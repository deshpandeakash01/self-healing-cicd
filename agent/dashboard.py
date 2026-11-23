"""
Runs dashboard: one self-contained HTML page showing every incident the
agent handled - outcome, model, time, tokens, what each fix changed and
what backed it.

WHERE THE RECORDS COME FROM
    - the local audit log (data/agent_runs.jsonl, or --file paths), and/or
    - with --repo, the "self-heal-record-*" artifacts the GitHub Action
      uploads after every run (needs GITHUB_TOKEN with Actions: read).
      Downloaded artifacts are cached in data/dashboard_cache/, so a second
      build downloads only new ones. GitHub deletes artifacts after their
      retention period (90 days by default); cached ones stay.

ONE INCIDENT = ONE FAILED CI RUN
    A run can produce several records: an overload retry, a fallback model,
    a re-run of the Self-heal job. They are grouped by (repo, run id); the
    LATEST record is the incident's result and the earlier ones are listed
    as "earlier tries".

SAFETY
    Records contain text from CI logs and from the LLM - untrusted. The data
    is embedded as JSON with every "<" escaped, and the page only ever
    inserts it with textContent, never as HTML.

RUN
    python -m agent.dashboard                       local records -> dashboard.html
    python -m agent.dashboard --repo owner/name     + the GitHub Action's records
    python -m agent.dashboard --file a.jsonl --file b.jsonl --out runs.html
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import zipfile
from pathlib import Path

from . import config

ARTIFACT_PREFIX = "self-heal-record-"
CACHE_DIR = config.PROJECT_ROOT / "data" / "dashboard_cache"


# ------------------------------------------------------------------ records
def read_jsonl(text: str, source: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue                    # a damaged line must not stop the page
        if isinstance(rec, dict):
            rec["_source"] = source
            out.append(rec)
    return out


def load_files(paths) -> list[dict]:
    records = []
    for p in paths:
        p = Path(p)
        if p.exists():
            records += read_jsonl(p.read_text(encoding="utf-8", errors="replace"), p.name)
    return records


def fetch_artifacts(gh, cache_dir: Path = CACHE_DIR, say=print) -> list[dict]:
    """The audit records the Action uploaded, downloading only new ones."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    records, new = [], 0
    for art in gh.list_artifacts(ARTIFACT_PREFIX):
        cached = cache_dir / f"{art['id']}.jsonl"
        if not cached.exists():
            if art.get("expired"):
                continue
            text = ""
            with zipfile.ZipFile(io.BytesIO(gh.download_artifact(art["id"]))) as z:
                for name in z.namelist():
                    if name.endswith(".jsonl"):
                        text += z.read(name).decode("utf-8", errors="replace") + "\n"
            cached.write_text(text, encoding="utf-8")
            new += 1
        records += read_jsonl(cached.read_text(encoding="utf-8"), art["name"])
    say(f"[dashboard] {new} new artifact(s) downloaded; {len(records)} record(s) from GitHub")
    return records


def dedupe(records: list[dict]) -> list[dict]:
    """The same record can arrive twice (local file and artifact)."""
    seen, out = set(), []
    for r in records:
        key = (r.get("time"), r.get("repo"), r.get("run_id"), r.get("outcome"),
               r.get("llm_calls"), r.get("seconds"))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def incidents(records: list[dict]) -> list[dict]:
    """Group records into incidents (one failed CI run), newest first."""
    groups: dict = {}
    for r in sorted(records, key=lambda r: r.get("time") or ""):
        key = (r.get("repo") or "(log file)", r.get("run_id") or r.get("job_name") or r.get("time"))
        groups.setdefault(key, []).append(r)
    out = []
    for (repo, _), recs in groups.items():
        final = recs[-1]
        out.append({"repo": repo, "final": final, "earlier": recs[:-1],
                    "llm_calls": sum(x.get("llm_calls") or 0 for x in recs),
                    "tokens_in": sum((x.get("llm_tokens") or {}).get("input") or 0 for x in recs),
                    "tokens_out": sum((x.get("llm_tokens") or {}).get("output") or 0 for x in recs),
                    "seconds": sum(x.get("seconds") or 0 for x in recs)})
    out.sort(key=lambda i: i["final"].get("time") or "", reverse=True)
    return out


def summarise(items: list[dict]) -> dict:
    """The headline numbers, over live incidents only (dry runs change nothing)."""
    live = [i for i in items if not i["final"].get("dry_run")]
    by = {}
    for i in live:
        by[i["final"].get("outcome")] = by.get(i["final"].get("outcome"), 0) + 1
    healed = [i for i in live if i["final"].get("outcome") == "healed"]
    times = sorted(i["seconds"] for i in healed)
    median = times[len(times) // 2] if len(times) % 2 else (
        (times[len(times) // 2 - 1] + times[len(times) // 2]) / 2 if times else None)
    return {"incidents": len(live), "healed": len(healed), "by_outcome": by,
            "median_heal_seconds": median,
            "llm_calls": sum(i["llm_calls"] for i in live),
            "tokens_in": sum(i["tokens_in"] for i in live),
            "tokens_out": sum(i["tokens_out"] for i in live),
            "dry_runs": len(items) - len(live)}


# --------------------------------------------------------------------- page
def safe_json(obj) -> str:
    """JSON that cannot end a <script> element or open a comment."""
    return (json.dumps(obj, ensure_ascii=False, default=str)
            .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def build_html(items: list[dict], title: str = "Self-heal runs") -> str:
    data = {"title": title, "incidents": items, "summary": summarise(items)}
    return PAGE.replace("__TITLE__", title.replace("<", "&lt;")).replace(
        "__DATA__", safe_json(data))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the self-heal runs dashboard.")
    ap.add_argument("--repo", help='also fetch the GitHub Action\'s records: "owner/name"')
    ap.add_argument("--file", action="append", help="audit log(s) to read "
                    f"(default: {config.AUDIT_LOG})")
    ap.add_argument("--out", default="dashboard.html")
    args = ap.parse_args(argv)

    records = load_files(args.file or [config.AUDIT_LOG])
    if args.repo:
        if not os.environ.get("GITHUB_TOKEN"):
            print("GITHUB_TOKEN is not set (needed to download the records).", file=sys.stderr)
            return 2
        from .github_client import GitHubClient, GitHubError
        try:
            records += fetch_artifacts(GitHubClient(args.repo, read_only=True))
        except GitHubError as e:
            print(f"GitHub error: {e}", file=sys.stderr)
            return 1
    items = incidents(dedupe(records))
    title = f"Self-heal runs \u2014 {args.repo}" if args.repo else "Self-heal runs"
    Path(args.out).write_text(build_html(items, title), encoding="utf-8")
    s = summarise(items)
    print(f"[dashboard] {len(items)} incident(s) ({s['incidents']} live, "
          f"{s['healed']} healed) -> {Path(args.out).resolve()}")
    return 0


# The page. Plain HTML/CSS/JS, no libraries, works offline. Colours: the
# status palette for outcomes (always with an icon and a label), recessive
# greys for everything else; light and dark each have their own steps.
PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fcfcfb;--panel:#ffffff;--line:#e4e3df;--ink:#0b0b0b;--ink2:#52514e;--ink3:#7a7974;
 --good:#0ca30c;--warning:#fab219;--serious:#ec835a;--critical:#d03b3b;--neutral:#a8a7a1;
 --code:#f4f3f0;--add:#e6f4e6;--del:#fbe9e9;--focus:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--panel:#222220;--line:#383835;--ink:#f0efec;
 --ink2:#c3c2b7;--ink3:#8d8c85;--neutral:#6b6a64;--code:#2a2a28;--add:#1d3320;--del:#3a2222;--focus:#3987e5}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--ink2);margin:0 0 22px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:18px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.tile .k{color:var(--ink2);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.tile .v{font-size:28px;font-weight:650;margin-top:2px;font-variant-numeric:tabular-nums}
.tile .n{color:var(--ink3);font-size:12px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:18px}
.panel h2{font-size:14px;margin:0 0 10px}
.bar{display:flex;gap:2px;height:22px;border-radius:4px;overflow:hidden}
.bar div{height:100%;min-width:4px;cursor:default}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin-top:10px;color:var(--ink2)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:10px;height:10px;border-radius:2px;display:inline-block}
.filters{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:10px}
select,input[type=search]{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font:inherit}
label.chk{color:var(--ink2);display:inline-flex;gap:6px;align-items:center}
table{width:100%;border-collapse:collapse}
th{text-align:left;font-weight:600;color:var(--ink2);font-size:12px;border-bottom:1px solid var(--line);padding:8px 6px}
td{border-bottom:1px solid var(--line);padding:9px 6px;vertical-align:top}
tr.row{cursor:pointer}
tr.row:hover td{background:var(--code)}
tr.row:focus{outline:2px solid var(--focus);outline-offset:-2px}
.num{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
.nw{white-space:nowrap}
.badge{display:inline-flex;align-items:center;gap:5px;white-space:nowrap;font-weight:600}
.muted{color:var(--ink3)}
.detail td{background:var(--code);padding:14px 16px}
.detail h3{font-size:13px;margin:12px 0 4px}
.detail h3:first-child{margin-top:0}
.ev{display:inline-block;font-size:12px;padding:1px 6px;border-radius:4px;border:1px solid var(--line);margin-left:6px}
pre{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px;overflow:auto;font:12px/1.4 ui-monospace,Consolas,monospace;max-height:340px}
.diff .a{background:var(--add)}.diff .d{background:var(--del)}
a{color:var(--focus)}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);padding:6px 9px;border-radius:6px;font-size:12px;display:none;z-index:9}
.empty{color:var(--ink2);padding:24px 6px}
</style></head>
<body><main>
<h1 id="title"></h1>
<p class="sub" id="sub"></p>
<section class="tiles" id="tiles"></section>
<section class="panel"><h2>Outcomes of live incidents</h2><div class="bar" id="bar" role="img"></div><div class="legend" id="legend"></div></section>
<section class="panel">
 <div class="filters">
  <select id="fRepo" aria-label="Repository"></select>
  <select id="fOutcome" aria-label="Outcome"></select>
  <label class="chk"><input type="checkbox" id="fDry"> include dry runs</label>
  <input type="search" id="fText" placeholder="Search diagnosis, files…" aria-label="Search">
 </div>
 <table><thead><tr><th>When</th><th>Repository · job</th><th>Classifier</th><th>Diagnosis → action</th><th>Outcome</th>
 <th class="num" title="Attempts in the final try (+ earlier tries for the same run: overload retries, fallback model, re-runs)">Tries</th><th>Model</th><th class="num">Time</th><th class="num">LLM calls</th><th>PR</th></tr></thead>
 <tbody id="rows"></tbody></table>
</section>
<div id="tip"></div>
</main>
<script type="application/json" id="data">__DATA__</script>
<script>
"use strict";
const D = JSON.parse(document.getElementById("data").textContent);
const OUT = {healed:["good","\u2714","healed"], escalated:["warning","\u270B","handed to a person"],
  failed:["serious","\u2716","fix failed"], timeout:["serious","\u23F1","timed out"],
  error:["critical","\u26A0","error"], unverified:["neutral","?","unverified"], dry_run:["neutral","\u25CB","dry run"]};
const outInfo = o => OUT[o] || ["neutral","\u2022", String(o || "unknown")];
const $ = id => document.getElementById(id);
function el(tag, text, cls){ const e=document.createElement(tag); if(text!=null) e.textContent=String(text); if(cls) e.className=cls; return e; }
function secs(s){ if(s==null) return "–"; s=Math.round(s); return s<60? s+" s" : Math.floor(s/60)+" min "+String(s%60).padStart(2,"0")+" s"; }
const MON=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
function when(t){ const m=/^(\d{4})-(\d\d)-(\d\d)T(\d\d:\d\d)/.exec(String(t||"")); return m? (+m[3])+" "+MON[+m[2]-1]+" "+m[1]+" "+m[4] : (t||"–"); }
function badge(o){ const [c,i,l]=outInfo(o); const b=el("span",null,"badge"); const sw=el("span",i); sw.style.color="var(--"+c+")"; b.append(sw, el("span",l)); return b; }
const fmt = n => Number(n||0).toLocaleString();

$("title").textContent = D.title;
const S = D.summary;
$("sub").textContent = S.incidents + " live incident(s)" + (S.dry_runs? " and "+S.dry_runs+" dry run(s)" : "") + " · generated " + new Date().toLocaleString();
[["Live incidents", S.incidents, ""],
 ["Healed", S.healed, S.incidents? Math.round(100*S.healed/S.incidents)+"% of live incidents" : ""],
 ["Median time to heal", secs(S.median_heal_seconds), "whole incident, incl. waiting for CI"],
 ["LLM calls", fmt(S.llm_calls), fmt(S.tokens_in)+" tokens in · "+fmt(S.tokens_out)+" out"]
].forEach(([k,v,n])=>{ const t=el("div",null,"tile"); t.append(el("div",k,"k"), el("div",v,"v"), el("div",n,"n")); $("tiles").append(t); });

// outcome bar: one segment per outcome, status colours, label + icon in the legend
const order=["healed","escalated","failed","timeout","error","unverified"];
const counts=Object.entries(S.by_outcome).sort((a,b)=>{const ia=order.indexOf(a[0]),ib=order.indexOf(b[0]);return (ia<0?99:ia)-(ib<0?99:ib);});
const tip=$("tip");
$("bar").setAttribute("aria-label", counts.map(([o,n])=>outInfo(o)[2]+": "+n).join(", ") || "no live incidents");
counts.forEach(([o,n])=>{
  const [c,i,l]=outInfo(o); const seg=el("div"); seg.style.flex=n; seg.style.background="var(--"+c+")";
  seg.onmousemove=e=>{tip.textContent=l+": "+n+" of "+S.incidents; tip.style.display="block"; tip.style.left=(e.clientX+12)+"px"; tip.style.top=(e.clientY+12)+"px";};
  seg.onmouseleave=()=>tip.style.display="none";
  $("bar").append(seg);
  const lg=el("span"); const sw=el("span",null,"sw"); sw.style.background="var(--"+c+")"; lg.append(sw, el("span", i+" "+l+" "+n)); $("legend").append(lg);
});
if(!counts.length){ $("bar").append(el("div",null)); $("legend").append(el("span","No live incidents yet.","muted")); }

// filters
const repos=[...new Set(D.incidents.map(x=>x.repo))].sort();
$("fRepo").append(new Option("All repositories",""), ...repos.map(r=>new Option(r,r)));
const outs=[...new Set(D.incidents.map(x=>x.final.outcome))];
$("fOutcome").append(new Option("All outcomes",""), ...outs.map(o=>new Option(outInfo(o)[2],o)));
["fRepo","fOutcome","fDry","fText"].forEach(id=>$(id).addEventListener("input",render));

function haystack(x){ const f=x.final; return JSON.stringify([x.repo,f.job_name,f.model,(f.attempts||[]).map(a=>[a.proposal,a.edit_paths])]).toLowerCase(); }

function detail(x){
  const f=x.final, box=el("div");
  const last=(f.attempts||[]).slice(-1)[0]||{}; const p=last.proposal||{};
  box.append(el("h3","Root cause"), el("div", p.root_cause||"–"));
  if(p.explanation){ box.append(el("h3","Explanation"), el("div", p.explanation)); }
  (f.attempts||[]).forEach((a,k)=>{
    const ar=a.action_result||{}, v=a.verify||{}, ap=a.proposal||{};
    box.append(el("h3","Attempt "+(k+1)+": "+(ap.category||"?")+" → "+(ar.action||"?")+" → "+(v.status||"?")+(v.note? " ("+v.note+")":"")));
    const g=a.guardrail||{}; if(g.reasons&&g.reasons.length){ const ul=el("ul"); g.reasons.forEach(r=>ul.append(el("li","blocked: "+r))); box.append(ul); }
    const ev=ar.evidence||[];
    if(ev.length){ const ul=el("ul"); ev.forEach(e=>{ const li=el("li"); li.append(el("code",e.path));
        li.append(el("span", e.evidence==="log"? "CI log lines "+e.log_lines : "code only — check first","ev"));
        if(e.why) li.append(el("span"," — "+e.why)); if(e.note) li.append(el("span"," ("+e.note+")","muted")); ul.append(li); }); box.append(ul); }
    else if((a.edit_paths||[]).length){ box.append(el("div","Files: "+a.edit_paths.join(", "),"muted")); }
    if(ar.diff){ const pre=el("pre",null,"diff"); String(ar.diff).split("\n").forEach(line=>{ const s=el("div",line||" ");
        if(line.startsWith("+")&&!line.startsWith("+++")) s.className="a"; else if(line.startsWith("-")&&!line.startsWith("---")) s.className="d"; pre.append(s); }); box.append(pre); }
    const steps=(a.tool_calls||[]).length; if(steps) box.append(el("div",steps+" LLM step(s): "+a.tool_calls.map(s=>s.map(c=>String(c).split("(")[0]).join(", ")).join(" | "),"muted"));
  });
  if(x.earlier.length){ box.append(el("h3","Earlier tries for this run"));
    const ul=el("ul"); x.earlier.forEach(r=>ul.append(el("li", when(r.time)+" · "+(r.model||"no model")+" · "+outInfo(r.outcome)[2]+(r.llm_error? " — "+r.llm_error : "")))); box.append(ul); }
  return box;
}

function render(){
  const repo=$("fRepo").value, out=$("fOutcome").value, dry=$("fDry").checked, q=$("fText").value.trim().toLowerCase();
  const rows=$("rows"); rows.replaceChildren();
  const list=D.incidents.filter(x=>(dry||!x.final.dry_run)&&(!repo||x.repo===repo)&&(!out||x.final.outcome===out)&&(!q||haystack(x).includes(q)));
  if(!list.length){ const tr=el("tr"); const td=el("td","No incidents match.","empty"); td.colSpan=10; tr.append(td); rows.append(tr); return; }
  list.forEach(x=>{
    const f=x.final, last=(f.attempts||[]).slice(-1)[0]||{}, p=last.proposal||{}, t=f.triage||{};
    const tr=el("tr",null,"row"); tr.tabIndex=0;
    const repoCell=el("td"); repoCell.append(el("div",x.repo), el("div",(f.job_name||"")+(f.run_id? " · run "+f.run_id:""),"muted"));
    const pr=el("td"); if(f.pr&&f.pr.url&&/^https:\/\/github\.com\//.test(f.pr.url)){ const a=el("a","#"+f.pr.number); a.href=f.pr.url; a.target="_blank"; a.rel="noopener"; a.onclick=e=>e.stopPropagation(); pr.append(a);} else pr.append(el("span","–","muted"));
    const oc=el("td"); oc.append(badge(f.outcome)); if(f.dry_run) oc.append(el("div","dry run","muted"));
    tr.append(el("td",when(f.time),"nw"), repoCell, el("td",(t.label||"–")+(t.confidence!=null? " ("+Number(t.confidence).toFixed(2)+")":"")),
      el("td",(p.category||"–")+" → "+(p.action||"–")), oc, el("td",(f.attempts||[]).length+(x.earlier.length? " (+"+x.earlier.length+")":""),"num"),
      el("td",f.model||"–","nw"), el("td",secs(x.seconds),"num"), el("td",fmt(x.llm_calls),"num"), pr);
    const dr=el("tr",null,"detail"); dr.hidden=true; const dtd=el("td"); dtd.colSpan=10; dr.append(dtd);
    const toggle=()=>{ if(!dtd.firstChild) dtd.append(detail(x)); dr.hidden=!dr.hidden; };
    tr.onclick=toggle; tr.onkeydown=e=>{ if(e.key==="Enter"||e.key===" "){ e.preventDefault(); toggle(); } };
    rows.append(tr, dr);
  });
}
render();
</script></body></html>
"""

if __name__ == "__main__":
    sys.exit(main())
