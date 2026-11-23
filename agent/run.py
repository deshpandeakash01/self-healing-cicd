"""
Command line for the self-healing agent.

DRY-RUN IS THE DEFAULT
    Without --live the agent reads from GitHub but writes nothing: it prints
    the rerun or pull request it WOULD make. This is deliberate. Trying the
    agent must be safe; changing a repository must be a decision.

EXAMPLES (from the project root)

  Diagnose a saved log file, no GitHub needed:
      python -m agent.run --log-file "dataset\\data\\raw\\logs\\go\\etcd-io__etcd__116750249015.txt"

  Plan a fix for a failed run in your test repo (writes nothing):
      python -m agent.run --repo you/your-test-repo --run-id 1234567890

  Really do it, and wait for CI to confirm the fix:
      python -m agent.run --repo you/your-test-repo --run-id 1234567890 --live

  As the automatic trigger runs it (examples in test_repo/self-heal.yml):
      python -m agent.run --repo you/your-test-repo --run-id 1234567890 --live --retry-overloaded 3

ENVIRONMENT
    GITHUB_TOKEN     needed with --repo
    GOOGLE_API_KEY   needed unless --mode classifier_only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import config
from .github_client import GitHubClient, GitHubError
from .graph import Agent


def overloaded(state: dict) -> bool:
    """The LLM service was busy (503): worth trying again in a few minutes.
    A used-up daily quota is not: that lasts until the next day."""
    return (state.get("outcome") == "error"
            and "overloaded" in (state.get("llm_error") or "").lower()
            and not state.get("pr"))            # never redo a run that already wrote


def run_with_retries(start, retries: int, wait: float, sleep=time.sleep, say=print) -> dict:
    """
    Run the agent; when Gemini answers "overloaded" (503), wait and start
    again, at most `retries` more times. `start` is a function that runs the
    agent once and returns its final state. Each try writes its own audit
    record, so the waits are visible afterwards.
    """
    state = start()
    for n in range(1, retries + 1):
        if not overloaded(state):
            break
        say(f"[retry] Gemini is overloaded; trying again in {wait:.0f}s ({n} of {retries})")
        sleep(wait)
        state = start()
    return state


def unavailable(state: dict) -> bool:
    """The LLM could not be used at all - overloaded or out of quota - and
    nothing was written. Another model may still be available."""
    return (state.get("outcome") == "error" and bool(state.get("llm_error"))
            and not state.get("pr"))


def run_with_fallback(start, models: list, retries: int, wait: float,
                      sleep=time.sleep, say=print) -> dict:
    """
    Try each model in turn (with the overload retries for each) until one
    is available. `start(model)` runs the agent once with that model. A
    model that answered is never swapped mid-incident: the switch happens
    only when the previous one could not be used at all.
    """
    state = {}
    for i, model in enumerate(models):
        if i:
            say(f"[fallback] {models[i - 1]} is unavailable; trying {model}")
        state = run_with_retries(lambda m=model: start(m), retries, wait, sleep, say)
        if not unavailable(state):
            break
    return state


def error_comment(state: dict, repo: str) -> str:
    inc = state.get("incident") or {}
    run_id = inc.get("run_id", state.get("run_id"))
    pr = state.get("pr")
    if pr:
        # An earlier attempt already changed something (log 3.47): never
        # say "nothing was changed" then.
        return "\n".join([
            "**Self-heal stopped before this failure was fixed.**",
            "",
            f"Run [{run_id}](https://github.com/{repo}/actions/runs/{run_id}) failed. "
            f"The agent opened [pull request #{pr.get('number')}]({pr.get('url')}), "
            "but CI does not pass on it yet, and the language model became "
            "unavailable before a second attempt. A human needs to take over.",
            "",
            f"Reason: {state.get('llm_error')}",
        ])
    return "\n".join([
        "**Self-heal could not look at this failure.**",
        "",
        f"Run [{run_id}](https://github.com/{repo}/actions/runs/{run_id}) failed, "
        "but the agent's language model was not available, so nothing was "
        "diagnosed and nothing was changed.",
        "",
        f"Reason: {state.get('llm_error')}",
    ])


def job_summary(state: dict, repo: str | None, live: bool) -> str:
    """
    Markdown for the GitHub Actions run page ($GITHUB_STEP_SUMMARY): what
    failed, what the agent decided, what it changed and what backs each
    change. Text from the LLM is shown as quoted text, never as HTML.
    """
    p = state.get("proposal") or {}
    d = state.get("decision") or {}
    outcome = state.get("outcome")
    icon = {"healed": "\u2705", "dry_run": "\U0001F50D", "escalated": "\U0001F64B",
            "unverified": "\u2753"}.get(outcome, "\u274C")
    inc = state.get("incident") or {}
    run_id = inc.get("run_id", state.get("run_id"))
    lines = [f"### Self-heal: {icon} {outcome}", ""]
    if repo and run_id:
        lines.append(f"**Failed run:** [{run_id}](https://github.com/{repo}/actions/runs/{run_id})"
                     f" \u00b7 job `{inc.get('job_name', state.get('job_name'))}`")
    tri = state.get("first_triage") or state.get("triage") or {}
    if tri:
        lines.append(f"**Classifier:** {tri.get('label')} ({tri.get('confidence')})")
    lines += [f"**Model:** {state.get('model') or 'none'}"
              + ("" if live else " \u00b7 **dry run** (nothing was changed)"),
              f"**Diagnosis:** {p.get('category')} \u2192 {d.get('action')}"
              + ("" if d.get("ok", True) else " (proposal blocked by the guardrail)"),
              "", "> " + str(p.get("root_cause") or "").replace("\n", "\n> "), ""]
    for r in d.get("reasons") or []:
        lines.append(f"- blocked: {r}")
    attempts = state.get("attempts") or []
    if len(attempts) > 1:
        lines += ["", "**Attempts**", ""]
        for a in attempts:
            ap_, av = a.get("proposal") or {}, a.get("verify") or {}
            files = ", ".join(f"`{x}`" for x in a.get("edit_paths") or []) or "no files"
            lines.append(f"{a.get('attempt', 0) + 1}. {ap_.get('category')} \u2192 "
                         f"{(a.get('action_result') or {}).get('action')} [{files}] \u2192 "
                         f"{av.get('status')}")
    evidence = [ev for a in attempts for ev in (a.get("action_result") or {}).get("evidence") or []]
    if evidence:
        lines += ["", "**What backs each change**", ""]
        for ev in evidence:
            what = (f"CI log lines {ev['log_lines']}" if ev["evidence"] == "log"
                    else "found in the code only \u2014 check first")
            lines.append(f"- `{ev['path']}`: {what}")
    diffs = [(a.get("action_result") or {}).get("diff") for a in attempts]
    diffs = [x for x in diffs if x]
    if diffs:
        lines += ["", "<details><summary>Change</summary>", "", "```diff",
                  "\n".join(diffs).replace("```", "``\u200b`"), "```", "", "</details>"]
    if state.get("pr"):
        lines += ["", f"**Pull request:** {state['pr'].get('url')}"]
    lines += ["", f"LLM calls: {state.get('llm_calls', 0)} \u00b7 tokens in "
                  f"{state.get('llm_tokens_in', 0):,}, out {state.get('llm_tokens_out', 0):,}"]
    return "\n".join(lines) + "\n"


def write_github_outputs(state: dict, repo: str | None, live: bool) -> None:
    """When running inside GitHub Actions: the run-page summary, and outputs
    (outcome, pull-request, model) that later workflow steps can use."""
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(job_summary(state, repo, live))
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"outcome={state.get('outcome') or ''}\n")
            f.write(f"pull-request={(state.get('pr') or {}).get('url') or ''}\n")
            f.write(f"model={state.get('model') or ''}\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Self-healing CI agent.")
    ap.add_argument("--repo", help='GitHub repository, "owner/name"')
    ap.add_argument("--run-id", type=int, help="failed workflow run id")
    ap.add_argument("--job-id", type=int,
                    help="failed job id (default: the run's first failed job)")
    ap.add_argument("--log-file", help="diagnose a saved raw log instead of a run")
    ap.add_argument("--live", action="store_true",
                    help="actually rerun jobs / open pull requests")
    ap.add_argument("--no-wait", action="store_true",
                    help="do not wait for CI to verify the result")
    ap.add_argument("--mode", choices=config.MODES, default=config.MODE)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-memory", action="store_true",
                    help="do not show the LLM earlier verified fixes (for clean trials)")
    ap.add_argument("--retry-overloaded", type=int, default=0, metavar="N",
                    help="when Gemini is overloaded (503), wait and try again N times")
    ap.add_argument("--retry-wait", type=float, default=120.0, metavar="SECONDS")
    ap.add_argument("--fallback-model", default=config.GEMINI_FALLBACK_MODEL, metavar="NAME",
                    help="Gemini model to try when the main one is overloaded or out of "
                         "quota (default %(default)s; '' to switch off)")
    ap.add_argument("--no-notify", action="store_true",
                    help="in live mode, do not comment on the failing commit when "
                         "handing over to a human")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")     # Windows consoles

    if not args.log_file and not (args.repo and (args.run_id or args.job_id)):
        ap.error("give --log-file, or --repo with --run-id (or --job-id)")
    if args.live and args.log_file:
        ap.error("--live needs a real run (--repo and --run-id), not --log-file")

    if args.no_memory:
        config.MEMORY_TOP_K = 0

    gh = None
    if args.repo and not args.log_file:
        if not os.environ.get("GITHUB_TOKEN"):
            print("GITHUB_TOKEN is not set.", file=sys.stderr)
            return 2
        gh = GitHubClient(args.repo, read_only=not args.live)

    models, llms = [None], {}
    if args.mode != "classifier_only":
        from .llm import make_llm
        models = [config.GEMINI_MODEL]
        if args.fallback_model and args.fallback_model != config.GEMINI_MODEL:
            models.append(args.fallback_model)
        try:
            llms[config.GEMINI_MODEL] = make_llm()
        except RuntimeError as e:
            print(e, file=sys.stderr)
            return 2

    def start(model=None) -> dict:
        # A fresh agent each time, so a retry starts from a clean state.
        if model and model not in llms:
            llms[model] = make_llm(model)
        agent = Agent(llms.get(model), gh, dry_run=not args.live, wait=not args.no_wait,
                      mode=args.mode, verbose=not args.quiet, notify=not args.no_notify)
        if args.log_file:
            with open(args.log_file, encoding="utf-8", errors="replace") as f:
                raw = f.read()
            return agent.run(raw_log=raw, job_name=os.path.basename(args.log_file))
        return agent.run(repo=args.repo, run_id=args.run_id, job_id=job_id)

    print(f"mode={args.mode}  " + ("LIVE - will write to GitHub" if args.live
                                   else "DRY-RUN - nothing will be changed"))
    try:
        job_id = args.job_id
        if not args.log_file and job_id is None:
            failed = gh.failed_jobs(args.run_id)
            if not failed:
                print(f"Run {args.run_id} has no failed jobs.")
                return 0
            job_id = failed[0]["id"]
        state = run_with_fallback(start, models, args.retry_overloaded, args.retry_wait)
    except GitHubError as e:
        print(f"GitHub error: {e}", file=sys.stderr)
        return 1

    # The LLM was unavailable (quota or still overloaded). With nobody
    # watching, say so where a person will see it, so the failure is not
    # silently ignored.
    sha = (state.get("incident") or {}).get("head_sha") or state.get("head_sha")
    if (state.get("outcome") == "error" and state.get("llm_error") and args.live
            and not args.no_notify and gh is not None and sha):
        try:
            gh.comment_on_commit(sha, error_comment(state, args.repo))
            print("[report] the error was posted as a comment on the failing commit")
        except GitHubError as e:
            print(f"[report] could not post the error comment: {e}", file=sys.stderr)

    try:
        write_github_outputs(state, args.repo, args.live)
    except OSError as e:                          # a summary must not fail the run
        print(f"[report] could not write the GitHub summary: {e}", file=sys.stderr)

    p = state.get("proposal") or {}
    result = state.get("action_result") or {}
    print("\n" + "=" * 72)
    print(f"outcome    : {state.get('outcome')}")
    print(f"model      : {state.get('model') or 'none'}")
    print(f"category   : {p.get('category')}")
    print(f"root cause : {p.get('root_cause')}")
    print(f"action     : {state['decision']['action']}"
          + ("" if state["decision"]["ok"] else "  (proposal was blocked by the guardrail)"))
    for reason in state["decision"]["reasons"]:
        print(f"  blocked  : {reason}")
    print(f"explanation: {p.get('explanation')}")
    if result.get("plan"):
        print("would do   : " + json.dumps(result["plan"], indent=2))
    attempts = state.get("attempts") or []
    if len(attempts) > 1:
        # The lines above describe the LAST attempt. Show the whole story.
        print(f"attempts   : {len(attempts)}")
        for a in attempts:
            ap_, av = a.get("proposal") or {}, a.get("verify") or {}
            files = ", ".join(a.get("edit_paths") or []) or "no files"
            print(f"  {a.get('attempt', 0) + 1}. {ap_.get('category')} -> "
                  f"{(a.get('action_result') or {}).get('action')} [{files}] -> "
                  f"{av.get('status')}" + (f": {av['note']}" if av.get("note") else ""))
    diffs = [(a.get("action_result") or {}).get("diff") for a in attempts]
    diffs = [d for d in diffs if d]
    if diffs:
        print("change     :\n" + "\n".join(diffs))
    evidence = [ev for a in attempts for ev in (a.get("action_result") or {}).get("evidence") or []]
    if evidence:
        print("backed by  :")
        for ev in evidence:
            what = (f"log lines {ev['log_lines']}" if ev["evidence"] == "log"
                    else "CODE ONLY - check first")
            print(f"  {ev['path']}: {what}" + (f"  ({ev['note']})" if ev.get("note") else ""))
    if state.get("pr"):
        print(f"pull request: {state['pr'].get('url')}")
    tokens = (f"  ({state['llm_tokens_in']:,} tokens in, {state.get('llm_tokens_out', 0):,} out)"
              if state.get("llm_tokens_in") else "")
    print(f"LLM calls  : {state.get('llm_calls', 0)}{tokens}")
    print(f"audit log  : {config.AUDIT_LOG}")
    # 0 when the incident was handled or cleanly handed to a human.
    return 0 if state.get("outcome") in ("healed", "dry_run", "escalated",
                                         "unverified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
