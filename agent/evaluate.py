"""
Feature 9 - Evaluation of DIAGNOSIS against the hand-labelled sample.

THE QUESTION (PROJECT_LOG section 1)
    How much of the LLM's work can the cheap classifier take over without
    losing accuracy, and does that hold on a language it has never seen?

WHAT IS COMPARED, on the same hand-labelled logs
    rules            the regex label from clean_and_label.py
    classifier_only  the classifier's label ("other" when below the threshold)
    llm_only         the LLM investigates every log
    cascade(t)       the classifier's label when its confidence is >= t,
                     otherwise the LLM's answer; for several thresholds t
    For each: accuracy, macro-F1, and how many LLM calls it needed.

HOW THE CASCADE IS COMPUTED
    The LLM is run ONCE per log (in llm_only mode, with no classifier hint)
    and the result is cached. Each cascade(t) is then put together from the
    cache: no extra LLM calls. So "cascade" here means "classifier if sure,
    else the un-hinted LLM"; the live agent additionally passes the
    classifier's hint to the LLM, which this does not measure.

WHAT THIS DOES NOT MEASURE
    Whether a FIX works. That needs real runs against a repository. This is
    the diagnosis half only: did the system name the right kind of failure?

RULES OF THE GAME
    - Rows with human_label empty or "unclear" are skipped.
    - A log matched by the "failed because another job failed" rule is
      predicted needs_traceback by every system, since every mode applies
      that rule before anything else.
    - The cache (data/eval_llm_cache.jsonl) means an interrupted run resumes
      where it stopped and costs nothing to repeat. Delete it to start over.
    - Cached answers are kept per model. After a model change the other
      model's answers stay in the file but are not used, so two models can
      be compared by running the evaluation once with each.
    - The fix memory is switched off during evaluation, so earlier runs
      cannot leak answers in.

CAVEAT ON classifier_only
    The deployed model was trained on logs that are also in the sample, so
    its score here is optimistic. For an honest figure run
    train_holdout_model.py first and point SH_MODEL_PATH at its output.

USAGE (from the project root)
    python -m agent.evaluate --no-llm          rules and classifier only; no key
    python -m agent.evaluate                   full run (needs GOOGLE_API_KEY)
    python -m agent.evaluate --limit 20 --delay 5      a small, slow first try
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from . import config
from .llm import model_name
from .triage import triage

csv.field_size_limit(sys.maxsize)

SAMPLE = config.PROJECT_ROOT / "data" / "hand_label_sample.csv"
LABELLED = config.PROJECT_ROOT / "data" / "labelled_logs.csv"
LOG_ROOT = config.PROJECT_ROOT / "dataset" / "data" / "raw" / "logs"
CACHE = config.PROJECT_ROOT / "data" / "eval_llm_cache.jsonl"
EVAL_AUDIT = config.PROJECT_ROOT / "data" / "eval_audit.jsonl"
RESULTS = config.PROJECT_ROOT / "data" / "eval_diagnosis_results.csv"
PER_LOG = config.PROJECT_ROOT / "data" / "eval_diagnosis_per_log.csv"

HUMAN_LABELS = ("dependency", "flaky_infra", "lint_build", "other_bot_infra",
                "test_failure", "config", "needs_traceback", "other")
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def find_raw_log(language: str, log_id: str) -> Path | None:
    """Raw logs are named <owner>__<repo>__<job id>.txt in the language folder."""
    hits = list((LOG_ROOT / language).glob(f"*__{log_id}.txt"))
    return hits[0] if hits else None


def load_sample(path: Path) -> tuple[list[dict], Counter]:
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        rows = list(csv.DictReader(f))
    skipped = Counter()
    usable = []
    for r in rows:
        label = (r.get("human_label") or "").strip().lower()
        if not label:
            skipped["not labelled yet"] += 1
        elif label == "unclear":
            skipped["labelled 'unclear'"] += 1
        elif label not in HUMAN_LABELS:
            skipped[f"unknown label {label!r}"] += 1
        else:
            r["human_label"] = label
            usable.append(r)
    return usable, skipped


def load_rule_labels(ids: set) -> dict:
    out = {}
    with open(LABELLED, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            if r["log_id"] in ids:
                out[r["log_id"]] = r["category"]
    return out


def read_cache(model: str | None) -> tuple[dict, Counter]:
    """
    Cached LLM diagnoses made by `model`, plus a count of the answers in the
    file that other models made. Those are left in the file but not used:
    scoring one model with another model's answers would be a silent mistake
    after a model change.
    """
    cache, others = {}, Counter()
    if CACHE.exists():
        with open(CACHE, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("model") == model:
                        cache[rec["log_id"]] = rec
                    else:
                        others[str(rec.get("model"))] += 1
                except (ValueError, KeyError):
                    continue
    return cache, others


def macro_f1(truth: list, pred: list) -> float:
    """Macro-F1 over the classes that occur in the human labels."""
    scores = []
    for c in sorted(set(truth)):
        tp = sum(1 for t, p in zip(truth, pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(truth, pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(truth, pred) if t == c and p != c)
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def score(name: str, truth: list, pred: list, llm_calls, n_total: int) -> dict:
    n = len(truth)
    right = sum(1 for t, p in zip(truth, pred) if t == p)
    return {"system": name, "n": n,
            "accuracy": right / n if n else 0.0,
            "macro_f1": macro_f1(truth, pred),
            "llm_logs": llm_calls,          # logs that needed the LLM
            "llm_share": (llm_calls / n_total) if (llm_calls is not None and n_total) else None}


def run_llm(rows: list[dict], cache: dict, llm, model: str | None, delay: float,
            limit: int | None, verbose: bool) -> None:
    """Fill the cache with one LLM diagnosis per log that does not have one."""
    from .graph import Agent

    # Evaluation must not write into the real audit log or read the fix memory.
    saved = (config.AUDIT_LOG, config.FIX_MEMORY)
    config.AUDIT_LOG = EVAL_AUDIT
    config.FIX_MEMORY = config.PROJECT_ROOT / "data" / "_no_memory_during_eval.jsonl"
    done = 0
    try:
        agent = Agent(llm, None, mode="llm_only", verbose=False)
        todo = [r for r in rows if r["log_id"] not in cache and not r["_derivative"]]
        if limit is not None:
            todo = todo[:limit]
        print(f"LLM diagnoses needed: {len(todo)} "
              f"({len(cache)} already in the cache)")
        for i, r in enumerate(todo, 1):
            raw = r["_raw_path"].read_text(encoding="utf-8", errors="replace")
            started = time.monotonic()
            state = agent.run(raw_log=raw, job_name=r["_raw_path"].name)
            p = state["proposal"]
            if p.get("explanation", "").startswith("LLM call failed"):
                # Quota or network: do not record a made-up answer. Stop; the
                # next run resumes from here.
                print(f"\nStopped at log {r['log_id']}: {p['explanation'][:300]}")
                print("Nothing was recorded for it. Run the command again to resume.")
                return
            rec = {"log_id": r["log_id"], "category": p["category"],
                   "action": p["action"], "root_cause": p["root_cause"],
                   "gave_up": p["root_cause"].startswith("The investigation did not"),
                   "llm_steps": state.get("llm_calls", 0),
                   "seconds": round(time.monotonic() - started, 1),
                   "model": model}
            with open(CACHE, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cache[r["log_id"]] = rec
            done += 1
            if verbose:
                print(f"  [{i}/{len(todo)}] {r['log_id']} human={r['human_label']:16s} "
                      f"llm={rec['category']:16s} steps={rec['llm_steps']}", flush=True)
            if delay:
                time.sleep(delay)
    finally:
        config.AUDIT_LOG, config.FIX_MEMORY = saved
        if done:
            print(f"Added {done} LLM diagnoses to {CACHE}")


def evaluate(sample_path: Path = SAMPLE, llm=None, use_llm: bool = True,
             delay: float = 0.0, limit: int | None = None,
             verbose: bool = True) -> list[dict]:
    rows, skipped = load_sample(sample_path)
    print(f"Sample: {sample_path}")
    print(f"Usable hand-labelled logs: {len(rows)}")
    for why, n in skipped.items():
        print(f"  skipped {n}: {why}")
    if not rows:
        print("\nNothing to evaluate yet. Fill in human_label in the sample "
              "(guide: PROJECT_LOG.txt, Appendix A).")
        return []

    rule_labels = load_rule_labels({r["log_id"] for r in rows})
    missing = 0
    for r in rows:
        r["_rule"] = rule_labels.get(r["log_id"], "unlabelled")
        r["_raw_path"] = find_raw_log(r["language"], r["log_id"])
        if r["_raw_path"] is None:
            missing += 1
            continue
        t = triage(r["_raw_path"].read_text(encoding="utf-8", errors="replace"))
        r["_triage"] = t
        r["_derivative"] = t.decided_by == "derivative_rule"
    if missing:
        print(f"  skipped {missing}: raw log file not found under {LOG_ROOT}")
        rows = [r for r in rows if r["_raw_path"] is not None]

    # The model the LLM answers belong to: the LLM object's own name when one
    # was given, otherwise the configured model (for --no-llm re-scoring).
    model = model_name(llm) if llm is not None else config.GEMINI_MODEL
    cache, others = read_cache(model)
    print(f"LLM model: {model}")
    for other, k in sorted(others.items()):
        print(f"  ignoring {k} cached answers made by {other}")
    if use_llm:
        run_llm(rows, cache, llm, model, delay, limit, verbose)

    truth = [r["human_label"] for r in rows]
    n = len(rows)
    results = []

    # --- rules: only where a rule fired -------------------------------------
    fired = [r for r in rows if r["_rule"] != "unlabelled"]
    res = score("rules (where a rule fired)", [r["human_label"] for r in fired],
                [r["_rule"] for r in fired], 0, n)
    res["note"] = f"rules labelled {len(fired)} of {n}"
    results.append(res)

    # --- classifier only ------------------------------------------------------
    def clf_label(r, threshold):
        t = r["_triage"]
        if r["_derivative"]:
            return "needs_traceback"
        return t.best_guess if t.confidence >= threshold else None

    pred = [clf_label(r, config.CONFIDENCE_THRESHOLD) or "other" for r in rows]
    res = score(f"classifier_only (t={config.CONFIDENCE_THRESHOLD})", truth, pred, 0, n)
    answered = [(t, p) for t, p, r in zip(truth, pred, rows)
                if clf_label(r, config.CONFIDENCE_THRESHOLD)]
    res["note"] = (f"answered {len(answered)} of {n}; right on "
                   f"{sum(1 for t, p in answered if t == p)} of those")
    results.append(res)

    # --- LLM only and cascade: only on logs that have an LLM answer -----------
    have = [r for r in rows if r["_derivative"] or r["log_id"] in cache]
    if len(have) < n:
        print(f"\nLLM answers exist for {len(have)} of {n} logs; llm_only and "
              "cascade are scored on those only.")
    if any(r["log_id"] in cache for r in rows):
        h_truth = [r["human_label"] for r in have]

        def llm_label(r):
            return "needs_traceback" if r["_derivative"] else cache[r["log_id"]]["category"]

        needs_llm = sum(1 for r in have if not r["_derivative"])
        results.append(score("llm_only", h_truth, [llm_label(r) for r in have],
                             needs_llm, len(have)))
        for t in THRESHOLDS:
            pred, calls = [], 0
            for r in have:
                c = clf_label(r, t)
                if c is None:
                    calls += 1
                    c = llm_label(r)
                pred.append(c)
            results.append(score(f"cascade (t={t})", h_truth, pred, calls, len(have)))
        steps = [cache[r["log_id"]]["llm_steps"] for r in have if r["log_id"] in cache]
        gave_up = sum(1 for r in have if r["log_id"] in cache and cache[r["log_id"]]["gave_up"])
        print(f"LLM: {sum(steps) / len(steps):.1f} model calls per log on average; "
              f"gave up on {gave_up} of {len(steps)} logs.")

    # --- print ----------------------------------------------------------------
    print(f"\n{'system':32s} {'n':>4s} {'accuracy':>9s} {'macro-F1':>9s} "
          f"{'logs sent to LLM':>17s}")
    for r in results:
        share = "" if r["llm_share"] is None else f"{r['llm_logs']:4d} ({r['llm_share']:.0%})"
        print(f"{r['system']:32s} {r['n']:4d} {r['accuracy']:9.1%} {r['macro_f1']:9.3f} "
              f"{share:>17s}  {r.get('note', '')}")

    # --- what the unlabelled logs really are ----------------------------------
    unl = Counter(r["human_label"] for r in rows if r["_rule"] == "unlabelled")
    if unl:
        total = sum(unl.values())
        print(f"\nWhat the {total} rule-unlabelled logs really are (your labels):")
        for label, k in unl.most_common():
            print(f"  {label:16s} {k:3d}  ({k / total:.0%})")

    # --- by language: classifier versus LLM -----------------------------------
    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["language"]].append(r)
    print(f"\n{'language':12s} {'n':>3s} {'classifier right':>17s} {'LLM right':>10s}")
    for lang in sorted(by_lang):
        rs = by_lang[lang]
        c_ok = sum(1 for r in rs if (clf_label(r, config.CONFIDENCE_THRESHOLD) or "other")
                   == r["human_label"])
        with_llm = [r for r in rs if r["_derivative"] or r["log_id"] in cache]
        l_ok = sum(1 for r in with_llm
                   if ("needs_traceback" if r["_derivative"] else cache[r["log_id"]]["category"])
                   == r["human_label"])
        llm_txt = f"{l_ok}/{len(with_llm)}" if with_llm else "-"
        print(f"{lang:12s} {len(rs):3d} {c_ok:>10d}/{len(rs):<6d} {llm_txt:>10s}")

    # --- save -----------------------------------------------------------------
    with open(RESULTS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "n", "accuracy", "macro_f1", "logs_sent_to_llm",
                    "share_sent_to_llm", "note", "llm_model"])
        for r in results:
            w.writerow([r["system"], r["n"], f"{r['accuracy']:.4f}", f"{r['macro_f1']:.4f}",
                        "" if r["llm_logs"] is None else r["llm_logs"],
                        "" if r["llm_share"] is None else f"{r['llm_share']:.4f}",
                        r.get("note", ""),
                        (model or "") if r["llm_logs"] else ""])
    with open(PER_LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["log_id", "language", "human_label", "rule_label", "classifier_guess",
                    "classifier_confidence", "derivative_rule", "llm_category",
                    "llm_steps", "llm_root_cause"])
        for r in rows:
            c = cache.get(r["log_id"], {})
            t = r["_triage"]
            w.writerow([r["log_id"], r["language"], r["human_label"], r["_rule"],
                        t.best_guess, f"{t.confidence:.4f}", int(r["_derivative"]),
                        c.get("category", ""), c.get("llm_steps", ""),
                        c.get("root_cause", "")])
    print(f"\nSaved {RESULTS}\nSaved {PER_LOG}")
    print("\nReminder: classes were sampled in equal numbers, so 'accuracy' here "
          "is not accuracy on real traffic (PROJECT_LOG known issue 16).")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate diagnosis on the hand-labelled sample.")
    ap.add_argument("--sample", default=str(SAMPLE))
    ap.add_argument("--no-llm", action="store_true",
                    help="score rules and classifier only (uses cached LLM answers if any)")
    ap.add_argument("--limit", type=int, help="at most this many NEW LLM diagnoses")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="seconds to wait between logs (for rate limits)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    llm = None
    if not args.no_llm:
        from .llm import make_llm
        try:
            llm = make_llm()
        except RuntimeError as e:
            print(e, file=sys.stderr)
            print("Use --no-llm to score the rules and the classifier without a key.",
                  file=sys.stderr)
            return 2
    evaluate(Path(args.sample), llm=llm, use_llm=not args.no_llm, delay=args.delay,
             limit=args.limit, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
