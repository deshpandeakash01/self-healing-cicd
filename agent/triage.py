"""
Feature 1 - Triage: raw failed-job log in, routing decision out.

This is the agent's single entry point for "what kind of failure is this?".
It does three things, in this order:

  1. CLEAN   Runs clean_and_anchor() from clean_and_label.py - the exact
             function used to build the training data. The classifier was
             trained on cleaned 45-line windows, so it must be given the same
             kind of text. (predict.py skipped this step.)

  2. RULE    Checks the "derivative" patterns (job failed only because another
             job failed). The classifier was never trained on that category,
             so it cannot recognise it; the regex rules can. Bot/infra patterns
             are checked first and win, the same order clean_and_label.py uses.

  3. MODEL   Asks the classifier for a label and a confidence. If confidence
             is below the threshold the label becomes "unknown": the model must
             always pick one of its 5 classes, so low confidence is the only
             warning sign that it is guessing.

Possible labels returned:
    dependency, flaky_infra, lint_build, other_bot_infra, test_failure
    needs_traceback   -> the agent should look at the upstream job's log
    unknown           -> the agent should investigate from scratch

USAGE (from the project root)
    python -m agent.triage path\\to\\raw_log.txt
    python -m agent.triage log1.txt log2.txt --threshold 0.7 --show-log
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache

import joblib

from . import config

# clean_and_label.py lives in the project root, one level above this package.
if str(config.PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(config.PROJECT_ROOT))

# Imported, not copied, so cleaning and rules can never drift apart from the
# ones the training data was built with.
from clean_and_label import (  # noqa: E402
    BOT_INFRA_PATTERNS,
    DERIVATIVE_SIGNALS,
    clean_and_anchor,
)

_RULE_FLAGS = re.IGNORECASE | re.MULTILINE   # same flags as clean_and_label.classify


@dataclass
class TriageResult:
    label: str                    # final routing label (see module docstring)
    decided_by: str               # "classifier" | "low_confidence" |
                                  # "derivative_rule" | "empty_log"
    confidence: float             # classifier's highest class probability
    best_guess: str               # classifier's top class, even when not trusted
    probabilities: dict = field(default_factory=dict)   # class -> probability
    matched_rule: str = ""        # the regex that fired, for derivative_rule
    cleaned_log: str = ""         # the text the classifier actually saw

    def summary(self) -> str:
        probs = ", ".join(f"{c} {p:.2f}" for c, p in self.probabilities.items())
        lines = [
            f"label       : {self.label}",
            f"decided by  : {self.decided_by}",
            f"confidence  : {self.confidence:.3f}  (best guess: {self.best_guess})",
            f"all classes : {probs}",
        ]
        if self.matched_rule:
            lines.append(f"matched rule: {self.matched_rule}")
        return "\n".join(lines)


@lru_cache(maxsize=1)
def _load_bundle():
    """Load the model once and reuse it; loading on every call would be slow."""
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {config.MODEL_PATH}. "
            "Run train_classifiers.py first, or set SH_MODEL_PATH."
        )
    bundle = joblib.load(config.MODEL_PATH)
    return bundle["vectorizer"], bundle["classifier"]


def _first_match(patterns, text: str) -> str:
    for pat in patterns:
        if re.search(pat, text, _RULE_FLAGS):
            return pat
    return ""


def triage(raw_log: str, threshold: float | None = None) -> TriageResult:
    """Classify one raw GitHub Actions job log.

    raw_log   : the full log text exactly as downloaded
    threshold : confidence below which the label becomes "unknown";
                defaults to config.CONFIDENCE_THRESHOLD
    """
    if threshold is None:
        threshold = config.CONFIDENCE_THRESHOLD

    cleaned = clean_and_anchor(raw_log or "")
    if not cleaned.strip():
        # Nothing to classify. Do not let the model guess from empty text.
        return TriageResult(label="unknown", decided_by="empty_log",
                            confidence=0.0, best_guess="", cleaned_log="")

    vec, clf = _load_bundle()
    probs = clf.predict_proba(vec.transform([cleaned]))[0]
    ranked = sorted(zip(clf.classes_, probs), key=lambda cp: cp[1], reverse=True)
    probabilities = {str(c): float(p) for c, p in ranked}
    best_guess, confidence = str(ranked[0][0]), float(ranked[0][1])

    # Rule check for "failed because another job failed". Bot/infra patterns
    # take priority, mirroring the order in clean_and_label.classify().
    if not _first_match(BOT_INFRA_PATTERNS, cleaned):
        rule = _first_match(DERIVATIVE_SIGNALS, cleaned)
        if rule:
            return TriageResult(label="needs_traceback",
                                decided_by="derivative_rule",
                                confidence=confidence, best_guess=best_guess,
                                probabilities=probabilities,
                                matched_rule=rule, cleaned_log=cleaned)

    if confidence < threshold:
        return TriageResult(label="unknown", decided_by="low_confidence",
                            confidence=confidence, best_guess=best_guess,
                            probabilities=probabilities, cleaned_log=cleaned)

    return TriageResult(label=best_guess, decided_by="classifier",
                        confidence=confidence, best_guess=best_guess,
                        probabilities=probabilities, cleaned_log=cleaned)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Triage raw CI failure logs.")
    ap.add_argument("logs", nargs="+", help="path(s) to raw job log file(s)")
    ap.add_argument("--threshold", type=float, default=None,
                    help=f"confidence threshold (default {config.CONFIDENCE_THRESHOLD})")
    ap.add_argument("--show-log", action="store_true",
                    help="also print the cleaned text the classifier saw")
    args = ap.parse_args(argv)

    # Logs contain characters some Windows consoles cannot print.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    for path in args.logs:
        # Read the same way clean_and_label.py does, so results match training.
        with open(path, encoding="utf-8", errors="replace") as f:
            raw = f.read()
        result = triage(raw, args.threshold)
        print("=" * 72)
        print(path)
        print(result.summary())
        if args.show_log:
            print("-" * 72)
            print(result.cleaned_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
