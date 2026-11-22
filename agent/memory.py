"""
Fix memory and audit log.

AUDIT LOG  (data/agent_runs.jsonl)
    One JSON line per incident: what the classifier said, what the LLM
    diagnosed, what the guardrail decided, what was done, how it ended.
    WHY: every action must be explainable afterwards, and these records are
    the raw material for the evaluation and for retraining the classifier.

FIX MEMORY  (data/agent_fix_memory.jsonl)
    One JSON line per fix that PASSED verification. For a new failure the
    most similar past fixes are looked up and shown to the LLM.
    WHY: the same failures come back. Only verified fixes are stored, so the
    agent never learns from a fix that did not work.

Similarity uses the classifier's own TF-IDF vectorizer: it is already loaded,
needs no extra dependency, and is good at "this is the same error text".
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import config


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except ValueError:
                    continue          # a damaged line must not stop the agent
    return records


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# ------------------------------------------------------------------ audit log
def write_audit(record: dict) -> None:
    _append(config.AUDIT_LOG, record)


# ----------------------------------------------------------------- fix memory
def remember_fix(*, repo, category, root_cause, action, explanation,
                 edits, cleaned_log) -> None:
    _append(config.FIX_MEMORY, {
        "time": now(),
        "repo": repo,
        "category": category,
        "root_cause": root_cause,
        "action": action,
        "explanation": explanation,
        # Paths only: file contents can be large, and the explanation is what
        # helps the LLM next time.
        "files": [e["path"] for e in edits],
        "cleaned_log": cleaned_log,
    })


def recall_fixes(cleaned_log: str, k: int | None = None,
                 min_similarity: float | None = None) -> list[dict]:
    """Most similar verified fixes for this log, best first."""
    k = config.MEMORY_TOP_K if k is None else k
    min_similarity = (config.MEMORY_MIN_SIMILARITY
                      if min_similarity is None else min_similarity)
    records = [r for r in _read(config.FIX_MEMORY) if r.get("cleaned_log")]
    if not records or not cleaned_log.strip():
        return []

    from .triage import _load_bundle          # reuse the loaded vectorizer
    vec, _ = _load_bundle()
    query = vec.transform([cleaned_log])
    stored = vec.transform([r["cleaned_log"] for r in records])
    # TF-IDF rows are length-normalised, so the dot product is cosine similarity.
    sims = (stored @ query.T).toarray().ravel()

    ranked = sorted(zip(sims, records), key=lambda sr: sr[0], reverse=True)
    out = []
    for sim, rec in ranked[:k]:
        if sim >= min_similarity:
            item = {key: rec.get(key) for key in
                    ("time", "repo", "category", "root_cause", "action",
                     "explanation", "files")}
            item["similarity"] = round(float(sim), 3)
            out.append(item)
    return out
