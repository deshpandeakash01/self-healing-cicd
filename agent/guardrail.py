"""
Guardrail: plain-code checks on a proposed action BEFORE anything happens.

WHY THIS IS CODE AND NOT A PROMPT
    The LLM reads CI logs and repository files. That text can be written by
    anyone who can open a pull request, so it may contain instructions aimed
    at the agent ("edit the workflow to print the secrets"). A prompt asking
    the LLM to behave can be talked around; an if-statement cannot.

RULE
    A proposal that fails ANY check is not "partly allowed". It becomes
    "escalate" and the reasons are recorded, so a human sees what was blocked.
"""

from __future__ import annotations

import re

from . import config
from .playbooks import ACTIONS, CATEGORIES, allowed_actions


def normalise_path(path: str) -> str:
    """Repo-relative path with forward slashes, or "" if it is not acceptable."""
    p = (path or "").strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    if not p or p.startswith("/") or (len(p) > 1 and p[1] == ":"):
        return ""                      # empty or absolute
    if any(part in ("..", "") for part in p.split("/")):
        return ""                      # escapes the repo, or has "//"
    return p


def is_protected(path: str) -> bool:
    low = path.lower()
    name = low.rsplit("/", 1)[-1]
    return (low.startswith(config.PROTECTED_PREFIXES)
            or name in config.PROTECTED_NAMES
            or name.endswith(config.PROTECTED_SUFFIXES))


def apply_replacements(text: str, replacements: list) -> tuple[str, list[str]]:
    """
    Apply exact text replacements to a file's content.

    Each replacement is {"old", "new"}. "old" must be found EXACTLY ONCE in
    the file as it is at that moment, so there is never a guess about which
    place was meant. Returns (new_text, problems); with any problem the text
    must not be used.

    A file with Windows line endings is handled: the LLM sees and writes
    "\n", so "old" and "new" are converted to "\r\n" when that is what
    the file uses.
    """
    problems: list[str] = []
    if not isinstance(text, str) or not text:
        return "", ["the file's current content could not be read"]
    if "\ufffd" in text:
        return text, ["the file is not valid UTF-8 text; editing it could corrupt it"]
    crlf = "\r\n" in text
    for i, r in enumerate(replacements or [], 1):
        old, new = (r or {}).get("old"), (r or {}).get("new")
        if not isinstance(old, str) or not old.strip():
            problems.append(f"replacement {i}: 'old' is empty")
            continue
        if not isinstance(new, str):
            problems.append(f"replacement {i}: 'new' is missing")
            continue
        if crlf:
            old = old.replace("\r\n", "\n").replace("\n", "\r\n")
            new = new.replace("\r\n", "\n").replace("\n", "\r\n")
        if old == new:
            problems.append(f"replacement {i}: 'old' and 'new' are identical")
            continue
        found = text.count(old)
        if found == 0:
            problems.append(f"replacement {i}: 'old' was not found in the file. It "
                            "must match the file exactly, including indentation")
        elif found > 1:
            problems.append(f"replacement {i}: 'old' occurs {found} times in the "
                            "file; include more surrounding lines so it is unique")
        else:
            text = text.replace(old, new)
    return text, problems


def check(proposal: dict, *, attempt: int, reruns_done: int, online: bool,
          existing_files: dict | None = None,
          files_read_full: set | None = None,
          current_files: dict | None = None) -> dict:
    """
    proposal        : {"category", "action", "root_cause", "explanation",
                       "file_edits": [{"path", "content"} or
                                      {"path", "replacements": [{"old", "new"}]}]}
    attempt         : how many actions were already tried for this incident
    reruns_done     : how many reruns were already done for this incident
    online          : False when there is no repository (log-file mode)
    existing_files  : {path: True/False} - does each edited file exist now?
    files_read_full : paths the LLM read completely during this investigation
    current_files   : {path: text} - current content of each edited file that
                      exists (needed to apply "replacements")

    Returns {"action", "ok", "reasons", "edits"}. Each item of "edits" is
    {"path", "content", "kind"}: the cleaned path and the COMPLETE new
    content, whichever way the edit was given.
    """
    existing_files = dict(existing_files or {})
    files_read_full = files_read_full or set()
    current_files = current_files or {}
    for path in current_files:
        existing_files.setdefault(path, True)
    reasons: list[str] = []
    action = proposal.get("action")
    category = proposal.get("category")
    edits_in = proposal.get("file_edits") or []
    edits: list[dict] = []

    if action not in ACTIONS:
        reasons.append(f"unknown action {action!r}")
    if category not in CATEGORIES:
        reasons.append(f"unknown category {category!r}")

    if not reasons and action != "escalate":
        # 1. Is this action allowed for this kind of failure?
        if action not in allowed_actions(category):
            reasons.append(f"action {action!r} is not allowed for category "
                           f"{category!r} (allowed: {allowed_actions(category)})")
        # 2. Attempt limit.
        if attempt >= config.MAX_FIX_ATTEMPTS:
            reasons.append(f"attempt limit reached ({config.MAX_FIX_ATTEMPTS})")
        # 3. No repository to act on.
        if not online:
            reasons.append("no repository connected (log-file mode): diagnosis only")

    if not reasons and action == "rerun" and reruns_done >= 1:
        # A second rerun of the same failure is hoping, not fixing.
        reasons.append("this failure was already re-run once")

    if not reasons and action == "fix_pr":
        if not edits_in:
            reasons.append("fix_pr without any file edits")
        if len(edits_in) > config.MAX_EDIT_FILES:
            reasons.append(f"too many files edited ({len(edits_in)} > "
                           f"{config.MAX_EDIT_FILES})")
        seen = set()
        for e in edits_in:
            raw_path = e.get("path", "")
            path = normalise_path(raw_path)
            content = e.get("content")
            if not path:
                reasons.append(f"unacceptable path {raw_path!r}")
                continue
            if path in seen:
                reasons.append(f"{path}: edited twice in one proposal")
            seen.add(path)
            if is_protected(path):
                reasons.append(f"{path}: protected path, the agent may not edit it")
            replacements = e.get("replacements") or []
            if replacements:
                # Exact text replacements: only the quoted text changes, so
                # the file need not have been read in full and may be large.
                if isinstance(content, str) and content.strip():
                    reasons.append(f"{path}: give either 'content' or "
                                   "'replacements', not both")
                if len(replacements) > config.MAX_REPLACEMENTS:
                    reasons.append(f"{path}: too many replacements "
                                   f"({len(replacements)} > {config.MAX_REPLACEMENTS})")
                added = sum(len(str((r or {}).get("new") or "").encode("utf-8"))
                            for r in replacements)
                if added > config.MAX_EDIT_BYTES:
                    reasons.append(f"{path}: edit larger than {config.MAX_EDIT_BYTES} bytes")
                if not existing_files.get(path) or path not in current_files:
                    reasons.append(f"{path}: 'replacements' need an existing file "
                                   "whose content could be read")
                    continue
                new_text, problems = apply_replacements(current_files[path], replacements)
                reasons.extend(f"{path}: {p}" for p in problems)
                edits.append({"path": path, "content": new_text, "kind": "replacements"})
                continue
            if not isinstance(content, str) or not content.strip():
                reasons.append(f"{path}: empty content (would wipe the file)")
            elif len(content.encode("utf-8")) > config.MAX_EDIT_BYTES:
                reasons.append(f"{path}: edit larger than {config.MAX_EDIT_BYTES} bytes")
            # An edit replaces the whole file. Replacing a file the LLM never
            # read in full would silently destroy whatever it did not see.
            if existing_files.get(path) and path not in files_read_full:
                reasons.append(f"{path}: existing file was not read in full "
                               "before being edited")
            edits.append({"path": path, "content": content, "kind": "whole_file"})

    if reasons:
        return {"action": "escalate", "ok": False, "reasons": reasons, "edits": []}
    return {"action": action, "ok": True, "reasons": [],
            "edits": edits if action == "fix_pr" else []}


# --- Evidence: which changes does the CI log actually back? (log 3.44) ------
#
# The agent may fix problems it finds by reading the code, even when no log
# line shows them yet (the gauntlet, 3.43). That is allowed, but a reviewer
# must be able to tell those changes apart. So every edit says what backs it,
# and a claim of "the log shows this" is CHECKED here: the cited log lines
# must mention the file or the text being replaced. A claim that fails the
# check is not rejected (that would block a good fix over a wrong line
# number); it is downgraded to "found by reading the code" and the PR says so.

_STOPWORDS = {
    "import", "from", "return", "assert", "self", "true", "false", "none",
    "print", "class", "with", "else", "elif", "pass", "raise", "async",
    "await", "lambda", "yield", "while", "error", "test", "tests",
}
_LOG_WINDOW = 1            # a cited line may be off by this many lines
_MAX_CITED_LINES = 120


def parse_line_spec(spec: str, total: int) -> list[int]:
    """'129-131, 284' -> [129, 130, 131, 284], kept within 1..total."""
    lines: list[int] = []
    for a, b, single in re.findall(r"(\d+)\s*(?:-|–|to)\s*(\d+)|(\d+)", spec or ""):
        lo, hi = (int(a), int(b)) if a else (int(single), int(single))
        if hi < lo:
            lo, hi = hi, lo
        hi = min(hi, lo + 29)                     # one range: at most 30 lines
        lines += [n for n in range(lo, hi + 1) if 1 <= n <= total]
        if len(lines) >= _MAX_CITED_LINES:
            break
    return sorted(set(lines))[:_MAX_CITED_LINES]


def _edit_tokens(path: str, replacements: list) -> set[str]:
    """Words a log line about this edit would be expected to contain."""
    name = path.rsplit("/", 1)[-1].lower()
    tokens = {name}
    stem = name.rsplit(".", 1)[0]
    if len(stem) >= 3 and stem not in _STOPWORDS:
        tokens.add(stem)
    for r in replacements or []:
        for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", str((r or {}).get("old") or "")):
            if w.lower() not in _STOPWORDS:
                tokens.add(w.lower())
    return tokens


def check_evidence(edit: dict, log_lines: list[str]) -> dict:
    """
    edit      : one proposed file edit, with the optional fields
                "evidence" ("log" or "code"), "log_lines" (e.g. "129-131")
                and "why" (one sentence)
    log_lines : the log the LLM was given (main job + other failed jobs)

    Returns {"path", "evidence", "log_lines", "why", "note"}. "evidence" is
    "log" only when the cited lines really mention the file or the replaced
    text; anything else is "code" (shown to the reviewer as not backed by
    the log). "note" explains a downgrade.
    """
    path = normalise_path(edit.get("path", "")) or str(edit.get("path", ""))
    claimed = str(edit.get("evidence") or "code").strip().lower()
    spec = str(edit.get("log_lines") or "").strip()
    out = {"path": path, "evidence": "code", "log_lines": "",
           "why": str(edit.get("why") or "").strip(), "note": ""}
    if claimed != "log":
        return out
    cited = parse_line_spec(spec, len(log_lines))
    if not cited:
        out["note"] = (f"claimed log evidence, but cited no valid log line "
                       f"({spec or 'none given'})")
        return out
    window = sorted({n for c in cited
                     for n in range(c - _LOG_WINDOW, c + _LOG_WINDOW + 1)
                     if 1 <= n <= len(log_lines)})
    text = "\n".join(log_lines[n - 1] for n in window).lower()
    tokens = _edit_tokens(path, edit.get("replacements") or [])
    if any(t in text for t in tokens):
        out.update(evidence="log", log_lines=spec)
    else:
        out["note"] = (f"cited log lines {spec}, but they do not mention "
                       f"{path} or the text being changed")
    return out
