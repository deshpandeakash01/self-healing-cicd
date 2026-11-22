"""
Feature 3 - Tools: the only things the LLM can do.

WHY TOOLS
    The LLM cannot touch GitHub or the disk. It can only ask for one of the
    functions below to be run. So this list IS the agent's set of abilities
    during an investigation: look at the log, look at the repository, and hand
    in a diagnosis. None of them changes anything; acting happens later, in
    code, after the guardrail.

    submit_diagnosis is how the LLM answers. Making the answer a tool call
    with typed fields (instead of free text to be parsed) means the result is
    always structured the same way.

Every tool returns text and never raises: an error comes back as "ERROR: ..."
so the LLM can react to it instead of the run crashing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from . import config
from .guardrail import apply_replacements, normalise_path

# Same noise removal as clean_and_anchor() in clean_and_label.py, applied to
# the WHOLE log instead of a 45-line window, so the LLM can look anywhere.
_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T[\d:.]+Z")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def log_to_lines(raw_log: str) -> list[str]:
    lines = []
    for line in (raw_log or "").splitlines():
        line = _ANSI_RE.sub("", _TS_RE.sub("", line)).strip()
        if line:
            lines.append(line)
    return lines


@dataclass
class Context:
    """What the tools work on for one investigation."""
    log_lines: list = field(default_factory=list)
    gh: object = None                 # GitHubClient, or None in log-file mode
    ref: str | None = None            # commit / branch the files are read at
    files_read_full: set = field(default_factory=set)
    proposal: dict | None = None      # filled in by submit_diagnosis


class Replacement(BaseModel):
    old: str = Field(description="Text to replace, copied EXACTLY from the file "
                                 "(with its indentation). Must occur exactly once "
                                 "in the file; include neighbouring lines if needed.")
    new: str = Field(description="The text that takes its place")


class FileEdit(BaseModel):
    """One file to change. Give EITHER replacements OR content, not both."""
    path: str = Field(description="Repository-relative path, e.g. src/app.py")
    replacements: list[Replacement] = Field(
        default_factory=list,
        description="Preferred for an existing file: exact text replacements. "
                    "Works for files of any size.")
    content: str = Field(
        default="",
        description="Only for a NEW file, or a small file you read completely: "
                    "the COMPLETE new content of the file.")
    evidence: Literal["log", "code"] = Field(
        default="code",
        description="What backs this change. 'log': a line of the CI log shows "
                    "the problem this edit fixes (give log_lines). 'code': you "
                    "found it by reading the repository and no log line shows "
                    "it. Be honest: 'log' claims are checked against the log.")
    log_lines: str = Field(
        default="",
        description="For evidence='log': the log line numbers that show the "
                    "problem, e.g. '129-131' or '57, 212'.")
    why: str = Field(
        default="",
        description="One sentence: what is wrong in this file and how you know.")


def _cap(text: str, limit: int | None = None) -> str:
    limit = limit or config.MAX_TOOL_OUTPUT_CHARS
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more characters]"


def make_tools(ctx: Context) -> list:
    """Build the tool list bound to one investigation's context."""

    @tool
    def search_log(pattern: str, max_matches: int = 20) -> str:
        """Search the whole job log for a regular expression (case-insensitive).
        Returns matching lines with their line numbers."""
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(pattern), re.IGNORECASE)
        max_matches = max(1, min(int(max_matches), 50))
        hits = [f"{i}: {line[:400]}" for i, line in enumerate(ctx.log_lines, 1)
                if rx.search(line)]
        if not hits:
            return f"No lines match. The log has {len(ctx.log_lines)} lines."
        more = f"\n... {len(hits) - max_matches} more matches" if len(hits) > max_matches else ""
        return _cap("\n".join(hits[:max_matches]) + more)

    @tool
    def read_log_lines(start: int, end: int) -> str:
        """Read log lines start..end (1-based, inclusive, at most 120 lines)."""
        total = len(ctx.log_lines)
        start = max(1, int(start))
        end = min(total, int(end), start + 119)
        if start > total or end < start:
            return f"ERROR: nothing in that range. The log has {total} lines."
        return _cap("\n".join(f"{i}: {ctx.log_lines[i - 1][:400]}"
                              for i in range(start, end + 1)))

    @tool
    def list_repo_files(path: str = "") -> str:
        """List files and folders in a repository directory ("" is the root)."""
        if ctx.gh is None:
            return "ERROR: no repository connected in this run."
        clean = normalise_path(path) if path.strip(" /.") else ""
        try:
            entries = ctx.gh.list_dir(clean, ref=ctx.ref)
        except Exception as e:                       # noqa: BLE001
            return f"ERROR: {e}"
        return _cap("\n".join(f"{e['type']:4s} {e['path']}" for e in entries[:300])
                    or "(empty directory)")

    def _fetch(path: str):
        """(clean path, text, error). The text of a file at the failing commit."""
        if ctx.gh is None:
            return "", None, "ERROR: no repository connected in this run."
        clean = normalise_path(path)
        if not clean:
            return "", None, f"ERROR: unacceptable path {path!r}"
        try:
            f = ctx.gh.get_file(clean, ref=ctx.ref)
        except Exception as e:                       # noqa: BLE001
            return clean, None, f"ERROR: {e}"
        if f is None:
            return clean, None, f"ERROR: {clean} does not exist."
        return clean, f["text"], None

    @tool
    def read_repo_file(path: str, start_line: int = 0, end_line: int = 0) -> str:
        """Read a file from the repository at the failing commit.

        Without start_line/end_line the whole file is returned if it is small
        enough; a large file returns its first part and its line count. Give
        start_line and end_line (1-based, inclusive) to read part of a file,
        for example the lines around the one named in a traceback.

        A first line in [square brackets] is a note from the tool, not part
        of the file."""
        clean, text, error = _fetch(path)
        if error:
            return error
        lines = text.splitlines(keepends=True)
        total = len(lines)
        start, end = int(start_line or 0), int(end_line or 0)
        if start <= 0 and end <= 0:
            if len(text) <= config.MAX_FILE_READ_CHARS:
                ctx.files_read_full.add(clean)       # may be replaced whole
                return text
            shown, size = 0, 0
            for line in lines:
                if size + len(line) > config.MAX_FILE_READ_CHARS:
                    break
                size, shown = size + len(line), shown + 1
            return (f"[{clean}: {total} lines, {len(text)} characters - too large "
                    f"to return whole. Below are lines 1-{shown}. Read other parts "
                    "with start_line and end_line. To change this file use "
                    "'replacements' in submit_diagnosis, not 'content'.]\n"
                    + "".join(lines[:shown]))
        start = max(1, start)
        end = min(total, end if end > 0 else start + config.MAX_FILE_READ_LINES - 1,
                  start + config.MAX_FILE_READ_LINES - 1)
        if start > total or end < start:
            return f"ERROR: nothing in that range. {clean} has {total} lines."
        part = "".join(lines[start - 1:end])
        if len(part) > config.MAX_FILE_READ_CHARS:
            return (f"ERROR: lines {start}-{end} of {clean} are more than "
                    f"{config.MAX_FILE_READ_CHARS} characters. Ask for fewer lines.")
        return f"[{clean} lines {start}-{end} of {total}]\n" + part

    @tool
    def submit_diagnosis(
        category: Literal["dependency", "flaky_infra", "lint_build",
                          "other_bot_infra", "test_failure", "config", "other"],
        root_cause: str,
        action: Literal["rerun", "fix_pr", "escalate"],
        explanation: str,
        file_edits: list[FileEdit] = Field(default_factory=list),
    ) -> str:
        """Hand in the final diagnosis and ONE proposed action. Call this
        exactly once, when the investigation is finished.

        category    : the kind of failure
        root_cause  : one or two sentences naming the actual cause
        action      : rerun (transient, no change needed), fix_pr (file edits),
                      or escalate (a human must handle it)
        explanation : the evidence, and for fix_pr what the change does
        file_edits  : for fix_pr only; one item per file. For an existing
                      file give 'replacements' (exact old text -> new text).
                      For a new file give 'content' (the complete file).
                      Every edit also says what backs it: evidence 'log'
                      (with log_lines) or 'code', and a one-sentence 'why'.
        """
        edits = [e.model_dump() if isinstance(e, BaseModel) else dict(e)
                 for e in (file_edits or [])]
        # Check replacements against the real file NOW, so a mis-quoted "old"
        # comes back as an error the LLM can correct. Without this, the
        # guardrail (which checks again, and is the authority) would turn a
        # one-character quoting slip straight into an escalation.
        if action == "fix_pr" and ctx.gh is not None:
            problems = []
            for e in edits:
                if not e.get("replacements"):
                    continue
                clean, text, error = _fetch(e.get("path", ""))
                if error:
                    problems.append(f"{e.get('path')}: {error}")
                    continue
                problems += [f"{clean}: {p}" for p in
                             apply_replacements(text, e["replacements"])[1]]
            if problems:
                return ("ERROR: the diagnosis was NOT accepted because an edit "
                        "cannot be applied:\n- " + "\n- ".join(problems)
                        + "\nRead the lines again, correct the edit and call "
                          "submit_diagnosis again, or escalate.")
        ctx.proposal = {
            "category": category,
            "root_cause": root_cause,
            "action": action,
            "explanation": explanation,
            "file_edits": edits,
        }
        return "Diagnosis received."

    return [search_log, read_log_lines, list_repo_files, read_repo_file,
            submit_diagnosis]
