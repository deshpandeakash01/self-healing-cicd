"""
Tests for the agent. No API key, no network, nothing written to GitHub.

HOW
    The two outside services are replaced by stand-ins:
      ScriptedLLM  - plays back a fixed list of replies instead of calling Gemini
      FakeGitHub   - an in-memory repository with jobs, logs, branches and CI
    Everything else is the real code: the real classifier, the real cleaning,
    the real graph, guardrail, memory and audit log, on four real CI logs
    from the dataset, kept in tests/fixtures/ so the tests run on a fresh
    clone. The evaluation test also needs data/labelled_logs.csv (not in
    the repository); without it that one test is skipped.

WHAT THIS PROVES AND WHAT IT DOES NOT
    It proves the agent's own logic: routing, the tool loop, the guardrail,
    acting, verifying, retrying, remembering. It does NOT prove that Gemini
    diagnoses well or that the real GitHub API behaves as FakeGitHub assumes.
    Those need a real run against your test repository.

RUN (from the project root)
    python tests\\test_agent.py            or            python -m pytest tests
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402

from agent import config, guardrail, memory  # noqa: E402
from agent.github_client import ReadOnlyError  # noqa: E402
from agent.graph import Agent  # noqa: E402

# Four real failed-CI logs (public GitHub Actions runs), copied from the
# dataset so the tests need nothing outside the repository.
LOGS = ROOT / "tests" / "fixtures"
LOG_FLAKY = LOGS / "etcd-io__etcd__116750249015.txt"              # flaky_infra 0.97
LOG_TEST = LOGS / "pytest-dev__pytest__122700520097.txt"           # test_failure 0.93
LOG_DERIV = LOGS / "pytest-dev__pytest__124132246976.txt"          # needs_traceback
LOG_UNKNOWN = LOGS / "elastic__elasticsearch__121492986837.txt"    # low confidence


class Skipped(Exception):
    """A test that cannot run here (missing optional data)."""


def skip(reason: str):
    import os
    if os.environ.get("PYTEST_CURRENT_TEST"):     # running under pytest
        import pytest
        pytest.skip(reason)
    raise Skipped(reason)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# ----------------------------------------------------------------- stand-ins
_n = 0


def call(name, **args):
    """An LLM reply that calls one tool."""
    global _n
    _n += 1
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": f"call_{_n}", "type": "tool_call"}])


class ScriptedLLM:
    def __init__(self, steps, model=None):
        self.steps, self.calls, self.seen = list(steps), 0, []
        if model:
            self.model = model

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        self.seen.append(list(messages))
        step = self.steps.pop(0) if self.steps else AIMessage(content="I am done.")
        return step(messages) if callable(step) else step


class FakeGitHub:
    """In-memory repository. `ci` decides whether a tree passes CI."""

    def __init__(self, files, jobs, read_only=False, rerun_passes=True, ci=None):
        self.read_only = read_only
        self.trees = {"sha0": dict(files)}
        self.branches = {"main": "sha0"}
        self.jobs = {j["id"]: j for j in jobs}
        self.run = {"id": 100, "status": "completed", "conclusion": "failure",
                    "run_attempt": 1, "head_branch": "main", "head_sha": "sha0"}
        self.rerun_passes = rerun_passes
        self.ci = ci or (lambda tree: True)
        self.checks = {}
        self.writes, self.prs, self.comments = [], [], []
        self.commit_comments = []
        self.commit_messages = []
        self._next = 9000

    def _w(self, what):
        if self.read_only:
            raise ReadOnlyError(what)
        self.writes.append(what)

    def _tree(self, ref):
        return self.trees[self.branches.get(ref, ref)]

    # reads
    def get_run(self, run_id):
        if run_id != 100:       # CI on a fix branch is a different run, as on GitHub
            fix = next(b for b in self.branches if b.startswith("self-heal/"))
            return dict(self.run, id=run_id, head_branch=fix, head_sha=self.branches[fix])
        return dict(self.run)

    def get_job(self, job_id):
        j = self.jobs[job_id]
        return {"id": j["id"], "run_id": j.get("run_id", 100), "name": j["name"],
                "conclusion": j.get("conclusion", "failure"),
                "head_sha": j.get("head_sha", "sha0"),
                "steps": [{"name": "Run tests", "conclusion": "failure"}]}

    def get_job_log(self, job_id):
        return self.jobs[job_id]["log"]

    def failed_jobs(self, run_id):
        failed = [j for j in self.jobs.values() if j.get("conclusion", "failure") == "failure"
                  and j.get("attempt", 1) == self.run["run_attempt"]]
        return sorted(failed, key=lambda j: j.get("completed_at", ""))

    def list_check_runs(self, ref):
        return self.checks.get(ref, [])

    def get_default_branch(self):
        return "main"

    def branch_sha(self, branch):
        return self.branches.get(branch)

    def get_file(self, path, ref=None):
        tree = self._tree(ref or "main")
        return {"path": path, "sha": "x", "text": tree[path]} if path in tree else None

    def list_dir(self, path="", ref=None):
        return [{"path": p, "type": "file"} for p in sorted(self._tree(ref or "main"))]

    # writes
    def rerun_failed_jobs(self, run_id):
        self._w("rerun")
        self.run["run_attempt"] += 1
        self.run["conclusion"] = "success" if self.rerun_passes else "failure"
        if not self.rerun_passes:
            self._next += 1
            old = next(iter(self.jobs.values()))
            self.jobs[self._next] = dict(old, id=self._next,
                                         attempt=self.run["run_attempt"])

    def create_branch(self, branch, from_sha):
        self._w(f"branch {branch}")
        self.branches[branch] = from_sha

    def commit_files(self, branch, parent_sha, files, message, create_branch):
        """One commit with every file, as the real client (3.41). CI runs once."""
        self._w(f"branch {branch}" if create_branch else f"update {branch}")
        self.commit_messages.append(message)
        self._next += 1
        sha = f"sha{self._next}"
        self.trees[sha] = dict(self.trees[parent_sha], **dict(files))
        self.branches[branch] = sha
        for path, _ in files:
            self._w(f"commit {path}")
        passed = self.ci(self.trees[sha])
        self._next += 1
        self.checks[sha] = [{"id": self._next, "name": "test", "status": "completed",
                             "conclusion": "success" if passed else "failure"}]
        if not passed:
            self.jobs[self._next] = {"id": self._next, "name": "test on fix branch",
                                     "log": read(LOG_TEST), "head_sha": sha,
                                     "attempt": -1, "run_id": 500}
        return sha

    def put_file(self, path, text, branch, message):
        self._w(f"commit {path}")
        self._next += 1
        sha = f"sha{self._next}"
        self.trees[sha] = dict(self._tree(branch), **{path: text})
        self.branches[branch] = sha
        passed = self.ci(self.trees[sha])
        self._next += 1
        self.checks[sha] = [{"id": self._next, "name": "test", "status": "completed",
                             "conclusion": "success" if passed else "failure"}]
        if not passed:
            self.jobs[self._next] = {"id": self._next, "name": "test on fix branch",
                                     "log": read(LOG_TEST), "head_sha": sha,
                                     "attempt": -1, "run_id": 500}
        return sha

    def open_pull_request(self, title, head, base, body):
        self._w("pull request")
        self.prs.append({"title": title, "head": head, "base": base, "body": body})
        return {"number": len(self.prs), "url": f"https://example.test/pr/{len(self.prs)}"}

    def comment(self, number, body):
        self._w("comment")
        self.comments.append(body)

    def comment_on_commit(self, sha, body):
        self._w("commit comment")
        self.commit_comments.append((sha, body))
        return f"https://example.test/commit/{sha}#comment"


BROKEN = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"
STILL_WRONG = "def add(a, b):\n    return a * b\n"
FILES = {"calc.py": BROKEN, "test_calc.py": "from calc import add\n",
         ".github/workflows/ci.yml": "name: CI\n"}


def changes(gh):
    """Writes that change the repository or CI. A comment on the failing commit
    (the hand-over notice, 17 Nov) is not one."""
    return [w for w in gh.writes if w != "commit comment"]


def job(job_id, log_path, name="test", **extra):
    return dict({"id": job_id, "name": name, "log": read(log_path)}, **extra)


def fresh(tmp):
    """Send audit log and fix memory to a temp folder; make polling instant."""
    config.AUDIT_LOG = Path(tmp) / "runs.jsonl"
    config.FIX_MEMORY = Path(tmp) / "memory.jsonl"
    config.POLL_INTERVAL = 0
    config.VERIFY_START_TIMEOUT = 0.2
    config.VERIFY_TIMEOUT = 5


def audit():
    return [json.loads(line) for line in config.AUDIT_LOG.read_text().splitlines()]


def fix_script(content=FIXED):
    return [call("read_repo_file", path="calc.py"),
            call("submit_diagnosis", category="test_failure",
                 root_cause="add() subtracts instead of adding",
                 action="fix_pr", explanation="Changed - to + in add().",
                 file_edits=[{"path": "calc.py", "content": content}])]


def passes(tree):
    return "a + b" in tree["calc.py"]


# --------------------------------------------------------------------- tests
def test_flaky_is_rerun_without_calling_the_llm(tmp):
    gh = FakeGitHub(FILES, [job(1, LOG_FLAKY)])
    llm = ScriptedLLM([])
    s = Agent(llm, gh, dry_run=False, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert s["outcome"] == "healed"
    assert llm.calls == 0 and s["llm_calls"] == 0
    assert gh.writes == ["rerun"]
    assert audit()[0]["attempts"][0]["proposal"]["source"] == "classifier"
    assert len(memory.recall_fixes(s["triage"]["cleaned_log"])) == 1


def test_flaky_below_the_rerun_threshold_goes_to_the_llm(tmp):
    """Confident enough to be a label (>= 0.6) but not to act alone (< 0.99)."""
    saved = config.DIRECT_RERUN_THRESHOLD
    config.DIRECT_RERUN_THRESHOLD = 0.99          # the test log is at 0.97
    try:
        gh = FakeGitHub(FILES, [job(1, LOG_FLAKY)])
        llm = ScriptedLLM([call("submit_diagnosis", category="flaky_infra",
                                root_cause="network timeout", action="rerun",
                                explanation="transient")])
        s = Agent(llm, gh, dry_run=False, verbose=False).run(
            repo="o/r", run_id=100, job_id=1)
        assert llm.calls == 1 and s["outcome"] == "healed"
        assert "'flaky_infra' at 0.97" in llm.seen[0][1].content    # passed as a hint
        s2 = Agent(None, FakeGitHub(FILES, [job(1, LOG_FLAKY)]), dry_run=False,
                   mode="classifier_only", verbose=False).run(
            repo="o/r", run_id=100, job_id=1)
        assert s2["outcome"] == "escalated"        # no LLM, and not sure enough to rerun
    finally:
        config.DIRECT_RERUN_THRESHOLD = saved


def test_dry_run_writes_nothing(tmp):
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)], read_only=True)
    s = Agent(ScriptedLLM(fix_script()), gh, dry_run=True, verbose=False).run(
        repo="o/r", run_id=100, job_id=1)
    assert s["outcome"] == "dry_run"
    assert gh.writes == []
    plan = s["action_result"]["plan"]
    assert plan["files"][0]["path"] == "calc.py" and plan["base"] == "main"
    assert not config.FIX_MEMORY.exists()        # unverified -> not remembered


def test_fix_pr_heals(tmp):
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)], ci=passes)
    llm = ScriptedLLM(fix_script())
    s = Agent(llm, gh, dry_run=False, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert s["outcome"] == "healed"
    assert llm.calls == 2
    assert gh.writes == ["branch self-heal/run-100-job-1", "commit calc.py", "pull request"]
    assert gh.prs[0]["base"] == "main" and "Review it before merging" in gh.prs[0]["body"]
    assert gh.branches["main"] == "sha0"         # default branch never touched
    assert gh._tree("self-heal/run-100-job-1")["calc.py"] == FIXED


def test_error_while_checking_ci_still_leaves_a_record(tmp):
    """The pull request exists by then; losing the record would hide a change."""
    from agent.github_client import GitHubError

    class CannotReadChecks(FakeGitHub):
        def list_check_runs(self, ref):
            raise GitHubError(502, "Bad Gateway")

    gh = CannotReadChecks(FILES, [job(1, LOG_TEST)], ci=passes)
    s = Agent(ScriptedLLM(fix_script()), gh, dry_run=False,
              verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert s["outcome"] == "unverified" and "could not check CI" in s["verify"]["note"]
    assert gh.writes[-1] == "pull request"
    assert audit()[-1]["outcome"] == "unverified" and audit()[-1]["pr"]


class NoCheckRuns(FakeGitHub):
    """A private repository with a fine-grained token: check runs are closed,
    the Actions runs of a commit are not."""

    def list_check_runs(self, ref):
        from agent.github_client import GitHubError
        raise GitHubError(403, "Resource not accessible by personal access token")

    def list_workflow_runs(self, head_sha):
        runs = FakeGitHub.list_check_runs(self, head_sha)
        return [{"id": 900 + i, "name": "CI", "status": r["status"],
                 "conclusion": r["conclusion"], "_job": r["id"]}
                for i, r in enumerate(runs)]

    def failed_jobs(self, run_id):
        if run_id >= 900:                   # a run on the fix branch
            sha = self.branches[next(b for b in self.branches if b.startswith("self-heal/"))]
            return [{"id": r["id"], "name": r["name"]}
                    for r in FakeGitHub.list_check_runs(self, sha)
                    if r["conclusion"] == "failure"]
        return FakeGitHub.failed_jobs(self, run_id)


def test_ci_is_verified_without_check_runs(tmp):
    gh = NoCheckRuns(FILES, [job(1, LOG_TEST)], ci=passes)
    s = Agent(ScriptedLLM(fix_script()), gh, dry_run=False,
              verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert s["outcome"] == "healed"
    # and a fix that does not work is still noticed and retried
    gh = NoCheckRuns(FILES, [job(1, LOG_TEST)], ci=passes)
    llm = ScriptedLLM(fix_script(STILL_WRONG) + fix_script(FIXED))
    s = Agent(llm, gh, dry_run=False, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert s["outcome"] == "healed" and len(s["attempts"]) == 2


def test_failed_fix_is_retried_with_the_new_log(tmp):
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)], ci=passes)
    llm = ScriptedLLM(fix_script(STILL_WRONG) + fix_script(FIXED))
    s = Agent(llm, gh, dry_run=False, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert s["outcome"] == "healed"
    assert len(s["attempts"]) == 2 and s["attempts"][0]["verify"]["status"] == "failed"
    assert len(gh.prs) == 1                      # second attempt reuses the PR
    second_prompt = llm.seen[2][1].content
    assert "PREVIOUS ATTEMPT DID NOT FIX" in second_prompt
    # the retry read the file from the fix branch, i.e. saw the first edit
    assert llm.seen[3][-1].content == STILL_WRONG
    # Seen on 16 Nov with real GitHub: the failing job of the fix branch belongs
    # to another run. The record must still name the run first reported ...
    rec = audit()[-1]
    assert rec["run_id"] == 100 and rec["job_name"] == "test"
    assert [a["job_id"] for a in rec["attempts"]] == [1, s["attempts"][1]["job_id"]]
    # ... and the second commit is explained on the pull request already open
    assert len(gh.comments) == 1 and "attempt 2" in gh.comments[0]
    assert "add() subtracts instead of adding" in gh.comments[0]
    assert gh.writes[-1] == "comment" or "comment" in gh.writes


def test_gives_up_after_the_attempt_limit(tmp):
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)], ci=passes)
    llm = ScriptedLLM(fix_script(STILL_WRONG) + fix_script("def add(a, b):\n    return 0\n")
                      + fix_script(FIXED))
    s = Agent(llm, gh, dry_run=False, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert s["outcome"] == "failed"
    assert len(s["attempts"]) == config.MAX_FIX_ATTEMPTS
    # one comment explaining the second commit, then the hand-over to a human
    assert len(gh.comments) == 2 and "attempt 2" in gh.comments[0]
    assert "attempts did not make it pass" in gh.comments[-1]
    assert "A human needs to take over" in gh.comments[-1]
    assert not config.FIX_MEMORY.exists()


def test_guardrail_blocks_workflow_edit(tmp):
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)])
    llm = ScriptedLLM([
        call("read_repo_file", path=".github/workflows/ci.yml"),
        call("submit_diagnosis", category="test_failure", root_cause="x",
             action="fix_pr", explanation="y",
             file_edits=[{"path": ".github/workflows/ci.yml",
                          "content": "name: CI\n# print secrets\n"}])])
    s = Agent(llm, gh, dry_run=False, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert s["outcome"] == "escalated" and changes(gh) == []
    assert "protected path" in s["decision"]["reasons"][0]


def test_handover_is_posted_on_the_failing_commit_in_live_mode(tmp):
    """Automated runs have nobody watching: a hand-over must reach a person."""
    blocked = [call("read_repo_file", path=".github/workflows/ci.yml"),
               call("submit_diagnosis", category="config", root_cause="typo in ci.yml",
                    action="fix_pr", explanation="ruff chekc should be ruff check",
                    file_edits=[{"path": ".github/workflows/ci.yml", "content": "x\n"}])]
    gh = FakeGitHub(FILES, [job(1, LOG_TEST, head_sha="sha0")])
    s = Agent(ScriptedLLM(blocked), gh, dry_run=False, verbose=False).run(
        repo="o/r", run_id=100, job_id=1)
    assert s["outcome"] == "escalated" and gh.writes == ["commit comment"]
    sha, body = gh.commit_comments[0]
    assert sha == "sha0" and "needs a person" in body and "typo in ci.yml" in body
    assert "safety rules blocked it" in body and "not allowed for category" in body
    assert "actions/runs/100" in body and audit()[-1]["notified"]
    # dry run: nothing is posted
    gh = FakeGitHub(FILES, [job(1, LOG_TEST, head_sha="sha0")], read_only=True)
    Agent(ScriptedLLM(blocked), gh, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert gh.commit_comments == [] and audit()[-1]["notified"] is None
    # notify=False (--no-notify): nothing is posted
    gh = FakeGitHub(FILES, [job(1, LOG_TEST, head_sha="sha0")])
    Agent(ScriptedLLM(blocked), gh, dry_run=False, verbose=False, notify=False).run(
        repo="o/r", run_id=100, job_id=1)
    assert gh.commit_comments == []
    # a fix that worked is not announced on the commit (the pull request says it)
    gh = FakeGitHub(FILES, [job(1, LOG_TEST, head_sha="sha0")], ci=passes)
    s = Agent(ScriptedLLM(fix_script()), gh, dry_run=False, verbose=False).run(
        repo="o/r", run_id=100, job_id=1)
    assert s["outcome"] == "healed" and gh.commit_comments == []


def test_overloaded_gemini_is_retried_but_a_used_up_quota_is_not(tmp):
    from agent.run import error_comment, overloaded, run_with_retries
    busy = {"outcome": "error", "llm_error": "Gemini is overloaded (503: ...)"}
    quota = {"outcome": "error", "llm_error": "Gemini quota used up for x"}
    done = {"outcome": "escalated"}
    results, waits = [busy, busy, done], []
    s = run_with_retries(lambda: results.pop(0), 3, 120, sleep=waits.append, say=lambda m: 0)
    assert s is done and waits == [120, 120]
    results, waits = [quota, done], []
    s = run_with_retries(lambda: results.pop(0), 3, 120, sleep=waits.append, say=lambda m: 0)
    assert s is quota and waits == []                   # a quota lasts until tomorrow
    results, waits = [busy] * 5, []
    s = run_with_retries(lambda: results.pop(0), 2, 1, sleep=waits.append, say=lambda m: 0)
    assert overloaded(s) and len(waits) == 2           # gives up after N retries
    assert not overloaded(dict(busy, pr={"number": 1}))   # never redo a run that wrote
    text = error_comment(dict(quota, incident={"run_id": 7}), "o/r")
    assert "actions/runs/7" in text and "quota used up" in text


def test_other_failed_jobs_are_shown_to_the_llm(tmp):
    """Two jobs failing for different reasons: one fix must cover both (3.38)."""
    tail_marker = read(LOG_FLAKY).splitlines()[-1].strip()
    gh = FakeGitHub(FILES, [job(1, LOG_TEST, completed_at="1"),
                            job(2, LOG_FLAKY, name="lint", completed_at="2")])
    done = call("submit_diagnosis", category="test_failure", root_cause="x",
                action="escalate", explanation="y")
    llm = ScriptedLLM([call("search_log", pattern="ALSO FAILED"), done])
    s = Agent(llm, gh, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    opening = llm.seen[0][1].content
    assert "ALSO FAILED in this run: lint" in opening
    assert s["other_jobs"] == ["lint"] and tail_marker in s["other_jobs_log"]
    assert "lint" in llm.seen[1][-1].content            # search_log found the section
    # the classifier still judged the main job only
    assert s["triage"]["cleaned_log"] and tail_marker not in s["triage"]["cleaned_log"]
    # a run with one failed job adds nothing
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)])
    llm = ScriptedLLM([done])
    s = Agent(llm, gh, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert s["other_jobs"] == [] and "ALSO FAILED" not in llm.seen[0][1].content


def test_a_fix_is_one_commit_and_the_branch_starts_at_it(tmp):
    """3.41: no CI run on the broken commit, none on half-fixed states."""
    two_files = [call("read_repo_file", path="calc.py"),
                 call("read_repo_file", path="test_calc.py"),
                 call("submit_diagnosis", category="test_failure", root_cause="x",
                      action="fix_pr", explanation="y",
                      file_edits=[{"path": "calc.py", "content": FIXED},
                                  {"path": "test_calc.py", "content": "from calc import add\n# ok\n"}])]
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)], ci=passes)
    s = Agent(ScriptedLLM(two_files), gh, dry_run=False, verbose=False).run(
        repo="o/r", run_id=100, job_id=1)
    assert s["outcome"] == "healed"
    assert gh.commit_messages == ["self-heal: fix calc.py, test_calc.py"]
    assert list(gh.checks) == [s["fix_head_sha"]]       # CI ran once, on the whole fix
    tree = gh.trees[gh.branches[s["fix_branch"]]]
    assert tree["calc.py"] == FIXED and tree["test_calc.py"].endswith("# ok\n")
    # a second attempt is a second commit ON TOP of the first, branch updated
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)], ci=passes)
    llm = ScriptedLLM(fix_script(STILL_WRONG) + fix_script(FIXED))
    s = Agent(llm, gh, dry_run=False, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert s["outcome"] == "healed" and len(gh.commit_messages) == 2
    assert gh.commit_messages[1] == "self-heal: fix calc.py (attempt 2)"
    assert any(w.startswith("update self-heal/") for w in gh.writes)



def test_evidence_claims_are_checked_against_the_log(tmp):
    """3.44: 'the log shows this' must be true of the cited lines."""
    log = ["Collecting pytset==9.1.1",
           "ERROR: No matching distribution found for pytset==9.1.1",
           "Process completed with exit code 1."]
    req = {"path": "requirements.txt", "evidence": "log", "log_lines": "2",
           "replacements": [{"old": "pytset==9.1.1", "new": "pytest==9.1.1"}],
           "why": "misspelt package"}
    ev = guardrail.check_evidence(req, log)
    assert ev["evidence"] == "log" and ev["log_lines"] == "2" and not ev["note"]
    # off by one line is accepted (line 3 is next to line 2)
    assert guardrail.check_evidence(dict(req, log_lines="3"), log)["evidence"] == "log"
    # a cited line that says nothing about the file or the text -> downgraded
    far = log + ["ok"] * 10
    ev = guardrail.check_evidence(dict(req, log_lines="9"), far)
    assert ev["evidence"] == "code" and "do not mention requirements.txt" in ev["note"]
    # no usable line number -> downgraded; out-of-range numbers are ignored
    ev = guardrail.check_evidence(dict(req, log_lines="see above"), log)
    assert ev["evidence"] == "code" and "no valid log line" in ev["note"]
    assert guardrail.check_evidence(dict(req, log_lines="400"), log)["evidence"] == "code"
    # honest "code" stays "code" with no note; a missing field means "code"
    imp = {"path": "orders.py", "evidence": "code", "why": "unused import",
           "replacements": [{"old": "import json\n", "new": ""}]}
    assert guardrail.check_evidence(imp, log) == {
        "path": "orders.py", "evidence": "code", "log_lines": "",
        "why": "unused import", "note": ""}
    assert guardrail.check_evidence({"path": "a.py", "content": "x"}, log)["evidence"] == "code"
    # the file's name in the cited line is enough (a traceback names the file)
    tb = ["tests/test_orders.py:7: AssertionError"]
    t = {"path": "tests/test_orders.py", "evidence": "log", "log_lines": "1",
         "replacements": [{"old": "== 104", "new": "== 100"}]}
    assert guardrail.check_evidence(t, tb)["evidence"] == "log"
    assert guardrail.parse_line_spec("129-131, 284, 5 to 3", 300) == [3, 4, 5, 129, 130, 131, 284]


def test_pull_request_lists_changes_the_log_does_not_show(tmp):
    """Option (b) of 3.43: fix what it can see, but say what no log backs."""
    from agent.tools import log_to_lines
    raw = read(LOG_TEST) + "\ncalc.py:2: AssertionError: add(1, 2) returned -1\n"
    n = next(i for i, line in enumerate(log_to_lines(raw), 1) if "returned -1" in line)
    gh = FakeGitHub(dict(FILES, **{"util.py": "import os\nX = 1\n"}),
                    [{"id": 1, "name": "test", "log": raw}], ci=passes)
    llm = ScriptedLLM([
        call("read_repo_file", path="calc.py"),
        call("submit_diagnosis", category="test_failure", root_cause="add() subtracts",
             action="fix_pr", explanation="see below", file_edits=[
                 {"path": "calc.py", "evidence": "log", "log_lines": str(n),
                  "why": "add() subtracts", "replacements": [{"old": "a - b", "new": "a + b"}]},
                 {"path": "util.py", "evidence": "code", "why": "unused import os",
                  "replacements": [{"old": "import os\n", "new": ""}]},
                 {"path": "test_calc.py", "evidence": "log", "log_lines": "1",
                  "why": "claimed, not true", "replacements": [
                      {"old": "from calc import add\n", "new": "from calc import add  # x\n"}]}])])
    s = Agent(llm, gh, dry_run=False, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert s["outcome"] == "healed"
    body = gh.prs[0]["body"]
    backed, unbacked = body.split("**Changes the CI log does not show")
    assert "**Changes backed by the CI log**" in backed and f"`calc.py` (log lines {n})" in backed
    assert "`util.py` \u2014 unused import os" in unbacked
    assert "`test_calc.py`" in unbacked and "do not mention test_calc.py" in unbacked
    assert "util.py" not in backed and "test_calc.py" not in backed
    ev = {e["path"]: e["evidence"] for e in audit()[-1]["attempts"][0]["evidence"]}
    assert ev == {"calc.py": "log", "util.py": "code", "test_calc.py": "code"}
    # a fix with no log-backed change at all says so plainly
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)], ci=passes)
    s = Agent(ScriptedLLM(fix_script()), gh, dry_run=False, verbose=False).run(
        repo="o/r", run_id=100, job_id=1)
    assert "None of the changes in this commit is tied to a line" in gh.prs[0]["body"]



def test_a_red_pull_request_is_never_left_without_a_note(tmp):
    """3.47: attempt 1's fix fails CI, then the LLM runs out of quota."""
    from agent.run import error_comment

    def boom(messages):
        raise QUOTA_ERROR
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)], ci=passes)
    s = Agent(ScriptedLLM(fix_script(STILL_WRONG) + [boom]), gh, dry_run=False,
              verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert s["outcome"] == "error" and s["pr"]
    note = gh.comments[-1]
    assert "A human needs to take over" in note and "unavailable" in note
    text = error_comment(s, "o/r")
    assert "pull request #1" in text and "nothing was changed" not in text
    assert "nothing was changed" in error_comment(dict(s, pr=None), "o/r")
    # a later attempt that ESCALATES also leaves a note, with its diagnosis
    give_up = call("submit_diagnosis", category="other", root_cause="unclear now",
                   action="escalate", explanation="cannot tell")
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)], ci=passes)
    s = Agent(ScriptedLLM(fix_script(STILL_WRONG) + [give_up]), gh, dry_run=False,
              verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert s["outcome"] == "escalated"
    assert "decided a person must handle" in gh.comments[-1]
    assert "unclear now" in gh.comments[-1]



def test_action_guard_rules(tmp):
    """3.49: the loop guards now run inside the Action, for every user."""
    from agent.action_guard import should_run
    ok = {"SH_EVENT_NAME": "workflow_run", "SH_EVENT_CONCLUSION": "failure",
          "SH_EVENT_ATTEMPT": "1", "SH_EVENT_BRANCH": "case-13",
          "SH_EVENT_HEAD_REPO": "o/r", "SH_REPOSITORY": "o/r", "SH_EVENT_RUN_ID": "55"}
    assert should_run(ok) == (True, "run 55 on 'case-13' failed")
    for change, words in [({"SH_EVENT_NAME": "push"}, "not by a finished workflow run"),
                          ({"SH_EVENT_CONCLUSION": "success"}, "not 'failure'"),
                          ({"SH_EVENT_ATTEMPT": "2"}, "a re-run"),
                          ({"SH_EVENT_BRANCH": "self-heal/run-1-job-2"}, "own fix branches"),
                          ({"SH_EVENT_HEAD_REPO": "stranger/r"}, "a fork"),
                          ({"SH_EVENT_HEAD_REPO": ""}, "a fork"),
                          ({"SH_EVENT_RUN_ID": ""}, "no run id")]:
        run, reason = should_run(dict(ok, **change))
        assert not run and words in reason, (change, reason)
    # the Action passes these names; a renamed variable would silently skip everything
    action = (ROOT / "action.yml").read_text(encoding="utf-8")
    for name in ok:
        assert f"{name}:" in action, name


def test_run_summary_and_outputs_for_github_actions(tmp):
    """The run page shows what happened; later steps get outcome / PR / model."""
    import os
    from agent.run import job_summary, write_github_outputs
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)], ci=passes)
    s = Agent(ScriptedLLM(fix_script(), model="gemini-test"), gh, dry_run=False,
              verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert s["model"] == "gemini-test"
    text = job_summary(s, "o/r", live=True)
    assert "Self-heal:" in text and "healed" in text and "gemini-test" in text
    assert "actions/runs/100" in text and "https://example.test/pr/1" in text
    assert "`calc.py`: found in the code only" in text and "```diff" in text
    assert "dry run" in job_summary(s, "o/r", live=False)
    summary, out = Path(tmp) / "summary.md", Path(tmp) / "out.txt"
    saved = {k: os.environ.get(k) for k in ("GITHUB_STEP_SUMMARY", "GITHUB_OUTPUT")}
    os.environ.update(GITHUB_STEP_SUMMARY=str(summary), GITHUB_OUTPUT=str(out))
    try:
        write_github_outputs(s, "o/r", True)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert summary.read_text(encoding="utf-8") == text
    assert out.read_text(encoding="utf-8").splitlines() == [
        "outcome=healed", "pull-request=https://example.test/pr/1", "model=gemini-test"]


def test_dashboard_groups_tries_and_never_runs_record_text(tmp):
    """3.49: one incident per failed run; text from logs/LLM is only data."""
    import io
    import re
    import zipfile
    from agent import dashboard
    evil = '</script><img src=x onerror="alert(1)"> & <!-- x'

    def rec(time, outcome, run_id=7, dry=False, calls=2, model="m1", **kw):
        return dict({"time": time, "repo": "o/r", "run_id": run_id, "job_name": "lint",
                     "outcome": outcome, "dry_run": dry, "llm_calls": calls, "model": model,
                     "seconds": 10.0, "llm_tokens": {"input": 100, "output": 10},
                     "attempts": [{"proposal": {"root_cause": evil, "category": "other"}}]}, **kw)
    records = [rec("2026-11-23T01:00:00+0530", "error", calls=1, llm_error="overloaded"),
               rec("2026-11-23T01:05:00+0530", "healed", model="m2"),
               rec("2026-11-23T02:00:00+0530", "escalated", run_id=8),
               rec("2026-11-23T03:00:00+0530", "dry_run", run_id=9, dry=True)]
    items = dashboard.incidents(dashboard.dedupe(records + records[:1]))   # a duplicate
    assert len(items) == 3 and items[-1]["final"]["outcome"] == "healed"
    first = items[-1]
    assert len(first["earlier"]) == 1 and first["llm_calls"] == 3 and first["seconds"] == 20.0
    sm = dashboard.summarise(items)
    assert sm["incidents"] == 2 and sm["healed"] == 1 and sm["dry_runs"] == 1
    assert sm["by_outcome"] == {"healed": 1, "escalated": 1} and sm["median_heal_seconds"] == 20.0
    html = dashboard.build_html(items)
    assert "</script><img" not in html and "<!-- x" not in html and 'onerror="' not in html
    data = re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S).group(1)
    assert json.loads(data)["incidents"][-1]["final"]["attempts"][0]["proposal"]["root_cause"] == evil
    assert "innerHTML" not in dashboard.PAGE            # text is only ever inserted as text

    # records from the Action's artifacts, downloaded once and then cached
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("agent_runs.jsonl", json.dumps(rec("2026-11-23T04:00:00+0530", "healed", run_id=10)) + "\n")

    class GH:
        downloads = 0
        def list_artifacts(self, prefix):
            assert prefix == "self-heal-record-"
            return [{"id": 31, "name": "self-heal-record-10-1", "expired": False},
                    {"id": 30, "name": "self-heal-record-6-1", "expired": True}]
        def download_artifact(self, artifact_id):
            GH.downloads += 1
            return buf.getvalue()
    cache = Path(tmp) / "cache"
    got = dashboard.fetch_artifacts(GH(), cache, say=lambda m: 0)
    assert [r["run_id"] for r in got] == [10] and got[0]["_source"] == "self-heal-record-10-1"
    dashboard.fetch_artifacts(GH(), cache, say=lambda m: 0)
    assert GH.downloads == 1                             # second build: from the cache
    out = Path(tmp) / "d.html"
    assert dashboard.main(["--file", str(Path(tmp) / "none.jsonl"), "--out", str(out)]) == 0
    empty = re.search(r'id="data">(.*?)</script>', out.read_text(encoding="utf-8"), re.S).group(1)
    assert json.loads(empty)["incidents"] == [] and json.loads(empty)["summary"]["incidents"] == 0


def test_github_client_commits_several_files_at_once(tmp):
    """The real client's calls, checked against a recorded session (no network)."""
    from agent.github_client import GitHubClient

    class Resp:
        def __init__(self, data):
            self.status_code, self._data, self.headers = 200, data, {}
            self.content = b"x"

        def json(self):
            return self._data

    class Session:
        headers = {}

        def __init__(self):
            self.calls = []

        def request(self, method, url, params=None, json=None, timeout=None):
            self.calls.append((method, url.split("/repos/o/r")[1], json))
            if url.endswith("/git/commits/base1"):
                return Resp({"tree": {"sha": "tree0"}})
            if "/git/trees/tree0" in url:
                return Resp({"tree": [{"path": "run.sh", "type": "blob", "mode": "100755"},
                                      {"path": "a.py", "type": "blob", "mode": "100644"}]})
            if url.endswith("/git/trees"):
                return Resp({"sha": "tree1"})
            if url.endswith("/git/commits"):
                return Resp({"sha": "commit1"})
            return Resp({})

    sess = Session()
    gh = GitHubClient("o/r", token="t", read_only=False, session=sess)
    sha = gh.commit_files("self-heal/x", "base1", [("a.py", "A\n"), ("run.sh", "B\n")],
                          "self-heal: fix a.py, run.sh", create_branch=True)
    assert sha == "commit1"
    methods = [(m, u) for m, u, _ in sess.calls]
    assert methods == [("GET", "/git/commits/base1"), ("GET", "/git/trees/tree0"),
                       ("POST", "/git/trees"), ("POST", "/git/commits"), ("POST", "/git/refs")]
    tree_body = sess.calls[2][2]
    assert tree_body["base_tree"] == "tree0"
    assert {e["path"]: e["mode"] for e in tree_body["tree"]} == {"a.py": "100644",
                                                                 "run.sh": "100755"}
    assert sess.calls[3][2]["parents"] == ["base1"]
    assert sess.calls[4][2] == {"ref": "refs/heads/self-heal/x", "sha": "commit1"}
    # an existing branch is moved, never forced
    sess.calls.clear()
    gh.commit_files("self-heal/x", "base1", [("a.py", "C\n")], "m", create_branch=False)
    assert sess.calls[-1] == ("PATCH", "/git/refs/heads/self-heal/x",
                              {"sha": "commit1", "force": False})
    # and a read-only client refuses
    from agent.github_client import ReadOnlyError
    try:
        GitHubClient("o/r", token="t", session=Session()).commit_files(
            "b", "base1", [("a.py", "x")], "m", create_branch=True)
        assert False, "should have refused"
    except ReadOnlyError:
        pass


def test_model_queue_order_tiers_and_keys(tmp):
    """3.52: the queue is read from SH_MODEL_QUEUE; only models with a key join."""
    from agent import models as mq
    specs = mq.parse_queue("gemini:gemini-4-hydrogen, openrouter:nvidia/x-ultra:free, "
                           "groq:llama-3.3-70b-versatile@diagnose")
    assert [s.id for s in specs] == ["gemini:gemini-4-hydrogen", "openrouter:nvidia/x-ultra:free",
                                     "groq:llama-3.3-70b-versatile"]
    assert [s.tier for s in specs] == ["fix", "fix", "diagnose"]
    for bad in ("grok:x", "groq:", "groq:x@expert"):
        try:
            mq.parse_queue(bad)
            raise AssertionError(f"{bad!r} should be refused")
        except ValueError:
            pass
    # the default queue: best first, Gemini slots from the old settings
    default = [s.id for s in mq.configured_queue(main="gemini-4-hydrogen", fallback="gemini-3.5-flash")]
    assert default[0] == "gemini:gemini-4-hydrogen" and default[3] == "gemini:gemini-3.5-flash"
    assert default[1] == "groq:openai/gpt-oss-120b" and len(default) == 6
    assert "gemini:gemini-3.5-flash" not in [s.id for s in mq.configured_queue(fallback="")]
    # only providers with a key are used
    said = []
    q = mq.build_queue(env={"GROQ_API_KEY": "k"}, say=said.append, state_path=None,
                       factory=lambda spec: None)
    assert all(i.startswith("groq:") for i in q.ids) and "no key for" in said[0]
    try:
        mq.build_queue(env={}, say=said.append, state_path=None)
        raise AssertionError("an empty queue must be refused")
    except RuntimeError as e:
        assert "GROQ_API_KEY" in str(e)


def test_model_queue_sorts_errors(tmp):
    """Busy, per-minute and per-day limits, bad keys and unknown models differ."""
    from agent.models import classify, retry_after

    class APIError(Exception):
        def __init__(self, status, text):
            super().__init__(text)
            self.status_code = status
    cases = [
        (QUOTA_ERROR, "quota"),                                         # Gemini, per day
        (RuntimeError("503 UNAVAILABLE: the model is experiencing high demand"), "busy"),
        (APIError(429, "Rate limit reached for model `openai/gpt-oss-120b` on requests per "
                       "minute (RPM): Limit 30. Please try again in 1.5s."), "rate"),
        (APIError(429, "Rate limit reached for model `llama-3.3-70b-versatile` on tokens per "
                       "day (TPD): Limit 100000. Please try again in 7m12.5s."), "quota"),
        (APIError(429, "Rate limit exceeded: free-models-per-day. Add 10 credits"), "quota"),
        (APIError(402, "Insufficient credits"), "quota"),
        (APIError(401, "Invalid API Key"), "auth"),
        (APIError(404, "No endpoints found for nvidia/x:free"), "missing"),
        (APIError(400, "tool schema not supported"), "rejected"),
        (TimeoutError("Request timed out."), "busy"),
    ]
    for error, kind in cases:
        assert classify(error)[0] == kind, (str(error)[:60], classify(error))
    assert retry_after("Please retry in 14h40m7.36s") == 14 * 3600 + 40 * 60 + 7.36
    assert retry_after("Please try again in 7m12.5s") == 432.5
    assert retry_after("{'retryDelay': '41s'}") == 41
    assert classify(cases[2][0])[1] == 1.5


class FakeChat:
    """A stand-in provider model: answers, or raises, step by step."""
    def __init__(self, name, steps):
        self.name, self.steps, self.seen, self.bound = name, list(steps), [], []

    def bind_tools(self, tools, tool_choice=None):
        self.bound.append(tool_choice)
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        step = self.steps.pop(0) if self.steps else AIMessage(content="I am done.")
        if isinstance(step, Exception):
            raise step
        return step(messages) if callable(step) else step


def queue_of(chats, clock=None, **kw):
    from agent import models as mq
    specs = mq.parse_queue(", ".join(c.name for c in chats))
    by_id = {s.id: c for s, c in zip(specs, chats)}
    t = {"now": 1_800_000_000.0}
    kw.setdefault("say", lambda m: None)
    q = mq.ModelQueue(specs, factory=lambda spec: by_id[spec.id], clock=clock or (lambda: t["now"]),
                      sleep=lambda secs: t.__setitem__("now", t["now"] + secs), **kw)
    return q, t


def test_queue_switches_model_mid_investigation(tmp):
    """Model A answers step 1, then is busy: B continues from step 2 with A's work."""
    a = FakeChat("gemini:a", [call("read_repo_file", path="calc.py"),
                              RuntimeError("503 UNAVAILABLE high demand")])
    b = FakeChat("groq:b", fix_script()[1:])           # carries on: straight to the fix
    q, _ = queue_of([a, b], state_path=str(Path(tmp) / "q.json"))
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)], ci=passes)
    s = Agent(q, gh, dry_run=False, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert s["outcome"] == "healed" and s["model"] == "groq:b"
    assert s["step_models"][:2] == ["gemini:a", "groq:b"]
    # B was given A's steps as plain text, in one message after the system prompt
    first_b = b.seen[0]
    assert len(first_b) == 2 and "EARLIER STEPS" in first_b[1].content
    assert "[called] read_repo_file" in first_b[1].content and "[result of read_repo_file]" in first_b[1].content
    rec = audit()[-1]
    assert rec["models_used"] == ["gemini:a", "groq:b"] and rec["model"] == "groq:b"
    assert rec["queue_events"][0]["model"] == "gemini:a" and rec["queue_events"][0]["kind"] == "busy"
    # the busy model rests (and the rest is remembered in the state file)
    saved = json.loads((Path(tmp) / "q.json").read_text(encoding="utf-8"))
    assert saved["gemini:a"]["kind"] == "busy"


def test_queue_remembers_a_used_up_quota_until_the_reset(tmp):
    from agent import models as mq
    state = str(Path(tmp) / "q.json")
    a = FakeChat("gemini:a", [QUOTA_ERROR])
    b = FakeChat("groq:b", fix_script())
    q, t = queue_of([a, b], state_path=state)
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)], ci=passes)
    assert Agent(q, gh, dry_run=False, verbose=False).run(repo="o/r", run_id=100, job_id=1)["outcome"] == "healed"
    # a NEW run (new queue, same file) does not even try A: it is out of quota
    a2, b2 = FakeChat("gemini:a", fix_script()), FakeChat("groq:b", fix_script())
    q2, _ = queue_of([a2, b2], clock=lambda: t["now"] + 60, state_path=state)
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)], ci=passes)
    Agent(q2, gh, dry_run=False, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert a2.seen == [] and b2.seen and "quota, back in 14h" in q2.describe()
    # ... and uses it again once the quota has reset
    a3 = FakeChat("gemini:a", fix_script())
    q3, _ = queue_of([a3, FakeChat("groq:b", [])], clock=lambda: t["now"] + 15 * 3600,
                     state_path=state)
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)], ci=passes)
    Agent(q3, gh, dry_run=False, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert a3.seen
    # no retry time given -> the provider's next daily reset
    assert mq.next_reset("groq", 1_800_000_000.0) % 86400 == 0


def test_queue_waits_briefly_then_gives_up_cleanly(tmp):
    """All busy: wait for the first to return if soon; otherwise hand over."""
    a = FakeChat("gemini:a", [RuntimeError("503 UNAVAILABLE"), call("read_repo_file", path="calc.py")]
                 + fix_script()[1:])
    q, t = queue_of([a])
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)], ci=passes)
    s = Agent(q, gh, dry_run=False, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert s["outcome"] == "healed" and q.waited == 120          # rested 2 min, then answered
    # every model out of quota: no waiting, a clean error naming each model
    a = FakeChat("gemini:a", [QUOTA_ERROR])
    b = FakeChat("openrouter:b:free", [RuntimeError("429 Rate limit exceeded: free-models-per-day")])
    q, _ = queue_of([a, b])
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)], ci=passes)
    s = Agent(q, gh, dry_run=False, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert s["outcome"] == "error" and q.waited == 0
    assert "out of quota or unusable" in s["llm_error"] and "openrouter:b:free" in s["llm_error"]
    assert "overloaded" not in s["llm_error"]           # so run.py does not retry pointlessly
    assert gh.prs == []


def test_diagnose_only_model_may_not_open_a_pull_request(tmp):
    weak = FakeChat("groq:small@diagnose", fix_script())
    q, _ = queue_of([weak])
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)], ci=passes)
    s = Agent(q, gh, dry_run=False, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert s["outcome"] == "escalated" and gh.prs == []
    assert "diagnose-only" in " ".join(s["decision"]["reasons"])


def test_queue_works_with_the_real_openai_compatible_client(tmp):
    """Offline: the request ChatOpenAI would send to Groq / OpenRouter is valid,
    for our tools, the forced last step, and a replayed history."""
    from langchain_openai import ChatOpenAI
    from agent import models as mq
    from agent.tools import Context, make_tools
    tools = make_tools(Context())
    chat = ChatOpenAI(model="openai/gpt-oss-120b", base_url=mq.PROVIDERS["groq"]["base_url"],
                      api_key="test", max_retries=0)
    bound = chat.bind_tools(tools)
    sent = json.dumps(bound.kwargs["tools"])
    assert "$ref" not in sent and "submit_diagnosis" in sent and "file_edits" in sent
    last = chat.bind_tools([t for t in tools if t.name == "submit_diagnosis"],
                           tool_choice="submit_diagnosis")
    assert last.kwargs["tool_choice"]["function"]["name"] == "submit_diagnosis"
    # a history started by Gemini, replayed for Groq, builds a valid request
    a_msg = AIMessage(content="", tool_calls=[{"name": "read_repo_file", "args": {"path": "calc.py"},
                                               "id": "g-1"}], response_metadata={"sh_model": "gemini:a"})
    msgs = [SystemMessage("sys"), HumanMessage("A job failed."), a_msg,
            ToolMessage("def add(a, b): ...", name="read_repo_file", tool_call_id="g-1")]
    payload = chat._get_request_payload(mq.portable(msgs, "groq:openai/gpt-oss-120b"))
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    # unchanged when the same model continues
    assert mq.portable(msgs, "gemini:a") is msgs

def test_guardrail_blocks_edit_of_unread_file(tmp):
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)])
    llm = ScriptedLLM(fix_script()[1:])          # submits without reading calc.py
    s = Agent(llm, gh, dry_run=False, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert s["outcome"] == "escalated" and changes(gh) == []
    assert "not read in full" in s["decision"]["reasons"][0]


def test_action_must_match_category(tmp):
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)])
    llm = ScriptedLLM([call("submit_diagnosis", category="other_bot_infra",
                            root_cause="x", action="rerun", explanation="y")])
    s = Agent(llm, gh, dry_run=False, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert s["outcome"] == "escalated" and changes(gh) == []


def test_upstream_job_is_followed(tmp):
    gh = FakeGitHub(FILES, [
        job(2, LOG_DERIV, name="all-green", completed_at="2026-01-01T10:05:00Z"),
        job(1, LOG_FLAKY, name="build", completed_at="2026-01-01T10:01:00Z")])
    llm = ScriptedLLM([])
    s = Agent(llm, gh, dry_run=False, verbose=False).run(repo="o/r", run_id=100, job_id=2)
    assert s["upstream_hops"] == 1 and s["job_id"] == 1
    assert s["first_triage"]["label"] == "needs_traceback"
    assert s["outcome"] == "healed" and gh.writes == ["rerun"]


def test_upstream_missing_is_escalated(tmp):
    gh = FakeGitHub(FILES, [job(2, LOG_DERIV)])
    s = Agent(ScriptedLLM([]), gh, dry_run=False, verbose=False).run(
        repo="o/r", run_id=100, job_id=2)
    assert s["outcome"] == "escalated" and changes(gh) == []


def test_failed_rerun_leads_to_investigation(tmp):
    gh = FakeGitHub(FILES, [job(1, LOG_FLAKY)], rerun_passes=False)
    llm = ScriptedLLM([call("search_log", pattern="error"),
                       call("submit_diagnosis", category="flaky_infra",
                            root_cause="still timing out", action="rerun",
                            explanation="try again")])
    s = Agent(llm, gh, dry_run=False, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert changes(gh) == ["rerun"]              # only ONE rerun, ever
    assert llm.calls == 2 and s["outcome"] == "escalated"
    assert "already re-run once" in s["decision"]["reasons"][0]


def test_step_limit(tmp):
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)])
    llm = ScriptedLLM([lambda m: call("search_log", pattern="x")] * 100)
    s = Agent(llm, gh, dry_run=False, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert llm.calls == config.MAX_LLM_STEPS and s["outcome"] == "escalated"
    assert "Step limit" in s["proposal"]["explanation"]


def test_last_step_must_submit_a_diagnosis(tmp):
    """3.50: on case-14 the fallback model read files until the step limit and
    handed over with nothing. The last call now sees only submit_diagnosis."""
    bound = []

    class Recording(ScriptedLLM):
        def bind_tools(self, tools, tool_choice=None):
            bound.append(([t.name for t in tools], tool_choice))
            return self
    last = config.MAX_LLM_STEPS - 1
    steps = [lambda m: call("read_repo_file", path="calc.py")] * last + [
        lambda m: call("submit_diagnosis", category="test_failure", root_cause="add() subtracts",
                       action="fix_pr", explanation="seen in calc.py",
                       file_edits=[{"path": "calc.py", "evidence": "code",
                                    "replacements": [{"old": "a - b", "new": "a + b"}]}])]
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)], ci=passes)
    llm = Recording(steps)
    s = Agent(llm, gh, dry_run=False, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert s["outcome"] == "healed" and llm.calls == config.MAX_LLM_STEPS
    assert ["submit_diagnosis"] == bound[-1][0] and bound[-1][1] == "submit_diagnosis"
    assert len(bound[0][0]) > 1                       # normal steps keep every tool
    assert "LAST step" in llm.seen[-1][-1].content    # only the last call is told
    assert all("LAST step" not in getattr(m[-1], "content", "") for m in llm.seen[:-1])


def test_model_that_never_answers_is_nudged_then_escalated(tmp):
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)])
    llm = ScriptedLLM([AIMessage(content="I think it is a test."),
                       AIMessage(content="Still thinking.")])
    s = Agent(llm, gh, dry_run=False, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert llm.calls == 2 and s["outcome"] == "escalated"
    assert "submit_diagnosis" in llm.seen[1][-1].content      # the reminder


def test_llm_error_ends_cleanly(tmp):
    def boom(messages):
        raise RuntimeError("quota exceeded")
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)])
    s = Agent(ScriptedLLM([boom]), gh, dry_run=False, verbose=False).run(
        repo="o/r", run_id=100, job_id=1)
    # "error", not "escalated": the LLM never decided anything (changed 15 Nov)
    assert s["outcome"] == "error" and "quota exceeded" in s["proposal"]["explanation"]
    assert len(audit()) == 1 and gh.writes == []


def test_log_file_mode_can_only_diagnose(tmp):
    llm = ScriptedLLM([call("list_repo_files"),
                       call("submit_diagnosis", category="test_failure", root_cause="x",
                            action="fix_pr", explanation="y",
                            file_edits=[{"path": "a.py", "content": "x = 1\n"}])])
    s = Agent(llm, None, verbose=False).run(raw_log=read(LOG_TEST))
    assert s["outcome"] == "escalated"
    assert "no repository connected" in s["decision"]["reasons"][0]
    assert "no repository connected" in llm.seen[1][-1].content   # tool said so


def test_low_confidence_is_passed_on_as_a_weak_hint(tmp):
    llm = ScriptedLLM([call("submit_diagnosis", category="config", root_cause="x",
                            action="escalate", explanation="y")])
    s = Agent(llm, None, verbose=False).run(raw_log=read(LOG_UNKNOWN))
    assert s["label"] == "unknown" and "NOT confident" in llm.seen[0][1].content


# A file far above the 30,000-character read limit, with one bug in the middle.
_BLOCKS = [f"def f{i}():\n    return {i}\n\n\n" for i in range(2500)]
BIG_BROKEN = ("".join(_BLOCKS[:1250]) + "def add(a, b):\n    return a - b\n\n\n"
              + "".join(_BLOCKS[1250:]))
BIG_FIXED = BIG_BROKEN.replace("return a - b", "return a + b")
BIG_FILES = {"big.py": BIG_BROKEN, ".github/workflows/ci.yml": "name: CI\n"}
BIG_EDIT = [{"path": "big.py",
             "replacements": [{"old": "    return a - b\n", "new": "    return a + b\n"}]}]


def big_passes(tree):
    return "return a + b" in tree["big.py"]


def big_submit(edits):
    return call("submit_diagnosis", category="test_failure",
                root_cause="add() subtracts instead of adding", action="fix_pr",
                explanation="Changed - to + in add().", file_edits=edits)


def test_large_file_is_fixed_by_an_exact_replacement(tmp):
    """Found on the first real GitHub run (15 Nov): a one-line fix in a file
    over the read limit used to be impossible."""
    assert len(BIG_BROKEN) > 2 * config.MAX_FILE_READ_CHARS
    gh = FakeGitHub(BIG_FILES, [job(1, LOG_TEST)], ci=big_passes)
    llm = ScriptedLLM([call("read_repo_file", path="big.py"),
                       call("read_repo_file", path="big.py", start_line=4995, end_line=5010),
                       big_submit(BIG_EDIT)])
    s = Agent(llm, gh, dry_run=False, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    whole, part = llm.seen[1][-1].content, llm.seen[2][-1].content
    assert whole.startswith("[big.py: 10004 lines") and "replacements" in whole
    assert len(whole) <= config.MAX_FILE_READ_CHARS + 400     # still capped
    assert part.startswith("[big.py lines 4995-5010 of 10004]\n")
    assert "def add(a, b):\n    return a - b\n" in part      # exact text, no line numbers
    assert s["outcome"] == "healed"
    assert gh._tree("self-heal/run-100-job-1")["big.py"] == BIG_FIXED   # nothing else changed
    assert s["action_result"]["files"][0]["kind"] == "replacements"


def test_misquoted_replacement_can_be_corrected(tmp):
    gh = FakeGitHub(BIG_FILES, [job(1, LOG_TEST)], ci=big_passes)
    wrong = [{"path": "big.py", "replacements": [{"old": "return a-b", "new": "return a+b"}]}]
    llm = ScriptedLLM([big_submit(wrong), big_submit(BIG_EDIT)])
    s = Agent(llm, gh, dry_run=False, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    answer = llm.seen[1][-1].content
    assert answer.startswith("ERROR") and "was not found" in answer
    assert s["outcome"] == "healed" and llm.calls == 2


def test_dry_run_shows_the_change_it_would_make(tmp):
    gh = FakeGitHub(BIG_FILES, [job(1, LOG_TEST)], read_only=True)
    s = Agent(ScriptedLLM([big_submit(BIG_EDIT)]), gh,
              verbose=False).run(repo="o/r", run_id=100, job_id=1)
    diff = s["action_result"]["diff"]
    assert s["outcome"] == "dry_run" and gh.writes == []
    assert "-    return a - b" in diff and "+    return a + b" in diff
    assert len(diff) < 600                              # the change, not the file
    assert "return a + b" in json.dumps(audit()[-1])    # and it is in the audit log


QUOTA_ERROR = RuntimeError("""Error calling model 'gemini-4-hydrogen' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://aistudio.google.com/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-4-hydrogen, description: Generate content requests per day limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.\\nPlease retry in 14h40m7.361636948s.', 'status': 'RESOURCE_EXHAUSTED'}}""")


def test_llm_failure_is_an_error_not_a_diagnosis(tmp):
    """Seen on 15 Nov: the free tier's 20 requests a day ran out mid-run."""
    from agent.graph import _llm_error_text
    text = _llm_error_text(QUOTA_ERROR)
    assert text == ("Gemini quota used up for gemini-4-hydrogen: the free tier limit is 20 "
                    "requests per day. Google says to retry in about 14h40m. Until then, "
                    "use another model (SH_GEMINI_MODEL) or a key with billing enabled.")
    assert len(_llm_error_text(RuntimeError("x" * 5000))) < 310      # anything else: cut
    overloaded = RuntimeError("503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This "
                              "model is currently experiencing high demand.'}}")
    assert _llm_error_text(overloaded).startswith("Gemini is overloaded (503")
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)], read_only=True)
    def out_of_quota(messages):
        raise QUOTA_ERROR

    llm = ScriptedLLM([call("read_repo_file", path="calc.py"), out_of_quota])
    s = Agent(llm, gh, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    rec = audit()[-1]
    assert s["outcome"] == "error" and rec["outcome"] == "error"
    assert rec["llm_error"] == text and rec["llm_calls"] == 1
    # the record shows what the LLM looked at before the failure
    assert rec["attempts"][0]["tool_calls"] == [["read_repo_file(path=calc.py)"]]
    assert gh.writes == [] and s["decision"]["action"] == "escalate"
    # a run without a failure has no error and counts tokens when the service reports them
    ok = Agent(ScriptedLLM([call("submit_diagnosis", category="test_failure", root_cause="x",
                                 action="escalate", explanation="y")]),
               None, verbose=False).run(raw_log=read(LOG_TEST))
    assert ok["outcome"] == "escalated" and audit()[-1]["llm_error"] is None
    assert audit()[-1]["memory_hits"] == 0           # nothing remembered in this test
    assert audit()[-1]["llm_tokens"] == {"input": 0, "output": 0}


def test_pull_request_title_is_cut_between_words(tmp):
    from agent.graph import _short
    real = ("`test_subtest_with_mocked_clock` calls `Path.read_text()` without an "
            "explicit `encoding` argument, causing an `EncodingWarning`.")
    cut = _short(real)
    assert cut == "`test_subtest_with_mocked_clock` calls `Path.read_text()` without..."
    assert len(cut) <= 70 and _short("short  one\nline") == "short one line"
    assert len(_short("x" * 200)) == 70          # one long word is still cut


def test_replacement_rules(tmp):
    apply = guardrail.apply_replacements
    # must be unique
    assert "occurs 2501 times" in apply(BIG_BROKEN, [{"old": "    return", "new": "x"}])[1][0]
    assert "was not found" in apply("a = 1\n", [{"old": "b = 2", "new": "b = 3"}])[1][0]
    assert "identical" in apply("a = 1\n", [{"old": "a = 1", "new": "a = 1"}])[1][0]
    assert "empty" in apply("a = 1\n", [{"old": "  ", "new": "x"}])[1][0]
    assert apply("", [{"old": "a", "new": "b"}])[1]             # unreadable file
    assert apply("caf\ufffd = 1\n", [{"old": "1", "new": "2"}])[1]   # not valid UTF-8
    # two replacements, applied in order
    assert apply("a = 1\nb = 2\n", [{"old": "a = 1", "new": "a = 5"},
                                     {"old": "b = 2", "new": "b = 6"}]) == ("a = 5\nb = 6\n", [])
    # a file with Windows line endings keeps them
    assert apply("a = 1\r\nb = 2\r\n", [{"old": "a = 1\nb = 2\n", "new": "a = 1\nb = 3\n"}]) \
        == ("a = 1\r\nb = 3\r\n", [])

    def check(edits, **kw):
        p = {"category": "test_failure", "action": "fix_pr", "root_cause": "r",
             "explanation": "e", "file_edits": edits}
        return guardrail.check(p, attempt=0, reruns_done=0, online=True, **kw)
    now = {"big.py": BIG_BROKEN, ".github/workflows/ci.yml": "name: CI\n"}
    ok = check(BIG_EDIT, current_files=now)
    assert ok["ok"] and ok["edits"][0]["content"] == BIG_FIXED
    # still blocked: protected path, both kinds at once, a file that is not there,
    # too many replacements, and replacing a large file whole without reading it
    assert not check([{"path": ".github/workflows/ci.yml",
                       "replacements": [{"old": "CI", "new": "X"}]}], current_files=now)["ok"]
    assert not check([dict(BIG_EDIT[0], content="x = 1\n")], current_files=now)["ok"]
    assert not check([{"path": "new.py", "replacements": [{"old": "a", "new": "b"}]}],
                     current_files=now, existing_files={"new.py": False})["ok"]
    many = [{"old": f"def f{i}():", "new": f"def g{i}():"} for i in range(11)]
    assert not check([{"path": "big.py", "replacements": many}], current_files=now)["ok"]
    assert not check([{"path": "big.py", "content": BIG_FIXED}], current_files=now)["ok"]


def test_audit_log_records_the_model(tmp):
    """Two runs on one log are only comparable if the record names the model."""
    done = call("submit_diagnosis", category="test_failure", root_cause="x",
                action="escalate", explanation="y")
    Agent(ScriptedLLM([done], model="models/gemini-x"), None,
          verbose=False).run(raw_log=read(LOG_TEST))
    Agent(ScriptedLLM([done]), None, verbose=False).run(raw_log=read(LOG_TEST))
    assert [r["model"] for r in audit()] == ["gemini-x", None]
    assert all(0 <= r["llm_seconds"] <= r["seconds"] for r in audit())


def test_make_llm_passes_the_thinking_level_only_when_set(tmp):
    import os
    from agent import llm as llm_mod
    try:
        import langchain_google_genai  # noqa: F401
    except ImportError:
        return                          # package not installed: nothing to check
    saved = (os.environ.get("GOOGLE_API_KEY"), config.GEMINI_THINKING_LEVEL)
    os.environ["GOOGLE_API_KEY"] = "not-a-real-key"     # no call is made
    try:
        config.GEMINI_THINKING_LEVEL = ""
        plain = llm_mod.make_llm("gemini-x")
        assert plain.thinking_config is None and llm_mod.model_name(plain) == "gemini-x"
        config.GEMINI_THINKING_LEVEL = "low"
        low = llm_mod.make_llm("gemini-x")
        assert dict(low.thinking_config) == {"thinking_level": "low"}
        assert llm_mod.thinking_level(low) == "low" and llm_mod.thinking_level(plain) is None
        # and the audit record carries it
        stand_in = ScriptedLLM([call("submit_diagnosis", category="test_failure",
                                     root_cause="x", action="escalate", explanation="y")],
                               model="gemini-x")
        stand_in.thinking_config = {"thinking_level": "low"}
        Agent(stand_in, None, verbose=False).run(raw_log=read(LOG_TEST))
        assert audit()[-1]["thinking_level"] == "low"
    finally:
        config.GEMINI_THINKING_LEVEL = saved[1]
        if saved[0] is None:
            os.environ.pop("GOOGLE_API_KEY", None)
        else:
            os.environ["GOOGLE_API_KEY"] = saved[0]


def test_classifier_only_mode_needs_no_llm(tmp):
    gh = FakeGitHub(FILES, [job(1, LOG_TEST)])
    s = Agent(None, gh, dry_run=False, mode="classifier_only", verbose=False).run(
        repo="o/r", run_id=100, job_id=1)
    assert s["outcome"] == "escalated" and s["proposal"]["category"] == "test_failure"
    assert s["llm_calls"] == 0 and changes(gh) == []


def test_llm_only_mode_ignores_the_classifier(tmp):
    gh = FakeGitHub(FILES, [job(1, LOG_FLAKY)])
    llm = ScriptedLLM([call("submit_diagnosis", category="flaky_infra", root_cause="x",
                            action="rerun", explanation="y")])
    s = Agent(llm, gh, dry_run=False, mode="llm_only", verbose=False).run(
        repo="o/r", run_id=100, job_id=1)
    assert llm.calls == 1 and s["outcome"] == "healed"
    assert "No classifier hint" in llm.seen[0][1].content


def test_verified_fix_is_recalled_next_time(tmp):
    for _ in range(2):
        gh = FakeGitHub(FILES, [job(1, LOG_TEST)], ci=passes)
        llm = ScriptedLLM(fix_script())
        Agent(llm, gh, dry_run=False, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    assert "SIMILAR FAILURES THAT WERE FIXED" in llm.seen[0][1].content
    assert "add() subtracts instead of adding" in llm.seen[0][1].content
    assert [r["memory_hits"] for r in audit()] == [0, 1]     # the record says so
    # SH_MEMORY_TOP_K=0 switches recall off (used for the hard cases, PROJECT_LOG 3.28)
    saved = config.MEMORY_TOP_K
    config.MEMORY_TOP_K = 0
    try:
        gh = FakeGitHub(FILES, [job(1, LOG_TEST)], ci=passes)
        llm = ScriptedLLM(fix_script())
        Agent(llm, gh, dry_run=False, verbose=False).run(repo="o/r", run_id=100, job_id=1)
    finally:
        config.MEMORY_TOP_K = saved
    assert "SIMILAR FAILURES" not in llm.seen[0][1].content
    assert audit()[-1]["memory_hits"] == 0


def test_guardrail_rules(tmp):
    base = dict(attempt=0, reruns_done=0, online=True)

    def fix(path, content="x = 1\n", **kw):
        p = {"category": "lint_build", "action": "fix_pr", "root_cause": "r",
             "explanation": "e", "file_edits": [{"path": path, "content": content}]}
        return guardrail.check(p, **dict(base, **kw))

    assert fix("src/a.py")["ok"]
    assert fix("./src\\a.py")["edits"][0]["path"] == "src/a.py"
    for bad in ("../a.py", "/etc/passwd", "C:\\x.py", "", ".github/workflows/ci.yml",
                ".git/config", ".env", "deploy/server.pem", "a//b.py"):
        assert not fix(bad)["ok"], bad
    assert not fix("a.py", content="   ")["ok"]
    assert not fix("a.py", content="x" * (config.MAX_EDIT_BYTES + 1))["ok"]
    assert not fix("a.py", attempt=config.MAX_FIX_ATTEMPTS)["ok"]
    assert not fix("a.py", online=False)["ok"]
    assert not fix("a.py", existing_files={"a.py": True})["ok"]
    assert fix("a.py", existing_files={"a.py": True}, files_read_full={"a.py"})["ok"]
    many = {"category": "lint_build", "action": "fix_pr", "root_cause": "r", "explanation": "e",
            "file_edits": [{"path": f"f{i}.py", "content": "x\n"}
                           for i in range(config.MAX_EDIT_FILES + 1)]}
    assert not guardrail.check(many, **base)["ok"]
    rerun = {"category": "flaky_infra", "action": "rerun", "file_edits": []}
    assert guardrail.check(rerun, **base)["ok"]
    assert not guardrail.check(rerun, **dict(base, reruns_done=1))["ok"]
    esc = {"category": "other", "action": "escalate", "file_edits": []}
    assert guardrail.check(esc, attempt=99, reruns_done=9, online=False)["ok"]


def test_evaluation_scores_rules_classifier_llm_and_cascade(tmp):
    """agent.evaluate on four logs with known answers, with a stand-in LLM."""
    import csv
    from agent import evaluate as ev

    if not ev.LABELLED.exists():
        skip(f"needs {ev.LABELLED.name} (research data, not in the repository)")
    tmp = Path(tmp)
    sample = tmp / "sample.csv"
    rows = [("116750249015", "go", "flaky_infra"),         # classifier: flaky 0.97
            ("122700520097", "python", "test_failure"),    # classifier: test 0.93
            ("124132246976", "python", "needs_traceback"),  # caught by the rule
            ("121492986837", "java", "config"),            # classifier unsure (0.49)
            ("121613017731", "ruby", ""),                  # not labelled -> skipped
            ("121885784167", "csharp", "unclear")]         # unclear -> skipped
    with open(sample, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "log_id", "language", "repo", "run_url",
                    "cleaned_log", "human_label", "notes"])
        for i, (log_id, lang, label) in enumerate(rows, 1):
            w.writerow([i, log_id, lang, "", "", "", label, ""])
    saved = (ev.CACHE, ev.RESULTS, ev.PER_LOG, ev.EVAL_AUDIT, config.AUDIT_LOG)
    saved_model = config.GEMINI_MODEL
    ev.CACHE, ev.RESULTS, ev.PER_LOG = tmp / "c.jsonl", tmp / "r.csv", tmp / "p.csv"
    ev.EVAL_AUDIT = tmp / "eval_audit.jsonl"
    real_stdout, sys.stdout = sys.stdout, open(tmp / "printed.txt", "w", encoding="utf-8")

    def says_config(messages):          # this "LLM" calls everything config
        return call("submit_diagnosis", category="config", root_cause="r",
                    action="escalate", explanation="e")

    def quota(messages):
        raise RuntimeError("quota exceeded")
    try:
        # 1. an API failure records nothing
        ev.evaluate(sample, llm=ScriptedLLM([quota], model="model-a"), verbose=False)
        assert not ev.CACHE.exists()
        # 2. a full run
        llm = ScriptedLLM([says_config] * 10, model="model-a")
        res = {r["system"]: r for r in ev.evaluate(sample, llm=llm, verbose=False)}
        assert llm.calls == 3                       # the rule-caught log needs no LLM
        assert res["rules (where a rule fired)"]["n"] == 4       # every usable row has a rule label
        assert abs(res["classifier_only (t=0.6)"]["accuracy"] - 0.75) < 1e-9
        assert abs(res["llm_only"]["accuracy"] - 0.50) < 1e-9
        assert res["llm_only"]["llm_logs"] == 3
        best = res["cascade (t=0.6)"]               # classifier where sure, LLM for the rest
        assert best["accuracy"] == 1.0 and best["llm_logs"] == 1
        assert res["cascade (t=0.95)"]["llm_logs"] == 2
        assert abs(res["cascade (t=0.95)"]["accuracy"] - 0.75) < 1e-9
        # 3. a second run is free: everything comes from the cache
        again = ScriptedLLM([quota], model="model-a")
        ev.evaluate(sample, llm=again, verbose=False)
        assert again.calls == 0
        # 4. --no-llm still reports the cached LLM answers (of the configured model)
        config.GEMINI_MODEL = "model-a"
        res2 = ev.evaluate(sample, use_llm=False, verbose=False)
        assert any(r["system"] == "llm_only" for r in res2)
        assert ev.RESULTS.exists() and ev.PER_LOG.exists()
        assert "model-a" in ev.RESULTS.read_text(encoding="utf-8")
        assert config.AUDIT_LOG == saved[4]         # the real audit log was left alone
        # 5. another model does not reuse model-a's answers ...
        config.GEMINI_MODEL = "model-b"
        res3 = ev.evaluate(sample, use_llm=False, verbose=False)
        assert not any(r["system"] == "llm_only" for r in res3)
        other = ScriptedLLM([says_config] * 10, model="model-b")
        ev.evaluate(sample, llm=other, verbose=False)
        assert other.calls == 3
        # ... and model-a's answers are still there afterwards
        back = ScriptedLLM([quota], model="model-a")
        ev.evaluate(sample, llm=back, verbose=False)
        assert back.calls == 0
    finally:
        sys.stdout.close()
        sys.stdout = real_stdout
        ev.CACHE, ev.RESULTS, ev.PER_LOG, ev.EVAL_AUDIT = saved[:4]
        config.GEMINI_MODEL = saved_model


def test_real_github_client_against_canned_responses(tmp):
    """The real GitHubClient, with the network replaced by canned replies."""
    import base64
    from agent.github_client import GitHubClient, GitHubError

    class Resp:
        def __init__(self, status=200, data=None, content=None):
            self.status_code, self._data, self.headers = status, data, {}
            self.content = content if content is not None else json.dumps(data).encode()
            self.text = self.content.decode()

        def json(self):
            return self._data

    class Session:
        def __init__(self):
            self.headers, self.sent = {}, []

        def request(self, method, url, params=None, json=None, timeout=None):
            self.sent.append((method, url.split("/repos/o/r")[-1], params, json))
            path = self.sent[-1][1]
            if path == "/contents/calc.py":
                if method == "PUT":
                    return Resp(200, {"commit": {"sha": "newsha"}})
                return Resp(200, {"path": "calc.py", "sha": "blob1",
                                  "content": base64.b64encode(BROKEN.encode()).decode()})
            if path == "/contents/missing.py":
                return Resp(404, {"message": "Not Found"})
            if path == "/actions/runs/100/jobs":
                return Resp(200, {"jobs": [
                    {"id": 2, "conclusion": "failure", "completed_at": "2026-01-01T10:05:00Z"},
                    {"id": 3, "conclusion": "success", "completed_at": "2026-01-01T10:02:00Z"},
                    {"id": 1, "conclusion": "failure", "completed_at": "2026-01-01T10:01:00Z"}]})
            if path == "/actions/jobs/1/logs":
                return Resp(200, None, content="line one\nline two\n".encode())
            if path == "/actions/runs/100/rerun-failed-jobs":
                return Resp(201, {})
            if path == "/actions/artifacts":
                return Resp(200, {"artifacts": [{"id": 5, "name": "self-heal-record-1-1"},
                                                {"id": 6, "name": "other"}]})
            if path == "/actions/artifacts/5/zip":
                return Resp(200, None, content=b"PK-zip-bytes")
            return Resp(500, {"message": "unexpected " + path})

    session = Session()
    gh = GitHubClient("o/r", token="t", read_only=True, session=session)
    assert session.headers["Authorization"] == "Bearer t"
    assert gh.get_file("calc.py", ref="abc")["text"] == BROKEN
    assert session.sent[-1][2] == {"ref": "abc"}
    assert gh.get_file("missing.py") is None
    assert [j["id"] for j in gh.failed_jobs(100)] == [1, 2]     # earliest first
    assert gh.get_job_log(1) == "line one\nline two\n"
    assert [a["id"] for a in gh.list_artifacts("self-heal-record-")] == [5]
    assert gh.download_artifact(5) == b"PK-zip-bytes"
    for write in (lambda: gh.rerun_failed_jobs(100),
                  lambda: gh.create_branch("b", "sha"),
                  lambda: gh.put_file("calc.py", FIXED, "b", "m"),
                  lambda: gh.open_pull_request("t", "b", "main", "body"),
                  lambda: gh.comment(1, "hi")):
        try:
            write()
            raise AssertionError("a read-only client performed a write")
        except ReadOnlyError:
            pass
    assert not [s for s in session.sent if s[0] != "GET"]       # nothing was sent

    live = GitHubClient("o/r", token="t", read_only=False, session=session)
    live.rerun_failed_jobs(100)
    assert session.sent[-1][:2] == ("POST", "/actions/runs/100/rerun-failed-jobs")
    assert live.put_file("calc.py", FIXED, "fix", "msg") == "newsha"
    body = session.sent[-1][3]
    assert body["sha"] == "blob1" and body["branch"] == "fix"   # update, not create
    assert base64.b64decode(body["content"]).decode() == FIXED
    try:
        GitHubClient("not-a-repo")
        raise AssertionError("bad repo name accepted")
    except ValueError:
        pass
    assert isinstance(GitHubError(404, "x"), RuntimeError)


def main():
    missing =[p for p in (LOG_FLAKY, LOG_TEST, LOG_DERIV, LOG_UNKNOWN) if not p.exists()]
    if missing:
        print("Missing test logs, cannot run:", *missing, sep="\n  ")
        return 2
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    saved = (config.AUDIT_LOG, config.FIX_MEMORY)
    failed = skipped = 0
    for name, fn in tests:
        with tempfile.TemporaryDirectory() as tmp:
            fresh(tmp)
            try:
                fn(tmp)
                print(f"PASS  {name}")
            except Skipped as e:
                skipped += 1
                print(f"SKIP  {name}: {e}")
            except Exception as e:          # noqa: BLE001
                failed += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    config.AUDIT_LOG, config.FIX_MEMORY = saved
    ran = len(tests) - skipped
    print(f"\n{ran - failed} of {ran} passed" + (f", {skipped} skipped" if skipped else ""))
    return 1 if failed else 0


# pytest support: gives every test a clean temp folder.
try:
    import pytest

    @pytest.fixture
    def tmp(tmp_path):
        fresh(tmp_path)
        return tmp_path
except ImportError:                          # pytest is optional
    pass


if __name__ == "__main__":
    raise SystemExit(main())
