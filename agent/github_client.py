"""
Feature 2 - GitHub client: every call the agent makes to GitHub is here.

WHY ONE MODULE
    Keeping all GitHub access in one place means there is exactly one list of
    things the agent is able to do to a repository. Anything not in this file
    cannot happen.

READ AND WRITE ARE SEPARATED
    Read methods can never change anything. Write methods (rerun, branch,
    commit, pull request) first call _require_write(). A client created with
    read_only=True refuses them, so in dry-run mode a bug elsewhere still
    cannot write to GitHub.

The session / retry pattern is the one from dataset/fetch.py.
"""

from __future__ import annotations

import base64
import os
import time

import requests
from requests.exceptions import RequestException

API = "https://api.github.com"


class GitHubError(RuntimeError):
    def __init__(self, status: int, message: str, url: str = ""):
        super().__init__(f"GitHub API {status}: {message} ({url})")
        self.status = status


class ReadOnlyError(RuntimeError):
    """Raised when a write is attempted on a read-only client (dry-run)."""


class GitHubClient:
    def __init__(self, repo: str, token: str | None = None,
                 read_only: bool = True, session=None):
        """
        repo      : "owner/name"
        token     : defaults to the GITHUB_TOKEN environment variable
        read_only : True (default) blocks every write method
        """
        if repo.count("/") != 1:
            raise ValueError(f'repo must look like "owner/name", got {repo!r}')
        self.repo = repo
        self.read_only = read_only
        token = token or os.environ.get("GITHUB_TOKEN", "")
        self.session = session or requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    # ------------------------------------------------------------------ core
    def _request(self, method: str, path: str, *, params=None, json=None,
                 max_retries: int = 4, raw: bool = False):
        url = path if path.startswith("http") else f"{API}/repos/{self.repo}{path}"
        resp = None
        for attempt in range(max_retries):
            try:
                resp = self.session.request(method, url, params=params,
                                            json=json, timeout=(20, 120))
            except RequestException:
                time.sleep(min(2 ** attempt, 30))
                continue
            # Rate limited: wait as told, otherwise back off (as in fetch.py).
            limited = resp.status_code == 429 or (
                resp.status_code == 403
                and resp.headers.get("x-ratelimit-remaining") == "0")
            if limited:
                wait = int(resp.headers.get("retry-after") or 0) or min(2 ** attempt, 60)
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                time.sleep(min(2 ** attempt, 30))
                continue
            break
        if resp is None:
            raise GitHubError(0, "no response (network error)", url)
        if resp.status_code >= 400:
            try:
                message = resp.json().get("message", resp.text[:200])
            except ValueError:
                message = resp.text[:200]
            raise GitHubError(resp.status_code, message, url)
        if raw:
            return resp
        return resp.json() if resp.content else {}

    def _require_write(self, what: str):
        if self.read_only:
            raise ReadOnlyError(
                f"Blocked write ({what}): this client is read-only (dry-run).")

    # ------------------------------------------------------------- read: CI
    def get_run(self, run_id) -> dict:
        return self._request("GET", f"/actions/runs/{run_id}")

    def get_job(self, job_id) -> dict:
        return self._request("GET", f"/actions/jobs/{job_id}")

    def list_run_jobs(self, run_id) -> list[dict]:
        """All jobs of the run's latest attempt, with status and times."""
        jobs, page = [], 1
        while True:
            data = self._request("GET", f"/actions/runs/{run_id}/jobs",
                                 params={"filter": "latest", "per_page": 100,
                                         "page": page})
            batch = data.get("jobs", [])
            jobs.extend(batch)
            if len(batch) < 100:
                return jobs
            page += 1

    def failed_jobs(self, run_id) -> list[dict]:
        """Failed jobs of a run, the one that finished first comes first."""
        failed = [j for j in self.list_run_jobs(run_id)
                  if j.get("conclusion") == "failure"]
        failed.sort(key=lambda j: (j.get("completed_at") or "",
                                   j.get("started_at") or ""))
        return failed

    def get_job_log(self, job_id) -> str:
        resp = self._request("GET", f"/actions/jobs/{job_id}/logs", raw=True)
        return resp.content.decode("utf-8", errors="replace")

    def list_check_runs(self, ref: str) -> list[dict]:
        data = self._request("GET", f"/commits/{ref}/check-runs",
                             params={"per_page": 100})
        return data.get("check_runs", [])

    def list_workflow_runs(self, head_sha: str) -> list[dict]:
        """GitHub Actions runs for one commit. Needs only the Actions
        permission, which a fine-grained token can have; check runs on a
        private repository may be closed to such a token."""
        data = self._request("GET", "/actions/runs",
                             params={"head_sha": head_sha, "per_page": 100})
        return data.get("workflow_runs", [])

    def list_artifacts(self, prefix: str = "", max_pages: int = 20) -> list[dict]:
        """Workflow artifacts of the repository whose name starts with
        `prefix`, newest first (the runs dashboard reads the agent's run
        records this way)."""
        found = []
        for page in range(1, max_pages + 1):
            data = self._request("GET", "/actions/artifacts",
                                 params={"per_page": 100, "page": page})
            batch = data.get("artifacts", [])
            found += [a for a in batch if str(a.get("name", "")).startswith(prefix)]
            if len(batch) < 100:
                break
        return found

    def download_artifact(self, artifact_id) -> bytes:
        """The artifact as a zip file. GitHub answers with a redirect to
        storage; requests drops the token when following it to another host."""
        return self._request("GET", f"/actions/artifacts/{artifact_id}/zip", raw=True).content

    # ----------------------------------------------------------- read: repo
    def get_default_branch(self) -> str:
        return self._request("GET", "")["default_branch"]

    def branch_sha(self, branch: str) -> str | None:
        try:
            return self._request("GET", f"/git/ref/heads/{branch}")["object"]["sha"]
        except GitHubError as e:
            if e.status == 404:
                return None
            raise

    def get_file(self, path: str, ref: str | None = None) -> dict | None:
        """Returns {"path", "sha", "text"} or None if the file does not exist."""
        try:
            data = self._request("GET", f"/contents/{path.lstrip('/')}",
                                 params={"ref": ref} if ref else None)
        except GitHubError as e:
            if e.status == 404:
                return None
            raise
        if isinstance(data, list):
            raise GitHubError(400, f"{path} is a directory, not a file")
        raw = base64.b64decode(data.get("content", "") or "")
        return {"path": data["path"], "sha": data["sha"],
                "text": raw.decode("utf-8", errors="replace")}

    def list_dir(self, path: str = "", ref: str | None = None) -> list[dict]:
        """Returns [{"path", "type"}] for a directory ("file" or "dir")."""
        data = self._request("GET", f"/contents/{path.strip('/')}",
                             params={"ref": ref} if ref else None)
        if isinstance(data, dict):
            data = [data]
        return [{"path": d["path"], "type": d["type"]} for d in data]

    # ---------------------------------------------------------------- write
    def rerun_failed_jobs(self, run_id) -> None:
        self._require_write("rerun failed jobs")
        self._request("POST", f"/actions/runs/{run_id}/rerun-failed-jobs")

    def create_branch(self, branch: str, from_sha: str) -> None:
        self._require_write("create branch")
        self._request("POST", "/git/refs",
                      json={"ref": f"refs/heads/{branch}", "sha": from_sha})

    def commit_files(self, branch: str, parent_sha: str, files: list[tuple[str, str]],
                     message: str, create_branch: bool) -> str:
        """
        Commit several files as ONE commit on top of `parent_sha`, then point
        `branch` at it (creating the branch if asked). Returns the commit sha.

        Why not create_branch + put_file: that pushes the branch first, at the
        still-broken commit (CI runs once on broken code), then one commit per
        file (CI runs on every half-fixed state). Here GitHub sees a single
        push of a single commit that contains the whole fix (PROJECT_LOG 3.41).
        Files keep their mode (an executable stays executable).
        """
        self._require_write("commit files")
        parent = self._request("GET", f"/git/commits/{parent_sha}")
        base_tree = parent["tree"]["sha"]
        modes = {}
        tree = self._request("GET", f"/git/trees/{base_tree}", params={"recursive": "1"})
        for entry in tree.get("tree", []):
            if entry.get("type") == "blob":
                modes[entry["path"]] = entry.get("mode", "100644")
        new_tree = self._request("POST", "/git/trees", json={
            "base_tree": base_tree,
            "tree": [{"path": path.lstrip("/"), "mode": modes.get(path.lstrip("/"), "100644"),
                      "type": "blob", "content": text} for path, text in files],
        })
        commit = self._request("POST", "/git/commits", json={
            "message": message, "tree": new_tree["sha"], "parents": [parent_sha]})
        if create_branch:
            self._request("POST", "/git/refs",
                          json={"ref": f"refs/heads/{branch}", "sha": commit["sha"]})
        else:   # never force: if the branch moved meanwhile, fail rather than overwrite
            self._request("PATCH", f"/git/refs/heads/{branch}",
                          json={"sha": commit["sha"], "force": False})
        return commit["sha"]

    def put_file(self, path: str, text: str, branch: str, message: str) -> str:
        """Create or update one file on a branch. Returns the new commit sha."""
        self._require_write("commit file")
        existing = self.get_file(path, ref=branch)
        body = {
            "message": message,
            "content": base64.b64encode(text.encode("utf-8")).decode("ascii"),
            "branch": branch,
        }
        if existing:
            body["sha"] = existing["sha"]
        data = self._request("PUT", f"/contents/{path.lstrip('/')}", json=body)
        return data["commit"]["sha"]

    def open_pull_request(self, title: str, head: str, base: str, body: str) -> dict:
        self._require_write("open pull request")
        data = self._request("POST", "/pulls", json={
            "title": title, "head": head, "base": base, "body": body})
        return {"number": data["number"], "url": data["html_url"]}

    def comment_on_commit(self, sha: str, body: str) -> str | None:
        """Comment on a commit; it shows next to the red mark of the failed run.
        Returns the comment's web address."""
        self._require_write("comment on commit")
        data = self._request("POST", f"/commits/{sha}/comments", json={"body": body})
        return (data or {}).get("html_url")

    def comment(self, issue_number: int, body: str) -> None:
        """Comment on an issue or pull request."""
        self._require_write("comment")
        self._request("POST", f"/issues/{issue_number}/comments", json={"body": body})
