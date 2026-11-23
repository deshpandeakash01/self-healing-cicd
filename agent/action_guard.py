"""
The first step of the GitHub Action (action.yml): should the agent run at all?

WHY IN THE ACTION, NOT ONLY IN THE CALLER'S WORKFLOW
    The loop guards used to live in the "if:" of each repository's
    self-heal.yml (3.39). Anyone copying a shorter workflow would lose them.
    Here they run for every user of the Action, whatever their workflow says.

THE RULES (each one a reason to do nothing)
    - the event is not a finished workflow_run          -> nothing to heal
    - the run did not fail                               -> nothing to heal
    - it is a re-run (run_attempt > 1)                   -> a re-run the agent
      started itself failed again; starting again would loop
    - the branch is one of the agent's own fix branches  -> the agent already
      watches those itself
    - the run came from a fork                           -> a stranger's pull
      request must not drive an agent holding secrets and write access

Standard library only: it runs before Python packages are installed.
Event fields arrive as environment variables (set in action.yml), never
pasted into a shell command, so a branch name cannot inject commands.
"""

from __future__ import annotations

import os
import sys

FIX_BRANCH_PREFIX = "self-heal/"     # same as config.FIX_BRANCH_PREFIX


def should_run(env: dict) -> tuple[bool, str]:
    """(run?, reason) from the SH_EVENT_* variables action.yml passes in."""
    if env.get("SH_EVENT_NAME") != "workflow_run":
        return False, (f"started by '{env.get('SH_EVENT_NAME') or 'unknown'}', not by a "
                       "finished workflow run; use 'on: workflow_run' in the caller")
    if env.get("SH_EVENT_CONCLUSION") != "failure":
        return False, f"the run's conclusion is '{env.get('SH_EVENT_CONCLUSION')}', not 'failure'"
    try:
        attempt = int(env.get("SH_EVENT_ATTEMPT") or 1)
    except ValueError:
        attempt = 1
    if attempt > 1:
        return False, f"this is attempt {attempt} of the run (a re-run); not starting again"
    branch = env.get("SH_EVENT_BRANCH") or ""
    if branch.startswith(FIX_BRANCH_PREFIX):
        return False, f"'{branch}' is one of the agent's own fix branches"
    head_repo, repo = env.get("SH_EVENT_HEAD_REPO") or "", env.get("SH_REPOSITORY") or ""
    if not head_repo or head_repo != repo:
        return False, f"the run came from '{head_repo or 'unknown'}', not from {repo} (a fork?)"
    if not env.get("SH_EVENT_RUN_ID"):
        return False, "the event has no run id"
    return True, f"run {env.get('SH_EVENT_RUN_ID')} on '{branch}' failed"


def main() -> int:
    run, reason = should_run(dict(os.environ))
    print(("[guard] running: " if run else "[guard] skipped: ") + reason)
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"run={'true' if run else 'false'}\n")
            f.write(f"reason={reason.replace(chr(10), ' ')}\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary and not run:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(f"### Self-heal: skipped\n\n{reason}.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
