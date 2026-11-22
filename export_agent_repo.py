"""
Copy just what the agent needs to run into a separate folder, ready to push
as its own GitHub repository (deshpandeakash01/self-healing-agent).

WHY
    The automatic trigger (test_repo/self-heal.yml) runs on GitHub's
    machines, which must download the agent's code from somewhere. Pushing
    this whole project is not an option: dataset/ is about 215 MB, and the
    hand labels and research data do not belong next to running code.

WHAT IS COPIED
    agent/*.py                                   the agent
    clean_and_label.py                           cleaning + rules, used by triage
    requirements-agent.txt                       packages
    data/classifier_logreg_shared_final.joblib   the classifier
    plus a short README.md and .gitignore written here.

USAGE (from the project root)
    python export_agent_repo.py D:\\self-healing-agent

    Run it again after any change to the agent; it replaces the copied files
    and leaves the folder's .git alone, so you can commit and push the update.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = ["clean_and_label.py", "requirements-agent.txt",
         "data/classifier_logreg_shared_final.joblib"]

README = """# Self-healing agent

The code that the Self-heal workflow runs when CI fails. Exported from the
research project with `export_agent_repo.py`; do not edit here, edit there
and export again.

    python -m agent.run --repo OWNER/REPO --run-id RUN_ID          # dry run
    python -m agent.run --repo OWNER/REPO --run-id RUN_ID --live   # writes

Needs GITHUB_TOKEN and GOOGLE_API_KEY in the environment.
"""

GITIGNORE = """__pycache__/
*.pyc
# records the agent writes while it runs
data/*.jsonl
"""


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    target = Path(sys.argv[1]).resolve()
    if target == ROOT or ROOT in target.parents:
        sys.exit("Choose a folder OUTSIDE the project, e.g. D:\\self-healing-agent")
    target.mkdir(parents=True, exist_ok=True)

    agent_dir = target / "agent"
    if agent_dir.exists():
        shutil.rmtree(agent_dir)            # so deleted files do not linger
    agent_dir.mkdir()
    copied = []
    for src in sorted((ROOT / "agent").glob("*.py")):
        shutil.copy2(src, agent_dir / src.name)
        copied.append(f"agent/{src.name}")
    for rel in FILES:
        src = ROOT / rel
        if not src.exists():
            sys.exit(f"Missing {rel}. Is this the project root?")
        (target / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target / rel)
        copied.append(rel)
    (target / "README.md").write_text(README, encoding="utf-8")
    (target / ".gitignore").write_text(GITIGNORE, encoding="utf-8")

    print(f"Copied {len(copied)} files to {target}:")
    for rel in copied:
        print("  " + rel)
    print("\nNext, in that folder:  git init -b main ; git add . ; git commit -m "
          '"Agent" ; then add the GitHub remote and push (README: "Automatic healing").')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
