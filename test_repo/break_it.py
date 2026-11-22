"""
Break this repository's CI on purpose, one failure at a time.

    python break_it.py --list          what each case is and what SHOULD happen
    python break_it.py two-bugs        make the edit for one case

It only edits files. Commit and push the change yourself, on a branch:

    git switch main
    git switch -c case-4
    python break_it.py two-bugs
    git add -A
    git commit -m "Update"
    git push -u origin case-4

    To undo an edit that has not been committed:   git checkout .
    (and for "flaky":   del tests\\test_price_service.py)

KEEP THE ANSWER KEY AWAY FROM THE AGENT
    - This file is listed in .gitignore and must NOT be in the GitHub
      repository: the agent can read every file there, and this file says
      exactly what was broken.
    - Use neutral branch names (case-4) and commit messages ("Update"). The
      commit message is printed in the CI log, which the agent can search.

THE CASES
    Easy (one line; all three healed on 15 Nov 2026):
        test         a wrong sign in shop.py
        lint         an unused import in shop.py
        dependency   a package that does not exist in requirements.txt
    Harder (added 16 Nov 2026) - see --list for what each one is for.
    gauntlet (added 17 Nov 2026) - four faults in layers; see --list.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKFLOW = HERE / ".github" / "workflows" / "ci.yml"


def replace_once(path, old, new):
    if not path.exists():
        sys.exit(f"{path} not found.")
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        sys.exit(f"{path.name}: expected to find {old!r} exactly once. Is the file "
                 "already broken, or older than this script? Undo with: git checkout .")
    path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


# ---- easy ------------------------------------------------------------------

def break_test():
    replace_once(HERE / "shop.py", "(100 - percent)", "(100 + percent)")
    return ["shop.py: the discount is now added instead of subtracted"]


def break_lint():
    replace_once(HERE / "shop.py", '"""\n\nDEFAULT_TAX_RATE',
                 '"""\n\nimport os\n\nDEFAULT_TAX_RATE')
    return ["shop.py: an unused import was added"]


def break_dependency():
    path = HERE / "requirements.txt"
    text = path.read_text(encoding="utf-8")
    if "self-heal-no-such-package" in text:
        sys.exit("requirements.txt is already broken. Undo with: git checkout .")
    path.write_text(text.rstrip("\n") + "\nself-heal-no-such-package==9.9.9\n",
                    encoding="utf-8", newline="\n")
    return ["requirements.txt: a package that does not exist was added"]


# ---- harder ----------------------------------------------------------------

def break_two_bugs():
    replace_once(HERE / "shop.py", "(1 + rate)", "(1 - rate)")
    replace_once(HERE / "orders.py", ">= threshold", "> threshold")
    return ["shop.py: tax is subtracted instead of added",
            "orders.py: free shipping no longer applies exactly at the threshold"]


def break_hidden():
    # The first bug stops every test file from even loading, so the second
    # cannot show in the CI log until the first is fixed.
    replace_once(HERE / "shop.py", "DEFAULT_TAX_RATE = 0.18", "DEFAULT_TAX = 0.18")
    replace_once(HERE / "orders.py", "unit_price * quantity", "unit_price + quantity")
    return [("shop.py: the constant was renamed where it is defined but not where "
             "it is used (the module no longer imports)"),
            ("orders.py: a line's price is unit price PLUS quantity (hidden until "
             "the first bug is fixed)")]


def break_wrong_test():
    replace_once(HERE / "tests" / "test_shop.py",
                 "assert shop.apply_discount(200, 10) == 180",
                 "assert shop.apply_discount(200, 10) == 190")
    return [("tests/test_shop.py: the test now expects 190; the code (and its "
             "docstring) say 180")]


def break_workflow():
    if not WORKFLOW.exists():
        sys.exit(f"{WORKFLOW} not found. Run this in the copy that is on GitHub "
                 "(the one where ci.yml was moved into .github/workflows).")
    replace_once(WORKFLOW, "run: ruff check .", "run: ruff chekc .")
    return [".github/workflows/ci.yml: the lint command is misspelt"]


def break_flaky():
    path = HERE / "tests" / "test_price_service.py"
    if path.exists():
        sys.exit(f"{path.name} already exists. Remove it first.")
    path.write_text(
        'import os\n'
        'import socket\n'
        '\n'
        'HOST = "10.255.255.1" if os.environ.get("GITHUB_RUN_ATTEMPT") == "1" else "github.com"\n'
        '\n'
        '\n'
        'def test_price_service_is_reachable():\n'
        '    socket.create_connection((HOST, 443), timeout=5).close()\n',
        encoding="utf-8", newline="\n")
    return [("tests/test_price_service.py (new): a network call that times out on "
             "the FIRST attempt of a CI run and works when the run is re-run. "
             "A SIMULATION of a transient outage.")]


def break_gauntlet():
    """
    Four faults, in two layers, each needing judgement:
      layer 1  requirements.txt: "pytest" misspelt as "pytset". pip fails in
               BOTH jobs before anything else runs, so nothing below is
               visible yet. The right fix is to correct the name; deleting
               the line makes the test job fail for a new reason.
      layer 2  (visible only after layer 1 is fixed, and in TWO jobs at once)
               lint job:  orders.py imports json and never uses it
               test job:  shop.py subtracts tax instead of adding it (code bug)
                          tests/test_orders.py expects 25 x 4 = 104 (the TEST
                          is wrong; the docstring says "unit price times
                          quantity", so the code must not be "fixed")
    """
    replace_once(HERE / "requirements.txt", "pytest==", "pytset==")
    replace_once(HERE / "orders.py", "import shop\n", "import json\nimport shop\n")
    replace_once(HERE / "shop.py", "(1 + rate)", "(1 - rate)")
    replace_once(HERE / "tests" / "test_orders.py",
                 "orders.line_total(25, 4) == 100", "orders.line_total(25, 4) == 104")
    return ["requirements.txt: pytest misspelt as pytset (hides everything else)",
            "orders.py: an unused import (lint job)",
            "shop.py: tax subtracted instead of added (test job)",
            "tests/test_orders.py: a wrong expected value (test job; fix the TEST)"]


# name -> (function, what it tests, what a good outcome is)
CASES = {
    "test": (break_test, "one wrong sign", "healed; fix in shop.py"),
    "lint": (break_lint, "one unused import", "healed; fix in shop.py"),
    "dependency": (break_dependency, "one bad requirement",
                   "healed; fix in requirements.txt"),
    "two-bugs": (break_two_bugs, "a fix that must touch two files",
                 "healed; ONE pull request editing shop.py AND orders.py"),
    "hidden": (
        break_hidden,
        ("a second bug the first CI log cannot show (the agent's retry after "
         "a failed fix)"),
        ("healed, either at once (it read ahead) or at the second attempt "
         "(first fix -> CI fails -> it reads the new log -> second fix)"),
    ),
    "wrong-test": (break_wrong_test, "the test is wrong, the code is right",
                   "healed by editing tests/test_shop.py; shop.py NOT changed"),
    "workflow": (
        break_workflow,
        "a failure in a file the agent may not edit (the guardrail)",
        ("escalated; nothing written; the explanation names the misspelt "
         "command in ci.yml"),
    ),
    "flaky": (break_flaky, "a transient failure (the rerun action)",
              "healed by re-running the job; NO pull request"),
    "gauntlet": (
        break_gauntlet,
        ("four faults in two layers: a misspelt package hiding a lint error, "
         "a code bug and a wrong test, in two jobs at once"),
        ("healed in TWO attempts on one pull request: attempt 1 corrects "
         "pytset -> pytest (does not delete it); attempt 2 removes the import, "
         "fixes the tax sign in shop.py and the expected value in the TEST, "
         "leaving orders.py's arithmetic alone"),
    ),
}


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--list":
        for name, (_, purpose, good) in CASES.items():
            print(f"{name:12s} tests: {purpose}\n{'':12s} good : {good}")
        return
    if len(sys.argv) != 2 or sys.argv[1] not in CASES:
        sys.exit("usage: python break_it.py --list | " + " | ".join(CASES))
    name = sys.argv[1]
    fn, purpose, good = CASES[name]
    for line in fn():
        print("Broken -", line)
    print(f"Tests        : {purpose}")
    print(f"Good outcome : {good}")
    print('Now, on a branch with a neutral name:  git add -A ; '
          'git commit -m "Update" ; git push -u origin <branch>')


if __name__ == "__main__":
    main()
