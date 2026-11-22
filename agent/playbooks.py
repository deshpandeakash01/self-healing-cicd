"""
Playbooks: which actions are allowed for which kind of failure.

WHY
    Different failures deserve different treatment. Re-running fixes a network
    blip but not a broken test; a code change fixes a broken test but is
    pointless for a network blip. Writing the allowed actions down as data
    lets the guardrail enforce them in code instead of trusting the LLM.

    The playbook is chosen by the DIAGNOSED category (the LLM's, or the
    classifier's when no LLM was used). The classifier's label is given to the
    LLM as a hint only: it can be wrong, and a wrong hint should not forbid
    the right action.
"""

ACTIONS = ("rerun", "fix_pr", "escalate")

# Categories a diagnosis may use. The first five are the classifier's classes.
CATEGORIES = (
    "dependency", "flaky_infra", "lint_build", "other_bot_infra",
    "test_failure", "config", "other",
)

PLAYBOOKS = {
    "flaky_infra": {
        "allowed": ("rerun", "escalate"),
        "guidance": "Transient problem outside the code (network, timeout, "
                    "runner). Re-running should pass. No code change.",
    },
    "dependency": {
        "allowed": ("fix_pr", "escalate"),
        "guidance": "A package could not be found, resolved or installed. "
                    "Fix the dependency file (add, pin or correct a version).",
    },
    "lint_build": {
        "allowed": ("fix_pr", "escalate"),
        "guidance": "The code does not compile, or a linter / type checker / "
                    "formatter rejected it. Fix the reported lines.",
    },
    "test_failure": {
        "allowed": ("fix_pr", "escalate"),
        "guidance": "A test failed. Decide whether the code or the test is "
                    "wrong; normally fix the code. Never weaken a test.",
    },
    "other_bot_infra": {
        "allowed": ("escalate",),
        "guidance": "The automation around the code failed (bots, caches, "
                    "containers, credentials). Not fixable by editing the repo.",
    },
    "config": {
        "allowed": ("escalate",),
        "guidance": "The workflow or its environment is misconfigured. The "
                    "agent may not edit workflow files or secrets.",
    },
    "other": {
        "allowed": ("escalate",),
        "guidance": "Does not fit a known category. A human should look.",
    },
}


def allowed_actions(category: str) -> tuple:
    return PLAYBOOKS.get(category, PLAYBOOKS["other"])["allowed"]


def guidance(category: str) -> str:
    return PLAYBOOKS.get(category, PLAYBOOKS["other"])["guidance"]
