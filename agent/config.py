"""
Settings shared by every agent feature.

Each value can be overridden with an environment variable, so nothing has to
be edited in code to try a different model, threshold or limit.

Secrets are read from the environment and never stored in this file:
    GITHUB_TOKEN     GitHub token (needed for anything that talks to GitHub)
    GOOGLE_API_KEY   Gemini API key (GEMINI_API_KEY also works)
"""

import os
from pathlib import Path


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


# --- Paths -------------------------------------------------------------------
# The project root is the folder that contains this "agent" folder. Paths are
# built from it so the agent works no matter which folder it is started from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The saved classifier bundle written by train_classifiers.py.
MODEL_PATH = Path(os.environ.get(
    "SH_MODEL_PATH",
    PROJECT_ROOT / "data" / "classifier_logreg_shared_final.joblib",
))

# One JSON line per incident the agent handled: the audit trail.
AUDIT_LOG = Path(os.environ.get(
    "SH_AUDIT_LOG", PROJECT_ROOT / "data" / "agent_runs.jsonl"))

# One JSON line per fix that passed verification: the agent's memory.
FIX_MEMORY = Path(os.environ.get(
    "SH_FIX_MEMORY", PROJECT_ROOT / "data" / "agent_fix_memory.jsonl"))

# --- Triage ------------------------------------------------------------------
# Below this confidence the classifier's answer is not trusted and the failure
# is treated as "unknown". 0.6 is a starting value (PROJECT_LOG.txt, 3.5); it
# is to be re-tuned once the hand-labelled sample exists.
CONFIDENCE_THRESHOLD = _float("SH_CONFIDENCE_THRESHOLD", 0.6)

# A flaky_infra label must be at least this confident before the job is
# re-run with NO LLM call. It is higher than the threshold above because the
# generalisation study (experiments/generalisation_study.py, PROJECT_LOG 3.13)
# found that on repositories the classifier has not seen, "flaky_infra at
# >= 0.6" was right only 55% of the time, against 85% at >= 0.8.
DIRECT_RERUN_THRESHOLD = _float("SH_DIRECT_RERUN_THRESHOLD", 0.8)

# How the classifier and the LLM are combined (needed for the evaluation):
#   cascade          classifier first; LLM investigates unless the classifier
#                    alone is enough (confident flaky_infra -> rerun)
#   llm_only         classifier ignored; every failure goes to the LLM
#   classifier_only  no LLM at all; flaky_infra -> rerun, everything else is
#                    reported with the classifier's label
MODE = os.environ.get("SH_MODE", "cascade")
MODES = ("cascade", "llm_only", "classifier_only")

# --- LLM ---------------------------------------------------------------------
# Kept as a setting because Google's model list changes often. Check
# https://ai.google.dev/gemini-api/docs/models for current names.
# "or": an empty variable (as a workflow passes an unset value) means the default.
GEMINI_MODEL = os.environ.get("SH_GEMINI_MODEL") or "gemini-4-hydrogen"
# Tried when the main model stays unavailable (503 after the retries, or its
# daily quota used up). Each model has its own load and its own free quota.
# "none" switches it off. (PROJECT_LOG 3.42)
GEMINI_FALLBACK_MODEL = os.environ.get("SH_GEMINI_FALLBACK_MODEL") or "gemini-3.5-flash"
if GEMINI_FALLBACK_MODEL.lower() == "none":
    GEMINI_FALLBACK_MODEL = ""

# How hard the model thinks before answering: "low", "medium", "high" or
# "xhigh" on Gemini 4. Empty = do not send the setting, so the model uses its
# own default (reported to be "high" for gemini-4-hydrogen). Lower is faster
# and cheaper. UNTESTED against the real service (PROJECT_LOG 3.15).
GEMINI_THINKING_LEVEL = os.environ.get("SH_GEMINI_THINKING_LEVEL", "").strip().lower()

# --- Limits (the agent stops and hands over to a human beyond these) ----------
# The model queue (agent/models.py, log 3.52): its order, and where it
# remembers which models are resting (busy or out of quota) between runs.
MODEL_QUEUE = os.environ.get("SH_MODEL_QUEUE", "").strip()
QUEUE_STATE = os.environ.get("SH_QUEUE_STATE") or str(PROJECT_ROOT / "data" / "model_queue_state.json")
QUEUE_MAX_WAIT = float(os.environ.get("SH_QUEUE_MAX_WAIT") or 180)   # seconds

MAX_LLM_STEPS = _int("SH_MAX_LLM_STEPS", 15)       # LLM calls per investigation
MAX_FIX_ATTEMPTS = _int("SH_MAX_FIX_ATTEMPTS", 2)  # actions tried per incident
MAX_UPSTREAM_HOPS = _int("SH_MAX_UPSTREAM_HOPS", 1)
# When several jobs of a run fail, the LLM also sees the end of the other
# failed jobs' logs, so one fix can cover every cause (PROJECT_LOG 3.38).
MAX_OTHER_JOBS = _int("SH_MAX_OTHER_JOBS", 3)
OTHER_JOB_TAIL_LINES = _int("SH_OTHER_JOB_TAIL_LINES", 150)

# --- Guardrail ---------------------------------------------------------------
MAX_EDIT_FILES = _int("SH_MAX_EDIT_FILES", 5)
MAX_EDIT_BYTES = _int("SH_MAX_EDIT_BYTES", 100_000)   # per file
# A fix may also be given as exact text replacements inside a file (needed for
# files too large to send back whole). At most this many per file.
MAX_REPLACEMENTS = _int("SH_MAX_REPLACEMENTS", 10)
# The agent may never edit these. CI definitions and secrets are exactly what
# someone would target by planting instructions in a log.
PROTECTED_PREFIXES = (".github/", ".git/")
PROTECTED_NAMES = (".env", ".npmrc", ".pypirc", "id_rsa")
PROTECTED_SUFFIXES = (".pem", ".key", ".p12", ".pfx")

# --- Tools -------------------------------------------------------------------
MAX_TOOL_OUTPUT_CHARS = _int("SH_MAX_TOOL_OUTPUT_CHARS", 6_000)
MAX_FILE_READ_CHARS = _int("SH_MAX_FILE_READ_CHARS", 30_000)
MAX_FILE_READ_LINES = _int("SH_MAX_FILE_READ_LINES", 400)   # per line-range read

# --- GitHub writes -----------------------------------------------------------
FIX_BRANCH_PREFIX = "self-heal/"

# --- Verification ------------------------------------------------------------
POLL_INTERVAL = _float("SH_POLL_INTERVAL", 15)               # seconds
VERIFY_START_TIMEOUT = _float("SH_VERIFY_START_TIMEOUT", 300)  # wait for CI to start
VERIFY_TIMEOUT = _float("SH_VERIFY_TIMEOUT", 1800)            # wait for CI to finish

# --- Fix memory --------------------------------------------------------------
MEMORY_TOP_K = _int("SH_MEMORY_TOP_K", 2)
MEMORY_MIN_SIMILARITY = _float("SH_MEMORY_MIN_SIMILARITY", 0.35)
