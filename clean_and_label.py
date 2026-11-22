import csv
import glob
import os
import re

RAW_DIR = "dataset/data/raw"
LOG_ROOT = os.path.join(RAW_DIR, "logs")
OUTPUT_PATH = "data/labelled_logs.csv"

WINDOW_LINES = 45          # lines kept, ending at the last error marker
ERROR_MARKER = "##[error]"
BOT_INFRA_PATTERNS = [
    r"Dependabot encountered an error",
    r"Label prediction cannot proceed",
    r"was not restored from cache",
    r"couldn't find remote ref",
    r"Dev container build failed",
    r"docker buildx build",
    r"Loaded a configuration file for version .* but running version",
    r"Could not find the PR that triggered this workflow",
    r"scala-steward",
    r"runner has received a shutdown signal",
    r"binary not found on PATH",
    r"could not read Username for",
    r"not found in this repository",
]
DERIVATIVE_SIGNALS = [
    r"No test report files were found",
    r"no artifacts? (were )?found",
    r"could not download artifact",
    r"required to succeed jobs failed",
    r"Job statuses:",
]
LANGUAGE_RULES = {
    "python": [
        ("dependency", [r"ModuleNotFoundError", r"Could not find a version",
                         r"No matching distribution found",
                         r"ImportError: No module named",
                         r"ResolutionImpossible"]),
        ("test_failure", [r"AssertionError", r"FAILED tests?/",
                           r"short test summary", r"\d+ failed,? \d* ?passed"]),
        ("lint_build", [r"SyntaxError", r"flake8", r"mypy", r"error: "]),
        ("flaky_infra", [r"Timeout", r"ConnectionError", r"Connection reset",
                          r"Could not resolve host"]),
        ("config", [r"Missing environment variable", r"Invalid workflow file",
                     r"Input required and not supplied"]),
    ],
    "javascript": [
        ("dependency", [r"npm ERR!", r"Cannot find module", r"ERESOLVE",
                         r"peer dep"]),
        ("test_failure", [r"Tests:\s*\d+ failed", r"\u25cf ", r"expect\(received\)",
                           r"AssertionError: expected", r"\[ELIFECYCLE\]"]),
        ("lint_build", [r"eslint", r"error TS\d+", r"Unexpected token",
                         r"Module build failed"]),
        ("flaky_infra", [r"ECONNRESET", r"ETIMEDOUT", r"network timeout"]),
        ("config", [r"Missing environment variable", r"Invalid workflow file",
                     r"Input required and not supplied"]),
    ],
    "java": [
        ("dependency", [r"Could not resolve dependencies",
                         r"package .* does not exist"]),
        ("test_failure", [r"Tests run: \d+.*Failures: [1-9]",
                           r"There was \d+ failure"]),
        ("lint_build", [r"cannot find symbol", r"COMPILATION ERROR",
                         r"checkstyle", r"spotbugs",
                         r"MojoFailureException", r"\[ERROR\] \[Help \d+\]"]),
        ("flaky_infra", [r"Timeout", r"Connection reset", r"SocketTimeoutException"]),
        ("config", [r"Missing environment variable", r"Invalid workflow file",
                     r"Input required and not supplied"]),
    ],
    "go": [
        ("dependency", [r"no required module provides package",
                         r"missing go\.sum entry"]),
        ("test_failure", [r"--- FAIL: Test", r"^FAIL\t"]),
        ("lint_build", [r"undefined: ", r"declared and not used",
                         r"govulncheck", r"\.go:\d+:\d+: "]),
        ("flaky_infra", [r"i/o timeout", r"TLS handshake timeout",
                          r"dial tcp", r"no such host"]),
        ("config", [r"Missing environment variable", r"Invalid workflow file",
                     r"Input required and not supplied"]),
    ],
    "rust": [
        ("dependency", [r"error: failed to select a version",
                         r"no matching package"]),
        ("test_failure", [r"test result: FAILED", r"panicked at",
                           r"\d+ passed, \d+ failed"]),
        ("lint_build", [r"error: could not compile", r"error\[E\d+\]"]),
        ("flaky_infra", [r"Timeout", r"Connection reset"]),
        ("config", [r"Missing environment variable", r"Invalid workflow file",
                     r"Input required and not supplied"]),
    ],
    "cpp": [
        ("dependency", [r"Could not find package", r"Could NOT find"]),
        ("test_failure", [r"tests? failed", r"FAILED", r"\d+ assertions? failed"]),
        ("lint_build", [r"error: ", r"warnings-as-errors", r"ninja: build stopped",
                         r"gmake: \*\*\*", r"fatal error:"]),
        ("flaky_infra", [r"Timeout", r"Connection reset"]),
        ("config", [r"Missing environment variable", r"Invalid workflow file",
                     r"Input required and not supplied"]),
    ],
    "csharp": [
        ("dependency", [r"NU1101", r"Unable to resolve"]),
        ("test_failure", [r"Failed!\s*-\s*Failed:", r"\[FAIL\]"]),
        ("lint_build", [r"error CS\d+"]),
        ("flaky_infra", [r"Timeout", r"Connection reset"]),
        ("config", [r"Missing environment variable", r"Invalid workflow file",
                     r"Input required and not supplied"]),
    ],
    "kotlin": [
        ("dependency", [r"Could not resolve", r"Could not find \S+\.jar"]),
        ("test_failure", [r"\d+ tests? failed", r"FAILED$"]),
        ("lint_build", [r"Execution failed for task", r"BUILD FAILED",
                         r"e: .*\.kt:"]),
        ("flaky_infra", [r"Timeout", r"Connection reset"]),
        ("config", [r"Missing environment variable", r"Invalid workflow file",
                     r"Input required and not supplied"]),
    ],
    "scala": [
        ("dependency", [r"unresolved dependency", r"Could not find or load main class"]),
        ("test_failure", [r"\*\*\* FAILED \*\*\*", r"Tests: .*Failed",
                           r"\(scripted\) Failed tests", r"Failed tests:"]),
        ("lint_build", [r"\[error\].*\.scala:\d+", r"error: .*scala"]),
        ("flaky_infra", [r"Timeout", r"Connection reset"]),
        ("config", [r"Missing environment variable", r"Invalid workflow file",
                     r"Input required and not supplied"]),
    ],
    "ruby": [
        ("dependency", [r"Could not find gem", r"Bundler::GemNotFound"]),
        ("test_failure", [r"Failures:", r"\d+ examples?, \d+ failures?",
                           r"rake aborted!"]),
        ("lint_build", [r"RuboCop", r"SyntaxError", r"\[ERROR\] Found \d+ error"]),
        ("flaky_infra", [r"Timeout", r"Connection reset"]),
        ("config", [r"Missing environment variable", r"Invalid workflow file",
                     r"Input required and not supplied"]),
    ],
    "php": [
        ("dependency", [r"but it conflicts with your root composer\.json",
                         r"Your requirements could not be resolved"]),
        ("test_failure", [r"There was \d+ failure", r"FAILURES!",
                           r"Tests: \d+, Assertions: \d+, Failures: [1-9]"]),
        ("lint_build", [r"\[ERROR\] Found \d+ error", r"PHP Fatal error",
                         r"PHP Parse error"]),
        ("flaky_infra", [r"Timeout", r"Connection reset"]),
        ("config", [r"Missing environment variable", r"Invalid workflow file",
                     r"Input required and not supplied"]),
    ],
}
def clean_and_anchor(raw: str) -> str:
    """Strip timestamps/ANSI noise, then return a window ending at the LAST
    real error marker instead of the raw file's tail (see fix #1 above)."""
    ts_re = re.compile(r"\d{4}-\d{2}-\d{2}T[\d:.]+Z")
    ansi_re = re.compile(r"\x1b\[[0-9;]*m")
    lines = []
    for line in raw.splitlines():
        line = ts_re.sub("", line)
        line = ansi_re.sub("", line)
        line = line.strip()
        if line:
            lines.append(line)

    err_idx = None
    for i, l in enumerate(lines):
        if ERROR_MARKER in l:
            err_idx = i
    if err_idx is None:
        # no marker found at all -- fall back to the plain tail, but this
        # should be rare (97% of real logs had a marker in our sample)
        window = lines[-WINDOW_LINES:]
    else:
        start = max(0, err_idx - (WINDOW_LINES - 1))
        window = lines[start:err_idx + 1]
    deduped = []
    for l in window:
        if not deduped or deduped[-1] != l:
            deduped.append(l)
    return "\n".join(deduped)
def classify(language: str, cleaned_text: str) -> str:
    for pat in BOT_INFRA_PATTERNS:
        if re.search(pat, cleaned_text, re.IGNORECASE | re.MULTILINE):
            return "other_bot_infra"
    for pat in DERIVATIVE_SIGNALS:
        if re.search(pat, cleaned_text, re.IGNORECASE | re.MULTILINE):
            return "needs_traceback"

    rules = LANGUAGE_RULES.get(language, [])
    for category, patterns in rules:
        for pat in patterns:
            if re.search(pat, cleaned_text, re.IGNORECASE | re.MULTILINE):
                return category
    return "unlabelled"
def iter_log_files():
    for lang_dir in sorted(glob.glob(os.path.join(LOG_ROOT, "*"))):
        if not os.path.isdir(lang_dir):
            continue
        language = os.path.basename(lang_dir)
        for path in sorted(glob.glob(os.path.join(lang_dir, "*.txt"))):
            yield language, path
def main():
    rows = []
    counts = {}
    seen_files = set()

    for language, path in iter_log_files():
        job_id = os.path.splitext(os.path.basename(path))[0].split("__")[-1]
        if job_id in seen_files:        # guards against duplicate filenames
            continue
        seen_files.add(job_id)

        raw = open(path, encoding="utf-8", errors="replace").read()
        cleaned = clean_and_anchor(raw)
        if not cleaned.strip():
            continue
        category = classify(language, cleaned)
        counts[(language, category)] = counts.get((language, category), 0) + 1

        rows.append({
            "log_id": job_id,
            "language": language,
            "cleaned_log": cleaned,
            "category": category,
            "label_source": "rule_based",
            "reviewed_by_human": False,
        })

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "log_id", "language", "cleaned_log", "category",
            "label_source", "reviewed_by_human"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} labelled rows to {OUTPUT_PATH}\n")
    print(f"{'language':12s} {'category':18s} count")
    for (lang, cat), n in sorted(counts.items()):
        print(f"{lang:12s} {cat:18s} {n}")
if __name__ == "__main__":
    main()
