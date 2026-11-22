# Self-healing CI/CD pipeline

An agent that is started when a GitHub Actions run fails. It works out why the run failed, takes one action, and checks that the action worked:

- **re-run** the failed jobs, for a transient failure such as a network timeout;
- **open a pull request** with a fix, for a broken dependency, lint error or failing test;
- **hand it to a person** (a comment on the failing commit), for anything it may not or cannot fix.

A failure only counts as healed when CI passes afterwards. Every run is written to an audit log.

## How it works

```
failed run ─► ingest ─► triage ──► confident flaky failure ─────────────► re-run
                          │
                          └──────► investigate (LLM + tools) ─► guardrail ─► act ─► verify
                                                                               ▲        │
                                                                               └ retry ◄┘ (CI still red)
```

1. **Ingest:** downloads the failed job's log, plus the ends of the logs of other jobs that failed in the same run.
2. **Triage:** a small classifier (TF-IDF + logistic regression, built from a dataset of 8,912 failed GitHub Actions logs from 106 repositories in 11 languages) labels the failure. It re-runs a clearly flaky failure on its own, with no LLM call. Anything else goes to the LLM, with the classifier's label as a hint only.
3. **Investigate:** an LLM (Google Gemini, through LangGraph) searches the log and reads repository files with tools. It then submits one diagnosis and one action. Fixes are exact text replacements, so it can safely edit files of any size.
4. **Guardrail:** plain code, not a prompt, decides whether the action may happen. Each kind of failure has its own allowed actions. Workflow files, secrets and keys are never edited, a fix may change at most 5 files, and a whole file is replaced only after it has been read in full. Anything blocked becomes a hand-over, with the reasons recorded.
5. **Act:** a fix is one commit on a new `self-heal/…` branch, and a pull request is opened against the branch that failed. The agent never pushes to an existing branch of yours.
6. **Verify and retry:** the agent waits for CI on the fix. If CI is still red, it reads the new log and tries once more on the same pull request. After that it hands over.

**What backs each change.** An early failure can hide later ones: a broken install stops lint and tests from ever running. The agent may fix faults it finds by reading the code, but it labels every edit either as backed by the CI log (with line numbers) or as found in the code only. "Log" claims are checked against the cited lines. The pull request lists code-only changes separately, under *please check these first*.

## Results so far

These results come from a small test repository, [`selfheal-test`](https://github.com/deshpandeakash01/selfheal-test), which is broken on purpose with `test_repo/break_it.py`. Each case was run on real GitHub with `gemini-4-hydrogen`.

| Case | What it tests | Result |
|---|---|---|
| test, lint, dependency | one-line faults | healed (pull request) |
| two-bugs | one fix that must touch two files | healed, one pull request |
| hidden | a second bug that the first log cannot show | healed |
| wrong-test | the test is wrong, the code is right | healed by fixing the test; code left alone |
| workflow | a fault in a file the agent may not edit | handed over; nothing written |
| flaky | a transient failure | healed by re-running; no pull request |
| gauntlet | four faults in two layers, two jobs failing at once | healed automatically: all four fixed in one commit, first attempt |

These are single runs on a test repository, with faults written by the author. They show the agent *can* do these things, not how often it does. The model ignores the temperature setting, so runs are not exactly repeatable. A systematic evaluation is future work.

## Repository layout

```
agent/              the agent (LangGraph graph, tools, guardrail, GitHub client, CLI)
  run.py            command-line entry point
  graph.py          the steps above
  guardrail.py      the rules, in code
  tools.py          what the LLM may do: search/read the log, list/read files, submit
  triage.py         the classifier step
  evaluate.py       scores rules / classifier / LLM / cascade on hand-labelled logs
clean_and_label.py  log cleaning used by triage (and, originally, dataset labelling)
data/classifier_logreg_shared_final.joblib   the trained classifier (scikit-learn 1.9.0)
tests/test_agent.py 42 tests with a stand-in LLM and a stand-in GitHub; no key, no network
tests/fixtures/     four real failed-CI logs the tests use
test_repo/          the demo repository: code, its CI, the Self-heal workflow, break_it.py
export_agent_repo.py copies just the agent into its own folder, for the workflow to check out
```

The dataset of 8,912 logs, the labelled data and the research scripts are not in this repository. Comments such as "PROJECT_LOG 3.38" or "log 3.44" refer to the author's working log, which is kept private.

## Running it

Requires Python 3.12+.

```bash
pip install -r requirements-agent.txt
python tests/test_agent.py              # 41 of 41 passed, 1 skipped (needs research data)
```

The run commands below need two settings in your environment, never in a file:
- `GITHUB_TOKEN`: a fine-grained token for the repository.
- `GOOGLE_API_KEY`: a Gemini key.

Set them in your shell's environment variables (for example in PowerShell, `$env:GOOGLE_API_KEY = "..."`), and never commit them.

Diagnose a failed run without changing anything (a dry run, the default):

```bash
python -m agent.run --repo owner/name --run-id 123456789
```

Diagnose a saved log file (no GitHub needed):

```bash
python -m agent.run --log-file path/to/log.txt
```

`--live` lets it re-run jobs and open pull requests. **Use it only on a repository that exists to be broken**, such as the test repository.

Useful options:
- `--retry-overloaded 3`: retry when Gemini is busy.
- `--fallback-model NAME`: the model to try when the main one is unavailable.
- `--no-memory`: don't show the LLM earlier fixes.
- `--mode classifier_only` or `--mode llm_only`: run only one of the two stages.

Run `python -m agent.run --help` for the full list.

## Automatic healing

To make a repository heal itself whenever its CI fails:

1. Put the agent in its own GitHub repository: `python export_agent_repo.py <folder>`, then push that folder.
2. Copy `test_repo/self-heal.yml` to `.github/workflows/self-heal.yml` on the default branch of the repository to be healed. It listens for the workflow named `CI`. Change the agent repository named in it to yours.
3. In that repository, add two secrets under *Settings → Secrets and variables → Actions*:
   - `SELF_HEAL_TOKEN`: a fine-grained token with Contents, Pull requests and Actions set to read/write on this repository, and Contents read-only on the agent repository. A personal token is needed because changes pushed with GitHub's built-in token do not start CI, so the agent could never verify its fix.
   - `GOOGLE_API_KEY`: your Gemini key.

The workflow skips the agent's own `self-heal/` branches, re-runs and forks, so the agent cannot set itself off in a loop.

## Limitations

- It has been tested on one small Python repository. Logs from other languages are triaged, but repairing them is untested.
- The Gemini free tier allows about 20 requests a day per model and is often overloaded. One investigation uses a few requests, up to about 10 for a hard case.
- A fix can be wrong in ways CI does not catch. Every change arrives as a pull request for a person to review; nothing is merged automatically.
