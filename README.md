# Self-healing CI/CD pipeline

An agent that is started when a GitHub Actions run fails. It works out why the run failed, takes one action, and checks that the action worked:

- **re-run** the failed jobs, for a transient failure such as a network timeout;
- **open a pull request** with a fix, for a broken dependency, lint error or failing test;
- **hand it to a person** (a comment on the failing commit), for anything it may not or cannot fix.

A failure only counts as healed when CI passes afterwards. Every run is written to an audit log, and a dashboard shows them all.

It is packaged as a **GitHub Action**: a repository adds self-healing with one short workflow file (see *Automatic healing*).

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
3. **Investigate:** an LLM, taken from the **model queue** (below) and run through LangGraph, searches the log and reads repository files with tools. It then submits one diagnosis and one action. Fixes are exact text replacements, so it can safely edit files of any size.
4. **Guardrail:** plain code, not a prompt, decides whether the action may happen. Each kind of failure has its own allowed actions. Workflow files, secrets and keys are never edited, a fix may change at most 5 files, and a whole file is replaced only after it has been read in full. Anything blocked becomes a hand-over, with the reasons recorded.
5. **Act:** a fix is one commit on a new `self-heal/…` branch, and a pull request is opened against the branch that failed. The agent never pushes to an existing branch of yours.
6. **Verify and retry:** the agent waits for CI on the fix. If CI is still red, it reads the new log and tries once more on the same pull request. After that it hands over.

**The model queue.** Free LLM tiers are small and often busy, so the agent works through a queue of free models, best first:

| # | Model | Provider | May open fix PRs |
|---|---|---|---|
| 1 | `gemini-4-hydrogen` | Gemini | yes |
| 2 | `openai/gpt-oss-120b` | Groq | yes |
| 3 | `nvidia/nemotron-3-ultra-550b-a55b:free` | OpenRouter | yes |
| 4 | `gemini-3.5-flash` | Gemini | yes |
| 5 | `llama-3.3-70b-versatile` | Groq | no, diagnose only |
| 6 | `google/gemma-4-26b-a4b-it:free` | OpenRouter | no, diagnose only |

When a model is busy (503, timeout) it rests for a few minutes. When it runs out of its daily quota it is skipped until the reset. The **next model continues the same investigation**, given the earlier steps as plain text, so nothing read so far is lost. Only providers whose key is set join the queue. The queue remembers resting models between runs. Change it with `SH_MODEL_QUEUE` (`provider:model[@diagnose], ...`) or `--models`. Positions 1 and 4 are measured on the test repository; the others are ranked from their providers' descriptions, to be re-ranked from test results.

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
action.yml          the GitHub Action (runs the agent when CI fails)
agent/              the agent (LangGraph graph, tools, guardrail, GitHub client, CLI)
  run.py            command-line entry point
  action_guard.py   the Action's loop guards (when NOT to run)
  dashboard.py      builds the runs dashboard (one HTML page)
  models.py         the model queue (which LLM answers; switching when one is busy or out of quota)
  graph.py          the steps above
  guardrail.py      the rules, in code
  tools.py          what the LLM may do: search/read the log, list/read files, submit
  triage.py         the classifier step
  evaluate.py       scores rules / classifier / LLM / cascade on hand-labelled logs
clean_and_label.py  log cleaning used by triage (and, originally, dataset labelling)
data/classifier_logreg_shared_final.joblib   the trained classifier (scikit-learn 1.9.0)
tests/test_agent.py 52 tests with a stand-in LLM and a stand-in GitHub; no key, no network
tests/fixtures/     four real failed-CI logs the tests use
test_repo/          the demo repository: code, its CI, the Self-heal workflow, break_it.py
export_agent_repo.py older deployment helper (copies the agent to its own repo); not needed with the Action
```

The dataset of 8,912 logs, the labelled data and the research scripts are not in this repository. Comments such as "PROJECT_LOG 3.38" or "log 3.44" refer to the author's working log, which is kept private.

## Running it

Requires Python 3.12+.

```bash
pip install -r requirements-agent.txt
python tests/test_agent.py              # 51 of 51 passed, 1 skipped (needs research data)
```

The run commands below need settings in your environment, never in a file:
- `GITHUB_TOKEN`: a fine-grained token for the repository.
- At least one model key: `GOOGLE_API_KEY` (Gemini), `GROQ_API_KEY` (Groq) or `OPENROUTER_API_KEY` (OpenRouter). Each key you set adds that provider's models to the queue.

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
- `--models "gemini:gemini-4-hydrogen, groq:openai/gpt-oss-120b"`: use your own model queue, best first.
- `--no-memory`: don't show the LLM earlier fixes.
- `--mode classifier_only` or `--mode llm_only`: run only one of the two stages.

Run `python -m agent.run --help` for the full list.

## Automatic healing (the GitHub Action)

To make a repository heal itself whenever its CI fails:

1. Add two secrets under *Settings → Secrets and variables → Actions*:
   - `SELF_HEAL_TOKEN`: a fine-grained personal access token for that repository with Contents, Pull requests and Actions set to read and write. GitHub's built-in token can't be used, because changes pushed with it don't start CI, so the agent could never verify its fix.
   - `GOOGLE_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`: model keys. Set at least one; each adds its provider's models to the queue.
2. Add `.github/workflows/self-heal.yml` on the default branch. `workflows:` must name your CI workflow:

```yaml
name: Self-heal
on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]
permissions:
  contents: read
jobs:
  heal:
    if: github.event.workflow_run.conclusion == 'failure'
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - uses: deshpandeakash01/self-healing-cicd@v1
        with:
          github-token: ${{ secrets.SELF_HEAL_TOKEN }}
          google-api-key: ${{ secrets.GOOGLE_API_KEY }}
          groq-api-key: ${{ secrets.GROQ_API_KEY }}
          openrouter-api-key: ${{ secrets.OPENROUTER_API_KEY }}
          # live: true      # leave out at first: dry run, diagnosis only
```

Without `live: true` the agent only diagnoses. The diagnosis appears on the run's summary page and nothing in the repository changes. Read a few dry runs before turning `live` on.

The Action has its own loop guards. It never runs for a re-run, for the agent's own `self-heal/` branches, or for a run from a fork, whatever the calling workflow says. `test_repo/self-heal.yml` is the full version used by the test repository.

| Input | Default | Meaning |
|---|---|---|
| `github-token` | (required) | token described above |
| `google-api-key`, `groq-api-key`, `openrouter-api-key` | (at least one) | model keys; each adds its provider's models to the queue |
| `model-queue` | built-in queue | your own queue, best first: `provider:model[@diagnose], ...` |
| `live` | `false` | `true` lets it re-run jobs, push fix branches, open pull requests and comment |
| `model` | agent default | the first Gemini model in the built-in queue |
| `fallback-model` | agent default | the second Gemini model in the built-in queue; `none` removes it |
| `retry-overloaded` | `3` | how often to start again when every model is busy |
| `notify` | `true` | comment on the failing commit when handing over |
| `python-version` | `3.12` | Python used to run the agent |

Outputs: `outcome`, `pull-request`, `model` and `skipped-reason`, for later steps (for example, a chat notification).

## Runs dashboard

Every run of the Action uploads its audit record as an artifact (`self-heal-record-…`). This command collects them, together with any local runs, into one self-contained page:

```bash
python -m agent.dashboard --repo owner/name      # needs GITHUB_TOKEN with Actions: read
python -m agent.dashboard                        # local runs only (data/agent_runs.jsonl)
```

It writes `dashboard.html`. The page shows outcome counts, median time to heal and LLM usage. Each incident's row expands to show the diagnosis, every attempt with its diff, what backed each change (a CI log line, or found in the code only), and earlier tries such as overload retries or a fallback model. Downloaded records are cached in `data/dashboard_cache/`, because GitHub deletes artifacts after 90 days by default.

## Limitations

- It has been tested on one small Python repository. Logs from other languages are triaged, but repairing them is untested.
- Free tiers are small (Gemini about 20 requests a day per model; OpenRouter free models 50 a day) and often busy. The model queue spreads the work across providers, but a day of heavy testing can still use them all up. One investigation uses a few requests, up to about 15 for a hard case.
- Free models may log prompts and use them for training. For a private repository, set your own `model-queue` with providers you accept.
- A fix can be wrong in ways CI does not catch. Every change arrives as a pull request for a person to review; nothing is merged automatically.
