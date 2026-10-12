# Contributing

How the group works on this repository. Keep it short; update it when the process changes.

## Setup

```bash
git clone https://github.com/shanuu23/llm-fact-checker.git
cd llm-fact-checker
uv sync                  # one command installs everything
make test                # must pass before you open a PR
```

You need [uv](https://docs.astral.sh/uv/) and a local [Ollama](https://ollama.com/) with the model
you plan to run. Raw FEVER data is optional, but running the VitaminC sensitivity experiment needs
`test.jsonl`; see [data/README.md](data/README.md) for what each command requires.

## Branches and pull requests

- Branch from an up-to-date `main`; name it `<your-name>/<task>`, for example
  `shanu/data-auditing`.
- Keep a PR to one purpose. A refactor and a behaviour change belong in separate PRs.
- Before pushing: `make test` and `make lint`.
- A PR needs one teammate's review before it merges to `main`. Say in the description what you ran
  and what changed in `results/`.
- Do not push to `main` directly or force-push shared branches.

## Frozen files

These define the experiments. Results from different models are comparable only while they stay
identical, so changing them needs a team decision and its own PR with the reason in the
description.

| Frozen | Where |
|---|---|
| The prompt, labels and parser | `llm_fact_checker/prompts.py` |
| The two samples and their hashes | `data/*/processed/`, `data/HASHES.txt` |
| Experiment settings (temperatures, seeds, sample sizes) | constants in `llm_fact_checker/experiments/` |
| Committed pilot results | `results/*/llama3.2_3b/` |

The golden tests in `tests/test_golden_results.py` fail if the prompt, a metric, or a report
changes, so you will find out immediately.

## Results policy

See [results/README.md](results/README.md). In short: never hand-edit results, commit only your own
model's folders, keep the model digest, and share very large files by link instead of committing
them.

## Data

- Never commit `data/*/raw/`; it is gitignored.
- Do not add new files under `data/*/processed/` without agreeing on them, and record any new
  file's hash in `data/HASHES.txt`.

## Code conventions

- The experiment pipeline uses only the standard library. Do not add runtime dependencies without
  discussing it; notebook and tooling libraries go in the `notebooks` or `dev` groups in
  `pyproject.toml`.
- Shared helpers live in `common.py`. Do not copy a helper into a second module.
- Formatting and imports are enforced by ruff (`make format`).
- Tests use small synthetic fixtures; never call Ollama or read the raw Wikipedia dump in a test.
- Notebooks are for audits and analysis only. Experiment runs are CLI commands.

## Commit messages

First line: what the change does, in the imperative, under about 70 characters. Then a short body
saying *why*. Example from this repository:

```text
Consolidate shared helpers into common.py and prompts.py
```

## Workstreams and owners

Fill in as the group assigns work. Each workstream owns its files so two people rarely edit the
same one.

| Workstream | Files | Owner |
|---|---|---|
| Multi-model support (`--model`, per-model results paths) | `cli.py`, `experiments/` | TBD |
| Full-dataset run and resumable runner | `data_prep/`, `experiments/` | TBD |
| Metrics, confidence intervals, scorecard | `evaluation.py`, new modules | TBD |
| Model runs: qwen2.5:7b | `results/*/qwen2.5_7b/` | TBD |
| Model runs: gemma3:4b | `results/*/gemma3_4b/` | TBD |
| Comparison page and Streamlit app (built on `results.py`) | `app/`, new `compare.py` | TBD |
| Report, poster, recommendations | `docs/` | TBD |
