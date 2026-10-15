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
you plan to run. No raw data is needed to run the experiments: the samples, the whole FEVER dev set and
VitaminC's `test.jsonl` are committed. See [data/README.md](data/README.md) for what each command
requires.

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

## Adding a model

Follow the step-by-step guide in the README, **"Run a new model"**. In short:

1. `ollama pull <tag>` and check the exact tag appears in `ollama list`.
2. Smoke test: `uv run factcheck run --model <tag> --limit 3`, then delete the `smoke_3` folder.
3. Run the four experiments on the frozen samples, then (overnight)
   `scripts/run_whole_fever.sh <tag>` for the whole FEVER dev set.
4. Commit only `results/*/<model_slug>/` (the slug is the tag with `:` replaced by `_`).
5. Open a PR named `<your-name>/add-<model_slug>`; say in the description which experiments
   you ran, the four headline numbers, and anything unusual (for example a high invalid-format
   rate).

Nothing else changes: the prompt, samples and seeds are shared, so your numbers sit next to
everyone else's. One owner per model avoids two people overwriting the same folder.

## Results policy

See [results/README.md](results/README.md). In short: never hand-edit results, commit only your own
model's folders, keep the model digest, and share very large files by link instead of committing
them.

## Data

- Never commit anything else under `data/*/raw/` (the FEVER dev claims, the Wikipedia dump, other
  VitaminC splits); only VitaminC's `test.jsonl` is allowed through `.gitignore`.
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
| Full-dataset run and resumable runner | `data_prep/`, `experiments/` | TBD |
| Metrics, confidence intervals, scorecard | `evaluation.py`, new modules | TBD |
| Model runs: qwen2.5:7b | `results/*/qwen2.5_7b/` | Aryan Goel (`16aryan`) |
| Model runs: gemma3:4b | `results/*/gemma3_4b/` | TBD |
| Comparison tooling & scorecard (built on `results.py`) | `scripts/compare_models.py`, `results/CROSS_MODEL_SCORECARD.md`, `notebooks/02_*.ipynb` | Aryan Goel (`16aryan`) |
| Report, poster, recommendations | `docs/` | TBD |

## Completed Contributions: Aryan Goel (`16aryan`)

1. **`qwen2.5:7b` Benchmark Suite Execution**:
   - Evaluated `qwen2.5:7b` across all four core test-bench dimensions on the frozen evaluation samples (1,261 model calls via local Ollama inference, 0 inference failures):
     - **Baseline Correctness**: 90.67% accuracy (136/150; SUPPORTED 88.0%, REFUTED 93.33%; only 1 NEI).
     - **5-Run Stability**: 98.0% exact 5/5 consistency (147/150 cases); 99.86% modal agreement; only 1 unstable claim across 750 requests.
     - **Noise Robustness**: 90.9% reorder invariance (10/11); 95.24% distractor invariance (140/147) with 88.0% post-distractor accuracy and 0 non-NEI to NEI collapses.
     - **VitaminC Sensitivity**: 72.0% correct sensitivity (72/100 pairs); 77.55% flip rate; 88.0% support-side accuracy and 83.0% refute-side accuracy.
   - Committed all standardized output files to `results/fever/qwen2.5_7b/` and `results/vitaminc/qwen2.5_7b/`.

2. **Cross-Model Comparison & Scorecard Tooling**:
   - Created `scripts/compare_models.py`: A clean CLI utility that leverages `llm_fact_checker.results` to discover and aggregate headline metrics across all benchmarked models (`llama3.2_1b`, `llama3.2_3b`, `phi4-mini_3.8b`, `qwen2.5_7b`) without running inference.
   - Generated `results/CROSS_MODEL_SCORECARD.md`: An audited comparative scorecard summarizing model performance across Correctness, Stability, Robustness, and Sensitivity.
   - Authored `notebooks/02_model_benchmark_comparison.ipynb`: An interactive Jupyter notebook with 4-dimension bar charts, contradiction vs. confirmation balance breakdowns, and failure mode disagreement case studies.

