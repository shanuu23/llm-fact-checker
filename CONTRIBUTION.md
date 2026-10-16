# Contributing

Developer setup and maintenance rules for the delivered ModelLens prototype. For using the app
without a checkout, see the [Mac recipient instructions](docs/macos-start-here.txt).

## Setup

```bash
git clone https://github.com/shanuu23/llm-fact-checker.git
cd llm-fact-checker
uv sync --locked --group app
```

This installs the locked app dependencies as well as the default developer/notebook groups.
Plain `uv sync` does not select the app group, which the UI tests need. Python 3.12 is selected
by `.python-version`; uv can obtain it if it is missing.

You need uv for this developer workflow. Ollama and an installed model are needed only for real
inference, not saved comparisons or offline tests. Public prepared inputs are committed; raw
Wikipedia is needed only to rebuild FEVER evidence. See [data/README.md](data/README.md).
For launching the interface and packaging it, use the [app guide](app/README.md).

## Verification

Run from the repository root:

```bash
uv run --group app python -m unittest discover -s tests -v
make lint
```

The explicit app group makes the test command work on a fresh checkout. Tests use fixtures,
saved artifacts and fake model responses; this is not a full benchmark inference run.
For a focused UI change, the affected tests can be run first:

```bash
uv run --group app python -m unittest tests.test_app tests.test_charts tests.test_live_evaluator -v
```

The current Ruff configuration covers `app/` and `llm_fact_checker/`, not `scripts/` or `tests/`.
When changing the latter, check the touched files explicitly, for example:

```bash
uv run ruff check --config 'include = ["scripts/**/*.py", "tests/**/*.py"]' tests/test_app.py
```

Report which checks passed and which were not run. Real-model quick demos, browser inspection
and fresh-machine release acceptance are separate checks; do not claim them from offline tests.

## Branches and pull requests

- Branch from an up-to-date `main`; name it `<your-name>/<task>`, for example
  `shanu/data-auditing`.
- Keep a PR to one purpose. A refactor and a behaviour change belong in separate PRs.
- Before pushing, run the offline suite and lint commands in [Verification](#verification).
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

Use the [research workflow in the README](README.md#run-a-new-model) for additional recorded
benchmark experiments. Coordinate ownership before writing to a model's results directory;
preserve completed runs and report the model digest, workload, settings and invalid/failure counts.
Do not change the shared prompt, cases or seeds to improve an individual model's score.

An app user evaluating a model does not need to add it to the repository. Local runs are saved
under `.modellens/runs/` and compared through **Local evaluator runs**, separate from the
committed research experiments. Do not commit those private local outputs.

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
  discussing it. App libraries go in the `app` group; notebook and tooling libraries go in
  the `notebooks` or `dev` groups in `pyproject.toml`.
- Shared helpers live in `common.py`. Do not copy a helper into a second module.
- Follow Ruff formatting/import rules; see the current coverage limits in [Verification](#verification).
- Tests may read saved benchmark artifacts but must not call real models or read the raw Wikipedia dump.
- Notebooks are for audits and analysis. Research experiments run through the CLI; interactive
  local evaluations run through the app's worker.
- Do not regenerate results, rebuild ZIPs or change dependencies as incidental cleanup.

## Commit messages

First line: what the change does, in the imperative, under about 70 characters. Then a short body
saying *why*. Example from this repository:

```text
Consolidate shared helpers into common.py and prompts.py
```

## Maintenance coordination

The app and recorded full experiments are delivered; this document is not an outstanding-work
checklist. Assign an owner and scope for each new issue or change before editing shared files.
Use [the app release guide](app/README.md#share-and-verify) for packaging and acceptance checks.
Preserve contributor attribution and historical records when reorganising documentation.

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

