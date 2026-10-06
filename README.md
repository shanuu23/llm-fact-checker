# LLM fact checker

This repository currently contains one evidence-grounded baseline: can local `llama3.2:3b` classify FEVER SUPPORTS and REFUTES claims when it receives a complete gold Wikipedia evidence set?

The model may predict `SUPPORTED`, `REFUTED`, or `NOT ENOUGH INFORMATION`. The fixed benchmark sample contains no gold NOT ENOUGH INFO cases, so it does not measure NEI accuracy. This baseline also does not test evidence retrieval, robustness, stability, sensitivity, calibration, or general real-world fact-checking.

## Data layout

Download FEVER separately and keep the immutable files at:

```text
data/fever/raw/
├── shared_task_dev.jsonl
└── wiki-pages/wiki-pages/
    ├── wiki-001.jsonl
    ├── ...
    └── wiki-109.jsonl
```

`data/fever/raw/` is ignored by Git because the extracted Wikipedia dump is several gigabytes. The preparation command streams those files; it never loads or stores whole Wikipedia articles.

## Environment

The project uses Python 3.12 and `uv`:

```bash
uv sync
ollama list  # must include llama3.2:3b
```

The experiment calls the local Ollama HTTP API directly and never downloads a model.

## Prepare the fixed 150-case sample

```bash
uv run python -m llm_fact_checker.baseline prepare
```

Preparation selects 75 SUPPORTS and 75 REFUTES claims with seed 42. It removes duplicate IDs and claim text, chooses one complete gold evidence set per claim (fewest unique sentences, then stable page/sentence tie-break), and streams the wiki files to resolve only the required sentences. Unresolvable cases are logged and replaced from the deterministic candidate order.

The auditable output is `data/fever/processed/fever_baseline_150.jsonl`. Re-running preparation against identical raw files produces the same sample.

## Run inference

A smoke test uses its own output directory and cannot overwrite the full run:

```bash
uv run python -m llm_fact_checker.baseline run --limit 3
```

Run all 150 fixed cases:

```bash
uv run python -m llm_fact_checker.baseline run
```

Existing predictions are protected. Use `--overwrite` only when intentionally replacing a run.

Prepare and run the complete pipeline in one command:

```bash
uv run python -m llm_fact_checker.baseline all
```

## Re-evaluate saved predictions

```bash
uv run python -m llm_fact_checker.baseline evaluate
uv run python -m llm_fact_checker.baseline evaluate --results-dir results/fever/llama3.2_3b/smoke_3
```

The full run is stored under `results/fever/llama3.2_3b/baseline/`:

```text
predictions.jsonl
metrics.json
confusion_matrix.csv
errors.csv
report.md
run_config.json
```

Raw model responses, invalid formats, inference failures, and incorrect predictions are retained. Gold labels are never placed in prompts.

## Run the five-run consistency experiment

The consistency experiment reuses the exact prepared dataset, prompt, evidence formatting, and strict parser from the baseline. It first verifies that the prepared dataset SHA-256 is `6bc7cf910cb0b31186e5fc397ebc7cefab2de1ee65d29ae601e369d8758344cb`, then runs all cases five times at temperature 0.2 with seeds 101, 202, 303, 404, and 505.

Run a three-case development check (15 inference requests):

```bash
uv run python -m llm_fact_checker.baseline consistency --limit 3
```

Run the complete experiment (750 inference requests):

```bash
uv run python -m llm_fact_checker.baseline consistency
```

Smoke results are isolated under `results/fever/llama3.2_3b/consistency_5run_smoke_3/`. Full results are written under `results/fever/llama3.2_3b/consistency_5run/`. Existing results are protected unless `--overwrite` is passed explicitly.

The experiment reports exact 5/5 consistency, mean modal agreement, per-run FEVER accuracy, stable-correct versus stable-wrong cases, consensus predictions, NEI behaviour, invalid outputs, and inference failures. Consistency measures repeatability, not correctness or robustness to changed prompts or evidence.

## Run the evidence-robustness experiment

First generate deterministic control, reversed-evidence, and distractor-evidence files plus a 20-case human audit:

```bash
uv run python -m llm_fact_checker.baseline robustness-prepare
```

Inspect `results/fever/llama3.2_3b/robustness/prepared/distractor_audit.md`, then run the isolated three-case smoke test:

```bash
uv run python -m llm_fact_checker.baseline robustness-smoke
```

Only after approving the distractors, run all 150 cases:

```bash
uv run python -m llm_fact_checker.baseline robustness
```

The experiment uses the unchanged fixed dataset, baseline prompt and strict parser at temperature 0 and seed 42. It reports reorder and distractor invariance separately, along with paired FEVER accuracy changes, correctness outcomes, label transitions, and NEI transitions. Existing robustness artifacts are protected unless `--overwrite` is passed explicitly.

## Run the VitaminC sensitivity experiment

The first VitaminC experiment uses only `data/vitaminc/raw/vitaminc/test.jsonl`. Contrastive rows are linked by the explicit `case_id` plus exact claim text. A valid pair must contain exactly one SUPPORTS row and one REFUTES row with different evidence, matching page, revision type, and source revision identifier. The one ambiguous contradictory group is excluded.

Generate 100 deterministic, unique-claim pairs and the 20-pair audit:

```bash
uv run python -m llm_fact_checker.baseline vitaminc-prepare
```

Inspect `results/vitaminc/llama3.2_3b/sensitivity/prepared/pair_audit.md`, then run the isolated three-pair smoke test:

```bash
uv run python -m llm_fact_checker.baseline vitaminc-smoke
```

Only after approving the audit, run all 100 pairs (200 model calls):

```bash
uv run python -m llm_fact_checker.baseline vitaminc-sensitivity
```

The primary metric requires both predictions to be correct: SUPPORTED for the support evidence and REFUTED for the refute evidence. Ordinary verdict flips, individual accuracy, and NEI behavior are reported separately. Existing VitaminC artifacts are protected unless `--overwrite` is passed explicitly.

## Tests

Tests use only tiny synthetic fixtures and do not call Ollama or scan the real dump:

```bash
uv run python -m unittest discover -s tests -v
```
