# Results

Every experiment writes its outputs here. Results are committed so the whole team (and the
planned Streamlit app) can read them without running a model or downloading raw data.

## Layout

```text
results/<dataset>/<model_slug>/<experiment>/
```

| Part | Meaning | Example |
|---|---|---|
| `<dataset>` | benchmark the cases come from | `fever`, `vitaminc` |
| `<model_slug>` | Ollama tag with `:` replaced by `_` | `llama3.2_3b`, `qwen2.5_7b` |
| `<experiment>` | one folder per experiment | `baseline`, `consistency_5run`, `robustness`, `sensitivity` |

Folders named `smoke*` (for example `smoke_3`, `consistency_5run_smoke_3`, `robustness/smoke`) are
3-case development checks, not results. Do not cite them.

> The model slug is currently written by a constant in `llm_fact_checker/cli.py` and is
> `llama3.2_3b` for every run. Running another model correctly is the next planned change.

## Files per experiment

Each experiment folder contains a `report.md` for people, a `run_config.json` for audit, and the
machine-readable files below.

### `fever/<model>/baseline/`

| File | Contents |
|---|---|
| `predictions.jsonl` | one row per case: `fever_id`, `claim`, `evidence`, `gold_label`, `raw_model_output`, `parsed_prediction`, `correct`, `valid_format`, `inference_latency_seconds`, `inference_error`, `ollama_response_metadata` |
| `metrics.json` | `total_cases`, `number_correct`, `overall_accuracy`, `class_accuracy`, `prediction_counts`, `invalid_format_*`, `inference_failure_count`, `confusion_matrix` |
| `confusion_matrix.csv` | gold label by predicted label (`SUPPORTED`, `REFUTED`, `NOT ENOUGH INFORMATION`, `INVALID`) |
| `errors.csv` | incorrect cases only: `fever_id, claim, gold_label, parsed_prediction, valid_format, inference_error, raw_model_output` |

### `fever/<model>/consistency_5run/`

| File | Contents |
|---|---|
| `runs/run_01…05_predictions.jsonl` | the five full prediction files, same schema as the baseline |
| `consistency_by_case.csv` | per claim: five predictions, `unique_label_count`, `modal_label`, `modal_agreement`, `exact_agreement`, `consistency_classification` (`STABLE_CORRECT` / `STABLE_WRONG` / `UNSTABLE` / `INCOMPLETE`) |
| `run_accuracy.csv` | `run, seed, correct, total, fever_accuracy` |
| `consistency_metrics.json` | exact consistency, modal agreement, stability classes, consensus, NEI behaviour, accuracy summary |

### `fever/<model>/robustness/`

| File | Contents |
|---|---|
| `prepared/` | the perturbed inputs (`control`, `reordered_evidence`, `distractor_evidence`) and `distractor_audit.md` for human review |
| `predictions/{control,reorder,distractor}_predictions.jsonl` | predictions per variant |
| `robustness_by_case.csv` | per claim: control, reorder and distractor predictions, whether each changed, outcome class |
| `robustness_metrics.json` | per variant: invariance, paired accuracy and delta, outcomes, label transitions, NEI analysis, technical validity |

### `vitaminc/<model>/sensitivity/`

| File | Contents |
|---|---|
| `prepared/` | `selected_pairs.jsonl` and `pair_audit.md` for human review |
| `predictions/predictions.jsonl` | 200 rows; `pair_id` and `side` (`support` / `refute`) identify each |
| `sensitivity_by_pair.csv` | per pair: both predictions, `prediction_changed`, `correct_sensitivity`, `outcome` |
| `sensitivity_metrics.json` | correct-flip rate, flip rate, pair outcomes, per-side accuracy, NEI analysis, technical validity |

### `run_config.json` (every experiment)

Prompt (system and user template), model tag, temperature, seed(s), dataset path and hash, Ollama
version, model metadata including its digest, Python version, UTC timestamp and git commit. This is
what makes a result auditable: two runs are comparable only if prompt, seeds and dataset hash
match.

## Rules

1. **Never edit results by hand.** They are produced by the runner. To change a report or metric,
   change the code and regenerate (`evaluate` re-scores saved predictions without a model).
2. **Do not overwrite.** The CLI refuses to replace existing results unless `--overwrite` is
   passed. Golden tests pin the committed pilot numbers; if one fails, a metric changed.
3. **Commit your own model's folders only.** Each owner commits their own `<model_slug>`
   directories, so branches do not collide.
4. **Record the model digest.** The same tag can point to different builds; `run_config.json`
   stores the digest, so keep it.
5. **Large outputs.** The frozen samples produce about 1 to 2 MB per model, which belongs in git. If
   a run produces files over roughly 10 MB (for example a full-dataset `predictions.jsonl`), commit
   the metrics and per-case CSVs and share the large file by link, noting the link in the folder.

## Reading results in code

Use the committed files directly:

```python
import json
from pathlib import Path

metrics = json.loads(Path("results/fever/llama3.2_3b/baseline/metrics.json").read_text())
print(metrics["overall_accuracy"])
```

A shared loader that discovers every `<dataset>/<model>/<experiment>` folder is planned for the
comparison page and the Streamlit app.
