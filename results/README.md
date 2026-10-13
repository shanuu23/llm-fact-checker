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

The folder is derived from `--model`, so each model gets its own tree and nothing is shared
between models. The perturbed inputs and audit files that robustness and sensitivity read are
derived from the frozen samples, not from a model, so they live in `data/fever/prepared/` and
`data/vitaminc/prepared/` rather than here.

## Files per experiment

Each experiment folder contains a `report.md` for people, a `run_config.json` for audit, and the
machine-readable files below.

### `fever/<model>/baseline/`

| File | Contents |
|---|---|
| `predictions.jsonl` | one row per case: `fever_id`, `claim`, `evidence`, `gold_label`, `raw_model_output`, `parsed_prediction`, `correct`, `valid_format`, `inference_latency_seconds`, `inference_error`, `ollama_response_metadata` |
| `metrics.json` | `total_cases`, `number_correct`, `overall_accuracy`, `class_accuracy`, `prediction_counts`, `invalid_format_*`, `inference_failure_count`, `confusion_matrix` |
| `confusion_matrix.csv` | gold label by predicted label (`SUPPORTED`, `REFUTED`, `NOT ENOUGH INFORMATION`, `INVALID`) |
| `metrics_excluding_flagged.json` | only for whole-dataset runs: the same metrics without audit-flagged claims, plus `excluded_flagged_cases` and `flag_counts` |
| `predictions.partial.jsonl` | only while a run is in progress or was interrupted; `--resume` continues from it and it is deleted when the run completes |
| `errors.csv` | incorrect cases only: `fever_id, claim, gold_label, parsed_prediction, valid_format, inference_error, raw_model_output` |

Whole-dataset runs are stored next to the 150-case ones as `fever/<model>/baseline_full/`,
`robustness_full/` and `consistency_5run_full/` (see the README, "Run the whole FEVER dev set").
Each `run_config.json` records the dataset hash, so a comparison can tell them apart from the
150-case runs. Their large per-case files (`predictions.jsonl`, `predictions/`, `runs/`) are
git-ignored; the metrics, reports and configs are committed.

Whole-set runs also write the second result without the audit-flagged claims:
`metrics_excluding_flagged.json` (baseline), `robustness_metrics_excluding_flagged.json` and
`consistency_metrics_excluding_flagged.json`.

**Crash safety.** Every result file is written atomically (a crash leaves the old file or the
complete new one, never half), answers are saved as they arrive in `*.partial.jsonl`, and the
metrics file is written last because its existence is what marks a run as finished. `--resume`
relies on all three.

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
| `predictions/{control,reorder,distractor}_predictions.jsonl` | predictions per variant |
| `robustness_by_case.csv` | per claim: control, reorder and distractor predictions, whether each changed, outcome class |
| `robustness_metrics.json` | per variant: invariance, paired accuracy and delta, outcomes, label transitions, NEI analysis, technical validity |

### `vitaminc/<model>/sensitivity/`

| File | Contents |
|---|---|
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

`llm_fact_checker.results` is a read-only loader (standard library only; it never writes and never
calls a model). Use it instead of re-implementing paths and file names:

```python
from llm_fact_checker.results import discover_runs, select

runs = discover_runs()                          # every experiment under results/, smoke runs excluded
for run in select(runs, kind="baseline"):       # or filter by dataset= / model=
    print(run.model, run.headline()["accuracy"])

run = select(runs, model="llama3.2_3b", kind="robustness")[0]
run.metrics()       # the full metrics JSON as a dict
run.config()        # run_config.json
run.provenance()    # model tag and digest, seeds, dataset hash, git commit, timestamp
run.cases()         # per-case rows as dicts: pandas.DataFrame(run.cases()) if pandas is installed
```

`headline()` returns a small set of comparable numbers with the same names for every model:

| Kind | Keys |
|---|---|
| `baseline` | `cases`, `accuracy`, `supported_accuracy`, `refuted_accuracy`, `nei_rate`, `invalid_format_rate` |
| `consistency` | `cases`, `exact_consistency`, `mean_modal_agreement`, `mean_accuracy`, `stable_wrong_rate`, `unstable_rate` |
| `robustness` | `cases`, `control_accuracy`, `reorder_eligible`, `reorder_invariance`, `distractor_invariance`, `distractor_accuracy`, `distractor_accuracy_delta` |
| `sensitivity` | `pairs`, `correct_sensitivity`, `flip_rate`, `support_accuracy`, `refute_accuracy`, `nei_pair_rate` |

Per-case CSV values are typed for the columns above (counts as `int`, rates as `float`, flags as
`bool`, an empty cell as `None`); other columns, such as claim text, stay strings.
