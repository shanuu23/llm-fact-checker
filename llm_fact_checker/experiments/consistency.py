"""Five-run repeatability experiment for the fixed FEVER baseline dataset."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import statistics
import sys
from typing import Any

from llm_fact_checker.common import (
    file_sha256,
    git_commit,
    read_jsonl,
    write_csv,
    write_jsonl,
    write_text_atomic,
)
from llm_fact_checker.data_prep.fever import PREPARED_PATH
from llm_fact_checker.evaluation import calculate_metrics
from llm_fact_checker.models import DEFAULT_MODEL
from llm_fact_checker.ollama import OllamaClient, infer_cases_resumable
from llm_fact_checker.prompts import LABELS, SYSTEM_PROMPT

EXPECTED_DATASET_SHA256 = "6bc7cf910cb0b31186e5fc397ebc7cefab2de1ee65d29ae601e369d8758344cb"
TEMPERATURE = 0.2
RUN_SEEDS = (101, 202, 303, 404, 505)


def validate_prepared_dataset(
    path: Path, frozen_path: Path = PREPARED_PATH
) -> tuple[list[dict[str, Any]], str]:
    """Load a prepared FEVER file and check it can be scored.

    The frozen 150-case sample is protected: the file at ``frozen_path`` must match the pilot's
    hash, and any file with that hash must have exactly 150 cases (75/75). Any other file, for
    example the whole dev set, only has to be a well-formed set of cases; its hash is returned
    so results can be tied to it.
    """
    actual_hash = file_sha256(path)
    if path.resolve() == frozen_path.resolve() and actual_hash != EXPECTED_DATASET_SHA256:
        raise ValueError(
            f"Prepared dataset SHA-256 mismatch: expected {EXPECTED_DATASET_SHA256}, "
            f"found {actual_hash}. Refusing to run the consistency experiment."
        )
    records = list(read_jsonl(path))
    if not records:
        raise ValueError(f"Prepared dataset is empty: {path}")
    if len({record["fever_id"] for record in records}) != len(records):
        raise ValueError("Prepared dataset contains duplicate FEVER IDs")
    unscorable = {record["gold_label"] for record in records} - {"SUPPORTED", "REFUTED"}
    if unscorable:
        raise ValueError(f"Prepared dataset has labels this experiment cannot score: {unscorable}")
    if actual_hash == EXPECTED_DATASET_SHA256:
        counts = Counter(record["gold_label"] for record in records)
        if len(records) != 150 or counts != {"SUPPORTED": 75, "REFUTED": 75}:
            raise ValueError(
                f"Prepared dataset must contain 150 cases (75/75); found {dict(counts)}"
            )
    return records, actual_hash


def case_consistency(predictions: list[str | None], gold_label: str) -> dict[str, Any]:
    if len(predictions) != 5:
        raise ValueError(f"Expected five predictions, found {len(predictions)}")
    complete = all(prediction in LABELS for prediction in predictions)
    counts = Counter(prediction for prediction in predictions if prediction in LABELS)
    modal_count = max(counts.values(), default=0)
    modes = sorted(label for label, count in counts.items() if count == modal_count)
    tie = complete and len(modes) > 1
    modal_label = modes[0] if complete and len(modes) == 1 else None
    exact_agreement = complete and len(counts) == 1
    if not complete:
        classification = "INCOMPLETE"
    elif exact_agreement and modal_label == gold_label:
        classification = "STABLE_CORRECT"
    elif exact_agreement:
        classification = "STABLE_WRONG"
    else:
        classification = "UNSTABLE"
    return {
        "unique_label_count": len(counts),
        "modal_label": modal_label,
        "modal_count": modal_count if complete else None,
        "modal_agreement": modal_count / 5 if complete else None,
        "exact_agreement": exact_agreement,
        "tie": tie,
        "incomplete": not complete,
        "consensus_prediction": "TIE" if tie else modal_label,
        "consistency_classification": classification,
    }


def build_case_rows(runs: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if len(runs) != 5 or not runs:
        raise ValueError("Consistency analysis requires exactly five runs")
    expected_ids = [item["fever_id"] for item in runs[0]]
    if any([item["fever_id"] for item in run] != expected_ids for run in runs[1:]):
        raise ValueError("Run files do not contain identical FEVER cases in identical order")
    rows = []
    for items in zip(*runs, strict=True):
        predictions = [item["parsed_prediction"] for item in items]
        display_predictions = [
            item["parsed_prediction"]
            or ("INFERENCE_FAILURE" if item["inference_error"] else "INVALID")
            for item in items
        ]
        summary = case_consistency(predictions, items[0]["gold_label"])
        rows.append(
            {
                "fever_id": items[0]["fever_id"],
                "claim": items[0]["claim"],
                "gold_label": items[0]["gold_label"],
                **{
                    f"run_{index}_prediction": prediction
                    for index, prediction in enumerate(display_predictions, 1)
                },
                **summary,
            }
        )
    return rows


def calculate_consistency_metrics(
    case_rows: list[dict[str, Any]],
    runs: list[list[dict[str, Any]]],
    run_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    total_cases = len(case_rows)
    complete = [row for row in case_rows if not row["incomplete"]]
    exact_count = sum(row["exact_agreement"] for row in case_rows)
    modal_agreements = [row["modal_agreement"] for row in complete]
    distribution = {
        f"{count}_of_5": sum(row["modal_count"] == count for row in complete)
        for count in (5, 4, 3, 2)
    }
    unique_distribution = {
        str(count): sum(row["unique_label_count"] == count for row in complete)
        for count in (1, 2, 3)
    }
    classifications = Counter(row["consistency_classification"] for row in case_rows)
    all_predictions = [item for run in runs for item in run]
    successful = sum(not item["inference_error"] for item in all_predictions)
    failures = sum(bool(item["inference_error"]) for item in all_predictions)
    invalid = sum(
        not item["inference_error"] and not item["valid_format"] for item in all_predictions
    )
    nei_by_case = [
        [item["parsed_prediction"] for item in items] for items in zip(*runs, strict=True)
    ]
    accuracies = [metrics["overall_accuracy"] for metrics in run_metrics]
    consensus_eligible = [
        row for row in complete if not row["tie"] and row["consensus_prediction"] is not None
    ]
    consensus_correct = sum(
        row["consensus_prediction"] == row["gold_label"] for row in consensus_eligible
    )
    return {
        "total_cases": total_cases,
        "run_count": len(runs),
        "expected_inference_attempts": total_cases * len(runs),
        "successful_inference_requests": successful,
        "inference_failure_count": failures,
        "invalid_format_count": invalid,
        "complete_case_count": len(complete),
        "incomplete_case_count": total_cases - len(complete),
        "exact_consistency_count": exact_count,
        "exact_consistency_rate": exact_count / total_cases if total_cases else 0.0,
        "mean_modal_agreement": statistics.fmean(modal_agreements) if modal_agreements else None,
        "agreement_distribution": distribution,
        "unique_label_count_distribution": unique_distribution,
        "tie_case_count": sum(row["tie"] for row in complete),
        "stability_classification": {
            label: {
                "count": classifications[label],
                "percentage": classifications[label] / total_cases if total_cases else 0.0,
            }
            for label in ("STABLE_CORRECT", "STABLE_WRONG", "UNSTABLE", "INCOMPLETE")
        },
        "consensus": {
            "eligible_non_tied_cases": len(consensus_eligible),
            "tie_count": sum(row["tie"] for row in complete),
            "correct_count": consensus_correct,
            "fever_accuracy": consensus_correct / len(consensus_eligible)
            if consensus_eligible
            else None,
        },
        "nei_behavior": {
            "cases_with_nei_at_least_once": sum(
                "NOT ENOUGH INFORMATION" in predictions for predictions in nei_by_case
            ),
            "cases_with_nei_all_five_runs": sum(
                predictions.count("NOT ENOUGH INFORMATION") == 5 for predictions in nei_by_case
            ),
            "cases_switching_between_nei_and_verdict": sum(
                "NOT ENOUGH INFORMATION" in predictions
                and any(label in predictions for label in ("SUPPORTED", "REFUTED"))
                for predictions in nei_by_case
            ),
            "total_nei_predictions": sum(
                item["parsed_prediction"] == "NOT ENOUGH INFORMATION" for item in all_predictions
            ),
        },
        "run_accuracy": [
            {
                "run": index,
                "seed": RUN_SEEDS[index - 1],
                "correct": metrics["number_correct"],
                "total": metrics["total_cases"],
                "fever_accuracy": metrics["overall_accuracy"],
            }
            for index, metrics in enumerate(run_metrics, 1)
        ],
        "accuracy_summary": {
            "mean": statistics.fmean(accuracies),
            "population_standard_deviation": statistics.pstdev(accuracies),
            "minimum": min(accuracies),
            "maximum": max(accuracies),
        },
    }


CASE_CSV_FIELDS = [
    "fever_id",
    "claim",
    "gold_label",
    *(f"run_{index}_prediction" for index in range(1, 6)),
    "unique_label_count",
    "modal_label",
    "modal_count",
    "modal_agreement",
    "exact_agreement",
    "tie",
    "incomplete",
    "consensus_prediction",
    "consistency_classification",
]
RUN_ACCURACY_FIELDS = ["run", "seed", "correct", "total", "fever_accuracy"]


def _write_case_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows, CASE_CSV_FIELDS)


def _write_run_accuracy(path: Path, metrics: dict[str, Any]) -> None:
    write_csv(path, metrics["run_accuracy"], RUN_ACCURACY_FIELDS)


def _report(metrics: dict[str, Any], config: dict[str, Any]) -> str:
    distribution = metrics["agreement_distribution"]
    states = metrics["stability_classification"]
    nei = metrics["nei_behavior"]
    mean_modal = metrics["mean_modal_agreement"]
    consensus_accuracy = metrics["consensus"]["fever_accuracy"]
    mean_modal_text = "N/A" if mean_modal is None else f"{mean_modal:.3%}"
    consensus_text = "N/A" if consensus_accuracy is None else f"{consensus_accuracy:.3%}"
    accuracy_rows = "\n".join(
        f"| {row['run']} | {row['seed']} | {row['correct']}/{row['total']} | "
        f"{row['fever_accuracy']:.3%} |"
        for row in metrics["run_accuracy"]
    )
    frozen = config["prepared_dataset_sha256"] == config["expected_dataset_sha256"]
    balance = " / ".join(
        f"{count} {label}" for label, count in config["full_dataset_class_balance"].items()
    )
    return f"""# FEVER five-run consistency experiment

## Purpose

This experiment measures how consistently `{config["ollama_model"]}` returns the same verdict when given the exact same claim and gold evidence five times. Consistency does not imply correctness.

## Dataset and model

- Prepared dataset: `{config["prepared_dataset_path"]}`
- {"Verified" if frozen else "Dataset"} SHA-256: `{config["prepared_dataset_sha256"]}`
- Cases: {config["sample_size"]} from the {"unchanged" if frozen else "prepared"} {balance} dataset
- Model: `{config["ollama_model"]}`
- Execution: sequential requests through the local Ollama HTTP API

## Generation settings

- Temperature: {config["temperature"]}
- Seeds: {config["run_seeds"]}
- Prompt and evidence formatting: identical to the baseline

Temperature 0.2 permits limited sampling variation, while fixed distinct seeds make the five repetitions controlled and reproducible. All other generation settings are identical.

## Primary stability results

- Exact 5/5 consistency: {metrics["exact_consistency_count"]}/{metrics["total_cases"]} ({metrics["exact_consistency_rate"]:.3%})
- Mean modal agreement: {mean_modal_text}
- Complete cases: {metrics["complete_case_count"]}
- Incomplete cases: {metrics["incomplete_case_count"]}

### Agreement distribution

| Maximum agreement | Cases |
|---|---:|
| 5/5 | {distribution["5_of_5"]} |
| 4/5 | {distribution["4_of_5"]} |
| 3/5 | {distribution["3_of_5"]} |
| 2/5 | {distribution["2_of_5"]} |

- One unique label: {metrics["unique_label_count_distribution"]["1"]}
- Two unique labels: {metrics["unique_label_count_distribution"]["2"]}
- Three unique labels: {metrics["unique_label_count_distribution"]["3"]}
- Modal ties: {metrics["tie_case_count"]}

## FEVER gold-label accuracy

| Run | Seed | Correct | FEVER accuracy |
|---:|---:|---:|---:|
{accuracy_rows}

- Mean: {metrics["accuracy_summary"]["mean"]:.3%}
- Population standard deviation: {metrics["accuracy_summary"]["population_standard_deviation"]:.3%}
- Minimum: {metrics["accuracy_summary"]["minimum"]:.3%}
- Maximum: {metrics["accuracy_summary"]["maximum"]:.3%}
- Non-tied consensus FEVER accuracy: {consensus_text} ({metrics["consensus"]["correct_count"]}/{metrics["consensus"]["eligible_non_tied_cases"]})
- Consensus ties: {metrics["consensus"]["tie_count"]}

FEVER accuracy means agreement with FEVER gold labels, not absolute real-world truth.

## Stable-correct versus stable-wrong

- Stable correct: {states["STABLE_CORRECT"]["count"]} ({states["STABLE_CORRECT"]["percentage"]:.3%})
- Stable wrong: {states["STABLE_WRONG"]["count"]} ({states["STABLE_WRONG"]["percentage"]:.3%})
- Unstable: {states["UNSTABLE"]["count"]} ({states["UNSTABLE"]["percentage"]:.3%})
- Incomplete: {states["INCOMPLETE"]["count"]} ({states["INCOMPLETE"]["percentage"]:.3%})

## NOT ENOUGH INFORMATION behaviour

- Cases producing NEI at least once: {nei["cases_with_nei_at_least_once"]}
- Cases producing NEI in all five runs: {nei["cases_with_nei_all_five_runs"]}
- Cases switching between NEI and SUPPORTED/REFUTED: {nei["cases_switching_between_nei_and_verdict"]}
- Total NEI predictions: {nei["total_nei_predictions"]} / {metrics["expected_inference_attempts"]}

## Technical validity

- Expected inference attempts: {metrics["expected_inference_attempts"]}
- Successful inference requests: {metrics["successful_inference_requests"]}
- Inference failures: {metrics["inference_failure_count"]}
- Invalid-format outputs: {metrics["invalid_format_count"]}

## Limitations

This experiment tests repeated-run stability only. It does not test prompt robustness, evidence perturbations, sensitivity to meaningful evidence changes, retrieval, or real-world fact-checking. Consistency does not imply correctness: a verdict can be repeated five times and still disagree with FEVER. FEVER also contains known annotation noise, so FEVER accuracy and repeatability are separate concepts.
"""


def _flagged_section(metrics: dict[str, Any], excluded: dict[str, Any]) -> str:
    def row(label: str, key: str, fmt: str) -> str:
        values = []
        for block in (metrics, excluded):
            value = block[key]
            values.append("N/A" if value is None else format(value, fmt))
        return f"| {label} | {values[0]} | {values[1]} |"

    return f"""
## Result excluding flagged records

The FEVER data audit flagged {excluded["excluded_flagged_cases"]} of these claims as possibly
inconsistent. The main result above keeps them; this section removes them, as the audit decided.
Flags are possible label inconsistencies, not proven errors.

| | All cases | Excluding flagged |
|---|---:|---:|
| Cases | {metrics["total_cases"]} | {excluded["total_cases"]} |
{row("Exact 5/5 consistency", "exact_consistency_rate", ".3%")}
{row("Mean modal agreement", "mean_modal_agreement", ".3%")}
| Mean FEVER accuracy across runs | {metrics["accuracy_summary"]["mean"]:.3%} | {excluded["accuracy_summary"]["mean"]:.3%} |
"""


def run_consistency_experiment(
    prepared_path: Path,
    results_dir: Path,
    limit: int | None = None,
    overwrite: bool = False,
    base_url: str = "http://localhost:11434",
    model: str = DEFAULT_MODEL,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the five-run experiment.

    Answers are saved as they arrive and each finished run is kept, so ``resume=True`` carries on
    an interrupted experiment (and does nothing if it already finished).
    """
    dataset, dataset_hash = validate_prepared_dataset(prepared_path)
    cases = dataset if limit is None else dataset[:limit]
    metrics_path = results_dir / "consistency_metrics.json"
    if resume and metrics_path.exists():
        print(f"Already complete, nothing to do: {results_dir}", file=sys.stderr)
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    if results_dir.exists() and any(results_dir.rglob("*")) and not (overwrite or resume):
        raise FileExistsError(
            f"Refusing to overwrite {results_dir}; pass --resume to continue an interrupted "
            "run or --overwrite to replace it"
        )

    client = OllamaClient(base_url=base_url)
    ollama_metadata = client.check(model)
    results_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = results_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    config = {
        "experiment": "FEVER five-run repeatability",
        "prepared_dataset_path": str(prepared_path),
        "prepared_dataset_sha256": dataset_hash,
        "expected_dataset_sha256": EXPECTED_DATASET_SHA256,
        "full_dataset_size": len(dataset),
        "full_dataset_class_balance": dict(Counter(row["gold_label"] for row in dataset)),
        "sample_size": len(cases),
        "ollama_model": model,
        "temperature": TEMPERATURE,
        "run_seeds": list(RUN_SEEDS),
        "prompt_template": {
            "system": SYSTEM_PROMPT,
            "user": "Claim:\n<claim>\n\nEvidence:\n<numbered evidence>",
        },
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "ollama_version": ollama_metadata["version"],
        "ollama_model_metadata": ollama_metadata["model"],
        "relevant_package_versions": {
            "llm_fact_checker": "0.0.1",
            "http_client": "urllib (standard library)",
        },
        "git_commit": git_commit(),
    }
    (results_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    runs = []
    run_metrics = []
    for run_number, seed in enumerate(RUN_SEEDS, 1):
        run_path = runs_dir / f"run_{run_number:02d}_predictions.jsonl"
        finished = list(read_jsonl(run_path)) if resume and run_path.exists() else None
        if finished is not None and len(finished) == len(cases):
            predictions = finished  # this run finished before the interruption
        else:
            partial_path = runs_dir / f"run_{run_number:02d}_predictions.partial.jsonl"
            predictions = infer_cases_resumable(
                client,
                model,
                cases,
                seed,
                TEMPERATURE,
                partial_path,
                resume=resume,
                progress_prefix=f"Run {run_number}/5",
            )
            write_jsonl(run_path, predictions)
            partial_path.unlink(missing_ok=True)
        runs.append(predictions)
        run_metrics.append(calculate_metrics(predictions))

    case_rows = build_case_rows(runs)
    metrics = calculate_consistency_metrics(case_rows, runs, run_metrics)
    _write_case_csv(results_dir / "consistency_by_case.csv", case_rows)
    _write_run_accuracy(results_dir / "run_accuracy.csv", metrics)
    report = _report(metrics, config)
    flagged_ids = {row["fever_id"] for row in runs[0] if row.get("flags")}
    excluded_path = results_dir / "consistency_metrics_excluding_flagged.json"
    if flagged_ids:
        kept = [[row for row in run if row["fever_id"] not in flagged_ids] for run in runs]
        excluded = calculate_consistency_metrics(
            build_case_rows(kept), kept, [calculate_metrics(run) for run in kept]
        )
        excluded["excluded_flagged_cases"] = len(flagged_ids)
        write_text_atomic(excluded_path, json.dumps(excluded, indent=2, ensure_ascii=False) + "\n")
        report += _flagged_section(metrics, excluded)
    elif excluded_path.exists():
        excluded_path.unlink()
    write_text_atomic(results_dir / "report.md", report)
    # last: its existence is what marks the experiment as finished
    write_text_atomic(metrics_path, json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    return metrics
