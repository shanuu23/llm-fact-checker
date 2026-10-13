"""Metrics and research-artifact generation for the FEVER baseline."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any

from llm_fact_checker.common import read_jsonl, write_text_atomic
from llm_fact_checker.prompts import LABELS, SYSTEM_PROMPT

GOLD_LABELS = ("SUPPORTED", "REFUTED")
# A whole-dataset run has thousands of errors; the table lists this many, errors.csv has all.
MAX_LISTED_ERRORS = 200
PREDICTION_LABELS = (*LABELS, "INVALID")


def calculate_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = {gold: {predicted: 0 for predicted in PREDICTION_LABELS} for gold in GOLD_LABELS}
    prediction_counts = Counter()
    class_totals = Counter()
    class_correct = Counter()
    correct = 0
    formatting_failures = 0
    inference_failures = 0
    for item in predictions:
        gold = item["gold_label"]
        predicted = item.get("parsed_prediction") or "INVALID"
        if gold not in GOLD_LABELS:
            raise ValueError(f"Unexpected gold label in baseline: {gold!r}")
        if predicted not in PREDICTION_LABELS:
            raise ValueError(f"Unexpected prediction: {predicted!r}")
        confusion[gold][predicted] += 1
        prediction_counts[predicted] += 1
        class_totals[gold] += 1
        is_correct = predicted == gold
        correct += is_correct
        class_correct[gold] += is_correct
        inference_failures += bool(item.get("inference_error"))
        formatting_failures += not item.get("inference_error") and not item.get("valid_format")
    total = len(predictions)
    return {
        "total_cases": total,
        "number_correct": correct,
        "overall_accuracy": correct / total if total else 0.0,
        "class_accuracy": {
            label: class_correct[label] / class_totals[label] if class_totals[label] else None
            for label in GOLD_LABELS
        },
        "prediction_counts": {label: prediction_counts[label] for label in PREDICTION_LABELS},
        "invalid_format_count": formatting_failures,
        "invalid_format_rate": formatting_failures / total if total else 0.0,
        "inference_failure_count": inference_failures,
        "confusion_matrix": confusion,
    }


def _write_confusion(path: Path, confusion: dict[str, dict[str, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold_label", *PREDICTION_LABELS])
        for gold in GOLD_LABELS:
            writer.writerow([gold, *(confusion[gold][label] for label in PREDICTION_LABELS)])


def _write_errors(path: Path, predictions: list[dict[str, Any]]) -> None:
    fields = [
        "fever_id",
        "claim",
        "gold_label",
        "parsed_prediction",
        "valid_format",
        "inference_error",
        "raw_model_output",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for item in predictions:
            if not item["correct"]:
                writer.writerow(
                    {**item, "parsed_prediction": item.get("parsed_prediction") or "INVALID"}
                )


def _report(
    metrics: dict[str, Any], predictions: list[dict[str, Any]], config: dict[str, Any]
) -> str:
    confusion = metrics["confusion_matrix"]
    supported_accuracy = metrics["class_accuracy"]["SUPPORTED"]
    refuted_accuracy = metrics["class_accuracy"]["REFUTED"]
    supported_text = "N/A" if supported_accuracy is None else f"{supported_accuracy:.3f}"
    refuted_text = "N/A" if refuted_accuracy is None else f"{refuted_accuracy:.3f}"
    balance_text = ", ".join(
        f"{count} {label}" for label, count in config["class_balance"].items()
    )
    count_text = ", ".join(
        f"{label}: {count}" for label, count in metrics["prediction_counts"].items()
    )
    incorrect = [item for item in predictions if not item["correct"]]
    incorrect_rows = (
        "\n".join(
            f"| {item['fever_id']} | {item['gold_label']} | "
            f"{item.get('parsed_prediction') or 'INVALID'} | "
            f"{str(item['claim']).replace('|', chr(92) + '|')} |"
            for item in incorrect[:MAX_LISTED_ERRORS]
        )
        or "| — | — | — | None |"
    )
    if len(incorrect) > MAX_LISTED_ERRORS:
        incorrect_rows += (
            f"\n\n*{len(incorrect) - MAX_LISTED_ERRORS} more incorrect cases are not listed here; "
            "all are in `errors.csv`.*"
        )
    return f"""# FEVER gold-evidence baseline: {config["ollama_model"]}

## Purpose

This baseline tests whether `{config["ollama_model"]}` classifies FEVER SUPPORTS and REFUTES claims when supplied with one complete gold Wikipedia evidence set. It uses only the supplied evidence in the prompt.

## Dataset and sample construction

- Source: `{config["raw_dataset_path"]}`
- Prepared data: `{config["prepared_dataset_path"]}`
- Cases: {metrics["total_cases"]} ({balance_text})
- Sampling seed: {config["sampling_seed"]}
- Construction: claims are ID/claim-deduplicated, deterministically shuffled per class, and assigned the valid gold evidence set with the fewest unique sentences (stable page/sentence tie-break).
- Extraction: required sentence references are resolved by streaming `wiki-*.jsonl`; complete articles are not retained.

## Model configuration

- Model: `{config["ollama_model"]}`
- Temperature: {config["temperature"]}
- Model seed: {config["model_seed"]}
- Execution: sequential, non-streaming requests through the local Ollama HTTP API

## Prompt

```text
{SYSTEM_PROMPT}
```

The user message contains the claim and numbered evidence sentences. Gold labels are not included.

## Results

- Correct: {metrics["number_correct"]} / {metrics["total_cases"]}
- Overall accuracy: {metrics["overall_accuracy"]:.3f}
- SUPPORTED accuracy: {supported_text}
- REFUTED accuracy: {refuted_text}
- Invalid-format outputs: {metrics["invalid_format_count"]} ({metrics["invalid_format_rate"]:.3%})
- Inference failures: {metrics["inference_failure_count"]}
- Prediction counts: {count_text}

## Confusion matrix

| Gold | SUPPORTED | REFUTED | NOT ENOUGH INFORMATION | INVALID |
|---|---:|---:|---:|---:|
| SUPPORTED | {confusion["SUPPORTED"]["SUPPORTED"]} | {confusion["SUPPORTED"]["REFUTED"]} | {confusion["SUPPORTED"]["NOT ENOUGH INFORMATION"]} | {confusion["SUPPORTED"]["INVALID"]} |
| REFUTED | {confusion["REFUTED"]["SUPPORTED"]} | {confusion["REFUTED"]["REFUTED"]} | {confusion["REFUTED"]["NOT ENOUGH INFORMATION"]} | {confusion["REFUTED"]["INVALID"]} |

## Incorrect cases

| FEVER ID | Gold | Prediction | Claim |
|---:|---|---|---|
{incorrect_rows}

Full raw outputs and error details are preserved in `predictions.jsonl` and `errors.csv`.

## Limitations

The gold sample contains no NOT ENOUGH INFO cases, so this run does not measure gold NEI accuracy. It does not test evidence retrieval, robustness, stability, sensitivity, calibration, or real-world fact-checking ability. It measures only classification of FEVER SUPPORTS/REFUTES cases supplied with valid gold evidence.
"""


def _flagged_section(metrics: dict[str, Any], excluded: dict[str, Any]) -> str:
    def accuracy(block: dict[str, Any]) -> str:
        return "N/A" if not block["total_cases"] else f"{block['overall_accuracy']:.3f}"

    def class_accuracy(block: dict[str, Any], label: str) -> str:
        value = block["class_accuracy"][label]
        return "N/A" if value is None else f"{value:.3f}"

    flag_text = ", ".join(f"{name}: {count}" for name, count in excluded["flag_counts"].items())
    return f"""
## Result excluding flagged records

The FEVER data audit flagged {excluded["excluded_flagged_cases"]} of these claims as possibly
inconsistent ({flag_text}). The main result above keeps them. This section removes them, as the
audit decided. Flags are possible label inconsistencies, not proven errors, and no label was
changed.

| | All cases | Excluding flagged |
|---|---:|---:|
| Cases | {metrics["total_cases"]} | {excluded["total_cases"]} |
| Correct | {metrics["number_correct"]} | {excluded["number_correct"]} |
| Overall accuracy | {accuracy(metrics)} | {accuracy(excluded)} |
| SUPPORTED accuracy | {class_accuracy(metrics, "SUPPORTED")} | {class_accuracy(excluded, "SUPPORTED")} |
| REFUTED accuracy | {class_accuracy(metrics, "REFUTED")} | {class_accuracy(excluded, "REFUTED")} |
"""


def evaluate_results(results_dir: Path) -> dict[str, Any]:
    predictions = list(read_jsonl(results_dir / "predictions.jsonl"))
    config = json.loads((results_dir / "run_config.json").read_text(encoding="utf-8"))
    metrics = calculate_metrics(predictions)
    _write_confusion(results_dir / "confusion_matrix.csv", metrics["confusion_matrix"])
    _write_errors(results_dir / "errors.csv", predictions)
    report = _report(metrics, predictions, config)
    flagged = [item for item in predictions if item.get("flags")]
    excluded_path = results_dir / "metrics_excluding_flagged.json"
    if flagged:
        excluded = calculate_metrics([item for item in predictions if not item.get("flags")])
        excluded["excluded_flagged_cases"] = len(flagged)
        excluded["flag_counts"] = dict(Counter(f for item in flagged for f in item["flags"]))
        write_text_atomic(excluded_path, json.dumps(excluded, indent=2, ensure_ascii=False) + "\n")
        report += _flagged_section(metrics, excluded)
    elif excluded_path.exists():
        excluded_path.unlink()  # a re-score of unflagged predictions must not leave a stale file
    write_text_atomic(results_dir / "report.md", report)
    # last: other code treats this file's existence as "this run is complete"
    write_text_atomic(
        results_dir / "metrics.json", json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    )
    return metrics
