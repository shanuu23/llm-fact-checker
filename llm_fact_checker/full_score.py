"""Score the frozen FEVER and VitaminC full runs without calling a model."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from math import sqrt
from pathlib import Path
from typing import Any

from llm_fact_checker.common import file_sha256, read_jsonl
from llm_fact_checker.evaluation import GOLD_LABELS, calculate_metrics
from llm_fact_checker.full_run import (
    PAIR_LABELS,
    _digest,
    _saved_predictions,
    fever_cases,
    vitaminc_cases,
)
from llm_fact_checker.prompts import parse_prediction

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"llama3.2:3b": "llama3.2_3b", "qwen2.5:7b": "qwen2.5_7b"}
REFUTED_TYPOS = {"REFUTATED", "REFUTTED", "REFUTUTED", "REFUTUED"}


def typo_tolerant_label(row: dict[str, Any]) -> str | None:
    """Post-hoc interpretation of four unambiguous REFUTED misspellings."""
    if row["inference_error"] is not None:
        return None
    raw = row.get("raw_model_output")
    if raw is not None and raw.strip().upper() in REFUTED_TYPOS:
        return "REFUTED"
    return row["parsed_prediction"]


def rate(count: int, total: int) -> dict[str, Any]:
    """A proportion with a descriptive 95% Wilson interval."""
    if not 0 <= count <= total:
        raise ValueError(f"Invalid proportion: {count}/{total}")
    if not total:
        return {"count": count, "total": total, "rate": None, "wilson_95": None}
    z = 1.959963984540054
    p = count / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half_width = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {
        "count": count,
        "total": total,
        "rate": p,
        "wilson_95": [max(0.0, centre - half_width), min(1.0, centre + half_width)],
    }


def score_fever(predictions: list[dict[str, Any]], pilot_ids: set[int]) -> dict[str, Any]:
    metrics = calculate_metrics(predictions)
    ids = {row["fever_id"] for row in predictions}
    if len(ids) != len(predictions) or not pilot_ids <= ids:
        raise ValueError("FEVER IDs are duplicated or pilot IDs are missing")
    unflagged = [row for row in predictions if not row["audit_flags"]]
    non_pilot = [row for row in predictions if row["fever_id"] not in pilot_ids]
    return {
        "accuracy": rate(metrics["number_correct"], metrics["total_cases"]),
        "typo_tolerant_accuracy": rate(
            sum(typo_tolerant_label(row) == row["gold_label"] for row in predictions),
            len(predictions),
        ),
        "by_gold_label": {
            label: rate(
                sum(row["correct"] for row in predictions if row["gold_label"] == label),
                sum(row["gold_label"] == label for row in predictions),
            )
            for label in GOLD_LABELS
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "prediction_counts": metrics["prediction_counts"],
        "invalid_format_count": metrics["invalid_format_count"],
        "inference_failure_count": metrics["inference_failure_count"],
        "audit_flag_counts": dict(
            Counter(flag for row in predictions for flag in row["audit_flags"])
        ),
        "without_flagged": rate(sum(row["correct"] for row in unflagged), len(unflagged)),
        "pilot_overlap_count": len(pilot_ids),
        "without_pilot_ids": rate(sum(row["correct"] for row in non_pilot), len(non_pilot)),
    }


def vitamin_pairs(
    predictions: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if len(predictions) % 2:
        raise ValueError("VitaminC has an odd number of predictions")
    pairs = []
    for index in range(0, len(predictions), 2):
        left, right = predictions[index : index + 2]
        pair_type = left["pair_type"]
        if (
            pair_type not in PAIR_LABELS
            or left["side"] != "left"
            or right["side"] != "right"
            or left["pair_id"] != right["pair_id"]
            or right["pair_type"] != pair_type
            or (left["gold_label"], right["gold_label"]) != PAIR_LABELS[pair_type]
            or left["claim"] != right["claim"]
            or left["audit_flags"] != right["audit_flags"]
            or left["revision_type"] != right["revision_type"]
        ):
            raise ValueError(f"Invalid VitaminC pair at prediction {index + 1}")
        pairs.append((left, right))
    if len({left["pair_id"] for left, _ in pairs}) != len(pairs):
        raise ValueError("Duplicate VitaminC pair ID")
    return pairs


def _pair_metrics(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    valid = [
        (left, right)
        for left, right in pairs
        if left["parsed_prediction"] is not None and right["parsed_prediction"] is not None
    ]
    return {
        "both_sides_correct": rate(
            sum(left["correct"] and right["correct"] for left, right in pairs), len(pairs)
        ),
        "typo_tolerant_both_sides_correct": rate(
            sum(
                typo_tolerant_label(left) == left["gold_label"]
                and typo_tolerant_label(right) == right["gold_label"]
                for left, right in pairs
            ),
            len(pairs),
        ),
        "left_accuracy": rate(sum(left["correct"] for left, _ in pairs), len(pairs)),
        "right_accuracy": rate(sum(right["correct"] for _, right in pairs), len(pairs)),
        "valid_pair_count": len(valid),
        "prediction_change_among_valid": rate(
            sum(left["parsed_prediction"] != right["parsed_prediction"] for left, right in valid),
            len(valid),
        ),
        "invalid_format_sides": sum(
            row["inference_error"] is None and not row["valid_format"]
            for pair in pairs
            for row in pair
        ),
        "inference_failure_sides": sum(
            row["inference_error"] is not None for pair in pairs for row in pair
        ),
    }


def score_vitaminc(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = vitamin_pairs(predictions)
    result = {}
    for pair_type in PAIR_LABELS:
        group = [pair for pair in pairs if pair[0]["pair_type"] == pair_type]
        unflagged = [pair for pair in group if not pair[0]["audit_flags"]]
        metrics = _pair_metrics(group)
        metrics["without_flagged"] = _pair_metrics(unflagged)["both_sides_correct"]
        metrics["audit_flag_counts"] = dict(
            Counter(flag for left, _ in group for flag in left["audit_flags"])
        )
        metrics["by_source"] = {
            source: _pair_metrics([pair for pair in group if pair[0]["revision_type"] == source])[
                "both_sides_correct"
            ]
            for source in sorted({left["revision_type"] for left, _ in group})
        }
        result[pair_type] = metrics
    return result


def paired_outcomes(first: list[bool], second: list[bool]) -> dict[str, int]:
    if len(first) != len(second):
        raise ValueError("Models have different numbers of evaluation units")
    outcomes = Counter(zip(first, second, strict=True))
    return {
        "both_correct": outcomes[True, True],
        "llama_only_correct": outcomes[True, False],
        "qwen_only_correct": outcomes[False, True],
        "both_wrong": outcomes[False, False],
    }


def _load_run(
    directory: Path, model: str, cases: list[dict[str, Any]], sources: dict[str, Path]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    contract = config["contract"]
    if (
        contract["model"] != model
        or contract["case_count"] != len(cases)
        or contract["cases_sha256"] != _digest(cases)
        or contract["source_hashes"] != {key: file_sha256(path) for key, path in sources.items()}
    ):
        raise ValueError(f"Run configuration does not match frozen inputs: {directory}")
    predictions = _saved_predictions(directory / "predictions.jsonl", cases)
    if len(predictions) != len(cases):
        raise ValueError(f"Incomplete run: {directory} ({len(predictions)}/{len(cases)})")
    for index, row in enumerate(predictions, 1):
        parsed = (
            parse_prediction(row["raw_model_output"])
            if row["raw_model_output"] is not None
            else None
        )
        if (
            row["parsed_prediction"] != parsed
            or row["valid_format"] != (parsed is not None)
            or row["correct"] != (parsed == row["gold_label"])
        ):
            raise ValueError(f"Inconsistent saved prediction at {directory}, line {index}")
    return config, predictions


def score_full_runs(root: Path = ROOT) -> dict[str, Any]:
    fever_input = root / "data/fever/processed/fever_gold_full_13332.jsonl"
    vitamin_sources = {
        "rows": root / "data/vitaminc/processed/train_full/rows.jsonl",
        "pairs": root / "data/vitaminc/processed/train_full/pairs.jsonl",
        "selected_pairs": root / "data/vitaminc/processed/vitaminc_train_eval_pair_ids.jsonl",
    }
    inputs = {
        "fever": ({"prepared": fever_input}, fever_cases(fever_input)),
        "vitaminc": (
            vitamin_sources,
            vitaminc_cases(
                vitamin_sources["rows"],
                vitamin_sources["pairs"],
                vitamin_sources["selected_pairs"],
            ),
        ),
    }
    pilot_path = root / "data/fever/processed/fever_baseline_150.jsonl"
    pilot_ids = {row["fever_id"] for row in read_jsonl(pilot_path)}
    if len(pilot_ids) != 150:
        raise ValueError("Expected 150 unique FEVER pilot IDs")
    output: dict[str, Any] = {
        "schema_version": 1,
        "posthoc_refuted_spellings": sorted(REFUTED_TYPOS),
        "input_sha256": {
            dataset: {key: file_sha256(path) for key, path in sources.items()}
            for dataset, (sources, _) in inputs.items()
        },
        "fever_pilot_sha256": file_sha256(pilot_path),
        "models": {},
        "paired_comparison": {},
    }
    saved: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for dataset, (sources, cases) in inputs.items():
        shared_contract = None
        saved[dataset] = {}
        for model, folder in MODELS.items():
            experiment = "full_gold_13332" if dataset == "fever" else "full_eval_2102_pairs"
            directory = root / "results" / dataset / folder / experiment
            config, predictions = _load_run(directory, model, cases, sources)
            contract = {key: value for key, value in config["contract"].items() if key != "model"}
            if shared_contract is not None and contract != shared_contract:
                raise ValueError(f"Models did not use the same {dataset} inputs and settings")
            shared_contract = contract
            saved[dataset][model] = predictions
            metrics = (
                score_fever(predictions, pilot_ids)
                if dataset == "fever"
                else score_vitaminc(predictions)
            )
            model_output = output["models"].setdefault(
                model, {"model_digest": config["model_digest"], "predictions_sha256": {}}
            )
            model_output["predictions_sha256"][dataset] = file_sha256(
                directory / "predictions.jsonl"
            )
            model_output[dataset] = metrics
        llama = saved[dataset]["llama3.2:3b"]
        qwen = saved[dataset]["qwen2.5:7b"]
        if dataset == "fever":
            output["paired_comparison"][dataset] = paired_outcomes(
                [row["correct"] for row in llama], [row["correct"] for row in qwen]
            )
        else:
            llama_pairs, qwen_pairs = vitamin_pairs(llama), vitamin_pairs(qwen)
            if [left["pair_id"] for left, _ in llama_pairs] != [
                left["pair_id"] for left, _ in qwen_pairs
            ]:
                raise ValueError("Models have different VitaminC pair order")
            output["paired_comparison"][dataset] = {
                pair_type: paired_outcomes(
                    [
                        left["correct"] and right["correct"]
                        for left, right in llama_pairs
                        if left["pair_type"] == pair_type
                    ],
                    [
                        left["correct"] and right["correct"]
                        for left, right in qwen_pairs
                        if left["pair_type"] == pair_type
                    ],
                )
                for pair_type in PAIR_LABELS
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/full_evaluation_summary.json"
    )
    args = parser.parse_args()
    summary = score_full_runs()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Scored four completed runs: {args.output}")


if __name__ == "__main__":
    main()
