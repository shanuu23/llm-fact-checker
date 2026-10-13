"""Contrastive VitaminC sensitivity experiment for local Llama inference."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import sys
from typing import Any

from llm_fact_checker.common import (
    git_commit,
    percentage,
    read_jsonl,
    write_csv,
    write_jsonl,
    write_text_atomic,
)
from llm_fact_checker.data_prep.vitaminc import (
    EXPECTED_VALID_PAIRS,
    identify_contrastive_pairs,
    load_vitaminc_test,
    validate_pairs,
    validate_pairs_against_candidates,
)
from llm_fact_checker.models import DEFAULT_MODEL
from llm_fact_checker.ollama import OllamaClient, infer_cases_resumable
from llm_fact_checker.prompts import ALLOWED_LABELS, SYSTEM_PROMPT

PAIR_OUTCOMES = (
    "CORRECT_FLIP",
    "NO_CHANGE_CORRECT_ON_SUPPORT",
    "NO_CHANGE_CORRECT_ON_REFUTE",
    "NEI_INVOLVED",
    "WRONG_DIRECTION",
    "OTHER_ERROR",
)
TEMPERATURE = 0
SEED = 42


def classify_pair(support_prediction: str | None, refute_prediction: str | None) -> str:
    if support_prediction not in ALLOWED_LABELS or refute_prediction not in ALLOWED_LABELS:
        return "OTHER_ERROR"
    if "NOT ENOUGH INFORMATION" in {support_prediction, refute_prediction}:
        return "NEI_INVOLVED"
    if support_prediction == "SUPPORTED" and refute_prediction == "REFUTED":
        return "CORRECT_FLIP"
    if support_prediction == refute_prediction == "SUPPORTED":
        return "NO_CHANGE_CORRECT_ON_SUPPORT"
    if support_prediction == refute_prediction == "REFUTED":
        return "NO_CHANGE_CORRECT_ON_REFUTE"
    if support_prediction != refute_prediction:
        return "WRONG_DIRECTION"
    return "OTHER_ERROR"


def _inference_cases(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for pair in pairs:
        for side_name in ("support", "refute"):
            side = pair[side_name]
            metadata = side["source_metadata"]
            cases.append(
                {
                    "prediction_id": f"{pair['pair_id']}:{side_name}",
                    "pair_id": pair["pair_id"],
                    "side": side_name,
                    "claim": pair["claim"],
                    "evidence": [
                        {
                            "page": metadata["page"],
                            "sentence_id": metadata["unique_id"],
                            "text": side["evidence"],
                        }
                    ],
                    "gold_label": side["canonical_label"],
                }
            )
    return cases


def build_pair_rows(
    pairs: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_pair_side = {(item["pair_id"], item["side"]): item for item in predictions}
    rows = []
    for pair in pairs:
        support = by_pair_side[(pair["pair_id"], "support")]
        refute = by_pair_side[(pair["pair_id"], "refute")]
        support_prediction = support["parsed_prediction"]
        refute_prediction = refute["parsed_prediction"]
        complete = support_prediction in ALLOWED_LABELS and refute_prediction in ALLOWED_LABELS
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "claim": pair["claim"],
                "support_prediction": support_prediction,
                "refute_prediction": refute_prediction,
                "complete": complete,
                "prediction_changed": (
                    support_prediction != refute_prediction if complete else None
                ),
                "correct_sensitivity": (
                    support_prediction == "SUPPORTED" and refute_prediction == "REFUTED"
                ),
                "outcome": classify_pair(support_prediction, refute_prediction),
                "support_inference_error": support["inference_error"],
                "refute_inference_error": refute["inference_error"],
                "support_valid_format": support["valid_format"],
                "refute_valid_format": refute["valid_format"],
            }
        )
    return rows


def calculate_sensitivity_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    complete = [row for row in rows if row["complete"]]
    correct = sum(row["correct_sensitivity"] for row in rows)
    flips = sum(row["prediction_changed"] for row in complete)
    support_correct = sum(row["support_prediction"] == "SUPPORTED" for row in rows)
    refute_correct = sum(row["refute_prediction"] == "REFUTED" for row in rows)
    outcomes = Counter(row["outcome"] for row in rows)
    nei = "NOT ENOUGH INFORMATION"
    return {
        "total_pairs": total,
        "complete_pairs": len(complete),
        "incomplete_pairs": total - len(complete),
        "correct_sensitivity_count": correct,
        "correct_sensitivity_rate": correct / total if total else None,
        "flip_count": flips,
        "flip_rate": flips / len(complete) if complete else None,
        "correct_flip_count": outcomes["CORRECT_FLIP"],
        "incorrect_flip_count": flips - outcomes["CORRECT_FLIP"],
        "pair_outcomes": {name: outcomes[name] for name in PAIR_OUTCOMES},
        "individual_accuracy": {
            "support_correct": support_correct,
            "support_accuracy": support_correct / total if total else None,
            "refute_correct": refute_correct,
            "refute_accuracy": refute_correct / total if total else None,
            "overall_correct": support_correct + refute_correct,
            "overall_rows": total * 2,
            "overall_accuracy": (support_correct + refute_correct) / (total * 2)
            if total
            else None,
        },
        "nei_analysis": {
            "nei_on_support": sum(row["support_prediction"] == nei for row in rows),
            "nei_on_refute": sum(row["refute_prediction"] == nei for row in rows),
            "pairs_with_nei_either_side": sum(
                nei in {row["support_prediction"], row["refute_prediction"]} for row in rows
            ),
            "pairs_with_nei_both_sides": sum(
                row["support_prediction"] == row["refute_prediction"] == nei for row in rows
            ),
        },
        "technical_validity": {
            "expected_requests": total * 2,
            "inference_failures": sum(
                bool(row["support_inference_error"]) + bool(row["refute_inference_error"])
                for row in rows
            ),
            "invalid_formats": sum(
                (not row["support_inference_error"] and not row["support_valid_format"])
                + (not row["refute_inference_error"] and not row["refute_valid_format"])
                for row in rows
            ),
        },
    }


def _write_pair_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows)


def _report(metrics: dict[str, Any], config: dict[str, Any]) -> str:
    whole_note = ""
    if config.get("pair_selection") == "whole":
        whole_note = (
            f" (every valid contrastive pair in the test set; {config['unique_claims']} distinct "
            "claims, a claim that occurs in several pairs is kept in each)"
        )
    accuracy = metrics["individual_accuracy"]
    nei = metrics["nei_analysis"]
    outcomes = metrics["pair_outcomes"]
    return f"""# VitaminC contrastive sensitivity experiment

This experiment tests whether `{config["ollama_model"]}` changes its verdict correctly when the same claim receives SUPPORTS versus REFUTES evidence from an explicit VitaminC case group.

## Setup

- Test SHA-256: `{config["raw_test_sha256"]}`
- Selected pairs: {metrics["total_pairs"]}{whole_note}
- Model temperature / seed: {config["temperature"]} / {config["model_seed"]}
- Pairing: exact `case_id` plus exact claim, with one SUPPORTS and one REFUTES row

## Primary result

- Correct sensitivity: {metrics["correct_sensitivity_count"]}/{metrics["total_pairs"]} ({percentage(metrics["correct_sensitivity_rate"])})
- Verdict flips: {metrics["flip_count"]}/{metrics["complete_pairs"]} ({percentage(metrics["flip_rate"])})
- Correct flips: {metrics["correct_flip_count"]}
- Incorrect flips: {metrics["incorrect_flip_count"]}

## Pair outcomes

- Correct flip: {outcomes["CORRECT_FLIP"]}
- No change, correct on SUPPORT side: {outcomes["NO_CHANGE_CORRECT_ON_SUPPORT"]}
- No change, correct on REFUTE side: {outcomes["NO_CHANGE_CORRECT_ON_REFUTE"]}
- NEI involved: {outcomes["NEI_INVOLVED"]}
- Wrong direction: {outcomes["WRONG_DIRECTION"]}
- Other error: {outcomes["OTHER_ERROR"]}

## Individual accuracy

- SUPPORT side: {accuracy["support_correct"]}/{metrics["total_pairs"]} ({percentage(accuracy["support_accuracy"])})
- REFUTE side: {accuracy["refute_correct"]}/{metrics["total_pairs"]} ({percentage(accuracy["refute_accuracy"])})
- Overall rows: {accuracy["overall_correct"]}/{accuracy["overall_rows"]} ({percentage(accuracy["overall_accuracy"])})

## NEI behavior

- NEI on SUPPORT evidence: {nei["nei_on_support"]}
- NEI on REFUTE evidence: {nei["nei_on_refute"]}
- Pairs with NEI on either side: {nei["pairs_with_nei_either_side"]}
- Pairs with NEI on both sides: {nei["pairs_with_nei_both_sides"]}

Full per-pair predictions and technical validity are retained in the JSONL, CSV, and metrics artifacts.
"""


def run_vitaminc_sensitivity(
    raw_path: Path,
    processed_path: Path,
    sensitivity_dir: Path,
    limit: int | None = None,
    overwrite: bool = False,
    base_url: str = "http://localhost:11434",
    model: str = DEFAULT_MODEL,
    whole: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the contrastive sensitivity experiment.

    By default on the 100 frozen pairs in ``processed_path``. With ``whole`` on every valid
    contrastive pair of the test set (20,232), rebuilt from ``raw_path`` and not de-duplicated.
    Each answer is saved as it arrives (``predictions/predictions.partial.jsonl``); ``resume``
    keeps them, retries failures and runs only what is missing. The metrics file is written
    last: its existence is what marks a run as finished.
    """
    source_rows, source_summary = load_vitaminc_test(raw_path)
    candidates = identify_contrastive_pairs(source_rows)
    if whole:
        pairs = candidates  # every valid pair, already sorted by pair_id
        validate_pairs(pairs, expected_count=EXPECTED_VALID_PAIRS, unique_claims=False)
    else:
        pairs = list(read_jsonl(processed_path))
        validate_pairs(pairs, expected_count=100)
        validate_pairs_against_candidates(pairs, candidates)
    selected = pairs if limit is None else pairs[:limit]

    results_dir = sensitivity_dir if limit is None else sensitivity_dir / "smoke"
    predictions_path = results_dir / "predictions" / "predictions.jsonl"
    partial_path = results_dir / "predictions" / "predictions.partial.jsonl"
    metrics_path = results_dir / "sensitivity_metrics.json"
    if resume and predictions_path.exists() and metrics_path.exists():
        print(f"Already complete, nothing to do: {results_dir}", file=sys.stderr)
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    output_paths = [
        predictions_path,
        results_dir / "sensitivity_by_pair.csv",
        metrics_path,
        results_dir / "run_config.json",
        results_dir / "report.md",
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not (overwrite or resume):
        raise FileExistsError(f"Refusing to overwrite VitaminC sensitivity results: {existing}")
    if partial_path.exists() and not (resume or overwrite):
        raise FileExistsError(
            f"{partial_path} holds an interrupted run; pass --resume to continue it "
            "or --overwrite to discard it"
        )

    client = OllamaClient(base_url=base_url)
    ollama_metadata = client.check(model)
    cases = _inference_cases(selected)
    finished = list(read_jsonl(predictions_path)) if resume and predictions_path.exists() else None
    if finished is not None and len(finished) == len(cases):
        predictions = finished  # inference finished; only scoring is missing
    else:
        predictions = infer_cases_resumable(
            client,
            model,
            cases,
            SEED,
            TEMPERATURE,
            partial_path,
            resume=resume,
            progress_prefix="VitaminC",
            id_field="prediction_id",
            id_label="row",
        )
        for case, prediction in zip(cases, predictions, strict=True):
            prediction["pair_id"] = case["pair_id"]
            prediction["side"] = case["side"]
        write_jsonl(predictions_path, predictions)
        partial_path.unlink(missing_ok=True)

    pair_rows = build_pair_rows(selected, predictions)
    metrics = calculate_sensitivity_metrics(pair_rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_pair_csv(results_dir / "sensitivity_by_pair.csv", pair_rows)
    config = {
        "experiment": "VitaminC contrastive sensitivity",
        "raw_test_path": str(raw_path),
        "raw_test_sha256": source_summary["sha256"],
        "raw_test_rows": source_summary["row_count"],
        "raw_label_counts": source_summary["label_counts"],
        "processed_pairs_path": str(processed_path),
        "pair_selection": "whole" if whole else "frozen sample",
        "full_pair_count": len(pairs),
        "unique_claims": len({pair["claim"] for pair in selected}),
        "sample_size_pairs": len(selected),
        "pairing_method": "exact case_id plus exact claim; exactly one SUPPORTS and one REFUTES row; different evidence",
        "ollama_model": model,
        "temperature": TEMPERATURE,
        "model_seed": SEED,
        "prompt_template": {
            "system": SYSTEM_PROMPT,
            "user": "Claim:\n<claim>\n\nEvidence:\n<numbered evidence>",
        },
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "ollama_version": ollama_metadata["version"],
        "ollama_model_metadata": ollama_metadata["model"],
        "git_commit": git_commit(),
    }
    write_text_atomic(
        results_dir / "run_config.json", json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    )
    write_text_atomic(results_dir / "report.md", _report(metrics, config))
    write_text_atomic(metrics_path, json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    return metrics
