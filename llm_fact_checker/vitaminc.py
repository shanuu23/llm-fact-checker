"""Contrastive VitaminC sensitivity experiment for local Llama inference."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import random
from typing import Any

from llm_fact_checker.common import (
    file_sha256,
    git_commit,
    percentage,
    read_jsonl,
    write_csv,
    write_jsonl,
)
from llm_fact_checker.ollama import OllamaClient, infer_cases
from llm_fact_checker.prompts import ALLOWED_LABELS, SYSTEM_PROMPT

EXPECTED_TEST_SHA256 = "7f799e076dec184e31533d46704fef2022d08ef68a0db2c44f138d2b4e552f36"
EXPECTED_TEST_ROWS = 55_197
EXPECTED_LABEL_COUNTS = {"SUPPORTS": 27_664, "REFUTES": 20_265, "NOT ENOUGH INFO": 7_268}
REQUIRED_FIELDS = {"unique_id", "case_id", "label", "claim", "evidence", "page", "revision_type"}
LABEL_MAP = {
    "SUPPORTS": "SUPPORTED",
    "REFUTES": "REFUTED",
    "NOT ENOUGH INFO": "NOT ENOUGH INFORMATION",
}
PAIR_OUTCOMES = (
    "CORRECT_FLIP",
    "NO_CHANGE_CORRECT_ON_SUPPORT",
    "NO_CHANGE_CORRECT_ON_REFUTE",
    "NEI_INVOLVED",
    "WRONG_DIRECTION",
    "OTHER_ERROR",
)
MODEL = "llama3.2:3b"
TEMPERATURE = 0
SEED = 42


def map_vitaminc_label(label: str) -> str:
    try:
        return LABEL_MAP[label]
    except KeyError as error:
        raise ValueError(f"Unsupported VitaminC label: {label!r}") from error


def load_vitaminc_test(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    digest = file_sha256(path)
    if digest != EXPECTED_TEST_SHA256:
        raise ValueError(
            f"VitaminC test SHA-256 mismatch: expected {EXPECTED_TEST_SHA256}, found {digest}"
        )
    rows = list(read_jsonl(path))
    if len(rows) != EXPECTED_TEST_ROWS:
        raise ValueError(f"Expected {EXPECTED_TEST_ROWS} VitaminC test rows, found {len(rows)}")
    missing = [
        row.get("unique_id", index)
        for index, row in enumerate(rows)
        if not REQUIRED_FIELDS <= row.keys()
    ]
    if missing:
        raise ValueError(f"VitaminC rows missing required fields: {missing[:5]}")
    labels = Counter(row["label"] for row in rows)
    if dict(labels) != EXPECTED_LABEL_COUNTS:
        raise ValueError(f"Unexpected VitaminC label counts: {dict(labels)}")
    return rows, {
        "sha256": digest,
        "row_count": len(rows),
        "field_names": sorted({key for row in rows for key in row}),
        "field_presence": dict(sorted(Counter(key for row in rows for key in row).items())),
        "label_counts": dict(labels),
        "revision_type_counts": dict(Counter(row["revision_type"] for row in rows)),
    }


def _pair_side(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim": row["claim"],
        "evidence": row["evidence"],
        "original_label": row["label"],
        "canonical_label": map_vitaminc_label(row["label"]),
        "source_metadata": {
            key: value for key, value in row.items() if key not in {"claim", "evidence", "label"}
        },
    }


def _source_anchor(row: dict[str, Any]) -> tuple[str, Any]:
    key = "wiki_revision_id" if "wiki_revision_id" in row else "FEVER_id"
    return key, row[key]


def identify_contrastive_pairs(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Link only rows sharing explicit case_id and exact claim text."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["case_id"], row["claim"])].append(row)

    pairs = []
    for (case_id, claim), group in groups.items():
        supports = [row for row in group if row["label"] == "SUPPORTS"]
        refutes = [row for row in group if row["label"] == "REFUTES"]
        if len(supports) != 1 or len(refutes) != 1:
            continue
        support, refute = supports[0], refutes[0]
        if support["evidence"] == refute["evidence"]:
            continue
        if (
            support["page"] != refute["page"]
            or support["revision_type"] != refute["revision_type"]
            or _source_anchor(support) != _source_anchor(refute)
        ):
            continue
        pair_id = f"{support['unique_id']}__{refute['unique_id']}"
        pairs.append(
            {
                "pair_id": pair_id,
                "claim": claim,
                "support": _pair_side(support),
                "refute": _pair_side(refute),
                "source_metadata": {
                    "case_id": case_id,
                    "pairing_key": ["case_id", "claim"],
                    "support_unique_id": support["unique_id"],
                    "refute_unique_id": refute["unique_id"],
                },
            }
        )
    return sorted(pairs, key=lambda pair: pair["pair_id"])


def select_pairs(
    candidates: list[dict[str, Any]], count: int = 100, seed: int = SEED
) -> list[dict[str, Any]]:
    unique_claim_candidates = []
    seen_claims = set()
    for pair in sorted(candidates, key=lambda item: item["pair_id"]):
        if pair["claim"] in seen_claims:
            continue
        seen_claims.add(pair["claim"])
        unique_claim_candidates.append(pair)
    if len(unique_claim_candidates) < count:
        raise ValueError(
            f"Need {count} unique-claim contrastive pairs, found {len(unique_claim_candidates)}"
        )
    selected = random.Random(seed).sample(unique_claim_candidates, count)
    return sorted(selected, key=lambda pair: pair["pair_id"])


def validate_pairs(pairs: list[dict[str, Any]], expected_count: int | None = None) -> None:
    if expected_count is not None and len(pairs) != expected_count:
        raise ValueError(f"Expected {expected_count} pairs, found {len(pairs)}")
    if len({pair["pair_id"] for pair in pairs}) != len(pairs):
        raise ValueError("Duplicate VitaminC pair IDs")
    if len({pair["claim"] for pair in pairs}) != len(pairs):
        raise ValueError("Duplicate claims in selected VitaminC pairs")
    for pair in pairs:
        support, refute = pair["support"], pair["refute"]
        if support["claim"] != pair["claim"] or refute["claim"] != pair["claim"]:
            raise ValueError(f"Pair {pair['pair_id']} does not use the exact same claim")
        if support["original_label"] != "SUPPORTS" or refute["original_label"] != "REFUTES":
            raise ValueError(f"Pair {pair['pair_id']} does not have SUPPORTS/REFUTES labels")
        if support["canonical_label"] != "SUPPORTED" or refute["canonical_label"] != "REFUTED":
            raise ValueError(f"Pair {pair['pair_id']} has invalid canonical labels")
        if support["evidence"] == refute["evidence"]:
            raise ValueError(f"Pair {pair['pair_id']} has identical evidence")
        if support["source_metadata"]["case_id"] != refute["source_metadata"]["case_id"]:
            raise ValueError(f"Pair {pair['pair_id']} crosses case IDs")
        if support["source_metadata"]["page"] != refute["source_metadata"]["page"]:
            raise ValueError(f"Pair {pair['pair_id']} crosses pages")
        if (
            support["source_metadata"]["revision_type"]
            != refute["source_metadata"]["revision_type"]
        ):
            raise ValueError(f"Pair {pair['pair_id']} crosses revision types")
        if _source_anchor(support["source_metadata"]) != _source_anchor(refute["source_metadata"]):
            raise ValueError(f"Pair {pair['pair_id']} crosses source revisions")


def validate_pairs_against_candidates(
    selected: list[dict[str, Any]], candidates: list[dict[str, Any]]
) -> None:
    candidate_by_id = {pair["pair_id"]: pair for pair in candidates}
    for pair in selected:
        if candidate_by_id.get(pair["pair_id"]) != pair:
            raise ValueError(f"Selected pair is not an unchanged candidate: {pair['pair_id']}")


def _audit_markdown(pairs: list[dict[str, Any]], count: int = 20) -> str:
    sections = [
        "# VitaminC sensitivity pair audit",
        "",
        "Pairs share the same `case_id` and exact claim. Each has one unchanged VitaminC SUPPORTS row and one unchanged REFUTES row with different evidence.",
    ]
    for pair in pairs[:count]:
        sections.extend(
            [
                "",
                f"## PAIR ID: {pair['pair_id']}",
                "",
                "**Claim:**",
                "",
                pair["claim"],
                "",
                "**Evidence A:**",
                "",
                pair["support"]["evidence"],
                "",
                "**Gold: SUPPORTED**",
                "",
                "**Evidence B:**",
                "",
                pair["refute"]["evidence"],
                "",
                "**Gold: REFUTED**",
            ]
        )
    return "\n".join(sections) + "\n"


def prepare_vitaminc_pairs(
    raw_path: Path,
    processed_path: Path,
    prepared_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_paths = [
        processed_path,
        prepared_dir / "selected_pairs.jsonl",
        prepared_dir / "pair_audit.md",
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite VitaminC prepared files: {existing}")
    rows, source_summary = load_vitaminc_test(raw_path)
    candidates = identify_contrastive_pairs(rows)
    selected = select_pairs(candidates)
    validate_pairs(selected, expected_count=100)
    validate_pairs_against_candidates(selected, candidates)
    write_jsonl(processed_path, selected)
    write_jsonl(prepared_dir / "selected_pairs.jsonl", selected)
    (prepared_dir / "pair_audit.md").write_text(_audit_markdown(selected), encoding="utf-8")
    return {
        **source_summary,
        "pairing_method": "exact case_id plus exact claim; exactly one SUPPORTS and one REFUTES row; different evidence",
        "valid_contrastive_pairs": len(candidates),
        "unique_claim_candidate_pairs": len({pair["claim"] for pair in candidates}),
        "selected_pairs": len(selected),
        "selection_seed": SEED,
    }


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
    accuracy = metrics["individual_accuracy"]
    nei = metrics["nei_analysis"]
    outcomes = metrics["pair_outcomes"]
    return f"""# VitaminC contrastive sensitivity experiment

This experiment tests whether `{config["ollama_model"]}` changes its verdict correctly when the same claim receives SUPPORTS versus REFUTES evidence from an explicit VitaminC case group.

## Setup

- Test SHA-256: `{config["raw_test_sha256"]}`
- Selected pairs: {metrics["total_pairs"]}
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
) -> dict[str, Any]:
    source_rows, source_summary = load_vitaminc_test(raw_path)
    candidates = identify_contrastive_pairs(source_rows)
    pairs = list(read_jsonl(processed_path))
    prepared_pairs = list(read_jsonl(sensitivity_dir / "prepared" / "selected_pairs.jsonl"))
    if pairs != prepared_pairs:
        raise ValueError("Processed and prepared VitaminC pair files differ")
    validate_pairs(pairs, expected_count=100)
    validate_pairs_against_candidates(pairs, candidates)
    selected = pairs if limit is None else pairs[:limit]

    results_dir = sensitivity_dir if limit is None else sensitivity_dir / "smoke"
    predictions_path = results_dir / "predictions" / "predictions.jsonl"
    output_paths = [
        predictions_path,
        results_dir / "sensitivity_by_pair.csv",
        results_dir / "sensitivity_metrics.json",
        results_dir / "run_config.json",
        results_dir / "report.md",
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite VitaminC sensitivity results: {existing}")

    client = OllamaClient(base_url=base_url)
    ollama_metadata = client.check(MODEL)
    cases = _inference_cases(selected)
    predictions = infer_cases(
        client,
        MODEL,
        cases,
        seed=SEED,
        temperature=TEMPERATURE,
        progress_prefix="VitaminC",
        id_field="prediction_id",
        id_label="row",
    )
    for case, prediction in zip(cases, predictions, strict=True):
        prediction["pair_id"] = case["pair_id"]
        prediction["side"] = case["side"]
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(predictions_path, predictions)

    pair_rows = build_pair_rows(selected, predictions)
    metrics = calculate_sensitivity_metrics(pair_rows)
    _write_pair_csv(results_dir / "sensitivity_by_pair.csv", pair_rows)
    config = {
        "experiment": "VitaminC contrastive sensitivity",
        "raw_test_path": str(raw_path),
        "raw_test_sha256": source_summary["sha256"],
        "raw_test_rows": source_summary["row_count"],
        "raw_label_counts": source_summary["label_counts"],
        "processed_pairs_path": str(processed_path),
        "full_pair_count": len(pairs),
        "sample_size_pairs": len(selected),
        "pairing_method": "exact case_id plus exact claim; exactly one SUPPORTS and one REFUTES row; different evidence",
        "ollama_model": MODEL,
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
    (results_dir / "sensitivity_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (results_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (results_dir / "report.md").write_text(_report(metrics, config), encoding="utf-8")
    return metrics
