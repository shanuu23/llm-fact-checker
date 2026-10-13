"""Checkpointed Ollama inference for frozen full-evaluation inputs.

This is separate from the original pilot commands. VitaminC requires an explicit
pair-ID selection; its full eligibility index is not an experiment workload.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

from llm_fact_checker.common import file_sha256, read_jsonl
from llm_fact_checker.ollama import OllamaClient, OllamaResponseError
from llm_fact_checker.prompts import ALLOWED_LABELS, SYSTEM_PROMPT, parse_prediction

PAIR_LABELS = {
    "SUPPORTS_REFUTES": ("SUPPORTED", "REFUTED"),
    "SUPPORTS_NEI": ("SUPPORTED", "NOT ENOUGH INFORMATION"),
    "REFUTES_NEI": ("REFUTED", "NOT ENOUGH INFORMATION"),
}


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _case(case_id: str, row: dict[str, Any], **extra: Any) -> dict[str, Any]:
    claim, evidence, gold = row["claim"], row["evidence"], row["gold_label"]
    if not isinstance(claim, str) or not claim.strip() or gold not in ALLOWED_LABELS:
        raise ValueError(f"Invalid claim or label for {case_id}")
    if (
        not isinstance(evidence, list)
        or not evidence
        or any(
            not isinstance(item, dict)
            or not isinstance(item.get("page"), str)
            or not isinstance(item.get("text"), str)
            or not item["text"].strip()
            or "sentence_id" not in item
            for item in evidence
        )
    ):
        raise ValueError(f"Invalid evidence for {case_id}")
    return {
        "case_id": case_id,
        "claim": claim,
        "evidence": evidence,
        "gold_label": gold,
        "audit_flags": row.get("audit_flags", []),
        **extra,
    }


def fever_cases(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")
    cases = []
    seen = set()
    for row in read_jsonl(path):
        case_id = f"fever:{row['fever_id']}"
        if case_id in seen:
            raise ValueError(f"Duplicate FEVER ID: {case_id}")
        seen.add(case_id)
        cases.append(_case(case_id, row, fever_id=row["fever_id"]))
        if len(cases) == limit:
            break
    if not cases:
        raise ValueError(f"No FEVER cases in {path}")
    return cases


def vitaminc_cases(rows_path: Path, pairs_path: Path, selected_path: Path) -> list[dict[str, Any]]:
    selected = [row["pair_id"] for row in read_jsonl(selected_path)]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Selected pair IDs must be nonempty and unique")
    wanted = set(selected)
    pairs = {}
    for pair in read_jsonl(pairs_path):
        pair_id = pair["pair_id"]
        if pair_id in wanted:
            if pair_id in pairs:
                raise ValueError(f"Duplicate pair in index: {pair_id}")
            pairs[pair_id] = pair
    missing = wanted - pairs.keys()
    if missing:
        raise ValueError(f"Selected pair IDs not in index: {sorted(missing)[:5]}")

    wanted_rows = {
        pair[f"{side}_unique_id"] for pair in pairs.values() for side in ("left", "right")
    }
    rows = {}
    for row in read_jsonl(rows_path):
        unique_id = row["unique_id"]
        if unique_id in wanted_rows:
            if unique_id in rows:
                raise ValueError(f"Duplicate VitaminC row: {unique_id}")
            rows[unique_id] = row
    missing = wanted_rows - rows.keys()
    if missing:
        raise ValueError(f"Selected VitaminC rows not in index: {sorted(missing)[:5]}")

    cases = []
    for pair_id in selected:
        pair = pairs[pair_id]
        if (
            pair["pair_type"] not in PAIR_LABELS
            or (pair["left_gold_label"], pair["right_gold_label"])
            != PAIR_LABELS[pair["pair_type"]]
        ):
            raise ValueError(f"Invalid transition labels: {pair_id}")
        for side in ("left", "right"):
            row = rows[pair[f"{side}_unique_id"]]
            if (
                any(row[key] != pair[key] for key in ("case_id", "claim", "page"))
                or row["revision_type"] != pair["revision_type"]
                or row["gold_label"] != pair[f"{side}_gold_label"]
            ):
                raise ValueError(f"Row/pair mismatch: {pair_id}, {side}")
            cases.append(
                _case(
                    f"vitaminc:{row['unique_id']}",
                    {
                        **row,
                        "audit_flags": pair["audit_flags"],
                        "evidence": [
                            {
                                "page": row["page"],
                                "sentence_id": row["unique_id"],
                                "text": row["evidence"],
                            }
                        ],
                    },
                    pair_id=pair_id,
                    pair_type=pair["pair_type"],
                    side=side,
                    revision_type=pair["revision_type"],
                )
            )
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("A VitaminC row appears in more than one selected pair")
    return cases


def _saved_predictions(path: Path, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    saved = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line_number > len(cases):
                raise ValueError("Checkpoint has more predictions than selected cases")
            try:
                prediction = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid checkpoint JSON at line {line_number}") from error
            case = cases[line_number - 1]
            if prediction.get("case_id") != case["case_id"] or prediction.get(
                "input_sha256"
            ) != _digest(case):
                raise ValueError(f"Checkpoint/input mismatch at line {line_number}")
            saved.append(prediction)
    return saved


def run(
    cases: list[dict[str, Any]],
    results_dir: Path,
    model: str,
    source_hashes: dict[str, str],
    client: OllamaClient,
    *,
    seed: int = 42,
    temperature: float = 0,
) -> dict[str, Any]:
    """Append one durable result per call; never overwrite a previous result."""
    if not cases or len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Cases must be nonempty and have unique IDs")
    contract = {
        "schema_version": 1,
        "model": model,
        "provider": "ollama",
        "ollama_url": client.base_url,
        "seed": seed,
        "temperature": temperature,
        "transport_retries": 2,
        "source_hashes": source_hashes,
        "cases_sha256": _digest(cases),
        "case_count": len(cases),
        "prompt_source_sha256": file_sha256(Path(__file__).with_name("prompts.py")),
        "system_prompt": SYSTEM_PROMPT,
    }
    config_path = results_dir / "run_config.json"
    predictions_path = results_dir / "predictions.jsonl"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("contract") != contract:
            raise ValueError("Run configuration or inputs changed; refusing to resume")
    elif results_dir.exists() and any(results_dir.iterdir()):
        raise FileExistsError(f"Results directory is not an empty new run: {results_dir}")
    else:
        config = None
    saved = _saved_predictions(predictions_path, cases)
    if len(saved) < len(cases):
        metadata = client.check(model)
        model_digest = metadata["model"].get("digest")
        if not model_digest:
            raise ValueError("Ollama did not provide a model digest; cannot freeze this run")
        if config is not None:
            if config["model_digest"] != model_digest:
                raise ValueError("Installed model digest changed; refusing to resume")
        else:
            results_dir.mkdir(parents=True, exist_ok=True)
            config = {
                "contract": contract,
                "model_digest": model_digest,
                "ollama_version": metadata["version"],
                "created_utc": datetime.now(UTC).isoformat(),
            }
            with config_path.open("x", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())

        with predictions_path.open("a", encoding="utf-8") as handle:
            for index in range(len(saved), len(cases)):
                case = cases[index]
                print(f"Inference {index + 1}/{len(cases)}: {case['case_id']}", file=sys.stderr)
                raw_output = latency = error = None
                response_metadata = {}
                try:
                    raw_output, latency, response_metadata = client.chat(
                        model,
                        case["claim"],
                        case["evidence"],
                        seed=seed,
                        temperature=temperature,
                        retries=2,
                    )
                except (RuntimeError, OllamaResponseError) as caught:
                    error = str(caught)
                parsed = parse_prediction(raw_output) if raw_output is not None else None
                result = {
                    **case,
                    "input_sha256": _digest(case),
                    "raw_model_output": raw_output,
                    "parsed_prediction": parsed,
                    "correct": parsed == case["gold_label"],
                    "valid_format": parsed is not None,
                    "inference_latency_seconds": latency,
                    "inference_error": error,
                    "ollama_response_metadata": response_metadata,
                    "completed_utc": datetime.now(UTC).isoformat(),
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                saved.append(result)
    return {
        "completed": len(saved),
        "total": len(cases),
        "correct": sum(row["correct"] for row in saved),
        "invalid_format": sum(
            row["inference_error"] is None and not row["valid_format"] for row in saved
        ),
        "inference_failures": sum(row["inference_error"] is not None for row in saved),
        "results_dir": str(results_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    subparsers = parser.add_subparsers(dest="dataset", required=True)
    fever = subparsers.add_parser("fever")
    fever.add_argument("--prepared", type=Path, required=True)
    fever.add_argument("--limit", type=int)
    vitaminc = subparsers.add_parser("vitaminc")
    vitaminc.add_argument("--rows", type=Path, required=True)
    vitaminc.add_argument("--pairs", type=Path, required=True)
    vitaminc.add_argument("--selected-pairs", type=Path, required=True)
    args = parser.parse_args()
    if args.dataset == "fever":
        cases = fever_cases(args.prepared, args.limit)
        source_hashes = {"prepared": file_sha256(args.prepared)}
    else:
        cases = vitaminc_cases(args.rows, args.pairs, args.selected_pairs)
        source_hashes = {
            "rows": file_sha256(args.rows),
            "pairs": file_sha256(args.pairs),
            "selected_pairs": file_sha256(args.selected_pairs),
        }
    summary = run(
        cases,
        args.results_dir,
        args.model,
        source_hashes,
        OllamaClient(base_url=args.ollama_url),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
