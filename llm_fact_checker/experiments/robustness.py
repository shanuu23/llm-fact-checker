"""Evidence-perturbation robustness experiment for the fixed FEVER sample."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import re
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
from llm_fact_checker.experiments.consistency import (
    EXPECTED_DATASET_SHA256,
    validate_prepared_dataset,
)
from llm_fact_checker.models import DEFAULT_MODEL
from llm_fact_checker.ollama import OllamaClient, infer_cases_resumable
from llm_fact_checker.prompts import ALLOWED_LABELS, LABELS, SYSTEM_PROMPT

TEMPERATURE = 0
SEED = 42
VARIANTS = ("control", "reorder", "distractor")
OUTCOMES = (
    "STAYED_CORRECT",
    "HARMFUL_FLIP",
    "BENEFICIAL_FLIP",
    "STAYED_WRONG",
    "INCOMPLETE",
)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "was",
    "were",
    "which",
    "who",
    "with",
}
TOKEN_RE = re.compile(r"[a-z0-9]+")


def important_words(*texts: str) -> set[str]:
    return {
        token
        for text in texts
        for token in TOKEN_RE.findall(text.lower().replace("_", " "))
        if len(token) > 2 and token not in STOPWORDS
    }


def has_substantial_overlap(candidate_text: str, target_texts: list[str]) -> bool:
    """Treat two shared non-stopwords as substantial lexical overlap."""
    return len(important_words(candidate_text) & important_words(*target_texts)) >= 2


def reverse_evidence_case(case: dict[str, Any]) -> dict[str, Any]:
    evidence = [dict(item) for item in case["evidence"]]
    eligible = len(evidence) >= 2
    return {
        **case,
        "evidence": list(reversed(evidence)) if eligible else evidence,
        "reorder_eligible": eligible,
    }


def _distractor_pool(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pool = [
        {
            "distractor_source_fever_id": case["fever_id"],
            "distractor_page": item["page"],
            "distractor_sentence_id": item["sentence_id"],
            "distractor_text": item["text"],
        }
        for case in dataset
        for item in case["evidence"]
    ]
    return sorted(
        pool,
        key=lambda item: (
            item["distractor_source_fever_id"],
            item["distractor_page"],
            item["distractor_sentence_id"],
            item["distractor_text"],
        ),
    )


def choose_distractor(target: dict[str, Any], pool: list[dict[str, Any]]) -> dict[str, Any]:
    target_pages = {item["page"] for item in target["evidence"]}
    target_texts = [
        target["claim"],
        *(item["text"] for item in target["evidence"]),
        *target_pages,
    ]
    start = next(
        (
            index
            for index, candidate in enumerate(pool)
            if candidate["distractor_source_fever_id"] > target["fever_id"]
        ),
        0,
    )
    for candidate in pool[start:] + pool[:start]:
        if candidate["distractor_source_fever_id"] == target["fever_id"]:
            continue
        if candidate["distractor_page"] in target_pages:
            continue
        if has_substantial_overlap(candidate["distractor_text"], target_texts):
            continue
        return dict(candidate)
    raise ValueError(f"No valid distractor found for FEVER {target['fever_id']}")


def add_distractor_case(case: dict[str, Any], pool: list[dict[str, Any]]) -> dict[str, Any]:
    distractor = choose_distractor(case, pool)
    evidence = [dict(item) for item in case["evidence"]]
    evidence.append(
        {
            "page": distractor["distractor_page"],
            "sentence_id": distractor["distractor_sentence_id"],
            "text": distractor["distractor_text"],
        }
    )
    return {**case, "evidence": evidence, **distractor}


def build_perturbations(
    dataset: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    control = [{**case, "evidence": [dict(item) for item in case["evidence"]]} for case in dataset]
    reordered = [reverse_evidence_case(case) for case in dataset]
    pool = _distractor_pool(dataset)
    distractors = [add_distractor_case(case, pool) for case in dataset]
    return control, reordered, distractors


def validate_perturbations(
    source: list[dict[str, Any]],
    control: list[dict[str, Any]],
    reordered: list[dict[str, Any]],
    distractors: list[dict[str, Any]],
) -> None:
    if not (len(source) == len(control) == len(reordered) == len(distractors)):
        raise ValueError(
            "Source and prepared robustness files must contain the same cases: "
            f"{len(source)} source, {len(control)} control, {len(reordered)} reordered, "
            f"{len(distractors)} distractor"
        )
    source_by_id = {case["fever_id"]: case for case in source}
    pool = _distractor_pool(source)
    for control_case, reorder_case, distractor_case in zip(
        control, reordered, distractors, strict=True
    ):
        fever_id = control_case["fever_id"]
        original = source_by_id.get(fever_id)
        if (
            original is None
            or reorder_case["fever_id"] != fever_id
            or distractor_case["fever_id"] != fever_id
        ):
            raise ValueError(
                "Prepared files do not contain identical FEVER IDs in identical order"
            )
        if (
            control_case["claim"] != original["claim"]
            or control_case["evidence"] != original["evidence"]
        ):
            raise ValueError(f"Control altered FEVER {fever_id}")
        eligible = len(original["evidence"]) >= 2
        expected_reorder = (
            list(reversed(original["evidence"])) if eligible else original["evidence"]
        )
        if (
            reorder_case.get("reorder_eligible") != eligible
            or reorder_case["evidence"] != expected_reorder
        ):
            raise ValueError(f"Invalid evidence reversal for FEVER {fever_id}")
        if distractor_case["evidence"][:-1] != original["evidence"]:
            raise ValueError(
                f"Distractor preparation altered original evidence for FEVER {fever_id}"
            )
        added = distractor_case["evidence"][-1]
        if len(distractor_case["evidence"]) != len(original["evidence"]) + 1:
            raise ValueError(f"FEVER {fever_id} does not have exactly one added distractor")
        if distractor_case["distractor_source_fever_id"] == fever_id:
            raise ValueError(f"FEVER {fever_id} uses a distractor from itself")
        if added["page"] in {item["page"] for item in original["evidence"]}:
            raise ValueError(f"FEVER {fever_id} uses a distractor from the same page")
        if added != {
            "page": distractor_case["distractor_page"],
            "sentence_id": distractor_case["distractor_sentence_id"],
            "text": distractor_case["distractor_text"],
        }:
            raise ValueError(f"Distractor metadata mismatch for FEVER {fever_id}")
        expected_distractor = choose_distractor(original, pool)
        if any(distractor_case[key] != value for key, value in expected_distractor.items()):
            raise ValueError(f"Distractor selection is not deterministic for FEVER {fever_id}")


def _audit_markdown(
    control: list[dict[str, Any]], distractors: list[dict[str, Any]], count: int = 20
) -> str:
    sections = [
        "# FEVER distractor audit",
        "",
        "These are the first 20 cases in the fixed, FEVER-ID-sorted dataset. The added sentence is shown after the unchanged genuine evidence.",
    ]
    for case, perturbed in list(zip(control, distractors, strict=True))[:count]:
        real_evidence = "\n".join(
            f"{index}. [{item['page']} #{item['sentence_id']}] {item['text']}"
            for index, item in enumerate(case["evidence"], 1)
        )
        sections.extend(
            [
                "",
                f"## CASE ID: {case['fever_id']}",
                "",
                "**Claim:**",
                "",
                case["claim"],
                "",
                "**Real evidence:**",
                "",
                real_evidence,
                "",
                "**Added distractor:**",
                "",
                f"[{perturbed['distractor_page']} #{perturbed['distractor_sentence_id']}; source FEVER {perturbed['distractor_source_fever_id']}] {perturbed['distractor_text']}",
            ]
        )
    return "\n".join(sections) + "\n"


def prepare_robustness_files(
    prepared_path: Path, output_dir: Path, overwrite: bool = False
) -> dict[str, int | str]:
    dataset, dataset_hash = validate_prepared_dataset(prepared_path)
    paths = {
        "control": output_dir / "control.jsonl",
        "reorder": output_dir / "reordered_evidence.jsonl",
        "distractor": output_dir / "distractor_evidence.jsonl",
        "audit": output_dir / "distractor_audit.md",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite prepared robustness files: {existing}")
    control, reordered, distractors = build_perturbations(dataset)
    validate_perturbations(dataset, control, reordered, distractors)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths["control"], control)
    write_jsonl(paths["reorder"], reordered)
    write_jsonl(paths["distractor"], distractors)
    paths["audit"].write_text(_audit_markdown(control, distractors), encoding="utf-8")
    return {
        "dataset_sha256": dataset_hash,
        "cases": len(dataset),
        "reorder_eligible_cases": sum(case["reorder_eligible"] for case in reordered),
        "distractors_generated": len(distractors),
    }


def outcome_category(
    control_prediction: str | None, perturbed_prediction: str | None, gold_label: str
) -> str:
    if control_prediction not in ALLOWED_LABELS or perturbed_prediction not in ALLOWED_LABELS:
        return "INCOMPLETE"
    control_correct = control_prediction == gold_label
    perturbed_correct = perturbed_prediction == gold_label
    if control_correct and perturbed_correct:
        return "STAYED_CORRECT"
    if control_correct:
        return "HARMFUL_FLIP"
    if perturbed_correct:
        return "BENEFICIAL_FLIP"
    return "STAYED_WRONG"


def prediction_changed(
    control_prediction: str | None, perturbed_prediction: str | None
) -> bool | None:
    if control_prediction not in ALLOWED_LABELS or perturbed_prediction not in ALLOWED_LABELS:
        return None
    return control_prediction != perturbed_prediction


def invariance_metrics(pairs: list[tuple[str | None, str | None, str]]) -> dict[str, Any]:
    complete = [pair for pair in pairs if pair[0] in ALLOWED_LABELS and pair[1] in ALLOWED_LABELS]
    invariant = sum(control == perturbed for control, perturbed, _ in complete)
    return {
        "complete_pair_count": len(complete),
        "invariant_count": invariant,
        "invariance_rate": invariant / len(complete) if complete else None,
    }


def transition_counts(
    pairs: list[tuple[str | None, str | None, str]],
) -> dict[str, int]:
    counts = {f"{source} -> {target}": 0 for source in LABELS for target in LABELS}
    for control, perturbed, _ in pairs:
        if control in ALLOWED_LABELS and perturbed in ALLOWED_LABELS:
            counts[f"{control} -> {perturbed}"] += 1
    return counts


def nei_transition_counts(
    pairs: list[tuple[str | None, str | None, str]],
) -> dict[str, int]:
    complete = [pair for pair in pairs if pair[0] in ALLOWED_LABELS and pair[1] in ALLOWED_LABELS]
    nei = "NOT ENOUGH INFORMATION"
    return {
        "control_nei_count": sum(control == nei for control, _, _ in complete),
        "perturbed_nei_count": sum(perturbed == nei for _, perturbed, _ in complete),
        "non_nei_to_nei": sum(
            control != nei and perturbed == nei for control, perturbed, _ in complete
        ),
        "nei_to_non_nei": sum(
            control == nei and perturbed != nei for control, perturbed, _ in complete
        ),
        "nei_to_nei": sum(
            control == nei and perturbed == nei for control, perturbed, _ in complete
        ),
    }


def calculate_pair_metrics(
    pairs: list[tuple[str | None, str | None, str]],
) -> dict[str, Any]:
    total = len(pairs)
    invariance = invariance_metrics(pairs)
    outcomes = Counter(outcome_category(*pair) for pair in pairs)
    control_correct = sum(control == gold for control, _, gold in pairs)
    perturbed_correct = sum(perturbed == gold for _, perturbed, gold in pairs)
    control_accuracy = control_correct / total if total else None
    perturbed_accuracy = perturbed_correct / total if total else None
    return {
        "eligible_pair_count": total,
        **invariance,
        "prediction_changed_count": invariance["complete_pair_count"]
        - invariance["invariant_count"],
        "control_correct": control_correct,
        "control_accuracy": control_accuracy,
        "perturbed_correct": perturbed_correct,
        "perturbed_accuracy": perturbed_accuracy,
        "accuracy_delta": perturbed_accuracy - control_accuracy if total else None,
        "outcomes": {name: outcomes[name] for name in OUTCOMES},
        "transitions": transition_counts(pairs),
        "nei_analysis": nei_transition_counts(pairs),
    }


def _prediction_map(predictions: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    return {item["fever_id"]: item for item in predictions}


def _pairs(
    control: list[dict[str, Any]], perturbed: list[dict[str, Any]]
) -> list[tuple[str | None, str | None, str]]:
    controls = _prediction_map(control)
    return [
        (
            controls[item["fever_id"]]["parsed_prediction"],
            item["parsed_prediction"],
            item["gold_label"],
        )
        for item in perturbed
    ]


def calculate_robustness_metrics(
    control: list[dict[str, Any]],
    reorder: list[dict[str, Any]],
    distractor: list[dict[str, Any]],
) -> dict[str, Any]:
    control_accuracy = sum(item["correct"] for item in control) / len(control) if control else None
    metrics = {
        "case_count": len(control),
        "control": {
            "correct": sum(item["correct"] for item in control),
            "accuracy": control_accuracy,
        },
        "reorder": calculate_pair_metrics(_pairs(control, reorder)),
        "distractor": calculate_pair_metrics(_pairs(control, distractor)),
        "technical_validity": {},
    }
    for name, predictions in zip(VARIANTS, (control, reorder, distractor), strict=True):
        metrics["technical_validity"][name] = {
            "requests": len(predictions),
            "inference_failures": sum(bool(item["inference_error"]) for item in predictions),
            "invalid_formats": sum(
                not item["inference_error"] and not item["valid_format"] for item in predictions
            ),
        }
    return metrics


def _case_rows(
    control: list[dict[str, Any]],
    reorder: list[dict[str, Any]],
    distractor: list[dict[str, Any]],
    reorder_eligible_ids: set[Any],
) -> list[dict[str, Any]]:
    reorder_by_id = _prediction_map(reorder)
    distractor_by_id = _prediction_map(distractor)
    rows = []
    for item in control:
        fever_id = item["fever_id"]
        control_prediction = item["parsed_prediction"]
        distractor_prediction = distractor_by_id[fever_id]["parsed_prediction"]
        reorder_item = reorder_by_id.get(fever_id)
        reorder_prediction = reorder_item["parsed_prediction"] if reorder_item else None
        rows.append(
            {
                "fever_id": fever_id,
                "claim": item["claim"],
                "gold_label": item["gold_label"],
                "control_prediction": control_prediction,
                "reorder_eligible": fever_id in reorder_eligible_ids,
                "reorder_prediction": reorder_prediction,
                "reorder_prediction_changed": (
                    prediction_changed(control_prediction, reorder_prediction)
                    if reorder_item
                    else None
                ),
                "reorder_outcome": (
                    outcome_category(control_prediction, reorder_prediction, item["gold_label"])
                    if reorder_item
                    else "INELIGIBLE"
                ),
                "distractor_prediction": distractor_prediction,
                "distractor_prediction_changed": prediction_changed(
                    control_prediction, distractor_prediction
                ),
                "distractor_outcome": outcome_category(
                    control_prediction, distractor_prediction, item["gold_label"]
                ),
            }
        )
    return rows


def _write_case_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows)


def _report(metrics: dict[str, Any], config: dict[str, Any]) -> str:
    reorder = metrics["reorder"]
    distractor = metrics["distractor"]
    return f"""# FEVER evidence-perturbation robustness experiment

This experiment compares a deterministic control with two meaning-preserving evidence changes. Invariance measures whether the prediction stayed the same; FEVER accuracy measures agreement with the dataset label.

## Setup

- Dataset SHA-256: `{config["prepared_dataset_sha256"]}`
- Cases: {config["sample_size"]}
- Model: `{config["ollama_model"]}`
- Temperature / seed: {config["temperature"]} / {config["model_seed"]}
- Reorder-eligible cases: {config["reorder_eligible_cases"]}

## Results

| Variant | Complete pairs | Invariance | Paired control accuracy | Perturbed accuracy | Delta |
|---|---:|---:|---:|---:|---:|
| Reorder | {reorder["complete_pair_count"]} | {percentage(reorder["invariance_rate"])} | {percentage(reorder["control_accuracy"])} | {percentage(reorder["perturbed_accuracy"])} | {percentage(reorder["accuracy_delta"], signed=True)} |
| Distractor | {distractor["complete_pair_count"]} | {percentage(distractor["invariance_rate"])} | {percentage(distractor["control_accuracy"])} | {percentage(distractor["perturbed_accuracy"])} | {percentage(distractor["accuracy_delta"], signed=True)} |

Control FEVER accuracy across all {metrics["case_count"]} cases: {percentage(metrics["control"]["accuracy"])}.

Detailed outcome, transition, NEI, and technical-validity counts are in `robustness_metrics.json` and `robustness_by_case.csv`.
"""


def _flagged_section(metrics: dict[str, Any], excluded: dict[str, Any]) -> str:
    def pct(value: float | None) -> str:
        return percentage(value)

    rows = [
        ("Cases", metrics["case_count"], excluded["case_count"], False),
        (
            "Control FEVER accuracy",
            metrics["control"]["accuracy"],
            excluded["control"]["accuracy"],
            True,
        ),
        (
            "Reorder invariance",
            metrics["reorder"]["invariance_rate"],
            excluded["reorder"]["invariance_rate"],
            True,
        ),
        (
            "Distractor invariance",
            metrics["distractor"]["invariance_rate"],
            excluded["distractor"]["invariance_rate"],
            True,
        ),
        (
            "Distractor FEVER accuracy",
            metrics["distractor"]["perturbed_accuracy"],
            excluded["distractor"]["perturbed_accuracy"],
            True,
        ),
    ]
    table = "\n".join(
        f"| {name} | {pct(a) if is_rate else a} | {pct(b) if is_rate else b} |"
        for name, a, b, is_rate in rows
    )
    return f"""
## Result excluding flagged records

The FEVER data audit flagged {excluded["excluded_flagged_cases"]} of these claims as possibly
inconsistent. The main result above keeps them; this section removes them, as the audit decided.
Flags are possible label inconsistencies, not proven errors.

| | All cases | Excluding flagged |
|---|---:|---:|
{table}
"""


def run_robustness_experiment(
    prepared_path: Path,
    robustness_dir: Path,
    limit: int | None = None,
    overwrite: bool = False,
    base_url: str = "http://localhost:11434",
    model: str = DEFAULT_MODEL,
    prepared_dir: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run control, reorder and distractor passes, saving answers as they arrive.

    ``resume=True`` carries on an interrupted experiment (finished passes are kept, a partly
    finished pass continues, failures are retried) and does nothing if it already finished.
    """
    source, dataset_hash = validate_prepared_dataset(prepared_path)
    prepared_dir = prepared_dir if prepared_dir is not None else robustness_dir / "prepared"
    control = list(read_jsonl(prepared_dir / "control.jsonl"))
    reordered = list(read_jsonl(prepared_dir / "reordered_evidence.jsonl"))
    distractors = list(read_jsonl(prepared_dir / "distractor_evidence.jsonl"))
    validate_perturbations(source, control, reordered, distractors)

    selected_ids = {case["fever_id"] for case in (control if limit is None else control[:limit])}
    control_cases = [case for case in control if case["fever_id"] in selected_ids]
    reorder_cases = [
        case for case in reordered if case["fever_id"] in selected_ids and case["reorder_eligible"]
    ]
    distractor_cases = [case for case in distractors if case["fever_id"] in selected_ids]
    results_dir = robustness_dir if limit is None else robustness_dir / "smoke"
    predictions_dir = results_dir / "predictions"
    output_paths = [
        predictions_dir / "control_predictions.jsonl",
        predictions_dir / "reorder_predictions.jsonl",
        predictions_dir / "distractor_predictions.jsonl",
        results_dir / "robustness_by_case.csv",
        results_dir / "robustness_metrics.json",
        results_dir / "run_config.json",
        results_dir / "report.md",
    ]
    metrics_path = results_dir / "robustness_metrics.json"
    if resume and metrics_path.exists():
        print(f"Already complete, nothing to do: {results_dir}", file=sys.stderr)
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    existing = [path for path in output_paths if path.exists()]
    if existing and not (overwrite or resume):
        raise FileExistsError(
            f"Refusing to overwrite robustness results: {existing}; pass --resume to continue "
            "an interrupted run or --overwrite to replace it"
        )

    client = OllamaClient(base_url=base_url)
    ollama_metadata = client.check(model)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    def predictions_for(name: str, variant_cases: list[dict[str, Any]], prefix: str):
        final_path = predictions_dir / f"{name}_predictions.jsonl"
        if resume and final_path.exists():
            finished = list(read_jsonl(final_path))
            if len(finished) == len(variant_cases):
                return finished  # this pass finished before the interruption
        partial_path = predictions_dir / f"{name}_predictions.partial.jsonl"
        rows = infer_cases_resumable(
            client,
            model,
            variant_cases,
            SEED,
            TEMPERATURE,
            partial_path,
            resume=resume,
            progress_prefix=prefix,
        )
        write_jsonl(final_path, rows)
        partial_path.unlink(missing_ok=True)
        return rows

    control_predictions = predictions_for("control", control_cases, "Control")
    reorder_predictions = predictions_for("reorder", reorder_cases, "Reorder")
    distractor_predictions = predictions_for("distractor", distractor_cases, "Distractor")

    metrics = calculate_robustness_metrics(
        control_predictions, reorder_predictions, distractor_predictions
    )
    rows = _case_rows(
        control_predictions,
        reorder_predictions,
        distractor_predictions,
        {case["fever_id"] for case in reorder_cases},
    )
    _write_case_csv(results_dir / "robustness_by_case.csv", rows)
    config = {
        "experiment": "FEVER evidence-perturbation robustness",
        "prepared_dataset_path": str(prepared_path),
        "prepared_dataset_sha256": dataset_hash,
        "expected_dataset_sha256": EXPECTED_DATASET_SHA256,
        "full_dataset_size": len(source),
        "full_dataset_class_balance": dict(Counter(row["gold_label"] for row in source)),
        "sample_size": len(control_cases),
        "reorder_eligible_cases": len(reorder_cases),
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
    report = _report(metrics, config)
    flagged_ids = {row["fever_id"] for row in control_predictions if row.get("flags")}
    excluded_path = results_dir / "robustness_metrics_excluding_flagged.json"
    if flagged_ids:

        def kept(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [row for row in rows if row["fever_id"] not in flagged_ids]

        excluded = calculate_robustness_metrics(
            kept(control_predictions), kept(reorder_predictions), kept(distractor_predictions)
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
