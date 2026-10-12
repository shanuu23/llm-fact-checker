"""FEVER gold-evidence baseline: one deterministic pass over the fixed sample."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import sys
from typing import Any

from llm_fact_checker.common import file_sha256, git_commit, read_jsonl, write_jsonl
from llm_fact_checker.data_prep.fever import DEV_PATH
from llm_fact_checker.evaluation import evaluate_results
from llm_fact_checker.models import DEFAULT_MODEL
from llm_fact_checker.ollama import OllamaClient, infer_cases_resumable
from llm_fact_checker.prompts import SYSTEM_PROMPT


def run_inference(
    prepared_path: Path,
    results_dir: Path,
    limit: int | None = None,
    overwrite: bool = False,
    base_url: str = "http://localhost:11434",
    model: str = DEFAULT_MODEL,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the baseline. Each answer is saved as it arrives (``predictions.partial.jsonl``).

    If a run is interrupted, ``resume=True`` keeps every answer already saved, retries cases
    that failed, and runs only what is missing. The partial file is removed once
    ``predictions.jsonl`` has been written.
    """
    predictions_path = results_dir / "predictions.jsonl"
    partial_path = results_dir / "predictions.partial.jsonl"
    metrics_path = results_dir / "metrics.json"
    if resume and predictions_path.exists() and metrics_path.exists():
        print(f"Already complete, nothing to do: {results_dir}", file=sys.stderr)
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    if predictions_path.exists() and not (overwrite or resume):
        raise FileExistsError(
            f"Refusing to overwrite {predictions_path}; pass --overwrite explicitly"
        )
    if partial_path.exists() and not (resume or overwrite):
        raise FileExistsError(
            f"{partial_path} holds an interrupted run; pass --resume to continue it "
            "or --overwrite to discard it"
        )
    dataset = list(read_jsonl(prepared_path))
    if not dataset:
        raise ValueError(f"Prepared dataset is empty: {prepared_path}")
    cases = dataset if limit is None else dataset[:limit]
    client = OllamaClient(base_url=base_url)
    ollama_metadata = client.check(model)
    results_dir.mkdir(parents=True, exist_ok=True)

    finished = list(read_jsonl(predictions_path)) if resume and predictions_path.exists() else None
    if finished is not None and len(finished) == len(cases):
        results = finished  # inference finished; only scoring is missing
    else:
        results = infer_cases_resumable(client, model, cases, 42, 0, partial_path, resume=resume)
        write_jsonl(predictions_path, results)
        partial_path.unlink(missing_ok=True)
    counts = Counter(case["gold_label"] for case in cases)
    config = {
        "dataset_name": "FEVER shared_task_dev gold-evidence baseline",
        "raw_dataset_path": str(DEV_PATH),
        "prepared_dataset_path": str(prepared_path),
        "prepared_dataset_sha256": file_sha256(prepared_path),
        "sample_size": len(cases),
        "class_balance": dict(counts),
        "sampling_seed": 42,
        "ollama_model": model,
        "temperature": 0,
        "model_seed": 42,
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
    return evaluate_results(results_dir)
