"""FEVER gold-evidence baseline: one deterministic pass over the fixed sample."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
from typing import Any

from llm_fact_checker.common import file_sha256, git_commit, read_jsonl, write_jsonl
from llm_fact_checker.data_prep.fever import DEV_PATH
from llm_fact_checker.evaluation import evaluate_results
from llm_fact_checker.models import DEFAULT_MODEL
from llm_fact_checker.ollama import OllamaClient, infer_cases
from llm_fact_checker.prompts import SYSTEM_PROMPT


def run_inference(
    prepared_path: Path,
    results_dir: Path,
    limit: int | None = None,
    overwrite: bool = False,
    base_url: str = "http://localhost:11434",
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    predictions_path = results_dir / "predictions.jsonl"
    if predictions_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {predictions_path}; pass --overwrite explicitly"
        )
    dataset = list(read_jsonl(prepared_path))
    if not dataset:
        raise ValueError(f"Prepared dataset is empty: {prepared_path}")
    cases = dataset if limit is None else dataset[:limit]
    client = OllamaClient(base_url=base_url)
    ollama_metadata = client.check(model)
    results = infer_cases(client, model, cases, seed=42, temperature=0)
    results_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(predictions_path, results)
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
