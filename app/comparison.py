"""Small, read-only adapters for the four full evaluation experiments."""

from __future__ import annotations

from pathlib import Path

from llm_fact_checker.results import DEFAULT_RESULTS_DIR, ResultRun, discover_runs

EXPERIMENTS = {
    "baseline": ("fever", "baseline_full", "Correctness"),
    "consistency": ("fever", "consistency_5run_full", "Consistency"),
    "robustness": ("fever", "robustness_full", "Robustness"),
    "sensitivity": ("vitaminc", "sensitivity_full", "Sensitivity"),
}


def load_models(root: Path = DEFAULT_RESULTS_DIR) -> dict[str, dict[str, ResultRun]]:
    """Keep full runs only; never substitute a pilot or smoke result."""
    models: dict[str, dict[str, ResultRun]] = {}
    for run in discover_runs(root):
        dataset, experiment, _ = EXPERIMENTS[run.kind]
        if (run.dataset, run.experiment) == (dataset, experiment):
            models.setdefault(run.model, {})[run.kind] = run
    return models


def comparison_issues(left: ResultRun, right: ResultRun) -> list[str]:
    """Check recorded inputs and settings before displaying a score difference."""
    configs = [run.config() for run in (left, right)]
    provenance = [run.provenance() for run in (left, right)]
    checks = {
        "experiment": [run.kind for run in (left, right)],
        "dataset fingerprint": [item["dataset_sha256"] for item in provenance],
        "prompt": [item.get("prompt_template") for item in configs],
        "temperature": [item["temperature"] for item in provenance],
        "seeds": [item["seeds"] for item in provenance],
        "case count": [item.get("sample_size", item.get("sample_size_pairs")) for item in configs],
    }
    if left.kind == "sensitivity":
        for key in ("pair_selection", "pairing_method"):
            checks[key.replace("_", " ")] = [item.get(key) for item in configs]
    if left.kind == "robustness":
        checks["reordering case count"] = [item.get("reorder_eligible_cases") for item in configs]
    return [
        f"{name}: {'missing information' if any(v is None for v in values) else 'different values'}"
        for name, values in checks.items()
        if any(value is None for value in values) or values[0] != values[1]
    ]


def headline_score(kind: str, metrics: dict) -> tuple[str, float | None]:
    """Read the result card's score without re-scoring saved answers."""
    if kind == "baseline":
        return "Overall accuracy", metrics["overall_accuracy"]
    if kind == "consistency":
        return "Same answer in all five runs", metrics["exact_consistency_rate"]
    if kind == "sensitivity":
        return "Both evidence versions correct", metrics["correct_sensitivity_rate"]
    raise ValueError(f"Unsupported headline score: {kind}")
