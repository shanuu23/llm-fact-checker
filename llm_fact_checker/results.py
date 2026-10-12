"""Read-only access to committed experiment results.

Discovers ``results/<dataset>/<model_slug>/<experiment>/`` folders and exposes their metrics,
run configuration and per-case tables as plain Python data. It uses only the standard library,
never writes, and never calls a model, so a comparison page or the Streamlit app can import it
safely::

    from llm_fact_checker.results import discover_runs, select

    runs = select(discover_runs(), kind="baseline")
    for run in runs:
        print(run.model, run.headline()["accuracy"])

The per-case tables are lists of dicts, so ``pandas.DataFrame(run.cases())`` works where pandas
is available.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from llm_fact_checker.common import read_jsonl

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

NEI = "NOT ENOUGH INFORMATION"

# Experiment kind -> the metrics file that identifies it. ``metrics.json`` is checked last
# because only the baseline uses that generic name.
_METRICS_FILES = {
    "consistency": "consistency_metrics.json",
    "robustness": "robustness_metrics.json",
    "sensitivity": "sensitivity_metrics.json",
    "baseline": "metrics.json",
}
_CASE_FILES = {
    "baseline": "predictions.jsonl",
    "consistency": "consistency_by_case.csv",
    "robustness": "robustness_by_case.csv",
    "sensitivity": "sensitivity_by_pair.csv",
}
# CSV cells are text. Only the columns listed here are converted; everything else stays a string
# (so a claim such as "1990" is never turned into a number). An empty cell is always None.
_COLUMN_TYPES: dict[str, dict[str, type]] = {
    "consistency": {
        "fever_id": int,
        "unique_label_count": int,
        "modal_count": int,
        "modal_agreement": float,
        "exact_agreement": bool,
        "tie": bool,
        "incomplete": bool,
    },
    "robustness": {
        "fever_id": int,
        "reorder_eligible": bool,
        "reorder_prediction_changed": bool,
        "distractor_prediction_changed": bool,
    },
    "sensitivity": {
        "complete": bool,
        "prediction_changed": bool,
        "correct_sensitivity": bool,
        "support_valid_format": bool,
        "refute_valid_format": bool,
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing results file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _convert(value: str, column_type: type | None) -> Any:
    if value == "":
        return None
    if column_type is None:
        return value
    if column_type is bool:
        if value not in ("True", "False"):
            raise ValueError(f"Expected True or False, found {value!r}")
        return value == "True"
    return column_type(value)


def _detect_kind(directory: Path) -> str | None:
    for kind, filename in _METRICS_FILES.items():
        if (directory / filename).is_file():
            return kind
    return None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def _headline(kind: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """The few numbers used to compare models, with stable names across models."""
    if kind == "baseline":
        total = metrics["total_cases"]
        return {
            "cases": total,
            "accuracy": metrics["overall_accuracy"],
            "supported_accuracy": metrics["class_accuracy"]["SUPPORTED"],
            "refuted_accuracy": metrics["class_accuracy"]["REFUTED"],
            "nei_rate": _ratio(metrics["prediction_counts"].get(NEI, 0), total),
            "invalid_format_rate": metrics["invalid_format_rate"],
        }
    if kind == "consistency":
        states = metrics["stability_classification"]
        return {
            "cases": metrics["total_cases"],
            "exact_consistency": metrics["exact_consistency_rate"],
            "mean_modal_agreement": metrics["mean_modal_agreement"],
            "mean_accuracy": metrics["accuracy_summary"]["mean"],
            "stable_wrong_rate": states["STABLE_WRONG"]["percentage"],
            "unstable_rate": states["UNSTABLE"]["percentage"],
        }
    if kind == "robustness":
        reorder, distractor = metrics["reorder"], metrics["distractor"]
        return {
            "cases": metrics["case_count"],
            "control_accuracy": metrics["control"]["accuracy"],
            "reorder_eligible": reorder["eligible_pair_count"],
            "reorder_invariance": reorder["invariance_rate"],
            "distractor_invariance": distractor["invariance_rate"],
            "distractor_accuracy": distractor["perturbed_accuracy"],
            "distractor_accuracy_delta": distractor["accuracy_delta"],
        }
    if kind == "sensitivity":
        total = metrics["total_pairs"]
        return {
            "pairs": total,
            "correct_sensitivity": metrics["correct_sensitivity_rate"],
            "flip_rate": metrics["flip_rate"],
            "support_accuracy": metrics["individual_accuracy"]["support_accuracy"],
            "refute_accuracy": metrics["individual_accuracy"]["refute_accuracy"],
            "nei_pair_rate": _ratio(metrics["nei_analysis"]["pairs_with_nei_either_side"], total),
        }
    raise ValueError(f"Unknown experiment kind: {kind!r}")


@dataclass(frozen=True)
class ResultRun:
    """One experiment's results for one model on one dataset."""

    dataset: str
    model: str
    experiment: str
    kind: str
    path: Path
    is_smoke: bool = False

    @property
    def label(self) -> str:
        return f"{self.dataset}/{self.model}/{self.experiment}"

    def metrics(self) -> dict[str, Any]:
        return _read_json(self.path / _METRICS_FILES[self.kind])

    def config(self) -> dict[str, Any]:
        return _read_json(self.path / "run_config.json")

    def headline(self) -> dict[str, Any]:
        return _headline(self.kind, self.metrics())

    def provenance(self) -> dict[str, Any]:
        """What produced these numbers, for display and for checking runs are comparable."""
        config = self.config()
        model_meta = config.get("ollama_model_metadata") or {}
        seeds = config.get("run_seeds")
        if seeds is None and config.get("model_seed") is not None:
            seeds = [config["model_seed"]]
        return {
            "model_tag": config.get("ollama_model"),
            "model_digest": model_meta.get("digest"),
            "ollama_version": config.get("ollama_version"),
            "temperature": config.get("temperature"),
            "seeds": seeds,
            "dataset_sha256": config.get("prepared_dataset_sha256")
            or config.get("raw_test_sha256"),
            "git_commit": config.get("git_commit"),
            "timestamp_utc": config.get("run_timestamp_utc"),
        }

    def cases(self) -> list[dict[str, Any]]:
        """The per-case table: one dict per claim (or per pair for VitaminC)."""
        path = self.path / _CASE_FILES[self.kind]
        if not path.is_file():
            raise FileNotFoundError(f"Missing results file: {path}")
        if path.suffix == ".jsonl":
            return list(read_jsonl(path))
        types = _COLUMN_TYPES[self.kind]
        with path.open(newline="", encoding="utf-8") as handle:
            return [
                {column: _convert(value, types.get(column)) for column, value in row.items()}
                for row in csv.DictReader(handle)
            ]


def _is_smoke(name: str) -> bool:
    return name.startswith("smoke") or "_smoke" in name


def _run(dataset: str, model: str, experiment: str, path: Path, smoke: bool) -> ResultRun | None:
    kind = _detect_kind(path)
    if kind is None:
        return None
    return ResultRun(dataset, model, experiment, kind, path, is_smoke=smoke)


def discover_runs(
    root: Path | str = DEFAULT_RESULTS_DIR, *, include_smoke: bool = False
) -> list[ResultRun]:
    """Find every experiment folder under ``root``, sorted by dataset, model and experiment.

    Folders without a recognised metrics file are skipped. Smoke-test folders (``smoke_3``,
    ``consistency_5run_smoke_3``, and ``<experiment>/smoke``) are 3-case development checks
    and are left out unless ``include_smoke`` is true.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Results directory not found: {root}")
    runs: list[ResultRun] = []
    for dataset in sorted(p for p in root.iterdir() if p.is_dir()):
        for model in sorted(p for p in dataset.iterdir() if p.is_dir()):
            for experiment in sorted(p for p in model.iterdir() if p.is_dir()):
                smoke = _is_smoke(experiment.name)
                if smoke and not include_smoke:
                    continue
                found = _run(dataset.name, model.name, experiment.name, experiment, smoke)
                if found is None:
                    continue
                runs.append(found)
                nested = experiment / "smoke"
                if include_smoke and nested.is_dir():
                    inner = _run(
                        dataset.name, model.name, f"{experiment.name}/smoke", nested, True
                    )
                    if inner is not None:
                        runs.append(inner)
    return sorted(runs, key=lambda run: (run.dataset, run.model, run.experiment))


def select(
    runs: list[ResultRun],
    *,
    dataset: str | None = None,
    model: str | None = None,
    kind: str | None = None,
) -> list[ResultRun]:
    """Filter runs by any combination of dataset, model slug and experiment kind."""
    return [
        run
        for run in runs
        if (dataset is None or run.dataset == dataset)
        and (model is None or run.model == model)
        and (kind is None or run.kind == kind)
    ]
