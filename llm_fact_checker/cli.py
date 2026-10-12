"""Command-line interface for the FEVER and VitaminC experiments."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from llm_fact_checker.data_prep.fever import (
    DEV_PATH,
    FULL_PATH,
    PREPARED_PATH,
    WIKI_DIR,
    prepare_dataset,
)
from llm_fact_checker.data_prep.vitaminc import (
    VITAMINC_PROCESSED_PATH,
    VITAMINC_RAW_PATH,
    prepare_vitaminc_pairs,
)
from llm_fact_checker.evaluation import evaluate_results
from llm_fact_checker.experiments.baseline import run_inference
from llm_fact_checker.experiments.consistency import run_consistency_experiment
from llm_fact_checker.experiments.robustness import (
    prepare_robustness_files,
    run_robustness_experiment,
)
from llm_fact_checker.experiments.sensitivity import run_vitaminc_sensitivity
from llm_fact_checker.models import DEFAULT_MODEL, model_slug

RESULTS_ROOT = Path("results")
# Perturbed inputs and audit files are derived from the frozen samples, not from any model, so
# they live with the data and are shared by every model's run.
FEVER_PREPARED_DIR = Path("data/fever/prepared")
VITAMINC_PREPARED_DIR = Path("data/vitaminc/prepared")
OLLAMA_URL = "http://localhost:11434"


def experiment_dir(dataset: str, model: str, experiment: str) -> Path:
    """``results/<dataset>/<model_slug>/<experiment>`` for a model tag such as ``qwen2.5:7b``."""
    return RESULTS_ROOT / dataset / model_slug(model) / experiment


def _results_dir(limit: int | None, requested: Path | None, model: str = DEFAULT_MODEL) -> Path:
    if requested:
        return requested
    base = experiment_dir("fever", model, "baseline")
    return base if limit is None else base.parent / f"smoke_{limit}"


def _prepare(args: argparse.Namespace) -> None:
    whole = getattr(args, "all", False)
    output = args.output or (FULL_PATH if whole else PREPARED_PATH)
    prepared = prepare_dataset(args.dev_path, args.wiki_dir, output, seed=42, whole=whole)
    counts = Counter(item["gold_label"] for item in prepared)
    print(f"Prepared {len(prepared)} cases: {dict(counts)} -> {output}")
    if whole:
        flagged = sum(bool(item["flags"]) for item in prepared)
        print(f"{flagged} cases carry an audit flag (kept in the file; reported separately)")
    for item in prepared[:3]:
        print(json.dumps(item, ensure_ascii=False, indent=2))


def _run(args: argparse.Namespace) -> None:
    limit = getattr(args, "limit", None)
    results_dir = _results_dir(limit, args.results_dir, args.model)
    metrics = run_inference(
        args.prepared,
        results_dir,
        limit,
        args.overwrite,
        args.ollama_url,
        model=args.model,
        resume=getattr(args, "resume", False),
    )
    print(json.dumps(metrics, indent=2))
    print(f"Results written to {results_dir}")


def _evaluate(args: argparse.Namespace) -> None:
    results_dir = args.results_dir or experiment_dir("fever", args.model, "baseline")
    print(json.dumps(evaluate_results(results_dir), indent=2))


def _consistency(args: argparse.Namespace) -> None:
    base = experiment_dir("fever", args.model, "consistency_5run")
    results_dir = args.results_dir or (
        base if args.limit is None else base.parent / f"consistency_5run_smoke_{args.limit}"
    )
    metrics = run_consistency_experiment(
        args.prepared,
        results_dir,
        limit=args.limit,
        overwrite=args.overwrite,
        base_url=args.ollama_url,
        model=args.model,
        resume=args.resume,
    )
    print(json.dumps(metrics, indent=2))
    print(f"Consistency results written to {results_dir}")


def _robustness_prepare(args: argparse.Namespace) -> None:
    summary = prepare_robustness_files(args.prepared, args.output_dir, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2))
    print(f"Prepared robustness files written to {args.output_dir}")


def _robustness(args: argparse.Namespace) -> None:
    robustness_dir = args.robustness_dir or experiment_dir("fever", args.model, "robustness")
    metrics = run_robustness_experiment(
        args.prepared,
        robustness_dir,
        limit=args.limit,
        overwrite=args.overwrite,
        base_url=args.ollama_url,
        model=args.model,
        prepared_dir=args.prepared_dir,
        resume=args.resume,
    )
    print(json.dumps(metrics, indent=2))
    destination = robustness_dir if args.limit is None else robustness_dir / "smoke"
    print(f"Robustness results written to {destination}")


def _vitaminc_prepare(args: argparse.Namespace) -> None:
    summary = prepare_vitaminc_pairs(
        args.raw,
        args.processed,
        args.prepared_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))
    print(f"Prepared VitaminC pairs written to {args.processed}; audit in {args.prepared_dir}")


def _vitaminc_sensitivity(args: argparse.Namespace) -> None:
    sensitivity_dir = args.sensitivity_dir or experiment_dir("vitaminc", args.model, "sensitivity")
    metrics = run_vitaminc_sensitivity(
        args.raw,
        args.processed,
        sensitivity_dir,
        limit=args.limit,
        overwrite=args.overwrite,
        base_url=args.ollama_url,
        model=args.model,
    )
    print(json.dumps(metrics, indent=2))
    destination = sensitivity_dir if args.limit is None else sensitivity_dir / "smoke"
    print(f"VitaminC sensitivity results written to {destination}")


def _all(args: argparse.Namespace) -> None:
    _prepare(args)
    args.prepared = args.output
    _run(args)


def _add_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model tag, exactly as `ollama list` shows it (default: {DEFAULT_MODEL}); "
        "results go to results/<dataset>/<model_slug>/...",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m llm_fact_checker", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="prepare the FEVER sample or the whole set")
    prepare.add_argument("--dev-path", type=Path, default=DEV_PATH)
    prepare.add_argument("--wiki-dir", type=Path, default=WIKI_DIR)
    prepare.add_argument(
        "--all",
        action="store_true",
        help="every SUPPORTS/REFUTES claim (13,332), nothing de-duplicated, audit-flagged claims "
        "marked, accented titles matched; writes data/fever/full/fever_dev_full.jsonl",
    )
    prepare.add_argument(
        "--output",
        type=Path,
        help=f"default: {PREPARED_PATH} for the sample, {FULL_PATH} with --all",
    )
    prepare.set_defaults(function=_prepare)

    run = subparsers.add_parser("run", help="run inference and evaluate results")
    run.add_argument("--prepared", type=Path, default=PREPARED_PATH)
    _add_model(run)
    run.add_argument("--results-dir", type=Path)
    run.add_argument("--limit", type=int)
    run.add_argument("--ollama-url", default=OLLAMA_URL)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted run: keep saved answers, retry failures, run the rest",
    )
    run.set_defaults(function=_run)

    evaluate = subparsers.add_parser("evaluate", help="regenerate metrics and report")
    _add_model(evaluate)
    evaluate.add_argument(
        "--results-dir",
        type=Path,
        help="results folder to re-score (default: the baseline folder of --model)",
    )
    evaluate.set_defaults(function=_evaluate)

    consistency = subparsers.add_parser(
        "consistency", help="run five seeded repetitions on the fixed FEVER dataset"
    )
    consistency.add_argument("--prepared", type=Path, default=PREPARED_PATH)
    _add_model(consistency)
    consistency.add_argument("--results-dir", type=Path)
    consistency.add_argument("--limit", type=int)
    consistency.add_argument("--ollama-url", default=OLLAMA_URL)
    consistency.add_argument("--overwrite", action="store_true")
    consistency.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted run: keep saved answers, retry failures, run the rest; "
        "does nothing if the experiment already finished",
    )
    consistency.set_defaults(function=_consistency)

    robustness_prepare = subparsers.add_parser(
        "robustness-prepare", help="prepare deterministic evidence perturbations for audit"
    )
    robustness_prepare.add_argument("--prepared", type=Path, default=PREPARED_PATH)
    robustness_prepare.add_argument("--output-dir", type=Path, default=FEVER_PREPARED_DIR)
    robustness_prepare.add_argument("--overwrite", action="store_true")
    robustness_prepare.set_defaults(function=_robustness_prepare)

    for name, help_text, limit in (
        ("robustness-smoke", "run the isolated three-case robustness smoke test", 3),
        ("robustness", "run the full audited evidence-perturbation experiment", None),
    ):
        robustness = subparsers.add_parser(name, help=help_text)
        robustness.add_argument("--prepared", type=Path, default=PREPARED_PATH)
        _add_model(robustness)
        robustness.add_argument("--robustness-dir", type=Path)
        robustness.add_argument("--prepared-dir", type=Path, default=FEVER_PREPARED_DIR)
        robustness.add_argument("--ollama-url", default=OLLAMA_URL)
        robustness.add_argument("--overwrite", action="store_true")
        robustness.add_argument(
            "--resume",
            action="store_true",
            help="continue an interrupted run: keep saved answers, retry failures, run the rest; "
            "does nothing if the experiment already finished",
        )
        robustness.set_defaults(function=_robustness, limit=limit)

    vitaminc_prepare = subparsers.add_parser(
        "vitaminc-prepare", help="prepare and audit 100 contrastive VitaminC pairs"
    )
    vitaminc_prepare.add_argument("--raw", type=Path, default=VITAMINC_RAW_PATH)
    vitaminc_prepare.add_argument("--processed", type=Path, default=VITAMINC_PROCESSED_PATH)
    vitaminc_prepare.add_argument("--prepared-dir", type=Path, default=VITAMINC_PREPARED_DIR)
    vitaminc_prepare.add_argument("--overwrite", action="store_true")
    vitaminc_prepare.set_defaults(function=_vitaminc_prepare)

    for name, help_text, limit in (
        (
            "vitaminc-smoke",
            "run the isolated three-pair VitaminC sensitivity smoke test",
            3,
        ),
        (
            "vitaminc-sensitivity",
            "run the full audited 100-pair VitaminC experiment",
            None,
        ),
    ):
        sensitivity = subparsers.add_parser(name, help=help_text)
        sensitivity.add_argument("--raw", type=Path, default=VITAMINC_RAW_PATH)
        sensitivity.add_argument("--processed", type=Path, default=VITAMINC_PROCESSED_PATH)
        _add_model(sensitivity)
        sensitivity.add_argument("--sensitivity-dir", type=Path)
        sensitivity.add_argument("--ollama-url", default=OLLAMA_URL)
        sensitivity.add_argument("--overwrite", action="store_true")
        sensitivity.set_defaults(function=_vitaminc_sensitivity, limit=limit)

    all_steps = subparsers.add_parser("all", help="prepare, run all 150 cases, and evaluate")
    all_steps.add_argument("--dev-path", type=Path, default=DEV_PATH)
    all_steps.add_argument("--wiki-dir", type=Path, default=WIKI_DIR)
    all_steps.add_argument("--output", type=Path, default=PREPARED_PATH)
    _add_model(all_steps)
    all_steps.add_argument("--results-dir", type=Path)
    all_steps.add_argument("--ollama-url", default=OLLAMA_URL)
    all_steps.add_argument("--overwrite", action="store_true")
    all_steps.set_defaults(function=_all)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "limit", None) is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    args.function(args)


if __name__ == "__main__":
    main()
