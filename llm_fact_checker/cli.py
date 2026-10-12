"""Command-line interface for the FEVER and VitaminC experiments."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from llm_fact_checker.data_prep.fever import DEV_PATH, PREPARED_PATH, WIKI_DIR, prepare_dataset
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

BASE_RESULTS_DIR = Path("results/fever/llama3.2_3b/baseline")
CONSISTENCY_RESULTS_DIR = Path("results/fever/llama3.2_3b/consistency_5run")
ROBUSTNESS_RESULTS_DIR = Path("results/fever/llama3.2_3b/robustness")
VITAMINC_RESULTS_DIR = Path("results/vitaminc/llama3.2_3b/sensitivity")


def _results_dir(limit: int | None, requested: Path | None) -> Path:
    if requested:
        return requested
    return BASE_RESULTS_DIR if limit is None else BASE_RESULTS_DIR.parent / f"smoke_{limit}"


def _prepare(args: argparse.Namespace) -> None:
    prepared = prepare_dataset(args.dev_path, args.wiki_dir, args.output, seed=42)
    counts = Counter(item["gold_label"] for item in prepared)
    print(f"Prepared {len(prepared)} cases: {dict(counts)}")
    for item in prepared[:3]:
        print(json.dumps(item, ensure_ascii=False, indent=2))


def _run(args: argparse.Namespace) -> None:
    results_dir = _results_dir(getattr(args, "limit", None), args.results_dir)
    metrics = run_inference(
        args.prepared,
        results_dir,
        getattr(args, "limit", None),
        args.overwrite,
        args.ollama_url,
    )
    print(json.dumps(metrics, indent=2))
    print(f"Results written to {results_dir}")


def _evaluate(args: argparse.Namespace) -> None:
    print(json.dumps(evaluate_results(args.results_dir), indent=2))


def _consistency(args: argparse.Namespace) -> None:
    results_dir = args.results_dir or (
        CONSISTENCY_RESULTS_DIR
        if args.limit is None
        else CONSISTENCY_RESULTS_DIR.parent / f"consistency_5run_smoke_{args.limit}"
    )
    metrics = run_consistency_experiment(
        args.prepared,
        results_dir,
        limit=args.limit,
        overwrite=args.overwrite,
        base_url=args.ollama_url,
    )
    print(json.dumps(metrics, indent=2))
    print(f"Consistency results written to {results_dir}")


def _robustness_prepare(args: argparse.Namespace) -> None:
    summary = prepare_robustness_files(
        args.prepared,
        args.robustness_dir / "prepared",
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))
    print(f"Prepared robustness files written to {args.robustness_dir / 'prepared'}")


def _robustness(args: argparse.Namespace) -> None:
    metrics = run_robustness_experiment(
        args.prepared,
        args.robustness_dir,
        limit=args.limit,
        overwrite=args.overwrite,
        base_url=args.ollama_url,
    )
    print(json.dumps(metrics, indent=2))
    destination = args.robustness_dir if args.limit is None else args.robustness_dir / "smoke"
    print(f"Robustness results written to {destination}")


def _vitaminc_prepare(args: argparse.Namespace) -> None:
    summary = prepare_vitaminc_pairs(
        args.raw,
        args.processed,
        args.sensitivity_dir / "prepared",
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))
    print(f"Prepared VitaminC pairs written to {args.sensitivity_dir / 'prepared'}")


def _vitaminc_sensitivity(args: argparse.Namespace) -> None:
    metrics = run_vitaminc_sensitivity(
        args.raw,
        args.processed,
        args.sensitivity_dir,
        limit=args.limit,
        overwrite=args.overwrite,
        base_url=args.ollama_url,
    )
    print(json.dumps(metrics, indent=2))
    destination = args.sensitivity_dir if args.limit is None else args.sensitivity_dir / "smoke"
    print(f"VitaminC sensitivity results written to {destination}")


def _all(args: argparse.Namespace) -> None:
    _prepare(args)
    args.prepared = args.output
    _run(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m llm_fact_checker", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="prepare the balanced FEVER dataset")
    prepare.add_argument("--dev-path", type=Path, default=DEV_PATH)
    prepare.add_argument("--wiki-dir", type=Path, default=WIKI_DIR)
    prepare.add_argument("--output", type=Path, default=PREPARED_PATH)
    prepare.set_defaults(function=_prepare)

    run = subparsers.add_parser("run", help="run inference and evaluate results")
    run.add_argument("--prepared", type=Path, default=PREPARED_PATH)
    run.add_argument("--results-dir", type=Path)
    run.add_argument("--limit", type=int)
    run.add_argument("--ollama-url", default="http://localhost:11434")
    run.add_argument("--overwrite", action="store_true")
    run.set_defaults(function=_run)

    evaluate = subparsers.add_parser("evaluate", help="regenerate metrics and report")
    evaluate.add_argument("--results-dir", type=Path, default=BASE_RESULTS_DIR)
    evaluate.set_defaults(function=_evaluate)

    consistency = subparsers.add_parser(
        "consistency", help="run five seeded repetitions on the fixed FEVER dataset"
    )
    consistency.add_argument("--prepared", type=Path, default=PREPARED_PATH)
    consistency.add_argument("--results-dir", type=Path)
    consistency.add_argument("--limit", type=int)
    consistency.add_argument("--ollama-url", default="http://localhost:11434")
    consistency.add_argument("--overwrite", action="store_true")
    consistency.set_defaults(function=_consistency)

    robustness_prepare = subparsers.add_parser(
        "robustness-prepare", help="prepare deterministic evidence perturbations for audit"
    )
    robustness_prepare.add_argument("--prepared", type=Path, default=PREPARED_PATH)
    robustness_prepare.add_argument("--robustness-dir", type=Path, default=ROBUSTNESS_RESULTS_DIR)
    robustness_prepare.add_argument("--overwrite", action="store_true")
    robustness_prepare.set_defaults(function=_robustness_prepare)

    robustness_smoke = subparsers.add_parser(
        "robustness-smoke", help="run the isolated three-case robustness smoke test"
    )
    robustness_smoke.add_argument("--prepared", type=Path, default=PREPARED_PATH)
    robustness_smoke.add_argument("--robustness-dir", type=Path, default=ROBUSTNESS_RESULTS_DIR)
    robustness_smoke.add_argument("--ollama-url", default="http://localhost:11434")
    robustness_smoke.add_argument("--overwrite", action="store_true")
    robustness_smoke.set_defaults(function=_robustness, limit=3)

    robustness = subparsers.add_parser(
        "robustness", help="run the full audited evidence-perturbation experiment"
    )
    robustness.add_argument("--prepared", type=Path, default=PREPARED_PATH)
    robustness.add_argument("--robustness-dir", type=Path, default=ROBUSTNESS_RESULTS_DIR)
    robustness.add_argument("--ollama-url", default="http://localhost:11434")
    robustness.add_argument("--overwrite", action="store_true")
    robustness.set_defaults(function=_robustness, limit=None)

    vitaminc_prepare = subparsers.add_parser(
        "vitaminc-prepare", help="prepare and audit 100 contrastive VitaminC pairs"
    )
    vitaminc_prepare.add_argument("--raw", type=Path, default=VITAMINC_RAW_PATH)
    vitaminc_prepare.add_argument("--processed", type=Path, default=VITAMINC_PROCESSED_PATH)
    vitaminc_prepare.add_argument("--sensitivity-dir", type=Path, default=VITAMINC_RESULTS_DIR)
    vitaminc_prepare.add_argument("--overwrite", action="store_true")
    vitaminc_prepare.set_defaults(function=_vitaminc_prepare)

    vitaminc_smoke = subparsers.add_parser(
        "vitaminc-smoke", help="run the isolated three-pair VitaminC sensitivity smoke test"
    )
    vitaminc_smoke.add_argument("--raw", type=Path, default=VITAMINC_RAW_PATH)
    vitaminc_smoke.add_argument("--processed", type=Path, default=VITAMINC_PROCESSED_PATH)
    vitaminc_smoke.add_argument("--sensitivity-dir", type=Path, default=VITAMINC_RESULTS_DIR)
    vitaminc_smoke.add_argument("--ollama-url", default="http://localhost:11434")
    vitaminc_smoke.add_argument("--overwrite", action="store_true")
    vitaminc_smoke.set_defaults(function=_vitaminc_sensitivity, limit=3)

    vitaminc_sensitivity = subparsers.add_parser(
        "vitaminc-sensitivity", help="run the full audited 100-pair VitaminC experiment"
    )
    vitaminc_sensitivity.add_argument("--raw", type=Path, default=VITAMINC_RAW_PATH)
    vitaminc_sensitivity.add_argument("--processed", type=Path, default=VITAMINC_PROCESSED_PATH)
    vitaminc_sensitivity.add_argument("--sensitivity-dir", type=Path, default=VITAMINC_RESULTS_DIR)
    vitaminc_sensitivity.add_argument("--ollama-url", default="http://localhost:11434")
    vitaminc_sensitivity.add_argument("--overwrite", action="store_true")
    vitaminc_sensitivity.set_defaults(function=_vitaminc_sensitivity, limit=None)

    all_steps = subparsers.add_parser("all", help="prepare, run all 150 cases, and evaluate")
    all_steps.add_argument("--dev-path", type=Path, default=DEV_PATH)
    all_steps.add_argument("--wiki-dir", type=Path, default=WIKI_DIR)
    all_steps.add_argument("--output", type=Path, default=PREPARED_PATH)
    all_steps.add_argument("--results-dir", type=Path)
    all_steps.add_argument("--ollama-url", default="http://localhost:11434")
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
