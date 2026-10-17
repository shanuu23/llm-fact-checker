#!/usr/bin/env python3
"""Cross-Model Benchmark Comparison and Error Analysis.

Reads committed results under results/ via the read-only loader
llm_fact_checker.results and generates comparative scorecards,
disagreement analysis, and optional figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from llm_fact_checker.results import discover_runs, select


def generate_scorecard_data() -> list[dict[str, Any]]:
    runs = discover_runs()
    models = sorted({r.model for r in runs if not r.is_smoke and not r.experiment.endswith("_full")})

    rows = []
    for m in models:
        b_runs = [r for r in select(runs, model=m, kind="baseline") if r.experiment == "baseline"]
        c_runs = [
            r for r in select(runs, model=m, kind="consistency") if r.experiment == "consistency_5run"
        ]
        r_runs = [
            r for r in select(runs, model=m, kind="robustness") if r.experiment == "robustness"
        ]
        s_runs = [
            r for r in select(runs, model=m, kind="sensitivity") if r.experiment == "sensitivity"
        ]

        b_head = b_runs[0].headline() if b_runs else {}
        c_head = c_runs[0].headline() if c_runs else {}
        r_head = r_runs[0].headline() if r_runs else {}
        s_head = s_runs[0].headline() if s_runs else {}

        rows.append(
            {
                "model": m,
                "fever_accuracy": b_head.get("accuracy"),
                "fever_supported_acc": b_head.get("supported_accuracy"),
                "fever_refuted_acc": b_head.get("refuted_accuracy"),
                "fever_nei_rate": b_head.get("nei_rate"),
                "consistency_5_of_5": c_head.get("exact_consistency"),
                "modal_agreement": c_head.get("mean_modal_agreement"),
                "stable_wrong_rate": c_head.get("stable_wrong_rate"),
                "unstable_rate": c_head.get("unstable_rate"),
                "reorder_invariance": r_head.get("reorder_invariance"),
                "distractor_invariance": r_head.get("distractor_invariance"),
                "distractor_accuracy": r_head.get("distractor_accuracy"),
                "distractor_delta": r_head.get("distractor_accuracy_delta"),
                "sensitivity_correct": s_head.get("correct_sensitivity"),
                "sensitivity_flip_rate": s_head.get("flip_rate"),
                "sensitivity_support_acc": s_head.get("support_accuracy"),
                "sensitivity_refute_acc": s_head.get("refute_accuracy"),
            }
        )
    return rows


def format_pct(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def print_markdown_scorecard(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Cross-Model Fact-Checking Reliability Scorecard",
        "",
        "Benchmarked on the frozen 150-claim FEVER sample and 100-pair VitaminC sample.",
        "",
        "| Model | FEVER Acc | SUPP Acc | REF Acc | NEI Rate | 5/5 Consistency | Distractor Invariance | Post-Distractor Acc | VitaminC Sensitivity |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        lines.append(
            f"| **{r['model']}** | "
            f"{format_pct(r['fever_accuracy'])} | "
            f"{format_pct(r['fever_supported_acc'])} | "
            f"{format_pct(r['fever_refuted_acc'])} | "
            f"{format_pct(r['fever_nei_rate'])} | "
            f"{format_pct(r['consistency_5_of_5'])} | "
            f"{format_pct(r['distractor_invariance'])} | "
            f"{format_pct(r['distractor_accuracy'])} | "
            f"{format_pct(r['sensitivity_correct'])} |"
        )
    text = "\n".join(lines)
    print(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cross-model comparison scorecard.")
    parser.add_argument("--save-markdown", type=Path, help="Path to save markdown summary table")
    args = parser.parse_args()

    rows = generate_scorecard_data()
    md = print_markdown_scorecard(rows)

    if args.save_markdown:
        args.save_markdown.write_text(md + "\n", encoding="utf-8")
        print(f"\nSaved scorecard to: {args.save_markdown}")


if __name__ == "__main__":
    main()
