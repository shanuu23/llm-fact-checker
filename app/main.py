"""Run from the repository root: uv run --group app streamlit run app/main.py."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from app.comparison import (
    EXPERIMENTS,
    comparison_issues,
    load_models,
    result_examples,
    robustness_rows,
    score_rows,
)

COLORS = ["#0f766e", "#c45b2b"]
DESCRIPTIONS = {
    "baseline": "Does the answer match the dataset's expected label?",
    "consistency": "Does the model give the same answer across five runs?",
    "robustness": "What happens when we reorder evidence or add a distractor?",
    "sensitivity": "Does the answer change correctly when the evidence meaning changes?",
}


def show_scores(left_rows, right_rows, names):
    frame = pd.DataFrame(
        {
            "Measure": [row[0] for row in left_rows],
            names[0]: [row[1] * 100 for row in left_rows],
            names[1]: [row[1] * 100 for row in right_rows],
        }
    )
    st.bar_chart(
        frame,
        x="Measure",
        y=names,
        color=COLORS,
        horizontal=True,
        stack=False,
        x_label="Percent (%)",
        height=max(210, len(frame) * 80),
    )
    details = frame.copy()
    details["Model 2 − Model 1 (pp)"] = details[names[1]] - details[names[0]]
    details["Interpretation"] = [
        {
            "higher": "Higher is better",
            "lower": "Lower is better",
            "context": "Read with accuracy",
        }[row[2]]
        for row in left_rows
    ]
    st.dataframe(
        details,
        hide_index=True,
        width="stretch",
        column_config={
            names[0]: st.column_config.NumberColumn(format="%.2f%%"),
            names[1]: st.column_config.NumberColumn(format="%.2f%%"),
            "Model 2 − Model 1 (pp)": st.column_config.NumberColumn(format="%+.2f"),
        },
    )


def show_quality(kind, metrics, names):
    rows = []
    for name, m in zip(names, metrics):
        if kind == "baseline":
            requests, failures, invalid = (
                m["total_cases"],
                m["inference_failure_count"],
                m["invalid_format_count"],
            )
        elif kind == "consistency":
            requests, failures, invalid = (
                m["expected_inference_attempts"],
                m["inference_failure_count"],
                m["invalid_format_count"],
            )
        elif kind == "robustness":
            blocks = list(m["technical_validity"].values())
            requests = sum(item["requests"] for item in blocks)
            failures = sum(item["inference_failures"] for item in blocks)
            invalid = sum(item["invalid_formats"] for item in blocks)
        else:
            validity = m["technical_validity"]
            requests, failures, invalid = (
                validity["expected_requests"],
                validity["inference_failures"],
                validity["invalid_formats"],
            )
        rows.append(
            {
                "Model": name,
                "Expected requests": requests,
                "Failed calls": failures,
                "Invalid answers": invalid,
            }
        )
    with st.expander("Run quality"):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            "Invalid answers and failed calls remain part of the saved evaluation. The app does not change the scoring rules."
        )


def show_examples(runs, kind, names):
    with st.expander("Explore saved examples"):
        st.caption(
            "First 10 recorded errors per model; these are not a representative accuracy sample."
            if kind == "baseline"
            else "First 10 saved cases per model, in file order. Full scores above use all eligible cases."
        )
        for name, run in zip(names, runs):
            st.markdown(f"**{name}**")
            examples = result_examples(run)
            if examples:
                st.dataframe(pd.DataFrame(examples), hide_index=True, width="stretch")
            else:
                st.info("No examples in the saved table.")


def show_experiment(kind, runs, names):
    st.markdown(DESCRIPTIONS[kind])
    if any(run is None for run in runs):
        st.info("This experiment is not available for both selected models yet.")
        return
    try:
        issues = comparison_issues(*runs)
        if issues:
            st.warning("These runs cannot be compared directly: " + "; ".join(issues) + ".")
            return
        metrics = [run.metrics() for run in runs]
        left, right = metrics
        if kind == "baseline":
            st.caption(
                f"FEVER · {left['total_cases']:,} gold-evidence claims per model · higher accuracy is better"
            )
            show_scores(score_rows(kind, left), score_rows(kind, right), names)
        elif kind == "consistency":
            st.caption(
                f"FEVER · {left['total_cases']:,} claims · {left['run_count']} runs per claim"
            )
            st.info(
                "A model can repeat the same wrong answer. Read agreement alongside consistently correct and consistently wrong."
            )
            show_scores(score_rows(kind, left), score_rows(kind, right), names)
            st.caption(
                "Percentages use all claims. Incomplete means at least one run lacked a valid label."
            )
        elif kind == "robustness":
            for variant, title in (
                ("reorder", "Reordered evidence"),
                ("distractor", "Added distractor"),
            ):
                st.markdown(f"#### {title}")
                blocks = [m[variant] for m in metrics]
                st.caption(
                    f"{blocks[0]['eligible_pair_count']:,} eligible claims per model. Original and changed accuracy use this same subset."
                )
                show_scores(robustness_rows(left, variant), robustness_rows(right, variant), names)
                for name, block in zip(names, blocks):
                    st.caption(
                        f"{name}: accuracy change {block['accuracy_delta'] * 100:+.2f} pp; unchanged verdicts measured on {block['complete_pair_count']:,} complete pairs."
                    )
            st.info(
                "An unchanged verdict may still be wrong. Use the original and changed accuracies to judge the effect."
            )
        else:
            st.caption(
                f"VitaminC · {left['total_pairs']:,} SUPPORTS–REFUTES pairs per model · two evidence versions for each claim"
            )
            show_scores(score_rows(kind, left), score_rows(kind, right), names)
            st.info(
                "Correct sensitivity requires both answers to be right. A verdict change by itself can be incorrect."
            )
            for name, m in zip(names, metrics):
                st.caption(
                    f"{name}: {m['correct_sensitivity_count']:,}/{m['total_pairs']:,} pairs correct on both sides; {m['incomplete_pairs']:,} incomplete. Any-change rate uses {m['complete_pairs']:,} complete pairs."
                )
        show_quality(kind, metrics, names)
        with st.expander("Input and model details"):
            for name, run in zip(names, runs):
                st.markdown(f"**{name}**")
                st.json(run.provenance(), expanded=False)
        show_examples(runs, kind, names)
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.error(f"Could not load this experiment's saved results: {error}")


def main():
    st.set_page_config(page_title="LLM Evaluator", page_icon="◈", layout="wide")
    st.html(f"<style>{Path(__file__).with_name('style.css').read_text(encoding='utf-8')}</style>")
    st.html('<div class="eyebrow">EVIDENCE · RELIABILITY · EVALUATION</div>')
    st.title("LLM Evaluator")
    st.markdown("### Beyond a single accuracy score.")
    st.write(
        "Explore how language models respond to evidence: when they get it right, whether they repeat their answers, and how those answers change."
    )

    st.markdown("## Project overview")
    st.write(
        "We evaluate pretrained language models with FEVER and VitaminC benchmark data. Four tests reveal different aspects of their behaviour."
    )
    with st.container(key="test_cards"):
        for column, (kind, (_, _, title)) in zip(st.columns(4, border=True), EXPERIMENTS.items()):
            with column:
                st.markdown(f"#### {title}")
                st.write(DESCRIPTIONS[kind])

    try:
        models = load_models()
        labels = {
            model: next(iter(runs.values())).config()["ollama_model"]
            for model, runs in models.items()
        }
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Saved results could not be loaded: {error}")
        return
    st.caption(
        f"{len(models)} models with full-run result files · saved experiments · no model installation needed to compare"
    )

    compare_card, demo_card = st.columns(2)
    with compare_card, st.container(border=True, key="compare_card"):
        st.markdown("### Compare models")
        st.write("Choose two models and explore their recorded results across all four tests.")
        if st.button("Open comparison", type="primary", key="open_compare"):
            st.session_state["show_comparison"] = True
    with demo_card, st.container(border=True, key="demo_card"):
        st.caption("COMING SOON")
        st.markdown("### Try the evaluator")
        st.write("Run a small evaluation with a connected model and inspect its responses.")
        st.button("Try the evaluator", disabled=True, key="try_evaluator")

    if st.session_state.get("show_comparison"):
        st.divider()
        st.markdown("## Compare models")
        st.caption("Select two different models, then compare their saved full-run results.")
        if len(models) < 2:
            st.info("Add full-run results for at least two models to enable comparison.")
        else:
            options = sorted(models, key=lambda model: labels[model])
            with st.form("model_comparison"):
                left_col, button_col, right_col = st.columns(
                    [4, 1.5, 4], vertical_alignment="bottom"
                )
                with left_col:
                    left = st.selectbox("Model 1", options, format_func=labels.get, key="model_1")
                with right_col:
                    right = st.selectbox(
                        "Model 2", options, index=1, format_func=labels.get, key="model_2"
                    )
                with button_col:
                    submitted = st.form_submit_button("Compare", type="primary", width="stretch")
            if submitted:
                st.session_state["compared_models"] = (left, right) if left != right else None
                if left == right:
                    st.warning("Choose two different models to compare.")
            selected = st.session_state.get("compared_models")
            if selected and all(model in models for model in selected):
                names = [labels[model] for model in selected]
                st.html(
                    f'<div class="comparison-heading"><span style="color:{COLORS[0]}">{escape(names[0])}</span><span class="versus">vs</span><span style="color:{COLORS[1]}">{escape(names[1])}</span></div>'
                )
                st.caption(
                    "Differences are Model 2 minus Model 1, in percentage points (pp). Each test measures a different behaviour."
                )
                tabs = st.tabs([value[2] for value in EXPERIMENTS.values()])
                for tab, kind in zip(tabs, EXPERIMENTS):
                    with tab:
                        show_experiment(
                            kind, [models[model].get(kind) for model in selected], names
                        )

    with st.expander("About the data and scoring"):
        st.write(
            "FEVER evaluates claims with gold supporting or refuting evidence. VitaminC evaluates pairs where the same claim has one supporting and one refuting evidence passage. The saved test runs do not include examples whose expected label is NOT ENOUGH INFORMATION (NEI), although models can predict it."
        )
        st.write(
            "This view uses the full saved metrics, including flagged cases. Scores show agreement with dataset labels, which can contain annotation errors. Only full-run folders are used; pilot and smoke results are excluded."
        )
        st.write(
            "Labels are scored using the recorded evaluation rules. Invalid output and failed requests are reported. Results are shown by test, with no combined model ranking."
        )
    st.caption("LLM Evaluator · FEVER & VitaminC · Group 10")


if __name__ == "__main__":
    main()
