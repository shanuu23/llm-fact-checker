"""Local ModelLens UI: launch with make app (no inference until explicitly requested)."""

from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st

from app.charts import COLORS, show_comparison_charts, show_validity_warnings
from app.comparison import (
    EXPERIMENTS,
    comparison_issues,
    load_models,
)
from app.live import evaluator_help, show_evaluator, show_local_comparison

DESCRIPTIONS = {
    "baseline": "Does the model’s answer match the expected verdict?",
    "consistency": "Does the model give the same verdict across five runs?",
    "robustness": "Does the verdict stay stable when evidence is reordered or a distractor is added?",
    "sensitivity": "Does the model correctly handle both supporting and refuting evidence for the same claim?",
}


def show_experiment(kind, runs, names):
    if any(run is None for run in runs):
        st.info("This experiment is not available for both selected models yet.")
        return
    try:
        issues = comparison_issues(*runs)
        if issues:
            st.warning("These runs cannot be compared directly: " + "; ".join(issues) + ".")
            return
        metrics = [run.metrics() for run in runs]
        show_comparison_charts(kind, metrics, names)
        show_validity_warnings(kind, metrics, names)
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.error(f"Could not load this experiment's saved results: {error}")


def main():
    st.set_page_config(page_title="ModelLens", page_icon="◈", layout="wide")
    st.html(f"<style>{Path(__file__).with_name('style.css').read_text(encoding='utf-8')}</style>")
    with st.container(key="hero"):
        st.title("ModelLens", text_alignment="center")
        st.markdown(
            "### Go beyond accuracy when comparing language models.",
            text_alignment="center",
            width="stretch",
        )
        st.markdown(
            "Evaluate how correctly and consistently models answer, and whether their answers "
            "change appropriately when the evidence changes.",
            text_alignment="center",
            width="stretch",
        )
        st.caption(
            "Built for AI engineers and researchers · By Evidence Lab",
            text_alignment="center",
        )

    st.write(
        "ModelLens helps AI engineers and researchers evaluate local language models using "
        "FEVER and VitaminC benchmark data. Explore saved comparisons or test your own model "
        "across four dimensions."
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
        models, labels = {}, {}

    compare_card, demo_card = st.columns(2)
    with compare_card, st.container(border=True, key="compare_card"):
        st.markdown("### Compare models")
        st.write("Choose two models and explore their recorded results across all four tests.")
        if st.button("Open comparison", type="primary", key="open_compare"):
            st.session_state["show_comparison"] = True
            st.session_state["show_evaluator"] = False
    with demo_card, st.container(border=True, key="demo_card"):
        st.markdown("### Try the evaluator")
        st.write("Run a small evaluation with a connected model and inspect its responses.")
        if st.button("Try the evaluator", key="try_evaluator"):
            st.session_state["show_evaluator"] = True
            st.session_state["show_comparison"] = False

    if st.session_state.get("show_evaluator"):
        show_evaluator()

    if st.session_state.get("show_comparison"):
        st.divider()
        st.markdown("## Compare models")
        st.write(
            "Compare how two models perform across correctness, consistency, robustness and "
            "evidence sensitivity. Explore recorded results without loading models or running "
            "evaluations again."
        )
        source = st.radio(
            "Comparison source",
            ["Saved full experiments", "Local evaluator runs"],
            horizontal=True,
            key="comparison_source",
        )
        if source == "Local evaluator runs":
            show_local_comparison()
        else:
            show_full_comparison(models, labels)

    with st.popover("Evaluation guide & help"):
        evaluator_help()
        st.markdown("#### Data and scoring")
        st.write(
            "FEVER and VitaminC provide evidence-based benchmark cases. The selected experiments "
            "use SUPPORTS/REFUTES gold labels, not gold NEI, although models may predict NEI. "
            "Local demo/custom runs are separate from saved full experiments."
        )
        st.write(
            "Scores include flagged cases; labels can contain annotation errors. "
            "Invalid outputs and failed calls remain visible. "
            "There is no combined model ranking or claim of universal model quality."
        )
    st.caption("ModelLens · Evidence Lab · FEVER & VitaminC")


def show_full_comparison(models, labels):
    st.caption(
        "Results from our completed benchmark evaluations. Select two different models to compare."
    )
    if len(models) < 2:
        st.info("Add full-run results for at least two models to enable comparison.")
    else:
        options = sorted(models, key=lambda model: labels[model])
        with st.form("model_comparison"):
            left_col, button_col, right_col = st.columns([4, 1.5, 4], vertical_alignment="bottom")
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
            tabs = st.tabs([value[2] for value in EXPERIMENTS.values()])
            for tab, kind in zip(tabs, EXPERIMENTS):
                with tab:
                    show_experiment(kind, [models[model].get(kind) for model in selected], names)


if __name__ == "__main__":
    main()
