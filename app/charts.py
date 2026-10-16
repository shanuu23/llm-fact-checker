"""Shared comparison visuals; all values come from recorded metrics."""

import altair as alt
import pandas as pd
import streamlit as st

from app.interpretation import MEASURES, interpretation

COLORS = ["#0f766e", "#2563eb"]
STATES = [
    ("STABLE_CORRECT", "Correct every run", "#0f766e"),
    ("STABLE_WRONG", "Wrong every run", "#d9776a"),
    ("UNSTABLE", "Unstable", "#d99a20"),
    ("INCOMPLETE", "Incomplete", "#94a3b8"),
]


def show_validity_warnings(kind, metrics, names):
    """Keep failed or invalid requests visible without a raw metrics panel."""
    for name, block in zip(names, metrics, strict=True):
        validity = block.get("technical_validity", {})
        counts = list(validity.values()) if kind == "robustness" else [validity or block]
        failures = sum(
            v.get("inference_failures", v.get("inference_failure_count", 0)) for v in counts
        )
        invalid = sum(v.get("invalid_formats", v.get("invalid_format_count", 0)) for v in counts)
        if failures or invalid:
            st.warning(f"{name}: {failures:,} inference failures · {invalid:,} invalid responses")


def percent(value):
    return "N/A" if value is None else f"{value:.1%}"


def paired_chart(labels, values, names):
    rows = [
        {
            "Measure": label,
            "Model": name,
            "Value": value,
            "Position": 0 if value is None else value,
            "Label": percent(value),
        }
        for name, scores in zip(names, values, strict=True)
        for label, value in zip(labels, scores, strict=True)
    ]
    single = len(labels) == 1
    base = alt.Chart(pd.DataFrame(rows)).encode(
        y=alt.Y(
            "Model:N" if single else "Measure:N",
            sort=names if single else labels,
            title=None,
            axis=alt.Axis(labelLimit=230, labelPadding=10),
            scale=alt.Scale(paddingInner=0.35, paddingOuter=0.2),
        ),
        tooltip=["Model:N", "Measure:N", alt.Tooltip("Label:N", title="Score")],
    )
    if not single:
        base = base.encode(
            yOffset=alt.YOffset(
                "Model:N", sort=names, scale=alt.Scale(paddingInner=0.25, paddingOuter=0.1)
            )
        )
    bars = base.mark_bar(size=26, cornerRadiusEnd=3).encode(
        x=alt.X(
            "Value:Q",
            scale=alt.Scale(domain=[0, 1]),
            title=None,
            axis=alt.Axis(format=".0%", values=[0, 0.25, 0.5, 0.75, 1]),
        ),
        color=alt.Color(
            "Model:N",
            scale=alt.Scale(domain=names, range=COLORS),
            legend=None if single else alt.Legend(title=None, orient="top", offset=12),
        ),
    )
    # Keep 100% labels inside the plot; missing values have text but no bar.
    low = (
        base.transform_filter("datum.Position < 0.9")
        .mark_text(align="left", dx=5, color="#172b3a")
        .encode(
            x=alt.X(
                "Position:Q",
                scale=alt.Scale(domain=[0, 1]),
                title=None,
                axis=alt.Axis(format=".0%", values=[0, 0.25, 0.5, 0.75, 1]),
            ),
            text="Label:N",
        )
    )
    high = (
        base.transform_filter("datum.Position >= 0.9")
        .mark_text(align="right", dx=-5, color="white")
        .encode(
            x=alt.X(
                "Position:Q",
                scale=alt.Scale(domain=[0, 1]),
                title=None,
                axis=alt.Axis(format=".0%", values=[0, 0.25, 0.5, 0.75, 1]),
            ),
            text="Label:N",
        )
    )
    return (
        (bars + low + high)
        .properties(
            height=140 if single else len(labels) * 112,
            autosize=alt.AutoSizeParams(type="fit-x", contains="padding"),
        )
        .configure_view(stroke=None)
        .configure_axis(
            labelFontSize=15, labelColor="#334155", gridColor="#e2e8f0", domain=False, tickSize=0
        )
        .configure_legend(labelFontSize=15, labelColor="#334155")
        .configure_text(fontSize=15)
    )


def consistency_chart(metrics, names):
    rows = []
    for name, metric in zip(names, metrics, strict=True):
        start = 0
        for key, label, _ in STATES:
            value = metric["stability_classification"][key]["percentage"]
            rows.append(
                {
                    "Model": name,
                    "Outcome": label,
                    "Start": start,
                    "End": start + value,
                    "Middle": start + value / 2,
                    "Value": value,
                    "Label": percent(value),
                }
            )
            start += value
    base = alt.Chart(pd.DataFrame(rows)).encode(
        y=alt.Y(
            "Model:N",
            sort=names,
            title=None,
            axis=alt.Axis(labelLimit=230, labelPadding=10),
            scale=alt.Scale(paddingInner=0.4, paddingOuter=0.25),
        ),
        tooltip=["Model:N", "Outcome:N", alt.Tooltip("Value:Q", format=".1%", title="Share")],
    )
    bars = base.mark_bar(size=42).encode(
        x=alt.X(
            "Start:Q",
            scale=alt.Scale(domain=[0, 1]),
            title=None,
            axis=alt.Axis(format=".0%", values=[0, 0.25, 0.5, 0.75, 1]),
        ),
        x2="End:Q",
        color=alt.Color(
            "Outcome:N",
            scale=alt.Scale(domain=[s[1] for s in STATES], range=[s[2] for s in STATES]),
            legend=alt.Legend(title=None, orient="bottom", columns=2),
        ),
    )
    labels = (
        base.transform_filter("datum.Value >= 0.08")
        .mark_text(color="white", fontWeight="bold")
        .encode(
            x=alt.X(
                "Middle:Q",
                scale=alt.Scale(domain=[0, 1]),
                title=None,
                axis=alt.Axis(format=".0%", values=[0, 0.25, 0.5, 0.75, 1]),
            ),
            text="Label:N",
        )
    )
    return (
        (bars + labels)
        .properties(height=190, autosize=alt.AutoSizeParams(type="fit-x", contains="padding"))
        .configure_view(stroke=None)
        .configure_axis(
            labelFontSize=15, labelColor="#334155", gridColor="#e2e8f0", domain=False, tickSize=0
        )
        .configure_legend(labelFontSize=15, labelColor="#334155")
        .configure_text(fontSize=15)
    )


def show_centered_chart(chart, key):
    # A fixed-width container shrinks to its parent on smaller screens.
    with st.container(horizontal_alignment="center"), st.container(width=900):
        st.altair_chart(chart, width="stretch", key=key)


def show_comparison_charts(kind, metrics, names, *, demo=False):
    st.markdown("#### What this chart measures")
    st.write(MEASURES[kind])
    left = metrics[0]
    if kind != "robustness":
        field, title = {
            "baseline": ("overall_accuracy", "Overall accuracy"),
            "consistency": ("exact_consistency_rate", "Same answer in all five runs"),
            "sensitivity": ("correct_sensitivity_rate", "Both evidence versions correct"),
        }[kind]
        st.caption(title)
        for column, name, metric in zip(st.columns(2), names, metrics, strict=True):
            with column:
                st.markdown(f"**{name} · {percent(metric[field])}**")
    if kind == "baseline":
        st.caption(f"FEVER · {left['total_cases']:,} claims per model · higher accuracy is better")
        values = [
            [
                m["overall_accuracy"],
                m["class_accuracy"]["SUPPORTED"],
                m["class_accuracy"]["REFUTED"],
            ]
            for m in metrics
        ]
        show_centered_chart(
            paired_chart(["Overall", "Supports", "Refutes"], values, names),
            key="comparison_baseline_chart",
        )
    elif kind == "consistency":
        st.caption(f"FEVER · {left['total_cases']:,} claims · {left['run_count']} runs per claim")
        show_centered_chart(consistency_chart(metrics, names), key="comparison_consistency_chart")
        st.caption(
            "Repeated answers can still be wrong. All claims are included; incomplete means at least one invalid or missing answer."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Model": name,
                        **{
                            label: m["stability_classification"][key]["count"]
                            for key, label, _ in STATES
                        },
                    }
                    for name, m in zip(names, metrics, strict=True)
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    elif kind == "robustness":
        for column, (variant, title) in zip(
            st.columns(2),
            [("reorder", "Reordered evidence"), ("distractor", "Added distractor")],
            strict=True,
        ):
            with column:
                st.markdown(f"#### {title}")
                blocks = [m[variant] for m in metrics]
                st.caption(f"{blocks[0]['eligible_pair_count']:,} eligible claims per model")
                st.altair_chart(
                    paired_chart(
                        ["Original", "Changed"],
                        [[b["control_accuracy"], b["perturbed_accuracy"]] for b in blocks],
                        names,
                    ),
                    width="stretch",
                    key=f"comparison_{variant}_chart",
                )
                for name, b in zip(names, blocks, strict=True):
                    st.caption(
                        f"{name}: verdict unchanged {percent(b['invariance_rate'])} · {b['complete_pair_count']:,} complete pairs"
                    )
        st.caption(
            "Original and changed accuracy use the same eligible subset. An unchanged verdict may still be wrong."
        )
    else:
        st.caption(f"VitaminC · {left['total_pairs']:,} pairs per model")
        show_centered_chart(
            paired_chart(
                ["Both versions correct"],
                [[m["correct_sensitivity_rate"]] for m in metrics],
                names,
            ),
            key="comparison_sensitivity_chart",
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Model": name,
                        "Supporting accuracy": percent(
                            m["individual_accuracy"]["support_accuracy"]
                        ),
                        "Refuting accuracy": percent(m["individual_accuracy"]["refute_accuracy"]),
                        "Incomplete pairs": m["incomplete_pairs"],
                    }
                    for name, m in zip(names, metrics, strict=True)
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Both answers must be correct. A verdict change alone is not success. N/A means no eligible observations, not zero."
        )
    st.markdown("#### Results interpretation")
    st.write(interpretation(kind, metrics, names, demo=demo))
