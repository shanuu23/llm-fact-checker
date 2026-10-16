"""Local evaluator UI. Model calls and worker launches require explicit button clicks."""

from __future__ import annotations

from html import escape
from io import BytesIO
import os
from zipfile import ZIP_DEFLATED, ZipFile

import streamlit as st

from app import jobs
from app.charts import show_comparison_charts, show_validity_warnings
from app.comparison import robustness_rows, score_rows
from app.evaluator import (
    FEVER_FULL,
    FEVER_SAMPLE,
    PAIRS_SAMPLE,
    TESTS,
    VITAMINC_TEST,
    LocalOllamaClient,
    calibrate,
    estimated_seconds,
    make_plan,
    protocol_fingerprint,
    stages,
)


def local_enabled():
    return os.environ.get("MODELLENS_LOCAL") == "1"


@st.cache_data(max_entries=3, show_spinner=False)
def cached_plan(mode, percentage, tests, source_stamps, protocol):
    return make_plan(mode, percentage, tests)


def duration(seconds):
    if seconds is None:
        return "Not yet available"
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    if seconds < 3600:
        return f"{seconds / 60:.1f} minutes"
    return f"{seconds / 3600:.1f} hours"


def run_label(run, include_model=True):
    config = run["config"]
    size = config["mode"] if config["mode"] == "Quick demo" else f"Custom {config['percentage']}%"
    prefix = f"{run['model']} | " if include_model else ""
    return (
        f"{prefix}{size} | {run['created'][:19]} UTC | {run['status']['state']} | {run['id'][:8]}"
    )


def metric_rows(kind, metrics):
    if kind == "robustness":
        return [
            (f"{variant}: {label}", value, direction)
            for variant in ("reorder", "distractor")
            for label, value, direction in robustness_rows(metrics, variant)
        ]
    return score_rows(kind, metrics)


def results_download(path):
    """Build the export only when requested, never on progress refreshes."""
    output = BytesIO()
    names = {"metrics.json", "manifest.json", "plan.json", "status.json"}
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for file in sorted(path.iterdir()):
            if (
                file.is_file()
                and not file.is_symlink()
                and (
                    file.name in names
                    or (file.suffix == ".jsonl" and ".partial." not in file.name)
                )
            ):
                archive.write(file, arcname=file.name)
    return output.getvalue()


def run_header(run):
    left, right = st.columns([4, 1])
    left.markdown(f"### {run['model']}")
    right.badge(
        run["status"]["state"].capitalize(),
        color="green" if run["status"]["state"] == "complete" else "blue",
    )
    config = run["config"]
    size = "Quick demo" if config["mode"] == "Quick demo" else f"Custom {config['percentage']}%"
    st.caption(f"{size} · {config['expected_calls']:,} model calls")


def new_evaluation():
    st.session_state.pop("live_current_run", None)
    st.session_state.pop("live_pending_run", None)
    st.session_state.pop("live_current_observed_status", None)
    st.session_state["live_tab"] = "Evaluation"
    st.session_state["live_reset_view"] = True


def result_stat(label, value, detail):
    score = "N/A" if value is None else f"{value:.1%}"
    st.html(
        '<div class="result-stat">'
        f'<div class="result-stat-label">{escape(label)}</div>'
        f'<div class="result-stat-value">{score}</div>'
        f'<div class="result-stat-detail">{escape(detail)}</div>'
        "</div>"
    )


def display_results(run, key_prefix="live", history=False):
    run_header(run)
    metrics = jobs.read_json(run["path"] / "metrics.json")
    selected = [(kind, title) for kind, title in TESTS.items() if kind in metrics]
    for offset in range(0, len(selected), 2):
        for column, (kind, title) in zip(st.columns(2), selected[offset : offset + 2]):
            with column, st.container(border=True, key=f"{key_prefix}_result_card_{kind}"):
                st.markdown(f"#### {title}")
                block = metrics[kind]
                if kind == "robustness":
                    for variant_column, (variant, label) in zip(
                        st.columns(2),
                        (("reorder", "Reordered evidence"), ("distractor", "Added distractor")),
                    ):
                        with variant_column:
                            value = block[variant]["invariance_rate"]
                            result_stat(
                                label,
                                value,
                                f"{block[variant]['invariant_count']:,} of {block[variant]['complete_pair_count']:,} unchanged",
                            )
                    st.caption("Verdict unchanged; incomplete cases excluded.")
                else:
                    label, value, _ = metric_rows(kind, block)[0]
                    if kind == "baseline":
                        detail = f"{block['number_correct']:,} of {block['total_cases']:,} claims correct"
                    elif kind == "consistency":
                        detail = f"{block['exact_consistency_count']:,} of {block['total_cases']:,} claims stable across {block['run_count']} runs"
                    else:
                        detail = f"{block['correct_sensitivity_count']:,} of {block['total_pairs']:,} pairs both correct"
                    result_stat(label, value, detail)
                validity = block.get("technical_validity", {})
                counts = list(validity.values()) if kind == "robustness" else [validity or block]
                failures = sum(
                    v.get("inference_failures", v.get("inference_failure_count", 0))
                    for v in counts
                )
                invalid = sum(
                    v.get("invalid_formats", v.get("invalid_format_count", 0)) for v in counts
                )
                if failures or invalid:
                    st.warning(f"{failures:,} inference failures · {invalid:,} invalid responses")
                incomplete = block.get("incomplete_case_count", block.get("incomplete_pairs", 0))
                if kind == "robustness":
                    incomplete = sum(
                        block[v]["eligible_pair_count"] - block[v]["complete_pair_count"]
                        for v in ("reorder", "distractor")
                    )
                if incomplete:
                    st.warning(f"{incomplete:,} incomplete cases/pairs")
    st.caption(
        ("Quick demo only. Not a model ranking. " if run["config"]["mode"] == "Quick demo" else "")
        + "Stability does not imply correctness."
    )
    new, compare = st.columns(2)
    new.button(
        "New evaluation",
        key=f"{key_prefix}_new",
        on_click=new_evaluation,
        disabled=jobs.busy(),
        type="primary",
        width="stretch",
    )
    if compare.button("Compare local runs", key=f"{key_prefix}_open_compare", width="stretch"):
        st.session_state["show_comparison"] = True
        st.session_state["show_evaluator"] = False
        st.session_state["comparison_source"] = "Local evaluator runs"
        st.rerun()
    if history:
        st.download_button(
            "Export results",
            data=lambda: results_download(run["path"]),
            file_name=f"modellens-{run['id'][:8]}-results.zip",
            mime="application/zip",
            key=f"{key_prefix}_download",
            on_click="ignore",
        )


@st.fragment(run_every=2)
def run_monitor(history=True):
    if st.session_state.pop("live_reset_view", False):
        st.rerun()
    runs = jobs.list_runs()
    if not runs:
        if history:
            st.info("No saved runs yet. Start a run in the Evaluation tab.")
        return
    by_id = {r["id"]: r for r in runs}
    prefix = "live" if history else "live_current"
    if history:
        models = list(dict.fromkeys(r["model"] for r in runs))
        if st.session_state.get("live_history_model") not in models:
            st.session_state["live_history_model"] = models[0]
        model = st.selectbox("Model history", models, key="live_history_model")
        options = [r["id"] for r in runs if r["model"] == model]
        if st.session_state.get("live_history") not in options:
            st.session_state["live_history"] = options[0]
        selected = st.selectbox(
            "Saved run (newest first)",
            options,
            format_func=lambda key: run_label(by_id[key], include_model=False),
            key="live_history",
        )
    else:
        selected = next(
            (r["id"] for r in runs if r["status"]["state"] in {"running", "starting"}),
            st.session_state.get("live_current_run"),
        )
        if selected not in by_id:
            return
        st.session_state["live_current_run"] = selected
    run = by_id[selected]
    status = run["status"]
    snapshot = (selected, status["state"], jobs.busy())
    previous = st.session_state.get(f"{prefix}_observed_status")
    st.session_state[f"{prefix}_observed_status"] = snapshot
    if previous and previous != snapshot:
        # Completion can precede lock release. Refresh again when controls become usable.
        st.rerun()
    if status["state"] == "complete":
        display_results(run, prefix, history=history)
        return
    run_header(run)
    total = status.get("total", run["config"]["expected_calls"])
    count = status.get("completed", 0)
    st.progress(
        min(count / total, 1) if total else 0, text=f"{count:,}/{total:,} recorded responses"
    )
    if status.get("stage"):
        st.caption(
            f"Pass: {status['stage']} | Active time this session: {duration(status.get('active_seconds'))}"
        )
    if status["state"] in {"running", "starting"}:
        st.caption("Approximate remaining time: " + duration(status.get("remaining_seconds")))
        if st.button("Pause after current response", key=f"{prefix}_pause"):
            jobs.request_pause(selected)
            st.info(
                "Pause requested. Waiting for the current response or request timeout; progress will be saved."
            )
    else:
        if status.get("error"):
            st.warning(status["error"])
        if st.button("Resume selected run", key=f"{prefix}_resume", disabled=jobs.busy()):
            try:
                jobs.start_run(resume_id=selected)
                st.session_state["live_current_run"] = selected
                st.rerun()
            except (OSError, ValueError, RuntimeError) as error:
                st.error(str(error))
        st.button(
            "New evaluation", key=f"{prefix}_new", on_click=new_evaluation, disabled=jobs.busy()
        )
    st.caption("Progress is saved on this computer. Pause waits for the current response.")


def show_local_comparison():
    st.caption(
        "Results from your own evaluations, compared using matching cases and settings. "
        "These runs are kept separate from saved full experiments."
    )
    if not local_enabled():
        st.info("Open the local launcher to compare your own saved runs.")
        return
    runs = [r for r in jobs.list_runs() if r["status"]["state"] == "complete"]
    if len(runs) < 2:
        st.info("Complete at least two matching local runs to compare them.")
        return
    by_id = {r["id"]: r for r in runs}
    models = list(dict.fromkeys(r["model"] for r in runs))
    left_col, middle, right_col = st.columns([4, 1.5, 4], vertical_alignment="bottom")
    with left_col:
        left_model = st.selectbox("Model 1", models, key="local_model_1")
    with right_col:
        right_model = st.selectbox(
            "Model 2", models, index=min(1, len(models) - 1), key="local_model_2"
        )
    left_runs = [r for r in runs if r["model"] == left_model]
    right_runs = [r for r in runs if r["model"] == right_model]
    model_pair = (left_model, right_model)
    if st.session_state.get("local_selected_models") != model_pair:
        # Prefer the newest compatible pair, not necessarily each model's newest run.
        default_pair = next(
            (
                (a, b)
                for a in left_runs
                for b in right_runs
                if a["id"] != b["id"] and not jobs.compatibility(a, b)
            ),
            (left_runs[0], right_runs[0]),
        )
        st.session_state["local_run_1"], st.session_state["local_run_2"] = (
            r["id"] for r in default_pair
        )
        st.session_state["local_selected_models"] = model_pair
    with left_col:
        left = st.selectbox(
            "Saved run 1",
            [r["id"] for r in left_runs],
            format_func=lambda k: run_label(by_id[k], include_model=False),
            key="local_run_1",
        )
    if st.session_state.get("local_selected_left") != left:
        matching = [
            r for r in right_runs if r["id"] != left and not jobs.compatibility(by_id[left], r)
        ]
        if matching and st.session_state.get("local_run_2") not in [r["id"] for r in matching]:
            st.session_state["local_run_2"] = matching[0]["id"]
        st.session_state["local_selected_left"] = left
    with right_col:
        right = st.selectbox(
            "Saved run 2",
            [r["id"] for r in right_runs],
            format_func=lambda k: run_label(by_id[k], include_model=False),
            key="local_run_2",
        )
    with middle:
        compare = st.button("Compare", key="local_compare", width="stretch")
    st.caption(
        "One entry per model; each evaluation is kept as a separate saved run. You can also compare two runs of the same model."
    )
    if compare:
        st.session_state["local_comparison_ids"] = (left, right)
    selected = st.session_state.get("local_comparison_ids")
    if selected != (left, right) or any(key not in by_id for key in selected):
        return
    left, right = [by_id[key] for key in selected]
    issue = (
        "Choose two different runs."
        if left["id"] == right["id"]
        else jobs.compatibility(left, right)
    )
    if issue:
        st.warning(issue)
        return
    st.text("Run 1: " + run_label(left))
    st.text("Run 2: " + run_label(right))
    st.caption(
        f"Matching workload: {left['config']['workload_id'][:16]} | "
        f"{left['config']['expected_calls']:,} requests each. Small demos are illustrative, not rankings."
    )
    lm, rm = [jobs.read_json(r["path"] / "metrics.json") for r in (left, right)]
    for tab, kind in zip(st.tabs(list(TESTS.values())), TESTS):
        with tab:
            if kind not in lm:
                st.info("Not evaluated.")
                continue
            names = [left["model"], right["model"]]
            if names[0] == names[1]:
                names = [f"{r['model']} · {r['id'][:8]}" for r in (left, right)]
            show_comparison_charts(
                kind,
                [lm[kind], rm[kind]],
                names,
                demo=left["config"]["mode"] == "Quick demo",
            )
            show_validity_warnings(kind, [lm[kind], rm[kind]], names)


def show_evaluator():
    st.divider()
    st.markdown("## Try the evaluator")
    if not local_enabled():
        st.info(
            "Local evaluation is disabled in hosted/read-only mode. Use Launch ModelLens.command "
            "on your Mac, or run `make app` from the project folder. No ngrok account is needed."
        )
        return
    st.caption("Evaluate an installed model using reproducible benchmarks, on this computer.")
    st.session_state.pop("live_reset_view", None)
    for key, value in st.session_state.get("live_setup_choices", {}).items():
        if key not in st.session_state:
            st.session_state[key] = value
    pending = st.session_state.pop("live_pending_run", None)
    if pending:
        st.session_state["live_current_run"] = pending
        run = next((r for r in jobs.list_runs() if r["id"] == pending), None)
        if run:
            st.session_state["live_history_model"] = run["model"]
            st.session_state["live_history"] = pending
    runs = jobs.list_runs()
    current = next(
        (r for r in runs if r["status"]["state"] in {"running", "starting"}),
        next((r for r in runs if r["id"] == st.session_state.get("live_current_run")), None),
    )
    if current:
        st.session_state["live_current_run"] = current["id"]
    new, history = st.tabs(["Evaluation", "Run history"], key="live_tab", on_change="rerun")
    with new:
        if not current:
            show_run_controls()
        run_monitor(history=False)
    with history:
        run_monitor()


def evaluator_help():
    st.markdown("#### Evaluation protocol")
    st.write(
        "Quick demo uses fixed cases. Custom percentages use the full eligible test populations, "
        "not pilot samples. Cases have SUPPORTS/REFUTES gold labels, not gold NEI. "
        "Selection is seeded and nested, with FEVER labels interleaved."
    )
    st.write(
        "Consistency keeps the claim and evidence fixed across five runs, using temperature 0.2 "
        "and different fixed seeds (101, 202, 303, 404 and 505). This tests verdict stability "
        "under controlled sampling variation. Evidence reordering applies only "
        "to eligible multi-sentence cases; distractors come from the selected FEVER pool. "
        "Sensitivity requires correct answers for both evidence versions of a claim. "
        "Small demos are not model rankings, and stable answers can still be wrong."
    )
    st.caption(
        "Export a completed run from Run history for its exact settings, selected cases, "
        "recorded responses and metrics."
    )
    st.markdown("#### Running and troubleshooting")
    st.write(
        "Keep Ollama running and your computer awake. Models run locally at "
        "127.0.0.1:11434; ModelLens does not upload or download models."
    )
    st.write(
        "The worker survives a page refresh. A shutdown requires Resume from Run history; "
        "sleep pauses computation. Pause waits for the current response or timeout. "
        "Failed calls in unfinished passes are retried on resume."
    )
    st.write(
        "Cold loading can exceed the 120-second request timeout. If it does, warm the model "
        "in Ollama first and start a new quick demo. Avoid other model jobs during evaluation."
    )
    st.caption(
        "Runtime estimates extrapolate observed sample times, not a confidence interval or guarantee. "
        "Heating and cold loading affect speed. Estimates expire after one hour; refresh models after changes."
    )


def show_run_controls():
    layout = st.container(key="evaluation_layout")
    setup, summary = layout.columns([1.8, 1], gap="medium", border=True)
    with setup, st.container(key="evaluation_setup"):
        st.markdown("### Evaluation setup")
        plan, model = evaluation_setup()
    with summary, st.container(key="evaluation_summary"):
        st.markdown("### Run summary")
        if plan is None:
            st.caption("Connect a model and choose tests to prepare your workload.")
            st.button(
                "Run evaluation", type="primary", key="live_run", disabled=True, width="stretch"
            )
            return
        run_summary(plan, model)
    if plan["config"]["mode"] == "Quick demo":
        st.info("Quick demo checks the workflow. It is not a model ranking.")
    else:
        st.caption("Compare only runs with matching cases, tests and settings.")


def evaluation_setup():
    if st.button(
        "Refresh models" if st.session_state.get("live_models") else "Connect to Ollama",
        key="live_connect",
        disabled=jobs.busy(),
    ):
        st.session_state.pop("live_models", None)
        st.session_state.pop("live_timing", None)
        try:
            st.session_state["live_models"] = LocalOllamaClient().list_models()
            if not st.session_state["live_models"]:
                st.info(
                    "No local text models found. Install a local model in Ollama, then refresh. ModelLens does not download models."
                )
        except (OSError, ValueError, RuntimeError) as error:
            st.error(str(error))
    models = st.session_state.get("live_models")
    if not models:
        st.caption("Start Ollama, then connect to see your installed models.")
        return None, None
    if st.session_state.get("live_model") not in models:
        st.session_state["live_model"] = models[0]
    model = st.selectbox("Installed model", models, key="live_model")
    st.caption("Ollama connected · local models only")
    st.markdown("**Evaluation tests**")
    descriptions = {
        "baseline": "Match expected labels",
        "consistency": "Repeat across five runs",
        "robustness": "Reorder and distract",
        "sensitivity": "Respond to changed evidence",
    }
    tests = []
    columns = st.columns(2)
    for index, (kind, title) in enumerate(TESTS.items()):
        with columns[index % 2]:
            if st.checkbox(title, value=True, key=f"live_test_{kind}"):
                tests.append(kind)
            st.caption(descriptions[kind])
    mode = st.radio("Dataset size", ["Quick demo", "Custom"], horizontal=True, key="live_mode")
    percentage = (
        st.slider("Evaluation dataset size (%)", 1, 100, 1, key="live_percentage")
        if mode == "Custom"
        else 1
    )
    st.session_state["live_setup_choices"] = {
        key: st.session_state[key]
        for key in (
            "live_model",
            "live_mode",
            "live_percentage",
            *(f"live_test_{kind}" for kind in TESTS),
        )
        if key in st.session_state
    }
    if not tests:
        st.info("Choose at least one evaluation test.")
        return None, model
    try:
        paths = (FEVER_SAMPLE, FEVER_FULL, PAIRS_SAMPLE, VITAMINC_TEST)
        stamps = tuple(
            (str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths if p.exists()
        )
        with st.spinner("Preparing a reproducible workload (no model calls)…"):
            plan = cached_plan(mode, percentage, tuple(tests), stamps, protocol_fingerprint())
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Could not prepare the benchmark: {error}")
        return None, model
    data = plan["inputs"]
    st.caption(f"{len(data['fever']):,} FEVER claims and {len(data['pairs']):,} VitaminC pairs")
    if "robustness" in tests:
        st.caption(f"Reordering applies to {len(data['reorder']):,} eligible claims.")
    return plan, model


def run_summary(plan, model):
    data, config = plan["inputs"], plan["config"]
    for label, value in (
        ("FEVER claims", len(data["fever"])),
        ("VitaminC pairs", len(data["pairs"])),
        ("Model calls", config["expected_calls"]),
    ):
        label_col, value_col = st.columns([2, 1])
        label_col.write(label)
        value_col.write(f"**{value:,}**")
    st.divider()
    calibration_calls = sum(
        min(3, len(cases))
        for name, cases, _, _, _ in stages(plan)
        if not name.startswith("consistency_") or name == "consistency_1"
    )
    estimate_slot = st.empty()
    if st.button("Estimate runtime", key="live_measure", disabled=jobs.busy(), width="stretch"):
        try:
            with jobs.exclusive(), st.spinner("Measuring local speed…"):
                st.session_state["live_timing"] = calibrate(LocalOllamaClient(), model, plan)
        except (OSError, ValueError, RuntimeError) as error:
            st.error(str(error))
    timing = st.session_state.get("live_timing")
    estimate = estimated_seconds(plan, timing) if timing and timing["model"] == model else None
    estimate_slot.markdown(
        "**Estimated duration**  \n"
        + (
            f"{duration(estimate[0])} – {duration(estimate[1])}"
            if estimate
            else "Not measured for this workload"
        )
    )
    st.caption(f"Optional · up to {calibration_calls} extra model calls, not scored.")
    confirm = True
    if config["expected_calls"] > 1000:
        confirm = st.checkbox(
            "I understand this workload may take hours and will save results locally."
        )
    if jobs.busy():
        st.info(
            "A local job is running. See progress below; new runs are disabled until it finishes or pauses."
        )
    if st.button(
        "Run evaluation",
        type="primary",
        key="live_run",
        disabled=jobs.busy() or not confirm,
        width="stretch",
    ):
        try:
            with st.spinner("Checking the model and starting the evaluation…"):
                run_id = jobs.start_run(model, plan)
            st.session_state["live_pending_run"] = run_id
            st.rerun()
        except (OSError, ValueError, RuntimeError) as error:
            st.error(str(error))
    st.caption("Results are saved automatically on this computer.")
