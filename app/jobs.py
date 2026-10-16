"""Disk-backed local runs, using the existing inference/checkpoint and metric functions."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from app.evaluator import (
    ROOT,
    LocalOllamaClient,
    fingerprint,
    model_identity,
    protocol_fingerprint,
    stages,
)
from llm_fact_checker.common import read_jsonl, write_jsonl, write_text_atomic
from llm_fact_checker.evaluation import calculate_metrics
from llm_fact_checker.experiments.consistency import build_case_rows, calculate_consistency_metrics
from llm_fact_checker.experiments.robustness import calculate_robustness_metrics
from llm_fact_checker.experiments.sensitivity import build_pair_rows, calculate_sensitivity_metrics
from llm_fact_checker.ollama import infer_cases_resumable

STORE = ROOT / ".modellens" / "runs"


class Paused(Exception):
    pass


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


@contextmanager
def exclusive(root=STORE):
    """One evaluation or calibration per installation. Kernel releases lock after a crash."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".worker.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(
                "An evaluation or speed check is already running. Wait or pause it first."
            ) from None
        (root / ".active-run").unlink(missing_ok=True)
        yield handle


def busy(root=STORE):
    path = root / ".worker.lock"
    if not path.exists():
        return False
    with path.open() as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def check_plan(plan):
    config = plan["config"]
    if config["protocol"] != protocol_fingerprint():
        raise ValueError("Evaluation code changed. Create a new run rather than mixing protocols.")
    if config["inputs_sha256"] != fingerprint(plan["inputs"]):
        raise ValueError("Saved input snapshot changed; refusing to resume.")
    if config["workload_id"] != fingerprint(
        {k: v for k, v in config.items() if k != "workload_id"}
    ):
        raise ValueError("Saved workload configuration changed.")


def run_directory(root, run_id):
    if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
        raise ValueError("Invalid run identifier.")
    path = root / run_id
    if path.resolve().parent != root.resolve():
        raise ValueError("Run must be inside the local results folder.")
    return path


def start_run(model=None, plan=None, *, resume_id=None, root=STORE):
    if os.environ.get("MODELLENS_LOCAL") != "1":
        raise ValueError("Use the local launcher to enable evaluation. Hosted mode is read-only.")
    with exclusive(root) as lock:
        client = LocalOllamaClient()
        if resume_id:
            path = run_directory(root, resume_id)
            manifest = read_json(path / "manifest.json")
            plan = read_json(path / "plan.json")
            model = manifest["model"]
            if (path / "metrics.json").exists():
                raise ValueError(
                    "This run is already complete. Start a new run to evaluate again."
                )
        else:
            check_plan(plan)
            metadata = client.check(model)
            path = run_directory(root, uuid.uuid4().hex)
            path.mkdir()
            manifest = {
                "model": model,
                "identity": model_identity(metadata),
                "metadata": metadata,
                "created": datetime.now(UTC).isoformat(),
                "config": plan["config"],
            }
            save_json(path / "plan.json", plan)
            save_json(path / "manifest.json", manifest)
        check_plan(plan)
        if resume_id and model_identity(client.check(model)) != manifest["identity"]:
            raise ValueError("Model or Ollama version changed. Start a new run.")
        (path / "stop").unlink(missing_ok=True)
        save_json(
            path / "status.json",
            {"state": "starting", "completed": 0, "total": plan["config"]["expected_calls"]},
        )
        write_text_atomic(root / ".active-run", path.name)
        try:
            with (path / "worker.log").open("a") as log:
                # Inherit the already-held lock: no launch gap and no concurrent inference.
                subprocess.Popen(
                    [sys.executable, "-m", "app.jobs", str(path), str(lock.fileno())],
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                    pass_fds=(lock.fileno(),),
                )
        except OSError:
            save_json(path / "status.json", {"state": "error", "error": "Worker could not start."})
            raise
    return path.name


def request_pause(run_id, root=STORE):
    (run_directory(root, run_id) / "stop").touch()


def list_runs(root=STORE):
    if not root.exists():
        return []
    runs = []
    active_path = root / ".active-run"
    active = active_path.read_text() if active_path.exists() and busy(root) else None
    for path in sorted(root.iterdir(), reverse=True):
        if not path.is_dir():
            continue
        try:
            path = run_directory(root, path.name)
            manifest = read_json(path / "manifest.json")
            status = read_json(path / "status.json")
            if (path / "metrics.json").exists():
                status["state"] = "complete"
            elif status["state"] in {"starting", "running"} and active != path.name:
                status["state"] = "interrupted"
            runs.append({"id": path.name, "path": path, **manifest, "status": status})
        except (OSError, ValueError, KeyError):
            continue  # A failed creation/corrupt local run is not a comparable result.
    return sorted(runs, key=lambda r: r["created"], reverse=True)


def compatibility(left, right):
    if any(r["status"]["state"] != "complete" for r in (left, right)):
        return "Only completed runs can be compared."
    if left["config"]["workload_id"] != right["config"]["workload_id"]:
        return (
            "Different cases, tests, settings or protocol. Choose runs with the same workload ID."
        )
    versions = [run.get("identity", {}).get("version") for run in (left, right)]
    if not all(versions) or versions[0] != versions[1]:
        return "Missing or different Ollama versions. Use the same version for both model runs."
    return None


def execute(path, client=None):
    """Called only by a locked worker (or an isolated test with a fake client)."""
    plan = read_json(path / "plan.json")
    manifest = read_json(path / "manifest.json")
    check_plan(plan)
    if manifest["config"] != plan["config"]:
        raise ValueError("Run manifest does not match the saved workload.")
    if (path / "metrics.json").exists():
        return

    def stop_check():
        if (path / "stop").exists():
            raise Paused("Paused after saving the current response.")

    client = client or LocalOllamaClient(before_call=stop_check)
    if model_identity(client.check(manifest["model"])) != manifest["identity"]:
        raise ValueError("Model or Ollama version changed. Start a new run.")
    state = {
        "state": "running",
        "completed": 0,
        "total": plan["config"]["expected_calls"],
        "stage": "preparing",
        "active_seconds": 0,
        "remaining_seconds": None,
    }
    started = time.monotonic()
    new_latencies = []
    outputs = {}
    for name, cases, seed, temperature, id_field in stages(plan):
        stop_check()
        state["stage"] = name
        final = path / f"{name}.jsonl"
        checkpoint = path / f"{name}.partial.jsonl"
        expected = {c[id_field]: c for c in cases}

        def validate_rows(rows, expected=expected, id_field=id_field):
            for row in rows:
                case = expected.get(row[id_field])
                if case is None or any(
                    row[key] != case[key] for key in ("claim", "evidence", "gold_label")
                ):
                    raise ValueError("Saved predictions do not match this workload's inputs.")

        if final.exists():
            rows = list(read_jsonl(final))
            validate_rows(rows)
            if [r[id_field] for r in rows] != [c[id_field] for c in cases]:
                raise ValueError("Completed stage does not match its selected cases.")
            state["completed"] += len(rows)
        else:
            # Count valid saved IDs; the existing resumable helper repairs a torn last line.
            saved = {}
            if checkpoint.exists():
                lines = checkpoint.read_text(encoding="utf-8").splitlines()
                for i, line in enumerate(lines):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        if i != len(lines) - 1:
                            raise
                        break
                    saved[row[id_field]] = row
            expected_ids = {c[id_field] for c in cases}
            if saved.keys() - expected_ids:
                raise ValueError("Checkpoint contains cases outside this workload.")
            validate_rows(saved.values())
            state["completed"] += sum(not r["inference_error"] for r in saved.values())

            def progress(row):
                state["completed"] += 1
                state["active_seconds"] = time.monotonic() - started
                if row["inference_latency_seconds"] is not None:
                    new_latencies.append(row["inference_latency_seconds"])
                if new_latencies:
                    state["remaining_seconds"] = (
                        sum(new_latencies)
                        / len(new_latencies)
                        * (state["total"] - state["completed"])
                    )
                save_json(path / "status.json", state)
                stop_check()  # response has already been flushed to the checkpoint

            save_json(path / "status.json", state)
            rows = infer_cases_resumable(
                client,
                manifest["model"],
                cases,
                seed,
                temperature,
                checkpoint,
                resume=True,
                id_field=id_field,
                progress_prefix=name,
                on_result=progress,
            )
            if name == "sensitivity":
                for case, row in zip(cases, rows, strict=True):
                    row.update(pair_id=case["pair_id"], side=case["side"])
            write_jsonl(final, rows)
            checkpoint.unlink(missing_ok=True)
        outputs[name] = rows
        save_json(path / "status.json", state)
    metrics = {}
    for test in plan["config"]["tests"]:
        if test == "baseline":
            metrics[test] = calculate_metrics(outputs[test])
        elif test == "consistency":
            runs = [outputs[f"consistency_{i}"] for i in range(1, 6)]
            metrics[test] = calculate_consistency_metrics(
                build_case_rows(runs), runs, [calculate_metrics(r) for r in runs]
            )
        elif test == "robustness":
            metrics[test] = calculate_robustness_metrics(
                *(outputs[f"robustness_{key}"] for key in ("control", "reorder", "distractor"))
            )
        else:
            metrics[test] = calculate_sensitivity_metrics(
                build_pair_rows(plan["inputs"]["pairs"], outputs[test])
            )
    save_json(path / "metrics.json", metrics)
    state.update(state="complete", remaining_seconds=0, active_seconds=time.monotonic() - started)
    save_json(path / "status.json", state)


def worker(path, client=None):
    try:
        execute(path, client)
    except Exception as error:  # noqa: BLE001 - persist unexpected worker failures for the UI
        status_path = path / "status.json"
        state = read_json(status_path) if status_path.exists() else {}
        state.update(state="paused" if isinstance(error, Paused) else "error", error=str(error))
        save_json(status_path, state)


if __name__ == "__main__":
    if os.environ.get("MODELLENS_LOCAL") != "1":
        raise SystemExit("Enable local mode through the ModelLens launcher.")
    # The launching parent retains the lock until this inherited descriptor closes.
    with os.fdopen(int(sys.argv[2]), "a"):
        worker(Path(sys.argv[1]))
