"""Offline local evaluator checks: no real Ollama, workers or model inference."""

from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

from app import jobs
from app.evaluator import (
    MAX_RESPONSE_BYTES,
    ROOT,
    TESTS,
    LocalOllamaClient,
    calibrate,
    estimated_seconds,
    make_plan,
    model_identity,
    stages,
)
from llm_fact_checker.common import read_jsonl, write_jsonl
from llm_fact_checker.ollama import OllamaRequestError
from llm_fact_checker.prompts import SYSTEM_PROMPT

MODEL = "test-model:small"
METADATA = {
    "version": "test-version",
    "model": {"name": MODEL, "digest": "test-digest"},
    "details_sha256": "test-details",
}


class FakeClient:
    def __init__(self, action=None):
        self.calls = []
        self.action = action

    def check(self, model):
        return deepcopy(METADATA)

    def chat(self, model, claim, evidence, seed=42, temperature=0):
        self.calls.append((claim, evidence, seed, temperature))
        if self.action:
            self.action(len(self.calls))
        return "SUPPORTED", 0.2, {}


def response(data=None, status=200, raw=None):
    result = MagicMock(status=status)
    result.read.return_value = (
        raw if raw is not None else json.dumps(data if data is not None else {}).encode()
    )
    return result


@patch("app.evaluator.http.client.HTTPConnection")
class LocalClientTests(unittest.TestCase):
    def test_listing_is_loopback_only_and_does_not_infer(self, transport):
        transport.return_value.getresponse.return_value = response(
            {
                "models": [
                    {"name": MODEL},
                    {"name": "remote:cloud"},
                    {"name": "alias", "remote_host": "cloud"},
                ]
            }
        )
        client = LocalOllamaClient()
        transport.assert_not_called()
        self.assertEqual(client.list_models(), [MODEL])
        transport.assert_called_once_with("127.0.0.1", 11434, timeout=10)
        request = transport.return_value.request.call_args
        self.assertEqual(request.args, ("GET", "/api/tags"))
        self.assertNotIn("Authorization", request.kwargs["headers"])

    def test_uses_existing_prompt_and_generation_settings(self, transport):
        transport.return_value.getresponse.return_value = response(
            {"message": {"content": "SUPPORTED"}, "done": True}
        )
        LocalOllamaClient().chat(
            MODEL,
            "claim",
            [{"page": "Example", "sentence_id": 0, "text": "evidence"}],
            seed=101,
            temperature=0.2,
        )
        payload = json.loads(transport.return_value.request.call_args.kwargs["body"])
        self.assertEqual(payload["messages"][0]["content"], SYSTEM_PROMPT)
        self.assertEqual(payload["options"], {"seed": 101, "temperature": 0.2})
        self.assertFalse(payload["stream"])

    def test_rejects_redirects_bad_json_and_incomplete_answers(self, transport):
        for reply in (
            response(status=302),
            response(raw=b"secret invalid JSON"),
            response(raw=b"x" * (MAX_RESPONSE_BYTES + 1)),
            response([]),
            response({"done": False}),
            response({"done": True, "message": []}),
        ):
            transport.return_value.getresponse.return_value = reply
            with self.assertRaises(OllamaRequestError) as caught:
                LocalOllamaClient()._request("/api/chat", {})
            self.assertNotIn("secret", str(caught.exception))

    def test_timeout_does_not_retry(self, transport):
        transport.return_value.request.side_effect = TimeoutError("private")
        with self.assertRaisesRegex(OllamaRequestError, "timed out"):
            LocalOllamaClient().chat(MODEL, "claim", [])
        self.assertEqual(transport.return_value.request.call_count, 1)

    def test_no_pull_or_other_endpoints(self, transport):
        with self.assertRaises(OllamaRequestError):
            LocalOllamaClient()._request("/api/pull", {"model": MODEL})
        transport.assert_not_called()

    def test_cloud_embeddings_and_missing_identity_rejected(self, transport):
        for detail in (
            {"capabilities": ["embedding"]},
            {"capabilities": "completion"},
            {"capabilities": ["completion"], "remote_model": "cloud"},
        ):
            transport.return_value.getresponse.side_effect = [
                response({"version": "test"}),
                response({"models": [METADATA["model"]]}),
                response(detail),
            ]
            with self.assertRaises(OllamaRequestError):
                LocalOllamaClient().check(MODEL)


class PlanTests(unittest.TestCase):
    def test_quick_plan_is_fixed_and_covers_all_tests(self):
        plan = make_plan("Quick demo", 1, list(TESTS))
        self.assertEqual(plan, make_plan("Quick demo", 100, list(reversed(TESTS))))
        self.assertEqual(len(plan["inputs"]["fever"]), 3)
        self.assertEqual(len(plan["inputs"]["pairs"]), 3)
        self.assertTrue(plan["inputs"]["reorder"])
        self.assertEqual(
            {c["gold_label"] for c in plan["inputs"]["fever"]}, {"SUPPORTED", "REFUTED"}
        )
        self.assertEqual(plan["config"]["expected_calls"], 31)
        self.assertEqual(
            [seed for name, _, seed, _, _ in stages(plan) if name.startswith("consistency")],
            [101, 202, 303, 404, 505],
        )

    def test_custom_uses_full_population_and_nested_selection(self):
        small = make_plan("Custom", 1, ["baseline", "sensitivity"])
        larger = make_plan("Custom", 2, ["baseline", "sensitivity"])
        self.assertEqual(small["config"]["sources"]["fever"]["population"], 13332)
        self.assertEqual(small["config"]["sources"]["vitaminc"]["population"], 20232)
        self.assertEqual(len(small["inputs"]["fever"]), 134)
        self.assertEqual(len(small["inputs"]["pairs"]), 203)
        for key in ("fever", "pairs"):
            self.assertEqual(
                small["inputs"][key], larger["inputs"][key][: len(small["inputs"][key])]
            )

    def test_plan_tampering_and_code_change_rejected(self):
        plan = make_plan("Quick demo", 1, ["baseline"])
        changed = deepcopy(plan)
        changed["inputs"]["fever"][0]["claim"] = "Changed"
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            jobs.check_plan(changed)
        with (
            patch("app.jobs.protocol_fingerprint", return_value="new-code"),
            self.assertRaisesRegex(ValueError, "code changed"),
        ):
            jobs.check_plan(plan)

    def test_timing_is_opt_in_not_scored_and_expires(self):
        plan = make_plan("Quick demo", 1, list(TESTS))
        client = FakeClient()
        timing = calibrate(client, MODEL, plan)
        self.assertEqual(len(client.calls), 16)
        lower, upper = estimated_seconds(plan, timing)
        self.assertAlmostEqual(lower, 6.2)
        self.assertAlmostEqual(upper, 6.2)
        timing["time"] -= 3601
        self.assertIsNone(estimated_seconds(plan, timing))


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def create(self, tests=tuple(TESTS), run_id="a" * 32):
        path = self.root / run_id
        path.mkdir()
        plan = make_plan("Quick demo", 1, tests)
        jobs.save_json(path / "plan.json", plan)
        jobs.save_json(
            path / "manifest.json",
            {
                "model": MODEL,
                "identity": model_identity(METADATA),
                "config": plan["config"],
                "created": "2026-10-16T00:00:00Z",
            },
        )
        jobs.save_json(path / "status.json", {"state": "starting"})
        return path

    def test_four_tests_complete_and_repeat_execute_makes_no_calls(self):
        path = self.create()
        client = FakeClient()
        jobs.worker(path, client)
        status = jobs.read_json(path / "status.json")
        self.assertEqual(status["state"], "complete", status)
        self.assertEqual(status["completed"], 31)
        self.assertEqual(len(client.calls), 31)
        metrics = jobs.read_json(path / "metrics.json")
        self.assertEqual(set(metrics), set(TESTS))
        self.assertEqual(metrics["sensitivity"]["total_pairs"], 3)
        jobs.execute(path, client)
        self.assertEqual(len(client.calls), 31)
        self.assertFalse(list(path.glob("*.partial.jsonl")))

    def test_pause_flushes_and_resume_skips_answered_calls(self):
        path = self.create(["baseline"])

        def stop(_):
            (path / "stop").touch()

        client = FakeClient(stop)
        jobs.worker(path, client)
        self.assertEqual(jobs.read_json(path / "status.json")["state"], "paused")
        self.assertEqual(len(list(read_jsonl(path / "baseline.partial.jsonl"))), 1)
        (path / "stop").unlink()
        client.action = None
        jobs.worker(path, client)
        self.assertEqual(jobs.read_json(path / "status.json")["state"], "complete")
        self.assertEqual(len(client.calls), 3)

    def test_failed_and_invalid_responses_remain_in_metrics(self):
        path = self.create(["baseline"])
        client = FakeClient()
        client.chat = MagicMock(
            side_effect=[
                ("SUPPORTED", 0.1, {}),
                ("bad output", 0.1, {}),
                OllamaRequestError("offline"),
            ]
        )
        jobs.worker(path, client)
        result = jobs.read_json(path / "metrics.json")["baseline"]
        self.assertEqual(result["total_cases"], 3)
        self.assertEqual(result["inference_failure_count"], 1)
        self.assertGreaterEqual(result["invalid_format_count"], 1)

    def test_changed_model_and_corrupt_saved_prediction_rejected(self):
        path = self.create(["baseline"])
        client = FakeClient()
        client.check = MagicMock(return_value={**METADATA, "version": "different"})
        with self.assertRaisesRegex(ValueError, "Model or Ollama"):
            jobs.execute(path, client)
        jobs.execute(path, FakeClient())
        (path / "metrics.json").unlink()
        rows = list(read_jsonl(path / "baseline.jsonl"))
        rows[0]["evidence"] = []
        write_jsonl(path / "baseline.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "predictions do not match"):
            jobs.execute(path, FakeClient())

    def test_lock_prevents_overlap_and_marks_crashed_runs_interrupted(self):
        self.create(["baseline"])
        with jobs.exclusive(self.root):
            self.assertTrue(jobs.busy(self.root))
            with self.assertRaises(ValueError), jobs.exclusive(self.root):
                pass
            self.assertEqual(jobs.list_runs(self.root)[0]["status"]["state"], "interrupted")
        self.assertFalse(jobs.busy(self.root))

    @patch.dict(os.environ, {"MODELLENS_LOCAL": "1"})
    @patch("app.jobs.subprocess.Popen")
    @patch("app.jobs.LocalOllamaClient", return_value=FakeClient())
    def test_start_is_explicit_and_hands_lock_to_worker(self, _factory, popen):
        plan = make_plan("Quick demo", 1, ["baseline"])
        run_id = jobs.start_run(MODEL, plan, root=self.root)
        self.assertTrue((self.root / run_id / "plan.json").exists())
        kwargs = popen.call_args.kwargs
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(len(kwargs["pass_fds"]), 1)
        self.assertEqual(jobs.list_runs(self.root)[0]["status"]["state"], "interrupted")

    def test_hosted_start_blocked_and_matching_required(self):
        with patch.dict(os.environ, {"MODELLENS_LOCAL": "0"}), self.assertRaises(ValueError):
            jobs.start_run(MODEL, {}, root=self.root)
        a = {"status": {"state": "complete"}, "config": {"workload_id": "a"}}
        b = deepcopy(a)
        a["identity"] = b["identity"] = model_identity(METADATA)
        self.assertIsNone(jobs.compatibility(a, b))
        b["config"]["workload_id"] = "b"
        self.assertIsNotNone(jobs.compatibility(a, b))
        b = deepcopy(a)
        b["identity"]["version"] = "other-version"
        self.assertIsNotNone(jobs.compatibility(a, b))
        for run_id in ("../data", "x" * 32):
            with self.assertRaises(ValueError):
                jobs.run_directory(self.root, run_id)


class LiveUITests(unittest.TestCase):
    def test_download_preserves_records_and_excludes_unrelated_files(self):
        from app.live import results_download

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("metrics.json", "manifest.json", "plan.json", "status.json"):
                jobs.save_json(root / name, {"test": name})
            write_jsonl(
                root / "baseline.jsonl", [{"claim": "example", "raw_model_output": "SUPPORTED"}]
            )
            write_jsonl(root / "baseline.partial.jsonl", [{"unfinished": True}])
            jobs.save_json(root / "unrelated.json", {"private": True})
            with ZipFile(BytesIO(results_download(root))) as archive:
                self.assertEqual(
                    set(archive.namelist()),
                    {
                        "metrics.json",
                        "manifest.json",
                        "plan.json",
                        "status.json",
                        "baseline.jsonl",
                    },
                )
                self.assertIn(b"SUPPORTED", archive.read("baseline.jsonl"))

    def app(self):
        from streamlit.testing.v1 import AppTest

        return AppTest.from_file(str(ROOT / "app/main.py"), default_timeout=30).run()

    def saved_run(self, root, letter, model=MODEL, state="complete"):
        path = root / (letter * 32)
        path.mkdir()
        jobs.save_json(path / "metrics.json", {})
        return {
            "id": path.name,
            "path": path,
            "model": model,
            "config": make_plan("Quick demo", 1, list(TESTS))["config"],
            "identity": model_identity(METADATA),
            "created": "2026-10-16T00:00:00Z",
            "status": {"state": state, "completed": 31 if state == "complete" else 0},
        }

    @patch.dict(os.environ, {"MODELLENS_LOCAL": "0"})
    @patch("app.live.LocalOllamaClient")
    def test_hosted_mode_never_connects(self, factory):
        app = self.app()
        app.button(key="try_evaluator").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("disabled in hosted" in m.value for m in app.info))
        factory.assert_not_called()
        self.assertEqual(len(app.text_input), 0)

    @patch.dict(os.environ, {"MODELLENS_LOCAL": "1"})
    @patch("app.live.jobs.list_runs", return_value=[])
    @patch("app.live.jobs.busy", return_value=False)
    @patch("app.live.jobs.start_run", return_value="a" * 32)
    @patch("app.live.LocalOllamaClient")
    def test_connect_and_run_only_on_click_and_refresh_no_repeat(
        self, factory, start, _busy, _runs
    ):
        factory.return_value.list_models.return_value = [MODEL]
        app = self.app()
        app.button(key="try_evaluator").click().run()
        factory.assert_not_called()
        app.button(key="live_connect").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.selectbox(key="live_model").value, MODEL)
        start.assert_not_called()
        app.button(key="live_run").click().run()
        self.assertEqual(len(app.exception), 0)
        start.assert_called_once()
        app.run()
        start.assert_called_once()
        factory.return_value.chat.assert_not_called()

    @patch.dict(os.environ, {"MODELLENS_LOCAL": "1"})
    @patch("app.live.LocalOllamaClient")
    def test_completed_history_results_and_comparison(self, factory):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = make_plan("Quick demo", 1, list(TESTS))
            for run_id in ("a" * 32, "b" * 32):
                path = root / run_id
                path.mkdir()
                jobs.save_json(path / "plan.json", plan)
                jobs.save_json(
                    path / "manifest.json",
                    {
                        "model": MODEL,
                        "identity": model_identity(METADATA),
                        "config": plan["config"],
                        "created": "2026-10-16T00:00:00Z",
                    },
                )
                jobs.execute(path, FakeClient())
            runs = jobs.list_runs(root)
            with patch("app.live.jobs.list_runs", return_value=runs):
                app = self.app()
                app.button(key="try_evaluator").click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(any(MODEL in item.value for item in app.markdown))
                stats = [
                    element.proto.body for element in app.get("html")
                    if 'class="result-stat"' in element.proto.body
                ]
                self.assertEqual(len(stats), 5)
                metrics = jobs.read_json(runs[0]["path"] / "metrics.json")
                expected_values = (
                    metrics["baseline"]["overall_accuracy"],
                    metrics["consistency"]["exact_consistency_rate"],
                    metrics["robustness"]["reorder"]["invariance_rate"],
                    metrics["robustness"]["distractor"]["invariance_rate"],
                    metrics["sensitivity"]["correct_sensitivity_rate"],
                )
                for body, value in zip(stats, expected_values, strict=True):
                    self.assertIn(f'class="result-stat-value">{value:.1%}</div>', body)
                for body, label in zip(stats, (
                    "Overall accuracy", "Same answer in all five runs",
                    "Reordered evidence", "Added distractor", "Both evidence versions correct",
                )):
                    self.assertIn(label, body)
                    self.assertIn('class="result-stat-value"', body)
                    self.assertIn('class="result-stat-detail"', body)
                self.assertEqual(len(app.selectbox(key="live_history_model").options), 1)
                self.assertEqual(len(app.selectbox(key="live_history").options), 2)
                app.button(key="live_open_compare").click().run()
                self.assertEqual(len(app.exception), 0)
                app.button(key="local_compare").click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.tabs), 4)
                factory.assert_not_called()

    @patch.dict(os.environ, {"MODELLENS_LOCAL": "1"})
    @patch("app.live.LocalOllamaClient")
    @patch("app.live.jobs.busy", return_value=False)
    def test_unique_models_default_to_latest_compatible_runs(self, busy, factory):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            newest = self.saved_run(root, "c")
            newest["config"]["workload_id"] = "another-workload"
            older = self.saved_run(root, "a")
            other = self.saved_run(root, "b", "another-model:small")
            with patch("app.live.jobs.list_runs", return_value=[newest, older, other]):
                app = self.app()
                app.button(key="try_evaluator").click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.selectbox(key="live_history_model").options), 2)
                self.assertEqual(len(app.selectbox(key="live_history").options), 2)
                app.button(key="live_open_compare").click().run()
                self.assertEqual(len(app.selectbox(key="local_model_1").options), 2)
                self.assertEqual(app.selectbox(key="local_run_1").value, older["id"])
                self.assertEqual(app.selectbox(key="local_run_2").value, other["id"])
                app.selectbox(key="local_run_1").select(newest["id"]).run()
                app.button(key="local_compare").click().run()
                self.assertTrue(any("Different cases" in m.value for m in app.warning))
                self.assertEqual(len(app.tabs), 0)
                app.session_state["live_pending_run"] = other["id"]
                app.button(key="try_evaluator").click().run()
                self.assertEqual(app.selectbox(key="live_history_model").value, other["model"])
                self.assertEqual(app.selectbox(key="live_history").value, other["id"])
                self.assertEqual(len(app.exception), 0)
                factory.assert_not_called()

    @patch.dict(os.environ, {"MODELLENS_LOCAL": "1"})
    @patch("app.live.jobs.list_runs", return_value=[])
    @patch("app.live.jobs.busy", return_value=False)
    @patch("app.live.jobs.start_run", return_value="a" * 32)
    @patch("app.live.LocalOllamaClient")
    def test_setup_tabs_selection_and_no_automatic_inference(self, factory, start, _busy, _runs):
        app = self.app()
        app.session_state["live_models"] = [MODEL]
        app.button(key="try_evaluator").click().run()
        self.assertEqual([tab.label for tab in app.tabs], ["Evaluation", "Run history"])
        self.assertEqual(len(app.slider), 0)
        for kind in ("consistency", "robustness", "sensitivity"):
            app.checkbox(key=f"live_test_{kind}").uncheck()
        app.run()
        factory.assert_not_called()
        start.assert_not_called()
        app.button(key="live_run").click().run()
        self.assertEqual(start.call_args.args[1]["config"]["tests"], ["baseline"])
        app.checkbox(key="live_test_baseline").uncheck().run()
        self.assertTrue(app.button(key="live_run").disabled)
        self.assertTrue(any("at least one" in m.value for m in app.info))
        self.assertEqual(len(app.exception), 0)

    @patch.dict(os.environ, {"MODELLENS_LOCAL": "1"})
    @patch("app.live.jobs.list_runs", return_value=[])
    @patch("app.live.jobs.busy", return_value=False)
    @patch("app.live.cached_plan")
    @patch("app.live.LocalOllamaClient")
    def test_custom_workload_confirmation(self, factory, prepare, _busy, _runs):
        plan = make_plan("Quick demo", 1, list(TESTS))
        plan["config"].update(mode="Custom", expected_calls=2000)
        prepare.return_value = plan
        app = self.app()
        app.session_state["live_models"] = [MODEL]
        app.button(key="try_evaluator").click().run()
        app.radio(key="live_mode").set_value("Custom").run()
        app.slider(key="live_percentage").set_value(25).run()
        self.assertEqual(prepare.call_args.args[:2], ("Custom", 25))
        self.assertTrue(app.button(key="live_run").disabled)
        next(c for c in app.checkbox if "may take hours" in c.label).check().run()
        self.assertFalse(app.button(key="live_run").disabled)
        self.assertEqual(len(app.exception), 0)
        factory.assert_not_called()

    @patch.dict(os.environ, {"MODELLENS_LOCAL": "1"})
    @patch("app.live.jobs.list_runs", return_value=[])
    @patch("app.live.jobs.busy", return_value=False)
    @patch("app.live.LocalOllamaClient")
    def test_connection_error_does_not_enable_run(self, factory, _busy, _runs):
        factory.return_value.list_models.side_effect = RuntimeError("Ollama is unavailable")
        app = self.app()
        app.button(key="try_evaluator").click().run()
        app.button(key="live_connect").click().run()
        self.assertTrue(app.button(key="live_run").disabled)
        self.assertTrue(any("Ollama is unavailable" in e.value for e in app.error))
        self.assertEqual(len(app.exception), 0)
        factory.return_value.chat.assert_not_called()

    @patch.dict(os.environ, {"MODELLENS_LOCAL": "1"})
    @patch("app.live.LocalOllamaClient")
    @patch("app.live.jobs.busy", return_value=True)
    def test_progress_completion_and_lock_release_refresh_controls(self, busy, factory):
        with tempfile.TemporaryDirectory() as directory:
            run = self.saved_run(Path(directory), "a", state="running")
            with patch("app.live.jobs.list_runs", side_effect=lambda: [deepcopy(run)]):
                app = self.app()
                app.session_state["live_models"] = [MODEL]
                app.button(key="try_evaluator").click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertFalse(any(b.key == "live_run" for b in app.button))
                self.assertTrue(any(e.type == "progress" for e in app.main))
                run["status"].update(state="complete", completed=31)
                app.run()
                self.assertTrue(any(MODEL in item.value for item in app.markdown))
                self.assertFalse(any(e.type == "progress" for e in app.main))
                self.assertTrue(app.button(key="live_current_new").disabled)
                self.assertEqual(
                    [item.proto.label for item in app.get("download_button")],
                    ["Export results"],
                )
                self.assertFalse(
                    any(e.label == "Dataset and protocol details" for e in app.expander)
                )
                busy.return_value = False
                app.run()
                self.assertFalse(app.button(key="live_current_new").disabled)
                self.assertEqual(
                    app.session_state["live_observed_status"], (run["id"], "complete", False)
                )
                self.assertEqual(len(app.exception), 0)
                factory.assert_not_called()
                app.button(key="live_current_new").click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertFalse(app.button(key="live_run").disabled)
                self.assertEqual(app.selectbox(key="live_model").value, MODEL)

    @patch.dict(os.environ, {"MODELLENS_LOCAL": "1"})
    @patch("app.live.LocalOllamaClient")
    @patch("app.live.jobs.busy", return_value=True)
    @patch("app.live.jobs.request_pause")
    def test_history_selection_does_not_hide_active_run(self, pause, _busy, factory):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active = self.saved_run(root, "a", state="running")
            older = self.saved_run(root, "b")
            with patch("app.live.jobs.list_runs", return_value=[active, older]):
                app = self.app()
                app.button(key="try_evaluator").click().run()
                app.selectbox(key="live_history").select(older["id"]).run()
                self.assertEqual(app.session_state["live_current_run"], active["id"])
                self.assertEqual(app.session_state["live_history"], older["id"])
                app.button(key="live_current_pause").click().run()
                pause.assert_called_once_with(active["id"])
                self.assertEqual(len(app.exception), 0)
                factory.assert_not_called()

    @patch.dict(os.environ, {"MODELLENS_LOCAL": "1"})
    @patch("app.live.LocalOllamaClient")
    @patch("app.live.jobs.start_run")
    @patch("app.live.jobs.busy", return_value=False)
    def test_new_evaluation_restores_choices_without_restarting(self, busy, start, factory):
        with tempfile.TemporaryDirectory() as directory:
            run = self.saved_run(Path(directory), "a", state="running")
            runs = []
            with patch("app.live.jobs.list_runs", side_effect=lambda: deepcopy(runs)):
                app = self.app()
                app.session_state["live_models"] = [MODEL]
                app.button(key="try_evaluator").click().run()
                app.checkbox(key="live_test_sensitivity").uncheck().run()

                def begin(*args):
                    runs.append(run)
                    busy.return_value = True
                    return run["id"]

                start.side_effect = begin
                app.button(key="live_run").click().run()
                self.assertFalse(any(b.key == "live_run" for b in app.button))
                run["status"].update(state="paused")
                busy.return_value = False
                app.run()
                self.assertTrue(any(b.key == "live_current_resume" for b in app.button))
                self.assertFalse(any(b.key == "live_run" for b in app.button))
                run["status"].update(state="complete")
                app.run()
                self.assertFalse(any(e.type == "progress" for e in app.main))
                app.button(key="live_current_new").click().run()
                self.assertFalse(app.checkbox(key="live_test_sensitivity").value)
                self.assertEqual(app.selectbox(key="live_model").value, MODEL)
                self.assertEqual(app.session_state["live_tab"], "Evaluation")
                start.assert_called_once()
                factory.assert_not_called()
                self.assertTrue((run["path"] / "metrics.json").exists())
                self.assertEqual(len(app.exception), 0)


class PackageTests(unittest.TestCase):
    def test_allowlist_contains_inputs_and_no_private_or_large_logs(self):
        from scripts.package_app import package_files

        paths = [p.relative_to(ROOT) for p in package_files()]
        self.assertIn(Path("data/fever/full/fever_dev_full.jsonl"), paths)
        self.assertIn(Path("data/vitaminc/raw/vitaminc/test.jsonl"), paths)
        self.assertIn(Path("data/README.md"), paths)
        self.assertIn(Path("scripts/macos_setup.sh"), paths)
        for path in paths:
            self.assertFalse(
                set(path.parts)
                & {".git", ".venv", ".modellens", "__pycache__", "smoke", "smoke_3", "train_full"}
            )
            self.assertNotIn(path.name, {".env", "secrets.toml", "predictions.jsonl"})


if __name__ == "__main__":
    unittest.main()
