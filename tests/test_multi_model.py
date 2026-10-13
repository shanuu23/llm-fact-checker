"""Multi-model support: model slugs, result paths, CLI flags, and the VitaminC export handling.

The end-to-end tests talk to a tiny fake Ollama server, so they exercise the real HTTP client and
run orchestration without a model, and without touching the committed results.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

try:
    from tests.fake_ollama import FakeOllama
except ModuleNotFoundError:  # run with tests/ itself on sys.path
    from fake_ollama import FakeOllama

from llm_fact_checker.cli import _results_dir, build_parser, experiment_dir
from llm_fact_checker.common import file_sha256, read_jsonl
from llm_fact_checker.experiments.baseline import run_inference
from llm_fact_checker.experiments.consistency import run_consistency_experiment
from llm_fact_checker.experiments.robustness import run_robustness_experiment
from llm_fact_checker.experiments.sensitivity import run_vitaminc_sensitivity
from llm_fact_checker.models import DEFAULT_MODEL, model_slug

ROOT = Path(__file__).resolve().parents[1]
FEVER_SAMPLE = ROOT / "data" / "fever" / "processed" / "fever_baseline_150.jsonl"
FEVER_PREPARED = ROOT / "data" / "fever" / "prepared"
VITAMINC_PAIRS = ROOT / "data" / "vitaminc" / "processed" / "vitaminc_sensitivity_100_pairs.jsonl"
VITAMINC_RAW = ROOT / "data" / "vitaminc" / "raw" / "vitaminc" / "test.jsonl"


def quiet():
    return contextlib.redirect_stderr(io.StringIO())


class ModelSlugTests(unittest.TestCase):
    def test_default_model_slug_matches_the_committed_pilot_folder(self):
        self.assertEqual(model_slug(DEFAULT_MODEL), "llama3.2_3b")

    def test_tags_become_folder_names(self):
        self.assertEqual(model_slug("qwen2.5:7b"), "qwen2.5_7b")
        self.assertEqual(model_slug("gemma3:4b"), "gemma3_4b")
        self.assertEqual(model_slug("hf.co/org/model:Q4_K_M"), "hf.co_org_model_Q4_K_M")

    def test_unsafe_tags_are_rejected(self):
        for bad in ("", "   ", "../escape", "has space", ".hidden"):
            with self.subTest(tag=bad), self.assertRaises(ValueError):
                model_slug(bad)


class PathAndFlagTests(unittest.TestCase):
    def test_experiment_dir_is_derived_from_the_model(self):
        self.assertEqual(
            experiment_dir("fever", "qwen2.5:7b", "baseline"),
            Path("results/fever/qwen2.5_7b/baseline"),
        )
        self.assertEqual(
            experiment_dir("vitaminc", DEFAULT_MODEL, "sensitivity"),
            Path("results/vitaminc/llama3.2_3b/sensitivity"),
        )

    def test_default_paths_are_unchanged_for_the_default_model(self):
        self.assertEqual(
            _results_dir(None, None), Path("results/fever/llama3.2_3b/baseline")
        )
        self.assertEqual(
            _results_dir(3, None), Path("results/fever/llama3.2_3b/smoke_3")
        )

    def test_smoke_runs_stay_inside_the_model_folder(self):
        self.assertEqual(
            _results_dir(3, None, "qwen2.5:7b"), Path("results/fever/qwen2.5_7b/smoke_3")
        )

    def test_explicit_results_dir_wins(self):
        self.assertEqual(_results_dir(None, Path("elsewhere"), "qwen2.5:7b"), Path("elsewhere"))

    def test_every_model_running_command_takes_a_model_flag(self):
        parser = build_parser()
        for command in (
            "run",
            "evaluate",
            "consistency",
            "robustness",
            "robustness-smoke",
            "vitaminc-sensitivity",
            "vitaminc-smoke",
            "all",
        ):
            with self.subTest(command=command):
                self.assertEqual(parser.parse_args([command]).model, DEFAULT_MODEL)
                parsed = parser.parse_args([command, "--model", "qwen2.5:7b"])
                self.assertEqual(parsed.model, "qwen2.5:7b")

    def test_prepare_commands_do_not_need_a_model(self):
        parser = build_parser()
        for command in ("prepare", "robustness-prepare", "vitaminc-prepare"):
            with self.subTest(command=command):
                self.assertFalse(hasattr(parser.parse_args([command]), "model"))

    def test_robustness_prepared_files_default_to_the_shared_data_folder(self):
        args = build_parser().parse_args(["robustness"])
        self.assertEqual(args.prepared_dir, Path("data/fever/prepared"))
        self.assertIsNone(args.robustness_dir)


@unittest.skipUnless(FEVER_SAMPLE.exists(), "committed FEVER sample not present")
class EndToEndWithFakeModelTests(unittest.TestCase):
    """Run each experiment against a fake server with a model that is not the default."""

    MODEL = "fake-model:1b"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_baseline_uses_the_requested_model_and_records_it(self):
        with FakeOllama() as server, quiet():
            metrics = run_inference(
                FEVER_SAMPLE, self.out / "baseline", limit=3, base_url=server.url, model=self.MODEL
            )
        self.assertEqual(metrics["total_cases"], 3)
        config = json.loads((self.out / "baseline" / "run_config.json").read_text())
        self.assertEqual(config["ollama_model"], self.MODEL)
        self.assertEqual(config["ollama_model_metadata"]["digest"], "digest-of-fake-model:1b")
        self.assertEqual(config["prepared_dataset_sha256"], file_sha256(FEVER_SAMPLE))
        self.assertEqual({r["model"] for r in server.chat_requests}, {self.MODEL})
        self.assertTrue(all(r["options"]["temperature"] == 0 for r in server.chat_requests))

    def test_requesting_a_model_that_is_not_installed_fails_clearly(self):
        with FakeOllama(installed=("other:1b",)) as server, quiet():
            with self.assertRaisesRegex(RuntimeError, "not installed"):
                run_inference(
                    FEVER_SAMPLE, self.out / "b", limit=1, base_url=server.url, model=self.MODEL
                )

    def test_consistency_report_names_the_requested_model(self):
        with FakeOllama() as server, quiet():
            run_consistency_experiment(
                FEVER_SAMPLE, self.out / "c", limit=2, base_url=server.url, model=self.MODEL
            )
        report = (self.out / "c" / "report.md").read_text(encoding="utf-8")
        self.assertIn(f"`{self.MODEL}`", report)
        self.assertNotIn("llama3.2", report)
        self.assertEqual(len(server.chat_requests), 2 * 5)  # two cases, five runs

    def test_robustness_reads_shared_prepared_files_not_the_model_folder(self):
        robustness_dir = self.out / "fever" / "fake-model_1b" / "robustness"
        self.assertFalse((robustness_dir / "prepared").exists())
        with FakeOllama() as server, quiet():
            metrics = run_robustness_experiment(
                FEVER_SAMPLE,
                robustness_dir,
                limit=3,
                base_url=server.url,
                model=self.MODEL,
                prepared_dir=FEVER_PREPARED,
            )
        self.assertEqual(metrics["case_count"], 3)
        smoke = robustness_dir / "smoke"
        self.assertTrue((smoke / "robustness_metrics.json").exists())
        self.assertEqual(
            json.loads((smoke / "run_config.json").read_text())["ollama_model"], self.MODEL
        )

    @unittest.skipUnless(VITAMINC_RAW.exists(), "raw VitaminC test.jsonl not downloaded")
    def test_sensitivity_runs_without_a_prepared_copy_in_the_model_folder(self):
        sensitivity_dir = self.out / "vitaminc" / "fake-model_1b" / "sensitivity"
        with FakeOllama() as server, quiet():
            metrics = run_vitaminc_sensitivity(
                VITAMINC_RAW,
                VITAMINC_PAIRS,
                sensitivity_dir,
                limit=2,
                base_url=server.url,
                model=self.MODEL,
            )
        self.assertEqual(metrics["total_pairs"], 2)
        self.assertEqual(len(server.chat_requests), 4)  # two pairs, support + refute
        predictions = list(read_jsonl(sensitivity_dir / "smoke" / "predictions" / "predictions.jsonl"))
        self.assertEqual(len(predictions), 4)

    def test_default_model_and_other_model_never_share_a_results_folder(self):
        self.assertNotEqual(
            experiment_dir("fever", DEFAULT_MODEL, "baseline"),
            experiment_dir("fever", self.MODEL, "baseline"),
        )


if __name__ == "__main__":
    unittest.main()
