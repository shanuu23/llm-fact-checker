from __future__ import annotations

from copy import deepcopy
import importlib.util
from itertools import combinations
from pathlib import Path
import unittest
from unittest.mock import patch

from app.comparison import (
    comparison_issues,
    load_models,
    result_examples,
    robustness_rows,
    score_rows,
)
from llm_fact_checker.results import ResultRun

ROOT = Path(__file__).resolve().parents[1]


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.models = load_models()
        self.phi = self.models["phi4-mini_3.8b"]

    def test_only_full_results_are_discovered(self):
        self.assertGreaterEqual(len(self.models), 2)
        for runs in self.models.values():
            for run in runs.values():
                self.assertTrue(run.experiment.endswith("_full"))
                self.assertFalse(run.is_smoke)

    def test_current_models_have_matching_recorded_inputs_and_settings(self):
        for left, right in combinations(self.models.values(), 2):
            for kind in left.keys() & right.keys():
                with self.subTest(kind=kind, model1=left[kind].model, model2=right[kind].model):
                    self.assertEqual(comparison_issues(left[kind], right[kind]), [])

    def test_missing_or_different_inputs_are_not_accepted(self):
        left = self.phi["baseline"]
        right = self.models["llama3.2_3b"]["baseline"]
        config = left.config()
        for field, value in (("prepared_dataset_sha256", "changed"), ("temperature", None)):
            modified = deepcopy(config)
            modified[field] = value
            with patch.object(
                ResultRun, "config", side_effect=[config, modified, config, modified]
            ):
                self.assertTrue(comparison_issues(left, right))

    def test_sensitivity_uses_both_correct_not_any_flip(self):
        metrics = self.phi["sensitivity"].metrics()
        rows = score_rows("sensitivity", metrics)
        self.assertEqual(rows[0][1], metrics["correct_sensitivity_count"] / metrics["total_pairs"])
        self.assertEqual(rows[1][1], metrics["flip_count"] / metrics["complete_pairs"])
        self.assertNotEqual(rows[0][1], rows[1][1])

    def test_reordering_uses_control_on_the_same_subset(self):
        metrics = self.phi["robustness"].metrics()
        original = robustness_rows(metrics, "reorder")[0][1]
        self.assertEqual(original, metrics["reorder"]["control_accuracy"])
        self.assertNotEqual(original, metrics["control"]["accuracy"])

    def test_examples_work_without_reading_ignored_prediction_logs(self):
        with patch.object(
            ResultRun, "cases", side_effect=AssertionError("prediction log accessed")
        ):
            for run in self.phi.values():
                rows = result_examples(run, limit=2)
                self.assertLessEqual(len(rows), 2)
                self.assertIn("claim", rows[0])


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "install the app dependency group")
class DashboardTests(unittest.TestCase):
    def app(self):
        from streamlit.testing.v1 import AppTest

        return AppTest.from_file(str(ROOT / "app" / "main.py"), default_timeout=30).run()

    def test_overview_and_comparison_use_saved_results_only(self):
        with patch(
            "llm_fact_checker.ollama.OllamaClient.chat", side_effect=AssertionError("live call")
        ):
            app = self.app()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(app.button(key="try_evaluator").disabled)
            self.assertEqual(len(app.selectbox), 0)
            app.button(key="open_compare").click().run()
            app.selectbox(key="model_1").select("llama3.2_3b")
            app.selectbox(key="model_2").select("phi4-mini_3.8b")
            app.button(key="FormSubmitter:model_comparison-Compare").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.error), 0)
            self.assertEqual(len(app.warning), 0)
            self.assertEqual(
                [tab.label for tab in app.tabs],
                ["Correctness", "Consistency", "Robustness", "Sensitivity"],
            )
            self.assertEqual(
                app.session_state["compared_models"], ("llama3.2_3b", "phi4-mini_3.8b")
            )
            app.run()
            self.assertEqual(len(app.tabs), 4)

    def test_same_model_is_rejected(self):
        app = self.app()
        app.button(key="open_compare").click().run()
        selected = app.selectbox(key="model_1").value
        app.selectbox(key="model_2").select(selected)
        app.button(key="FormSubmitter:model_comparison-Compare").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.tabs), 0)
        self.assertIn("two different models", app.warning[0].value)

    def test_missing_experiment_is_displayed_as_unavailable(self):
        models = load_models()
        models["phi4-mini_3.8b"].pop("sensitivity")
        with patch("app.comparison.load_models", return_value=models):
            app = self.app()
            app.button(key="open_compare").click().run()
            app.selectbox(key="model_1").select("llama3.2_3b")
            app.selectbox(key="model_2").select("phi4-mini_3.8b")
            app.button(key="FormSubmitter:model_comparison-Compare").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("not available" in item.value for item in app.info))

    def test_empty_results_show_guidance(self):
        with patch("app.comparison.load_models", return_value={}):
            app = self.app()
            app.button(key="open_compare").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("at least two models" in item.value for item in app.info))


if __name__ == "__main__":
    unittest.main()
