"""Narratives follow the selected scores without inference or causal claims."""

import unittest

from app.comparison import load_models
from app.interpretation import compare_scores, interpretation


class InterpretationTests(unittest.TestCase):
    def test_winner_changes_with_pair_and_scores(self):
        self.assertIn("A scores higher", compare_scores(["A", "B"], [0.8, 0.6], "accuracy"))
        text = compare_scores(["New model", "Other model"], [0.2, 0.9], "accuracy")
        self.assertIn("Other model scores higher", text)
        self.assertIn("90.0%", text)
        self.assertIn("20.0%", text)

    def test_ties_rounding_missing_and_zero(self):
        self.assertIn("equal", compare_scores(["A", "B"], [0, 0], "accuracy"))
        self.assertIn("unrounded", compare_scores(["A", "B"], [0.60001, 0.60002], "accuracy"))
        self.assertIn("unavailable for A", compare_scores(["A", "B"], [None, 0], "accuracy"))
        self.assertIn("B scores higher", compare_scores(["A", "B"], [0, 1], "accuracy"))

    def test_mixed_class_results_and_demo(self):
        metrics = [
            {"overall_accuracy": 0.784, "class_accuracy": {"SUPPORTED": 0.941, "REFUTED": 0.628}},
            {"overall_accuracy": 0.599, "class_accuracy": {"SUPPORTED": 0.964, "REFUTED": 0.233}},
        ]
        text = interpretation("baseline", metrics, ["Gemma", "Phi"], demo=True)
        self.assertIn("Gemma scores higher on overall accuracy", text)
        self.assertIn("Phi scores higher on supporting cases", text)
        self.assertIn("Gemma scores higher on refuting cases", text)
        self.assertIn("too small", text)
        swapped = interpretation("baseline", metrics[::-1], ["Phi", "Gemma"])
        self.assertIn("Gemma scores higher on overall accuracy", swapped)

    def test_all_saved_metric_schemas_and_equal_scores(self):
        runs = load_models()["phi4-mini_3.8b"]
        for kind, run in runs.items():
            with self.subTest(kind=kind):
                metric = run.metrics()
                text = interpretation(kind, [metric, metric], ["First", "Second"])
                self.assertIn("equal", text)
                self.assertIn("First", text)
                self.assertIn("Second", text)
                self.assertIn("not overall model quality", text)
                if kind == "consistency":
                    self.assertIn("wrong every run", text)
                if kind == "robustness":
                    self.assertIn("original score", text)


if __name__ == "__main__":
    unittest.main()
