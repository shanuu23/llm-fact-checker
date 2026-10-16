"""Chart contracts, without a browser or model calls."""

import unittest
from unittest.mock import patch

from app.charts import COLORS, STATES, consistency_chart, paired_chart, show_validity_warnings
from app.comparison import load_models


class ChartTests(unittest.TestCase):
    @patch("app.charts.st.warning")
    def test_validity_warnings_only_for_failed_or_invalid_requests(self, warning):
        show_validity_warnings("baseline", [{"inference_failure_count": 0}], ["A"])
        warning.assert_not_called()
        show_validity_warnings(
            "sensitivity",
            [{"technical_validity": {"inference_failures": 1, "invalid_formats": 2}}],
            ["B"],
        )
        warning.assert_called_once_with("B: 1 inference failures · 2 invalid responses")
        warning.reset_mock()
        show_validity_warnings(
            "robustness",
            [
                {
                    "technical_validity": {
                        "control": {"inference_failure_count": 2},
                        "reorder": {"invalid_format_count": 3},
                    }
                }
            ],
            ["C"],
        )
        warning.assert_called_once_with("C: 2 inference failures · 3 invalid responses")

    def test_fixed_axes_labels_and_missing_values(self):
        names = ["Same model · run-a", "Same model · run-b"]
        spec = paired_chart(["Overall", "Refutes"], [[1, None], [0, 0.625]], names).to_dict()
        self.assertEqual(spec["autosize"]["type"], "fit-x")
        self.assertEqual(spec["height"], 224)
        self.assertEqual(spec["config"]["axis"]["labelFontSize"], 15)
        self.assertEqual(spec["config"]["legend"]["labelFontSize"], 15)
        self.assertEqual(spec["config"]["text"]["fontSize"], 15)
        self.assertEqual(spec["layer"][0]["mark"]["size"], 26)
        self.assertGreater(spec["layer"][0]["encoding"]["yOffset"]["scale"]["paddingInner"], 0)
        rows = next(iter(spec["datasets"].values()))
        self.assertEqual([r["Label"] for r in rows], ["100.0%", "N/A", "0.0%", "62.5%"])
        self.assertIsNone(rows[1]["Value"])
        self.assertEqual({r["Model"] for r in rows}, set(names))
        for layer in spec["layer"]:
            axis = layer["encoding"]["x"]
            self.assertEqual(axis["scale"]["domain"], [0, 1])
            self.assertIsNone(axis["title"])
            self.assertEqual(axis["axis"]["format"], ".0%")
        self.assertEqual(spec["layer"][0]["encoding"]["color"]["scale"]["range"], COLORS)

    def test_stacks_preserve_saved_outcomes_and_total(self):
        metric = load_models()["phi4-mini_3.8b"]["consistency"].metrics()
        spec = consistency_chart([metric, metric], ["Phi A", "Phi B"]).to_dict()
        self.assertEqual(spec["autosize"]["type"], "fit-x")
        self.assertEqual(spec["height"], 190)
        self.assertEqual(spec["config"]["axis"]["labelFontSize"], 15)
        self.assertEqual(spec["config"]["legend"]["labelFontSize"], 15)
        self.assertEqual(spec["config"]["text"]["fontSize"], 15)
        self.assertEqual(spec["layer"][0]["mark"]["size"], 42)
        self.assertGreater(spec["height"] / 2, spec["layer"][0]["mark"]["size"])
        rows = next(iter(spec["datasets"].values()))
        for name in ("Phi A", "Phi B"):
            selected = [r for r in rows if r["Model"] == name]
            self.assertAlmostEqual(selected[-1]["End"], 1)
            for row, (key, _, _) in zip(selected, STATES, strict=True):
                self.assertEqual(
                    row["Value"], metric["stability_classification"][key]["percentage"]
                )
        self.assertEqual(spec["layer"][0]["encoding"]["x"]["scale"]["domain"], [0, 1])

    def test_zero_eligible_scores_remain_na(self):
        spec = paired_chart(
            ["Original", "Changed"], [[None, None], [None, None]], ["A", "B"]
        ).to_dict()
        rows = next(iter(spec["datasets"].values()))
        self.assertTrue(all(row["Value"] is None and row["Label"] == "N/A" for row in rows))

    def test_single_measure_labels_each_model_even_at_zero(self):
        spec = paired_chart(["Both correct"], [[0], [1]], ["Llama", "Granite"]).to_dict()
        self.assertEqual(spec["height"], 140)
        for layer in spec["layer"]:
            self.assertEqual(layer["encoding"]["y"]["field"], "Model")
            self.assertNotIn("yOffset", layer["encoding"])
        self.assertIsNone(spec["layer"][0]["encoding"]["color"]["legend"])

    @patch("app.charts.st.altair_chart")
    @patch("app.charts.st.container")
    def test_full_width_charts_use_centered_width_cap(self, container, render):
        from app.charts import show_centered_chart

        chart = paired_chart(["Overall"], [[0], [1]], ["A", "B"])
        show_centered_chart(chart, "test_chart")
        self.assertEqual(container.call_args_list[0].kwargs, {"horizontal_alignment": "center"})
        self.assertEqual(container.call_args_list[1].kwargs, {"width": 900})
        render.assert_called_once_with(chart, width="stretch", key="test_chart")

    def test_identical_robustness_panels_have_distinct_widget_keys(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_string("""
from app.charts import show_comparison_charts
block = {"eligible_pair_count": 3, "control_accuracy": 1.0,
         "perturbed_accuracy": 1.0, "invariance_rate": 1.0, "complete_pair_count": 3}
metrics = {"reorder": block, "distractor": block}
show_comparison_charts("robustness", [metrics, metrics], ["Model A", "Model B"])
""").run()
        self.assertEqual(len(app.exception), 0)

    def test_reordering_chart_uses_control_on_the_same_subset(self):
        from streamlit.testing.v1 import AppTest

        metrics = load_models()["phi4-mini_3.8b"]["robustness"].metrics()
        with patch("app.charts.paired_chart", wraps=paired_chart) as chart:
            app = AppTest.from_string("""
from app.charts import show_comparison_charts
from app.comparison import load_models
metrics = load_models()["phi4-mini_3.8b"]["robustness"].metrics()
show_comparison_charts("robustness", [metrics, metrics], ["Model A", "Model B"])
""").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(chart.call_count, 2)
        original = chart.call_args_list[0].args[1][0][0]
        self.assertEqual(original, metrics["reorder"]["control_accuracy"])
        self.assertNotEqual(original, metrics["control"]["accuracy"])


if __name__ == "__main__":
    unittest.main()
