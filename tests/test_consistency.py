from __future__ import annotations

import unittest

from llm_fact_checker.experiments.consistency import (
    calculate_consistency_metrics,
    case_consistency,
)


SUPPORTED = "SUPPORTED"
REFUTED = "REFUTED"
NEI = "NOT ENOUGH INFORMATION"


def prediction(fever_id: int, gold: str, predicted: str) -> dict:
    return {
        "fever_id": fever_id,
        "gold_label": gold,
        "parsed_prediction": predicted,
        "correct": predicted == gold,
        "valid_format": True,
        "inference_error": None,
    }


class ConsistencyTests(unittest.TestCase):
    def test_five_of_five_is_exact_and_stable_correct(self):
        result = case_consistency([SUPPORTED] * 5, SUPPORTED)
        self.assertEqual(result["modal_agreement"], 1.0)
        self.assertTrue(result["exact_agreement"])
        self.assertEqual(result["consistency_classification"], "STABLE_CORRECT")

    def test_four_of_five(self):
        result = case_consistency([SUPPORTED] * 4 + [REFUTED], SUPPORTED)
        self.assertEqual(result["modal_agreement"], 0.8)
        self.assertFalse(result["exact_agreement"])
        self.assertFalse(result["tie"])

    def test_three_of_five(self):
        result = case_consistency([SUPPORTED] * 3 + [REFUTED, NEI], SUPPORTED)
        self.assertEqual(result["modal_agreement"], 0.6)
        self.assertEqual(result["modal_label"], SUPPORTED)

    def test_two_two_one_is_a_tie(self):
        result = case_consistency([SUPPORTED, SUPPORTED, REFUTED, REFUTED, NEI], SUPPORTED)
        self.assertEqual(result["modal_agreement"], 0.4)
        self.assertIsNone(result["modal_label"])
        self.assertEqual(result["consensus_prediction"], "TIE")
        self.assertTrue(result["tie"])

    def test_stable_wrong(self):
        result = case_consistency([REFUTED] * 5, SUPPORTED)
        self.assertEqual(result["consistency_classification"], "STABLE_WRONG")

    def test_mean_modal_agreement(self):
        labels = [
            [SUPPORTED] * 5,
            [SUPPORTED] * 4 + [REFUTED],
            [SUPPORTED] * 3 + [REFUTED, NEI],
        ]
        case_rows = [
            {
                **case_consistency(predictions, SUPPORTED),
                "gold_label": SUPPORTED,
            }
            for predictions in labels
        ]
        runs = [
            [prediction(index, SUPPORTED, labels[index][run]) for index in range(3)]
            for run in range(5)
        ]
        run_metrics = [
            {
                "overall_accuracy": sum(item["correct"] for item in run) / len(run),
                "number_correct": sum(item["correct"] for item in run),
                "total_cases": len(run),
            }
            for run in runs
        ]
        metrics = calculate_consistency_metrics(case_rows, runs, run_metrics)
        self.assertAlmostEqual(metrics["mean_modal_agreement"], 0.8)


if __name__ == "__main__":
    unittest.main()
