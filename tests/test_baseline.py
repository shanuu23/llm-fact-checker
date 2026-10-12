from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from llm_fact_checker.evaluation import calculate_metrics
from llm_fact_checker.data_prep import map_label
from llm_fact_checker.data_prep.fever import (
    candidate_order,
    scan_wikipedia,
    select_evidence_set,
)
from llm_fact_checker.prompts import parse_prediction


def fever_record(fever_id: int, label: str, claim: str) -> dict:
    return {
        "id": fever_id,
        "label": label,
        "claim": claim,
        "evidence": [[[1, 2, f"Page_{fever_id}", 0]]],
    }


class BaselineTests(unittest.TestCase):
    def test_label_mapping(self):
        self.assertEqual(map_label("SUPPORTS", "FEVER"), "SUPPORTED")
        self.assertEqual(map_label("REFUTES", "FEVER"), "REFUTED")
        self.assertEqual(map_label("NOT ENOUGH INFO", "FEVER"), "NOT ENOUGH INFORMATION")
        with self.assertRaises(ValueError):
            map_label("UNKNOWN", "FEVER")

    def test_deterministic_sampling_order_and_claim_deduplication(self):
        records = [fever_record(index, "SUPPORTS", f"Claim {index}") for index in range(10)]
        records.append(fever_record(99, "SUPPORTS", "Claim 1"))
        first = candidate_order(records, "SUPPORTS", 42)
        second = candidate_order(reversed(records), "SUPPORTS", 42)
        self.assertEqual(
            [item["fever_id"] for item in first], [item["fever_id"] for item in second]
        )
        self.assertEqual(len(first), 10)

    def test_evidence_set_selection(self):
        evidence = [
            [[1, 1, "Page_B", 2], [1, 2, "Page_A", 1]],
            [[2, 3, "Page_C", 4]],
            [[3, None, None, None]],
        ]
        self.assertEqual(select_evidence_set(evidence), [{"page": "Page_C", "sentence_id": 4}])

    def test_wikipedia_sentence_parsing_from_tiny_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wiki-001.jsonl"
            article = {
                "id": "Page_A",
                "text": "First sentence. Second sentence.",
                "lines": "0\tFirst sentence.\tlink\ttarget\n1\tSecond sentence.\n2\t",
            }
            path.write_text(json.dumps(article) + "\n", encoding="utf-8")
            found, missing = scan_wikipedia([path], {("Page_A", 1)})
        self.assertEqual(found, {("Page_A", 1): "Second sentence."})
        self.assertEqual(missing, set())

    def test_strict_model_output_parsing(self):
        self.assertEqual(parse_prediction(" supported\n"), "SUPPORTED")
        self.assertEqual(parse_prediction("REFUTED"), "REFUTED")
        self.assertEqual(parse_prediction("not enough information"), "NOT ENOUGH INFORMATION")
        self.assertIsNone(parse_prediction("The answer is SUPPORTED because..."))
        self.assertIsNone(parse_prediction("SUPPORTED."))

    def test_metrics(self):
        predictions = [
            {
                "gold_label": "SUPPORTED",
                "parsed_prediction": "SUPPORTED",
                "valid_format": True,
                "inference_error": None,
            },
            {
                "gold_label": "REFUTED",
                "parsed_prediction": "NOT ENOUGH INFORMATION",
                "valid_format": True,
                "inference_error": None,
            },
            {
                "gold_label": "REFUTED",
                "parsed_prediction": None,
                "valid_format": False,
                "inference_error": None,
            },
            {
                "gold_label": "SUPPORTED",
                "parsed_prediction": None,
                "valid_format": False,
                "inference_error": "timeout",
            },
        ]
        metrics = calculate_metrics(predictions)
        self.assertEqual(metrics["number_correct"], 1)
        self.assertEqual(metrics["overall_accuracy"], 0.25)
        self.assertEqual(metrics["prediction_counts"]["INVALID"], 2)
        self.assertEqual(metrics["invalid_format_count"], 1)
        self.assertEqual(metrics["inference_failure_count"], 1)
        self.assertEqual(metrics["confusion_matrix"]["REFUTED"]["NOT ENOUGH INFORMATION"], 1)


if __name__ == "__main__":
    unittest.main()
