from __future__ import annotations

import unittest

from llm_fact_checker.full_score import (
    paired_outcomes,
    rate,
    score_fever,
    score_vitaminc,
    typo_tolerant_label,
    vitamin_pairs,
)


def prediction(
    case_id: str, gold: str, parsed: str | None, *, flags: list[str] | None = None
) -> dict:
    return {
        "case_id": case_id,
        "gold_label": gold,
        "parsed_prediction": parsed,
        "correct": parsed == gold,
        "valid_format": parsed is not None,
        "inference_error": None,
        "audit_flags": flags or [],
    }


def pair(
    pair_id: str, kind: str, left: str, right: str, *, flags: list[str] | None = None
) -> list[dict]:
    labels = {
        "SUPPORTS_REFUTES": ("SUPPORTED", "REFUTED"),
        "SUPPORTS_NEI": ("SUPPORTED", "NOT ENOUGH INFORMATION"),
        "REFUTES_NEI": ("REFUTED", "NOT ENOUGH INFORMATION"),
    }[kind]
    rows = [
        prediction(f"{pair_id}:{side}", gold, parsed, flags=flags)
        for side, gold, parsed in (("left", labels[0], left), ("right", labels[1], right))
    ]
    for side, row in zip(("left", "right"), rows, strict=True):
        row.update(
            pair_id=pair_id,
            pair_type=kind,
            side=side,
            claim=pair_id,
            revision_type="real",
        )
    return rows


class FullScoreTests(unittest.TestCase):
    def test_posthoc_refuted_typos_leave_strict_scores_unchanged(self) -> None:
        for raw in ("REFUTATED", "REFUTTED", "REFUTUTED", "REFUTUED"):
            row = {**prediction("x", "REFUTED", None), "raw_model_output": f" {raw.lower()}\n"}
            self.assertEqual(typo_tolerant_label(row), "REFUTED")
        row["raw_model_output"] = "REFUTATED because of the evidence"
        self.assertIsNone(typo_tolerant_label(row))
        row["raw_model_output"] = "REFUTATED"
        row["inference_error"] = "timeout"
        self.assertIsNone(typo_tolerant_label(row))

        fever = [
            {
                **prediction("fever:1", "REFUTED", None),
                "fever_id": 1,
                "raw_model_output": "REFUTATED",
            }
        ]
        scored = score_fever(fever, {1})
        self.assertEqual(scored["accuracy"]["count"], 0)
        self.assertEqual(scored["typo_tolerant_accuracy"]["count"], 1)

        vitamin = pair("a", "SUPPORTS_REFUTES", "SUPPORTED", None)
        vitamin[1]["raw_model_output"] = "REFUTATED"
        vitamin += pair("b", "SUPPORTS_NEI", "SUPPORTED", None)
        vitamin[3]["raw_model_output"] = "REFUTATED"
        scored_pairs = score_vitaminc(vitamin)
        self.assertEqual(scored_pairs["SUPPORTS_REFUTES"]["both_sides_correct"]["count"], 0)
        self.assertEqual(
            scored_pairs["SUPPORTS_REFUTES"]["typo_tolerant_both_sides_correct"]["count"], 1
        )
        self.assertEqual(
            scored_pairs["SUPPORTS_NEI"]["typo_tolerant_both_sides_correct"]["count"], 0
        )

    def test_fever_primary_and_secondary_denominators(self) -> None:
        rows = [
            {**prediction("fever:1", "SUPPORTED", "SUPPORTED"), "fever_id": 1},
            {
                **prediction("fever:2", "REFUTED", None, flags=["conflict"]),
                "fever_id": 2,
            },
            {**prediction("fever:3", "REFUTED", "REFUTED"), "fever_id": 3},
        ]
        metrics = score_fever(rows, {3})
        self.assertEqual(metrics["accuracy"]["count"], 2)
        self.assertEqual(metrics["accuracy"]["total"], 3)
        self.assertEqual(metrics["without_flagged"]["rate"], 1)
        self.assertEqual(metrics["without_pilot_ids"]["rate"], 0.5)
        self.assertEqual(metrics["invalid_format_count"], 1)
        self.assertEqual(rate(0, 0)["rate"], None)
        self.assertEqual(rate(3, 3)["wilson_95"][1], 1)

    def test_vitaminc_pairs_and_comparison(self) -> None:
        rows = [
            *pair("a", "SUPPORTS_REFUTES", "SUPPORTED", "REFUTED"),
            *pair("b", "SUPPORTS_NEI", "REFUTED", "NOT ENOUGH INFORMATION"),
            *pair("c", "REFUTES_NEI", "REFUTED", None, flags=["review"]),
        ]
        metrics = score_vitaminc(rows)
        self.assertEqual(metrics["SUPPORTS_REFUTES"]["both_sides_correct"]["count"], 1)
        self.assertEqual(metrics["SUPPORTS_NEI"]["both_sides_correct"]["count"], 0)
        self.assertEqual(metrics["REFUTES_NEI"]["invalid_format_sides"], 1)
        self.assertEqual(metrics["REFUTES_NEI"]["without_flagged"]["total"], 0)
        self.assertEqual(metrics["SUPPORTS_REFUTES"]["prediction_change_among_valid"]["count"], 1)
        self.assertEqual(len(vitamin_pairs(rows)), 3)
        self.assertEqual(
            paired_outcomes([True, True, False, False], [True, False, True, False]),
            {
                "both_correct": 1,
                "llama_only_correct": 1,
                "qwen_only_correct": 1,
                "both_wrong": 1,
            },
        )
        with self.assertRaisesRegex(ValueError, "odd number"):
            vitamin_pairs(rows[:-1])


if __name__ == "__main__":
    unittest.main()
