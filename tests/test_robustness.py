from __future__ import annotations

import unittest

from llm_fact_checker.experiments.robustness import (
    _distractor_pool,
    calculate_pair_metrics,
    choose_distractor,
    nei_transition_counts,
    outcome_category,
    reverse_evidence_case,
    transition_counts,
)

SUPPORTED = "SUPPORTED"
REFUTED = "REFUTED"
NEI = "NOT ENOUGH INFORMATION"


def case(fever_id: int, claim: str, page: str, text: str, second: str | None = None) -> dict:
    evidence = [{"page": page, "sentence_id": 0, "text": text}]
    if second is not None:
        evidence.append({"page": f"{page}_2", "sentence_id": 1, "text": second})
    return {
        "fever_id": fever_id,
        "claim": claim,
        "gold_label": SUPPORTED,
        "evidence": evidence,
    }


class PerturbationTests(unittest.TestCase):
    def test_reversal_preserves_every_sentence_exactly(self):
        original = case(1, "Claim", "Page_A", "First.", "Second.")
        result = reverse_evidence_case(original)
        self.assertTrue(result["reorder_eligible"])
        self.assertEqual(result["evidence"], list(reversed(original["evidence"])))
        self.assertCountEqual(
            [tuple(item.values()) for item in result["evidence"]],
            [tuple(item.values()) for item in original["evidence"]],
        )

    def test_single_sentence_is_reorder_ineligible(self):
        original = case(1, "Claim", "Page_A", "Only sentence.")
        result = reverse_evidence_case(original)
        self.assertFalse(result["reorder_eligible"])
        self.assertEqual(result["evidence"], original["evidence"])

    def test_distractor_never_uses_same_fever_id(self):
        target = case(1, "Mars is red", "Mars", "Mars appears red.")
        other = case(2, "Ocean claim", "Ocean", "Ocean tides are influenced by the Moon.")
        selected = choose_distractor(target, _distractor_pool([target, other]))
        self.assertEqual(selected["distractor_source_fever_id"], 2)

    def test_distractor_skips_same_page_when_other_page_exists(self):
        target = case(1, "Target claim", "Shared_Page", "Target evidence.")
        same_page = case(2, "Other claim", "Shared_Page", "Unrelated wording here.")
        other_page = case(3, "Third claim", "Different_Page", "A distant sentence.")
        selected = choose_distractor(target, _distractor_pool([target, same_page, other_page]))
        self.assertEqual(selected["distractor_source_fever_id"], 3)
        self.assertEqual(selected["distractor_page"], "Different_Page")

    def test_distractor_selection_is_deterministic(self):
        target = case(3, "Target claim", "Target_Page", "Target evidence.")
        candidates = [
            case(2, "Second", "Page_B", "Another unrelated sentence."),
            case(1, "First", "Page_A", "A completely distant fact."),
            target,
        ]
        first = choose_distractor(target, _distractor_pool(candidates))
        second = choose_distractor(target, _distractor_pool(reversed(candidates)))
        self.assertEqual(first, second)
        self.assertEqual(first["distractor_source_fever_id"], 1)

    def test_obvious_lexical_overlap_is_rejected(self):
        target = case(3, "Roberto Orci wrote Star Trek", "Star_Trek", "Roberto Orci wrote it.")
        overlap = case(1, "Other", "Other_Page", "Roberto Orci produced a television series.")
        valid = case(2, "Other", "Distant_Page", "Meteora is an album by Linkin Park.")
        selected = choose_distractor(target, _distractor_pool([target, overlap, valid]))
        self.assertEqual(selected["distractor_source_fever_id"], 2)


class MetricTests(unittest.TestCase):
    def test_invariance_calculation(self):
        pairs = [
            (SUPPORTED, SUPPORTED, SUPPORTED),
            (REFUTED, SUPPORTED, REFUTED),
            (None, SUPPORTED, SUPPORTED),
        ]
        metrics = calculate_pair_metrics(pairs)
        self.assertEqual(metrics["complete_pair_count"], 2)
        self.assertEqual(metrics["invariant_count"], 1)
        self.assertEqual(metrics["invariance_rate"], 0.5)

    def test_stayed_correct(self):
        self.assertEqual(outcome_category(SUPPORTED, SUPPORTED, SUPPORTED), "STAYED_CORRECT")

    def test_stayed_wrong(self):
        self.assertEqual(outcome_category(REFUTED, NEI, SUPPORTED), "STAYED_WRONG")

    def test_harmful_flip(self):
        self.assertEqual(outcome_category(SUPPORTED, REFUTED, SUPPORTED), "HARMFUL_FLIP")

    def test_beneficial_flip(self):
        self.assertEqual(outcome_category(REFUTED, SUPPORTED, SUPPORTED), "BENEFICIAL_FLIP")

    def test_transition_counts(self):
        pairs = [
            (SUPPORTED, SUPPORTED, SUPPORTED),
            (SUPPORTED, REFUTED, SUPPORTED),
            (REFUTED, NEI, REFUTED),
        ]
        counts = transition_counts(pairs)
        self.assertEqual(counts["SUPPORTED -> SUPPORTED"], 1)
        self.assertEqual(counts["SUPPORTED -> REFUTED"], 1)
        self.assertEqual(counts["REFUTED -> NOT ENOUGH INFORMATION"], 1)

    def test_nei_transition_counts(self):
        pairs = [
            (SUPPORTED, NEI, SUPPORTED),
            (NEI, REFUTED, REFUTED),
            (NEI, NEI, SUPPORTED),
        ]
        counts = nei_transition_counts(pairs)
        self.assertEqual(counts["control_nei_count"], 2)
        self.assertEqual(counts["perturbed_nei_count"], 2)
        self.assertEqual(counts["non_nei_to_nei"], 1)
        self.assertEqual(counts["nei_to_non_nei"], 1)
        self.assertEqual(counts["nei_to_nei"], 1)


if __name__ == "__main__":
    unittest.main()
