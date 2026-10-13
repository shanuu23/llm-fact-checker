from __future__ import annotations

import unittest

from llm_fact_checker.data_prep.vitaminc_train import (
    select_full_eval_pairs,
    strict_transition_pairs,
)


def row(unique_id: str, label: str, evidence: str, **changes: str) -> dict:
    item = {
        "unique_id": unique_id,
        "label": label,
        "evidence": evidence,
        "page": "Example",
        "revision_type": "real",
        "source_anchor": ("wiki_revision_id", "rev-1"),
        "audit_flags": [],
    }
    item.update(changes)
    return item


class VitaminCTrainTests(unittest.TestCase):
    def test_three_transitions_and_rejects_non_pairs(self) -> None:
        groups = {
            ("s-r", "Claim A"): [
                row("a-s", "SUPPORTS", "before"),
                row("a-r", "REFUTES", "after"),
            ],
            ("s-n", "Claim B"): [
                row("b-n", "NOT ENOUGH INFO", "before"),
                row("b-s", "SUPPORTS", "after"),
            ],
            ("r-n", "Claim C"): [
                row("c-r", "REFUTES", "before", audit_flags=["page_title_mismatch"]),
                row("c-n", "NOT ENOUGH INFO", "after"),
            ],
            ("same-evidence", "Claim D"): [
                row("d-s", "SUPPORTS", "same"),
                row("d-r", "REFUTES", "same"),
            ],
            ("different-page", "Claim E"): [
                row("e-s", "SUPPORTS", "before"),
                row("e-r", "REFUTES", "after", page="Other"),
            ],
        }
        pairs = list(strict_transition_pairs(groups))
        self.assertEqual(
            [pair["pair_type"] for pair in pairs],
            ["REFUTES_NEI", "SUPPORTS_NEI", "SUPPORTS_REFUTES"],
        )
        self.assertEqual(pairs[0]["audit_flags"], ["page_title_mismatch"])
        self.assertEqual(pairs[1]["left_gold_label"], "SUPPORTED")
        self.assertEqual(pairs[1]["right_gold_label"], "NOT ENOUGH INFORMATION")

    def test_full_selection_is_deterministic_and_excludes_pilot_claims(self) -> None:
        pair_types = ("SUPPORTS_REFUTES", "SUPPORTS_NEI", "REFUTES_NEI")
        pairs = [
            {
                "pair_id": f"{pair_type}-{number}-b",
                "pair_type": pair_type,
                "claim": f"Claim {pair_type} {number}",
                "left_unique_id": f"{pair_type}-{number}-left",
                "right_unique_id": f"{pair_type}-{number}-right",
            }
            for pair_type in pair_types
            for number in range(4)
        ]
        for pair in pairs:
            pair.setdefault("audit_flags", [])
        pairs.append({**pairs[0], "pair_id": "SUPPORTS_REFUTES-0-a"})
        counts = {"SUPPORTS_REFUTES": 2, "SUPPORTS_NEI": 2, "REFUTES_NEI": None}
        first, available = select_full_eval_pairs(pairs, {"Claim SUPPORTS_NEI 0"}, counts)
        second, _ = select_full_eval_pairs(reversed(pairs), {"Claim SUPPORTS_NEI 0"}, counts)
        self.assertEqual(first, second)
        self.assertEqual(available, {"SUPPORTS_REFUTES": 4, "SUPPORTS_NEI": 3, "REFUTES_NEI": 4})
        self.assertEqual(len(first), 8)
        self.assertFalse(any(pair["claim"] == "Claim SUPPORTS_NEI 0" for pair in first))
        self.assertNotIn("SUPPORTS_REFUTES-0-b", [pair["pair_id"] for pair in first])
        self.assertEqual(sum(pair["pair_type"] == "REFUTES_NEI" for pair in first), 4)

        with self.assertRaisesRegex(ValueError, "Not enough eligible"):
            select_full_eval_pairs(
                pairs, set(), {"SUPPORTS_REFUTES": 2, "SUPPORTS_NEI": 5, "REFUTES_NEI": None}
            )


if __name__ == "__main__":
    unittest.main()
