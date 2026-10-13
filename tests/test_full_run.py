from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from llm_fact_checker.full_run import fever_cases, run, vitaminc_cases


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class FakeClient:
    base_url = "http://localhost:11434"

    def __init__(self, answers: list[str | BaseException], digest: str = "fixed-digest"):
        self.answers = iter(answers)
        self.calls = 0
        self.digest = digest

    def check(self, model: str) -> dict:
        return {"version": "test", "model": {"digest": self.digest}}

    def chat(self, model: str, claim: str, evidence: list[dict], **settings: object) -> tuple:
        self.calls += 1
        answer = next(self.answers)
        if isinstance(answer, BaseException):
            raise answer
        return answer, 0.1, {}


class FullRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_fever_resume_skips_saved_answers(self) -> None:
        prepared = self.root / "fever.jsonl"
        write_jsonl(
            prepared,
            [
                {
                    "fever_id": number,
                    "claim": f"Claim {number}",
                    "gold_label": label,
                    "evidence": [{"page": "Page", "sentence_id": 0, "text": "Evidence"}],
                    "audit_flags": [],
                }
                for number, label in ((1, "SUPPORTED"), (2, "REFUTED"))
            ],
        )
        cases = fever_cases(prepared)
        results_dir = self.root / "results"
        first = FakeClient(["SUPPORTED", KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            run(cases, results_dir, "example:1", {"prepared": "hash"}, first)
        self.assertEqual(first.calls, 2)
        self.assertEqual(len((results_dir / "predictions.jsonl").read_text().splitlines()), 1)

        second = FakeClient(["REFUTED"])
        summary = run(cases, results_dir, "example:1", {"prepared": "hash"}, second)
        self.assertEqual(second.calls, 1)
        self.assertEqual(summary["completed"], 2)
        self.assertEqual(summary["correct"], 2)

        with self.assertRaisesRegex(ValueError, "configuration or inputs changed"):
            run(cases, results_dir, "other:1", {"prepared": "hash"}, FakeClient([]))

    def test_changed_model_digest_refuses_partial_resume(self) -> None:
        cases = [
            {
                "case_id": "fever:1",
                "claim": "A claim",
                "evidence": [{"page": "P", "sentence_id": 0, "text": "evidence"}],
                "gold_label": "SUPPORTED",
                "audit_flags": [],
            },
            {
                "case_id": "fever:2",
                "claim": "Another claim",
                "evidence": [{"page": "P", "sentence_id": 1, "text": "evidence"}],
                "gold_label": "REFUTED",
                "audit_flags": [],
            },
        ]
        results = self.root / "results"
        with self.assertRaises(KeyboardInterrupt):
            run(cases, results, "example:1", {}, FakeClient(["SUPPORTED", KeyboardInterrupt()]))
        changed = FakeClient(["REFUTED"], digest="different")
        with self.assertRaisesRegex(ValueError, "model digest changed"):
            run(cases, results, "example:1", {}, changed)
        self.assertEqual(changed.calls, 0)

    def test_checkpoint_failure_is_counted_and_not_repeated(self) -> None:
        cases = [
            {
                "case_id": "fever:1",
                "claim": "A claim",
                "evidence": [{"page": "P", "sentence_id": 0, "text": "evidence"}],
                "gold_label": "SUPPORTED",
                "audit_flags": [],
            }
        ]
        client = FakeClient([RuntimeError("timeout")])
        summary = run(cases, self.root / "results", "example:1", {}, client)
        self.assertEqual(summary["inference_failures"], 1)
        self.assertEqual(summary["invalid_format"], 0)
        self.assertEqual(summary["correct"], 0)
        resumed = FakeClient([])
        run(cases, self.root / "results", "example:1", {}, resumed)
        self.assertEqual(resumed.calls, 0)

    def test_invalid_label_is_saved_without_retry(self) -> None:
        cases = [
            {
                "case_id": "fever:1",
                "claim": "A claim",
                "evidence": [{"page": "P", "sentence_id": 0, "text": "evidence"}],
                "gold_label": "SUPPORTED",
                "audit_flags": [],
            }
        ]
        client = FakeClient(["SUPPORTED because I think so"])
        summary = run(cases, self.root / "results", "example:1", {}, client)
        self.assertEqual(client.calls, 1)
        self.assertEqual(summary["invalid_format"], 1)
        self.assertEqual(summary["inference_failures"], 0)
        self.assertEqual(summary["correct"], 0)

    def test_vitaminc_selection_expands_both_sides(self) -> None:
        selected = self.root / "selected.jsonl"
        pairs = self.root / "pairs.jsonl"
        rows = self.root / "rows.jsonl"
        write_jsonl(selected, [{"pair_id": "a__b"}])
        write_jsonl(
            pairs,
            [
                {
                    "pair_id": "a__b",
                    "case_id": "one",
                    "claim": "A claim",
                    "page": "P",
                    "pair_type": "SUPPORTS_NEI",
                    "revision_type": "real",
                    "left_unique_id": "a",
                    "right_unique_id": "b",
                    "left_gold_label": "SUPPORTED",
                    "right_gold_label": "NOT ENOUGH INFORMATION",
                    "audit_flags": ["review"],
                }
            ],
        )
        write_jsonl(
            rows,
            [
                {
                    "unique_id": unique_id,
                    "case_id": "one",
                    "claim": "A claim",
                    "page": "P",
                    "evidence": evidence,
                    "gold_label": label,
                    "revision_type": "real",
                    "audit_flags": [],
                }
                for unique_id, evidence, label in (
                    ("a", "supports", "SUPPORTED"),
                    ("b", "insufficient", "NOT ENOUGH INFORMATION"),
                )
            ],
        )
        cases = vitaminc_cases(rows, pairs, selected)
        self.assertEqual([case["side"] for case in cases], ["left", "right"])
        self.assertEqual(cases[1]["evidence"][0]["text"], "insufficient")
        self.assertEqual(cases[1]["gold_label"], "NOT ENOUGH INFORMATION")
        self.assertEqual(cases[0]["audit_flags"], ["review"])
        self.assertEqual(cases[1]["audit_flags"], ["review"])
        self.assertEqual(cases[0]["pair_id"], cases[1]["pair_id"])


if __name__ == "__main__":
    unittest.main()
