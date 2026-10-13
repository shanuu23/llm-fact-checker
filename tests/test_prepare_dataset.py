"""prepare_dataset builds the frozen FEVER sample; these tests use a tiny synthetic dump."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

try:
    from tests.fever_fixtures import fever_record, wiki_page, write_jsonl
except ModuleNotFoundError:  # run with tests/ itself on sys.path
    from fever_fixtures import fever_record, wiki_page, write_jsonl

from llm_fact_checker.data_prep.fever import candidate_order, prepare_dataset


class PrepareDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dev = self.root / "dev.jsonl"
        self.wiki = self.root / "wiki"
        self.wiki.mkdir()
        self.out = self.root / "sample.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, records, pages, **kwargs):
        write_jsonl(self.dev, records)
        write_jsonl(self.wiki / "wiki-001.jsonl", pages)
        with contextlib.redirect_stderr(io.StringIO()):
            return prepare_dataset(self.dev, self.wiki, self.out, **kwargs)

    @staticmethod
    def simple_world(per_class: int = 5):
        """per_class SUPPORTS and per_class REFUTES claims, each with one resolvable sentence."""
        records, pages = [], []
        for i in range(per_class):
            records.append(fever_record(100 + i, "SUPPORTS", f"Supported claim {i}.", [[(f"S{i}", 0)]]))
            records.append(fever_record(200 + i, "REFUTES", f"Refuted claim {i}.", [[(f"R{i}", 1)]]))
            pages.append(wiki_page(f"S{i}", {0: f"Supported evidence {i}."}))
            pages.append(wiki_page(f"R{i}", {0: "Intro.", 1: f"Refuted evidence {i}."}))
        return records, pages

    def test_builds_a_balanced_sample_with_the_evidence_text_attached(self):
        records, pages = self.simple_world()
        sample = self.build(records, pages, per_label=3, seed=42)
        self.assertEqual(len(sample), 6)
        self.assertEqual([r["gold_label"] for r in sample].count("SUPPORTED"), 3)
        self.assertEqual([r["gold_label"] for r in sample].count("REFUTED"), 3)
        self.assertEqual([r["fever_id"] for r in sample], sorted(r["fever_id"] for r in sample))
        for row in sample:
            (evidence,) = row["evidence"]
            kind = "Supported" if row["gold_label"] == "SUPPORTED" else "Refuted"
            self.assertTrue(evidence["text"].startswith(f"{kind} evidence"))
            self.assertEqual(row["wikipedia_page_ids"], [evidence["page"]])
            self.assertEqual(row["sentence_ids"], [evidence["sentence_id"]])
        self.assertEqual(len(self.out.read_text().splitlines()), 6)

    def test_the_same_inputs_give_byte_identical_output(self):
        records, pages = self.simple_world()
        self.build(records, pages, per_label=3, seed=42)
        first = self.out.read_bytes()
        self.out.unlink()
        self.build(records, pages, per_label=3, seed=42)
        self.assertEqual(self.out.read_bytes(), first)

    def test_not_enough_info_duplicates_and_unusable_evidence_are_never_selected(self):
        records = [
            fever_record(1, "SUPPORTS", "Alpha is real.", [[("P1", 0)]]),
            fever_record(2, "SUPPORTS", "  alpha   IS real. ", [[("P1", 0)]]),  # duplicate text
            fever_record(3, "SUPPORTS", "Beta is real.", [[("P2", 0)]]),
            fever_record(4, "SUPPORTS", "Has no usable evidence."),  # null page
            fever_record(5, "NOT ENOUGH INFO", "Unverifiable claim."),
            fever_record(6, "REFUTES", "Gamma is false.", [[("P3", 0)]]),
            fever_record(7, "REFUTES", "Delta is false.", [[("P4", 0)]]),
        ]
        pages = [wiki_page(p, {0: f"Text of {p}."}) for p in ("P1", "P2", "P3", "P4")]
        sample = self.build(records, pages, per_label=2, seed=1)
        ids = {r["fever_id"] for r in sample}
        self.assertEqual(ids & {4, 5}, set())  # unusable evidence and NEI never appear
        self.assertEqual(len(ids & {1, 2}), 1)  # exactly one of the duplicate claims survives
        self.assertEqual(len(sample), 4)

    def test_an_unresolvable_candidate_is_replaced_by_the_next_in_order(self):
        records, pages = self.simple_world(per_class=5)
        order = [c["fever_id"] for c in candidate_order(records, "SUPPORTS", seed=42)]
        broken = order[0]  # first choice for SUPPORTS: remove its page from the dump
        pages = [p for p in pages if p["id"] != f"S{broken - 100}"]
        sample = self.build(records, pages, per_label=2, seed=42)
        supported = {r["fever_id"] for r in sample if r["gold_label"] == "SUPPORTED"}
        self.assertNotIn(broken, supported)
        # first slot dropped, so the survivor of the first two plus the next candidate
        self.assertEqual(supported, {order[1], order[2]})

    def test_a_sentence_missing_from_its_page_counts_as_unresolvable(self):
        records = [
            fever_record(1, "SUPPORTS", "Needs a sentence that is absent.", [[("P1", 9)]]),
            fever_record(2, "SUPPORTS", "Fine claim.", [[("P2", 0)]]),
            fever_record(3, "REFUTES", "Refuted claim.", [[("P3", 0)]]),
        ]
        pages = [wiki_page(p, {0: f"Text of {p}."}) for p in ("P1", "P2", "P3")]
        sample = self.build(records, pages, per_label=1, seed=0)
        self.assertEqual({r["fever_id"] for r in sample}, {2, 3})

    def test_fails_clearly_when_there_are_not_enough_resolvable_candidates(self):
        records, pages = self.simple_world(per_class=2)
        pages = [p for p in pages if p["id"] != "S0"]  # only one resolvable SUPPORTS left
        with self.assertRaisesRegex(RuntimeError, "Not enough resolvable SUPPORTS"):
            self.build(records, pages, per_label=2, seed=42)

    def test_multi_sentence_evidence_keeps_every_sentence(self):
        records = [
            fever_record(1, "SUPPORTS", "Needs two pages.", [[("PageB", 2), ("PageA", 0)]]),
            fever_record(2, "REFUTES", "Single sentence.", [[("PageC", 0)]]),
        ]
        pages = [
            wiki_page("PageA", {0: "From A."}),
            wiki_page("PageB", {2: "From B."}),
            wiki_page("PageC", {0: "From C."}),
        ]
        sample = self.build(records, pages, per_label=1, seed=0)
        row = next(r for r in sample if r["fever_id"] == 1)
        self.assertEqual(
            [(e["page"], e["sentence_id"], e["text"]) for e in row["evidence"]],
            [("PageA", 0, "From A."), ("PageB", 2, "From B.")],  # stable, sorted by reference
        )
        self.assertEqual(row["wikipedia_page_ids"], ["PageA", "PageB"])

    def test_smallest_complete_evidence_set_is_preferred(self):
        records = [
            fever_record(1, "SUPPORTS", "Has two evidence sets.", [[("P1", 0), ("P2", 0)], [("P3", 0)]]),
            fever_record(2, "REFUTES", "Refuted.", [[("P4", 0)]]),
        ]
        pages = [wiki_page(p, {0: f"Text {p}."}) for p in ("P1", "P2", "P3", "P4")]
        sample = self.build(records, pages, per_label=1, seed=0)
        row = next(r for r in sample if r["fever_id"] == 1)
        self.assertEqual([e["page"] for e in row["evidence"]], ["P3"])

    def test_a_missing_dump_directory_is_reported(self):
        write_jsonl(self.dev, [fever_record(1, "SUPPORTS", "x", [[("P", 0)]])])
        with self.assertRaisesRegex(FileNotFoundError, "wiki-"):
            prepare_dataset(self.dev, self.wiki, self.out, per_label=1)


if __name__ == "__main__":
    unittest.main()
