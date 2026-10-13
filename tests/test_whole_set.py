"""The whole-dataset run: audit flags, whole-set preparation, flagged results, and resume."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import re
import tempfile
import unicodedata
import unittest
from unittest import mock

try:
    from tests.fake_ollama import FakeOllama
    from tests.fever_fixtures import fever_record, wiki_page, write_jsonl
except ModuleNotFoundError:  # run with tests/ itself on sys.path
    from fake_ollama import FakeOllama
    from fever_fixtures import fever_record, wiki_page, write_jsonl

from llm_fact_checker import cli
from llm_fact_checker.common import file_sha256, read_jsonl
from llm_fact_checker.data_prep.fever import audit_flags, prepare_dataset
from llm_fact_checker.evaluation import MAX_LISTED_ERRORS, evaluate_results
from llm_fact_checker import ollama
from llm_fact_checker.experiments import baseline

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "fever" / "processed" / "fever_baseline_150.jsonl"
PILOT_CONFIG = ROOT / "results" / "fever" / "llama3.2_3b" / "baseline" / "run_config.json"
MODEL = "fake-model:1b"


def quiet():
    return contextlib.redirect_stderr(io.StringIO())


def claim_record(record_id: int, label: str, claim: str) -> dict:
    return {"id": record_id, "label": label, "claim": claim}


class AuditFlagTests(unittest.TestCase):
    def test_flags_follow_the_audit_rules(self):
        records = [
            claim_record(1, "SUPPORTS", "Alpha is real."),
            claim_record(2, "REFUTES", "Alpha is real."),  # same text, different label
            claim_record(3, "SUPPORTS", "Beta is real."),
            claim_record(4, "SUPPORTS", "Beta is real."),  # same text, same label: fine
            claim_record(5, "SUPPORTS", "Gamma is here."),
            claim_record(6, "NOT ENOUGH INFO", "Gamma is here."),  # conflict with NEI counts
            claim_record(7, "SUPPORTS", "Delta is in north Georgia."),
            claim_record(8, "NOT ENOUGH INFO", "Delta is in North Georgia."),  # only case differs
            claim_record(9, "SUPPORTS", "Eps has won an award."),
            claim_record(10, "SUPPORTS", "Eps has won an Award."),  # variant, same label: fine
            claim_record(11, "REFUTES", "A claim nobody repeats."),
        ]
        flags = audit_flags(records)
        self.assertEqual(
            flags,
            {
                1: ["conflicting_duplicate_label"],
                2: ["conflicting_duplicate_label"],
                5: ["conflicting_duplicate_label"],
                6: ["conflicting_duplicate_label"],
                7: ["normalized_claim_conflict"],
                8: ["normalized_claim_conflict"],
            },
        )

    def test_unicode_spelling_variants_with_different_labels_are_flagged(self):
        composed = "Café is a place."
        decomposed = unicodedata.normalize("NFD", composed)
        self.assertNotEqual(composed, decomposed)
        flags = audit_flags(
            [claim_record(1, "SUPPORTS", composed), claim_record(2, "REFUTES", decomposed)]
        )
        self.assertEqual(sorted(flags), [1, 2])


class WholeSetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dev, self.wiki, self.out = self.root / "dev.jsonl", self.root / "wiki", self.root / "o"
        self.wiki.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, records, pages, **kwargs):
        write_jsonl(self.dev, records)
        write_jsonl(self.wiki / "wiki-001.jsonl", pages)
        with quiet():
            return prepare_dataset(self.dev, self.wiki, self.out, **kwargs)

    def test_whole_mode_keeps_every_claim_and_marks_the_flagged_ones(self):
        records = [
            fever_record(1, "SUPPORTS", "Alpha is real.", [[("P1", 0)]]),
            fever_record(2, "REFUTES", "Alpha is real.", [[("P2", 0)]]),  # conflicting duplicate
            fever_record(3, "SUPPORTS", "Beta is real.", [[("P3", 0)]]),
            fever_record(4, "SUPPORTS", "Beta is real.", [[("P3", 0)]]),  # repeat, same label
            fever_record(5, "SUPPORTS", "Gamma.", [[("P4", 0)]]),
            fever_record(6, "NOT ENOUGH INFO", "Nothing known."),
        ]
        pages = [wiki_page(f"P{i}", {0: f"Text {i}."}) for i in range(1, 5)]
        rows = self.build(records, pages, whole=True)
        self.assertEqual([r["fever_id"] for r in rows], [1, 2, 3, 4, 5])  # NEI only is excluded
        self.assertEqual(
            {r["fever_id"]: r["flags"] for r in rows},
            {
                1: ["conflicting_duplicate_label"],
                2: ["conflicting_duplicate_label"],
                3: [],
                4: [],
                5: [],
            },
        )
        # the claim text of ids 3 and 4 is identical and both are kept
        self.assertEqual([r["claim"] for r in rows].count("Beta is real."), 2)

    def test_whole_mode_is_not_limited_to_a_balanced_sample(self):
        records = [
            fever_record(i, "SUPPORTS", f"Supported {i}.", [[(f"P{i}", 0)]]) for i in range(1, 6)
        ] + [fever_record(10, "REFUTES", "Refuted.", [[("P10", 0)]])]
        pages = [wiki_page(f"P{i}", {0: "Text."}) for i in (*range(1, 6), 10)]
        rows = self.build(records, pages, whole=True)
        self.assertEqual(len(rows), 6)

    def test_whole_mode_matches_an_accented_title_written_in_another_unicode_form(self):
        title = "Ricardo_Montalbán"
        stored = unicodedata.normalize("NFD", title)
        self.assertNotEqual(title, stored)
        records = [
            fever_record(1, "SUPPORTS", "Accent claim.", [[(title, 0)]]),
            fever_record(2, "REFUTES", "Plain claim.", [[("P2", 0)]]),
        ]
        pages = [wiki_page(stored, {0: "He was an actor."}), wiki_page("P2", {0: "Other."})]
        rows = self.build(records, pages, whole=True)
        accent = next(r for r in rows if r["fever_id"] == 1)
        self.assertEqual(accent["evidence"][0]["text"], "He was an actor.")
        self.assertEqual(accent["evidence"][0]["page"], title)  # title kept as requested

    def test_the_frozen_sample_mode_deliberately_does_not_match_accents(self):
        title = "Ricardo_Montalbán"
        records = [
            fever_record(1, "SUPPORTS", "Accent claim.", [[(title, 0)]]),
            fever_record(2, "SUPPORTS", "Plain claim.", [[("P2", 0)]]),
            fever_record(3, "REFUTES", "Refuted claim.", [[("P3", 0)]]),
        ]
        pages = [
            wiki_page(unicodedata.normalize("NFD", title), {0: "He was an actor."}),
            wiki_page("P2", {0: "Other."}),
            wiki_page("P3", {0: "Third."}),
        ]
        sample = self.build(records, pages, per_label=1, seed=0)
        self.assertNotIn(1, {r["fever_id"] for r in sample})  # skipped, as in the frozen sample
        self.assertTrue(all("flags" not in r for r in sample))

    def test_whole_mode_reports_and_drops_claims_whose_evidence_is_missing(self):
        records = [
            fever_record(1, "SUPPORTS", "Findable.", [[("P1", 0)]]),
            fever_record(2, "REFUTES", "Not in the dump.", [[("Gone", 0)]]),
        ]
        write_jsonl(self.dev, records)
        write_jsonl(self.wiki / "wiki-001.jsonl", [wiki_page("P1", {0: "Text."})])
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rows = prepare_dataset(self.dev, self.wiki, self.out, whole=True)
        self.assertEqual([r["fever_id"] for r in rows], [1])
        self.assertIn("Dropped 1 claims", err.getvalue())

    @unittest.skipUnless(SAMPLE.exists(), "committed sample not present")
    def test_the_committed_sample_has_no_flags_field(self):
        first = next(iter(read_jsonl(SAMPLE)))
        self.assertNotIn("flags", first)


WHOLE_SET = ROOT / "data" / "fever" / "full" / "fever_dev_full.jsonl"
HASHES = ROOT / "data" / "HASHES.txt"


@unittest.skipUnless(WHOLE_SET.exists(), "whole-dataset file not present")
class CommittedWholeSetTests(unittest.TestCase):
    """Pins the committed whole-set file to what the FEVER data audit established."""

    @classmethod
    def setUpClass(cls):
        cls.rows = list(read_jsonl(WHOLE_SET))

    def test_every_supports_and_refutes_claim_is_present_and_balanced(self):
        labels = {}
        for row in self.rows:
            labels[row["gold_label"]] = labels.get(row["gold_label"], 0) + 1
        self.assertEqual(labels, {"SUPPORTED": 6666, "REFUTED": 6666})
        self.assertEqual(len({row["fever_id"] for row in self.rows}), 13332)

    def test_every_case_carries_its_evidence_text(self):
        for row in self.rows:
            self.assertTrue(row["evidence"])
            self.assertTrue(all(item["text"].strip() for item in row["evidence"]))

    def test_the_audit_flags_match_the_notebook(self):
        flagged = [row for row in self.rows if row["flags"]]
        self.assertEqual(len(flagged), 68)
        self.assertEqual(
            {label: sum(r["gold_label"] == label for r in flagged) for label in ("SUPPORTED", "REFUTED")},
            {"SUPPORTED": 21, "REFUTED": 47},
        )
        self.assertEqual(sum(len(r["evidence"]) >= 2 for r in self.rows), 1204)

    def test_the_file_matches_the_recorded_hash(self):
        recorded = {
            line.split()[1]: line.split()[0]
            for line in HASHES.read_text().splitlines()
            if line.strip()
        }
        key = "data/fever/full/fever_dev_full.jsonl"
        self.assertIn(key, recorded)
        self.assertEqual(file_sha256(WHOLE_SET), recorded[key])


def make_prepared(path: Path, flagged_ids: set[int], count: int = 6) -> list[dict]:
    rows = []
    for row in list(read_jsonl(SAMPLE))[:count]:
        row = dict(row)
        row["flags"] = ["conflicting_duplicate_label"] if row["fever_id"] in flagged_ids else []
        rows.append(row)
    write_jsonl(path, rows)
    return rows


@unittest.skipUnless(SAMPLE.exists(), "committed sample not present")
class FlaggedResultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_baseline(self, prepared: Path, results: Path):
        with FakeOllama(installed=(MODEL,)) as server, quiet():
            return baseline.run_inference(
                prepared, results, base_url=server.url, model=MODEL
            ), server

    def test_flags_reach_the_predictions_and_a_second_result_is_written(self):
        prepared = self.root / "prepared.jsonl"
        rows = make_prepared(prepared, {r["fever_id"] for r in list(read_jsonl(SAMPLE))[:2]})
        results = self.root / "full"
        metrics, _ = self.run_baseline(prepared, results)
        self.assertEqual(metrics["total_cases"], 6)  # the main result keeps flagged cases
        excluded = json.loads((results / "metrics_excluding_flagged.json").read_text())
        self.assertEqual(excluded["total_cases"], 4)
        self.assertEqual(excluded["excluded_flagged_cases"], 2)
        self.assertEqual(excluded["flag_counts"], {"conflicting_duplicate_label": 2})
        predictions = list(read_jsonl(results / "predictions.jsonl"))
        self.assertEqual([bool(p["flags"]) for p in predictions], [bool(r["flags"]) for r in rows])
        report = (results / "report.md").read_text(encoding="utf-8")
        self.assertIn("## Result excluding flagged records", report)
        self.assertIn(f"`{MODEL}`", report)

    def test_without_flags_there_is_no_second_result(self):
        prepared = self.root / "prepared.jsonl"
        write_jsonl(prepared, list(read_jsonl(SAMPLE))[:3])
        results = self.root / "plain"
        self.run_baseline(prepared, results)
        self.assertFalse((results / "metrics_excluding_flagged.json").exists())
        self.assertNotIn("excluding flagged", (results / "report.md").read_text())

    def test_the_report_title_names_the_model_that_ran(self):
        prepared = self.root / "prepared.jsonl"
        write_jsonl(prepared, list(read_jsonl(SAMPLE))[:2])
        self.run_baseline(prepared, self.root / "r")
        first_line = (self.root / "r" / "report.md").read_text().splitlines()[0]
        self.assertEqual(first_line, f"# FEVER gold-evidence baseline: {MODEL}")

    @unittest.skipUnless(PILOT_CONFIG.exists(), "pilot run config not present")
    def test_a_huge_error_list_is_capped_in_the_report_but_complete_in_errors_csv(self):
        results = self.root / "many"
        results.mkdir()
        rows = [
            {
                "fever_id": i,
                "claim": f"Claim {i}",
                "evidence": [],
                "gold_label": "SUPPORTED",
                "raw_model_output": "REFUTED",
                "parsed_prediction": "REFUTED",
                "correct": False,
                "valid_format": True,
                "inference_error": None,
            }
            for i in range(MAX_LISTED_ERRORS + 25)
        ]
        write_jsonl(results / "predictions.jsonl", rows)
        (results / "run_config.json").write_text(PILOT_CONFIG.read_text(), encoding="utf-8")
        evaluate_results(results)
        report = (results / "report.md").read_text(encoding="utf-8")
        listed = [line for line in report.splitlines() if re.match(r"\| \d+ \| SUPPORTED \| REFUTED \|", line)]
        self.assertEqual(len(listed), MAX_LISTED_ERRORS)
        self.assertIn("25 more incorrect cases are not listed here", report)
        self.assertEqual(len((results / "errors.csv").read_text().splitlines()), len(rows) + 1)


@unittest.skipUnless(SAMPLE.exists(), "committed sample not present")
class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.prepared = self.root / "prepared.jsonl"
        write_jsonl(self.prepared, list(read_jsonl(SAMPLE))[:6])

    def tearDown(self):
        self.tmp.cleanup()

    def run_baseline(self, results: Path, server: FakeOllama, **kwargs):
        with quiet():
            return baseline.run_inference(
                self.prepared, results, base_url=server.url, model=MODEL, **kwargs
            )

    def reference_predictions(self) -> list[dict]:
        with FakeOllama(installed=(MODEL,)) as server:
            self.run_baseline(self.root / "reference", server)
        return list(read_jsonl(self.root / "reference" / "predictions.jsonl"))

    @staticmethod
    def answers(rows):
        return [(r["fever_id"], r["parsed_prediction"]) for r in rows]

    def interrupt_after(self, results: Path, rows_before_crash: int):
        """Run the baseline but stop the process (KeyboardInterrupt) after N saved answers."""
        real = ollama.infer_cases
        seen = []

        def crashing(*args, on_result=None, **kwargs):
            def wrapped(row):
                on_result(row)
                seen.append(row)
                if len(seen) >= rows_before_crash:
                    raise KeyboardInterrupt

            return real(*args, on_result=wrapped, **kwargs)

        with FakeOllama(installed=(MODEL,)) as server, quiet():
            with mock.patch.object(ollama, "infer_cases", crashing):
                with self.assertRaises(KeyboardInterrupt):
                    baseline.run_inference(
                        self.prepared, results, base_url=server.url, model=MODEL
                    )

    def test_a_crashed_run_keeps_what_it_had_and_resume_finishes_the_rest(self):
        results = self.root / "run"
        self.interrupt_after(results, rows_before_crash=3)
        self.assertFalse((results / "predictions.jsonl").exists())
        self.assertEqual(len((results / "predictions.partial.jsonl").read_text().splitlines()), 3)

        with FakeOllama(installed=(MODEL,)) as server:
            self.run_baseline(results, server, resume=True)
            self.assertEqual(len(server.chat_requests), 3)  # only the missing cases were asked
        final = list(read_jsonl(results / "predictions.jsonl"))
        self.assertEqual(self.answers(final), self.answers(self.reference_predictions()))
        self.assertFalse((results / "predictions.partial.jsonl").exists())

    def test_a_plain_rerun_refuses_to_touch_an_interrupted_run(self):
        results = self.root / "run"
        self.interrupt_after(results, rows_before_crash=2)
        with FakeOllama(installed=(MODEL,)) as server:
            with self.assertRaisesRegex(FileExistsError, "--resume"):
                self.run_baseline(results, server)
            self.assertEqual(server.chat_requests, [])

    def test_overwrite_discards_the_partial_run_and_starts_over(self):
        results = self.root / "run"
        self.interrupt_after(results, rows_before_crash=2)
        with FakeOllama(installed=(MODEL,)) as server:
            self.run_baseline(results, server, overwrite=True)
            self.assertEqual(len(server.chat_requests), 6)
        self.assertEqual(len(list(read_jsonl(results / "predictions.jsonl"))), 6)

    def test_cases_that_failed_before_the_interruption_are_retried(self):
        results = self.root / "run"
        results.mkdir()
        reference = self.reference_predictions()
        failed = dict(reference[1], parsed_prediction=None, raw_model_output=None,
                      correct=False, valid_format=False, inference_error="Ollama request failed")
        partial = [reference[0], failed]
        (results / "predictions.partial.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in partial), encoding="utf-8"
        )
        with FakeOllama(installed=(MODEL,)) as server:
            self.run_baseline(results, server, resume=True)
            self.assertEqual(len(server.chat_requests), 5)  # case 2 is asked again, with 3..6
        final = list(read_jsonl(results / "predictions.jsonl"))
        self.assertEqual(self.answers(final), self.answers(reference))
        self.assertTrue(all(r["inference_error"] is None for r in final))

    def test_resume_on_a_fresh_folder_just_runs_everything(self):
        with FakeOllama(installed=(MODEL,)) as server:
            self.run_baseline(self.root / "fresh", server, resume=True)
            self.assertEqual(len(server.chat_requests), 6)


class CommandLineTests(unittest.TestCase):
    def test_prepare_all_defaults_to_the_derived_full_file_not_the_frozen_sample(self):
        args = cli.build_parser().parse_args(["prepare", "--all"])
        self.assertTrue(args.all)
        self.assertIsNone(args.output)  # resolved to FULL_PATH by the handler
        plain = cli.build_parser().parse_args(["prepare"])
        self.assertFalse(plain.all)

    def test_run_accepts_resume(self):
        args = cli.build_parser().parse_args(["run", "--resume"])
        self.assertTrue(args.resume)
        self.assertFalse(cli.build_parser().parse_args(["run"]).resume)


if __name__ == "__main__":
    unittest.main()
