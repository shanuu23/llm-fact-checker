"""Consistency and robustness on a dataset other than the frozen 150 cases, and resuming them."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    from tests.fake_ollama import FakeOllama
    from tests.fever_fixtures import write_jsonl
except ModuleNotFoundError:  # run with tests/ itself on sys.path
    from fake_ollama import FakeOllama
    from fever_fixtures import write_jsonl

from llm_fact_checker import ollama
from llm_fact_checker.common import file_sha256, read_jsonl
from llm_fact_checker.experiments.consistency import (
    EXPECTED_DATASET_SHA256,
    run_consistency_experiment,
    validate_prepared_dataset,
)
from llm_fact_checker.experiments.robustness import (
    prepare_robustness_files,
    run_robustness_experiment,
    validate_perturbations,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "fever" / "processed" / "fever_baseline_150.jsonl"
MODEL = "fake-model:1b"
CASES = 24


def quiet():
    return contextlib.redirect_stderr(io.StringIO())


def crash_after(rows_before_crash: int):
    """Patch inference so the process 'dies' (KeyboardInterrupt) after N saved answers."""
    real = ollama.infer_cases
    seen: list = []

    def crashing(*args, on_result=None, **kwargs):
        def wrapped(row):
            on_result(row)
            seen.append(row)
            if len(seen) >= rows_before_crash:
                raise KeyboardInterrupt

        return real(*args, on_result=wrapped, **kwargs)

    return mock.patch.object(ollama, "infer_cases", crashing)


@unittest.skipUnless(SAMPLE.exists(), "committed sample not present")
class WholeExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rows = [dict(row) for row in list(read_jsonl(SAMPLE))[:CASES]]
        self.flagged = {rows[1]["fever_id"], rows[4]["fever_id"]}
        for row in rows:  # the shape `prepare --all` writes
            row["flags"] = ["conflicting_duplicate_label"] if row["fever_id"] in self.flagged else []
        self.prepared = self.root / "whole.jsonl"
        write_jsonl(self.prepared, rows)
        self.perturbed = self.root / "perturbed"
        with quiet():
            prepare_robustness_files(self.prepared, self.perturbed)

    def tearDown(self):
        self.tmp.cleanup()

    # -- helpers -----------------------------------------------------------------------------
    def consistency(self, results: Path, server: FakeOllama, **kwargs):
        with quiet():
            return run_consistency_experiment(
                self.prepared, results, base_url=server.url, model=MODEL, **kwargs
            )

    def robustness(self, results: Path, server: FakeOllama, **kwargs):
        with quiet():
            return run_robustness_experiment(
                self.prepared,
                results,
                base_url=server.url,
                model=MODEL,
                prepared_dir=self.perturbed,
                **kwargs,
            )

    # -- the gate ----------------------------------------------------------------------------
    def test_any_well_formed_file_is_accepted_and_its_hash_is_returned(self):
        records, digest = validate_prepared_dataset(self.prepared)
        self.assertEqual(len(records), CASES)
        self.assertEqual(digest, file_sha256(self.prepared))
        self.assertNotEqual(digest, EXPECTED_DATASET_SHA256)

    def test_the_frozen_sample_must_still_match_its_hash_when_passed_in_place(self):
        edited = self.root / "frozen_copy.jsonl"
        edited.write_text(SAMPLE.read_text().replace("Roberto Orci", "Somebody Else", 1))
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_prepared_dataset(edited, frozen_path=edited)

    def test_the_untouched_frozen_sample_passes_its_strict_checks(self):
        records, digest = validate_prepared_dataset(SAMPLE)
        self.assertEqual((len(records), digest), (150, EXPECTED_DATASET_SHA256))

    def test_malformed_files_are_rejected(self):
        rows = list(read_jsonl(self.prepared))
        cases = {
            "empty": [],
            "duplicate id": [rows[0], rows[0]],
            "unscorable label": [{**rows[0], "gold_label": "NOT ENOUGH INFORMATION"}],
        }
        for name, content in cases.items():
            with self.subTest(case=name):
                path = self.root / f"bad_{name.replace(' ', '_')}.jsonl"
                write_jsonl(path, content)
                with self.assertRaises(ValueError):
                    validate_prepared_dataset(path)

    # -- consistency -------------------------------------------------------------------------
    def test_consistency_runs_on_a_dataset_that_is_not_the_frozen_sample(self):
        with FakeOllama(installed=(MODEL,)) as server:
            metrics = self.consistency(self.root / "c", server)
            self.assertEqual(len(server.chat_requests), CASES * 5)
        self.assertEqual(metrics["total_cases"], CASES)
        config = json.loads((self.root / "c" / "run_config.json").read_text())
        self.assertEqual(config["prepared_dataset_sha256"], file_sha256(self.prepared))
        report = (self.root / "c" / "report.md").read_text()
        self.assertIn("Dataset SHA-256", report)
        self.assertNotIn("unchanged", report)  # it is not the frozen 75/75 sample

    def test_consistency_reports_a_second_result_without_flagged_cases(self):
        with FakeOllama(installed=(MODEL,)) as server:
            self.consistency(self.root / "c", server)
        excluded = json.loads(
            (self.root / "c" / "consistency_metrics_excluding_flagged.json").read_text()
        )
        self.assertEqual(excluded["total_cases"], CASES - len(self.flagged))
        self.assertEqual(excluded["excluded_flagged_cases"], len(self.flagged))
        self.assertIn(
            "## Result excluding flagged records", (self.root / "c" / "report.md").read_text()
        )
        runs = list(read_jsonl(self.root / "c" / "runs" / "run_01_predictions.jsonl"))
        self.assertEqual({r["fever_id"] for r in runs if r["flags"]}, self.flagged)

    def test_an_interrupted_consistency_run_resumes_to_the_same_result(self):
        with FakeOllama(installed=(MODEL,)) as server:
            reference = self.consistency(self.root / "reference", server)

        results = self.root / "crashed"
        with FakeOllama(installed=(MODEL,)) as server, quiet(), crash_after(CASES * 2 + 7):
            with self.assertRaises(KeyboardInterrupt):
                run_consistency_experiment(
                    self.prepared, results, base_url=server.url, model=MODEL
                )
        runs = results / "runs"
        self.assertTrue((runs / "run_02_predictions.jsonl").exists())  # two runs finished
        self.assertFalse((runs / "run_03_predictions.jsonl").exists())
        self.assertEqual(
            len((runs / "run_03_predictions.partial.jsonl").read_text().splitlines()), 7
        )

        with FakeOllama(installed=(MODEL,)) as server:
            with self.assertRaisesRegex(FileExistsError, "--resume"):
                self.consistency(results, server)  # a plain rerun must not touch it
            resumed = self.consistency(results, server, resume=True)
            self.assertEqual(len(server.chat_requests), CASES * 5 - (CASES * 2 + 7))
        self.assertEqual(resumed, reference)
        self.assertFalse(list(runs.glob("*.partial.jsonl")))

    def test_resuming_a_finished_consistency_run_asks_the_model_nothing(self):
        with FakeOllama(installed=(MODEL,)) as server:
            first = self.consistency(self.root / "c", server)
        with FakeOllama(installed=(MODEL,)) as server:
            again = self.consistency(self.root / "c", server, resume=True)
            self.assertEqual(server.chat_requests, [])
        self.assertEqual(again, first)

    # -- robustness --------------------------------------------------------------------------
    def test_robustness_runs_on_a_dataset_that_is_not_the_frozen_sample(self):
        with FakeOllama(installed=(MODEL,)) as server:
            metrics = self.robustness(self.root / "r", server)
        self.assertEqual(metrics["case_count"], CASES)
        self.assertEqual(
            len(list(read_jsonl(self.root / "r" / "predictions" / "distractor_predictions.jsonl"))),
            CASES,
        )

    def test_robustness_reports_a_second_result_without_flagged_cases(self):
        with FakeOllama(installed=(MODEL,)) as server:
            self.robustness(self.root / "r", server)
        excluded = json.loads(
            (self.root / "r" / "robustness_metrics_excluding_flagged.json").read_text()
        )
        self.assertEqual(excluded["case_count"], CASES - len(self.flagged))
        self.assertEqual(excluded["excluded_flagged_cases"], len(self.flagged))
        self.assertIn(
            "## Result excluding flagged records", (self.root / "r" / "report.md").read_text()
        )

    def test_an_interrupted_robustness_run_resumes_to_the_same_result(self):
        with FakeOllama(installed=(MODEL,)) as server:
            reference = self.robustness(self.root / "reference", server)
            total_requests = len(server.chat_requests)

        results = self.root / "crashed"
        saved_before_crash = CASES + 5  # the whole control pass and 5 rows of the next one
        with FakeOllama(installed=(MODEL,)) as server, quiet(), crash_after(saved_before_crash):
            with self.assertRaises(KeyboardInterrupt):
                run_robustness_experiment(
                    self.prepared,
                    results,
                    base_url=server.url,
                    model=MODEL,
                    prepared_dir=self.perturbed,
                )
        self.assertTrue((results / "predictions" / "control_predictions.jsonl").exists())
        self.assertFalse((results / "robustness_metrics.json").exists())

        with FakeOllama(installed=(MODEL,)) as server:
            with self.assertRaisesRegex(FileExistsError, "--resume"):
                self.robustness(results, server)
            resumed = self.robustness(results, server, resume=True)
            self.assertEqual(len(server.chat_requests), total_requests - saved_before_crash)
        self.assertEqual(resumed, reference)
        self.assertFalse(list((results / "predictions").glob("*.partial.jsonl")))

    def test_resuming_a_finished_robustness_run_asks_the_model_nothing(self):
        with FakeOllama(installed=(MODEL,)) as server:
            first = self.robustness(self.root / "r", server)
        with FakeOllama(installed=(MODEL,)) as server:
            again = self.robustness(self.root / "r", server, resume=True)
            self.assertEqual(server.chat_requests, [])
        self.assertEqual(again, first)

    def test_mismatched_perturbation_files_are_rejected(self):
        source = list(read_jsonl(self.prepared))
        control = list(read_jsonl(self.perturbed / "control.jsonl"))
        reordered = list(read_jsonl(self.perturbed / "reordered_evidence.jsonl"))
        distractors = list(read_jsonl(self.perturbed / "distractor_evidence.jsonl"))
        validate_perturbations(source, control, reordered, distractors)  # consistent: fine
        with self.assertRaisesRegex(ValueError, "same cases"):
            validate_perturbations(source, control[:-1], reordered, distractors)


if __name__ == "__main__":
    unittest.main()
