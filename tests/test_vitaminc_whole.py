"""The VitaminC sensitivity experiment on every valid pair, not just the 100 frozen ones.

The end-to-end tests read the committed ``test.jsonl`` and talk to a fake Ollama server, with
``limit`` keeping each run to a handful of pairs. A real whole-set run is 40,464 requests.
"""

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
    from tests.test_crash_safety import crash_after
except ModuleNotFoundError:  # run with tests/ itself on sys.path
    from fake_ollama import FakeOllama
    from test_crash_safety import crash_after

from llm_fact_checker import cli
from llm_fact_checker.common import read_jsonl
from llm_fact_checker.data_prep.vitaminc import (
    EXPECTED_VALID_PAIRS,
    identify_contrastive_pairs,
    load_vitaminc_test,
    validate_pairs,
)
from llm_fact_checker.experiments import sensitivity
from llm_fact_checker.experiments.sensitivity import run_vitaminc_sensitivity

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "vitaminc" / "raw" / "vitaminc" / "test.jsonl"
FROZEN = ROOT / "data" / "vitaminc" / "processed" / "vitaminc_sensitivity_100_pairs.jsonl"
MODEL = "fake-model:1b"
PAIRS = 5  # pairs per test run (two requests each)


def quiet():
    return contextlib.redirect_stderr(io.StringIO())


def answers(results: Path) -> list[tuple]:
    """What the model said for each row; latency is the one field that differs run to run."""
    rows = read_jsonl(results / "predictions" / "predictions.jsonl")
    return [
        (
            r["prediction_id"],
            r["pair_id"],
            r["side"],
            r["raw_model_output"],
            r["parsed_prediction"],
        )
        for r in rows
    ]


def pair(case_id: str, claim: str) -> dict:
    def make(label: str, canonical: str, evidence: str) -> dict:
        return {
            "claim": claim,
            "evidence": evidence,
            "original_label": label,
            "canonical_label": canonical,
            "source_metadata": {
                "case_id": case_id,
                "page": "Page",
                "revision_type": "real",
                "wiki_revision_id": "r1",
            },
        }

    return {
        "pair_id": f"{case_id}-s__{case_id}-r",
        "claim": claim,
        "support": make("SUPPORTS", "SUPPORTED", "Before."),
        "refute": make("REFUTES", "REFUTED", "After."),
    }


class RepeatedClaimValidationTests(unittest.TestCase):
    def test_the_sample_check_still_rejects_a_repeated_claim(self):
        with self.assertRaisesRegex(ValueError, "Duplicate claims"):
            validate_pairs([pair("1", "Same claim"), pair("2", "Same claim")])

    def test_the_whole_set_check_accepts_a_claim_in_two_cases(self):
        validate_pairs([pair("1", "Same claim"), pair("2", "Same claim")], unique_claims=False)

    def test_a_repeated_pair_id_is_rejected_either_way(self):
        with self.assertRaisesRegex(ValueError, "Duplicate VitaminC pair IDs"):
            validate_pairs([pair("1", "A"), pair("1", "B")], unique_claims=False)

    def test_every_other_check_still_applies_to_the_whole_set(self):
        broken = pair("1", "Claim")
        broken["refute"]["evidence"] = broken["support"]["evidence"]
        with self.assertRaisesRegex(ValueError, "identical evidence"):
            validate_pairs([broken], unique_claims=False)


@unittest.skipUnless(RAW.exists(), "committed VitaminC test.jsonl not present")
class WholeSetDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows, _ = load_vitaminc_test(RAW)
        cls.candidates = identify_contrastive_pairs(rows)

    def test_the_whole_set_is_every_valid_pair(self):
        self.assertEqual(len(self.candidates), EXPECTED_VALID_PAIRS)
        validate_pairs(self.candidates, EXPECTED_VALID_PAIRS, unique_claims=False)

    def test_claims_really_do_repeat_across_pairs(self):
        """The reason the whole set cannot go through the sample's duplicate-claim check."""
        claims = [item["claim"] for item in self.candidates]
        self.assertLess(len(set(claims)), len(claims))
        with self.assertRaises(ValueError):
            validate_pairs(self.candidates)

    def test_the_frozen_sample_is_inside_the_whole_set(self):
        ids = {item["pair_id"] for item in self.candidates}
        self.assertTrue({item["pair_id"] for item in read_jsonl(FROZEN)} <= ids)


@unittest.skipUnless(RAW.exists(), "committed VitaminC test.jsonl not present")
class WholeSetRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_whole(self, results: Path, server: FakeOllama, **kwargs):
        kwargs.setdefault("limit", PAIRS)
        with quiet():
            return run_vitaminc_sensitivity(
                RAW, FROZEN, results, base_url=server.url, model=MODEL, whole=True, **kwargs
            )

    def test_it_takes_pairs_from_the_whole_set_not_the_frozen_file(self):
        results = self.root / "run"
        with FakeOllama(installed=(MODEL,)) as server:
            # a path that does not exist: the whole set must not need the frozen sample
            with quiet():
                metrics = run_vitaminc_sensitivity(
                    RAW,
                    self.root / "absent.jsonl",
                    results,
                    limit=PAIRS,
                    base_url=server.url,
                    model=MODEL,
                    whole=True,
                )
            self.assertEqual(len(server.chat_requests), 2 * PAIRS)
        self.assertEqual(metrics["total_pairs"], PAIRS)
        config = json.loads((results / "smoke" / "run_config.json").read_text())
        self.assertEqual(config["pair_selection"], "whole")
        self.assertEqual(config["full_pair_count"], EXPECTED_VALID_PAIRS)
        self.assertEqual(config["sample_size_pairs"], PAIRS)

    def test_the_report_says_it_covers_every_pair(self):
        with FakeOllama(installed=(MODEL,)) as server:
            self.run_whole(self.root / "run", server)
        report = (self.root / "run" / "smoke" / "report.md").read_text(encoding="utf-8")
        self.assertIn("every valid contrastive pair", report)

    def test_the_frozen_sample_report_is_unchanged_by_the_new_line(self):
        with FakeOllama(installed=(MODEL,)) as server, quiet():
            run_vitaminc_sensitivity(
                RAW, FROZEN, self.root / "run", limit=2, base_url=server.url, model=MODEL
            )
        report = (self.root / "run" / "smoke" / "report.md").read_text(encoding="utf-8")
        self.assertNotIn("every valid contrastive pair", report)
        config = json.loads((self.root / "run" / "smoke" / "run_config.json").read_text())
        self.assertEqual(config["pair_selection"], "frozen sample")

    def test_an_interrupted_run_resumes_to_the_same_result(self):
        with FakeOllama(installed=(MODEL,)) as server:
            reference = self.run_whole(self.root / "reference", server)
        reference_rows = answers(self.root / "reference" / "smoke")

        results = self.root / "run"
        saved_before_crash = 4
        with FakeOllama(installed=(MODEL,)) as server, quiet(), crash_after(saved_before_crash):
            with self.assertRaises(KeyboardInterrupt):
                run_vitaminc_sensitivity(
                    RAW,
                    FROZEN,
                    results,
                    limit=PAIRS,
                    base_url=server.url,
                    model=MODEL,
                    whole=True,
                )
        smoke = results / "smoke"
        self.assertTrue((smoke / "predictions" / "predictions.partial.jsonl").exists())
        self.assertFalse((smoke / "sensitivity_metrics.json").exists())

        with FakeOllama(installed=(MODEL,)) as server:
            resumed = self.run_whole(results, server, resume=True)
            self.assertEqual(len(server.chat_requests), 2 * PAIRS - saved_before_crash)
        self.assertEqual(resumed, reference)
        self.assertEqual(answers(smoke), reference_rows)
        self.assertFalse((smoke / "predictions" / "predictions.partial.jsonl").exists())

    def test_resuming_a_finished_run_asks_the_model_nothing(self):
        results = self.root / "run"
        with FakeOllama(installed=(MODEL,)) as server:
            first = self.run_whole(results, server)
        with FakeOllama(installed=(MODEL,)) as server:
            again = self.run_whole(results, server, resume=True)
            self.assertEqual(server.chat_requests, [])
        self.assertEqual(again, first)

    def test_an_interrupted_run_is_not_silently_discarded(self):
        results = self.root / "run"
        with FakeOllama(installed=(MODEL,)) as server, quiet(), crash_after(2):
            with self.assertRaises(KeyboardInterrupt):
                run_vitaminc_sensitivity(
                    RAW,
                    FROZEN,
                    results,
                    limit=PAIRS,
                    base_url=server.url,
                    model=MODEL,
                    whole=True,
                )
        with FakeOllama(installed=(MODEL,)) as server:
            with self.assertRaisesRegex(FileExistsError, "--resume"):
                self.run_whole(results, server)
            self.assertEqual(server.chat_requests, [])

    def test_the_metrics_file_is_written_last(self):
        results = self.root / "run"
        real = sensitivity.write_text_atomic

        def die_on_metrics(path, text):
            if path.name == "sensitivity_metrics.json":
                raise KeyboardInterrupt  # killed just before the "finished" marker
            real(path, text)

        with FakeOllama(installed=(MODEL,)) as server:
            with mock.patch.object(sensitivity, "write_text_atomic", die_on_metrics):
                with self.assertRaises(KeyboardInterrupt):
                    self.run_whole(results, server)
        smoke = results / "smoke"
        self.assertFalse((smoke / "sensitivity_metrics.json").exists())
        self.assertTrue((smoke / "report.md").exists())  # everything else was already there

        with FakeOllama(installed=(MODEL,)) as server:
            metrics = self.run_whole(results, server, resume=True)
            self.assertEqual(server.chat_requests, [])  # answers kept; only scoring was redone
        self.assertEqual(metrics["total_pairs"], PAIRS)
        self.assertTrue((smoke / "sensitivity_metrics.json").exists())


class CommandLineTests(unittest.TestCase):
    def parse(self, *argv: str):
        return cli.build_parser().parse_args(["vitaminc-sensitivity", *argv])

    def test_all_and_resume_are_off_by_default(self):
        args = self.parse()
        self.assertFalse(args.all)
        self.assertFalse(args.resume)

    def run_and_capture_dir(self, *argv: str) -> tuple[Path, dict]:
        args = self.parse(*argv)
        with mock.patch.object(cli, "run_vitaminc_sensitivity", return_value={}) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                cli._vitaminc_sensitivity(args)
        return run.call_args.args[2], run.call_args.kwargs

    def test_the_frozen_sample_keeps_its_folder(self):
        directory, kwargs = self.run_and_capture_dir("--model", "qwen2.5:7b")
        self.assertEqual(directory, Path("results/vitaminc/qwen2.5_7b/sensitivity"))
        self.assertFalse(kwargs["whole"])

    def test_the_whole_set_goes_to_its_own_full_folder(self):
        directory, kwargs = self.run_and_capture_dir("--all", "--resume", "--model", "qwen2.5:7b")
        self.assertEqual(directory, Path("results/vitaminc/qwen2.5_7b/sensitivity_full"))
        self.assertTrue(kwargs["whole"])
        self.assertTrue(kwargs["resume"])


if __name__ == "__main__":
    unittest.main()
