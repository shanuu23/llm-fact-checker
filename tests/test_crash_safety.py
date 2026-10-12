"""A long unattended run must survive being killed at any moment, including mid-write."""

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
    from tests.fever_fixtures import write_jsonl as write_plain_jsonl
except ModuleNotFoundError:  # run with tests/ itself on sys.path
    from fake_ollama import FakeOllama
    from fever_fixtures import write_jsonl as write_plain_jsonl

from llm_fact_checker import evaluation, ollama
from llm_fact_checker.common import read_jsonl, write_jsonl, write_text_atomic
from llm_fact_checker.experiments import baseline
from llm_fact_checker.experiments.consistency import run_consistency_experiment

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "fever" / "processed" / "fever_baseline_150.jsonl"
MODEL = "fake-model:1b"
CASES = 6


def quiet():
    return contextlib.redirect_stderr(io.StringIO())


def crash_after(rows_before_crash: int):
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


class AtomicWriteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_failure_while_writing_rows_leaves_the_old_file_and_no_debris(self):
        target = self.dir / "predictions.jsonl"
        write_jsonl(target, [{"n": 1}, {"n": 2}])

        def rows():
            yield {"n": 3}
            raise KeyboardInterrupt  # killed halfway through writing

        with self.assertRaises(KeyboardInterrupt):
            write_jsonl(target, rows())
        self.assertEqual([r["n"] for r in read_jsonl(target)], [1, 2])  # untouched
        self.assertEqual([p.name for p in self.dir.iterdir()], ["predictions.jsonl"])

    def test_text_files_are_replaced_whole_or_not_at_all(self):
        target = self.dir / "metrics.json"
        write_text_atomic(target, "old")
        write_text_atomic(target, "new")
        self.assertEqual(target.read_text(), "new")
        self.assertEqual([p.name for p in self.dir.iterdir()], ["metrics.json"])


@unittest.skipUnless(SAMPLE.exists(), "committed sample not present")
class ResumeCrashSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.prepared = self.root / "prepared.jsonl"
        write_plain_jsonl(self.prepared, list(read_jsonl(SAMPLE))[:CASES])

    def tearDown(self):
        self.tmp.cleanup()

    def baseline(self, results: Path, server: FakeOllama, **kwargs):
        with quiet():
            return baseline.run_inference(
                self.prepared, results, base_url=server.url, model=MODEL, **kwargs
            )

    def consistency(self, results: Path, server: FakeOllama, **kwargs):
        with quiet():
            return run_consistency_experiment(
                self.prepared, results, base_url=server.url, model=MODEL, **kwargs
            )

    def test_a_cut_off_last_line_in_the_partial_file_is_just_asked_again(self):
        with FakeOllama(installed=(MODEL,)) as server:
            reference = self.baseline(self.root / "reference", server)
        ref_rows = list(read_jsonl(self.root / "reference" / "predictions.jsonl"))

        results = self.root / "run"
        results.mkdir()
        good = "".join(json.dumps(r) + "\n" for r in ref_rows[:3])
        cut_off = json.dumps(ref_rows[3])[:40]  # the crash happened mid-write
        (results / "predictions.partial.jsonl").write_text(good + cut_off, encoding="utf-8")

        with FakeOllama(installed=(MODEL,)) as server:
            resumed = self.baseline(results, server, resume=True)
            self.assertEqual(len(server.chat_requests), CASES - 3)  # the cut-off case is re-asked
        self.assertEqual(resumed, reference)
        self.assertEqual(len(list(read_jsonl(results / "predictions.jsonl"))), CASES)

    def test_corruption_in_the_middle_of_a_partial_file_is_reported_not_hidden(self):
        results = self.root / "run"
        results.mkdir()
        (results / "predictions.partial.jsonl").write_text(
            '{"fever_id": 1}\nnot json\n{"fever_id": 2}\n', encoding="utf-8"
        )
        with FakeOllama(installed=(MODEL,)) as server:
            with self.assertRaisesRegex(ValueError, "line 2"):
                self.baseline(results, server, resume=True)

    def test_a_truncated_predictions_file_is_not_mistaken_for_a_finished_run(self):
        results = self.root / "run"
        results.mkdir()
        rows = list(read_jsonl(SAMPLE))[:CASES]
        write_plain_jsonl(results / "predictions.jsonl", rows[:2])  # only 2 of 6
        with FakeOllama(installed=(MODEL,)) as server:
            self.baseline(results, server, resume=True)
            self.assertEqual(len(server.chat_requests), CASES)
        self.assertEqual(len(list(read_jsonl(results / "predictions.jsonl"))), CASES)

    def test_a_truncated_consistency_run_file_is_redone_not_trusted(self):
        """The failure seen in the overnight rehearsal: a kill while run_01 was being written."""
        with FakeOllama(installed=(MODEL,)) as server:
            reference = self.consistency(self.root / "reference", server)

        results = self.root / "run"
        with FakeOllama(installed=(MODEL,)) as server, quiet(), crash_after(CASES + 2):
            with self.assertRaises(KeyboardInterrupt):
                run_consistency_experiment(
                    self.prepared, results, base_url=server.url, model=MODEL
                )
        run_one = results / "runs" / "run_01_predictions.jsonl"
        lines = run_one.read_text().splitlines()
        run_one.write_text("\n".join(lines[: CASES // 2]) + "\n")  # simulate the half-written file

        with FakeOllama(installed=(MODEL,)) as server:
            resumed = self.consistency(results, server, resume=True)
            # run 1 again (6) + the rest of run 2 (4 of 6 left) + runs 3, 4, 5 (18)
            self.assertEqual(len(server.chat_requests), CASES + (CASES - 2) + 3 * CASES)
        self.assertEqual(resumed, reference)

    def test_the_metrics_file_is_written_last_so_a_crash_before_it_is_not_a_finished_run(self):
        results = self.root / "run"
        real = evaluation.write_text_atomic

        def die_on_metrics(path, text):
            if path.name == "metrics.json":
                raise KeyboardInterrupt  # killed just before the "finished" marker
            real(path, text)

        with FakeOllama(installed=(MODEL,)) as server, quiet():
            with mock.patch.object(evaluation, "write_text_atomic", die_on_metrics):
                with self.assertRaises(KeyboardInterrupt):
                    baseline.run_inference(
                        self.prepared, results, base_url=server.url, model=MODEL
                    )
        self.assertFalse((results / "metrics.json").exists())
        self.assertTrue((results / "report.md").exists())  # everything else was already there

        with FakeOllama(installed=(MODEL,)) as server:
            metrics = self.baseline(results, server, resume=True)
            self.assertEqual(server.chat_requests, [])  # answers were kept; only scoring was redone
        self.assertEqual(metrics["total_cases"], CASES)
        self.assertTrue((results / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
