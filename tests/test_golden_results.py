"""Golden regression tests: recompute every experiment's metrics and report from the
committed predictions and require an exact match with the committed result files.

These pin the pilot (Llama 3.2 3B) numbers, so a refactor or a change to metric code
cannot silently alter published results. They never call Ollama.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from llm_fact_checker.experiments import consistency, robustness, sensitivity
from llm_fact_checker.evaluation import calculate_metrics, evaluate_results
from llm_fact_checker.prompts import SYSTEM_PROMPT, build_user_prompt
from llm_fact_checker.common import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
FEVER = ROOT / "results" / "fever" / "llama3.2_3b"
VITAMINC = ROOT / "results" / "vitaminc" / "llama3.2_3b" / "sensitivity"
PAIRS = ROOT / "data" / "vitaminc" / "processed" / "vitaminc_sensitivity_100_pairs.jsonl"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalise(value):
    """Round-trip through JSON so int keys, tuples and floats compare like the stored file."""
    return json.loads(json.dumps(value))


@unittest.skipUnless((FEVER / "baseline").exists(), "committed baseline results not present")
class PromptGoldenTests(unittest.TestCase):
    """The frozen prompt must equal what every committed run recorded in run_config.json."""

    def test_system_prompt_matches_every_recorded_run(self):
        configs = sorted(ROOT.glob("results/**/run_config.json"))
        self.assertGreaterEqual(len(configs), 4)
        for path in configs:
            with self.subTest(run=str(path.relative_to(ROOT))):
                recorded = load_json(path)["prompt_template"]["system"]
                self.assertEqual(SYSTEM_PROMPT, recorded)

    def test_user_prompt_format(self):
        evidence = [
            {"page": "Page_A", "sentence_id": 3, "text": "First."},
            {"page": "Page_B", "sentence_id": 0, "text": "Second."},
        ]
        self.assertEqual(
            build_user_prompt("A claim.", evidence),
            "Claim:\nA claim.\n\nEvidence:\n"
            "[1] Page: Page_A | Sentence: 3\nFirst.\n\n"
            "[2] Page: Page_B | Sentence: 0\nSecond.",
        )


@unittest.skipUnless((FEVER / "baseline").exists(), "committed baseline results not present")
class BaselineGoldenTests(unittest.TestCase):
    directory = FEVER / "baseline"

    def test_metrics_match_committed_file(self):
        predictions = list(read_jsonl(self.directory / "predictions.jsonl"))
        self.assertEqual(
            normalise(calculate_metrics(predictions)), load_json(self.directory / "metrics.json")
        )

    def test_regenerated_artifacts_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "baseline"
            shutil.copytree(self.directory, copy)
            evaluate_results(copy)
            for name in ("metrics.json", "confusion_matrix.csv", "errors.csv", "report.md"):
                with self.subTest(file=name):
                    self.assertEqual(
                        (copy / name).read_bytes(), (self.directory / name).read_bytes()
                    )


@unittest.skipUnless((FEVER / "consistency_5run").exists(), "committed consistency results absent")
class ConsistencyGoldenTests(unittest.TestCase):
    directory = FEVER / "consistency_5run"

    def setUp(self):
        self.runs = [
            list(read_jsonl(self.directory / "runs" / f"run_{number:02d}_predictions.jsonl"))
            for number in range(1, 6)
        ]
        self.case_rows = consistency.build_case_rows(self.runs)
        run_metrics = [calculate_metrics(run) for run in self.runs]
        self.metrics = consistency.calculate_consistency_metrics(
            self.case_rows, self.runs, run_metrics
        )

    def test_metrics_match_committed_file(self):
        self.assertEqual(
            normalise(self.metrics), load_json(self.directory / "consistency_metrics.json")
        )

    def test_report_matches_committed_file(self):
        config = load_json(self.directory / "run_config.json")
        self.assertEqual(
            consistency._report(self.metrics, config),
            (self.directory / "report.md").read_text(encoding="utf-8"),
        )


@unittest.skipUnless((FEVER / "robustness").exists(), "committed robustness results absent")
class RobustnessGoldenTests(unittest.TestCase):
    directory = FEVER / "robustness"

    def setUp(self):
        predictions = self.directory / "predictions"
        self.metrics = robustness.calculate_robustness_metrics(
            list(read_jsonl(predictions / "control_predictions.jsonl")),
            list(read_jsonl(predictions / "reorder_predictions.jsonl")),
            list(read_jsonl(predictions / "distractor_predictions.jsonl")),
        )

    def test_metrics_match_committed_file(self):
        self.assertEqual(
            normalise(self.metrics), load_json(self.directory / "robustness_metrics.json")
        )

    def test_report_matches_committed_file(self):
        config = load_json(self.directory / "run_config.json")
        self.assertEqual(
            robustness._report(self.metrics, config),
            (self.directory / "report.md").read_text(encoding="utf-8"),
        )


@unittest.skipUnless(VITAMINC.exists() and PAIRS.exists(), "committed VitaminC results absent")
class VitaminCGoldenTests(unittest.TestCase):
    def setUp(self):
        pairs = list(read_jsonl(PAIRS))
        predictions = list(read_jsonl(VITAMINC / "predictions" / "predictions.jsonl"))
        self.rows = sensitivity.build_pair_rows(pairs, predictions)
        self.metrics = sensitivity.calculate_sensitivity_metrics(self.rows)

    def test_metrics_match_committed_file(self):
        self.assertEqual(
            normalise(self.metrics), load_json(VITAMINC / "sensitivity_metrics.json")
        )

    def test_report_matches_committed_file(self):
        config = load_json(VITAMINC / "run_config.json")
        self.assertEqual(
            sensitivity._report(self.metrics, config),
            (VITAMINC / "report.md").read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
