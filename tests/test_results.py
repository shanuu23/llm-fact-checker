from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from llm_fact_checker.results import DEFAULT_RESULTS_DIR, ResultRun, discover_runs, select

NEI = "NOT ENOUGH INFORMATION"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv_file(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def baseline_metrics() -> dict:
    return {
        "total_cases": 4,
        "number_correct": 3,
        "overall_accuracy": 0.75,
        "class_accuracy": {"SUPPORTED": 1.0, "REFUTED": 0.5},
        "prediction_counts": {"SUPPORTED": 2, "REFUTED": 1, NEI: 1, "INVALID": 0},
        "invalid_format_rate": 0.0,
    }


def make_baseline(directory: Path) -> None:
    write_json(directory / "metrics.json", baseline_metrics())
    write_json(
        directory / "run_config.json",
        {
            "ollama_model": "demo:1b",
            "model_seed": 42,
            "temperature": 0,
            "ollama_version": "0.0.1",
            "ollama_model_metadata": {"digest": "abc123"},
            "git_commit": "deadbeef",
        },
    )
    (directory / "predictions.jsonl").write_text(
        json.dumps({"fever_id": 1, "gold_label": "SUPPORTED", "parsed_prediction": "SUPPORTED"})
        + "\n",
        encoding="utf-8",
    )


def make_consistency(directory: Path) -> None:
    states = {
        label: {"count": 1, "percentage": 0.25}
        for label in ("STABLE_CORRECT", "STABLE_WRONG", "UNSTABLE", "INCOMPLETE")
    }
    write_json(
        directory / "consistency_metrics.json",
        {
            "total_cases": 4,
            "exact_consistency_rate": 0.75,
            "mean_modal_agreement": 0.9,
            "accuracy_summary": {"mean": 0.7},
            "stability_classification": states,
        },
    )
    write_json(
        directory / "run_config.json",
        {"ollama_model": "demo:1b", "run_seeds": [1, 2, 3, 4, 5], "temperature": 0.2},
    )
    write_csv_file(
        directory / "consistency_by_case.csv",
        [
            {
                "fever_id": "42",
                "claim": "1990",
                "unique_label_count": "2",
                "modal_label": "SUPPORTED",
                "modal_count": "4",
                "modal_agreement": "0.8",
                "exact_agreement": "False",
                "tie": "False",
                "incomplete": "False",
                "consensus_prediction": "",
            }
        ],
    )


def make_tree(root: Path) -> None:
    fever = root / "fever" / "demo_1b"
    make_baseline(fever / "baseline")
    make_consistency(fever / "consistency_5run")
    make_baseline(fever / "smoke_3")
    make_consistency(fever / "consistency_5run_smoke_3")
    make_baseline(fever / "baseline" / "smoke")  # nested smoke dir, not a separate experiment
    (fever / "notes").mkdir()
    (fever / "notes" / "todo.txt").write_text("not a result", encoding="utf-8")
    make_baseline(root / "fever" / "other_7b" / "baseline")


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "results"
        make_tree(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_finds_experiments_and_skips_smoke_and_unknown_folders(self):
        runs = discover_runs(self.root)
        self.assertEqual(
            [(run.model, run.experiment, run.kind) for run in runs],
            [
                ("demo_1b", "baseline", "baseline"),
                ("demo_1b", "consistency_5run", "consistency"),
                ("other_7b", "baseline", "baseline"),
            ],
        )
        self.assertFalse(any(run.is_smoke for run in runs))

    def test_include_smoke_adds_smoke_and_nested_smoke_runs(self):
        runs = discover_runs(self.root, include_smoke=True)
        smoke = {run.label for run in runs if run.is_smoke}
        self.assertEqual(
            smoke,
            {
                "fever/demo_1b/smoke_3",
                "fever/demo_1b/consistency_5run_smoke_3",
                "fever/demo_1b/baseline/smoke",
            },
        )

    def test_missing_root_raises_instead_of_returning_nothing(self):
        with self.assertRaises(FileNotFoundError):
            discover_runs(self.root / "does-not-exist")

    def test_select_filters_by_model_and_kind(self):
        runs = discover_runs(self.root)
        self.assertEqual(len(select(runs, model="demo_1b")), 2)
        self.assertEqual(len(select(runs, kind="baseline")), 2)
        self.assertEqual([r.model for r in select(runs, model="other_7b", kind="baseline")], ["other_7b"])
        self.assertEqual(select(runs, dataset="vitaminc"), [])


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "results"
        make_tree(self.root)
        self.runs = {run.experiment: run for run in discover_runs(self.root)}

    def tearDown(self):
        self.tmp.cleanup()

    def test_headline_uses_stable_names(self):
        self.assertEqual(
            self.runs["baseline"].headline(),
            {
                "cases": 4,
                "accuracy": 0.75,
                "supported_accuracy": 1.0,
                "refuted_accuracy": 0.5,
                "nei_rate": 0.25,
                "invalid_format_rate": 0.0,
            },
        )
        consistency = self.runs["consistency_5run"].headline()
        self.assertEqual(consistency["exact_consistency"], 0.75)
        self.assertEqual(consistency["stable_wrong_rate"], 0.25)

    def test_csv_cells_are_typed_by_column_and_text_is_left_alone(self):
        (row,) = self.runs["consistency_5run"].cases()
        self.assertEqual(row["fever_id"], 42)
        self.assertEqual(row["modal_count"], 4)
        self.assertEqual(row["modal_agreement"], 0.8)
        self.assertIs(row["exact_agreement"], False)
        self.assertEqual(row["claim"], "1990")  # text that looks numeric stays a string
        self.assertIsNone(row["consensus_prediction"])  # empty cell is None

    def test_jsonl_cases_are_returned_as_stored(self):
        (row,) = self.runs["baseline"].cases()
        self.assertEqual(row["parsed_prediction"], "SUPPORTED")

    def test_provenance_reads_model_digest_and_seeds(self):
        baseline = self.runs["baseline"].provenance()
        self.assertEqual(baseline["model_tag"], "demo:1b")
        self.assertEqual(baseline["model_digest"], "abc123")
        self.assertEqual(baseline["seeds"], [42])
        self.assertEqual(self.runs["consistency_5run"].provenance()["seeds"], [1, 2, 3, 4, 5])

    def test_missing_file_names_the_path(self):
        run = ResultRun("fever", "m", "baseline", "baseline", self.root / "nope")
        with self.assertRaisesRegex(FileNotFoundError, "metrics.json"):
            run.metrics()


@unittest.skipUnless(
    (DEFAULT_RESULTS_DIR / "fever" / "llama3.2_3b" / "baseline").exists(),
    "committed pilot results not present",
)
class CommittedPilotResultsTests(unittest.TestCase):
    """Pin the loader against the real committed Llama 3.2 3B results."""

    def setUp(self):
        # whole-dataset runs (*_full) sit beside the pilot runs; these tests pin the pilot ones
        self.runs = [
            run
            for run in select(discover_runs(), model="llama3.2_3b")
            if not run.experiment.endswith("_full")
        ]
        self.by_kind = {run.kind: run for run in self.runs}

    def test_finds_exactly_the_four_experiments_and_no_smoke_runs(self):
        self.assertEqual(
            sorted(run.label for run in self.runs),
            [
                "fever/llama3.2_3b/baseline",
                "fever/llama3.2_3b/consistency_5run",
                "fever/llama3.2_3b/robustness",
                "vitaminc/llama3.2_3b/sensitivity",
            ],
        )

    def test_case_tables_have_the_expected_row_counts(self):
        counts = {kind: len(run.cases()) for kind, run in self.by_kind.items()}
        self.assertEqual(
            counts,
            {"baseline": 150, "consistency": 150, "robustness": 150, "sensitivity": 100},
        )

    def test_headline_numbers_match_the_published_pilot_results(self):
        baseline = self.by_kind["baseline"].headline()
        self.assertAlmostEqual(baseline["accuracy"], 97 / 150)
        self.assertAlmostEqual(baseline["nei_rate"], 35 / 150)

        consistency = self.by_kind["consistency"].headline()
        self.assertAlmostEqual(consistency["exact_consistency"], 120 / 150)

        robustness = self.by_kind["robustness"].headline()
        self.assertAlmostEqual(robustness["reorder_invariance"], 8 / 11)
        self.assertAlmostEqual(robustness["distractor_invariance"], 59 / 150)
        self.assertAlmostEqual(robustness["distractor_accuracy"], 22 / 150)

        sensitivity = self.by_kind["sensitivity"].headline()
        self.assertAlmostEqual(sensitivity["correct_sensitivity"], 0.27)
        self.assertAlmostEqual(sensitivity["flip_rate"], 0.72)

    def test_whole_dataset_runs_are_separate_runs_with_their_own_dataset_hash(self):
        runs = select(discover_runs(), model="llama3.2_3b", kind="baseline")
        by_name = {run.experiment: run for run in runs}
        pilot = by_name["baseline"].provenance()["dataset_sha256"]
        for name, run in by_name.items():
            if name.endswith("_full"):
                self.assertNotEqual(run.provenance()["dataset_sha256"], pilot)
                self.assertGreater(run.headline()["cases"], 150)

    def test_provenance_identifies_the_model_build(self):
        for run in self.runs:
            with self.subTest(run=run.label):
                provenance = run.provenance()
                self.assertEqual(provenance["model_tag"], "llama3.2:3b")
                self.assertTrue(provenance["model_digest"].startswith("a80c4f17"))

    def test_typed_values_in_real_tables(self):
        row = self.by_kind["consistency"].cases()[0]
        self.assertIsInstance(row["fever_id"], int)
        self.assertIsInstance(row["exact_agreement"], bool)
        pair = self.by_kind["sensitivity"].cases()[0]
        self.assertIsInstance(pair["correct_sensitivity"], bool)


if __name__ == "__main__":
    unittest.main()
