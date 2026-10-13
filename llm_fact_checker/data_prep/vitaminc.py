"""VitaminC test-set loading, contrastive pairing, and deterministic pair selection."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
import random
from typing import Any

from llm_fact_checker.common import (
    file_sha256,
    read_jsonl,
    write_jsonl,
)
from llm_fact_checker.data_prep import map_label

EXPECTED_TEST_SHA256 = "7f799e076dec184e31533d46704fef2022d08ef68a0db2c44f138d2b4e552f36"
# Two byte-different exports of the same test set are accepted. After `_normalise_row` they
# give identical rows, the same 20,232 valid pairs, and byte-identical selected pairs.
ACCEPTED_TEST_SHA256 = {
    EXPECTED_TEST_SHA256: "original release used for the pilot",
    "7ad1808dbc30c62e0a1427a53022d0dfaff668a1fde3c4b612a2d266edd753ad": "Hugging Face export (tals/vitaminc)",
}
# Row fields in the order of the original release; anything else is appended after.
_ROW_FIELD_ORDER = (
    "unique_id",
    "case_id",
    "wiki_revision_id",
    "FEVER_id",
    "label",
    "claim",
    "evidence",
    "page",
    "revision_type",
)
EXPECTED_TEST_ROWS = 55_197
EXPECTED_LABEL_COUNTS = {"SUPPORTS": 27_664, "REFUTES": 20_265, "NOT ENOUGH INFO": 7_268}
# Every contrastive pair `identify_contrastive_pairs` finds in the test set. 18,352 distinct
# claims: 1,051 claims appear in more than one pair (different cases), so the whole set is the
# only selection that does not de-duplicate claims.
EXPECTED_VALID_PAIRS = 20_232
REQUIRED_FIELDS = {"unique_id", "case_id", "label", "claim", "evidence", "page", "revision_type"}
SEED = 42
VITAMINC_RAW_PATH = Path("data/vitaminc/raw/vitaminc/test.jsonl")
VITAMINC_PROCESSED_PATH = Path("data/vitaminc/processed/vitaminc_sensitivity_100_pairs.jsonl")


def _normalise_row(row: dict[str, Any]) -> dict[str, Any]:
    """Give every accepted export the shape of the original release.

    The Hugging Face export adds a ``big_bench_canary`` field to every row, always includes
    both ``wiki_revision_id`` and ``FEVER_id`` (one of them an empty string), and orders keys
    differently. Dropping the canary and the empty id and restoring the key order makes the
    two exports indistinguishable to pairing, validation and the committed samples.
    """
    cleaned = {
        key: value
        for key, value in row.items()
        if key != "big_bench_canary"
        and not (key in ("wiki_revision_id", "FEVER_id") and value == "")
    }
    ordered = {key: cleaned[key] for key in _ROW_FIELD_ORDER if key in cleaned}
    ordered.update({key: value for key, value in cleaned.items() if key not in ordered})
    return ordered


def load_vitaminc_test(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    digest = file_sha256(path)
    if digest not in ACCEPTED_TEST_SHA256:
        accepted = ", ".join(
            f"{value} ({key[:8]}…)" for key, value in ACCEPTED_TEST_SHA256.items()
        )
        raise ValueError(
            f"VitaminC test SHA-256 {digest} is not a known export. Accepted: {accepted}"
        )
    rows = [_normalise_row(row) for row in read_jsonl(path)]
    if len(rows) != EXPECTED_TEST_ROWS:
        raise ValueError(f"Expected {EXPECTED_TEST_ROWS} VitaminC test rows, found {len(rows)}")
    missing = [
        row.get("unique_id", index)
        for index, row in enumerate(rows)
        if not REQUIRED_FIELDS <= row.keys()
    ]
    if missing:
        raise ValueError(f"VitaminC rows missing required fields: {missing[:5]}")
    labels = Counter(row["label"] for row in rows)
    if dict(labels) != EXPECTED_LABEL_COUNTS:
        raise ValueError(f"Unexpected VitaminC label counts: {dict(labels)}")
    return rows, {
        "sha256": digest,
        "row_count": len(rows),
        "field_names": sorted({key for row in rows for key in row}),
        "field_presence": dict(sorted(Counter(key for row in rows for key in row).items())),
        "label_counts": dict(labels),
        "revision_type_counts": dict(Counter(row["revision_type"] for row in rows)),
    }


def _pair_side(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim": row["claim"],
        "evidence": row["evidence"],
        "original_label": row["label"],
        "canonical_label": map_label(row["label"], "VitaminC"),
        "source_metadata": {
            key: value for key, value in row.items() if key not in {"claim", "evidence", "label"}
        },
    }


def _source_anchor(row: dict[str, Any]) -> tuple[str, Any]:
    key = "wiki_revision_id" if "wiki_revision_id" in row else "FEVER_id"
    return key, row[key]


def identify_contrastive_pairs(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Link only rows sharing explicit case_id and exact claim text."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["case_id"], row["claim"])].append(row)

    pairs = []
    for (case_id, claim), group in groups.items():
        supports = [row for row in group if row["label"] == "SUPPORTS"]
        refutes = [row for row in group if row["label"] == "REFUTES"]
        if len(supports) != 1 or len(refutes) != 1:
            continue
        support, refute = supports[0], refutes[0]
        if support["evidence"] == refute["evidence"]:
            continue
        if (
            support["page"] != refute["page"]
            or support["revision_type"] != refute["revision_type"]
            or _source_anchor(support) != _source_anchor(refute)
        ):
            continue
        pair_id = f"{support['unique_id']}__{refute['unique_id']}"
        pairs.append(
            {
                "pair_id": pair_id,
                "claim": claim,
                "support": _pair_side(support),
                "refute": _pair_side(refute),
                "source_metadata": {
                    "case_id": case_id,
                    "pairing_key": ["case_id", "claim"],
                    "support_unique_id": support["unique_id"],
                    "refute_unique_id": refute["unique_id"],
                },
            }
        )
    return sorted(pairs, key=lambda pair: pair["pair_id"])


def select_pairs(
    candidates: list[dict[str, Any]], count: int = 100, seed: int = SEED
) -> list[dict[str, Any]]:
    unique_claim_candidates = []
    seen_claims = set()
    for pair in sorted(candidates, key=lambda item: item["pair_id"]):
        if pair["claim"] in seen_claims:
            continue
        seen_claims.add(pair["claim"])
        unique_claim_candidates.append(pair)
    if len(unique_claim_candidates) < count:
        raise ValueError(
            f"Need {count} unique-claim contrastive pairs, found {len(unique_claim_candidates)}"
        )
    selected = random.Random(seed).sample(unique_claim_candidates, count)
    return sorted(selected, key=lambda pair: pair["pair_id"])


def validate_pairs(
    pairs: list[dict[str, Any]],
    expected_count: int | None = None,
    *,
    unique_claims: bool = True,
) -> None:
    """Check every pair. ``unique_claims=False`` is for the whole set, where a claim can occur
    in several pairs (each with its own case); pair IDs must still be unique."""
    if expected_count is not None and len(pairs) != expected_count:
        raise ValueError(f"Expected {expected_count} pairs, found {len(pairs)}")
    if len({pair["pair_id"] for pair in pairs}) != len(pairs):
        raise ValueError("Duplicate VitaminC pair IDs")
    if unique_claims and len({pair["claim"] for pair in pairs}) != len(pairs):
        raise ValueError("Duplicate claims in selected VitaminC pairs")
    for pair in pairs:
        support, refute = pair["support"], pair["refute"]
        if support["claim"] != pair["claim"] or refute["claim"] != pair["claim"]:
            raise ValueError(f"Pair {pair['pair_id']} does not use the exact same claim")
        if support["original_label"] != "SUPPORTS" or refute["original_label"] != "REFUTES":
            raise ValueError(f"Pair {pair['pair_id']} does not have SUPPORTS/REFUTES labels")
        if support["canonical_label"] != "SUPPORTED" or refute["canonical_label"] != "REFUTED":
            raise ValueError(f"Pair {pair['pair_id']} has invalid canonical labels")
        if support["evidence"] == refute["evidence"]:
            raise ValueError(f"Pair {pair['pair_id']} has identical evidence")
        if support["source_metadata"]["case_id"] != refute["source_metadata"]["case_id"]:
            raise ValueError(f"Pair {pair['pair_id']} crosses case IDs")
        if support["source_metadata"]["page"] != refute["source_metadata"]["page"]:
            raise ValueError(f"Pair {pair['pair_id']} crosses pages")
        if (
            support["source_metadata"]["revision_type"]
            != refute["source_metadata"]["revision_type"]
        ):
            raise ValueError(f"Pair {pair['pair_id']} crosses revision types")
        if _source_anchor(support["source_metadata"]) != _source_anchor(refute["source_metadata"]):
            raise ValueError(f"Pair {pair['pair_id']} crosses source revisions")


def validate_pairs_against_candidates(
    selected: list[dict[str, Any]], candidates: list[dict[str, Any]]
) -> None:
    candidate_by_id = {pair["pair_id"]: pair for pair in candidates}
    for pair in selected:
        if candidate_by_id.get(pair["pair_id"]) != pair:
            raise ValueError(f"Selected pair is not an unchanged candidate: {pair['pair_id']}")


def _audit_markdown(pairs: list[dict[str, Any]], count: int = 20) -> str:
    sections = [
        "# VitaminC sensitivity pair audit",
        "",
        "Pairs share the same `case_id` and exact claim. Each has one unchanged VitaminC SUPPORTS row and one unchanged REFUTES row with different evidence.",
    ]
    for pair in pairs[:count]:
        sections.extend(
            [
                "",
                f"## PAIR ID: {pair['pair_id']}",
                "",
                "**Claim:**",
                "",
                pair["claim"],
                "",
                "**Evidence A:**",
                "",
                pair["support"]["evidence"],
                "",
                "**Gold: SUPPORTED**",
                "",
                "**Evidence B:**",
                "",
                pair["refute"]["evidence"],
                "",
                "**Gold: REFUTED**",
            ]
        )
    return "\n".join(sections) + "\n"


def prepare_vitaminc_pairs(
    raw_path: Path,
    processed_path: Path,
    prepared_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_paths = [processed_path, prepared_dir / "pair_audit.md"]
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite VitaminC prepared files: {existing}")
    rows, source_summary = load_vitaminc_test(raw_path)
    candidates = identify_contrastive_pairs(rows)
    selected = select_pairs(candidates)
    validate_pairs(selected, expected_count=100)
    validate_pairs_against_candidates(selected, candidates)
    write_jsonl(processed_path, selected)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    (prepared_dir / "pair_audit.md").write_text(_audit_markdown(selected), encoding="utf-8")
    return {
        **source_summary,
        "pairing_method": "exact case_id plus exact claim; exactly one SUPPORTS and one REFUTES row; different evidence",
        "valid_contrastive_pairs": len(candidates),
        "unique_claim_candidate_pairs": len({pair["claim"] for pair in candidates}),
        "selected_pairs": len(selected),
        "selection_seed": SEED,
    }
