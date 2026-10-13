"""Prepare reproducible experiment inputs for the audited VitaminC train split."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from typing import Any

from llm_fact_checker.common import file_sha256, read_jsonl, write_jsonl
from llm_fact_checker.data_prep.vitaminc import LABEL_MAP, REQUIRED_FIELDS

TRAIN_PATH = Path("data/vitaminc/raw/vitaminc/train.jsonl")
TRAIN_OUTPUT_DIR = Path("data/vitaminc/processed/train_full")
EXPECTED_TRAIN_SHA256 = "446163d81ec9b654d2514e8ba5b8f2e4bd8938342e34c98e7660fe05a8a3ca39"
EXPECTED_ROWS = 370_653
EXPECTED_LABEL_COUNTS = {
    "SUPPORTS": 185_714,
    "REFUTES": 131_958,
    "NOT ENOUGH INFO": 52_981,
}
EXPECTED_PAIR_COUNTS = {
    "SUPPORTS_REFUTES": 131_709,
    "SUPPORTS_NEI": 52_779,
    "REFUTES_NEI": 102,
}
LABEL_ORDER = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
PAIR_TYPES = {
    ("SUPPORTS", "REFUTES"): "SUPPORTS_REFUTES",
    ("SUPPORTS", "NOT ENOUGH INFO"): "SUPPORTS_NEI",
    ("REFUTES", "NOT ENOUGH INFO"): "REFUTES_NEI",
}
KNOWN_CASE_FLAGS = {"5ed4de06c9e77c0008489b3e": "page_title_mismatch"}
FULL_EVAL_COUNTS = {"SUPPORTS_REFUTES": 1000, "SUPPORTS_NEI": 1000, "REFUTES_NEI": None}
FULL_EVAL_SELECTION = Path("data/vitaminc/processed/vitaminc_train_eval_pair_ids.jsonl")
PILOT_PAIRS = Path("data/vitaminc/processed/vitaminc_sensitivity_100_pairs.jsonl")
EXPECTED_PAIRS_SHA256 = "92947ec1006da123750078c56f10d0ee47332b191b57e053ad99b431364d08f8"


def _source_anchor(row: dict[str, Any]) -> tuple[str, Any]:
    key = "wiki_revision_id" if row["revision_type"] == "real" else "FEVER_id"
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing {key} for VitaminC row {row['unique_id']}")
    return key, value


def _flags(row: dict[str, Any]) -> list[str]:
    flags = []
    if row["case_id"] in KNOWN_CASE_FLAGS:
        flags.append(KNOWN_CASE_FLAGS[row["case_id"]])
    if "\ufffd" in row["claim"] or "\ufffd" in row["evidence"]:
        flags.append("replacement_character")
    return flags


def strict_transition_pairs(
    groups: dict[tuple[str, str], list[dict[str, Any]]],
) -> Iterator[dict[str, Any]]:
    """Yield only exact two-row, two-evidence pairs from one source and page."""
    for (case_id, claim), group in sorted(groups.items()):
        if len(group) != 2 or group[0]["evidence"] == group[1]["evidence"]:
            continue
        left, right = sorted(group, key=lambda row: LABEL_ORDER[row["label"]])
        pair_type = PAIR_TYPES.get((left["label"], right["label"]))
        if pair_type is None or any(
            left[field] != right[field] for field in ("page", "revision_type", "source_anchor")
        ):
            continue
        yield {
            "pair_id": f"{left['unique_id']}__{right['unique_id']}",
            "case_id": case_id,
            "claim": claim,
            "pair_type": pair_type,
            "left_unique_id": left["unique_id"],
            "right_unique_id": right["unique_id"],
            "left_gold_label": LABEL_MAP[left["label"]],
            "right_gold_label": LABEL_MAP[right["label"]],
            "revision_type": left["revision_type"],
            "page": left["page"],
            "audit_flags": sorted(set(left["audit_flags"] + right["audit_flags"])),
        }


def _index_rows(
    rows: Iterable[dict[str, Any]], row_path: Path
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], Counter, Counter, Counter]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    label_counts: Counter = Counter()
    source_counts: Counter = Counter()
    flag_counts: Counter = Counter()
    seen_ids: set[str] = set()
    with row_path.open("w", encoding="utf-8") as output:
        for index, row in enumerate(rows, 1):
            missing = REQUIRED_FIELDS - row.keys()
            if missing:
                raise ValueError(f"VitaminC row {index} is missing fields: {sorted(missing)}")
            for field in ("unique_id", "case_id", "claim", "evidence", "page"):
                if not isinstance(row[field], str) or not row[field].strip():
                    raise ValueError(f"VitaminC row {index} has empty or invalid {field}")
            if row["label"] not in LABEL_MAP or row["revision_type"] not in {"real", "synthetic"}:
                raise ValueError(f"VitaminC row {index} has an unexpected label or source")
            unique_id = row["unique_id"]
            if unique_id in seen_ids:
                raise ValueError(f"Duplicate VitaminC unique_id: {unique_id}")
            seen_ids.add(unique_id)
            anchor = _source_anchor(row)
            flags = _flags(row)
            label_counts[row["label"]] += 1
            source_counts[row["revision_type"]] += 1
            flag_counts.update(flags)
            output.write(
                json.dumps(
                    {
                        "unique_id": unique_id,
                        "case_id": row["case_id"],
                        "claim": row["claim"],
                        "evidence": row["evidence"],
                        "page": row["page"],
                        "gold_label": LABEL_MAP[row["label"]],
                        "revision_type": row["revision_type"],
                        "source_anchor": {anchor[0]: anchor[1]},
                        "audit_flags": flags,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            groups[(row["case_id"], row["claim"])].append(
                {
                    "unique_id": unique_id,
                    "label": row["label"],
                    "evidence": row["evidence"],
                    "page": row["page"],
                    "revision_type": row["revision_type"],
                    "source_anchor": anchor,
                    "audit_flags": flags,
                }
            )
    return groups, label_counts, source_counts, flag_counts


def prepare_vitaminc_train(
    raw_path: Path = TRAIN_PATH, output_dir: Path = TRAIN_OUTPUT_DIR
) -> dict[str, Any]:
    """Validate the frozen raw file and publish unchanged evidence with pair indexes."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite VitaminC train preparation: {output_dir}")
    raw_digest = file_sha256(raw_path)
    if raw_digest != EXPECTED_TRAIN_SHA256:
        raise ValueError(f"VitaminC train SHA-256 mismatch: {raw_digest}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".vitaminc_train_", dir=output_dir.parent) as directory:
        temporary = Path(directory)
        rows_path = temporary / "rows.jsonl"
        groups, labels, sources, flags = _index_rows(read_jsonl(raw_path), rows_path)
        if sum(labels.values()) != EXPECTED_ROWS or dict(labels) != EXPECTED_LABEL_COUNTS:
            raise ValueError(f"Unexpected VitaminC train rows or labels: {dict(labels)}")
        pairs_path = temporary / "pairs.jsonl"
        pair_counts: Counter = Counter()

        def counted_pairs() -> Iterator[dict[str, Any]]:
            for pair in strict_transition_pairs(groups):
                pair_counts[pair["pair_type"]] += 1
                yield pair

        write_jsonl(pairs_path, counted_pairs())
        if dict(pair_counts) != EXPECTED_PAIR_COUNTS:
            raise ValueError(f"Unexpected VitaminC train pair counts: {dict(pair_counts)}")
        summary = {
            "raw_path": str(raw_path),
            "raw_sha256": raw_digest,
            "row_count": sum(labels.values()),
            "label_counts": dict(labels),
            "source_counts": dict(sources),
            "strict_pair_counts": dict(pair_counts),
            "audit_flag_counts": dict(flags),
            "pairing_rule": "same case_id and exact claim; exactly two rows with different evidence, labels, same page, source type and source revision",
            "rows_sha256": file_sha256(rows_path),
            "pairs_sha256": file_sha256(pairs_path),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if output_dir.exists():
            raise FileExistsError(
                f"Refusing to overwrite VitaminC train preparation: {output_dir}"
            )
        temporary.rename(output_dir)
    return summary


def select_full_eval_pairs(
    pairs: Iterable[dict[str, Any]], pilot_claims: set[str], counts: dict[str, int | None]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select one pair per exact claim and type, then sample fixed pair counts."""
    pools: dict[str, dict[str, dict[str, Any]]] = {pair_type: {} for pair_type in counts}
    for pair in pairs:
        pair_type, claim = pair["pair_type"], pair["claim"]
        if pair_type not in pools:
            raise ValueError(f"Unexpected pair type: {pair_type}")
        if claim in pilot_claims:
            continue
        previous = pools[pair_type].get(claim)
        if previous is None or pair["pair_id"] < previous["pair_id"]:
            pools[pair_type][claim] = pair

    selected = []
    eligible_counts = {}
    for pair_type, count in counts.items():
        candidates = sorted(pools[pair_type].values(), key=lambda pair: pair["pair_id"])
        eligible_counts[pair_type] = len(candidates)
        sample_size = len(candidates) if count is None else count
        if not candidates or sample_size < 1 or sample_size > len(candidates):
            raise ValueError(f"Not enough eligible {pair_type} pairs: {len(candidates)}")
        selected.extend(
            sorted(random.Random(42).sample(candidates, sample_size), key=lambda pair: pair["pair_id"])
        )
    row_ids = [
        pair[f"{side}_unique_id"] for pair in selected for side in ("left", "right")
    ]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("Selected pairs reuse a VitaminC row")
    return selected, eligible_counts


def prepare_full_eval_selection(
    pairs_path: Path = TRAIN_OUTPUT_DIR / "pairs.jsonl",
    pilot_path: Path = PILOT_PAIRS,
    output_path: Path = FULL_EVAL_SELECTION,
) -> dict[str, Any]:
    """Freeze selected IDs and provenance without changing prepared or raw data."""
    manifest_path = output_path.with_suffix(".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation selection: {output_path}")
    pairs_hash = file_sha256(pairs_path)
    if pairs_hash != EXPECTED_PAIRS_SHA256:
        raise ValueError(f"VitaminC pair index SHA-256 mismatch: {pairs_hash}")
    pilot_pairs = list(read_jsonl(pilot_path))
    pilot_claims = {pair["claim"] for pair in pilot_pairs}
    if len(pilot_pairs) != 100 or len(pilot_claims) != 100:
        raise ValueError("Expected 100 distinct pilot claims")
    selected, eligible_counts = select_full_eval_pairs(
        read_jsonl(pairs_path), pilot_claims, FULL_EVAL_COUNTS
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for pair in selected:
            handle.write(json.dumps({"pair_id": pair["pair_id"]}) + "\n")
    summary = {
        "selection_path": str(output_path),
        "selection_sha256": file_sha256(output_path),
        "pairs_path": str(pairs_path),
        "pairs_sha256": pairs_hash,
        "pilot_path": str(pilot_path),
        "pilot_sha256": file_sha256(pilot_path),
        "seed_per_pair_type": 42,
        "rule": "exclude pilot exact claims; lowest pair_id per exact claim and type; sorted IDs; sample without replacement",
        "eligible_unique_claim_counts": eligible_counts,
        "selected_pair_counts": dict(Counter(pair["pair_type"] for pair in selected)),
        "selected_source_counts": {
            pair_type: dict(
                Counter(pair["revision_type"] for pair in selected if pair["pair_type"] == pair_type)
            )
            for pair_type in FULL_EVAL_COUNTS
        },
        "selected_flag_counts": dict(
            Counter(flag for pair in selected for flag in pair["audit_flags"])
        ),
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary
