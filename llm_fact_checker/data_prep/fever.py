"""FEVER loading, deterministic sampling, and gold-evidence extraction."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import random
import sys
from typing import Any

from llm_fact_checker.common import read_jsonl, write_jsonl
from llm_fact_checker.data_prep import map_label

TARGET_LABELS = ("SUPPORTS", "REFUTES")
DEV_PATH = Path("data/fever/raw/shared_task_dev.jsonl")
WIKI_DIR = Path("data/fever/raw/wiki-pages/wiki-pages")
PREPARED_PATH = Path("data/fever/processed/fever_baseline_150.jsonl")


def select_evidence_set(evidence_sets: list[list[list[Any]]]) -> list[dict[str, Any]] | None:
    """Choose the smallest complete evidence set, then break ties by reference."""
    candidates: list[tuple[tuple[tuple[str, int], ...], list[dict[str, Any]]]] = []
    for evidence_set in evidence_sets:
        if not evidence_set:
            continue
        references: set[tuple[str, int]] = set()
        complete = True
        for item in evidence_set:
            if len(item) < 4 or item[2] is None or item[3] is None:
                complete = False
                break
            references.add((str(item[2]), int(item[3])))
        if complete and references:
            key = tuple(sorted(references))
            selected = [{"page": page, "sentence_id": sentence_id} for page, sentence_id in key]
            candidates.append((key, selected))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (len(item[0]), item[0]))[1]


def candidate_order(
    records: Iterable[dict[str, Any]], label: str, seed: int
) -> list[dict[str, Any]]:
    """Return a reproducibly shuffled, ID- and claim-deduplicated candidate list."""
    seen_ids: set[int] = set()
    seen_claims: set[str] = set()
    eligible = []
    for record in records:
        if record.get("label") != label:
            continue
        evidence_set = select_evidence_set(record.get("evidence", []))
        fever_id = int(record["id"])
        claim_key = " ".join(str(record["claim"]).split()).casefold()
        if evidence_set:
            eligible.append(
                {
                    "fever_id": fever_id,
                    "claim": record["claim"],
                    "claim_key": claim_key,
                    "original_fever_label": label,
                    "gold_label": map_label(label, "FEVER"),
                    "selected_evidence_set": evidence_set,
                }
            )
    candidates = []
    for item in sorted(eligible, key=lambda candidate: candidate["fever_id"]):
        if item["fever_id"] not in seen_ids and item["claim_key"] not in seen_claims:
            seen_ids.add(item["fever_id"])
            seen_claims.add(item.pop("claim_key"))
            candidates.append(item)
    random.Random(seed).shuffle(candidates)
    return candidates


def parse_wiki_lines(lines: str) -> dict[int, str]:
    """Parse FEVER sentence-id/text pairs while ignoring hyperlink columns."""
    sentences = {}
    for row in lines.splitlines():
        parts = row.split("\t", 2)
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].strip():
            sentences[int(parts[0])] = parts[1].strip()
    return sentences


def scan_wikipedia(
    wiki_files: Iterable[Path], needed: set[tuple[str, int]]
) -> tuple[dict[tuple[str, int], str], set[tuple[str, int]]]:
    """Stream Wikipedia files and retain only requested sentences."""
    remaining = set(needed)
    found: dict[tuple[str, int], str] = {}
    needed_by_page: dict[str, set[int]] = {}
    for page, sentence_id in needed:
        needed_by_page.setdefault(page, set()).add(sentence_id)
    if not remaining:
        return found, remaining
    paths = sorted(wiki_files)
    for file_number, path in enumerate(paths, 1):
        print(
            f"Scanning Wikipedia {file_number}/{len(paths)}: {path.name} "
            f"({len(remaining)} references remaining)",
            file=sys.stderr,
        )
        for article in read_jsonl(path):
            page = article.get("id")
            if not page:
                continue
            page_needs = needed_by_page.get(page)
            if not page_needs:
                continue
            sentences = parse_wiki_lines(article.get("lines", ""))
            for sentence_id in page_needs:
                text = sentences.get(sentence_id)
                if text:
                    reference = (page, sentence_id)
                    found[reference] = text
                    remaining.remove(reference)
            needed_by_page.pop(page)
            if not remaining:
                return found, remaining
    return found, remaining


def _references(record: dict[str, Any]) -> set[tuple[str, int]]:
    return {(item["page"], item["sentence_id"]) for item in record["selected_evidence_set"]}


def _attach_evidence(
    record: dict[str, Any], resolved: dict[tuple[str, int], str]
) -> dict[str, Any]:
    """Add the sentence text; every reference is resolved by the time this is called."""
    result = dict(record)
    result["evidence"] = [
        {
            "page": item["page"],
            "sentence_id": item["sentence_id"],
            "text": resolved[(item["page"], item["sentence_id"])],
        }
        for item in record["selected_evidence_set"]
    ]
    result["wikipedia_page_ids"] = sorted({item["page"] for item in result["evidence"]})
    result["sentence_ids"] = [item["sentence_id"] for item in result["evidence"]]
    return result


def prepare_dataset(
    dev_path: Path,
    wiki_dir: Path,
    output_path: Path,
    per_label: int = 75,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build a balanced sample, replacing unresolved cases deterministically."""
    records = list(read_jsonl(dev_path))
    orders = {label: candidate_order(records, label, seed) for label in TARGET_LABELS}
    selected = {label: candidates[:per_label] for label, candidates in orders.items()}
    next_index = {label: per_label for label in TARGET_LABELS}
    resolved: dict[tuple[str, int], str] = {}
    wiki_files = list(wiki_dir.glob("wiki-*.jsonl"))
    if not wiki_files:
        raise FileNotFoundError(f"No wiki-*.jsonl files found in {wiki_dir}")

    while True:
        needed = set().union(
            *(_references(record) for group in selected.values() for record in group)
        )
        found, unresolved = scan_wikipedia(wiki_files, needed - resolved.keys())
        resolved.update(found)
        unusable = {
            label: [record for record in group if not _references(record) <= resolved.keys()]
            for label, group in selected.items()
        }
        if not any(unusable.values()):
            break
        print(f"Unresolved evidence references: {sorted(unresolved)!r}", file=sys.stderr)
        for label in TARGET_LABELS:
            if not unusable[label]:
                continue
            selected[label] = [
                record for record in selected[label] if record not in unusable[label]
            ]
            while len(selected[label]) < per_label:
                if next_index[label] >= len(orders[label]):
                    raise RuntimeError(f"Not enough resolvable {label} candidates")
                selected[label].append(orders[label][next_index[label]])
                next_index[label] += 1

    final = sorted(
        (
            _attach_evidence(record, resolved)
            for label in TARGET_LABELS
            for record in selected[label]
        ),
        key=lambda item: item["fever_id"],
    )
    counts = {
        label: sum(item["gold_label"] == map_label(label, "FEVER") for item in final)
        for label in TARGET_LABELS
    }
    if len(final) != per_label * 2 or counts != {label: per_label for label in TARGET_LABELS}:
        raise AssertionError(f"Invalid prepared dataset: total={len(final)}, counts={counts}")
    write_jsonl(output_path, final)
    return final
