"""FEVER loading, deterministic sampling, and gold-evidence extraction."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import random
import sys
from typing import Any
import unicodedata

from llm_fact_checker.common import read_jsonl, write_jsonl
from llm_fact_checker.data_prep import map_label

TARGET_LABELS = ("SUPPORTS", "REFUTES")
DEV_PATH = Path("data/fever/raw/shared_task_dev.jsonl")
WIKI_DIR = Path("data/fever/raw/wiki-pages/wiki-pages")
PREPARED_PATH = Path("data/fever/processed/fever_baseline_150.jsonl")
# The whole dev set (derived, about 8 MB, not committed).
FULL_PATH = Path("data/fever/full/fever_dev_full.jsonl")


def nfc(text: str) -> str:
    """Canonical Unicode form, so an accented title typed two ways still compares equal."""
    return unicodedata.normalize("NFC", text)


def audit_flags(records: Iterable[dict[str, Any]]) -> dict[int, list[str]]:
    """Mark records the FEVER audit found possibly inconsistent. Nothing is removed or relabelled.

    - ``conflicting_duplicate_label``: the exact claim text appears more than once with
      different labels (54 groups, 124 records in the dev file)
    - ``normalized_claim_conflict``: claims that differ only in case, spacing or Unicode form
      carry different labels (one group, 2 records)

    Flags are possible label inconsistencies, not proven errors.
    """
    exact: dict[str, list[dict[str, Any]]] = {}
    normalised: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        exact.setdefault(record["claim"], []).append(record)
        key = " ".join(nfc(record["claim"]).casefold().split())
        normalised.setdefault(key, []).append(record)
    flags: dict[int, set[str]] = {}
    for group in exact.values():
        if len(group) > 1 and len({item["label"] for item in group}) > 1:
            for item in group:
                flags.setdefault(int(item["id"]), set()).add("conflicting_duplicate_label")
    for group in normalised.values():
        differently_written = len({item["claim"] for item in group}) > 1
        if differently_written and len({item["label"] for item in group}) > 1:
            for item in group:
                flags.setdefault(int(item["id"]), set()).add("normalized_claim_conflict")
    return {record_id: sorted(found) for record_id, found in flags.items()}


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
    records: Iterable[dict[str, Any]], label: str, seed: int, dedupe: bool = True
) -> list[dict[str, Any]]:
    """Return a reproducibly shuffled candidate list, de-duplicated by ID and by claim text.

    ``dedupe=False`` keeps repeated claim text (IDs stay unique); the frozen sample uses the
    default.
    """
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
        claim_key = item.pop("claim_key")
        if item["fever_id"] in seen_ids or (dedupe and claim_key in seen_claims):
            continue
        seen_ids.add(item["fever_id"])
        seen_claims.add(claim_key)
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
    wiki_files: Iterable[Path], needed: set[tuple[str, int]], match_nfc: bool = False
) -> tuple[dict[tuple[str, int], str], set[tuple[str, int]]]:
    """Stream Wikipedia files and retain only requested sentences.

    With ``match_nfc`` titles are compared in NFC form, because the same accented title can be
    encoded two ways (the FEVER audit found 32 pages that only match this way). It is off by
    default so the frozen 150-case sample rebuilds exactly as committed: one of its candidates
    (FEVER 95159, "Ricardo_Montalb\u00e1n") was skipped because of this mismatch. Results are keyed
    by the title exactly as it was requested.
    """
    normalise = nfc if match_nfc else str
    remaining = set(needed)
    found: dict[tuple[str, int], str] = {}
    needed_by_page: dict[str, dict[int, set[str]]] = {}
    for page, sentence_id in needed:
        needed_by_page.setdefault(normalise(page), {}).setdefault(sentence_id, set()).add(page)
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
            key = normalise(page)
            page_needs = needed_by_page.get(key)
            if not page_needs:
                continue
            sentences = parse_wiki_lines(article.get("lines", ""))
            for sentence_id, requested_pages in page_needs.items():
                text = sentences.get(sentence_id)
                if text:
                    for requested in requested_pages:
                        found[(requested, sentence_id)] = text
                        remaining.discard((requested, sentence_id))
            needed_by_page.pop(key)
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


def _prepare_whole(
    records: list[dict[str, Any]], wiki_files: list[Path], output_path: Path
) -> list[dict[str, Any]]:
    """Every SUPPORTS/REFUTES claim, as decided in the FEVER data audit.

    Nothing is de-duplicated or removed because it was flagged; flagged claims stay in and carry
    their flags so a second result without them can be reported. Page titles are matched in NFC
    form. A claim whose evidence still cannot be found is dropped and reported, never guessed.
    """
    flags = audit_flags(records)
    candidates = [
        candidate
        for label in TARGET_LABELS
        for candidate in candidate_order(records, label, seed=0, dedupe=False)
    ]
    needed = set().union(*(_references(candidate) for candidate in candidates))
    found, unresolved = scan_wikipedia(wiki_files, needed, match_nfc=True)
    usable = [c for c in candidates if _references(c) <= found.keys()]
    if len(usable) != len(candidates):
        print(
            f"Dropped {len(candidates) - len(usable)} claims whose evidence could not be found: "
            f"{sorted(unresolved)[:10]!r}",
            file=sys.stderr,
        )
    final = sorted(
        (
            {**_attach_evidence(candidate, found), "flags": flags.get(candidate["fever_id"], [])}
            for candidate in usable
        ),
        key=lambda item: item["fever_id"],
    )
    write_jsonl(output_path, final)
    return final


def prepare_dataset(
    dev_path: Path,
    wiki_dir: Path,
    output_path: Path,
    per_label: int = 75,
    seed: int = 42,
    *,
    whole: bool = False,
) -> list[dict[str, Any]]:
    """Build a balanced sample, replacing unresolved cases deterministically.

    ``whole=True`` instead returns every SUPPORTS/REFUTES claim (see ``_prepare_whole``);
    ``per_label`` and ``seed`` are then unused.
    """
    records = list(read_jsonl(dev_path))
    wiki_files = list(wiki_dir.glob("wiki-*.jsonl"))
    if not wiki_files:
        raise FileNotFoundError(f"No wiki-*.jsonl files found in {wiki_dir}")
    if whole:
        return _prepare_whole(records, wiki_files, output_path)
    orders = {label: candidate_order(records, label, seed) for label in TARGET_LABELS}
    selected = {label: candidates[:per_label] for label, candidates in orders.items()}
    next_index = {label: per_label for label in TARGET_LABELS}
    resolved: dict[tuple[str, int], str] = {}

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
