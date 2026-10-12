"""Synthetic FEVER dev records and Wikipedia pages for tests."""

from __future__ import annotations

import json
from pathlib import Path


def fever_record(fever_id: int, label: str, claim: str, evidence_sets=None) -> dict:
    """A FEVER dev record; ``evidence_sets`` is a list of lists of (page, sentence_id)."""
    if label == "NOT ENOUGH INFO" or evidence_sets is None:
        evidence = [[[fever_id * 10, None, None, None]]]
    else:
        evidence = [
            [
                [fever_id * 10 + i, fever_id * 100 + i, page, sentence_id]
                for i, (page, sentence_id) in enumerate(group)
            ]
            for group in evidence_sets
        ]
    return {
        "id": fever_id,
        "label": label,
        "claim": claim,
        "verifiable": "VERIFIABLE",
        "evidence": evidence,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def wiki_page(page: str, sentences: dict[int, str]) -> dict:
    lines = "\n".join(f"{index}\t{text}\t" for index, text in sorted(sentences.items()))
    return {"id": page, "text": " ".join(sentences.values()), "lines": lines}
