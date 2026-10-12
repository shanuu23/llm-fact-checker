"""Small helpers shared by the data-preparation and experiment modules."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path} at line {line_number}") from error


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write records so that a crash cannot leave a half-written file at ``path``.

    The rows go to a temporary file that replaces ``path`` only once complete.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_text_atomic(path: Path, text: str) -> None:
    """Write a text file so that a crash cannot leave it half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str] | None = None
) -> None:
    """Write dict rows as CSV; the header defaults to the first row's keys."""
    fields = list(fieldnames) if fieldnames is not None else list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def percentage(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.3%}" if signed else f"{value:.3%}"
