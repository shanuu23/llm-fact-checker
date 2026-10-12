"""Model identifiers: the default model and the folder-safe slug used in result paths."""

from __future__ import annotations

import re

DEFAULT_MODEL = "llama3.2:3b"

_SAFE_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def model_slug(tag: str) -> str:
    """Folder-safe name for an Ollama model tag, for example ``qwen2.5:7b`` -> ``qwen2.5_7b``.

    ``:`` and ``/`` become ``_``. The exact tag is always kept in ``run_config.json``, so the
    slug only has to be stable and safe to use as a directory name.
    """
    slug = tag.strip().replace(":", "_").replace("/", "_")
    if not _SAFE_SLUG.match(slug):
        raise ValueError(f"Cannot make a folder name from model tag {tag!r}")
    return slug
