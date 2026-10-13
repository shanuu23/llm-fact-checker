"""The frozen prompt, label set, and strict output parser.

Every experiment and every model must use this prompt unchanged so results stay
comparable. Changing anything here invalidates comparison with existing results and
needs a team decision.
"""

from __future__ import annotations

from typing import Any

# The three answers the model may give, in the order reports list them.
LABELS = ("SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION")
ALLOWED_LABELS = set(LABELS)
SYSTEM_PROMPT = """You are an evidence-grounded fact verifier.

Determine whether the provided evidence supports, refutes, or does not provide enough information to determine the claim.

Use only the supplied evidence. Do not rely on outside knowledge.

Return exactly one of:

SUPPORTED
REFUTED
NOT ENOUGH INFORMATION

Return the label only. Do not provide an explanation."""


def parse_prediction(raw_output: str) -> str | None:
    normalized = raw_output.strip().upper()
    return normalized if normalized in ALLOWED_LABELS else None


def build_user_prompt(claim: str, evidence: list[dict[str, Any]]) -> str:
    evidence_text = "\n\n".join(
        f"[{index}] Page: {item['page']} | Sentence: {item['sentence_id']}\n{item['text']}"
        for index, item in enumerate(evidence, 1)
    )
    return f"Claim:\n{claim}\n\nEvidence:\n{evidence_text}"
