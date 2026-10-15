# VitaminC contrastive sensitivity experiment

This experiment tests whether `gemma3:4b` changes its verdict correctly when the same claim receives SUPPORTS versus REFUTES evidence from an explicit VitaminC case group.

## Setup

- Test SHA-256: `7ad1808dbc30c62e0a1427a53022d0dfaff668a1fde3c4b612a2d266edd753ad`
- Selected pairs: 100
- Model temperature / seed: 0 / 42
- Pairing: exact `case_id` plus exact claim, with one SUPPORTS and one REFUTES row

## Primary result

- Correct sensitivity: 53/100 (53.000%)
- Verdict flips: 64/100 (64.000%)
- Correct flips: 53
- Incorrect flips: 11

## Pair outcomes

- Correct flip: 53
- No change, correct on SUPPORT side: 28
- No change, correct on REFUTE side: 8
- NEI involved: 8
- Wrong direction: 3
- Other error: 0

## Individual accuracy

- SUPPORT side: 88/100 (88.000%)
- REFUTE side: 61/100 (61.000%)
- Overall rows: 149/200 (74.500%)

## NEI behavior

- NEI on SUPPORT evidence: 0
- NEI on REFUTE evidence: 8
- Pairs with NEI on either side: 8
- Pairs with NEI on both sides: 0

Full per-pair predictions and technical validity are retained in the JSONL, CSV, and metrics artifacts.
