# VitaminC contrastive sensitivity experiment

This experiment tests whether `phi4-mini:3.8b` changes its verdict correctly when the same claim receives SUPPORTS versus REFUTES evidence from an explicit VitaminC case group.

## Setup

- Test SHA-256: `7ad1808dbc30c62e0a1427a53022d0dfaff668a1fde3c4b612a2d266edd753ad`
- Selected pairs: 3
- Model temperature / seed: 0 / 42
- Pairing: exact `case_id` plus exact claim, with one SUPPORTS and one REFUTES row

## Primary result

- Correct sensitivity: 0/3 (0.000%)
- Verdict flips: 2/3 (66.667%)
- Correct flips: 0
- Incorrect flips: 2

## Pair outcomes

- Correct flip: 0
- No change, correct on SUPPORT side: 1
- No change, correct on REFUTE side: 0
- NEI involved: 2
- Wrong direction: 0
- Other error: 0

## Individual accuracy

- SUPPORT side: 3/3 (100.000%)
- REFUTE side: 0/3 (0.000%)
- Overall rows: 3/6 (50.000%)

## NEI behavior

- NEI on SUPPORT evidence: 0
- NEI on REFUTE evidence: 2
- Pairs with NEI on either side: 2
- Pairs with NEI on both sides: 0

Full per-pair predictions and technical validity are retained in the JSONL, CSV, and metrics artifacts.
