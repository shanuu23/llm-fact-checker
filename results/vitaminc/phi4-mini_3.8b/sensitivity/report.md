# VitaminC contrastive sensitivity experiment

This experiment tests whether `phi4-mini:3.8b` changes its verdict correctly when the same claim receives SUPPORTS versus REFUTES evidence from an explicit VitaminC case group.

## Setup

- Test SHA-256: `7ad1808dbc30c62e0a1427a53022d0dfaff668a1fde3c4b612a2d266edd753ad`
- Selected pairs: 100
- Model temperature / seed: 0 / 42
- Pairing: exact `case_id` plus exact claim, with one SUPPORTS and one REFUTES row

## Primary result

- Correct sensitivity: 16/100 (16.000%)
- Verdict flips: 58/100 (58.000%)
- Correct flips: 16
- Incorrect flips: 42

## Pair outcomes

- Correct flip: 16
- No change, correct on SUPPORT side: 37
- No change, correct on REFUTE side: 0
- NEI involved: 47
- Wrong direction: 0
- Other error: 0

## Individual accuracy

- SUPPORT side: 93/100 (93.000%)
- REFUTE side: 17/100 (17.000%)
- Overall rows: 110/200 (55.000%)

## NEI behavior

- NEI on SUPPORT evidence: 7
- NEI on REFUTE evidence: 45
- Pairs with NEI on either side: 47
- Pairs with NEI on both sides: 5

Full per-pair predictions and technical validity are retained in the JSONL, CSV, and metrics artifacts.
