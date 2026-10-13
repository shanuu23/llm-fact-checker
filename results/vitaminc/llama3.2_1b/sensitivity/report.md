# VitaminC contrastive sensitivity experiment

This experiment tests whether `llama3.2:1b` changes its verdict correctly when the same claim receives SUPPORTS versus REFUTES evidence from an explicit VitaminC case group.

## Setup

- Test SHA-256: `7ad1808dbc30c62e0a1427a53022d0dfaff668a1fde3c4b612a2d266edd753ad`
- Selected pairs: 100
- Model temperature / seed: 0 / 42
- Pairing: exact `case_id` plus exact claim, with one SUPPORTS and one REFUTES row

## Primary result

- Correct sensitivity: 1/100 (1.000%)
- Verdict flips: 5/100 (5.000%)
- Correct flips: 1
- Incorrect flips: 4

## Pair outcomes

- Correct flip: 1
- No change, correct on SUPPORT side: 85
- No change, correct on REFUTE side: 0
- NEI involved: 14
- Wrong direction: 0
- Other error: 0

## Individual accuracy

- SUPPORT side: 90/100 (90.000%)
- REFUTE side: 1/100 (1.000%)
- Overall rows: 91/200 (45.500%)

## NEI behavior

- NEI on SUPPORT evidence: 10
- NEI on REFUTE evidence: 14
- Pairs with NEI on either side: 14
- Pairs with NEI on both sides: 10

Full per-pair predictions and technical validity are retained in the JSONL, CSV, and metrics artifacts.
