# VitaminC contrastive sensitivity experiment

This experiment tests whether `qwen2.5:7b` changes its verdict correctly when the same claim receives SUPPORTS versus REFUTES evidence from an explicit VitaminC case group.

## Setup

- Test SHA-256: `7ad1808dbc30c62e0a1427a53022d0dfaff668a1fde3c4b612a2d266edd753ad`
- Selected pairs: 100
- Model temperature / seed: 0 / 42
- Pairing: exact `case_id` plus exact claim, with one SUPPORTS and one REFUTES row

## Primary result

- Correct sensitivity: 72/100 (72.000%)
- Verdict flips: 76/98 (77.551%)
- Correct flips: 72
- Incorrect flips: 4

## Pair outcomes

- Correct flip: 72
- No change, correct on SUPPORT side: 12
- No change, correct on REFUTE side: 10
- NEI involved: 3
- Wrong direction: 1
- Other error: 2

## Individual accuracy

- SUPPORT side: 88/100 (88.000%)
- REFUTE side: 83/100 (83.000%)
- Overall rows: 171/200 (85.500%)

## NEI behavior

- NEI on SUPPORT evidence: 1
- NEI on REFUTE evidence: 2
- Pairs with NEI on either side: 3
- Pairs with NEI on both sides: 0

Full per-pair predictions and technical validity are retained in the JSONL, CSV, and metrics artifacts.
