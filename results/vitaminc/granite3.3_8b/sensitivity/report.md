# VitaminC contrastive sensitivity experiment

This experiment tests whether `granite3.3:8b` changes its verdict correctly when the same claim receives SUPPORTS versus REFUTES evidence from an explicit VitaminC case group.

## Setup

- Test SHA-256: `7ad1808dbc30c62e0a1427a53022d0dfaff668a1fde3c4b612a2d266edd753ad`
- Selected pairs: 100
- Model temperature / seed: 0 / 42
- Pairing: exact `case_id` plus exact claim, with one SUPPORTS and one REFUTES row

## Primary result

- Correct sensitivity: 44/100 (44.000%)
- Verdict flips: 54/97 (55.670%)
- Correct flips: 44
- Incorrect flips: 10

## Pair outcomes

- Correct flip: 44
- No change, correct on SUPPORT side: 14
- No change, correct on REFUTE side: 27
- NEI involved: 11
- Wrong direction: 1
- Other error: 3

## Individual accuracy

- SUPPORT side: 67/100 (67.000%)
- REFUTE side: 74/100 (74.000%)
- Overall rows: 141/200 (70.500%)

## NEI behavior

- NEI on SUPPORT evidence: 5
- NEI on REFUTE evidence: 8
- Pairs with NEI on either side: 11
- Pairs with NEI on both sides: 2

Full per-pair predictions and technical validity are retained in the JSONL, CSV, and metrics artifacts.
