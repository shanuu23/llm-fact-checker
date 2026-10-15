# VitaminC contrastive sensitivity experiment

This experiment tests whether `gemma3:4b` changes its verdict correctly when the same claim receives SUPPORTS versus REFUTES evidence from an explicit VitaminC case group.

## Setup

- Test SHA-256: `7ad1808dbc30c62e0a1427a53022d0dfaff668a1fde3c4b612a2d266edd753ad`
- Selected pairs: 20232 (every valid contrastive pair in the test set; 18352 distinct claims, a claim that occurs in several pairs is kept in each)
- Model temperature / seed: 0 / 42
- Pairing: exact `case_id` plus exact claim, with one SUPPORTS and one REFUTES row

## Primary result

- Correct sensitivity: 9877/20232 (48.819%)
- Verdict flips: 12439/20232 (61.482%)
- Correct flips: 9877
- Incorrect flips: 2562

## Pair outcomes

- Correct flip: 9877
- No change, correct on SUPPORT side: 4969
- No change, correct on REFUTE side: 2712
- NEI involved: 2216
- Wrong direction: 458
- Other error: 0

## Individual accuracy

- SUPPORT side: 16547/20232 (81.786%)
- REFUTE side: 12691/20232 (62.727%)
- Overall rows: 29238/40464 (72.257%)

## NEI behavior

- NEI on SUPPORT evidence: 248
- NEI on REFUTE evidence: 2080
- Pairs with NEI on either side: 2216
- Pairs with NEI on both sides: 112

Full per-pair predictions and technical validity are retained in the JSONL, CSV, and metrics artifacts.
