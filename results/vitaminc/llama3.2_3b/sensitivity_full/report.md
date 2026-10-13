# VitaminC contrastive sensitivity experiment

This experiment tests whether `llama3.2:3b` changes its verdict correctly when the same claim receives SUPPORTS versus REFUTES evidence from an explicit VitaminC case group.

## Setup

- Test SHA-256: `7ad1808dbc30c62e0a1427a53022d0dfaff668a1fde3c4b612a2d266edd753ad`
- Selected pairs: 20232 (every valid contrastive pair in the test set; 18352 distinct claims, a claim that occurs in several pairs is kept in each)
- Model temperature / seed: 0 / 42
- Pairing: exact `case_id` plus exact claim, with one SUPPORTS and one REFUTES row

## Primary result

- Correct sensitivity: 6246/20232 (30.872%)
- Verdict flips: 13174/20232 (65.115%)
- Correct flips: 6246
- Incorrect flips: 6928

## Pair outcomes

- Correct flip: 6246
- No change, correct on SUPPORT side: 1813
- No change, correct on REFUTE side: 3696
- NEI involved: 8198
- Wrong direction: 279
- Other error: 0

## Individual accuracy

- SUPPORT side: 11698/20232 (57.819%)
- REFUTE side: 10729/20232 (53.030%)
- Overall rows: 22427/40464 (55.425%)

## NEI behavior

- NEI on SUPPORT evidence: 2397
- NEI on REFUTE evidence: 7350
- Pairs with NEI on either side: 8198
- Pairs with NEI on both sides: 1549

Full per-pair predictions and technical validity are retained in the JSONL, CSV, and metrics artifacts.
