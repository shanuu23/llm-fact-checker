# VitaminC contrastive sensitivity experiment

This experiment tests whether `llama3.2:1b` changes its verdict correctly when the same claim receives SUPPORTS versus REFUTES evidence from an explicit VitaminC case group.

## Setup

- Test SHA-256: `7ad1808dbc30c62e0a1427a53022d0dfaff668a1fde3c4b612a2d266edd753ad`
- Selected pairs: 20232 (every valid contrastive pair in the test set; 18352 distinct claims, a claim that occurs in several pairs is kept in each)
- Model temperature / seed: 0 / 42
- Pairing: exact `case_id` plus exact claim, with one SUPPORTS and one REFUTES row

## Primary result

- Correct sensitivity: 89/20232 (0.440%)
- Verdict flips: 2687/20232 (13.281%)
- Correct flips: 89
- Incorrect flips: 2598

## Pair outcomes

- Correct flip: 89
- No change, correct on SUPPORT side: 15426
- No change, correct on REFUTE side: 53
- NEI involved: 4630
- Wrong direction: 34
- Other error: 0

## Individual accuracy

- SUPPORT side: 17397/20232 (85.988%)
- REFUTE side: 156/20232 (0.771%)
- Overall rows: 17553/40464 (43.379%)

## NEI behavior

- NEI on SUPPORT evidence: 2727
- NEI on REFUTE evidence: 3969
- Pairs with NEI on either side: 4630
- Pairs with NEI on both sides: 2066

Full per-pair predictions and technical validity are retained in the JSONL, CSV, and metrics artifacts.
