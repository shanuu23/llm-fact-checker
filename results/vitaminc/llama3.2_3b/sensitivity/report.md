# VitaminC contrastive sensitivity experiment

This experiment tests whether `llama3.2:3b` changes its verdict correctly when the same claim receives SUPPORTS versus REFUTES evidence from an explicit VitaminC case group.

## Setup

- Test SHA-256: `7f799e076dec184e31533d46704fef2022d08ef68a0db2c44f138d2b4e552f36`
- Selected pairs: 100
- Model temperature / seed: 0 / 42
- Pairing: exact `case_id` plus exact claim, with one SUPPORTS and one REFUTES row

## Primary result

- Correct sensitivity: 27/100 (27.000%)
- Verdict flips: 72/100 (72.000%)
- Correct flips: 27
- Incorrect flips: 45

## Pair outcomes

- Correct flip: 27
- No change, correct on SUPPORT side: 8
- No change, correct on REFUTE side: 11
- NEI involved: 52
- Wrong direction: 2
- Other error: 0

## Individual accuracy

- SUPPORT side: 66/100 (66.000%)
- REFUTE side: 42/100 (42.000%)
- Overall rows: 108/200 (54.000%)

## NEI behavior

- NEI on SUPPORT evidence: 15
- NEI on REFUTE evidence: 46
- Pairs with NEI on either side: 52
- Pairs with NEI on both sides: 9

Full per-pair predictions and technical validity are retained in the JSONL, CSV, and metrics artifacts.
