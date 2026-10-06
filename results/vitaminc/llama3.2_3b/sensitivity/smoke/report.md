# VitaminC contrastive sensitivity experiment

This experiment tests whether `llama3.2:3b` changes its verdict correctly when the same claim receives SUPPORTS versus REFUTES evidence from an explicit VitaminC case group.

## Setup

- Test SHA-256: `7f799e076dec184e31533d46704fef2022d08ef68a0db2c44f138d2b4e552f36`
- Selected pairs: 3
- Model temperature / seed: 0 / 42
- Pairing: exact `case_id` plus exact claim, with one SUPPORTS and one REFUTES row

## Primary result

- Correct sensitivity: 0/3 (0.000%)
- Verdict flips: 1/3 (33.333%)
- Correct flips: 0
- Incorrect flips: 1

## Pair outcomes

- Correct flip: 0
- No change, correct on SUPPORT side: 0
- No change, correct on REFUTE side: 1
- NEI involved: 2
- Wrong direction: 0
- Other error: 0

## Individual accuracy

- SUPPORT side: 0/3 (0.000%)
- REFUTE side: 1/3 (33.333%)
- Overall rows: 1/6 (16.667%)

## NEI behavior

- NEI on SUPPORT evidence: 2
- NEI on REFUTE evidence: 1
- Pairs with NEI on either side: 2
- Pairs with NEI on both sides: 1

Full per-pair predictions and technical validity are retained in the JSONL, CSV, and metrics artifacts.
