# FEVER five-run consistency experiment

## Purpose

This experiment measures how consistently `phi4-mini:3.8b` returns the same verdict when given the exact same claim and gold evidence five times. Consistency does not imply correctness.

## Dataset and model

- Prepared dataset: `data/fever/processed/fever_baseline_150.jsonl`
- Verified SHA-256: `6bc7cf910cb0b31186e5fc397ebc7cefab2de1ee65d29ae601e369d8758344cb`
- Cases: 150 from the unchanged 75 SUPPORTED / 75 REFUTED dataset
- Model: `phi4-mini:3.8b`
- Execution: sequential requests through the local Ollama HTTP API

## Generation settings

- Temperature: 0.2
- Seeds: [101, 202, 303, 404, 505]
- Prompt and evidence formatting: identical to the baseline

Temperature 0.2 permits limited sampling variation, while fixed distinct seeds make the five repetitions controlled and reproducible. All other generation settings are identical.

## Primary stability results

- Exact 5/5 consistency: 135/150 (90.000%)
- Mean modal agreement: 97.867%
- Complete cases: 150
- Incomplete cases: 0

### Agreement distribution

| Maximum agreement | Cases |
|---|---:|
| 5/5 | 135 |
| 4/5 | 14 |
| 3/5 | 1 |
| 2/5 | 0 |

- One unique label: 135
- Two unique labels: 15
- Three unique labels: 0
- Modal ties: 0

## FEVER gold-label accuracy

| Run | Seed | Correct | FEVER accuracy |
|---:|---:|---:|---:|
| 1 | 101 | 87/150 | 58.000% |
| 2 | 202 | 86/150 | 57.333% |
| 3 | 303 | 86/150 | 57.333% |
| 4 | 404 | 86/150 | 57.333% |
| 5 | 505 | 88/150 | 58.667% |

- Mean: 57.733%
- Population standard deviation: 0.533%
- Minimum: 57.333%
- Maximum: 58.667%
- Non-tied consensus FEVER accuracy: 57.333% (86/150)
- Consensus ties: 0

FEVER accuracy means agreement with FEVER gold labels, not absolute real-world truth.

## Stable-correct versus stable-wrong

- Stable correct: 81 (54.000%)
- Stable wrong: 54 (36.000%)
- Unstable: 15 (10.000%)
- Incomplete: 0 (0.000%)

## NOT ENOUGH INFORMATION behaviour

- Cases producing NEI at least once: 44
- Cases producing NEI in all five runs: 29
- Cases switching between NEI and SUPPORTED/REFUTED: 15
- Total NEI predictions: 186 / 750

## Technical validity

- Expected inference attempts: 750
- Successful inference requests: 750
- Inference failures: 0
- Invalid-format outputs: 0

## Limitations

This experiment tests repeated-run stability only. It does not test prompt robustness, evidence perturbations, sensitivity to meaningful evidence changes, retrieval, or real-world fact-checking. Consistency does not imply correctness: a verdict can be repeated five times and still disagree with FEVER. FEVER also contains known annotation noise, so FEVER accuracy and repeatability are separate concepts.
