# FEVER five-run consistency experiment

## Purpose

This experiment measures how consistently `granite3.3:8b` returns the same verdict when given the exact same claim and gold evidence five times. Consistency does not imply correctness.

## Dataset and model

- Prepared dataset: `data\fever\processed\fever_baseline_150.jsonl`
- Verified SHA-256: `6bc7cf910cb0b31186e5fc397ebc7cefab2de1ee65d29ae601e369d8758344cb`
- Cases: 150 from the unchanged 75 SUPPORTED / 75 REFUTED dataset
- Model: `granite3.3:8b`
- Execution: sequential requests through the local Ollama HTTP API

## Generation settings

- Temperature: 0.2
- Seeds: [101, 202, 303, 404, 505]
- Prompt and evidence formatting: identical to the baseline

Temperature 0.2 permits limited sampling variation, while fixed distinct seeds make the five repetitions controlled and reproducible. All other generation settings are identical.

## Primary stability results

- Exact 5/5 consistency: 145/150 (96.667%)
- Mean modal agreement: 99.728%
- Complete cases: 147
- Incomplete cases: 3

### Agreement distribution

| Maximum agreement | Cases |
|---|---:|
| 5/5 | 145 |
| 4/5 | 2 |
| 3/5 | 0 |
| 2/5 | 0 |

- One unique label: 145
- Two unique labels: 2
- Three unique labels: 0
- Modal ties: 0

## FEVER gold-label accuracy

| Run | Seed | Correct | FEVER accuracy |
|---:|---:|---:|---:|
| 1 | 101 | 131/150 | 87.333% |
| 2 | 202 | 131/150 | 87.333% |
| 3 | 303 | 131/150 | 87.333% |
| 4 | 404 | 131/150 | 87.333% |
| 5 | 505 | 133/150 | 88.667% |

- Mean: 87.600%
- Population standard deviation: 0.533%
- Minimum: 87.333%
- Maximum: 88.667%
- Non-tied consensus FEVER accuracy: 89.116% (131/147)
- Consensus ties: 0

FEVER accuracy means agreement with FEVER gold labels, not absolute real-world truth.

## Stable-correct versus stable-wrong

- Stable correct: 130 (86.667%)
- Stable wrong: 15 (10.000%)
- Unstable: 2 (1.333%)
- Incomplete: 3 (2.000%)

## NOT ENOUGH INFORMATION behaviour

- Cases producing NEI at least once: 5
- Cases producing NEI in all five runs: 3
- Cases switching between NEI and SUPPORTED/REFUTED: 1
- Total NEI predictions: 21 / 750

## Technical validity

- Expected inference attempts: 750
- Successful inference requests: 750
- Inference failures: 0
- Invalid-format outputs: 10

## Limitations

This experiment tests repeated-run stability only. It does not test prompt robustness, evidence perturbations, sensitivity to meaningful evidence changes, retrieval, or real-world fact-checking. Consistency does not imply correctness: a verdict can be repeated five times and still disagree with FEVER. FEVER also contains known annotation noise, so FEVER accuracy and repeatability are separate concepts.
