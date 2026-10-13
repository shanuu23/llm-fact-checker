# FEVER five-run consistency experiment

## Purpose

This experiment measures how consistently `llama3.2:1b` returns the same verdict when given the exact same claim and gold evidence five times. Consistency does not imply correctness.

## Dataset and model

- Prepared dataset: `data/fever/processed/fever_baseline_150.jsonl`
- Verified SHA-256: `6bc7cf910cb0b31186e5fc397ebc7cefab2de1ee65d29ae601e369d8758344cb`
- Cases: 150 from the unchanged 75 SUPPORTED / 75 REFUTED dataset
- Model: `llama3.2:1b`
- Execution: sequential requests through the local Ollama HTTP API

## Generation settings

- Temperature: 0.2
- Seeds: [101, 202, 303, 404, 505]
- Prompt and evidence formatting: identical to the baseline

Temperature 0.2 permits limited sampling variation, while fixed distinct seeds make the five repetitions controlled and reproducible. All other generation settings are identical.

## Primary stability results

- Exact 5/5 consistency: 106/150 (70.667%)
- Mean modal agreement: 93.733%
- Complete cases: 150
- Incomplete cases: 0

### Agreement distribution

| Maximum agreement | Cases |
|---|---:|
| 5/5 | 106 |
| 4/5 | 41 |
| 3/5 | 3 |
| 2/5 | 0 |

- One unique label: 106
- Two unique labels: 43
- Three unique labels: 1
- Modal ties: 0

## FEVER gold-label accuracy

| Run | Seed | Correct | FEVER accuracy |
|---:|---:|---:|---:|
| 1 | 101 | 74/150 | 49.333% |
| 2 | 202 | 76/150 | 50.667% |
| 3 | 303 | 76/150 | 50.667% |
| 4 | 404 | 76/150 | 50.667% |
| 5 | 505 | 66/150 | 44.000% |

- Mean: 49.067%
- Population standard deviation: 2.585%
- Minimum: 44.000%
- Maximum: 50.667%
- Non-tied consensus FEVER accuracy: 50.667% (76/150)
- Consensus ties: 0

FEVER accuracy means agreement with FEVER gold labels, not absolute real-world truth.

## Stable-correct versus stable-wrong

- Stable correct: 62 (41.333%)
- Stable wrong: 44 (29.333%)
- Unstable: 44 (29.333%)
- Incomplete: 0 (0.000%)

## NOT ENOUGH INFORMATION behaviour

- Cases producing NEI at least once: 44
- Cases producing NEI in all five runs: 6
- Cases switching between NEI and SUPPORTED/REFUTED: 38
- Total NEI predictions: 97 / 750

## Technical validity

- Expected inference attempts: 750
- Successful inference requests: 750
- Inference failures: 0
- Invalid-format outputs: 0

## Limitations

This experiment tests repeated-run stability only. It does not test prompt robustness, evidence perturbations, sensitivity to meaningful evidence changes, retrieval, or real-world fact-checking. Consistency does not imply correctness: a verdict can be repeated five times and still disagree with FEVER. FEVER also contains known annotation noise, so FEVER accuracy and repeatability are separate concepts.
