# FEVER five-run consistency experiment

## Purpose

This experiment measures how consistently `gemma3:4b` returns the same verdict when given the exact same claim and gold evidence five times. Consistency does not imply correctness.

## Dataset and model

- Prepared dataset: `data/fever/processed/fever_baseline_150.jsonl`
- Verified SHA-256: `6bc7cf910cb0b31186e5fc397ebc7cefab2de1ee65d29ae601e369d8758344cb`
- Cases: 150 from the unchanged 75 SUPPORTED / 75 REFUTED dataset
- Model: `gemma3:4b`
- Execution: sequential requests through the local Ollama HTTP API

## Generation settings

- Temperature: 0.2
- Seeds: [101, 202, 303, 404, 505]
- Prompt and evidence formatting: identical to the baseline

Temperature 0.2 permits limited sampling variation, while fixed distinct seeds make the five repetitions controlled and reproducible. All other generation settings are identical.

## Primary stability results

- Exact 5/5 consistency: 150/150 (100.000%)
- Mean modal agreement: 100.000%
- Complete cases: 150
- Incomplete cases: 0

### Agreement distribution

| Maximum agreement | Cases |
|---|---:|
| 5/5 | 150 |
| 4/5 | 0 |
| 3/5 | 0 |
| 2/5 | 0 |

- One unique label: 150
- Two unique labels: 0
- Three unique labels: 0
- Modal ties: 0

## FEVER gold-label accuracy

| Run | Seed | Correct | FEVER accuracy |
|---:|---:|---:|---:|
| 1 | 101 | 121/150 | 80.667% |
| 2 | 202 | 121/150 | 80.667% |
| 3 | 303 | 121/150 | 80.667% |
| 4 | 404 | 121/150 | 80.667% |
| 5 | 505 | 121/150 | 80.667% |

- Mean: 80.667%
- Population standard deviation: 0.000%
- Minimum: 80.667%
- Maximum: 80.667%
- Non-tied consensus FEVER accuracy: 80.667% (121/150)
- Consensus ties: 0

FEVER accuracy means agreement with FEVER gold labels, not absolute real-world truth.

## Stable-correct versus stable-wrong

- Stable correct: 121 (80.667%)
- Stable wrong: 29 (19.333%)
- Unstable: 0 (0.000%)
- Incomplete: 0 (0.000%)

## NOT ENOUGH INFORMATION behaviour

- Cases producing NEI at least once: 10
- Cases producing NEI in all five runs: 10
- Cases switching between NEI and SUPPORTED/REFUTED: 0
- Total NEI predictions: 50 / 750

## Technical validity

- Expected inference attempts: 750
- Successful inference requests: 750
- Inference failures: 0
- Invalid-format outputs: 0

## Limitations

This experiment tests repeated-run stability only. It does not test prompt robustness, evidence perturbations, sensitivity to meaningful evidence changes, retrieval, or real-world fact-checking. Consistency does not imply correctness: a verdict can be repeated five times and still disagree with FEVER. FEVER also contains known annotation noise, so FEVER accuracy and repeatability are separate concepts.
