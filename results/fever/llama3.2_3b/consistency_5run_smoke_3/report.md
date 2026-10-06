# FEVER five-run consistency experiment

## Purpose

This experiment measures how consistently `llama3.2:3b` returns the same verdict when given the exact same claim and gold evidence five times. Consistency does not imply correctness.

## Dataset and model

- Prepared dataset: `data/fever/processed/fever_baseline_150.jsonl`
- Verified SHA-256: `6bc7cf910cb0b31186e5fc397ebc7cefab2de1ee65d29ae601e369d8758344cb`
- Cases: 3 from the unchanged 75 SUPPORTED / 75 REFUTED dataset
- Model: `llama3.2:3b`
- Execution: sequential requests through the local Ollama HTTP API

## Generation settings

- Temperature: 0.2
- Seeds: [101, 202, 303, 404, 505]
- Prompt and evidence formatting: identical to the baseline

Temperature 0.2 permits limited sampling variation, while fixed distinct seeds make the five repetitions controlled and reproducible. All other generation settings are identical.

## Primary stability results

- Exact 5/5 consistency: 2/3 (66.667%)
- Mean modal agreement: 93.333%
- Complete cases: 3
- Incomplete cases: 0

### Agreement distribution

| Maximum agreement | Cases |
|---|---:|
| 5/5 | 2 |
| 4/5 | 1 |
| 3/5 | 0 |
| 2/5 | 0 |

- One unique label: 2
- Two unique labels: 1
- Three unique labels: 0
- Modal ties: 0

## FEVER gold-label accuracy

| Run | Seed | Correct | FEVER accuracy |
|---:|---:|---:|---:|
| 1 | 101 | 0/3 | 0.000% |
| 2 | 202 | 0/3 | 0.000% |
| 3 | 303 | 0/3 | 0.000% |
| 4 | 404 | 0/3 | 0.000% |
| 5 | 505 | 1/3 | 33.333% |

- Mean: 6.667%
- Population standard deviation: 13.333%
- Minimum: 0.000%
- Maximum: 33.333%
- Non-tied consensus FEVER accuracy: 0.000% (0/3)
- Consensus ties: 0

FEVER accuracy means agreement with FEVER gold labels, not absolute real-world truth.

## Stable-correct versus stable-wrong

- Stable correct: 0 (0.000%)
- Stable wrong: 2 (66.667%)
- Unstable: 1 (33.333%)
- Incomplete: 0 (0.000%)

## NOT ENOUGH INFORMATION behaviour

- Cases producing NEI at least once: 1
- Cases producing NEI in all five runs: 0
- Cases switching between NEI and SUPPORTED/REFUTED: 1
- Total NEI predictions: 4 / 15

## Technical validity

- Expected inference attempts: 15
- Successful inference requests: 15
- Inference failures: 0
- Invalid-format outputs: 0

## Limitations

This experiment tests repeated-run stability only. It does not test prompt robustness, evidence perturbations, sensitivity to meaningful evidence changes, retrieval, or real-world fact-checking. Consistency does not imply correctness: a verdict can be repeated five times and still disagree with FEVER. FEVER also contains known annotation noise, so FEVER accuracy and repeatability are separate concepts.
