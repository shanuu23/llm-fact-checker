# FEVER five-run consistency experiment

## Purpose

This experiment measures how consistently `qwen2.5:7b` returns the same verdict when given the exact same claim and gold evidence five times. Consistency does not imply correctness.

## Dataset and model

- Prepared dataset: `data/fever/processed/fever_baseline_150.jsonl`
- Verified SHA-256: `6bc7cf910cb0b31186e5fc397ebc7cefab2de1ee65d29ae601e369d8758344cb`
- Cases: 150 from the unchanged 75 SUPPORTED / 75 REFUTED dataset
- Model: `qwen2.5:7b`
- Execution: sequential requests through the local Ollama HTTP API

## Generation settings

- Temperature: 0.2
- Seeds: [101, 202, 303, 404, 505]
- Prompt and evidence formatting: identical to the baseline

Temperature 0.2 permits limited sampling variation, while fixed distinct seeds make the five repetitions controlled and reproducible. All other generation settings are identical.

## Primary stability results

- Exact 5/5 consistency: 147/150 (98.000%)
- Mean modal agreement: 99.865%
- Complete cases: 148
- Incomplete cases: 2

### Agreement distribution

| Maximum agreement | Cases |
|---|---:|
| 5/5 | 147 |
| 4/5 | 1 |
| 3/5 | 0 |
| 2/5 | 0 |

- One unique label: 147
- Two unique labels: 1
- Three unique labels: 0
- Modal ties: 0

## FEVER gold-label accuracy

| Run | Seed | Correct | FEVER accuracy |
|---:|---:|---:|---:|
| 1 | 101 | 135/150 | 90.000% |
| 2 | 202 | 136/150 | 90.667% |
| 3 | 303 | 136/150 | 90.667% |
| 4 | 404 | 136/150 | 90.667% |
| 5 | 505 | 137/150 | 91.333% |

- Mean: 90.667%
- Population standard deviation: 0.422%
- Minimum: 90.000%
- Maximum: 91.333%
- Non-tied consensus FEVER accuracy: 91.216% (135/148)
- Consensus ties: 0

FEVER accuracy means agreement with FEVER gold labels, not absolute real-world truth.

## Stable-correct versus stable-wrong

- Stable correct: 135 (90.000%)
- Stable wrong: 12 (8.000%)
- Unstable: 1 (0.667%)
- Incomplete: 2 (1.333%)

## NOT ENOUGH INFORMATION behaviour

- Cases producing NEI at least once: 1
- Cases producing NEI in all five runs: 1
- Cases switching between NEI and SUPPORTED/REFUTED: 0
- Total NEI predictions: 5 / 750

## Technical validity

- Expected inference attempts: 750
- Successful inference requests: 750
- Inference failures: 0
- Invalid-format outputs: 6

## Limitations

This experiment tests repeated-run stability only. It does not test prompt robustness, evidence perturbations, sensitivity to meaningful evidence changes, retrieval, or real-world fact-checking. Consistency does not imply correctness: a verdict can be repeated five times and still disagree with FEVER. FEVER also contains known annotation noise, so FEVER accuracy and repeatability are separate concepts.
