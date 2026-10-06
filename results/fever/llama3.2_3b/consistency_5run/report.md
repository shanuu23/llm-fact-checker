# FEVER five-run consistency experiment

## Purpose

This experiment measures how consistently `llama3.2:3b` returns the same verdict when given the exact same claim and gold evidence five times. Consistency does not imply correctness.

## Dataset and model

- Prepared dataset: `data/fever/processed/fever_baseline_150.jsonl`
- Verified SHA-256: `6bc7cf910cb0b31186e5fc397ebc7cefab2de1ee65d29ae601e369d8758344cb`
- Cases: 150 from the unchanged 75 SUPPORTED / 75 REFUTED dataset
- Model: `llama3.2:3b`
- Execution: sequential requests through the local Ollama HTTP API

## Generation settings

- Temperature: 0.2
- Seeds: [101, 202, 303, 404, 505]
- Prompt and evidence formatting: identical to the baseline

Temperature 0.2 permits limited sampling variation, while fixed distinct seeds make the five repetitions controlled and reproducible. All other generation settings are identical.

## Primary stability results

- Exact 5/5 consistency: 120/150 (80.000%)
- Mean modal agreement: 95.600%
- Complete cases: 150
- Incomplete cases: 0

### Agreement distribution

| Maximum agreement | Cases |
|---|---:|
| 5/5 | 120 |
| 4/5 | 27 |
| 3/5 | 3 |
| 2/5 | 0 |

- One unique label: 120
- Two unique labels: 30
- Three unique labels: 0
- Modal ties: 0

## FEVER gold-label accuracy

| Run | Seed | Correct | FEVER accuracy |
|---:|---:|---:|---:|
| 1 | 101 | 98/150 | 65.333% |
| 2 | 202 | 97/150 | 64.667% |
| 3 | 303 | 97/150 | 64.667% |
| 4 | 404 | 97/150 | 64.667% |
| 5 | 505 | 99/150 | 66.000% |

- Mean: 65.067%
- Population standard deviation: 0.533%
- Minimum: 64.667%
- Maximum: 66.000%
- Non-tied consensus FEVER accuracy: 64.667% (97/150)
- Consensus ties: 0

FEVER accuracy means agreement with FEVER gold labels, not absolute real-world truth.

## Stable-correct versus stable-wrong

- Stable correct: 85 (56.667%)
- Stable wrong: 35 (23.333%)
- Unstable: 30 (20.000%)
- Incomplete: 0 (0.000%)

## NOT ENOUGH INFORMATION behaviour

- Cases producing NEI at least once: 46
- Cases producing NEI in all five runs: 21
- Cases switching between NEI and SUPPORTED/REFUTED: 25
- Total NEI predictions: 171 / 750

## Technical validity

- Expected inference attempts: 750
- Successful inference requests: 750
- Inference failures: 0
- Invalid-format outputs: 0

## Limitations

This experiment tests repeated-run stability only. It does not test prompt robustness, evidence perturbations, sensitivity to meaningful evidence changes, retrieval, or real-world fact-checking. Consistency does not imply correctness: a verdict can be repeated five times and still disagree with FEVER. FEVER also contains known annotation noise, so FEVER accuracy and repeatability are separate concepts.
