# FEVER five-run consistency experiment

## Purpose

This experiment measures how consistently `phi4-mini:3.8b` returns the same verdict when given the exact same claim and gold evidence five times. Consistency does not imply correctness.

## Dataset and model

- Prepared dataset: `data/fever/full/fever_dev_full.jsonl`
- Dataset SHA-256: `46d03c136a5774811e00a92c67c9ad84f78b1daad981bf17c7a07a74f5474882`
- Cases: 13332 from the prepared 6666 SUPPORTED / 6666 REFUTED dataset
- Model: `phi4-mini:3.8b`
- Execution: sequential requests through the local Ollama HTTP API

## Generation settings

- Temperature: 0.2
- Seeds: [101, 202, 303, 404, 505]
- Prompt and evidence formatting: identical to the baseline

Temperature 0.2 permits limited sampling variation, while fixed distinct seeds make the five repetitions controlled and reproducible. All other generation settings are identical.

## Primary stability results

- Exact 5/5 consistency: 12113/13332 (90.857%)
- Mean modal agreement: 97.986%
- Complete cases: 13307
- Incomplete cases: 25

### Agreement distribution

| Maximum agreement | Cases |
|---|---:|
| 5/5 | 12113 |
| 4/5 | 1052 |
| 3/5 | 138 |
| 2/5 | 4 |

- One unique label: 12113
- Two unique labels: 1161
- Three unique labels: 33
- Modal ties: 4

## FEVER gold-label accuracy

| Run | Seed | Correct | FEVER accuracy |
|---:|---:|---:|---:|
| 1 | 101 | 7979/13332 | 59.848% |
| 2 | 202 | 7981/13332 | 59.863% |
| 3 | 303 | 7981/13332 | 59.863% |
| 4 | 404 | 7981/13332 | 59.863% |
| 5 | 505 | 7966/13332 | 59.751% |

- Mean: 59.838%
- Population standard deviation: 0.044%
- Minimum: 59.751%
- Maximum: 59.863%
- Non-tied consensus FEVER accuracy: 59.926% (7972/13303)
- Consensus ties: 4

FEVER accuracy means agreement with FEVER gold labels, not absolute real-world truth.

## Stable-correct versus stable-wrong

- Stable correct: 7526 (56.451%)
- Stable wrong: 4587 (34.406%)
- Unstable: 1194 (8.956%)
- Incomplete: 25 (0.188%)

## NOT ENOUGH INFORMATION behaviour

- Cases producing NEI at least once: 3059
- Cases producing NEI in all five runs: 2076
- Cases switching between NEI and SUPPORTED/REFUTED: 973
- Total NEI predictions: 12843 / 66660

## Technical validity

- Expected inference attempts: 66660
- Successful inference requests: 66660
- Inference failures: 0
- Invalid-format outputs: 36

## Limitations

This experiment tests repeated-run stability only. It does not test prompt robustness, evidence perturbations, sensitivity to meaningful evidence changes, retrieval, or real-world fact-checking. Consistency does not imply correctness: a verdict can be repeated five times and still disagree with FEVER. FEVER also contains known annotation noise, so FEVER accuracy and repeatability are separate concepts.

## Result excluding flagged records

The FEVER data audit flagged 68 of these claims as possibly
inconsistent. The main result above keeps them; this section removes them, as the audit decided.
Flags are possible label inconsistencies, not proven errors.

| | All cases | Excluding flagged |
|---|---:|---:|
| Cases | 13332 | 13264 |
| Exact 5/5 consistency | 90.857% | 90.878% |
| Mean modal agreement | 97.986% | 97.991% |
| Mean FEVER accuracy across runs | 59.838% | 59.931% |
