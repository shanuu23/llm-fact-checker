# FEVER five-run consistency experiment

## Purpose

This experiment measures how consistently `llama3.2:1b` returns the same verdict when given the exact same claim and gold evidence five times. Consistency does not imply correctness.

## Dataset and model

- Prepared dataset: `data/fever/full/fever_dev_full.jsonl`
- Dataset SHA-256: `46d03c136a5774811e00a92c67c9ad84f78b1daad981bf17c7a07a74f5474882`
- Cases: 13332 from the prepared 6666 SUPPORTED / 6666 REFUTED dataset
- Model: `llama3.2:1b`
- Execution: sequential requests through the local Ollama HTTP API

## Generation settings

- Temperature: 0.2
- Seeds: [101, 202, 303, 404, 505]
- Prompt and evidence formatting: identical to the baseline

Temperature 0.2 permits limited sampling variation, while fixed distinct seeds make the five repetitions controlled and reproducible. All other generation settings are identical.

## Primary stability results

- Exact 5/5 consistency: 10018/13332 (75.143%)
- Mean modal agreement: 94.280%
- Complete cases: 13325
- Incomplete cases: 7

### Agreement distribution

| Maximum agreement | Cases |
|---|---:|
| 5/5 | 10018 |
| 4/5 | 2837 |
| 3/5 | 436 |
| 2/5 | 34 |

- One unique label: 10018
- Two unique labels: 3116
- Three unique labels: 191
- Modal ties: 34

## FEVER gold-label accuracy

| Run | Seed | Correct | FEVER accuracy |
|---:|---:|---:|---:|
| 1 | 101 | 6460/13332 | 48.455% |
| 2 | 202 | 6465/13332 | 48.492% |
| 3 | 303 | 6465/13332 | 48.492% |
| 4 | 404 | 6466/13332 | 48.500% |
| 5 | 505 | 6268/13332 | 47.015% |

- Mean: 48.191%
- Population standard deviation: 0.588%
- Minimum: 47.015%
- Maximum: 48.500%
- Non-tied consensus FEVER accuracy: 48.559% (6454/13291)
- Consensus ties: 34

FEVER accuracy means agreement with FEVER gold labels, not absolute real-world truth.

## Stable-correct versus stable-wrong

- Stable correct: 5648 (42.364%)
- Stable wrong: 4370 (32.778%)
- Unstable: 3307 (24.805%)
- Incomplete: 7 (0.053%)

## NOT ENOUGH INFORMATION behaviour

- Cases producing NEI at least once: 3312
- Cases producing NEI in all five runs: 414
- Cases switching between NEI and SUPPORTED/REFUTED: 2898
- Total NEI predictions: 7951 / 66660

## Technical validity

- Expected inference attempts: 66660
- Successful inference requests: 66660
- Inference failures: 0
- Invalid-format outputs: 14

## Limitations

This experiment tests repeated-run stability only. It does not test prompt robustness, evidence perturbations, sensitivity to meaningful evidence changes, retrieval, or real-world fact-checking. Consistency does not imply correctness: a verdict can be repeated five times and still disagree with FEVER. FEVER also contains known annotation noise, so FEVER accuracy and repeatability are separate concepts.

## Result excluding flagged records

The FEVER data audit flagged 68 of these claims as possibly
inconsistent. The main result above keeps them; this section removes them, as the audit decided.
Flags are possible label inconsistencies, not proven errors.

| | All cases | Excluding flagged |
|---|---:|---:|
| Cases | 13332 | 13264 |
| Exact 5/5 consistency | 75.143% | 75.136% |
| Mean modal agreement | 94.280% | 94.275% |
| Mean FEVER accuracy across runs | 48.191% | 48.304% |
