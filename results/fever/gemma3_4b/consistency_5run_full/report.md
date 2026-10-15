# FEVER five-run consistency experiment

## Purpose

This experiment measures how consistently `gemma3:4b` returns the same verdict when given the exact same claim and gold evidence five times. Consistency does not imply correctness.

## Dataset and model

- Prepared dataset: `data/fever/full/fever_dev_full.jsonl`
- Dataset SHA-256: `46d03c136a5774811e00a92c67c9ad84f78b1daad981bf17c7a07a74f5474882`
- Cases: 13332 from the prepared 6666 SUPPORTED / 6666 REFUTED dataset
- Model: `gemma3:4b`
- Execution: sequential requests through the local Ollama HTTP API

## Generation settings

- Temperature: 0.2
- Seeds: [101, 202, 303, 404, 505]
- Prompt and evidence formatting: identical to the baseline

Temperature 0.2 permits limited sampling variation, while fixed distinct seeds make the five repetitions controlled and reproducible. All other generation settings are identical.

## Primary stability results

- Exact 5/5 consistency: 13275/13332 (99.572%)
- Mean modal agreement: 99.907%
- Complete cases: 13332
- Incomplete cases: 0

### Agreement distribution

| Maximum agreement | Cases |
|---|---:|
| 5/5 | 13275 |
| 4/5 | 52 |
| 3/5 | 5 |
| 2/5 | 0 |

- One unique label: 13275
- Two unique labels: 57
- Three unique labels: 0
- Modal ties: 0

## FEVER gold-label accuracy

| Run | Seed | Correct | FEVER accuracy |
|---:|---:|---:|---:|
| 1 | 101 | 10453/13332 | 78.405% |
| 2 | 202 | 10454/13332 | 78.413% |
| 3 | 303 | 10454/13332 | 78.413% |
| 4 | 404 | 10454/13332 | 78.413% |
| 5 | 505 | 10453/13332 | 78.405% |

- Mean: 78.410%
- Population standard deviation: 0.004%
- Minimum: 78.405%
- Maximum: 78.413%
- Non-tied consensus FEVER accuracy: 78.413% (10454/13332)
- Consensus ties: 0

FEVER accuracy means agreement with FEVER gold labels, not absolute real-world truth.

## Stable-correct versus stable-wrong

- Stable correct: 10433 (78.255%)
- Stable wrong: 2842 (21.317%)
- Unstable: 57 (0.428%)
- Incomplete: 0 (0.000%)

## NOT ENOUGH INFORMATION behaviour

- Cases producing NEI at least once: 913
- Cases producing NEI in all five runs: 867
- Cases switching between NEI and SUPPORTED/REFUTED: 46
- Total NEI predictions: 4454 / 66660

## Technical validity

- Expected inference attempts: 66660
- Successful inference requests: 66660
- Inference failures: 0
- Invalid-format outputs: 0

## Limitations

This experiment tests repeated-run stability only. It does not test prompt robustness, evidence perturbations, sensitivity to meaningful evidence changes, retrieval, or real-world fact-checking. Consistency does not imply correctness: a verdict can be repeated five times and still disagree with FEVER. FEVER also contains known annotation noise, so FEVER accuracy and repeatability are separate concepts.

## Result excluding flagged records

The FEVER data audit flagged 68 of these claims as possibly
inconsistent. The main result above keeps them; this section removes them, as the audit decided.
Flags are possible label inconsistencies, not proven errors.

| | All cases | Excluding flagged |
|---|---:|---:|
| Cases | 13332 | 13264 |
| Exact 5/5 consistency | 99.572% | 99.585% |
| Mean modal agreement | 99.907% | 99.911% |
| Mean FEVER accuracy across runs | 78.410% | 78.468% |
