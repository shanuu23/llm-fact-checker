# FEVER five-run consistency experiment

## Purpose

This experiment measures how consistently `llama3.2:3b` returns the same verdict when given the exact same claim and gold evidence five times. Consistency does not imply correctness.

## Dataset and model

- Prepared dataset: `data/fever/full/fever_dev_full.jsonl`
- Dataset SHA-256: `46d03c136a5774811e00a92c67c9ad84f78b1daad981bf17c7a07a74f5474882`
- Cases: 13332 from the prepared 6666 SUPPORTED / 6666 REFUTED dataset
- Model: `llama3.2:3b`
- Execution: sequential requests through the local Ollama HTTP API

## Generation settings

- Temperature: 0.2
- Seeds: [101, 202, 303, 404, 505]
- Prompt and evidence formatting: identical to the baseline

Temperature 0.2 permits limited sampling variation, while fixed distinct seeds make the five repetitions controlled and reproducible. All other generation settings are identical.

## Primary stability results

- Exact 5/5 consistency: 10456/13332 (78.428%)
- Mean modal agreement: 95.197%
- Complete cases: 13332
- Incomplete cases: 0

### Agreement distribution

| Maximum agreement | Cases |
|---|---:|
| 5/5 | 10456 |
| 4/5 | 2550 |
| 3/5 | 326 |
| 2/5 | 0 |

- One unique label: 10456
- Two unique labels: 2849
- Three unique labels: 27
- Modal ties: 0

## FEVER gold-label accuracy

| Run | Seed | Correct | FEVER accuracy |
|---:|---:|---:|---:|
| 1 | 101 | 9358/13332 | 70.192% |
| 2 | 202 | 9369/13332 | 70.275% |
| 3 | 303 | 9369/13332 | 70.275% |
| 4 | 404 | 9369/13332 | 70.275% |
| 5 | 505 | 9283/13332 | 69.629% |

- Mean: 70.129%
- Population standard deviation: 0.252%
- Minimum: 69.629%
- Maximum: 70.275%
- Non-tied consensus FEVER accuracy: 70.275% (9369/13332)
- Consensus ties: 0

FEVER accuracy means agreement with FEVER gold labels, not absolute real-world truth.

## Stable-correct versus stable-wrong

- Stable correct: 8120 (60.906%)
- Stable wrong: 2336 (17.522%)
- Unstable: 2876 (21.572%)
- Incomplete: 0 (0.000%)

## NOT ENOUGH INFORMATION behaviour

- Cases producing NEI at least once: 3345
- Cases producing NEI in all five runs: 1047
- Cases switching between NEI and SUPPORTED/REFUTED: 2298
- Total NEI predictions: 10756 / 66660

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
| Exact 5/5 consistency | 78.428% | 78.498% |
| Mean modal agreement | 95.197% | 95.219% |
| Mean FEVER accuracy across runs | 70.129% | 70.246% |
