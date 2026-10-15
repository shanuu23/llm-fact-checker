# FEVER gold-evidence baseline: qwen2.5:7b

## Purpose

This baseline tests whether `qwen2.5:7b` classifies FEVER SUPPORTS and REFUTES claims when supplied with one complete gold Wikipedia evidence set. It uses only the supplied evidence in the prompt.

## Dataset and sample construction

- Source: `data/fever/raw/shared_task_dev.jsonl`
- Prepared data: `data/fever/processed/fever_baseline_150.jsonl`
- Cases: 150 (75 SUPPORTED, 75 REFUTED)
- Sampling seed: 42
- Construction: claims are ID/claim-deduplicated, deterministically shuffled per class, and assigned the valid gold evidence set with the fewest unique sentences (stable page/sentence tie-break).
- Extraction: required sentence references are resolved by streaming `wiki-*.jsonl`; complete articles are not retained.

## Model configuration

- Model: `qwen2.5:7b`
- Temperature: 0
- Model seed: 42
- Execution: sequential, non-streaming requests through the local Ollama HTTP API

## Prompt

```text
You are an evidence-grounded fact verifier.

Determine whether the provided evidence supports, refutes, or does not provide enough information to determine the claim.

Use only the supplied evidence. Do not rely on outside knowledge.

Return exactly one of:

SUPPORTED
REFUTED
NOT ENOUGH INFORMATION

Return the label only. Do not provide an explanation.
```

The user message contains the claim and numbered evidence sentences. Gold labels are not included.

## Results

- Correct: 136 / 150
- Overall accuracy: 0.907
- SUPPORTED accuracy: 0.880
- REFUTED accuracy: 0.933
- Invalid-format outputs: 1 (0.667%)
- Inference failures: 0
- Prediction counts: SUPPORTED: 70, REFUTED: 78, NOT ENOUGH INFORMATION: 1, INVALID: 1

## Confusion matrix

| Gold | SUPPORTED | REFUTED | NOT ENOUGH INFORMATION | INVALID |
|---|---:|---:|---:|---:|
| SUPPORTED | 66 | 8 | 1 | 0 |
| REFUTED | 4 | 70 | 0 | 1 |

## Incorrect cases

| FEVER ID | Gold | Prediction | Claim |
|---:|---|---|---|
| 16007 | SUPPORTED | NOT ENOUGH INFORMATION | Riverdale is an adaptation. |
| 17218 | SUPPORTED | REFUTED | Pharmacology deals with the elucidation of molecular and organismal function in relation to chemicals. |
| 19177 | REFUTED | SUPPORTED | Victoria Palace Theatre is on the same side as a central London railway terminus. |
| 51757 | SUPPORTED | REFUTED | The KGB was not a military service. |
| 58415 | SUPPORTED | REFUTED | Ludwig van Beethoven was part of the classical era. |
| 60763 | REFUTED | SUPPORTED | Dan Trachtenberg is a radio show host. |
| 76166 | REFUTED | SUPPORTED | Camden, New Jersey is in a county in New Jersey. |
| 144188 | SUPPORTED | REFUTED | Always was directed by a director. |
| 144715 | SUPPORTED | REFUTED | Ian Gillan was born on August 19, 1945. |
| 158017 | SUPPORTED | REFUTED | Two of Mozart's sons were alive after his death. |
| 163812 | SUPPORTED | REFUTED | The Ukrainian Soviet Socialist Republic became Ukraine before the founding of the United Nations. |
| 177855 | REFUTED | SUPPORTED | Milk is based on the life of a businessman. |
| 196158 | SUPPORTED | REFUTED | Sixteen Primetime Emmy Award nominations have gone to actor Jon Hamm. |
| 217212 | REFUTED | INVALID | A monk only avoids practicing religious asceticism. |

Full raw outputs and error details are preserved in `predictions.jsonl` and `errors.csv`.

## Limitations

The gold sample contains no NOT ENOUGH INFO cases, so this run does not measure gold NEI accuracy. It does not test evidence retrieval, robustness, stability, sensitivity, calibration, or real-world fact-checking ability. It measures only classification of FEVER SUPPORTS/REFUTES cases supplied with valid gold evidence.
