# FEVER gold-evidence baseline: granite3.3:8b

## Purpose

This baseline tests whether `granite3.3:8b` classifies FEVER SUPPORTS and REFUTES claims when supplied with one complete gold Wikipedia evidence set. It uses only the supplied evidence in the prompt.

## Dataset and sample construction

- Source: `data\fever\raw\shared_task_dev.jsonl`
- Prepared data: `data\fever\processed\fever_baseline_150.jsonl`
- Cases: 150 (75 SUPPORTED, 75 REFUTED)
- Sampling seed: 42
- Construction: claims are ID/claim-deduplicated, deterministically shuffled per class, and assigned the valid gold evidence set with the fewest unique sentences (stable page/sentence tie-break).
- Extraction: required sentence references are resolved by streaming `wiki-*.jsonl`; complete articles are not retained.

## Model configuration

- Model: `granite3.3:8b`
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

- Correct: 131 / 150
- Overall accuracy: 0.873
- SUPPORTED accuracy: 0.840
- REFUTED accuracy: 0.907
- Invalid-format outputs: 3 (2.000%)
- Inference failures: 0
- Prediction counts: SUPPORTED: 66, REFUTED: 77, NOT ENOUGH INFORMATION: 4, INVALID: 3

## Confusion matrix

| Gold | SUPPORTED | REFUTED | NOT ENOUGH INFORMATION | INVALID |
|---|---:|---:|---:|---:|
| SUPPORTED | 63 | 9 | 1 | 2 |
| REFUTED | 3 | 68 | 3 | 1 |

## Incorrect cases

| FEVER ID | Gold | Prediction | Claim |
|---:|---|---|---|
| 1423 | SUPPORTED | REFUTED | Roberto Orci wrote Star Trek. |
| 7702 | SUPPORTED | REFUTED | An American directed One True Thing. |
| 16007 | SUPPORTED | INVALID | Riverdale is an adaptation. |
| 19177 | REFUTED | SUPPORTED | Victoria Palace Theatre is on the same side as a central London railway terminus. |
| 34105 | REFUTED | SUPPORTED | Phoenix, Arizona is the most populous country. |
| 34496 | REFUTED | NOT ENOUGH INFORMATION | Email filtering output is capable of redirecting birds elsewhere. |
| 39599 | REFUTED | NOT ENOUGH INFORMATION | Red Holt died before Steve Wozniak was born. |
| 51757 | SUPPORTED | REFUTED | The KGB was not a military service. |
| 58415 | SUPPORTED | REFUTED | Ludwig van Beethoven was part of the classical era. |
| 60763 | REFUTED | SUPPORTED | Dan Trachtenberg is a radio show host. |
| 73466 | SUPPORTED | REFUTED | Colombiana was released after 2000. |
| 147730 | SUPPORTED | INVALID | Danger UXB is on a commercial TV network in Europe. |
| 150636 | SUPPORTED | REFUTED | Kareena Kapoor wrote negative reviews. |
| 158017 | SUPPORTED | NOT ENOUGH INFORMATION | Two of Mozart's sons were alive after his death. |
| 163812 | SUPPORTED | REFUTED | The Ukrainian Soviet Socialist Republic became Ukraine before the founding of the United Nations. |
| 185167 | SUPPORTED | REFUTED | You Only Live Twice was directed by Lewis Gilbert. |
| 201748 | REFUTED | INVALID | North Vietnam existed from 1945 to 1978. |
| 202050 | REFUTED | NOT ENOUGH INFORMATION | Tamerlan Tsarnaev was an only child. |
| 224558 | SUPPORTED | REFUTED | Vampire Academy was a critical failure. |

Full raw outputs and error details are preserved in `predictions.jsonl` and `errors.csv`.

## Limitations

The gold sample contains no NOT ENOUGH INFO cases, so this run does not measure gold NEI accuracy. It does not test evidence retrieval, robustness, stability, sensitivity, calibration, or real-world fact-checking ability. It measures only classification of FEVER SUPPORTS/REFUTES cases supplied with valid gold evidence.
