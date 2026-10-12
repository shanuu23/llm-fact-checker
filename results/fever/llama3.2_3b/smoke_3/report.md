# FEVER gold-evidence baseline: llama3.2:3b

## Purpose

This baseline tests whether `llama3.2:3b` classifies FEVER SUPPORTS and REFUTES claims when supplied with one complete gold Wikipedia evidence set. It uses only the supplied evidence in the prompt.

## Dataset and sample construction

- Source: `data/fever/raw/shared_task_dev.jsonl`
- Prepared data: `data/fever/processed/fever_baseline_150.jsonl`
- Cases: 3 (2 SUPPORTED, 1 REFUTED)
- Sampling seed: 42
- Construction: claims are ID/claim-deduplicated, deterministically shuffled per class, and assigned the valid gold evidence set with the fewest unique sentences (stable page/sentence tie-break).
- Extraction: required sentence references are resolved by streaming `wiki-*.jsonl`; complete articles are not retained.

## Model configuration

- Model: `llama3.2:3b`
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

- Correct: 0 / 3
- Overall accuracy: 0.000
- SUPPORTED accuracy: 0.000
- REFUTED accuracy: 0.000
- Invalid-format outputs: 0 (0.000%)
- Inference failures: 0
- Prediction counts: SUPPORTED: 0, REFUTED: 2, NOT ENOUGH INFORMATION: 1, INVALID: 0

## Confusion matrix

| Gold | SUPPORTED | REFUTED | NOT ENOUGH INFORMATION | INVALID |
|---|---:|---:|---:|---:|
| SUPPORTED | 0 | 2 | 0 | 0 |
| REFUTED | 0 | 0 | 1 | 0 |

## Incorrect cases

| FEVER ID | Gold | Prediction | Claim |
|---:|---|---|---|
| 1423 | SUPPORTED | REFUTED | Roberto Orci wrote Star Trek. |
| 1917 | SUPPORTED | REFUTED | The current Duke of York has been unmarried since his 1996 divorce. |
| 2466 | REFUTED | NOT ENOUGH INFORMATION | Cheese in the Trap (TV series) stars Batman. |

Full raw outputs and error details are preserved in `predictions.jsonl` and `errors.csv`.

## Limitations

The gold sample contains no NOT ENOUGH INFO cases, so this run does not measure gold NEI accuracy. It does not test evidence retrieval, robustness, stability, sensitivity, calibration, or real-world fact-checking ability. It measures only classification of FEVER SUPPORTS/REFUTES cases supplied with valid gold evidence.
