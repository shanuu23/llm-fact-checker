# FEVER gold-evidence baseline: gemma3:4b

## Purpose

This baseline tests whether `gemma3:4b` classifies FEVER SUPPORTS and REFUTES claims when supplied with one complete gold Wikipedia evidence set. It uses only the supplied evidence in the prompt.

## Dataset and sample construction

- Source: `data/fever/raw/shared_task_dev.jsonl`
- Prepared data: `data/fever/processed/fever_baseline_150.jsonl`
- Cases: 150 (75 SUPPORTED, 75 REFUTED)
- Sampling seed: 42
- Construction: claims are ID/claim-deduplicated, deterministically shuffled per class, and assigned the valid gold evidence set with the fewest unique sentences (stable page/sentence tie-break).
- Extraction: required sentence references are resolved by streaming `wiki-*.jsonl`; complete articles are not retained.

## Model configuration

- Model: `gemma3:4b`
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

- Correct: 121 / 150
- Overall accuracy: 0.807
- SUPPORTED accuracy: 0.933
- REFUTED accuracy: 0.680
- Invalid-format outputs: 0 (0.000%)
- Inference failures: 0
- Prediction counts: SUPPORTED: 84, REFUTED: 56, NOT ENOUGH INFORMATION: 10, INVALID: 0

## Confusion matrix

| Gold | SUPPORTED | REFUTED | NOT ENOUGH INFORMATION | INVALID |
|---|---:|---:|---:|---:|
| SUPPORTED | 70 | 5 | 0 | 0 |
| REFUTED | 14 | 51 | 10 | 0 |

## Incorrect cases

| FEVER ID | Gold | Prediction | Claim |
|---:|---|---|---|
| 4058 | REFUTED | SUPPORTED | Balibo (film) involves zero journalist. |
| 19177 | REFUTED | SUPPORTED | Victoria Palace Theatre is on the same side as a central London railway terminus. |
| 34496 | REFUTED | SUPPORTED | Email filtering output is capable of redirecting birds elsewhere. |
| 39599 | REFUTED | SUPPORTED | Red Holt died before Steve Wozniak was born. |
| 51757 | SUPPORTED | REFUTED | The KGB was not a military service. |
| 58415 | SUPPORTED | REFUTED | Ludwig van Beethoven was part of the classical era. |
| 59144 | SUPPORTED | REFUTED | Kevin Kline stars in The Road to El Dorado. |
| 60763 | REFUTED | SUPPORTED | Dan Trachtenberg is a radio show host. |
| 70022 | REFUTED | SUPPORTED | Veep is not a television show. |
| 76166 | REFUTED | SUPPORTED | Camden, New Jersey is in a county in New Jersey. |
| 82450 | REFUTED | SUPPORTED | One True Thing is a book. |
| 85702 | REFUTED | SUPPORTED | The Challenge XXX: Dirty 30 is a season of a show on Syfy. |
| 87483 | REFUTED | NOT ENOUGH INFORMATION | Dakota Fanning is a dancer. |
| 92687 | REFUTED | SUPPORTED | Guillermo del Toro is not a novelist. |
| 105025 | REFUTED | NOT ENOUGH INFORMATION | The Faroe Islands disappeared in 1035. |
| 106928 | REFUTED | SUPPORTED | Mamata Banerjee has only ever been called Mamata Banerjee. |
| 113214 | REFUTED | NOT ENOUGH INFORMATION | The Atlanta metropolitan area spans only 10 counties. |
| 113990 | REFUTED | NOT ENOUGH INFORMATION | Chile is a food. |
| 123785 | REFUTED | NOT ENOUGH INFORMATION | Olympic medals have always displayed the Colosseum. |
| 131006 | REFUTED | NOT ENOUGH INFORMATION | You Belong with Me is a fan club. |
| 140032 | REFUTED | SUPPORTED | Georges Gilles de la Tourette failed to describe symptoms in nine patients. |
| 150096 | REFUTED | NOT ENOUGH INFORMATION | Trollhunters is a squid. |
| 150357 | REFUTED | NOT ENOUGH INFORMATION | Musala is a valley. |
| 163812 | SUPPORTED | REFUTED | The Ukrainian Soviet Socialist Republic became Ukraine before the founding of the United Nations. |
| 177855 | REFUTED | SUPPORTED | Milk is based on the life of a businessman. |
| 182936 | REFUTED | SUPPORTED | The soundtrack of Far from the Madding Crowd was only by a French composer. |
| 184420 | REFUTED | NOT ENOUGH INFORMATION | Premam had a theatrical run of 151 days in 2012. |
| 202050 | REFUTED | NOT ENOUGH INFORMATION | Tamerlan Tsarnaev was an only child. |
| 224558 | SUPPORTED | REFUTED | Vampire Academy was a critical failure. |

Full raw outputs and error details are preserved in `predictions.jsonl` and `errors.csv`.

## Limitations

The gold sample contains no NOT ENOUGH INFO cases, so this run does not measure gold NEI accuracy. It does not test evidence retrieval, robustness, stability, sensitivity, calibration, or real-world fact-checking ability. It measures only classification of FEVER SUPPORTS/REFUTES cases supplied with valid gold evidence.
