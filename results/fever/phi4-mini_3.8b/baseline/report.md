# FEVER gold-evidence baseline: phi4-mini:3.8b

## Purpose

This baseline tests whether `phi4-mini:3.8b` classifies FEVER SUPPORTS and REFUTES claims when supplied with one complete gold Wikipedia evidence set. It uses only the supplied evidence in the prompt.

## Dataset and sample construction

- Source: `data/fever/raw/shared_task_dev.jsonl`
- Prepared data: `data/fever/processed/fever_baseline_150.jsonl`
- Cases: 150 (75 SUPPORTED, 75 REFUTED)
- Sampling seed: 42
- Construction: claims are ID/claim-deduplicated, deterministically shuffled per class, and assigned the valid gold evidence set with the fewest unique sentences (stable page/sentence tie-break).
- Extraction: required sentence references are resolved by streaming `wiki-*.jsonl`; complete articles are not retained.

## Model configuration

- Model: `phi4-mini:3.8b`
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

- Correct: 87 / 150
- Overall accuracy: 0.580
- SUPPORTED accuracy: 0.960
- REFUTED accuracy: 0.200
- Invalid-format outputs: 0 (0.000%)
- Inference failures: 0
- Prediction counts: SUPPORTED: 97, REFUTED: 16, NOT ENOUGH INFORMATION: 37, INVALID: 0

## Confusion matrix

| Gold | SUPPORTED | REFUTED | NOT ENOUGH INFORMATION | INVALID |
|---|---:|---:|---:|---:|
| SUPPORTED | 72 | 1 | 2 | 0 |
| REFUTED | 25 | 15 | 35 | 0 |

## Incorrect cases

| FEVER ID | Gold | Prediction | Claim |
|---:|---|---|---|
| 2466 | REFUTED | NOT ENOUGH INFORMATION | Cheese in the Trap (TV series) stars Batman. |
| 4058 | REFUTED | SUPPORTED | Balibo (film) involves zero journalist. |
| 10810 | REFUTED | NOT ENOUGH INFORMATION | Halsey signed her first recording contract in 1914. |
| 19177 | REFUTED | SUPPORTED | Victoria Palace Theatre is on the same side as a central London railway terminus. |
| 20493 | REFUTED | NOT ENOUGH INFORMATION | Super Bowl XLV was played on February 6, 2011 in Jamaica. |
| 34105 | REFUTED | NOT ENOUGH INFORMATION | Phoenix, Arizona is the most populous country. |
| 34496 | REFUTED | SUPPORTED | Email filtering output is capable of redirecting birds elsewhere. |
| 35708 | REFUTED | NOT ENOUGH INFORMATION | Seohyun is only German. |
| 36501 | REFUTED | SUPPORTED | Bhagat Singh was not executed. |
| 37250 | SUPPORTED | NOT ENOUGH INFORMATION | Production took place in Sydney for the movie Australia (2008 film). |
| 39599 | REFUTED | NOT ENOUGH INFORMATION | Red Holt died before Steve Wozniak was born. |
| 44347 | REFUTED | SUPPORTED | Duff McKagan was born in May of 1964. |
| 49775 | REFUTED | SUPPORTED | Baloch people rarely speak the Balochi language. |
| 51757 | SUPPORTED | REFUTED | The KGB was not a military service. |
| 54701 | SUPPORTED | NOT ENOUGH INFORMATION | Shannon Lee acts. |
| 60763 | REFUTED | NOT ENOUGH INFORMATION | Dan Trachtenberg is a radio show host. |
| 65790 | REFUTED | SUPPORTED | Randy Savage has a high voice. |
| 70022 | REFUTED | SUPPORTED | Veep is not a television show. |
| 75217 | REFUTED | NOT ENOUGH INFORMATION | The Gettysburg Address was exclusively a movie. |
| 76166 | REFUTED | SUPPORTED | Camden, New Jersey is in a county in New Jersey. |
| 82450 | REFUTED | NOT ENOUGH INFORMATION | One True Thing is a book. |
| 85191 | REFUTED | SUPPORTED | Scotty Moore died on June 28, 2010. |
| 85702 | REFUTED | SUPPORTED | The Challenge XXX: Dirty 30 is a season of a show on Syfy. |
| 86829 | REFUTED | SUPPORTED | Chris Kyle did not die on February 2, 2013. |
| 87483 | REFUTED | NOT ENOUGH INFORMATION | Dakota Fanning is a dancer. |
| 90517 | REFUTED | NOT ENOUGH INFORMATION | Colombiana was released in 2001. |
| 92687 | REFUTED | SUPPORTED | Guillermo del Toro is not a novelist. |
| 93416 | REFUTED | SUPPORTED | Aubrey Anderson-Emmons did not star on Modern Family. |
| 94861 | REFUTED | NOT ENOUGH INFORMATION | The Winds of Winter is the eighth book in a series. |
| 99904 | REFUTED | NOT ENOUGH INFORMATION | Ashley Cole was born December 10, 1977. |
| 103298 | REFUTED | SUPPORTED | Phoenix, Arizona is not the capital of Arizona. |
| 104735 | REFUTED | SUPPORTED | Victor Manuelle was uninvolved with every version of Despacito. |
| 105025 | REFUTED | NOT ENOUGH INFORMATION | The Faroe Islands disappeared in 1035. |
| 106928 | REFUTED | NOT ENOUGH INFORMATION | Mamata Banerjee has only ever been called Mamata Banerjee. |
| 113214 | REFUTED | NOT ENOUGH INFORMATION | The Atlanta metropolitan area spans only 10 counties. |
| 114748 | REFUTED | SUPPORTED | Kung Fu Panda made $75 million during its opening weekend. |
| 123285 | REFUTED | NOT ENOUGH INFORMATION | Noah Cyrus was born in 1991. |
| 123785 | REFUTED | NOT ENOUGH INFORMATION | Olympic medals have always displayed the Colosseum. |
| 124730 | REFUTED | NOT ENOUGH INFORMATION | The vacuum cleaner is one of the ABC Islands. |
| 126292 | REFUTED | NOT ENOUGH INFORMATION | Camden, New Jersey is a county. |
| 131006 | REFUTED | NOT ENOUGH INFORMATION | You Belong with Me is a fan club. |
| 138356 | REFUTED | NOT ENOUGH INFORMATION | Murda Beatz was born on February 11, 1984. |
| 139433 | REFUTED | NOT ENOUGH INFORMATION | Elizabeth of York was married to King George. |
| 139636 | REFUTED | NOT ENOUGH INFORMATION | Stan Beeman is only in modern comedies. |
| 140032 | REFUTED | SUPPORTED | Georges Gilles de la Tourette failed to describe symptoms in nine patients. |
| 140353 | REFUTED | SUPPORTED | Henry II of France did not die. |
| 144611 | REFUTED | NOT ENOUGH INFORMATION | Agent Raghav – Crime Branch lacked any nominations. |
| 149501 | REFUTED | SUPPORTED | Kung Fu Panda was number three at the box office. |
| 150096 | REFUTED | NOT ENOUGH INFORMATION | Trollhunters is a squid. |
| 150357 | REFUTED | NOT ENOUGH INFORMATION | Musala is a valley. |
| 155704 | REFUTED | NOT ENOUGH INFORMATION | West Virginia borders Maine and Kentucky to the southwest. |
| 156098 | REFUTED | NOT ENOUGH INFORMATION | Benjamin Franklin was born in 1790. |
| 161007 | REFUTED | SUPPORTED | Gal Gadot was ranked behind Esti Ginzburgh for highest earning actress/models in Israel. |
| 175643 | REFUTED | NOT ENOUGH INFORMATION | Fabian Nicieza only writes screenplays. |
| 182936 | REFUTED | SUPPORTED | The soundtrack of Far from the Madding Crowd was only by a French composer. |
| 184420 | REFUTED | NOT ENOUGH INFORMATION | Premam had a theatrical run of 151 days in 2012. |
| 185288 | REFUTED | NOT ENOUGH INFORMATION | Bradley Fuller is the sole owner of Platinum Dunes. |
| 186995 | REFUTED | NOT ENOUGH INFORMATION | The Bermuda Triangle is located completely within the Dead Sea. |
| 198084 | REFUTED | SUPPORTED | Story of My Life wasn't recorded by One Direction. |
| 201115 | REFUTED | NOT ENOUGH INFORMATION | Marcus Bentley was born on November 5th, 1267. |
| 201748 | REFUTED | SUPPORTED | North Vietnam existed from 1945 to 1978. |
| 202050 | REFUTED | NOT ENOUGH INFORMATION | Tamerlan Tsarnaev was an only child. |
| 217212 | REFUTED | SUPPORTED | A monk only avoids practicing religious asceticism. |

Full raw outputs and error details are preserved in `predictions.jsonl` and `errors.csv`.

## Limitations

The gold sample contains no NOT ENOUGH INFO cases, so this run does not measure gold NEI accuracy. It does not test evidence retrieval, robustness, stability, sensitivity, calibration, or real-world fact-checking ability. It measures only classification of FEVER SUPPORTS/REFUTES cases supplied with valid gold evidence.
