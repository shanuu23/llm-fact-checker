# FEVER gold-evidence baseline: llama3.2:1b

## Purpose

This baseline tests whether `llama3.2:1b` classifies FEVER SUPPORTS and REFUTES claims when supplied with one complete gold Wikipedia evidence set. It uses only the supplied evidence in the prompt.

## Dataset and sample construction

- Source: `data/fever/raw/shared_task_dev.jsonl`
- Prepared data: `data/fever/processed/fever_baseline_150.jsonl`
- Cases: 150 (75 SUPPORTED, 75 REFUTED)
- Sampling seed: 42
- Construction: claims are ID/claim-deduplicated, deterministically shuffled per class, and assigned the valid gold evidence set with the fewest unique sentences (stable page/sentence tie-break).
- Extraction: required sentence references are resolved by streaming `wiki-*.jsonl`; complete articles are not retained.

## Model configuration

- Model: `llama3.2:1b`
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

- Correct: 76 / 150
- Overall accuracy: 0.507
- SUPPORTED accuracy: 0.960
- REFUTED accuracy: 0.053
- Invalid-format outputs: 0 (0.000%)
- Inference failures: 0
- Prediction counts: SUPPORTED: 131, REFUTED: 4, NOT ENOUGH INFORMATION: 15, INVALID: 0

## Confusion matrix

| Gold | SUPPORTED | REFUTED | NOT ENOUGH INFORMATION | INVALID |
|---|---:|---:|---:|---:|
| SUPPORTED | 72 | 0 | 3 | 0 |
| REFUTED | 59 | 4 | 12 | 0 |

## Incorrect cases

| FEVER ID | Gold | Prediction | Claim |
|---:|---|---|---|
| 2466 | REFUTED | SUPPORTED | Cheese in the Trap (TV series) stars Batman. |
| 3149 | REFUTED | SUPPORTED | No Country for Old Men starred Harrison Ford. |
| 4058 | REFUTED | SUPPORTED | Balibo (film) involves zero journalist. |
| 10810 | REFUTED | NOT ENOUGH INFORMATION | Halsey signed her first recording contract in 1914. |
| 11730 | REFUTED | SUPPORTED | Harold Macmillan was born on February 10, 1794. |
| 19177 | REFUTED | SUPPORTED | Victoria Palace Theatre is on the same side as a central London railway terminus. |
| 20493 | REFUTED | NOT ENOUGH INFORMATION | Super Bowl XLV was played on February 6, 2011 in Jamaica. |
| 34105 | REFUTED | SUPPORTED | Phoenix, Arizona is the most populous country. |
| 34496 | REFUTED | NOT ENOUGH INFORMATION | Email filtering output is capable of redirecting birds elsewhere. |
| 35291 | REFUTED | SUPPORTED | Samsung exited the electronics industry in the late 1960s. |
| 35708 | REFUTED | SUPPORTED | Seohyun is only German. |
| 36501 | REFUTED | SUPPORTED | Bhagat Singh was not executed. |
| 39599 | REFUTED | SUPPORTED | Red Holt died before Steve Wozniak was born. |
| 44347 | REFUTED | SUPPORTED | Duff McKagan was born in May of 1964. |
| 48047 | REFUTED | SUPPORTED | Konidela Production Company was established by Amitabh Bachchan. |
| 49775 | REFUTED | SUPPORTED | Baloch people rarely speak the Balochi language. |
| 60763 | REFUTED | SUPPORTED | Dan Trachtenberg is a radio show host. |
| 65790 | REFUTED | SUPPORTED | Randy Savage has a high voice. |
| 70022 | REFUTED | SUPPORTED | Veep is not a television show. |
| 76166 | REFUTED | SUPPORTED | Camden, New Jersey is in a county in New Jersey. |
| 77162 | REFUTED | SUPPORTED | The Mod Squad is a Peruvian television series. |
| 82450 | REFUTED | SUPPORTED | One True Thing is a book. |
| 85191 | REFUTED | SUPPORTED | Scotty Moore died on June 28, 2010. |
| 85702 | REFUTED | SUPPORTED | The Challenge XXX: Dirty 30 is a season of a show on Syfy. |
| 86829 | REFUTED | SUPPORTED | Chris Kyle did not die on February 2, 2013. |
| 87483 | REFUTED | SUPPORTED | Dakota Fanning is a dancer. |
| 88425 | REFUTED | SUPPORTED | Exit the King is by Alexander Hamilton. |
| 90517 | REFUTED | NOT ENOUGH INFORMATION | Colombiana was released in 2001. |
| 90865 | REFUTED | SUPPORTED | Salman Rushdie is a German politician. |
| 92687 | REFUTED | SUPPORTED | Guillermo del Toro is not a novelist. |
| 93416 | REFUTED | SUPPORTED | Aubrey Anderson-Emmons did not star on Modern Family. |
| 94861 | REFUTED | NOT ENOUGH INFORMATION | The Winds of Winter is the eighth book in a series. |
| 97589 | REFUTED | NOT ENOUGH INFORMATION | The Adventures of Pluto Nash was released in Africa. |
| 99904 | REFUTED | SUPPORTED | Ashley Cole was born December 10, 1977. |
| 103298 | REFUTED | SUPPORTED | Phoenix, Arizona is not the capital of Arizona. |
| 104735 | REFUTED | SUPPORTED | Victor Manuelle was uninvolved with every version of Despacito. |
| 105025 | REFUTED | NOT ENOUGH INFORMATION | The Faroe Islands disappeared in 1035. |
| 106928 | REFUTED | SUPPORTED | Mamata Banerjee has only ever been called Mamata Banerjee. |
| 112585 | SUPPORTED | NOT ENOUGH INFORMATION | A sailor shirt is typically worn by Donald Duck. |
| 113214 | REFUTED | NOT ENOUGH INFORMATION | The Atlanta metropolitan area spans only 10 counties. |
| 113990 | REFUTED | SUPPORTED | Chile is a food. |
| 114748 | REFUTED | SUPPORTED | Kung Fu Panda made $75 million during its opening weekend. |
| 123285 | REFUTED | SUPPORTED | Noah Cyrus was born in 1991. |
| 123785 | REFUTED | SUPPORTED | Olympic medals have always displayed the Colosseum. |
| 124730 | REFUTED | NOT ENOUGH INFORMATION | The vacuum cleaner is one of the ABC Islands. |
| 126292 | REFUTED | SUPPORTED | Camden, New Jersey is a county. |
| 131006 | REFUTED | SUPPORTED | You Belong with Me is a fan club. |
| 131495 | REFUTED | SUPPORTED | Kuching is in the state of Johor. |
| 138356 | REFUTED | SUPPORTED | Murda Beatz was born on February 11, 1984. |
| 139636 | REFUTED | SUPPORTED | Stan Beeman is only in modern comedies. |
| 140353 | REFUTED | SUPPORTED | Henry II of France did not die. |
| 144611 | REFUTED | NOT ENOUGH INFORMATION | Agent Raghav – Crime Branch lacked any nominations. |
| 149501 | REFUTED | SUPPORTED | Kung Fu Panda was number three at the box office. |
| 150096 | REFUTED | NOT ENOUGH INFORMATION | Trollhunters is a squid. |
| 150357 | REFUTED | SUPPORTED | Musala is a valley. |
| 155704 | REFUTED | SUPPORTED | West Virginia borders Maine and Kentucky to the southwest. |
| 156098 | REFUTED | NOT ENOUGH INFORMATION | Benjamin Franklin was born in 1790. |
| 158017 | SUPPORTED | NOT ENOUGH INFORMATION | Two of Mozart's sons were alive after his death. |
| 161007 | REFUTED | SUPPORTED | Gal Gadot was ranked behind Esti Ginzburgh for highest earning actress/models in Israel. |
| 161787 | REFUTED | SUPPORTED | Planet Hollywood Las Vegas is owned by Universal Studios. |
| 175643 | REFUTED | SUPPORTED | Fabian Nicieza only writes screenplays. |
| 177855 | REFUTED | SUPPORTED | Milk is based on the life of a businessman. |
| 182936 | REFUTED | SUPPORTED | The soundtrack of Far from the Madding Crowd was only by a French composer. |
| 183468 | SUPPORTED | NOT ENOUGH INFORMATION | Henry III of France was succeeded by another Henry. |
| 184420 | REFUTED | SUPPORTED | Premam had a theatrical run of 151 days in 2012. |
| 185288 | REFUTED | SUPPORTED | Bradley Fuller is the sole owner of Platinum Dunes. |
| 186995 | REFUTED | SUPPORTED | The Bermuda Triangle is located completely within the Dead Sea. |
| 198084 | REFUTED | SUPPORTED | Story of My Life wasn't recorded by One Direction. |
| 200279 | REFUTED | SUPPORTED | Natural Born Killers was based upon Tarantino's original screenplay without revision. |
| 201115 | REFUTED | SUPPORTED | Marcus Bentley was born on November 5th, 1267. |
| 201748 | REFUTED | SUPPORTED | North Vietnam existed from 1945 to 1978. |
| 202050 | REFUTED | SUPPORTED | Tamerlan Tsarnaev was an only child. |
| 211278 | REFUTED | SUPPORTED | The Closer's final season began in July 2013. |
| 217212 | REFUTED | SUPPORTED | A monk only avoids practicing religious asceticism. |

Full raw outputs and error details are preserved in `predictions.jsonl` and `errors.csv`.

## Limitations

The gold sample contains no NOT ENOUGH INFO cases, so this run does not measure gold NEI accuracy. It does not test evidence retrieval, robustness, stability, sensitivity, calibration, or real-world fact-checking ability. It measures only classification of FEVER SUPPORTS/REFUTES cases supplied with valid gold evidence.
