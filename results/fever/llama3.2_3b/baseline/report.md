# FEVER gold-evidence baseline: llama3.2:3b

## Purpose

This baseline tests whether `llama3.2:3b` classifies FEVER SUPPORTS and REFUTES claims when supplied with one complete gold Wikipedia evidence set. It uses only the supplied evidence in the prompt.

## Dataset and sample construction

- Source: `data/fever/raw/shared_task_dev.jsonl`
- Prepared data: `data/fever/processed/fever_baseline_150.jsonl`
- Cases: 150 (75 SUPPORTED, 75 REFUTED)
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

- Correct: 97 / 150
- Overall accuracy: 0.647
- SUPPORTED accuracy: 0.773
- REFUTED accuracy: 0.520
- Invalid-format outputs: 0 (0.000%)
- Inference failures: 0
- Prediction counts: SUPPORTED: 62, REFUTED: 53, NOT ENOUGH INFORMATION: 35, INVALID: 0

## Confusion matrix

| Gold | SUPPORTED | REFUTED | NOT ENOUGH INFORMATION | INVALID |
|---|---:|---:|---:|---:|
| SUPPORTED | 58 | 14 | 3 | 0 |
| REFUTED | 4 | 39 | 32 | 0 |

## Incorrect cases

| FEVER ID | Gold | Prediction | Claim |
|---:|---|---|---|
| 1423 | SUPPORTED | REFUTED | Roberto Orci wrote Star Trek. |
| 1917 | SUPPORTED | REFUTED | The current Duke of York has been unmarried since his 1996 divorce. |
| 2466 | REFUTED | NOT ENOUGH INFORMATION | Cheese in the Trap (TV series) stars Batman. |
| 3149 | REFUTED | NOT ENOUGH INFORMATION | No Country for Old Men starred Harrison Ford. |
| 10810 | REFUTED | NOT ENOUGH INFORMATION | Halsey signed her first recording contract in 1914. |
| 17218 | SUPPORTED | REFUTED | Pharmacology deals with the elucidation of molecular and organismal function in relation to chemicals. |
| 19177 | REFUTED | SUPPORTED | Victoria Palace Theatre is on the same side as a central London railway terminus. |
| 20493 | REFUTED | NOT ENOUGH INFORMATION | Super Bowl XLV was played on February 6, 2011 in Jamaica. |
| 34105 | REFUTED | NOT ENOUGH INFORMATION | Phoenix, Arizona is the most populous country. |
| 35291 | REFUTED | NOT ENOUGH INFORMATION | Samsung exited the electronics industry in the late 1960s. |
| 35708 | REFUTED | NOT ENOUGH INFORMATION | Seohyun is only German. |
| 37250 | SUPPORTED | NOT ENOUGH INFORMATION | Production took place in Sydney for the movie Australia (2008 film). |
| 39599 | REFUTED | NOT ENOUGH INFORMATION | Red Holt died before Steve Wozniak was born. |
| 48047 | REFUTED | NOT ENOUGH INFORMATION | Konidela Production Company was established by Amitabh Bachchan. |
| 49775 | REFUTED | NOT ENOUGH INFORMATION | Baloch people rarely speak the Balochi language. |
| 51757 | SUPPORTED | REFUTED | The KGB was not a military service. |
| 51885 | SUPPORTED | REFUTED | A Good Day to Die Hard was directed by John Moore. |
| 54701 | SUPPORTED | REFUTED | Shannon Lee acts. |
| 58415 | SUPPORTED | REFUTED | Ludwig van Beethoven was part of the classical era. |
| 59144 | SUPPORTED | REFUTED | Kevin Kline stars in The Road to El Dorado. |
| 60763 | REFUTED | NOT ENOUGH INFORMATION | Dan Trachtenberg is a radio show host. |
| 73466 | SUPPORTED | REFUTED | Colombiana was released after 2000. |
| 75217 | REFUTED | NOT ENOUGH INFORMATION | The Gettysburg Address was exclusively a movie. |
| 76166 | REFUTED | SUPPORTED | Camden, New Jersey is in a county in New Jersey. |
| 85702 | REFUTED | SUPPORTED | The Challenge XXX: Dirty 30 is a season of a show on Syfy. |
| 87483 | REFUTED | NOT ENOUGH INFORMATION | Dakota Fanning is a dancer. |
| 97589 | REFUTED | NOT ENOUGH INFORMATION | The Adventures of Pluto Nash was released in Africa. |
| 99251 | SUPPORTED | REFUTED | In 2015, among Americans, more than 50% of adults had consumed alcoholic drink in the last year. |
| 112585 | SUPPORTED | REFUTED | A sailor shirt is typically worn by Donald Duck. |
| 113990 | REFUTED | NOT ENOUGH INFORMATION | Chile is a food. |
| 114748 | REFUTED | NOT ENOUGH INFORMATION | Kung Fu Panda made $75 million during its opening weekend. |
| 123785 | REFUTED | NOT ENOUGH INFORMATION | Olympic medals have always displayed the Colosseum. |
| 124730 | REFUTED | NOT ENOUGH INFORMATION | The vacuum cleaner is one of the ABC Islands. |
| 131006 | REFUTED | NOT ENOUGH INFORMATION | You Belong with Me is a fan club. |
| 131495 | REFUTED | NOT ENOUGH INFORMATION | Kuching is in the state of Johor. |
| 139433 | REFUTED | NOT ENOUGH INFORMATION | Elizabeth of York was married to King George. |
| 139636 | REFUTED | NOT ENOUGH INFORMATION | Stan Beeman is only in modern comedies. |
| 140032 | REFUTED | NOT ENOUGH INFORMATION | Georges Gilles de la Tourette failed to describe symptoms in nine patients. |
| 147730 | SUPPORTED | NOT ENOUGH INFORMATION | Danger UXB is on a commercial TV network in Europe. |
| 149501 | REFUTED | NOT ENOUGH INFORMATION | Kung Fu Panda was number three at the box office. |
| 150096 | REFUTED | NOT ENOUGH INFORMATION | Trollhunters is a squid. |
| 150357 | REFUTED | SUPPORTED | Musala is a valley. |
| 150636 | SUPPORTED | REFUTED | Kareena Kapoor wrote negative reviews. |
| 156098 | REFUTED | NOT ENOUGH INFORMATION | Benjamin Franklin was born in 1790. |
| 158017 | SUPPORTED | REFUTED | Two of Mozart's sons were alive after his death. |
| 163812 | SUPPORTED | REFUTED | The Ukrainian Soviet Socialist Republic became Ukraine before the founding of the United Nations. |
| 184420 | REFUTED | NOT ENOUGH INFORMATION | Premam had a theatrical run of 151 days in 2012. |
| 185288 | REFUTED | NOT ENOUGH INFORMATION | Bradley Fuller is the sole owner of Platinum Dunes. |
| 186995 | REFUTED | NOT ENOUGH INFORMATION | The Bermuda Triangle is located completely within the Dead Sea. |
| 195122 | SUPPORTED | NOT ENOUGH INFORMATION | Backing vocalists sing with other backing vocalists. |
| 201748 | REFUTED | NOT ENOUGH INFORMATION | North Vietnam existed from 1945 to 1978. |
| 202050 | REFUTED | NOT ENOUGH INFORMATION | Tamerlan Tsarnaev was an only child. |
| 211278 | REFUTED | NOT ENOUGH INFORMATION | The Closer's final season began in July 2013. |

Full raw outputs and error details are preserved in `predictions.jsonl` and `errors.csv`.

## Limitations

The gold sample contains no NOT ENOUGH INFO cases, so this run does not measure gold NEI accuracy. It does not test evidence retrieval, robustness, stability, sensitivity, calibration, or real-world fact-checking ability. It measures only classification of FEVER SUPPORTS/REFUTES cases supplied with valid gold evidence.
