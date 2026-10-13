# FEVER gold-evidence baseline: llama3.2:1b

## Purpose

This baseline tests whether `llama3.2:1b` classifies FEVER SUPPORTS and REFUTES claims when supplied with one complete gold Wikipedia evidence set. It uses only the supplied evidence in the prompt.

## Dataset and sample construction

- Source: `data/fever/raw/shared_task_dev.jsonl`
- Prepared data: `data/fever/full/fever_dev_full.jsonl`
- Cases: 13332 (6666 SUPPORTED, 6666 REFUTED)
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

- Correct: 6465 / 13332
- Overall accuracy: 0.485
- SUPPORTED accuracy: 0.950
- REFUTED accuracy: 0.020
- Invalid-format outputs: 1 (0.008%)
- Inference failures: 0
- Prediction counts: SUPPORTED: 11742, REFUTED: 161, NOT ENOUGH INFORMATION: 1428, INVALID: 1

## Confusion matrix

| Gold | SUPPORTED | REFUTED | NOT ENOUGH INFORMATION | INVALID |
|---|---:|---:|---:|---:|
| SUPPORTED | 6335 | 31 | 300 | 0 |
| REFUTED | 5407 | 130 | 1128 | 1 |

## Incorrect cases

| FEVER ID | Gold | Prediction | Claim |
|---:|---|---|---|
| 12 | REFUTED | SUPPORTED | Carlos Santana disbanded Santana in 1965. |
| 83 | REFUTED | SUPPORTED | Honeymoon is the third major-label record by Lana Del Rey. |
| 304 | REFUTED | SUPPORTED | Aphrodite is unmarried. |
| 316 | REFUTED | SUPPORTED | Yemen has an area of 250,750 square miles. |
| 327 | SUPPORTED | NOT ENOUGH INFORMATION | Donald Duck typically wears a cap. |
| 354 | REFUTED | NOT ENOUGH INFORMATION | LinkedIn is available in zero languages as of 2013. |
| 369 | REFUTED | SUPPORTED | The State of Palestine claims a territory in Western Asia. |
| 370 | REFUTED | SUPPORTED | The State of Palestine claims a territory in East Asia. |
| 420 | REFUTED | SUPPORTED | Toy Story 2 was released by Universal Pictures. |
| 422 | REFUTED | SUPPORTED | John Krasinski is an astronaut. |
| 446 | REFUTED | SUPPORTED | Benjamin Franklin earned the title of The Dead Body. |
| 493 | SUPPORTED | NOT ENOUGH INFORMATION | Nuuk is a seat of government. |
| 499 | REFUTED | SUPPORTED | Paul Nicholls is a singer. |
| 529 | REFUTED | SUPPORTED | Musala is 2,525 meters. |
| 531 | REFUTED | NOT ENOUGH INFORMATION | Keegan-Michael Key played zero characters. |
| 573 | REFUTED | SUPPORTED | Doxycycline may not cause vomiting. |
| 619 | REFUTED | SUPPORTED | Hannah and Her Sisters' ending is unrelated to Thanksgiving. |
| 631 | REFUTED | SUPPORTED | The Adventures of Pluto Nash is an Australian-American science fiction action comedy film from the 1960s. |
| 637 | REFUTED | NOT ENOUGH INFORMATION | Liberal Socialists were the only people to form The Blue Dog Coalition. |
| 673 | REFUTED | SUPPORTED | Saxony is only a town. |
| 753 | REFUTED | NOT ENOUGH INFORMATION | Tottenham Hotspur F.C. is a basketball team. |
| 787 | REFUTED | SUPPORTED | Shawn Carlson is only German. |
| 850 | REFUTED | SUPPORTED | John S. McCain Jr. graduated in 1901. |
| 916 | REFUTED | SUPPORTED | Pulmonary embolism is not indicated by rapid breathing. |
| 972 | REFUTED | SUPPORTED | Stan Beeman is a fictional setting. |
| 976 | REFUTED | SUPPORTED | Doug Petrie has yet to appear on the web. |
| 1032 | REFUTED | NOT ENOUGH INFORMATION | Yin and yang do not have a portion of the opposite element in each section. |
| 1094 | REFUTED | SUPPORTED | Henry Cavill is not part of the DC Extended Universe. |
| 1126 | REFUTED | SUPPORTED | Wildfang is a US-based men's apparel company featuring clothing that is tomboyish in style. |
| 1136 | REFUTED | SUPPORTED | Tiber Oil Field was discovered in France. |
| 1180 | REFUTED | SUPPORTED | Renato Balestra refused to study for an engineering degree. |
| 1188 | REFUTED | SUPPORTED | Pearl Jam is a type of dressing. |
| 1244 | REFUTED | SUPPORTED | The Nottingham Forest F.C. is based in Lincolnshire, England. |
| 1251 | SUPPORTED | NOT ENOUGH INFORMATION | Watertown, Massachusetts is in the most populous state in New England. |
| 1325 | REFUTED | NOT ENOUGH INFORMATION | The Winds of Winter is being written by J. R. R. Tolkien. |
| 1362 | REFUTED | SUPPORTED | Deepika Padukone only acts in tragedies. |
| 1481 | REFUTED | SUPPORTED | EA Black Box was founded in 1999. |
| 1598 | REFUTED | SUPPORTED | The Hunger Games are a film trilogy. |
| 1599 | REFUTED | SUPPORTED | The Hunger Games are series of three films. |
| 1746 | REFUTED | SUPPORTED | Sheryl Lee has yet to work with Woody Allen. |
| 1799 | REFUTED | SUPPORTED | Aphrodite plays tennis with Hephaestus. |
| 1831 | REFUTED | NOT ENOUGH INFORMATION | DNA is a single. |
| 1931 | SUPPORTED | NOT ENOUGH INFORMATION | Team Fortress 2 development was led by John Cook. |
| 2043 | REFUTED | SUPPORTED | Pakistan sits outside of Central and Western Asia. |
| 2151 | REFUTED | SUPPORTED | One True Thing is a German film. |
| 2317 | REFUTED | SUPPORTED | Edison Machine Works was set up to produce cups. |
| 2358 | SUPPORTED | NOT ENOUGH INFORMATION | Computer chips became Virginia's most valuable export in 2002. |
| 2359 | REFUTED | SUPPORTED | Virginia keeps all computer chips manufactured within the state for use in Virginian electronics. |
| 2372 | REFUTED | SUPPORTED | Guillermo del Toro works in the coal industry. |
| 2374 | REFUTED | SUPPORTED | Anthony Scaramucci is not a political figure. |
| 2421 | REFUTED | NOT ENOUGH INFORMATION | Beaverton, Oregon's city center is in the Grand Canyon. |
| 2466 | REFUTED | SUPPORTED | Cheese in the Trap (TV series) stars Batman. |
| 2521 | REFUTED | SUPPORTED | Akkineni Nageswara Rao plays in Daag. |
| 2531 | REFUTED | SUPPORTED | Dopamine prevents neuromodulation. |
| 2797 | REFUTED | SUPPORTED | Night of the Living Dead is a 1990 American horror film. |
| 2798 | REFUTED | SUPPORTED | Night of the Living Dead is a Spanish comic book. |
| 2938 | REFUTED | SUPPORTED | Legendary Entertainment is an independent company. |
| 2948 | REFUTED | SUPPORTED | The Mod Squad is an American movie. |
| 2984 | REFUTED | SUPPORTED | Kevin Costner refused the role of Devil Anse Hatfield in 2012. |
| 3008 | REFUTED | NOT ENOUGH INFORMATION | Jamie Lee Curtis' mother is a Palestinian. |
| 3019 | REFUTED | SUPPORTED | The horse was domesticated on a broad scale by 3103 BC. |
| 3033 | REFUTED | NOT ENOUGH INFORMATION | The Gifted is from outside of America. |
| 3063 | SUPPORTED | NOT ENOUGH INFORMATION | Ronaldinho has played 90 matches. |
| 3111 | REFUTED | SUPPORTED | Luis Fonsi was born in the eighties. |
| 3119 | REFUTED | NOT ENOUGH INFORMATION | The University of Illinois at Chicago is located in Jamaica. |
| 3149 | REFUTED | SUPPORTED | No Country for Old Men starred Harrison Ford. |
| 3160 | REFUTED | SUPPORTED | University of Chicago Law School is ranked first in the 2016 QS World University Rankings. |
| 3202 | REFUTED | SUPPORTED | Zoe Saldana was born in May of 1978. |
| 3210 | REFUTED | SUPPORTED | The Challenge XXX: Dirty 30 is the thirteenth season of a reality game show. |
| 3305 | REFUTED | SUPPORTED | Julianne Moore was not in the television series As the World Turns. |
| 3338 | REFUTED | SUPPORTED | Mike Huckabee has been mayor of Pittsburgh for his entire political career. |
| 3371 | REFUTED | NOT ENOUGH INFORMATION | Maria Theresa reigned for only one year. |
| 3413 | REFUTED | SUPPORTED | Samsung was separated into ten business groups in 1987. |
| 3483 | REFUTED | SUPPORTED | Brie Larson was denied any role in the movie Short Term 12. |
| 3494 | REFUTED | SUPPORTED | English people are unconnected to the Saxons. |
| 3518 | REFUTED | SUPPORTED | Exercise is bad for heart health. |
| 3546 | REFUTED | SUPPORTED | Villa Park hosted a tennis match on August 12, 2012. |
| 3556 | REFUTED | SUPPORTED | A River Runs Through It has lost every nomination. |
| 3562 | REFUTED | SUPPORTED | Fargo (season 3) has yet to air. |
| 3588 | REFUTED | NOT ENOUGH INFORMATION | Jed Whedon has only ever lived on Mars. |
| 3598 | REFUTED | SUPPORTED | Ice-T began his career as a rapper in 1984. |
| 3638 | REFUTED | SUPPORTED | There are currently 15,882,417 Mormon members as of 2007. |
| 3656 | REFUTED | SUPPORTED | John DiMaggio refused to work on any shows from Cartoon Network. |
| 3668 | REFUTED | SUPPORTED | Andrew Moray refused to cede the territory he gained to John Balliol. |
| 3682 | REFUTED | SUPPORTED | English people are disconnected from the Jutes and Frisians. |
| 3718 | REFUTED | SUPPORTED | Victoria Palace Theatre is outside of Victoria Street. |
| 3783 | REFUTED | SUPPORTED | The American actor that plays Chumlee was born on the 10th. |
| 3795 | REFUTED | SUPPORTED | The Fame has sold over 4.5 billion copies. |
| 3809 | REFUTED | NOT ENOUGH INFORMATION | Sonny & Cher began in the mid-1950's. |
| 4058 | REFUTED | SUPPORTED | Balibo (film) involves zero journalist. |
| 4091 | REFUTED | SUPPORTED | Zac Efron's full name is Zac David Alexander Efron. |
| 4100 | SUPPORTED | NOT ENOUGH INFORMATION | Sonny & Cher began in the mid-1930's. |
| 4172 | REFUTED | SUPPORTED | A monster is always a person. |
| 4179 | SUPPORTED | NOT ENOUGH INFORMATION | The 14th Dalai Lama fled to India during the Tibetan uprising. |
| 4246 | REFUTED | SUPPORTED | Meteora is a country album. |
| 4251 | REFUTED | NOT ENOUGH INFORMATION | Solanum contains the leopards. |
| 4280 | REFUTED | NOT ENOUGH INFORMATION | Ekta Kapoor did not produce an Indian soap opera. |
| 4500 | REFUTED | SUPPORTED | Sledgehammer was first performed live in 2004. |
| 4595 | SUPPORTED | NOT ENOUGH INFORMATION | The Burj Khalifa contains 8 escalators. |
| 4675 | REFUTED | SUPPORTED | Aphrodite is not married to Hephaestus. |
| 4713 | REFUTED | SUPPORTED | Tool has won three Oscars. |
| 4717 | SUPPORTED | NOT ENOUGH INFORMATION | Moscovium is an element. |
| 4818 | REFUTED | SUPPORTED | English people are descended from the Romans. |
| 4819 | REFUTED | SUPPORTED | English people are unconnected to the Romans. |
| 4824 | REFUTED | NOT ENOUGH INFORMATION | Watertown, Massachusetts is in France. |
| 4916 | REFUTED | SUPPORTED | Paul Nicholls created Joe Wicks. |
| 4987 | REFUTED | SUPPORTED | Southern Hospitality peaked at number 21 on the Top 40. |
| 5004 | SUPPORTED | NOT ENOUGH INFORMATION | The Crips are a street gang. |
| 5075 | SUPPORTED | NOT ENOUGH INFORMATION | The State of Palestine claims the West Bank. |
| 5076 | REFUTED | NOT ENOUGH INFORMATION | The State of Palestine claims Algeria. |
| 5080 | REFUTED | SUPPORTED | The Times began printing before or during 1785. |
| 5111 | SUPPORTED | NOT ENOUGH INFORMATION | Michael Hutchence died in Sydney, Australia. |
| 5127 | REFUTED | SUPPORTED | Kellyanne Conway did not claim Michael Flynn had the full confidence of the president hours before he was dismissed. |
| 5169 | REFUTED | SUPPORTED | Jonah Hill is not a screenwriter. |
| 5177 | REFUTED | SUPPORTED | The Armenian Genocide also fails to be known as the Armenian Holocaust. |
| 5187 | REFUTED | SUPPORTED | Honeymoon is the third perfume line by Lana Del Rey. |
| 5314 | REFUTED | SUPPORTED | Younger is a town. |
| 5324 | SUPPORTED | REFUTED | The current Duke of York has been unmarried since his June 1996 divorce. |
| 5346 | REFUTED | SUPPORTED | Ripon College is in California. |
| 5421 | REFUTED | SUPPORTED | Stephen Hillenburg developed an interest in everything but art. |
| 5454 | REFUTED | SUPPORTED | Jack Falahee's middle name is Bryan. |
| 5466 | REFUTED | SUPPORTED | Due Date was shot in Los Angeles, Alabama. |
| 5513 | SUPPORTED | NOT ENOUGH INFORMATION | Shadowhunters premiered in 2016. |
| 5514 | REFUTED | NOT ENOUGH INFORMATION | Shadowhunters premiered in 2018. |
| 5545 | REFUTED | SUPPORTED | Atlanta metropolitan area is located in north Florida. |
| 5599 | REFUTED | SUPPORTED | San Francisco Bay Area is home to approximately 10 million people. |
| 5689 | REFUTED | SUPPORTED | Amyotrophic lateral sclerosis is a cure. |
| 5705 | REFUTED | SUPPORTED | Emma Watson is a French-British car. |
| 5719 | REFUTED | NOT ENOUGH INFORMATION | Qui-Gon Jinn is married to Liam Neeson. |
| 5738 | REFUTED | SUPPORTED | Melancholia was directed by a Japanese sushi chef. |
| 5745 | REFUTED | SUPPORTED | One Flew Over the Cuckoo's Nest only won one Oscar, for sound design. |
| 5766 | REFUTED | SUPPORTED | Hot Right Now is from the author Bob Stanley. |
| 5781 | REFUTED | SUPPORTED | Raja Hindustani was directed solely by Mariah Carey. |
| 5833 | REFUTED | SUPPORTED | The 100 is a TV series following a group of swans. |
| 5900 | REFUTED | NOT ENOUGH INFORMATION | The Paper stars a horse. |
| 5980 | REFUTED | NOT ENOUGH INFORMATION | Camden, New Jersey is the home of Yale School of Medicine. |
| 6017 | SUPPORTED | NOT ENOUGH INFORMATION | Stanley Williams died in California. |
| 6032 | REFUTED | SUPPORTED | Aruba is the only ABC Island. |
| 6275 | REFUTED | SUPPORTED | Recovery is an album focusing on themes of isolation and addiction. |
| 6317 | REFUTED | SUPPORTED | Loretta Sanchez was born on January 7, 1950. |
| 6334 | REFUTED | SUPPORTED | The Challenge XXX: Dirty 30 is not a season of a reality game show. |
| 6390 | REFUTED | SUPPORTED | Spider-Man 2 is a store. |
| 6414 | REFUTED | SUPPORTED | The Godfather Part II is a political party. |
| 6462 | REFUTED | NOT ENOUGH INFORMATION | Emperor Norton is not immortalized by authors from different countries. |
| 6572 | REFUTED | SUPPORTED | The Winds of Winter was not called "very surprising and satisfying." |
| 6668 | REFUTED | SUPPORTED | Louis Malle worked solely in pharmaceuticals. |
| 6743 | REFUTED | SUPPORTED | Jens Stoltenberg was the first Prime Minister of Norway from 2005 to 2013. |
| 6834 | REFUTED | SUPPORTED | Chris Benoit's middle name is Jason. |
| 7000 | REFUTED | SUPPORTED | Moves Like Jagger is the lead single on Overexposed. |
| 7026 | REFUTED | NOT ENOUGH INFORMATION | Uranium uses the symbol M. |
| 7041 | REFUTED | SUPPORTED | Kareem Abdul-Jabbar is not ranked 3rd all-time in blocks. |
| 7105 | REFUTED | SUPPORTED | Zac Efron is Somali. |
| 7138 | REFUTED | SUPPORTED | The Tenth Doctor travels alone during the time period 2009-2010. |
| 7183 | REFUTED | SUPPORTED | Sands Hotel and Casino closed in 1952. |
| 7209 | REFUTED | SUPPORTED | Colombiana is a Irish film. |
| 7265 | REFUTED | SUPPORTED | SpongeBob SquarePants is a documentary series. |
| 7286 | REFUTED | NOT ENOUGH INFORMATION | Croatia is incapable of rendering services. |
| 7336 | REFUTED | SUPPORTED | Lizzy Caplan is incapable of appearing in multiple television shows. |
| 7418 | SUPPORTED | NOT ENOUGH INFORMATION | The Faroe Islands existed in 1035. |
| 7436 | REFUTED | SUPPORTED | Simon Pegg was born on February 14th, 1860. |
| 7624 | SUPPORTED | NOT ENOUGH INFORMATION | Ghost was released in 1990. |
| 7625 | REFUTED | NOT ENOUGH INFORMATION | Ghost, the film, was released in 1994. |
| 7697 | REFUTED | SUPPORTED | Riddick is in a 2023 film. |
| 7728 | REFUTED | SUPPORTED | Hinduism has zero textual resources. |
| 7744 | REFUTED | SUPPORTED | Mount Hood is the sixth highest point in its mountain range. |
| 7758 | REFUTED | NOT ENOUGH INFORMATION | Ragtime was released in 1965. |
| 7769 | REFUTED | NOT ENOUGH INFORMATION | Tottenham Hotspur F.C. is Scottish. |
| 7811 | REFUTED | SUPPORTED | Adobe Photoshop is incapable of being part of the Adobe Systems. |
| 7876 | SUPPORTED | NOT ENOUGH INFORMATION | The ruins of the ancient Roman town of Herculaneum lie near Naples. |
| 7877 | REFUTED | SUPPORTED | The ruins of the ancient Roman town of Herculaneum were teleported to a location outside Naples. |
| 7982 | REFUTED | NOT ENOUGH INFORMATION | The Siege of Fort Stanwix took place in 2000. |
| 8022 | REFUTED | SUPPORTED | Colin Kaepernick was not backup to Alex Smith. |
| 8065 | SUPPORTED | NOT ENOUGH INFORMATION | Accessing email and the World Wide Web requires Internet Access. |
| 8167 | REFUTED | SUPPORTED | Caesar is a 1937 adaptation of The Tempest. |
| 8222 | REFUTED | SUPPORTED | Ashley Cole is only Japanese. |
| 8248 | REFUTED | SUPPORTED | The Good Wife is a movie. |
| 8298 | REFUTED | SUPPORTED | Simon Pegg is only a banker. |
| 8304 | REFUTED | SUPPORTED | Therasia is outside of the Greek Cyclades. |
| 8404 | REFUTED | NOT ENOUGH INFORMATION | US Airways Flight 1549 had 1978 people on board. |
| 8429 | REFUTED | SUPPORTED | The series finale of Make It or Break It is ending on its 5th season. |
| 8443 | REFUTED | SUPPORTED | A Milli is by Justin Bieber. |
| 8464 | REFUTED | SUPPORTED | I Kissed a Girl is only an EP. |
| 8538 | REFUTED | SUPPORTED | Hermit crabs are arachnids. |
| 8577 | REFUTED | SUPPORTED | The Dodo has been depicted with a naked head and arm. |
| 8581 | REFUTED | SUPPORTED | Johann Wolfgang von Goethe failed to publish Wilhelm Meister's Apprenticeship. |
| 8585 | REFUTED | SUPPORTED | Tiber Oil Field is operated by Pfizer. |
| 8603 | REFUTED | NOT ENOUGH INFORMATION | The Indian Institute of Management Bangalore refuses to offer a doctoral program. |
| 8725 | REFUTED | SUPPORTED | Sonny & Cher are rock singers. |
| 8830 | REFUTED | NOT ENOUGH INFORMATION | Helmand Province is far away from Lashkargah. |
| 8834 | REFUTED | NOT ENOUGH INFORMATION | Antigua and Barbuda was named for the first and only time in 1450. |
| 8847 | REFUTED | SUPPORTED | Ludwig van Beethoven was a saxophonist. |
| 8867 | REFUTED | NOT ENOUGH INFORMATION | Moscovium is atomic weight 115. |
| 8926 | REFUTED | SUPPORTED | Advertising is rarely a visual form of marketing communication. |
| 9075 | REFUTED | SUPPORTED | Eric Church was born May 3, 1975. |
| 9152 | REFUTED | SUPPORTED | Riz Ahmed acted in an Portuguese miniseries based on a British television series. |
| 9282 | SUPPORTED | NOT ENOUGH INFORMATION | Gamal Abdel Nasser's funeral was in the capital of Egypt. |
| 9643 | REFUTED | SUPPORTED | Sledgehammer was first performed live in America. |
| 9713 | REFUTED | SUPPORTED | Recovery is Eminem's first album. |
| 9825 | REFUTED | SUPPORTED | SZA's music does not combine elements from different genres. |
| 9833 | REFUTED | NOT ENOUGH INFORMATION | The State of Palestine claims East Lebanon. |

*6667 more incorrect cases are not listed here; all are in `errors.csv`.*

Full raw outputs and error details are preserved in `predictions.jsonl` and `errors.csv`.

## Limitations

The gold sample contains no NOT ENOUGH INFO cases, so this run does not measure gold NEI accuracy. It does not test evidence retrieval, robustness, stability, sensitivity, calibration, or real-world fact-checking ability. It measures only classification of FEVER SUPPORTS/REFUTES cases supplied with valid gold evidence.

## Result excluding flagged records

The FEVER data audit flagged 68 of these claims as possibly
inconsistent (conflicting_duplicate_label: 67, normalized_claim_conflict: 1). The main result above keeps them. This section removes them, as the
audit decided. Flags are possible label inconsistencies, not proven errors, and no label was
changed.

| | All cases | Excluding flagged |
|---|---:|---:|
| Cases | 13332 | 13264 |
| Correct | 6465 | 6447 |
| Overall accuracy | 0.485 | 0.486 |
| SUPPORTED accuracy | 0.950 | 0.951 |
| REFUTED accuracy | 0.020 | 0.020 |
