# FEVER gold-evidence baseline: phi4-mini:3.8b

## Purpose

This baseline tests whether `phi4-mini:3.8b` classifies FEVER SUPPORTS and REFUTES claims when supplied with one complete gold Wikipedia evidence set. It uses only the supplied evidence in the prompt.

## Dataset and sample construction

- Source: `data/fever/raw/shared_task_dev.jsonl`
- Prepared data: `data/fever/full/fever_dev_full.jsonl`
- Cases: 13332 (6666 SUPPORTED, 6666 REFUTED)
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

- Correct: 7981 / 13332
- Overall accuracy: 0.599
- SUPPORTED accuracy: 0.964
- REFUTED accuracy: 0.233
- Invalid-format outputs: 3 (0.023%)
- Inference failures: 0
- Prediction counts: SUPPORTED: 9159, REFUTED: 1596, NOT ENOUGH INFORMATION: 2574, INVALID: 3

## Confusion matrix

| Gold | SUPPORTED | REFUTED | NOT ENOUGH INFORMATION | INVALID |
|---|---:|---:|---:|---:|
| SUPPORTED | 6428 | 43 | 195 | 0 |
| REFUTED | 2731 | 1553 | 2379 | 3 |

## Incorrect cases

| FEVER ID | Gold | Prediction | Claim |
|---:|---|---|---|
| 12 | REFUTED | NOT ENOUGH INFORMATION | Carlos Santana disbanded Santana in 1965. |
| 304 | REFUTED | SUPPORTED | Aphrodite is unmarried. |
| 305 | SUPPORTED | NOT ENOUGH INFORMATION | Ragtime features Samuel L. Jackson. |
| 316 | REFUTED | SUPPORTED | Yemen has an area of 250,750 square miles. |
| 354 | REFUTED | SUPPORTED | LinkedIn is available in zero languages as of 2013. |
| 369 | REFUTED | SUPPORTED | The State of Palestine claims a territory in Western Asia. |
| 370 | REFUTED | NOT ENOUGH INFORMATION | The State of Palestine claims a territory in East Asia. |
| 422 | REFUTED | NOT ENOUGH INFORMATION | John Krasinski is an astronaut. |
| 446 | REFUTED | NOT ENOUGH INFORMATION | Benjamin Franklin earned the title of The Dead Body. |
| 499 | REFUTED | NOT ENOUGH INFORMATION | Paul Nicholls is a singer. |
| 519 | SUPPORTED | NOT ENOUGH INFORMATION | The Hundred Years' War includes the Civil War. |
| 529 | REFUTED | SUPPORTED | Musala is 2,525 meters. |
| 531 | REFUTED | SUPPORTED | Keegan-Michael Key played zero characters. |
| 619 | REFUTED | SUPPORTED | Hannah and Her Sisters' ending is unrelated to Thanksgiving. |
| 637 | REFUTED | NOT ENOUGH INFORMATION | Liberal Socialists were the only people to form The Blue Dog Coalition. |
| 787 | REFUTED | NOT ENOUGH INFORMATION | Shawn Carlson is only German. |
| 850 | REFUTED | NOT ENOUGH INFORMATION | John S. McCain Jr. graduated in 1901. |
| 972 | REFUTED | SUPPORTED | Stan Beeman is a fictional setting. |
| 976 | REFUTED | SUPPORTED | Doug Petrie has yet to appear on the web. |
| 1032 | REFUTED | SUPPORTED | Yin and yang do not have a portion of the opposite element in each section. |
| 1076 | SUPPORTED | NOT ENOUGH INFORMATION | James Jones missed the finals once. |
| 1094 | REFUTED | SUPPORTED | Henry Cavill is not part of the DC Extended Universe. |
| 1136 | REFUTED | NOT ENOUGH INFORMATION | Tiber Oil Field was discovered in France. |
| 1180 | REFUTED | SUPPORTED | Renato Balestra refused to study for an engineering degree. |
| 1188 | REFUTED | NOT ENOUGH INFORMATION | Pearl Jam is a type of dressing. |
| 1220 | SUPPORTED | REFUTED | The Vandals were an East Germanic tribe or group of tribes in the third century AD. |
| 1244 | REFUTED | SUPPORTED | The Nottingham Forest F.C. is based in Lincolnshire, England. |
| 1313 | SUPPORTED | NOT ENOUGH INFORMATION | English people are descended from the Saxons, |
| 1362 | REFUTED | SUPPORTED | Deepika Padukone only acts in tragedies. |
| 1481 | REFUTED | SUPPORTED | EA Black Box was founded in 1999. |
| 1598 | REFUTED | SUPPORTED | The Hunger Games are a film trilogy. |
| 1746 | REFUTED | SUPPORTED | Sheryl Lee has yet to work with Woody Allen. |
| 1799 | REFUTED | NOT ENOUGH INFORMATION | Aphrodite plays tennis with Hephaestus. |
| 1830 | SUPPORTED | REFUTED | DNA is an album. |
| 1831 | REFUTED | NOT ENOUGH INFORMATION | DNA is a single. |
| 1931 | SUPPORTED | NOT ENOUGH INFORMATION | Team Fortress 2 development was led by John Cook. |
| 2043 | REFUTED | SUPPORTED | Pakistan sits outside of Central and Western Asia. |
| 2231 | SUPPORTED | NOT ENOUGH INFORMATION | Mick Thomson was born in Iowa. |
| 2359 | REFUTED | NOT ENOUGH INFORMATION | Virginia keeps all computer chips manufactured within the state for use in Virginian electronics. |
| 2372 | REFUTED | SUPPORTED | Guillermo del Toro works in the coal industry. |
| 2374 | REFUTED | SUPPORTED | Anthony Scaramucci is not a political figure. |
| 2421 | REFUTED | NOT ENOUGH INFORMATION | Beaverton, Oregon's city center is in the Grand Canyon. |
| 2466 | REFUTED | NOT ENOUGH INFORMATION | Cheese in the Trap (TV series) stars Batman. |
| 2521 | REFUTED | SUPPORTED | Akkineni Nageswara Rao plays in Daag. |
| 2531 | REFUTED | NOT ENOUGH INFORMATION | Dopamine prevents neuromodulation. |
| 2984 | REFUTED | SUPPORTED | Kevin Costner refused the role of Devil Anse Hatfield in 2012. |
| 3008 | REFUTED | NOT ENOUGH INFORMATION | Jamie Lee Curtis' mother is a Palestinian. |
| 3019 | REFUTED | SUPPORTED | The horse was domesticated on a broad scale by 3103 BC. |
| 3033 | REFUTED | NOT ENOUGH INFORMATION | The Gifted is from outside of America. |
| 3063 | SUPPORTED | REFUTED | Ronaldinho has played 90 matches. |
| 3111 | REFUTED | SUPPORTED | Luis Fonsi was born in the eighties. |
| 3202 | REFUTED | SUPPORTED | Zoe Saldana was born in May of 1978. |
| 3210 | REFUTED | SUPPORTED | The Challenge XXX: Dirty 30 is the thirteenth season of a reality game show. |
| 3305 | REFUTED | SUPPORTED | Julianne Moore was not in the television series As the World Turns. |
| 3338 | REFUTED | NOT ENOUGH INFORMATION | Mike Huckabee has been mayor of Pittsburgh for his entire political career. |
| 3413 | REFUTED | NOT ENOUGH INFORMATION | Samsung was separated into ten business groups in 1987. |
| 3483 | REFUTED | SUPPORTED | Brie Larson was denied any role in the movie Short Term 12. |
| 3494 | REFUTED | SUPPORTED | English people are unconnected to the Saxons. |
| 3518 | REFUTED | SUPPORTED | Exercise is bad for heart health. |
| 3546 | REFUTED | SUPPORTED | Villa Park hosted a tennis match on August 12, 2012. |
| 3556 | REFUTED | SUPPORTED | A River Runs Through It has lost every nomination. |
| 3562 | REFUTED | SUPPORTED | Fargo (season 3) has yet to air. |
| 3588 | REFUTED | NOT ENOUGH INFORMATION | Jed Whedon has only ever lived on Mars. |
| 3598 | REFUTED | SUPPORTED | Ice-T began his career as a rapper in 1984. |
| 3638 | REFUTED | SUPPORTED | There are currently 15,882,417 Mormon members as of 2007. |
| 3656 | REFUTED | NOT ENOUGH INFORMATION | John DiMaggio refused to work on any shows from Cartoon Network. |
| 3682 | REFUTED | SUPPORTED | English people are disconnected from the Jutes and Frisians. |
| 3718 | REFUTED | SUPPORTED | Victoria Palace Theatre is outside of Victoria Street. |
| 3783 | REFUTED | SUPPORTED | The American actor that plays Chumlee was born on the 10th. |
| 3795 | REFUTED | NOT ENOUGH INFORMATION | The Fame has sold over 4.5 billion copies. |
| 3809 | REFUTED | NOT ENOUGH INFORMATION | Sonny & Cher began in the mid-1950's. |
| 4058 | REFUTED | SUPPORTED | Balibo (film) involves zero journalist. |
| 4091 | REFUTED | SUPPORTED | Zac Efron's full name is Zac David Alexander Efron. |
| 4100 | SUPPORTED | NOT ENOUGH INFORMATION | Sonny & Cher began in the mid-1930's. |
| 4172 | REFUTED | NOT ENOUGH INFORMATION | A monster is always a person. |
| 4251 | REFUTED | NOT ENOUGH INFORMATION | Solanum contains the leopards. |
| 4280 | REFUTED | SUPPORTED | Ekta Kapoor did not produce an Indian soap opera. |
| 4500 | REFUTED | NOT ENOUGH INFORMATION | Sledgehammer was first performed live in 2004. |
| 4537 | SUPPORTED | REFUTED | Eric Church died on May 3, 1977. |
| 4675 | REFUTED | SUPPORTED | Aphrodite is not married to Hephaestus. |
| 4713 | REFUTED | NOT ENOUGH INFORMATION | Tool has won three Oscars. |
| 4728 | REFUTED | NOT ENOUGH INFORMATION | James VI and I was a major adversary of a single parliament for Scotland. |
| 4818 | REFUTED | SUPPORTED | English people are descended from the Romans. |
| 4819 | REFUTED | SUPPORTED | English people are unconnected to the Romans. |
| 4916 | REFUTED | SUPPORTED | Paul Nicholls created Joe Wicks. |
| 4987 | REFUTED | SUPPORTED | Southern Hospitality peaked at number 21 on the Top 40. |
| 5076 | REFUTED | NOT ENOUGH INFORMATION | The State of Palestine claims Algeria. |
| 5080 | REFUTED | NOT ENOUGH INFORMATION | The Times began printing before or during 1785. |
| 5092 | SUPPORTED | NOT ENOUGH INFORMATION | Richard Kuklinski led a double life. |
| 5093 | SUPPORTED | NOT ENOUGH INFORMATION | Richard Kuklinski led other lives. |
| 5127 | REFUTED | SUPPORTED | Kellyanne Conway did not claim Michael Flynn had the full confidence of the president hours before he was dismissed. |
| 5169 | REFUTED | SUPPORTED | Jonah Hill is not a screenwriter. |
| 5177 | REFUTED | SUPPORTED | The Armenian Genocide also fails to be known as the Armenian Holocaust. |
| 5314 | REFUTED | NOT ENOUGH INFORMATION | Younger is a town. |
| 5346 | REFUTED | SUPPORTED | Ripon College is in California. |
| 5421 | REFUTED | SUPPORTED | Stephen Hillenburg developed an interest in everything but art. |
| 5454 | REFUTED | NOT ENOUGH INFORMATION | Jack Falahee's middle name is Bryan. |
| 5466 | REFUTED | NOT ENOUGH INFORMATION | Due Date was shot in Los Angeles, Alabama. |
| 5545 | REFUTED | SUPPORTED | Atlanta metropolitan area is located in north Florida. |
| 5599 | REFUTED | NOT ENOUGH INFORMATION | San Francisco Bay Area is home to approximately 10 million people. |
| 5719 | REFUTED | NOT ENOUGH INFORMATION | Qui-Gon Jinn is married to Liam Neeson. |
| 5738 | REFUTED | NOT ENOUGH INFORMATION | Melancholia was directed by a Japanese sushi chef. |
| 5744 | SUPPORTED | NOT ENOUGH INFORMATION | One Flew Over the Cuckoo's Nest won Best Actor at the Academy Awards. |
| 5745 | REFUTED | NOT ENOUGH INFORMATION | One Flew Over the Cuckoo's Nest only won one Oscar, for sound design. |
| 5766 | REFUTED | NOT ENOUGH INFORMATION | Hot Right Now is from the author Bob Stanley. |
| 5781 | REFUTED | NOT ENOUGH INFORMATION | Raja Hindustani was directed solely by Mariah Carey. |
| 5900 | REFUTED | NOT ENOUGH INFORMATION | The Paper stars a horse. |
| 5980 | REFUTED | NOT ENOUGH INFORMATION | Camden, New Jersey is the home of Yale School of Medicine. |
| 6032 | REFUTED | NOT ENOUGH INFORMATION | Aruba is the only ABC Island. |
| 6275 | REFUTED | NOT ENOUGH INFORMATION | Recovery is an album focusing on themes of isolation and addiction. |
| 6317 | REFUTED | NOT ENOUGH INFORMATION | Loretta Sanchez was born on January 7, 1950. |
| 6334 | REFUTED | SUPPORTED | The Challenge XXX: Dirty 30 is not a season of a reality game show. |
| 6390 | REFUTED | NOT ENOUGH INFORMATION | Spider-Man 2 is a store. |
| 6414 | REFUTED | NOT ENOUGH INFORMATION | The Godfather Part II is a political party. |
| 6462 | REFUTED | SUPPORTED | Emperor Norton is not immortalized by authors from different countries. |
| 6668 | REFUTED | NOT ENOUGH INFORMATION | Louis Malle worked solely in pharmaceuticals. |
| 6743 | REFUTED | SUPPORTED | Jens Stoltenberg was the first Prime Minister of Norway from 2005 to 2013. |
| 6834 | REFUTED | SUPPORTED | Chris Benoit's middle name is Jason. |
| 6876 | SUPPORTED | NOT ENOUGH INFORMATION | The Wallace  is historically inaccurate. |
| 6925 | REFUTED | NOT ENOUGH INFORMATION | Ed Gein committed crimes around Milwaukee, Wisconsin. |
| 7000 | REFUTED | NOT ENOUGH INFORMATION | Moves Like Jagger is the lead single on Overexposed. |
| 7041 | REFUTED | SUPPORTED | Kareem Abdul-Jabbar is not ranked 3rd all-time in blocks. |
| 7105 | REFUTED | NOT ENOUGH INFORMATION | Zac Efron is Somali. |
| 7138 | REFUTED | SUPPORTED | The Tenth Doctor travels alone during the time period 2009-2010. |
| 7183 | REFUTED | SUPPORTED | Sands Hotel and Casino closed in 1952. |
| 7278 | SUPPORTED | NOT ENOUGH INFORMATION | Exit the King is by man. |
| 7286 | REFUTED | SUPPORTED | Croatia is incapable of rendering services. |
| 7336 | REFUTED | SUPPORTED | Lizzy Caplan is incapable of appearing in multiple television shows. |
| 7436 | REFUTED | NOT ENOUGH INFORMATION | Simon Pegg was born on February 14th, 1860. |
| 7625 | REFUTED | NOT ENOUGH INFORMATION | Ghost, the film, was released in 1994. |
| 7697 | REFUTED | SUPPORTED | Riddick is in a 2023 film. |
| 7728 | REFUTED | SUPPORTED | Hinduism has zero textual resources. |
| 7744 | REFUTED | NOT ENOUGH INFORMATION | Mount Hood is the sixth highest point in its mountain range. |
| 7758 | REFUTED | NOT ENOUGH INFORMATION | Ragtime was released in 1965. |
| 7811 | REFUTED | SUPPORTED | Adobe Photoshop is incapable of being part of the Adobe Systems. |
| 7877 | REFUTED | NOT ENOUGH INFORMATION | The ruins of the ancient Roman town of Herculaneum were teleported to a location outside Naples. |
| 8022 | REFUTED | SUPPORTED | Colin Kaepernick was not backup to Alex Smith. |
| 8167 | REFUTED | SUPPORTED | Caesar is a 1937 adaptation of The Tempest. |
| 8298 | REFUTED | NOT ENOUGH INFORMATION | Simon Pegg is only a banker. |
| 8304 | REFUTED | SUPPORTED | Therasia is outside of the Greek Cyclades. |
| 8403 | SUPPORTED | NOT ENOUGH INFORMATION | US Airways Flight 1549 had 155 people on board. |
| 8404 | REFUTED | NOT ENOUGH INFORMATION | US Airways Flight 1549 had 1978 people on board. |
| 8538 | REFUTED | NOT ENOUGH INFORMATION | Hermit crabs are arachnids. |
| 8546 | REFUTED | SUPPORTED | Prague's historic center is not on UNESCO's list of World Heritage Sites. |
| 8577 | REFUTED | SUPPORTED | The Dodo has been depicted with a naked head and arm. |
| 8581 | REFUTED | SUPPORTED | Johann Wolfgang von Goethe failed to publish Wilhelm Meister's Apprenticeship. |
| 8603 | REFUTED | SUPPORTED | The Indian Institute of Management Bangalore refuses to offer a doctoral program. |
| 8725 | REFUTED | SUPPORTED | Sonny & Cher are rock singers. |
| 8830 | REFUTED | NOT ENOUGH INFORMATION | Helmand Province is far away from Lashkargah. |
| 8847 | REFUTED | NOT ENOUGH INFORMATION | Ludwig van Beethoven was a saxophonist. |
| 8864 | SUPPORTED | REFUTED | Great white sharks have never killed people. |
| 8867 | REFUTED | SUPPORTED | Moscovium is atomic weight 115. |
| 8926 | REFUTED | SUPPORTED | Advertising is rarely a visual form of marketing communication. |
| 9075 | REFUTED | SUPPORTED | Eric Church was born May 3, 1975. |
| 9101 | REFUTED | SUPPORTED | Key & Peele has only won one Primetime Emmy Award. |
| 9152 | REFUTED | NOT ENOUGH INFORMATION | Riz Ahmed acted in an Portuguese miniseries based on a British television series. |
| 9643 | REFUTED | NOT ENOUGH INFORMATION | Sledgehammer was first performed live in America. |
| 9817 | SUPPORTED | NOT ENOUGH INFORMATION | Shannon Lee is a citizen. |
| 9825 | REFUTED | SUPPORTED | SZA's music does not combine elements from different genres. |
| 9833 | REFUTED | NOT ENOUGH INFORMATION | The State of Palestine claims East Lebanon. |
| 9849 | REFUTED | SUPPORTED | Michigan is not a U.S. state. |
| 9860 | REFUTED | NOT ENOUGH INFORMATION | Bob Ross created the Internet. |
| 9913 | REFUTED | NOT ENOUGH INFORMATION | Pirates of the Caribbean was added to Magic Kingdom in 1993. |
| 9915 | REFUTED | SUPPORTED | The Vandals were North Germanic tribes or groups of tribes. |
| 9945 | REFUTED | SUPPORTED | The Armenian Genocide took place outside the Ottoman Empire and the Republic of Turkey. |
| 10052 | REFUTED | NOT ENOUGH INFORMATION | Stomp the Yard is a clothing brand. |
| 10103 | REFUTED | NOT ENOUGH INFORMATION | Tiber Oil Field is operated by BP as of 2008. |
| 10146 | REFUTED | NOT ENOUGH INFORMATION | TakePart is a division of 20th Century Fox. |
| 10158 | REFUTED | NOT ENOUGH INFORMATION | Emma Watson was killed in 1990. |
| 10262 | REFUTED | NOT ENOUGH INFORMATION | The English Wikipedia is an edition of a free online chat room. |
| 10272 | REFUTED | SUPPORTED | Taran Killam isn't a writer. |
| 10328 | REFUTED | NOT ENOUGH INFORMATION | Sidse Babett Knudsen graduated on November 22nd, 1968. |
| 10357 | REFUTED | NOT ENOUGH INFORMATION | Brian Michael Bendis has spent his entire career as a professional baseball player. |
| 10379 | REFUTED | SUPPORTED | The horse was not called a Eohippus when it had multiple toes. |
| 10455 | REFUTED | NOT ENOUGH INFORMATION | The Cyclades are in the Dead Sea. |
| 10530 | REFUTED | SUPPORTED | Fuller House (TV series) is unable to be a family sitcom. |
| 10567 | REFUTED | SUPPORTED | Google Search shows no home information. |
| 10635 | REFUTED | SUPPORTED | Unforced labor is a reason for human trafficking. |
| 10670 | REFUTED | SUPPORTED | Flaked was cancelled in 2016. |
| 10689 | REFUTED | SUPPORTED | Cleopatre premiered in a manner. |
| 10707 | REFUTED | SUPPORTED | Chris Kyle was born on June 8, 1974. |
| 10712 | REFUTED | NOT ENOUGH INFORMATION | Speech recognition is ignored in the computer science fields. |
| 10765 | REFUTED | NOT ENOUGH INFORMATION | Mount Hood is the highest point of a roller coaster. |
| 10798 | REFUTED | SUPPORTED | Carlos Santana formed Santana in Florida. |
| 10810 | REFUTED | NOT ENOUGH INFORMATION | Halsey signed her first recording contract in 1914. |
| 10838 | REFUTED | NOT ENOUGH INFORMATION | Saxony is a horse. |
| 11091 | REFUTED | SUPPORTED | Hinduism contains zero philosophies. |
| 11194 | REFUTED | NOT ENOUGH INFORMATION | Ashley Cole graduated college on December 20, 1980. |
| 11290 | REFUTED | SUPPORTED | Talulah Riley did not appear in The Boat That Rocked. |
| 11352 | REFUTED | NOT ENOUGH INFORMATION | The Hit was released on a mural. |
| 11432 | REFUTED | SUPPORTED | The first inauguration of Bill Clinton made him the 50th President of the United States. |
| 11441 | REFUTED | SUPPORTED | The Bassoon King's working title was My Life in Art, Faith, and Idiocy. |
| 11443 | REFUTED | SUPPORTED | David Attenborough served as director of BBC Two. |
| 11479 | REFUTED | NOT ENOUGH INFORMATION | Kenny Chesney graduated from high school on March 26th, 1968. |
| 11554 | SUPPORTED | NOT ENOUGH INFORMATION | Pharrell Williams plays an instrument. |
| 11570 | REFUTED | SUPPORTED | José Ferrer only won a Tony Award in 1930. |
| 11572 | REFUTED | SUPPORTED | Samsung was separated into four business groups in Italy. |
| 11666 | REFUTED | NOT ENOUGH INFORMATION | A near-Earth object is not a dead body. |
| 11823 | REFUTED | SUPPORTED | Saxony only has a population of 2 million people. |
| 11914 | REFUTED | NOT ENOUGH INFORMATION | Temple Grandin features only male characters. |

*5151 more incorrect cases are not listed here; all are in `errors.csv`.*

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
| Correct | 7981 | 7952 |
| Overall accuracy | 0.599 | 0.600 |
| SUPPORTED accuracy | 0.964 | 0.964 |
| REFUTED accuracy | 0.233 | 0.233 |
