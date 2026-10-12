# FEVER gold-evidence baseline: llama3.2:3b

## Purpose

This baseline tests whether `llama3.2:3b` classifies FEVER SUPPORTS and REFUTES claims when supplied with one complete gold Wikipedia evidence set. It uses only the supplied evidence in the prompt.

## Dataset and sample construction

- Source: `data/fever/raw/shared_task_dev.jsonl`
- Prepared data: `data/fever/full/fever_dev_full.jsonl`
- Cases: 13332 (6666 SUPPORTED, 6666 REFUTED)
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

- Correct: 9369 / 13332
- Overall accuracy: 0.703
- SUPPORTED accuracy: 0.807
- REFUTED accuracy: 0.598
- Invalid-format outputs: 0 (0.000%)
- Inference failures: 0
- Prediction counts: SUPPORTED: 6279, REFUTED: 4935, NOT ENOUGH INFORMATION: 2118, INVALID: 0

## Confusion matrix

| Gold | SUPPORTED | REFUTED | NOT ENOUGH INFORMATION | INVALID |
|---|---:|---:|---:|---:|
| SUPPORTED | 5381 | 947 | 338 | 0 |
| REFUTED | 898 | 3988 | 1780 | 0 |

## Incorrect cases

| FEVER ID | Gold | Prediction | Claim |
|---:|---|---|---|
| 161 | SUPPORTED | NOT ENOUGH INFORMATION | Yin and yang is a taoist concept. |
| 223 | SUPPORTED | REFUTED | Advertising is an audio form of marketing communication. |
| 305 | SUPPORTED | REFUTED | Ragtime features Samuel L. Jackson. |
| 315 | SUPPORTED | REFUTED | Yemen has an area of 203,850 square miles. |
| 316 | REFUTED | NOT ENOUGH INFORMATION | Yemen has an area of 250,750 square miles. |
| 327 | SUPPORTED | NOT ENOUGH INFORMATION | Donald Duck typically wears a cap. |
| 369 | REFUTED | SUPPORTED | The State of Palestine claims a territory in Western Asia. |
| 370 | REFUTED | NOT ENOUGH INFORMATION | The State of Palestine claims a territory in East Asia. |
| 420 | REFUTED | NOT ENOUGH INFORMATION | Toy Story 2 was released by Universal Pictures. |
| 422 | REFUTED | NOT ENOUGH INFORMATION | John Krasinski is an astronaut. |
| 446 | REFUTED | NOT ENOUGH INFORMATION | Benjamin Franklin earned the title of The Dead Body. |
| 519 | SUPPORTED | REFUTED | The Hundred Years' War includes the Civil War. |
| 531 | REFUTED | NOT ENOUGH INFORMATION | Keegan-Michael Key played zero characters. |
| 532 | SUPPORTED | REFUTED | The New England Patriots won five Super Bowls. |
| 607 | SUPPORTED | REFUTED | Cher was united in marriage to Sonny Bono. |
| 637 | REFUTED | NOT ENOUGH INFORMATION | Liberal Socialists were the only people to form The Blue Dog Coalition. |
| 673 | REFUTED | SUPPORTED | Saxony is only a town. |
| 833 | SUPPORTED | REFUTED | TakePart is the digital division of Participant Media. |
| 850 | REFUTED | NOT ENOUGH INFORMATION | John S. McCain Jr. graduated in 1901. |
| 915 | SUPPORTED | REFUTED | Pulmonary embolism is indicated by rapid breathing. |
| 963 | SUPPORTED | NOT ENOUGH INFORMATION | Clueless (film) was followed by a spin-off television sitcom. |
| 1076 | SUPPORTED | NOT ENOUGH INFORMATION | James Jones missed the finals once. |
| 1180 | REFUTED | NOT ENOUGH INFORMATION | Renato Balestra refused to study for an engineering degree. |
| 1188 | REFUTED | NOT ENOUGH INFORMATION | Pearl Jam is a type of dressing. |
| 1220 | SUPPORTED | REFUTED | The Vandals were an East Germanic tribe or group of tribes in the third century AD. |
| 1244 | REFUTED | SUPPORTED | The Nottingham Forest F.C. is based in Lincolnshire, England. |
| 1272 | SUPPORTED | REFUTED | Renato Balestra came from an architect family. |
| 1313 | SUPPORTED | NOT ENOUGH INFORMATION | English people are descended from the Saxons, |
| 1423 | SUPPORTED | REFUTED | Roberto Orci wrote Star Trek. |
| 1536 | SUPPORTED | NOT ENOUGH INFORMATION | Sledgehammer made an appearance on Good Morning America. |
| 1667 | SUPPORTED | REFUTED | Ed Wood is not about the eponymous filmmaker. |
| 1729 | SUPPORTED | REFUTED | Joni Mitchell released 17 albums of original songs. |
| 1746 | REFUTED | SUPPORTED | Sheryl Lee has yet to work with Woody Allen. |
| 1831 | REFUTED | NOT ENOUGH INFORMATION | DNA is a single. |
| 1917 | SUPPORTED | REFUTED | The current Duke of York has been unmarried since his 1996 divorce. |
| 1931 | SUPPORTED | NOT ENOUGH INFORMATION | Team Fortress 2 development was led by John Cook. |
| 2197 | SUPPORTED | REFUTED | Key & Peele has won two Primetime Emmy Awards. |
| 2359 | REFUTED | NOT ENOUGH INFORMATION | Virginia keeps all computer chips manufactured within the state for use in Virginian electronics. |
| 2372 | REFUTED | NOT ENOUGH INFORMATION | Guillermo del Toro works in the coal industry. |
| 2466 | REFUTED | NOT ENOUGH INFORMATION | Cheese in the Trap (TV series) stars Batman. |
| 2530 | SUPPORTED | NOT ENOUGH INFORMATION | Dopamine induces neuromodulation. |
| 2531 | REFUTED | NOT ENOUGH INFORMATION | Dopamine prevents neuromodulation. |
| 2536 | SUPPORTED | REFUTED | Paris (Paris Hilton album) incorporates elements of a music genre that originated in Jamaica in the late 1960s. |
| 2594 | SUPPORTED | REFUTED | Chris Eubank Jr. is an athlete. |
| 2699 | SUPPORTED | NOT ENOUGH INFORMATION | A River Runs Through It has won an Academy Award. |
| 2739 | SUPPORTED | REFUTED | Randy Savage can speak. |
| 2834 | SUPPORTED | REFUTED | Flaked was renewed for a six episode season. |
| 2937 | SUPPORTED | REFUTED | Legendary Entertainment is owned by the owner of Wanda Cinemas. |
| 2948 | REFUTED | SUPPORTED | The Mod Squad is an American movie. |
| 2984 | REFUTED | SUPPORTED | Kevin Costner refused the role of Devil Anse Hatfield in 2012. |
| 3019 | REFUTED | NOT ENOUGH INFORMATION | The horse was domesticated on a broad scale by 3103 BC. |
| 3063 | SUPPORTED | REFUTED | Ronaldinho has played 90 matches. |
| 3113 | SUPPORTED | REFUTED | Daggering is associated with dancehall music. |
| 3149 | REFUTED | NOT ENOUGH INFORMATION | No Country for Old Men starred Harrison Ford. |
| 3264 | SUPPORTED | REFUTED | Baloch people speak the Balochi language. |
| 3338 | REFUTED | NOT ENOUGH INFORMATION | Mike Huckabee has been mayor of Pittsburgh for his entire political career. |
| 3348 | SUPPORTED | REFUTED | Heaven Can Wait is a film adaptation of Harry Segall's play of the same name. |
| 3371 | REFUTED | NOT ENOUGH INFORMATION | Maria Theresa reigned for only one year. |
| 3413 | REFUTED | NOT ENOUGH INFORMATION | Samsung was separated into ten business groups in 1987. |
| 3432 | SUPPORTED | REFUTED | The year in which Noah Cyrus was born was 1992 or later. |
| 3475 | SUPPORTED | REFUTED | Sonny & Cher was made up of a couple. |
| 3483 | REFUTED | SUPPORTED | Brie Larson was denied any role in the movie Short Term 12. |
| 3518 | REFUTED | NOT ENOUGH INFORMATION | Exercise is bad for heart health. |
| 3546 | REFUTED | SUPPORTED | Villa Park hosted a tennis match on August 12, 2012. |
| 3556 | REFUTED | SUPPORTED | A River Runs Through It has lost every nomination. |
| 3588 | REFUTED | NOT ENOUGH INFORMATION | Jed Whedon has only ever lived on Mars. |
| 3598 | REFUTED | NOT ENOUGH INFORMATION | Ice-T began his career as a rapper in 1984. |
| 3638 | REFUTED | NOT ENOUGH INFORMATION | There are currently 15,882,417 Mormon members as of 2007. |
| 3668 | REFUTED | NOT ENOUGH INFORMATION | Andrew Moray refused to cede the territory he gained to John Balliol. |
| 3682 | REFUTED | SUPPORTED | English people are disconnected from the Jutes and Frisians. |
| 3761 | SUPPORTED | NOT ENOUGH INFORMATION | The Premier League Asia Trophy is takes place every two years in Asia. |
| 3794 | SUPPORTED | NOT ENOUGH INFORMATION | The Fame has sold over 4.5 million copies. |
| 3795 | REFUTED | NOT ENOUGH INFORMATION | The Fame has sold over 4.5 billion copies. |
| 3809 | REFUTED | NOT ENOUGH INFORMATION | Sonny & Cher began in the mid-1950's. |
| 4041 | SUPPORTED | REFUTED | Jayasudha stars in Daag. |
| 4100 | SUPPORTED | NOT ENOUGH INFORMATION | Sonny & Cher began in the mid-1930's. |
| 4132 | SUPPORTED | REFUTED | A.J. Styles is a heavyweight. |
| 4172 | REFUTED | NOT ENOUGH INFORMATION | A monster is always a person. |
| 4246 | REFUTED | SUPPORTED | Meteora is a country album. |
| 4251 | REFUTED | NOT ENOUGH INFORMATION | Solanum contains the leopards. |
| 4391 | SUPPORTED | REFUTED | The Greek language is spoken in Albania. |
| 4500 | REFUTED | NOT ENOUGH INFORMATION | Sledgehammer was first performed live in 2004. |
| 4537 | SUPPORTED | REFUTED | Eric Church died on May 3, 1977. |
| 4595 | SUPPORTED | REFUTED | The Burj Khalifa contains 8 escalators. |
| 4635 | SUPPORTED | NOT ENOUGH INFORMATION | 59 was the debut number of Shut Up. |
| 4713 | REFUTED | NOT ENOUGH INFORMATION | Tool has won three Oscars. |
| 4728 | REFUTED | NOT ENOUGH INFORMATION | James VI and I was a major adversary of a single parliament for Scotland. |
| 4737 | SUPPORTED | REFUTED | Sausage Party was directed by Greg Tiernan and Conrad Vernon. |
| 4738 | SUPPORTED | REFUTED | Greg Tiernan and Conrad Vernon directed Sausage Party. |
| 4807 | SUPPORTED | REFUTED | Dark matter is distinguishable from dark energy. |
| 4818 | REFUTED | NOT ENOUGH INFORMATION | English people are descended from the Romans. |
| 4863 | SUPPORTED | REFUTED | Janelle Monáe's full name is Janelle Monáe Robinson. |
| 5076 | REFUTED | NOT ENOUGH INFORMATION | The State of Palestine claims Algeria. |
| 5093 | SUPPORTED | NOT ENOUGH INFORMATION | Richard Kuklinski led other lives. |
| 5136 | SUPPORTED | REFUTED | The Greek language is spoken in Italy. |
| 5177 | REFUTED | SUPPORTED | The Armenian Genocide also fails to be known as the Armenian Holocaust. |
| 5314 | REFUTED | SUPPORTED | Younger is a town. |
| 5421 | REFUTED | NOT ENOUGH INFORMATION | Stephen Hillenburg developed an interest in everything but art. |
| 5514 | REFUTED | NOT ENOUGH INFORMATION | Shadowhunters premiered in 2018. |
| 5578 | SUPPORTED | REFUTED | Australia (2008 film) production took place in Kununurra. |
| 5579 | SUPPORTED | REFUTED | Australia (2008 film) production took place in Bowen. |
| 5599 | REFUTED | NOT ENOUGH INFORMATION | San Francisco Bay Area is home to approximately 10 million people. |
| 5689 | REFUTED | NOT ENOUGH INFORMATION | Amyotrophic lateral sclerosis is a cure. |
| 5744 | SUPPORTED | REFUTED | One Flew Over the Cuckoo's Nest won Best Actor at the Academy Awards. |
| 5900 | REFUTED | NOT ENOUGH INFORMATION | The Paper stars a horse. |
| 5980 | REFUTED | NOT ENOUGH INFORMATION | Camden, New Jersey is the home of Yale School of Medicine. |
| 6032 | REFUTED | NOT ENOUGH INFORMATION | Aruba is the only ABC Island. |
| 6275 | REFUTED | NOT ENOUGH INFORMATION | Recovery is an album focusing on themes of isolation and addiction. |
| 6390 | REFUTED | NOT ENOUGH INFORMATION | Spider-Man 2 is a store. |
| 6414 | REFUTED | NOT ENOUGH INFORMATION | The Godfather Part II is a political party. |
| 6807 | SUPPORTED | REFUTED | Camden, New Jersey is the home of a school that was founded as the South Jersey Law School in 1926. |
| 6810 | SUPPORTED | REFUTED | The Hundred Years' War includes the Edwardian Era War and was the first phase in the war. |
| 6876 | SUPPORTED | REFUTED | The Wallace  is historically inaccurate. |
| 6893 | SUPPORTED | REFUTED | Underdog features the voice talents of Amy Adams. |
| 6925 | REFUTED | NOT ENOUGH INFORMATION | Ed Gein committed crimes around Milwaukee, Wisconsin. |
| 7012 | SUPPORTED | REFUTED | Penélope Cruz has modeled for Mango. |
| 7032 | SUPPORTED | REFUTED | The left hand side of the Trebbia River is the location of the Battle of the Trebia. |
| 7041 | REFUTED | NOT ENOUGH INFORMATION | Kareem Abdul-Jabbar is not ranked 3rd all-time in blocks. |
| 7105 | REFUTED | NOT ENOUGH INFORMATION | Zac Efron is Somali. |
| 7138 | REFUTED | SUPPORTED | The Tenth Doctor travels alone during the time period 2009-2010. |
| 7183 | REFUTED | SUPPORTED | Sands Hotel and Casino closed in 1952. |
| 7278 | SUPPORTED | REFUTED | Exit the King is by man. |
| 7286 | REFUTED | NOT ENOUGH INFORMATION | Croatia is incapable of rendering services. |
| 7697 | REFUTED | NOT ENOUGH INFORMATION | Riddick is in a 2023 film. |
| 7744 | REFUTED | NOT ENOUGH INFORMATION | Mount Hood is the sixth highest point in its mountain range. |
| 7757 | SUPPORTED | REFUTED | Ragtime is the final film appearance of James Cagney. |
| 7768 | SUPPORTED | REFUTED | Tottenham Hotspur F.C. is British. |
| 7877 | REFUTED | NOT ENOUGH INFORMATION | The ruins of the ancient Roman town of Herculaneum were teleported to a location outside Naples. |
| 7922 | SUPPORTED | REFUTED | Talulah Riley appeared in Pride & Prejudice. |
| 8067 | SUPPORTED | REFUTED | Beaverton, Oregon is a city. |
| 8167 | REFUTED | SUPPORTED | Caesar is a 1937 adaptation of The Tempest. |
| 8190 | SUPPORTED | REFUTED | James, Viscount Severn is the only son of Prince Edward and Sophie, Countess of Wessex. |
| 8219 | SUPPORTED | REFUTED | The Danish language is spoken in Spain. |
| 8298 | REFUTED | NOT ENOUGH INFORMATION | Simon Pegg is only a banker. |
| 8403 | SUPPORTED | REFUTED | US Airways Flight 1549 had 155 people on board. |
| 8404 | REFUTED | NOT ENOUGH INFORMATION | US Airways Flight 1549 had 1978 people on board. |
| 8505 | SUPPORTED | NOT ENOUGH INFORMATION | Jenna Jameson was photographed in the past. |
| 8538 | REFUTED | NOT ENOUGH INFORMATION | Hermit crabs are arachnids. |
| 8577 | REFUTED | SUPPORTED | The Dodo has been depicted with a naked head and arm. |
| 8603 | REFUTED | NOT ENOUGH INFORMATION | The Indian Institute of Management Bangalore refuses to offer a doctoral program. |
| 8725 | REFUTED | SUPPORTED | Sonny & Cher are rock singers. |
| 8727 | SUPPORTED | REFUTED | Sayyeshaa made her Bollywood debut in the 2016 film Shivaay. |
| 8829 | SUPPORTED | REFUTED | Helmand Province contains Lashkargah. |
| 8863 | SUPPORTED | REFUTED | Great white sharks have killed people. |
| 8864 | SUPPORTED | REFUTED | Great white sharks have never killed people. |
| 8867 | REFUTED | SUPPORTED | Moscovium is atomic weight 115. |
| 8870 | SUPPORTED | NOT ENOUGH INFORMATION | Pharmacology deals with the research, discovery, and characterization of turnips which show biological effects. |
| 8905 | SUPPORTED | REFUTED | The Raven (2012 film) was released in a sovereign country in western Europe in March of 2012. |
| 9119 | SUPPORTED | REFUTED | Mom (TV series) has been nominated at the Critics' Choice Television Awards. |
| 9234 | SUPPORTED | REFUTED | Wilhelmina Slater is portrayed in a comedy-drama television series. |
| 9410 | SUPPORTED | REFUTED | In the End was released in 2000. |
| 9503 | SUPPORTED | NOT ENOUGH INFORMATION | Rhythm Nation was covered by Pink. |
| 9504 | SUPPORTED | REFUTED | Pink covered the song Rhythm Nation. |
| 9629 | SUPPORTED | NOT ENOUGH INFORMATION | The 100 also follows the teens cousins on The Ark. |
| 9643 | REFUTED | NOT ENOUGH INFORMATION | Sledgehammer was first performed live in America. |
| 9693 | SUPPORTED | REFUTED | Touchscreens are used in game consoles. |
| 9817 | SUPPORTED | REFUTED | Shannon Lee is a citizen. |
| 9833 | REFUTED | NOT ENOUGH INFORMATION | The State of Palestine claims East Lebanon. |
| 9860 | REFUTED | NOT ENOUGH INFORMATION | Bob Ross created the Internet. |
| 9868 | SUPPORTED | REFUTED | Google Search shows home listings. |
| 9913 | REFUTED | NOT ENOUGH INFORMATION | Pirates of the Caribbean was added to Magic Kingdom in 1993. |
| 9938 | SUPPORTED | REFUTED | Byron Howard won a Golden Globe for Zootopia. |
| 10052 | REFUTED | NOT ENOUGH INFORMATION | Stomp the Yard is a clothing brand. |
| 10103 | REFUTED | NOT ENOUGH INFORMATION | Tiber Oil Field is operated by BP as of 2008. |
| 10146 | REFUTED | NOT ENOUGH INFORMATION | TakePart is a division of 20th Century Fox. |
| 10158 | REFUTED | NOT ENOUGH INFORMATION | Emma Watson was killed in 1990. |
| 10262 | REFUTED | SUPPORTED | The English Wikipedia is an edition of a free online chat room. |
| 10277 | SUPPORTED | REFUTED | Hush (2016 film) was produced by Trevor Macy. |
| 10278 | REFUTED | NOT ENOUGH INFORMATION | Hush (2016 film) was produced by children. |
| 10324 | SUPPORTED | REFUTED | Cleopatre is one of three operas and it is significant. |
| 10328 | REFUTED | SUPPORTED | Sidse Babett Knudsen graduated on November 22nd, 1968. |
| 10357 | REFUTED | NOT ENOUGH INFORMATION | Brian Michael Bendis has spent his entire career as a professional baseball player. |
| 10468 | SUPPORTED | REFUTED | PacSun sells accessories. |
| 10530 | REFUTED | NOT ENOUGH INFORMATION | Fuller House (TV series) is unable to be a family sitcom. |
| 10635 | REFUTED | SUPPORTED | Unforced labor is a reason for human trafficking. |
| 10671 | REFUTED | NOT ENOUGH INFORMATION | NXT Tag Team Championship is a singing talent competition. |
| 10689 | REFUTED | NOT ENOUGH INFORMATION | Cleopatre premiered in a manner. |
| 10698 | SUPPORTED | NOT ENOUGH INFORMATION | The dress was scientifically investigated. |
| 10712 | REFUTED | NOT ENOUGH INFORMATION | Speech recognition is ignored in the computer science fields. |
| 10765 | REFUTED | NOT ENOUGH INFORMATION | Mount Hood is the highest point of a roller coaster. |
| 10810 | REFUTED | NOT ENOUGH INFORMATION | Halsey signed her first recording contract in 1914. |
| 10942 | SUPPORTED | REFUTED | Speech recognition incorporates knowledge and research in the computer science fields. |
| 11091 | REFUTED | NOT ENOUGH INFORMATION | Hinduism contains zero philosophies. |
| 11237 | SUPPORTED | NOT ENOUGH INFORMATION | Acetylcholine induces neuromodulation. |
| 11432 | REFUTED | NOT ENOUGH INFORMATION | The first inauguration of Bill Clinton made him the 50th President of the United States. |
| 11441 | REFUTED | SUPPORTED | The Bassoon King's working title was My Life in Art, Faith, and Idiocy. |
| 11443 | REFUTED | SUPPORTED | David Attenborough served as director of BBC Two. |
| 11479 | REFUTED | SUPPORTED | Kenny Chesney graduated from high school on March 26th, 1968. |
| 11497 | SUPPORTED | REFUTED | Ashton Kutcher was in two films in 2005. |
| 11528 | SUPPORTED | REFUTED | Bones is loosely based on the life and writings of Kathy Reichs. |
| 11554 | SUPPORTED | REFUTED | Pharrell Williams plays an instrument. |
| 11666 | REFUTED | NOT ENOUGH INFORMATION | A near-Earth object is not a dead body. |
| 11823 | REFUTED | NOT ENOUGH INFORMATION | Saxony only has a population of 2 million people. |
| 11853 | REFUTED | NOT ENOUGH INFORMATION | Robinson Crusoe on Mars was distributed by an American film studio in 2006. |
| 11915 | SUPPORTED | REFUTED | Mars is the site of Olympus Mons. |
| 11940 | REFUTED | NOT ENOUGH INFORMATION | Edward Cullen was created outside of Stephenie Meyer's work. |
| 12015 | SUPPORTED | REFUTED | Floyd Mayweather Jr. has made 30 career boxing knockouts. |
| 12016 | REFUTED | NOT ENOUGH INFORMATION | Floyd Mayweather Jr. is incapable of boxing. |
| 12084 | REFUTED | NOT ENOUGH INFORMATION | Acetylcholine prevents neuromodulation. |
| 12088 | SUPPORTED | REFUTED | Global warming causes the abandonment of populated areas due to rising sea levels. |

*3763 more incorrect cases are not listed here; all are in `errors.csv`.*

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
| Correct | 9369 | 9340 |
| Overall accuracy | 0.703 | 0.704 |
| SUPPORTED accuracy | 0.807 | 0.808 |
| REFUTED accuracy | 0.598 | 0.600 |
