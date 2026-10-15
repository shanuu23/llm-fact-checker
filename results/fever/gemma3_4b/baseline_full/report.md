# FEVER gold-evidence baseline: gemma3:4b

## Purpose

This baseline tests whether `gemma3:4b` classifies FEVER SUPPORTS and REFUTES claims when supplied with one complete gold Wikipedia evidence set. It uses only the supplied evidence in the prompt.

## Dataset and sample construction

- Source: `data/fever/raw/shared_task_dev.jsonl`
- Prepared data: `data/fever/full/fever_dev_full.jsonl`
- Cases: 13332 (6666 SUPPORTED, 6666 REFUTED)
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

- Correct: 10454 / 13332
- Overall accuracy: 0.784
- SUPPORTED accuracy: 0.941
- REFUTED accuracy: 0.628
- Invalid-format outputs: 0 (0.000%)
- Inference failures: 0
- Prediction counts: SUPPORTED: 7972, REFUTED: 4469, NOT ENOUGH INFORMATION: 891, INVALID: 0

## Confusion matrix

| Gold | SUPPORTED | REFUTED | NOT ENOUGH INFORMATION | INVALID |
|---|---:|---:|---:|---:|
| SUPPORTED | 6270 | 285 | 111 | 0 |
| REFUTED | 1702 | 4184 | 780 | 0 |

## Incorrect cases

| FEVER ID | Gold | Prediction | Claim |
|---:|---|---|---|
| 70 | SUPPORTED | REFUTED | David Packouz was born in February of 1982. |
| 223 | SUPPORTED | REFUTED | Advertising is an audio form of marketing communication. |
| 369 | REFUTED | SUPPORTED | The State of Palestine claims a territory in Western Asia. |
| 519 | SUPPORTED | NOT ENOUGH INFORMATION | The Hundred Years' War includes the Civil War. |
| 531 | REFUTED | SUPPORTED | Keegan-Michael Key played zero characters. |
| 619 | REFUTED | SUPPORTED | Hannah and Her Sisters' ending is unrelated to Thanksgiving. |
| 910 | SUPPORTED | REFUTED | Shut Up is a title. |
| 972 | REFUTED | SUPPORTED | Stan Beeman is a fictional setting. |
| 1188 | REFUTED | NOT ENOUGH INFORMATION | Pearl Jam is a type of dressing. |
| 1667 | SUPPORTED | REFUTED | Ed Wood is not about the eponymous filmmaker. |
| 1746 | REFUTED | SUPPORTED | Sheryl Lee has yet to work with Woody Allen. |
| 1831 | REFUTED | NOT ENOUGH INFORMATION | DNA is a single. |
| 1931 | SUPPORTED | NOT ENOUGH INFORMATION | Team Fortress 2 development was led by John Cook. |
| 2106 | SUPPORTED | REFUTED | Johann Wolfgang von Goethe published the first part of Faust in 1808. |
| 2359 | REFUTED | NOT ENOUGH INFORMATION | Virginia keeps all computer chips manufactured within the state for use in Virginian electronics. |
| 2372 | REFUTED | NOT ENOUGH INFORMATION | Guillermo del Toro works in the coal industry. |
| 2521 | REFUTED | SUPPORTED | Akkineni Nageswara Rao plays in Daag. |
| 2531 | REFUTED | NOT ENOUGH INFORMATION | Dopamine prevents neuromodulation. |
| 2948 | REFUTED | SUPPORTED | The Mod Squad is an American movie. |
| 2984 | REFUTED | SUPPORTED | Kevin Costner refused the role of Devil Anse Hatfield in 2012. |
| 3019 | REFUTED | SUPPORTED | The horse was domesticated on a broad scale by 3103 BC. |
| 3063 | SUPPORTED | REFUTED | Ronaldinho has played 90 matches. |
| 3111 | REFUTED | SUPPORTED | Luis Fonsi was born in the eighties. |
| 3371 | REFUTED | SUPPORTED | Maria Theresa reigned for only one year. |
| 3483 | REFUTED | SUPPORTED | Brie Larson was denied any role in the movie Short Term 12. |
| 3494 | REFUTED | SUPPORTED | English people are unconnected to the Saxons. |
| 3518 | REFUTED | SUPPORTED | Exercise is bad for heart health. |
| 3556 | REFUTED | SUPPORTED | A River Runs Through It has lost every nomination. |
| 3638 | REFUTED | SUPPORTED | There are currently 15,882,417 Mormon members as of 2007. |
| 3656 | REFUTED | NOT ENOUGH INFORMATION | John DiMaggio refused to work on any shows from Cartoon Network. |
| 3668 | REFUTED | SUPPORTED | Andrew Moray refused to cede the territory he gained to John Balliol. |
| 3682 | REFUTED | SUPPORTED | English people are disconnected from the Jutes and Frisians. |
| 4058 | REFUTED | SUPPORTED | Balibo (film) involves zero journalist. |
| 4091 | REFUTED | SUPPORTED | Zac Efron's full name is Zac David Alexander Efron. |
| 4100 | SUPPORTED | REFUTED | Sonny & Cher began in the mid-1930's. |
| 4251 | REFUTED | NOT ENOUGH INFORMATION | Solanum contains the leopards. |
| 4458 | SUPPORTED | NOT ENOUGH INFORMATION | Edouard Seguin worked in the United States. |
| 4537 | SUPPORTED | REFUTED | Eric Church died on May 3, 1977. |
| 4713 | REFUTED | NOT ENOUGH INFORMATION | Tool has won three Oscars. |
| 4778 | SUPPORTED | REFUTED | Betsy Hodges was a candidate for a 2013 election. |
| 5076 | REFUTED | NOT ENOUGH INFORMATION | The State of Palestine claims Algeria. |
| 5169 | REFUTED | SUPPORTED | Jonah Hill is not a screenwriter. |
| 5314 | REFUTED | SUPPORTED | Younger is a town. |
| 5421 | REFUTED | SUPPORTED | Stephen Hillenburg developed an interest in everything but art. |
| 5454 | REFUTED | SUPPORTED | Jack Falahee's middle name is Bryan. |
| 5578 | SUPPORTED | NOT ENOUGH INFORMATION | Australia (2008 film) production took place in Kununurra. |
| 5579 | SUPPORTED | NOT ENOUGH INFORMATION | Australia (2008 film) production took place in Bowen. |
| 5705 | REFUTED | SUPPORTED | Emma Watson is a French-British car. |
| 5744 | SUPPORTED | NOT ENOUGH INFORMATION | One Flew Over the Cuckoo's Nest won Best Actor at the Academy Awards. |
| 5781 | REFUTED | NOT ENOUGH INFORMATION | Raja Hindustani was directed solely by Mariah Carey. |
| 5900 | REFUTED | NOT ENOUGH INFORMATION | The Paper stars a horse. |
| 5980 | REFUTED | NOT ENOUGH INFORMATION | Camden, New Jersey is the home of Yale School of Medicine. |
| 6032 | REFUTED | NOT ENOUGH INFORMATION | Aruba is the only ABC Island. |
| 6275 | REFUTED | NOT ENOUGH INFORMATION | Recovery is an album focusing on themes of isolation and addiction. |
| 6334 | REFUTED | SUPPORTED | The Challenge XXX: Dirty 30 is not a season of a reality game show. |
| 6414 | REFUTED | NOT ENOUGH INFORMATION | The Godfather Part II is a political party. |
| 6834 | REFUTED | SUPPORTED | Chris Benoit's middle name is Jason. |
| 6893 | SUPPORTED | REFUTED | Underdog features the voice talents of Amy Adams. |
| 6925 | REFUTED | SUPPORTED | Ed Gein committed crimes around Milwaukee, Wisconsin. |
| 7041 | REFUTED | SUPPORTED | Kareem Abdul-Jabbar is not ranked 3rd all-time in blocks. |
| 7286 | REFUTED | NOT ENOUGH INFORMATION | Croatia is incapable of rendering services. |
| 7336 | REFUTED | SUPPORTED | Lizzy Caplan is incapable of appearing in multiple television shows. |
| 7418 | SUPPORTED | REFUTED | The Faroe Islands existed in 1035. |
| 7546 | SUPPORTED | REFUTED | Brown University's main campus is located in the most populous city in Rhode Island. |
| 7697 | REFUTED | NOT ENOUGH INFORMATION | Riddick is in a 2023 film. |
| 7728 | REFUTED | SUPPORTED | Hinduism has zero textual resources. |
| 7744 | REFUTED | NOT ENOUGH INFORMATION | Mount Hood is the sixth highest point in its mountain range. |
| 7811 | REFUTED | SUPPORTED | Adobe Photoshop is incapable of being part of the Adobe Systems. |
| 7877 | REFUTED | NOT ENOUGH INFORMATION | The ruins of the ancient Roman town of Herculaneum were teleported to a location outside Naples. |
| 8167 | REFUTED | SUPPORTED | Caesar is a 1937 adaptation of The Tempest. |
| 8298 | REFUTED | NOT ENOUGH INFORMATION | Simon Pegg is only a banker. |
| 8464 | REFUTED | SUPPORTED | I Kissed a Girl is only an EP. |
| 8546 | REFUTED | SUPPORTED | Prague's historic center is not on UNESCO's list of World Heritage Sites. |
| 8577 | REFUTED | SUPPORTED | The Dodo has been depicted with a naked head and arm. |
| 8603 | REFUTED | SUPPORTED | The Indian Institute of Management Bangalore refuses to offer a doctoral program. |
| 8725 | REFUTED | SUPPORTED | Sonny & Cher are rock singers. |
| 8830 | REFUTED | SUPPORTED | Helmand Province is far away from Lashkargah. |
| 8864 | SUPPORTED | REFUTED | Great white sharks have never killed people. |
| 8867 | REFUTED | SUPPORTED | Moscovium is atomic weight 115. |
| 8905 | SUPPORTED | REFUTED | The Raven (2012 film) was released in a sovereign country in western Europe in March of 2012. |
| 8926 | REFUTED | SUPPORTED | Advertising is rarely a visual form of marketing communication. |
| 9101 | REFUTED | SUPPORTED | Key & Peele has only won one Primetime Emmy Award. |
| 9152 | REFUTED | SUPPORTED | Riz Ahmed acted in an Portuguese miniseries based on a British television series. |
| 9267 | SUPPORTED | NOT ENOUGH INFORMATION | Edouard Seguin was employed in the United States. |
| 9643 | REFUTED | SUPPORTED | Sledgehammer was first performed live in America. |
| 9825 | REFUTED | SUPPORTED | SZA's music does not combine elements from different genres. |
| 9860 | REFUTED | NOT ENOUGH INFORMATION | Bob Ross created the Internet. |
| 9913 | REFUTED | NOT ENOUGH INFORMATION | Pirates of the Caribbean was added to Magic Kingdom in 1993. |
| 9945 | REFUTED | SUPPORTED | The Armenian Genocide took place outside the Ottoman Empire and the Republic of Turkey. |
| 10262 | REFUTED | SUPPORTED | The English Wikipedia is an edition of a free online chat room. |
| 10272 | REFUTED | SUPPORTED | Taran Killam isn't a writer. |
| 10328 | REFUTED | SUPPORTED | Sidse Babett Knudsen graduated on November 22nd, 1968. |
| 10379 | REFUTED | SUPPORTED | The horse was not called a Eohippus when it had multiple toes. |
| 10530 | REFUTED | SUPPORTED | Fuller House (TV series) is unable to be a family sitcom. |
| 10567 | REFUTED | SUPPORTED | Google Search shows no home information. |
| 10635 | REFUTED | SUPPORTED | Unforced labor is a reason for human trafficking. |
| 10689 | REFUTED | SUPPORTED | Cleopatre premiered in a manner. |
| 10712 | REFUTED | SUPPORTED | Speech recognition is ignored in the computer science fields. |
| 10765 | REFUTED | NOT ENOUGH INFORMATION | Mount Hood is the highest point of a roller coaster. |
| 11290 | REFUTED | SUPPORTED | Talulah Riley did not appear in The Boat That Rocked. |
| 11310 | SUPPORTED | REFUTED | Papua comprised all of Indonesian New Guinea. |
| 11352 | REFUTED | NOT ENOUGH INFORMATION | The Hit was released on a mural. |
| 11441 | REFUTED | SUPPORTED | The Bassoon King's working title was My Life in Art, Faith, and Idiocy. |
| 11443 | REFUTED | SUPPORTED | David Attenborough served as director of BBC Two. |
| 11572 | REFUTED | SUPPORTED | Samsung was separated into four business groups in Italy. |
| 11666 | REFUTED | NOT ENOUGH INFORMATION | A near-Earth object is not a dead body. |
| 12016 | REFUTED | SUPPORTED | Floyd Mayweather Jr. is incapable of boxing. |
| 12084 | REFUTED | SUPPORTED | Acetylcholine prevents neuromodulation. |
| 12369 | REFUTED | NOT ENOUGH INFORMATION | James VI and I was a major advocate of women's rights for Scotland. |
| 12403 | REFUTED | SUPPORTED | The Challenge XXX: Dirty 30 is a contestant in a reality game show. |
| 12454 | REFUTED | NOT ENOUGH INFORMATION | Tangled is a silent film. |
| 12467 | SUPPORTED | REFUTED | SpongeBob SquarePants is not the most distributed property of MTV Networks. |
| 12469 | REFUTED | SUPPORTED | Argentina is not also subdivided into one autonomous city. |
| 12553 | REFUTED | SUPPORTED | Taran Killam wasn't born in April. |
| 12610 | REFUTED | SUPPORTED | Paul Nicholls did not play Joe Wicks. |
| 12684 | REFUTED | SUPPORTED | The CONCACAF Champions League is rarely organized by CONCACAF. |
| 12691 | REFUTED | SUPPORTED | San Diego Comic-Con was not founded in 1970. |
| 12748 | SUPPORTED | NOT ENOUGH INFORMATION | The ruins of the ancient Roman town of Herculaneum lie two hundred miles from Naples. |
| 12936 | REFUTED | NOT ENOUGH INFORMATION | Maria Theresa reigned over Spain. |
| 12967 | REFUTED | NOT ENOUGH INFORMATION | The Good Wife is a theme park. |
| 12990 | REFUTED | SUPPORTED | Lee Daniels has only directed terrible films that go unrecognized. |
| 13001 | SUPPORTED | REFUTED | For four years, Ann Richards was the Governor of Texas. |
| 13076 | SUPPORTED | REFUTED | The White House Press Secretary's primary responsibility is to be a father. |
| 13086 | REFUTED | NOT ENOUGH INFORMATION | Chris Paul is a two-time Nobel Prize winner. |
| 13155 | REFUTED | SUPPORTED | James Earl Jones was not the voice of Darth Vader. |
| 13210 | REFUTED | SUPPORTED | Vampires were lacking from European folklore. |
| 13230 | SUPPORTED | REFUTED | A View to a Kill is the third James Bond film directed by John Glen. |
| 13236 | REFUTED | SUPPORTED | The Silver Surfer appears in manga published by Marvel Comics. |
| 13242 | REFUTED | SUPPORTED | The United States dollar is not used in the Caribbean. |
| 13829 | SUPPORTED | REFUTED | Balibo (film) starts in the year 1975. |
| 13929 | SUPPORTED | NOT ENOUGH INFORMATION | The WWE created it's developmental brand NXT after the start of the NXT Tag Team Championship. |
| 13942 | SUPPORTED | NOT ENOUGH INFORMATION | Wish Upon starred a person. |
| 14112 | SUPPORTED | REFUTED | Harris Jayaraj was born in January of 1975. |
| 14123 | REFUTED | NOT ENOUGH INFORMATION | The Last Song (film) began filming on June 14th, 2009. |
| 14286 | REFUTED | SUPPORTED | Pearl (Steven Universe) projects a holographic butterfly. |
| 14367 | REFUTED | SUPPORTED | Michael Hutchence died on November 12th, 1997. |
| 14415 | SUPPORTED | REFUTED | July 1974 is the birthday of Jed Whedon. |
| 14430 | REFUTED | SUPPORTED | Pocahontas was the daughter of the paramount chief of Wahunsenacah. |
| 14431 | REFUTED | SUPPORTED | Pocahontas was not the daughter of the paramount chief of Wahunsenacah. |
| 14731 | REFUTED | SUPPORTED | Gamal Abdel Nasser's graduation ceremony drew five million mourners. |
| 14855 | REFUTED | SUPPORTED | Rupert Murdoch is a broadcaster, forming BSkyB. |
| 15155 | SUPPORTED | REFUTED | Luke Cage did not team up with Iron Fist. |
| 15483 | REFUTED | SUPPORTED | South Island is sometimes referred to as the "mainland" of Australia. |
| 15489 | REFUTED | SUPPORTED | See You on the Other Side is a book. |
| 15559 | REFUTED | SUPPORTED | Times Higher Education World University Rankings is a publication of illustrations. |
| 15674 | SUPPORTED | REFUTED | Soul Food was produced by Tracey Edmonds. |
| 15855 | REFUTED | SUPPORTED | The Columbia River has yet to undergo dredging. |
| 16080 | REFUTED | NOT ENOUGH INFORMATION | Solanum contains animals with ornamental flowers. |
| 16115 | REFUTED | NOT ENOUGH INFORMATION | Ayananka Bose works on books. |
| 16325 | REFUTED | NOT ENOUGH INFORMATION | Hinduism forbids pilgrimage to sacred sites. |
| 16382 | REFUTED | NOT ENOUGH INFORMATION | The Africa Cup of Nations is a friendly global soccer exhibition. |
| 16629 | REFUTED | NOT ENOUGH INFORMATION | Brie Larson and Destin Daniel refused to ever work with each other. |
| 16951 | REFUTED | NOT ENOUGH INFORMATION | The Premier League Asia Trophy is a battle. |
| 17145 | REFUTED | SUPPORTED | John Goodman starred in Cloverfield. |
| 17214 | REFUTED | SUPPORTED | Cars 3 isn't a computer animated movie. |
| 17274 | REFUTED | NOT ENOUGH INFORMATION | The Catcher in the Rye is a bar. |
| 17314 | REFUTED | SUPPORTED | Telemundo is incapable of being owned by Comcast. |
| 17452 | REFUTED | NOT ENOUGH INFORMATION | Camden, New Jersey is a publication. |
| 17616 | REFUTED | SUPPORTED | Manchester United is not based in Old Trafford, Greater Manchester, England. |
| 17628 | REFUTED | SUPPORTED | Jack Lowden is incapable of being an actor. |
| 17634 | REFUTED | NOT ENOUGH INFORMATION | I Kissed a Girl is only part of the album Things Fall Apart. |
| 17744 | SUPPORTED | REFUTED | Heavy Metal music was first developed in the late 1960's. |
| 17841 | REFUTED | SUPPORTED | Food Network is unavailable to 83.3% of households with television. |
| 17847 | REFUTED | SUPPORTED | The Road to El Dorado is an artistic work. |
| 17855 | REFUTED | SUPPORTED | One True Thing was undirected. |
| 17907 | REFUTED | SUPPORTED | Konidela Production Company was disestablished by an actor. |
| 17933 | REFUTED | SUPPORTED | Catherine Keener appeared in the 1997 film Synecdoche, New York. |
| 17973 | SUPPORTED | NOT ENOUGH INFORMATION | Part of Dilwale Dulhania Le Jayenge was shot in August. |
| 18051 | REFUTED | SUPPORTED | Kate Walsh has yet to appear on an ABC television drama. |
| 18065 | REFUTED | SUPPORTED | Colombiana was produced by Olivier Megaton. |
| 18066 | REFUTED | SUPPORTED | Yin and yang show a balance between magnetized entities. |
| 18197 | SUPPORTED | NOT ENOUGH INFORMATION | Johnny Van Zant became the lead vocalist for Lynyrd Skynyrd in concerts. |
| 18487 | REFUTED | SUPPORTED | The United Nations Charter was destroyed in 1945. |
| 18497 | REFUTED | SUPPORTED | Maria Theresa was the rival of Emperor Charles VI. |
| 18723 | REFUTED | NOT ENOUGH INFORMATION | Wildfang was founded in orange. |
| 18823 | REFUTED | NOT ENOUGH INFORMATION | The Quran is a WWE wrestler.. |
| 18835 | REFUTED | SUPPORTED | Rhythm Nation was covered by an American singer, and songwriter Crystal Kay. |
| 18951 | REFUTED | NOT ENOUGH INFORMATION | The Colosseum is a resort. |
| 19169 | REFUTED | SUPPORTED | Poseidon lost $181,674,817. |
| 19177 | REFUTED | SUPPORTED | Victoria Palace Theatre is on the same side as a central London railway terminus. |
| 19302 | REFUTED | SUPPORTED | Sausage Party was directed by two cats. |
| 19418 | SUPPORTED | REFUTED | The Bengal tiger is a member of the largest cat species. |
| 19711 | REFUTED | SUPPORTED | The Icelandic Coast Guard is barely known as Landhelgisgaeslan. |
| 19721 | REFUTED | SUPPORTED | The Indian Institute of Management Bangalore refuses to provide a post graduate program. |
| 19766 | REFUTED | NOT ENOUGH INFORMATION | Pakistan is a store. |
| 19898 | SUPPORTED | NOT ENOUGH INFORMATION | Birthday Song (2 Chainz song) was produced by Mike Dean. |
| 19902 | SUPPORTED | NOT ENOUGH INFORMATION | One of the newest Netflix TV series's is GLOW. |
| 20026 | SUPPORTED | REFUTED | Aphrodite is the daughter of a Titaness in Homer's Iliad. |
| 20202 | REFUTED | NOT ENOUGH INFORMATION | Watertown, Massachusetts is in the least populous state in New England. |
| 20458 | REFUTED | SUPPORTED | Yves Saint Laurent lacks the ability to make clothes. |
| 20461 | REFUTED | NOT ENOUGH INFORMATION | Daggering is associated with a genre of Filipino music. |
| 20708 | REFUTED | SUPPORTED | Black Canary is a character in comic books published by an American entrepreneur. |
| 20770 | REFUTED | SUPPORTED | How to Train Your Dragon 2 refused to use scalable multicore processing. |
| 21033 | REFUTED | SUPPORTED | Victoria Palace Theatre is on the same side of Victoria Station. |
| 21270 | REFUTED | SUPPORTED | Yugoslavia failed to exist during most of the 20th century. |
| 21405 | REFUTED | NOT ENOUGH INFORMATION | Brie Larson is an anarchist. |
| 21483 | REFUTED | SUPPORTED | Jack Lowden is incapable of being in films. |
| 21804 | REFUTED | SUPPORTED | The United Nations Charter was vetoed in San Francisco, United States. |
| 21933 | REFUTED | SUPPORTED | Team Fortress 2 is incapable of being a video game. |
| 21959 | SUPPORTED | NOT ENOUGH INFORMATION | The Incredibles 2 is scheduled to be released at 2 p. m. on June 15, 2018. |

*2678 more incorrect cases are not listed here; all are in `errors.csv`.*

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
| Correct | 10454 | 10408 |
| Overall accuracy | 0.784 | 0.785 |
| SUPPORTED accuracy | 0.941 | 0.941 |
| REFUTED accuracy | 0.628 | 0.628 |
