# Full evaluation protocol

This protocol evaluates **existing LLMs** on supplied claim–evidence inputs; it does not train a
model or build an evidence-retrieval fact checker. These scoring rules are fixed before full model
inference. The number of VitaminC pairs and the model lineup will be fixed after a short timing
check, before comparing models. The earlier 150-case FEVER and 100-pair VitaminC experiments remain
pilots; their results are not full-evaluation results.

## Fixed inputs and units

| Dataset | Input | Evaluation unit | Gold labels |
|---|---|---|---|
| FEVER | `data/fever/processed/fever_gold_full_13332.jsonl` (SHA-256 `2511951db73f1f7c266f06e41256fa67b2219a759c25304de4c9289d78f31aec`) | One claim with its selected gold evidence | SUPPORTED, REFUTED |
| VitaminC | `data/vitaminc/processed/train_full/rows.jsonl` and `pairs.jsonl`, derived from `train.jsonl` (SHA-256 `446163d81ec9b654d2514e8ba5b8f2e4bd8938342e34c98e7660fe05a8a3ca39`) | One fixed claim with two different evidence passages | SUPPORTED, REFUTED, NOT ENOUGH INFORMATION |

The prepared FEVER file contains 13,332 verifiable claims, not all 19,998 shared-task development
claims. Gold FEVER NOT ENOUGH INFO cases have no supplied gold evidence and are outside this
gold-evidence test. Do not describe its accuracy as three-class FEVER accuracy. VitaminC NEI sides
**do** contain evidence; that evidence is insufficient for the claim.

VitaminC pairs must have the same `case_id` and exact claim, exactly two rows with different
evidence and labels, and the same page, revision type and source revision. The audited train file
has 131,709 eligible SUPPORTS–REFUTES pairs, 52,779 SUPPORTS–NEI pairs and 102 REFUTES–NEI pairs.
These are **eligibility counts**, not the committed run size. Keep each pair intact when sampling.
The generated row and pair files have SHA-256 hashes
`ea2e06f4beb1dce166908ad9a0b285db3e0649c27f6572c04c0ca80dd9892836` and
`92947ec1006da123750078c56f10d0ee47332b191b57e053ad99b431364d08f8`, respectively.
They are local, regenerable files (`python -m llm_fact_checker vitaminc-train-prepare`), not Git
artifacts.

## Selection before inference

- FEVER's primary result uses all 13,332 prepared rows. All 150 FEVER pilot IDs are within that
  file; disclose this and also report a secondary score excluding those IDs. Do not silently call
  that secondary score the full benchmark result.
- For each VitaminC transition type, keep only the lowest `pair_id` for each **exact claim** before
  sampling. This prevents a frequently repeated claim from receiving extra weight. Sort the
  remaining pair IDs, sample without replacement using a fresh `random.Random(42)` for each type,
  and save the chosen IDs, counts, seed and file hash before model calls. The same IDs go to every
  model.
- Exclude an exact claim if it appeared in the 100-pair VitaminC pilot before sampling. The current
  train pair pool has one such pair. The REFUTES–NEI pool has only 102 pairs, so report its actual
  denominator and avoid treating a small result as equally precise as the larger types.
- Set the number of pairs per type after a timing check but **before** looking at full-run
  predictions. Record the real/synthetic composition of the SUPPORTS–REFUTES selection.

The frozen VitaminC selection is
`data/vitaminc/processed/vitaminc_train_eval_pair_ids.jsonl` (SHA-256
`5ceaa412aa0b96474608f80e104662bb60d72a50c7a2b9d8588431a462eb31f8`). It contains
1,000 SUPPORTS–REFUTES, 1,000 SUPPORTS–NEI and all 102 REFUTES–NEI pairs after the rules above
(2,102 pairs; 4,204 model inputs). The SUPPORTS–REFUTES sample has 558 real and 442 synthetic
pairs. Its adjacent `.manifest.json` records the source hashes, seed, counts and audit flags.

## Scoring

**FEVER:** A prediction is correct only when its parsed label equals the gold label. Report
correct/total and accuracy overall and separately for SUPPORTED and REFUTED, plus a confusion
matrix. Invalid outputs and inference failures remain in the total and count as incorrect; report
their counts separately.

**VitaminC:** Score each transition type separately. A pair is correct only when **both** sides
match their gold labels: SUPPORTED→REFUTED, SUPPORTED→NOT ENOUGH INFORMATION, or
REFUTED→NOT ENOUGH INFORMATION. The primary result for each type is correct pairs / all selected
pairs of that type. An invalid output or inference failure makes its pair incorrect and is reported
separately. Also report per-side accuracy and the rate of *any* prediction change among pairs with
two valid predictions. A changed verdict alone is not success. Do not substitute a pooled
VitaminC percentage for the three pair-type results; their available populations differ greatly.
For SUPPORTS–REFUTES, additionally show real versus synthetic source results; the two NEI pair
types in the audited train file are real-source only.

**Data quality:** Retain raw labels and evidence. Include flagged examples in the primary scores,
then recompute the same scores with flagged examples excluded as a clearly labelled sensitivity
analysis. For VitaminC, exclude a whole pair if either side is flagged. Filtering saved predictions
does not require a second model run. A flag signals review, not proof that a label is wrong.

For every primary rate, show its numerator and denominator as well as a 95% Wilson interval. The
interval is descriptive: related benchmark examples and possible pretraining exposure mean it is
not a guarantee of performance on new claims. Compare models on their *paired* shared cases, not
by reading small differences between two headline percentages as proof of superiority.

## Comparable runs and limits

- Freeze a deterministic list of case/pair IDs and its hash before the full comparison. Every
  model receives the same selected inputs, evidence ordering, prompt and strict label parser.
- Request deterministic decoding where supported; record actual provider, model/version, settings,
  prompt, input hashes, timestamps, raw outputs, parsed labels, errors and latency. Document any
  provider setting that cannot be matched exactly.
- Fix the technical-failure policy in the runner: retry transport/time-out failures at most twice,
  never re-prompt merely because a model returned an invalid label, and count unresolved failures
  as incorrect while reporting them separately. Save each response so a resumed run does not
  repeat successful calls.
- Keep FEVER and VitaminC results separate: FEVER counts individual claims; VitaminC counts
  two-call pairs. Neither benchmark measures evidence retrieval or unrestricted truth checking.
- Run an isolated smoke test and timing check first. Fix the VitaminC sample size and model lineup
  from runtime/resource constraints **before** examining full results. A small app demo is not an
  accuracy estimate.

Using VitaminC `train.jsonl` is valid for evaluating existing models because this project does not
train on it. However, both benchmarks are public and may have appeared in model pretraining; do
not describe these scores as performance on guaranteed unseen data.

Checkpoint/resume for Ollama inference is implemented in `llm_fact_checker/full_run.py`.
The completed runs are scored without model calls by `python -m llm_fact_checker.full_score`,
which writes `results/full_evaluation_summary.json`. Cross-provider execution is **not yet
implemented**.

The summary also reports a post-hoc typo-tolerant score: only `REFUTATED`, `REFUTTED`,
`REFUTUTED`, and `REFUTUED` are interpreted as `REFUTED`. Strict scores and raw outputs remain
unchanged. A mapped typo counts as correct only when the gold label is `REFUTED`.
