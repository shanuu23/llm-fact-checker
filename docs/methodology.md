# Methodology

This project measures whether a model behaves like a reliable evidence-grounded fact-checker. It
evaluates existing models; nothing is trained. Everything below describes what is *implemented and
frozen* so results from different models are comparable.

## Common setup

- **Task:** given a claim and supplied evidence, answer `SUPPORTED`, `REFUTED` or
  `NOT ENOUGH INFORMATION` (NEI).
- **Prompt:** one fixed system prompt in `llm_fact_checker/prompts.py`. It tells the model to use
  only the supplied evidence and to return the label alone. Gold labels are never in the prompt.
- **User message:** the claim followed by numbered evidence sentences, each tagged with its page and
  sentence id.
- **Output parsing:** strict. The reply is stripped and upper-cased and must equal one of the three
  labels exactly. Anything else is recorded as an invalid format, never silently repaired.
- **Execution:** sequential, non-streaming requests to a local Ollama server. Raw outputs, parsed
  labels, latencies and errors are all saved.
- **Reproducibility:** each run writes `run_config.json` with the prompt, model tag and digest,
  Ollama version, seeds, dataset hash and git commit.

## Datasets

| Dataset | Used for | Sample |
|---|---|---|
| FEVER shared-task dev | Correctness, stability, robustness | 150 claims: 75 SUPPORTS + 75 REFUTES, seed 42 |
| VitaminC test | Sensitivity | 100 contrastive pairs, seed 42 |

**FEVER sample.** Claims are de-duplicated by id and by claim text, shuffled deterministically per
class, and each is given the valid gold evidence set with the fewest unique sentences (stable
page/sentence tie-break). Required sentences are resolved by streaming the Wikipedia dump. Gold
NOT ENOUGH INFO claims are excluded: FEVER provides no evidence for them, so they do not fit a
gold-evidence experiment.

**VitaminC pairs.** Rows are linked by `case_id` plus exact claim text. A valid pair has exactly one
SUPPORTS row and one REFUTES row with different evidence, the same page, revision type and source
revision identifier. The one ambiguous contradictory group is excluded. Claims are unique across
the selected pairs.

## Whole-dataset run and flagged records

The baseline can also run on the whole FEVER dev set (13,332 SUPPORTS and REFUTES claims),
following the FEVER data audit. Two results are reported:

- **Main result**: every claim, including the 68 the audit flagged as possibly inconsistent.
- **Second result**: the same run without the flagged claims (13,264).

The flags (`conflicting_duplicate_label`, `normalized_claim_conflict`) are possible label
inconsistencies, not proven errors, and no label is ever changed. A large gap between the two
results would suggest label noise is moving the number; a small gap means it is not. NOT ENOUGH
INFO claims remain out of scope because FEVER provides them with no gold evidence.

## Experiments and metrics

### 1. Correctness (baseline)

Temperature 0, seed 42, one pass over the 150 claims.

- **Overall accuracy:** parsed prediction equals the gold label.
- **Class accuracy:** the same, per gold class.
- **Confusion matrix:** gold against predicted, including `INVALID`.
- **NEI answers** are reported separately. They count as incorrect here, since the sample has no
  gold NEI, but they are *not* the same failure as the opposite verdict: they usually mean the model
  failed to recognise a contradiction.

### 2. Stability (five-run consistency)

The same 150 claims, five times, temperature 0.2, seeds 101, 202, 303, 404, 505.

- **Exact consistency:** all five answers identical.
- **Modal agreement:** share of runs agreeing with the most common answer, averaged over claims.
- **Stable correct / stable wrong / unstable:** identical and equal to gold, identical and not
  equal to gold, or not identical.
- **Per-run accuracy:** mean, standard deviation, min and max across runs.

Consistency is **not** correctness: a model can be wrong the same way every time.

### 3. Robustness (evidence perturbations)

Temperature 0, seed 42, so any change is attributable to the perturbation.

- **Reorder:** reverse the order of the evidence sentences. Only claims with two or more evidence
  sentences are eligible (11 of 150), so this is a small-sample observation.
- **Distractor:** append one unrelated evidence sentence taken from another case. A candidate is
  rejected if it comes from the same case, the same Wikipedia page, or shares two or more
  non-stopword terms with the claim or its evidence. Selection is deterministic.

For each variant, against an unperturbed control run:

- **Invariance:** the prediction did not change.
- **Accuracy delta:** perturbed accuracy minus control accuracy on the same cases.
- **Transitions and NEI flow:** which labels moved to which, and how many definite verdicts
  collapsed into NEI.

A robust model should be invariant: the perturbations do not change what the evidence says.

### 4. Sensitivity (VitaminC contrastive pairs)

Temperature 0, seed 42, two calls per pair: the same claim with SUPPORTS evidence and with REFUTES
evidence. A sensitive model must change its verdict, and in the right direction.

| Outcome | Meaning |
|---|---|
| `CORRECT_FLIP` | SUPPORTED on the support evidence and REFUTED on the refute evidence |
| `NO_CHANGE_CORRECT_ON_SUPPORT` / `_REFUTE` | Same verdict twice; right on only one side |
| `WRONG_DIRECTION` | Verdicts differ, but reversed |
| `NEI_INVOLVED` | At least one side answered NEI |
| `OTHER_ERROR` | Invalid format or inference failure |

The **primary metric** is the correct-flip rate. Verdict flips (any change), individual accuracy
per side and NEI behaviour are reported separately, because a model that flips often but wrongly
should not score well.

The frozen comparison uses 100 pairs, one per claim. The same experiment also runs on every valid
pair in the test set (`vitaminc-sensitivity --all`: 20,232 pairs, 18,352 distinct claims). That run
keeps a claim that occurs in several pairs in each of them, because each pair has its own case and
evidence; only the sample de-duplicates claims.

## Interpreting results

- The four dimensions are separate. Report them side by side; do not average them away.
- NEI is not automatically bad. It can be appropriate caution or a missed contradiction. The
  reports show which.
- With 150 claims and 100 pairs, one case is about 0.7 and 1 percentage point. Treat differences of
  a few points as noise unless a confidence interval says otherwise.
- FEVER labels contain known annotation noise. The dataset's own audit flags 54 duplicate-claim
  groups (124 records) with conflicting labels in the full dev file; labels are never changed here.
- Both datasets derive from Wikipedia, which models may have seen in pretraining. Scores measure
  agreement with the benchmark, not proof the model never saw the claims.

## What is deliberately not tested

Evidence retrieval, calibration or confidence, prompt paraphrase sensitivity, and any real-world
fact-checking claim beyond these benchmarks.
