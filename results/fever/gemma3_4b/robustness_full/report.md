# FEVER evidence-perturbation robustness experiment

This experiment compares a deterministic control with two meaning-preserving evidence changes. Invariance measures whether the prediction stayed the same; FEVER accuracy measures agreement with the dataset label.

## Setup

- Dataset SHA-256: `46d03c136a5774811e00a92c67c9ad84f78b1daad981bf17c7a07a74f5474882`
- Cases: 13332
- Model: `gemma3:4b`
- Temperature / seed: 0 / 42
- Reorder-eligible cases: 1204

## Results

| Variant | Complete pairs | Invariance | Paired control accuracy | Perturbed accuracy | Delta |
|---|---:|---:|---:|---:|---:|
| Reorder | 1204 | 83.472% | 77.076% | 76.910% | -0.166% |
| Distractor | 13332 | 82.816% | 78.413% | 72.870% | -5.543% |

Control FEVER accuracy across all 13332 cases: 78.413%.

Detailed outcome, transition, NEI, and technical-validity counts are in `robustness_metrics.json` and `robustness_by_case.csv`.

## Result excluding flagged records

The FEVER data audit flagged 68 of these claims as possibly
inconsistent. The main result above keeps them; this section removes them, as the audit decided.
Flags are possible label inconsistencies, not proven errors.

| | All cases | Excluding flagged |
|---|---:|---:|
| Cases | 13332 | 13264 |
| Control FEVER accuracy | 78.413% | 78.468% |
| Reorder invariance | 83.472% | 83.501% |
| Distractor invariance | 82.816% | 82.841% |
| Distractor FEVER accuracy | 72.870% | 72.964% |
