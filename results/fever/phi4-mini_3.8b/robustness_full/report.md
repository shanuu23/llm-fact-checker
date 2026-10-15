# FEVER evidence-perturbation robustness experiment

This experiment compares a deterministic control with two meaning-preserving evidence changes. Invariance measures whether the prediction stayed the same; FEVER accuracy measures agreement with the dataset label.

## Setup

- Dataset SHA-256: `46d03c136a5774811e00a92c67c9ad84f78b1daad981bf17c7a07a74f5474882`
- Cases: 13332
- Model: `phi4-mini:3.8b`
- Temperature / seed: 0 / 42
- Reorder-eligible cases: 1204

## Results

| Variant | Complete pairs | Invariance | Paired control accuracy | Perturbed accuracy | Delta |
|---|---:|---:|---:|---:|---:|
| Reorder | 1200 | 86.083% | 58.638% | 58.804% | +0.166% |
| Distractor | 13277 | 58.417% | 59.863% | 33.378% | -26.485% |

Control FEVER accuracy across all 13332 cases: 59.863%.

Detailed outcome, transition, NEI, and technical-validity counts are in `robustness_metrics.json` and `robustness_by_case.csv`.

## Result excluding flagged records

The FEVER data audit flagged 68 of these claims as possibly
inconsistent. The main result above keeps them; this section removes them, as the audit decided.
Flags are possible label inconsistencies, not proven errors.

| | All cases | Excluding flagged |
|---|---:|---:|
| Cases | 13332 | 13264 |
| Control FEVER accuracy | 59.863% | 59.952% |
| Reorder invariance | 86.083% | 86.050% |
| Distractor invariance | 58.417% | 58.392% |
| Distractor FEVER accuracy | 33.378% | 33.474% |
