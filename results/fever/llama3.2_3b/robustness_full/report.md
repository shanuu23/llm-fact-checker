# FEVER evidence-perturbation robustness experiment

This experiment compares a deterministic control with two meaning-preserving evidence changes. Invariance measures whether the prediction stayed the same; FEVER accuracy measures agreement with the dataset label.

## Setup

- Dataset SHA-256: `46d03c136a5774811e00a92c67c9ad84f78b1daad981bf17c7a07a74f5474882`
- Cases: 13332
- Model: `llama3.2:3b`
- Temperature / seed: 0 / 42
- Reorder-eligible cases: 1204

## Results

| Variant | Complete pairs | Invariance | Paired control accuracy | Perturbed accuracy | Delta |
|---|---:|---:|---:|---:|---:|
| Reorder | 1204 | 79.568% | 73.007% | 73.007% | +0.000% |
| Distractor | 13332 | 30.873% | 70.275% | 17.559% | -52.715% |

Control FEVER accuracy across all 13332 cases: 70.275%.

Detailed outcome, transition, NEI, and technical-validity counts are in `robustness_metrics.json` and `robustness_by_case.csv`.

## Result excluding flagged records

The FEVER data audit flagged 68 of these claims as possibly
inconsistent. The main result above keeps them; this section removes them, as the audit decided.
Flags are possible label inconsistencies, not proven errors.

| | All cases | Excluding flagged |
|---|---:|---:|
| Cases | 13332 | 13264 |
| Control FEVER accuracy | 70.275% | 70.416% |
| Reorder invariance | 79.568% | 79.648% |
| Distractor invariance | 30.873% | 30.745% |
| Distractor FEVER accuracy | 17.559% | 17.544% |
