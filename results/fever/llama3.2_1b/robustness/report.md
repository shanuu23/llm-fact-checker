# FEVER evidence-perturbation robustness experiment

This experiment compares a deterministic control with two meaning-preserving evidence changes. Invariance measures whether the prediction stayed the same; FEVER accuracy measures agreement with the dataset label.

## Setup

- Dataset SHA-256: `6bc7cf910cb0b31186e5fc397ebc7cefab2de1ee65d29ae601e369d8758344cb`
- Cases: 150
- Model: `llama3.2:1b`
- Temperature / seed: 0 / 42
- Reorder-eligible cases: 11

## Results

| Variant | Complete pairs | Invariance | Paired control accuracy | Perturbed accuracy | Delta |
|---|---:|---:|---:|---:|---:|
| Reorder | 11 | 100.000% | 72.727% | 72.727% | +0.000% |
| Distractor | 150 | 83.333% | 50.667% | 46.000% | -4.667% |

Control FEVER accuracy across all 150 cases: 50.667%.

Detailed outcome, transition, NEI, and technical-validity counts are in `robustness_metrics.json` and `robustness_by_case.csv`.
