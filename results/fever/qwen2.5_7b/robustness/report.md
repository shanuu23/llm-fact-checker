# FEVER evidence-perturbation robustness experiment

This experiment compares a deterministic control with two meaning-preserving evidence changes. Invariance measures whether the prediction stayed the same; FEVER accuracy measures agreement with the dataset label.

## Setup

- Dataset SHA-256: `6bc7cf910cb0b31186e5fc397ebc7cefab2de1ee65d29ae601e369d8758344cb`
- Cases: 150
- Model: `qwen2.5:7b`
- Temperature / seed: 0 / 42
- Reorder-eligible cases: 11

## Results

| Variant | Complete pairs | Invariance | Paired control accuracy | Perturbed accuracy | Delta |
|---|---:|---:|---:|---:|---:|
| Reorder | 11 | 90.909% | 81.818% | 72.727% | -9.091% |
| Distractor | 147 | 95.238% | 90.667% | 88.000% | -2.667% |

Control FEVER accuracy across all 150 cases: 90.667%.

Detailed outcome, transition, NEI, and technical-validity counts are in `robustness_metrics.json` and `robustness_by_case.csv`.
