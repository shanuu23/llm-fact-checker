# Cross-Model Fact-Checking Reliability Scorecard

Benchmarked on the frozen 150-claim FEVER sample and 100-pair VitaminC sample.

| Model | FEVER Acc | SUPP Acc | REF Acc | NEI Rate | 5/5 Consistency | Distractor Invariance | Post-Distractor Acc | VitaminC Sensitivity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **llama3.2_1b** | 50.7% | 96.0% | 5.3% | 10.0% | 70.7% | 83.3% | 46.0% | 1.0% |
| **llama3.2_3b** | 64.7% | 77.3% | 52.0% | 23.3% | 80.0% | 39.3% | 14.7% | 27.0% |
| **phi4-mini_3.8b** | 58.0% | 96.0% | 20.0% | 24.7% | 90.0% | 62.4% | 32.0% | 16.0% |
| **qwen2.5_7b** | 90.7% | 88.0% | 93.3% | 0.7% | 98.0% | 95.2% | 88.0% | 72.0% |
