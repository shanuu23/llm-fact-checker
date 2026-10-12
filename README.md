# LLM fact-checker reliability tests

Can a small local LLM be trusted as an **evidence-grounded fact-checker**? This project does not ask
"is this claim true?". It gives a model a claim plus supplied Wikipedia evidence and measures how
reliably it behaves, from four angles.

| Dimension | Question | Experiment | Dataset |
|---|---|---|---|
| Correctness | Does it return the expected verdict? | Baseline | FEVER, 150 claims |
| Stability | Does it repeat the same verdict on the same input? | Five-run consistency | same 150 claims |
| Robustness | Does irrelevant evidence leave the verdict unchanged? | Reorder and distractor | same 150 claims |
| Sensitivity | Does a meaningful evidence change flip the verdict correctly? | Contrastive pairs | VitaminC, 100 pairs |

The model may answer `SUPPORTED`, `REFUTED` or `NOT ENOUGH INFORMATION` (NEI). Definitions and
caveats are in [docs/methodology.md](docs/methodology.md).

## Pilot results: `llama3.2:3b`

| Dimension | Headline result |
|---|---|
| Correctness | 97/150 = **64.67%** (SUPPORTED 77.33%, REFUTED 52.00%; 35 answers were NEI) |
| Stability | **80%** of claims identical in 5/5 runs; 35 claims wrong the same way every run |
| Robustness | Reordering: 8/11 unchanged. One distractor sentence: only **39.33%** unchanged, accuracy 64.67% to 14.67%, NEI answers 35 to 104 |
| Sensitivity | Verdict changed on 72/100 pairs but was **correct on only 27/100** |

Repeatable but often wrong, and easily thrown by irrelevant evidence. Full reports are under
[results/](results/README.md).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/).

```bash
uv sync                       # installs everything: dev tools and notebook libraries
make test                     # unit + golden tests; no Ollama or raw data needed
ollama list                   # must include the exact tag llama3.2:3b
# only `llama3.2:latest` listed?  ollama cp llama3.2:latest llama3.2:3b
```

The experiments call the local Ollama HTTP API directly and never download a model.

The command-line interface can be started three equivalent ways:

```bash
uv run factcheck <command>
uv run python -m llm_fact_checker <command>
uv run python -m llm_fact_checker.baseline <command>   # original form, still supported
```

## Data

The two fixed samples are committed under `data/*/processed/`. The raw datasets are several GB and
are downloaded separately; see [data/README.md](data/README.md) for sources, layout and hashes. You
only need the raw data to re-create a sample, not to run the experiments.

## Run the experiments

Every command refuses to overwrite existing results unless `--overwrite` is passed, and smoke tests
(`--limit`, `*-smoke`) write to their own folders.

```bash
# FEVER baseline
uv run factcheck prepare                        # rebuild the 150-case sample from raw data
uv run factcheck run --limit 3                  # 3-case smoke test
uv run factcheck run                            # all 150 cases
uv run factcheck evaluate                       # re-score saved predictions, no Ollama

# Five-run consistency (750 requests, temperature 0.2, seeds 101..505)
uv run factcheck consistency --limit 3
uv run factcheck consistency

# Evidence robustness: generate perturbations, audit them by eye, then run
uv run factcheck robustness-prepare
uv run factcheck robustness-smoke
uv run factcheck robustness

# VitaminC sensitivity (200 requests)
uv run factcheck vitaminc-prepare
uv run factcheck vitaminc-smoke
uv run factcheck vitaminc-sensitivity
```

Inspect `results/.../prepared/distractor_audit.md` and `pair_audit.md` before running the full
robustness and sensitivity experiments.

## Repository layout

```text
llm_fact_checker/
├── cli.py            command-line interface (argparse only)
├── common.py         JSONL/CSV helpers, SHA-256, git commit, formatting
├── prompts.py        the frozen prompt, label set and strict output parser
├── ollama.py         Ollama HTTP client and inference loop
├── evaluation.py     baseline metrics and report
├── data_prep/        fever.py, vitaminc.py: sampling and evidence extraction
└── experiments/      baseline.py, consistency.py, robustness.py, sensitivity.py
data/                 raw (ignored) and processed samples, HASHES.txt
results/              <dataset>/<model>/<experiment>/ outputs, committed
notebooks/            data audits and analysis
docs/                 methodology
tests/                unit tests and golden regression tests
```

## Tests

```bash
make test        # or: uv run python -m unittest discover -s tests -v
make lint        # ruff check + format check
```

Unit tests use tiny synthetic fixtures. The golden tests recompute every metric and report from the
committed predictions and require an exact match, so published numbers cannot change silently.

## Scope and limitations

- The FEVER sample contains no gold NOT ENOUGH INFO cases, so NEI accuracy is not measured, and any
  NEI answer counts as wrong on this sample.
- FEVER accuracy means agreement with FEVER labels, which contain known annotation noise. It is not
  absolute truth.
- Consistency measures repeatability, not correctness.
- The experiments supply gold evidence. They do not test evidence retrieval, calibration, or
  real-world fact-checking.
- Samples are small (150 claims, 100 pairs); differences of a few points are within noise.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for branches, frozen files and the results policy.
