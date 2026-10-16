# ModelLens: local LLM reliability evaluation

ModelLens helps **AI engineers and researchers** compare language models beyond a single
accuracy score. It supplies claims and evidence from FEVER and VitaminC, then measures model
behaviour across four dimensions. It evaluates models; it does not train them, retrieve evidence
from the web or act as an end-user fact-checking service.

| Dimension | Question | Experiment | Dataset |
|---|---|---|---|
| Correctness | Does it return the expected verdict? | Baseline | FEVER |
| Consistency | Does it repeat the same verdict on the same input? | Five runs with fixed seeds | FEVER |
| Robustness | Does the verdict stay stable when evidence is reordered or a distractor is added? | Reorder and distractor | FEVER |
| Evidence sensitivity | Are both supporting and refuting evidence versions answered correctly? | Contrastive pairs | VitaminC |

The model may answer `SUPPORTED`, `REFUTED` or `NOT ENOUGH INFORMATION` (NEI). Definitions and
caveats are in [docs/methodology.md](docs/methodology.md).

## Current project

The local Streamlit app and guided macOS launcher are implemented. The app opens in a browser
but runs on the user's computer, with two workflows:

- **Compare models:** explore recorded full experiments, or compare compatible saved local runs.
  Saved comparisons do not require loading a model or repeating inference.
- **Try the evaluator:** connect to local Ollama, select an installed text model and tests,
  choose a quick demo or custom workload, optionally estimate runtime, run/pause/resume,
  inspect scores and save/export results.

Saved full experiments cover **gemma3:4b, llama3.2:1b, llama3.2:3b and phi4-mini:3.8b** across
all four dimensions. Keep the evaluation scopes distinct:

| Scope | FEVER claims | VitaminC pairs | Purpose |
|---|---|---|---|
| Historical frozen pilots | 150 | 100 | Initial research experiments and regression checks |
| Saved full experiments | 13,332 | 20,232 | Recorded full-benchmark comparisons |
| Local quick demo | 3 | 3 | Check the workflow, not rank models |
| Local custom workload | 1–100% of the eligible full population | 1–100% of the eligible full population | User-selected local evaluation |

Only datasets needed by the selected tests are used. Reordering requires multi-sentence evidence.
Local runs remain separate from historical experiments even at 100% workload.

## Pilot results: `llama3.2:3b`

| Dimension | Headline result |
|---|---|
| Correctness | 97/150 = **64.67%** (SUPPORTED 77.33%, REFUTED 52.00%; 35 answers were NEI) |
| Stability | **80%** of claims identical in 5/5 runs; 35 claims wrong the same way every run |
| Robustness | Reordering: 8/11 unchanged. One distractor sentence: only **39.33%** unchanged, accuracy 64.67% to 14.67%, NEI answers 35 to 104 |
| Sensitivity | Verdict changed on 72/100 pairs but was **correct on only 27/100** |

These historical pilot results illustrate why the four dimensions are reported separately.
They are not the full-run scores shown in the app. Reports and schemas are under
[results/](results/README.md).

## Quickstart

### Local ModelLens app

For a shared Mac ZIP, extract it and open **ModelLens.app**, following
[the recipient instructions](docs/macos-start-here.txt). No repository clone is needed.
From a developer checkout, run `make app`; the launcher offers guided setup if needed.
Model inference requires Ollama and a compatible installed local text model. No ngrok or hosted
model service is required. The prototype is not Apple-notarized.

See the [app guide](app/README.md) for setup, controls, storage, limitations and packaging.

### Experiment setup

Follow [contributor setup and checks](CONTRIBUTING.md#setup) for the Python environment.
Research inference requires running Ollama with the exact installed tag passed to `--model`.
`llama3.2:3b` is the CLI default, not a requirement to use that model.
The experiments call the Ollama HTTP API and do not download models automatically.

**Running a model for the team?** Follow [Run a new model](#run-a-new-model).

The command-line interface can be started three equivalent ways:

```bash
uv run factcheck <command>
uv run python -m llm_fact_checker <command>
uv run python -m llm_fact_checker.baseline <command>   # original form, still supported
```

## Data

The two fixed samples are committed under `data/*/processed/`. The raw datasets are downloaded
separately (FEVER's Wikipedia dump is several GB); see [data/README.md](data/README.md) for sources,
layout and hashes. Which raw data you need depends on what you run:

| You want to | Raw data needed |
|---|---|
| Run the FEVER baseline, consistency or robustness | None (the pilot and full prepared claims are committed) |
| Run the VitaminC sensitivity experiment, on the 100 pairs or on every pair | None: `test.jsonl` is committed (the run re-validates the pairs against it) |
| Re-create the FEVER sample (`prepare`) | FEVER dev file + Wikipedia dump |

## Run the experiments

Inference commands protect existing prediction files unless resume/replacement is requested,
and smoke tests (`--limit`, `*-smoke`) write to their own folders. Every command that calls a model takes
`--model` (default `llama3.2:3b`); see [Run a new model](#run-a-new-model).

The repository already contains completed experiments. The commands below document the research
workflow, not work still required to use the app. Do not overwrite the committed results.

**CLI resume limitation:** keep the model weights, inputs, prompts and settings unchanged when
resuming. Unlike the app's worker, the research CLI does not fully validate run identity before
reusing answers. For changed experiments, select a new output directory. Do not use an existing
completed directory for an interrupted `--overwrite` replacement: old completion files can remain.

```bash
# FEVER baseline
uv run factcheck prepare                        # rebuild the 150-case sample from raw data
uv run factcheck run --limit 3                  # 3-case smoke test
uv run factcheck run                            # all 150 cases
uv run factcheck evaluate                       # re-score saved predictions, no Ollama

# Five-run consistency (750 requests, temperature 0.2, seeds 101..505)
uv run factcheck consistency --limit 3
uv run factcheck consistency

# Evidence robustness (the perturbed inputs are committed under data/fever/prepared/)
uv run factcheck robustness-smoke
uv run factcheck robustness

# VitaminC sensitivity (200 requests; reads the committed data/vitaminc/raw/vitaminc/test.jsonl)
uv run factcheck vitaminc-smoke
uv run factcheck vitaminc-sensitivity

# The same on every valid pair of the test set (40,464 requests); see "Run the whole VitaminC
# test set" below
uv run factcheck vitaminc-sensitivity --all --resume
```

`robustness-prepare` and `vitaminc-prepare` regenerate the perturbed inputs and pair audit from the
frozen samples. They do not depend on a model, so their output is shared and committed under
`data/fever/prepared/` and `data/vitaminc/prepared/`. Read `distractor_audit.md` and
`pair_audit.md` there if you change how perturbations or pairs are built.

## Run a new model

Everything you need is in the repo: the samples, the whole FEVER dev set and VitaminC's test file
are committed, so there is **nothing to download except your model**. The prompt, the data and the
seeds are identical for every model, which is what makes the numbers comparable, so do not change
them. Results land in `results/<dataset>/<model_slug>/<experiment>/`, so your run never touches
anyone else's.

**Before you start:** complete [contributor setup](CONTRIBUTING.md#setup), start Ollama and choose
a model that fits your hardware. Coordinate ownership of the result folder before running it.
The examples use Qwen, which already has pilot results here; use a new model/output location for
new work rather than replacing the recorded experiment.

### 1. Set up (once)

Use the environment and offline checks in [CONTRIBUTING.md](CONTRIBUTING.md#setup), then create
a branch for the additional experiment. Setup instructions are maintained there rather than
duplicated for each research workflow.

### 2. Get the model and name it

```bash
ollama pull qwen2.5:7b
ollama list                                      # the tag must match what you pass to --model
MODEL=qwen2.5:7b                                 # set these two in the terminal you will use
SLUG=qwen2.5_7b                                  # the tag with ":" and "/" replaced by "_"
```

Use the exact tag shown by `ollama list`. Renaming a model does not establish which weights it
contains; retain the model digest recorded with the experiment.

### 3. Smoke test (about a minute)

```bash
uv run factcheck run --model $MODEL --limit 3
cat results/fever/$SLUG/smoke_3/metrics.json     # want: inference_failure_count 0
# Keep smoke outputs separate from reported experiments; do not commit or cite them.
```

If this fails, fix it here before spending hours: see the table at the end.

### 4. Run the experiments

**First, the frozen samples** (150 FEVER claims, 100 VitaminC pairs). These are the numbers every
model is compared on. About 1,400 model calls: roughly 6 minutes for a 3B model, probably 15 to
30 for a 7B.

```bash
uv run factcheck run --model $MODEL                      # correctness
uv run factcheck consistency --model $MODEL              # stability (5 runs)
uv run factcheck robustness --model $MODEL               # reorder + distractor
uv run factcheck vitaminc-sensitivity --model $MODEL     # sensitivity (100 pairs)
```

**Then the whole FEVER dev set** (13,332 claims), overnight. One script runs baseline, robustness
and consistency in order:

```bash
caffeinate -i scripts/run_whole_fever.sh $MODEL 2>&1 | tee /tmp/whole_$SLUG.log
```

| Whole-set run | Time |
|---|---|
| Llama 3.2 3B on a fanless MacBook Air (measured) | about 1 h for the baseline, then 6.6 h for robustness and consistency |
| A 7B model | slower, probably 2 to 3 times (not measured) |

Keep the laptop **plugged in with the lid open** and do not switch branches or edit files while it
runs. If it stops for any reason (sleep, Ctrl-C), run the **same command again**: every step
resumes where it left off and finished steps are skipped.

**Optionally, the whole VitaminC test set** (20,232 contrastive pairs, 40,464 requests; see
[Run the whole VitaminC test set](#run-the-whole-vitaminc-test-set)):

```bash
caffeinate -i uv run factcheck vitaminc-sensitivity --all --resume --model $MODEL
```

### 5. Check the results, then commit and open a PR

```bash
uv run python - <<EOF
from llm_fact_checker.results import discover_runs, select
for run in select(discover_runs(), model="$SLUG"):
    print(run.label, run.headline())
EOF

git add results/fever/$SLUG results/vitaminc/$SLUG
git status                          # only results/*/<slug>/ files should be listed
git commit -m "Add $MODEL results"
git push -u origin HEAD
```

Open a pull request and put in the description: the model tag (the digest is in each
`run_config.json`), which experiments you ran, the four headline numbers (baseline accuracy,
consistency 5/5 agreement, distractor invariance, both-evidence-correct sensitivity), and anything unusual, such as
a high invalid-format rate. The large per-case files of whole-set runs are git-ignored on purpose.

Re-score saved baseline predictions at any time, without the model, with
`uv run factcheck evaluate --model $MODEL`.

### If something goes wrong

| You see | What it means and what to do |
|---|---|
| `Required Ollama model '…' is not installed` | The tag does not match `ollama list` exactly. Copy it from there, or alias it with `ollama cp` |
| `Ollama is not reachable` | Start the Ollama app (or `ollama serve`) and run the command again |
| `Refusing to overwrite …` | Preserve completed records. Resume only an unchanged interrupted experiment; use a new output directory for a replacement |
| The run stopped partway | Run the same command again with `--resume` (the whole-set script resumes by itself) |
| High `invalid_format_count` | The model does not follow "return the label only". That is a result, not a bug: do not change the prompt, mention it in your PR |
| `inference_failure_count` above 0 | Some requests were rejected or timed out. They are recorded per case, not lost; report the count in your PR |
| Very slow, laptop hot | Expected for 7B and larger. Plug in, lid open, and start with the frozen samples only |
| Out of memory | Use a smaller model or a more heavily quantised tag |

## Run the whole FEVER dev set

The 150-case sample is the frozen comparison set. The baseline can also be run on every
SUPPORTS/REFUTES claim in the FEVER dev file, as decided in the data audit
(`notebooks/01_data_audit_fever.ipynb`):

```bash
# data/fever/full/fever_dev_full.jsonl is already in the repo. Only to rebuild it from raw data:
# uv run factcheck prepare --all        # 13 s; needs data/fever/raw/ and the Wikipedia dump
uv run factcheck run --model llama3.2:3b --limit 3 \
    --prepared data/fever/full/fever_dev_full.jsonl --results-dir /tmp/smoke      # smoke test
caffeinate -i uv run factcheck run --model llama3.2:3b \
    --prepared data/fever/full/fever_dev_full.jsonl \
    --results-dir results/fever/llama3.2_3b/baseline_full
```

- **What is in the file:** all 13,332 claims (6,666 SUPPORTED + 6,666 REFUTED). Nothing is
  de-duplicated and nothing is removed because it was flagged. Accented Wikipedia titles are
  matched in NFC form, so all evidence resolves. The 68 claims the audit flagged as possibly
  inconsistent carry a `flags` field.
- **Two results:** `metrics.json` is the main result on all 13,332 claims;
  `metrics_excluding_flagged.json` and a section of `report.md` give the second result without the
  flagged claims (13,264), as the audit decided.
- **If the run is interrupted** (sleep, closed lid, Ctrl-C): every answer is saved as it arrives.
  Run the same command with `--resume`; it keeps what was saved, retries any failed cases and
  asks the model only for the missing ones.
- **Time:** about an hour for a 3B model on a fanless Apple-silicon laptop (the Llama run took
  about 0.25 s per call once the machine warmed up; the first minutes were 0.16 s). Other machines
  and larger models differ. `predictions.jsonl` is about 9.5 MB: commit the metrics and report and
  share the predictions by link.

### Everything on the whole set, unattended

Robustness and consistency also run on the whole set, with the same flagged-records second result.
One script runs all five steps in order (build the file, build the perturbed inputs, baseline,
robustness, consistency):

```bash
caffeinate -i scripts/run_whole_fever.sh llama3.2:3b 2>&1 | tee whole_run.log
```

For an unchanged experiment, every step saves answers as it runs, resumes from checkpoints,
and is skipped once finished. The CLI resume limitation above still applies.
Results go to `results/fever/<model_slug>/baseline_full/`,
`robustness_full/` and `consistency_5run_full/`. About 108,000 model calls in total. Measured for
Llama 3.2 3B on a MacBook Air: baseline about 1 h, then robustness and consistency together
6.6 h (94,528 calls). Plan for 7 to 8 hours per 3B model. The individual commands
(`factcheck run`, `robustness`, `consistency`, each with `--prepared`, `--results-dir` and
`--resume`) are inside the script.

## Run the whole VitaminC test set

The frozen comparison uses 100 pairs. The sensitivity experiment can also run on **every valid
contrastive pair** in VitaminC's test file: rows that share a `case_id` and an exact claim and
have exactly one SUPPORTS and one REFUTES row with different evidence. That is 20,232 pairs, so
40,464 requests. `data/vitaminc/raw/vitaminc/test.jsonl` is committed and the pairs are rebuilt
from it on every run, so there is nothing to download or prepare.

```bash
# smoke test: the first 3 pairs of the whole set, written to its own smoke/ folder
uv run factcheck vitaminc-smoke --all --model llama3.2:3b

# the whole set
caffeinate -i uv run factcheck vitaminc-sensitivity --all --resume --model llama3.2:3b \
    2>&1 | tee vitaminc_whole.log
```

- **Where it goes:** `results/vitaminc/<model_slug>/sensitivity_full/`, next to the 100-pair
  `sensitivity/` folder, which is untouched. The metrics, report, per-pair CSV and `run_config.json`
  are committed; `predictions/` is git-ignored, like the other whole-set runs.
- **What is in the set:** nothing is de-duplicated. 18,352 distinct claims occur in the 20,232
  pairs; 1,051 claims appear in more than one pair, each from a different case with its own
  evidence, and each pair is kept. `run_config.json` records `pair_selection: whole`,
  `full_pair_count` and `unique_claims`. The 100-pair sample, by contrast, has one pair per claim.
- **If it is interrupted** (sleep, closed lid, Ctrl-C): every answer is saved as it arrives. Run
  the same command again; `--resume` keeps what was saved, retries failed requests and asks the
  model only for the rest. The metrics file is written last, so a run is "finished" only once it
  exists.
- **Time:** 40,464 requests. Measured for Llama 3.2 3B on a fanless MacBook Air: 0.24 s per call
  on average, 2.7 hours of model time in total. A 6-call smoke test is faster (0.14 s per call)
  because the laptop has not warmed up yet, so do not extrapolate from it. Larger models take
  longer.
- **Read it with the sample's result in mind:** the sensitivity rate over 20,232 pairs is a much
  tighter estimate than 27/100, and the two should agree within the sample's noise. If they do
  not, the 100 pairs were not representative; say so in the PR.

## Repository layout

```text
app/                  local Streamlit UI, comparison charts, workload planning and worker jobs
llm_fact_checker/
├── cli.py            command-line interface (argparse only)
├── common.py         JSONL/CSV helpers, SHA-256, git commit, formatting
├── prompts.py        the frozen prompt, label set and strict output parser
├── models.py         default model and the model-tag to folder-name slug
├── ollama.py         Ollama HTTP client and inference loop
├── evaluation.py     baseline metrics and report
├── results.py        read-only loader for results/ (used by comparisons and the app)
├── data_prep/        fever.py, vitaminc.py: sampling and evidence extraction
└── experiments/      baseline.py, consistency.py, robustness.py, sensitivity.py
data/                 raw (ignored), processed samples, shared prepared/audit files, HASHES.txt
results/              research outputs; summaries tracked, large full predictions ignored
.modellens/           local evaluator history and tool state, ignored
scripts/              Mac launcher/packaging and research workflow helpers
notebooks/            data audits and analysis
docs/                 methodology and Mac recipient instructions
tests/                core, UI, worker, packaging and golden regression tests
dist/                 generated release ZIPs and verification records, ignored
```

## Tests

See [contributor verification](CONTRIBUTING.md#verification) for the test and lint commands.
Offline tests use fixtures, fake model responses and saved records, not real model inference.
Golden tests protect the recorded pilot metrics, reports and prompt. Passing tests is separate
from visually checking the app and testing an exact release ZIP on another Mac.

## Scope and limitations

- The selected FEVER/VitaminC evaluations use SUPPORTS/REFUTES gold labels, not gold NEI cases.
  NEI predictions do not count as correct on these cases.
- FEVER accuracy means agreement with FEVER labels, which contain known annotation noise. It is not
  absolute truth.
- Consistency measures repeatability, not correctness.
- The experiments supply gold evidence. They do not test evidence retrieval, calibration, or
  real-world fact-checking.
- Quick demos and historical pilots are small. Do not treat demo scores as model rankings or
  assume small score differences establish a meaningful performance advantage.
- Private dataset import and non-Ollama model backends are not implemented. Local execution is
  not a privacy/compliance certification. Review your model and environment before sensitive use.
- The Mac launcher has no automatic updater or migration of local runs between different builds.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for branches, frozen files and the results policy.
