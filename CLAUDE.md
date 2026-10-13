# llm-fact-checker: project context

Group 10 project. A **reliability test bench for evidence-grounded LLM fact-checkers**, not a
fact-checking app. It answers "can this model be trusted to judge a claim from supplied evidence?"
by measuring four things separately: correctness, stability, robustness, sensitivity. Models are
local, via Ollama. Nothing is trained.

Read `docs/methodology.md` for definitions and `results/README.md` for the output schema.

## Commands

```bash
uv sync                               # installs everything (dev + notebooks groups)
make test                             # unit + golden tests; no Ollama or raw data needed
make lint                             # ruff check + format check
uv run factcheck <command>            # same as: python -m llm_fact_checker <command>
                                      # (python -m llm_fact_checker.baseline also works)
```

Commands: `prepare`, `run`, `evaluate`, `consistency`, `robustness-prepare`, `robustness-smoke`,
`robustness`, `vitaminc-prepare`, `vitaminc-smoke`, `vitaminc-sensitivity`, `all`. Run from the
repository root: paths are relative.

## Layout

```text
llm_fact_checker/
  cli.py          argparse only          prompts.py     FROZEN prompt, labels, parser
  common.py       shared helpers         ollama.py      HTTP client + inference loop
  evaluation.py   baseline metrics       data_prep/     fever.py, vitaminc.py
  results.py      read-only results loader   models.py   default model + model_slug()
  experiments/    baseline, consistency, robustness, sensitivity
data/             raw (gitignored), processed samples (tracked), HASHES.txt
results/          <dataset>/<model_slug>/<experiment>/, committed
tests/            unit tests + tests/test_golden_results.py
```

## Frozen: do not change without an explicit team decision

- `prompts.py` (`SYSTEM_PROMPT`, labels, parser) and the user-prompt format.
- `data/fever/processed/fever_baseline_150.jsonl` (sha256 `6bc7cf91…`) and
  `data/vitaminc/processed/vitaminc_sensitivity_100_pairs.jsonl`.
- Experiment settings: baseline temp 0 seed 42; stability temp 0.2 seeds 101/202/303/404/505;
  robustness and sensitivity temp 0 seed 42.
- Committed pilot results under `results/*/llama3.2_3b/`. Never hand-edit results or pass
  `--overwrite` on them. `tests/test_golden_results.py` pins their metrics, reports and the prompt.

## Conventions

- The pipeline is **standard library only**. Do not add runtime dependencies; notebook and tooling
  libraries belong in the `notebooks` / `dev` groups.
- Shared helpers go in `common.py`; do not duplicate them per module.
- Ruff: line length 99, import sorting on. Keep `make lint` clean.
- A refactor must not change behaviour: after it, the golden tests pass and `git status results`
  is empty after `factcheck evaluate`.
- Tests use tiny synthetic fixtures and never call Ollama or read the Wikipedia dump.
- Use `git mv` for moves so history follows files.
- Result files are written atomically (`common.write_jsonl`, `write_text_atomic`) and the metrics
  file is written LAST, because `--resume` treats its existence as "this run is finished". Keep
  that order when adding outputs.

## Gotchas

- `ollama.py` requires the **exact** model tag `llama3.2:3b`. If `ollama list` only shows
  `llama3.2:latest`, alias it: `ollama cp llama3.2:latest llama3.2:3b` (same weights, no download).
- The Wikipedia dump path is doubled on purpose: `data/fever/raw/wiki-pages/wiki-pages/`.
- The FEVER sample has no gold NEI cases, so any NEI answer counts as wrong there.
- NFC page-title matching is opt-in (`scan_wikipedia(match_nfc=True)`, used by `prepare --all`):
  turning it on for the frozen sample would change it (candidate FEVER 95159 would replace
  186995) and break its hash.
- `vitaminc-sensitivity` needs the raw `data/vitaminc/raw/vitaminc/test.jsonl` (it re-validates the
  committed pairs against it). The FEVER experiments run from the committed sample alone.
- `VIRTUAL_ENV` warnings from uv are harmless; the project uses its own `.venv`.

## Current state and pending work

Done: the Llama 3.2 3B pilot on all four experiments (committed under `results/`).

Pending, in order:

1. **Run the other models** (`qwen2.5:7b`, `gemma3:4b`). Multi-model support is in place: every
   model-calling command takes `--model` and writes to `results/<dataset>/<model_slug>/...`.
   Perturbed inputs and audits are model-independent and live in `data/*/prepared/`.
2. **Full-dataset run**: done for the FEVER baseline. `data/fever/full/fever_dev_full.jsonl` (all
   13,332 claims) is committed, so no raw data is needed to run it; `factcheck prepare --all`
   rebuilds it from raw data (same bytes) and builds all 13,332
   claims (no de-duplication, audit flags in `flags`, NFC title matching) and `factcheck run
   --prepared data/fever/full/fever_dev_full.jsonl --results-dir …/baseline_full [--resume]` runs
   it, saving each answer as it arrives and writing `metrics_excluding_flagged.json` alongside
   `metrics.json`. Consistency and robustness run on it too (`scripts/run_whole_fever.sh <model>`
   runs all steps unattended and resumes). Still frozen-sample only: VitaminC sensitivity (100
   pairs hard-coded).
3. **Sample size is still hard-coded** to 150 claims / 100 pairs in `consistency.py`,
   `robustness.py`, `sensitivity.py` and `data_prep/fever.py` (`per_label`). A larger sample needs
   those checks generalised.
4. **Baseline hash gate.** `run` now records the sample hash in `run_config.json` (as consistency
   and robustness do) but, unlike them, does not refuse to run on a different sample.
5. **Comparison and a Streamlit app** that reads `results/` only and never runs inference. The
   read-only loader `llm_fact_checker/results.py` exists (`discover_runs`, `select`,
   `ResultRun.headline/metrics/config/provenance/cases`); build the comparison and app on it
   rather than reading result files directly.
