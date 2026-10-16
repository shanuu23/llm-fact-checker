# ModelLens: project context

Evidence Lab / Group 10 project. ModelLens is a delivered **local LLM evaluation prototype**
for AI engineers and researchers, not an end-user fact-checking service. It measures correctness,
consistency, robustness and evidence sensitivity separately using supplied FEVER/VitaminC
evidence. The Streamlit app supports saved comparisons and live evaluation through local Ollama.
Nothing is trained, and no combined model ranking is calculated.

Read `README.md` for the product overview and research commands, `app/README.md` for the app
workflow and packaging, `CONTRIBUTING.md` for developer setup and checks,
`docs/methodology.md` for definitions, and `results/README.md` for the research output schema.

## Commands

```bash
make app                             # local app; launcher offers setup if needed
uv run factcheck <command>            # research CLI, not the app's worker
```

CLI commands: `prepare`, `run`, `evaluate`, `consistency`, `robustness-prepare`, `robustness-smoke`,
`robustness`, `vitaminc-prepare`, `vitaminc-smoke`, `vitaminc-sensitivity`, `all`. Run from the
repository root: paths are relative.

## Layout

```text
app/              Streamlit UI, charts, interpretation, local workloads and worker jobs
llm_fact_checker/
  cli.py          argparse only          prompts.py     FROZEN prompt, labels, parser
  common.py       shared helpers         ollama.py      HTTP client + inference loop
  evaluation.py   baseline metrics       data_prep/     fever.py, vitaminc.py
  results.py      read-only results loader   models.py   default model + model_slug()
  experiments/    baseline, consistency, robustness, sensitivity
data/             benchmark inputs, prepared cases, audits and HASHES.txt
results/          research outputs; summaries tracked, large full predictions ignored
.modellens/       private local runs and tool state, Git-ignored
scripts/          Mac packaging/launch support and research commands
tests/            offline core, UI, worker, packaging and golden regression checks
notebooks/        data audits and historical comparison analysis
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

- The core experiment pipeline is **standard library only**. UI dependencies belong in `app`;
  notebook and tooling libraries belong in the `notebooks` / `dev` groups.
- Shared helpers go in `common.py`; do not duplicate them per module.
- Ruff: line length 99, import sorting on. Keep `make lint` clean.
- A refactor must not change scoring or saved records. Run the affected offline tests and check
  the diff; do not regenerate committed results merely to validate a refactor.
- Tests use synthetic fixtures, fake model responses and saved benchmark artifacts. They do not
  call a real Ollama model or read the Wikipedia dump. Follow `CONTRIBUTING.md` for test setup.
- Use `git mv` for moves so history follows files.
- Preserve atomic prediction/metric writes and checkpoint flushing. CLI runners use the final
  metrics file as a completion marker; the app also tracks worker status. These are distinct paths.
- Core Python files plus `app/evaluator.py` and `app/jobs.py` contribute to the local protocol
  fingerprint. Changes can prevent old runs from resuming or comparing with new runs.
- Do not rebuild/share ZIPs, change frozen artifacts or run real inference as an incidental
  documentation or UI cleanup step.

## Gotchas

- `llama3.2:3b` is the CLI default, not the only supported model. Use the exact installed tag
  with `--model`; the app lists compatible local text models.
- The Wikipedia dump path is doubled on purpose: `data/fever/raw/wiki-pages/wiki-pages/`.
- The FEVER sample has no gold NEI cases, so any NEI answer counts as wrong there.
- NFC page-title matching is opt-in (`scan_wikipedia(match_nfc=True)`, used by `prepare --all`):
  turning it on for the frozen sample would change it (candidate FEVER 95159 would replace
  186995) and break its hash.
- `vitaminc-sensitivity` re-validates the committed pairs against `data/vitaminc/raw/vitaminc/test.jsonl`,
  which is committed (via `.gitignore` exceptions: the rest of `raw/` stays untracked).
- `VIRTUAL_ENV` warnings from uv are harmless; the project uses its own `.venv`.

## Delivered functionality

- Saved full comparisons cover `gemma3:4b`, `llama3.2:1b`, `llama3.2:3b` and `phi4-mini:3.8b`
  across all four tests. FEVER uses 13,332 claims; VitaminC uses 20,232 contrastive pairs.
  Historical pilot results remain separate, including Qwen 2.5 7B. Discover actual result folders
  through `llm_fact_checker.results`; do not substitute pilot scores for a missing full run.
- The local evaluator offers model selection, test selection, a quick demo or custom percentage,
  optional runtime estimation, background execution, pause/resume, saved history, result export
  and compatible local-run comparisons. See `app/README.md` for exact controls and constraints.
- Quick demo uses three FEVER claims and three VitaminC pairs when relevant. The historical
  frozen pilots use 150 claims / 100 pairs. Custom local workloads sample from the full eligible
  populations, not those pilot files. Do not merge these three scopes.
- The macOS ZIP launches `ModelLens.app`, which uses the `.command` files internally. It offers
  guided dependency setup and runs in a per-build workspace. Preserve both launch scripts.

## Limits and maintenance cautions

- This is a single-user local prototype, not a hosted service or compliance-certified product.
  Private dataset import and additional model-serving backends are not implemented.
- Mac builds are ad-hoc signed, not Apple-notarized. New payloads use separate workspaces;
  there is no automatic updater or run migration. Never delete old workspaces as cleanup.
- Research CLI resume assumes unchanged inputs, model and settings; it does not enforce the
  app's full identity checks. Use a new output location for changed experiments. Avoid restarting
  an interrupted overwrite in the old directory, where old completion files can remain.
- The baseline CLI records its input hash but does not enforce the frozen sample hash.
  Protect the committed samples and results independently of this permissive CLI path.
- Historical full comparisons check recorded inputs/settings, not identical hardware/runtime.
  Local comparisons additionally require matching workload IDs and Ollama versions.
