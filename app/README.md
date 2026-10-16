# ModelLens — local evaluator

ModelLens is a single-user research prototype for evaluating evidence-grounded text models.
It uses Ollama on the same computer. No ngrok, account, tunnel, hosted model service or model
upload is required. This version targets macOS; the worker uses POSIX file locking.

## Setup and launch

For the recipient ZIP, extract it and open **ModelLens.app**. Read **START HERE.txt** first.
The app opens guided setup in Terminal; no shell commands need to be typed. It is ad-hoc signed
for bundle integrity, **not Apple-notarized**. A trusted copy may need normal macOS Open Anyway
approval and permission for ModelLens to control Terminal. Do not disable security or clear
quarantine. If macOS reports damage/harm or approval is unavailable, stop and contact the team.

The app prepares a writable workspace at
`~/Library/Application Support/ModelLens/<build-id>/ModelLens/`. Code, Python and saved runs
stay outside the sealed app bundle, including when macOS translocates the downloaded app.
Reopening the **same build** reuses this workspace and opens its existing local server if running.
It selects a free local port, independent of an existing developer server. Different payload
builds have separate workspaces; no updater, migration or deletion of previous runs is automatic.

From a developer checkout, **Setup ModelLens.command** and **Launch ModelLens.command** still work.
The supported target is macOS 14 or newer, Apple Silicon or Intel. Intel inference is CPU-only
and can be much slower. See [Ollama's Mac requirements](https://docs.ollama.com/macos).

Setup is guided, not silent. Type `y` and press Enter to approve an action; Enter alone means No.

1. Detect existing uv, including common user/Homebrew paths. Reuse it without upgrading it.
   If missing, offer the official, version-pinned uv installer (0.13.1), scoped to
   `.modellens/tools/`, with no shell-profile changes. This uses uv's documented
   [unmanaged installation option](https://docs.astral.sh/uv/reference/installer/).
2. Ask before setting up `.venv` with locked app dependencies. Reuse a compatible Python 3.12
   or let uv download it. System Python is not changed. Existing extra developer packages are
   retained. uv's normal per-user cache/Python locations may also be used.
3. Detect running Ollama, its CLI, or its app in system/user Applications. If missing, offer
   the [official Mac download](https://ollama.com/download/mac), check its code signature and
   Gatekeeper assessment, and copy it to `~/Applications/Ollama.app`. Never overwrite an existing
   installation, run `sudo`, disable Gatekeeper, change PATH, or download a model.
4. Offer to open the installed Ollama app, then wait briefly for its local API. You can decline
   and still use saved comparisons. On first launch, macOS or Ollama may require approval;
   moving it to system Applications or adding its CLI is not required by ModelLens.

Close ModelLens and pause evaluations before running setup. Initial downloads need internet
access and enough free storage. An organisation's IT restrictions still apply. If an existing
tool is broken/outdated, setup reports the failure rather than replacing/upgrading it silently.
An Ollama app in a nonstandard location can be opened manually; a CLI-only installation should
be started with `ollama serve` in a separate terminal. No service is started without permission.

For developers running from the repository, the equivalent is:

```bash
bash "Setup ModelLens.command"
bash "Launch ModelLens.command"
```

If setup was skipped, the launcher offers to open it. Subsequent launches reuse the environment
and can offer to start Ollama; they do not install or update dependencies automatically.
Choose/install a compatible **local** text model in Ollama yourself, then refresh the installed
model list in ModelLens. Model downloads can be large and are intentionally not automatic.
The launcher binds Streamlit
to `127.0.0.1` and disables its usage telemetry. The inference client connects only to
`127.0.0.1:11434`, bypasses proxies, rejects redirects and rejects cloud-routed/embedding-only
models. Do not expose this single-user app on a public interface. These controls are not a
compliance certification or a substitute for reviewing your Ollama/model configuration.

## Evaluate

1. Open **Try the evaluator**, then **Connect to Ollama** (or **Refresh models**).
2. Choose a model and one or more tests.
3. Start with **Quick demo**: three fixed FEVER claims and three complete VitaminC pairs.
   All four tests together require **31 calls**, including five consistency passes.
4. Optionally choose **Custom** and a 1–100% slider. It selects from **13,332 eligible FEVER
   gold-evidence claims** and **20,232 eligible VitaminC test SUPPORTS–REFUTES pairs**, not from
   the pilot files. The app shows actual counts and expected requests before starting.
5. Optionally click **Estimate runtime**. This makes up to three calls per distinct pass family
   (16 for the all-test quick demo). These are extra, unscored calls. The range extrapolates
   observed minimum/maximum latency, expires after an hour and is not a guaranteed finish time.
6. Click **Run evaluation**. No inference happens just from opening the page or moving the slider.

The existing prompts, exact-label parser, scoring functions and seed/temperature settings are
reused. Baseline, robustness and sensitivity use seed 42, temperature 0; consistency uses five
passes with seeds 101/202/303/404/505 and temperature 0.2. FEVER sampling interleaves deterministic
label-specific orders. Percentage selections are nested. Quick demo deliberately includes a
multi-sentence case; custom reordering only applies where evidence has multiple sentences.
Distractors come from the selected FEVER pool. Input snapshots and protocol hashes identify
the exact workload. Neither scope tests **gold NEI** accuracy. Known input flags are preserved
in predictions; local headline metrics include flagged cases. Dataset labels are not infallible.

## Save, pause and compare

Each run has its own folder in the workspace's `.modellens/runs/` (Git-ignored), with inputs, model identity,
settings, prediction checkpoints, status and final metrics. Existing `results/` is read-only.
Only one local evaluation/speed check can run at a time. Separate CLI jobs are not controlled
by this app: do not run those alongside it.

**Pause after current response** flushes the current response before stopping. A page refresh or
closed browser does not stop the detached worker. Pause through the app before shutting it down.
After a restart, reopen the app and select **Resume selected run**. Successful saved calls are
skipped; failed calls in unfinished passes are retried. A torn final checkpoint line is discarded.
A call interrupted before its answer was saved can be repeated. Sleep pauses computation; an
in-flight request can time out. Model digest, Ollama version, input or evaluation-code changes
block resume rather than silently mixing experiments. Do not update code/models during a run.
Completed runs with failures remain completed and show those failures in their metrics.

The **Evaluation** tab shows one stage at a time: setup, progress with Pause/Resume, then
results. Completion replaces the setup and progress display with visible score cards.
**New evaluation** returns to setup with previous choices retained; it does not start a run
or delete results. **Run history** keeps separate runs per model, with timestamp (UTC), status
and run ID. Repeated evaluations are never overwritten.

**Export results** in **Run history** creates a ZIP on demand containing metrics, model/run metadata, the
input plan, status and completed prediction logs. Raw responses and configuration stay out
of the results screen but remain available in this export. Large runs produce larger downloads.
**Evaluation guide & help** contains troubleshooting and scoring notes.

**Compare models → Local evaluator runs** offers separate model and saved-run selectors.
It defaults to the newest compatible pair for the selected models, where one exists. You can
also compare two distinct runs of the same model. Comparison
requires the same workload ID and Ollama version: identical selected cases, tests, settings and evaluation code.
Unrun tests are marked unavailable. Small demos are illustrative, not model rankings. Even a
100% local run stays separate from historical full experiments; the app does not assert they
share an identical protocol. Consistency/invariance can be high for consistently wrong answers.

## Saved full experiments

Models are discovered through `llm_fact_checker.results` from the four standard full-run folders:
FEVER `baseline_full`, `consistency_5run_full`, `robustness_full`, and VitaminC `sensitivity_full`.
Pilot and smoke runs are not used. An unavailable experiment is displayed as missing, not zero.
The app checks the recorded input hashes, prompts, temperatures, seeds and case counts before
comparing each experiment. This checks the recorded settings, not identical hardware or runtimes.

Charts and dynamic result interpretations use the saved full metrics. Reordering accuracy uses the
reordering subset on both sides; invariance and VitaminC flip rates retain their recorded
complete-pair denominators. No overall ranking or scoring changes are introduced.

Keep `results/` alongside the app. Saved-comparison models come from those results, not Ollama.

Both comparison sources share the same four chart tabs: paired accuracy bars, consistency
outcome stacks, separate reorder/distractor accuracy panels, and both-evidence-correct sensitivity.
Axes stay at 0–100%, missing scores display N/A, and labels identify models (with run IDs
when comparing the same model). Consistency colours describe outcomes, not model identity.
Compatibility checks still run before charts are displayed; no combined ranking is calculated.
Running Streamlit without `MODELLENS_LOCAL=1` leaves the evaluator disabled (read-only/hosted mode).
The app no longer reads or sends tunnel credentials. Your existing private ngrok configuration
and account are not modified; stop any old tunnel yourself if it is still running.

## Share and verify

`make app-package` builds `dist/modellens-macos-app.zip` on macOS, using Apple's `osacompile`,
`codesign` (ad-hoc only) and `ditto`. It contains **ModelLens.app** and **START HERE.txt**.
The embedded source payload comes from an explicit file allow-list, including
public benchmark inputs, their attribution and saved full-result summaries/case tables.
It excludes Git history, environments, credentials, smoke runs, private local results and large
prediction logs. It refuses to overwrite an existing bundle. This is a guided app launcher,
**not** a notarized standalone application with a bundled Python runtime. Recipients do not need Git, a repository clone,
Homebrew or preinstalled Python. Guided setup offers missing prerequisites; permission prompts
and model installation are still required. Retain
`data/README.md` for dataset attribution/licensing. No publication or release upload is automatic.

`make app-source-package` retains the old developer-only source export (`dist/modellens-macos.zip`).
Do not distribute it as the Finder-launch fix. To build a separately named candidate, use
`python scripts/package_app.py --macos-app --output dist/your-candidate.zip`.
Identical source payloads produce the same workspace ID, regardless of source file timestamps.
See [recipient instructions](../docs/macos-start-here.txt). Apple's guidance:
[embedded tools](https://developer.apple.com/forums/thread/706379) and
[normal app approval](https://support.apple.com/en-au/102445).

Check the dashboard and comparison adapters without calling any models:

```bash
uv run --group app python -m unittest discover -s tests -v
```

Automated tests use fake model responses. A real-model quick demo and a teammate's clean-machine
setup are separate acceptance checks, not implied by passing offline tests.
For each shared app build, also download/extract the exact ZIP, verify its bundle signature,
complete normal Mac approval, test first setup and reopen, and confirm existing results survive.
A passed harmless launch probe does not substitute for that full acceptance check.
