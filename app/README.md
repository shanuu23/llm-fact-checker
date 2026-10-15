# LLM Evaluator dashboard

From the repository root:

```bash
uv run --group app streamlit run app/main.py
```

The overview explains the four tests. **Open comparison** reveals two model dropdowns and a
**Compare** button. Submitted results stay visible while using other controls. **Try the evaluator**
is a disabled card; no live inference is implemented.

Models are discovered through `llm_fact_checker.results` from the four standard full-run folders:
FEVER `baseline_full`, `consistency_5run_full`, `robustness_full`, and VitaminC `sensitivity_full`.
Pilot and smoke runs are not used. An unavailable experiment is displayed as missing, not zero.
The app checks the recorded input hashes, prompts, temperatures, seeds and case counts before
comparing each experiment. This checks the recorded settings, not identical hardware or runtimes.

Charts and percentage-point differences use the saved full metrics. Reordering accuracy uses the
reordering subset on both sides; invariance and VitaminC flip rates retain their recorded
complete-pair denominators. No overall ranking or scoring changes are introduced.

The example tables read committed CSVs, so ignored prediction logs and raw datasets are not needed.
Baseline examples contain errors only. The other examples are the first 10 rows in file order.
Keep `results/` alongside the app when deploying. Model names are populated from those results;
installed Ollama models are not needed for this dashboard.

Check the dashboard and comparison adapters without calling any models:

```bash
uv run --group app python -m unittest discover -s tests -p 'test_app.py' -v
```
