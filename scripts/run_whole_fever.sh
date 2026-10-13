#!/usr/bin/env bash
# Run every FEVER experiment on the WHOLE dev set for one model, one after another.
#
#   caffeinate -i scripts/run_whole_fever.sh llama3.2:3b
#
# Safe to stop and start again: every step resumes where it left off, and a step that already
# finished is skipped. Results go to results/fever/<model_slug>/{baseline,robustness,
# consistency_5run}_full/. Set OLLAMA_URL to use a server other than the default.
#
# Needs only the committed data/fever/full/fever_dev_full.jsonl: the perturbed inputs are rebuilt
# from it in about 12 s, so neither the raw FEVER data nor the Wikipedia dump is required.
set -euo pipefail

MODEL="${1:?usage: scripts/run_whole_fever.sh <ollama-model-tag>   e.g. llama3.2:3b}"
SLUG="${MODEL//[:\/]/_}"
URL="${OLLAMA_URL:-http://localhost:11434}"
FULL="data/fever/full/fever_dev_full.jsonl"
PREPARED="data/fever/full/prepared"
OUT="results/fever/${SLUG}"

cd "$(dirname "$0")/.."
started=$(date +%s)
step() { printf '\n=== [%s] %s\n' "$(date +%H:%M:%S)" "$1"; }

step "1/5 whole dataset file ($FULL)"
if [ -f "$FULL" ]; then echo "already built"; else uv run factcheck prepare --all; fi

step "2/5 perturbed inputs for robustness ($PREPARED)"
if [ -f "$PREPARED/distractor_evidence.jsonl" ]; then
  echo "already built"
else
  uv run factcheck robustness-prepare --prepared "$FULL" --output-dir "$PREPARED"
fi

step "3/5 baseline (correctness)  ->  $OUT/baseline_full"
uv run factcheck run --model "$MODEL" --prepared "$FULL" \
  --results-dir "$OUT/baseline_full" --ollama-url "$URL" --resume

step "4/5 robustness (reorder + distractor)  ->  $OUT/robustness_full"
uv run factcheck robustness --model "$MODEL" --prepared "$FULL" --prepared-dir "$PREPARED" \
  --robustness-dir "$OUT/robustness_full" --ollama-url "$URL" --resume

step "5/5 consistency (5 runs)  ->  $OUT/consistency_5run_full"
uv run factcheck consistency --model "$MODEL" --prepared "$FULL" \
  --results-dir "$OUT/consistency_5run_full" --ollama-url "$URL" --resume

printf '\n=== finished in %d min. Results are under %s\n' "$(( ($(date +%s) - started) / 60 ))" "$OUT"
