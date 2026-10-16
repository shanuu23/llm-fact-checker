#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
source scripts/macos_setup.sh
trap pause_on_exit EXIT
require_mac

if ! app_ready; then
    ask "The app environment is missing/incomplete. Open guided setup now?" || exit 0
    MODELLENS_SETUP_CONTINUE=1 bash "Setup ModelLens.command"
    app_ready
fi
offer_start_ollama

# Local opt-in, loopback binding, no Streamlit usage telemetry.
export MODELLENS_LOCAL=1
if [ "${MODELLENS_DESKTOP:-0}" = 1 ]; then
    exec .venv/bin/python scripts/desktop_server.py
fi
exec .venv/bin/python -m streamlit run app/main.py \
    --server.address=127.0.0.1 --browser.gatherUsageStats=false
