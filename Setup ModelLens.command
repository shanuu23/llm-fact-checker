#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
source scripts/macos_setup.sh
trap pause_on_exit EXIT
require_mac

echo "ModelLens setup. Close the app and pause any evaluations before continuing."
if ! MODELLENS_UV=$(find_tool uv); then
    install_uv
    MODELLENS_UV=$(find_tool uv)
fi
"$MODELLENS_UV" --version
echo "Using existing uv at $MODELLENS_UV; no uv upgrade will be performed."
if [ -L .venv ]; then
    echo "The .venv folder is a symlink. Refusing to modify an external environment."
    exit 1
fi
echo "This step sets up .venv here, reusing Python 3.12 if available or downloading it if needed."
echo "uv may also store downloads/Python in its normal per-user cache. No system Python changes."
ask "Set up the locked app dependencies (internet may be required)?" || exit 0
# Preserve extra developer packages when setup is run from a working checkout.
UV_PROJECT_ENVIRONMENT="$PWD/.venv" "$MODELLENS_UV" sync --locked --inexact \
    --no-default-groups --group app --python 3.12
app_ready

if ! ollama_running && ! find_ollama_app >/dev/null && ! find_tool ollama >/dev/null; then
    install_ollama
fi
offer_start_ollama
if [ "${MODELLENS_SETUP_CONTINUE:-0}" = 1 ]; then
    echo "App setup complete. ModelLens will open next."
else
    echo "App setup complete. Open ModelLens.app (or Launch ModelLens.command in a checkout)."
fi
echo "Models are not downloaded automatically. Choose/install your own local text model in Ollama."
