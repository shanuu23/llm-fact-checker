#!/bin/bash
# Shared by the two macOS launchers. Sourcing this file performs no installation.

ask() {
    local answer
    read -r -p "$1 [y/N] " answer || return 1
    case "$answer" in y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
}

require_mac() {
    if [ "$(uname -s)" != Darwin ]; then
        echo "This launcher supports macOS only."
        return 1
    fi
    local version
    version=$(sw_vers -productVersion)
    if [ "${version%%.*}" -lt 14 ]; then
        echo "This setup requires macOS 14 or newer for Ollama."
        return 1
    fi
    case "$(uname -m)" in
        arm64) ;;
        x86_64) echo "Intel Mac: Ollama uses CPU inference; evaluations can be slow." ;;
        *) echo "Unsupported Mac architecture."; return 1 ;;
    esac
}

find_tool() {
    local name="$1" candidate
    candidate=$(command -v "$name" || true)
    if [ -n "$candidate" ]; then printf '%s\n' "$candidate"; return 0; fi
    # Finder-launched terminals do not always inherit Homebrew/user PATH entries.
    for candidate in "$PWD/.modellens/tools/$name" "$HOME/.local/bin/$name" \
        "$HOME/.cargo/bin/$name" "/opt/homebrew/bin/$name" "/usr/local/bin/$name"; do
        if [ -e "$candidate" ] || [ -L "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

find_ollama_app() {
    local candidate
    for candidate in /Applications/Ollama.app "$HOME/Applications/Ollama.app"; do
        if [ -e "$candidate" ] || [ -L "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

ollama_running() {
    curl --noproxy '*' --fail --silent --max-time 2 \
        http://127.0.0.1:11434/api/version >/dev/null 2>&1
}

download() {
    curl --proto '=https' --proto-redir '=https' --tlsv1.2 --fail --location \
        --show-error --connect-timeout 15 --max-time 1800 --output "$2" "$1"
}

install_uv() (
    set -e
    echo "uv is missing. It manages the app's Python and locked dependencies."
    echo "Source: https://astral.sh/uv/0.13.1/install.sh"
    echo "Destination: $PWD/.modellens/tools (no shell profile changes)."
    ask "Download and run the official uv installer?" || return 1
    local work
    work=$(mktemp -d)
    trap 'rm -f -- "$work/uv-install.sh"; rmdir -- "$work"' EXIT
    download https://astral.sh/uv/0.13.1/install.sh "$work/uv-install.sh"
    env UV_UNMANAGED_INSTALL="$PWD/.modellens/tools" sh "$work/uv-install.sh"
)

install_ollama() (
    set -e
    local destination="$HOME/Applications/Ollama.app" work mounted=0
    echo "Ollama was not found in the usual Applications folders or command paths."
    echo "Already installed elsewhere? Choose No and open that copy yourself."
    echo "Download: https://ollama.com/download/Ollama.dmg"
    echo "Destination: $destination. No models, sudo or PATH changes."
    ask "Download and install Ollama for this user?" || return 0
    # Never invoke Ollama's install.sh: it can delete an existing system installation.
    if find_ollama_app >/dev/null || find_tool ollama >/dev/null || ollama_running; then
        echo "Ollama is now present; skipping installation."
        return 0
    fi
    work=$(mktemp -d)
    cleanup_ollama_download() {
        if [ "$mounted" = 1 ]; then
            hdiutil detach "$work/mount" -quiet || {
                echo "Could not unmount $work/mount. Eject it in Finder; download retained."
                return
            }
        fi
        rm -f -- "$work/Ollama.dmg"
        rmdir -- "$work/mount" "$work" 2>/dev/null || true
    }
    trap cleanup_ollama_download EXIT
    download https://ollama.com/download/Ollama.dmg "$work/Ollama.dmg"
    mkdir "$work/mount"
    hdiutil attach -readonly -nobrowse -mountpoint "$work/mount" "$work/Ollama.dmg" -quiet
    mounted=1
    codesign --verify --deep --strict "$work/mount/Ollama.app"
    spctl --assess --type execute "$work/mount/Ollama.app"
    mkdir -p "$HOME/Applications"
    # mkdir fails if anything already occupies the exact destination, including a symlink.
    mkdir "$destination"
    if ! ditto "$work/mount/Ollama.app" "$destination"; then
        echo "Copy failed. Review the partial installation at $destination before retrying."
        return 1
    fi
    echo "Ollama installed. macOS/Ollama may ask for first-launch approval."
)

offer_start_ollama() {
    local app attempt
    if ollama_running; then echo "Ollama is already running locally."; return 0; fi
    if app=$(find_ollama_app); then
        if ask "Start the installed Ollama app?"; then
            echo "Approve any normal macOS/Ollama first-launch prompts."
            echo "You can decline moving Ollama or adding its CLI; ModelLens uses the local API."
            if open "$app"; then
                for attempt in {1..15}; do
                    if ollama_running; then echo "Ollama is ready."; return 0; fi
                    sleep 2
                done
            fi
        fi
    elif find_tool ollama >/dev/null; then
        echo "CLI-only Ollama found. Start 'ollama serve' in a separate terminal."
    else
        echo "Ollama is missing. Run Setup ModelLens.command when you want live evaluation."
    fi
    echo "Ollama is not ready. Saved comparisons still work; connect later for evaluation."
}

app_ready() {
    [ -x .venv/bin/python ] && .venv/bin/python -B -c \
        'import sys, streamlit, pandas; assert sys.version_info[:2] == (3, 12)' >/dev/null 2>&1
}

pause_on_exit() {
    local result=$?
    if [ "$result" -ne 0 ]; then
        echo "Setup/launch stopped. Review the error above; nothing will retry automatically."
        echo "Existing results and models have not been deleted."
    fi
    if [ -t 0 ]; then
        if [ "$result" = 0 ] && [ "${MODELLENS_SETUP_CONTINUE:-0}" = 1 ]; then
            read -r -p "Press Enter to continue to ModelLens. " _ || true
        else
            read -r -p "Press Enter to close. " _ || true
        fi
    fi
}
