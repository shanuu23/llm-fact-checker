#!/bin/bash
# Called by the approved app before starting Terminal; never write into the signed bundle.
set -euo pipefail
umask 077
resources=$(dirname -- "$1")
support="$2"
fail() { printf '%s\n' "$1" >&2; exit 1; }

release=$(cat "$resources/release-id.txt")
[[ "$release" =~ ^[0-9a-f]{64}$ ]] || fail "Invalid ModelLens build identifier. Get a fresh ZIP."
actual=$(/usr/bin/shasum -a 256 "$resources/payload.zip")
[ "${actual%% *}" = "$release" ] || fail "ModelLens payload is damaged. Get a fresh ZIP."
[[ "$support" = /* ]] || fail "ModelLens needs an absolute Application Support path."
ancestor="${support%/}"
while [ -n "$ancestor" ] && [ "$ancestor" != / ]; do
    [ ! -L "$ancestor" ] || fail "ModelLens Application Support contains a symlink; nothing was changed."
    ancestor=$(dirname -- "$ancestor")
done
mkdir -p -- "$support"
workspace="$support/$release"
[ ! -L "$workspace" ] || fail "ModelLens workspace is a symlink; nothing was changed."

if [ ! -e "$workspace" ]; then
    # Exclusive mkdir prevents two first launches from extracting into the same workspace.
    mkdir -- "$workspace" || fail "Another launch is preparing ModelLens. Try reopening shortly."
    /usr/bin/ditto -x -k "$resources/payload.zip" "$workspace"
    printf '%s\n' "$release" > "$workspace/.payload-id"
fi
[ -f "$workspace/.payload-id" ] && [ ! -L "$workspace/.payload-id" ] && \
    [ "$(cat "$workspace/.payload-id")" = "$release" ] || \
    fail "Incomplete or unrecognised ModelLens workspace: $workspace. Nothing was overwritten. If setup was interrupted, retain this folder for recovery and contact the team."
[ -d "$workspace/ModelLens" ] && [ ! -L "$workspace/ModelLens" ] && \
    [ -f "$workspace/ModelLens/Launch ModelLens.command" ] && \
    [ ! -L "$workspace/ModelLens/Launch ModelLens.command" ] || \
    fail "ModelLens workspace is incomplete: $workspace. Nothing was overwritten."
printf '%s\n' "$workspace/ModelLens"
