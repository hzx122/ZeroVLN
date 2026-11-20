#!/usr/bin/env bash
# Provision pi's search tools from the shared, preinstalled runtime bundle.
set -euo pipefail
: "${PI_CODING_AGENT_DIR:?Set PI_CODING_AGENT_DIR before preparing pi tools}"
VLN_PI_TOOLS_DIR=${VLN_PI_TOOLS_DIR:-/opt/external/user-d5121be6/tools/pi-search/linux-x64}
mkdir -p "$PI_CODING_AGENT_DIR/bin"
for tool in fd rg; do
    target="$PI_CODING_AGENT_DIR/bin/$tool"
    if [[ -x "$target" ]] && "$target" --version >/dev/null 2>&1; then
        continue
    fi
    source_tool="$VLN_PI_TOOLS_DIR/$tool"
    if [[ ! -x "$source_tool" ]] || ! "$source_tool" --version >/dev/null 2>&1; then
        echo "[pi-tools] Missing or unusable $source_tool; provision the shared fd/rg bundle first" >&2
        exit 1
    fi
    ln -sfn "$source_tool" "$target"
    echo "[pi-tools] $tool ready: $target"
done
