#!/usr/bin/env bash
set -euo pipefail

# Compatibility for existing demo commands. Production launch ownership lives
# on the Verl Agent server.
REPO=/workspace/ZeroVLN
exec bash "$REPO/scripts/serve/action_head/vllm-qwen36-hidden.sh" "$@"
