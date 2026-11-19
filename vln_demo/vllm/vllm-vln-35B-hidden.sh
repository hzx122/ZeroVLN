#!/usr/bin/env bash
set -euo pipefail

# Compatibility for existing demo commands. Production launch ownership lives
# on the Verl Agent server.
REPO=/workspace/ZeroVLN
exec bash "$REPO/verl/action_head/serve/vllm-qwen36-hidden.sh" "$@"
