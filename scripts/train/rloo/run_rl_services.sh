#!/usr/bin/env bash
# Every RL node: use the SAME rloo.* Hydra overrides as run_rloo.sh.
# Foreground run owns renderer/Gym children; Ctrl-C cleans only those children.
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="$REPO/scripts/train:$REPO/scripts/eval:$REPO/verl:$REPO/vln_demo/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec "${PYTHON:-python3}" -B "$REPO/scripts/train/run_rl_services_gpu.py" "$@"
