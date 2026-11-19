#!/usr/bin/env bash
# Direct execution on the current GPU host; no scheduler or platform submission.
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$REPO/scripts/train:$REPO/scripts/eval:$REPO/verl:$REPO/vln_demo/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GPU_DIRECT_RUN=1
exec "${PYTHON:-python3}" -B "$REPO/scripts/train/run_collection_gpu.py" "$@"
