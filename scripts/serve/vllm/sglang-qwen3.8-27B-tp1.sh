#!/usr/bin/env bash
# Single-card sidecar for the 16-GPU DLC layout; keep the TP8 launcher separate.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=15
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
PYTHON=${VLN_27B_PYTHON:-/usr/local/bin/python3}
if [[ "${1:-}" == --check ]]; then
  exec "$PYTHON" -m sglang.launch_server --help
fi
exec "$PYTHON" -u -m sglang.launch_server \
  --model-path /opt/external/user-d5121be6/model/Qwen3.8-27B \
  --served-model-name qwen3.8-27b \
  --tp-size 1 --context-length 65536 --max-running-requests 4 \
  --mem-fraction-static 0.8 --host 0.0.0.0 --port 9000 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-metrics --enable-cache-report
