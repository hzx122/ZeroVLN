#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5-4B raw instruct baseline for the VLN demo.  The model path and vLLM
# architecture support are reused from VLA-RLVR's established 4B experiments.
MODEL=${QWEN35_4B_MODEL:-/opt/external/user-e7a01947/Model/Qwen3.5-4B}
VLLM_GPU=${VLLM_GPU:-4}
VLLM_PORT=${VLLM_PORT:-28901}

export PYTHONPATH=/workspace:${PYTHONPATH:-}
export VLLM_LOGGING_CONFIG_PATH=/workspace/vln_demo/vllm/vllm_log_hidden_id.json

CUDA_VISIBLE_DEVICES="$VLLM_GPU" \
vllm serve "$MODEL" \
    --allowed-local-media-path / \
    --served-model-name Qwen3.5-4B \
    --port "$VLLM_PORT" \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --max-model-len 65536 \
    --limit-mm-per-prompt '{"image":64}' \
    --enable-log-requests
