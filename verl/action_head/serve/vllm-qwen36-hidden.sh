#!/usr/bin/env bash
set -euo pipefail

# Qwen3.6 VLN server with vLLM 0.18 native verifier-hidden extraction.
# Correctness/latency experiment only: ExampleHiddenStatesConnector writes the
# full selected layer to CPFS.  Production will replace this connector with
# in-worker token selection + ActionFormer compression after parity is proven.
REPO=/workspace/ZeroVLN
MODEL=/opt/external/user-e7a01947/Model/Qwen3.6-35B-A3B
HIDDEN_CACHE=${ACTION_HEAD_HIDDEN_CACHE:-/workspace/action_head_hidden_cache}
COMPILE_CACHE=${ACTION_HEAD_COMPILE_CACHE:-/workspace/action_head_compile_cache}
export ACTION_HEAD_COMPILE_CACHE="$COMPILE_CACHE"
PORT=${ACTION_HEAD_VLLM_PORT:-28899}
GPU_IDS=${ACTION_HEAD_VLLM_GPUS:-14,15}
MAX_MODEL_LEN=${ACTION_HEAD_MAX_MODEL_LEN:-32768}
MAX_IMAGES=${ACTION_HEAD_MAX_IMAGES:-8}
GPU_MEMORY_UTILIZATION=${ACTION_HEAD_GPU_MEMORY_UTILIZATION:-0.90}
HIDDEN_LAYER=${ACTION_HEAD_HIDDEN_LAYER:-40}
export ACTION_HEAD_VLLM_DIAGNOSTIC=${ACTION_HEAD_VLLM_DIAGNOSTIC:-/workspace/discussion_logs/vllm_qwen36_hidden_diagnostic.log}

# Non-interactive DSW SSH sessions do not source the profile fragment that
# exposes the PPU CUDA/HGGC runtime.  Without this, torch/vLLM repeatedly emits
# "libhggcrt1.so: cannot open shared object file" and reports zero devices.
PPU_LIBS=/opt/accl-p:/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/lib:/usr/local/lib
export LD_LIBRARY_PATH="$PPU_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_PATH=${CUDA_PATH:-/usr/local/PPU_SDK/CUDA_SDK}
export PPU_SDK=${PPU_SDK:-/usr/local/PPU_SDK}
export PATH="$CUDA_PATH/bin:$PATH"
export PYTHONPATH="$REPO/verl/action_head/serve/bootstrap:$REPO/verl${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="$COMPILE_CACHE/xdg"
export TORCHINDUCTOR_CACHE_DIR="$COMPILE_CACHE/torchinductor"
export TRITON_CACHE_DIR="$COMPILE_CACHE/triton"
export CUDA_CACHE_PATH="$COMPILE_CACHE/cuda"
export DG_JIT_CACHE_DIR="$COMPILE_CACHE/deep_gemm"
export DG_CACHE_DIR="$DG_JIT_CACHE_DIR"

mkdir -p "$HIDDEN_CACHE" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" \
    "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$DG_JIT_CACHE_DIR"
# Qwen3.6 requests num_hidden_layers (=40), which vLLM documents as the final
# text-block output before output norm. Training, SFT bundles, and serving must
# keep this exact convention; overrides are explicit ablations, not
# checkpoint-compatible substitutions.
SPECULATIVE_CONFIG=$(printf '{"method":"extract_hidden_states","num_speculative_tokens":1,"draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":[%s]}}}' "$HIDDEN_LAYER")
KV_TRANSFER_CONFIG=$(printf '{"kv_connector":"ActionHeadHiddenStatesConnector","kv_connector_module_path":"action_head.serve.hidden_states_connector","kv_role":"kv_producer","kv_connector_extra_config":{"shared_storage_path":"%s"}}' "$HIDDEN_CACHE")

CUDA_VISIBLE_DEVICES="$GPU_IDS" exec python3 "$REPO/verl/action_head/serve/launch_vllm_hidden.py" serve "$MODEL" \
    --allowed-local-media-path / \
    --served-model-name Qwen3.6-35B \
    --port "$PORT" \
    --tensor-parallel-size 2 \
    --data-parallel-size 1 \
    --worker-cls action_head.serve.qwen_hidden_worker.ActionHeadHiddenWorker \
    --enforce-eager \
    --no-async-scheduling \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --limit-mm-per-prompt "{\"image\":$MAX_IMAGES}" \
    --no-enable-chunked-prefill \
    --no-disable-hybrid-kv-cache-manager \
    --speculative-config "$SPECULATIVE_CONFIG" \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    --enable-log-requests
