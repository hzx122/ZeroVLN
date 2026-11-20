#!/usr/bin/env bash
# Qwen3.6-35B 四-query Action Head 服务。
# 普通 VLN 推理请使用 vllm-vln-35B.sh，避免自定义 worker 泄漏到通用服务。
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../vln_demo/vllm" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
: "${ACTION_HEAD_QUERY_CHECKPOINT:?ACTION_HEAD_QUERY_CHECKPOINT is required}"

VLLM_MODEL=${VLLM_MODEL:-/opt/external/user-e7a01947/Model/Qwen3.6-35B-A3B}
VLLM_GPUS=${VLLM_GPUS:-14,15}
VLLM_PORT=${VLLM_PORT:-28899}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-131072}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.9}

# 这些 token ID 已在 Qwen3.6-35B tokenizer 上核验。query token 的 embedding
# 会在进入模型前替换，不作为普通文本 token 使用。
export ACTION_HEAD_QUERY_TOKEN_ID=${ACTION_HEAD_QUERY_TOKEN_ID:-248060}
export ACTION_HEAD_QUERY_EOS_TOKEN_ID=${ACTION_HEAD_QUERY_EOS_TOKEN_ID:-248046}
export ACTION_HEAD_TOOL_END_TOKEN_IDS=${ACTION_HEAD_TOOL_END_TOKEN_IDS:-248059}
export ACTION_HEAD_RUN_TOOL_TOKEN_IDS=${ACTION_HEAD_RUN_TOOL_TOKEN_IDS:-5917,7681,12756}
export ACTION_HEAD_QUERY_CACHE=${ACTION_HEAD_QUERY_CACHE:-/dev/shm/action_head_queries}

# CPFS 上的 torch.compile 原子写偶发 ESTALE，编译缓存必须放节点本地盘。
ACTION_HEAD_COMPILE_CACHE_ROOT=${ACTION_HEAD_COMPILE_CACHE_ROOT:-/tmp/action_head_vllm_cache}
mkdir -p "$ACTION_HEAD_COMPILE_CACHE_ROOT/torchinductor" \
    "$ACTION_HEAD_COMPILE_CACHE_ROOT/triton" \
    "$ACTION_HEAD_COMPILE_CACHE_ROOT/xdg"
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ACTION_HEAD_COMPILE_CACHE_ROOT/torchinductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$ACTION_HEAD_COMPILE_CACHE_ROOT/triton}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$ACTION_HEAD_COMPILE_CACHE_ROOT/xdg}
# action_head is under verl/; the repository root contains the logging filter
# imported by vllm_log_hidden_id.json.
export PYTHONPATH="$REPO/verl:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_LOGGING_CONFIG_PATH="${ACTION_HEAD_LOGGING_CONFIG_OVERRIDE:-$HERE/vllm_log_hidden_id.json}"

QUERY_ARGS=(
    --worker-cls action_head.serve.qwen_query_worker.ActionHeadQueryWorker
    --logits-processors action_head.serve.dualvln_query_runtime:DualVlnQueryLogitsProcessor
)
if [[ "${ACTION_HEAD_CAPTURE_ROUTING:-0}" == "1" ]]; then
    QUERY_ARGS+=(--enable-return-routed-experts)
fi

# Full token/request dumps grow with every turn and are not evaluation audit.
# Canonical conversations and native audit remain in per-case artifacts.
LOG_ARGS=()
if [[ "${VLLM_ENABLE_LOG_REQUESTS:-0}" == "1" ]]; then
    LOG_ARGS+=(--enable-log-requests --max-log-len 256)
fi
PREFIX_CACHE_ARGS=()
if [[ "${VLLM_ENABLE_PREFIX_CACHING:-1}" == "1" ]]; then
    PREFIX_CACHE_ARGS+=(--enable-prefix-caching)
fi
SCHEDULER_ARGS=()
if [[ "${ACTION_HEAD_R3_CAPTURE_MODE:-legacy}" == "native" ]]; then
    SCHEDULER_ARGS+=(--no-async-scheduling)
    if [[ "${ACTION_HEAD_V8_PREFIX_CACHE:-0}" == "1" ]]; then
        SCHEDULER_ARGS+=(--mamba-cache-mode align)
    fi
fi
if [[ -n "${VLLM_SERVER_SEED:-}" ]]; then
    SCHEDULER_ARGS+=(--seed "$VLLM_SERVER_SEED")
fi
if (( ${VLLM_MAX_NUM_SEQS:-0} > 0 )); then
    SCHEDULER_ARGS+=(--max-num-seqs "$VLLM_MAX_NUM_SEQS")
fi
if (( ${VLLM_MAX_NUM_BATCHED_TOKENS:-0} > 0 )); then
    SCHEDULER_ARGS+=(--max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS")
fi

# vLLM 0.18 不允许 custom logits processor 与 speculative decoding 并用，
# 因此这个专用入口固定不启用 MTP。
exec env CUDA_VISIBLE_DEVICES="$VLLM_GPUS" \
    vllm serve "$VLLM_MODEL" \
    --allowed-local-media-path / \
    --served-model-name Qwen3.6-35B \
    --port "$VLLM_PORT" \
    --dtype "${VLLM_DTYPE:-auto}" \
    --tensor-parallel-size 2 \
    --data-parallel-size 1 \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --limit-mm-per-prompt "{\"image\":${VLLM_MAX_IMAGES:-64}}" \
    "${QUERY_ARGS[@]}" \
    "${PREFIX_CACHE_ARGS[@]}" \
    "${SCHEDULER_ARGS[@]}" \
    "${LOG_ARGS[@]}"
