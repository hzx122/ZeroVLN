#!/usr/bin/env bash
# 普通 Qwen3.6-35B VLN 服务。
# 带 Action Head（四-query/query-tail）的改动见 vllm-action-head-query.sh。
set -euo pipefail

CHECK_ARGS=()
case "${1:-start}" in
    start) ;;
    --check) CHECK_ARGS+=(--help) ;;
    *) echo "Usage: $0 [start|--check]" >&2; exit 2 ;;
esac

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../vln_demo/vllm" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
# vllm_log_hidden_id.json imports vllm_log_filters.py from the repository's
# root directory. This path belongs to logging, not to Action Head runtime.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_LOGGING_CONFIG_PATH="$HERE/vllm_log_hidden_id.json"

# 本地默认卡位保持为 2,3；其他拓扑可通过 VLLM_GPUS 显式覆盖。
VLLM_MODEL=${VLLM_MODEL:-/opt/external/user-e7a01947/Model/Qwen3.6-35B-A3B}
VLLM_GPUS=${VLLM_GPUS:-2,3}
VLLM_TP_SIZE=${VLLM_TP_SIZE:-2}
VLLM_PORT=${VLLM_PORT:-28900}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-131072}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.9}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-0}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-0}
VLLM_MTP_TOKENS=${VLLM_MTP_TOKENS:-3}
ACTION_HEAD_ENABLE_QUERY_TAIL=${ACTION_HEAD_ENABLE_QUERY_TAIL:-0}
QUERY_ARGS=()
if [[ "$ACTION_HEAD_ENABLE_QUERY_TAIL" == "1" ]]; then
    : "${ACTION_HEAD_QUERY_CHECKPOINT:?ACTION_HEAD_QUERY_CHECKPOINT is required}"
    # Verified against the live Qwen3.6-35B tokenizer on 2026-08-24.
    # 248060 is the existing non-EOS <|fim_prefix|> reserved special token;
    # its embedding is replaced before the model sees it.
    export ACTION_HEAD_QUERY_TOKEN_ID=${ACTION_HEAD_QUERY_TOKEN_ID:-248060}
    export ACTION_HEAD_QUERY_EOS_TOKEN_ID=${ACTION_HEAD_QUERY_EOS_TOKEN_ID:-248046}
    export ACTION_HEAD_TOOL_END_TOKEN_IDS=${ACTION_HEAD_TOOL_END_TOKEN_IDS:-248059}
    export ACTION_HEAD_RUN_TOOL_TOKEN_IDS=${ACTION_HEAD_RUN_TOOL_TOKEN_IDS:-5917,7681,12756}
    export ACTION_HEAD_QUERY_CACHE=${ACTION_HEAD_QUERY_CACHE:-/dev/shm/action_head_queries}
    # Compilation on CPFS can fail with ESTALE. Keep Torch/Triton/vLLM caches
    # on node-local storage; this also makes restarts reuse the compiled graph.
    ACTION_HEAD_COMPILE_CACHE_ROOT=${ACTION_HEAD_COMPILE_CACHE_ROOT:-/tmp/action_head_vllm_cache}
    mkdir -p "$ACTION_HEAD_COMPILE_CACHE_ROOT/torchinductor" \
        "$ACTION_HEAD_COMPILE_CACHE_ROOT/triton" \
        "$ACTION_HEAD_COMPILE_CACHE_ROOT/xdg"
    export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ACTION_HEAD_COMPILE_CACHE_ROOT/torchinductor}
    export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$ACTION_HEAD_COMPILE_CACHE_ROOT/triton}
    export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$ACTION_HEAD_COMPILE_CACHE_ROOT/xdg}
    # Load the process-local vLLM 0.18 compatibility patches in the API,
    # EngineCore, and TP worker subprocesses. Qwen3.6's 30 linear-attention
    # plus 10 full-attention layers form four hybrid KV groups.
    export PYTHONPATH="$REPO/verl/action_head/serve/bootstrap:$REPO/verl${PYTHONPATH:+:$PYTHONPATH}"
    QUERY_ARGS+=(
        --worker-cls action_head.serve.qwen_query_worker.ActionHeadQueryWorker
        --logits-processors action_head.serve.dualvln_query_runtime:DualVlnQueryLogitsProcessor
        # Keep the full-route training service synchronous until the reordered
        # input-batch row mapping has also passed a real multi-group GPU gate.
        --no-async-scheduling
    )
    if [[ "${ACTION_HEAD_CAPTURE_ROUTING:-0}" == "1" ]]; then
        # Diagnostic-only: vLLM records logical top-k expert IDs for every
        # token/layer; the query worker persists only the four query rows.
        QUERY_ARGS+=(--enable-return-routed-experts)
    fi
fi
VLLM_PYTHON=${VLLM_PYTHON:-/usr/local/bin/python3}

if ! [[ "$VLLM_MTP_TOKENS" =~ ^[0-9]+$ ]]; then
    echo "VLLM_MTP_TOKENS must be a non-negative integer" >&2
    exit 2
fi

# vLLM 0.18 rejects speculative decoding whenever any custom logits processor
# is registered. Keep MTP for ordinary VLN generation, but disable it for the
# Action Head query-tail service until capture no longer uses logits processors.
if [[ "$ACTION_HEAD_ENABLE_QUERY_TAIL" == "1" ]] && (( VLLM_MTP_TOKENS > 0 )); then
    echo "Action Head query tail uses a custom logits processor; disabling MTP for vLLM 0.18" >&2
    VLLM_MTP_TOKENS=0
fi

MTP_ARGS=()
if (( VLLM_MTP_TOKENS > 0 )); then
    MTP_ARGS+=(
        --speculative-config
        "{\"method\":\"mtp\",\"num_speculative_tokens\":$VLLM_MTP_TOKENS}"
    )
fi
PREFIX_CACHE_ARGS=()
if [[ "${VLLM_ENABLE_PREFIX_CACHING:-1}" == "1" ]]; then
    PREFIX_CACHE_ARGS+=(--enable-prefix-caching)
    if [[ "${ACTION_HEAD_V8_PREFIX_CACHE:-0}" == "1" ]]; then
        PREFIX_CACHE_ARGS+=(--mamba-cache-mode align)
    fi
elif [[ "${ACTION_HEAD_CONTINUOUS_CONTEXT:-0}" == "1" ]]; then
    PREFIX_CACHE_ARGS+=(--no-enable-prefix-caching)
fi
SCHEDULER_ARGS=()
if [[ "${RULE_TEACHER_SAMPLING_PROFILE:-}" == "v6_depth" ]]; then
    SCHEDULER_ARGS+=(--no-async-scheduling)
fi
if [[ "${VLLM_ENFORCE_EAGER:-0}" == "1" ]]; then
    SCHEDULER_ARGS+=(--enforce-eager)
fi
if (( VLLM_MAX_NUM_SEQS > 0 )); then
    SCHEDULER_ARGS+=(--max-num-seqs "$VLLM_MAX_NUM_SEQS")
fi
if (( VLLM_MAX_NUM_BATCHED_TOKENS > 0 )); then
    SCHEDULER_ARGS+=(--max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS")
fi
if [[ "${ACTION_HEAD_VLLM_ENABLE_SLEEP_MODE:-0}" == 1 ]]; then
    SCHEDULER_ARGS+=(--enable-sleep-mode)
    # Independent sleep diagnostics may observe subprocess exits. Unset in
    # production; keep the normal compatibility bootstrap as the default.
    if [[ -n "${ACTION_HEAD_VLLM_DIAGNOSTIC_BOOTSTRAP:-}" ]]; then
        export PYTHONPATH="$ACTION_HEAD_VLLM_DIAGNOSTIC_BOOTSTRAP:$PYTHONPATH"
    fi
fi
if [[ "${ACTION_HEAD_NATIVE_COLOCATED_WEIGHTS:-0}" == 1 ]]; then
    SCHEDULER_ARGS+=(--worker-extension-cls "action_head_rl.native_weight_receiver.NativeColocatedWeightExtension")
elif [[ "${ACTION_HEAD_VLLM_DIRECT_WEIGHT_SYNC:-0}" == 1 ]]; then
    SCHEDULER_ARGS+=(--worker-extension-cls "action_head_rl.direct_weight_sync.DirectWeightSyncExtension")
elif [[ "${ACTION_HEAD_VLLM_AUDIT_WEIGHTS:-0}" == 1 ]]; then
    SCHEDULER_ARGS+=(--worker-extension-cls "${ACTION_HEAD_VLLM_AUDIT_EXTENSION:-action_head_rl.weight_sync_audit.WeightSyncAuditExtension}")
fi

# The image's generic vllm CLI imports benchmark-only datasets/pyarrow modules.
# Use its supported server entrypoint directly; --check exercises this same
# interpreter, logging configuration and argument parser without loading a model.
exec env CUDA_VISIBLE_DEVICES="$VLLM_GPUS" \
    "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server --model "$VLLM_MODEL" \
    --allowed-local-media-path / \
    --served-model-name Qwen3.6-35B \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$VLLM_TP_SIZE" \
    --data-parallel-size 1 \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --limit-mm-per-prompt "{\"image\":${VLLM_MAX_IMAGES:-64}}" \
    "${SCHEDULER_ARGS[@]}" \
    "${QUERY_ARGS[@]}" \
    "${PREFIX_CACHE_ARGS[@]}" \
    "${MTP_ARGS[@]}" \
    --enable-log-requests \
    "${CHECK_ARGS[@]}"
