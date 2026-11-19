#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/workspace/ZeroVLN}
VERL_REPO=${VERL_REPO:-/workspace/verl}
PYTHON=${PYTHON:-/usr/local/bin/python3.12}
EXPR_NAME=${EXPR_NAME:?EXPR_NAME is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
HC_DIR=${HC_DIR:-/data/zerovln/vln_verl/dlc}

write_failure_done() {
  status=$?
  if [[ "$status" -ne 0 && "${VALIDATE_ONLY:-0}" != 1 ]]; then
    DONE_FILE="$HC_DIR/trainer_done_${EXPR_NAME}"
    [[ -e "$DONE_FILE" ]] || printf '{"status":"failed","exit_code":%d}\n' "$status" > "$DONE_FILE"
  fi
  case "${ACTION_HEAD_LOCAL_RUNTIME_ROOT:-}" in
    /tmp/action_head_rl/*) rm -rf -- "$ACTION_HEAD_LOCAL_RUNTIME_ROOT" ;;
    "") ;;
    *) echo "refusing to clean unexpected runtime root: $ACTION_HEAD_LOCAL_RUNTIME_ROOT" >&2 ;;
  esac
}
trap write_failure_done EXIT

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
# Caches and temporary files live on the container-local disk. Persistent
# checkpoints, metrics and logs remain under OUTPUT_DIR on the data volume.
export ACTION_HEAD_LOCAL_RUNTIME_ROOT=${ACTION_HEAD_LOCAL_RUNTIME_ROOT:-/tmp/action_head_rl/${EXPR_NAME}/${HOSTNAME:-driver}}
export ACTION_HEAD_RAY_CACHE_ROOT=${ACTION_HEAD_RAY_CACHE_ROOT:-$ACTION_HEAD_LOCAL_RUNTIME_ROOT/cache/ray}
export ACTION_HEAD_PROCESS_COMPILE_CACHE_ROOT=${ACTION_HEAD_PROCESS_COMPILE_CACHE_ROOT:-$ACTION_HEAD_LOCAL_RUNTIME_ROOT/cache/process_compile}
export ACTION_HEAD_COMPILE_CACHE_ROOT=${ACTION_HEAD_COMPILE_CACHE_ROOT:-$ACTION_HEAD_LOCAL_RUNTIME_ROOT/cache/vllm}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$ACTION_HEAD_RAY_CACHE_ROOT/xdg}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ACTION_HEAD_RAY_CACHE_ROOT/torchinductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$ACTION_HEAD_RAY_CACHE_ROOT/triton}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$ACTION_HEAD_RAY_CACHE_ROOT/cuda}
export PYTORCH_KERNEL_CACHE_PATH=${PYTORCH_KERNEL_CACHE_PATH:-$ACTION_HEAD_RAY_CACHE_ROOT/torch/kernels}
export HF_HOME=${HF_HOME:-$ACTION_HEAD_RAY_CACHE_ROOT/huggingface}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$ACTION_HEAD_RAY_CACHE_ROOT/torch_extensions}
export TMPDIR=${ACTION_HEAD_TMPDIR:-$ACTION_HEAD_LOCAL_RUNTIME_ROOT/tmp}
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
  "$CUDA_CACHE_PATH" "$PYTORCH_KERNEL_CACHE_PATH" "$HF_HOME" "$TORCH_EXTENSIONS_DIR"
if [[ -n "${FAST_KERNEL_OVERLAY:-}" ]]; then
  [[ "${TRAIN_SCOPE:-}" == full ]] || {
    echo "FAST_KERNEL_OVERLAY is Full-only" >&2
    exit 2
  }
  [[ -f "$FAST_KERNEL_OVERLAY/causal_conv1d/__init__.py" ]] || {
    echo "invalid FAST_KERNEL_OVERLAY=$FAST_KERNEL_OVERLAY" >&2
    exit 2
  }
  export PYTHONPATH="$FAST_KERNEL_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
  export ACTION_HEAD_REQUIRE_QWEN_FAST_PATH=1
fi
export PYTHONPATH="$VERL_REPO:$REPO:$REPO/verl:$REPO/vln_demo/src:${PYTHONPATH:-}"
export ACTION_HEAD_SLOW_MODEL="${SLOW_MODEL:?SLOW_MODEL is required}"
export ACTION_HEAD_HEAD_CHECKPOINT="${HEAD_CHECKPOINT:?HEAD_CHECKPOINT is required}"
export ACTION_HEAD_QUERY_CHECKPOINT="${QUERY_CHECKPOINT:?QUERY_CHECKPOINT is required}"
export ACTION_HEAD_TRAIN_SCOPE="${TRAIN_SCOPE:-head_query}"
export ACTION_HEAD_ROUTER_REPLAY=R3
export ACTION_HEAD_CAPTURE_ROUTING=1
export ACTION_HEAD_CAPTURE_FULL_ROUTING=1

BENCH_URLS=${BENCH_URLS:-}
if [[ "${VALIDATE_ONLY:-0}" == 1 && -z "$BENCH_URLS" ]]; then
  BENCH_URLS=http://127.0.0.1:1
fi
if [[ -z "$BENCH_URLS" ]]; then
  BENCH_EXPR=${BENCH_EXPR:-$EXPR_NAME}
  READY_FILE="$HC_DIR/bench_ready_${BENCH_EXPR}.txt"
  echo "[ray-entry] waiting for $READY_FILE"
  for _ in $(seq 1 720); do
    [[ -s "$READY_FILE" ]] && break
    sleep 5
  done
  [[ -s "$READY_FILE" ]] || { echo "bench discovery timed out: $READY_FILE" >&2; exit 2; }
  BENCH_URLS=$(tr '\n' ',' < "$READY_FILE" | sed 's/,$//')
fi

ARGS=(
  --repo "$REPO"
  --slow-model "$SLOW_MODEL"
  --fast-backbone "$FAST_BACKBONE"
  --head-checkpoint "$HEAD_CHECKPOINT"
  --query-checkpoint "$QUERY_CHECKPOINT"
  --output-dir "$OUTPUT_DIR"
  --bench-urls "$BENCH_URLS"
  --nnodes "${NNODES:-1}"
  --learner-gpus-per-node "${LEARNER_GPUS_PER_NODE:-13}"
  --fsdp-size "${FSDP_SIZE:--1}"
  --rollout-actors "${ROLLOUT_ACTORS:-1}"
  --rollout-gpu-fraction "${ROLLOUT_GPU_FRACTION:-1.0}"
  --fast-policy-workers "${FAST_POLICY_WORKERS:-${NNODES:-1}}"
  --fast-policy-max-batch-size "${FAST_POLICY_MAX_BATCH_SIZE:-8}"
  --fast-policy-batch-wait-ms "${FAST_POLICY_BATCH_WAIT_MS:-12}"
  --rollout-groups-in-flight "${ROLLOUT_GROUPS_IN_FLIGHT:-2}"
  --rollout-scheduler "${ROLLOUT_SCHEDULER:-wave}"
  --vllm-max-num-seqs "${VLLM_MAX_NUM_SEQS:-0}"
  --vllm-max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS:-0}"
  --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
  --vllm-tp-size "${VLLM_TP_SIZE:-2}"
  --vllm-replicas "${VLLM_REPLICAS:-1}"
  --colocated-vllm-replicas "${COLOCATED_VLLM_REPLICAS:-0}"
  --benchmark-variants "${BENCHMARK_VARIANTS:-baseline16,bulk16,pipeline64}"
  --train-scope "$ACTION_HEAD_TRAIN_SCOPE"
  --router-replay R3
  --opt-steps "${OPT_STEPS:-1}"
  --group-size "${GROUP_SIZE:-2}"
  --num-groups-per-step "${NUM_GROUPS_PER_STEP:-2}"
  --smoke-cases "${SMOKE_CASES:-2}"
  --stop-after-step "${STOP_AFTER_STEP:-0}"
  --keep-last-checkpoints "${KEEP_LAST_CHECKPOINTS:-0}"
  --checkpoint-interval "${CHECKPOINT_INTERVAL:-1}"
  --dataset-split "${DATASET_SPLIT:-action_head_train_p0_unified}"
  --harness-version "${HARNESS_VERSION:-v1}"
  --learning-rate "${LEARNING_RATE:-1e-6}"
  --weight-decay "${WEIGHT_DECAY:-0.01}"
  --vlm-learning-rate "${VLM_LEARNING_RATE:-0}"
  --old-logprob-mode "${OLD_LOGPROB_MODE:-recompute}"
  --full-weight-residency-gib "${FULL_WEIGHT_RESIDENCY_GIB:-0}"
  --r3-capture-mode "${ACTION_HEAD_R3_CAPTURE_MODE:-legacy}"
  --r3-audit-dir "${ACTION_HEAD_R3_AUDIT_DIR:-}"
  --stage-records-per-rpc "${STAGE_RECORDS_PER_RPC:-1}"
  --stream-groups-per-update "${STREAM_GROUPS_PER_UPDATE:-1}"
  --multi-condition-batch-size "${MULTI_CONDITION_BATCH_SIZE:-1}"
  --max-condition-tokens "${MAX_CONDITION_TOKENS:-0}"
  --condition-token-bucket-size "${CONDITION_TOKEN_BUCKET_SIZE:-0}"
  --weight-sync-bucket-mb "${WEIGHT_SYNC_BUCKET_MB:-2048}"
  --sde-steps "${SDE_STEPS:-3}"
  --sde-noise "${SDE_NOISE:-0.08}"
  --terminal-success-reward "${TERMINAL_SUCCESS_REWARD:-1.0}"
  --direction-match-reward "${DIRECTION_MATCH_REWARD:-0.05}"
  --collision-penalty "${COLLISION_PENALTY:-0.05}"
  --discount-gamma "${DISCOUNT_GAMMA:-1.0}"
  --max-tool-calls "${MAX_TOOL_CALLS:-2}"
  --vllm-port "${VLLM_PORT:-28899}"
  --ray-address "${RAY_ADDRESS_MODE:-auto}"
  --resume-mode "${RESUME_MODE:-disable}"
)
case "${ACTION_HEAD_EPISODE_GRPO:-0}" in
  1) ARGS+=(--episode-grpo) ;;
  0) ;;
  *) echo "ACTION_HEAD_EPISODE_GRPO must be 0 or 1" >&2; exit 2 ;;
esac
if [[ "$ACTION_HEAD_TRAIN_SCOPE" != full && "${ACTION_HEAD_EPISODE_GRPO:-0}" != 1 ]]; then
  ARGS+=(
    --max-turns "${MAX_TURNS:-2}"
    --fm-chunks-per-episode "${FM_CHUNKS_PER_EPISODE:-8}"
  )
fi
if [[ -n "${CASE_SHUFFLE_SEED:-}" ]]; then
  ARGS+=(--case-shuffle-seed "$CASE_SHUFFLE_SEED")
fi
if [[ "${GRADIENT_CHECKPOINTING:-1}" == 1 ]]; then
  ARGS+=(--gradient-checkpointing)
else
  ARGS+=(--no-gradient-checkpointing)
fi
if [[ "${PIOLD_OPTION_REUSE:-0}" == 1 ]]; then
  ARGS+=(--piold-option-reuse)
fi
if [[ "${FROZEN_WEIGHT_RESIDENCY:-0}" == 1 ]]; then
  ARGS+=(--frozen-weight-residency)
fi
if [[ "${ACTION_HEAD_DEFER_QUERY_CAPTURE:-0}" == 1 ]]; then
  ARGS+=(--defer-query-capture)
fi
[[ "${ACTION_HEAD_DEFER_GRADIENT_SYNC:-0}" == 1 ]] && ARGS+=(--defer-gradient-sync)
if [[ "${OPTION_CONDITION_REUSE:-0}" == 1 ]]; then
  ARGS+=(--option-condition-reuse)
else
  ARGS+=(--no-option-condition-reuse)
fi
if [[ -n "${RESUME_FROM_PATH:-}" ]]; then
  ARGS+=(--resume-from-path "$RESUME_FROM_PATH")
fi
[[ "${RESUME_AUDIT:-0}" == 1 ]] && ARGS+=(--resume-audit)
if [[ "${ALLOW_WEIGHTS_ONLY_RESUME:-0}" == 1 ]]; then
  ARGS+=(--allow-weights-only-resume)
fi
if [[ -n "${LEARNER_GPUS_LAYOUT:-}" ]]; then
  ARGS+=(--learner-gpus-layout "$LEARNER_GPUS_LAYOUT")
fi
if [[ "${FSDP_FORWARD_PREFETCH:-0}" == 1 ]]; then
  ARGS+=(--fsdp-forward-prefetch)
else
  ARGS+=(--no-fsdp-forward-prefetch)
fi
if [[ "${VALIDATE_ONLY:-0}" == 1 ]]; then
  ARGS+=(--validate-only)
fi

[[ "${FAST_POLICY_PIPELINE:-0}" == 1 ]] && ARGS+=(--fast-policy-pipeline)
[[ "${BULK_ROUTE_CAPTURE:-0}" == 1 ]] && ARGS+=(--bulk-route-capture)
[[ "${TOKEN_BALANCED_SHARDING:-0}" == 1 ]] && ARGS+=(--token-balanced-sharding)
[[ "${ZERO_COPY_ROLLOUT_PAYLOAD:-0}" == 1 ]] && ARGS+=(--zero-copy-rollout-payload)
[[ "${DIRECT_VLLM_WEIGHT_SYNC:-0}" == 1 ]] && ARGS+=(--direct-vllm-weight-sync)
[[ "${OVERLAP_ROLLOUT_LEARNER:-0}" == 1 ]] && ARGS+=(--overlap-rollout-learner)
[[ "${ROLLOUT_BENCHMARK:-0}" == 1 ]] && ARGS+=(--rollout-benchmark)
cd "$REPO"
"$PYTHON" -u "$REPO/verl/action_head_rl/trainer.py" "${ARGS[@]}"
