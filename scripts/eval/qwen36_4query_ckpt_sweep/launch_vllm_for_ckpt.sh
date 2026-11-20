#!/usr/bin/env bash
# Launch the four-query Qwen3.6 service through its dedicated Action Head entry.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../verl/scripts/eval/qwen36_4query_ckpt_sweep" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
: "${CHECKPOINT_DIR:?CHECKPOINT_DIR is required}"
: "${QUERY_CACHE:?QUERY_CACHE is required}"
: "${VLLM_LOG:?VLLM_LOG is required}"
: "${VLLM_PID_FILE:?VLLM_PID_FILE is required}"

QUERY_CKPT=$CHECKPOINT_DIR/dualvln_query/dualvln_query_embeddings.pt
VLLM_PORT_VALUE=${VLLM_PORT_OVERRIDE:-28899}
VLLM_GPUS_VALUE=${VLLM_GPUS_OVERRIDE:-14,15}
COMPILE_CACHE_VALUE=${ACTION_HEAD_COMPILE_CACHE_ROOT_OVERRIDE:-/tmp/action_head_vllm_cache}
MAX_MODEL_LEN_VALUE=${VLLM_MAX_MODEL_LEN_OVERRIDE:-131072}
GPU_MEMORY_UTILIZATION_VALUE=${VLLM_GPU_MEMORY_UTILIZATION_OVERRIDE:-0.9}
MAX_NUM_SEQS_VALUE=${VLLM_MAX_NUM_SEQS_OVERRIDE:-0}
MAX_NUM_BATCHED_TOKENS_VALUE=${VLLM_MAX_NUM_BATCHED_TOKENS_OVERRIDE:-0}
test -f "$QUERY_CKPT"
if [[ "${COMPANION_USE_CHECKPOINT_QWEN:-0}" == 1 ]]; then
  test -s "$CHECKPOINT_DIR/qwen/model.safetensors.index.json"
  VLLM_MODEL_VALUE=$CHECKPOINT_DIR/qwen
  VLLM_DTYPE_VALUE=bfloat16
else
  VLLM_MODEL_VALUE=${VLLM_MODEL:-/opt/external/user-e7a01947/Model/Qwen3.6-35B-A3B}
  VLLM_DTYPE_VALUE=${VLLM_DTYPE:-auto}
fi
mkdir -p "$QUERY_CACHE" "$(dirname "$VLLM_LOG")" "$(dirname "$VLLM_PID_FILE")"

if [[ -s "$VLLM_PID_FILE" ]]; then
  old_pid=$(cat "$VLLM_PID_FILE")
  if [[ "$old_pid" =~ ^[0-9]+$ ]]; then
    kill -TERM -- "-$old_pid" 2>/dev/null || true
    for _ in $(seq 1 60); do
      live=$(ps -eo pgid=,stat= | awk -v pgid="$old_pid" '$1 == pgid && $2 !~ /^Z/ {n++} END {print n+0}')
      [[ "$live" == "0" ]] && break
      sleep 1
    done
    if [[ "$live" != "0" ]]; then
      kill -KILL -- "-$old_pid" 2>/dev/null || true
      sleep 2
    fi
  fi
  rm -f "$VLLM_PID_FILE"
fi

setsid env \
  VLLM_GPUS="$VLLM_GPUS_VALUE" \
  VLLM_PORT="$VLLM_PORT_VALUE" \
  VLLM_MAX_MODEL_LEN="$MAX_MODEL_LEN_VALUE" \
  VLLM_GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION_VALUE" \
  VLLM_MAX_NUM_SEQS="$MAX_NUM_SEQS_VALUE" \
  VLLM_MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS_VALUE" \
  VLLM_MTP_TOKENS=0 \
  VLLM_MODEL="$VLLM_MODEL_VALUE" \
  VLLM_DTYPE="$VLLM_DTYPE_VALUE" \
  ACTION_HEAD_ENABLE_QUERY_TAIL=1 \
  ACTION_HEAD_QUERY_CHECKPOINT="$QUERY_CKPT" \
  ACTION_HEAD_QUERY_CACHE="$QUERY_CACHE" \
  ACTION_HEAD_COMPILE_CACHE_ROOT="$COMPILE_CACHE_VALUE" \
  bash "$REPO/scripts/serve/vllm/vllm-action-head-query.sh" \
  > "$VLLM_LOG" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$VLLM_PID_FILE"

deadline=$((SECONDS + 3600))
until curl -fsS -m 5 "http://127.0.0.1:$VLLM_PORT_VALUE/health" >/dev/null \
  && curl -fsS -m 5 "http://127.0.0.1:$VLLM_PORT_VALUE/v1/models" >/dev/null; do
  kill -0 "$pid" 2>/dev/null || { tail -n 200 "$VLLM_LOG"; exit 3; }
  if grep -Eq 'Engine core initialization failed|Stale file handle|UnboundLocalError:.*tracer_output' "$VLLM_LOG"; then
    echo "vLLM startup hit a fatal compiler/runtime error" >&2
    kill -TERM -- "-$pid" 2>/dev/null || true
    tail -n 200 "$VLLM_LOG"
    exit 5
  fi
  (( SECONDS < deadline )) || { echo "vLLM startup timeout"; tail -n 200 "$VLLM_LOG"; exit 4; }
  sleep 5
done

if grep -Eq 'Engine core initialization failed|Stale file handle|UnboundLocalError:.*tracer_output|pickle data was truncated' "$VLLM_LOG"; then
  echo "vLLM startup log contains a fatal compiler/runtime error" >&2
  kill -TERM -- "-$pid" 2>/dev/null || true
  tail -n 200 "$VLLM_LOG"
  exit 5
fi

grep -Eq 'enforce_eager.?=.?False|enforce_eager=False' "$VLLM_LOG"
grep -Eq 'VLLM_COMPILE|CompilationMode' "$VLLM_LOG"
grep -Eq 'FULL_AND_PIECEWISE|CUDAGraphMode' "$VLLM_LOG"
grep -Eq 'Graph capturing finished|graph capturing finished' "$VLLM_LOG"
echo "[vllm] ready pid=$pid port=$VLLM_PORT_VALUE gpus=$VLLM_GPUS_VALUE checkpoint=$CHECKPOINT_DIR max_num_seqs=$MAX_NUM_SEQS_VALUE max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS_VALUE gpu_memory_utilization=$GPU_MEMORY_UTILIZATION_VALUE"
