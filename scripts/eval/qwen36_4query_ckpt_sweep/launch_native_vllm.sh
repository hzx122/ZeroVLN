#!/usr/bin/env bash
# Launch the audited Qwen3.6 service without the Action Head query tail.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../verl/scripts/eval/qwen36_4query_ckpt_sweep" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
: "${VLLM_LOG:?VLLM_LOG is required}"
: "${VLLM_PID_FILE:?VLLM_PID_FILE is required}"
VLLM_PORT_VALUE=${VLLM_PORT_OVERRIDE:-28899}
VLLM_GPUS_VALUE=${VLLM_GPUS_OVERRIDE:-14,15}
mkdir -p "$(dirname "$VLLM_LOG")" "$(dirname "$VLLM_PID_FILE")"

if [[ -s "$VLLM_PID_FILE" ]]; then
  old_pid=$(cat "$VLLM_PID_FILE")
  if [[ "$old_pid" =~ ^[0-9]+$ ]]; then
    kill -TERM -- "-$old_pid" 2>/dev/null || true
    for _ in $(seq 1 60); do
      live=$(ps -eo pgid=,stat= | awk -v pgid="$old_pid" '$1 == pgid && $2 !~ /^Z/ {n++} END {print n+0}')
      [[ "$live" == "0" ]] && break
      sleep 1
    done
    [[ "$live" == "0" ]] || kill -KILL -- "-$old_pid" 2>/dev/null || true
  fi
  rm -f "$VLLM_PID_FILE"
fi

setsid env \
  VLLM_GPUS="$VLLM_GPUS_VALUE" \
  VLLM_PORT="$VLLM_PORT_VALUE" \
  VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN_OVERRIDE:-131072}" \
  VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION_OVERRIDE:-0.9}" \
  VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS_OVERRIDE:-0}" \
  VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS_OVERRIDE:-0}" \
  ACTION_HEAD_ENABLE_QUERY_TAIL=0 \
  VLLM_MTP_TOKENS=0 \
  bash "$REPO/scripts/serve/vllm/vllm-vln-35B.sh" \
  > "$VLLM_LOG" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$VLLM_PID_FILE"

deadline=$((SECONDS + 3600))
until curl -fsS -m 5 "http://127.0.0.1:$VLLM_PORT_VALUE/health" >/dev/null \
  && curl -fsS -m 5 "http://127.0.0.1:$VLLM_PORT_VALUE/v1/models" >/dev/null; do
  kill -0 "$pid" 2>/dev/null || { tail -n 200 "$VLLM_LOG"; exit 3; }
  (( SECONDS < deadline )) || { echo "native vLLM startup timeout"; tail -n 200 "$VLLM_LOG"; exit 4; }
  sleep 5
done

grep -Eq 'enforce_eager.?=.?False|enforce_eager=False' "$VLLM_LOG"
grep -Eq 'VLLM_COMPILE|CompilationMode' "$VLLM_LOG"
grep -Eq 'FULL_AND_PIECEWISE|CUDAGraphMode' "$VLLM_LOG"
grep -Eq 'Graph capturing finished|graph capturing finished' "$VLLM_LOG"
if grep -q 'ActionHeadQueryWorker' "$VLLM_LOG"; then
  echo "native baseline unexpectedly enabled Action Head query worker" >&2
  exit 5
fi
echo "[native-vllm] ready pid=$pid port=$VLLM_PORT_VALUE gpus=$VLLM_GPUS_VALUE"
