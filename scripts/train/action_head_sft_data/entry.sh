#!/usr/bin/env bash
# Build one scene shard of the Action Head P0 nested-SFT dataset. Each DLC job
# owns one shard and starts its own TP2 slow VLM, Sage renderer and fast visual
# encoder. Shard 0 merges only after all independent jobs publish done markers.
set -euo pipefail

ACTION_HEAD_AS_TOOL_VERSION=v1
while (($#)); do
  case "$1" in
    --action-head-as-tool-version) ACTION_HEAD_AS_TOOL_VERSION=${2:?}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$ACTION_HEAD_AS_TOOL_VERSION" == v1 || "$ACTION_HEAD_AS_TOOL_VERSION" == v2 || "$ACTION_HEAD_AS_TOOL_VERSION" == v3 || "$ACTION_HEAD_AS_TOOL_VERSION" == v4.3 || "$ACTION_HEAD_AS_TOOL_VERSION" == v6 ]] || {
  echo "--action-head-as-tool-version must be v1, v2, or v3" >&2; exit 2;
}

REPO=/workspace/ZeroVLN
GV=/opt/navigation-runtime/venv
RS=/workspace/agentic_sim/repo/ABot-Navigation/render_server
TGT=$REPO/vln_demo/render_libs
FM_VENV=$REPO/vln_demo/.venv_fm
DATA_ROOT=${DATA_ROOT:?}
SHARD_INDEX=${SHARD_INDEX:?}
SHARD_COUNT=${SHARD_COUNT:?}
RUN_NAME=${RUN_NAME:?}
LOCAL_SERVICES=${LOCAL_SERVICES:-0}
RAW_ONLY=${RAW_ONLY:-0}
SKIP_RENDER_SERVICE=${SKIP_RENDER_SERVICE:-0}
RAW_JSONL_NAME=${RAW_JSONL_NAME:-sage_sft.jsonl}
RAW_PART_INDEX=${RAW_PART_INDEX:-}
RAW_SCENE_SHARD_INDEX=${RAW_SCENE_SHARD_INDEX:-$SHARD_INDEX}
RAW_SCENE_SHARD_COUNT=${RAW_SCENE_SHARD_COUNT:-$SHARD_COUNT}
MAX_SCENES=${MAX_SCENES:-0}
MAX_TASKS=${MAX_TASKS:-0}
MAX_ANCHORS=${MAX_ANCHORS:-0}
RENDER_URL=${RENDER_URL:-http://127.0.0.1:7038}
SLOW_VLLM_URL=${SLOW_VLLM_URL:-http://127.0.0.1:28899}
VLLM_PORT=${VLLM_PORT:-28899}
RENDER_PORT=${RENDER_PORT:-7038}
VLLM_GPUS=${VLLM_GPUS:-0,1}
RENDER_GPU=${RENDER_GPU:-2}
QUERY_CHECKPOINT=${QUERY_CHECKPOINT:-/data/zerovln/action_head/integration/qwen36_four_query_5001_migration_20260824/query/dualvln_query_embeddings.pt}
FAST_BACKBONE=${FAST_BACKBONE:-/opt/external/user-e7a01947/Model/Qwen3.5-4B}
SPLIT_MANIFEST=${SPLIT_MANIFEST:-/data/zerovln/action_head/curriculum/train_v1/p0_unified/manifest.json}
SHARD_ROOT=$DATA_ROOT/shards/$(printf 'shard_%03d' "$SHARD_INDEX")
LOG_ROOT=$DATA_ROOT/logs
# Query-tail .safetensors cache on tmpfs (/dev/shm), NOT CPFS. The build reads
# one small .safetensors per anchor (VllmQueryStateProvider); CPFS small-file
# read latency dominated per-anchor processing (~2.86s/anchor, mostly IO).
# /dev/shm is per-pod, shared by the TP4 vLLM (writer) and build (reader).
# Pod shared_memory is ~240Gi so this is ample; provider cleanup=True bounds it.
ACTION_HEAD_JOB_LOCAL_ROOT=${ACTION_HEAD_JOB_LOCAL_ROOT:-/tmp/ahdata/$(hostname)}
QUERY_CACHE=${QUERY_CACHE:-$ACTION_HEAD_JOB_LOCAL_ROOT/query_shard$(printf '%03d' "$SHARD_INDEX")}
export VLLM_RPC_BASE_PATH="@ahdata-${SHARD_INDEX}-$$"
mkdir -p "$SHARD_ROOT/raw" "$SHARD_ROOT/bundles" "$QUERY_CACHE" "$LOG_ROOT" "$DATA_ROOT/status"

PIDS=()
cleanup() {
  code=$?
  trap - EXIT INT TERM
  for pid in "${PIDS[@]:-}"; do kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
  exit "$code"
}
trap cleanup EXIT INT TERM

stop_services() {
  for pid in "${PIDS[@]:-}"; do kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
  PIDS=()
}

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/opt/accl-p:/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/lib:$GV/lib:/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO/verl:$REPO/verl/workers:$REPO/vln_demo/src${PYTHONPATH:+:$PYTHONPATH}"
export SAGE_HFOV=115
if [[ "$LOCAL_SERVICES" != 1 ]]; then
  # torch.compile/Triton cache must stay on node-local storage. CPFS can return
  # ESTALE while multiple TP workers atomically publish compiled artifacts.
  NODE_CACHE=$ACTION_HEAD_JOB_LOCAL_ROOT/cache/$(printf 'shard_%03d' "$SHARD_INDEX")
  export ACTION_HEAD_COMPILE_CACHE_ROOT=$NODE_CACHE/vllm
  export TORCH_EXTENSIONS_DIR=$NODE_CACHE/extensions
  export CUDA_CACHE_PATH=$NODE_CACHE/cuda
  export PYTORCH_KERNEL_CACHE_PATH=$NODE_CACHE/kernels
  export XDG_CACHE_HOME=$NODE_CACHE/xdg
  export TORCHINDUCTOR_CACHE_DIR=$NODE_CACHE/torchinductor
  export TRITON_CACHE_DIR=$NODE_CACHE/triton
  mkdir -p "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
    "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" "$PYTORCH_KERNEL_CACHE_PATH"
fi

# Action-Head vLLM env (query tail + ports) needed by BOTH the legacy TP2
# service-start below and the TP4 build-phase restart further down. Export
# unconditionally so a build-only resume (SKIP_RAW+SKIP_PLAN+BUILD_TP4) still
# configures the four-query tail worker when the TP4 vLLM comes up.
export ACTION_HEAD_ENABLE_QUERY_TAIL=1
export ACTION_HEAD_QUERY_CHECKPOINT="$QUERY_CHECKPOINT"
export ACTION_HEAD_QUERY_CACHE="$QUERY_CACHE"
export VLLM_GPUS VLLM_PORT VLLM_MTP_TOKENS=0 VLLM_MAX_MODEL_LEN=32768
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.70}

# Build-only resume: SKIP_RAW+SKIP_PLAN+BUILD_TP4. raw/plan reused from a prior
# run, so render is not needed and the TP2 vLLM below would be killed by the
# TP4 block anyway. But killing a TP2 vLLM that has just finished loading 35B
# does not release its GPU memory fast enough (vllm subprocess / CUDA context
# linger), so the TP4 vLLM then OOMs on startup — `Free memory on device cuda:0
# (8.72/96.0 GiB) ... less than desired GPU memory utilization (0.95, 91.2 GiB)`.
# Skip the initial TP2 vLLM+render+health-checks entirely; the TP4 block starts
# the only vLLM on a clean GPU.
BUILD_ONLY_RESUME=0
if [[ "${SKIP_RAW:-0}" == 1 && "${SKIP_PLAN:-0}" == 1 && "${BUILD_TP4_GPU:-1}" == 1 && "$LOCAL_SERVICES" != 1 ]]; then
  BUILD_ONLY_RESUME=1
  echo "[entry] build-only resume: skip initial TP2 vLLM+render, go straight to TP4 block"
fi

if [[ "$LOCAL_SERVICES" != 1 && "$BUILD_ONLY_RESUME" != 1 && "$RAW_ONLY" != 1 ]]; then
  setsid bash "$REPO/scripts/serve/vllm/vllm-vln-35B.sh" >"$LOG_ROOT/vllm_shard${SHARD_INDEX}.log" 2>&1 &
  VLLM_SERVICE_PID=$!
  PIDS+=("$VLLM_SERVICE_PID")
fi
if [[ "$LOCAL_SERVICES" != 1 && "$BUILD_ONLY_RESUME" != 1 && "$SKIP_RENDER_SERVICE" != 1 ]]; then
  SAGE_ROOT=/data/zerovln/SAGE-3D
  SCENES_ROOT=$SAGE_ROOT/sage_gs_root
  setsid env CUDA_VISIBLE_DEVICES="$RENDER_GPU" \
    PYTHONPATH="$TGT:$RS/submodules/diff-plane-rasterization:$RS/submodules/simple-knn:$RS" \
    "$GV/bin/python" "$RS/render_sim_dynamic.py" \
    --scenes_root "$SCENES_ROOT" --port "$RENDER_PORT" --gpus 0 --sh_degree 3 \
    --max_scenes_per_gpu "${MAX_SCENES_PER_GPU:-4}" \
    >"$LOG_ROOT/render_shard${SHARD_INDEX}${RAW_PART_INDEX:+_part${RAW_PART_INDEX}}.log" 2>&1 &
  PIDS+=("$!")
fi

if [[ "$BUILD_ONLY_RESUME" != 1 && "$RAW_ONLY" != 1 ]]; then
  deadline=$((SECONDS + 1800))
  until curl -sf -m 5 "$SLOW_VLLM_URL/v1/models" >/dev/null; do
    if [[ -n "${VLLM_SERVICE_PID:-}" ]] && ! kill -0 "$VLLM_SERVICE_PID" 2>/dev/null; then
      echo "vLLM process exited before readiness; inspect $LOG_ROOT/vllm_shard${SHARD_INDEX}.log" >&2
      exit 3
    fi
    (( SECONDS < deadline )) || { echo "vLLM startup timeout"; exit 3; }
    sleep 5
  done
fi
if [[ "$BUILD_ONLY_RESUME" != 1 && "$SKIP_RENDER_SERVICE" != 1 ]]; then
  deadline=$((SECONDS + 1800))
  until curl -sf -m 5 "$RENDER_URL/ping" | grep -q pong; do
    (( SECONDS < deadline )) || { echo "render startup timeout"; exit 4; }
    sleep 3
  done
fi

# node_entry gives every TP2 service a private 100-port range and starts them
# together. Hold already-healthy lanes here until all four services are ready so the
# timed build begins concurrently on all 16 GPUs instead of letting lane 0
# finish useful work while lanes 1-3 are still compiling.
if [[ "$RAW_ONLY" != 1 && -n "${BUILD_START_BARRIER:-}" ]]; then
  echo "[entry] vLLM healthy; waiting for node build barrier: $BUILD_START_BARRIER"
  deadline=$((SECONDS + ${BUILD_BARRIER_TIMEOUT:-3600}))
  until [[ -e "$BUILD_START_BARRIER" ]]; do
    (( SECONDS < deadline )) || { echo "build barrier timeout" >&2; exit 6; }
    sleep 2
  done
  echo "[entry] node build barrier released"
fi

started=$(date +%s)
BUILD_ARGS=(
  --render-url "$RENDER_URL"
  --sage-split "${SAGE_SPLIT:-action_head_train_p0_unified}"
  --out-dir "$SHARD_ROOT/raw"
  --split-manifest "$SPLIT_MANIFEST"
  --instruction-index 0
  --stride 2
  --history-len 8
  --action-chunk 16
  --step-m 0.1
  --allow-partial-horizon
  --fail-on-render-error
  --min-future-m 0.4
  --scene-shard-index "$RAW_SCENE_SHARD_INDEX"
  --scene-shard-count "$RAW_SCENE_SHARD_COUNT"
  --jsonl-name "$RAW_JSONL_NAME"
)
if (( MAX_SCENES > 0 )); then BUILD_ARGS+=(--max-scenes "$MAX_SCENES"); fi
if (( MAX_TASKS > 0 )); then BUILD_ARGS+=(--max-tasks "$MAX_TASKS"); fi
if [[ "$ACTION_HEAD_AS_TOOL_VERSION" == v6 && "${SKIP_RAW:-0}" == 1 && ! -s "$SHARD_ROOT/raw/sage_sft.jsonl" ]]; then
  echo "v6 read-only raw source is missing; refusing to render into a reused input" >&2
  exit 2
fi
if [[ "${SKIP_RAW:-0}" != "1" || ! -s "$SHARD_ROOT/raw/sage_sft.jsonl" ]]; then
  "$FM_VENV/bin/python" "$REPO/verl/sft/build_sage_sft_data.py" "${BUILD_ARGS[@]}"
else
  echo "[entry] SKIP_RAW=1 且 raw 已存在, 跳过渲染阶段: $SHARD_ROOT/raw/sage_sft.jsonl"
fi

if [[ "$RAW_ONLY" == 1 ]]; then
  stop_services
  echo "[action-head-data] raw-only shard=$SHARD_INDEX scene_partition=$RAW_SCENE_SHARD_INDEX/$RAW_SCENE_SHARD_COUNT output=$SHARD_ROOT/raw/$RAW_JSONL_NAME"
  exit 0
fi

PLAN_ARGS=(
  --raw-jsonl "$SHARD_ROOT/raw/sage_sft.jsonl"
  --split-manifest "$SPLIT_MANIFEST"
  --output "$SHARD_ROOT/nested_plan.jsonl"
  --slow-stride-m "$([[ "$ACTION_HEAD_AS_TOOL_VERSION" != v1 ]] && echo 1.6 || echo 0.8)"
  --slow-coverage-m 1.6
  --fast-stride-m 0.4
  --sampling-seed "${SAMPLING_SEED:-20260827}"
  --min-valid-action-steps 4
  --action-head-as-tool-version "$ACTION_HEAD_AS_TOOL_VERSION"
)
if [[ "$ACTION_HEAD_AS_TOOL_VERSION" == v1 ]]; then
  PLAN_ARGS+=(
    --fast-stride-min-m "${FAST_STRIDE_MIN_M:-0.2}"
    --fast-stride-max-m "${FAST_STRIDE_MAX_M:-0.6}"
  )
fi
if (( MAX_ANCHORS > 0 )); then PLAN_ARGS+=(--max-anchors "$MAX_ANCHORS"); fi
if [[ "${SKIP_PLAN:-0}" != "1" || ! -s "$SHARD_ROOT/nested_plan.jsonl" ]]; then
  "$FM_VENV/bin/python" "$REPO/verl/sft/plan_fm_nested_sft.py" "${PLAN_ARGS[@]}"
else
  echo "[entry] SKIP_PLAN=1 且 plan 已存在, 跳过 plan: $SHARD_ROOT/nested_plan.jsonl"
fi

# Build 阶段: 16 GPU 全给 vLLM (TP2 -> TP4, 吸收 render/visual 的 2 个闲置 GPU)。
# raw 阶段需要 render(3DGS) + visual(快编码), 那些 GPU 在 build 阶段完全空闲,
# 拉低整节点 SM% 导致 Lingjun 巡检判"利用率不达标"停 job。这里 raw 结束后停掉
# raw 阶段的 TP2 vLLM + render, 用全部 4 GPU/lane 起 TP4 vLLM, 让 build 阶段
# 16/16 卡都在做 vLLM 推理(SM% 高), 不再有闲置 GPU。
if [[ "${BUILD_TP4_GPU:-1}" == "1" && "$LOCAL_SERVICES" != 1 ]]; then
  echo "[entry] build 阶段切 TP4: 释放 raw 的 TP2 vLLM + render, 起 4 GPU TP4 vLLM"
  # 停 raw 阶段起的所有子进程(vLLM + render, 它们的 GPU 交给 TP4 vLLM)
  for pid in "${PIDS[@]:-}"; do
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  done
  sleep 5
  PIDS=()
  # TP4 vLLM 占全 4 GPU/lane: VLLM_GPUS(base,base+1) + RENDER_GPU(base+2) + VISUAL_GPU(base+3)
  export VLLM_GPUS="${VLLM_GPUS:-},${RENDER_GPU:-},${VISUAL_GPU:-}"
  export VLLM_TP_SIZE=4
  export VLLM_GPU_MEMORY_UTILIZATION=${BUILD_TP4_UTIL:-0.95}
  export VLLM_MAX_MODEL_LEN=32768
  setsid bash "$REPO/scripts/serve/vllm/vllm-vln-35B.sh" >"$LOG_ROOT/vllm_shard${SHARD_INDEX}_tp4.log" 2>&1 &
  PIDS+=("$!")
  deadline=$((SECONDS + 1800))
  until curl -sf -m 5 "$SLOW_VLLM_URL/v1/models" >/dev/null; do
    (( SECONDS < deadline )) || { echo "TP4 vLLM startup timeout"; exit 3; }
    sleep 5
  done
  echo "[entry] TP4 vLLM 健康 (${VLLM_GPUS})"
fi

VISUAL_GPUS=${VISUAL_GPUS:-${VISUAL_GPU:-3}}
IFS=',' read -r -a _VISUAL_GPU_ARRAY <<< "$VISUAL_GPUS"
BUILD_WORKERS=${BUILD_WORKERS:-${#_VISUAL_GPU_ARRAY[@]}}
(( BUILD_WORKERS >= 1 && BUILD_WORKERS <= ${#_VISUAL_GPU_ARRAY[@]} )) || {
  echo "BUILD_WORKERS must be within available VISUAL_GPUS" >&2; exit 7;
}
_BUNDLE_VISUAL_DEVICES=cuda:0
_SINGLE_WORKER_VISIBLE_GPUS=${_VISUAL_GPU_ARRAY[0]}
if (( BUILD_WORKERS == 1 && ${#_VISUAL_GPU_ARRAY[@]} > 1 )); then
  # Compatibility resume for an existing unsplit bundles/ checkpoint. One
  # process owns the old causal state, but its visual runner may still use both
  # free lane GPUs, keeping the 4-card lane occupied without duplicating data.
  _SINGLE_WORKER_VISIBLE_GPUS=$(IFS=,; echo "${_VISUAL_GPU_ARRAY[*]}")
  _local_visual_devices=()
  for ((index=0; index<${#_VISUAL_GPU_ARRAY[@]}; index++)); do
    _local_visual_devices+=("cuda:$index")
  done
  _BUNDLE_VISUAL_DEVICES=$(IFS=,; echo "${_local_visual_devices[*]}")
fi
COMMON_BUNDLE_ARGS=(
  --beam-width "${BEAM_WIDTH:-4}" \
  --beam-seed "${BEAM_SEED:-20260922}" \
  --render-url "$RENDER_URL" \
  --nested-plan "$SHARD_ROOT/nested_plan.jsonl" \
  --frame-dir "$SHARD_ROOT/raw" \
  --backbone "$FAST_BACKBONE" \
  --vllm-url "$SLOW_VLLM_URL" \
  --model Qwen3.6-35B \
  --slow-hidden-size 2048 \
  --cache-root "$QUERY_CACHE" \
  --device cuda:0 \
  --visual-devices "$_BUNDLE_VISUAL_DEVICES" \
  --visual-prefetch "${VISUAL_PREFETCH:-4}" \
  --pipeline-mode "${PIPELINE_MODE:-stream}" \
  --npz-compress-level "${NPZ_COMPRESS_LEVEL:-1}" \
  --max-anchors "${MAX_ANCHORS:-0}" \
  --pass-k "${PASS_K:-2}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --decision-filter vlm \
  --stop-success-radius-m 1.0 \
  --timeout "${REQUEST_TIMEOUT:-$([[ "$ACTION_HEAD_AS_TOOL_VERSION" != v1 ]] && echo 600 || echo 60)}" \
  --action-head-as-tool-version "$ACTION_HEAD_AS_TOOL_VERSION"
)
if [[ "$ACTION_HEAD_AS_TOOL_VERSION" == v4.3 && ! -f "$SHARD_ROOT/v43_smoke/smoke_validated.json" ]]; then
  CUDA_VISIBLE_DEVICES="$_SINGLE_WORKER_VISIBLE_GPUS" /usr/local/bin/python3.12 \
    "$REPO/verl/sft/build_fm_nested_bundles.py" \
    "${COMMON_BUNDLE_ARGS[@]}" --max-anchors "${V43_SMOKE_ANCHORS:-32}" \
    --output-dir "$SHARD_ROOT/v43_smoke"
  /usr/local/bin/python3.12 "$REPO/verl/sft/verify_v43_teacher_bundle.py" \
    --bundle "$SHARD_ROOT/v43_smoke"
fi

if [[ "$ACTION_HEAD_AS_TOOL_VERSION" == v6 && ! -f "$SHARD_ROOT/v6_smoke/smoke_validated.json" ]]; then
  CUDA_VISIBLE_DEVICES="$_SINGLE_WORKER_VISIBLE_GPUS" /usr/local/bin/python3.12 \
    "$REPO/verl/sft/build_fm_nested_bundles.py" \
    "${COMMON_BUNDLE_ARGS[@]}" --max-anchors "${V6_SMOKE_ANCHORS:-32}" \
    --output-dir "$SHARD_ROOT/v6_smoke"
  /usr/local/bin/python3.12 "$REPO/verl/sft/verify_v43_teacher_bundle.py" \
    --bundle "$SHARD_ROOT/v6_smoke" --action-head-as-tool-version v6
fi

if (( BUILD_WORKERS == 1 )); then
  CUDA_VISIBLE_DEVICES="$_SINGLE_WORKER_VISIBLE_GPUS" /usr/local/bin/python3.12 \
    "$REPO/verl/sft/build_fm_nested_bundles.py" \
    --output-dir "$SHARD_ROOT/bundles" \
    "${COMMON_BUNDLE_ARGS[@]}"
else
  BUILD_PIDS=()
  for ((worker=0; worker<BUILD_WORKERS; worker++)); do
    output="$SHARD_ROOT/bundles_part_$(printf '%02d' "$worker")"
    CUDA_VISIBLE_DEVICES="${_VISUAL_GPU_ARRAY[$worker]}" /usr/local/bin/python3.12 \
      "$REPO/verl/sft/build_fm_nested_bundles.py" \
      --output-dir "$output" \
      --task-shard-index "$worker" --task-shard-count "$BUILD_WORKERS" \
      "${COMMON_BUNDLE_ARGS[@]}" &
    BUILD_PIDS+=("$!")
    PIDS+=("$!")
  done
  build_status=0
  for pid in "${BUILD_PIDS[@]}"; do
    wait "$pid" || build_status=1
  done
  (( build_status == 0 )) || { echo "one or more visual build workers failed" >&2; exit 8; }
  /usr/local/bin/python3.12 "$REPO/verl/sft/merge_fm_nested_partitions.py" \
    --shard-root "$SHARD_ROOT" --partition-count "$BUILD_WORKERS"
fi

elapsed=$(( $(date +%s) - started ))
/usr/local/bin/python3.12 - "$SHARD_ROOT" "$elapsed" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); elapsed=int(sys.argv[2])
rows=sum(1 for line in (root/'raw/sage_sft.jsonl').open() if line.strip())
plan=sum(1 for line in (root/'nested_plan.jsonl').open() if line.strip())
samples=sum(1 for line in (root/'bundles/manifest.jsonl').open() if line.strip())
(root/'timing.json').write_text(json.dumps({'elapsed_seconds':elapsed,'raw_rows':rows,'planned_windows':plan,'accepted_samples':samples},indent=2)+'\n')
PY
stop_services
touch "$DATA_ROOT/status/$(printf 'shard_%03d.done' "$SHARD_INDEX")"

if [[ "$SHARD_INDEX" == 0 && "${MERGE_ON_SHARD0:-1}" == 1 ]]; then
  deadline=$((SECONDS + ${MERGE_WAIT_SECONDS:-7200}))
  while (( $(find "$DATA_ROOT/status" -name 'shard_*.done' | wc -l) < SHARD_COUNT )); do
    (( SECONDS < deadline )) || { echo "timed out waiting for shard markers"; exit 5; }
    sleep 15
  done
  /usr/local/bin/python3.12 "$REPO/verl/sft/merge_fm_nested_shards.py" \
    --data-root "$DATA_ROOT" --shard-count "$SHARD_COUNT" \
    --action-head-as-tool-version "$ACTION_HEAD_AS_TOOL_VERSION"
  if [[ "$LOCAL_SERVICES" == 1 || "${SKIP_SPLIT_AUDIT:-0}" == 1 ]]; then
    echo "[action-head-data] smoke has one split; full split audit deferred to formal merge"
  else
    /usr/local/bin/python3.12 "$REPO/verl/sft/audit_action_head_split.py" \
      --manifest "$DATA_ROOT/manifest.jsonl" \
      --action-head-as-tool-version "$ACTION_HEAD_AS_TOOL_VERSION"
  fi
fi

echo "[action-head-data] shard=$SHARD_INDEX/$SHARD_COUNT elapsed_seconds=$elapsed complete"
