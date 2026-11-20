#!/usr/bin/env bash
# v6 direct policy on the frozen English R2R/RxR validation set.
set -euo pipefail
REPO=/workspace/ZeroVLN
ART=/data/zerovln/artifacts
GV=/opt/navigation-runtime/venv
NAV=/opt/navigation-runtime/VLMEvalKit/3rdparty/RynnBrain/navigation
SWEEP=$REPO/verl/scripts/eval/qwen36_4query_ckpt_sweep
BOOT=$REPO/verl/scripts/train/action_head_bootstrap
: "${EXPR_NAME:?}" "${CHECKPOINT_DIR:?}" "${BACKBONE:?}"
[[ "$EXPR_NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 2
OUTPUT=$ART/$EXPR_NAME
# One task per DLC container; keep Unix-domain socket paths below 108 bytes.
RUNTIME=/tmp/v6vlnce
MESA_SEED=/data/zerovln/vln_verl/runtime_inputs/habitat-mesa-25.2.8-ubuntu24.04-amd64
MESA_RUNTIME=$RUNTIME/mesa
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy CUDA_VISIBLE_DEVICES
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
unset SAGE_BENCH_PROTOCOL_ROOT
export TMPDIR=$RUNTIME/tmp XDG_CACHE_HOME=$RUNTIME/xdg HF_HOME=$RUNTIME/hf
export TRITON_CACHE_DIR=$RUNTIME/triton TORCHINDUCTOR_CACHE_DIR=$RUNTIME/torchinductor
export TORCH_EXTENSIONS_DIR=$RUNTIME/torch_extensions PYTORCH_KERNEL_CACHE_PATH=$RUNTIME/torch_kernels
export CUDA_CACHE_PATH=$RUNTIME/cuda RTC_CACHE_PATH=$RUNTIME/ppu_rtc PPU_LIB_RTC_CACHE_PATH=$RUNTIME/ppu_rtc
export MPLCONFIGDIR=$RUNTIME/matplotlib NUMBA_CACHE_DIR=$RUNTIME/numba
export VLLM_RPC_BASE_PATH=$RUNTIME/rpc VLLM_CACHE_ROOT=$RUNTIME/vllm DG_CACHE_DIR=$RUNTIME/dg
export PIP_CACHE_DIR=$RUNTIME/pip HF_DATASETS_CACHE=$RUNTIME/hf_datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PPU_SDK=/usr/local/PPU_SDK CUDA_PATH=/usr/local/PPU_SDK/CUDA_SDK
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=/opt/accl-p:$CUDA_PATH/lib64:$PPU_SDK/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$REPO:$REPO/verl:$REPO/vln_demo/src:$SWEEP:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export COMPANION_BRAIN_MODE=sft ACTION_HEAD_QUERY_TEXT_BUDGET=256
export ACTION_HEAD_INVALID_TOOL_CALL_POLICY=feedback ACTION_HEAD_V43_PROMPT_REVISION=legacy
export VLN_HABITAT_RENDER_URL=http://127.0.0.1:7040 VLN_RENDER_MODE=batch
export RXR_LANGUAGES=en-US,en-IN RXR_ROLES=guide RXR_EPISODES_PER_SCENE=0
export RYNNBRAIN_NAV_DIR=$NAV RYNNVLN_LANGUAGE=en RYNNVLN_TURN_ANGLE=30
export RYNNVLN_HFOV=120 RYNNVLN_ALLOW_SLIDING=1
export QUERY_CACHE=$RUNTIME/query_handoff
mkdir -p "$OUTPUT/logs" "$OUTPUT/evaluation" "$RUNTIME" "$QUERY_CACHE"
for key in TMPDIR XDG_CACHE_HOME HF_HOME TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR \
  TORCH_EXTENSIONS_DIR PYTORCH_KERNEL_CACHE_PATH CUDA_CACHE_PATH RTC_CACHE_PATH \
  MPLCONFIGDIR NUMBA_CACHE_DIR VLLM_RPC_BASE_PATH VLLM_CACHE_ROOT DG_CACHE_DIR \
  PIP_CACHE_DIR HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE; do
  mkdir -p "${!key}"
done
PIDS=(); PID_FILES=()
GPU_IDS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
cleanup() {
  code=$?
  trap - EXIT INT TERM
  args=()
  for pid in "${PIDS[@]}"; do args+=(--pid "$pid"); done
  for file in "${PID_FILES[@]}"; do args+=(--pid-file "$file"); done
  /usr/local/bin/python3.12 -B "$BOOT/collection_runtime.py" stop \
    --gpu-ids "$GPU_IDS" --output "$OUTPUT/logs/cleanup.json" "${args[@]}" || code=1
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  rm -rf -- "$RUNTIME"
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
wait_ready() {
  local url=$1 pid=$2 logfile=$3 deadline=$((SECONDS+1800))
  until curl -fsS --max-time 5 "$url" >/dev/null; do
    kill -0 "$pid" 2>/dev/null || { tail -60 "$logfile"; return 1; }
    (( SECONDS < deadline )) || { tail -60 "$logfile"; return 1; }
    sleep 3
  done
}
/usr/local/bin/python3.12 -B - "$OUTPUT/logs/runtime_environment.json" <<'PY'
import json, platform, sys
from pathlib import Path
record = dict(python=sys.version, libc=platform.libc_ver(), machine=platform.machine(),
              os_release=Path('/etc/os-release').read_text())
Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
PY
/usr/local/bin/python3.12 -B "$REPO/scripts/eval/submit_v6_vlnce_dlc.py" \
  --check-only --expected-record "$OUTPUT/launch/inputs.json" \
  --checkpoint "$CHECKPOINT_DIR" --ladder-stage "${VLNCE_LADDER_STAGE:-full}"
ports=7040,28900,29000,29100,29200,29300,29400
for actor in $(seq 0 31); do ports+=,$((9100+actor)); done
/usr/local/bin/python3.12 -B "$BOOT/collection_runtime.py" preflight \
  --ports "$ports" --gpu-ids "$GPU_IDS" --output "$OUTPUT/logs/preflight.json"
# The PPU model image has no system Mesa. Copy the version-bound, verified
# dependency input to container-local storage and explicitly select its ICD.
mkdir -p "$MESA_RUNTIME"
cp -a "$MESA_SEED/." "$MESA_RUNTIME/"

# CPU Habitat renderer is shared; each Gym retains independent navigation state.
setsid env LD_LIBRARY_PATH="$GV/lib:$MESA_RUNTIME/lib:$LD_LIBRARY_PATH" \
  __EGL_VENDOR_LIBRARY_FILENAMES="$MESA_RUNTIME/50_mesa.json" LIBGL_DRIVERS_PATH="$MESA_RUNTIME/dri" \
  VLN_RENDER_SCENE_CACHE_SIZE=128 VLN_RENDER_THREADS_PER_WORKER=4 \
  bash "$REPO/scripts/demo/remote_habitat/start.sh" --port 7040 \
  >"$OUTPUT/logs/habitat_render.log" 2>&1 &
PIDS+=("$!")
wait_ready "$VLN_HABITAT_RENDER_URL/health" "${PIDS[-1]}" "$OUTPUT/logs/habitat_render.log"
readarray -t INITIAL < <(/usr/local/bin/python3.12 -B - "$OUTPUT/launch/inputs.json" <<'PY'
import json, sys
record = json.load(open(sys.argv[1]))
for dataset in ('r2r', 'rxr'):
    row = record['initial_cases'][dataset]
    print(row['scene'])
    print(row['episode_id'])
PY
)
for actor in $(seq 0 31); do
  dataset=r2r; (( actor < 16 )) || dataset=rxr
  initial=0; (( actor < 16 )) || initial=2
  work=$RUNTIME/gym_$actor
  mkdir -p "$work"
  ln -s "$NAV/data" "$work/data"
  cp "$REPO/vln_demo/src/vln_demo.yaml" "$work/vln_demo.yaml"
  setsid bash -c 'cd "$1"; exec env LD_LIBRARY_PATH="$2/lib:$3/lib:$LD_LIBRARY_PATH" __EGL_VENDOR_LIBRARY_FILENAMES="$3/50_mesa.json" LIBGL_DRIVERS_PATH="$3/dri" "$2/bin/python" -B -u -m vln_gym.vlnce_action_head --backend "$4" --port "$5" --split val_unseen --max_steps 0 --output_path "$6" --scene "$7" --episode_id "$8"' \
    _ "$work" "$GV" "$MESA_RUNTIME" "$dataset" "$((9100+actor))" "$OUTPUT/evaluation/gym_actor_$actor" "${INITIAL[$initial]}" "${INITIAL[$((initial+1))]}" \
    >"$OUTPUT/logs/gym_$actor.log" 2>&1 &
  PIDS+=("$!")
done
for actor in $(seq 0 31); do
  wait_ready "http://127.0.0.1:$((9100+actor))/health" "${PIDS[$((actor+1))]}" "$OUTPUT/logs/gym_$actor.log"
done
# Every actor can draw either dataset from the shared queue. Check switching
# both ways on the same real Gym before loading any model.
/usr/local/bin/python3.12 -B "$REPO/scripts/eval/smoke_v6_vlnce.py" \
  --bench-url http://127.0.0.1:9100 --dataset both \
  --output "$OUTPUT/validation/backend_switch_http_smoke.json" \
  >"$OUTPUT/logs/backend_switch_http_smoke.log" 2>&1

# Keep request IDs and inference metrics, without writing full prompts at DEBUG.
export ACTION_HEAD_LOGGING_CONFIG_OVERRIDE=$RUNTIME/vllm-log-info.json
/usr/local/bin/python3.12 -B - "$REPO/vln_demo/vllm/vllm_log_hidden_id.json" "$ACTION_HEAD_LOGGING_CONFIG_OVERRIDE" <<'PY'
import json, sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text())
value['loggers']['vllm.entrypoints.logger']['level'] = 'INFO'
value['handlers']['request_debug']['level'] = 'INFO'
Path(sys.argv[2]).write_text(json.dumps(value) + '\n')
PY

# Preserve the established 6xTP2 Qwen query-tail services and compiled path.
launch_brain() {
  local replica=$1 cache=$RUNTIME/compile_$1
  env QUERY_CACHE="$QUERY_CACHE" CHECKPOINT_DIR="$CHECKPOINT_DIR" \
    VLLM_LOG="$OUTPUT/logs/vllm_$replica.log" VLLM_PID_FILE="$RUNTIME/vllm_$replica.pid" \
    VLLM_GPUS_OVERRIDE="$((replica*2)),$((replica*2+1))" VLLM_PORT_OVERRIDE="$((28900+replica*100))" \
    ACTION_HEAD_COMPILE_CACHE_ROOT_OVERRIDE="$cache" TORCHINDUCTOR_CACHE_DIR="$cache/torchinductor" \
    TRITON_CACHE_DIR="$cache/triton" XDG_CACHE_HOME="$cache/xdg" \
    VLLM_MAX_MODEL_LEN_OVERRIDE=131072 VLLM_MAX_NUM_SEQS_OVERRIDE=64 \
    VLLM_MAX_NUM_BATCHED_TOKENS_OVERRIDE=32768 VLLM_GPU_MEMORY_UTILIZATION_OVERRIDE=0.92 \
    bash "$REPO/scripts/eval/qwen36_4query_ckpt_sweep/launch_vllm_for_ckpt.sh"
}
for replica in $(seq 0 5); do PID_FILES+=("$RUNTIME/vllm_$replica.pid"); done
export -f launch_brain
export RUNTIME OUTPUT SWEEP QUERY_CACHE CHECKPOINT_DIR
launchers=()
for replica in $(seq 0 5); do
  setsid bash -c 'launch_brain "$1"' _ "$replica" >"$OUTPUT/logs/vllm_launcher_$replica.log" 2>&1 &
  launchers+=("$!"); PIDS+=("$!")
done
failed=0
for pid in "${launchers[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 3
urls=()
for actor in $(seq 0 31); do urls+=("http://127.0.0.1:$((9100+actor))"); done
csv=$(IFS=,; echo "${urls[*]}")
setsid /usr/local/bin/python3.12 -B -u "$REPO/scripts/eval/v6_vlnce_eval.py" \
  run-pool --dataset both --scoring-mode merged_batch \
  --ladder-stage "${VLNCE_LADDER_STAGE:-full}" \
  --checkpoint-dir "$CHECKPOINT_DIR" --backbone "$BACKBONE" --query-cache "$QUERY_CACHE" \
  --bench-urls "$csv" --fm-gpus 12,13,14,15 \
  --vllm-urls http://127.0.0.1:28900,http://127.0.0.1:29000,http://127.0.0.1:29100,http://127.0.0.1:29200,http://127.0.0.1:29300,http://127.0.0.1:29400 \
  --output "$OUTPUT/evaluation" --max-tool-calls 100 --max-turns 0 \
  --max-batch-size 8 --batch-wait-ms 12 \
  >"$OUTPUT/logs/policy_pool.log" 2>&1 &
PIDS+=("$!")
failed=0
wait "${PIDS[-1]}" || failed=1
/usr/local/bin/python3.12 -B "$REPO/scripts/eval/v6_vlnce_eval.py" aggregate \
  --dataset both --scoring-mode merged_batch \
  --ladder-stage "${VLNCE_LADDER_STAGE:-full}" \
  --checkpoint-dir "$CHECKPOINT_DIR" --backbone "$BACKBONE" \
  --output "$OUTPUT/evaluation" --max-tool-calls 100 --max-turns 0 || failed=1
(( failed == 0 ))
