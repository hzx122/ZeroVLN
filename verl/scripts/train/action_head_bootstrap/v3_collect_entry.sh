#!/usr/bin/env bash
# One-node v3 bootstrap collection: traced rollout -> mismatch -> A* -> R1-R4.
set -euo pipefail

VERSION=
BOOTSTRAP_ROUND=
while (($#)); do
  case "$1" in
    --action-head-as-tool-version) VERSION=${2:?}; shift 2 ;;
    --bootstrap-round) BOOTSTRAP_ROUND=${2:?}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$VERSION" == v3 || "$VERSION" == v4.3 || "$VERSION" == v6 || "${COMPANION_EVAL:-0}" == 1 ]] || { echo "collector requires v3, v4.3 or v6" >&2; exit 2; }
[[ "$BOOTSTRAP_ROUND" =~ ^[1-9][0-9]*$ ]] || { echo "--bootstrap-round must be positive" >&2; exit 2; }

REPO=/workspace/ZeroVLN
HERE=$REPO/verl/scripts/train/action_head_bootstrap
GV=/opt/navigation-runtime/venv
RS=/workspace/agentic_sim/repo/ABot-Navigation/render_server
RENDER_LIBS=$REPO/vln_demo/render_libs
: "${OUTPUT_DIR:?}" "${MANIFEST:?}" "${CHECKPOINT_DIR:?}" "${BACKBONE:?}"
if [[ "${COMPANION_EVAL:-0}" == 1 ]]; then
  : "${EVAL_OUTPUT:?}" "${SUBSET_MANIFEST:?}"
  [[ "${ROLLOUT_ONLY:-0}" == 1 && "${COLLECTION_PHASE:-all}" == all ]] || exit 2
  case "$VERSION" in v1|v2|v3|v4|v4.[1-9]|v4.10|v4.11|v6|primitive_map) ;; *) echo "unsupported query protocol: $VERSION" >&2; exit 2;; esac
fi
MAX_CASES=${MAX_CASES:-5203}
ROLLOUT_ACTORS=${ROLLOUT_ACTORS:-160}
MAX_CORRECTION_ANCHORS=${MAX_CORRECTION_ANCHORS:-1000}
RENDERER_CACHE_SCENES=${RENDERER_CACHE_SCENES:-64}
# Physical GPU IDs: local smoke uses only 8..15; DLC retains 0..15.
COLLECTION_GPU_START=${COLLECTION_GPU_START:-0}
COLLECTION_GPU_COUNT=${COLLECTION_GPU_COUNT:-16}
VLLM_REPLICAS=${VLLM_REPLICAS:-6}
BUILD_LANES=${BUILD_LANES:-$((COLLECTION_GPU_COUNT / 2))}
BUILD_WORKERS=${BUILD_WORKERS:-$BUILD_LANES}
V6_GOLDEN_EXTRA_MAX=${V6_GOLDEN_EXTRA_MAX:-0}
COLLECTION_PHASE=${COLLECTION_PHASE:-all}
[[ "$COLLECTION_PHASE" == all || "$COLLECTION_PHASE" == build ]] || exit 2
SOURCE_ROLLOUT_DIR=${SOURCE_ROLLOUT_DIR:-}
ROLLOUT_GYM_BASE_PORT=${ROLLOUT_GYM_BASE_PORT:-9000}
BUILD_GYM_BASE_PORT=${BUILD_GYM_BASE_PORT:-8938}
CASE_SCHEDULING=${CASE_SCHEDULING:-static}
MAX_ACTIVE_SCENES_PER_RENDERER=${MAX_ACTIVE_SCENES_PER_RENDERER:-48}
MAX_ACTIVE_CASES_PER_RENDERER=${MAX_ACTIVE_CASES_PER_RENDERER:-64}
AH_MAX_BATCH_SIZE=${AH_MAX_BATCH_SIZE:-32}
AH_BATCH_WAIT_MS=${AH_BATCH_WAIT_MS:-50}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-64}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.92}
VLLM_STARTUP_MODE=${VLLM_STARTUP_MODE:-warm-one-parallel}
GYM_CPU_THREADS=${GYM_CPU_THREADS:-1}
NPZ_COMPRESS_LEVEL=${NPZ_COMPRESS_LEVEL:-1}
MAX_TURNS=${MAX_TURNS:-12}
MAX_TOOL_CALLS=${MAX_TOOL_CALLS:-12}
CASE_SEED=${CASE_SEED:-20260903}
BOOTSTRAP_NODE_INDEX=${BOOTSTRAP_NODE_INDEX:-0}
BOOTSTRAP_NODE_COUNT=${BOOTSTRAP_NODE_COUNT:-1}
COLLECTION_ITERATION=${COLLECTION_ITERATION:-1}
SAGE_ASSET_SET=${SAGE_ASSET_SET:-train_p0_unified}
SAGE_DATASET_SPLIT=${SAGE_DATASET_SPLIT:-action_head_train_p0_unified}
[[ "$BOOTSTRAP_NODE_COUNT" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$BOOTSTRAP_NODE_INDEX" =~ ^[0-9]+$ ]] || exit 2
(( BOOTSTRAP_NODE_INDEX < BOOTSTRAP_NODE_COUNT )) || exit 2
[[ "$COLLECTION_ITERATION" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$VLLM_STARTUP_MODE" == sequential || "$VLLM_STARTUP_MODE" == warm-one-parallel ]] || {
  echo "VLLM_STARTUP_MODE must be sequential or warm-one-parallel" >&2; exit 2;
}
(( COLLECTION_GPU_START >= 0 && COLLECTION_GPU_COUNT == 2 * VLLM_REPLICAS + 4 &&
   VLLM_REPLICAS > 0 && BUILD_LANES > 0 && BUILD_LANES <= COLLECTION_GPU_COUNT / 2 &&
   BUILD_WORKERS >= BUILD_LANES )) || {
  echo "invalid collection GPU layout" >&2; exit 2;
}
[[ "$V6_GOLDEN_EXTRA_MAX" =~ ^[0-3]$ ]] || { echo "V6_GOLDEN_EXTRA_MAX must be 0..3" >&2; exit 2; }
if (( V6_GOLDEN_EXTRA_MAX > 0 )); then
  [[ "$VERSION" == v6 ]] || { echo "golden extras require v6" >&2; exit 2; }
  (( BUILD_LANES <= COLLECTION_GPU_COUNT / 2 - 2 )) || {
    echo "golden extras reserve two build GPUs for TP2 vLLM; reduce BUILD_LANES" >&2; exit 2;
  }
  : "${VLLM_MODEL:?VLLM_MODEL is required for golden extra anchors}"
fi
if [[ "$(hostname)" == dsw-* ]] &&
   (( COLLECTION_GPU_START != 8 || COLLECTION_GPU_COUNT != 8 )) &&
   [[ "${ALLOW_DSW_FULL16:-0}" != 1 ]]; then
  echo "DSW smoke is restricted to physical GPU 8..15 unless ALLOW_DSW_FULL16=1" >&2; exit 2
fi
# Renderer --gpus takes physical IDs; only model children receive a CUDA mask.
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "unset outer CUDA_VISIBLE_DEVICES; use COLLECTION_GPU_START/COUNT" >&2; exit 2; }
RENDER_GPU_START=$((COLLECTION_GPU_START + VLLM_REPLICAS * 2))
AH_GPU=$((COLLECTION_GPU_START + COLLECTION_GPU_COUNT - 1))
VISION_GPU_START=$((COLLECTION_GPU_START + COLLECTION_GPU_COUNT / 2))
BUILD_RENDER_GPU_START=$((COLLECTION_GPU_START + (V6_GOLDEN_EXTRA_MAX > 0 ? 2 : 0)))
COLLECTION_GPU_IDS=$(seq -s, "$COLLECTION_GPU_START" "$AH_GPU")
NODE_CACHE=${NODE_CACHE_ROOT:-$OUTPUT_DIR/node_cache}
VLLM_COMPILE_CACHE_ROOT=${VLLM_COMPILE_CACHE_ROOT:-${TMPDIR:-/tmp}/action_head_bootstrap_vllm/$(hostname)}
# Request-scoped query handoff is transient. v6 joint training reconstructs
# queries from saved slow context; only legacy/explicit extra-anchor modes archive them.
QUERY_HANDOFF_CACHE="$NODE_CACHE/query_cache"
if [[ "$VERSION" == v4.3 || "$VERSION" == v4.11 || "$VERSION" == v6 ]]; then
  QUERY_HANDOFF_CACHE="${TMPDIR:-/tmp/v43_bootstrap}/query_handoff"
  [[ "$QUERY_HANDOFF_CACHE" == /tmp/* ]] || { echo "v4.3 query handoff must be container-local" >&2; exit 2; }
fi
mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/rollout" "$OUTPUT_DIR/corrections/lanes" "$QUERY_HANDOFF_CACHE"

PIDS=()
VLLM_PID_FILES=()
PHASE=initial
stop_phase() {
  local args=() file pid
  for file in "${VLLM_PID_FILES[@]}"; do args+=(--pid-file "$file"); done
  for pid in "${PIDS[@]}"; do args+=(--pid "$pid"); done
  if (( ${#args[@]} )); then
    /usr/local/bin/python3.12 "$HERE/collection_runtime.py" stop \
      --gpu-ids "$COLLECTION_GPU_IDS" --output "$OUTPUT_DIR/logs/${PHASE}_cleanup.json" "${args[@]}"
  fi
  wait 2>/dev/null || true
  PIDS=(); VLLM_PID_FILES=()
}

cleanup() { code=$?; trap - EXIT INT TERM; stop_phase; exit "$code"; }
trap cleanup EXIT INT TERM

export PYTHONUNBUFFERED=1
export PPU_SDK=/usr/local/PPU_SDK PPU_HOME=/usr/local/PPU_SDK
export CUDA_PATH=/usr/local/PPU_SDK/CUDA_SDK
export CUDA_HOME=$CUDA_PATH
export PATH="$CUDA_PATH/bin:$PPU_SDK/bin:$PATH"
export SAGE_COLLISION=usd_rooms SAGE_FURNITURE_COLLISION=none
export SAGE_ROBOT_RADIUS=0.07 SAGE_CAMERA_HEIGHT=1.2 SAGE_HFOV=115 SAGE_EPISODES_PER_SCENE=0
export SAGE_USD_COLLISION_MASK_ROOT=/data/zerovln/SAGE-3D/sage_usd_rooms_masks
export SAGE_USD_STATIC_BOUNDARY_MASK_ROOT=$SAGE_USD_COLLISION_MASK_ROOT
export PYTHONPATH="$RENDER_LIBS:$RS/submodules/diff-plane-rasterization:$RS/submodules/simple-knn:$RS:$REPO:$REPO/verl:$REPO/verl/sft:$REPO/vln_demo/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/opt/accl-p:/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/lib:$GV/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"

# Refuse occupied ports before any service starts, including later build ports.
required_ports=()
if [[ "$COLLECTION_PHASE" == all ]]; then
  required_ports+=(7038 28899 7060 7061 7062)
  for replica in $(seq 0 $((VLLM_REPLICAS-1))); do required_ports+=("$((28900+replica*100))"); done
  for actor in $(seq 0 $((ROLLOUT_ACTORS-1))); do required_ports+=("$((ROLLOUT_GYM_BASE_PORT+actor))"); done
fi
if [[ "${ROLLOUT_ONLY:-0}" != 1 || "$COLLECTION_PHASE" == build ]]; then
  for lane in $(seq 0 $((BUILD_LANES-1))); do required_ports+=("$((7060+lane))"); done
  for worker in $(seq 0 $((BUILD_WORKERS-1))); do required_ports+=("$((BUILD_GYM_BASE_PORT+worker))"); done
fi
if [[ "$COLLECTION_PHASE" == build && "$V6_GOLDEN_EXTRA_MAX" != 0 ]]; then required_ports+=(28900); fi
port_list=$(IFS=,; echo "${required_ports[*]}")
/usr/local/bin/python3.12 "$HERE/collection_runtime.py" preflight --ports "$port_list" \
  --gpu-ids "$COLLECTION_GPU_IDS" --output "$OUTPUT_DIR/logs/preflight.json"
if [[ "$COLLECTION_PHASE" == all ]]; then
PHASE=rollout
# Phase A: TP2 Qwen replicas, 3 renderers, one dynamically-batched Action Head.
for renderer in 0 1 2; do
  setsid "$GV/bin/python" "$HERE/single_gpu_renderer.py" "$RS/render_sim_dynamic.py" \
    --scenes_root /data/zerovln/SAGE-3D/sage_gs_root \
    --port "$((7060+renderer))" --gpus "$((RENDER_GPU_START+renderer))" --sh_degree 3 \
    --max_scenes_per_gpu "$RENDERER_CACHE_SCENES" >"$OUTPUT_DIR/logs/rollout_render_${renderer}.log" 2>&1 &
  PIDS+=("$!")
done
for renderer in 0 1 2; do
  deadline=$((SECONDS+1200)); until curl -fsS -m5 "http://127.0.0.1:$((7060+renderer))/ping" | grep -q pong; do
    ((SECONDS<deadline)) || exit 3; sleep 3; done
done
setsid "$GV/bin/python" "$HERE/service_router.py" --port 7038 --mode scene \
  --upstreams http://127.0.0.1:7060,http://127.0.0.1:7061,http://127.0.0.1:7062 \
  >"$OUTPUT_DIR/logs/rollout_render_router.log" 2>&1 & PIDS+=("$!")

for actor in $(seq 0 $((ROLLOUT_ACTORS-1))); do
  port=$((ROLLOUT_GYM_BASE_PORT+actor))
  lazy_episodes=${ROLLOUT_LAZY_EPISODES:-${SAGE_LAZY_EPISODES:-0}}
  # Keep one full catalog for the companion's immutable /spec audit. Other
  # actors retain bootstrap's lazy episode lookup and independent environments.
  if [[ "${COMPANION_EVAL:-0}" == 1 && "$actor" == 0 ]]; then lazy_episodes=0; fi
  setsid env OMP_NUM_THREADS="$GYM_CPU_THREADS" MKL_NUM_THREADS="$GYM_CPU_THREADS" \
    OPENBLAS_NUM_THREADS="$GYM_CPU_THREADS" NUMEXPR_NUM_THREADS="$GYM_CPU_THREADS" \
    SAGE_LAZY_EPISODES="$lazy_episodes" \
    GYM_WORK_DIR="$NODE_CACHE/gyms/$port" \
    DEMO_HTTP_PORT=$port SAGE_RENDER_URL=http://127.0.0.1:7038 SAGE_ACTION_HEAD_ASSET_SET="$SAGE_ASSET_SET" \
    bash "$REPO/vln_demo/run_action_head_curriculum_testset.sh" gym >"$OUTPUT_DIR/logs/gym_${port}.log" 2>&1 & PIDS+=("$!")
done
for actor in $(seq 0 $((ROLLOUT_ACTORS-1))); do
  port=$((ROLLOUT_GYM_BASE_PORT+actor)); deadline=$((SECONDS+1200)); until curl -fsS -m5 "http://127.0.0.1:$port/health" >/dev/null; do
    ((SECONDS<deadline)) || exit 4; sleep 2; done
done
if [[ "${GYM_STARTUP_ONLY:-0}" == 1 ]]; then
  printf '{"actors":%s,"ready_epoch":%s}\n' "$ROLLOUT_ACTORS" "$(date +%s)" >"$OUTPUT_DIR/GYM_READY.json"
  exit 0 # EXIT trap verifies renderer/Gym teardown; no VLM or AH is launched.
fi

VLLM_WARM_MARKER=$VLLM_COMPILE_CACHE_ROOT/.replicas_${VLLM_REPLICAS}_ready
launch_vllm_replica() {
  local replica=$1
  local port=$((28900+replica*100))
  local pid_file="$OUTPUT_DIR/vllm_${replica}.pid"
  local compile_cache="$VLLM_COMPILE_CACHE_ROOT/replica_${replica}"
  local launcher="$REPO/verl/scripts/eval/qwen36_4query_ckpt_sweep/launch_vllm_for_ckpt.sh"
  if [[ "${COMPANION_CONDITION_MODE:-query}" == v5_native || "$VERSION" == primitive_map ]]; then
    [[ "${COMPANION_EVAL:-0}" == 1 ]] || exit 2
    launcher="$REPO/verl/scripts/eval/qwen36_4query_ckpt_sweep/launch_native_vllm.sh"
  fi
  env QUERY_CACHE="$QUERY_HANDOFF_CACHE" VLLM_LOG="$OUTPUT_DIR/logs/vllm_${replica}.log" VLLM_PID_FILE="$pid_file" \
    TORCHINDUCTOR_CACHE_DIR="$compile_cache/torchinductor" TRITON_CACHE_DIR="$compile_cache/triton" \
    XDG_CACHE_HOME="$compile_cache/xdg" \
    CHECKPOINT_DIR="$CHECKPOINT_DIR" VLLM_GPUS_OVERRIDE="$((COLLECTION_GPU_START+replica*2)),$((COLLECTION_GPU_START+replica*2+1))" VLLM_PORT_OVERRIDE="$port" \
    VLLM_MAX_MODEL_LEN_OVERRIDE=131072 ACTION_HEAD_COMPILE_CACHE_ROOT_OVERRIDE="$compile_cache" \
    VLLM_MAX_NUM_SEQS_OVERRIDE="$VLLM_MAX_NUM_SEQS" \
    VLLM_MAX_NUM_BATCHED_TOKENS_OVERRIDE="$VLLM_MAX_NUM_BATCHED_TOKENS" \
    VLLM_GPU_MEMORY_UTILIZATION_OVERRIDE="$VLLM_GPU_MEMORY_UTILIZATION" \
    bash "$launcher"
}

ports=(); VLLM_PID_FILES=()
for replica in $(seq 0 $((VLLM_REPLICAS-1))); do
  port=$((28900+replica*100)); ports+=("$port")
  VLLM_PID_FILES+=("$OUTPUT_DIR/vllm_${replica}.pid")
done

parallel_start=0
if [[ "$VLLM_STARTUP_MODE" == warm-one-parallel ]]; then
  if [[ ! -f "$VLLM_WARM_MARKER" ]]; then
    # Limit simultaneous initialization on first use. Replica caches are
    # independent; subsequent uses of this root can start every replica at once.
    launch_vllm_replica 0 >"$OUTPUT_DIR/logs/vllm_launcher_0.log" 2>&1
    mkdir -p "$VLLM_COMPILE_CACHE_ROOT"
    parallel_start=1
  fi
  launchers=()
  for replica in $(seq "$parallel_start" $((VLLM_REPLICAS-1))); do
    launch_vllm_replica "$replica" >"$OUTPUT_DIR/logs/vllm_launcher_${replica}.log" 2>&1 &
    launchers+=("$!")
  done
  failed=0
  for pid in "${launchers[@]:-}"; do wait "$pid" || failed=1; done
  [[ "$failed" == 0 ]] || { echo "parallel vLLM startup failed" >&2; exit 5; }
  touch "$VLLM_WARM_MARKER"
else
  for replica in $(seq 0 $((VLLM_REPLICAS-1))); do
    launch_vllm_replica "$replica" >"$OUTPUT_DIR/logs/vllm_launcher_${replica}.log" 2>&1
  done
fi
for pid_file in "${VLLM_PID_FILES[@]}"; do PIDS+=("$(<"$pid_file")"); done
upstreams=$(printf 'http://127.0.0.1:%s,' "${ports[@]}"); upstreams=${upstreams%,}
setsid "$GV/bin/python" "$HERE/service_router.py" --port 28899 --mode least-inflight --upstreams "$upstreams" \
  >"$OUTPUT_DIR/logs/qwen_router.log" 2>&1 & PIDS+=("$!")
deadline=$((SECONDS+120)); until curl -fsS -m5 http://127.0.0.1:28899/health >/dev/null; do ((SECONDS<deadline)) || exit 5; sleep 2; done

setsid bash -c 'while true; do date +%s; nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader,nounits | while IFS= read -r row; do case ",$1," in *",${row%%,*},"*) echo "$row";; esac; done; sleep 5; done' _ "$COLLECTION_GPU_IDS" \
  >"$OUTPUT_DIR/logs/gpu_rollout.csv" 2>&1 & PIDS+=("$!")
urls=$(seq 0 $((ROLLOUT_ACTORS-1)) | awk -v base="$ROLLOUT_GYM_BASE_PORT" 'BEGIN{s=""}{s=s (NR==1?"":",") "http://127.0.0.1:" base+$1}END{print s}')
if [[ "${COMPANION_EVAL:-0}" == 1 ]]; then
  eval_args=()
  [[ "${ALLOW_PARTIAL_CLOSEDLOOP:-0}" == 1 ]] && eval_args+=(--allow-partial)
  curl -fsS "http://127.0.0.1:$ROLLOUT_GYM_BASE_PORT/spec" >"$OUTPUT_DIR/gym_spec.json"
  /usr/local/bin/python3.12 - "$OUTPUT_DIR/gym_spec.json" "$EXPECTED_CLOSEDLOOP_SCENES" "$EXPECTED_CLOSEDLOOP_CASES" "$SUBSET_MANIFEST" <<'PY_SPEC'
import json, sys
spec = json.load(open(sys.argv[1]))
cases = json.load(open(sys.argv[4]))['cases']
assert len(spec['scenes']) == int(sys.argv[2]), 'scene count mismatch'
assert len(cases) == int(sys.argv[3]), 'selected case count mismatch'
assert len({case['episode_id'] for case in cases}) == len(cases), 'duplicate selected episode'
available = {scene: set(episodes) for scene, episodes in spec['scene_eps'].items()}
missing = [(case['scene'], case['episode_id']) for case in cases
           if case['episode_id'] not in available.get(case['scene'], set())]
assert not missing, f'selected cases absent from gym: {missing[:3]}'
PY_SPEC
  if [[ "$VERSION" == primitive_map ]]; then
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      /usr/local/bin/python3.12 "$REPO/verl/scripts/eval/qwen36_4query_ckpt_sweep/primitive_map_eval.py" \
      --subset-manifest "$SUBSET_MANIFEST" --bench-urls "$urls" \
      --vllm-url http://127.0.0.1:28899 --output "$EVAL_OUTPUT" \
      --max-turns "$MAX_TURNS" --max-tool-calls "$MAX_TOOL_CALLS" \
      --repeats "${CLOSEDLOOP_REPEATS:-8}" --trajectory-map \
      >"$OUTPUT_DIR/logs/closedloop_worker.log" 2>&1
    exit 0
  fi
  if [[ "${COMPANION_REPORT_PASS_AT_K:-1}" == 0 ]]; then
    [[ "${CLOSEDLOOP_REPEATS:-4}" == 1 ]] || { echo 'no pass@k requires one rollout per case' >&2; exit 2; }
    eval_args+=(--no-pass-at-k)
  fi
  CUDA_VISIBLE_DEVICES="$AH_GPU" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /usr/local/bin/python3.12 "$REPO/verl/scripts/eval/qwen36_4query_ckpt_sweep/closedloop_eval.py" run \
    --checkpoint-dir "$CHECKPOINT_DIR" --subset-manifest "$SUBSET_MANIFEST" --backbone "$BACKBONE" \
    --query-cache "$QUERY_HANDOFF_CACHE" --vllm-url http://127.0.0.1:28899 \
    --bench-url "http://127.0.0.1:$ROLLOUT_GYM_BASE_PORT" --bench-urls "$urls" \
    --output "$EVAL_OUTPUT" --max-turns "$MAX_TURNS" --max-tool-calls "$MAX_TOOL_CALLS" \
    --repeats "${CLOSEDLOOP_REPEATS:-4}" --limit-cases "${LIMIT_CLOSEDLOOP_CASES:-0}" \
    --max-batch-size "$AH_MAX_BATCH_SIZE" --batch-wait-ms "$AH_BATCH_WAIT_MS" \
    --action-head-as-tool-version "$VERSION" "${eval_args[@]}" \
    >"$OUTPUT_DIR/logs/closedloop_worker.log" 2>&1
  exit 0 # Existing trap reaps all owned services before CAST/open-loop can run.
fi
policy_args=()
if [[ "${AH_DEDUPLICATE_FIRST_VISUAL:-0}" == 1 ]]; then
  policy_args+=(--deduplicate-first-visual)
fi
if [[ "${ARCHIVE_QUERY_HIDDEN:-0}" == 1 || "$V6_GOLDEN_EXTRA_MAX" != 0 ]]; then
  policy_args+=(--archive-query-hidden)
fi
CUDA_VISIBLE_DEVICES="$AH_GPU" /usr/local/bin/python3.12 "$HERE/batch_rollout_worker.py" \
  --manifest "$MANIFEST" --checkpoint-dir "$CHECKPOINT_DIR" --backbone "$BACKBONE" \
  --query-cache "$QUERY_HANDOFF_CACHE" --vllm-url http://127.0.0.1:28899 --bench-urls "$urls" \
  --output-dir "$OUTPUT_DIR/rollout" --worker-index "$BOOTSTRAP_NODE_INDEX" \
  --worker-count "$BOOTSTRAP_NODE_COUNT" --worker-shard-mode strided --max-cases "$MAX_CASES" \
  --case-seed "$CASE_SEED" \
  --max-turns "$MAX_TURNS" --max-tool-calls "$MAX_TOOL_CALLS" --max-batch-size "$AH_MAX_BATCH_SIZE" \
  --batch-wait-ms "$AH_BATCH_WAIT_MS" --action-head-as-tool-version "$VERSION" \
  --case-scheduling "$CASE_SCHEDULING" \
  --max-active-scenes-per-renderer "$MAX_ACTIVE_SCENES_PER_RENDERER" \
  --max-active-cases-per-renderer "$MAX_ACTIVE_CASES_PER_RENDERER" \
  "${policy_args[@]}" \
  >"$OUTPUT_DIR/logs/rollout_worker.log" 2>&1

/usr/local/bin/python3.12 - "$OUTPUT_DIR/rollout/worker_$(printf '%02d' "$BOOTSTRAP_NODE_INDEX")/rollouts" "$VERSION" "$V6_GOLDEN_EXTRA_MAX" "${ARCHIVE_QUERY_HIDDEN:-0}" <<'PY'
import json, pathlib, sys
paths=list(pathlib.Path(sys.argv[1]).glob("*.json"))
assert paths, "no rollouts"
call_count=0
all_calls_have_trace=True
has_slow_messages=False
has_query_archive=False
for path in paths:
    payload=json.load(open(path))
    for turn in payload.get("turns",[]):
        has_slow_messages = has_slow_messages or bool(turn.get("slow_messages"))
        for call in turn.get("calls",[]):
            if call.get("name") != "run_action_head":
                continue
            call_count += 1
            trace = call.get("action_trace") or {}
            all_calls_have_trace = all_calls_have_trace and bool(trace)
            artifact = trace.get("query_artifact") or {}
            has_query_archive = has_query_archive or bool(artifact.get("archive_path"))
assert call_count and all_calls_have_trace, "missing action_trace"
assert has_slow_messages, "missing slow_messages"
if sys.argv[2] != 'v6' or sys.argv[3] != '0' or sys.argv[4] == '1':
    assert has_query_archive, "missing query archive"
print({"rollouts":len(paths),"run_action_head_calls":call_count,"trace_contract":"ok"})
PY
stop_phase
/usr/local/bin/python3.12 - "$OUTPUT_DIR/ROLLOUT_DONE.json" "$BOOTSTRAP_NODE_INDEX" "$BOOTSTRAP_NODE_COUNT" "$BOOTSTRAP_ROUND" "$COLLECTION_ITERATION" <<'PY'
import json, pathlib, sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "ok": True,
    "phase": "rollout",
    "node_index": int(sys.argv[2]),
    "node_count": int(sys.argv[3]),
    "bootstrap_round": int(sys.argv[4]),
    "collection_iteration": int(sys.argv[5]),
}) + "\n", encoding="utf-8")
PY
fi

if [[ "${ROLLOUT_ONLY:-0}" == 1 ]]; then
  exit 0
fi
PHASE=build
SOURCE_ROLLOUT_DIR=${SOURCE_ROLLOUT_DIR:-$OUTPUT_DIR/rollout/worker_$(printf '%02d' "$BOOTSTRAP_NODE_INDEX")}
[[ -d "$SOURCE_ROLLOUT_DIR/rollouts" ]] || { echo "missing source rollouts: $SOURCE_ROLLOUT_DIR" >&2; exit 6; }
# Phase B: independent renderer/visual lanes. Selection and A* stay on CPU.
if (( V6_GOLDEN_EXTRA_MAX > 0 )); then
  build_vllm_cache="$VLLM_COMPILE_CACHE_ROOT/golden_build"
  build_vllm_pid="$OUTPUT_DIR/vllm_golden_build.pid"
  env QUERY_CACHE="$QUERY_HANDOFF_CACHE" VLLM_LOG="$OUTPUT_DIR/logs/vllm_golden_build.log" \
    VLLM_PID_FILE="$build_vllm_pid" CHECKPOINT_DIR="$CHECKPOINT_DIR" \
    VLLM_GPUS_OVERRIDE="$COLLECTION_GPU_START,$((COLLECTION_GPU_START+1))" \
    VLLM_PORT_OVERRIDE=28900 VLLM_MAX_MODEL_LEN_OVERRIDE=131072 \
    ACTION_HEAD_COMPILE_CACHE_ROOT_OVERRIDE="$build_vllm_cache" \
    TORCHINDUCTOR_CACHE_DIR="$build_vllm_cache/torchinductor" \
    TRITON_CACHE_DIR="$build_vllm_cache/triton" XDG_CACHE_HOME="$build_vllm_cache/xdg" \
    VLLM_MAX_NUM_SEQS_OVERRIDE="$VLLM_MAX_NUM_SEQS" \
    VLLM_MAX_NUM_BATCHED_TOKENS_OVERRIDE="$VLLM_MAX_NUM_BATCHED_TOKENS" \
    VLLM_GPU_MEMORY_UTILIZATION_OVERRIDE="$VLLM_GPU_MEMORY_UTILIZATION" \
    bash "$REPO/verl/scripts/eval/qwen36_4query_ckpt_sweep/launch_vllm_for_ckpt.sh" \
    >"$OUTPUT_DIR/logs/vllm_golden_launcher.log" 2>&1
  VLLM_PID_FILES+=("$build_vllm_pid")
  PIDS+=("$(<"$build_vllm_pid")")
  deadline=$((SECONDS+120)); until curl -fsS -m5 http://127.0.0.1:28900/health >/dev/null; do
    ((SECONDS<deadline)) || exit 5; sleep 2
  done
fi
for lane in $(seq 0 $((BUILD_LANES-1))); do
  setsid "$GV/bin/python" "$HERE/single_gpu_renderer.py" "$RS/render_sim_dynamic.py" \
    --scenes_root /data/zerovln/SAGE-3D/sage_gs_root \
    --port "$((7060+lane))" --gpus "$((BUILD_RENDER_GPU_START+lane))" --sh_degree 3 --max_scenes_per_gpu 16 \
    >"$OUTPUT_DIR/logs/build_render_${lane}.log" 2>&1 & PIDS+=("$!")
done
for lane in $(seq 0 $((BUILD_LANES-1))); do
  deadline=$((SECONDS+1200)); until curl -fsS -m5 "http://127.0.0.1:$((7060+lane))/ping" | grep -q pong; do ((SECONDS<deadline)) || exit 6; sleep 3; done
done
for worker in $(seq 0 $((BUILD_WORKERS-1))); do
  lane=$((worker % BUILD_LANES))
  port=$((BUILD_GYM_BASE_PORT+worker)); setsid env OMP_NUM_THREADS="$GYM_CPU_THREADS" MKL_NUM_THREADS="$GYM_CPU_THREADS" \
    OPENBLAS_NUM_THREADS="$GYM_CPU_THREADS" NUMEXPR_NUM_THREADS="$GYM_CPU_THREADS" \
    DEMO_HTTP_PORT=$port SAGE_RENDER_URL="http://127.0.0.1:$((7060+lane))" SAGE_ACTION_HEAD_ASSET_SET="$SAGE_ASSET_SET" \
    bash "$REPO/vln_demo/run_action_head_curriculum_testset.sh" gym >"$OUTPUT_DIR/logs/build_gym_${worker}.log" 2>&1 & PIDS+=("$!")
done
for worker in $(seq 0 $((BUILD_WORKERS-1))); do port=$((BUILD_GYM_BASE_PORT+worker)); deadline=$((SECONDS+1200)); until curl -fsS -m5 "http://127.0.0.1:$port/health" >/dev/null; do ((SECONDS<deadline)) || exit 7; sleep 2; done; done
setsid bash -c 'while true; do date +%s; nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader,nounits | while IFS= read -r row; do case ",$1," in *",${row%%,*},"*) echo "$row";; esac; done; sleep 5; done' _ "$COLLECTION_GPU_IDS" \
  >"$OUTPUT_DIR/logs/gpu_build.csv" 2>&1 & PIDS+=("$!")
builders=()
V6_TEACHER_ARGS=()
if [[ "$VERSION" == v6 ]]; then
  V6_TEACHER_ARGS+=(--v6-teacher-mode "${V6_TEACHER_MODE:-audit_only}")
  V6_TEACHER_ARGS+=(--v6-anchor-mode "${V6_ANCHOR_MODE:-collision-only}")
  if (( V6_GOLDEN_EXTRA_MAX > 0 )); then
    V6_TEACHER_ARGS+=(--v6-golden-extra-max "$V6_GOLDEN_EXTRA_MAX" \
      --v6-golden-vllm-url http://127.0.0.1:28900 --v6-golden-vllm-model "$VLLM_MODEL")
  fi
  [[ -z "${V6_JUDGE_URL:-}" ]] || V6_TEACHER_ARGS+=(--v6-judge-url "$V6_JUDGE_URL")
  [[ -z "${V6_JUDGE_MODEL:-}" ]] || V6_TEACHER_ARGS+=(--v6-judge-model "$V6_JUDGE_MODEL")
  [[ -z "${V6_REVIEWED_CORRECTIONS:-}" ]] || V6_TEACHER_ARGS+=(--v6-reviewed-corrections "$V6_REVIEWED_CORRECTIONS")
fi
for worker in $(seq 0 $((BUILD_WORKERS-1))); do
  lane=$((worker % BUILD_LANES))
  lane_out="$OUTPUT_DIR/corrections/lanes/lane_$(printf '%02d' "$worker")"
  global_shard_index=$((BOOTSTRAP_NODE_INDEX + worker * BOOTSTRAP_NODE_COUNT))
  global_shard_count=$((BOOTSTRAP_NODE_COUNT * BUILD_WORKERS))
  setsid env CUDA_VISIBLE_DEVICES=$((VISION_GPU_START+lane)) /usr/local/bin/python3.12 "$REPO/verl/sft/collect_noncollision_option_corrections.py" \
    --source-recovery-dir "$SOURCE_ROLLOUT_DIR" --base-manifest "$MANIFEST" \
    --checkpoint-dir "$CHECKPOINT_DIR" --backbone "$BACKBONE" --query-cache "$QUERY_HANDOFF_CACHE" \
    --bench-url "http://127.0.0.1:$((BUILD_GYM_BASE_PORT+worker))" --render-url "http://127.0.0.1:$((7060+lane))" \
    --output-dir "$lane_out" --max-cases "$MAX_CASES" --max-correction-anchors "$MAX_CORRECTION_ANCHORS" \
    --case-shard-index "$global_shard_index" --case-shard-count "$global_shard_count" --action-head-as-tool-version "$VERSION" \
    --bootstrap-round "$BOOTSTRAP_ROUND" \
    --collection-iteration "$COLLECTION_ITERATION" \
    --sage-split "$SAGE_DATASET_SPLIT" \
    --npz-compress-level "$NPZ_COMPRESS_LEVEL" \
    --seed "$CASE_SEED" \
    "${V6_TEACHER_ARGS[@]}" \
    >"$OUTPUT_DIR/logs/build_lane_${worker}.log" 2>&1 & builders+=("$!"); PIDS+=("$!")
done
failed=0; for pid in "${builders[@]}"; do wait "$pid" || failed=1; done; [[ $failed == 0 ]]
/usr/local/bin/python3.12 "$HERE/merge_v3_corrections.py" --output-dir "$OUTPUT_DIR/corrections" \
  --lanes "$BUILD_WORKERS" --max-correction-anchors "$MAX_CORRECTION_ANCHORS" \
  --bootstrap-round "$BOOTSTRAP_ROUND"
/usr/local/bin/python3.12 - "$OUTPUT_DIR/DONE" "$BOOTSTRAP_NODE_INDEX" "$BOOTSTRAP_NODE_COUNT" <<'PY'
import json, pathlib, sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "ok": True, "node_index": int(sys.argv[2]), "node_count": int(sys.argv[3])
}) + "\n", encoding="utf-8")
PY
