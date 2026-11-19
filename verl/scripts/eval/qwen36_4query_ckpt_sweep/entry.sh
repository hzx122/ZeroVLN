#!/usr/bin/env bash
# One-node companion job: shared bootstrap services plus open/closed loop.
set -euo pipefail

ACTION_HEAD_AS_TOOL_VERSION=v1
while (($#)); do
  case "$1" in
    --action-head-as-tool-version) ACTION_HEAD_AS_TOOL_VERSION=${2:?}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$ACTION_HEAD_AS_TOOL_VERSION" in v1|v2|v3|v4|v4.[1-9]|v4.10|v4.11|v6) ;; *) exit 2;; esac
export EVAL_LAYOUT=${EVAL_LAYOUT:-bootstrap}
[[ "$EVAL_LAYOUT" == serial || "$EVAL_LAYOUT" == bootstrap ]] || exit 2

REPO=/workspace/ZeroVLN
HERE=$REPO/verl/scripts/eval/qwen36_4query_ckpt_sweep
GV=/opt/navigation-runtime/venv
RS=/workspace/agentic_sim/repo/ABot-Navigation/render_server
RENDER_LIBS=$REPO/vln_demo/render_libs
: "${EXPR_NAME:?EXPR_NAME is required}"
: "${UPSTREAM_JOB_ID:?UPSTREAM_JOB_ID is required}"
: "${CHECKPOINT_ROOT:?CHECKPOINT_ROOT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${SFT_MANIFEST:?SFT_MANIFEST is required}"
: "${SUBSET_MANIFEST:?SUBSET_MANIFEST is required}"
STEP_START=${STEP_START:-200}
STEP_STOP=${STEP_STOP:-5000}
STEP_INTERVAL=${STEP_INTERVAL:-200}
STEP_LIST=${STEP_LIST:-}
FINAL_STEP=${FINAL_STEP:-0}
CHECKPOINT_INDEX=${CHECKPOINT_INDEX:-}
EXPECTED_CHECKPOINT_COUNT=${EXPECTED_CHECKPOINT_COUNT:-0}
MAX_OPENLOOP_ANCHORS=${MAX_OPENLOOP_ANCHORS:-0}
LIMIT_CLOSEDLOOP_CASES=${LIMIT_CLOSEDLOOP_CASES:-0}
CLOSEDLOOP_REPEATS=${CLOSEDLOOP_REPEATS:-4}
ALLOW_PARTIAL_CLOSEDLOOP=${ALLOW_PARTIAL_CLOSEDLOOP:-0}
CLOSEDLOOP_ONLY=${CLOSEDLOOP_ONLY:-0}
MAX_TURNS=${MAX_TURNS:-30}
MAX_TOOL_CALLS=${MAX_TOOL_CALLS:-30}
CLOSEDLOOP_ASSET_SET=${CLOSEDLOOP_ASSET_SET:-test}
EXPECTED_CLOSEDLOOP_SCENES=${EXPECTED_CLOSEDLOOP_SCENES:-29}
EXPECTED_CLOSEDLOOP_CASES=${EXPECTED_CLOSEDLOOP_CASES:-262}
WAIT_FOR_SFT_MANIFEST=${WAIT_FOR_SFT_MANIFEST:-0}
SFT_MANIFEST_WAIT_SECONDS=${SFT_MANIFEST_WAIT_SECONDS:-21600}

if [[ ! -f "$SFT_MANIFEST" ]]; then
  [[ "$WAIT_FOR_SFT_MANIFEST" == 1 ]] || {
    echo "missing SFT manifest: $SFT_MANIFEST" >&2
    exit 2
  }
  deadline=$((SECONDS + SFT_MANIFEST_WAIT_SECONDS))
  echo "[entry] waiting for SFT manifest: $SFT_MANIFEST"
  until [[ -f "$SFT_MANIFEST" ]]; do
    (( SECONDS < deadline )) || {
      echo "timed out waiting for SFT manifest: $SFT_MANIFEST" >&2
      exit 2
    }
    sleep 30
  done
  echo "[entry] SFT manifest ready: $SFT_MANIFEST"
fi

mkdir -p "$OUTPUT_ROOT/logs"
PIDS=()
cleanup() {
  code=$?
  trap - EXIT INT TERM
  for pid in "${PIDS[@]:-}"; do
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  done
  find "$OUTPUT_ROOT" -name vllm_28899.pid -type f -print0 2>/dev/null \
    | while IFS= read -r -d '' file; do
        pid=$(cat "$file" 2>/dev/null || true)
        [[ "$pid" =~ ^[0-9]+$ ]] && { kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true; }
      done
  wait 2>/dev/null || true
  echo "[entry] cleanup rc=$code time=$(date -Is)"
  exit "$code"
}
trap cleanup EXIT INT TERM

export PYTHONUNBUFFERED=1
export PPU_SDK=/usr/local/PPU_SDK PPU_HOME=/usr/local/PPU_SDK
export CUDA_PATH=/usr/local/PPU_SDK/CUDA_SDK CUDA_HOME=$CUDA_PATH
export SAGE_COLLISION=usd_rooms SAGE_FURNITURE_COLLISION=none
export SAGE_ROBOT_RADIUS=0.07 SAGE_CAMERA_HEIGHT=1.2 SAGE_HFOV=115
export SAGE_EPISODES_PER_SCENE=0
export SAGE_USD_COLLISION_MASK_ROOT=/data/zerovln/SAGE-3D/sage_usd_rooms_masks
export SAGE_USD_STATIC_BOUNDARY_MASK_ROOT=$SAGE_USD_COLLISION_MASK_ROOT
export PYTHONPATH="$RENDER_LIBS:$RS/submodules/diff-plane-rasterization:$RS/submodules/simple-knn:$RS:$REPO:$REPO/verl:$REPO/vln_demo/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/opt/accl-p:/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/lib:$GV/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"

if [[ "$CLOSEDLOOP_ASSET_SET" == test ]]; then
  if [[ -f "$SUBSET_MANIFEST" ]]; then
    echo "[entry] reuse immutable fixed subset: $SUBSET_MANIFEST"
  else
    /usr/local/bin/python3.12 "$HERE/build_fixed_subset.py" \
      --output "$SUBSET_MANIFEST" \
      --source-manifest /data/zerovln/action_head/curriculum/testset/Trajectory_demo/manifest.json \
      --seed 20260828
  fi
else
  test -f "$SUBSET_MANIFEST" || { echo "missing closed-loop subset: $SUBSET_MANIFEST" >&2; exit 2; }
fi

if [[ "$EVAL_LAYOUT" == serial ]]; then
setsid "$GV/bin/python" "$REPO/tmp/sage_usd_render_server.py" \
  --scenes_root /data/zerovln/SAGE-3D/sage_gs_root \
  --port 7038 --gpus 10 --sh_degree 3 --max_scenes_per_gpu 2 \
  > "$OUTPUT_ROOT/logs/render_7038.log" 2>&1 &
PIDS+=("$!")
deadline=$((SECONDS + 1200))
until curl -fsS -m 5 http://127.0.0.1:7038/ping | grep -q pong; do
  (( SECONDS < deadline )) || { tail -n 200 "$OUTPUT_ROOT/logs/render_7038.log"; exit 3; }
  sleep 3
done

setsid env DEMO_HTTP_PORT=8938 SAGE_RENDER_URL=http://127.0.0.1:7038 \
  SAGE_ACTION_HEAD_ASSET_SET="$CLOSEDLOOP_ASSET_SET" \
  bash "$REPO/vln_demo/run_action_head_curriculum_testset.sh" gym \
  > "$OUTPUT_ROOT/logs/gym_8938.log" 2>&1 &
PIDS+=("$!")
deadline=$((SECONDS + 1200))
until curl -fsS -m 5 http://127.0.0.1:8938/health >/dev/null; do
  (( SECONDS < deadline )) || { tail -n 200 "$OUTPUT_ROOT/logs/gym_8938.log"; exit 4; }
  sleep 3
done
curl -fsS http://127.0.0.1:8938/spec > "$OUTPUT_ROOT/gym_spec.json"
/usr/local/bin/python3.12 - "$OUTPUT_ROOT/gym_spec.json" \
  "$EXPECTED_CLOSEDLOOP_SCENES" "$EXPECTED_CLOSEDLOOP_CASES" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
episodes=sum(len(items) for items in (x.get("scene_eps") or {}).values())
expected_scenes, expected_cases = map(int, sys.argv[2:4])
if len(x.get("scenes") or []) != expected_scenes or episodes != expected_cases:
    raise SystemExit(f"gym /spec mismatch: scenes={len(x.get('scenes') or [])} episodes={episodes}")
PY

fi # serial services; bootstrap owns services per checkpoint.

if [[ -n "$CHECKPOINT_INDEX" ]]; then
  WATCH_STEP_ARGS=(
    --checkpoint-index "$CHECKPOINT_INDEX"
    --expected-checkpoint-count "$EXPECTED_CHECKPOINT_COUNT"
  )
elif [[ -n "$STEP_LIST" ]]; then
  IFS=',' read -r -a EXPLICIT_STEPS <<< "$STEP_LIST"
  FIRST_STEP=${EXPLICIT_STEPS[0]}
  WATCH_STEP_ARGS=(
    --step-start "$FIRST_STEP"
    --step-stop "$FIRST_STEP"
    --step-interval 1
  )
  for step in "${EXPLICIT_STEPS[@]:1}"; do
    WATCH_STEP_ARGS+=(--include-step "$step")
  done
else
  WATCH_STEP_ARGS=(--step-start "$STEP_START" --step-stop "$STEP_STOP" --step-interval "$STEP_INTERVAL")
  if [[ "$FINAL_STEP" -gt 0 ]]; then
    WATCH_STEP_ARGS+=(--include-step "$FINAL_STEP")
  fi
fi
if [[ "$ALLOW_PARTIAL_CLOSEDLOOP" == 1 ]]; then
  WATCH_STEP_ARGS+=(--allow-partial-closedloop)
fi
if [[ "$CLOSEDLOOP_ONLY" == 1 ]]; then
  WATCH_STEP_ARGS+=(--closedloop-only)
fi

/usr/local/bin/python3.12 "$HERE/watch_checkpoints.py" \
  --upstream-job-id "$UPSTREAM_JOB_ID" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --sft-manifest "$SFT_MANIFEST" \
  --subset-manifest "$SUBSET_MANIFEST" \
  --slow-model /opt/external/user-e7a01947/Model/Qwen3.6-35B-A3B \
  --backbone "${ACTION_HEAD_VISUAL_BACKBONE:-/opt/external/user-e7a01947/Model/Qwen3.5-4B}" \
  --bench-url http://127.0.0.1:8938 \
  --max-openloop-anchors "$MAX_OPENLOOP_ANCHORS" \
  --limit-closedloop-cases "$LIMIT_CLOSEDLOOP_CASES" \
  --closedloop-repeats "$CLOSEDLOOP_REPEATS" \
  --max-turns "$MAX_TURNS" \
  --max-tool-calls "$MAX_TOOL_CALLS" \
  --action-head-as-tool-version "$ACTION_HEAD_AS_TOOL_VERSION" \
  "${WATCH_STEP_ARGS[@]}"
