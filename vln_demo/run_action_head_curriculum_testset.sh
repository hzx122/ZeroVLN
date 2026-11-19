#!/usr/bin/env bash
# One physical-contract entry point for the fixed test set and audited train candidates.
set -euo pipefail

# PPU's runtime dlopens libhggcrt1.so by alias; ldconfig only indexes its
# versioned SONAME. Non-login console launches need the SDK search path too.
if [[ -d /usr/local/PPU_SDK/lib ]]; then
  export LD_LIBRARY_PATH="/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/.." && pwd)
SRC="$HERE/src"
PY=${VENV:-/opt/navigation-runtime/venv}/bin/python
ROOT=/data/zerovln/action_head/curriculum
ASSET_SET=${SAGE_ACTION_HEAD_ASSET_SET:-test}
case "$ASSET_SET" in
  official_sage_bench_v1)
    : "${SAGE_BENCH_PROTOCOL_ROOT:?official protocol root required}"
    DATASET="$SAGE_BENCH_PROTOCOL_ROOT/dataset"
    MANIFEST="$DATASET/manifest.json"
    SPLIT=official_sage_bench_v1
    AUDIT_OUTPUT="$DATASET/usd_rooms_replay_audit.json"
    ;;
  test)
    DATASET="$ROOT/testset/Trajectory_demo"
    MANIFEST="$DATASET/manifest.json"
    SPLIT=action_head_curriculum
    AUDIT_OUTPUT="$ROOT/testset/usd_rooms_replay_audit.json"
    ;;
  p0_pilot)
    DATASET="$ROOT/pilot_v1/p0/Trajectory_demo"
    MANIFEST="$ROOT/pilot_v1/p0/manifest.json"
    SPLIT=action_head_train_p0_pilot
    AUDIT_OUTPUT="$ROOT/pilot_v1/p0/usd_rooms_replay_audit.json"
    ;;
  train_p0_unified)
    DATASET="$ROOT/train_v1/p0_unified/Trajectory_demo"
    MANIFEST="$ROOT/train_v1/p0_unified/manifest.json"
    SPLIT=action_head_train_p0_unified
    AUDIT_OUTPUT="$ROOT/train_v1/p0_unified/usd_rooms_replay_audit.json"
    ;;
  sample50)
    DATASET="/data/zerovln/SAGE-3D/official_gvln_action_head_p0_sample50_20260907"
    MANIFEST="$DATASET/manifest.json"
    SPLIT=official_gvln_action_head_p0_sample50
    AUDIT_OUTPUT="$DATASET/usd_rooms_replay_audit.json"
    ;;
  official_gvln_v3test100)
    DATASET="/data/zerovln/SAGE-3D/official_gvln_action_head_p0_v3test100_20260908"
    MANIFEST="$DATASET/manifest.json"
    SPLIT=official_gvln_action_head_p0_v3test100
    AUDIT_OUTPUT="$DATASET/usd_rooms_replay_audit.json"
    ;;
  official_gvln_v3full95val)
    DATASET="/data/zerovln/artifacts/sage-fullval-clean-20260919/dataset"
    MANIFEST="$DATASET/manifest.json"
    SPLIT=official_gvln_action_head_p0_v3full95val
    AUDIT_OUTPUT="$DATASET/usd_rooms_replay_audit.json"
    ;;
  official_gvln_v3full827train)
    DATASET="/data/zerovln/SAGE-3D/official_gvln_action_head_p0_v3full_827train_20260907"
    MANIFEST="$DATASET/manifest.json"
    SPLIT=official_gvln_action_head_p0_v3full827train
    AUDIT_OUTPUT="$DATASET/usd_rooms_replay_audit.json"
    ;;
  *)
    echo "unknown SAGE_ACTION_HEAD_ASSET_SET=$ASSET_SET; use test, p0_pilot, train_p0_unified, sample50, official_gvln_v3test100, or official_gvln_v3full827train" >&2
    exit 2
    ;;
esac
PORT=${VLN_DEMO_PORT:-5001}
MODE=${1:-console}
shift || true

physical_contract() {
  : "${SAGE_COLLISION:=usd_rooms}"
  : "${SAGE_FURNITURE_COLLISION:=none}"
  : "${SAGE_ROBOT_RADIUS:=0.07}"
  : "${SAGE_CAMERA_HEIGHT:=1.2}"
  : "${SAGE_HFOV:=115}"
  : "${SAGE_EPISODES_PER_SCENE:=0}"
  : "${SAGE_USD_COLLISION_MASK_ROOT:=/data/zerovln/SAGE-3D/sage_usd_rooms_masks}"
  export SAGE_COLLISION SAGE_FURNITURE_COLLISION SAGE_ROBOT_RADIUS
  export SAGE_CAMERA_HEIGHT SAGE_HFOV SAGE_USD_COLLISION_MASK_ROOT
  export SAGE_EPISODES_PER_SCENE
  [[ "$SAGE_COLLISION" == "usd_rooms" ]] || { echo "refuse collision=$SAGE_COLLISION; curriculum golden requires usd_rooms" >&2; exit 2; }
  [[ "$SAGE_FURNITURE_COLLISION" == "none" ]] || { echo "refuse furniture=$SAGE_FURNITURE_COLLISION; curriculum golden requires none" >&2; exit 2; }
  [[ "$SAGE_ROBOT_RADIUS" == "0.07" ]] || { echo "refuse robot radius=$SAGE_ROBOT_RADIUS; curriculum golden requires 0.07" >&2; exit 2; }
  [[ "$SAGE_HFOV" == "115" ]] || { echo "refuse HFOV=$SAGE_HFOV; curriculum observations require 115" >&2; exit 2; }
  [[ "$SAGE_EPISODES_PER_SCENE" == "0" ]] || { echo "refuse episodes_per_scene=$SAGE_EPISODES_PER_SCENE; curriculum console/eval must load all cases" >&2; exit 2; }
}

case "$MODE" in
  build)
    "$PY" "$HERE/assemble_action_head_curriculum_testset.py" --output "$DATASET" "$@"
    ;;
  inspect)
    test -f "$MANIFEST" || { echo "missing $MANIFEST; run: $0 build" >&2; exit 2; }
    "$PY" - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
keys = ("scene_count", "trajectory_count", "task_count", "stage_counts", "source_root")
print(json.dumps({k: m[k] for k in keys if k in m}, ensure_ascii=False, indent=2))
PY
    ;;
  audit)
    test -f "$MANIFEST" || { echo "missing $MANIFEST; run: $0 build" >&2; exit 2; }
    physical_contract
    cd "$SRC"
    PYTHONPATH=. "$PY" "$HERE/audit_action_head_curriculum.py" \
      --split "$SPLIT" --output "$AUDIT_OUTPUT" "$@"
    ;;
  console)
    test -f "$MANIFEST" || { echo "missing $MANIFEST; run: $0 build" >&2; exit 2; }
    physical_contract
    cd "$SRC"
    export PYTHONPATH=".:$REPO/verl${PYTHONPATH:+:$PYTHONPATH}"
    export DEMO_BACKEND=sage SAGE_SPLIT="$SPLIT"
    export SAGE_RENDER_URL=${SAGE_RENDER_URL:-http://127.0.0.1:7038}
    # Device visibility must be fixed before Python imports torch.  Setting it
    # inside the asynchronous loader is too late on PPU/CUDA runtimes and can
    # make the first Action Head forward pass terminate the web process.
    ACTION_HEAD_CONFIG=${ACTION_HEAD_CONFIG:-$HERE/config/action_head_runtime.json}
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -f "$ACTION_HEAD_CONFIG" ]]; then
      CUDA_VISIBLE_DEVICES=$("$PY" -c \
        'import json,sys; print((json.load(open(sys.argv[1])).get("defaults") or {}).get("visible_devices", ""))' \
        "$ACTION_HEAD_CONFIG")
      export CUDA_VISIBLE_DEVICES
    fi
    exec "$PY" "$SRC/vln_model.py" --backend sage --sage_split "$SPLIT" --port "$PORT" "$@"
    ;;
  gym)
    test -f "$MANIFEST" || { echo "missing $MANIFEST; run: $0 build" >&2; exit 2; }
    physical_contract
    export DEMO_BACKEND=sage DEMO_SPLIT="$SPLIT" SAGE_SPLIT="$SPLIT"
    export SAGE_RENDER_URL=${SAGE_RENDER_URL:-http://127.0.0.1:7038}
    exec "$HERE/vln_gym_server.sh" "$@"
    ;;
  *)
    echo "usage: $0 {build|inspect|audit|console|gym} [args...]" >&2
    exit 2
    ;;
esac
