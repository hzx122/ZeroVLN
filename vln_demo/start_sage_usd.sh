#!/usr/bin/env bash
# 一键起 Sage USD Physical：官方 collision USD 生成的 2D 物理 mask + nurec 3DGS。
# 与 start_sage_visual.sh 平行；同样占用 render 7038 + console 5000。
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
GV=/opt/navigation-runtime/venv
RS=/workspace/agentic_sim/repo/ABot-Navigation/render_server
TGT="$HERE/render_libs"
NAV=/opt/navigation-runtime/VLMEvalKit/3rdparty/RynnBrain/navigation
SRC="$HERE/src"
SAGE_ROOT=${SAGE_ROOT:-/data/zerovln/SAGE-3D}
SCENES_ROOT=${SAGE_GS_ROOT:-$SAGE_ROOT/sage_gs_root}
MASK_ROOT=${SAGE_USD_COLLISION_MASK_ROOT:-$SAGE_ROOT/sage_usd_rooms_masks}
VISUAL_MASK_ROOT=${SAGE_FURNITURE_MASK_ROOT:-$SAGE_ROOT/sage_collision_masks}
USD_MODE=${SAGE_USD_COLLISION_MODE:-usd_rooms}
SAGE_DEMO_SPLIT=${SAGE_SPLIT:-scene}
SAGE_DEMO_SCENES=${SAGE_SCENES:-}
SAGE_EPS=${SAGE_EPISODES_PER_SCENE:-5}
RENDER_GPU=${SAGE_RENDER_GPU:-1}
RGB_CLEANUP=${SAGE_RGB_OUTSIDE_CLEANUP:-0}
BOUNDARY_CLIP=${SAGE_USD_BOUNDARY_CLIP:-0}
BOUNDARY_BLUE_ONLY=${SAGE_USD_BOUNDARY_BLUE_ONLY:-0}
STATIC_BOUNDARY_ROOT=${SAGE_USD_STATIC_BOUNDARY_MASK_ROOT:-}
STATIC_BOUNDARY_MARGIN=${SAGE_USD_STATIC_BOUNDARY_MARGIN_M:-0.30}
STATIC_BLUE_MARGIN=${SAGE_USD_STATIC_BLUE_MARGIN_M:-0.10}
BACKGROUND_GRAY=${SAGE_RENDER_BACKGROUND_GRAY:-0.12}
LOG_ROOT=${SAGE_USD_LOG_ROOT:-$REPO/tmp/sage_usd}
RENDER_ENTRY="$REPO/tmp/sage_usd_render_server.py"

mkdir -p "$LOG_ROOT" "$REPO/tmp"

if [[ "$RGB_CLEANUP" == 1 ]] && ! grep -q "suppress_blue_gaussians" "$RS/render_sim_dynamic.py"; then
  echo "[warn] render server 缺少 RGB cleanup 可选扩展，异常帧将 fail-open 返回原图" >&2
  echo "       git -C $(dirname "$RS") apply $HERE/patches/sage_rgb_outside_cleanup_render_server.patch" >&2
fi

if [[ "$USD_MODE" != "usd" && "$USD_MODE" != "usd_visual" && "$USD_MODE" != "usd_rooms" ]]; then
  echo "[error] SAGE_USD_COLLISION_MODE 只支持 usd / usd_visual / usd_rooms：$USD_MODE" >&2
  exit 2
fi

if [[ "$USD_MODE" == "usd_rooms" && -z "$STATIC_BOUNDARY_ROOT" ]]; then
  STATIC_BOUNDARY_ROOT="$MASK_ROOT"
fi

if [[ -n "$STATIC_BOUNDARY_ROOT" ]] \
   && ! grep -q "_apply_static_boundary_mask" "$RS/render_sim_dynamic.py"; then
  echo "[error] render server 缺少 USD 静态边界过滤；拒绝静默启动假蓝版本" >&2
  echo "先运行：git -C $(dirname "$RS") apply $HERE/patches/sage_usd_static_boundary_render_server.patch" >&2
  exit 2
fi

# Refuse a silent per-scene fallback to official occupancy.  A production USD
# launch is useful only when every selected scene has a generated mask.
missing=$(
  SAGE_ROOT="$SAGE_ROOT" SAGE_SPLIT="$SAGE_DEMO_SPLIT" \
  SAGE_SCENES="$SAGE_DEMO_SCENES" MASK_ROOT="$MASK_ROOT" USD_MODE="$USD_MODE" PYTHONPATH="$SRC" \
  "$GV/bin/python" - <<'PY'
import os
import numpy as np
from sage.datasets import SageSet
ds = SageSet(os.environ["SAGE_SPLIT"], root=os.environ["SAGE_ROOT"])
raw = [x.strip() for x in os.environ.get("SAGE_SCENES", "").split(",") if x.strip()]
scenes = [x if x.startswith("sage-") else ds.qualify(x) for x in raw] or ds.list_scenes()
missing = []
for scene in scenes:
    bare = ds.bare(scene)
    path = os.path.join(os.environ["MASK_ROOT"], bare + ".npz")
    try:
        with np.load(path, allow_pickle=False) as data:
            version = int(np.asarray(data["version"]).reshape(()))
            valid = version >= 2
            if os.environ["USD_MODE"] == "usd_rooms":
                valid = (version >= 4 and all(key in data.files for key in
                         ("rooms_free", "rooms_candidate", "door_opening",
                          "render_domain", "exterior_door_opening")))
    except Exception:
        valid = False
    if not valid:
        missing.append(bare)
print(",".join(missing))
PY
)
if [[ -n "$missing" ]]; then
  echo "[error] USD collision mask 缺失或版本过旧：$missing" >&2
  echo "先运行：cd $SRC && PYTHONPATH=.:$SAGE_ROOT/usd_core_py312 $GV/bin/python -m sage.build_usd_collision --all-splits --out $MASK_ROOT --workers 8 --skip-existing" >&2
  exit 2
fi

# The guarded mode needs the independent nurec/PLY physical artifact. Raw
# `usd` remains available for exact reproduction and skips this check.
if [[ "$USD_MODE" == "usd_visual" ]]; then
  visual_missing=$(
    SAGE_ROOT="$SAGE_ROOT" SAGE_SPLIT="$SAGE_DEMO_SPLIT" \
    SAGE_SCENES="$SAGE_DEMO_SCENES" MASK_ROOT="$VISUAL_MASK_ROOT" PYTHONPATH="$SRC" \
    "$GV/bin/python" - <<'PY'
import os
import numpy as np
from sage.datasets import SageSet
ds = SageSet(os.environ["SAGE_SPLIT"], root=os.environ["SAGE_ROOT"])
raw = [x.strip() for x in os.environ.get("SAGE_SCENES", "").split(",") if x.strip()]
scenes = [x if x.startswith("sage-") else ds.qualify(x) for x in raw] or ds.list_scenes()
missing = []
for scene in scenes:
    path = os.path.join(os.environ["MASK_ROOT"], ds.bare(scene) + ".npz")
    try:
        with np.load(path, allow_pickle=False) as data:
            valid = (int(np.asarray(data["version"]).reshape(())) >= 3
                     and "visual_free" in data.files)
    except Exception:
        valid = False
    if not valid:
        missing.append(ds.bare(scene))
print(",".join(missing))
PY
  )
  if [[ -n "$visual_missing" ]]; then
    echo "[error] USD 视觉校验 mask 缺失或版本过旧：$visual_missing" >&2
    echo "先运行：cd $SRC && PYTHONPATH=. $GV/bin/python -m sage.build_geom_collision --split $SAGE_DEMO_SPLIT" >&2
    exit 2
  fi
fi

export LD_LIBRARY_PATH="/opt/accl-p:/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"
export SAGE_USD_STATIC_BOUNDARY_MASK_ROOT="$STATIC_BOUNDARY_ROOT"
export SAGE_USD_STATIC_BOUNDARY_MARGIN_M="$STATIC_BOUNDARY_MARGIN"
export SAGE_USD_STATIC_BLUE_MARGIN_M="$STATIC_BLUE_MARGIN"
export SAGE_RENDER_BACKGROUND_GRAY="$BACKGROUND_GRAY"

# Same port-switch semantics as Visual v3.  Use a distinct argv name so an
# unrelated broad `pkill -f render_sim_dynamic.py` cannot reap this server.
pkill -9 -f "sage_usd_render_server.py.*--port 7038" 2>/dev/null || true
pkill -9 -f "vln_model.py.*--port 5000" 2>/dev/null || true
for port in 5000 7038; do fuser -k "$port/tcp" 2>/dev/null || true; done
sleep 2

ln -sfn "$RS/render_sim_dynamic.py" "$RENDER_ENTRY"
cd "$RS"
PYTHONPATH="$TGT:$RS/submodules/diff-plane-rasterization:$RS/submodules/simple-knn:$RS" \
  setsid "$GV/bin/python" "$RENDER_ENTRY" \
  --scenes_root "$SCENES_ROOT" --port 7038 --gpus "$RENDER_GPU" --sh_degree 3 \
  --max_scenes_per_gpu 2 > "$LOG_ROOT/render_7038.log" 2>&1 < /dev/null &
echo "[start] render pid=$!"

ready=0
for _ in $(seq 1 20); do
  sleep 3
  if curl -s -m 3 http://127.0.0.1:7038/ping 2>/dev/null | grep -q pong; then
    ready=1
    break
  fi
done
if [[ "$ready" != 1 ]]; then
  echo "[error] render 7038 未就绪；日志：$LOG_ROOT/render_7038.log" >&2
  tail -n 80 "$LOG_ROOT/render_7038.log" >&2 || true
  exit 3
fi

cd "$NAV"
export PATH="$GV/bin:$PATH"
export LD_LIBRARY_PATH="$GV/lib:${LD_LIBRARY_PATH:-}"
export EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export HABITAT_SIM_GPU_DEVICE_ID=-1 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet PYTHONUNBUFFERED=1
export DEMO_BACKEND=sage SAGE_COLLISION="$USD_MODE" SAGE_SPLIT="$SAGE_DEMO_SPLIT"
export SAGE_SCENES="$SAGE_DEMO_SCENES" SAGE_EPISODES_PER_SCENE="$SAGE_EPS"
export SAGE_RENDER_URL=http://127.0.0.1:7038 SAGE_BLOCK_FURNITURE=0
export SAGE_FURNITURE_COLLISION=none SAGE_ROBOT_RADIUS=0.07
export SAGE_CAMERA_HEIGHT="${SAGE_CAMERA_HEIGHT:-1.2}" SAGE_HFOV="${SAGE_HFOV:-105.3}"
export SAGE_USD_COLLISION_MASK_ROOT="$MASK_ROOT"
export SAGE_FURNITURE_MASK_ROOT="$VISUAL_MASK_ROOT"
export SAGE_RGB_OUTSIDE_CLEANUP="$RGB_CLEANUP"
export SAGE_USD_BOUNDARY_CLIP="$BOUNDARY_CLIP"
export SAGE_USD_BOUNDARY_BLUE_ONLY="$BOUNDARY_BLUE_ONLY"
setsid "$GV/bin/python" "$SRC/vln_model.py" --backend sage --port 5000 \
  --sage_split "$SAGE_DEMO_SPLIT" --sage_scenes "$SAGE_DEMO_SCENES" \
  --sage_episodes_per_scene "$SAGE_EPS" \
  > "$LOG_ROOT/console_5000.log" 2>&1 < /dev/null &
echo "[start] console pid=$!"
echo "[start] USD Physical: mode=$USD_MODE split=$SAGE_DEMO_SPLIT scenes=${SAGE_DEMO_SCENES:-all} boundary_clip=$SAGE_USD_BOUNDARY_CLIP rgb_cleanup=$SAGE_RGB_OUTSIDE_CLEANUP"
echo "[start] http://127.0.0.1:5000"
