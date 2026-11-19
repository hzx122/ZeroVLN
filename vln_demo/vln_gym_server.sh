#!/usr/bin/env bash
# vln_gym HTTP bench 启动器（单 session，纯 CPU sim，不含 vLLM）。
# 复用 run_demo_ext.sh 的 env 设置（gordon venv + mesa 软渲染四件套 +
# RynnBrain NAV 只读 import；在独立 GYM_WORK_DIR 内让 hydra 解析 config）。
# policy 外置在 verl，bench 只暴露 gym HTTP 接口。
#
# 用法：
#   bash vln_gym_server.sh                      # 默认 2azQ1b91cZZ/ep11/8899/r2r
#   DEMO_HTTP_PORT=8901 DEMO_SCENE=... bash vln_gym_server.sh
#   GYM_WORK_DIR=/path/to/workspace/gym/8901 DEMO_HTTP_PORT=8901 bash vln_gym_server.sh
#   DEMO_BACKEND=hsim  bash vln_gym_server.sh   # 切后端: r2r/rxr/rs3d/hsim/abotn/mall
#     rs3d  : 需先起 3DGS render server :7036 (RS3D_RENDER_URL 可覆盖)
#     abotn : 需先起 3DGS render server :7037 (ABOTN_RENDER_URL)，ABOTN_SET=poi|in|out
#     hsim  : HSIM_SET=partnr|hab，资产在 AGENTIC_SIM_ROOT/benchmark
#     rxr   : 需 RxR json.gz (RXR_DATA_PATH)，复用 R2R 11 场景
#     mall  : 需先起 3DGS render server (MALL_RENDER_URL，默认 7039；--sh_degree 0，
#             scenes_root 指到 zekang 的 ppu_sim/3drs_gs_root)，资产目录 MALL_ASSETS_DIR
# 迁移到新机器：改下面 VENV/NAV/DATA_ROOT 三个路径变量（同 run_demo_ext.sh）。
set -uo pipefail

# ---- 路径(迁移到新机器时改这里) ----
VENV=/opt/navigation-runtime/venv
NAV=/opt/navigation-runtime/VLMEvalKit/3rdparty/RynnBrain/navigation
DATA_ROOT=/data/shared/groups/mfm/common/anno/vlmevalkit/RynnBrain/data
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)   # vln_demo 根

PORT=${DEMO_HTTP_PORT:-8899}
SCENE=${DEMO_SCENE:-2azQ1b91cZZ}
EPISODE_ID=${DEMO_EPISODE_ID:-11}
SPLIT=${DEMO_SPLIT:-val_unseen}
MAX_STEPS=${DEMO_MAX_STEPS:-500}
BACKEND=${DEMO_BACKEND:-r2r}    # r2r(默认) / rxr / rs3d / hsim / abotn
GYM_WORK_DIR=${GYM_WORK_DIR:-$HERE/../tmp/vln_gym/$PORT}

echo "[vln_gym_server] backend=$BACKEND port=$PORT scene=$SCENE ep=$EPISODE_ID split=$SPLIT"
echo "[vln_gym_server] VENV=$VENV"
echo "[vln_gym_server] NAV=$NAV"
echo "[vln_gym_server] DATA_ROOT=$DATA_ROOT"

source "$VENV/bin/activate"

# ---- 软渲染四件套 + libEGL/libOpenGL 路径 ----
export LD_LIBRARY_PATH="$HERE/lib:$VENV/lib:${LD_LIBRARY_PATH:-}"
export EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 \
       MESA_LOADER_DRIVER_OVERRIDE=llvmpipe GALLIUM_DRIVER=llvmpipe \
       HABITAT_SIM_GPU_DEVICE_ID=-1 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

# ---- RynnBrain 仿真参数（语言/转角/HFOV/sliding 同 vln_demo 默认）----
export RYNNVLN_LANGUAGE=en RYNNVLN_TURN_ANGLE=30 RYNNVLN_HFOV=120 RYNNVLN_ALLOW_SLIDING=1
export RYNNBRAIN_NAV_DIR="$NAV"
# Imported modules under the shared NAV/venv remain read-only as well.
export PYTHONDONTWRITEBYTECODE=1
# bench 不含 vLLM，MODEL_PATH 占位（env.py 里 setdefault，VLNEvaluator model=None 不会真加载）
export MODEL_PATH="${MODEL_PATH:-/dummy/vln_gym_no_vllm}"

# ---- 每个 gym 在自己的 workspace 内挂 data + 放 yaml，NAV 保持只读 ----
# Hydra 和数据路径仍按 CWD 解析；按端口隔离避免几十个 gym 同时覆盖配置。
mkdir -p "$GYM_WORK_DIR" || exit 2
ln -sfn "$DATA_ROOT" "$GYM_WORK_DIR/data" || exit 2
cp -f "$HERE/src/vln_demo.yaml" "$GYM_WORK_DIR/vln_demo.yaml" || exit 2

# vln_gym 包在 $HERE/src 下；PYTHONPATH 带上让它能 -m 启动，agent_tools 同理
export PYTHONPATH="$HERE/src:$NAV/src:${PYTHONPATH:-}"
cd "$GYM_WORK_DIR" || exit 2
echo "[vln_gym_server] GYM_WORK_DIR=$(pwd)"

echo "[vln_gym_server] starting HTTP gym server on 0.0.0.0:$PORT backend=$BACKEND (CPU soft-render, no vLLM)"
exec python3 -m vln_gym.server \
  --port "$PORT" --scene "$SCENE" --episode_id "$EPISODE_ID" \
  --split "$SPLIT" --max_steps "$MAX_STEPS" --backend "$BACKEND"
