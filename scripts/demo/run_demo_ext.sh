#!/usr/bin/env bash
# vln_demo 外部-vLLM 跑法(已验证可跑通):
#   1) 先起 vLLM:  bash vllm/vllm-vln-2B.sh   (GPU13/端口28898/模型MoBot_VLN-2B)
#                  bash vllm/vllm-vln-35B.sh  (GPU14,15/端口28899/模型Qwen3.6-35B)
#      等 /health 返回 200 (PPU warmup 约 5 分钟)。
#   2) 再跑本脚本:  bash run_demo_ext.sh
# 本脚本只跑 demo(habitat 软渲染 + RynnBrain R2R 3 episode), 不起 vLLM。
# run_demo.sh 是另一种自包含入口(自己起 vLLM), 但其默认 MODEL_PATH/chat-template
# 与 vllm-vln-2B.sh 不一致, 推荐用本脚本 + 外部 vllm-vln-*.sh。
set -uo pipefail

# ---- 路径(迁移到新机器时改这里) ----
VENV=/opt/navigation-runtime/venv
NAV=/opt/navigation-runtime/VLMEvalKit/3rdparty/RynnBrain/navigation
DATA_ROOT=/data/shared/groups/mfm/common/anno/vlmevalkit/RynnBrain/data
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../vln_demo" && pwd)   # 本脚本所在=vln_demo 根

# ---- 选哪个后端(对应 vllm-vln-*.sh 起的 served-model-name / 端口) ----
BACKEND=${BACKEND:-2B}        # 2B => MoBot_VLN-2B@28898 ; 35B => Qwen3.6-35B@28899
case "$BACKEND" in
  2B)  VLLM_URL=http://127.0.0.1:28898; VLLM_NAME=MoBot_VLN-2B
       MODEL_PATH=${MODEL_PATH:-/data/shared/groups/mfm/vl/rl/pretrain_models/final_mix_mt_me6/hf_iter_0001250} ;;
  35B) VLLM_URL=http://127.0.0.1:28899; VLLM_NAME=Qwen3.6-35B
       MODEL_PATH=${MODEL_PATH:-/opt/external/user-e7a01947/Model/Qwen3.6-35B-A3B} ;;
  *) echo "BACKEND must be 2B or 35B"; exit 1 ;;
esac

DEMO_OUT=${DEMO_OUT:-$HERE/outputs/run_ext_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$DEMO_OUT"
echo "[run_demo_ext] BACKEND=$BACKEND vllm=$VLLM_URL model=$MODEL_PATH out=$DEMO_OUT"

source "$VENV/bin/activate"

# ---- 软渲染四件套 + libEGL/libOpenGL 路径 ----
# $HERE/lib 自带 libEGL.so.1 + libegl-mesa0(.2) 等; 系统也需装 mesa .2 全套(见 MIGRATION.md)
export LD_LIBRARY_PATH="$HERE/lib:$VENV/lib:${LD_LIBRARY_PATH:-}"
export EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe GALLIUM_DRIVER=llvmpipe HABITAT_SIM_GPU_DEVICE_ID=-1 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

# ---- RynnBrain / vLLM 推理参数 ----
export RYNNVLN_LANGUAGE=en RYNNVLN_TURN_ANGLE=30 RYNNVLN_HFOV=120 RYNNVLN_ALLOW_SLIDING=1
export RYNNVLN_LOG_RAW_OUTPUT=1 RYNNVLN_VLLM_IMAGE_FORMAT=JPG RYNNVLN_VLLM_IMAGE_QUALITY=90 MAX_NEW_TOKENS=2048
export RYNNVLN_VLLM_SERVER_URLS="$VLLM_URL" RYNNVLN_VLLM_MODEL_NAME="$VLLM_NAME" RYNNBRAIN_NAV_DIR="$NAV"
export MODEL_PATH DEMO_OUTPUT_PATH="$DEMO_OUT"

# ---- 给 NAV 目录挂 data + 放 yaml(hydra 按 CWD 解析) ----
ln -sfn "$DATA_ROOT" "$NAV/data"
cp -f "$HERE/src/vln_demo.yaml" "$NAV/vln_demo.yaml"
cd "$NAV"

# ---- 选 episode 的开关(可环境变量覆盖) ----
# DEMO_SCENE / DEMO_EPISODE_IDS / DEMO_NUM_EPISODES / DEMO_MAX_STEPS / DEMO_SPLIT
python3 "$HERE/src/vln_demo.py"
