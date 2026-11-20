#!/usr/bin/env bash
# VLN 交互控制台: 网页上选场景/回合/模型/提示词, 手动驾驶或交给模型, 模型输出流式展示。
# 需要常驻 vLLM 在跑, 两个后端各起一个 (见 ~/vllm/vllm-vln-35B.sh / vllm-vln-2B.sh):
#   28899 Qwen3.6-35B (2 卡)   28898 MoBot_VLN-2B (1 卡)
# 网页上的「模型」下拉决定用哪个, 下面的环境变量只是给 rynnvln_eval 打底。
# 仿真渲染走 CPU 软渲染, 不占卡。
# 用法:
#   bash model.sh                                  # 默认场景/回合
#   DEMO_SCENE=xxx DEMO_EPISODE_ID=16 bash model.sh
#   DEMO_HTTP_PORT=8899 bash model.sh              # 换端口(并行开第二个)
set -uo pipefail

VENV=/opt/navigation-runtime/venv
NAV=/opt/navigation-runtime/VLMEvalKit/3rdparty/RynnBrain/navigation
# 跟随脚本自身位置, 这样在 git worktree / 别的 clone 里也能直接跑
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../vln_demo" && pwd)

source "$VENV/bin/activate"
export LD_LIBRARY_PATH="$HERE/lib:$VENV/lib:${LD_LIBRARY_PATH:-}"
export EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 \
       MESA_LOADER_DRIVER_OVERRIDE=llvmpipe GALLIUM_DRIVER=llvmpipe
export HABITAT_SIM_GPU_DEVICE_ID=-1 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export RYNNVLN_LANGUAGE=en RYNNVLN_VLLM_IMAGE_FORMAT=JPG RYNNVLN_VLLM_IMAGE_QUALITY=90
# 输出长度现在由 src/prompts.py 的 MODES 控制(思考 40960 / 非思考 32768 tokens),
# 这个变量只影响 rynnvln_eval 内部的非流式路径, demo 已不走那条路。
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
# 底层 VLN 模型。可选后端写在 src/prompts.py 的 MODELS 里, 网页下拉切换;
# 下面这组只是 rynnvln_eval 构造时的默认值, 每次运行都会被 apply_model() 覆盖。
# 两个后端各用自己的真名(Qwen3.6-35B / MoBot_VLN-2B)而不是 default: 同名的话
# URL 配错也能拿到正常回复, 等于静默测错模型; 用真名配错就是 404。
export MODEL_PATH=${MODEL_PATH:-/opt/external/user-e7a01947/Model/Qwen3.6-35B-A3B}
export RYNNVLN_VLLM_SERVER_URLS=${RYNNVLN_VLLM_SERVER_URLS:-http://127.0.0.1:28899}
export RYNNVLN_VLLM_MODEL_NAME=${RYNNVLN_VLLM_MODEL_NAME:-Qwen3.6-35B}
export DEMO_SCENE=${DEMO_SCENE:-2azQ1b91cZZ}
export DEMO_EPISODE_ID=${DEMO_EPISODE_ID:-11}
export DEMO_HTTP_PORT=${DEMO_HTTP_PORT:-5000}
export DEMO_OUTPUT_PATH=${DEMO_OUTPUT_PATH:-$HERE/outputs/model}

# 检查常驻 vLLM。两个后端各查一次, 全都不在才退出 —— 只起了一个也应该能开页面,
# 网页上选到没起的那个时 vln_model.py 会当场报错, 不至于让整个控制台起不来。
UP=0
for U in http://127.0.0.1:28899 http://127.0.0.1:28898; do
  if curl -s --noproxy '*' -o /dev/null -m 4 "$U/health"; then
    echo "[vLLM] $U 在线"; UP=$((UP+1))
  else
    echo "[vLLM] $U 离线(网页上选到它会报错)"
  fi
done
if [ "$UP" -eq 0 ]; then
  echo "!! 两个 vLLM 都没在跑, 先起服务: bash ~/vllm/vllm-vln-35B.sh / vllm-vln-2B.sh !!"
  exit 1
fi

cp -f "$HERE/src/vln_demo.yaml" "$NAV/vln_demo.yaml"
cd "$NAV"

PORT=$DEMO_HTTP_PORT
echo "=================================================================="
echo " VLN 交互控制台"
echo "   浏览器打开 http://127.0.0.1:$PORT"
echo "   本地转发:  ssh -L $PORT:127.0.0.1:$PORT <你连这台机的地址>"
echo " 页面上可切场景/回合、切模型、切驱动/提示词、手动 W/A/D 驾驶、模型流式输出"
echo "=================================================================="
python3 "$HERE/src/vln_model.py" --scene "$DEMO_SCENE" --episode_id "$DEMO_EPISODE_ID" --port "$PORT"
