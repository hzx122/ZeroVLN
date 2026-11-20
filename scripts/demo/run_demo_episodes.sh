#!/usr/bin/env bash
# 3DGS 多类型指令用例 demo：造新用例 + 可视化。
#
#   1. 确保 render 7038 在线（目标 RGB/depth 是生产硬校验）
#   2. 跑 build_demo_episodes.py：ECM + 平滑 + grounded-v4 两类确定性指令
#   3. 重启 render 7038 + console 5000（SAGE_SPLIT=demo）
#
# 指令事实和文本都由受控模板生成；Qwen 只做无红框 RGB 目标定位质检。
# 默认不带参数读取 config/sage_demo_pipeline.yaml；实验 wrapper 只用
# --set dotted.key=value 覆盖与默认配置不同的项。
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../vln_demo" && pwd)          # vln_demo
REPO=$(cd "$HERE/.." && pwd)                                 # closed_loop_3dgs_rl
SRC="$HERE/src"
GV=/opt/navigation-runtime/venv
PIPELINE_CONFIG="$HERE/config/sage_demo_pipeline.yaml"
OVERRIDES=()
PRINT_CONFIG=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || { echo "[error] --config requires a path" >&2; exit 2; }
      PIPELINE_CONFIG="$2"
      shift 2
      ;;
    --set)
      [[ $# -ge 2 ]] || { echo "[error] --set requires dotted.key=value" >&2; exit 2; }
      OVERRIDES+=(--set "$2")
      shift 2
      ;;
    --print-config)
      PRINT_CONFIG=1
      shift
      ;;
    *)
      echo "[error] unknown argument: $1 (use --config PATH, --set dotted.key=value, or --print-config)" >&2
      exit 2
      ;;
  esac
done

CFG_ARGS=(--config "$PIPELINE_CONFIG" "${OVERRIDES[@]}")
if [[ "$PRINT_CONFIG" == 1 ]]; then
  PYTHONPATH="$SRC" "$GV/bin/python" -m sage.demo_pipeline_config "${CFG_ARGS[@]}"
  exit 0
fi
cfg_get() {
  PYTHONPATH="$SRC" "$GV/bin/python" -m sage.demo_pipeline_config \
    "${CFG_ARGS[@]}" --get "$1"
}
SCENES=$(cfg_get runtime.scenes)
SAGE_ROOT=$(cfg_get runtime.sage_root)
RENDER_URL=$(cfg_get runtime.render_url)
HFOV=$(cfg_get camera.hfov_deg)
CAMERA_HEIGHT=$(cfg_get camera.height_m)
VLM_MODELS_URL=$(cfg_get visibility.vlm_models_url)
VLM_MODEL=$(cfg_get visibility.vlm_model)
TERMINAL_MODE=$(cfg_get goal.terminal_mode)
echo "[config] $PIPELINE_CONFIG"
echo "[config] overrides=${OVERRIDES[*]:-none} scenes=$SCENES hfov=$HFOV terminal=$TERMINAL_MODE"

if ! curl -s -m 3 "$RENDER_URL/ping" 2>/dev/null | grep -q pong; then
  echo "[1/3] bootstrap render 7038 for goal-visibility validation"
  env SAGE_ROOT="$SAGE_ROOT" SAGE_HFOV="$HFOV" SAGE_CAMERA_HEIGHT="$CAMERA_HEIGHT" \
    SAGE_SPLIT=demo SAGE_SCENES="$SCENES" SAGE_EPISODES_PER_SCENE=0 \
    SAGE_GOLDEN_ASTAR=0 bash "$REPO/scripts/demo/start_sage_usd.sh"
else
  echo "[1/3] render 7038 already ready"
fi
if ! curl -s -m 5 "$VLM_MODELS_URL" 2>/dev/null | grep -q "$VLM_MODEL"; then
  echo "[error] grounded-v4 requires $VLM_MODEL at $VLM_MODELS_URL" >&2
  exit 2
fi

echo "[2/3] build production demo episodes (ECM + smoothing + grounded-v4)"
cd "$SRC"
PYTHONPATH=. "$GV/bin/python" -m sage.build_demo_episodes "${CFG_ARGS[@]}"

echo "[3/3] restart console (render 7038 + console 5000, SAGE_SPLIT=demo)"
echo "  open http://<this-node>:5000  -> backend sage -> scene 840483 -> pick episode"
echo "  橙线=ECM centred-A* golden, 绿圈=goal, 「走专家轨迹」沿路径逐帧 render"
exec env \
  SAGE_ROOT="$SAGE_ROOT" \
  SAGE_HFOV="$HFOV" \
  SAGE_CAMERA_HEIGHT="$CAMERA_HEIGHT" \
  SAGE_SPLIT=demo \
  SAGE_SCENES="$SCENES" \
  SAGE_EPISODES_PER_SCENE=0 \
  SAGE_GOLDEN_ASTAR=0 \
  bash "$REPO/scripts/demo/start_sage_usd.sh"
