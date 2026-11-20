#!/usr/bin/env bash
# 一键起「可视化页面跑训练配置 VLN(FM) 模型」：render(7038 sage) + console(5000)。
# 与 start_sage_visual.sh 的区别：控制台用 **.venv_fm**（不是 gordon venv）起——这样
# FM driver（run_fm → _load_fm）才能加载 SocialNav forked transformers + GPU 的 FM head；
# _load_fm 默认走 closed_loop 训练配置那份 fm_policy（<repo>/verl/workers/fm_policy.py
# + <repo>/third_party/SocialNav），即 SFT/RL 用的同一份（含 PROMPT_VARIANT / 0729 base）。
#
# 用法：
#   bash start_sage_fm_viz.sh                      # 默认 SAGE_SPLIT=scene, FM_CKPT=0729 base
#   SAGE_SPLIT=instr bash start_sage_fm_viz.sh
#   FM_CKPT=/data/storage/.../sage_sft_*/checkpoint-N bash start_sage_fm_viz.sh   # 换你的 SFT/RL ckpt
#
# 起完浏览器开 http://<本机IP>:5000，驱动选 "FM 流匹配"，点 Start 即在 sage 场景里跑你的 vln。
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../vln_demo" && pwd)
REPO=$(cd "$HERE/.." && pwd)            # closed_loop_3dgs_rl
GV=/opt/navigation-runtime/venv
VENV_FM="$REPO/vln_demo/.venv_fm"
RS=/workspace/agentic_sim/repo/ABot-Navigation/render_server
TGT="$HERE/render_libs"
SCENES_ROOT=/data/zerovln/SAGE-3D/sage_gs_root
NAV=/opt/navigation-runtime/VLMEvalKit/3rdparty/RynnBrain/navigation
SRC="$HERE/src"

SAGE_SPLIT=${SAGE_SPLIT:-scene}
RENDER_GPU=${RENDER_GPU:-1}
CONSOLE_PORT=${DEMO_HTTP_PORT:-5000}
FM_DEVICE=${FM_DEVICE:-cuda:0}
RGB_CLEANUP=${SAGE_RGB_OUTSIDE_CLEANUP:-1}

# PPU runtime（render_sim + .venv_fm torch 都需要）
export LD_LIBRARY_PATH="/opt/accl-p:/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"

# 自清残留
pkill -9 -f "render_sim_dynamic.*7038" 2>/dev/null || true
pkill -9 -f "vln_model.py" 2>/dev/null || true
for p in "$CONSOLE_PORT" 7038; do fuser -k $p/tcp 2>/dev/null || true; done
sleep 2

echo "[fm-viz] start=$(date -Is) split=$SAGE_SPLIT render_gpu=$RENDER_GPU fm_device=$FM_DEVICE"

# ---- ① render server 7038 (sage, GPU$RENDER_GPU, SH3) — gordon venv ----
PYTHONPATH="$TGT:$RS/submodules/diff-plane-rasterization:$RS/submodules/simple-knn:$RS" \
  setsid "$GV/bin/python" "$RS/render_sim_dynamic.py" \
  --scenes_root "$SCENES_ROOT" --port 7038 --gpus "$RENDER_GPU" --sh_degree 3 \
  --max_scenes_per_gpu 20 > /tmp/sage_render_7038.log 2>&1 < /dev/null &
echo "[fm-viz] render pid=$! (gordon venv, GPU$RENDER_GPU)"

for i in $(seq 1 30); do
  sleep 3
  curl -s -m3 http://127.0.0.1:7038/ping 2>/dev/null | grep -q pong && { echo "[fm-viz] render pong @ ${i}x3s"; break; }
done
curl -s -m3 http://127.0.0.1:7038/ping >/dev/null 2>&1 || { echo "[fm-viz] render 没起来，看 /tmp/sage_render_7038.log"; tail -15 /tmp/sage_render_7038.log; exit 1; }

# ---- ② console 5000 (sage 后端, FM driver 走 closed_loop fm_policy) — .venv_fm ----
# PYTHONPATH 让控制台进程能 import: closed_loop verl/workers+bench（fm_policy/bench_client）
# + third_party/SocialNav(+train_clean) + vln_demo/src（sage/vln_model）+ render_libs + NAV/src
export PYTHONPATH="$REPO/verl/workers:$REPO/verl/bench:$REPO/third_party/SocialNav:$REPO/third_party/SocialNav/train_clean:$SRC:$TGT:$NAV/src:${PYTHONPATH:-}"
export PATH="$VENV_FM/bin:$PATH"
export RYNNBRAIN_NAV_DIR="$NAV"
export RYNNVLN_LANGUAGE="en"
# sage 后端 env（同 start_sage_visual.sh）
export DEMO_BACKEND=sage SAGE_COLLISION=visual SAGE_SPLIT="$SAGE_SPLIT" \
       SAGE_RENDER_URL=http://127.0.0.1:7038 SAGE_BLOCK_FURNITURE=0 \
       SAGE_FURNITURE_COLLISION=geom SAGE_ROBOT_RADIUS=0.07 \
       SAGE_RGB_OUTSIDE_CLEANUP="$RGB_CLEANUP"
# FM 模型：0729 base（训练配置默认）；FM_CKPT env 覆盖可换 SFT/RL ckpt。FM_DEVICE=cuda:0。
export FM_DEVICE="$FM_DEVICE"
: "${FM_CKPT:=/data/shared/groups/mfm/vl/exp/qwen3_vl_2b_fm/fm_baseline_fixed/qwen3vl_2b_fm_0729-baseline-cocofix-0p5-align_tp2_16nodes_mbs8_20260802_012647/checkpoint_tp2_hf_16k}"
export FM_CKPT
# 页面起来即默认选中 VLN-FM 模型（vlnfm0729）→ 驱动锁死 FM·流匹配，直接 Start 就跑你的 vln。
export DEMO_DEFAULT_MODEL="${DEMO_DEFAULT_MODEL:-vlnfm0729}"
echo "[fm-viz] FM_CKPT=$FM_CKPT  model=$DEMO_DEFAULT_MODEL  (env 可覆盖)"

cd "$NAV"
setsid "$VENV_FM/bin/python" "$SRC/vln_model.py" \
  --backend sage --port "$CONSOLE_PORT" --sage_split "$SAGE_SPLIT" \
  > /tmp/sage_console_fm.log 2>&1 < /dev/null &
echo "[fm-viz] console pid=$! (.venv_fm, port $CONSOLE_PORT)"

# 等控制台 /status 起来
for i in $(seq 1 20); do
  sleep 2
  curl -s -m2 "http://127.0.0.1:$CONSOLE_PORT/status" >/dev/null 2>&1 && { echo "[fm-viz] console up @ ${i}x2s"; break; }
done
echo "[fm-viz] done. 浏览器开："
python3 - "$CONSOLE_PORT" <<'PY'
import socket,sys
port=sys.argv[1]
ips=["127.0.0.1"]
try:
    s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect(("8.8.8.8",80)); ips.append(s.getsockname()[0]); s.close()
except Exception: pass
print("  http://127.0.0.1:%s"%port)
print("  http://%s:%s"%(ips[-1],port))
PY
echo "[fm-viz] 驱动选「FM 流匹配」→ Start。tail -f /tmp/sage_console_fm.log 看就绪/FM 加载。"
