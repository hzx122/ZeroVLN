#!/usr/bin/env bash
# 一键起 Sage Visual Physical v3：render server(7038) + console(5000)，全程 setsid 脱离，
# 供 harness 之外常驻。日志 /tmp/sage_render_7038.log 与 /tmp/sage_console_hybrid.log。
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GV=/opt/navigation-runtime/venv
RS=/workspace/agentic_sim/repo/ABot-Navigation/render_server
TGT="$HERE/render_libs"
SCENES_ROOT=/data/zerovln/SAGE-3D/sage_gs_root
NAV=/opt/navigation-runtime/VLMEvalKit/3rdparty/RynnBrain/navigation
SRC="$HERE/src"
SAGE_DEMO_SPLIT=${SAGE_SPLIT:-scene}
# 固定约定：7036/7037/7038 依次用 PPU 0/1/2（见 start_all.sh）。本脚本只起 sage 一个资产集，
# 要 8 个资产集全量用 start_all.sh。
SAGE_GPU=${SAGE_GPU:-2}
RGB_CLEANUP=${SAGE_RGB_OUTSIDE_CLEANUP:-1}

if [[ "$RGB_CLEANUP" == 1 ]] && ! grep -q "suppress_blue_gaussians" "$RS/render_sim_dynamic.py"; then
  echo "[warn] render server 缺少 RGB cleanup 可选扩展，异常帧将 fail-open 返回原图" >&2
  echo "       git -C $(dirname "$RS") apply $HERE/patches/sage_rgb_outside_cleanup_render_server.patch" >&2
fi

# Non-interactive SSH does not inherit DSW's PPU runtime paths.  The renderer
# otherwise exits before Python starts with "libhggcrt1.so: cannot open".
export LD_LIBRARY_PATH="/opt/accl-p:/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"

pkill -9 -f "render_sim_dynamic.*7038" 2>/dev/null || true
pkill -9 -f "vln_model.py" 2>/dev/null || true
for p in 5000 7038; do fuser -k $p/tcp 2>/dev/null || true; done
sleep 2

# ---- render server 7038 (PPU 2 默认, SH3) ----
PYTHONPATH="$TGT:$RS/submodules/diff-plane-rasterization:$RS/submodules/simple-knn:$RS" \
  setsid "$GV/bin/python" "$RS/render_sim_dynamic.py" \
  --scenes_root "$SCENES_ROOT" --port 7038 --gpus "$SAGE_GPU" --sh_degree 3 \
  --max_scenes_per_gpu 20 > /tmp/sage_render_7038.log 2>&1 < /dev/null &
echo "[start] render pid=$!"

# 等 render pong（最多 60s）
for i in $(seq 1 20); do
  sleep 3
  curl -s -m3 http://127.0.0.1:7038/ping 2>/dev/null | grep -q pong && { echo "[start] render pong @ ${i}x3s"; break; }
done

# ---- console 5000 (sage, visual/physical 碰撞) ----
cd "$NAV"
export PATH="$GV/bin:$PATH" LD_LIBRARY_PATH="$GV/lib:${LD_LIBRARY_PATH:-}"
export EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export HABITAT_SIM_GPU_DEVICE_ID=-1 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet PYTHONUNBUFFERED=1
export DEMO_BACKEND=sage SAGE_COLLISION=visual SAGE_SPLIT="$SAGE_DEMO_SPLIT" \
       SAGE_RENDER_URL=http://127.0.0.1:7038 SAGE_BLOCK_FURNITURE=0 \
       SAGE_FURNITURE_COLLISION=geom SAGE_ROBOT_RADIUS=0.07 \
       SAGE_RGB_OUTSIDE_CLEANUP="$RGB_CLEANUP"
setsid "$GV/bin/python" "$SRC/vln_model.py" --backend sage --port 5000 --sage_split "$SAGE_DEMO_SPLIT" \
  > /tmp/sage_console_hybrid.log 2>&1 < /dev/null &
echo "[start] console pid=$!"
echo "[start] done; tail /tmp/sage_console_hybrid.log for readiness"
