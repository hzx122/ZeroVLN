#!/usr/bin/env bash
# 一键起「4 个 3DGS 渲染服务 + 8 资产集控制台」。
#
#   bash scripts/demo/start_all.sh            # 渲染服务缺谁补谁，控制台重起
#   bash scripts/demo/start_all.sh restart    # 连渲染服务一起强制重起（换资产 root 时用）
#   bash scripts/demo/start_all.sh stop
#   bash scripts/demo/start_all.sh status
#
# 分工照搬原版 vln_demo（/opt/external/user-d5121be6/vln_demo）的安排：
#
#   * 控制台交给 src/run_console.sh，本脚本不重复它那套 venv/软件 GL/NAV_DIR 环境。
#     它是唯一入口，别直接 python vln_model.py。
#   * 渲染服务**故意不由 run_console.sh 起**：它把高斯留在显存里按 LRU 池着，要活得比
#     很多次控制台运行更久。所以这里默认**只补不杀**——端口已经 pong 且 scenes_root 对
#     得上就原样复用，不白白丢掉已经加载的场景。要换 root 才用 restart。
#
# 端口 → PPU → 资产（一个服务独占一张卡，不与别的渲染服务挤同一张）：
#
#   7036  PPU 0  sh0  3drs_gs_root   → 资产集 3drs        （自采 3DGS，free-roam）
#   7037  PPU 1  sh3  abotn_gs_root  → 资产集 poi/in/out  （ABotN，有 goal）
#   7038  PPU 2  sh3  sage_gs_root   → 资产集 sage        （SAGE-3D GVLN，有 goal）
#   7039  PPU 3  sh0  zekang 3drs    → 资产集 mall        （万科里商场，degree-0 但 root 不同）
#
# 为什么必须四个进程：--sh_degree 按进程定死（load_gs_ply 断言系数个数），3DRS/mall 的
# ply 一个 f_rest 都没有（degree 0）而 ABotN/SAGE 是 degree 3，一个进程装不下两套。ABotN
# 和 SAGE 虽同为 degree 3，也还是分开：一个服务只认一个 scenes_root，合成一棵树意味着任何
# 一边加场景都要重建它。mall 虽与 3DRS 同为 degree 0，scenes_root 在 zekang 盘上、与 3DRS
# 不同，同理要独立进程独立卡。habitat 系（habitat/rxr）不用渲染服务，走 CPU llvmpipe。
# PARTNR 同属 habitat 系，但渲染质量太差、已在 BACKENDS 里标 hidden，默认不再启动；
# 要单独调它用 DEMO_BACKEND=partnr。
#
# 主策略占 14 张 PPU：渲染 0-3，Qwen3.8-Flash-Next vLLM×8 占 4-11，35B×2 占
# 12-13；控制台默认模型为 35B（28900）。本脚本只管渲染和控制台，模型服务单独启动，见
# vln_demo/vllm/README.md。14-15 为外挂槽；LightNav 使用 14，运行双卡 Action Head 前
# 需自行协调同槽服务。
set -u

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../vln_demo" && pwd)   # vln_demo 根
ACTION=${1:-start}

# ---- 外部依赖（机器特定，迁机器改这两个）----
GV=${GV:-/opt/navigation-runtime/venv}
RS=${RS:-/workspace/agentic_sim/repo/ABot-Navigation/render_server}

# ---- 端口 / PPU / 资产 root / LRU 上限 ----
RS3D_PORT=${RS3D_PORT:-7036};   RS3D_GPU=${RS3D_GPU:-0};   RS3D_SPG=${RS3D_SPG:-2}
ABOTN_PORT=${ABOTN_PORT:-7037}; ABOTN_GPU=${ABOTN_GPU:-1}; ABOTN_SPG=${ABOTN_SPG:-3}
SAGE_PORT=${SAGE_PORT:-7038};   SAGE_GPU=${SAGE_GPU:-2};   SAGE_SPG=${SAGE_SPG:-20}
MALL_PORT=${MALL_PORT:-7039};   MALL_GPU=${MALL_GPU:-3};   MALL_SPG=${MALL_SPG:-2}
RS3D_ROOT=${RS3D_ROOT:-/workspace/agentic_sim/3drs_gs_root}
ABOTN_ROOT=${ABOTN_ROOT:-/workspace/agentic_sim/abotn_gs_root}
SAGE_ROOT=${SAGE_ROOT:-/data/zerovln/SAGE-3D/sage_gs_root}
MALL_ROOT=${MALL_ROOT:-/opt/external/user-9656ce61/ppu_sim/3drs_gs_root}
# 3DRS 关超采样与原版一致（原版：--render_scale 1.0）；另外几套用服务端默认 1.5
RS3D_RENDER_SCALE=${RS3D_RENDER_SCALE:-1.0}
# mall 资产目录（walk.json / floor_meta.json / mall_walk_union_topo.npy）；不在 scenes_root 下
MALL_ASSETS_DIR=${MALL_ASSETS_DIR:-/opt/external/user-9656ce61/grpo/locnav_grpo/exp0822/assets_arr5/mall}

CONSOLE_PORT=${CONSOLE_PORT:-5000}
DEMO_BACKENDS=${DEMO_BACKEND:-all}   # all = 8 个资产集（hidden 的 hab/partnr 不含在内）
SAGE_SPLIT=${SAGE_SPLIT:-scene}      # train(929)/instr(15)/scene(15)/traj(14)

RENDER_PORTS=($RS3D_PORT $ABOTN_PORT $SAGE_PORT $MALL_PORT)
# DLC task entrypoints can retain logs/PIDs on mounted storage. Local defaults
# and the existing render-reuse behavior remain the same.
LOG_DIR=${VLN_LOG_DIR:-/tmp}
PID_DIR=${VLN_PID_DIR:-}
STRICT_STARTUP=${VLN_STRICT_STARTUP:-0}
LOG_CONSOLE=$LOG_DIR/console_${CONSOLE_PORT}.log
mkdir -p "$LOG_DIR"
if [ -n "$PID_DIR" ]; then mkdir -p "$PID_DIR"; fi

# 非交互 shell 不继承 DSW 的 PPU runtime 路径，渲染器会在 python 起来前就
# "libhggcrt1.so: cannot open" 退出。
PPU_LIBS=/opt/accl-p:/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/lib:/usr/local/lib
PYPATH="$HERE/render_libs:$RS/submodules/diff-plane-rasterization:$RS/submodules/simple-knn:$RS"

# 本机没装 fuser（command not found，静默失败），只能走 ss 查监听 pid
port_pids() { ss -lntpH 2>/dev/null | grep -F ":$1 " | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u; }
pong()      { [ "$(curl -s -m3 "http://127.0.0.1:$1/ping" 2>/dev/null)" = pong ]; }
served_root() { curl -s -m5 "http://127.0.0.1:$1/status" 2>/dev/null |
                grep -oE '"scenes_root":"[^"]*"' | cut -d'"' -f4; }

kill_port() {  # 按端口停，不用 pkill -f render_sim_dynamic：那会连别人/RL 脚本起的一起杀
  local pids; pids=$(port_pids "$1")
  [ -n "$pids" ] && { kill -9 $pids 2>/dev/null; echo "[stop] :$1 killed (pid $(echo $pids | tr '\n' ' '))"; }
  return 0
}

stop_render() {
  local p
  for p in "${RENDER_PORTS[@]}"; do kill_port "$p"; done
  sleep 2
  # 抢端口失败、没在 listen 但还活着的残留（不清掉会让下一轮以为端口空着）
  local stale
  stale=$(ps -eo pid,args | grep -E "[r]ender_sim_dynamic\.py .*--port ($(IFS='|'; echo "${RENDER_PORTS[*]}"))( |$)" | awk '{print $1}')
  [ -n "$stale" ] && { kill -9 $stale 2>/dev/null; echo "[stop] 残留 render killed (pid $(echo $stale | tr '\n' ' '))"; }
  return 0
}

stop_console() {
  kill_port "$CONSOLE_PORT"; sleep 1
  local stale
  stale=$(ps -eo pid,args | grep "[v]ln_model\.py .*--port $CONSOLE_PORT" | awk '{print $1}')
  [ -n "$stale" ] && { kill -9 $stale 2>/dev/null; echo "[stop] 残留 console killed (pid $(echo $stale | tr '\n' ' '))"; }
  return 0
}

status_all() {
  local p root
  for p in "${RENDER_PORTS[@]}"; do
    if pong "$p"; then root=$(served_root "$p"); printf '  render :%s  pong  %s\n' "$p" "$root"
    else printf '  render :%s  (无响应)\n' "$p"; fi
  done
  printf '  console :%s  HTTP %s\n' "$CONSOLE_PORT" \
    "$(curl -s -o /dev/null -w '%{http_code}' -m5 "http://127.0.0.1:$CONSOLE_PORT/" 2>/dev/null || echo 无响应)"
}

# 只补不杀：端口已 pong 且 scenes_root 一致就复用（保住已加载的场景）
ensure_render() {  # name root port gpu sh spg [render_scale]
  local name=$1 root=$2 port=$3 gpu=$4 sh=$5 spg=$6 scale=${7:-}
  if pong "$port"; then
    [ "$STRICT_STARTUP" != 1 ] || { echo "[error] strict startup refuses an existing render on :$port"; return 1; }
    local cur; cur=$(served_root "$port")
    if [ "$cur" = "$root" ]; then echo "[keep ] render $name :$port 已在跑，复用"; return 0; fi
    echo "[warn ] :$port 上跑的是别的 root（$cur），重起成 $root"
    kill_port "$port"; sleep 2
  fi
  [ -e "$root" ] || { echo "[error] $name 资产 root 不存在，跳过：$root"; return 1; }
  # 不静默跳过占用：否则新进程绑不上端口就死，而 wait_pong 会被旧服务的 pong 骗过
  [ -n "$(port_pids "$port")" ] && { echo "[error] :$port 被占用且不响应 /ping，$name 不启动"; return 1; }
  local log=$LOG_DIR/render_${port}.log
  LD_LIBRARY_PATH="$PPU_LIBS:${LD_LIBRARY_PATH:-}" PYTHONPATH="$PYPATH" \
    setsid "$GV/bin/python" "$RS/render_sim_dynamic.py" \
    --scenes_root "$root" --port "$port" --gpus "$gpu" \
    --sh_degree "$sh" --max_scenes_per_gpu "$spg" \
    ${scale:+--render_scale "$scale"} > "$log" 2>&1 < /dev/null &
  local pid=$!
  if [ -n "$PID_DIR" ]; then printf '%s\n' "$pid" > "$PID_DIR/render_${port}.pid"; fi
  echo "[start] render $name :$port PPU$gpu sh$sh pid=$pid log=$log"
}

wait_pong() {  # 等全部 render pong，最多 ~120s
  local ok i p
  for i in $(seq 1 40); do
    ok=0
    for p in "${RENDER_PORTS[@]}"; do pong "$p" && ok=$((ok+1)); done
    [ "$ok" -eq ${#RENDER_PORTS[@]} ] && { echo "[start] 渲染服务全部 pong"; return 0; }
    sleep 3
  done
  echo "[warn ] 只有 $ok/${#RENDER_PORTS[@]} 个 render pong，控制台会跳过起不来的资产集"
  [ "$STRICT_STARTUP" != 1 ]
}

start_console() {
  # 渲染服务地址 + Sage Visual Physical v3（代码默认是 official/0.10m，必须显式给）
  export RS3D_RENDER_URL="http://127.0.0.1:$RS3D_PORT"
  export ABOTN_RENDER_URL="http://127.0.0.1:$ABOTN_PORT"
  export SAGE_RENDER_URL="http://127.0.0.1:$SAGE_PORT"
  export MALL_RENDER_URL="http://127.0.0.1:$MALL_PORT"
  export LEPU_RENDER_URL=${LEPU_RENDER_URL:-$MALL_RENDER_URL}
  export MALL_ASSETS_DIR="$MALL_ASSETS_DIR"
  export SAGE_COLLISION=${SAGE_COLLISION:-visual}
  export SAGE_FURNITURE_COLLISION=${SAGE_FURNITURE_COLLISION:-geom}
  export SAGE_BLOCK_FURNITURE=${SAGE_BLOCK_FURNITURE:-0}
  export SAGE_ROBOT_RADIUS=${SAGE_ROBOT_RADIUS:-0.07}
  export SAGE_SPLIT="$SAGE_SPLIT"
  export DEMO_BACKEND="$DEMO_BACKENDS"
  # DEMO_LOCK=1 让控制台一起来就是「前端锁」状态（只有后台 agent 能操作）。默认 0：
  # 锁是给 agent 调试用的开关，不该是常态，不然人打开页面点什么都被拒还不知道为什么。
  export DEMO_LOCK=${DEMO_LOCK:-0}
  export PYTHONUNBUFFERED=1
  # habitat/rxr 构造 RynnBrain VLNEvaluator 时要一个非空 vllm URL，否则这两个资产集
  # 以 "vllm_server backend requires --vllm_server_urls" 被静默跳过。
  export RYNNVLN_VLLM_SERVER_URLS=${RYNNVLN_VLLM_SERVER_URLS:-http://127.0.0.1:28899}
  # 软件 GL / venv / NAV_DIR 全在 run_console.sh 里，这里不重复
  setsid bash "$HERE/../scripts/demo/run_console.sh" "$CONSOLE_PORT" --sage_split "$SAGE_SPLIT" \
    > "$LOG_CONSOLE" 2>&1 < /dev/null &
  local pid=$!
  if [ -n "$PID_DIR" ]; then printf '%s\n' "$pid" > "$PID_DIR/console_${CONSOLE_PORT}.pid"; fi
  echo "[start] console :$CONSOLE_PORT backend=$DEMO_BACKENDS sage_split=$SAGE_SPLIT pid=$pid log=$LOG_CONSOLE"
}

case "$ACTION" in
  status)  status_all; exit 0 ;;
  stop)    stop_console; stop_render; echo "[stop] done"; exit 0 ;;
  restart) stop_render ;;
  start)   ;;
  *)       echo "用法：bash start_all.sh [start|restart|stop|status]"; exit 1 ;;
esac

echo "[start] $(date -Is)  repo=$(cd "$HERE/.." && pwd)"
ensure_render rs3d  "$RS3D_ROOT"  "$RS3D_PORT"  "$RS3D_GPU"  0 "$RS3D_SPG" "$RS3D_RENDER_SCALE" || { [ "$STRICT_STARTUP" != 1 ] || exit 1; }
ensure_render abotn "$ABOTN_ROOT" "$ABOTN_PORT" "$ABOTN_GPU" 3 "$ABOTN_SPG" || { [ "$STRICT_STARTUP" != 1 ] || exit 1; }
ensure_render sage  "$SAGE_ROOT"  "$SAGE_PORT"  "$SAGE_GPU"  3 "$SAGE_SPG" || { [ "$STRICT_STARTUP" != 1 ] || exit 1; }
ensure_render mall  "$MALL_ROOT"  "$MALL_PORT"  "$MALL_GPU"  0 "$MALL_SPG" || { [ "$STRICT_STARTUP" != 1 ] || exit 1; }
wait_pong || exit 1
if [ "$STRICT_STARTUP" != 1 ]; then stop_console; fi
start_console

echo "[start] 等控制台 listen（habitat 软件 GL 加载 ~2.5min）..."
for i in $(seq 1 100); do
  sleep 3
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m5 "http://127.0.0.1:$CONSOLE_PORT/" 2>/dev/null)" = 200 ] && break
done
echo "[start] 状态："
status_all
grep -E '起不来，跳过' "$LOG_CONSOLE" 2>/dev/null || true
if [ "$STRICT_STARTUP" = 1 ]; then
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m5 "http://127.0.0.1:$CONSOLE_PORT/" 2>/dev/null)" = 200 ] || exit 1
  # Optional backends (e.g. the LightNav sidecar) may be offline. The DLC
  # supervisor checks VLN_REQUIRED_BACKENDS against the actual console assets.
fi
echo "[start] 打开 http://$(hostname -I 2>/dev/null | awk '{print $1}'):$CONSOLE_PORT"
