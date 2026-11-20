#!/usr/bin/env bash
# bench 节点 entry（3DGS render × N + bench × 16 + CPFS 握手）。
# 跑在 DLC PyTorchJob Worker pod（16 PPU）。后端契约（scenes_root / sh_degree /
# render 端口段 / render-url env / DEMO_BACKEND / split / eps / 相机 / DLC 布局）
# 全从**统一配置** verl/dlc/backends.py 读——加后端在那加一条，本脚本零分支改动：
#   rs3d  : 3drs_gs_root  sh0  render port 7036/38/40/42  RS3D_RENDER_URL  (free-roam 无 goal → reward 恒 0)
#   abotn : abotn_gs_root sh3  render port 7037..7052     ABOTN_RENDER_URL (有 goal → 真实 reward)
#   sage  : sage_gs_root  sh3  render port 7060..7075     SAGE_RENDER_URL  (有 goal → 真实 reward，SAGE-3D GVLN)
# 渲染两套高斯分进程分端口（SH degree 按进程固定，load_gs_ply 会断言，见 manual §2）。
#
# 与 trainer job 的握手（跨 pod HTTP，同 VPC）：
#   bench 16 health ok 后 → 写 $HC/bench_ready_${EXPR_NAME}.txt（逗号分隔 16 个 URL）
#   同时 curl :8914/spec 取第一个 scene:episode_id 写 $HC/bench_episodes_${EXPR_NAME}.txt（trainer 不用手查场景名）
#   trainer 跑完 → 写 $HC/trainer_done_${EXPR_NAME}；本脚本 poll 到即 pkill 退出。
#
# env（由 submit 脚本 heredoc export）：
#   EXPR_NAME, BACKEND（rs3d|abotn|sage, 默认 rs3d），
#   ABOTN_SET（abotn 时 poi|in|out, 默认 poi）/ SAGE_SPLIT（sage 时 train|instr|scene|traj, 默认 instr）/ SAGE_EPS_PER_SCENE。
set -euo pipefail

# ---- 路径（迁移机器改这里）----
GV=/opt/navigation-runtime/venv
RS=/workspace/agentic_sim/repo/ABot-Navigation/render_server
REPO=${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
TGT=$REPO/vln_demo/render_libs
DEMO=$REPO/vln_demo
ASIM=/workspace/agentic_sim
HC=${HC_DIR:-/data/zerovln/vln_verl/dlc}

EXPR_NAME=${EXPR_NAME:?EXPR_NAME must be exported}
BACKEND=${BACKEND:-rs3d}
ABOTN_SET=${ABOTN_SET:-poi}          # abotn 用；sage/rs3d 忽略
SAGE_SPLIT=${SAGE_SPLIT:-instr}      # sage 用；abotn/rs3d 忽略
SAGE_EPS_PER_SCENE=${SAGE_EPS_PER_SCENE:-0}
BENCH_REPLICAS=${BENCH_REPLICAS:-16}
GYM_CPU_THREADS=${GYM_CPU_THREADS:-1}
BENCH_ARTIFACT_ROOT=${BENCH_ARTIFACT_ROOT:-/data/zerovln/artifacts/$EXPR_NAME}
BENCH_LOCAL_RUNTIME_ROOT=${BENCH_LOCAL_RUNTIME_ROOT:-/tmp/action_head_rl/$EXPR_NAME/${HOSTNAME:-worker}}
GYM_WORK_ROOT=${GYM_WORK_ROOT:-$BENCH_LOCAL_RUNTIME_ROOT/gym}
export TMPDIR=$BENCH_LOCAL_RUNTIME_ROOT/tmp XDG_CACHE_HOME=$BENCH_LOCAL_RUNTIME_ROOT/cache
export HF_HOME=$XDG_CACHE_HOME/huggingface TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
export TORCHINDUCTOR_CACHE_DIR=$XDG_CACHE_HOME/torchinductor CUDA_CACHE_PATH=$XDG_CACHE_HOME/cuda
export PYTORCH_KERNEL_CACHE_PATH=$XDG_CACHE_HOME/kernels TORCH_EXTENSIONS_DIR=$XDG_CACHE_HOME/extensions
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
IFS=',' read -r -a CLIENT_EXPRS <<< "${BENCH_CLIENT_EXPRS:-}"
if (( ${#CLIENT_EXPRS[@]} > 0 )); then
  (( BENCH_REPLICAS % ${#CLIENT_EXPRS[@]} == 0 )) || exit 2
fi
if ! [[ "$GYM_CPU_THREADS" =~ ^[1-9][0-9]*$ ]]; then
  echo "GYM_CPU_THREADS must be a positive integer" >&2
  exit 2
fi
mkdir -p "$HC"
POD_IP=$(hostname -I | awk '{print $NF}')
READY=$HC/bench_ready_${EXPR_NAME}.txt
EPS=$HC/bench_episodes_${EXPR_NAME}.txt
VAL=$HC/bench_val_${EXPR_NAME}.txt
DONE=$HC/trainer_done_${EXPR_NAME}

# ---- 后端契约：从统一配置 backends.py 注入（零分支，加后端只改 backends.py）----
# 注入变量：SCENES_ROOT / SH_DEG / MAX_SPG / RENDER_URL_ENV / DEMO_BACKEND /
# BENCH_PER_RENDER / RENDER_PORTS(数组) / GPU_GROUPS(数组) / PRELOAD_SCENES / BENCH_EXTRA(数组)。
# ★单卡/render server（同 run_local_abotn.sh manual §2）：render_sim.py 的 GSRender 把
# background/camera_center 硬绑 "cuda"（=进程当前默认 device），SceneManager._load_scene 用
# torch.cuda.set_device(gpu_id) 切场景到不同 GPU——多卡/server 时 set_device 是进程全局态，
# 加载场景A(GPU0)→加载场景B(GPU1)后默认 device 变 1，再渲染A 时 .cuda() 落 GPU1 但高斯在
# GPU0 → 跨设备 CUDA illegal access → render 坏态不出图 → reset 无 rgb → reward=nan
# （abotn_mc/mc2 16 rank×11 场景反复跨卡切，两次都挂此；本地 8+8 单卡/server + 单 ep smoke 不切卡故 OK）。
# 故 abotn/sage 用 16 render server × 1 GPU + 16 bench × 1:1（铺满 16 卡，同本地已验模式）。
eval "$(python3 "$REPO/verl/dlc/backends.py" "$BACKEND" --emit-sh)"
if [[ -n "${BENCH_RENDER_PORTS:-}" ]]; then
  read -r -a RENDER_PORTS <<< "$BENCH_RENDER_PORTS"
  read -r -a GPU_GROUPS <<< "${BENCH_RENDER_GPU_GROUPS:?required with BENCH_RENDER_PORTS}"
  [[ ${#RENDER_PORTS[@]} == ${#GPU_GROUPS[@]} ]] || exit 2
fi
N_RENDER=${#RENDER_PORTS[@]}
CHILD_PIDS=()
cleanup_children() {
  for child in "${CHILD_PIDS[@]}"; do
    kill -TERM -- "-$child" 2>/dev/null || true
  done
  for child in "${CHILD_PIDS[@]}"; do wait "$child" 2>/dev/null || true; done
  rm -f "$READY" "$EPS" "$VAL"
  for client in "${CLIENT_EXPRS[@]}"; do rm -f "$HC/bench_ready_${client}.txt"; done
  rm -rf -- "${RENDER_BUSY_DIR:-/dev/shm/render_busy_${EXPR_NAME}}"
  case "$BENCH_LOCAL_RUNTIME_ROOT" in
    /tmp/action_head_rl/*) rm -rf -- "$BENCH_LOCAL_RUNTIME_ROOT" ;;
  esac
}
trap cleanup_children EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ---- 渲染感知 GPU 占卡器：防 bench 节点 16 卡 renderer 空闲被平台低利用巡检 kill ----
# RENDER_BUSY_DIR：gs_client(rs3d/gs_client.py 单点包了 self._session.post)在每个
# 发往 render server 的 POST 期间写独立 rendering_<port>_<pid>_<request> flag，
# POST 返回后仅删自身文件；gpu_occupier 按端口聚合，渲染中暂停该 GPU 的 matmul，
# 空闲时跑 matmul 把利用率顶上去防 kill。/dev/shm 同 pod 共享 + 快。export 让 gym
# 进程(vln_gym_server.sh→gs_client)继承。关掉则不写 flag，占卡器仍持续占卡(等同原模式)。
RENDER_BUSY_DIR=/dev/shm/render_busy_${EXPR_NAME}
export RENDER_BUSY_DIR
mkdir -p "$RENDER_BUSY_DIR"
# 建 port→gpu 映射(同端口多 GPU 用 | 分隔，rs3d 4GPU/server)：RENDER_PORTS[i] ↔ GPU_GROUPS[i]。
RENDER_PORT_GPU=""
for i in "${!RENDER_PORTS[@]}"; do
  P=${RENDER_PORTS[$i]}; G=${GPU_GROUPS[$i]}
  GPIPE=${G//,/|}
  RENDER_PORT_GPU="${RENDER_PORT_GPU}${P}:${GPIPE},"
done
RENDER_PORT_GPU=${RENDER_PORT_GPU%,}

echo "[bench] EXPR=$EXPR_NAME backend=$BACKEND pod_ip=$POD_IP host=$(hostname) start=$(date -Is)"

# render 日志写 CPFS（持久，pod 退出后可查；原 /tmp/render_*.log pod 退出即丢）
RENDER_LOG_DIR=$BENCH_ARTIFACT_ROOT/logs
mkdir -p "$RENDER_LOG_DIR"

# ---- ① N_RENDER 个 render server（abotn=16×1GPU / rs3d=4×4GPU；PRELOAD_SCENES 非空则启动时预加载全部场景）----
PRELOAD_ARG=()
if [ -n "$PRELOAD_SCENES" ]; then
  PRELOAD_ARG=(--preload_scenes "$PRELOAD_SCENES")
  echo "[bench] preload scenes ($(echo "$PRELOAD_SCENES" | tr ',' '\n' | wc -l)): ${PRELOAD_SCENES}"
fi
for i in "${!RENDER_PORTS[@]}"; do
  P=${RENDER_PORTS[$i]}; G=${GPU_GROUPS[$i]}
  RENDER_LAUNCH=("$RS/render_sim_dynamic.py")
  if (( BENCH_REPLICAS > N_RENDER )); then
    # Reuse bootstrap's single-owner renderer: the upstream scene cache can
    # evict (and delete gaussians from) an object held by another HTTP thread.
    RENDER_LAUNCH=("$REPO/verl/scripts/train/action_head_bootstrap/single_gpu_renderer.py"
                   "$RS/render_sim_dynamic.py")
  fi
  PYTHONPATH="$TGT:$RS/submodules/diff-plane-rasterization:$RS/submodules/simple-knn:$RS" \
    setsid "$GV/bin/python" "${RENDER_LAUNCH[@]}" \
    --scenes_root "$SCENES_ROOT" --port "$P" --gpus "$G" \
    --sh_degree "$SH_DEG" --max_scenes_per_gpu "$MAX_SPG" \
    ${PRELOAD_ARG[@]+"${PRELOAD_ARG[@]}"} > "$RENDER_LOG_DIR/render_${EXPR_NAME}_${P}.log" 2>&1 &
  CHILD_PIDS+=("$!")
  echo "[bench] render $i port=$P gpus=$G sh=$SH_DEG spg=$MAX_SPG preload=${#PRELOAD_ARG[@]} pid=$!"
done

echo "[bench] waiting $N_RENDER render pong（预加载时 /ping 在 app.run 前，启动耗时含 ply 加载）..."
ok=0
waits=0
until [ "$ok" -eq "$N_RENDER" ]; do
  ok=0
  for P in "${RENDER_PORTS[@]}"; do
    curl -s -m 3 "http://127.0.0.1:$P/ping" | grep -q pong && ok=$((ok+1)) || true
  done
  [ "$ok" -lt "$N_RENDER" ] && { waits=$((waits+1)); [ $((waits % 20)) -eq 0 ] && echo "[bench] still waiting render pong ($waits×3s, $ok/$N_RENDER, preload in progress)..."; sleep 3; } || true
done
echo "[bench] $N_RENDER render ready"

# ---- ①.⑤ 预加载校验：各 render server /status 看 loaded_scenes_count（是否真加载了全部场景）----
echo "[bench] render /status (preload verify):"
for P in "${RENDER_PORTS[@]}"; do
  N=$(curl -s -m 5 "http://127.0.0.1:$P/status" 2>/dev/null | python3 -c "import json,sys
try:
    d=json.load(sys.stdin); print(d.get('loaded_scenes_count','?'), '|', d.get('gpu_scene_count',{}))
except Exception as e:
    print('ERR',e)" 2>/dev/null)
  echo "  render $P: loaded_scenes=$N"
done

# ---- ①.⑥ 启动渲染感知 GPU 占卡器（render server 已 ready，gs_client 写 flag 即生效）----
# matmul 占满 16 卡防平台低利用巡检 kill；渲染进行时(render server POST 期间)按 flag 暂停
# 让出 SM，保证出图速度与无占卡器时基本一致。trainer_done 在 §④ 一并 kill。
if [[ -n "$RENDER_PORT_GPU" && "${BENCH_GPU_OCCUPIER:-1}" == 1 ]]; then
  setsid "$GV/bin/python" "$REPO/verl/scripts/train/action_head_sft_data/gpu_occupier.py" \
    --render-busy-dir "$RENDER_BUSY_DIR" --render-port-gpu "$RENDER_PORT_GPU" \
    > "$RENDER_LOG_DIR/gpu_occupier_${EXPR_NAME}.log" 2>&1 &
  OCCUPIER_PID=$!
  CHILD_PIDS+=("$OCCUPIER_PID")
  echo "[bench] gpu_occupier(render-aware) pid=$OCCUPIER_PID port_gpu=$RENDER_PORT_GPU"
else
  OCCUPIER_PID=""
  echo "[bench] gpu_occupier skipped (no render-port-gpu map)"
fi

# ---- ② BENCH_REPLICAS 个独立 stateful gym；超过 renderer 数时 round-robin 共享 renderer ----
BENCH_PORTS=()
for i in $(seq 0 $((BENCH_REPLICAS - 1))); do BENCH_PORTS+=( $((${BENCH_PORT_BASE:-8914} + i)) ); done
for i in $(seq 0 $((BENCH_REPLICAS - 1))); do
  BP=${BENCH_PORTS[$i]}
  if [ "$BENCH_REPLICAS" -le "$N_RENDER" ]; then
    RI=$((i * N_RENDER / BENCH_REPLICAS))
  else
    RI=$((i % N_RENDER))
  fi
  RP=${RENDER_PORTS[$RI]}
  env DEMO_BACKEND="$BACKEND" "$RENDER_URL_ENV=http://127.0.0.1:$RP" \
    ${BENCH_EXTRA[@]+"${BENCH_EXTRA[@]}"} DEMO_HTTP_PORT="$BP" \
    GYM_WORK_DIR="$GYM_WORK_ROOT/$BP" \
    OMP_NUM_THREADS="$GYM_CPU_THREADS" MKL_NUM_THREADS="$GYM_CPU_THREADS" \
    OPENBLAS_NUM_THREADS="$GYM_CPU_THREADS" NUMEXPR_NUM_THREADS="$GYM_CPU_THREADS" \
    setsid bash "$REPO/scripts/demo/vln_gym_server.sh" > "$RENDER_LOG_DIR/vln_gym_${EXPR_NAME}_${BP}.log" 2>&1 &
  CHILD_PIDS+=("$!")
  echo "[bench] bench $i port=$BP render=$RP cpu_threads=$GYM_CPU_THREADS work=$GYM_WORK_ROOT/$BP pid=$!"
done

echo "[bench] waiting $BENCH_REPLICAS bench health..."
ok=0
until [ "$ok" -eq "$BENCH_REPLICAS" ]; do
  ok=0
  for BP in "${BENCH_PORTS[@]}"; do
    curl -s -m 3 "http://127.0.0.1:$BP/health" | grep -q . && ok=$((ok+1)) || true
  done
  [ "$ok" -lt "$BENCH_REPLICAS" ] && sleep 3 || true
done
echo "[bench] $BENCH_REPLICAS bench ready"

# ---- ③ 写 ready 文件 + 从 /spec 切 train/val episode 列表给 trainer ----
# hold out 末 3 scene 作 unseen-scene val（同分布内多留 scene 提稳统计量）；
# rs3d 无 goal 时 val 也 nan，但 trainer 的 info_breakdown 跳过 nan 字段不崩。
URLS=""
for BP in "${BENCH_PORTS[@]}"; do URLS="${URLS}http://${POD_IP}:${BP},"; done
URLS=${URLS%,}
echo "$URLS" > "$READY"
if (( ${#CLIENT_EXPRS[@]} > 0 )); then
  PER_CLIENT=$((BENCH_REPLICAS / ${#CLIENT_EXPRS[@]}))
  for ci in "${!CLIENT_EXPRS[@]}"; do
    CLIENT_URLS=""
    for ((j=ci*PER_CLIENT; j<(ci+1)*PER_CLIENT; j++)); do
      CLIENT_URLS+="http://${POD_IP}:${BENCH_PORTS[$j]},"
    done
    CLIENT_READY="$HC/bench_ready_${CLIENT_EXPRS[$ci]}.txt"
    printf '%s\n' "${CLIENT_URLS%,}" > "${CLIENT_READY}.tmp"
    mv "${CLIENT_READY}.tmp" "$CLIENT_READY"
  done
fi

curl -s -m 10 "http://127.0.0.1:${BENCH_PORTS[0]}/spec" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); sc=d['scenes']; n=len(sc)
train=','.join(f'{s}:{e}' for s in sc[:-3] for e in d['scene_eps'][s]) if n>3 else ','.join(f'{s}:{e}' for s in sc for e in d['scene_eps'][s])
val=','.join(f'{s}:{e}' for s in sc[-3:] for e in d['scene_eps'][s]) if n>3 else ''
open('$EPS','w').write(train); open('$VAL','w').write(val)
nt=train.count(',')+1 if train else 0; nv=val.count(',')+1 if val else 0
print(f'train={nt} val={nv} (scenes={n} val_scenes={min(3,n)})')"
echo "[bench] wrote $READY"
echo "[bench] ready: $URLS"
echo "[bench] train eps: $(head -c 80 $EPS 2>/dev/null)..."
echo "[bench] val eps: $(cat $VAL 2>/dev/null | head -c 80)..."

# ---- ④ 等 trainer 跑完 → 清理退出 ----
echo "[bench] waiting trainer_done..."
if (( ${#CLIENT_EXPRS[@]} > 0 )); then
  for CLIENT_EXPR in "${CLIENT_EXPRS[@]}"; do
    until [ -f "$HC/trainer_done_${CLIENT_EXPR}" ]; do sleep 10; done
  done
else
  until [ -f "$DONE" ]; do sleep 10; done
fi
echo "[bench] trainer_done seen, cleaning up"
echo "[bench] end=$(date -Is)"
