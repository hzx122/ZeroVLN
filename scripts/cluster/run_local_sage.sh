#!/usr/bin/env bash
# 本地 8+8 卡调试启动器（sage 后端 = SAGE-3D GVLN，InteriorGS 高斯 + 真值轨迹）。
#   render：PPU 0-7（8 卡，2 个 render server 各 4 卡，sh_degree 3，端口 7038/7040）
#   bench ：8 个 sage bench，8940..8947（前 4 绑 7038，后 4 绑 7040）
#   trainer：PPU 8（cuda:8，2B FM 单进程）
# 与 run_local_abotn.sh 同构——只是 render 资产 abotn_gs_root→sage_gs_root、
# bench DEMO_BACKEND=abotn→sage + SAGE_RENDER_URL + SAGE_SPLIT、trainer 命令逐字不变
# （trainer 后端无关纯 HTTP，零改）。
#
# ★前置：sage_gs_root 下得有场景 ply（make_scenes_root --split <s> --source nurec）。
#   本脚本自动把 /spec 枚举的场景与「已建 ply 的场景」求交，只用建好的，没建的全滤掉。
#
# ★多轮闭环阻塞（继承 abotn，未解）：多轮 trainer 尚未真机 smoke；abotn 多轮 eval
#   success=0（covariate shift / 动作尺度 / cond）；sage 无 domain-matched SFT ckpt，
#   从 fm_baseline 起 = 已知 success=0。本脚本只验「sage 接进 abotn 同款链路、链路不崩」。
#
# 默认 smoke 参数（快速验链路/val/计时），跑正式用 env 覆盖：
#   OPT_STEPS=200 GROUP_SIZE=16 ACCUM_STEPS=4 EVAL_EVERY=10 bash run_local_sage.sh
set -euo pipefail

# ---- 路径 ----
GV=/opt/navigation-runtime/venv
RS=/workspace/agentic_sim/repo/ABot-Navigation/render_server
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)   # 本脚本所在仓根
TGT=$REPO/vln_demo/render_libs
DEMO=$REPO/vln_demo
VENV=$REPO/vln_demo/.venv_fm
# 后端加载参数从统一配置读（scenes_root/sh_degree/max_scenes_per_gpu/render_url_env），
# 本地拓扑（2 render×4gpu + 8 bench + 端口 7038/7040/8940..）仍是本机调试布局。
eval "$(python3 "$REPO/verl/dlc/backends.py" sage --emit-sh)"
SCENES_ROOT=$SCENES_ROOT            # = /data/storage/.../SAGE-3D/sage_gs_root（来自配置）
SH_DEG=$SH_DEG                      # = 3
MAX_SCENES_PER_GPU=${MAX_SCENES_PER_GPU:-$MAX_SPG}   # 配置 5；env 覆盖
TB_ROOT=/data/zerovln/vln_verl/tb
SAGE_SPLIT=${SAGE_SPLIT:-instr}        # train(929) / instr(15) / scene(15) / traj(14)
# 每场景限量解析的 episode 数（cap 大 split 启动开销）。0=全量（instr/scene/traj
# 小 split 没事）；train(929 场景 × ~2385 task ≈ 2.2M) 必须设 >0，否则 bench 启动
# ~6min + ~2GB RAM/进程。限 N → 每场景只取前 N 条 task（trainer 从这个子池采）。
SAGE_EPS_PER_SCENE=${SAGE_EPS_PER_SCENE:-0}
TRAINER_DEVICE=${TRAINER_DEVICE:-cuda:8}
# smoke 用小数据快速验代码；正式 env 覆盖 TRAIN_LIMIT=0 VAL_LIMIT=0（0=全部已建 ply 场景）
TRAIN_LIMIT=${TRAIN_LIMIT:-4}
VAL_LIMIT=${VAL_LIMIT:-2}

# ---- 训练超参（默认 smoke，env 覆盖跑正式）----
EPISODES=${EPISODES:-}          # 空 → /spec ∩ 已建 ply 自动切 train
VAL_EPISODES=${VAL_EPISODES:-}  # 空 → /spec ∩ 已建 ply 自动切 val（hold out 末 3 scene）
OPT_STEPS=${OPT_STEPS:-2}
GROUP_SIZE=${GROUP_SIZE:-4}
NUM_SDE_STEPS=${NUM_SDE_STEPS:-5}
SIGMA=${SIGMA:-0.3}
ACCUM_STEPS=${ACCUM_STEPS:-1}
LR=${LR:-1e-4}
TRAIN_PARAMS=${TRAIN_PARAMS:-fm_head}
EVAL_EVERY=${EVAL_EVERY:-1}
EVAL_TRAIN_SAMPLE=${EVAL_TRAIN_SAMPLE:-3}
MAX_CHUNKS=${MAX_CHUNKS:-8}
HISTORY_LEN=${HISTORY_LEN:-8}   # 滚动历史帧/waypoint 数（对齐 SocialNav N=8）
FM_CKPT=${FM_CKPT:-}
CKPT_ARG=()
[ -n "$FM_CKPT" ] && CKPT_ARG=(--ckpt "$FM_CKPT")
DIST_CKPT_ARG=()
[ -n "$FM_CKPT" ] && DIST_CKPT_ARG=(--fm-ckpt "$FM_CKPT")
# 多卡分布式 trainer（同 run_local_abotn.sh §8）：NPROC>1 时 torchrun 起 dist trainer，
# GPUs 由 GPUS_FOR_TRAINER 给。GROUP_SIZE 建议 = NPROC（每 rank 1 rollout）。
NPROC=${NPROC:-1}
MASTER_PORT=${MASTER_PORT:-29512}
GPUS_FOR_TRAINER=${GPUS_FOR_TRAINER:-8,9,10,11,12,13,14,15}

RENDER_PORTS=(7038 7040)
GPU_GROUPS=("0" "4")
BENCH_PORTS=(8940 8941 8942 8943 8944 8945 8946 8947)

# 启动前自清残留（别人起的 render 或上次没杀干净的 bench 占着端口段）
pkill -9 -f render_sim_dynamic 2>/dev/null || true
pkill -9 -f "vln_gym.server" 2>/dev/null || true
for p in "${RENDER_PORTS[@]}" "${BENCH_PORTS[@]}"; do fuser -k $p/tcp 2>/dev/null || true; done
sleep 2
TB=$TB_ROOT/sage_local_$(date +%H%M%S)
mkdir -p "$TB_ROOT"

echo "[local] start=$(date -Is) trainer_device=$TRAINER_DEVICE sage_split=$SAGE_SPLIT eps_per_scene=$SAGE_EPS_PER_SCENE tb=$TB"
echo "[local] scenes_root=$SCENES_ROOT  (built scenes: $(ls "$SCENES_ROOT" 2>/dev/null | grep -c '^sage-'))"

# ---- ① 2 render server（PPU 0-7，各 4 卡，sh_degree 3，sage_gs_root）----
for i in 0 1; do
  P=${RENDER_PORTS[$i]}; G=${GPU_GROUPS[$i]}
  PYTHONPATH="$TGT:$RS/submodules/diff-plane-rasterization:$RS/submodules/simple-knn:$RS" \
    setsid "$GV/bin/python" "$RS/render_sim_dynamic.py" \
    --scenes_root "$SCENES_ROOT" --port "$P" --gpus "$G" \
    --sh_degree "$SH_DEG" --max_scenes_per_gpu "$MAX_SCENES_PER_GPU" > /tmp/sage_render_${P}.log 2>&1 &
  echo "[local] render $i port=$P gpus=$G pid=$!"
done
trap 'pkill -f render_sim_dynamic.py 2>/dev/null||true; pkill -f vln_gym.server 2>/dev/null||true; echo "[local] cleaned $(date -Is)"' EXIT

echo "[local] waiting 2 render pong..."
ok=0
until [ "$ok" -eq 2 ]; do
  ok=0
  for P in "${RENDER_PORTS[@]}"; do
    curl -s -m 3 "http://127.0.0.1:$P/ping" | grep -q pong && ok=$((ok+1)) || true
  done
  [ "$ok" -lt 2 ] && sleep 3 || true
done
echo "[local] 2 render ready"

# ---- ② 8 bench（前 4 绑 7038，后 4 绑 7040）----
for i in $(seq 0 7); do
  BP=${BENCH_PORTS[$i]}; RP=${RENDER_PORTS[$((i/4))]}
  DEMO_BACKEND=sage SAGE_SPLIT=$SAGE_SPLIT SAGE_EPS_PER_SCENE=$SAGE_EPS_PER_SCENE \
    SAGE_RENDER_URL=http://127.0.0.1:$RP \
    DEMO_HTTP_PORT=$BP setsid bash "$REPO/scripts/demo/vln_gym_server.sh" > /tmp/vln_gym_sage_${BP}.log 2>&1 &
  echo "[local] bench $i port=$BP render=$RP pid=$!"
done

echo "[local] waiting 8 bench health..."
ok=0
until [ "$ok" -eq 8 ]; do
  ok=0
  for BP in "${BENCH_PORTS[@]}"; do
    curl -s -m 3 "http://127.0.0.1:$BP/health" | grep -q . && ok=$((ok+1)) || true
  done
  [ "$ok" -lt 8 ] && sleep 3 || true
done
echo "[local] 8 bench up, waiting /spec (env loaded, sage 建 occupancy 慢)..."
SPEC=""
for i in $(seq 1 60); do
  SPEC=$(curl -s -m 5 "http://127.0.0.1:${BENCH_PORTS[0]}/spec" 2>/dev/null || true)
  if [ -n "$SPEC" ] && echo "$SPEC" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('scenes') else 1)" 2>/dev/null; then
    echo "[local] /spec ready after ${i}x3s"; break
  fi
  sleep 3
done
if [ -z "$SPEC" ] || ! echo "$SPEC" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('scenes') else 1)" 2>/dev/null; then
  echo "[local] /spec empty after 180s, bench env load failed. tail /tmp/vln_gym_sage_${BENCH_PORTS[0]}.log:"; tail -20 /tmp/vln_gym_sage_${BENCH_PORTS[0]}.log; exit 1
fi

# ---- ③ /spec ∩ 已建 ply 场景 → 切 train/val（hold out 末 3 scene，smoke 按 TRAIN/VAL_LIMIT 取前 N）----
# sage /spec 枚举的是 SageSet 轨迹场景（按 split），未必都有 ply；只保留 gs_root 下建好的。
TRAIN=$(echo "$SPEC" | python3 -c "
import json,sys,os
d=json.load(sys.stdin); sc=d['scenes']
built=[s for s in sc if os.path.exists('$SCENES_ROOT/'+s+'/gs/0/'+s+'.ply')]
if not built: sys.stderr.write('[local] WARN: /spec 场景都没建 ply，检查 make_scenes_root\\n'); built=sc
n=len(built)
eps=lambda s: d['scene_eps'].get(s, [d['scene_eps'][s][0] if d['scene_eps'].get(s) else '0'])[0] if d['scene_eps'].get(s) else '0'
t=[f'{s}:{e}' for s in built[:-3] for e in d['scene_eps'][s]] if n>3 else [f'{s}:{e}' for s in built for e in d['scene_eps'][s]]
t=t if ${TRAIN_LIMIT}<=0 else t[:${TRAIN_LIMIT}]
print(','.join(t))")
VAL=$(echo "$SPEC" | python3 -c "
import json,sys,os
d=json.load(sys.stdin); sc=d['scenes']
built=[s for s in sc if os.path.exists('$SCENES_ROOT/'+s+'/gs/0/'+s+'.ply')]
n=len(built)
v=[f'{s}:{e}' for s in built[-3:] for e in d['scene_eps'][s]] if n>3 else []
v=v if ${VAL_LIMIT}<=0 else v[:${VAL_LIMIT}]
print(','.join(v))")
[ -z "$EPISODES" ] && EPISODES="$TRAIN"
[ -z "$VAL_EPISODES" ] && VAL_EPISODES="$VAL"
echo "[local] train_eps=$(echo "$EPISODES"|tr ',' '\n'|wc -l) val_eps=$(echo "$VAL_EPISODES"|tr ',' '\n'|grep -c .)"
echo "[local] train sample: $(echo "$EPISODES"|head -c 80)"
echo "[local] val sample:   $(echo "$VAL_EPISODES"|head -c 80)"
if [ -z "$EPISODES" ]; then
  echo "[local] no train episodes (built ply scenes=0?). build first: make_scenes_root --split $SAGE_SPLIT --source nurec"; exit 1
fi

# ---- ④ trainer（.venv_fm，PPU 8）----
source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$REPO"
BENCH_URL=$(IFS=,; echo "${BENCH_PORTS[*]}" | sed 's/\([0-9]\+\)/http:\/\/127.0.0.1:\1/g')
echo "[local] bench_url=$BENCH_URL"

MODE=${MODE:-train}
if [ "$MODE" = "eval_multiturn" ]; then
  # Phase A：多轮 episode eval（验 sage 多轮链路通，不训练；sage 无 SFT → success 预期 0）
  echo "[local] MODE=eval_multiturn max_chunks=$MAX_CHUNKS history_len=$HISTORY_LEN eval_train_sample=$EVAL_TRAIN_SAMPLE device=$TRAINER_DEVICE"
  python3 verl/workers/fm_grpo_env_trainer.py \
    --eval-multiturn --max-chunks "$MAX_CHUNKS" --history-len "$HISTORY_LEN" \
    "${CKPT_ARG[@]}" \
    --episodes "$EPISODES" --val-episodes "$VAL_EPISODES" \
    --eval-train-sample "$EVAL_TRAIN_SAMPLE" \
    --device "$TRAINER_DEVICE" --bench-url "$BENCH_URL" --tb-dir "$TB"
  echo "[local] end=$(date -Is) tb=$TB"
  exit 0
fi

COMMON_ARGS=(
  --episodes "$EPISODES" --val-episodes "$VAL_EPISODES"
  --opt-steps "$OPT_STEPS" --group-size "$GROUP_SIZE"
  --num-sde-steps "$NUM_SDE_STEPS" --sigma "$SIGMA" --accum-steps "$ACCUM_STEPS"
  --lr "$LR" --train-params "$TRAIN_PARAMS" --eval-every "$EVAL_EVERY"
  --eval-train-sample "$EVAL_TRAIN_SAMPLE"
  --max-chunks "$MAX_CHUNKS" --history-len "$HISTORY_LEN"
  --bench-url "$BENCH_URL" --tb-dir "$TB"
)
if [ "$NPROC" -gt 1 ]; then
  # 多卡分布式 trainer（同 run_local_abotn.sh §8）：torchrun + 手动 all_reduce（不套 DDP），
  # bench URL 数 ≥ NPROC，GROUP_SIZE 是 NPROC 的倍数。trainer 后端无关纯 HTTP，dist 版与
  # 单卡版共用 bench。
  echo "[local] running dist trainer: nproc=$NPROC gpus=$GPUS_FOR_TRAINER opt=$OPT_STEPS group=$GROUP_SIZE accum=$ACCUM_STEPS max_chunks=$MAX_CHUNKS history=$HISTORY_LEN eval=$EVAL_EVERY"
  CUDA_VISIBLE_DEVICES="$GPUS_FOR_TRAINER" python3 -m torch.distributed.run --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    verl/workers/fm_grpo_env_trainer_dist.py "${DIST_CKPT_ARG[@]}" "${COMMON_ARGS[@]}"
else
  echo "[local] running trainer(multiturn): opt=$OPT_STEPS group=$GROUP_SIZE accum=$ACCUM_STEPS max_chunks=$MAX_CHUNKS history_len=$HISTORY_LEN eval=$EVAL_EVERY device=$TRAINER_DEVICE"
  python3 verl/workers/fm_grpo_env_trainer.py "${CKPT_ARG[@]}" "${COMMON_ARGS[@]}" --device "$TRAINER_DEVICE"
fi
echo "[local] end=$(date -Is) tb=$TB"
