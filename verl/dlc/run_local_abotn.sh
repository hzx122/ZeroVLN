#!/usr/bin/env bash
# 本地 8+8 卡调试启动器（abotn 后端，abotn = PPU 3DGS + 有 goal）。
#   render：PPU 0-7（8 卡，2 个 render server 各 4 卡，sh_degree 3，端口 7037/7039）
#   bench ：8 个 rs3d-bench? abotn bench，8914..8921（前 4 绑 7037，后 4 绑 7039）
#   trainer：PPU 8（cuda:8，2B FM 单进程，占 8-15 但只用 1 张）
# 单脚本：起 render+bench → /spec 切 train/val（hold out 末 3 scene）→ 跑 trainer。
# trap EXIT pkill render+bench，Ctrl-C 或 trainer 结束即清理。
#
# 默认 smoke 参数（快速验新指标/val/计时），跑正式用 env 覆盖：
#   OPT_STEPS=200 GROUP_SIZE=16 ACCUM_STEPS=4 EVAL_EVERY=10 bash run_local_abotn.sh
set -euo pipefail

# ---- 路径 ----
GV=/opt/navigation-runtime/venv
RS=/workspace/agentic_sim/repo/ABot-Navigation/render_server
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)   # 本脚本所在仓根
TGT=$REPO/vln_demo/render_libs
DEMO=$REPO/vln_demo
VENV=$REPO/vln_demo/.venv_fm
SCENES_ROOT=/workspace/agentic_sim/abotn_gs_root
TB_ROOT=/data/zerovln/vln_verl/tb
ABOTN_SET=${ABOTN_SET:-poi}
export ABOTN_BASE_CONVENTION=${ABOTN_BASE_CONVENTION:-1}
TRAINER_DEVICE=${TRAINER_DEVICE:-cuda:8}
# smoke 用小数据快速验代码；正式 env 覆盖 TRAIN_LIMIT=0 VAL_LIMIT=0（0=全 /spec）
TRAIN_LIMIT=${TRAIN_LIMIT:-4}
VAL_LIMIT=${VAL_LIMIT:-2}

# ---- 训练超参（默认 smoke，env 覆盖跑正式）----
EPISODES=${EPISODES:-}          # 空 → /spec 自动切 train
VAL_EPISODES=${VAL_EPISODES:-}  # 空 → /spec 自动切 val
OPT_STEPS=${OPT_STEPS:-2}
GROUP_SIZE=${GROUP_SIZE:-4}
NUM_SDE_STEPS=${NUM_SDE_STEPS:-5}
SIGMA=${SIGMA:-0.3}
ACCUM_STEPS=${ACCUM_STEPS:-1}
LR=${LR:-1e-4}
TRAIN_PARAMS=${TRAIN_PARAMS:-fm_head}
EVAL_EVERY=${EVAL_EVERY:-1}
EVAL_TRAIN_SAMPLE=${EVAL_TRAIN_SAMPLE:-3}
# 多轮 episode 每条 rollout 最大交互轮数（train rollout + eval_multiturn 共用）。
# 训练默认 8（每轮 ~32 点 ~3m，8 轮 ~24m 够短程）；正式 eval 可 MAX_CHUNKS=16。
MAX_CHUNKS=${MAX_CHUNKS:-8}
HISTORY_LEN=${HISTORY_LEN:-8}   # 滚动历史帧/waypoint 数（对齐 SocialNav N=8）
# FM checkpoint：空 → trainer 默认 fm_baseline_fixed（未 domain-adapt）；
# 正式验 SFT/RL 必须显式指向 domain-matched SFT ckpt（如 abotn_sft_*/checkpoint-2000）
FM_CKPT=${FM_CKPT:-}
CKPT_ARG=()
[ -n "$FM_CKPT" ] && CKPT_ARG=(--ckpt "$FM_CKPT")
DIST_CKPT_ARG=()
[ -n "$FM_CKPT" ] && DIST_CKPT_ARG=(--fm-ckpt "$FM_CKPT")
# 多卡分布式 trainer：NPROC>1 时 torchrun 起 dist trainer（GPUs 由 GPUS_FOR_TRAINER 给）
NPROC=${NPROC:-1}
MASTER_PORT=${MASTER_PORT:-29512}
GPUS_FOR_TRAINER=${GPUS_FOR_TRAINER:-8,9,10,11,12,13,14,15}

RENDER_PORTS=(7037 7039)
# 单卡/server（manual §2 同款；多卡一个 render server 切 scene 时 PPU CUDA illegal access）
GPU_GROUPS=("0" "4")
BENCH_PORTS=(8940 8941 8942 8943 8944 8945 8946 8947)

# 启动前自清残留（别人起的 http.server 或上次没杀干净的 bench 占着端口段）
pkill -9 -f render_sim_dynamic 2>/dev/null || true
pkill -9 -f "vln_gym.server" 2>/dev/null || true
for p in "${RENDER_PORTS[@]}" "${BENCH_PORTS[@]}"; do fuser -k $p/tcp 2>/dev/null || true; done
sleep 2
TB=$TB_ROOT/abotn_local_$(date +%H%M%S)
mkdir -p "$TB_ROOT"

echo "[local] start=$(date -Is) trainer_device=$TRAINER_DEVICE abotn_set=$ABOTN_SET tb=$TB"

# ---- ① 2 render server（PPU 0-7，各 4 卡，sh_degree 3）----
for i in 0 1; do
  P=${RENDER_PORTS[$i]}; G=${GPU_GROUPS[$i]}
  PYTHONPATH="$TGT:$RS/submodules/diff-plane-rasterization:$RS/submodules/simple-knn:$RS" \
    setsid "$GV/bin/python" "$RS/render_sim_dynamic.py" \
    --scenes_root "$SCENES_ROOT" --port "$P" --gpus "$G" \
    --sh_degree 3 --max_scenes_per_gpu 11 > /tmp/render_${P}.log 2>&1 &
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

# ---- ② 8 bench（前 4 绑 7037，后 4 绑 7039）----
for i in $(seq 0 7); do
  BP=${BENCH_PORTS[$i]}; RP=${RENDER_PORTS[$((i/4))]}
  DEMO_BACKEND=abotn ABOTN_SET=$ABOTN_SET ABOTN_RENDER_URL=http://127.0.0.1:$RP \
    DEMO_HTTP_PORT=$BP setsid bash "$DEMO/vln_gym_server.sh" > /tmp/vln_gym_${BP}.log 2>&1 &
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
echo "[local] 8 bench up, waiting /spec (env loaded, abotn 加载 11 scene navmesh 慢)..."
SPEC=""
for i in $(seq 1 60); do
  SPEC=$(curl -s -m 5 "http://127.0.0.1:${BENCH_PORTS[0]}/spec" 2>/dev/null || true)
  if [ -n "$SPEC" ] && echo "$SPEC" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('scenes') else 1)" 2>/dev/null; then
    echo "[local] /spec ready after ${i}x3s"; break
  fi
  sleep 3
done
if [ -z "$SPEC" ] || ! echo "$SPEC" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('scenes') else 1)" 2>/dev/null; then
  echo "[local] /spec empty after 180s, bench env load failed. tail /tmp/vln_gym_${BENCH_PORTS[0]}.log:"; tail -20 /tmp/vln_gym_${BENCH_PORTS[0]}.log; exit 1
fi

# ---- ③ /spec 切 train/val（hold out 末 3 scene，smoke 时按 TRAIN/VAL_LIMIT 取前 N）----
TRAIN=$(echo "$SPEC" | python3 -c "import json,sys; d=json.load(sys.stdin); sc=d['scenes']; n=len(sc);
t=[f'{s}:{e}' for s in sc[:-3] for e in d['scene_eps'][s]] if n>3 else [f'{s}:{e}' for s in sc for e in d['scene_eps'][s]];
print(','.join(t if ${TRAIN_LIMIT}<=0 else t[:${TRAIN_LIMIT}]))")
VAL=$(echo "$SPEC" | python3 -c "import json,sys; d=json.load(sys.stdin); sc=d['scenes']; n=len(sc);
v=[f'{s}:{e}' for s in sc[-3:] for e in d['scene_eps'][s]] if n>3 else [];
print(','.join(v if ${VAL_LIMIT}<=0 else v[:${VAL_LIMIT}]))")
[ -z "$EPISODES" ] && EPISODES="$TRAIN"
[ -z "$VAL_EPISODES" ] && VAL_EPISODES="$VAL"
echo "[local] train_eps=$(echo "$EPISODES"|tr ',' '\n'|wc -l) val_eps=$(echo "$VAL_EPISODES"|tr ',' '\n'|grep -c .)"
echo "[local] val sample: $(echo "$VAL_EPISODES"|head -c 80)"

# ---- ④ trainer（.venv_fm，PPU 8）----
source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$REPO"
BENCH_URL=$(IFS=,; echo "${BENCH_PORTS[*]}" | sed 's/\([0-9]\+\)/http:\/\/127.0.0.1:\1/g')
echo "[local] bench_url=$BENCH_URL"

MODE=${MODE:-train}
if [ "$MODE" = "eval_multiturn" ]; then
  # Phase A：多轮 episode eval（验 SFT 多轮 success 非零，不训练）
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
  echo "[local] running dist trainer: nproc=$NPROC gpus=$GPUS_FOR_TRAINER opt=$OPT_STEPS group=$GROUP_SIZE accum=$ACCUM_STEPS max_chunks=$MAX_CHUNKS history=$HISTORY_LEN eval=$EVAL_EVERY"
  CUDA_VISIBLE_DEVICES="$GPUS_FOR_TRAINER" python3 -m torch.distributed.run --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    verl/workers/fm_grpo_env_trainer_dist.py "${DIST_CKPT_ARG[@]}" "${COMMON_ARGS[@]}"
else
  echo "[local] running trainer(multiturn): opt=$OPT_STEPS group=$GROUP_SIZE accum=$ACCUM_STEPS max_chunks=$MAX_CHUNKS history_len=$HISTORY_LEN eval=$EVAL_EVERY device=$TRAINER_DEVICE"
  python3 verl/workers/fm_grpo_env_trainer.py "${CKPT_ARG[@]}" "${COMMON_ARGS[@]}" --device "$TRAINER_DEVICE"
fi
echo "[local] end=$(date -Is) tb=$TB"
