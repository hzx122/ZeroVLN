#!/usr/bin/env bash
# trainer 节点 entry（fm_grpo_env_trainer 闭环 RL，单卡或 torchrun 多卡）。
# 跑在 DLC PyTorchJob Worker pod（16 PPU）。
#   NPROC=1（默认）：单进程 verl/workers/fm_grpo_env_trainer.py（cuda:0，原单卡 reference）
#   NPROC>1        ：python3 -m torch.distributed.run --nproc_per_node=$NPROC verl/workers/fm_grpo_env_trainer_dist.py
#                   （每 rank cuda:LOCAL_RANK，bench URL 内部 strided 分片，grad all_reduce）
#
# 与 bench job 握手：poll $HC/bench_ready_${EXPR_NAME}.txt 拿到 16 个 bench URL，
# 跑完写 $HC/trainer_done_${EXPR_NAME}（bench 见到即清理退出）。
#
# env（由 submit_fm_grpo_dlc.py 的 heredoc export 进来）：EXPR_NAME + 训练超参（见下方 default）。
set -euo pipefail

WORKSPACE=/workspace
REPO=$WORKSPACE/closed_loop_3dgs_rl
VENV=$REPO/vln_demo/.venv_fm
HC=${HC_DIR:-/data/zerovln/vln_verl/dlc}

EXPR_NAME=${EXPR_NAME:?EXPR_NAME must be exported by submit script}
READY=$HC/bench_ready_${EXPR_NAME}.txt
EPS=$HC/bench_episodes_${EXPR_NAME}.txt
VAL=$HC/bench_val_${EXPR_NAME}.txt
DONE=$HC/trainer_done_${EXPR_NAME}
RC=$HC/trainer_rc_${EXPR_NAME}

# ---- 训练超参（manual §4 同款默认值，submit 脚本可覆盖）----
EPISODES=${EPISODES:-}
VAL_EPISODES=${VAL_EPISODES:-}
OPT_STEPS=${OPT_STEPS:-30}
GROUP_SIZE=${GROUP_SIZE:-8}
NUM_SDE_STEPS=${NUM_SDE_STEPS:-5}
SIGMA=${SIGMA:-0.3}
ACCUM_STEPS=${ACCUM_STEPS:-1}
LR=${LR:-1e-4}
TRAIN_PARAMS=${TRAIN_PARAMS:-fm_head}
EVAL_EVERY=${EVAL_EVERY:-5}
EVAL_TRAIN_SAMPLE=${EVAL_TRAIN_SAMPLE:-10}
MAX_CHUNKS=${MAX_CHUNKS:-8}
HISTORY_LEN=${HISTORY_LEN:-8}
NPROC=${NPROC:-1}
MASTER_PORT=${MASTER_PORT:-29512}
# 多机多卡（NNODES>1）：DLC PyTorchJob pod_count=nnodes 注入 NODE_RANK/MASTER_ADDR/MASTER_PORT
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
NPROC_PER_NODE=${NPROC_PER_NODE:-$NPROC}
# FM ckpt：空 → trainer 默认 fm_baseline_fixed（未 domain-adapt）；正式验 SFT/RL 必须指向 domain-matched SFT ckpt
FM_CKPT=${FM_CKPT:-}

echo "[trainer] EXPR=$EXPR_NAME host=$(hostname) start=$(date -Is) nproc=$NPROC"

# ---- 等 bench 起齐 ----
echo "[trainer] waiting bench_ready..."
until [ -s "$READY" ]; do sleep 10; done
BENCH_URL=$(cat "$READY")
echo "[trainer] bench_url=$BENCH_URL"
if [ -z "$EPISODES" ]; then
  echo "[trainer] EPISODES empty, reading train eps from $EPS"
  until [ -s "$EPS" ]; do sleep 5; done
  EPISODES=$(cat "$EPS")
fi
if [ -z "$VAL_EPISODES" ] && [ -s "$VAL" ]; then VAL_EPISODES=$(cat "$VAL"); fi
echo "[trainer] train_eps=$(echo "$EPISODES" | tr ',' '\n' | wc -l) val_eps=$(echo "$VAL_EPISODES" | tr ',' '\n' | grep -c .)"

# ---- env ----
source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$REPO"

TB=/data/zerovln/vln_verl/tb/${EXPR_NAME}_$(date +%H%M%S)
mkdir -p "$(dirname "$TB")"

CKPT_ARG=()
[ -n "$FM_CKPT" ] && CKPT_ARG=(--ckpt "$FM_CKPT")
DIST_CKPT_ARG=()
[ -n "$FM_CKPT" ] && DIST_CKPT_ARG=(--fm-ckpt "$FM_CKPT")

COMMON_ARGS=(
  --episodes "$EPISODES" --val-episodes "$VAL_EPISODES"
  --opt-steps "$OPT_STEPS" --group-size "$GROUP_SIZE"
  --num-sde-steps "$NUM_SDE_STEPS" --sigma "$SIGMA" --accum-steps "$ACCUM_STEPS"
  --lr "$LR" --train-params "$TRAIN_PARAMS" --eval-every "$EVAL_EVERY"
  --eval-train-sample "$EVAL_TRAIN_SAMPLE"
  --max-chunks "$MAX_CHUNKS" --history-len "$HISTORY_LEN"
  --bench-url "$BENCH_URL" --tb-dir "$TB"
)

# ---- 跑闭环 FM-GRPO trainer ----
set +e
if [ "$NPROC" -gt 1 ]; then
  if [ "$NNODES" -gt 1 ]; then
    TORCHRUN_ARGS=(--nnodes="$NNODES" --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" --nproc_per_node="$NPROC_PER_NODE")
    echo "[trainer] multinode torchrun nnodes=$NNODES node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT nproc=$NPROC_PER_NODE"
  else
    TORCHRUN_ARGS=(--nproc_per_node="$NPROC" --master_port="$MASTER_PORT")
    echo "[trainer] single-node torchrun nproc=$NPROC master_port=$MASTER_PORT (dist trainer)"
  fi
  python3 -m torch.distributed.run "${TORCHRUN_ARGS[@]}" \
    verl/workers/fm_grpo_env_trainer_dist.py \
    "${DIST_CKPT_ARG[@]}" "${COMMON_ARGS[@]}"
else
  echo "[trainer] single-process (original trainer, cuda:0)"
  python3 verl/workers/fm_grpo_env_trainer.py \
    "${CKPT_ARG[@]}" "${COMMON_ARGS[@]}" --device cuda:0
fi
code=$?
set -e

echo "$code" > "$RC"
touch "$DONE"
echo "[trainer] end=$(date -Is) rc=$code tb=$TB"
exit $code
