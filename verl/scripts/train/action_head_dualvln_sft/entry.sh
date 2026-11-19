#!/usr/bin/env bash
# Shared single-/multi-node torchrun entry for VERL-native Action Head SFT.
set -euo pipefail

ACTION_HEAD_AS_TOOL_VERSION=v1
while (($#)); do
  case "$1" in
    --action-head-as-tool-version) ACTION_HEAD_AS_TOOL_VERSION=${2:?}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$ACTION_HEAD_AS_TOOL_VERSION" == v1 || "$ACTION_HEAD_AS_TOOL_VERSION" == v2 || "$ACTION_HEAD_AS_TOOL_VERSION" == v3 || "$ACTION_HEAD_AS_TOOL_VERSION" == v4.3 || "$ACTION_HEAD_AS_TOOL_VERSION" == v6 || "$ACTION_HEAD_AS_TOOL_VERSION" == v8 ]] || exit 2

REPO=${ACTION_HEAD_CODE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}
VERL_REPO=${ACTION_HEAD_VERL_REPO:?Set the installed VERL source root}
ENV_ROOT=${ENV_ROOT:-$REPO/.venv}
# The isolated venv contains optional PPU wheels that are not ABI-safe in all
# DLC images. The base image's Python 3.12 already carries the validated
# Torch/Transformers/DeepSpeed stack used by the successful 35B smoke.
PYTHON=${PYTHON:-python3}
PPU_CAUSAL_SITE=${PPU_CAUSAL_SITE:-$ENV_ROOT/ppu_site_causal_conv1d_1_6_2_post1}

NNODES=${NNODES:-${WORLD_SIZE:-1}}
# DLC PyTorchJob injects RANK/WORLD_SIZE at node granularity. This matches the
# known-good VLA-RLVR multinode SFT entry; PET_NODE_RANK and hostname parsing
# remain compatibility fallbacks.
NODE_RANK=${NODE_RANK:-${RANK:-${PET_NODE_RANK:-}}}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29518}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
GATE=${GATE:-train}
MANIFEST=${MANIFEST:-/data/zerovln/action_head/sft_data/fm_head_joint_v2_smoke_20260826/bundles/manifest.jsonl}
SLOW_MODEL=${SLOW_MODEL:-/opt/external/user-e7a01947/Model/Qwen3.6-35B-A3B}
HEAD_CHECKPOINT=${HEAD_CHECKPOINT:-/data/zerovln/action_head/integration/action_head_wan22_init_v1}
QUERY_CHECKPOINT=${QUERY_CHECKPOINT:-/data/zerovln/action_head/integration/qwen36_four_query_5001_migration_20260824/query}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
MAX_STEPS=${MAX_STEPS:-2}
TOTAL_EPOCHS=${ACTION_HEAD_TOTAL_EPOCHS:-$MAX_STEPS}
[[ "$TOTAL_EPOCHS" =~ ^[1-9][0-9]*$ ]] || {
  echo "ACTION_HEAD_TOTAL_EPOCHS must be a positive integer" >&2; exit 2;
}
SAVE_FREQ=${SAVE_FREQ:--1}
VAL_FREQ=${VAL_FREQ:-0}
ACTION_HEAD_ANCHORS_PER_GPU=${ACTION_HEAD_ANCHORS_PER_GPU:-1}
ACTION_HEAD_VAL_ANCHORS_PER_GPU=${ACTION_HEAD_VAL_ANCHORS_PER_GPU:-1}
# Resolve complete profiles here so dated launchers cannot silently retain a
# mixture of old and new flags. Custom is reserved for explicit experiments.
export ACTION_HEAD_TRAIN_PROFILE=${ACTION_HEAD_TRAIN_PROFILE:-auto}
if [[ "$ACTION_HEAD_TRAIN_PROFILE" == auto ]]; then
  if [[ ( "$ACTION_HEAD_AS_TOOL_VERSION" == v4.3 || "$ACTION_HEAD_AS_TOOL_VERSION" == v6 || "$ACTION_HEAD_AS_TOOL_VERSION" == v8 ) && "${ACTION_HEAD_CONDITION_MODE:-}" != v5_native ]]; then
    # Keep local and DLC defaults identical; multi-anchor packing is opt-in.
    ACTION_HEAD_TRAIN_PROFILE=suffix_v1
  else
    ACTION_HEAD_TRAIN_PROFILE=custom
  fi
fi
case "$ACTION_HEAD_TRAIN_PROFILE" in
  suffix_v1|packed_v1|legacy)
    [[ ( "$ACTION_HEAD_AS_TOOL_VERSION" == v4.3 || "$ACTION_HEAD_AS_TOOL_VERSION" == v6 || "$ACTION_HEAD_AS_TOOL_VERSION" == v8 ) && "${ACTION_HEAD_CONDITION_MODE:-}" != v5_native ]] || {
      echo "suffix_v1/packed_v1/legacy profiles require v4.3 frozen Qwen training" >&2; exit 2;
    }
    if [[ "$ACTION_HEAD_TRAIN_PROFILE" != legacy ]]; then
      export ACTION_HEAD_QUERY_SUFFIX_BACKWARD=1 ACTION_HEAD_ANCHOR_MICROBATCH=1
      export ACTION_HEAD_CPU_PREFETCH=1 ACTION_HEAD_FM_BATCH_WINDOWS=1
      GRADIENT_CHECKPOINTING=false
    else
      export ACTION_HEAD_QUERY_SUFFIX_BACKWARD=0 ACTION_HEAD_ANCHOR_MICROBATCH=0
      export ACTION_HEAD_CPU_PREFETCH=0 ACTION_HEAD_FM_BATCH_WINDOWS=0
      GRADIENT_CHECKPOINTING=true
    fi
    export ACTION_HEAD_GC_SKIP_EVERY=0 ACTION_HEAD_FSDP_RESIDENT_MAX_TOKENS=0
    export ACTION_HEAD_SLOW_BATCH_MODE=sequential
    if [[ "$ACTION_HEAD_TRAIN_PROFILE" == packed_v1 ]]; then
      export ACTION_HEAD_SLOW_BATCH_MODE=packed ACTION_HEAD_PACK_EXECUTION=layerwise ACTION_HEAD_PACK_CONV_BACKEND=segmented
      export ACTION_HEAD_PACK_ANCHORS=2 ACTION_HEAD_PACK_MAX_TOKENS=16000
    fi
    FSDP_RESHARD_AFTER_FORWARD=true
    ;;
  custom) ;;
  *) echo "ACTION_HEAD_TRAIN_PROFILE must be auto, suffix_v1, packed_v1, legacy or custom" >&2; exit 2 ;;
esac
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}
if [[ "$ACTION_HEAD_AS_TOOL_VERSION" == v6 || "$ACTION_HEAD_AS_TOOL_VERSION" == v8 ]]; then
  FSDP_SIZE=${FSDP_SIZE:-1}
else
  FSDP_SIZE=${FSDP_SIZE:--1}
fi
FSDP_RESHARD_AFTER_FORWARD=${FSDP_RESHARD_AFTER_FORWARD:-true}
if [[ "$ACTION_HEAD_AS_TOOL_VERSION" == v8 && "${ACTION_HEAD_FM_PREFIX_REUSE:-0}" == 1 ]]; then
  # RL FM-stage history cache: plain HF modules, no suffix-backward patch.
  [[ "${ACTION_HEAD_FROZEN_QUERY_BACKEND:-}" == legacy_suffix && "$FSDP_SIZE" == 1 && "${ACTION_HEAD_FLA_L2NORM_DYNAMIC_NB:-1}" == 1 \
     && "${ACTION_HEAD_PREFIX_REUSE:-0}" == 0 && "$ACTION_HEAD_TRAIN_PROFILE" == suffix_v1 \
     && ( "${ACTION_HEAD_EPISODE_SAMPLER:-0}" == 1 || "${ACTION_HEAD_LOCAL_PROFILE:-}" == fm_prefix_chain ) ]] || {
    echo "v8 FM prefix reuse requires legacy_suffix, FSDP_SIZE=1, episode sampler and no MoE prefix reuse" >&2; exit 2;
  }
  export ACTION_HEAD_FM_PREFIX_REUSE=1 ACTION_HEAD_QUERY_SUFFIX_BACKWARD=0
  export ACTION_HEAD_FLA_L2NORM_DYNAMIC_NB=1
elif [[ "$ACTION_HEAD_AS_TOOL_VERSION" == v6 || "$ACTION_HEAD_AS_TOOL_VERSION" == v8 ]]; then
  [[ "${ACTION_HEAD_FROZEN_QUERY_BACKEND:-query_only}" == query_only && "$FSDP_SIZE" == 1 && "${ACTION_HEAD_FLA_L2NORM_DYNAMIC_NB:-1}" == 1 ]] || {
    echo "v6 requires query_only, FSDP_SIZE=1 and dynamic FLA normalization" >&2; exit 2;
  }
  export ACTION_HEAD_FROZEN_QUERY_BACKEND=query_only
  export ACTION_HEAD_FLA_L2NORM_DYNAMIC_NB=1
fi
[[ "$FSDP_RESHARD_AFTER_FORWARD" == true || "$FSDP_RESHARD_AFTER_FORWARD" == false ]] || {
  echo "FSDP_RESHARD_AFTER_FORWARD must be true or false" >&2; exit 2;
}
export ACTION_HEAD_FLA_L2NORM_DYNAMIC_NB=${ACTION_HEAD_FLA_L2NORM_DYNAMIC_NB:-0}
export ACTION_HEAD_FM_BATCH_WINDOWS=${ACTION_HEAD_FM_BATCH_WINDOWS:-0}
export ACTION_HEAD_CPU_PREFETCH=${ACTION_HEAD_CPU_PREFETCH:-0}
export ACTION_HEAD_QUERY_SUFFIX_BACKWARD=${ACTION_HEAD_QUERY_SUFFIX_BACKWARD:-0}
export ACTION_HEAD_ANCHOR_MICROBATCH=${ACTION_HEAD_ANCHOR_MICROBATCH:-0}
export ACTION_HEAD_FROZEN_QUERY_BACKEND=${ACTION_HEAD_FROZEN_QUERY_BACKEND:-legacy_suffix}
export ACTION_HEAD_DETERMINISTIC_ALGORITHMS=${ACTION_HEAD_DETERMINISTIC_ALGORITHMS:-0}
[[ "$ACTION_HEAD_DETERMINISTIC_ALGORITHMS" == 0 || "$ACTION_HEAD_DETERMINISTIC_ALGORITHMS" == 1 ]] || {
  echo "ACTION_HEAD_DETERMINISTIC_ALGORITHMS must be 0 or 1" >&2; exit 2;
}
case "$ACTION_HEAD_FROZEN_QUERY_BACKEND" in
  legacy_suffix) ;;
  query_only)
    if [[ "${ACTION_HEAD_SLOW_BATCH_MODE:-sequential}" != sequential || "$ACTION_HEAD_ANCHOR_MICROBATCH" != 1 || "$GRADIENT_CHECKPOINTING" != false || "${ACTION_HEAD_CONDITION_MODE:-}" == v5_native ]]; then
      echo "query_only requires frozen Qwen, sequential B1 anchor microbatches and GRADIENT_CHECKPOINTING=false" >&2; exit 2
    fi
    ;;
  *) echo "ACTION_HEAD_FROZEN_QUERY_BACKEND must be legacy_suffix or query_only" >&2; exit 2 ;;
esac
if [[ "$ACTION_HEAD_DETERMINISTIC_ALGORITHMS" == 1 ]]; then
  export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
fi
export ACTION_HEAD_PACK_ANCHORS=${ACTION_HEAD_PACK_ANCHORS:-2}
export ACTION_HEAD_PACK_MAX_TOKENS=${ACTION_HEAD_PACK_MAX_TOKENS:-12000}
if [[ "${ACTION_HEAD_SLOW_BATCH_MODE:-sequential}" == packed ]]; then
  [[ "$ACTION_HEAD_PACK_ANCHORS" =~ ^[1-9][0-9]*$ && "$ACTION_HEAD_PACK_MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || exit 2
  if [[ "$ACTION_HEAD_QUERY_SUFFIX_BACKWARD" != 1 || "$ACTION_HEAD_ANCHOR_MICROBATCH" != 1 || "$GRADIENT_CHECKPOINTING" != false ]]; then
    echo "packed anchors require suffix backward, anchor microbatch and GRADIENT_CHECKPOINTING=false" >&2; exit 2
  fi
fi
for action_head_bool in ACTION_HEAD_QUERY_SUFFIX_BACKWARD ACTION_HEAD_ANCHOR_MICROBATCH; do
  [[ "${!action_head_bool}" == 0 || "${!action_head_bool}" == 1 ]] || {
    echo "$action_head_bool must be 0 or 1" >&2; exit 2;
  }
done
export ACTION_HEAD_GC_SKIP_EVERY=${ACTION_HEAD_GC_SKIP_EVERY:-0}
export ACTION_HEAD_GC_MAX_TOKENS=${ACTION_HEAD_GC_MAX_TOKENS:-5120}
[[ "$ACTION_HEAD_GC_SKIP_EVERY" == 0 || "$ACTION_HEAD_GC_SKIP_EVERY" == 2 || "$ACTION_HEAD_GC_SKIP_EVERY" == 4 ]] || {
  echo "ACTION_HEAD_GC_SKIP_EVERY must be 0, 2 or 4" >&2; exit 2;
}
[[ "$ACTION_HEAD_GC_MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || exit 2
if [[ "$ACTION_HEAD_GC_SKIP_EVERY" != 0 && "$GRADIENT_CHECKPOINTING" != true ]]; then
  echo "selective checkpointing requires GRADIENT_CHECKPOINTING=true" >&2; exit 2
fi
[[ "$ACTION_HEAD_CPU_PREFETCH" == 0 || "$ACTION_HEAD_CPU_PREFETCH" == 1 ]] || {
  echo "ACTION_HEAD_CPU_PREFETCH must be 0 or 1" >&2; exit 2;
}
[[ "$ACTION_HEAD_FM_BATCH_WINDOWS" == 0 || "$ACTION_HEAD_FM_BATCH_WINDOWS" == 1 ]] || {
  echo "ACTION_HEAD_FM_BATCH_WINDOWS must be 0 or 1" >&2; exit 2;
}
[[ "$ACTION_HEAD_FLA_L2NORM_DYNAMIC_NB" == 0 || "$ACTION_HEAD_FLA_L2NORM_DYNAMIC_NB" == 1 ]] || {
  echo "ACTION_HEAD_FLA_L2NORM_DYNAMIC_NB must be 0 or 1" >&2; exit 2;
}
[[ "$FSDP_SIZE" == -1 || "$FSDP_SIZE" =~ ^[1-9][0-9]*$ ]] || exit 2
if [[ "$FSDP_SIZE" != -1 ]]; then
  (( (NNODES * NPROC_PER_NODE) % FSDP_SIZE == 0 )) || {
    echo "FSDP_SIZE must divide total training ranks" >&2; exit 2;
  }
fi
[[ "$GRADIENT_CHECKPOINTING" == true || "$GRADIENT_CHECKPOINTING" == false ]] || {
  echo "GRADIENT_CHECKPOINTING must be true or false" >&2; exit 2;
}
echo "[entry] train_profile=$ACTION_HEAD_TRAIN_PROFILE suffix=$ACTION_HEAD_QUERY_SUFFIX_BACKWARD anchor_microbatch=$ACTION_HEAD_ANCHOR_MICROBATCH fm_batch=$ACTION_HEAD_FM_BATCH_WINDOWS cpu_prefetch=$ACTION_HEAD_CPU_PREFETCH gradient_checkpointing=$GRADIENT_CHECKPOINTING reshard=$FSDP_RESHARD_AFTER_FORWARD slow_batch_mode=${ACTION_HEAD_SLOW_BATCH_MODE:-sequential} pack_execution=${ACTION_HEAD_PACK_EXECUTION:-layerwise} pack_anchors=$ACTION_HEAD_PACK_ANCHORS pack_max_tokens=$ACTION_HEAD_PACK_MAX_TOKENS"
echo "[entry] frozen_query_backend=$ACTION_HEAD_FROZEN_QUERY_BACKEND deterministic_algorithms=$ACTION_HEAD_DETERMINISTIC_ALGORITHMS cublas_workspace_config=${CUBLAS_WORKSPACE_CONFIG:-unset} fsdp_size=$FSDP_SIZE"

# VERL exposes only one periodic hook (trainer.save_freq). Use the greatest
# common divisor so the project handler sees every requested save/eval step,
# then let it decide whether to write a snapshot, run validation, or both.
HOOK_FREQ=$SAVE_FREQ
if [[ "$VAL_FREQ" -gt 0 ]]; then
  if [[ "$SAVE_FREQ" -le 0 ]]; then
    HOOK_FREQ=$VAL_FREQ
  else
    GCD_A=$SAVE_FREQ
    GCD_B=$VAL_FREQ
    while [[ "$GCD_B" -ne 0 ]]; do
      GCD_R=$((GCD_A % GCD_B))
      GCD_A=$GCD_B
      GCD_B=$GCD_R
    done
    HOOK_FREQ=$GCD_A
  fi
fi

# The checkpoint handler can select global boundaries before the trainer's
# cache-cleanup hook. Do not shrink HOOK_FREQ to gcd(offset, interval): that
# would force cache cleanup on intermediate non-save steps.
export ACTION_HEAD_SAVE_STEP_BASIS=${ACTION_HEAD_SAVE_STEP_BASIS:-local}
case "$ACTION_HEAD_SAVE_STEP_BASIS" in
  local|global) ;;
  *) echo "ACTION_HEAD_SAVE_STEP_BASIS must be local or global" >&2; exit 2 ;;
esac

# Derive the node index only for nonstandard launchers that expose neither
# DLC's RANK nor PET_NODE_RANK.
POD_HOSTNAME=$(hostname)
if [[ -z "$NODE_RANK" ]]; then
  case "$POD_HOSTNAME" in
    *-master-*) NODE_RANK=0 ;;
    *-worker-*)
      WORKER_INDEX=${POD_HOSTNAME##*-worker-}
      if [[ ! "$WORKER_INDEX" =~ ^[0-9]+$ ]]; then
        echo "cannot derive worker index from hostname: $POD_HOSTNAME" >&2
        exit 2
      fi
      NODE_RANK=$((WORKER_INDEX + 1))
      ;;
    *)
      if [[ "$NNODES" == "1" ]]; then
        NODE_RANK=0
      else
        echo "NODE_RANK is unset and hostname has no DLC node suffix: $POD_HOSTNAME" >&2
        exit 2
      fi
      ;;
  esac
fi
echo "[entry] host=$POD_HOSTNAME nnodes=$NNODES node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT nproc_per_node=$NPROC_PER_NODE"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export ACTION_HEAD_MANIFEST="$MANIFEST"
export ACTION_HEAD_HEAD_CHECKPOINT="$HEAD_CHECKPOINT"
export ACTION_HEAD_QUERY_CHECKPOINT="$QUERY_CHECKPOINT"
export ACTION_HEAD_OUTPUT_DIR="$OUTPUT_DIR"
export ACTION_HEAD_GATE="$GATE"
export ACTION_HEAD_SAVE_FREQ="$SAVE_FREQ"
export ACTION_HEAD_VAL_FREQ="$VAL_FREQ"
export ACTION_HEAD_ANCHORS_PER_GPU
export ACTION_HEAD_VAL_ANCHORS_PER_GPU
# Mirror verl Tracking metrics (train/loss, train/grad_norm, ...) into a
# TensorBoard event dir living next to the checkpoint. verl's
# _TensorboardAdapter reads TENSORBOARD_DIR; validation flow_loss/ade/fde are
# additionally written by ActionHeadVerlSFTTrainer._log_validation_tb.
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-$OUTPUT_DIR/tensorboard}"
if [[ "${USE_PPU_CAUSAL_SITE:-0}" == "1" ]]; then
  export PYTHONPATH="$PPU_CAUSAL_SITE:$VERL_REPO:$REPO/verl:$REPO/scripts/train:$REPO/scripts/eval:$REPO/vln_demo/src"
else
  # The separately built causal-conv1d extension is not ABI-safe in every DLC
  # image. Keep it opt-in; Qwen's torch fallback is the correctness baseline.
  export PYTHONPATH="$VERL_REPO:$REPO/verl:$REPO/scripts/train:$REPO/scripts/eval:$REPO/vln_demo/src"
fi
cd "$REPO"

case "$GATE" in import|load|train) ;; *) echo "GATE must be import, load, or train" >&2; exit 2 ;; esac

# Precompute the frozen frontend once, then release all encoder workers before
# launching cached DDP training in the same allocation.
if [[ "${ACTION_HEAD_NATIVE_PREPARE:-0}" == "1" ]]; then
  [[ "$NNODES" == "1" && "$NPROC_PER_NODE" == "16" && "$GATE" == "train" ]]
  [[ "${ACTION_HEAD_CONDITION_MODE:-}" == "v5_native" && "${ACTION_HEAD_NATIVE_DDP:-0}" == "1" ]]
  "$PYTHON" -m sft.native_feature_dataset prepare \
    --manifest "$MANIFEST" --output "${ACTION_HEAD_NATIVE_CACHE:?}" \
    --model "$SLOW_MODEL" --world 16 \
    --expected-anchors "${ACTION_HEAD_NATIVE_EXPECTED_ANCHORS:?}" \
    --expected-windows "${ACTION_HEAD_NATIVE_EXPECTED_WINDOWS:?}"
fi

# TensorDict's outer dimension remains one packed project batch per rank;
# ACTION_HEAD_ANCHORS_PER_GPU controls the true anchor batch inside it.
EXTRA=()
exec "$PYTHON" -m torch.distributed.run \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  --nproc_per_node="$NPROC_PER_NODE" \
  "$REPO/verl/sft/train_action_head_verl.py" \
  model.path="$SLOW_MODEL" \
  model.external_lib=sft.verl_action_head_engine \
  model.use_remove_padding=false \
  model.enable_gradient_checkpointing="$GRADIENT_CHECKPOINTING" \
  +model.override_config.attn_implementation=sdpa \
  engine=fsdp \
  engine.strategy=fsdp2 \
  engine.fsdp_size="$FSDP_SIZE" \
  engine.reshard_after_forward="$FSDP_RESHARD_AFTER_FORWARD" \
  engine.model_dtype=bf16 \
  optim=fsdp \
  optim.lr="${LEARNING_RATE:-3e-5}" \
  optim.lr_scheduler_type=constant \
  optim.lr_warmup_steps="${ACTION_HEAD_LR_WARMUP_STEPS:-0}" \
  optim.clip_grad="${ACTION_HEAD_CLIP_GRAD:-1.0}" \
  data.train_files="$MANIFEST" \
  data.val_files=null \
  data.train_batch_size="$((NNODES * NPROC_PER_NODE * ACTION_HEAD_ANCHORS_PER_GPU))" \
  data.micro_batch_size_per_gpu=1 \
  data.use_dynamic_bsz=false \
  data.pad_mode=right \
  data.num_workers=0 \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.total_training_steps="$MAX_STEPS" \
  trainer.save_freq="$HOOK_FREQ" \
  trainer.test_freq=-1 \
  trainer.logger="['console']" \
  trainer.project_name=action-head-sft \
  trainer.experiment_name="$(basename "$OUTPUT_DIR")" \
  trainer.nnodes="$NNODES" \
  trainer.n_gpus_per_node="$NPROC_PER_NODE" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.resume_mode=disable \
  checkpoint.save_contents="[]" \
  checkpoint.load_contents="[]" \
  +action_head_as_tool_version="$ACTION_HEAD_AS_TOOL_VERSION" \
  "${EXTRA[@]}"
