#!/usr/bin/env bash
# VLN goal-point GRPO (Phase 1): single-turn, image-conditioned, distance reward.
# Policy = Qwen3-VL-2B-vla-fm; reward = parse "goal point: [x,y]" -> gaussian L2 on
# a [0,1000) canvas (verl_nav/reward/vln_goal_reward.py). No agent-loop, no judge.
# Derived from VLA-RLVR/scripts/verl/grpo_causal_mcq_twoturn.sh (stripped to single-turn).
set -xeu

WORKSPACE=/workspace
# date_mfm repo root (holds third-party/verl + verl_nav)
REPO_ROOT="${DATE_MFM_REPO:-${WORKSPACE}/date_mfm}"
VLA_ROOT="${REPO_ROOT}/projects/MegaTE/vla"
VERL_ROOT="${VERL_ROOT:-${REPO_ROOT}/third-party/verl}"
VENV="${VENV:-${WORKSPACE}/envs/megate_py312}"
VENV_SITE="${VENV}/lib/python3.12/site-packages"

source "${VENV}/bin/activate"
cd "${REPO_ROOT}"

# PPU shared-object shims (mirror VLA-RLVR entry); harmless if absent (e.g. on GPU).
GLOBAL_SITE="/usr/local/lib/python3.12/site-packages"
mkdir -p "${VENV_SITE}/lib/th_op" "${VENV_SITE}/include"
if [ -f "${GLOBAL_SITE}/lib/libacext.so" ]; then
    ln -sfn "${GLOBAL_SITE}/lib/libacext.so" "${VENV_SITE}/lib/libacext.so"
    ln -sfn "${GLOBAL_SITE}/lib/libacext_op.so" "${VENV_SITE}/lib/libacext_op.so"
    ln -sfn "${GLOBAL_SITE}/lib/th_op/libth_op.so" "${VENV_SITE}/lib/th_op/libth_op.so"
    ln -sfn "${GLOBAL_SITE}/include/acext" "${VENV_SITE}/include/acext"
fi

# verl first on PYTHONPATH so the vendored copy wins; VLA_ROOT for MegaTE data utils.
export PYTHONPATH="${REPO_ROOT}:${VERL_ROOT}:${VLA_ROOT}:${VENV_SITE}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${VENV}/local_libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
export VERL_SKIP_BAD_VISION_SAMPLE="${VERL_SKIP_BAD_VISION_SAMPLE:-1}"

python3 -c "import verl; print('verl:', verl.__file__)"

# --- data (verl-native jsonl from build_vln_rl_data.py) ---
DATA_ROOT="${DATA_ROOT:-/data/zerovln/vln_verl/data}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_ROOT}/vln_goal_rl_train.jsonl}"
VAL_FILE="${VAL_FILE:-${TRAIN_FILE}}"
train_path="[\"${TRAIN_FILE}\"]"
test_path="[\"${VAL_FILE}\"]"

# --- policy base (Qwen3-VL-2B-vla-fm SFT ckpt; HF format) ---
MODEL_BASE="${MODEL_BASE:-/data/shared/groups/vla/open_model_weights/Qwen3-VL-2B-Instruct-vla-fm}"

# --- reward ---
REWARD_PATH="${REWARD_PATH:-${VLA_ROOT}/verl_nav/reward/vln_goal_reward.py}"
REWARD_NAME="${REWARD_NAME:-compute_score}"
REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-16}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
PROJECT_NAME="VLN-GoalPoint-RL-Qwen3VL-2B"
EXPR_NAME="${EXPR_NAME:-GRPO_VLNgoal_${TIMESTAMP}}"

RUN_DIR1=/workspace/verl_run
RUN_DIR2=/data/zerovln/verl_run
DEFAULT_LOCAL_DIR="${RUN_DIR2}/checkpoints/${PROJECT_NAME}/${EXPR_NAME}"
ROLLOUT_DATA_DIR="${RUN_DIR1}/log/${PROJECT_NAME}/${EXPR_NAME}/rollout"
VALIDATION_DATA_DIR="${RUN_DIR1}/log/${PROJECT_NAME}/${EXPR_NAME}/validation"
export TENSORBOARD_DIR="${RUN_DIR1}/tensorboard_log/${PROJECT_NAME}/${EXPR_NAME}"

# --- generation lengths (single-turn goal point: short response) ---
max_prompt_length="${MAX_PROMPT_LENGTH:-8192}"      # 8 image frames dominate the prompt
max_response_length="${MAX_RESPONSE_LENGTH:-128}"   # "goal point: [x, y]; <flow_matching_policy>"
temperature=1.0
top_p=0.95
top_k=-1
val_temperature="${VAL_TEMPERATURE:-1.0}"
val_top_p="${VAL_TOP_P:-0.95}"
n="${ROLLOUT_N:-8}"
val_n="${VAL_N:-1}"

train_batch_size="${TRAIN_BATCH_SIZE:-32}"
val_batch_size="${VAL_BATCH_SIZE:-32}"
train_mini_batch_size="${TRAIN_MINI_BATCH_SIZE:-32}"
train_max_samples="${TRAIN_MAX_SAMPLES:-100000}"
val_max_samples="${VAL_MAX_SAMPLES:-64}"

# --- optim / loss (GRPO defaults from causal-mcq run) ---
loss_mode=vanilla
clip_ratio_low=0.2
clip_ratio_high=0.28
clip_ratio_c=3.0
entropy_coeff=0
grad_clip=1.0
loss_agg_mode=token-mean
use_kl_in_reward=False
kl_coef=0.04
use_kl_loss=True
kl_loss_coef=0.001
weight_decay=0.01
lr_scheduler_type=constant
lr_warmup_steps_ratio=0.0
lora_rank=0
lora_alpha=32

rollout_tp="${ROLLOUT_TP:-1}"
train_sp="${TRAIN_SP:-1}"
use_remove_padding="${USE_REMOVE_PADDING:-False}"
use_dynamic_bsz=False
micro_batch_size_per_gpu=1
max_token_len=$((max_prompt_length + max_response_length))
actor_ppo_max_token_len=$(( max_token_len / train_sp ))
infer_ppo_max_token_len=$(( max_token_len / train_sp ))
rollout_max_num_batched_tokens=${max_token_len}
rollout_max_model_len="${ROLLOUT_MAX_MODEL_LEN:-16384}"
rollout_gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}"
offload=True
fsdp_model_dtype="${FSDP_MODEL_DTYPE:-bf16}"
nnodes="${NNODES:-1}"
num_gpus="${NUM_GPUS:-16}"
fsdp_size="${FSDP_SIZE:-${num_gpus}}"
save_freq="${SAVE_FREQ:-10}"
test_freq="${TEST_FREQ:--1}"
val_before_train="${VAL_BEFORE_TRAIN:-False}"
val_only="${VAL_ONLY:-False}"
total_epochs="${TOTAL_EPOCHS:-3}"
total_training_steps="${TOTAL_TRAINING_STEPS:-940}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    data.train_files=${train_path} \
    data.val_files=${test_path} \
    data.train_max_samples=${train_max_samples} \
    data.val_max_samples=${val_max_samples} \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=${val_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.image_key=images \
    data.video_key=videos \
    data.trust_remote_code=True \
    reward.custom_reward_function.path="${REWARD_PATH}" \
    reward.custom_reward_function.name=${REWARD_NAME} \
    reward.num_workers=${REWARD_NUM_WORKERS} \
    actor_rollout_ref.model.path=${MODEL_BASE} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    actor_rollout_ref.actor.optim.lr_scheduler_type=${lr_scheduler_type} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${lr_warmup_steps_ratio} \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora.rank=${lora_rank} \
    actor_rollout_ref.model.lora.alpha=${lora_alpha} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.grad_clip=${grad_clip} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=${fsdp_model_dtype} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${train_sp} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${train_sp} \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n} \
    actor_rollout_ref.rollout.n=${n} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.ref.fsdp_config.model_dtype=${fsdp_model_dtype} \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPR_NAME} \
    trainer.default_local_dir=${DEFAULT_LOCAL_DIR} \
    trainer.rollout_data_dir=${ROLLOUT_DATA_DIR} \
    trainer.validation_data_dir=${VALIDATION_DATA_DIR} \
    +ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR=${TENSORBOARD_DIR} \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="${PYTHONPATH}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.CUDA_VISIBLE_DEVICES="\"${CUDA_VISIBLE_DEVICES}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="\"${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_SKIP_BAD_VISION_SAMPLE="\"${VERL_SKIP_BAD_VISION_SAMPLE}\"" \
    trainer.n_gpus_per_node=${num_gpus} \
    trainer.nnodes=${nnodes} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.val_before_train=${val_before_train} \
    trainer.val_only=${val_only} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps}
