#!/usr/bin/env bash
# VLN goal-point GRPO DLC RayJob entry.
# Prepares the vendored verl on all alive Ray nodes, then runs grpo_vln_goal.sh.
# Derived from VLA-RLVR/scripts/dlc/causal_mcq_rl_entry.sh (single-turn, no traj-token,
# no judge, no agent-loop; date_mfm paths with third-party/verl).
set -xeu

WORKSPACE=/workspace
REPO=${DATE_MFM_REPO:-${WORKSPACE}/date_mfm}
VLA_ROOT=${REPO}/projects/MegaTE/vla
VERL_ROOT=${REPO}/third-party/verl
VENV=${WORKSPACE}/envs/megate_py312
VENV_SITE=${VENV}/lib/python3.12/site-packages

source "${VENV}/bin/activate"
export PYTHONPATH="${REPO}:${VERL_ROOT}:${VLA_ROOT}:${VENV_SITE}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${VENV}/local_libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export REPO VENV VENV_SITE VERL_ROOT

# Prepare vendored verl on every alive Ray node.
# Shared cpfs venv: editable install (pip install -e) runs ONCE on head, visible to all
# nodes via the shared venv. Concurrent writes to the same site-packages .pth from other
# nodes race the .pth mid-replace -> OSError [Errno 116] Stale file handle -> setup crash.
# Other nodes only do an import check, no concurrent pip install.
python3 - <<'PY'
import os
import subprocess
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

repo = os.environ["REPO"]
venv = os.environ["VENV"]
venv_site = os.environ["VENV_SITE"]
verl_root = os.environ["VERL_ROOT"]
vla_root = repo + "/projects/MegaTE/vla"
ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")

_common = f"""
set -xeu
source {venv}/bin/activate
cd {repo}
export PYTHONPATH={repo}:{verl_root}:{vla_root}:{venv_site}:${{PYTHONPATH:-}}
export LD_LIBRARY_PATH={ld_library_path}:${{LD_LIBRARY_PATH:-}}
"""
_verify = """
python3 -c "import verl; print('verl:', verl.__file__)"
"""
install_cmd = _common + f"pip install --no-deps -e {verl_root}\n" + _verify
verify_cmd = _common + _verify


def setup(cmd):
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")
    return os.uname().nodename


setup(install_cmd)  # head: install into shared venv once (visible to all nodes)
ray.init(address="auto")


@ray.remote(num_cpus=0)
def remote_setup():
    return setup(verify_cmd)  # other nodes: import check only, no concurrent venv write


refs = [
    remote_setup.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=node["NodeID"],
            soft=False,
        )
    ).remote()
    for node in ray.nodes()
    if node["Alive"]
]
print(ray.get(refs))
ray.shutdown()
PY

cd "${REPO}"
# Defaults (overridable via bootstrap export). Single-turn goal point: short response.
DATA_ROOT="/data/zerovln/vln_verl/data"
export TRAIN_FILE="${TRAIN_FILE:-${DATA_ROOT}/vln_goal_rl_train.jsonl}"
export VAL_FILE="${VAL_FILE:-${TRAIN_FILE}}"
export NNODES="${NNODES:-1}"
export NUM_GPUS="${NUM_GPUS:-16}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
export TRAIN_MINI_BATCH_SIZE="${TRAIN_MINI_BATCH_SIZE:-32}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-128}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-16384}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-234}"
export SAVE_FREQ="${SAVE_FREQ:-30}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-64}"
export TEST_FREQ="${TEST_FREQ:-30}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}"
export MODEL_BASE="${MODEL_BASE:-/data/shared/groups/vla/open_model_weights/Qwen3-VL-2B-Instruct-vla-fm}"
export REWARD_PATH="${REWARD_PATH:-${VLA_ROOT}/verl_nav/reward/vln_goal_reward.py}"
export REWARD_NAME="${REWARD_NAME:-compute_score}"

bash projects/MegaTE/vla/verl_nav/scripts/grpo_vln_goal.sh
