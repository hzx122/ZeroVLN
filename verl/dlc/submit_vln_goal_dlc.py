#!/usr/bin/env python3
"""提交 VLN goal-point GRPO RL(Phase 1)的 DLC RayJob。

基于 VLA-RLVR/scripts/dlc/submit_causal_mcq_rl_dlc.py 精简:
- 单轮 goal-point(无 two-turn agent-loop / 无 judge / 无 traj-token)
- **不 git clone**: 直接读共享 CPFS 上的 date_mfm(所有节点可见), 绕开 date.git origin
  里的明文 GitLab PAT, 也省掉分支/commit 校验。bootstrap 只 export env + 跑 entry。
- 默认单节点(nnodes=1, worker pod_count=1)。

用法:
  python3 submit_vln_goal_dlc.py --display-name VLNgoal_smoke --nnodes 1 --dry-run
  python3 submit_vln_goal_dlc.py --display-name VLNgoal_smoke --nnodes 1
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("ALIBABA_CLOUD_CREDENTIALS_URI", "http://localhost:7002/api/v1/credentials/0")

from alibabacloud_credentials.client import Client as CredClient
from alibabacloud_credentials.models import Config as CredConfig
from alibabacloud_pai_dlc20201203.client import Client
from alibabacloud_pai_dlc20201203.models import (
    CreateJobRequest,
    CreateJobRequestDataSources,
    CreateJobRequestUserVpc,
    JobSpec,
    ResourceConfig,
)
from alibabacloud_tea_openapi.models import Config

WORKSPACE_ROOT = Path("/workspace")
DATE_MFM_REPO = WORKSPACE_ROOT / "date_mfm"
VLA_ROOT = DATE_MFM_REPO / "projects/MegaTE/vla"
DLC_CLI_DIR = WORKSPACE_ROOT / "dlc_cli"
if str(DLC_CLI_DIR) not in sys.path:
    sys.path.insert(0, str(DLC_CLI_DIR))

from workspace_config import DLC_ENDPOINT, build_data_sources, resolve_workspace

WORKSPACE = "vla_xpu"
PRIORITY = 7
IMAGE = "registry.example.invalid/aliyun-wl-gpu02/ppu-training:verl-veomni-eval-lite-zhouyang-0518"
GPUS_PER_NODE = 16

DATA_ROOT = Path("/data/zerovln/vln_verl/data")
DEFAULT_TRAIN_FILE = DATA_ROOT / "vln_goal_rl_train.jsonl"
DEFAULT_VAL_FILE = DATA_ROOT / "vln_goal_rl_val.jsonl"
DEFAULT_MODEL_BASE = Path("/data/shared/groups/vla/open_model_weights/Qwen3-VL-2B-Instruct-vla-fm")
DEFAULT_REWARD_PATH = VLA_ROOT / "verl_nav/reward/vln_goal_reward.py"


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def get_dlc_client() -> Client:
    cred_config = CredConfig(type="credentials_uri")
    cred_client = CredClient(cred_config)
    config = Config(credential=cred_client, endpoint=DLC_ENDPOINT)
    return Client(config)


def user_vpc() -> CreateJobRequestUserVpc:
    return CreateJobRequestUserVpc(
        vpc_id="vpc-0jl6vz0vw0bf48e7v8v2j",
        switch_id="vsw-0jl7b93vkb5938rqipb81",
        security_group_id="sg-0jlbiowdxq51f35s1bpp",
        extended_cidrs=[
            "127.0.0.1/25",
            "127.0.0.1/20",
            "127.0.0.1/20",
            "127.0.0.1/20",
            "127.0.0.1/26",
            "127.0.0.1/26",
            "127.0.0.1/26",
            "127.0.0.1/28",
            "127.0.0.1/28",
            "127.0.0.1/19",
        ],
    )


def build_bootstrap(
    display_name: str,
    train_file: str,
    val_file: str,
    nnodes: int,
    model_base: str,
    reward_path: str,
    reward_name: str,
    train_batch_size: int,
    train_mini_batch_size: int,
    train_max_samples: int,
    max_prompt_length: int,
    max_response_length: int,
    rollout_max_model_len: int,
    rollout_n: int,
    total_epochs: int,
    total_training_steps: int,
    save_freq: int,
    test_freq: int,
    val_max_samples: int,
    rollout_gpu_memory_utilization: float,
) -> str:
    log_dir = WORKSPACE_ROOT / "discussion_logs" / "vln_goal_rl_dlc"
    entry = VLA_ROOT / "verl_nav/dlc/vln_goal_entry.sh"
    return f"""#!/usr/bin/env bash
set -xeu

DISPLAY_NAME={shell_quote(display_name)}
TRAIN_FILE={shell_quote(train_file)}
VAL_FILE={shell_quote(val_file)}
LOG_DIR={shell_quote(str(log_dir))}
LOG_FILE="${{LOG_DIR}}/${{DISPLAY_NAME}}.log"

mkdir -p "${{LOG_DIR}}"
exec >> "${{LOG_FILE}}" 2>&1

echo "display_name=${{DISPLAY_NAME}}"
echo "start_time=$(date -Is)"
echo "host=$(hostname)"
nvidia-smi || true
test -s "${{TRAIN_FILE}}"
test -s "${{VAL_FILE}}"

# 直接读共享 CPFS 上的 date_mfm(不 clone, 绕开 origin 明文 PAT); 全节点可见。
export DATE_MFM_REPO={shell_quote(str(DATE_MFM_REPO))}
cd "${{DATE_MFM_REPO}}"
echo "repo_head=$(git rev-parse HEAD 2>/dev/null || echo 'no-git')"

export EXPR_NAME="${{DISPLAY_NAME}}"
export TRAIN_FILE="${{TRAIN_FILE}}"
export VAL_FILE="${{VAL_FILE}}"
export NNODES={nnodes}
export NUM_GPUS={GPUS_PER_NODE}
export MODEL_BASE={shell_quote(model_base)}
export REWARD_PATH={shell_quote(reward_path)}
export REWARD_NAME={shell_quote(reward_name)}
export TRAIN_BATCH_SIZE={train_batch_size}
export TRAIN_MINI_BATCH_SIZE={train_mini_batch_size}
export TRAIN_MAX_SAMPLES={train_max_samples}
export MAX_PROMPT_LENGTH={max_prompt_length}
export MAX_RESPONSE_LENGTH={max_response_length}
export ROLLOUT_MAX_MODEL_LEN={rollout_max_model_len}
export ROLLOUT_N={rollout_n}
export TOTAL_EPOCHS={total_epochs}
export TOTAL_TRAINING_STEPS={total_training_steps}
export SAVE_FREQ={save_freq}
export TEST_FREQ={test_freq}
export VAL_MAX_SAMPLES={val_max_samples}
export ROLLOUT_GPU_MEMORY_UTILIZATION={rollout_gpu_memory_utilization}

bash "{entry}"

echo "end_time=$(date -Is)"
"""


def submit(params: dict, dry_run: bool = False) -> str | None:
    workspace_profile = resolve_workspace(WORKSPACE)
    user_command = build_bootstrap(**params)
    if dry_run:
        print(f"[dry-run] job_type=RayJob nnodes={params['nnodes']} gpus_per_node={GPUS_PER_NODE}")
        print("[dry-run] user_command:\n" + user_command)
        return None

    head_spec = JobSpec(
        type="Head",
        pod_count=1,
        image=IMAGE,
        resource_config=ResourceConfig(cpu="32", memory="128Gi", shared_memory="128Gi", gpu="0"),
    )
    worker_spec = JobSpec(
        type="Worker",
        pod_count=params["nnodes"],
        image=IMAGE,
        resource_config=ResourceConfig(cpu="184", memory="1800Gi", shared_memory="1800Gi", gpu=str(GPUS_PER_NODE)),
    )
    request = CreateJobRequest(
        display_name=params["display_name"],
        job_type="RayJob",
        user_command=user_command,
        data_sources=build_data_sources(workspace_profile, CreateJobRequestDataSources),
        resource_id=workspace_profile.resource_id,
        workspace_id=workspace_profile.workspace_id,
        user_vpc=user_vpc(),
        priority=PRIORITY,
        job_specs=[head_spec, worker_spec],
    )
    return get_dlc_client().create_job(request).body.job_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--train-file", default=str(DEFAULT_TRAIN_FILE))
    parser.add_argument("--val-file", default=str(DEFAULT_VAL_FILE))
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--model-base", default=str(DEFAULT_MODEL_BASE))
    parser.add_argument("--reward-path", default=str(DEFAULT_REWARD_PATH))
    parser.add_argument("--reward-name", default="compute_score")
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--train-mini-batch-size", type=int, default=32)
    parser.add_argument("--train-max-samples", type=int, default=100000)
    parser.add_argument("--max-prompt-length", type=int, default=8192)
    parser.add_argument("--max-response-length", type=int, default=128)
    parser.add_argument("--rollout-max-model-len", type=int, default=16384)
    parser.add_argument("--rollout-n", type=int, default=8)
    parser.add_argument("--total-epochs", type=int, default=3)
    parser.add_argument("--total-training-steps", type=int, default=234)
    parser.add_argument("--save-freq", type=int, default=30)
    parser.add_argument("--test-freq", type=int, default=30)
    parser.add_argument("--val-max-samples", type=int, default=64)
    parser.add_argument("--rollout-gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    params = dict(
        display_name=safe_name(args.display_name),
        train_file=args.train_file,
        val_file=args.val_file,
        nnodes=args.nnodes,
        model_base=args.model_base,
        reward_path=args.reward_path,
        reward_name=args.reward_name,
        train_batch_size=args.train_batch_size,
        train_mini_batch_size=args.train_mini_batch_size,
        train_max_samples=args.train_max_samples,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        rollout_max_model_len=args.rollout_max_model_len,
        rollout_n=args.rollout_n,
        total_epochs=args.total_epochs,
        total_training_steps=args.total_training_steps,
        save_freq=args.save_freq,
        test_freq=args.test_freq,
        val_max_samples=args.val_max_samples,
        rollout_gpu_memory_utilization=args.rollout_gpu_memory_utilization,
    )
    job_id = submit(params, dry_run=args.dry_run)
    if job_id:
        print(f"submitted job_id={job_id} display_name={params['display_name']}")


if __name__ == "__main__":
    main()
