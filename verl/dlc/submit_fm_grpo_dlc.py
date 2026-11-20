#!/usr/bin/env python3
"""提交 VLN 闭环 FM-GRPO RL trainer DLC PyTorchJob（两节点部署的 trainer 端）。

1 节点 16 PPU。NPROC=1（默认）单进程 cuda:0；NPROC>1 torchrun 多卡
（verl/workers/fm_grpo_env_trainer_dist.py，每 rank cuda:LOCAL_RANK，bench URL strided
分片，grad all_reduce）。poll bench job 写的 $HC/bench_ready_<EXPR>.txt 拿 bench URL
→ 跑 trainer → 写 trainer_done_<EXPR>。详见 verl/dlc/README.md。

trainer job 必须用与 bench job 同名的 --display-name（握手文件按它配对）。

用法:
  python3 submit_fm_grpo_dlc.py --display-name fmgrpo_smoke --dry-run
  python3 submit_fm_grpo_dlc.py --display-name fmgrpo_smoke --nproc-per-node 16 --opt-steps 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE_ROOT = Path("/workspace")
REPO = WORKSPACE_ROOT / "closed_loop_3dgs_rl"
DLC_CLI_DIR = WORKSPACE_ROOT / "dlc_cli"
if str(DLC_CLI_DIR) not in sys.path:
    sys.path.insert(0, str(DLC_CLI_DIR))
sys.path.insert(0, str(REPO / "verl" / "dlc"))

from _dlc_common import (  # noqa: E402
    GPUS_PER_NODE, HC_DIR, IMAGE, PRIORITY, WORKSPACE,
    get_dlc_client, safe_name, shell_quote, user_vpc,
)
from alibabacloud_pai_dlc20201203.models import (  # noqa: E402
    CreateJobRequest, CreateJobRequestDataSources, JobSpec, ResourceConfig,
)
from workspace_config import build_data_sources, resolve_workspace  # noqa: E402

ENTRY = REPO / "scripts/cluster/fm_grpo_entry.sh"

# 默认超参 = docs/实验记录.md#legacy-docs-20260921 §4 同款。EPISODES/VAL_EPISODES 空 → entry 从 bench /spec 自动读。
DEFAULTS = dict(
    episodes="", val_episodes="", opt_steps="30", group_size="8",
    num_sde_steps="5", sigma="0.3", accum_steps="1", lr="1e-4",
    train_params="fm_head", eval_every="5", eval_train_sample="10",
    max_chunks="8", history_len="8", nproc="1", master_port="29512", fm_ckpt="",
)


def build_bootstrap(expr_name: str, **kw: str) -> str:
    log_file = WORKSPACE_ROOT / "discussion_logs" / f"fm_grpo_{expr_name}.log"
    exports = "\n".join(
        f"export {k.upper()}={shell_quote(str(v))}" for k, v in kw.items())
    # POSIX heredoc：只 export + exec bash entry。超参从 env 进 entry 的 default 兜底。
    return f"""#!/usr/bin/env bash
set -xeu
mkdir -p $(dirname {shell_quote(str(log_file))})
exec >> {shell_quote(str(log_file))} 2>&1
echo "start_time=$(date -Is) host=$(hostname)"

export EXPR_NAME={shell_quote(expr_name)}
export HC_DIR={shell_quote(str(HC_DIR))}
{exports}
exec bash {shell_quote(str(ENTRY))}

echo "end_time=$(date -Is)"
"""


def submit(expr_name: str, dry_run: bool, **kw: str) -> str | None:
    workspace_profile = resolve_workspace(WORKSPACE)
    user_command = build_bootstrap(expr_name, **kw)
    if dry_run:
        print(f"[dry-run] job_type=PyTorchJob nnodes=1 gpu={GPUS_PER_NODE} workspace={WORKSPACE}")
        print("[dry-run] user_command:\n" + user_command)
        return None

    # trainer 单进程或 torchrun 都用 ≤16 PPU；节点 shape 固定 16，请 16（pool 无更小 shape）。
    worker_spec = JobSpec(
        type="Worker",
        pod_count=1,
        image=IMAGE,
        resource_config=ResourceConfig(
            cpu="184", memory="1800Gi", shared_memory="1800Gi", gpu=str(GPUS_PER_NODE)),
    )
    request = CreateJobRequest(
        display_name=expr_name + "_trainer",
        job_type="PyTorchJob",
        user_command=user_command,
        data_sources=build_data_sources(workspace_profile, CreateJobRequestDataSources),
        resource_id=workspace_profile.resource_id,
        workspace_id=workspace_profile.workspace_id,
        user_vpc=user_vpc(),
        priority=PRIORITY,
        job_specs=[worker_spec],
    )
    return get_dlc_client().create_job(request).body.job_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--display-name", required=True,
                    help="实验名；必须与 bench job 同名（握手文件按它配对）")
    ap.add_argument("--episodes", default=DEFAULTS["episodes"])
    ap.add_argument("--val-episodes", default=DEFAULTS["val_episodes"])
    ap.add_argument("--opt-steps", default=DEFAULTS["opt_steps"])
    ap.add_argument("--group-size", default=DEFAULTS["group_size"])
    ap.add_argument("--num-sde-steps", default=DEFAULTS["num_sde_steps"])
    ap.add_argument("--sigma", default=DEFAULTS["sigma"])
    ap.add_argument("--accum-steps", default=DEFAULTS["accum_steps"])
    ap.add_argument("--lr", default=DEFAULTS["lr"])
    ap.add_argument("--train-params", default=DEFAULTS["train_params"],
                    choices=["fm_head", "fm_head+actionformer", "fm_head+actionformer+backbone"])
    ap.add_argument("--eval-every", default=DEFAULTS["eval_every"])
    ap.add_argument("--eval-train-sample", default=DEFAULTS["eval_train_sample"])
    ap.add_argument("--max-chunks", default=DEFAULTS["max_chunks"])
    ap.add_argument("--history-len", default=DEFAULTS["history_len"])
    ap.add_argument("--nproc-per-node", dest="nproc", default=DEFAULTS["nproc"],
                    help="1=单卡（原 trainer）；>1=torchrun 多卡（dist trainer）")
    ap.add_argument("--master-port", default=DEFAULTS["master_port"])
    ap.add_argument("--fm-ckpt", default=DEFAULTS["fm_ckpt"],
                    help="FM/SFT ckpt 起点；空=trainer 默认 fm_baseline_fixed")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    expr_name = safe_name(args.display_name)
    kw = {k.replace("-", "_"): getattr(args, k.replace("-", "_")) for k in DEFAULTS}
    job_id = submit(expr_name, args.dry_run, **kw)
    if job_id:
        print(f"submitted trainer job_id={job_id} expr={expr_name} nproc={args.nproc}")
        print(f"  reads: {HC_DIR}/bench_ready_{expr_name}.txt")


if __name__ == "__main__":
    main()
