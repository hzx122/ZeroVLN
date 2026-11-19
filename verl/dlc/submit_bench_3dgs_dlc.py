#!/usr/bin/env python3
"""提交 3DGS + bench 服务节点 DLC PyTorchJob（两节点部署的 bench 端）。

1 节点 16 PPU：4 render server（各 4 卡）+ 16 rs3d bench，起齐后写
$HC/bench_ready_<EXPR>.txt 给 trainer job 读。trainer 跑完写 trainer_done_<EXPR>，
本 job poll 到即 pkill 退出。详见 verl/dlc/README.md。

照 VLA-RLVR/scripts/dlc/submit_decode_internal_frames_dlc.py 的最简形态
（PyTorchJob + 单 Worker + dry-run + heredoc），逻辑全在 bench_3dgs_entry.sh。

用法:
  python3 submit_bench_3dgs_dlc.py --display-name fmgrpo_smoke --dry-run
  python3 submit_bench_3dgs_dlc.py --display-name fmgrpo_smoke
"""

from __future__ import annotations

import argparse
import sys
import time
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
from backends import BACKENDS, fields  # noqa: E402

ENTRY = REPO / "verl" / "dlc" / "bench_3dgs_entry.sh"
ARTIFACT_ROOT = Path('/data/zerovln/artifacts')


def build_bootstrap(expr_name: str, backend: str, split: str | None,
                    eps_per_scene: int,
                    sage_action_head_contract: bool = False,
                    bench_replicas: int = 16,
                    entry_path: Path = ENTRY,
                    client_exprs: tuple[str, ...] = ()) -> str:
    artifact = ARTIFACT_ROOT / expr_name
    log_file = artifact / 'logs/driver.log'
    # POSIX heredoc（DLC user_command 跑在 /bin/sh）：只 export + exec bash entry。
    # split/eps env 按后端配置（backends.py: split_env/eps_env）导出——abotn 导 ABOTN_SET，
    # sage 导 SAGE_SPLIT+SAGE_EPS_PER_SCENE，rs3d 无 split env。entry 内 eval backends.py 读。
    cfg = fields(backend)
    extras = [f"export BENCH_REPLICAS={bench_replicas}"]
    extras.extend([
        f"export REPO={shell_quote(str(REPO))}",
        f"export BENCH_ARTIFACT_ROOT={shell_quote(str(artifact))}",
        f"export BENCH_CLIENT_EXPRS={shell_quote(','.join(client_exprs))}",
    ])
    if cfg["split_env"] and split is not None:
        extras.append(f"export {cfg['split_env']}={shell_quote(split)}")
    if cfg["eps_env"]:
        extras.append(f"export {cfg['eps_env']}={int(eps_per_scene)}")
    if backend == "sage" and sage_action_head_contract:
        extras.extend([
            f"export DEMO_SPLIT={shell_quote(split)}",
            "export SAGE_COLLISION=usd_rooms",
            "export SAGE_FURNITURE_COLLISION=none",
            "export SAGE_ROBOT_RADIUS=0.07",
            "export SAGE_CAMERA_HEIGHT=1.2",
            "export SAGE_HFOV=115",
            ("export SAGE_USD_COLLISION_MASK_ROOT="
             "/data/zerovln/"
             "SAGE-3D/sage_usd_rooms_masks"),
            ("export SAGE_USD_STATIC_BOUNDARY_MASK_ROOT="
             "/data/zerovln/"
             "SAGE-3D/sage_usd_rooms_masks"),
        ])
    extra = "\n".join(extras) + ("\n" if extras else "")
    return f"""#!/usr/bin/env bash
set -xeu
mkdir -p $(dirname {shell_quote(str(log_file))})
exec >> {shell_quote(str(log_file))} 2>&1
echo "start_time=$(date -Is) host=$(hostname)"

export EXPR_NAME={shell_quote(expr_name)}
export HC_DIR={shell_quote(str(HC_DIR))}
export BACKEND={shell_quote(backend)}
{extra}exec bash {shell_quote(str(entry_path))}

echo "end_time=$(date -Is)"
"""


def submit(expr_name: str, backend: str, split: str | None,
           eps_per_scene: int, dry_run: bool,
           sage_action_head_contract: bool = False,
           bench_replicas: int = 16,
           client_exprs: tuple[str, ...] = (),
           max_running_minutes: int = 0) -> str | None:
    workspace_profile = resolve_workspace(WORKSPACE)
    # Keep the two-level directory depth used by the entry's REPO discovery.
    # Bash reads a long-lived script incrementally; freeze its bytes per job.
    entry_path = ARTIFACT_ROOT / expr_name / "entry_snapshots/bench_3dgs_entry.sh"
    user_command = build_bootstrap(
        expr_name, backend, split, eps_per_scene,
        sage_action_head_contract=sage_action_head_contract,
        bench_replicas=bench_replicas, entry_path=entry_path, client_exprs=client_exprs)
    if dry_run:
        print(f"[dry-run] job_type=PyTorchJob nnodes=1 gpu={GPUS_PER_NODE} cpu=184 backend={backend} workspace={WORKSPACE} max_running_minutes={max_running_minutes}")
        print("[dry-run] user_command:\n" + user_command)
        return None

    entry_path.parent.mkdir(parents=True, exist_ok=True)
    with entry_path.open("xb") as snapshot:
        snapshot.write(ENTRY.read_bytes())

    worker_spec = JobSpec(
        type="Worker",
        pod_count=1,
        image=IMAGE,
        resource_config=ResourceConfig(
            cpu="184", memory="1800Gi", shared_memory="1800Gi", gpu=str(GPUS_PER_NODE)),
    )
    request = CreateJobRequest(
        display_name=expr_name + "_bench",
        job_type="PyTorchJob",
        user_command=user_command,
        data_sources=build_data_sources(workspace_profile, CreateJobRequestDataSources),
        resource_id=workspace_profile.resource_id,
        workspace_id=workspace_profile.workspace_id,
        user_vpc=user_vpc(),
        priority=PRIORITY,
        job_specs=[worker_spec],
        job_max_running_time_minutes=max_running_minutes or None,
    )
    return get_dlc_client().create_job(request).body.job_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--display-name", required=True,
                    help="实验名；trainer job 必须用同名（握手文件按它配对）")
    ap.add_argument("--backend", choices=list(BACKENDS), default="rs3d",
                    help="后端取自 verl/dlc/backends.py 配置：rs3d=3DRS free-roam(无goal,reward恒0) "
                         "/ abotn=ABotN 有goal(真实reward) / sage=SAGE-3D GVLN 有goal")
    ap.add_argument("--abotn-set", choices=["poi", "in", "out"], default="poi",
                    help="abotn benchmark set（仅 backend=abotn 生效）")
    ap.add_argument("--sage-split", choices=BACKENDS["sage"]["split_choices"],
                    default="instr", help="sage GVLN split（仅 backend=sage 生效）")
    ap.add_argument("--sage-eps-per-scene", type=int, default=0,
                    help="sage 每场景限量 episode 数（train 929 场景必设 >0 省 RAM；0=全量）")
    ap.add_argument("--sage-action-head-contract", action="store_true",
                    help="固定 Action Head RL 物理口径：P0 split、usd_rooms、r=0.07、cam=1.2、HFOV=115")
    ap.add_argument("--bench-replicas", type=int, default=16,
                    help="独立 stateful gym URL 数；2x16 用16，4x16满卡布局用18")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument('--client-exprs', default='', help='Comma-separated trainer experiment names; disjoint equal gym partitions, wait for all trainers')
    ap.add_argument('--max-running-minutes', type=int, default=0,
                    help='DLC runtime cap in minutes; 0 leaves it unset')
    args = ap.parse_args()
    if args.bench_replicas <= 0 or args.bench_replicas > 64:
        raise ValueError("--bench-replicas must be in [1, 64]")
    expr_name = safe_name(args.display_name)
    clients = tuple(safe_name(value) for value in args.client_exprs.split(',') if value)
    if len(set(clients)) != len(clients) or (clients and args.bench_replicas % len(clients)):
        raise ValueError('client names must be distinct and evenly divide bench replicas')
    if args.max_running_minutes < 0:
        raise ValueError('max-running-minutes must be nonnegative')
    cfg = fields(args.backend)
    # 按后端配置选 split 值（abotn→ABOTN_SET 值 / sage→SAGE_SPLIT 值 / rs3d→None）
    split = None
    if cfg["split_env"]:
        split = args.abotn_set if args.backend == "abotn" else args.sage_split
    eps = args.sage_eps_per_scene if cfg.get("eps_env") else 0
    if args.sage_action_head_contract:
        if args.backend != "sage" or args.sage_split not in ("action_head_train_p0_unified", "official_gvln_action_head_p0_v3full827train"):
            raise ValueError(
                "--sage-action-head-contract requires --backend sage "
                "--sage-split action_head_train_p0_unified")
    job_id = submit(
        expr_name, args.backend, split, eps, args.dry_run,
        sage_action_head_contract=args.sage_action_head_contract,
        bench_replicas=args.bench_replicas, client_exprs=clients,
        max_running_minutes=args.max_running_minutes)
    if job_id:
        print(f"submitted bench job_id={job_id} expr={expr_name} backend={args.backend}")
        print(f"  ready: {HC_DIR}/bench_ready_{expr_name}.txt  episodes: {HC_DIR}/bench_episodes_{expr_name}.txt")


if __name__ == "__main__":
    main()
