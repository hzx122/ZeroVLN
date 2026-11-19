#!/usr/bin/env python3
"""Submit direct v6 action expert evaluation on locked English R2R/RxR data."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/workspace')
REPO = ROOT / 'closed_loop_3dgs_rl'
ART = Path('/data/zerovln/artifacts')
CHECKPOINT = ART / 'action-head-v43-bootstrap-from17000-4node-20round-1ep-20260918-r5/checkpoints/global_step_017566'
BACKBONE = Path('/opt/external/user-e7a01947/Model/Qwen3.5-4B')
EVAL_SET = REPO / 'vln_demo/batch/eval_sets/english-val-unseen-v1'
MESA_SEED = ART.parent / 'runtime_inputs/habitat-mesa-25.2.8-ubuntu24.04-amd64'
RUNNER = REPO / 'scripts/eval/run_v6_vlnce_dlc.sh'
sys.path[:0] = [str(ROOT / 'dlc_cli'), str(REPO), str(REPO / 'verl'),
               str(REPO / 'verl/dlc'), str(REPO / 'vln_demo/src'),
               str(REPO / 'vln_demo/batch'),
               str(REPO / 'verl/scripts/eval/qwen36_4query_ckpt_sweep')]
from common import atomic_json, checkpoint_identity, sha256_file
from eval_sets import load_ladder


def validate_inputs(checkpoint=None, ladder_stage='full'):
    checkpoint = Path(checkpoint or CHECKPOINT)
    navmesh_root = os.environ.get('VLNCE_NAVMESH_ROOT')
    navmesh_identity = None
    if navmesh_root:
        from vln_gym.vlnce_action_head import navmesh_override_identity
        navmesh_identity = navmesh_override_identity(navmesh_root)
    from prepare_vlnce_mesa_runtime import verify_seed
    verify_seed(MESA_SEED)
    ladder = load_ladder(EVAL_SET, verify_sources=True)
    cases = ladder[ladder_stage]
    counts = dict(Counter(row['dataset'] for row in cases))
    expected_counts = ({'r2r': 1839, 'rxr': 3669} if ladder_stage == 'full'
                       else {'r2r': int(ladder_stage), 'rxr': int(ladder_stage)})
    if counts != expected_counts:
        raise ValueError(f'Unexpected evaluation population: {counts}')
    from watch_checkpoints import assert_checkpoint_protocol
    from action_head.protocol import protocol_module
    report = json.loads((checkpoint / 'report.json').read_text())
    assert_checkpoint_protocol(report, 'v6', allow_legacy_v43_prompt=True)
    required = [BACKBONE / 'config.json', checkpoint / 'dualvln_query/dualvln_query_embeddings.pt',
                checkpoint / 'action_head/fast_slow_action_head.pt',
                Path('/opt/navigation-runtime/venv/bin/python')]
    scene_root = Path('/data/shared/groups/mfm/common/anno/vlmevalkit/RynnBrain/data/scene_datasets/mp3d')
    for scene in sorted({row['scene'] for row in cases}):
        required.extend([scene_root / scene / f'{scene}.glb', scene_root / scene / f'{scene}.navmesh'])
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    code = [RUNNER, Path(__file__), REPO / 'scripts/eval/v6_vlnce_eval.py',
            REPO / 'scripts/eval/smoke_v6_vlnce.py',
            REPO / 'scripts/eval/prepare_vlnce_mesa_runtime.py',
            REPO / 'vln_demo/src/vln_gym/vlnce_action_head.py',
            REPO / 'vln_demo/src/vln_gym/env.py',
            REPO / 'vln_demo/src/remote_habitat/client.py',
            REPO / 'vln_demo/src/remote_habitat/server.py',
            REPO / 'verl/scripts/eval/action_head_fm_p0_p2/curriculum_action_head_eval.py',
            REPO / 'verl/scripts/eval/qwen36_4query_ckpt_sweep/closedloop_eval.py',
            REPO / 'verl/scripts/train/action_head_bootstrap/case_scheduler.py',
            REPO / 'vln_demo/batch/model_gateway.py',
            REPO / 'vln_demo/vllm/vllm-action-head-query.sh',
            REPO / 'vln_demo/vllm/vllm_log_hidden_id.json',
            REPO / 'vln_demo/src/verl_access/bench_client.py',
            REPO / 'vln_demo/src/vln_gym/client.py',
            REPO / 'vln_demo/src/vln_demo.yaml']
    from v6_vlnce_eval import SOURCE_FILES
    code = sorted(set(code) | {REPO / name for name in SOURCE_FILES})
    return dict(schema='v6-vlnce-merged-batch-english-full-v3', counts=counts, repeats=1,
                initial_cases={dataset: next(row for row in cases if row['dataset'] == dataset)
                               for dataset in ('r2r', 'rxr')},
                checkpoint=checkpoint_identity(checkpoint), backbone=str(BACKBONE),
                checkpoint_selection=str(checkpoint), ladder_stage=ladder_stage,
                navmesh_override=navmesh_identity,
                protocol=protocol_module('v6').protocol_identity(),
                evaluation_set=str(EVAL_SET), evaluation_lock_sha256=sha256_file(EVAL_SET / 'lock.json'),
                mesa_seed=str(MESA_SEED), mesa_manifest_sha256=sha256_file(MESA_SEED / 'manifest.json'),
                source_sha256={str(path): sha256_file(path) for path in code},
                max_tool_calls=100, max_turns=0, native_max_steps=0, episode_timeout=0,
                scoring_mode='merged_batch',
                success_contract='Habitat terminal STOP after normal or tool-budget completion; geodesic distance < 3m',
                scheduling='one shared cross-dataset rolling queue, 32 actors; retain backend preference then share the exhausted-dataset tail',
                model_routing='actor-affine across six replicas, per-replica transparent AIMD',
                infrastructure_retries=1,
                gpus=16, brain_replicas=6, tensor_parallel=2, fm_replicas=4, gym_actors=32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expr-name', default='v6-r2r-rxr-merged100-full-20260923-r2')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--expected-record', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--ladder-stage', choices=('100', '200', '500', '1000', 'full'), default='full')
    parser.add_argument('--navmesh-root', type=Path,
                        help='Explicit versioned navmesh override; requires a new evaluation output')
    args = parser.parse_args()
    if args.navmesh_root:
        os.environ['VLNCE_NAVMESH_ROOT'] = str(args.navmesh_root.resolve(strict=True))
    # DLC uses private endpoints and the local credentials service, never a proxy.
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(key, None)
    from _dlc_common import get_dlc_client, safe_name, shell_quote, user_vpc, IMAGE, PRIORITY, WORKSPACE
    from workspace_config import resolve_workspace, build_data_sources
    if args.expr_name != safe_name(args.expr_name):
        parser.error('invalid EXPR_NAME')
    inputs = validate_inputs(args.checkpoint, args.ladder_stage)
    if args.expected_record and json.loads(args.expected_record.read_text()) != inputs:
        raise RuntimeError('inputs or implementation changed since submission; revalidate before running')
    if args.check_only:
        print(json.dumps(dict(ok=True, **inputs), ensure_ascii=False))
        return
    output = ART / args.expr_name
    launch = output / 'launch'
    command = '\n'.join([
        '#!/usr/bin/env bash', 'set -euo pipefail',
        f'export EXPR_NAME={shell_quote(args.expr_name)}',
        f'export CHECKPOINT_DIR={shell_quote(str(args.checkpoint))}',
        f'export VLNCE_LADDER_STAGE={shell_quote(args.ladder_stage)}',
        f'export BACKBONE={shell_quote(str(BACKBONE))}',
        (f'export VLNCE_NAVMESH_ROOT={shell_quote(inputs["navmesh_override"]["root"])}'
         if inputs['navmesh_override'] else 'unset VLNCE_NAVMESH_ROOT'),
        f'mkdir -p {shell_quote(str(output / "logs"))}',
        f'exec >{shell_quote(str(output / "logs/dlc_entry.log"))} 2>&1',
        f'exec bash {shell_quote(str(RUNNER))}', ''])
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, output=str(output), command=command, inputs=inputs), ensure_ascii=False))
        return
    launch.mkdir(parents=True, exist_ok=True)
    with (launch / 'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (launch / 'submission.json').exists():
            raise FileExistsError('EXPR_NAME has already been submitted')
        atomic_json(launch / 'inputs.json', inputs)
        (launch / 'entry.sh').write_text(command)
        from alibabacloud_pai_dlc20201203.models import (
            CreateJobRequest, CreateJobRequestDataSources, JobSpec, ResourceConfig)
        profile = resolve_workspace(WORKSPACE)
        request = CreateJobRequest(
            display_name=args.expr_name, job_type='PyTorchJob',
            user_command='bash -c ' + shell_quote(command),
            data_sources=build_data_sources(profile, CreateJobRequestDataSources),
            resource_id=profile.resource_id, workspace_id=profile.workspace_id,
            user_vpc=user_vpc(), priority=PRIORITY,
            job_specs=[JobSpec(type='Worker', pod_count=1, image=IMAGE,
                resource_config=ResourceConfig(cpu='184', memory='1760Gi',
                                               shared_memory='1760Gi', gpu='16'))])
        job_id = get_dlc_client().create_job(request).body.job_id
        record = dict(job_id=job_id, utc=datetime.now(timezone.utc).isoformat(),
                      expr_name=args.expr_name, output=str(output), inputs=inputs)
        atomic_json(launch / 'submission.json', record)
        with (REPO / 'docs/实验记录.md').open('a') as stream:
            stream.write(f'\n\n{record["utc"]}：v6 Action Head as Tool 英文全量R2R/RxR评测提交 `{job_id}`；'
                f'EXPR_NAME=`{args.expr_name}`。启动 `bash {launch}/entry.sh`；提交器 `{Path(__file__)}`；'
                f'主脚本 `{RUNNER}`；冻结清单 `{EVAL_SET}/full.json`，输入锁 `{launch}/inputs.json`；'
                f'权重 `{args.checkpoint}`；输出 `{output}/evaluation`，日志 `{output}/logs`，无TensorBoard。'
                f'固定清单档位{args.ladder_stage}、数量{inputs["counts"]}，英文val_unseen guide，每条1次；对齐合并链路100工具调用，'
                '无独立模型轮数/回合墙钟/原生步数上限，正式SR/SPL按结束或工具预算耗尽后的Habitat终态STOP及3m判定，另保留模型主动STOP指标。'
                '1节点16卡（6×TP2大脑+4×FM），32独立Gym共用跨数据集队列；actor优先保持当前backend，某数据集耗尽后动态借槽，避免交替清单触发反复重建；actor固定脑副本、每脑独立透明AIMD；CPU远程Habitat渲染。'
                '旧200预算对模型可见，未复用旧回合。'
                '运行时缓存 `/tmp/v6vlnce/`（容器隔离，短Unix socket路径），提交成功不等于完成。\n')
        print(json.dumps(dict(job_id=job_id, output=str(output), counts=inputs['counts'], gpus=16)))


if __name__ == "__main__":
    raise SystemExit("DLC submission entry removed in the open-source edition; use the root GPU run scripts.")
