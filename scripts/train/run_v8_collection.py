"""Prepare a pinned v8 checkout and detach the proven resident starter once.

This is a container-side bootstrap, not another collection engine or a DLC
submission tool. Without --execute it only prints the exact launch plan.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

from prepare_dlc_workspace_recovery import CACHES
from rule_teacher_training_handoff import save
from run_git_task import checkout

ART = Path('/data/zerovln/artifacts')
REPOSITORY = 'https://git.example.invalid/anonymous/closed_loop_3dgs_rl.git'
COLLECTION_COMMIT = '635ecac7f54f1891540ef6ad7dd3bc07601d79b9'
NAVIGATION_REPOSITORY = 'https://github.com/alibaba-damo-academy/RynnBrain.git'
NAVIGATION_COMMIT = '40f19547941454835ff5e63c22994da74605cfea'
ENTRY = 'scripts/train/start_rule_teacher_v8_resident.py'


def plan(args):
    revision = getattr(args, 'collection_commit', None)
    if revision is not None and not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('collection commit must be an explicit full Git SHA')
    if getattr(args, 'prefix_cache', False) and revision is None:
        raise ValueError('cached bootstrap requires an explicit pushed collection commit')
    output_root = args.output_root.resolve()
    if output_root == ART or not output_root.is_relative_to(ART):
        raise ValueError('a new experiment directory on the authorized data disk is required')
    if not 0 <= args.rank < 4:
        raise ValueError('each source has four nodes: rank must be 0..3')
    if args.resume_manifest and not args.resume_manifest.resolve().is_relative_to(ART):
        raise ValueError('resume manifest must be on the authorized data disk')
    seed = getattr(args, 'navigation_seed', None)
    if seed and not seed.resolve().is_relative_to('/root/rl-code'):
        raise ValueError('navigation seed must be a container-local Git checkout')
    local = Path('/root/rl-code') / f'{output_root.name}-{args.source}-n{args.rank}'
    repo = local / 'repo'
    runtime = Path('/tmp/v8-entry') / f'{output_root.name}-{args.source}-n{args.rank}'
    output = output_root / args.source / f'node{args.rank}'
    command = [sys.executable, '-B', str(repo / ENTRY),
               '--output', str(output), '--debug-output', str(args.debug_output.resolve()),
               '--source', args.source, '--rank', str(args.rank), '--shards', '4',
               '--collectors', '8' if args.source == 'sage' else '7',
               '--actors', '64' if args.source == 'sage' else '16',
               '--max-calls', '5' if args.smoke else '100', '--http-concurrency', '128',
               '--stdout-pid', str(args.stdout_pid)]
    if args.smoke:
        command += ['--limit', '24' if args.source == 'sage' else '21']
    if args.resume_manifest:
        command += ['--resume-manifest', str(args.resume_manifest.resolve())]
    if getattr(args, 'prefix_cache', False):
        command += ['--prefix-cache']
    return dict(schema='v8_r5_collection_bootstrap_v1', repository=REPOSITORY,
                commit=revision or COLLECTION_COMMIT,
                prefix_cache=bool(getattr(args, 'prefix_cache', False)),
                navigation_repository=NAVIGATION_REPOSITORY,
                navigation_commit=NAVIGATION_COMMIT, checkout=str(repo),
                navigation_seed=str(seed.resolve()) if seed else None,
                navigation_checkout=str(local / 'navigation'), runtime=str(runtime),
                output=str(output), receipt=str(output_root / 'platform/bootstrap' /
                    f'{args.source}-node{args.rank}-launch.json'), command=command,
                smoke=args.smoke, changes_platform_task=False,
                automatic_business_restart=False)


def runtime_environment(spec):
    runtime = Path(spec['runtime'])
    env = dict(os.environ)
    for name in CACHES.split():
        folder = runtime / 'cache' / name.lower()
        folder.mkdir(parents=True, exist_ok=True)
        env[name] = str(folder)
    repo = Path(spec['checkout'])
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               RULE_COLLECTION_RUNTIME_ROOT=str(runtime / 'collection'),
               PYTHONPATH=os.pathsep.join(str(repo / p) for p in
                   ('scripts/train', 'scripts/eval', 'verl', 'vln_demo/src')),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    # Use the same SDK setup as the existing resident. Its output is captured,
    # never echoed: the environment may contain Git credentials.
    sdk = '/usr/local/PPU_SDK/envsetup.sh'
    raw = subprocess.check_output(['bash', '-c',
        'export LIBRARY_PATH=${LIBRARY_PATH:-}; source "$1" cuda >&2; env -0',
        'v8-sdk', sdk], env=env)
    result = dict(item.decode().split('=', 1) for item in raw.split(b'\0') if b'=' in item)
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                 'http_proxy', 'https_proxy', 'all_proxy'):
        result.pop(name, None)
    return result


def execute(spec):
    if not Path(__file__).resolve().is_relative_to('/root/rl-code'):
        raise ValueError('execute the bootstrap from a container-local Git checkout')
    output, receipt = Path(spec['output']), Path(spec['receipt'])
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return execute_once(spec, output, receipt)


def execute_once(spec, output, receipt):
    if output.exists() or receipt.exists():
        raise ValueError('attempt already exists; inspect receipts instead of replaying')
    env = runtime_environment(spec)
    repo = Path(spec['checkout'])
    repo.parent.mkdir(parents=True, exist_ok=True)
    code = checkout(spec['repository'], spec['commit'], repo)
    dependency = checkout(spec.get('navigation_seed') or spec['navigation_repository'], spec['navigation_commit'],
                          Path(spec['navigation_checkout']))
    # Reuse the actual starter's PID/start-ticks identity check and owner. All
    # GPU release, liveness, failure cleanup and idle protection stay there.
    log = receipt.with_suffix('.log')
    with log.open('xb') as stream:
        child = subprocess.Popen(spec['command'], cwd=repo, env=env,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
    fields = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()
    result = dict(spec, code=code, navigation=dependency, pid=child.pid,
                  start_ticks=fields[19], hostname=os.uname().nodename,
                  log=str(log), created_at=time.time(),
                  stage='starter_dispatched_not_collection_acceptance')
    save(receipt, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('sage', 'r2r', 'rxr'), required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--debug-output', type=Path, required=True)
    parser.add_argument('--resume-manifest', type=Path)
    parser.add_argument('--collection-commit', help='explicit pushed revision for a new collection attempt')
    parser.add_argument('--prefix-cache', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--navigation-seed', type=Path,
                        help='clone an existing container-local navigation Git repository at the pinned commit')
    parser.add_argument('--stdout-pid', type=int, default=16)
    parser.add_argument('--smoke', action='store_true',
                        help='24 SAGE or 21 native cases with 5 calls; separate output required')
    parser.add_argument('--execute', action='store_true',
                        help='clone pinned code and launch once in this idle resident')
    args = parser.parse_args(argv)
    spec = plan(args)
    print(json.dumps(execute(spec) if args.execute else spec), flush=True)


if __name__ == '__main__':
    main()
