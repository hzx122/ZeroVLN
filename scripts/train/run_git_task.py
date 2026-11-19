"""Clone the main repository into a container-local directory and run a task.

The specification is configuration, not a source bundle. Authentication uses
the container's Git credentials; repository URLs must not contain credentials.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import traceback
import time
from urllib.parse import urlsplit

import align_three_source_observations as io


def clean_repository(value):
    parsed = urlsplit(value)
    if parsed.scheme in ('http', 'https') and (parsed.username or parsed.password):
        raise ValueError('use Git credential helpers, not credentials in repository URLs')
    if '\n' in value or value.startswith('-'):
        raise ValueError('invalid Git repository')
    return value


def checkout(repository, revision, destination):
    """A private, clean checkout; refuse reuse of a changed or different repo."""
    repository = clean_repository(repository)
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('task code must specify a pushed 40-character Git commit')
    destination = Path(destination)
    fresh = not destination.exists()
    if fresh:
        subprocess.run(['git', 'clone', '--quiet', '--no-hardlinks', '--no-checkout', '--', repository,
                        str(destination)], check=True)
    actual = subprocess.check_output(['git', '-C', str(destination), 'remote',
                                     'get-url', 'origin'], text=True).strip()
    if actual != repository:
        raise ValueError('existing checkout belongs to a different repository')
    dirty = subprocess.check_output(['git', '-C', str(destination), 'status',
                                    '--porcelain', '--untracked-files=no'], text=True)
    if dirty and not fresh:
        raise ValueError('refuse to overwrite local code changes')
    subprocess.run(['git', '-C', str(destination), 'fetch', '--quiet', 'origin', revision], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', '--quiet', '--detach', revision], check=True)
    commit = subprocess.check_output(['git', '-C', str(destination), 'rev-parse',
                                     'HEAD'], text=True).strip()
    if commit != revision:
        raise ValueError('Git checkout did not resolve the requested commit')
    return dict(repository=repository, commit=commit, path=str(destination))


def task_node_rank(spec, env):
    """Use the experiment node count, not a possibly cleared torchrun world."""
    from v6_rl_dlc_worker import node_rank
    nodes = spec.get("node_count")
    if spec.get("kind", "rl") == "rl":
        nodes = int(io.load(spec["launch_plan"])["config"]["nnodes"])
    return node_rank(env, nodes=nodes)


def local_plan(spec, repo, dependencies, local):
    """Rebind code paths only; keep the experiment's data and initialization."""
    plan = io.load(spec['launch_plan'])
    if 'native_admission' in spec:
        binding = spec['native_admission']
        raw = io.read_bytes(binding['path'])
        if io.digest(raw) != binding['sha256']:
            raise ValueError('native admission report changed')
        admission = json.loads(raw)
        if (admission['schema'] not in ('native_rl_start_goal_admission_v1', 'three_source_navigation_pool_v2')
                or io.digest(io.read_bytes(plan['config']['case_manifest'])) != admission['train_cases_sha256']):
            raise ValueError('native admission case manifest changed')
        if admission['schema'] == 'three_source_navigation_pool_v2':
            from rl_task_config import apply_navigation_pool
            expected_config = dict(plan['config'])
            apply_navigation_pool(expected_config, Path(binding['path']).parent)
            if expected_config != plan['config']:
                raise ValueError('formal navigation sampling differs from bound 2:1:1 pool')
        for mesh in admission.get('meshes', {}).values():
            if io.digest(io.read_bytes(mesh['path'])) != mesh['sha256']:
                raise ValueError('native admission navmesh changed: ' + mesh['path'])
    plan.update(code_mode='git_checkout', code_commit=spec['commit'], launcher_manages_standby=True,
                snapshot_repo=str(repo),
                node_worker=str(repo / 'scripts/train/v6_rl_dlc_worker.py'),
                upstream_repo=dependencies['verl']['path'],
                renderer_repo=str(repo / 'third_party/abot_renderer'),
                runtime_config=str(local / 'config.json'))
    plan['snapshot_upstream'] = plan['upstream_repo']  # compatibility field; no copy
    if 'source_object_map' in spec:
        plan['source_object_map'] = spec['source_object_map']
    plan['components'] = spec.get('components', [])
    plan['execution_profile'] = spec.get('execution_profile', 'v7_v3_collision')
    if 'navigation' in dependencies:
        plan['navigation_repo'] = str(Path(dependencies['navigation']['path']) /
            spec['dependencies']['navigation'].get('subdirectory', 'navigation'))
    plan['fast_kernel_overlay'] = spec.get('fast_kernel_overlay', '')
    for key in ('resume_from_path',):
        if key in spec:
            plan[key] = spec[key]
    io.write_json(local / 'config.json', plan['config'])
    io.write_json(local / 'plan.json', plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--preflight', action='store_true',
                        help='validate the full RL trainer command without starting GPU work')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    spec = io.load(args.spec)
    root = Path(spec['checkout_root']).resolve()
    if root.is_relative_to('/data/storage') or root.is_relative_to('/data/storage'):
        raise ValueError('Git task checkout must be on container-local storage')
    output = Path(spec['output_root']).resolve()
    if not output.is_relative_to(io.ART) or output == io.ART:
        raise ValueError('task artifacts must be on the authorized data disk')
    rank = task_node_rank(spec, os.environ)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'task.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(phase='cloning', pid=os.getpid(), rank=rank,
                 hostname=os.uname().nodename, business_pid=None, business_returncode=None)
    stop = threading.Event()
    def heartbeat():
        while not stop.is_set():
            print(json.dumps(dict(event='git_task_liveness', time=time.time(), **state)), flush=True)
            stop.wait(20)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    child = None
    receipt = output / 'git_runtime' / f'node_{rank:02d}'
    try:
        try:
            code = checkout(spec['repository'], spec['commit'], root / 'repo')
            local_entry = root / 'repo/scripts/train/run_git_task.py'
            if Path(__file__).resolve() != local_entry.resolve():
                # Supervisors and I/O helpers also come from the local checkout.
                os.execve(sys.executable, [sys.executable, '-B', str(local_entry),
                          '--spec', str(args.spec.resolve()),
                          *(['--prepare-only'] if args.prepare_only else []),
                          *(['--preflight'] if args.preflight else [])], dict(os.environ))
            deps = {}
            (root / 'dependencies').mkdir(exist_ok=True)
            for name, item in spec.get('dependencies', {}).items():
                if not re.fullmatch(r'[a-z][a-z0-9_-]*', name):
                    raise ValueError('invalid dependency name')
                deps[name] = checkout(item['repository'], item['commit'], root / 'dependencies' / name)
            receipt.mkdir(parents=True, exist_ok=True)
            io.write_json(receipt / 'checkout.json', dict(time=time.time(), **state,
                code=code, dependencies=deps, spec=str(args.spec.resolve())))
            state['phase'] = 'prepared'
            if args.prepare_only and not args.preflight:
                return 0
            repo = root / 'repo'
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
            if spec.get('kind', 'rl') == 'rl':
                if not {'verl', 'navigation'} <= set(deps):
                    raise ValueError('RL needs local Git checkouts of VERL and navigation')
                local = root / 'runtime'; local.mkdir(exist_ok=True)
                plan = local_plan(spec, repo, deps, local)
                if Path(plan['output_root']).resolve() != output:
                    raise ValueError('spec and launch plan output roots differ')
                interpreter = sys.executable
                if plan['config'].get('text_backend') == 'megatron':
                    environment = root / 'environment'
                    state['phase'] = 'preparing_megatron_environment'
                    if spec.get('megatron_environment_ready'):
                        ready = io.load(spec['megatron_environment_ready'])
                        if (not ready['ready'] or ready['main_commit'] != spec['commit']
                                or Path(ready['python']).resolve() != (environment/'venv/bin/python').resolve()):
                            raise ValueError('prepared Megatron environment binding changed')
                    else:
                        raise ValueError('prepare the Megatron environment while the existing GPU holder remains active')
                    plan['megatron_environment'] = str(environment)
                    interpreter = str(environment/'venv/bin/python')
                    io.write_json(local/'plan.json', plan)
                command = [interpreter, '-B', plan['node_worker'], '--launch-plan', str(local / 'plan.json')]
            else:
                entry = Path(spec['entry'])
                if entry.is_absolute() or '..' in entry.parts:
                    raise ValueError('task entry must be relative to its Git checkout')
                command = [sys.executable, '-B', str(repo / entry), *spec.get('arguments', [])]
                env['PYTHONPATH'] = os.pathsep.join([str(repo / 'scripts/train'), str(repo / 'verl'),
                    str(repo / 'vln_demo/src'), *[d['path'] for d in deps.values()]])
            if args.preflight:
                if spec.get('kind', 'rl') != 'rl':
                    raise ValueError('RL preflight requires kind=rl')
                from v6_rl_dlc_worker import node_environment, training_initialization_kwargs
                import v6_rl
                check_root = output / 'preflight' / f'node_{rank:02d}'
                check_root.mkdir(parents=True, exist_ok=True)
                env = node_environment(plan, rank)
                # Build in the same environment as the real worker, including
                # local dependency routing and all optimization flags.
                os.environ.clear()
                os.environ.update(env)
                training = v6_rl.build_plan(plan['config'],
                    head_checkpoint=plan['initial_checkpoint']+'/action_head',
                    query_checkpoint=plan['initial_checkpoint']+'/dualvln_query',
                    bench_urls=','.join(f'http://127.0.0.1:{8950+i}'
                                        for i in range(plan['config']['rollout_actors'])),
                    output_dir=str(check_root / 'train'), **training_initialization_kwargs(plan))
                command = [*training['command'], '--validate-only']
                state['phase'] = 'validating_trainer'
                with (check_root / 'trainer.log').open('ab', buffering=0) as log:
                    child = subprocess.Popen(command, cwd=repo, env={**env, **training['env']},
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    state['business_pid'] = child.pid
                    code = child.wait()
                io.write_json(check_root / 'result.json', dict(returncode=code,
                    code_commit=spec['commit'], command=command, time=time.time(),
                    gpu_training_started=False))
                return code
            state['phase'] = 'running'
            child = subprocess.Popen(command, cwd=repo, env=env, start_new_session=True)
            state['business_pid'] = child.pid
            code = child.wait()
            state.update(phase='succeeded' if code == 0 else 'failed', business_returncode=code)
            io.write_json(receipt / 'business_exit.json', dict(time=time.time(), **state))
            if (code == 0 and not spec.get('standby_after_success', False)) or (
                    code != 0 and not spec.get('standby_after_failure', True)):
                return code
        except Exception as exc:
            traceback.print_exc()
            code = 1
            state.update(phase='failed', business_returncode=code,
                         error=f'{type(exc).__name__}: {exc}')
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=300)
            receipt.mkdir(parents=True, exist_ok=True)
            io.write_json(receipt / 'business_exit.json', dict(time=time.time(), **state))
            repo = root / 'repo'
            if args.prepare_only or not spec.get('standby_after_failure', True) or not repo.exists():
                return code
        # This process stays in the original console and reports the real error.
        holder = repo / 'verl/scripts/train/action_head_sft_data/gpu_occupier.py'
        child = subprocess.Popen([sys.executable, '-B', str(holder), '--gpus',
            ','.join(map(str, range(16))), '--require-gpu-count', '16',
            '--ready-file', str(root / 'standby_ready.json')], cwd=repo, start_new_session=True)
        state.update(phase='reserving_after_success' if code == 0 else 'reserving_after_failure',
                     standby_pid=child.pid)
        deadline = time.monotonic() + 100
        while not (root / 'standby_ready.json').exists():
            if child.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError('failure GPU reservation did not become ready')
            time.sleep(1)
        state['phase'] = 'succeeded_standby' if code == 0 else 'failed_standby'
        child.wait()
        raise RuntimeError('failure standby exited; original business returncode=' + str(code))
    finally:
        stop.set()
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                # The worker reaps trainer/Ray groups and then its services.
                # Killing it after 20s can interrupt that cleanup and strand
                # GPU children started in their own process groups.
                child.wait(timeout=300)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)


if __name__ == '__main__':
    raise SystemExit(main())
