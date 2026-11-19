#!/usr/bin/env python3
"""One DLC worker owns its configured GPU layout and isolated CPU gyms."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

from rl_task_config import rollout_gyms_per_node, native_scene_cache_settings


def require_unused_ray_head(ip, port=6379):
    """Never advertise a stale cluster as this attempt's newly started head."""
    with socket.socket() as connection:
        connection.settimeout(1)
        if connection.connect_ex((ip, port)) == 0:
            raise RuntimeError(f'Ray head address {ip}:{port} is already occupied; '
                               'verify and retire the previous attempt before startup')


def node_bench_urls(plan, ip):
    """Advertise every independent Gym launched by node_environment."""
    return [f'http://{ip}:{8950 + index}'
            for index in range(rollout_gyms_per_node(plan['config']))]


def atomic_json(path: Path, data):
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def record_failure(status, progress, exc):
    """Preserve the cause in the Pod console even if shared logs are unwritable."""
    storage = []
    for path in (progress.output, Path('/tmp')):
        item = {'path': str(path)}
        try:
            disk = os.statvfs(path)
            item.update(available_bytes=disk.f_bavail * disk.f_frsize,
                        available_inodes=disk.f_favail)
        except OSError as probe_error:
            item['statvfs_error'] = str(probe_error)
        storage.append(item)
    event = {'phase': 'failed', 'error': f'{type(exc).__name__}: {exc}',
             'traceback': ''.join(traceback.format_exception(exc)), 'storage': storage}
    try:
        progress.emit(event)
    except OSError:
        pass  # A broken Pod console must not replace the original exception.
    try:
        status('failed', error=event['error'])
    except BaseException as record_error:
        try:
            progress.emit({'phase': 'failed', 'error': event['error'],
                           'failure_record_error': f'{type(record_error).__name__}: {record_error}'})
        except OSError:
            pass


class ConsoleProgress:
    """Expose actual file activity to DLC without claiming optimizer progress.

    File logs remain authoritative. Unchanged files produce no heartbeat, so
    this cannot turn a stalled workload into an endlessly healthy log stream.
    The worker's own log is excluded to avoid observing our own status output.
    """

    def __init__(self, output: Path, rank: int, console_fd: int):
        self.output, self.rank, self.console_fd = output, rank, console_fd
        self.seen = {}
        self.last_poll = None

    def emit(self, event):
        payload = (json.dumps({'rank': self.rank, **event}) + '\n').encode()
        while payload:
            payload = payload[os.write(self.console_fd, payload):]

    def poll(self, now=None):
        now = time.monotonic() if now is None else now
        if self.last_poll is not None and now - self.last_poll < 30:
            return
        self.last_poll = now
        paths = [self.output / 'logs' / f'{label}_node_{self.rank:02d}.log'
                 for label in ('ray', 'trainer')]
        paths += list((self.output / 'services' / f'node_{self.rank:02d}' / 'logs').glob('*.log'))
        # Inference actors may be placed on any Ray node. These shared files
        # report real sampler activity even while learners wait for a wave.
        paths += list((self.output / 'train' / 'logs').glob('vllm_*.log'))
        # Learners report completed condition batches even between updates.
        # No timer-generated records: a stalled backward stops this activity.
        paths += list((self.output / 'train' / 'learner_progress').glob('rank_*.jsonl'))
        paths += [self.output / 'train' / name for name in
                  ('metrics.jsonl', 'rollouts.jsonl', 'training_updates.jsonl')]
        changed = []
        for path in paths:
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if stat.st_size and self.seen.get(path) != signature:
                changed.append({'file': str(path.relative_to(self.output)),
                                'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
            self.seen[path] = signature
        if changed:
            self.emit({'event': 'component_log_activity', 'files': changed,
                       'meaning': 'file activity, not proof of a completed training update'})


def node_rank(env, nodes=None):
    nodes = int(env.get('WORLD_SIZE', 4)) if nodes is None else nodes
    if type(nodes) is not int or nodes < 1:
        raise ValueError('DLC node count must be a positive integer')
    for name in ('RANK', 'PET_NODE_RANK', 'NODE_RANK'):
        if env.get(name) is not None:
            rank = int(env[name])
            if not 0 <= rank < nodes:
                raise ValueError(f'{name}={rank} outside {nodes}-node job')
            return rank
    raise RuntimeError('DLC did not provide a node rank')


def training_initialization_kwargs(plan: dict) -> dict:
    """Forward the experiment initialization mode to the actual remote trainer."""
    resume = plan.get('resume_from_path')
    warm = plan.get('policy_warm_start_checkpoint')
    transition = plan.get('allow_rule_credit_transition', False)
    if not isinstance(transition, bool):
        raise ValueError('allow_rule_credit_transition must be an explicit boolean')
    if transition and (not resume or warm or plan.get('same_training_objective') is not False):
        raise ValueError('rule-credit transition requires complete resume and an explicitly changed objective')
    if resume and warm:
        raise ValueError('exact resume and policy warm-start are mutually exclusive')
    if resume:
        return {'resume_from_path': resume,
                **({'allow_weights_only_resume': True} if plan.get('allow_weights_only_resume') else {}),
                **({'allow_rule_credit_transition': True} if transition else {})}
    if warm:
        return {'policy_warm_start_checkpoint': warm}
    return {}


def apply_training_compiler_cache(training_plan: dict, environment: dict) -> None:
    """Bind the root actually consumed by learner rank-cache initialization."""
    value = environment.get('ACTION_HEAD_TRAINING_PROCESS_COMPILE_CACHE_ROOT')
    if not value:
        return
    path = Path(value)
    resolved = path.resolve()
    if (not path.is_absolute() or not any(resolved.is_relative_to(base)
            for base in ('/root/rl-code', '/tmp'))):
        raise ValueError('training compiler cache must resolve to container-local storage')
    training_plan['env']['ACTION_HEAD_PROCESS_COMPILE_CACHE_ROOT'] = str(resolved)


def node_environment(plan: dict, rank: int) -> dict:
    repo = plan['snapshot_repo']
    runtime = '/tmp/v6rd/' + hashlib.sha256(plan['output_root'].encode()).hexdigest()[:12]
    cache = runtime + '/cache'
    env = dict(os.environ)
    mapping = plan['source_object_map']
    env.update(SAGE_SOURCE_OBJECT_MAP=mapping['path'],
        SAGE_SOURCE_OBJECT_MAP_SHA256=mapping['sha256'],
        SAGE_REQUIRE_SOURCE_OBJECT_MAP='1')
    # This is a Ray job inside PyTorch worker containers, not a torchrun child.
    for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
                'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy',
                'SAGE_BENCH_PROTOCOL_ROOT'):
        env.pop(key, None)
    env.update(REPO=repo, EXPR_NAME=plan['display_name'],
        VERL_SFT_CONFIG_DIR=plan['upstream_repo'] + '/verl/trainer/config',
        VERL_SOURCE_ROOT=plan['upstream_repo'],
        RENDER_SERVER_ROOT=plan['renderer_repo'],
        RYNNBRAIN_NAV_DIR=plan['navigation_repo'],
        SERVICES_OUTPUT_DIR=f"{plan['output_root']}/services/node_{rank:02d}",
        TRAIN_GYMS=str(rollout_gyms_per_node(plan['config'])), VALIDATION_GYMS='0', PYTHON=sys.executable,
        THREE_SOURCE_CASE_MANIFEST=plan['config']['case_manifest'], CPU_RENDER_PORT='18480',
        SAGE_SPLIT='train', SAGE_GEOMETRY_MODE='legacy_xy',
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        PYTHONPATH=':'.join((plan['fast_kernel_overlay'], repo + '/verl',
                            plan['upstream_repo'], repo + '/vln_demo/src')),
        LD_LIBRARY_PATH='/opt/accl-p:/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/lib:'
                        '/usr/local/PPU_SDK/sailSHMEM/lib:/usr/local/lib:' + env.get('LD_LIBRARY_PATH', ''),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        ACTION_HEAD_REQUIRE_QWEN_FAST_PATH='1', ACTION_HEAD_FLA_L2NORM_DYNAMIC_NB='1',
        ACTION_HEAD_FLA_UNUSED_SHAPE_DYNAMIC='1', ACTION_HEAD_VLLM_WEIGHT_SYNC='inplace',
        RAY_DEDUP_LOGS='1', TMPDIR=runtime + '/tmp', XDG_CACHE_HOME=cache,
        HF_HOME=cache + '/hf', TRITON_CACHE_DIR=cache + '/triton',
        TORCHINDUCTOR_CACHE_DIR=cache + '/torchinductor', CUDA_CACHE_PATH=cache + '/cuda',
        PYTORCH_KERNEL_CACHE_PATH=cache + '/kernels', TORCH_EXTENSIONS_DIR=cache + '/extensions',
        RTC_CACHE_PATH=cache + '/rtc', PPU_LIB_RTC_CACHE_PATH=cache + '/rtc',
        VLLM_RPC_BASE_PATH=runtime + '/rpc',
        MPLCONFIGDIR=cache + '/matplotlib', NUMBA_CACHE_DIR=cache + '/numba',
        V6_DLC_LOCAL_RUNTIME=runtime)
    env['ACTION_HEAD_QUEUE_MMAP_DIR'] = runtime + '/queue-records'
    env['SAGE_RENDER_DEVICE'] = 'cpu' if plan.get('resource_mode') == 'cpu_colocated' else 'cuda'
    sage_parallelism = plan.get('sage_cpu_render_parallelism', 1)
    if (type(sage_parallelism) is not int or not 1 <= sage_parallelism <= 16
            or (env['SAGE_RENDER_DEVICE'] != 'cpu' and sage_parallelism != 1)):
        raise ValueError('SAGE CPU render parallelism must be 1..16; GPU mode requires 1')
    env['SAGE_CPU_RENDER_PARALLELISM'] = str(sage_parallelism)
    episode_cache = plan.get('native_episode_cache', False)
    if type(episode_cache) is not bool:
        raise ValueError('native episode cache must be an explicit boolean')
    env['VLN_NATIVE_EPISODE_CACHE'] = '1' if episode_cache else '0'
    native_cache = native_scene_cache_settings(plan)
    env['CPU_RENDER_SHARED_CACHE'] = '1' if native_cache['shared'] else '0'
    env['CPU_RENDER_CACHE_MAX_GIB'] = str(native_cache['max_gib'])
    env['CPU_RENDER_SCENE_LOAD_RESERVE_GIB'] = str(native_cache['reserve_gib'])
    if native_cache['shared']:
        # This formal cache treatment keeps the validated 32-context/4-queue
        # layout. Do not let a parent shell silently multiply residency.
        env['CPU_RENDER_CACHE_SCENES'] = '32'
        env['CPU_RENDER_PARALLELISM'] = '4'
    preparation_tasks = plan.get('cpu_group_preparation_tasks_per_node', 1)
    if (type(preparation_tasks) is not int or preparation_tasks not in (1, 2)
            or (preparation_tasks != 1 and (plan.get('resource_mode') != 'cpu_colocated'
                or not plan['config'].get('cpu_group_preparation')
                or not plan['config'].get('overlap_cpu_group_preparation')))):
        raise ValueError('two CPU preparation tasks per node require colocated CPU overlap')
    env['ACTION_HEAD_CPU_GROUP_PREP_TASKS_PER_NODE'] = str(preparation_tasks)
    env['SAGE_CPU_RENDER_CACHE'] = cache + '/sage_cpu_render'
    env['V6_SERVICES_RUNTIME_ROOT'] = runtime + '/services'
    env['V6_SERVICE_IO_ISOLATION'] = '1'
    env['ACTION_HEAD_FINAL_ABLATIONS'] = ','.join(plan.get('components', []))
    env['ACTION_HEAD_EXECUTION_PROFILE'] = plan.get('execution_profile', 'v7_v3_collision')
    env['RAY_agent_register_timeout_ms'] = '180000'
    env['ACTION_HEAD_LEARNER_PROGRESS_DIR'] = plan['output_root'] + '/train/learner_progress'
    env.pop('ACTION_HEAD_BACKWARD_REPLAY_DIR', None)
    rule_credit = plan.get('config', {}).get('tool_credit_mode') == 'rule_call_loo'
    # Gyms are launched before the trainer and do not inherit its Ray actor env.
    # Bind privileged scoring/depth collection explicitly on every DLC node.
    env['SAGE_RULE_CREDIT'] = '1' if rule_credit else '0'
    env['ACTION_HEAD_RULE_CREDIT'] = '1' if rule_credit else '0'
    env['ACTION_HEAD_RULE_TERMINAL_SUCCESS_REWARD'] = str(plan['config'].get('terminal_success_reward', 1.))
    env['ACTION_HEAD_QUEUE_PAYLOAD_CODEC'] = plan.get('queue_payload_codec', 'torch')
    env['ACTION_HEAD_TRAIN_ACTION_HEAD'] = '0' if plan.get('freeze_action_head') else '1'
    env['ACTION_HEAD_TRAIN_TEXT_POLICY'] = '0' if plan.get('freeze_text_policy') else '1'
    env['ACTION_HEAD_RULE_FIRST_ARRIVAL_REWARD'] = str(plan.get('first_arrival_reward', 1.))
    env['ACTION_HEAD_RETENTION_EVAL_ROOT'] = plan.get('checkpoint_retention_evaluation_root', '')
    env['ACTION_HEAD_SEQUENCE_RLOO'] = '1' if plan.get('sequence_rloo', False) else '0'
    coefficients = plan.get('rule_reward_coefficients', dict(distance=.2, information=.05))
    if set(coefficients) != {'distance', 'information'}:
        raise ValueError('rule reward coefficients require exactly distance and information')
    for name, key in [('distance', 'ACTION_HEAD_RULE_DISTANCE_COEFFICIENT'),
                      ('information', 'ACTION_HEAD_RULE_INFORMATION_COEFFICIENT')]:
        value = coefficients[name]
        if type(value) not in (int, float) or not __import__('math').isfinite(value) or value < 0:
            raise ValueError('invalid independent rule reward coefficient')
        env[key] = str(value)
    env['ACTION_HEAD_FINAL_CONTRACT'] = ('three-source-boundary-stop-v1'
        if rule_credit and plan['config'].get('rule_credit_version') == 'v4_boundary_stop_spl' else '')
    # A v3 task must not inherit the v4 masked policy from its parent shell.
    env['ACTION_HEAD_EXECUTABLE_POLICY'] = ('v6-executable-tools-v1'
        if env['ACTION_HEAD_FINAL_CONTRACT'] else '')
    if rule_credit and plan.get('config', {}).get('joint_text_fm_forward'):
        env['ACTION_HEAD_RULE_SHARED_AUDIT_DIR'] = plan['output_root'] + '/shared_gradient_audit'
    else:
        env.pop('ACTION_HEAD_RULE_SHARED_AUDIT_DIR', None)
    if plan['config'].get('text_backend') == 'megatron':
        if plan.get('resource_mode') == 'cpu_colocated':
            # The parent WebTerminal defaults disable IB. The validated PPU
            # multi-node recipe must not depend on incidental launcher exports:
            # r14 passed128 ranks and two full updates with this exact tuple.
            env.update(NCCL_IB_DISABLE='0', NCCL_NET='IB', NCCL_NET_PLUGIN='none',
                       NCCL_MIN_NCHANNELS='4', NCCL_MAX_NCHANNELS='4')
            # PCCL's optional per-collective statistics allocate from a finite
            # nccl_call pool, independent of CUDA/Work completion. Long Bridge
            # exports exhausted it even with per-collective waits. Disable
            # that telemetry, not communication or platform hang detection.
            env['ACCL_C4_STATS_MODE'] = 'OFF'
        environment = Path(plan['megatron_environment'])
        if not environment.is_absolute() or str(environment.resolve()).startswith(('/data/storage', '/data/storage')):
            raise ValueError('Megatron dependencies must be container-local Git checkouts')
        for name, revision in [('mcore', '4b4acac9a1d28ea6829c8d4f566d75698a21249d'),
                               ('bridge', 'c0e164ed2aedac4ad1c877780e2564a19d5d54ec')]:
            actual = subprocess.check_output(['git', '-C', str(environment/name), 'rev-parse', 'HEAD'], text=True).strip()
            if actual != revision:
                raise ValueError('Megatron dependency revision mismatch: '+name)
        env['PYTHONPATH'] = ':'.join((str(environment/'mcore'), str(environment/'bridge/src'), env['PYTHONPATH']))
        env['ACTION_HEAD_MEGATRON_ENV'] = str(environment)
        env['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
    return env


def validate_cluster(nodes: list[dict], expected_ips: set[str], *, ray_gpus_per_node=15) -> dict:
    active = [node for node in nodes if node.get('Alive')]
    by_ip = {node['NodeManagerAddress']: node for node in active}
    if not expected_ips or set(by_ip) != expected_ips or len(active) != len(expected_ips):
        raise ValueError(f'expected {len(expected_ips)} isolated Ray nodes {expected_ips}; got {set(by_ip)}')
    for ip, node in by_ip.items():
        if node['Resources'].get('GPU') != ray_gpus_per_node:
            raise ValueError(f'{ip} advertises {node["Resources"].get("GPU")} GPUs; renderer layout expects {ray_gpus_per_node}')
        if node['Resources'].get('CPU', 0) < 138:
            raise ValueError(f'{ip} has too few Ray CPUs for 12 learners and 16 rollouts')
    return {'ray_nodes': len(active), 'ray_gpus': ray_gpus_per_node * len(active),
            'renderer_gpus': (16 - ray_gpus_per_node) * len(active),
            'total_gpus': 16 * len(active), 'node_ips': sorted(by_ip)}


def audit_ray_cluster(ray, address, expected_ips, nodes, ray_gpus_per_node, until):
    """Ray installs driver signal handlers; retain the worker's cleanup handler."""
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    def restore():
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    try:
        ray.init(address=address, log_to_driver=False)
        restore()
        until(lambda: len([item for item in ray.nodes() if item.get('Alive')]) == nodes)
        return validate_cluster(ray.nodes(), expected_ips, ray_gpus_per_node=ray_gpus_per_node)
    finally:
        try:
            ray.shutdown()
        finally:
            restore()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-plan', type=Path, required=True)
    args = parser.parse_args()
    # Bind the existing storage retry helper by exact frozen file, before
    # shared coordinator reads. Never hide non-ESTALE or unrelated ENOENT.
    retry_path = Path(__file__).with_name('dlc_control_io_retry.py')
    retry_spec = importlib.util.spec_from_file_location('_v6_worker_control_io_retry', retry_path)
    retry_module = importlib.util.module_from_spec(retry_spec)
    retry_spec.loader.exec_module(retry_module)
    with retry_module.bounded(time.monotonic() + 1800):
        plan = json.loads(retry_module.retry_estale(
            lambda: args.launch_plan.read_text(), f'read_launch_plan:{args.launch_plan}'))
    retry_module.install(sys.modules[__name__], Path(plan['output_root']), stats=True)
    print(json.dumps(dict(event='git_worker_storage_retry',
        helper_file=retry_module.__file__,
        helper_sha256=hashlib.sha256(retry_path.read_text().encode()).hexdigest(),
        retries='ESTALE_reopen_and_exact_own_atomic_temporary_only', stats=True)), flush=True)
    nodes = plan['nodes']
    rank = node_rank(os.environ, nodes)
    if plan['config']['nnodes'] != nodes or os.environ.get('WORLD_SIZE') not in (None, str(nodes)):
        raise RuntimeError('DLC WORLD_SIZE and trainer nnodes must match the launch plan')
    output = Path(plan['output_root'])
    repo = Path(plan['snapshot_repo'])
    coord = output / 'coordination'
    logs = output / 'logs'
    coord.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    # Preserve the real Pod console before redirecting verbose logs to disk.
    # AIMaster otherwise sees no output and kills healthy long rollout steps.
    console_fd = os.dup(1)
    progress = ConsoleProgress(output, rank, console_fd)
    from dlc_workspace_stdout_heartbeat import identity
    controller = {key: value for key, value in identity(os.getpid()).items()
                  if key in ('pid', 'start_ticks')}
    status_path = coord / f'node_{rank:02d}.json'
    if status_path.exists():
        raise RuntimeError('this DLC worker rank already started; use a new attempt')
    atomic_json(status_path, dict(rank=rank, pid=os.getpid(), controller=controller,
        hostname=socket.gethostname(), phase='initializing', stage='initializing',
        gpu_mode='unknown', occupier_pid=None, time=time.time()))
    heartbeat_root = output / 'heartbeat' / f'node_{rank:02d}'
    heartbeat_root.mkdir(parents=True, exist_ok=False)
    with os.fdopen(os.dup(console_fd), 'wb', buffering=0) as console:
        heartbeat = subprocess.Popen([sys.executable, '-B', '-u',
            str(repo / 'scripts/train/dlc_workspace_stdout_heartbeat.py'),
            '--controller-pid', str(controller['pid']),
            '--controller-start-ticks', controller['start_ticks'],
            '--phase-file', str(status_path), '--progress-root', str(output),
            '--output', str(heartbeat_root)], stdout=console, stderr=console,
            stdin=subprocess.DEVNULL, start_new_session=True)
    atomic_json(heartbeat_root / 'launch.json', dict(pid=heartbeat.pid,
        controller=controller, original_stdout_inherited=True, interval_seconds=20))
    # All stdout and trainer/Ray/service logs are persistent experiment data.
    log = (logs / f'worker_{rank:02d}_{socket.gethostname()}.log').open('a', buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    env = node_environment(plan, rank)
    runtime = Path(env['V6_DLC_LOCAL_RUNTIME'])
    for key in ('TMPDIR', 'XDG_CACHE_HOME', 'HF_HOME', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR',
                'CUDA_CACHE_PATH', 'PYTORCH_KERNEL_CACHE_PATH', 'TORCH_EXTENSIONS_DIR',
                'RTC_CACHE_PATH', 'MPLCONFIGDIR', 'NUMBA_CACHE_DIR'):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    os.environ.clear()
    os.environ.update(env)
    import ray
    from ray._private.services import get_node_ip_address
    ip = get_node_ip_address()
    if ip.startswith('127.'):
        raise RuntimeError('cross-node Ray requires a routable local address')
    record = {'rank': rank, 'hostname': socket.gethostname(), 'ip': ip, 'pid': os.getpid()}
    occupiers = {}
    def status(phase, **fields):
        standby = next((child.pid for child in occupiers.values() if child.poll() is None), None)
        atomic_json(status_path, {**record, 'controller': controller, 'phase': phase,
            'stage': phase, 'gpu_mode': 'standby' if standby else 'component_processes',
            'occupier_pid': standby, 'time': time.time(), **fields})
        print(json.dumps({'rank': rank, 'phase': phase, **fields}), flush=True)
        progress.emit({'phase': phase, **fields})
    processes = []
    handles = []
    service_started = False
    service_script = repo / 'scripts/train/v6_rl_local_services.sh'
    def spawn(command, label, child_env=env):
        handle = (logs / f'{label}_node_{rank:02d}.log').open('ab', buffering=0)
        handles.append(handle)
        child = subprocess.Popen(command, env=child_env, cwd=repo, stdout=handle,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        processes.append((label, child))
        return child
    def standby(gpus, label):
        ready = output / 'standby' / f'node_{rank:02d}_{label}.json'
        child = spawn([sys.executable, '-B', '-u',
            str(repo / 'verl/scripts/train/action_head_sft_data/gpu_occupier.py'),
            '--gpus', gpus, '--ready-file', str(ready)],
            'standby_' + label)
        occupiers[label] = child
        until(lambda: ready.exists(), timeout=120)
        return child
    def release_standby(label):
        child = occupiers.pop(label, None)
        if child is None:
            return
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
        processes.remove(('standby_' + label, child))
    def check_children(ignore_labels=()):
        if heartbeat.poll() is not None:
            raise RuntimeError('independent original-stdout heartbeat exited')
        if (coord / 'trainer_starting.json').exists():
            release_standby('learner')
            released = coord / f'standby_released_{rank:02d}.json'
            if not released.exists():
                # Refresh ownership before acknowledging the barrier. Peers
                # otherwise retain a dead occupier PID throughout training.
                status('starting_trainer' if rank == 0 else 'serving',
                       learner_standby_released=True)
                atomic_json(released, dict(rank=rank, time=time.time(), own_standby_released=True))
        for label, child in processes:
            if label not in ignore_labels and child.poll() is not None:
                raise RuntimeError(f'{label} exited unexpectedly: {child.returncode}')
        for path in coord.glob('node_*.json'):
            peer = json.loads(path.read_text())
            if peer['phase'] == 'failed':
                raise RuntimeError(f"worker {peer['rank']} failed: {peer.get('error')}")
        if service_started:
            records = [json.loads(line) for line in (Path(env['SERVICES_OUTPUT_DIR']) / 'owned_pids.jsonl').read_text().splitlines()]
            for owned in records:
                try:
                    fields = Path(f"/proc/{owned['pid']}/stat").read_text().rsplit(')', 1)[1].split()
                except FileNotFoundError:
                    raise RuntimeError(f"service {owned['kind']} exited") from None
                if fields[19] != owned['start_ticks'] or fields[0] == 'Z':
                    raise RuntimeError(f"service {owned['kind']} no longer has the recorded process identity")
        progress.poll()
    def until(predicate, timeout=1800):
        deadline = time.monotonic() + timeout
        with retry_module.bounded(deadline):
            while not predicate():
                check_children()
                if time.monotonic() > deadline:
                    raise TimeoutError('node startup coordination timed out')
                time.sleep(3)
    def signal_failure(signum, _frame):
        raise InterruptedError(f'received signal {signum}')
    signal.signal(signal.SIGTERM, signal_failure)
    signal.signal(signal.SIGINT, signal_failure)
    try:
        status('verifying')
        sys.path.insert(0, str(repo / 'scripts/train'))
        if plan.get('code_mode') != 'git_checkout' or repo.resolve() != Path(__file__).resolve().parents[2]:
            raise ValueError('formal worker must execute from its container-local Git checkout')
        if repo.resolve().is_relative_to('/data/storage') or repo.resolve().is_relative_to('/data/storage'):
            raise ValueError('formal worker checkout cannot be on shared storage')
        actual_commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        if actual_commit != plan['code_commit']:
            raise ValueError('formal worker Git commit differs from the task specification')
        status('verified', code_commit=actual_commit, code_path=str(repo), helper_file=retry_module.__file__)
        count = subprocess.check_output([sys.executable, '-B', '-c',
            'import torch; print(torch.cuda.device_count())'], env=env, text=True).strip()
        if count != '16':
            raise RuntimeError(f'worker must own a complete 16-GPU node; detected {count!r}')
        ray_gpus_per_node = 16 if plan.get('resource_mode') == 'cpu_colocated' else 15
        standby(','.join(map(str, range(ray_gpus_per_node))), 'learner')
        if ray_gpus_per_node == 15:
            standby('15', 'renderer')
        status('starting_services')
        if ray_gpus_per_node == 15:
            release_standby('renderer')
        service_started = True
        subprocess.run(['bash', str(service_script), 'start'], env=env, check=True)
        benches = node_bench_urls(plan, ip)
        atomic_json(coord / f'bench_{rank:02d}.json', {**record, 'urls': benches})
        status('starting_ray')
        ray_env = {**env, 'CUDA_VISIBLE_DEVICES': ','.join(map(str, range(ray_gpus_per_node)))}
        command = [sys.executable, '-m', 'ray.scripts.scripts', 'start', '--block',
                   '--node-ip-address', ip, '--num-gpus', str(ray_gpus_per_node), '--num-cpus', '160',
                   '--disable-usage-stats']
        if rank == 0:
            require_unused_ray_head(ip)
            command += ['--head', '--port', '6379', '--include-dashboard', 'false',
                        '--temp-dir', str(runtime / 'ray')]
            spawn(command, 'ray', ray_env)
            def head_ready():
                with socket.socket() as connection:
                    connection.settimeout(1)
                    return connection.connect_ex((ip, 6379)) == 0
            until(head_ready)
            atomic_json(coord / 'ray_head.json', {'ip': ip, 'address': f'{ip}:6379'})
        else:
            until(lambda: (coord / 'ray_head.json').is_file())
            head = json.loads((coord / 'ray_head.json').read_text())
            command += ['--address', head['address']]
            spawn(command, 'ray', ray_env)
        status('ray_joining')
        if rank == 0:
            until(lambda: all((coord / f'bench_{n:02d}.json').is_file() for n in range(nodes)))
            manifests = [json.loads((coord / f'bench_{n:02d}.json').read_text()) for n in range(nodes)]
            expected_ips = {item['ip'] for item in manifests}
            if len(expected_ips) != nodes:
                raise ValueError(f'{nodes} DLC workers must occupy {nodes} distinct nodes')
            audit = audit_ray_cluster(ray, f'{ip}:6379', expected_ips, nodes,
                                      ray_gpus_per_node, until)
            atomic_json(output / 'launch/resource_audit.json', audit)
            benches = [url for item in manifests for url in item['urls']]
            if (len(benches) != plan['config']['rollout_actors']
                    or len(set(benches)) != plan['config']['rollout_actors']
                    or any(len(item['urls']) != rollout_gyms_per_node(plan['config'])
                           for item in manifests)):
                raise ValueError('every rollout actor requires its own independent gym')
            # Every node executed reset+JPEG probes; the trainer will additionally
            # validate its native harness, task manifest and full-R3 contract.
            import v6_rl
            recipe = json.loads(Path(plan['runtime_config']).read_text())
            recipe['ray_address'] = f'{ip}:6379'
            initialization_kwargs = training_initialization_kwargs(plan)
            training_plan = v6_rl.build_plan(recipe,
                head_checkpoint=plan['initial_checkpoint'] + '/action_head',
                query_checkpoint=plan['initial_checkpoint'] + '/dualvln_query',
                bench_urls=','.join(benches), output_dir=str(output / 'train'), **initialization_kwargs)
            apply_training_compiler_cache(training_plan, ray_env)
            training_env = {**ray_env, **training_plan['env']}
            training_env['PYTHONPATH'] = training_env['PYTHONPATH'].replace(
                '/workspace/verl', plan['snapshot_upstream'])
            if recipe.get('text_backend') == 'megatron':
                # build_plan constructs the legacy FSDP import path. Preserve
                # the node's already verified MCore/Bridge roots ahead of the
                # vendor namespace in both driver and Ray worker processes.
                training_env['PYTHONPATH'] = ray_env['PYTHONPATH']
            training_plan['env']['PYTHONPATH'] = training_env['PYTHONPATH']
            training_plan['env']['VERL_SFT_CONFIG_DIR'] = training_env['VERL_SFT_CONFIG_DIR']
            for key, value in training_env.items():
                if key in ('TMPDIR', 'XDG_CACHE_HOME', 'HF_HOME', 'TRITON_CACHE_DIR',
                           'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH', 'PYTORCH_KERNEL_CACHE_PATH',
                           'TORCH_EXTENSIONS_DIR'):
                    Path(value).mkdir(parents=True, exist_ok=True)
            atomic_json(output / 'launch/effective_training_plan.json', training_plan)
            atomic_json(coord / 'trainer_starting.json', dict(time=time.time(), rank=0))
            release_standby('learner')
            until(lambda: all((coord / f'standby_released_{n:02d}.json').exists()
                              for n in range(1, nodes)))
            status('training', resource_audit=audit)
            trainer = spawn(training_plan['command'], 'trainer', training_env)
            while trainer.poll() is None:
                # Exclude the trainer from generic checks so normal completion
                # between poll and check cannot be mistaken for a failure.
                # Keep it registered throughout: a peer/service failure here
                # must still reap the trainer in the ownership cleanup.
                check_children(ignore_labels=('trainer',))
                time.sleep(5)
            processes.remove(('trainer', trainer))
            if trainer.returncode:
                raise RuntimeError(f'trainer exited with code {trainer.returncode}')
            atomic_json(coord / 'training_done.json', {'status': 'succeeded', 'time': time.time()})
        else:
            status('serving')
            until(lambda: (coord / 'training_done.json').is_file(), timeout=60 * 60 * 24 * 14)
        status('succeeded')
        return 0
    except BaseException as exc:
        record_failure(status, progress, exc)
        try:
            traceback.print_exc()
        except OSError:
            pass
        return 1
    finally:
        for _, child in reversed(processes):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
        if service_started:
            subprocess.run(['bash', str(service_script), 'stop'], env=env, check=False)
        # The task owns an exclusive container. Restore the existing standby
        # workload after real GPU children exit; Pod teardown will reap it.
        ready = output / 'standby' / f'node_{rank:02d}_postrun.json'
        if plan.get('launcher_manages_standby'):
            # The Git launcher reserves after this worker has reaped every GPU
            # child. A second holder here would duplicate CUDA allocations.
            if heartbeat.poll() is None:
                os.killpg(heartbeat.pid, signal.SIGTERM)
                heartbeat.wait(timeout=10)
            for handle in handles:
                handle.close()
            os.close(console_fd)
        else:
            restore_postrun_standby(output, rank, repo, env, logs, status_path, progress)
            if heartbeat.poll() is None:
                os.killpg(heartbeat.pid, signal.SIGTERM)
                heartbeat.wait(timeout=10)
            for handle in handles:
                handle.close()
            os.close(console_fd)


def restore_postrun_standby(output, rank, repo, env, logs, status_path, progress):
    ready = output / 'standby' / f'node_{rank:02d}_postrun.json'
    with (logs / f'standby_postrun_node_{rank:02d}.log').open('ab', buffering=0) as standby_log:
        restored = subprocess.Popen([sys.executable, '-B', '-u',
            str(repo / 'verl/scripts/train/action_head_sft_data/gpu_occupier.py'),
            '--gpus', ','.join(map(str, range(16))), '--require-gpu-count', '16',
            '--ready-file', str(ready)], env=env, stdout=standby_log,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    deadline = time.monotonic() + 120
    while not ready.exists() and restored.poll() is None and time.monotonic() < deadline:
        time.sleep(1)
    atomic_json(output / 'standby' / f'node_{rank:02d}_postrun_launch.json',
        dict(pid=restored.pid, ready=ready.exists(), returncode=restored.poll(),
             lifecycle='owned exclusive Pod teardown', time=time.time()))
    postrun = json.loads(status_path.read_text())
    postrun.update(gpu_mode='standby' if ready.exists() else 'unknown',
        occupier_pid=restored.pid if restored.poll() is None else None,
        postrun_standby_ready=ready.exists(), time=time.time())
    atomic_json(status_path, postrun)
    progress.emit(dict(event='postrun_standby', pid=restored.pid,
                       ready=ready.exists(), returncode=restored.poll()))
    # Containers are exclusive to this launch. Cached kernels remain local
    # until all owned processes stop, then container teardown releases them.


if __name__ == '__main__':
    raise SystemExit(main())
