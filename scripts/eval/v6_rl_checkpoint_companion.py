#!/usr/bin/env python3
"""Retain committed RL inference weights and evaluate the fixed SAGE companion.

`stage` runs on the submitting workstation, including while the GPU job queues.
`evaluate` runs in the single-node DLC job. Neither mode modifies training.
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[2]
SWEEP = REPO / 'verl/scripts/eval/qwen36_4query_ckpt_sweep'
sys.path[:0] = [str(SWEEP), str(REPO / 'verl'), str(REPO / 'vln_demo/src')]
from common import atomic_json, checkpoint_identity, sha256_file

TERMINAL = {'Stopped', 'Failed', 'Succeeded', 'Completed', 'Cancelled'}


def pin_checkpoint(source: Path, destination: Path) -> dict:
    """Hard-link immutable inference files, surviving the trainer's unlink.

    No optimizer state is retained. Publication is an atomic directory rename;
    interrupted staging stays private and is rebuilt on retry.
    """
    if destination.exists():
        return checkpoint_identity(destination)
    identity = checkpoint_identity(source)
    stage = destination.parent / ('.' + destination.name + '.staging')
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    files = [source / 'report.json']
    for name in ('qwen', 'action_head', 'dualvln_query'):
        files.extend(p for p in (source / name).rglob('*') if p.is_file())
    manifest = []
    for path in files:
        if path.is_symlink():
            raise ValueError(f'checkpoint must contain regular files: {path}')
        relative = path.relative_to(source)
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, target)  # Same data filesystem; fail instead of silently copying TBs.
        a, b = path.stat(), target.stat()
        if (a.st_dev, a.st_ino, a.st_size) != (b.st_dev, b.st_ino, b.st_size):
            raise ValueError(f'checkpoint retention mismatch: {path}')
        manifest.append({'path': str(relative), 'bytes': a.st_size,
                         'inode': a.st_ino, 'device': a.st_dev})
    if checkpoint_identity(source) != identity:
        raise ValueError('checkpoint changed while retaining inference weights')
    atomic_json(stage / 'retention.json', {'source': str(source),
                'source_identity': identity, 'files': manifest, 'time': time.time()})
    stage.rename(destination)
    return checkpoint_identity(destination)


def stage_once(config: dict, upstream: dict) -> dict:
    root = Path(config['output_root'])
    source = Path(config['training_root']) / 'train'
    retained = root / 'checkpoints'
    retained.mkdir(parents=True, exist_ok=True)
    contract_path = source / 'contract.json'
    deferred = config.get('training_contract_sha256') is None
    if deferred:
        # New training and its companion can queue together before the trainer
        # publishes its runtime contract. Bind the immutable plan immediately.
        plan_path = Path(config['training_root']) / 'launch/submit_plan.json'
        if sha256_file(plan_path) != config['training_plan_sha256']:
            raise ValueError('bound training plan changed')
        if not contract_path.exists():
            state = {'time': time.time(), 'upstream': upstream,
                     'latest_committed': 0, 'staged_steps': [], 'errors': {},
                     'waiting_for_training_contract': True}
            atomic_json(root / 'staging_status.json', state)
            return state
        contract = json.loads(contract_path.read_text())
        for key, expected in config['expected_training_contract'].items():
            if contract.get(key) != expected:
                raise ValueError(f'bound training contract differs: {key}')
    elif sha256_file(contract_path) != config['training_contract_sha256']:
        raise ValueError('bound training contract changed')
    retained_contract = retained / 'contract.json'
    if retained_contract.exists():
        if sha256_file(retained_contract) != sha256_file(contract_path):
            raise ValueError('retained training contract changed')
    else:
        shutil.copy2(contract_path, retained_contract)
    latest_file = source / 'latest_checkpointed_iteration.txt'
    latest = int(latest_file.read_text().strip()) if latest_file.exists() else 0
    staged, errors = [], {}
    for step in config['steps']:
        if step > latest:
            continue
        name = f'global_step_{step:06d}'
        try:
            identity = pin_checkpoint(source / name, retained / name)
            staged.append(step)
            record = root / 'retention' / (name + '.json')
            if not record.exists():
                atomic_json(record, {'checkpoint': identity,
                    'retention': json.loads((retained / name / 'retention.json').read_text())})
        except (OSError, ValueError) as exc:
            errors[step] = repr(exc)
    state = {'time': time.time(), 'upstream': upstream, 'latest_committed': latest,
             'staged_steps': staged, 'errors': errors}
    atomic_json(root / 'staging_status.json', state)
    return state


def platform_training_status(config: dict) -> dict:
    """Query the bound task through the authorized platform CLI only."""
    task_id = str(config['training_task_id'])
    result = subprocess.run([
        '/workspace/bin/train-platform',
        'detail', '--task-id', task_id,
    ], capture_output=True, text=True, check=True, timeout=45)
    envelope = json.loads(result.stdout)
    if not 200 <= envelope['http_status'] < 300 or envelope['response']['code'] != 2000:
        raise RuntimeError('platform task detail request failed')
    body = envelope['response']['data']
    expected = {'taskId': task_id, 'jobId': config['training_job_id'],
                'createdBy': 'anonymous', 'quotaId': 'quota-anonymous',
                'platType': '2', 'spaceId': '150120'}
    for key, value in expected.items():
        if str(body.get(key)) != value:
            raise ValueError(f'bound training task differs: {key}')
    return {'task_id': task_id, 'job_id': body['jobId'],
            'status': body['status'], 'observed_at': time.time()}


def stage(config: dict) -> None:
    root = Path(config['output_root'])
    with (root / 'staging.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(root / 'staging_process.json', {'pid': os.getpid(),
                    'started': time.time(), 'config': config})
        while True:
            upstream = {'job_id': config['training_job_id'], 'status': 'Unknown'}
            try:
                upstream = platform_training_status(config)
            except (OSError, subprocess.SubprocessError, RuntimeError,
                    json.JSONDecodeError, KeyError) as exc:
                upstream['error'] = type(exc).__name__
            try:
                state = stage_once(config, upstream)
            except OSError as exc:
                if exc.errno != errno.ESTALE:
                    raise
                # Reopen shared CPFS paths on the next poll. An unavailable
                # read is not a new publication or a successful retention.
                print(json.dumps({'event': 'checkpoint_staging_read_retry',
                      'time': time.time(), 'error': repr(exc),
                      'upstream': upstream}), flush=True)
                time.sleep(30)
                continue
            print(json.dumps(state), flush=True)
            if len(state['staged_steps']) == len(config['steps']):
                return
            if upstream['status'] in TERMINAL:
                return  # Remote consumer reports missing steps explicitly.
            if (root / 'evaluation/SWEEP_COMPLETE.json').exists():
                return
            time.sleep(30)


def activity(stop: threading.Event, directory: Path) -> None:
    """Expose real file activity to Pod logs; unchanged files are not progress."""
    previous = {}
    while not stop.wait(30):
        changed = []
        for pattern in ('logs/*.log', 'bootstrap_runtime/logs/*.log',
                        'closedloop/summary.json', 'closedloop/rollouts/**/*.json'):
            for p in directory.glob(pattern):
                try:
                    st = p.stat()
                except FileNotFoundError:
                    continue
                signature = (st.st_size, st.st_mtime_ns)
                if previous.get(str(p)) != signature and st.st_size:
                    changed.append({'path': str(p.relative_to(directory)), 'bytes': st.st_size})
                    previous[str(p)] = signature
        if changed:
            print(json.dumps({'event': 'evaluation_file_activity', 'time': time.time(),
                  'changed_count': len(changed), 'files': changed[:12],
                  'meaning': 'file activity; not proof of valid completed episodes'}), flush=True)


class WaitingGpuReservation:
    """Reuse the GPU occupier only between evaluations; CUDA lives in the child."""
    def __init__(self, config: dict, output: Path, stop_process_group):
        self.config = config.get('waiting_gpu_reservation', {})
        self.output = output
        self.stop_process_group = stop_process_group
        self.process = None
        self.ready = output / 'waiting_gpu_ready.json'

    def status(self, active: bool, reason: str) -> None:
        state = {'time': time.time(), 'active': active, 'reason': reason,
                 'phase': 'waiting_checkpoint_gpu_reservation',
                 'meaning': 'reservation compute; not evaluation progress',
                 'pid': self.process.pid if self.process is not None else None,
                 'gpu_count': self.config.get('gpu_count', 16)}
        atomic_json(self.output / 'waiting_gpu_status.json', state)
        print(json.dumps({'event': 'waiting_gpu_reservation', **state}), flush=True)

    def check(self) -> None:
        if self.process is not None and self.process.poll() is not None:
            code = self.process.returncode
            self.status(False, f'unexpected_exit_{code}')
            raise RuntimeError(f'waiting GPU occupier exited unexpectedly: {code}')

    def start(self) -> None:
        if not self.config.get('enabled', False):
            return
        self.check()
        if self.process is not None:
            return
        self.ready.unlink(missing_ok=True)
        log = self.output / 'logs/waiting_gpu_occupier.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [self.config['python'], self.config['script'], '--auto',
                   '--require-gpu-count', str(self.config['gpu_count']),
                   '--ready-file', str(self.ready)]
        with log.open('a') as sink:
            self.process = subprocess.Popen(command, stdout=sink, stderr=subprocess.STDOUT,
                                            start_new_session=True)
        self.status(False, 'starting')
        deadline = time.monotonic() + self.config.get('startup_timeout', 120)
        try:
            while not self.ready.exists():
                self.check()
                if time.monotonic() >= deadline:
                    raise TimeoutError('waiting GPU occupier did not become ready')
                time.sleep(0.2)
            self.check()
            ready = json.loads(self.ready.read_text())
            count = self.config['gpu_count']
            if ready['pid'] != self.process.pid or ready['gpu_count'] != count or ready['gpus'] != list(range(count)):
                raise ValueError('waiting GPU occupier readiness does not cover all requested GPUs')
            self.status(True, 'waiting_for_checkpoint')
        except BaseException:
            self.stop('startup_failed')
            raise

    def stop(self, reason: str) -> None:
        if self.process is not None:
            self.stop_process_group(self.process)
            if self.process.poll() is None:
                raise RuntimeError('GPU occupier still alive; refusing to start evaluation')
            self.process = None
            self.ready.unlink(missing_ok=True)
            self.status(False, reason)

    def wait(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.check()
            time.sleep(min(5, max(0, deadline - time.monotonic())))


def evaluate(config: dict) -> int:
    import watch_checkpoints as sweep
    root = Path(config['output_root'])
    output = root / 'evaluation'
    output.mkdir(exist_ok=True)
    reservation = WaitingGpuReservation(config, output, sweep.stop_process_group)
    try:
        return _evaluate(config, sweep, reservation)
    finally:
        reservation.stop('watcher_exit')


def _evaluate(config: dict, sweep, reservation: WaitingGpuReservation) -> int:
    root = Path(config['output_root'])
    output = root / 'evaluation'
    for key in ('subset', 'protocol'):
        if sha256_file(config[key]) != config[key + '_sha256']:
            raise ValueError(f'{key} changed')
    os.environ['COMPANION_RL_TRAIN_ROOT'] = str(root / 'checkpoints')
    args = argparse.Namespace(**config['eval_args'], output_root=str(output))
    repeats = args.closedloop_repeats
    failures = {}
    attempts = {}
    while True:
        for step in config['steps']:
            dest = output / f'global_step_{step:06d}'
            checkpoint = root / 'checkpoints' / dest.name
            if sweep.step_complete(dest, repeats) or not checkpoint.is_dir():
                continue
            if attempts.get(step, 0) >= 2:
                continue
            reservation.check()
            reservation.stop('before_checkpoint_evaluation')
            attempts[step] = attempts.get(step, 0) + 1
            print(json.dumps({'event': 'evaluate', 'step': step, 'time': time.time()}), flush=True)
            stop = threading.Event()
            worker = threading.Thread(target=activity, args=(stop, dest), daemon=True)
            worker.start()
            try:
                sweep.evaluate_one(args, checkpoint, dest)
                if not sweep.step_complete(dest, repeats):
                    raise ValueError('evaluation returned without a valid completion')
                failures.pop(step, None)
            except Exception as exc:
                failures[step] = repr(exc)
                atomic_json(dest / f'FAILED_ATTEMPT_{attempts[step]}.json',
                            {'step': step, 'error': repr(exc)})
                print(json.dumps({'event': 'evaluation_failed', 'step': step,
                                  'error': repr(exc)}), flush=True)
            finally:
                stop.set()
                worker.join(timeout=2)
        done = [s for s in config['steps'] if sweep.step_complete(
                output / f'global_step_{s:06d}', repeats)]
        producer_path = root / 'staging_status.json'
        try:
            producer = json.loads(producer_path.read_text()) if producer_path.exists() else {}
        except OSError as exc:
            if exc.errno != errno.ESTALE:
                raise
            # Reopen the shared path on the next poll, as in stage(). Keep
            # attempts and valid results; unavailable status cannot end a sweep.
            print(json.dumps({'event': 'checkpoint_producer_read_retry',
                  'time': time.time(), 'error': repr(exc),
                  'completed_steps': done, 'attempts': attempts}), flush=True)
            reservation.start()
            reservation.wait(config.get('poll_seconds', 60))
            continue
        terminal = producer.get('upstream', {}).get('status') in TERMINAL
        all_staged = set(config['steps']).issubset(producer.get('staged_steps', []))
        state = {'time': time.time(), 'completed_steps': done, 'failed_steps': failures,
                 'attempts': attempts, 'producer': producer}
        atomic_json(output / 'watcher_status.json', state)
        pending = [s for s in config['steps'] if s not in done]
        if not pending or ((terminal or all_staged) and all(
                attempts.get(s, 0) >= 2 or s not in producer.get('staged_steps', []) for s in pending)):
            atomic_json(output / 'SWEEP_COMPLETE.json', dict(state, evaluation_valid=not pending,
                                                           missing_or_failed_steps=pending))
            return int(bool(pending))
        print(json.dumps({'event': 'waiting_for_checkpoint_or_retry', **state}), flush=True)
        reservation.start()
        reservation.wait(config.get('poll_seconds', 60))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['stage', 'evaluate'])
    p.add_argument('--config', required=True, type=Path)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if args.mode == 'stage':
        stage(config)
    else:
        def terminate(signum, frame):
            raise SystemExit(128 + signum)  # evaluate's finally releases all GPUs.
        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        lock = (Path(config['output_root']) / 'evaluation.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.exit(evaluate(config))


if __name__ == '__main__':
    main()
