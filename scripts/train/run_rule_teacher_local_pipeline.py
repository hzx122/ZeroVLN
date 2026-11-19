#!/usr/bin/env python3
"""Detached, fail-closed local collect -> validate -> export -> assemble -> train.

The plan supplies existing stage commands, not a second collection or training
implementation. Commands within a stage run concurrently; stages are serial.
Each stage requires explicit JSON completion evidence in addition to exit zero.
A failed attempt is immutable and is never silently retried.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'scripts/train'), str(REPO/'scripts/eval')]
from prepare_rule_teacher_three_source import ART, save, sha256
from supervise_rule_teacher_collection import CACHE_KEYS, read_status_json, collection_runtime_root
from watch_v6_rl_local_credit import process_identity, same_process_alive
from v6_rl_checkpoint_companion import WaitingGpuReservation
from watch_checkpoints import stop_process_group

KINDS = ('collect', 'validate', 'export', 'assemble', 'initialize', 'train')


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def verify_hashes(values):
    for name, expected in values.items():
        if sha256(Path(name)) != expected:
            raise ValueError(f'frozen input changed: {name}')


def verify_receipts(receipts):
    if not receipts:
        raise ValueError('every stage requires explicit completion/admission receipts')
    observed = []
    for item in receipts:
        path = Path(item['path'])
        if item.get('sha256') and sha256(path) != item['sha256']:
            raise ValueError(f'receipt changed: {path}')
        value = json.loads(path.read_text())
        if not item.get('equals'):
            raise ValueError('receipt must assert explicit JSON fields')
        for dotted, expected in item['equals'].items():
            actual = value
            for key in dotted.split('.'):
                actual = actual[key]
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(f'gate not met: {path}: {dotted}: {actual!r}')
        observed.append(dict(path=str(path.resolve()), sha256=sha256(path)))
    return observed


def validate_plan(plan, plan_path):
    output = Path(plan['output_root']).resolve()
    runtime = Path(plan['runtime']).resolve()
    allowed = collection_runtime_root()
    if (plan.get('schema') != 'rule_teacher_local_pipeline_v1'
            or not output.is_relative_to(ART) or output != plan_path.resolve().parent
            or not runtime.is_relative_to(allowed) or runtime == allowed):
        raise ValueError('pipeline schema/output/runtime boundary is invalid')
    stages = plan['stages']
    kinds = [entry['kind'] for entry in stages]
    if kinds != list(KINDS):
        raise ValueError(f'one serial stage of each kind required: {KINDS}')
    names = set()
    for stage in stages:
        if not stage.get('commands') or not stage.get('required_receipts'):
            raise ValueError('stage commands and required receipts are mandatory')
        idle = stage.get('idle_gpu_ids', [])
        if (len(set(idle)) != len(idle) or any(type(i) is not int or not 0 <= i < 16 for i in idle)):
            raise ValueError('idle GPU IDs must be distinct physical IDs in 0..15')
        if stage['kind'] in ('initialize', 'train') and idle:
            raise ValueError('training/initialization require releasing all reservations')
        for entry in stage['commands']:
            name = entry['name']
            if name in names or not name.replace('-', '').replace('_', '').isalnum():
                raise ValueError('commands require unique simple names')
            names.add(name)
    for entry in plan.get('services', []):
        if (entry['name'] in names
                or not entry['name'].replace('-', '').replace('_', '').isalnum()):
            raise ValueError('services require unique simple names')
        names.add(entry['name'])
        if not entry.get('ready_receipts'):
            raise ValueError('managed services require readiness evidence')
        if entry.get('stop_before', 'initialize') not in KINDS[1:]:
            raise ValueError('managed service stop_before must name a later stage')
    if not plan.get('input_sha256'):
        raise ValueError('a frozen input/source hash map is required')
    verify_hashes(plan['input_sha256'])
    return output, runtime


def runtime_environment(runtime, name, extra):
    env = dict(os.environ, **{k: str(v) for k, v in extra.items()})
    for key in CACHE_KEYS:
        path = runtime/name/key.lower()
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['ACTION_HEAD_LOCAL_RANK_CACHE_ROOT'] = str(runtime/name/'ranks')
    env['LOCAL_RUNTIME_ROOT'] = str(runtime/name)
    return env


@contextmanager
def temporary_environment(env):
    previous = os.environ.copy()
    os.environ.clear()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def assert_gpus_idle(ids):
    """Observe all requested physical cards; never kill someone else's process."""
    for gpu in ids:
        result = subprocess.run(['nvidia-smi', '-i', str(gpu),
            '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
            check=True, capture_output=True, text=True, timeout=15)
        if result.stdout.strip():
            raise RuntimeError(f'GPU {gpu} still has compute processes: {result.stdout.strip()}')


def stop_owned(process, identity):
    if process.poll() is None:
        if identity is None or not same_process_alive(identity):
            raise RuntimeError(f'owned process identity changed; refusing signal: {process.pid}')
        stop_process_group(process, timeout=20)
    if process.poll() is None:
        raise RuntimeError(f'owned process did not stop: {process.pid}')


class Pipeline:
    def __init__(self, plan, path):
        self.plan, self.path = plan, path
        self.output, self.runtime = validate_plan(plan, path)
        self.plan_sha256 = sha256(path)
        self.owned, self.results = [], []
        self.phase, self.status = 'preflight', 'starting'
        self.reservation = None

    def snapshot(self):
        save(self.output/'status.json', dict(schema='rule_teacher_pipeline_status_v1',
            status=self.status, phase=self.phase, updated_utc=utc(),
            supervisor=process_identity(os.getpid()), plan_sha256=self.plan_sha256, results=self.results,
            processes=[dict(name=e['name'], identity=identity, returncode=p.poll())
                       for e, p, identity in self.owned]))

    def start(self, entry):
        extra = dict(self.plan.get('environment', {}))
        command = entry.get('command')
        launch_proof = None
        if 'launch_plan' in entry:
            launch_path = Path(entry['launch_plan'])
            launch = json.loads(launch_path.read_text())
            previous = {item['path']: item['sha256'] for result in self.results for item in result['receipts']}
            if previous.get(str(launch_path.resolve())) != sha256(launch_path):
                raise ValueError('dynamic launch plan must be an immutable prior-stage receipt')
            launch_proof = dict(path=str(launch_path.resolve()), sha256=sha256(launch_path))
            # This artifact is produced by the already verified assemble stage.
            verify_hashes({launch['experiment_config']: launch['experiment_config_sha256']})
            command = launch[entry['argv_key']]
            extra.update(launch['environment'])
        if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
            raise ValueError('command must be an argv string list')
        extra.update(entry.get('environment', {}))
        env = runtime_environment(self.runtime, entry['name'], extra)
        log = self.output/'logs'/f"{entry['name']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        cwd = Path(entry.get('cwd', REPO)).resolve()
        if not cwd.is_relative_to(REPO):
            raise ValueError('stage cwd must be inside its frozen repository')
        with log.open('a') as stream:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        identity = process_identity(process.pid)
        self.owned.append((entry, process, identity))
        save(self.output/'processes'/f"{entry['name']}.json", dict(command=command,
            environment={k: env[k] for k in (*CACHE_KEYS, 'CUDA_VISIBLE_DEVICES') if k in env},
            identity=identity, log=str(log), launch_plan=launch_proof, started_utc=utc()))
        self.snapshot()
        return process

    def release_reservation(self, reason):
        if self.reservation is not None:
            self.reservation.stop(reason)
            self.reservation = None

    def reserve(self, stage):
        ids = stage.get('idle_gpu_ids', [])
        if not ids:
            return
        assert_gpus_idle(ids)
        config = dict(self.plan['gpu_reservation'], enabled=True, gpu_count=len(ids))
        folder = self.output/'reservations'/stage['kind']
        folder.mkdir(parents=True, exist_ok=False)
        self.reservation = WaitingGpuReservation({'waiting_gpu_reservation': config}, folder,
            lambda process: stop_owned(process, self.reservation_identity))
        env = runtime_environment(self.runtime, 'reservation-'+stage['kind'],
            dict(self.plan.get('environment', {}), CUDA_VISIBLE_DEVICES=','.join(map(str, ids))))
        # Existing helper owns readiness validation and graceful release. Record
        # identity as soon as its child is observable, including startup failure.
        original_check = self.reservation.check
        self.reservation_identity = None
        def check():
            if self.reservation.process is not None and self.reservation_identity is None:
                self.reservation_identity = process_identity(self.reservation.process.pid)
            original_check()
        self.reservation.check = check
        with temporary_environment(env):
            self.reservation.start()
        save(folder/'physical-gpus.json', dict(ids=ids, identity=self.reservation_identity))

    def wait_jobs(self, jobs, services):
        while True:
            if self.reservation is not None:
                self.reservation.check()
            for entry, process in services:
                if process.poll() is not None:
                    raise RuntimeError(f"managed service exited: {entry['name']}: {process.returncode}")
            for entry, process, start_time in jobs:
                code = process.poll()
                if code is not None and code != 0:
                    raise RuntimeError(f"stage command failed: {entry['name']}: {code}")
                if code is None and entry.get('timeout_seconds') and time.monotonic()-start_time > entry['timeout_seconds']:
                    raise TimeoutError(f"stage timeout: {entry['name']}")
            self.snapshot()
            if all(process.poll() is not None for _, process, _ in jobs):
                return
            time.sleep(min(5., float(self.plan.get('poll_seconds', 5.))))

    def run(self):
        error, service_jobs = None, []
        try:
            for entry in self.plan.get('services', []):
                process = self.start(entry)
                service_jobs.append((entry, process))
                deadline = time.monotonic()+entry.get('startup_timeout', 300)
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"service exited before ready: {entry['name']}")
                    try:
                        verify_receipts(entry['ready_receipts'])
                        break
                    except (OSError, ValueError, KeyError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"service readiness timeout: {entry['name']}")
                    self.snapshot()
                    time.sleep(1)
            for stage in self.plan['stages']:
                self.release_reservation('stage_transition')
                self.phase, self.status = stage['kind'], 'running'
                verify_hashes({str(self.path): self.plan_sha256, **self.plan['input_sha256']})
                for completed in self.results:
                    verify_hashes({item['path']: item['sha256'] for item in completed['receipts']})
                if stage.get('preconditions'):
                    verify_receipts(stage['preconditions'])
                stopping = {entry['name'] for entry in self.plan.get('services', [])
                    if entry.get('stop_before', 'initialize') == stage['kind']
                    or stage['kind'] in ('initialize', 'train')}
                for entry, process, identity in reversed(self.owned):
                    if entry['name'] in stopping:
                        stop_owned(process, identity)
                service_jobs = [(entry, process) for entry, process in service_jobs
                                if entry['name'] not in stopping]
                if stage['kind'] in ('initialize', 'train'):
                    # Mandatory observation on all 16 cards, even if no managed
                    # services were declared. External occupancy is a blocker.
                    assert_gpus_idle(list(range(16)))
                self.reserve(stage)
                started = utc()
                jobs = [(entry, self.start(entry), time.monotonic()) for entry in stage['commands']]
                self.wait_jobs(jobs, service_jobs)
                proofs = verify_receipts(stage['required_receipts'])
                result = dict(kind=stage['kind'], status='complete', started_utc=started,
                    finished_utc=utc(), results={entry['name']: p.returncode for entry, p, _ in jobs},
                    receipts=proofs)
                self.results.append(result)
                save(self.output/'stages'/f"{stage['kind']}.json", result)
            self.status = 'complete'
        except BaseException:
            self.status = 'blocked'
            error = traceback.format_exc()
        finally:
            cleanup_errors = []
            try:
                self.release_reservation('pipeline_exit')
            except BaseException:
                cleanup_errors.append(traceback.format_exc())
            for _, process, identity in reversed(self.owned):
                try:
                    stop_owned(process, identity)
                except BaseException:
                    cleanup_errors.append(traceback.format_exc())
            if cleanup_errors:
                self.status = 'blocked'
            else:
                shutil.rmtree(self.runtime, ignore_errors=True)
            self.snapshot()
            completion = dict(status=self.status, phase=self.phase, results=self.results,
                error=error, cleanup_errors=cleanup_errors, finished_utc=utc(),
                runtime_cleaned=not self.runtime.exists(),
                training_started=any(entry.get('pipeline_kind') == 'train' for entry, _, _ in self.owned))
            save(self.output/'completion.json', completion)
        return 0 if self.status == 'complete' else 1


def execute(path):
    plan = json.loads(path.read_text())
    output, _ = validate_plan(plan, path)
    with (output/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output/'completion.json').exists() or (output/'status.json').exists():
            raise ValueError('attempt already started; inspect receipts, never replay automatically')
        runtime = Path(plan['runtime']).resolve()
        token = os.environ.get('RULE_TEACHER_PIPELINE_LAUNCH_TOKEN')
        if runtime.exists():
            owner = read_status_json(output/'runtime-owner.json')
            if not token or owner != dict(token=token, plan_sha256=sha256(path)):
                raise ValueError('existing runtime does not belong to this launch')
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'pipeline received {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        for stage in plan['stages']:
            for command in stage['commands']:
                command['pipeline_kind'] = stage['kind']
        return Pipeline(plan, path).run()


def launch(path):
    plan = json.loads(path.read_text())
    output, runtime = validate_plan(plan, path)
    with (output/'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((output/name).exists() for name in ('launch.json', 'completion.json', 'status.json')):
            raise ValueError('refusing duplicate or implicit recovery launch')
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--plan', str(path.resolve()), '--run']
        if runtime.exists():
            raise ValueError('refusing pre-existing runtime')
        token = uuid.uuid4().hex
        save(output/'runtime-owner.json', dict(token=token, plan_sha256=sha256(path)))
        env = runtime_environment(runtime, 'pipeline', plan.get('environment', {}))
        env['RULE_TEACHER_PIPELINE_LAUNCH_TOKEN'] = token
        with (output/'pipeline.log').open('a') as stream:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = dict(command=command, identity=process_identity(process.pid),
            plan_sha256=sha256(path), started_utc=utc())
        save(output/'launch.json', receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--launch', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--status', action='store_true')
    args = parser.parse_args()
    if args.status:
        result = read_status_json(args.plan.parent/'status.json')
        identity = result.get('supervisor')
        result['supervisor_alive'] = bool(identity and same_process_alive(identity))
        print(json.dumps(result, ensure_ascii=False))
    elif args.launch:
        print(json.dumps(launch(args.plan), ensure_ascii=False))
    else:
        raise SystemExit(execute(args.plan))


if __name__ == '__main__':
    main()
