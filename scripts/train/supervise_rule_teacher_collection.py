"""Run frozen local collection commands with durable receipts and owned cleanup.

The caller prepares an explicit plan from the original launch receipts. Dataset,
teacher, rendering and resume behavior remain in their existing entrypoints.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import runpy
import signal
import socket
import subprocess
import sys
import time
import traceback
from urllib.request import ProxyHandler, build_opener

from prepare_rule_teacher_three_source import ART, REPO, save, sha256, verify_plan
from watch_v6_rl_local_credit import process_identity
# Load the existing service helper by path: vln_demo.py shadows the package.
proc_info = runpy.run_path(str(REPO / 'vln_demo/dlc/supervise.py'),
                          run_name='_rule_teacher_service_identity')['proc_info']
routing_shm = runpy.run_path(str(REPO / 'verl/action_head/serve/routing_shm.py'),
                            run_name='_owned_routing_shm')


CACHE_KEYS = (
    'TMPDIR', 'XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'HF_HOME', 'TORCH_HOME',
    'TRITON_CACHE_DIR', 'TRITON_HOME', 'TRITON_DUMP_DIR', 'TRITON_OVERRIDE_DIR',
    'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH', 'PYTORCH_KERNEL_CACHE_PATH',
    'TORCH_EXTENSIONS_DIR', 'MPLCONFIGDIR', 'NUMBA_CACHE_DIR', 'DG_CACHE_DIR',
    'DG_JIT_CACHE_DIR', 'RTC_CACHE_PATH', 'PPU_LIB_RTC_CACHE_PATH', 'HGRTC_CACHE_PATH',
    'FLASHINFER_WORKSPACE_BASE', 'FLASHINFER_CUBIN_DIR', 'VLLM_CACHE_ROOT',
    'VLLM_ASSETS_CACHE', 'VLLM_CONFIG_ROOT', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE',
    'HF_MODULES_CACHE', 'HF_DATASETS_CACHE', 'TRANSFORMERS_CACHE', 'PIP_CACHE_DIR',
)

COMPILE_CACHE_KEYS = (
    'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH',
    'PYTORCH_KERNEL_CACHE_PATH', 'TORCH_EXTENSIONS_DIR', 'DG_CACHE_DIR',
    'DG_JIT_CACHE_DIR', 'RTC_CACHE_PATH', 'PPU_LIB_RTC_CACHE_PATH',
    'HGRTC_CACHE_PATH', 'VLLM_CACHE_ROOT',
)


def reusable_compile_environment(entry, runtime, allowed_runtime):
    """Keep compiler caches across stages, with exclusive per-replica ownership."""
    value = entry.get('reusable_compile_cache')
    if value is None:
        return {}, None
    cache = Path(value).resolve()
    base = (allowed_runtime / 'compile-cache').resolve()
    if (not cache.is_relative_to(base) or len(cache.relative_to(base).parts) != 2
            or not re.fullmatch(r'[0-9a-f]{64}', cache.parent.name)
            or not re.fullmatch(r'brain_[0-9]+', cache.name)
            or cache.is_relative_to(runtime) or runtime.is_relative_to(cache)):
        raise ValueError('reusable compiler cache must be a version-bound local sibling of runtime')
    cache.mkdir(parents=True, exist_ok=True)
    lock = (cache / 'owner.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = {}
        for key in COMPILE_CACHE_KEYS:
            path = cache / key.lower()
            path.mkdir(exist_ok=True)
            result[key] = str(path)
        result['ACTION_HEAD_COMPILE_CACHE_ROOT'] = str(cache)
        return result, lock
    except BaseException:
        lock.close()
        raise


class CollectorFailed(RuntimeError):
    """A collector's failure requires stopping this supervisor's owned jobs."""



def owned_live_process_groups(groups):
    """Reuse the service supervisor's stat reader; exited leaders may have workers."""
    live = set()
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            info = proc_info(int(path.name))
            if info and info['state'] != 'Z' and info['pgid'] in groups:
                live.add(info['pgid'])
    return live


@contextmanager
def uninterrupted_cleanup():
    """A second termination request must not abandon owned CUDA descendants."""
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        for sig in previous: signal.signal(sig,signal.SIG_IGN)
        yield
    finally:
        for sig,handler in previous.items(): signal.signal(sig,handler)


def stop_owned_process_groups(owned, timeout=20, *, kill_timeout=60):
    groups = {process.pid for _, process in owned}
    if os.getpgrp() in groups or any(group <= 1 for group in groups):
        raise ValueError('owned cleanup requires dedicated child process groups')
    active = owned_live_process_groups(groups)
    active.update(process.pid for _, process in owned if process.poll() is None)
    for group in active:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    for _, process in reversed(owned):
        try:
            process.wait(timeout=max(.1, deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    active = owned_live_process_groups(groups)
    while active and time.monotonic() < deadline:
        time.sleep(.1)
        active = owned_live_process_groups(groups)
    for group in active:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for _, process in reversed(owned):
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
    # Accelerator workers can remain alive while the driver releases CUDA
    # state after SIGKILL. Three seconds was shorter than real PPU teardown:
    # clean collectors were marked failed and their log archives abandoned.
    # Observe actual owned groups for a bounded grace period, never declare
    # them gone merely because the parent returned or SIGKILL was sent.
    deadline = time.monotonic() + kill_timeout
    next_notice = 0.
    active = owned_live_process_groups(groups)
    while active and time.monotonic() < deadline:
        if time.monotonic() >= next_notice:
            print(json.dumps(dict(event='waiting_owned_process_group_exit',
                time=time.time(),pid=os.getpid(),groups=sorted(active),
                remaining_seconds=max(0.,deadline-time.monotonic()),
                meaning='cleanup only; no evaluation progress')),flush=True)
            next_notice = time.monotonic()+20
        time.sleep(.1)
        active = owned_live_process_groups(groups)
    return sorted(active)

def reconcile_completed_cleanup(launch, recovery, *, release_timeout=60):
    """Record late resource release and recover logs without rewriting a failure.

    The original supervisor must be gone and every owned process group empty.
    This receipt concerns resource ownership only, never evaluation success.
    """
    import align_three_source_observations as io
    from watch_v6_rl_local_credit import same_process_alive
    launch, recovery = Path(launch), Path(recovery)
    if recovery.resolve().is_relative_to(launch.resolve()):
        raise ValueError('cleanup recovery must preserve the original attempt')
    completion = io.load(launch/'completion.json')
    owner = io.load(launch/'supervisor-launch.json')
    state = io.load(launch/'status.json')
    plan = io.load(launch/'plan.json')
    if same_process_alive(owner):
        raise RuntimeError('original service supervisor is still alive')
    groups = {p['pid'] for p in state['processes']}
    remaining = owned_live_process_groups(groups)
    deadline=time.monotonic()+release_timeout
    while remaining and time.monotonic()<deadline:
        time.sleep(.2)
        remaining=owned_live_process_groups(groups)
    if remaining:
        raise RuntimeError('original owned service groups are still alive: '+str(sorted(remaining)))
    if Path(plan['runtime']).exists() or any(Path(p).exists() for p in completion.get('native_workspace_roots',[])):
        raise RuntimeError('original runtime or native workspaces still exist')
    recovery.mkdir(parents=True,exist_ok=True)
    if (recovery/'cleanup-release.json').exists():
        raise ValueError('cleanup release already recorded; inspect instead of rewriting')
    records=[]
    for record in completion.get('log_archives',[]):
        archive=Path(record['archive'])
        before=io.read_bytes(archive) if io.exists(archive) else b''
        local=Path(record['local_spool'])
        if record.get('local_spool_retained'):
            if not local.is_relative_to('/tmp') or not archive.is_relative_to(launch):
                raise ValueError('retained log paths escaped this service attempt')
            if io.exists(local):
                data=io.read_bytes(local)
                reclaimed=int(record.get('reclaimed_bytes',0))
                if not 0<=reclaimed<=len(before)<=len(data) or data[reclaimed:len(before)]!=before[reclaimed:]:
                    raise ValueError('retained log prefix differs from its archived bytes')
                recovered=before+data[len(before):]
            elif record.get('pending_bytes')==0 and len(before)>=record['bytes']:
                recovered=before
            else:
                raise ValueError('unarchived local log bytes are unavailable')
            target=recovery/'logs'/archive.name
            target.parent.mkdir(parents=True,exist_ok=True)
            io.write_bytes(target,recovered)
            records.append(dict(source_archive=str(archive),local_spool=str(local),
                recovered_archive=str(target),bytes=len(recovered),original_preserved=True))
    result=dict(schema='owned_service_cleanup_release_v1',hostname=socket.gethostname(),
        source_completion=str(launch/'completion.json'),supervisor=owner,
        checked_groups=sorted(groups),owned_process_groups_cleaned=True,
        runtime_cleaned=True,native_workspaces_cleaned=True,logs=records,
        original_status=completion.get('status'),evaluation_success_claimed=False,time=time.time())
    io.write_json(recovery/'cleanup-release.json',result)
    return result


def finish_log_spools(spools, *, timeout=120.):
    """Bound total drain time; unarchived bytes remain outside runtime cleanup."""
    deadline = time.monotonic()+timeout
    # Publish EOF to every archive before any shared-deadline wait. Sequential
    # close() otherwise spends one polling interval per idle service and marks
    # the remaining, already archived logs as failed when the budget expires.
    before = [(name, spool, spool.status()) for name, spool in spools]
    for _, spool, _ in before:
        spool.finish()
    result = []
    for name, spool, state in before:
        drained = spool.close(max(0., deadline-time.monotonic()))
        final = dict(state, archived_bytes=state['bytes'], pending_bytes=0, error=None) if drained else spool.status()
        result.append(dict(name=name, **final, drained=drained, local_spool_retained=not drained))
    return result


def collection_runtime_root():
    """Use a container-local root for DLC, retaining the existing local default."""
    override = os.environ.get('RULE_COLLECTION_RUNTIME_ROOT')
    if not override:
        # Executable source snapshots live on the artifact disk. The local
        # writable cache remains in the user's workspace, independent of code.
        return Path('/tmp/closed-loop-collection').resolve()
    supplied = Path(override)
    root = supplied.resolve()
    if os.environ.get('GPU_DIRECT_RUN') == '1' and os.environ.get('GPU_CACHE_ROOT'):
        cache = Path(os.environ['GPU_CACHE_ROOT']).resolve()
        if supplied.is_absolute() and root != cache and root.is_relative_to(cache):
            return root
    if not supplied.is_absolute() or not any(root.is_relative_to(base) and root != Path(base)
                                           for base in ('/tmp', '/dev/shm')):
        raise ValueError('RULE_COLLECTION_RUNTIME_ROOT must be a dedicated container-local /tmp or /dev/shm directory')
    return root


def read_status_json(path):
    """Read small receipts only; never load requests, images or rollout bodies."""
    try:
        with path.open('rb') as stream:
            body = stream.read(2 * 1024 * 1024 + 1)
        if len(body) > 2 * 1024 * 1024:
            return {}
        value = json.loads(body)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, RecursionError):
        return {}


def status_text(value, limit=160):
    return value[:limit] if isinstance(value, str) else None


def status_integer(value):
    return value if type(value) is int and abs(value) < 2 ** 63 else None


def status_entries(value, label, warnings):
    if not isinstance(value, list):
        warnings.append(label + ' is not a list')
        return []
    entries = [entry for entry in value if isinstance(entry, dict)]
    if len(entries) != len(value):
        warnings.append(label + ' contains non-object entries')
    return entries


def file_age(path, now):
    try:
        return round(max(0, now - path.stat().st_mtime), 1)
    except OSError:
        return None


def inspect_process(pid, command, *, proc_root=Path('/proc')):
    """Check command identity as well as PID existence (old receipts lack ticks)."""
    pid = pid if type(pid) is int and 0 < pid < 2 ** 31 else None
    try:
        identity = process_identity(int(pid), proc_root=proc_root)
        actual = (proc_root / str(pid) / 'cmdline').read_bytes().split(b'\0')
        actual = [part.decode(errors='replace') for part in actual if part]
    except (OSError, TypeError, ValueError):
        identity = None
        actual = []
    matches = bool(command and actual == command)
    return dict(pid=pid, live=bool(identity and identity['state'] not in ('Z', 'X')),
                command_matches=matches,
                verified_live=bool(identity and identity['state'] not in ('Z', 'X') and matches),
                state=identity['state'] if identity else None,
                start_ticks=identity['start_ticks'] if identity else None)


def collector_output(entry):
    command = entry.get('command')
    if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
        raise ValueError('collector command unavailable')

    def option(name, default=None):
        return command[command.index(name) + 1] if name in command else default

    if '--output' in command:
        return Path(option('--output'))
    if '--root' in command and '--backend' in command:
        return Path(option('--root')) / 'collections' / option('--backend') / f"shard_{int(option('--shard', '0')):03d}"
    return None


def collector_serving_failure(entry, attempt_root):
    """Inspect only this attempt's planned collector, never older run receipts."""
    output = collector_output(entry)
    if output is None or not output.resolve().is_relative_to(attempt_root.resolve()):
        return None
    for path in sorted(output.glob('case_*/candidates/decision_*/serving_error.json')):
        if not path.resolve().is_relative_to(output.resolve()):
            continue
        audit = read_status_json(path)
        summary = {key: status_text(audit.get(key), 160) for key in
            ('phase', 'failure_kind', 'error_type', 'finish_reason', 'request_sha256')}
        summary.update({key: status_integer(audit.get(key)) for key in
                        ('http_status', 'emitted_text_chars')})
        usage = audit.get('usage')
        if isinstance(usage, dict):
            summary['usage'] = {key:status_integer(usage.get(key)) for key in
                               ('prompt_tokens', 'completion_tokens', 'total_tokens')}
        return dict(collector=entry['name'], receipt_path=str(path),
                    receipt_sha256=sha256(path), audit_summary=summary)
    return None


def collector_progress(entry, now):
    output = collector_output(entry)
    if output is None:
        raise ValueError('collector output unavailable')
    summary_path = output / 'summary.json'
    summary = read_status_json(summary_path)
    cases = sorted(path for path in output.glob('case_*') if path.is_dir())
    unfinished = [path for path in cases if not (path / 'result.json').exists()]
    case = unfinished[-1] if unfinished else cases[-1] if cases else None
    recent = []
    decisions = []
    if case:
        decisions = sorted(case.glob('decision_*.json'))
        recent = decisions[-1:] + [case / name for name in ('case.json', 'result.json', 'rollout.json')]
        # Only inspect the newest decision/option directory, never image or JSON bodies.
        for folder in ('candidates', 'options'):
            children = sorted(path for path in (case / folder).glob('*') if path.is_dir())
            if children:
                recent.extend(path for path in children[-1].iterdir() if path.is_file())
    newest = max(((path, file_age(path, now)) for path in recent if path.exists()),
                 key=lambda item: -(item[1] if item[1] is not None else float('inf')), default=(None, None))
    return dict(name=status_text(entry.get('name'), 80), output=str(output)[:1024],
                completed_result_files=len(cases) - len(unfinished),
                summary_completed=status_integer(summary.get('completed')), summary_age_seconds=file_age(summary_path, now),
                unfinished_cases=len(unfinished), active_case=case.name if unfinished else None,
                latest_decision=decisions[-1].name if decisions else None,
                latest_decision_age_seconds=file_age(decisions[-1], now) if decisions else None,
                latest_artifact=str(newest[0].relative_to(output))[:256] if newest[0] else None,
                latest_artifact_age_seconds=newest[1])


def collection_status(plan_path, *, now=None, proc_root=Path('/proc')):
    """A read-only observation; a live supervisor is not evidence of data progress."""
    now = time.time() if now is None else now
    plan = read_status_json(plan_path)
    output = plan_path.resolve().parent
    status = read_status_json(output / 'status.json')
    launch = read_status_json(output / 'supervisor-launch.json')
    completion = read_status_json(output / 'completion.json')
    warnings = []
    services = status_entries(plan.get('services', []), 'plan.services', warnings)
    planned_collectors = status_entries(plan.get('collectors', []), 'plan.collectors', warnings)
    entries = {entry['name']: entry for entry in services + planned_collectors if isinstance(entry.get('name'), str)}
    recorded = status_entries(status.get('processes', []), 'status.processes', warnings)
    processes = []
    for row in recorded[:16]:
        name = status_text(row.get('name'), 80)
        expected = entries.get(row['name'], {}) if isinstance(row.get('name'), str) else {}
        processes.append(dict(name=name,
            recorded_returncode=status_integer(row.get('returncode')),
            **inspect_process(row.get('pid'), expected.get('command'), proc_root=proc_root)))
    collectors = []
    for entry in planned_collectors[:8]:
        try:
            collectors.append(collector_progress(entry, now))
        except (OSError, ValueError, TypeError, IndexError):
            collectors.append(dict(name=status_text(entry.get('name'), 80), error='progress unavailable'))
    age = file_age(output / 'status.json', now)
    return dict(schema='rule_teacher_collection_observation_v1',
        observed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(now)),
        recorded_status=status_text(status.get('status')), heartbeat_age_seconds=age,
        heartbeat_stale=age is None or age > 30,
        supervisor=inspect_process(status.get('supervisor_pid', launch.get('pid')), launch.get('argv'), proc_root=proc_root),
        completion_status=status_text(completion.get('status')), processes=processes, collectors=collectors,
        receipt_warnings=warnings,
        omitted_processes=max(0, len(recorded) - 16), omitted_collectors=max(0, len(planned_collectors) - 8),
        note='Read-only. PID identity uses current command line; legacy receipts lack original start ticks. '
             'Process/heartbeat liveness is not data progress; compare artifact/decision ages across snapshots. '
             'Completed cases are not reviewed training tasks. Missing/malformed receipts report unknown.')


def format_status(report):
    """Keep even unexpected large plans below the 8 KiB console output budget."""
    report = {key: value[:512] if isinstance(value, str) else value for key, value in report.items()}
    for field in ('processes', 'collectors'):
        report[field] = list(report.get(field) or [])
        report['omitted_' + field] = status_integer(report.get('omitted_' + field)) or 0
    while True:
        body = json.dumps(report, ensure_ascii=False, separators=(',', ':'))
        if len(body.encode()) + 1 <= 8192:
            return body
        for field in ('processes', 'collectors'):
            if report[field]:
                report[field].pop()
                report['omitted_' + field] += 1
                break
        else:
            return json.dumps(dict(schema='rule_teacher_collection_observation_v1',
                recorded_status='unknown', error='status metadata exceeds output budget'))


def prepare_native_gym_workspace(entry, runtime, frozen_inputs):
    """Reuse the native Gym launcher's private YAML + read-only data link.

    Only the workspace is created here; shared NAV/data inputs are never
    copied or modified. Returning cwd preserves the original argv and PID.
    """
    workspace = entry.get('native_gym_workspace')
    if workspace is None:
        return REPO
    name = entry['name']
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('native Gym actor name must be a single directory name')
    runtime = Path(runtime).resolve()
    cwd = Path(workspace['cwd'])
    workspace_root = (REPO / '.runtime/native-gym' / runtime.name
                      if workspace.get('inside_checkout') else runtime)
    expected = workspace_root / name / 'work'
    if (not cwd.is_absolute() or cwd != expected or cwd.resolve() != expected
            or not cwd.resolve().is_relative_to(workspace_root)):
        raise ValueError('native Gym cwd must be its private runtime/actor/work')
    config = Path(workspace['config_source'])
    expected_sha = workspace['config_sha256']
    if (not config.is_absolute() or frozen_inputs.get(str(config)) != expected_sha
            or sha256(config) != expected_sha):
        raise ValueError('native Gym YAML is not frozen or changed')
    data = Path(workspace['data_target'])
    if not data.is_absolute() or not data.is_dir() or data.resolve() != data:
        raise ValueError('native Gym data target must be an existing resolved directory')
    # A fresh directory also rejects stale files and symlinks before writing.
    cwd.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config, cwd / 'vln_demo.yaml')
    if sha256(cwd / 'vln_demo.yaml') != expected_sha:
        raise ValueError('native Gym YAML changed during workspace copy')
    (cwd / 'data').symlink_to(data, target_is_directory=True)
    return cwd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--status', action='store_true', help='read-only bounded process and data progress snapshot')
    args = parser.parse_args()
    suite_plan = os.environ.get('ACTION_HEAD_SUITE_PLAN')
    if suite_plan and not args.status:
        sys.path.insert(0, str(REPO / 'scripts/eval'))
        from action_head_suite_runtime import install_control_io
        install_control_io(Path(suite_plan).parent, sys.modules[__name__])
    if args.status:
        print(format_status(collection_status(args.plan)))
        return
    plan = json.loads(args.plan.read_text())
    output = args.plan.resolve().parent
    runtime = Path(plan['runtime']).resolve()
    allowed_runtime = collection_runtime_root()
    if not output.is_relative_to(ART) or not runtime.is_relative_to(allowed_runtime) or runtime == allowed_runtime:
        raise ValueError('artifact output or local runtime outside authorized experiment roots')
    lock = (output / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / 'completion.json').exists():
        raise ValueError('completed attempt is immutable; prepare a new launch plan')
    source_plan = json.loads(Path(plan['source_plan']).read_text())
    verify_plan(source_plan)
    for entry in plan['services']:
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', entry['port'])) == 0:
                raise RuntimeError(f"refusing to replace existing service on {entry['port']}")
    opener = build_opener(ProxyHandler({}))
    owned = []
    compile_cache_locks = []
    service_log_spools = []
    native_workspaces = set()
    routing_namespaces = []
    routing_cleanup = None
    log_archives = None
    results = {}
    status = 'starting'
    error = None
    serving_failure = None
    independent = plan.get('independent_collectors', False)

    def snapshot():
        save(output / 'status.json', dict(status=status, updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            supervisor_pid=os.getpid(), results=results,
            processes=[dict(name=e['name'], pid=p.pid, returncode=p.poll(), log=e['log']) for e,p in owned],
            routing_shm_namespaces=routing_namespaces,
            log_archives=(log_archives if log_archives is not None else
                [dict(name=name, **spool.status()) for name, spool in service_log_spools])))

    def start(entry):
        environment = os.environ.copy()
        for key in plan.get('unset_environment', []):
            environment.pop(key, None)
        environment.update(plan['environment'])
        environment.update(entry.get('environment', {}))
        if entry['name'].startswith('brain_'):
            namespace = routing_shm['new_namespace'](entry['name'])
            environment[routing_shm['NAMESPACE_ENV']] = namespace
            routing_namespaces.append(namespace)
        # Reuse run_v6_vlnce_dlc.sh and the installed-runtime cache bindings;
        # override inherited aliases before any service imports their consumers.
        for key in CACHE_KEYS:
            cache = runtime / entry['name'] / key.lower()
            cache.mkdir(parents=True, exist_ok=True)
            environment[key] = str(cache)
        reusable, cache_lock = reusable_compile_environment(entry, runtime, allowed_runtime)
        if cache_lock is not None:
            compile_cache_locks.append(cache_lock)
            environment.update(reusable)
            print(json.dumps(dict(event='reusable_compile_cache_bound',
                name=entry['name'], path=entry['reusable_compile_cache'],
                pid=os.getpid(), hostname=socket.gethostname())), flush=True)
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        cwd = prepare_native_gym_workspace(entry, runtime, source_plan.get('input_sha256', {}))
        if entry.get('native_gym_workspace', {}).get('inside_checkout'):
            native_workspaces.add(cwd.parent.parent)
        logfile = Path(entry['log']).resolve()
        if not logfile.is_relative_to(ART):
            raise ValueError('log must be on artifact disk')
        logfile.parent.mkdir(parents=True, exist_ok=True)
        if plan.get('service_log_spool'):
            # Same LogSpool as production RL. GPU services never wait on CPFS
            # logging; the archive owns the sole writer and retains every byte.
            from service_log_spool import LogSpool
            spool = LogSpool(logfile, allowed_runtime/'service-log-spool'/runtime.name)
            service_log_spools.append((entry['name'], spool))
            process = subprocess.Popen(entry['command'], cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                stdout=spool.handle, stderr=subprocess.STDOUT, start_new_session=True)
        else:
            with logfile.open('a') as stream:
                process = subprocess.Popen(entry['command'], cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        owned.append((entry, process))
        snapshot()
        return process

    def check_log_capacity():
        for name, spool in service_log_spools:
            if spool.status()['capacity_exceeded']:
                raise RuntimeError('local log spool capacity exceeded: '+name)

    def wait_ready(entry, process):
        deadline = time.monotonic() + entry.get("startup_timeout", 180)
        while time.monotonic() < deadline:
            check_log_capacity()
            if process.poll() is not None:
                raise RuntimeError(f"{entry['name']} exited during startup: {process.returncode}")
            try:
                with opener.open(entry['health_url'], timeout=3) as response:
                    if response.status == 200:
                        return
            except OSError:
                pass
            snapshot()
            time.sleep(1)
        raise RuntimeError(f"{entry['name']} health check timed out")

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'supervisor received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        # Consecutive entries in one explicit group may start together.
        # Separate renderers from their dependent Gym actors.
        from itertools import groupby
        for _, group in groupby(plan['services'], key=lambda e: e.get('startup_group', e['name'])):
            group = list(group)
            width = max(1, min(16, int(plan.get('service_start_batch_size', 1))))
            for offset in range(0, len(group), width):
                pending = [(entry, start(entry)) for entry in group[offset:offset + width]]
                for entry, process in pending:
                    wait_ready(entry, process)
        jobs = [(entry, start(entry)) for entry in plan['collectors']]
        status = 'collecting'
        while True:
            check_log_capacity()
            running = False
            for entry,process in jobs:
                # V8 uses this file only as a launcher/waiter. Never scan case
                # receipts or terminate peers in response to a child failure.
                serving_failure = None if independent else collector_serving_failure(entry, output)
                if serving_failure is not None:
                    save(output / 'collector-serving-error.json', serving_failure)
                    raise CollectorFailed(f"collector {entry['name']} serving failure: {serving_failure['receipt_path']}")
                code = process.poll()
                if code is None:
                    running = True
                else:
                    results[entry['name']] = code
                    if code != 0 and not independent:
                        raise CollectorFailed(f"collector {entry['name']} exited: {code}")
            if not running:
                break
            for entry,process in owned[:len(plan['services'])]:
                if process.poll() is not None and not independent:
                    raise RuntimeError(f"owned service {entry['name']} exited: {process.returncode}")
            snapshot()
            time.sleep(5)
        results.update({entry['name']:process.returncode for entry,process in jobs})
        status = 'collector_error' if any(results.values()) else 'complete'
    except BaseException as exc:
        status = ('interrupted' if isinstance(exc, KeyboardInterrupt) else
                  'collector_error' if isinstance(exc, CollectorFailed) else 'error')
        error = traceback.format_exc()
    finally:
        with uninterrupted_cleanup():
            remaining_groups = stop_owned_process_groups(owned)
            if remaining_groups:
                status = 'cleanup_error'
                error = (error or '') + '\nowned process groups survived cleanup: ' + str(remaining_groups)
            elif routing_namespaces:
                try:
                    routing_cleanup = routing_shm['cleanup_namespaces'](routing_namespaces)
                    if routing_cleanup['referenced']:
                        status = 'cleanup_error'
                        error = (error or '') + '\nowned routing buffers still have live readers'
                except Exception:
                    status = 'cleanup_error'
                    error = (error or '') + '\nrouting buffer cleanup failed:\n' + traceback.format_exc()
            log_archives = ([dict(name=name, **spool.status(), drained=False, local_spool_retained=True)
                for name, spool in service_log_spools] if remaining_groups else
                finish_log_spools(service_log_spools, timeout=120.))
            if any(not row['drained'] for row in log_archives):
                if status == 'complete':
                    status = 'log_archive_error'
                error = (error or '') + '\nunarchived service log bytes retained in container-local spool'
            if not remaining_groups:
                for workspace in native_workspaces:
                    shutil.rmtree(workspace, ignore_errors=True)
            native_workspaces_cleaned = all(not workspace.exists() for workspace in native_workspaces)
            if not native_workspaces_cleaned:
                status = 'cleanup_error'
                error = (error or '') + '\nnative Gym workspaces retained in container checkout'
            shutil.rmtree(runtime, ignore_errors=True)
            # Children have exited before releasing compiler-cache ownership.
            for cache_lock in compile_cache_locks:
                cache_lock.close()
            snapshot()
            save(output / 'completion.json', dict(status=status, results=results, error=error,
                serving_failure=serving_failure, log_archives=log_archives,
                native_workspaces_cleaned=native_workspaces_cleaned,
                native_workspace_roots=sorted(map(str, native_workspaces)),
                runtime_cleaned=not runtime.exists(), owned_process_groups_cleaned=not remaining_groups,
                remaining_process_groups=remaining_groups, formal_training_started=False,
                routing_shm_cleanup=routing_cleanup,
                finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    if status != 'complete':
        raise SystemExit(error or str(results))


if __name__ == '__main__':
    main()
