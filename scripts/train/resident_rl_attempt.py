"""Replace one RL business attempt inside an existing, identity-bound DLC Pod.

`prepare` records a plan only; `status` is read-only. `stop-old` signals the
verified worker PID once, never its process group (the original group is 1).
`start-new` detaches one owner after all four shutdown proofs exist. No command
creates/stops a DLC job, edits a frozen launch, or replays an uncertain launch.

This composes the existing bounded control I/O, process identity helpers,
worker cleanup/postrun reservation, and resident active/idle controller. The
collection-specific owner cannot consume an RL launch plan, hence this adapter.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import align_three_source_observations as io
from watch_v6_rl_local_credit import process_identity, same_process_alive

ART = io.ART
SCHEMA = 'resident-rl-attempt-v1'
SELF = Path(__file__).resolve()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read_optional(path):
    try:
        return io.load(path)
    except FileNotFoundError:
        return None


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=io.digest(io.read_bytes(path)))


def check_binding(record):
    require(binding(record['path']) == record, 'bound file changed: ' + record['path'])
    return Path(record['path'])


def proc(pid):
    identity = process_identity(pid)
    if not identity or identity['state'] in ('Z', 'X'):
        return None
    try:
        directory = Path('/proc') / str(pid)
        fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[19] != identity['start_ticks']:
            return None
        argv = (directory / 'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        return dict(identity, ppid=int(fields[1]), pgid=int(fields[2]),
                    argv_sha256=hashlib.sha256(b'\0'.join(argv)).hexdigest(),
                    argv=[s.decode() for s in argv])
    except FileNotFoundError:
        return None


def exact_live(expected, *, argv=True):
    actual = proc(expected['pid'])
    return bool(actual and actual['start_ticks'] == str(expected['start_ticks'])
                and (not argv or not expected.get('argv_sha256')
                     or actual['argv_sha256'] == expected['argv_sha256']))


def signal_pid(expected, protected):
    """A verified individual PID only. Original RL workers share PGID 1."""
    require(expected['pid'] not in protected and expected['pid'] > 1,
            'refusing to signal a protected process')
    require(exact_live(expected), 'process identity/argv changed before signal')
    require(hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'),
            'safe recovery requires Linux pidfd signals')
    # Reuse the handoff_v6_rl_local pidfd pattern: retain the kernel process
    # reference, then recheck identity. PID recycling after this point cannot
    # redirect SIGTERM to a different process.
    pidfd = os.pidfd_open(expected['pid'])
    try:
        require(exact_live(expected), 'process identity/argv changed after pidfd open')
        signal.pidfd_send_signal(pidfd, signal.SIGTERM)
    finally:
        os.close(pidfd)


def root_path(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_relative_to(ART)
            and path != ART, 'attempt must be a real directory on the artifact disk')
    return path


def runtime(root):
    return Path('/tmp/v6rd') / hashlib.sha256(str(root).encode()).hexdigest()[:12]


def directory(config):
    return Path(config['new_root']) / 'recovery' / f"node_{config['rank']:02d}"


def debug_root(config):
    return Path(config['resident_root']) / 'resident/nodes' / config['hostname'] / 'node-debug'


def validate(config, *, files=True):
    require(config.get('schema') == SCHEMA and config.get('node_count') == 4,
            'unsupported recovery schema/node count')
    require(config['hostname'] == os.uname().nodename and 0 <= config['rank'] < 4,
            'wrong Pod/rank for this recovery')
    old, new, resident = (root_path(config[k]) for k in ('old_root', 'new_root', 'resident_root'))
    require(old != new and not new.is_relative_to(old) and not old.is_relative_to(new),
            'attempt output roots overlap')
    require(config['job_id'] and config['pod_uid'] and config['hostname'].startswith(config['job_id'] + '-'),
            'missing official job/Pod UID binding')
    require(exact_live(config['outer_shell']), 'original resident shell identity changed')
    require(config['outer_shell']['pid'] != config['old_worker']['pid'],
            'worker must differ from platform shell')
    if files:
        for record in config['files'].values():
            check_binding(record)
        plan = io.load(new / 'launch/submit_plan.json')
        require(plan['output_root'] == str(new), 'new plan does not own its isolated output')
        if config.get('git_task'):
            spec = io.load(check_binding(config['files']['git_task']))
            checkout = Path(spec['checkout_root']) / 'repo'
            require(spec['output_root'] == str(new) and checkout.is_absolute()
                    and not checkout.is_relative_to('/data/storage') and not checkout.is_relative_to('/data/storage'),
                    'Git task must own its container-local checkout and output')
            commit = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
            require(commit == spec['commit'], 'prepared local Git commit differs from the task')
        else:
            require(plan['snapshot_repo'] == str(new / 'launch/source'),
                    'legacy plan does not own its isolated source')
        require(runtime(old) != runtime(new), 'runtime namespace collision')
    return old, new, resident


def prepare(old_root, new_root, rank, job_id, pod_uid, config_path, resident_root=None):
    """Run in the target Pod through official exec; no process is changed."""
    old, new = root_path(old_root), root_path(new_root)
    resident = root_path(resident_root or old)
    host = os.uname().nodename
    node = io.load(old / 'coordination' / f'node_{rank:02d}.json')
    require(node['hostname'] == host and node['rank'] == rank, 'old worker Pod/rank mismatch')
    expected = node['controller']
    actual = proc(expected['pid'])
    if actual:
        require(actual['start_ticks'] == str(expected['start_ticks']), 'old PID reused')
        old_plan = io.load(old / 'launch/submit_plan.json')
        require(actual['argv'][-3:] == [old_plan.get('node_worker', str(old / 'launch/source/scripts/train/v6_rl_dlc_worker.py')),
                '--launch-plan', str(old / 'launch/submit_plan.json')], 'old worker argv mismatch')
        expected = dict(expected, argv_sha256=actual['argv_sha256'])
    else:
        require(node['phase'] in ('failed', 'succeeded'), 'old worker vanished without terminal evidence')
    heartbeat_path = Path('/tmp') / resident.joinpath('resident').name / host / 'stdout-heartbeat/status.json'
    heartbeat = io.load(heartbeat_path)
    require(heartbeat['hostname'] == host and time.time() - heartbeat['time'] < 60,
            'resident stdout heartbeat is not fresh')
    outer = proc(heartbeat['controller']['pid'])
    require(outer and outer['start_ticks'] == str(heartbeat['controller']['start_ticks']),
            'resident heartbeat identity mismatch')
    control = Path('/tmp/resident') / host / 'control-source/scripts/train/run_dlc_debug_reservation.py'
    git_task = io.exists(new/'launch/git_task.json')
    config = dict(schema=SCHEMA, git_task=git_task, old_root=str(old), new_root=str(new), resident_root=str(resident),
        rank=rank, node_count=4, hostname=host, job_id=job_id, pod_uid=pod_uid,
        old_worker=expected, outer_shell={k:outer[k] for k in ('pid','start_ticks','argv_sha256')},
        files=dict(new_entry=binding(new/'launch/entry.sh'), new_plan=binding(new/'launch/submit_plan.json'),
                   **({'git_task': binding(new/'launch/git_task.json')} if git_task else
                      {'new_manifest': binding(new/'launch/source_sha256.json')}),
                   old_plan=binding(old/'launch/submit_plan.json'),
                   control=binding(control), control_io=binding(control.with_name('dlc_control_io_retry.py')),
                   implementation=binding(SELF), io_adapter=binding(Path(io.__file__)),
                   io_helper=binding(SELF.with_name('dlc_control_io_retry.py')),
                   identities=binding(SELF.with_name('watch_v6_rl_local_credit.py'))))
    validate(config)
    target = Path(config_path).resolve()
    require(target == directory(config)/'config.json', 'config must use new_root/recovery/node_XX/config.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    with command_lock(config):
        previous = read_optional(target)
        require(previous is None or previous == config, 'different recovery config already exists')
        if previous is None:
            io.write_json(target, config)
    return config


@contextlib.contextmanager
def command_lock(config, name='command.lock'):
    with (directory(config) / name).open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def gpu_clients():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=20)
    return [dict(gpu=line.split(',')[0].strip(), pid=int(line.split(',')[1]))
            for line in result.stdout.splitlines() if line.strip()]


def postrun(config, root):
    rank = config['rank']
    launch = read_optional(root/'standby'/f'node_{rank:02d}_postrun_launch.json')
    if not launch:
        return None
    ready_path = root/'standby'/f'node_{rank:02d}_postrun.json'
    ready = read_optional(ready_path)
    actual = proc(launch['pid'])
    if not actual:
        return None
    require(ready and ready['pid'] == actual['pid'] and ready['gpus'] == list(range(16)),
            'postrun ready receipt differs from live occupier')
    expected_tail = [str(root/'launch/source/verl/scripts/train/action_head_sft_data/gpu_occupier.py'),
        '--gpus', ','.join(map(str, range(16))), '--require-gpu-count', '16', '--ready-file', str(ready_path)]
    require(actual['argv'][-len(expected_tail):] == expected_tail,
            'postrun occupier is not the bound attempt executable/ready path')
    return {k: actual[k] for k in ('pid', 'start_ticks', 'argv_sha256')}


def business_processes(config, root):
    """Read-only namespace scan catches Ray children that outlived their parent."""
    protected = {1, os.getpid(), config['outer_shell']['pid']}
    needles = [str(root/'launch').encode(), str(root/'services').encode(), str(runtime(root)).encode()]
    if config.get('git_task') and root == Path(config['new_root']):
        needles.append(io.load(root/'launch/git_task.json')['checkout_root'].encode())
    # During PPU/pyarrow bootstrap, children inherit this separate startup
    # cache namespace before the worker replaces it with /tmp/v6rd/<hash>.
    for line in io.read_bytes(root/'launch/entry.sh').decode().splitlines():
        if line.startswith('RUNTIME=/tmp/v6startup/'):
            startup = line.partition('=')[2]
            require(all(c in '0123456789abcdef/' for c in startup.removeprefix('/tmp/v6startup/')),
                    'unexpected startup runtime expression')
            needles.append(startup.encode())
    found = []
    for item in Path('/proc').iterdir():
        if not item.name.isdigit() or int(item.name) in protected:
            continue
        try:
            argv = (item/'cmdline').read_bytes()
            environment = (item/'environ').read_bytes()
        except FileNotFoundError:
            continue
        # Inspect namespace membership only; never persist process environments.
        if any(needle in argv or needle in environment for needle in needles):
            actual = proc(int(item.name))
            if actual:
                found.append({k:actual[k] for k in ('pid','start_ticks','argv_sha256')})
    return found


def bound_debug(config):
    state = io.load(debug_root(config)/'debug-status.json')
    start = io.load(debug_root(config)/'debug-start.json')
    actual = proc(state['pid'])
    require(state['hostname'] == config['hostname'] == start['hostname']
            and state['pid'] == start['pid'] and time.time()-state['heartbeat'] < 60,
            'resident reservation state stale or mismatched')
    require(actual and config['files']['control']['path'] in actual['argv']
            and str(debug_root(config)) in actual['argv'], 'resident reservation argv changed')
    proof = read_optional(directory(config)/'shutdown.json')
    if proof and proof.get('debug_controller'):
        require(exact_live(proof['debug_controller']), 'resident debug controller identity changed since shutdown')
    return state


def status(config):
    validate(config, files=False)
    old = Path(config['old_root'])
    return dict(schema=SCHEMA, hostname=config['hostname'], rank=config['rank'],
        worker_alive=exact_live(config['old_worker']), outer_alive=True,
        old_node=read_optional(old/'coordination'/f"node_{config['rank']:02d}.json"),
        old_business_exit=read_optional(Path(config['resident_root'])/'resident/nodes'/config['hostname']/'rl-exit.json'),
        postrun=postrun(config, old), debug=read_optional(debug_root(config)/'debug-status.json'),
        shutdown=read_optional(directory(config)/'shutdown.json'),
        owner=read_optional(directory(config)/'owner-start.json'),
        business_exit=read_optional(directory(config)/'business-exit.json'))


def wait_until(predicate, seconds, message):
    deadline = time.monotonic()+seconds
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise TimeoutError(message)
        time.sleep(1)


def config_hash(config):
    return io.digest(json.dumps(config, sort_keys=True, separators=(',', ':')).encode())


def shutdown_snapshot(config):
    validate(config, files=False)
    old = Path(config['old_root'])
    if exact_live(config['old_worker']):
        return None
    state = io.load(old/'coordination'/f"node_{config['rank']:02d}.json")
    require(state['controller'] == {k:config['old_worker'][k] for k in ('pid','start_ticks')},
            'old worker receipt identity changed')
    if state['phase'] not in ('failed','succeeded'):
        return None
    exit_record = read_optional(Path(config['resident_root'])/'resident/nodes'/config['hostname']/'rl-exit.json')
    if not exit_record:
        return None
    holder = postrun(config, old)
    remaining = business_processes(config, old)
    if any(not holder or r['pid'] != holder['pid'] for r in remaining):
        return None
    debug = read_optional(debug_root(config)/'debug-status.json')
    if not debug:
        return None
    bound_debug(config)
    clients = gpu_clients()
    if not holder and not (debug.get('mode')=='idle' and debug.get('transition')=='ready' and debug.get('occupier_pid')):
        return None
    if len(clients) != 16 or len({r['gpu'] for r in clients}) != 16 or len({r['pid'] for r in clients}) != 1:
        return None
    return dict(schema=SCHEMA, config_sha256=config_hash(config), hostname=config['hostname'],
        rank=config['rank'], old_root=config['old_root'], new_root=config['new_root'],
        old_worker=config['old_worker'], outer_shell=config['outer_shell'],
        postrun=holder, debug_controller=proc(debug['pid']), business_exit=exit_record,
        cleanup_verified=True, standby_restored=True, gpu_clients=clients,
        gpu_ownership_verified=False,
        gpu_ownership_note='ready container occupier and 16-card single host client; mapping requires the owned release-to-zero check',
        time=time.time())


def stop_old(config, timeout):
    validate(config)
    root = directory(config)
    with command_lock(config):
        proof = read_optional(root/'shutdown.json')
        if proof:
            require(proof['config_sha256']==config_hash(config), 'shutdown proof belongs to another plan')
            require(shutdown_snapshot(config) is not None, 'previous shutdown proof no longer holds')
            return proof
        intent = read_optional(root/'stop-intent.json')
        if intent:
            require(intent['config_sha256']==config_hash(config), 'stop intent belongs to another plan')
        else:
            actual = proc(config['old_worker']['pid'])
            require(not actual or exact_live(config['old_worker']), 'old PID identity changed; refuse signal')
            node = io.load(Path(config['old_root'])/'coordination'/f"node_{config['rank']:02d}.json")
            require(node['controller']=={k:config['old_worker'][k] for k in ('pid','start_ticks')},
                    'old coordinator identity changed before stop')
            cleaning = node['phase'] in ('failed','succeeded')
            io.write_json(root/'stop-intent.json', dict(config_sha256=config_hash(config), time=time.time(),
                worker=config['old_worker'], outer=config['outer_shell'],
                signal='SIGTERM' if actual and not cleaning else 'none', already_cleaning=cleaning))
            if actual and not cleaning:
                # A terminal phase precedes finally. A second exception there
                # would interrupt service cleanup and postrun occupation.
                latest = io.load(Path(config['old_root'])/'coordination'/f"node_{config['rank']:02d}.json")
                require(latest['controller']==node['controller'],'worker changed before signal')
                if latest['phase'] not in ('failed','succeeded'):
                    signal_pid(config['old_worker'], {config['outer_shell']['pid'], os.getpid()})
        # A repeated command only observes the original intent; it never sends
        # another TERM that could interrupt the worker's own finally cleanup.
        proof = wait_until(lambda: shutdown_snapshot(config), timeout,
            'old business cleanup/standby not verified; inspect before any further action')
        io.write_json(root/'shutdown.json', proof)
        return proof


def control(config, mode):
    bound_debug(config)
    result = subprocess.run([sys.executable, '-B', config['files']['control']['path'], mode,
        '--output', str(debug_root(config)), '--lease-seconds', '600'],
        capture_output=True, text=True, timeout=160)
    require(result.returncode == 0, 'resident lease transition failed: '+result.stderr[-500:])
    state = bound_debug(config)
    require(state['mode'] == mode or mode=='idle' and state['transition']=='waiting_for_gpu_work',
            'resident lease acknowledgement mismatch')
    return state


def release_postrun(config, root, expected=None):
    holder = postrun(config, root)
    if not holder:
        require(expected is None or not exact_live(expected), 'bound postrun process lost its receipt')
        return
    if expected:
        require(holder==expected, 'postrun PID/argv identity changed')
    signal_pid(holder, {config['outer_shell']['pid'], os.getpid()})
    wait_until(lambda: not exact_live(holder), 45, 'owned postrun occupier did not exit')


def start_new(config, config_path):
    old, new, _ = validate(config)
    root = directory(config)
    with command_lock(config):
        intent = read_optional(root/'start-intent.json')
        if intent:
            require(intent['config_sha256']==config_hash(config), 'existing start intent has another binding')
            launch = read_optional(root/'owner-start.json')
            require(launch is not None, 'uncertain start intent; inspect owner process, never replay')
            return dict(already_dispatched=True, owner=launch)
        require(shutdown_snapshot(config) is not None, 'old local business is not safely stopped')
        for rank in range(4):
            proof = io.load(new/'recovery'/f'node_{rank:02d}'/'shutdown.json')
            require(proof['new_root']==str(new) and proof['rank']==rank and proof['cleanup_verified']
                    and proof['standby_restored'], 'all four nodes must acknowledge shutdown before restart')
        require(not io.exists(new/'coordination'/f"node_{config['rank']:02d}.json")
                and not runtime(new).exists(), 'new worker/runtime already started; refuse replay')
        command = [sys.executable, '-B', str(SELF), 'owner', '--config', str(config_path)]
        io.write_json(root/'start-intent.json', dict(config_sha256=config_hash(config), command=command,time=time.time()))
        with (root/'owner.log').open('xb') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        identity = proc(child.pid)
        require(identity is not None, 'owner exited before launch acknowledgement; inspect without replay')
        launch = dict(schema=SCHEMA, identity={k:identity[k] for k in ('pid','start_ticks','argv_sha256')},
                      config_sha256=config_hash(config), command=command, time=time.time())
        io.write_json(root/'owner-start.json', launch)
        return launch


class OwnerHeartbeat:
    """Existing owner pattern: no shared-disk operation on the stdout thread."""
    def __init__(self, config, event='resident_rl_attempt_liveness'):
        require(exact_live(config['outer_shell']), 'original stdout shell identity changed')
        self.config, self.phase = config, dict(phase='owner_starting', worker_pid=None)
        self.event = event
        self.owner = {k:v for k,v in process_identity(os.getpid()).items() if k in ('pid','start_ticks')}
        self.fd = os.open(f"/proc/{config['outer_shell']['pid']}/fd/1", os.O_WRONLY | os.O_NONBLOCK)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def update(self, **values):
        self.phase = dict(self.phase, **values)

    def run(self):
        while not self.stop.is_set():
            record = dict(event=self.event, time=time.time(),
                hostname=self.config['hostname'], rank=self.config['rank'], owner=self.owner,
                outer_shell=self.config['outer_shell'], **self.phase,
                meaning='actual attempt phase and process identity; not optimizer success')
            try:
                os.write(self.fd, (json.dumps(record)+'\n').encode())
            except BlockingIOError:
                pass
            self.stop.wait(20)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        os.close(self.fd)


def child_to_signal(config, expected):
    """Bind the exec'd worker again, rather than assuming a bootstrap shell."""
    require(exact_live(expected), 'new child PID was reused')
    new = Path(config['new_root'])
    actual = proc(expected['pid'])
    require(actual is not None, 'new child exited during identity verification')
    if config.get('git_task'):
        spec = io.load(check_binding(config['files']['git_task']))
        require(actual['argv'][-3:] == [str(Path(spec['checkout_root'])/'repo/scripts/train/run_git_task.py'),
                    '--spec', config['files']['git_task']['path']],
                'Git launcher process does not match the bound local checkout/spec')
        return {k:actual[k] for k in ('pid','start_ticks','argv_sha256')}
    node = read_optional(new/'coordination'/f"node_{config['rank']:02d}.json")
    plan = io.load(check_binding(config['files']['new_plan']))
    worker_tail = [plan.get('node_worker', str(new/'launch/source/scripts/train/v6_rl_dlc_worker.py')),
                   '--launch-plan', str(new/'launch/submit_plan.json')]
    if node:
        require(node['hostname']==config['hostname'] and node['rank']==config['rank']
                and node['controller']['pid']==expected['pid']
                and str(node['controller']['start_ticks'])==str(expected['start_ticks'])
                and actual['argv'][-3:]==worker_tail,
                'new coordinator does not bind the actual worker process')
        if node['phase'] in ('failed','succeeded'):
            return None  # Its own finally is already cleaning; do not interrupt it.
    elif actual['argv'][-3:] != worker_tail:
        require(actual['argv'][-1]==config['files']['new_entry']['path'] and not gpu_clients(),
                'bootstrap identity or GPU ownership is unproven; refuse signal')
    return {k:actual[k] for k in ('pid','start_ticks','argv_sha256')}


def owner(config):
    # The lock includes every cleanup action. A duplicate owner must fail
    # before acquiring a lease or touching either reservation owner.
    with command_lock(config, 'owner.lock'):
        require(not io.exists(directory(config)/'business-start.json')
                and not io.exists(directory(config)/'business-exit.json'),
                'owner already executed; preserve its existing receipts')
        return _owner_locked(config)


def stop_child_if_running(config, child, expected):
    if child.poll() is not None:
        return
    try:
        target = child_to_signal(config, expected)
        if target:
            signal_pid(target, {config['outer_shell']['pid'],os.getpid()})
    except (RuntimeError, ProcessLookupError):
        # Popen, unlike an arbitrary PID lookup, can prove our exact child
        # ended between poll and signal. A live mismatched PID still fails.
        if child.poll() is None:
            raise
        return
    child.wait(timeout=300)


def _owner_locked(config):
    root = directory(config)
    heartbeat = OwnerHeartbeat(config)
    child = identity = None
    error = cleanup_error = None
    restored = False
    transition_started = False
    def interrupted(signum, _frame):
        raise InterruptedError(f'owned RL attempt interrupted: {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        validate(config)
        require(not io.exists(root/'business-start.json') and not io.exists(root/'business-exit.json'),
                'owner already executed; never replay the same attempt')
        proof = io.load(root/'shutdown.json')
        require(proof['config_sha256']==config_hash(config) and shutdown_snapshot(config),
                'shutdown evidence changed before start')
        heartbeat.update(phase='releasing_owned_standby')
        transition_started = True
        control(config, 'active')
        release_postrun(config, Path(config['old_root']), proof['postrun'])
        wait_until(lambda: not gpu_clients(), 45, 'unknown GPU clients remain; refuse business launch')
        io.write_json(root/'standby-release.json', dict(time=time.time(),
            old_postrun=proof['postrun'], prior_gpu_clients=proof.get('gpu_clients'),
            resident_active_ack=True, remaining_gpu_clients=[],
            ownership_method='only_identity_bound_reservations_released_then_zero_host_clients',
            gpu_ownership_verified=True))
        new = Path(config['new_root'])
        require(not runtime(new).exists(), 'new runtime unexpectedly exists')
        env = dict(os.environ, RANK=str(config['rank']), WORLD_SIZE='4', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        # A fresh worker inherits the original shell stdout for its own existing
        # heartbeat, then redirects detailed logs to the new attempt directory.
        require(exact_live(config['outer_shell']), 'outer shell changed before launching worker')
        if config.get('git_task'):
            spec = io.load(check_binding(config['files']['git_task']))
            cwd = Path(spec['checkout_root'])/'repo'
            command = [sys.executable, '-B', str(cwd/'scripts/train/run_git_task.py'),
                       '--spec', config['files']['git_task']['path']]
        else:
            cwd = new/'launch/source'
            command = ['bash', config['files']['new_entry']['path']]
        with open(f"/proc/{config['outer_shell']['pid']}/fd/1", 'wb', buffering=0) as console:
            child = subprocess.Popen(command, env=env,
                cwd=cwd, stdin=subprocess.DEVNULL, stdout=console,
                stderr=console, start_new_session=True)
        identity = proc(child.pid)
        require(identity is not None, 'new business exited before PID receipt')
        identity = {k:identity[k] for k in ('pid','start_ticks')}
        io.write_json(root/'business-start.json', dict(identity=identity, new_root=str(new),
            rank=config['rank'], entry=config['files']['new_entry'], config_sha256=config_hash(config),time=time.time()))
        heartbeat.update(phase='business_running', worker_pid=child.pid)
        renewed = time.monotonic()
        while child.poll() is None:
            require(exact_live(config['outer_shell']), 'resident shell exited during business')
            node = read_optional(new/'coordination'/f"node_{config['rank']:02d}.json") or {}
            heartbeat.update(phase='business_'+node.get('phase','starting'), worker_pid=child.pid)
            io.write_json(root/'owner-status.json', dict(heartbeat.phase,time=time.time(),owner=heartbeat.owner))
            if time.monotonic()-renewed >= 240:
                control(config, 'active')
                renewed = time.monotonic()
            time.sleep(5)
    except BaseException as exc:
        error = repr(exc)
        heartbeat.update(phase='business_error_cleanup', error=error)
    finally:
        try:
            require(transition_started, 'preflight failed before any lease/child takeover; reservations unchanged')
            if child is not None and child.poll() is None:
                stop_child_if_running(config, child, identity)
            # No hard kill or global Ray cleanup: insufficient ownership keeps
            # this attempt failed and the original resident shell untouched.
            new = Path(config['new_root'])
            holder = postrun(config, new)
            remaining = business_processes(config, new)
            require(not any(not holder or r['pid']!=holder['pid'] for r in remaining),
                    'new business still has owned processes; refusing to declare cleanup')
            heartbeat.update(phase='restoring_resident_standby', returncode=child.poll() if child else None)
            release_postrun(config, new, holder)
            control(config, 'idle')
            def standby_ready():
                state = bound_debug(config)
                return state if state.get('mode')=='idle' and state.get('transition')=='ready' and state.get('occupier_pid') else None
            wait_until(standby_ready, 150, 'resident standby restoration not verified')
            restored = True
        except BaseException as exc:
            cleanup_error = repr(exc)
        code = child.poll() if child else None
        record = dict(schema=SCHEMA, hostname=config['hostname'], rank=config['rank'],job_id=config['job_id'],
            pod_uid=config['pod_uid'],new_root=config['new_root'],worker_identity=identity,
            entry_sha256=config['files']['new_entry']['sha256'],returncode=code,time=time.time(),
            cleanup_verified=transition_started and cleanup_error is None,standby_restored=restored,
            error=error,cleanup_error=cleanup_error,automatic_retry=False)
        heartbeat.update(phase='finished' if code==0 and not error and not cleanup_error else 'failed',
                         returncode=code, cleanup_verified=cleanup_error is None,
                         standby_restored=restored, error=error, cleanup_error=cleanup_error)
        io.write_json(root/'business-exit.json',record)
        heartbeat.close()
    return 0 if code==0 and not error and not cleanup_error else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','status','stop-old','start-new','owner'))
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--old-root',type=Path)
    parser.add_argument('--new-root',type=Path)
    parser.add_argument('--resident-root',type=Path)
    parser.add_argument('--rank',type=int)
    parser.add_argument('--job-id')
    parser.add_argument('--pod-uid')
    parser.add_argument('--timeout',type=int,default=300)
    args=parser.parse_args()
    require(1 <= args.timeout <= 600,'timeout must be bounded at 1..600 seconds')
    if args.action=='prepare':
        require(all(value is not None for value in (args.old_root,args.new_root,args.rank,args.job_id,args.pod_uid)),
                'prepare needs old/new roots, rank, official job ID and Pod UID')
        result=prepare(args.old_root,args.new_root,args.rank,args.job_id,args.pod_uid,args.config,args.resident_root)
    else:
        config=io.load(args.config)
        require(args.config.resolve()==directory(config)/'config.json','unexpected config location')
        if args.action=='owner':
            raise SystemExit(owner(config))
        result={'status':status,'stop-old':lambda c:stop_old(c,args.timeout),
                'start-new':lambda c:start_new(c,args.config)}[args.action](config)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
