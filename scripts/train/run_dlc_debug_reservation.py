#!/usr/bin/env python3
"""Keep a debug node reserved when idle, using the existing GPU reservation."""
import argparse
import errno
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'scripts/eval'))
from v6_rl_checkpoint_companion import WaitingGpuReservation, atomic_json
from watch_checkpoints import stop_process_group

# The checkpoint helpers prepend their own search paths. Load this dependency
# by its sibling path so an older PYTHONPATH copy cannot change the CLI ABI.
_io_spec = importlib.util.spec_from_file_location(
    '_workspace_control_io_retry', Path(__file__).with_name('dlc_control_io_retry.py'))
CONTROL_IO = importlib.util.module_from_spec(_io_spec)
_io_spec.loader.exec_module(CONTROL_IO)


class GPUProbeError(RuntimeError):
    pass


def compute_pids():
    try:
        query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], capture_output=True, text=True,
            check=True, timeout=15)
        return {int(line.strip()) for line in query.stdout.splitlines() if line.strip()}
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        raise GPUProbeError(str(exc)) from exc


def transition_reservation(reservation, mode, current):
    """Defer idle occupation while real work runs; never stop that work."""
    if mode == current:
        return current, 'ready'
    if mode == 'idle':
        if compute_pids():
            return current or 'active', 'waiting_for_gpu_work'
        reservation.start()
    else:
        owned_pid = reservation.process.pid if reservation.process else None
        reservation.stop('user_debug_active')
        # Other GPU clients are allowed to continue. Verify only our occupier.
        if owned_pid is not None:
            deadline = time.monotonic() + 30
            while owned_pid in compute_pids():
                if time.monotonic() >= deadline:
                    raise RuntimeError('owned occupier CUDA context remains; debug release unverified')
                time.sleep(1)
    return mode, 'ready'


def read_request(control, previous):
    """A missing receipt never revokes the last acknowledged GPU ownership."""
    try:
        return CONTROL_IO.retry_estale(lambda: json.loads(control.read_text()), str(control))
    except FileNotFoundError:
        if previous is not None:
            print(json.dumps(dict(event='control_request_unavailable', time=time.time(),
                progress_unavailable=True, retaining_previous_request=True)), flush=True)
            return previous
        return dict(mode='idle', requested_at=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'idle', 'active', 'status'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sleep-pid', type=int)
    parser.add_argument('--lease-seconds', type=int, default=600)
    args = parser.parse_args()
    root = args.output.resolve()
    if not root.is_relative_to('/data/zerovln/artifacts'):
        parser.error('output must be on the authorized artifact disk')
    root.mkdir(parents=True, exist_ok=True)
    control, status = root/'requested-mode.json', root/'debug-status.json'
    CONTROL_IO.install(sys.modules[__name__], root, reads=args.action == 'run')
    if args.action == 'status':
        print(status.read_text() if status.exists() else '{"state":"not_started"}')
        return
    if args.action in ('active', 'idle'):
        from dlc_reservation_request import request_mode
        state=request_mode(root,args.action,lease_seconds=args.lease_seconds,write=atomic_json)
        print(json.dumps(state))
        return
    if not args.sleep_pid:
        parser.error('run requires the owned sleep PID')
    with (root/'debug-supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stopping = False
        def stop(*_):
            nonlocal stopping
            stopping = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        config = dict(waiting_gpu_reservation=dict(enabled=True, gpu_count=16,
            python=sys.executable, script=str(REPO/'verl/scripts/train/action_head_sft_data/gpu_occupier.py'),
            startup_timeout=120))
        reservation = WaitingGpuReservation(config, root, stop_process_group)
        atomic_json(root/'debug-start.json', dict(pid=os.getpid(), hostname=os.uname().nodename,
                    sleep_pid=args.sleep_pid, source=str(Path(__file__).resolve()), started_at=time.time()))
        current = None
        request = None
        next_console_heartbeat = 0.
        try:
            while not stopping:
                if not Path(f'/proc/{args.sleep_pid}').exists():
                    raise RuntimeError('owned sleep process exited')
                request = read_request(control, request)
                mode = request['mode']
                if mode not in ('active','idle'):
                    raise ValueError('unsupported debug mode')
                transition = 'ready'
                try:
                    if mode == 'active' and time.time() > request.get('lease_expires_at', 0):
                        if not compute_pids():
                            request = dict(mode='idle', requested_at=time.time(), reason='expired_debug_lease')
                            atomic_json(control, request)
                            mode = 'idle'
                    current, transition = transition_reservation(reservation, mode, current)
                except GPUProbeError as exc:
                    # A failed GPU probe does not prove the node is idle.
                    transition = 'gpu_probe_retry'
                    print(json.dumps(dict(event=transition, error=repr(exc), time=time.time())), flush=True)
                reservation.check()
                state = dict(mode=current, requested_at=request['requested_at'],
                    requested_mode=mode, transition=transition,
                    heartbeat=time.time(), hostname=os.uname().nodename, pid=os.getpid(),
                    sleep_pid=args.sleep_pid, occupier_pid=reservation.process.pid if reservation.process else None,
                    lease_expires_at=request.get('lease_expires_at'),
                    gpu_count=16, meaning='debug node reservation; no training progress')
                if 'RL_EXIT_CODE' in os.environ:
                    code = int(os.environ['RL_EXIT_CODE'])
                    state.update(business_exit_code=code, automatic_rl_retry=False,
                        stage='rl_failed_reserving' if code else 'rl_completed_reserving',
                        meaning='RL exited; only standby GPU reservation remains')
                atomic_json(status, state)
                if time.monotonic() >= next_console_heartbeat:
                    print(json.dumps(dict(event='workspace_liveness', **state)), flush=True)
                    next_console_heartbeat = time.monotonic()+30
                time.sleep(5)
        finally:
            reservation.stop('debug_supervisor_exit')


if __name__ == '__main__':
    main()
