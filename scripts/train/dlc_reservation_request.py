"""Lightweight request/ack client for the existing GPU reservation controller.

Avoid importing the GPU/service stack for each lease renewal. A timeout remains
failure; unreadable state never permits GPU work or a second request.
"""
import errno
import json
import math
import os
import sys
import time
from pathlib import Path


def request_mode(root, action, *, lease_seconds=600, write, renewal=None):
    root=Path(root)
    if action not in ('active','idle'):
        raise ValueError('invalid reservation mode')
    control,status=root/'requested-mode.json',root/'debug-status.json'
    wait_seconds = 150.
    identity = None
    if renewal is not None:
        previous, identity = renewal['acknowledgement'], renewal['controller']
        expiry = previous.get('lease_expires_at', 0)
        remaining = float(expiry) - time.time()
        if (action != 'active' or previous.get('mode') != 'active'
                or previous.get('transition') != 'ready' or previous.get('occupier_pid') is not None
                or previous.get('pid') != identity['pid'] or not math.isfinite(remaining)
                or remaining <= 15):
            raise ValueError('renewal requires an acknowledged active lease with cleanup margin')
        # A delayed renewal ACK must not revoke an already granted lease.
        # Still stop BEFORE its original expiry, allowing owned cleanup.
        wait_seconds = min(remaining, float(lease_seconds)) - 15
    request = dict(mode=action, requested_at=time.time())
    if action == 'active':
        if lease_seconds <= 0:
            raise ValueError('active lease must be positive')
        request['lease_expires_at'] = time.time() + lease_seconds
    started = time.monotonic()
    deadline = started + wait_seconds
    next_warning = started + 30
    write(control, request)
    stale_reads = 0
    while time.monotonic() < deadline:
        if identity is not None:
            try:
                fields = Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(')', 1)[1].split()
            except FileNotFoundError:
                raise RuntimeError('reservation controller exited during renewal')
            if fields[19] != str(identity['start_ticks']) or fields[0] in ('Z', 'X'):
                raise RuntimeError('reservation controller identity changed during renewal')
        try:
            state = json.loads(status.read_text())
        except FileNotFoundError:
            state = {}
        except OSError as exc:
            if exc.errno != errno.ESTALE:
                raise
            # Reopen the same receipt on the next poll. Do not publish a
            # second request or interpret an unreadable acknowledgement
            # as a failed transition of the running GPU work.
            stale_reads += 1
            if stale_reads <= 3 or stale_reads % 30 == 0:
                print(json.dumps(dict(event='debug_ack_read_retry', path=str(status),
                    operation='reopen_acknowledgement', errno=exc.errno, attempt=stale_reads,
                    pid=os.getpid(), node=os.uname().nodename, progress_unavailable=True,
                    requested_at=request['requested_at'], error=repr(exc),
                    time=time.time())), file=sys.stderr, flush=True)
            state = {}
        if state:
            if (identity is None or state.get('pid') == identity['pid']) and state.get('requested_at') == request['requested_at'] and (
                state.get('mode') == action or (action == 'idle'
                    and state.get('transition') == 'waiting_for_gpu_work')):
                return state
        if time.monotonic() >= next_warning:
            print(json.dumps(dict(event='reservation_ack_delayed', path=str(status),
                node=os.uname().nodename, pid=os.getpid(), action=action,
                requested_at=request['requested_at'], waited_seconds=time.monotonic()-started,
                acknowledged_lease_expires_at=(renewal['acknowledgement']['lease_expires_at']
                    if renewal is not None else None), progress_unavailable=True)), flush=True)
            next_warning = time.monotonic()+30
        time.sleep(1)
    raise TimeoutError('debug mode transition not acknowledged; do not start GPU debug work')


class ActiveLeaseRenewer:
    """Renew independently of watcher status writes and queue/barrier reads.

    The granted lease and process identity rules remain in request_mode. A
    failed renewal interrupts the owner so its existing finally cleans GPU work.
    """
    def __init__(self, renew, on_failure, *, interval=120.):
        import threading
        self.renew, self.on_failure, self.interval = renew, on_failure, interval
        self.stop_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True, name='resident-lease-renewal')

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        while not self.stop_event.wait(self.interval):
            try:
                self.renew()
            except Exception as exc:
                self.error = exc
                print(json.dumps(dict(event='resident_lease_renewal_failed',
                    time=time.time(), pid=os.getpid(), error=repr(exc))), flush=True)
                self.on_failure()
                return

    def close(self):
        self.stop_event.set()
        # Never race a pending active request with the final idle handoff.
        self.thread.join()

    def check(self):
        if self.error is not None:
            raise RuntimeError('independent resident lease renewal failed') from self.error
