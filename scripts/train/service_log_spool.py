"""Keep service stdout off shared storage; archive an exclusive local spool.

The archive thread may block without blocking the service. Spools are never
deleted until all bytes have been archived. A full spool fails visibly rather
than silently dropping logs. This does not make checkpoints or assets local.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid


class LogSpool:
    def __init__(self, destination, local_root, *, interval=1., max_bytes=8 << 30):
        self.destination = Path(destination)
        root = Path(local_root)
        root.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(str(self.destination).encode()).hexdigest()[:16]
        self.path = root / f'{key}-{os.getpid()}-{uuid.uuid4().hex}.log'
        self.handle = self.path.open('xb', buffering=0)
        self.offset = 0
        self.reclaimed_bytes = 0
        self.reclaim_error = None
        self.base = None
        self.error = None
        self.interval, self.max_bytes = interval, max_bytes
        self.finished = threading.Event()
        self.drained = threading.Event()
        self.wake = threading.Event()
        self.thread = threading.Thread(target=self._archive, daemon=True,
                                       name='service-log-archive')
        self.thread.start()

    def _copy_once(self):
        # Only this thread touches the archive. Reopen after each operation;
        # pwrite at a stable offset makes a partial write/ESTALE retry idempotent.
        # Idle services must not poll shared storage on every local tick.
        # Read the local tail first; touch CPFS only when bytes need archiving.
        with self.path.open('rb') as source:
            source.seek(self.offset)
            chunk = source.read(1 << 20)
        if not chunk:
            return False
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.destination, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            if self.base is None:
                self.base = os.fstat(fd).st_size
            start = self.base + self.offset
            size = os.fstat(fd).st_size
            if not start <= size <= start + len(chunk):
                raise RuntimeError('log archive changed by another writer')
            # A previously interrupted partial write must contain our bytes.
            overlap = os.pread(fd, size - start, start)
            if overlap != chunk[:len(overlap)]:
                raise RuntimeError('log archive content changed by another writer')
            done = 0
            while done < len(chunk):
                count = os.pwrite(fd, chunk[done:], start + done)
                if count <= 0:
                    raise OSError('zero-length archive write')
                done += count
            os.fsync(fd)
            self.offset += len(chunk)
            return True
        finally:
            os.close(fd)

    def _reclaim_archived(self):
        # Punch only whole blocks already fsynced to the archive. Keep the file
        # size and writer offset intact: a child may be concurrently appending.
        end = self.offset // (1 << 20) * (1 << 20)
        if end <= self.reclaimed_bytes:
            return
        fd = os.open(self.path, os.O_RDWR)
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            fallocate = libc.fallocate
            fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong]
            fallocate.restype = ctypes.c_int
            if fallocate(fd, 3, self.reclaimed_bytes, end-self.reclaimed_bytes):
                code = ctypes.get_errno()
                raise OSError(code, os.strerror(code))
            self.reclaimed_bytes = end
            self.reclaim_error = None
        except (OSError, AttributeError) as exc:
            # Unsupported filesystems retain all local bytes. Physical-capacity
            # protection remains active; never pretend reclamation succeeded.
            self.reclaim_error = str(exc)
        finally:
            os.close(fd)

    def _archive(self):
        while True:
            self.wake.clear()
            try:
                copied = self._copy_once()
                self.error = None
                self._reclaim_archived()
                if self.finished.is_set() and self.offset == self.path.stat().st_size:
                    self.drained.set()
                    return
                if copied:
                    continue
            except Exception as exc:
                # Failure remains visible; retained local bytes are the source
                # of truth until successful archiving, never an empty success.
                self.error = f'{type(exc).__name__}: {exc}'
            self.wake.wait(self.interval)

    def status(self):
        offset = self.offset
        stat = self.path.stat()
        size = stat.st_size
        allocated = stat.st_blocks * 512
        return dict(local_spool=str(self.path), archive=str(self.destination),
                    bytes=size, archived_bytes=offset,
                    pending_bytes=size-offset, error=self.error,
                    allocated_bytes=allocated, reclaimed_bytes=self.reclaimed_bytes,
                    reclaim_error=self.reclaim_error,
                    capacity_exceeded=max(size-offset, allocated) >= self.max_bytes)

    def finish(self):
        """Signal EOF without waiting, so many archives can drain together."""
        self.handle.close()
        self.finished.set()
        self.wake.set()

    def close(self, timeout=5.):
        self.finish()
        if not self.drained.wait(timeout):
            return False  # Thread retains the spool and may finish later.
        self.path.unlink()
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--local-root', required=True)
    parser.add_argument('--drain-timeout', type=float, default=30.)
    parser.add_argument('--max-bytes', type=int, default=8 << 30)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.max_bytes <= 0 or args.drain_timeout < 0:
        parser.error('command and positive capacity required')
    spool = LogSpool(args.archive, args.local_root, max_bytes=args.max_bytes)
    # The launcher's setsid owns this entire group, including service children.
    child = subprocess.Popen(command, stdout=spool.handle, stderr=subprocess.STDOUT)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: child.send_signal(signum)
                      if child.poll() is None else None)
    last_report = time.monotonic()
    while child.poll() is None:
        state = spool.status()
        if state['capacity_exceeded']:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            print(json.dumps(dict(event='service_log_capacity_exceeded', **state)), flush=True)
            break
        if time.monotonic() - last_report >= 20:
            print(json.dumps(dict(event='service_log_archive', time=time.time(),
                child_pid=child.pid, **state)), flush=True)
            last_report = time.monotonic()
        time.sleep(.2)
    drained = spool.close(args.drain_timeout)
    if not drained:
        print(json.dumps(dict(event='service_log_not_archived', **spool.status())), flush=True)
    return child.returncode if child.returncode else (0 if drained else 74)


if __name__ == '__main__':
    raise SystemExit(main())
