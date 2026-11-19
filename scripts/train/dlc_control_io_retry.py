"""Reopen transient stale CPFS control receipts without changing GPU ownership."""
import errno
from contextlib import contextmanager
from contextvars import ContextVar
import functools
import json
import os
import socket
from pathlib import Path
import sys
import time


_deadline = ContextVar('control_io_deadline', default=None)


@contextmanager
def bounded(deadline):
    previous = _deadline.get()
    token = _deadline.set(min(previous, deadline) if previous is not None else deadline)
    try:
        yield
    finally:
        _deadline.reset(token)


def check_deadline(label):
    limit = _deadline.get()
    if limit is not None and time.monotonic() >= limit:
        raise TimeoutError(f'control I/O deadline expired: {label}')


def retry_estale(operation, label):
    attempt = 0
    while True:
        check_deadline(label)
        try:
            return operation()
        except OSError as exc:
            if exc.errno != errno.ESTALE:
                raise
            attempt += 1
            if attempt <= 3 or attempt % 30 == 0:
                print(json.dumps(dict(event='control_io_estale_retry', time=time.time(),
                    operation=label, attempt=attempt, error=repr(exc),
                    errno=exc.errno, pid=os.getpid(), node=socket.gethostname(),
                    progress_unavailable=True)), flush=True)
            time.sleep(1)


def retry_atomic_json(operation, path):
    """Recreate the complete write when CPFS loses its own temporary file.

    ENOENT from unrelated inputs is not retryable. The established writer's
    temporary basename is bound to this target and this process.
    """
    path = Path(path)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    attempt = 0
    while True:
        try:
            return retry_estale(operation, f'atomic_json:{path}')
        except FileNotFoundError as exc:
            if exc.filename != str(temporary):
                raise
            attempt += 1
            if attempt <= 3 or attempt % 20 == 0:
                print(json.dumps(dict(event='control_atomic_missing_temp_retry',
                    time=time.time(), target=str(path), attempt=attempt,
                    errno=exc.errno, pid=os.getpid(), node=socket.gethostname(),
                    progress_unavailable=True)), flush=True)
            time.sleep(1)


def install(controller, root, *, reads=True, stats=False):
    """Scope the old controller's receipt reads and atomic writes to its root.

    Install before startup from the container-local Git checkout. Never inject
    it into an existing live interpreter or mutate its running code.
    """
    prefix = str(Path(root)).rstrip('/') + '/'
    if getattr(controller, '_control_io_retry_root', None) is not None:
        if controller._control_io_retry_root != prefix:
            raise ValueError('controller retry is already bound to another root')
        return

    def owns(path):
        value = str(path)
        return value == prefix[:-1] or value.startswith(prefix)

    def scoped(original):
        @functools.wraps(original)
        def wrapped(path, *args, **kwargs):
            if not owns(path):
                return original(path, *args, **kwargs)
            return retry_estale(lambda: original(path, *args, **kwargs),
                f'{original.__name__}:{path}')
        return wrapped

    # read_text retries a fresh open+read, rather than an already stale FD.
    if reads:
        Path.read_text = scoped(Path.read_text)
        Path.exists = scoped(Path.exists)
    if stats:
        # Path.is_file/exists and component progress polling also use stat.
        Path.stat = scoped(Path.stat)
    original_atomic = controller.atomic_json
    def atomic(path, *args, **kwargs):
        if not owns(path):
            return original_atomic(path, *args, **kwargs)
        return retry_atomic_json(lambda: original_atomic(path, *args, **kwargs), path)
    controller.atomic_json = atomic
    reservation = sys.modules.get('v6_rl_checkpoint_companion')
    if reservation is not None:
        reservation.atomic_json = atomic
    controller._control_io_retry_root = prefix
