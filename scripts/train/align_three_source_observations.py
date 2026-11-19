"""Shared bounded ESTALE-safe control I/O for container-local Git tasks."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

ART = Path('/data/zerovln/artifacts')

def _io():
    path = Path(__file__).resolve().with_name('dlc_control_io_retry.py')
    spec = importlib.util.spec_from_file_location('_three_source_camera_io', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bounded(operation, label):
    """Retain the existing ESTALE retry, with a finite preparation deadline."""
    until = time.monotonic() + 20
    def attempt():
        if time.monotonic() >= until:
            raise TimeoutError('three-source camera control I/O deadline: ' + label)
        return operation()
    return _io().retry_estale(attempt, label)


def read_bytes(path):
    path = Path(path)
    return bounded(path.read_bytes, 'three_source_camera_read:' + str(path))


def exists(path):
    return bounded(Path(path).exists, 'three_source_camera_exists:' + str(path))


def write_bytes(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    until = time.monotonic() + 20
    def operation():
        if time.monotonic() >= until:
            raise TimeoutError('three-source camera atomic-write deadline: ' + str(path))
        temporary.write_bytes(data)
        temporary.replace(path)
    _io().retry_atomic_json(operation, path)


def load(path):
    return json.loads(read_bytes(path))


def write_json(path, value):
    write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def digest(data):
    return hashlib.sha256(data).hexdigest()
