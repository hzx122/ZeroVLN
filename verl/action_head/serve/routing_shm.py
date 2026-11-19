"""Own vLLM route buffers by service attempt and reclaim only unused files."""
from __future__ import annotations

import os
from pathlib import Path
import re
import stat
import uuid

BASE = 'vllm_routed_experts_buffer'
NAMESPACE_ENV = 'ACTION_HEAD_ROUTING_SHM_PREFIX'
OWNED = re.compile(r'vllm_routed_experts_buffer_ah_[0-9]+_[a-zA-Z0-9_-]+_[0-9a-f]{32}')
LEGACY = re.compile(r'vllm_routed_experts_buffer_[0-9]+_[0-9]+')


def new_namespace(service):
    return f'{BASE}_ah_{os.getpid()}_{service}_{uuid.uuid4().hex}'


def configure_namespace(module):
    """Worker and EngineCore inherit the same attempt-specific prefix."""
    prefix = os.environ.get(NAMESPACE_ENV)
    if prefix:
        if not OWNED.fullmatch(prefix):
            raise ValueError('invalid owned routing-buffer namespace')
        module._BUFFER_PREFIX = prefix


def live_references(names, *, shm_root=Path('/dev/shm'), proc_root=Path('/proc')):
    paths = {str(shm_root / name): name for name in names}
    live = {}
    for process in proc_root.iterdir():
        if not process.name.isdigit():
            continue
        try:
            for line in (process / 'maps').read_text().splitlines():
                fields = line.split(maxsplit=5)
                if len(fields) == 6 and fields[5] in paths:
                    live.setdefault(paths[fields[5]], set()).add(int(process.name))
            for fd in (process / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                except FileNotFoundError:
                    continue
                if target in paths:
                    live.setdefault(paths[target], set()).add(int(process.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return {name: sorted(pids) for name, pids in live.items()}


def _cleanup(candidates, *, shm_root, proc_root):
    snapshots = {}
    for path in candidates:
        try:
            value = path.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(value.st_mode) or value.st_uid != os.geteuid():
            raise ValueError('routing buffer is not an owned regular file: ' + str(path))
        snapshots[path.name] = value
    referenced = live_references(snapshots, shm_root=shm_root, proc_root=proc_root)
    removed = []
    for name, before in snapshots.items():
        if name in referenced:
            continue
        path = shm_root / name
        try:
            current = path.lstat()
        except FileNotFoundError:
            continue
        if (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError('routing buffer was replaced during cleanup: ' + name)
        path.unlink()
        removed.append(dict(name=name, bytes=before.st_size,
                            allocated_bytes=before.st_blocks * 512))
    return dict(removed=removed, referenced=referenced,
                released_bytes=sum(r['allocated_bytes'] for r in removed))


def cleanup_namespaces(prefixes, *, shm_root=Path('/dev/shm'), proc_root=Path('/proc')):
    prefixes = tuple(prefixes)
    if any(not OWNED.fullmatch(prefix) for prefix in prefixes):
        raise ValueError('cleanup requires exact service-owned namespaces')
    candidates = [p for p in shm_root.iterdir()
                  if any(p.name.startswith(prefix + '_') for prefix in prefixes)]
    return _cleanup(candidates, shm_root=shm_root, proc_root=proc_root)


def cleanup_legacy_orphans(*, shm_root=Path('/dev/shm'), proc_root=Path('/proc')):
    """Explicit repair only, after the bound evaluation owner has retired."""
    candidates = [p for p in shm_root.iterdir() if LEGACY.fullmatch(p.name)]
    return _cleanup(candidates, shm_root=shm_root, proc_root=proc_root)
