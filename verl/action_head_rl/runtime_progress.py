"""Completed learner work and bounded RPC acknowledgements, without CUDA calls."""
from __future__ import annotations

import json
import time
from pathlib import Path


LEARNER_RUNTIME_REVISION = "completed_work_bounded_rpc_v2"


class LearnerProgress:
    """Write only on completed work; an idle/stalled worker emits nothing."""

    def __init__(self, directory, rank, interval=30.0):
        self.path = Path(directory) / f"rank_{int(rank):04d}.jsonl"
        self.rank = int(rank)
        self.interval = interval
        self.last_write = None

    def completed(self, phase, *, force=False, now=None, **fields):
        now = time.monotonic() if now is None else now
        if not force and self.last_write is not None and now - self.last_write < self.interval:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        event = dict(rank=self.rank, phase=phase, unix_time=time.time(),
                     runtime_revision=LEARNER_RUNTIME_REVISION, **fields)
        with self.path.open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        self.last_write = now
        return True


def resolve_ray_results(value, *, ray_api, timeout=None, phase="learner RPC", clock=time.monotonic):
    """Resolve nested refs with one deadline for all ranks, including data fetch.

    Fetch ready refs immediately so a failed rank cannot hide behind its peers.
    Return only after all refs succeed. On timeout, preserve paths
    into the worker group's return value to identify the missing rank replies.
    No retries or partial success can release a staging barrier.
    """
    if timeout is not None and timeout <= 0:
        raise ValueError("RPC timeout must be positive")
    deadline = None if timeout is None else clock() + timeout
    resolved = {}

    def remaining():
        return None if deadline is None else max(0.0, deadline - clock())

    def collect(item, path, refs):
        if isinstance(item, ray_api.ObjectRef):
            if item in resolved:
                collect(resolved[item], path, refs)
            else:
                refs.setdefault(item, path)
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                collect(child, f"{path}[{index}]", refs)
        elif isinstance(item, dict):
            for key, child in item.items():
                collect(child, f"{path}[{key!r}]", refs)

    while True:
        refs = {}
        collect(value, "result", refs)
        if not refs:
            break
        pending = list(refs)
        total = len(pending)
        while pending:
            ready, missing = ray_api.wait(pending, num_returns=1,
                                          timeout=remaining(), fetch_local=False)
            if not ready:
                raise TimeoutError(f"{phase}: deadline {timeout:g}s; replies "
                                   f"{total-len(pending)}/{total}; pending paths="
                                   f"{[refs[ref] for ref in pending]}")
            for ref in ready:
                try:
                    resolved[ref] = ray_api.get(ref, timeout=remaining())
                except ray_api.exceptions.GetTimeoutError as error:
                    raise TimeoutError(f"{phase}: deadline {timeout}s fetching {refs[ref]}") from error
                except Exception as error:
                    error.add_note(f"{phase}: failed reply at {refs[ref]}")
                    raise
            pending = missing

    def rebuild(item):
        if isinstance(item, ray_api.ObjectRef):
            return rebuild(resolved[item])
        if isinstance(item, list):
            return [rebuild(child) for child in item]
        if isinstance(item, tuple):
            return tuple(rebuild(child) for child in item)
        if isinstance(item, dict):
            return {key: rebuild(child) for key, child in item.items()}
        return item

    return rebuild(value)
