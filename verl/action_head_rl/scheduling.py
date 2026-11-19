"""Bounded full-episode scheduling, independent of the Ray transport.

The driver already expands cases into a stable G x K episode sequence. This
module changes only when and where those calls run; results are emitted in the
same group/sample order for reward normalization and the on-policy update.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Hashable, Iterator, Sequence
from typing import Any


def iter_grouped_rollouts(
    episode_specs: Sequence[Any],
    workers: Sequence[Any],
    *,
    group_size: int,
    max_in_flight_groups: int,
    submit: Callable[[Any, Any], Hashable],
    wait_one: Callable[[Sequence[Hashable]], Hashable],
    resolve: Callable[[Hashable], Any],
    scheduler: str = "completion",
) -> Iterator[tuple[int, Sequence[Any], list[Any]]]:
    """Yield complete ordered groups while keeping idle workers supplied.

    ``submit(worker, spec)`` must run a complete independent episode using the
    seed in ``spec``. ``wait_one`` returns one completed submitted handle;
    ``resolve`` retrieves its result or raises its original execution error.
    There is at most one outstanding call per worker in completion mode. At
    most ``max_in_flight_groups`` groups have been admitted but not yet yielded,
    including completed results waiting behind an earlier group. No episode is
    skipped, retried, truncated, or sampled again to handle a straggler.

    ``wave`` retains the original fixed actor queues and whole-wave admission
    barrier, for an otherwise identical performance comparison.
    """
    if group_size <= 0 or max_in_flight_groups <= 0:
        raise ValueError("group_size and max_in_flight_groups must be positive")
    if len(episode_specs) % group_size:
        raise ValueError("episode_specs must contain complete groups")
    if not workers:
        raise ValueError("at least one rollout worker is required")
    if scheduler not in {"completion", "wave"}:
        raise ValueError(f"unknown rollout scheduler: {scheduler}")

    total = len(episode_specs)
    if scheduler == "wave":
        stride = group_size * max_in_flight_groups
        for wave_start in range(0, total, stride):
            wave_end = min(total, wave_start + stride)
            handles = [
                submit(workers[index % len(workers)], episode_specs[index])
                for index in range(wave_start, wave_end)
            ]
            for local_start in range(0, len(handles), group_size):
                start = wave_start + local_start
                results = [resolve(handle) for handle in
                           handles[local_start:local_start + group_size]]
                yield start // group_size, episode_specs[start:start + group_size], results
            # Release the references before admitting the next tensor-rich wave.
            del handles, results
        return

    idle = deque(range(len(workers)))
    pending: dict[Hashable, tuple[int, int]] = {}
    completed: dict[int, dict[int, Any]] = {}
    next_submit = 0
    next_group = 0

    def fill_idle_workers() -> None:
        nonlocal next_submit
        admitted_end = min(total, (next_group + max_in_flight_groups) * group_size)
        while idle and next_submit < admitted_end:
            worker_index = idle.popleft()
            handle = submit(workers[worker_index], episode_specs[next_submit])
            if handle in pending:
                raise RuntimeError("submit returned a duplicate pending handle")
            pending[handle] = (worker_index, next_submit)
            next_submit += 1

    fill_idle_workers()
    while next_group < total // group_size:
        group_results = completed.get(next_group)
        if group_results is not None and len(group_results) == group_size:
            group_index = next_group
            results = [group_results[index] for index in range(group_size)]
            del completed[next_group], group_results
            next_group += 1
            # Admit the replacement group before the consumer processes this
            # group, so its work also overlaps normalization and log writes.
            fill_idle_workers()
            start = group_index * group_size
            yield group_index, episode_specs[start:start + group_size], results
            del results
            continue
        if not pending:
            raise RuntimeError("rollout scheduler exhausted work before completing a group")
        handle = wait_one(tuple(pending))
        if handle not in pending:
            raise RuntimeError("wait_one returned a handle that is not pending")
        worker_index, flat_index = pending.pop(handle)
        # Resolve before resubmission: an execution failure is propagated,
        # never replaced with a navigation reward or hidden behind another task.
        result = resolve(handle)
        group_index, sample_index = divmod(flat_index, group_size)
        completed.setdefault(group_index, {})[sample_index] = result
        del result, handle
        idle.append(worker_index)
        fill_idle_workers()
