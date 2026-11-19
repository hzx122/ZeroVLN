"""Share bounded native GL residency across the existing serial render queues.

SingleWorker still owns exactly one scene/camera and its original IPC/rendering.
Only idle entries can be evicted. Retiring children remain charged until close
returns; cold loads reserve memory before starting. RSS is conservative because
it includes shared pages. This is admission accounting, not an OS memory cap:
an underestimated scene fails visibly and is closed instead of silently raising
the budget. Active/recent queues retain a fair slot preference; idle capacity
can be borrowed after the grace interval. Never share navigation or task state.
"""
from pathlib import Path
from threading import Condition
import time

from .worker import SceneWorkers, SingleWorker, scene_key


class SharedSceneWorkers(SceneWorkers):
    def __init__(self, root, cache_size, timeout=None, *, max_bytes, load_reserve_bytes,
                 parallelism=1, idle_grace_seconds=30.):
        super().__init__(root, cache_size, timeout)
        if (type(max_bytes) is not int or type(load_reserve_bytes) is not int
                or not 0 < load_reserve_bytes <= max_bytes):
            raise ValueError('positive byte budget and bounded scene load reserve required')
        self.max_bytes, self.load_reserve_bytes = max_bytes, load_reserve_bytes
        if (type(parallelism) is not int or not 1 <= parallelism <= cache_size
                or idle_grace_seconds < 0):
            raise ValueError('valid cache queue count and idle grace required')
        self._parallelism, self._idle_grace = parallelism, idle_grace_seconds
        self._owners = {}
        self._queued = [0] * parallelism
        self._last_active = [float('-inf')] * parallelism
        self._condition = Condition()
        self._busy, self._pending, self._closing = set(), set(), set()
        self._memory, self._reserved, self._retiring = {}, {}, {}
        self._closed, self._error = False, None
        self.cache_hits = 0
        self.cache_wait_seconds = self.cold_seconds = self.warm_seconds = 0.
        self.queue_wait_seconds = self.queue_wait_max_seconds = 0.
        self.peak_accounted_bytes = self.peak_contexts = 0
        self.max_context_peak_rss_bytes = 0

    def _charge(self):
        return sum(self._memory.values()) + sum(self._reserved.values()) + sum(self._retiring.values())

    def _count(self):
        return len(self.workers) + len(self._reserved) + len(self._retiring)

    def _check_open(self):
        if self._error is not None:
            raise RuntimeError('native scene cache failed') from self._error
        if self._closed:
            raise RuntimeError('native scene cache is closed')

    def _record_peaks(self):
        self.peak_accounted_bytes = max(self.peak_accounted_bytes, self._charge())
        self.peak_contexts = max(self.peak_contexts, self._count())

    def _retire(self, key):
        worker = self.workers.pop(key)
        self._retiring[worker] = self._memory.pop(key)
        self._owners.pop(key)
        self.evictions += 1
        return worker

    def note_pending(self, shard, delta):
        with self._condition:
            self._queued[shard] += delta
            if self._queued[shard] < 0:
                raise RuntimeError('native render pending count became negative')
            self._last_active[shard] = time.monotonic()

    def _idle_victim(self, incoming_shard):
        """Preserve fair shares of active/recent queues; lend idle capacity.

        This prevents an over-capacity scan on one queue from evicting every
        warm key on the other queues. Fairness is a preference: byte pressure
        must still make progress if every idle entry is below its count share.
        """
        counts = [0] * self._parallelism
        for owner in self._owners.values():
            counts[owner] += 1
        now = time.monotonic()
        oldest = None
        for key in self.workers:
            if key in self._busy:
                continue
            oldest = oldest or key
            owner = self._owners[key]
            share = self.cache_size // self._parallelism + int(owner < self.cache_size % self._parallelism)
            active = self._queued[owner] or now - self._last_active[owner] <= self._idle_grace
            if owner == incoming_shard or counts[owner] > share or not active:
                return key
        return oldest

    def _close_worker(self, worker):
        with self._condition:
            self._closing.add(worker)
        try:
            worker.close()
        except BaseException as exc:
            # Keep ownership and its memory charge if shutdown was not proven.
            with self._condition:
                self._error = exc
            raise
        else:
            with self._condition:
                self._retiring.pop(worker)
        finally:
            with self._condition:
                self._closing.remove(worker)
                self._condition.notify_all()

    def _acquire(self, key, shard):
        with self._condition:
            while True:
                self._check_open()
                if key in self._busy or key in self._pending:
                    self._condition.wait()
                    continue
                if key in self.workers:
                    if self._owners[key] != shard:
                        raise RuntimeError('native scene changed its owning render queue')
                    self._busy.add(key)
                    self.workers.move_to_end(key)
                    self.cache_hits += 1
                    return self.workers[key], False
                self._pending.add(key)
                break
        try:
            while True:
                with self._condition:
                    self._check_open()
                    if (self._count() < self.cache_size and
                            self._charge() + self.load_reserve_bytes <= self.max_bytes):
                        self._reserved[key] = self.load_reserve_bytes
                        self._record_peaks()
                        break
                    # Wait for a retirement already in progress; do not evict
                    # extra scenes while their eventual free space is pending.
                    idle = self._idle_victim(shard)
                    if self._retiring or idle is None:
                        self._condition.wait()
                        continue
                    victim = self._retire(idle)
                # Never hold the cache lock during native teardown. Warm
                # renders on other queues may continue while this one closes.
                self._close_worker(victim)
            worker = SingleWorker(self.root, 1, self.timeout)
            with self._condition:
                self._check_open()
                self.workers[key] = worker
                self._owners[key] = shard
                self._memory[key] = self._reserved.pop(key)
                self._pending.remove(key)
                self._busy.add(key)
                self.loads += 1
                self._condition.notify_all()
            return worker, True
        except BaseException:
            with self._condition:
                self._pending.discard(key)
                self._reserved.pop(key, None)
                self._condition.notify_all()
            raise

    def _trim(self):
        while True:
            with self._condition:
                if self._closed:
                    return
                self._check_open()
                if self._charge() <= self.max_bytes:
                    return
                idle = next((k for k in self.workers if k not in self._busy), None)
                if self._retiring or idle is None:
                    self._condition.wait()
                    continue
                victim = self._retire(idle)
            self._close_worker(victim)

    def render(self, request, *, queued_at=None, shard=0):
        if type(shard) is not int or not 0 <= shard < self._parallelism:
            raise ValueError('invalid owning native render queue')
        key = scene_key(self.root, request)
        started = time.monotonic()
        worker, cold = self._acquire(key, shard)
        acquired = time.monotonic()
        success = False
        try:
            result = worker.render(request)
            success = True
            return result
        finally:
            ended = time.monotonic()
            try:
                memory = worker.resident_bytes()
                peak_memory = worker.peak_resident_bytes()
            except BaseException as exc:
                with self._condition:
                    self._error = exc
                    self._busy.remove(key)
                    self._condition.notify_all()
                raise
            oversized = max(memory, peak_memory) > self.load_reserve_bytes
            memory_error = RuntimeError('native context exceeded scene load memory reservation: '
                f'scene={key[0]} rss={memory} peak={peak_memory} reserve={self.load_reserve_bytes}')
            with self._condition:
                self._memory[key] = memory
                self.max_context_peak_rss_bytes = max(self.max_context_peak_rss_bytes, peak_memory)
                self._busy.remove(key)
                self.requests += int(success and not oversized)
                queued_seconds = max(0., started - queued_at) if queued_at is not None else 0.
                self.queue_wait_seconds += queued_seconds
                self.queue_wait_max_seconds = max(self.queue_wait_max_seconds, queued_seconds)
                self.cache_wait_seconds += acquired - started
                if cold:
                    self.cold_seconds += ended - acquired
                else:
                    self.warm_seconds += ended - acquired
                # Dead/failed child instances must acquire a fresh cold-load
                # reservation before SingleWorker is allowed to respawn them.
                victim = self._retire(key) if oversized or not memory else None
                if oversized:
                    self._error = memory_error
                self._record_peaks()
                self._condition.notify_all()
            if victim is not None:
                self._close_worker(victim)
            if oversized:
                raise memory_error
            self._trim()

    def health(self):
        with self._condition:
            items = list(self.workers.items())
            states = [worker.health() for _, worker in items]
            return dict(ok=not self._closed and self._error is None and all(s['ok'] for s in states),
                protocol=1, stateless=True, service_instances=1,
                renderer='habitat-mesh-software-egl',
                isolation='one_scene_camera_per_gl_process_v1', shared_scene_cache=True,
                cache_capacity=self.cache_size, cache_max_bytes=self.max_bytes,
                scene_load_reserve_bytes=self.load_reserve_bytes,
                accounted_bytes=self._charge(), peak_accounted_bytes=self.peak_accounted_bytes,
                resident_rss_bytes=sum(self._memory.values()), peak_contexts=self.peak_contexts,
                max_context_peak_rss_bytes=self.max_context_peak_rss_bytes,
                gl_workers=len(states), retiring_workers=len(self._retiring),
                pending_scene_loads=len(self._pending), active_renders=len(self._busy),
                queue_pending=list(self._queued), idle_grace_seconds=self._idle_grace,
                cached_contexts_per_queue=[sum(owner == i for owner in self._owners.values())
                                           for i in range(self._parallelism)],
                requests=self.requests, scene_loads=self.loads, evictions=self.evictions,
                cache_hits=self.cache_hits, cache_wait_seconds=self.cache_wait_seconds,
                queue_wait_seconds=self.queue_wait_seconds,
                queue_wait_max_seconds=self.queue_wait_max_seconds,
                cold_seconds=self.cold_seconds, warm_seconds=self.warm_seconds,
                cached_scenes=[Path(key[0]).stem for key, _ in items],
                worker_pids=[s['worker_pid'] for s in states],
                active_scenes=[Path(key[0]).stem for key in self._busy],
                error=str(self._error) if self._error is not None else None)

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            while self._pending or self._busy or self._closing:
                self._condition.wait()
            for key, worker in self.workers.items():
                self._retiring[worker] = self._memory[key]
            workers = list(self._retiring)
            self.workers.clear()
            self._owners.clear()
            self._memory.clear()
        errors = []
        for worker in workers:
            try:
                self._close_worker(worker)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError('native scene cache failed to close owned workers') from errors[0]
