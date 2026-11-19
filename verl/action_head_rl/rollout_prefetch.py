"""Keep admitted rollout actors busy while the driver waits for a learner wave.

One producer owns submission/completion handling. The consumer owns only the
ordered group cursor. Admission is bounded against that CONSUMED cursor, so
background progress never expands the existing in-flight group limit.
"""
from __future__ import annotations

from collections import deque
import threading
import time


class BackgroundGroupedRollouts:
    """Context-managed completion scheduler with cancellable polling.

    wait_one must return a ready handle or None within a bounded interval;
    resolve must only retrieve a ready result. Actor calls themselves are not
    cancelled here: the trainer's failure teardown owns its Ray actors.
    """

    def __init__(self, episode_specs, workers, *, group_size, max_in_flight_groups,
                 submit, wait_one, resolve):
        if group_size <= 0 or max_in_flight_groups <= 0 or not workers:
            raise ValueError('positive group bounds and rollout workers required')
        if len(episode_specs) % group_size:
            raise ValueError('rollout specs must contain complete groups')
        self.specs, self.workers = episode_specs, workers
        self.group_size, self.window = group_size, max_in_flight_groups
        self.submit, self.wait_one, self.resolve = submit, wait_one, resolve
        self.condition = threading.Condition()
        self.completed = {}
        self.next_group = 0
        self.stopped = self.finished = False
        self.error = None
        self.thread = None
        self.submitted = self.resolved = self.peak_unconsumed_groups = 0
        self.consumer_wait_seconds = 0.

    def __enter__(self):
        if self.thread is not None:
            raise RuntimeError('rollout prefetch context cannot be reused')
        self.thread = threading.Thread(target=self._produce, name='v6-rollout-prefetch', daemon=True)
        self.thread.start()
        return self

    def _produce(self):
        idle, pending = deque(range(len(self.workers))), {}
        try:
            while True:
                with self.condition:
                    if self.stopped:
                        return
                    end = min(len(self.specs), (self.next_group + self.window) * self.group_size)
                    while idle and self.submitted < end:
                        worker = idle.popleft()
                        index = self.submitted
                        handle = self.submit(self.workers[worker], self.specs[index])
                        if handle is None or handle in pending:
                            raise RuntimeError('rollout submission returned an invalid/duplicate handle')
                        pending[handle] = worker, index
                        self.submitted += 1
                        admitted = (self.submitted + self.group_size - 1) // self.group_size - self.next_group
                        self.peak_unconsumed_groups = max(self.peak_unconsumed_groups, admitted)
                    if not pending:
                        if self.submitted == len(self.specs):
                            self.finished = True
                            self.condition.notify_all()
                            return
                        # Every admitted group is complete; consumer progress
                        # releases the next slot. Do not spin or over-admit.
                        self.condition.wait(timeout=.1)
                        continue
                    handles = tuple(pending)
                handle = self.wait_one(handles)
                if handle is None:
                    continue
                if handle not in pending:
                    raise RuntimeError('wait returned an unknown rollout handle')
                worker, index = pending.pop(handle)
                result = self.resolve(handle)
                with self.condition:
                    group, sample = divmod(index, self.group_size)
                    self.completed.setdefault(group, {})[sample] = result
                    self.resolved += 1
                    idle.append(worker)
                    self.condition.notify_all()
                del result, handle
        except BaseException as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()
        finally:
            # Do not keep completed payloads in the thread's local frame.
            pending.clear()

    def __iter__(self):
        return self

    def __next__(self):
        if self.thread is None:
            raise RuntimeError('rollout prefetch requires its context manager')
        started = time.perf_counter()
        with self.condition:
            try:
                while True:
                    if self.error is not None:
                        raise self.error
                    if self.stopped or self.next_group == len(self.specs) // self.group_size:
                        raise StopIteration
                    group = self.completed.get(self.next_group)
                    if group is not None and len(group) == self.group_size:
                        index = self.next_group
                        result = [group[i] for i in range(self.group_size)]
                        del self.completed[index]
                        self.next_group += 1
                        self.condition.notify_all()
                        start = index * self.group_size
                        return index, self.specs[start:start+self.group_size], result
                    if self.finished:
                        raise RuntimeError('prefetch ended before an ordered group completed')
                    self.condition.wait(timeout=.1)
            finally:
                self.consumer_wait_seconds += time.perf_counter() - started

    def stats(self):
        with self.condition:
            return dict(submitted_episodes=self.submitted, resolved_episodes=self.resolved,
                consumed_groups=self.next_group, max_in_flight_groups=self.window,
                peak_unconsumed_groups=self.peak_unconsumed_groups,
                consumer_wait_seconds=self.consumer_wait_seconds)

    def close(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=5.)
            if self.thread.is_alive():
                raise RuntimeError('rollout prefetch polling did not stop; refuse to advance policy')
        with self.condition:
            self.completed.clear()

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except RuntimeError as cleanup:
            if exc is None:
                raise
            exc.add_note(str(cleanup))
        return False
