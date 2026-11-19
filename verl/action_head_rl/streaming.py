"""Bounded, ordered learner accumulation; independent of Ray and CUDA.

The rollout scheduler still owns the ordered G x K groups. This controller
allows one learner wave and one driver-prepared wave, never an unbounded queue
of Ray calls. Async Ray actors do not promise FIFO execution: the previous
all-rank result must complete before staging or dispatching its successor.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable


class BoundedLearnerStream:
    def __init__(self, *, world: int, expected_waves: int,
                 submit: Callable, resolve: Callable, clock: Callable = time.perf_counter,
                 prepare: Callable | None = None):
        if world <= 0 or expected_waves <= 0:
            raise ValueError("stream world and expected waves must be positive")
        self.world = world
        self.expected_waves = expected_waves
        self.submit_wave = submit
        self.prepare_wave = prepare
        self.resolve = resolve
        self.clock = clock
        self.pending = None
        self.failed = False
        self.submitted = 0
        self.logical_valid = 0.0
        self.rank_record_counts = [0] * world
        self.rank_metrics = []
        self.wait_seconds = 0.0
        self.submission_seconds = 0.0
        self.driver_wave_timings = []
        self.preparation_seconds = 0.0

    def drain(self):
        """Wait for every rank, releasing the sole learner handle afterwards."""
        if self.pending is None:
            return
        handle, self.pending = self.pending, None
        started = self.clock()
        try:
            metrics = self.resolve(handle)
            if not isinstance(metrics, (list, tuple)) or len(metrics) != self.world:
                raise RuntimeError("streamed accumulation did not acknowledge every rank")
            self.rank_metrics.append(metrics)
        except BaseException:
            self.failed = True
            raise
        finally:
            seconds = self.clock() - started
            self.wait_seconds += seconds
            if self.driver_wave_timings:
                self.driver_wave_timings[-1]['result_wait_seconds'] += seconds

    def submit(self, records):
        if self.failed or self.submitted >= self.expected_waves or not records:
            raise RuntimeError("invalid streamed wave submission")
        # CPU-only preparation may overlap the previous all-rank backward.
        # Staging must still wait for its acknowledgement: actors are not FIFO.
        prepared = records
        prepare_started = self.clock()
        try:
            if self.prepare_wave is not None:
                prepared = self.prepare_wave(records)
        except BaseException:
            self.failed = True
            raise
        prepare_seconds = self.clock() - prepare_started
        self.preparation_seconds += prepare_seconds
        self.drain()
        started = self.clock()
        try:
            handle, mass, counts = self.submit_wave(prepared)
            self.pending = handle
            if (not math.isfinite(mass) or mass <= 0 or len(counts) != self.world
                    or len(set(counts)) != 1 or counts[0] <= 0):
                raise RuntimeError("invalid streamed wave mass or rank alignment")
            self.logical_valid += mass
            self.rank_record_counts = [a + b for a, b in zip(self.rank_record_counts, counts)]
            self.submitted += 1
        except BaseException:
            self.failed = True
            raise
        finally:
            seconds = self.clock() - started
            self.submission_seconds += seconds
            self.driver_wave_timings.append(dict(
                submission_seconds=seconds, result_wait_seconds=0.0,
                **({"preparation_seconds": prepare_seconds} if self.prepare_wave is not None else {})))

    def complete(self, *, expected_logical_valid: float | None = None):
        self.drain()
        if self.failed or self.submitted != self.expected_waves:
            raise RuntimeError("incomplete streamed logical batch cannot take an optimizer step")
        if expected_logical_valid is not None and not math.isclose(
                self.logical_valid, expected_logical_valid, rel_tol=1e-10, abs_tol=1e-8):
            self.failed = True
            raise RuntimeError("streamed text mass differs from the original episode denominator")
        return self.rank_metrics


def stage_aligned_wave(rank_records, *, stage_records_per_rpc: int,
                       shared_keys, clear: Callable, stage: Callable,
                       accumulate: Callable, resolve: Callable, preflight: Callable | None = None):
    """Complete all CPU payload staging before any rank enters FSDP.

One staged shard, split into bounded transport RPCs, is retained per learner.
Only scalar stage acknowledgements reach the driver. Resolver errors therefore
cannot release some ranks into a collective while another rank never entered.
"""
    counts = [len(rows) for rows in rank_records]
    if (not counts or len(set(counts)) != 1 or not counts[0]
            or stage_records_per_rpc <= 0):
        raise ValueError("streamed rank records must be nonempty and aligned")
    cleared = resolve(clear())
    if cleared != [0] * len(rank_records):
        raise RuntimeError("streamed staging clear was not acknowledged by every rank")
    previous = [None] * len(rank_records)
    for start in range(0, counts[0], stage_records_per_rpc):
        waves = [[] for _ in rank_records]
        for rank, rows in enumerate(rank_records):
            for source in rows[start:start + stage_records_per_rpc]:
                record = dict(source)
                identity = record.get("condition_id")
                if identity and identity == previous[rank]:
                    for key in shared_keys:
                        record.pop(key, None)
                    record["_reuse_staged_option_payload"] = True
                else:
                    previous[rank] = identity
                waves[rank].append(record)
        staged = resolve(stage(waves))
        expected_count = min(start + stage_records_per_rpc, counts[0])
        if staged != [expected_count] * len(rank_records):
            raise RuntimeError("streamed staging count was not acknowledged by every rank")
    if preflight is not None:
        preflight()
    return accumulate()


def validate_streamed_optimizer_result(results, *, world: int, logical_valid: float):
    """Fail before publishing a partial or nonfinite streamed update."""
    if (not isinstance(results, (list, tuple)) or len(results) != world
            or any(not isinstance(row, dict) for row in results)
            or any(type(row.get("rank")) is not int for row in results)
            or {row.get("rank") for row in results} != set(range(world))):
        raise RuntimeError("streamed optimizer requires all-rank commit reports")
    for row in results:
        if (type(row.get("optimizer_steps")) is not int or row["optimizer_steps"] != 1
                or not math.isfinite(float(row.get("grad_norm", float("nan"))))
                or not math.isclose(float(row.get("logical_batch_valid", -1)), logical_valid,
                                    rel_tol=1e-10, abs_tol=1e-8)):
            raise RuntimeError("streamed optimizer did not commit a finite update with the original denominator")
