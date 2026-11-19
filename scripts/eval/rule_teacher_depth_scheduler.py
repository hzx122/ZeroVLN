"""Ready-frontier HTTP batching for independently owned, resumable Gym episodes.

This reuses the bounded ThreadPoolExecutor pattern of
verl/sft/golden_context_beam.py. Unlike that open-loop builder, consuming a
reply here executes the real tools. HTTP sampling uses a separate pool. Optional
actor lanes keep each generator and its rule validation on one fixed thread.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import monotonic

from verl_access.serving_error import ServingGenerationError


def run_depth_cohort(entries, *, max_workers, on_result=None, on_error=None,
                     on_depth=None, refill=None, actor_workers=1,
                     overlap_validation=False, depth_synchronous=False,
                     parallel_initialization=False, stop_on_serving_error=True):
    """Run ``(key, rollout_steps, candidate_adapter)`` entries in ready batches.

    depth_synchronous initializes the entire instruction shard before sending
    HTTP, batches only its minimum unresolved depth, and blocks every next-depth
    request until all initial/retry work and execution at that depth finishes.
    It never rolling-refills instructions or mixes depths.
    Optional parallel_initialization drains the same fixed actor lanes before
    the first depth starts; callbacks retain original input order. Queued
    initializers are cancelled on a serving failure, and running ones drain.

    Adapters expose prepare(*args, **kwargs), next_wave(ctx), sample(ctx, index)
    and resolve(ctx, {index: reply}). A None resolution requests another wave.
    Optional refill(active_keys) supplies new entries only for released actors. All entries
    must own distinct Gym actors; sharing a stateless renderer is supported.
    An accepted state executes before the next batch; its next query joins
    retries of unresolved states, even when their episode depths differ.

    on_depth retains its callback name and start/completed phases for callers.
    Events identify the batch and each case's depth and candidate wave. Wave
    wall time includes sample submission/HTTP/archive/drain and rule
    validation, but excludes preparation and real execution. Mixed batches
    cannot attribute their shared wall time exclusively to retry requests.

    With stop_on_serving_error=False, serving failures are recorded on their
    own episode; peers keep their depth barrier, pending requests and tools.
    The default preserves the legacy behavior described below.
    Serving failures stop the cohort immediately. The failing episode receives
    the original exception and may persist its model_error result; remaining
    episodes are closed at the query boundary, without executing their tools.
    The re-raised exception carries partial_results. In-flight HTTP calls have
    their own transport timeouts and are not retried or waited on here.
    Unhandled episode errors (including reset before the first query) are
    persisted by on_error(key, exception); its return value becomes that case's
    result. Without this callback the exception propagates. A serving error
    always propagates after the callback, even during initial episode setup.
    actor_workers=1 preserves caller-thread execution. Larger values use up to
    eight persistent single-thread lanes; each case's initialization, prepare, resolve,
    generator resume and close use the same lane (including SQLite ownership).
    Scheduler state and callbacks remain on the caller. A whole sampling and
    validation wave must finish without a serving failure before any tool runs.
    overlap_validation starts each case's validation on its owning actor lane
    once all its HTTP replies arrive. Other cases may still be sampling. Tools
    still wait for the entire wave; validation_seconds measures the remaining
    validation tail after HTTP drain when overlap is enabled.
    """
    # V8 collection records a failed episode and lets its peers continue.
    # Older callers retain their original fail-fast behavior.
    entries = list(entries)
    if not isinstance(depth_synchronous, bool):
        raise ValueError("depth_synchronous must be boolean")
    if depth_synchronous and refill is not None:
        raise ValueError("depth-synchronous collection initializes all instructions; rolling refill is forbidden")
    if int(max_workers) < 1:
        raise ValueError("max_workers must be positive")
    if isinstance(actor_workers, bool) or not isinstance(actor_workers, int) or not 1 <= actor_workers <= 8:
        raise ValueError("actor_workers must be an integer in 1..8")
    if not isinstance(overlap_validation, bool) or (overlap_validation and actor_workers == 1):
        raise ValueError("overlap_validation requires at least two actor lanes")
    if len({key for key, _, _ in entries}) != len(entries):
        raise ValueError("cohort case keys must be unique")
    if len({id(steps) for _, steps, _ in entries}) != len(entries):
        raise ValueError("cohort rollout generators must be distinct")
    if not isinstance(parallel_initialization, bool):
        raise ValueError("parallel_initialization must be boolean")
    if parallel_initialization and not depth_synchronous:
        raise ValueError("parallel initialization requires the complete depth-synchronous cohort")
    results, active, depths = {}, {}, {}
    lanes = ([ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"rule-actor-{i}")
              for i in range(actor_workers)] if actor_workers > 1 else [])
    case_lanes = {}

    def actor_call(key, function, *args, **kwargs):
        if not lanes:
            return function(*args, **kwargs)
        lane = case_lanes.setdefault(key, len(case_lanes) % len(lanes))
        return lanes[lane].submit(function, *args, **kwargs).result()

    def actor_batch(calls):
        """Drain actor work before committing callbacks or closing generators.

        Each call is unique per case. Serving exceptions take priority over
        ordinary failures, and queued work is cancelled, but already running
        actor operations must finish before their generators can be closed.
        No HTTP sampling futures pass through this helper.
        """
        pending = {}
        for key, function, args, kwargs in calls:
            lane = case_lanes[key]
            pending[lanes[lane].submit(function, *args, **kwargs)] = key
        return actor_drain(pending)

    def actor_drain(pending):
        """Reuse the same failure priority and owned-thread drain for queued work."""
        outcomes = {}
        fatal_error = None
        for future in as_completed(pending):
            if future.cancelled():
                continue
            key = pending[future]
            try:
                outcomes[key] = (True, future.result())
            except Exception as exc:
                outcomes[key] = (False, exc)
                if stop_on_serving_error and isinstance(exc, ServingGenerationError) and fatal_error is None:
                    fatal_error = key, exc
                    for queued in pending:
                        queued.cancel()
        return outcomes, fatal_error

    def resume(steps, *, reply=None, error=None, initial=False):
        try:
            query = (next(steps) if initial else
                     steps.throw(error) if error is not None else steps.send(reply))
        except StopIteration as done:
            return "done", done.value
        return "query", query

    def commit_error(key, exc):
        active.pop(key, None)
        if stop_on_serving_error and isinstance(exc, ServingGenerationError):
            try:
                if on_error is not None:
                    results[key] = on_error(key, exc)
            except Exception as finalization_error:
                exc.finalization_error = type(finalization_error).__name__
            exc.partial_results = dict(results)
            raise exc
        if on_error is None:
            raise exc
        results[key] = on_error(key, exc)

    def commit_advance(key, steps, adapter, outcome, *, initial=False):
        kind, value = outcome
        if kind == "done":
            results[key] = value
            active.pop(key, None)
            if on_result is not None:
                on_result(key, value)
        else:
            active[key] = (steps, adapter, value)
            depths[key] = 1 if initial else depths[key] + 1

    def advance(key, steps, adapter, *, reply=None, error=None, initial=False):
        try:
            outcome = actor_call(key, resume, steps, reply=reply, error=error, initial=initial)
        except Exception as exc:
            commit_error(key, exc)
        else:
            commit_advance(key, steps, adapter, outcome, initial=initial)

    def fatal(key, exc):
        steps, adapter, _ = active[key]
        try:
            advance(key, steps, adapter, error=exc)
        except Exception as finalization_error:
            # Preserve the typed serving failure if scoring/archiving fails.
            if finalization_error is not exc:
                exc.finalization_error = type(finalization_error).__name__
        exc.partial_results = dict(results)
        raise exc

    try:
        if parallel_initialization and lanes:
            # Reuse the fixed actor lanes. Cases sharing a lane initialize in
            # FIFO order; all lanes drain before callbacks or depth-1 HTTP.
            for key, _, _ in entries:
                case_lanes.setdefault(key, len(case_lanes) % len(lanes))
            initial_outcomes, initial_fatal = actor_batch([
                (key, resume, (steps,), {"initial": True})
                for key, steps, _ in entries])
            if initial_fatal is not None:
                # Running initializers have drained; queued work was cancelled.
                # Persist the original typed serving failure before any sample.
                commit_error(*initial_fatal)
            for key, steps, adapter in entries:
                succeeded, value = initial_outcomes[key]
                if succeeded:
                    commit_advance(key, steps, adapter, value, initial=True)
                else:
                    commit_error(key, value)
        else:
            for key, steps, adapter in entries:
                advance(key, steps, adapter, initial=True)
        batch_index = 0
        contexts, wave_numbers = {}, {}
        seen = {key for key, _, _ in entries}
        while True:
            added = False
            if refill is not None:
                for key, steps, adapter in refill(set(active)):
                    if key in seen or any(steps is old for _, old, _ in entries):
                        raise ValueError("refill must provide a new case and generator")
                    added = True
                    seen.add(key)
                    entries.append((key, steps, adapter))
                    advance(key, steps, adapter, initial=True)
            if not active:
                if added:
                    continue
                break
            batch_index += 1
            batch_started = monotonic()
            frontier = dict(active)
            if depth_synchronous:
                current_depth = min(depths[key] for key in active)
                frontier = {key: value for key, value in active.items()
                            if depths[key] == current_depth}
            case_depths = {key: depths[key] for key in frontier}
            depth_distribution = {str(depth): list(case_depths.values()).count(depth)
                                  for depth in sorted(set(case_depths.values()))}
            metadata = dict(scheduler=("v6_same_depth_batch" if depth_synchronous else "mixed_depth_ready_batch"), batch_index=batch_index,
                            actor_workers=actor_workers,
                            depth=(next(iter(case_depths.values()))
                                   if len(depth_distribution) == 1 else None),
                            case_depths=case_depths, depth_distribution=depth_distribution,
                            active_cases=list(frontier))
            if on_depth is not None:
                on_depth(dict(metadata, phase="start"))
            resolved, failures = {}, {}
            if lanes:
                preparation_outcomes, preparation_fatal = actor_batch([
                    (key, adapter.prepare, query.args, query.kwargs)
                    for key, (_, adapter, query) in frontier.items() if key not in contexts])
                if preparation_fatal is not None:
                    fatal(*preparation_fatal)
            for key, (steps, adapter, query) in frontier.items():
                if key in contexts:
                    continue
                try:
                    if lanes:
                        ok, context = preparation_outcomes[key]
                        if not ok:
                            raise context
                    else:
                        context = adapter.prepare(*query.args, **query.kwargs)
                    contexts[key] = context
                    wave_numbers[key] = 0
                except ServingGenerationError as exc:
                    if stop_on_serving_error:
                        fatal(key, exc)
                    failures[key] = exc
                except Exception as exc:
                    failures[key] = exc
            prepare_seconds = monotonic() - batch_started
            jobs = []
            case_waves = {}
            for key, ctx in contexts.items():
                if key not in frontier:
                    raise RuntimeError("candidate context crossed the depth barrier")
                indices = list(frontier[key][1].next_wave(ctx))
                if not indices:
                    raise RuntimeError("unresolved candidate context has no next wave")
                if len(set(indices)) != len(indices):
                    raise ValueError("candidate indices within a wave must be unique")
                wave_numbers[key] += 1
                case_waves[key] = wave_numbers[key]
                jobs.extend((key, ctx, index) for index in indices)
            new_requests = sum(case_waves[key] == 1 for key, _, _ in jobs)
            retry_requests = len(jobs) - new_requests
            request_depth_distribution = {
                str(depth): sum(depths[key] == depth for key, _, _ in jobs)
                for depth in sorted({depths[key] for key, _, _ in jobs})}
            batch_kind = ("mixed" if new_requests and retry_requests else
                          "retry" if retry_requests else "new" if jobs else "empty")
            wave_metadata = dict(metadata, case_waves=case_waves,
                                 overlap_validation=overlap_validation,
                                 request_count=len(jobs), client_batch_requests=len(jobs),
                                 batch_size_scope="client_submission_not_gpu_execution",
                                 http_concurrency=min(int(max_workers), len(jobs)),
                                 sample_case_count=len(case_waves), new_requests=new_requests,
                                 retry_requests=retry_requests, batch_kind=batch_kind,
                                 request_depth_distribution=request_depth_distribution)
            wave_seconds = sample_seconds = validation_seconds = 0.
            if jobs:
                wave_started = monotonic()
                if on_depth is not None:
                    on_depth(dict(wave_metadata, phase="wave_start"))
                pool = ThreadPoolExecutor(max_workers=min(int(max_workers), len(jobs)))
                futures = {}
                replies = {key: {} for key in contexts}
                sample_errors = {}
                validation_pending = {}
                remaining = {key: sum(job_key == key for job_key, _, _ in jobs)
                             for key in contexts}
                failed = False
                try:
                    for key, ctx, index in jobs:
                        future = pool.submit(frontier[key][1].sample, ctx, index)
                        futures[future] = (key, index)
                    for future in as_completed(futures):
                        key, index = futures[future]
                        try:
                            replies[key][index] = future.result()
                        except ServingGenerationError as exc:
                            if not stop_on_serving_error:
                                sample_errors[key] = exc
                                remaining[key] -= 1
                                continue
                            failed = True
                            for queued in futures:
                                queued.cancel()
                            # Do not let the pool context manager drain requests
                            # before the caller/supervisor can see this failure.
                            pool.shutdown(wait=False, cancel_futures=True)
                            # A running validator owns actor state. Drain it
                            # before throwing into or closing that generator.
                            for queued in validation_pending:
                                queued.cancel()
                            actor_drain(validation_pending)
                            fatal(key, exc)
                        except Exception as exc:
                            sample_errors.setdefault(key, exc)
                        remaining[key] -= 1
                        if overlap_validation and remaining[key] == 0 and key not in sample_errors:
                            future = lanes[case_lanes[key]].submit(
                                frontier[key][1].resolve, contexts[key], dict(replies[key]))
                            validation_pending[future] = key
                finally:
                    if not failed:
                        pool.shutdown(wait=True)
                sample_finished = monotonic()
                sample_seconds = sample_finished - wave_started
                if lanes:
                    validation_outcomes, validation_fatal = (actor_drain(validation_pending)
                        if overlap_validation else actor_batch([
                        (key, frontier[key][1].resolve, (ctx, replies[key]), {})
                        for key, ctx in contexts.items() if key not in sample_errors]))
                    if validation_fatal is not None:
                        fatal(*validation_fatal)
                for key, ctx in contexts.items():
                    if key in sample_errors:
                        failures[key] = sample_errors[key]
                        continue
                    try:
                        if lanes:
                            ok, reply = validation_outcomes[key]
                            if not ok:
                                raise reply
                        else:
                            reply = frontier[key][1].resolve(ctx, replies[key])
                    except ServingGenerationError as exc:
                        if stop_on_serving_error:
                            fatal(key, exc)
                        failures[key] = exc
                    except Exception as exc:
                        failures[key] = exc
                    else:
                        if reply is not None:
                            resolved[key] = reply
                wave_finished = monotonic()
                validation_seconds = wave_finished - sample_finished
                wave_seconds = wave_finished - wave_started
                if on_depth is not None:
                    on_depth(dict(wave_metadata, phase="wave_completed",
                                  elapsed_seconds=wave_seconds, sample_seconds=sample_seconds,
                                  validation_seconds=validation_seconds))
            # All HTTP in this batch has finished before any real tool executes.
            # Only resolved cases advance; retries retain their exact context.
            execution_started = monotonic()
            if lanes:
                execution_outcomes, execution_fatal = actor_batch([
                    (key, resume, (steps,), dict(reply=resolved.get(key), error=failures.get(key)))
                    for key, (steps, _, _) in frontier.items() if key in resolved or key in failures])
            for key, (steps, adapter, _) in frontier.items():
                if key not in resolved and key not in failures:
                    continue
                contexts.pop(key, None)
                wave_numbers.pop(key, None)
                if lanes:
                    if key not in execution_outcomes:
                        continue  # Cancelled after a different actor failed.
                    ok, outcome = execution_outcomes[key]
                    # Commit successful peers before propagating a fatal actor
                    # error, so their already executed prefix stays archived.
                    if execution_fatal is not None and key == execution_fatal[0]:
                        continue
                    try:
                        if ok:
                            commit_advance(key, steps, adapter, outcome)
                        else:
                            commit_error(key, outcome)
                    except Exception as finalization_error:
                        if execution_fatal is None:
                            raise
                        execution_fatal[1].finalization_error = type(finalization_error).__name__
                else:
                    advance(key, steps, adapter, reply=resolved.get(key),
                            error=failures.get(key))
            if lanes and execution_fatal is not None:
                key, exc = execution_fatal
                commit_error(key, exc)
            batch_finished = monotonic()
            if on_depth is not None:
                on_depth(dict(wave_metadata, phase="completed", waves=int(bool(jobs)),
                              requests=len(jobs), remaining_cases=list(active),
                              elapsed_seconds=batch_finished - batch_started,
                              prepare_seconds=prepare_seconds, wave_seconds=wave_seconds,
                              sample_seconds=sample_seconds, validation_seconds=validation_seconds,
                              execution_seconds=batch_finished - execution_started))
        return results
    finally:
        try:
            for key, steps, _ in entries:
                actor_call(key, steps.close)
        finally:
            for lane in lanes:
                lane.shutdown(wait=True, cancel_futures=True)
