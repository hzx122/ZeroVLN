"""Diagnostic-only TP observations for the standalone sleep/weight-sync gate.

The optional bootstrap records actual multiprocessing exit statuses. A separate
probe-only opt-in fixes teardown ownership; it never changes model forwards or
production worker source, and refuses to destroy pools with live allocations.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import time


def install_exit_audit():
    import multiprocessing.popen_fork

    root = Path(os.environ['ACTION_HEAD_SLEEP_EXIT_AUDIT_DIR'])
    root.mkdir(parents=True, exist_ok=True)
    original = multiprocessing.popen_fork.Popen.poll
    if getattr(original, '_action_head_exit_audit', False):
        return
    # Capture the writer functions before interpreter teardown changes globals.
    dumps, open_fd, write_fd, close_fd, rename, getpid = (
        json.dumps, os.open, os.write, os.close, os.replace, os.getpid)
    prefix = str(root)
    monotonic = time.monotonic
    write_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC

    def poll(self, flag=os.WNOHANG):
        code = original(self, flag)
        if code is not None and not getattr(self, '_action_head_exit_reported', False):
            pid = getpid()
            path = f'{prefix}/{pid}_{self.pid}.json'
            temporary = path + '.tmp'
            value = dumps({'observer_pid': pid, 'pid': self.pid, 'returncode': code,
                           'observed_monotonic': monotonic()}).encode()
            try:
                descriptor = open_fd(temporary, write_flags, 0o600)
                try:
                    written = 0
                    while written < len(value):
                        written += write_fd(descriptor, value[written:])
                finally:
                    close_fd(descriptor)
                rename(temporary, path)
                self._action_head_exit_reported = True
            except Exception:
                # Missing evidence fails the parent gate; never alter exit code.
                pass
        return code

    poll._action_head_exit_audit = True
    multiprocessing.popen_fork.Popen.poll = poll
    (root / f'installed_{os.getpid()}.json').write_text(json.dumps({'pid': os.getpid()}))
    if os.environ.get('ACTION_HEAD_SLEEP_PROBE_SAFE_SHUTDOWN', '0') == '1':
        try:
            install_probe_shutdown(root)
        except Exception as error:
            # Python otherwise prints and ignores ordinary sitecustomize errors.
            # A requested candidate must never continue as the unpatched worker.
            raise SystemExit(f'probe shutdown bootstrap failed: {error}') from error


def shutdown_with_minimum_timeout(original, seconds=120.0):
    """Keep the native shutdown implementation, allowing its workers to exit."""
    import functools

    # Fail during bootstrap if this installed API has no compatible timeout.
    inspect.signature(original).bind(None, timeout=seconds)

    @functools.wraps(original)
    def shutdown(self, timeout=None):
        return original(self, timeout=max(seconds, float(timeout or 0.0)))

    shutdown._action_head_probe_shutdown = True
    return shutdown


def release_worker_model(worker, *, cuda, clear_graphs, reset_compiler, observe):
    """Use installed graph/KV cleanup before dropping the complete model runner.

    This is called only from native WorkerProc.shutdown, after its busy loop has
    stopped. It is destructive and deliberately unavailable as a serving RPC.
    """
    import gc
    import weakref

    cuda.synchronize()
    clear_graphs()
    observe({'event': 'graphs_cleared'})
    runner = getattr(worker, 'model_runner', None)
    runner_ref = weakref.ref(runner) if runner is not None else lambda: None
    if runner is not None:
        # Both methods already exist in GPUModelRunner; the profiling cleanup
        # also severs attention-layer KV references in static_forward_context.
        runner.reset_encoder_cache()
        runner._cleanup_profiling_kv_cache()
        runner.compilation_config.static_forward_context.clear()
        worker.model_runner = None
        del runner
    # Compiled callables and query closures may form cycles holding the model.
    # Clear the process-local compiler cache via Torch's existing public reset.
    reset_compiler()
    gc.collect()
    cuda.synchronize()
    cuda.empty_cache()
    observe({'event': 'model_runner_released', 'runner_collected': runner_ref() is None})
    if runner_ref() is not None:
        raise RuntimeError('model runner still has external references after normal cleanup')


def probe_worker_subclass(base, teardown):
    """Replace only this probe process's class alias; preserve native shutdown."""
    class ProbeActionHeadQueryWorker(base):
        _action_head_probe_worker = True

        def shutdown(self):
            if getattr(self, '_action_head_probe_shutdown_done', False):
                return
            # KV transfer/profiler/weight-transfer cleanup precedes model release;
            # WorkerProc still owns live distributed groups until this returns.
            super().shutdown()
            teardown(self)
            self._action_head_probe_shutdown_done = True

    return ProbeActionHeadQueryWorker


def install_probe_shutdown(root):
    """Install the default-off real-TP teardown candidate in the probe bootstrap."""
    from action_head.serve import qwen_query_worker
    from vllm.v1.engine.async_llm import AsyncLLM

    if getattr(AsyncLLM.shutdown, '_action_head_probe_shutdown', False):
        return
    AsyncLLM.shutdown = shutdown_with_minimum_timeout(AsyncLLM.shutdown)
    # The parent's existing *_*.json glob is exclusively actual child exits.
    # Keep teardown diagnostics separate so they cannot masquerade as waitpid.
    shutdown_root = Path(root) / 'shutdown'
    shutdown_root.mkdir(parents=True, exist_ok=True)
    report_path = shutdown_root / f'{os.getpid()}.json'

    def teardown(worker):
        import torch
        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        from vllm.device_allocator.cumem import CuMemAllocator
        from action_head_d16_sleep_probe import release_allocator_pools

        report = {'status': 'running', 'pid': os.getpid(), 'events': []}

        def observe(row):
            report['events'].append(row)
            report_path.write_text(json.dumps(report, indent=2) + '\n')

        try:
            observe({'event': 'native_worker_shutdown_entered'})
            release_worker_model(worker, cuda=torch.cuda,
                clear_graphs=CUDAGraphWrapper.clear_all_graphs,
                reset_compiler=torch._dynamo.reset, observe=observe)
            allocator = CuMemAllocator.instance
            if allocator is not None:
                # Reuse the r2 GPU-validated order and all its live-owner guards.
                # Never erase pointer_to_data or discard pool snapshots.
                report['allocator_teardown'] = release_allocator_pools(
                    allocator, torch.cuda, observe)
            else:
                report['allocator_teardown'] = {'completed': True, 'pool_count': 0,
                                                'remaining_allocations': 0}
            report['status'] = 'completed'
        except BaseException as error:
            report.update(status='error', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            report_path.write_text(json.dumps(report, indent=2) + '\n')

    original = qwen_query_worker.ActionHeadQueryWorker
    candidate = probe_worker_subclass(original, teardown)
    candidate.__module__ = __name__
    candidate.__qualname__ = 'ProbeActionHeadQueryWorker'
    globals()['ProbeActionHeadQueryWorker'] = candidate
    qwen_query_worker.ActionHeadQueryWorker = candidate
    helper_path = Path(__file__).with_name('action_head_d16_sleep_probe.py')
    (shutdown_root / f'patch-{os.getpid()}.json').write_text(json.dumps({
        'pid': os.getpid(), 'api_to_core_minimum_timeout_seconds': 120,
        'worker_class': f'{candidate.__module__}.{candidate.__qualname__}',
        'allocator_helper': str(helper_path),
        'allocator_helper_sha256': hashlib.sha256(helper_path.read_bytes()).hexdigest(),
        'diagnostic_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2) + '\n')


def tensor_sha(tensor):
    import torch
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def tensor_family(model, name, include_hashes):
    iterator = model.named_parameters() if name == 'parameters' else model.named_buffers()
    rows = []
    digest = hashlib.sha256()
    for key, tensor in sorted(iterator):
        row = {'name': key, 'dtype': str(tensor.dtype), 'shape': list(tensor.shape),
               'numel': tensor.numel(), 'pointer': tensor.data_ptr()}
        if include_hashes:
            row['sha256'] = tensor_sha(tensor)
            digest.update(json.dumps({k: v for k, v in row.items() if k != 'pointer'},
                                     sort_keys=True).encode())
        rows.append(row)
    return {'sha256': digest.hexdigest() if include_hashes else None,
            'elements': sum(row['numel'] for row in rows), 'tensors': rows}


def learned_query_state(model, include_hashes):
    state = inspect.getclosurevars(model.embed_input_ids).nonlocals
    learned = state['learned']
    return {'shape': list(learned.shape), 'dtype': str(learned.dtype),
            'pointer': learned.data_ptr(),
            'sha256': tensor_sha(learned) if include_hashes else None,
            'checkpoint_version': list(state['checkpoint_version']),
            'pending_query_phases': len(state['query_phases'])}


class SleepWeightAuditExtension:
    def action_head_sleep_snapshot(self, include_hashes: bool = True):
        import torch
        from vllm.device_allocator.cumem import CuMemAllocator

        candidate_requested = os.environ.get('ACTION_HEAD_SLEEP_PROBE_SAFE_SHUTDOWN', '0') == '1'
        candidate_installed = getattr(self, '_action_head_probe_worker', False)
        if candidate_requested and not candidate_installed:
            raise RuntimeError('requested probe shutdown worker was not installed')
        torch.cuda.synchronize()
        allocator = CuMemAllocator.get_instance()
        model = self.model_runner.get_model()
        by_tag = {}
        for allocation in allocator.pointer_to_data.values():
            by_tag[allocation.tag] = by_tag.get(allocation.tag, 0) + allocation.handle[1]
        free, total = torch.cuda.mem_get_info()
        # Never dereference unmapped sleeping allocations: asleep observations
        # pass include_hashes=False and inspect metadata/pointers only.
        return {
            'pid': os.getpid(), 'parent_pid': os.getppid(),
            'tp_rank': int(self.rank), 'local_rank': int(self.local_rank),
            'probe_safe_shutdown_installed': candidate_installed,
            'worker_class': f'{type(self).__module__}.{type(self).__qualname__}',
            'enable_sleep_mode': self.vllm_config.model_config.enable_sleep_mode,
            'free_bytes': free, 'total_bytes': total,
            'allocated_bytes': torch.cuda.memory_allocated(),
            'reserved_bytes': torch.cuda.memory_reserved(),
            'allocator_bytes': sum(by_tag.values()), 'allocator_bytes_by_tag': by_tag,
            'hashes_included': include_hashes,
            'parameters': tensor_family(model, 'parameters', include_hashes),
            'buffers': tensor_family(model, 'buffers', include_hashes),
            # Hooks are attached after load_model wraps the raw model with
            # CUDAGraphWrapper. get_model() unwraps it and loses this closure.
            'query': learned_query_state(self.model_runner.model, include_hashes),
        }
