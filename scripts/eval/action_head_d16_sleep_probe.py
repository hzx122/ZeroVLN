#!/usr/bin/env python3
"""Small PPU gate for the installed vLLM allocator's sleep/wake mechanism.

Reuses CuMemAllocator directly and the native R3 probe's local-cache policy.
This validates memory release and byte preservation, not a vLLM model service.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import math
import os
import runpy
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time


REPO = Path('/workspace/ZeroVLN')
BASE = Path('/data/zerovln/action_head/rl/'
            'd16_resume_acceptance_20260911/optimization')
LOCAL_RUNTIME_PARENT = Path('/tmp/action_head_rl')


def release_allocator_pools(allocator, cuda, observe=lambda row: None):
    """Release pool objects while every custom allocator is still strongly held.

    The installed vLLM retains (MemPool, CUDAPluggableAllocator) tuples. Python
    may destroy the allocator first when releasing such a tuple, whereas the
    installed Torch MemPool stores a raw CUDAAllocator pointer. Do not leave
    this ordering to interpreter shutdown, or manually discard free callbacks.
    The caller must first drop all tensors and finish outstanding GPU work.
    """
    cuda.synchronize()
    gc.collect()
    cuda.empty_cache()
    metadata = []
    for tag, pair in allocator.allocator_and_pools.items():
        if len(pair) != 2:
            raise ValueError('unexpected installed allocator/pool tuple')
        active_bytes = sum(row['allocated_size'] for row in pair[0].snapshot())
        metadata.append({'tag': tag, 'pool_id': list(pair[0].id),
                         'active_bytes': active_bytes, 'use_count': pair[0].use_count()})
        if active_bytes:
            raise RuntimeError(f'cannot destroy pool {tag} with live tensor allocations')
        if metadata[-1]['use_count'] != 1:
            raise RuntimeError(f'cannot release allocator while pool {tag} has external users')
    # Do not keep the loop's last tuple alive after the dictionary is cleared.
    if allocator.allocator_and_pools:
        del pair
    allocators = [entry[1] for entry in allocator.allocator_and_pools.values()]
    pools = [entry[0] for entry in allocator.allocator_and_pools.values()]
    allocator.allocator_and_pools.clear()
    observe({'event': 'tuples_detached', 'pools': metadata})
    while pools:
        pool = pools.pop()
        pool_id = list(pool.id)
        del pool
        gc.collect()
        cuda.empty_cache()
        observe({'event': 'pool_released', 'pool_id': pool_id,
                 'remaining_allocations': len(allocator.pointer_to_data)})
    cuda.synchronize()
    gc.collect()
    cuda.empty_cache()
    remaining = len(allocator.pointer_to_data)
    if remaining:
        # Keep allocators alive if Torch still owns cached memory. Never erase
        # pointer_to_data: the installed free callback still needs those handles.
        allocator._probe_cleanup_keepalive = allocators
        raise RuntimeError(f'{remaining} allocations remain after pool destruction')
    allocators.clear()
    gc.collect()
    observe({'event': 'allocators_released', 'remaining_allocations': 0})
    return {'completed': True, 'pool_count': len(metadata), 'remaining_allocations': 0,
            'order': 'all MemPools, then all CUDAPluggableAllocators', 'pools': metadata}


def check_owned_runtime(runtime, token, output):
    runtime = Path(runtime)
    if (runtime.parent != LOCAL_RUNTIME_PARENT or not runtime.name.startswith('sleep-')
            or runtime.resolve() != runtime or runtime.stat().st_uid != os.geteuid()):
        raise ValueError('runtime must be this probe\'s owned real sleep-* directory')
    owner = json.loads((runtime / 'owner.json').read_text())
    if owner['token'] != token or owner['output'] != str(output):
        raise ValueError('runtime owner does not match this probe')


def cleanup_owned_runtime(runtime, token, output):
    check_owned_runtime(runtime, token, output)
    shutil.rmtree(runtime)
    return not Path(runtime).exists()


def merge_process_result(worker_report, returncode, runtime_cleaned, cleanup_error=None):
    report = dict(worker_report)
    worker_status = report.get('status', 'missing_report')
    checks_passed = worker_status == 'checks_passed' and report.get('allocator_teardown', {}).get('completed') is True
    if returncode is None:
        status = 'process_not_started'
    elif returncode < 0:
        status = 'process_abort'
    elif returncode != 0:
        status = 'process_error'
    elif not checks_passed:
        status = 'checks_failed'
    else:
        status = 'pass'
    if not runtime_cleaned or cleanup_error:
        status = 'cleanup_error'
    report.update(status=status, worker_status=worker_status, checks_passed=checks_passed,
                  child_returncode=returncode, runtime_cleaned=runtime_cleaned,
                  process_exit_verified=returncode is not None,
                  exit_signal=-returncode if returncode is not None and returncode < 0 else None,
                  shell_equivalent_exit_code=(128 - returncode if returncode is not None and returncode < 0 else returncode))
    if cleanup_error:
        report['cleanup_error'] = cleanup_error
    return report


def supervise_child(command, *, output, runtime, token, environment, timeout_seconds, write_json):
    """Keep the supervisor free of Torch so it can observe native exit crashes."""
    process = None
    worker_report = {}
    supervisor_error = cleanup_error = None
    runtime_cleaned = False
    started = time.monotonic()
    try:
        with (output / 'worker.log').open('x') as log:
            process = subprocess.Popen(command, env=environment, cwd=runtime,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write_json(output / 'supervisor.json', {'pid': os.getpid(), 'child_pid': process.pid,
                'runtime': str(runtime), 'output': str(output), 'status': 'running'})
            process.wait(timeout=timeout_seconds)
    except (Exception, KeyboardInterrupt) as error:
        supervisor_error = f'{type(error).__name__}: {error}'
    finally:
        if process is not None and process.poll() is None:
            # This is a new process group created solely for the owned child.
            try:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        child_path = output / 'worker_report.json'
        if child_path.exists():
            try:
                worker_report = json.loads(child_path.read_text())
            except (ValueError, OSError) as error:
                worker_report = {'status': 'unreadable_report', 'report_error': str(error)}
        try:
            runtime_cleaned = cleanup_owned_runtime(runtime, token, output)
        except Exception as error:
            cleanup_error = f'{type(error).__name__}: {error}'
        report = merge_process_result(worker_report, process.returncode if process else None,
                                      runtime_cleaned, cleanup_error)
        report.update(runtime=str(runtime), output=str(output), supervised_seconds=time.monotonic() - started,
                      worker_log=str(output / 'worker.log'), supervisor_pid=os.getpid(),
                      child_pid=process.pid if process else None)
        if supervisor_error:
            report.update(supervisor_error=supervisor_error,
                          status='supervisor_error' if report['status'] == 'pass' else report['status'])
        write_json(output / 'report.json', report)
    return report


def run_worker(args, output, runtime, write_json):
    # Native aborts must not leave a large core file in the workspace/runtime.
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    check_owned_runtime(runtime, args.owner_token, output)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    report = {'status': 'running', 'gpu': args.gpu, 'runtime': str(runtime),
              'scope': 'allocator only; no model/TP/graph/weight reload/resume validation',
              'cycles': [], 'teardown_events': []}
    allocator = weights = kv_cache = torch = None
    mapping_state = 'awake'

    def save_event(row):
        report['teardown_events'].append(row)
        write_json(output / 'worker_report.json', report)

    write_json(output / 'worker_report.json', report)
    try:
        import torch
        from vllm.device_allocator.cumem import CuMemAllocator, cumem_available
        report['cumem_available'] = cumem_available
        if not cumem_available:
            raise RuntimeError('installed vLLM has no CuMemAllocator')
        allocator = CuMemAllocator.get_instance()
        count = 64 * 1024 * 1024  # 256 MiB each, 512 MiB in the allocator.
        with allocator.use_memory_pool(tag='weights'):
            weights = torch.arange(count, dtype=torch.int32, device='cuda')
        with allocator.use_memory_pool(tag='kv_cache'):
            kv_cache = torch.zeros(count, dtype=torch.int32, device='cuda')
        torch.cuda.synchronize()

        def digest(tensor):
            host = tensor.cpu().numpy()
            return hashlib.sha256(memoryview(host).cast('B')).hexdigest()

        initial_digest = digest(weights)
        pointer = weights.data_ptr()
        report.update(weight_sha256=initial_digest, tensor_bytes=weights.numel() * weights.element_size(),
                      allocator_bytes=sum(row.handle[1] for row in allocator.pointer_to_data.values()))
        for cycle in range(2):
            torch.cuda.synchronize()
            before = torch.cuda.mem_get_info()[0]
            started = time.monotonic()
            mapping_state = 'sleep_transition'
            allocator.sleep(offload_tags=('weights',))
            mapping_state = 'asleep'
            sleep_seconds = time.monotonic() - started
            after_sleep = torch.cuda.mem_get_info()[0]
            started = time.monotonic()
            mapping_state = 'wake_transition'
            allocator.wake_up()
            torch.cuda.synchronize()
            mapping_state = 'awake'
            wake_seconds = time.monotonic() - started
            exact = digest(weights) == initial_digest
            stable_pointer = pointer == weights.data_ptr()
            # KV content is intentionally discarded. Reinitialize before use.
            kv_cache.fill_(cycle + 1)
            torch.cuda.synchronize()
            row = {'cycle': cycle, 'sleep_seconds': sleep_seconds, 'wake_seconds': wake_seconds,
                   'free_before_bytes': before, 'free_asleep_bytes': after_sleep,
                   'released_bytes': after_sleep - before, 'weights_exact': exact,
                   'pointer_stable': stable_pointer}
            report['cycles'].append(row)
            write_json(output / 'worker_report.json', report)
            if not exact or not stable_pointer or after_sleep - before < report['allocator_bytes'] * .9:
                raise RuntimeError('allocator did not preserve weights or release the expected GPU memory')
        report['status'] = 'checks_completed_pending_teardown'
        write_json(output / 'worker_report.json', report)
    except BaseException as error:
        report.update(status='error', error=f'{type(error).__name__}: {error}')
    finally:
        if allocator is not None:
            try:
                if mapping_state == 'asleep':
                    allocator.wake_up()
                    mapping_state = 'awake'
                if mapping_state != 'awake':
                    raise RuntimeError(f'cannot safely tear down partially mapped memory: {mapping_state}')
                weights = kv_cache = None
                torch.cuda.synchronize()
                gc.collect()
                report['allocator_teardown'] = release_allocator_pools(allocator, torch.cuda, save_event)
                if report['status'] == 'checks_completed_pending_teardown':
                    # The supervisor alone may promote this to pass after exit0.
                    report['status'] = 'checks_passed'
            except BaseException as error:
                report.update(status='error', teardown_error=f'{type(error).__name__}: {error}')
        write_json(output / 'worker_report.json', report)
    return 0 if report['status'] == 'checks_passed' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expr-name', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--timeout-seconds', type=float, default=600)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--runtime', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--owner-token', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if Path(args.expr_name).name != args.expr_name or args.expr_name in ('', '.', '..'):
        parser.error('expr-name must be one directory name')
    if not 0 <= args.gpu < 16:
        parser.error('gpu must be a local physical GPU in 0..15')
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0 or args.timeout_seconds > 3600:
        parser.error('timeout-seconds must be in (0, 3600]')
    output = BASE / args.expr_name
    helpers = runpy.run_path(str(REPO / 'scripts/eval/action_head_d16_native_r3_probe.py'))
    if args.worker:
        if args.runtime is None or not args.owner_token or not output.is_dir():
            parser.error('worker requires the supervisor-created runtime and output')
        return run_worker(args, output, args.runtime, helpers['write_json'])
    output.mkdir(exist_ok=False)
    LOCAL_RUNTIME_PARENT.mkdir(parents=True, exist_ok=True)
    runtime = Path(tempfile.mkdtemp(prefix='sleep-', dir=LOCAL_RUNTIME_PARENT))
    token = secrets.token_hex(16)
    helpers['write_json'](runtime / 'owner.json', {'token': token, 'output': str(output), 'pid': os.getpid()})
    environment = dict(os.environ)
    environment.update(helpers['runtime_environment'](runtime))
    environment['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    command = [sys.executable, str(Path(__file__).resolve()), '--expr-name', args.expr_name,
               '--gpu', str(args.gpu), '--worker', '--runtime', str(runtime), '--owner-token', token]
    report = supervise_child(command, output=output, runtime=runtime, token=token,
        environment=environment, timeout_seconds=args.timeout_seconds, write_json=helpers['write_json'])
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
