"""Resident CPU compilation using the original native map and compiler.

Requests contain the actual numeric pose, original program and independent
geometry. They never contain an instruction, reference route or episode goal.
"""
from collections import OrderedDict, deque
from concurrent.futures import Future, ProcessPoolExecutor
import atexit
import hashlib
import json
import multiprocessing
import os
import pickle
from pathlib import Path
import sys
from threading import Lock
import time
import uuid

import numpy as np

_MAPS = OrderedDict()
_MAP_BYTES = 0
MAX_MAPS = 8
MAX_MAP_BYTES = 256 * 1024 * 1024
MAX_OUTCOMES = 4096
MAX_OUTCOME_BYTES = 64 * 1024 * 1024
_POOL = None
_POOL_LOCK = Lock()


class NativeCompileInfrastructureError(RuntimeError):
    pass


def array_identity(value):
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), dtype=value.dtype.str,
                sha256=hashlib.sha256(value.tobytes()).hexdigest())


def map_descriptor(occ, pose):
    from mp3d_rule_teacher import MP3DNavmeshMap
    if type(occ) is not MP3DNavmeshMap or not getattr(occ, '_native_managed_pathfinder', False):
        raise NativeCompileInfrastructureError('only production managed native maps can be offloaded')
    if not hasattr(occ, '_compile_map_token'):
        occ._compile_map_token = uuid.uuid4().hex
    return dict(token=occ._compile_map_token, navmesh=occ.navmesh_path,
        navmesh_sha256=hashlib.sha256(Path(occ.navmesh_path).read_bytes()).hexdigest(),
        construction_start=list(occ._native_construction_start),
        floor_height=occ.floor_height.hex(), mpp=occ.mpp.hex(),
        floor_tolerance=occ.floor_tolerance.hex(), world_frame=occ.world_frame,
        lower=list(occ.lower), upper=list(occ.upper),
        free=array_identity(occ.free), clearance=array_identity(occ.clearance_m))


def _initialize_worker(runtime):
    # Reuse the pipeline's complete runtime cache configuration.
    source = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(source/'vln_demo/src'), str(source/'verl'),
                    str(source/'scripts/train'), str(source)]
    from run_rule_teacher_local_pipeline import runtime_environment
    env = runtime_environment(Path(runtime), f'cpu-worker-{os.getpid()}',
        dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
    os.environ.update(env)
    import mp3d_rule_teacher
    import privileged_teacher
    from sage import planner
    for module, relative in ((mp3d_rule_teacher, 'vln_demo/src/mp3d_rule_teacher.py'),
            (privileged_teacher, 'vln_demo/src/privileged_teacher.py'),
            (planner, 'vln_demo/src/sage/planner.py')):
        if Path(module.__file__).resolve() != source/relative:
            raise NativeCompileInfrastructureError('worker imported code from a different source')


def _native_map(descriptor):
    global _MAP_BYTES
    from mp3d_rule_teacher import MP3DNavmeshMap
    key = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
    navmesh = Path(descriptor['navmesh'])
    if hashlib.sha256(navmesh.read_bytes()).hexdigest() != descriptor['navmesh_sha256']:
        raise NativeCompileInfrastructureError('native worker navmesh content changed')
    if key in _MAPS:
        occ, size = _MAPS.pop(key)
        _MAPS[key] = occ, size
        return occ, True
    occ = MP3DNavmeshMap(navmesh, descriptor['construction_start'],
        mpp=float.fromhex(descriptor['mpp']),
        floor_tolerance=float.fromhex(descriptor['floor_tolerance']),
        world_frame=descriptor['world_frame'], cache_floor_slice=False)
    if (occ.floor_height.hex() != descriptor['floor_height']
            or list(occ.lower) != descriptor['lower'] or list(occ.upper) != descriptor['upper']
            or array_identity(occ.free) != descriptor['free']
            or array_identity(occ.clearance_m) != descriptor['clearance']
            or hashlib.sha256(navmesh.read_bytes()).hexdigest() != descriptor['navmesh_sha256']):
        raise NativeCompileInfrastructureError('native worker reconstructed different geometry')
    size = occ.free.nbytes + occ.clearance_m.nbytes + occ.occ.nbytes
    while _MAPS and (len(_MAPS) >= MAX_MAPS or _MAP_BYTES + size > MAX_MAP_BYTES):
        _, (_, removed) = _MAPS.popitem(last=False)
        _MAP_BYTES -= removed
        # The original class-level lru_cache holds strong self references.
        MP3DNavmeshMap.snap_xy.cache_clear()
        MP3DNavmeshMap.native_clearance.cache_clear()
    if size <= MAX_MAP_BYTES:
        _MAPS[key] = occ, size
        _MAP_BYTES += size
    return occ, False


def _compile_request(descriptor, program, pose, objects, openings):
    from privileged_teacher import PrivilegedTeacher, TeacherRejected
    started = time.monotonic()
    occ, hit = _native_map(descriptor)
    try:
        outcome = dict(state=PrivilegedTeacher.__new__(PrivilegedTeacher).compile(
            program, pose, occ, objects, openings))
    except TeacherRejected as error:
        outcome = dict(rejected=str(error))
    except (ValueError, KeyError) as error:
        outcome = dict(compiler_error=dict(kind=type(error).__name__, args=error.args))
    finally:
        if occ.free.nbytes + occ.clearance_m.nbytes + occ.occ.nbytes > MAX_MAP_BYTES:
            type(occ).snap_xy.cache_clear()
            type(occ).native_clearance.cache_clear()
    return dict(outcome=outcome, pid=os.getpid(), map_cache_hit=hit,
        seconds=time.monotonic()-started, map_token=descriptor['token'])


def _prepare_stair_request(descriptor, object_id, obj, actual_start, motion,
                           component, pose, minimum_chunks, exact_chunks, prefix_steps=4):
    from native_stair_route import prepare_stair_state
    from privileged_teacher import TeacherRejected
    started = time.monotonic()
    occ, hit = _native_map(descriptor)
    audit = []
    try:
        state = prepare_stair_state(occ, object_id, obj, actual_start, motion,
            component, pose, minimum_chunks, exact_chunks, diagnostics=audit, prefix_steps=prefix_steps)
        outcome = dict(state=(state, audit))
    except TeacherRejected as error:
        outcome = dict(rejected=str(error), stair_audit=audit)
    except (ValueError, KeyError) as error:
        outcome = dict(compiler_error=dict(kind=type(error).__name__, args=error.args))
    finally:
        if occ.free.nbytes + occ.clearance_m.nbytes + occ.occ.nbytes > MAX_MAP_BYTES:
            type(occ).snap_xy.cache_clear()
            type(occ).native_clearance.cache_clear()
    return dict(outcome=outcome, pid=os.getpid(), map_cache_hit=hit,
        seconds=time.monotonic()-started, map_token=descriptor['token'])


class NativeCompilePool:
    def __init__(self, workers, runtime, *, cache_outcomes=False, coalesce_inflight=True):
        runtime = Path(runtime)
        if (not runtime.is_absolute() or not any(runtime.is_relative_to(root)
                and runtime != Path(root) for root in ('/tmp', '/dev/shm'))):
            raise ValueError('native workers require a dedicated container-local runtime')
        if not 1 <= workers <= 8:
            raise ValueError('native worker count must be between one and eight')
        self.executor = ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=_initialize_worker, initargs=(str(runtime),))
        self.metrics = deque(maxlen=256)
        self.closed = False
        self.cache_outcomes = cache_outcomes
        self.coalesce_inflight = coalesce_inflight
        self.outcomes = OrderedDict()
        self.outcome_bytes = 0
        self.outcome_lock = Lock()
        self.inflight = {}
        self.request_totals = dict(requests=0, worker_submissions=0,
            result_cache_hits=0, inflight_hits=0, worker_seconds=0.,
            result_cache_seconds=0., inflight_wait_seconds=0.)
        self.worker_pids = set()

    def snapshot_metrics(self):
        with self.outcome_lock:
            return dict(self.request_totals, inflight_requests=len(getattr(self, 'inflight', {})),
                outcome_cache_entries=len(self.outcomes),
                outcome_cache_bytes=self.outcome_bytes,
                outcome_cache_max_entries=MAX_OUTCOMES,
                outcome_cache_max_bytes=MAX_OUTCOME_BYTES,
                result_cache_enabled=self.cache_outcomes,
                worker_pids=sorted(self.worker_pids))

    def _cached_outcome(self, key):
        with self.outcome_lock:
            encoded = self.outcomes.get(key)
            if encoded is not None:
                self.outcomes.move_to_end(key)
            # Decoding makes returned paths/programs independent of the cache.
            return pickle.loads(encoded) if encoded is not None else None

    def _remember_outcome(self, key, outcome):
        encoded = pickle.dumps(outcome, protocol=5)
        size = len(encoded) + len(key)
        if size > MAX_OUTCOME_BYTES:
            return
        with self.outcome_lock:
            if self.closed:
                return
            previous = self.outcomes.pop(key, None)
            if previous is not None:
                self.outcome_bytes -= len(previous) + len(key)
            self.outcomes[key] = encoded
            self.outcome_bytes += size
            while (len(self.outcomes) > MAX_OUTCOMES or
                   self.outcome_bytes > MAX_OUTCOME_BYTES):
                old_key, discarded = self.outcomes.popitem(last=False)
                self.outcome_bytes -= len(old_key) + len(discarded)

    def compile(self, program, pose, occ, objects, openings):
        if self.closed:
            raise NativeCompileInfrastructureError('native worker pool is closed')
        try:
            descriptor = map_descriptor(occ, pose)
            object_ids = {s.get('object') for s in program.get('steps', []) if 'object' in s}
            opening_ids = {s.get('opening') for s in program.get('steps', []) if 'opening' in s}
            args = (descriptor, program, pose,
                {k:objects[k] for k in object_ids if k in objects},
                {k:openings[k] for k in opening_ids if k in openings})
        except Exception as error:
            raise NativeCompileInfrastructureError(str(error)) from error
        return self._execute_request(_compile_request, args)

    def prepare_stair(self, occ, object_id, obj, actual_start, motion, component,
                      pose, minimum_chunks, exact_chunks, *, prefix_steps=4):
        if self.closed:
            raise NativeCompileInfrastructureError('native worker pool is closed')
        try:
            descriptor = map_descriptor(occ, pose)
            # Language parsing/grounding remains per original call in the
            # parent. The physical stair compiler reads only direction; clause
            # wording and plan position do not change its geometry. Keep every
            # physical argument byte-exact for cached/inflight reuse.
            from native_stair_route import stair_search_policy
            policy = stair_search_policy()
            # Bind even the default policy explicitly: a resident worker can
            # retain an older environment after the parent changes its policy.
            geometry_motion = dict(direction=motion['direction'], search_policy=policy)
            args = (descriptor, object_id, obj, actual_start, geometry_motion, component,
                    pose, minimum_chunks, exact_chunks, prefix_steps)
        except Exception as error:
            raise NativeCompileInfrastructureError(str(error)) from error
        return self._execute_request(_prepare_stair_request, args)

    def _execute_request(self, worker, args):
        if self.closed:
            raise NativeCompileInfrastructureError('native worker pool is closed')
        try:
            started = time.monotonic()
            descriptor = args[0]
            key = None
            if getattr(self, 'cache_outcomes', False):
                # Exact operation and complete original input bytes bind reuse.
                encoded = pickle.dumps(args, protocol=5)
                key = hashlib.sha256(worker.__name__.encode()+b'\0'+encoded).digest()
                args = pickle.loads(encoded)
            result = self._resident_result(worker, args, key, started)
        except Exception as error:
            raise NativeCompileInfrastructureError(str(error)) from error
        self.metrics.append({k:v for k,v in result.items() if k != 'outcome'})
        if hasattr(self, 'request_totals'):
            with self.outcome_lock:
                self.request_totals['requests'] += 1
                if result['result_cache_hit']:
                    self.request_totals['result_cache_hits'] += 1
                    self.request_totals['result_cache_seconds'] += result['seconds']
                elif result.get('inflight_hit', False):
                    self.request_totals.setdefault('inflight_hits', 0)
                    self.request_totals.setdefault('inflight_wait_seconds', 0.)
                    self.request_totals['inflight_hits'] += 1
                    self.request_totals['inflight_wait_seconds'] += result['seconds']
                else:
                    self.request_totals['worker_submissions'] += 1
                    self.request_totals['worker_seconds'] += result['seconds']
                    self.worker_pids.add(result['pid'])
        outcome = result['outcome']
        if 'rejected' in outcome:
            from privileged_teacher import TeacherRejected
            error = TeacherRejected(outcome['rejected'])
            if 'stair_audit' in outcome:
                error.native_stair_audit = outcome['stair_audit']
            raise error
        if 'compiler_error' in outcome:
            error = outcome['compiler_error']
            raise {'ValueError':ValueError, 'KeyError':KeyError}[error['kind']](*error['args'])
        return outcome['state']

    def _resident_result(self, worker, args, key, started):
        if key is None:
            result = self.executor.submit(worker, *args).result()
            result['result_cache_hit'] = False
            return result
        if not getattr(self, 'coalesce_inflight', True):
            outcome = self._cached_outcome(key)
            if outcome is not None:
                return dict(outcome=outcome, pid=None, map_cache_hit=None,
                    result_cache_hit=True, seconds=time.monotonic()-started,
                    map_token=args[0]['token'])
            result = self.executor.submit(worker, *args).result()
            self._remember_outcome(key, result['outcome'])
            result['result_cache_hit'] = False
            return result
        # Lookup and publish an immutable result future atomically. Never wait
        # on a future while holding this lock or the native parser lock.
        with self.outcome_lock:
            encoded = self.outcomes.get(key)
            if encoded is not None:
                self.outcomes.move_to_end(key)
                return dict(outcome=pickle.loads(encoded), pid=None,
                    map_cache_hit=None, result_cache_hit=True, inflight_hit=False,
                    seconds=time.monotonic()-started, map_token=args[0]['token'])
            if not hasattr(self, 'inflight'):
                self.inflight = {}
            future = self.inflight.get(key)
            owner = future is None
            if owner:
                future = Future()
                self.inflight[key] = future
        if owner:
            try:
                result = self.executor.submit(worker, *args).result()
                payload = pickle.dumps(result, protocol=5)
                self._remember_outcome(key, result['outcome'])
            except BaseException as error:
                # Wake all identical callers with the original infrastructure
                # failure; never retain it as a compiler rejection or retry it.
                future.set_exception(error)
                raise
            else:
                future.set_result(payload)
            finally:
                with self.outcome_lock:
                    if self.inflight.get(key) is future:
                        del self.inflight[key]
        else:
            payload = future.result()
        # Each caller, including the owner, receives an independent state/audit.
        result = pickle.loads(payload)
        result['result_cache_hit'] = False
        result['inflight_hit'] = not owner
        if not owner:
            result['pid'] = None
            result['map_cache_hit'] = None
            result['seconds'] = time.monotonic()-started
        return result

    def close(self):
        if not self.closed:
            self.closed = True
            self.executor.shutdown(wait=True, cancel_futures=True)
            if hasattr(self, 'outcome_lock'):
                with self.outcome_lock:
                    self.outcomes.clear()
                    self.outcome_bytes = 0


def get_compile_pool():
    global _POOL
    try:
        workers = int(os.environ.get('RULE_NATIVE_COMPILE_WORKERS', '0'))
    except ValueError as error:
        raise NativeCompileInfrastructureError('invalid native worker count') from error
    if workers == 0:
        return None
    with _POOL_LOCK:
        if _POOL is None:
            runtime = os.environ.get('RULE_COLLECTION_RUNTIME_ROOT')
            if not runtime:
                raise NativeCompileInfrastructureError('native worker runtime is missing')
            try:
                cache = os.environ.get('RULE_NATIVE_COMPILE_RESULT_CACHE', '0')
                if cache not in ('0', '1'):
                    raise ValueError('native result cache must be zero or one')
                _POOL = NativeCompilePool(workers, Path(runtime)/'native-compile',
                    cache_outcomes=cache == '1')
            except (ValueError, OSError) as error:
                raise NativeCompileInfrastructureError(str(error)) from error
            atexit.register(_POOL.close)
        return _POOL
