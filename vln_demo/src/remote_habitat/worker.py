"""One GL worker behind the service, so native mesh loads cannot block health."""
import multiprocessing as mp
from pathlib import Path
from collections import OrderedDict
import json
import os


def scene_key(root, request):
    scene = (root / request['scene']).resolve(strict=True)
    if not scene.is_relative_to(root) or scene.suffix != '.glb':
        raise ValueError('scene must be an MP3D glb below scenes_root')
    return str(scene), json.dumps(request['sensors'], sort_keys=True)


def run(connection, root, cache_size):
    from .server import Renderer
    renderer = Renderer(root,cache_size)
    try:
        while True:
            request = connection.recv()
            if request is None:
                break
            try:
                observations = renderer.render(request)
                connection.send((True,observations,renderer.health()))
            except Exception as exc:
                connection.send((False,f'{type(exc).__name__}: {exc}',renderer.health()))
    except EOFError:
        pass
    finally:
        renderer.close()
        connection.close()


class SingleWorker:
    def __init__(self, root, cache_size, timeout=None):
        self.root = Path(root).resolve(strict=True)
        self.cache_size, self.timeout = cache_size, timeout
        self.process = self.connection = None

        self.status = dict(ok=True,protocol=1,stateless=True,service_instances=1,gl_workers=1,
                           cache_capacity=cache_size,cached_scenes=[],requests=0,scene_loads=0,evictions=0)
        self.loading_scene = None

    def render(self, request):
        if self.process is None:
            context = mp.get_context('spawn')
            self.connection,child = context.Pipe()
            self.process = context.Process(target=run,args=(child,str(self.root),self.cache_size),daemon=True)
            self.process.start()
            child.close()
        self.loading_scene = request.get('scene')
        try:
            self.connection.send(request)
            if not self.connection.poll(self.timeout):
                raise TimeoutError('GL worker timed out')
            ok,result,status = self.connection.recv()
            self.status = status
            if not ok:
                raise ValueError(result)
            return result
        except (OSError,EOFError,TimeoutError):
            self.close()
            raise
        finally:
            self.loading_scene = None

    def health(self):
        process = self.process
        return dict(self.status,ok=process is None or process.is_alive(),
                    active_scene=self.loading_scene,gl_workers=1,
                    worker_pid=process.pid if process else None)

    def resident_bytes(self):
        """Conservative RSS charge for this owned child, including shared pages."""
        process = self.process
        if process is None or not process.is_alive():
            return 0
        try:
            return int(Path(f'/proc/{process.pid}/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        except (FileNotFoundError, ProcessLookupError):
            return 0

    def peak_resident_bytes(self):
        """Kernel high-water RSS includes transient native scene initialization."""
        process = self.process
        if process is None or not process.is_alive():
            return 0
        try:
            for line in Path(f'/proc/{process.pid}/status').read_text().splitlines():
                if line.startswith('VmHWM:'):
                    return int(line.split()[1]) * 1024
        except (FileNotFoundError, ProcessLookupError):
            return 0
        raise RuntimeError('owned GL worker has no kernel memory high-water mark')

    def close(self):
        if self.process is None:
            return
        if self.process.is_alive():
            try:
                self.connection.send(None)
            except OSError:
                pass
            self.process.join(5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(5)
            if self.process.is_alive():
                raise RuntimeError('owned GL worker did not exit after kill')
        self.connection.close()
        self.process = self.connection = None


class SceneWorkers:
    """Bounded persistent cache with one native Simulator in each GL process.

    Reuse SingleWorker's IPC and failure recovery. Never reload a different
    scene/camera in a process with a resident Simulator: native texture/context
    teardown otherwise corrupts another cached instance in this Habitat build.
    The HTTP server still serializes rendering on its original single thread.
    """
    def __init__(self, root, cache_size, timeout=None):
        self.root = Path(root).resolve(strict=True)
        if not 1 <= cache_size <= 512:
            raise ValueError('scene cache size must be 1..512')
        self.cache_size, self.timeout = cache_size, timeout
        self.workers = OrderedDict()
        self.requests = self.loads = self.evictions = 0
        self.loading_scene = None

    def render(self, request):
        key = scene_key(self.root, request)
        scene = Path(key[0])
        self.loading_scene = scene.stem
        try:
            if key not in self.workers:
                if len(self.workers) >= self.cache_size:
                    _, old = self.workers.popitem(last=False)
                    old.close()
                    self.evictions += 1
                self.workers[key] = SingleWorker(self.root, 1, self.timeout)
                self.loads += 1
            worker = self.workers[key]
            self.workers.move_to_end(key)
            result = worker.render(request)
            self.requests += 1
            return result
        finally:
            self.loading_scene = None

    def health(self):
        states = [worker.health() for worker in list(self.workers.values())]
        return dict(ok=all(state['ok'] for state in states), protocol=1,
            stateless=True, service_instances=1, gl_workers=len(states),
            cache_capacity=self.cache_size, requests=self.requests,
            scene_loads=self.loads, evictions=self.evictions,
            cached_scenes=[Path(key[0]).stem for key in list(self.workers)],
            worker_pids=[state['worker_pid'] for state in states],
            active_scene=self.loading_scene,
            renderer='habitat-mesh-software-egl',
            isolation='one_scene_camera_per_gl_process_v1')

    def close(self):
        errors = []
        while self.workers:
            _, worker = self.workers.popitem(last=False)
            try:
                worker.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError('native scene workers failed to close') from errors[0]


class RenderScheduler:
    """Independent serial queues; a cold scene cannot stall every GL worker.

    Scene/camera affinity keeps each native context on its owning process.
    Legacy capacity is partitioned. The opt-in shared cache lends idle queue
    capacity while retaining one global count/byte admission budget.
    """
    def __init__(self, root, cache_size, parallelism=1, *,
                 shared_cache=False, cache_max_bytes=128 << 30,
                 scene_load_reserve_bytes=16 << 30):
        from concurrent.futures import ThreadPoolExecutor
        if not 1 <= parallelism <= min(cache_size, 16):
            raise ValueError('render parallelism must be 1..min(cache_size, 16)')
        self.root = Path(root).resolve(strict=True)
        self.shared_cache = None
        if shared_cache:
            from .scene_cache import SharedSceneWorkers
            self.shared_cache = SharedSceneWorkers(self.root, cache_size,
                max_bytes=cache_max_bytes, load_reserve_bytes=scene_load_reserve_bytes,
                parallelism=parallelism)
            self.pools = [self.shared_cache] * parallelism
        else:
            self.pools = [SceneWorkers(self.root, cache_size // parallelism +
                          int(i < cache_size % parallelism)) for i in range(parallelism)]
        self.queues = [ThreadPoolExecutor(max_workers=1, thread_name_prefix=f'habitat-gl-{i}')
                       for i in range(parallelism)]

    def shard(self, request):
        import hashlib
        scene = (self.root / request['scene']).resolve(strict=True)
        if not scene.is_relative_to(self.root) or scene.suffix != '.glb':
            raise ValueError('scene must be an MP3D glb below scenes_root')
        key = json.dumps([str(scene), request['sensors']], sort_keys=True).encode()
        return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % len(self.pools)

    def submit(self, request):
        index = self.shard(request)
        if self.shared_cache is not None:
            import time
            self.shared_cache.note_pending(index, 1)
            try:
                future = self.queues[index].submit(self.shared_cache.render, request,
                                                  queued_at=time.monotonic(), shard=index)
            except BaseException:
                self.shared_cache.note_pending(index, -1)
                raise
            future.add_done_callback(lambda _: self.shared_cache.note_pending(index, -1))
            return future
        return self.queues[index].submit(self.pools[index].render, request)

    def health(self):
        if self.shared_cache is not None:
            return dict(self.shared_cache.health(), render_parallelism=len(self.queues))
        states = [pool.health() for pool in self.pools]
        return dict(ok=all(s['ok'] for s in states), protocol=1, stateless=True,
                    service_instances=1, render_parallelism=len(states),
                    renderer='habitat-mesh-software-egl',
                    isolation='one_scene_camera_per_gl_process_v1',
                    **{key: sum(s[key] for s in states) for key in
                       ('cache_capacity', 'requests', 'scene_loads', 'evictions', 'gl_workers')},
                    cached_scenes=[x for s in states for x in s['cached_scenes']],
                    worker_pids=[x for s in states for x in s['worker_pids']],
                    active_scenes=[s['active_scene'] for s in states if s['active_scene']])

    def close(self):
        if self.shared_cache is not None:
            for queue in self.queues:
                queue.shutdown(wait=True)
            self.shared_cache.close()
            return
        futures = [queue.submit(pool.close) for queue, pool in zip(self.queues, self.pools)]
        errors = []
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                errors.append(exc)
        for queue in self.queues:
            queue.shutdown()
        if errors:
            raise RuntimeError('render scheduler cleanup failed') from errors[0]
