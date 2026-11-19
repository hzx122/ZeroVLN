"""One Habitat service with a bounded, persistent multi-scene cache.

Clients own navigation, collision, episode and scoring state. Each render
request supplies its complete scene, camera and pose; no reset/step API exists.
"""
import argparse
import io
import json
import math
import os
import signal
import socket
import zipfile
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import OrderedDict
from threading import BoundedSemaphore
from .worker import RenderScheduler

import numpy as np


class Renderer:
    def __init__(self, scenes_root, cache_size=128):
        self.root = Path(scenes_root).resolve(strict=True)
        if not 1 <= cache_size <= 512:
            raise ValueError('scene cache size must be 1..512')
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.requests = self.loads = self.evictions = 0
        self.status = {}
        self.loading_scene = None
        self.update_status()

    def update_status(self):
        self.status = dict(ok=True,protocol=1,stateless=True,service_instances=1,gl_threads=1,
            requests=self.requests,scene_loads=self.loads,evictions=self.evictions,cache_capacity=self.cache_size,
            cached_scenes=[Path(key[0]).stem for key in self.cache],renderer='habitat-mesh-software-egl')

    def health(self):
        return dict(self.status,loading_scene=self.loading_scene)

    def close(self):
        for sim in self.cache.values():
            sim.close()
        self.cache.clear()

    def render(self, request):
        import habitat_sim
        from habitat_sim.utils.common import quat_from_coeffs
        scene = (self.root / request['scene']).resolve(strict=True)
        if not scene.is_relative_to(self.root) or scene.suffix != '.glb':
            raise ValueError('scene must be an MP3D glb below scenes_root')
        position = np.asarray(request['position'], dtype=np.float32)
        rotation = np.asarray(request['rotation'], dtype=np.float64)
        if position.shape != (3,) or rotation.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise ValueError('invalid pose')
        if abs(float(np.linalg.norm(rotation)) - 1) > 1e-3:
            raise ValueError('rotation must be a unit quaternion in xyzw order')
        sensors = request['sensors']
        if not 1 <= len(sensors) <= 4:
            raise ValueError('1 to 4 sensors required')
        specs = []
        canonical_sensors = []
        for sensor in sensors:
            h, w = sensor['resolution']
            if not all(type(v) is int and 1 <= v <= 1024 for v in (h, w)):
                raise ValueError('resolution must be 1..1024')
            hfov = float(sensor['hfov'])
            if not math.isfinite(hfov) or not 1 <= hfov <= 179:
                raise ValueError('invalid hfov')
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid = sensor['uuid']
            if spec.uuid not in ('rgb', 'depth'):
                raise ValueError('only rgb and depth supported')
            if sensor['type'] not in ('COLOR', 'DEPTH'):
                raise ValueError('unsupported sensor type')
            spec.sensor_type = getattr(habitat_sim.SensorType, sensor['type'])
            spec.resolution = [h, w]
            spec.hfov = hfov
            spec.position = sensor['position']
            spec.orientation = sensor['orientation']
            spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            specs.append(spec)
            canonical_sensors.append(dict(uuid=spec.uuid,type=sensor['type'],resolution=[h,w],
                hfov=hfov,position=[float(v) for v in sensor['position']],
                orientation=[float(v) for v in sensor['orientation']]))
        key = (str(scene), json.dumps(canonical_sensors, sort_keys=True))
        if key not in self.cache:
            if len(self.cache) >= self.cache_size:
                _, old = self.cache.popitem(last=False)
                old.close()
                self.evictions += 1
            self.loading_scene = scene.stem
            cfg = habitat_sim.SimulatorConfiguration()
            cfg.scene_id = str(scene)
            cfg.gpu_device_id = -1
            cfg.enable_physics = False
            agent = habitat_sim.agent.AgentConfiguration()
            agent.sensor_specifications = specs
            try:
                self.cache[key] = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))
                self.loads += 1
            finally:
                self.loading_scene = None
        sim = self.cache[key]
        self.cache.move_to_end(key)
        state = habitat_sim.AgentState()
        state.position = position
        state.rotation = quat_from_coeffs(rotation)
        sim.get_agent(0).set_state(state, reset_sensors=True)
        observations = {key: value.copy() for key,value in sim.get_sensor_observations().items()}
        self.requests += 1
        self.update_status()
        return observations


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--port', type=int, default=7040)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--scenes-root', required=True)
    parser.add_argument('--scene-cache-size', type=int, default=int(os.environ.get('VLN_RENDER_SCENE_CACHE_SIZE', '128')))
    parser.add_argument('--render-parallelism', type=int, default=1)
    parser.add_argument('--shared-scene-cache', action='store_true')
    parser.add_argument('--scene-cache-max-gib', type=int, default=128)
    parser.add_argument('--scene-load-reserve-gib', type=int, default=16)
    parser.add_argument('--queue-size', type=int, default=128)
    args = parser.parse_args()
    if args.queue_size < 1:
        parser.error('queue-size must be positive')
    if not 0 < args.scene_load_reserve_gib <= args.scene_cache_max_gib:
        parser.error('positive scene memory reserve within cache budget required')
    renderer = RenderScheduler(args.scenes_root, args.scene_cache_size, args.render_parallelism,
        shared_cache=args.shared_scene_cache, cache_max_bytes=args.scene_cache_max_gib << 30,
        scene_load_reserve_bytes=args.scene_load_reserve_gib << 30)
    capacity = BoundedSemaphore(args.queue_size)

    class Handler(BaseHTTPRequestHandler):
        def log_request(self, code='-', size='-'):
            # High-frequency successful frame requests need no access log.
            # Keep failures visible; independent task liveness is external.
            if code != '-' and int(code) >= 400:
                super().log_request(code, size)

        def do_GET(self):
            if self.path != '/health':
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            hostname = socket.gethostname()
            self.wfile.write(json.dumps(dict(renderer.health(), hostname=hostname,
                job_id=hostname.rsplit('-master-', 1)[0] if '-master-' in hostname else None)).encode())

        def do_POST(self):
            if self.path != '/render':
                self.send_error(404)
                return
            if not capacity.acquire(blocking=False):
                self.send_error(503, 'renderer queue full')
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 65536:
                    raise ValueError('invalid request size')
                payload = json.loads(self.rfile.read(length))
                observations = renderer.submit(payload).result()
                # Compress/send on HTTP threads while the GL thread renders the
                # next request. Arrays are private copies of this exact pose.
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
                    for key,value in dict(request_id=np.array(payload['request_id']),**observations).items():
                        with archive.open(key+'.npy','w',force_zip64=True) as stream:
                            np.lib.format.write_array(stream,np.asarray(value),allow_pickle=False)
                result = buffer.getvalue()
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Length', str(len(result)))
                self.end_headers()
                self.wfile.write(result)
            except (ValueError, KeyError, TypeError) as exc:
                self.send_error(400, str(exc))
            except (OSError, RuntimeError, EOFError) as exc:
                self.send_error(503, str(exc))
            finally:
                capacity.release()

    print(json.dumps({'listening': args.port, 'scenes_root': str(renderer.root)}), flush=True)
    class RenderHTTPServer(ThreadingHTTPServer):
        request_queue_size = args.queue_size
    server = RenderHTTPServer((args.host, args.port), Handler)
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        renderer.close()


if __name__ == '__main__':
    main()
