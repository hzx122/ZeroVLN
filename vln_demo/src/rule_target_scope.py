"""Local target admission from actual images still present in the brain request.

Reuse the camera/depth oracle and archived executed observations. Scene labels,
prospective controller poses and the trajectory map are never historical views.
"""
from collections import OrderedDict
import base64
import hashlib
import io
import json
import math
import importlib.util
import sys
import uuid
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image

from privileged_teacher import TeacherRejected
from rule_map_grounding import mesh_fingerprint
from sage.route_extractor import _bbox_distance_xy
from sage.visibility_audit import project_bbox, depth_support
from verl_access.vlm_policy import _image_to_data_url

VERSION = 'local-visible-target-scope-v2'
RADIUS_M = 16.
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_DEPTHS = OrderedDict()
_DEPTH_BYTES = 0
_DEPTH_LOCK = Lock()
_MAX_DEPTH_BYTES = 64 * 1024 * 1024


def _retry_io(operation, label):
    # Reuse the supervisor's existing fresh-open CPFS retry helper, including
    # standalone teacher entrypoints without scripts/train on PYTHONPATH.
    try:
        from dlc_control_io_retry import retry_estale
    except ImportError:
        source = Path(sys.modules['privileged_teacher'].__file__).resolve().parents[2]
        helper = Path(__file__).with_name('dlc_control_io_retry.py')
        if not helper.is_file():
            helper = source/'scripts/train/dlc_control_io_retry.py'
        spec = importlib.util.spec_from_file_location('dlc_control_io_retry',
            helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules['dlc_control_io_retry'] = module
        retry_estale = module.retry_estale
    return retry_estale(operation, label)


def _read_json(path):
    return json.loads(_retry_io(path.read_text, str(path)))


def _read_image(path):
    def read():
        with Image.open(path) as image:
            return image.convert('RGB').copy()
    return _retry_io(read, str(path))


def image_key(url):
    return hashlib.sha256(url.encode()).hexdigest()


def request_images(snapshot):
    images = {}
    for message in (snapshot or {}).get('messages', []):
        content = message.get('content', [])
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get('type') == 'image_url':
                url = part['image_url']['url']
                images[image_key(url)] = url
    return images


def _cached_depth(key):
    with _DEPTH_LOCK:
        value = _DEPTHS.get(key)
        if value is not None:
            _DEPTHS.move_to_end(key)
        return value


def _retain_depth(key, depth):
    global _DEPTH_BYTES
    depth = np.asarray(depth, dtype=np.float32).copy()
    depth.setflags(write=False)
    if depth.nbytes > _MAX_DEPTH_BYTES:
        return depth
    with _DEPTH_LOCK:
        if key not in _DEPTHS:
            _DEPTHS[key] = depth
            _DEPTH_BYTES += depth.nbytes
        while _DEPTH_BYTES > _MAX_DEPTH_BYTES:
            _, old = _DEPTHS.popitem(last=False)
            _DEPTH_BYTES -= old.nbytes
    return depth


class TargetSearchScope:
    def __init__(self, teacher, *, scene, episode, pose, rgb, depth, snapshot):
        self.teacher, self.scene, self.episode = teacher, scene, str(episode)
        self.pose = np.asarray(pose, float)
        self.images = request_images(snapshot)
        self.current_key = image_key(_image_to_data_url(Image.fromarray(np.asarray(rgb, np.uint8)), 90))
        # The explicit current observation is always valid; when messages are
        # supplied it must be the actual current camera image in that request.
        if self.images and self.current_key not in self.images:
            raise TeacherRejected('target scope: current observation absent from request')
        self.current_depth = depth
        self.frames = None
        self.audit = dict(schema=VERSION, radius_m=RADIUS_M,
            scope_source_sha256=SOURCE_SHA256,
            runtime_digest=getattr(teacher, '_target_scope_runtime_digest', None),
            distance_metric='horizontal distance to nearest target bbox point',
            center_world_xy=self.pose[:2].tolist(),
            visibility='positive metric-depth support in current or retained actual image',
            current_image_sha256=self.current_key, retained_image_count=len(self.images),
            outside_radius=0, outside_retained_views=0, current_visible=0,
            history_visible=0, historical_render_count=0, history_render_mismatches=0,
            unavailable_history_views=0, physical_reject_cache_hits=0,
            physical_reject_cache_misses=0)
        self.physical_key = None
        camera = getattr(teacher, 'camera', None)
        camera_id = tuple(getattr(camera, k, None) for k in ('width', 'height', 'fx', 'fy'))
        geometry = getattr(teacher, '_sage_floor_contract', None)
        if not hasattr(teacher, '_target_scope_instance_token'):
            teacher._target_scope_instance_token = uuid.uuid4().hex
        self.geometry_key = (teacher._target_scope_instance_token, scene, self.episode, camera_id,
            hashlib.sha256(json.dumps(geometry, sort_keys=True, default=str).encode()).hexdigest())
        key = self.geometry_key
        cache = getattr(teacher, '_target_visibility_cache', None)
        if cache is None or cache['geometry'] != key:
            cache = dict(geometry=key, views={})
            teacher._target_visibility_cache = cache
        self.cache = cache['views']
        # A view leaving the actual request cannot support a future candidate.
        for old in list(self.cache):
            if old not in self.images and old != self.current_key:
                del self.cache[old]

    def nearby(self, obj):
        distance = _bbox_distance_xy(self.pose[:2], obj['bbox_xyxy'])
        if not math.isfinite(distance) or distance > RADIUS_M:
            self.audit['outside_radius'] += 1
            return False
        return True

    def compile_candidate(self, program, occ, objects, openings):
        """Reuse only deterministic physical rejections at the identical state.

        No accepted plan, semantic admission failure or infrastructure exception
        is cached. Actual image/pose, full geometry and compiler changes reset
        the bounded per-teacher cache, including rejected waves at one depth.
        """
        from sage import planner
        compile_fn = self.teacher.compile
        code = getattr(getattr(compile_fn, '__func__', compile_fn), '__code__', None)
        if code is None or not hasattr(occ, 'free'):
            return compile_fn(program, self.pose.tolist(), occ, objects, openings)
        if self.physical_key is None:
            grid = np.ascontiguousarray(occ.free)
            geometry = hashlib.sha256(grid.tobytes())
            geometry.update(json.dumps(dict(shape=grid.shape, dtype=str(grid.dtype),
                objects={oid:dict(fingerprint=mesh_fingerprint(obj), centre_xy=obj['centre_xy'])
                         for oid,obj in objects.items()},
                openings=openings,
                map_geometry={name:getattr(occ, name, None) for name in
                              ('lower', 'mpp', 'world_frame', 'navmesh_manifest_sha256')}), sort_keys=True,
                default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)).encode())
            self.physical_key = (self.geometry_key, tuple(self.pose), self.current_key,
                tuple(self.teacher.camera_position(self.pose)), id(occ), geometry.hexdigest(),
                code, tuple((name, fn.__code__) for name,fn in sorted(vars(planner).items())
                            if hasattr(fn, '__code__')),
                tuple(getattr(getattr(getattr(occ, name), '__func__', getattr(occ, name)),
                              '__code__', None) for name in ('is_free', 'seg_free')
                      if hasattr(occ, name)))
        cache = getattr(self.teacher, '_target_physical_rejections', None)
        if cache is None or cache['state'] != self.physical_key:
            cache = dict(state=self.physical_key, errors=OrderedDict())
            self.teacher._target_physical_rejections = cache
        key = json.dumps(program, sort_keys=True, allow_nan=False)
        errors = cache['errors']
        if key in errors:
            errors.move_to_end(key)
            self.audit['physical_reject_cache_hits'] += 1
            error_type, reason = errors[key]
            raise error_type(reason)
        self.audit['physical_reject_cache_misses'] += 1
        try:
            return compile_fn(program, self.pose.tolist(), occ, objects, openings)
        except (TeacherRejected, ValueError) as error:
            if not isinstance(error, TeacherRejected) and str(error) != 'original centred route is unsafe':
                raise
            errors[key] = (type(error), str(error))
            while len(errors) > 512:
                errors.popitem(last=False)
            raise

    def _actual_frames(self):
        if self.frames is not None:
            return self.frames
        self.frames = {}
        artifact = getattr(self.teacher, 'artifact_dir', None)
        if artifact is None or not self.images:
            return self.frames
        case = Path(artifact).parent
        # Only this exact case's completed actual executions. The last 12
        # executions cover the v6 8-slow/3-fast observation window, with margin.
        folders = []
        for kind in ('options', 'turns'):
            root = case / kind
            if _retry_io(root.is_dir, str(root)):
                children = _retry_io(lambda: list(root.iterdir()), str(root))
                folders.extend(p for p in children if _retry_io(p.is_dir, str(p)))
        completed = 0
        for folder in sorted(folders, key=lambda p: p.name, reverse=True):
            receipt = folder / 'execution.json'
            if not _retry_io(receipt.is_file, str(receipt)):
                continue
            row = _read_json(receipt)
            if row.get('scene', self.scene) != self.scene or str(row.get('episode', self.episode)) != self.episode:
                continue
            state = row.get('state') or {}
            poses = [state.get('initial_pose')] + state.get('actual_poses', [])
            if not any(actual is not None for actual in poses):
                continue
            completed += 1
            for index, actual in enumerate(poses):
                path = folder / f'actual_chunk_{index}.png'
                if actual is None or not _retry_io(path.is_file, str(path)):
                    continue
                actual = np.asarray(actual, float)
                if actual.shape != (3,) or not np.isfinite(actual).all():
                    continue
                key = image_key(_image_to_data_url(_read_image(path), 90))
                if key in self.images and key != self.current_key:
                    self.frames.setdefault(key, dict(pose=actual.tolist(), path=str(path)))
            if completed >= 12:
                break
        self.audit['unavailable_history_views'] = len(set(self.images) - set(self.frames) - {self.current_key})
        return self.frames

    def _historical_depth(self, key, frame, camera_position):
        cache_key = (self.geometry_key, key, tuple(frame['pose']), tuple(camera_position))
        cached = _cached_depth(cache_key)
        if cached is not None:
            return cached
        rendered, depth = self.teacher.renderer.render_one(self.scene, camera_position,
            math.radians(frame['pose'][2]), want_depth=True)
        self.audit['historical_render_count'] += 1
        # Existing observations have passed the Gym JPEG codec. Verify the
        # reconstructed view against those pixels before trusting its depth.
        from vln_gym.env import _jpg_b64
        comparable = np.asarray(Image.open(io.BytesIO(base64.b64decode(_jpg_b64(rendered)))).convert('RGB'), float)
        original = np.asarray(_read_image(Path(frame['path'])), float)
        if comparable.shape != original.shape or np.abs(comparable - original).mean() > 3:
            self.audit['history_render_mismatches'] += 1
            return None
        if np.asarray(depth).ndim != 2:
            return None
        return _retain_depth(cache_key, depth)

    def admit(self, oid, obj, projection, support):
        fingerprint = mesh_fingerprint(obj)
        self.cache.setdefault(self.current_key, {})[fingerprint] = bool(support.get('visible_rays', 0) > 0)
        if support.get('visible_rays', 0) > 0:
            self.audit['current_visible'] += 1
            return dict(source='current', image_sha256=self.current_key, object_fingerprint=fingerprint)
        for key, values in self.cache.items():
            if key in self.images and values.get(fingerprint):
                self.audit['history_visible'] += 1
                return dict(source='history', image_sha256=key, object_fingerprint=fingerprint)
        for key, frame in self._actual_frames().items():
            values = self.cache.setdefault(key, {})
            if fingerprint in values:
                continue
            historical_pose = frame['pose']
            camera_position = self.teacher.camera_position(historical_pose)
            projected = project_bbox(self.teacher.camera, camera_position,
                math.radians(historical_pose[2]), obj)
            if not projected or projected['projected_area_ratio'] <= 0:
                values[fingerprint] = False
                continue
            depth = self._historical_depth(key, frame, camera_position)
            if depth is None:
                continue
            evidence = depth_support(depth, projected['bbox_px'], self.teacher.camera,
                camera_position, math.radians(historical_pose[2]), obj, stride=8)
            values[fingerprint] = evidence['visible_rays'] > 0
            if values[fingerprint]:
                self.audit['history_visible'] += 1
                return dict(source='history', image_sha256=key, object_fingerprint=fingerprint,
                    actual_pose=historical_pose, observation_path=frame['path'])
        self.audit['outside_retained_views'] += 1
        return None
