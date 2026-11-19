"""Native MP3D geometry/render adapters for instruction-conditioned RuleTeacher.

Episode goals and reference paths are deliberately absent from the task view.
Only the original instruction/start, the actual simulator pose, native navmesh,
and independent .house semantic annotations enter this evaluator adapter.
The first adapter is conservative: local plans must stay on one flat floor.
"""
from __future__ import annotations

from functools import lru_cache, wraps
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
from collections import OrderedDict
import hashlib
import gzip
import io
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace, MappingProxyType
from urllib.request import Request, ProxyHandler, build_opener
import uuid
from threading import RLock

import numpy as np
from PIL import Image

from rs3d.gs_client import CameraSpec
from mp3d_teacher_semantics import DEFAULT_MP3D_ROOT
from sage.occmap import OccMap
from rule_teacher import RuleTeacher, TeacherRejected
from privileged_teacher import RADIUS


DATASETS_ROOT = Path('/data/shared/groups/mfm/common/anno/vlmevalkit/RynnBrain/data/datasets')
SCENES_ROOT = Path('/data/shared/groups/mfm/common/anno/vlmevalkit/RynnBrain/data/scene_datasets/mp3d')
HOUSE_ROOT = DEFAULT_MP3D_ROOT


_NATIVE_GRAMMAR_LOCK = RLock()

# Reuse the bounded process-local LRU pattern from verl_access.vlm_policy.
# Only content-bound immutable raster results are shared. Simulator state,
# PathFinder objects and each actor's native query caches remain independent.
_FLOOR_SLICE_MAX_ENTRIES = 8
_FLOOR_SLICE_MAX_BYTES = 256 * 1024 * 1024
_floor_slices = OrderedDict()
_floor_slice_bytes = 0
_floor_slice_lock = RLock()


def _navmesh_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.digest()


def _cached_floor_slice(key):
    with _floor_slice_lock:
        value = _floor_slices.get(key)
        if value is not None:
            _floor_slices.move_to_end(key)
        return value


def _retain_floor_slice(key, free, clearance):
    global _floor_slice_bytes
    size = free.nbytes + clearance.nbytes
    if size > _FLOOR_SLICE_MAX_BYTES:
        return
    # Copy before freezing: callers cannot alter an entry via its owner buffer.
    free, clearance = free.copy(), clearance.copy()
    free.setflags(write=False)
    clearance.setflags(write=False)
    with _floor_slice_lock:
        if key in _floor_slices:
            _floor_slices.move_to_end(key)
            return
        _floor_slices[key] = (free, clearance)
        _floor_slice_bytes += size
        while (len(_floor_slices) > _FLOOR_SLICE_MAX_ENTRIES or
               _floor_slice_bytes > _FLOOR_SLICE_MAX_BYTES):
            _, removed = _floor_slices.popitem(last=False)
            _floor_slice_bytes -= sum(value.nbytes for value in removed)



def _serialized_native_scope(scope):
    """Restore shared parser helpers before another actor can replace them.

    A prepare call nests all four scopes. A reentrant process-local lock keeps
    their combined lifetime atomic; separate collection workers stay parallel.
    """
    @wraps(scope)
    @contextmanager
    def serialized(*args, **kwargs):
        with _NATIVE_GRAMMAR_LOCK:
            with scope(*args, **kwargs):
                yield
    return serialized


@_serialized_native_scope
@contextmanager
def native_room_grammar():
    """Reuse the room parser with the exact native h-region taxonomy.

    This process-local parsing scope leaves original calls/NLP evidence
    untouched and restores the shared SAGE vocabulary afterward. A hallway
    still needs a witnessed native portal and an annotated destination room.
    """
    import re
    import rule_lexicon
    import rule_semantics
    rooms = dict(rule_lexicon.ROOMS)
    patterns = rule_lexicon.ROOM_PATTERN, rule_semantics.ROOM_PATTERN
    try:
        rule_lexicon.ROOMS.update(hallway='hallway', corridor='hallway', hall='hallway')
        rule_lexicon.ROOMS.update({'laundry room': 'laundryroom/mudroom',
                                  'laundry area': 'laundryroom/mudroom'})
        rule_lexicon.ROOMS.update({'game room': 'rec/game', 'recreation room': 'rec/game'})
        # Canonical compiler area names are valid spellings too; they do not
        # change which native region can establish the requested function.
        rule_lexicon.ROOMS.update({value: value for value in rule_lexicon.ROOMS.values()})
        pattern = '(?:' + '|'.join(sorted(map(re.escape, rule_lexicon.ROOMS), key=len, reverse=True)) + ')'
        rule_lexicon.ROOM_PATTERN = rule_semantics.ROOM_PATTERN = pattern
        yield
    finally:
        rule_lexicon.ROOMS.clear()
        rule_lexicon.ROOMS.update(rooms)
        rule_lexicon.ROOM_PATTERN, rule_semantics.ROOM_PATTERN = patterns


@_serialized_native_scope
@contextmanager
def native_target_grammar():
    """Preserve native target qualifiers as explicit geometry constraints.

    This scope accepts ``large`` only on the current main object. References
    with size adjectives remain unsupported by the shared relation parser.
    Original text and NLP graphs are retained by RuleTeacher unchanged.
    """
    import re
    import rule_teacher
    from rule_lexicon import ROOMS
    simple = rule_teacher.parse_simple_contract
    select = rule_teacher.select_navigation_prefix
    target_direction = rule_teacher.parse_target_direction_contract
    words = set(rule_teacher.WORDS)
    names = dict(rule_teacher.NAMES)
    noun = rule_teacher.NOUN
    primitive = rule_teacher.parse_contract

    def room_endpoint(target):
        """Keep an explicitly named far room, including its object qualifier."""
        target = rule_teacher.normalized(target).removeprefix('the ')
        room_pattern = '(?:' + '|'.join(map(re.escape, sorted(ROOMS, key=len, reverse=True))) + ')'
        room = re.fullmatch('(?P<room>' + room_pattern + r')(?: with (?:the |a |an )?(?P<reference>' + rule_teacher.NOUN.pattern + '))?', target)
        object_room = re.fullmatch(r'(?:a |an )?' + rule_teacher.NOUN.pattern + r' in (?:the )?(?P<room>' + room_pattern + ')', target)
        if room:
            return dict(destination_room=ROOMS[room['room']],
                        reference_family=rule_teacher.NAMES[room['reference']] if room['reference'] else None)
        if object_room:
            return dict(destination_room=ROOMS[object_room['room']], reference_family=None)
        # An explicit room qualifier cannot vanish just because this call's
        # first navigation endpoint is a portal. Unresolved suite/attributes
        # remain a rejection instead of an unrestricted doorway label.
        if re.match(room_pattern + r'\b', target) or re.search(r'\bin (?:the )?' + room_pattern + r'\b', target):
            raise TeacherRejected('native: original portal destination qualifier is unresolved')
        return None

    def sized_simple(contract):
        target = rule_teacher.normalized(contract.get('target', ''))
        where = rule_teacher.normalized(contract.get('where', ''))
        if re.search(r'\blarge\b', where):
            raise TeacherRejected('native: size-qualified location/reference is unresolved')
        size = bool(re.search(r'\blarge\b', target))
        if size and not re.match(r'^(?:the |a |an )?large\s+' + rule_teacher.NOUN.pattern, target):
            raise TeacherRejected('native: large must qualify the main object')
        spec = simple(contract)
        if size:
            spec['native_size_constraint'] = dict(adjective='large', comparison='same-floor same-family inventory',
                source_target=contract['target'], min_linear_ratio=1.25)
        return spec

    def selected(contract, graph=None, *, open_vocabulary=False):
        # An object's qualified identity is this call's actual endpoint. The
        # target/direction compiler already ignores later plan obligations and
        # supports a witnessed portal followed by the full object approach.
        # Turning that call into a doorway label would lose its relation/room
        # constraints and finish before reaching the requested current target.
        original_target = rule_teacher.normalized(contract.get('target', '')).removeprefix('the ')
        original_noun = rule_teacher.NOUN.search(original_target)
        room_pattern = '(?:' + '|'.join(map(re.escape, sorted(ROOMS, key=len, reverse=True))) + ')'
        if (original_noun and rule_teacher.NAMES[original_noun.group()] != 'doorway'
                and not re.match(room_pattern + r'\b', original_target)):
            from copy import deepcopy
            return deepcopy(contract), dict(mode='original_current_object_target',
                retained_target=contract.get('target', ''), retained_where=contract.get('where', ''),
                reason='full current object endpoint; original qualifiers and initial motion retained')
        if (original_noun and rule_teacher.NAMES[original_noun.group()]=='doorway'
                and re.search(r'\b(?:to|into|leading)\b',original_target[original_noun.end():])):
            from copy import deepcopy
            return deepcopy(contract), dict(mode='original_qualified_portal_target',
                retained_target=contract.get('target',''), retained_where=contract.get('where',''),
                reason='door destination belongs to the current target identity')
        local, prefix = select(contract, graph, open_vocabulary=open_vocabulary)
        where = rule_teacher.normalized(contract.get('where', ''))
        portal = re.search(r'\b(?:through|via)\s+((?:the |a |an )?(?:open\s+)?(?:doorway|opening)\b.*)', where)
        target_noun = rule_teacher.NOUN.search(rule_teacher.normalized(local.get('target', '')))
        if (portal and target_noun
                and rule_teacher.NAMES[target_noun.group()] == 'doorway'
                and prefix.get('mode') == 'first_navigation_endpoint'):
            endpoint = room_endpoint(contract.get('target', ''))
            from rule_nlp_grounding import EGO_PREFIX
            ego = where[:portal.start()].strip(' ,')
            ego = ego if set(re.findall(r'[a-z]+', ego)) <= EGO_PREFIX | {'forward'} else ''
            local['where'] = (ego + ' ' + portal[1]).strip()
            prefix = dict(prefix, retained_where=local['where'])
            if endpoint is None:
                return local, prefix
            # Shared prefix selection may already have retained an exact
            # room. Do not duplicate it or replace a conflicting destination.
            if not re.search(r'\bleading (?:to|into)\b', local['where']):
                local['where'] += ' leading to the ' + endpoint['destination_room']
            local['_native_original_destination'] = dict(endpoint, source_target=contract['target'])
            prefix = dict(prefix, retained_where=local['where'],
                          native_destination_preserved=endpoint['destination_room'])
        elif (target_noun and rule_teacher.NAMES[target_noun.group()] == 'doorway'
              and prefix.get('mode') == 'first_navigation_endpoint'
              and re.search(r'\b(?:through|via)\b', where)):
            # A location such as "straight ahead through the archway" must
            # not vanish when a plan abbreviates it as "the open doorway".
            # Preserve the original location verbatim for the existing
            # parser. An unsupported portal identity remains a rejection;
            # it does not become any convenient annotated doorframe.
            local['where'] = contract.get('where', '')
            prefix = dict(prefix, retained_where=local['where'],
                          native_unresolved_portal_location_preserved=True)
        return local, prefix

    def connected_primitive(contract):
        # "Adjacent room" asserts the far side of the selected portal, not
        # a named functional destination. Reuse the existing connected-room
        # guard, after original NLP evidence has been recorded unchanged.
        from copy import deepcopy
        local = deepcopy(contract)
        adjacent = []
        current_noun = rule_teacher.NOUN.search(rule_teacher.normalized(local.get('target', '')))
        object_endpoint = current_noun and rule_teacher.NAMES[current_noun.group()] != 'doorway'
        for field in ('target', 'where'):
            local[field] = re.sub(r'\b(leading to|leading into|into|to) (the |a )?adjacent room\b',
                                  r'\1 \2next room', local.get(field, ''), flags=re.I)
            if object_endpoint:
                pattern = r'\bin (?:the |an |a )?adjacent room\b'
                adjacent.extend(dict(field=field, phrase=m.group())
                                for m in re.finditer(pattern, local[field], flags=re.I))
                local[field] = re.sub(pattern, '', local[field], flags=re.I).strip(' ,')
        spec = primitive(local)
        if adjacent:
            # Only the exact topological phrase is lowered here. Furniture
            # relations, direction and other words still use the shared
            # parser; the original call/NLP graph are never rewritten.
            spec['native_target_adjacent_room'] = dict(source_phrases=adjacent,
                evidence_required='unique witnessed portal from current room to object room')
            spec['via_portal'] = True
        return spec

    def constrained_target_direction(contract, graph=None, *, select_prefix=True, entity_spec=None):
        spec = target_direction(contract, graph, select_prefix=select_prefix, entity_spec=entity_spec)
        if re.search(r'\bdouble doors\b', rule_teacher.normalized(contract.get('target', ''))):
            spec['native_double_door_group'] = dict(source_target=contract['target'], minimum_leaves=2)
        constraint = contract.get('_native_original_destination')
        if constraint:
            if spec.get('destination_room') != constraint['destination_room']:
                raise TeacherRejected('native: prefix destination differs from original target room')
            spec['native_original_destination'] = dict(constraint)
        return spec

    rule_teacher.NAMES['double doors'] = 'doorway'
    rule_teacher.NOUN = re.compile(r'\b(' + '|'.join(re.escape(x) for x in sorted(rule_teacher.NAMES, key=len, reverse=True)) + r')\b')
    rule_teacher.WORDS.add('large')
    rule_teacher.parse_contract = connected_primitive
    rule_teacher.parse_simple_contract = sized_simple
    rule_teacher.select_navigation_prefix = selected
    rule_teacher.parse_target_direction_contract = constrained_target_direction
    try:
        yield
    finally:
        rule_teacher.WORDS.clear()
        rule_teacher.WORDS.update(words)
        rule_teacher.NAMES.clear()
        rule_teacher.NAMES.update(names)
        rule_teacher.NOUN = noun
        rule_teacher.parse_contract = primitive
        rule_teacher.parse_simple_contract = simple
        rule_teacher.select_navigation_prefix = select
        rule_teacher.parse_target_direction_contract = target_direction


def habitat_position(world_xy, height):
    """Inverse of VLNCEActionHeadEnv's fixed forward/left planar frame."""
    return np.asarray([-float(world_xy[1]), float(height), -float(world_xy[0])], dtype=np.float32)


def habitat_yaw_xyzw(yaw):
    return [0., math.sin(float(yaw) / 2), 0., math.cos(float(yaw) / 2)]


def rotation_yaw(rotation):
    q = np.asarray(rotation, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1.) > 1e-3:
        raise ValueError('MP3D start rotation must be a finite xyzw unit quaternion')
    x, y, z, w = q
    if abs(x) > 1e-5 or abs(z) > 1e-5:
        raise ValueError('MP3D teacher currently requires an untilted agent')
    return math.atan2(2 * w * y, 1 - 2 * y * y)


_instruction_sets = OrderedDict()
_instruction_sets_lock = RLock()
_INSTRUCTION_SET_MAX_ENTRIES = 4
_INSTRUCTION_SET_MAX_BYTES = 256 * 1024 * 1024
_instruction_set_bytes = 0


def _instruction_templates(backend, source, compressed=None):
    """Keep only original language/start fields in immutable shared templates."""
    stream = (gzip.open(source, 'rt') if compressed is None else
              io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(compressed))))
    tasks, size = {}, 0
    with stream:
        for row in json.load(stream)['episodes']:
            instruction = row['instruction']
            if backend == 'rxr' and instruction.get('language') not in ('en-US', 'en-IN'):
                continue
            scene = Path(row['scene_id']).stem
            position = np.asarray(row['start_position'], dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError('invalid native start position')
            yaw = rotation_yaw(row['start_rotation'])
            task_id = str(row['episode_id'])
            scene_tasks = tasks.setdefault(scene, {})
            if task_id in scene_tasks:
                raise ValueError(f'duplicate {backend} task identity {scene}/{task_id}')
            text = instruction['instruction_text']
            scene_tasks[task_id] = (task_id, text, -float(position[2]), -float(position[0]),
                math.degrees(yaw), tuple(position.tolist()), tuple(row['start_rotation']))
            # Conservative accounting for the strings, scalar tuple and map row.
            size += len(text.encode('utf-8')) + len(task_id) + len(scene) + 1024
    return MappingProxyType({scene: tuple(rows.values()) for scene, rows in tasks.items()}), size


def _cached_instruction_templates(backend, split, source):
    global _instruction_set_bytes
    try:
        compressed = source.read_bytes()
    except FileNotFoundError:
        # Preserve callers that inject gzip streams (including existing tests).
        # A real missing source still fails in the original gzip loader.
        templates, _ = _instruction_templates(backend, source)
        return templates, None, False
    digest = hashlib.sha256(compressed).hexdigest()
    key = (backend, split, digest)
    with _instruction_sets_lock:
        cached = _instruction_sets.get(key)
        if cached is not None:
            _instruction_sets.move_to_end(key)
            return cached[0], digest, True
        templates, size = _instruction_templates(backend, source, compressed)
        if size <= _INSTRUCTION_SET_MAX_BYTES:
            while _instruction_sets and (len(_instruction_sets) >= _INSTRUCTION_SET_MAX_ENTRIES or
                    _instruction_set_bytes + size > _INSTRUCTION_SET_MAX_BYTES):
                _, (_, removed_size) = _instruction_sets.popitem(last=False)
                _instruction_set_bytes -= removed_size
            _instruction_sets[key] = (templates, size)
            _instruction_set_bytes += size
        return templates, digest, False


class MP3DInstructionSet:
    """Minimal task view reused by PrivilegedTeacher, with no route oracle."""

    def __init__(self, backend, *, split='train', source=None):
        if backend not in ('r2r', 'rxr'):
            raise ValueError('MP3D teacher dataset must be r2r or rxr')
        self.backend, self.split = backend, split
        self.source = Path(source) if source is not None else DATASETS_ROOT / backend / split / (
            f'{split}.json.gz' if backend == 'r2r' else f'{split}_guide.json.gz')
        self._tasks, self.instruction_source_sha256, self.instruction_cache_hit = (
            _cached_instruction_templates(backend, split, self.source))
        self.scenes = sorted(self._tasks)

    def tasks(self, scene):
        # Mutable caller views are independent, including nested start fields.
        return [SimpleNamespace(task_id=task_id, instruction=text,
            start=dict(x=x, y=y, yaw=yaw), native_start_position=list(position),
            native_start_rotation=list(rotation))
            for task_id, text, x, y, yaw, position, rotation in self._tasks[scene.removeprefix('rxr-')]]


class MP3DNavmeshMap(OccMap):
    """Reuse OccMap/planner APIs over a floor slice of the true Habitat navmesh."""

    def __init__(self, navmesh, start_position, *, mpp=.05, floor_tolerance=.06, pathfinder=None,
                 world_frame='mp3d', cache_floor_slice=True):
        # Injected PathFinders may be mutable and are never cacheable.
        managed_native = pathfinder is None
        self._native_managed_pathfinder = managed_native
        self._native_construction_start = tuple(float(v) for v in start_position)
        digest = _navmesh_sha256(navmesh) if managed_native and cache_floor_slice else None
        self._native_loaded_navmesh_sha256 = digest
        self.floor_slice_cache_hit = False
        if world_frame not in ('mp3d', 'sage'):
            raise ValueError('unknown navigation world frame')
        self.world_frame = world_frame
        if pathfinder is None:
            import habitat_sim
            pathfinder = habitat_sim.PathFinder()
            if not pathfinder.load_nav_mesh(str(navmesh)):
                raise FileNotFoundError(f'failed native navmesh load: {navmesh}')
        self.pathfinder = pathfinder
        self.navmesh_path = str(Path(navmesh))
        self.scene_dir = str(Path(navmesh).parent)
        self.floor_height = float(start_position[1])
        self.floor_tolerance = float(floor_tolerance)
        self.scale = self.mpp = float(mpp)
        if not .01 <= self.mpp <= .1 or not 0 < self.floor_tolerance <= .1:
            raise ValueError('MP3D floor map requires a fine grid and a strict flat-floor tolerance')
        lower, _ = [np.asarray(v, float) for v in pathfinder.get_bounds()]
        if digest is not None and _navmesh_sha256(navmesh) != digest:
            raise TeacherRejected('native navmesh changed during PathFinder load')
        key = (digest, self.floor_height.hex(), self.mpp.hex(),
               self.floor_tolerance.hex(), world_frame, tuple(lower)) if digest is not None else None
        cached = _cached_floor_slice(key) if key is not None else None
        if cached is not None:
            self.free, self.clearance_m = cached
            self.floor_slice_cache_hit = True
        else:
            native = np.asarray(pathfinder.get_topdown_view(
                meters_per_pixel=self.mpp, height=self.floor_height, eps=self.floor_tolerance), bool)
            if native.ndim != 2 or not native.any():
                raise TeacherRejected('MP3D native navmesh has no walkable current-floor slice')
            # Native rows increase Habitat Z, columns increase Habitat X. OccMap
            # rows decrease world y=-X and columns increase world x=-Z.
            self.free = np.ascontiguousarray(native if world_frame == 'sage' else native[::-1, :].T)
        self.h, self.w = self.free.shape
        self.lower = [-(lower[2] + (self.w - 1) * self.mpp),
                      -(lower[0] + (self.h - 1) * self.mpp), self.floor_height]
        if world_frame == 'sage':
            self.lower = [lower[0], -(lower[2] + (self.h-1)*self.mpp), self.floor_height]
        self.upper = [self.lower[0] + (self.w - 1) * self.mpp,
                      self.lower[1] + (self.h - 1) * self.mpp, self.floor_height]
        self.occ = self.free.astype(np.uint8) * 255
        self.collision_mode, self.furniture_mode = 'habitat_navmesh', 'native'
        # A height-slice boundary on a ramp is not an obstacle. Keep the slice
        # as the allowed centre positions, but query physical clearance on the
        # original navmesh. Otherwise EDT erodes valid starts near that boundary.
        if cached is None:
            self.clearance_m = np.zeros(self.free.shape, dtype=float)
            for row, col in np.argwhere(self.free):
                xy = self.to_world(col, row)
                self.clearance_m[row, col] = self.native_clearance(*xy)
            if key is not None:
                if _navmesh_sha256(navmesh) != digest:
                    raise TeacherRejected('native navmesh changed while deriving floor slice')
                _retain_floor_slice(key, self.free, self.clearance_m)
        start_xy = self.world_xy(np.asarray(start_position, float))
        if self.snap_xy(*start_xy) is None:
            raise TeacherRejected('MP3D actual start is off the selected native floor')

    def world_xy(self, point):
        if self.world_frame == 'sage':
            return float(point[0]), -float(point[2])
        return -float(point[2]), -float(point[0])

    def native_position(self, xy):
        if self.world_frame == 'sage':
            return np.asarray([xy[0], self.floor_height, -xy[1]], np.float32)
        return habitat_position(xy, self.floor_height)

    @lru_cache(maxsize=32768)
    def snap_xy(self, x, y):
        requested = self.native_position((x, y))
        snapped = np.asarray(self.pathfinder.snap_point(requested), dtype=float)
        if (snapped.shape != (3,) or not np.isfinite(snapped).all()
                or abs(snapped[1] - self.floor_height) > self.floor_tolerance
                or np.linalg.norm(snapped[[0, 2]] - requested[[0, 2]]) > .025):
            return None
        return tuple(float(v) for v in snapped)

    def is_free(self, x, y, radius_m=0.):
        return bool(super().is_free(x, y, radius_m)
                    and self.native_clearance(float(x), float(y)) > radius_m)

    @lru_cache(maxsize=32768)
    def native_clearance(self, x, y):
        point = self.snap_xy(float(x), float(y))
        if point is None:
            return 0.
        value = float(self.pathfinder.distance_to_closest_obstacle(np.asarray(point, np.float32), 2.))
        return value if math.isfinite(value) and value >= 0 else 0.

    def seg_free(self, x0, y0, x1, y1, radius_m=0.):
        if not self.is_free(x0, y0, radius_m):
            return False
        start, end = self.snap_xy(float(x0), float(y0)), self.snap_xy(float(x1), float(y1))
        if start is None or end is None:
            return False
        actual = np.asarray(self.pathfinder.try_step_no_sliding(
            np.asarray(start, np.float32), np.asarray(end, np.float32)), float)
        # A fixed 1.5 cm allowance accepts a completely blocked 1 cm action.
        # Keep short actions valid, but require the oracle to execute them.
        distance = float(np.linalg.norm(np.asarray(end)-start))
        tolerance = min(.015, .05*distance + 1e-5)
        # String pulling probes many long, obstructed shortcuts. Reject those
        # with the same native edge oracle before sampling their full length.
        # Edges that pass still need every original footprint/clearance check.
        if not (np.isfinite(actual).all() and np.linalg.norm(actual - end) <= tolerance
                and abs(actual[1] - self.floor_height) <= self.floor_tolerance):
            return False
        grid = self._grid_segment_free(x0, y0, x1, y1, radius_m)
        if grid is False:
            return False
        if grid is not None and type(self).is_free is MP3DNavmeshMap.is_free:
            # The batched base predicate has checked every original sample;
            # preserve the extra native snap/floor/clearance query at each one.
            return all(self.native_clearance(float(x), float(y)) > radius_m
                       for x, y in self._samples(x0, y0, x1, y1))
        return bool(super().seg_free(x0, y0, x1, y1, radius_m))

    def advance(self, x0, y0, x1, y1, radius_m=0.):
        last = (x0, y0)
        for x, y in self._samples(x0, y0, x1, y1):
            if not self.seg_free(*last, x, y, radius_m):
                break
            last = (x, y)
        return last

    def repair_grid_path(self, start_xy, goal_xy, radius_m):
        """Try a native route, retaining exact endpoints and all safety checks.

        Native shortest paths can also hug a wall or leave the selected floor.
        They are usable only after the same footprint and native edge oracle
        certify the entire route. Failure stays an ordinary unreachable plan.
        """
        if not callable(getattr(self.pathfinder, 'find_path', None)):
            return None
        import habitat_sim
        from sage.planner import _path_is_safe
        start, end = self.snap_xy(*map(float, start_xy)), self.snap_xy(*map(float, goal_xy))
        if start is None or end is None:
            return None
        query = habitat_sim.ShortestPath()
        query.requested_start = np.asarray(start, np.float32)
        query.requested_end = np.asarray(end, np.float32)
        if not self.pathfinder.find_path(query):
            return None
        points = np.asarray(query.points, float)
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
                or not np.isfinite(points).all()
                or np.any(np.abs(points[:, 1]-self.floor_height) > self.floor_tolerance)):
            return None
        route = np.asarray([self.world_xy(point) for point in points])
        route[0], route[-1] = start_xy, goal_xy
        if not _path_is_safe(self, route, radius_m):
            return None
        return [tuple(point) for point in route]


def native_door_group_leaves(semantic, oid, obj, midpoint, tangent, floor_height):
    """Witness two independent door panels on opposite sides of one free gap.

    Original OBB footprints must intersect; opened leaves may intersect only
    at their tips. This is spatial evidence, not a fabricated parent/child
    relation. A standalone door or two panels on the same side cannot pass.
    """
    import cv2
    def footprint(value):
        corners = np.asarray(value.get('obb_world', {}).get('corners', []), np.float32)
        if corners.shape != (8, 3) or not np.isfinite(corners).all():
            return None
        return cv2.convexHull(np.ascontiguousarray(corners[:, :2]))
    group = footprint(obj)
    if group is None:
        return None
    leaves = []
    for peer_id, peer in sorted(semantic.objects.items()):
        if peer_id == oid or peer['label'].replace(' ', '').replace('_', '').lower() != 'door':
            continue
        obb = peer.get('obb_world', {})
        axes, half = np.asarray(obb.get('axes', []), float), np.asarray(obb.get('half_lengths', []), float)
        if (axes.shape != (3, 3) or half.shape != (3,) or not np.isfinite(axes).all()
                or not np.isfinite(half).all() or np.any(half <= 0)
                or not np.allclose(axes @ axes.T, np.eye(3), atol=2e-3, rtol=0)):
            continue
        horizontal = [i for i, axis in enumerate(axes) if abs(axis[2]) < .02]
        if (len(horizontal) != 2 or abs(peer['z_range'][0] - floor_height) > .25
                or peer['z_range'][1] < floor_height + 1.5
                or max(half[i] for i in horizontal) < .2
                or max(half[i] for i in horizontal) <= 2 * min(half[i] for i in horizontal)):
            continue
        panel = footprint(peer)
        if panel is None:
            continue
        area, intersection = cv2.intersectConvexConvex(group, panel)
        side = float((np.asarray(peer['centre_xy']) - midpoint) @ tangent)
        if area <= 1e-6 or abs(side) <= RADIUS:
            continue
        leaves.append(dict(id=peer_id, native_object_id=peer.get('native_object_id'),
            tangential_offset_m=side, group_intersection_area_m2=float(area),
            leaf_intersection_fraction=float(area / cv2.contourArea(panel)),
            intersection_polygon_xy=intersection.reshape(-1, 2).tolist()))
    if not any(x['tangential_offset_m'] < 0 for x in leaves) or not any(x['tangential_offset_m'] > 0 for x in leaves):
        return None
    return dict(schema='mp3d_native_double_door_group_v1', leaves=leaves,
                evidence='independent same-floor upright door panels intersect group OBB on opposite sides of unique native free gap; no parent-child assertion')


def native_doorframe_openings(semantic, occ):
    """Verify apertures inside annotated doorframes on the actual floor.

    A door panel, a panorama P record, or adjacent room polygons alone do not
    establish an opening. Each retained strip has an upright native frame,
    collision-free native crossings, and two distinct unambiguous rooms.
    The room/direction/visibility guards and portal compiler remain unchanged.
    """
    openings, rejected = {}, []
    spacing = min(float(occ.mpp) / 2, .025)
    for oid, obj in sorted(semantic.objects.items()):
        kind = obj['label'].replace(' ', '').replace('_', '').lower()
        if kind not in ('doorframe', 'door'):
            continue
        obb = obj.get('obb_world', {})
        axes = np.asarray(obb.get('axes', []), float)
        half = np.asarray(obb.get('half_lengths', []), float)
        if (axes.shape != (3, 3) or half.shape != (3,) or not np.isfinite(axes).all()
                or not np.isfinite(half).all() or np.any(half <= 0)
                or not np.allclose(axes @ axes.T, np.eye(3), atol=2e-3, rtol=0)):
            rejected.append(dict(id=oid, reason='invalid native frame OBB'))
            continue
        horizontal = [i for i, axis in enumerate(axes) if abs(axis[2]) < .02]
        if (len(horizontal) != 2 or min(abs(z - occ.floor_height) for z in obj['z_range']) > .25
                or obj['z_range'][1] < occ.floor_height + 1.5):
            rejected.append(dict(id=oid, reason='frame is not upright on the actual floor'))
            continue
        along = max(horizontal, key=lambda i: half[i])
        across = min(horizontal, key=lambda i: half[i])
        if along == across or half[along] <= half[across] * 2 or half[along] <= RADIUS:
            rejected.append(dict(id=oid, reason='frame has no distinct horizontal aperture axis'))
            continue
        tangent = axes[along, :2].copy()
        tangent /= np.linalg.norm(tangent)
        if tangent[np.argmax(abs(tangent))] < 0:
            tangent = -tangent
        normal = np.asarray([-tangent[1], tangent[0]])
        center = np.asarray(obj['centre_xy'], float)
        if kind == 'door' and (half[along] < .7 or native_door_group_leaves(
                semantic, oid, obj, center, tangent, occ.floor_height) is None):
            rejected.append(dict(id=oid, reason='door is not an independently witnessed two-sided broad group'))
            continue
        limit = float(half[along] - RADIUS)
        offsets = np.linspace(-limit, limit, math.ceil(2 * limit / spacing) + 1)
        def traversable_runs(distance):
            runs, run = [], []
            for offset in offsets:
                point = center + offset * tangent
                near, far = point - distance * normal, point + distance * normal
                rooms = (semantic.room_at(near), semantic.room_at(far))
                valid = (None not in rooms and rooms[0] != rooms[1]
                         and occ.seg_free(*near, *far, RADIUS))
                if run and (not valid or rooms != run[-1][2]):
                    runs.append(run)
                    run = []
                if valid:
                    run.append((float(offset), point, rooms))
            if run:
                runs.append(run)
            return [r for r in runs if r[-1][0] - r[0][0] >= 2 * RADIUS]

        # Keep the established crossing where it is safe. A narrow toilet
        # can have a valid aperture while a fixed .55m endpoint hits its
        # opposite wall. Shorter candidates must lie outside the annotated
        # frame thickness plus the body footprint, and still prove two
        # distinct rooms with native collision checks.
        crossing = .55
        runs = traversable_runs(crossing)
        if not runs:
            minimum = float(half[across]) + RADIUS + .025
            candidates = np.arange(math.ceil(minimum / .025) * .025, .55, .025)
            verified = [(float(d), traversable_runs(float(d))) for d in candidates]
            verified = [(d, r) for d, r in verified if r]
            pairs = {tuple(x[2]) for _, groups in verified for r in groups for x in r}
            if verified and len(pairs) == 1 and all(len(r) == 1 for _, r in verified):
                crossing, runs = verified[0]
        if len(runs) != 1:
            rejected.append(dict(id=oid, reason='no unique traversable strip across two rooms',
                                 verified_strip_count=len(runs)))
            continue
        run = runs[0]
        ends = np.asarray([run[0][1], run[-1][1]])
        midpoint = ends.mean(axis=0)
        near, far = midpoint - crossing * normal, midpoint + crossing * normal
        if ((semantic.room_at(near), semantic.room_at(far)) != run[0][2]
                or not occ.seg_free(*near, *far, RADIUS)):
            rejected.append(dict(id=oid, reason='verified strip midpoint is not traversable'))
            continue
        group = None
        if kind == 'door':
            group = native_door_group_leaves(semantic, oid, obj, midpoint, tangent, occ.floor_height)
            if group is None:
                rejected.append(dict(id=oid, reason='door has no independently witnessed two-sided panel group'))
                continue
        aperture = dict(center=midpoint.tolist(), ends=ends.tolist(),
            native_evidence=dict(schema='mp3d_native_doorframe_aperture_v1',
                native_object_ids=[oid], native_region_id=obj['native_region_id'],
                source_path=getattr(getattr(semantic, 'native_map', None), 'source_path', None),
                navmesh_path=occ.navmesh_path, floor_height=occ.floor_height,
                frame_obb=obb, room_pair=list(run[0][2]),
                crossing_offset_m=crossing, footprint_radius_m=RADIUS,
                sampling_spacing_m=float(offsets[1] - offsets[0]),
                verified_crossings=[dict(xy=p.tolist(), rooms=list(rooms)) for _, p, rooms in run],
                evidence='native doorframe OBB, same-floor navmesh strip and two native room polygons'))
        if group is not None:
            aperture['native_evidence']['double_door_group'] = group
            aperture['native_evidence']['evidence'] = 'native door group OBB, independent panels, same-floor free strip and two native room polygons'
        # Some .house files annotate the same frame at different extents.
        # Collapse only nearly coplanar, parallel, overlapping verified strips
        # between the same rooms; the selected strip itself stays unchanged.
        duplicate = None
        for prior in openings.values():
            a, b = np.asarray(prior['ends'])
            axis = (b - a) / np.linalg.norm(b - a)
            prior_rooms = prior['native_evidence']['room_pair']
            intervals = sorted(float((p - a) @ axis) for p in ends)
            overlap = min(np.linalg.norm(b - a), intervals[1]) - max(0., intervals[0])
            minimum_width = min(np.linalg.norm(b - a), np.linalg.norm(ends[1] - ends[0]))
            if (set(prior_rooms) == set(run[0][2]) and abs(axis @ tangent) > .999
                    and abs((ends.mean(axis=0) - a) @ np.asarray([-axis[1], axis[0]])) <= .05
                    and overlap >= .7 * minimum_width):
                duplicate = prior
                break
        if duplicate is not None:
            duplicate['native_evidence']['native_object_ids'].append(oid)
            if group is not None:
                duplicate['native_evidence']['double_door_group'] = group
        else:
            openings['native_aperture_' + oid] = aperture
    return openings, dict(schema='mp3d_native_opening_inventory_v1',
                          openings=openings, rejected_frames=rejected)


def native_directed_portal_program(opening, pose, normal=None, *, original=None):
    """Reuse generic direction/phase validation with witnessed native offsets."""
    if original is None:
        from rule_teacher import directed_portal_program as original
    program, direction = original(opening, pose, normal)
    distance = opening.get('native_evidence', {}).get('crossing_offset_m', .55)
    if distance == .55:
        return program, direction
    center, direction = np.asarray(opening['center']), np.asarray(direction)
    signed = float((np.asarray(pose[:2]) - center) @ direction)
    near = -min(.35, distance)
    goals = [(center + d * direction).tolist() for d in (near, 0., distance)
             if d > signed + (.05 if d == near else .015)]
    if not goals:
        raise TeacherRejected('rule: retained portal traversal already complete')
    return dict(status='ok', steps=[dict(kind='via', xy=goal) for goal in goals]), direction.tolist()


@_serialized_native_scope
@contextmanager
def native_portal_programs():
    """Install a native-only helper during synchronous candidate preparation."""
    import rule_teacher
    original = rule_teacher.directed_portal_program
    def scoped(opening, pose, normal=None):
        return native_directed_portal_program(opening, pose, normal, original=original)
    rule_teacher.directed_portal_program = scoped
    try:
        yield
    finally:
        rule_teacher.directed_portal_program = original


@_serialized_native_scope
@contextmanager
def native_opening_visibility(openings):
    """Reuse aperture visibility in the witnessed native floor's frame.

    The shared SAGE helper builds doorway boxes at z=0..2.3 and tests inner
    aperture rays at z=.6..1.8. Native upper floors require their annotation's
    actual vertical extent for projection and the same inner window translated
    by the measured navmesh floor. RGB/depth thresholds remain unchanged.
    """
    import rule_teacher
    project, depth_test = rule_teacher.project_bbox, rule_teacher.opening_depth_evidence

    def geometry(obj):
        opening = openings.get(obj.get('ins_id')) if obj.get('label') == 'doorway' else None
        matched_id = obj.get('ins_id') if opening is not None else None
        if opening is None and set(obj) == {'bbox_xyxy', 'z_range'} and obj.get('z_range') == [0, 2.3]:
            # connecting_portals uses this exact helper box without an ID.
            # Recover provenance only from a unique, equal verified strip;
            # ordinary object boxes cannot acquire doorway identity here.
            bbox = np.asarray(obj['bbox_xyxy'], float)
            matches = []
            for oid, candidate in openings.items():
                ends = np.asarray(candidate.get('ends', []), float)
                if (candidate.get('native_evidence') and ends.shape == (2, 2)
                        and bbox.shape == (4,) and np.isfinite(ends).all()
                        and np.allclose(bbox, np.r_[ends.min(axis=0), ends.max(axis=0)], rtol=0., atol=1e-9)):
                    matches.append((oid, candidate))
            if len(matches) > 1:
                raise TeacherRejected('native: anonymous portal box has ambiguous verified provenance')
            if matches:
                matched_id, opening = matches[0]
        if opening is None:
            return None
        evidence = opening['native_evidence']
        corners = np.asarray(evidence['frame_obb'].get('corners', []), float)
        if corners.shape != (8, 3) or not np.isfinite(corners).all():
            raise TeacherRejected('native: opening lacks exact annotated vertical bounds')
        floor = float(evidence['floor_height'])
        bounds = [float(corners[:, 2].min()), float(corners[:, 2].max())]
        if not math.isfinite(floor) or bounds[0] > floor + .6 or bounds[1] < floor + 1.8:
            raise TeacherRejected('native: annotated frame does not contain the standard inner aperture window')
        return dict(floor_height=floor, frame_z_range=bounds,
                    aperture_z_range=[floor + .6, floor + 1.8],
                    matched_native_opening_id=matched_id,
                    native_object_ids=evidence['native_object_ids'],
                    evidence='native frame OBB and measured navmesh floor; unchanged shared visibility thresholds')

    def projected(camera, camera_pos, yaw, obj):
        native = geometry(obj)
        value = project(camera, camera_pos, yaw,
                        dict(obj, z_range=native['frame_z_range']) if native else obj)
        if value is not None and native:
            value = dict(value, native_vertical_geometry=native)
        return value

    def tested(depth, camera, pose, obj, projection, *, camera_xyz=None):
        native = geometry(obj)
        if native:
            if camera_xyz is None:
                raise TeacherRejected('native: exact camera height is required for aperture depth')
            camera_xyz = [*camera_xyz[:2], float(camera_xyz[2]) - native['floor_height']]
        value = depth_test(depth, camera, pose, obj, projection, camera_xyz=camera_xyz)
        return dict(value, native_vertical_geometry=native) if native else value

    rule_teacher.project_bbox, rule_teacher.opening_depth_evidence = projected, tested
    try:
        yield
    finally:
        rule_teacher.project_bbox, rule_teacher.opening_depth_evidence = project, depth_test


def current_floor_pathfinder(previous, navmesh):
    """Opt-in reload of a known immutable production mesh for a new floor.

    External/injected PathFinders retain the original local path. An owned map
    with a load-time full-content binding may reload that same mesh through the
    original constructor, so the new floor can use verified resident workers.
    The caller must already have validated its actual floor-transition witness.
    """
    if previous is None:
        return None
    import os
    if os.environ.get('RULE_NATIVE_MANAGED_FLOOR_RELOAD', '0') != '1':
        return previous.pathfinder
    digest = getattr(previous, '_native_loaded_navmesh_sha256', None)
    if (type(previous) is not MP3DNavmeshMap
            or not getattr(previous, '_native_managed_pathfinder', False)
            or digest is None):
        return previous.pathfinder
    if Path(navmesh).resolve() != Path(previous.navmesh_path).resolve():
        raise TeacherRejected('native floor reload differs from the owned map source')
    if _navmesh_sha256(navmesh) != digest:
        raise TeacherRejected('native navmesh changed since the owned PathFinder load')
    return None


class MP3DRenderClient:
    """Use the existing stateless remote_habitat /render NPZ protocol."""

    def __init__(self, url, *, timeout=180):
        self.url, self.timeout = url.rstrip('/'), timeout
        self.sensors = None

    def bind_sensors(self, sensors):
        by_id = {s['uuid']: dict(s) for s in sensors}
        if set(by_id) != {'rgb', 'depth'}:
            raise TeacherRejected('native MP3D RGB/depth camera specifications unavailable')
        rgb, depth = by_id['rgb'], by_id['depth']
        for sensor in (rgb, depth):
            position, orientation = np.asarray(sensor['position'], float), np.asarray(sensor['orientation'], float)
            if (position.shape != (3,) or not np.isfinite(position).all()
                    or not np.allclose(position[[0, 2]], 0., atol=1e-6)
                    or position[1] <= 0 or not np.allclose(orientation, 0., atol=1e-6)):
                raise TeacherRejected('MP3D rule teacher requires upright, vertically offset RGB/depth cameras')
        for key in ('resolution', 'hfov', 'position', 'orientation'):
            if rgb[key] != depth[key]:
                raise TeacherRejected('MP3D RGB/depth camera geometry differs')
        if rgb['type'] != 'COLOR' or depth['type'] != 'DEPTH':
            raise TeacherRejected('MP3D native sensor types differ from RGB/metric depth')
        self.sensors = [rgb, depth]
        self.sensor_height = float(rgb['position'][1])
        height, width = map(int, rgb['resolution'])
        return CameraSpec(width=width, height=height, hfov_deg=float(rgb['hfov']))

    def render_one(self, scene, position, yaw, *, want_depth=True):
        if self.sensors is None:
            raise TeacherRejected('MP3D renderer must bind the actual simulator camera first')
        request_id = uuid.uuid4().hex
        scene = scene.removeprefix('rxr-')
        if not scene or Path(scene).name != scene:
            raise ValueError('invalid MP3D scene identity')
        position = np.asarray(position, float)
        if position.shape != (3,) or not np.isfinite(position).all() or not math.isfinite(yaw):
            raise ValueError('invalid MP3D render pose')
        payload = dict(request_id=request_id, scene=f'{scene}/{scene}.glb',
            position=habitat_position(position[:2], position[2] - self.sensor_height).tolist(),
            rotation=habitat_yaw_xyzw(yaw), sensors=self.sensors)
        # Reconstructing an observed quaternion/floor through Euler angles and
        # navmesh snapping introduces small float errors. At the actual pose,
        # preserve the native state exactly, so the observation guard compares
        # the same rendered camera frame rather than a nearby approximation.
        actual = getattr(self, 'actual_geometry', None)
        if actual is not None:
            native = np.asarray(actual['position'], float)
            actual_xy = [-native[2], -native[0]]
            actual_yaw = rotation_yaw(actual['rotation_xyzw'])
            error = math.atan2(math.sin(yaw - actual_yaw), math.cos(yaw - actual_yaw))
            if (np.allclose(position[:2], actual_xy, atol=1e-7, rtol=0)
                    and abs(position[2] - native[1] - self.sensor_height) <= 1e-7
                    and abs(error) <= 1e-7):
                payload['position'] = list(actual['position'])
                payload['rotation'] = list(actual['rotation_xyzw'])
        request = Request(self.url + '/render', data=json.dumps(payload).encode(),
                          headers={'Content-Type': 'application/json'})
        with build_opener(ProxyHandler({})).open(request, timeout=self.timeout) as response:
            body = response.read(32 * 1024 * 1024)
        with np.load(io.BytesIO(body), allow_pickle=False) as arrays:
            if str(arrays['request_id']) != request_id:
                raise RuntimeError('MP3D counterfactual render response identity mismatch')
            rgb, depth = arrays['rgb'].copy(), arrays['depth'].copy()
        h, w = self.sensors[0]['resolution']
        if rgb.shape != (h, w, 4) and rgb.shape != (h, w, 3):
            raise RuntimeError('MP3D renderer returned the wrong RGB shape')
        if rgb.dtype != np.uint8 or depth.shape not in ((h, w), (h, w, 1)):
            raise RuntimeError('MP3D renderer returned an incompatible RGB/depth frame')
        return Image.fromarray(rgb[:, :, :3]), depth.reshape(h, w) if want_depth else None


# Reuse the four restoring contexts. Only PrivilegedTeacher.compile (which
# uses explicit geometry and no rule_teacher parser/visibility globals) may
# run while they are suspended. Each actor retains its own context/PathFinder.
_NATIVE_PREPARE_SCOPE = ContextVar('native_prepare_scope', default=None)


class _NativePrepareScope:
    def __init__(self, owner, openings):
        self.owner, self.openings = owner, openings
        self.stack = None
        self.outer = _NATIVE_PREPARE_SCOPE.get()

    def enter(self):
        stack = ExitStack()
        try:
            for context in (native_room_grammar(), native_target_grammar(),
                            native_portal_programs(), native_opening_visibility(self.openings)):
                stack.enter_context(context)
        except BaseException:
            stack.close()
            raise
        self.stack = stack

    def close(self):
        if self.stack is not None:
            self.stack.close()
            self.stack = None

    @contextmanager
    def activated(self):
        token = _NATIVE_PREPARE_SCOPE.set(self)
        try:
            self.enter()
            yield
        finally:
            self.close()
            _NATIVE_PREPARE_SCOPE.reset(token)

    @contextmanager
    def geometry_only(self):
        # Restore every patched global BEFORE releasing the last lock. Reenter
        # all contexts BEFORE returning to any parser/visibility caller.
        self.close()
        try:
            yield
        finally:
            self.enter()


class MP3DRuleTeacher(RuleTeacher):
    def prepare_for_execution(self, **kwargs):
        state = super().prepare_for_execution(**kwargs)
        return self._bind_flat_native_witness(state, kwargs['scene'])

    def _bind_flat_native_witness(self, state, scene):
        if state.get('native_surface_contract'):
            return state
        # Validate the unchanged flat subdivision before selecting a candidate.
        # This is a plan/start witness, not proof of any executed waypoint.
        # FullRoutePolicy.commit still retains fresh simulator feedback.
        from native_surface_route import native_route_safe
        from agent_tools import SUBSTEP_M
        occ, _, _ = self.scene(scene)
        actual = np.asarray(self.actual_geometry()['position'], float)
        targets = np.asarray(state['route_target_world_poses'], float)
        if (actual.shape != (3,) or not np.isfinite(actual).all()
                or targets.shape != (state.get('route_prefix_steps', 4)*state['route_chunks']+1, 3)
                or not np.isfinite(targets).all()
                or math.dist(occ.world_xy(actual), targets[0, :2]) > .002
                or abs(actual[1]-occ.floor_height) > occ.floor_tolerance):
            raise TeacherRejected('native: flat route lacks the actual native start witness')
        planned = []
        for xy in targets[:, :2]:
            snapped = occ.snap_xy(*map(float, xy))
            if snapped is None:
                raise TeacherRejected('native: flat route target leaves the selected floor')
            point = occ.native_position(xy).astype(float)
            point[1] = snapped[1]
            planned.append(point)
        planned = np.asarray(planned)
        planned[0] = actual
        # Retain the flat planner's original extra clearance and floor bound;
        # do not switch to the terrain compiler or resample its XY labels.
        if not native_route_safe(occ.pathfinder, planned, radius=.07,
                                 height_tolerance=occ.floor_tolerance):
            raise TeacherRejected('native: flat route fails physical witness validation')
        state['route_target_native_positions'] = planned.tolist()
        state['execution_native_positions'] = [actual.tolist()]
        state['native_surface_contract'] = dict(schema='habitat_surface_route_v1',
            world_frame=occ.world_frame, radius_m=.07, probe_spacing_m=.025,
            executor_substep_m=SUBSTEP_M, height_tolerance_m=occ.floor_tolerance,
            planned_native_path=planned.tolist(),
            planned_native_path_m=float(np.linalg.norm(np.diff(planned, axis=0), axis=1).sum()),
            terrain_following=False, witness_source='actual native start and simulator waypoint feedback')
        return state

    def prepare_native_navigation(self, **kwargs):
        from native_stair_route import prepare_stair_route
        return prepare_stair_route(self, **kwargs)

    def retain_native_transition(self, state, execution):
        """Authorize a new floor slice only after a measured complete route."""
        points = np.asarray(state.get('execution_native_positions', []), float)
        actual = np.asarray(self.actual_geometry()['position'], float)
        if (not state.get('full_route_complete') or state.get('tracking_failure')
                or points.shape != (state.get('route_prefix_steps', 4)*state.get('route_chunks', 4)+1, 3)
                or not np.isfinite(points).all()
                or np.linalg.norm(points[-1]-actual) > .002):
            raise TeacherRejected('MP3D floor refresh lacks a verified actual surface transition')
        self._native_transition = dict(scene=self._current_scene, start=points[0].tolist(),
                                      end=points[-1].tolist(), execution=str(execution))

    def validate_first_navigation_motion(self, contract, graph):
        """Do not lower an unsupported first stair traversal to an XY ray.

        Approaching stairs, rotations and later plan stages retain the local
        supervision scope. A first stair flight is handled separately by the
        native compiler; unsupported transitions must never fall back to XY.
        """
        from rule_plan_chunks import segment_plan
        if graph is None or graph.get('original_call') != contract:
            raise TeacherRejected('native: first navigation motion requires the original language graph')
        fields = {f['field']: f for f in graph['fields']}
        first = None
        for index, text in enumerate(contract.get('plan', [])):
            if not text.strip():
                continue
            clauses = segment_plan(fields[f'plan[{index}]'])['clauses']
            if clauses:
                first = clauses[0]
            if first is not None:
                break
        if first is None or first['negated'] or first['kind'] == 'rotation':
            return
        if first['action'] not in {'go', 'move', 'walk', 'proceed', 'head', 'continue',
                                   'advance', 'navigate', 'follow', 'ascend', 'descend', 'climb'}:
            return
        text = first['text'].lower()
        stairs = r'(?:stair(?:s|case|way)?|steps)'
        vertical = (
            re.search(r'\b(?:up|down)\s+(?:(?:the|a|an|these|those)\s+)?'+stairs+r'\b', text)
            or re.search(r'\b(?:ascend|descend|climb)\b[^.;]*\b'+stairs+r'\b', text)
            or re.search(r'\b(?:go|move|walk|proceed|head|continue)\s+(?:straight\s+)?(?:upstairs|downstairs)\b', text)
            or re.search(r'\b(?:go|move|walk|proceed|head|continue)\b[^.;]*\b(?:upper|lower|next)\s+(?:floor|level)\b', text))
        if vertical:
            raise TeacherRejected('native: first navigation stage requests a floor transition; '
                                  'this adapter has no verified stair traversal compiler')

    def prepare_stair_state(self, occ, object_id, obj, actual_start, motion,
                            component, pose, minimum_chunks, exact_chunks, *, diagnostics=None, prefix_steps=4):
        import os
        from native_compile_worker import get_compile_pool, NativeCompileInfrastructureError
        from native_stair_route import prepare_stair_state
        args = (occ, object_id, obj, actual_start, motion, component, pose,
                minimum_chunks, exact_chunks)
        scope = _NATIVE_PREPARE_SCOPE.get()
        if (os.environ.get('RULE_NATIVE_STAIR_WORKERS', '0') == '1'
                and scope is not None and scope.owner is self and scope.outer is None):
            try:
                pool = get_compile_pool()
                if pool is not None:
                    with scope.geometry_only():
                        state, audit = pool.prepare_stair(*args, prefix_steps=prefix_steps)
                    if diagnostics is not None:
                        diagnostics.extend(audit)
                    return state
            except TeacherRejected as error:
                if diagnostics is not None:
                    diagnostics.extend(getattr(error, 'native_stair_audit', ()))
                raise
            except NativeCompileInfrastructureError as error:
                print(json.dumps(dict(event='native_stair_worker_fallback', error=str(error))))
        return prepare_stair_state(*args, diagnostics=diagnostics, prefix_steps=prefix_steps)

    def compile(self, program, pose, occ, objects, openings):
        from native_compile_worker import get_compile_pool, NativeCompileInfrastructureError
        scope = _NATIVE_PREPARE_SCOPE.get()
        if scope is None or scope.owner is not self or scope.outer is not None:
            return super().compile(program, pose, occ, objects, openings)
        try:
            pool = get_compile_pool()
            if pool is not None:
                with scope.geometry_only():
                    return pool.compile(program, pose, occ, objects, openings)
        except NativeCompileInfrastructureError as error:
            # All grammar helpers have been reentered before local fallback.
            print(json.dumps(dict(event='native_compile_worker_fallback', error=str(error))))
        return super().compile(program, pose, occ, objects, openings)

    def prepare(self, **kwargs):
        openings = self.scene(kwargs['scene'])[2]
        with _NativePrepareScope(self, openings).activated():
            state = super().prepare(**kwargs)
        output = Path(kwargs['output'])
        try:
            state = self._bind_flat_native_witness(state, kwargs['scene'])
        except TeacherRejected as exc:
            self.last_rejection = str(exc)
            accepted = output/'accepted.json'
            if accepted.exists():
                accepted.rename(output/'compiled-before-native-rejection.json')
            (output/'rejected.json').write_text(json.dumps(dict(
                reason=str(exc), original_call=kwargs['contract']), ensure_ascii=False, indent=2))
            raise
        (output/'accepted.json').write_text(json.dumps(state, ensure_ascii=False, indent=2))
        return state

    """Apply unchanged local call grounding/compilation to actual MP3D state."""

    def __init__(self, *, backend, dataset_source=None, scenes_root=SCENES_ROOT,
                 house_root=HOUSE_ROOT, split='train', instruction_overrides=None, **kwargs):
        self.backend = backend
        # The collector verifies synthesis provenance before constructing this
        # instruction-only mapping. Native episode goals/routes never enter it.
        self.instruction_overrides = instruction_overrides or {}
        self.scenes_root, self.house_root = Path(scenes_root), Path(house_root)
        self._current_scene = None
        dataset = MP3DInstructionSet(backend, split=split, source=dataset_source)
        renderer = MP3DRenderClient(kwargs['render_url'])
        super().__init__(split=split, dataset=dataset, renderer=renderer,
                         camera=CameraSpec(width=640, height=480, hfov_deg=120), **kwargs)

    def actual_geometry(self):
        value = self.render_client.last_info.get('oracle_habitat_geometry')
        if not isinstance(value, dict) or value.get('schema') != 'habitat_actual_agent_geometry_v1':
            raise TeacherRejected('MP3D backend lacks exact evaluator-only native pose/camera evidence')
        position = np.asarray(value['position'], float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise TeacherRejected('invalid actual MP3D simulator position')
        rotation_yaw(value['rotation_xyzw'])
        return value

    def reset_environment(self, **kwargs):
        task = self.task(kwargs['scene'], str(kwargs['episode_id']))
        requested = kwargs.get('instruction', task.instruction)
        allowed = self.instruction_overrides.get((kwargs['scene'], str(kwargs['episode_id'])))
        if requested != task.instruction and requested != allowed:
            raise TeacherRejected('MP3D instruction lacks verified collector provenance')
        # Reset the unchanged official episode/start through the strict native
        # Gym API. The slow loop reads its separately verified case wording.
        kwargs['instruction'] = task.instruction
        obs, info = super().reset_environment(**kwargs)
        geometry = self.actual_geometry()
        if np.linalg.norm(np.asarray(geometry['position']) - task.native_start_position) > .02:
            raise TeacherRejected('MP3D reset actual 3D position differs from native task start')
        actual_yaw = rotation_yaw(geometry['rotation_xyzw'])
        expected_yaw = rotation_yaw(task.native_start_rotation)
        error = math.atan2(math.sin(actual_yaw - expected_yaw), math.cos(actual_yaw - expected_yaw))
        if abs(error) > math.radians(.5):
            raise TeacherRejected('MP3D reset actual heading differs from native task start')
        if info.get('backend') != self.backend:
            raise TeacherRejected('MP3D reset substituted dataset backend')
        self.camera = self.renderer.bind_sensors(geometry.get('sensors', []))
        self._current_scene = kwargs['scene']
        self._native_transition = None
        self.maps.clear()
        self.semantic_scenes.clear()
        self.scene(self._current_scene)
        return obs, info

    def scene(self, scene):
        if scene != self._current_scene:
            raise TeacherRejected('MP3D teacher scene must match the actual reset scene')
        geometry = self.actual_geometry()
        previous = self.maps.get(scene)
        pathfinder = previous[0].pathfinder if previous else None
        if previous is not None and abs(float(geometry['position'][1])-previous[0].floor_height) > previous[0].floor_tolerance:
            witness = getattr(self, '_native_transition', None) or {}
            if (witness.get('scene') != scene
                    or np.linalg.norm(np.asarray(witness.get('end', [math.inf]*3))-geometry['position']) > .002
                    or abs(float(witness.get('start', [0., math.inf, 0.])[1])-previous[0].floor_height) > .06):
                raise TeacherRejected('MP3D changed floor lacks a measured execution witness')
            # Reuse the immutable native mesh; select semantic rooms and door
            # openings using measured Y rather than the desired endpoint.
            native_scene = scene.removeprefix('rxr-')
            navmesh = self.scenes_root / native_scene / f'{native_scene}.navmesh'
            pathfinder = current_floor_pathfinder(previous[0], navmesh)
            self.maps.pop(scene)
            self.semantic_scenes.pop(scene, None)
        if scene not in self.maps:
            from mp3d_teacher_semantics import load_mp3d_scene
            native_scene = scene.removeprefix('rxr-')
            navmesh = self.scenes_root / native_scene / f'{native_scene}.navmesh'
            occ = MP3DNavmeshMap(navmesh, geometry['position'], pathfinder=pathfinder)
            semantic_map = load_mp3d_scene(native_scene, root=self.house_root)
            # Native region polygons on another floor must not ground a local
            # call solely because their XY projection overlaps this floor.
            semantic = semantic_map.make_semantic_scene(occ=occ)
            openings, inventory = native_doorframe_openings(semantic, occ)
            semantic.openings = openings
            semantic.native_opening_inventory = inventory
            self.semantic_scenes[scene] = semantic
            self.maps[scene] = occ, semantic.objects, openings
        occ = self.maps[scene][0]
        if abs(float(geometry['position'][1]) - occ.floor_height) > occ.floor_tolerance:
            raise TeacherRejected('MP3D map does not match the measured current surface')
        return self.maps[scene]

    def semantic_scene(self, scene, objects, openings):
        return self.semantic_scenes[scene]

    def camera_position(self, pose):
        occ = self.scene(self._current_scene)[0]
        actual = self.actual_geometry()
        self.renderer.actual_geometry = actual
        native = np.asarray(actual['position'], float)
        if np.allclose(pose[:2], [-native[2], -native[0]], atol=1e-7, rtol=0):
            return float(pose[0]), float(pose[1]), float(native[1]) + self.renderer.sensor_height
        snapped = occ.snap_xy(float(pose[0]), float(pose[1]))
        if snapped is None:
            raise TeacherRejected('MP3D planned viewpoint is off the actual flat navmesh floor')
        return float(pose[0]), float(pose[1]), snapped[1] + self.renderer.sensor_height
