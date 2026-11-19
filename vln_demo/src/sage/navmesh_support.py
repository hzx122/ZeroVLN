"""Height-aware SAGE physics shared by offline teachers and future adapters.

No task resampling, XY snapping or cross-floor teleportation. Source trajectory
z=0.5 is not floor evidence; resolve the original XY against a nearby source
object and the actual navigation surfaces, or explicitly refuse the task.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np


class FloorUnresolved(ValueError):
    pass


class SourceCaseGeometryUnsupported(FloorUnresolved):
    """The unchanged released case has no unambiguous physical support.

    Missing assets, invalid runtime bindings and execution errors retain the
    generic FloorUnresolved type and must not become source-case abstentions.
    """
    pass


def to_nav(xyz):
    x, y, z = xyz
    return np.asarray([x, z, -y], dtype=np.float32)


def from_nav(position):
    x, z, neg_y = position
    return np.asarray([x, -neg_y, z], dtype=float)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=2048)
def _file_digest_at_stat(path, size, mtime_ns, ctime_ns):
    return digest(path)


def checked_digest(path):
    """Reuse immutable asset hashes; invalidate if the on-disk file changes."""
    path = Path(path).resolve()
    stat = path.stat()
    return _file_digest_at_stat(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def floor_contract(nav, audit, *, scene, episode, reset_pose, reset_ground_z, source_reference=None):
    """Private reset evidence, carried with saved cases and complete anchors."""
    if (np.asarray(reset_pose).shape != (3,) or not np.isfinite(reset_pose).all()
            or not np.isfinite(reset_ground_z)
            or not np.allclose(reset_pose[:2], audit['source_xy'], atol=1e-7, rtol=0)
            or abs(float(reset_ground_z)-audit['ground_z_m']) > .06):
        raise FloorUnresolved('recorded reset differs from resolved source floor')
    return dict(schema='sage-executed-floor-v1', scene=scene, episode=str(episode),
        **({'source_references': source_reference} if source_reference is not None else {}),
        manifest_path=str(nav.manifest_path), manifest_sha256=audit['manifest_sha256'],
        start_audit=dict(audit), reset_pose_xy_yaw_deg=list(reset_pose),
        reset_ground_z_m=float(reset_ground_z), quality_approved=False)


def verify_floor_contract(proof, task, objects, *, scene, episode, cache=None):
    """Recheck a saved source reset without relying on the auditor's env vars."""
    if (proof.get('schema') != 'sage-executed-floor-v1' or proof.get('scene') != scene
            or proof.get('episode') != str(episode) or proof.get('quality_approved') is not False):
        raise FloorUnresolved('invalid saved SAGE floor identity')
    path = Path(proof['manifest_path']).resolve()
    expected = proof['manifest_sha256']
    if digest(path) != expected:
        raise FloorUnresolved('saved SAGE floor manifest changed')
    key = (str(path), expected, scene)
    nav = cache.get(key) if cache is not None else None
    if nav is None:
        nav = NavMeshSupport(path, scene=scene)
        if cache is not None:
            cache[key] = nav
    # Recheck inputs even when the geometry object was cached for another case.
    for name in ('usd', 'navmesh'):
        if digest(nav.manifest[name+'_path']) != nav.manifest[name+'_sha256']:
            raise FloorUnresolved('saved SAGE '+name+' source changed')
    if proof.get('source_references') is not None:
        from .source_objects import load_source_label_index, require_source_references
        from .datasets import SageSet
        objects = load_source_label_index(SageSet('train').scene_dir(scene),
                                         identity=proof['source_references'])
        require_source_references(task, objects)
    audit = nav.resolve_start(task, objects)
    if audit != proof['start_audit']:
        raise FloorUnresolved('saved SAGE start floor proof differs from source')
    checked = floor_contract(nav, audit, scene=scene, episode=episode,
        reset_pose=proof['reset_pose_xy_yaw_deg'], reset_ground_z=proof['reset_ground_z_m'],
        source_reference=proof.get('source_references'))
    if checked != proof:
        raise FloorUnresolved('saved SAGE reset evidence differs')
    return nav, audit


class NavMeshSupport:
    def __init__(self, manifest, *, scene=None):
        import habitat_sim as hs
        self.hs = hs
        self.manifest_path = Path(manifest).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        m = self.manifest
        if m['schema'] != 'sage-usd-navmesh-v1' or m['coordinate_transform'] != 'habitat=[sage.x,sage.z,-sage.y]':
            raise ValueError('unsupported SAGE navmesh contract')
        if scene is not None and m['scene'] != str(scene).removeprefix('sage-'):
            raise ValueError('navigation seed belongs to another scene')
        for key in ('navmesh', 'usd'):
            if digest(m[key+'_path']) != m[key+'_sha256']:
                raise ValueError(key+' source hash mismatch')
        self.radius = float(m['settings']['agent_radius'])
        self.pathfinder = hs.PathFinder()
        if not self.pathfinder.load_nav_mesh(m['navmesh_path']):
            raise ValueError('failed to load SAGE navmesh')
        vertices = np.asarray(self.pathfinder.build_navmesh_vertices(), dtype=float)
        indices = np.asarray(self.pathfinder.build_navmesh_vertex_indices(), dtype=int).reshape(-1, 3)
        self.triangles = vertices[indices][:, :, [0, 2, 1]] * np.array([1., -1., 1.])
        self.structure = None
        if 'structure_path' in m:
            if checked_digest(m['structure_path']) != m['structure_sha256']:
                raise ValueError('structural support source hash mismatch')
            self.structure = np.load(m['structure_path'], allow_pickle=False)
            if self.structure.ndim != 3 or self.structure.shape[1:] != (3, 3) or not np.isfinite(self.structure).all():
                raise ValueError('invalid architectural support triangles')
        self.position = None

    def surface_heights(self, x, y):
        return self._surface_heights(self.triangles, x, y)

    @staticmethod
    def _surface_heights(tri, x, y):
        a, b, c = tri[:, 0, :2], tri[:, 1, :2], tri[:, 2, :2]
        ab, ac, q = b-a, c-a, np.asarray([x, y])-a
        det = ab[:, 0]*ac[:, 1]-ab[:, 1]*ac[:, 0]
        nonzero = np.abs(det) > 1e-10
        safe = np.where(nonzero, det, 1.)
        u = (q[:, 0]*ac[:, 1]-q[:, 1]*ac[:, 0])/safe
        v = (ab[:, 0]*q[:, 1]-ab[:, 1]*q[:, 0])/safe
        inside = nonzero & (u >= -1e-7) & (v >= -1e-7) & (u+v <= 1+1e-7)
        heights = tri[:, 0, 2]+u*(tri[:, 1, 2]-tri[:, 0, 2])+v*(tri[:, 2, 2]-tri[:, 0, 2])
        return sorted(set(round(float(z), 4) for z in heights[inside]))

    def object_support_height(self, info):
        """Architectural surface directly below the object's footprint.

        Multiple footprint probes must agree. Never use the agent position or
        reference endpoint to choose an object's floor, or treat a table as a
        building storey. No claim is made when source structure is unavailable.
        """
        structure = getattr(self, 'structure', None)
        z_range = info.get('z_range')
        if structure is None or z_range is None or len(z_range) != 2:
            return None
        base = float(z_range[0])
        x0, y0, x1, y1 = info['bbox_xyxy']
        # Wall caps and window sills also belong to SM_floorplan. A room
        # support must additionally have native standing area near the object;
        # a downward ray hitting a narrow sill alone is not floor evidence.
        tri = self.triangles
        nearby = tri[(tri[:, :, 0].max(axis=1) >= x0-1.) &
                     (tri[:, :, 0].min(axis=1) <= x1+1.) &
                     (tri[:, :, 1].max(axis=1) >= y0-1.) &
                     (tri[:, :, 1].min(axis=1) <= y1+1.)]
        levels = []
        for u, v in ((.5,.5),(.2,.2),(.2,.8),(.8,.2),(.8,.8)):
            heights = self._surface_heights(structure, x0+u*(x1-x0), y0+v*(y1-y0))
            below = [z for z in heights if z <= base+.001 and
                     any(np.max(np.abs(t[:, 2]-z)) <= .15 for t in nearby)]
            if below:
                levels.append(max(below))
        if len(levels) < 3 or max(levels)-min(levels) > .15:
            return None
        return float(np.median(levels))

    def resolve_start(self, task, objects):
        from .goal_region import semantic_family
        x, y = float(task.start['x']), float(task.start['y'])
        heights = self.surface_heights(x, y)
        info = objects.get(str(task.start_label or '').strip())
        anchor = None
        if info is not None:
            family = semantic_family(info.get('label', ''))
            box = info['bbox_xyxy']
            distance = np.hypot(max(box[0]-x, 0., x-box[2]), max(box[1]-y, 0., y-box[3]))
            if family in {'chair', 'table', 'sofa', 'cabinet', 'plant', 'bed',
                          'single_bed', 'double_bed', 'bunk_bed'} and distance <= 1.5:
                z = info.get('z_range')
                if z is not None and len(z) == 2:
                    anchor = float(z[0])
            structural_anchor = self.object_support_height(info)
            if structural_anchor is not None and distance <= 1.5:
                anchor = structural_anchor
        candidates = heights if anchor is None else [z for z in heights if abs(z-anchor) <= .35]
        # Coplanar Recast triangle boundaries may differ a few millimetres.
        unique = []
        for z in candidates:
            if not unique or z-unique[-1] > .05:
                unique.append(z)
        audit = dict(schema='sage-navmesh-start-v1', source_xy=[x, y],
                     source_z=task.start.get('z'), surface_heights_m=heights,
                     source_anchor_base_m=anchor, candidates_m=unique,
                     changes_source_xy=False, manifest_sha256=digest(self.manifest_path))
        if len(unique) != 1:
            audit['reason'] = 'source_floor_ambiguous' if unique else 'source_xy_not_on_supported_floor'
            raise SourceCaseGeometryUnsupported(json.dumps(audit, sort_keys=True))
        z = unique[0]
        point = to_nav([x, y, z])
        if not self.pathfinder.is_navigable(point, .05):
            audit['reason'] = 'original_source_pose_not_navigable'
            raise SourceCaseGeometryUnsupported(json.dumps(audit, sort_keys=True))
        self.position = np.asarray([x, y, z], dtype=float)
        audit.update(ground_z_m=z, reason='source_floor_resolved', quality_approved=False)
        return audit

    def ground_z(self, x, y):
        if self.position is None:
            raise FloorUnresolved('source floor has not been resolved')
        heights = self.surface_heights(x, y)
        if not heights:
            raise FloorUnresolved('no navigable support at requested XY')
        z = min(heights, key=lambda z: abs(z-self.position[2]))
        if abs(z-self.position[2]) > .35:
            raise FloorUnresolved('support requires an unexecuted floor transition')
        return z

    def diagnostic_stance(self, xyz, *, height_tolerance=.15, max_xy=1.):
        """Nearest supported XY on the requested floor, for diagnostics only.

        Habitat's unconstrained snap_point can choose a chair top. Project
        onto navigation triangle edges, then enforce the requested height.
        This method never changes the source pose or the active position.
        """
        xyz = np.asarray(xyz, dtype=float)
        candidates = []
        for z in self.surface_heights(*xyz[:2]):
            candidates.append(np.array([*xyz[:2], z]))
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = self.triangles[:, i], self.triangles[:, j]
            edge = b-a
            denominator = np.sum(edge[:, :2]**2, axis=1)
            t = np.clip(np.sum((xyz[:2]-a[:, :2])*edge[:, :2], axis=1) /
                        np.maximum(denominator, 1e-12), 0., 1.)
            candidates.extend(a+edge*t[:, None])
        candidates = np.asarray(candidates)
        distances = np.linalg.norm(candidates[:, :2]-xyz[:2], axis=1)
        valid = (np.abs(candidates[:, 2]-xyz[2]) <= height_tolerance) & (distances <= max_xy)
        indices = np.flatnonzero(valid)
        for index in indices[np.argsort(distances[indices])]:
            point = candidates[index]
            if self.pathfinder.is_navigable(to_nav(point), .05):
                return point.copy()
        return None

    def step(self, target_xy, *, slide=False):
        if self.position is None:
            raise FloorUnresolved('source floor has not been resolved')
        end = to_nav([*target_xy, self.position[2]])
        start = to_nav(self.position)
        call = self.pathfinder.try_step if slide else self.pathfinder.try_step_no_sliding
        new = from_nav(call(start, end))
        if not np.isfinite(new).all():
            raise FloorUnresolved('navigation step returned no physical support')
        settings = self.manifest['settings']
        slope = np.tan(np.deg2rad(settings.get('agent_max_slope', 45.)))
        horizontal = float(np.linalg.norm(new[:2]-self.position[:2]))
        if abs(new[2]-self.position[2]) > horizontal*slope + settings.get('agent_max_climb', .20) + .05:
            raise FloorUnresolved('navigation step requires a discontinuous vertical jump')
        self.position = new
        return new.copy()

    def path(self, start_xyz, end_xyz):
        query = self.hs.ShortestPath()
        query.requested_start = to_nav(start_xyz)
        query.requested_end = to_nav(end_xyz)
        if not self.pathfinder.find_path(query):
            return None
        return np.asarray([from_nav(p) for p in query.points])
