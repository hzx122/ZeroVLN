"""Default SAGE geometry: executed 3-D support, exact target, no XY fallback.

Geometry manifests are immutable USD/Recast evidence, not proof that the source
benchmark is physically executable. Ambiguous source/target support is an error.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from .navmesh_support import FloorUnresolved, SourceCaseGeometryUnsupported, NavMeshSupport, digest, checked_digest, to_nav

SCHEMA = 'sage-designated-target-navmesh-v2'
DEFAULT_ROOT = '/data/zerovln/artifacts/sage-physical-navigation-v2/navmeshes-structural-v1'


def identity():
    """Common scoring identity; per-scene and per-target evidence stays separate."""
    from . import benchmark_protocol
    sources = {name: checked_digest(Path(__file__).with_name(name)) for name in
               ('physical_navigation.py', 'navmesh_support.py', 'goal_region.py', 'source_objects.py')}
    from .source_objects import source_object_identity
    spec = dict(schema=SCHEMA, sources=sources, expansion_radius_m=1.,
                benchmark=benchmark_protocol.identity() if benchmark_protocol.enabled() else None,
                source_references=source_object_identity())
    return dict(schema=SCHEMA, object_success_schema=SCHEMA,
                contract_sha256=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(),
                expansion_radius_m=1., requires_agent_stop=False,
                distance_metric='supported_target_navmesh_geodesic')


def training_geometry_contract(cases):
    """Freeze scene manifests as well as scoring code across exact resume.

    Missing assets stay explicit; reset/preflight must reject them, never fill
    them with an XY simulator. Non-SAGE training keeps its existing contract.
    """
    scenes = sorted({str(row['scene']) for row in cases
                     if str(row.get('scene', '')).startswith('sage-')})
    if not scenes:
        return None
    from .goal_region import official_success_contract_identity
    from .source_objects import source_object_identity
    result = dict(official_success_contract_identity(), mode='physical' if enabled() else 'legacy_xy')
    if enabled():
        result['source_references'] = source_object_identity()
        root = Path(os.environ.get('SAGE_NAVMESH_ROOT', DEFAULT_ROOT)).resolve()
        manifests = {}
        for scene in scenes:
            path = root / scene.removeprefix('sage-') / 'manifest.json'
            manifests[scene] = checked_digest(path) if path.is_file() else None
        result.update(navmesh_root=str(root), manifests=manifests)
    return result


def enabled():
    mode = os.environ.get('SAGE_GEOMETRY_MODE', 'physical')
    if mode not in ('physical', 'legacy_xy'):
        raise ValueError('SAGE_GEOMETRY_MODE must be physical or explicit legacy_xy')
    return mode == 'physical'


def load_navigation(scene, radius):
    root = Path(os.environ.get('SAGE_NAVMESH_ROOT', DEFAULT_ROOT))
    manifest = root / str(scene).removeprefix('sage-') / 'manifest.json'
    if not manifest.is_file():
        raise FloorUnresolved(f'missing 3-D geometry: {manifest}; prepare audited navmeshes before running, no XY fallback')
    nav = NavMeshSupport(manifest, scene=scene)
    if abs(nav.radius - radius) > 1e-6:
        raise FloorUnresolved(f'navmesh body radius {nav.radius} != configured radius {radius}')
    return nav


def resolve_task(nav, task, labels):
    """Shared reset/preflight admission, preserving the original case and XY."""
    from . import benchmark_protocol
    from .source_objects import require_source_references
    if os.environ.get('SAGE_SOURCE_OBJECT_MAP'):
        require_source_references(task, labels)
    audit = nav.resolve_start(task, labels)
    target_id = str(task.goal_label or '').strip()
    if benchmark_protocol.enabled():
        designated = str(benchmark_protocol.bundle()[2][task.task_id]['end_item']).strip()
        if designated != target_id:
            raise FloorUnresolved('dataset goal_label differs from benchmark end_item')
    if target_id not in labels:
        raise SourceCaseGeometryUnsupported(f'missing designated target {target_id!r}')
    target_info = dict(labels[target_id])
    if benchmark_protocol.enabled():
        _, target_info['official_bbox_xyxy'] = benchmark_protocol.designated_bbox(task.scene, task.task_id)
    target = TargetRegion(nav, target_id, target_info)
    shortest = target.distance(nav.position)
    if shortest is None:
        raise SourceCaseGeometryUnsupported('designated target has no connected 3-D route from the original start')
    return audit, target, shortest


def clip_triangle(triangle, bounds):
    """Sutherland-Hodgman clipping against the task's closed XY rectangle."""
    points = [np.asarray(p, dtype=float) for p in triangle]
    for axis, bound, sign in ((0,bounds[0],1),(0,bounds[2],-1),
                              (1,bounds[1],1),(1,bounds[3],-1)):
        output = []
        for a,b in zip(points, points[1:]+points[:1]):
            da,db = sign*(a[axis]-bound),sign*(b[axis]-bound)
            if da >= 0:
                output.append(a)
            if (da >= 0) != (db >= 0):
                output.append(a+(b-a)*(da/(da-db)))
        points = output
        if not points:
            break
    return np.asarray(points)


def polygon_contains(polygon, xy):
    edges = np.roll(polygon,-1,axis=0)-polygon
    relative = np.asarray(xy)-polygon
    cross = edges[:,0]*relative[:,1]-edges[:,1]*relative[:,0]
    return bool(np.all(cross >= -1e-8) or np.all(cross <= 1e-8))


class TargetRegion:
    """Expanded designated bbox intersected with an unambiguous support layer.

    Grounded objects use their base as evidence. For other objects (e.g. lamps),
    accept only a unique underlying navigation layer; never choose the layer
    nearest the agent or use trajectory z=0.5. Ambiguity requires asset repair.
    """
    def __init__(self, nav, target_id, obj):
        from .goal_region import official_object_arrival, semantic_family
        self.nav, self.target_id = nav, str(target_id)
        self.target_ins_id = obj.get('ins_id')
        self.xy_detail = official_object_arrival(obj, (0., 0.))
        # Physical support and object geometry must share the USD world frame.
        # Frozen raster-map bounds are retained only for the separate XY
        # diagnostic: some released maps have an offset from the source USD.
        physical_obj = dict(obj)
        physical_obj.pop('official_bbox_xyxy', None)
        self.region_detail = official_object_arrival(physical_obj, (0., 0.))
        self.bbox = self.region_detail['expanded_bbox_xyxy']
        z_range = np.asarray(obj.get('z_range', []), dtype=float)
        if z_range.shape != (2,) or not np.isfinite(z_range).all():
            raise SourceCaseGeometryUnsupported(f'target {target_id} lacks vertical geometry')
        base = float(z_range[0])
        grounded = semantic_family(obj.get('label', '')) in {
            'chair', 'table', 'sofa', 'cabinet', 'plant', 'bed',
            'single_bed', 'double_bed', 'bunk_bed'}
        structural_height = nav.object_support_height(obj)
        patches = []
        for tri in nav.triangles:
            if np.any(tri[:, :2].max(axis=0) < np.asarray(self.bbox[:2])) or np.any(tri[:, :2].min(axis=0) > np.asarray(self.bbox[2:])):
                continue
            coords = clip_triangle(tri[:, :2], self.bbox)
            if len(coords) < 3:
                continue
            area = abs(np.dot(coords[:,0],np.roll(coords[:,1],-1))-
                       np.dot(coords[:,1],np.roll(coords[:,0],-1)))/2
            if area < 1e-9:
                continue
            plane = np.linalg.solve(np.column_stack((tri[:, :2], np.ones(3))), tri[:, 2])
            zs = coords @ plane[:2] + plane[2]
            if zs.max() > base + .10:
                continue
            if structural_height is None and grounded and np.max(np.abs(zs-base)) > .35:
                continue
            if structural_height is not None and np.max(np.abs(zs-structural_height)) > .15:
                continue
            patches.append((coords, plane, float(zs.mean())))
        levels = []
        for z in sorted(p[2] for p in patches):
            if not levels or z-levels[-1][-1] > .15:
                levels.append([z])
            else:
                levels[-1].append(z)
        if len(levels) != 1 or max(levels[0])-min(levels[0]) > .35:
            raise SourceCaseGeometryUnsupported(f'target {target_id}: ambiguous or absent support layers {[(min(x),max(x)) for x in levels]}')
        self.patches = patches
        points = []
        for polygon, plane, _ in patches:
            coords = list(polygon)+[polygon[0]]
            for a, b in zip(coords, coords[1:]):
                count = max(1, math.ceil(math.dist(a, b)/.25))
                for f in np.arange(count)/count:
                    xy = np.asarray(a)*(1-f)+np.asarray(b)*f
                    points.append([*xy, float(xy @ plane[:2]+plane[2])])
            p = polygon.mean(axis=0)
            points.append([*p, float(p @ plane[:2]+plane[2])])
        self.points = np.unique(np.round(points, 5), axis=0)
        self.hash = hashlib.sha256(json.dumps(dict(schema=SCHEMA,
            navmesh=digest(nav.manifest_path), target_id=self.target_id,
            bbox=self.bbox, z_range=z_range.tolist(), samples=self.points.tolist()),sort_keys=True).encode()).hexdigest()
        self.structural_support_height = structural_height

    def contains(self, xyz):
        p = np.asarray(xyz, dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all():
            return False
        if not self.nav.pathfinder.is_navigable(to_nav(p), .05):
            return False
        return any(polygon_contains(poly, p[:2]) and abs(p[2]-(p[:2] @ plane[:2]+plane[2])) <= .06
                   for poly, plane, _ in self.patches)

    def distance(self, xyz):
        if self.contains(xyz):
            return 0.
        query = self.nav.hs.MultiGoalShortestPath()
        query.requested_start = to_nav(xyz)
        query.requested_ends = np.asarray([to_nav(p) for p in self.points])
        if not self.nav.pathfinder.find_path(query):
            return None
        distance = float(query.geodesic_distance)
        return distance if math.isfinite(distance) else None

    def score(self, xyz):
        x,y,z = map(float, xyz)
        b = self.xy_detail['expanded_bbox_xyxy']
        xy_success = b[0] <= x <= b[2] and b[1] <= y <= b[3]
        return dict(self.region_detail, schema=SCHEMA, available=True,
            success=self.contains(xyz), strict_target_success=xy_success,
            designated_target_xy_success=xy_success, target_object_id=self.target_id,
            target_ins_id=self.target_ins_id,
            position_xy=[x,y], ground_position=[x,y,z],
            official_position=[-x,-y,z], contract_sha256=identity()['contract_sha256'],
            target_contract_sha256=self.hash, navmesh_manifest_path=str(self.nav.manifest_path),
            navmesh_manifest_sha256=checked_digest(self.nav.manifest_path),
            designated_target_xy_bbox_xyxy=b,
            structural_support_height_m=self.structural_support_height,
            metric_semantics='designated_target_supported_3d_arrival')
