"""Opt-in open-loop floor maps over the shared native collision adapter.

Plans stay on the resolved floor. A stair/other-floor route requires a future
3-D teacher compiler; it must never be flattened into a valid XY label.
"""
from pathlib import Path

import numpy as np

from .navmesh_support import NavMeshSupport, FloorUnresolved, to_nav


def manifest_for(root, scene):
    return Path(root) / str(scene).removeprefix('sage-') / 'manifest.json'


def floor_structure(structure, floor):
    """Keep only geometry with an explicit vertical witness for this floor."""
    def supported(item, key):
        points = np.asarray(item.get('profile', []), float)
        if points.ndim != 2 or points.shape[1] < 3 or not len(points):
            return False
        lo, hi = points[:, 2].min(), points[:, 2].max()
        return abs(lo-floor) <= .35 if key == 'rooms' else lo <= floor+.35 and hi >= floor+1.
    result = dict(structure)
    for key in ('rooms', 'holes', 'walls'):
        result[key] = [item for item in structure.get(key, []) if supported(item, key)]
    return result


def floor_objects(objects, floor):
    from .goal_region import object_height_relation
    return {oid: obj for oid, obj in objects.items()
            if object_height_relation(obj, floor)['compatible']}


def floor_map(nav, audit, *, mpp=.05):
    # Lazy import keeps the ordinary SAGE backend independent of the teacher
    # and reuses its established grid/native-segment contract.
    from mp3d_rule_teacher import MP3DNavmeshMap

    class SageNavmeshMap(MP3DNavmeshMap):
        preserve_source_xy = True

        def ground_z(self, x, y, default=None):
            if np.allclose([x, y], nav.position[:2], atol=1e-7, rtol=0):
                return float(nav.position[2])
            point = self.snap_xy(float(x), float(y))
            if point is None:
                raise FloorUnresolved('no support on the active SAGE floor')
            return float(point[1])

        def try_step(self, x0, y0, x1, y1, radius_m=0., slide=True):
            """Execute on the connected 3-D surface, retaining actual height.

            The floor raster is a planning view, not a physical barrier at a
            staircase. Never choose a surface by unconstrained XY snapping.
            """
            before = nav.position.copy()
            if not np.allclose([x0, y0], before[:2], atol=1e-6, rtol=0):
                raise FloorUnresolved('SAGE simulator XY differs from native physical position')
            new = nav.step([x1, y1], slide=slide)
            clearance = float(nav.pathfinder.distance_to_closest_obstacle(to_nav(new), 2.))
            if not np.isfinite(clearance) or clearance <= radius_m:
                nav.position = before
                return x0, y0, True
            requested_distance = float(np.hypot(x1-x0, y1-y0))
            tolerance = min(.015, .05*requested_distance+1e-5)
            collided = float(np.linalg.norm(new[:2]-[x1, y1])) > tolerance
            return float(new[0]), float(new[1]), collided

    occ = SageNavmeshMap(nav.manifest['navmesh_path'], to_nav(nav.position),
                         pathfinder=nav.pathfinder, world_frame='sage', mpp=mpp)
    occ.floor_audit = dict(audit)
    occ.sage_navmesh_support = nav
    occ.navmesh_manifest_sha256 = audit['manifest_sha256']
    return occ
