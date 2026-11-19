"""Build a height-preserving navigation seed from released USD collisions.

Reuse build_usd_collision's source resolution and world transforms and hsim's
Recast path. Runtime needs Habitat only, never USD. Building a mesh is not a
quality admission: original starts and executed labels still need auditing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .build_usd_collision import _asset_path, _load_usd, _world_points, DEFAULT_USD_ROOT


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def world_mesh(path, *, structure_only=False):
    Usd, UsdGeom = _load_usd()
    stage = Usd.Stage.Open(str(path))
    if not stage or UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError('expected Z-up USD')
    if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
        raise ValueError('USD units must be metres; unverified scale is forbidden')
    xforms = UsdGeom.XformCache()
    vertices, faces, offset = [], [], 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if structure_only and prim.GetName() != 'SM_floorplan':
            continue
        if prim.GetAttribute('physics:collisionEnabled').Get() is False:
            continue
        mesh = UsdGeom.Mesh(prim)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        if np.any(counts != 3):
            raise ValueError('nontriangular collision faces require validated triangulation: ' + str(prim.GetPath()))
        pts = _world_points(mesh, xforms)
        ind = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
        if not len(pts):
            continue
        if not np.isfinite(pts).all() or np.any(ind < 0) or np.any(ind >= len(pts)):
            raise ValueError('invalid collision mesh')
        # Proper rotation: SAGE Z-up -> Habitat Y-up, no reflection/XY remap.
        vertices.append(pts[:, [0, 2, 1]] * np.array([1., 1., -1.]))
        faces.append(ind + offset)
        offset += len(pts)
    if not vertices:
        raise ValueError('empty collision payload')
    return np.concatenate(vertices).astype('<f4'), np.concatenate(faces).astype('<i4')


def write_structure(source, output):
    """Freeze upward architectural support faces, excluding furniture tops.

    These are source surfaces, not a generated floor label. Runtime intersects
    them vertically below the object and still requires native navigability.
    """
    vertices, faces = world_mesh(source, structure_only=True)
    triangles = vertices[faces][:, :, [0, 2, 1]] * np.array([1., -1., 1.])
    normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    length = np.linalg.norm(normal, axis=1)
    triangles = triangles[(length > 1e-10) & (normal[:, 2] >= length*np.cos(np.pi/4))]
    if not len(triangles):
        raise ValueError('source has no upward architectural support faces')
    with Path(output).open('xb') as f:
        np.save(f, triangles, allow_pickle=False)
    return dict(structure_path=str(Path(output).resolve()), structure_sha256=digest(output))


def write_obj(path, vertices, faces):
    # Habitat interprets PLY stages as Matterport PTex assets. OBJ uses its
    # generic triangle importer without texture/semantic/GPU dependencies.
    with Path(path).open('w') as f:
        np.savetxt(f, vertices, fmt='v %.9g %.9g %.9g')
        np.savetxt(f, faces + 1, fmt='f %d %d %d')


def walkable_height_chain(vertices, faces, settings, start_z, goal_z):
    """Necessary height support only; connected XY/body clearance still required.

    Inputs use Habitat Y-up. Inspect both face windings so a reported missing
    height band cannot be blamed on the authored floor winding. A positive
    result is deliberately not a path or permission to construct labels.
    """
    tri = np.asarray(vertices)[np.asarray(faces)]
    normal = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    length = np.linalg.norm(normal, axis=1)
    eligible = ((length > 1e-10) &
        (np.abs(normal[:, 1]) >= length*np.cos(np.deg2rad(settings['agent_max_slope']))-1e-10))
    low, high = sorted((float(start_z), float(goal_z)))
    heights = tri[eligible, :, 1]
    ranges = np.column_stack((heights.min(axis=1), heights.max(axis=1))) if len(heights) else np.empty((0, 2))
    ranges = ranges[(ranges[:, 1] >= low) & (ranges[:, 0] <= high)]
    ranges = np.clip(ranges, low, high)
    merged = []
    for a, b in sorted(ranges.tolist()):
        if merged and a <= merged[-1][1]+1e-5:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    # Include uncovered endpoints, rather than assuming either is supported.
    gaps, cursor = [], low
    budget = settings['agent_max_climb']+2*settings['cell_height']
    for a, b in merged+[[high, high]]:
        if a-cursor > budget+1e-5:
            gaps.append(dict(z_range_m=[cursor, a], gap_m=a-cursor))
        cursor = max(cursor, b)
    return dict(schema='sage-walkable-height-chain-diagnostic-v1',
        requested_ground_range_m=[low, high], source_triangles=len(tri),
        slope_eligible_triangles=int(eligible.sum()), slope_limit_deg=settings['agent_max_slope'],
        climb_budget_with_voxel_margin_m=budget, slope_surface_height_intervals_m=merged,
        unsupported_height_gaps=gaps, has_necessary_height_chain=not gaps,
        winding_independent=True, connected_path_proven=False, training_admitted=False)


def build(scene, output, usd_root=DEFAULT_USD_ROOT, collision_sidedness='authored', radius=.10):
    import habitat_sim as hs
    started = time.time()
    out = Path(output).resolve()
    allowed = Path('/data/zerovln')
    if not out.is_relative_to(allowed) or out.exists():
        raise ValueError('output must be a fresh directory on authorized data disk')
    source = Path(_asset_path(usd_root, scene))
    vertices, faces = world_mesh(source)
    source_faces = len(faces)
    if collision_sidedness == 'both':
        # Explicit diagnostic alternative for mixed structural winding. Keep
        # the authored mesh unchanged; no seed is quality-approved by build.
        faces = np.concatenate((faces, faces[:, ::-1]))
    elif collision_sidedness != 'authored':
        raise ValueError('unknown collision sidedness')
    out.mkdir(parents=True)
    asset = out / 'collision.obj'
    write_obj(asset, vertices, faces)
    cfg = hs.SimulatorConfiguration()
    cfg.scene_id = str(asset)
    cfg.create_renderer = False
    cfg.enable_physics = False
    agent = hs.agent.AgentConfiguration()
    agent.sensor_specifications = []
    sim = hs.Simulator(hs.Configuration(cfg, [agent]))
    try:
        ns = hs.nav.NavMeshSettings()
        ns.set_defaults()
        if not 0 < radius < 1:
            raise ValueError('agent radius must be in (0, 1) metres')
        ns.agent_radius = radius
        ns.agent_height = 1.55
        ns.agent_max_climb = .20
        ns.agent_max_slope = 45.
        ns.cell_size = .025
        ns.cell_height = .025
        if not sim.recompute_navmesh(sim.pathfinder, ns):
            raise RuntimeError('Recast failed to build a navigation mesh')
        target = out / 'scene.navmesh'
        sim.pathfinder.save_nav_mesh(str(target))
        settings = {k: float(getattr(ns, k)) for k in (
            'agent_radius', 'agent_height', 'agent_max_climb', 'agent_max_slope',
            'cell_size', 'cell_height')}
        report = dict(schema='sage-usd-navmesh-v1', scene=scene,
                      coordinate_transform='habitat=[sage.x,sage.z,-sage.y]',
                      usd_path=str(source), usd_sha256=digest(source),
                      mesh_path=str(asset), mesh_sha256=digest(asset),
                      navmesh_path=str(target), navmesh_sha256=digest(target),
                      builder_sha256=digest(__file__), settings=settings,
                      source_vertices=len(vertices), source_faces=source_faces,
                      exported_faces=len(faces), collision_sidedness=collision_sidedness,
                      area_m2=float(sim.pathfinder.navigable_area),
                      islands=int(sim.pathfinder.num_islands),
                      quality_approved=False, elapsed_seconds=time.time()-started)
        report.update(write_structure(source, out / 'structure.npy'))
        (out / 'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
        return report
    finally:
        sim.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--usd-root', default=DEFAULT_USD_ROOT)
    p.add_argument('--radius', type=float, default=.10)
    p.add_argument('--collision-sidedness', choices=('authored', 'both'), default='authored')
    args = p.parse_args()
    build(args.scene, args.output, args.usd_root, args.collision_sidedness, args.radius)


if __name__ == '__main__':
    main()
