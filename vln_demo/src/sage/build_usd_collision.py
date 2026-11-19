#!/usr/bin/env python3
"""Rasterize SAGE-3D's official USD collision payload onto the 5 cm map.

The official release keeps photorealistic 3DGS and physics geometry separate.
Each ``<scene>_collision.usd`` contains a floor-plan shell plus independently
transformed object collision meshes.  This builder applies every transform in
that payload and projects body-height faces into the InteriorGS occupancy frame.

Only the offline builder needs Pixar USD (``usd-core``/Isaac Python).  Runtime
loads a compact NPZ through ``SAGE_COLLISION=usd`` and has no USD dependency.
The existing official and visual-v3 paths are not modified or overwritten.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from .datasets import SageSet
from .occmap import OccMap, USD_COLLISION_ROOT, _point_in_poly


DEFAULT_USD_ROOT = os.environ.get(
    "SAGE_COLLISION_USD_ROOT",
    "/data/zerovln/SAGE-3D/"
    "SAGE-3D_Collision_Mesh/Collision_Mesh")

_WORKER_DATASET = None


def _init_worker(split: str, root: str):
    global _WORKER_DATASET
    _WORKER_DATASET = SageSet(split, root=root)


def _build_worker(job):
    scene, usd_root, base_root, out_root, min_z, max_z = job
    return build_one(_WORKER_DATASET, scene, usd_root, base_root, out_root,
                     min_z, max_z)


def _load_usd():
    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:
        raise RuntimeError(
            "build_usd_collision requires Pixar USD at build time; install "
            "usd-core or run it with Isaac Sim's Python") from exc
    return Usd, UsdGeom


def _asset_path(root: str, bare: str) -> str:
    candidates = (
        os.path.join(root, bare, bare + "_collision.usd"),
        os.path.join(root, "Collision_Mesh", bare, bare + "_collision.usd"),
        os.path.join(root, bare + "_collision.usd"),
    )
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "collision USD not found; tried: " + ", ".join(candidates))


def _world_points(mesh, xforms) -> np.ndarray:
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    matrix = np.asarray(xforms.GetLocalToWorldTransform(mesh.GetPrim()),
                        dtype=np.float64)
    # Gf.Matrix4d uses row-vector convention; translation is in the last row.
    homogeneous = np.concatenate(
        (points, np.ones((len(points), 1), dtype=np.float64)), axis=1)
    return (homogeneous @ matrix)[:, :3]


def _face_ranges(counts: Iterable[int]):
    start = 0
    for count in counts:
        end = start + int(count)
        yield start, end
        start = end


def _clip_z(points: np.ndarray, bound: float, keep_above: bool) -> np.ndarray:
    """Clip one 3-D polygon against a horizontal half-space."""
    if len(points) == 0:
        return points
    out = []
    previous = points[-1]
    previous_inside = (previous[2] >= bound) if keep_above else (previous[2] <= bound)
    for current in points:
        current_inside = ((current[2] >= bound) if keep_above
                          else (current[2] <= bound))
        if current_inside != previous_inside:
            dz = current[2] - previous[2]
            if abs(dz) > 1e-12:
                alpha = (bound - previous[2]) / dz
                out.append(previous + alpha * (current - previous))
        if current_inside:
            out.append(current)
        previous, previous_inside = current, current_inside
    return np.asarray(out, dtype=np.float64)


def _body_slice(face: np.ndarray, min_z: float, max_z: float) -> np.ndarray:
    return _clip_z(_clip_z(face, min_z, keep_above=True),
                   max_z, keep_above=False)


def _room_region(occ: OccMap, structure) -> np.ndarray:
    xs = occ.lower[0] + (np.arange(occ.w) + 0.5) * occ.scale
    ys = occ.lower[1] + (occ.h - 0.5 - np.arange(occ.h)) * occ.scale
    grid_x, grid_y = np.meshgrid(xs, ys)
    result = np.zeros_like(occ.free)
    for room in structure.get("rooms", []):
        profile = room.get("profile") or []
        if len(profile) >= 3:
            result |= _point_in_poly(
                grid_x, grid_y,
                [(float(point[0]), float(point[1])) for point in profile])
    return result


def _opening_region(occ: OccMap, structure, opening_type: str,
                    normal_margin_m: float = 0.12,
                    room_region: np.ndarray | None = None,
                    interior_only: bool = False) -> np.ndarray:
    """Rasterize structure ``holes`` as finite-width wall apertures.

    A hole profile is a vertical rectangle, hence only two distinct XY
    endpoints.  Expanding it by half the recorded wall thickness plus a small
    grid margin produces a corridor through the whole wall shell.  Doors and
    windows are kept separate: doors alter navigation topology; windows are
    only metadata for render-boundary diagnostics.
    """
    canvas = Image.new("1", (occ.w, occ.h), 0)
    draw = ImageDraw.Draw(canvas)
    wanted = opening_type.strip().upper()
    for opening in structure.get("holes", []):
        if str(opening.get("type", "")).strip().upper() != wanted:
            continue
        profile = opening.get("profile") or []
        xy = np.asarray([[float(p[0]), float(p[1])] for p in profile
                         if len(p) >= 2], dtype=np.float64)
        if len(xy) < 2:
            continue
        # Pick the farthest pair so duplicated top/bottom vertices collapse to
        # the full opening span.
        delta = xy[:, None, :] - xy[None, :, :]
        i, j = np.unravel_index(np.argmax(np.sum(delta * delta, axis=-1)),
                                (len(xy), len(xy)))
        p0, p1 = xy[i], xy[j]
        tangent = p1 - p0
        length = float(np.linalg.norm(tangent))
        if length < 1e-6:
            continue
        tangent /= length
        normal = np.asarray([-tangent[1], tangent[0]])
        half_width = 0.5 * float(opening.get("thickness", 0.0))
        half_width += max(0.0, float(normal_margin_m))
        if interior_only:
            if room_region is None:
                continue
            # A semantic DOOR may be an exterior entrance. Only carve it when
            # both sides, sampled beyond the wall shell, belong to declared
            # rooms. This opens room-to-room portals without letting the robot
            # walk out of the reconstructed scene.
            center = 0.5 * (p0 + p1)
            probe = half_width + max(0.15, 3.0 * occ.scale)
            sides = (center + normal * probe, center - normal * probe)
            side_inside = []
            for point in sides:
                col, row = occ.to_pixel(float(point[0]), float(point[1]))
                side_inside.append(
                    0 <= col < occ.w and 0 <= row < occ.h
                    and bool(room_region[row, col]))
            if not all(side_inside):
                continue
        corners = (p0 + normal * half_width,
                   p1 + normal * half_width,
                   p1 - normal * half_width,
                   p0 - normal * half_width)
        pixels = [((float(p[0]) - occ.lower[0]) / occ.scale,
                   (occ.h - 1) - (float(p[1]) - occ.lower[1]) / occ.scale)
                  for p in corners]
        draw.polygon(pixels, fill=1)
    return np.asarray(canvas, dtype=bool)


def _raster_payload(path: str, occ: OccMap, min_z: float, max_z: float):
    Usd, UsdGeom = _load_usd()
    stage = Usd.Stage.Open(path)
    if not stage:
        raise RuntimeError("unable to open USD stage: " + path)
    if str(UsdGeom.GetStageUpAxis(stage)).upper() != "Z":
        raise ValueError("collision payload must be Z-up: " + path)

    canvas = Image.new("1", (occ.w, occ.h), 0)
    draw = ImageDraw.Draw(canvas)
    structure_canvas = Image.new("1", (occ.w, occ.h), 0)
    structure_draw = ImageDraw.Draw(structure_canvas)
    object_canvas = Image.new("1", (occ.w, occ.h), 0)
    object_draw = ImageDraw.Draw(object_canvas)
    xforms = UsdGeom.XformCache()
    mesh_count = face_count = kept_faces = point_count = 0
    bounds_lo = np.full(3, np.inf)
    bounds_hi = np.full(3, -np.inf)

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        collision_enabled = prim.GetAttribute("physics:collisionEnabled").Get()
        if collision_enabled is False:
            continue
        mesh = UsdGeom.Mesh(prim)
        is_structure = prim.GetName() == "SM_floorplan"
        part_draw = structure_draw if is_structure else object_draw
        points = _world_points(mesh, xforms)
        if not len(points):
            continue
        bounds_lo = np.minimum(bounds_lo, points.min(axis=0))
        bounds_hi = np.maximum(bounds_hi, points.max(axis=0))
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        mesh_count += 1
        point_count += len(points)
        face_count += len(counts)

        for start, end in _face_ranges(counts):
            source_face = points[indices[start:end]]
            # SM_floorplan contains both vertical wall shells and horizontal
            # floor/ceiling caps.  Some releases put the floor exactly at
            # min_z=0.10; projecting that cap would mark the whole room blocked
            # (840171 is the concrete regression).  A ground robot collides
            # with the wall sides, not with the support plane it stands on.
            # Keep sloped/vertical structure faces and every object face.
            if is_structure and np.ptp(source_face[:, 2]) < 1e-4:
                continue
            face = _body_slice(source_face, min_z, max_z)
            if len(face) < 2:
                continue
            xy = [((float(p[0]) - occ.lower[0]) / occ.scale,
                   (occ.h - 1) - (float(p[1]) - occ.lower[1]) / occ.scale)
                  for p in face]
            draw.polygon(xy, fill=1)
            # Preserve edge-on walls whose XY projection is degenerate.
            draw.line(xy + [xy[0]], fill=1, width=1)
            part_draw.polygon(xy, fill=1)
            part_draw.line(xy + [xy[0]], fill=1, width=1)
            kept_faces += 1

    if mesh_count == 0 or kept_faces == 0:
        raise ValueError("USD payload contains no body-height collision faces: " + path)
    return (np.asarray(canvas, dtype=bool),
            np.asarray(structure_canvas, dtype=bool),
            np.asarray(object_canvas, dtype=bool)), {
        "meshes": mesh_count,
        "points": point_count,
        "faces": face_count,
        "body_faces": kept_faces,
        "bounds_lo": bounds_lo,
        "bounds_hi": bounds_hi,
    }


def build_one(ds: SageSet, scene: str, usd_root: str, base_root: str,
              out_root: str,
              min_z: float = 0.10, max_z: float = 1.55) -> str:
    bare = ds.bare(scene)
    scene_dir = ds.scene_dir(scene)
    occ = OccMap(scene_dir, collision_mode="official", furniture_mode="none")
    base_path = os.path.join(base_root, bare + ".npz")
    if os.path.isfile(base_path):
        # v2 is the exact official-USD raster. Reusing it keeps v4 additive
        # and avoids reopening very large payloads such as 840007 (156 MB),
        # which is unstable in some usd-core builds.
        with np.load(base_path, allow_pickle=False) as base:
            base_version = int(np.asarray(base["version"]).reshape(()))
            obstacle = np.asarray(base["obstacle"], dtype=bool)
            structure_obstacle = np.asarray(
                base["structure_obstacle"], dtype=bool)
            object_obstacle = np.asarray(base["object_obstacle"], dtype=bool)
        if base_version < 2 or obstacle.shape != occ.occ.shape:
            raise ValueError("invalid base USD collision artifact: " + base_path)
        stats = None
    else:
        usd_path = _asset_path(usd_root, bare)
        masks, stats = _raster_payload(usd_path, occ, min_z, max_z)
        obstacle, structure_obstacle, object_obstacle = masks

    with open(os.path.join(scene_dir, "structure.json"), encoding="utf-8") as f:
        structure = json.load(f)
    room_region = _room_region(occ, structure)
    all_door_opening = (_opening_region(occ, structure, "DOOR")
                        | _opening_region(occ, structure, "HOLE"))
    door_opening = (
        _opening_region(occ, structure, "DOOR", room_region=room_region,
                        interior_only=True)
        | _opening_region(occ, structure, "HOLE", room_region=room_region,
                          interior_only=True))
    exterior_door_opening = all_door_opening & ~door_opening
    window_opening = _opening_region(occ, structure, "WINDOW", 0.0)
    # Door panels may swing outside the thin navigation corridor.  Rendering
    # gets a wider visibility bridge so valid door geometry is not mistaken
    # for exterior leakage; collision still uses the narrow physical aperture.
    render_domain = (room_region
                     | _opening_region(occ, structure, "DOOR",
                                       normal_margin_m=0.20,
                                       room_region=room_region,
                                       interior_only=True)
                     | _opening_region(occ, structure, "HOLE",
                                       normal_margin_m=0.20,
                                       room_region=room_region,
                                       interior_only=True))

    # Keep the existing USD fields byte-for-byte in meaning.  The room-topology
    # path is additive: all declared rooms become candidate floor, door holes
    # bridge wall shells, while USD walls and objects remain collision geometry.
    candidate = (occ.occ == 255) | room_region
    free = candidate & ~obstacle
    rooms_candidate = room_region | door_opening
    rooms_obstacle = obstacle & ~door_opening
    rooms_free = rooms_candidate & ~rooms_obstacle

    Path(out_root).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_root, bare + ".npz")
    tmp_path = out_path + f".tmp.{os.getpid()}.npz"
    np.savez_compressed(
        tmp_path,
        free=free,
        obstacle=obstacle,
        structure_obstacle=structure_obstacle,
        object_obstacle=object_obstacle,
        candidate=candidate,
        room_region=room_region,
        door_opening=door_opening,
        exterior_door_opening=exterior_door_opening,
        window_opening=window_opening,
        rooms_candidate=rooms_candidate,
        rooms_obstacle=rooms_obstacle,
        rooms_free=rooms_free,
        render_domain=render_domain,
        scale=np.asarray(occ.scale, np.float32),
        lower=np.asarray(occ.lower, np.float32),
        shape=np.asarray(free.shape, np.int32),
        min_z=np.asarray(min_z, np.float32),
        max_z=np.asarray(max_z, np.float32),
        version=np.asarray(4, np.int32),
    )
    os.replace(tmp_path, out_path)
    source_summary = (f"base_v2={base_path} " if stats is None else
        f"meshes={stats['meshes']} points={stats['points']} "
        f"faces={stats['faces']} body_faces={stats['body_faces']} "
        f"bounds={stats['bounds_lo'].round(3).tolist()}.."
        f"{stats['bounds_hi'].round(3).tolist()} ")
    print(
        f"{scene}: {source_summary}"
        f"blocked={int(obstacle.sum())}/{obstacle.size}={obstacle.mean():.2%} "
        f"structure={structure_obstacle.mean():.2%} "
        f"objects={object_obstacle.mean():.2%} "
        f"free={free.mean():.2%} rooms_free={rooms_free.mean():.2%} "
        f"doors={door_opening.mean():.2%} -> {out_path}", flush=True)
    return out_path


def _has_room_topology(path: str) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            return (int(np.asarray(data["version"]).reshape(())) >= 4
                    and all(key in data.files for key in
                            ("room_region", "door_opening", "rooms_free",
                             "render_domain", "exterior_door_opening")))
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="instr")
    parser.add_argument("--all-splits", action="store_true",
                        help="build the union of train/instr/scene/traj")
    parser.add_argument("--scenes", default="",
                        help="comma-separated bare or sage-prefixed ids")
    parser.add_argument("--usd-root", default=DEFAULT_USD_ROOT)
    parser.add_argument("--base-root", default=USD_COLLISION_ROOT,
                        help="existing official-USD v2 NPZ root (preferred)")
    parser.add_argument("--out", default=USD_COLLISION_ROOT)
    parser.add_argument("--min-z", type=float, default=0.10)
    parser.add_argument("--max-z", type=float, default=1.55)
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel USD readers (default: 1)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="resume without rebuilding existing NPZ files")
    args = parser.parse_args()
    if args.min_z >= args.max_z:
        parser.error("--min-z must be smaller than --max-z")

    ds = SageSet(args.split)
    requested = [
        ds.qualify(value.strip())
        if not value.strip().startswith("sage-") else value.strip()
        for value in args.scenes.split(",") if value.strip()
    ]
    if requested:
        scenes = requested
    elif args.all_splits:
        scenes = sorted({scene for split in ("train", "instr", "scene", "traj")
                         for scene in SageSet(split, root=ds.root).list_scenes()})
    else:
        scenes = ds.list_scenes()
    if args.skip_existing:
        before = len(scenes)
        scenes = [scene for scene in scenes
                  if not _has_room_topology(
                      os.path.join(args.out, ds.bare(scene) + ".npz"))]
        print(f"skip-existing: {before - len(scenes)} v4 present, {len(scenes)} pending",
              flush=True)
    if not scenes:
        print("nothing to build", flush=True)
        return
    workers = max(1, int(args.workers))
    if workers == 1:
        for scene in scenes:
            build_one(ds, scene, args.usd_root, args.base_root, args.out,
                      args.min_z, args.max_z)
        return
    jobs = [(scene, args.usd_root, args.base_root, args.out,
             args.min_z, args.max_z)
            for scene in scenes]
    with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker,
            initargs=(args.split, ds.root)) as executor:
        list(executor.map(_build_worker, jobs))


if __name__ == "__main__":
    main()
