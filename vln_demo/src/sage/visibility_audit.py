"""Camera-geometry and metric-depth evidence shared by SAGE builders.

The helpers are category- and scene-agnostic.  Semantic localization remains a
separate gate; this module only answers whether a 3-D annotation projects into
the actual camera and whether rendered metric depth supports that annotation.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np


def visible_to_bbox_2d(occ, stance: Sequence[float], info: dict,
                       robot_radius_m: float) -> bool:
    """Collision-aware 2-D LOS ending just before the target's own bbox.

    This is the canonical demo builder's geometry gate factored into the shared
    visibility module so offline semantic-region scoring uses the exact same
    ray contract instead of growing a second implementation.
    """
    sx, sy = float(stance[0]), float(stance[1])
    x0, y0, x1, y1 = info["bbox_xyxy"]
    cx, cy = info["centre_xy"]
    dx, dy = cx - sx, cy - sy
    intersections = []
    if abs(dx) > 1e-9:
        for x in (x0, x1):
            t = (x - sx) / dx
            y = sy + t * dy
            if 0.0 < t <= 1.0 and y0 - 1e-6 <= y <= y1 + 1e-6:
                intersections.append(t)
    if abs(dy) > 1e-9:
        for y in (y0, y1):
            t = (y - sy) / dy
            x = sx + t * dx
            if 0.0 < t <= 1.0 and x0 - 1e-6 <= x <= x1 + 1e-6:
                intersections.append(t)
    if not intersections:
        return False
    hit_t = min(intersections)
    distance = math.hypot(dx, dy)
    safe_t = max(0.0, hit_t - 0.08 / max(distance, 1e-6))
    ex, ey = sx + safe_t * dx, sy + safe_t * dy
    return occ.seg_free(sx, sy, ex, ey, float(robot_radius_m))


def project_bbox(camera, camera_pos: Sequence[float], yaw: float,
                 info: dict) -> Optional[dict]:
    x0, y0, x1, y1 = info["bbox_xyxy"]
    z0, z1 = info["z_range"]
    c, s = math.cos(yaw), math.sin(yaw)
    uv = []
    for x in (x0, x1):
        for y in (y0, y1):
            for z in (z0, z1):
                dx, dy, dz = x - camera_pos[0], y - camera_pos[1], z - camera_pos[2]
                x_cam = s * dx - c * dy
                y_cam = -dz
                z_cam = c * dx + s * dy
                if z_cam > 1e-3:
                    uv.append((camera.fx * x_cam / z_cam + camera.width / 2.0,
                               camera.fy * y_cam / z_cam + camera.height / 2.0))
    if not uv:
        return None
    us, vs = [p[0] for p in uv], [p[1] for p in uv]
    raw = (min(us), min(vs), max(us), max(vs))
    clip = (max(0.0, raw[0]), max(0.0, raw[1]),
            min(float(camera.width), raw[2]), min(float(camera.height), raw[3]))
    width, height = max(0.0, clip[2] - clip[0]), max(0.0, clip[3] - clip[1])
    return {
        "bbox_px": [round(v, 1) for v in clip],
        "width_px": round(width, 1), "height_px": round(height, 1),
        "projected_area_ratio": round(
            width * height / float(camera.width * camera.height), 6),
        "center_px": [round(0.5 * (clip[0] + clip[2]), 1),
                      round(0.5 * (clip[1] + clip[3]), 1)],
    }


def ray_box_interval(camera, camera_pos: Sequence[float], yaw: float,
                     u: float, v: float, info: dict):
    """Camera-forward-depth interval in which a pixel ray crosses an AABB."""
    nx = (u - camera.width / 2.0) / camera.fx
    ny = (v - camera.height / 2.0) / camera.fy
    c, s = math.cos(yaw), math.sin(yaw)
    direction = (c + nx * s, s - nx * c, -ny)
    bounds = ((info["bbox_xyxy"][0], info["bbox_xyxy"][2]),
              (info["bbox_xyxy"][1], info["bbox_xyxy"][3]),
              (info["z_range"][0], info["z_range"][1]))
    near, far = -float("inf"), float("inf")
    for origin, delta, (lo, hi) in zip(camera_pos, direction, bounds):
        if abs(delta) < 1e-9:
            if origin < lo or origin > hi:
                return None
            continue
        a, b = (lo - origin) / delta, (hi - origin) / delta
        if a > b:
            a, b = b, a
        near, far = max(near, a), min(far, b)
        if near > far:
            return None
    if far <= 0:
        return None
    return max(near, 0.0), far


def depth_support(depth, bbox_px, camera, camera_pos, yaw: float, info: dict,
                  stride: int = 4, tolerance_m: float = .10) -> dict:
    """Measure target-consistent depth rays inside an image rectangle."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        return {"tested_rays": 0, "visible_rays": 0,
                "visible_fraction": 0.0, "visible_area_ratio": 0.0}
    x0, y0, x1, y1 = [int(round(v)) for v in bbox_px]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(depth.shape[1] - 1, x1), min(depth.shape[0] - 1, y1)
    tested = visible = 0
    stride = max(1, int(stride))
    for v in range(y0, y1 + 1, stride):
        for u in range(x0, x1 + 1, stride):
            interval = ray_box_interval(camera, camera_pos, yaw, u + .5, v + .5, info)
            if interval is None:
                continue
            tested += 1
            value = float(depth[v, u])
            if (math.isfinite(value) and value > 0
                    and interval[0] - tolerance_m <= value <= interval[1] + tolerance_m):
                visible += 1
    pixels = visible * stride * stride
    return {"tested_rays": tested, "visible_rays": visible,
            "visible_fraction": visible / float(max(tested, 1)),
            "visible_area_ratio": pixels / float(camera.width * camera.height)}


def visible_object_ids(objects, depth, camera, camera_pos, yaw, *, candidate=None):
    """One visibility gate shared by SAGE and native R2R/RxR rewards."""
    visible = []
    for key, obj in objects.items():
        if candidate is not None and not candidate(obj):
            continue
        projection = project_bbox(camera, camera_pos, yaw, obj)
        if not projection or projection['projected_area_ratio'] < .001:
            continue
        evidence = depth_support(depth, projection['bbox_px'], camera, camera_pos,
                                 yaw, obj, stride=8)
        if evidence['visible_rays'] >= 4 and evidence['visible_fraction'] >= .25:
            visible.append(key)
    return visible
