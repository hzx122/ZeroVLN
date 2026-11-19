"""Geometry-backed semantic terminal regions for SAGE object goals.

The legacy benchmark scores distance to one golden endpoint.  This module
instead asks whether an agent pose is in safe, visible, reachable free space
beside the target object's XY bounding box.  It deliberately reuses the same
``OccMap``/room/LOS contracts as the canonical demo builder and does not render
or contact a model, so immutable rollouts can be rescored offline.
"""
from __future__ import annotations

import json
import hashlib
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from . import route_extractor
from . import occmap as occmap_module
from .occmap import OccMap
from .planner import passable_grid
from .visibility_audit import project_bbox, visible_to_bbox_2d
from rs3d.gs_client import CameraSpec


SEMANTIC_TERMINAL_SCHEMA = "sage-semantic-terminal-v1"
OFFICIAL_OBJECT_SUCCESS_SCHEMA = "sage-official-expanded-object-bbox-v1"
OFFICIAL_OBJECT_SUCCESS_CONTRACT_PATH = (
    Path(__file__).resolve().parents[2] / "config" /
    "sage_official_object_success_v1.json")
_OFFICIAL_OBJECT_SUCCESS_CONTRACT_BYTES = (
    OFFICIAL_OBJECT_SUCCESS_CONTRACT_PATH.read_bytes())
OFFICIAL_OBJECT_SUCCESS_CONTRACT_SHA256 = hashlib.sha256(
    _OFFICIAL_OBJECT_SUCCESS_CONTRACT_BYTES).hexdigest()
OFFICIAL_OBJECT_SUCCESS_CONTRACT = json.loads(
    _OFFICIAL_OBJECT_SUCCESS_CONTRACT_BYTES)
OFFICIAL_OBJECT_EXPANSION_RADIUS_M = float(
    OFFICIAL_OBJECT_SUCCESS_CONTRACT["expansion_radius_m"])
FROZEN_REGION_CONFIG_PATH = (Path(__file__).resolve().parents[2] / "config" /
                             "sage_semantic_terminal_v1.json")
with open(FROZEN_REGION_CONFIG_PATH, encoding="utf-8") as _handle:
    _FROZEN_CONTRACT = json.load(_handle)
DEFAULT_REGION_CONFIG = dict(_FROZEN_CONTRACT["region_config"])
SENSITIVITY_MAX_DISTANCE_M = tuple(
    float(value) for value in _FROZEN_CONTRACT[
        "sensitivity_max_surface_distance_m"])

_NON_OBJECT_LABELS = {
    "ceiling", "floor", "wall", "baseboard", "door", "interior_doors",
    "window", "other", "unknown",
}


def semantic_family(label: str) -> str:
    """Canonical coarse family used to identify plausible proxy instances."""
    value = str(label).strip().lower().replace(" ", "_")
    if "cabinet" in value or "cupboard" in value or value in ("dresser", "wardrobe"):
        return "cabinet"
    if "chair" in value or value in ("stool", "armchair"):
        return "chair"
    if "table" in value or value in ("desk", "counter"):
        return "table"
    if "sofa" in value or "couch" in value:
        return "sofa"
    if "door" in value:
        return "door"
    return value


def label_index_from_items(items: Iterable[dict]) -> Dict[str, dict]:
    """Build the canonical ``label_N`` index used by GVLN/SAGE manifests."""
    counts: Dict[str, int] = {}
    index: Dict[str, dict] = {}
    for item in items:
        raw = str(item.get("label") or "unknown").strip() or "unknown"
        normalized = raw.lower().replace(" ", "_")
        counts[normalized] = counts.get(normalized, 0) + 1
        object_id = "%s_%d" % (normalized, counts[normalized])
        bbox = item.get("bounding_box") or []
        xs = [float(point["x"]) for point in bbox]
        ys = [float(point["y"]) for point in bbox]
        zs = [float(point.get("z", 0.0)) for point in bbox]
        if not xs or not ys:
            continue
        index[object_id] = {
            "label": raw,
            "bbox_xyxy": (min(xs), min(ys), max(xs), max(ys)),
            "centre_xy": (0.5 * (min(xs) + max(xs)),
                          0.5 * (min(ys) + max(ys))),
            "z_range": (min(zs), max(zs)),
            "ins_id": str(item.get("ins_id", "")),
        }
    return index


def _polygon_covers_point(vertices: Sequence[Sequence[float]], x: float,
                          y: float, eps: float = 1e-10) -> bool:
    """Dependency-free equivalent of Shapely ``Polygon.covers(Point)``."""
    inside = False
    for index, first in enumerate(vertices):
        second = vertices[(index + 1) % len(vertices)]
        x0, y0 = float(first[0]), float(first[1])
        x1, y1 = float(second[0]), float(second[1])
        cross = (x - x0) * (y1 - y0) - (y - y0) * (x1 - x0)
        if (abs(cross) <= eps and min(x0, x1) - eps <= x <= max(x0, x1) + eps
                and min(y0, y1) - eps <= y <= max(y0, y1) + eps):
            return True
        if (y0 > y) != (y1 > y):
            intersect_x = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < intersect_x:
                inside = not inside
    return inside


def load_label_index(scene_dir: str) -> Dict[str, dict]:
    root = Path(scene_dir)
    with open(root / "labels.json", encoding="utf-8") as handle:
        items = json.load(handle)
    index = label_index_from_items(items)
    # Reuse the official semantic-map builder's cell-centre ``Polygon.covers``
    # rule so the local-frame bbox includes exactly the same raster cells as
    # ``bbox_m`` before its mirror transform.  This keeps boundary behavior
    # identical without materializing a semantic-map JSON per gym process.
    try:
        from PIL import Image
        with open(root / "occupancy.json", encoding="utf-8") as handle:
            meta = json.load(handle)
        scale = float(meta["scale"])
        x_min, y_min = (float(value) for value in meta["min"][:2])
        with Image.open(root / "occupancy.png") as image:
            occupancy = np.asarray(image.convert("L"))
        height, width = occupancy.shape
        map_cols: list[int] = []
        map_rows: list[int] = []
        object_map_bboxes: dict[str, tuple[float, float, float, float]] = {}
        counts: Dict[str, int] = {}
        for item in items:
            label = str(item.get("label") or "unknown").strip().lower().replace(" ", "_")
            counts[label] = counts.get(label, 0) + 1
            object_id = "%s_%d" % (label, counts[label])
            if object_id not in index:
                continue
            polygon = [(float(v["x"]), float(v["y"]))
                       for v in (item.get("bounding_box") or [])[:4]]
            if len(polygon) < 3:
                continue
            x0, y0, x1, y1 = index[object_id]["bbox_xyxy"]
            cols = range(max(0, int(math.floor((x0 - x_min) / scale))),
                         min(width - 1, int(math.floor((x1 - x_min) / scale))) + 1)
            rows = range(max(0, int(math.floor((y0 - y_min) / scale))),
                         min(height - 1, int(math.floor((y1 - y_min) / scale))) + 1)
            covered = [(col, row) for col in cols for row in rows
                       if _polygon_covers_point(
                           polygon, x_min + (col + 0.5) * scale,
                           y_min + (row + 0.5) * scale)]
            if covered:
                c0, c1 = min(x[0] for x in covered), max(x[0] for x in covered)
                r0, r1 = min(x[1] for x in covered), max(x[1] for x in covered)
                # Official builder mirrors both object-mask axes before writing
                # bbox_m/mask_coords_m, then rounds every coordinate to 2 dp.
                mc0, mc1 = width - 1 - c1, width - 1 - c0
                mr0, mr1 = height - 1 - r1, height - 1 - r0
                object_map_bboxes[object_id] = tuple(float(f"{value:.2f}") for value in (
                    x_min + mc0 * scale, y_min + mr0 * scale,
                    x_min + (mc1 + 1) * scale, y_min + (mr1 + 1) * scale))
                map_cols.extend(width - 1 - col for col, _ in covered)
                map_rows.extend(height - 1 - row for _, row in covered)

        # Match the official builder's remaining semantic-map entries because
        # ObjectBasedSuccessEvaluator derives its reverse-map origin from the
        # extrema of *all* emitted mask coordinates.  Walls/unable regions flip
        # only the image row in upstream code (yes, differently from objects).
        pixels, pixel_counts = np.unique(occupancy.reshape(-1), return_counts=True)
        candidate_walls = [int(value) for value in pixels if 0 < value < 250]
        wall_value = (candidate_walls[int(np.argmax([
            pixel_counts[np.where(pixels == value)[0][0]]
            for value in candidate_walls]))] if candidate_walls else int(pixels[0]))
        wall_y, wall_x = np.where(np.flipud(occupancy == wall_value))
        map_cols.extend(int(value) for value in wall_x)
        map_rows.extend(int(value) for value in wall_y)
        from scipy.ndimage import label as component_label
        unable_labels, unable_count = component_label(
            np.flipud(occupancy == 0), structure=np.ones((3, 3)))
        for component in range(1, unable_count + 1):
            ys, xs = np.where(unable_labels == component)
            if xs.size >= 5:
                map_cols.extend(int(value) for value in xs)
                map_rows.extend(int(value) for value in ys)
        if map_cols and map_rows:
            map_x = [float(f"{x_min + (value + 0.5) * scale:.2f}")
                     for value in (min(map_cols), max(map_cols))]
            map_y = [float(f"{y_min + (value + 0.5) * scale:.2f}")
                     for value in (min(map_rows), max(map_rows))]
            origin_x, origin_y = sum(map_x), sum(map_y)
            for object_id, bbox in object_map_bboxes.items():
                index[object_id]["official_map_bbox_xyxy"] = bbox
                index[object_id]["official_bbox_xyxy"] = (
                    origin_x - bbox[2], origin_y - bbox[3],
                    origin_x - bbox[0], origin_y - bbox[1])
    except (OSError, KeyError, TypeError, ValueError):
        # Unit fixtures and old assets may omit occupancy metadata.  The raw
        # annotation bbox remains a conservative compatibility fallback.
        pass
    return index


def official_success_contract_identity() -> dict:
    """Return the frozen official object-success contract provenance."""
    from . import physical_navigation
    if physical_navigation.enabled():
        return physical_navigation.identity()
    from . import benchmark_protocol
    if benchmark_protocol.enabled():
        return benchmark_protocol.identity()
    return {
        "schema": OFFICIAL_OBJECT_SUCCESS_CONTRACT["schema"],
        "contract_path": str(OFFICIAL_OBJECT_SUCCESS_CONTRACT_PATH),
        "contract_sha256": OFFICIAL_OBJECT_SUCCESS_CONTRACT_SHA256,
        "object_success_schema": OFFICIAL_OBJECT_SUCCESS_SCHEMA,
        "expansion_radius_m": OFFICIAL_OBJECT_EXPANSION_RADIUS_M,
        "requires_agent_stop": bool(
            OFFICIAL_OBJECT_SUCCESS_CONTRACT["requires_agent_stop"]),
    }


def success_contract_matches(row, contract=None):
    """Reject saved XY results and changed physical geometry on resume/aggregate."""
    contract = contract or official_success_contract_identity()
    detail = row.get('official_object_success_detail') or {}
    if (row.get('success_contract') != contract['object_success_schema']
            or detail.get('contract_sha256') != contract['contract_sha256']
            or detail.get('expansion_radius_m') != contract['expansion_radius_m']):
        return False
    if contract['object_success_schema'] == 'sage-designated-target-navmesh-v2':
        import os
        from .navmesh_support import checked_digest
        from .physical_navigation import DEFAULT_ROOT
        try:
            path = Path(detail['navmesh_manifest_path'])
            if (not detail.get('target_contract_sha256') or not detail.get('target_object_id')
                    or checked_digest(path) != detail['navmesh_manifest_sha256']):
                return False
            spec = json.loads(path.read_text())
            if str(row.get('scene', '')).removeprefix('sage-') != spec['scene']:
                return False
            current = Path(os.environ.get('SAGE_NAVMESH_ROOT', DEFAULT_ROOT)) / spec['scene'] / 'manifest.json'
            if checked_digest(current) != detail['navmesh_manifest_sha256']:
                return False
            sources = ('usd', 'navmesh', 'structure') if 'structure_path' in spec else ('usd', 'navmesh')
            return all(checked_digest(spec[k+'_path']) == spec[k+'_sha256'] for k in sources)
        except (OSError, KeyError, TypeError, ValueError):
            return False
    return True


def official_object_arrival(info: dict, xy: Sequence[float],
                            expansion_radius_m: Optional[float] = None) -> dict:
    """Apply the official SAGE object-arrival geometry in the local frame.

    SAGE's ``ObjectBasedSuccessEvaluator`` reverse-maps an Isaac trajectory
    pose into the semantic-map frame and tests the closed, axis-aligned target
    bbox expanded by one metre.  ``SageEnv`` and ``labels.json`` already share
    that semantic-map geometry frame, so applying the same interval test here
    avoids both coordinate round trips and the upstream adapter bug that drops
    ``instructions[0].end`` and accidentally selects a fallback object.

    Deliberately absent: stop, visibility, reachability, collision, and
    point-goal checks.  The official evaluator does not gate object success on
    any of them.
    """
    radius = (OFFICIAL_OBJECT_EXPANSION_RADIUS_M
              if expansion_radius_m is None else float(expansion_radius_m))
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("expansion_radius_m must be finite and non-negative")
    if not math.isclose(radius, OFFICIAL_OBJECT_EXPANSION_RADIUS_M,
                        rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "official object success expansion radius is frozen at "
            f"{OFFICIAL_OBJECT_EXPANSION_RADIUS_M}m")
    x, y = float(xy[0]), float(xy[1])
    bbox = info.get("official_bbox_xyxy") or info["bbox_xyxy"]
    x0, y0, x1, y1 = (float(value) for value in bbox)
    expanded = (x0 - radius, y0 - radius, x1 + radius, y1 + radius)
    success = expanded[0] <= x <= expanded[2] and expanded[1] <= y <= expanded[3]
    return {
        "schema": OFFICIAL_OBJECT_SUCCESS_SCHEMA,
        "success": bool(success),
        "position_xy": [x, y],
        "target_bbox_xyxy": [x0, y0, x1, y1],
        "expanded_bbox_xyxy": list(expanded),
        "expansion_radius_m": radius,
        "contract_sha256": OFFICIAL_OBJECT_SUCCESS_CONTRACT_SHA256,
    }


def resolve_scene_dir(sage_root: str, scene: str) -> str:
    bare = str(scene).removeprefix("sage-")
    matches = [path for path in Path(sage_root, "InteriorGS").glob("*_%s" % bare)
               if path.is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(
            "expected one InteriorGS scene for %s, found %d" % (bare, len(matches)))
    return str(matches[0])


def goal_stances(info: dict, standoff_m: Sequence[float]) -> list[tuple[float, float]]:
    """Candidate poses on all four outside bbox faces (canonical builder logic)."""
    x0, y0, x1, y1 = info["bbox_xyxy"]
    out = []
    for distance in standoff_m:
        for fraction in (0.2, 0.5, 0.8):
            x = x0 + fraction * (x1 - x0)
            y = y0 + fraction * (y1 - y0)
            out.extend(((x0 - distance, y), (x1 + distance, y),
                        (x, y0 - distance), (x, y1 + distance)))
    return out


def _component_map(occ: OccMap, robot_radius_m: float) -> np.ndarray:
    from scipy.ndimage import label

    labels, _ = label(passable_grid(occ, robot_radius_m),
                      structure=np.ones((3, 3), dtype=int))
    return labels


def _map_value(grid: Optional[np.ndarray], occ: OccMap,
               xy: Sequence[float]) -> int:
    if grid is None:
        return 0
    col, row = occ.to_pixel(float(xy[0]), float(xy[1]))
    if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]):
        return 0
    return int(grid[row, col])


def _infer_target_room(occ: OccMap, room_lab: Optional[np.ndarray], info: dict,
                       goal_xy: Optional[Sequence[float]], config: dict) -> tuple[int, int]:
    """Use the frozen endpoint room, with canonical accessible sides as fallback."""
    endpoint_room = _map_value(room_lab, occ, goal_xy) if goal_xy is not None else 0
    valid_stances = []
    for stance in goal_stances(info, config["goal_standoff_m"]):
        room = _map_value(room_lab, occ, stance)
        if (room > 0 and occ.is_free(
                float(stance[0]), float(stance[1]),
                float(config["robot_radius_m"]) + float(config["planning_margin_m"]))
                and visible_to_bbox_2d(
                    occ, stance, info, float(config["robot_radius_m"]))):
            valid_stances.append(room)
    if endpoint_room > 0:
        return endpoint_room, len(valid_stances)
    if not valid_stances:
        return 0, 0
    return max(set(valid_stances), key=valid_stances.count), len(valid_stances)


def _point_membership(occ: OccMap, info: dict, xy: Sequence[float],
                      config: dict, room_lab: Optional[np.ndarray] = None,
                      target_room: int = 0,
                      component_lab: Optional[np.ndarray] = None,
                      start_component: int = 0) -> dict:
    """Evaluate one pose without discretizing a target into a centre circle."""
    x, y = float(xy[0]), float(xy[1])
    bbox = info["bbox_xyxy"]
    surface_distance = route_extractor._bbox_distance_xy((x, y), bbox)
    planning_radius = (float(config["robot_radius_m"])
                       + float(config["planning_margin_m"]))
    outside_bbox = surface_distance > 1e-9
    within_distance = surface_distance <= float(config["max_surface_distance_m"]) + 1e-9
    footprint_free = occ.is_free(x, y, planning_radius)
    clearance = float(occ.clearance(x, y))
    camera = CameraSpec(int(config.get("camera_width", 640)),
                        int(config.get("camera_height", 480)),
                        float(config.get("camera_hfov_deg", 115.0)))
    cx, cy = info["centre_xy"]
    target_yaw = math.atan2(float(cy) - y, float(cx) - x)
    projected = project_bbox(
        camera, (x, y, float(occ.ground_z(x, y))
                         + float(config.get("camera_height_m", 1.20))),
        target_yaw, info) if outside_bbox else None
    projection_ok = bool(
        projected is not None
        and projected["width_px"] >= float(config.get("min_bbox_width_px", 32.0))
        and projected["height_px"] >= float(config.get("min_bbox_height_px", 32.0))
        and projected["projected_area_ratio"] >= float(
            config.get("min_projected_area_ratio", 0.01))
        and 0.2 * camera.width <= projected["center_px"][0] <= 0.8 * camera.width
        and 0.1 * camera.height <= projected["center_px"][1] <= 0.9 * camera.height)
    # Projection is analytic and much cheaper than the collision-aware ray.
    # A weak/behind-camera target cannot enter the region, so avoid tracing a
    # LOS for those cells while retaining a strict fail-closed result.
    visible = (outside_bbox and projection_ok and visible_to_bbox_2d(
        occ, (x, y), info, float(config["robot_radius_m"])))
    room = _map_value(room_lab, occ, (x, y))
    component = _map_value(component_lab, occ, (x, y))
    room_ok = target_room <= 0 or room == target_room
    component_ok = start_component <= 0 or component == start_component
    accepted = bool(outside_bbox and within_distance and footprint_free and
                    visible and projection_ok and room_ok and component_ok)
    return {
        "accepted": accepted,
        "xy": [round(x, 4), round(y, 4)],
        "surface_distance_m": round(surface_distance, 4),
        "ground_z_m": round(float(occ.ground_z(x, y)), 4),
        "clearance_m": round(clearance, 4),
        "outside_bbox": outside_bbox,
        "within_max_distance": within_distance,
        "footprint_free": bool(footprint_free),
        "visible_to_bbox_2d": bool(visible),
        "target_facing_projection_ok": projection_ok,
        "target_facing_projection": projected,
        "room_id": room,
        "target_room_id": int(target_room),
        "room_ok": bool(room_ok),
        "component_id": component,
        "start_component_id": int(start_component),
        "component_ok": bool(component_ok),
    }


def object_side_region_cells(occ: OccMap, info: dict,
                             start_xy: Sequence[float],
                             goal_xy: Sequence[float],
                             config: Optional[dict] = None) -> dict:
    """Enumerate the 5 cm map cells accepted by the object-side contract.

    This is intended for cached top-down visualization and audit, not for
    scoring a continuous rollout pose (which uses ``_point_membership``
    directly).  Every cell still passes the exact same footprint, room,
    component, bbox-surface distance, and LOS checks as the scorer.
    """
    cfg = dict(DEFAULT_REGION_CONFIG)
    if config:
        cfg.update(config)
    room_lab = route_extractor._room_label_map(occ)
    component_lab = _component_map(occ, float(cfg["robot_radius_m"]))
    start_component = _map_value(component_lab, occ, start_xy)
    target_room, valid_stances = _infer_target_room(
        occ, room_lab, info, goal_xy, cfg)
    if target_room <= 0 or start_component <= 0:
        return {"available": False, "reason": "room_or_component_not_resolved",
                "cells_xy": [], "cell_size_m": float(occ.mpp)}

    x0, y0, x1, y1 = [float(value) for value in info["bbox_xyxy"]]
    margin = float(cfg["max_surface_distance_m"])
    corners = [occ.to_pixel(x0 - margin, y0 - margin),
               occ.to_pixel(x1 + margin, y1 + margin)]
    cols = range(max(0, min(point[0] for point in corners)),
                 min(occ.w - 1, max(point[0] for point in corners)) + 1)
    rows = range(max(0, min(point[1] for point in corners)),
                 min(occ.h - 1, max(point[1] for point in corners)) + 1)
    cells = []
    for row in rows:
        for col in cols:
            xy = occ.to_world(col, row)
            membership = _point_membership(
                occ, info, xy, cfg, room_lab=room_lab, target_room=target_room,
                component_lab=component_lab, start_component=start_component)
            if membership["accepted"]:
                cells.append((float(xy[0]), float(xy[1])))
    return {"available": True, "cells_xy": cells,
            "cell_size_m": float(occ.mpp), "config": cfg,
            "target_room_id": target_room,
            "start_component_id": start_component,
            "canonical_goal_stances_valid": valid_stances}


def _rollout_positions(rollout: dict) -> list[dict]:
    positions = []

    def append(source: str, info: dict, call_name: Optional[str] = None):
        xy = info.get("world_xy") or info.get("gps")
        if xy is not None:
            positions.append({"source": source, "call_name": call_name,
                              "xy": [float(xy[0]), float(xy[1])]})

    append("initial", rollout.get("initial_info") or {})
    for turn in rollout.get("turns", []):
        for call in turn.get("calls", []):
            append("call", call.get("info") or {}, call.get("name"))
    append("final", rollout.get("final") or {})
    return positions


def _stopped(rollout: dict) -> bool:
    return any(call.get("name") == "stop" or call.get("stop") is True
               for turn in rollout.get("turns", [])
               for call in turn.get("calls", []))


def _proxy_hits(occ: OccMap, label_index: Dict[str, dict], target_id: str,
                proxy_ids: set[str],
                xy: Sequence[float], config: dict,
                component_lab: Optional[np.ndarray], start_component: int) -> list[dict]:
    hits = []
    for object_id, info in label_index.items():
        label = str(info.get("label") or "").strip().lower().replace(" ", "_")
        if (object_id == target_id or object_id not in proxy_ids
                or not label or label in _NON_OBJECT_LABELS):
            continue
        bbox = info.get("bbox_xyxy")
        if bbox is None or route_extractor._bbox_distance_xy(
                (float(xy[0]), float(xy[1])), bbox) > float(
                    config["max_surface_distance_m"]) + 1e-9:
            continue
        membership = _point_membership(
            occ, info, xy, config, component_lab=component_lab,
            start_component=start_component)
        if membership["accepted"]:
            hits.append({"object_id": object_id, "label": info.get("label"),
                         "surface_distance_m": membership["surface_distance_m"]})
    return sorted(hits, key=lambda item: (item["surface_distance_m"], item["object_id"]))


def _evaluate_once(case: dict, rollout: dict, occ: OccMap,
                   label_index: Dict[str, dict], config: dict,
                   room_lab: Optional[np.ndarray], component_lab: np.ndarray,
                   target_room: int, start_component: int) -> dict:
    target_id = str(case.get("target_object_id") or "")
    target = label_index[target_id]
    explicit_proxies = set((case.get("oracle") or {}).get(
        "candidate_goal_xy", {}).keys()) - {target_id}
    family = semantic_family(target.get("label", ""))
    proxy_ids = explicit_proxies | {
        object_id for object_id, info in label_index.items()
        if object_id != target_id and semantic_family(info.get("label", "")) == family}
    positions = _rollout_positions(rollout)
    evaluated = []
    for position in positions:
        membership = _point_membership(
            occ, target, position["xy"], config, room_lab=room_lab,
            target_room=target_room, component_lab=component_lab,
            start_component=start_component)
        evaluated.append({**position, "target_region": membership})

    final_xy = ((rollout.get("final") or {}).get("world_xy") or
                (rollout.get("final") or {}).get("gps"))
    if final_xy is None and positions:
        final_xy = positions[-1]["xy"]
    final_target = (_point_membership(
        occ, target, final_xy, config, room_lab=room_lab,
        target_room=target_room, component_lab=component_lab,
        start_component=start_component) if final_xy is not None else None)
    proxies = (_proxy_hits(occ, label_index, target_id, proxy_ids, final_xy, config,
                           component_lab, start_component)
               if final_xy is not None else [])
    entered = any(item["target_region"]["accepted"] for item in evaluated)
    stopped = _stopped(rollout)
    target_at_stop = bool(stopped and final_target and final_target["accepted"])
    ambiguous = bool(target_at_stop and proxies)
    if ambiguous:
        event = "ambiguous_region_stop"
    elif target_at_stop:
        event = "correct_region_stop"
    elif stopped and proxies:
        event = "proxy_target_stop"
    elif entered:
        event = "pass_without_stop"
    elif stopped:
        event = "premature_stop"
    else:
        event = "no_stop_no_region_entry"
    return {
        "max_surface_distance_m": float(config["max_surface_distance_m"]),
        "event": event,
        "success": event == "correct_region_stop",
        "evaluation_valid": not ambiguous,
        "stopped": stopped,
        "entered_target_region": entered,
        "final_target_region": final_target,
        "final_proxy_regions": proxies,
        "trace": evaluated,
    }


def evaluate_semantic_terminal(case: dict, rollout: dict, sage_root: str,
                               config: Optional[dict] = None) -> dict:
    """Rescore one immutable rollout under primary and sensitivity contracts."""
    cfg = dict(DEFAULT_REGION_CONFIG)
    if config:
        cfg.update(config)
    target_id = str(case.get("target_object_id") or "")
    scene_dir = resolve_scene_dir(sage_root, str(case.get("scene") or ""))
    labels = load_label_index(scene_dir)
    if target_id not in labels:
        return {"schema_version": SEMANTIC_TERMINAL_SCHEMA, "available": False,
                "reason": "target_object_not_in_label_index", "target_object_id": target_id}
    rooms_root = str(Path(sage_root, "sage_usd_rooms_masks"))
    if Path(rooms_root).is_dir():
        # OccMap keeps this root as a module constant because env construction
        # normally sets it before import. Offline rescore must be equally exact
        # even when goal_region is imported directly from a clean process.
        occmap_module.USD_COLLISION_ROOT = rooms_root
    occ = OccMap(scene_dir, collision_mode="usd_rooms", furniture_mode="none")
    if occ.collision_mode != "usd_rooms":
        raise RuntimeError("semantic region requires usd_rooms collision artifacts")
    room_lab = route_extractor._room_label_map(occ)
    component_lab = _component_map(occ, float(cfg["robot_radius_m"]))
    golden = case.get("golden_path_xy") or []
    start_xy = golden[0] if golden else None
    goal_xy = golden[-1] if golden else None
    start_component = _map_value(component_lab, occ, start_xy) if start_xy else 0
    target_room, valid_stances = _infer_target_room(
        occ, room_lab, labels[target_id], goal_xy, cfg)
    if target_room <= 0:
        return {"schema_version": SEMANTIC_TERMINAL_SCHEMA, "available": False,
                "reason": "target_room_not_resolved", "target_object_id": target_id}
    if start_component <= 0:
        return {"schema_version": SEMANTIC_TERMINAL_SCHEMA, "available": False,
                "reason": "start_component_not_resolved", "target_object_id": target_id}

    primary = _evaluate_once(case, rollout, occ, labels, cfg, room_lab,
                             component_lab, target_room, start_component)
    sensitivity = {}
    for distance in SENSITIVITY_MAX_DISTANCE_M:
        probe = dict(cfg, max_surface_distance_m=float(distance))
        result = _evaluate_once(case, rollout, occ, labels, probe, room_lab,
                                component_lab, target_room, start_component)
        sensitivity["%.2f" % distance] = {
            "event": result["event"], "success": result["success"],
            "evaluation_valid": result["evaluation_valid"]}
    return {
        "schema_version": SEMANTIC_TERMINAL_SCHEMA,
        "available": True,
        "scene": str(case.get("scene")),
        "target_object_id": target_id,
        "target_label": labels[target_id].get("label"),
        "target_bbox_xyxy": [float(x) for x in labels[target_id]["bbox_xyxy"]],
        "config": cfg,
        "target_room_id": target_room,
        "start_component_id": start_component,
        "canonical_goal_stances_total": len(goal_stances(
            labels[target_id], cfg["goal_standoff_m"])),
        "canonical_goal_stances_valid": valid_stances,
        "construction_target_visible": (case.get("oracle") or {}).get("target_visible"),
        "primary": primary,
        "sensitivity": sensitivity,
    }


def object_height_relation(info: dict, ground_z: float) -> dict:
    """Conservative body-height overlap, not a proof of floor connectivity.

    Keep the official XY-only score separate. A disjoint object cannot be a
    physical arrival on the active floor; overlap alone does not establish
    that a staircase or a complete route exists.
    """
    try:
        low, high = (float(v) for v in info['z_range'])
        ground = float(ground_z)
        if not all(math.isfinite(v) for v in (low, high, ground)) or low > high:
            raise ValueError('invalid heights')
    except (KeyError, TypeError, ValueError):
        return dict(schema='sage-object-height-overlap-v1', available=False,
                    compatible=False, reason='object_height_unavailable')
    compatible = high >= ground - .10 and low <= ground + 1.55
    return dict(schema='sage-object-height-overlap-v1', available=True,
                compatible=compatible, ground_z_m=ground,
                object_z_range=[low, high], body_band_m=[ground-.10, ground+1.55],
                reason='overlap_not_connectivity_proof' if compatible
                       else 'object_disjoint_from_active_floor_body_band')


def architectural_anchor_floor(occ, info):
    """Use the native adapter's independently bound building support."""
    support = getattr(occ, 'sage_navmesh_support', None)
    if support is None:
        return None
    height = support.object_support_height(info)
    if height is not None and not math.isfinite(height):
        raise ValueError('invalid architectural anchor support')
    return height


def source_start_height_audit(task, occ, objects: dict) -> dict:
    """Reject only a nearby floor-standing source anchor on another height.

    Released trajectory Z is often a constant 0.5; never reinterpret it as
    a verified floor and never change the source XY to make the mask fit.
    Ceiling, wall-mounted and other ambiguous anchors remain unresolved.
    """
    oid = str(task.start_label or '').strip()
    info = objects.get(oid)
    x, y = float(task.start['x']), float(task.start['y'])
    ground = float(occ.ground_z(x, y))
    result = dict(schema='sage-source-start-height-audit-v1', start_object_id=oid,
                  source_xy=[x, y], source_z=task.start.get('z'),
                  active_ground_z_m=ground, reject=False,
                  reason='start_anchor_height_unresolved', changes_source_pose=False)
    if info is None:
        return result
    relation = object_height_relation(info, ground)
    family = semantic_family(info.get('label', ''))
    floor_standing = family in {'chair', 'table', 'sofa', 'cabinet', 'plant',
                               'bed', 'single_bed', 'double_bed', 'bunk_bed'}
    distance = route_extractor._bbox_distance_xy((x, y), info['bbox_xyxy'])
    result.update(height=relation, anchor_surface_distance_m=float(distance),
                  floor_standing_anchor=floor_standing)
    if getattr(occ, 'sage_navmesh_support', None) is not None and distance <= 1.5:
        anchor_floor = architectural_anchor_floor(occ, info)
        result.update(architectural_anchor_floor_m=anchor_floor,
                      navmesh_manifest_sha256=occ.navmesh_manifest_sha256,
                      height_evidence='architectural_support')
        if anchor_floor is None:
            result['reason'] = 'source_anchor_architectural_floor_unresolved'
        else:
            # Reuse resolve_start's existing .35m anchor/floor agreement bound.
            conflict = abs(anchor_floor-ground) > .35
            result.update(reject=conflict,
                          reason='start_anchor_architectural_floor_conflicts_with_active_floor'
                          if conflict else 'source_anchor_architectural_floor_matches_active_floor')
        return result
    if floor_standing and distance <= 1.5 and relation['available']:
        result.update(reject=not relation['compatible'],
                      reason='start_anchor_height_conflicts_with_active_floor'
                             if not relation['compatible']
                             else 'height_overlap_only_not_full_quality_approval')
    return result


def same_floor_task_audit(task, occ, objects: dict) -> dict:
    """Collection scope guard; height overlap is not a semantic approval.

    Do not infer floors from the released constant golden Z, or reject a
    staircase used solely as a landmark. Unknown target heights do not establish a cross-floor task.
    """
    text = str(getattr(task, 'instruction', '')).lower()
    motions = re.finditer(
        r"\b(?:go|walk|head|move|proceed|continue|climb|ascend|descend)\s+"
        r"(?:(?:up|down)\s+(?:the\s+)?(?:stairs|staircase|steps)|upstairs|downstairs)\b",
        text)
    motion = any(not re.search(r"(?:not|never|don't|avoid)\s+$", text[max(0, m.start()-16):m.start()])
                 for m in motions)
    opposite = bool(re.search(r'\bupstairs\b', text) and
                    re.search(r'\bdownstairs\b', text))
    ground = float(occ.ground_z(float(task.start['x']), float(task.start['y'])))
    goal = object_height_relation(objects.get(str(getattr(task, 'goal_label', '')).strip(), {}), ground)
    # Object body height is not floor height: chandeliers and wall fixtures
    # routinely lie above the agent on the very same floor. Compare only
    # nearby source and target floor-standing anchors with low body extents.
    families = {'chair', 'table', 'sofa', 'bed', 'single_bed', 'double_bed'}
    def floor_anchor(oid):
        info = objects.get(str(oid or '').strip(), {})
        if semantic_family(info.get('label', '')) not in families:
            return None
        if getattr(occ, 'sage_navmesh_support', None) is not None:
            # A chair on a platform or furniture is not another building floor.
            # Missing architectural evidence stays unknown, rather than using
            # an object's base as a substitute for a storey.
            return architectural_anchor_floor(occ, info)
        relation = object_height_relation(info, ground)
        if not relation['available']:
            return None
        low, high = relation['object_z_range']
        if high-low > 1.8:
            return None
        return low
    source_id = getattr(task, 'start_label', '')
    start_floor = floor_anchor(source_id)
    goal_floor = floor_anchor(getattr(task, 'goal_label', ''))
    nearby = False
    if start_floor is not None:
        bbox = objects[str(source_id).strip()].get('bbox_xyxy')
        nearby = bbox is not None and route_extractor._bbox_distance_xy(
            (float(task.start['x']), float(task.start['y'])), bbox) <= 1.5
    different = bool(nearby and goal_floor is not None and
                     abs(start_floor-goal_floor) >= 2.0)
    reason = ('sage_cross_floor_instruction_excluded' if motion or opposite else
              'sage_cross_floor_standing_anchors_excluded' if different else
              'cross_floor_not_established_requires_trajectory_review')
    return dict(schema='sage-same-floor-scope-v2', reject=bool(motion or opposite or different),
                reason=reason, goal_height=goal, source_anchor_floor_m=start_floor,
                goal_anchor_floor_m=goal_floor, nearby_source_anchor=nearby,
                quality_approved=False)
