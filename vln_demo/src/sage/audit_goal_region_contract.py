"""Calibrate the SAGE semantic terminal-region contract on construction scenes.

This is a geometry-only audit.  It reuses the frozen construction/evaluation
scene split, physically reachable train tasks, :mod:`sage.goal_region`, and the
canonical ``usd_rooms`` collision map.  It never reads evaluation manifests,
contacts a policy model, or mutates benchmark data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from . import occmap as occmap_module
from . import route_extractor
from .capability_scene_split import (DEFAULT_POOL, DEFAULT_SALT,
                                     load_scene_pool, partition_scenes)
from .datasets import SAGE_ROOT, SageSet
from .goal_region import (DEFAULT_REGION_CONFIG, SENSITIVITY_MAX_DISTANCE_M,
                          _component_map, _infer_target_room, _map_value,
                          _point_membership, goal_stances, load_label_index,
                          semantic_family)
from .occmap import OccMap
from .visibility_audit import visible_to_bbox_2d


SCHEMA_VERSION = "sage-semantic-terminal-calibration-v1"
ALLOWED_FAMILIES = frozenset(("cabinet", "chair", "table", "sofa"))
SIZE_SMALL_MAX_M = 0.75
SIZE_LARGE_MIN_M = 1.50
LOCAL_DISTRACTOR_MAX_M = 6.0
SAMPLES_PER_STRATUM = 3
MAX_SCENES = 160


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_object_id(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _bbox_gap(left: Sequence[float], right: Sequence[float]) -> float:
    lx0, ly0, lx1, ly1 = [float(v) for v in left]
    rx0, ry0, rx1, ry1 = [float(v) for v in right]
    dx = max(lx0 - rx1, rx0 - lx1, 0.0)
    dy = max(ly0 - ry1, ry0 - ly1, 0.0)
    return math.hypot(dx, dy)


def _size_bucket(info: dict) -> Optional[str]:
    x0, y0, x1, y1 = info["bbox_xyxy"]
    span = max(float(x1) - float(x0), float(y1) - float(y0))
    if span <= SIZE_SMALL_MAX_M:
        return "small"
    if span >= SIZE_LARGE_MIN_M:
        return "large"
    return None


def _stance_records(occ: OccMap, info: dict, config: dict,
                    room_lab: np.ndarray, target_room: int,
                    component_lab: np.ndarray, start_component: int) -> list[dict]:
    x0, y0, x1, y1 = info["bbox_xyxy"]
    records = []
    for distance in config["goal_standoff_m"]:
        for fraction in (0.2, 0.5, 0.8):
            x = x0 + fraction * (x1 - x0)
            y = y0 + fraction * (y1 - y0)
            for face, xy in (("left", (x0 - distance, y)),
                             ("right", (x1 + distance, y)),
                             ("bottom", (x, y0 - distance)),
                             ("top", (x, y1 + distance))):
                membership = _point_membership(
                    occ, info, xy, config, room_lab=room_lab,
                    target_room=target_room, component_lab=component_lab,
                    start_component=start_component)
                records.append({"face": face, "distance_m": float(distance),
                                "xy": [float(xy[0]), float(xy[1])],
                                "accepted": membership["accepted"]})
    return records


def _path_point_before_end(path: Sequence[Sequence[float]], back_m: float):
    if not path:
        return None
    left = float(back_m)
    for index in range(len(path) - 1, 0, -1):
        x1, y1 = float(path[index][0]), float(path[index][1])
        x0, y0 = float(path[index - 1][0]), float(path[index - 1][1])
        length = math.hypot(x1 - x0, y1 - y0)
        if length >= left and length > 1e-9:
            ratio = left / length
            return [x1 + (x0 - x1) * ratio, y1 + (y0 - y1) * ratio]
        left -= length
    return [float(path[0][0]), float(path[0][1])]


def _local_distractors(occ: OccMap, labels: dict, target_id: str,
                       target_room: int, component_lab: np.ndarray,
                       start_component: int, config: dict) -> list[str]:
    target = labels[target_id]
    family = semantic_family(target.get("label", ""))
    found = []
    for object_id, info in labels.items():
        if object_id == target_id or semantic_family(info.get("label", "")) != family:
            continue
        if _bbox_gap(target["bbox_xyxy"], info["bbox_xyxy"]) > LOCAL_DISTRACTOR_MAX_M:
            continue
        valid = False
        for stance in goal_stances(info, config["goal_standoff_m"]):
            membership = _point_membership(
                occ, info, stance, config, room_lab=None, target_room=0,
                component_lab=component_lab, start_component=start_component)
            if membership["accepted"]:
                valid = True
                break
        if valid:
            found.append(object_id)
    return sorted(found)


def _region_cells(occ: OccMap, info: dict, config: dict,
                  room_lab: np.ndarray, target_room: int,
                  component_lab: np.ndarray, start_component: int) -> list[list[float]]:
    x0, y0, x1, y1 = [float(value) for value in info["bbox_xyxy"]]
    margin = float(config["max_surface_distance_m"])
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
            if _point_membership(
                    occ, info, xy, config, room_lab=room_lab,
                    target_room=target_room, component_lab=component_lab,
                    start_component=start_component)["accepted"]:
                cells.append([float(xy[0]), float(xy[1])])
    return cells


def _rejection_counts(occ: OccMap, info: dict, config: dict,
                      room_lab: np.ndarray, target_room: int,
                      component_lab: np.ndarray, start_component: int) -> dict:
    """Count why geometrically nearby cells do not enter the region."""
    x0, y0, x1, y1 = [float(value) for value in info["bbox_xyxy"]]
    margin = float(config["max_surface_distance_m"])
    corners = [occ.to_pixel(x0 - margin, y0 - margin),
               occ.to_pixel(x1 + margin, y1 + margin)]
    counts = Counter()
    for row in range(max(0, min(p[1] for p in corners)),
                     min(occ.h - 1, max(p[1] for p in corners)) + 1):
        for col in range(max(0, min(p[0] for p in corners)),
                         min(occ.w - 1, max(p[0] for p in corners)) + 1):
            xy = occ.to_world(col, row)
            result = _point_membership(
                occ, info, xy, config, room_lab=room_lab,
                target_room=target_room, component_lab=component_lab,
                start_component=start_component)
            if not result["outside_bbox"] or not result["within_max_distance"]:
                continue
            if result["accepted"]:
                counts["accepted"] += 1
            else:
                for key in ("footprint_free", "visible_to_bbox_2d", "room_ok",
                            "target_facing_projection_ok", "component_ok"):
                    if not result[key]:
                        counts["rejected_" + key] += 1
    return dict(sorted(counts.items()))


def _overlap_cells(occ: OccMap, cells: Iterable[Sequence[float]], labels: dict,
                   proxy_ids: Sequence[str], config: dict,
                   component_lab: np.ndarray, start_component: int) -> int:
    overlap = 0
    for xy in cells:
        if any(_point_membership(
                occ, labels[proxy_id], xy, config, room_lab=None, target_room=0,
                component_lab=component_lab,
                start_component=start_component)["accepted"]
               for proxy_id in proxy_ids):
            overlap += 1
    return overlap


def _task_candidates(ds: SageSet, bare: str, kept_ids: set[str]) -> list:
    candidates = []
    seen_targets = set()
    for task in ds.tasks(ds.qualify(bare), 0):
        if task.task_id not in kept_ids:
            continue
        target_id = _normalize_object_id(task.goal_label)
        if target_id in seen_targets:
            continue
        seen_targets.add(target_id)
        candidates.append((task, target_id))
    return candidates


def _load_kept(path: str) -> dict[str, set[str]]:
    out = defaultdict(set)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            out[str(item["scene"])].add(str(item["task_id"]))
    return out


def calibrate(sage_root: str, pool_path: str, output_path: str,
              max_scenes: int = MAX_SCENES,
              samples_per_stratum: int = SAMPLES_PER_STRATUM) -> dict:
    rooms_root = str(Path(sage_root, "sage_usd_rooms_masks"))
    if not Path(rooms_root).is_dir():
        raise FileNotFoundError("missing usd_rooms masks: " + rooms_root)
    occmap_module.USD_COLLISION_ROOT = rooms_root
    kept = _load_kept(pool_path)
    construction = partition_scenes(load_scene_pool(pool_path), "construction",
                                    max_scenes, DEFAULT_SALT)
    ds = SageSet("train", root=sage_root)
    selected = defaultdict(list)
    target_per_stratum = int(samples_per_stratum)
    wanted = {(size, layout, distractor)
              for size in ("small", "large")
              for layout in ("wall_adjacent", "open")
              for distractor in ("with_distractor", "without_distractor")}

    for bare in construction:
        labels = load_label_index(ds.scene_dir(bare))
        raw_candidates = _task_candidates(ds, bare, kept.get(bare, set()))
        ds._tasks.clear()
        if not raw_candidates:
            continue
        occ = OccMap(ds.scene_dir(bare), collision_mode="usd_rooms",
                     furniture_mode="none")
        if occ.collision_mode != "usd_rooms":
            raise RuntimeError("usd_rooms artifact missing for " + bare)
        room_lab = route_extractor._room_label_map(occ)
        component_lab = _component_map(occ, DEFAULT_REGION_CONFIG["robot_radius_m"])
        for task, target_id in raw_candidates:
            if target_id not in labels:
                continue
            info = labels[target_id]
            family = semantic_family(info.get("label", ""))
            size = _size_bucket(info)
            if family not in ALLOWED_FAMILIES or size is None:
                continue
            start_xy = [float(task.start["x"]), float(task.start["y"])]
            goal_xy = [float(task.goal_xy[0]), float(task.goal_xy[1])]
            start_component = _map_value(component_lab, occ, start_xy)
            target_room, _ = _infer_target_room(
                occ, room_lab, info, goal_xy, DEFAULT_REGION_CONFIG)
            if start_component <= 0 or target_room <= 0:
                continue
            stance_records = _stance_records(
                occ, info, DEFAULT_REGION_CONFIG, room_lab, target_room,
                component_lab, start_component)
            valid_faces = sorted({item["face"] for item in stance_records
                                  if item["accepted"]})
            if not valid_faces:
                continue
            layout = "wall_adjacent" if len(valid_faces) <= 2 else "open"
            proxies = _local_distractors(
                occ, labels, target_id, target_room, component_lab,
                start_component, DEFAULT_REGION_CONFIG)
            distractor = "with_distractor" if proxies else "without_distractor"
            key = (size, layout, distractor)
            if key not in wanted or len(selected[key]) >= target_per_stratum:
                continue

            threshold_metrics = {}
            path_xy = [[float(p[0]), float(p[1])] for p in task.waypoints]
            early_xy = {"1.0": _path_point_before_end(path_xy, 1.0),
                        "2.0": _path_point_before_end(path_xy, 2.0)}
            for threshold in SENSITIVITY_MAX_DISTANCE_M:
                cfg = dict(DEFAULT_REGION_CONFIG,
                           max_surface_distance_m=float(threshold))
                cells = _region_cells(
                    occ, info, cfg, room_lab, target_room, component_lab,
                    start_component)
                endpoint = _point_membership(
                    occ, info, goal_xy, cfg, room_lab=room_lab,
                    target_room=target_room, component_lab=component_lab,
                    start_component=start_component)
                stances = _stance_records(
                    occ, info, cfg, room_lab, target_room, component_lab,
                    start_component)
                early = {distance: (_point_membership(
                    occ, info, xy, cfg, room_lab=room_lab,
                    target_room=target_room, component_lab=component_lab,
                    start_component=start_component) if xy is not None else None)
                         for distance, xy in early_xy.items()}
                overlap = _overlap_cells(
                    occ, cells, labels, proxies, cfg, component_lab,
                    start_component) if proxies else 0
                threshold_metrics["%.2f" % threshold] = {
                    "region_cells": len(cells),
                    "region_area_m2": round(len(cells) * occ.mpp * occ.mpp, 4),
                    "golden_endpoint": endpoint,
                    "canonical_stances_accepted": sum(
                        int(item["accepted"]) for item in stances),
                    "canonical_stances_total": len(stances),
                    "early_path": early,
                    "proxy_overlap_cells": overlap,
                    "proxy_overlap_fraction": round(
                        overlap / float(max(len(cells), 1)), 6),
                    "rejection_counts": _rejection_counts(
                        occ, info, cfg, room_lab, target_room,
                        component_lab, start_component),
                }
            primary_cells = _region_cells(
                occ, info, DEFAULT_REGION_CONFIG, room_lab, target_room,
                component_lab, start_component)
            if not primary_cells:
                continue
            x0, y0, x1, y1 = info["bbox_xyxy"]
            max_span = max(float(x1) - float(x0), float(y1) - float(y0))
            representative = min(primary_cells, key=lambda xy: abs(
                route_extractor._bbox_distance_xy(xy, info["bbox_xyxy"]) - 0.90))
            selected[key].append({
                "scene": bare,
                "task_id": task.task_id,
                "target_object_id": target_id,
                "target_label": info.get("label"),
                "semantic_family": family,
                "bbox_xyxy": [float(v) for v in info["bbox_xyxy"]],
                "bbox_z_range": [float(v) for v in info["z_range"]],
                "max_horizontal_span_m": round(max_span, 4),
                "size_bucket": size,
                "layout_bucket": layout,
                "distractor_bucket": distractor,
                "local_distractor_ids": proxies,
                "valid_approach_faces": valid_faces,
                "start_xy": start_xy,
                "golden_endpoint_xy": goal_xy,
                "representative_region_xy": representative,
                "target_room_id": target_room,
                "start_component_id": start_component,
                "thresholds": threshold_metrics,
            })
        if all(len(selected[key]) >= target_per_stratum for key in wanted):
            break

    cases = []
    for key in sorted(wanted):
        cases.extend(selected.get(key, []))
    aggregate = {}
    for threshold in SENSITIVITY_MAX_DISTANCE_M:
        name = "%.2f" % threshold
        rows = [case["thresholds"][name] for case in cases]
        endpoints = [row["golden_endpoint"]["accepted"] for row in rows]
        early_1 = [row["early_path"]["1.0"]["accepted"] for row in rows]
        early_2 = [row["early_path"]["2.0"]["accepted"] for row in rows]
        stance_num = sum(row["canonical_stances_accepted"] for row in rows)
        stance_den = sum(row["canonical_stances_total"] for row in rows)
        aggregate[name] = {
            "cases": len(rows),
            "nonempty_region_rate": (sum(row["region_cells"] > 0 for row in rows)
                                     / float(max(len(rows), 1))),
            "golden_endpoint_accept_rate": sum(endpoints) / float(max(len(rows), 1)),
            "canonical_stance_accept_rate": stance_num / float(max(stance_den, 1)),
            "early_1m_accept_rate": sum(early_1) / float(max(len(rows), 1)),
            "early_2m_accept_rate": sum(early_2) / float(max(len(rows), 1)),
            "cases_with_proxy_overlap": sum(row["proxy_overlap_cells"] > 0
                                             for row in rows),
            "mean_region_area_m2": (sum(row["region_area_m2"] for row in rows)
                                    / float(max(len(rows), 1))),
        }
    coverage = {"/".join(key): len(selected.get(key, [])) for key in sorted(wanted)}
    marginal_coverage = {
        "size": dict(Counter(case["size_bucket"] for case in cases)),
        "layout": dict(Counter(case["layout_bucket"] for case in cases)),
        "distractor": dict(Counter(case["distractor_bucket"] for case in cases)),
    }
    marginal_complete = all(
        marginal_coverage[axis].get(value, 0) >= target_per_stratum
        for axis, values in {
            "size": ("small", "large"),
            "layout": ("wall_adjacent", "open"),
            "distractor": ("with_distractor", "without_distractor"),
        }.items() for value in values)
    max_stance_rate = max((row["canonical_stance_accept_rate"]
                           for row in aggregate.values()), default=0.0)
    retained = [float(name) for name, row in aggregate.items()
                if abs(row["canonical_stance_accept_rate"] - max_stance_rate) < 1e-12]
    recommended_threshold = min(retained) if retained else None
    for row in aggregate.values():
        row["canonical_stance_retention_vs_best"] = (
            row["canonical_stance_accept_rate"] / max(max_stance_rate, 1e-12))
    result = {
        "schema_version": SCHEMA_VERSION,
        "partition": "construction",
        "scene_split_salt": DEFAULT_SALT,
        "evaluation_data_used_for_tuning": False,
        "sage_root": str(Path(sage_root).resolve()),
        "reachable_pool": str(Path(pool_path).resolve()),
        "reachable_pool_sha256": _sha256(pool_path),
        "selection": {
            "max_scenes": int(max_scenes),
            "scenes_scanned": construction[:max_scenes],
            "allowed_semantic_families": sorted(ALLOWED_FAMILIES),
            "size_small_max_m": SIZE_SMALL_MAX_M,
            "size_large_min_m": SIZE_LARGE_MIN_M,
            "layout_definition": "wall_adjacent iff <=2 bbox approach faces have an accepted canonical stance; open iff >=3",
            "local_distractor_definition": "same semantic family, bbox gap <=6m, with a reachable visible canonical stance in the start component",
            "samples_per_stratum": target_per_stratum,
        },
        "stratum_coverage": coverage,
        "marginal_coverage": marginal_coverage,
        "marginal_coverage_complete": marginal_complete,
        "factorial_coverage_complete": all(
            value >= target_per_stratum for value in coverage.values()),
        "missing_factorial_strata": [key for key, value in coverage.items()
                                     if value < target_per_stratum],
        "cases": cases,
        "aggregate": aggregate,
        "decision": {
            "recommended_max_surface_distance_m": recommended_threshold,
            "rule": "choose the smallest threshold retaining all canonical stances accepted by any tested threshold; use proxy overlap and premature-path rates to reject larger ties",
            "raw_gvln_endpoint_is_diagnostic_only": True,
            "reason": "raw GVLN endpoints were not produced by the canonical stance builder and are not used to tune the semantic threshold",
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sage-root", default=SAGE_ROOT)
    parser.add_argument("--pool", default=DEFAULT_POOL)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-scenes", type=int, default=MAX_SCENES)
    parser.add_argument("--samples-per-stratum", type=int,
                        default=SAMPLES_PER_STRATUM)
    args = parser.parse_args()
    result = calibrate(args.sage_root, args.pool, args.output,
                       args.max_scenes, args.samples_per_stratum)
    print(json.dumps({"output": str(Path(args.output).resolve()),
                      "cases": len(result["cases"]),
                      "marginal_coverage_complete": result["marginal_coverage_complete"],
                      "factorial_coverage_complete": result["factorial_coverage_complete"],
                      "stratum_coverage": result["stratum_coverage"],
                      "aggregate": result["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
