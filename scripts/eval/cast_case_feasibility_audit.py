"""CPU-only geometric probes for a saved, deterministic v4.3 case sample.

Reuses SAGE planning, smoothing, replay and label readers. These are candidate
paths, not accepted counterfactual training labels or policy evaluations.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vln_demo/src"))

from sage.datasets import SageSet
from sage.occmap import OccMap
from sage.planner import plan_path, smooth_path, _path_is_safe
from sage.build_demo_episodes import _execution_replay_free
from sage.goal_region import load_label_index
from sage.visibility_audit import project_bbox, visible_to_bbox_2d
from rs3d.gs_client import CameraSpec

OUTPUT = Path("/data/zerovln/"
              "vln_verl/artifacts/sage_full_status_analysis_20260914/cast_case_audit")
RADIUS = 0.07


def length(points):
    return float(np.linalg.norm(np.diff(np.asarray(points)[:, :2], axis=0), axis=1).sum())


def checked_path(occ, start, end):
    if not occ.is_free(*end, RADIUS):
        return None, {"reason": "endpoint_blocked"}
    raw = plan_path(occ, start, end, RADIUS, mode="ecm")
    if raw is None:
        return None, {"reason": "no_path"}
    points, smoothing = smooth_path(occ, raw, RADIUS)
    if not _path_is_safe(occ, np.asarray(points), RADIUS):
        return None, {"reason": "unsafe_smoothed_path"}
    ok, replay = _execution_replay_free(occ, points, step_m=0.1, radius_m=RADIUS)
    if not ok:
        return None, {"reason": "replay_failed", "replay": replay}
    return points, {"length_m": length(points), "replay": replay,
                    "smoothing": smoothing}


def main():
    source = OUTPUT / "sampled_anchors.json"
    sample = json.loads(source.read_text())
    dataset = SageSet(split="official_gvln_action_head_p0_v3full827train")
    camera = CameraSpec(width=640, height=480, hfov_deg=115)
    results = []
    for index, row in enumerate(sample["anchors"]):
        scene_dir = dataset.scene_dir(row["source_scene"])
        occ = OccMap(scene_dir, collision_mode="usd_rooms", furniture_mode="none")
        task = next(t for t in dataset.tasks(row["source_scene"])
                    if t.task_id == row["source_task_id"])
        x, y, yaw = row["source_pose_xyyaw"]
        baseline, baseline_info = checked_path(occ, (x, y), task.goal_xy)
        labels = load_label_index(scene_dir)
        objects = []
        for oid, info in labels.items():
            distance = math.dist((x, y), info["centre_xy"])
            if distance > 5:
                continue
            projection = project_bbox(camera, (x, y, 1.2), yaw, info)
            if projection is None or projection["projected_area_ratio"] <= 0:
                continue
            objects.append({"id": oid, "label": info["label"],
                            "distance_m": distance, "projection": projection,
                            "los_2d": visible_to_bbox_2d(occ, (x, y), info, RADIUS),
                            "bbox_xyxy": info["bbox_xyxy"]})
        candidates = []
        for distance in (0.8, 1.2, 1.6):
            for angle in (-45, -20, 0, 20, 45):
                heading = yaw + math.radians(angle)
                endpoint = (x + distance * math.cos(heading),
                            y + distance * math.sin(heading))
                path, info = checked_path(occ, (x, y), endpoint)
                candidate = {"distance_m": distance, "bearing_left_deg": angle,
                             "endpoint_xy": endpoint, "local_check": info,
                             "path_xy": path, "within_1p6m_window": False,
                             "geometric_progress": False}
                if path is not None and info["length_m"] <= 1.65:
                    candidate["within_1p6m_window"] = True
                    continuation, continuation_info = checked_path(occ, endpoint, task.goal_xy)
                    candidate["continuation_check"] = continuation_info
                    if continuation is not None and baseline is not None:
                        remaining = continuation_info["length_m"]
                        candidate["remaining_length_m"] = remaining
                        candidate["geometric_progress"] = remaining < baseline_info["length_m"] - 0.1
                        candidate["total_detour_m"] = info["length_m"] + remaining - baseline_info["length_m"]
                candidates.append(candidate)
        with np.load(occ.usd_mask_path, allow_pickle=False) as mask:
            object_cells = int(np.count_nonzero(mask["object_obstacle"]))
        goal_info = labels.get(task.goal_label)
        result = {"index": index, "anchor": row["slow_anchor_id"],
                  "instruction": row["instruction"], "contract": row["contract"],
                  "current_image": row["current_image"], "scene_dir": scene_dir,
                  "pose_xyyaw": [x, y, yaw], "goal_xy": task.goal_xy,
                  "goal_label": task.goal_label, "goal_object": goal_info,
                  "goal_projection_without_distance_cutoff": (
                      project_bbox(camera, (x, y, 1.2), yaw, goal_info) if goal_info else None),
                  "mask_path": occ.usd_mask_path, "mask_version": occ._usd_version,
                  "object_obstacle_cells": object_cells,
                  "golden_suffix_xy": [list(p[:2]) for p in task.waypoints[row["source_point_index"]:]],
                  "baseline_path_xy": baseline, "baseline_check": baseline_info,
                  "nearby_projected_objects": sorted(objects, key=lambda o: o["distance_m"]),
                  "candidates": candidates}
        results.append(result)
        valid = [c for c in candidates if c["within_1p6m_window"] and c["geometric_progress"]]
        print(json.dumps({"case": index, "anchor": row["slow_anchor_id"],
                          "goal": task.goal_label,
                          "geometric_progress_candidates": [(c["distance_m"], c["bearing_left_deg"]) for c in valid],
                          "object_obstacle_cells": object_cells}), flush=True)
    output = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "script": str(Path(__file__).resolve()),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "limitations": ["Geometric progress does not certify instruction correctness or visibility.",
                              "No new RGB/depth renders or policy execution were performed.",
                              "Projected object boxes and 2D LOS are not semantic/depth verification.",
                              "The small deterministic sample is not a dataset-wide yield estimate.",
                              "Polar endpoint probes are not exhaustive route-topology search."],
              "cases": results}
    (OUTPUT / "geometry_probes.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
