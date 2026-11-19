#!/usr/bin/env python3
"""Replay every merged curriculum path against the fixed usd_rooms contract."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def replay(occ, waypoints, radius: float, step_m: float) -> dict:
    points = np.asarray([(float(p[0]), float(p[1])) for p in waypoints], dtype=float)
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(segments.sum())
    if len(points) < 2 or total < 1e-6:
        return {"ok": False, "reason": "short_or_zero_path"}
    cum = np.concatenate([[0.0], np.cumsum(segments)])
    count = max(2, int(math.ceil(total / step_m)))
    distances = np.linspace(0.0, total, count)
    path = np.column_stack((np.interp(distances, cum, points[:, 0]),
                            np.interp(distances, cum, points[:, 1])))
    x, y = map(float, path[0])
    hits = 0
    for index, target in enumerate(path[1:], 1):
        tx, ty = map(float, target)
        requested = math.hypot(tx - x, ty - y)
        nx, ny, hit = occ.try_step(x, y, tx, ty, radius, True)
        moved = math.hypot(nx - x, ny - y)
        hits += int(bool(hit))
        if requested > 1e-9 and moved < requested * 0.95:
            return {"ok": False, "reason": "blocked", "step": index,
                    "requested_m": requested, "moved_m": moved,
                    "xy": [x, y], "target_xy": [tx, ty], "hits": hits}
        x, y = nx, ny
    return {"ok": True, "steps": len(path) - 1, "hits": hits,
            "end_error_m": math.hypot(x - path[-1, 0], y - path[-1, 1])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="action_head_curriculum")
    parser.add_argument("--radius-m", type=float, default=0.07)
    parser.add_argument("--step-m", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from sage.datasets import available
    from sage.occmap import OccMap

    dataset = available()[args.split]
    rows, cache = [], {}
    for scene in dataset.list_scenes():
        if scene not in cache:
            cache[scene] = OccMap(dataset.scene_dir(scene), collision_mode="usd_rooms",
                                  furniture_mode="none")
        occ = cache[scene]
        for task in dataset.tasks(scene, 0):
            stage = task.instruction_type.rsplit("/", 1)[-1].upper()
            result = replay(occ, task.waypoints, args.radius_m, args.step_m)
            rows.append({"scene": scene, "episode_id": task.task_id,
                         "stage": stage, **result})
    bad = [row for row in rows if not row["ok"]]
    report = {"schema": "action_head_curriculum_usd_rooms_replay_v1",
              "contract": {"collision": "usd_rooms", "furniture": "none",
                           "robot_radius_m": args.radius_m, "step_m": args.step_m},
              "trajectory_count": len(rows), "failed_count": len(bad),
              "stage_counts": dict(Counter(row["stage"] for row in rows)),
              "failed": bad}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("trajectory_count", "failed_count", "stage_counts")},
                     ensure_ascii=False, indent=2))
    if bad:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
