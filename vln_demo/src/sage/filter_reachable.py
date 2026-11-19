"""Filter SAGE-3D GVLN episodes for physical trainability under the default
`usd_rooms` render/collision mode.

An episode is kept iff, on the scene's `usd_rooms` occupancy grid:
  1. start is on open floor outside furniture -- `is_free(start, radius)` True;
  2. goal  is on open floor outside furniture -- `is_free(goal , radius)` True;
  3. goal is reachable from start -- both land in the same connected component
     of the radius-eroded free grid (flood-fill over `clearance_m > radius`).

Reuses `sage.datasets.SageSet` (offset-corrected start/goal enumeration) and
`sage.occmap.OccMap` (`is_free`, `clearance_m`, `to_pixel`). Connectivity is the
only new bit -- the repo has no BFS/flood-fill -- done with `scipy.ndimage.label`
on the same passable mask `is_free` implies (8-connectivity; a disk robot can
cross a diagonal pinch only where clearance already exceeds the radius).

Radius defaults to 0.07 m to match `start_sage_usd.sh` (canonical usd_rooms
launcher); mask root defaults to `sage_usd_rooms_masks` (the v4 rooms masks).

Run (no CLI needed for the common case; flags are inlined-overridable):
    cd vln_demo/src && PYTHONPATH=. \
      /opt/navigation-runtime/venv/bin/python \
      -m sage.filter_reachable --split instr
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from . import occmap as _occmap_mod
from .datasets import SAGE_ROOT, SageSet
from .occmap import OccMap

try:
    from scipy.ndimage import label as _cc_label
    _EIGHT = np.ones((3, 3), dtype=np.uint8)
except Exception:                                   # pragma: no cover
    _cc_label = None
    _EIGHT = None


def _passable(occ: OccMap, radius_m: float) -> np.ndarray:
    """Boolean grid the robot centre can occupy: free AND clearance > radius.

    Mirrors `OccMap.is_free` exactly so component membership agrees with the
    per-point checks. Falls back to raw `free` when scipy/EDT is unavailable.
    """
    cm = getattr(occ, "clearance_m", None)
    if cm is None:
        return np.asarray(occ.free, dtype=bool)
    return np.asarray(occ.free, dtype=bool) & (np.asarray(cm) > radius_m)


def _components(passable: np.ndarray):
    if _cc_label is None:
        return None
    lab, _ = _cc_label(passable, structure=_EIGHT)
    return lab


def _label_at(lab, occ: OccMap, x: float, y: float) -> int:
    px, py = occ.to_pixel(x, y)
    if not (0 <= px < occ.w and 0 <= py < occ.h):
        return 0
    return int(lab[py, px])


def filter_scene(occ: OccMap, tasks, radius_m: float):
    """Classify a scene's tasks. Returns (kept, stats dict, per-route dict)."""
    passable = _passable(occ, radius_m)
    lab = _components(passable)
    kept = []
    n_start_bad = n_goal_bad = n_unreach = 0
    routes = {}                     # (start,goal) rounded -> reachable? (dedup)
    for t in tasks:
        sx, sy = t.start["x"], t.start["y"]
        gx, gy = t.goal_xy
        s_ok = occ.is_free(sx, sy, radius_m)
        g_ok = occ.is_free(gx, gy, radius_m)
        if not s_ok:
            n_start_bad += 1
        if not g_ok:
            n_goal_bad += 1
        reach = False
        if s_ok and g_ok:
            if lab is None:                         # no scipy: fall back to LoS
                reach = occ.seg_free(sx, sy, gx, gy, radius_m)
            else:
                ls = _label_at(lab, occ, sx, sy)
                lg = _label_at(lab, occ, gx, gy)
                reach = ls != 0 and ls == lg
            if not reach:
                n_unreach += 1
        if s_ok and g_ok and reach:
            kept.append(t)
        rk = (round(sx, 2), round(sy, 2), round(gx, 2), round(gy, 2))
        routes.setdefault(rk, s_ok and g_ok and reach)
    stats = dict(total=len(tasks), kept=len(kept),
                 start_bad=n_start_bad, goal_bad=n_goal_bad,
                 unreachable=n_unreach,
                 routes_total=len(routes),
                 routes_kept=sum(1 for v in routes.values() if v))
    return kept, stats


def run(split: str, radius_m: float, mask_root: str, collision: str,
        out_dir: str, limit_scenes: int = 0):
    os.environ["SAGE_COLLISION"] = collision
    os.environ["SAGE_USD_COLLISION_MASK_ROOT"] = mask_root
    _occmap_mod.USD_COLLISION_ROOT = mask_root   # captured at import; override
    ds = SageSet(split=split)
    scenes = ds.list_scenes()
    if limit_scenes:
        scenes = scenes[:limit_scenes]
    print(f"[filter] split={split} collision={collision} radius={radius_m}m "
          f"mask_root={mask_root}", flush=True)
    print(f"[filter] {len(scenes)} scenes with traj+occupancy", flush=True)

    agg = dict(scenes=0, scenes_with_kept=0, scenes_mask_missing=0,
               total=0, kept=0, routes_total=0, routes_kept=0,
               start_bad=0, goal_bad=0, unreachable=0)
    per_scene = []
    kept_records = []
    t0 = time.time()
    for i, scene in enumerate(scenes):
        try:
            occ = OccMap(ds.scene_dir(scene), collision_mode=collision,
                         furniture_mode="none")
        except Exception as e:
            print(f"  [{i+1}/{len(scenes)}] {scene} OCCMAP-FAIL {e}", flush=True)
            continue
        mask_missing = bool(getattr(occ, "usd_mask_missing", False)
                            or getattr(occ, "usd_rooms_mask_missing", False))
        tasks = ds.tasks(scene, limit=0)
        kept, st = filter_scene(occ, tasks, radius_m)
        agg["scenes"] += 1
        if mask_missing:
            agg["scenes_mask_missing"] += 1
        if st["kept"]:
            agg["scenes_with_kept"] += 1
        for k in ("total", "kept", "start_bad", "goal_bad", "unreachable",
                  "routes_total", "routes_kept"):
            agg[k] += st[k]
        st["scene"] = ds.bare(scene)
        st["mask_missing"] = mask_missing
        per_scene.append(st)
        for t in kept:
            kept_records.append(dict(scene=ds.bare(scene), task_id=t.task_id,
                                     start=[t.start["x"], t.start["y"]],
                                     goal=list(t.goal_xy), ref_len=t.ref_len))
        flag = " MASK-MISSING" if mask_missing else ""
        print(f"  [{i+1}/{len(scenes)}] {ds.bare(scene)} "
              f"tasks={st['total']:5d} kept={st['kept']:5d} "
              f"(routes {st['routes_kept']}/{st['routes_total']}) "
              f"startBad={st['start_bad']} goalBad={st['goal_bad']} "
              f"unreach={st['unreachable']}{flag}", flush=True)

    dt = time.time() - t0
    print(f"\n[filter] === {split} summary ({dt:.0f}s) ===", flush=True)
    print(f"  scenes total/with-kept/mask-missing: "
          f"{agg['scenes']}/{agg['scenes_with_kept']}/{agg['scenes_mask_missing']}",
          flush=True)
    print(f"  episodes total={agg['total']}  kept={agg['kept']} "
          f"({100.0*agg['kept']/max(1,agg['total']):.1f}%)", flush=True)
    print(f"  unique routes total={agg['routes_total']}  kept={agg['routes_kept']} "
          f"({100.0*agg['routes_kept']/max(1,agg['routes_total']):.1f}%)", flush=True)
    print(f"  reject: startBad={agg['start_bad']} goalBad={agg['goal_bad']} "
          f"unreachable={agg['unreachable']}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    summ = os.path.join(out_dir, f"filter_{split}_r{radius_m}.json")
    with open(summ, "w", encoding="utf-8") as f:
        json.dump(dict(split=split, radius_m=radius_m, collision=collision,
                       mask_root=mask_root, agg=agg, per_scene=per_scene),
                  f, ensure_ascii=False, indent=2)
    recs = os.path.join(out_dir, f"kept_{split}_r{radius_m}.jsonl")
    with open(recs, "w", encoding="utf-8") as f:
        for r in kept_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[filter] wrote {summ}\n[filter] wrote {recs}", flush=True)
    return agg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="instr",
                    choices=["train", "instr", "scene", "traj"])
    ap.add_argument("--radius", type=float, default=0.07)
    ap.add_argument("--collision", default="usd_rooms")
    ap.add_argument("--mask-root",
                    default=os.path.join(SAGE_ROOT, "sage_usd_rooms_masks"))
    ap.add_argument("--out-dir",
                    default="/data/shared/groups/mfm/vl/rl/"
                            "anonymous/SAGE-3D/episode_filter")
    ap.add_argument("--limit-scenes", type=int, default=0)
    a = ap.parse_args(argv)
    run(a.split, a.radius, a.mask_root, a.collision, a.out_dir, a.limit_scenes)


if __name__ == "__main__":
    main()
