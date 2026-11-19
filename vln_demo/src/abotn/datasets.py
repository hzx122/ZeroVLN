"""The three ABotN benchmarks, as separate asset sets.

    ABotN-POIBench            11 scenes  163 tasks   中文 POI 指令 + 真值轨迹
    ABotN-PointBench/Indoor   16 scenes  240 tasks   point-goal, 室内
    ABotN-PointBench/Outdoor  15 scenes  225 tasks   point-goal, 室外

They stay three sets rather than one "ABotN" bucket: a POI task is a language
instruction scored against a named shop, a point-goal task is a coordinate with
no instruction at all, and indoor/outdoor differ by an order of magnitude in
scale.  Mixing them into one dropdown would hide exactly the distinction you
pick a scene for.

Scene ids are prefixed per set (``poi-``, ``in-``, ``out-``) because four scene
directories appear in *both* POIBench and PointBench/Outdoor -- the console
keys its scene->asset-set map by scene id, so a bare id would make those four
ambiguous, and the render server would serve one scene under a name that means
two different sets of tasks.

The layout each set expects, which is also what the render server wants::

    <bench>/occmaps/<scene>/gs/floor_0_block_0/*.ply   Gaussians
    <bench>/occmaps/<scene>/map/occ_map*.{png,tiff,txt}
    <bench>/annotations/<scene>/traj_*.json            tasks
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

ZYTOSS = os.environ.get("ABOTN_ROOT", "/workspace/agentic_sim/zytoss")


@dataclass
class Task:
    """One ``traj_*.json``: a start pose, a goal point and a reference path."""

    task_id: str
    scene: str
    start: dict                 # x/y/z/roll/pitch/yaw, ABotN world (z up)
    goal_xy: tuple              # (x, y) -- label.extend.end_point
    instruction: str
    goal_label: str
    ref_len: float              # GT path length, the SPL denominator
    euclidean: float
    distance_bucket: str = ""
    waypoints: tuple = ()       # (x, y, z) of the annotated path


@dataclass
class ABotNSet:
    """One benchmark directory."""

    key: str                    # backend key, also the scene-id prefix
    label: str                  # 资产集 dropdown text
    desc: str                   # one-line hint under the dropdown
    bench_dir: str
    arrive_threshold: float     # metres, ABotN's own value for this benchmark
    spl_slack: float = 0.0      # subtracted from the shortest path (POI only)
    _scenes: Optional[List[str]] = field(default=None, repr=False)
    _tasks: Dict[str, List[Task]] = field(default_factory=dict, repr=False)
    _areas: Dict[str, float] = field(default_factory=dict, repr=False)

    # -- paths -------------------------------------------------------------
    @property
    def occ_root(self) -> str:
        return os.path.join(self.bench_dir, "occmaps")

    @property
    def ann_root(self) -> str:
        return os.path.join(self.bench_dir, "annotations")

    def scene_dir(self, scene: str) -> str:
        return os.path.join(self.occ_root, self.bare(scene))

    def bare(self, scene: str) -> str:
        """Strip the set prefix off a console scene id."""
        p = self.key + "-"
        return scene[len(p):] if scene.startswith(p) else scene

    def qualify(self, bare: str) -> str:
        return "%s-%s" % (self.key, bare)

    # -- scenes ------------------------------------------------------------
    def available(self) -> bool:
        return os.path.isdir(self.occ_root) and bool(self.list_scenes())

    def list_scenes(self) -> List[str]:
        """Prefixed ids of scenes that have both Gaussians and an occ map."""
        if self._scenes is None:
            out = []
            if os.path.isdir(self.occ_root):
                for d in sorted(os.listdir(self.occ_root)):
                    p = os.path.join(self.occ_root, d)
                    if (glob.glob(os.path.join(p, "gs", "*", "*.ply"))
                            and os.path.exists(os.path.join(p, "map", "occ_map.png"))
                            and os.path.isdir(os.path.join(self.ann_root, d))):
                        out.append(self.qualify(d))
            self._scenes = out
        return list(self._scenes)

    # -- tasks -------------------------------------------------------------
    def tasks(self, scene: str) -> List[Task]:
        if scene not in self._tasks:
            self._tasks[scene] = self._read_tasks(scene)
        return self._tasks[scene]

    def _read_tasks(self, scene: str) -> List[Task]:
        out = []
        pat = os.path.join(self.ann_root, self.bare(scene), "traj_*.json")
        for p in sorted(glob.glob(pat), key=_traj_order):
            try:
                with open(p, encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:
                continue
            traj = d.get("trajectory") or []
            ext = (d.get("label") or {}).get("extend") or {}
            end = ext.get("end_point")
            if not traj or not end or len(end) < 2:
                continue
            out.append(Task(
                task_id=os.path.basename(p)[:-5].replace("traj_", ""),
                scene=scene,
                start=_pose(traj[0]),
                goal_xy=(float(end[0]), float(end[1])),
                instruction=(d.get("instruction") or "").strip(),
                goal_label=(ext.get("goal_label") or "").strip(),
                # actual_len is rounded to 0.1 m in the annotations, so prefer
                # the polyline we can measure ourselves and keep it as a fallback
                ref_len=_path_len(traj) or float(ext.get("actual_len") or 0.0),
                euclidean=float(ext.get("euclidean_distance") or 0.0),
                distance_bucket=(ext.get("distance_bucket") or "").strip(),
                waypoints=tuple((float(p["x"]), float(p["y"]),
                                 float(p.get("z", 0.0))) for p in traj)))
        return out

    # -- labels ------------------------------------------------------------
    def area_of(self, scene: str) -> float:
        """Walkable area in square metres, straight off the occupancy map.

        The point-goal scenes ship no description at all, and their ids are
        opaque (``0802_840243``), so this is the one number available to tell a
        40 m2 studio from a 100 m2 flat in the dropdown.  Only the png is read,
        never the height tiff, and only the total is kept -- the maps run to
        40 megapixels and holding 31 of them for a label would cost a gigabyte.
        """
        if scene not in self._areas:
            import numpy as np
            from PIL import Image

            from .occmap import FREE, parse_meta

            d = os.path.join(self.scene_dir(scene), "map")
            try:
                with open(os.path.join(d, "occ_map_meta.txt"), encoding="utf-8") as f:
                    m = parse_meta(f.read())
                occ = np.asarray(Image.open(os.path.join(d, "occ_map.png"))
                                 .convert("L"))
                self._areas[scene] = (
                    float((occ >= FREE).sum())
                    * m["COORDINATE_RANGE_X"] / occ.shape[1]
                    * m["COORDINATE_RANGE_Y"] / occ.shape[0])
            except Exception:
                self._areas[scene] = 0.0
        return self._areas[scene]

    def title_of(self, scene: str) -> str:
        raise NotImplementedError


class POIBench(ABotNSet):
    """Shop/POI goals with Chinese instructions -- the only scorable ABotN set
    with language in it."""

    def title_of(self, scene: str) -> str:
        ts = self.tasks(scene)
        names = []
        for t in ts:
            n = t.goal_label or ""
            if n and n not in names:
                names.append(n)
        if not names:
            return "POI 场景"
        head = names[0][:12]
        return head if len(names) == 1 else "%s 等 %d 个 POI" % (head, len(names))


class PointBench(ABotNSet):
    """Coordinate goals, no instruction text."""

    def __init__(self, *a, indoor: bool = True, **kw):
        super().__init__(*a, **kw)
        self.indoor = indoor

    def title_of(self, scene: str) -> str:
        bare = self.bare(scene)
        kind = "室内户型"
        if not self.indoor:
            kind = "室外街区"
            for pre, name in (("park", "公园"), ("cross", "路口"),
                              ("street", "街道")):
                if bare.startswith(pre):
                    kind = name
                    break
        area = self.area_of(scene)
        return "%s %.0f㎡" % (kind, area) if area > 1 else kind


# ----------------------------------------------------------------------------
def _pose(p: dict) -> dict:
    return {k: float(p.get(k, 0.0)) for k in ("x", "y", "z", "roll", "pitch", "yaw")}


def _path_len(traj) -> float:
    tot = 0.0
    for a, b in zip(traj, traj[1:]):
        tot += ((float(b["x"]) - float(a["x"])) ** 2
                + (float(b["y"]) - float(a["y"])) ** 2) ** 0.5
    return tot


def _traj_order(path: str):
    """Sort ``traj_2`` before ``traj_10``; ids stay the file's own number."""
    stem = os.path.basename(path)[:-5].replace("traj_", "")
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def all_sets(root: str = ZYTOSS) -> List[ABotNSet]:
    """Every benchmark descriptor, present on disk or not."""
    return [
        POIBench(key="poi", label="POI 导航 / ABotN-POIBench",
                 desc="真实街区高斯场景，中文 POI 指令（前往「111医药馆」），带真值轨迹，可算指标",
                 bench_dir=os.path.join(root, "ABotN-POIBench"),
                 # ABotN's own poi_goal default: shopfronts are annotated at the
                 # storefront, so 2 m is "arrived", and SPL gets the same slack
                 arrive_threshold=2.0, spl_slack=2.0),
        PointBench(key="in", label="室内点导航 / ABotN-PointBench",
                   desc="室内高斯场景，目标是一个坐标点而不是一句话，到达阈值 0.5 m",
                   bench_dir=os.path.join(root, "ABotN-PointBench", "Indoor"),
                   arrive_threshold=0.5, indoor=True),
        PointBench(key="out", label="室外点导航 / ABotN-PointBench",
                   desc="室外街区/公园高斯场景，目标是一个坐标点，地面有起伏，到达阈值 0.5 m",
                   bench_dir=os.path.join(root, "ABotN-PointBench", "Outdoor"),
                   arrive_threshold=0.5, indoor=False),
    ]


def available(root: str = ZYTOSS) -> Dict[str, ABotNSet]:
    """Only the benchmarks actually downloaded on this machine."""
    return {s.key: s for s in all_sets(root) if s.available()}
