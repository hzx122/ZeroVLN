"""GVLN trajectories as a split-selectable asset set.

SAGE-3D ships one directory of Gaussian scenes (``InteriorGS/<prefix>_<id>/``,
carrying ``occupancy.{json,png}`` and ``3dgs_compressed.ply``) and, separately,
GVLN reference trajectories split four ways::

    SAGE-3D_VLN_Data/train/Trajectory_train/<id>/train_trajectories_overall_*.json
    SAGE-3D_VLN_Data/test/Trajectory_test/Trajectory_test_Instruction_Unseen/<id>/*_simplify.json
    ...Scene_Unseen / ...Trajectory_Unseen

so this is one backend (``sage``) with a ``--sage_split`` selector rather than
four asset sets: the scenes and the kinematics are identical across splits, only
which trajectories you score against changes.

Each GVLN file is ``scenes[0].samples[]``; every sample is one reference path
(``points[]``, dense z-up poses at camera height z=0.5) plus a list of
``instructions`` phrased against the same path. One sample with three
instructions is three console tasks -- same route, different wording -- because
that is exactly the Instruction-Unseen axis the benchmark tests.

Scene ids are the short numeric suffix (``840898``) with a ``sage-`` console
prefix, matching the render-server ``scenes_root`` :mod:`sage.make_scenes_root`
builds; the on-disk InteriorGS directory keeps its ``<prefix>_`` and is resolved
once by scanning.
"""

from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

SAGE_ROOT = os.environ.get(
    "SAGE_ROOT", "/data/zerovln/SAGE-3D")

# console split key -> path under SAGE-3D_VLN_Data
SPLITS = {
    "official_sage_bench_v1": os.environ.get("SAGE_BENCH_PROTOCOL_ROOT", "/data/zerovln/artifacts/sage-official-bench-20260922-r1") + "/dataset/Trajectory_demo",
    "train": "train/Trajectory_train",
    "instr": "test/Trajectory_test/Trajectory_test_Instruction_Unseen",
    "scene": "test/Trajectory_test/Trajectory_test_Scene_Unseen",
    "traj":  "test/Trajectory_test/Trajectory_test_Trajectory_Unseen",
    "demo":  "demo/Trajectory_demo",
    # Smoke-scale action expert SFT source. It lives outside the canonical demo tree.
    "action_head_simple_construction": "/data/zerovln/"
                                         "action_head/simple_cases/construction/Trajectory_demo",
    "action_head_curriculum_p0": "/data/zerovln/action_head/curriculum/p0/Trajectory_demo",
    "action_head_curriculum_p1": "/data/zerovln/action_head/curriculum/p1/Trajectory_demo",
    "action_head_curriculum_p2": "/data/zerovln/action_head/curriculum/p2/Trajectory_demo",
    "action_head_curriculum": "/data/zerovln/action_head/curriculum/testset/Trajectory_demo",
    # Candidate training data, kept scene-disjoint from the fixed test set.
    "action_head_train_p0_pilot": "/data/zerovln/action_head/curriculum/pilot_v1/p0/Trajectory_demo",
    "action_head_train_p0_expand_b1": "/data/zerovln/action_head/curriculum/train_v1/p0_expand/Trajectory_demo",
    "action_head_train_p0_expand_b2": "/data/zerovln/action_head/curriculum/train_v1/p0_expand_b2/Trajectory_demo",
    "action_head_train_p0_unified": "/data/zerovln/action_head/curriculum/train_v1/p0_unified/Trajectory_demo",
    # Official train trajectories admitted only after the two-stage physical QC
    # in `sage.clean_official_gvln`.  This is deliberately a separate asset
    # root: it must never silently fall back to the unfiltered official split.
    "official_gvln_physical_clean": "/data/zerovln/SAGE-3D/official_gvln_physical_clean_v2/Trajectory_demo",
    "official_gvln_coordinate_rescued": "/data/zerovln/SAGE-3D/official_gvln_coordinate_rescued_v1/Trajectory_demo",
    "official_gvln_train_candidate": "/data/zerovln/SAGE-3D/official_gvln_train_candidate_v1/Trajectory_demo",
    # action expert P0 training data: official GVLN start/goal + goal object,
    # path re-planned with ECM centered A* + collision-aware smoothing by
    # `sage.build_official_action_head_demo`.  Collision contract = usd_rooms +
    # furniture=none (v4 USD mask's object_obstacle blocks furniture).  Terminal
    # VLM visibility / bearing gate skipped (downstream SFT chain replays a
    # window at every slow anchor, not endpoint).  See build_provenance.json.
    "official_gvln_action_head_p0": "/data/zerovln/SAGE-3D/official_gvln_action_head_p0_v1/Trajectory_demo",
    # Pure-random 50-trajectory subset of official_gvln_action_head_p0's 265
    # train scenes (seed=20260907, same-scene multi-trajectory allowed),
    # materialized by sage.sample_official_gvln_subset.  Same physical contract
    # as the parent split (usd_rooms / furniture=none / r0.07 / cam1.2 / HFOV115).
    "official_gvln_action_head_p0_sample50": "/data/zerovln/SAGE-3D/official_gvln_action_head_p0_sample50_20260907",
    # v3-protocol smoke episode pool: first 8 train scenes of the candidate
    # train split, ECM re-planned by sage.build_official_action_head_demo for
    # local_smoke v3 SFT data construction (quality check before full DLC).
    "official_gvln_action_head_p0_v3smoke": "/data/zerovln/SAGE-3D/official_gvln_action_head_p0_v3smoke_20260907/Trajectory_demo",
    # Full-827 train-scene episode pool for the v3-protocol official-gvln SFT
    # data construction (ECM re-planned by sage.build_official_action_head_demo,
    # 810 scenes / ~54.8k trajectories, persisted once and reused by all DLC
    # construction runs; 95 val scenes not included).
    "official_gvln_action_head_p0_v3full827train": "/data/zerovln/SAGE-3D/official_gvln_action_head_p0_v3full_827train_20260907/Trajectory_demo",
    # Fixed random 100-trajectory test subset drawn only from the 95 held-out
    # official-gvln validation scenes (seed=20260908).  Every trajectory has
    # passed the same ECM replanning and physical gates as the v3 train pool.
    "official_gvln_action_head_p0_v3test100": "/data/zerovln/SAGE-3D/official_gvln_action_head_p0_v3test100_20260908",
    "official_gvln_action_head_p0_v3full95val": "/data/zerovln/artifacts/sage-fullval-clean-20260919/dataset/Trajectory_demo",
    # Unified v3 capability page. All five families are materialized into this
    # one GVLN directory by build_capability_cases --console-output.
    "capability": os.environ.get(
        "SAGE_CAPABILITY_VLN_ROOT",
        "/workspace/ZeroVLN/"
        "tmp/capability_v3_unified/Trajectory_demo"),
    # Non-canonical diagnostic sources. They live outside SAGE_ROOT so profile
    # generation can never overwrite the canonical demo split.
    "cap_perception": "/data/zerovln/"
                      "capability_benchmark/sources/perception/Trajectory_demo",
    "cap_execution": "/data/zerovln/"
                     "capability_benchmark/sources/execution/Trajectory_demo",
    "cap_grounding": "/data/zerovln/"
                     "capability_benchmark/sources/grounding/Trajectory_demo",
    "cap_construct_perception": "/data/zerovln/"
                                "capability_benchmark/sources/construction/perception/Trajectory_demo",
    "cap_construct_execution": "/data/zerovln/"
                               "capability_benchmark/sources/construction/execution/Trajectory_demo",
    "cap_construct_grounding": "/data/zerovln/"
                               "capability_benchmark/sources/construction/grounding/Trajectory_demo",
    "cap_eval_perception": "/data/zerovln/"
                           "capability_benchmark/sources/evaluation/perception/Trajectory_demo",
    "cap_eval_execution": "/data/zerovln/"
                          "capability_benchmark/sources/evaluation/execution/Trajectory_demo",
    "cap_eval_grounding": "/data/zerovln/"
                          "capability_benchmark/sources/evaluation/grounding/Trajectory_demo",
    "cap_construct_long_route": "/data/zerovln/"
                                "capability_benchmark/sources/construction/long_route/Trajectory_demo",
    "cap_eval_long_route": "/data/zerovln/"
                           "capability_benchmark/sources/evaluation/long_route/Trajectory_demo",
}
SPLIT_LABEL = {"train": "训练集", "instr": "Instruction-Unseen",
               "scene": "Scene-Unseen", "traj": "Trajectory-Unseen",
               "demo": "Demo(自造用例)",
               "action_head_simple_construction": "Action Head SFT Smoke / Construction",
               "action_head_curriculum_p0": "Action Head Curriculum / P0 Control Calibration",
               "action_head_curriculum_p1": "Action Head Curriculum / P1 Visible-Goal Approach",
               "action_head_curriculum_p2": "Action Head Curriculum / P2 Short Option",
               "action_head_curriculum": "Action Head Curriculum / P0-P2 Test Set",
               "action_head_train_p0_pilot": "Action Head Train Candidate / P0 Pilot",
               "action_head_train_p0_expand_b1": "Action Head Train Candidate / P0 Expand B1",
               "action_head_train_p0_expand_b2": "Action Head Train Candidate / P0 Expand B2",
               "action_head_train_p0_unified": "Action Head Train / P0 Unified (1537)",
               "official_gvln_physical_clean": "Official GVLN / Physical QC Clean",
               "official_gvln_coordinate_rescued": "Official GVLN / Coordinate-Corrected Rescued",
               "official_gvln_train_candidate": "Official GVLN / Physical Train Candidate (Text Unreviewed)",
               "official_gvln_action_head_p0_v3test100": "Official GVLN / v3 Held-out Test 100",
               "capability": "Capability v3 / Unified Five Families"}
SPLIT_LABEL.update({"cap_perception": "Capability/Perception",
                    "cap_execution": "Capability/Execution",
                    "cap_grounding": "Capability/Grounding",
                    "cap_construct_perception": "Capability Construction/Perception",
                    "cap_construct_execution": "Capability Construction/Execution",
                    "cap_construct_grounding": "Capability Construction/Grounding",
                    "cap_eval_perception": "Capability Eval/Perception",
                    "cap_eval_execution": "Capability Eval/Execution",
                    "cap_eval_grounding": "Capability Eval/Grounding",
                    "cap_construct_long_route": "Capability Construction/Long Route",
                    "cap_eval_long_route": "Capability Eval/Long Route"})

# Some diagnostic splits expose exactly one selected instruction per physical route.
CONSOLE_INSTRUCTION_INDEX = {"action_head_simple_construction": 1}

PREFIX = "sage"                 # console scene-id prefix and backend key
ARRIVE_THRESHOLD = 1.0         # metres; GVLN goal is a point, 1 m is "arrived"


def yaw_from_wxyz(rot) -> float:
    """Heading (rad, from +x towards +y) from an Isaac/USD ``[w,x,y,z]`` quat.

    GVLN stores ``point['rotation']`` scalar-**first**, the order Isaac's
    ``set_world_pose`` consumes, and every stored pose is a pure yaw about world
    z (x/y components are 0). Verified against the reference paths: over 200+
    samples in two scenes the start heading this returns lands within 9 deg of
    the initial path direction, so it is the navigation heading, not a mount
    artefact. (SAGE's own ``yaw_from_quaternion`` assumes scalar-last and reads
    every stored quat as yaw 0 -- a bug we deliberately do not copy.)
    """
    w, x, y, z = float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def trajectory_group_id(task_id: str) -> str:
    """Task id -> base route id shared by dimension-specific v3 cases."""
    trajectory = str(task_id).rsplit("_", 1)[0]
    return trajectory.split("__", 1)[0]


def _to_geometry_frame(x: float, y: float, yaw: float):
    """Map a GVLN stored ``(x, y, yaw)`` into the InteriorGS geometry frame
    (the frame the gaussian ply, occupancy grid, collision mesh and
    ``labels.json`` all share).

    The SAGE-3D ``*_simplify.json`` ships positions already transformed by
    ``trajectory_2d_to_3d.py`` with ``flip_x = flip_y = True, negate = False`` =
    a 180° rotation about the scene origin. Isaac eval reads them verbatim into
    a USDA scene whose frame is the rot180 of our gaussian-ply frame, so to land
    on our ply/occupancy/labels frame we apply the same rot180 to position and
    heading: ``(-x, -y, yaw + pi)``. Verified across instr / scene / traj / train
    (start -> named object 0.3-0.5 m, goal 0.4-1.1 m; the ~1 m residual is the
    agent's standing position beside the object, not error). The old ``+S``
    translation picker could not express a rotation and landed 5-10 m off.
    """
    y2 = yaw + math.pi
    if y2 > math.pi:
        y2 -= 2 * math.pi
    return -x, -y, y2


@dataclass
class Task:
    """One GVLN reference path under one instruction."""

    task_id: str
    scene: str                  # prefixed console id, e.g. "sage-840898"
    start: dict                 # x/y/z/roll/pitch/yaw, SAGE world (z up)
    goal_xy: tuple              # (x, y) -- points[-1].position[:2]
    instruction: str
    ref_len: float              # GT path length (xy polyline), the SPL denominator
    waypoints: tuple = ()       # (x, y, z) of the reference path
    yaws: tuple = ()            # per-point heading (rad, yaw from +x→+y)，SFT 渲染+dyaw 用
    instruction_type: str = ""
    start_label: str = ""
    goal_label: str = ""


@dataclass
class SageSet:
    """One GVLN split over the shared InteriorGS scenes."""

    split: str
    root: str = SAGE_ROOT
    key: str = PREFIX
    arrive_threshold: float = ARRIVE_THRESHOLD
    _scenes: Optional[List[str]] = field(default=None, repr=False)
    _tasks: Dict[tuple, List[Task]] = field(default_factory=dict, repr=False)
    _dirmap: Optional[Dict[str, str]] = field(default=None, repr=False)

    # -- paths -------------------------------------------------------------
    @property
    def label(self) -> str:
        return "SAGE-3D GVLN / %s" % SPLIT_LABEL.get(self.split, self.split)

    @property
    def vln_dir(self) -> str:
        configured = SPLITS[self.split]
        if os.path.isabs(configured):
            return configured
        return os.path.join(self.root, "SAGE-3D_VLN_Data", configured)

    @property
    def interior_root(self) -> str:
        return os.path.join(self.root, "InteriorGS")

    def bare(self, scene: str) -> str:
        p = self.key + "-"
        return scene[len(p):] if scene.startswith(p) else scene

    def qualify(self, bare: str) -> str:
        return "%s-%s" % (self.key, bare)

    def _interior_map(self) -> Dict[str, str]:
        """suffix id (``840898``) -> InteriorGS dir name (``0474_840898``)."""
        if self._dirmap is None:
            m = {}
            if os.path.isdir(self.interior_root):
                for d in os.listdir(self.interior_root):
                    suf = d.split("_")[-1]
                    m.setdefault(suf, d)
            self._dirmap = m
        return self._dirmap

    def scene_dir(self, scene: str) -> str:
        """InteriorGS directory (occupancy + compressed ply) for a scene."""
        return os.path.join(self.interior_root, self._interior_map().get(
            self.bare(scene), self.bare(scene)))

    def traj_file(self, scene: str) -> Optional[str]:
        hits = glob.glob(os.path.join(self.vln_dir, self.bare(scene), "*.json"))
        hits = [h for h in hits if "summary" not in os.path.basename(h).lower()]
        return sorted(hits)[0] if hits else None

    # -- scenes ------------------------------------------------------------
    def available(self) -> bool:
        return os.path.isdir(self.vln_dir) and bool(self.list_scenes())

    def list_scenes(self) -> List[str]:
        """Prefixed ids of scenes with a trajectory file and an occupancy map."""
        if self._scenes is None:
            out = []
            if os.path.isdir(self.vln_dir):
                for d in sorted(os.listdir(self.vln_dir)):
                    if not os.path.isdir(os.path.join(self.vln_dir, d)):
                        continue
                    sd = self.scene_dir(d)
                    if (self.traj_file(d)
                            and os.path.exists(os.path.join(sd, "occupancy.json"))
                            and os.path.exists(os.path.join(sd, "occupancy.png"))):
                        out.append(self.qualify(d))
            self._scenes = out
        return list(self._scenes)

    # -- tasks -------------------------------------------------------------
    def tasks(self, scene: str, limit: int = 0) -> List[Task]:
        """This scene's tasks. ``limit>0`` stops parsing after N tasks——caps
        SageEnv startup on big splits (train ≈ 2.2M episodes across 929 scenes:
        full eager parse ~6 min + ~2 GB/bench; with limit=10 → ~5 s + ~10 MB)."""
        key = (scene, int(limit))
        if key not in self._tasks:
            self._tasks[key] = self._read_tasks(scene, limit)
        return self._tasks[key]

    def traj_tasks(self, scene: str, traj_id) -> List[Task]:
        """一条参考轨迹（trajectory_id）下的全部指令用例。

        task_id 形如 ``"<trajectory_id>_<instruction_idx>"``；用 rsplit("_", 1)
        恢复前缀，即使 trajectory_id 本身含下划线也对。全量解析（limit=0）有
        独立缓存键，一次 json + 几何变换约百毫秒，之后命中缓存。
        """
        traj = str(traj_id).split("__", 1)[0]
        return [t for t in self.tasks(scene, 0)
                if trajectory_group_id(t.task_id) == traj]

    def _read_tasks(self, scene: str, limit: int = 0) -> List[Task]:
        path = self.traj_file(scene)
        if not path:
            return []
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            return []
        out = []
        for sc in d.get("scenes", []):
            for smp in sc.get("samples", []):
                pts = smp.get("points") or []
                if len(pts) < 2:
                    continue
                p0 = pts[0]
                pos0 = p0["position"]
                sx, sy, syaw = _to_geometry_frame(
                    float(pos0[0]), float(pos0[1]), yaw_from_wxyz(p0["rotation"]))
                start = {"x": sx, "y": sy, "z": float(pos0[2]),
                         "roll": 0.0, "pitch": 0.0, "yaw": syaw}
                goal = pts[-1]["position"]
                gx, gy, _ = _to_geometry_frame(float(goal[0]), float(goal[1]), 0.0)
                waypoints = tuple(
                    (*_to_geometry_frame(float(p["position"][0]),
                                         float(p["position"][1]), 0.0)[:2],
                     float(p["position"][2])) for p in pts)
                yaws = tuple(
                    _to_geometry_frame(0.0, 0.0, yaw_from_wxyz(p["rotation"]))[2]
                    for p in pts)
                ref_len = _path_len(waypoints)
                tid = str(smp.get("trajectory_id", len(out)))
                instrs = smp.get("instructions") or [{}]
                for i, ins in enumerate(instrs):
                    selected_index = CONSOLE_INSTRUCTION_INDEX.get(self.split)
                    if selected_index is not None and i != selected_index:
                        continue
                    out.append(Task(
                        task_id="%s_%d" % (tid, i),
                        scene=scene,
                        start=dict(start),
                        goal_xy=(gx, gy),
                        instruction=(ins.get("generated_instruction") or "").strip(),
                        ref_len=ref_len,
                        waypoints=waypoints,
                        yaws=yaws,
                        instruction_type=(ins.get("instruction_type") or "").strip(),
                        start_label=(ins.get("start") or "").strip(),
                        goal_label=(ins.get("end") or "").strip()))
                    if limit and len(out) >= limit:
                        return out
        return out

    # -- labels ------------------------------------------------------------
    def title_of(self, scene: str) -> str:
        ts = self.tasks(scene)
        return "%s · %d 回合" % (self.bare(scene), len(ts))


# ----------------------------------------------------------------------------
def _path_len(waypoints) -> float:
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1])
                     for a, b in zip(waypoints, waypoints[1:])))


def all_sets(root: str = SAGE_ROOT) -> Dict[str, SageSet]:
    return {s: SageSet(split=s, root=root) for s in SPLITS}


def available(root: str = SAGE_ROOT) -> Dict[str, SageSet]:
    """Splits whose trajectory directory exists and has usable scenes."""
    return {s: ds for s, ds in all_sets(root).items() if ds.available()}
