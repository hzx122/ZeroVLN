"""Build trajectory-grounded demo episodes for SAGE-3D GVLN (usd_rooms branch).

A self-contained pipeline that constructs NEW episodes (not filtering official
ones) on the sage ``usd_rooms`` render branch, with instructions grounded in the
A* path the robot actually travels -- the opposite of the official pipeline,
which feeds the LLM only a start/end id and an LLM-fabricated scene text.

Stages (each writes an intermediate artifact so it can be rerun alone):

  A  scene index    OccMap(usd_rooms) + labels.json -> label_N -> {bbox, centre, room}
  B  pose bank      Poisson-disk starts from ``candidate``; each goal is a free,
                    visible stance around a target object's bbox, not an
                    arbitrary free cell later snapped to the nearest object
  C  A* golden       centred A* + collision-aware curve smoothing + exact replay
  D  observation    every concrete class uses the same blind RGB localization,
                    3-D/depth agreement and nearby-view confirmation gates
  E  instruction    deterministic fact specs + controlled templates produce
                    Object-Goal / Area-based x {high,low}; strict validation;
                    assemble GVLN json in the stored rot180 frame SageSet reads.

Run with the canonical config (no overrides)::

    bash scripts/demo/run_demo_episodes.sh

Experiment wrappers may pass only changed dotted keys with ``--set``. The
default production contract lives in ``vln_demo/config/sage_demo_pipeline.yaml``.

The output lands under $SAGE_ROOT/SAGE-3D_VLN_Data/demo/Trajectory_demo/<bare>/
and is visible in the console via ``SAGE_SPLIT=demo`` (see run_demo_episodes.sh).
"""

from __future__ import annotations

import json
import math
import os
import base64
import io
import re
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .demo_pipeline_config import pipeline_config_from_argv

# --------------------------------------------------------------------------- #
# strict single-source configuration
# --------------------------------------------------------------------------- #
PIPELINE_CONFIG, PIPELINE_CONFIG_META = pipeline_config_from_argv()
_RUNTIME = PIPELINE_CONFIG["runtime"]
_CAMERA = PIPELINE_CONFIG["camera"]
_ROBOT = PIPELINE_CONFIG["robot"]
_PLANNER = PIPELINE_CONFIG["planner"]
_SAMPLING = PIPELINE_CONFIG["sampling"]
_GOAL = PIPELINE_CONFIG["goal"]
_VISIBILITY = PIPELINE_CONFIG["visibility"]
_CAPABILITY = PIPELINE_CONFIG["capability_profile"]

SAGE_ROOT = str(_RUNTIME["sage_root"])
DEMO_ROOT = str(_RUNTIME["demo_root"])
OUT_VLN = str(_RUNTIME["out_vln"])
SCENES = [str(x) for x in _RUNTIME["scenes"]]
EPISODES_PER_SCENE = int(_RUNTIME["episodes_per_scene"])
REQUIRE_ALL_SCENES = bool(_RUNTIME["require_all_scenes"])
TRACE_REJECTIONS = bool(_RUNTIME["trace_rejections"])
PIPELINE_NAME = str(PIPELINE_CONFIG["pipeline"]["name"])
PIPELINE_VERSION = str(PIPELINE_CONFIG["pipeline"]["version"])
RENDER_URL = str(_RUNTIME["render_url"])
CAMERA_WIDTH = int(_CAMERA["width"])
CAMERA_HEIGHT_PX = int(_CAMERA["height"])
CAMERA_HFOV_DEG = float(_CAMERA["hfov_deg"])
CAMERA_Z_M = float(_CAMERA["height_m"])
ROBOT_RADIUS_M = float(_ROBOT["physical_radius_m"])
PLANNING_MARGIN_M = float(_ROBOT["planning_margin_m"])
PLANNING_RADIUS_M = ROBOT_RADIUS_M + PLANNING_MARGIN_M
REPLAY_STEP_M = float(_ROBOT["replay_step_m"])
PLANNER_MODE = str(_PLANNER["mode"])
MIN_ACCEPTED_SMOOTHING_BLEND = float(
    _PLANNER["min_accepted_smoothing_blend"])
PAIR_EUCLIDEAN_M = tuple(map(float, _SAMPLING["pair_euclidean_m"]))
PAIR_GEODESIC_M = tuple(map(float, _SAMPLING["pair_geodesic_m"]))
START_MIN_SPACING_M = float(_SAMPLING["start_min_spacing_m"])
GOAL_STANDOFF_M = tuple(map(float, _SAMPLING["goal_standoff_m"]))
GOAL_MAX_EPISODES_PER_OBJECT = int(_SAMPLING["max_episodes_per_object"])
GOAL_MAX_STANCES_PER_OBJECT = int(_SAMPLING["max_stances_per_object"])
GOAL_TERMINAL_MODE = str(_GOAL["terminal_mode"])
GOAL_APPROACH_M = float(_GOAL["approach_m"])
GOAL_ROOM_MARGIN_M = float(_GOAL["room_margin_m"])
GOAL_ROOM_TAIL_M = float(_GOAL["room_tail_m"])
GOAL_MAX_NATURAL_BEARING_ERROR_DEG = float(_GOAL["max_bearing_error_deg"])
GOAL_MIN_BBOX_WIDTH_PX = int(_VISIBILITY["min_bbox_width_px"])
GOAL_MIN_BBOX_HEIGHT_PX = int(_VISIBILITY["min_bbox_height_px"])
GOAL_MIN_PROJECTED_AREA_RATIO = float(_VISIBILITY["min_projected_area_ratio"])
GOAL_MIN_DEPTH_AREA_RATIO = float(_VISIBILITY["min_depth_area_ratio"])
GOAL_DEPTH_TOLERANCE_M = float(_VISIBILITY["depth_tolerance_m"])
GOAL_DEPTH_SAMPLE_STRIDE = int(_VISIBILITY["depth_sample_stride"])
GOAL_CONFIRMATION_OFFSET_M = float(_VISIBILITY["confirmation_offset_m"])
GOAL_MIN_DETECTION_WIDTH_PX = int(_VISIBILITY["min_detection_width_px"])
GOAL_MIN_DETECTION_HEIGHT_PX = int(_VISIBILITY["min_detection_height_px"])
GOAL_MIN_DETECTION_AREA_RATIO = float(_VISIBILITY["min_detection_area_ratio"])
GOAL_MIN_DETECTION_DEPTH_FRACTION = float(
    _VISIBILITY["min_detection_depth_fraction"])
GOAL_VLM_URL = str(_VISIBILITY["vlm_url"])
GOAL_VLM_MODEL = str(_VISIBILITY["vlm_model"])
CAPABILITY_PROFILE = str(_CAPABILITY["name"])
REQUIRE_CROSS_ROOM = bool(_CAPABILITY["require_cross_room"])
START_TARGET_VISIBILITY_MAX_ARC_M = float(
    _CAPABILITY["start_target_visibility_max_arc_m"])
MIN_ROUTE_TURNS = int(_CAPABILITY["min_turns"])
MAX_ROUTE_TURNS = int(_CAPABILITY["max_turns"])
MIN_ROUTE_DOORS = int(_CAPABILITY["min_doors"])
MAX_ROUTE_DOORS = int(_CAPABILITY["max_doors"])
MIN_ROUTE_DECISIONS = int(_CAPABILITY["min_decisions"])
MIN_ROUTE_LANDMARKS = int(_CAPABILITY["min_landmarks"])
MIN_ROUTE_CLEARANCE_M = float(_CAPABILITY["min_clearance_m"])
MAX_ROUTE_CLEARANCE_M = float(_CAPABILITY["max_clearance_m"])
MAX_VISIBLE_SAME_FAMILY_DISTRACTORS = int(
    _CAPABILITY["max_visible_same_family_distractors"])
ALLOWED_TARGET_FAMILIES = set(str(x) for x in
                              _CAPABILITY["allowed_target_families"])
GROUNDING_PAIR_CONTRACT = {
    "max_length_gap_m": float(_CAPABILITY["grounding_max_length_gap_m"]),
    "max_length_ratio": float(_CAPABILITY["grounding_max_length_ratio"]),
    "max_turn_gap": int(_CAPABILITY["grounding_max_turn_gap"]),
    "max_door_gap": int(_CAPABILITY["grounding_max_door_gap"]),
    "min_goal_separation_m": float(_CAPABILITY["grounding_min_goal_separation_m"]),
}
GROUNDING_MAX_CANDIDATES_PER_START = int(
    _CAPABILITY["grounding_max_candidates_per_start"])
SEED = int(_RUNTIME["seed"])
TRAJ_Z = 0.5          # stored agent body z (render uses ground+1.2; see env.py)
INSTRUCTION_TYPES = ("Object-Goal", "Area-based")
EXPECTED_INSTRUCTION_PAIRS = {
    (instruction_type, level)
    for instruction_type in INSTRUCTION_TYPES
    for level in ("high", "low")
}
CURRICULUM_OPTION_PROFILES = {"action_head_p0", "action_head_p1", "action_head_p2"}

# usd_rooms collision mode env MUST be set before importing occmap, so the
# module-level USD_COLLISION_ROOT reads the rooms artifact (it has rooms_free /
# room_region / door_opening the route extractor needs).
os.environ.setdefault("SAGE_COLLISION", "usd_rooms")
os.environ.setdefault("SAGE_USD_COLLISION_MASK_ROOT",
                      os.path.join(SAGE_ROOT, "sage_usd_rooms_masks"))

from .occmap import OccMap                       # noqa: E402
from .planner import (CenteredAStarConfig, PathSmoothingConfig, plan_path,
                      passable_grid, smooth_path)  # noqa: E402
from . import route_extractor                    # noqa: E402
from .gs_client import XYZEulerClient             # noqa: E402
from .visibility_audit import (depth_support, project_bbox,
                               ray_box_interval,
                               visible_to_bbox_2d)  # noqa: E402
from .goal_region import (goal_stances, load_label_index,
                          semantic_family)  # noqa: E402
from .capability_benchmark import select_grounding_pair  # noqa: E402
from rs3d.gs_client import CameraSpec             # noqa: E402

rng = np.random.default_rng(SEED)
CENTERED_CONFIG = CenteredAStarConfig(**_PLANNER["ecm"])
_smoothing_kwargs = dict(_PLANNER["smoothing"])
_smoothing_kwargs["blend_strengths"] = tuple(
    map(float, _smoothing_kwargs["blend_strengths"]))
SMOOTHING_CONFIG = PathSmoothingConfig(**_smoothing_kwargs)
VISIBILITY_CAMERA = CameraSpec(CAMERA_WIDTH, CAMERA_HEIGHT_PX, CAMERA_HFOV_DEG)
CONFIG_IS_CANONICAL = bool(PIPELINE_CONFIG["pipeline"]["canonical"]
                           and PIPELINE_CONFIG_META["uses_default_config"]
                           and not PIPELINE_CONFIG_META["overrides"])


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _interior_dir(bare: str) -> str:
    """``InteriorGS/<prefix>_<bare>`` for a scene id."""
    root = os.path.join(SAGE_ROOT, "InteriorGS")
    for d in os.listdir(root):
        if d.split("_")[-1] == bare:
            return os.path.join(root, d)
    raise FileNotFoundError("no InteriorGS dir for %s under %s" % (bare, root))


def _load_occ(bare: str) -> OccMap:
    return OccMap(_interior_dir(bare), collision_mode="usd_rooms", furniture_mode="none")


def _build_label_index(bare: str) -> Dict[str, dict]:
    """label_N -> {label, bbox_xyxy, centre_xy, ins_id} matching GVLN strings.

    Mirrors :func:`SAGE-3D_Official.semantic_map_builder.normalize_label` +
    per-label running counter in labels.json list order, so the produced ids
    (``bed_2``, ``fridge_1``) align with official GVLN ``start``/``end``.
    """
    return load_label_index(_interior_dir(bare))


def _snap_object(xy: Tuple[float, float], label_index: Dict[str, dict]) -> str:
    """Nearest object id by bbox-centre distance to ``xy`` (geometry frame)."""
    best, best_d = "", float("inf")
    for label_N, info in label_index.items():
        cx, cy = info["centre_xy"]
        d = (cx - xy[0]) ** 2 + (cy - xy[1]) ** 2
        if d < best_d:
            best, best_d = label_N, d
    return best


def _poisson_disk_starts(occ: OccMap, n: int,
                         min_spacing_m: float) -> List[Tuple[float, float]]:
    """``n`` candidate-pixel starts, each >= ``min_spacing_m`` apart (world m)."""
    cand = getattr(occ, "_usd_candidate", None)
    free = occ.free if cand is None else cand
    ys, xs = np.nonzero(free)
    if len(xs) == 0:
        return []
    order = rng.permutation(len(xs))
    out: List[Tuple[float, float]] = []
    for i in order:
        col, row = int(xs[i]), int(ys[i])
        x, y = occ.to_world(col, row)
        if occ.clearance(x, y) <= PLANNING_RADIUS_M:
            continue
        if all(math.hypot(x - ox, y - oy) >= min_spacing_m for ox, oy in out):
            out.append((x, y))
        if len(out) >= n:
            break
    return out


def _connected_components(passable: np.ndarray) -> np.ndarray:
    from scipy.ndimage import label
    lab, _ = label(passable, structure=np.ones((3, 3), dtype=int))
    return lab


# These are not navigable object instances at all.  This is intentionally a
# small ontology-level exclusion, not a whitelist of classes that happened to
# work in one batch. Every concrete label is admitted to the same downstream
# geometry, localization, depth and cross-view checks.
_NON_OBJECT_LABELS = {
    "ceiling", "floor", "wall", "baseboard", "door", "interior_doors",
    "window", "other", "unknown",
}


def _goal_stances(info: dict) -> List[Tuple[float, float]]:
    """Candidate poses on all four *outside* faces of an object's XY bbox.

    The occupancy/LOS checks below remove wall-backed faces.  Sampling several
    lateral positions prevents a cabinet corner or adjacent chair from making
    an otherwise useful object unreachable.
    """
    return goal_stances(info, GOAL_STANDOFF_M)


def _visible_to_bbox(occ: OccMap, stance: Tuple[float, float], info: dict) -> bool:
    """2-D line of sight from stance to just outside the target bbox.

    Testing the centre would intentionally enter the target's own collider and
    fail.  Instead, stop 8 cm before the first bbox intersection: every tested
    point is navigable space, while the ray still terminates at the object.
    """
    return visible_to_bbox_2d(occ, stance, info, ROBOT_RADIUS_M)


def _room_id(room_lab: np.ndarray, occ: OccMap,
             xy: Tuple[float, float]) -> int:
    if room_lab is None:
        return 0
    col, row = occ.to_pixel(xy[0], xy[1])
    if not (0 <= row < room_lab.shape[0] and 0 <= col < room_lab.shape[1]):
        return 0
    return int(room_lab[row, col])


def _inside_room(room_lab: np.ndarray, occ: OccMap,
                 xy: Tuple[float, float], room_id: int,
                 margin_m: float = GOAL_ROOM_MARGIN_M) -> bool:
    """Pose and a small surrounding disk must belong to ``room_id``.

    This prevents a nominal goal from landing on a wall/door threshold or just
    outside the room even when the target bbox itself touches that boundary.
    """
    if room_lab is None or room_id <= 0:
        return False
    col, row = occ.to_pixel(xy[0], xy[1])
    rad = max(1, int(math.ceil(margin_m / occ.scale)))
    if row - rad < 0 or row + rad >= room_lab.shape[0] \
            or col - rad < 0 or col + rad >= room_lab.shape[1]:
        return False
    yy, xx = np.ogrid[-rad:rad + 1, -rad:rad + 1]
    disk = xx * xx + yy * yy <= rad * rad
    patch = room_lab[row - rad:row + rad + 1, col - rad:col + rad + 1]
    return bool(np.all(patch[disk] == room_id))


def _path_finishes_in_room(occ: OccMap, waypoints,
                           room_lab: np.ndarray, room_id: int,
                           tail_m: float = GOAL_ROOM_TAIL_M) -> bool:
    """The last ``tail_m`` of the planned route must be inside target room."""
    if room_lab is None or len(waypoints) < 2 or room_id <= 0:
        return False
    left = tail_m
    for i in range(len(waypoints) - 1, 0, -1):
        x1, y1 = waypoints[i][:2]
        x0, y0 = waypoints[i - 1][:2]
        seg = math.hypot(x1 - x0, y1 - y0)
        take = min(left, seg)
        n = max(1, int(math.ceil(take / 0.05)))
        for k in range(n + 1):
            back = take * k / float(n)
            t = back / max(seg, 1e-9)
            xy = (x1 + (x0 - x1) * t, y1 + (y0 - y1) * t)
            if _room_id(room_lab, occ, xy) != room_id:
                return False
        left -= take
        if left <= 1e-6:
            return True
    return False


def _goal_approach_point(occ: OccMap, goal_xy: Tuple[float, float], info: dict,
                         comp: np.ndarray, room_lab: np.ndarray,
                         room_id: int) -> Optional[Tuple[float, float]]:
    """A safe point behind the stance, producing a natural target-facing tail.

    The robot should see the target while completing its final forward motion,
    not arrive side-on and rotate in place.  Extend from the target centre
    through the selected stance into free space, then require the final 0.6 m
    segment to be collision-free and inside the same topology room.
    """
    cx, cy = info["centre_xy"]
    vx, vy = goal_xy[0] - cx, goal_xy[1] - cy
    n = math.hypot(vx, vy)
    if n < 1e-6:
        return None
    ax = goal_xy[0] + GOAL_APPROACH_M * vx / n
    ay = goal_xy[1] + GOAL_APPROACH_M * vy / n
    col, row = occ.to_pixel(ax, ay)
    gcol, grow = occ.to_pixel(goal_xy[0], goal_xy[1])
    if not (0 <= row < comp.shape[0] and 0 <= col < comp.shape[1]
            and 0 <= grow < comp.shape[0] and 0 <= gcol < comp.shape[1]):
        return None
    if comp[row, col] == 0 or comp[row, col] != comp[grow, gcol]:
        return None
    if occ.clearance(ax, ay) <= PLANNING_RADIUS_M:
        return None
    if not _inside_room(room_lab, occ, (ax, ay), room_id, margin_m=0.10):
        return None
    if not occ.seg_free(ax, ay, goal_xy[0], goal_xy[1], PLANNING_RADIUS_M):
        return None
    return (ax, ay)


def _goal_candidates(occ: OccMap, start_xy: Tuple[float, float],
                     comp: np.ndarray, room_lab: np.ndarray,
                     label_index: Dict[str, dict], used_objects: dict
                     ) -> List[Tuple[Tuple[float, float], str, int]]:
    """Enumerate feasible object/stance pairs without a target-class whitelist.

    A candidate is only a proposal.  The caller still has to plan the complete
    route and prove the target in RGB/depth.  Returning several poses is
    important: one occluded face or one bad annotation must not make us either
    accept a weak view or discard an otherwise valid object without trying its
    other accessible sides.
    """
    s_col, s_row = occ.to_pixel(start_xy[0], start_xy[1])
    s_comp = comp[s_row, s_col] if (0 <= s_row < comp.shape[0] and 0 <= s_col < comp.shape[1]) else 0
    s_room = room_lab[s_row, s_col] if room_lab is not None and 0 <= s_row < room_lab.shape[0] and 0 <= s_col < room_lab.shape[1] else 0
    if s_comp == 0 or s_room == 0:
        return []
    objects = []
    for label_N, info in label_index.items():
        label = info["label"].strip().lower().replace(" ", "_")
        if (used_objects.get(label_N, 0) >= GOAL_MAX_EPISODES_PER_OBJECT
                or not label or label in _NON_OBJECT_LABELS):
            continue
        if ALLOWED_TARGET_FAMILIES and _semantic_family(label) not in ALLOWED_TARGET_FAMILIES:
            continue
        x0, y0, x1, y1 = info["bbox_xyxy"]
        z0, z1 = info.get("z_range", (0.0, 0.0))
        dims = (x1 - x0, y1 - y0, z1 - z0)
        if (not all(math.isfinite(v) and v > 0.0 for v in dims)
                or max(dims) > 4.0 or dims[2] < 0.08):
            continue
        objects.append((label_N, info))
    if not objects:
        return []
    ranked_candidates = []
    for oi in rng.permutation(len(objects)):
        label_N, info = objects[int(oi)]
        stances = _goal_stances(info)
        # Infer object-room membership from its physically accessible sides,
        # never from bbox centre (wall-mounted objects often have centre on wall).
        by_room: Dict[int, List[Tuple[float, float]]] = {}
        for x, y in stances:
            col, row = occ.to_pixel(x, y)
            if not (0 <= row < comp.shape[0] and 0 <= col < comp.shape[1]):
                continue
            if occ.clearance(x, y) <= PLANNING_RADIUS_M:
                continue
            rid = _room_id(room_lab, occ, (x, y))
            if rid <= 0 or not _visible_to_bbox(occ, (x, y), info):
                continue
            by_room.setdefault(rid, []).append((x, y))
        if not by_room:
            continue
        target_room = max(by_room, key=lambda rid: len(by_room[rid]))
        if REQUIRE_CROSS_ROOM and target_room == int(s_room):
            continue
        room_stances = by_room[target_room]
        valid = []
        for gi in rng.permutation(len(room_stances)):
            x, y = room_stances[int(gi)]
            col, row = occ.to_pixel(x, y)
            if comp[row, col] != s_comp:                  # must be reachable
                continue
            d = math.hypot(x - start_xy[0], y - start_xy[1])
            if not (PAIR_EUCLIDEAN_M[0] <= d <= PAIR_EUCLIDEAN_M[1]):
                continue
            if not _inside_room(room_lab, occ, (x, y), target_room):
                continue
            if (GOAL_TERMINAL_MODE == "radial_approach"
                    and _goal_approach_point(
                        occ, (x, y), info, comp, room_lab, target_room) is None):
                continue
            area = route_extractor._functional_area(
                (x, y), occ, label_index, room_lab, label_N)
            if area.get("status") != "verified":
                continue
            cx, cy = info["centre_xy"]
            yaw = math.atan2(cy - y, cx - x)
            camera_pos = (x, y, float(occ.ground_z(x, y)) + CAMERA_Z_M)
            projected = _project_goal_bbox(camera_pos, yaw, info)
            if (projected is None
                    or projected["width_px"] < GOAL_MIN_BBOX_WIDTH_PX
                    or projected["height_px"] < GOAL_MIN_BBOX_HEIGHT_PX
                    or projected["projected_area_ratio"]
                    < GOAL_MIN_PROJECTED_AREA_RATIO):
                continue
            valid.append(((x, y), projected["projected_area_ratio"]
                          + 0.001 * occ.clearance(x, y)))
        # Prefer clearance, but keep several alternatives for the expensive
        # visual/route gates. This ranking contains no object-class knowledge.
        valid.sort(key=lambda item: item[1], reverse=True)
        ranked_candidates.extend((score, xy, label_N, target_room)
                                 for xy, score in valid[:GOAL_MAX_STANCES_PER_OBJECT])
    ranked_candidates.sort(key=lambda item: item[0], reverse=True)
    return [(xy, label_N, target_room)
            for _, xy, label_N, target_room in ranked_candidates]


def _swept_free(occ: OccMap, waypoints, step_m: float = 0.05) -> bool:
    """Every point along the polyline (resampled at step_m) must be free."""
    for i in range(len(waypoints) - 1):
        x0, y0 = waypoints[i][:2]
        x1, y1 = waypoints[i + 1][:2]
        seg = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(seg / step_m) + 1)
        for k in range(n + 1):
            t = k / float(n)
            if not occ.is_free(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, ROBOT_RADIUS_M):
                return False
    return True


def _execution_replay_free(occ: OccMap, waypoints, step_m: Optional[float] = None,
                           radius_m: float = ROBOT_RADIUS_M) -> Tuple[bool, dict]:
    """Replay the exact geometric contract used by ``run_golden``.

    The older swept-point check could miss a one-pixel obstacle depending on
    sampling phase.  Here the path is arc-resampled exactly like the console,
    and every target is executed through ``OccMap.try_step`` with sliding.  A
    step that achieves <95% of the requested displacement is the same hard
    block that ``agent_tools.drive_forward`` reports at runtime.
    """
    step_m = REPLAY_STEP_M if step_m is None else float(step_m)
    pts = np.asarray([[float(p[0]), float(p[1])] for p in waypoints], dtype=float)
    if len(pts) < 2:
        return False, {"reason": "short_path"}
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total < 1e-6:
        return False, {"reason": "zero_path"}
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(math.ceil(total / step_m)))
    targets = np.linspace(0.0, total, n)
    path = np.column_stack((np.interp(targets, cum, pts[:, 0]),
                            np.interp(targets, cum, pts[:, 1])))
    x, y = float(path[0, 0]), float(path[0, 1])
    collisions = 0
    for k in range(1, len(path)):
        tx, ty = float(path[k, 0]), float(path[k, 1])
        requested = math.hypot(tx - x, ty - y)
        nx, ny, hit = occ.try_step(x, y, tx, ty, radius_m, True)
        moved = math.hypot(nx - x, ny - y)
        collisions += int(bool(hit))
        if requested > 1e-9 and moved < requested * 0.95:
            return False, {"reason": "blocked", "step": k,
                           "requested_m": requested, "moved_m": moved,
                           "xy": [x, y], "target_xy": [tx, ty]}
        x, y = nx, ny
    return True, {"steps": len(path) - 1, "collisions": collisions,
                  "end_error_m": math.hypot(x - path[-1, 0], y - path[-1, 1])}


def _yaw_to_quat_wxyz(yaw: float) -> List[float]:
    """scalar-first [w,x,y,z] for a pure-z yaw (matches datasets.yaw_from_wxyz)."""
    h = 0.5 * yaw
    return [math.cos(h), 0.0, 0.0, math.sin(h)]


def _waypoint_yaws(waypoints) -> List[float]:
    """Tangent heading per waypoint (rad, +x->+y); last copies previous."""
    yaws = []
    for i in range(len(waypoints)):
        if i < len(waypoints) - 1:
            dx = waypoints[i + 1][0] - waypoints[i][0]
            dy = waypoints[i + 1][1] - waypoints[i][1]
            yaws.append(math.atan2(dy, dx))
        else:
            yaws.append(yaws[-1] if yaws else 0.0)
    return yaws


def _project_goal_bbox(camera_pos, yaw: float, info: dict) -> Optional[dict]:
    return project_bbox(VISIBILITY_CAMERA, camera_pos, yaw, info)


def _ray_box_interval(camera_pos, yaw: float, u: float, v: float, info: dict):
    return ray_box_interval(VISIBILITY_CAMERA, camera_pos, yaw, u, v, info)


def _detection_from_object(obj: dict) -> Optional[dict]:
    if not isinstance(obj, dict) or obj.get("visible") is False:
        return None
    bbox = obj.get("bbox_2d", obj.get("bbox"))
    if bbox is None and obj.get("visible") is not True:
        return None
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        return None
    # Qwen's native visual grounding convention is normalized 0..1000 even
    # when pixel coordinates are requested. Accept both conventions explicitly.
    normalized = ("bbox_2d" in obj or max(x0, x1) > CAMERA_WIDTH
                  or max(y0, y1) > CAMERA_HEIGHT_PX)
    if normalized:
        x0, x1 = x0 * CAMERA_WIDTH / 1000.0, x1 * CAMERA_WIDTH / 1000.0
        y0, y1 = y0 * CAMERA_HEIGHT_PX / 1000.0, y1 * CAMERA_HEIGHT_PX / 1000.0
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0, x1 = max(0.0, min(float(CAMERA_WIDTH), x0)), \
        max(0.0, min(float(CAMERA_WIDTH), x1))
    y0, y1 = max(0.0, min(float(CAMERA_HEIGHT_PX), y0)), \
        max(0.0, min(float(CAMERA_HEIGHT_PX), y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return {"bbox_px": [x0, y0, x1, y1],
            "width_px": x1 - x0, "height_px": y1 - y0,
            "area_ratio": (x1 - x0) * (y1 - y0)
                          / float(CAMERA_WIDTH * CAMERA_HEIGHT_PX)}


def _parse_detection_answers(answer: str) -> List[dict]:
    """Parse one or many VLM boxes from JSON/fenced JSON."""
    text = str(answer).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text,
                      flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        match = re.search(r"(\[.*\]|\{.*\})", text, flags=re.DOTALL)
        if match is None:
            return []
        try:
            payload = json.loads(match.group(1))
        except (TypeError, ValueError):
            return []
    objects = payload if isinstance(payload, list) else [payload]
    return [det for det in (_detection_from_object(obj) for obj in objects)
            if det is not None]


def _parse_detection_answer(answer: str) -> Optional[dict]:
    """Backward-compatible single-box helper used by unit tests."""
    detections = _parse_detection_answers(answer)
    return detections[0] if detections else None


def _detection_matches_projection(detection: dict, projected: dict) -> bool:
    """Require a localized semantic object to agree with the 3-D annotation.

    The semantic model sees an unannotated RGB frame, so it cannot simply agree
    with a suggested red box.  Geometry is consulted only after localization.
    """
    if (detection["width_px"] < GOAL_MIN_DETECTION_WIDTH_PX
            or detection["height_px"] < GOAL_MIN_DETECTION_HEIGHT_PX
            or detection["area_ratio"] < GOAL_MIN_DETECTION_AREA_RATIO):
        return False
    dx0, dy0, dx1, dy1 = detection["bbox_px"]
    px0, py0, px1, py1 = projected["bbox_px"]
    ix0, iy0 = max(dx0, px0), max(dy0, py0)
    ix1, iy1 = min(dx1, px1), min(dy1, py1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    d_area = (dx1 - dx0) * (dy1 - dy0)
    dcx, dcy = 0.5 * (dx0 + dx1), 0.5 * (dy0 + dy1)
    return (px0 <= dcx <= px1 and py0 <= dcy <= py1
            and inter / max(d_area, 1e-9) >= 0.60)


def _semantic_localize(rgb, label: str) -> Tuple[List[dict], dict]:
    """Locate a category in raw RGB without exposing the projected 3-D box."""
    import requests
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=92)
    image_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    prompt = (
        "Inspect this unannotated 640x480 RGB image. Locate one clearly and "
        "recognizably visible instance of the exact object category '%s'. "
        "Do not infer an object from room context. If no instance is visually "
        "unambiguous, return {\"visible\":false,\"bbox\":null}. Otherwise "
        "return {\"visible\":true,\"bbox\":[x0,y0,x1,y1]} using pixel "
        "coordinates. Return JSON only." % label)
    payload = {"model": GOAL_VLM_MODEL, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": prompt}]}], "temperature": 0, "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False}}
    response = requests.post(GOAL_VLM_URL, json=payload, timeout=120)
    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"].strip()
    detections = _parse_detection_answers(answer)
    return detections, {"model": GOAL_VLM_MODEL, "mode": "blind_localization",
                        "temperature": 0, "answer": answer[:500],
                        "detections": len(detections)}


def _goal_view(render_client, scene: str, occ: OccMap, xy, yaw: float,
               info: dict, save_path: str = "") -> dict:
    """Class-agnostic gate: blind localization -> 3-D match -> metric depth."""
    camera_pos = (float(xy[0]), float(xy[1]),
                  float(occ.ground_z(float(xy[0]), float(xy[1]))) + CAMERA_Z_M)
    projected = _project_goal_bbox(camera_pos, yaw, info)
    result = {"status": "rejected", "camera_xy": [round(xy[0], 3), round(xy[1], 3)],
              "camera_yaw": round(float(yaw), 6)}
    if projected is None:
        result["reason"] = "behind_camera"
        return result
    result.update(projected)
    cx, cy = projected["center_px"]
    geometry_ok = (projected["width_px"] >= GOAL_MIN_BBOX_WIDTH_PX
                   and projected["height_px"] >= GOAL_MIN_BBOX_HEIGHT_PX
                   and projected["projected_area_ratio"] >= GOAL_MIN_PROJECTED_AREA_RATIO
                   and 0.2 * CAMERA_WIDTH <= cx <= 0.8 * CAMERA_WIDTH
                   and 0.1 * CAMERA_HEIGHT_PX <= cy <= 0.9 * CAMERA_HEIGHT_PX)
    if not geometry_ok:
        result["reason"] = "weak_projection"
        return result
    rgb, depth = render_client.render_one(
        "sage-" + scene, camera_pos, yaw, want_depth=True)
    if depth is None or np.asarray(depth).ndim != 2:
        raise RuntimeError("render server returned no metric depth for goal visibility")
    depth = np.asarray(depth, dtype=np.float32)
    label = str(info.get("label", "target")).replace("_", " ").lower()
    try:
        detections, verifier = _semantic_localize(rgb, label)
    except Exception as exc:
        raise RuntimeError("mandatory blind goal localizer failed: %s" % exc) from exc
    result["semantic_verifier"] = verifier
    if not detections:
        result["reason"] = "semantic_target_not_localized"
        return result
    matching = [det for det in detections
                if _detection_matches_projection(det, projected)]
    if not matching:
        result["candidate_detected_bboxes_px"] = [
            [round(v, 1) for v in det["bbox_px"]] for det in detections]
        result["reason"] = "semantic_geometry_mismatch"
        return result
    detection = max(matching, key=lambda det: det["area_ratio"])
    result["detected_bbox_px"] = [round(v, 1) for v in detection["bbox_px"]]
    result["detected_area_ratio"] = round(detection["area_ratio"], 6)
    # Depth support is measured inside the independently detected object box,
    # not over the possibly loose projected AABB. This prevents background
    # walls inside a large annotation box from masquerading as target pixels.
    support = depth_support(
        depth, detection["bbox_px"], VISIBILITY_CAMERA, camera_pos, yaw, info,
        GOAL_DEPTH_SAMPLE_STRIDE, GOAL_DEPTH_TOLERANCE_M)
    visible_fraction = support["visible_fraction"]
    depth_area_ratio = support["visible_area_ratio"]
    result.update({"depth_tested_rays": support["tested_rays"],
                   "visible_target_rays": support["visible_rays"],
                   "visible_target_pixels_est": support["visible_rays"]
                                                * GOAL_DEPTH_SAMPLE_STRIDE ** 2,
                   "visible_bbox_fraction": round(visible_fraction, 6),
                   "depth_visible_area_ratio": round(depth_area_ratio, 6)})
    if (depth_area_ratio < GOAL_MIN_DEPTH_AREA_RATIO
            or visible_fraction < GOAL_MIN_DETECTION_DEPTH_FRACTION):
        result["reason"] = "occluded_or_too_small"
        return result
    try:
        from PIL import ImageDraw
        audit = rgb.copy()
        draw = ImageDraw.Draw(audit)
        draw.rectangle(projected["bbox_px"], outline=(255, 0, 0), width=3)
        draw.rectangle(detection["bbox_px"], outline=(0, 255, 0), width=3)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            audit.save(save_path, quality=90)
    except Exception as exc:
        raise RuntimeError("goal audit image failed: %s" % exc) from exc
    result["status"] = "verified"
    result["reason"] = "blind_localization_geometry_and_depth"
    return result


def _confirmation_goal_view(render_client, scene: str, occ: OccMap, xy,
                            yaw: float, info: dict, room_lab: np.ndarray,
                            room_id: int, save_path: str = "") -> dict:
    """Require the same category/geometry agreement from a nearby viewpoint."""
    side = (-math.sin(yaw), math.cos(yaw))
    cx, cy = info["centre_xy"]
    failures = []
    for sign in (-1.0, 1.0):
        probe = (xy[0] + sign * GOAL_CONFIRMATION_OFFSET_M * side[0],
                 xy[1] + sign * GOAL_CONFIRMATION_OFFSET_M * side[1])
        if (occ.clearance(probe[0], probe[1]) <= PLANNING_RADIUS_M
                or not _inside_room(room_lab, occ, probe, room_id,
                                    margin_m=0.10)
                or not occ.seg_free(xy[0], xy[1], probe[0], probe[1],
                                    PLANNING_RADIUS_M)):
            failures.append({"camera_xy": [round(probe[0], 3), round(probe[1], 3)],
                             "reason": "unsafe_confirmation_pose"})
            continue
        probe_yaw = math.atan2(cy - probe[1], cx - probe[0])
        suffix = ".confirm.jpg" if save_path.endswith(".jpg") else ".confirm"
        probe_path = save_path[:-4] + suffix if save_path.endswith(".jpg") else save_path + suffix
        view = _goal_view(render_client, scene, occ, probe, probe_yaw,
                          info, probe_path)
        if view.get("status") == "verified":
            view["offset_from_terminal_m"] = GOAL_CONFIRMATION_OFFSET_M
            return view
        failures.append(view)
    return {"status": "rejected", "reason": "cross_view_confirmation_failed",
            "attempts": failures}


# --------------------------------------------------------------------------- #
# Stage D: deterministic realization + grounding check
# --------------------------------------------------------------------------- #
FORBIDDEN_NAV_VERBS = (
    "take ", "carry ", "pick up", "place it", "place the", "place a", "bring ", "grab ", "fetch ",
    "retrieve ", "lie down",
)


def _label(label_index: Dict[str, dict], object_id: str) -> str:
    return str(label_index[object_id]["label"]).strip().lower().replace("_", " ")


def _semantic_family(label: str) -> str:
    """Coarse families used only to keep perception cases unambiguous."""
    return semantic_family(label)


def _visible_same_family_distractors(occ: OccMap, xy, yaw: float,
                                     target_id: str,
                                     label_index: Dict[str, dict]) -> list:
    """Conservative panoramic distractors near the early observation pose.

    A wrong turn can reveal an object just outside the current camera frustum,
    as happened with cabinet vs wine_cabinet.  Perception cases therefore
    exclude same-family objects within 6 m and 2-D line of sight, not merely
    those already projected into the current frame.
    """
    target_family = _semantic_family(label_index[target_id]["label"])
    out = []
    for object_id, info in label_index.items():
        if object_id == target_id or _semantic_family(info["label"]) != target_family:
            continue
        x0, y0, x1, y1 = info["bbox_xyxy"]
        distance = math.hypot(max(x0 - xy[0], 0.0, xy[0] - x1),
                              max(y0 - xy[1], 0.0, xy[1] - y1))
        if distance <= 6.0 and _visible_to_bbox(occ, xy, info):
            out.append(object_id)
    return out


def _door_phrase(count: int) -> str:
    if count <= 0:
        return ""
    return "through the doorway" if count == 1 else "through %d doorways" % count


def _instruction(t: str, level: str, route: dict, text: str, *,
                 object_ids=(), rooms=(), door_count=None, turns=(),
                 landmark_ids=(), purpose_id=None) -> dict:
    return {
        "instruction_type": t,
        "level": level,
        "start": route["start"]["object"],
        "end": route["goal"]["object"],
        "generated_instruction": text,
        "grounding": {
            "generator": "deterministic_grounded_v2",
            "object_ids": list(object_ids),
            "rooms": list(rooms),
            "door_count": door_count,
            "turns": list(turns),
            "landmark_ids": list(landmark_ids),
            "purpose_id": purpose_id,
        },
    }


def _grounding_ok(instr: dict, route: dict, label_index: Dict[str, dict]) -> bool:
    """Strict validation of the deterministic fact specification."""
    if instr.get("start") != route["start"]["object"] \
            or instr.get("end") != route["goal"]["object"]:
        return False
    grounding = instr.get("grounding")
    if not isinstance(grounding, dict) \
            or grounding.get("generator") != "deterministic_grounded_v2":
        return False
    goal_id = route["goal"]["object"]
    landmark_ids = [x["object_id"] for x in route.get("landmarks", [])]
    allowed_objects = {route["start"]["object"], goal_id, *landmark_ids}
    if any(x not in allowed_objects or x not in label_index
           for x in grounding.get("object_ids", [])):
        return False
    allowed_rooms = set(route.get("cross_rooms", [])) | {
        route["start"]["room"], route["goal"]["room"]}
    if any(room not in allowed_rooms for room in grounding.get("rooms", [])):
        return False
    if grounding.get("door_count") is not None \
            and grounding["door_count"] != route["doors_crossed"]:
        return False
    if grounding.get("turns", []) != (route["turns"] if
                                      instr["instruction_type"] == "Maneuver-based" else []):
        return False
    used_landmarks = grounding.get("landmark_ids", [])
    if instr["instruction_type"] == "Landmark-sequence":
        if not used_landmarks or used_landmarks != landmark_ids[:len(used_landmarks)]:
            return False
    elif used_landmarks:
        return False
    text = str(instr.get("generated_instruction", "")).strip().lower()
    return bool(text) and not any(verb in text for verb in FORBIDDEN_NAV_VERBS)


def _instruction_contract_ok(instructions: list, route: dict,
                             label_index: Dict[str, dict]) -> bool:
    """Validate either the canonical four-way contract or one curriculum option."""
    if CAPABILITY_PROFILE in CURRICULUM_OPTION_PROFILES:
        return (isinstance(instructions, list) and len(instructions) == 1
                and isinstance(instructions[0], dict)
                and instructions[0].get("instruction_type") == "Maneuver-based"
                and instructions[0].get("level") == CAPABILITY_PROFILE
                and _grounding_ok(instructions[0], route, label_index))
    if (not isinstance(instructions, list) or len(instructions) != 4
            or any(not isinstance(ins, dict) for ins in instructions)):
        return False
    pairs = {(ins.get("instruction_type"), ins.get("level"))
             for ins in instructions if isinstance(ins, dict)}
    return (pairs == EXPECTED_INSTRUCTION_PAIRS
            and all(str(ins.get("generated_instruction", "")).strip()
                    and _grounding_ok(ins, route, label_index)
                    for ins in instructions))


def _gen_action_head_option(route: dict, label_index: Dict[str, dict]) -> list:
    """One grounded local option for the P0--P2 action expert curriculum."""
    profile = CAPABILITY_PROFILE
    goal_id = route["goal"]["object"]
    goal = _label(label_index, goal_id)
    turns = list(route.get("turns", []))
    directional = next((turn for turn in turns if turn in {"left", "right"}), None)
    doors = int(route.get("doors_crossed") or 0)
    if profile == "action_head_p0":
        text = ("Turn %s and stop beside the %s." % (directional, goal)
                if directional else "Go straight and stop beside the %s." % goal)
    elif profile == "action_head_p1":
        text = ("Turn %s and approach the %s." % (directional, goal)
                if directional else "Go straight to the %s." % goal)
    else:
        if doors:
            prefix = "Go through the doorway"
            text = ("%s, turn %s, then stop beside the %s." %
                    (prefix, directional, goal) if directional else
                    "%s and stop beside the %s." % (prefix, goal))
        else:
            text = ("Turn %s, then approach and stop beside the %s." %
                    (directional, goal) if directional else
                    "Go straight to the %s and stop beside it." % goal)
    return [_instruction("Maneuver-based", profile, route, text,
                         object_ids=[goal_id], rooms=[route["goal"]["room"]],
                         door_count=doors if doors else None, turns=turns)]


def _gen_instructions(route: dict, label_index: Dict[str, dict]) -> Optional[list]:
    """Build facts first, then realize them with deterministic safe templates."""
    if CAPABILITY_PROFILE in CURRICULUM_OPTION_PROFILES:
        return _gen_action_head_option(route, label_index)
    goal_id = route["goal"]["object"]
    goal = _label(label_index, goal_id)
    gr = route["goal"]["room"]
    area_grounding = route.get("goal_area_grounding", {})
    if area_grounding.get("status") != "verified" or not area_grounding.get("anchors"):
        return None
    doors = int(route["doors_crossed"])
    door = _door_phrase(doors)
    object_low = ("Go %s and stop beside the %s." % (door, goal)
                  if door else "Navigate to the %s and stop beside it." % goal)
    area_low = ("Go %s and stop in the %s." % (door, gr)
                if door else "Navigate to the %s and stop there." % gr)
    out = [
        _instruction("Object-Goal", "high", route, "Go to the %s in the %s." % (goal, gr), object_ids=[goal_id], rooms=[gr]),
        _instruction("Object-Goal", "low", route, object_low, object_ids=[goal_id], rooms=[gr], door_count=doors if doors else None),
        _instruction("Area-based", "high", route, "Go to the %s." % gr, rooms=[gr]),
        _instruction("Area-based", "low", route, area_low, rooms=[gr], door_count=doors if doors else None),
    ]
    if not _instruction_contract_ok(out, route, label_index):
        raise RuntimeError("deterministic instruction generator violated contract")
    return out



# --------------------------------------------------------------------------- #
# Stage D: assemble GVLN json (geometry -> stored rot180 frame)
# --------------------------------------------------------------------------- #
def _to_stored(gx: float, gy: float, gyaw: float) -> dict:
    """geometry-frame (x,y,yaw) -> json point in the SAGE-stored rot180 frame.

    ``datasets._to_geometry_frame`` is self-inverse (rot180 twice = identity),
    so writing the SAME rot180 the reader applies makes the reader recover the
    geometry frame verbatim. Verified round-trip <1 px before batch write.
    """
    yaw_stored = gyaw + math.pi
    if yaw_stored > math.pi:
        yaw_stored -= 2 * math.pi
    return {"position": [-gx, -gy, TRAJ_Z], "rotation": _yaw_to_quat_wxyz(yaw_stored)}


def _assemble_sample(traj_id: str, waypoints, yaws, instructions: list) -> dict:
    points = []
    for (x, y), yaw in zip(waypoints, yaws):
        points.append(_to_stored(x, y, yaw))
    return {"trajectory_id": traj_id, "points": points, "instructions": instructions}


def _roundtrip_check(sample: dict, start_xy: Tuple[float, float],
                      goal_xy: Tuple[float, float]) -> bool:
    """Read back one sample's stored points through _to_geometry_frame; start/goal
    must match the geometry-frame endpoints within 1 occupancy pixel."""
    from .datasets import _to_geometry_frame, yaw_from_wxyz
    pts = sample["points"]
    p0 = pts[0]
    sx, sy, _ = _to_geometry_frame(float(p0["position"][0]),
                                    float(p0["position"][1]), 0.0)
    pg = pts[-1]
    gx, gy, _ = _to_geometry_frame(float(pg["position"][0]),
                                    float(pg["position"][1]), 0.0)
    tol = 0.06                # ~1 pixel at 5 cm grid
    ok = (abs(sx - start_xy[0]) < tol and abs(sy - start_xy[1]) < tol
          and abs(gx - goal_xy[0]) < tol and abs(gy - goal_xy[1]) < tol)
    if not ok:
        print("   [roundtrip] FAIL start(%.2f,%.2f)->(%.2f,%.2f) goal(%.2f,%.2f)->(%.2f,%.2f)"
              % (start_xy[0], start_xy[1], sx, sy, goal_xy[0], goal_xy[1], gx, gy))
    return ok


# --------------------------------------------------------------------------- #
# per-scene pipeline
# --------------------------------------------------------------------------- #
def _try_candidate(bare: str, ep: int, occ: OccMap, render_client,
                   comp: np.ndarray, room_lab: np.ndarray,
                   label_index: Dict[str, dict], start_xy: Tuple[float, float],
                   candidate) -> Optional[Tuple[dict, dict]]:
    """Run every hard gate for one proposed goal stance; never relax a gate."""
    goal_xy, g_obj, goal_room = candidate
    plan_goal_xy = goal_xy
    if GOAL_TERMINAL_MODE == "radial_approach":
        plan_goal_xy = _goal_approach_point(
            occ, goal_xy, label_index[g_obj], comp, room_lab, goal_room)
        if plan_goal_xy is None:
            return None
    raw_wp = plan_path(
        occ, start_xy, plan_goal_xy, PLANNING_RADIUS_M, mode=PLANNER_MODE,
        centered_config=CENTERED_CONFIG)
    if raw_wp is None or len(raw_wp) < 2:
        return None
    raw_wp = list(raw_wp)
    if math.hypot(raw_wp[-1][0] - goal_xy[0], raw_wp[-1][1] - goal_xy[1]) > 1e-6:
        raw_wp.append(goal_xy)
    try:
        wp, smoothing = smooth_path(occ, raw_wp, PLANNING_RADIUS_M,
                                    SMOOTHING_CONFIG)
    except ValueError:
        return None
    if smoothing.get("blend_strength", 0.0) + 1e-9 < MIN_ACCEPTED_SMOOTHING_BLEND:
        if TRACE_REJECTIONS:
            print("    reject %s smoothing_blend=%.3g < %.3g" % (
                g_obj, smoothing.get("blend_strength", 0.0),
                MIN_ACCEPTED_SMOOTHING_BLEND), flush=True)
        return None
    if not _swept_free(occ, wp):
        return None
    replay_ok, replay = _execution_replay_free(occ, wp)
    if not replay_ok or not _path_finishes_in_room(occ, wp, room_lab, goal_room):
        return None
    length = sum(math.hypot(wp[i + 1][0] - wp[i][0],
                            wp[i + 1][1] - wp[i][1])
                 for i in range(len(wp) - 1))
    if not (PAIR_GEODESIC_M[0] <= length <= PAIR_GEODESIC_M[1]):
        return None
    s_obj = _snap_object(start_xy, label_index)
    route = route_extractor.extract(
        wp, occ, label_index, bare, s_obj, g_obj, turn_waypoints=raw_wp)
    route_turns = len([x for x in route.get("turns", []) if x != "straight"])
    route_doors = int(route.get("doors_crossed") or 0)
    route_landmarks = len(route.get("landmarks") or [])
    min_clearance = min(float(occ.clearance(x, y)) for x, y in wp)
    if not (MIN_ROUTE_TURNS <= route_turns <= MAX_ROUTE_TURNS
            and MIN_ROUTE_DOORS <= route_doors <= MAX_ROUTE_DOORS
            and route_turns + route_doors >= MIN_ROUTE_DECISIONS
            and route_landmarks >= MIN_ROUTE_LANDMARKS
            and MIN_ROUTE_CLEARANCE_M <= min_clearance <= MAX_ROUTE_CLEARANCE_M):
        return None
    yaws = _waypoint_yaws(wp)
    gcx, gcy = label_index[g_obj]["centre_xy"]
    target_yaw = math.atan2(gcy - goal_xy[1], gcx - goal_xy[0])
    target_bearing_error = abs(math.atan2(
        math.sin(target_yaw - yaws[-1]), math.cos(target_yaw - yaws[-1])))
    if math.degrees(target_bearing_error) > GOAL_MAX_NATURAL_BEARING_ERROR_DEG:
        return None
    start_visibility = None
    if START_TARGET_VISIBILITY_MAX_ARC_M > 0:
        arc = 0.0
        probes = []
        for i in range(len(wp)):
            if i:
                arc += math.hypot(wp[i][0] - wp[i - 1][0],
                                  wp[i][1] - wp[i - 1][1])
            if arc > START_TARGET_VISIBILITY_MAX_ARC_M + 1e-6:
                break
            if not probes or arc - probes[-1][0] >= 0.25 or i == 0:
                probes.append((arc, i))
        for probe_arc, i in probes:
            view = _goal_view(render_client, bare, occ, wp[i], yaws[i],
                              label_index[g_obj])
            if view.get("status") == "verified":
                start_visibility = {"first_visible_arc_m": round(probe_arc, 3),
                                    "path_index": i, "view": view}
                break
        if start_visibility is None:
            return None
        distractors = _visible_same_family_distractors(
            occ, wp[start_visibility["path_index"]],
            yaws[start_visibility["path_index"]], g_obj, label_index)
        if len(distractors) > MAX_VISIBLE_SAME_FAMILY_DISTRACTORS:
            return None
        start_visibility["visible_same_family_distractors"] = distractors
    instrs = _gen_instructions(route, label_index)
    if instrs is None:
        return None
    traj_id = "demo_%s_%d" % (bare, ep)
    audit_path = os.path.join(DEMO_ROOT, "goal_visibility", bare,
                              "%s.jpg" % traj_id)
    goal_visibility = _goal_view(render_client, bare, occ, goal_xy,
                                 yaws[-1], label_index[g_obj], audit_path)
    if goal_visibility.get("status") != "verified":
        if TRACE_REJECTIONS:
            print("    reject %s terminal_view=%s answer=%r" %
                  (g_obj, goal_visibility.get("reason"),
                   goal_visibility.get("semantic_verifier", {}).get("answer")),
                  flush=True)
        return None
    confirmation = _confirmation_goal_view(
        render_client, bare, occ, goal_xy, yaws[-1], label_index[g_obj],
        room_lab, goal_room, audit_path)
    if confirmation.get("status") != "verified":
        if TRACE_REJECTIONS:
            print("    reject %s confirmation=%s" %
                  (g_obj, confirmation.get("reason")), flush=True)
        return None
    goal_visibility["required_views"] = 2
    goal_visibility["confirmation_view"] = confirmation
    sample = _assemble_sample(traj_id, wp, yaws, instrs)
    sample["route_grounding"] = route
    sample["goal_constraints"] = {
        "object": g_obj,
        "room_id": int(goal_room),
        "visible_from_goal": True,
        "terminal_view_is_path_tangent": True,
        "target_bearing_error_deg": round(math.degrees(target_bearing_error), 3),
        "terminal_mode": GOAL_TERMINAL_MODE,
        "planner": PLANNER_MODE,
        "planning_radius_m": PLANNING_RADIUS_M,
        "path_smoothing": smoothing,
        "execution_replay": replay,
        "goal_visibility": goal_visibility,
        "capability_profile": CAPABILITY_PROFILE,
        "route_min_clearance_m": round(min_clearance, 4),
        "start_target_visibility": start_visibility,
    }
    if bare == SCENES[0] and ep == 0 \
            and not _roundtrip_check(sample, start_xy, goal_xy):
        raise RuntimeError("geometry/stored-frame round-trip failed")
    meta = {"goal": g_obj, "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
            "goal_room": goal_room, "length": length,
            "smoothing": smoothing, "replay": replay,
            "goal_visibility": goal_visibility,
            "target_bearing_error": target_bearing_error,
            "standoff": math.hypot(gcx - goal_xy[0], gcy - goal_xy[1]),
            "route": route, "instructions": instrs}
    return sample, meta


def _grounding_record(accepted, label_index: Dict[str, dict]) -> dict:
    sample, meta = accepted
    route = meta["route"]
    return {"target_object_id": meta["goal"],
            "target_term": _label(label_index, meta["goal"]),
            "goal_xy": meta["goal_xy"], "path_length_m": meta["length"],
            "turn_count": len([x for x in route.get("turns", []) if x != "straight"]),
            "door_count": int(route.get("doors_crossed") or 0),
            "accepted": accepted}


def _attach_grounding_pair(bare: str, pair_index: int, pair) -> list:
    records = list(pair)
    contrast_id = "grounding_%s_%d" % (bare, pair_index)
    candidates = [{"target_object_id": r["target_object_id"],
                   "target_term": r["target_term"], "goal_xy": r["goal_xy"],
                   "path_length_m": round(float(r["path_length_m"]), 4),
                   "turn_count": r["turn_count"], "door_count": r["door_count"]}
                  for r in records]
    out = []
    for slot, record in enumerate(records):
        sample, meta = record["accepted"]
        sample["trajectory_id"] = "demo_%s_%d" % (bare, pair_index * 2 + slot)
        sample["capability_grounding"] = {
            "contract": "shared-start-balanced-routes-v1",
            "contrast_set_id": contrast_id,
            "target_object_id": record["target_object_id"],
            "candidates": candidates,
            "balance": dict(GROUNDING_PAIR_CONTRACT),
        }
        out.append((sample, meta))
    return out


def _build_grounding_pair_for_start(bare, pair_index, occ, render_client,
                                    comp, room_lab, label_index, start_xy,
                                    candidates):
    """Evaluate ordinary valid routes, then select one balanced shared-start pair."""
    records = []
    for candidate_no, candidate in enumerate(
            candidates[:GROUNDING_MAX_CANDIDATES_PER_START]):
        accepted = _try_candidate(
            bare, pair_index * 1000 + candidate_no, occ, render_client,
            comp, room_lab, label_index, start_xy, candidate)
        if accepted is not None:
            records.append(_grounding_record(accepted, label_index))
    pair = select_grounding_pair(records, GROUNDING_PAIR_CONTRACT)
    return _attach_grounding_pair(bare, pair_index, pair) if pair else []


def build_scene(bare: str) -> dict:
    print("\n=== scene %s ===" % bare)
    occ = _load_occ(bare)
    render_client = XYZEulerClient(url=RENDER_URL, camera=VISIBILITY_CAMERA)
    if not render_client.ping():
        raise RuntimeError("goal visibility requires render server at %s" % RENDER_URL)
    label_index = _build_label_index(bare)
    print("  [A] occ %s mode=%s free=%d labels=%d" % (
        bare, occ.collision_mode, int(occ.free.sum()), len(label_index)))

    passable = passable_grid(occ, ROBOT_RADIUS_M)
    comp = _connected_components(passable)
    room_lab = route_extractor._room_label_map(occ)
    room_names = route_extractor._room_names(room_lab, occ, label_index)
    print("  [A] rooms=%d comps=%d" % (
        (int(room_lab.max()) if room_lab is not None else 0), int(comp.max())))

    os.makedirs(os.path.join(DEMO_ROOT, "scene_index"), exist_ok=True)
    with open(os.path.join(DEMO_ROOT, "scene_index", "%s.json" % bare), "w",
              encoding="utf-8") as f:
        json.dump({"label_index": label_index, "room_names": room_names}, f)

    # Stage B + C: sample episodes
    starts = _poisson_disk_starts(occ, EPISODES_PER_SCENE * 8, START_MIN_SPACING_M)
    samples: List[dict] = []
    used_goal_objects: Dict[str, int] = {}
    ep = 0
    if CAPABILITY_PROFILE == "grounding" and EPISODES_PER_SCENE % 2:
        raise ValueError("grounding profile episodes_per_scene must be even")
    for start_xy in starts:
        if ep >= EPISODES_PER_SCENE:
            break
        accepted = None
        candidates = _goal_candidates(
            occ, start_xy, comp, room_lab, label_index, used_goal_objects)
        if CAPABILITY_PROFILE == "grounding":
            paired = _build_grounding_pair_for_start(
                bare, ep // 2, occ, render_client, comp, room_lab,
                label_index, start_xy, candidates)
            if not paired:
                continue
            for sample, meta in paired:
                samples.append(sample)
                goal = meta["goal"]
                used_goal_objects[goal] = used_goal_objects.get(goal, 0) + 1
                print("  [Grounding] pair=%d target=%s len=%.1fm turns=%d doors=%d" % (
                    ep // 2, goal, meta["length"],
                    len([x for x in meta["route"].get("turns", []) if x != "straight"]),
                    int(meta["route"].get("doors_crossed") or 0)))
            ep += 2
            continue
        for candidate_no, candidate in enumerate(candidates, 1):
            if TRACE_REJECTIONS:
                print("  [try] ep%d candidate=%d/%d goal=%s xy=(%.2f,%.2f)" %
                      (ep, candidate_no, len(candidates), candidate[1],
                       candidate[0][0], candidate[0][1]), flush=True)
            accepted = _try_candidate(
                bare, ep, occ, render_client, comp, room_lab, label_index,
                start_xy, candidate)
            if accepted is not None:
                break
        if accepted is None:
            continue
        sample, meta = accepted
        g_obj = meta["goal"]
        goal_visibility = meta["goal_visibility"]
        route = meta["route"]
        samples.append(sample)
        used_goal_objects[g_obj] = used_goal_objects.get(g_obj, 0) + 1
        print("  [C-E] ep%d len=%.1fm smooth=%.3g goal=%s area=%s rgb=%.2f%% depth=%.2f%% standoff=%.2fm facing_err=%.3fdeg doors=%d turns=%s instrs=%d views=2 candidates=%d"
              % (ep, meta["length"], meta["smoothing"]["blend_strength"], g_obj,
                 route["goal"]["room"],
                 100.0 * goal_visibility["projected_area_ratio"],
                 100.0 * goal_visibility["depth_visible_area_ratio"],
                 meta["standoff"],
                 math.degrees(meta["target_bearing_error"]), route["doors_crossed"],
                 "+".join(route["turns"])[:40], len(meta["instructions"]),
                 len(candidates)))
        ep += 1

    if len(samples) != EPISODES_PER_SCENE:
        if CAPABILITY_PROFILE in CURRICULUM_OPTION_PROFILES and samples:
            print("  [D] PARTIAL curriculum scene: built %d/%d; committing valid trajectories" %
                  (len(samples), EPISODES_PER_SCENE))
        else:
            print("  [D] FAIL incomplete scene: built %d/%d; preserving previous output" %
                  (len(samples), EPISODES_PER_SCENE))
            return {"scene": bare, "ok": False, "n": len(samples),
                    "err": "incomplete production batch"}
    if CAPABILITY_PROFILE in CURRICULUM_OPTION_PROFILES:
        invalid = any(
            len(sample.get("instructions", [])) != 1
            or sample["instructions"][0].get("instruction_type") != "Maneuver-based"
            or sample["instructions"][0].get("level") != CAPABILITY_PROFILE
            for sample in samples)
    else:
        invalid = any(len(sample.get("instructions", [])) != 4 or
                      {(ins.get("instruction_type"), ins.get("level"))
                       for ins in sample.get("instructions", [])} != EXPECTED_INSTRUCTION_PAIRS
                      for sample in samples)
    if invalid:
        raise RuntimeError("instruction contract changed before write")

    # Stage GVLN JSON beside the destination. main() commits all three only
    # after every scene passes, so a later scene failure preserves the complete
    # previous batch rather than leaving mixed pipeline generations.
    os.makedirs(os.path.join(OUT_VLN, bare), exist_ok=True)
    out_path = os.path.join(OUT_VLN, bare, "%s_demo.json" % bare)
    doc = {
        "pipeline": {
            "name": PIPELINE_NAME,
            "version": PIPELINE_VERSION,
            "canonical": CONFIG_IS_CANONICAL,
            "config": PIPELINE_CONFIG_META,
            "planner": PLANNER_MODE,
            "planner_config": asdict(CENTERED_CONFIG),
            "min_accepted_smoothing_blend": MIN_ACCEPTED_SMOOTHING_BLEND,
            "path_smoothing": asdict(SMOOTHING_CONFIG),
            "robot_radius_m": ROBOT_RADIUS_M,
            "planning_margin_m": PLANNING_MARGIN_M,
            "planning_radius_m": PLANNING_RADIUS_M,
            "goal_room_margin_m": GOAL_ROOM_MARGIN_M,
            "goal_room_tail_m": GOAL_ROOM_TAIL_M,
            "goal_terminal_mode": GOAL_TERMINAL_MODE,
            "goal_approach_m": GOAL_APPROACH_M,
            "execution_replay_step_m": REPLAY_STEP_M,
            "seed": SEED,
            "episodes_per_scene": EPISODES_PER_SCENE,
            "instructions_per_episode": (1 if CAPABILITY_PROFILE in CURRICULUM_OPTION_PROFILES else 4),
            "instruction_types": (["Maneuver-based"] if CAPABILITY_PROFILE in CURRICULUM_OPTION_PROFILES else list(INSTRUCTION_TYPES)),
            "capability_profile": dict(_CAPABILITY),
            "goal_visibility": {
                "camera": {"width": CAMERA_WIDTH, "height": CAMERA_HEIGHT_PX,
                           "hfov_deg": CAMERA_HFOV_DEG, "height_m": CAMERA_Z_M},
                "min_bbox_width_px": GOAL_MIN_BBOX_WIDTH_PX,
                "min_bbox_height_px": GOAL_MIN_BBOX_HEIGHT_PX,
                "min_projected_area_ratio": GOAL_MIN_PROJECTED_AREA_RATIO,
                "min_depth_area_ratio": GOAL_MIN_DEPTH_AREA_RATIO,
                "max_natural_bearing_error_deg": GOAL_MAX_NATURAL_BEARING_ERROR_DEG,
                "semantic_mode": "blind_localization",
                "required_views": 2,
                "confirmation_offset_m": GOAL_CONFIRMATION_OFFSET_M,
                "min_detection_width_px": GOAL_MIN_DETECTION_WIDTH_PX,
                "min_detection_height_px": GOAL_MIN_DETECTION_HEIGHT_PX,
                "min_detection_area_ratio": GOAL_MIN_DETECTION_AREA_RATIO,
                "min_detection_depth_fraction": GOAL_MIN_DETECTION_DEPTH_FRACTION,
                "target_class_whitelist": None,
            },
        },
        "scenes": [{"scene_id": "demo_%s" % bare, "scene_name": bare,
                    "samples": samples}],
    }
    tmp_path = "%s.tmp.%d" % (out_path, os.getpid())
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    print("  [D] staged %d episodes -> %s" % (len(samples), tmp_path))
    return {"scene": bare, "ok": True, "n": len(samples),
            "tmp_path": tmp_path, "out_path": out_path}


def main():
    print("SAGE-3D demo episode builder | scenes=%s | planner=%s | terminal=%s | hfov=%.1f | canonical=%s" %
          (SCENES, PLANNER_MODE, GOAL_TERMINAL_MODE, CAMERA_HFOV_DEG,
           CONFIG_IS_CANONICAL))
    print("config=%s sha256=%s overrides=%s" % (
        PIPELINE_CONFIG_META["config_path"],
        PIPELINE_CONFIG_META["config_sha256"],
        PIPELINE_CONFIG_META["overrides"] or "none"))
    os.makedirs(DEMO_ROOT, exist_ok=True)
    os.makedirs(OUT_VLN, exist_ok=True)
    results = []
    for bare in SCENES:
        try:
            results.append(build_scene(bare))
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"scene": bare, "ok": False, "err": repr(e)})
    print("\n=== summary ===")
    for r in results:
        print("  %s: %s" % (r["scene"],
              "ok n=%d" % r.get("n", 0) if r.get("ok") else "FAIL %s" % r.get("err", "")))
    successful = [r for r in results if r.get("ok")]
    ok = (len(successful) == len(results) if REQUIRE_ALL_SCENES
          else bool(successful))
    if ok:
        for r in successful:
            os.replace(r["tmp_path"], r["out_path"])
            print("  [commit] %s" % r["out_path"])
        print("%s -- %d/%d complete scenes committed" % (
            "ALL OK" if len(successful) == len(results) else "PARTIAL OK",
            len(successful), len(results)))
    else:
        for r in results:
            tmp_path = r.get("tmp_path")
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        print("SOME FAILED -- staged files removed; previous batch preserved")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
