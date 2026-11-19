"""Compatibility entry point for depth-backed visual waypoint navigation.

The implementation is deliberately split by responsibility:

* :mod:`visual_navigation.targets` validates image coordinates and samples
  same-frame metric depth;
* :mod:`visual_navigation.planning` turns a projected target into a safe path;
* :mod:`visual_navigation.control` follows that path and publishes motion.

Point targets are not classified as ground. Any image pixel with valid
same-frame metric depth may be used; collision maps remain a controller concern
and may snap the projected target to a reachable position during planning.
"""

from __future__ import annotations

import math
from remote_habitat.policy import terminal_frame
from typing import Sequence

from visual_navigation.control import (
    ANIMATION_STEP_M,
    ANIMATION_TURN_DEG,
    GOAL_TOLERANCE_M,
    follow_embodied_habitat,
    follow_path,
    follow_replanning,
    habitat_target_y,
    path_length,
)
from visual_navigation.planning import (
    PLANNER_CLEARANCE_M,
    plan_habitat,
    plan_mall,
    plan_occupancy,
    plan_path,
    plan_rs3d,
)
from visual_navigation.targets import (
    MAX_VALID_DEPTH_M,
    MAX_WAYPOINT_M,
    bbox_sample,
    camera_hfov,
    camera_world_y,
    depth_image,
    local_waypoint,
    parse_bbox_2d,
    parse_point_2d,
    point_sample,
    project_world_point,
    target_habitat_xz,
)


POINT_GOAL_RADIUS_M = 0.20
BBOX_GOAL_RADIUS_M = 0.25

# Private aliases retained for focused controller/planner tests and existing
# internal imports. New code should import from the responsibility module.
_depth2d = depth_image
_target_habitat_xz = target_habitat_xz
_plan_occupancy = plan_occupancy
_plan_habitat = plan_habitat
_plan_mall = plan_mall
_plan_rs3d = plan_rs3d
_follow_path = follow_path
_habitat_target_y = habitat_target_y
_follow_embodied_habitat = follow_embodied_habitat
_follow_replanning = follow_replanning
_path_length = path_length


@terminal_frame
def execute(env, obs, mode: str, coords: Sequence[float], on_step=None,
            abort=None, on_target=None, on_invalid_point=None):
    """Project one visual target and run the matching map-aware controller."""
    import agent_tools as agent

    if mode == "point":
        try:
            u, v, depth_m, size = point_sample(obs, coords)
        except (TypeError, ValueError):
            if on_invalid_point is not None:
                on_invalid_point(coords)
            raise
        radius = POINT_GOAL_RADIUS_M
        tool = "move_to_point"
    elif mode == "bbox":
        u, v, depth_m, size = bbox_sample(obs, coords)
        radius = BBOX_GOAL_RADIUS_M
        tool = "move_to_bbox"
    else:
        raise ValueError(f"unknown visual target mode {mode!r}")

    local_x, local_y, local_z = local_waypoint(
        u, v, depth_m, size, camera_hfov(env))
    target_hab, requested = target_habitat_xz(env, local_x, local_z)
    target_y = camera_world_y(env)-local_y
    bearing = math.degrees(math.atan2(local_x, local_z))
    if on_target is not None:
        on_target({
            "kind": mode,
            "world_xyz": (float(target_hab[0]), float(target_y),
                          float(target_hab[1])),
            "goal_radius_m": float(radius),
            "source_pixel": (int(u), int(v)),
            "depth_m": float(depth_m),
        })

    if requested <= radius:
        start = agent._xz(env)
        control = {
            "obs": obs, "start": start, "target": target_hab,
            "snapped": start, "moved": 0.0, "steps": 0, "turned": 0.0,
            "blocked": False, "reason": "goal_radius",
            "planner": "already_at_goal", "replans": 0,
            "initial_path": [start],
        }
    else:
        control = None
        if mode == "point":
            control = follow_embodied_habitat(
                env, obs, target_hab, habitat_target_y(env, local_y),
                radius, on_step, abort)
        if control is None:
            control = follow_replanning(
                env, obs, target_hab, radius, mode, on_step, abort)

    current = control["obs"]
    moved, steps, turned = (
        control["moved"], control["steps"], control["turned"])
    blocked, reason, planner = (
        control["blocked"], control["reason"], control["planner"])
    path = control["initial_path"]
    snapped = tuple(control["snapped"])
    end = agent._xz(env)
    raw_snap_error = math.hypot(
        target_hab[0]-snapped[0], target_hab[1]-snapped[1])
    final_error = math.hypot(end[0]-snapped[0], end[1]-snapped[1])
    reached = final_error <= radius+GOAL_TOLERANCE_M
    if reached:
        reason = "goal_radius"
    elif reason == "goal_radius":
        reason = "controller_stopped_short"

    raw = {
        "tool": tool, "ok": reached, "reached": reached,
        "input_per_mille": [round(float(value), 3) for value in coords],
        "sample_pixel": [u, v], "depth_m": round(depth_m, 3),
        "local_waypoint_m": [round(local_x, 3), round(local_y, 3),
                             round(local_z, 3)],
        "world_target_xz": [round(target_hab[0], 3),
                            round(target_hab[1], 3)],
        "world_target_xyz": [round(target_hab[0], 3), round(target_y, 3),
                             round(target_hab[1], 3)],
        "snapped_target_xz": [round(snapped[0], 3), round(snapped[1], 3)],
        "end_xz": [round(end[0], 3), round(end[1], 3)],
        "snap_offset_m": round(raw_snap_error, 3),
        "final_goal_error_m": round(final_error, 3),
        "goal_radius_m": round(radius, 3),
        "bearing_deg": round(bearing, 2),
        "turned_deg": round(turned, 2),
        "requested_m": round(requested, 3),
        "moved_m": round(moved, 3),
        "planner": planner,
        "path_m": round(path_length(path), 3),
        "path_waypoints": len(path),
        "replans": int(control["replans"]),
        "n_steps": steps,
        "blocked": blocked,
        "reason": reason,
    }
    return current, raw
