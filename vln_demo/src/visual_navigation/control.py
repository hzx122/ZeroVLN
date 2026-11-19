"""Motion controllers that follow planned visual-target paths."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from types import SimpleNamespace

import numpy as np

from .planning import plan_path


ANIMATION_STEP_M = 0.10
ANIMATION_TURN_DEG = 5.0
MAX_CONTROLLER_STEPS = 500
MAX_BLOCKED_REPLANS = 3
GOAL_TOLERANCE_M = 0.01

_UNILAVIRA_FOLLOWER_CLASS = None


def _unilavira_follower():
    """Instantiate Uni-LaViRA's vendored pure-pursuit controller."""
    global _UNILAVIRA_FOLLOWER_CLASS
    if _UNILAVIRA_FOLLOWER_CLASS is None:
        root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "third_party",
            "uni-lavira", "real-world-code", "cobot_magic"))
        config_path = os.path.join(root, "config.py")
        utils_path = os.path.join(root, "utils.py")
        if not os.path.isfile(config_path) or not os.path.isfile(utils_path):
            raise ValueError("找不到 Uni-LaViRA Cobot 控制器源码")

        config_spec = importlib.util.spec_from_file_location(
            "_vln_unilavira_config", config_path)
        config_module = importlib.util.module_from_spec(config_spec)
        config_spec.loader.exec_module(config_module)

        sentinel = object()
        previous = sys.modules.get("config", sentinel)
        sys.modules["config"] = config_module
        try:
            utils_spec = importlib.util.spec_from_file_location(
                "_vln_unilavira_utils", utils_path)
            utils_module = importlib.util.module_from_spec(utils_spec)
            utils_spec.loader.exec_module(utils_module)
        finally:
            if previous is sentinel:
                sys.modules.pop("config", None)
            else:
                sys.modules["config"] = previous
        _UNILAVIRA_FOLLOWER_CLASS = utils_module.TrajectoryFollower
    return _UNILAVIRA_FOLLOWER_CLASS()


def follow_path(env, obs, path, on_step, abort):
    """Follow planner short-term goals with smoothly rendered primitives."""
    import agent_tools as agent

    current, moved, steps, turned = obs, 0.0, 0, 0.0
    blocked, reason = False, "ok"
    for waypoint in path[1:]:
        x0, z0 = agent._xz(env)
        dx, dz = float(waypoint[0])-x0, float(waypoint[1])-z0
        distance = math.hypot(dx, dz)
        if distance < agent.MIN_FWD_M:
            continue
        desired = math.degrees(math.atan2(dx, -dz))
        delta = agent.wrap180(desired-agent._yaw(env))
        while abs(delta) >= agent.MIN_TURN_DEG:
            if abort is not None and abort():
                return current, moved, steps, turned, False, "aborted"
            amount = min(ANIMATION_TURN_DEG, abs(delta))
            direction = "right" if delta > 0 else "left"
            next_obs, info = agent.drive_turn(env, direction, amount, on_step)
            current = next_obs or current
            actual = float(info["turned_deg"])
            turned += actual
            steps += int(info["n_steps"])
            if actual < amount*0.95:
                return current, moved, steps, turned, True, "turn_incomplete"
            delta = agent.wrap180(desired-agent._yaw(env))
        remaining = math.hypot(float(waypoint[0])-agent._xz(env)[0],
                               float(waypoint[1])-agent._xz(env)[1])
        while remaining >= agent.MIN_FWD_M:
            if abort is not None and abort():
                return current, moved, steps, turned, False, "aborted"
            amount = min(ANIMATION_STEP_M, remaining)
            next_obs, info = agent.drive_forward(env, amount, on_step, abort)
            current = next_obs or current
            actual = float(info["moved_m"])
            moved += actual
            steps += int(info["n_steps"])
            if info["blocked"] or actual < amount*0.95:
                return current, moved, steps, turned, True, "blocked"
            remaining = math.hypot(float(waypoint[0])-agent._xz(env)[0],
                                   float(waypoint[1])-agent._xz(env)[1])
    return current, moved, steps, turned, blocked, reason


def path_length(path):
    return sum(math.hypot(b[0]-a[0], b[1]-a[1])
               for a, b in zip(path, path[1:]))


def habitat_target_y(env, local_down):
    """Projected world Y using the real depth/RGB sensor pose when present."""
    try:
        state = env.sim.get_agent_state()
        sensors = getattr(state, "sensor_states", {}) or {}
        sensor = (sensors.get("depth") or sensors.get("depth_sensor")
                  or sensors.get("rgb") or sensors.get("rgb_sensor"))
        if sensor is not None:
            return float(sensor.position[1])-float(local_down)
        return float(state.position[1])-float(local_down)
    except Exception:
        return None


def follow_embodied_habitat(env, obs, target_hab, target_y, radius,
                             on_step, abort):
    """Use Embodied-Navigator's ShortestPathFollower when available."""
    import agent_tools as agent

    try:
        from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
        from habitat.sims.habitat_simulator.actions import HabitatSimActions
    except ImportError:
        return None
    pathfinder = getattr(env.sim, "pathfinder", None)
    if pathfinder is None or not getattr(pathfinder, "is_loaded", True):
        return None

    state = env.sim.get_agent_state()
    proposed = np.array([
        target_hab[0],
        float(state.position[1]) if target_y is None else target_y,
        target_hab[1],
    ], dtype=np.float32)
    goal = np.asarray(pathfinder.snap_point(proposed), dtype=np.float32)
    if not np.isfinite(goal).all():
        raise ValueError("视觉目标无法 snap 到 Habitat navmesh")
    snapped = (float(goal[0]), float(goal[2]))
    initial_path, _ = plan_path(env, target_hab)
    if getattr(env.sim, "habitat_config", None) is None:
        env.sim.habitat_config = SimpleNamespace(scene=getattr(env, "scene", ""))
    follower = ShortestPathFollower(
        env.sim, goal_radius=float(radius), return_one_hot=False)
    action_names = {
        int(HabitatSimActions.move_forward): "move_forward",
        int(HabitatSimActions.turn_left): "turn_left",
        int(HabitatSimActions.turn_right): "turn_right",
    }
    stop_id = int(HabitatSimActions.stop)
    current, moved, steps, turned = obs, 0.0, 0, 0.0
    blocked, reason = False, "goal_radius"
    start = agent._xz(env)

    for _ in range(MAX_CONTROLLER_STEPS):
        if abort is not None and abort():
            reason = "aborted"
            break
        before, yaw0 = agent._xz(env), agent._yaw(env)
        if math.hypot(before[0]-snapped[0], before[1]-snapped[1]) <= radius:
            break
        action = follower.get_next_action(goal)
        if action is None or int(action) == stop_id:
            break
        name = action_names.get(int(action))
        if name is None:
            raise ValueError(f"ShortestPathFollower 返回未知动作 {action!r}")
        current = env.step(agent.action_id(env, name))
        steps += 1
        if on_step is not None:
            on_step(current, steps)
        after = agent._xz(env)
        if name == "move_forward":
            delta = math.hypot(after[0]-before[0], after[1]-before[1])
            moved += delta
            if delta < 1e-3:
                blocked, reason = True, "blocked"
                break
        else:
            turned += abs(agent.wrap180(agent._yaw(env)-yaw0))
    else:
        blocked, reason = True, "max_controller_steps"

    return {
        "obs": current, "start": start, "target": target_hab,
        "snapped": snapped, "moved": moved, "steps": steps,
        "turned": turned, "blocked": blocked, "reason": reason,
        "planner": "embodied_shortest_path_follower", "replans": steps,
        "initial_path": initial_path,
    }


def _global_path_to_robot(path, env):
    """Convert a habitat X/Z path into Uni-LaViRA (forward, left)."""
    import agent_tools as agent

    x0, z0 = agent._xz(env)
    yaw = math.radians(agent._yaw(env))
    points = []
    for x, z in path:
        dx, dz = float(x)-x0, float(z)-z0
        forward = dx*math.sin(yaw)-dz*math.cos(yaw)
        right = dx*math.cos(yaw)+dz*math.sin(yaw)
        points.append([forward, -right])
    return np.asarray(points, dtype=float)


def follow_replanning(env, obs, target_hab, radius, mode, on_step, abort):
    """Replan from the measured pose after every rendered motion increment."""
    import agent_tools as agent

    follower = _unilavira_follower() if mode == "bbox" else None
    current, moved, steps, turned = obs, 0.0, 0, 0.0
    blocked, reason, replans = False, "goal_radius", 0
    blocked_replans = 0
    start = agent._xz(env)
    snapped = initial_path = planner_name = None

    for _ in range(MAX_CONTROLLER_STEPS):
        if abort is not None and abort():
            reason = "aborted"
            break
        path, planner_name = plan_path(env, target_hab)
        replans += 1
        if initial_path is None:
            initial_path = list(path)
        snapped = tuple(path[-1])
        here = agent._xz(env)
        goal_error = math.hypot(snapped[0]-here[0], snapped[1]-here[1])
        if goal_error <= radius+GOAL_TOLERANCE_M:
            break
        if len(path) < 2:
            blocked, reason = True, "planner_no_progress"
            break

        waypoint = path[1]
        for candidate in path[1:]:
            if math.hypot(candidate[0]-here[0], candidate[1]-here[1]) >= 0.03:
                waypoint = candidate
                break
        dx, dz = float(waypoint[0])-here[0], float(waypoint[1])-here[1]
        desired = math.degrees(math.atan2(dx, -dz))
        heading_error = agent.wrap180(desired-agent._yaw(env))
        turn_cap = ANIMATION_TURN_DEG
        if follower is not None:
            local_path = _global_path_to_robot(path, env)
            linear, angular = follower.compute_velocity(local_path)
            if abs(angular) > 0.01:
                horizon = max(0.05, ANIMATION_STEP_M/max(abs(linear), 0.05))
                turn_cap = min(
                    ANIMATION_TURN_DEG,
                    max(agent.MIN_TURN_DEG,
                        math.degrees(abs(angular)*horizon)),
                )

        if abs(heading_error) >= agent.MIN_TURN_DEG:
            amount = min(turn_cap, abs(heading_error))
            direction = "right" if heading_error > 0 else "left"
            next_obs, info = agent.drive_turn(env, direction, amount, on_step)
            current = next_obs or current
            actual = float(info["turned_deg"])
            turned += actual
            steps += int(info["n_steps"])
            blocked_replans = blocked_replans+1 if actual < amount*0.95 else 0
        else:
            waypoint_error = math.hypot(dx, dz)
            amount = min(ANIMATION_STEP_M, waypoint_error,
                         max(0.0, goal_error-radius))
            if amount <= 1e-4:
                break
            next_obs, info = agent.drive_forward(
                env, amount, on_step, abort, min_m=0.0)
            current = next_obs or current
            actual = float(info["moved_m"])
            moved += actual
            steps += int(info["n_steps"])
            blocked_replans = (blocked_replans+1
                               if info["blocked"] or actual < amount*0.95 else 0)
        if blocked_replans >= MAX_BLOCKED_REPLANS:
            blocked, reason = True, "blocked_after_replanning"
            break
    else:
        blocked, reason = True, "max_controller_steps"

    return {
        "obs": current, "start": start, "target": target_hab,
        "snapped": snapped or target_hab, "moved": moved, "steps": steps,
        "turned": turned, "blocked": blocked, "reason": reason,
        "planner": (("unilavira_dynamic_" if mode == "bbox" else
                     "embodied_dynamic_")+str(planner_name)),
        "replans": replans, "initial_path": initial_path or [start],
    }
