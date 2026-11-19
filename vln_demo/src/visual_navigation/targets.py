"""Image-coordinate validation, metric-depth sampling, and projection."""

from __future__ import annotations

import math

import numpy as np


MAX_WAYPOINT_M = 10.0
MAX_VALID_DEPTH_M = 20.0


def _numbers(value, size: int, field: str):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{field} 必须是长度为 {size} 的数组")
    output = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"{field} 只能包含 0~1000 的数字")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} 只能包含 0~1000 的数字") from exc
        if not math.isfinite(number) or not 0 <= number <= 1000:
            raise ValueError(f"{field} 坐标必须在 0~1000 内")
        output.append(number)
    return output


def parse_point_2d(value):
    return _numbers(value, 2, "point_2d")


def parse_bbox_2d(value):
    box = _numbers(value, 4, "bbox_2d")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("bbox_2d 必须满足 x2>x1 且 y2>y1")
    return box


def depth_image(obs):
    """Return same-frame metric depth after shape validation."""
    if not isinstance(obs, dict) or obs.get("depth") is None:
        raise ValueError("当前 observation 没有同帧 depth；该视觉移动工具不能安全执行")
    depth = np.asarray(obs["depth"], dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"depth 应为 H×W，实际 shape={depth.shape}")
    rgb = obs.get("rgb")
    if rgb is not None and np.asarray(rgb).shape[:2] != depth.shape:
        raise ValueError("RGB 与 depth 分辨率不一致，不能按图像坐标安全反投影")
    return depth


def _pixel(value, extent):
    return int(np.clip(round(float(value) * (extent - 1) / 1000.0),
                       0, extent - 1))


def point_sample(obs, point_2d):
    """Read the selected pixel's exact metric depth.

    This intentionally performs no ground, normal, occupancy, or navigability
    classification. A point is rejected only when its coordinates are invalid
    or the selected pixel has no usable same-frame metric depth.
    """
    depth = depth_image(obs)
    height, width = depth.shape
    x, y = parse_point_2d(point_2d)
    u, v = _pixel(x, width), _pixel(y, height)
    value = float(depth[v, u])
    if not math.isfinite(value) or value <= 1e-4 or value > MAX_VALID_DEPTH_M:
        raise ValueError(f"point_2d 对应像素 ({u},{v}) 没有有效 metric depth")
    return u, v, value, (width, height)


def bbox_sample(obs, bbox_2d):
    """Sample robust metric depth around a bbox's bottom centre."""
    depth = depth_image(obs)
    height, width = depth.shape
    x1, y1, x2, y2 = parse_bbox_2d(bbox_2d)
    left, top = _pixel(x1, width), _pixel(y1, height)
    right = max(left + 1, min(width, int(math.ceil(x2 * width / 1000.0))))
    bottom = max(top + 1, min(height, int(math.ceil(y2 * height / 1000.0))))
    u = _pixel((x1 + x2) / 2.0, width)
    v = _pixel(y2, height)

    radius = 5
    local = depth[max(0, v-radius):min(height, v+radius),
                  max(0, u-radius):min(width, u+radius)]
    valid = local[np.isfinite(local) & (local > 1e-4)
                  & (local <= MAX_VALID_DEPTH_M)]
    if not valid.size:
        strip_height = max(1, int(math.ceil((bottom-top) * 0.20)))
        lower = depth[max(top, bottom-strip_height):bottom, left:right]
        valid = lower[np.isfinite(lower) & (lower > 1e-4)
                      & (lower <= MAX_VALID_DEPTH_M)]
    if not valid.size:
        raise ValueError("bbox_2d 底边中心及底部区域没有有效 metric depth")
    return u, v, float(np.median(valid)), (width, height)


def camera_hfov(env, default=120.0):
    for owner in (getattr(env, "client", None),
                  getattr(getattr(env, "sim", None), "client", None)):
        camera = getattr(owner, "camera", None)
        if camera is not None:
            value = getattr(camera, "hfov_deg", getattr(camera, "hfov", None))
            if value is not None:
                return float(value)
    geom = getattr(env, "_geom", None)
    if isinstance(geom, dict) and geom.get("hfov") is not None:
        return float(geom["hfov"])
    return float(default)


def camera_world_y(env):
    """Return the current optical-centre height in habitat's Y-up frame."""
    state = env.sim.get_agent_state()
    sensors = getattr(state, "sensor_states", {}) or {}
    for key in ("depth", "depth_sensor", "rgb", "rgb_sensor"):
        sensor = sensors.get(key)
        if sensor is not None and getattr(sensor, "position", None) is not None:
            return float(sensor.position[1])
    if getattr(env.sim, "camera_z", None) is not None:
        return float(env.sim.camera_z)
    if (getattr(env.sim, "nav", None) is not None
            and getattr(env.sim, "sensor_height", None) is not None):
        return float(state.position[1]) + float(env.sim.sensor_height)
    return float(state.position[1])


def project_world_point(env, world_xyz, image_size):
    """Project a habitat-frame world point into the current camera view."""
    import agent_tools as agent

    width, height = image_size
    x, y, z = (float(value) for value in world_xyz)
    x0, z0 = agent._xz(env)
    yaw = math.radians(agent._yaw(env))
    dx, dz = x-x0, z-z0
    right = dx*math.cos(yaw) + dz*math.sin(yaw)
    forward = dx*math.sin(yaw) - dz*math.cos(yaw)
    if forward <= 1e-4:
        return None
    focal = width / (2.0 * math.tan(math.radians(camera_hfov(env)) / 2.0))
    u = (width-1)/2.0 + focal*right/forward
    v = (height-1)/2.0 + focal*(camera_world_y(env)-y)/forward
    return float(u), float(v), float(forward)


def local_waypoint(u, v, depth_m, image_size, hfov_deg):
    """Return camera-local (right, down, forward) coordinates in metres."""
    width, height = image_size
    focal = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    x = (float(u) - (width - 1) / 2.0) * depth_m / focal
    y = (float(v) - (height - 1) / 2.0) * depth_m / focal
    return x, y, float(depth_m)


def target_habitat_xz(env, local_right, local_forward, standoff_m=0.0):
    """Convert a camera-local target into habitat-shaped world X/Z."""
    import agent_tools as agent

    planar = math.hypot(local_right, local_forward)
    distance = max(0.0, min(MAX_WAYPOINT_M, planar-standoff_m))
    if planar <= 1e-6:
        return agent._xz(env), distance
    scale = distance / planar
    right, forward = local_right*scale, local_forward*scale
    x0, z0 = agent._xz(env)
    yaw = math.radians(agent._yaw(env))
    return (x0 + right*math.cos(yaw) + forward*math.sin(yaw),
            z0 + right*math.sin(yaw) - forward*math.cos(yaw)), distance
