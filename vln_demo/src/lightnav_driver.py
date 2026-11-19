# -*- coding: utf-8 -*-
"""LightNav-0 WebSocket transport and simulator-independent action adapter.

The model stays in the LightNav repository and process.  This module owns only
the public JSON protocol plus the narrow projection from LightNav's unicycle
velocity action to the continuous primitives shared by this console's
simulators.

LightNav's native Habitat policy executes one 0.1 s velocity-control step:

* select the first waypoint whose forward or yaw component is non-zero;
* ignore lateral displacement (the controller is a 2-DoF unicycle);
* clip forward motion to [0, 0.25] m and yaw to [-30, 30] degrees.

Our environments expose turn/forward primitives rather than a simultaneous
velocity command.  A symmetric turn-chord-turn decomposition preserves the
ideal constant-twist SE(2) endpoint.  Collision handling can still differ, so
the returned log retains every dropped/clipped component for parity audits.
"""
from __future__ import annotations

import base64
import io
import json
import math
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_URL = "ws://127.0.0.1:8050"
DEFAULT_DT = 0.1
DEFAULT_LINEAR_RANGE_MPS = (0.0, 2.5)
DEFAULT_ANGULAR_RANGE_DPS = (-300.0, 300.0)

# Keep the colours aligned with LightNav-0's own visualizer.  They are used
# only for human-facing previews; the image sent to the policy stays clean.
POINTING_COLOURS = {
    "apos": (128, 255, 214),
    "opos": (255, 96, 172),
}
TRAJECTORY_NEAR = (48, 176, 255)
TRAJECTORY_FAR = (214, 246, 255)
TRAJECTORY_RAIL = (236, 252, 255)
TRAJECTORY_WIDTH_FRAC = 0.25


@dataclass(frozen=True)
class Prediction:
    seq: int
    action_step: int
    waypoints: tuple[tuple[float, float, float], ...]
    stop: bool
    visible: bool | None
    latency_ms: float
    raw_text: str
    pointing: dict[str, Any] | None
    timings_ms: dict[str, Any] | None


@dataclass(frozen=True)
class ProjectedWaypoint:
    index: int
    source_forward_m: float
    source_lateral_m: float
    source_yaw_rad: float
    forward_m: float
    yaw_deg: float
    lateral_dropped_m: float
    forward_clipped_m: float
    yaw_clipped_deg: float


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be a finite number")
    return out


def _optional_bool(value: Any, name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool or null")
    return value


def parse_waypoints(value: Any) -> tuple[tuple[float, float, float], ...]:
    """Parse the documented ``actions`` payload and reject ambiguous shapes."""
    trajectories = value.get("actions") if isinstance(value, dict) else value
    if not isinstance(trajectories, (list, tuple)) or not trajectories:
        raise ValueError("actions.actions must be a non-empty waypoint list")
    if all(isinstance(v, (int, float)) and not isinstance(v, bool)
           for v in trajectories):
        if len(trajectories) % 3:
            raise ValueError("flat waypoint list length must be divisible by 3")
        trajectories = [trajectories[i:i + 3]
                        for i in range(0, len(trajectories), 3)]
    out = []
    for i, waypoint in enumerate(trajectories):
        if not isinstance(waypoint, (list, tuple)) or len(waypoint) != 3:
            raise ValueError(f"waypoint {i} must contain exactly 3 numbers")
        out.append(tuple(_finite_float(v, f"waypoint[{i}][{j}]")
                         for j, v in enumerate(waypoint)))
    return tuple(out)


def select_action_waypoint(
        waypoints: Iterable[Iterable[float]], atol: float = 1e-6,
) -> tuple[int, tuple[float, float, float]]:
    """Mirror ``lightnav.habitat.policy.select_action_waypoint`` exactly."""
    parsed = tuple(tuple(float(x) for x in waypoint) for waypoint in waypoints)
    if not parsed:
        raise ValueError("waypoints must not be empty")
    for i, waypoint in enumerate(parsed):
        if len(waypoint) != 3 or not all(math.isfinite(x) for x in waypoint):
            raise ValueError(f"waypoint {i} must contain 3 finite numbers")
        if abs(waypoint[0]) > atol or abs(waypoint[2]) > atol:
            return i, waypoint
    return 0, parsed[0]


def project_waypoint(
        waypoints: Iterable[Iterable[float]], *,
        dt: float = DEFAULT_DT,
        linear_range_mps: tuple[float, float] = DEFAULT_LINEAR_RANGE_MPS,
        angular_range_dps: tuple[float, float] = DEFAULT_ANGULAR_RANGE_DPS,
) -> ProjectedWaypoint:
    """Apply the native Habitat velocity limits to one LightNav prediction."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a positive finite number")
    if len(linear_range_mps) != 2 or len(angular_range_dps) != 2:
        raise ValueError("velocity ranges must be pairs")
    lin_lo, lin_hi = (_finite_float(v, "linear velocity range")
                      for v in linear_range_mps)
    ang_lo, ang_hi = (_finite_float(v, "angular velocity range")
                      for v in angular_range_dps)
    if lin_lo > lin_hi or ang_lo > ang_hi:
        raise ValueError("velocity range minimum exceeds maximum")

    index, waypoint = select_action_waypoint(waypoints)
    source_forward, source_lateral, source_yaw = waypoint
    applied_forward = min(lin_hi * dt, max(lin_lo * dt, source_forward))
    source_yaw_deg = math.degrees(source_yaw)
    applied_yaw_deg = min(ang_hi * dt, max(ang_lo * dt, source_yaw_deg))
    return ProjectedWaypoint(
        index=index,
        source_forward_m=source_forward,
        source_lateral_m=source_lateral,
        source_yaw_rad=source_yaw,
        forward_m=applied_forward,
        yaw_deg=applied_yaw_deg,
        lateral_dropped_m=source_lateral,
        forward_clipped_m=source_forward - applied_forward,
        yaw_clipped_deg=source_yaw_deg - applied_yaw_deg,
    )


def constant_twist_segments(
        forward_m: float, yaw_deg: float, *, min_turn_deg: float = 0.5,
) -> tuple[tuple[str, float], ...]:
    """Return turn/forward segments with the same ideal unicycle endpoint."""
    forward_m = _finite_float(forward_m, "forward_m")
    yaw_deg = _finite_float(yaw_deg, "yaw_deg")
    if forward_m < 0:
        raise ValueError("forward_m must be non-negative after projection")
    if min_turn_deg < 0:
        raise ValueError("min_turn_deg must be non-negative")
    if forward_m <= 1e-9:
        return (("turn", yaw_deg),) if abs(yaw_deg) >= min_turn_deg else ()
    if abs(yaw_deg) < min_turn_deg:
        return (("forward", forward_m),)

    half = yaw_deg / 2.0
    # Two sub-threshold half turns would silently lose the native rotation.
    if abs(half) < min_turn_deg:
        return (("forward", forward_m), ("turn", yaw_deg))
    theta = math.radians(yaw_deg)
    chord = 2.0 * forward_m * math.sin(theta / 2.0) / theta
    return (("turn", half), ("forward", chord), ("turn", half))


def execute_projected_waypoint(
        env: Any,
        projected: ProjectedWaypoint,
        tools: Any,
        *,
        on_step: Callable[[Any, int], None] | None = None,
        abort: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Execute one projected waypoint through ``agent_tools`` primitives."""
    segments = constant_twist_segments(
        projected.forward_m, projected.yaw_deg,
        min_turn_deg=float(tools.MIN_TURN_DEG),
    )
    result: dict[str, Any] = {
        "waypoint_index": projected.index,
        "source_forward_m": projected.source_forward_m,
        "source_lateral_m": projected.source_lateral_m,
        "source_yaw_rad": projected.source_yaw_rad,
        "forward_m": projected.forward_m,
        "yaw_deg": projected.yaw_deg,
        "lateral_dropped_m": projected.lateral_dropped_m,
        "forward_clipped_m": projected.forward_clipped_m,
        "yaw_clipped_deg": projected.yaw_clipped_deg,
        "segments": list(segments),
        "moved_m": 0.0,
        "turned_deg": 0.0,
        "n_steps": 0,
        "collided": False,
        "blocked": False,
        "terminated_mid": False,
        "observation": None,
    }

    callback_step = 0

    def step_callback(obs: Any, _local_step: int) -> None:
        nonlocal callback_step
        callback_step += 1
        result["observation"] = obs
        if on_step is not None:
            on_step(obs, callback_step)

    for kind, amount in segments:
        if getattr(env, "episode_over", False) or (abort is not None and abort()):
            result["terminated_mid"] = True
            break
        if kind == "turn":
            direction = "left" if amount > 0 else "right"
            obs, info = tools.drive_turn(
                env, direction, abs(amount), on_step=step_callback, abort=abort)
            result["turned_deg"] += float(info.get("turned_deg", 0.0))
        else:
            obs, info = tools.drive_forward(
                env, amount, on_step=step_callback, abort=abort, min_m=0.0)
            result["moved_m"] += float(info.get("moved_m", 0.0))
            result["collided"] = result["collided"] or bool(info.get("collided"))
            result["blocked"] = result["blocked"] or bool(info.get("blocked"))
        result["n_steps"] += int(info.get("n_steps", 0))
        if obs is not None:
            result["observation"] = obs
        if info.get("reason") in ("episode_over", "aborted"):
            result["terminated_mid"] = True
            break
    result["moved_m"] = round(result["moved_m"], 4)
    result["turned_deg"] = round(result["turned_deg"], 3)
    return result


def pointing_pixels(pointing: Any, target_size: tuple[int, int]) -> dict[str, tuple[int, int]]:
    """Return APOS/OPOS ``point`` states rescaled into ``target_size``.

    LightNav reports coordinates in the frame it received and includes that
    frame's ``[width, height]``.  Non-point states (rotation, stop, not visible)
    intentionally have no image location and are omitted.  Malformed optional
    display metadata fails soft instead of aborting an otherwise valid action.
    """
    if not isinstance(pointing, dict):
        return {}
    try:
        target_width, target_height = (int(target_size[0]), int(target_size[1]))
        source_size = pointing.get("frame_size")
        if (not isinstance(source_size, (list, tuple)) or len(source_size) != 2
                or isinstance(source_size[0], bool) or isinstance(source_size[1], bool)):
            return {}
        source_width, source_height = int(source_size[0]), int(source_size[1])
    except (TypeError, ValueError, IndexError):
        return {}
    if min(target_width, target_height, source_width, source_height) <= 0:
        return {}

    output: dict[str, tuple[int, int]] = {}
    for name in ("apos", "opos"):
        if pointing.get(f"{name}_state") != "point":
            continue
        value = pointing.get(f"{name}_px")
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            continue
        try:
            x = _finite_float(value[0], f"{name}_px[0]")
            y = _finite_float(value[1], f"{name}_px[1]")
        except ValueError:
            continue
        if not (0 <= x < source_width and 0 <= y < source_height):
            continue
        # Pixel-centre preserving scale, including one-pixel edge cases.
        u = round(x * (target_width - 1) / max(1, source_width - 1))
        v = round(y * (target_height - 1) / max(1, source_height - 1))
        output[name] = (u, v)
    return output


def draw_pointing_overlay(image: Image.Image, pointing: Any) -> Image.Image:
    """Draw LightNav APOS/OPOS on a copy of ``image`` for display only."""
    output = image.convert("RGB").copy()
    points = pointing_pixels(pointing, output.size)
    if not points:
        return output
    width, height = output.size
    draw = ImageDraw.Draw(output)
    stroke = max(2, min(width, height) // 180)
    radius = max(9, min(width, height) // 34)
    outline = (15, 18, 22)
    for name in ("apos", "opos"):
        if name not in points:
            continue
        u, v = points[name]
        colour = POINTING_COLOURS[name]
        draw.ellipse((u-radius, v-radius, u+radius, v+radius),
                     outline=outline, width=stroke+2)
        draw.ellipse((u-radius, v-radius, u+radius, v+radius),
                     outline=colour, width=stroke)
        draw.line((u-radius-4, v, u+radius+4, v), fill=outline, width=stroke+2)
        draw.line((u, v-radius-4, u, v+radius+4), fill=outline, width=stroke+2)
        draw.line((u-radius-4, v, u+radius+4, v), fill=colour, width=stroke)
        draw.line((u, v-radius-4, u, v+radius+4), fill=colour, width=stroke)
        label = name.upper()
        tx = (u+radius+6 if u+radius+52 < width
              else max(2, u-radius-49))
        ty = max(2, min(height-17, v-radius))
        draw.rectangle((tx-3, ty-2, tx+43, ty+13), fill=outline)
        draw.text((tx, ty), label, fill=colour)
    return output


def _trajectory_array(waypoints: Any) -> np.ndarray | None:
    try:
        array = np.asarray(waypoints, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if (array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 3
            or not np.all(np.isfinite(array[:, :3]))):
        return None
    return array[:, :3].copy()


def trajectory_bottom_edge_depth(
        image_size: tuple[int, int], hfov_deg: float, cam_height_m: float,
) -> float:
    """Native LightNav ``auto`` forward offset for keeping near poses visible."""
    width, height = image_size
    if min(width, height) <= 0 or cam_height_m <= 0:
        return 0.0
    focal = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    return 2.0 * focal * float(cam_height_m) / float(height)


def project_trajectory_pixels(
        waypoints: Any, image_size: tuple[int, int], *, hfov_deg: float,
        cam_height_m: float, forward_offset_m: float = 0.0,
) -> np.ndarray:
    """Project metric ego waypoints to pixels using LightNav's pinhole convention.

    Input rows are cumulative ``[forward_m, lateral_m, yaw_rad]`` poses with
    positive lateral pointing left.  Output is ``(H, 2)`` ``[u, v]``; poses at
    or behind the camera plane are NaN.  This mirrors
    ``lightnav.viz.projection.project_waypoints_to_image``.
    """
    array = _trajectory_array(waypoints)
    if array is None:
        return np.empty((0, 2), dtype=np.float64)
    width, height = (int(image_size[0]), int(image_size[1]))
    if min(width, height) <= 0:
        return np.full((array.shape[0], 2), np.nan, dtype=np.float64)
    hfov = _finite_float(hfov_deg, "hfov_deg")
    cam_height = _finite_float(cam_height_m, "cam_height_m")
    offset = _finite_float(forward_offset_m, "forward_offset_m")
    if not 0 < hfov < 180 or cam_height <= 0 or offset < 0:
        raise ValueError("trajectory projection requires valid hfov/camera height/offset")

    forward = array[:, 0].copy()
    lateral = array[:, 1].copy()
    yaw = array[:, 2]
    yaw_only = ((np.abs(forward) <= 0.05) & (np.abs(lateral) <= 0.05)
                & (np.abs(yaw) > 1e-8))
    for index in np.flatnonzero(yaw_only):
        if index:
            forward[index] = forward[index-1]
            lateral[index] = lateral[index-1]
    forward += offset
    valid = forward > 0.05
    safe = np.where(valid, forward, 1.0)
    focal = width / (2.0 * math.tan(math.radians(hfov) / 2.0))
    u = width/2.0 + focal*(-lateral/safe)
    v = height/2.0 + focal*(cam_height/safe)
    output = np.stack((u, v), axis=1)
    output[~valid] = np.nan
    return output


def draw_trajectory_overlay(
        image: Image.Image, waypoints: Any, *, hfov_deg: float,
        cam_height_m: float, forward_offset_m: float | None = None,
) -> Image.Image:
    """Draw the complete predicted chunk as a blue ground-plane ribbon."""
    output = image.convert("RGB").copy()
    array = _trajectory_array(waypoints)
    if array is None or array.shape[0] < 2:
        return output
    width, height = output.size
    offset = (trajectory_bottom_edge_depth(output.size, hfov_deg, cam_height_m)
              if forward_offset_m is None else float(forward_offset_m))
    path = np.vstack((np.zeros((1, 3), dtype=np.float64), array))
    half_width_m = (TRAJECTORY_WIDTH_FRAC * width/max(1, height)
                    * float(cam_height_m))
    left = path.copy()
    right = path.copy()
    left[:, 1] += half_width_m
    right[:, 1] -= half_width_m
    try:
        left_px = project_trajectory_pixels(
            left, output.size, hfov_deg=hfov_deg, cam_height_m=cam_height_m,
            forward_offset_m=offset)
        right_px = project_trajectory_pixels(
            right, output.size, hfov_deg=hfov_deg, cam_height_m=cam_height_m,
            forward_offset_m=offset)
        centres = project_trajectory_pixels(
            array, output.size, hfov_deg=hfov_deg, cam_height_m=cam_height_m,
            forward_offset_m=offset)
        heading_poses = array.copy()
        heading_poses[:, 0] += 0.10*np.cos(array[:, 2])
        heading_poses[:, 1] += 0.10*np.sin(array[:, 2])
        headings = project_trajectory_pixels(
            heading_poses, output.size, hfov_deg=hfov_deg,
            cam_height_m=cam_height_m, forward_offset_m=offset)
    except ValueError:
        return output

    overlay = Image.new("RGBA", output.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    limit = 8*max(width, height)

    def pixel(row):
        if not np.all(np.isfinite(row)):
            return None
        return (int(np.clip(round(row[0]), -limit, limit)),
                int(np.clip(round(row[1]), -limit, limit)))

    segments = []
    denom = max(1, len(path)-2)
    for index in range(len(path)-1):
        points = (pixel(left_px[index]), pixel(left_px[index+1]),
                  pixel(right_px[index+1]), pixel(right_px[index]))
        if any(point is None for point in points):
            continue
        t = index/denom
        colour = tuple(round(TRAJECTORY_NEAR[k]
                             + (TRAJECTORY_FAR[k]-TRAJECTORY_NEAR[k])*t)
                       for k in range(3))
        alpha = round(220*(1.0 if t <= 0.5 else max(0.0, 2.0*(1.0-t))**1.15))
        segments.append((points, colour, alpha))
    # Far first, near last, keeping the near corridor legible on overlap.
    for points, colour, alpha in reversed(segments):
        draw.polygon(points, fill=(*colour, alpha))
    rail_width = max(2, round(2*height/270.0))
    for edge in (left_px, right_px):
        points = [pixel(row) for row in edge]
        for index in range(len(points)-1):
            if points[index] is not None and points[index+1] is not None:
                draw.line((points[index], points[index+1]),
                          fill=(*TRAJECTORY_RAIL, 225), width=rail_width)
    node_radius = max(3, min(width, height)//100)
    for centre, heading in zip(centres, headings):
        p0, p1 = pixel(centre), pixel(heading)
        if p0 is not None and p1 is not None:
            draw.line((p0, p1), fill=(255, 255, 255, 235),
                      width=max(1, node_radius//2))
    for index, row in enumerate(centres, start=1):
        point = pixel(row)
        if point is None:
            continue
        u, v = point
        draw.ellipse((u-node_radius, v-node_radius, u+node_radius, v+node_radius),
                     fill=(*TRAJECTORY_NEAR, 235), outline=(255, 255, 255, 245),
                     width=max(1, node_radius//3))
    return Image.alpha_composite(output.convert("RGBA"), overlay).convert("RGB")


def draw_prediction_overlay(
        image: Image.Image, waypoints: Any, pointing: Any, *, hfov_deg: float,
        cam_height_m: float,
) -> Image.Image:
    """Compose trajectory first, then APOS/OPOS, without mutating ``image``."""
    trajectory = draw_trajectory_overlay(
        image, waypoints, hfov_deg=hfov_deg, cam_height_m=cam_height_m)
    return draw_pointing_overlay(trajectory, pointing)


class LightNavClient:
    """Persistent synchronous client for one LightNav episode."""

    def __init__(self, url: str = DEFAULT_URL, *, timeout: float = 30.0,
                 client_id: str | None = None, connector: Callable[..., Any] | None = None):
        self.url = str(url)
        self.timeout = float(timeout)
        self.client_id = client_id or f"closed_loop_{uuid.uuid4().hex[:8]}"
        self._connector = connector
        self._ws: Any = None
        self._seq = 0

    def connect(self) -> None:
        if self._ws is not None:
            return
        connector = self._connector
        if connector is None:
            try:
                import websocket  # websocket-client, not stdlib
            except ImportError as exc:
                raise RuntimeError(
                    "LightNav driver requires websocket-client (pip install websocket-client)"
                ) from exc
            connector = websocket.create_connection
        self._ws = connector(self.url, timeout=self.timeout, http_proxy_host=None)
        try:
            data = self._request("login", {"clientId": self.client_id})
            if data.get("msg") != "ok":
                raise RuntimeError(f"LightNav login failed: {data.get('msg', 'unknown error')}")
        except Exception:
            self.close()
            raise

    def reset(self) -> None:
        self.connect()
        data = self._request("reset", {})
        if data.get("msg") != "ok":
            raise RuntimeError(f"LightNav reset failed: {data.get('msg', 'unknown error')}")
        self._seq = 0

    def predict(self, image: Image.Image | np.ndarray | bytes,
                instruction: str) -> Prediction:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        self.connect()
        seq = self._seq
        self._seq += 1
        data = self._request("next", {
            "seq": seq,
            "image": self._encode_image(image),
            "instruction": instruction,
        })
        if data.get("seq") != seq:
            raise RuntimeError(
                f"LightNav response sequence mismatch: expected {seq}, got {data.get('seq')!r}")
        if data.get("actions") is None:
            raise RuntimeError(f"LightNav returned no actions: {data.get('msg', 'unknown error')}")
        actions_value = data["actions"]
        action_step = actions_value.get("step", 0) if isinstance(actions_value, dict) else 0
        try:
            action_step = int(action_step)
        except (TypeError, ValueError) as exc:
            raise ValueError("actions.step must be an integer") from exc
        pointing = data.get("pointing")
        if pointing is not None and not isinstance(pointing, dict):
            raise ValueError("pointing must be an object or null")
        timings = data.get("timings_ms")
        if timings is not None and not isinstance(timings, dict):
            raise ValueError("timings_ms must be an object or null")
        return Prediction(
            seq=seq,
            action_step=action_step,
            waypoints=parse_waypoints(actions_value),
            stop=bool(_optional_bool(data.get("stop"), "stop") or False),
            visible=_optional_bool(data.get("visible"), "visible"),
            latency_ms=_finite_float(data.get("latency_ms", 0.0), "latency_ms"),
            raw_text=str(data.get("raw_text") or ""),
            pointing=dict(pointing) if pointing is not None else None,
            timings_ms=dict(timings) if timings is not None else None,
        )

    def close(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def _request(self, action: str, data: dict[str, Any]) -> dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("LightNav WebSocket is not connected")
        self._ws.send(json.dumps({"action": action, "data": data}, ensure_ascii=False))
        try:
            response = json.loads(self._ws.recv())
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError("LightNav returned invalid JSON") from exc
        if not isinstance(response, dict) or response.get("action") != action:
            raise RuntimeError(f"LightNav returned an unexpected {action!r} response")
        payload = response.get("data")
        if not isinstance(payload, dict):
            raise RuntimeError("LightNav response data must be an object")
        if payload.get("rc") != 0:
            raise RuntimeError(
                f"LightNav {action} failed (rc={payload.get('rc')}): "
                f"{payload.get('msg', 'unknown error')}"
            )
        return payload

    @staticmethod
    def _encode_image(image: Image.Image | np.ndarray | bytes) -> str:
        if isinstance(image, bytes):
            if not image:
                raise ValueError("JPEG image bytes must not be empty")
            # The LightNav MuJoCo backend already receives the exact JPEG that
            # its native VlnClient would forward. Preserve it byte-for-byte.
            return base64.b64encode(image).decode("ascii")
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image, dtype=np.uint8))
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def __enter__(self) -> "LightNavClient":
        self.connect()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
