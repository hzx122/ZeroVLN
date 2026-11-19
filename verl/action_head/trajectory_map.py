"""Deterministic, non-oracle trajectory-map context for the v6 harness.

The state is reconstructed only from model-visible successful execution
results.  It deliberately contains no scene map, obstacle map, goal position,
reference trajectory, or evaluator-only pose.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


TRAJECTORY_MAP_SCHEMA = "actual_trajectory_map_v1"
TRAJECTORY_MAP_SIZE = 448
_HALF_EXTENTS_M = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
_RENDERER_CONFIG = {
    "schema": TRAJECTORY_MAP_SCHEMA,
    "renderer_revision": "20260921_r2",
    "size": TRAJECTORY_MAP_SIZE,
    "half_extents_m": _HALF_EXTENTS_M,
    "coordinates": "start_relative_forward_left",
    "content": "actual_execution_only",
}
TRAJECTORY_MAP_RENDERER_SHA256 = hashlib.sha256(
    json.dumps(_RENDERER_CONFIG, sort_keys=True).encode("utf-8")
).hexdigest()

MAP_CONTEXT_TEXT = (
    "[Latest actual trajectory map]\n"
    "This start-relative top-down trace contains actual executed motion only. "
    "S is the start, C is the current position, and the arrow is the current "
    "heading. Up is the starting forward direction and left is the starting "
    "left direction."
)
CURRENT_VIEW_TEXT = "[Current first-person view]"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


@dataclass
class TrajectoryMapState:
    """Cumulative actual pose in the episode-start coordinate frame."""

    forward_m: float = 0.0
    left_m: float = 0.0
    heading_deg: float = 0.0
    points: list[tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0)])
    blocked_points: list[tuple[float, float]] = field(default_factory=list)
    collision_points: list[tuple[float, float]] = field(default_factory=list)

    def update_action_head(self, frames: Iterable[dict[str, Any]]) -> None:
        """Compose option-relative cumulative chunk poses into episode frame."""
        frame_list = list(frames or ())
        if not frame_list:
            return
        start_forward = self.forward_m
        start_left = self.left_m
        start_heading = self.heading_deg
        theta = math.radians(start_heading)
        last_pose = None
        for frame in frame_list:
            pose = frame.get("actual_end_pose") or {}
            x_m, y_m = _finite(pose.get("x_m")), _finite(pose.get("y_m"))
            yaw_deg = _finite(pose.get("yaw_deg"))
            if x_m is None or y_m is None:
                continue
            forward = start_forward + math.cos(theta) * x_m - math.sin(theta) * y_m
            left = start_left + math.sin(theta) * x_m + math.cos(theta) * y_m
            point = (forward, left)
            if point != self.points[-1]:
                self.points.append(point)
            if bool(frame.get("blocked")):
                self.blocked_points.append(point)
            if bool(frame.get("collision")):
                self.collision_points.append(point)
            self.forward_m, self.left_m = point
            last_pose = yaw_deg
        if last_pose is not None:
            self.heading_deg = _wrap_degrees(start_heading + last_pose)

    def update_turn(self, raw: dict[str, Any] | None) -> None:
        """Apply only a successfully executed in-place turn."""
        raw = raw or {}
        if raw.get("ok") is not True:
            return
        turned = _finite(raw.get("turned_deg"))
        direction = raw.get("direction")
        if turned is None or direction not in ("left", "right"):
            return
        signed = turned if direction == "left" else -turned
        self.heading_deg = _wrap_degrees(self.heading_deg + signed)

    def render(self) -> Image.Image:
        return render_trajectory_map(self)


def _wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _half_extent(state: TrajectoryMapState) -> float:
    coordinates = [abs(v) for point in state.points for v in point]
    maximum = max(coordinates or [0.0]) + 0.6
    return next((value for value in _HALF_EXTENTS_M if value >= maximum),
                _HALF_EXTENTS_M[-1])


def render_trajectory_map(state: TrajectoryMapState) -> Image.Image:
    size = TRAJECTORY_MAP_SIZE
    margin = 38
    half = _half_extent(state)
    pixels_per_m = (size - 2 * margin) / (2.0 * half)
    image = Image.new("RGB", (size, size), (248, 249, 251))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    def pixel(point: tuple[float, float]) -> tuple[float, float]:
        forward, left = point
        return (size / 2.0 - left * pixels_per_m,
                size / 2.0 - forward * pixels_per_m)

    grid_step = max(1.0, half / 4.0)
    grid = int(half // grid_step)
    for index in range(-grid, grid + 1):
        delta = index * grid_step * pixels_per_m
        draw.line((margin, size / 2 + delta, size - margin, size / 2 + delta),
                  fill=(222, 226, 232), width=1)
        draw.line((size / 2 + delta, margin, size / 2 + delta, size - margin),
                  fill=(222, 226, 232), width=1)

    path = [pixel(point) for point in state.points]
    if len(path) > 1:
        draw.line(path, fill=(35, 102, 190), width=5, joint="curve")
    start = pixel((0.0, 0.0))
    current = pixel((state.forward_m, state.left_m))
    at_start = math.hypot(state.forward_m, state.left_m) < 0.05
    if at_start:
        draw.ellipse((start[0] - 11, start[1] - 11, start[0] + 11, start[1] + 11),
                     fill=(37, 155, 87), outline=(20, 92, 51), width=2)
        draw.ellipse((start[0] - 6, start[1] - 6, start[0] + 6, start[1] + 6),
                     fill=(35, 102, 190), outline=(18, 57, 110), width=2)
        draw.text((start[0] + 13, start[1] + 2), "S/C",
                  fill=(18, 57, 110), font=font)
    else:
        draw.ellipse((start[0] - 8, start[1] - 8, start[0] + 8, start[1] + 8),
                     fill=(37, 155, 87), outline=(20, 92, 51), width=2)
        draw.text((start[0] + 10, start[1] + 2), "S",
                  fill=(20, 92, 51), font=font)
        draw.ellipse((current[0] - 9, current[1] - 9,
                      current[0] + 9, current[1] + 9),
                     fill=(35, 102, 190), outline=(18, 57, 110), width=2)
        draw.text((current[0] + 11, current[1] + 2), "C",
                  fill=(18, 57, 110), font=font)
    angle = math.radians(state.heading_deg)
    arrow = (current[0] - math.sin(angle) * 27,
             current[1] - math.cos(angle) * 27)
    draw.line((current[0], current[1], arrow[0], arrow[1]),
              fill=(18, 57, 110), width=5)

    # Execution flags are drawn last so they remain visible at the current pose.
    for point in state.blocked_points:
        x, y = pixel(point)
        draw.rectangle((x - 7, y - 7, x + 7, y + 7),
                       outline=(225, 145, 32), width=3)
    for point in state.collision_points:
        x, y = pixel(point)
        draw.line((x - 10, y - 10, x + 10, y + 10),
                  fill=(196, 49, 49), width=3)
        draw.line((x - 10, y + 10, x + 10, y - 10),
                  fill=(196, 49, 49), width=3)

    draw.text((12, 10), "ACTUAL TRAJECTORY  |  start frame",
              fill=(35, 39, 47), font=font)
    draw.text((12, size - 28), f"half-width {half:g} m",
              fill=(70, 75, 85), font=font)
    return image


def inject_trajectory_map(messages: Iterable[dict[str, Any]],
                          state: TrajectoryMapState) -> list[dict[str, Any]]:
    """Insert exactly one latest map immediately before the current RGB.

    Call this on a freshly stripped canonical history.  The canonical history
    is never mutated, so previous maps cannot accumulate across decisions.
    """
    copied = copy.deepcopy(list(messages))
    image_positions: list[tuple[int, int]] = []
    for message_index, message in enumerate(copied):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part_index, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") in ("image", "image_url"):
                image_positions.append((message_index, part_index))
    if not image_positions:
        raise ValueError("v6 trajectory-map injection requires a current RGB image")
    message_index, part_index = image_positions[-1]
    content = copied[message_index]["content"]
    content[part_index:part_index] = [
        {"type": "text", "text": MAP_CONTEXT_TEXT},
        {"type": "image", "image": state.render()},
        {"type": "text", "text": CURRENT_VIEW_TEXT},
    ]
    return copied
