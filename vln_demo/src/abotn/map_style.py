"""Light, model-readable styling for ABotN top-down maps only."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np

# Pale blue-green outside, white navigable floor and a thin teal boundary.
CANVAS = (229, 240, 239)
WALKABLE = (253, 254, 250)
BOUNDARY = (62, 99, 104)

# Navigation overlays intentionally match the original console: a light-blue
# trail, yellow one-metre heading ruler, blue arrow and simple green goal ring.
TRAIL = (90, 190, 255)
RULER = (255, 220, 0)
AGENT = (55, 105, 200)
GOAL = (120, 255, 140)


def _inner_boundary(mask: np.ndarray) -> np.ndarray:
    """Return the one-pixel boundary inside a boolean region."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("walkability mask must be two-dimensional")
    interior = mask.copy()
    interior[1:, :] &= mask[:-1, :]
    interior[:-1, :] &= mask[1:, :]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    return mask & ~interior


def walkability_rgb(free: np.ndarray) -> np.ndarray:
    """Render an ABotN walkability mask with clear region boundaries."""
    free = np.asarray(free, dtype=bool)
    rgb = np.empty(free.shape + (3,), dtype=np.uint8)
    rgb[:] = CANVAS
    rgb[free] = WALKABLE
    rgb[_inner_boundary(free)] = BOUNDARY
    return rgb


def _point(point: Sequence[float]) -> tuple[float, float]:
    return float(point[0]), float(point[1])


def draw_agent_arrow(draw, center: Sequence[float], heading_tip: Sequence[float],
                     radius: float = 8.0) -> None:
    """Draw the original simple blue agent arrow inside a small circle."""
    cx, cy = _point(center)
    tx, ty = _point(heading_tip)
    dx, dy = tx - cx, ty - cy
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux

    ring = radius + 1.0
    draw.ellipse([cx - ring, cy - ring, cx + ring, cy + ring],
                 fill=(238, 240, 242), outline=(80, 86, 94), width=1)
    front = (cx + ux * radius, cy + uy * radius)
    left = (cx - ux * radius * 0.68 + px * radius * 0.62,
            cy - uy * radius * 0.68 + py * radius * 0.62)
    notch = (cx - ux * radius * 0.20, cy - uy * radius * 0.20)
    right = (cx - ux * radius * 0.68 - px * radius * 0.62,
             cy - uy * radius * 0.68 - py * radius * 0.62)
    draw.polygon([front, left, notch, right], fill=AGENT)


def draw_navigation_overlay(draw, trail: Iterable[Sequence[float]],
                            center: Sequence[float], heading_tip: Sequence[float],
                            goal: Optional[Sequence[float]] = None) -> None:
    """Draw the trajectory, optional goal, one-metre ruler and agent."""
    points = [_point(point) for point in trail]
    if len(points) > 1:
        draw.line(points, fill=TRAIL, width=2)

    if goal is not None:
        gx, gy = _point(goal)
        radius = 7
        draw.ellipse([gx - radius, gy - radius, gx + radius, gy + radius],
                     outline=GOAL, width=3)

    cx, cy = _point(center)
    tx, ty = _point(heading_tip)
    draw.line([(cx, cy), (tx, ty)], fill=RULER, width=3)
    draw_agent_arrow(draw, (cx, cy), (tx, ty))
