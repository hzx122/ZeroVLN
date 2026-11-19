"""Occupancy-grid kinematics for ABotN scenes.

ABotN ships no navmesh.  What every scene *does* have is ``map/``::

    occ_map.png          uint8 greyscale, >=128 free, <128 obstacle
    occ_map_height.tiff  per-pixel ground height in metres -- but 8-bit, so
                         quantised to whole metres (see :mod:`abotn.floorfit`)
    occ_map_meta.txt     the four map corners in world metres + image size

so this module is to ABotN what ``rs3d.navmesh``/``rs3d.nav`` are to 3DRS: the
one place that answers "may I stand here" and "how far do I actually get".

World convention is ABotN's own: **z up**, x/y horizontal, yaw measured from +x
towards +y, camera at ``ground_z + 0.65``.  The conversion to habitat's y-up
frame happens in :mod:`abotn.env`, not here -- keeping this module in the same
frame as the annotations means the pixel math can be a transcription of
``abotn_evaluator/scene.py`` (``convert_actual_to_pixel`` and friends) rather
than a re-derivation, and a bug here shows up as a mismatch against ABotN's own
occ-map overlays.

numpy + pillow only: the console runs inside gordon's habitat venv, which has
no cv2, so the greyscale read and the footprint check are done by hand.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import numpy as np

from .map_style import walkability_rgb

FREE = 128  # ABotN's threshold: >=128 is free space


def parse_meta(text: str) -> dict:
    """Parse ``occ_map_meta.txt``.

    Transcribed from ``abotn_evaluator.scene._parse_meta_text``: the ranges are
    corner *distances*, so they stay positive whichever way the axes run, and
    the direction is recovered separately as the sx/sy signs below.
    """
    d = {}
    for line in text.strip().splitlines():
        line = line.strip()
        for tag, key in (("Top Left:", "TOP_LEFT"), ("Top Right:", "TOP_RIGHT"),
                         ("Bottom Left:", "BOTTOM_LEFT"),
                         ("Bottom Right:", "BOTTOM_RIGHT")):
            if line.startswith(tag):
                a, b = line.split(":", 1)[1].strip().strip("()").split(",")
                d[key + "_X"], d[key + "_Y"] = float(a), float(b)
        if line.startswith("Image size in pixels:"):
            w, h = line.split(":", 1)[1].strip().split(",")
            d["IMAGE_WIDTH"], d["IMAGE_HEIGHT"] = int(w), int(h)
    d["COORDINATE_RANGE_X"] = math.hypot(d["BOTTOM_RIGHT_X"] - d["BOTTOM_LEFT_X"],
                                         d["BOTTOM_RIGHT_Y"] - d["BOTTOM_LEFT_Y"])
    d["COORDINATE_RANGE_Y"] = math.hypot(d["BOTTOM_LEFT_X"] - d["TOP_LEFT_X"],
                                         d["BOTTOM_LEFT_Y"] - d["TOP_LEFT_Y"])
    return d


class OccMap:
    """One scene's walkability, in world metres."""

    def __init__(self, scene_dir: str):
        from PIL import Image

        map_dir = os.path.join(scene_dir, "map")
        self.scene_dir = scene_dir
        with open(os.path.join(map_dir, "occ_map_meta.txt"), encoding="utf-8") as f:
            self.meta = parse_meta(f.read())
        occ = Image.open(os.path.join(map_dir, "occ_map.png"))
        if occ.mode not in ("L", "I;16", "I"):
            occ = occ.convert("L")
        self.occ = np.asarray(occ, dtype=np.uint8)
        self.free = self.occ >= FREE
        self.h, self.w = self.occ.shape[:2]

        hp = os.path.join(map_dir, "occ_map_height.tiff")
        self.height = np.asarray(Image.open(hp), dtype=np.float32) \
            if os.path.exists(hp) else None
        # ABotN reads anything above this as a height-map hole and falls back to
        # 0.  Its own value, kept as-is: the 6 m and 20 m patches in two
        # POIBench scenes look like an elevated walkway but the Gaussian clouds
        # put the floor there at z ~= -0.4 like everywhere else, so they are
        # buildings and noise.  See ``abotn.env._occ_for``.
        self.hole_above = 5.0
        # Optional sub-metre correction to ``ground_z``, attached by
        # ``abotn.env`` -- the tiff is 8-bit, so it is quantised to whole
        # metres.  See :mod:`abotn.floorfit`.
        self.z_delta = None

        m = self.meta
        # which way world x/y run as the pixel column/row grows
        self.sx = 1.0 if (m["TOP_RIGHT_X"] - m["TOP_LEFT_X"]) >= 0 else -1.0
        self.sy = 1.0 if (m["BOTTOM_LEFT_Y"] - m["TOP_LEFT_Y"]) >= 0 else -1.0
        self.mpp_x = m["COORDINATE_RANGE_X"] / m["IMAGE_WIDTH"]
        self.mpp_y = m["COORDINATE_RANGE_Y"] / m["IMAGE_HEIGHT"]
        self.mpp = (self.mpp_x + self.mpp_y) / 2.0

    # -- coordinates -------------------------------------------------------
    def to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """World metres -> ``(col, row)``, ABotN's ``convert_actual_to_pixel``."""
        m = self.meta
        px = (x - m["TOP_LEFT_X"]) * self.sx / self.mpp_x
        py = (y - m["TOP_LEFT_Y"]) * self.sy / self.mpp_y
        return int(px), int(py)

    def to_world(self, px: float, py: float) -> Tuple[float, float]:
        m = self.meta
        return (m["TOP_LEFT_X"] + self.sx * px * self.mpp_x,
                m["TOP_LEFT_Y"] + self.sy * py * self.mpp_y)

    def ground_z(self, x: float, y: float, default: float = -0.1) -> float:
        """Ground height under ``(x, y)``.

        Outdoor scenes are not flat -- their trajectories sit anywhere from
        -0.8 m to several metres -- so the camera height has to be added to
        this rather than to a constant.

        Two details are ABotN's, not ours.  Its ``get_z_value`` scales by the
        *height map's* own pixel dimensions rather than the meta file's, which
        matters in the one scene where the tiff is a pixel narrower than the
        png.  And it reads anything above ``hole_above`` (5 m) as a hole and
        falls back to 0.

        On top of that comes ``z_delta`` if one is attached: the tiff is 8-bit
        greyscale, so ABotN's ground is quantised to whole metres, and the
        correction measured off the Gaussians (:mod:`abotn.floorfit`) is what
        keeps the camera from starting half a metre inside the pavement.  It
        applies to the hole fallback too -- a hole is exactly where the height
        map knows least.
        """
        if self.height is None:
            return default
        ih, iw = self.height.shape[0], self.height.shape[1]
        m = self.meta
        px = int((x - m["TOP_LEFT_X"]) * self.sx * iw / m["COORDINATE_RANGE_X"])
        py = int((y - m["TOP_LEFT_Y"]) * self.sy * ih / m["COORDINATE_RANGE_Y"])
        if not (0 <= px < iw and 0 <= py < ih):
            return default
        z = float(self.height[py, px])
        if not math.isfinite(z) or z > self.hole_above:
            z = 0.0
        if self.z_delta is not None:
            z += self.z_delta.at(x, y)
        return z

    # -- walkability -------------------------------------------------------
    def radius_px(self, radius_m: float) -> int:
        return max(0, int(round(radius_m / self.mpp)))

    def is_free(self, x: float, y: float, radius_m: float = 0.0) -> bool:
        """Is a square footprint of ``radius_m`` around ``(x, y)`` clear?

        Square, not round, because that is what ABotN's collision check uses --
        matching it matters more than the 20% of a pixel the corners cost.
        """
        px, py = self.to_pixel(x, y)
        if not (0 <= px < self.w and 0 <= py < self.h):
            return False
        r = self.radius_px(radius_m)
        if r == 0:
            return bool(self.free[py, px])
        patch = self.free[max(0, py - r):py + r + 1, max(0, px - r):px + r + 1]
        return bool(patch.size and patch.all())

    def seg_free(self, x0, y0, x1, y1, radius_m: float = 0.0) -> bool:
        for x, y in self._samples(x0, y0, x1, y1):
            if not self.is_free(x, y, radius_m):
                return False
        return True

    def _samples(self, x0, y0, x1, y1):
        """Points along a segment, one per half pixel (endpoint included)."""
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / (self.mpp * 0.5)) + 1)
        for i in range(1, n + 1):
            t = i / float(n)
            yield x0 + (x1 - x0) * t, y0 + (y1 - y0) * t

    def advance(self, x0, y0, x1, y1, radius_m: float = 0.0):
        """Furthest free point along the segment, starting from ``(x0, y0)``."""
        last = (x0, y0)
        for x, y in self._samples(x0, y0, x1, y1):
            if not self.is_free(x, y, radius_m):
                return last
            last = (x, y)
        return last

    def try_step(self, x0, y0, x1, y1, radius_m: float = 0.0, slide: bool = True):
        """Move towards ``(x1, y1)``; return ``(x, y, collided)``.

        Sliding is done by retrying the two axis components separately and
        keeping whichever gets further, which is the grid equivalent of
        habitat's wall slide: walking into a wall at an angle keeps the
        along-wall part of the motion instead of stopping dead.
        """
        if self.seg_free(x0, y0, x1, y1, radius_m):
            return x1, y1, False
        bx, by = self.advance(x0, y0, x1, y1, radius_m)
        best = (bx - x0) ** 2 + (by - y0) ** 2
        if slide:
            for cx, cy in ((x1, y0), (x0, y1)):
                ax, ay = self.advance(x0, y0, cx, cy, radius_m)
                d = (ax - x0) ** 2 + (ay - y0) ** 2
                if d > best:
                    bx, by, best = ax, ay, d
        return bx, by, True

    def sample_free(self, rng, radius_m: float = 0.2, tries: int = 400):
        """A random free point, biased towards open floor.

        Same shape as ``rs3d``'s spawn sampler: keep the roomiest of a bounded
        number of candidates rather than the first hit, so a spawn does not
        land in the one-pixel gap behind a sofa.
        """
        ys, xs = np.nonzero(self.free)
        if len(xs) == 0:
            return None
        best = None
        for _ in range(tries):
            i = int(rng.integers(len(xs)))
            x, y = self.to_world(float(xs[i]), float(ys[i]))
            r = self.clearance(x, y)
            if best is None or r > best[0]:
                best = (r, x, y)
            if r > 1.0:
                break
        if best is None or best[0] < radius_m:
            return None
        return best[1], best[2]

    def clearance(self, x: float, y: float, limit_m: float = 2.0) -> float:
        """Distance to the nearest obstacle, in metres, capped at ``limit_m``."""
        px, py = self.to_pixel(x, y)
        if not (0 <= px < self.w and 0 <= py < self.h) or not self.free[py, px]:
            return 0.0
        lim = self.radius_px(limit_m)
        for r in range(1, lim + 1):
            patch = self.free[max(0, py - r):py + r + 1, max(0, px - r):px + r + 1]
            if patch.size != (2 * r + 1) ** 2 or not patch.all():
                return (r - 1) * self.mpp
        return limit_m

    # -- visualisation -----------------------------------------------------
    def top_down(self, size: int = 512):
        """``(PIL image, to_px)`` -- free space light, obstacles dark.

        The maps are up to 4500 px on a side, so this downsamples to the widget
        size once and hands back the world->widget mapping; callers draw the
        trail and the agent on a copy.
        """
        from PIL import Image

        scale = min(size / float(self.w), size / float(self.h))
        tw, th = max(1, int(self.w * scale)), max(1, int(self.h * scale))
        mask = Image.fromarray(self.free.astype(np.uint8) * 255).resize(
            (tw, th), Image.NEAREST)
        img = Image.fromarray(walkability_rgb(np.asarray(mask) > 0))
        ox, oy = (size - tw) // 2, (size - th) // 2

        def to_px(x: float, y: float):
            px, py = self.to_pixel(x, y)
            return (ox + px * scale, oy + py * scale)

        return img, to_px, (size, size)


def load(scene_dir: str, cache: Optional[dict] = None) -> OccMap:
    if cache is None:
        return OccMap(scene_dir)
    if scene_dir not in cache:
        cache[scene_dir] = OccMap(scene_dir)
    return cache[scene_dir]
