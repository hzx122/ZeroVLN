"""Render client for SAGE-3D scenes -- the z-up euler route, port 7038.

Identical contract to :mod:`abotn.gs_client`: GVLN poses are already
``(x, y, z, roll, pitch, yaw)`` in a z-up world, so they go verbatim to the
server's ``/render_gs_xyz_euler`` and let it do its own COLMAP conversion. The
only differences are the default URL (a *separate* degree-3 server instance,
port 7038, so it can coexist with ABotN's 7037 and 3DRS's 7036) -- the ply
loader asserts the SH coefficient count, so the degree is per process.
"""

from __future__ import annotations

import os
import math
import base64
import io

import numpy as np

from abotn.gs_client import XYZEulerClient as _XYZEulerClient
from rs3d.gs_client import _to_image, rotmat2qvec

SAGE_RENDER_URL = os.environ.get("SAGE_RENDER_URL", "http://127.0.0.1:7038")
_USD_MASK_ROOT = os.environ.get(
    "SAGE_USD_COLLISION_MASK_ROOT",
    "/data/zerovln/SAGE-3D/"
    "sage_usd_collision_masks")

# --- optional tonemap on the returned RGB ---------------------------------
# InteriorGS gaussians render very bright/washed-out (scene mean ~211/255 on
# 841479): the right-of-frame glare and grazing-angle SH streaks are baked into
# the asset (verified: no gaussian prune removes them, MAE < 0.7/255 even at
# full SH deg3 -- see docs/实验记录.md#legacy-docs-20260921 2026-08-12). The one lever that
# helps the overall wash-out is a post-hoc exposure/gamma curve on the output
# RGB. We apply it *here*, in our own client, because the render server source
# lives on a read-only share and Isaac's own tonemap has no copyable spec.
#
#   out = clip(rgb/255 * EXP, 0, 1) ** GAMMA        (curve "A")
#
# Off by default (SAGE_TONEMAP=0) so nothing changes unless asked. Train and
# eval share the same client, so enabling it keeps observations consistent.
_TONEMAP = os.environ.get("SAGE_TONEMAP", "0") == "1"
_TM_EXP = float(os.environ.get("SAGE_TONEMAP_EXP", "0.72"))
_TM_GAMMA = float(os.environ.get("SAGE_TONEMAP_GAMMA", "0.9"))

# InteriorGS occasionally exposes a large smooth blue/purple sheet when the
# camera looks through an opening beyond the photographed part of a room. It
# is not the renderer background (the server background is black): depth is
# finite and SH0 reproduces it, so the bad colour is already in the Gaussian
# DC/base colour. Keep cleanup opt-in and conservative. The launchers enable
# it, while SAGE_RGB_OUTSIDE_CLEANUP=0 preserves the original byte path.
_OUTSIDE_CLEANUP = os.environ.get("SAGE_RGB_OUTSIDE_CLEANUP", "0") == "1"
_BLUE_EXCESS = float(os.environ.get("SAGE_RGB_BLUE_EXCESS", "16"))
_BLUE_MIN_FRACTION = float(os.environ.get("SAGE_RGB_BLUE_MIN_FRACTION", "0.04"))
_BLUE_MIN_HEIGHT = float(os.environ.get("SAGE_RGB_BLUE_MIN_HEIGHT", "0.45"))
_BLUE_STRENGTH = float(os.environ.get("SAGE_RGB_BLUE_STRENGTH", "0.98"))

# Geometry-based scene-domain clipping.  Unlike the legacy colour detector,
# this follows USD room/door topology and therefore also handles neutral or
# differently coloured leakage through windows and at the outer scene edge.
_USD_BOUNDARY_CLIP = os.environ.get("SAGE_USD_BOUNDARY_CLIP", "0") == "1"
_USD_BOUNDARY_MARGIN_M = float(os.environ.get(
    "SAGE_USD_BOUNDARY_MARGIN_M", "0.10"))
_USD_BOUNDARY_BLUE_ONLY = os.environ.get(
    "SAGE_USD_BOUNDARY_BLUE_ONLY", "1") == "1"
_USD_BOUNDARY_EXIT_M = float(os.environ.get(
    "SAGE_USD_BOUNDARY_EXIT_M", "0.25"))
_USD_BOUNDARY_MASK_SCALE = float(os.environ.get(
    "SAGE_USD_BOUNDARY_MASK_SCALE", "0.25"))
_USD_BOUNDARY_DEPTH_DOWNSAMPLE = int(os.environ.get(
    "SAGE_USD_BOUNDARY_DEPTH_DOWNSAMPLE", "4"))
_USD_BOUNDARY_LUMA_SCALE = float(os.environ.get(
    "SAGE_USD_BOUNDARY_LUMA_SCALE", "0.26"))
_USD_BOUNDARY_LUMA_FLOOR = float(os.environ.get(
    "SAGE_USD_BOUNDARY_LUMA_FLOOR", "18"))
_USD_PORTAL_NEAR_M = float(os.environ.get(
    "SAGE_USD_PORTAL_NEAR_M", "0.55"))
_USD_PORTAL_MIN_DEPTH_M = float(os.environ.get(
    "SAGE_USD_PORTAL_MIN_DEPTH_M", "0.55"))
_USD_PORTAL_STRONG_FRACTION = float(os.environ.get(
    "SAGE_USD_PORTAL_STRONG_FRACTION", "0.20"))
_USD_DOMAIN_CACHE = {}


def _apply_tonemap(img):
    """Exposure+gamma curve on a PIL RGB image; returns a PIL RGB image."""
    if img is None:
        return img
    from PIL import Image
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = np.clip(a * _TM_EXP, 0.0, 1.0) ** _TM_GAMMA
    return Image.fromarray((a * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def _outside_blue_alpha(img):
    """Return a feathered mask for a large enclosed blue artifact, if any."""
    if img is None:
        return None
    try:
        import cv2
    except Exception:
        return None

    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    red, green, blue = np.moveaxis(rgb, -1, 0)
    candidate = (((blue - np.maximum(red, green)) >= _BLUE_EXCESS)
                 & (blue >= 80.0)).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate, connectivity=8)
    if count <= 1:
        return None

    h, w = candidate.shape
    min_pixels = max(64, int(round(h * w * _BLUE_MIN_FRACTION)))
    keep = np.zeros_like(candidate)
    for component in range(1, count):
        x, y, width, height, area = stats[component]
        if area < min_pixels or height < h * _BLUE_MIN_HEIGHT:
            continue
        # Real outdoor sky normally reaches the frame edge. The failure mode
        # is a portal-sized component enclosed by a doorway or window frame.
        if x <= 2 or y <= 2 or x + width >= w - 2 or y + height >= h - 2:
            continue
        keep |= labels == component
    if not keep.any():
        return None

    keep = cv2.dilate(keep.astype(np.uint8), kernel, iterations=2)
    alpha = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), sigmaX=2.0)
    return np.clip(alpha * _BLUE_STRENGTH, 0.0, 1.0)[..., None]


def _blend_artifact_fallback(img, fallback, alpha):
    """Replace only the detected portal region with a filtered re-render."""
    if fallback is None or alpha is None:
        return img
    from PIL import Image
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    clean = np.asarray(fallback.convert("RGB"), dtype=np.float32)
    out = rgb * (1.0 - alpha) + clean * alpha
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), mode="RGB")


def _dilate_domain(mask, iterations):
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(iterations))):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        out = np.logical_or.reduce([
            padded[0:-2, 0:-2], padded[0:-2, 1:-1], padded[0:-2, 2:],
            padded[1:-1, 0:-2], padded[1:-1, 1:-1], padded[1:-1, 2:],
            padded[2:, 0:-2], padded[2:, 1:-1], padded[2:, 2:],
        ])
    return out


def _usd_room_domain(scene_id):
    bare = str(scene_id).removeprefix("sage-")
    key = (bare, round(_USD_BOUNDARY_MARGIN_M, 4))
    if key in _USD_DOMAIN_CACHE:
        return _USD_DOMAIN_CACHE[key]
    path = os.path.join(_USD_MASK_ROOT, bare + ".npz")
    try:
        with np.load(path, allow_pickle=False) as data:
            version = int(np.asarray(data["version"]).reshape(()))
            domain = np.asarray(data["render_domain"], dtype=bool)
            exterior = np.asarray(data["exterior_door_opening"], dtype=bool)
            scale = float(np.asarray(data["scale"]).reshape(()))
            lower = np.asarray(data["lower"], dtype=np.float64)
            shape = tuple(int(v) for v in np.asarray(data["shape"]).tolist())
        if (version < 4 or domain.shape != shape or exterior.shape != shape
                or lower.size < 2
                or scale <= 0):
            raise ValueError("invalid USD room-topology artifact")
        margin_px = int(math.ceil(max(0.0, _USD_BOUNDARY_MARGIN_M) / scale))
        portal_px = int(math.ceil(max(0.0, _USD_PORTAL_NEAR_M) / scale))
        value = (_dilate_domain(domain, margin_px),
                 _dilate_domain(exterior, portal_px), scale, lower[:2])
    except Exception:
        value = None
    _USD_DOMAIN_CACHE[key] = value
    return value


def _clip_to_usd_room_domain(img, depth, scene_id, position, yaw, pitch,
                             camera):
    """Mask a pixel once its camera ray exits the USD room/door domain."""
    artifact = _usd_room_domain(scene_id)
    if img is None or depth is None or artifact is None:
        return img, 0.0
    from PIL import Image

    domain, near_exterior, scale, lower = artifact
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    h, w = rgb.shape[:2]
    if depth.shape == (h, w):
        mask_scale = min(1.0, max(0.125, _USD_BOUNDARY_MASK_SCALE))
        stride = max(1, int(round(1.0 / mask_scale)))
        mask_depth = depth[::stride, ::stride]
    else:
        dh, dw = depth.shape if depth.ndim == 2 else (0, 0)
        stride_y = int(round(h / float(dh))) if dh else 0
        stride_x = int(round(w / float(dw))) if dw else 0
        if (stride_x < 1 or stride_x != stride_y
                or abs(dw * stride_x - w) > stride_x
                or abs(dh * stride_y - h) > stride_y):
            return img, 0.0
        stride = stride_x
        mask_depth = depth
    mh, mw = mask_depth.shape

    # Renderer route convention: yaw=0 faces +x; image-right is -y.  Depth is
    # camera-forward z depth, so the pinhole ray is f + nx*r + ny*down.
    rows, cols = np.indices((mh, mw), dtype=np.float32)
    nx = (cols * stride + 0.5 - w * 0.5) / float(camera.fx)
    ny = (rows * stride + 0.5 - h * 0.5) / float(camera.fy)
    cy, sy = math.cos(float(yaw)), math.sin(float(yaw))
    cp, sp = math.cos(float(pitch)), math.sin(float(pitch))
    forward = np.asarray([cy * cp, sy * cp])
    right = np.asarray([sy, -cy])
    down = np.asarray([cy * sp, sy * sp])
    ray_x = forward[0] + nx * right[0] + ny * down[0]
    ray_y = forward[1] + nx * right[1] + ny * down[1]
    end_x = float(position[0]) + mask_depth * ray_x
    end_y = float(position[1]) + mask_depth * ray_y
    distance = np.hypot(end_x - float(position[0]),
                        end_y - float(position[1]))
    valid = (np.isfinite(mask_depth) & (mask_depth > 1e-4)
             & np.isfinite(distance))
    if not valid.any():
        return img, 0.0

    gh, gw = domain.shape

    def sample_domain(world_x, world_y):
        col = np.rint((world_x - lower[0]) / scale).astype(np.int32)
        row = np.rint((gh - 1) - (world_y - lower[1]) / scale).astype(np.int32)
        inside = (col >= 0) & (col < gw) & (row >= 0) & (row < gh)
        sampled = np.zeros(world_x.shape, dtype=bool)
        sampled[inside] = domain[row[inside], col[inside]]
        return sampled

    # A stale/misaligned artifact must never black out the whole observation.
    camera_col = int(round((float(position[0]) - lower[0]) / scale))
    camera_row = int(round((gh - 1) -
                           (float(position[1]) - lower[1]) / scale))
    if (camera_col < 0 or camera_col >= gw or camera_row < 0
            or camera_row >= gh):
        return img, 0.0
    near_portal = bool(near_exterior[camera_row, camera_col])
    if not domain[camera_row, camera_col] and not near_portal:
        return img, 0.0

    outside = np.zeros((mh, mw), dtype=bool)
    outside_run = np.zeros((mh, mw), dtype=np.uint16)
    step_m = max(scale, 0.05)
    exit_steps = max(1, int(math.ceil(max(0.0, _USD_BOUNDARY_EXIT_M) /
                                      step_m)))
    max_steps = min(256, int(math.ceil(float(np.nanmax(distance[valid])) /
                                       step_m)))
    safe_distance = np.maximum(distance, step_m)
    for step in range(1, max_steps + 1):
        active = valid & ~outside & (distance > (step - 1) * step_m)
        if not active.any():
            break
        fraction = np.minimum(1.0, step * step_m / safe_distance)
        world_x = float(position[0]) + (end_x - float(position[0])) * fraction
        world_y = float(position[1]) + (end_y - float(position[1])) * fraction
        in_domain = sample_domain(world_x, world_y)
        outside_run[active & in_domain] = 0
        outside_run[active & ~in_domain] += 1
        outside |= active & (outside_run >= exit_steps)

    depth_full = mask_depth
    if stride > 1:
        try:
            import cv2
            outside = cv2.resize(outside.astype(np.uint8), (w, h),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
            depth_full = cv2.resize(mask_depth, (w, h),
                                    interpolation=cv2.INTER_NEAREST)
        except Exception:
            outside = np.repeat(np.repeat(outside, stride, axis=0),
                                stride, axis=1)[:h, :w]
            depth_full = np.repeat(np.repeat(mask_depth, stride, axis=0),
                                   stride, axis=1)[:h, :w]
    clipped_fraction = float(outside.mean())
    if _USD_BOUNDARY_BLUE_ONLY and outside.any():
        escaped = outside.copy()
        red, green, blue = np.moveaxis(rgb, -1, 0)
        blue_leak = ((blue - np.maximum(red, green) >= 8.0)
                     & (blue >= 70.0))
        enclosed_seeded = np.zeros_like(escaped)
        try:
            import cv2
            blue_leak = cv2.morphologyEx(
                blue_leak.astype(np.uint8), cv2.MORPH_CLOSE,
                np.ones((3, 3), np.uint8)).astype(bool)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                blue_leak.astype(np.uint8), connectivity=8)
            seeded = np.zeros_like(escaped)
            near_escaped = cv2.dilate(
                escaped.astype(np.uint8), np.ones((11, 11), np.uint8),
                iterations=1).astype(bool)
            for component in range(1, count):
                area = int(stats[component, cv2.CC_STAT_AREA])
                component_mask = labels == component
                overlap = int((escaped & component_mask).sum())
                near_overlap = int((near_escaped & component_mask).sum())
                selected = overlap >= max(16, int(round(area * 0.02)))
                selected |= near_overlap >= max(24, int(round(area * 0.10)))
                if selected:
                    seeded |= component_mask
                    x = int(stats[component, cv2.CC_STAT_LEFT])
                    y = int(stats[component, cv2.CC_STAT_TOP])
                    width = int(stats[component, cv2.CC_STAT_WIDTH])
                    height = int(stats[component, cv2.CC_STAT_HEIGHT])
                    # A compact component enclosed by a portal is exterior
                    # leakage even when its own bad depth is too short.  A
                    # wallpaper component reaches the image edge and stays
                    # constrained by the geometric support below.
                    if (x > 2 and y > 2 and x + width < w - 2
                            and y + height < h - 2):
                        enclosed_seeded |= component_mask
            blue_leak = seeded
            # Bad Gaussians inside a window can report a shorter, bogus depth,
            # leaving blue islands fully enclosed by the ray-exit mask.  Fill
            # only those enclosed holes, then add a tiny antialiasing margin.
            # Indoor walls/wallpaper remain connected to the image boundary,
            # so this cannot turn the old colour component back into a global
            # wallpaper eraser.
            inverse = (~escaped).astype(np.uint8)
            holes = np.zeros_like(escaped)
            hole_count, hole_labels, hole_stats, _ = \
                cv2.connectedComponentsWithStats(inverse, connectivity=8)
            for hole in range(1, hole_count):
                x = int(hole_stats[hole, cv2.CC_STAT_LEFT])
                y = int(hole_stats[hole, cv2.CC_STAT_TOP])
                width = int(hole_stats[hole, cv2.CC_STAT_WIDTH])
                height = int(hole_stats[hole, cv2.CC_STAT_HEIGHT])
                if (x > 0 and y > 0 and x + width < w
                        and y + height < h):
                    holes |= hole_labels == hole
            support = escaped | holes
            support = cv2.dilate(
                support.astype(np.uint8), np.ones((5, 5), np.uint8),
                iterations=1).astype(bool)
        except Exception:
            blue_leak = escaped & blue_leak
        # Connected components are used only to decide which blue regions are
        # genuine exterior leaks. Never let that colour component grow beyond
        # the USD ray-exit mask, otherwise a blue wallpaper/door frame joined
        # to the window would be erased as well.
            support = escaped
        outside = (support & blue_leak) | enclosed_seeded
        clipped_fraction = float(outside.mean())
    elif outside.any():
        # In production USD mode the whole ray-confirmed exterior is neutral,
        # regardless of its colour.  A bad Gaussian can still claim a short
        # depth and punch a blue notch into a window.  Complete only blue
        # pixels inside the bounding envelope of a substantial exterior
        # component; this closes portal holes without touching room-wide blue
        # walls or arbitrary indoor objects.
        try:
            import cv2
            escaped = outside.copy()
            red, green, blue = np.moveaxis(rgb, -1, 0)
            blue_hole = ((blue - np.maximum(red, green) >= 8.0)
                         & (blue >= 70.0))
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                escaped.astype(np.uint8), connectivity=8)
            envelope = np.zeros_like(escaped)
            min_area = max(64, int(round(h * w * 0.002)))
            for component in range(1, count):
                if int(stats[component, cv2.CC_STAT_AREA]) < min_area:
                    continue
                x = int(stats[component, cv2.CC_STAT_LEFT])
                y = int(stats[component, cv2.CC_STAT_TOP])
                width = int(stats[component, cv2.CC_STAT_WIDTH])
                height = int(stats[component, cv2.CC_STAT_HEIGHT])
                envelope[y:y + height, x:x + width] = True
            outside |= envelope & blue_hole
        except Exception:
            pass
        clipped_fraction = float(outside.mean())
    # At an exterior portal the bad sheet can report a bogus short depth and
    # still end inside room_region, so the ray-exit test alone misses it.  Use
    # v4's explicit exterior-door mask only when strong blue dominates the
    # frame; a depth floor protects the nearby real door leaf/frame.
    portal_fallback = False
    if near_portal:
        red, green, blue = np.moveaxis(rgb, -1, 0)
        strong_blue = ((blue - np.maximum(red, green) >= 16.0)
                       & (blue >= 80.0))
        if float(strong_blue.mean()) >= _USD_PORTAL_STRONG_FRACTION:
            portal_blue = ((blue - np.maximum(red, green) >= 8.0)
                           & (blue >= 70.0)
                           & np.isfinite(depth_full)
                           & (depth_full >= _USD_PORTAL_MIN_DEPTH_M))
            try:
                import cv2
                portal_blue = cv2.morphologyEx(
                    portal_blue.astype(np.uint8), cv2.MORPH_CLOSE,
                    np.ones((5, 5), np.uint8)).astype(bool)
            except Exception:
                pass
            outside |= portal_blue
            clipped_fraction = float(outside.mean())
            portal_fallback = True
    if clipped_fraction <= 0.0:
        return img, 0.0
    # Guard against catastrophic frame mismatch while still allowing large
    # windows/boundaries. A geometry fix should fail open, not erase the frame.
    if clipped_fraction > 0.80 and not portal_fallback:
        return img, 0.0
    try:
        import cv2
        alpha = cv2.GaussianBlur(outside.astype(np.float32), (0, 0), 1.0)
    except Exception:
        alpha = outside.astype(np.float32)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    # A hard black cutout is as distracting as the original blue sheet. Keep
    # only a heavily darkened neutral luminance cue: this reads as unmodelled
    # exterior/tinted glass, removes the blue light, and does not invent RGB
    # texture that is absent from the asset.
    red, green, blue = np.moveaxis(rgb, -1, 0)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    neutral = np.clip(
        luminance * max(0.0, _USD_BOUNDARY_LUMA_SCALE)
        + max(0.0, _USD_BOUNDARY_LUMA_FLOOR), 0.0, 96.0)
    neutral = np.repeat(neutral[..., None], 3, axis=2)
    out = rgb * (1.0 - alpha) + neutral * alpha
    return (Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8),
                            mode="RGB"), clipped_fraction)


class XYZEulerClient(_XYZEulerClient):
    """SAGE geometry-frame client with optional cleanup and tonemap.

    SAGE geometry/labels/occupancy use a z-up frame that is not the pose frame
    assumed by the server's ABotN-specific ``/render_gs_xyz_euler`` shortcut.
    Build COLMAP extrinsics directly and use ``/render_gs`` so rendered pixels
    align with labels and navigation without hidden position/yaw transforms.
    """

    def __init__(self, url: str = SAGE_RENDER_URL, timeout: float | None = None,
                 camera=None):
        super().__init__(url=url, timeout=timeout, camera=camera)
        self.last_boundary_clipped_fraction = 0.0

    @staticmethod
    def _extrinsics(position, yaw, pitch=0.0):
        c, s = math.cos(float(yaw)), math.sin(float(yaw))
        cp, sp = math.cos(float(pitch)), math.sin(float(pitch))
        forward = np.array([c * cp, s * cp, sp], dtype=float)
        right = np.array([s, -c, 0.0], dtype=float)
        down = np.cross(forward, right)
        R = np.stack([right, down, forward])
        t = -R @ np.asarray(position, dtype=float)
        qw, qx, qy, qz = rotmat2qvec(R)
        return ("1 %.9f %.9f %.9f %.9f %.9f %.9f %.9f 1 000000.jpg" %
                (qw, qx, qy, qz, t[0], t[1], t[2]))

    def _render_geometry(self, scene_id, position, yaw, pitch=0.0,
                         want_depth=False, render_scale=None,
                         suppress_blue=False, depth_downsample=1):
        payload = {
            "scene_id": scene_id,
            "cam_extrinsics": self._extrinsics(position, yaw, pitch),
            "cam_intrinsics": self.camera.intrinsics_txt(),
        }
        if want_depth:
            payload["return_depth_npz"] = True
            payload["depth_downsample"] = max(1, int(depth_downsample))
        if suppress_blue:
            payload["suppress_blue_gaussians"] = True
        if render_scale is not None:
            payload["render_scale"] = float(render_scale)
        response = self._session.post(
            self.url + "/render_gs", json=payload, timeout=self.timeout)
        if response.status_code != 200:
            raise RuntimeError("SAGE geometry render failed %d: %s" %
                               (response.status_code, response.text[:400]))
        if not want_depth:
            return _to_image(response.content), None
        data = response.json()
        depth = None
        if "depth_npz" in data:
            with np.load(io.BytesIO(base64.b64decode(data["depth_npz"]))) as z:
                depth = z["depth"]
        return _to_image(base64.b64decode(data["image"])), depth

    def _render_without_cool_gaussians(self, scene_id, position, yaw,
                                       pitch=0.0, render_scale=None):
        """Optional server extension; the normal render contract is untouched."""
        return self._render_geometry(
            scene_id, position, yaw, pitch, render_scale=render_scale,
            suppress_blue=True)[0]

    def _render_with_boundary_depth(self, scene_id, position, yaw, pitch,
                                    render_scale):
        """Request low-resolution depth for USD ray-boundary clipping."""
        return self._render_geometry(
            scene_id, position, yaw, pitch, want_depth=True,
            render_scale=render_scale,
            depth_downsample=_USD_BOUNDARY_DEPTH_DOWNSAMPLE)

    def render_one(self, scene_id, position, yaw, pitch: float = 0.0,
                   want_depth: bool = False, render_scale=None):
        if _USD_BOUNDARY_CLIP and not want_depth:
            rgb, depth = self._render_with_boundary_depth(
                scene_id, position, yaw, pitch, render_scale)
        else:
            request_depth = want_depth or _USD_BOUNDARY_CLIP
            rgb, depth = self._render_geometry(
                scene_id, position, yaw, pitch=pitch,
                want_depth=request_depth, render_scale=render_scale)
        if _USD_BOUNDARY_CLIP:
            rgb, self.last_boundary_clipped_fraction = _clip_to_usd_room_domain(
                rgb, depth, scene_id, position, yaw, pitch, self.camera)
        if _OUTSIDE_CLEANUP:
            alpha = _outside_blue_alpha(rgb)
            if alpha is not None:
                try:
                    fallback = self._render_without_cool_gaussians(
                        scene_id, position, yaw, pitch=pitch,
                        render_scale=render_scale)
                except Exception:
                    # Old render servers do not implement the optional flag.
                    # Fail open to the unmodified observation.
                    fallback = None
                rgb = _blend_artifact_fallback(rgb, fallback, alpha)
        if _TONEMAP:
            rgb = _apply_tonemap(rgb)
        return rgb, depth if want_depth else None
