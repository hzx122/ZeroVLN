"""Occupancy-grid kinematics for SAGE-3D scenes.

Same role as :mod:`abotn.occmap` -- the one place that answers "may I stand
here" and "how far do I actually get" -- but SAGE ships a simpler map. Each
InteriorGS scene has::

    occupancy.json   {scale, lower[x,y,z], upper[x,y,z], center}
    occupancy.png    HxW uint8: 255 free, 127 obstacle, 0 unknown/outside

The world<->pixel mapping is a 5 cm regular grid with the origin at ``lower``;
row 0 of the png is the **top**, i.e. world +y max (verified by rendering an
eroded free cell in both row conventions -- the +y-max-at-row-0 one lands the
camera in open floor, the other lands it inside furniture). Scenes are flat, so
there is no height map: the ground is z=0 and the renderer eye sits at the
official 1.2 m camera height (GVLN stores the trajectory/body at z=0.5 m).

numpy + pillow only, to stay importable in the console's habitat venv.
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional, Tuple

import numpy as np

FREE = 255          # SAGE occupancy.png: exactly 255 is free space
WALL = 127          # official occupancy.png wall value (kept blocking in every mode)
GROUND_Z = 0.0      # scenes are flat; GVLN poses carry the eye height in z

# Collision model selector (env SAGE_COLLISION, or ctor arg):
#   "official" -- walkable = occupancy.png free (occ>=255). Faithful to the
#                 SAGE-3D benchmark, but 3DGS renders visible floor over the
#                 ~30% "Unable Area" (occ==0) cells the map marks unwalkable,
#                 so free-roam/RL hit invisible air-walls on that floor.
#   "hybrid"    -- walkable = official-free UNION structure.json room floor,
#                 walls (occ==127) still block. RECOMMENDED: a strict superset
#                 of official-free, so it removes the air-walls (recovers ~22%
#                 room-covered floor) AND raises usable golden episodes (instr
#                 60.5%->86.3% fully-walkable) while never clipping real walls.
#   "visual"    -- PLY-first physical map precomputed from the same Gaussians
#                 as RGB: candidate floor (official free UNION rooms) minus
#                 coherent body-height geometry. This is the only mode whose
#                 static obstacles are designed to match the rendered scene.
#   "self"      -- walkable = structure.json rooms minus structure.json walls
#                 (ignores occupancy.png entirely). Pure self-built collision;
#                 undercovers benchmark circulation space so golden fit drops.
# Both non-official modes rasterize structure.json {rooms,walls} onto the same
# grid; furniture blocking stays orthogonal (SAGE_FURNITURE_COLLISION).
COLLISION_MODES = ("official", "hybrid", "visual", "self", "usd",
                   "usd_visual", "usd_rooms")
FURNITURE_MODES = ("none", "labels", "geom")
GEOM_FURNITURE_ROOT = os.environ.get(
    "SAGE_FURNITURE_MASK_ROOT",
    "/data/zerovln/SAGE-3D/sage_collision_masks")
DEFAULT_USD_COLLISION_ROOT = (
    "/data/zerovln/SAGE-3D/"
    "sage_usd_collision_masks")
DEFAULT_USD_ROOMS_ROOT = (
    "/data/zerovln/SAGE-3D/"
    "sage_usd_rooms_masks")
USD_COLLISION_ROOT = os.environ.get(
    "SAGE_USD_COLLISION_MASK_ROOT", DEFAULT_USD_COLLISION_ROOT)


def usd_collision_root(collision_mode: str) -> str:
    """Resolve the artifact root without letting ``usd_rooms`` read v2.

    The environment variable remains the explicit override.  A few offline
    tools historically monkeypatch ``USD_COLLISION_ROOT`` after import, so keep
    honoring a non-default value.  Otherwise direct ``vln_model.py`` launches
    get the v4 rooms root automatically instead of silently opening the legacy
    v2 USD mask.
    """
    explicit = os.environ.get("SAGE_USD_COLLISION_MASK_ROOT", "").strip()
    if explicit:
        return explicit
    if USD_COLLISION_ROOT != DEFAULT_USD_COLLISION_ROOT:
        return USD_COLLISION_ROOT
    return (DEFAULT_USD_ROOMS_ROOT
            if str(collision_mode).lower() == "usd_rooms"
            else DEFAULT_USD_COLLISION_ROOT)

# labels.json instances kept walkable even when they touch the floor band: they
# are openings/passages, not solid furniture.
FURNITURE_PASSABLE = {"door", "interior doors", "door curtain"}


class OccMap:
    """One scene's walkability, in world metres."""

    def __init__(self, scene_dir: str, block_furniture: bool = False,
                 collision_mode: Optional[str] = None,
                 furniture_mode: Optional[str] = None):
        from PIL import Image

        self.scene_dir = scene_dir
        with open(os.path.join(scene_dir, "occupancy.json"), encoding="utf-8") as f:
            meta = json.load(f)
        self.scale = float(meta["scale"])                 # metres per pixel
        self.lower = [float(v) for v in meta["lower"]]    # world (x, y, z) of grid origin
        self.upper = [float(v) for v in meta["upper"]]

        occ = Image.open(os.path.join(scene_dir, "occupancy.png"))
        if occ.mode != "L":
            occ = occ.convert("L")
        self.occ = np.asarray(occ, dtype=np.uint8)
        self.free = self.occ >= FREE
        self.h, self.w = self.occ.shape[:2]
        self.mpp = self.scale
        # Collision model: default official; env SAGE_COLLISION or ctor arg may
        # switch to the self-built structure.json mask. Applied here so it
        # re-writes ``self.free`` before furniture blocking / clearance, keeping
        # every downstream method (is_free/clearance/seg_free/try_step) unchanged.
        if collision_mode is None:
            collision_mode = os.environ.get("SAGE_COLLISION", "official").strip().lower()
        if collision_mode not in COLLISION_MODES:
            collision_mode = "official"
        self.collision_mode = collision_mode
        if collision_mode == "visual":
            # Build a useful hybrid fallback first, then replace it with the
            # offline PLY-first map when the artifact is present and valid.
            self._apply_structure_mode("hybrid")
            self.collision_mode = "visual"
            self._apply_visual_collision_mask()
        elif collision_mode in ("usd", "usd_visual", "usd_rooms"):
            # Optional side path built from SAGE-3D's separately released USD
            # collision payload.  It is deliberately opt-in: the established
            # official/visual modes and their artifacts remain untouched.
            self._apply_usd_collision_mask()
            if collision_mode == "usd_rooms" and self.usd_mask_missing:
                raise RuntimeError(
                    "usd_rooms requires a valid v4 collision artifact; "
                    f"failed to load {self.usd_mask_path}. Set "
                    "SAGE_USD_COLLISION_MASK_ROOT to sage_usd_rooms_masks")
            if collision_mode == "usd_rooms" and not self.usd_mask_missing:
                self.collision_mode = "usd_rooms"
                self._apply_usd_rooms()
            elif collision_mode == "usd_visual" and not self.usd_mask_missing:
                self.collision_mode = "usd_visual"
                self._apply_usd_visual_guard()
        elif collision_mode != "official":
            self._apply_structure_mode(collision_mode)
        # SAGE's occupancy.png marks only walls (127) and unable-area (0) as
        # obstacles -- every piece of furniture is 255 (free), matching the
        # official benchmark's furniture-passable collision. For the console we
        # optionally block floor-standing furniture so the agent stops at
        # visible wardrobes/stools/plants instead of walking through them.
        # Furniture collision is separate from floor/wall collision. ``labels``
        # is the legacy axis-aligned bbox implementation (kept only for exact
        # reproduction); ``geom`` loads a small precomputed mask projected from
        # the same PLY that produces RGB. The old boolean/env remains a backwards
        # compatible alias for ``labels``.
        if furniture_mode is None:
            furniture_mode = os.environ.get("SAGE_FURNITURE_COLLISION", "").strip().lower()
        if not furniture_mode:
            furniture_mode = "labels" if block_furniture else "none"
        if furniture_mode not in FURNITURE_MODES:
            furniture_mode = "none"
        self.furniture_mode = furniture_mode
        self.block_furniture = furniture_mode != "none"
        if furniture_mode == "labels":
            self._block_furniture_from_labels()
        elif furniture_mode == "geom":
            self._block_furniture_from_geom_mask()
        self._rebuild_clearance()

    # -- self-built collision (structure.json) -----------------------------
    def _apply_structure_mode(self, mode: str):
        """Re-write ``self.free`` from ``structure.json`` room/wall polygons.

        ``structure.json`` (present for every InteriorGS scene) carries the
        architect floor plan in the SAME geometry frame as occupancy.png::

            rooms[].profile  2D interior floor polygons  -> walkable
            walls[].location [[x,y],[x,y]] + thickness   -> obstacle strips
            holes[]          DOOR/WINDOW openings         -> ignored (passable)

        ``hybrid`` = official-free UNION rooms, minus official walls (occ==127)
        -- a superset of official-free that fills the air-wall floor. ``self`` =
        rooms minus structure walls, ignoring occupancy.png. numpy-only (no
        matplotlib/scipy) to stay importable in the console's habitat venv; if
        structure.json is missing or unreadable, silently keeps official free."""
        path = os.path.join(self.scene_dir, "structure.json")
        if not os.path.exists(path):
            self.collision_mode = "official"
            return
        try:
            with open(path, encoding="utf-8") as f:
                st = json.load(f)
        except Exception:
            self.collision_mode = "official"
            return
        # cell-center world coords; row0 = +y max (matches to_pixel)
        xs = self.lower[0] + (np.arange(self.w) + 0.5) * self.scale
        ys = self.lower[1] + (self.h - 0.5 - np.arange(self.h)) * self.scale
        gx, gy = np.meshgrid(xs, ys)                      # (h, w)
        room = np.zeros((self.h, self.w), bool)
        for r in st.get("rooms", []):
            prof = r.get("profile") or []
            if len(prof) >= 3:
                room |= _point_in_poly(gx, gy, [(float(p[0]), float(p[1])) for p in prof])
        wall = np.zeros((self.h, self.w), bool)
        for wseg in st.get("walls", []):
            loc = wseg.get("location") or []
            if len(loc) < 2:
                continue
            (ax, ay), (bx, by) = loc[0][:2], loc[1][:2]
            half = float(wseg.get("thickness", 0.2)) / 2.0 + 0.02
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy + 1e-9
            t = np.clip(((gx - ax) * dx + (gy - ay) * dy) / L2, 0.0, 1.0)
            cx, cy = ax + t * dx, ay + t * dy
            wall |= ((gx - cx) ** 2 + (gy - cy) ** 2) <= half * half
        if mode == "hybrid":
            self.free = (self.free | room) & (self.occ != WALL)
        else:  # "self": pure structure.json floor plan
            self.free = room & ~wall

    def _rebuild_clearance(self):
        """Euclidean distance (metres) from every free pixel to the nearest
        obstacle, via scipy's EDT -- the exact kernel the official SAGE-3D
        ``collision_detector.py`` uses for robot-radius inflation
        (``distance_transform_edt`` + ``dist_m <= robot_radius``). Cached on
        ``self.clearance_m``; rebuilt whenever ``self.free`` changes (e.g.
        ``block_furniture`` re-marks floor-standing labels). Falls back to
        ``None`` if scipy is unavailable, in which case ``is_free`` reverts to
        the legacy square-patch test (still correct, just slightly stricter)."""
        try:
            from scipy.ndimage import distance_transform_edt
        except Exception:
            self.clearance_m = None
            return
        self.clearance_m = distance_transform_edt(self.free, sampling=self.mpp)

    def _apply_visual_collision_mask(self):
        """Replace static walkability with the RGB-PLY-derived physical map.

        The artifact is built offline by :mod:`sage.build_geom_collision` so
        runtime workers never parse the multi-hundred-MB PLY. Missing assets
        fall back to hybrid instead of turning the scene into an empty map.
        """
        bare = os.path.basename(os.path.normpath(self.scene_dir)).split("_")[-1]
        path = os.path.join(GEOM_FURNITURE_ROOT, bare + ".npz")
        self.visual_mask_path = path
        try:
            with np.load(path, allow_pickle=False) as data:
                visual_free = np.asarray(data["visual_free"], dtype=bool)
        except Exception:
            self.visual_mask_missing = True
            self.collision_mode = "hybrid"
            return
        if visual_free.shape != self.free.shape:
            self.visual_mask_missing = True
            self.collision_mode = "hybrid"
            return
        self.visual_mask_missing = False
        self.free = visual_free

    def _apply_usd_collision_mask(self):
        """Load a walkability mask rasterized from the official USD payload.

        The payload-local coordinates already match InteriorGS occupancy and
        labels.  ``Data/template.usda`` applies an additional rotation only
        when composing the payload into Isaac's stage; applying it here would
        rotate an already aligned mask a second time.

        Invalid or missing artifacts fail back to the byte-for-byte official
        occupancy map and expose ``usd_mask_missing`` for diagnostics.
        """
        bare = os.path.basename(os.path.normpath(self.scene_dir)).split("_")[-1]
        root = usd_collision_root(self.collision_mode)
        path = os.path.join(root, bare + ".npz")
        self.usd_mask_root = root
        self.usd_mask_path = path
        try:
            with np.load(path, allow_pickle=False) as data:
                free = np.asarray(data["free"], dtype=bool)
                shape = tuple(int(v) for v in np.asarray(data["shape"]).tolist())
                scale = float(np.asarray(data["scale"]).reshape(()))
                lower = np.asarray(data["lower"], dtype=float)
                version = int(np.asarray(data["version"]).reshape(()))
                obstacle = np.asarray(data["obstacle"], dtype=bool)
                candidate = np.asarray(data["candidate"], dtype=bool)
                rooms_free = (np.asarray(data["rooms_free"], dtype=bool)
                              if "rooms_free" in data.files else None)
                room_region = (np.asarray(data["room_region"], dtype=bool)
                               if "room_region" in data.files else None)
                door_opening = (np.asarray(data["door_opening"], dtype=bool)
                                if "door_opening" in data.files else None)
        except Exception:
            self.usd_mask_missing = True
            self.collision_mode = "official"
            return
        if (version < 2 or free.shape != self.free.shape
                or shape != self.free.shape
                or obstacle.shape != self.free.shape
                or candidate.shape != self.free.shape
                or abs(scale - self.scale) > 1e-6
                or lower.ndim != 1 or lower.size < 2
                or not np.allclose(lower[:2], self.lower[:2], atol=1e-4)):
            self.usd_mask_missing = True
            self.collision_mode = "official"
            return
        self.usd_mask_missing = False
        self._usd_obstacle = obstacle
        self._usd_candidate = candidate
        self._usd_version = version
        self._usd_rooms_free = rooms_free
        self._usd_room_region = room_region
        self._usd_door_opening = door_opening
        self.free = free

    def _apply_usd_rooms(self):
        """Use USD room topology while retaining USD walls/object collision.

        The v3 artifact expands navigation to every declared room and carves
        only explicit ``structure.holes[type=DOOR|HOLE]`` through the collision
        shell.  It therefore removes doorway air walls without reopening
        windows or dropping furniture collision in formerly unable rooms.
        """
        rooms_free = self._usd_rooms_free
        room_region = self._usd_room_region
        door_opening = self._usd_door_opening
        if (rooms_free is None or room_region is None or door_opening is None
                or rooms_free.shape != self.free.shape
                or room_region.shape != self.free.shape
                or door_opening.shape != self.free.shape):
            self.usd_rooms_mask_missing = True
            raise RuntimeError(
                "usd_rooms requires rooms_free/room_region/door_opening v4 "
                f"fields, but {self.usd_mask_path} is a legacy/incompatible "
                "artifact. Point SAGE_USD_COLLISION_MASK_ROOT at "
                "sage_usd_rooms_masks")
        self.usd_rooms_mask_missing = False
        self.usd_room_pixels = int(room_region.sum())
        self.usd_door_pixels = int(door_opening.sum())
        self.free = rooms_free

    def _apply_usd_visual_guard(self):
        """Keep only USD collision cells supported by nearby RGB geometry.

        ``usd`` remains the untouched projection of the official collision
        asset. ``usd_visual`` is the production physical mode: an independent
        nurec/PLY mask rejects released collision pieces that are visibly
        absent, while a tolerance keeps slightly misregistered real surfaces.
        """
        bare = os.path.basename(os.path.normpath(self.scene_dir)).split("_")[-1]
        path = os.path.join(GEOM_FURNITURE_ROOT, bare + ".npz")
        self.usd_visual_mask_path = path
        try:
            with np.load(path, allow_pickle=False) as data:
                visual_free = np.asarray(data["visual_free"], dtype=bool)
                shape = tuple(int(v) for v in np.asarray(data["shape"]).tolist())
                scale = float(np.asarray(data["scale"]).reshape(()))
                lower = np.asarray(data["lower"], dtype=float)
                version = int(np.asarray(data["version"]).reshape(()))
        except Exception:
            self.usd_visual_mask_missing = True
            self.collision_mode = "usd"
            return
        if (version < 3 or visual_free.shape != self.free.shape
                or shape != self.free.shape
                or abs(scale - self.scale) > 1e-6
                or lower.ndim != 1 or lower.size < 2
                or not np.allclose(lower[:2], self.lower[:2], atol=1e-4)):
            self.usd_visual_mask_missing = True
            self.collision_mode = "usd"
            return
        try:
            from scipy.ndimage import binary_dilation
        except Exception:
            self.usd_visual_mask_missing = True
            self.collision_mode = "usd"
            return

        margin_m = max(0.0, float(os.environ.get("SAGE_USD_VISUAL_GUARD_M", "0.10")))
        iterations = int(math.ceil(margin_m / self.mpp))
        visual_obstacle = self._usd_candidate & ~visual_free
        support = (binary_dilation(visual_obstacle, iterations=iterations)
                   if iterations else visual_obstacle)
        guarded_obstacle = self._usd_obstacle & support
        self.usd_visual_mask_missing = False
        self.usd_raw_obstacle_pixels = int(self._usd_obstacle.sum())
        self.usd_guarded_obstacle_pixels = int(guarded_obstacle.sum())
        self.usd_unsupported_pixels = int((self._usd_obstacle & ~support).sum())
        self.free = self._usd_candidate & ~guarded_obstacle

    def _block_furniture_from_labels(self, floor_band_m: float = 0.35):
        """Mark floor-standing furniture bboxes (from ``labels.json``) as blocked.

        Each label instance carries an axis-aligned world-xy bounding box and a
        z range. Items whose ``z_min`` sits within ``floor_band_m`` of the floor
        occupy the agent's body height, so they should stop it; items that begin
        higher (ceiling lights, wall mirrors, counter-top props, windows) are
        overhead/mounted and stay passable. Openings (doors/curtains) are kept
        walkable by an explicit allow-list."""
        path = os.path.join(self.scene_dir, "labels.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                labels = json.load(f)
        except Exception:
            return
        blocked = 0
        for inst in labels:
            label = str(inst.get("label", "")).strip().lower()
            if label in FURNITURE_PASSABLE:
                continue
            bb = inst.get("bounding_box") or []
            if not bb:
                continue
            xs = [float(p["x"]) for p in bb]
            ys = [float(p["y"]) for p in bb]
            zs = [float(p["z"]) for p in bb]
            if min(zs) > floor_band_m:          # overhead / mounted -> passable
                continue
            self._block_box(min(xs), max(xs), min(ys), max(ys))
            blocked += 1
        self._furniture_blocked = blocked

    def _block_furniture_from_geom_mask(self):
        """Block PLY-derived body-height surfaces within furniture regions.

        Masks are produced offline by ``sage.build_geom_collision``. Runtime
        loading is intentionally tiny and deterministic; reading a 0.2--0.7 GB
        PLY in every console/bench worker would make this mode unusable.
        Missing/mismatched masks fail open and are recorded on the instance so
        callers can surface the asset problem without inventing air-walls.
        """
        bare = os.path.basename(os.path.normpath(self.scene_dir)).split("_")[-1]
        path = os.path.join(GEOM_FURNITURE_ROOT, bare + ".npz")
        self.furniture_mask_path = path
        if not os.path.exists(path):
            self.furniture_mask_missing = True
            self._furniture_blocked = 0
            return
        try:
            with np.load(path, allow_pickle=False) as data:
                mask = np.asarray(data["mask"], dtype=bool)
                wall_remove = (np.asarray(data["wall_remove"], dtype=bool)
                               if "wall_remove" in data.files else None)
        except Exception:
            self.furniture_mask_missing = True
            self._furniture_blocked = 0
            return
        if mask.shape != self.free.shape:
            self.furniture_mask_missing = True
            self._furniture_blocked = 0
            return
        self.furniture_mask_missing = False
        # A very small number of scene-specific occupancy wall pixels have
        # been verified against all three available geometry sources (RGB PLY,
        # structure walls and labels) and are known stale annotations.  Repairs
        # live in the offline artifact rather than broad runtime heuristics.
        # They are enabled only in hybrid/physical mode; official mode remains
        # byte-for-byte faithful to the benchmark occupancy.
        self.wall_repaired = 0
        if (self.collision_mode == "hybrid" and wall_remove is not None
                and wall_remove.shape == self.free.shape):
            self.free |= wall_remove
            self.wall_repaired = int(wall_remove.sum())
        self.furniture_mask = mask
        self.free &= ~mask
        self._furniture_blocked = int(mask.sum())

    def _block_box(self, x0: float, x1: float, y0: float, y1: float):
        """Clear ``self.free`` over the world-xy rectangle ``[x0,x1]x[y0,y1]``."""
        c0, r1 = self.to_pixel(x0, y0)          # (col, row); row flips y
        c1, r0 = self.to_pixel(x1, y1)
        cmin, cmax = sorted((c0, c1))
        rmin, rmax = sorted((r0, r1))
        cmin = max(0, cmin); rmin = max(0, rmin)
        cmax = min(self.w - 1, cmax); rmax = min(self.h - 1, rmax)
        if cmax >= cmin and rmax >= rmin:
            self.free[rmin:rmax + 1, cmin:cmax + 1] = False

    # -- coordinates -------------------------------------------------------
    def to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """World metres -> ``(col, row)``; row 0 is world +y max."""
        col = (x - self.lower[0]) / self.scale
        row = (self.h - 1) - (y - self.lower[1]) / self.scale
        return int(round(col)), int(round(row))

    def to_world(self, col: float, row: float) -> Tuple[float, float]:
        return (self.lower[0] + col * self.scale,
                self.lower[1] + (self.h - 1 - row) * self.scale)

    def ground_z(self, x: float, y: float, default: float = GROUND_Z) -> float:
        return GROUND_Z

    # -- walkability -------------------------------------------------------
    def radius_px(self, radius_m: float) -> int:
        return max(0, int(round(radius_m / self.mpp)))

    def is_free(self, x: float, y: float, radius_m: float = 0.0) -> bool:
        """Is a circular footprint of ``radius_m`` around ``(x, y)`` clear?

        Walkable iff the point is itself free **and** its clearance (EDT
        distance to the nearest obstacle) strictly exceeds ``radius_m`` --
        the exact inverse of the official ``dist_m <= robot_radius`` obstacle
        test, so a 0.1 m robot matches the SAGE-3D benchmark collision. With
        scipy absent (``clearance_m`` is None), falls back to the legacy
        square-patch test (slightly stricter by ~4 %, see update log).
        """
        px, py = self.to_pixel(x, y)
        if not (0 <= px < self.w and 0 <= py < self.h):
            return False
        if not self.free[py, px]:
            return False
        cm = getattr(self, "clearance_m", None)
        if cm is None:
            r = self.radius_px(radius_m)
            if r == 0:
                return True
            patch = self.free[max(0, py - r):py + r + 1,
                              max(0, px - r):px + r + 1]
            return bool(patch.size and patch.all())
        return float(cm[py, px]) > radius_m

    def _grid_segment_free(self, x0, y0, x1, y1, radius_m=0.):
        """Original half-cell samples and banker rounding, in one array query.

        This is only the base grid predicate. Subclass native/physical
        oracles must still run for every originally required sample.
        """
        cm = getattr(self, "clearance_m", None)
        if (cm is None or type(self).to_pixel is not OccMap.to_pixel
                or type(self)._samples is not OccMap._samples
                or not all(math.isfinite(v) for v in (x0, y0, x1, y1, radius_m))):
            return None
        n = max(2, int(math.hypot(x1-x0, y1-y0)/(self.mpp*.5))+1)
        t = np.arange(1, n+1, dtype=np.float64)/float(n)
        x, y = x0+(x1-x0)*t, y0+(y1-y0)*t
        col = np.rint((x-self.lower[0])/self.scale)
        row = np.rint((self.h-1)-(y-self.lower[1])/self.scale)
        if not np.all((col >= 0) & (col < self.w) & (row >= 0) & (row < self.h)):
            return False
        col, row = col.astype(np.intp), row.astype(np.intp)
        return bool(np.all(self.free[row, col] & (cm[row, col] > radius_m)))

    def seg_free(self, x0, y0, x1, y1, radius_m: float = 0.0) -> bool:
        if type(self).is_free is OccMap.is_free:
            result = self._grid_segment_free(x0, y0, x1, y1, radius_m)
            if result is not None:
                return result
        for x, y in self._samples(x0, y0, x1, y1):
            if not self.is_free(x, y, radius_m):
                return False
        return True

    def _samples(self, x0, y0, x1, y1):
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
        """Move towards ``(x1, y1)``; return ``(x, y, collided)`` with wall slide."""
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
        """A random free point biased towards open floor (roomiest of N tries)."""
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
        cm = getattr(self, "clearance_m", None)
        if cm is not None:
            return min(float(cm[py, px]), limit_m)
        lim = self.radius_px(limit_m)
        for r in range(1, lim + 1):
            patch = self.free[max(0, py - r):py + r + 1, max(0, px - r):px + r + 1]
            if patch.size != (2 * r + 1) ** 2 or not patch.all():
                return (r - 1) * self.mpp
        return limit_m

    # -- visualisation -----------------------------------------------------
    def top_down(self, size: int = 512):
        """``(PIL image, to_px, canvas)`` -- free space light, obstacles dark."""
        from PIL import Image

        scale = min(size / float(self.w), size / float(self.h))
        tw, th = max(1, int(self.w * scale)), max(1, int(self.h * scale))
        rgb = np.where(self.free[:, :, None],
                       np.array([86, 96, 112], np.uint8),
                       np.array([18, 20, 24], np.uint8))
        img = Image.fromarray(rgb).resize((tw, th), Image.NEAREST)
        ox, oy = (size - tw) // 2, (size - th) // 2

        def to_px(x: float, y: float):
            px, py = self.to_pixel(x, y)
            return (ox + px * scale, oy + py * scale)

        return img, to_px, (size, size)


def load(scene_dir: str, cache: Optional[dict] = None,
         block_furniture: bool = False,
         collision_mode: Optional[str] = None,
         furniture_mode: Optional[str] = None) -> OccMap:
    key = (scene_dir, collision_mode, furniture_mode, bool(block_furniture))
    if cache is None:
        return OccMap(scene_dir, block_furniture=block_furniture,
                      collision_mode=collision_mode, furniture_mode=furniture_mode)
    if key not in cache:
        cache[key] = OccMap(scene_dir, block_furniture=block_furniture,
                            collision_mode=collision_mode, furniture_mode=furniture_mode)
    return cache[key]


def _point_in_poly(gx: "np.ndarray", gy: "np.ndarray", poly) -> "np.ndarray":
    """Vectorised even-odd point-in-polygon over a grid (numpy only).

    ``gx``/``gy`` are HxW cell-center coordinates; ``poly`` is a list of
    ``(x, y)`` ring vertices. Returns a HxW bool mask. Ray-cast crossing test,
    no matplotlib so it stays importable in the console's habitat venv."""
    inside = np.zeros(gx.shape, bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > gy) != (yj > gy)) & (
            gx < (xj - xi) * (gy - yi) / (yj - yi + 1e-12) + xi)
        inside ^= cond
        j = i
    return inside
