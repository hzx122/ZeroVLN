"""Navmesh queries: the kinematics half of the 3DRS simulator.

The 3DGS render server gives us pixels; this module gives us everything
habitat's ``PathFinder`` used to: where the floor is, whether a point is
navigable, and where a step actually ends when it runs into a wall.

Only *walkable* polygons participate (see ``NavMesh.walkable``), and only their
detail triangles -- that is the same surface habitat reports as navigable area.
Triangles are bucketed into a uniform x/z grid so a point query touches a
handful of them instead of all ~30k.

Frame: y-up world, same as ``<scene>.gs.ply``.  Heading convention matches
habitat: yaw 0 faces -z and grows counter-clockwise seen from above.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .navmesh import NavMesh

# Habitat's PathFinder.is_navigable default: a point counts as on the navmesh
# when a triangle is within this vertical distance of it.
MAX_Y_DELTA = 0.5


class NavQuery:
    """Spatial index over one scene's walkable surface."""

    def __init__(self, nav: NavMesh, cell: float = 1.0):
        self.nav = nav
        self.cell = float(cell)

        tris = nav.tris if nav.tris is not None else np.zeros((0, 3, 3), np.float32)
        tri_poly = (nav.tri_poly if nav.tri_poly is not None
                    else np.zeros(len(tris), np.int64))
        if len(tris) and nav.walkable is not None and nav.tri_poly is not None:
            keep = nav.walkable[nav.tri_poly]
            tris = tris[keep]
            tri_poly = tri_poly[keep]
        self.tris = np.ascontiguousarray(tris, dtype=np.float64)
        # Kept in the same filtered order as ``self.tris`` so higher-level
        # waypoint followers can lift a point query onto the polygon graph.
        self.tri_poly = np.ascontiguousarray(tri_poly, dtype=np.int64)

        # per-triangle xz bounding box, for the grid and for early rejection
        if len(self.tris):
            self.tmin = self.tris.min(axis=1)
            self.tmax = self.tris.max(axis=1)
            self.origin = self.tmin[:, [0, 2]].min(axis=0) - 1e-6
        else:
            self.tmin = self.tmax = np.zeros((0, 3))
            self.origin = np.zeros(2)
        self._grid = self._build_grid()
        self._islands: Optional[List[np.ndarray]] = None
        self._boundary: Optional[np.ndarray] = None

    # -- index -------------------------------------------------------------
    def _cells(self, lo, hi):
        i0, j0 = np.floor((lo - self.origin) / self.cell).astype(int)
        i1, j1 = np.floor((hi - self.origin) / self.cell).astype(int)
        return i0, j0, i1, j1

    def _build_grid(self):
        grid = {}
        for t in range(len(self.tris)):
            lo = self.tmin[t][[0, 2]]
            hi = self.tmax[t][[0, 2]]
            i0, j0, i1, j1 = self._cells(lo, hi)
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    grid.setdefault((i, j), []).append(t)
        return {k: np.asarray(v, dtype=np.int64) for k, v in grid.items()}

    def candidates(self, x: float, z: float) -> np.ndarray:
        key = (int(math.floor((x - self.origin[0]) / self.cell)),
               int(math.floor((z - self.origin[1]) / self.cell)))
        return self._grid.get(key, _EMPTY)

    # -- point queries -----------------------------------------------------
    def heights_at(self, x: float, z: float) -> np.ndarray:
        """Surface heights of every walkable triangle covering ``(x, z)``."""
        idx = self.candidates(x, z)
        if not len(idx):
            return _EMPTY_F
        t = self.tris[idx]
        w = _bary_xz(t, x, z)
        hit = np.all(w >= -1e-9, axis=1)
        if not hit.any():
            return _EMPTY_F
        w, t = w[hit], t[hit]
        return (w * t[:, :, 1]).sum(axis=1)   # barycentric interpolation of y

    def height_at(self, x: float, z: float, y_ref: float = 0.0) -> Optional[float]:
        """Height of the surface under ``(x, z)`` nearest ``y_ref``, or None."""
        h = self.heights_at(x, z)
        if not len(h):
            return None
        return float(h[np.argmin(np.abs(h - y_ref))])

    def is_navigable(self, p: Sequence[float], max_y_delta: float = MAX_Y_DELTA) -> bool:
        y = self.height_at(p[0], p[2], p[1])
        return y is not None and abs(y - p[1]) <= max_y_delta

    def snap(self, p: Sequence[float], max_y_delta: float = MAX_Y_DELTA):
        """Drop ``p`` onto the surface below/above it; None if off-navmesh."""
        y = self.height_at(p[0], p[2], p[1])
        if y is None or abs(y - p[1]) > max_y_delta:
            return None
        return np.array([p[0], y, p[2]], dtype=float)

    def snap_nearest(self, p: Sequence[float], radius: float = 2.0):
        """Nearest point on the walkable surface within ``radius`` (3D)."""
        if not len(self.tris):
            return None
        p = np.asarray(p, dtype=float)
        near = np.where((self.tmax[:, 0] >= p[0] - radius) &
                        (self.tmin[:, 0] <= p[0] + radius) &
                        (self.tmax[:, 2] >= p[2] - radius) &
                        (self.tmin[:, 2] <= p[2] + radius))[0]
        if not len(near):
            return None
        pts = _closest_on_tris(self.tris[near], p)
        d = np.linalg.norm(pts - p, axis=1)
        k = int(np.argmin(d))
        return None if d[k] > radius else pts[k]

    # -- connectivity ------------------------------------------------------
    def islands(self) -> List[np.ndarray]:
        """Walkable polygon islands as triangle-index arrays, largest first."""
        if self._islands is None:
            nav = self.nav
            keep = np.ones(len(nav.polys), dtype=bool) if nav.walkable is None \
                else nav.walkable.copy()
            tri_poly = nav.tri_poly[keep[nav.tri_poly]] if nav.tri_poly is not None \
                else np.zeros(0, np.int64)
            # triangle order here must match self.tris' filtering above
            out = []
            for comp in nav.islands():
                comp = [c for c in comp if keep[c]]
                if not comp:
                    continue
                sel = np.where(np.isin(tri_poly, comp))[0]
                if len(sel):
                    out.append(sel)
            self._islands = out
        return self._islands

    def sample_navigable(self, rng: np.random.Generator, island: int = 0):
        """Uniform-by-area point on island ``island`` (0 = largest)."""
        isl = self.islands()
        if not isl:
            return None
        tris = self.tris[isl[min(island, len(isl) - 1)]]
        a = _tri_areas(tris)
        t = tris[rng.choice(len(tris), p=a / a.sum())]
        u, v = rng.random(), rng.random()
        if u + v > 1.0:
            u, v = 1.0 - u, 1.0 - v
        return t[0] + u * (t[1] - t[0]) + v * (t[2] - t[0])

    def boundary_edges(self) -> np.ndarray:
        """(E, 2, 3) edges owned by exactly one triangle -- the walls."""
        if self._boundary is None:
            if not len(self.tris):
                self._boundary = np.zeros((0, 2, 3))
                return self._boundary
            v = self.tris.reshape(-1, 3)
            # quantise so shared vertices from different triangles compare equal
            key = np.round(v, 4)
            _, inv = np.unique(key, axis=0, return_inverse=True)
            inv = inv.reshape(-1, 3)
            e = np.concatenate([inv[:, [0, 1]], inv[:, [1, 2]], inv[:, [2, 0]]], 0)
            es = np.sort(e, axis=1)
            uniq, counts = np.unique(es, axis=0, return_counts=True)
            once = uniq[counts == 1]
            pos = {}
            for row, iv in zip(v, inv.ravel()):
                pos.setdefault(iv, row)
            self._boundary = np.array(
                [[pos[a], pos[b]] for a, b in once]) if len(once) else np.zeros((0, 2, 3))
        return self._boundary

    # -- motion ------------------------------------------------------------
    def try_step(self, start: Sequence[float], end: Sequence[float],
                 slide: bool = True, step: float = 0.05,
                 max_y_delta: float = MAX_Y_DELTA):
        """Move from ``start`` towards ``end``, stopping at geometry.

        Returns ``(position, collided)``.  The straight path is walked in
        ``step``-metre increments so a step never tunnels through a wall; the
        height is re-snapped at every increment, which is what lets the agent
        walk up ramps and stairs.  When ``slide`` is set and the path is
        blocked, the *remaining* motion is projected onto the nearest wall edge
        and retried -- habitat's ``allow_sliding`` in spirit: grazing a corridor
        wall keeps you moving instead of stopping you dead.
        """
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        pos, collided = self._walk(start, end, step, max_y_delta)
        if not collided or not slide:
            return pos, collided
        remaining = end - pos
        remaining[1] = 0.0
        if np.linalg.norm(remaining) < 1e-6:
            return pos, True
        edge = self._nearest_boundary_dir(pos)
        if edge is None:
            return pos, True
        proj = float(np.dot(remaining, edge)) * edge
        if np.linalg.norm(proj) < 1e-6:
            return pos, True
        slid, _ = self._walk(pos, pos + proj, step, max_y_delta)
        return slid, True

    def _walk(self, start, end, step, max_y_delta):
        d = end - start
        d[1] = 0.0
        dist = float(np.linalg.norm(d))
        if dist < 1e-9:
            return start.copy(), False
        n = max(1, int(math.ceil(dist / step)))
        dir_ = d / dist
        pos = start.copy()
        for i in range(1, n + 1):
            t = min(dist, i * step)
            cand = start + dir_ * t
            y = self.height_at(cand[0], cand[2], pos[1])
            if y is None or abs(y - pos[1]) > max_y_delta:
                return pos, True
            pos = np.array([cand[0], y, cand[2]])
        return pos, False

    def _nearest_boundary_dir(self, p, radius: float = 1.0):
        """Unit direction of the wall edge nearest ``p`` (xz plane)."""
        e = self.boundary_edges()
        if not len(e):
            return None
        mid = e.mean(axis=1)
        near = np.where(np.abs(mid[:, 0] - p[0]) + np.abs(mid[:, 2] - p[2]) < radius * 4)[0]
        if not len(near):
            near = np.arange(len(e))
        seg = e[near]
        d = _point_seg_dist_xz(seg, p)
        k = int(np.argmin(d))
        v = seg[k, 1] - seg[k, 0]
        v[1] = 0.0
        n = np.linalg.norm(v)
        return None if n < 1e-9 else v / n


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
_EMPTY = np.zeros(0, dtype=np.int64)
_EMPTY_F = np.zeros(0, dtype=float)


def _bary_xz(t: np.ndarray, x: float, z: float) -> np.ndarray:
    """Barycentric coords of ``(x, z)`` in each triangle's xz projection."""
    ax, az = t[:, 0, 0], t[:, 0, 2]
    bx, bz = t[:, 1, 0], t[:, 1, 2]
    cx, cz = t[:, 2, 0], t[:, 2, 2]
    d = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz)
    d = np.where(np.abs(d) < 1e-12, 1e-12, d)
    w0 = ((bz - cz) * (x - cx) + (cx - bx) * (z - cz)) / d
    w1 = ((cz - az) * (x - cx) + (ax - cx) * (z - cz)) / d
    return np.stack([w0, w1, 1.0 - w0 - w1], axis=1)


def _tri_areas(t: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)


def _closest_on_tris(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Closest point to ``p`` on each triangle (3D), vectorised."""
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = np.where(np.abs(va + vb + vc) < 1e-20, 1e-20, va + vb + vc)
    v = np.clip(vb / denom, 0.0, 1.0)
    w = np.clip(vc / denom, 0.0, 1.0)
    s = v + w
    over = s > 1.0
    v = np.where(over, v / np.where(s == 0, 1.0, s), v)
    w = np.where(over, w / np.where(s == 0, 1.0, s), w)
    out = a + v[:, None] * ab + w[:, None] * ac
    # vertex/edge regions handled by the clamp above; refine against corners
    for corner, cond in ((a, (d1 <= 0) & (d2 <= 0)),
                         (b, (d3 >= 0) & (d4 <= d3)),
                         (c, (d6 >= 0) & (d5 <= d6))):
        out = np.where(cond[:, None], corner, out)
    return out


def _point_seg_dist_xz(seg: np.ndarray, p: np.ndarray) -> np.ndarray:
    a = seg[:, 0][:, [0, 2]]
    b = seg[:, 1][:, [0, 2]]
    q = np.asarray(p, dtype=float)[[0, 2]]
    ab = b - a
    L = np.einsum("ij,ij->i", ab, ab)
    L = np.where(L < 1e-18, 1e-18, L)
    t = np.clip(np.einsum("ij,ij->i", q - a, ab) / L, 0.0, 1.0)
    return np.linalg.norm(a + t[:, None] * ab - q, axis=1)


def forward_xz(yaw: float) -> np.ndarray:
    """Horizontal forward vector for ``yaw`` radians (habitat convention)."""
    return np.array([-math.sin(yaw), 0.0, -math.cos(yaw)])


def yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    """``(w, x, y, z)`` for a yaw-only rotation about +y."""
    return (math.cos(yaw / 2.0), 0.0, math.sin(yaw / 2.0), 0.0)


def quat_to_yaw(w: float, x: float, y: float, z: float) -> float:
    """Inverse of :func:`yaw_to_quat`, tolerant of full quaternions.

    Reads the heading off the rotated forward vector, so a quaternion carrying
    some pitch/roll still yields the horizontal heading.
    """
    fx = -2.0 * (x * z + w * y)              # forward = R @ (0, 0, -1)
    fz = -(1.0 - 2.0 * (x * x + y * y))
    return math.atan2(-fx, -fz)
