"""A* path planning on the usd_rooms free grid, to build physically-executable
golden references (the official GVLN golden was authored on furniture-passable
occupancy and clips furniture colliders under the physical modes).

Plans on the same radius-eroded passable mask ``OccMap.is_free`` implies
(``free & clearance_m > radius``), 8-connected with Euclidean step costs, then
The default ``astar`` mode is the historical shortest path.  ``ecm`` mode uses
the clearance objective from mfm-sim-env's Explicit Corridor Map planner on
the complete 5 cm grid: length + a soft ESDF penalty below 0.6 m.  Searching
the full grid avoids sparse-graph bridge failures while preserving the same
centering behaviour.  Its simplifier also preserves clearance cost; the old
unconstrained LOS string-pull would otherwise pull a centred route back to a
wall-hugging chord.

Reuses ``OccMap`` (``to_pixel``/``to_world``/``clearance_m``/``seg_free``); the
only new bit is the grid search — the repo had no A*/BFS.
"""

from __future__ import annotations

import heapq
import hashlib
from collections import OrderedDict
from threading import RLock
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np

from .occmap import OccMap

_SQRT2 = math.sqrt(2.0)
# 8-neighbour offsets (dr, dc, step-cost)
_NBRS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, _SQRT2), (-1, 1, _SQRT2), (1, -1, _SQRT2), (1, 1, _SQRT2)]


@dataclass(frozen=True)
class CenteredAStarConfig:
    """ECM-compatible knobs, matching the reference branch's test defaults."""

    c_pref_m: float = 0.6
    w_length: float = 1.0
    w_clearance: float = 0.8
    alpha: float = 2.0
    max_detour_ratio: Optional[float] = 0.25
    shortcut_cost_ratio: float = 1.01


@dataclass(frozen=True)
class PathSmoothingConfig:
    """Collision-aware post-processing for the centred A* polyline.

    The curve stages mirror ``scan_planner_navmesh``: centripetal Catmull-Rom,
    a short moving average and equal-arclength resampling.  Unlike the reference
    point-wise snap repair, this implementation validates every output segment
    with the same radius-aware ``OccMap`` oracle used by planning.  If the full
    curve is unsafe it blends back towards the original polyline; strength zero
    is an exact, safe fallback rather than a best-effort repair.
    """

    catmull_alpha: float = 0.5
    samples_per_segment: int = 8
    position_window: int = 5
    output_step_m: float = 0.10
    blend_strengths: Tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.0)
    max_length_ratio: float = 1.05
    max_clearance_loss_m: float = 0.05


def _clearance_penalty(clearance_m: float, cfg: CenteredAStarConfig) -> float:
    if clearance_m >= cfg.c_pref_m:
        return 0.0
    return ((cfg.c_pref_m - max(0.0, clearance_m)) / cfg.c_pref_m) ** cfg.alpha


def passable_grid(occ: OccMap, radius_m: float) -> np.ndarray:
    """Cells whose centre the robot can occupy: free AND clearance > radius."""
    cm = getattr(occ, "clearance_m", None)
    free = np.asarray(occ.free, dtype=bool)
    if cm is None:
        return free
    return free & (np.asarray(cm) > radius_m)


def _nearest_passable(passable: np.ndarray, rc: Tuple[int, int],
                      limit_px: int = 40) -> Optional[Tuple[int, int]]:
    """Nearest True cell to ``rc`` within a square radius (spiral by ring)."""
    r0, c0 = rc
    H, W = passable.shape
    if 0 <= r0 < H and 0 <= c0 < W and passable[r0, c0]:
        return rc
    for rad in range(1, limit_px + 1):
        best = None
        best_d = None
        rlo, rhi = max(0, r0 - rad), min(H - 1, r0 + rad)
        clo, chi = max(0, c0 - rad), min(W - 1, c0 + rad)
        sub = passable[rlo:rhi + 1, clo:chi + 1]
        if not sub.any():
            continue
        ys, xs = np.nonzero(sub)
        ys = ys + rlo
        xs = xs + clo
        d2 = (ys - r0) ** 2 + (xs - c0) ** 2
        i = int(np.argmin(d2))
        return (int(ys[i]), int(xs[i]))
    return None


def _astar_grid_search(passable, start, goal, heuristics, neighbors):
    """Original heap, predecessor and tie ordering, with optional exact lookup."""
    H, W = passable.shape
    if not (passable[start] and passable[goal]):
        return None
    gr, gc = goal
    g = {start: 0.0}
    came: dict = {}
    h0 = (math.hypot(start[0] - gr, start[1] - gc) if heuristics is None
          else heuristics[abs(start[0] - gr), abs(start[1] - gc)])
    openh = [(h0, 0.0, start)]
    closed = np.zeros((H, W), dtype=np.bool_)
    while openh:
        _, gc_cur, cur = heapq.heappop(openh)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path
        cr, cc = cur
        if closed[cr, cc]:
            continue
        closed[cr, cc] = True
        for dr, dc, step in neighbors:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < H and 0 <= nc < W) or not passable[nr, nc] \
                    or closed[nr, nc]:
                continue
            # forbid corner-cutting between two diagonal obstacles
            if dr and dc and not (passable[cr, nc] and passable[nr, cc]):
                continue
            ng = gc_cur + step
            nb = (nr, nc)
            if ng < g.get(nb, math.inf):
                g[nb] = ng
                came[nb] = cur
                h = (math.hypot(nr - gr, nc - gc) if heuristics is None
                     else heuristics[abs(nr - gr), abs(nc - gc)])
                f = ng + h
                heapq.heappush(openh, (f, ng, nb))
    return None


@lru_cache(maxsize=1)
def _compiled_grid_search():
    # Reuse the existing optional, version-compatible native Numba capability.
    # No writable JIT cache or relaxed floating-point arithmetic is introduced.
    from native_resource_search import _compiled_frontier_insert
    if _compiled_frontier_insert() is None:
        return None
    from numba import njit
    return njit(cache=False, fastmath=False, nogil=True)(_astar_grid_search)


@lru_cache(maxsize=2)
def _exact_grid_hypot(height, width):
    # CPython and LLVM hypot can round differently. Preserve CPython's exact
    # original values, so symmetric ties choose the same predecessor/path.
    table = np.fromiter((math.hypot(row, column) for row in range(height)
                         for column in range(width)), dtype=np.float64,
                        count=height*width).reshape(height, width)
    table.flags.writeable = False
    return table


def astar_grid(passable: np.ndarray, start: Tuple[int, int],
               goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """8-connected A*; preserve original costs, corners and canonical ties."""
    eligible = (isinstance(passable, np.ndarray) and passable.ndim == 2
        and passable.dtype == np.bool_ and passable.size <= 16*1024*1024
        and all(isinstance(point, tuple) and len(point) == 2
                and all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
                        and 0 <= v < limit for v, limit in zip(point, passable.shape))
                for point in (start, goal)))
    kernel = _compiled_grid_search() if eligible else None
    if kernel is not None:
        start, goal = tuple(map(int, start)), tuple(map(int, goal))
        result = kernel(passable, start, goal, _exact_grid_hypot(*passable.shape), tuple(_NBRS))
        return list(result) if result is not None else None
    # Larger/custom grids and environments without Numba retain the full
    # original search. The table bound limits cache memory, never search labels.
    return _astar_grid_search(passable, start, goal, None, _NBRS)


def _centered_astar_grid_reference(passable: np.ndarray, clearance_m: np.ndarray,
                        start: Tuple[int, int], goal: Tuple[int, int],
                        mpp: float, config: Optional[CenteredAStarConfig] = None
                        ) -> Optional[List[Tuple[int, int]]]:
    """Full-grid A* with the ECM clearance-weighted edge objective."""
    cfg = config or CenteredAStarConfig()
    H, W = passable.shape
    if not (passable[start] and passable[goal]):
        return None
    gr, gc = goal
    g = {start: 0.0}
    came: dict = {}
    h0 = cfg.w_length * mpp * math.hypot(start[0] - gr, start[1] - gc)
    openh = [(h0, 0.0, start)]
    closed = np.zeros((H, W), dtype=bool)
    while openh:
        _, gc_cur, cur = heapq.heappop(openh)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path
        cr, cc = cur
        if closed[cr, cc]:
            continue
        closed[cr, cc] = True
        for dr, dc, step in _NBRS:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < H and 0 <= nc < W) or not passable[nr, nc] \
                    or closed[nr, nc]:
                continue
            if dr and dc and not (passable[cr, nc] and passable[nr, cc]):
                continue
            edge_m = step * mpp
            edge_clearance = min(float(clearance_m[cr, cc]),
                                 float(clearance_m[nr, nc]))
            factor = (cfg.w_length
                      + cfg.w_clearance * _clearance_penalty(edge_clearance, cfg))
            ng = gc_cur + edge_m * factor
            nb = (nr, nc)
            if ng < g.get(nb, math.inf):
                g[nb] = ng
                came[nb] = cur
                heuristic = cfg.w_length * mpp * math.hypot(nr - gr, nc - gc)
                heapq.heappush(openh, (ng + heuristic, ng, nb))
    return None



def _centered_grid_search(passable, clearance_m, factors, start, goal, mpp, w_length, heuristics, neighbors):
    """Original heap/cost/predecessor ordering, using exact Python lookup values."""
    H, W = passable.shape
    if not (passable[start] and passable[goal]):
        return None
    gr, gc = goal
    g = {start: 0.0}
    came: dict = {}
    h0 = w_length * mpp * heuristics[abs(start[0] - gr), abs(start[1] - gc)]
    openh = [(h0, 0.0, start)]
    closed = np.zeros((H, W), dtype=np.bool_)
    while openh:
        _, gc_cur, cur = heapq.heappop(openh)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path
        cr, cc = cur
        if closed[cr, cc]:
            continue
        closed[cr, cc] = True
        for dr, dc, step in neighbors:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < H and 0 <= nc < W) or not passable[nr, nc] \
                    or closed[nr, nc]:
                continue
            if dr and dc and not (passable[cr, nc] and passable[nr, cc]):
                continue
            edge_m = step * mpp
            factor = (factors[cr, cc] if clearance_m[cr, cc] <= clearance_m[nr, nc]
                      else factors[nr, nc])
            ng = gc_cur + edge_m * factor
            nb = (nr, nc)
            if ng < g.get(nb, math.inf):
                g[nb] = ng
                came[nb] = cur
                heuristic = w_length * mpp * heuristics[abs(nr - gr), abs(nc - gc)]
                heapq.heappush(openh, (ng + heuristic, ng, nb))
    return None


@lru_cache(maxsize=1)
def _compiled_centered_grid_search():
    if _compiled_grid_search() is None:
        return None
    from numba import njit
    return njit(cache=False, fastmath=False, nogil=True)(_centered_grid_search)



# Bounded content-based LRU, as used for native floor rasters and image bytes.
# Rechecking every byte permits mutable callers without stale cost reuse.
_CENTERED_FACTOR_MAX_ENTRIES = 8
_CENTERED_FACTOR_MAX_BYTES = 256 * 1024 * 1024
_centered_factors = OrderedDict()
_centered_factor_bytes = 0
_centered_factor_lock = RLock()


def _centered_cost_factors(clearance_m, cfg, *, segment_penalties=False):
    global _centered_factor_bytes
    key = (clearance_m.shape, clearance_m.dtype.str,
           hashlib.sha256(clearance_m.tobytes()).digest(),
           *(float(value).hex() for value in
             (cfg.w_length, cfg.w_clearance, cfg.c_pref_m, cfg.alpha)),
           segment_penalties)
    with _centered_factor_lock:
        factors = _centered_factors.get(key)
        if factors is not None:
            _centered_factors.move_to_end(key)
            return factors
    values, inverse = np.unique(clearance_m, return_inverse=True)
    if segment_penalties:
        # Match OccMap.clearance's default cap and CPython pow exactly. Do not
        # recover penalties from weighted edge factors: that loses cost bits.
        factor_values = np.fromiter((_clearance_penalty(min(float(value), 2.), cfg)
            for value in values), dtype=np.float64, count=len(values))
    else:
        factor_values = np.fromiter((cfg.w_length + cfg.w_clearance *
            _clearance_penalty(float(value), cfg) for value in values),
            dtype=np.float64, count=len(values))
    factors = factor_values[inverse].reshape(clearance_m.shape)
    factors.setflags(write=False)
    with _centered_factor_lock:
        cached = _centered_factors.get(key)
        if cached is not None:
            _centered_factors.move_to_end(key)
            return cached
        if factors.nbytes <= _CENTERED_FACTOR_MAX_BYTES:
            _centered_factors[key] = factors
            _centered_factor_bytes += factors.nbytes
            while (len(_centered_factors) > _CENTERED_FACTOR_MAX_ENTRIES or
                   _centered_factor_bytes > _CENTERED_FACTOR_MAX_BYTES):
                _, discarded = _centered_factors.popitem(last=False)
                _centered_factor_bytes -= discarded.nbytes
    return factors

def centered_astar_grid(passable: np.ndarray, clearance_m: np.ndarray,
                        start: Tuple[int, int], goal: Tuple[int, int],
                        mpp: float, config: Optional[CenteredAStarConfig] = None
                        ) -> Optional[List[Tuple[int, int]]]:
    """Preserve the ECM route and costs while releasing the GIL during search."""
    cfg = config or CenteredAStarConfig()
    eligible = (isinstance(passable, np.ndarray) and passable.ndim == 2
        and passable.dtype == np.bool_ and passable.size <= 16*1024*1024
        and isinstance(clearance_m, np.ndarray) and clearance_m.shape == passable.shape
        and clearance_m.dtype.kind == 'f' and np.isfinite(clearance_m).all()
        and all(math.isfinite(float(v)) for v in
                (mpp, cfg.w_length, cfg.w_clearance, cfg.c_pref_m, cfg.alpha))
        and cfg.c_pref_m > 0
        and all(isinstance(point, tuple) and len(point) == 2
                and all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
                        and 0 <= v < limit for v, limit in zip(point, passable.shape))
                for point in (start, goal)))
    kernel = _compiled_centered_grid_search() if eligible else None
    if kernel is None:
        return _centered_astar_grid_reference(passable, clearance_m, start, goal, mpp, cfg)
    # Each edge uses the lower endpoint clearance. Calculate its factor with
    # CPython once per distinct value; LLVM pow/hypot rounding cannot alter ties.
    factors = _centered_cost_factors(clearance_m, cfg)
    result = kernel(passable, clearance_m, factors, tuple(map(int, start)),
        tuple(map(int, goal)), mpp, cfg.w_length, _exact_grid_hypot(*passable.shape), tuple(_NBRS))
    return list(result) if result is not None else None


def _string_pull(occ: OccMap, pts_xy: List[Tuple[float, float]],
                 radius_m: float) -> List[Tuple[float, float]]:
    """Greedy line-of-sight simplification: from each kept point, jump to the
    farthest later point still reachable by a ``seg_free`` straight segment."""
    if len(pts_xy) <= 2:
        return list(pts_xy)
    out = [pts_xy[0]]
    i = 0
    n = len(pts_xy)
    while i < n - 1:
        j = n - 1
        while j > i + 1 and not occ.seg_free(*pts_xy[i], *pts_xy[j], radius_m):
            j -= 1
        out.append(pts_xy[j])
        i = j
    return out


def _segment_objective(occ: OccMap, a: Tuple[float, float],
                       b: Tuple[float, float], cfg: CenteredAStarConfig) -> float:
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    if length < 1e-9:
        return 0.0
    n = max(2, int(math.ceil(length / max(occ.scale, 0.02))) + 1)
    penalties = []
    for k in range(n):
        t = k / float(n - 1)
        penalties.append(_clearance_penalty(
            occ.clearance(a[0] + (b[0] - a[0]) * t,
                          a[1] + (b[1] - a[1]) * t), cfg))
    return length * (cfg.w_length + cfg.w_clearance * float(np.mean(penalties)))


def _segment_objective_evaluator(occ, cfg):
    """Reuse exact grid penalties for one read-only simplification operation.

    Custom coordinate/clearance implementations retain the scalar oracle.
    Geometry is rebound by content for each operation, never by object id.
    """
    reference = lambda a, b: _segment_objective(occ, a, b, cfg)
    cm = getattr(occ, 'clearance_m', None)
    if not (getattr(occ.clearance, '__func__', None) is OccMap.clearance
            and getattr(occ.to_pixel, '__func__', None) is OccMap.to_pixel
            and isinstance(cm, np.ndarray) and cm.dtype.kind == 'f'
            and cm.shape == occ.free.shape == (occ.h, occ.w)
            and np.isfinite(cm).all()
            and math.isfinite(float(occ.scale)) and occ.scale > 0
            and np.isfinite(occ.lower[:2]).all()
            and all(math.isfinite(float(v)) for v in
                    (cfg.w_length, cfg.w_clearance, cfg.c_pref_m, cfg.alpha))
            and cfg.c_pref_m > 0):
        return reference
    grid = _centered_cost_factors(cm, cfg, segment_penalties=True)
    blocked_penalty = _clearance_penalty(0., cfg)

    def objective(a, b):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        if length < 1e-9:
            return 0.0
        n = max(2, int(math.ceil(length / max(occ.scale, 0.02))) + 1)
        # Tiny segments and unusual coordinates keep the established oracle.
        if n < 8 or n > 1000000:
            return reference(a, b)
        t = np.arange(n, dtype=np.float64) / float(n - 1)
        x = a[0] + (b[0] - a[0]) * t
        y = a[1] + (b[1] - a[1]) * t
        col = (x - occ.lower[0]) / occ.scale
        row = (occ.h - 1) - (y - occ.lower[1]) / occ.scale
        if not (np.isfinite(col).all() and np.isfinite(row).all()
                and np.max(np.abs(col)) < 2**51
                and np.max(np.abs(row)) < 2**51):
            return reference(a, b)
        # IEEE round-to-even matches the original int(round(float)) pixels.
        cols, rows = np.rint(col).astype(np.int64), np.rint(row).astype(np.int64)
        valid = (cols >= 0) & (cols < occ.w) & (rows >= 0) & (rows < occ.h)
        penalties = np.full(n, blocked_penalty, dtype=np.float64)
        indexes = np.flatnonzero(valid)
        indexes = indexes[occ.free[rows[indexes], cols[indexes]]]
        penalties[indexes] = grid[rows[indexes], cols[indexes]]
        # Retain NumPy's original reduction order and the outer scalar formula.
        return length * (cfg.w_length + cfg.w_clearance * float(np.mean(penalties)))

    return objective


def _centered_string_pull(occ: OccMap, pts_xy: List[Tuple[float, float]],
                          radius_m: float,
                          cfg: CenteredAStarConfig) -> List[Tuple[float, float]]:
    """LOS simplification that cannot increase the ECM objective materially."""
    if len(pts_xy) <= 2:
        return list(pts_xy)
    objective = _segment_objective_evaluator(occ, cfg)
    prefix = [0.0]
    for i in range(len(pts_xy) - 1):
        prefix.append(prefix[-1] + objective(pts_xy[i], pts_xy[i + 1]))
    out = [pts_xy[0]]
    i = 0
    n = len(pts_xy)
    while i < n - 1:
        chosen = i + 1
        for j in range(n - 1, i + 1, -1):
            original_cost = prefix[j] - prefix[i]
            shortcut_cost = objective(pts_xy[i], pts_xy[j])
            # The objective is read-only and usually a vectorized grid lookup.
            # Reject costly shortcuts before native navmesh segment sampling;
            # admitted shortcuts still pass the identical collision oracle.
            if (shortcut_cost <= original_cost * cfg.shortcut_cost_ratio + 1e-6
                    and occ.seg_free(*pts_xy[i], *pts_xy[j], radius_m)):
                chosen = j
                break
        out.append(pts_xy[chosen])
        i = chosen
    return out


def _grid_path_length(grid: List[Tuple[int, int]], mpp: float) -> float:
    return sum(math.hypot(grid[i + 1][0] - grid[i][0],
                          grid[i + 1][1] - grid[i][1]) * mpp
               for i in range(len(grid) - 1))


def _polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _resample_polyline(points: np.ndarray, *, step_m: Optional[float] = None,
                       count: Optional[int] = None,
                       preserve_vertices: bool = True) -> np.ndarray:
    """Equal-arclength resampling with exact endpoint preservation.

    For a step-based final path, original vertices are included in the target
    arc lengths.  A plain global ``linspace`` can put one sample immediately
    before a sharp corner and the next immediately after it, silently replacing
    the two safe legs by an unsafe diagonal chord.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate(([True], seg > 1e-9))
    pts = pts[keep]
    if len(pts) < 2:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cum[-1])
    if count is None:
        count = max(2, int(math.ceil(total / max(float(step_m or 0.1), 1e-6))) + 1)
    targets = np.linspace(0.0, total, max(2, int(count)))
    if preserve_vertices and step_m is not None:
        targets = np.unique(np.concatenate((targets, cum)))
    out = np.column_stack([np.interp(targets, cum, pts[:, axis])
                           for axis in range(pts.shape[1])])
    out[0], out[-1] = pts[0], pts[-1]
    return out


def _catmull_rom_chain(points: np.ndarray, alpha: float,
                       samples_per_segment: int) -> np.ndarray:
    """Centripetal Catmull-Rom curve through all 2-D control points."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return _resample_polyline(pts, count=max(2, samples_per_segment + 1))
    phantom = np.vstack((2.0 * pts[0] - pts[1], pts,
                         2.0 * pts[-1] - pts[-2]))
    out = [pts[0]]
    density = max(2, int(samples_per_segment))
    for i in range(len(pts) - 1):
        p0, p1, p2, p3 = phantom[i:i + 4]
        d01 = max(float(np.linalg.norm(p1 - p0)) ** alpha, 1e-10)
        d12 = max(float(np.linalg.norm(p2 - p1)) ** alpha, 1e-10)
        d23 = max(float(np.linalg.norm(p3 - p2)) ** alpha, 1e-10)
        t0, t1, t2, t3 = 0.0, d01, d01 + d12, d01 + d12 + d23
        for sample in range(1, density + 1):
            t = t1 + (t2 - t1) * (sample / float(density))
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1
                       + (t - t1) / (t2 - t1) * b2)
    curve = np.asarray(out, dtype=np.float64)
    curve[0], curve[-1] = pts[0], pts[-1]
    return curve


def _smooth_positions(points: np.ndarray, window: int) -> np.ndarray:
    """Edge-padded moving average, preserving the route endpoints."""
    pts = np.asarray(points, dtype=np.float64)
    window = min(max(1, int(window)), len(pts))
    if window % 2 == 0:
        window -= 1
    if len(pts) < 3 or window <= 1:
        return pts.copy()
    pad = window // 2
    padded = np.pad(pts, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    out = np.column_stack([np.convolve(padded[:, axis], kernel, mode="valid")
                           for axis in range(2)])
    out[0], out[-1] = pts[0], pts[-1]
    return out


def _path_min_clearance(occ: OccMap, points: np.ndarray) -> float:
    return min((occ.clearance(float(x), float(y)) for x, y in points),
               default=0.0)


def _path_is_safe(occ: OccMap, points: np.ndarray, radius_m: float) -> bool:
    if len(points) < 2:
        return False
    if not all(occ.is_free(float(x), float(y), radius_m) for x, y in points):
        return False
    return all(occ.seg_free(float(a[0]), float(a[1]), float(b[0]), float(b[1]),
                            radius_m) for a, b in zip(points, points[1:]))


def smooth_path(occ: OccMap, points, radius_m: float,
                config: Optional[PathSmoothingConfig] = None
                ) -> Tuple[List[Tuple[float, float]], dict]:
    """Round an A*/ECM polyline without weakening its collision contract.

    Returns the smoothed path plus provenance useful for production audits.
    Endpoints always match the input exactly.  The output is sampled at about
    ``output_step_m`` so downstream replay and yaw generation see the curve,
    rather than reconstructing the original sparse corner geometry.
    """
    cfg = config or PathSmoothingConfig()
    raw = np.asarray([[float(p[0]), float(p[1])] for p in points], dtype=np.float64)
    if len(raw) < 3:
        return [tuple(map(float, p)) for p in raw], {
            "applied": False, "reason": "short_path", "blend_strength": 0.0,
            "input_points": int(len(raw)), "output_points": int(len(raw)),
        }
    raw_dense = _resample_polyline(raw, step_m=cfg.output_step_m)
    # Validate the exact planner segments.  Re-segmenting a corner at different
    # floating-point sample phases can be stricter than the source segment in a
    # raster map, even though every interpolated point is free; the production
    # replay below remains the authority for the eventual 0.1 m motion chunks.
    if not _path_is_safe(occ, raw, radius_m):
        raise ValueError("input path violates the smoothing collision radius")

    curve = _catmull_rom_chain(raw, cfg.catmull_alpha, cfg.samples_per_segment)
    curve = _smooth_positions(curve, cfg.position_window)
    curve = _resample_polyline(curve, step_m=cfg.output_step_m,
                               preserve_vertices=False)
    baseline = _resample_polyline(raw, count=len(curve), preserve_vertices=False)
    raw_length = _polyline_length(raw)
    raw_clearance = _path_min_clearance(occ, raw_dense)
    min_allowed_clearance = max(float(radius_m),
                                raw_clearance - cfg.max_clearance_loss_m)

    accepted = None
    accepted_strength = 0.0
    for strength in cfg.blend_strengths:
        strength = float(strength)
        candidate = baseline + strength * (curve - baseline)
        candidate[0], candidate[-1] = raw[0], raw[-1]
        # The candidate is already a smooth curve sampled in common arc-length
        # coordinates; a second vertex-preserving union would nearly double the
        # requested 0.1 m density without adding safety.
        candidate = _resample_polyline(candidate, step_m=cfg.output_step_m,
                                       preserve_vertices=False)
        length = _polyline_length(candidate)
        if length > raw_length * cfg.max_length_ratio + 1e-6:
            continue
        if not _path_is_safe(occ, candidate, radius_m):
            continue
        if _path_min_clearance(occ, candidate) + 1e-6 < min_allowed_clearance:
            continue
        accepted, accepted_strength = candidate, strength
        break
    if accepted is None:
        # Preserve the exact known-safe planner polyline.  The builder's swept
        # check and exact 0.1 m execution replay still run after this fallback.
        accepted = raw
        accepted_strength = 0.0

    out_length = _polyline_length(accepted)
    out_clearance = _path_min_clearance(occ, accepted)
    result = [tuple(map(float, p)) for p in accepted]
    return result, {
        "applied": bool(accepted_strength > 0.0),
        "blend_strength": accepted_strength,
        "input_points": int(len(raw)),
        "output_points": int(len(result)),
        "input_length_m": round(raw_length, 4),
        "output_length_m": round(out_length, 4),
        "input_min_clearance_m": round(raw_clearance, 4),
        "output_min_clearance_m": round(out_clearance, 4),
    }


# Eight-neighbour connectivity is a superset of the search edges (the search
# also forbids corner cutting). Different labels prove rejection; equal labels
# never prove a valid route. Reuse the dataset filter's component kernel.
_COMPONENT_MAX_ENTRIES = 8
_COMPONENT_MAX_BYTES = 256 * 1024 * 1024
_components_cache = OrderedDict()
_component_bytes = 0
_component_lock = RLock()


def _cached_components(passable):
    global _component_bytes
    from .filter_reachable import _components
    key = (passable.shape, passable.dtype.str,
           hashlib.sha256(passable.tobytes()).digest())
    with _component_lock:
        labels = _components_cache.get(key)
        if labels is not None:
            _components_cache.move_to_end(key)
            return labels
    labels = _components(passable)
    if labels is None:
        return None
    labels.setflags(write=False)
    with _component_lock:
        existing = _components_cache.get(key)
        if existing is not None:
            _components_cache.move_to_end(key)
            return existing
        if labels.nbytes <= _COMPONENT_MAX_BYTES:
            _components_cache[key] = labels
            _component_bytes += labels.nbytes
            while (len(_components_cache) > _COMPONENT_MAX_ENTRIES or
                   _component_bytes > _COMPONENT_MAX_BYTES):
                _, discarded = _components_cache.popitem(last=False)
                _component_bytes -= discarded.nbytes
    return labels


def _provably_disconnected(passable, start, goal):
    if (not isinstance(passable, np.ndarray) or passable.ndim != 2 or
            passable.dtype != np.bool_ or passable.size > 16 * 1024 * 1024):
        return False
    labels = _cached_components(passable)
    return labels is not None and labels[start] != labels[goal]


def plan_path(occ: OccMap, start_xy, goal_xy, radius_m: float = 0.07,
              simplify: bool = True, mode: str = "astar",
              centered_config: Optional[CenteredAStarConfig] = None
              ) -> Optional[List[Tuple[float, float]]]:
    """A* golden from ``start_xy`` to ``goal_xy`` in world metres, or ``None``.

    Endpoints are snapped to the nearest passable cell (kept episodes already
    satisfy ``is_free``, so this is usually a no-op).
    """
    passable = passable_grid(occ, radius_m)
    sc, sr = occ.to_pixel(start_xy[0], start_xy[1])
    gc_, gr = occ.to_pixel(goal_xy[0], goal_xy[1])
    s = _nearest_passable(passable, (sr, sc))
    gpt = _nearest_passable(passable, (gr, gc_))
    if s is None or gpt is None:
        return None
    mode = str(mode).strip().lower()
    if mode not in ("astar", "ecm"):
        raise ValueError("planner mode must be astar or ecm, got %r" % mode)
    cfg = centered_config or CenteredAStarConfig()
    clearance = getattr(occ, "clearance_m", None)
    # Keep unusual/invalid ECM inputs on the original validation/search path.
    normal_ecm = (mode == "astar" or clearance is None or
        (isinstance(clearance, np.ndarray) and clearance.shape == passable.shape
         and clearance.dtype.kind == 'f' and np.isfinite(clearance).all()
         and all(math.isfinite(float(v)) for v in
                 (occ.scale, cfg.w_length, cfg.w_clearance, cfg.c_pref_m, cfg.alpha))
         and cfg.c_pref_m > 0))
    if (mode == "astar" or normal_ecm) and _provably_disconnected(passable, s, gpt):
        return None
    baseline = None
    if mode == "ecm":
        clearance = getattr(occ, "clearance_m", None)
        if clearance is None:
            mode = "astar"
        else:
            grid = centered_astar_grid(passable, np.asarray(clearance), s, gpt,
                                       float(occ.scale), cfg)
            if grid is not None and cfg.max_detour_ratio is not None:
                baseline = astar_grid(passable, s, gpt)
                if baseline is not None and _grid_path_length(grid, occ.scale) > \
                        _grid_path_length(baseline, occ.scale) * (1.0 + cfg.max_detour_ratio):
                    grid = baseline
                    mode = "astar"
    if mode == "astar":
        grid = baseline if baseline is not None else astar_grid(passable, s, gpt)
    if grid is None:
        return None
    pts = [occ.to_world(c, r) for (r, c) in grid]
    # keep true endpoints (not the snapped cell centres)
    pts[0] = (float(start_xy[0]), float(start_xy[1]))
    pts[-1] = (float(goal_xy[0]), float(goal_xy[1]))
    if simplify:
        pts = (_centered_string_pull(occ, pts, radius_m, cfg)
               if mode == "ecm" else _string_pull(occ, pts, radius_m))
    repair = getattr(occ, 'repair_grid_path', None)
    if callable(repair):
        if not _path_is_safe(occ, np.asarray(pts), radius_m):
            return repair(start_xy, goal_xy, radius_m)
    return pts
