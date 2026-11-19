"""Ground-height correction for ABotN scenes, measured off the Gaussians.

Why this exists
---------------
``map/occ_map_height.tiff`` is the only ground reference ABotN ships, and it is
**8-bit greyscale** -- a plain ``uint8``, no scale or offset anywhere in
``occ_map_meta.txt``.  So the ground it reports is quantised to whole metres,
and the camera, which ABotN places at ``ground_z + 0.65``, inherits up to half a
metre of error.  Most of the time that is invisible: the floor in these scenes
sits near z = 0, the tiff says 0, and the eye ends up at a plausible 0.65 m.
Where the real floor sits near a step in the quantisation it is very visible ---
飘宿's ep0 spawns on ground the cloud puts at +0.68 while the tiff says 0, so the
eye lands 0.03 m *below* the floor and you start the episode inside the pavement.

The fix is to measure the floor where we actually have sub-metre geometry -- the
Gaussian cloud, ``gs/*/point_cloud_rotated.ply`` -- and keep only the *difference*
against the tiff:

    delta(x, y) = floor_from_cloud(x, y) - ground_z_from_tiff(x, y)

A correction rather than a replacement, and a clipped, one-directional one, on
purpose.  The clouds are photogrammetry: cells that caught only an awning, a
tree canopy or a passing car have a "floor" metres off, in three PointBench
scenes the cloud covers barely a fifth of the walkable area, and even a good
cell's low percentile sits a little *under* the visible surface because of the
skirt of splats below it.  So the correction is clipped to ``TRUST`` (1.5 m) and
only ever applied when it *lifts* the camera (see :meth:`Delta.at`): standing
below the floor is a visibly broken frame, standing a bit high is just a taller
robot.  Worst case the whole thing degrades to ABotN's own number.

Cost and caching
----------------
The clouds are 0.4-1.6 GB each, 36.5 GB over the 42 scenes, so this is not
something to do per run: the grid is a few tens of kB, cached as ``.npz`` under
``$ABOTN_FLOOR_CACHE`` (default ``$ABOTN_ROOT/.floor_cache``).  Building one takes
1-20 s depending on cloud size, and happens lazily on first use of a scene if the
cache is cold.  Warm the whole cache up front with::

    python -m abotn.floorfit            # all three benchmarks
    python -m abotn.floorfit poi        # or just one

Set ``ABOTN_FLOOR_FIT=0`` to switch the correction off and get ABotN's raw
height-map behaviour back.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import numpy as np

CELL = 1.0        # metres per correction cell; the tiff's own quantisation step
PCT = 20.0        # percentile of z within a cell that counts as its floor
MIN_PTS = 8       # cells with fewer splats than this are not evidence
TRUST = 1.5       # metres; corrections beyond this are cloud noise, not ground
FILL_ITERS = 12   # how far (in cells) a measured correction spreads into holes
SMOOTH = 2        # 3x3 mean passes -- also what removes the tiff's 1 m staircase
VERSION = 1       # bump to invalidate every cached grid


def _cache_root() -> str:
    from . import datasets
    return os.environ.get("ABOTN_FLOOR_CACHE") \
        or os.path.join(datasets.ZYTOSS, ".floor_cache")


def enabled() -> bool:
    return os.environ.get("ABOTN_FLOOR_FIT", "1") not in ("0", "no", "false")


# -- reading the cloud -----------------------------------------------------
def _ply_header(f):
    hdr = b""
    while b"end_header" not in hdr:
        line = f.readline()
        if not line:
            raise ValueError("truncated ply header")
        hdr += line
    n = 0
    props = []
    fmt = None
    for ln in hdr.decode("ascii", "replace").splitlines():
        p = ln.split()
        if not p:
            continue
        if p[0] == "format":
            fmt = p[1]
        elif p[0] == "element" and p[1] == "vertex":
            n = int(p[2])
        elif p[0] == "property":
            props.append((p[1], p[2]))
    tmap = {"float": "f4", "float32": "f4", "double": "f8", "uchar": "u1",
            "int": "i4", "uint": "u4", "short": "i2", "ushort": "u2", "char": "i1"}
    return np.dtype([(nm, tmap[t]) for t, nm in props]), n, fmt


def read_xyz(path: str, chunk: int = 500_000):
    """xyz of every splat centre, read in chunks.

    Chunked because the row dtype is ~250 B wide (62 float properties for a
    degree-3 cloud): slurping 1.6 GB of it at once costs a gigabyte of transient
    RSS inside the console process for three columns we keep.
    """
    with open(path, "rb") as f:
        dt, n, fmt = _ply_header(f)
        if fmt != "binary_little_endian":
            raise ValueError("unexpected ply format %r" % fmt)
        xs, ys, zs = [], [], []
        left = n
        while left > 0:
            a = np.fromfile(f, dtype=dt, count=min(chunk, left))
            if not len(a):
                break
            left -= len(a)
            xs.append(a["x"].astype(np.float32))
            ys.append(a["y"].astype(np.float32))
            zs.append(a["z"].astype(np.float32))
    if not xs:
        raise ValueError("empty ply %s" % path)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return x[ok], y[ok], z[ok]


# -- the correction grid ---------------------------------------------------
class Delta:
    """Ground correction in metres, on a regular world-aligned grid."""

    def __init__(self, grid: np.ndarray, x0: float, y0: float, cell: float,
                 coverage: float = 0.0):
        self.grid = np.asarray(grid, np.float32)
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.cell = float(cell)
        self.coverage = float(coverage)

    def at(self, x: float, y: float) -> float:
        """Correction at ``(x, y)``, in metres.  Never negative -- see below.

        The grid stores the measured *signed* difference, but only the lifting
        half is handed out.  Two reasons, both measured rather than assumed:

        * The asymmetry is real.  A camera below the floor is a broken frame --
          the bottom of the picture is the underside of the pavement, the horizon
          is squeezed to the top edge.  A camera half a metre high is just a
          taller robot; nobody can tell from the image.
        * The cloud's own floor estimate is biased *low*.  Photogrammetry leaves
          a skirt of splats under the real surface, so the low percentile of a
          cell sits below what you can see.  Checked against a third,
          independent measurement -- back-projecting the render server's own
          depth map and reading off the ground plane -- the signed correction
          overshoots downwards: 太田's spawn measures 0.70 m above the visible
          road with ABotN's raw height map and 0.25 m with the signed
          correction, i.e. the correction introduced the very defect it is here
          to remove.  Lifting-only keeps 0.70 and still fixes 飘宿's spawn, which
          measures -0.03 m raw (visibly inside the pavement) and +0.46 m lifted.
        """
        j = int((x - self.x0) / self.cell)
        i = int((y - self.y0) / self.cell)
        if 0 <= i < self.grid.shape[0] and 0 <= j < self.grid.shape[1]:
            return max(0.0, float(self.grid[i, j]))
        return 0.0

    # -- disk ----------------------------------------------------------------
    def save(self, path: str) -> None:
        tmp = path + ".tmp%d" % os.getpid()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        meta = np.array([self.x0, self.y0, self.cell, self.coverage, VERSION],
                        np.float64)
        # write through a handle: savez_compressed would append its own ".npz"
        # to a bare temp name and then os.replace would miss the file
        with open(tmp, "wb") as f:
            np.savez_compressed(f, grid=self.grid, meta=meta)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "Delta":
        with np.load(path) as z:
            m = z["meta"]
            if int(m[4]) != VERSION:
                raise ValueError("stale floor cache")
            return cls(z["grid"], m[0], m[1], m[2], m[3])


def _fill_and_smooth(raw: np.ndarray) -> np.ndarray:
    """Spread measured cells into their neighbourhood, then blur.

    Two jobs in one.  Filling stops the camera from stepping up and down as it
    crosses between measured and unmeasured cells -- a discontinuity is worse to
    watch than a constant offset.  Blurring is what actually erases the tiff's
    1 m staircase, since ``delta`` carries the negative of it.
    """
    cur = raw.copy()
    for _ in range(FILL_ITERS):
        holes = ~np.isfinite(cur)
        if not holes.any():
            break
        s = np.zeros_like(cur)
        n = np.zeros(cur.shape, np.float32)
        val = np.where(np.isfinite(cur), cur, 0.0).astype(np.float32)
        got = np.isfinite(cur).astype(np.float32)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            s += np.roll(val, (di, dj), (0, 1))
            n += np.roll(got, (di, dj), (0, 1))
        near = holes & (n > 0)
        cur[near] = s[near] / n[near]
    out = np.where(np.isfinite(cur), cur, 0.0).astype(np.float32)
    for _ in range(SMOOTH):
        acc = out.copy()
        cnt = np.ones_like(out)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di or dj:
                    acc += np.roll(out, (di, dj), (0, 1))
                    cnt += 1
        out = acc / cnt
    return np.clip(out, -TRUST, TRUST).astype(np.float32)


def build(scene_dir: str, occ=None, verbose: bool = False) -> Optional[Delta]:
    """Measure the correction for one scene.  ``None`` if it has no cloud."""
    import glob
    from . import occmap as occmod

    plys = sorted(glob.glob(os.path.join(scene_dir, "gs", "*",
                                         "point_cloud_rotated.ply")))
    if not plys:
        return None
    if occ is None:
        occ = occmod.OccMap(scene_dir)
    # measure against the *raw* height map even if a correction is already
    # attached, or a rebuild would fold the old delta into the new one
    prev, occ.z_delta = getattr(occ, "z_delta", None), None

    t0 = time.time()
    x, y, z = read_xyz(plys[0])

    # Grid over the occ map's own extent, never the cloud's bounding box: a
    # handful of stray splats put the cloud's bbox kilometres out, and the map
    # extent is both bounded and exactly the region we can stand in anyway.
    cx = [occ.to_world(px, py)[0] for px in (0, occ.w) for py in (0, occ.h)]
    cy = [occ.to_world(px, py)[1] for px in (0, occ.w) for py in (0, occ.h)]
    x0 = float(np.floor(min(cx) / CELL) * CELL)
    y0 = float(np.floor(min(cy) / CELL) * CELL)
    W = int((max(cx) - x0) / CELL) + 2
    H = int((max(cy) - y0) / CELL) + 2

    ix = np.floor((x - x0) / CELL).astype(np.int64)
    iy = np.floor((y - y0) / CELL).astype(np.int64)
    inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    ix, iy, z = ix[inside], iy[inside], z[inside]
    if not len(z):
        return None

    # per-cell PCT-th percentile of z, via one sort of (cell, z)
    key = iy * W + ix
    order = np.lexsort((z, key))
    ks, zs = key[order], z[order]
    uk, start, cnt = np.unique(ks, return_index=True, return_counts=True)
    pick = start + np.minimum(cnt - 1, (cnt * PCT / 100.0).astype(np.int64))
    keep = cnt >= MIN_PTS
    uk, floor = uk[keep], zs[pick][keep]

    # delta against the tiff, at each measured cell's centre
    ci, cj = uk // W, uk % W
    wx = x0 + (cj + 0.5) * CELL
    wy = y0 + (ci + 0.5) * CELL
    tiff = np.array([occ.ground_z(float(a), float(b), default=np.nan)
                     for a, b in zip(wx, wy)], np.float64)
    d = floor - tiff
    good = np.isfinite(d) & (np.abs(d) <= TRUST)

    raw = np.full((H, W), np.nan, np.float32)
    raw[ci[good], cj[good]] = d[good]

    # coverage is quoted over free space, since that is where it matters
    free_cells = np.zeros((H, W), bool)
    fr, fc = np.nonzero(occ.free)
    if len(fr):
        step = max(1, len(fr) // 200_000)
        for r, c in zip(fr[::step], fc[::step]):
            wxx, wyy = occ.to_world(float(c), float(r))
            jj = int((wxx - x0) / CELL)
            ii = int((wyy - y0) / CELL)
            if 0 <= ii < H and 0 <= jj < W:
                free_cells[ii, jj] = True
    cov = float(np.isfinite(raw)[free_cells].mean()) if free_cells.any() else 0.0

    out = Delta(_fill_and_smooth(raw), x0, y0, CELL, cov)
    occ.z_delta = prev
    if verbose:
        print("    %d 点 -> %d 个可信格子, free 覆盖 %.0f%%, 修正量 "
              "p5/中位/p95 = %.2f/%.2f/%.2f m  (%.1fs)"
              % (len(z), int(good.sum()), 100 * cov,
                 float(np.percentile(d[good], 5)) if good.any() else 0.0,
                 float(np.median(d[good])) if good.any() else 0.0,
                 float(np.percentile(d[good], 95)) if good.any() else 0.0,
                 time.time() - t0))
    return out


def cache_path(scene_dir: str) -> str:
    """Flat name under the cache root, so the asset tree stays read-only."""
    from . import datasets
    rel = os.path.relpath(os.path.abspath(scene_dir),
                          os.path.abspath(datasets.ZYTOSS))
    return os.path.join(_cache_root(),
                        rel.replace(os.sep, "__").strip("._") + ".npz")


def delta_for(scene_dir: str, occ=None, log=None) -> Optional[Delta]:
    """Cached correction for a scene; builds it on a cold cache.

    Never raises: a missing cloud, an unreadable ply or a read-only cache root
    all degrade to ``None``, which means "use ABotN's height map as-is".
    """
    if not enabled():
        return None
    p = cache_path(scene_dir)
    try:
        return Delta.load(p)
    except Exception:
        pass
    try:
        d = build(scene_dir, occ=occ)
    except Exception as e:                       # noqa: BLE001 -- never fatal
        if log:
            log("[abotn] 地面修正算不出来，退回原高度图：%s" % e)
        return None
    if d is None:
        return None
    try:
        d.save(p)
    except Exception as e:                       # noqa: BLE001
        if log:
            log("[abotn] 地面修正缓存写不进去（%s），这次现算" % e)
    return d


# -- warm the cache --------------------------------------------------------
def main(argv) -> int:
    from . import datasets
    sets = datasets.available(datasets.ZYTOSS)
    keys = [a for a in argv if a in sets] or list(sets)
    print("缓存目录 %s" % _cache_root())
    for k in keys:
        ds = sets[k]
        for sc in ds.list_scenes():
            d = ds.scene_dir(sc)
            print("[%s] %s" % (k, sc))
            p = cache_path(d)
            if os.path.exists(p):
                try:
                    Delta.load(p)
                    print("    已缓存")
                    continue
                except Exception:
                    pass
            g = build(d, verbose=True)
            if g is None:
                print("    没有点云，跳过")
                continue
            g.save(p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
