"""Reader for the 3DRS ``.navmesh`` files (habitat NavMeshSet **version 2**).

The local habitat-sim (0.2.4) only loads version 1, so we parse the Detour tile
data ourselves.  We only need the geometry, not Detour's query machinery:

    NavMeshSetHeader   40 B   magic 'MSET', version, numTiles, dtNavMeshParams
    NavMeshSettings    56 B   13 floats + 1 packed int (3 bools)   [version >= 2]
    per tile:  NavMeshTileHeader 8 B (dtTileRef, dataSize) + dtMeshHeader + data

Verified against ``<scene>_metainfo.json`` (``navigable_area_m2``,
``num_islands``, ``largest_island_area_m2``), which the asset pipeline computed
with real habitat -- see ``check_navmesh.py``.

Coordinate frame: the same y-up world frame as ``<scene>.gs.ply`` /
``<scene>.mesh.ply``, so x/z is the ground plane and y is height.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

MSET_MAGIC = b"MSET"[::-1]  # stored little-endian as b'TESM'
DT_NAVMESH_MAGIC = 0x444E4156  # 'DNAV'
DT_EXT_LINK = 0x8000
DT_POLYTYPE_OFFMESH_CONNECTION = 1
POLYFLAGS_WALK = 0x01  # habitat's only walkable flag

_SET_HEADER = struct.Struct("<4sii3fffii")  # magic, ver, numTiles, params
_SETTINGS_SIZE = 56
_TILE_HEADER = struct.Struct("<Ii")  # dtTileRef (u32), dataSize
_MESH_HEADER = struct.Struct("<iiiii I iiiiiiiii fff 3f 3f f")
_POLY = struct.Struct("<I6H6HHBB")  # firstLink, verts, neis, flags, vertCount, areaAndtype
_LINK_SIZE = 12  # dtLink: ref, next, edge, side, bmin, bmax
_BVNODE_SIZE = 16  # dtBVNode: bmin[3]+bmax[3] as u16, int i
_OFFMESH_SIZE = 36  # dtOffMeshConnection
_POLY_DETAIL = np.dtype({"names": ["vertBase", "triBase", "vertCount", "triCount"],
                         "formats": ["<u4", "<u4", "u1", "u1"],
                         "offsets": [0, 4, 8, 9], "itemsize": 12})


@dataclass
class NavMesh:
    """Walkable surface of one scene, as a polygon soup."""

    verts: np.ndarray  # (V, 3) float32, world frame (y up)
    polys: List[np.ndarray]  # per polygon: vertex indices into ``verts``
    neighbours: List[List[int]]  # per polygon: adjacent polygon indices
    bmin: np.ndarray
    bmax: np.ndarray
    walkable_height: float
    walkable_radius: float
    walkable_climb: float
    settings: Optional[dict] = None
    tris: Optional[np.ndarray] = None  # (T, 3, 3) detail-mesh triangles
    tri_poly: Optional[np.ndarray] = None  # (T,) owning polygon of each triangle
    walkable: Optional[np.ndarray] = None  # (P,) bool, False = never step here

    # -- derived -----------------------------------------------------------
    def poly_area(self, i: int) -> float:
        """Surface area of polygon ``i``.

        Uses the detail triangles when available (that is what habitat reports
        as navigable area); falls back to the coarse polygon otherwise.
        """
        if self.tris is not None:
            t = self.tris[self.tri_poly == i]
            if len(t):
                return float(_tri_areas_3d(t).sum())
        p = self.verts[self.polys[i]]
        x, z = p[:, 0], p[:, 2]
        return 0.5 * abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1)))

    @property
    def area(self) -> float:
        """Navigable area in m^2, matching habitat's ``navigable_area``.

        habitat sums the true 3D triangle areas, so sloped surfaces count for
        more than their horizontal footprint.
        """
        if self.tris is not None and len(self.tris):
            return float(_tri_areas_3d(self.tris).sum())
        return float(sum(self.poly_area(i) for i in range(len(self.polys))))

    def island_areas(self) -> List[float]:
        per_poly = self.poly_areas()
        return [float(per_poly[c].sum()) for c in self.islands()]

    def poly_areas(self) -> np.ndarray:
        """Per-polygon 3D area, computed once."""
        if getattr(self, "_areas", None) is None:
            per = np.zeros(len(self.polys))
            if self.tris is not None and len(self.tris):
                np.add.at(per, self.tri_poly, _tri_areas_3d(self.tris))
            else:
                for i in range(len(self.polys)):
                    p = self.verts[self.polys[i]]
                    x, z = p[:, 0], p[:, 2]
                    per[i] = 0.5 * abs(np.dot(x, np.roll(z, -1)) -
                                       np.dot(z, np.roll(x, -1)))
            self._areas = per
        return self._areas

    def islands(self) -> List[List[int]]:
        """Connected components of the polygon adjacency graph, largest first."""
        areas = self.poly_areas()
        seen = [False] * len(self.polys)
        out = []
        for start in range(len(self.polys)):
            if seen[start]:
                continue
            stack, comp = [start], []
            seen[start] = True
            while stack:
                p = stack.pop()
                comp.append(p)
                for n in self.neighbours[p]:
                    if not seen[n]:
                        seen[n] = True
                        stack.append(n)
            out.append(comp)
        out.sort(key=lambda c: -areas[c].sum())
        return out


def _tri_areas_xz(t: np.ndarray) -> np.ndarray:
    """Horizontally projected areas of triangles ``t`` of shape (T, 3, 3)."""
    a = t[:, 1, [0, 2]] - t[:, 0, [0, 2]]
    b = t[:, 2, [0, 2]] - t[:, 0, [0, 2]]
    return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def _tri_areas_3d(t: np.ndarray) -> np.ndarray:
    """True 3D areas of triangles ``t`` of shape (T, 3, 3)."""
    a = t[:, 1] - t[:, 0]
    b = t[:, 2] - t[:, 0]
    return 0.5 * np.linalg.norm(np.cross(a, b), axis=1)


def load_navmesh(path: str) -> NavMesh:
    with open(path, "rb") as f:
        blob = f.read()

    magic, version, num_tiles, ox, oy, oz, tile_w, tile_h, max_tiles, max_polys = \
        _SET_HEADER.unpack_from(blob, 0)
    if magic != MSET_MAGIC:
        raise ValueError("not a habitat navmesh: magic %r" % magic)
    off = _SET_HEADER.size

    settings = None
    if version >= 2:
        f13 = struct.unpack_from("<13f", blob, off)
        flags = struct.unpack_from("<i", blob, off + 52)[0]
        settings = dict(
            cell_size=f13[0], cell_height=f13[1], agent_height=f13[2],
            agent_radius=f13[3], agent_max_climb=f13[4], agent_max_slope=f13[5],
            region_min_size=f13[6], region_merge_size=f13[7], edge_max_len=f13[8],
            edge_max_error=f13[9], verts_per_poly=f13[10],
            detail_sample_dist=f13[11], detail_sample_max_error=f13[12],
            filter_flags=flags,
        )
        off += _SETTINGS_SIZE

    all_verts: List[np.ndarray] = []
    all_polys: List[np.ndarray] = []
    all_neis: List[List[int]] = []
    all_tris: List[np.ndarray] = []
    all_tri_poly: List[np.ndarray] = []
    all_walk: List[bool] = []
    bmin = np.array([np.inf] * 3)
    bmax = np.array([-np.inf] * 3)
    wh = wr = wc = 0.0

    for _ in range(num_tiles):
        _tile_ref, data_size = _TILE_HEADER.unpack_from(blob, off)
        off += _TILE_HEADER.size
        if data_size <= 0:
            continue
        vert_base = sum(v.shape[0] for v in all_verts)
        poly_base = len(all_polys)
        verts, polys, neis, hdr, tris, tri_poly, walk = _parse_tile(blob, off, data_size)
        all_verts.append(verts)
        all_polys.extend([p + vert_base for p in polys])
        all_neis.extend([[n + poly_base for n in ns] for ns in neis])
        all_walk.extend(walk)
        if len(tris):
            all_tris.append(tris)
            all_tri_poly.append(tri_poly + poly_base)
        bmin = np.minimum(bmin, hdr["bmin"])
        bmax = np.maximum(bmax, hdr["bmax"])
        wh, wr, wc = hdr["walkable_height"], hdr["walkable_radius"], hdr["walkable_climb"]
        off += data_size

    verts = np.concatenate(all_verts, 0) if all_verts else np.zeros((0, 3), np.float32)
    tris = np.concatenate(all_tris, 0) if all_tris else np.zeros((0, 3, 3), np.float32)
    tri_poly = np.concatenate(all_tri_poly, 0) if all_tri_poly else np.zeros(0, np.int64)
    return NavMesh(verts=verts, polys=all_polys, neighbours=all_neis,
                   bmin=bmin, bmax=bmax, walkable_height=wh,
                   walkable_radius=wr, walkable_climb=wc, settings=settings,
                   tris=tris, tri_poly=tri_poly,
                   walkable=np.asarray(all_walk, dtype=bool))


def _parse_tile(blob: bytes, off: int, data_size: int):
    """Parse one serialised ``dtMeshTile``; returns verts, polys, neighbours."""
    (magic, version, _x, _y, _layer, _user_id, poly_count, vert_count,
     max_link_count, detail_mesh_count, detail_vert_count, detail_tri_count,
     bv_node_count, off_mesh_con_count, _off_mesh_base,
     walkable_height, walkable_radius, walkable_climb,
     bminx, bminy, bminz, bmaxx, bmaxy, bmaxz, _bv_quant) = \
        _MESH_HEADER.unpack_from(blob, off)
    if magic != DT_NAVMESH_MAGIC:
        raise ValueError("bad Detour tile magic 0x%x" % magic)

    p = off + _MESH_HEADER.size
    verts = np.frombuffer(blob, dtype="<f4", count=vert_count * 3,
                          offset=p).reshape(-1, 3).copy()
    p += vert_count * 12

    polys, neis, walk = [], [], []
    for i in range(poly_count):
        rec = _POLY.unpack_from(blob, p + i * _POLY.size)
        pv = rec[1:7]
        pn = rec[7:13]
        n_v = rec[14]
        area_and_type = rec[15]
        if (area_and_type >> 6) == DT_POLYTYPE_OFFMESH_CONNECTION:
            polys.append(np.zeros(0, np.int64))
            neis.append([])
            walk.append(False)
            continue
        # habitat marks walkable polygons with POLYFLAGS_WALK; anything else
        # (e.g. flag 4) still counts as an island but contributes no area and
        # must never be stepped on.
        walk.append(bool(rec[13] & POLYFLAGS_WALK))
        polys.append(np.asarray(pv[:n_v], dtype=np.int64))
        nb = []
        for k in range(n_v):
            v = pn[k]
            if v and not (v & DT_EXT_LINK):
                nb.append(int(v) - 1)
        neis.append(nb)

    # sections after the polygons: links, detailMeshes, detailVerts, detailTris
    p += poly_count * _POLY.size
    p += max_link_count * _LINK_SIZE
    detail = np.frombuffer(blob, dtype=_POLY_DETAIL, count=detail_mesh_count,
                           offset=p) if detail_mesh_count else None
    p += detail_mesh_count * _POLY_DETAIL.itemsize
    dverts = np.frombuffer(blob, dtype="<f4", count=detail_vert_count * 3,
                           offset=p).reshape(-1, 3) if detail_vert_count else \
        np.zeros((0, 3), np.float32)
    p += detail_vert_count * 12
    dtris = np.frombuffer(blob, dtype="<u1", count=detail_tri_count * 4,
                          offset=p).reshape(-1, 4) if detail_tri_count else \
        np.zeros((0, 4), np.uint8)

    expect = (_MESH_HEADER.size + vert_count * 12 + poly_count * _POLY.size +
              max_link_count * _LINK_SIZE + detail_mesh_count * _POLY_DETAIL.itemsize +
              detail_vert_count * 12 + detail_tri_count * 4 +
              bv_node_count * _BVNODE_SIZE + off_mesh_con_count * _OFFMESH_SIZE)
    if expect != data_size:
        raise ValueError("tile size mismatch: computed %d, stored %d "
                         "(struct layout assumption is wrong)" % (expect, data_size))

    tris, tri_poly = _detail_triangles(verts, polys, detail, dverts, dtris, walk)

    keep = [i for i, q in enumerate(polys) if q.size >= 3]
    remap = {old: new for new, old in enumerate(keep)}
    polys = [polys[i] for i in keep]
    neis = [[remap[n] for n in neis[i] if n in remap] for i in keep]
    walk = [walk[i] for i in keep]
    if len(tri_poly):
        sel = np.array([t in remap for t in tri_poly], dtype=bool)
        tris = tris[sel]
        tri_poly = np.array([remap[t] for t in tri_poly[sel]], dtype=np.int64)

    hdr = dict(bmin=np.array([bminx, bminy, bminz]),
               bmax=np.array([bmaxx, bmaxy, bmaxz]),
               walkable_height=walkable_height,
               walkable_radius=walkable_radius,
               walkable_climb=walkable_climb,
               counts=(poly_count, vert_count, max_link_count, detail_mesh_count,
                       detail_vert_count, detail_tri_count, bv_node_count,
                       off_mesh_con_count))
    return verts, polys, neis, hdr, tris, tri_poly, walk


def _detail_triangles(verts, polys, detail, dverts, dtris, walk):
    """Expand Detour's detail meshes into explicit world-space triangles.

    A detail triangle indexes the owning polygon's own vertices when the index
    is below the polygon's vertex count, and the tile's detail vertices
    otherwise.  Polygons without a detail mesh fall back to a fan
    triangulation of the coarse polygon.
    """
    tris, owner = [], []
    for i, pv in enumerate(polys):
        if pv.size < 3 or not walk[i]:
            continue
        pverts = verts[pv]
        if detail is not None and i < len(detail):
            vb = int(detail["vertBase"][i])
            tb = int(detail["triBase"][i])
            nv = int(detail["vertCount"][i])
            nt = int(detail["triCount"][i])
        else:
            nt = 0
            vb = tb = nv = 0
        if nt:
            for t in dtris[tb:tb + nt]:
                pts = []
                for k in range(3):
                    idx = int(t[k])
                    pts.append(pverts[idx] if idx < len(pv) else dverts[vb + idx - len(pv)])
                tris.append(np.stack(pts, 0))
                owner.append(i)
        else:
            for k in range(1, len(pv) - 1):
                tris.append(np.stack([pverts[0], pverts[k], pverts[k + 1]], 0))
                owner.append(i)
    if not tris:
        return np.zeros((0, 3, 3), np.float32), np.zeros(0, np.int64)
    return (np.stack(tris, 0).astype(np.float32),
            np.asarray(owner, dtype=np.int64))
