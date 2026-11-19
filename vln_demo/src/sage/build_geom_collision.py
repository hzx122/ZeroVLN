#!/usr/bin/env python3
"""Precompute RGB-geometry furniture collision masks for SAGE-3D scenes.

The base floor/wall policy remains in ``occupancy.png``/``structure.json``.
This script adds only body-height, vertically-oriented PLY surfaces that fall
inside plausible furniture label regions. It avoids the legacy behavior of
filling each label's entire axis-aligned box.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import (binary_closing, binary_dilation,
                           distance_transform_edt, label as nd_label)

from .datasets import SAGE_ROOT, SageSet
from .nurec_cpu import load_std_ply
from .occmap import FURNITURE_PASSABLE, OccMap, _point_in_poly

DEFAULT_OUT = os.environ.get(
    "SAGE_FURNITURE_MASK_ROOT",
    os.path.join(SAGE_ROOT, "sage_collision_masks"))

SKIP_WORDS = (
    "carpet", "rug", "blanket", "quilt", "pillow", "mattress", "sheet",
    "curtain", "window", "door", "wall design", "wall panel", "plaster",
    "painting", "picture", "mirror", "ceiling", "floor", "baseboard",
    "skirting", "light strip", "screening",
)

# Verified local occupancy defects.  Do not turn this into a global threshold:
# some scenes have incomplete PLY wall surfaces.  841459's 0.5 m-outlier wall
# pixels were checked against RGB PLY, structure.json and labels.json; they form
# an unsupported L-shaped spur in the living room while the real armchair is
# about 0.75 m farther ahead.
# bare id -> (seed_x, seed_y, minimum RGB-geometry distance).  Only the
# unsupported component containing the reviewed seed is removed.
VERIFIED_WALL_REPAIRS = {"841459": (.035, -1.42, .50)}
VERIFIED_SOLID_FURNITURE_SCENES = {"841459"}


def _candidate_region(occ: OccMap, labels) -> np.ndarray:
    """Broad semantic ROI used by the conservative surface mask."""
    mask = np.zeros_like(occ.free)
    for inst in labels:
        label = str(inst.get("label", "")).strip().lower()
        if label in FURNITURE_PASSABLE or any(w in label for w in SKIP_WORDS):
            continue
        pts = inst.get("bounding_box") or []
        if not pts:
            continue
        xs = [float(p["x"]) for p in pts]
        ys = [float(p["y"]) for p in pts]
        zs = [float(p["z"]) for p in pts]
        dx, dy, dz = max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)
        area = dx * dy
        if max(zs) < .20 or min(zs) > .75 or dz < .20 or min(dx, dy) < .08:
            continue
        if not (.05 <= area <= 6.0):
            continue
        c0, r1 = occ.to_pixel(min(xs), min(ys))
        c1, r0 = occ.to_pixel(max(xs), max(ys))
        ca, cb = sorted((c0, c1)); ra, rb = sorted((r0, r1))
        ca=max(0,ca); cb=min(occ.w-1,cb); ra=max(0,ra); rb=min(occ.h-1,rb)
        if ca <= cb and ra <= rb:
            mask[ra:rb+1, ca:cb+1] = True
    return mask


def _convex_hull(points):
    """2-D monotonic-chain hull (bbox corners are not guaranteed ordered)."""
    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return ((a[0] - o[0]) * (b[1] - o[1])
                - (a[1] - o[1]) * (b[0] - o[0]))

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _solid_furniture(occ: OccMap, labels, geom: np.ndarray) -> np.ndarray:
    """Fill exact oriented label footprints only when RGB PLY confirms them.

    Surface-only masks have pinholes: a 5 cm motion sample can slip between
    sparse Gaussians and walk through an armchair.  Conversely, the legacy code
    filled each label's axis-aligned min/max rectangle, grossly overblocking
    rotated furniture.  Here the label's real convex OBB footprint supplies a
    watertight solid, while nearby body-height PLY surfaces prove that the
    labelled object is actually present in the rendered asset.
    """
    mask = np.zeros_like(occ.free)
    xs_grid = occ.lower[0] + (np.arange(occ.w) + .5) * occ.scale
    ys_grid = occ.lower[1] + (occ.h - .5 - np.arange(occ.h)) * occ.scale
    gx, gy = np.meshgrid(xs_grid, ys_grid)
    support = binary_dilation(geom, iterations=2)
    for inst in labels:
        label = str(inst.get("label", "")).strip().lower()
        if label in FURNITURE_PASSABLE or any(w in label for w in SKIP_WORDS):
            continue
        pts = inst.get("bounding_box") or []
        if not pts:
            continue
        xs=[float(p["x"]) for p in pts]; ys=[float(p["y"]) for p in pts]
        zs=[float(p["z"]) for p in pts]
        dx=max(xs)-min(xs); dy=max(ys)-min(ys); dz=max(zs)-min(zs)
        area=dx*dy
        if max(zs)<.20 or min(zs)>.75 or dz<.20 or min(dx,dy)<.08:
            continue
        if not (.05 <= area <= 6.0):
            continue
        hull = _convex_hull((float(p["x"]), float(p["y"])) for p in pts)
        if len(hull) < 3:
            continue
        region = _point_in_poly(gx, gy, hull)
        if int((region & support).sum()) >= 2:
            mask |= region
    return mask


def _raster(pos, keep, occ: OccMap) -> np.ndarray:
    x=pos[keep,0]; y=pos[keep,1]
    c=np.rint((x-occ.lower[0])/occ.scale).astype(np.int32)
    r=np.rint((occ.h-1)-(y-occ.lower[1])/occ.scale).astype(np.int32)
    good=(c>=0)&(c<occ.w)&(r>=0)&(r<occ.h)
    out=np.zeros_like(occ.free); out[r[good],c[good]]=True
    return out


def _room_region(occ: OccMap, structure) -> np.ndarray:
    """Rasterize room interiors for conservative stale-wall repair."""
    xs = occ.lower[0] + (np.arange(occ.w) + .5) * occ.scale
    ys = occ.lower[1] + (occ.h - .5 - np.arange(occ.h)) * occ.scale
    gx, gy = np.meshgrid(xs, ys)
    out = np.zeros_like(occ.free)
    for room in structure.get("rooms", []):
        profile = room.get("profile") or []
        if len(profile) >= 3:
            out |= _point_in_poly(
                gx, gy, [(float(p[0]), float(p[1])) for p in profile])
    return out


def _verified_wall_remove(bare: str, occ: OccMap, structure, geom: np.ndarray):
    repair = VERIFIED_WALL_REPAIRS.get(bare)
    if repair is None:
        return np.zeros_like(occ.free)
    seed_x, seed_y, threshold = repair
    distance = distance_transform_edt(~geom, sampling=occ.scale)
    candidate = ((occ.occ == 127) & _room_region(occ, structure)
                 & (distance > threshold))
    components, _ = nd_label(candidate, structure=np.ones((3, 3), np.uint8))
    col, row = occ.to_pixel(seed_x, seed_y)
    if not (0 <= row < occ.h and 0 <= col < occ.w):
        return np.zeros_like(candidate)
    component = int(components[row, col])
    return components == component if component else np.zeros_like(candidate)


def _remove_small(mask: np.ndarray, min_pixels: int = 3) -> np.ndarray:
    """Drop isolated Gaussian speckles while retaining thin coherent surfaces."""
    components, count = nd_label(mask, structure=np.ones((3, 3), np.uint8))
    if not count:
        return mask
    sizes = np.bincount(components.ravel())
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[components]


def build_one(ds: SageSet, scene: str, out_root: str):
    bare=ds.bare(scene); sd=ds.scene_dir(scene)
    ply=os.path.join(SAGE_ROOT,"sage_gs_root",scene,"gs","0",scene+".ply")
    labels=json.load(open(os.path.join(sd,"labels.json"),encoding="utf-8"))
    structure=json.load(open(os.path.join(sd,"structure.json"),encoding="utf-8"))
    occ=OccMap(sd,False,"hybrid",furniture_mode="none")
    g=load_std_ply(ply)
    pos=g["pos"]; scale=g["scale"]; opacity=g["opacity"]
    rot=g["rot"].as_matrix()
    axis=np.argmin(scale,axis=1)
    normal=rot[np.arange(len(rot)),:,axis]
    nz=np.abs(normal[:,2])
    flat=np.min(scale,axis=1)/np.maximum(np.median(scale,axis=1),1e-6)
    keep=(opacity>.15)&(nz<.35)&(flat<.40)&(np.max(scale,axis=1)<.35)&(pos[:,2]>.10)&(pos[:,2]<1.35)
    wall_keep=(opacity>.10)&(nz<.45)&(flat<.50)&(np.max(scale,axis=1)<.50)&(pos[:,2]>.05)&(pos[:,2]<1.55)
    wall_geom=_raster(pos,wall_keep,occ)
    wall_remove=_verified_wall_remove(bare,occ,structure,wall_geom)
    geom=_raster(pos,keep,occ)
    if bare in VERIFIED_SOLID_FURNITURE_SCENES:
        mask=_solid_furniture(occ,labels,geom)
    else:
        # Keep the conservative PLY surface mask until a scene is visually
        # audited. Solid OBBs are intentionally not enabled wholesale: several
        # official GVLN starts lie inside labelled furniture volumes.
        geom &= ~binary_dilation(occ.occ==127,iterations=3)
        mask=geom&_candidate_region(occ,labels)
    # Full visual/physical map. Unlike hybrid it does not carry occupancy's
    # stale 127 lines forward: visible body-height PLY surfaces define walls
    # and furniture, while official-free UNION room polygons define candidate
    # floor. Closing seals one-cell sampling holes; the runtime robot-radius EDT
    # supplies the final 10 cm safety margin.
    visual_geom=_remove_small(_raster(pos,keep,occ),min_pixels=3)
    visual_geom=binary_closing(visual_geom,structure=np.ones((3,3),bool),iterations=1)
    room=_room_region(occ,structure)
    visual_candidate=(occ.occ==255)|room
    visual_free=visual_candidate&~(visual_geom|mask)
    Path(out_root).mkdir(parents=True,exist_ok=True)
    path=os.path.join(out_root,bare+".npz")
    np.savez_compressed(path,mask=mask,scale=np.float32(occ.scale),
                        lower=np.asarray(occ.lower,np.float32),
                        shape=np.asarray(mask.shape,np.int32),
                        wall_remove=wall_remove,
                        visual_free=visual_free,
                        version=np.asarray(3,np.int32))
    print(f"{scene}: gaussians={len(pos)} selected={int(keep.sum())} "
          f"blocked={int(mask.sum())}/{mask.size}={mask.mean():.2%} "
          f"wall_repaired={int(wall_remove.sum())} "
          f"visual_free={visual_free.mean():.2%} -> {path}",flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--split",default="instr")
    ap.add_argument("--scenes",default="",help="comma-separated bare or sage-prefixed ids")
    ap.add_argument("--out",default=DEFAULT_OUT)
    a=ap.parse_args()
    ds=SageSet(a.split)
    scenes=[ds.qualify(x.strip()) if not x.strip().startswith("sage-") else x.strip()
            for x in a.scenes.split(",") if x.strip()] or ds.list_scenes()
    for scene in scenes: build_one(ds,scene,a.out)


if __name__=="__main__": main()
