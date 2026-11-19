"""Turn InteriorGS scenes into a render-server ``scenes_root``.

Unlike :mod:`abotn.make_scenes_root`, which only symlinks (ABotN already ships
standard plys), SAGE scenes are PlayCanvas-*compressed*, so each one has to be
**converted** (:mod:`sage.conv_ply`) into a real standard degree-3 ply the
render server can load. The tree the server scans is::

    <gs_root>/sage-<id>/gs/0/sage-<id>.ply

named with the console's *prefixed* id so the render ``scene_id`` and the
console scene id are the same string (see :mod:`sage.env`). ~230 MB per scene,
written under the authorised data root, skipped if already present.

    python3 -m sage.make_scenes_root --split instr           # convert one split
    python3 -m sage.make_scenes_root --split instr --scenes 839992,840007

Then point a *second* render server at the tree (degree is per process)::

    render_sim_dynamic.py --scenes_root <gs_root> --port 7038 --gpus 0 --sh_degree 3
"""

from __future__ import annotations

import argparse
import os
import sys

from .conv_ply import convert, nurec_to_ply
from .datasets import SAGE_ROOT, SageSet

GS_ROOT = os.environ.get(
    "SAGE_GS_ROOT",
    "/data/zerovln/SAGE-3D/sage_gs_root")


def _usdz_path(root: str, bare: str) -> str:
    """Official per-scene usdz (carries the lossless .nurec), named by bare id."""
    return os.path.join(root, "SAGE-3D_InteriorGS_usdz", "InteriorGS_usdz",
                        "%s.usdz" % bare)


def build(split: str, gs_root: str = GS_ROOT, root: str = SAGE_ROOT,
          scenes=None, force: bool = False, source: str = "compressed"):
    """Build a render-server scenes_root; ``source`` picks the ply origin.

    ``compressed`` (default): SuperSplat ``3dgs_compressed.ply`` (lossy, ~18-36%
    of gaussians pruned). ``nurec``: the lossless ``.nurec`` inside the official
    ``<id>.usdz`` -- full gaussian count, unclipped SH -- via
    :func:`sage.conv_ply.nurec_to_ply`. Same output tree/naming either way, so
    swapping ``--source`` just re-points which ply the render server loads.
    """
    ds = SageSet(split=split, root=root)
    want = [ds.qualify(s) if not s.startswith(ds.key + "-") else s
            for s in (scenes or [])] or ds.list_scenes()
    os.makedirs(gs_root, exist_ok=True)
    made, skipped = [], []
    for scene in want:
        if source == "nurec":
            src = _usdz_path(root, ds.bare(scene))
            missing = "no usdz (%s)" % os.path.basename(src)
        else:
            src = os.path.join(ds.scene_dir(scene), "3dgs_compressed.ply")
            missing = "no 3dgs_compressed.ply"
        if not os.path.exists(src):
            print("  [skip] %s: %s" % (scene, missing))
            continue
        out_dir = os.path.join(gs_root, scene, "gs", "0")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "%s.ply" % scene)
        if os.path.exists(out) and not force:
            skipped.append(scene)
            continue
        if source == "nurec":
            nurec_to_ply(src, out)
        else:
            convert(src, out, "aabb")
        made.append(scene)
    print("converted %d, skipped %d (already present) into %s [source=%s]"
          % (len(made), len(skipped), gs_root, source))
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="instr", choices=["train", "instr", "scene", "traj"])
    ap.add_argument("--gs-root", default=GS_ROOT)
    ap.add_argument("--root", default=SAGE_ROOT)
    ap.add_argument("--scenes", default="", help="逗号分隔的 scene id（默认整个 split）")
    ap.add_argument("--force", action="store_true", help="已存在也重转")
    ap.add_argument("--source", default="compressed", choices=["compressed", "nurec"],
                    help="ply 来源：compressed=SuperSplat压缩(有损,少18-36%%高斯); "
                         "nurec=官方usdz内.nurec无损全量")
    a = ap.parse_args()
    build(a.split, a.gs_root, a.root,
          [s for s in a.scenes.split(",") if s] or None, a.force, a.source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
