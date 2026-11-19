"""3DRS scenes as a 3DGS render-server ``scenes_root``.

The ABotN render server expects ``<scenes_root>/<scene_id>/gs/<block>/*.ply``
and loads *every* ply it finds in a block as a separate Gaussian model -- and
without ``polygon_*.txt`` block bounds it only ever renders block 0.  3DRS
instead stores ``<scene>/<scene>.gs.ply`` (the scene) next to
``background.ply`` (a far-field/sky shell), so we build a symlink tree with
exactly one ply per block.

    python3 -m rs3d.make_scenes_root [--with-background] [--scenes a,b,c]

Nothing is copied: 126 GB of assets stay where they are.
"""

from __future__ import annotations

import argparse
import os
import sys

RS3D_ROOT = os.environ.get("RS3D_ROOT", "/workspace/agentic_sim/3DRS")
GS_ROOT = os.environ.get(
    "RS3D_GS_ROOT", "/workspace/agentic_sim/3drs_gs_root")


def list_scenes(root: str = RS3D_ROOT):
    out = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "%s.gs.ply" % d)):
            out.append(d)
    return out


def build(scenes=None, with_background=False, gs_root=GS_ROOT, rs3d_root=RS3D_ROOT):
    scenes = scenes or list_scenes(rs3d_root)
    made = 0
    for s in scenes:
        block = os.path.join(gs_root, s, "gs", "block0")
        os.makedirs(block, exist_ok=True)
        link = os.path.join(block, "%s.gs.ply" % s)
        target = os.path.join(rs3d_root, s, "%s.gs.ply" % s)
        _relink(link, target)
        # background.ply would be loaded as a *second* Gaussian model that the
        # server never renders, so it goes in its own block only on request.
        bg_block = os.path.join(gs_root, s, "gs", "block1")
        bg_link = os.path.join(bg_block, "background.ply")
        if with_background:
            os.makedirs(bg_block, exist_ok=True)
            _relink(bg_link, os.path.join(rs3d_root, s, "background.ply"))
        elif os.path.islink(bg_link):
            os.unlink(bg_link)
        made += 1
    return made


def _relink(link: str, target: str):
    if os.path.islink(link) or os.path.exists(link):
        if os.path.realpath(link) == os.path.realpath(target):
            return
        os.unlink(link)
    os.symlink(target, link)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="", help="comma-separated subset")
    ap.add_argument("--with-background", action="store_true",
                    help="also link background.ply (as block1)")
    ap.add_argument("--gs-root", default=GS_ROOT)
    ap.add_argument("--rs3d-root", default=RS3D_ROOT)
    a = ap.parse_args()
    scenes = [s for s in a.scenes.split(",") if s] or None
    n = build(scenes, a.with_background, a.gs_root, a.rs3d_root)
    print("linked %d scenes into %s" % (n, a.gs_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
