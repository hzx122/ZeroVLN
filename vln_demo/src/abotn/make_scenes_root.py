"""ABotN scenes as one 3DGS render-server ``scenes_root``.

Three benchmarks live in three directories, the render server takes one root,
and four scene ids appear in two benchmarks at once -- so we build a symlink
tree whose entries are the console's *prefixed* ids::

    <gs_root>/poi-20260212130612  -> ABotN-POIBench/occmaps/20260212130612
    <gs_root>/out-20260212160022  -> ABotN-PointBench/Outdoor/occmaps/20260212160022

The scene directories already have the ``gs/<block>/*.ply`` layout the server
scans for, so unlike ``rs3d.make_scenes_root`` this links whole directories and
invents no block structure.  Nothing is copied: 35 GB stays where it is.

    python3 -m abotn.make_scenes_root

Then point a *second* server instance at the tree -- second because
``--sh_degree`` is process-wide and these plys are degree 3 while 3DRS's are
degree 0::

    python render_sim_dynamic.py --scenes_root <gs_root> --port 7037 \\
        --gpus 0 --sh_degree 3
"""

from __future__ import annotations

import argparse
import os
import sys

from .datasets import ZYTOSS, available

GS_ROOT = os.environ.get(
    "ABOTN_GS_ROOT", "/workspace/agentic_sim/abotn_gs_root")


def build(gs_root: str = GS_ROOT, root: str = ZYTOSS, sets=None):
    os.makedirs(gs_root, exist_ok=True)
    made = []
    for key, ds in available(root).items():
        if sets and key not in sets:
            continue
        for scene in ds.list_scenes():
            link = os.path.join(gs_root, scene)
            _relink(link, ds.scene_dir(scene))
            made.append(scene)
    return made


def _relink(link: str, target: str):
    if os.path.islink(link):
        if os.path.realpath(link) == os.path.realpath(target):
            return
        os.unlink(link)
    elif os.path.exists(link):
        raise RuntimeError("%s 已存在且不是软链，不动它" % link)
    os.symlink(target, link)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gs-root", default=GS_ROOT)
    ap.add_argument("--root", default=ZYTOSS)
    ap.add_argument("--sets", default="", help="逗号分隔，默认全部 (poi,in,out)")
    a = ap.parse_args()
    made = build(a.gs_root, a.root, [s for s in a.sets.split(",") if s] or None)
    print("linked %d scenes into %s" % (len(made), a.gs_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
