"""Validate the navmesh v2 parser against the numbers habitat itself produced.

``<scene>_metainfo.json`` carries ``navigable_area_m2``, ``num_islands`` and
``largest_island_area_m2`` computed by the 3DRS asset pipeline with real
habitat, which makes it a ground-truth check for our own reader.

    python3 -m rs3d.check_navmesh [N_SCENES]
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rs3d.navmesh import load_navmesh  # noqa: E402

RS3D_ROOT = os.environ.get("RS3D_ROOT", "/workspace/agentic_sim/3DRS")


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    scenes = sorted(d for d in os.listdir(RS3D_ROOT) if os.path.isdir(os.path.join(RS3D_ROOT, d)))
    ok = bad = 0
    print("%-12s %9s %9s %6s %5s %9s %9s %s" %
          ("scene", "area", "ref_area", "isl", "ref", "big", "ref_big", "verdict"))
    for s in scenes[:limit]:
        d = os.path.join(RS3D_ROOT, s)
        meta = json.load(open(os.path.join(d, "%s_metainfo.json" % s)))["navmesh"]
        nm = load_navmesh(os.path.join(d, "%s.navmesh" % s))
        isl = nm.islands()
        area = nm.area
        big = nm.island_areas()[0] if isl else 0.0
        ra, ri, rb = (meta["navigable_area_m2"], meta["num_islands"],
                      meta["largest_island_area_m2"])
        good = abs(area - ra) < max(0.5, 0.01 * ra) and len(isl) == ri and \
            abs(big - rb) < max(0.5, 0.01 * rb)
        ok, bad = (ok + 1, bad) if good else (ok, bad + 1)
        print("%-12s %9.2f %9.2f %6d %5d %9.2f %9.2f %s" %
              (s, area, ra, len(isl), ri, big, rb, "OK" if good else "MISMATCH"))
    print("\n%d ok, %d mismatched" % (ok, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
