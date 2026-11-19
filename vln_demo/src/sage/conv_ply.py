"""PlayCanvas-compressed .ply (InteriorGS ``3dgs_compressed.ply``) -> standard 3DGS .ply.

InteriorGS ships each scene as a PlayCanvas/SuperSplat *compressed* ply (a
``chunk`` element of per-256-vertex AABBs + a ``vertex`` element of packed
uint32s + an ``sh`` element of uint8 rest coefficients). The render server's
``GaussianModel.load_ply`` only understands a *standard* degree-3 ply, so this
turns one into the other.

Dequantisation conventions -- verified by rendering scene 0474_840898 and
checking geometry, colour and heading against the GVLN reference path:

* **position / scale**: ``min + q/qmax * (max-min)`` with a 11-10-11 bit split,
  vertex ``i`` reads chunk ``i//256``. Scale stays in **log space** -- the
  server's ``get_scaling`` does the ``exp`` -- so it is written raw.
* **opacity**: the packed RGBA alpha byte is the already-sigmoided opacity in
  ``[0,1]``; the server's ``get_opacity`` applies sigmoid, so we write the
  **logit**.
* **colour (f_dc)**: the packed RGB, mapped through the chunk colour AABB, is
  the *display base colour* ``color01 = C0*f_dc + 0.5`` in ``[0,1]`` -- NOT the
  SH DC coefficient. The rasteriser computes ``C0*f_dc + 0.5`` itself, so
  feeding ``color01`` in as ``f_dc`` double-brightens and washes the scene white
  (the original PoC bug). We invert it: ``f_dc = (color01 - 0.5) / C0``.
* **rotation**: smallest-three, 3x10 bit + 2 bit index, components in
  ``[-1/√2, 1/√2]``; the missing one is ``sqrt(1 - sum²)``. Written as a unit
  quaternion (server's ``get_rotation`` normalises anyway).

Usage::

    python3 -m sage.conv_ply <compressed.ply> <out.ply> [aabb|aabb_raw|sh0]
    python3 -m sage.conv_ply --nurec <id.usdz> <out.ply>
"""

import pathlib
import sys

import numpy as np

# plyfile lives in the render venv's site-packages *or* the repo's render_libs;
# add the latter so conv_ply runs from any interpreter that can import numpy.
_RENDER_LIBS = str(pathlib.Path(__file__).resolve().parents[2] / "render_libs")
if _RENDER_LIBS not in sys.path:
    sys.path.insert(0, _RENDER_LIBS)
from plyfile import PlyData, PlyElement  # noqa: E402

C0 = 0.28209479177387814  # SH band-0 constant, rendered colour = C0*f_dc + 0.5

# .nurec ``features_specular`` -> INRIA/render-server ``f_rest`` convention.
# Measured empirically against the known-good compressed ply on scene 839992:
# the per-coefficient slope is a flat 0.2500 +/- 0.0001 across SH deg 1/2/3
# (a global 1/4, not a per-band basis constant), while DC/scale/opacity/rot are
# 1:1 direct copies. So the only rest-specific transform is this 1/4 scale.
SPEC_TO_FREST = 0.25
_STD_PLY_NAMES = (['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2']
                  + [f'f_rest_{k}' for k in range(45)]
                  + ['opacity', 'scale_0', 'scale_1', 'scale_2',
                     'rot_0', 'rot_1', 'rot_2', 'rot_3'])


def _write_std_ply(out, xyz, f_dc, f_rest, opacity, scale, quat):
    """Assemble the render-server standard degree-3 ply from column arrays.

    Shapes: xyz/scale/f_dc[N,3], quat[N,4] (wxyz), f_rest[N,45] (channel-major),
    opacity[N] (logit). Shared by both source converters so the field layout /
    ordering lives in exactly one place.
    """
    N = xyz.shape[0]
    rec = np.empty(N, dtype=np.dtype([(n, '<f4') for n in _STD_PLY_NAMES]))
    rec['x'], rec['y'], rec['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec['nx'] = rec['ny'] = rec['nz'] = 0
    rec['f_dc_0'], rec['f_dc_1'], rec['f_dc_2'] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    for k in range(45):
        rec[f'f_rest_{k}'] = f_rest[:, k]
    rec['opacity'] = np.asarray(opacity).reshape(-1)
    rec['scale_0'], rec['scale_1'], rec['scale_2'] = scale[:, 0], scale[:, 1], scale[:, 2]
    q = quat / (np.linalg.norm(quat, axis=1, keepdims=True) + 1e-12)
    rec['rot_0'], rec['rot_1'], rec['rot_2'], rec['rot_3'] = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    PlyData([PlyElement.describe(rec, 'vertex')], text=False).write(out)
    return N


def _load_nurec_raw(usdz_path):
    """Decode the raw float32 gaussian arrays out of the ``.nurec`` in a usdz.

    Returns the arrays in their *stored* form (scale=log, densities=logit,
    albedo=SH DC, specular=[15,3] coeff-major SH rest, rotations=wxyz), in the
    local == occupancy == compressed-ply frame (no Isaac ``M`` placement).
    """
    import gzip
    import zipfile
    import msgpack
    z = zipfile.ZipFile(usdz_path)
    name = [n for n in z.namelist() if n.endswith(".nurec")][0]
    sd = msgpack.unpackb(gzip.decompress(z.read(name)), raw=False,
                         strict_map_key=False)["nre_data"]["state_dict"]
    g = ".gaussians_nodes.gaussians."

    def arr(k):
        return np.frombuffer(sd[g + k], dtype=np.float16).reshape(
            sd[g + k + ".shape"]).astype(np.float32)

    return dict(positions=arr("positions"), rotations=arr("rotations"),
                scales=arr("scales"), densities=arr("densities"),
                features_albedo=arr("features_albedo"),
                features_specular=arr("features_specular"))


def nurec_to_ply(usdz: str, out: str) -> int:
    """Lossless ``.nurec`` (inside ``<id>.usdz``) -> standard degree-3 ply.

    The .nurec already holds *standard* 3DGS params, so this is a direct copy
    plus one reshape and one scale:

    * position <- positions; scale(log) <- scales; opacity(logit) <- densities;
      rotation(wxyz) <- rotations; f_dc <- features_albedo (all 1:1).
    * f_rest <- features_specular, which is stored **coeff-major ``[15,3]``**
      (SH-coeff outer, channel inner); the server wants **channel-major
      ``[3,15]``**, so reshape+transpose, then scale by :data:`SPEC_TO_FREST`.

    Both the reshape and the 1/4 were verified against the compressed ply
    (ordering corr 0.07 direct vs 0.98 reshaped; slope 0.2500 +/- 0.0001).
    """
    g = _load_nurec_raw(usdz)
    N = g["positions"].shape[0]
    xyz = g["positions"]
    scale = g["scales"]                                   # log space, direct
    opacity = g["densities"][:, 0]                        # logit, direct
    quat = g["rotations"]                                 # wxyz, normalised in writer
    f_dc = g["features_albedo"]                           # SH DC, direct
    # [N,15,3] coeff-major -> [N,3,15] channel-major -> [N,45], then 1/4 scale
    f_rest = (g["features_specular"].reshape(N, 15, 3).transpose(0, 2, 1)
              .reshape(N, 45) * SPEC_TO_FREST)
    n = _write_std_ply(out, xyz, f_dc, f_rest, opacity, scale, quat)
    print("wrote %s N=%d source=nurec  scale %.3f..%.3f  f_dc %.3f..%.3f  f_rest %.3f..%.3f"
          % (out, n, scale.min(), scale.max(), f_dc.min(), f_dc.max(),
             f_rest.min(), f_rest.max()))
    return n


def convert(src: str, out: str, fdc_mode: str = "aabb") -> int:
    d = PlyData.read(src)
    ch = d["chunk"]; v = d["vertex"]; sh = d["sh"]; N = len(v.data)
    CHUNK = 256; ci = np.arange(N) // CHUNK

    def stackc(names):
        return np.stack([ch[n] for n in names], 1)[ci]

    min_xyz = stackc(["min_x", "min_y", "min_z"]); max_xyz = stackc(["max_x", "max_y", "max_z"])
    min_sc = stackc(["min_scale_x", "min_scale_y", "min_scale_z"])
    max_sc = stackc(["max_scale_x", "max_scale_y", "max_scale_z"])
    min_rgb = stackc(["min_r", "min_g", "min_b"]); max_rgb = stackc(["max_r", "max_g", "max_b"])

    pp = np.asarray(v["packed_position"], dtype=np.uint64)
    pr = np.asarray(v["packed_rotation"], dtype=np.uint64)
    ps = np.asarray(v["packed_scale"], dtype=np.uint64)
    pc = np.asarray(v["packed_color"], dtype=np.uint64)

    def q3(u):  # 11-10-11 bits, x in the high bits
        x = ((u >> 21) & 0x7FF); y = ((u >> 11) & 0x3FF); z = (u & 0x7FF)
        return np.stack([x / 2047.0, y / 1023.0, z / 2047.0], 1)

    P = q3(pp); S = q3(ps)
    xyz = min_xyz + P * (max_xyz - min_xyz)
    scale = min_sc + S * (max_sc - min_sc)                   # log space; server exps

    # packed_color uint32 little-endian: byte0=A(LSB), then R, G, B
    A = (pc & 0xFF).astype(np.float64)
    R = ((pc >> 8) & 0xFF).astype(np.float64)
    G = ((pc >> 16) & 0xFF).astype(np.float64)
    B = ((pc >> 24) & 0xFF).astype(np.float64)
    p_op = np.clip(A / 255.0, 1e-4, 1 - 1e-4)
    opacity = np.log(p_op / (1 - p_op))[:, None]             # logit; server sigmoids

    def color01():
        return np.stack([min_rgb[:, 0] + R / 255 * (max_rgb[:, 0] - min_rgb[:, 0]),
                         min_rgb[:, 1] + G / 255 * (max_rgb[:, 1] - min_rgb[:, 1]),
                         min_rgb[:, 2] + B / 255 * (max_rgb[:, 2] - min_rgb[:, 2])], 1)

    if fdc_mode == "aabb":
        f_dc = (color01() - 0.5) / C0                        # correct: recover SH DC
    elif fdc_mode == "aabb_raw":
        f_dc = color01()                                     # PoC bug, kept for A/B only
    else:  # sh0: ignore the chunk AABB, treat R/255 as an sRGB colour
        f_dc = np.stack([(R / 255 - 0.5) / C0, (G / 255 - 0.5) / C0,
                         (B / 255 - 0.5) / C0], 1)

    # rotation smallest-three; components in [-1/√2, 1/√2]
    SQRT2 = np.sqrt(2.0)
    a = ((pr & 0x3FF).astype(np.float64) / 1023.0 * 2 - 1) / SQRT2
    b = (((pr >> 10) & 0x3FF).astype(np.float64) / 1023.0 * 2 - 1) / SQRT2
    c = (((pr >> 20) & 0x3FF).astype(np.float64) / 1023.0 * 2 - 1) / SQRT2
    idx = ((pr >> 30) & 0x3).astype(int)
    miss = np.sqrt(np.clip(1 - (a * a + b * b + c * c), 0, None))
    qout = np.zeros((N, 4))
    for i in range(4):
        m = (idx == i); rest = [j for j in range(4) if j != i]
        qout[m, i] = miss[m]; qout[m, rest[0]] = a[m]; qout[m, rest[1]] = b[m]; qout[m, rest[2]] = c[m]
    qout /= np.linalg.norm(qout, axis=1, keepdims=True)

    names = (['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2']
             + [f'f_rest_{k}' for k in range(45)]
             + ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'])
    rec = np.empty(N, dtype=np.dtype([(n, '<f4') for n in names]))
    rec['x'], rec['y'], rec['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec['nx'] = rec['ny'] = rec['nz'] = 0
    rec['f_dc_0'], rec['f_dc_1'], rec['f_dc_2'] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    for k in range(45):
        rec[f'f_rest_{k}'] = (np.asarray(sh[f'f_rest_{k}']).astype(np.float64) - 128.0) / 128.0
    rec['opacity'] = opacity[:, 0]
    rec['scale_0'], rec['scale_1'], rec['scale_2'] = scale[:, 0], scale[:, 1], scale[:, 2]
    rec['rot_0'], rec['rot_1'], rec['rot_2'], rec['rot_3'] = qout[:, 0], qout[:, 1], qout[:, 2], qout[:, 3]
    PlyData([PlyElement.describe(rec, 'vertex')], text=False).write(out)
    print("wrote %s N=%d fdc=%s  scale %.3f..%.3f  f_dc %.3f..%.3f"
          % (out, N, fdc_mode, scale.min(), scale.max(), f_dc.min(), f_dc.max()))
    return N


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--nurec":
        nurec_to_ply(sys.argv[2], sys.argv[3])
        return 0
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    convert(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "aabb")
    return 0


if __name__ == "__main__":
    sys.exit(main())
