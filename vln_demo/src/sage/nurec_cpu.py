"""Extract official SAGE-3D NuRec gaussians (usdz) and render on CPU (no Isaac, no GPU).

nurec = gzip(msgpack): state_dict has standard 3DGS params in float16 (local frame):
  positions[N,3], rotations[N,4], scales[N,3]=log, densities[N,1]=logit opacity,
  features_albedo[N,3]=SH DC, features_specular[N,45]=SH rest, SH degree 3.
usdz gauss.usda places the volume with xformOp:transform M (a proper rotation),
so world(=trajectory) frame = local @ M. We render DC colour only (view-independent,
frame-invariant -> no SH rotation needed) with a from-scratch numpy EWA splatter.
"""
import sys, zipfile, gzip, math
import numpy as np
import msgpack
from scipy.spatial.transform import Rotation

C0 = 0.28209479177387814
# usdz xformOp:transform (row-vector: p_world = p_local @ M), constant across scenes
M = np.array([[-1.,0.,0.],[0.,0.,-1.],[0.,-1.,0.]])


def load_nurec(usdz_path, apply_M=False):
    z = zipfile.ZipFile(usdz_path)
    nurec = [n for n in z.namelist() if n.endswith(".nurec")][0]
    sd = msgpack.unpackb(gzip.decompress(z.read(nurec)), raw=False,
                         strict_map_key=False)["nre_data"]["state_dict"]
    G = ".gaussians_nodes.gaussians."

    def arr(k):
        shp = sd[G + k + ".shape"]
        return np.frombuffer(sd[G + k], dtype=np.float16).reshape(shp).astype(np.float32)

    pos = arr("positions")                 # local frame == occupancy/compressed-ply frame
    rot = arr("rotations")
    logscale = arr("scales")
    opacity = 1.0 / (1.0 + np.exp(-arr("densities")[:, 0]))     # sigmoid
    albedo = arr("features_albedo")        # SH DC
    rgb = np.clip(C0 * albedo + 0.5, 0.0, 1.0)
    scale = np.exp(logscale)

    q = rot / (np.linalg.norm(rot, axis=1, keepdims=True) + 1e-9)
    q_xyzw = np.stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]], 1)  # wxyz -> xyzw
    R_w = Rotation.from_quat(q_xyzw)
    if apply_M:                            # Isaac Y-up placement; NOT our frame
        pos = pos @ M
        R_w = Rotation.from_matrix(M) * R_w
    return dict(pos=pos.astype(np.float32), rot=R_w, scale=scale.astype(np.float32),
                opacity=opacity.astype(np.float32), rgb=rgb.astype(np.float32))


def render_cpu(g, cam_xyz, yaw, W=640, H=480, fx=184.75, fy=184.75,
               near=0.05, far=60.0, tile=16, bg=(0.0, 0.0, 0.0)):
    cx, cy = W / 2.0, H / 2.0
    psi = float(yaw)
    # world->cam rotation (rows = right, down, forward); forward = (cos,sin,0)
    fwd = np.array([math.cos(psi), math.sin(psi), 0.0])
    right = np.array([math.sin(psi), -math.cos(psi), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    Rwc = np.stack([right, down, fwd], 0)                  # 3x3
    C = np.asarray(cam_xyz, np.float32)

    Xc = (g["pos"] - C) @ Rwc.T                            # N x 3 camera space
    z = Xc[:, 2]
    front = (z > near) & (z < far)
    # project
    u = fx * Xc[:, 0] / z + cx
    v = fy * Xc[:, 1] / z + cy
    inb = front & (u > -32) & (u < W + 32) & (v > -32) & (v < H + 32)
    idx = np.nonzero(inb)[0]
    if len(idx) == 0:
        return np.zeros((H, W, 3), np.uint8)

    # 3D covariance in world -> camera
    Rg = g["rot"][idx].as_matrix()                         # k x 3 x 3
    s = g["scale"][idx]                                    # k x 3
    S2 = s[:, :, None] * s[:, None, :]                     # outer not needed; build diag
    Sig = np.einsum("kij,kj,klj->kil", Rg, s * s, Rg)      # R diag(s^2) R^T
    Wc = Rwc                                               # world->cam
    Sig_c = np.einsum("ij,kjl,ml->kim", Wc, Sig, Wc)       # k x 3 x 3
    zc = z[idx]; xc = Xc[idx, 0]; yc = Xc[idx, 1]
    # Jacobian per gaussian
    J = np.zeros((len(idx), 2, 3), np.float32)
    J[:, 0, 0] = fx / zc; J[:, 0, 2] = -fx * xc / (zc * zc)
    J[:, 1, 1] = fy / zc; J[:, 1, 2] = -fy * yc / (zc * zc)
    cov2d = np.einsum("kij,kjl,kml->kim", J, Sig_c, J)     # k x 2 x 2
    cov2d[:, 0, 0] += 0.3; cov2d[:, 1, 1] += 0.3
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2
    ok = det > 1e-9
    # conic = inverse cov2d
    inv = np.zeros_like(cov2d)
    inv[:, 0, 0] = cov2d[:, 1, 1]; inv[:, 1, 1] = cov2d[:, 0, 0]
    inv[:, 0, 1] = -cov2d[:, 0, 1]; inv[:, 1, 0] = -cov2d[:, 1, 0]
    inv = inv / np.maximum(det, 1e-9)[:, None, None]
    # 3-sigma radius in px
    tr = cov2d[:, 0, 0] + cov2d[:, 1, 1]
    disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0))
    lam = tr / 2 + disc
    rad = np.ceil(3.0 * np.sqrt(np.maximum(lam, 1e-6))).astype(np.int32)
    rad = np.clip(rad, 1, 128)

    up = u[idx]; vp = v[idx]; op = g["opacity"][idx]; col = g["rgb"][idx]
    keep = ok & (op > 1.0 / 255)
    order = np.argsort(zc)                                  # near -> far (front-to-back)
    order = order[keep[order]]

    img = np.zeros((H, W, 3), np.float32)
    Tacc = np.ones((H, W), np.float32)                     # transmittance
    # front-to-back: process near first, accumulate C += T*a*c ; T*=(1-a)
    for i in order:
        r = int(rad[i]); cu = up[i]; cv = vp[i]
        x0 = max(0, int(cu - r)); x1 = min(W, int(cu + r + 1))
        y0 = max(0, int(cv - r)); y1 = min(H, int(cv + r + 1))
        if x0 >= x1 or y0 >= y1:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        dx = xs - cu; dy = ys - cv
        A = inv[i]
        power = -0.5 * (A[0, 0] * dx * dx + 2 * A[0, 1] * dx * dy + A[1, 1] * dy * dy)
        a = op[i] * np.exp(np.clip(power, -30, 0))
        a = np.clip(a, 0, 0.99)
        Tt = Tacc[y0:y1, x0:x1]
        w = Tt * a
        img[y0:y1, x0:x1, 0] += w * col[i, 0]
        img[y0:y1, x0:x1, 1] += w * col[i, 1]
        img[y0:y1, x0:x1, 2] += w * col[i, 2]
        Tacc[y0:y1, x0:x1] = Tt * (1 - a)
    img += Tacc[:, :, None] * np.asarray(bg, np.float32)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


if __name__ == "__main__":
    import glob
    from PIL import Image
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from sage.datasets import available, SageSet
    suf = sys.argv[1] if len(sys.argv) > 1 else "839992"
    split = sys.argv[2] if len(sys.argv) > 2 else "instr"
    outdir = "/root/.claude/jobs/b2ac06e0/tmp"
    base = "/data/zerovln/SAGE-3D"
    usdz = f"{base}/SAGE-3D_InteriorGS_usdz/InteriorGS_usdz/{suf}.usdz"
    print("loading nurec (local frame, no M)...", flush=True)
    g = load_nurec(usdz, apply_M=False)
    print(f"gaussians: {len(g['pos'])}  world x[{g['pos'][:,0].min():.1f},{g['pos'][:,0].max():.1f}] "
          f"y[{g['pos'][:,1].min():.1f},{g['pos'][:,1].max():.1f}] z[{g['pos'][:,2].min():.1f},{g['pos'][:,2].max():.1f}]", flush=True)
    # use sage.datasets offset-corrected trajectory (same frame as occupancy/our ply)
    ds = available().get(split) or SageSet(split=split)
    scene = ds.qualify(suf)
    t = ds.tasks(scene, limit=1)[0]
    wp = t.waypoints
    n = len(wp)
    for tag, k in [("start", 0), ("q", n // 4), ("mid", n // 2), ("goal", n - 1)]:
        x, y, _ = wp[k]
        j = min(k + 3, n - 1)
        yaw = math.atan2(wp[j][1] - y, wp[j][0] - x) if j != k else t.start["yaw"]
        img = render_cpu(g, [x, y, 0.5], yaw)
        Image.fromarray(img).save(f"{outdir}/nurec_{suf}_{tag}.png")
        print(f"  {tag} k={k} pos=({x:.2f},{y:.2f}) yaw={math.degrees(yaw):.0f} saved", flush=True)


def load_std_ply(ply_path):
    """Load our converted standard degree-3 ply into the same gaussian dict (DC colour)."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "render_libs"))
    from plyfile import PlyData
    v = PlyData.read(ply_path)["vertex"]
    pos = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)  # wxyz
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    R_w = Rotation.from_quat(np.stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]], 1))
    scale = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float32))
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], np.float32)))
    albedo = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)
    rgb = np.clip(C0 * albedo + 0.5, 0.0, 1.0)
    return dict(pos=pos, rot=R_w, scale=scale, opacity=opacity, rgb=rgb.astype(np.float32))
