"""CPU forward rasterization of the existing SAGE Gaussian observations.

Projection/covariance/SH and the C++ tile compositor follow the maintained
ABot diff-plane rasterizer. This is an inference backend, never a training
surrogate. Real-scene GPU parity is required before enabling it in a recipe.
"""
from __future__ import annotations

import ctypes
import fcntl
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from utils.general_utils import build_rotation
from utils.sh_utils import eval_sh


@lru_cache(maxsize=1)
def _compositor():
    # No implicit shared-disk build/cache and no writing into the Git checkout.
    root = Path(os.environ["SAGE_CPU_RENDER_CACHE"]).resolve()
    source = Path(__file__).with_name("cpu_composite.cpp")
    key = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"composite_{key}.so"
    with (root / "compile.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not target.exists():
            temporary = target.with_suffix(f".{os.getpid()}.tmp.so")
            try:
                subprocess.run(["g++", "-std=c++17", "-O3", "-fPIC", "-shared", "-fopenmp",
                                "-ffp-contract=off", str(source), "-o", str(temporary)], check=True)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    library = ctypes.CDLL(str(target))
    function = library.composite_cpu
    pointer = ctypes.c_void_p
    function.argtypes = ([ctypes.c_int]*4 + [pointer]*7 + [ctypes.c_float]*4 + [pointer]*2)
    function.restype = ctypes.c_int
    return function


def composite(xy, depth, radii, conic_opacity, rgb, maps, background,
              *, width, height, fx, fy, cx, cy, threads=1):
    """Composite projected Gaussians in depth order, preserving CUDA thresholds."""
    n = len(depth)
    shapes = ((n, 2), (n,), (n,), (n, 4), (n, 3), (3,))
    arrays = [np.ascontiguousarray(value, dtype=np.int32 if i == 2 else np.float32)
              for i, value in enumerate((xy, depth, radii, conic_opacity, rgb, background))]
    if any(a.shape != shape or not np.isfinite(a).all() for a, shape in zip(arrays, shapes)):
        raise ValueError("invalid Gaussian compositor input")
    if (width <= 0 or height <= 0 or threads <= 0 or fx <= 0 or fy <= 0
            or not np.isfinite([fx, fy, cx, cy]).all()):
        raise ValueError("invalid CPU render camera or thread count")
    if maps is not None:
        maps = np.ascontiguousarray(maps, dtype=np.float32)
        if maps.shape != (n, 5) or not np.isfinite(maps).all():
            raise ValueError("invalid Gaussian geometry maps")
    image = np.empty((height, width, 3), dtype=np.float32)
    plane = np.zeros((height, width), dtype=np.float32)
    xy, depth, radii, co, rgb, bg = arrays
    pointers = [a.ctypes.data if a is not None else None
                for a in (xy, depth, radii, co, rgb, maps, bg)]
    result = _compositor()(n, width, height, threads, *pointers,
                           fx, fy, cx, cy, image.ctypes.data, plane.ctypes.data)
    if result:
        raise RuntimeError("CPU Gaussian compositor failed (allocation or native error)")
    return image, plane


@torch.no_grad()
def render(view, pc, background, *, render_scale=0., return_plane=False,
           suppress_blue_gaussians=False, threads=1):
    """Render the existing camera/model contract, including plane depth."""
    xyz, scales, rotations = pc.get_xyz, pc.get_scaling, pc.get_rotation
    opacity = pc.get_opacity.reshape(-1)
    sh = pc.get_features
    if any(t.device.type != "cpu" for t in (xyz, scales, rotations, opacity, sh, background)):
        raise ValueError("CPU renderer received a GPU tensor")
    width = int(view.image_width * render_scale) if render_scale > 0 else int(view.image_width)
    height = int(view.image_height * render_scale) if render_scale > 0 else int(view.image_height)
    tanx, tany = np.tan(view.FoVx * .5), np.tan(view.FoVy * .5)
    fx, fy = width / (2 * tanx), height / (2 * tany)
    cx = int(view.cx_pixel * render_scale) if render_scale > 0 else view.cx_pixel
    cy = int(view.cy_pixel * render_scale) if render_scale > 0 else view.cy_pixel
    matrix = view.world_view_transform.cpu().float()
    camera_xyz = xyz @ matrix[:3, :3] + matrix[3, :3]
    keep = (camera_xyz[:, 2] > .2) & (opacity > 1.e-6)
    if suppress_blue_gaussians:
        dc = sh[:, 0, :] * .28209479177387814 + .5
        keep &= ((dc[:, 2] - torch.maximum(dc[:, 0], dc[:, 1]) < 2./255.)
                 | (dc[:, 2] < 50./255.))
    xyz, scales, rotations, opacity, sh, camera_xyz = [
        t[keep] for t in (xyz, scales, rotations, opacity, sh, camera_xyz)]
    n = len(xyz)
    rotation = build_rotation(rotations)
    # CUDA's GLM column-major construction is R^T, yielding R S^2 R^T.
    scaled_rotation = rotation * scales[:, None, :]
    covariance = scaled_rotation @ scaled_rotation.transpose(1, 2)
    z = camera_xyz[:, 2]
    jacobian = torch.zeros((n, 2, 3), dtype=torch.float32)
    jacobian[:, 0, 0] = fx / z
    jacobian[:, 1, 1] = fy / z
    jacobian[:, 0, 2] = -fx * (camera_xyz[:, 0] / z).clamp(-1.3*tanx, 1.3*tanx) / z
    jacobian[:, 1, 2] = -fy * (camera_xyz[:, 1] / z).clamp(-1.3*tany, 1.3*tany) / z
    transform = jacobian @ matrix[:3, :3].T
    cov2 = transform @ covariance @ transform.transpose(1, 2)
    cov2[:, 0, 0] += .3
    cov2[:, 1, 1] += .3
    a, b, c = cov2[:, 0, 0], cov2[:, 0, 1], cov2[:, 1, 1]
    determinant = a*c-b*b
    valid = determinant != 0
    mid = .5*(a+c)
    radius = torch.ceil(3*torch.sqrt(mid + torch.sqrt((mid*mid-determinant).clamp_min(.1))))
    co = torch.stack((c/determinant, -b/determinant, a/determinant, opacity), dim=1)
    homogeneous = torch.cat((xyz, torch.ones((n, 1))), dim=1) @ view.full_proj_transform.cpu()
    projected = homogeneous[:, :2] / (homogeneous[:, 3:4] + 1.e-7)
    xy = ((projected + 1) * torch.tensor([width, height]) - 1) * .5
    direction = xyz - view.camera_center.cpu()
    direction = direction / direction.norm(dim=1, keepdim=True)
    rgb = (eval_sh(pc.active_sh_degree, sh.transpose(1, 2), direction) + .5).clamp_min(0)
    maps = None
    if return_plane:
        axes = scales.argmin(dim=1)
        normal = rotation[torch.arange(n), :, axes]
        sign = (normal * (view.camera_center.cpu() - xyz)).sum(-1) < 0
        normal[sign] = -normal[sign]
        local_normal = normal @ matrix[:3, :3]
        distance = (local_normal * camera_xyz).sum(-1).abs()
        maps = torch.cat((local_normal, torch.ones((n, 1)), distance[:, None]), dim=1)[valid].numpy()
    image, plane = composite(
        xy[valid].numpy(), z[valid].numpy(), radius[valid].to(torch.int32).numpy(),
        co[valid].numpy(), rgb[valid].numpy(), maps, background.numpy(),
        width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy, threads=threads)
    return {"render": torch.from_numpy(image).permute(2, 0, 1),
            "plane_depth": torch.from_numpy(plane).unsqueeze(0)}



def render_pgsr(view, pc, pipeline, background, *, app_color=None,
                app_color_beta=None, app_opacity=None, return_plane=True,
                img_res_scale=0, suppress_blue_gaussians=False):
    if any(value is not None for value in (app_color, app_color_beta, app_opacity)):
        raise ValueError("CPU renderer does not support appearance-network overrides")
    return render(view, pc, background, render_scale=img_res_scale,
                  return_plane=return_plane,
                  suppress_blue_gaussians=suppress_blue_gaussians,
                  threads=int(os.environ.get("SAGE_CPU_RENDER_THREADS", "8")))
