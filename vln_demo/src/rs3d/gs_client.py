"""HTTP client for the ABotN 3DGS render server.

The server speaks COLMAP: ``cam_extrinsics`` is an ``images.txt`` body whose
quaternion/translation are world-to-camera, ``cam_intrinsics`` a
``cameras.txt`` body.  Its ``/render_gs_xyz_euler`` convenience route assumes a
z-up world, while 3DRS assets are y-up, so we always build the matrices here
and post to ``/render_gs``.

Camera convention (COLMAP/OpenCV): +x right, +y down, +z forward.
World convention (3DRS/habitat): +y up, agent at yaw 0 faces -z, yaw grows
counter-clockwise seen from above (i.e. towards -x).
"""

from __future__ import annotations

import base64
import io
import math
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Optional, Sequence
from urllib.parse import urlparse

import numpy as np

try:  # pillow is only needed by callers that want decoded frames
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

import requests

UP = np.array([0.0, 1.0, 0.0])


def forward_vector(yaw: float, pitch: float = 0.0) -> np.ndarray:
    """Unit forward vector for ``yaw``/``pitch`` in radians (y-up world)."""
    cp = math.cos(pitch)
    return np.array([-math.sin(yaw) * cp, math.sin(pitch), -math.cos(yaw) * cp])


def look_at_w2c(position: Sequence[float], yaw: float, pitch: float = 0.0):
    """Return ``(R_w2c, t_w2c)`` for an agent camera."""
    f = forward_vector(yaw, pitch)
    right = np.cross(f, UP)
    n = np.linalg.norm(right)
    right = np.array([1.0, 0.0, 0.0]) if n < 1e-8 else right / n
    down = np.cross(f, right)
    R = np.stack([right, down, f])  # rows are camera axes in world coords
    t = -R @ np.asarray(position, dtype=float)
    return R, t


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> COLMAP quaternion ``(w, x, y, z)``."""
    m = np.asarray(R, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, \
            (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, \
            (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, \
            (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, \
            (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


@dataclass
class CameraSpec:
    width: int = 640
    height: int = 480
    hfov_deg: float = 79.0  # habitat VLN default

    @property
    def fx(self) -> float:
        return self.width / (2.0 * math.tan(math.radians(self.hfov_deg) / 2.0))

    @property
    def fy(self) -> float:
        return self.fx  # square pixels

    def intrinsics_txt(self) -> str:
        return "1 PINHOLE %d %d %.6f %.6f %.6f %.6f" % (
            self.width, self.height, self.fx, self.fy,
            self.width / 2.0, self.height / 2.0)


def extrinsics_txt(poses, name_fmt: str = "%06d.jpg") -> str:
    """``poses``: sequence of ``(position, yaw, pitch)`` -> images.txt body."""
    lines = []
    for i, p in enumerate(poses):
        position, yaw = p[0], p[1]
        pitch = p[2] if len(p) > 2 else 0.0
        R, t = look_at_w2c(position, yaw, pitch)
        qw, qx, qy, qz = rotmat2qvec(R)
        lines.append("%d %.9f %.9f %.9f %.9f %.9f %.9f %.9f 1 %s" % (
            i + 1, qw, qx, qy, qz, t[0], t[1], t[2], name_fmt % i))
    return "\n".join(lines)


class GSRenderClient:
    """Thin wrapper over the render server's HTTP API."""

    def __init__(self, url: str = "http://127.0.0.1:7036", timeout: float = 300.0,
                 camera: Optional[CameraSpec] = None):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.camera = camera or CameraSpec()
        self._session = requests.Session()
        # Render-aware GPU occupier（可选）：bench 节点单卡 renderer 在 RL rollout 中
        # 墙钟大部分时间空闲，会被平台低利用率巡检 kill；co-located gpu_occupier 用
        # matmul 占卡，但渲染进行时必须让出 SM，否则拖慢出图。设 RENDER_BUSY_DIR 后，
        # 每个发往本 render server 的 POST（render/preload/unload，皆用 GPU）期间写一个
        # 含 render 端口的独立 request flag，占卡器按端口聚合是否暂停 matmul。
        # 多个 gym 共享 renderer 时不能共写/共删单文件，否则先完成的 POST 会
        # 清掉其它仍在渲染的请求信号。每次请求独占文件，仅清理自己的文件。
        # flag 覆盖 POST 全程（含 HTTP 往返 + 光栅化），渲染一启动占卡器即让出（~一次
        # matmul 迭代，几 ms），从而保证渲染速度与无占卡器时基本一致。仅 wrap post，
        # /ping /scenes 等轻量 get 不置位。
        _busy_dir = os.environ.get("RENDER_BUSY_DIR")
        if _busy_dir:
            _port = urlparse(self.url).port
            if _port:
                _orig_post = self._session.post

                def _flagged_post(url, *a, **kw):
                    _flag = None
                    try:
                        os.makedirs(_busy_dir, exist_ok=True)
                        fd, _flag = tempfile.mkstemp(
                            dir=_busy_dir,
                            prefix="rendering_%s_%s_" % (_port, os.getpid()))
                        os.close(fd)
                    except OSError:
                        pass
                    try:
                        return _orig_post(url, *a, **kw)
                    finally:
                        if _flag is not None:
                            try:
                                os.remove(_flag)
                            except OSError:
                                pass

                self._session.post = _flagged_post

    # -- plumbing ---------------------------------------------------------
    def ping(self) -> bool:
        try:
            r = self._session.get(self.url + "/ping", timeout=10)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def wait_ready(self, seconds: float = 600.0, poll: float = 2.0) -> bool:
        deadline = time.time() + seconds
        while time.time() < deadline:
            if self.ping():
                return True
            time.sleep(poll)
        return False

    def scenes(self):
        r = self._session.get(self.url + "/scenes", timeout=30)
        r.raise_for_status()
        return r.json()

    def preload(self, scene_id: str):
        r = self._session.post(self.url + "/preload", json={"scene_id": scene_id},
                               timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def unload(self, scene_id: str):
        r = self._session.post(self.url + "/unload", json={"scene_id": scene_id},
                               timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # -- rendering --------------------------------------------------------
    def render_one(self, scene_id: str, position, yaw: float, pitch: float = 0.0,
                   want_depth: bool = False, render_scale: Optional[float] = None):
        """Render a single view.

        Returns ``(rgb, depth)`` where ``rgb`` is a PIL image (raw JPEG bytes if
        pillow is missing) and ``depth`` a float32 metric depth array or None.
        ``/render_gs`` streams a bare JPEG unless depth is requested, in which
        case the reply is JSON with base64 fields.
        """
        # the endpoint wraps both fields in a one-element list of its own, so
        # each must be a single COLMAP *line*, not a whole file body.
        payload = {
            "scene_id": scene_id,
            "cam_extrinsics": extrinsics_txt([(position, yaw, pitch)]),
            "cam_intrinsics": self.camera.intrinsics_txt(),
        }
        if render_scale is not None:
            payload["render_scale"] = float(render_scale)
        if want_depth:
            payload["return_depth_npz"] = True
        r = self._session.post(self.url + "/render_gs", json=payload,
                               timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError("render failed %d: %s" % (r.status_code, r.text[:400]))
        if not want_depth:
            return _to_image(r.content), None
        data = r.json()
        depth = None
        if "depth_npz" in data:
            with np.load(io.BytesIO(base64.b64decode(data["depth_npz"]))) as z:
                depth = z["depth"]
        return _to_image(base64.b64decode(data["image"])), depth

    def render(self, scene_id: str, poses, want_depth: bool = False,
               workers: int = 1):
        """Render several views; ``poses`` are ``(position, yaw[, pitch])``."""
        poses = [tuple(p) if len(p) > 2 else (p[0], p[1], 0.0) for p in poses]
        if workers <= 1 or len(poses) == 1:
            return [self.render_one(scene_id, p[0], p[1], p[2], want_depth)
                    for p in poses]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(
                lambda p: self.render_one(scene_id, p[0], p[1], p[2], want_depth),
                poses))


def _to_image(raw: bytes):
    if Image is None:
        return raw
    return Image.open(io.BytesIO(raw)).convert("RGB")
