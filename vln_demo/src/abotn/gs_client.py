"""Render client for ABotN scenes.

The 3DGS server offers two routes.  ``/render_gs`` wants COLMAP extrinsics and
is what :mod:`rs3d` uses, because 3DRS assets are y-up and the server's
convenience route is not.  ABotN is the case that route was written for: its
poses are already ``(x, y, z, roll, pitch, yaw)`` in a z-up world, exactly as
they appear in the ``traj_*.json`` annotations, so we post them verbatim to
``/render_gs_xyz_euler`` and let the server do its own COLMAP conversion.  That
keeps our frames pixel-identical to ABotN's own evaluator renders instead of
merely equivalent-looking.

The server is a *separate instance* from the 3DRS one: ``--sh_degree`` is
process-wide and its ply loader asserts the spherical-harmonic coefficient
count, so a server started at 0 for 3DRS cannot load these degree-3 plys at
all.  See ``abotn/README.md`` for the launch line.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Optional, Sequence

import numpy as np

from rs3d.gs_client import CameraSpec, GSRenderClient, _to_image

ABOTN_RENDER_URL = os.environ.get("ABOTN_RENDER_URL", "http://127.0.0.1:7037")


class XYZEulerClient(GSRenderClient):
    """:class:`~rs3d.gs_client.GSRenderClient` speaking the z-up euler route."""

    def __init__(self, url: str = ABOTN_RENDER_URL, timeout: float = 300.0,
                 camera: Optional[CameraSpec] = None):
        super().__init__(url=url, timeout=timeout, camera=camera)

    def render_one(self, scene_id: str, position: Sequence[float], yaw: float,
                   pitch: float = 0.0, want_depth: bool = False,
                   render_scale: Optional[float] = None):
        """``position`` is ABotN world ``(x, y, z)``; ``yaw``/``pitch`` radians.

        Same return contract as the base class: ``(rgb, depth)``.
        """
        payload = {
            "scene_id": scene_id,
            "x": float(position[0]), "y": float(position[1]), "z": float(position[2]),
            "roll": 0.0, "pitch": float(pitch), "yaw": float(yaw),
            "degree_flag": False,
            "cam_intrinsics": self.camera.intrinsics_txt(),
        }
        if render_scale is not None:
            payload["render_scale"] = float(render_scale)
        if want_depth:
            payload["return_depth_npz"] = True
        r = self._session.post(self.url + "/render_gs_xyz_euler", json=payload,
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
