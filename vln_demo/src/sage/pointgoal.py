"""Point-Goal 工具模块（sage 后端）。

对齐 ABot-N1 Point-Goal 链路（``docs/实验记录.md#legacy-docs-20260921`` 2026-08-14 sage SFT 系列）：
在 curated 专家轨迹上选一个 ~look_m（室内默认 3m，可调）前方的 safe
traversable waypoint，traversability + LOS 检查，不行沿 path backtrack。

**可视化**（vln_demo 控制台「走专家轨迹」）：只把前瞻 waypoint 画在 top-down
地图上（青色标记，``SageEnv.render_top_down`` 读 ``env._pointgoal_wp``），不在
RGB 图像上画点——FOV/可见性几何限制下点不一定可见（地面点 <1.22m 正前方会
掉出画面下沿、路径转弯处后方点背后），按用户定调「不可见就只在地图上可视化」。

**SFT pixel GT**（后续）：``project_pixel`` 把 3D waypoint 投到当前 RGB 得
(u, v)，作为 Slow-VLM 的 Point-Goal label；当前只在地图可视化，离线 dump
label 时再调用（``build_sage_sft_data`` import 本模块即可）。

相机约定：sage z-up 世界、yaw 从 +x→+y；render server ``/render_gs_xyz_euler``
经 ``xyz_euler_trans_gs_colmap`` 建 COLMAP（OpenCV：+x right/+y down/+z forward）
w2c。从 sage ``gs_client`` 的 2D ray 模型（``gs_client.py:215-225``：
forward=(cos yaw, sin yaw)、right=(sin yaw, -cos yaw)）+ z-up 推出世界系相机轴：

    R_w2c row0 (cam x=right) = ( sin yaw, -cos yaw, 0)
    R_w2c row1 (cam y=down ) = ( 0,       0,       -1)   # 世界 +z = 图像上
    R_w2c row2 (cam z=fwd  ) = ( cos yaw,  sin yaw, 0)

自验：地面点 3m 正前方、相机下方 1.2m → p_cam[1]>0 → v>cy（图下半部），
p_cam[2]>0 → 可见。
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------------
# look-ahead waypoint on the expert path (safety + LOS, no projection)
# ----------------------------------------------------------------------------
def lookahead_waypoint(occ, path, cur_xy, look_m: float,
                       radius: float) -> Optional[Tuple[float, float]]:
    """在专家 path 上选 ~look_m 前方的 safe traversable waypoint（世界 x,y）。

    链路：从 cur_xy 投影到 path 的最近点 → 累积弧长取前方 look_m 处作候选 →
    traversability（``OccMap.is_free``）+ LOS（``OccMap.seg_free`` from cur）→
    不行就沿 path **backtrack**（缩短弧长，远的先试以偏好 ~look_m）→ 返回
    第一个通过的有效点；全受阻返回 None。

    纯 safety/LOS 选点，不含投影——给「地图可视化」与「SFT pixel GT 投影前
    的选点」共用。
    """
    pts = [(float(p[0]), float(p[1])) for p in path]
    if len(pts) < 2:
        return None
    seg = [math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
           for i in range(len(pts) - 1)]
    cum = [0.0]
    for s in seg:
        cum.append(cum[-1] + s)
    if cum[-1] < 1e-6:
        return None
    cx, cy = float(cur_xy[0]), float(cur_xy[1])
    i0 = int(np.argmin([(pts[i][0] - cx) ** 2 + (pts[i][1] - cy) ** 2
                        for i in range(len(pts))]))
    arc0 = cum[i0]
    target_arc = min(arc0 + float(look_m), cum[-1])
    cand = [k for k in range(len(pts))
            if arc0 + 0.05 < cum[k] <= target_arc + 1e-9]
    cand.sort(key=lambda k: cum[k], reverse=True)   # 远的先试
    for k in cand:
        x, y = pts[k]
        if not occ.is_free(x, y, radius):
            continue
        if not occ.seg_free(cx, cy, x, y, radius):
            continue
        return (x, y)
    return None


# ----------------------------------------------------------------------------
# camera projection (SFT pixel GT — 后续 build_sage_sft_data 调)
# ----------------------------------------------------------------------------
def _w2c_rows(yaw: float):
    """World->camera rotation rows (right, down, forward) for sage z-up yaw。

    forward = (cos yaw, sin yaw, 0)；right = forward 绕 z 顺时针 90° =
    (sin yaw, -cos yaw, 0)；down = -世界 z = (0, 0, -1)。
    """
    c, s = math.cos(yaw), math.sin(yaw)
    return (s, -c, 0.0), (0.0, 0.0, -1.0), (c, s, 0.0)


def project_pixel(camera, cam_pos, cam_yaw: float,
                  world_pt: Sequence[float], pitch: float = 0.0) -> Optional[Tuple[float, float]]:
    """世界 (x, y, z) 点投影到 sage 相机图像平面，返回像素 (u, v)。

    点在相机背后（z_cam<=0）或落在画面外返回 None。``camera`` 是 sage
    ``CameraSpec``（fx/fy/width/height）；``cam_pos``/``cam_yaw`` 是 agent 相机的
    sage 世界位姿（z-up，yaw +x→+y）。
    """
    fx, fy = float(camera.fx), float(camera.fy)
    cx, cy = float(camera.width) / 2.0, float(camera.height) / 2.0
    right, down, fwd = _w2c_rows(float(cam_yaw))
    # Same z-up camera basis as sage.gs_client.XYZEulerClient._extrinsics.
    cp, sp = math.cos(pitch), math.sin(pitch)
    down = (fwd[0]*sp, fwd[1]*sp, -cp)
    fwd = (fwd[0]*cp, fwd[1]*cp, sp)
    dp = (float(world_pt[0]) - float(cam_pos[0]),
          float(world_pt[1]) - float(cam_pos[1]),
          float(world_pt[2]) - float(cam_pos[2]))
    x_cam = right[0] * dp[0] + right[1] * dp[1] + right[2] * dp[2]
    y_cam = down[0] * dp[0] + down[1] * dp[1] + down[2] * dp[2]
    z_cam = fwd[0] * dp[0] + fwd[1] * dp[1] + fwd[2] * dp[2]
    if z_cam <= 1e-3:
        return None                 # 背后 / 贴着相机 -> 不是有效 pixel
    u = fx * x_cam / z_cam + cx
    v = fy * y_cam / z_cam + cy
    w, h = float(camera.width), float(camera.height)
    if u < 0.0 or v < 0.0 or u >= w or v >= h:
        return None                 # 前方但出画面
    return float(u), float(v)
