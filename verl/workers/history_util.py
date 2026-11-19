# -*- coding: utf-8 -*-
"""history_util — 闭环多轮 rollout 的滚动历史窗口 + ego 对齐（train/eval 共用）。

问题：当前闭环推理每轮只喂**单帧 + 零 history**（fm_policy 用例4、trainer 的
run_episode_ode_multiturn 用 wp_zero=zeros((1,2))），而 SocialNav 原生 FM 训练带
**8 帧图像 + 8 个 input_waypoint**（历史帧位姿变换到当前帧 ego 系的相对 xy）。缺
历史 → 闭环 covariate shift，动作把 agent 带偏。

本模块把 SocialNav 的历史机制搬到闭环 rollout：
  - HistoryBuffer：deque(maxlen=N) 维护 (frame, x, y, yaw)，reset 后用当前帧重复
    N 份初始化（对齐 SFT 起步 linspace 补满，避免起步分布突变）。
  - ego_align_history：把 buffer 里 N 个历史位姿变换到**当前帧 ego 系**取相对 xy，
    再按 _ZYT stats 归一化（除 dx/dy range，同 data.py:_normalize_input_waypoints）。

坐标系（对齐 build_abotn_sft_data.relative_future + SocialNav _compute_zyt_waypoints）：
  abotn world 的 yaw 从 +x→+y。历史点 (x_k,y_k) 相对当前 (x0,y0,yaw0) 的 ego：
      dx =  cos(yaw0)*(x_k-x0) + sin(yaw0)*(y_k-y0)   # 前
      dy = -sin(yaw0)*(x_k-x0) + cos(yaw0)*(y_k-y0)   # 左
  这就是 R(-yaw0) 旋转，与 relative_future / SocialNav 的 t_ref_inv@pose_k 2D 等价。
  历史点在 agent 后方 → dx 多为负（走过来的路在身后），可据此自检方向。

归一化复用 fm_policy._norm_input_wp（除 range），stats 用 fm_policy._ZYT，单点维护
不重造。
"""
from __future__ import annotations

import math
import os
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

# 复用 fm_policy 的归一化 + stats（同目录，trainer 已把 verl/workers 加进 sys.path）
from fm_policy import _norm_input_wp, _ZYT  # noqa: E402

# 坐标约定开关：
#   ABOTN_BASE_CONVENTION=1（默认）= GT(output) 与 deploy 输出翻号对齐 0729 base 的
#     Y右/yaw顺时针约定（build.relative_future 翻 dy/dyaw + fm_action.denormalize 翻
#     dy/dyaw，两者互逆）。让模型输出侧用上 base 先验。
#   ABOTN_FLIP_INPUT=0（默认）= input_waypoint(历史点) **不翻号**，保持 abotn sim 原始
#     Y左约定（与 3DGS 图像物理同系）。即 Design C：模型看原始坐标系、只翻译动作输出。
#     设 1 = Design A（input 也翻 Y右，与图像物理异系，实测退步）。
# 详见 docs/实验记录.md#legacy-docs-20260921 2026-08-13 坐标映射 / Design C 条。
_BASE_CONV = os.environ.get("ABOTN_BASE_CONVENTION", "1") == "1"
_FLIP_INPUT = os.environ.get("ABOTN_FLIP_INPUT", "0") == "1"


def ego_relative_xy(hist_xy: np.ndarray, x0: float, y0: float, yaw0: float) -> np.ndarray:
    """(N,2) world (x,y) 历史点 → 当前帧 (x0,y0,yaw0) ego 系相对 (dx前, dy左)。

    R(-yaw0) 旋转，对齐 build_abotn_sft_data.relative_future 的 2D 部分。
    ABOTN_FLIP_INPUT=1 时 dy 取负（Y右约定，Design A；与图像物理异系，实测退步）；
    默认 0 保持 Y左原始（Design C：input 与 3DGS 图像同系，只翻译动作输出）。
    """
    c, s = math.cos(yaw0), math.sin(yaw0)
    dx = c * (hist_xy[:, 0] - x0) + s * (hist_xy[:, 1] - y0)
    dy = -s * (hist_xy[:, 0] - x0) + c * (hist_xy[:, 1] - y0)
    if _FLIP_INPUT:
        dy = -dy
    return np.stack([dx, dy], axis=1).astype(np.float32)


class HistoryBuffer:
    """滚动历史窗口：维护最近 N 个 (frame, x, y, yaw)。

    - reset(frame, x, y, yaw)：用首帧重复 N 份初始化（对齐 SFT 起步补满，避免
      起步只有 1 帧、模型吃到与训练分布不一致的短序列）。
    - append(frame, x, y, yaw)：滚动加入新观测，deque 自动丢最旧。
    - frames()：返回最近 N 帧 PIL（时序：旧→新，最后一个是当前帧）。
    - input_waypoints_norm(cur_xy, cur_yaw)：N 个历史位姿在当前帧 ego 系的归一化
      相对 xy，(N,2)。传入 cur 用最新观测的 pose（= 最后一次 append 的 pose）。
    """

    def __init__(self, n: int = 8):
        self.n = int(n)
        self._buf: Deque[Tuple[object, float, float, float]] = deque(maxlen=self.n)

    def reset(self, frame, x: float, y: float, yaw: float) -> None:
        self._buf.clear()
        for _ in range(self.n):
            self._buf.append((frame, float(x), float(y), float(yaw)))

    def append(self, frame, x: float, y: float, yaw: float) -> None:
        self._buf.append((frame, float(x), float(y), float(yaw)))

    def frames(self) -> List[object]:
        return [b[0] for b in self._buf]

    def poses_xy(self) -> np.ndarray:
        return np.array([[b[1], b[2]] for b in self._buf], np.float32)

    def input_waypoints_norm(self, cur_xy: Optional[Tuple[float, float]] = None,
                             cur_yaw: Optional[float] = None) -> np.ndarray:
        """(N,2) 归一化 input_waypoints（当前帧 ego 系）。

        cur_xy/cur_yaw 缺省用 buffer 里最新一个 pose（即当前观测），与 frames()
        末帧同一时刻，保证 input_waypoint 与图像时序对齐。
        """
        if not self._buf:
            return np.zeros((self.n, 2), np.float32)
        if cur_xy is None or cur_yaw is None:
            _, cx, cy, cyaw = self._buf[-1]
        else:
            cx, cy, cyaw = cur_xy[0], cur_xy[1], cur_yaw
        ego = ego_relative_xy(self.poses_xy(), cx, cy, cyaw)  # (N,2) 米
        return _norm_input_wp(ego, _ZYT)                       # 除 range 归一化

    def __len__(self) -> int:
        return len(self._buf)


def world_pose_from_info(info: dict) -> Optional[Tuple[float, float, float]]:
    """从 bench info 取 backend world 系 (x, y, yaw_rad)。

    info 的 world_xy=[x,y]、world_yaw_deg 由 vln_gym._metrics 从 get_metrics 的
    position/heading_deg 透出（abotn/rs3d world 系，与 SFT 同系）。缺字段返回 None
    （调用方回退，不崩）。yaw 转弧度供 ego 对齐用。
    """
    xy = info.get("world_xy")
    yaw_deg = info.get("world_yaw_deg")
    if xy is None or yaw_deg is None or len(xy) < 2:
        return None
    return float(xy[0]), float(xy[1]), math.radians(float(yaw_deg))

