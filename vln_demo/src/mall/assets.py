# -*- coding: utf-8 -*-
"""mall_floor2 资产加载与几何查询。

资产目录（``MALL_ASSETS_DIR``，默认 zekang 训练侧 cpfs 正本）::

    walk.json                1225 个 starts {idx,x,z,yaw_deg} + 45 个 shops
                             {name, front_world:[x,z], side_m, plan_px, front_node}
    floor_meta.json          {px_per_m, x0, y1, W, H}（平面图↔世界标定）
    mall_walk_union_topo.npy 膨胀后的可走域 mask（**必须用这个，不是原 navmesh**：
                             原 navmesh 走廊太窄；= topo_poi 连线 0.1m 栅格化 +
                             8px L1 膨胀 ∪ navmesh floor mask，+15.5% 可走像素）
    walk_grpo.json           （可选）2851 节点拓扑图 {idx,x,z}+边{a,b,米}，
                             SPL 专家路径长用；缺失退回欧氏距离

世界↔像素（与训练侧 floor_meta 同口径，mask[row=q, col=p]）::

    p = (x - x0) * ppm        x = x0 + p / ppm
    q = (z + y1) * ppm        z = q / ppm - y1

世界系 = habitat y-up 系（x 横 / z 深 / y 上），yaw 0°=−Z、逆时针为正
（与 rs3d / mt_move_server_strict.py 完全一致，无坐标系转换）。

注意：walk.json 的 ``front_node`` 索引的是 arr5 的 1225 起点表而不是
walk_grpo 的 2851 节点图（实测对不上，最远差 78 m），所以专家距离一律按
**坐标最近**把端点 snap 到拓扑图节点再跑 Dijkstra，不读 front_node。
"""

from __future__ import annotations

import heapq
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

DEFAULT_ASSETS_DIR = ("/opt/external/user-9656ce61/grpo/"
                      "locnav_grpo/exp0822/assets_arr5/mall")

SCENE_ID = "mall_floor2"     # GS 渲染场景名 = scenes_root 下的目录名

# 这些「商铺」是设施不是店，不进目标池（名字命中任一关键词即排除）。
INFRA_KEYWORDS = ("出口", "出入口", "电梯", "扶梯", "楼梯", "服务台", "导览屏", "休息区")


class MallAssets:
    """一个 mall_floor2 资产目录。``available()`` 只查文件在不在，``load()`` 才读。"""

    scene_id = SCENE_ID
    scene_label = "万科里 mall_floor2"
    target_label = "商铺"
    floorplan_path = os.path.join(os.path.dirname(__file__), "floorplan.png")
    unload_env = "MALL_UNLOAD_ON_CLOSE"

    def __init__(self, root: Optional[str] = None):
        self.root = root or os.environ.get("MALL_ASSETS_DIR", DEFAULT_ASSETS_DIR)
        self.loaded = False
        self.ppm = 12.0
        self.x0 = 0.0
        self.y1 = 0.0
        self.mask: Optional[np.ndarray] = None          # bool (H, W)
        self.starts: List[dict] = []                    # {idx,x,z,yaw_deg}
        self.shops: List[dict] = []                     # {name, front_world,...}
        self._goals: Optional[List[dict]] = None        # 非设施商铺
        # 拓扑图（可选）
        self._node_xy: Optional[np.ndarray] = None      # (N, 2)  [x, z]
        self._node_idx: Optional[np.ndarray] = None     # (N,) 原始 idx
        self._adj: Dict[int, List[Tuple[int, float]]] = {}
        self._dist_cache: Dict[str, Dict[int, float]] = {}

    # ------------------------------------------------------------- files
    def walk_json(self) -> str:
        return os.path.join(self.root, "walk.json")

    def floor_meta(self) -> str:
        return os.path.join(self.root, "floor_meta.json")

    def mask_npy(self) -> str:
        return os.path.join(self.root, "mall_walk_union_topo.npy")

    def topo_json(self) -> str:
        return os.path.join(self.root, "walk_grpo.json")

    def available(self) -> bool:
        return (os.path.isfile(self.walk_json())
                and os.path.isfile(self.floor_meta())
                and os.path.isfile(self.mask_npy()))

    def load(self) -> "MallAssets":
        if self.loaded:
            return self
        with open(self.walk_json(), encoding="utf-8") as f:
            w = json.load(f)
        self.starts = list(w.get("starts") or [])
        self.shops = list(w.get("shops") or [])
        with open(self.floor_meta(), encoding="utf-8") as f:
            meta = json.load(f)
        self.ppm = float(meta["px_per_m"])
        self.x0 = float(meta["x0"])
        self.y1 = float(meta["y1"])
        # 校验尺寸一致，防止拿错 mask
        m = np.load(self.mask_npy())
        if m.shape != (int(meta["H"]), int(meta["W"])):
            raise RuntimeError(
                "mall 资产不一致：mask 形状 %s 与 floor_meta W/H=(%d,%d) 不符"
                % (m.shape, int(meta["W"]), int(meta["H"])))
        self.mask = m > 0
        self._load_topo()
        self.loaded = True
        return self

    def _load_topo(self) -> None:
        """拓扑图可选：缺失时专家距离退回欧氏，其余功能不受影响。"""
        p = self.topo_json()
        if not os.path.isfile(p):
            return
        with open(p, encoding="utf-8") as f:
            g = json.load(f)
        nodes = g.get("nodes") or []
        edges = g.get("edges") or []
        if not nodes:
            return
        self._node_idx = np.array([int(n["idx"]) for n in nodes], dtype=np.int64)
        self._node_xy = np.array(
            [[float(n["x"]), float(n["z"])] for n in nodes], dtype=np.float64)
        pos = {int(n["idx"]): i for i, n in enumerate(nodes)}   # idx -> 数组行
        adj: Dict[int, List[Tuple[int, float]]] = {}
        for e in edges:
            a, b = int(e[0]), int(e[1])
            ia, ib = pos.get(a), pos.get(b)
            if ia is None or ib is None:
                continue
            if len(e) >= 3:
                d = float(e[2])
            else:  # 个别边缺距离（如 [0,3144]），用端点欧氏距离补
                d = float(np.hypot(self._node_xy[ia][0] - self._node_xy[ib][0],
                                   self._node_xy[ia][1] - self._node_xy[ib][1]))
            adj.setdefault(ia, []).append((ib, d))
            adj.setdefault(ib, []).append((ia, d))
        self._adj = adj

    @property
    def has_topo(self) -> bool:
        return self._node_xy is not None and len(self._adj) > 0

    # ------------------------------------------------------------- goals
    def goal_shops(self) -> List[dict]:
        """可当导航目标的商铺（剔除设施：安全出口/电梯/服务台 等）。"""
        if self._goals is None:
            self._goals = [s for s in self.shops
                           if s.get("name")
                           and not any(k in s["name"] for k in INFRA_KEYWORDS)]
        return self._goals

    # ------------------------------------------------- world <-> pixel
    def to_pixel(self, x: float, z: float) -> Tuple[float, float]:
        """世界 -> 平面图像素 (p 列, q 行)。"""
        return ((float(x) - self.x0) * self.ppm, (float(z) + self.y1) * self.ppm)

    def to_world(self, p: float, q: float) -> Tuple[float, float]:
        """平面图像素 (p 列, q 行) -> 世界 (x, z)。"""
        return (self.x0 + float(p) / self.ppm, float(q) / self.ppm - self.y1)

    def walkable(self, x: float, z: float) -> bool:
        """可走判据：查膨胀 walk mask（越界=不可走）。"""
        p, q = self.to_pixel(x, z)
        xi, yi = int(round(p)), int(round(q))
        m = self.mask
        if 0 <= yi < m.shape[0] and 0 <= xi < m.shape[1]:
            return bool(m[yi, xi])
        return False

    def snap(self, x: float, z: float, limit_m: float = 1.5):
        """最近可走像素（环搜索，半径 ``limit_m``）；找不到返回 None。

        起点本来都在可走域上，这只是防御：标注点贴着店铺门、浮点漂移等。
        """
        if self.walkable(x, z):
            return float(x), float(z)
        p0, q0 = self.to_pixel(x, z)
        lim = int(round(limit_m * self.ppm))
        H, W = self.mask.shape
        for r in range(1, lim + 1):
            q_lo, q_hi = max(0, int(q0) - r), min(H, int(q0) + r + 1)
            p_lo, p_hi = max(0, int(p0) - r), min(W, int(p0) + r + 1)
            sub = self.mask[q_lo:q_hi, p_lo:p_hi]
            ys, xs = np.nonzero(sub)
            if len(xs) == 0:
                continue
            order = np.argsort((xs + p_lo - p0) ** 2 + (ys + q_lo - q0) ** 2)
            i = order[0]
            return self.to_world(float(xs[i] + p_lo), float(ys[i] + q_lo))
        return None

    # --------------------------------------------------- expert length
    def nearest_node(self, x: float, z: float) -> Optional[int]:
        """坐标最近拓扑图节点（数组行号）。walk.json 的 front_node 不能用
        （见模块 docstring），一律按坐标 snap。"""
        if self._node_xy is None:
            return None
        d2 = (self._node_xy[:, 0] - x) ** 2 + (self._node_xy[:, 1] - z) ** 2
        return int(np.argmin(d2))

    def dist_from_goal(self, goal_key: str, gx: float, gz: float) -> Optional[Dict[int, float]]:
        """从目标点出发到所有拓扑图节点的 Dijkstra 距离表（按目标缓存）。

        图无向，从目标跑一次即可供任意起点查询；每个目标最多跑一次。
        无拓扑图返回 None（调用方退回欧氏距离）。
        """
        if not self.has_topo:
            return None
        if goal_key in self._dist_cache:
            return self._dist_cache[goal_key]
        src = self.nearest_node(gx, gz)
        if src is None:
            return None
        dist: Dict[int, float] = {src: 0.0}
        pq = [(0.0, src)]
        adj = self._adj
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float("inf")):
                continue
            for v, w in adj.get(u, ()):
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        self._dist_cache[goal_key] = dist
        return dist

    def expert_length(self, x0: float, z0: float, x1: float, z1: float) -> float:
        """起点→目标的专家路径长：拓扑图 Dijkstra（端点坐标最近 snap）；
        图缺失或不连通时退回欧氏距离（乐观但诚实标注）。"""
        eucl = float(np.hypot(x1 - x0, z1 - z0))
        if not self.has_topo:
            return eucl
        dist = self.dist_from_goal("goal:%.2f,%.2f" % (x1, z1), x1, z1)
        s = self.nearest_node(x0, z0)
        if dist is None or s is None or s not in dist:
            return eucl
        return float(dist[s])
