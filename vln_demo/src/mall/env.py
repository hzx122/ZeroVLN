# -*- coding: utf-8 -*-
"""A habitat-shaped simulator for the mall_floor2 Gaussian scene.

Same contract as :mod:`rs3d.env` / :mod:`abotn.env` -- ``reset``/``step``/
``get_metrics``/``episode_over`` plus ``sim.agents[0].agent_config
.action_space``, ``sim.get_agent_state()`` and ``sim.previous_step_collided``
-- so ``agent_tools.py`` and the ReAct/FM drivers run unmodified.

What is specific to mall_floor2 (authoritative source: zekang's
``mt_move_server_strict.py``, the /move server used for closed-loop training):

* **kinematics are strict, off the dilated walk mask** -- a forward step moves
  only if BOTH the midpoint and the endpoint are walkable; anything less
  rejects the whole action (no move, no partial slide), exactly the training
  semantics. ``previous_step_collided`` is raised on such a rejection so
  ``agent_tools.drive_forward`` reads it as a stall/blocked step.
* **the episodes are real**: 1225 annotated starts (``walk.json``) each paired
  with a deterministic seeded shop goal, instruction ``前往<店名>``, scored at
  2 m arrival radius like ABotN-POIBench. SPL's shortest path is a Dijkstra
  over the 2851-node topo graph (``walk_grpo.json``), endpoints snapped by
  nearest coordinate -- the ``front_node`` field of ``walk.json`` indexes a
  different node table and must not be used (see ``mall/assets.py``).

Coordinate frame: the scene is natively habitat y-up (x/z ground plane, yaw
0° = −Z, counter-clockwise positive) -- the same frame the render server and
the move server speak, so there is no frame conversion anywhere in this file.

Metric frame (``get_metrics()["position"]`` / ``["heading_deg"]``): reported
as ``[x, −z, 1.5]`` with ``heading = habitat_yaw + 90°`` so that the ideal
kinematics ``forward = (cos h, sin h)`` that ``vln_gym.fm_action`` assumes for
chunk-frame trajectories hold (same two-frame boundary design abotn uses).
``obs`` gps/compass stay habitat-shaped via ``get_agent_state()``.
"""

from __future__ import annotations

import math
import os
import threading
import zlib
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from rs3d.env import (Agent, AgentConfig, AgentState, ActionSpec, Actuation,
                      Instruction, Quaternion, draw_agent_arrow)
from rs3d.gs_client import CameraSpec, GSRenderClient
from rs3d.nav import quat_to_yaw, yaw_to_quat

from .assets import SCENE_ID, MallAssets

STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2, 3

CAMERA_HEIGHT = 1.5           # 相机绝对高度（米），训练/渲染标定值
CAMERA_W, CAMERA_H = 1280, 720
CAMERA_HFOV = 90.0            # fx = 640（与训练数据同口径）
FORWARD_STEP = 0.25           # 前进原语（米）
TURN_ANGLE = 15.0             # 转向原语（度），08-24 起现行标准
ARRIVE_THRESHOLD = 2.0        # 到店门口 2 m 算到达（POIBench 同款）
# 目标筛选的专家距离区间（与训练侧 build 的 `4<d<80` 过滤同款）：
# 太近（起点就在店门口，Dijkstra 端点 snap 到同一节点 → expert≈0 → SPL 退化），
# 太远（超预算，学不出抵达）。
EXPERT_MIN_M = 4.0
EXPERT_MAX_M = 80.0

MALL_RENDER_URL = os.environ.get("MALL_RENDER_URL", "http://127.0.0.1:7039")


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Episode:
    """One start × shop-goal task in habitat-episode clothing.

    ``scene`` is a stored field (like abotn's): the ply path's
    ``scene_id.split('/')[-2]`` would say ``block0``, not ``mall_floor2``.
    """

    episode_id: str
    scene: str
    scene_id: str
    instruction: Instruction
    start_position: np.ndarray              # habitat y-up [x, CAMERA_HEIGHT, z]
    start_rotation: tuple                   # quaternion (w, x, y, z)
    goal_name: Optional[str] = None         # None = free-roam（无目标，指标 NaN）
    goal_xz: Optional[tuple] = None         # (x, z) world
    expert_len_m: float = 0.0
    info: dict = field(default_factory=dict)


class MallSim:
    """Pose bookkeeping + rendering for the single mall_floor2 scene."""

    def __init__(self, assets: MallAssets, client: GSRenderClient,
                 forward_step: float = FORWARD_STEP,
                 turn_angle: float = TURN_ANGLE, want_depth: bool = True):
        self.assets = assets
        self.client = client
        self.want_depth = bool(want_depth)
        self.previous_step_collided = False
        self._lock = threading.Lock()
        self.agents = [Agent(AgentConfig(
            action_space={
                MOVE_FORWARD: ActionSpec("move_forward", Actuation(float(forward_step))),
                TURN_LEFT: ActionSpec("turn_left", Actuation(float(turn_angle))),
                TURN_RIGHT: ActionSpec("turn_right", Actuation(float(turn_angle))),
            },
            sensor_height=CAMERA_HEIGHT))]
        self._x = 0.0
        self._z = 0.0
        self._yaw = 0.0

    # -- state -------------------------------------------------------------
    @property
    def xz(self):
        return self._x, self._z

    @property
    def yaw(self) -> float:
        """Heading, habitat convention: radians, 0 = −Z, CCW positive."""
        return self._yaw

    def get_agent_state(self, agent_id: int = 0) -> AgentState:
        return AgentState(np.array([self._x, CAMERA_HEIGHT, self._z]),
                          Quaternion(*yaw_to_quat(self._yaw)))

    def set_agent_state(self, position: Sequence[float],
                        rotation=None, agent_id: int = 0) -> bool:
        x, z = float(position[0]), float(position[2])
        if not self.assets.walkable(x, z):
            snapped = self.assets.snap(x, z)
            if snapped is not None:
                x, z = snapped
        self._x, self._z = x, z
        if rotation is not None:
            if np.iterable(rotation) and len(rotation) == 4:
                self._yaw = quat_to_yaw(*rotation)
            else:
                self._yaw = float(np.ravel(rotation)[0])
        self.previous_step_collided = False
        return True

    # -- actions -----------------------------------------------------------
    def act(self, action_id: int) -> None:
        """One kinematic primitive. Reads ``actuation.amount`` at call time so
        ``agent_tools``' temporary-amount trick works (negative = backward).

        Forward semantics = training-strict: midpoint AND endpoint must be
        walkable for the step to pass at all; otherwise the whole action is
        rejected (position untouched) and ``previous_step_collided`` goes up.
        No partial displacement, no wall sliding.
        """
        spec = self.agents[0].agent_config.action_space.get(action_id)
        if spec is None:
            raise KeyError("no action %r" % (action_id,))
        amount = float(spec.actuation.amount)
        if spec.name == "move_forward":
            dx = -math.sin(self._yaw) * amount
            dz = -math.cos(self._yaw) * amount
            tx, tz = self._x + dx, self._z + dz
            ok = (self.assets.walkable(tx, tz)
                  and self.assets.walkable(self._x + dx / 2.0,
                                           self._z + dz / 2.0))
            if ok:
                self._x, self._z = tx, tz
            self.previous_step_collided = not ok
        elif spec.name in ("turn_left", "turn_right"):
            sign = 1.0 if spec.name == "turn_left" else -1.0
            self._yaw = _wrap_pi(self._yaw + sign * math.radians(amount))
            self.previous_step_collided = False

    # -- sensors -----------------------------------------------------------
    def observe(self) -> dict:
        with self._lock:  # one in-flight render keeps the server calm
            rgb, depth = self.client.render_one(
                self.assets.scene_id, (self._x, CAMERA_HEIGHT, self._z), self._yaw,
                want_depth=self.want_depth)
        obs = {"rgb": np.asarray(rgb, dtype=np.uint8)}
        if depth is not None:
            obs["depth"] = depth
        return obs


class MallEnv:
    """``habitat.Env``-shaped wrapper over one :class:`MallAssets` directory."""

    def __init__(self, assets: MallAssets, render_url: str = MALL_RENDER_URL,
                 width: int = CAMERA_W, height: int = CAMERA_H,
                 hfov: float = CAMERA_HFOV, forward_step: float = FORWARD_STEP,
                 turn_angle: float = TURN_ANGLE, seed: int = 0,
                 max_episode_steps: int = 0, episodes: int = 0,
                 client: Optional[GSRenderClient] = None):
        if not assets.loaded:
            assets.load()
        self.assets = assets
        self.seed = int(seed)
        self.max_episode_steps = int(max_episode_steps)
        self.client = client or GSRenderClient(
            url=render_url, camera=CameraSpec(width, height, hfov))
        self._map_cache: dict = {}
        self._map_inv: dict = {}

        starts = assets.starts
        if episodes and episodes > 0:
            starts = starts[:episodes]
        self.episodes: List[Episode] = [self._make_episode(s) for s in starts]
        if not self.episodes:
            raise RuntimeError("mall 资产里没有可用起点（%s）" % assets.root)
        self.current_episode = self.episodes[0]

        self.scene = self.assets.scene_id
        self.sim = MallSim(assets, self.client, forward_step, turn_angle)
        self._stopped = False
        self._steps = 0
        self.trail: List[tuple] = []
        self.collisions = 0

    # -- episodes ----------------------------------------------------------
    def _pick_goal(self, x: float, z: float, rng) -> tuple:
        """从专家距离落在 ``[EXPERT_MIN_M, EXPERT_MAX_M]`` 的商铺里确定性抽一个。

        太近（端点 snap 到同一拓扑节点 → expert≈0 → SPL 退化）或太远的都不进
        目标池，与训练侧 build 的 ``4<d<80`` 过滤同款。全不合格时退回「离区间
        中值最近」的那家，保证总有非退化目标。返回 ``(goal, expert_len)``。"""
        goals = self.assets.goal_shops()
        if not goals:
            return None, 0.0
        scored = []
        for g in goals:
            gx, gz = float(g["front_world"][0]), float(g["front_world"][1])
            scored.append((g, gx, gz, self.assets.expert_length(x, z, gx, gz)))
        valid = [t for t in scored if EXPERT_MIN_M <= t[3] <= EXPERT_MAX_M]
        if valid:
            g, gx, gz, L = valid[int(rng.integers(len(valid)))]
        else:  # 兜底：离区间中值最近（仍非退化）
            mid = 0.5 * (EXPERT_MIN_M + EXPERT_MAX_M)
            g, gx, gz, L = min(scored, key=lambda t: abs(t[3] - mid))
        return {"name": g["name"], "front_world": [gx, gz]}, L

    def _make_episode(self, start: dict) -> Episode:
        """One start → one task. The goal shop is drawn deterministically from
        (seed, start idx), so episode N is the same task on every boot."""
        rng = np.random.default_rng(
            [self.seed, zlib.crc32(self.assets.scene_id.encode()), int(start["idx"])])
        x, z = float(start["x"]), float(start["z"])
        goal, expert = self._pick_goal(x, z, rng)
        gx = gz = None
        if goal is not None:
            gx, gz = float(goal["front_world"][0]), float(goal["front_world"][1])
        yaw = math.radians(float(start.get("yaw_deg", 0.0)))
        return Episode(
            episode_id=str(start["idx"]),
            scene=self.assets.scene_id,
            scene_id=os.path.join(self.assets.root, self.assets.scene_id + ".gs.ply"),
            instruction=Instruction(
                ("前往%s" % goal["name"]) if goal else ""),
            start_position=np.array([x, CAMERA_HEIGHT, z]),
            start_rotation=yaw_to_quat(yaw),
            goal_name=goal["name"] if goal else None,
            goal_xz=(gx, gz) if goal else None,
            expert_len_m=float(expert),
            info={"goal_label": goal["name"] if goal else "",
                  "euclidean_distance":
                      float(math.hypot(gx - x, gz - z)) if goal else 0.0})

    def new_episode(self, scene: Optional[str] = None) -> Episode:
        """A sampled free-roam spawn (no goal, hence NaN metrics) -- the
        console's "drop me somewhere" affordance."""
        rng = np.random.default_rng(
            [self.seed, zlib.crc32(self.assets.scene_id.encode()), len(self.episodes)])
        m = self.assets.mask
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            raise RuntimeError("mall 资产没有可走像素")
        i = int(rng.integers(len(xs)))
        x, z = self.assets.to_world(float(xs[i]), float(ys[i]))
        ep = Episode(
            episode_id="free%d" % sum(1 for e in self.episodes
                                       if e.goal_xz is None),
            scene=self.assets.scene_id,
            scene_id=os.path.join(self.assets.root, self.assets.scene_id + ".gs.ply"),
            instruction=Instruction(""),
            start_position=np.array([x, CAMERA_HEIGHT, z]),
            start_rotation=yaw_to_quat(float(rng.random() * 2 * math.pi)),
            info={})
        self.episodes.append(ep)
        self.current_episode = ep
        return ep

    def title_of(self, scene: str) -> str:
        return self.assets.scene_label

    def description_of(self, scene: str) -> str:
        g = self.assets.goal_shops()
        names = [s["name"] for s in g[:8]]
        return "%s · %d 起点 × %d %s目标（%s…）" % (
            self.assets.scene_id, len(self.assets.starts), len(g),
            self.assets.target_label, "、".join(names))

    @property
    def scene_description(self) -> str:
        return self.description_of(self.scene)

    # -- gym-ish API -------------------------------------------------------
    def reset(self) -> dict:
        ep = self.current_episode
        self.sim.set_agent_state(ep.start_position, ep.start_rotation)
        self._stopped = False
        self._steps = 0
        self.collisions = 0
        self.trail = [self.sim.xz]
        return self.sim.observe()

    def step(self, action, render: bool = True, **kwargs) -> dict:
        """One kinematic sub-step. ``render=False`` runs identical kinematics
        but skips the render HTTP (closed-loop FM chunks render once at the
        stopping pose); returns ``{"rgb": None}`` then."""
        assert not self.episode_over, "episode is over; call reset()"
        aid = action if isinstance(action, int) else int(action.get("action"))
        if aid == STOP:
            self._stopped = True
        else:
            self.sim.act(aid)
            if self.sim.previous_step_collided:
                self.collisions += 1
        self._steps += 1
        self.trail.append(self.sim.xz)
        if not render:
            return {"rgb": None}
        return self.sim.observe()

    @property
    def episode_over(self) -> bool:
        if self._stopped:
            return True
        return self.max_episode_steps != 0 and self._steps >= self.max_episode_steps

    def teleport(self, position) -> dict:
        """Jump to a world point keeping the heading, and re-observe."""
        st = self.sim.get_agent_state()
        self.sim.set_agent_state(position, st.rotation)
        self.trail = [self.sim.xz]
        return self.sim.observe()

    # -- scoring -----------------------------------------------------------
    def distance_to_goal(self) -> float:
        g = self.current_episode.goal_xz
        if not g:
            return float("nan")
        x, z = self.sim.xz
        return float(math.hypot(x - g[0], z - g[1]))

    def get_metrics(self) -> dict:
        """habitat-shaped metrics, scored ABotN-POIBench style.

        ``success`` turns 1 only once STOP is called inside the 2 m arrival
        radius; SPL uses the topo-graph Dijkstra length (euclidean fallback).

        ``position``/``heading_deg`` are reported in the FM chunk frame
        (``[x, −z, 1.5]``, ``habitat_yaw + 90°``) so ``forward=(cos h, sin h)``
        -- the convention ``vln_gym.env._world_pose_rad`` / ``fm_action`` read.
        """
        ep = self.current_episode
        d = self.distance_to_goal()
        walked = self.path_length()
        if ep.goal_xz is None:
            succ = spl = short = float("nan")
        else:
            succ = 1.0 if (self._stopped and d <= ARRIVE_THRESHOLD) else 0.0
            short = max(ep.expert_len_m, 1e-3)
            spl = succ * short / max(walked, short)
        x, z = self.sim.xz
        return {
            "distance_to_goal": d,
            "success": succ,
            "spl": spl,
            "success_radius": (float("nan") if ep.goal_xz is None
                               else ARRIVE_THRESHOLD),
            "shortest_path_m": short,
            "steps": self._steps,
            "collisions": self.collisions,
            "position": [round(x, 3), round(-z, 3), CAMERA_HEIGHT],
            "heading_deg": round((math.degrees(self.sim.yaw) + 90.0) % 360.0, 1),
            "path_length_m": round(walked, 2),
            "top_down_map": None,  # use render_top_down()
        }

    def path_length(self) -> float:
        return float(sum(math.hypot(b[0] - a[0], b[1] - a[1])
                         for a, b in zip(self.trail, self.trail[1:])))

    def close(self) -> None:
        """Deliberately does NOT unload the scene from the render server: the
        GS process is typically shared with training move servers, and an
        unload there would force a ~40 s cold reload on their next render.
        Set ``MALL_UNLOAD_ON_CLOSE=1`` to opt in."""
        if os.environ.get(self.assets.unload_env) == "1":
            try:
                self.client.unload(self.assets.scene_id)
            except Exception:
                pass

    # -- visualisation -----------------------------------------------------
    def render_top_down(self, size: int = 512, trail: bool = True,
                        show_goal: bool = False) -> np.ndarray:
        """Bird's-eye view: the floorplan (or the dilated walk mask if the
        image is missing) as base, trail, agent arrow and the goal marker."""
        from PIL import Image, ImageDraw

        base, to_px = self._top_down_base(size)
        img = base.copy()
        d = ImageDraw.Draw(img)
        ep = self.current_episode
        if trail and len(self.trail) > 1:
            d.line([to_px(p) for p in self.trail], fill=(90, 190, 255), width=2)
        x, z = self.sim.xz
        cx, cy = to_px((x, z))
        tip = to_px((x - math.sin(self.sim.yaw), z - math.cos(self.sim.yaw)))
        d.line([(cx, cy), tip], fill=(255, 220, 0), width=3)
        draw_agent_arrow(d, (cx, cy), tip)
        if show_goal and ep.goal_xz:
            gp = to_px(ep.goal_xz)
            d.ellipse([gp[0] - 6, gp[1] - 6, gp[0] + 6, gp[1] + 6],
                      fill=(255, 80, 80), outline=(120, 20, 20), width=2)
        return np.asarray(img)

    def map_heading_degrees(self, size: int = 512) -> float:
        """Clockwise heading on the rendered map: north/up is zero degrees."""
        _, to_px = self._top_down_base(size)
        x, z = self.sim.xz
        center = to_px((x, z))
        tip = to_px((x - math.sin(self.sim.yaw), z - math.cos(self.sim.yaw)))
        return math.degrees(
            math.atan2(tip[0] - center[0], center[1] - tip[1])) % 360.0

    def _top_down_base(self, size: int):
        from PIL import Image

        cached = self._map_cache.get(size)
        if cached is not None:
            return cached
        a = self.assets
        H, W = a.mask.shape
        pad = 8
        s = (size - 2 * pad) / float(max(W, H))       # px per mask pixel
        img = Image.new("RGB", (size, size), (18, 20, 24))
        # 底图优先用仓库自带平面图（与 floor_meta 标定同源，像素 1:1 对 mask）；
        # 缺失时退回可走域灰度图（可走=亮、障碍=暗）。
        fp = self.assets.floorplan_path
        if os.path.isfile(fp):
            small = Image.open(fp).convert("RGB").resize(
                (max(1, int(W * s)), max(1, int(H * s))), Image.LANCZOS)
        else:
            small = Image.fromarray(
                np.where(a.mask, 96, 24).astype(np.uint8), mode="L") \
                .resize((max(1, int(W * s)), max(1, int(H * s))), Image.NEAREST) \
                .convert("RGB")
        ox = (size - small.width) // 2
        oy = (size - small.height) // 2
        img.paste(small, (ox, oy))

        def to_px(xz):
            p, q = a.to_pixel(float(xz[0]), float(xz[1]))
            return (ox + p * s, oy + q * s)

        def from_px(px, py):
            return a.to_world((px - ox) / s, (py - oy) / s)

        self._map_cache[size] = (img, to_px)
        self._map_inv[size] = from_px
        return img, to_px

    def map_to_world(self, fx: float, fy: float):
        """A click on the bird's-eye view -> a habitat-shaped world point."""
        self._top_down_base(512)                        # ensure mapping cached
        from_px = self._map_inv[512]
        x, z = from_px(fx * 512.0, fy * 512.0)
        return np.array([x, CAMERA_HEIGHT, z], dtype=float)
