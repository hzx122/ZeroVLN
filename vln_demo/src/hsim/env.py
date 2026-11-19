"""A habitat_sim env for scene datasets that ship no VLN trajectory.

``vln_demo`` drives a simulator through a small duck-typed surface: ``reset()``,
``step()``, ``get_metrics()``, ``episode_over``, ``episodes``,
``current_episode`` -- and, from ``agent_tools``, ``env.sim.agents[0]``,
``env.sim.get_agent_state()`` and ``env.sim.previous_step_collided``.  Those last
three belong to habitat-*lab*'s wrapper, not to habitat_sim's own ``Simulator``,
so :class:`SimFacade` adds exactly them and delegates everything else through.

Two asset sets ride on this (see :mod:`hsim.datasets`):

  * **PARTNR / HSSD** -- 50 textured HSSD scenes with the official PARTNR
    episodes, which do carry a start pose and an English instruction.
  * **Habitat-HAB / ReplicaCAD** -- the two hab2 benchmark stages, which ship no
    navigation episodes, so spawns are sampled the way 3DRS's are.

Neither ships a ground-truth *path*: PARTNR scores rearrangement propositions,
not a trajectory.  ``distance_to_goal``/``success``/``spl`` are therefore NaN and
the classic driver stays off, exactly as for 3DRS.

Rendering is CPU (llvmpipe) like the R2R backend -- ``gpu_device_id`` follows
``HABITAT_SIM_GPU_DEVICE_ID`` so it never fights the vLLM servers for PPU memory.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from rs3d.env import draw_agent_arrow

STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2, 3


class SimFacade:
    """habitat_sim ``Simulator`` plus the three members habitat-lab would add.

    ``agent_tools`` reads ``sim.agents[0].agent_config.action_space`` to find and
    temporarily rewrite actuation amounts, ``sim.get_agent_state()`` for the pose
    and ``sim.previous_step_collided`` after each step.  habitat_sim spells the
    first two differently and reports collision as ``Agent.act()``'s return value,
    so the env sets ``previous_step_collided`` from there.  Everything else --
    ``pathfinder``, ``step``, ``reconfigure``, ``close`` -- passes through.
    """

    def __init__(self, sim):
        self._sim = sim
        self.previous_step_collided = False

    def __getattr__(self, name):
        return getattr(self._sim, name)

    @property
    def agents(self):
        return [self._sim.get_agent(0)]

    def get_agent_state(self, agent_id: int = 0):
        return self._sim.get_agent(agent_id).get_state()

    def set_agent_state(self, position, rotation=None, agent_id: int = 0) -> bool:
        agent = self._sim.get_agent(agent_id)
        st = agent.get_state()
        st.position = np.asarray(position, dtype=np.float32)
        if rotation is not None:
            st.rotation = rotation
        agent.set_state(st)
        self.previous_step_collided = False
        return True


@dataclass
class Instruction:
    """habitat's episode instruction object, of which only the text is read."""

    instruction_text: str


@dataclass
class Episode:
    """Minimal stand-in for a habitat episode.

    ``scene_id`` is synthesised as ``<root>/<scene>/<scene>.scene_instance.json``
    rather than pointing at the real file: the console identifies a scene as
    ``scene_id.split('/')[-2]``, and HSSD keeps every instance config in one flat
    directory, so the real path would name the wrong thing.
    """

    episode_id: str
    scene_id: str
    instruction: Instruction
    start_position: Optional[np.ndarray] = None
    start_rotation: Optional[Sequence[float]] = None  # (x, y, z, w)
    info: dict = field(default_factory=dict)

    @property
    def scene(self) -> str:
        return self.scene_id.split("/")[-2]


class HabSceneEnv:
    """``habitat.Env``-shaped wrapper around one habitat_sim ``Simulator``."""

    def __init__(self, dataset, scenes: Optional[Sequence[str]] = None,
                 width: int = 640, height: int = 480, hfov: float = 120.0,
                 forward_step: float = 0.25, turn_angle: float = 30.0,
                 sensor_height: float = 1.25, allow_sliding: bool = True,
                 want_depth: bool = False, seed: int = 0,
                 max_episode_steps: int = 0, episodes_per_scene: int = 5,
                 agent_height: float = 1.5, agent_radius: float = 0.1):
        import habitat_sim
        from habitat_sim.utils.common import quat_from_coeffs, quat_rotate_vector

        self._hs = habitat_sim
        self._quat_from_coeffs = quat_from_coeffs
        self._quat_rotate = quat_rotate_vector
        self.dataset = dataset
        self.seed = int(seed)
        self.max_episode_steps = int(max_episode_steps)
        self.want_depth = bool(want_depth)
        self._geom = dict(width=int(width), height=int(height), hfov=float(hfov),
                          forward_step=float(forward_step),
                          turn_angle=float(turn_angle),
                          sensor_height=float(sensor_height),
                          allow_sliding=bool(allow_sliding),
                          agent_height=float(agent_height),
                          agent_radius=float(agent_radius))
        self._lock = threading.Lock()
        self._map_cache: Dict[str, tuple] = {}
        # the same mapping the other way round, for map-click teleports
        self._map_inv: Dict[str, tuple] = {}

        self.episodes: List[Episode] = dataset.episodes(
            scenes, episodes_per_scene=episodes_per_scene, seed=self.seed)
        if not self.episodes:
            raise RuntimeError("%s 没有可用回合" % dataset.name)
        self.current_episode = self.episodes[0]

        self.scene = self.current_episode.scene
        self.sim = SimFacade(self._hs.Simulator(self._sim_config(self.scene)))
        self._ensure_navmesh()
        self._stopped = False
        self._steps = 0
        self.trail: List[np.ndarray] = []
        self.collisions = 0

    # -- configuration -----------------------------------------------------
    def _sim_config(self, scene: str):
        hs, g = self._hs, self._geom
        sim_cfg = hs.SimulatorConfiguration()
        sim_cfg.scene_dataset_config_file = self.dataset.config_file
        sim_cfg.scene_id = scene
        sim_cfg.enable_physics = False
        sim_cfg.allow_sliding = g["allow_sliding"]
        # -1 keeps the rasteriser on llvmpipe; the PPUs belong to vLLM.
        sim_cfg.gpu_device_id = int(os.environ.get("HABITAT_SIM_GPU_DEVICE_ID", "-1"))

        specs = []
        rgb = hs.CameraSensorSpec()
        rgb.uuid = "rgb"
        rgb.sensor_type = hs.SensorType.COLOR
        rgb.resolution = [g["height"], g["width"]]
        rgb.position = [0.0, g["sensor_height"], 0.0]
        rgb.hfov = g["hfov"]
        specs.append(rgb)
        if self.want_depth:
            d = hs.CameraSensorSpec()
            d.uuid = "depth"
            d.sensor_type = hs.SensorType.DEPTH
            d.resolution = [g["height"], g["width"]]
            d.position = [0.0, g["sensor_height"], 0.0]
            d.hfov = g["hfov"]
            specs.append(d)

        agent = hs.agent.AgentConfiguration()
        agent.height = g["agent_height"]
        agent.radius = g["agent_radius"]
        agent.sensor_specifications = specs
        # Keyed by the task's action indices so env.step(1) means forward for
        # both backends; agent_tools looks specs up by .name either way.
        agent.action_space = {
            MOVE_FORWARD: hs.agent.ActionSpec(
                "move_forward", hs.agent.ActuationSpec(amount=g["forward_step"])),
            TURN_LEFT: hs.agent.ActionSpec(
                "turn_left", hs.agent.ActuationSpec(amount=g["turn_angle"])),
            TURN_RIGHT: hs.agent.ActionSpec(
                "turn_right", hs.agent.ActuationSpec(amount=g["turn_angle"])),
        }
        return hs.Configuration(sim_cfg, [agent])

    def _ensure_navmesh(self) -> None:
        """Recompute the navmesh when the scene ships none.

        HSSD scene instances name a ``navmesh_instance`` but the file is not in
        the partnr download, and the hab2 stages have none at all -- Recast
        rebuilds one from the stage geometry in well under a second.
        """
        if self.sim.pathfinder.is_loaded:
            return
        ns = self._hs.nav.NavMeshSettings()
        ns.set_defaults()
        ns.agent_radius = self._geom["agent_radius"]
        ns.agent_height = self._geom["agent_height"]
        if not self.sim.recompute_navmesh(self.sim.pathfinder, ns):
            raise RuntimeError("场景 %s 建不出 navmesh" % self.scene)

    def _use_scene(self, scene: str) -> None:
        if scene == self.scene:
            return
        self.sim.reconfigure(self._sim_config(scene))
        self.scene = scene
        self._ensure_navmesh()

    # -- episodes ----------------------------------------------------------
    def _resolve_spawn(self, ep: Episode) -> None:
        """Fill in a start pose for ``ep`` if the dataset gave none (idempotent).

        Sampled from an rng seeded by (scene, episode_id), so "back to start"
        returns to the same pose and two console runs agree on what episode 3 is
        -- the same contract as the 3DRS backend.
        """
        if ep.start_position is not None:
            return
        import zlib
        rng = np.random.default_rng(
            [self.seed, zlib.crc32(ep.scene.encode()),
             int(ep.episode_id) if str(ep.episode_id).isdigit()
             else zlib.crc32(str(ep.episode_id).encode())])
        pf = self.sim.pathfinder
        best = None
        for _ in range(200):  # prefer the roomiest island, as free roam wants space
            p = pf.get_random_navigable_point()
            if not np.all(np.isfinite(p)):
                continue
            r = float(pf.distance_to_closest_obstacle(p, 2.0))
            if best is None or r > best[0]:
                best = (r, np.asarray(p, dtype=float))
            if r > 1.0:
                break
        if best is None:
            raise RuntimeError("场景 %s 采不到可走点" % ep.scene)
        ep.start_position = best[1]
        yaw = float(rng.random() * 2 * math.pi)
        ep.start_rotation = [0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2)]

    def description_of(self, scene: str) -> str:
        return self.dataset.description_of(scene)

    def title_of(self, scene: str) -> str:
        return self.dataset.title_of(scene)

    # -- gym-ish API -------------------------------------------------------
    def _obs(self, raw: dict) -> dict:
        # habitat_sim hands back RGBA; every consumer here wants three channels.
        out = {"rgb": np.asarray(raw["rgb"], dtype=np.uint8)[:, :, :3]}
        if "depth" in raw:
            out["depth"] = np.asarray(raw["depth"])
        return out

    def _position(self) -> np.ndarray:
        return np.asarray(self.sim.get_agent_state().position, dtype=float)

    def reset(self) -> dict:
        ep = self.current_episode
        self._use_scene(ep.scene)
        self._resolve_spawn(ep)
        rot = None
        if ep.start_rotation is not None:
            rot = self._quat_from_coeffs(
                np.asarray(ep.start_rotation, dtype=np.float32))
        self.sim.set_agent_state(ep.start_position, rot)
        self._stopped = False
        self._steps = 0
        self.collisions = 0
        self.trail = [self._position()]
        with self._lock:
            return self._obs(self.sim.get_sensor_observations())

    def step(self, action, **kwargs) -> dict:
        assert not self.episode_over, "episode is over; call reset()"
        aid = action if isinstance(action, int) else int(action.get("action"))
        with self._lock:
            if aid == STOP:
                self._stopped = True
            else:
                # Act through the agent rather than sim.step(): it returns the
                # collision flag, which is what agent_tools' stall detection
                # reads back off sim.previous_step_collided.
                collided = bool(self.sim.agents[0].act(aid))
                self.sim.previous_step_collided = collided
                if collided:
                    self.collisions += 1
            obs = self._obs(self.sim.get_sensor_observations())
        self._steps += 1
        self.trail.append(self._position())
        return obs

    @property
    def episode_over(self) -> bool:
        if self._stopped:
            return True
        return self.max_episode_steps != 0 and self._steps >= self.max_episode_steps

    @property
    def forward_xz(self) -> np.ndarray:
        """Unit heading in world XZ, straight off the agent's quaternion."""
        v = self._quat_rotate(self.sim.get_agent_state().rotation,
                              np.array([0.0, 0.0, -1.0]))
        return np.asarray([float(v[0]), 0.0, float(v[2])])

    def get_metrics(self) -> dict:
        """habitat-shaped metrics; goal-relative entries are NaN by design --
        neither asset set ships a ground-truth path to score against."""
        f = self.forward_xz
        return {
            "distance_to_goal": float("nan"),
            "success": float("nan"),
            "spl": float("nan"),
            "steps": self._steps,
            "collisions": self.collisions,
            "position": self._position().tolist(),
            "heading_deg": round(math.degrees(math.atan2(-f[0], -f[2])), 1),
            "path_length_m": round(float(sum(
                np.linalg.norm(b - a) for a, b in zip(self.trail, self.trail[1:]))), 2),
            "top_down_map": None,  # render_top_down() draws it off the pathfinder
        }

    def close(self) -> None:
        try:
            self.sim.close()
        except Exception:
            pass

    # -- visualisation -----------------------------------------------------
    def render_top_down(self, size: int = 512, trail: bool = True,
                        show_goal: bool = False) -> np.ndarray:
        """Bird's-eye view of the walkable surface, the trail and the agent.

        Rasterised from the pathfinder rather than habitat's ``top_down_map``
        measure (which only exists inside a habitat-lab task), and cached per
        scene so per-frame cost is just the trail and the marker.
        """
        from PIL import Image, ImageDraw

        base, to_px = self._top_down_base(size)
        img = base.copy()
        d = ImageDraw.Draw(img)
        if trail and len(self.trail) > 1:
            d.line([to_px(p) for p in self.trail], fill=(90, 190, 255), width=2)
        p = self._position()
        cx, cy = to_px(p)
        tip = to_px(p + self.forward_xz * 1.0)
        d.line([(cx, cy), tip], fill=(255, 220, 0), width=3)
        draw_agent_arrow(d, (cx, cy), tip)
        return np.asarray(img)

    def map_heading_degrees(self, size: int = 512) -> float:
        """Clockwise heading on the rendered map: north/up is zero degrees."""
        _, to_px = self._top_down_base(size)
        position = self._position()
        center = to_px(position)
        tip = to_px(position + self.forward_xz)
        return math.degrees(math.atan2(tip[0] - center[0], center[1] - tip[1])) % 360.0

    def _top_down_base(self, size: int):
        from PIL import Image

        height = float(self._position()[1])
        cached = self._map_cache.get(self.scene)
        if cached is not None and cached[0] == size and len(cached) > 3 and abs(cached[3] - height) < 0.05:
            return cached[1], cached[2]
        pf = self.sim.pathfinder
        lo, hi = (np.asarray(b, dtype=float) for b in pf.get_bounds())
        span = max(hi[0] - lo[0], hi[2] - lo[2], 1e-6)
        pad = 8
        scale = (size - 2 * pad) / span
        mpp = 1.0 / scale

        def to_px(p):
            return (pad + (p[0] - lo[0]) * scale, pad + (p[2] - lo[2]) * scale)

        def from_px(px, py):
            return (lo[0] + (px - pad) / scale, lo[2] + (py - pad) / scale)

        # [z][x] of booleans at the agent's own height, so upper floors of a
        # multi-storey scene do not paint over the one we are standing on.
        mask = np.asarray(pf.get_topdown_view(mpp, height))
        rgb = np.full(mask.shape + (3,), (18, 20, 24), dtype=np.uint8)
        rgb[mask] = (86, 96, 112)
        img = Image.fromarray(rgb).resize(
            (max(1, int(round((hi[0] - lo[0]) * scale))),
             max(1, int(round((hi[2] - lo[2]) * scale)))), Image.NEAREST)
        canvas = Image.new("RGB", (size, size), (18, 20, 24))
        canvas.paste(img, (pad, pad))
        self._map_cache[self.scene] = (size, canvas, to_px, height)
        self._map_inv[self.scene] = (size, size, from_px)
        return canvas, to_px

    def map_to_world(self, fx: float, fy: float):
        """A click on the bird's-eye view -> a navigable world point, or None.

        ``fx``/``fy`` are fractions of the rendered map, so they survive the
        rescaling between here and the browser.  Unlike the two Gaussian
        backends, ``SimFacade.set_agent_state`` does not snap, so the pathfinder
        does it here -- an unreachable click is refused rather than dropping the
        agent through the floor.
        """
        self._top_down_base(512)          # ensure the mapping exists (cached)
        w, h, from_px = self._map_inv[self.scene]
        x, z = from_px(fx * w, fy * h)
        y = float(self._position()[1])
        p = self.sim.pathfinder.snap_point(np.array([x, y, z], dtype=np.float32))
        p = np.asarray(p, dtype=float)
        return None if not np.all(np.isfinite(p)) else p

    def teleport(self, position) -> dict:
        """Jump to a world point keeping the heading, and re-observe.

        The trail restarts at the new pose: a teleport is not a walk, and the
        straight line from the old position would read as one.
        """
        with self._lock:
            self.sim.set_agent_state(np.asarray(position, dtype=np.float32))
            obs = self._obs(self.sim.get_sensor_observations())
        self.trail = [self._position()]
        return obs
