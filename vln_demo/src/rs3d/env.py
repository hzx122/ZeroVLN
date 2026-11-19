"""A habitat-shaped simulator for 3DRS scenes.

``vln_demo`` drives habitat through a small duck-typed surface: ``env.reset()``,
``env.step(action_id)``, ``env.get_metrics()``, ``env.episode_over`` and --
from ``agent_tools`` -- ``env.sim.agents[0].agent_config.action_space``,
``env.sim.get_agent_state()`` and ``env.sim.previous_step_collided``.  This
module implements exactly that surface on top of

  * the ABotN 3DGS render server for RGB/depth  (``rs3d.gs_client``), and
  * the scene's own ``.navmesh`` for kinematics  (``rs3d.nav``),

so ``agent_tools.py`` -- sub-stepping, stall detection, actuation save/restore
-- runs against 3DRS unmodified.  habitat is never imported.

Defaults mirror ``src/vln_demo.yaml`` (640x480, hfov 120, 0.25 m forward step,
30 deg turn, sliding on, unlimited steps) so behaviour is comparable with the
habitat runs.

What is deliberately *not* provided: goals.  3DRS ships scenes, not episodes,
so there is no ground-truth path and ``distance_to_goal``/``success`` are NaN.
Free roam and instruction-following demos work; VLN-CE scoring does not.
"""

from __future__ import annotations

import json
import math
import os
import threading
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .gs_client import CameraSpec, GSRenderClient
from .nav import NavQuery, forward_xz, quat_to_yaw, yaw_to_quat
from .navmesh import load_navmesh

RS3D_ROOT = os.environ.get("RS3D_ROOT", "/workspace/agentic_sim/3DRS")

STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2, 3
ACTION_NAME = {MOVE_FORWARD: "move_forward", TURN_LEFT: "turn_left",
               TURN_RIGHT: "turn_right", STOP: "stop"}


def draw_agent_arrow(draw, center, heading_tip, radius: float = 8.0) -> None:
    """Draw an R2R-like agent arrow centred at ``center``.

    ``heading_tip`` only supplies the direction; the separately drawn yellow
    one-metre ruler remains responsible for showing scale.
    """
    cx, cy = center
    dx, dy = heading_tip[0] - cx, heading_tip[1] - cy
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    ring = radius + 1.0
    draw.ellipse([cx - ring, cy - ring, cx + ring, cy + ring],
                 fill=(238, 240, 242), outline=(80, 86, 94), width=1)
    front = (cx + ux * radius, cy + uy * radius)
    left = (cx - ux * radius * 0.68 + px * radius * 0.62,
            cy - uy * radius * 0.68 + py * radius * 0.62)
    notch = (cx - ux * radius * 0.20, cy - uy * radius * 0.20)
    right = (cx - ux * radius * 0.68 - px * radius * 0.62,
             cy - uy * radius * 0.68 - py * radius * 0.62)
    draw.polygon([front, left, notch, right], fill=(55, 105, 200))


# ----------------------------------------------------------------------------
# habitat look-alikes (only the attributes vln_demo actually reads)
# ----------------------------------------------------------------------------
@dataclass
class Actuation:
    amount: float


@dataclass
class ActionSpec:
    name: str
    actuation: Actuation


@dataclass
class AgentConfig:
    action_space: Dict[int, ActionSpec]
    height: float = 1.5
    radius: float = 0.1
    sensor_height: float = 1.25


@dataclass
class Quaternion:
    w: float
    x: float
    y: float
    z: float


@dataclass
class AgentState:
    position: np.ndarray
    rotation: Quaternion


@dataclass
class Agent:
    agent_config: AgentConfig


@dataclass
class Instruction:
    """habitat's episode instruction object, of which only the text is read."""

    instruction_text: str


@dataclass
class Episode:
    """Minimal stand-in for a habitat VLN episode.

    ``start_position`` is None until the episode is first reset: resolving it
    means loading that scene's navmesh, and the console builds the whole episode
    list up front just to fill a dropdown.  The spawn is then sampled from an
    rng seeded by (scene, episode_id), so "back to start" really does return to
    the same pose, and two runs of the console agree on what "episode 3" is.
    """

    episode_id: str
    scene_id: str  # ".../<scene>/<scene>.gs.ply" so scene_id.split('/')[-2] works
    instruction: Instruction
    start_position: Optional[np.ndarray] = None
    start_rotation: Optional[Sequence[float]] = None  # (w, x, y, z)
    info: dict = field(default_factory=dict)

    @property
    def scene(self) -> str:
        return self.scene_id.split("/")[-2]


# ----------------------------------------------------------------------------
# the simulator
# ----------------------------------------------------------------------------
class RS3DSim:
    """Pose bookkeeping + rendering for one scene."""

    def __init__(self, scene: str, nav: NavQuery, client: GSRenderClient,
                 forward_step: float = 0.25, turn_angle: float = 30.0,
                 sensor_height: float = 1.25, allow_sliding: bool = True,
                 want_depth: bool = False):
        self.scene = scene
        self.nav = nav
        self.client = client
        self.allow_sliding = allow_sliding
        self.want_depth = want_depth
        self.previous_step_collided = False
        self._lock = threading.Lock()
        self.agents = [Agent(AgentConfig(
            action_space={
                MOVE_FORWARD: ActionSpec("move_forward", Actuation(float(forward_step))),
                TURN_LEFT: ActionSpec("turn_left", Actuation(float(turn_angle))),
                TURN_RIGHT: ActionSpec("turn_right", Actuation(float(turn_angle))),
            },
            sensor_height=float(sensor_height)))]
        self._pos = np.zeros(3)
        self._yaw = 0.0

    # -- state -------------------------------------------------------------
    @property
    def sensor_height(self) -> float:
        return self.agents[0].agent_config.sensor_height

    def get_agent_state(self, agent_id: int = 0) -> AgentState:
        return AgentState(self._pos.copy(), Quaternion(*yaw_to_quat(self._yaw)))

    def set_agent_state(self, position: Sequence[float],
                        rotation: Optional[Sequence[float]] = None,
                        agent_id: int = 0) -> bool:
        p = np.asarray(position, dtype=float)
        snapped = self.nav.snap(p) if self.nav.is_navigable(p) else self.nav.snap_nearest(p)
        self._pos = snapped if snapped is not None else p
        if rotation is not None:
            self._yaw = quat_to_yaw(*rotation) if len(rotation) == 4 else float(rotation[0])
        self.previous_step_collided = False
        return True

    @property
    def yaw(self) -> float:
        return self._yaw

    # -- actions -----------------------------------------------------------
    def act(self, action_id: int) -> None:
        """Apply one action using the *current* actuation amount.

        Reading the amount at call time is what makes ``agent_tools``' trick of
        temporarily rewriting ``actuation.amount`` work -- exactly as it does
        against habitat_sim.
        """
        spec = self.agents[0].agent_config.action_space.get(action_id)
        if spec is None:
            raise KeyError("no action %r" % (action_id,))
        amount = float(spec.actuation.amount)
        if spec.name == "move_forward":
            target = self._pos + forward_xz(self._yaw) * amount
            new, hit = self.nav.try_step(self._pos, target, slide=self.allow_sliding)
            self._pos = new
            self.previous_step_collided = bool(hit)
        elif spec.name in ("turn_left", "turn_right"):
            sign = 1.0 if spec.name == "turn_left" else -1.0
            self._yaw = _wrap_pi(self._yaw + sign * math.radians(amount))
            self.previous_step_collided = False

    # -- sensors -----------------------------------------------------------
    def observe(self) -> dict:
        eye = self._pos + np.array([0.0, self.sensor_height, 0.0])
        with self._lock:  # one in-flight render per scene keeps the server calm
            rgb, depth = self.client.render_one(
                self.scene, eye, self._yaw, want_depth=self.want_depth)
        obs = {"rgb": np.asarray(rgb, dtype=np.uint8)}
        if depth is not None:
            obs["depth"] = depth
        return obs


class RS3DEnv:
    """``habitat.Env``-shaped wrapper around :class:`RS3DSim`."""

    def __init__(self, scenes, client: Optional[GSRenderClient] = None,
                 rs3d_root: str = RS3D_ROOT, width: int = 640, height: int = 480,
                 hfov: float = 120.0, forward_step: float = 0.25,
                 turn_angle: float = 30.0, sensor_height: float = 1.25,
                 allow_sliding: bool = True, want_depth: bool = False,
                 seed: int = 0, max_episode_steps: int = 0,
                 episodes_per_scene: int = 1,
                 render_url: str = "http://127.0.0.1:7036"):
        if isinstance(scenes, str):
            scenes = [scenes]
        self.scenes = list(scenes)
        self.rs3d_root = rs3d_root
        self.max_episode_steps = int(max_episode_steps)
        self.seed = int(seed)

        self.client = client or GSRenderClient(
            url=render_url, camera=CameraSpec(width, height, hfov))
        self._nav_cache: Dict[str, NavQuery] = {}
        self._meta_cache: Dict[str, dict] = {}
        self._map_cache: Dict[str, tuple] = {}
        # the same mapping the other way round, for map-click teleports
        self._map_inv: Dict[str, tuple] = {}

        self.scene = self.scenes[0]
        self.nav = self._nav_for(self.scene)
        self.sim = RS3DSim(self.scene, self.nav, self.client, forward_step,
                           turn_angle, sensor_height, allow_sliding, want_depth)

        self.episodes: List[Episode] = [
            self._make_episode(s, i)
            for s in self.scenes for i in range(episodes_per_scene)]
        self.current_episode = self.episodes[0]
        self._stopped = False
        self._steps = 0
        self.trail: List[np.ndarray] = []
        self.collisions = 0

    # -- scenes ------------------------------------------------------------
    def _nav_for(self, scene: str) -> NavQuery:
        if scene not in self._nav_cache:
            path = os.path.join(self.rs3d_root, scene, "%s.navmesh" % scene)
            self._nav_cache[scene] = NavQuery(load_navmesh(path))
        return self._nav_cache[scene]

    def _meta_for(self, scene: str) -> dict:
        if scene not in self._meta_cache:
            self._meta_cache[scene] = _load_meta(self.rs3d_root, scene)
        return self._meta_cache[scene]

    def _use_scene(self, scene: str) -> None:
        """Point the sim at ``scene``, loading it on the render server.

        Called from ``reset()``, which is how the console switches scenes: it
        assigns ``env.current_episode`` and resets, exactly as against habitat.
        The server keeps its own LRU pool, so we only ever ask it to preload.
        """
        if scene == self.scene and self.nav is self._nav_cache.get(scene):
            return
        self.nav = self._nav_for(scene)
        self.scene = scene
        self.sim.scene = scene
        self.sim.nav = self.nav
        self.client.preload(scene)

    # -- episodes ----------------------------------------------------------
    def _make_episode(self, scene: str, i: int) -> Episode:
        desc = (self._meta_for(scene).get("metainfo", {}) or {}).get("description", "")
        return Episode(
            episode_id=str(i),
            scene_id=os.path.join(self.rs3d_root, scene, "%s.gs.ply" % scene),
            instruction=Instruction(default_instruction(desc)),
            info={"scene_description": desc})

    def _resolve_spawn(self, ep: Episode) -> None:
        """Fill in a reproducible spawn pose for ``ep`` (idempotent)."""
        if ep.start_position is not None:
            return
        nav = self._nav_for(ep.scene)
        rng = np.random.default_rng(
            [self.seed, zlib.crc32(ep.scene.encode()), _int_or_hash(ep.episode_id)])
        p = nav.sample_navigable(rng)
        if p is None:
            raise RuntimeError("scene %s has no walkable surface" % ep.scene)
        ep.start_position = p
        ep.start_rotation = yaw_to_quat(float(rng.random() * 2 * math.pi))

    def new_episode(self, scene: Optional[str] = None) -> Episode:
        """Append a freshly sampled spawn (3DRS ships no episode list)."""
        scene = scene or self.scene
        n = sum(1 for e in self.episodes if e.scene == scene)
        ep = self._make_episode(scene, n)
        self.episodes.append(ep)
        self.current_episode = ep
        return ep

    @property
    def scene_description(self) -> str:
        return self.description_of(self.scene)

    def description_of(self, scene: str) -> str:
        """The metainfo caption for any scene, not just the loaded one."""
        return (self._meta_for(scene).get("metainfo", {}) or {}).get("description", "")

    def title_of(self, scene: str) -> str:
        """Short label for the scene dropdown; "" when the scene has no caption.

        Cheap to ask for every scene: the metainfo is already cached, since
        building the episode list reads it for all of them.
        """
        return scene_title(self.description_of(scene))

    @property
    def scene_type(self) -> str:
        return (self._meta_for(self.scene).get("metainfo", {}) or {}).get("scene_type", "")

    # -- gym-ish API -------------------------------------------------------
    def reset(self) -> dict:
        ep = self.current_episode
        self._use_scene(ep.scene)
        self._resolve_spawn(ep)
        self.sim.set_agent_state(ep.start_position, ep.start_rotation)
        self._stopped = False
        self._steps = 0
        self.collisions = 0
        self.trail = [self.sim.get_agent_state().position.copy()]
        return self.sim.observe()

    def step(self, action, **kwargs) -> dict:
        assert not self.episode_over, "episode is over; call reset()"
        aid = action if isinstance(action, int) else int(action.get("action"))
        if aid == STOP:
            self._stopped = True
        else:
            self.sim.act(aid)
            if self.sim.previous_step_collided:
                self.collisions += 1
        self._steps += 1
        self.trail.append(self.sim.get_agent_state().position.copy())
        return self.sim.observe()

    def teleport(self, position) -> dict:
        """Jump to a world point keeping the heading, and re-observe.

        Used by the double-click-to-jump on the bird's-eye view.  The trail
        restarts at the new pose: a teleport is not a walk, and drawing the
        straight line from the old position would read as one.
        """
        st = self.sim.get_agent_state()
        self.sim.set_agent_state(position, st.rotation)
        self.trail = [self.sim.get_agent_state().position.copy()]
        return self.sim.observe()

    @property
    def episode_over(self) -> bool:
        if self._stopped:
            return True
        return self.max_episode_steps != 0 and self._steps >= self.max_episode_steps

    def get_metrics(self) -> dict:
        """habitat-shaped metrics.  Goal-relative entries are NaN by design --
        3DRS has no ground-truth episodes, so there is nothing to score."""
        st = self.sim.get_agent_state()
        return {
            "distance_to_goal": float("nan"),
            "success": float("nan"),
            "spl": float("nan"),
            "steps": self._steps,
            "collisions": self.collisions,
            "position": st.position.tolist(),
            "heading_deg": round(math.degrees(self.sim.yaw), 1),
            "path_length_m": round(float(sum(
                np.linalg.norm(b - a) for a, b in zip(self.trail, self.trail[1:]))), 2),
            "top_down_map": None,  # use render_top_down(); habitat's format is not reproduced
        }

    def close(self) -> None:
        try:
            self.client.unload(self.scene)
        except Exception:
            pass

    # -- visualisation -----------------------------------------------------
    def render_top_down(self, size: int = 512, trail: bool = True,
                        show_goal: bool = False) -> np.ndarray:
        """Bird's-eye view of the walkable surface, the trail and the agent.

        Drawn from the navmesh rather than habitat's ``top_down_map`` measure:
        the base layer is rasterised once per scene and cached, so per-frame
        cost is just the trail and the agent marker.
        """
        from PIL import Image, ImageDraw

        base, to_px = self._top_down_base(size)
        img = base.copy()
        d = ImageDraw.Draw(img)
        if trail and len(self.trail) > 1:
            d.line([to_px(p) for p in self.trail], fill=(90, 190, 255), width=2)
        st = self.sim.get_agent_state()
        cx, cy = to_px(st.position)
        f = forward_xz(self.sim.yaw)
        tip = to_px(st.position + f * 1.0)
        d.line([(cx, cy), tip], fill=(255, 220, 0), width=3)
        draw_agent_arrow(d, (cx, cy), tip)
        return np.asarray(img)

    def map_heading_degrees(self, size: int = 512) -> float:
        """Clockwise heading on the rendered map: north/up is zero degrees."""
        _, to_px = self._top_down_base(size)
        state = self.sim.get_agent_state()
        center = to_px(state.position)
        tip = to_px(state.position + forward_xz(self.sim.yaw))
        return math.degrees(math.atan2(tip[0] - center[0], center[1] - tip[1])) % 360.0

    def _top_down_base(self, size: int):
        from PIL import Image, ImageDraw

        cached = self._map_cache.get(self.scene)
        if cached is not None and cached[0] == size:
            return cached[1], cached[2]
        tris = self.nav.tris
        lo = tris.reshape(-1, 3).min(axis=0)
        hi = tris.reshape(-1, 3).max(axis=0)
        span = max(hi[0] - lo[0], hi[2] - lo[2], 1e-6)
        pad = 8
        scale = (size - 2 * pad) / span

        def to_px(p):
            return (pad + (p[0] - lo[0]) * scale, pad + (p[2] - lo[2]) * scale)

        def from_px(px, py):
            return (lo[0] + (px - pad) / scale, lo[2] + (py - pad) / scale)

        img = Image.new("RGB", (size, size), (18, 20, 24))
        d = ImageDraw.Draw(img)
        # every island in a dim slate, the largest one brighter: the agent can
        # only ever be on one island, and knowing which matters when reading it
        for k, sel in enumerate(self.nav.islands()):
            col = (86, 96, 112) if k == 0 else (46, 52, 62)
            for t in tris[sel]:
                d.polygon([to_px(t[0]), to_px(t[1]), to_px(t[2])], fill=col)
        self._map_cache[self.scene] = (size, img, to_px)
        self._map_inv[self.scene] = (size, size, from_px)
        return img, to_px

    def map_to_world(self, fx: float, fy: float):
        """A click on the bird's-eye view -> a world point, or ``None``.

        ``fx``/``fy`` are fractions of the rendered map's width/height, so they
        survive whatever the browser and the JPEG path scaled the image to.
        Height comes from the current pose and gets snapped by
        ``set_agent_state``; the navmesh is what decides where we actually land.
        """
        self._top_down_base(512)          # ensure the mapping exists (cached)
        w, h, from_px = self._map_inv[self.scene]
        x, z = from_px(fx * w, fy * h)
        y = float(self.sim.get_agent_state().position[1])
        return np.array([x, y, z], dtype=float)


# ----------------------------------------------------------------------------
def list_scenes(rs3d_root: str = RS3D_ROOT) -> List[str]:
    out = []
    for d in sorted(os.listdir(rs3d_root)):
        if os.path.exists(os.path.join(rs3d_root, d, "%s.navmesh" % d)):
            out.append(d)
    return out


def _load_meta(rs3d_root: str, scene: str) -> dict:
    p = os.path.join(rs3d_root, scene, "%s_metainfo.json" % scene)
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def default_instruction(description: str) -> str:
    """The scene's own metainfo caption, verbatim.

    habitat episodes come with a dataset instruction that success is scored
    against. 3DRS ships no such thing -- only this caption, one VLM-written
    paragraph about what the place looks like. It is not a navigation order and
    nothing is scored against it, but it is the one piece of text the asset
    actually carries, so it goes in the instruction box as the starting point:
    you see what the scene is and edit it into a task, instead of facing an
    empty box. Nothing is rewritten or invented here -- what you read is what
    the asset says. Captionless scenes still yield "" and the console then asks
    you to type something.
    """
    return (description or "").strip()


def scene_title(description: str, limit: int = 22) -> str:
    """A short human-readable title out of the metainfo caption.

    The captions are one VLM-written paragraph that opens by saying what the
    place *is* ("该场景为城市公园或绿地中的步行道区域，地面由方形石板铺成，…"),
    so the first clause with the boilerplate opener removed is already the
    title. Empty for scenes with no caption -- callers fall back to the id.
    """
    t = (description or "").strip()
    for opener in ("该场景为", "本场景为", "该场景是", "这是一个", "场景为", "该场景"):
        if t.startswith(opener):
            t = t[len(opener):]
            break
    for sep in ("，", "；", "。", ",", ";"):
        i = t.find(sep)
        if i > 0:
            t = t[:i]
            break
    t = t.strip(" 、,;：:")
    return t[:limit] + "…" if len(t) > limit else t


def _int_or_hash(s: str) -> int:
    return int(s) if str(s).isdigit() else zlib.crc32(str(s).encode())


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
