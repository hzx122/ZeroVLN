"""A habitat-shaped simulator for ABotN's Gaussian benchmarks.

Same contract as :mod:`rs3d.env` -- ``reset``/``step``/``get_metrics``/
``episode_over`` plus ``sim.agents[0].agent_config.action_space``,
``sim.get_agent_state()`` and ``sim.previous_step_collided`` -- so
``agent_tools.py`` and the ReAct loop run against ABotN unmodified.  What
differs from 3DRS is underneath:

* kinematics come from the scene's occupancy grid (:mod:`abotn.occmap`)
  rather than a navmesh, and
* **the episodes are real**.  Every ``traj_*.json`` carries a start pose, a
  goal point, a human reference path and (in POIBench) a Chinese instruction,
  so ``distance_to_goal``/``success``/``spl`` are actual numbers here, scored
  the way ABotN's own evaluator scores them.  This is the only non-R2R asset
  set in the console where that is true.

Two coordinate frames meet in this file, so keep them straight:

* **ABotN world** -- z up, x/y horizontal, yaw from +x towards +y, camera at
  ``ground_z + 0.65``.  Everything stored on the sim, sent to the renderer and
  reported in ``get_metrics()["position"]`` is in this frame, because that is
  the frame the annotations, the occ maps and the render server all speak.
* **habitat-shaped** -- y up, forward is -z.  Only ``get_agent_state()`` speaks
  it, and only because ``agent_tools`` reads ``position[0]``/``position[2]``
  and decodes a habitat quaternion.  The map is ``(x, y, z) -> (x, z, -y)``
  and ``psi = yaw - pi/2``; it is a rigid motion, so every distance and
  relative heading ``agent_tools`` measures is unaffected.
"""

from __future__ import annotations

import math
import os
import threading
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from rs3d.env import (Agent, AgentConfig, AgentState, ActionSpec, Actuation,
                      Instruction, Quaternion)
from rs3d.gs_client import CameraSpec
from rs3d.nav import quat_to_yaw, yaw_to_quat

from . import floorfit
from .map_style import CANVAS, draw_navigation_overlay
from . import occmap as occmod
from .datasets import ABotNSet, Task
from .gs_client import ABOTN_RENDER_URL, XYZEulerClient

STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2, 3

CAMERA_HEIGHT = 0.65        # ABotN's extrinsic_height, and what the GT poses use
ROBOT_RADIUS = 0.10         # footprint for the occ-map collision check


def to_habitat(x: float, y: float, z: float) -> np.ndarray:
    """ABotN world -> the y-up frame ``agent_tools`` expects."""
    return np.array([x, z, -y], dtype=float)


def from_habitat(p: Sequence[float]):
    return float(p[0]), -float(p[2]), float(p[1])


@dataclass
class Episode:
    """One ABotN task, in habitat-episode clothing.

    ``start_position``/``start_rotation`` are **ABotN world**, not habitat --
    nothing outside this module reads them, and keeping them native means the
    pose that goes to the renderer is byte-for-byte the annotated one.

    ``scene`` is a stored field rather than ``scene_id.split('/')[-2]`` the way
    the other backends do it: our scene ids carry a set prefix that the
    directory on disk does not, and deriving it from the path would hand the
    render server ``20260212160022`` -- a name that exists in two benchmarks.
    """

    episode_id: str
    scene: str                  # prefixed console id, e.g. "poi-20260212130612"
    scene_id: str               # on-disk path, for anything that wants a file
    instruction: Instruction
    start_position: Optional[np.ndarray] = None   # (x, y, z), z up
    start_rotation: float = 0.0                   # yaw, radians
    goal_xy: Optional[tuple] = None
    ref_len: float = 0.0
    waypoints: tuple = ()          # golden reference path (x, y, z), for the map
    arrive_threshold: float = 0.5
    spl_slack: float = 0.0
    z_offset: Optional[float] = None   # camera height above the height map
    info: dict = field(default_factory=dict)


class ABotNSim:
    """Pose bookkeeping + rendering for one scene."""

    def __init__(self, scene: str, occ: occmod.OccMap, client: XYZEulerClient,
                 z_offset: float = CAMERA_HEIGHT, forward_step: float = 0.25,
                 turn_angle: float = 30.0, allow_sliding: bool = True,
                 want_depth: bool = False):
        self.scene = scene
        self.occ = occ
        self.client = client
        self.z_offset = z_offset
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
            sensor_height=CAMERA_HEIGHT))]
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0

    # -- state -------------------------------------------------------------
    @property
    def xy(self):
        return self._x, self._y

    @property
    def yaw(self) -> float:
        """Heading in the *ABotN* frame (from +x towards +y)."""
        return self._yaw

    @property
    def camera_z(self) -> float:
        """Eye height in world metres.

        The ground comes from ``occ_map_height.tiff`` -- these scenes are not
        flat, and two POIBench ones are an elevated walkway 6 m and 20 m up --
        and ``z_offset`` is the eye height above it, 0.65 m for every annotated
        pose in all three benchmarks.
        """
        return self.ground(self._x, self._y) + self.z_offset

    def ground(self, x: float, y: float) -> float:
        return self.occ.ground_z(x, y)

    def get_agent_state(self, agent_id: int = 0) -> AgentState:
        """habitat-shaped state -- see the module docstring on the two frames."""
        return AgentState(to_habitat(self._x, self._y, self.camera_z),
                          Quaternion(*yaw_to_quat(self._yaw - math.pi / 2.0)))

    def set_agent_state(self, position: Sequence[float],
                        rotation=None, agent_id: int = 0, native: bool = False) -> bool:
        x, y = (float(position[0]), float(position[1])) if native \
            else from_habitat(position)[:2]
        if not self.occ.is_free(x, y, ROBOT_RADIUS):
            snapped = self._snap(x, y)
            if snapped is not None:
                x, y = snapped
        self._x, self._y = x, y
        if rotation is not None:
            if native:
                self._yaw = float(rotation)
            elif np.iterable(rotation) and len(rotation) == 4:
                self._yaw = quat_to_yaw(*rotation) + math.pi / 2.0
            else:
                self._yaw = float(np.ravel(rotation)[0])
        self.previous_step_collided = False
        return True

    def _snap(self, x: float, y: float, limit_m: float = 1.5):
        """Nearest free pixel within ``limit_m``.

        Annotated start poses sit right against shopfronts, and roughly 1% of
        them fall inside the robot footprint on the annotators' own map; rather
        than refuse the episode we nudge it out, exactly as ABotN's evaluator
        tolerates by dilating free space.
        """
        px, py = self.occ.to_pixel(x, y)
        lim = self.occ.radius_px(limit_m)
        r_px = self.occ.radius_px(ROBOT_RADIUS)
        for r in range(1, lim + 1):
            y0, y1 = max(0, py - r), min(self.occ.h, py + r + 1)
            x0, x1 = max(0, px - r), min(self.occ.w, px + r + 1)
            sub = self.occ.free[y0:y1, x0:x1]
            ys, xs = np.nonzero(sub)
            if len(xs) == 0:
                continue
            # ring order: closest candidate first, and it must fit the footprint
            order = np.argsort((xs + x0 - px) ** 2 + (ys + y0 - py) ** 2)
            for i in order:
                cx, cy = int(xs[i]) + x0, int(ys[i]) + y0
                patch = self.occ.free[max(0, cy - r_px):cy + r_px + 1,
                                      max(0, cx - r_px):cx + r_px + 1]
                if patch.size and patch.all():
                    return self.occ.to_world(float(cx), float(cy))
        return None

    # -- actions -----------------------------------------------------------
    def act(self, action_id: int) -> None:
        spec = self.agents[0].agent_config.action_space.get(action_id)
        if spec is None:
            raise KeyError("no action %r" % (action_id,))
        amount = float(spec.actuation.amount)
        if spec.name == "move_forward":
            tx = self._x + math.cos(self._yaw) * amount
            ty = self._y + math.sin(self._yaw) * amount
            nx, ny, hit = self.occ.try_step(self._x, self._y, tx, ty,
                                            ROBOT_RADIUS, self.allow_sliding)
            self._x, self._y = nx, ny
            self.previous_step_collided = bool(hit)
        elif spec.name in ("turn_left", "turn_right"):
            sign = 1.0 if spec.name == "turn_left" else -1.0
            self._yaw = _wrap_pi(self._yaw + sign * math.radians(amount))
            self.previous_step_collided = False

    # -- sensors -----------------------------------------------------------
    def observe(self) -> dict:
        with self._lock:  # one in-flight render per scene keeps the server calm
            rgb, depth = self.client.render_one(
                self.scene, (self._x, self._y, self.camera_z), self._yaw,
                want_depth=self.want_depth)
        obs = {"rgb": np.asarray(rgb, dtype=np.uint8)}
        if depth is not None:
            obs["depth"] = depth
        return obs


class ABotNEnv:
    """``habitat.Env``-shaped wrapper over one :class:`~abotn.datasets.ABotNSet`."""

    def __init__(self, dataset: ABotNSet, scenes=None, width: int = 640,
                 height: int = 480, hfov: float = 120.0, forward_step: float = 0.25,
                 turn_angle: float = 30.0, allow_sliding: bool = True,
                 want_depth: bool = False, seed: int = 0,
                 max_episode_steps: int = 0, episodes_per_scene: int = 0,
                 render_url: str = ABOTN_RENDER_URL,
                 client: Optional[XYZEulerClient] = None):
        self.ds = dataset
        self.seed = int(seed)
        self.max_episode_steps = int(max_episode_steps)
        want = [s for s in (scenes or ()) if s] or dataset.list_scenes()
        self.scenes = [s if s in dataset.list_scenes() else dataset.qualify(s)
                       for s in want]
        self.client = client or XYZEulerClient(
            url=render_url, camera=CameraSpec(width, height, hfov))
        self._occ_cache: Dict[str, occmod.OccMap] = {}
        self._map_cache: Dict[str, tuple] = {}
        # the same mapping the other way round, for map-click teleports
        self._map_inv: Dict[str, tuple] = {}

        self.episodes: List[Episode] = [
            self._make_episode(t)
            for s in self.scenes
            for t in _limit(dataset.tasks(s), episodes_per_scene)]
        if not self.episodes:
            raise RuntimeError("%s 里没有可用回合" % dataset.key)
        self.current_episode = self.episodes[0]

        self.scene = self.current_episode.scene
        self.sim = ABotNSim(self.scene, self._occ_for(self.scene), self.client,
                            self._z_for(self.scene), forward_step, turn_angle,
                            allow_sliding, want_depth)
        self._stopped = False
        self._steps = 0
        self.trail: List[tuple] = []
        self.collisions = 0

    # -- scenes ------------------------------------------------------------
    def _occ_for(self, scene: str) -> occmod.OccMap:
        if scene not in self._occ_cache:
            # Keep ABotN's flat 5 m hole guard. It used to be lifted per scene
            # to ``max waypoint z + 1``, on the reading that the height map's
            # 6 m and 20 m patches were a genuinely elevated walkway that the
            # 5 m ceiling was flattening to street level. The Gaussian clouds
            # say otherwise: in both of those scenes the floor sits at z ~= -0.4
            # like everywhere else, so those patches are buildings and noise in
            # the height map, not ground. Trusting them put the camera 7 m up in
            # 麦当劳 and 22 m up in 太田 -- looking down on a rooftop instead of
            # standing in the street. Checked the other way round against
            # point_cloud_rotated.ply for all 26 POI/outdoor scenes: with the
            # 5 m guard every one lands 0.7-2.2 m above its own floor.
            occ = occmod.OccMap(self.ds.scene_dir(scene))
            # What the 5 m guard cannot fix is that the tiff is 8-bit, i.e.
            # quantised to whole metres. The sub-metre correction comes off the
            # Gaussians, cached on disk; None means "no cloud / switched off",
            # in which case the height map is used raw, exactly as ABotN does.
            occ.z_delta = floorfit.delta_for(self.ds.scene_dir(scene), occ=occ,
                                             log=print)
            self._occ_cache[scene] = occ
        return self._occ_cache[scene]

    def _z_for(self, scene: str) -> float:
        """Eye height above the height map's ground, for this scene.

        ABotN's ``_adjust_pose_z`` is unconditional -- ``ground_z + 0.65``, the
        rig's ``extrinsic_height``, with no per-episode term -- so this is a
        constant, and it is checked rather than assumed: over all 628 tasks in
        the three benchmarks ``start.z - ground_z(start)`` is exactly 0.65 in
        610 of them. The 18 that differ are the 麦当劳 and 太田 tasks whose
        annotated start z (6.65 / 20.65) came from the height-map patches the
        5 m hole guard rejects, and taking the offset from those poses is what
        used to hoist the camera onto a rooftop. See ``_occ_for``.
        """
        return CAMERA_HEIGHT

    def _ground(self, scene: str, x: float, y: float) -> float:
        """Ground reference at ``(x, y)``, same rule the sim uses."""
        return self._occ_for(scene).ground_z(x, y)

    def _use_scene(self, scene: str) -> None:
        if scene == self.scene:
            return
        self.scene = scene
        self.sim.scene = scene
        self.sim.occ = self._occ_for(scene)
        self.sim.z_offset = self._z_for(scene)
        self.client.preload(scene)

    # -- episodes ----------------------------------------------------------
    def _make_episode(self, t: Task) -> Episode:
        return Episode(
            episode_id=t.task_id,
            scene=t.scene,
            scene_id=os.path.join(self.ds.scene_dir(t.scene),
                                  "%s.json" % self.ds.bare(t.scene)),
            instruction=Instruction(default_instruction(t)),
            start_position=np.array([t.start["x"], t.start["y"], t.start["z"]]),
            start_rotation=float(t.start["yaw"]),
            goal_xy=t.goal_xy,
            ref_len=float(t.ref_len),
            waypoints=t.waypoints,
            arrive_threshold=self.ds.arrive_threshold,
            spl_slack=self.ds.spl_slack,
            # Same constant every episode, not ``start.z - ground(start)``:
            # ABotN's own runtime overrides the annotated z with
            # ``ground_z + extrinsic_height`` too. See ``_z_for``.
            z_offset=CAMERA_HEIGHT,
            info={"scene_description": scene_blurb(self.ds, t.scene),
                  "goal_label": t.goal_label,
                  "distance_bucket": t.distance_bucket,
                  "euclidean_distance": t.euclidean})

    def new_episode(self, scene: Optional[str] = None) -> Episode:
        """A sampled free-roam spawn in ``scene`` (no goal, hence no score).

        The benchmarks ship enough real episodes that inventing one is rarely
        the right move, but "drop me somewhere in this scene" is still the
        fastest way to look around, so it stays available -- clearly marked
        with no goal rather than dressed up as a task.
        """
        scene = scene or self.scene
        occ = self._occ_for(scene)
        rng = np.random.default_rng([self.seed, zlib.crc32(scene.encode()),
                                     len(self.episodes)])
        p = occ.sample_free(rng, ROBOT_RADIUS)
        if p is None:
            raise RuntimeError("场景 %s 没有可站立的自由空间" % scene)
        ep = Episode(
            episode_id="free%d" % sum(1 for e in self.episodes
                                      if e.scene == scene and e.goal_xy is None),
            scene=scene,
            scene_id=os.path.join(self.ds.scene_dir(scene),
                                  "%s.json" % self.ds.bare(scene)),
            instruction=Instruction(""),
            start_position=np.array([p[0], p[1],
                                     self._ground(scene, *p) + self._z_for(scene)]),
            start_rotation=float(rng.random() * 2 * math.pi),
            z_offset=self._z_for(scene),
            info={"scene_description": scene_blurb(self.ds, scene)})
        self.episodes.append(ep)
        self.current_episode = ep
        return ep

    def title_of(self, scene: str) -> str:
        return self.ds.title_of(scene)

    def description_of(self, scene: str) -> str:
        return scene_blurb(self.ds, scene)

    @property
    def scene_description(self) -> str:
        return self.description_of(self.scene)

    # -- gym-ish API -------------------------------------------------------
    def reset(self) -> dict:
        ep = self.current_episode
        self._use_scene(ep.scene)
        if ep.z_offset is not None:
            self.sim.z_offset = float(ep.z_offset)
        self.sim.set_agent_state(ep.start_position, ep.start_rotation, native=True)
        self._stopped = False
        self._steps = 0
        self.collisions = 0
        self.trail = [self.sim.xy]
        return self.sim.observe()

    def step(self, action, render: bool = True, **kwargs) -> dict:
        """Advance one kinematic sub-step.

        ``render=False`` runs the exact same kinematics -- ``sim.act`` moves the
        agent, the occ-map still reports collisions and they still accumulate --
        but skips the ``sim.observe()`` render HTTP.  A closed-loop FM chunk is
        dozens of sub-steps, and rendering every intermediate one is what makes
        ``env_http`` dominate the step time; the caller (``VLNEnv``'s
        ``fm_trajectory`` branch) executes the whole chunk with ``render=False``
        and renders once at the chunk's real stopping pose.  Collision/stall
        semantics are untouched because they live in ``sim.act``, not in the
        render.  Returns ``{"rgb": None}`` when not rendering so callers that
        only read pose (``agent_tools`` measures it off ``sim`` directly) do not
        trip on a missing key.
        """
        assert not self.episode_over, "episode is over; call reset()"
        aid = action if isinstance(action, int) else int(action.get("action"))
        if aid == STOP:
            self._stopped = True
        else:
            self.sim.act(aid)
            if self.sim.previous_step_collided:
                self.collisions += 1
        self._steps += 1
        self.trail.append(self.sim.xy)
        if not render:
            return {"rgb": None}
        return self.sim.observe()

    @property
    def episode_over(self) -> bool:
        if self._stopped:
            return True
        return self.max_episode_steps != 0 and self._steps >= self.max_episode_steps

    # -- scoring -----------------------------------------------------------
    def distance_to_goal(self) -> float:
        g = self.current_episode.goal_xy
        if not g:
            return float("nan")
        x, y = self.sim.xy
        return float(math.hypot(x - g[0], y - g[1]))

    def get_metrics(self) -> dict:
        """habitat-shaped metrics, scored ABotN's way.

        ``success`` only turns 1 once the agent has called STOP inside the
        arrive threshold -- same rule habitat's R2R measure uses, so the number
        in the console means the same thing across asset sets.  SPL uses the
        annotated reference path as the shortest-path length (POIBench
        additionally subtracts its 2 m arrival slack, as its evaluator does),
        which is what ABotN compares against.
        """
        ep = self.current_episode
        d = self.distance_to_goal()
        walked = self.path_length()
        if ep.goal_xy is None:
            succ = spl = short = float("nan")
        else:
            succ = 1.0 if (self._stopped and d <= ep.arrive_threshold) else 0.0
            short = max(ep.ref_len - ep.spl_slack, 1e-3)
            spl = succ * short / max(walked, short)
        x, y = self.sim.xy
        return {
            "distance_to_goal": d,
            "success": succ,
            "spl": spl,
            # The console never steps STOP into the sim (see vln_model's module
            # docstring), so it re-derives the verdict from the distance; these
            # two are what it needs to do that with ABotN's numbers rather than
            # R2R's 3 m.
            "success_radius": (float("nan") if ep.goal_xy is None
                               else ep.arrive_threshold),
            "shortest_path_m": short,
            "steps": self._steps,
            "collisions": self.collisions,
            "position": [round(x, 3), round(y, 3), round(self.sim.camera_z, 3)],
            "heading_deg": round(math.degrees(self.sim.yaw), 1),
            "path_length_m": round(walked, 2),
            "top_down_map": None,  # use render_top_down()
        }

    def path_length(self) -> float:
        return float(sum(math.hypot(b[0] - a[0], b[1] - a[1])
                         for a, b in zip(self.trail, self.trail[1:])))

    def close(self) -> None:
        try:
            self.client.unload(self.scene)
        except Exception:
            pass

    # -- visualisation -----------------------------------------------------
    def render_top_down(self, size: int = 512, trail: bool = True,
                        show_goal: bool = False) -> np.ndarray:
        from PIL import ImageDraw

        base, to_px, canvas = self._top_down_base(size)
        img = base.copy()
        d = ImageDraw.Draw(img)
        ep = self.current_episode
        # ABotN maps show only what has actually happened.  Its episode
        # waypoints are benchmark ground truth, not a user-visible route; the
        # orange expert polyline is reserved for SAGE's golden-path debugger.
        x, y = self.sim.xy
        cx, cy = to_px(x, y)
        tip = to_px(x + math.cos(self.sim.yaw) * 1.0,
                    y + math.sin(self.sim.yaw) * 1.0)
        draw_navigation_overlay(
            d,
            [to_px(*p) for p in self.trail] if trail else [],
            (cx, cy),
            tip,
            goal=(to_px(*ep.goal_xy) if show_goal and ep.goal_xy else None),
        )
        return np.asarray(img)

    def map_heading_degrees(self, size: int = 512) -> float:
        """Clockwise heading on the rendered map: north/up is zero degrees."""
        _, to_px, _ = self._top_down_base(size)
        x, y = self.sim.xy
        center = to_px(x, y)
        tip = to_px(x + math.cos(self.sim.yaw), y + math.sin(self.sim.yaw))
        return math.degrees(math.atan2(tip[0] - center[0], center[1] - tip[1])) % 360.0

    def _top_down_base(self, size: int):
        from PIL import Image

        cached = self._map_cache.get(self.scene)
        if cached is not None and cached[0] == size:
            return cached[1], cached[2], cached[3]
        small, to_px, canvas = self._occ_for(self.scene).top_down(size)
        img = Image.new("RGB", canvas, CANVAS)
        img.paste(small, ((canvas[0] - small.width) // 2,
                          (canvas[1] - small.height) // 2))
        self._map_cache[self.scene] = (size, img, to_px, canvas)
        self._map_inv[self.scene] = canvas
        return img, to_px, canvas

    def map_to_world(self, fx: float, fy: float):
        """A click on the bird's-eye view -> a habitat-shaped world point.

        ``fx``/``fy`` are fractions of the rendered map.  ``OccMap.top_down``
        centre-pastes a uniformly scaled copy of the occupancy grid on a square
        canvas, so undoing it is that same scale and offset followed by
        ``OccMap.to_world``.  ``set_agent_state`` snaps into free space from
        there, exactly as it does for an annotated start pose inside a shopfront.
        """
        self._top_down_base(512)          # ensure the mapping exists (cached)
        cw, ch = self._map_inv[self.scene]
        occ = self._occ_for(self.scene)
        scale = min(cw / float(occ.w), ch / float(occ.h))
        ox, oy = (cw - int(occ.w * scale)) // 2, (ch - int(occ.h * scale)) // 2
        x, y = occ.to_world((fx * cw - ox) / scale, (fy * ch - oy) / scale)
        return to_habitat(x, y, self.sim.camera_z)

    def teleport(self, position) -> dict:
        """Jump to a world point keeping the heading, and re-observe.

        The trail restarts at the new pose: a teleport is not a walk, and the
        straight line from the old position would read as one.
        """
        self.sim.set_agent_state(position)
        self.trail = [self.sim.xy]
        return self.sim.observe()


# ----------------------------------------------------------------------------
def default_instruction(t: Task) -> str:
    """What goes in the instruction box for this task.

    POIBench ships the real thing ("前往111医药馆"), so it is used verbatim.
    PointBench ships an empty string and a coordinate; writing "walk to
    (0.21, -0.90)" would be inventing an instruction the benchmark does not
    have, so the box states the target plainly and leaves it to you -- the goal
    is scored from the coordinate either way, not from this text.
    """
    if t.instruction:
        return t.instruction
    if t.goal_label:
        return "前往%s" % t.goal_label
    return "目标点 (%.2f, %.2f)，直线距离约 %.1f m%s" % (
        t.goal_xy[0], t.goal_xy[1], t.euclidean,
        "（%s）" % t.distance_bucket if t.distance_bucket else "")


def scene_blurb(ds: ABotNSet, scene: str) -> str:
    """One line about the scene, from its own task list."""
    ts = ds.tasks(scene)
    labels = []
    for t in ts:
        if t.goal_label and t.goal_label not in labels:
            labels.append(t.goal_label)
    head = "%s · %s · %d 个回合" % (ds.label, ds.bare(scene), len(ts))
    if labels:
        head += "；POI：" + "、".join(labels[:8]) + ("…" if len(labels) > 8 else "")
    return head


def _limit(tasks, n: int):
    return tasks if not n or n <= 0 else tasks[:n]


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
