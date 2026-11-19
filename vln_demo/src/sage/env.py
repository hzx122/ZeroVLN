"""A habitat-shaped simulator for SAGE-3D's GVLN benchmark.

Same contract as :mod:`abotn.env` -- ``reset``/``step``/``get_metrics``/
``episode_over`` plus ``sim.agents[0].agent_config.action_space``,
``sim.get_agent_state()`` and ``sim.previous_step_collided`` -- so
``agent_tools.py`` and the ReAct loop run against SAGE unmodified. It is the
ABotN facade with three simplifications the SAGE data allows:

* kinematics default to USD-derived 3-D navmesh support; the camera follows
  executed ground height. ``SAGE_GEOMETRY_MODE=legacy_xy`` is an explicit
  historical reproduction mode, never an automatic fallback.
* **episodes are real**: every GVLN sample carries a start pose, a goal point, a
  dense reference path and an English instruction, so ``distance_to_goal`` /
  ``success`` / ``spl`` are actual numbers, scored the R2R way.
* the render scene id is the *prefixed* console id (``sage-840898``), which is
  also the symlink name :mod:`sage.make_scenes_root` builds, so nothing has to
  strip a prefix.

Two coordinate frames meet here (as in ABotN):

* **SAGE world** -- z up, x/y horizontal, yaw from +x towards +y, camera at
  ``ground_z + CAMERA_HEIGHT``. Everything stored on the sim, sent to the
  renderer and reported in ``get_metrics()["position"]`` is in this frame.
* **habitat-shaped** -- y up, forward -z. Only ``get_agent_state()`` speaks it,
  because ``agent_tools`` reads ``position[0]``/``position[2]`` and a habitat
  quaternion. Map is ``(x, y, z) -> (x, z, -y)`` and ``psi = yaw - pi/2``.
"""

from __future__ import annotations

import math
import os
import json
import threading
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from rs3d.env import (Agent, AgentConfig, AgentState, ActionSpec, Actuation,
                      Instruction, Quaternion, draw_agent_arrow)
from rs3d.gs_client import CameraSpec
from rs3d.nav import quat_to_yaw, yaw_to_quat

from . import occmap as occmod
from .datasets import SageSet, Task
from .gs_client import SAGE_RENDER_URL, XYZEulerClient

STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2, 3


def _load_keep_set(path: str) -> Optional[Set[Tuple[str, str]]]:
    """Read a ``kept_*.jsonl`` allow-list into a ``{(bare_scene, task_id)}`` set.

    Returns ``None`` when ``path`` is empty/missing so callers keep every
    episode. Records are the reachability filter's output
    (``sage.filter_reachable``): each line has ``scene`` (bare id) + ``task_id``.
    """
    if not path or not os.path.isfile(path):
        return None
    keep: Set[Tuple[str, str]] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            keep.add((str(r["scene"]), str(r["task_id"])))
    return keep or None

from vln_gym.observation_profile import requested_profile
_CAMERA_PROFILE = requested_profile()
CAMERA_HEIGHT = (_CAMERA_PROFILE['height_m'] if _CAMERA_PROFILE else
                 float(os.environ.get("SAGE_CAMERA_HEIGHT", "1.2")))
# Official SimpleVLNEnv stores trajectory/agent positions at z=0.5 but renders
# its child camera at world z=1.2.  Treating the trajectory z as eye height put
# our camera near the floor and exaggerated through-wall views at collisions.
CAMERA_HFOV = float(os.environ.get("SAGE_HFOV", "105.3"))
# Official benchmark sets the USD camera focal length to 8 mm. With USD's
# default 20.955 mm horizontal aperture this is about 105.3 degrees, not the
# 120-degree Habitat/Navid sensor setting used by the other demo backends.
# Official SAGE uses 0.10 m. Visual/physical maps can opt into a smaller body
# through the environment; 0.07 m is the useful lower knee on the 5 cm grid
# (smaller values recover no additional cells in the audited scene split).
ROBOT_RADIUS = float(os.environ.get("SAGE_ROBOT_RADIUS", "0.10"))


def to_habitat(x: float, y: float, z: float) -> np.ndarray:
    """SAGE world -> the y-up frame ``agent_tools`` expects."""
    return np.array([x, z, -y], dtype=float)


def from_habitat(p: Sequence[float]):
    return float(p[0]), -float(p[2]), float(p[1])


@dataclass
class Episode:
    """One GVLN task, in habitat-episode clothing (poses stay SAGE-world)."""

    episode_id: str
    scene: str                  # prefixed console id, e.g. "sage-840898"
    scene_id: str               # on-disk InteriorGS dir, for anything file-ish
    instruction: Instruction
    start_position: Optional[np.ndarray] = None   # (x, y, z), z up
    start_rotation: float = 0.0                   # yaw, radians
    goal_xy: Optional[tuple] = None
    ref_len: float = 0.0
    waypoints: tuple = ()          # golden reference path (x, y, z), for the map
    arrive_threshold: float = 1.0
    z_offset: Optional[float] = None
    info: dict = field(default_factory=dict)


class SageSim:
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
        self.want_depth = want_depth or os.environ.get("SAGE_RULE_CREDIT", "0") == "1"
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

        self.navigation = None

    # -- state -------------------------------------------------------------
    @property
    def xy(self):
        return self._x, self._y

    @property
    def yaw(self) -> float:
        return self._yaw

    @property
    def camera_z(self) -> float:
        return self.ground(self._x, self._y) + self.z_offset

    def ground(self, x: float, y: float) -> float:
        if self.navigation is not None:
            return self.navigation.ground_z(x, y)
        return self.occ.ground_z(x, y)

    def get_agent_state(self, agent_id: int = 0) -> AgentState:
        return AgentState(to_habitat(self._x, self._y, self.camera_z),
                          Quaternion(*yaw_to_quat(self._yaw - math.pi / 2.0)))

    def set_agent_state(self, position: Sequence[float],
                        rotation=None, agent_id: int = 0, native: bool = False) -> bool:
        x, y = (float(position[0]), float(position[1])) if native \
            else from_habitat(position)[:2]
        # Public benchmark starts must stay at the released pose. Snapping to
        # this simulator's occupancy mask changes the task before evaluation.
        from . import benchmark_protocol
        if self.navigation is not None:
            if not np.allclose([x,y], self.navigation.position[:2], atol=1e-7, rtol=0):
                raise ValueError("physical SAGE pose changes must execute through navmesh; teleport requires a new resolved reset")
        elif getattr(self.occ, 'preserve_source_xy', False) and not self.occ.is_free(x, y, ROBOT_RADIUS):
            from .navmesh_support import SourceCaseGeometryUnsupported
            raise SourceCaseGeometryUnsupported('original SAGE pose is not supported; automatic XY snapping is forbidden')
        elif not benchmark_protocol.enabled() and not self.occ.is_free(x, y, ROBOT_RADIUS):
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
        """Nearest free pixel within ``limit_m`` that fits the footprint."""
        px, py = self.occ.to_pixel(x, y)
        lim = self.occ.radius_px(limit_m)
        for r in range(1, lim + 1):
            y0, y1 = max(0, py - r), min(self.occ.h, py + r + 1)
            x0, x1 = max(0, px - r), min(self.occ.w, px + r + 1)
            sub = self.occ.free[y0:y1, x0:x1]
            ys, xs = np.nonzero(sub)
            if len(xs) == 0:
                continue
            order = np.argsort((xs + x0 - px) ** 2 + (ys + y0 - py) ** 2)
            for i in order:
                cx, cy = int(xs[i]) + x0, int(ys[i]) + y0
                wx, wy = self.occ.to_world(float(cx), float(cy))
                if self.occ.is_free(wx, wy, ROBOT_RADIUS):
                    return wx, wy
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
            if self.navigation is not None:
                moved = self.navigation.step((tx,ty), slide=self.allow_sliding)
                nx,ny = moved[:2]
                tolerance = max(1e-5, min(.005, abs(amount)*.05))
                hit = math.hypot(nx-tx,ny-ty) > tolerance
            else:
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
        with self._lock:
            rgb, depth = self.client.render_one(
                self.scene, (self._x, self._y, self.camera_z), self._yaw,
                want_depth=self.want_depth,
                pitch=getattr(self, 'camera_pitch', 0.0))
        obs = {"rgb": np.asarray(rgb, dtype=np.uint8)}
        if depth is not None:
            obs["depth"] = depth
        return obs


class SageEnv:
    """``habitat.Env``-shaped wrapper over one :class:`~sage.datasets.SageSet`."""

    def __init__(self, dataset: SageSet, scenes=None, width: int = 640,
                 height: int = 480, hfov: Optional[float] = None,
                 forward_step: float = 0.25,
                 turn_angle: float = 30.0, allow_sliding: bool = True,
                 want_depth: bool = False, seed: int = 0,
                 max_episode_steps: int = 0, episodes_per_scene: int = 0,
                 render_url: str = SAGE_RENDER_URL,
                 client: Optional[XYZEulerClient] = None,
                 block_furniture: Optional[bool] = None,
                 collision_mode: Optional[str] = None,
                 furniture_mode: Optional[str] = None,
                 unload_on_close: Optional[bool] = None):
        self.ds = dataset
        self.unload_on_close = (os.environ.get("SAGE_UNLOAD_ON_CLOSE", "1") != "0"
            if unload_on_close is None else bool(unload_on_close))
        self.seed = int(seed)
        self.robot_radius = ROBOT_RADIUS
        self.max_episode_steps = int(max_episode_steps)
        # Furniture collision is independent of the floor/wall model. ``geom``
        # uses a mask projected from the RGB renderer's PLY; the old boolean is
        # retained only as an alias for the legacy labels/AABB implementation.
        if block_furniture is None:
            block_furniture = os.environ.get("SAGE_BLOCK_FURNITURE", "0") == "1"
        if furniture_mode is None:
            furniture_mode = os.environ.get("SAGE_FURNITURE_COLLISION", "").strip().lower()
        if not furniture_mode:
            furniture_mode = "labels" if block_furniture else "none"
        if furniture_mode not in occmod.FURNITURE_MODES:
            furniture_mode = "none"
        self.furniture_mode = furniture_mode
        self.block_furniture = furniture_mode != "none"
        # Collision model (official/hybrid/visual/self): the one switch that decides
        # whether walkability comes from the official occupancy.png or our
        # self-built structure.json floor plan. Read once here so BOTH the
        # console (build_sage) and the bench/training server (vln_gym) -- both
        # construct SageEnv -- honour SAGE_COLLISION identically.
        if collision_mode is None:
            collision_mode = os.environ.get("SAGE_COLLISION", "official").strip().lower()
        self.collision_mode = collision_mode
        want = [s for s in (scenes or ()) if s] or dataset.list_scenes()
        self.scenes = [s if s in dataset.list_scenes() else dataset.qualify(s)
                       for s in want]
        if hfov is None:
            hfov = CAMERA_HFOV
        self.client = client or XYZEulerClient(
            url=render_url, camera=CameraSpec(width, height, float(hfov)))
        self._occ_cache: Dict[str, occmod.OccMap] = {}
        self._map_cache: Dict[str, tuple] = {}
        self._map_inv: Dict[str, tuple] = {}
        self.show_semantic_region = os.environ.get(
            "SAGE_SHOW_SEMANTIC_REGION", "0") == "1"
        self.semantic_region_max_m = max(0.05, float(os.environ.get(
            "SAGE_SEMANTIC_REGION_MAX_M", "1.20") or "1.20"))
        self._semantic_region_cache: Dict[tuple, dict] = {}
        self._label_index_cache: Dict[str, dict] = {}

        # Optional episode allow-list: SAGE_EPISODE_KEEP points at a
        # ``kept_<split>.jsonl`` (records with ``scene`` bare-id + ``task_id``)
        # so the console/bench can show ONLY the physically-trainable use cases
        # the reachability filter kept. Empty/unset -> every episode as before.
        keep = _load_keep_set(os.environ.get("SAGE_EPISODE_KEEP", "").strip())
        self._keep = keep
        # SAGE_GOLDEN_ASTAR=1 replaces the official (furniture-clipping) golden
        # shown on the map with an A* path re-planned on the usd_rooms free grid
        # (physically executable). Lazy + cached per episode.
        self.golden_astar = os.environ.get("SAGE_GOLDEN_ASTAR", "0") == "1"
        self._astar_cache: Dict[str, tuple] = {}

        self.episodes: List[Episode] = [
            self._make_episode(t)
            for s in self.scenes
            for t in dataset.tasks(s, limit=(0 if keep else episodes_per_scene))
            if keep is None or (dataset.bare(s), t.task_id) in keep]
        if not self.episodes:
            raise RuntimeError("%s 里没有可用回合" % dataset.split)
        self.current_episode = self.episodes[0]

        self.scene = self.current_episode.scene
        self.sim = SageSim(self.scene, self._occ_for(self.scene), self.client,
                           CAMERA_HEIGHT, forward_step, turn_angle,
                           allow_sliding, want_depth)
        self._stopped = False
        self._steps = 0
        self.trail: List[tuple] = []
        self.collisions = 0

    # -- scenes ------------------------------------------------------------
    def _occ_for(self, scene: str) -> occmod.OccMap:
        if scene not in self._occ_cache:
            self._occ_cache[scene] = occmod.OccMap(
                self.ds.scene_dir(scene), block_furniture=self.block_furniture,
                collision_mode=self.collision_mode,
                furniture_mode=self.furniture_mode)
        return self._occ_cache[scene]

    def _ground(self, scene: str, x: float, y: float) -> float:
        return self._occ_for(scene).ground_z(x, y)

    def _use_scene(self, scene: str) -> None:
        if scene == self.scene:
            return
        self.scene = scene
        self.sim.scene = scene
        self.sim.occ = self._occ_for(scene)
        self.sim.z_offset = CAMERA_HEIGHT
        self.client.preload(scene)

    # -- episodes ----------------------------------------------------------
    def _make_episode(self, t: Task) -> Episode:
        return Episode(
            episode_id=t.task_id,
            scene=t.scene,
            scene_id=self.ds.scene_dir(t.scene),
            instruction=Instruction(default_instruction(t)),
            start_position=np.array([t.start["x"], t.start["y"], t.start["z"]]),
            start_rotation=float(t.start["yaw"]),
            goal_xy=t.goal_xy,
            ref_len=float(t.ref_len),
            waypoints=t.waypoints,
            arrive_threshold=self.ds.arrive_threshold,
            z_offset=CAMERA_HEIGHT,
            info={"scene_description": scene_blurb(self.ds, t.scene),
                  "instruction_type": t.instruction_type,
                  "goal_label": t.goal_label,
                  "start_label": t.start_label})

    def episode_for(self, scene: str, task_id: str) -> Optional[Episode]:
        """按 task_id 取回合；自载 episodes 里没有时全量解析定位并 append.

        供 console 的 switch 对 ``--sage_episodes_per_scene`` 截断之外的用例
        （如当前轨迹的其他指令）延迟物化。与 :meth:`traj_cases` 同口径
        honours ``SAGE_EPISODE_KEEP``。
        """
        tid = str(task_id)
        for e in self.episodes:
            if e.scene == scene and str(e.episode_id) == tid:
                return e
        for t in self.ds.tasks(scene, 0):
            if t.task_id == tid and (
                    self._keep is None
                    or (self.ds.bare(scene), t.task_id) in self._keep):
                ep = self._make_episode(t)
                self.episodes.append(ep)
                return ep
        return None

    def traj_cases(self, scene: str, traj_id,
                   dimension: Optional[str] = None) -> List[dict]:
        """该轨迹当前能力维度的用例；不传dimension时兼容返回全部。"""
        out = []
        for t in self.ds.traj_tasks(scene, traj_id):
            if self._keep is not None and (
                    self.ds.bare(scene), t.task_id) not in self._keep:
                continue
            task_dimension = str(t.instruction_type).split("/", 1)[-1].lower()
            if dimension and task_dimension != str(dimension).lower():
                continue
            out.append({"task_id": t.task_id, "scene": t.scene,
                        "instruction": t.instruction,
                        "instruction_type": t.instruction_type,
                        "start_label": t.start_label,
                        "goal_label": t.goal_label,
                        "ref_len": t.ref_len})
        return out

    def new_episode(self, scene: Optional[str] = None) -> Episode:
        """A sampled free-roam spawn in ``scene`` (no goal, hence no score)."""
        scene = scene or self.scene
        occ = self._occ_for(scene)
        rng = np.random.default_rng([self.seed, zlib.crc32(scene.encode()),
                                     len(self.episodes)])
        from .physical_navigation import enabled
        if enabled():
            from .navmesh_support import from_nav, digest
            nav = self._navigation_for(scene)
            nav.pathfinder.seed(int(rng.integers(0, 2**31)))
            ground = from_nav(nav.pathfinder.get_random_navigable_point())
            if not np.isfinite(ground).all():
                raise RuntimeError('scene has no navigable 3-D free-roam spawn')
            p = ground[:2]
            evidence = {'free_roam_ground_xyz': ground.tolist(),
                        'navmesh_manifest_sha256': digest(nav.manifest_path)}
        else:
            p = occ.sample_free(rng, ROBOT_RADIUS)
            evidence = {}
        if p is None:
            raise RuntimeError("场景 %s 没有可站立的自由空间" % scene)
        ep = Episode(
            episode_id="free%d" % sum(1 for e in self.episodes
                                      if e.scene == scene and e.goal_xy is None),
            scene=scene,
            scene_id=self.ds.scene_dir(scene),
            instruction=Instruction(""),
            start_position=np.array([p[0], p[1],
                                     (ground[2] if enabled() else self._ground(scene, *p)) + CAMERA_HEIGHT]),
            start_rotation=float(rng.random() * 2 * math.pi),
            z_offset=CAMERA_HEIGHT,
            info={"scene_description": scene_blurb(self.ds, scene), **evidence})
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
    def _navigation_for(self, scene):
        from .physical_navigation import load_navigation
        if not hasattr(self, '_navigation_cache'):
            self._navigation_cache = {}
        if scene not in self._navigation_cache:
            self._navigation_cache[scene] = load_navigation(scene, self.robot_radius)
        return self._navigation_cache[scene]

    def reset(self) -> dict:
        ep = self.current_episode
        from .physical_navigation import enabled, resolve_task
        navigation, physical_target, floor_audit = None, None, None
        if enabled():
            from .goal_region import load_label_index
            nav = self._navigation_for(ep.scene)
            if ep.goal_xy is None:
                from .navmesh_support import digest, to_nav, FloorUnresolved
                ground = np.asarray((ep.info or {}).get('free_roam_ground_xyz', []))
                if (ground.shape != (3,) or not np.isfinite(ground).all()
                        or not np.allclose(ground[:2], ep.start_position[:2], atol=1e-7, rtol=0)
                        or ep.info.get('navmesh_manifest_sha256') != digest(nav.manifest_path)
                        or not nav.pathfinder.is_navigable(to_nav(ground), .05)):
                    raise FloorUnresolved('free-roam reset lacks a verified 3-D spawn')
                nav.position = ground.copy()
                floor_audit = {'reason': 'native_navmesh_free_roam', **ep.info}
            else:
                task = next(t for t in self.ds.tasks(ep.scene, 0) if t.task_id == ep.episode_id)
                from .source_objects import load_source_label_index
                labels = load_source_label_index(self.ds.scene_dir(ep.scene))
                floor_audit, physical_target, _ = resolve_task(nav, task, labels)
            navigation = nav
        # Reject unsupported original geometry before preloading a GPU scene
        # or replacing the previous simulator state. Keep the physical checks
        # and actual reset pose unchanged for admissible cases.
        self._use_scene(ep.scene)
        self.sim.navigation = navigation
        self._physical_target = physical_target
        if floor_audit is not None:
            self._floor_audit = floor_audit
        if ep.z_offset is not None:
            self.sim.z_offset = float(ep.z_offset)
        self.sim.set_agent_state(ep.start_position, ep.start_rotation, native=True)
        self._stopped = False
        self._steps = 0
        self.collisions = 0
        self.trail = [self.sim.xy]
        self.trail_xyz = [(*self.sim.xy, self.sim.ground(*self.sim.xy))]
        self._physical_shortest_path = (self._physical_target.distance(self.trail_xyz[0])
                                       if self._physical_target is not None else None)
        if self._physical_target is not None and self._physical_shortest_path is None:
            from .navmesh_support import SourceCaseGeometryUnsupported
            raise SourceCaseGeometryUnsupported('designated target has no connected 3-D route from the original start')
        return self.sim.observe()

    def step(self, action, render: bool = True, **kwargs) -> dict:
        """Advance physics; like ABotNEnv, optionally skip the sensor HTTP call."""
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
        self.trail_xyz.append((*self.sim.xy, self.sim.ground(*self.sim.xy)))
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
        if getattr(self, '_physical_target', None) is not None:
            distance = self._physical_target.distance((*self.sim.xy, self.sim.ground(*self.sim.xy)))
            return distance if distance is not None else float('inf')
        g = self.current_episode.goal_xy
        if not g:
            return float("nan")
        x, y = self.sim.xy
        return float(math.hypot(x - g[0], y - g[1]))

    def get_metrics(self) -> dict:
        ep = self.current_episode
        d = self.distance_to_goal()
        walked = self.path_length()
        if ep.goal_xy is None:
            succ = spl = short = float("nan")
        else:
            succ = 1.0 if (self._stopped and d <= ep.arrive_threshold) else 0.0
            short = max(ep.ref_len, 1e-3)
            spl = succ * short / max(walked, short)
        x, y = self.sim.xy
        official = self._official_object_arrival(ep, (x, y))
        physical = getattr(self, '_physical_target', None) is not None
        if physical:
            succ = float(self._stopped and official['success'])
            optimum = self._physical_shortest_path
            short = optimum
            spl = succ * optimum/max(optimum, walked, 1e-9) if optimum else succ if optimum == 0 else 0.
        return {
            **({'geometry_schema': 'sage-executed-navmesh-v2',
                'ground_z_m': self.sim.ground(x,y),
                'navmesh_manifest_sha256': official['navmesh_manifest_sha256'],
                'sage_floor_audit': self._floor_audit,
                'designated_target_xy_success': official['designated_target_xy_success'],
                'physical_success': bool(official['success'])} if physical else {}),
            "distance_to_goal": d,
            "success": succ,
            "spl": spl,
            "success_radius": (float("nan") if ep.goal_xy is None
                               else ep.arrive_threshold),
            "shortest_path_m": short,
            "steps": self._steps,
            "collisions": self.collisions,
            "position": [round(x, 3), round(y, 3), round(self.sim.camera_z, 3)],
            "heading_deg": round(math.degrees(self.sim.yaw), 1),
            **({"oracle_world_pose": [float(x),float(y),float(math.degrees(self.sim.yaw))]}
               if os.environ.get("SAGE_PRIVILEGED_STATE", "0") == "1" else {}),
            "official_object_success": bool(official.get("success", False)),
            "official_object_success_detail": official,
            "path_length_m": round(walked, 2),
            "top_down_map": None,
        }

    def _official_object_arrival(self, ep, xy) -> dict:
        """Target-specific official bbox success, evaluated without a stop gate."""
        if getattr(self, '_physical_target', None) is not None:
            return self._physical_target.score((*xy, self.sim.ground(*xy)))
        if ep.goal_xy is None:
            return {'success': False, 'available': False, 'reason': 'free_roam_has_no_target'}
        from . import benchmark_protocol
        if benchmark_protocol.enabled():
            return benchmark_protocol.score(ep.scene, ep.episode_id, xy)
        target_id = str((ep.info or {}).get("goal_label") or "").strip()
        if not target_id:
            return {"schema": "sage-official-expanded-object-bbox-v1",
                    "success": False, "available": False,
                    "reason": "missing_target_object_id"}
        try:
            from .goal_region import load_label_index, official_object_arrival
            labels = self._label_index_cache.get(ep.scene)
            if labels is None:
                from .source_objects import load_source_label_index
                labels = load_source_label_index(self.ds.scene_dir(ep.scene))
                self._label_index_cache[ep.scene] = labels
            info = labels[target_id]
            result = official_object_arrival(info, xy)
            result.update({"available": True, "target_object_id": target_id})
            return result
        except Exception as exc:
            return {"schema": "sage-official-expanded-object-bbox-v1",
                    "success": False, "available": False,
                    "target_object_id": target_id, "reason": str(exc)}

    def path_length(self) -> float:
        if getattr(self.sim, 'navigation', None) is not None:
            return float(sum(math.dist(a, b) for a, b in zip(self.trail_xyz, self.trail_xyz[1:])))
        return float(sum(math.hypot(b[0] - a[0], b[1] - a[1])
                         for a, b in zip(self.trail, self.trail[1:])))

    def close(self) -> None:
        # The shared renderer owns residency across private sessions.
        if self.unload_on_close:
            try:
                self.client.unload(self.scene)
            except Exception:
                pass

    # -- visualisation -----------------------------------------------------
    def _golden_pts(self, ep):
        """Waypoints to draw as golden: official ``ep.waypoints`` by default, or
        an A* replan on the usd_rooms free grid when ``SAGE_GOLDEN_ASTAR``."""
        if not self.golden_astar or not ep.goal_xy or ep.start_position is None:
            return getattr(ep, "waypoints", ())
        key = ep.episode_id
        if key not in self._astar_cache:
            from . import planner
            occ = self._occ_for(ep.scene)
            path = planner.plan_path(
                occ, (float(ep.start_position[0]), float(ep.start_position[1])),
                ep.goal_xy, radius_m=self.robot_radius)
            # fall back to the official golden if A* cannot connect
            self._astar_cache[key] = tuple(path) if path else getattr(
                ep, "waypoints", ())
        return self._astar_cache[key]

    def render_top_down(self, size: int = 512, trail: bool = True,
                        show_goal: bool = False) -> np.ndarray:
        from PIL import Image, ImageDraw

        base, to_px, canvas = self._top_down_base(size)
        img = base.copy()
        ep = self.current_episode
        # Use the exact evaluator geometry, without extra free-space/LOS gates.
        # Tool/model maps must not reveal the annotated target or success area.
        region = self._official_object_arrival(ep, self.sim.xy) if show_goal else {}
        if region.get('schema') == 'sage-designated-target-xy-v2' and region.get('available'):
            from .benchmark_protocol import designated_bbox
            # The public scorer returns identity/success, not the legacy display
            # fields. Derive both overlays from its exact designated target.
            region = {**region,
                'target_bbox_xyxy': designated_bbox(ep.scene, ep.episode_id)[1],
                'expanded_bbox_xyxy': designated_bbox(ep.scene, ep.episode_id,
                    region['expansion_radius_m'])[1]}
        if region.get("available"):
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            for key, fill, outline in (
                    ("expanded_bbox_xyxy", (40, 220, 120, 65), (20, 180, 80, 230)),
                    ("target_bbox_xyxy", (40, 220, 120, 95), (0, 110, 45, 255))):
                x0, y0, x1, y1 = region[key]
                a, b = to_px(x0, y0), to_px(x1, y1)
                od.rectangle([min(a[0], b[0]), min(a[1], b[1]),
                              max(a[0], b[0]), max(a[1], b[1])],
                             fill=fill, outline=outline, width=2)
            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        d = ImageDraw.Draw(img)
        # Expert routes are human display overlays, never agent observations.
        if show_goal:
            wps = self._golden_pts(ep)
            if wps and len(wps) > 1:
                pts = [to_px(w[0], w[1]) for w in wps]
                d.line(pts, fill=(255, 165, 40), width=3)
                sx, sy = pts[0]
                d.ellipse([sx-5, sy-5, sx+5, sy+5], fill=(255, 165, 40))
        if show_goal and ep.goal_xy:
            gx, gy = to_px(*ep.goal_xy)
            r = 7
            d.ellipse([gx - r, gy - r, gx + r, gy + r], outline=(255, 165, 40), width=2)
        if trail and len(self.trail) > 1:
            d.line([to_px(*p) for p in self.trail], fill=(90, 190, 255), width=2)
        # Point-Goal 前瞻 waypoint（走专家轨迹时 run_golden 设进 self._pointgoal_wp）：
        # 青色实心+黑边，画在 trail 之上、agent 之下，标 ~look_m 前方的 safe+LOS 点。
        wp = getattr(self, "_pointgoal_wp", None)
        if show_goal and wp is not None:
            wx, wy = to_px(wp[0], wp[1])
            d.ellipse([wx - 7, wy - 7, wx + 7, wy + 7], fill=(0, 220, 220),
                      outline=(0, 0, 0), width=2)
        x, y = self.sim.xy
        cx, cy = to_px(x, y)
        tip = to_px(x + math.cos(self.sim.yaw) * 1.0,
                    y + math.sin(self.sim.yaw) * 1.0)
        d.line([(cx, cy), tip], fill=(255, 220, 0), width=3)
        draw_agent_arrow(d, (cx, cy), tip)
        return np.asarray(img)

    def _semantic_region_for(self, ep) -> dict:
        if not self.show_semantic_region or not ep.goal_xy:
            return {"available": False, "cells_xy": []}
        target_id = str((ep.info or {}).get("goal_label") or "").strip()
        if not target_id:
            return {"available": False, "cells_xy": []}
        key = (ep.scene, target_id, round(self.semantic_region_max_m, 3),
               round(self.robot_radius, 3))
        cached = self._semantic_region_cache.get(key)
        if cached is not None:
            return cached
        try:
            from .goal_region import (load_label_index,
                                      object_side_region_cells)
            from .source_objects import load_source_label_index
            labels = load_source_label_index(self.ds.scene_dir(ep.scene))
            info = labels[target_id]
            start_xy = (float(ep.start_position[0]), float(ep.start_position[1]))
            result = object_side_region_cells(
                self._occ_for(ep.scene), info, start_xy, ep.goal_xy,
                {"robot_radius_m": self.robot_radius,
                 "max_surface_distance_m": self.semantic_region_max_m})
        except Exception as exc:
            result = {"available": False, "reason": str(exc), "cells_xy": []}
        self._semantic_region_cache[key] = result
        return result

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
        img = Image.new("RGB", canvas, (18, 20, 24))
        img.paste(small, ((canvas[0] - small.width) // 2,
                          (canvas[1] - small.height) // 2))
        self._map_cache[self.scene] = (size, img, to_px, canvas)
        self._map_inv[self.scene] = canvas
        return img, to_px, canvas

    def map_to_world(self, fx: float, fy: float):
        self._top_down_base(512)
        cw, ch = self._map_inv[self.scene]
        occ = self._occ_for(self.scene)
        scale = min(cw / float(occ.w), ch / float(occ.h))
        ox, oy = (cw - int(occ.w * scale)) // 2, (ch - int(occ.h * scale)) // 2
        x, y = occ.to_world((fx * cw - ox) / scale, (fy * ch - oy) / scale)
        return to_habitat(x, y, self.sim.camera_z)

    def teleport(self, position) -> dict:
        if self.sim.navigation is not None:
            raise ValueError('physical SAGE teleport is unsupported; reset a designated episode')
        self.sim.set_agent_state(position)
        self.trail = [self.sim.xy]
        self.trail_xyz = [(*self.sim.xy, self.sim.ground(*self.sim.xy))]
        return self.sim.observe()


# ----------------------------------------------------------------------------
def default_instruction(t: Task) -> str:
    """GVLN always ships a real English instruction; use it verbatim."""
    if t.instruction:
        return t.instruction
    if t.goal_label:
        return "Navigate to the %s." % t.goal_label
    return "Navigate to (%.2f, %.2f)." % (t.goal_xy[0], t.goal_xy[1])


def scene_blurb(ds: SageSet, scene: str) -> str:
    ts = ds.tasks(scene)
    return "%s · %s · %d 个回合" % (ds.label, ds.bare(scene), len(ts))


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
