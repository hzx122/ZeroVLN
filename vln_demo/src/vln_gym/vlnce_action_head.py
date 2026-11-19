"""Strict R2R/RxR Habitat adapter for the existing action expert HTTP runtime.

The shared VLNEnv owns movement and native scoring. This adapter supplies the
fixed forward/left coordinate convention required by its FM executor, fresh
terminal camera frames, and exact episode selection for benchmark execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from .env import VLNEnv


NATIVE_SUCCESS_CONTRACT = "habitat_vlnce_native_stop_3m_v1"


def navmesh_override_identity(root):
    """Validate a versioned candidate without changing shared scene assets."""
    root = Path(root).expanduser()
    if not root.is_absolute():
        raise ValueError('navmesh override must use an explicit absolute directory')
    root = root.resolve(strict=True)
    manifest = root / 'manifest.json'
    raw = manifest.read_bytes()
    record = json.loads(raw)
    if record.get('schema') != 'vlnce_standard_navmesh_rebake_v1':
        raise ValueError('unsupported navmesh override manifest')
    assets = {}
    for row in record['reports']:
        scene = row['scene']
        if Path(scene).name != scene or scene in assets:
            raise ValueError('invalid or duplicate override scene')
        path = root / scene / (scene + '.navmesh')
        if (not path.resolve().is_relative_to(root)
                or hashlib.sha256(path.read_bytes()).hexdigest() != row['navmesh_sha256']):
            raise ValueError('navmesh override hash mismatch: ' + scene)
        assets[scene] = {k: row[k] for k in ('navmesh_sha256', 'mesh_sha256', 'settings')}
    return dict(root=str(root), manifest_sha256=hashlib.sha256(raw).hexdigest(), assets=assets)


def install_navmesh_override(sim, root):
    """Load before task reset computes DistanceToGoal and SPL's denominator."""
    identity = navmesh_override_identity(root)
    previous = getattr(sim, '_vlnce_navmesh_override', None)
    if previous is not None:
        if previous != identity:
            raise ValueError('cannot change navmesh version on a live simulator')
        return
    original = sim.reconfigure

    def reconfigure(*args, **kwargs):
        result = original(*args, **kwargs)
        mesh = Path(sim.sim_config.sim_cfg.scene_id)
        scene = mesh.stem
        asset = identity['assets'].get(scene)
        if asset is None:
            sim._vlnce_navmesh_active = None
            return result
        active = dict(scene=scene, manifest_sha256=identity['manifest_sha256'], **asset)
        if getattr(sim, '_vlnce_navmesh_active', None) != active:
            if hashlib.sha256(mesh.read_bytes()).hexdigest() != asset['mesh_sha256']:
                raise ValueError('scene mesh differs from navmesh build input: ' + scene)
            agent = sim.sim_config.agents[0]
            if (not math.isclose(agent.radius, asset['settings']['agent_radius'], abs_tol=1e-6)
                    or not math.isclose(agent.height, asset['settings']['agent_height'], abs_tol=1e-6)):
                raise ValueError('agent dimensions differ from navmesh build input')
            # Verify again at load, rather than silently accepting a changed file.
            path = Path(identity['root']) / scene / (scene + '.navmesh')
            if hashlib.sha256(path.read_bytes()).hexdigest() != asset['navmesh_sha256']:
                raise ValueError('navmesh changed after manifest validation')
            if not sim.pathfinder.load_nav_mesh(str(path)):
                raise RuntimeError('cannot load versioned navmesh: ' + str(path))
            sim._vlnce_navmesh_active = active
        return result

    sim.reconfigure = reconfigure
    sim._vlnce_navmesh_override = identity


def episode_scene(episode):
    return getattr(episode, "scene", None) or episode.scene_id.split("/")[-2]


def canonical_scene(scene, backend):
    value = str(scene)
    return value.removeprefix("rxr-") if backend == "rxr" else value


def install_rotation_only_controls(sim):
    """Keep native yaw actions in place; retain filtering for translation.

    Habitat applies its navmesh move filter to every body action, including
    turns. At a nonzero navmesh height offset a zero-length move snaps the
    agent vertically. A yaw-only tool must not manufacture position reward.
    Bind only this simulator's controls; leave registry and other gyms alone.
    """
    controls = sim.get_agent(0).controls
    if getattr(controls, '_vlnce_rotation_only', False):
        return
    original_action = controls.action

    def action(obj, action_name, actuation_spec, apply_filter=True):
        if action_name not in ('turn_left', 'turn_right'):
            return original_action(obj, action_name, actuation_spec,
                                   apply_filter=apply_filter)
        before = np.array(obj.absolute_translation, copy=True)
        collided = original_action(obj, action_name, actuation_spec,
                                   apply_filter=False)
        if not np.array_equal(before, np.asarray(obj.absolute_translation)):
            raise RuntimeError('native yaw control translated agent without move filter')
        return collided

    controls.action = action
    controls._vlnce_rotation_only = True


class VLNCEActionHeadEnv(VLNEnv):
    """Keep Habitat simulation/scoring while adapting actual planar odometry."""

    def __init__(self, *args, backend="r2r", **kwargs):
        if backend not in ("r2r", "rxr"):
            raise ValueError("Action Head VLN-CE adapter supports only r2r/rxr")
        if os.environ.get("VLN_HABITAT_RENDER_URL"):
            from remote_habitat.client import install
            install()
        if backend == "rxr":
            # rxr.py caches these at import time. Explicitly update the loader's
            # settings even if another module imported it before this adapter.
            os.environ["RXR_EPISODES_PER_SCENE"] = "0"
            os.environ["RXR_LANGUAGES"] = "en-US,en-IN"
            os.environ["RXR_ROLES"] = "guide"
            import rxr
            rxr.EPISODES_PER_SCENE = 0
            rxr.LANGUAGES = "en-US,en-IN"
            rxr.ROLES = "guide"
        super().__init__(*args, backend=backend, **kwargs)

    def _select_episode(self, scene, ep_id):
        wanted_scene = canonical_scene(scene, self.backend)
        matches = [episode for episode in self.env.episodes
                   if canonical_scene(episode_scene(episode), self.backend) == wanted_scene
                   and str(episode.episode_id) == str(ep_id)]
        if len(matches) != 1:
            raise ValueError(f"expected one {self.backend} episode for "
                             f"scene={scene!r} id={ep_id!r}, found {len(matches)}")
        return matches[0]

    def reset(self, *, scene=None, episode_id=None, instruction=None,
              seed=None, options=None):
        scene = self.scene if scene is None else scene
        episode_id = self.episode_id if episode_id is None else str(episode_id)
        expected = self._select_episode(scene, episode_id)
        expected_instruction = expected.instruction.instruction_text
        if instruction is not None and instruction != expected_instruction:
            raise ValueError(f"manifest instruction mismatch for {scene}/{episode_id}")
        # A Habitat recovery rebuild uses the underlying R2R evaluator. Retain
        # RxR's full episode catalog when the existing recovery branch runs.
        previous_env = self.env
        episodes = self.env.episodes
        navmesh_root = os.environ.get('VLNCE_NAVMESH_ROOT')
        if navmesh_root:
            install_navmesh_override(self.env.sim, navmesh_root)
        if seed is not None:
            self.env.seed(int(seed))
        self._native_rl_credit = None
        observation, info = super().reset(
            scene=scene, episode_id=episode_id, instruction=expected_instruction,
            seed=seed, options=options)
        if self.backend == "rxr" and self.env is not previous_env:
            from rxr import SharedSimEnv
            self.env = SharedSimEnv(self.env, episodes)
        if navmesh_root and getattr(self.env.sim, '_vlnce_navmesh_override', None) is None:
            # The shared recovery branch may have rebuilt the simulator. Redo
            # this reset before returning metrics, never score on its old mesh.
            install_navmesh_override(self.env.sim, navmesh_root)
            observation, info = super().reset(
                scene=scene, episode_id=episode_id, instruction=expected_instruction,
                seed=seed, options=options)
        actual = self.env.current_episode
        if (canonical_scene(episode_scene(actual), self.backend)
                != canonical_scene(scene, self.backend)
                or str(actual.episode_id) != str(episode_id)
                or actual.instruction.instruction_text != expected_instruction):
            raise RuntimeError(f"Habitat reset substituted episode {scene}/{episode_id}")
        install_rotation_only_controls(self.env.sim)
        info["actual_instruction"] = actual.instruction.instruction_text
        info["backend"] = self.backend
        return observation, info

    def _world_pose_rad(self):
        # Habitat: +Y up, identity rotation faces -Z, left points -X.
        # This is a fixed, right-handed planar frame: forward = +x, left = +y,
        # positive yaw = left. Episodic GPS/compass have a different convention
        # and compass is radians, so neither is used for FM execution.
        position = self.env.sim.get_agent_state().position
        yaw_deg = -self.A._yaw(self.env)
        return -float(position[2]), -float(position[0]), math.radians(yaw_deg)

    def _private_waypoint_geometry(self):
        """Evaluator-only measured height; never part of the FM action label."""
        return [float(v) for v in self.env.sim.get_agent_state().position]

    def _render_current(self):
        # get_observations_at() uses Habitat's sensor suite, preserving metric
        # depth and sensor processing. It takes no action and returns a frame
        # at the actual final pose, including after collision or wall sliding.
        sim = self.env.sim
        depth = getattr(sim, "_remote_defer_depth", 0)
        try:
            if getattr(sim, "_remote_render_installed", False):
                sim._remote_defer_depth = 0
            observation = sim.get_observations_at()
        finally:
            if getattr(sim, "_remote_render_installed", False):
                sim._remote_defer_depth = depth
        if observation is None or "rgb" not in observation or "depth" not in observation:
            raise RuntimeError("Habitat final-pose RGB/depth unavailable")
        return dict(observation)

    def step(self, action):
        mode, payload = self._classify(action)
        sim = self.env.sim
        defer = (mode == "fm_trajectory" and payload.get("render_final_only", False)
                 and getattr(sim, "_remote_render_installed", False))
        depth = getattr(sim, "_remote_defer_depth", 0)
        try:
            if defer:
                # Nest the existing primitive terminal-frame decorators under
                # one FM prefix. _render_current supplies the sole final frame.
                sim._remote_defer_depth = depth + 1
            return super().step(action)
        finally:
            if defer:
                sim._remote_defer_depth = depth

    def _observe(self, obs):
        value = super()._observe(obs)
        if value:
            x, y, yaw = self._world_pose_rad()
            value.update(gps=[x, y], yaw_deg=math.degrees(yaw))
        return value

    def _metrics(self, obs, extra=None):
        value = super()._metrics(obs, extra=extra)
        x, y, yaw = self._world_pose_rad()
        value.update(world_xy=[x, y], world_yaw_deg=math.degrees(yaw),
                     gps=[x, y], yaw_deg=math.degrees(yaw),
                     success_radius=3.0, backend=self.backend,
                     success_contract=NATIVE_SUCCESS_CONTRACT)
        # Evaluator-only geometry, retained in RemoteRenderClient.last_info.
        # Planar odometry alone cannot distinguish overlapping MP3D floors or
        # certify a counterfactual RGB/depth render from the same camera.
        sim = self.env.sim
        state = sim.get_agent_state()
        geometry = {
            "schema": "habitat_actual_agent_geometry_v1",
            "position": [float(v) for v in state.position],
            "rotation_xyzw": [float(getattr(state.rotation, k)) for k in ("x", "y", "z", "w")],
        }
        specs = getattr(sim, "_remote_specs", None)
        if specs is None and getattr(sim, "sim_config", None) is not None:
            specs = [dict(uuid=s.uuid, type=s.sensor_type.name,
                          resolution=[int(v) for v in s.resolution], hfov=float(s.hfov),
                          position=[float(v) for v in s.position],
                          orientation=[float(v) for v in s.orientation])
                     for s in sim.sim_config.agents[0].sensor_specifications]
        if specs is not None:
            geometry["sensors"] = [dict(s) for s in specs if s["uuid"] in ("rgb", "depth")]
        value["oracle_habitat_geometry"] = geometry
        active_navmesh = getattr(sim, '_vlnce_navmesh_active', None)
        if active_navmesh is not None:
            value['navmesh_identity'] = active_navmesh
        measures = self.env.task.measurements.measures
        spl = measures["spl"]
        # Use the same accumulated 3D distance and starting geodesic distance
        # as Habitat's native SPL measure; do not replace either with planar
        # displacement or a sum of requested commands.
        value["path_length_m"] = float(spl._agent_episode_distance)
        value["shortest_path_m"] = float(spl._start_end_episode_distance)
        native = self.env.get_metrics()
        for key in ("ndtw", "sdtw", "oracle_spl"):
            if key in native:
                value[key] = native[key]
        if isinstance(native.get("collisions"), dict):
            value["collisions"] = int(native["collisions"].get("count", 0))
        if obs is not None and "rgb" in obs and "depth" in obs:
            rgb = np.asarray(obs["rgb"])
            depth = np.asarray(obs["depth"])
            finite = np.isfinite(depth)
            finite_depth = depth[finite]
            value["sensor_audit"] = {
                "schema": "habitat_same_frame_rgb_metric_depth_v1",
                "rgb_shape": list(rgb.shape), "rgb_dtype": str(rgb.dtype),
                "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "depth_shape": list(depth.shape), "depth_dtype": str(depth.dtype),
                "depth_normalized": bool(sim.sensor_suite.sensors['depth'].normalize_depth),
                "depth_sha256": hashlib.sha256(depth.tobytes()).hexdigest(),
                "depth_finite_fraction": float(finite.mean()) if depth.size else 0.,
                "depth_min_m": float(finite_depth.min()) if finite_depth.size else None,
                "depth_max_m": float(finite_depth.max()) if finite_depth.size else None,
                "world_xy": [x, y], "world_yaw_deg": math.degrees(yaw),
            }
        if os.environ.get("SAGE_RULE_CREDIT", "0") == "1":
            from .mp3d_rule_credit_state import snapshot, call_success_detail
            value["rule_credit_state"] = snapshot(self, obs, geometry)
            value["native_goal_success_detail"] = call_success_detail(
                self, value['rule_credit_state'], native, [x, y])
        return value


def main():
    from . import server

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=int(os.environ.get("DEMO_HTTP_PORT", "8899")))
    parser.add_argument("--scene", default=os.environ.get("DEMO_SCENE", "2azQ1b91cZZ"))
    parser.add_argument("--episode_id", default=os.environ.get("DEMO_EPISODE_ID", "11"))
    parser.add_argument("--split", default=os.environ.get("DEMO_SPLIT", "val_unseen"))
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--backend", choices=("r2r", "rxr"), required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    args = parser.parse_args()
    root = Path(os.environ.get("RL_ARTIFACT_ROOT", "/data/closed_loop_3dgs_rl")).resolve()
    output_path = args.output_path.resolve()
    if not output_path.is_relative_to(root):
        parser.error("--output_path must be inside the authorized data disk root")

    def build_env(scene, episode_id, split, max_steps, port, backend="r2r"):
        env = VLNCEActionHeadEnv(
            scene=scene, episode_id=episode_id, split=split,
            max_steps=max_steps, backend=backend, output_path=str(output_path))
        scene_eps = {}
        for episode in env.env.episodes:
            scene_eps.setdefault(episode_scene(episode), []).append(str(episode.episode_id))
        for episodes in scene_eps.values():
            episodes.sort()
        server._SPEC.update(scenes=sorted(scene_eps), scene_eps=scene_eps,
                            splits=[split], backend=backend, backends=["r2r", "rxr"],
                            max_steps=max_steps, port=port,
                            success_contract=NATIVE_SUCCESS_CONTRACT)
        return env

    # Process-local factory replacement preserves the shared HTTP contracts,
    # including backend rebuilds, without changing SAGE or its services.
    server._build_env = build_env
    server._SPEC.update(max_steps=args.max_steps, port=args.port)
    server._ENV = build_env(args.scene, args.episode_id, args.split,
                            args.max_steps, args.port, args.backend)
    print(f"[vlnce_action_head] ready backend={args.backend} "
          f"scenes={len(server._SPEC['scenes'])} "
          f"episodes={sum(map(len, server._SPEC['scene_eps'].values()))} "
          f"port={args.port}", flush=True)
    http = server.HTTPServer(("0.0.0.0", args.port), server.Handler)
    try:
        http.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        http.server_close()
        if server._ENV is not None:
            server._ENV.close()


if __name__ == "__main__":
    main()
