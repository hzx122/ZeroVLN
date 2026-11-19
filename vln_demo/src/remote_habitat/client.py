"""Opt-in Habitat-Lab adapter: local navmesh/actions, remote RGB + metric depth."""
import io
import json
import os
from pathlib import Path
import uuid
from types import SimpleNamespace
from urllib.request import Request, ProxyHandler, build_opener

import numpy as np


def install(url=None):
    url = url or os.environ.get('VLN_HABITAT_RENDER_URL')
    if not url:
        return
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
    from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis
    if getattr(HabitatSim, '_remote_render_installed', False):
        return
    create = HabitatSim.create_sim_config
    get_state = HabitatSim.get_agent_state
    original_reconfigure = HabitatSim.reconfigure
    original_step = HabitatSim.step

    def reconfigure(self, habitat_config, ep_info=None, should_close_on_new_scene=True):
        # This Habitat build crashes in native close(destroy=False) when no GL
        # context was created. Its supported in-place reconfigure path reloads
        # geometry/navmesh without that renderer-specific close operation.
        return original_reconfigure(self, habitat_config, ep_info, should_close_on_new_scene=False)

    def state_with_sensors(self, agent_id=0):
        import quaternion
        state = get_state(self, agent_id)
        for spec in getattr(self, '_remote_specs', []):
            pitch, yaw, roll = map(float, spec['orientation'])
            if yaw != 0 or roll != 0:
                raise ValueError('remote navigation adapter supports camera pitch; yaw/roll must be zero')
            rotation = (state.rotation * quat_from_angle_axis(pitch, np.array([1., 0., 0.]))
                        if pitch else state.rotation)
            state.sensor_states[spec['uuid']] = SimpleNamespace(
                position=state.position + quaternion.rotate_vectors(state.rotation, spec['position']),
                rotation=rotation)
        return state

    def config(self, suite):
        cfg = create(self, suite)
        self._remote_last_raw = None
        specs = cfg.agents[0].sensor_specifications
        self._remote_specs = [dict(uuid=s.uuid, type=s.sensor_type.name,
                                   resolution=[int(v) for v in s.resolution], hfov=float(s.hfov),
                                   position=[float(v) for v in s.position],
                                   orientation=[float(v) for v in s.orientation]) for s in specs]
        self._remote_scene = str(Path(cfg.sim_cfg.scene_id).resolve())
        cfg.agents[0].sensor_specifications = []
        cfg.sim_cfg.create_renderer = False
        return cfg

    def observe(self, agent_ids=0):
        if isinstance(agent_ids, (list, tuple, range)):
            return {agent_id: observe(self, agent_id) for agent_id in agent_ids}
        if agent_ids != 0:
            raise ValueError('remote Habitat adapter supports one agent per private Env')
        if getattr(self, '_remote_defer_depth', 0):
            cached = getattr(self, '_remote_last_raw', None)
            if cached is None:
                raise RuntimeError('batch motion requires an initial observation')
            # Habitat needs sensor keys while stepping; never expose these
            # old pixels as a current observation outside this adapter.
            return dict(cached)
        state = self.get_agent_state()
        scene = Path(self._remote_scene)
        request_id = uuid.uuid4().hex
        payload = dict(request_id=request_id, scene='/'.join(scene.parts[-2:]),
                       position=state.position.tolist(), rotation=quat_to_coeffs(state.rotation).tolist(),
                       sensors=self._remote_specs)
        request = Request(url.rstrip('/') + '/render', data=json.dumps(payload).encode(),
                          headers={'Content-Type': 'application/json'})
        timeout = float(os.environ.get('VLN_RENDER_TIMEOUT', '0'))
        with build_opener(ProxyHandler({})).open(request, timeout=None if timeout == 0 else timeout) as response:
            body = response.read(32 * 1024 * 1024)
        with np.load(io.BytesIO(body), allow_pickle=False) as arrays:
            if str(arrays['request_id']) != request_id:
                raise RuntimeError('remote rendering response identity mismatch')
            raw = {key: arrays[key].copy() for key in arrays.files if key != 'request_id'}
        self._remote_last_raw = raw
        return dict(raw)

    def step(self, action):
        observations = original_step(self, action)
        if getattr(self, '_remote_defer_depth', 0):
            self._remote_deferred_steps = getattr(self, '_remote_deferred_steps', 0) + 1
            observations = {k: v for k, v in observations.items() if k not in ('rgb', 'depth')}
            observations['_render_deferred'] = True
        return observations

    HabitatSim.step = step
    HabitatSim.create_sim_config = config
    HabitatSim.get_sensor_observations = observe
    HabitatSim.get_agent_state = state_with_sensors
    HabitatSim.reconfigure = reconfigure
    HabitatSim._remote_render_installed = True
