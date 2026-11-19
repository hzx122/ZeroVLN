"""Private native VLN-CE goal evidence for the existing call-credit algorithm.

No route, goal, distance or semantic oracle is added to model observations.
Arrival is checked after actual simulator primitives, never interpolated.
"""
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'mp3d-rule-credit-state-v1'
ARRIVAL = 'habitat_native_goal_geodesic_3m_actual_steps_v1'
TERMINAL = 'habitat_vlnce_native_stop_3m_v1'
_mesh_hashes = {}


def call_success_detail(gym, state, native, world_xy):
    """Bind measured native STOP success to private geodesic call state."""
    distance = state['goal_distance_m']
    stopped = gym.env.task.is_stop_called
    if type(stopped) is not bool or native.get('success') not in (0, 1, False, True):
        raise RuntimeError('native STOP/success evidence unavailable')
    success = bool(native['success'])
    if success != (stopped and distance < 3.):
        raise RuntimeError('native success contradicts actual STOP/geodesic evidence')
    binding = dict(schema=TERMINAL, map_sha256=state['map_sha256'],
                   goal_contract_sha256=state['goal_contract_sha256'], radius_m=3.)
    return dict(**binding, available=True,
        contract_sha256=hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest(),
        position_xy=list(world_xy), goal_distance_m=distance,
        inside_goal_range=distance < 3., native_stop_called=stopped, success=success)


class NativeRuleCreditState:
    def __init__(self, gym):
        self.gym = gym
        episode = gym.env.current_episode
        self.goals = [[float(v) for v in goal.position] for goal in episode.goals]
        self.ever_goal_reached = False
        self.initialized = False
        self.information = None
        mesh = Path(episode.scene_id).with_suffix('.navmesh').resolve()
        stat = mesh.stat()
        key = (str(mesh), stat.st_size, stat.st_mtime_ns)
        if key not in _mesh_hashes:
            _mesh_hashes[key] = hashlib.sha256(mesh.read_bytes()).hexdigest()
        self.map_hash = _mesh_hashes[key]
        self.goal_hash = hashlib.sha256(json.dumps(dict(backend=gym.backend,
            scene=episode.scene_id, episode_id=str(episode.episode_id),
            goals=self.goals, radius_m=3.0), sort_keys=True).encode()).hexdigest()

    def observe_actual_position(self):
        sim = self.gym.env.sim
        position = sim.get_agent_state().position
        distance = float(sim.geodesic_distance(position, self.goals))
        if not math.isfinite(distance) or distance < 0:
            raise RuntimeError('native goal geodesic unavailable; no Euclidean fallback; '
                f'backend={self.gym.backend} episode={self.gym.env.current_episode.episode_id} '
                f'scene={self.gym.env.current_episode.scene_id} '
                f'position={list(map(float, position))} goals={self.goals} distance={distance}')
        self.position_xyz = [float(v) for v in position]
        self.distance = distance
        self.ever_goal_reached |= distance < 3.0
        self.initialized = True

    def snapshot(self, obs=None, geometry=None):
        self.observe_actual_position()
        native = self.gym.env.get_metrics()
        collision = native.get('collisions', {})
        information = dict(information_available=False, relevant_entities=[],
            information_unavailable_reason='no observation supplied to geometry-only caller')
        if obs is not None:
            from .mp3d_information_reward import InformationState
            if self.information is None:
                self.information = InformationState(self.gym.env.current_episode)
            information = self.information.snapshot(self.gym, obs, geometry)
        return dict(schema=SCHEMA, backend=self.gym.backend,
            map_sha256=self.map_hash, goal_contract_sha256=self.goal_hash,
            target_object_id='native_episode_goal', goal_distance_m=self.distance,
            position_xyz=list(self.position_xyz),
            ever_goal_reached=bool(self.ever_goal_reached), arrival_evidence=ARRIVAL,
            collisions=int(collision.get('count', 0)) if isinstance(collision, dict) else int(collision),
            **information)


def snapshot(gym, obs=None, geometry=None):
    state = getattr(gym, '_native_rl_credit', None)
    if state is None:
        state = gym._native_rl_credit = NativeRuleCreditState(gym)
        if not getattr(gym.env, '_native_rl_step_observer', False):
            original = gym.env.step

            def measured_step(*args, **kwargs):
                result = original(*args, **kwargs)
                current = getattr(gym, '_native_rl_credit', None)
                if current is not None:
                    current.observe_actual_position()
                return result

            gym.env.step = measured_step
            gym.env._native_rl_step_observer = True
    return state.snapshot(obs, geometry)
