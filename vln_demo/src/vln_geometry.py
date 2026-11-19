"""Dataset-specific collision bodies for the shared R2R/RxR simulator.

RxR dimensions are from the official VLN-CE rxr_vlnce_english_task.yaml.
Camera, actions and controller parameters remain the console's configuration.
"""
from functools import wraps

RXR_BODY = (0.88, 0.18)


def configure_episode(env, episode):
    if 'mp3d' not in str(getattr(episode, 'scene_id', '')).split('/'):
        return
    config = getattr(env, '_config', None)
    if config is None or not hasattr(episode, 'instruction'):
        return
    from habitat.config import read_write
    agent = config.simulator.agents[config.simulator.agents_order[0]]
    if not hasattr(env, '_vln_default_body'):
        env._vln_default_body = (float(agent.height), float(agent.radius))
    rxr = str(getattr(episode, 'scene', '')).startswith('rxr-')
    height, radius = RXR_BODY if rxr else env._vln_default_body
    with read_write(config):
        agent.height, agent.radius = height, radius


def register():
    """Habitat-Lab skips native reconfigure for same-scene episode switches."""
    import habitat_sim
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
    if getattr(HabitatSim, '_vln_body_switch_installed', False):
        return
    original = HabitatSim.reconfigure

    @wraps(original)
    def reconfigure(self, *args, **kwargs):
        original(self, *args, **kwargs)
        actual = self.config.agents[0]
        requested = self.sim_config.agents[0]
        if (actual.height, actual.radius) != (requested.height, requested.radius):
            habitat_sim.Simulator.reconfigure(self, self.sim_config)
            self._update_agents_state()

    HabitatSim.reconfigure = reconfigure
    HabitatSim._vln_body_switch_installed = True
