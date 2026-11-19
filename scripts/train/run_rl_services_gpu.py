"""Run local renderer/Gym services using the training YAML's RLOO settings."""
import os
from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig

from gpu_runtime import REPO
from run_rloo import settings_from_config


def run(config):
    mode = config.run.mode
    if mode not in ('run', 'start', 'stop', 'status'):
        raise ValueError('service run.mode must be run/start/stop/status; inspect with --cfg job')
    settings = settings_from_config(config)
    env = dict(os.environ, **settings.environment())
    env.update(REPO=str(REPO), PYTHON=sys.executable, SAGE_RENDER_DEVICE='cpu', VALIDATION_GYMS='0',
               THREE_SOURCE_CASE_MANIFEST=config.recipe.case_manifest, CPU_RENDER_SHARED_CACHE='1')
    for key, value in dict(SAGE_CPU_RENDER_PARALLELISM='8', TRAIN_GYMS='64',
                          CPU_RENDER_CACHE_MAX_GIB='128', CPU_RENDER_SCENE_LOAD_RESERVE_GIB='16').items():
        env.setdefault(key, value)
    env['SAGE_CPU_RENDER_CACHE'] = str(Path(os.environ.get('GPU_CACHE_ROOT', '/tmp/closed-loop-gpu'))/'sage-render')
    os.execvpe('bash', ['bash', str(REPO/'scripts/train/v6_rl_local_services.sh'), mode], env)


@hydra.main(version_base=None, config_path='../../configs/gpu', config_name='rloo')
def main(config: DictConfig):
    run(config)


if __name__ == '__main__':
    main()
