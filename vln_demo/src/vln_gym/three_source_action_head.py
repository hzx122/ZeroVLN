"""Route the existing RL HTTP environment to SAGE or native R2R/RxR.

Factories, motion, sensors and scoring come from the existing adapters. This
entrypoint is exclusive to the independent three-source RL version.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .env import VLNEnv
from .vlnce_action_head import VLNCEActionHeadEnv, episode_scene
from .rl_observation_contract import (
    ProfiledSageTrainingEnv, ProfiledNativeTrainingEnv)

BACKENDS = ("sage", "r2r", "rxr")


def install_factory(server, output_path):
    from action_head_rl.cpu_memory import release_cpu_memory
    from .native_episode_cache import episode_catalog
    output_path = Path(output_path).resolve()
    allowed = Path(os.environ.get('RL_ARTIFACT_ROOT', '/data/zerovln')).resolve()
    if not output_path.is_relative_to(allowed):
        raise ValueError('environment outputs must be on the authorized data disk')
    if not os.environ.get('VLN_HABITAT_RENDER_URL'):
        raise ValueError('R2R/RxR require an explicit CPU renderer URL')
    server._on_session_closed = release_cpu_memory

    def build_env(scene, episode_id, split, max_steps, port, backend='sage', *,
                  defer_initial_reset=False):
        if backend not in BACKENDS:
            raise ValueError('unsupported three-source backend: ' + str(backend))
        if split != 'train':
            raise ValueError('three-source training only accepts the frozen train split')
        cls = ProfiledSageTrainingEnv if backend == 'sage' else ProfiledNativeTrainingEnv
        kwargs = dict(scene=scene, episode_id=str(episode_id), split=split,
            max_steps=max_steps, backend=backend,
            output_path=str(output_path / backend))
        if backend == 'sage':
            # The first real /reset supplies the original instruction and ID.
            kwargs['defer_initial_reset'] = True
        env = cls(**kwargs)
        scene_eps = episode_catalog(env.env.episodes)
        for episodes in scene_eps.values():
            episodes.sort()
        server._SPEC.update(scenes=sorted(scene_eps), scene_eps=scene_eps,
            splits=[split], backend=backend, backends=list(BACKENDS),
            max_steps=max_steps, port=port,
            renderer_contract='sage_gpu_3dgs_r2r_rxr_cpu_habitat_v1')
        return env

    server._build_env = build_env
    return build_env


def main():
    from . import server
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--episode_id', required=True)
    parser.add_argument('--backend', choices=BACKENDS, required=True)
    parser.add_argument('--split', choices=('train',), default='train')
    parser.add_argument('--max_steps', type=int, default=0)
    parser.add_argument('--output_path', type=Path, required=True)
    args = parser.parse_args()
    factory = install_factory(server, args.output_path)
    server._ENV = factory(args.scene, args.episode_id, args.split,
                          args.max_steps, args.port, args.backend)
    print(f'[three_source_action_head] ready backend={args.backend} '
          f'port={args.port}', flush=True)
    http = server.HTTPServer(('0.0.0.0', args.port), server.Handler)
    try:
        http.serve_forever()
    finally:
        http.server_close()
        for env in [server._ENV, *server._SESSIONS.values()]:
            if env is not None:
                env.close()


if __name__ == '__main__':
    main()
