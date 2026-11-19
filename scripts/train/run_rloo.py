"""Launch the existing VERL navigation trainer with native Hydra overrides."""
from dataclasses import fields
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from gpu_runtime import REPO, artifact_root, environment, execute, save
from rloo_variants import RLOOSettings


def settings_from_config(config):
    """All six values must come from YAML, never implicit Python preset defaults."""
    values = OmegaConf.to_container(config.rloo, resolve=True, throw_on_missing=True)
    expected = {field.name for field in fields(RLOOSettings)}
    if set(values) != expected:
        raise ValueError('rloo must define exactly these fields: ' + ', '.join(sorted(expected)))
    return RLOOSettings(**values).validate()


def configuration(config):
    settings = settings_from_config(config)
    document = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    recipe = document['recipe']
    if document['run']['batch_size'] is not None:
        from prepare_git_rl_task import configure
        batch = document['run']['batch_size']
        if type(batch) is not int:
            raise ValueError('run.batch_size must be an integer')
        configure({'config': recipe}, batch_size=batch)
    for name in ('opt_steps', 'max_tool_calls'):
        if type(recipe[name]) is not int or recipe[name] < 1:
            raise ValueError('recipe.' + name + ' must be a positive integer')
    if recipe['terminal_success_reward'] != settings.terminal_success_reward:
        raise ValueError('set rloo.terminal_success_reward; recipe must retain its interpolation')
    return document, settings


def build_launch(config):
    from v6_rl import build_plan
    document, settings = configuration(config)
    if not document['run']['output_dir']:
        raise ValueError('run.output_dir is required for prepare/run')
    root = artifact_root(document['run']['output_dir'])
    checkpoint = Path(document['checkpoint'])
    recipe = document['recipe']
    for path in [checkpoint/'action_head', checkpoint/'dualvln_query',
                 Path(recipe['slow_model']), Path(recipe['fast_backbone']),
                 Path(recipe['case_manifest'])]:
        if not path.is_absolute() or not path.exists():
            raise ValueError('Missing absolute input path: ' + str(path))
    urls = [url.strip() for url in document['bench_urls'].split(',') if url.strip()]
    if len(urls) != recipe['rollout_actors'] or len(set(urls)) != len(urls):
        raise ValueError('bench_urls must supply one distinct Gym per rollout actor')
    resume = document['run']['resume_from']
    launch = build_plan(recipe, head_checkpoint=str(checkpoint/'action_head'),
        query_checkpoint=str(checkpoint/'dualvln_query'), bench_urls=','.join(urls),
        output_dir=str(root/'train'), resume_from_path=str(Path(resume).resolve()) if resume else '')
    env = environment(root, document.get('environment'))
    env.update(launch['env'])
    env.update(settings.environment())
    env['VERL_SFT_CONFIG_DIR'] = str(Path(env['VERL_SOURCE_ROOT'])/'verl/trainer/config')
    env['PYTHONPATH'] = os.pathsep.join((str(REPO/'verl'), env['VERL_SOURCE_ROOT'], str(REPO/'vln_demo/src'), env['PYTHONPATH']))
    if recipe.get('text_backend') == 'megatron':
        mcore = Path(env['ACTION_HEAD_MEGATRON_ENV'])
        for path in (mcore/'mcore', mcore/'bridge/src'):
            if not path.is_dir():
                raise ValueError('Missing Megatron dependency: '+str(path))
        env['PYTHONPATH'] = os.pathsep.join((str(mcore/'mcore'), str(mcore/'bridge/src'), env['PYTHONPATH']))
        env['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
    return root, document, launch['command'], env


def run(config):
    mode = config.run.mode
    if mode not in ('prepare', 'run'):
        raise ValueError('run.mode must be prepare or run; use --cfg job to view configuration')
    root, document, command, env = build_launch(config)
    if mode == 'prepare':
        root.mkdir(parents=True, exist_ok=False)
    else:
        if (root/'train').exists() or (root/'launch/trainer.started.json').exists():
            raise ValueError('Use a new output directory; this attempt already ran')
        root.mkdir(parents=True, exist_ok=True)
    # Fully resolved YAML includes CLI overrides and any batch/source rebalancing.
    save(root/'launch/config.yaml', document)
    save(root/'launch/recipe.yaml', document['recipe'])
    save(root/'launch/command.json', dict(command=command, cwd=str(REPO)))
    print(OmegaConf.to_yaml(OmegaConf.create(dict(output=str(root), rloo=document['rloo'], command=command))), flush=True)
    return 0 if mode == 'prepare' else execute(command, env, root/'launch/trainer.json')


@hydra.main(version_base=None, config_path='../../configs/gpu', config_name='rloo')
def main(config: DictConfig):
    raise SystemExit(run(config))


if __name__ == '__main__':
    main()
