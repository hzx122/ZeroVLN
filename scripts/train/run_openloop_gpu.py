"""Prepare/init/train the existing published-input exec4 SFT recipe directly."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from gpu_runtime import REPO, artifact_root, environment, execute, load, save


def prepare_smoke(root, steps):
    """Explicit four-GPU smoke; preserve the 1:1:2 source ratio and real trainer."""
    launch = load(root/'launch-plan.json')
    config = load(root/'experiment_config.json')
    config['training'].update(nnodes=1, nproc_per_node=4, anchors_per_gpu=1, total_epochs=1)
    config['scales']['D16']['max_steps'] = steps
    save(root/'experiment_config.json', config)
    checksum = hashlib.sha256((root/'experiment_config.json').read_bytes()).hexdigest()
    launch['experiment_config_sha256'] = checksum
    launch['environment'].update(NNODES='1', NPROC_PER_NODE='4',
        ACTION_HEAD_TOTAL_EPOCHS='1', MAX_STEPS=str(steps),
        ACTION_HEAD_CHECKPOINT_STEPS=str(steps),
        ACTION_HEAD_EXPERIMENT_CONFIG_SHA256=checksum)
    save(root/'launch-plan.json', launch)
    schedule = load(root/'schedule.json')
    anchors = schedule['anchors']
    schedule.update(smoke_test=True, global_batch=4, epochs=1, max_steps=steps,
        steps_per_epoch=(anchors+3)//4, checkpoint_steps=[steps], checkpoints=1,
        exposures_per_epoch=4*steps,
        exposure_per_source={name:steps*weight for name,weight in
                             schedule['named_source_sampling_ratio'].items()},
        training_layout=dict(mode='explicit_smoke', devices=[[0,1,2,3]],
            nproc_per_node=4, world_size=4, anchors_per_gpu=1, global_batch=4,
            idle_devices=[list(range(4,16))]),
        config=dict(path=str(root/'experiment_config.json'), sha256=checksum),
        validation_scope='A bounded real-data training run, not a paper-result reproduction.')
    save(root/'schedule.json', schedule)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['show-config', 'prepare', 'initialize', 'run'])
    p.add_argument('--config', type=Path, default=REPO/'configs/gpu/openloop.json')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--node-rank', type=int, default=0)
    p.add_argument('--master-addr', default='127.0.0.1')
    p.add_argument('--master-port', type=int, default=29561)
    p.add_argument('--smoke-steps', type=int, default=0,
                   help='prepare only: explicit one-node/four-GPU bounded real-data training')
    a = p.parse_args(argv)
    if a.smoke_steps < 0 or (a.smoke_steps and a.mode != 'prepare'):
        p.error('--smoke-steps must be positive and is only valid with prepare')
    config = load(a.config, strict=a.mode != 'show-config')
    if a.mode == 'show-config':
        print(json.dumps(config, indent=2)); return 0
    root = artifact_root(a.output_dir)
    if a.mode == 'prepare':
        root.mkdir(parents=True, exist_ok=False)
        base = load(REPO/'configs/action_head/v6_official_gvln_d16_beam4_10epochs.json')
        base['shared']['slow_model'] = config.pop('slow_model')
        base['initialization'].update(config.pop('initialization'))
        # These values are replaced by prepare_published_training's published snapshot.
        base['shared']['manifest'] = ''
        base['shared']['validation_manifest'] = ''
        template = root/'structure.json'
        save(template, base)
        config.update(root=str(root), run_name=root.name, python=sys.executable,
            training_source=str(REPO),
            git_runtime={'verl_checkout': config.pop('verl_source_root')},
            v6_template=dict(path=str(template),sha256=hashlib.sha256(template.read_bytes()).hexdigest()))
        from rule_teacher_training_handoff import prepare_published_training
        prepare_published_training(config)
        if a.smoke_steps:
            prepare_smoke(root, a.smoke_steps)
        save(root/'gpu-config.json', config)
        return 0
    launch = load(root/'launch-plan.json')
    nodes = int(launch['environment']['NNODES'])
    if not 0 <= a.node_rank < nodes:
        p.error(f'node rank must be in 0..{nodes-1} for the prepared plan')
    env = environment(root)
    env.update(launch['environment'])
    env.update(PYTHON=sys.executable, NODE_RANK=str(a.node_rank), MASTER_ADDR=a.master_addr,
               MASTER_PORT=str(a.master_port))
    # SDK and communication paths come from the activated environment, never an image.
    for key in ('PPU_SDK', 'CUDA_PATH'):
        import os
        if key in os.environ: env[key] = os.environ[key]
        else: env.pop(key, None)
    if a.mode == 'initialize':
        if a.node_rank != 0: p.error('initialize runs once on rank0')
        command = launch['initialize_argv']
    else:
        receipt = root/'launch/initialize-node0.json'
        initialized = root/'initialization/d16'
        if (not receipt.is_file() or load(receipt).get('returncode') != 0
                or not (initialized/'scale_report.json').is_file()
                or not all((initialized/name).is_dir() for name in ('head', 'query'))):
            p.error('initialize must finish successfully before starting all four training nodes')
        command = launch['train_argv']
    return execute(command, env, root/'launch'/f'{a.mode}-node{a.node_rank}.json')


if __name__ == '__main__':
    sys.exit(main())
