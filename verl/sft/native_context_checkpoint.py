"""Publish the trainable v5 condition aggregator and the standard FM head."""
import json
import os
from pathlib import Path
import shutil

import torch

from action_head import protocol


def publish_native_checkpoint(output, step, selected, frozen_audit,
                              aggregator_delta, head_delta, version, engine):
    head = {name.split('action_head.', 1)[1]: value for name, value in selected.items()
            if 'action_head.' in name}
    aggregator = {name.split('context_aggregator.', 1)[1]: value
                  for name, value in selected.items() if 'context_aggregator.' in name}
    if not head or not aggregator:
        raise RuntimeError('v5 checkpoint is missing head or aggregator parameters')
    target = Path(output) / f'global_step_{step:06d}'
    pending = target.with_name(target.name + '.publishing')
    if pending.exists() or target.exists():
        raise FileExistsError(f'refusing to overwrite v5 checkpoint {target}')
    (pending / 'action_head').mkdir(parents=True)
    source = Path(os.environ['ACTION_HEAD_HEAD_CHECKPOINT'])
    shutil.copy2(source / 'fast_slow_action_head_config.json',
                 pending / 'action_head/fast_slow_action_head_config.json')
    torch.save(head, pending / 'action_head/fast_slow_action_head.pt')
    torch.save(aggregator, pending / 'context_aggregator.pt')
    report = {
        'schema': 'action_head_native_context_sft_report_v1',
        'condition_mode': 'v5_native', 'global_step': step,
        'local_step': step - int(os.environ.get('ACTION_HEAD_STEP_OFFSET', '0')),
        'step_offset': int(os.environ.get('ACTION_HEAD_STEP_OFFSET', '0')),
        'optimizer_state_restored': False,
        'data_resume_steps': int(os.environ.get('ACTION_HEAD_RESUME_DATA_STEPS', '0')),
        'warm_start_checkpoint': os.environ.get('ACTION_HEAD_WARM_START_CHECKPOINT', ''),
        'aggregator_initialization': os.environ.get('ACTION_HEAD_AGGREGATOR_CHECKPOINT') or None,
        'parameter_delta_reference': 'warm_start_boundary',
        'query_count': 0, 'condition_token_count': 16, 'condition_dim': 384,
        'world_size': torch.distributed.get_world_size(),
        'anchors_per_gpu': int(os.environ.get('ACTION_HEAD_ANCHORS_PER_GPU', '1')),
        'global_batch': torch.distributed.get_world_size() * int(os.environ.get('ACTION_HEAD_ANCHORS_PER_GPU', '1')),
        'backend': ('verl_sft_training_worker_ddp' if os.environ.get('ACTION_HEAD_NATIVE_DDP') == '1'
                    else 'verl_sft_training_worker_fsdp2'),
        'native_feature_cache': os.environ.get('ACTION_HEAD_NATIVE_CACHE') or None,
        'pin_memory': os.environ.get('ACTION_HEAD_PIN_MEMORY') == '1',
        'action_head_as_tool_version': version,
        **protocol.protocol_module(version).protocol_identity(),
        'frontend_model': os.environ['SLOW_MODEL'],
        'frontend_frozen': True, 'language_backbone_loaded': False,
        'history_image_limit': 11, 'history_slow_images': 8, 'history_fast_images': 3,
        'manifest': os.environ['ACTION_HEAD_MANIFEST'],
        'head_initialization': str(source),
        'aggregator_parameter_delta_l2': aggregator_delta,
        'action_head_parameter_delta_l2': head_delta,
        'frontend_audit': frozen_audit,
        'aggregator_parameter_count': sum(v.numel() for v in aggregator.values()),
        'action_head_parameter_count': sum(v.numel() for v in head.values()),
        'trainable_parameter_count': sum(p.numel() for p in engine.module.parameters() if p.requires_grad),
        'ok': aggregator_delta > 0 and head_delta > 0 and all(v == 0 for v in frozen_audit.values()),
    }
    config = os.environ.get('ACTION_HEAD_EXPERIMENT_CONFIG')
    if config:
        shutil.copy2(config, pending / 'training_experiment.json')
    (pending / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    pending.replace(target)
    tb = os.environ.get('TENSORBOARD_DIR')
    if tb:
        from torch.utils.tensorboard import SummaryWriter
        with SummaryWriter(str(Path(tb) / Path(output).name)) as writer:
            writer.add_scalar('train/aggregator_delta_l2', aggregator_delta, step)
            writer.add_scalar('train/head_delta_l2', head_delta, step)
