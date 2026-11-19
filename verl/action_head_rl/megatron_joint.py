"""Schedule existing FM replay and Megatron text without changing loss mass."""
from __future__ import annotations

import math
import struct
import torch

from action_head_rl.text_policy import is_text_record
from action_head_rl.megatron_trajectory_text import shard_text_trajectories


def text_transport_dp_waves(waves, placements):
    """Recover logical DP shards for upstream ``dispatch_nd_compute``.

    This runs after credit, padding and sharding. Scalars must match exactly;
    tensors/containers must be the SAME objects, not merely share an episode ID.
    A divergent model-parallel replica is an error, never silently overwritten.
    Return None for mixed/FM waves, whose physical-rank ownership is unchanged.
    """
    if not waves or not all(wave and all(is_text_record(row) for row in wave)
                            for wave in waves):
        return None
    if sorted(p['rank'] for p in placements) != list(range(len(waves))):
        raise ValueError('need one placement per physical learner rank')
    dp_size = placements[0]['text_dp_size']
    if dp_size <= 0 or any(p['text_dp_size'] != dp_size for p in placements):
        raise ValueError('inconsistent Megatron DP sizes')

    def identity(value):
        kind = type(value)
        if kind in (str, bytes, int, bool, type(None)):
            return kind, value
        if kind is float:
            return kind, struct.pack('!d', value)
        return kind, id(value)

    signatures = {}
    logical_waves = [None] * dp_size
    mapping = [None] * len(waves)
    for placement in placements:
        rank, dp_rank = placement['rank'], placement['text_dp_rank']
        if not 0 <= dp_rank < dp_size:
            raise ValueError('invalid Megatron DP rank')
        wave = waves[rank]
        signature = tuple(tuple((key, identity(value))
                                for key, value in sorted(row.items()))
                          for row in wave)
        if dp_rank in signatures and signatures[dp_rank] != signature:
            raise ValueError('text payload differs within a Megatron DP replica')
        signatures[dp_rank] = signature
        logical_waves[dp_rank] = wave
        mapping[rank] = dp_rank
    if any(wave is None for wave in logical_waves):
        raise ValueError('incomplete Megatron DP placement')
    return mapping, logical_waves


def shard_joint_records(records, placements, *, fm_sharder, text_sharder=shard_text_trajectories):
    """Physical-rank transport: FM is FSDP-DP, text is Megatron dense-DP.

    Only the transport copies text across TP/PP/CP ranks. Megatron's pipeline
    and loss reduction, rather than the driver, own those parallel replicas.
    """
    world = len(placements)
    if sorted(p['rank'] for p in placements) != list(range(world)):
        raise ValueError('need one placement per physical learner rank')
    dp_size = placements[0]['text_dp_size']
    if any(p['text_dp_size'] != dp_size for p in placements):
        raise ValueError('inconsistent Megatron DP sizes')
    counts = [sum(p['text_dp_rank'] == rank for p in placements) for rank in range(dp_size)]
    if not counts or len(set(counts)) != 1 or not all(counts) or sum(counts) != world:
        raise ValueError('incomplete Megatron DP replica placement')
    text = [r for r in records if is_text_record(r)]
    if not text or any(float(r.get('sample_valid', 1.)) != 1. for r in text):
        raise ValueError('schedule unpadded complete trajectories')
    if len({r['condition_id'] for r in text}) != len(text):
        raise ValueError('duplicate trajectory in whole-step update')
    logical_valid = sum(float(r.get('loss_weight', 1.)) for r in text)
    if not math.isfinite(logical_valid) or logical_valid <= 0:
        raise ValueError('invalid original text loss mass')
    text_shards = text_sharder(text, dp_size=dp_size,
        policy_version=text[0].get('update_policy_version', text[0]['policy_version']))
    fm = [r for r in records if not is_text_record(r)]
    shards = fm_sharder(fm, world)[0] if fm else [[] for _ in range(world)]
    for placement in placements:
        shards[placement['rank']].extend(dict(r, sample_valid=valid)
            for r, valid in text_shards[placement['text_dp_rank']])
    if len(set(map(len, shards))) != 1:
        raise ValueError('physical-rank transport waves must remain aligned')
    return shards, logical_valid


def fm_parameters(engine):
    """Qwen is a differentiable condition, never an FM optimization target."""
    module = engine.module
    if any(p.grad is not None for p in module.qwen.parameters()):
        raise RuntimeError('FM replay unexpectedly wrote Qwen gradients')
    if getattr(engine, 'scaler', None) is not None:
        raise ValueError('joint BF16 update cannot use an independent loss scaler')
    return list(module.query_encoder.parameters()) + list(module.action_head.parameters())


def reduced_fm_norm(engine):
    from verl.utils.fsdp_utils import fsdp2_clip_grad_norm_
    from torch.distributed.tensor import DTensor
    # Infinity computes the existing distributed norm without clipping. Text
    # and action leaves are clipped together only after text backward finishes.
    norm = fsdp2_clip_grad_norm_(fm_parameters(engine), max_norm=float('inf'))
    if isinstance(norm, DTensor):
        norm = norm.full_tensor()
    value = float(norm)
    if not math.isfinite(value):
        raise RuntimeError('non-finite FM gradients; neither optimizer may step')
    return value


@torch.no_grad()
def step_fm(engine, *, clip_factor):
    if not math.isfinite(clip_factor) or not 0 < clip_factor <= 1:
        raise ValueError('invalid joint clip factor')
    for parameter in fm_parameters(engine):
        if parameter.grad is not None:
            parameter.grad.mul_(clip_factor)
    engine.optimizer.step()
    engine.lr_scheduler_step()


@torch.no_grad()
def refresh_fm_condition(module, name, value):
    """Refresh the old-policy FSDP Qwen mirror from Bridge's same HF stream."""
    from torch.distributed.tensor import DTensor, distribute_tensor
    from action_head_rl.checkpoint import hf_qwen_export_key
    parameters = {hf_qwen_export_key(k): p for k, p in module.qwen.named_parameters()}
    if name not in parameters:
        raise ValueError('Megatron exported unknown FM condition parameter: ' + name)
    target = parameters[name]
    if tuple(value.shape) != tuple(target.shape):
        raise ValueError('FM condition parameter shape changed: ' + name)
    value = value.to(device=target.device, dtype=target.dtype)
    if isinstance(target, DTensor):
        value = distribute_tensor(value, target.device_mesh, target.placements)
    target.copy_(value)
