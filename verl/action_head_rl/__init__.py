"""Closed-loop reinforcement learning for the ZeroVLN brain and action expert."""

import os

if os.environ.get('ACTION_HEAD_MEGATRON_ENV'):
    # The upstream engine registry imports MCore before worker.reset(). Bind
    # the same PPU TE compatibility used by the accepted history runner first.
    from sft.probe_qwen36_megatron import expose_transformer_engine_pytorch
    expose_transformer_engine_pytorch()

from .config import RouterReplayMode, TrainScope, validate_training_contract
from .sde import ReverseSDEConfig, recompute_reverse_logprob, sample_reverse_sde

__all__ = [
    "ReverseSDEConfig",
    "RouterReplayMode",
    "TrainScope",
    "recompute_reverse_logprob",
    "sample_reverse_sde",
    "validate_training_contract",
]
