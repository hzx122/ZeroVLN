"""Experimental host offsets for PPU's native grouped-MM fallback.

Not installed in a training runtime. The native fallback copies offsets to
CPU in forward, dX and dW; preserving one host copy may remove repeat syncs.
Real-device dispatch, gradients and speed must be checked before integration.
"""
from __future__ import annotations

import torch

_CACHE_ATTRIBUTE = '_v6_grouped_mm_host_offsets'


def host_offsets(offsets: torch.Tensor) -> torch.Tensor:
    if offsets.ndim != 1 or offsets.dtype != torch.int32:
        raise ValueError('native grouped-MM offsets must be one-dimensional int32')
    if offsets.device.type == 'cpu':
        return offsets
    if offsets.device.type != 'cuda':
        raise ValueError('host-offset candidate is limited to CPU reference and CUDA/PPU')
    # The cache lives on this exact short-lived offset tensor. A view's version
    # changes with its base; neither mutation nor allocator reuse may reuse a
    # stale host copy. Inference tensors have no version and are not cached.
    try:
        version = offsets._version
    except RuntimeError:
        return offsets.detach().cpu()
    cached = getattr(offsets, _CACHE_ATTRIBUTE, None)
    if cached is None or cached[0] != version:
        cached = (version, offsets.detach().cpu())
        setattr(offsets, _CACHE_ATTRIBUTE, cached)
    return cached[1]


def grouped_mm_with_host_offsets(inputs, weights, *, offs):
    """Use the original native matmul/autograd, changing only offset placement."""
    return torch._grouped_mm(inputs, weights, offs=host_offsets(offs))
