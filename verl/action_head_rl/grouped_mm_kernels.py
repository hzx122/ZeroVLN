"""Experimental native forward with full-gradient VeOmni GEMM backward.

Only replaces HF's grouped matrix multiplication. Routing, activation, and
routing-weight placement remain in the original HF implementation. This is
not enabled by a training configuration until real-device validation passes.
"""
from contextlib import contextmanager
from functools import lru_cache
import os

import torch

from .grouped_mm_offsets import host_offsets


@lru_cache(maxsize=1)
def _kernels():
    # Importing the VeOmni package otherwise installs unrelated attention
    # patches. Request its documented HF backend just for this lazy import.
    original = os.environ.get('MODELING_BACKEND')
    os.environ['MODELING_BACKEND'] = 'hf'
    try:
        from veomni.ops.kernels.moe._kernels.kernel.group_gemm import (
            group_gemm_same_mn, group_gemm_same_nk,
        )
    finally:
        if original is None:
            os.environ.pop('MODELING_BACKEND', None)
        else:
            os.environ['MODELING_BACKEND'] = original
    return group_gemm_same_nk, group_gemm_same_mn


class _TrainableGroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, offsets):
        if (inputs.ndim != 2 or weights.ndim != 3
                or inputs.shape[1] != weights.shape[1]
                or offsets.numel() != weights.shape[0]):
            raise ValueError('require [tokens,in] @ [experts,in,out] and expert offsets')
        if (inputs.device.type != 'cuda' or weights.device != inputs.device
                or offsets.device != inputs.device):
            raise ValueError('full-gradient grouped GEMM requires CUDA/PPU operands and offsets')
        if inputs.dtype != torch.bfloat16 or weights.dtype != torch.bfloat16:
            raise ValueError('full-gradient grouped GEMM is limited to BF16')
        # Reuse the version-checked host-copy helper, but pass CUDA offsets to
        # the kernels. HF reuses one offsets tensor for both projections.
        host = host_offsets(offsets)
        counts = torch.diff(host, prepend=host.new_zeros(1))
        if bool((counts < 0).any()) or int(host[-1]) != inputs.shape[0]:
            raise ValueError('expert offsets must partition all input rows')
        max_rows = int(counts.max())
        # HF supplies a transpose of [experts,out,in], so this generally is a
        # view of existing weights, not another complete weight allocation.
        weight_nt = weights.transpose(-2, -1).contiguous()
        inputs = inputs.contiguous()
        # Preserve the native forward rounding, including checkpoint replay.
        # Only dX/dW are candidates for replacement; the failed all-kernel
        # variant remains frozen in its separate diagnostic artifacts.
        result = torch._grouped_mm(inputs, weights, offs=offsets)
        ctx.save_for_backward(inputs, weight_nt, offsets)
        ctx.max_rows = max_rows
        return result

    @staticmethod
    def backward(ctx, gradient):
        group_gemm_same_nk, group_gemm_same_mn = _kernels()
        inputs, weight_nt, offsets = ctx.saved_tensors
        gradient = gradient.contiguous()
        input_grad = weight_grad = None
        if ctx.needs_input_grad[0]:
            input_grad = group_gemm_same_nk(gradient, weight_nt, offsets,
                                           ctx.max_rows, transpose_b=False)
        if ctx.needs_input_grad[1]:
            weight_grad_nt = torch.empty_like(weight_nt)
            group_gemm_same_mn(gradient, inputs, weight_grad_nt, offsets,
                               ctx.max_rows, transpose_a=True)
            weight_grad = weight_grad_nt.transpose(-2, -1)
        return input_grad, weight_grad, None


def trainable_grouped_mm(inputs, weights, *, offs):
    # Match HF's original grouped-MM dispatch under autocast.
    return _TrainableGroupedMM.apply(inputs.to(weights.dtype), weights, offs)


@contextmanager
def trainable_expert_grouped_mm():
    """Install only within one isolated worker's candidate diagnostic scope."""
    import transformers.integrations.moe as moe
    original = moe._grouped_mm
    moe._grouped_mm = trainable_grouped_mm
    try:
        yield
    finally:
        moe._grouped_mm = original
