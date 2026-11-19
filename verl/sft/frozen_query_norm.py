"""Frozen FLA gated RMSNorm VJP using the original forward statistics.

Recomputing RMS statistics on only the query rows can change FP32 rounding
when Triton chooses another tile. Keep the statistics from the full native
forward, just as attention retains its original log-sum-exp.
"""
from functools import lru_cache
import hashlib
from importlib import import_module, metadata
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable


EXPECTED_SOURCE_SHA256 = 'd3a40dee12c22b8416d1cf61483c0e8eaa87e44765489b1b2f22c1525f0394cf'


@lru_cache(maxsize=1)
def _fla_norm_ops():
    if metadata.version('fla-core') != '0.5.0':
        raise RuntimeError('Frozen-query gated norm requires validated fla-core 0.5.0')
    module = import_module('fla.modules.fused_norm_gate')
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != EXPECTED_SOURCE_SHA256:
        raise RuntimeError('Frozen-query gated norm FLA source differs from validated source')
    return module


class _FrozenQueryFusedNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, gate, weight, bias, eps, activation, count):
        if inputs.ndim != 2 or inputs.shape != gate.shape or not 0 < count <= inputs.shape[0]:
            raise ValueError('Frozen-query gated norm requires matching flat rows and a valid suffix')
        if any(value is not None and value.requires_grad for value in (weight, bias)):
            raise ValueError('Frozen-query gated norm weights must be frozen')
        ops = _fla_norm_ops()
        # Match the original input_guard and LayerNormGatedFunction forward.
        inputs, gate = inputs.contiguous(), gate.contiguous()
        output, mean, rstd, residual_out = ops.layer_norm_gated_fwd(
            x=inputs, g=gate, weight=weight, bias=bias, activation=activation,
            eps=eps, residual=None, residual_dtype=None, is_rms_norm=True)
        if mean is not None or residual_out.data_ptr() != inputs.data_ptr():
            raise RuntimeError('Unexpected native gated RMSNorm forward contract')
        ctx.save_for_backward(inputs[-count:].detach().clone(), gate[-count:].detach().clone(),
                              weight, bias, rstd[-count:].detach().clone())
        ctx.shape, ctx.count = inputs.shape, count
        ctx.input_dtype, ctx.gate_dtype = inputs.dtype, gate.dtype
        ctx.eps, ctx.activation = eps, activation
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        inputs, gate, weight, bias, rstd = ctx.saved_tensors
        dx, dg, _, _, _ = _fla_norm_ops().layer_norm_gated_bwd(
            dy=gradient[-ctx.count:].contiguous(), x=inputs, g=gate,
            weight=weight, bias=bias, activation=ctx.activation, eps=ctx.eps,
            mean=None, rstd=rstd, dresidual=None, has_residual=False,
            is_rms_norm=True, x_dtype=ctx.input_dtype)
        gradients = []
        for needed, tail, dtype in zip(ctx.needs_input_grad[:2], (dx, dg),
                                       (ctx.input_dtype, ctx.gate_dtype)):
            if not needed:
                gradients.append(None)
                continue
            full = torch.zeros(ctx.shape, device=tail.device, dtype=dtype)
            full[-ctx.count:] = tail
            gradients.append(full)
        return *gradients, None, None, None, None, None


def frozen_query_gated_norm(inputs, gate, weight, bias, eps, activation, count):
    return _FrozenQueryFusedNorm.apply(inputs, gate, weight, bias, eps, activation, count)
