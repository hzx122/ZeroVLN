"""Full FLA forward with a VJP restricted to trainable suffix tokens.

This is an operator for a frozen, causal backbone, not a general GDN gradient.
Only the final ``query_count`` inputs are differentiable. Prefix inputs must
be independent of every optimized parameter. The caller must enforce this
contract at the backbone boundary; requires_grad on a concatenated tensor
cannot express it.

The installed FLA forward body and all kernels remain unchanged. A private
function binding captures the original tail chunk states and corrected values.
Backward runs the installed FLA backward body on those same aligned chunks,
substituting the captured states for its usual full-prefix recomputation.
Saving every tail chunk is essential: FLA stores h in BF16 but accumulates the
recurrence in FP32, so restarting from a single BF16 h can change the next chunk.
No installed module globals or package files are modified.

Scope: FLA 0.5.0, B=1, contiguous suffix, no cache, packed/CP layout, fused gate,
or higher-order differentiation. Unsupported contracts raise explicitly.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib
from importlib import metadata
from pathlib import Path
from types import FunctionType

import torch
from torch.autograd.function import once_differentiable


CHUNK_SIZE = 64
EXPECTED_FLA_VERSION = "0.5.0"
EXPECTED_SOURCE_SHA256 = "2bce57074e4bfb11660622d41178d264aa2da9eed7e63717e8d59bd90c13ee5f"


def aligned_tail_start(length: int, query_count: int) -> int:
    if not isinstance(query_count, int) or isinstance(query_count, bool):
        raise ValueError("query_count must be an integer")
    if not 0 < query_count <= length:
        raise ValueError("query_count must be positive and no larger than the sequence")
    return ((length - query_count) // CHUNK_SIZE) * CHUNK_SIZE


@lru_cache(maxsize=1)
def _fla_ops():
    if metadata.version("fla-core") != EXPECTED_FLA_VERSION:
        raise RuntimeError("Frozen-query GDN requires validated fla-core 0.5.0")
    module = importlib.import_module("fla.ops.gated_delta_rule.chunk")
    digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if digest != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Frozen-query GDN FLA source SHA256 differs from validated source")
    return module


def _bind_state_function(function, replacement):
    """Reuse the exact installed Python body without changing its globals."""
    if not isinstance(function, FunctionType):
        raise RuntimeError("Expected the source-bound FLA Python orchestration function")
    namespace = dict(function.__globals__)
    namespace["chunk_gated_delta_rule_fwd_h"] = replacement
    bound = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    return bound


def _owned_tail(value, start):
    # contiguous() alone may retain the complete underlying prefix allocation.
    return None if value is None else value[:, start:].clone(memory_format=torch.contiguous_format)


class _FrozenQueryGDN(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, q, k, v, g, beta, query_count, scale, normalize, transpose):
        ops = _fla_ops()
        length = q.shape[1]
        start = aligned_tail_start(length, query_count)
        qr, kr = None, None
        if normalize:
            q, qr = ops.l2norm_fwd(q)
            k, kr = ops.l2norm_fwd(k)
        captured = []

        def capture_state(**kwargs):
            h, corrected_value, final_state = ops.chunk_gated_delta_rule_fwd_h(**kwargs)
            captured.append((_owned_tail(h, start // CHUNK_SIZE),
                             _owned_tail(corrected_value, start)))
            return h, corrected_value, final_state

        full_forward = _bind_state_function(ops.chunk_gated_delta_rule_fwd, capture_state)
        gates, output, auxiliary, final_state, _, _ = full_forward(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            initial_state=None, output_final_state=False,
            transpose_state_layout=transpose)
        if final_state is not None or len(captured) != 1:
            raise RuntimeError("Unexpected state production in source-bound FLA forward")
        h, corrected_value = captured[0]
        ctx.save_for_backward(*[_owned_tail(x, start) for x in
                                (q, qr, k, kr, v, gates, beta, auxiliary)],
                              h, corrected_value)
        ctx.length, ctx.start, ctx.query_count = length, start, query_count
        ctx.scale, ctx.normalize, ctx.transpose = scale, normalize, transpose
        ctx.input_shapes = (q.shape, k.shape, v.shape, g.shape, beta.shape)
        ctx.input_dtypes = (q.dtype, k.dtype, v.dtype, g.dtype, beta.dtype)
        return output.to(q.dtype)

    @staticmethod
    @once_differentiable
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        ops = _fla_ops()
        q, qr, k, kr, v, gates, beta, auxiliary, h, corrected_value = ctx.saved_tensors
        # Prefix outputs cannot depend on the trainable suffix. Their incoming
        # gradients are irrelevant even if downstream attention produced them.
        do = grad_output[:, ctx.start:].contiguous().clone()
        prefix_in_tail = do.shape[1] - ctx.query_count
        if prefix_in_tail:
            do[:, :prefix_in_tail] = 0

        def retained_state(**kwargs):
            if kwargs["k"].shape != k.shape or kwargs.get("output_final_state"):
                raise RuntimeError("Unexpected FLA backward state reconstruction contract")
            return h, corrected_value, None

        tail_backward = _bind_state_function(ops.chunk_gated_delta_rule_bwd, retained_state)
        dq, dk, dv, db, dg, _, _, _ = tail_backward(
            q=q, k=k, v=v, g=gates, beta=beta, A=auxiliary,
            scale=ctx.scale, initial_state=None, do=do, dht=None,
            transpose_state_layout=ctx.transpose)
        if ctx.normalize:
            dq = ops.l2norm_bwd(q, qr, dq)
            dk = ops.l2norm_bwd(k, kr, dk)
        gradients = []
        for needed, tail, shape, dtype in zip(ctx.needs_input_grad[:5],
                                             (dq, dk, dv, dg, db),
                                             ctx.input_shapes, ctx.input_dtypes):
            if not needed:
                gradients.append(None)
                continue
            full = torch.zeros(shape, dtype=dtype, device=tail.device)
            full[:, -ctx.query_count:] = tail[:, -ctx.query_count:]
            gradients.append(full)
        return (*gradients, None, None, None, None)


def frozen_query_gated_delta_rule(
    q, k, v, g, beta, *, query_count=16, scale=None,
    initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=False, transpose_state_layout=False,
    cu_seqlens=None, cu_seqlens_cpu=None, cp_context=None, **kwargs,
):
    """Return ``(full_output, None)`` and differentiate only suffix inputs.

    Identical tensor layout and scaling to FLA's chunk_gated_delta_rule.
    This deliberately does not accept a recurrent cache or a final-state loss.
    Every query token's q/k/v/g/beta derivatives remain present, including
    query-to-query recurrence. Only derivatives of frozen prefix inputs vanish.
    """
    if initial_state is not None or output_final_state:
        raise ValueError("Frozen-query GDN does not support cache states/final-state outputs")
    if cu_seqlens is not None or cu_seqlens_cpu is not None or cp_context is not None:
        raise ValueError("Frozen-query GDN requires one unpadded sequence, without packed/CP layout")
    allowed_defaults = {"use_gate_in_kernel": False, "A_log": None, "dt_bias": None}
    for name, value in kwargs.items():
        if name not in allowed_defaults or value != allowed_defaults[name]:
            raise ValueError(f"Unsupported frozen-query GDN argument: {name}")
    values = (q, k, v, g, beta)
    if q.ndim != 4 or q.shape[0] != 1 or k.shape != q.shape:
        raise ValueError("Frozen-query GDN requires matching [1,T,H,K] q/k tensors")
    aligned_tail_start(q.shape[1], query_count)
    if v.ndim != 4 or v.shape[:2] != q.shape[:2]:
        raise ValueError("v must have shape [1,T,HV,V]")
    if v.shape[2] % q.shape[2] or g.shape != v.shape[:3] or beta.shape != g.shape:
        raise ValueError("GDN head counts or gate/beta shapes are incompatible")
    if q.shape[-1] > 256:
        raise ValueError("FLA GDN supports key head dimension at most 256")
    if any(x.device != q.device for x in values):
        raise ValueError("All GDN inputs must be on one device")
    if q.device.type != "cuda":
        raise ValueError("Frozen-query GDN requires a CUDA-compatible device")
    if not all(x.is_floating_point() for x in values):
        raise ValueError("GDN inputs must be floating-point tensors")
    if q.dtype != k.dtype or v.dtype != q.dtype:
        raise ValueError("q/k/v must have the same floating-point dtype")
    scale = k.shape[-1] ** -0.5 if scale is None else float(scale)
    output = _FrozenQueryGDN.apply(
        *(x.contiguous() for x in values), query_count, scale,
        bool(use_qk_l2norm_in_kernel), bool(transpose_state_layout))
    return output, None
