"""Original FlashAttention forward with a frozen-prefix, query-only VJP.

This operator is deliberately narrower than general FlashAttention: one
unpadded causal sequence, no dropout, ALiBi, softcap or sliding window.  The
prefix must not depend on trainable parameters.  Query K/V remain trainable:
later query outputs contribute to earlier query K/V gradients.

The full forward is PyTorch's native SDPA FlashAttention forward, unchanged.
Its suffix output and log-sum-exp are passed to the same installed backward
with Q query rows and L key/value rows.  The PPU kernel's bottom-right causal
alignment gives exactly the original last Q attention rows.  No alternate
forward, KV cache, or recomputed softmax is used.
"""
from __future__ import annotations

from functools import lru_cache

import torch
from torch.autograd.function import once_differentiable


@lru_cache(maxsize=None)
def _check_native_device(device):
    if device.type != 'cuda' or 'PPU' not in torch.cuda.get_device_name(device):
        raise RuntimeError('query-only native SDPA backward is validated only on the PPU backend')


def _native_sdpa_forward(q, k, v, scale):
    # Do not substitute the separately installed flash_attn extension: on this
    # PPU it differs from HF's native SDPA forward even for the same tensors.
    _check_native_device(q.device)
    q, k, v = (x.transpose(0, 1).unsqueeze(0) for x in (q, k, v))
    output, lse, _, _, _, _, rng, unused, _ = (
        torch.ops.aten._scaled_dot_product_flash_attention.default(
            q, k, v, 0.0, True, False, scale=scale))
    return output.squeeze(0).transpose(0, 1), lse.squeeze(0), unused, rng


def _native_sdpa_backward(do, q, k, v, output, lse, scale, rng_state, philox_offset):
    max_q, max_k = q.shape[0], k.shape[0]
    do, q, k, v, output = (x.transpose(0, 1).unsqueeze(0)
                           for x in (do, q, k, v, output))
    gradients = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
        do, q, k, v, output, lse.unsqueeze(0), None, None,
        max_q, max_k, 0.0, True, rng_state, philox_offset, scale=scale)
    return tuple(value.squeeze(0).transpose(0, 1) for value in gradients)


def _flash_functions():
    return _native_sdpa_forward, _native_sdpa_backward


def _validate(q, k, v, query_count):
    if q.ndim not in (3, 4) or k.ndim != q.ndim or v.ndim != q.ndim:
        raise ValueError('expected [T,H,D] or [1,T,H,D] query/key/value tensors')
    if q.ndim == 4 and (q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1):
        raise ValueError('frozen query attention requires batch size one')
    if q.shape[-3] != k.shape[-3] or k.shape != v.shape:
        raise ValueError('full query/key/value sequence lengths must match')
    if (not q.shape[-2] or not k.shape[-2]
            or q.shape[-1] != k.shape[-1] or q.shape[-2] % k.shape[-2]):
        raise ValueError('invalid GQA heads or head dimensions')
    if not isinstance(query_count, int) or not 1 <= query_count <= q.shape[-3]:
        raise ValueError('query_count must be within the full sequence')
    if q.dtype != k.dtype or k.dtype != v.dtype or q.device != k.device or k.device != v.device:
        raise ValueError('query/key/value must share dtype and device')
    if not q.is_floating_point() or q.shape[-1] % 8:
        raise ValueError('floating point tensors and head dimension divisible by eight required')


def _full_forward(ctx, q, k, v, query_count, scale, deterministic):
    _validate(q, k, v, query_count)
    if deterministic and not torch.are_deterministic_algorithms_enabled():
        raise ValueError('native SDPA follows torch deterministic policy; no local deterministic override')
    ctx.batched = q.ndim == 4
    if ctx.batched:
        q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    length = q.shape[0]
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    forward, backward = _flash_functions()
    output, lse, auxiliary, rng = forward(q, k, v, scale)
    # The native adapter returns [heads,total_q]. Reject another layout rather
    # than silently selecting a head dimension as the query sequence.
    if tuple(lse.shape) != (q.shape[1], length):
        raise RuntimeError(f'unsupported native FlashAttention LSE shape: {tuple(lse.shape)}')
    # clone is intentional: a suffix view would keep all Q/O/LSE storage alive.
    ctx.save_for_backward(q[-query_count:].clone(), k, v,
                          output[-query_count:].clone(),
                          lse[:, -query_count:].contiguous().clone(), rng, auxiliary)
    ctx.query_count, ctx.length = query_count, length
    ctx.scale, ctx.deterministic, ctx.kernel_backward = scale, bool(deterministic), backward
    ctx.set_materialize_grads(False)
    return output.unsqueeze(0) if ctx.batched else output


def _suffix_backward(ctx, gradient):
    q, k, v, output, lse, rng, auxiliary = ctx.saved_tensors
    if ctx.batched:
        gradient = gradient.squeeze(0)
    gradient = gradient.contiguous()
    # Unequal lengths use bottom-right alignment, so row i may attend keys
    # j <= L-Q+i. The low-level kernel also sums GQA contributions into K/V.
    gradients = ctx.kernel_backward(gradient, q, k, v, output, lse,
                                    ctx.scale, rng, auxiliary)
    return gradients


class _FrozenQueryAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, query_count, scale, deterministic):
        return _full_forward(ctx, q, k, v, query_count, scale, deterministic)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        if gradient is None:
            return (None,) * 6
        dq, dk, dv = _suffix_backward(ctx, gradient[..., -ctx.query_count:, :, :])
        # Prefix gradients cannot reach the trainable suffix in a frozen
        # causal backbone. Return zeros instead of differentiating constants.
        full_dq = dq.new_zeros((ctx.length, *dq.shape[1:]))
        full_dq[-ctx.query_count:] = dq
        dk[:-ctx.query_count].zero_()
        dv[:-ctx.query_count].zero_()
        if ctx.batched:
            full_dq, dk, dv = (x.unsqueeze(0) for x in (full_dq, dk, dv))
        return full_dq, dk, dv, None, None, None


def frozen_query_flash_attention(q, k, v, query_count, *, softmax_scale=None,
                                 deterministic=False):
    """Return the original full causal output; differentiate only its queries.

    Inputs use [T,H,D] or [1,T,H,D]. The prefix portion of each input is
    treated as constant, including when the full tensor requires gradients.
    Only losses depending on the last ``query_count`` outputs are supported.
    Prefer ``frozen_query_flash_attention_parts`` when prefix and query
    tensors are already separate, to avoid full-length zero input gradients.
    """
    return _FrozenQueryAttention.apply(q, k, v, query_count, softmax_scale, deterministic)


def frozen_query_sdpa(q, k, v, query_count, attention_mask=None, scale=None):
    """HF-compatible [1,H,T,D] causal SDPA with frozen-prefix gradients.

    This narrowly targets the verified native PPU FlashAttention dispatch
    used by HF SDPA with ``enable_gqa=True`` and no attention mask. Explicit
    masks and padded batches must be handled by a separately verified path.
    """
    if attention_mask is not None:
        raise ValueError('frozen_query_sdpa requires an unpadded sequence and attention_mask=None')
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError('frozen_query_sdpa requires [1,H,T,D] tensors')
    output = frozen_query_flash_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), query_count,
        softmax_scale=scale)
    return output.transpose(1, 2)


class _FrozenQueryAttentionParts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_prefix, k_prefix, v_prefix, q_query, k_query, v_query,
                scale, deterministic):
        if any(x.requires_grad for x in (q_prefix, k_prefix, v_prefix)):
            raise ValueError('prefix tensors must be detached in the parts interface')
        count = q_query.shape[-3]
        if (k_query.shape[-3] != count or v_query.shape[-3] != count
                or k_prefix.shape[-3] != q_prefix.shape[-3]
                or v_prefix.shape[-3] != q_prefix.shape[-3]):
            raise ValueError('the three prefix lengths and three query lengths must match')
        q, k, v = (torch.cat((prefix, query), dim=-3) for prefix, query in
                   ((q_prefix, q_query), (k_prefix, k_query), (v_prefix, v_query)))
        output = _full_forward(ctx, q, k, v, count, scale, deterministic)
        prefix_output, query_output = output[..., :-count, :, :], output[..., -count:, :, :]
        ctx.mark_non_differentiable(prefix_output)
        return prefix_output, query_output

    @staticmethod
    @once_differentiable
    def backward(ctx, prefix_gradient, query_gradient):
        if query_gradient is None:
            return (None,) * 8
        dq, dk, dv = _suffix_backward(ctx, query_gradient)
        # Copy the tiny suffixes so downstream autograd cannot retain the
        # temporary full-length dK/dV kernel workspaces.
        dk, dv = dk[-ctx.query_count:].clone(), dv[-ctx.query_count:].clone()
        if ctx.batched:
            dq, dk, dv = (x.unsqueeze(0) for x in (dq, dk, dv))
        return None, None, None, dq, dk, dv, None, None


def frozen_query_flash_attention_parts(q_prefix, k_prefix, v_prefix,
                                       q_query, k_query, v_query, *,
                                       softmax_scale=None, deterministic=False):
    """Return (detached full-prefix output, differentiable query output).

    Prefix inputs must be explicitly detached; the three query tensors retain
    all causal dependencies among query tokens. Layout matches the full API.
    No full-length zero dQ/dK/dV tensors are returned to upstream autograd.
    """
    return _FrozenQueryAttentionParts.apply(q_prefix, k_prefix, v_prefix,
                                            q_query, k_query, v_query,
                                            softmax_scale, deterministic)
