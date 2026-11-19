"""Explicit frozen causal Qwen modules for suffix-query differentiation.

The native decoder/parameter tree and full forward are retained. Only the
appended query rows have input derivatives propagated between decoder layers.
This backend is deliberately limited to unpadded B1, no cache, frozen weights,
and an ordinary causal prefix. It is installed before FSDP owns the parameters.
"""
from __future__ import annotations

from functools import partial

import torch
from torch import nn
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention, Qwen3_5MoeExperts, Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeMLP, Qwen3_5MoeRMSNorm, Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeTopKRouter, apply_rotary_pos_emb,
)

from sft.frozen_expert_grouped_mm import _SuffixExperts, _SuffixLinear
from sft.frozen_query_pointwise import (
    FrozenQueryRouter, FrozenQuerySharedMLP, FrozenQuerySparseMoeBlock,
)


def _tail(value, count, axis):
    return value.narrow(axis, value.shape[axis] - count, count)


class _SuffixPointwise(torch.autograd.Function):
    """Run the original pointwise forward; retain and differentiate only tails."""

    @staticmethod
    def forward(ctx, original, count, axis, *inputs):
        if any(x.shape[axis] < count for x in inputs):
            raise ValueError('query tail exceeds pointwise input')
        ctx.original, ctx.count, ctx.axis = original, count, axis
        ctx.shapes = [x.shape for x in inputs]
        ctx.autocast = torch.is_autocast_enabled('cuda')
        ctx.dtype = torch.get_autocast_dtype('cuda')
        # A view would retain the entire prefix allocation until backward.
        ctx.save_for_backward(*[_tail(x, count, axis).detach().clone() for x in inputs])
        return original(*inputs)

    @staticmethod
    def backward(ctx, gradient):
        with torch.enable_grad(), torch.autocast('cuda', enabled=ctx.autocast, dtype=ctx.dtype):
            inputs = [x.detach().requires_grad_(True) for x in ctx.saved_tensors]
            output = ctx.original(*inputs)
            gradients = torch.autograd.grad(output, inputs, _tail(gradient, ctx.count, ctx.axis))
        result = []
        for shape, grad in zip(ctx.shapes, gradients):
            full = grad.new_zeros(shape)
            _tail(full, ctx.count, ctx.axis).copy_(grad)
            result.append(full)
        return None, None, None, *result


class _SuffixCausalConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias, original, activation, count):
        if weight.requires_grad or (bias is not None and bias.requires_grad):
            raise ValueError('causal convolution weights must be frozen')
        ctx.original, ctx.activation, ctx.count = original, activation, count
        ctx.shape = inputs.shape
        ctx.autocast = torch.is_autocast_enabled('cuda')
        ctx.dtype = torch.get_autocast_dtype('cuda')
        ctx.save_for_backward(inputs[..., -(count + weight.shape[-1] - 1):].detach().clone(),
                              weight, bias)
        return original(x=inputs, weight=weight, bias=bias, activation=activation, seq_idx=None)

    @staticmethod
    def backward(ctx, gradient):
        tail, weight, bias = ctx.saved_tensors
        with torch.enable_grad(), torch.autocast('cuda', enabled=ctx.autocast, dtype=ctx.dtype):
            tail = tail.detach().requires_grad_(True)
            output = ctx.original(x=tail, weight=weight, bias=bias,
                                  activation=ctx.activation, seq_idx=None)
            grad, = torch.autograd.grad(output[..., -ctx.count:], tail,
                                       gradient[..., -ctx.count:].contiguous())
        full = grad.new_zeros(ctx.shape)
        full[..., -ctx.count:] = grad[..., -ctx.count:]
        return full, None, None, None, None, None


class FrozenQueryLinear(nn.Linear):
    def __init__(self, source, count):
        nn.Module.__init__(self)
        self.in_features, self.out_features = source.in_features, source.out_features
        self.weight, self.bias, self.query_count = source.weight, source.bias, count

    def _native(self, inputs):
        return nn.Linear.forward(self, inputs)

    def forward(self, inputs):
        return _SuffixLinear.apply(inputs, self.weight, self.bias, self._native, self.query_count)


class FrozenQueryExperts(Qwen3_5MoeExperts):
    def __init__(self, source, count):
        nn.Module.__init__(self)
        for name in ('num_experts', 'hidden_dim', 'intermediate_dim', 'act_fn',
                     'gate_up_proj', 'down_proj', 'config', 'has_gate', 'has_bias', 'is_transposed'):
            setattr(self, name, getattr(source, name))
        self.query_count = count

    def _native(self, inputs, indices, routing):
        return Qwen3_5MoeExperts.forward(self, inputs, indices, routing)

    def forward(self, inputs, indices, routing):
        return _SuffixExperts.apply(inputs, indices, routing, self._native,
                                    self.query_count, None, True)


class FrozenQueryRMSNorm(Qwen3_5MoeRMSNorm):
    def __init__(self, source, count):
        nn.Module.__init__(self)
        self.eps, self.weight, self.query_count = source.eps, source.weight, count

    def _native(self, inputs):
        return Qwen3_5MoeRMSNorm.forward(self, inputs)

    def forward(self, inputs):
        return _SuffixPointwise.apply(self._native, self.query_count, 1, inputs)


class FrozenQueryGatedNorm(nn.Module):
    def __init__(self, source, count):
        super().__init__()
        if type(source).__name__ not in ('FusedRMSNormGated', 'Qwen3_5MoeRMSNormGated'):
            raise TypeError(f'unsupported GDN norm: {type(source).__name__}')
        self.weight = source.weight
        for name in ('bias', 'eps', 'activation', 'hidden_size', 'elementwise_affine', 'variance_epsilon'):
            if hasattr(source, name):
                setattr(self, name, getattr(source, name))
        self._native_type, self.query_rows = type(source), count

    def _native(self, inputs, gate):
        return self._native_type.forward(self, inputs, gate)

    def forward(self, inputs, gate):
        if self._native_type.__name__ == 'FusedRMSNormGated':
            from sft.frozen_query_norm import frozen_query_gated_norm
            return frozen_query_gated_norm(inputs, gate, self.weight, self.bias,
                                           self.eps, self.activation, self.query_rows)
        return _SuffixPointwise.apply(self._native, self.query_rows, 0, inputs, gate)


class FrozenQueryGatedDeltaNet(Qwen3_5MoeGatedDeltaNet):
    def __init__(self, source, count):
        nn.Module.__init__(self)
        for name in ('hidden_size', 'num_v_heads', 'num_k_heads', 'head_k_dim',
                     'head_v_dim', 'key_dim', 'value_dim', 'conv_kernel_size',
                     'layer_idx', 'activation', 'act', 'layer_norm_epsilon', 'conv_dim',
                     'dt_bias', 'A_log', 'conv1d', 'out_proj', 'in_proj_qkv',
                     'in_proj_z', 'in_proj_b', 'in_proj_a'):
            setattr(self, name, getattr(source, name))
        self.query_count = count
        self._native_conv = source.causal_conv1d_fn or self._torch_convolve
        self.norm = FrozenQueryGatedNorm(source.norm, count * self.num_v_heads)
        from sft.frozen_query_gdn import frozen_query_gated_delta_rule
        self.chunk_gated_delta_rule = partial(frozen_query_gated_delta_rule, query_count=count)
        self.causal_conv1d_fn = self._convolve

    def _torch_convolve(self, *, x, weight, bias, activation, seq_idx):
        # The production image has no causal_conv1d extension. Preserve HF's
        # exact native Conv1d + SiLU forward rather than installing another op.
        return torch.nn.functional.silu(self.conv1d(x)[..., :x.shape[-1]])

    def _convolve(self, *, x, weight, bias, activation, seq_idx):
        if seq_idx is not None:
            raise ValueError('query-only GDN does not support packed sequences')
        return _SuffixCausalConv.apply(x, weight, bias, self._native_conv, activation, self.query_count)

    def forward(self, hidden_states, cache_params=None, attention_mask=None):
        if cache_params is not None or hidden_states.shape[0] != 1:
            raise ValueError('query-only GDN requires B1 without recurrent cache')
        return Qwen3_5MoeGatedDeltaNet.forward(self, hidden_states, None, attention_mask)


def _gated_attention(inputs, gate):
    return inputs * torch.sigmoid(gate)


class FrozenQueryAttention(Qwen3_5MoeAttention):
    def __init__(self, source, count):
        nn.Module.__init__(self)
        for name in ('config', 'layer_idx', 'head_dim', 'num_key_value_groups',
                     'scaling', 'attention_dropout', 'is_causal', 'q_proj', 'k_proj',
                     'v_proj', 'o_proj', 'q_norm', 'k_norm'):
            setattr(self, name, getattr(source, name))
        self.query_count = count
        if self.attention_dropout != 0:
            raise ValueError('query-only attention requires dropout=0')
        if self.config._attn_implementation != 'sdpa':
            raise ValueError('query-only attention is verified against the native SDPA backend')

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kwargs):
        if hidden_states.shape[0] != 1 or past_key_values is not None:
            raise ValueError('query-only attention requires B1 without KV cache')
        if kwargs.get('output_attentions', False):
            raise ValueError('query-only attention does not return attention matrices')
        shape = hidden_states.shape[:-1]
        head_shape = (*shape, -1, self.head_dim)
        queries, gate = torch.chunk(self.q_proj(hidden_states).view(
            *shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*shape, -1)
        queries = self.q_norm(queries.view(head_shape)).transpose(1, 2)
        keys = self.k_norm(self.k_proj(hidden_states).view(head_shape)).transpose(1, 2)
        values = self.v_proj(hidden_states).view(head_shape).transpose(1, 2)
        queries, keys = apply_rotary_pos_emb(queries, keys, *position_embeddings)
        from sft.frozen_query_attention import frozen_query_sdpa
        output = frozen_query_sdpa(queries, keys, values, self.query_count,
                                   attention_mask=attention_mask, scale=self.scaling)
        output = output.transpose(1, 2).reshape(*shape, -1).contiguous()
        output = _SuffixPointwise.apply(_gated_attention, self.query_count, 1, output, gate)
        return self.o_proj(output), None


def install_frozen_query_backbone(backbone, query_count):
    """Replace modules once before FSDP, preserving every parameter key/object."""
    if query_count <= 0 or any(p.requires_grad for p in backbone.parameters()):
        raise ValueError('query-only backbone requires frozen weights and positive query count')
    if getattr(backbone, '_frozen_query_backend', None) is not None:
        raise ValueError('query-only backbone is already installed')
    before = {name: id(p) for name, p in backbone.named_parameters()}
    counts = {}
    replacements = {
        nn.Linear: FrozenQueryLinear,
        Qwen3_5MoeExperts: FrozenQueryExperts,
        Qwen3_5MoeRMSNorm: FrozenQueryRMSNorm,
        Qwen3_5MoeGatedDeltaNet: FrozenQueryGatedDeltaNet,
        Qwen3_5MoeAttention: FrozenQueryAttention,
        Qwen3_5MoeTopKRouter: FrozenQueryRouter,
        Qwen3_5MoeMLP: FrozenQuerySharedMLP,
        Qwen3_5MoeSparseMoeBlock: FrozenQuerySparseMoeBlock,
    }

    def replace(parent):
        for name, child in list(parent.named_children()):
            target = replacements.get(type(child))
            if target is not None:
                native_name = type(child).__name__
                was_training = child.training
                child = target(child, query_count)
                child.train(was_training)
                setattr(parent, name, child)
                counts[native_name] = counts.get(native_name, 0) + 1
            replace(child)

    replace(backbone)
    after = {name: id(p) for name, p in backbone.named_parameters()}
    if before != after:
        raise RuntimeError('query-only backend changed parameter names or identity')
    backbone._frozen_query_backend = {'name': 'query_only', 'query_count': query_count,
                                      'modules': counts, 'scope': 'unpadded_b1_frozen_causal'}
    return backbone._frozen_query_backend
