"""Frozen Qwen router/shared-MLP operators with query-only activation graphs.

These modules preserve the original full-sequence forward and all parameter
objects. They are opt-in building blocks, not installed by importing this
module. Only suffix query inputs may depend on trainable parameters; prefix
router auxiliary losses and trainable router/shared-expert weights are outside
this contract.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import once_differentiable
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeMLP, Qwen3_5MoeSparseMoeBlock, Qwen3_5MoeTopKRouter,
)


def _validate_source(source, count):
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError('query_count must be a positive integer')
    if any(parameter.requires_grad for parameter in source.parameters()):
        raise ValueError('frozen query pointwise modules require frozen weights')


def _validate_inputs(inputs, count):
    if inputs.ndim not in (2, 3) or (inputs.ndim == 3 and inputs.shape[0] != 1):
        raise ValueError('query pointwise modules require [T,D] or [1,T,D]')
    if inputs.shape[-2] < count:
        raise ValueError('query_count exceeds input sequence length')


class _FrozenQueryRouterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, top_k, count):
        _validate_inputs(inputs, count)
        if weight.requires_grad:
            raise ValueError('query router weight must be frozen')
        if inputs.shape[-1] != weight.shape[-1] or not 1 <= top_k <= weight.shape[0]:
            raise ValueError('router dimensions or top_k are inconsistent')
        flat = inputs.reshape(-1, weight.shape[-1])
        # This is the original HF operation sequence, including FP32 softmax
        # and the in-place normalization of topk values. Do not recompute the
        # projections or choose routes on just the query rows.
        logits = F.linear(flat, weight)
        probabilities = F.softmax(logits, dtype=torch.float, dim=-1)
        scores, indices = torch.topk(probabilities, top_k, dim=-1)
        scores /= scores.sum(dim=-1, keepdim=True)
        scores = scores.to(probabilities.dtype)
        ctx.save_for_backward(logits[-count:].detach().clone(),
                              indices[-count:].clone(), weight)
        ctx.input_shape, ctx.input_dtype, ctx.count = inputs.shape, inputs.dtype, count
        ctx.autocast = torch.is_autocast_enabled('cuda')
        ctx.autocast_dtype = torch.get_autocast_dtype('cuda')
        ctx.mark_non_differentiable(indices)
        ctx.set_materialize_grads(False)
        return probabilities, scores, indices

    @staticmethod
    @once_differentiable
    def backward(ctx, probability_gradient, score_gradient, index_gradient):
        if probability_gradient is None and score_gradient is None:
            return None, None, None, None
        logits, indices, weight = ctx.saved_tensors
        with torch.enable_grad(), torch.autocast('cuda', enabled=ctx.autocast,
                                                dtype=ctx.autocast_dtype):
            logits = logits.detach().requires_grad_(True)
            probabilities = F.softmax(logits, dtype=torch.float, dim=-1)
            # Saved original indices preserve every route even at a topk tie.
            selected = probabilities.gather(-1, indices)
            scores = selected / selected.sum(dim=-1, keepdim=True)
            outputs, gradients = [], []
            if probability_gradient is not None:
                outputs.append(probabilities)
                gradients.append(probability_gradient[-ctx.count:])
            if score_gradient is not None:
                outputs.append(scores)
                gradients.append(score_gradient[-ctx.count:])
            logit_gradient, = torch.autograd.grad(outputs, logits, gradients)
            query_gradient = logit_gradient @ weight.to(logit_gradient.dtype)
        full = torch.zeros(ctx.input_shape, device=query_gradient.device, dtype=ctx.input_dtype)
        full.reshape(-1, full.shape[-1])[-ctx.count:] = query_gradient.to(ctx.input_dtype)
        return full, None, None, None


class FrozenQueryRouter(Qwen3_5MoeTopKRouter):
    """Original router output triplet with differentiable query rows only."""
    def __init__(self, source, query_count):
        nn.Module.__init__(self)
        _validate_source(source, query_count)
        for name in ('top_k', 'num_experts', 'hidden_dim', 'weight'):
            setattr(self, name, getattr(source, name))
        self.query_count, self.training = query_count, source.training

    def forward(self, hidden_states):
        return _FrozenQueryRouterFunction.apply(hidden_states, self.weight,
                                                self.top_k, self.query_count)


class FrozenQuerySharedMLP(Qwen3_5MoeMLP):
    """Keep each full projection; retain only query gate/up activations."""
    def __init__(self, source, query_count):
        nn.Module.__init__(self)
        _validate_source(source, query_count)
        for name in ('config', 'hidden_size', 'intermediate_size', 'gate_proj',
                     'up_proj', 'down_proj', 'act_fn'):
            setattr(self, name, getattr(source, name))
        self.query_count, self.training = query_count, source.training

    def _activate_product(self, gate, up):
        return self.act_fn(gate) * up

    def forward(self, inputs):
        _validate_inputs(inputs, self.query_count)
        # Import locally so the explicit backbone installer can later import
        # these classes without a circular dependency.
        from sft.frozen_query_backbone import _SuffixPointwise
        gate, up = self.gate_proj(inputs), self.up_proj(inputs)
        activated = _SuffixPointwise.apply(self._activate_product, self.query_count,
                                           inputs.ndim - 2, gate, up)
        return self.down_proj(activated)


def _shared_expert_product(gate, shared):
    return F.sigmoid(gate) * shared


class FrozenQuerySparseMoeBlock(Qwen3_5MoeSparseMoeBlock):
    """Original MoE order with query-only shared-expert gate multiplication.

    This wrapper owns only the outer pointwise operation. The installer must
    recursively replace its router, shared MLP, experts and linear children;
    constructing this module alone does not optimize their activation graphs.
    """
    def __init__(self, source, query_count):
        nn.Module.__init__(self)
        _validate_source(source, query_count)
        for name in ('gate', 'experts', 'shared_expert', 'shared_expert_gate'):
            setattr(self, name, getattr(source, name))
        self.query_count, self.training = query_count, source.training

    def forward(self, hidden_states):
        cache = getattr(self, '_prefix_reuse', None)
        if cache is not None:
            return cache.run(self._prefix_reuse_key, self._forward_uncached,
                             hidden_states, self.query_count)
        return self._forward_uncached(hidden_states)

    def _forward_uncached(self, hidden_states):
        _validate_inputs(hidden_states, self.query_count)
        if hidden_states.ndim != 3:
            raise ValueError('sparse MoE block requires [1,T,D] hidden states')
        from sft.frozen_query_backbone import _SuffixPointwise

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flattened = hidden_states.view(-1, hidden_dim)
        # Match the original full forward, including its module call order.
        shared = self.shared_expert(flattened)
        _, routing_weights, selected_experts = self.gate(flattened)
        experts = self.experts(flattened, selected_experts, routing_weights)
        gate = self.shared_expert_gate(flattened)
        shared = _SuffixPointwise.apply(_shared_expert_product, self.query_count,
                                        0, gate, shared)
        return (experts + shared).reshape(batch_size, sequence_length, hidden_dim)
