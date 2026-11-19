"""Validated frozen-query suffix VJPs and optional experimental grouped GEMM.

Only frozen expert weights are supported: backward computes input gradients,
which are needed to train latent queries through frozen Qwen. No quantization.
Suffix VJPs are enabled by the v4.3 suffix_v1 launch profile. The alternative
grouped GEMM kernels remain diagnostic-only.
"""
import torch
from contextlib import contextmanager
import types
from contextvars import ContextVar

_QUERY_ROWS = ContextVar("action_head_query_rows", default=None)


@contextmanager
def query_token_indices(indices):
    """Select all segment suffixes; each autograd node saves its own layout."""
    token = _QUERY_ROWS.set(indices)
    try:
        yield
    finally:
        _QUERY_ROWS.reset(token)



class _SuffixLinear(torch.autograd.Function):
    """Original full forward; only the appended queries need input gradients."""

    @staticmethod
    def forward(ctx, inputs, weight, bias, original, count):
        if weight.requires_grad or (bias is not None and bias.requires_grad):
            raise ValueError('suffix backward requires frozen linear weights')
        if inputs.ndim not in (2, 3) or (inputs.ndim == 3 and inputs.shape[0] != 1):
            raise ValueError('suffix diagnostic requires unpadded batch-one tokens')
        rows = _QUERY_ROWS.get()
        ctx.save_for_backward(weight, rows)
        ctx.count = count
        ctx.shape, ctx.input_dtype = inputs.shape, inputs.dtype
        return original(inputs)

    @staticmethod
    def backward(ctx, gradient):
        weight, rows = ctx.saved_tensors
        tail = gradient[..., -ctx.count:, :] if rows is None else gradient.index_select(-2, rows)
        # Match autocast Linear's operand precision, then restore input dtype.
        value = tail @ weight.to(tail.dtype)
        result = torch.zeros(ctx.shape, device=value.device, dtype=ctx.input_dtype)
        if rows is None:
            result[..., -ctx.count:, :] = value
        else:
            result.index_copy_(-2, rows, value)
        return result, None, None, None, None


class _SuffixExperts(torch.autograd.Function):
    """Keep full MoE forward and its exact selected routes/weights.

    The VJP recomputes only query tokens through the existing HF experts path.
    This avoids changing forward routing or routing-weight placement.
    """

    @staticmethod
    def forward(ctx, inputs, indices, routing_weights, original, count, backward_original,
                own_saved_tail=False):
        rows = _QUERY_ROWS.get()
        def select(x):
            tail = x[-count:] if rows is None else x.index_select(0, rows)
            # The explicit query backend must own its saved suffix storage;
            # otherwise a 16-row view pins the complete long-prefix tensor.
            return tail.clone() if own_saved_tail else tail
        ctx.count = count
        ctx.save_for_backward(select(inputs).detach(), select(indices), select(routing_weights).detach(), rows)
        ctx.original = backward_original or original
        ctx.input_shape, ctx.routing_shape = inputs.shape, routing_weights.shape
        ctx.autocast = torch.is_autocast_enabled('cuda')
        ctx.autocast_dtype = torch.get_autocast_dtype('cuda')
        return original(inputs, indices, routing_weights)

    @staticmethod
    def backward(ctx, gradient):
        inputs, indices, routing, rows = ctx.saved_tensors
        with torch.enable_grad(), torch.autocast('cuda', dtype=ctx.autocast_dtype, enabled=ctx.autocast):
            inputs = inputs.detach().requires_grad_(True)
            routing = routing.detach().requires_grad_(True)
            output = ctx.original(inputs, indices, routing)
            input_grad, routing_grad = torch.autograd.grad(output, (inputs, routing), gradient[-ctx.count:] if rows is None else gradient.index_select(0, rows))
        full_input = input_grad.new_zeros(ctx.input_shape)
        full_routing = routing_grad.new_zeros(ctx.routing_shape)
        if rows is None:
            full_input[-ctx.count:] = input_grad
            full_routing[-ctx.count:] = routing_grad
        else:
            full_input.index_copy_(0, rows, input_grad)
            full_routing.index_copy_(0, rows, routing_grad)
        result = (full_input, None, full_routing, None, None, None, None)
        return result[:len(ctx.needs_input_grad)]


def veomni_experts_forward(module, inputs, indices, routing):
    """Diagnostic merged MoE adapter; preserves routes, casts scores to BF16.

    The installed VeOmni kernel applies scores before the down projection,
    unlike HF. Same real-arithmetic equation, different floating-point path.
    """
    from veomni.ops.kernels.moe.group_gemm import MergedFc1TritonFusedMoeExpertFunction
    return MergedFc1TritonFusedMoeExpertFunction.apply(
        module.num_experts, routing.to(inputs.dtype), indices, inputs,
        module.gate_up_proj, module.down_proj)


@contextmanager
def query_suffix_backward(backbone, query_count, expert_backward='hf'):
    """VJP pruning for a frozen causal backbone, B1 unpadded tail queries.

    Prefix activations do not depend on the appended trainable queries. Their
    VJPs can be discarded at token-wise operations without changing the query
    derivative mathematically. Attention still computes its normal backward.
    Do not apply to LoRA, trainable prefix tokens, or bidirectional models.
    """
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise ValueError('query suffix backward requires a fully frozen backbone')
    if expert_backward not in ('hf', 'veomni'):
        raise ValueError('unknown suffix expert backward backend')
    originals = []
    try:
        for module in backbone.modules():
            original = module.forward
            if isinstance(module, torch.nn.Linear):
                def linear(self, inputs, fn=original):
                    return _SuffixLinear.apply(inputs, self.weight, self.bias, fn, query_count)
                module.forward = types.MethodType(linear, module)
            elif type(module).__name__ == 'Qwen3_5MoeExperts':
                def experts(self, inputs, indices, routing, fn=original):
                    recompute = (lambda x, i, r: veomni_experts_forward(self, x, i, r)) if expert_backward == 'veomni' else None
                    return _SuffixExperts.apply(inputs, indices, routing, fn, query_count, recompute)
                module.forward = types.MethodType(experts, module)
            else:
                continue
            originals.append((module, original))
        yield
    finally:
        for module, original in originals:
            module.forward = original


def _multiply(inputs, weights_nt, counts, backend='deep_gemm'):
    if backend == 'veomni':
        from veomni.ops.kernels.moe._kernels.kernel.group_gemm import group_gemm_same_nk
        # Preserve HF routing and weighting placement. Reuse only the GEMM,
        # not VeOmni's full MoE, whose BF16 routing-weight placement differs.
        return group_gemm_same_nk(
            a=inputs.contiguous(), b=weights_nt.contiguous(),
            cumsum_M=counts.cumsum(0).to(torch.int32),
            max_M=int(counts.max().item()), transpose_b=True)
    if backend != 'deep_gemm':
        raise ValueError(backend)
    from deep_gemm import m_grouped_gemm_bf16_bf16_bf16_nt_nopad
    inputs = inputs.contiguous()
    output = inputs.new_empty((inputs.shape[0], weights_nt.shape[1]))
    # With counts supplied the kernel uses counts, not row IDs; its public
    # wrapper still requires an int32 tensor with one entry per input row.
    row_ids = torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device, dtype=torch.int32),
        counts, output_size=inputs.shape[0])
    m_grouped_gemm_bf16_bf16_bf16_nt_nopad(
        inputs, weights_nt.contiguous(), output, row_ids, m_rows=counts)
    return output


class _FrozenGroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, offsets, backend):
        if weights.requires_grad:
            raise ValueError('PPU expert grouped GEMM requires frozen weights')
        if inputs.dtype != torch.bfloat16 or weights.dtype != torch.bfloat16:
            raise ValueError('PPU expert grouped GEMM only supports BF16')
        counts = torch.diff(offsets, prepend=offsets.new_zeros(1)).to(torch.int32)
        ctx.save_for_backward(weights, counts)
        ctx.backend = backend
        return _multiply(inputs, weights.transpose(-2, -1), counts, backend)

    @staticmethod
    def backward(ctx, gradient):
        weights, counts = ctx.saved_tensors
        return _multiply(gradient, weights, counts, ctx.backend), None, None, None


def frozen_grouped_mm(inputs, weights, *, offs):
    return _FrozenGroupedMM.apply(inputs, weights, offs, 'deep_gemm')


def veomni_grouped_mm(inputs, weights, *, offs):
    return _FrozenGroupedMM.apply(inputs, weights, offs, 'veomni')


def padded_grouped_mm(inputs, weights, *, offs, stats=None):
    """Reuse torch.bmm on padded expert groups without expanding expert weights."""
    if weights.requires_grad:
        raise ValueError('expert BMM probe requires frozen weights')
    counts = torch.diff(offs, prepend=offs.new_zeros(1)).long()
    capacity = int(counts.max().item())
    if stats is not None:
        stats['calls'] = stats.get('calls', 0) + 1
        stats['max_padding_ratio'] = max(stats.get('max_padding_ratio', 0),
                                        capacity * weights.shape[0] / max(1, inputs.shape[0]))
    if capacity * weights.shape[0] > 4 * inputs.shape[0]:
        if stats is not None:
            stats['fallbacks'] = stats.get('fallbacks', 0) + 1
        return torch._grouped_mm(inputs, weights, offs=offs)
    if not capacity:
        return inputs @ weights[0]
    expert = torch.repeat_interleave(torch.arange(weights.shape[0], device=offs.device),
                                    counts, output_size=inputs.shape[0])
    starts = offs.long() - counts
    slots = expert * capacity + torch.arange(inputs.shape[0], device=offs.device) - starts[expert]
    padded = inputs.new_zeros(weights.shape[0] * capacity, inputs.shape[1])
    padded = padded.index_copy(0, slots, inputs).view(weights.shape[0], capacity, inputs.shape[1])
    return torch.bmm(padded, weights).reshape(-1, weights.shape[-1])[slots]
