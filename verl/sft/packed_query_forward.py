"""Unpadded multi-anchor dispatch for the frozen HF Qwen3.5-MoE backbone.

Default layerwise execution shares each decoder's FSDP gather while preserving
all historical per-anchor operator shapes. Explicit varlen execution batches
projections/experts and uses FLA cu_seqlens for GDN; BF16 parity is weaker.
Images remain live inputs. Checkpoint recomputation is not yet supported.
"""
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import accumulate
import importlib
import os
import types

import torch

from sft.frozen_expert_grouped_mm import query_token_indices

_LAYOUT = ContextVar('action_head_packed_layout', default=None)


def install_conv_dynamic_nb():
    """Reuse the audited unused-argument patch; preserve all convolution math."""
    from sft.fla_unused_shape_dynamic import ModulePatch, create_patched_kernels
    module = importlib.import_module('fla.modules.conv.triton.kernels')
    consumer = importlib.import_module('fla.modules.conv.triton.ops')
    spec = ModulePatch(module_name=module.__name__,
        source_sha256='553024cf1f91415dd7828dbd49122887b975bccb5e540a51f76f122f7cb99ea7',
        kernel_arguments=(('causal_conv1d_fwd_kernel', 'NB'), ('causal_conv1d_bwd_kernel', 'NB')),
        wrapper_names=())
    installed = getattr(module, '_action_head_conv_dynamic_nb', None)
    if installed is not None:
        for name, kernel in installed['kernels'].items():
            if getattr(module, name) is not kernel or getattr(consumer, name) is not kernel:
                raise RuntimeError('packed convolution kernel was replaced after installation')
        return installed['audit']
    kernels, audit = create_patched_kernels(module, spec)
    for name in kernels:
        if getattr(module, name) is not getattr(consumer, name):
            raise RuntimeError('unexpected FLA convolution consumer kernel binding')
    module.__dict__.update(kernels)
    consumer.__dict__.update(kernels)
    module._action_head_conv_dynamic_nb = {'kernels': kernels, 'audit': audit}
    return audit


@dataclass(frozen=True)
class PackedLayout:
    boundaries: tuple[int, ...]
    cu_seqlens: torch.Tensor
    query_indices: torch.Tensor

    @classmethod
    def build(cls, lengths, query_count, device):
        if not lengths or query_count <= 0 or any(n <= query_count for n in lengths):
            raise ValueError('each packed anchor needs a nonempty prefix and query suffix')
        boundaries = (0, *accumulate(lengths))
        indices = [i for end in boundaries[1:] for i in range(end - query_count, end)]
        return cls(boundaries, torch.tensor(boundaries, device=device, dtype=torch.int32),
                   torch.tensor(indices, device=device, dtype=torch.int64))


def _packed_attention(self, hidden_states, position_embeddings, attention_mask=None,
                      past_key_values=None, **kwargs):
    """HF Qwen attention algebra, with independent causal SDPA per segment."""
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import apply_rotary_pos_emb
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    layout = _LAYOUT.get()
    if past_key_values is not None:
        raise ValueError('packed anchors cannot share a KV cache')
    shape = hidden_states.shape[:-1]
    heads = (*shape, -1, self.head_dim)
    q, gate = self.q_proj(hidden_states).view(*shape, -1, self.head_dim * 2).chunk(2, dim=-1)
    q = self.q_norm(q.reshape(heads)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(heads)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(heads).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
    pieces = [sdpa_attention_forward(self, q[:, :, start:end], k[:, :, start:end],
                                    v[:, :, start:end], None, dropout=0.0,
                                    scaling=self.scaling, is_causal=True)[0]
              for start, end in zip(layout.boundaries, layout.boundaries[1:])]
    value = torch.cat(pieces, dim=1).reshape(*shape, -1)
    return self.o_proj(value * gate.reshape(*shape, -1).sigmoid()), None


def install_packed_forward(backbone):
    """Install dispatch once, before FSDP wrapping; normal calls stay native HF."""
    if getattr(backbone, '_action_head_packed_installed', False):
        return
    if backbone.config.model_type != 'qwen3_5_moe_text':
        raise ValueError('packed adapter is restricted to Qwen3.5 MoE text')
    execution = os.environ.get('ACTION_HEAD_PACK_EXECUTION', 'layerwise')
    if execution not in ('layerwise', 'varlen'):
        raise ValueError('ACTION_HEAD_PACK_EXECUTION must be layerwise or varlen')
    conv_backend = os.environ.get('ACTION_HEAD_PACK_CONV_BACKEND', 'segmented')
    if conv_backend not in ('segmented', 'fla'):
        raise ValueError('ACTION_HEAD_PACK_CONV_BACKEND must be segmented or fla')
    if execution == 'varlen' and conv_backend == 'fla' and os.environ.get('ACTION_HEAD_PACK_CONV_DYNAMIC_NB', '1') == '1':
        backbone._action_head_conv_compile_audit = install_conv_dynamic_nb()
    for layer in backbone.layers:
        if execution == 'layerwise':
            original_layer = layer.forward

            def decoder_dispatch(self, hidden_states, position_embeddings=None,
                                 attention_mask=None, fn=original_layer, **kwargs):
                layout = _LAYOUT.get()
                if layout is None:
                    return fn(hidden_states, position_embeddings=position_embeddings,
                              attention_mask=attention_mask, **kwargs)
                pieces = []
                # The outer decoder __call__ owns one FSDP gather. Calling the
                # original forward here reuses that weight materialization for
                # every segment, preserving the historical operator shapes.
                for start, end in zip(layout.boundaries, layout.boundaries[1:]):
                    token = _LAYOUT.set(None)
                    try:
                        with query_token_indices(None):
                            pieces.append(fn(hidden_states[:, start:end],
                                position_embeddings=tuple(x[..., start:end, :] for x in position_embeddings),
                                attention_mask=None, **kwargs))
                    finally:
                        _LAYOUT.reset(token)
                return torch.cat(pieces, dim=1)

            layer.forward = types.MethodType(decoder_dispatch, layer)
            continue
        linear = getattr(layer, 'linear_attn', None)
        if linear is not None:
            conv, chunk = linear.causal_conv1d_fn, linear.chunk_gated_delta_rule

            def convolution(*, x, weight, bias=None, activation=None, seq_idx=None, fn=conv, module=linear, **kwargs):
                layout = _LAYOUT.get()
                if layout is None:
                    if fn is None:
                        return torch.nn.functional.silu(module.conv1d(x)[:, :, :x.shape[-1]])
                    return fn(x=x, weight=weight, bias=bias, activation=activation, seq_idx=seq_idx, **kwargs)
                if conv_backend == 'segmented':
                    pieces = []
                    for start, end in zip(layout.boundaries, layout.boundaries[1:]):
                        part = x[..., start:end]
                        pieces.append(torch.nn.functional.silu(module.conv1d(part)[:, :, :end-start])
                            if fn is None else fn(x=part, weight=weight, bias=bias,
                                                activation=activation, seq_idx=None, **kwargs))
                    return torch.cat(pieces, dim=-1)
                from fla.modules.convolution import causal_conv1d
                return causal_conv1d(x=x.transpose(1, 2), weight=weight, bias=bias,
                    activation=activation, backend='triton', cu_seqlens=layout.cu_seqlens)[0].transpose(1, 2)

            def delta_rule(*args, fn=chunk, **kwargs):
                layout = _LAYOUT.get()
                if layout is not None:
                    kwargs['cu_seqlens'] = layout.cu_seqlens
                return fn(*args, **kwargs)

            # Varlen GDN requires FLA; ordinary convolution retains its native
            # callable or the original HF torch fallback.
            if 'torch_chunk' in getattr(chunk, '__name__', ''):
                raise RuntimeError('packed anchors require FLA')
            linear.causal_conv1d_fn, linear.chunk_gated_delta_rule = convolution, delta_rule
        else:
            attention = layer.self_attn
            original = attention.forward

            def attention_dispatch(self, *args, fn=original, **kwargs):
                return fn(*args, **kwargs) if _LAYOUT.get() is None else _packed_attention(self, *args, **kwargs)

            attention.forward = types.MethodType(attention_dispatch, attention)

    original = backbone.forward

    def text_dispatch(self, *args, packed_layout=None, **kwargs):
        if packed_layout is None:
            return original(*args, **kwargs)
        if args or kwargs.get('use_cache') or kwargs.get('past_key_values') is not None:
            raise ValueError('packed training requires keyword embeddings and no cache')
        if any(layer.gradient_checkpointing for layer in self.layers):
            raise ValueError('packed layout must not escape into checkpoint recomputation; disable GC')
        if any(p.requires_grad for p in self.parameters()):
            raise ValueError('packed query training requires a frozen backbone')
        hidden = kwargs['inputs_embeds']
        positions = kwargs['position_ids']
        if hidden.shape[0] != 1 or hidden.shape[1] != packed_layout.boundaries[-1]:
            raise ValueError('packed layout does not match embeddings')
        # HF uses the first of four position planes only for mask generation;
        # the remaining three are MRoPE, preserved separately for each anchor.
        if positions.ndim == 3 and positions.shape[0] == 4:
            positions = positions[1:]
        token = _LAYOUT.set(packed_layout)
        try:
            with query_token_indices(packed_layout.query_indices):
                if execution == 'layerwise':
                    parts = [self.rotary_emb(hidden[:, start:end], positions[..., start:end])
                             for start, end in zip(packed_layout.boundaries, packed_layout.boundaries[1:])]
                    rope = tuple(torch.cat([part[i] for part in parts], dim=-2) for i in range(2))
                else:
                    rope = self.rotary_emb(hidden, positions)
                for layer in self.layers:
                    hidden = layer(hidden, position_embeddings=rope, attention_mask=None,
                                   past_key_values=None, use_cache=False)
                from transformers.modeling_outputs import BaseModelOutputWithPast
                normalized = (torch.cat([self.norm(hidden[:, start:end]) for start, end in
                    zip(packed_layout.boundaries, packed_layout.boundaries[1:])], dim=1)
                    if execution == 'layerwise' else self.norm(hidden))
                return BaseModelOutputWithPast(last_hidden_state=normalized)
        finally:
            _LAYOUT.reset(token)

    backbone.forward = types.MethodType(text_dispatch, backbone)
    backbone._action_head_packed_installed = True
