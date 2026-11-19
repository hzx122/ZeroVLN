"""Certified causal unpadding for Qwen SDPA with unchanged FSDP/R3 layout."""
from __future__ import annotations

import functools
import inspect
import threading
import weakref

import torch

_UNPAD_MARKER = '_action_head_sdpa_unpadded_causal'
_LOCK = threading.RLock()


def causal_left_padding(mask, length):
    """Certify the complete mask, not just its last row or a length guess.

    Return None for arbitrary masks, trainable biases, non-prefix padding,
    or fully empty samples. Tile comparisons to avoid another quadratic
    allocation. Only fully masked query rows may be removed.
    """
    if (mask is None or mask.shape != (1, 1, length, length)
            or mask.requires_grad or length < 2
            or (mask.dtype != torch.bool and not mask.is_floating_point())):
        return None
    last = mask[0, 0, -1]
    allowed = last if mask.dtype == torch.bool else last == 0
    padding = int((~allowed).sum())
    if not 0 < padding < length:
        return None
    columns = torch.arange(length, device=mask.device)
    for start in range(0, length, 512):
        stop = min(start + 512, length)
        expected = ((columns[None] <= torch.arange(start, stop, device=mask.device)[:, None])
                    & (columns[None] >= padding))
        actual = mask[0, 0, start:stop]
        if mask.dtype == torch.bool:
            matches = actual == expected
        else:
            matches = torch.where(expected, actual == 0, torch.isneginf(actual))
        if not bool(matches.all()):
            return None
    return padding


def wrap_unpadded_sdpa(original):
    # A weak reference never extends the lifetime of a quadratic mask. Verify
    # its version as well as identity before reusing a certificate across
    # decoder layers/checkpoint recomputation.
    cached = [None]

    @functools.wraps(original)
    def forward(module, query, key, value, attention_mask, *args, **kwargs):
        eligible = (module.__class__.__name__ == 'Qwen3_5MoeAttention'
            and module.__class__.__module__.startswith('transformers.models.qwen3_5_moe.')
            and not args and query.ndim == key.ndim == value.ndim == 4
            and query.shape[0] == key.shape[0] == value.shape[0] == 1
            and query.shape[-2] == key.shape[-2] == value.shape[-2]
            and kwargs.get('dropout', 0.) == 0.
            and kwargs.get('is_causal', getattr(module, 'is_causal', True)) is True
            and attention_mask is not None)
        padding = None
        if eligible:
            certificate = cached[0]
            version = None if torch.is_inference(attention_mask) else attention_mask._version
            if (version is not None and certificate is not None
                    and certificate[0]() is attention_mask
                    and certificate[1] == version and certificate[3] == query.shape[-2]):
                padding = certificate[2]
            else:
                padding = causal_left_padding(attention_mask, query.shape[-2])
                cached[0] = (weakref.ref(attention_mask), version, padding, query.shape[-2])
        if padding is None:
            return original(module, query, key, value, attention_mask, *args, **kwargs)
        # RoPE has already been applied. Slicing never renumbers positions,
        # touches routes, truncates history or detaches the autograd graph.
        output, weights = original(module, query[..., padding:, :],
            key[..., padding:, :], value[..., padding:, :], None,
            **{**kwargs, 'is_causal': True})
        output = torch.nn.functional.pad(output, (0, 0, 0, 0, padding, 0))
        forward.unpadded_calls += 1
        return output, weights
    forward.unpadded_calls = 0
    return forward


def unpadding_count():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    marker = getattr(ALL_ATTENTION_FUNCTIONS, _UNPAD_MARKER, None)
    return None if marker is None else marker['forward'].unpadded_calls


def install_unpadded_causal_sdpa():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    with _LOCK:
        marker = getattr(ALL_ATTENTION_FUNCTIONS, _UNPAD_MARKER, None)
        if marker is not None:
            if ALL_ATTENTION_FUNCTIONS.get_interface('sdpa', None) is not marker['forward']:
                raise RuntimeError('Qwen unpadded SDPA adapter was replaced')
            return dict(marker['audit'], status='already_applied')
        original = ALL_ATTENTION_FUNCTIONS.get_interface('sdpa', None)
        if original is None:
            raise RuntimeError('native SDPA interface missing')
        forward = wrap_unpadded_sdpa(original)
        audit = dict(enabled=True, revision='qwen35_moe_sdpa_certified_unpadding_v1',
            status='applied', native_module=original.__module__,
            native_file=inspect.getfile(original), scope='certified left-padded causal self-attention',
            attention_backend='sdpa', real_token_visibility_changed=False,
            outer_sequence_shape_changed=False)
        ALL_ATTENTION_FUNCTIONS.register('sdpa', forward)
        setattr(ALL_ATTENTION_FUNCTIONS, _UNPAD_MARKER, dict(forward=forward, audit=audit))
        return audit
