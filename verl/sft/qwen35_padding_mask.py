"""Correct the installed Qwen3.5 MoE mask helper for padded batch-one RL.

The source-bound upstream helper skips masking when B=1. RL's synchronized
rank layouts can left-pad even a single condition, so those tokens must not
enter the recurrent state. This changes no installed package files.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import threading


PADDING_MASK_REVISION = "qwen35_moe_padding_mask_all_batches_v1"
UPSTREAM_HELPER_SHA256 = "694b0342a3bfdf0403562fb16ebadb4edea90521dfcf07114322d4737ca49a35"
_MARKER = "_action_head_single_batch_padding_fix"
_LOCK = threading.RLock()


def apply_mask_to_padding_states(hidden_states, attention_mask):
    # Preserve upstream B>1 and single-token behavior; remove only the B>1
    # condition that incorrectly exempts padded batch-one prefill/training.
    if attention_mask is not None and attention_mask.shape[1] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


def install_single_batch_padding_fix():
    module = importlib.import_module("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
    with _LOCK:
        marker = getattr(module, _MARKER, None)
        if marker is not None:
            if module.apply_mask_to_padding_states is not apply_mask_to_padding_states:
                raise RuntimeError("Qwen padding mask helper was replaced after installation")
            return {**marker, "status": "already_applied"}
        source = inspect.getsource(module.apply_mask_to_padding_states)
        source_sha = hashlib.sha256(source.encode()).hexdigest()
        if source_sha != UPSTREAM_HELPER_SHA256:
            raise RuntimeError("Qwen padding mask helper source SHA256 differs; review upstream behavior")
        module.apply_mask_to_padding_states = apply_mask_to_padding_states
        audit = {"enabled": True, "status": "applied", "revision": PADDING_MASK_REVISION,
                 "upstream_helper_sha256": source_sha, "scope": "Qwen3.5 MoE linear attention"}
        setattr(module, _MARKER, audit)
        return dict(audit)
