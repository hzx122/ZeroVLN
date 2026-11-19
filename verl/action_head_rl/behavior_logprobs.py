"""Sampled-token behavior probabilities for resumable navigation rollouts.

Use vLLM's token IDs, never re-tokenize streamed text (which omits forced,
reasoning and stop markers). At temperature one with untruncated sampling,
raw and processed log-softmax agree on every natural trainable token.
"""
from __future__ import annotations

import math
import torch


MODE = 'untruncated_temperature_one'


def sampled_logprobs(choice):
    rows = (choice.get('logprobs') or {}).get('content') or []
    result = []
    for row in rows:
        token, value = row.get('token'), row.get('logprob')
        if (not isinstance(token, str) or not token.startswith('token_id:')
                or not token[9:].isdigit() or isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(value)
                or not -9999. < value <= 0.):
            raise ValueError('behavior probability requires exact token IDs and unclamped finite logprobs')
        result.append((int(token[9:]), float(value)))
    return result


def attach_behavior_logprobs(record, usage):
    """Align probabilities to the existing decision loss mask, including STOP."""
    capture = usage.get('behavior_logprobs')
    if capture is None:
        return
    if capture.get('mode') != MODE or record['text_temperature'] != 1.:
        raise ValueError('behavior probability sampling contract differs from text loss')
    ids, values = capture.get('token_ids', []), capture.get('logprobs', [])
    if (not ids or len(ids) != len(values)
            or any(type(token) is not int or token < 0 for token in ids)
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or not -9999. < value <= 0. for value in values)):
        raise ValueError('invalid sampled-token behavior probability capture')
    prompt, end = record['text_prompt_length'], record['text_response_end']
    single = record.get('text_single_token_target_id')
    expected = ([single] if single is not None else record['all_input_ids'][prompt:end].tolist())
    if ids[:len(expected)] != expected or len(ids) < len(expected):
        raise ValueError('API behavior tokens differ from native R3 decision tokens')
    probabilities = torch.zeros_like(record['all_input_ids'], dtype=torch.float32)
    probabilities[record['text_response_mask']] = torch.tensor(values[:len(expected)], dtype=torch.float32)
    record['text_behavior_logprobs'] = probabilities
    record['text_behavior_logprob_mode'] = MODE


def correction_weights(current, behavior, valid):
    """Reference token TIS, detached and capped at 2, without a second forward.

    This is an explicitly off-policy extension of v3 GPG. Same-policy exact
    probabilities give unit weights; it is not a claim of unbiased trajectory
    importance sampling. Both FM chunks and text tokens use their own behavior.
    """
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_weights
    if current.shape != behavior.shape or current.shape != valid.shape:
        raise ValueError('behavior correction requires aligned values and validity')
    mask = valid.to(dtype=torch.bool)
    if not torch.isfinite(current[mask]).all() or not torch.isfinite(behavior[mask]).all():
        raise ValueError('non-finite live behavior probability')
    if not mask.any():
        return torch.zeros_like(current), {}
    # A padded microbatch must not create a NaN via 0 * exp(NaN).
    ratio = torch.where(mask, current.float().detach() - behavior.float().detach(), 0.)
    weights, metrics = compute_rollout_correction_weights(ratio.reshape(1, -1),
        mask.reshape(1, -1), rollout_is='token', rollout_is_threshold=2.,
        rollout_is_batch_normalize=False)
    return weights.reshape_as(current), metrics
