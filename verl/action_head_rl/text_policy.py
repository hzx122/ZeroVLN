"""One-pass on-policy language GPG for independently recorded decisions.

The objective follows VERL's ``compute_policy_loss_gpg`` and sequence-mean
aggregation.  It deliberately has no PPO ratio: the rollout and learner must
use one unchanged policy version until their single logical optimizer step.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


TEXT_RECORD_KIND = "text_policy"


def is_text_record(record: dict) -> bool:
    return record.get("record_kind") == TEXT_RECORD_KIND


def validate_response_mask(input_ids: Tensor, response_mask: Tensor, *,
                           prompt_length: int, response_end: int | None = None,
                           forced_token_mask: Tensor | None = None,
                           single_token_target_id: int | None = None) -> Tensor:
    """Validate an unpadded exact-token mask, or the explicit single-label slot.

    Only the certified single-token boundary uses a prediction-position mask
    at the final prompt input. Its target is external, never that prompt ID.
    """
    ids = torch.as_tensor(input_ids)
    mask = torch.as_tensor(response_mask)
    single = single_token_target_id is not None
    if ids.ndim != 1 or mask.shape != ids.shape or ids.numel() < (1 if single else 2):
        raise ValueError("text policy ids and response mask must be aligned 1D tensors")
    end = ids.numel() if response_end is None else int(response_end)
    boundary_valid = (1 <= int(prompt_length) == end == ids.numel()) if single else (
        1 <= int(prompt_length) < end <= ids.numel())
    if not boundary_valid:
        raise ValueError("text policy prompt/response boundaries are invalid")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("text response mask must be binary")
    mask = mask.to(dtype=torch.bool)
    if single:
        if (type(single_token_target_id) is not int or single_token_target_id < 0
                or mask[:-1].any() or not mask[-1]):
            raise ValueError("single-token target requires one last-prompt prediction slot and a real token ID")
    elif mask[:prompt_length].any() or mask[end:].any():
        raise ValueError("text response mask includes history, prompt, or forced tail")
    if forced_token_mask is not None:
        forced = torch.as_tensor(forced_token_mask, dtype=torch.bool)
        if forced.shape != mask.shape or (forced & mask).any():
            raise ValueError("text response mask includes a forced token")
    if not mask.any():
        raise ValueError("text policy decision has no sampled response tokens")
    return mask


def selected_token_logprobs(hidden: Tensor, lm_head: nn.Module, input_ids: Tensor,
                            response_mask: Tensor, temperature: float = 1.0, *,
                            single_token_target_ids: Tensor | None = None,
                            executable_metadata: Tensor | None = None,
                            trajectory_targets: Tensor | None = None) -> Tensor:
    """Compute logits only at current-response predecessor positions.

    Output has the same [batch, token] layout as the labels.  Prompt/history,
    tool inputs and query carriers never enter the logits or loss reduction.
    Their hidden states can still affect later predictions causally.
    The explicit single-token boundary instead selects the last prompt hidden
    state itself to predict the external real sampled label.
    """
    if executable_metadata is not None:
        if single_token_target_ids is not None:
            raise ValueError("executable policy cannot use single-token fallback")
        from action_head_rl.executable_policy_learning import selected_policy_statistics
        return selected_policy_statistics(hidden, lm_head, input_ids, response_mask,
                                          temperature, executable_metadata)
    if hidden.shape[:2] != input_ids.shape or response_mask.shape != input_ids.shape:
        raise ValueError("text hidden states, ids and response mask must align")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("text policy temperature must be finite and positive")
    mask = response_mask.to(device=hidden.device, dtype=torch.bool)
    single = single_token_target_ids is not None
    if trajectory_targets is not None:
        if single or executable_metadata is not None or trajectory_targets.shape != input_ids.shape:
            raise ValueError("whole-trajectory labels require an independent aligned text branch")
        indices = mask.nonzero(as_tuple=False)
        logits = lm_head(hidden[indices[:, 0], indices[:, 1]]).float() / float(temperature)
        labels = trajectory_targets.to(hidden.device)[mask]
        selected = -torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        return selected.new_zeros(input_ids.shape).index_put((indices[:, 0], indices[:, 1]), selected)
    if (not single and mask[:, 0].any()) or not mask.any():
        raise ValueError("text policy requires sampled tokens with causal predecessors")
    indices = mask.nonzero(as_tuple=False)
    if single:
        targets = single_token_target_ids
        if (targets.dtype != torch.long or targets.numel() != hidden.shape[0]
                or (targets < 0).any() or not mask[:, -1].all()
                or mask[:, :-1].any()):
            raise ValueError("single-token targets must align with last-prompt prediction slots")
        predecessor = hidden[indices[:, 0], indices[:, 1]]
        labels = targets.to(hidden.device).reshape(-1)
    else:
        predecessor = hidden[indices[:, 0], indices[:, 1] - 1]
        labels = input_ids.to(hidden.device)[indices[:, 0], indices[:, 1]]
    logits = lm_head(predecessor).float() / float(temperature)
    selected = -torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    return selected.new_zeros(input_ids.shape).index_put(
        (indices[:, 0], indices[:, 1]), selected)


def text_gpg_loss(logprobs: Tensor, response_mask: Tensor, advantage: Tensor, *,
                  sample_valid: Tensor | None = None,
                  loss_weight: Tensor | None = None,
                  coefficient: float = 1.0,
                  global_weight: Tensor | float = 1.0,
                  dp_size: int = 1, stop_target=None, stop_weight=None,
                  path_advantage=None) -> Tensor:
    """Decision-equal GPG; one record per decision, independent of FM chunks."""
    if logprobs.ndim == 3 and logprobs.shape[-1] == 5:
        from action_head_rl.executable_policy_learning import final_text_loss
        return final_text_loss(logprobs, response_mask, advantage,
            sample_valid=sample_valid, loss_weight=loss_weight, coefficient=coefficient,
            global_weight=global_weight, dp_size=dp_size, stop_target=stop_target,
            stop_weight=stop_weight, path_advantage=path_advantage)
    if any(value is not None for value in (stop_target, stop_weight, path_advantage)):
        raise ValueError("final credit cannot train unconstrained language statistics")
    if logprobs.ndim != 2 or response_mask.shape != logprobs.shape:
        raise ValueError("text GPG requires [decision, token] logprobs and mask")
    if not math.isfinite(float(coefficient)) or coefficient < 0:
        raise ValueError("text policy coefficient must be finite and nonnegative")
    mask = response_mask.to(logprobs)
    counts = mask.sum(-1)
    if (counts <= 0).any():
        raise ValueError("text GPG decision has no response tokens")
    scores = (logprobs.float() * mask).sum(-1) / counts
    advantage = advantage.reshape_as(scores).to(scores).detach()
    valid = torch.ones_like(scores) if sample_valid is None else sample_valid.reshape_as(scores).to(scores)
    weight = torch.ones_like(scores) if loss_weight is None else loss_weight.reshape_as(scores).to(scores)
    denominator = torch.as_tensor(global_weight, device=scores.device, dtype=scores.dtype)
    if denominator.numel() != 1 or denominator.item() <= 0:
        raise ValueError("text GPG logical denominator must be positive")
    return -(scores * advantage * valid * weight).sum() * (
        float(coefficient) * int(dp_size) / denominator.reshape(()))
