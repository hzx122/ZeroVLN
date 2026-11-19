"""One causal text graph per trajectory, with unchanged v3 decision credit.

Never concatenate independent prompts. The last captured sequence must contain
every earlier sampled prefix, including identical native MoE routing. Query
carrier branches are absent from these text-only captures.
"""
from __future__ import annotations

import math
import torch

from .text_policy import is_text_record, validate_response_mask


def validate_v8_options(options):
    required = dict(tool_credit_mode="rule_call_loo", multi_condition_batch_size=1)
    budget = options.get("max_tool_calls")
    if type(budget) is not int or budget < 1:
        raise ValueError("v8 requires a positive integer tool-call budget")
    if (options.get("rule_credit_version") not in ("v3_first_arrival", "v3_terminal_local")
            or any(options.get(k) != v for k, v in required.items())):
        raise ValueError("v8 requires a v3 reward version, rule_call_loo and condition batch one")
    disabled = ("joint_text_fm_forward", "zero_copy_rollout_payload", "direct_vllm_weight_sync",
                "compact_zero_advantage_waves", "fm_suffix_vjp", "fm_core_suffix_vjp",
                "dual_shared_recompute", "fast_causal_attention")
    if any(options.get(k, False) for k in disabled):
        raise ValueError("v8 requires separate text/FM graphs and the four legacy optimizations disabled")


def merge_trajectory_text(records, *, allow_mixed_policy=False):
    decisions = [row for row in records if is_text_record(row)]
    if not decisions:
        raise ValueError("v8 trajectory has no captured text decisions")
    last = decisions[-1]
    ids = last["all_input_ids"]
    mask = torch.zeros_like(ids, dtype=torch.bool)
    targets = torch.zeros_like(ids)
    weights = torch.zeros_like(ids, dtype=torch.float32)
    behavior = torch.zeros_like(ids, dtype=torch.float32) if allow_mixed_policy else None
    versions = torch.full_like(ids, -1) if allow_mixed_policy else None
    spans = []
    total_mass = 0.0
    previous_index = -1
    for row in decisions:
        index = int(row["decision_index"])
        if index <= previous_index:
            raise ValueError("v8 decision indices must increase")
        previous_index = index
        for key in (("update_policy_version", "text_temperature", "text_loss_coef")
                    if allow_mixed_policy else ("policy_version", "text_temperature", "text_loss_coef")):
            if row[key] != last[key]:
                raise ValueError(f"v8 trajectory changed {key}")
        if allow_mixed_policy:
            from action_head_rl.behavior_logprobs import MODE
            source = row['policy_version']
            if (type(source) is not int or not 0 <= source <= row['update_policy_version']
                    or row.get('text_behavior_logprob_mode') != MODE
                    or row['text_behavior_logprobs'].shape != row['all_input_ids'].shape
                    or not torch.isfinite(row['text_behavior_logprobs']).all()):
                raise ValueError('partial trajectory has invalid source versions or behavior probabilities')
        if row.get("text_executable_policy") is not None:
            raise ValueError("v8 uses v3 unconstrained text credit")
        n = row["all_input_ids"].numel()
        for key in ("all_input_ids", "all_positions", "all_routed_experts"):
            if n > ids.numel() or not torch.equal(row[key], last[key][:n]):
                raise ValueError(f"v8 non-continuous {key} at decision {index}; refusing inexact replay")
        original = validate_response_mask(row["all_input_ids"], row["text_response_mask"],
            prompt_length=row["text_prompt_length"], response_end=row.get("text_response_end"),
            forced_token_mask=row.get("text_forced_token_mask"),
            single_token_target_id=row.get("text_single_token_target_id"))
        positions = original.nonzero().flatten()
        source_positions = positions.clone()
        if row.get("text_single_token_target_id") is not None:
            labels = torch.full_like(positions, row["text_single_token_target_id"])
        else:
            labels = row["all_input_ids"][positions]
            positions = positions - 1
        if mask[positions].any():
            raise ValueError("v8 response spans overlap")
        advantage, mass = float(row["advantage"]), float(row["loss_weight"])
        if not math.isfinite(advantage) or not math.isfinite(mass) or mass < 0:
            raise ValueError("v8 decision credit must be finite with nonnegative mass")
        if float(row.get("sample_valid", 1.0)) != 1.0:
            raise ValueError("v8 must merge before distributed padding")
        mask[positions] = True
        targets[positions] = labels
        weights[positions] = advantage * mass / len(positions)
        if allow_mixed_policy:
            behavior[positions] = row['text_behavior_logprobs'][source_positions]
            versions[positions] = row['policy_version']
        total_mass += mass
        spans.append(dict(decision_index=index, tokens=len(positions), advantage=advantage,
                          loss_weight=mass, first_prediction=int(positions[0]),
                          last_prediction=int(positions[-1])))
        if allow_mixed_policy:
            spans[-1]['policy_version'] = row['policy_version']
    if total_mass <= 0:
        raise ValueError("v8 trajectory has no positive loss mass")
    merged = dict(last, text_response_mask=mask, text_trajectory_targets=targets,
        text_trajectory_weights=weights, text_trajectory_spans=spans,
        text_trajectory_decisions=len(decisions), advantage=1.0,
        text_processor_prompt_only=last.get("text_single_token_target_id") is not None,
        loss_weight=total_mass, condition_id=last["condition_id"] + "/whole-trajectory",
        option_id="text-trajectory")
    for key in ("text_target_mode", "text_single_token_target_id", "text_forced_token_mask"):
        merged.pop(key, None)
    if allow_mixed_policy:
        merged.update(text_trajectory_behavior_logprobs=behavior,
                      text_trajectory_policy_versions=versions)
    return [row for row in records if not is_text_record(row)] + [merged]


def validate_trajectory_targets(ids, mask, targets, weights):
    if ids.ndim != 1 or any(x.shape != ids.shape for x in (mask, targets, weights)):
        raise ValueError("v8 text targets must align with the whole sequence")
    if mask.dtype != torch.bool or not mask.any() or targets.dtype != torch.long:
        raise ValueError("v8 requires explicit prediction slots and integer labels")
    if (targets[mask] < 0).any() or not torch.isfinite(weights).all() or weights[~mask].any():
        raise ValueError("v8 invalid labels or credit outside sampled prediction slots")


def trajectory_text_loss(logprobs, weights, *, sample_valid, coefficient, global_weight, dp_size=1):
    if logprobs.shape != weights.shape or logprobs.ndim != 2:
        raise ValueError("v8 logprobs and token credit must align")
    if not math.isfinite(float(coefficient)) or coefficient < 0:
        raise ValueError("v8 text coefficient must be finite and nonnegative")
    denominator = torch.as_tensor(global_weight, device=logprobs.device, dtype=torch.float32)
    if denominator.numel() != 1 or not torch.isfinite(denominator) or denominator <= 0:
        raise ValueError("v8 logical denominator must be positive and finite")
    return -(logprobs.float() * weights.detach().to(logprobs) *
             sample_valid.reshape(-1, 1).to(logprobs)).sum() * (coefficient * dp_size / denominator)
