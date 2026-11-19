"""Exact, opt-in high-level records for the v6 joint text/action objective.

These helpers only consume CPU rollout provenance. They do not tokenize a
second time, sample another answer, or turn privileged rewards into inputs.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Mapping

import torch


SINGLE_TOKEN_TARGET_MODE = "single_sampled_token_from_last_prompt_v1"
_COMPLETED_STATUSES = {"FINISHED_STOPPED": 1, "FINISHED_LENGTH_CAPPED": 2}


def decision_capture_tensors(*, all_ids, output_ids, token_count: int,
                             query_count: int, has_query_tail: bool,
                             temperature: float,
                             finished_status: str | None = None) -> dict[str, torch.Tensor]:
    """Persist authoritative scheduler boundaries alongside native R3 routes.

    The final sampled token has not been forwarded by the scheduler. We omit
    that token from the objective (normally natural EOS, or a truncated token)
    instead of inventing a route for it. Forced query carriers are also omitted.
    The previously undefined one-sampled-token boundary is explicit: its real
    label is predicted from the last prompt hidden state, with prompt-only R3.
    This exception does not add the final label to longer completions.
    """
    if (not output_ids or len(output_ids) > len(all_ids)
            or list(all_ids[-len(output_ids):]) != list(output_ids)):
        raise ValueError("decision output IDs do not match scheduler history")
    if token_count != len(all_ids) - 1:
        raise ValueError("decision capture requires native forwarded-token routes")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("text GPG requires a finite positive sampling temperature")
    prompt = len(all_ids) - len(output_ids)
    end = token_count - (query_count if has_query_tail else 0)
    single = (not has_query_tail and len(output_ids) == 1
              and 0 < prompt == end == token_count)
    if single and finished_status not in _COMPLETED_STATUSES:
        raise ValueError("single-token decision requires a verified stopped or length-capped request")
    if single and (type(output_ids[0]) is not int or output_ids[0] < 0):
        raise ValueError("single-token decision requires a nonnegative integer sampled ID")
    if not single and not 0 < prompt < end <= token_count:
        raise ValueError("decision has no forwarded natural completion token")
    result = {
        "decision_prompt_length": torch.tensor([prompt], dtype=torch.int64),
        "decision_response_end": torch.tensor([end], dtype=torch.int64),
        "decision_temperature": torch.tensor([temperature], dtype=torch.float64),
        "decision_unforwarded_final_id": torch.tensor([all_ids[-1]], dtype=torch.int64),
    }
    if single:
        result.update(
            decision_single_token_target_id=torch.tensor([output_ids[0]], dtype=torch.int64),
            decision_sampled_token_count=torch.tensor([1], dtype=torch.int64),
            decision_finished_status=torch.tensor([_COMPLETED_STATUSES[finished_status]], dtype=torch.int64))
    return result


def build_text_record(tensors: Mapping[str, torch.Tensor], *, request: Any,
                      response: Mapping[str, Any], decision_index: int,
                      policy_version: int) -> dict[str, Any]:
    """Create one text record per decision, independently of FM chunk count."""
    required = ("all_input_ids", "all_routed_experts", "all_positions",
                "decision_prompt_length", "decision_response_end", "decision_temperature")
    if any(key not in tensors for key in required):
        raise ValueError("v6 decision artifact lacks exact tokens/routes/boundaries")
    prompt = int(tensors["decision_prompt_length"].item())
    end = int(tensors["decision_response_end"].item())
    ids = tensors["all_input_ids"]
    routes, positions = tensors["all_routed_experts"], tensors["all_positions"]
    single = "decision_single_token_target_id" in tensors
    if single:
        evidence = ("decision_single_token_target_id", "decision_sampled_token_count",
                    "decision_finished_status", "decision_unforwarded_final_id")
        if (any(key not in tensors or tensors[key].numel() != 1
                or tensors[key].dtype != torch.int64 for key in evidence)
                or int(tensors["decision_sampled_token_count"].item()) != 1
                or int(tensors["decision_finished_status"].item()) not in _COMPLETED_STATUSES.values()
                or int(tensors["decision_single_token_target_id"].item()) < 0
                or not torch.equal(tensors["decision_single_token_target_id"],
                                   tensors["decision_unforwarded_final_id"])):
            raise ValueError("single-token decision lacks verified native target provenance")
    boundary_valid = (0 < prompt == end == ids.numel()) if single else (0 < prompt < end <= ids.numel())
    if (ids.ndim != 1 or routes.ndim != 3 or routes.shape[0] != ids.numel()
            or positions.shape != ids.shape or not boundary_valid
            or not torch.equal(positions.cpu(), torch.arange(ids.numel()))):
        raise ValueError("v6 decision artifact has inconsistent token alignment")
    usage_prompt = (response.get("usage") or {}).get("prompt_tokens")
    if usage_prompt is not None and int(usage_prompt) != prompt:
        raise ValueError("v6 decision scheduler/API prompt lengths differ")
    if single and (response.get("usage") or {}).get("completion_tokens") not in (None, 1):
        raise ValueError("single-token scheduler/API completion counts differ")
    mask = torch.zeros(end, dtype=torch.bool)
    mask[-1:] = single
    if not single:
        mask[prompt:] = True
    record = {
        "record_kind": "text_policy", "decision_index": int(decision_index),
        "option_id": f"text-decision-{decision_index}",
        "policy_version": int(policy_version),
        "slow_request": copy.deepcopy(request), "slow_response": copy.deepcopy(response),
        "all_input_ids": ids[:end].long().cpu(),
        "all_routed_experts": routes[:end].long().cpu(),
        "all_positions": positions[:end].long().cpu(),
        "text_prompt_length": prompt, "text_response_end": end,
        "text_response_mask": mask,
        "text_temperature": float(tensors["decision_temperature"].item()),
        "token_count": end,
        "text_history_ids": (torch.cat((ids[:end], tensors["decision_unforwarded_final_id"].reshape(-1)))
                             if end == ids.numel() and "decision_unforwarded_final_id" in tensors else ids[:end]).long().cpu(),
    }
    from action_head_rl.executable_policy import validate_captured_policy
    executable = validate_captured_policy(tensors)
    if executable is not None:
        if single:
            raise ValueError("executable tool cannot be a one-token completion")
        record["text_executable_policy"] = executable
    if single:
        record.update(text_target_mode=SINGLE_TOKEN_TARGET_MODE,
                      text_single_token_target_id=int(tensors["decision_single_token_target_id"].item()))
    from action_head_rl.behavior_logprobs import attach_behavior_logprobs
    attach_behavior_logprobs(record, response.get('usage') or {})
    return record


def attach_text_advantage(records: list[dict], *, advantage: float,
                          step: int, rollout_index: int,
                          policy_version: int, coefficient: float,
                          allow_mixed_policy: bool = False) -> list[dict]:
    """Give every episode unit text mass, without repeating calls per chunk."""
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("text loss coefficient must be finite and nonnegative")
    output = []
    for record in records:
        if record.get("record_kind") != "text_policy":
            raise ValueError("high-level decision list contains a non-text record")
        source_version = int(record.get('policy_version', -1))
        if ((not allow_mixed_policy and source_version != policy_version)
                or (allow_mixed_policy and not 0 <= source_version <= policy_version)):
            raise ValueError("high-level rollout policy-version mismatch")
        if allow_mixed_policy and 'text_behavior_logprobs' not in record:
            raise ValueError('partial decision lacks actual behavior logprobs')
        output.append({**record, "advantage": float(advantage),
                       "loss_weight": 1.0 / len(records), "text_loss_coef": coefficient,
                       "condition_id": f"step={step}/rollout={rollout_index}/text={record['decision_index']}"})
        if allow_mixed_policy:
            output[-1]['update_policy_version'] = policy_version
    return output
