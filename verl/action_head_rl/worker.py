"""Ray/VERL learner worker for one continuous-action GRPO microbatch."""
from __future__ import annotations

import json
import os
import shutil
import time
import gc
import math
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any

import torch

import action_head_rl.engine  # noqa: F401  (register custom FSDP engine)
from action_head_rl.checkpoint import (
    hf_qwen_export_key, normalize_saved_hf_qwen_visual_keys,
    save_hf_qwen_export, training_state_digest,
    validate_hf_qwen_export_keys,
    validate_optimizer_learning_rates,
)
from action_head_rl.text_policy import is_text_record, validate_response_mask
from action_head_rl.decision_records import SINGLE_TOKEN_TARGET_MODE
from action_head_rl.workload_metrics import condition_workload_metrics, sum_workload_metrics
from action_head_rl.runtime_progress import LearnerProgress
from sft.joint_fm_head_sft import (
    _slow_template_messages, build_slow_batch_inputs, build_slow_inputs,
)
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import tensordict_utils as tu
from verl.utils.seqlen_balancing import (
    calculate_workload, get_seqlen_balanced_partitions,
)
from verl.utils.tensordict_utils import maybe_fix_3d_position_ids
from verl.workers.engine_workers import TrainingWorker


FM_REPLAY_INPUT_REVISION = "exact_input_ids_mm_token_types_v1"


OPTION_SHARED_KEYS = (
    "slow_request", "slow_response", "query_routed_experts",
    "query_positions", "all_routed_experts", "all_positions", "all_input_ids",
    "reference_visual_tokens", "reference_visual_mask", "timesteps",
    "noise_scales", "signed_dt", "var_floor",
)


def _full_tensor(value: torch.Tensor) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    return value.full_tensor() if isinstance(value, DTensor) else value


def validate_direct_weight_bucket(state, bucket_size_mb: int) -> dict:
    """Check global tensor metadata before any NCCL buffer or transfer starts."""
    if type(bucket_size_mb) is not int or bucket_size_mb <= 0:
        raise ValueError("direct weight bucket must be a positive integer MiB")
    largest_bytes, largest_name, count = 0, None, 0
    for name, value in state.items():
        # DTensor.numel() describes its global shape. No full_tensor/all-gather
        # is needed, and floating transport is BF16 even with FP32 masters.
        size = value.numel() * (2 if value.is_floating_point() else value.element_size())
        if size > largest_bytes:
            largest_bytes, largest_name = size, name
        count += 1
    if not count or largest_bytes == 0:
        raise ValueError("direct publication requires nonempty Qwen weights")
    if largest_bytes > bucket_size_mb << 20:
        raise ValueError(f"direct bucket {bucket_size_mb} MiB is too small for "
                         f"{largest_name}: {largest_bytes} BF16 transport bytes")
    return {"tensor_count": count, "largest_tensor": largest_name,
            "largest_transport_tensor_bytes": largest_bytes,
            "bucket_bytes": bucket_size_mb << 20}


def record_to_tensordict(record: dict[str, Any], processor, device) -> Any:
    if is_text_record(record):
        return text_record_to_tensordict(record, processor, device)
    return record_group_to_tensordict([record], processor, device)


def _exact_mm_token_type_ids(processor, exact: torch.Tensor, *,
                             left_padding: int = 0) -> torch.Tensor:
    """Align modality positions with native R3 IDs, never a second template.

    The image processor still supplies pixels and grids. Its reconstructed
    chat text can omit historical thinking delimiters, shifting every later
    image span even when the sampled token IDs are subsequently restored.
    """
    if exact.ndim != 1 or not exact.numel() or left_padding < 0:
        raise ValueError("exact multimodal replay requires nonempty 1D IDs and nonnegative padding")
    builder = getattr(processor, "create_mm_token_type_ids", None)
    if not callable(builder):
        raise ValueError("exact multimodal replay requires the processor's exact token-type builder")
    types = torch.as_tensor(builder([exact.tolist()]), dtype=torch.long)
    if types.shape != (1, exact.numel()):
        raise ValueError("exact multimodal token types must align with authoritative input IDs")
    return torch.nn.functional.pad(types, (left_padding, 0), value=0)


def text_record_to_tensordict(record: dict[str, Any], processor, device) -> Any:
    """Pack one exact high-level decision, including STOP with no FM chunks."""
    if os.environ.get("ACTION_HEAD_TEXT_POLICY", "0") != "1":
        raise ValueError("text policy records require ACTION_HEAD_TEXT_POLICY=1")
    request = record["slow_request"]
    response = record["slow_response"]
    exact = torch.as_tensor(record["all_input_ids"], dtype=torch.long)
    single_target = record.get("text_single_token_target_id")
    if (record.get("text_target_mode") is not None or single_target is not None) and (
            record.get("text_target_mode") != SINGLE_TOKEN_TARGET_MODE or single_target is None):
        raise ValueError("single-token text target requires its explicit native boundary mode")
    if "text_trajectory_targets" in record:
        from action_head_rl.trajectory_text import validate_trajectory_targets
        mask = torch.as_tensor(record["text_response_mask"])
        validate_trajectory_targets(exact, mask, record["text_trajectory_targets"],
                                    record["text_trajectory_weights"])
    else:
        mask = validate_response_mask(
            exact, torch.as_tensor(record["text_response_mask"]),
            prompt_length=int(record["text_prompt_length"]),
            response_end=record.get("text_response_end"),
            forced_token_mask=record.get("text_forced_token_mask"),
            single_token_target_id=single_target)
    routes = torch.as_tensor(record["all_routed_experts"])
    positions = torch.as_tensor(record["all_positions"])
    if routes.ndim != 3 or routes.shape[0] != exact.numel() or positions.numel() != exact.numel():
        raise ValueError("text exact ids, R3 routes and positions must align")
    # Reuse the image/template conversion, but accept STOP and malformed tool
    # responses that build_slow_inputs intentionally rejects for FM anchors.
    messages, images = _slow_template_messages(request["messages"])
    prefix = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=bool(request.get("enable_thinking", False)),
        preserve_thinking=True)
    slow = dict(processor(
        # The one sampled label is not a forwarded input. In particular its
        # decoded spelling must not introduce a spurious visual placeholder.
        text=[prefix if single_target is not None or record.get("text_processor_prompt_only")
              else prefix + str(response.get("text", ""))], images=images,
        padding=True, return_tensors="pt"))
    width = max(exact.numel(), int(record.get("_condition_pack_token_width") or 0))
    padding = width - exact.numel()
    slow["input_ids"] = torch.nn.functional.pad(
        exact, (padding, 0), value=int(processor.tokenizer.pad_token_id)).unsqueeze(0)
    slow["attention_mask"] = torch.nn.functional.pad(
        torch.ones_like(exact), (padding, 0), value=0).unsqueeze(0)
    if "mm_token_type_ids" in slow:
        slow["mm_token_type_ids"] = _exact_mm_token_type_ids(
            processor, exact, left_padding=padding)
    values = {
        **slow,
        "all_routed_experts": torch.cat((routes[:1].expand(padding, -1, -1), routes)),
        "all_positions": torch.cat((positions[:1].expand(padding), positions.reshape(-1))),
        "text_response_mask": torch.nn.functional.pad(mask, (padding, 0)).unsqueeze(0),
        "text_temperature": torch.tensor([record.get("text_temperature", 1.0)]),
        "text_loss_coef": torch.tensor([record.get("text_loss_coef", 1.0)]),
        "advantage": torch.tensor([record["advantage"]], dtype=torch.float32),
        "sample_valid": torch.tensor([record.get("sample_valid", 1.0)], dtype=torch.float32),
        "loss_weight": torch.tensor([record.get("loss_weight", 1.0)], dtype=torch.float32),
    }
    if "text_trajectory_targets" in record:
        for key in ("text_trajectory_targets", "text_trajectory_weights"):
            values[key] = torch.nn.functional.pad(record[key], (padding, 0)).unsqueeze(0)
        values["text_trajectory_decisions"] = torch.tensor([record["text_trajectory_decisions"]])
    from action_head_rl.executable_policy_learning import pack_text_metadata
    values.update(pack_text_metadata(record, padding=padding))
    if single_target is not None:
        values["text_single_token_target_ids"] = torch.tensor([single_target], dtype=torch.long)
    if "logical_batch_valid" in record:
        values["logical_batch_valid"] = torch.tensor([record["logical_batch_valid"]], dtype=torch.float32)
    return tu.get_tensordict(tensor_dict={
        key: value.unsqueeze(0) for key, value in values.items()
        if torch.is_tensor(value)}).to(device)


def _condition_family(record):
    if not is_text_record(record):
        return "fm"
    return ("text_single" if record.get("text_single_token_target_id") is not None
            else "text")


def text_records_to_tensordict(records, processor, device):
    """Batch independent text decisions while preserving exact per-row targets.

    Ordinary sampled-token and last-prompt single-token supervision use
    separate aligned phases. Reuse the authoritative single-row processor
    path, then concatenate pixels/grids in the same row/image order as IDs.
    """
    if not records or any(not is_text_record(row) for row in records):
        raise ValueError("text batch requires text decisions")
    if len(records) == 1:
        return text_record_to_tensordict(records[0], processor, device)
    if any("text_trajectory_targets" in row for row in records):
        raise ValueError("v8 whole-trajectory text requires condition batch size one")
    if os.environ.get("ACTION_HEAD_FINAL_CONTRACT") == "three-source-boundary-stop-v1":
        raise ValueError("final executable policy requires B1 text packing")
    if len({_condition_family(row) for row in records}) != 1:
        raise ValueError("single-token and ordinary text targets require separate batches")
    from sft.verl_action_head_engine import unpack_anchor_batch
    width = max(max(torch.as_tensor(row["all_input_ids"]).numel(),
                    int(row.get("_condition_pack_token_width") or 0)) for row in records)
    rows = [unpack_anchor_batch(text_record_to_tensordict(
        dict(row, _condition_pack_token_width=width), processor, "cpu")) for row in records]
    shared_keys = {"text_temperature", "text_loss_coef", "logical_batch_valid"}
    concatenated_keys = {
        "input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw",
        "all_routed_experts", "all_positions", "text_response_mask",
        "text_single_token_target_ids", "advantage", "sample_valid", "loss_weight",
    }
    optional_images = {"pixel_values", "image_grid_thw"}
    keys = set().union(*(row.keys() for row in rows))
    if keys - shared_keys - concatenated_keys:
        raise ValueError(f"unsupported batched text fields: {sorted(keys - shared_keys - concatenated_keys)}")
    values = {}
    for key in keys:
        present = [row[key] for row in rows if key in row]
        if key not in optional_images and len(present) != len(rows):
            raise ValueError(f"batched text decisions disagree on {key}")
        if key in shared_keys:
            if any(not _same_tensor(present[0], value) for value in present[1:]):
                raise ValueError(f"batched text decisions disagree on {key}")
            values[key] = present[0]
        else:
            values[key] = torch.cat(present, dim=0)
    return tu.get_tensordict(tensor_dict={key: value.unsqueeze(0)
                                       for key, value in values.items()}).to(device)


def _same_tensor(left: Any, right: Any) -> bool:
    left = torch.as_tensor(left)
    right = torch.as_tensor(right)
    return left.shape == right.shape and left.dtype == right.dtype \
        and torch.equal(left, right)


def _validate_shared_option(records: list[dict[str, Any]]) -> None:
    """Fail closed unless records share the exact slow-policy condition."""
    first = records[0]
    identity = first.get("condition_id")
    if not identity:
        raise ValueError("option reuse requires a non-empty condition_id")
    tensor_keys = (
        "query_routed_experts", "all_routed_experts", "all_positions",
        "all_input_ids", "query_positions", "reference_visual_tokens",
        "reference_visual_mask", "timesteps", "noise_scales",
    )
    for record in records[1:]:
        if record.get("condition_id") != identity:
            raise ValueError("option group contains multiple condition_id values")
        if record["slow_request"] != first["slow_request"]:
            raise ValueError("option group changed its slow request")
        if record["slow_response"] != first["slow_response"]:
            raise ValueError("option group changed its slow response")
        for key in tensor_keys:
            if first.get(key) is None or record.get(key) is None:
                if first.get(key) is not record.get(key):
                    raise ValueError(f"option group changed optional field {key}")
            elif not _same_tensor(first[key], record[key]):
                raise ValueError(f"option group changed shared tensor {key}")
        for key in ("signed_dt", "var_floor"):
            if float(record[key]) != float(first[key]):
                raise ValueError(f"option group changed shared scalar {key}")


def canonicalize_option_payload(records: list[dict[str, Any]]) -> None:
    """Share immutable option objects so Ray pickles them once per rank arg."""
    if len(records) < 2:
        return
    _validate_shared_option(records)
    first = records[0]
    for record in records[1:]:
        for key in OPTION_SHARED_KEYS:
            record[key] = first[key]


def record_group_to_tensordict(
        records: list[dict[str, Any]], processor, device) -> Any:
    """Pack consecutive chunks while retaining one exact slow/R3 condition."""
    if not records:
        raise ValueError("record group must not be empty")
    if any(is_text_record(record) for record in records):
        if len(records) != 1:
            raise ValueError("text policy decisions must be recorded exactly once")
        return text_record_to_tensordict(records[0], processor, device)
    if len(records) > 1:
        _validate_shared_option(records)
    record = records[0]
    request = record["slow_request"]
    response = record["slow_response"]
    if not isinstance(request, dict) or not request.get("messages"):
        raise ValueError("rollout record has no exact slow request messages")
    completion = response.get("text") if isinstance(response, dict) else None
    if not completion:
        raise ValueError("rollout record has no exact slow completion")
    slow = build_slow_inputs(processor, request["messages"], completion,
                             torch.device("cpu"))
    if record.get("all_routed_experts") is not None:
        exact = torch.as_tensor(record["all_input_ids"], dtype=torch.long)
        query_count = int(record["query_routed_experts"].shape[0])
        if exact.numel() <= query_count:
            raise ValueError("captured token ids do not contain a slow prefix")
        target_full_width = max(exact.numel(), int(
            record.get("_condition_pack_token_width") or exact.numel()))
        prefix_padding = target_full_width - exact.numel()
        exact_prefix = torch.nn.functional.pad(
            exact[:-query_count], (prefix_padding, 0),
            value=int(processor.tokenizer.pad_token_id)).unsqueeze(0)
        slow["input_ids"] = exact_prefix
        slow["attention_mask"] = torch.nn.functional.pad(
            torch.ones_like(exact[:-query_count]), (prefix_padding, 0),
            value=0).unsqueeze(0)
        if "mm_token_type_ids" in slow:
            slow["mm_token_type_ids"] = _exact_mm_token_type_ids(
                processor, exact[:-query_count], left_padding=prefix_padding)
    values = {
        **slow,
        "query_routed_experts": record["query_routed_experts"],
        "reference_visual_tokens": record["reference_visual_tokens"],
        "reference_visual_mask": record["reference_visual_mask"],
        "current_visual_tokens": torch.cat([
            torch.as_tensor(row["current_visual_tokens"]) for row in records], dim=0),
        "current_visual_mask": torch.cat([
            torch.as_tensor(row["current_visual_mask"]) for row in records], dim=0),
        "states": torch.cat([
            torch.as_tensor(row["states"]) for row in records], dim=0),
        "successors": torch.cat([
            torch.as_tensor(row["successors"]) for row in records], dim=0),
        "timesteps": record["timesteps"],
        "signed_dt": torch.tensor([record["signed_dt"]], dtype=torch.float32),
        "noise_scales": record["noise_scales"],
        "old_logprob": torch.cat([
            torch.as_tensor(row["old_logprob"]).reshape(-1) for row in records]),
        "advantage": torch.tensor(
            [row["advantage"] for row in records], dtype=torch.float32),
        "var_floor": torch.tensor([record["var_floor"]], dtype=torch.float32),
        "sample_valid": torch.tensor(
            [row.get("sample_valid", 1.0) for row in records], dtype=torch.float32),
        "loss_weight": torch.tensor(
            [row.get("loss_weight", 1.0) for row in records], dtype=torch.float32),
    }
    if (os.environ.get("ACTION_HEAD_FINAL_CONTRACT") == "three-source-boundary-stop-v1"
            or any("action_path_advantage" in row for row in records)):
        if not all("action_path_advantage" in row for row in records):
            raise ValueError("final FM chunks require independent path credit")
        values["action_path_advantage"] = torch.tensor(
            [row["action_path_advantage"] for row in records], dtype=torch.float32)
    if "logical_batch_valid" in record:
        values["logical_batch_valid"] = torch.tensor(
            [record["logical_batch_valid"]], dtype=torch.float32)
    if any('update_policy_version' in row for row in records):
        if not all('update_policy_version' in row and
                0 <= row['policy_version'] <= row['update_policy_version'] for row in records):
            raise ValueError('partial FM condition lacks actual source/target versions')
        values['partial_policy_correction'] = torch.ones(1, dtype=torch.bool)
    if record.get("all_routed_experts") is not None:
        routes = torch.as_tensor(record["all_routed_experts"])
        positions = torch.as_tensor(record["all_positions"])
        padding = int(record.get("_condition_pack_token_width") or
                      routes.shape[0]) - routes.shape[0]
        values["all_routed_experts"] = torch.cat(
            (routes[:1].expand(padding, -1, -1), routes), dim=0)
        values["all_positions"] = torch.cat(
            (positions[:1].expand(padding), positions), dim=0)
    # The custom FSDP engine follows the existing action expert convention:
    # one outer anchor dimension, then the policy sample dimensions.
    tensors = {
        key: torch.as_tensor(value).unsqueeze(0)
        for key, value in values.items() if torch.is_tensor(value)
    }
    return tu.get_tensordict(tensor_dict=tensors).to(device)



def joint_group_to_tensordict(records, processor, position_fn):
    """CPU preflight of one declared pair, before any collective forward.

    Checking packed pixels/grids and actual model MRoPE catches differences
    hidden by otherwise matching scalar identities or native route positions.
    """
    from sft.verl_action_head_engine import unpack_anchor_batch

    family = records[0]["_joint_family"]
    fm = [row for row in records if not is_text_record(row)]
    texts = [row for row in records if is_text_record(row)]
    if family == "text":
        if fm or len(texts) != 1:
            raise ValueError("text family requires one decision and no FM records")
        return text_record_to_tensordict(texts[0], processor, "cpu")
    if not fm or (family == "fm" and texts) or (family == "joint" and len(texts) != 1):
        raise ValueError("joint family record cardinality mismatch")
    data = record_group_to_tensordict(fm, processor, "cpu")
    if family == "fm":
        return data
    query_count = int(fm[0]["query_routed_experts"].shape[0])
    text = dict(texts[0], _condition_pack_token_width=(
        int(fm[0]["_condition_pack_token_width"]) - query_count))
    if text.get("text_single_token_target_id") is not None:
        raise ValueError("a single sampled-token decision cannot have an Action Head query tail")
    for key in ("native_request_id", "slow_decision_index", "joint_episode_id", "policy_version"):
        if text.get(key) is None or text.get(key) != fm[0].get(key):
            raise ValueError(f"joint pair changed provenance {key}")
    if (text["slow_request"] != fm[0]["slow_request"]
            or text["slow_response"] != fm[0]["slow_response"]):
        raise ValueError("joint pair changed native request/response")
    natural = unpack_anchor_batch(text_record_to_tensordict(text, processor, "cpu"))
    full = unpack_anchor_batch(data)
    for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"):
        left, right = full.get(key), natural.get(key)
        if (left is None) != (right is None) or (left is not None and not _same_tensor(left, right)):
            raise ValueError(f"joint pair changed packed prefix {key}")
    for key in ("all_routed_experts", "all_positions"):
        if not _same_tensor(full[key][:-query_count], natural[key]):
            raise ValueError(f"joint pair changed native prefix {key}")
    if not _same_tensor(full["all_routed_experts"][-query_count:], full["query_routed_experts"]):
        raise ValueError("joint pair query tail routes mismatch")
    if not _same_tensor(full["all_positions"][-query_count:], fm[0]["query_positions"]):
        raise ValueError("joint pair query tail positions mismatch")
    left, right = position_fn(full), position_fn(natural)
    if (left is None) != (right is None) or (left is not None and not _same_tensor(left, right)):
        raise ValueError("joint pair changed actual 3D MRoPE positions")
    for source, target in (("text_response_mask", "joint_text_response_mask"),
                           ("text_temperature", "joint_text_temperature"),
                           ("text_loss_coef", "joint_text_loss_coef"),
                           ("advantage", "joint_text_advantage"),
                           ("sample_valid", "joint_text_sample_valid"),
                           ("loss_weight", "joint_text_loss_weight")):
        data[target] = natural[source].unsqueeze(0)
    from action_head_rl.executable_policy_learning import copy_joint_metadata
    copy_joint_metadata(data, natural)
    return data



def joint_condition_batch_to_tensordict(groups, processor, position_fn):
    """Reuse native pair checks and existing multi-anchor packing, on CPU."""
    from sft.verl_action_head_engine import unpack_anchor_batch
    if len(groups) == 1:
        return joint_group_to_tensordict(groups[0], processor, position_fn)
    if os.environ.get("ACTION_HEAD_FINAL_CONTRACT") == "three-source-boundary-stop-v1":
        raise ValueError("final executable policy requires B1 joint packing")
    # Validate each native pair before assembling a multi-anchor payload.
    for group in groups:
        joint_group_to_tensordict(group, processor, position_fn)
    family = groups[0][0]['_joint_family']
    if any(group[0]['_joint_family'] != family for group in groups):
        raise ValueError('joint batch changed forward family')
    if family == 'text':
        return text_records_to_tensordict([group[0] for group in groups], processor, 'cpu')
    fm_groups = [[row for row in group if not is_text_record(row)] for group in groups]
    data = record_condition_groups_to_tensordict(fm_groups, processor, 'cpu')
    if family == 'fm':
        return data
    query_count = int(fm_groups[0][0]['query_routed_experts'].shape[0])
    texts = [dict(next(row for row in group if is_text_record(row)),
                  _condition_pack_token_width=int(group[0]['_condition_pack_token_width']) - query_count)
             for group in groups]
    natural = unpack_anchor_batch(text_records_to_tensordict(texts, processor, 'cpu'))
    full = unpack_anchor_batch(data)
    for key in ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'mm_token_type_ids'):
        left, right = full.get(key), natural.get(key)
        if (left is None) != (right is None) or (left is not None and not _same_tensor(left, right)):
            raise ValueError(f'joint batch changed packed prefix {key}')
    for key in ('all_routed_experts', 'all_positions'):
        left = full[key].reshape(len(groups), full['input_ids'].shape[1] + query_count, *full[key].shape[1:])[:, :-query_count].flatten(0, 1)
        if not _same_tensor(left, natural[key]):
            raise ValueError(f'joint batch changed packed prefix {key}')
    if not _same_tensor(position_fn(full), position_fn(natural)):
        raise ValueError('joint batch changed actual 3D MRoPE positions')
    for key in ('text_temperature', 'text_loss_coef'):
        values = natural[key].reshape(-1)
        if not torch.equal(values, values[:1].expand_as(values)):
            raise ValueError(f'joint batch must share {key}')
    for source, target in (('text_response_mask', 'joint_text_response_mask'),
                           ('text_temperature', 'joint_text_temperature'),
                           ('text_loss_coef', 'joint_text_loss_coef'),
                           ('advantage', 'joint_text_advantage'),
                           ('sample_valid', 'joint_text_sample_valid'),
                           ('loss_weight', 'joint_text_loss_weight')):
        data[target] = natural[source].unsqueeze(0)
    from action_head_rl.executable_policy_learning import copy_joint_metadata
    copy_joint_metadata(data, natural)
    return data


def joint_payload_signature(data, *, family, width, fm_chunks, query_count,
                            layer_count, top_k, batch_size=1):
    """Verify actual collective/loop shapes, not merely the scheduler markers."""
    from action_head_rl.router_replay import validate_routes
    from sft.verl_action_head_engine import unpack_anchor_batch
    payload = unpack_anchor_batch(data)
    tail = query_count if family != "text" else 0
    if tuple(payload["input_ids"].shape) != (batch_size, width - tail):
        raise ValueError("joint preflight actual input width differs from scheduled width")
    if payload["attention_mask"].shape != payload["input_ids"].shape:
        raise ValueError("joint preflight attention mask shape mismatch")
    validate_routes(payload["all_routed_experts"], token_count=batch_size * width,
                    layer_count=layer_count, top_k=top_k)
    if payload["all_positions"].numel() != batch_size * width:
        raise ValueError("joint preflight native position width mismatch")
    signature = {"route_layers": layer_count, "top_k": top_k,
                 "vision_branch": "pixel_values" in payload,
                 "mm_types": "mm_token_type_ids" in payload}
    if fm_chunks:
        states = payload["states"]
        if states.ndim != 4 or states.shape[0] != batch_size * fm_chunks or min(states.shape) <= 0:
            raise ValueError("joint preflight actual FM chunk/state shape differs from schedule")
        if payload["successors"].shape != states.shape:
            raise ValueError("joint preflight SDE successor shape mismatch")
        steps = states.shape[1]
        if payload["timesteps"].shape != (steps,) or payload["noise_scales"].shape != (steps,):
            raise ValueError("joint preflight SDE loop length mismatch")
        if payload["current_visual_tokens"].shape[0] != batch_size * fm_chunks or payload["reference_visual_tokens"].shape[0] != batch_size:
            raise ValueError("joint preflight visual anchor batch mismatch")
        for key in ("advantage", "loss_weight", "sample_valid", "old_logprob"):
            if payload[key].numel() != batch_size * fm_chunks:
                raise ValueError(f"joint preflight FM target shape mismatch: {key}")
        anchors = payload.get("grpo_chunk_anchor_index", payload.get("chunk_anchor_index"))
        if anchors is None and batch_size == 1:
            anchors = torch.zeros(fm_chunks, dtype=torch.long)
        expected_anchors = torch.arange(batch_size).repeat_interleave(fm_chunks)
        if anchors is None or anchors.shape != expected_anchors.shape or not torch.equal(anchors.cpu(), expected_anchors):
            raise ValueError("joint preflight chunk anchor mapping mismatch")
        routes = payload['query_routed_experts'].reshape(batch_size * query_count, layer_count, top_k)
        validate_routes(routes, token_count=batch_size * query_count,
                        layer_count=layer_count, top_k=top_k)
        signature["sde_state_shape"] = list(states.shape[1:])
    if family in ("joint", "text"):
        key = "joint_text_response_mask" if family == "joint" else "text_response_mask"
        if payload[key].shape != payload["input_ids"].shape:
            raise ValueError("joint preflight actual text mask shape mismatch")
    return signature


def _cpu_model_positions(multimodal, payload):
    rope = getattr(multimodal, "compute_3d_position_ids", None)
    if not callable(rope):
        raise ValueError("joint preflight requires the model's actual MRoPE implementation")
    missing = object()
    saved = getattr(multimodal, "rope_deltas", missing)
    try:
        return rope(
            input_ids=payload["input_ids"], image_grid_thw=payload.get("image_grid_thw"),
            video_grid_thw=None, inputs_embeds=torch.empty(*payload["input_ids"].shape, 0),
            attention_mask=payload["attention_mask"], past_key_values=None,
            mm_token_type_ids=payload.get("mm_token_type_ids"))
    finally:
        if saved is missing:
            if hasattr(multimodal, "rope_deltas"):
                delattr(multimodal, "rope_deltas")
        else:
            multimodal.rope_deltas = saved


def record_condition_groups_to_tensordict(
        groups: list[list[dict[str, Any]]], processor, device) -> Any:
    """Batch independent option conditions behind one Qwen forward."""
    if not groups or any(not group for group in groups):
        raise ValueError("condition batch must contain non-empty groups")
    if len(groups) > 1 and os.environ.get("ACTION_HEAD_FINAL_CONTRACT") == "three-source-boundary-stop-v1":
        raise ValueError("final executable policy requires B1 FM packing")
    if len(groups) > 1 and any(is_text_record(row) for group in groups for row in group):
        if any(len(group) != 1 or not is_text_record(group[0]) for group in groups):
            raise ValueError("text and FM require separate condition batches")
        return text_records_to_tensordict([group[0] for group in groups], processor, device)
    if len(groups) == 1:
        if len(groups[0]) == 1:
            return record_to_tensordict(groups[0][0], processor, device)
        return record_group_to_tensordict(groups[0], processor, device)
    for group in groups:
        _validate_shared_option(group)
    anchors = [group[0] for group in groups]
    # The SDE loop and scalar reduction are shared across a packed forward.
    # Do not silently adopt the first option's settings for other options.
    for row in anchors[1:]:
        for key in ("timesteps", "noise_scales", "signed_dt", "var_floor",
                    "logical_batch_valid"):
            left, right = anchors[0].get(key), row.get(key)
            if ((left is None) != (right is None)
                    or (left is not None and not _same_tensor(left, right))):
                raise ValueError(f"batched FM conditions disagree on {key}")
        if torch.as_tensor(row["query_routed_experts"]).shape != torch.as_tensor(
                anchors[0]["query_routed_experts"]).shape:
            raise ValueError("batched FM conditions disagree on query/R3 shape")
    slow = build_slow_batch_inputs(
        processor,
        [row["slow_request"]["messages"] for row in anchors],
        [row["slow_response"]["text"] for row in anchors],
        torch.device("cpu"),
    )
    query_count = int(torch.as_tensor(
        anchors[0]["query_routed_experts"]).shape[0])
    exact_prefixes = [torch.as_tensor(row["all_input_ids"], dtype=torch.long)[:-query_count]
                      for row in anchors]
    global_full_width = max(int(row.get("_condition_pack_token_width") or
                                torch.as_tensor(row["all_input_ids"]).numel())
                            for row in anchors)
    prefix_width = max(max(value.numel() for value in exact_prefixes),
                       global_full_width - query_count)
    pad_token_id = int(processor.tokenizer.pad_token_id)
    slow["input_ids"] = torch.stack([
        torch.nn.functional.pad(value, (prefix_width - value.numel(), 0),
                                value=pad_token_id)
        for value in exact_prefixes
    ])
    slow["attention_mask"] = torch.stack([
        torch.nn.functional.pad(torch.ones_like(value),
                                (prefix_width - value.numel(), 0), value=0)
        for value in exact_prefixes
    ])
    if "mm_token_type_ids" in slow:
        slow["mm_token_type_ids"] = torch.cat([
            _exact_mm_token_type_ids(
                processor, prefix, left_padding=prefix_width - prefix.numel())
            for prefix in exact_prefixes
        ])

    route_rows, position_rows = [], []
    full_width = prefix_width + query_count
    for row, prefix in zip(anchors, exact_prefixes):
        routes = torch.as_tensor(row["all_routed_experts"])
        positions = torch.as_tensor(row["all_positions"])
        if routes.shape[0] != prefix.numel() + query_count:
            raise ValueError("condition R3 routes do not match exact prefix and queries")
        padding = full_width - routes.shape[0]
        route_rows.append(torch.cat((routes[:1].expand(padding, -1, -1), routes), dim=0))
        position_rows.append(torch.cat((positions[:1].expand(padding), positions), dim=0))

    counts = [len(group) for group in groups]
    values = {
        **slow,
        "query_routed_experts": torch.stack([
            torch.as_tensor(row["query_routed_experts"]) for row in anchors]),
        "reference_visual_tokens": torch.cat([
            torch.as_tensor(row["reference_visual_tokens"]) for row in anchors], dim=0),
        "reference_visual_mask": torch.cat([
            torch.as_tensor(row["reference_visual_mask"]) for row in anchors], dim=0),
        "current_visual_tokens": torch.cat([
            torch.as_tensor(row["current_visual_tokens"]) for group in groups for row in group], dim=0),
        "current_visual_mask": torch.cat([
            torch.as_tensor(row["current_visual_mask"]) for group in groups for row in group], dim=0),
        "states": torch.cat([
            torch.as_tensor(row["states"]) for group in groups for row in group], dim=0),
        "successors": torch.cat([
            torch.as_tensor(row["successors"]) for group in groups for row in group], dim=0),
        "timesteps": anchors[0]["timesteps"],
        "signed_dt": torch.tensor([anchors[0]["signed_dt"]], dtype=torch.float32),
        "noise_scales": anchors[0]["noise_scales"],
        "old_logprob": torch.cat([
            torch.as_tensor(row["old_logprob"]).reshape(-1)
            for group in groups for row in group]),
        "advantage": torch.tensor([
            row["advantage"] for group in groups for row in group], dtype=torch.float32),
        "var_floor": torch.tensor([anchors[0]["var_floor"]], dtype=torch.float32),
        "sample_valid": torch.tensor([
            row.get("sample_valid", 1.0) for group in groups for row in group], dtype=torch.float32),
        "loss_weight": torch.tensor([
            row.get("loss_weight", 1.0) for group in groups for row in group], dtype=torch.float32),
        "all_routed_experts": torch.stack(route_rows).flatten(0, 1),
        "all_positions": torch.stack(position_rows).flatten(),
        "grpo_chunk_anchor_index": torch.repeat_interleave(
            torch.arange(len(groups), dtype=torch.long), torch.tensor(counts)),
    }
    if "logical_batch_valid" in anchors[0]:
        values["logical_batch_valid"] = torch.tensor(
            [anchors[0]["logical_batch_valid"]], dtype=torch.float32)
    tensors = {key: torch.as_tensor(value).unsqueeze(0)
               for key, value in values.items() if torch.is_tensor(value)}
    return tu.get_tensordict(tensor_dict=tensors).to(device)


def pack_condition_groups(
        groups: list[list[dict[str, Any]]], size: int,
        max_padded_tokens: int = 0):
    """Form ordered dynamic batches bounded by count and padded token work."""
    if size <= 0:
        raise ValueError("condition batch size must be positive")
    if max_padded_tokens < 0:
        raise ValueError("max padded condition tokens must be nonnegative")
    aligned_ids = [group[0].get("_condition_pack_batch") for group in groups]
    if any(value is not None for value in aligned_ids):
        if any(value is None for value in aligned_ids):
            raise ValueError("aligned condition packing metadata is incomplete")
        batches = []
        previous_id = None
        for group, batch_id in zip(groups, aligned_ids):
            if not batches or batch_id != previous_id:
                batches.append([])
            batches[-1].append(group)
            previous_id = batch_id
        if any(len(batch) > size for batch in batches):
            raise ValueError("aligned condition batch exceeds configured size")
        if any(len({_condition_family(group[0]) for group in batch}) != 1 for batch in batches):
            raise ValueError("aligned condition batch mixes forward families")
        if max_padded_tokens and any(len(batch) > 1 and len(batch) * max(
                int(group[0].get("_condition_pack_token_width") or group[0].get("token_count")
                    or torch.as_tensor(group[0]["all_input_ids"]).numel()) for group in batch
                ) > max_padded_tokens for batch in batches):
            raise ValueError("aligned condition batch exceeds padded token cap")
        return batches

    batches, current, current_width = [], [], 0
    for group in groups:
        record = group[0]
        # Token width is needed only for the optional work cap. Keeping the
        # size-one path independent of VL tensors preserves the finalized
        # Head-only preprocessing and its lightweight tests.
        token_count = int(record.get("token_count") or (
            torch.as_tensor(record["all_input_ids"]).numel()
            if "all_input_ids" in record else 1))
        next_width = max(current_width, token_count)
        exceeds = bool(current) and (
            _condition_family(current[0][0]) != _condition_family(record)
            or len(current) >= size or (
                max_padded_tokens and next_width * (len(current) + 1) > max_padded_tokens))
        if exceeds:
            batches.append(current)
            current, current_width = [], 0
        current.append(group)
        current_width = max(current_width, token_count)
    if current:
        batches.append(current)
    return batches


def shard_record_batches(
        batches: list[list[dict[str, Any]]], world: int,
) -> tuple[list[list[dict[str, Any]]], float]:
    """Transpose world-sized microbatches into rank-local record streams.

    ``Dispatch.DP_COMPUTE`` expects one argument per learner rank. Performing
    this transpose on the driver prevents Ray from materializing the complete
    heterogeneous logical batch as one ``ONE_TO_ALL`` object.
    """
    if world <= 0:
        raise ValueError("learner world size must be positive")
    if not batches:
        raise ValueError("logical train batch contains no microbatches")
    for batch in batches:
        if len(batch) != world:
            raise ValueError(
                f"expected one record per learner rank, got {len(batch)} "
                f"for world={world}")
        if len({is_text_record(record) for record in batch}) > 1:
            raise ValueError("mixed text/FM ranks require option-affinity sharding")
    normalization_records = [record for batch in batches for record in batch]
    if any(is_text_record(row) for row in normalization_records):
        normalization_records = [row for row in normalization_records if is_text_record(row)]
    logical_valid = sum(
        float(record.get("sample_valid", 1.0))
        * float(record.get("loss_weight", 1.0))
        for record in normalization_records)
    if logical_valid <= 0:
        raise ValueError("logical train batch contains no valid chunks")
    return (
        [[batch[rank] for batch in batches] for rank in range(world)],
        logical_valid,
    )


def record_input_length(row):
    """Preserve exact original input length when scheduling queue descriptors."""
    if '_queue_input_numel' in row:
        if '_transfer_queue_record' not in row or type(row['_queue_input_numel']) is not int or row['_queue_input_numel'] < 0:
            raise ValueError('invalid queued input length descriptor')
        return row['_queue_input_numel']
    return int(torch.as_tensor(row.get('all_input_ids', [])).numel())


def shard_option_record_groups(
        records: list[dict[str, Any]], world: int,
        balance_by_token_flops: bool = False,
        condition_batch_size: int = 1,
        max_padded_tokens: int = 0,
        condition_token_bucket_size: int = 0,
        align_text_families: bool = False,
        text_before_fm: bool = False,
        trajectory_affinity: bool = False,
) -> tuple[list[list[dict[str, Any]]], float]:
    """Keep each option on one rank and align FSDP call/loop counts.

    One wave contains one option per rank. Shorter options and the final wave
    are padded to the wave's maximum chunk count with zero-valid copies, so
    every rank executes the same number of wrapped-module calls.
    """
    if world <= 0:
        raise ValueError("learner world size must be positive")
    if not records:
        raise ValueError("logical train batch contains no records")
    text_records = [row for row in records if is_text_record(row)]
    if trajectory_affinity and text_records:
        raise ValueError("FM trajectory affinity requires the separate Megatron text phase")
    if text_records:
        identities = [row.get("condition_id") for row in text_records
                      if float(row.get("sample_valid", 1.0)) > 0]
        if any(not key for key in identities) or len(set(identities)) != len(identities):
            raise ValueError("text policy requires one unique condition_id per decision")
        fm_records = [row for row in records if not is_text_record(row)]
        phases = []
        if condition_batch_size > 1 or align_text_families:
            phases.extend([[row for row in text_records if _condition_family(row) == family]
                           for family in ("text", "text_single")])
        else:
            phases.append(text_records)
        if fm_records:
            phases = phases + [fm_records] if text_before_fm else [fm_records] + phases
        phases = [phase for phase in phases if phase]
        if len(phases) > 1:
            # Every FSDP rank runs the same forward branch in the same order.
            # Mixing a STOP-only LM call with another rank's FM/query call can
            # otherwise mismatch nested module collectives.
            parts = [shard_option_record_groups(
                phase, world, balance_by_token_flops, condition_batch_size,
                max_padded_tokens, condition_token_bucket_size, align_text_families)
                for phase in phases]
            combined = [[] for _ in range(world)]
            offset = 0
            for shards, _ in parts:
                highest = -1
                for rank, shard in enumerate(shards):
                    for row in shard:
                        row = dict(row)
                        batch_id = int(row.get("_condition_pack_batch", 0))
                        row["_condition_pack_batch"] = batch_id + offset
                        highest = max(highest, batch_id)
                        combined[rank].append(row)
                offset += highest + 1
            # Every v6 episode contributes total text weight 1, including
            # STOP-only episodes. FM weight also sums to 1 when chunks exist,
            # or zero otherwise. Normalize both by the same episode count;
            # summing the two family masses would change LR with STOP rate.
            return combined, sum(float(row.get("sample_valid", 1.0))
                                 * float(row.get("loss_weight", 1.0)) for row in text_records)
    option_groups: list[list[dict[str, Any]]] = []
    for record in records:
        if not record.get("condition_id"):
            raise ValueError("option-affinity sharding requires condition_id")
        if (option_groups and record["condition_id"]
                == option_groups[-1][0]["condition_id"]):
            option_groups[-1].append(record)
        else:
            option_groups.append([record])
    for group in option_groups:
        if "slow_request" in group[0]:
            canonicalize_option_payload(group)
    rank_records: list[list[dict[str, Any]]] = [[] for _ in range(world)]
    logical_valid = sum(
        float(row.get("sample_valid", 1.0))
        * float(row.get("loss_weight", 1.0)) for row in records)
    if logical_valid <= 0:
        raise ValueError("logical train batch contains no valid chunks")
    # FSDP ranks must execute the same number of wrapped-module calls. Pad the
    # number of condition groups first, then either retain the established
    # chunk-length bucketing or reuse VERL's equal-cardinality Karmarkar-Karp
    # partitioner over its transformer FLOPs proxy.  A condition remains atomic
    # because the differentiable Qwen prefix is shared by all of its chunks.
    if trajectory_affinity:
        if condition_batch_size != 1:
            raise ValueError('FM prefix reuse requires one condition per forward')
        from action_head_rl.fm_prefix_scheduling import trajectory_rank_groups
        rank_groups = trajectory_rank_groups(option_groups, world)
    else:
        option_groups.sort(key=len, reverse=True)
        while len(option_groups) % world:
            template = min(option_groups, key=lambda group: (
                record_input_length(group[0]),
                len(group),
            ))
            padding = dict(template[0])
            padding["condition_id"] = (
                f"{template[0]['condition_id']}/rank-padding={len(option_groups)}")
            padding["sample_valid"] = 0.0
            padding["advantage"] = 0.0
            option_groups.append([padding])

        if balance_by_token_flops:
            sequence_lengths = [max(1, int(group[0].get("token_count") or record_input_length(group[0]))) for group in option_groups]
            workloads = calculate_workload(torch.tensor(
                sequence_lengths, dtype=torch.int64)).tolist()
            # VERL's equal-cardinality partitioner is a greedy list scheduler, but
            # it consumes its input in the supplied order.  Rollout records arrive
            # in episode/turn order, so passing them through unchanged can place
            # most long second-turn contexts on only a few ranks.  LPT ordering is
            # required for the scheduler's workload guarantee.  Keep reusing the
            # upstream FLOPs proxy and partitioner, then map the reordered indices
            # back to the condition groups.
            order = sorted(range(len(option_groups)),
                           key=lambda index: workloads[index], reverse=True)
            ordered_groups = [option_groups[index] for index in order]
            ordered_workloads = [int(workloads[index]) for index in order]
            partitions = get_seqlen_balanced_partitions(
                ordered_workloads, world, equal_size=True)
            rank_groups = [[ordered_groups[index] for index in partition]
                           for partition in partitions]
        else:
            rank_groups = [option_groups[rank::world] for rank in range(world)]
    group_count = len(rank_groups[0])
    if any(len(groups) != group_count for groups in rank_groups):
        raise RuntimeError("FSDP token-balanced shards have unequal condition counts")

    # FSDP forward all-gathers require every rank to execute identical batch
    # boundaries.  Rank-local greedy token caps can otherwise produce (for
    # example) one B2 batch on a short rank and two B1 batches on a long rank,
    # deadlocking the next collective.  Compute one conservative packing plan:
    # split whenever *any* rank would exceed the padded-token cap.
    if (condition_batch_size <= 0 or max_padded_tokens < 0
            or condition_token_bucket_size < 0):
        raise ValueError("invalid aligned condition packing limits")

    def bucket_width(width: int) -> int:
        if not condition_token_bucket_size:
            return width
        return ((width + condition_token_bucket_size - 1)
                // condition_token_bucket_size * condition_token_bucket_size)

    pack_id = 0
    pack_count = 0
    pack_widths = [0] * world
    for group_index in range(group_count):
        widths = [bucket_width(max(1, int(rank_groups[rank][group_index][0].get(
            "token_count") or record_input_length(rank_groups[rank][group_index][0])))) for rank in range(world)]
        next_widths = [max(old, new) for old, new in zip(pack_widths, widths)]
        exceeds = pack_count > 0 and (
            pack_count >= condition_batch_size or (
                max_padded_tokens and any(
                    width * (pack_count + 1) > max_padded_tokens
                    for width in next_widths)))
        if exceeds:
            pack_id += 1
            pack_count = 0
            pack_widths = [0] * world
            next_widths = widths
        for rank in range(world):
            for record in rank_groups[rank][group_index]:
                record["_condition_pack_batch"] = pack_id
        pack_count += 1
        pack_widths = next_widths
    # Hybrid linear-attention/R3 kernels in Qwen3.6 issue shape-dependent
    # collectives.  FSDP therefore needs the same padded sequence width, not
    # merely the same number and size of condition batches, on every rank.
    global_pack_widths: dict[int, int] = {}
    for groups in rank_groups:
        for group in groups:
            group_pack_id = int(group[0]["_condition_pack_batch"])
            width = bucket_width(max(1, int(group[0].get("token_count") or record_input_length(group[0]))))
            global_pack_widths[group_pack_id] = max(
                global_pack_widths.get(group_pack_id, 0), width)
    for groups in rank_groups:
        for group in groups:
            group_pack_id = int(group[0]["_condition_pack_batch"])
            for record in group:
                record["_condition_pack_token_width"] = global_pack_widths[
                    group_pack_id]
    for group_index in range(group_count):
        wave = [rank_groups[rank][group_index] for rank in range(world)]
        chunk_count = max(len(group) for group in wave)
        for rank, group in enumerate(wave):
            padded = list(group)
            while len(padded) < chunk_count:
                padding = dict(group[0])
                padding["sample_valid"] = 0.0
                padding["advantage"] = 0.0
                padded.append(padding)
            rank_records[rank].extend(padded)
    return rank_records, logical_valid


def consecutive_record_groups(records, reuse: bool):
    """Keep the same option boundaries for proximal inference and training."""
    groups = []
    for record in records:
        identity = record.get("condition_id")
        if (reuse and identity and groups and not is_text_record(record)
                and not is_text_record(groups[-1][0])
                and identity == groups[-1][0].get("condition_id")):
            groups[-1].append(record)
        else:
            groups.append([record])
    return groups


class ActionHeadGRPOWorker(TrainingWorker):
    """TrainingWorker with rank-local preprocessing for heterogeneous VL prompts."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def audit_qwen_fast_path(self) -> dict[str, Any]:
        """Fail closed when a Full candidate silently falls back to Torch."""
        import causal_conv1d
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe

        available = bool(modeling_qwen3_5_moe.is_fast_path_available)
        if not available:
            raise RuntimeError(
                "Qwen3.6 linear-attention fast path is required but unavailable")
        return {
            "rank": int(self.rank),
            "available": available,
            "causal_conv1d_version": getattr(causal_conv1d, "__version__", "unknown"),
            "causal_conv1d_path": str(Path(causal_conv1d.__file__).resolve()),
            "fla_l2norm_patch": dict(getattr(
                self.engine, "fla_l2norm_patch_audit",
                {"enabled": False, "status": "missing_audit"})),
            "fla_unused_shape_patch": dict(getattr(
                self.engine, "fla_unused_shape_patch_audit",
                {"enabled": False, "status": "missing_audit"})),
            "padding_mask_patch": dict(getattr(
                self.engine, "padding_mask_patch_audit",
                {"enabled": False, "status": "missing_audit"})),
            "sdpa_mask_patch": dict(getattr(
                self.engine, "sdpa_mask_patch_audit",
                {"enabled": False, "status": "missing_audit"})),
            "compile_cache": dict(getattr(
                self.engine, "compile_cache_audit",
                {"enabled": False, "status": "missing_audit"})),
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def audit_initial_collectives(self, output_dir):
        """Exercise the real Ray layout without allocating model weights."""
        from action_head_rl.engine import initialize_learner_collectives
        from action_head_rl.joint_audit import write_new_json
        import resource
        import socket
        import traceback
        rank = torch.distributed.get_rank()
        root = Path(output_dir) / "collective_audit"
        memory = Path("/sys/fs/cgroup/memory")
        report = dict(rank=rank, pid=os.getpid(), hostname=socket.gethostname(),
            affinity=sorted(os.sched_getaffinity(0)), memlock=resource.getrlimit(resource.RLIMIT_MEMLOCK),
            device=torch.cuda.current_device(), visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
            env={k:v for k,v in os.environ.items() if k.startswith(("NCCL_", "TORCH_NCCL_"))
                 or k in ("LD_LIBRARY_PATH", "PYTHONPATH", "CUDA_DEVICE_MAX_CONNECTIONS",
                          "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")},
            allocator_settings=torch.cuda.memory._snapshot().get("allocator_settings", {}),
            memory={name:(memory/name).read_text() for name in
                    ("memory.usage_in_bytes", "memory.stat", "memory.failcnt") if (memory/name).exists()},
            module_file=__file__, before_model_load=True, time=time.time())
        write_new_json(root / f"rank_{rank:03d}_before.json", report)
        try:
            result = initialize_learner_collectives()
        except BaseException as exc:
            write_new_json(root / f"rank_{rank:03d}_failure.json",
                           dict(error=repr(exc), traceback=traceback.format_exc(), time=time.time()))
            raise
        write_new_json(root / f"rank_{rank:03d}_passed.json", result)
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def init_direct_weight_sync(self, bucket_size_mb: int) -> None:
        """Attach VERL's NCCL checkpoint engine to every FSDP learner rank."""
        from action_head_rl.direct_nccl import SingleDeviceNCCLCheckpointEngine

        self._direct_bucket_metadata = validate_direct_weight_bucket(
            self.engine.module.qwen.state_dict(), bucket_size_mb)
        rank = torch.distributed.get_rank()
        self._direct_checkpoint_engine = SingleDeviceNCCLCheckpointEngine(
            bucket_size=int(bucket_size_mb) << 20,
            group_name="action_head_qwen_direct", rebuild_group=False,
            is_master=(rank == 0))

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        engine = getattr(self, "_direct_checkpoint_engine", None)
        if engine is None:
            raise RuntimeError("direct weight sync is not initialized")
        # NCCLCheckpointEngine.prepare owns CuPy buffers outside PyTorch's
        # allocator. Release inactive FSDP cache before its double allocation.
        if method == "prepare":
            torch.cuda.empty_cache()
        result = getattr(engine, method)(*args, **kwargs)
        if method == "finalize" and engine.is_master:
            # VERL's NCCL engine uses CuPy buffers on the trainer master so it
            # remains compatible with expandable_segments.  Assigning None in
            # finalize() returns those buffers to CuPy's cache, not CUDA.  A
            # full Qwen learner needs that space for the following FSDP
            # all-gather, so release the inactive transfer cache at the phase
            # boundary.  The persistent NCCL communicator remains alive.
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()
            torch.cuda.empty_cache()
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int | None = None) -> None:
        """Gather Qwen DTensors and stream HF-named tensors through VERL NCCL."""
        engine = getattr(self, "_direct_checkpoint_engine", None)
        if engine is None:
            raise RuntimeError("direct weight sync is not initialized")

        def qwen_weights():
            seen = set()
            for name, value in self.engine.module.qwen.state_dict().items():
                export_name = hf_qwen_export_key(name)
                if export_name in seen:
                    raise RuntimeError(
                        f"duplicate direct Qwen key after normalization: {export_name}")
                seen.add(export_name)
                full = _full_tensor(value.detach())
                if full.is_floating_point() and full.dtype != torch.bfloat16:
                    full = full.to(torch.bfloat16)
                yield export_name, full

        await engine.send_weights(qwen_weights())

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def colocated_placement(self) -> dict[str, Any] | None:
        """Return the learner GPU identity needed by a node-pinned service."""
        import ray

        visible = [value for value in os.environ.get(
            "CUDA_VISIBLE_DEVICES", "").split(",") if value]
        if len(visible) != 1:
            raise RuntimeError(
                f"colocation requires one GPU per learner rank, got {visible!r}")
        return {
            "rank": torch.distributed.get_rank(),
            "node_id": str(ray.get_runtime_context().get_node_id()),
            "node_ip": ray.util.get_node_ip_address(),
            "cuda_visible_device": visible[0],
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def offload_for_colocated_rollout(
            self, selected_ranks: list[int]) -> dict[str, Any] | None:
        """Release only learner ranks whose GPUs host colocated vLLM."""
        rank = torch.distributed.get_rank()
        selected = rank in {int(value) for value in selected_ranks}
        if getattr(self, "_action_head_colocated_offloaded", False):
            raise RuntimeError("learner is already offloaded for rollout")
        if not selected:
            return {"rank": rank, "selected": False, "seconds": 0.0}
        torch.cuda.synchronize()
        self.engine.optimizer_zero_grad()
        started = time.perf_counter()
        self.engine.to(
            device="cpu", model=True, optimizer=True, grad=True)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        self._action_head_colocated_offloaded = True
        return {
            "rank": rank,
            "selected": True,
            "seconds": time.perf_counter() - started,
            "free_gib": free / 1024 ** 3,
            "used_gib": (total - free) / 1024 ** 3,
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def restore_after_colocated_rollout(
            self, selected_ranks: list[int]) -> dict[str, Any] | None:
        """Restore the exact FSDP shards on colocated learner ranks."""
        rank = torch.distributed.get_rank()
        selected = rank in {int(value) for value in selected_ranks}
        if not selected:
            return {"rank": rank, "selected": False, "seconds": 0.0}
        if not getattr(self, "_action_head_colocated_offloaded", False):
            raise RuntimeError("learner was not offloaded for rollout")
        started = time.perf_counter()
        self.engine.to(
            device=self.device_name, model=True, optimizer=True, grad=True)
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        self._action_head_colocated_offloaded = False
        return {
            "rank": rank,
            "selected": True,
            "seconds": time.perf_counter() - started,
            "free_gib": free / 1024 ** 3,
            "used_gib": (total - free) / 1024 ** 3,
        }

    def _completed_learner_work(self, phase, *, force=False, **fields):
        directory = os.environ.get("ACTION_HEAD_LEARNER_PROGRESS_DIR")
        if not directory:
            return
        progress = getattr(self, "_learner_progress", None)
        if progress is None:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else int(self.rank)
            progress = self._learner_progress = LearnerProgress(directory, rank)
        progress.completed(phase, force=force,
            accumulation=int(getattr(self, "_streamed_train_count", 0)) + 1, **fields)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def clear_staged_records(self) -> int:
        """Reset rank-local rollout staging before a new logical batch."""
        self._staged_action_head_records = []
        self._staged_option_payloads = {}
        self._staged_rollout_payloads = {}
        self._joint_preflight_data = None
        return 0

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def stage_queued_record_batches(self, locators, records_per_fetch=4):
        """One metadata RPC per rank, bounded native TQ pulls without wave barriers."""
        import transfer_queue as tq
        from action_head_rl.queue_records import resolve_record_locators
        if not locators or type(records_per_fetch) is not int or records_per_fetch < 1:
            raise ValueError('queued staging requires records and a positive fetch bound')
        tq.init()
        cache = {}
        mapped_directory = os.environ.get('ACTION_HEAD_QUEUE_MMAP_DIR')
        self._staged_queue_mapped_paths = set()
        try:
            for start in range(0, len(locators), records_per_fetch):
                rows = resolve_record_locators(locators[start:start + records_per_fetch], cache,
                    mapped_directory=mapped_directory, mapped_paths=self._staged_queue_mapped_paths)
                self.stage_record_batches(rows)
            return len(self._staged_action_head_records)
        finally:
            cache.clear()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def release_staged_queue_files(self):
        """Only after every rank's staging ACK: mappings survive file unlink."""
        paths = getattr(self, '_staged_queue_mapped_paths', set())
        for path in paths:
            path.unlink(missing_ok=True)
            path.with_suffix('.lock').unlink(missing_ok=True)
        count = len(paths)
        paths.clear()
        from action_head_rl.cpu_memory import release_cpu_memory
        return dict(files=count, **release_cpu_memory())

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def stage_record_batches(self, records: list[dict[str, Any]]) -> int:
        """Retain one small aligned transport wave outside Ray's object store."""
        if not records:
            raise ValueError("cannot stage an empty record wave")
        staged = getattr(self, "_staged_action_head_records", None)
        if staged is None:
            raise RuntimeError("clear_staged_records must precede staging")
        self._joint_preflight_data = None
        payloads = self._staged_option_payloads
        for source in records:
            record = dict(source)
            payload_ref = record.pop("_rollout_payload_ref", None)
            if payload_ref is not None:
                import ray
                key = payload_ref.hex()
                packed = self._staged_rollout_payloads.get(key)
                if packed is None:
                    packed = ray.get(payload_ref)
                    self._staged_rollout_payloads[key] = packed
                chunk_index = int(record.pop("_rollout_payload_index"))
                record.update(packed["shared"])
                record.update(packed["chunks"][chunk_index])
            condition_id = record.get("condition_id")
            if record.pop("_reuse_staged_option_payload", False):
                if condition_id not in payloads:
                    raise RuntimeError(
                        f"staged option payload missing for {condition_id!r}")
                record.update(payloads[condition_id])
            elif condition_id:
                payloads[condition_id] = {
                    key: record[key] for key in OPTION_SHARED_KEYS if key in record
                }
            staged.append(record)
        self._completed_learner_work("stage_completed", staged_records=len(staged))
        return len(staged)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def validate_staged_joint_records(self, logical_valid: float) -> dict:
        """Resolve every local pair before the driver releases any rank to FSDP."""
        from action_head_rl.joint_policy import consecutive_joint_groups, consecutive_joint_batches
        if os.environ.get("ACTION_HEAD_JOINT_TEXT_FM_FORWARD", "0") != "1":
            raise ValueError("joint preflight is opt-in")
        if logical_valid <= 0:
            raise ValueError("joint preflight needs the original positive loss mass")
        records = getattr(self, "_staged_action_head_records", None)
        if not records:
            raise ValueError("joint preflight needs staged records")
        groups = consecutive_joint_groups(records)
        batches = consecutive_joint_batches(groups)
        positions = lambda payload: _cpu_model_positions(
            self.engine.module.query_forward.model.model, payload)
        prepared, schedule, signatures = {}, [], []
        expected_queries = self.engine.module.query_encoder.QUERY_COUNT
        for batch in batches:
            group = batch[0]
            flat = [row for group_rows in batch for row in group_rows]
            for row in flat:
                if not is_text_record(row) and row["query_routed_experts"].shape[0] != expected_queries:
                    raise ValueError("joint preflight query count differs from learner model")
            rows = [[dict(row, logical_batch_valid=logical_valid) for row in group_rows]
                    for group_rows in batch]
            prepared[group[0]["_joint_group_id"]] = joint_condition_batch_to_tensordict(
                rows, self.model_config.processor, positions)
            spec = {"family": group[0]["_joint_family"],
                    "width": int(group[0]["_condition_pack_token_width"]),
                    "fm_chunks": sum(not is_text_record(row) for row in group),
                    "records": len(flat),
                    **({"batch_size": len(batch)} if int(os.environ.get(
                        "ACTION_HEAD_MULTI_CONDITION_BATCH_SIZE", "1")) > 1 else {})}
            backbone = self.engine.module.query_forward._backbone()
            signatures.append(joint_payload_signature(
                prepared[group[0]["_joint_group_id"]],
                family=spec["family"], width=spec["width"], fm_chunks=spec["fm_chunks"],
                batch_size=len(batch), query_count=expected_queries, layer_count=len(backbone.layers),
                top_k=int(backbone.config.num_experts_per_tok)))
            schedule.append(spec)
            self._completed_learner_work("preflight_group_completed",
                                         completed=len(schedule), total=len(batches),
                                         force=len(schedule) == len(batches))
        self._joint_preflight_data = prepared
        self._joint_preflight_logical_valid = float(logical_valid)
        return {"schedule": schedule, "logical_valid": float(logical_valid),
                "collective_signatures": signatures}

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def train_staged_records(self, logical_valid: float) -> dict[str, Any] | None:
        """Consume all rank-local waves as the original one-step logical batch."""
        records = getattr(self, "_staged_action_head_records", None)
        if not records:
            raise RuntimeError("no staged Action Head records to train")
        try:
            replay_root = os.environ.get('ACTION_HEAD_UPDATE_REPLAY_DIR', '')
            if replay_root and not getattr(self, '_update_replay_saved', False):
                from action_head_rl.joint_fixture import capture_update_records
                text_engine = getattr(self, 'text_engine', self.engine)
                capture_update_records(replay_root, records,
                    rank=torch.distributed.get_rank(), world_size=torch.distributed.get_world_size(),
                    logical_weight=float(logical_valid), layout=dict(
                        text_backend='megatron' if hasattr(self, 'text_engine') else 'fsdp',
                        text_dp_rank=text_engine.get_data_parallel_rank(),
                        text_dp_size=text_engine.get_data_parallel_size()))
                self._update_replay_saved = True
            result = self.train_record_batches(records, logical_valid)
        finally:
            records.clear()
            self._staged_action_head_records = []
            self._staged_option_payloads = {}
            self._staged_rollout_payloads = {}
            self._joint_preflight_data = None
            from action_head_rl.cpu_memory import release_cpu_memory
            self._last_cpu_release = release_cpu_memory()
        if isinstance(result, dict):
            result['cpu_memory'] = self._last_cpu_release
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def accumulate_staged_records(self) -> dict[str, Any] | None:
        records = getattr(self, "_staged_action_head_records", None)
        if not records:
            raise RuntimeError("no staged Action Head records to accumulate")
        try:
            return self.accumulate_streamed_records(records)
        finally:
            self._staged_action_head_records = []
            self._staged_option_payloads = {}
            self._staged_rollout_payloads = {}
            self._joint_preflight_data = None

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def accumulate_streamed_waves(self, *waves) -> dict[str, Any] | None:
        """Stage and accumulate one rank shard after an all-rank driver barrier.

        Ray dependency readiness does not guarantee FIFO actor admission.
        The driver must drain every rank before another accumulation call;
        production streaming stages and acknowledges all ranks first, then
        invokes accumulate_staged_records to avoid resolver/collective races.
        """
        if not waves or any(not wave for wave in waves):
            raise ValueError("streamed accumulation requires non-empty rank waves")
        self._staged_action_head_records = []
        self._staged_option_payloads = {}
        self._staged_rollout_payloads = {}
        try:
            for wave in waves:
                self.stage_record_batches(wave)
            return self.accumulate_staged_records()
        finally:
            self._staged_action_head_records = []
            self._staged_option_payloads = {}
            self._staged_rollout_payloads = {}

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def compute_record_logprobs(self, record: dict[str, Any]) -> float | None:
        """Build the proximal anchor with the actor, under replayed routes.

        vLLM and HF/FSDP kernels are not bit-identical even with the same
        logical experts. Recomputing pi_old here mirrors standard VERL and
        prevents that inference/training gap from entering the PPO ratio.
        """
        data = record_to_tensordict(
            record, self.model_config.processor, self.device_name)
        with self.engine.eval_mode(disable_auto_offload=False):
            output = self.engine.infer_batch(data, loss_function=lambda **_: None)
        if not self.engine.is_mp_src_rank_with_outputs():
            return None
        value = output["model_output"]["new_logprob"]
        return float(torch.as_tensor(value).reshape(-1)[0])

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def compute_staged_logprobs(self) -> dict[str, Any] | None:
        """Freeze pi_old on each staged shard, reusing its exact option condition.

        No optimizer update occurs here. The driver waits for every rank before
        admitting training, and the proximal values remain with their records.
        """
        records = getattr(self, "_staged_action_head_records", None)
        if not records:
            raise RuntimeError("no staged Action Head records for proximal inference")
        if any(is_text_record(row) for row in records):
            raise ValueError("text GPG records do not use continuous proximal inference")
        groups = consecutive_record_groups(records, reuse=True)
        condition_batch_size = int(os.environ.get(
            "ACTION_HEAD_MULTI_CONDITION_BATCH_SIZE", "1"))
        max_condition_tokens = int(os.environ.get(
            "ACTION_HEAD_MAX_CONDITION_TOKENS", "0"))
        condition_batches = pack_condition_groups(
            groups, condition_batch_size, max_condition_tokens)
        audit_dir = os.environ.get("ACTION_HEAD_PIOLD_AUDIT_DIR", "")
        auditing = bool(audit_dir and not getattr(self, "_piold_audited", False))
        failures, baseline_seconds = [], 0.0
        rng_failures, baseline_logprobs = [], []
        max_abs, max_rel = 0.0, 0.0
        relative_denominator_floor = 1e-12
        started = time.perf_counter()
        with self.engine.eval_mode(disable_auto_offload=False):
            for condition_batch in condition_batches:
                group = [row for rows in condition_batch for row in rows]
                rng_before = self.engine.checkpoint_manager.get_rng_state() if auditing else None
                if len(condition_batch) == 1:
                    group = condition_batch[0]
                    data = record_group_to_tensordict(
                        group, self.model_config.processor, self.device_name)
                else:
                    data = record_condition_groups_to_tensordict(
                        condition_batch, self.model_config.processor, self.device_name)
                output = self.engine.infer_batch(data, loss_function=lambda **_: None)
                values = torch.as_tensor(output["model_output"]["new_logprob"]).float().reshape(-1).cpu()
                if values.numel() != len(group) or not torch.isfinite(values).all():
                    raise RuntimeError("invalid grouped proximal logprob shape or values")
                if auditing:
                    rng_after = self.engine.checkpoint_manager.get_rng_state()
                    rng_before_digest = training_state_digest(rng_before)
                    rng_after_digest = training_state_digest(rng_after)
                    if rng_after_digest != rng_before_digest:
                        rng_failures.append({
                            "condition_id": group[0].get("condition_id"),
                            "phase": "candidate", "before": rng_before_digest,
                            "after": rng_after_digest,
                        })
                    baseline_start = time.perf_counter()
                    try:
                        baseline = []
                        for chunk_index, record in enumerate(group):
                            self.engine.checkpoint_manager.load_rng_state(rng_before)
                            single = record_to_tensordict(
                                record, self.model_config.processor, self.device_name)
                            original = self.engine.infer_batch(single, loss_function=lambda **_: None)
                            baseline.append(float(torch.as_tensor(
                                original["model_output"]["new_logprob"]).reshape(-1)[0]))
                            baseline_rng_digest = training_state_digest(
                                self.engine.checkpoint_manager.get_rng_state())
                            if baseline_rng_digest != rng_before_digest:
                                rng_failures.append({
                                    "condition_id": record.get("condition_id"),
                                    "record_index": len(baseline_logprobs) + chunk_index,
                                    "phase": "baseline", "before": rng_before_digest,
                                    "after": baseline_rng_digest,
                                })
                        baseline_tensor = torch.tensor(baseline, dtype=values.dtype)
                        difference = (values - baseline_tensor).abs()
                        relative = difference / baseline_tensor.abs().clamp_min(
                            relative_denominator_floor)
                        finite = bool(torch.isfinite(difference).all())
                        group_max_abs = float(difference.max()) if finite else float("inf")
                        group_max_rel = float(relative.max()) if finite else float("inf")
                        max_abs, max_rel = max(max_abs, group_max_abs), max(max_rel, group_max_rel)
                        baseline_logprobs.extend(baseline)
                        if not finite or group_max_abs > 1e-5:
                            failures.append({"condition_id": group[0].get("condition_id"),
                                             "max_abs": group_max_abs, "max_rel": group_max_rel,
                                             "baseline": baseline, "candidate": values.tolist()})
                    finally:
                        self.engine.checkpoint_manager.load_rng_state(rng_after)
                    baseline_seconds += time.perf_counter() - baseline_start
                for record, value in zip(group, values):
                    record["old_logprob"] = value.reshape(1).clone()
        elapsed = time.perf_counter() - started
        report = {"records": len(records), "condition_forwards": len(condition_batches),
                  "condition_reuse_chunks": len(records) - len(groups),
                  "candidate_seconds": elapsed - baseline_seconds,
                  "baseline_seconds": baseline_seconds, "audited": auditing}
        if auditing:
            rank = torch.distributed.get_rank()
            root = Path(audit_dir)
            root.mkdir(parents=True, exist_ok=True)
            baseline_path = root / f"baseline_logprobs_rank_{rank:03d}.pt"
            report.update(
                rank=rank, failures=failures, rng_failures=rng_failures,
                passed=not failures and not rng_failures,
                max_abs=max_abs, max_rel=max_rel,
                logprob_max_abs_threshold=1e-5,
                relative_denominator_floor=relative_denominator_floor,
                rng_unchanged=not rng_failures,
                candidate_rng_checks=len(groups), baseline_rng_checks=len(records),
                baseline_logprobs_path=str(baseline_path), optimizer_steps=0)
            (root / f"rank_{rank:03d}.json").write_text(json.dumps(report, indent=2) + "\n")
            # Persist the small smoke shard for subsequent matched kernel and
            # gradient audits. Runtime compilation caches remain process-local.
            torch.save(records, root / f"records_rank_{rank:03d}.pt")
            # Keep both denominators aligned with the saved shard, including
            # padding. Do not inject audit-only fields into training records.
            torch.save({
                "rank": rank, "records": len(records),
                "baseline_old_logprob": torch.tensor(baseline_logprobs, dtype=torch.float32),
                "candidate_old_logprob": torch.cat([
                    record["old_logprob"].reshape(-1) for record in records]),
                "condition_ids": [record.get("condition_id") for record in records],
                "chunk_indices": [record.get("chunk_index") for record in records],
                "sample_valid": torch.tensor([
                    record.get("sample_valid", 1.0) for record in records], dtype=torch.float32),
            }, baseline_path)
            failed = torch.tensor(int(bool(failures or rng_failures)), device=self.device_name)
            torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
            if failed.item():
                raise RuntimeError("grouped pi_old failed numerical or RNG parity; see audit reports")
            self._piold_audited = True
        if not self.engine.is_mp_src_rank_with_outputs():
            return None
        return report

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def train_record_batches(
            self, records: list[dict[str, Any]],
            logical_valid: float) -> dict[str, Any] | None:
        """Accumulate one logical batch and perform exactly one optimizer step."""
        if not records:
            raise ValueError("logical train batch contains no microbatches")
        if logical_valid <= 0:
            raise ValueError("logical train batch contains no valid chunks")

        with self.engine.train_mode(disable_auto_offload=False):
            self.engine.optimizer_zero_grad()
            # Reset the CUDA peak counter once per logical batch so the
            # max_memory_allocated snapshots captured in engine.forward_backward_batch
            # reflect this step's high-water mark.  Pure observation; never
            # affects numerics or the optimizer.
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
            audit_dir = os.environ.get("ACTION_HEAD_SINGLE_FORWARD_AUDIT_DIR", "")
            auditing = bool(
                audit_dir and hasattr(self.engine, "checkpoint_manager")
                and not getattr(self, "_single_forward_audited", False))
            if auditing and any(is_text_record(row) for row in records):
                raise ValueError("continuous-only gradient audit cannot consume text decisions")
            rng_before = self.engine.checkpoint_manager.get_rng_state() if auditing else None
            outputs, groups, condition_batches = self._backward_record_batches(
                records, logical_valid)
            if auditing:
                self._audit_single_forward(groups, outputs, rng_before, audit_dir)
                self._single_forward_audited = True
            grad_norm = self.engine.optimizer_step()
            lr = self.engine.lr_scheduler_step()

        if not self.engine.is_mp_src_rank_with_outputs():
            return None
        metric_rows = [dict(output["metrics"]) for output in outputs]
        metrics = {}
        fm_diagnostics = {
            "ratio_mean", "approx_kl", "clip_fraction", "old_logprob_single_forward",
        }
        for key in set().union(*(row.keys() for row in metric_rows)):
            if key.startswith("workload_"):
                continue  # Raw Python counters are summed below, never tensorized.
            # Text-only forwards do not emit FM policy diagnostics. Treating
            # their missing values as zero makes an unchanged FM ratio look <1.
            rows = ([row for row in metric_rows if key in row]
                    if key in fm_diagnostics else metric_rows)
            values = [torch.as_tensor(row.get(key, 0.0)).float() for row in rows]
            metrics[key] = float(torch.stack(values).mean().item())
        # The generic mean above is wrong for counters and watermarks: time
        # accumulates across microbatches (SUM) and memory is a monotonic peak
        # (MAX).  Override the four engine-instrumented keys accordingly.
        try:
            metrics["forward_seconds"] = float(sum(
                float(row.get("forward_seconds", 0.0) or 0.0)
                for row in metric_rows))
            metrics["backward_seconds"] = float(sum(
                float(row.get("backward_seconds", 0.0) or 0.0)
                for row in metric_rows))
            metrics["max_memory_allocated_gb"] = float(max(
                float(row.get("max_memory_allocated_gb", 0.0) or 0.0)
                for row in metric_rows))
            metrics["memory_allocated_gb"] = float(max(
                float(row.get("memory_allocated_gb", 0.0) or 0.0)
                for row in metric_rows))
        except Exception:
            pass
        # New raw work counters keep the historical condition_* metrics intact.
        metrics.update(sum_workload_metrics(metric_rows))
        metrics.update({
            "grad_norm": float(grad_norm),
            "lr": float(lr),
            "microbatches": len(records),
            "condition_forwards": len(condition_batches),
            "condition_valid_tokens": sum(
                int(group[0].get("token_count") or (
                    torch.as_tensor(group[0]["all_input_ids"]).numel()
                    if "all_input_ids" in group[0] else 1))
                for group in groups),
            "condition_padded_tokens": sum(
                len(batch) * max(int(group[0].get("token_count") or (
                    torch.as_tensor(group[0]["all_input_ids"]).numel()
                    if "all_input_ids" in group[0] else 1))
                    for group in batch)
                for batch in condition_batches),
            "condition_reuse_chunks": len(records) - len(groups),
            "logical_valid_chunks": int(logical_valid),
            "optimizer_steps": 1,
        })
        for group in self.engine.optimizer.param_groups:
            name = str(group.get("name") or "default")
            metrics[f"lr/{name}"] = float(group["lr"])
        # Include optimizer allocation (especially the first Adam step), which
        # the per-forward engine snapshots above cannot observe.
        metrics["rank"] = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            metrics.update({
                "step_peak_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
                "step_peak_memory_reserved_gb": torch.cuda.max_memory_reserved() / 1024 ** 3,
                "post_update_memory_allocated_gb": torch.cuda.memory_allocated() / 1024 ** 3,
                "post_update_memory_reserved_gb": torch.cuda.memory_reserved() / 1024 ** 3,
                "post_update_device_used_gb": (total_bytes - free_bytes) / 1024 ** 3,
            })
        return metrics

    def _backward_record_batches(
            self, records, logical_valid,
            sync_final_backward: bool | None = None):
        reuse = os.environ.get("ACTION_HEAD_OPTION_CONDITION_REUSE", "0") == "1"
        if any(is_text_record(row) for row in records):
            if os.environ.get("ACTION_HEAD_TEXT_POLICY", "0") != "1":
                raise ValueError("text records require the opt-in text policy")
            identities = [row.get("condition_id") for row in records if is_text_record(row)
                          and float(row.get("sample_valid", 1.0)) > 0]
            if any(not key for key in identities) or len(set(identities)) != len(identities):
                raise ValueError("text policy requires one unique condition_id per decision")
        joint = os.environ.get("ACTION_HEAD_JOINT_TEXT_FM_FORWARD", "0") == "1"
        if joint:
            from action_head_rl.joint_policy import consecutive_joint_groups, consecutive_joint_batches
            groups = consecutive_joint_groups(records)
            joint_batches = consecutive_joint_batches(groups)
            cached = getattr(self, "_joint_preflight_data", None)
            if (cached is None or float(logical_valid) != self._joint_preflight_logical_valid
                    or set(cached) != {batch[0][0]["_joint_group_id"] for batch in joint_batches}):
                raise RuntimeError("joint training requires completed CPU preflight on every rank")
        else:
            groups = consecutive_record_groups(
                [dict(record, logical_batch_valid=logical_valid) for record in records], reuse)
        condition_batch_size = int(os.environ.get(
            "ACTION_HEAD_MULTI_CONDITION_BATCH_SIZE", "1"))
        max_condition_tokens = int(os.environ.get(
            "ACTION_HEAD_MAX_CONDITION_TOKENS", "0"))
        condition_batches = (joint_batches if joint else pack_condition_groups(
            groups, condition_batch_size, max_condition_tokens))
        if os.environ.get("ACTION_HEAD_PACKING_AUDIT", "0") == "1":
            rank = (torch.distributed.get_rank()
                    if torch.distributed.is_initialized() else 0)
            token_counts = [int(group[0].get("token_count") or
                torch.as_tensor(group[0].get("all_input_ids", [])).numel())
                for group in groups]
            token_flops_proxy = calculate_workload(torch.tensor(
                token_counts, dtype=torch.int64)).tolist()
            print(json.dumps({
                "packing_audit": True,
                "rank": rank,
                "condition_count": len(groups),
                "condition_forwards": len(condition_batches),
                "batch_sizes": [len(batch) for batch in condition_batches],
                "padded_token_widths": [max(int(group[0].get(
                    "_condition_pack_token_width") or group[0].get("token_count") or
                    torch.as_tensor(group[0].get("all_input_ids", [])).numel())
                    for group in batch) for batch in condition_batches],
                "token_counts": token_counts,
                "token_flops_proxy": token_flops_proxy,
                "token_flops_proxy_sum": sum(token_flops_proxy),
                "chunk_counts": [len(group) for group in groups],
                "valid_chunks": [sum(float(row.get("sample_valid", 1.0))
                    for row in group) for group in groups],
            }), flush=True)
        outputs = []
        defer_sync = (sync_final_backward is not None or
                      os.environ.get("ACTION_HEAD_DEFER_GRADIENT_SYNC", "0") == "1")
        from action_head_rl.residency import frozen_weight_residency, full_weight_residency
        from action_head_rl.rule_shared_audit import partial_checkpoint_mode
        checkpoint_layers_off = int(os.environ.get('ACTION_HEAD_FULL_ATTENTION_CHECKPOINT_LAYERS_OFF', '0'))
        checkpoint_context = (partial_checkpoint_mode(self.engine.module.query_forward._backbone(),
            checkpoint_layers_off, selection='full_attention') if checkpoint_layers_off else nullcontext())
        prefix_context = nullcontext(None)
        if os.environ.get('ACTION_HEAD_FM_PREFIX_REUSE', '0') == '1':
            if joint or not reuse or condition_batch_size != 1 or any(is_text_record(r) for r in records):
                raise ValueError('FM prefix reuse requires the separate one-condition FM phase')
            from action_head_rl.fm_prefix_cache import fm_prefix_reuse
            prefix_context = fm_prefix_reuse(self.engine.module.query_forward, aligned_forwards=True)
        previous_trajectory = None
        try:
            with checkpoint_context, prefix_context as prefix_cache, \
                    frozen_weight_residency(self.engine), full_weight_residency(self.engine):
                for batch_index, condition_batch in enumerate(condition_batches):
                    replay_capture = None
                    if prefix_cache is not None:
                        trajectory = condition_batch[0][0]['condition_id'].split('/option=', 1)[0]
                        if trajectory != previous_trajectory:
                            prefix_cache.clear()
                            previous_trajectory = trajectory
                    if defer_sync:
                        sync = (batch_index == len(condition_batches) - 1 and
                                sync_final_backward is not False)
                        self.engine.module.set_requires_gradient_sync(sync, recurse=True)
                        if hasattr(self.engine.module, "set_is_last_backward"):
                            self.engine.module.set_is_last_backward(sync)
                    if joint:
                        group = condition_batch[0]
                        data = cached[group[0]["_joint_group_id"]].to(self.device_name)
                    elif len(condition_batch) == 1:
                        group = condition_batch[0]
                        data = (record_group_to_tensordict(
                            group, self.model_config.processor, self.device_name)
                            if reuse else record_to_tensordict(
                                group[0], self.model_config.processor, self.device_name))
                    else:
                        data = record_condition_groups_to_tensordict(
                            condition_batch, self.model_config.processor, self.device_name)
                    maybe_fix_3d_position_ids(data)
                    if (joint and os.environ.get("ACTION_HEAD_RULE_SHARED_AUDIT_DIR")
                            and not getattr(self, "_rule_shared_audit_passed", False)):
                        if condition_batch[0][0]["_joint_family"] != "joint":
                            raise RuntimeError("first rule parity wave must contain a paired text/action condition")
                        from action_head_rl.rule_shared_audit import audit_first_shared_batch
                        report = audit_first_shared_batch(self.engine, data,
                            os.environ["ACTION_HEAD_RULE_SHARED_AUDIT_DIR"])
                        self._rule_shared_audit_passed = True
                        print(json.dumps({"rule_shared_parity": report}), flush=True)
                    output = self.engine.forward_backward_batch(
                        data, loss_function=lambda **_: None, forward_only=False)
                    output["metrics"].update(condition_workload_metrics(condition_batch, data))
                    outputs.append(output)
                    # Engine metrics already fetch detached scalars to CPU;
                    # this reports completed work without another CUDA sync.
                    self._completed_learner_work("backward_batch_completed",
                        completed=batch_index + 1, total=len(condition_batches),
                        force=batch_index + 1 == len(condition_batches))
                if prefix_cache is not None and outputs:
                    outputs[-1]['metrics'].update({
                        'workload_fm_prefix_' + key: value for key, value in prefix_cache.stats.items()})
                    if prefix_cache.images is not None:
                        outputs[-1]['metrics'].update({
                            'workload_fm_prefix_' + key: value
                            for key, value in prefix_cache.images.stats.items()})
            if prefix_cache is not None and prefix_cache.images is not None:
                if prefix_cache.images.retained_bytes != 0:
                    raise RuntimeError('FM image cache escaped its update scope')
        finally:
            # A streamed update deliberately leaves sync disabled across
            # completed groups; finish/abort restores it after the final call.
            if defer_sync and sync_final_backward is None:
                self.engine.module.set_requires_gradient_sync(True, recurse=True)
                if hasattr(self.engine.module, "set_is_last_backward"):
                    self.engine.module.set_is_last_backward(True)
        return outputs, groups, condition_batches

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def begin_streamed_train(self, expected_accumulations: int = 0) -> None:
        """Open one logical update while disjoint rollout GPUs keep collecting."""
        if getattr(self, "_streamed_train_context", None) is not None:
            raise RuntimeError("streamed learner update is already active")
        from action_head_rl.residency import optimizer_cpu_during_backward
        optimizer_cpu = os.environ.get('ACTION_HEAD_OPTIMIZER_CPU_DURING_BACKWARD', '0') == '1'
        context = ExitStack()
        if optimizer_cpu:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        try:
            context.enter_context(self.engine.train_mode(disable_auto_offload=False))
            optimizer_report = context.enter_context(
                optimizer_cpu_during_backward(self.engine, enabled=optimizer_cpu))
            self.engine.optimizer_zero_grad()
        except BaseException:
            context.close()
            raise
        self._streamed_train_context = context
        self._streamed_optimizer_residency_report = optimizer_report
        # forward_backward_batch may return tensors that still own saved
        # activations/checkpoint hooks. Keeping those result dictionaries for
        # every streamed wave grows device memory until finalization. Only the
        # number of completed forwards is needed by finish_streamed_train.
        self._streamed_train_condition_forwards = 0
        self._streamed_train_expected = int(expected_accumulations)
        self._streamed_train_count = 0
        if not optimizer_cpu:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def accumulate_streamed_records(
            self, records: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Accumulate an unnormalized group gradient before the final step."""
        if getattr(self, "_streamed_train_context", None) is None:
            raise RuntimeError("begin_streamed_train must precede accumulation")
        expected = int(getattr(self, "_streamed_train_expected", 0))
        count = int(getattr(self, "_streamed_train_count", 0))
        if expected and count >= expected:
            raise RuntimeError("received more streamed groups than declared")
        final_group = bool(expected and count + 1 == expected)
        started = time.perf_counter()
        # FSDP no-sync keeps full, unsharded gradients on every rank. That is
        # affordable for Head-only, but a 35B Full update reaches ~89 GiB and
        # OOMs before the final streamed wave. Full therefore reduce-scatters
        # every backward; the gradients remain sharded and still accumulate
        # until finish_streamed_train applies the logical-batch denominator.
        scope = getattr(getattr(
            self.engine, "action_head_train_scope", None), "value", None)
        streamed_sync_final = None if scope == "full" else final_group
        outputs, groups, batches = self._backward_record_batches(
            records, 1.0, sync_final_backward=streamed_sync_final)
        self._streamed_train_count = count + 1
        self._streamed_train_condition_forwards += len(outputs)
        if not self.engine.is_mp_src_rank_with_outputs():
            return None
        report = {
            "rank": (torch.distributed.get_rank()
                     if torch.distributed.is_initialized() else 0),
            "records": len(records), "conditions": len(groups),
            "condition_forwards": len(batches),
            "accumulation_seconds": time.perf_counter() - started,
            "dual_shared_recompute_conditions": sum(int(row["metrics"].get("dual_shared_recompute", 0)) for row in outputs),
            "dual_checkpoint_replay_count": sum(int(row["metrics"].get("dual_checkpoint_replay_count", 0)) for row in outputs),
            "dual_text_vjp_count": sum(int(row["metrics"].get("dual_text_vjp_count", 0)) for row in outputs),
            "dual_fm_vjp_count": sum(int(row["metrics"].get("dual_fm_vjp_count", 0)) for row in outputs),
            "fast_causal_condition_forwards": sum(int(row["metrics"].get("fast_causal_condition_forwards", 0)) for row in outputs),
            "fast_causal_fm_suffix_calls": sum(int(row["metrics"].get("fast_causal_fm_suffix_calls", 0)) for row in outputs),
            "fast_causal_calls": sum(int(row["metrics"].get("fast_causal_calls", 0)) for row in outputs),
            "fast_causal_padded_calls": sum(int(row["metrics"].get("fast_causal_padded_calls", 0)) for row in outputs),
            "forward_seconds": sum(float(row["metrics"].get(
                "forward_seconds", 0.0)) for row in outputs),
            "backward_seconds": sum(float(row["metrics"].get(
                "backward_seconds", 0.0)) for row in outputs),
            "selective_action_backwards": sum(int(row["metrics"].get(
                "action_selective_backward", 0)) for row in outputs),
            "shared_text_action_forwards": sum(int(row["metrics"].get(
                "shared_text_action_forward", 0)) for row in outputs),
        }
        report.update(sum_workload_metrics(row["metrics"] for row in outputs))
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            report.update({
                "peak_memory_allocated_gb": (
                    torch.cuda.max_memory_allocated() / 1024 ** 3),
                "peak_memory_reserved_gb": (
                    torch.cuda.max_memory_reserved() / 1024 ** 3),
                "post_accumulation_device_used_gb": (
                    (total_bytes - free_bytes) / 1024 ** 3),
            })
        if os.environ.get("ACTION_HEAD_PACKING_AUDIT", "0") == "1":
            print(json.dumps({"streamed_accumulation_audit": True, **report}),
                  flush=True)
        return report

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def finish_streamed_train(self, logical_valid: float) -> dict[str, Any] | None:
        """Normalize accumulated gradients, then take exactly one optimizer step."""
        context = getattr(self, "_streamed_train_context", None)
        if context is None or logical_valid <= 0:
            raise RuntimeError("invalid streamed learner finalization")
        try:
            expected = int(getattr(self, "_streamed_train_expected", 0))
            count = int(getattr(self, "_streamed_train_count", 0))
            if expected and count != expected:
                raise RuntimeError(
                    f"streamed group count mismatch: expected={expected}, got={count}")
            scale = 1.0 / float(logical_valid)
            for parameter in self.engine.module.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
            grad_norm = self.engine.optimizer_step()
            lr = self.engine.lr_scheduler_step()
            if not self.engine.is_mp_src_rank_with_outputs():
                return None
            return {"rank": (torch.distributed.get_rank()
                             if torch.distributed.is_initialized() else 0),
                    "grad_norm": float(grad_norm), "lr": float(lr),
                    # The FSDP engine clips once and skips Adam on nonfinite
                    # norm. Preserve its update/scheduler semantics and expose
                    # that decision rather than claiming an update occurred.
                    "optimizer_steps": int(math.isfinite(float(grad_norm))),
                    "optimizer_step_skipped": not math.isfinite(float(grad_norm)),
                    "grad_norm_finite": math.isfinite(float(grad_norm)),
                    "logical_batch_valid": float(logical_valid),
                    "logical_valid_chunks": float(logical_valid),
                    "accumulations": count,
                    "condition_forwards": int(
                        self._streamed_train_condition_forwards),
                    **({"optimizer_state_residency": self._streamed_optimizer_residency_report}
                       if getattr(self, '_streamed_optimizer_residency_report', None) is not None else {}),
                    **self.step_memory_report()}
        finally:
            if hasattr(self.engine.module, "set_requires_gradient_sync"):
                self.engine.module.set_requires_gradient_sync(True, recurse=True)
            if hasattr(self.engine.module, "set_is_last_backward"):
                self.engine.module.set_is_last_backward(True)
            self._streamed_train_condition_forwards = 0
            self._streamed_train_context = None
            self._streamed_train_expected = 0
            self._streamed_train_count = 0
            context.__exit__(None, None, None)
            self._streamed_optimizer_residency_report = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def step_memory_report(self):
        """Read cumulative peaks after Adam AND publication/weight synchronization.

        Peaks reset only at begin_streamed_train, never between waves or phases.
        Device-used is a point sample; it is deliberately not labelled a peak.
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if not torch.cuda.is_available():
            return {"rank": rank}
        free, total = torch.cuda.mem_get_info()
        return {"rank": rank,
                "step_peak_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "step_peak_memory_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
                "post_update_memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "post_update_device_used_gb": (total - free) / 1024**3}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def abort_streamed_train(self) -> bool:
        """Discard a partial streamed step after rollout or learner failure."""
        context = getattr(self, "_streamed_train_context", None)
        if context is None:
            return False
        try:
            self.engine.optimizer_zero_grad()
        finally:
            if hasattr(self.engine.module, "set_requires_gradient_sync"):
                self.engine.module.set_requires_gradient_sync(True, recurse=True)
            if hasattr(self.engine.module, "set_is_last_backward"):
                self.engine.module.set_is_last_backward(True)
            self._streamed_train_condition_forwards = 0
            self._streamed_train_context = None
            self._streamed_train_expected = 0
            self._streamed_train_count = 0
            self._staged_action_head_records = []
            self._staged_option_payloads = {}
            self._staged_rollout_payloads = {}
            context.__exit__(None, None, None)
            self._streamed_optimizer_residency_report = None
        return True

    def _audit_single_forward(self, groups, baseline_outputs, rng_before, audit_dir):
        """Compare real accumulated gradients, then restore the baseline update.

        Opt-in once per process. Reuses the exact training groups and VERL RNG
        helpers; it never steps an optimizer or alters the accepted gradient.
        """
        from torch.distributed.tensor import DTensor
        original_mode = os.environ.get("ACTION_HEAD_OLD_LOGPROB_MODE", "recompute")
        if original_mode != "recompute":
            raise ValueError("single-forward audit requires a recompute baseline")
        parameters = list(self.engine.module.named_parameters())
        baseline = {name: parameter.grad.detach().clone()
                    for name, parameter in parameters if parameter.grad is not None}
        rng_after = self.engine.checkpoint_manager.get_rng_state()
        candidate_outputs = []
        differences = []
        try:
            self.engine.optimizer_zero_grad()
            self.engine.checkpoint_manager.load_rng_state(rng_before)
            os.environ["ACTION_HEAD_OLD_LOGPROB_MODE"] = "single_forward"
            for group in groups:
                data = record_group_to_tensordict(
                    group, self.model_config.processor, self.device_name)
                maybe_fix_3d_position_ids(data)
                candidate_outputs.append(self.engine.forward_backward_batch(
                    data, loss_function=lambda **_: None, forward_only=False))
            for name, parameter in parameters:
                before, after = baseline.get(name), parameter.grad
                if before is None and after is None:
                    continue
                if before is None or after is None:
                    differences.append({"name": name, "missing_gradient": True})
                    continue
                left = before.to_local() if isinstance(before, DTensor) else before
                right = after.to_local() if isinstance(after, DTensor) else after
                left, right = left.float(), right.detach().float()
                error = (left - right).norm()
                norm = left.norm()
                relative = float((error / norm.clamp_min(1e-12)).item())
                maximum = float((left - right).abs().max().item()) if left.numel() else 0.0
                if not torch.isfinite(error) or relative > 1e-3:
                    differences.append({"name": name, "relative_l2": relative,
                                        "max_abs": maximum})
        finally:
            os.environ["ACTION_HEAD_OLD_LOGPROB_MODE"] = original_mode
            for name, parameter in parameters:
                parameter.grad = baseline.get(name)
            self.engine.checkpoint_manager.load_rng_state(rng_after)
        report = {
            "rank": torch.distributed.get_rank(), "scope": self.engine.action_head_train_scope.value,
            "gradient_tensor_count": len(baseline), "relative_l2_threshold": 1e-3,
            "gradient_failures": differences,
            "baseline_loss": [float(row["loss"]) for row in baseline_outputs],
            "candidate_loss": [float(row["loss"]) for row in candidate_outputs],
            "baseline_ratio": [float(row["metrics"]["ratio_mean"]) for row in baseline_outputs],
            "passed": not differences,
        }
        root = Path(audit_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"rank_{report['rank']:03d}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        os.replace(temporary, path)
        # A failed candidate does not interrupt the recompute baseline run.
        # The report gates any later switch to single_forward.

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def audit_learning_rates(self, action_head_lr: float, vlm_lr: float,
                            train_scope: str):
        report = validate_optimizer_learning_rates(
            self.engine.optimizer, self.engine.lr_scheduler,
            action_head_lr=action_head_lr, vlm_lr=vlm_lr, train_scope=train_scope)
        return {"rank": self.rank, **report}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def audit_training_state(self, output_dir: str, label: str,
                             compare_label: str = "") -> dict:
        """Acceptance-only, per-rank exact hashes before/after a fresh process load."""
        rank = torch.distributed.get_rank()
        module = self.engine.module
        report = {
            "rank": rank,
            "model": {name: training_state_digest(getattr(module, name).state_dict())
                      for name in ("qwen", "query_encoder", "action_head")},
            "optimizer": training_state_digest(self.engine.optimizer.state_dict()),
            "scheduler": training_state_digest(self.engine.lr_scheduler.state_dict()),
            "rng": training_state_digest(self.engine.checkpoint_manager.get_rng_state()),
            "trainable": self.engine.trainable_audit,
            "optimizer_step_values": sorted({float(state["step"].item())
                for state in self.engine.optimizer.state.values() if "step" in state}),
            "scheduler_last_epoch": self.engine.lr_scheduler.last_epoch,
        }
        root = Path(output_dir) / "resume_audit"
        root.mkdir(parents=True, exist_ok=True)
        if compare_label:
            prior = json.loads((root / f"{compare_label}_rank{rank}.json").read_text())
            changed = [key for key in ("model", "optimizer", "scheduler", "rng", "trainable")
                       if prior[key] != report[key]]
            report["compared_to"] = compare_label
            report["mismatches"] = changed
        if self.engine.action_head_train_scope.value == "head_only" and label.startswith("saved_"):
            initial = json.loads((root / f"loaded_000000_rank{rank}.json").read_text())
            report["frozen_mismatches"] = [name for name in ("qwen", "query_encoder")
                                            if initial["model"][name] != report["model"][name]]
        path = root / f"{label}_rank{rank}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        os.replace(temporary, path)
        if report.get("frozen_mismatches"):
            raise RuntimeError(f"frozen weights changed rank={rank}: {report['frozen_mismatches']}")
        if compare_label and report["mismatches"]:
            raise RuntimeError(f"resume state mismatch rank={rank}: {report['mismatches']}")
        return {"rank": rank, "audit": str(path)}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def trainable_audit(self) -> dict[str, int] | None:
        if not self.engine.is_mp_src_rank_with_outputs():
            return None
        return dict(self.engine.trainable_audit)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def publish_policy(self, output_dir: str, step: int,
                       export_qwen: bool, *, transport_only: bool = False,
                       qwen_export_dtype: str = "preserve",
                       bf16_gather_before_export: bool = False) -> dict[str, str] | None:
        """Collectively publish the exact next-rollout policy.

        Small query/head weights are always exported. Full mode also gathers
        the HF Qwen state so the Ray-owned vLLM server can restart on the new
        actor policy before another on-policy rollout is admitted.
        """
        from models.qwen36_fast_slow_action_head import Qwen36FastSlowActionHead

        if qwen_export_dtype not in ("preserve", "bf16"):
            raise ValueError("Qwen export dtype must be preserve or bf16")
        if qwen_export_dtype != "preserve" and not transport_only:
            raise ValueError("resumable checkpoints must preserve learner parameter precision")
        if bf16_gather_before_export:
            from action_head_rl.transport_gather import (
                prepare_qwen_transport_gather, qwen_transport_gather_consensus,
                validate_bf16_gather_before_export,
                validate_qwen_transport_gather_consensus,
            )
            validate_bf16_gather_before_export(
                enabled=True, transport_only=transport_only,
                qwen_export_dtype=qwen_export_dtype)
            if not export_qwen:
                raise ValueError("BF16 gather-before-export requires Qwen export")
            if self.engine.action_head_train_scope.value != "full":
                raise ValueError("BF16 gather-before-export requires full train scope")
        module = self.engine.module
        root = Path(output_dir)
        final = root / f"global_step_{int(step):06d}"
        temporary = final.with_name(final.name + ".publishing")
        rank = torch.distributed.get_rank()
        if rank == 0:
            if temporary.exists():
                shutil.rmtree(temporary)
            (temporary / "action_head").mkdir(parents=True)
            (temporary / "dualvln_query").mkdir(parents=True)

        query = _full_tensor(module.query_encoder.query_embeddings.detach()).float().cpu()
        head_state = {}
        for name, value in module.action_head.state_dict().items():
            full = _full_tensor(value.detach())
            if rank == 0:
                head_state[name] = full.float().cpu()
        qwen_state = None
        gather_audit = None
        if export_qwen:
            qwen_state = {} if rank == 0 else None
            if bf16_gather_before_export:
                try:
                    qwen_items = module.qwen.state_dict()
                    consensus = qwen_transport_gather_consensus(qwen_items.items())
                except Exception as exc:
                    consensus = {"ok": False, "error_type": type(exc).__name__}
                messages = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(messages, consensus)
                gather_audit = validate_qwen_transport_gather_consensus(messages)
            for name, value in (qwen_items.items() if bf16_gather_before_export
                                else module.qwen.state_dict().items()):
                if bf16_gather_before_export:
                    gather_input, _ = prepare_qwen_transport_gather(
                        value, enabled=True, transport_only=transport_only,
                        qwen_export_dtype=qwen_export_dtype)
                    full = _full_tensor(gather_input)
                    del gather_input
                else:
                    full = _full_tensor(value.detach())
                if rank == 0:
                    export_name = hf_qwen_export_key(name)
                    if export_name in qwen_state:
                        raise RuntimeError(
                            f"duplicate Qwen export key after normalization: {export_name}")
                    # Cast only the disposable inference copy. Original FSDP
                    # shards and optimizer state retain their full precision.
                    if qwen_export_dtype == "bf16" and full.is_floating_point():
                        full = full.to(torch.bfloat16)
                    qwen_state[export_name] = full.cpu()
                if bf16_gather_before_export:
                    del full
            if bf16_gather_before_export:
                del qwen_items
            if rank == 0:
                validate_hf_qwen_export_keys(qwen_state)
        torch.distributed.barrier()
        if rank != 0:
            return None

        torch.save({
            "query_embeddings": query,
            "query_count": int(query.shape[0]),
            "hidden_size": int(query.shape[1]),
            "hidden_convention": "final_norm",
        }, temporary / "dualvln_query/dualvln_query_embeddings.pt")
        self._export_head_config(
            temporary / "action_head" / Qwen36FastSlowActionHead.CONFIG_NAME)
        torch.save(head_state, temporary / "action_head" / Qwen36FastSlowActionHead.WEIGHTS_NAME)
        qwen_path = ""
        if export_qwen:
            qwen_dir = temporary / "qwen"
            serialization = save_hf_qwen_export(module.qwen, qwen_dir, qwen_state)
            del qwen_state
            remapped_visual_keys = normalize_saved_hf_qwen_visual_keys(qwen_dir)
            print(json.dumps({
                "event": "qwen_hf_export_normalized",
                **serialization,
                "remapped_visual_keys": remapped_visual_keys,
            }), flush=True)
            self.model_config.processor.save_pretrained(qwen_dir)
            qwen_path = str(final / "qwen")
        (temporary / "report.json").write_text(json.dumps({
            "schema": "action_head_ray_grpo_checkpoint",
            "step": int(step),
            "train_scope": self.engine.action_head_train_scope.value,
            "router_replay": "R3",
            "trainable_audit": self.engine.trainable_audit,
            "qwen_exported": bool(export_qwen),
            "publication_kind": "rollout_transport" if transport_only else "checkpoint",
            "qwen_export_dtype": qwen_export_dtype if export_qwen else None,
            **({"bf16_gather_before_export": gather_audit}
               if bf16_gather_before_export else {}),
        }, ensure_ascii=False, indent=2) + "\n")
        if final.exists():
            shutil.rmtree(final)
        os.replace(temporary, final)

        # Stable query path watched by the vLLM hook between rollout waves.
        stable = root / "current_query.pt"
        stable_tmp = stable.with_suffix(".pt.tmp")
        shutil.copy2(final / "dualvln_query/dualvln_query_embeddings.pt", stable_tmp)
        os.replace(stable_tmp, stable)
        return {
            "checkpoint": str(final),
            "action_head": str(final / "action_head"),
            "query": str(stable),
            "qwen": qwen_path,
        }

    def _export_head_config(self, destination: Path) -> None:
        # A resumed run can prune its source checkpoint after later commits.
        # Keep the immutable config in memory before that directory disappears.
        if not hasattr(self, "_head_export_config_bytes"):
            source = Path(os.environ["ACTION_HEAD_HEAD_CHECKPOINT"]) / destination.name
            self._head_export_config_bytes = source.read_bytes()
        destination.write_bytes(self._head_export_config_bytes)
