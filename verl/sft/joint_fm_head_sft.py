#!/usr/bin/env python3
"""Joint four-query + FM-head SFT with a frozen Qwen3.6 multimodal backbone.

Unlike :mod:`train_fm_head_sft`, this consumes the recorded slow chat prefix
from a v2 nested bundle, appends four learnable latent queries in the actual
Qwen computation graph, and optimizes only those query embeddings plus the
fast/slow FM head.  Qwen (including its vision tower) remains frozen.
"""
from __future__ import annotations

import argparse
import base64
import copy
import inspect
import io
import json
import os
import random
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from action_head.vllm_bridge import strip_query_tail
from models.dualvln_query_encoder import DualVlnQueryEncoder
from models.qwen36_fast_slow_action_head import Qwen36FastSlowActionHead
from sft.train_fm_head_sft import collate, evaluate, move, split_rows


class JointBundleDataset(Dataset):
    """Load the reusable fast tensors plus slow multimodal provenance."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if row.get('action_head_as_tool_version') == 'v8' and row.get('slow_condition_path'):
            from sft.v8_rule_training_data import load_condition
            row = load_condition(row)
        if not row.get("slow_messages") or not row.get("slow_completion"):
            raise ValueError(
                "joint SFT requires a v2 bundle with slow_messages and slow_completion; "
                "rebuild with build_fm_nested_bundles.py")
        from sft.live_fast_visual import LIVE_VISUAL, verify_image
        mode = row.get('visual_input_mode', 'preencoded')
        if mode not in ('preencoded', LIVE_VISUAL):
            raise ValueError('unknown fast visual input mode')
        if os.environ.get('ACTION_HEAD_FAST_VISUAL_MODE') == 'live' and mode != LIVE_VISUAL:
            raise ValueError('formal live training rejects preencoded visual features')
        with np.load(row["sample"], allow_pickle=False) as data:
            if mode == LIVE_VISUAL:
                if set(data.files) != {'target_actions', 'action_mask'}:
                    raise ValueError('live sample must contain action labels only')
                target, mask = data['target_actions'].astype(np.float32), data['action_mask'].astype(np.bool_)
                profile = os.environ.get('ACTION_HEAD_V8_LABEL_PROFILE', 'fine16')
                if row.get('action_head_as_tool_version') == 'v8' and profile != 'fine16':
                    from sft.v8_rule_training_data import coarse_route_labels, executed_points_mask
                    target, mask = coarse_route_labels(row)
                    # exec4: the evaluator executes only the first 4 points, so
                    # supervise exactly those and leave the rest unscored.
                    if profile == 'exec4':
                        mask = executed_points_mask()
                return dict(reference_image=verify_image(row['reference_image'], row['reference_image_sha256']),
                    current_image=verify_image(row['current_image'], row['current_image_sha256']),
                    target_actions=target, action_mask=mask,
                    slow_messages=row['slow_messages'], slow_completion=row['slow_completion'],
                    slow_anchor_id=row['slow_anchor_id'])
            return {
                "reference_visual_tokens": data["reference_visual_tokens"].astype(np.float32),
                "current_visual_tokens": data["current_visual_tokens"].astype(np.float32),
                "target_actions": data["target_actions"].astype(np.float32),
                "action_mask": data["action_mask"].astype(np.bool_),
                "slow_messages": row["slow_messages"],
                "slow_completion": row["slow_completion"],
                "slow_anchor_id": row["slow_anchor_id"],
            }


def joint_collate(samples):
    live = ['reference_image' in sample for sample in samples]
    if any(live) and not all(live):
        raise ValueError('mixed live and preencoded fast images')
    if all(live):
        tensors = dict(target_actions=torch.from_numpy(np.stack([x['target_actions'] for x in samples])),
                       action_mask=torch.from_numpy(np.stack([x['action_mask'] for x in samples])),
                       reference_image=[x['reference_image'] for x in samples],
                       current_image=[x['current_image'] for x in samples])
    else:
        tensors = collate([{key: value for key, value in sample.items()
                            if key not in ("slow_messages", "slow_completion", "slow_anchor_id")}
                           for sample in samples])
    tensors["slow_messages"] = [sample["slow_messages"] for sample in samples]
    tensors["slow_completion"] = [sample["slow_completion"] for sample in samples]
    tensors["slow_anchor_id"] = [sample["slow_anchor_id"] for sample in samples]
    return tensors


def anchor_grouped_shuffle(rows: list[dict], rng: random.Random) -> list[dict]:
    """Shuffle slow anchors while keeping their fast windows consecutive."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["slow_anchor_id"], []).append(row)
    keys = list(groups)
    rng.shuffle(keys)
    result = []
    for key in keys:
        windows = list(groups[key])
        rng.shuffle(windows)
        result.extend(windows)
    return result


def anchor_grouped_batches(rows: list[dict], rng: random.Random) -> list[list[dict]]:
    """Return one batch per slow anchor, with windows and anchors shuffled."""
    ordered = anchor_grouped_shuffle(rows, rng)
    result: list[list[dict]] = []
    for row in ordered:
        if not result or result[-1][0]["slow_anchor_id"] != row["slow_anchor_id"]:
            result.append([])
        result[-1].append(row)
    return result


def _filtered(module: nn.Module, values: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(module.forward)
    if any(item.kind == inspect.Parameter.VAR_KEYWORD
           for item in signature.parameters.values()):
        return values
    return {key: value for key, value in values.items() if key in signature.parameters}


def _append_positions(position_ids: torch.Tensor | None, query_count: int) -> torch.Tensor | None:
    if position_ids is None:
        return None
    # Qwen VL mRoPE uses [3,B,L]; text query tokens advance every axis.
    last = position_ids[..., -1:]
    offsets = torch.arange(1, query_count + 1, device=position_ids.device,
                           dtype=position_ids.dtype)
    return torch.cat((position_ids, last + offsets), dim=-1)


def _query_positions(position_ids: torch.Tensor | None,
                     query_count: int) -> torch.Tensor | None:
    if position_ids is None:
        return None
    offsets = torch.arange(1, query_count + 1, device=position_ids.device,
                           dtype=position_ids.dtype)
    return position_ids[..., -1:] + offsets


def _cached_chunk_linear_attention(self, hidden_states, cache_params=None,
                                   attention_mask=None, cu_seqlens=None,
                                   cu_seqlens_cpu=None, **kwargs):
    """Run a multi-token query chunk from Qwen's cached recurrent state.

    Transformers 5.5.4 only takes its cached GatedDeltaNet branch when the
    new sequence length is one.  Four action expert queries must stay in one
    causal chunk so their mutual attention and gradients are preserved.
    """
    # The verl decoder forwards these keywords even for an unpacked FM
    # prefix/query.  Do not pass None to older Transformers forwards, and
    # never silently interpret packed sequences as one cached trajectory.
    if (cu_seqlens is not None or cu_seqlens_cpu is not None
            or any(kwargs.get(key) is not None
                   for key in ('cu_seq_lens_q', 'cu_seq_lens_k', 'seq_idx'))):
        raise ValueError("cached action-head queries require unpacked sequences")
    batch_size, seq_len, _ = hidden_states.shape
    has_prefix = (cache_params is not None
                  and cache_params.has_previous_state(self.layer_idx))
    commit_prefix = getattr(cache_params, '_action_head_commit_prefix', False)
    if (not has_prefix and not commit_prefix) or (seq_len == 1 and not commit_prefix):
        return self._action_head_original_forward(
            hidden_states, cache_params=cache_params,
            attention_mask=attention_mask, **kwargs)
    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask[:, :, None].to(
            hidden_states.dtype)
    recurrent_state = (cache_params.layers[self.layer_idx].recurrent_states
                       if has_prefix else None)
    conv_state = cache_params.layers[self.layer_idx].conv_states if has_prefix else None
    if commit_prefix and torch.is_grad_enabled():
        raise ValueError('canonical FM prefix cache must be built without gradients')
    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    weight = self.conv1d.weight.squeeze(1)
    joined = (torch.cat((conv_state, mixed_qkv), dim=-1)
              if has_prefix else mixed_qkv).to(weight.dtype)
    if commit_prefix:
        cache_params.update_conv_state(
            F.pad(joined, (self.conv_kernel_size - joined.shape[-1], 0)), self.layer_idx)
    if not has_prefix and self.causal_conv1d_fn is not None:
        mixed_qkv = self.causal_conv1d_fn(x=joined, weight=weight,
            bias=self.conv1d.bias, activation=self.activation)
    else:
        mixed_qkv = F.conv1d(
            joined, weight.unsqueeze(1), self.conv1d.bias,
            padding=0 if has_prefix else self.conv_kernel_size - 1,
            groups=joined.shape[1])
        mixed_qkv = mixed_qkv[:, :, -seq_len:] if has_prefix else mixed_qkv[:, :, :seq_len]
        mixed_qkv = F.silu(mixed_qkv)
    mixed_qkv = mixed_qkv.to(hidden_states.dtype).transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
    z = self.in_proj_z(hidden_states).reshape(
        batch_size, seq_len, -1, self.head_v_dim)
    beta = self.in_proj_b(hidden_states).sigmoid()
    g = -self.A_log.float().exp() * F.softplus(
        self.in_proj_a(hidden_states).float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        repeats = self.num_v_heads // self.num_k_heads
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)
    core, final_state = self.chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta,
        initial_state=recurrent_state, output_final_state=commit_prefix,
        use_qk_l2norm_in_kernel=True)
    if commit_prefix:
        cache_params.update_recurrent_state(final_state, self.layer_idx)
    core = core.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core = self.norm(core, z).reshape(batch_size, seq_len, -1)
    return self.out_proj(core)


def _enable_cached_query_chunks(backbone: nn.Module) -> None:
    for layer in backbone.layers:
        linear = getattr(layer, "linear_attn", None)
        if linear is None or hasattr(linear, "_action_head_original_forward"):
            continue
        linear._action_head_original_forward = linear.forward
        linear.forward = types.MethodType(_cached_chunk_linear_attention, linear)


class FrozenQwenQueryForward(nn.Module):
    """Prepare Qwen multimodal embeddings then run four differentiable queries."""

    def __init__(self, model: nn.Module, query_encoder: DualVlnQueryEncoder):
        super().__init__()
        self.model = model
        self.query_encoder = query_encoder
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        backbone = getattr(getattr(self.model, "model", None), "language_model", None)
        if backbone is None:
            raise RuntimeError("Qwen wrapper has no text backbone .model.language_model")
        self.frozen_query_backend = os.environ.get('ACTION_HEAD_FROZEN_QUERY_BACKEND', 'legacy_suffix')
        if self.frozen_query_backend not in ('legacy_suffix', 'query_only'):
            raise ValueError(f'unknown frozen query backend: {self.frozen_query_backend}')
        if self.frozen_query_backend == 'query_only':
            if (os.environ.get('ACTION_HEAD_SLOW_BATCH_MODE') == 'packed'
                    or os.environ.get('ACTION_HEAD_PROFILE_PACKED') == '1'):
                raise ValueError('query-only backend does not support packed slow anchors')
            from sft.frozen_query_backbone import install_frozen_query_backbone
            self.backend_report = install_frozen_query_backbone(backbone, query_encoder.query_count)
            if os.environ.get('ACTION_HEAD_PREFIX_REUSE', '0') == '1':
                from sft.frozen_prefix_reuse import install_prefix_reuse
                self.prefix_reuse = install_prefix_reuse(backbone)
        else:
            _enable_cached_query_chunks(backbone)
        if (os.environ.get("ACTION_HEAD_SLOW_BATCH_MODE") == "packed"
                or os.environ.get("ACTION_HEAD_PROFILE_PACKED") == "1"):
            from sft.packed_query_forward import install_packed_forward
            install_packed_forward(backbone)
        if (self.frozen_query_backend == 'legacy_suffix'
                and os.environ.get('ACTION_HEAD_QUERY_SUFFIX_BACKWARD', '0') == '1'):
            from sft.frozen_expert_grouped_mm import query_suffix_backward
            self._suffix_backward_context = query_suffix_backward(backbone, query_encoder.query_count)
            self._suffix_backward_context.__enter__()
        if (os.environ.get('ACTION_HEAD_FM_PREFIX_REUSE', '0') == '1'
                and os.environ.get('ACTION_HEAD_ROUTER_REPLAY') != 'R3'):
            # RL owns an update-local R3 cache in its worker. Only native SFT
            # may install a cache that persists across optimizer updates.
            # Reuse the RL FM-stage KV/GDN history cache instead of MoE-only reuse.
            if (self.frozen_query_backend != 'legacy_suffix'
                    or os.environ.get('ACTION_HEAD_PREFIX_REUSE', '0') == '1'
                    or os.environ.get('ACTION_HEAD_QUERY_SUFFIX_BACKWARD', '0') == '1'):
                raise ValueError('FM prefix reuse requires plain legacy_suffix modules only')
            from transformers import AutoTokenizer
            from action_head_rl.fm_prefix_cache import install_persistent_fm_prefix_cache
            path = getattr(self.model.config, '_name_or_path', '') or os.environ.get('SLOW_MODEL', '')
            if not os.path.isdir(path):
                raise ValueError(f'FM prefix reuse cannot resolve the slow tokenizer: {path!r}')
            tokenizer = AutoTokenizer.from_pretrained(path)
            marker = tokenizer.convert_tokens_to_ids('<|im_start|>')
            if marker is None or marker == tokenizer.unk_token_id:
                raise ValueError('slow tokenizer has no <|im_start|> role marker')
            install_persistent_fm_prefix_cache(self, marker)
        self._cached_key = None
        self._cached_prefix = None

    def prefix_embeds(self, inputs: dict[str, torch.Tensor], *, image_cache=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        input_ids = inputs["input_ids"]
        embeds = self.model.get_input_embeddings()(input_ids)
        multimodal = getattr(self.model, "model", None)
        if multimodal is None:
            raise RuntimeError("Qwen wrapper has no multimodal .model")
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            # In Transformers 5.5 Qwen3.6 places visual + placeholder logic
            # under wrapper.model, and get_image_features returns one tensor
            # per input image.  Mirror its native multimodal forward exactly.
            if image_cache is None:
                image_output = multimodal.get_image_features(
                    pixel_values, inputs["image_grid_thw"], return_dict=True)
                image_embeds = torch.cat(image_output.pooler_output, dim=0)
            else:
                image_embeds = image_cache.features(
                    multimodal, pixel_values, inputs["image_grid_thw"])
            image_embeds = image_embeds.to(device=embeds.device, dtype=embeds.dtype)
            mask_fn = multimodal.get_placeholder_mask
            image_mask = mask_fn(input_ids, inputs_embeds=embeds,
                                 image_features=image_embeds)
            image_mask = image_mask[0] if isinstance(image_mask, tuple) else image_mask
            embeds = embeds.masked_scatter(image_mask, image_embeds)
        mask = inputs.get("attention_mask")
        if mask is None:
            mask = torch.ones(input_ids.shape, device=input_ids.device, dtype=torch.long)
        position_ids = None
        rope = getattr(multimodal, "compute_3d_position_ids", None)
        if rope is not None:
            position_ids = rope(
                input_ids=input_ids, image_grid_thw=inputs.get("image_grid_thw"),
                video_grid_thw=None, inputs_embeds=embeds,
                attention_mask=mask, past_key_values=None,
                mm_token_type_ids=inputs.get("mm_token_type_ids"))
        return embeds, mask, position_ids

    def _backbone(self) -> nn.Module:
        backbone = getattr(getattr(self.model, "model", None), "language_model", None)
        if backbone is None:
            raise RuntimeError("Qwen wrapper has no text backbone .model.language_model")
        return backbone

    def forward_full(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.frozen_query_backend == 'query_only':
            if inputs['input_ids'].shape[0] != 1:
                raise ValueError('query-only backend requires unpadded batch-one slow anchors')
            if inputs.get('attention_mask') is not None and not bool(inputs['attention_mask'].all()):
                raise ValueError('query-only backend cannot silently discard padded tokens')
        if getattr(self, '_persistent_fm_prefix', False):
            cache = self._fm_prefix_cache
            queries = cache.forward(self, inputs, None)
            if cache.stats['calls'] <= 3 or cache.stats['calls'] % 10 == 0:
                print(json.dumps(dict(event='fm_prefix_reuse', rank=int(os.environ.get('RANK', '0')),
                    scope='persistent_attention_gdn_history', native_routing=True,
                    cached_tokens=cache.cache.get_seq_length(), **cache.stats)), flush=True)
            return queries
        embeds, mask, positions = self.prefix_embeds(inputs)
        if self.frozen_query_backend == 'query_only' and embeds.requires_grad:
            raise ValueError('query-only backend cannot discard gradients of a trainable prefix')
        queries = self.query_encoder(
            self._backbone(), prefix_embeds=embeds, attention_mask=mask, use_cache=False,
            position_ids=_append_positions(positions, self.query_encoder.QUERY_COUNT))
        # device_map="auto" places Qwen's final text layers on a different
        # device from the small trainable head.  This copy is differentiable,
        # so it preserves the gradient path to the four query embeddings.
        cache = getattr(self, 'prefix_reuse', None)
        if cache is not None:
            self._prefix_reuse_steps = getattr(self, '_prefix_reuse_steps', 0) + 1
            if self._prefix_reuse_steps <= 3 or self._prefix_reuse_steps % 10 == 0:
                print(json.dumps(dict(event='slow_anchor_prefix_reuse', rank=int(os.environ.get('RANK', '0')),
                                      step_calls=self._prefix_reuse_steps, **cache.report())), flush=True)
        return queries.to(device=self.query_encoder.query_embeddings.device)

    def forward_packed(self, anchors):
        """Encode live images and MRoPE separately, then run one text forward."""
        from sft.packed_query_forward import PackedLayout
        segments, positions = [], []
        for inputs in anchors:
            embeds, mask, position = self.prefix_embeds(inputs)
            if position is None or not bool(mask.all()):
                raise ValueError('packed anchors require unpadded inputs and explicit MRoPE')
            segment, _ = self.query_encoder.append(embeds, mask)
            segments.append(segment)
            positions.append(_append_positions(position, self.query_encoder.query_count))
        joined = torch.cat(segments, dim=1)
        layout = PackedLayout.build([x.shape[1] for x in segments],
                                    self.query_encoder.query_count, joined.device)
        output = self._backbone()(inputs_embeds=joined, position_ids=torch.cat(positions, dim=-1),
                                  use_cache=False, return_dict=True, packed_layout=layout)
        queries = output.last_hidden_state.index_select(1, layout.query_indices)
        return queries.reshape(len(anchors), self.query_encoder.query_count, -1).to(
            device=self.query_encoder.query_embeddings.device)

    @torch.no_grad()
    def ensure_prefix_cache(self, inputs: dict[str, torch.Tensor],
                            cache_key: str) -> bool:
        if self._cached_key == cache_key and self._cached_prefix is not None:
            return True
        embeds, mask, positions = self.prefix_embeds(inputs)
        outputs = self._backbone()(
            inputs_embeds=embeds, attention_mask=mask,
            position_ids=positions, use_cache=True, return_dict=True)
        if outputs.past_key_values is None:
            raise RuntimeError("Qwen prefix forward did not return a cache")
        self._cached_key = cache_key
        self._cached_prefix = (outputs.past_key_values, mask.detach(),
                               None if positions is None else positions.detach())
        return False

    def forward_cached_chunk(self, query_dtype: torch.dtype | None = None) -> torch.Tensor:
        if self._cached_prefix is None:
            raise RuntimeError("ensure_prefix_cache must run before forward_cached")
        prefix_cache, prefix_mask, positions = self._cached_prefix
        working_cache = copy.deepcopy(prefix_cache)
        count = self.query_encoder.QUERY_COUNT
        query = self.query_encoder.query_embeddings.to(
            device=prefix_mask.device, dtype=query_dtype).unsqueeze(0)
        query_mask = torch.ones(
            (query.shape[0], count), device=prefix_mask.device,
            dtype=prefix_mask.dtype)
        outputs = self._backbone()(
            inputs_embeds=query,
            attention_mask=torch.cat((prefix_mask, query_mask), dim=1),
            position_ids=_query_positions(positions, count),
            past_key_values=working_cache, use_cache=False, return_dict=True)
        result = outputs.last_hidden_state
        expected = (1, count, self.query_encoder.hidden_size)
        if tuple(result.shape) != expected:
            raise RuntimeError(
                f"cached query output must have shape {expected}, got {tuple(result.shape)}")
        return result.to(device=self.query_encoder.query_embeddings.device)

    def forward_cached(self) -> torch.Tensor:
        """Decode four queries through Qwen's supported one-token cache path."""
        if self._cached_prefix is None:
            raise RuntimeError("ensure_prefix_cache must run before forward_cached")
        prefix_cache, prefix_mask, positions = self._cached_prefix
        working_cache = copy.deepcopy(prefix_cache)
        count = self.query_encoder.QUERY_COUNT
        queries = self.query_encoder.query_embeddings.to(
            device=prefix_mask.device).unsqueeze(0)
        outputs = []
        for index in range(count):
            step_mask = torch.ones(
                (1, index + 1), device=prefix_mask.device,
                dtype=prefix_mask.dtype)
            result = self._backbone()(
                inputs_embeds=queries[:, index:index + 1],
                attention_mask=torch.cat((prefix_mask, step_mask), dim=1),
                position_ids=_query_positions(positions, count)[..., index:index + 1],
                past_key_values=working_cache, use_cache=False, return_dict=True)
            outputs.append(result.last_hidden_state)
        result = torch.cat(outputs, dim=1)
        expected = (1, count, self.query_encoder.hidden_size)
        if tuple(result.shape) != expected:
            raise RuntimeError(
                f"cached query output must have shape {expected}, got {tuple(result.shape)}")
        return result.to(device=self.query_encoder.query_embeddings.device)

    def forward(self, inputs: dict[str, torch.Tensor],
                cache_key: str) -> torch.Tensor:
        self.ensure_prefix_cache(inputs, cache_key)
        return self.forward_cached()


def _slow_template_messages(messages, *, copy_native_images=True):
    """Convert persisted OpenAI ``image_url`` items to Qwen chat-template images."""
    converted, images = [], []
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            converted.append(message)
            continue
        parts = []
        for item in content:
            if item.get("type") == "image" and isinstance(item.get("image"), Image.Image):
                image = (item["image"] if not copy_native_images and item["image"].mode == "RGB"
                         else item["image"].convert("RGB"))
                images.append(image)
                parts.append({"type": "image", "image": image})
                continue
            if item.get("type") != "image_url":
                parts.append(item)
                continue
            url = item["image_url"]["url"]
            if url.startswith("file://"):
                image = Image.open(url[7:]).convert("RGB")
            elif url.startswith("data:image/") and ";base64," in url:
                encoded = url.split(";base64,", 1)[1]
                image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
            else:
                raise ValueError(
                    "joint SFT supports file:// or base64 data: slow images")
            images.append(image)
            # Qwen's template emits its image placeholder only for ``image``;
            # retaining OpenAI's ``image_url`` silently drops every placeholder.
            parts.append({"type": "image", "image": image})
        converted.append({**message, "content": parts})
    if not images:
        raise ValueError("slow_messages contains no RGB images")
    return converted, images


def build_online_prefix_text(processor, messages, completion):
    """Rebuild the exact vLLM prefix immediately before its first query.

    Online generation starts from ``add_generation_prompt=True`` with thinking
    disabled and then appends the generated Tool call.  Treating the completed
    response as an assistant chat message would add ``<|im_end|>`` after the
    Tool call and would therefore change both causal context and mRoPE offsets.
    """
    clean_completion, _ = strip_query_tail(completion)
    if not clean_completion.rstrip().endswith("</tool_call>"):
        raise ValueError("slow completion must end at a closed Tool call")
    prefix = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    return prefix + clean_completion


def build_slow_inputs(processor, messages, completion, device):
    template_messages, images = _slow_template_messages(messages)
    prompt = build_online_prefix_text(
        processor, template_messages, completion)
    values = processor(text=[prompt], images=images, padding=True, return_tensors="pt")
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in values.items()}


def build_slow_batch_inputs(processor, messages_batch, completions, device):
    """Batch independently rendered Qwen-VL prefixes without mixing images.

    Each slow anchor is first processed exactly like the historical batch-one
    path. Text-like tensors are then left padded, while Qwen's flattened image
    patches and per-image grids are concatenated in anchor order, matching the
    official Qwen-VL fine-tuning collator convention.
    """
    if len(messages_batch) != len(completions) or not messages_batch:
        raise ValueError("messages_batch and completions must have equal non-zero length")
    singles = [
        build_slow_inputs(processor, messages, completion, torch.device("cpu"))
        for messages, completion in zip(messages_batch, completions)
    ]

    def left_pad(key, value):
        tensors = [item[key] for item in singles]
        width = max(tensor.shape[-1] for tensor in tensors)
        return torch.cat([
            F.pad(tensor, (width - tensor.shape[-1], 0), value=value)
            for tensor in tensors
        ], dim=0)

    pad_token_id = int(processor.tokenizer.pad_token_id)
    result = {
        "input_ids": left_pad("input_ids", pad_token_id),
        "attention_mask": left_pad("attention_mask", 0),
    }
    if all("mm_token_type_ids" in item for item in singles):
        result["mm_token_type_ids"] = left_pad("mm_token_type_ids", 0)
    for key in ("pixel_values", "image_grid_thw"):
        present = [item[key] for item in singles if key in item]
        if present:
            if len(present) != len(singles):
                raise ValueError(f"mixed presence of {key!r} across slow anchors")
            result[key] = torch.cat(present, dim=0)
    # Preserve flattened multimodal boundaries for the optional numerical
    # parity gate. These small tensors are ignored during ordinary training.
    result["slow_image_count_per_anchor"] = torch.tensor([
        item.get("image_grid_thw", torch.empty(0)).shape[0] for item in singles
    ], dtype=torch.long)
    result["slow_pixel_count_per_anchor"] = torch.tensor([
        item.get("pixel_values", torch.empty(0)).shape[0] for item in singles
    ], dtype=torch.long)
    return {key: value.to(device) for key, value in result.items()}


def parameter_delta(parameters, initial):
    return float(sum((parameter.detach().float().cpu() - before).pow(2).sum()
                     for parameter, before in zip(parameters, initial)).sqrt())


def frozen_parameter_snapshot(module: nn.Module) -> list[tuple[int, int, int, tuple[float, ...]]]:
    """Audit a frozen backbone without cloning all 35B parameters to CPU."""
    parameters = list(module.parameters())
    sample_stride = max(1, len(parameters) // 32)
    snapshot = []
    for parameter_index, parameter in enumerate(parameters):
        flat = parameter.detach().reshape(-1)
        values = ()
        if flat.numel() and parameter_index % sample_stride == 0:
            indices = sorted({0, flat.numel() // 2, flat.numel() - 1})
            values = tuple(float(flat[index].float().cpu()) for index in indices)
        snapshot.append((id(parameter), parameter._version, parameter.data_ptr(), values))
    return snapshot


def audit_frozen_parameters(module: nn.Module,
                            before: list[tuple[int, int, int, tuple[float, ...]]],
                            optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    parameters = list(module.parameters())
    optimizer_ids = {id(parameter) for group in optimizer.param_groups
                     for parameter in group["params"]}
    after = frozen_parameter_snapshot(module)
    return {
        "parameter_count": len(parameters),
        "requires_grad_count": sum(parameter.requires_grad for parameter in parameters),
        "optimizer_overlap_count": sum(id(parameter) in optimizer_ids for parameter in parameters),
        "non_none_grad_count": sum(parameter.grad is not None for parameter in parameters),
        "changed_snapshot_count": sum(old != new for old, new in zip(before, after)),
        "snapshot_policy": "all identity/version/storage + 32 tensors first/middle/last value",
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate_joint(frozen, head, processor, rows, device, dtype, seed):
    """Fixed-noise validation loss plus masked waypoint ADE/FDE."""
    if not rows:
        return None
    head.eval()
    losses, ade_values, fde_values = [], [], []
    dataset = JointBundleDataset(rows)
    indexed_rows = [{**row, "_evaluation_index": index}
                    for index, row in enumerate(rows)]
    groups: dict[str, list[dict]] = {}
    for row in indexed_rows:
        groups.setdefault(row["slow_anchor_id"], []).append(row)
    for group in groups.values():
        samples = [dataset[row["_evaluation_index"]] for row in group]
        batch = joint_collate(samples)
        anchor_ids = batch.pop("slow_anchor_id")
        slow_messages = batch.pop("slow_messages")
        slow_completions = batch.pop("slow_completion")
        if len(set(anchor_ids)) != 1:
            raise RuntimeError("validation anchor batch mixed slow prefixes")
        slow = build_slow_inputs(processor, slow_messages[0],
                                 slow_completions[0], device)
        batch = move(batch, device, dtype)
        queries = frozen.forward_full(slow).expand(len(group), -1, -1)
        condition = head.encode_condition(
            queries, batch["reference_visual_tokens"], batch["current_visual_tokens"],
            batch["reference_visual_mask"], batch["current_visual_mask"])
        # Random initial noise per row, matching online sampling. Seeded by the
        # per-row evaluation index (offset from the loss generator below) so
        # validation ADE/FDE stay comparable across checkpoints and the decode
        # no longer masks under-denoising at the configured inference steps.
        initial_noise = torch.empty_like(batch["target_actions"])
        for local_index, row in enumerate(group):
            decode_generator = torch.Generator(device=device).manual_seed(
                seed + row["_evaluation_index"] + 1_000_000)
            initial_noise[local_index] = torch.randn(
                batch["target_actions"][local_index].shape, device=device,
                dtype=batch["target_actions"].dtype, generator=decode_generator)
        prediction = head.sample(condition=condition, initial_noise=initial_noise)
        distance = (prediction - batch["target_actions"]).norm(dim=-1)
        for local_index, row in enumerate(group):
            generator = torch.Generator(device=device).manual_seed(
                seed + row["_evaluation_index"])
            loss = head.flow_matching_loss(
                condition[local_index:local_index + 1],
                batch["target_actions"][local_index:local_index + 1],
                batch["action_mask"][local_index:local_index + 1], generator)
            valid = batch["action_mask"][local_index]
            ade_values.append(float(
                distance[local_index][valid].mean().float().cpu()))
            last = int(valid.sum().item()) - 1
            fde_values.append(float(distance[local_index, last].float().cpu()))
            losses.append(float(loss.float().cpu()))
    head.train()
    return {"flow_loss": float(np.mean(losses)), "ade": float(np.mean(ade_values)),
            "fde": float(np.mean(fde_values)), "noise_policy": "random_per_row"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Initial fast/slow Action Head.")
    parser.add_argument("--query-checkpoint", default="", help="Optional four-query initialization.")
    parser.add_argument("--slow-model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Deprecated; each optimizer microbatch is one complete slow-anchor group.")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--profile-timing", action="store_true",
                        help="Synchronize CUDA at phase boundaries and report wall time.")
    parser.add_argument("--require-fast-path", action="store_true",
                        help="Fail unless Qwen linear-attention CUDA kernels are importable.")
    parser.add_argument("--cache-equivalence-rtol", type=float, default=0.05,
                        help="Maximum relative RMS error for full vs cached queries.")
    args = parser.parse_args()
    if args.batch_size != 1:
        raise ValueError("--batch-size remains 1 because batching is automatic per slow anchor")
    if min(args.steps, args.grad_accum) <= 0:
        raise ValueError("steps and grad-accum must be positive")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    train_rows, validation_rows = split_rows(rows)
    if args.smoke:
        train_rows, validation_rows, args.steps = train_rows[:2], validation_rows[:1], min(args.steps, 2)
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe
    fast_path_available = bool(modeling_qwen3_5_moe.is_fast_path_available)
    if args.require_fast_path and not fast_path_available:
        raise RuntimeError(
            "Qwen3.5/3.6 linear-attention fast path is unavailable; both "
            "flash-linear-attention and causal-conv1d must import successfully")
    wall_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.slow_model)
    model_load_start = time.perf_counter()
    qwen = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        args.slow_model, dtype=dtype, device_map="auto", low_cpu_mem_usage=True,
        attn_implementation="sdpa")
    synchronize(device)
    model_load_seconds = time.perf_counter() - model_load_start
    hidden = int(qwen.config.text_config.hidden_size)
    query = (DualVlnQueryEncoder.from_pretrained(args.query_checkpoint)
             if args.query_checkpoint else DualVlnQueryEncoder(hidden))
    if query.hidden_size != hidden:
        raise ValueError("query checkpoint hidden size does not match slow model")
    query = query.to(device=device, dtype=dtype)
    frozen = FrozenQwenQueryForward(qwen, query)
    head = Qwen36FastSlowActionHead.from_pretrained(args.checkpoint).to(device=device, dtype=dtype).train()
    optimizer = torch.optim.AdamW(list(query.parameters()) + list(head.parameters()), lr=args.lr)
    qwen_before = frozen_parameter_snapshot(qwen)
    query_before = [p.detach().float().cpu().clone() for p in query.parameters()]
    head_before = [p.detach().float().cpu().clone() for p in head.parameters()]
    row_rng = random.Random(args.seed)
    def new_iterator():
        groups = anchor_grouped_batches(train_rows, row_rng)
        return iter([[JointBundleDataset(group)[index]
                      for index in range(len(group))] for group in groups])
    iterator = new_iterator(); losses = []
    cache_equivalence = {
        "enabled": False,
        "reason": "rejected: cached query chunk failed the full-forward equivalence gate",
        "training_path": "one exact full causal Qwen pass per slow anchor",
    }
    fixed_eval_seed = args.seed + 1_000_000
    initial_validation = evaluate_joint(
        frozen, head, processor, validation_rows, device, dtype, fixed_eval_seed)
    if initial_validation:
        print(json.dumps({"step": 0, "validation": initial_validation}), flush=True)
    timing_totals = {key: 0.0 for key in
                     ("input", "prefix_cache", "query_forward", "qwen_forward",
                      "head_forward", "backward", "optimizer", "step_total")}
    cache_hits = 0
    windows_processed = 0
    profile_timing = args.profile_timing or args.smoke
    train_start = time.perf_counter()
    for step in range(args.steps):
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True); total = 0.0
        for micro in range(args.grad_accum):
            phase_start = time.perf_counter()
            try: samples = next(iterator)
            except StopIteration: iterator = new_iterator(); samples = next(iterator)
            batch = joint_collate(samples)
            anchor_ids = batch.pop("slow_anchor_id")
            slow_messages = batch.pop("slow_messages")
            slow_completions = batch.pop("slow_completion")
            if len(set(anchor_ids)) != 1:
                raise RuntimeError("training anchor batch mixed slow prefixes")
            slow = build_slow_inputs(processor, slow_messages[0],
                                     slow_completions[0], device)
            batch = move(batch, device, dtype)
            window_count = len(anchor_ids)
            windows_processed += window_count
            if profile_timing: synchronize(device)
            timing_totals["input"] += time.perf_counter() - phase_start
            qwen_start = time.perf_counter()
            phase_start = qwen_start
            queries = frozen.forward_full(slow).expand(window_count, -1, -1)
            if profile_timing: synchronize(device)
            timing_totals["query_forward"] += time.perf_counter() - phase_start
            timing_totals["qwen_forward"] += time.perf_counter() - qwen_start
            phase_start = time.perf_counter()
            condition = head.encode_condition(
                queries, batch["reference_visual_tokens"],
                batch["current_visual_tokens"], batch["reference_visual_mask"],
                batch["current_visual_mask"])
            window_losses = []
            for window_index in range(window_count):
                generator = torch.Generator(device=device).manual_seed(
                    args.seed + step * args.grad_accum * 1000 + micro * 1000
                    + window_index)
                window_losses.append(head.flow_matching_loss(
                    condition[window_index:window_index + 1],
                    batch["target_actions"][window_index:window_index + 1],
                    batch["action_mask"][window_index:window_index + 1], generator))
            loss = torch.stack(window_losses).mean()
            if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss at step {step + 1}")
            if profile_timing: synchronize(device)
            timing_totals["head_forward"] += time.perf_counter() - phase_start
            phase_start = time.perf_counter()
            (loss / args.grad_accum).backward(); total += float(loss.detach().float().cpu())
            if profile_timing: synchronize(device)
            timing_totals["backward"] += time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        optimizer.step()
        if profile_timing: synchronize(device)
        timing_totals["optimizer"] += time.perf_counter() - phase_start
        losses.append(total / args.grad_accum)
        step_seconds = time.perf_counter() - step_start
        timing_totals["step_total"] += step_seconds
        print(json.dumps({"step": step + 1, "train_loss": losses[-1],
                          "step_seconds": step_seconds,
                          "windows": window_count}), flush=True)
    train_seconds = time.perf_counter() - train_start
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    query.save_pretrained(output / "dualvln_query")
    head.save_pretrained(output / "action_head")
    qwen_audit = audit_frozen_parameters(qwen, qwen_before, optimizer)
    qwen_unchanged = all(qwen_audit[key] == 0 for key in
                         ("requires_grad_count", "optimizer_overlap_count",
                          "non_none_grad_count", "changed_snapshot_count"))
    query_delta = parameter_delta(list(query.parameters()), query_before)
    head_delta = parameter_delta(list(head.parameters()), head_before)
    final_validation = evaluate_joint(
        frozen, head, processor, validation_rows, device, dtype, fixed_eval_seed)
    timing = {**timing_totals, "model_load": model_load_seconds,
              "train": train_seconds, "wall": time.perf_counter() - wall_start,
              "profile_synchronized": profile_timing}
    timing["windows_processed"] = windows_processed
    timing["seconds_per_window"] = train_seconds / max(windows_processed, 1)
    report = {"schema": "joint_fm_head_sft_report_v4", "ok": bool(np.isfinite(losses).all() and qwen_unchanged and query_delta > 0 and head_delta > 0), "manifest": str(Path(args.manifest).resolve()), "slow_model": args.slow_model, "steps": args.steps, "grad_accum": args.grad_accum, "first_loss": losses[0], "last_loss": losses[-1], "initial_validation": initial_validation, "final_validation": final_validation, "qwen_parameter_delta_l2": 0.0 if qwen_unchanged else None, "qwen_audit": qwen_audit, "qwen_linear_attention_fast_path": fast_path_available, "prefix_cache_equivalence": cache_equivalence, "prefix_cache_hits": cache_hits, "timing_seconds": timing, "query_parameter_delta_l2": query_delta, "action_head_parameter_delta_l2": head_delta, "window_weighting": "equal within each slow anchor; equal anchor groups per optimizer microbatch", "boundary": "Qwen vision+text frozen; one exact full causal pass shared by all fast windows under a slow anchor; only four query embeddings and FM head optimized"}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["ok"]: raise SystemExit(2)


if __name__ == "__main__":
    main()
