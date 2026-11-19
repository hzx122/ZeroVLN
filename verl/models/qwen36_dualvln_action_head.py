"""FSDP/ZeRO-compatible joint Qwen3.6 latent-query and FM-head model."""
from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn

from models.dualvln_query_encoder import DualVlnQueryEncoder
from models.qwen36_fast_slow_action_head import Qwen36FastSlowActionHead
from sft.joint_fm_head_sft import FrozenQwenQueryForward


def globally_weighted_window_loss(
        losses: torch.Tensor, window_valid: torch.Tensor | None = None,
        global_window_count: torch.Tensor | None = None) -> torch.Tensor:
    """Return a DDP/ZeRO loss whose averaged gradient is window-equal."""
    if losses.ndim != 1 or not losses.numel():
        raise ValueError("losses must be a non-empty one-dimensional tensor")
    valid = (window_valid.reshape(-1).to(device=losses.device, dtype=torch.float32)
             if window_valid is not None else torch.ones_like(losses, dtype=torch.float32))
    if valid.numel() != losses.numel():
        raise ValueError("window_valid must contain one value per window")
    local_count = valid.sum()
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if global_window_count is None:
            torch.distributed.all_reduce(local_count, op=torch.distributed.ReduceOp.SUM)
        world_size = torch.distributed.get_world_size()
    if global_window_count is not None:
        local_count = global_window_count.to(device=losses.device, dtype=torch.float32)
    return (losses * valid).sum() * (world_size / local_count.clamp_min(1.0))


class Qwen36DualVlnActionHead(nn.Module):
    """Freeze Qwen and optimize only latent queries plus the FM head."""

    def __init__(self, qwen: nn.Module, query_encoder: DualVlnQueryEncoder,
                 action_head: Qwen36FastSlowActionHead, *, fast_visual: nn.Module | None = None,
                 fast_visual_backbone: str = ""):
        super().__init__()
        self.query_forward = FrozenQwenQueryForward(qwen, query_encoder)
        self.action_head = action_head
        self.fast_visual = fast_visual
        self.fast_visual_backbone = fast_visual_backbone
        self._batch_parity_checked = False

    def can_generate(self) -> bool:
        """Expose the HuggingFace compatibility hook used by VERL checkpoints.

        This training wrapper does not implement ``generate``; rollout
        generation is served by the separate vLLM actor.
        """
        return False

    @staticmethod
    def _slice_slow_anchor(slow: dict[str, torch.Tensor], index: int,
                           image_counts: torch.Tensor,
                           pixel_counts: torch.Tensor) -> dict[str, torch.Tensor]:
        """Reconstruct the historical unpadded batch-one input for one anchor."""
        mask = slow["attention_mask"][index]
        start = int(torch.nonzero(mask, as_tuple=False)[0].item())
        result = {
            "input_ids": slow["input_ids"][index:index + 1, start:],
            "attention_mask": slow["attention_mask"][index:index + 1, start:],
        }
        if "mm_token_type_ids" in slow:
            result["mm_token_type_ids"] = slow["mm_token_type_ids"][
                index:index + 1, start:]
        image_start = int(image_counts[:index].sum().item())
        image_end = image_start + int(image_counts[index].item())
        pixel_start = int(pixel_counts[:index].sum().item())
        pixel_end = pixel_start + int(pixel_counts[index].item())
        if "image_grid_thw" in slow:
            result["image_grid_thw"] = slow["image_grid_thw"][image_start:image_end]
        if "pixel_values" in slow:
            result["pixel_values"] = slow["pixel_values"][pixel_start:pixel_end]
        return result

    def _check_batch_parity(self, slow: dict[str, torch.Tensor],
                            batched: torch.Tensor, image_counts: torch.Tensor,
                            pixel_counts: torch.Tensor) -> None:
        singles = []
        with torch.no_grad():
            for index in range(batched.shape[0]):
                item = self._slice_slow_anchor(
                    slow, index, image_counts, pixel_counts)
                singles.append(self.query_forward.forward_full(item))
        reference = torch.cat(singles, dim=0).to(dtype=torch.float32)
        candidate = batched.detach().to(dtype=torch.float32)
        difference = candidate - reference
        max_abs = float(difference.abs().max().item())
        mean_abs = float(difference.abs().mean().item())
        cosine = float(torch.nn.functional.cosine_similarity(
            candidate.reshape(candidate.shape[0], -1),
            reference.reshape(reference.shape[0], -1), dim=-1).min().item())
        tolerance = float(os.environ.get("ACTION_HEAD_BATCH_PARITY_MAX_ABS", "0.1"))
        print({"event": "action_head_batch_parity", "anchors": batched.shape[0],
               "max_abs": max_abs, "mean_abs": mean_abs,
               "min_cosine": cosine, "max_abs_tolerance": tolerance}, flush=True)
        if not torch.isfinite(difference).all() or max_abs > tolerance or cosine < 0.999:
            raise RuntimeError("batched slow-anchor query output failed batch-one parity gate")
        self._batch_parity_checked = True

    def _forward_slow_anchors(self, slow: dict[str, torch.Tensor],
                              image_counts: torch.Tensor | None,
                              pixel_counts: torch.Tensor | None) -> torch.Tensor:
        anchor_count = slow["input_ids"].shape[0]
        skip_every = int(os.environ.get("ACTION_HEAD_GC_SKIP_EVERY", "0"))
        if skip_every:
            from sft.selective_checkpointing import SelectiveCheckpointing, shared_token_budget
            if not hasattr(self, "_selective_checkpointing"):
                self._selective_checkpointing = SelectiveCheckpointing(
                    self.qwen.model.language_model.layers, skip_every,
                    int(os.environ.get("ACTION_HEAD_GC_MAX_TOKENS", "5120")))
            # Count padded tokens across every anchor, including latent queries.
            # This conservatively bounds both vectorized and sequential B>1.
            tokens = slow["input_ids"].numel() + anchor_count * self.query_encoder.query_count
            tokens = shared_token_budget(tokens, slow["input_ids"].device)
            skipped = self._selective_checkpointing.apply(tokens=tokens, training=self.training)
            if self.training and int(os.environ.get("RANK", "0")) == 0:
                print({"event": "action_head_selective_checkpointing",
                       "global_max_tokens": tokens, "skipped_layers": skipped,
                       "max_tokens": self._selective_checkpointing.max_tokens}, flush=True)
        mode = os.environ.get("ACTION_HEAD_SLOW_BATCH_MODE", "sequential")
        if anchor_count == 1 or mode == "vectorized":
            return self.query_forward.forward_full(slow)
        if mode not in ("sequential", "packed"):
            raise ValueError(f"unknown ACTION_HEAD_SLOW_BATCH_MODE={mode!r}")
        if image_counts is None or pixel_counts is None:
            raise RuntimeError("sequential slow-anchor mode requires multimodal boundaries")
        anchors = [self._slice_slow_anchor(slow, index, image_counts.reshape(-1),
                                          pixel_counts.reshape(-1)) for index in range(anchor_count)]
        if mode == "packed":
            return self.query_forward.forward_packed(anchors)
        queries = [self.query_forward.forward_full(anchor) for anchor in anchors]
        if not self._batch_parity_checked:
            print({"event": "action_head_batch_parity", "anchors": anchor_count,
                   "mode": "sequential_historical_batch_one", "exact_by_construction": True},
                  flush=True)
            self._batch_parity_checked = True
        return torch.cat(queries, dim=0)

    @property
    def qwen(self) -> nn.Module:
        return self.query_forward.model

    @property
    def query_encoder(self) -> DualVlnQueryEncoder:
        return self.query_forward.query_encoder

    def trainable_parameters(self):
        return [*self.query_encoder.parameters(), *self.action_head.parameters()]

    def _encode_condition(self, slow, reference, current, reference_mask, current_mask):
        return self.action_head.encode_condition(
            slow, reference, current, reference_mask, current_mask)

    def forward(self, *, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                pixel_values: torch.Tensor | None = None,
                image_grid_thw: torch.Tensor | None = None,
                mm_token_type_ids: torch.Tensor | None = None,
                cached_visual_tokens: torch.Tensor | None = None,
                fast_pixel_values: torch.Tensor | None = None,
                fast_image_grid_thw: torch.Tensor | None = None,
                fast_reference_image_index: torch.Tensor | None = None,
                fast_current_image_index: torch.Tensor | None = None,
                reference_visual_tokens: torch.Tensor | None = None,
                current_visual_tokens: torch.Tensor | None = None,
                reference_visual_mask: torch.Tensor | None = None,
                current_visual_mask: torch.Tensor | None = None,
                target_actions: torch.Tensor | None = None,
                action_mask: torch.Tensor | None = None,
                window_anchor_index: torch.Tensor | None = None,
                window_index_within_anchor: torch.Tensor | None = None,
                anchor_valid: torch.Tensor | None = None,
                noise_seed: torch.Tensor | None = None,
                slow_image_count_per_anchor: torch.Tensor | None = None,
                slow_pixel_count_per_anchor: torch.Tensor | None = None,
                compute_metrics: bool = False,
                global_window_count: torch.Tensor | None = None,
                grpo_query_routes: torch.Tensor | None = None,
                grpo_all_routes: torch.Tensor | None = None,
                grpo_all_positions: torch.Tensor | None = None,
                grpo_train_scope: str | None = None,
                grpo_states: torch.Tensor | None = None,
                grpo_successors: torch.Tensor | None = None,
                grpo_timesteps: torch.Tensor | None = None,
                grpo_signed_dt: float | None = None,
                grpo_noise_scales: torch.Tensor | None = None,
                grpo_var_floor: float | None = None,
                grpo_chunk_anchor_index: torch.Tensor | None = None,
                grpo_routes_already_installed: bool = False,
                grpo_text_response_mask: torch.Tensor | None = None,
                grpo_text_single_token_target_ids: torch.Tensor | None = None,
                grpo_text_trajectory_targets: torch.Tensor | None = None,
                grpo_text_temperature: float = 1.0,
                grpo_joint_text_response_mask: torch.Tensor | None = None,
                grpo_executable_policy: torch.Tensor | None = None,
                **_: Any) -> dict[str, torch.Tensor]:
        slow = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
            "cached_visual_tokens": cached_visual_tokens,
        }
        slow = {key: value for key, value in slow.items() if value is not None}
        if grpo_executable_policy is not None and grpo_text_response_mask is None and grpo_joint_text_response_mask is None:
            raise ValueError("executable policy metadata requires an actual text branch")
        if grpo_text_single_token_target_ids is not None and grpo_text_response_mask is None:
            raise ValueError("single-token causal targets require the explicit text branch")
        if grpo_joint_text_response_mask is not None and (
                grpo_train_scope != "full" or grpo_all_routes is None
                or grpo_states is None or grpo_text_response_mask is not None):
            raise ValueError("joint text/FM forward requires Full R3 FM inputs")
        if grpo_text_response_mask is not None:
            from action_head_rl.model import recompute_text_logprobs

            if grpo_train_scope != "full":
                raise ValueError("language policy gradients require full train scope")
            if grpo_states is not None:
                raise ValueError("text decisions must be separate from FM chunk records")
            logprobs = recompute_text_logprobs(
                self.query_forward, slow, response_mask=grpo_text_response_mask,
                all_routes=grpo_all_routes, all_positions=grpo_all_positions,
                temperature=grpo_text_temperature,
                routes_already_installed=grpo_routes_already_installed,
                trajectory_targets=grpo_text_trajectory_targets,
                single_token_target_ids=grpo_text_single_token_target_ids,
                executable_metadata=grpo_executable_policy,
            )
            return {"text_logprobs": logprobs}
        if grpo_all_routes is not None:
            # Keep the complete RL computation inside this root forward so
            # composable FSDP2 unshards root-owned vision/query/head weights
            # and installs the corresponding backward hooks.
            from action_head_rl.config import TrainScope
            from action_head_rl.model import recompute_query_hidden
            from action_head_rl.sde import recompute_reverse_logprob

            joint_text = {}
            if grpo_joint_text_response_mask is not None:
                from action_head_rl.model import recompute_joint_outputs
                queries, text_logprobs = recompute_joint_outputs(
                    self.query_forward, slow, response_mask=grpo_joint_text_response_mask,
                    all_routes=grpo_all_routes, all_positions=grpo_all_positions,
                    temperature=grpo_text_temperature,
                    routes_already_installed=grpo_routes_already_installed,
                    executable_metadata=grpo_executable_policy)
                joint_text["text_logprobs"] = text_logprobs
            else:
                queries = recompute_query_hidden(
                    self.query_forward, slow,
                    scope=TrainScope(grpo_train_scope or "head_query"),
                    query_routes=grpo_query_routes,
                    all_routes=grpo_all_routes,
                    all_positions=grpo_all_positions,
                    routes_already_installed=grpo_routes_already_installed,
                )
            if reference_visual_tokens is None or current_visual_tokens is None:
                raise ValueError("GRPO forward requires visual condition tokens")
            if (grpo_states is None or grpo_successors is None
                    or grpo_timesteps is None or grpo_signed_dt is None
                    or grpo_noise_scales is None or grpo_var_floor is None):
                raise ValueError("GRPO forward requires the sampled SDE trajectory")
            chunk_count = int(grpo_states.shape[0])
            anchor_count = int(queries.shape[0])
            if reference_visual_tokens.shape[0] != anchor_count:
                raise ValueError("GRPO reference batch must match slow conditions")
            if current_visual_tokens.shape[0] != chunk_count:
                raise ValueError("current visual batch must match SDE chunk count")
            chunk_anchor_index = (
                grpo_chunk_anchor_index.reshape(-1).to(
                    device=queries.device, dtype=torch.long)
                if grpo_chunk_anchor_index is not None else
                torch.zeros(chunk_count, device=queries.device, dtype=torch.long))
            if chunk_anchor_index.numel() != chunk_count:
                raise ValueError("GRPO chunk anchors must match SDE chunk count")
            if torch.any(chunk_anchor_index < 0) or torch.any(
                    chunk_anchor_index >= anchor_count):
                raise ValueError("GRPO chunk anchor index is outside the condition batch")
            # Keep historical batch-one action expert kernels and chunk order.
            # Only the expensive differentiable Qwen prefix/query graph above
            # is shared, exactly as a common subexpression within this root
            # FSDP forward. Backward therefore accumulates every chunk's query
            # gradient through the same deterministic Qwen Jacobian.
            logprobs = []
            for index in range(chunk_count):
                anchor = int(chunk_anchor_index[index].item())
                condition = self.action_head.encode_condition(
                    queries[anchor:anchor + 1],
                    reference_visual_tokens[anchor:anchor + 1],
                    current_visual_tokens[index:index + 1],
                    (None if reference_visual_mask is None else
                     reference_visual_mask[anchor:anchor + 1]),
                    (None if current_visual_mask is None else
                     current_visual_mask[index:index + 1]),
                )
                logprobs.append(recompute_reverse_logprob(
                    self.action_head.velocity, condition,
                    grpo_states[index:index + 1],
                    grpo_successors[index:index + 1], grpo_timesteps,
                    grpo_signed_dt, grpo_noise_scales, grpo_var_floor))
            new_logprob = torch.cat(logprobs, dim=0)
            # Capture root-owned *unsharded* leaves while inside the FSDP forward.
            # Selective backward can then prune Qwen parameter derivatives without
            # detaching the differentiable query path. Outside this forward the
            # module attributes may already refer to sharded DTensors again.
            if os.environ.get("ACTION_HEAD_RULE_CREDIT", "0") == "1":
                joint_text["action_parameters"] = tuple(self.query_encoder.parameters()) + tuple(self.action_head.parameters())
            return {"query_hidden": queries, "new_logprob": new_logprob, **joint_text}

        if target_actions is None or action_mask is None:
            raise ValueError("SFT forward requires target_actions and action_mask")
        live_inputs = (fast_pixel_values, fast_image_grid_thw,
                       fast_reference_image_index, fast_current_image_index)
        if any(value is not None for value in live_inputs):
            if not all(value is not None for value in live_inputs):
                raise ValueError('incomplete live fast visual inputs')
            if reference_visual_tokens is not None or current_visual_tokens is not None:
                raise ValueError('live images and preencoded visual features are mutually exclusive')
            if self.fast_visual is None:
                raise ValueError('live fast images require the bound original fast vision tower')
            from sft.live_fast_visual import encode_fast_inputs
            (reference_visual_tokens, current_visual_tokens,
             reference_visual_mask, current_visual_mask) = encode_fast_inputs(
                self.fast_visual, *live_inputs)
            self.fast_visual_input_mode = "live_frozen_visual_v1"
        elif os.environ.get('ACTION_HEAD_FAST_VISUAL_MODE') == 'live':
            raise ValueError('formal live model requires image inputs')
        queries = self._forward_slow_anchors(
            slow, slow_image_count_per_anchor, slow_pixel_count_per_anchor)
        if (not self._batch_parity_checked
                and os.environ.get("ACTION_HEAD_BATCH_PARITY_CHECK", "0") == "1"
                and queries.shape[0] > 1):
            if slow_image_count_per_anchor is None or slow_pixel_count_per_anchor is None:
                raise RuntimeError("batch parity check requires multimodal anchor boundaries")
            self._check_batch_parity(
                slow, queries, slow_image_count_per_anchor.reshape(-1),
                slow_pixel_count_per_anchor.reshape(-1))
        window_count = target_actions.shape[0]
        anchor_count = queries.shape[0]
        if window_anchor_index is None:
            if anchor_count != 1:
                raise ValueError("window_anchor_index is required for multiple slow anchors")
            window_anchor_index = torch.zeros(
                window_count, device=queries.device, dtype=torch.long)
        else:
            window_anchor_index = window_anchor_index.reshape(-1).to(
                device=queries.device, dtype=torch.long)
        if window_anchor_index.numel() != window_count:
            raise ValueError("window_anchor_index must contain one value per window")
        if torch.any(window_anchor_index < 0) or torch.any(window_anchor_index >= anchor_count):
            raise ValueError("window_anchor_index is outside the slow-anchor batch")
        condition = self._encode_condition(
            queries.index_select(0, window_anchor_index),
            reference_visual_tokens,
            current_visual_tokens,
            reference_visual_mask,
            current_visual_mask,
        )
        seeds = (noise_seed.reshape(-1).to(device=target_actions.device, dtype=torch.long)
                 if noise_seed is not None
                 else torch.zeros(anchor_count, device=target_actions.device, dtype=torch.long))
        if seeds.numel() != anchor_count:
            raise ValueError("noise_seed must contain one value per slow anchor")
        if window_index_within_anchor is None:
            window_index_within_anchor = torch.arange(
                window_count, device=target_actions.device, dtype=torch.long)
        else:
            window_index_within_anchor = window_index_within_anchor.reshape(-1).to(
                device=target_actions.device, dtype=torch.long)
        if window_index_within_anchor.numel() != window_count:
            raise ValueError("window_index_within_anchor must contain one value per window")
        anchor_valid = (anchor_valid.reshape(-1).to(device=target_actions.device,
                                                   dtype=torch.float32)
                        if anchor_valid is not None
                        else torch.ones(anchor_count, device=target_actions.device))
        if anchor_valid.numel() != anchor_count:
            raise ValueError("anchor_valid must contain one value per slow anchor")
        window_valid = anchor_valid.index_select(0, window_anchor_index)
        losses = []
        velocity_norms = []
        batch_windows = os.environ.get("ACTION_HEAD_FM_BATCH_WINDOWS", "0") == "1"
        generators = []
        for index in range(window_count):
            anchor_index = int(window_anchor_index[index].item())
            generator = torch.Generator(device=target_actions.device).manual_seed(
                int(seeds[anchor_index].item()) + int(window_index_within_anchor[index].item()))
            if batch_windows:
                generators.append(generator)
                continue
            window_loss, window_vel_norm = self.action_head.flow_matching_loss(
                condition[index:index + 1], target_actions[index:index + 1],
                action_mask[index:index + 1], generator,
                return_velocity=True)
            losses.append(window_loss)
            velocity_norms.append(window_vel_norm)
        if batch_windows:
            window_losses, velocity_norms = self.action_head.flow_matching_loss(
                condition, target_actions, action_mask, generators=generators,
                reduction="none", return_velocity=True)
        else:
            window_losses = torch.stack(losses)
            velocity_norms = torch.stack(velocity_norms)
        loss = globally_weighted_window_loss(window_losses, window_valid, global_window_count)
        # Per-row L2 norm of the fused condition (mean over rows), detached so
        # it never enters the backward graph. Health metric for the condition
        # channel (mirrors MegaTE's condition-feature diagnostics).
        cond_rows = condition.reshape(-1, condition.shape[-1]).float()
        cond_norm_mean = cond_rows.norm(dim=-1).mean()
        result = {
            "loss": loss,
            "local_loss_sum": (window_losses.detach() * window_valid).sum(),
            "local_window_count": window_valid.sum(),
            # Per-rank means (health metrics); the engine averages across DP.
            "local_velocity_mean": (
                velocity_norms.detach() * window_valid).sum() / window_valid.sum().clamp_min(1.0),
            "local_cond_norm_mean": cond_norm_mean.detach(),
        }
        if compute_metrics:
            # Deployment-shaped ODE decode: sigma 1 -> 0 from RANDOM initial
            # noise, matching online sampling (policy.predict draws randn).
            # Seeded by the per-anchor noise_seed so validation ADE/FDE stay
            # comparable across checkpoints — the validation collator resets
            # its call counter before each _fixed_validation pass, making the
            # per-anchor draw deterministic. The previous zero-initial-noise
            # decode masked under-denoising at the configured (3) inference
            # steps; random noise exposes the real deployment regime. The
            # decode generator is offset from every loss generator (which use
            # base_seed + window_index) so it never aliases their RNG stream.
            prediction = torch.empty_like(
                target_actions, device=condition.device, dtype=condition.dtype)
            for anchor_index in range(anchor_count):
                indices = torch.nonzero(
                    window_anchor_index == anchor_index, as_tuple=False).reshape(-1)
                if not indices.numel():
                    continue
                decode_generator = torch.Generator(
                    device=target_actions.device).manual_seed(
                        int(seeds[anchor_index].item()))
                anchor_noise = torch.randn(
                    (indices.numel(), *target_actions.shape[1:]),
                    device=condition.device, dtype=condition.dtype,
                    generator=decode_generator)
                prediction.index_copy_(0, indices, anchor_noise)
            steps = self.action_head.config.inference_steps
            step_size = -1.0 / steps
            for index in range(steps):
                sigma = torch.full(
                    (window_count,), 1.0 - index / steps,
                    device=prediction.device, dtype=torch.float32)
                prediction = prediction + self.action_head.velocity(
                    prediction, sigma, condition) * step_size
            mask = action_mask.to(device=prediction.device, dtype=torch.bool)
            valid_steps = mask.sum(dim=-1)
            if torch.any(valid_steps <= 0):
                raise ValueError("validation action mask has an empty window")
            distance = (prediction.float() - target_actions.float()).norm(dim=-1)
            per_window_ade = (distance * mask).sum(dim=-1) / valid_steps
            last = (valid_steps - 1).to(dtype=torch.long)
            per_window_fde = distance.gather(1, last[:, None]).squeeze(1)
            result.update({
                "local_ade_sum": (per_window_ade * window_valid).sum(),
                "local_fde_sum": (per_window_fde * window_valid).sum(),
                "local_metric_window_count": window_valid.sum(),
            })
        return result
