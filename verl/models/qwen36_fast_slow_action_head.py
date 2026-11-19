"""Qwen3.6-conditioned fast/slow action expert, first no-status version.

Project-specific contract
-------------------------
The slow Agent sees exactly the same messages as the primitive Tool-use Agent;
its only intervention is calling ``run_action_head(target, where, plan)``.
Final-normalized Qwen hidden states provide the slow condition. They are
produced by appending independently learnable query embeddings after the
completed Tool call, matching DualVLN's latent-query mechanism. The
queries are collaborative latent slots; this module assigns no target/where/
plan role to an individual slot.

The fast model additionally consumes frozen visual-encoder tokens from the
slow reference RGB and current fast RGB.  It predicts only one continuous
ego-relative trajectory.  There is deliberately no completion/replan/status
head in this first version; refresh and safety returns belong to the runtime.

DualVLN is used as an implementation reference for frozen slow conditioning,
reference/current visual fusion and asynchronous reuse.  This module does not
implement its pixel-goal interface, navigation-tuned System 2, trajectory
candidate averaging or fixed Habitat discrete-action adapter.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
from torch import Tensor, nn

try:  # package import
    from .qwen36_action_head import (
        ActionHeadConfig,
        ConditionalVelocityHead,
        sample_wan_flow_sigma,
        wan_flow_training_pair,
    )
except ImportError:  # direct script/test execution from this directory
    from qwen36_action_head import (
        ActionHeadConfig,
        ConditionalVelocityHead,
        sample_wan_flow_sigma,
        wan_flow_training_pair,
    )


@dataclass(frozen=True)
class FastSlowActionHeadConfig:
    backbone_model_type: str
    qwen_hidden_size: int
    visual_hidden_size: int
    fusion_dim: int
    action_dim: int
    horizon: int
    query_num_queries: int
    visual_num_queries: int
    visual_layers: int
    fusion_layers: int
    fusion_heads: int
    flow_model_dim: int
    flow_layers: int
    flow_heads: int
    inference_steps: int
    flow_timestep_sampling: str = "logit_normal"
    flow_logit_mean: float = 0.0
    flow_logit_std: float = 1.0
    flow_convention: str = "wan22_data_to_noise"
    flow_initialization: str = "wan22_xavier_zero_output_v1"

    def validate(self) -> None:
        if not self.backbone_model_type:
            raise ValueError("backbone_model_type must not be empty")
        values = asdict(self)
        non_integer = {
            "backbone_model_type", "flow_timestep_sampling", "flow_logit_mean",
            "flow_logit_std", "flow_convention", "flow_initialization",
        }
        bad = {k: v for k, v in values.items()
               if k not in non_integer
               and (not isinstance(v, int) or v <= 0)}
        if bad:
            raise ValueError(f"config integer fields must be positive: {bad}")
        divisibility = {
            "fusion_dim/fusion_heads": self.fusion_dim % self.fusion_heads,
            "flow_model_dim/flow_heads": self.flow_model_dim % self.flow_heads,
        }
        bad_div = [name for name, remainder in divisibility.items() if remainder]
        if bad_div:
            raise ValueError("incompatible attention dimensions: " + ", ".join(bad_div))
        if self.flow_timestep_sampling != "logit_normal":
            raise ValueError("Wan2.2 alignment requires logit_normal timestep sampling")
        if not isinstance(self.flow_logit_mean, (int, float)):
            raise ValueError("flow_logit_mean must be numeric")
        if not isinstance(self.flow_logit_std, (int, float)) or self.flow_logit_std <= 0:
            raise ValueError("flow_logit_std must be positive")
        if self.flow_convention != "wan22_data_to_noise":
            raise ValueError("unsupported flow convention")
        if self.flow_initialization != "wan22_xavier_zero_output_v1":
            raise ValueError("unsupported flow initialization")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FastSlowActionHeadConfig":
        config = cls(**dict(value))
        config.validate()
        return config


def _validate_tokens(name: str, tokens: Tensor, hidden_size: int,
                     mask: Optional[Tensor]) -> None:
    if tokens.ndim != 3 or tokens.shape[-1] != hidden_size:
        raise ValueError(
            f"{name} must have shape (B,L,{hidden_size}), got {tuple(tokens.shape)}")
    if mask is not None and tuple(mask.shape) != tuple(tokens.shape[:2]):
        raise ValueError(
            f"{name}_mask must match {name}[:2], got {tuple(mask.shape)}")


class _QueryPooler(nn.Module):
    def __init__(self, dim: int, num_queries: int, heads: int):
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, num_queries, dim))
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.output_norm = nn.LayerNorm(dim)
        nn.init.normal_(self.query, mean=0.0, std=0.02)

    def forward(self, context: Tensor,
                attention_mask: Optional[Tensor] = None) -> Tensor:
        query = self.query.expand(context.shape[0], -1, -1)
        padding_mask = None
        if attention_mask is not None:
            padding_mask = ~attention_mask.to(device=context.device, dtype=torch.bool)
            if torch.any(padding_mask.all(dim=1)):
                raise ValueError("each sample must contain at least one valid condition token")
        pooled, _ = self.attention(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        return self.output_norm(query + pooled)


class SlowQueryAdapter(nn.Module):
    """Project the Qwen query outputs without pooling them again."""

    def __init__(self, config: FastSlowActionHeadConfig):
        super().__init__()
        self.input_norm = nn.LayerNorm(config.qwen_hidden_size)
        self.input_projection = nn.Linear(config.qwen_hidden_size, config.fusion_dim)
        self.output_norm = nn.LayerNorm(config.fusion_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.output_norm(self.input_projection(self.input_norm(hidden_states)))


class TemporalVisualAdapter(nn.Module):
    """Fuse slow-reference and current fast visual tokens."""

    def __init__(self, config: FastSlowActionHeadConfig):
        super().__init__()
        self.input_projection = nn.Linear(config.visual_hidden_size, config.fusion_dim)
        self.time_embedding = nn.Parameter(torch.empty(2, config.fusion_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.fusion_dim,
            nhead=config.fusion_heads,
            dim_feedforward=config.fusion_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_fusion = nn.TransformerEncoder(layer, config.visual_layers)
        self.pooler = _QueryPooler(
            config.fusion_dim,
            config.visual_num_queries,
            config.fusion_heads,
        )
        nn.init.normal_(self.time_embedding, mean=0.0, std=0.02)

    def forward(self, reference_tokens: Tensor, current_tokens: Tensor,
                reference_mask: Optional[Tensor],
                current_mask: Optional[Tensor]) -> Tensor:
        if reference_tokens.shape[0] != current_tokens.shape[0]:
            raise ValueError("reference/current visual batch sizes must match")
        reference = self.input_projection(reference_tokens) + self.time_embedding[0]
        current = self.input_projection(current_tokens) + self.time_embedding[1]
        tokens = torch.cat((reference, current), dim=1)
        if reference_mask is None:
            reference_mask = torch.ones(
                reference.shape[:2], device=reference.device, dtype=torch.bool)
        if current_mask is None:
            current_mask = torch.ones(
                current.shape[:2], device=current.device, dtype=torch.bool)
        mask = torch.cat((reference_mask, current_mask), dim=1).to(
            device=tokens.device, dtype=torch.bool)
        fused = self.temporal_fusion(tokens, src_key_padding_mask=~mask)
        return self.pooler(fused, mask)


class Qwen36FastSlowActionHead(nn.Module):
    """No-status fast trajectory model for the action expert Tool v1."""

    CONFIG_NAME = "fast_slow_action_head_config.json"
    WEIGHTS_NAME = "fast_slow_action_head.pt"

    def __init__(self, config: FastSlowActionHeadConfig):
        super().__init__()
        config.validate()
        self.config = config

        self.slow_query_adapter = SlowQueryAdapter(config)
        self.visual_adapter = TemporalVisualAdapter(config)

        self.channel_embedding = nn.Parameter(torch.empty(2, config.fusion_dim))
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=config.fusion_dim,
            nhead=config.fusion_heads,
            dim_feedforward=config.fusion_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.condition_fusion = nn.TransformerEncoder(
            fusion_layer, config.fusion_layers)

        flow_config = ActionHeadConfig(
            backbone_model_type=config.backbone_model_type,
            hidden_size=config.fusion_dim,
            action_dim=config.action_dim,
            horizon=config.horizon,
            num_queries=1,
            action_former_layers=1,
            action_former_heads=config.fusion_heads,
            action_former_ffn_ratio=1,
            flow_model_dim=config.flow_model_dim,
            flow_layers=config.flow_layers,
            flow_heads=config.flow_heads,
            inference_steps=config.inference_steps,
        )
        self.velocity = ConditionalVelocityHead(flow_config)
        nn.init.normal_(self.channel_embedding, mean=0.0, std=0.02)

    def encode_condition(
        self,
        query_hidden_states: Tensor,
        reference_visual_tokens: Tensor,
        current_visual_tokens: Tensor,
        reference_visual_mask: Optional[Tensor] = None,
        current_visual_mask: Optional[Tensor] = None,
        *,
        use_slow_queries: bool = True,
        use_visual: bool = True,
        slow_condition_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        cfg = self.config
        if slow_condition_tokens is None:
            _validate_tokens(
                "query_hidden_states", query_hidden_states,
                cfg.qwen_hidden_size, None)
            if query_hidden_states.shape[1] != cfg.query_num_queries:
                raise ValueError(
                    f"query_hidden_states must contain exactly {cfg.query_num_queries} "
                    f"queries, got {query_hidden_states.shape[1]}")
            batch = query_hidden_states.shape[0]
        else:
            if query_hidden_states is not None:
                raise ValueError("provide either native context tokens or query hidden states")
            _validate_tokens("slow_condition_tokens", slow_condition_tokens,
                             self.channel_embedding.shape[-1], None)
            batch = slow_condition_tokens.shape[0]
        _validate_tokens(
            "reference_visual_tokens", reference_visual_tokens,
            cfg.visual_hidden_size, reference_visual_mask)
        _validate_tokens(
            "current_visual_tokens", current_visual_tokens,
            cfg.visual_hidden_size, current_visual_mask)
        if any(tokens.shape[0] != batch for tokens in (
                reference_visual_tokens,
                current_visual_tokens)):
            raise ValueError("all condition channels must have the same batch size")

        channels = []
        if use_slow_queries:
            latent = (self.slow_query_adapter(query_hidden_states)
                      if slow_condition_tokens is None else slow_condition_tokens)
            channels.append(latent + self.channel_embedding[0])
        if use_visual:
            visual = self.visual_adapter(
                reference_visual_tokens,
                current_visual_tokens,
                reference_visual_mask,
                current_visual_mask,
            )
            channels.append(visual + self.channel_embedding[1])
        if not channels:
            raise ValueError("at least one condition channel must be enabled")
        condition = torch.cat(channels, dim=1)
        return self.condition_fusion(condition)

    def flow_matching_loss(self, condition: Tensor, target_actions: Tensor,
                           action_mask: Optional[Tensor] = None,
                           generator: Optional[torch.Generator] = None,
                           return_velocity: bool = False, *,
                           generators: Optional[list[torch.Generator]] = None,
                           reduction: str = "mean"):
        expected = (self.config.horizon, self.config.action_dim)
        if target_actions.ndim != 3 or tuple(target_actions.shape[1:]) != expected:
            raise ValueError(
                f"target_actions must have shape (B,{expected[0]},{expected[1]})")
        target = target_actions.to(device=condition.device, dtype=condition.dtype)
        if reduction not in ("mean", "none"):
            raise ValueError("reduction must be mean or none")
        if generators is None:
            noise = torch.randn(
                target.shape, device=target.device, dtype=target.dtype,
                generator=generator)
            sigma = sample_wan_flow_sigma(
                target.shape[0], device=target.device, generator=generator,
                logit_mean=self.config.flow_logit_mean,
                logit_std=self.config.flow_logit_std)
        else:
            if generator is not None or len(generators) != target.shape[0]:
                raise ValueError("provide exactly one generator per window and no shared generator")
            # Preserve each historical batch-one RNG stream, including draw
            # shapes and noise-before-sigma order. Only the network is batched.
            noises, sigmas = [], []
            for item_generator in generators:
                noises.append(torch.randn(
                    (1, *target.shape[1:]), device=target.device,
                    dtype=target.dtype, generator=item_generator))
                sigmas.append(sample_wan_flow_sigma(
                    1, device=target.device, generator=item_generator,
                    logit_mean=self.config.flow_logit_mean,
                    logit_std=self.config.flow_logit_std))
            noise, sigma = torch.cat(noises), torch.cat(sigmas)
        noisy, velocity_target = wan_flow_training_pair(target, noise, sigma)
        velocity_pred = self.velocity(noisy, sigma, condition)
        # Wan trains the velocity regression in FP32 even when the network
        # forward uses BF16; retain that numerical boundary for trajectories.
        loss = (velocity_pred.float() - velocity_target.float()).square()
        # Mean predicted-velocity L2 norm over valid points; detached so it
        # never enters the backward graph. Used for the action_magnitude TB
        # metric (matches MegaTE's diagnostic velocity-norm).
        vel_norm_per_step = velocity_pred.float().norm(dim=-1)
        if reduction == "none":
            mask = (torch.ones(target.shape[:2], device=target.device, dtype=torch.bool)
                    if action_mask is None else action_mask.to(device=target.device, dtype=torch.bool))
            if tuple(mask.shape) != tuple(target.shape[:2]):
                raise ValueError("action_mask must match target_actions[:2]")
            counts = mask.sum(dim=-1)
            if torch.any(counts == 0):
                raise ValueError("action_mask has an empty trajectory window")
            loss_val = loss.masked_fill(~mask.unsqueeze(-1), 0).sum(dim=(1, 2)) / (counts * target.shape[-1])
            vel_norm = vel_norm_per_step.masked_fill(~mask, 0).sum(dim=-1) / counts
            return (loss_val, vel_norm.detach()) if return_velocity else loss_val
        if action_mask is None:
            loss_val = loss.mean()
            vel_norm = vel_norm_per_step.mean()
        else:
            if tuple(action_mask.shape) != tuple(target.shape[:2]):
                raise ValueError("action_mask must match target_actions[:2]")
            mask = action_mask.to(device=loss.device, dtype=torch.bool)
            if not torch.any(mask):
                raise ValueError("action_mask has no valid trajectory points")
            loss_val = loss[mask.unsqueeze(-1).expand_as(loss)].mean()
            vel_norm = vel_norm_per_step[mask].mean()
        if return_velocity:
            return loss_val, vel_norm.detach()
        return loss_val

    def forward(self, *, target_actions: Optional[Tensor] = None,
                action_mask: Optional[Tensor] = None,
                generator: Optional[torch.Generator] = None, **condition_inputs):
        condition = self.encode_condition(**condition_inputs)
        result = {"action_condition": condition}
        if target_actions is not None:
            result["action_loss"] = self.flow_matching_loss(
                condition, target_actions, action_mask, generator)
        return result

    @torch.no_grad()
    def sample(self, *, initial_noise: Optional[Tensor] = None,
               generator: Optional[torch.Generator] = None,
               inference_steps: Optional[int] = None, **condition_inputs) -> Tensor:
        condition = self.encode_condition(**condition_inputs)
        batch = condition.shape[0]
        shape = (batch, self.config.horizon, self.config.action_dim)
        if initial_noise is None:
            trajectory = torch.randn(
                shape, device=condition.device, dtype=condition.dtype,
                generator=generator)
        else:
            if tuple(initial_noise.shape) != shape:
                raise ValueError(
                    f"initial_noise must have shape {shape}, got {tuple(initial_noise.shape)}")
            trajectory = initial_noise.to(
                device=condition.device, dtype=condition.dtype)
        steps = inference_steps or self.config.inference_steps
        if steps <= 0:
            raise ValueError("inference_steps must be positive")
        step_size = -1.0 / steps
        for index in range(steps):
            sigma = torch.full(
                (batch,), 1.0 - index / steps,
                device=trajectory.device, dtype=torch.float32)
            trajectory = trajectory + self.velocity(
                trajectory, sigma, condition) * step_size
        return trajectory

    def save_pretrained(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / self.CONFIG_NAME).write_text(
            json.dumps(self.config.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        torch.save(self.state_dict(), output / self.WEIGHTS_NAME)

    @classmethod
    def from_pretrained(cls, input_dir: str | Path,
                        map_location: str | torch.device = "cpu") \
            -> "Qwen36FastSlowActionHead":
        source = Path(input_dir)
        config = FastSlowActionHeadConfig.from_dict(json.loads(
            (source / cls.CONFIG_NAME).read_text(encoding="utf-8")))
        model = cls(config)
        try:
            state = torch.load(
                source / cls.WEIGHTS_NAME,
                map_location=map_location,
                weights_only=True,
            )
        except TypeError:
            state = torch.load(source / cls.WEIGHTS_NAME,
                               map_location=map_location)
        model.load_state_dict(state, strict=True)
        return model
