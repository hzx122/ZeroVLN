"""Standalone action expert for Qwen3.6-35B-A3B hidden states.

This module deliberately does not patch the Qwen transformers implementation.
The backbone owns multimodal tokenization and returns hidden states; this head
owns action queries, conditional flow matching, trajectory loss and sampling.
That boundary lets the first SFT stage freeze the entire Qwen backbone and save
only a small randomly initialized action expert checkpoint.

The implementation follows the useful parts of the vendored SocialNav head:
learnable queries cross-attend to all multimodal hidden states, and a
conditional velocity model generates a fixed-shape local trajectory. It does
not inherit SocialNav weights, waypoint statistics, prompt tokens, hidden size
or the old 32-point horizon.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ActionHeadConfig:
    """All shape-affecting fields required to reproduce an action expert."""

    backbone_model_type: str
    hidden_size: int
    action_dim: int
    horizon: int
    num_queries: int
    action_former_layers: int
    action_former_heads: int
    action_former_ffn_ratio: int
    flow_model_dim: int
    flow_layers: int
    flow_heads: int
    inference_steps: int

    def validate(self) -> None:
        positive = {
            k: v for k, v in asdict(self).items()
            if k != "backbone_model_type"
        }
        bad = {k: v for k, v in positive.items()
               if not isinstance(v, int) or v <= 0}
        if bad:
            raise ValueError(f"Action Head integer fields must be positive: {bad}")
        if not self.backbone_model_type:
            raise ValueError("backbone_model_type must not be empty")
        if self.hidden_size % self.action_former_heads:
            raise ValueError("hidden_size must be divisible by action_former_heads")
        if self.flow_model_dim % self.flow_heads:
            raise ValueError("flow_model_dim must be divisible by flow_heads")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionHeadConfig":
        config = cls(**dict(value))
        config.validate()
        return config


def sample_wan_flow_sigma(batch_size: int, *, device: torch.device,
                          generator: Optional[torch.Generator],
                          logit_mean: float = 0.0,
                          logit_std: float = 1.0) -> Tensor:
    """Sample logit-normal data-to-noise levels for Wan-style FM."""
    logits = torch.randn(
        batch_size, device=device, dtype=torch.float32, generator=generator)
    return torch.sigmoid(logits * float(logit_std) + float(logit_mean))


def wan_flow_training_pair(target: Tensor, noise: Tensor,
                           sigma: Tensor) -> tuple[Tensor, Tensor]:
    """Return the Wan data-to-noise interpolation and velocity target."""
    if target.shape != noise.shape:
        raise ValueError("target and noise must have identical shapes")
    if sigma.ndim != 1 or sigma.shape[0] != target.shape[0]:
        raise ValueError("sigma must have shape (batch,)")
    mix = sigma.to(device=target.device, dtype=target.dtype).view(-1, 1, 1)
    noisy = (1.0 - mix) * target + mix * noise
    return noisy, noise - target


class _ActionFormerBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int, ffn_ratio: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(
            hidden_size, heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * ffn_ratio),
            nn.GELU(),
            nn.Linear(hidden_size * ffn_ratio, hidden_size),
        )

    def forward(self, query: Tensor, context: Tensor,
                key_padding_mask: Optional[Tensor]) -> Tensor:
        normalized_context = self.context_norm(context)
        update, _ = self.attention(
            self.query_norm(query), normalized_context, normalized_context,
            key_padding_mask=key_padding_mask, need_weights=False)
        query = query + update
        return query + self.ffn(self.ffn_norm(query))


class ActionFormer(nn.Module):
    def __init__(self, config: ActionHeadConfig):
        super().__init__()
        self.query = nn.Parameter(torch.empty(
            1, config.num_queries, config.hidden_size))
        self.blocks = nn.ModuleList([
            _ActionFormerBlock(
                config.hidden_size,
                config.action_former_heads,
                config.action_former_ffn_ratio,
            )
            for _ in range(config.action_former_layers)
        ])
        nn.init.normal_(self.query, mean=0.0, std=0.02)

    def forward(self, hidden_states: Tensor,
                attention_mask: Optional[Tensor] = None) -> Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                f"hidden_states must have shape (B,L,D), got {hidden_states.shape}")
        hidden_states = hidden_states.to(
            device=self.query.device, dtype=self.query.dtype)
        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != hidden_states.shape[:2]:
                raise ValueError(
                    "attention_mask shape must match hidden_states[:2], got "
                    f"{attention_mask.shape} vs {hidden_states.shape[:2]}")
            key_padding_mask = ~attention_mask.to(
                device=hidden_states.device, dtype=torch.bool)
        query = self.query.expand(hidden_states.shape[0], -1, -1)
        for block in self.blocks:
            query = block(query, hidden_states, key_padding_mask)
        return query


class _SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim < 4 or dim % 2:
            raise ValueError("time embedding dimension must be even and >= 4")
        self.dim = dim

    def forward(self, timestep: Tensor) -> Tensor:
        timestep = timestep.float().reshape(-1)
        half = self.dim // 2
        scale = -math.log(10000.0) / max(1, half - 1)
        frequencies = torch.exp(
            torch.arange(half, device=timestep.device, dtype=torch.float32)
            * scale)
        angles = timestep[:, None] * frequencies[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class ConditionalVelocityHead(nn.Module):
    """Predict a trajectory velocity conditioned on ActionFormer queries."""

    def __init__(self, config: ActionHeadConfig):
        super().__init__()
        d = config.flow_model_dim
        self.horizon = config.horizon
        self.action_dim = config.action_dim
        self.action_input = nn.Linear(config.action_dim, d)
        self.action_position = nn.Parameter(torch.empty(1, config.horizon, d))
        self.condition = nn.Linear(config.hidden_size, d)
        self.time_embedding = _SinusoidalTimeEmbedding(d)
        self.time_projection = nn.Sequential(
            nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=config.flow_heads,
            dim_feedforward=d * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.flow_layers)
        self.output_norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, config.action_dim)
        self.reset_parameters_wan22()

    def reset_parameters_wan22(self) -> None:
        """Use Wan2.2's Xavier backbone and zero velocity-output start."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
        for module in self.time_projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
        nn.init.normal_(self.action_position, mean=0.0, std=0.02)
        nn.init.zeros_(self.output.weight)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def forward(self, noisy_actions: Tensor, timestep: Tensor,
                condition: Tensor) -> Tensor:
        if noisy_actions.ndim != 3:
            raise ValueError("noisy_actions must have shape (B,T,A)")
        expected = (self.horizon, self.action_dim)
        if tuple(noisy_actions.shape[1:]) != expected:
            raise ValueError(
                f"expected trajectory shape (*,{expected[0]},{expected[1]}), "
                f"got {tuple(noisy_actions.shape)}")
        if condition.ndim != 3:
            raise ValueError("condition must have shape (B,Q,D)")
        dtype = self.action_input.weight.dtype
        action_tokens = self.action_input(noisy_actions.to(dtype))
        action_tokens = action_tokens + self.action_position.to(dtype)
        memory = self.condition(condition.to(dtype))
        time_token = self.time_projection(
            self.time_embedding(timestep).to(dtype)).unsqueeze(1)
        memory = torch.cat((time_token, memory), dim=1)
        decoded = self.decoder(action_tokens, memory)
        return self.output(self.output_norm(decoded))


class Qwen36ActionHead(nn.Module):
    """Randomly initialized ActionFormer + conditional flow-matching head."""

    CONFIG_NAME = "action_head_config.json"
    WEIGHTS_NAME = "action_head.pt"

    def __init__(self, config: ActionHeadConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.action_former = ActionFormer(config)
        self.velocity = ConditionalVelocityHead(config)

    def condition(self, hidden_states: Tensor,
                  attention_mask: Optional[Tensor] = None) -> Tensor:
        if hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"backbone hidden size {hidden_states.shape[-1]} does not match "
                f"Action Head config {self.config.hidden_size}")
        return self.action_former(hidden_states, attention_mask)

    def flow_matching_loss(self, condition: Tensor, target_actions: Tensor,
                           action_mask: Optional[Tensor] = None,
                           generator: Optional[torch.Generator] = None) -> Tensor:
        expected = (self.config.horizon, self.config.action_dim)
        if target_actions.ndim != 3 or tuple(target_actions.shape[1:]) != expected:
            raise ValueError(
                f"target_actions must have shape (B,{expected[0]},{expected[1]})")
        target = target_actions.to(
            device=condition.device, dtype=condition.dtype)
        noise = torch.randn(
            target.shape, device=target.device, dtype=target.dtype,
            generator=generator)
        sigma = sample_wan_flow_sigma(
            target.shape[0], device=target.device, generator=generator)
        noisy, velocity_target = wan_flow_training_pair(target, noise, sigma)
        velocity_pred = self.velocity(noisy, sigma, condition)
        loss = (velocity_pred.float() - velocity_target.float()).square()
        if action_mask is None:
            return loss.mean()
        if action_mask.shape != target.shape[:2]:
            raise ValueError(
                f"action_mask must have shape {target.shape[:2]}, got {action_mask.shape}")
        mask = action_mask.to(device=target.device, dtype=torch.bool)
        if not torch.any(mask):
            raise ValueError("action_mask has no valid trajectory points")
        expanded = mask.unsqueeze(-1).expand_as(loss)
        return loss[expanded].mean()

    def forward(self, hidden_states: Tensor,
                attention_mask: Optional[Tensor] = None,
                target_actions: Optional[Tensor] = None,
                action_mask: Optional[Tensor] = None,
                generator: Optional[torch.Generator] = None) -> Dict[str, Tensor]:
        condition = self.condition(hidden_states, attention_mask)
        result = {"action_condition": condition}
        if target_actions is not None:
            result["action_loss"] = self.flow_matching_loss(
                condition, target_actions, action_mask, generator)
        return result

    @torch.no_grad()
    def sample(self, hidden_states: Tensor,
               attention_mask: Optional[Tensor] = None,
               initial_noise: Optional[Tensor] = None,
               generator: Optional[torch.Generator] = None,
               inference_steps: Optional[int] = None) -> Tensor:
        condition = self.condition(hidden_states, attention_mask)
        batch = hidden_states.shape[0]
        shape = (batch, self.config.horizon, self.config.action_dim)
        if initial_noise is None:
            trajectory = torch.randn(
                shape, device=hidden_states.device, dtype=condition.dtype,
                generator=generator)
        else:
            if tuple(initial_noise.shape) != shape:
                raise ValueError(
                    f"initial_noise must have shape {shape}, got {initial_noise.shape}")
            trajectory = initial_noise.to(
                device=hidden_states.device, dtype=condition.dtype)
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
            json.dumps(self.config.to_dict(), ensure_ascii=False, indent=2)
            + "\n", encoding="utf-8")
        torch.save(self.state_dict(), output / self.WEIGHTS_NAME)

    @classmethod
    def from_pretrained(cls, input_dir: str | Path,
                        map_location: str | torch.device = "cpu") -> "Qwen36ActionHead":
        source = Path(input_dir)
        config = ActionHeadConfig.from_dict(json.loads(
            (source / cls.CONFIG_NAME).read_text(encoding="utf-8")))
        model = cls(config)
        try:
            state = torch.load(
                source / cls.WEIGHTS_NAME,
                map_location=map_location,
                weights_only=True)
        except TypeError:  # torch < 2.0 compatibility
            state = torch.load(source / cls.WEIGHTS_NAME,
                               map_location=map_location)
        model.load_state_dict(state, strict=True)
        return model


def extract_last_hidden_state(backbone_output: Any) -> Tensor:
    """Normalize common Hugging Face output forms to one (B,L,D) tensor."""
    hidden = getattr(backbone_output, "last_hidden_state", None)
    if hidden is not None:
        return hidden
    all_hidden = getattr(backbone_output, "hidden_states", None)
    if all_hidden is not None:
        if isinstance(all_hidden, (tuple, list)):
            if not all_hidden:
                raise ValueError("backbone output hidden_states is empty")
            return all_hidden[-1]
        if torch.is_tensor(all_hidden):
            return all_hidden
    if isinstance(backbone_output, Mapping):
        if backbone_output.get("last_hidden_state") is not None:
            return backbone_output["last_hidden_state"]
        value = backbone_output.get("hidden_states")
        if isinstance(value, (tuple, list)) and value:
            return value[-1]
        if torch.is_tensor(value):
            return value
    raise ValueError(
        "backbone output has no last_hidden_state/hidden_states; call Qwen with "
        "output_hidden_states=True")


def set_backbone_trainable(backbone: nn.Module,
                           trainable_name_fragments: Iterable[str] = ()) -> int:
    """Freeze a backbone, optionally reopening parameters matching fragments."""
    fragments = tuple(str(x) for x in trainable_name_fragments if str(x))
    count = 0
    for name, parameter in backbone.named_parameters():
        parameter.requires_grad = bool(
            fragments and any(fragment in name for fragment in fragments))
        if parameter.requires_grad:
            count += parameter.numel()
    return count
