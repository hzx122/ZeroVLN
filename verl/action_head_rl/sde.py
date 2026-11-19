"""Wan2.2 reverse-time SDE sampling for the existing action expert velocity.

The older :mod:`workers.fm_sde` implementation owns the Gaussian transition
log-probability and GRPO normalization.  This module reuses those primitives
while adapting the integration direction to the deployed action expert, whose
sampler runs from noise at sigma=1 to data at sigma=0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

from workers.fm_sde import _step_logprob, grpo_advantages


OLD_LOGPROB_MODES = ("single_forward", "recompute")


def select_proximal_old_logprob(
    new_logprob: torch.Tensor,
    recorded_old_logprob: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Choose the fixed denominator used by the one-pass GRPO surrogate.

    ``single_forward`` anchors the ratio to the current forward itself.  It is
    valid only while a logical rollout batch performs one optimizer update:
    parameters have not changed before this forward, so a separate proximal
    forward would produce the same detached denominator.  ``recompute`` keeps
    the former two-forward implementation for parity/debugging.
    """
    if mode == "single_forward":
        return new_logprob.detach()
    if mode == "recompute":
        return recorded_old_logprob.reshape_as(new_logprob)
    raise ValueError(
        f"old log-prob mode must be one of {OLD_LOGPROB_MODES}, got {mode!r}")


@dataclass(frozen=True)
class ReverseSDEConfig:
    horizon: int = 16
    action_dim: int = 3
    num_steps: int = 3
    noise_scale: float = 0.08
    var_floor: float = 1e-6

    def validate(self) -> None:
        if self.horizon <= 0 or self.action_dim <= 0 or self.num_steps <= 0:
            raise ValueError("horizon, action_dim and num_steps must be positive")
        if not math.isfinite(self.noise_scale) or self.noise_scale <= 0:
            raise ValueError("noise_scale must be finite and positive for RL")
        if not math.isfinite(self.var_floor) or self.var_floor <= 0:
            raise ValueError("var_floor must be finite and positive")


@torch.no_grad()
def sample_reverse_sde(
    velocity: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    condition: torch.Tensor,
    config: ReverseSDEConfig,
    *,
    generator: torch.Generator | None = None,
    initial_noise: torch.Tensor | None = None,
    step_noises: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | float]:
    """Sample the deployed reverse path and retain exact recompute material."""
    config.validate()
    batch = condition.shape[0]
    shape = (batch, config.horizon, config.action_dim)
    if initial_noise is None:
        current = torch.randn(
            shape, device=condition.device, dtype=condition.dtype,
            generator=generator)
    else:
        if tuple(initial_noise.shape) != shape:
            raise ValueError(f"initial_noise must have shape {shape}")
        current = initial_noise.to(condition)
    if step_noises is not None:
        expected = (config.num_steps, *shape)
        if tuple(step_noises.shape) != expected:
            raise ValueError(f"step_noises must have shape {expected}")
        step_noises = step_noises.to(condition)

    signed_dt = -1.0 / config.num_steps
    variance_dt = abs(signed_dt)
    states, successors, sigmas, logprobs = [], [], [], []
    times = []
    for index in range(config.num_steps):
        sigma = 1.0 - index / config.num_steps
        timestep = torch.full(
            (batch,), sigma, device=current.device, dtype=torch.float32)
        drift = velocity(current, timestep, condition).to(current.dtype)
        noise = (step_noises[index] if step_noises is not None else torch.randn(
            current.shape, device=current.device, dtype=current.dtype,
            generator=generator))
        successor = (
            current + drift * signed_dt
            + config.noise_scale * math.sqrt(variance_dt) * noise
        )
        states.append(current)
        successors.append(successor)
        times.append(sigma)
        sigmas.append(config.noise_scale)
        logprobs.append(_step_logprob(
            current, successor, drift, signed_dt,
            config.noise_scale, config.var_floor,
        ))
        current = successor
    per_step = torch.stack(logprobs, dim=1)
    return {
        "trajectory": current,
        "states": torch.stack(states, dim=1),
        "successors": torch.stack(successors, dim=1),
        "timesteps": torch.tensor(times, device=current.device, dtype=torch.float32),
        "signed_dt": signed_dt,
        "noise_scales": torch.tensor(
            sigmas, device=current.device, dtype=torch.float32),
        "old_logprob": per_step.sum(dim=1),
        "step_logprob": per_step,
    }


def recompute_reverse_logprob(
    velocity: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    condition: torch.Tensor,
    states: torch.Tensor,
    successors: torch.Tensor,
    timesteps: torch.Tensor,
    signed_dt: float,
    noise_scales: torch.Tensor,
    var_floor: float = 1e-6,
) -> torch.Tensor:
    """Recompute log p of stored reverse-SDE transitions with gradients."""
    if signed_dt >= 0:
        raise ValueError("Wan2.2 Action Head recomputation requires signed_dt < 0")
    total = states.new_zeros(states.shape[0])
    for index in range(states.shape[1]):
        timestep = timesteps[index].expand(states.shape[0]).to(torch.float32)
        drift = velocity(states[:, index], timestep, condition).to(states.dtype)
        total = total + _step_logprob(
            states[:, index], successors[:, index], drift, signed_dt,
            float(noise_scales[index].item()), var_floor,
        )
    return total


__all__ = [
    "OLD_LOGPROB_MODES",
    "ReverseSDEConfig",
    "grpo_advantages",
    "recompute_reverse_logprob",
    "select_proximal_old_logprob",
    "sample_reverse_sde",
]
