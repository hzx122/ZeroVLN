#!/usr/bin/env python3
"""Phase-2 core: SDE sampling + per-step log-prob for the Flow-Matching action expert.

The FM action expert is trained as an OT/linear-path velocity field
``v_theta(x_t, t, cond)`` that predicts ``u_t = x1 - x0`` and is *sampled*
deterministically with an Euler ODE (see
``reference/SocialNav/train_clean/infer.py:695-715``)::

    x_0 ~ N(0, I);  x_{t+dt} = x_t + v_theta(x_t, t) * dt   (deterministic)

A deterministic ODE trajectory is a delta distribution -> no tractable
sampling probability -> no policy gradient. To RL the continuous trajectory we
replace the Euler ODE by an Euler-Maruyama **SDE** with the SAME drift plus a
controlled Gaussian kick (DDPO / Flow-GRPO style)::

    x_{t+dt} = x_t + v_theta(x_t, t) * dt + sigma(t) * sqrt(dt) * eps,  eps~N(0,I)

Each step is then a Gaussian transition with a closed-form log-prob::

    mean_i   = x_i + v_theta(x_i, t_i) * dt
    var_i    = sigma(t_i)^2 * dt
    logp_i   = sum_dims log N(x_{i+1}; mean_i, var_i * I)
    logprob  = sum_i logp_i

This module is intentionally **model-agnostic**: it only needs a
``predict_velocity(cond, x_t, t) -> v`` callable (exactly the signature of
``FlowMatchingActionHead.predict_velocity``). Both the verl rollout worker
(sampling, no grad) and the actor worker (recompute log-prob, with grad) import
from here so the SDE math lives in one place.

Gradient note: during rollout we sample and *store* every visited state
``x_i`` and its realized successor ``x_{i+1}`` (both are data, no grad). The
actor later recomputes ``v_theta(x_i, t_i)`` -- the ONLY term carrying grad --
and re-evaluates ``logp_i`` of the stored transition. This mirrors how token
GRPO recomputes ``selective_log_softmax`` over stored completion ids.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch

_LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class SDEConfig:
    """Discretization + noise schedule for the FM SDE sampler.

    action_chunk / action_dim: trajectory shape (32 / 3 for the VLN FM head).
    num_steps: Euler-Maruyama steps over t in [0, 1]. Training used 3 ODE steps;
        RL benefits from a finer grid (more stochastic transitions -> smoother
        credit assignment). Default 10.
    sigma: noise level in *normalized* action space. Constant float, or a
        callable sigma(t: float) -> float for a schedule. 0.0 recovers the
        deterministic ODE (log-prob then degenerate -> only for A/B sanity).
    t_start / t_end: integration bounds. FM path is defined on [0, 1].
    var_floor: numerical floor on per-step variance so log(var) stays finite
        when sigma * sqrt(dt) is tiny.
    """

    action_chunk: int = 32
    action_dim: int = 3
    num_steps: int = 10
    sigma: float | Callable[[float], float] = 0.2
    t_start: float = 0.0
    t_end: float = 1.0
    var_floor: float = 1e-6

    def sigma_at(self, t: float) -> float:
        return float(self.sigma(t)) if callable(self.sigma) else float(self.sigma)


def _step_logprob(
    x_cur: torch.Tensor,     # (B, chunk, dim)
    x_next: torch.Tensor,    # (B, chunk, dim)
    v: torch.Tensor,         # (B, chunk, dim) = predict_velocity(cond, x_cur, t)
    dt: float,
    sigma_t: float,
    var_floor: float,
) -> torch.Tensor:
    """Gaussian transition log-prob of one Euler-Maruyama step, summed over
    (chunk, dim). Returns (B,).

    mean = x_cur + v * dt ;  var = sigma_t^2 * dt
    Only ``v`` need carry grad; x_cur / x_next are treated as data.
    """
    # Reverse-time samplers use a negative signed dt for the drift direction,
    # while Brownian variance always scales with the positive elapsed time.
    # Using signed dt here silently collapsed every reverse-SDE variance to
    # var_floor and amplified policy gradients by orders of magnitude.
    var = max(sigma_t * sigma_t * abs(dt), var_floor)
    mean = x_cur + v * dt
    diff = x_next - mean
    # elementwise log N(x_next; mean, var), then sum over the trajectory dims
    logp = -0.5 * (diff * diff / var + _LOG_2PI + math.log(var))
    return logp.flatten(start_dim=1).sum(dim=1)


@torch.no_grad()
def sample_sde(
    predict_velocity: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    cond: torch.Tensor,                 # (B, H) action_feature conditioning
    cfg: SDEConfig,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
    init_noise: Optional[torch.Tensor] = None,   # (B, chunk, dim) pin x_0 (optional)
) -> dict:
    """Roll out the FM SDE and return the trajectory + everything needed to
    recompute log-prob later.

    Returns dict:
      traj        : (B, chunk, dim)  final x at t_end  (the sampled action)
      xs          : (B, S, chunk, dim) visited states x_0..x_{S-1} (step inputs)
      x_nexts     : (B, S, chunk, dim) realized successors x_1..x_S
      ts          : (S,) step start-times t_i
      dt          : float
      sigmas      : (S,) sigma(t_i)
      old_logprob : (B,) sum of per-step log-probs at sample time (for PPO ratio)
      step_logprob: (B, S) per-step log-probs
    """
    B, H = cond.shape
    device = device if device is not None else cond.device
    chunk, dim = cfg.action_chunk, cfg.action_dim
    dt = (cfg.t_end - cfg.t_start) / cfg.num_steps

    if init_noise is not None:
        x = init_noise.to(device=device, dtype=dtype)
    else:
        x = torch.randn(B, chunk, dim, device=device, dtype=dtype, generator=generator)

    xs, x_nexts, ts, sigmas, step_lps = [], [], [], [], []
    for i in range(cfg.num_steps):
        t_val = cfg.t_start + i * dt
        t = torch.full((B,), t_val, device=device, dtype=dtype)
        v = predict_velocity(cond, x, t).to(dtype)
        sigma_t = cfg.sigma_at(t_val)
        eps = torch.randn(B, chunk, dim, device=device, dtype=dtype, generator=generator)
        x_next = x + v * dt + sigma_t * math.sqrt(dt) * eps
        lp = _step_logprob(x, x_next, v, dt, sigma_t, cfg.var_floor)

        xs.append(x)
        x_nexts.append(x_next)
        ts.append(t_val)
        sigmas.append(sigma_t)
        step_lps.append(lp)
        x = x_next

    step_logprob = torch.stack(step_lps, dim=1)          # (B, S)
    return {
        "traj": x,
        "xs": torch.stack(xs, dim=1),                    # (B, S, chunk, dim)
        "x_nexts": torch.stack(x_nexts, dim=1),          # (B, S, chunk, dim)
        "ts": torch.tensor(ts, device=device, dtype=dtype),
        "dt": dt,
        "sigmas": torch.tensor(sigmas, device=device, dtype=dtype),
        "old_logprob": step_logprob.sum(dim=1),          # (B,)
        "step_logprob": step_logprob,
    }


def recompute_logprob(
    predict_velocity: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    cond: torch.Tensor,          # (B, H) -- may carry grad (backbone) if desired
    xs: torch.Tensor,            # (B, S, chunk, dim) stored step inputs
    x_nexts: torch.Tensor,       # (B, S, chunk, dim) stored successors
    ts: torch.Tensor,            # (S,) step start-times
    dt: float,
    sigmas: torch.Tensor,        # (S,) sigma(t_i)
    var_floor: float = 1e-6,
) -> torch.Tensor:
    """Recompute the summed trajectory log-prob under the CURRENT policy for a
    stored rollout. Grad flows through ``predict_velocity`` (the FM head, and
    optionally the backbone via ``cond``). Returns (B,) log-prob.

    This is the actor-side counterpart of :func:`sample_sde`; the two must use
    identical step math so the PPO/GRPO ratio ``exp(new - old)`` is exact at
    sampling time.
    """
    B, S = xs.shape[0], xs.shape[1]
    dtype = xs.dtype
    total = xs.new_zeros(B)
    for i in range(S):
        x_cur = xs[:, i]
        x_next = x_nexts[:, i]
        t = ts[i].expand(B).to(dtype)
        v = predict_velocity(cond, x_cur, t).to(dtype)
        sigma_t = float(sigmas[i].item())
        total = total + _step_logprob(x_cur, x_next, v, dt, sigma_t, var_floor)
    return total


def grpo_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-6) -> torch.Tensor:
    """Group-normalized advantages (GRPO). rewards: (B,) laid out as
    ``num_groups`` contiguous blocks of ``group_size``. Returns (B,)
    standardized within each group.
    """
    assert rewards.numel() % group_size == 0, "rewards not divisible by group_size"
    r = rewards.view(-1, group_size)
    adv = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + eps)
    return adv.reshape(-1)
