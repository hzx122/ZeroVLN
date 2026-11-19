"""Preserve v8's single clipping operation across separate text/FM engines.

FM gradients must already be reduced and computed at the rollout policy. No
FM forward is performed here, so stepping text first cannot change its inputs.
Call collectively on Megatron ranks with the same reduced FM gradient norm.
"""
from __future__ import annotations

import math
import torch


def combined_clip(text_norm, fm_norm, max_norm):
    values = tuple(map(float, (text_norm, fm_norm, max_norm)))
    if not all(math.isfinite(v) and v >= 0 for v in values) or max_norm <= 0:
        raise ValueError("joint gradient norms must be finite/nonnegative and clip positive")
    norm = math.hypot(values[0], values[1])
    return norm, min(1.0, values[2] / (norm + 1e-6))


def _optimizers(optimizer):
    children = getattr(optimizer, "chained_optimizers", None)
    if children is None:
        yield optimizer
    else:
        for child in children:
            yield from _optimizers(child)


@torch.no_grad()
def step_text_with_fm_norm(engine, *, fm_grad_norm, max_norm,
                          rollout_policy_version, fm_gradient_policy_version):
    """Apply one global clip to Megatron's prepared FP32 master gradients.

    Returns the factor the FM owner must apply ONCE before its AdamW step.
    The caller publishes the next policy only after both optimizers complete.
    Native optimizer.step() is intentionally not called: it would clip text
    independently, whereas step_with_ready_grads() retains native parameter
    gathering and copies the updated masters back to the model.
    """
    if rollout_policy_version != fm_gradient_policy_version:
        raise ValueError("FM gradients were computed at a different policy version")
    combined_clip(0., fm_grad_norm, max_norm)  # validate before any mutation
    optimizer = engine.optimizer
    children = list(_optimizers(optimizer))
    for child in children:
        config = child.config
        if (getattr(config, "use_precision_aware_optimizer", False)
                or getattr(config, "overlap_param_gather_with_optimizer_step", False)):
            raise ValueError("v8 joint update requires ordinary FP32 master gradients")
        if math.isfinite(getattr(config, "grad_norm_skip_threshold", float("inf"))):
            raise ValueError("v8 does not skip finite gradients based on their magnitude")
    if optimizer.prepare_grads():
        raise RuntimeError("non-finite Megatron gradients; neither branch may step")
    text_norm = float(optimizer.get_grad_norm())
    norm, factor = combined_clip(text_norm, fm_grad_norm, max_norm)
    parameters = [p for child in children for p in child.get_parameters()]
    if len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("duplicate master parameter in joint gradient clipping")
    for p in parameters:
        if getattr(p, "grad_norm_group", None) is not None:
            raise ValueError("v8 joint update cannot use separate clipping groups")
        if p.grad is not None and p.grad.dtype != torch.float32:
            raise ValueError("v8 joint update requires FP32 master gradients")
    for p in parameters:
        if p.grad is not None:
            p.grad.mul_(factor)
    if not optimizer.step_with_ready_grads():
        raise RuntimeError("Megatron text step failed; do not publish a partial policy")
    engine.lr_scheduler_step()
    return dict(text_grad_norm=text_norm, fm_grad_norm=float(fm_grad_norm),
                grad_norm=norm, clip_factor=factor, text_optimizer_steps=1,
                policy_version=rollout_policy_version)
