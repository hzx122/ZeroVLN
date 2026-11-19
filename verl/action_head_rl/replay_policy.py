"""Experimental fixed-old-policy primitives; not wired into live training.

A multi-pass driver must bind cached probabilities to the behavior-policy
version and exact input/route/target identity, retain payloads across passes,
and separately count rollout batches and optimizer updates. These primitives
alone do not enable rollout reuse or establish GPU/learning equivalence.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor


class FrozenLogprobCache:
    """Per-batch CPU snapshots; consumers cannot overwrite the old likelihood."""
    def __init__(self, policy_version: int):
        if type(policy_version) is not int or policy_version < 0:
            raise ValueError('invalid behavior policy version')
        self.policy_version = policy_version
        self._entries = {}

    def put(self, key: str, identity_sha256: str, logprobs: Tensor):
        if not key or key in self._entries:
            raise ValueError('old-policy record requires a new nonempty key')
        if (not isinstance(identity_sha256, str) or len(identity_sha256) != 64
                or any(c not in '0123456789abcdef' for c in identity_sha256)):
            raise ValueError('exact input/route/target identity hash required')
        if not logprobs.is_floating_point() or not torch.isfinite(logprobs).all():
            raise ValueError('old-policy probabilities must be finite floating tensors')
        self._entries[key] = (identity_sha256, logprobs.detach().cpu().clone())

    def get(self, key: str, identity_sha256: str, *, policy_version: int) -> Tensor:
        if type(policy_version) is not int or policy_version != self.policy_version:
            raise ValueError('behavior policy version changed during replay')
        identity, value = self._entries[key]
        if identity != identity_sha256:
            raise ValueError('replay input/route/target identity changed')
        return value.clone()

    def clear(self):
        self._entries.clear()


def text_clipped_policy_loss(new_logprobs: Tensor, old_logprobs: Tensor,
                             response_mask: Tensor, advantage: Tensor, *,
                             sample_valid: Tensor | None = None,
                             loss_weight: Tensor | None = None,
                             coefficient: float = 1., global_weight: Tensor | float = 1.,
                             dp_size: int = 1, clip_ratio: float = .2):
    """Token PPO with the existing decision/trajectory weighting.

    Old probabilities are immutable, detached learner likelihoods collected
    before the first update. A future integration must check their relation
    to rollout likelihoods. Masked history/forced-tail positions contribute
    neither loss nor diagnostics. First-pass gradients match text GPG when
    old == new.detach(); loss values themselves need not match.
    """
    if (new_logprobs.ndim != 2 or old_logprobs.shape != new_logprobs.shape
            or response_mask.shape != new_logprobs.shape):
        raise ValueError('aligned [decision, token] probabilities/mask required')
    if old_logprobs.requires_grad:
        raise ValueError('old-policy likelihoods must be detached and frozen')
    if not math.isfinite(clip_ratio) or not 0 < clip_ratio < 1:
        raise ValueError('clip ratio must lie in (0, 1)')
    if (not math.isfinite(coefficient) or coefficient < 0
            or type(dp_size) is not int or dp_size < 1):
        raise ValueError('invalid coefficient or data-parallel size')
    mask = response_mask.to(device=new_logprobs.device)
    if not torch.all((mask == 0) | (mask == 1)) or (mask.sum(-1) <= 0).any():
        raise ValueError('binary mask with at least one response token required')
    mask = mask.bool()
    new = new_logprobs.float().masked_fill(~mask, 0.)
    old = old_logprobs.to(new).masked_fill(~mask, 0.)
    if not torch.isfinite(new).all() or not torch.isfinite(old).all():
        raise ValueError('nonfinite sampled-token likelihood')
    batch = new.shape[0]
    adv = advantage.to(new).detach().reshape(batch)
    valid = torch.ones_like(adv) if sample_valid is None else sample_valid.to(new).reshape(batch)
    weight = torch.ones_like(adv) if loss_weight is None else loss_weight.to(new).reshape(batch)
    if (not torch.isfinite(adv).all() or not torch.all((valid == 0) | (valid == 1))
            or not torch.isfinite(weight).all() or (weight < 0).any()):
        raise ValueError('invalid advantages, validity or loss weights')
    denominator = torch.as_tensor(global_weight, device=new.device, dtype=new.dtype)
    if denominator.numel() != 1 or not torch.isfinite(denominator).all() or denominator.item() <= 0:
        raise ValueError('positive finite original logical denominator required')
    log_ratio = new - old
    ratio = log_ratio.exp()
    if not torch.isfinite(ratio).all():
        raise ValueError('importance ratio overflow; refuse the replay update')
    gain = torch.minimum(ratio * adv[:, None],
                         ratio.clamp(1-clip_ratio, 1+clip_ratio) * adv[:, None])
    scores = (gain * mask).sum(-1) / mask.sum(-1)
    loss = -(scores * valid * weight).sum() * (coefficient * dp_size / denominator.reshape(()))
    diagnostic_mask = mask & ((valid * weight) > 0)[:, None]
    count = diagnostic_mask.sum().clamp_min(1)
    diagnostics = {
        'ratio_mean': (ratio.detach() * diagnostic_mask).sum() / count,
        'clip_fraction': (((ratio.detach()-1).abs() > clip_ratio) * diagnostic_mask).sum() / count,
        'approx_kl': (((ratio.detach()-1)-log_ratio.detach()) * diagnostic_mask).sum() / count,
    }
    return loss, diagnostics
