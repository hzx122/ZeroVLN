"""Exact masked language likelihood and separate navigation/STOP/path losses."""
from __future__ import annotations

import math
import torch
from action_head_rl.executable_policy import DecisionGrammar, POLICY_SCHEMA, codec


def selected_policy_statistics(hidden, lm_head, input_ids, response_mask,
                               temperature, metadata):
    """Return [B,T,5]: nav logp/mask, STOP/CONTINUE logp, choice mask.

    Navigation optimizes the selected continuation conditional on CONTINUE;
    it cannot suppress STOP by increasing an unconditional AH probability.
    Only the binary loss trains P(STOP) vs sum P(legal continuation tools).
    """
    if hidden.shape[:2] != input_ids.shape or response_mask.shape != input_ids.shape:
        raise ValueError('executable policy replay tensors must align')
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError('executable policy temperature must be positive')
    meta = metadata.detach().cpu().reshape(-1, 2).tolist()
    if len(meta) != input_ids.shape[0] or any(m not in ([1, 0], [1, 1]) for m in meta):
        raise ValueError('executable metadata must align with decision batch')
    ids_cpu = input_ids.detach().cpu()
    mask_cpu = response_mask.detach().cpu().bool()
    where = response_mask.bool().nonzero(as_tuple=False)
    if where.numel() == 0 or (where[:, 1] == 0).any():
        raise ValueError('executable replay needs forwarded response tokens')
    logits = lm_head(hidden[where[:, 0], where[:, 1] - 1]).float() / float(temperature)
    if logits.shape[-1] != codec().vocab_size:
        raise ValueError('learner vocabulary differs from sampler grammar')
    result = logits.new_zeros((*input_ids.shape, 5))
    offset = 0
    for batch in range(input_ids.shape[0]):
        positions = mask_cpu[batch].nonzero().flatten().tolist()
        if not positions or positions != list(range(positions[0], positions[-1] + 1)):
            raise ValueError('executable grammar requires an exact contiguous completion')
        state = DecisionGrammar(dict(schema=POLICY_SCHEMA, allow_turn=bool(meta[batch][1])))
        for pos in positions:
            token = int(ids_cpu[batch, pos])
            allowed, kind = state.allowed()
            row = logits[offset]
            # Clone before applying the exact inference mask; autograd must
            # retain all allowed logits for the softmax normalizer.
            masked = row.clone()
            state.mask_logits(masked)
            if kind == 'choice':
                stop = state.codec.choice['stop']
                continues = [idx for idx in allowed if idx != stop]
                normalizer = torch.logsumexp(masked[list(allowed)], 0)
                result[batch, pos, 2] = masked[stop] - normalizer
                continue_z = torch.logsumexp(masked[continues], 0)
                result[batch, pos, 3] = continue_z - normalizer
                result[batch, pos, 4] = 1.
                from .final_credit import ablated
                joint_choice = ablated('joint_tool_choice')
                if token != stop and (joint_choice or len(continues) > 1):
                    result[batch, pos, 0] = masked[token] - (normalizer if joint_choice else continue_z)
                    result[batch, pos, 1] = 1.
            elif kind == 'body' and state.tool != 'stop':
                # JSON punctuation with exactly one legal token is forced,
                # and has neither a gradient nor normalization mass.
                if int(torch.isfinite(masked).sum().item()) > 1:
                    result[batch, pos, 0] = masked[token] - torch.logsumexp(masked, 0)
                    result[batch, pos, 1] = 1.
            state.accept(token)
            offset += 1
        if not state.complete:
            raise ValueError('learner received incomplete executable tool call')
    return result


def final_text_loss(statistics, response_mask, advantage, *, sample_valid=None,
                    loss_weight=None, coefficient=1., global_weight=1., dp_size=1,
                    stop_target=None, stop_weight=None, path_advantage=None):
    if statistics.ndim != 3 or statistics.shape[-1] != 5:
        raise ValueError('final text loss requires constrained policy statistics')
    if statistics.shape[:2] != response_mask.shape:
        raise ValueError('final text loss mask differs from policy statistics')
    batch = statistics.shape[0]
    if stop_target is None or stop_weight is None or path_advantage is None:
        raise ValueError('final text loss needs independent STOP and path provenance')
    def vector(value, default=1.):
        return (statistics.new_full((batch,), default) if value is None
                else torch.as_tensor(value, device=statistics.device).reshape(batch).to(statistics))
    target, sw = vector(stop_target), vector(stop_weight)
    adv, path, valid, weight = map(vector, (advantage, path_advantage, sample_valid, loss_weight))
    if not (((target == 0) | (target == 1)).all() and (sw >= 0).all()
            and (weight >= 0).all() and (valid >= 0).all()):
        raise ValueError('invalid final text labels or weights')
    if not all(torch.isfinite(v).all() for v in (adv, path, valid, weight, sw)):
        raise ValueError('non-finite final text advantage or weight')
    choice = statistics[..., 4]
    if not torch.all(choice.sum(-1) == 1):
        raise ValueError('each final decision must have exactly one STOP/CONTINUE choice')
    nav_mask = statistics[..., 1] * response_mask.to(statistics)
    nav_score = (statistics[..., 0] * nav_mask).sum(-1) / nav_mask.sum(-1).clamp_min(1.)
    stop_lp = (statistics[..., 2] * choice).sum(-1)
    continue_lp = (statistics[..., 3] * choice).sum(-1)
    nav_loss = -(nav_score * adv.detach() * weight * valid).sum()
    path_loss = -(nav_score * path.detach() * weight * valid).sum()
    stop_loss = -((target * stop_lp + (1 - target) * continue_lp) * sw * valid).sum()
    denominator = torch.as_tensor(global_weight, device=statistics.device, dtype=statistics.dtype)
    if denominator.numel() != 1 or denominator.item() <= 0 or coefficient < 0:
        raise ValueError('final loss requires positive original episode count')
    return (nav_loss + path_loss + stop_loss) * (float(coefficient) * int(dp_size) / denominator)


def pack_text_metadata(record, *, padding):
    from action_head_rl.executable_policy import enabled
    metadata = record.get('text_executable_policy')
    if metadata is None:
        if enabled():
            raise ValueError('final learner record lacks masked sampling metadata')
        return {}
    for key in ('text_stop_target', 'text_stop_weight', 'text_path_advantage'):
        if key not in record:
            raise ValueError('final record lacks separate credit: ' + key)
    return dict(text_executable_policy=torch.as_tensor(metadata, dtype=torch.long).reshape(1, 2),
                text_stop_target=torch.tensor([record['text_stop_target']], dtype=torch.float32),
                text_stop_weight=torch.tensor([record['text_stop_weight']], dtype=torch.float32),
                text_path_advantage=torch.tensor([record['text_path_advantage']], dtype=torch.float32))


def copy_joint_metadata(data, natural):
    for name in ('executable_policy', 'stop_target', 'stop_weight', 'path_advantage'):
        key = 'text_' + name
        if key in natural:
            data['joint_' + key] = natural[key].unsqueeze(0)


def path_policy_loss(new_logprob, old_logprob, path_advantage, weighted_valid,
                     *, clip, global_valid, dp_size):
    """Independent successful-trajectory efficiency objective; no joint clipping."""
    path = path_advantage.reshape_as(new_logprob).to(new_logprob).detach()
    if not torch.isfinite(path).all():
        raise ValueError('non-finite success-only path advantage')
    ratio = (new_logprob.float() - old_logprob.float()).clamp(-20, 20).exp()
    surrogate = torch.minimum(ratio * path, ratio.clamp(1 - clip, 1 + clip) * path)
    return -(surrogate * weighted_valid).sum() * dp_size / global_valid.clamp_min(1.)
