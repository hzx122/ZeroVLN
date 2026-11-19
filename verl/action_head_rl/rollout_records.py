"""Common episode record assembly for the driver and CPU queue consumers."""
from __future__ import annotations

from action_head_rl.decision_records import attach_text_advantage


def credit_metadata(rollout):
    """The existing call-credit algorithm needs no policy/image tensors."""
    import json
    metadata = {key: rollout[key] for key in ('seed', 'case', 'reward', 'reward_components',
        'result', 'partial_rollout', 'episode_seconds', 'slow_qwen_seconds',
        'rollout_queue_wait_seconds', 'actor_max_memory_gb') if key in rollout}
    metadata['chunks'] = [{key: row[key] for key in ('option_id', 'chunk_index', 'gym_feedback') if key in row}
                          for row in rollout['chunks']]
    metadata['decisions'] = [dict(decision_index=row['decision_index']) for row in rollout['decisions']]
    json.dumps(metadata, allow_nan=False)
    return metadata


def attach_episode_records(rollout, *, step, rollout_index, policy_version,
                           trajectory_advantage, option_advantages,
                           text_policy, text_loss_coef, check_fm_version,
                           allow_mixed_policy=False):
    """Bind existing samples to credit without copying their large payloads.

    Rule credit is applied by the caller after this common episode-normalized
    assembly. Keeping the input FM dictionaries preserves the diagnostic log's
    references to the very same records subsequently passed to apply_records.
    """
    records = []
    chunks = rollout['chunks']
    for record in chunks:
        source_version = int(record.get('policy_version', -1))
        if check_fm_version and ((not allow_mixed_policy and source_version != policy_version)
                or (allow_mixed_policy and not 0 <= source_version <= policy_version)):
            raise RuntimeError('rollout record policy-version mismatch: '
                               f"expected={policy_version}, actual={record.get('policy_version')}")
        if allow_mixed_policy:
            if 'old_logprob' not in record:
                raise ValueError('partial FM chunk lacks sampled behavior probability')
            record['update_policy_version'] = policy_version
        record['condition_id'] = (
            f"step={step}/rollout={rollout_index}/option={record.get('option_id')}")
        option_id = str(record.get('option_id') or '')
        record['advantage'] = (trajectory_advantage if trajectory_advantage is not None
                               else float(option_advantages.get(option_id, 0.)))
        if trajectory_advantage is not None:
            record['loss_weight'] = 1. / max(len(chunks), 1)
        credit = next((item for item in rollout['reward_components']['option_credits']
                       if str(item['option_id']) == option_id), None)
        if credit is not None:
            record['option_credit'] = dict(credit)
        records.append(record)
    if text_policy:
        decisions = rollout.get('decisions')
        if not decisions:
            raise RuntimeError('v6 joint rollout has no trainable high-level decisions')
        records.extend(attach_text_advantage(decisions, advantage=trajectory_advantage,
            step=step, rollout_index=rollout_index, policy_version=policy_version,
            coefficient=text_loss_coef, allow_mixed_policy=allow_mixed_policy))
    return records


def prepare_v3_group(rollouts, *, step, group_index, policy_version,
                     max_tool_calls, reward_version, text_loss_coef, allow_mixed_policy=False):
    """Apply the production v3 rule credit and whole-trajectory text merge.

    Strict same-policy is the default. Explicit partial mode preserves actual
    source versions and requires behavior probabilities for corrected loss.
    """
    from action_head_rl.rule_credit import assign_group
    if reward_version not in ('v3_first_arrival', 'v3_terminal_local'):
        raise ValueError('queue credit preparation requires the v3 reward contract')
    if policy_version != step - 1:
        raise ValueError('queue credit must bind the acknowledged rollout policy')
    scores = assign_group(rollouts, step=step, max_tool_calls=max_tool_calls, version=reward_version)
    episodes = []
    for sample, (rollout, credit) in enumerate(zip(rollouts, scores, strict=True)):
        records = prepare_v3_episode(rollout, credit=credit, step=step,
            rollout_index=group_index * len(rollouts) + sample, policy_version=policy_version,
            text_loss_coef=text_loss_coef, allow_mixed_policy=allow_mixed_policy)
        episodes.append(records)
    return episodes, scores


def prepare_v3_episode(rollout, *, credit, step, rollout_index, policy_version,
                       text_loss_coef, allow_mixed_policy=False):
    from action_head_rl.rule_credit import apply_records
    from action_head_rl.trajectory_text import merge_trajectory_text
    records = attach_episode_records(rollout, step=step,
        rollout_index=rollout_index, policy_version=policy_version,
        trajectory_advantage=0., option_advantages=None, text_policy=True,
        text_loss_coef=text_loss_coef, check_fm_version=True, allow_mixed_policy=allow_mixed_policy)
    # Both branches use exactly the original complete-group call LOO credit.
    apply_records(records, credit)
    return merge_trajectory_text(records, allow_mixed_policy=allow_mixed_policy)
