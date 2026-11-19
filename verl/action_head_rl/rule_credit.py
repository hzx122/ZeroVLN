"""Rule-only tool-boundary returns with uniform collision-adjusted action credit.

No fitted critic, group-standard-deviation scaling, or privileged actor inputs.
"""
from __future__ import annotations
from collections import Counter
import math
import os
from typing import Mapping
from action_head_rl.tool_credit import build_call_ledger

MODE='rule_call_loo'
SCHEMA='v6-rule-call-credit-v2'
ACTION_CREDIT='call_advantage_minus_collision_clipped_v1'
ALPHA=.2
INFO_RATIO=.25
COLLISION=.25
VERSIONS=('v2', 'v3_first_arrival', 'v3_terminal_local', 'v3_reject_guard', 'v3_reject_guard_stop', 'v4_boundary_stop_spl')
GUARD_VERSIONS=('v3_reject_guard', 'v3_reject_guard_stop')

REWARD_ENV = dict(distance='ACTION_HEAD_RULE_DISTANCE_COEFFICIENT',
                  information='ACTION_HEAD_RULE_INFORMATION_COEFFICIENT')


def sequence_rloo_enabled(version):
    return version == 'v3_terminal_local' and os.environ.get('ACTION_HEAD_SEQUENCE_RLOO') == '1'


def reward_coefficients():
    """Independent, explicit coefficients shared by driver and CPU Ray workers."""
    values = {name:float(os.environ.get(key) or default) for name,key,default in (
        ('distance', REWARD_ENV['distance'], ALPHA),
        ('information', REWARD_ENV['information'], ALPHA*INFO_RATIO))}
    if any(not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError('rule reward coefficients must be finite and nonnegative')
    return values

def contract(*, selective_backward=False, version='v2'):
    if version == 'v4_boundary_stop_spl':
        from .final_credit import contract as final_contract
        return final_contract(selective_backward=selective_backward)
    if version not in VERSIONS:
        raise ValueError(f'unknown rule credit version: {version}')
    spec = dict(schema=SCHEMA,mode=MODE,alpha=ALPHA,information_ratio=INFO_RATIO,
                information_cap_entities=5,collision_coefficient=COLLISION,
                baseline='same_case_depth_leave_one_out_absorbing_zero',gamma=1.,
                critic_parameters=0,action_qwen_gradient=('selective_action_leaves_shared_graph_v1' if selective_backward else 'discard_fm_sharded_gradients_v1'),
                fm_normalization='episode_option_chunk_mean',action_credit=ACTION_CREDIT,
                action_advantage_clip=[-1.,1.])
    if version != 'v2':
        spec.update(schema='v6-rule-call-credit-v3-first-arrival',
                    first_arrival_reward=float(os.environ.get('ACTION_HEAD_RULE_FIRST_ARRIVAL_REWARD', '1.')), first_arrival_limit='once_per_episode',
                    arrival_evidence='official_goal_membership_native_trail_points_v1',
                    initial_arrival_bonus=False, terminal_success_reward=float(os.environ.get('ACTION_HEAD_RULE_TERMINAL_SUCCESS_REWARD', '1.')),
                    success_metric='unchanged_native_terminal_success')
    if version == 'v3_terminal_local':
        # Keep the existing recipe name, but version the corrected objective:
        # the event is local to the final call; its return reaches every prefix.
        spec.update(schema='v6-rule-call-credit-v3-terminal-rtg-v1',
                    terminal_reward_assignment='last_call_immediate_native_success',
                    credit_target='return_to_go_including_terminal_reward')
    if version in GUARD_VERSIONS:
        spec.update(schema='v6-rule-call-credit-'+version.replace('_','-'),
                    rejected_text_advantage='min_raw_advantage_zero',
                    rejection_evidence='harness_model_protocol_rejection_v1',
                    rejected_action_advantage='unchanged_raw_call_advantage',
                    text_normalization='unchanged_original_episode_decision_count',
                    terminal_reward_condition=('agent_legal_stop_and_native_success'
                        if version == 'v3_reject_guard_stop' else 'native_terminal_success'))
    if sequence_rloo_enabled(version):
        spec.update(sequence_rloo=True, baseline='same_case_sequence_leave_one_out',
                    credit_target='whole_episode_shaped_return', standard_deviation_normalization=False)
    coefficients = reward_coefficients()
    if coefficients != dict(distance=ALPHA, information=ALPHA*INFO_RATIO):
        spec.update(reward_coefficients=coefficients, alpha=coefficients['distance'],
                    information_ratio=(coefficients['information']/coefficients['distance']
                        if coefficients['distance'] else None))
    return spec

def _state(value):
    if not isinstance(value,Mapping) or value.get('schema') not in ('sage-rule-credit-state-v1', 'sage-rule-credit-state-v2', 'mp3d-rule-credit-state-v1'):
        raise ValueError('rule credit requires private state from the rule-enabled gym')
    return value

def _protocol_rejection_reason(turn):
    """Recognize harness protocol rejections, never a generic failed service.

    The broad rule_credit.rejected flag also includes infrastructure failures;
    it is deliberately insufficient evidence for suppressing actor credit.
    """
    calls=turn.get('calls') or []
    raw=turn.get('tool_call_error') if not calls else calls[0].get('result')
    if not isinstance(raw,Mapping) or raw.get('ok') is not False:
        return None
    if calls and calls[0].get('ok') is not False:
        return None
    name=calls[0].get('name') if calls else None
    reason=raw.get('return_reason')
    if (raw.get('schema')=='tool_call_rejection_v1' and reason=='rejected'
            and raw.get('action_executed') is False
            and raw.get('steps_used')==0 and raw.get('budget_cost')==1):
        code=raw.get('error_code')
        if not calls and code=='invalid_tool_call':
            return code
        if calls and raw.get('tool')==name and code=='invalid_tool_arguments':
            return code
    if (name=='turn' and raw.get('tool')=='turn'
            and raw.get('schema') in ('turn_tool_result_v1','turn_tool_result_v2')
            and raw.get('steps_used')==0):
        if reason in ('consecutive_turn_limit_reached','turn_not_available'):
            return reason
        if reason=='rejected' and isinstance(raw.get('error'),str) and raw['error']:
            return 'invalid_tool_arguments'
    if (name=='run_action_head' and raw.get('tool')==name and reason=='rejected'
            and str(raw.get('schema','')).startswith('action_head_tool_result_v')
            and all(raw.get(k)==0 for k in ('chunks','steps_used','path_m','net_m'))
            and isinstance(raw.get('error'),str) and raw['error']):
        return 'invalid_tool_arguments'
    return None

def _agent_stopped(result):
    """Require the model's accepted STOP, excluding evaluator terminal scoring."""
    flag=result.get('agent_stopped')
    if type(flag) is not bool or flag!=(result.get('why')=='agent_stop'):
        raise ValueError('guard credit requires consistent explicit agent_stopped evidence')
    if flag:
        last=result['turns'][-1]['calls'][-1]
        if (last.get('name')!='stop' or last.get('ok') is False
                or not isinstance(last.get('score_info'),Mapping)
                or _protocol_rejection_reason(result['turns'][-1])):
            raise ValueError('agent_stopped requires a final accepted model STOP')
    return flag

def score_episode(rollout, *, step, max_tool_calls=50, version='v2'):
    if version == 'v4_boundary_stop_spl':
        from .final_credit import score_episode as final_score
        return final_score(rollout, step=step, max_tool_calls=max_tool_calls)
    spec=contract(version=version)
    coefficients=reward_coefficients()
    baseline_coefficients = coefficients == dict(distance=ALPHA, information=ALPHA*INFO_RATIO)
    guarded=version in GUARD_VERSIONS
    result=rollout['result']
    ledger=build_call_ledger(result,rollout['chunks'],step=step,policy_version=step-1,max_tool_calls=max_tool_calls)
    initial=_state(result['initial_info'].get('rule_credit_state'))
    spec['geometry_state_schema'] = initial['schema']
    if initial['schema'] == 'mp3d-rule-credit-state-v1':
        spec['distance_metric'] = 'habitat_native_goal_geodesic'
        spec['arrival_evidence'] = 'habitat_native_goal_geodesic_3m_actual_steps_v1'
        spec['information_reward_available'] = initial.get('information_available') is True
        spec['information_evidence'] = initial.get('information_contract')
    if initial['schema'] == 'sage-rule-credit-state-v2':
        spec['distance_metric'] = 'supported_target_navmesh_geodesic'
        if version != 'v2':
            spec['arrival_evidence'] = 'supported_goal_membership_native_xyz_v2'
    turns={int(row['turn']):row for row in result['turns']}
    if any(len(row.get('calls') or [])>1 for row in turns.values()):
        raise ValueError('rule credit requires one tool call per sampled response')
    pairs=[]
    for call in ledger.calls:
        evidence=turns[call.decision_index].get('rule_credit') or {}
        before,after=_state(evidence.get('before')),_state(evidence.get('after'))
        pairs.append((call,evidence,before,after))
    states=[initial]+[pair[3] for pair in pairs]
    if version != 'v2':
        for state in states:
            if (state.get('arrival_evidence') != ('habitat_native_goal_geodesic_3m_actual_steps_v1' if state.get('schema') == 'mp3d-rule-credit-state-v1'
                    else 'supported_goal_membership_native_xyz_v2' if state.get('schema') == 'sage-rule-credit-state-v2'
                    else 'official_goal_membership_native_trail_points_v1')
                    or type(state.get('ever_goal_reached')) is not bool):
                raise ValueError('v3 requires native-trail arrival evidence; no distance-threshold fallback')
    for state in states:
        if any(state.get(k)!=initial.get(k) for k in ('schema','map_sha256','goal_contract_sha256','target_object_id')):
            raise ValueError('rule credit map/goal changed inside an episode')
    distance_available=all(isinstance(s.get('goal_distance_m'),(int,float)) and math.isfinite(s['goal_distance_m']) and s['goal_distance_m']>=0 for s in states)
    info_available=all(s.get('information_available') is True for s in states)
    D=max(initial['goal_distance_m'],1.) if distance_available else 1.
    seen=set(initial.get('relevant_entities') or [])
    discoveries=0; previous=initial; scored=[]
    for call,evidence,before,after in pairs:
        if before!=previous:
            raise ValueError('discontinuous private call-state chain')
        previous=after
        rejected=call.tool_name=='rejected' or evidence.get('rejected',False)
        active=call.tool_name not in ('stop','rejected') and not rejected
        dp=(D/(D+after['goal_distance_m'])-D/(D+before['goal_distance_m'])) if distance_available and active else 0.
        if call.tool_name=='turn':
            if abs(dp)>1e-6:
                raise ValueError('turn unexpectedly changed geodesic position')
            dp=0.
        new=set(after.get('relevant_entities') or [])-seen
        di=(min(5,discoveries+len(new))-min(5,discoveries))/5 if info_available and active else 0.
        if active:
            discoveries+=len(new)
        seen.update(after.get('relevant_entities') or [])
        collided=int(after['collisions'])>int(before['collisions'])
        if int(after['collisions'])<int(before['collisions']):
            raise ValueError('collision counter decreased')
        arrival=0.
        if version != 'v2':
            if before['ever_goal_reached'] and not after['ever_goal_reached']:
                raise ValueError('arrival ledger must be monotonic within an episode')
            if active and not before['ever_goal_reached'] and after['ever_goal_reached']:
                if call.tool_name == 'turn':
                    raise ValueError('turn cannot cause first positional arrival')
                arrival=spec['first_arrival_reward']
        scored.append(dict(call_id=call.call_id,decision_index=call.decision_index,
            tool=call.tool_name,option_id=call.option_id,chunk_count=call.chunk_count,
            distance_reward=coefficients['distance']*dp,information_reward=coefficients['information']*di,
            process_reward=(ALPHA*(dp+INFO_RATIO*di) if baseline_coefficients else
                coefficients['distance']*dp+coefficients['information']*di)+arrival,collision=collided))
        if version != 'v2':
            scored[-1]['first_arrival_reward']=arrival
        if guarded:
            rejection_reason=_protocol_rejection_reason(turns[call.decision_index])
            scored[-1].update(protocol_rejected=rejection_reason is not None,
                              protocol_rejection_reason=rejection_reason)
    agent_stopped=_agent_stopped(result) if guarded else None
    terminal_reward=float(os.environ.get('ACTION_HEAD_RULE_TERMINAL_SUCCESS_REWARD', '1.')) * float(ledger.native_success and
                          (version!='v3_reject_guard_stop' or agent_stopped))
    terminal_local = version == 'v3_terminal_local'
    total=0. if terminal_local else terminal_reward
    for row in reversed(scored):
        if terminal_local:
            # Pay once at the final call, then include it in the same RTG as
            # process rewards. Budget exhaustion and STOP both qualify.
            row['terminal_reward'] = terminal_reward if row is scored[-1] else 0.
            row['immediate_reward'] = row['process_reward'] + row['terminal_reward']
            total += row['immediate_reward']
        else:
            total += row['process_reward']
        row['return_to_go'] = total
        if terminal_local:
            row['credit_target'] = total
    score=dict(schema=spec['schema'],calls=scored,native_success=ledger.native_success,
                distance_available=distance_available,information_available=info_available,
                shaped_return=total,rule_state_seconds=sum(float(s.get('seconds',0)) for s in states))
    if not baseline_coefficients:
        score['reward_coefficients']=coefficients
    if guarded:
        score.update(agent_stopped=agent_stopped,
                     active_stop_success=agent_stopped and ledger.native_success,
                     terminal_reward=terminal_reward,
                     ever_goal_reached=states[-1]['ever_goal_reached'],
                     protocol_rejection_calls=sum(c['protocol_rejected'] for c in scored))
    return score

def assign_group(rollouts, *, step, max_tool_calls=50, version='v2'):
    if version == 'v4_boundary_stop_spl':
        from .final_credit import assign_group as final_assign
        return final_assign(rollouts, step=step, max_tool_calls=max_tool_calls)
    if len(rollouts)<2:
        raise ValueError('rule LOO requires at least two independent trajectories')
    ids={(r['result']['scene'],r['result']['episode_id']) for r in rollouts}
    seeds=[r['seed'] for r in rollouts]
    if len(ids)!=1 or len(set(seeds))!=len(seeds):
        raise ValueError('rule LOO needs one case and distinct trajectory seeds')
    scores=[score_episode(r,step=step,max_tool_calls=max_tool_calls,version=version) for r in rollouts]
    target_key = 'credit_target' if version == 'v3_terminal_local' else 'return_to_go'
    sequence = sequence_rloo_enabled(version)
    rewards = [s['shaped_return'] for s in scores]
    for i,score in enumerate(scores):
        if sequence:
            # Differences keep identical-return groups exactly zero without a tolerance.
            sequence_advantage = math.fsum(rewards[i]-v for j,v in enumerate(rewards) if j!=i)/(len(scores)-1)
            sequence_baseline = math.fsum(v for j,v in enumerate(rewards) if j!=i)/(len(scores)-1)
            score.update(sequence_rloo=True, sequence_reward=rewards[i],
                         sequence_baseline=sequence_baseline, sequence_advantage=sequence_advantage,
                         sequence_group_size=len(scores), sequence_zero_group=len(set(rewards))==1,
                         trajectory_call_count=len(score['calls']))
        text,options={},{}
        for t,row in enumerate(score['calls']):
            baseline=(sequence_baseline if sequence else
                      sum(s['calls'][t][target_key] if t<len(s['calls']) else 0.
                          for j,s in enumerate(scores) if j!=i)/(len(scores)-1))
            advantage=sequence_advantage if sequence else row[target_key]-baseline
            effective=advantage
            if version in GUARD_VERSIONS:
                if row['protocol_rejected']:
                    effective=min(advantage,0.)
                row.update(raw_text_advantage=advantage,effective_text_advantage=effective,
                           text_advantage_guarded=effective!=advantage)
            row.update(baseline=baseline,text_advantage=effective)
            text[row['decision_index']]=effective
            if row['chunk_count']:
                # The guard modifies only the text branch. Keep the original
                # collision-adjusted action objective and its normalization.
                action=max(-1.,min(1.,advantage-COLLISION*row['collision']))
                options[row['option_id']]=action; row['action_advantage']=action
                if sequence:
                    preclip=advantage-COLLISION*row['collision']
                    row.update(action_advantage_preclip=preclip, action_advantage_clipped=action!=preclip)
        score.update(text_advantages=text,option_advantages=options)
        expected={int(r['decision_index']) for r in rollouts[i].get('decisions',[])}
        if expected!=set(text):
            raise ValueError('rule credit text/ledger mismatch')
        if set(options)!={r['option_id'] for r in rollouts[i]['chunks']}:
            raise ValueError('rule credit option/ledger mismatch')
    return scores

def apply_records(records, credit):
    if credit.get('schema') == 'v4-boundary-stop-spl-credit-v1':
        from .final_credit import apply_records as final_apply
        return final_apply(records, credit)
    from action_head_rl.text_policy import is_text_record
    counts=Counter(str(r['option_id']) for r in records if not is_text_record(r))
    for row in records:
        if is_text_record(row):
            row['advantage']=credit['text_advantages'][int(row['decision_index'])]
        else:
            option=str(row['option_id'])
            row['advantage']=credit['option_advantages'][option]
            row['loss_weight']=1./(len(counts)*counts[option])


def summarize(scores):
    if any(s.get('schema') == 'v4-boundary-stop-spl-credit-v1' for s in scores):
        from .final_credit import summarize as final_summarize
        return final_summarize(scores)
    schemas={s['schema'] for s in scores}
    if len(schemas)>1:
        raise ValueError('cannot mix rule credit versions in one update')
    calls=[c for s in scores for c in s['calls']]
    actions=[c for c in calls if c['chunk_count']]
    advantages=[c['text_advantage'] for c in calls]
    result=dict(schema=next(iter(schemas),SCHEMA),mode=MODE,action_credit=ACTION_CREDIT,
        episodes=len(scores),calls=len(calls),action_calls=len(actions),
        collision_calls=sum(c['collision'] for c in actions),
        distance_available_episodes=sum(s['distance_available'] for s in scores),
        information_available_episodes=sum(s['information_available'] for s in scores),
        information_reward=sum(c['information_reward'] for c in calls),
        distance_reward=sum(c['distance_reward'] for c in calls),
        positive_calls=sum(v>0 for v in advantages),negative_calls=sum(v<0 for v in advantages),
        zero_calls=sum(v==0 for v in advantages),
        rule_state_seconds=sum(s['rule_state_seconds'] for s in scores))
    if schemas and SCHEMA not in schemas:
        result.update(first_arrival_reward=sum(c['first_arrival_reward'] for c in calls),
                      first_arrival_episodes=sum(any(c['first_arrival_reward'] for c in s['calls']) for s in scores))
    if schemas == {'v6-rule-call-credit-v3-terminal-rtg-v1'}:
        result.update(terminal_reward=sum(c['terminal_reward'] for c in calls),
                      native_success_episodes=sum(s['native_success'] for s in scores))
    if scores and 'terminal_reward' in scores[0]:
        rejected=[c for c in calls if c['protocol_rejected']]
        result.update(native_success_episodes=sum(s['native_success'] for s in scores),
            agent_stop_episodes=sum(s['agent_stopped'] for s in scores),
            active_stop_success_episodes=sum(s['active_stop_success'] for s in scores),
            ever_goal_reached_episodes=sum(s['ever_goal_reached'] for s in scores),
            terminal_reward=sum(s['terminal_reward'] for s in scores),
            protocol_rejection_calls=len(rejected),
            protocol_rejection_reasons=dict(Counter(c['protocol_rejection_reason'] for c in rejected)),
            guarded_calls=sum(c['text_advantage_guarded'] for c in calls),
            rejected_positive_calls_raw=sum(c['raw_text_advantage']>0 for c in rejected),
            rejected_positive_calls_effective=sum(c['text_advantage']>0 for c in rejected))
        for label,key in (('raw','raw_text_advantage'),('effective','text_advantage')):
            result['rejected_positive_advantage_weight_'+label]=sum(
                max(c[key],0.)/len(s['calls'])
                for s in scores for c in s['calls'] if c['protocol_rejected'])
    sequence_scores=[s for s in scores if s.get('sequence_rloo')]
    if sequence_scores:
        values=[s['sequence_advantage'] for s in sequence_scores]
        mean=math.fsum(values)/len(values)
        groups=math.fsum(1./s['sequence_group_size'] for s in sequence_scores)
        zero_groups=math.fsum(float(s['sequence_zero_group'])/s['sequence_group_size'] for s in sequence_scores)
        sequence_actions=[c for s in sequence_scores for c in s['calls'] if c['chunk_count']]
        clipped=sum(c['action_advantage_clipped'] for c in sequence_actions)
        result.update(sequence_rloo=True, sequence_advantage_mean=mean,
            sequence_advantage_std=math.sqrt(math.fsum((v-mean)**2 for v in values)/len(values)),
            sequence_advantage_min=min(values), sequence_advantage_max=max(values),
            sequence_advantage_abs_mean=math.fsum(abs(v) for v in values)/len(values),
            sequence_groups=groups, sequence_zero_groups=zero_groups,
            sequence_zero_group_fraction=zero_groups/groups,
            sequence_head_clipped_calls=clipped,
            sequence_head_clip_fraction=clipped/len(sequence_actions) if sequence_actions else 0.,
            sequence_trajectory_calls_mean=sum(s['trajectory_call_count'] for s in sequence_scores)/len(values))
    return result
