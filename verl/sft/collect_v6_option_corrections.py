"""v6 online corrections from observed collisions and replay-safe goal routes.

The opt-in reviewed and semantic-judge modes remain available for legacy v6
experiments. Privileged execution evidence stays out of model inputs.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from action_head import contract as A
from action_head import protocol as PROTOCOL
from action_head import protocol_v6 as P
from action_head.trajectory_map import MAP_CONTEXT_TEXT
from sft.golden_context_beam import validate_context
from sft.build_fm_nested_bundles import parse_tool_call_text

DATA_CONTRACT = 'v6_online_corrections_v1'
SELECTION_CONTRACT = 'v6_local_teacher_checked_v1'
COLLISION_CONTRACT = 'v6_collision_task_oracle_v1'
DIRECTION_CONTRACT = 'v6_direction_mismatch_task_oracle_v1'
PHYSICAL_PAIR_CONTRACT = 'v6_physical_failure_pair_v1'
GOLDEN_EXTRA_CONTRACT = 'v6_corrected_golden_slow_anchor_v1'
REVIEW_SCHEMA = 'v6_correction_review_v1'
FAILURE_MODES = ('collision', 'blocked', 'no_progress', 'direction_mismatch', 'overshoot')


def _legacy():
    # Lazy import keeps the existing executable's v3/v4.3 path unchanged.
    import collect_noncollision_option_corrections as C
    return C


def _canonical_messages(messages):
    result = copy.deepcopy(messages)
    for message in result:
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get('type') != 'image_url':
                continue
            url = str(part['image_url']['url'])
            if url.startswith('data:image/') and ';base64,' in url:
                raw = base64.b64decode(url.split(';base64,', 1)[1], validate=True)
            elif url.startswith('file://'):
                raw = Path(url[7:]).read_bytes()
            else:
                raise ValueError('condition binding requires archived local image bytes')
            part['image_url']['url'] = 'sha256:' + hashlib.sha256(raw).hexdigest()
    return result


def condition_binding(case, call, call_index, start_pose):
    """Bind an approval to observed pixels/text, call and physical start."""
    turn = call.get('turn') or {}
    if not turn.get('slow_messages'):
        raise ValueError('missing_slow_prefix')
    value = dict(protocol=P.protocol_identity(), scene=case['scene'],
                 episode_id=case['episode_id'], call_index=call_index,
                 start_pose=list(start_pose), arguments=call.get('arguments'),
                 name=call.get('name'), assistant=turn.get('assistant'),
                 messages=_canonical_messages(turn['slow_messages']))
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def load_reviews(path):
    if path is None:
        return {}
    reviews = {}
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('schema') != REVIEW_SCHEMA:
                raise ValueError('review schema mismatch')
            index = row.get('call_index')
            if type(index) is not int or index < 0:
                raise ValueError('review call_index must be a nonnegative integer')
            key = (row['scene'], row['episode_id'], index)
            if key in reviews:
                raise ValueError(f'duplicate reviewed decision: {key}')
            reviews[key] = row
    return reviews


def validate_review(review, case, call, index, start_pose):
    if review.get('schema') != REVIEW_SCHEMA or review.get('accepted') is not True:
        raise ValueError('teacher_unverified')
    if not str(review.get('reviewer') or '').strip() or not str(review.get('evidence') or '').strip():
        raise ValueError('review requires reviewer and evidence')
    if (review.get('scene'), review.get('episode_id'), review.get('call_index')) != (
            case['scene'], case['episode_id'], index):
        raise ValueError('review decision binding mismatch')
    expected = review.get('condition_start_world_pose')
    if (not isinstance(expected, list) or len(expected) != 3 or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in expected) or
            not _legacy().same_pose(tuple(expected), tuple(start_pose))):
        raise ValueError('review pose mismatch')
    if review.get('condition_sha256') != condition_binding(case, call, index, start_pose):
        raise ValueError('review condition binding mismatch')
    modes = review.get('approved_failure_modes', [])
    if not isinstance(modes, list) or any(mode not in FAILURE_MODES for mode in modes):
        raise ValueError('invalid reviewed failure modes')
    return review


def route_digest(route):
    """Stable geometry binding for explicit review of the generated trajectory."""
    return hashlib.sha256(json.dumps({key: list(map(float, route[key]))
        for key in ('xs', 'ys', 'yaws')}, sort_keys=True, allow_nan=False).encode()).hexdigest()


def execution_digest(call):
    """Reviewed failure categories bind to actual execution as well as inputs."""
    trace = call.get('action_trace') or {}
    value = {key: trace.get(key) for key in ('before_world_pose', 'after_world_pose',
        'option_actual_endpoint', 'option_endpoint_source', 'prefixes')}
    value['result'] = call.get('result')
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def validate_condition(call):
    """Reject changed or ambiguous calls instead of pairing their query with new text."""
    text = str((call.get('turn') or {}).get('assistant') or '')
    if len(A.TOOL_CALL_RE.findall(text)) != 1:
        raise ValueError('v6 correction requires one complete conditioning tool call')
    decision, _, reason = parse_tool_call_text(text, action_head_as_tool_version='v6')
    if decision != {'name': 'run_action_head', 'arguments': call.get('arguments')}:
        raise ValueError('condition_tool_call_mismatch:' + reason)


def failure_modes(call, review=None):
    C = _legacy()
    result, trace = call.get('result') or {}, call.get('action_trace') or {}
    prefixes = trace.get('prefixes') or []
    modes = []
    if result.get('collision') or any(row.get('collided') for row in prefixes):
        modes.append('collision')
    if result.get('blocked') or any(row.get('blocked') for row in prefixes):
        modes.append('blocked')
    no_progress = result.get('return_reason') in {'no_progress', 'no_executable_motion'}
    for row in prefixes:
        requested = float((row.get('requested_endpoint') or {}).get('path_m') or 0.)
        actual = float((row.get('actual_endpoint') or {}).get('path_m') or 0.)
        no_progress |= bool(row.get('no_progress')) or (
            requested >= .1 and actual < min(.02, requested * .1))
    if no_progress:
        modes.append('no_progress')
    planned = P.planned_option_direction(call.get('arguments') or {})
    actual, _ = C.stitched_direction(call, P)
    if planned in {'left', 'right', 'straight', 'turn_around'} and actual not in {
            'unknown', 'degenerate', planned}:
        modes.append('direction_mismatch')
    # Overshoot cannot be inferred from an arbitrary increase in task distance.
    # It requires review of this local target and the actual trajectory.
    for mode in (review or {}).get('approved_failure_modes', []):
        if mode not in modes:
            modes.append(mode)
    return modes


def observed_collision(call):
    """Only a collision reported by the executed action, not a blocked guess."""
    result, trace = call.get('result') or {}, call.get('action_trace') or {}
    return bool(result.get('collision') or any(
        prefix.get('collided') for prefix in trace.get('prefixes') or []))


def v43_direction_mismatch(call):
    """Reuse the v4.3 trigger gate, without its recovery (B) anchor."""
    C = _legacy()
    if call.get('name') != 'run_action_head' or C.execution_failed(call):
        return None
    expected = C.planned_direction(call.get('arguments') or {}, P)
    actual, endpoint = C.stitched_direction(call, P)
    if expected not in {'left', 'right', 'straight'} or actual in {
            'degenerate', 'unknown', expected}:
        return None
    if (C.call_pose(call) is None or C.call_pose(call, before=False) is None or
            (call.get('action_trace') or {}).get('option_endpoint_source') != 'world_pose'):
        return None
    return expected, actual, endpoint


def trajectory_map_path(messages):
    found = []
    for message in messages:
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for index, part in enumerate(content):
            if part.get('text') == MAP_CONTEXT_TEXT:
                following = content[index + 1:index + 2]
                url = (following[0].get('image_url') or {}).get('url', '') if following else ''
                if not url.startswith('file://'):
                    raise ValueError('v6 map must be an archived local image')
                found.append(url[7:])
    if len(found) != 1:
        raise ValueError('v6 requires exactly one trajectory map')
    validate_context(messages, found[0])
    return found[0]


def decision_pose(calls, index, initial_pose=None):
    C = _legacy()
    call = calls[index]
    direct = C.call_pose(call)
    if direct is not None:
        return direct
    direct = C.world_pose((call.get('execution_state_trace') or {}).get('before_world_pose'))
    if direct is not None:
        return direct
    # STOP/invalid decisions do not execute motion. Only use the immediately
    # preceding physical boundary, never a guessed or future state.
    if index:
        previous = calls[index - 1]
        if previous.get('ok') is True:
            return (C.call_pose(previous, before=False) or C.world_pose(
                (previous.get('execution_state_trace') or {}).get('after_world_pose')))
    return initial_pose if index == 0 else None


def high_level_records(calls, case, reviews, output, initial_pose=None):
    """Independent reviewed text SFT export; no model-generated pseudo-labels."""
    C = _legacy()
    audits, rows = [], []
    for index, call in enumerate(calls):
        audit = dict(scene=case['scene'], episode_id=case['episode_id'], call_index=index,
                     name=call.get('name'), accepted=False, reason='teacher_unverified')
        start = decision_pose(calls, index, initial_pose)
        if start is None:
            audit['reason'] = 'missing_physical_decision_pose'
            audits.append(audit)
            continue
        try:
            binding = condition_binding(case, call, index, start)
            audit.update(condition_sha256=binding, condition_start_world_pose=list(start))
            review = reviews.get((case['scene'], case['episode_id'], index))
            if not review or not review.get('teacher_tool_call'):
                audits.append(audit)
                continue
            validate_review(review, case, call, index, start)
            teacher = review['teacher_tool_call']
            name, args = teacher.get('name'), teacher.get('arguments')
            if name == 'run_action_head':
                valid, _ = A.validate_action_head_arguments(args)
            elif name == 'turn':
                valid, _ = PROTOCOL.validate_turn_arguments(args, 'v6')
            elif name == 'stop':
                valid, _ = P.validate_stop_arguments(args)
            else:
                valid = None
            if valid is None:
                raise ValueError('invalid_teacher_tool_call')
            messages = C.CR.externalize_messages(call['turn']['slow_messages'],
                Path(output) / 'high_level_images' / binding[:24])
            trajectory_map_path(messages)
            rows.append(dict(schema='v6_reviewed_high_level_sft_v1', **P.protocol_identity(),
                source_scene=case['scene'], source_task_id=case['episode_id'], split='train',
                messages=messages, completion=P.format_tool_call(name, valid),
                condition_sha256=binding, condition_start_world_pose=list(start),
                teacher_review=review))
            audit.update(accepted=True, reason='reviewed_high_level_label')
        except (ValueError, KeyError) as exc:
            audit['reason'] = str(exc)
        audits.append(audit)
    return audits, rows


def judge_local_route(url, model, timeout, call, image_paths):
    """Reuse the existing judge transport, with an explicit *local* contract."""
    C = _legacy()
    prompt = (
        'Review a candidate LOCAL action trajectory for the recorded run_action_head call. '
        'The original retained decision context appears first for grounding only. After the '
        'CANDIDATE ROUTE marker, the first image is the real decision state and later images follow the candidate '
        'supervised route in order, covering every future action horizon used in training. '
        'Accept only if the current target, where, and every relevant plan stage are grounded '
        'and the entire shown route fulfills that LOCAL request without passing a requested '
        'stopping/inspection point or approaching a different object. A route toward the overall '
        'task goal is NOT sufficient. Matching left/right/straight alone is NOT sufficient. '
        'If the target or local stopping point is ambiguous, reject. Do not rewrite the call. '
        'Return exactly LOCAL_ROUTE_ACCEPT or LOCAL_ROUTE_REJECT.\nCall: ' +
        json.dumps(call.get('arguments'), ensure_ascii=False))
    grounding = []
    for message in (call.get('turn') or {}).get('slow_messages', []):
        if message.get('role') == 'system':
            continue
        grounding.append({'type': 'text', 'text': 'RECORDED CONTEXT ' + str(message.get('role'))})
        content = message.get('content')
        if isinstance(content, list):
            grounding.extend(copy.deepcopy(content))
        elif isinstance(content, str):
            grounding.append({'type': 'text', 'text': content})
    response = C.CR.post_json(url.rstrip('/') + '/v1/chat/completions', {
        'model': model, 'temperature': 0., 'max_tokens': 16,
        'chat_template_kwargs': {'enable_thinking': False},
        'structured_outputs': {'choice': ['LOCAL_ROUTE_ACCEPT', 'LOCAL_ROUTE_REJECT']},
        'messages': [{'role': 'system', 'content': 'Only approve verified local action supervision.'},
                     {'role': 'user', 'content': [
                         *grounding, {'type': 'text', 'text': 'CANDIDATE ROUTE'},
                         *[{'type': 'image_url', 'image_url': {'url': 'file://' + str(p)}}
                           for p in image_paths], {'type': 'text', 'text': prompt}]}]}, timeout)
    answer = str(response['choices'][0]['message'].get('content') or '').strip()
    return answer == 'LOCAL_ROUTE_ACCEPT', answer


def build_candidate(*, calls, condition_index, trigger_index, sample_type, modes,
                    case, teacher_mode, reviews, judge_url, judge_model, timeout,
                    intervening_turns, golden_extra_max=0, golden_extra_seed=0,
                    golden_vllm_url=None, golden_vllm_model=None,
                    golden_query_cache=None, **kwargs):
    C = _legacy()
    current = calls[condition_index]
    start = C.call_pose(current)
    audit = dict(candidate_index=condition_index, sample_type=sample_type,
                 failure_modes=modes, accepted=False)
    if start is None or (current.get('ok') is not True and
                         not (teacher_mode == 'physical_oracle' and observed_collision(current))):
        return [], {**audit, 'reason': 'missing_executed_condition_state'}
    validate_condition(current)
    binding = condition_binding(case, current, condition_index, start)
    audit.update(condition_sha256=binding, condition_start_world_pose=list(start))
    if teacher_mode == 'audit_only':
        return [], {**audit, 'reason': 'teacher_unverified'}
    review = reviews.get((case['scene'], case['episode_id'], condition_index))
    if teacher_mode == 'reviewed':
        if review is None:
            return [], {**audit, 'reason': 'teacher_unverified'}
        validate_review(review, case, current, condition_index, start)
        if review.get('teacher_tool_call') and review['teacher_tool_call'] != {
                'name': current['name'], 'arguments': current['arguments']}:
            # This is valid independent text supervision, but its trajectory
            # cannot be paired with the original call's archived query.
            return [], {**audit, 'reason': 'review_changes_conditioning_toolcall'}
        goal = review.get('local_goal_xy')
        if (not isinstance(goal, list) or len(goal) != 2 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in goal)):
            return [], {**audit, 'reason': 'missing_reviewed_local_goal'}
    elif teacher_mode in ('semantic_judge', 'physical_oracle'):
        if teacher_mode == 'semantic_judge' and (not judge_url or not judge_model):
            raise ValueError('semantic_judge requires URL and model')
        goal = kwargs['task'].goal_xy
    else:
        raise ValueError(f'unknown teacher mode: {teacher_mode}')
    route, error = C.CR.oracle_route(kwargs['dataset'], case['scene'], start[:2],
                                    math.radians(start[2]), tuple(goal))
    if route is None:
        return [], {**audit, 'reason': error}
    route_start = C.CR.pose_at(route, 0.)
    if not C.same_pose(start, (route_start[0], route_start[1], math.degrees(route_start[2]))):
        return [], {**audit, 'reason': 'oracle_changed_condition_start'}
    if route['execution_replay'].get('collisions', 0):
        return [], {**audit, 'reason': 'oracle_execution_collision'}
    if not C.available_windows(route):
        return [], {**audit, 'reason': 'oracle_route_has_no_actionable_window'}
    if teacher_mode == 'physical_oracle':
        planned = C.planned_direction(current.get('arguments') or {}, P)
        oracle_direction, oracle_endpoint = C.oracle_prefix_direction(route, P)
        audit.update(planned_direction=planned, oracle_prefix_direction=oracle_direction,
                     oracle_prefix_endpoint=oracle_endpoint)
        if planned not in {'left', 'right', 'straight'}:
            return [], {**audit, 'reason': 'condition_toolcall_direction_unspecified'}
        if oracle_direction != planned:
            return [], {**audit, 'reason': 'condition_toolcall_oracle_direction_mismatch'}
    digest = route_digest(route)
    if teacher_mode == 'reviewed' and (review.get('route_review_scope') != 'entire_supervised_route'
                                      or review.get('route_sha256') != digest):
        return [], {**audit, 'reason': 'reviewed_route_not_bound', 'route_sha256': digest,
                    'candidate_route': {key: list(map(float, route[key]))
                                        for key in ('xs', 'ys', 'yaws')}}
    teacher = dict(mode=teacher_mode, condition_sha256=binding, route_sha256=digest,
                   goal_xy=list(goal), approved=teacher_mode in ('reviewed', 'physical_oracle'))
    if teacher_mode == 'physical_oracle':
        teacher.update(planned_direction=audit['planned_direction'],
                       oracle_prefix_direction=audit['oracle_prefix_direction'])
    if teacher_mode == 'reviewed':
        teacher['review'] = review
    elif teacher_mode == 'semantic_judge':
        # Last window starts at <=1.2m and predicts <=1.6m. Review that full
        # union rather than only the execution prefix used by the old gate.
        last = min(float(route['length_m']), C.available_windows(route)[-1][0] + 1.6)
        offsets = sorted(set([0., last] + [x * .4 for x in range(1, 8) if x * .4 < last]))
        evidence = [C.CR.render_pose(kwargs['render_url'], case['scene'], C.CR.pose_at(route, offset),
            Path(kwargs['output']) / 'teacher_evidence' / binding[:24] / f'{i:02d}.jpg',
            kwargs['preloaded']) for i, offset in enumerate(offsets)]
        accepted, answer = judge_local_route(judge_url, judge_model, timeout, current, evidence)
        teacher.update(approved=accepted, judge_model=judge_model, answer=answer,
                       evidence_images=[str(p) for p in evidence])
        if not accepted:
            return [], {**audit, 'reason': 'local_semantics_rejected', 'teacher': teacher}
    rows, built = C.build_rows(case=case, previous=calls[trigger_index], current=current,
        candidate_index=condition_index, sample_type=sample_type, trigger_index=trigger_index,
        intervening_turns=intervening_turns, protocol=P, prepared_route=route,
        teacher_audit=teacher, **kwargs)
    for row in rows:
        row.update(data_contract=DATA_CONTRACT, failure_modes=list(modes),
                   trajectory_map_path=trajectory_map_path(row['slow_messages']))
        row['bootstrap_provenance'].update(data_contract=DATA_CONTRACT,
                                           failure_modes=list(modes))
        row['bootstrap_provenance'][
            'route_validation' if teacher_mode == 'physical_oracle' else 'teacher_approval'] = teacher
    extra_audits = []
    if rows and teacher_mode == 'physical_oracle' and golden_extra_max:
        try:
            extra_rows, extra_audits = build_golden_extra(calls=calls,
                condition_index=condition_index, current=current, route=route,
                case=case, modes=modes, sample_type=sample_type,
                timeout=timeout, max_extra=golden_extra_max, seed=golden_extra_seed,
                vllm_url=golden_vllm_url, vllm_model=golden_vllm_model,
                query_cache=golden_query_cache, **kwargs)
            rows.extend(extra_rows)
        except Exception as exc:
            extra_audits = [dict(accepted=False, reason='golden_extra_error',
                error=f'{type(exc).__name__}:{exc}', parent_call_index=condition_index)]
    return rows, {**built, **audit, 'accepted': bool(rows),
                  'reason': built.get('reason'), 'teacher': teacher,
                  'golden_extra_audits': extra_audits}


def build_golden_extra(*, calls, condition_index, current, route, case, modes,
                       sample_type, timeout, max_extra, seed, vllm_url, vllm_model,
                       query_cache, evaluator, output, render_url, preloaded,
                       bootstrap_round, collection_iteration, npz_compress_level,
                       task, dataset):
    """Generate causal, independently conditioned slow anchors on one route."""
    from action_head.query_provider import VllmQueryStateProvider
    from sft import v6_golden_extra as G
    C = _legacy()
    if not vllm_url or not vllm_model or not query_cache:
        raise ValueError('golden extra anchors require vLLM model, URL and local query cache')
    source_query = (current.get('action_trace') or {}).get('query_artifact') or {}
    source_hidden = np.load(source_query['archive_path'], mmap_mode='r')
    if source_hidden.ndim != 2 or source_hidden.shape[0] != 4:
        raise ValueError('golden extra source must use the four-query v6 contract')
    provider = VllmQueryStateProvider(str(query_cache), hidden_size=source_hidden.shape[1],
                                    query_count=source_hidden.shape[0])
    key = f'{seed}:{bootstrap_round}:{collection_iteration}:{case["scene"]}:{case["episode_id"]}:{condition_index}:{route_digest(route)}'
    selected = set(G.sampled_offsets(route, key, max_extra))
    if not selected:
        return [], []
    root_map = trajectory_map_path(C.CR.externalize_messages(
        current['turn']['slow_messages'], Path(output) / 'golden_extra' / 'source_maps' /
        hashlib.sha256(key.encode()).hexdigest()[:16]))
    try:
        state = G.state_before_call(calls, condition_index, root_map)
        history = G.canonical_history(current['turn']['slow_messages'])
    except ValueError as exc:
        return [], [dict(accepted=False, reason=str(exc), route_offset_m=offset,
                         parent_call_index=condition_index) for offset in sorted(selected)]
    rows, audits = [], []
    previous_completion = current['turn']['assistant']
    previous_contract = current['arguments']
    directory = Path(output) / 'golden_extra' / hashlib.sha256(key.encode()).hexdigest()[:16]
    last_step = int(round(max(selected) / G.OPTION_M))
    for step in range(1, last_step + 1):
        start_m, offset = G.OPTION_M * (step - 1), round(G.OPTION_M * step, 6)
        option_id = f'golden:{case["episode_id"]}:{condition_index}:{step}'
        summary, frames = G.golden_feedback(route, start_m, offset, previous_contract,
            option_id, case['scene'], render_url, directory / f'option_{step:02d}', preloaded)
        P.append_tool_exchange(history, completion=previous_completion,
                               summary=summary, frames=frames)
        state.update_action_head(frames)
        # The generator uses the same request construction as v6 open-loop.
        try:
            decision, messages, reason = G.generate_decision(history=history, state=state,
                instruction=case['instruction'], map_path=directory / f'map_{step:02d}.png',
                vllm_url=vllm_url, vllm_model=vllm_model, timeout=timeout,
                query_provider=provider, max_attempts=8)
        except Exception as exc:
            audits.append(dict(accepted=False, reason='golden_slow_request_error',
                error=f'{type(exc).__name__}:{exc}', route_offset_m=offset,
                parent_call_index=condition_index))
            break
        if decision is None:
            audits.append(dict(accepted=False, reason='slow_decision:' + reason,
                route_offset_m=offset, parent_call_index=condition_index))
            break
        suffix = G.route_from_offset(route, offset)
        planned = C.planned_direction(decision['arguments'], P)
        oracle, _ = C.oracle_prefix_direction(suffix, P)
        previous_completion, previous_contract = decision['completion'], decision['arguments']
        if offset not in selected:
            continue
        synthetic_index = len(calls) + condition_index * 1000 + step
        pose = C.CR.pose_at(route, offset)
        query_path = directory / f'query_{step:02d}.npy'
        query_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(query_path, np.asarray(decision['query_hidden'].query_hidden, dtype=np.float32))
        synthetic = dict(name='run_action_head', arguments=decision['arguments'], ok=True,
            turn=dict(slow_messages=messages, assistant=decision['completion']),
            action_trace=dict(before_world_pose=dict(world_xy=[pose[0], pose[1]],
                world_yaw_deg=math.degrees(pose[2])),
                query_artifact=dict(archive_path=str(query_path),
                    request_id_with_digest=decision['query_hidden'].request_id)))
        binding = condition_binding(case, synthetic, synthetic_index,
                                      C.call_pose(synthetic))
        teacher = dict(mode='physical_oracle', approved=True,
            condition_sha256=binding, route_sha256=route_digest(suffix),
            parent_route_sha256=route_digest(route), parent_call_index=condition_index,
            route_offset_m=offset, planned_direction=planned,
            oracle_prefix_direction=oracle, goal_xy=list(task.goal_xy))
        try:
            new_rows, detail = C.build_rows(case=case, previous=current, current=synthetic,
                candidate_index=synthetic_index, sample_type='golden_route_relabel',
                trigger_index=condition_index, intervening_turns=[], protocol=P,
                prepared_route=suffix, teacher_audit=teacher, evaluator=evaluator,
                output=output, task=task, dataset=dataset, render_url=render_url,
                preloaded=preloaded, bootstrap_round=bootstrap_round,
                collection_iteration=collection_iteration,
                npz_compress_level=npz_compress_level)
        except Exception as exc:
            audits.append(dict(accepted=False, reason='golden_bundle_error',
                error=f'{type(exc).__name__}:{exc}', route_offset_m=offset,
                parent_call_index=condition_index))
            continue
        for row in new_rows:
            row.update(data_contract=DATA_CONTRACT, failure_modes=list(modes),
                       trajectory_map_path=trajectory_map_path(row['slow_messages']))
            row['bootstrap_provenance'].update(data_contract=DATA_CONTRACT,
                failure_modes=list(modes), parent_sample_type=sample_type,
                parent_condition_call_index=condition_index, golden_offset_m=offset,
                route_validation=teacher)
        rows.extend(new_rows)
        audits.append(dict(accepted=bool(new_rows), reason=detail.get('reason'),
            route_offset_m=offset, parent_call_index=condition_index,
            slow_anchor_id=detail.get('slow_anchor_id'), samples=len(new_rows)))
    return rows, audits


def build_episode_rows(*, calls, counters, case, source_path, protocol=P, max_anchors=None,
                       teacher_mode='audit_only', reviews=None, judge_url=None,
                       judge_model=None, timeout=600., anchor_mode='collision-only', **kwargs):
    C = _legacy()
    reviews = reviews or {}
    if teacher_mode == 'physical_oracle':
        if anchor_mode not in ('collision-only', 'failure-pair'):
            raise ValueError(f'unknown physical anchor mode: {anchor_mode}')
        limit = len(calls) if max_anchors is None else max_anchors
        rows, audits, accepted = [], [], 0
        # Every eligible recorded decision owns its own pose, context, and
        # query. Golden future frames are fast windows, not new slow anchors.
        for index, call in enumerate(calls):
            if accepted >= limit:
                break
            if call.get('name') != 'run_action_head':
                continue
            mismatch = (v43_direction_mismatch(call) if anchor_mode == 'failure-pair'
                        else None)
            if mismatch is not None:
                kind, mode, contract = ('direction_mismatch_relabel',
                                        'direction_mismatch', DIRECTION_CONTRACT)
            elif observed_collision(call):
                kind, mode, contract = 'collision_relabel', 'collision', COLLISION_CONTRACT
            else:
                continue
            counters['failure/' + mode] += 1
            audit = dict(scene=case['scene'], episode_id=case['episode_id'],
                         sample_type=kind, trigger_call_index=index,
                         condition_call_index=index, failure_modes=[mode],
                         source_rollout=str(source_path), accepted=False,
                         selection_contract=contract,
                         **({'previous_contract_direction': mismatch[0],
                             'previous_actual_direction': mismatch[1],
                             'previous_actual_endpoint': mismatch[2]}
                            if mismatch is not None else {}))
            if C.call_pose(call) is None:
                audits.append({**audit, 'reason': 'missing_physical_trigger_trace'})
                continue
            counters['candidate_attempts'] += 1
            try:
                extra_cap = min(int(kwargs.get('golden_extra_max', 0)),
                                max(0, limit - accepted - 1))
                samples, detail = build_candidate(calls=calls, condition_index=index,
                    trigger_index=index, sample_type=kind, modes=[mode],
                    case=case, teacher_mode=teacher_mode, reviews={}, judge_url=None,
                    judge_model=None, timeout=timeout, intervening_turns=[],
                    **{**kwargs, 'golden_extra_max': extra_cap})
            except ValueError as exc:
                audits.append({**audit, 'reason': f'invalid_condition:{exc}'})
                continue
            extra_audits = detail.pop('golden_extra_audits', [])
            audits.append({**audit, **detail})
            for extra in extra_audits:
                audits.append(dict(scene=case['scene'], episode_id=case['episode_id'],
                    sample_type='golden_route_relabel', source_rollout=str(source_path),
                    selection_contract=GOLDEN_EXTRA_CONTRACT, **extra))
            if samples:
                ids = list(dict.fromkeys(row['slow_anchor_id'] for row in samples))
                for priority, anchor_id in enumerate(ids, accepted):
                    for row in samples:
                        if row['slow_anchor_id'] == anchor_id:
                            row['bootstrap_provenance']['episode_anchor_priority'] = priority
                rows.extend(samples)
                accepted += len(ids)
                counters[kind + '_anchors_accepted'] += 1
                counters['golden_route_anchors_accepted'] += len(ids) - 1
        counters['anchors_accepted'] += accepted
        return rows, audits
    max_anchors = 2 if max_anchors is None else max_anchors
    rows, audits, accepted, used = [], [], set(), set()
    for trigger_index, trigger in enumerate(calls):
        if len(accepted) >= max_anchors:
            break
        if trigger.get('name') != 'run_action_head':
            continue
        review = reviews.get((case['scene'], case['episode_id'], trigger_index))
        # Reviewed failure annotations are used only after exact binding.
        if review:
            start = C.call_pose(trigger)
            if start is not None:
                validate_review(review, case, trigger, trigger_index, start)
            if review.get('approved_failure_modes') and review.get('execution_sha256') != execution_digest(trigger):
                raise ValueError('review execution binding mismatch')
        modes = failure_modes(trigger, review)
        if not modes:
            continue
        if (C.call_pose(trigger) is None or C.call_pose(trigger, before=False) is None or
                (trigger.get('action_trace') or {}).get('option_endpoint_source') != 'world_pose'):
            audits.append(dict(accepted=False, trigger_call_index=trigger_index,
                               reason='missing_physical_trigger_trace', failure_modes=modes))
            continue
        counters.update({'failure/' + mode: 1 for mode in modes})
        recovery, turns, recovery_error = C.next_recovery_call(calls, trigger_index,
                                                             protocol_version='v6')
        for kind, index in (('execution_relabel', trigger_index), ('recovery', recovery)):
            if len(accepted) >= max_anchors or kind in accepted:
                continue
            audit = dict(scene=case['scene'], episode_id=case['episode_id'], sample_type=kind,
                trigger_call_index=trigger_index, condition_call_index=index,
                failure_modes=modes, source_rollout=str(source_path), accepted=False,
                selection_contract=SELECTION_CONTRACT)
            if index is None or index in used:
                audits.append({**audit, 'reason': recovery_error if index is None else 'duplicate_condition_anchor'})
                continue
            counters['candidate_attempts'] += 1
            samples, detail = build_candidate(calls=calls, condition_index=index,
                trigger_index=trigger_index, sample_type=kind, modes=modes, case=case,
                teacher_mode=teacher_mode, reviews=reviews, judge_url=judge_url,
                judge_model=judge_model, timeout=timeout,
                intervening_turns=turns if kind == 'recovery' else [], **kwargs)
            audits.append({**audit, **detail})
            if samples:
                rows.extend(samples)
                accepted.add(kind)
                used.add(index)
                counters[kind + '_anchors_accepted'] += 1
    counters['anchors_accepted'] += len(accepted)
    return rows, audits
