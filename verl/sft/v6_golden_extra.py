"""Causal v6 slow decisions at later positions on a corrected golden route.

The original online decision remains the first anchor.  Later decisions are
generated only after replaying model-visible golden tool feedback.  No future
route geometry or task goal is placed in the slow request.
"""
from __future__ import annotations

import copy
import hashlib
import io
import math
import random
from pathlib import Path

import numpy as np

from action_head import protocol_v6 as P
from action_head.trajectory_map import CURRENT_VIEW_TEXT, MAP_CONTEXT_TEXT, TrajectoryMapState
from sft import build_fm_nested_bundles as B
from sft.golden_context_beam import request_messages


OPTION_M = 1.6


def sampled_offsets(route, seed_key, max_extra=3):
    """Choose 1..3 reproducible complete-option boundaries with a future."""
    if not 0 <= max_extra <= 3:
        raise ValueError('golden extra max must be in [0,3]')
    length = float(route['length_m'])
    candidates = [round(OPTION_M * step, 6) for step in range(1, int(length / OPTION_M) + 1)
                  if OPTION_M * step < length - 0.1 - 1e-6]
    if not candidates or max_extra == 0:
        return []
    seed = int.from_bytes(hashlib.sha256(str(seed_key).encode()).digest()[:8], 'big')
    rng = random.Random(seed)
    count = min(len(candidates), rng.randint(1, max_extra))
    return sorted(rng.sample(candidates, count))


def route_from_offset(route, offset):
    """Keep the checked route geometry, starting exactly at a slow boundary."""
    from sft.collect_action_head_recovery import pose_at
    start = pose_at(route, offset)
    keep = int(np.searchsorted(route['arc'], offset, side='right'))
    xs = [start[0], *route['xs'][keep:]]
    ys = [start[1], *route['ys'][keep:]]
    yaws = [start[2], *route['yaws'][keep:]]
    if len(xs) < 2:
        raise ValueError('golden route suffix has no future')
    arc = np.concatenate(([0.], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))))
    if arc[-1] < 0.1:
        raise ValueError('golden route suffix is too short')
    return dict(xs=xs, ys=ys, yaws=yaws, arc=arc, length_m=float(arc[-1]),
                smoothing=route['smoothing'], execution_replay=route['execution_replay'])


def canonical_history(slow_messages):
    """Undo the request-only map injection before extending the conversation."""
    messages = copy.deepcopy(slow_messages)
    found = 0
    for message in messages:
        content = message.get('content')
        if isinstance(content, str):
            # Online chat_query stores assistant content as a string, while
            # the v6 golden-context builder consumes canonical part lists.
            content = [{'type': 'text', 'text': content}]
            message['content'] = content
        if not isinstance(content, list):
            continue
        for index in range(len(content) - 2, -1, -1):
            if content[index].get('text') == MAP_CONTEXT_TEXT:
                if (content[index + 1].get('type') not in ('image_url', 'image') or
                        content[index + 2].get('text') != CURRENT_VIEW_TEXT):
                    raise ValueError('invalid archived v6 trajectory map placement')
                del content[index:index + 3]
                found += 1
    if found != 1:
        raise ValueError('expected one archived v6 map')
    return messages


def _world_pose(raw):
    try:
        return (float(raw['world_xy'][0]), float(raw['world_xy'][1]),
                math.radians(float(raw['world_yaw_deg'])))
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def _relative_pose(start, end):
    # Match the renderer's rounded actual_end_pose used by online map updates.
    from action_head.render_client import _option_ego_pose
    return _option_ego_pose(
        (start[0], start[1], math.degrees(start[2])),
        (end[0], end[1], math.degrees(end[2])))


def state_before_call(calls, index, archived_map):
    """Recreate the exact online map from observed calls; fail closed on drift."""
    state = TrajectoryMapState()
    for call in calls[:index]:
        if call.get('name') == 'turn':
            state.update_turn(call.get('result'))
        elif call.get('name') == 'run_action_head' and call.get('ok') is True:
            trace = call.get('action_trace') or {}
            start = _world_pose(trace.get('before_world_pose'))
            prefixes = trace.get('prefixes') or []
            if not prefixes:
                continue
            if start is None:
                raise ValueError('missing prior action-head physical start')
            frames = []
            for prefix in prefixes:
                end = _world_pose(prefix.get('after_world_pose'))
                if end is None:
                    raise ValueError('missing prior chunk physical end')
                frames.append(dict(actual_end_pose=_relative_pose(start, end),
                    blocked=bool(prefix.get('blocked')), collision=bool(prefix.get('collided'))))
            state.update_action_head(frames)
    expected_bytes = Path(archived_map).read_bytes()
    actual = state.render().convert('RGB')
    if Path(archived_map).suffix.lower() in ('.jpg', '.jpeg'):
        stream = io.BytesIO()
        actual.save(stream, format='JPEG', quality=90, optimize=True)
        actual_bytes = stream.getvalue()
    else:
        stream = io.BytesIO()
        actual.save(stream, format='PNG')
        actual_bytes = stream.getvalue()
    if expected_bytes != actual_bytes:
        raise ValueError('reconstructed online trajectory map differs from archived map')
    return state


def golden_feedback(route, start_m, end_m, contract, option_id, scene,
                    render_url, frame_dir, preloaded):
    """Use the same v6 tool-result structure as the open-loop golden builder."""
    from sft.collect_action_head_recovery import pose_at, render_pose
    start = pose_at(route, start_m)
    end = pose_at(route, end_m)
    frames = []
    chunk = 0
    for value in np.arange(start_m + 0.4, end_m + 1e-6, 0.4):
        chunk += 1
        at = min(float(value), end_m)
        pose = pose_at(route, at)
        frames.append(dict(chunk=chunk, actual_end_pose=_relative_pose(start, pose),
            path_m=round(at - start_m - (chunk - 1) * 0.4, 3), blocked=False,
            collision=False, no_progress=False,
            image_path=str(render_pose(render_url, scene, pose,
                frame_dir / f'chunk_{chunk:02d}.jpg', preloaded))))
    if not frames or end_m - (start_m + .4 * len(frames)) > 1e-4:
        chunk += 1
        frames.append(dict(chunk=chunk, actual_end_pose=_relative_pose(start, end),
            path_m=round(end_m - start_m - .4 * (chunk - 1), 3), blocked=False,
            collision=False, no_progress=False,
            image_path=str(render_pose(render_url, scene, end,
                frame_dir / f'chunk_{chunk:02d}.jpg', preloaded))))
    option_pose = _relative_pose(start, end)
    summary = dict(schema=P.TOOL_RESULT_SCHEMA, tool='run_action_head', option_id=option_id,
        ok=True, return_reason='fast_step_budget_reached', completion='unknown',
        chunks=len(frames), path_m=round(end_m - start_m, 3),
        net_m=round(math.hypot(option_pose['x_m'], option_pose['y_m']), 3),
        blocked=False, collision=False, steps_used=max(1, int(round((end_m-start_m)/.1))),
        direction_alignment=P.direction_alignment(contract, option_pose))
    return summary, frames


def generate_decision(*, history, state, instruction, map_path, vllm_url,
                      vllm_model, timeout, query_provider, max_attempts=1):
    """Pick a valid AH candidate from the v6 causal request, then fetch its query."""
    if max_attempts < 1:
        raise ValueError('max_attempts must be positive')
    messages = request_messages(B, instruction, [], history, state, map_path)
    reason = 'no_eligible_action_head'
    for _ in range(max_attempts):
        payload = dict(model=vllm_model, messages=messages, temperature=.7, top_p=.95,
            max_tokens=260, chat_template_kwargs={'enable_thinking': False})
        request_id = B.request_four_queries(payload)
        response = B.post_json(vllm_url.rstrip('/') + '/v1/chat/completions', payload, timeout)
        decision, text, reason = B.parse_tool_call(response, action_head_as_tool_version='v6')
        try:
            clean, tail_count = B.strip_query_tail(text)
        except ValueError as exc:
            decision, reason = None, f'invalid_query_tail:{exc}'
        if decision is not None and tail_count not in (0, 4):
            reason = f'invalid_query_tail_count:{tail_count}'
        if decision is None or decision['name'] != 'run_action_head' or tail_count not in (0, 4):
            (Path(query_provider.cache_root) / (request_id + '.safetensors')).unlink(missing_ok=True)
            continue
        hidden = query_provider(request={'messages': messages},
            response={'query_request_id': request_id}, contract=decision['arguments'])
        completion = P.format_tool_call('run_action_head', decision['arguments'])
        return dict(arguments=decision['arguments'], completion=completion,
                    query_hidden=hidden, request_id=request_id,
                    raw_completion=clean), messages, 'accepted'
    return None, messages, reason
