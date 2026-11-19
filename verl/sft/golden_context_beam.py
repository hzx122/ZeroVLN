"""Fixed-anchor v6 context beam; no direction filtering.

Only histories branch. Maps use completed model-visible execution feedback,
never future waypoints. Materialization reuses build_fm_nested_bundles.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from action_head import protocol_v6 as P
from action_head import contract as A
from action_head.trajectory_map import MAP_CONTEXT_TEXT, TrajectoryMapState


def select_branches(candidates, width, seed):
    """Seeded parent-stratified random selection; dedup only within a parent."""
    rng = random.Random(seed)
    pools, seen = defaultdict(list), set()
    for candidate in candidates:
        key = (candidate['parent_key'], candidate['normalized_completion'])
        if key in seen:
            continue
        seen.add(key)
        pools[candidate['parent_key']].append(candidate)
    for pool in pools.values():
        rng.shuffle(pool)
    selected = []
    while pools and len(selected) < width:
        parents = list(pools)
        rng.shuffle(parents)
        for parent in parents:
            selected.append(pools[parent].pop())
            if not pools[parent]:
                del pools[parent]
            if len(selected) == width:
                break
    return selected


def stable_seed(seed, anchor_id):
    return int.from_bytes(hashlib.sha256(
        f'{seed}:{anchor_id}'.encode()).digest()[:8], 'big')


def allocate_requests(parents, budget, seed):
    """Spend one fixed anchor budget, balancing existing parent contexts."""
    if not parents or budget < len(parents):
        raise ValueError('sampling budget must cover every live parent')
    base, remainder = divmod(budget, len(parents))
    order = list(parents)
    random.Random(seed).shuffle(order)
    extra = set(order[:remainder])
    return {parent: base + int(parent in extra) for parent in parents}


def sampled_tool_name(text):
    """Read the same tool envelope as the validator, including rejected calls."""
    match = A.TOOL_CALL_RE.search(text)
    if match is None:
        return None
    try:
        call = json.loads(A.strip_code_fence(match.group(1)))
    except (ValueError, TypeError):
        return None
    return call.get('name') if isinstance(call, dict) and isinstance(call.get('name'), str) else None


def summarize_attempts(attempts):
    tools = ('run_action_head', 'turn', 'stop')
    counts = dict(requests=len(attempts), **{
        name: sum(a.get('tool_name') == name for a in attempts) for name in tools})
    counts.update(
        other_tool=sum(a.get('tool_name') is not None and a['tool_name'] not in tools
                       for a in attempts),
        unparsed=sum(a.get('tool_name') is None and a.get('hard_reason') != 'request_error'
                     for a in attempts),
        request_error=sum(a.get('hard_reason') == 'request_error' for a in attempts),
        stop_valid=sum(a.get('hard_reason') == 'terminal_stop' for a in attempts),
        stop_premature=sum(a.get('hard_reason') == 'premature_stop' for a in attempts),
        stop_invalid=sum(a.get('tool_name') == 'stop' and a.get('hard_reason') not in
                         ('terminal_stop', 'premature_stop') for a in attempts),
        eligible_ah=sum(bool(a.get('eligible_ah')) for a in attempts),
        selected=sum(bool(a.get('accepted')) for a in attempts))
    return counts


def state_from_dict(value):
    state = TrajectoryMapState(**copy.deepcopy(value))
    for name in ('points', 'blocked_points', 'collision_points'):
        setattr(state, name, [tuple(point) for point in getattr(state, name)])
    return state


def request_messages(legacy, instruction, images, history, state, map_path):
    canonical = legacy.build_messages(
        instruction, images, history=history, action_head_as_tool_version='v6')
    effective = P.inject_trajectory_map(canonical, state)
    # Persist the exact online renderer output for both generation and SFT.
    for message in effective:
        for part in message.get('content', []):
            if isinstance(part, dict) and part.get('type') == 'image':
                part['image'].save(map_path)
                part.clear()
                part.update(type='image_url', image_url={'url': 'file://' + str(map_path)})
    validate_context(effective, str(map_path))
    return effective


def validate_context(messages, map_path):
    images, markers = [], 0
    for message in messages:
        content = message.get('content', [])
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get('type') in ('image_url', 'image'):
                images.append(part)
            if part.get('text') == MAP_CONTEXT_TEXT:
                markers += 1
    urls = [part.get('image_url', {}).get('url') for part in images]
    if markers != 1 or not 2 <= len(images) <= 12:
        raise ValueError('v6 requires 8 slow + <=3 latest fast + exactly one map')
    if urls.count('file://' + map_path) != 1 or urls[-2] != 'file://' + map_path:
        raise ValueError('v6 map must occur once immediately before current RGB')


def expand_layer(*, legacy, args, by_task, option_idx, frame_dir, output,
                 histories, last_option_end, beams, map_states,
                 broken_tasks, terminated_tasks):
    """Stream each task's AH branches, audits, decisions and next frontier.

    Selected caches are consumed by the existing provider; other caches are
    deleted. Caller commits the frontier only after materializing all records.
    """
    active, jobs = [], []
    map_dir = output / 'trajectory_maps'
    map_dir.mkdir(parents=True, exist_ok=True)
    for task, anchors in by_task.items():
        if option_idx >= len(anchors) or task in broken_tasks or task in terminated_tasks:
            continue
        golden_id, items = anchors[option_idx]
        row = items[0]['slow_anchor_row']
        images = legacy.resolve_images(row, frame_dir)
        instruction = legacy.instruction_from_record(row)
        state = state_from_dict(map_states.get(task, asdict(TrajectoryMapState())))
        path = map_dir / (hashlib.sha256(golden_id.encode()).hexdigest()[:24] + '.png')
        parents = beams.get(task, [task + ('root',)])
        allocation = allocate_requests(parents, args.beam_width * args.pass_k,
                                       stable_seed(args.beam_seed, golden_id + ':allocation'))
        task_jobs = []
        for parent in parents:
            history = histories.get(parent)
            if history is None and option_idx != 0:
                raise ValueError(f'missing beam history at {golden_id}: {parent}')
            if history is not None and Path(last_option_end[parent]).resolve() != Path(images[-1]).resolve():
                raise ValueError(f'v6 causal chain discontinuity: {golden_id}/{parent}')
            messages = request_messages(legacy, instruction, images, history, state, path)
            count = allocation[parent]
            for k in range(count):
                payload = {'model': args.model, 'messages': messages,
                           'temperature': args.temperature, 'top_p': .95,
                           'max_tokens': 256 + legacy.DEFAULT_QUERY_COUNT,
                           'chat_template_kwargs': {'enable_thinking': False}}
                rid = legacy.request_four_queries(payload)
                job = (parent, k + 1, rid, payload, messages, history)
                task_jobs.append(job)
                jobs.append(job)
        active.append((task, golden_id, items, instruction, images, path, task_jobs))

    def call(job):
        try:
            response = legacy.post_json(args.vllm_url.rstrip('/') + '/v1/chat/completions',
                                        job[3], args.timeout)
            return job, response, None
        except Exception as exc:
            return job, None, f'{type(exc).__name__}: {exc}'

    if not jobs:
        return
    with ThreadPoolExecutor(max_workers=min(len(jobs), args.beam_width * args.pass_k * 8)) as pool:
        results = iter(pool.map(call, jobs))
        for task, golden_id, items, instruction, images, map_path, task_jobs in active:
            candidates, attempts, stop_count = [], [], 0
            records, audits, terminals, next_beams, next_maps = [], [], [], {}, {}
            for _ in task_jobs:
                (parent, rank, rid, payload, messages, history), response, error = next(results)
                attempt = {'parent_branch_id': parent[-1], 'rank': rank,
                           'request_id': rid, 'accepted': False}
                attempts.append(attempt)
                if error:
                    attempt.update(hard_reason='request_error', error=error)
                    continue
                decision, text, reason = legacy.parse_tool_call(response, action_head_as_tool_version='v6')
                attempt.update(decision=decision, text=text, hard_reason=reason,
                               tool_name=sampled_tool_name(text))
                if decision is None:
                    continue
                try:
                    clean, query_tail_count = legacy.strip_query_tail(text)
                except ValueError as exc:
                    attempt.update(hard_reason=f'invalid_query_tail:{exc}')
                    continue
                completion = P.format_tool_call(decision['name'], decision['arguments'])
                if decision['name'] == 'stop':
                    valid = legacy.correct_terminal_stop(items[0], args.stop_success_radius_m)
                    attempt.update(hard_reason='terminal_stop' if valid else 'premature_stop', terminal=True)
                    if valid:
                        stop_count += 1
                        terminal = legacy.slow_decision_row(
                            anchor_id=golden_id + ':stop:' + rid, item=items[0],
                            instruction=instruction, messages=messages, completion=completion,
                            decision=decision, pass_rank=rank, protocol=P,
                            decision_filter='golden_success_radius')
                        terminal.update(golden_anchor_id=golden_id, parent_branch_id=parent[-1])
                        terminals.append(terminal)
                    continue
                # No direction_alignment gate and no semantic scorer.
                attempt['eligible_ah'] = True
                child_id = hashlib.sha256((str(parent) + completion).encode()).hexdigest()[:24]
                candidates.append(dict(tk=task + (child_id,), task=task,
                    anchor_id=golden_id + ':branch:' + child_id,
                    golden_anchor_id=golden_id, branch_id=child_id,
                    parent_branch_id=parent[-1], parent_key=parent,
                    items=items, instruction=instruction, anchor_images=images,
                    anchor_row=items[0]['slow_anchor_row'], history=history,
                    messages=messages, contract=decision['arguments'],
                    clean_text=clean, query_tail_count=query_tail_count,
                    normalized_completion=completion, rank=rank, attempts=[],
                    response=response, request_id=rid,
                    trajectory_map_path=str(map_path), decision=decision))
            selected = select_branches(candidates, args.beam_width,
                                       stable_seed(args.beam_seed, golden_id))
            selected_ids = {row['request_id'] for row in selected}
            for attempt in attempts:
                attempt['accepted'] = attempt['request_id'] in selected_ids
                if not attempt['accepted']:
                    (Path(args.cache_root) / (attempt['request_id'] + '.safetensors')).unlink(missing_ok=True)
            audits.append(dict(slow_anchor_id=golden_id, golden_anchor_id=golden_id,
                               option_sequence_index=option_idx,
                               source_scene=task[0], source_task_id=task[1],
                               counts=summarize_attempts(attempts),
                               parent_counts={parent_id: summarize_attempts([
                                   a for a in attempts if a['parent_branch_id'] == parent_id])
                                   for parent_id in dict.fromkeys(a['parent_branch_id'] for a in attempts)},
                               accepted=bool(selected), selected_count=len(selected),
                               terminal=bool(stop_count), attempts=attempts))
            if not selected:
                (terminated_tasks if stop_count else broken_tasks).add(task)
                yield records, audits, terminals, next_beams, next_maps
                continue
            for record in selected:
                record['branch_count'] = len(selected)
            records.extend(selected)
            next_beams[task] = [row['tk'] for row in selected]
            _, frames = legacy.golden_tool_result(golden_id, items, frame_dir,
                                                  protocol=P, contract=selected[0]['contract'])
            state = state_from_dict(map_states.get(task, asdict(TrajectoryMapState())))
            state.update_action_head(frames)
            next_maps[task] = asdict(state)
            yield records, audits, terminals, next_beams, next_maps
