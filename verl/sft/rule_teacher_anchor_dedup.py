"""Conservative, reversible deduplication of consecutive no-information holds.

The default policy never excludes an anchor on geometry alone. Identical local calls, programs,
all five visible frames, all four complete future labels and unchanged textual
history are required. A repeated run retains its first and last full anchors:
the latter preserves a representative of the longer conversation condition.
This is redundancy evidence, not instruction-quality or terminal-success review.
The opt-in selective policy also removes static actual calls with unchanged RGB
and negligible future motion, and compacts near-equivalent dense runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time

from sft.collect_action_head_recovery import atomic_json, atomic_jsonl, sha256_file
from sft.rule_teacher_progress import measured_poses, execution_progress

SCHEMA = 'rule_teacher_anchor_dedup_v1'
POLICY = 'exact_consecutive_hold_first_and_last_v4'
SELECTIVE_POLICY = 'selective_motion_dense_first_and_last_v2'
SELECTIVE_THRESHOLDS = dict(short_path_m=.1, useful_turn_deg=10.,
    dense_position_m=.1, dense_heading_deg=10., measured_trace_m=.02,
    measured_trace_heading_deg=2., future_step_m=.005, future_step_yaw_deg=2.,
    stationary_future_path_m=.01, stationary_future_turn_deg=5.)
SUPPORTED_POLICIES = {POLICY, SELECTIVE_POLICY, 'selective_motion_dense_first_and_last_v1',
                      'exact_consecutive_hold_first_and_last_v2',
                      'exact_consecutive_hold_first_and_last_v3'}
ART = Path('/data/zerovln/artifacts')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def pixel_digest(path):
    from PIL import Image
    with Image.open(path) as image:
        if image.mode != 'RGB':
            raise ValueError('non_rgb_frame')
        h = hashlib.sha256(str(image.size).encode())
        h.update(image.tobytes())
        return h.hexdigest()


def text_history(request):
    """Preserve every text fact; normalize only documented bookkeeping IDs.

    Archived older images can be omitted by the existing context budget. Their
    omission is not a textual task change. Current/fast observations are checked
    independently via exact decoded RGB hashes, never perceptual similarity.
    """
    result = []
    for message in request['messages']:
        content = message['content']
        parts = [content] if isinstance(content, str) else [
            item['text'] for item in content if item.get('type') == 'text'
            and item.get('text') != '[slow observation omitted]']
        normalized = []
        for value in parts:
            value = re.sub(r'(?<="option_id":")turn-\d+-call-\d+(?=")', 'OPTION', value)
            value = re.sub(r'(?<=\[option=)turn-\d+-call-\d+(?= chunk=)', 'OPTION', value)
            value = re.sub(r'(?<=omitted: option=)turn-\d+-call-\d+(?=, chunk=)', 'OPTION', value)
            value = re.sub(r'\[Tool-call budget: at most \d+ calls remain\.\]', '[Tool-call budget: remaining]', value)
            # These exact collector captions describe the presentation of an
            # image, not an observation fact. Image identity remains checked.
            value = value.replace('This start-relative top-down trace contains actual executed motion only. S is the start, C is the current position, and the arrow is the current heading. Up is the starting forward direction and left is the starting left direction.', '')
            value = value.replace('[Latest actual trajectory map]', '').replace('[Current first-person view]', '')
            value = re.sub(r'\[(?:slow observation|image) omitted: option=OPTION, chunk=[1-4]\]', '', value)
            if value.strip():
                normalized.append(value.strip())
        result.append(dict(role=message['role'], text=normalized))
    return result


def visible_image_history(request):
    """Bind retained earlier image payloads without printing or decoding them."""
    return [[digest(item['image_url']) for item in message['content']
             if item.get('type') == 'image_url']
            if isinstance(message['content'], list) else [] for message in request['messages']]


def omitted_images_only(previous, current):
    """Retained payloads are an ordered subsequence, with no substitutions."""
    old = iter(previous)
    return all(any(candidate == payload for candidate in old) for payload in current)


def repeated_feedback_only(previous, current):
    """A prefix-preserving extension with exactly one repeated call/result pair.

    New turns, stop decisions, instructions, feedback text, corrections, changed
    retained historical images or loss of unique context break the run. Omitted old
    images are allowed, but additions/substitutions in existing history are not.
    """
    a, b = previous['history'], current['history']
    if len(a) < 2:
        return False
    if b[-2:] != a[-2:] or [x['role'] for x in b[-2:]] != ['assistant', 'user']:
        return False
    old_images, new_images = previous['historical_images'], current['historical_images']
    # Current trajectory-map and view payloads must match even when a new
    # repeated pair is appended. Five execution RGB frames do not bind the map.
    if len(old_images) != len(a) or len(new_images) != len(b) or old_images[-2:] != new_images[-2:]:
        return False
    if len(b) == len(a) + 2 and b[:len(a)] == a:
        pairs = zip(old_images, new_images[:len(a)])
    elif b == a:
        pairs = zip(old_images, new_images)
    elif (len(a) == len(b) and len(a) >= 6 and a[:2] == b[:2]
          and a[4:] == b[2:-2] and a[2:4] == a[-2:]):
        # Existing collector's rolling context can remove an already identical
        # repeated pair. Removing a different fact is never accepted here.
        pairs = list(zip(old_images[:2], new_images[:2])) + list(zip(old_images[4:], new_images[2:-2]))
        # An already omitted image is absent from the previous model input;
        # removing its duplicate feedback cannot lose visible information.
        # A retained image must equal the repeated current pair's payload.
        if any(not omitted_images_only(new, old) for old, new in zip(old_images[2:4], new_images[-2:])):
            return False
    else:
        return False
    for old, new in pairs:
        if not omitted_images_only(old, new):
            return False
    return True


def evidence(path, *, exact_hold=True):
    """No tolerances for automatic equivalence: any real difference is retained."""
    path = Path(path).resolve()
    execution = json.loads(path.read_text())
    if execution.get('accepted_for_training') is not True:
        raise ValueError('not_rule_admitted')
    poses, _ = measured_poses(execution)
    state = execution['state']
    windows = state['training_windows']
    if len(windows) != 4 or any(w.get('executed') is not True for w in windows):
        raise ValueError('not_four_executed_windows')
    labels = [w['predicted_delta_xyyaw'] for w in windows]
    if any(len(window) != 16 or any(len(step) != 3 for step in window) for window in labels):
        raise ValueError('invalid_full_future_labels')
    # Only exact measured holds are automatically grouped. Tiny translation or
    # rotation is retained, regardless of being below any diagnostic threshold.
    if exact_hold and any(pose != poses[0] for pose in poses):
        raise ValueError('not_exact_measured_hold')
    request_path = path.parent/'brain_request.json'
    request = json.loads(request_path.read_text())
    frames = [path.parent/f'actual_chunk_{i}.png' for i in range(5)]
    facts = dict(scene=execution['scene'], episode=str(execution['episode']),
        call=execution['original_call'], program=state['program'], pose=poses[0],
        frames=[pixel_digest(p) for p in frames], labels=labels,
        semantic_texts=request['semantic_texts'])
    result = dict(execution_path=str(path), execution_sha256=sha256_file(path),
        request_sha256=sha256_file(request_path), exact_facts_sha256=digest(facts),
        labels_sha256=digest(labels), frame_pixel_sha256=facts['frames'],
        history=text_history(request), historical_images=visible_image_history(request))
    if not exact_hold:
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
               for window in labels for step in window for v in step):
            raise ValueError('nonfinite_future_labels')
        result.update(poses=poses, labels=labels, progress=execution_progress(execution),
            local_context_sha256=digest({k:v for k,v in facts.items() if k not in ('pose', 'labels')}))
        progress = result['progress']
        result['motion_category'] = ('turn' if progress['absolute_yaw_deg'] >= SELECTIVE_THRESHOLDS['useful_turn_deg']
            else 'short_translation' if 1e-6 < progress['actual_path_m'] < SELECTIVE_THRESHOLDS['short_path_m']
            else 'hold' if progress['actual_path_m'] <= 1e-6 else 'translation')
    return result


def _heading_distance(a, b):
    return abs((a - b + 180.) % 360. - 180.)


def _uninformative_hold(proof):
    """An AH call with no actual motion/RGB change and negligible future action.

    This does not remove a zero tail inside a moving four-window anchor.
    """
    progress = proof['progress']
    return (progress['actual_path_m'] <= 1e-6 and progress['absolute_yaw_deg'] <= 1e-3
        and len(set(proof['frame_pixel_sha256'])) == 1
        and all(sum(math.hypot(step[0],step[1]) for step in window) < SELECTIVE_THRESHOLDS['stationary_future_path_m']
                and sum(abs(math.degrees(step[2])) for step in window) < SELECTIVE_THRESHOLDS['stationary_future_turn_deg']
                for window in proof['labels']))


def _dense_equivalent(first, current):
    """Require identical call/program/RGB and no new conversational evidence.

    The full actual trace and all 64 future action labels must also agree within
    tight tolerances. Position alone is never a deduplication criterion.
    """
    t = SELECTIVE_THRESHOLDS
    if (first['local_context_sha256'] != current['local_context_sha256']
            or len(first['poses']) != len(current['poses'])):
        return False
    for index in (0, -1):
        a, b = first['poses'][index], current['poses'][index]
        if (math.hypot(a[0]-b[0], a[1]-b[1]) >= t['dense_position_m']
                or _heading_distance(a[2], b[2]) >= t['dense_heading_deg']):
            return False
    # Genuine turns remain separate even if the start and end RGB happen to match.
    if max(first['progress']['absolute_yaw_deg'], current['progress']['absolute_yaw_deg']) >= t['useful_turn_deg']:
        return False
    if any(math.hypot(a[0]-b[0], a[1]-b[1]) > t['measured_trace_m']
           or _heading_distance(a[2], b[2]) > t['measured_trace_heading_deg']
           for a,b in zip(first['poses'], current['poses'])):
        return False
    return all(math.hypot(a[0]-b[0], a[1]-b[1]) <= t['future_step_m']
               and abs(a[2]-b[2]) <= math.radians(t['future_step_yaw_deg'])
               for wa,wb in zip(first['labels'], current['labels']) for a,b in zip(wa,wb))


def filter_case(paths, *, policy=POLICY):
    """Every input execution gets an explicit decision; never merge across cases."""
    paths = [Path(path).resolve() for path in paths]
    if policy not in (POLICY, SELECTIVE_POLICY):
        raise ValueError('unknown_anchor_selection_policy')
    selective = policy == SELECTIVE_POLICY
    if len({path.parents[2] for path in paths}) > 1:
        raise ValueError('cannot_deduplicate_across_cases')
    decisions, groups, run = [], [], []

    def flush():
        if not run:
            return
        group_id = digest([item[0]['execution_path'] for item in run])
        members = [item[0]['execution_path'] for item in run]
        if len(run) >= 3:
            groups.append(dict(group_id=group_id, members=members,
                representatives=[members[0], members[-1]], count=len(run),
                reason=('near_equivalent_dense_motion_without_new_context' if selective
                        else 'identical_local_evidence_with_only_repeated_feedback_extension'),
                semantic_quality_approval=False))
        for index, (binding, proof) in enumerate(run):
            decision = 'drop_redundant' if len(run) >= 3 and 0 < index < len(run)-1 else 'keep'
            decisions.append(dict(**binding, decision=decision, group_id=group_id,
                group_size=len(run), representative=members[0],
                reason=('redundant_middle_of_dense_run' if selective else 'redundant_middle_of_exact_hold_run')
                    if decision != 'keep' else 'complete_anchor_representative',
                evidence={k:v for k,v in proof.items() if k not in ('history', 'historical_images', 'poses', 'labels')},
                normalized_history_sha256=digest(proof['history'])))
        run.clear()

    for path in paths:
        path = Path(path).resolve()
        binding = dict(execution_path=str(path), execution_sha256=sha256_file(path))
        try:
            proof = evidence(path, exact_hold=not selective)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            flush()
            decisions.append(dict(**binding, decision='keep', reason='not_proven_redundant:'+str(exc)))
            continue
        if selective and _uninformative_hold(proof):
            flush()
            decisions.append(dict(**binding, decision='drop_redundant',
                reason='static_actual_rgb_and_negligible_full_future_motion',
                evidence={k:v for k,v in proof.items() if k not in ('history', 'historical_images', 'poses', 'labels')}))
            continue
        if run:
            equivalent = (_dense_equivalent(run[0][1], proof) if selective else
                          proof['exact_facts_sha256'] == run[0][1]['exact_facts_sha256'])
            equivalent = equivalent and repeated_feedback_only(run[-1][1], proof)
        else:
            equivalent = True
        if not equivalent:
            flush()
        run.append((binding, proof))
    flush()
    return decisions, groups


def checked_decisions(ledger, expected_sha256):
    ledger = Path(ledger).resolve()
    if sha256_file(ledger) != expected_sha256:
        raise ValueError('dedup_ledger_hash_mismatch')
    report = json.loads(ledger.read_text())
    if report.get('schema') != SCHEMA or report.get('policy') not in SUPPORTED_POLICIES or report.get('complete') is not True:
        raise ValueError('invalid_dedup_ledger')
    binding = report['decisions']
    path = Path(binding['path'])
    if sha256_file(path) != binding['sha256']:
        raise ValueError('dedup_decisions_hash_mismatch')
    decisions = {}
    with path.open() as stream:
        for line in stream:
            item = json.loads(line)
            key = item['execution_path']
            if key in decisions or item['decision'] not in ('keep', 'drop_redundant'):
                raise ValueError('invalid_duplicate_dedup_decision')
            decisions[key] = item
    return decisions


def keep_execution(path, expected_execution_sha256, decisions):
    key = str(Path(path).resolve())
    if key not in decisions:
        raise ValueError('dedup_coverage_missing:'+key)
    item = decisions[key]
    if item['execution_sha256'] != expected_execution_sha256 or sha256_file(Path(key)) != expected_execution_sha256:
        raise ValueError('dedup_execution_hash_mismatch:'+key)
    if item['decision'] == 'drop_redundant' and item['reason'] in (
            'static_actual_rgb_and_negligible_full_future_motion', 'redundant_middle_of_dense_run'):
        proof = item['evidence']
        folder = Path(key).parent
        if sha256_file(folder/'brain_request.json') != proof['request_sha256']:
            raise ValueError('dedup_request_evidence_changed:'+key)
        if [pixel_digest(folder/f'actual_chunk_{i}.png') for i in range(5)] != proof['frame_pixel_sha256']:
            raise ValueError('dedup_frame_evidence_changed:'+key)
    return item['decision'] == 'keep'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy', choices=[POLICY, SELECTIVE_POLICY], default=POLICY)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_relative_to(ART) or (output/'dedup-report.json').exists():
        raise ValueError('use a fresh output on the data disk')
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.case_manifest.read_text())
    records, groups, counts, by_source, digest_inputs = [], [], Counter(), {}, hashlib.sha256()
    started = time.time()
    for case in manifest['cases']:
        folder = Path(case['path'])
        for name in ('case.json', 'result.json'):
            if sha256_file(folder/name) != case['sha256'][name]:
                raise ValueError('completed_case_changed:'+str(folder/name))
        # Freeze both membership and bytes; no late additions to the snapshot.
        paths = [folder/item['relative_path'] for item in case['executions']]
        for path, item in zip(paths, case['executions']):
            if sha256_file(path) != item['sha256']:
                raise ValueError('execution_snapshot_changed:'+str(path))
        decisions, local_groups = filter_case(paths, policy=args.policy)
        records.extend(decisions); groups.extend(local_groups)
        local = by_source.setdefault(case['source'], Counter())
        for item in decisions:
            counts[item['decision']] += 1; local[item['decision']] += 1
            digest_inputs.update(item['execution_path'].encode()); digest_inputs.update(item['execution_sha256'].encode())
        counts['cases'] += 1; local['cases'] += 1
        atomic_json(output/'progress.json', dict(cases=counts['cases'], counts=dict(counts),
            updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    decisions_path = output/'decisions.jsonl'
    atomic_jsonl(decisions_path, records)
    atomic_json(output/'groups.json', dict(groups=groups))
    atomic_jsonl(output/'kept-executions.jsonl', [item for item in records if item['decision'] == 'keep'])
    report = dict(schema=SCHEMA, policy=args.policy, complete=True, semantic_acceptance_passed=False,
        case_manifest=dict(path=str(args.case_manifest.resolve()), sha256=sha256_file(args.case_manifest)),
        decisions=dict(path=str(decisions_path), sha256=sha256_file(decisions_path)),
        counts=dict(counts), by_source={k:dict(v) for k,v in by_source.items()}, groups=len(groups),
        input_digest=digest_inputs.hexdigest(), wall_seconds=time.time()-started,
        original_files_modified=False, full_anchor_windows=4,
        selective_thresholds=SELECTIVE_THRESHOLDS if args.policy == SELECTIVE_POLICY else None,
        selection_reasons=dict(Counter(item['reason'] for item in records)),
        motion_categories=dict(Counter(item.get('evidence', {}).get('motion_category', 'unclassified') for item in records)),
        policy_limits=('drop static actual AH calls with unchanged RGB and sub-centimetre/<5deg full futures; compact equivalent dense runs '
            'within a real case to first and last complete anchors; preserve new target, RGB, '
            'feedback, turn or materially different actions; no terminal-success filter'
            if args.policy == SELECTIVE_POLICY else
            'exact holds only; no movement/yaw cutoff, no approximate matching, no terminal-success filter; keep first and last history conditions'))
    atomic_json(output/'dedup-report.json', report)
    print(json.dumps(dict(counts=report['counts'], groups=len(groups), output=str(output))), flush=True)


if __name__ == '__main__':
    main()
