#!/usr/bin/env python3
"""Export reviewed real rule-teacher calls into the existing v6 SFT loader.

Calls/history are preserved. Only image storage is externalized. Privileged
binding and geometry remain audit metadata, never slow-conditioning messages.
Historical records supervise predictions with four-step executed prefixes. New
full-route records verify every planned point against measured execution before
partitioning the complete call into 16-step prediction/four-step execution windows.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image

from action_head import contract as A
from action_head import protocol_v6 as P
from action_head.policy import TrajectoryTransform
from agent_tools import parse_tool_calls
from vln_gym.fm_action import chunk_frame_trajectory, world_to_chunk_pose
from sft.action_head_data_contract import (
    ANCHOR_SAMPLING, RULE_EXECUTION_CONTRACT, RULE_FULL_ROUTE_CONTRACT,
    RULE_SELECTION_CONTRACT, RULE_POLICY_SELECTION_CONTRACT, RULE_UNCLIPPED_NORMALIZATION,
    validate_v6_training_row, generated_instruction_task_id,
    ORIGINAL_TASK_GROUP_SCHEMA, original_task_group,
)
from sft.build_abotn_sft_data import ZYT_STATS, _flip_to_base, normalize_waypoints
from sft.build_fm_nested_bundles import save_npz
from sft.collect_action_head_recovery import (
    atomic_json, atomic_jsonl, externalize_messages, sha256_file,
)
from sft.collect_v6_option_corrections import condition_binding, trajectory_map_path

ARTIFACTS = Path('/data/zerovln/artifacts')


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def exact_pose(first, second):
    return np.array_equal(np.asarray(first, float), np.asarray(second, float))


def existing_instruction_proof(case, execution, case_dir=None):
    """Validate original SAGE wording without manufacturing generated identities."""
    proof = case.get('existing_instruction_provenance')
    if proof is None and case_dir is not None and str(execution['scene']).startswith('sage-'):
        selection_path = Path(case_dir).parent/'selection.json'
        if not selection_path.is_file():
            return None
        selection = json.loads(selection_path.read_text())
        index = int(Path(case_dir).name.removeprefix('case_'))
        row = selection['cases'][index]
        if row['scene'] != execution['scene'] or row['task'] != execution['episode']:
            raise ValueError('sage_original_selection_identity_mismatch')
        source_path = Path(row['source'])
        # The historical collector launch points at the frozen source plan.
        launch_path = next((parent/'launch.json' for parent in Path(case_dir).parents
                            if (parent/'launch.json').is_file()), None)
        if launch_path is None:
            raise ValueError('sage_original_source_plan_missing')
        launch = json.loads(launch_path.read_text())
        plan_path = Path(launch['source_plan'])
        plan = json.loads(plan_path.read_text())
        source_sha = sha256_file(source_path)
        if plan.get('source_sha256', {}).get(str(source_path)) != source_sha:
            raise ValueError('sage_original_historical_source_hash_mismatch')
        data = json.loads(source_path.read_text())
        matches = [(si, pi, ii, sample, record)
            for si, scene in enumerate(data['scenes']) for pi, sample in enumerate(scene['samples'])
            for ii, record in enumerate(sample['instructions'])
            if str(sample['trajectory_id'])+'_'+str(ii) == str(execution['episode'])]
        if len(matches) != 1:
            raise ValueError('sage_original_source_record_ambiguous')
        si, pi, ii, sample, record = matches[0]
        proof = dict(schema='sage_official_existing_instruction_v1', path=str(source_path.resolve()),
            sha256=sha256_file(source_path), scene=execution['scene'], source_task=execution['episode'],
            scene_index=si, sample_index=pi, instruction_index=ii, trajectory_id=str(sample['trajectory_id']),
            instruction_type=record['instruction_type'], record_sha256=digest(record),
            selection_path=str(selection_path.resolve()), selection_sha256=sha256_file(selection_path),
            source_binding_time='collection_source_plan', source_plan_path=str(plan_path.resolve()),
            source_plan_sha256=sha256_file(plan_path), launch_path=str(launch_path.resolve()),
            launch_sha256=sha256_file(launch_path))
    if proof is None:
        return None
    if (proof.get('schema') != 'sage_official_existing_instruction_v1' or
            proof.get('scene') != execution['scene'] or proof.get('source_task') != execution['episode'] or
            sha256_file(Path(proof['path'])) != proof['sha256']):
        raise ValueError('existing_instruction_source_identity_mismatch')
    data = json.loads(Path(proof['path']).read_text())
    for key in ('scene_index', 'sample_index', 'instruction_index'):
        if type(proof.get(key)) is not int or proof[key] < 0:
            raise ValueError('existing_instruction_invalid_index')
    scene = data['scenes'][proof['scene_index']]
    sample = scene['samples'][proof['sample_index']]
    record = sample['instructions'][proof['instruction_index']]
    scene_id = execution['scene'].removeprefix('sage-')
    if (str(scene['scene_id']).split('_')[-1] != scene_id or
            str(sample['trajectory_id']) != str(proof['trajectory_id']) or
            str(sample['trajectory_id'])+'_'+str(proof['instruction_index']) != str(execution['episode']) or
            record['generated_instruction'].strip() != case['instruction'] or
            record.get('instruction_type') != proof['instruction_type'] or digest(record) != proof['record_sha256']):
        raise ValueError('existing_instruction_record_mismatch')
    return dict(proof, start=record['start'], end=record['end'])


def training_task_identity(case, execution, case_dir=None):
    """Validate archived generation evidence before grouping a generated episode."""
    existing = existing_instruction_proof(case, execution, case_dir)
    if existing is not None:
        return execution['episode'], existing
    proof = case.get('instruction_provenance')
    if proof is None:
        return execution['episode'], None
    task_id = generated_instruction_task_id(execution['scene'], execution['episode'],
                                            case['instruction'], proof)
    path = Path(proof['path'])
    if sha256_file(path) != proof['sha256']:
        raise ValueError('generated_instruction_file_hash_mismatch')
    items = json.loads(path.read_text())
    if not isinstance(items, list) or proof['index'] >= len(items):
        raise ValueError('generated_instruction_index_mismatch')
    item = items[proof['index']]
    if (item.get('generated_instruction') != case['instruction'] or
            any(item.get(key) != proof.get(key) for key in ('start', 'end', 'instruction_type')) or
            case.get('generated_task_id', task_id) != task_id):
        raise ValueError('generated_instruction_text_or_identity_mismatch')
    return task_id, dict(schema=proof['schema'], source_task=execution['episode'],
        scene=execution['scene'], start=proof.get('start'), end=proof.get('end'),
        instruction_file_sha256=proof['sha256'], instruction_index=proof['index'],
        instruction_sha256=hashlib.sha256(case['instruction'].encode()).hexdigest())


def original_task_identity(case, execution, source_hint=None, case_dir=None):
    """Resolve only archived native identity or hash-verified SAGE endpoint evidence."""
    existing = existing_instruction_proof(case, execution, case_dir)
    proof = existing or case.get('instruction_provenance') or {}
    source = case.get('backend') or ('sage' if proof.get('schema') in
        ('sage_official_generated_instruction_v1', 'sage_official_existing_instruction_v1') else source_hint)
    if source_hint is not None and source != source_hint:
        raise ValueError('original_task_group_source_mismatch')
    if source not in ('sage', 'r2r', 'rxr'):
        return None
    identity = dict(schema=ORIGINAL_TASK_GROUP_SCHEMA, source_dataset=source,
                    scene=str(execution['scene']))
    if source == 'sage':
        if proof.get('schema') not in ('sage_official_generated_instruction_v1', 'sage_official_existing_instruction_v1'):
            raise ValueError('sage_original_endpoints_unverified_reexport_required')
        training_task_identity(case, execution, case_dir)
        identity.update(start=proof.get('start'), end=proof.get('end'))
    else:
        identity['episode'] = str(execution['episode'])
    original_task_group(identity)  # Validate endpoint/episode types before publishing.
    return identity


def execution_order(path, record):
    value = record.get('option_id') or path.parent.name
    match = re.search(r'(?:^|_)turn-(\d+)-call-(\d+)$', value)
    if match:
        # Timestamp disambiguates a physical retry of the same tool call.
        prefix = path.parent.name.split('_', 1)[0]
        return int(match[1]), int(match[2]), int(prefix) if prefix.isdigit() else 0
    match = re.fullmatch(r'option_(\d+)', path.parent.name)
    if match:
        return int(match[1]), 0, 0
    raise ValueError(f'unrecognized_execution_order:{path}')


class CaseArchiveIndex:
    """Reuse parsed archives within one case; never cache approval or hashes.

    Every lookup observes membership, file identity/size/times and content SHA.
    Content checks cover coarse timestamps and same-length in-place rewrites. Changed
    archives are reread; changes during a read fail closed. Source SHA checks in
    review/export/admission remain independent. A new case evicts the old case.
    The byte cap bounds cached source bytes per archive kind, not Python heap.
    """
    def __init__(self, max_source_bytes=256 * 1024 * 1024):
        self.max_source_bytes = max_source_bytes
        self.case_dir, self.cached = None, {}

    @staticmethod
    def signature(paths):
        result = []
        for path in paths:
            stat = path.stat()
            result.append((str(path), stat.st_dev, stat.st_ino, stat.st_size,
                           stat.st_mtime_ns, stat.st_ctime_ns, sha256_file(path)))
        return tuple(result)

    def records(self, case_dir, pattern):
        case_dir = Path(case_dir).resolve()
        if case_dir != self.case_dir:
            self.case_dir, self.cached = case_dir, {}
        paths = list(case_dir.glob(pattern))
        before = self.signature(paths)
        previous = self.cached.get(pattern)
        if previous is not None and previous[0] == before:
            return previous[1]
        records = [(path, json.loads(path.read_text())) for path in paths]
        after_paths = list(case_dir.glob(pattern))
        if self.signature(after_paths) != before:
            raise ValueError('case_archive_changed_during_read')
        self.cached.pop(pattern, None)
        if sum(item[3] for item in before) <= self.max_source_bytes:
            self.cached[pattern] = (before, records)
        return records

    def matching_decisions(self, case_dir, request, response):
        # Preserve Python/JSON equality, including numeric equality and every
        # ambiguous duplicate. A digest-only lookup could change that behavior.
        return [(path, record) for path, record in self.records(case_dir, 'decision_*.json')
                if record.get('request') == request and record.get('response') == response]

    def preceding_executions(self, execution_path):
        execution_path = Path(execution_path).resolve()
        records = self.records(execution_path.parents[2], 'options/*/execution.json')
        current = next((record for path, record in records if path == execution_path), None)
        if current is None:
            raise FileNotFoundError(execution_path)
        current_order = execution_order(execution_path, current)
        earlier = [(execution_order(path, record), record) for path, record in records
                   if execution_order(path, record) < current_order]
        return [record for _, record in sorted(earlier, key=lambda item: item[0])]


def preceding_executions(execution_path, archive_index=None):
    """Include unreviewed physical history; semantic approval remains a separate gate."""
    if archive_index is not None:
        return archive_index.preceding_executions(execution_path)
    current = json.loads(execution_path.read_text())
    current_order = execution_order(execution_path, current)
    records = []
    for path in execution_path.parent.parent.glob('*/execution.json'):
        record = json.loads(path.read_text())
        key = execution_order(path, record)
        if key < current_order:
            records.append((key, record))
    return [record for _, record in sorted(records, key=lambda item: item[0])]


def execution_contract(execution):
    contract = execution['state'].get('execution_contract')
    if contract is None:
        return RULE_EXECUTION_CONTRACT
    if contract != 'full_target_route_v1':
        raise ValueError('unsupported_execution_contract')
    return RULE_FULL_ROUTE_CONTRACT


def fixed_unclipped_normalization():
    """Reuse the deployment transform with fixed dataset statistics, without clip."""
    low = [ZYT_STATS[key]['q01'] for key in ('dx', 'dy', 'dyaw')]
    high = [ZYT_STATS[key]['q99'] for key in ('dx', 'dy', 'dyaw')]
    transform = TrajectoryTransform.from_quantiles(low, high, flip_y=True, flip_yaw=True)
    return transform, dict(name=RULE_UNCLIPPED_NORMALIZATION, clipping=None,
        stats_source='ZYT_STATS', stats_sha256=digest(ZYT_STATS), q01=low, q99=high,
        scale=list(transform.scale), offset=list(transform.offset),
        checkpoint_convention='x-forward/y-right/yaw-clockwise',
        physical_convention='fixed-chunk-frame x-forward/y-left/yaw-counterclockwise')


def normalize_full_route_actions(physical):
    transform, _ = fixed_unclipped_normalization()
    # Invert the exact metre/radian consumer, including its sign conversion.
    # No per-call length or teacher-private scale enters model conditioning.
    scale, offset = np.asarray(transform.scale), np.asarray(transform.offset)
    target = ((np.asarray(physical, np.float64) - offset) / scale).astype(np.float32)
    if (not np.isfinite(target).all() or not np.allclose(
            target.astype(np.float64) * scale + offset, physical, atol=1e-6, rtol=1e-6)):
        raise ValueError('full_route_normalization_not_lossless')
    return target


def pose_errors(first, second):
    first, second = np.asarray(first, np.float64), np.asarray(second, np.float64)
    xy = np.linalg.norm(first[..., :2] - second[..., :2], axis=-1)
    yaw = np.abs((first[..., 2] - second[..., 2] + 180.) % 360. - 180.)
    return xy, yaw


def assert_poses_close(first, second, reason, *, xy_tolerance=1e-5, yaw_tolerance=1e-3):
    first, second = np.asarray(first, np.float64), np.asarray(second, np.float64)
    if (first.shape != second.shape or first.shape[-1:] != (3,) or
            not np.isfinite(first).all() or not np.isfinite(second).all()):
        raise ValueError(reason)
    xy, yaw = pose_errors(first, second)
    if np.any(xy >= xy_tolerance) or np.any(yaw >= yaw_tolerance):
        raise ValueError(reason)


def world_poses_from_actions(start, actions):
    """Invert the shared fixed-chunk-frame executor, preserving variable distances."""
    cumulative = chunk_frame_trajectory(actions)
    angle = math.radians(start[2])
    c, s = math.cos(angle), math.sin(angle)
    return np.column_stack((start[0] + c*cumulative[:, 0] - s*cumulative[:, 1],
        start[1] + s*cumulative[:, 0] + c*cumulative[:, 1],
        start[2] + np.degrees(cumulative[:, 2])))


def full_route_actions(state, window):
    """16 planned future deltas; endpoint holds remain valid physical zero actions."""
    start = np.asarray(window['world_pose'], np.float64)
    targets = np.asarray(state['route_target_world_poses'], np.float64)
    progress = window['route_progress_start']
    future = targets[np.minimum(np.arange(progress + 1, progress + 17), len(targets) - 1)]
    yaw = np.unwrap(np.radians(np.r_[start[2], future[:, 2]]))
    cumulative = [world_to_chunk_pose(pose[:2], angle, start[:2], yaw[0])
                  for pose, angle in zip(future, yaw[1:])]
    return np.diff(np.vstack([np.zeros(3), cumulative]), axis=0), future


def validate_full_route(state, execution, windows):
    """Prove complete geometry execution before constructing any training window."""
    chunks = state.get('route_chunks')
    prefix_steps = state.get('route_prefix_steps', 4)
    if (prefix_steps not in (4, 16) or type(prefix_steps) is not int or
            state.get('execution_contract') != ('full_target_route_v1' if prefix_steps == 4
                                                else 'full_target_route_v2_fine16')):
        raise ValueError('full_route_invalid_resolution_contract')
    if (state.get('full_route_complete') is not True or
            execution.get('reason') != 'planned_route_complete' or
            type(chunks) is not int or chunks < 4 or chunks != len(windows)):
        raise ValueError('full_route_incomplete')
    planned = np.asarray(state.get('route_target_world_poses'), np.float64)
    measured = np.asarray(state.get('execution_world_poses'), np.float64)
    if planned.shape != (prefix_steps*chunks + 1, 3) or measured.shape != planned.shape:
        raise ValueError('full_route_missing_complete_waypoint_evidence')
    assert_poses_close(planned[0], state['initial_pose'], 'full_route_planned_start_mismatch')
    assert_poses_close(measured[0], state['initial_pose'], 'full_route_measured_start_mismatch')
    assert_poses_close(measured, planned, 'full_route_measured_plan_mismatch',
                       xy_tolerance=.035, yaw_tolerance=2.)
    for key in ('planned_path_m', 'execution_path_m', 'nominal_step_m',
                'goal_error_m', 'goal_yaw_error_deg'):
        value = state.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('full_route_invalid_geometry_metrics')
    measured_length = np.linalg.norm(np.diff(measured[:, :2], axis=0), axis=1).sum()
    if (not math.isclose(state['execution_path_m'], measured_length, abs_tol=1e-5, rel_tol=1e-6)
            or not math.isclose(state['nominal_step_m'], state['planned_path_m']/(prefix_steps*chunks),
                                abs_tol=1e-7, rel_tol=1e-6)):
        raise ValueError('full_route_path_metrics_mismatch')
    endpoint_xy, endpoint_yaw = pose_errors(measured[-1], planned[-1])
    if (endpoint_xy >= .035 or endpoint_yaw >= 2. or
            not math.isclose(state['goal_error_m'], endpoint_xy, abs_tol=1e-5, rel_tol=1e-5) or
            not math.isclose(state['goal_yaw_error_deg'], endpoint_yaw, abs_tol=1e-3, rel_tol=1e-5)):
        raise ValueError('full_route_endpoint_not_reached')
    path = np.asarray(state.get('path'), np.float64)
    if (path.ndim != 2 or path.shape[1:] != (2,) or len(path) < 1 or
            not np.isfinite(path).all() or np.linalg.norm(path[-1] - planned[-1, :2]) >= 1e-5):
        raise ValueError('full_route_planned_endpoint_mismatch')
    for index, window in enumerate(windows):
        start, end = prefix_steps*index, prefix_steps*(index+1)
        if (window.get('route_progress_start') != start or window.get('route_progress_end') != end):
            raise ValueError('full_route_nonconsecutive_progress')
        actual = measured[start:end+1]
        assert_poses_close(window.get('measured_world_poses'), actual,
                           'full_route_window_measured_evidence_mismatch')
        assert_poses_close(window['world_pose'], actual[0], 'full_route_window_start_mismatch')
        assert_poses_close(window['actual_end_pose'], actual[-1], 'full_route_window_end_mismatch')
        physical, future = full_route_actions(state, window)
        assert_poses_close(window.get('predicted_world_poses'), future,
                           'full_route_future_plan_mismatch')
        assert_poses_close(world_poses_from_actions(window['world_pose'],
                           window['predicted_delta_xyyaw']), future,
                           'full_route_prediction_action_reconstruction_mismatch')
        assert_poses_close(world_poses_from_actions(window['world_pose'],
                           window['executed_prefix']), planned[start+1:end+1],
                           'full_route_executed_action_reconstruction_mismatch')
        # Validate the labels independently of the controller's stored deltas.
        assert_poses_close(world_poses_from_actions(window['world_pose'], physical), future,
                           'full_route_label_action_reconstruction_mismatch')


def original_completion(response, call):
    """Validate the original envelope without normalizing or rewriting it."""
    completion = response['text']
    matches = A.TOOL_CALL_RE.findall(completion)
    if len(matches) != 1:
        raise ValueError('original response must contain exactly one tool call')
    # Use the live v6 parser's narrow brace tolerance, while preserving the
    # original text and checking the exact arguments that were executed.
    parsed = parse_tool_calls(completion)
    if (len(parsed) != 1 or parsed[0]['name'] != 'run_action_head'
            or parsed[0]['arguments'] != call):
        raise ValueError('original response/executed call mismatch')
    return completion


def validate_segment(segment, execution, request, *, execution_path, policy_review=None):
    """Quality selection plus hashes of the actual archived evidence."""
    if policy_review is None:
        if segment.get('semantic_conforms_current_scope') is not True:
            raise ValueError('quarantined_or_semantically_unverified')
    else:
        from sft.build_rule_teacher_policy_review import validate_policy_segment
        validate_policy_segment(segment, execution, execution_path=execution_path, review=policy_review)
        if execution_contract(execution) != RULE_FULL_ROUTE_CONTRACT or execution['state'].get('route_chunks') != 4:
            raise ValueError('policy_requires_four_complete_execution_windows')
    if execution.get('accepted_for_training') is not True or execution.get('ok') is not True:
        raise ValueError('execution_not_accepted')
    if execution.get('blocked') or segment.get('collision') or segment.get('blocked'):
        raise ValueError('blocked_or_collided')
    if any(row.get('blocked') or row.get('collided') or row.get('collision')
           for row in execution.get('internal_log', [])):
        raise ValueError('blocked_or_collided_actual_prefix')
    if not segment.get('reason') or (policy_review is None and not segment.get('reviewer')):
        raise ValueError('missing_review_evidence')
    call = execution['original_call']
    if call != segment['raw_call'] or execution['call_sha256'] != segment['raw_call_sha256']:
        raise ValueError('review_call_mismatch')
    if hashlib.sha256(json.dumps(call, sort_keys=True).encode()).hexdigest() != execution['call_sha256']:
        raise ValueError('raw_call_hash_mismatch')
    if sha256_file(Path(segment['request_path'])) != segment['request_file_sha256']:
        raise ValueError('review_request_mismatch')
    state = execution['state']
    full_route = execution_contract(execution) == RULE_FULL_ROUTE_CONTRACT
    if not exact_pose(state['initial_pose'], segment['initial_pose_world_xy_yaw_deg']):
        raise ValueError('review_start_pose_mismatch')
    folder = execution_path.parent
    # A controller can predict a terminal hold and stop before executing it.
    # Keep the contiguous actually executed prefix, not that unused prediction.
    windows = [window for window in state.get('training_windows', [])
               if window.get('executed') is True]
    if (not windows or (not full_route and len(windows) > 4)
            or execution['chunks'] != len(windows)):
        raise ValueError('missing_actual_16_4_window: invalid_window_count')
    previous = state['initial_pose']
    for index, window in enumerate(windows):
        actions = np.asarray(window['predicted_delta_xyyaw'], np.float32)
        prefix = np.asarray(window['executed_prefix'], np.float32)
        if (window.get('executed') is not True or window['chunk_index'] != index or
                actions.shape != (16, 3) or prefix.shape != (4, 3) or
                not np.isfinite(actions).all() or not np.isfinite(prefix).all()):
            raise ValueError('missing_actual_16_4_window')
        if not exact_pose(previous, window['world_pose']):
            raise ValueError('nonconsecutive_actual_fast_states')
        if not np.allclose(actions[:4], prefix, atol=1e-7, rtol=0):
            raise ValueError('teacher_prediction_executed_prefix_mismatch')
        previous = window['actual_end_pose']
        if not (folder/f'actual_chunk_{index}.png').is_file():
            raise ValueError('missing_actual_fast_observation')
    if not exact_pose(previous, segment['final_pose_world_xy_yaw_deg']):
        raise ValueError('review_end_pose_mismatch')
    for side in ('start', 'end'):
        path = Path(segment[f'{side}_png'])
        if sha256_file(path) != segment[f'{side}_png_sha256']:
            raise ValueError('review_boundary_frame_mismatch')
    if state.get('tracking_failure'):
        raise ValueError('tracking_failure')
    if full_route:
        validate_full_route(state, execution, windows)
    return windows


def export(reviews, output, encoder, *, fast_calls_per_anchor=None, visual_mode="preencoded",
           source_height_report=None, source_height_report_sha256=None):
    """Export unchanged labels/conditions with cached or live fast image inputs."""
    if visual_mode not in ("preencoded", "live"):
        raise ValueError("unknown visual mode")
    if visual_mode == "live" and encoder is not None:
        raise ValueError("live export must not instantiate a visual encoder")
    from sft.rule_teacher_source_height_filter import load_source_height_filter
    height_filter = load_source_height_filter(source_height_report, source_height_report_sha256)
    output = Path(output).resolve()
    if fast_calls_per_anchor not in (None, 4):
        raise ValueError('formal slow anchors require exactly four fast calls')
    if not output.is_relative_to(ARTIFACTS):
        raise ValueError('training output must be on the authorized data disk')
    if output.exists():
        raise FileExistsError('use a new frozen output directory')
    output.mkdir(parents=True)
    rows, audits, decisions, seen = [], [], [], set()
    archive_index = CaseArchiveIndex()
    sources = []
    for review_path in map(Path, reviews):
        review = json.loads(review_path.read_text())
        from sft.build_rule_teacher_policy_review import REVIEW_TYPE, validate_policy_review
        is_policy = review.get('review_type') == REVIEW_TYPE
        policy_provenance = validate_policy_review(review) if is_policy else None
        if not is_policy and review.get('schema') != 'rule-teacher-target-direction-quality-review-v1':
            raise ValueError('unsupported quality review schema')
        if not is_policy and review.get('review_type') not in ('assistant_review', 'human_review'):
            raise ValueError('unidentified quality reviewer')
        sources.append(dict(path=str(review_path.resolve()), sha256=sha256_file(review_path)))
        for segment in review['segments']:
            execution_path = Path(segment['execution_path']).resolve()
            audit = dict(execution_path=str(execution_path), review_path=str(review_path),
                         review_type=review['review_type'], accepted=False)
            try:
                if str(execution_path) in seen:
                    raise ValueError('duplicate_execution')
                seen.add(str(execution_path))
                execution = json.loads(execution_path.read_text())
                case = json.loads((execution_path.parents[2]/'case.json').read_text())
                source = original_task_identity(case, execution, case_dir=execution_path.parents[2])
                if source and source['source_dataset'] == 'sage' and height_filter is None:
                    raise ValueError('sage_same_floor_scope_requires_height_audit')
                if height_filter is not None:
                    reason = height_filter.rejection(execution_path, sha256_file(execution_path),
                        source['source_dataset'] if source else None)
                    if reason:
                        raise ValueError(reason)
                request = json.loads(Path(segment['request_path']).read_text())
                windows = validate_segment(segment, execution, request, execution_path=execution_path,
                    policy_review=review if is_policy else None)
                if fast_calls_per_anchor is not None and (
                        execution_contract(execution) != RULE_FULL_ROUTE_CONTRACT or
                        execution['state'].get('route_chunks') != fast_calls_per_anchor):
                    raise ValueError('formal_slow_anchor_requires_exactly_four_complete_fast_calls')
                data_contract = execution_contract(execution)
                full_route = data_contract == RULE_FULL_ROUTE_CONTRACT
                actual_progress = None
                if full_route:
                    from sft.rule_teacher_progress import execution_progress
                    actual_progress = execution_progress(execution,
                        preceding_executions(execution_path, archive_index))
                    if not actual_progress['admissible_geometry']:
                        raise ValueError('actual_progress_rejected:' + actual_progress['reason'])
                response = json.loads((execution_path.parent/'brain_response.json').read_text())
                completion = original_completion(response, execution['original_call'])
                decision = json.loads(Path(segment['decision_path']).read_text())
                if request != decision['request'] or response != decision['response']:
                    raise ValueError('execution_decision_context_mismatch')
                case_dir = execution_path.parent.parent.parent
                case = json.loads((case_dir/'case.json').read_text())
                training_task_id, instruction_identity = training_task_identity(case, execution, case_dir)
                task_identity = original_task_identity(case, execution, case_dir=case_dir)
                case.update(episode_id=training_task_id, scene=execution['scene'])
                slow_messages = externalize_messages(request['messages'], output/'condition_images')
                map_path = trajectory_map_path(slow_messages)
                call = dict(name='run_action_head', arguments=execution['original_call'],
                    turn=dict(slow_messages=slow_messages, assistant=completion))
                binding = condition_binding(case, call, decision['decision_index']-1,
                                            execution['state']['initial_pose'])
                execution_sha = sha256_file(execution_path)
                # The same original condition can have separate measured executions
                # across controller versions. Keep those labels and files distinct.
                artifact_binding = digest(dict(condition_sha256=binding,
                    execution_sha256=execution_sha)) if full_route else binding
                anchor = ('policy-rule:' if is_policy else 'reviewed-rule:') + artifact_binding[:24]
                images = [execution_path.parent/f'actual_chunk_{index}.png'
                          for index in range(len(windows))]
                if visual_mode == 'live':
                    # Immutable RGB copies and labels only; never fake visual tensors.
                    import shutil
                    frozen_images = []
                    for image in images:
                        target_image = output/'fast_images'/(sha256_file(image)+'.png')
                        target_image.parent.mkdir(exist_ok=True)
                        if not target_image.exists():
                            shutil.copyfile(image, target_image)
                        if sha256_file(image) != sha256_file(target_image):
                            raise ValueError('live_image_copy_mismatch')
                        frozen_images.append(target_image)
                    images = frozen_images
                    features = [None] * len(windows)
                else:
                    rgbs = [np.asarray(Image.open(path).convert('RGB')) for path in images]
                    features = encoder.encode_batch(rgbs)
                    features = [value.float().cpu().numpy().astype(np.float32) for value in features]
                    if len(features) != len(windows):
                        raise ValueError('visual_feature_count_mismatch')
                route_sha = digest(dict(windows=windows,
                    route_target_world_poses=execution['state']['route_target_world_poses'],
                    execution_world_poses=execution['state']['execution_world_poses'])) if full_route else digest(windows)
                provenance = dict(approved=True, review_type=review['review_type'],
                    human_verified=segment.get('human_verified', False),
                    scope='current_target_or_basic_direction', executed=True,
                    reviewed_match_branch=segment.get('reviewed_match_branch', 'target_or_direction'),
                    target_instance_verified=segment.get('target_instance_verified', False),
                    prediction_horizon=16, executed_horizon=4,
                    privileged_data_in_condition=False, condition_sha256=binding,
                    execution_sha256=execution_sha,
                    review_sha256=sha256_file(review_path), route_sha256=route_sha,
                    execution_path=str(execution_path), review_path=str(review_path.resolve()),
                    reviewer=segment.get('reviewer'), evidence=segment['reason'],
                    call_sha256=execution['call_sha256'])
                if is_policy:
                    from sft.build_rule_teacher_policy_review import digest as policy_digest
                    provenance.update(policy_provenance, human_verified=False, semantic_reviewed=False,
                        scope='rule_policy_execution', rule_accepted=True,
                        rule_evidence_sha256=policy_digest(segment['rule_evidence']))
                    provenance.pop('reviewer', None)
                    provenance.pop('reviewed_match_branch', None)
                provenance.update(source_reset_episode_id=execution['episode'],
                    case_path=str(case_dir/'case.json'),
                    case_sha256=sha256_file(case_dir/'case.json'))
                if task_identity is not None:
                    provenance['original_task_identity'] = task_identity
                if instruction_identity is not None:
                    provenance['existing_instruction_identity' if instruction_identity.get('schema') ==
                        'sage_official_existing_instruction_v1' else 'generated_instruction_identity'] = instruction_identity
                    provenance['source_reset_episode_id'] = execution['episode']
                if full_route:
                    provenance['actual_progress'] = actual_progress
                    state = execution['state']
                    provenance.update(execution_contract='full_target_route_v1',
                        full_route_complete=True, route_chunks=state['route_chunks'],
                        executed_route_steps=4*state['route_chunks'],
                        label_source='full_planned_route_verified_against_measured_execution')
                    provenance.update({key: state[key] for key in (
                        'planned_path_m', 'execution_path_m', 'nominal_step_m',
                        'goal_error_m', 'goal_yaw_error_deg')})
                anchor_rows = []
                for index, (window, feature, image) in enumerate(zip(windows, features, images)):
                    # The controller uses x-forward/y-left/yaw-counterclockwise.
                    if full_route:
                        physical, _ = full_route_actions(execution['state'], window)
                        target = normalize_full_route_actions(physical)
                    else:
                        physical = np.asarray(window['predicted_delta_xyyaw'], np.float32)
                        target = normalize_waypoints(_flip_to_base(physical, has_yaw=True))
                    mask = np.ones(16, np.bool_)  # Endpoint holds are valid controller labels.
                    sample = output/'samples'/f'{artifact_binding[:24]}_r{index+1}.npz'
                    sample.parent.mkdir(exist_ok=True)
                    if visual_mode == 'live':
                        save_npz(sample, target_actions=target, action_mask=mask)
                    else:
                        save_npz(sample, reference_visual_tokens=features[0],
                                 current_visual_tokens=feature, target_actions=target,
                                 action_mask=mask)
                    row = dict(schema=P.BUNDLE_SCHEMA, **P.protocol_identity(),
                        data_contract=data_contract,
                        sampling_contract=ANCHOR_SAMPLING,
                        decision_filter=RULE_POLICY_SELECTION_CONTRACT if is_policy else RULE_SELECTION_CONTRACT,
                        sample_type='rule_policy_execution' if is_policy else 'reviewed_rule_execution', split='train',
                        sample=str(sample), slow_anchor_id=anchor,
                        source_scene=execution['scene'], source_task_id=training_task_id,
                        source_point_index=None, option_sequence_index=decision['decision_index']-1,
                        fast_window_index=index, option_replan_index=index,
                        instruction=case['instruction'], contract=execution['original_call'],
                        slow_messages=slow_messages, slow_completion=completion,
                        trajectory_map_path=map_path, reference_image=str(images[0]),
                        current_image=str(image), valid_action_steps=16,
                        teacher_execution_provenance=provenance,
                        source_pose_xyyaw=[*window['world_pose'][:2], math.radians(window['world_pose'][2])])
                    if visual_mode == 'live':
                        row.update(visual_input_mode='live_frozen_visual_v1',
                            reference_image_sha256=sha256_file(images[0]),
                            current_image_sha256=sha256_file(image))
                    if full_route:
                        row['action_normalization'] = fixed_unclipped_normalization()[1]
                    if case.get('backend') in ('sage', 'r2r', 'rxr'):
                        row['source_dataset'] = case['backend']
                    if task_identity is not None:
                        row['source_dataset'] = task_identity['source_dataset']
                        row['original_task_group'] = original_task_group(task_identity)
                    validate_v6_training_row(row)
                    anchor_rows.append(row)
                # A failure in a later window must not publish part of the call.
                rows.extend(anchor_rows)
                decisions.append(dict(schema=P.SLOW_DECISION_SCHEMA, **P.protocol_identity(),
                    slow_anchor_id=anchor, split='train', source_scene=execution['scene'],
                    source_task_id=training_task_id, option_sequence_index=decision['decision_index']-1,
                    instruction=case['instruction'], slow_messages=slow_messages,
                    slow_completion=completion, decision_type='run_action_head',
                    arguments=execution['original_call'], data_contract=data_contract))
                audit.update(accepted=True, reason='rule_policy_admission' if is_policy else 'reviewed_current_target_or_basic_direction',
                    review_type=review['review_type'], semantic_reviewed=not is_policy,
                    reviewed_match_branch=provenance.get('reviewed_match_branch'),
                    target_instance_verified=provenance['target_instance_verified'],
                    slow_anchor_id=anchor, windows=len(windows), condition_sha256=binding,
                    data_contract=data_contract)
                print(json.dumps(dict(event='exported_rule_policy_call' if is_policy else 'exported_reviewed_call', anchor=anchor,
                    windows=len(windows), execution=str(execution_path))), flush=True)
            except (ValueError, KeyError, TypeError, OverflowError, FileNotFoundError) as exc:
                audit['reason'] = str(exc)
            audits.append(audit)
    atomic_jsonl(output/'manifest.jsonl', rows)
    atomic_jsonl(output/'slow_decisions.jsonl', decisions)
    atomic_jsonl(output/'selection_audit.jsonl', audits)
    contracts = Counter(row['data_contract'] for row in rows)
    has_full_route = RULE_FULL_ROUTE_CONTRACT in contracts
    report = dict(schema=RULE_FULL_ROUTE_CONTRACT if has_full_route else RULE_EXECUTION_CONTRACT,
        **P.protocol_identity(), data_contracts=dict(contracts),
        sampling_contract=ANCHOR_SAMPLING, review_sources=sources,
        source_height_audit=height_filter.binding if height_filter else None,
        visual_input_mode='live_frozen_visual_v1' if visual_mode == 'live' else 'preencoded',
        persisted_visual_features=visual_mode != 'live',
        admission_candidates=len(audits),
        reviewed_segments=sum(row.get('review_type') in ('assistant_review', 'human_review') for row in audits),
        accepted_by_review_type=dict(Counter(row.get('review_type') for row in audits if row['accepted'])),
        policy_admission_is_individual_semantic_review=False, accepted_calls=len(decisions), windows=len(rows),
        required_fast_calls_per_slow_anchor=fast_calls_per_anchor,
        excluded_by_reason=dict(Counter(row['reason'] for row in audits if not row['accepted'])),
        manifest=str(output/'manifest.jsonl'),
        full_episode_success_filter=False, assistant_review_is_human_verification=False,
        label_source=('full_planned_routes_verified_against_complete_measured_execution'
            if has_full_route and len(contracts) == 1 else 'per_row_versioned_rule_execution'
            if has_full_route else 'recorded_rule_controller_16_step_predictions_with_real_4_step_prefix_execution'),
        action_convention='quantile normalized single-step x-forward/y-right/yaw-clockwise',
        endpoint_holds='valid_zero_physical_actions_not_padding',
        condition_source='original_cropped_brain_request_and_complete_original_call')
    atomic_json(output/'provenance.json', report)
    return report


def verify_loader(manifest):
    from sft.action_head_joint_dataset import ActionHeadAnchorDataset, load_anchor_splits
    from sft.joint_fm_head_sft import joint_collate
    groups = load_anchor_splits(str(manifest), lambda row, index: validate_v6_training_row(row))
    dataset = ActionHeadAnchorDataset(str(manifest), groups=groups['train'])
    shapes, windows, image_counts = Counter(), 0, []
    for index in range(len(dataset)):
        features = dataset[index]
        batch = joint_collate(features)
        if batch['target_actions'].shape[1:] != (16, 3):
            raise ValueError('actual loader action shape mismatch')
        live = 'reference_image' in batch
        if not batch['action_mask'].all() or (not live and not batch['reference_visual_mask'].all()):
            raise ValueError('actual loader mask mismatch')
        shapes['live_rgb_paths' if live else str(tuple(batch['current_visual_tokens'].shape))] += 1
        windows += len(features)
        image_counts.append(sum(part.get('type') == 'image_url'
            for message in features[0]['slow_messages']
            for part in (message.get('content') if isinstance(message.get('content'), list) else [])))
    return dict(ok=True, loader='ActionHeadAnchorDataset/JointBundleDataset/joint_collate',
        anchors=len(dataset), windows=windows, visual_shapes=dict(shapes),
        max_condition_images=max(image_counts), max_rgb_images=max(image_counts)-1,
        branch_selection=dataset.branch_selection, action_shape=[16, 3])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-height-report', type=Path)
    parser.add_argument('--source-height-report-sha256')
    parser.add_argument('--backbone', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--visual-mode', choices=['preencoded', 'live'], default='preencoded')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16', 'float16'], default='float32')
    parser.add_argument('--fast-calls-per-anchor', type=int, choices=[4], default=4,
                        help='formal export requires four complete fast calls per slow anchor')
    args = parser.parse_args()
    import torch
    from action_head.policy import Qwen35VisualTokenEncoder
    torch.set_num_threads(args.threads)
    encoder = (None if args.visual_mode == 'live' else
        Qwen35VisualTokenEncoder.from_pretrained(args.backbone,
            device=args.device, dtype=getattr(torch, args.dtype)))
    report = export(args.review, args.output, encoder,
                    fast_calls_per_anchor=args.fast_calls_per_anchor, visual_mode=args.visual_mode,
                    source_height_report=args.source_height_report,
                    source_height_report_sha256=args.source_height_report_sha256)
    if not report['accepted_calls']:
        raise ValueError('no reviewed actual calls exported; inspect selection audit')
    verified = verify_loader(args.output/'manifest.jsonl')
    verified.update(backbone=str(args.backbone.resolve()), device=args.device, dtype=args.dtype,
                    weights='not_loaded_live_training' if args.visual_mode == 'live' else 'actual_frozen_visual_tower',
                    visual_input_mode=report['visual_input_mode'],
                    persisted_visual_features=report['persisted_visual_features'], zero_placeholder_features=False)
    atomic_json(args.output/'loader_verification.json', verified)
    print(json.dumps(dict(report=report, loader_verification=verified)), flush=True)


if __name__ == '__main__':
    main()
