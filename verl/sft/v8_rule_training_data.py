"""Complete v8 fast segments, independent of a future 4/8/16-point experiment.

Reuse the physical route verifier, action normalization and live-image dataset.
Never relabel append-only histories as v6 or impose its image-retention policy.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np

from action_head import protocol_v8 as P
from action_head.trajectory_map import MAP_CONTEXT_TEXT
from sft import export_rule_teacher_training as E
from sft.collect_action_head_recovery import externalize_messages
from sft.rule_teacher_progress import execution_progress

SCHEMA = 'v8_rule_teacher_complete_fast_segments_v1'
MIN_PATH_M = .1
BOUNDARY_TOLERANCE_M = 1e-6


class CompletedCaseArchive(E.CaseArchiveIndex):
    """Reuse archive matching after the case's result hash has been snapshotted.

    The live/archive index intentionally rehashes on each lookup. Ended cases
    have no writer: read each decision once, then recheck membership and all
    content hashes before publishing the entire case. Never use on a live case.
    """
    def __init__(self):
        super().__init__()
        self.completed_records = {}
        self.final_bindings = {}

    @staticmethod
    def signature(paths):
        # EFC can return different ctimes for the same immutable inode/bytes.
        # Ctime describes metadata, not content: retain path, device, inode,
        # size, mtime and the complete SHA instead. Live archive behavior stays
        # in the unchanged base class; this index is only for ended cases.
        return tuple(row[:5] + row[6:] for row in
                     E.CaseArchiveIndex.signature(sorted(paths,key=str)))

    def records(self, case_dir, pattern):
        key = (str(case_dir), pattern)
        if key not in self.completed_records:
            result = super().records(case_dir, pattern)
            self.completed_records[key] = result
            self.final_bindings[key] = self.signature([p for p, _ in result])
        return self.completed_records[key]

    def verify_unchanged(self):
        for (case_dir, pattern), expected in self.final_bindings.items():
            actual=self.signature(list(Path(case_dir).glob(pattern)))
            if actual != expected:
                before={row[0]:row for row in expected};after={row[0]:row for row in actual}
                changed=[dict(path=key,before=before.get(key),after=after.get(key))
                    for key in sorted(set(before)|set(after)) if before.get(key)!=after.get(key)]
                raise ValueError('v8_completed_archive_changed_before_publish:'+json.dumps(
                    dict(case_dir=case_dir,pattern=pattern,changes=changed[:3],changed_count=len(changed))))


def read_bytes(path):
    from dlc_control_io_retry import retry_estale
    return retry_estale(Path(path).read_bytes, str(path))


def checked_json(path, expected=None):
    raw = read_bytes(path)
    sha = hashlib.sha256(raw).hexdigest()
    if expected is not None and sha != expected:
        raise ValueError('v8_source_hash_changed:'+str(path))
    return json.loads(raw), sha


def keep_length(execution):
    progress = execution_progress(execution)
    length = progress.get('actual_path_m')
    if length is None:
        raise ValueError('v8_missing_measured_length')
    # This is a whole-anchor rule, with no rotation exception or per-window mask.
    return length + BOUNDARY_TOLERANCE_M >= MIN_PATH_M, progress


def validate_execution(value):
    state = value['state']
    windows = state.get('training_windows', [])
    if (value.get('accepted_for_training') is not True or value.get('ok') is not True
            or value.get('chunks') != 4 or len(windows) != 4
            or state.get('route_prefix_steps') != 16
            or state.get('execution_contract') != 'full_target_route_v2_fine16'
            or any(w.get('executed') is not True for w in windows)):
        raise ValueError('v8_requires_four_complete_fine16_windows')
    if value.get('blocked') or any(
            r.get('blocked') or r.get('collided') or r.get('collision')
            for r in value.get('internal_log', [])):
        raise ValueError('v8_blocked_or_collided')
    E.validate_full_route(state, value, windows)
    for i, w in enumerate(windows):
        if (w.get('chunk_index') != i or np.asarray(w['predicted_delta_xyyaw']).shape != (16, 3)
                or np.asarray(w['executed_prefix']).shape != (16, 3)):
            raise ValueError('v8_window_resolution_mismatch')
    return windows


def image_copy(path, images):
    raw = read_bytes(path)
    sha = hashlib.sha256(raw).hexdigest()
    target = images/(sha+Path(path).suffix)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    if hashlib.sha256(read_bytes(target)).hexdigest() != sha:
        raise ValueError('v8_fast_image_copy_mismatch')
    return str(target), sha


def validate_messages(messages):
    text = messages[0]['content']
    text = text if isinstance(text, str) else '\n'.join(p.get('text', '') for p in text)
    if text != P.SYSTEM_PROMPT:
        raise ValueError('v8_export_prompt_mismatch')
    maps = []
    for message in messages:
        parts = message.get('content')
        if not isinstance(parts, list):
            continue
        for i, part in enumerate(parts):
            if part.get('text') == MAP_CONTEXT_TEXT:
                following = parts[i+1:i+2]
                url = (following[0].get('image_url') or {}).get('url', '') if following else ''
                if not url.startswith('file://'):
                    raise ValueError('v8_map_not_externalized')
                maps.append(url[7:])
    if not maps:
        raise ValueError('v8_missing_trajectory_map')
    # All historical snapshots stay in the condition. Only metadata names the last.
    return maps[-1]


def coarse_route_labels(row):
    """v6-style window label rebuilt from the published v8 route (user 2026-10-20).

    v8 executes 16 fine points per window (64 per route). The coarse route is
    every 4th fine pose (17 poses). Window w predicts coarse points 4w+1..4w+16
    from its start pose, holding the endpoint past the route end, exactly like
    the v6 full-target-route labels: 4 executed points plus the remaining route.
    """
    trajectory, _ = checked_json(row['trajectory_path'], row['trajectory_sha256'])
    fine = np.asarray(trajectory['planned_world_poses'], np.float64)
    coarse = np.asarray(trajectory['coarse_world_poses'], np.float64)
    window = trajectory['windows'][row['fast_window_index']]
    if (fine.shape != (65, 3) or coarse.shape != (17, 3) or window['start'] % 4
            or not np.allclose(fine[::4, :2], coarse[:, :2], atol=1e-6)):
        raise ValueError('v8_coarse_route_unavailable')
    physical, _ = E.full_route_actions(dict(route_target_world_poses=coarse.tolist()),
        dict(world_pose=window['world_pose'], route_progress_start=window['start'] // 4))
    return E.normalize_full_route_actions(physical).astype(np.float32), np.ones(16, np.bool_)


def executed_points_mask():
    """exec4 trains only the points the evaluator actually executes."""
    mask = np.zeros(16, np.bool_)
    mask[:4] = True
    return mask


def load_condition(row):
    condition, _ = checked_json(row['slow_condition_path'], row['slow_condition_sha256'])
    if condition['schema'] != SCHEMA or condition['slow_anchor_id'] != row['slow_anchor_id']:
        raise ValueError('v8_condition_anchor_mismatch')
    return dict(row, slow_messages=condition['messages'], slow_completion=condition['completion'])


def original_completion(response, call):
    """Validate the same visible text as the collector; retain the raw prefix."""
    from agent_tools import split_thinking
    _, visible = split_thinking(response['text'])
    E.original_completion(dict(text=visible), call)
    return response['text']


def binding_call_matches(binding_call, executed_call):
    """The grounding policy strips whitespace; preserve raw calls in conditions."""
    from action_head.contract import validate_action_head_arguments
    normalized = [validate_action_head_arguments(call, max_plan_stages=None)[0]
                  for call in (binding_call, executed_call)]
    return normalized[0] is not None and normalized[0] == normalized[1]


def validate_training_row(row):
    P.assert_protocol_identity(row)
    if (row.get('data_contract') != SCHEMA or row.get('sampling_contract') != 'independent_anchors_v1'
            or row.get('valid_action_steps') != 16 or row.get('stored_trajectory_points') != 16
            or row.get('segment_count') != 4 or not 0 <= row.get('fast_window_index', -1) < 4):
        raise ValueError('v8_training_row_contract_mismatch')
    if row.get('visual_input_mode') != 'live_frozen_visual_v1':
        raise ValueError('v8_export_requires_live_visual_labels')
    for key in ('sample', 'trajectory_path', 'slow_condition_path', 'reference_image', 'current_image'):
        if not Path(row[key]).is_absolute():
            raise ValueError('v8_training_path_not_absolute:'+key)
    if row['action_normalization'] != E.fixed_unclipped_normalization()[1]:
        raise ValueError('v8_action_normalization_mismatch')
    if row['teacher_execution_provenance'].get('individual_semantic_review') is not False:
        raise ValueError('v8_export_must_not_invent_semantic_review')


def export_anchor(path, execution, case, output, *, source, archive, source_proof):
    """Build all four rows before publishing any of them to a case manifest."""
    from rule_teacher_training_handoff import save
    windows = validate_execution(execution)
    request, request_sha = checked_json(path.parent/'brain_request.json')
    response, response_sha = checked_json(path.parent/'brain_response.json')
    completion = original_completion(response, execution['original_call'])
    matches = archive.matching_decisions(path.parents[2], request, response)
    if len(matches) != 1:
        raise ValueError('v8_missing_or_ambiguous_original_decision')
    decision_path, decision = matches[0]
    candidates = path.parents[2]/'candidates'/f"decision_{decision['decision_index']:03d}"
    selection, selection_sha = checked_json(candidates/'selection.json')
    index = selection.get('selected_index')
    if type(index) is not int or not 0 <= index < 16 or selection.get('all_rejected') is not False:
        raise ValueError('v8_invalid_selected_candidate')
    candidate, candidate_sha = checked_json(candidates/f'candidate_{index:02d}.json')
    binding, binding_sha = checked_json(candidates/f'candidate_{index:02d}_binding/binding.json')
    accepted, accepted_sha = checked_json(candidates/f'candidate_{index:02d}_binding/accepted.json')
    if (candidate.get('selected') is not True or candidate.get('status') != 'selected'
            or candidate.get('request') != request or candidate.get('original_text') != completion
            or len(candidate.get('calls', [])) != 1
            or candidate['calls'][0].get('name') != 'run_action_head'
            or candidate['calls'][0].get('arguments') != execution['original_call']
            or not binding_call_matches(binding.get('original_call'), execution['original_call'])
            or str(binding.get('scene')) != str(execution['scene'])
            or str(binding.get('episode')) != str(execution['episode'])
            or binding.get('pose') != execution['state']['initial_pose']
            or accepted.get('execution_contract') != 'full_target_route_v2_fine16'
            or accepted.get('route_chunks') != 4
            or accepted.get('route_target_world_poses') != execution['state']['route_target_world_poses']):
        raise ValueError('v8_selected_candidate_execution_binding_mismatch')
    execution_sha = hashlib.sha256(read_bytes(path)).hexdigest()
    anchor = 'v8-rule:'+E.digest(dict(source=source, scene=execution['scene'],
        task=str(execution['episode']), execution_sha256=execution_sha,
        request_sha256=request_sha, response_sha256=response_sha))[:32]
    adir = output/'anchors'/anchor.split(':')[1]
    adir.mkdir(parents=True, exist_ok=True)
    messages = externalize_messages(request['messages'], output/'condition_images')
    map_path = validate_messages(messages)
    condition = adir/'condition.json'
    save(condition, dict(schema=SCHEMA, slow_anchor_id=anchor, messages=messages, completion=completion,
        original_request_sha256=request_sha, original_response_sha256=response_sha))
    condition_sha = hashlib.sha256(read_bytes(condition)).hexdigest()
    state = execution['state']
    trajectory = adir/'trajectory.json'
    save(trajectory, dict(schema=SCHEMA, slow_anchor_id=anchor,
        convention='world xy metres; yaw degrees', four_segments=True,
        future_prediction_points_are_training_configurable=[4, 8, 16],
        full_planner_path=state['path'], planned_world_poses=state['route_target_world_poses'],
        measured_world_poses=state['execution_world_poses'],
        coarse_world_poses=state.get('coarse_route_target_world_poses'),
        windows=[dict(index=i, start=w['route_progress_start'], end=w['route_progress_end'],
                      world_pose=w['world_pose'], actual_end_pose=w['actual_end_pose'])
                 for i, w in enumerate(windows)]))
    trajectory_sha = hashlib.sha256(read_bytes(trajectory)).hexdigest()
    images = [image_copy(path.parent/f'actual_chunk_{i}.png', output/'fast_images') for i in range(5)]
    provenance = dict(execution_path=str(path), execution_sha256=execution_sha,
        decision_path=str(decision_path), request_sha256=request_sha, response_sha256=response_sha,
        selection_sha256=selection_sha, candidate_sha256=candidate_sha, binding_sha256=binding_sha,
        accepted_sha256=accepted_sha, individual_semantic_review=False,
        admission='recorded_rule_acceptance_and_complete_measured_execution',
        collection_execution_points_per_window=16, source_proof=source_proof)
    rows = []
    for i, window in enumerate(windows):
        physical, future = E.full_route_actions(state, window)
        sample = adir/f'window_{i}.npz'
        E.save_npz(sample, target_actions=E.normalize_full_route_actions(physical),
                   action_mask=np.ones(16, np.bool_))
        row = dict(schema=P.BUNDLE_SCHEMA, **P.protocol_identity(), data_contract=SCHEMA,
            sampling_contract='independent_anchors_v1', split='train',
            slow_anchor_id=anchor, sample=str(sample), source_dataset=source,
            source_scene=execution['scene'], source_task_id=str(execution['episode']),
            instruction=case['instruction'], contract=execution['original_call'],
            option_sequence_index=decision['decision_index']-1, fast_window_index=i,
            option_replan_index=i, segment_count=4, stored_trajectory_points=16,
            valid_action_steps=16, action_normalization=E.fixed_unclipped_normalization()[1],
            slow_condition_path=str(condition), slow_condition_sha256=condition_sha,
            trajectory_path=str(trajectory), trajectory_sha256=trajectory_sha,
            trajectory_map_path=map_path, visual_input_mode='live_frozen_visual_v1',
            reference_image=images[0][0], reference_image_sha256=images[0][1],
            current_image=images[i][0], current_image_sha256=images[i][1],
            teacher_execution_provenance=provenance)
        validate_training_row(row)
        rows.append(row)
    return rows


def verify_manifest(manifest):
    from sft.action_head_joint_dataset import ActionHeadAnchorDataset, load_anchor_splits
    from sft.joint_fm_head_sft import joint_collate
    groups = load_anchor_splits(str(manifest), lambda row, _: validate_training_row(row))
    if not groups:
        return dict(ok=True, anchors=0, windows=0, loader='empty')
    data = ActionHeadAnchorDataset(str(manifest), groups=groups['train'])
    for group in data.groups:
        if len(group) != 4 or sorted(r['fast_window_index'] for r in group) != [0, 1, 2, 3]:
            raise ValueError('v8_export_does_not_have_exactly_four_windows')
        condition = load_condition(group[0])
        validate_messages(condition['slow_messages'])
        trajectory, _ = checked_json(group[0]['trajectory_path'], group[0]['trajectory_sha256'])
        if len(trajectory['measured_world_poses']) != 65:
            raise ValueError('v8_export_lost_canonical_trajectory')
    for i in range(len(data)):
        batch = joint_collate(data[i])
        if batch['target_actions'].shape != (4, 16, 3) or not batch['action_mask'].all():
            raise ValueError('v8_real_loader_shape_mismatch')
        for j, row in enumerate(data.groups[i]):
            trajectory = checked_json(row['trajectory_path'], row['trajectory_sha256'])[0]
            begin = trajectory['windows'][j]['start']
            end = trajectory['windows'][j]['end']
            normalization = row['action_normalization']
            physical = batch['target_actions'][j].numpy()*np.asarray(normalization['scale']) + np.asarray(normalization['offset'])
            E.assert_poses_close(E.world_poses_from_actions(trajectory['windows'][j]['world_pose'], physical),
                trajectory['planned_world_poses'][begin+1:end+1], 'v8_loader_label_endpoint_mismatch',
                xy_tolerance=1e-5, yaw_tolerance=1e-3)
    return dict(ok=True, anchors=len(data), windows=4*len(data),
        action_shape=[4, 16, 3], canonical_points_per_anchor=65,
        loader='ActionHeadAnchorDataset/JointBundleDataset/joint_collate')
