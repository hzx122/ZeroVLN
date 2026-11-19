"""Dataset sampling is independent of the action expert prompt protocol.

Historical v6 beam manifests keep their original default. Online corrections
are independent anchors; they must never invent golden branch ancestry.
"""
from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

BEAM_SAMPLING = 'fixed_two_complete_case_branches_v1'
ANCHOR_SAMPLING = 'independent_anchors_v1'
ONLINE_CONTRACT = 'v6_online_corrections_v1'
TEACHER_CONTRACT = 'v6_local_teacher_checked_v1'
COLLISION_CONTRACT = 'v6_collision_task_oracle_v1'
DIRECTION_CONTRACT = 'v6_direction_mismatch_task_oracle_v1'
GOLDEN_EXTRA_CONTRACT = 'v6_corrected_golden_slow_anchor_v1'
RULE_EXECUTION_CONTRACT = 'v6_rule_teacher_execution_v1'
RULE_FULL_ROUTE_CONTRACT = 'v6_rule_teacher_full_target_route_v1'
RULE_UNCLIPPED_NORMALIZATION = 'quantile_affine_unclipped_v1'
RULE_SELECTION_CONTRACT = 'v6_reviewed_target_direction_v1'
RULE_POLICY_SELECTION_CONTRACT = 'v6_rule_policy_admission_v1'
ORIGINAL_TASK_GROUP_SCHEMA = 'rule_teacher_original_task_group_v1'


def original_task_group(identity):
    """Stable holdout identity, independent of generated wording and sampling seed."""
    if not isinstance(identity, dict):
        raise ValueError('invalid_original_task_group_identity')
    source = identity.get('source_dataset')
    keys = ('start', 'end') if source == 'sage' else ('episode',)
    required = ('schema', 'source_dataset', 'scene', *keys)
    if (identity.get('schema') != ORIGINAL_TASK_GROUP_SCHEMA or
            source not in ('sage', 'r2r', 'rxr') or set(identity) != set(required) or
            any(not isinstance(identity.get(key), str) or not identity[key].strip()
                for key in required)):
        raise ValueError('invalid_original_task_group_identity')
    digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    return f'{source}:{digest}'


def validate_original_task_group(row):
    proof = row.get('teacher_execution_provenance', {}).get('original_task_identity')
    if proof is None or row.get('original_task_group') != original_task_group(proof):
        raise ValueError('original_task_group_missing_or_mismatched_reexport_required')
    if (proof['scene'] != row['source_scene'] or
            row.get('source_dataset', proof['source_dataset']) != proof['source_dataset']):
        raise ValueError('original_task_group_source_mismatch')
    return row['original_task_group']


def generated_instruction_task_id(scene, source_task, instruction, proof):
    """Keep generated instructions distinct while the real reset uses source_task."""
    if (proof.get('schema') not in ('sage_official_generated_instruction_v1',
                                   'mp3d_official_generated_instruction_v1') or
            proof.get('source_task') != source_task or proof.get('scene') != scene or
            not re.fullmatch(r'[0-9a-f]{64}', str(proof.get('sha256', ''))) or
            type(proof.get('index')) is not int or proof['index'] < 0 or
            not isinstance(instruction, str) or not instruction.strip()):
        raise ValueError('invalid_generated_instruction_identity')
    value = dict(schema=proof['schema'], scene=scene, source_task=source_task,
                 instruction_file_sha256=proof['sha256'], instruction_index=proof['index'],
                 instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest())
    digest = hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    return f'{source_task}__generated_{digest[:24]}'


def sampling_contract(row):
    value = row.get('sampling_contract')
    if value is None:
        value = (ANCHOR_SAMPLING if row.get('data_contract') in (
                     ONLINE_CONTRACT, RULE_EXECUTION_CONTRACT, RULE_FULL_ROUTE_CONTRACT)
                 or row.get('action_head_as_tool_version') != 'v6' else BEAM_SAMPLING)
    if value not in (BEAM_SAMPLING, ANCHOR_SAMPLING):
        raise ValueError(f'unknown sampling contract: {value}')
    return value


def manifest_sampling_contract(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get('split', 'train') == 'train':
                    return sampling_contract(row)
    raise ValueError(f'empty training manifest: {path}')


def validate_v6_training_row(row):
    from action_head import protocol_v6 as protocol
    from sft.golden_context_beam import validate_context
    protocol.assert_protocol_identity(row)
    if row.get('schema') != protocol.BUNDLE_SCHEMA:
        raise ValueError(f'expected {protocol.BUNDLE_SCHEMA}')
    validate_context(row['slow_messages'], row['trajectory_map_path'])
    visual_mode = row.get('visual_input_mode', 'preencoded')
    if visual_mode not in ('preencoded', 'live_frozen_visual_v1'):
        raise ValueError('unknown fast visual input contract')
    if visual_mode == 'live_frozen_visual_v1':
        for key in ('reference_image', 'current_image'):
            if (not Path(row.get(key, '')).is_absolute() or
                    not re.fullmatch(r'[0-9a-f]{64}', str(row.get(key+'_sha256', '')))):
                raise ValueError('live fast images require immutable RGB bindings')

    if row.get('data_contract') in (RULE_EXECUTION_CONTRACT, RULE_FULL_ROUTE_CONTRACT):
        approval = row.get('teacher_execution_provenance', {})
        if 'original_task_group' in row or 'original_task_identity' in approval:
            validate_original_task_group(row)
        is_policy = approval.get('review_type') == 'rule_policy_admission'
        if is_policy:
            if (row.get('data_contract') != RULE_FULL_ROUTE_CONTRACT or
                    row.get('decision_filter') != RULE_POLICY_SELECTION_CONTRACT or
                    row.get('sample_type') != 'rule_policy_execution' or
                    approval.get('scope') != 'rule_policy_execution' or
                    approval.get('human_verified') is not False or
                    approval.get('semantic_reviewed') is not False or
                    approval.get('rule_accepted') is not True or approval.get('route_chunks') != 4 or
                    approval.get('reviewer') is not None or
                    not re.fullmatch(r'[0-9a-f]{64}', str(approval.get('policy_sha256', ''))) or
                    not re.fullmatch(r'[0-9a-f]{64}', str(approval.get('rule_evidence_sha256', '')))):
                raise ValueError('v6 rule policy admission lacks honest policy/evidence provenance')
            validate_original_task_group(row)
            for key in ('policy_gate', 'validation_report'):
                item = approval.get(key, {})
                if not Path(item.get('path', '')).is_absolute() or not re.fullmatch(r'[0-9a-f]{64}', str(item.get('sha256', ''))):
                    raise ValueError('v6 rule policy admission lacks hash-pinned validation gate')
        elif (row.get('decision_filter') != RULE_SELECTION_CONTRACT or
                row.get('sample_type') != 'reviewed_rule_execution' or
                approval.get('review_type') not in ('assistant_review', 'human_review') or
                approval.get('scope') not in ('current_target_and_basic_direction',
                                              'current_target_or_basic_direction')):
            raise ValueError('v6 rule execution lacks reviewed local execution provenance')
        if (sampling_contract(row) != ANCHOR_SAMPLING or approval.get('approved') is not True or
                approval.get('executed') is not True or approval.get('prediction_horizon') != 16 or
                approval.get('executed_horizon') != 4 or approval.get('privileged_data_in_condition') is not False):
            raise ValueError('v6 rule execution lacks reviewed local execution provenance')
        if any(not re.fullmatch(r'[0-9a-f]{64}', str(approval.get(key, '')))
               for key in ('condition_sha256', 'execution_sha256', 'review_sha256', 'route_sha256')):
            raise ValueError('v6 rule execution lacks condition/execution/review/route binding')
        if row['data_contract'] == RULE_FULL_ROUTE_CONTRACT:
            import math
            normalization = row.get('action_normalization', {})
            chunks = approval.get('route_chunks')
            if (approval.get('execution_contract') != 'full_target_route_v1' or
                    approval.get('full_route_complete') is not True or
                    type(chunks) is not int or chunks < 4 or
                    approval.get('executed_route_steps') != 4 * chunks or
                    not 0 <= row.get('fast_window_index', -1) < chunks or
                    normalization.get('name') != RULE_UNCLIPPED_NORMALIZATION or
                    'clipping' not in normalization or normalization['clipping'] is not None or
                    not re.fullmatch(r'[0-9a-f]{64}', str(normalization.get('stats_sha256', '')))):
                raise ValueError('v6 full route lacks complete execution/unclipped normalization binding')
            from action_head.policy import TrajectoryTransform
            from sft.build_abotn_sft_data import ZYT_STATS
            import hashlib
            low = [ZYT_STATS[key]['q01'] for key in ('dx', 'dy', 'dyaw')]
            high = [ZYT_STATS[key]['q99'] for key in ('dx', 'dy', 'dyaw')]
            transform = TrajectoryTransform.from_quantiles(low, high, flip_y=True, flip_yaw=True)
            stats_sha = hashlib.sha256(json.dumps(ZYT_STATS, ensure_ascii=False, sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            if (normalization.get('stats_sha256') != stats_sha or
                    normalization.get('q01') != low or normalization.get('q99') != high or
                    normalization.get('scale') != list(transform.scale) or
                    normalization.get('offset') != list(transform.offset)):
                raise ValueError('v6 full route normalization differs from fixed deployment transform')
            for key, limit in (('goal_error_m', .035), ('goal_yaw_error_deg', 2.)):
                value = approval.get(key)
                if (not isinstance(value, (int, float)) or not math.isfinite(value)
                        or not 0 <= value < limit):
                    raise ValueError('v6 full route lacks verified local endpoint completion')
    elif row.get('data_contract') == ONLINE_CONTRACT:
        kind = row.get('sample_type')
        physical = kind in ('collision_relabel', 'direction_mismatch_relabel',
                            'golden_route_relabel')
        contract = (GOLDEN_EXTRA_CONTRACT if kind == 'golden_route_relabel' else
                    COLLISION_CONTRACT if kind == 'collision_relabel' else
                    DIRECTION_CONTRACT if kind == 'direction_mismatch_relabel' else TEACHER_CONTRACT)
        if row.get('decision_filter') != contract:
            raise ValueError('v6 correction selection contract mismatch')
        if not physical and kind not in ('execution_relabel', 'recovery'):
            raise ValueError('unknown v6 correction sample_type')
        if kind == 'golden_route_relabel':
            provenance = row.get('bootstrap_provenance', {})
            if (row.get('failure_modes') not in (['collision'], ['direction_mismatch']) or
                    type(provenance.get('trigger_call_index')) is not int or
                    provenance.get('parent_condition_call_index') != provenance['trigger_call_index'] or
                    type(provenance.get('condition_call_index')) is not int or
                    provenance['condition_call_index'] == provenance['trigger_call_index'] or
                    float(provenance.get('golden_offset_m') or 0.) < 1.6):
                raise ValueError('golden slow anchor lacks parent and route offset binding')
        elif physical:
            provenance = row.get('bootstrap_provenance', {})
            expected_mode = ('collision' if kind == 'collision_relabel' else 'direction_mismatch')
            if (row.get('failure_modes') != [expected_mode] or
                    provenance.get('trigger_call_index') != provenance.get('condition_call_index') or
                    type(provenance.get('trigger_call_index')) is not int):
                raise ValueError('physical correction must bind the failed call itself')
        if sampling_contract(row) != ANCHOR_SAMPLING:
            raise ValueError('v6 online corrections require independent anchor sampling')
        approval = row.get('bootstrap_provenance', {}).get(
            'route_validation' if physical else 'teacher_approval', {})
        modes = ('physical_oracle',) if physical else ('reviewed', 'semantic_judge')
        if approval.get('approved') is not True or approval.get('mode') not in modes:
            raise ValueError('v6 correction lacks a valid route record')
        if physical and kind != 'golden_route_relabel' and (
                approval.get('planned_direction') not in ('left', 'right', 'straight')
                or approval.get('oracle_prefix_direction') != approval['planned_direction']):
            raise ValueError('physical correction lacks plan/oracle direction alignment')
        if any(not re.fullmatch(r'[0-9a-f]{64}', str(approval.get(key, '')))
               for key in ('condition_sha256', 'route_sha256')):
            raise ValueError('v6 teacher approval lacks condition/route binding')
    elif row.get('decision_filter') != 'none' or not row.get('golden_anchor_id'):
        raise ValueError('v6 golden data requires no-filter beam provenance')
    sampling_contract(row)
