#!/usr/bin/env python3
"""Build mechanically verified admissions under an explicitly accepted policy.

This does not review individual instruction semantics. A separately produced,
hash-pinned small-batch validation gate is mandatory; this tool never creates it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REVIEW_TYPE = 'rule_policy_admission'
REVIEW_SCHEMA = 'rule_teacher_policy_admission_v1'
GATE_SCHEMA = 'rule_teacher_policy_validation_v1'
POLICY_FILES = (
    'vln_demo/src/rule_teacher.py', 'vln_demo/src/mp3d_rule_teacher.py',
    'vln_demo/src/rule_text_retrieval.py', 'vln_demo/src/vln_gym/fm_action.py',
    'vln_demo/src/privileged_teacher.py', 'vln_demo/src/sage/goal_region.py',
    'scripts/eval/rule_teacher_pilot.py', 'verl/sft/rule_teacher_source_height_filter.py',
    # Bind direct admission and executor dependencies as well as entrypoints.
    'vln_demo/src/rule_map_grounding.py', 'vln_demo/src/rule_match_policy.py',
    'vln_demo/src/rule_lexicon.py', 'vln_demo/src/rule_nlp_grounding.py',
    'vln_demo/src/rule_call_nlp.py', 'vln_demo/src/rule_route_execution.py',
    'vln_demo/src/rule_plan_chunks.py', 'vln_demo/src/mp3d_teacher_semantics.py',
    'vln_demo/src/native_surface_route.py', 'vln_demo/src/native_stair_route.py',
    'vln_demo/src/sage/same_floor_scope.py', 'vln_demo/src/sage/occmap.py',
    'vln_demo/src/sage/planner.py', 'vln_demo/src/agent_tools.py',
    'verl/sft/export_rule_teacher_training.py',
    'verl/sft/rule_teacher_progress.py',
    'scripts/eval/rule_teacher_depth_scheduler.py',
    'verl/scripts/eval/action_head_fm_p0_p2/curriculum_action_head_eval.py',
    'vln_demo/src/sage/navmesh_map.py', 'vln_demo/src/sage/navmesh_support.py',
    'vln_demo/src/sage/source_objects.py', 'vln_demo/src/sage/physical_navigation.py',
    'vln_demo/src/sage/env.py', 'vln_demo/src/sage/rule_credit_state.py',
    'vln_demo/src/sage/build_navmesh.py',
    'verl/action_head/contract.py',
    'vln_demo/src/native_resource_search.py', 'vln_demo/src/native_compile_worker.py',
    'vln_demo/src/verl_access/vlm_policy.py',
    'vln_demo/src/vln_gym/env.py', 'vln_demo/src/vln_gym/server.py',
    'vln_demo/src/rxr.py',
    'vln_demo/src/vln_gym/native_episode_cache.py',
)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def file_binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path))


def checked_json(binding):
    path = Path(binding['path'])
    if not path.is_absolute() or file_sha(path) != binding['sha256']:
        raise ValueError('policy_evidence_hash_mismatch:' + str(path))
    return json.loads(path.read_text())


def policy_identity(source_root, configuration):
    root = Path(source_root).resolve()
    if (set(configuration) != {'admission_mode', 'grounding_mode', 'match_policy'} or
            configuration != dict(admission_mode='relaxed', grounding_mode='map_open', match_policy='target_or_direction')):
        raise ValueError('unsupported_rule_policy_configuration')
    return dict(files={name: file_sha(root/name) for name in POLICY_FILES},
                configuration=configuration)


def validate_policy_review(review):
    if (review.get('schema') != REVIEW_SCHEMA or review.get('review_type') != REVIEW_TYPE or
            review.get('human_verified') is not False or review.get('semantic_reviewed') is not False):
        raise ValueError('policy_admission_must_not_claim_individual_semantic_review')
    gate = checked_json(review['policy_gate'])
    policy = review['policy']
    expected = policy_identity(review['policy_source_root'], policy['configuration'])
    if policy != expected or digest(policy) != review['policy_sha256']:
        raise ValueError('policy_code_or_configuration_changed')
    if (gate.get('schema') != GATE_SCHEMA or gate.get('accepted') is not True or
            gate.get('scope') != 'rule_policy' or gate.get('policy_sha256') != digest(policy)):
        raise ValueError('missing_accepted_small_batch_policy_gate')
    report = checked_json(gate['validation_report'])
    if report.get('accepted') is not True or report.get('policy_sha256') != digest(policy):
        raise ValueError('small_batch_report_not_accepted_for_policy')
    if not isinstance(report.get('evidence'), list) or not report['evidence']:
        raise ValueError('small_batch_report_requires_hashed_evidence')
    for evidence in report['evidence']:
        if file_sha(evidence['path']) != evidence['sha256']:
            raise ValueError('small_batch_evidence_changed')
    return dict(policy_sha256=digest(policy), policy_gate=review['policy_gate'],
                validation_report=gate['validation_report'])


def collection_policy_binding(case_dir, policy):
    """Bind admission policy to the collector's original frozen launch inputs."""
    case_dir = Path(case_dir).resolve()
    launch_path = next((parent/'launch.json' for parent in case_dir.parents
                        if (parent/'launch.json').is_file()), None)
    if launch_path is None:
        raise ValueError('missing_collection_launch_for_policy')
    launch = json.loads(launch_path.read_text())
    plan_path = Path(launch['source_plan']).resolve()
    plan = json.loads(plan_path.read_text())
    matches = []
    for collector in launch.get('collectors', []):
        argv = collector.get('command', [])
        if '--output' not in argv:
            continue
        output = Path(argv[argv.index('--output')+1]).resolve()
        if case_dir.parent != output:
            continue
        scripts = [Path(arg).resolve() for arg in argv if arg.endswith(
            ('/scripts/eval/rule_teacher_pilot.py', '/scripts/eval/mp3d_rule_teacher_pilot.py'))]
        if len(scripts) == 1:
            matches.append((collector, scripts[0].parents[2]))
    if len(matches) != 1:
        raise ValueError('missing_or_ambiguous_policy_collector')
    collector, source_root = matches[0]
    for name, expected in policy['files'].items():
        path = source_root/name
        if (plan.get('code_sha256', {}).get(str(path)) != expected or
                file_sha(path) != expected):
            raise ValueError('collection_policy_code_mismatch:' + name)
    environment = {**launch.get('environment', {}), **collector.get('environment', {})}
    for key, env_name in dict(admission_mode='RULE_TEACHER_ADMISSION',
            grounding_mode='RULE_TEACHER_GROUNDING', match_policy='RULE_TEACHER_MATCH_POLICY').items():
        if environment.get(env_name) != policy['configuration'][key]:
            raise ValueError('collection_policy_configuration_mismatch:' + key)
    selection_path = case_dir.parent/'selection.json'
    if plan.get('input_sha256', {}).get(str(selection_path)) != file_sha(selection_path):
        raise ValueError('collection_selection_not_frozen')
    return dict(collection_launch=file_binding(launch_path),
                collection_source_plan=file_binding(plan_path),
                collection_selection=file_binding(selection_path))


def validate_policy_segment(segment, execution, *, execution_path, review):
    """Verify the selected rule call and every observed frame, without semantics."""
    if (segment.get('human_verified') is not False or segment.get('semantic_reviewed') is not False or
            segment.get('semantic_conforms_current_scope') not in (None, False) or
            segment.get('reviewer') is not None):
        raise ValueError('policy_segment_falsely_claims_semantic_review')
    evidence = segment['rule_evidence']
    collection = collection_policy_binding(Path(execution_path).parents[2], review['policy'])
    if any(evidence.get(key) != value for key, value in collection.items()):
        raise ValueError('policy_collection_binding_mismatch')
    if evidence['execution'] != file_binding(execution_path):
        raise ValueError('policy_execution_binding_mismatch')
    candidate = checked_json(evidence['candidate'])
    selection = checked_json(evidence['selection'])
    binding = checked_json(evidence['binding'])
    accepted = checked_json(evidence['accepted'])
    request = checked_json(evidence['request'])
    response = checked_json(evidence['response'])
    decision = checked_json(evidence['decision'])
    if (candidate.get('selected') is not True or candidate.get('status') != 'selected' or
            candidate.get('index') != selection.get('selected_index') or selection.get('all_rejected') is not False or
            candidate.get('original_text') != response.get('text') or candidate.get('request') != request or
            decision.get('request') != request or decision.get('response') != response):
        raise ValueError('policy_candidate_not_the_executed_selection')
    calls = candidate.get('calls', [])
    if (len(calls) != 1 or calls[0].get('name') != 'run_action_head' or
            calls[0].get('arguments') != execution['original_call'] or
            binding.get('original_call') != execution['original_call'] or
            binding.get('scene') != execution['scene'] or binding.get('episode') != execution['episode'] or
            binding.get('pose') != execution['state']['initial_pose'] or
            candidate.get('real_pose') != execution['state']['initial_pose']):
        raise ValueError('policy_rule_call_or_state_mismatch')
    for key, value in review['policy']['configuration'].items():
        if binding.get(key) != value:
            raise ValueError('policy_rule_configuration_mismatch:' + key)
    state = execution['state']
    if (accepted.get('execution_contract') != 'full_target_route_v1' or
            accepted.get('route_chunks') != 4 or
            accepted.get('route_target_world_poses') != state.get('route_target_world_poses') or
            accepted.get('initial_pose') != state.get('initial_pose')):
        raise ValueError('policy_accepted_route_differs_from_execution')
    expected_frames = [file_binding(Path(execution_path).parent/f'actual_chunk_{i}.png') for i in range(5)]
    if evidence.get('frames') != expected_frames:
        raise ValueError('policy_actual_frame_hash_mismatch')
    if evidence.get('case') != file_binding(Path(execution_path).parents[2]/'case.json'):
        raise ValueError('policy_case_hash_mismatch')
    if segment['decision_path'] != evidence['decision']['path'] or segment['request_path'] != evidence['request']['path']:
        raise ValueError('policy_segment_evidence_path_mismatch')
    return dict(rule_evidence_sha256=digest(evidence), rule_accepted=True)


def build_segment(execution_path, review, archive_index=None):
    from sft import export_rule_teacher_training as E
    path = Path(execution_path).resolve()
    execution = json.loads(path.read_text())
    folder, case_dir = path.parent, path.parents[2]
    request_path, response_path = folder/'brain_request.json', folder/'brain_response.json'
    request, response = json.loads(request_path.read_text()), json.loads(response_path.read_text())
    # An option can follow turn/stop decisions; match actual archived context,
    # not an assumed one-to-one relation between option and decision numbers.
    if archive_index is None:
        matches = []
        for decision_path in case_dir.glob('decision_*.json'):
            decision = json.loads(decision_path.read_text())
            if decision.get('request') == request and decision.get('response') == response:
                matches.append((decision_path, decision))
    else:
        matches = archive_index.matching_decisions(case_dir, request, response)
    if len(matches) != 1:
        raise ValueError('missing_or_ambiguous_executed_decision')
    decision_path, decision = matches[0]
    candidates = case_dir/'candidates'/f"decision_{decision['decision_index']:03d}"
    selection_path = candidates/'selection.json'
    selection = json.loads(selection_path.read_text())
    selected = selection['selected_index']
    # v6_depth archives the initial eight and the one retry in a single
    # candidate namespace; retry candidates therefore use indices 8..15.
    limit = 16 if selection.get('strategy') == 'v6_depth' else 8
    capacity = selection.get('maximum_candidates', limit)
    if (type(capacity) is not int or not 1 <= capacity <= limit or
            type(selected) is not int or not 0 <= selected < capacity):
        raise ValueError('missing_selected_rule_candidate')
    candidate = candidates/f'candidate_{selected:02d}.json'
    binding_dir = candidates/f'candidate_{selected:02d}_binding'
    evidence = {name: file_binding(value) for name, value in dict(
        execution=path, candidate=candidate, selection=selection_path,
        binding=binding_dir/'binding.json', accepted=binding_dir/'accepted.json',
        request=request_path, response=response_path, decision=decision_path, case=case_dir/'case.json').items()}
    evidence['frames'] = [file_binding(folder/f'actual_chunk_{i}.png') for i in range(5)]
    evidence.update(collection_policy_binding(case_dir, review['policy']))
    state = execution['state']
    segment = dict(execution_path=str(path), request_path=str(request_path),
        request_file_sha256=file_sha(request_path), decision_path=str(decision_path),
        raw_call=execution['original_call'], raw_call_sha256=execution['call_sha256'],
        initial_pose_world_xy_yaw_deg=state['initial_pose'],
        final_pose_world_xy_yaw_deg=state['training_windows'][-1]['actual_end_pose'],
        start_png=str(folder/'actual_chunk_0.png'), end_png=str(folder/'actual_chunk_4.png'),
        start_png_sha256=evidence['frames'][0]['sha256'], end_png_sha256=evidence['frames'][4]['sha256'],
        human_verified=False, semantic_reviewed=False, rule_evidence=evidence,
        reason='Rule policy accepted; full route verified against four real execution windows.')
    E.validate_segment(segment, execution, request, execution_path=path, policy_review=review)
    from sft.rule_teacher_progress import execution_progress
    if not execution_progress(execution, E.preceding_executions(path, archive_index))['admissible_geometry']:
        raise ValueError('actual_geometry_rejected')
    E.original_completion(response, execution['original_call'])
    return segment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, action='append', required=True)
    parser.add_argument('--policy-source-root', type=Path, required=True)
    parser.add_argument('--policy-gate', type=Path, required=True)
    parser.add_argument('--policy-gate-sha256', required=True)
    parser.add_argument('--policy-configuration', required=True, help='JSON binding fields')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-height-report', type=Path)
    parser.add_argument('--source-height-report-sha256')
    parser.add_argument('--dedup-ledger', type=Path)
    parser.add_argument('--dedup-ledger-sha256')
    args = parser.parse_args(argv)
    from sft.export_rule_teacher_training import ARTIFACTS, atomic_json
    if not args.output.resolve().is_relative_to(ARTIFACTS) or args.output.exists():
        raise ValueError('use a new review on the authorized data disk')
    policy = policy_identity(args.policy_source_root, json.loads(args.policy_configuration))
    review = dict(schema=REVIEW_SCHEMA, review_type=REVIEW_TYPE, human_verified=False,
        semantic_reviewed=False, policy_source_root=str(args.policy_source_root.resolve()),
        policy=policy, policy_sha256=digest(policy), policy_gate=dict(
            path=str(args.policy_gate.resolve()), sha256=args.policy_gate_sha256), segments=[], exclusions=[])
    validate_policy_review(review)  # Must succeed before scanning or publishing admissions.
    from sft.rule_teacher_anchor_dedup import checked_decisions, keep_execution
    if bool(args.dedup_ledger) != bool(args.dedup_ledger_sha256):
        raise ValueError('dedup ledger and frozen sha256 are required together')
    dedup = checked_decisions(args.dedup_ledger, args.dedup_ledger_sha256) if args.dedup_ledger else None
    review['dedup_ledger'] = (dict(path=str(args.dedup_ledger.resolve()), sha256=args.dedup_ledger_sha256)
                              if dedup is not None else None)
    from sft.rule_teacher_source_height_filter import load_source_height_filter
    height_filter = load_source_height_filter(args.source_height_report, args.source_height_report_sha256)
    review['source_height_audit'] = height_filter.binding if height_filter else None
    from sft.export_rule_teacher_training import CaseArchiveIndex
    archive_index = CaseArchiveIndex()
    for root in args.collection:
        for path in sorted(root.glob('case_*/options/*/execution.json')):
            try:
                if height_filter is not None:
                    from sft.export_rule_teacher_training import original_task_identity
                    execution = json.loads(path.read_text())
                    case = json.loads((path.parents[2]/'case.json').read_text())
                    identity = original_task_identity(case, execution, case_dir=path.parents[2])
                    reason = height_filter.rejection(path, file_sha(path),
                        identity['source_dataset'] if identity else None)
                    if reason:
                        review['exclusions'].append(dict(execution_path=str(path.resolve()),
                            reason=reason, fast_windows_removed=4))
                        continue
                segment = build_segment(path, review, archive_index)
                if dedup is not None and not keep_execution(path, file_sha(path), dedup):
                    review['exclusions'].append(dict(execution_path=str(path.resolve()),
                        reason='redundant_middle_of_exact_hold_run', fast_windows_removed=4))
                    continue
                review['segments'].append(segment)
            except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
                review['exclusions'].append(dict(execution_path=str(path.resolve()), reason=str(exc)))
    atomic_json(args.output, review)
    print(json.dumps(dict(output=str(args.output), sha256=file_sha(args.output),
        admitted=len(review['segments']), excluded=len(review['exclusions']))))


if __name__ == '__main__':
    main()
