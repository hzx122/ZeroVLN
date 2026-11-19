"""Consume the existing SAGE height audit without granting semantic approval.

Reuse the policy review's hash-bound evidence reader. No existing dedup reader
checks source-floor evidence; this filter preserves complete four-window anchors
and rejects uncovered SAGE cases instead of silently calling them reviewed.
"""
from pathlib import Path

from sft.build_rule_teacher_policy_review import checked_json
from sft.collect_action_head_recovery import sha256_file


def case_execution_snapshot(case_path):
    """Bind the same option receipts as export, including rejected attempts.

    Reuse hash-checked evidence reads; freeze membership as well as file bytes
    so an empty case cannot later acquire an unaudited training anchor.
    """
    folder = Path(case_path).resolve().parent
    paths = sorted((folder/'options').glob('*/execution.json'))
    bindings = [dict(path=str(path.resolve()), sha256=sha256_file(path)) for path in paths]
    records = [checked_json(binding) for binding in bindings]
    if sorted((folder/'options').glob('*/execution.json')) != paths:
        raise ValueError('source_height_execution_membership_changed')
    return bindings, records


def verify_no_training_anchors(case):
    bindings, records = case_execution_snapshot(case['path'])
    if (case.get('training_anchor_count') != 0 or case.get('execution_snapshot') != bindings
            or any(record.get('accepted_for_training') for record in records)):
        raise ValueError('source_height_empty_case_has_changed_or_training_execution')


class SourceHeightFilter:
    def __init__(self, report_path, report_sha256):
        self.binding = dict(path=str(Path(report_path).resolve()), sha256=report_sha256)
        report = checked_json(self.binding)
        if (report.get('schema') != 'sage-source-height-audit-v1'
                or report.get('population_changed') is not False
                or report.get('quality_approved') is not False
                or report['counts'].get('audit_errors', 0)):
            raise ValueError('source_height_audit_incomplete')
        snapshot = checked_json(dict(path=report['case_manifest'],
                                     sha256=report['case_manifest_sha256']))
        if snapshot.get('schema') != 'sage-ended-case-height-snapshot-v1':
            raise ValueError('source_height_snapshot_schema')
        for path, expected in report['source_sha256'].items():
            if sha256_file(Path(path)) != expected:
                raise ValueError('source_height_geometry_or_source_changed:' + path)
        source_references = report.get('source_references')
        if source_references is not None:
            from sage.source_objects import SCHEMA, load_source_label_index
            from sage.datasets import SageSet
            if source_references.get('schema') != SCHEMA:
                raise ValueError('source_height_source_reference_schema')
            checked_json(source_references)
            # Load checks each scene file and its labels/occupancy hashes.
            for scene in {row['scene'] for row in snapshot['cases']}:
                load_source_label_index(SageSet('train').scene_dir(scene), identity=source_references)
        self.cases = {}
        frozen = {str(Path(row['path']).resolve()): row for row in snapshot['cases']}
        for case in report['cases']:
            path = str(Path(case['path']).resolve())
            original = frozen.get(path)
            if original is None or any(case[key] != original[key] for key in
                    ('sha256', 'result_path', 'result_sha256', 'scene')):
                raise ValueError('source_height_case_snapshot_mismatch')
            for prefix in ('', 'result_'):
                checked_json(dict(path=case[prefix+'path'], sha256=case[prefix+'sha256']))
            metadata = checked_json(dict(path=case['path'], sha256=case['sha256']))
            if case.get('status') == 'no_training_anchors':
                if path in self.cases or case.get('quality_approved') is not False:
                    raise ValueError('source_height_case_status_invalid')
                verify_no_training_anchors(case)
                self.cases[path] = case
                continue
            if metadata.get('native_route_scope') is not None:
                route_proof = metadata.get('sage_native_route_admission')
                if (not isinstance(route_proof, dict) or route_proof.get('status') != 'retained'
                        or route_proof.get('schema') != 'sage-native-whole-route-runtime-scope-v1'
                        or route_proof.get('ledger') != metadata['native_route_scope']
                        or case.get('sage_native_route_admission') != route_proof):
                    raise ValueError('source_height_whole_route_floor_proof_missing_or_mismatched')
                binding = route_proof['ledger']
                if report['source_sha256'].get(binding['path']) != binding['sha256']:
                    raise ValueError('source_height_whole_route_floor_ledger_not_frozen')
            if metadata.get('sage_floor_profile') or metadata.get('sage_floor_contract'):
                proof = metadata.get('sage_floor_contract')
                if (metadata.get('sage_floor_profile') != 'sage-executed-floor-v1'
                        or not isinstance(proof, dict) or case.get('sage_floor_contract') != proof):
                    raise ValueError('source_height_saved_floor_proof_missing_or_mismatched')
            if (path in self.cases or case.get('status') not in
                    ('start_height_conflict', 'no_proven_start_height_conflict')
                    or case['start']['reject'] != (case['status'] == 'start_height_conflict')):
                raise ValueError('source_height_case_status_invalid')
            scope = case.get('same_floor_scope', {})
            if source_references is not None:
                corrected = case.get('corrected_source_floor', {})
                if (corrected.get('source_references') != source_references
                        or corrected.get('original_proof_preserved') is not True
                        or not isinstance(corrected.get('reject'), bool)
                        or (corrected['reject'] and not scope.get('reject'))):
                    raise ValueError('corrected_source_floor_evidence_missing')
            if (scope.get('schema') != 'sage-same-floor-scope-v2'
                    or not isinstance(scope.get('reject'), bool) or not scope.get('reason')):
                raise ValueError('sage_same_floor_scope_missing_rerun_height_audit')
            self.cases[path] = case
        if len(self.cases) != len(frozen) or len(self.cases) != report['counts']['cases']:
            raise ValueError('source_height_audit_missing_case')

    def rejection(self, execution_path, execution_sha256, source):
        if source != 'sage':
            return None
        path = Path(execution_path).resolve()
        if sha256_file(path) != execution_sha256:
            raise ValueError('source_height_execution_changed')
        case_path = path.parents[2]/'case.json'
        case = self.cases.get(str(case_path))
        if case is None:
            raise ValueError('sage_case_not_covered_by_source_height_audit')
        if sha256_file(case_path) != case['sha256']:
            raise ValueError('source_height_case_changed')
        if case.get('status') == 'no_training_anchors':
            verify_no_training_anchors(case)
            # This case is represented in the population/failure ledger, but
            # grants no execution admission, including a zero-distance hold.
            return 'sage_case_has_no_training_anchors'
        if case.get('sage_floor_contract') is not None:
            execution = checked_json(dict(path=str(path), sha256=execution_sha256))
            if execution.get('state', {}).get('sage_floor_contract') != case['sage_floor_contract']:
                raise ValueError('source_height_execution_floor_proof_mismatch')
        if case.get('sage_native_route_admission') is not None:
            execution = checked_json(dict(path=str(path), sha256=execution_sha256))
            if execution.get('state', {}).get('sage_native_route_admission') != case['sage_native_route_admission']:
                raise ValueError('source_height_execution_whole_route_floor_proof_mismatch')
        if case['start']['reject']:
            return 'source_start_anchor_height_conflicts_with_active_floor'
        if case['same_floor_scope']['reject']:
            return case['same_floor_scope']['reason']
        return None


def load_source_height_filter(path=None, sha256=None):
    if bool(path) != bool(sha256):
        raise ValueError('source height report and frozen sha256 are required together')
    return SourceHeightFilter(path, sha256) if path else None


def main():
    """Produce a derived mechanical keep list; never a semantic approval."""
    import argparse
    from collections import Counter
    import json
    from sft.collect_action_head_recovery import atomic_json, atomic_jsonl
    from sft.rule_teacher_anchor_dedup import checked_decisions, keep_execution
    from sft.export_rule_teacher_training import ARTIFACTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-height-report', type=Path, required=True)
    parser.add_argument('--source-height-report-sha256', required=True)
    parser.add_argument('--dedup-ledger', type=Path)
    parser.add_argument('--dedup-ledger-sha256')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ARTIFACTS):
        raise ValueError('use a new output on the authorized data disk')
    check = load_source_height_filter(args.source_height_report, args.source_height_report_sha256)
    if bool(args.dedup_ledger) != bool(args.dedup_ledger_sha256):
        raise ValueError('dedup ledger and frozen hash are required together')
    dedup = checked_decisions(args.dedup_ledger, args.dedup_ledger_sha256) if args.dedup_ledger else None
    counts, keep, excludes = Counter(), [], []
    for case in check.cases.values():
        for path in sorted((Path(case['path']).parent/'options').glob('*/execution.json')):
            execution = json.loads(path.read_text())
            if execution.get('accepted_for_training') is not True:
                continue
            expected = sha256_file(path)
            height_reason = check.rejection(path, expected, 'sage')
            redundant = dedup is not None and not keep_execution(path, expected, dedup)
            row = dict(execution_path=str(path.resolve()), execution_sha256=expected,
                       case_path=case['path'], case_sha256=case['sha256'])
            counts['original_mechanically_admitted_anchors'] += 1
            counts['height_excluded_anchors'] += bool(height_reason)
            counts['redundant_anchors'] += bool(redundant)
            counts['overlap_exclusions'] += bool(height_reason and redundant)
            if height_reason or redundant:
                row['reasons'] = ([height_reason] if height_reason else []) + (
                    ['redundant_middle_of_exact_hold_run'] if redundant else [])
                excludes.append(row)
            else:
                keep.append(row)
    counts.update(mechanically_retained_anchors=len(keep), excluded_anchors=len(excludes))
    atomic_jsonl(args.output/'mechanical-keep.jsonl', keep)
    atomic_jsonl(args.output/'excluded.jsonl', excludes)
    atomic_json(args.output/'report.json', dict(schema='rule_teacher_joint_anchor_filter_v1',
        counts=dict(counts), source_height_audit=check.binding,
        dedup_ledger=(dict(path=str(args.dedup_ledger.resolve()), sha256=args.dedup_ledger_sha256)
                      if dedup is not None else None),
        semantic_quality_approved=False, formal_population_changed=False,
        original_files_modified=False, short_motion_or_hold_rejection=False,
        full_anchor_fast_windows=4, limitation='Retained anchors still require semantic and label quality admission.'))
    print(json.dumps(dict(counts=dict(counts), output=str(args.output))))


if __name__ == '__main__':
    main()
