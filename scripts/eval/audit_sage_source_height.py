#!/usr/bin/env python3
"""Audit frozen SAGE cases with the executor's source-start height guard.

No existing audit checks source anchor Z against the active collision layer.
Reuse official provenance verification, canonical labels, OccMap and the exact
runtime guard; do not rewrite calls, trajectories, starts or source population.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from rule_teacher_pilot import verify_existing_instruction_record, atomic_json
from sage.datasets import SageSet, _to_geometry_frame
from sage.goal_region import (load_label_index, source_start_height_audit,
                              label_index_from_items, object_height_relation, same_floor_task_audit)
from sage.occmap import OccMap
from sft.rule_teacher_source_height_filter import case_execution_snapshot


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def frozen_json(path, expected):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError('frozen case input changed: ' + str(path))
    return json.loads(content)


def recorded_floor(metadata, task, objects, root, *, nav_cache, legacy_cache):
    """New cases require their recorded floor; old archives keep old physics."""
    profile, proof = metadata.get('sage_floor_profile'), metadata.get('sage_floor_contract')
    if profile is not None or proof is not None:
        if profile != 'sage-executed-floor-v1' or not isinstance(proof, dict):
            raise ValueError('SAGE navmesh case is missing its executed floor proof')
        from sage.navmesh_support import verify_floor_contract
        from sage.navmesh_map import floor_map
        nav, audit = verify_floor_contract(proof, task, objects,
            scene=metadata['scene'], episode=metadata['episode_id'], cache=nav_cache)
        inputs = {str(nav.manifest_path): proof['manifest_sha256']}
        for key in ('usd', 'navmesh'):
            inputs[nav.manifest[key+'_path']] = nav.manifest[key+'_sha256']
        if metadata.get('native_route_scope') is not None:
            from sage.same_floor_scope import native_route_admission
            binding = metadata['native_route_scope']
            verified = native_route_admission(binding, scene=metadata['scene'],
                task_id=metadata['episode_id'], provenance=metadata.get('existing_instruction_provenance'),
                nav=nav, start_audit=audit)
            if verified['status'] != 'retained' or metadata.get('sage_native_route_admission') != verified:
                raise ValueError('SAGE original route floor proof missing or changed')
            inputs[binding['path']] = binding['sha256']
        return floor_map(nav, audit), inputs
    key = str(root)
    if key not in legacy_cache:
        legacy_cache[key] = OccMap(str(root), collision_mode='usd_rooms', furniture_mode='none')
    occ = legacy_cache[key]
    inputs = {str(path): digest(path) for path in
              (root/'occupancy.json', root/'occupancy.png', Path(occ.usd_mask_path))}
    return occ, inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--case-manifest', type=Path)
    inputs.add_argument('--collection', type=Path, action='append')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-object-map', type=Path, default=os.environ.get('SAGE_SOURCE_OBJECT_MAP'))
    parser.add_argument('--source-object-map-sha256', default=os.environ.get('SAGE_SOURCE_OBJECT_MAP_SHA256'))
    args = parser.parse_args(argv)
    if bool(args.source_object_map) != bool(args.source_object_map_sha256):
        raise ValueError('corrected source object map and SHA256 required together')
    source_identity = None
    if args.source_object_map:
        from sage.source_objects import SCHEMA, sha256
        if sha256(args.source_object_map) != args.source_object_map_sha256:
            raise ValueError('corrected source object map changed')
        source_identity = dict(schema=SCHEMA, path=str(args.source_object_map.resolve()),
                               sha256=args.source_object_map_sha256)
    if args.collection:
        rows = []
        for collection in args.collection:
            for path in sorted(collection.glob('case_*/case.json')):
                result = path.with_name('result.json')
                if not result.is_file():
                    raise ValueError('SAGE collection case is not ended: '+str(path))
                case = json.loads(path.read_text())
                rows.append(dict(path=str(path.resolve()), sha256=digest(path),
                    result_path=str(result.resolve()), result_sha256=digest(result), scene=case['scene']))
        if not rows or len({row['path'] for row in rows}) != len(rows):
            raise ValueError('empty or duplicate ended SAGE case snapshot')
        args.case_manifest = args.output/'case-manifest.json'
        if args.case_manifest.exists():
            raise FileExistsError('use a new source-height audit directory')
        atomic_json(args.case_manifest, dict(schema='sage-ended-case-height-snapshot-v1', cases=rows))
    manifest = json.loads(args.case_manifest.read_text())
    if manifest.get('schema') != 'sage-ended-case-height-snapshot-v1':
        raise ValueError('expected an explicitly frozen ended-case manifest')
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    counts, cases, exclusions, source_inputs = Counter(), [], [], {}
    if source_identity:
        source_inputs[source_identity['path']] = source_identity['sha256']
    dataset = SageSet('train')
    cached_scene, objects = None, None
    nav_cache, legacy_cache = {}, {}
    for row in sorted(manifest['cases'], key=lambda r: (r['scene'], r['path'])):
        metadata = frozen_json(row['path'], row['sha256'])
        frozen_json(row['result_path'], row['result_sha256'])
        scene = metadata['scene']
        if scene != row['scene']:
            raise ValueError('snapshot scene changed')
        case = dict(path=row['path'], sha256=row['sha256'],
                    result_path=row['result_path'], result_sha256=row['result_sha256'],
                    scene=scene, episode=metadata['episode_id'])
        counts['cases'] += 1
        try:
            proof = metadata['existing_instruction_provenance']
            sample, item = verify_existing_instruction_record(proof,
                metadata['instruction'], scene, str(metadata['episode_id']))
            source_inputs[proof['path']] = proof['sha256']
            bindings, records = case_execution_snapshot(row['path'])
            if not any(record.get('accepted_for_training') for record in records):
                # Reset can reject an unresolved floor before any action. Keep
                # that original ended case without inventing executed-floor
                # evidence, and bind every receipt so it cannot gain an anchor
                # after this audit. Accepted holds/tiny turns still take the
                # unchanged strict floor audit below.
                binding = metadata.get('native_route_scope')
                if binding is not None:
                    if digest(binding['path']) != binding['sha256']:
                        raise ValueError('unexecuted SAGE route ledger changed')
                    source_inputs[binding['path']] = binding['sha256']
                case.update(status='no_training_anchors', training_anchor_count=0,
                    execution_snapshot=bindings, quality_approved=False)
                counts['no_training_anchors'] += 1
                cases.append(case)
                atomic_json(args.output/'progress.json', dict(counts=dict(counts),
                    updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
                continue
            if scene != cached_scene:
                nav_cache.clear()
                legacy_cache.clear()
                root = Path(dataset.scene_dir(scene))
                # The original reset proof consumes raw bbox/Z only. Keep its
                # legacy IDs and verify it before evaluating the corrected IDs.
                objects = (label_index_from_items(json.loads((root/'labels.json').read_text()))
                           if source_identity else load_label_index(str(root)))
                source_objects = objects
                if source_identity:
                    from sage.source_objects import load_source_label_index
                    source_objects = load_source_label_index(root, identity=source_identity)
                for path in [root/'labels.json', root/'structure.json']:
                    source_inputs[str(path)] = digest(path)
                cached_scene = scene
            source_position = sample['points'][0]['position']
            x, y, _ = _to_geometry_frame(float(source_position[0]),
                                        float(source_position[1]), 0.)
            task = SimpleNamespace(instruction=metadata['instruction'],
                goal_label=(item.get('end') or '').strip(),
                start_label=(item.get('start') or '').strip(),
                start=dict(x=x,y=y,z=float(source_position[2])))
            occ, geometry_inputs = recorded_floor(metadata, task, objects, root,
                nav_cache=nav_cache, legacy_cache=legacy_cache)
            source_inputs.update(geometry_inputs)
            if metadata.get('sage_floor_contract'):
                case['sage_floor_contract'] = metadata['sage_floor_contract']
            if metadata.get('native_route_scope') is not None:
                case['sage_native_route_admission'] = metadata['sage_native_route_admission']
            start = source_start_height_audit(task, occ, source_objects)
            goal_id = (item.get('end') or '').strip()
            goal = object_height_relation(source_objects.get(goal_id, {}), occ.ground_z(x,y))
            scope = same_floor_task_audit(task, occ, source_objects)
            if source_identity:
                from sage.source_objects import require_source_references
                from sage.navmesh_support import SourceCaseGeometryUnsupported
                proof = metadata.get('sage_floor_contract')
                corrected = dict(source_references=source_identity, actual_ground_z_m=float(occ.ground_z(x,y)),
                                 original_proof_preserved=True, reject=False)
                try:
                    require_source_references(task, source_objects)
                    if not proof:
                        raise SourceCaseGeometryUnsupported('missing measured executed floor proof')
                    key = (str(Path(proof['manifest_path']).resolve()), proof['manifest_sha256'], scene)
                    nav = nav_cache[key]
                    fresh = nav.resolve_start(task, source_objects)
                    corrected['start_audit'] = fresh
                    corrected['reject'] = abs(fresh['ground_z_m']-corrected['actual_ground_z_m']) > .06
                    corrected['reason'] = ('corrected_source_floor_differs_from_recorded_reset' if corrected['reject']
                                           else 'corrected_source_floor_matches_recorded_reset')
                except SourceCaseGeometryUnsupported as exc:
                    corrected.update(reject=True, reason='corrected_source_floor_unresolved', detail=str(exc))
                case['corrected_source_floor'] = corrected
                if corrected['reject']:
                    scope.update(reject=True, reason=corrected['reason'])
            case['same_floor_scope'] = scope
            counts[scope['reason']] += 1
            case.update(start=start, goal_id=goal_id, goal_height=goal,
                quality_approved=False, status='start_height_conflict' if start['reject']
                                             else 'no_proven_start_height_conflict')
            counts[case['status']] += 1
            if goal['available'] and not goal['compatible']:
                counts['goal_disjoint_from_start_layer'] += 1
            if start['reject'] or scope['reject']:
                # All windows from this wrong-floor observation are excluded,
                # including holds. This is unrelated to movement length.
                for execution_path in sorted((Path(row['path']).parent/'options').glob('*/execution.json')):
                    raw = execution_path.read_bytes()
                    execution = json.loads(raw)
                    if execution.get('accepted_for_training'):
                        exclusions.append(dict(path=str(execution_path),
                            sha256=hashlib.sha256(raw).hexdigest(), case_path=row['path'],
                            case_sha256=row['sha256'],
                            reason=('source_start_anchor_height_conflicts_with_active_floor'
                                    if start['reject'] else scope['reason'])))
                        counts['excluded_mechanically_admitted_anchors'] += 1
        except Exception as exc:
            case.update(status='audit_error', error=str(exc)[:1200], quality_approved=False)
            counts['audit_errors'] += 1
        cases.append(case)
        atomic_json(args.output/'progress.json', dict(counts=dict(counts),
            updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    exclusion_path = args.output/'anchor-exclusions.json'
    atomic_json(exclusion_path, dict(schema='sage-source-height-anchor-exclusions-v1',
        case_manifest=str(args.case_manifest), case_manifest_sha256=digest(args.case_manifest),
        excludes=exclusions, modifies_archived_data=False,
        rejects_short_or_hold_anchors=False, population_gate_approved=False))
    report = dict(schema='sage-source-height-audit-v1', counts=dict(counts),
        **({'source_references': source_identity} if source_identity else {}),
        case_manifest=str(args.case_manifest),case_manifest_sha256=digest(args.case_manifest),
        source_sha256=source_inputs, cases=cases, exclusions=str(exclusion_path),
        exclusions_sha256=digest(exclusion_path), wall_seconds=time.time()-started,
        quality_approved=False, population_changed=False,
        limitation='Height overlap is necessary, not complete trajectory quality or floor connectivity.')
    atomic_json(args.output/'audit.json', report)
    print(json.dumps(dict(counts=dict(counts),wall_seconds=report['wall_seconds'])), flush=True)
    if counts['audit_errors']:
        raise RuntimeError('some frozen cases could not be audited; do not treat them as admitted')


if __name__ == '__main__':
    main()
