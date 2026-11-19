"""Bind completed original cases read-only in a fresh collection attempt.

The existing collector skips only result.json cases. All unfinished or
infrastructure-error cases remain absent and are recollected from original
instructions. Does not launch or alter an old output.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from dlc_control_io_retry import retry_estale
from formal_rule_teacher_four_node import selected_sources

ART = Path('/data/zerovln/artifacts')


def v8_resume_catalog(attempt_roots, source, *, node_indices=(0, 1, 2, 3), node_roots=None):
    """Index immutable completed episodes; retain every interrupted prefix.

    V8 changes scene placement across attempts, so match original identities,
    never collector indices. This is case-level continuation: interrupted
    episodes need a fresh environment; their prior labels remain archived.
    """
    completed, archives = {}, []
    for attempt in map(Path, attempt_roots):
        if not attempt.is_relative_to(ART):
            raise ValueError('resume lineage must stay on the authorized artifact disk')
        for node in node_indices:
            root = Path(node_roots[node]) if node_roots is not None else attempt/source/f'node{node}'
            if not root.is_relative_to(attempt):
                raise ValueError('resume node is outside its declared attempt')
            shutdown = read(root/'completion.json')
            owner = read(root/'owner/completion.json')
            if not (shutdown['runtime_cleaned'] and shutdown['owned_process_groups_cleaned']
                    and owner['idle_requested'] and not owner['cleanup_errors']):
                raise ValueError('old collection writer cleanup is not verified')
            plan = read(root/'source-plan.json')
            if plan['source'] != source or plan['tool_budget'] != 100:
                raise ValueError('resume source/tool budget differs')
            for worker in plan['workers']:
                collection = Path(worker['collection'])
                input_path = Path(worker['input'])
                input_bytes = retry_estale(input_path.read_bytes, str(input_path))
                if hashlib.sha256(input_bytes).hexdigest() != plan['input_sha256'][str(input_path)]:
                    raise ValueError('lineage input changed')
                rows = json.loads(input_bytes)['cases']
                if len(rows) != len(worker['identities']):
                    raise ValueError('lineage input and identity population differ')
                for index, (identity, row) in enumerate(zip(worker['identities'], rows)):
                    key = tuple(map(str, identity))
                    if key != (source, row['scene'], str(row['task'])):
                        raise ValueError('lineage input identity differs')
                    case = collection/f'case_{index:03d}'
                    result_path = case/'result.json'
                    record = dict(identity=list(key), instruction=row['instruction'],
                                  case_path=str(case), collection=str(collection))
                    if retry_estale(case.exists, str(case)):
                        record['input_path'] = str(input_path)
                        record['input_sha256'] = plan['input_sha256'][str(input_path)]
                        archives.append(record)
                    if not retry_estale(result_path.exists, str(result_path)):
                        continue
                    payload = retry_estale(result_path.read_bytes, str(result_path))
                    result = json.loads(payload)
                    if (result['index'] != index or result['scene'] != row['scene']
                            or str(result['episode']) != str(row['task'])):
                        raise ValueError('lineage result identity differs')
                    if result['status'] == 'infrastructure_error':
                        continue
                    if result['status'] not in ('rollout_ended', 'rule_abstention'):
                        raise ValueError('unsupported completed status')
                    # Never turn a transport failure into a reusable end state.
                    if list(case.glob('candidates/decision_*/serving_error.json')):
                        continue
                    record.update(result_path=str(result_path),
                        result_sha256=hashlib.sha256(payload).hexdigest(),
                        status=result['status'], accepted_ah=result['accepted_ah'])
                    previous = completed.get(key)
                    if previous and previous['instruction'] != row['instruction']:
                        raise ValueError('same identity has different instructions')
                    completed[key] = record
    return dict(schema='v8_case_resume_catalog_v1', source=source, tool_budget=100,
        created_at=time.time(), old_writers_stopped=True,
        attempts=list(map(str, attempt_roots)), completed_cases=list(completed.values()),
        preserved_cases=archives, reused_completed=len(completed),
        partial_episode_policy='retain_old_labels_and_recollect_interrupted_episode',
        training_ready=False)


def v8_node_resume_catalog(previous, attempt, source, node):
    """Resume only a stopped node, without reading or stealing live peers."""
    if previous['source'] != source or previous['tool_budget'] != 100 or node not in range(4):
        raise ValueError('incompatible node resume lineage')
    current = v8_resume_catalog([attempt], source, node_indices=(node,))
    plan = read(Path(attempt)/source/f'node{node}/source-plan.json')
    identities = [key for worker in plan['workers'] for key in worker['identities']]
    scope = previous.get('scope_identities', identities)
    if not set(map(tuple, identities)).issubset(set(map(tuple, scope))):
        raise ValueError('node resume changed its original population')
    completed = {tuple(row['identity']): row for row in previous['completed_cases']}
    for row in current['completed_cases']:
        key = tuple(row['identity'])
        if key in completed and completed[key]['instruction'] != row['instruction']:
            raise ValueError('node resume instruction changed')
        completed[key] = row
    return dict(current, attempts=previous['attempts'],
        completed_cases=list(completed.values()), reused_completed=len(completed),
        preserved_cases=previous['preserved_cases']+current['preserved_cases'],
        node_attempts=previous.get('node_attempts', [])+[str(Path(attempt)/source/f'node{node}')],
        scope_node=node, scope_identities=scope)


def v8_lane_recoveries(root, source, node_indices=(0, 1, 2, 3)):
    directory=Path(root)/'platform/lane-recoveries'
    recoveries=[]
    for path in retry_estale(lambda:sorted(directory.glob(f'{source}-*.json')),str(directory)):
        value=read(path)
        if value['source']==source and value['rank'] in node_indices:
            recoveries.append(value)
    keys=[(r['rank'],r['worker']) for r in recoveries]
    if len(keys)!=len(set(keys)):
        raise ValueError('ambiguous_lane_recovery_winner')
    return recoveries


def v8_export_scope_drained(root, source, node_indices=(0, 1, 2, 3)):
    """A failed original lane is finished only after its explicit replacement."""
    recovered={}
    for value in v8_lane_recoveries(root,source,node_indices):
        path=Path(value['replacement_attempt'])/'completion.json'
        if not retry_estale(path.exists,str(path)):return False
        receipt=read(path)
        if receipt.get('status')!='complete' or not receipt.get('owned_process_groups_cleaned'):
            return False
        recovered.setdefault(value['rank'],set()).add(f"{source}-collector-worker{value['worker']}")
    for node in node_indices:
        status=read(Path(root)/source/f'node{node}/status.json')
        if status['status']=='complete':continue
        failures={name for name,code in status.get('results',{}).items() if code}
        if (status['status']!='collector_error' or not failures
                or not failures.issubset(recovered.get(node,set()))):return False
    return True


def completed_v8_export_cases(root, source, *, node_indices=(0, 1, 2, 3)):
    """Snapshot the continuation's winning ended cases, never unfinished writers.

    Return original archived identities and hashes. Historical partial prefixes
    remain on disk but cannot silently duplicate a recollected instruction.
    """
    root = Path(root)
    records = {}
    plans={node:read(root/source/f'node{node}/source-plan.json') for node in node_indices}
    for node in node_indices:
        path=root/'platform/resume'/f'{source}-node{node}-catalog.json'
        if not retry_estale(path.exists,str(path)):
            if plans[node].get('schema')!='rule_teacher_v8_git_collection_v1':
                raise ValueError('missing_legacy_resume_catalog:'+str(path))
            continue
        catalog = read(path)
        for row in catalog['completed_cases']:
            key = tuple(map(str, row['identity']))
            if key in records and records[key] != row:
                raise ValueError('conflicting_resume_winner:'+str(key))
            records[key] = row
    def add_plan(plan, replacement_of=None):
        for worker in plan['workers']:
            collection = Path(worker['collection'])
            summary = read(collection/'summary.json')
            for row in summary.get('cases', []):
                if row['status'] not in ('rollout_ended', 'rule_abstention'):
                    continue
                key = tuple(map(str, worker['identities'][row['index']]))
                if key != (source, str(row['scene']), str(row['episode'])):
                    raise ValueError('completed_export_source_identity_mismatch')
                path = collection/f"case_{row['index']:03d}"
                raw = retry_estale((path/'result.json').read_bytes, str(path/'result.json'))
                result = json.loads(raw)
                if any(result.get(k) != row.get(k) for k in ('index', 'scene', 'episode', 'status', 'accepted_ah')):
                    raise ValueError('completed_export_result_summary_mismatch')
                if (result.get('evaluation_valid') is False or result.get('infra_error')
                        or result.get('status')=='infrastructure_error'):
                    continue
                if key in records:
                    if not replacement_of or records[key]['collection']!=replacement_of:
                        raise ValueError('current_case_duplicates_resume_winner:'+str(key))
                    previous = records[key]
                    # A fresh lane may link its already completed cases to the
                    # stopped original writer. Keep the original export record
                    # byte-for-byte so existing completion receipts still bind.
                    if retry_estale(path.resolve, str(path)) == retry_estale(
                            Path(previous['case_path']).resolve, previous['case_path']):
                        if (previous['result_sha256'] != hashlib.sha256(raw).hexdigest()
                                or previous['input_sha256'] != plan['input_sha256'][worker['input']]):
                            raise ValueError('linked_completed_case_binding_changed:'+str(key))
                        continue
                    if records[key]['accepted_ah'] and not result['accepted_ah']:
                        continue
                records[key] = dict(identity=list(key), case_path=str(path),
                    collection=str(collection), input_path=worker['input'],
                    input_sha256=plan['input_sha256'][worker['input']],
                    result_path=str(path/'result.json'), result_sha256=hashlib.sha256(raw).hexdigest(),
                    status=row['status'], accepted_ah=row['accepted_ah'])
    for plan in plans.values():add_plan(plan)
    for recovery in v8_lane_recoveries(root,source,node_indices):
        plan=read(Path(recovery['replacement_attempt'])/'source-plan.json')
        original=plans[recovery['rank']]
        worker=next(w for w in original['workers'] if w['worker']==recovery['worker'])
        expected=set(map(tuple,worker['identities']))
        actual={tuple(key) for w in plan['workers'] for key in w['identities']}
        if actual!=expected or worker['collection']!=recovery['old_collection']:
            raise ValueError('lane_recovery_changed_original_scope')
        add_plan(plan,replacement_of=recovery['old_collection'])
    return [records[key] for key in sorted(records)]


def v8_pending_population(rows, source, catalog):
    if (catalog['schema'] != 'v8_case_resume_catalog_v1' or catalog['source'] != source
            or catalog['tool_budget'] != 100 or not catalog['old_writers_stopped']):
        raise ValueError('incompatible v8 resume catalog')
    originals = {(source, row['scene'], str(row['task_id'])): row for row in rows}
    reused = set()
    for record in catalog['completed_cases']:
        key = tuple(record['identity'])
        if key in reused or key not in originals:
            raise ValueError('duplicate or foreign reused identity')
        if originals[key]['instruction'] != record['instruction']:
            raise ValueError('reused instruction differs')
        path = Path(record['result_path'])
        payload = retry_estale(path.read_bytes, str(path))
        if hashlib.sha256(payload).hexdigest() != record['result_sha256']:
            raise ValueError('reused completed result changed')
        reused.add(key)
    scope = set(map(tuple, catalog.get('scope_identities', originals)))
    if not scope.issubset(originals):
        raise ValueError('node resume scope contains foreign instructions')
    return [row for key, row in originals.items() if key not in reused and key in scope]


def v8_finished_node_catalog(root, source, node):
    """Reuse a naturally drained node, including its explicit replacement lanes.

    This is only case-level continuation. Never use it to stop an active
    episode or reconstruct a Gym state. The caller also verifies local process
    identities before publishing a node's handoff-ready receipt.
    """
    root = Path(root)
    catalog = v8_resume_catalog([root], source, node_indices=(node,))
    plan = read(root/source/f'node{node}/source-plan.json')
    catalog.update(scope_node=node, scope_identities=[
        key for w in plan['workers'] for key in w['identities']])
    completed = {tuple(r['identity']): r for r in catalog['completed_cases']}
    recoveries = v8_lane_recoveries(root, source, (node,))
    for recovery in recoveries:
        replacement = Path(recovery['replacement_attempt'])
        if not replacement.is_relative_to(ART):
            raise ValueError('replacement lineage must stay on the authorized artifact disk')
        replacement_plan = read(replacement/'source-plan.json')
        old_worker = next(w for w in plan['workers'] if w['worker'] == recovery['worker'])
        expected = set(map(tuple, old_worker['identities']))
        actual = {tuple(key) for w in replacement_plan['workers'] for key in w['identities']}
        if actual != expected or old_worker['collection'] != recovery['old_collection']:
            raise ValueError('replacement changed original lane scope')
        # Reuse the original result/input/hash admission exactly, including
        # terminal rule abstentions. Export's stricter training admission must
        # not accidentally send these same rejected starts back to collection.
        current = v8_resume_catalog([replacement], source, node_indices=(node,),
                                    node_roots={node: replacement})
        catalog['preserved_cases'].extend(current['preserved_cases'])
        for record in current['completed_cases']:
            key = tuple(record['identity'])
            instruction = record['instruction']
            previous = completed.get(key)
            if previous and previous['instruction'] != instruction:
                raise ValueError('replacement instruction changed')
            if previous and previous['accepted_ah'] and not record['accepted_ah']:
                continue
            completed[key] = record
    catalog.update(completed_cases=list(completed.values()), reused_completed=len(completed),
                   explicit_lane_recoveries_included=bool(recoveries))
    return catalog


def v8_sage_resource_groups(resource_groups):
    """Bind an ordered selection of available four-node pools, including SAGE."""
    groups = tuple(resource_groups)
    known = ('sage', 'r2r', 'rxr')
    if not groups or groups[0] != 'sage' or groups != tuple(g for g in known if g in groups):
        raise ValueError('SAGE resource pools must be distinct, ordered, and include sage first')
    return groups


def v8_rebalance_sage_catalogs(rows, catalogs, *, resource_groups=('sage', 'r2r', 'rxr'), seed=20261003):
    """Assign only stopped SAGE gaps once across selected four-node resource pools.

    Global completion lineage stays separate from each pending-only scope so
    starters do not reread or recollect all completed instructions.
    The local rank remains 0..3 inside each pool; identities remain SAGE.
    """
    from prepare_rule_teacher_formal_local16_collection import balance_scenes, identity
    resource_groups = v8_sage_resource_groups(resource_groups)
    if len(catalogs) != 4 or sorted(c.get('scope_node', -1) for c in catalogs) != list(range(4)):
        raise ValueError('SAGE redistribution requires four original nodes')
    completed, preserved, scope, attempts = [], [], set(), []
    for catalog in catalogs:
        if (catalog.get('schema') != 'v8_case_resume_catalog_v1' or catalog.get('source') != 'sage'
                or catalog.get('tool_budget') != 100 or catalog.get('old_writers_stopped') is not True):
            raise ValueError('active or incompatible SAGE lineage cannot be redistributed')
        selected = set(map(tuple, catalog['scope_identities']))
        if scope & selected:
            raise ValueError('overlapping original SAGE node scopes')
        if any(tuple(r['identity']) not in selected for r in catalog['completed_cases']):
            raise ValueError('completed SAGE identity is outside its original node scope')
        scope.update(selected)
        completed.extend(catalog['completed_cases'])
        preserved.extend(catalog['preserved_cases'])
        attempts.extend(p for p in catalog['attempts'] if p not in attempts)
    combined = dict(schema='v8_case_resume_catalog_v1', source='sage', tool_budget=100,
        old_writers_stopped=True, created_at=time.time(), attempts=attempts,
        completed_cases=completed, preserved_cases=preserved, scope_identities=sorted(scope),
        reused_completed=len(completed), training_ready=False)
    pending = v8_pending_population(rows, 'sage', combined)
    expected = {identity(r) for r in pending}
    slots = [(group, node) for node in range(4) for group in resource_groups]
    scene_count = len({r['scene'] for r in pending})
    count = min(len(slots), scene_count)
    buckets = balance_scenes(pending, seed, workers=count) if count else []
    buckets += [[] for _ in range(len(slots)-len(buckets))]
    seen, assignments = set(), []
    for (group, node), bucket in zip(slots, buckets):
        keys = {identity(r) for r in bucket}
        if seen & keys:
            raise ValueError('duplicate SAGE redistribution assignment')
        seen.update(keys)
        records = [r for r in preserved if tuple(r['identity']) in keys]
        catalog = dict(schema='v8_case_resume_catalog_v1', source='sage', tool_budget=100,
            old_writers_stopped=True, created_at=combined['created_at'], attempts=attempts,
            scope_node=node, scope_group=group, scope_identities=sorted(keys),
            completed_cases=[], reused_completed=0, preserved_cases=records, training_ready=False,
            partial_episode_policy=combined.get('partial_episode_policy',
                'retain_old_labels_and_recollect_interrupted_episode'))
        assignments.append(dict(resource_group=group, node=node, count=len(bucket), catalog=catalog))
    if seen != expected:
        raise ValueError('SAGE redistribution lost original pending identities')
    sizes = [a['count'] for a in assignments]
    return dict(schema='v8_sage_three_pool_pending_partition_v1', source='sage', seed=seed,
        original_scope=len(scope), reused_completed=len(completed), pending=len(pending),
        resource_groups=list(resource_groups), resource_slots=len(slots), whole_scenes_preserved=True,
        min_slot_cases=min(sizes), max_slot_cases=max(sizes),
        completed_catalog=combined, assignments=assignments, training_ready=False)


def read(path):
    return json.loads(retry_estale(Path(path).read_text, str(path)))


def prepare(config_path, roots_path, shutdown_paths, owner_shutdown_paths=()):
    config = read(config_path); root = Path(config['output_root'])
    sources = selected_sources(config)
    if len(sources) != 1 or not root.is_relative_to(ART):
        raise ValueError('one original source in a fresh authorized attempt required')
    source = sources[0]
    roots = read(roots_path)
    if set(roots) != set(map(str, range(4))):
        raise ValueError('explicit old lineage required for all four nodes')
    if not shutdown_paths and not owner_shutdown_paths:
        raise ValueError('verified old writer shutdown is required')
    if owner_shutdown_paths:
        verified_nodes = set()
        for path in owner_shutdown_paths:
            proof = read(path)
            node = proof.get('node')
            if (proof.get('schema') != 'rule_teacher_owner_shutdown_v1'
                    or type(node) is not int or node not in range(4) or node in verified_nodes
                    or proof.get('source') != source
                    or proof.get('old_root') != roots[str(node)]
                    or proof.get('remaining_processes') != []
                    or proof.get('controller_alive') is not True
                    or proof.get('idle_restored') is not True
                    or proof.get('owner_stopped') is not True
                    or proof.get('pipeline_stopped') is not True):
                raise ValueError('invalid live-workspace owner shutdown proof')
            verified_nodes.add(node)
        if verified_nodes != set(range(4)):
            raise ValueError('owner shutdown proofs must cover all four nodes')
    for proof in shutdown_paths:
        envelope = read(proof)
        if envelope['http_status'] != 200 or envelope['response']['code'] != 2000:
            raise ValueError('shutdown proof platform query failed')
        task = envelope['response']['data']
        if task['createdBy'] != 'anonymous' or task['quotaId'] != 'quota-anonymous':
            raise ValueError('old lineage task identity differs')
        if task['status'] not in ('Stopped', 'Failed', 'Succeeded', 'Completed', 'Cancelled'):
            raise ValueError('old source task is still active')
    ledger = root/'platform/completed-resume-lineage.json'
    if ledger.exists() or any((root/f'node{n}/node-owner/owner-start.json').exists() for n in range(4)):
        raise ValueError('resume may only prepare an unlaunched fresh attempt once')
    records, counts, pending = [], Counter(), Counter()
    for node in range(4):
        old = Path(roots[str(node)]).resolve(strict=True)
        if not old.is_relative_to(ART) or old == root:
            raise ValueError('old lineage must be a different authorized attempt')
        for worker in range(8):
            relative = Path(f'node{node}/collection/{source}/worker{worker}')
            before, after = old/relative, root/relative
            old_inputs, new_inputs = read(before/'inputs.json'), read(after/'inputs.json')
            if old_inputs['cases'] != new_inputs['cases']:
                raise ValueError(f'original input rows changed for {node}/{worker}')
            destination = after/f'collections/{source}/shard_000'
            destination.mkdir(parents=True, exist_ok=True)
            for index, row in enumerate(new_inputs['cases']):
                original = before/f'collections/{source}/shard_000/case_{index:03d}'
                path = original/'result.json'
                if not retry_estale(path.exists, str(path)):
                    pending['unfinished'] += 1; continue
                payload = retry_estale(path.read_bytes, str(path))
                result = json.loads(payload)
                if result['index'] != index or result['scene'] != row['scene'] or str(result['episode']) != str(row['task']):
                    raise ValueError('result identity does not match the original case')
                status = result['status']
                if status == 'infrastructure_error':
                    pending['infrastructure_error'] += 1; continue
                if status not in ('rollout_ended', 'rule_abstention'):
                    raise ValueError('unsupported completed result status')
                target = destination/f'case_{index:03d}'
                if target.exists() or target.is_symlink():
                    raise ValueError('new case is not empty; refusing overwrite')
                counts[status] += 1
                records.append(dict(node=node,worker=worker,index=index,scene=row['scene'],task=str(row['task']),
                    old_result=str(path.resolve(strict=True)),new_case=str(target),status=status,
                    result_sha256=hashlib.sha256(payload).hexdigest(),accepted_ah=result['accepted_ah']))
    # Verify the complete selected lineage before creating its first case link.
    for record in records:
        target = Path(record['new_case'])
        target.symlink_to(Path(record['old_result']).parent, target_is_directory=True)
    report=dict(schema='rule_teacher_completed_case_resume_v1',source=source,new_root=str(root),
        original_config=str(config_path),old_roots=roots,shutdown_proofs=list(map(str,shutdown_paths)),
        owner_shutdown_proofs=list(map(str,owner_shutdown_paths)),
        completed_cases=records,reused_counts=dict(counts),pending_counts=dict(pending),
        reused=len(records),pending=sum(pending.values()),training_ready=False)
    with ledger.open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='completed_cases'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--old-roots',type=Path,required=True)
    p.add_argument('--shutdown-proof',type=Path,action='append',default=[])
    p.add_argument('--owner-shutdown-proof',type=Path,action='append',default=[])
    args=p.parse_args()
    prepare(args.config,args.old_roots,args.shutdown_proof,args.owner_shutdown_proof)
