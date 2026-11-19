"""Seed a fresh companion attempt from audited results without editing its source.

This does not resume an owner or copy terminal/launch markers.  The normal suite
owns scheduling, service cleanup, retries, barriers, scoring and GPU reservation.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/train'))
from rule_teacher_training_handoff import load, save, sha
from dlc_control_io_retry import retry_estale
from action_head_benchmark_suite import digest, partition, stage_root, verify, GENERAL


def _exists(path):
    return retry_estale(lambda: Path(path).exists(), f'exists:{path}')


def _glob(path, pattern):
    return retry_estale(lambda: sorted(Path(path).glob(pattern)), f'glob:{path}/{pattern}')


def _normalize(value, plan):
    """Only relocate frozen attempt paths; model/data paths remain identities."""
    base = Path(plan['runtime_repo']).parent.parent
    if isinstance(value, str):
        return '$ATTEMPT' + value[len(str(base)):] if value.startswith(str(base) + '/') else value
    if isinstance(value, list):
        return [_normalize(v, plan) for v in value]
    if isinstance(value, dict):
        return {_normalize(k, plan): _normalize(v, plan) for k, v in value.items()}
    return value


def _code_changes(new, old):
    """Record source changes for provenance; never gate an authorized recovery."""
    changes = []
    for section in ('source_sha256', 'code_sha256'):
        a = _attempt_normalize(old.get(section, {}), old)
        b = _attempt_normalize(new.get(section, {}), new)
        for path in sorted(a.keys() | b.keys()):
            if a.get(path) != b.get(path):
                changes.append(dict(section=section, path=path,
                                    old_sha256=a.get(path), new_sha256=b.get(path)))
    return changes


def compatible_plans(new, old, allowed_code_changes=(), rerun_benchmarks=(), navigation_only=False):
    """Legacy API: provenance only; no config/source compatibility gate."""
    if Path(new['output']).resolve() == Path(old['output']).resolve():
        raise ValueError('repair must use a fresh output, never the original attempt')
    return _code_changes(new, old)


def seed_checkpoint(new_plan, old_plan_path, *, allowed_code_changes=(), rerun_benchmarks=(), navigation_only=False, dry_run=False):
    """Copy valid cases/node receipts into a new plan; failed work stays pending.

    Run before any rank starts the new suite.  A completed seed can be called
    again only for the same two plan hashes.  An interrupted seed is deliberately
    not replayed blindly: its intent and partial copies remain for inspection.
    """
    old_plan_path = Path(old_plan_path)
    output = Path(new_plan['output'])
    report_path = output / 'reuse-audit.json'
    intent = output / 'repair-seed-intent.json'
    if not _exists(old_plan_path):
        report = dict(schema='rule_teacher_companion_repair_seed_v1',
                      old_plan=str(old_plan_path), old_plan_sha256=None,
                      new_plan_sha256=new_plan['plan_sha256'], reused_nodes=0,
                      reused_cases=0, previous_checkpoint_missing=True, dry_run=dry_run,
                      nodes=[], files=[])
        if not dry_run:
            save(report_path, report)
        return report
    old = load(old_plan_path)
    if _exists(report_path):
        report = load(report_path)
        if report['old_plan_sha256'] != old['plan_sha256'] or report['new_plan_sha256'] != new_plan['plan_sha256']:
            raise ValueError('existing seed belongs to different plans')
        return report
    if _exists(intent):
        raise ValueError('interrupted repair seed requires inspection; do not overwrite partial copies')
    changes = compatible_plans(new_plan, old, allowed_code_changes, rerun_benchmarks, navigation_only)
    operations, nodes, rebound_contracts = [], [], []

    def add(source, destination, replacement=None):
        if _exists(destination):
            raise ValueError('repair destination already exists: ' + str(destination))
        operations.append(dict(source=str(source), destination=str(destination),
                               source_sha256=sha(source), replacement=replacement))

    for before, stage in zip(old['stages'], new_plan['stages']):
        old_manifest, manifest = load(before['manifest']), load(stage['manifest'])
        if old_manifest != manifest:
            raise ValueError('repair changed exact benchmark manifest')
        for rank in range(new_plan['config']['nodes']):
            source = stage_root(old, before) / f'node_{rank}'
            dest = stage_root(new_plan, stage) / f'node_{rank}'
            cases = partition(manifest['cases'], rank, new_plan['config']['nodes'])
            expected = {c['logical_id']: c for c in cases}
            receipt_path = source / 'result.json'
            receipt = load(receipt_path) if _exists(receipt_path) else {}
            if receipt and (receipt.get('plan_sha256') != old['plan_sha256'] or
                            receipt.get('stage') != before['index'] or receipt.get('rank') != rank):
                raise ValueError('old node receipt identity differs')
            if receipt.get('evaluation_valid') and receipt.get('logical_ids') != list(expected):
                raise ValueError('old node receipt has incomplete/changed case identities')
            # Services must have relinquished ownership even when collection failed.
            launch = source / 'launch/plan.json'
            if _exists(launch):
                completion = source / 'launch/completion.json'
                if not _exists(completion) or not load(completion).get('owned_process_groups_cleaned'):
                    raise ValueError('old services have not released ownership: ' + str(source))
            if stage['benchmark'] in set(rerun_benchmarks):
                old_ckpt = next(c for c in old['config']['checkpoints'] if c['name'] == before['checkpoint'])
                new_ckpt = next(c for c in new_plan['config']['checkpoints'] if c['name'] == stage['checkpoint'])
                nodes.append(dict(stage=stage['index'], benchmark=stage['benchmark'], rank=rank,
                    reused_node=False, reused_cases=0, expected=len(cases), pending_cases=len(cases),
                    full_rerun=True, reason='explicit camera-contract correction; original cases retained',
                    old_config_observation_profile=old['config'].get('observation_profiles', {}).get(stage['benchmark']),
                    new_config_observation_profile=new_plan['config'].get('observation_profiles', {}).get(stage['benchmark']),
                    old_observation_profile=old_ckpt.get('observation_profiles', {}).get(stage['benchmark']),
                    new_observation_profile=new_ckpt.get('observation_profiles', {}).get(stage['benchmark'])))
                continue
            result = copy.deepcopy(receipt.get('result') or {})
            benchmark, valid_ids = stage['benchmark'], set()
            if benchmark in GENERAL:
                if receipt.get('evaluation_valid'):
                    prediction = Path(result['predictions'])
                    if sha(prediction) != result['predictions_sha256'] or result.get('completed') != len(cases):
                        raise ValueError('general prediction receipt/hash/count differs')
                    if not prediction.resolve().is_relative_to(source.resolve()):
                        raise ValueError('general prediction escapes its original node')
                    target = dest / prediction.relative_to(source)
                    add(prediction, target)
                    result['predictions'] = str(target)
                    valid_ids = set(expected)
            else:
                ckpt = next(c for c in old['config']['checkpoints'] if c['name'] == before['checkpoint'])
                node_cases = source / 'cases.json'
                if _exists(node_cases):
                    if load(node_cases)['cases'] != cases:
                        raise ValueError('old node cases differ from frozen partition')
                    add(node_cases, dest / 'cases.json')
                native = benchmark in ('r2r', 'rxr')
                contract = new_contract = None
                if native:
                    contract_path = source / 'evaluation' / benchmark / 'contract.json'
                    if _exists(contract_path):
                        contract = load(contract_path)
                        if (contract.get('dataset') != benchmark or contract.get('case_pool_sha256') != digest(cases)
                                or contract.get('suite_cases_sha256') != sha(node_cases)
                                or contract.get('checkpoint', {}).get('action_head_dir') != ckpt['identity']['action_head_dir']):
                            raise ValueError('native contract differs from frozen cases/checkpoint')
                        # Validate the original contract before explicitly rebinding
                        # only reviewed source changes in the new attempt.
                        new_contract = copy.deepcopy(contract)
                        if contract.get('eval_set'):
                            old_eval_set = Path(contract['eval_set'])
                            old_runtime = Path(old['runtime_repo'])
                            if old_eval_set.is_relative_to(old_runtime):
                                relative_set = old_eval_set.relative_to(old_runtime)
                            elif not old['config'].get('vlnce_eval_set') and old_eval_set.parts[-4:] == (
                                    'vln_demo','batch','eval_sets','english-val-unseen-v1'):
                                # A prior scope-only copy may still point to its
                                # parent package. Rebind only the canonical set,
                                # validating both immutable locks and all source
                                # files below; never rewrite a failed old receipt.
                                relative_set = Path('vln_demo/batch/eval_sets/english-val-unseen-v1')
                            else:
                                relative_set = None
                            if relative_set is not None:
                                new_eval_set = Path(new_plan['runtime_repo']) / relative_set
                                new_contract['eval_set'] = str(new_eval_set.resolve())
                                new_contract['manifest_lock_sha256'] = sha(new_eval_set/'lock.json')
                        for name, expected_sha in contract.get('source_sha256', {}).items():
                            actual = sha(Path(new_plan['runtime_repo']) / name)
                            new_contract['source_sha256'][name] = actual
                        if new_contract != contract:
                            rebound_contracts.append(dict(stage=stage['index'], rank=rank,
                                source=str(contract_path), old_contract_sha256=digest(contract),
                                new_contract_sha256=digest(new_contract), old_contract=contract,
                                new_contract=new_contract,
                                reason='reviewed repair source changes; valid old cases only; case/config/model unchanged'))
                        add(contract_path, dest / contract_path.relative_to(source), new_contract)
                pattern = ('evaluation/' + benchmark + '/rollouts/**/*.json' if native
                           else 'evaluation/worker_*/rollouts/**/*.json')
                for path in _glob(source, pattern):
                    row = load(path)
                    if not row.get('evaluation_valid') or row.get('infra_error'):
                        continue
                    case = expected.get(row.get('logical_id'))
                    if not case or row.get('case_sha256') != digest(case) or row.get('repeat_index') != 1:
                        raise ValueError('saved valid row has wrong case identity: ' + str(path))
                    if case['logical_id'] in valid_ids:
                        raise ValueError('duplicate saved case: ' + case['logical_id'])
                    replacement = None
                    if native:
                        if not contract or row.get('contract_sha256') != digest(contract) or row.get('dataset') != benchmark:
                            raise ValueError('saved native row has wrong contract')
                        if new_contract != contract:
                            replacement = {'contract_sha256': digest(new_contract)}
                    else:
                        if row.get('suite_plan_sha256') != old['plan_sha256'] or row.get('checkpoint') != ckpt['identity']:
                            raise ValueError('saved SAGE row has wrong plan/checkpoint')
                        replacement = {'suite_plan_sha256': new_plan['plan_sha256']}
                    add(path, dest / path.relative_to(source), replacement)
                    valid_ids.add(case['logical_id'])
            reused = bool(receipt.get('evaluation_valid') and valid_ids == set(expected))
            if receipt.get('evaluation_valid') and not reused:
                raise ValueError('valid node receipt lacks complete validated outputs')
            if reused:
                if benchmark in ('r2r', 'rxr') and new_contract is not None:
                    result['contract_sha256'] = digest(new_contract)
                rebound = {**receipt, 'plan_sha256': new_plan['plan_sha256'], 'result': result,
                           'repair_source': dict(path=str(receipt_path), sha256=sha(receipt_path))}
                add(receipt_path, dest / 'result.json', rebound)
            nodes.append(dict(stage=stage['index'], benchmark=benchmark, rank=rank,
                              reused_node=reused, reused_cases=len(valid_ids), expected=len(cases),
                              pending_cases=len(cases) - len(valid_ids)))
    # No old launch/start/result-failure/aggregate is copied.  The fresh suite
    # recomputes aggregate validity from the full expected population.
    report = dict(schema='rule_teacher_companion_repair_seed_v1',
                  old_plan=str(old_plan_path), old_plan_sha256=old['plan_sha256'],
                  new_plan_sha256=new_plan['plan_sha256'], code_changes=changes,
                  allowed_code_changes=sorted(allowed_code_changes), nodes=nodes,
                  rerun_benchmarks=sorted(rerun_benchmarks),
                  rebound_contracts=rebound_contracts,
                  reused_nodes=sum(n['reused_node'] for n in nodes),
                  reused_cases=sum(n['reused_cases'] for n in nodes),
                  copied_files=len(operations), dry_run=dry_run, files=operations)
    if dry_run:
        return report
    save(intent, dict(old_plan_sha256=old['plan_sha256'], new_plan_sha256=new_plan['plan_sha256']))
    for operation in operations:
        source, dest = Path(operation['source']), Path(operation['destination'])
        if sha(source) != operation['source_sha256']:
            raise ValueError('source changed after reuse audit: ' + str(source))
        replacement = operation['replacement']
        if replacement is not None:
            value = load(source)
            value.update(replacement)
            save(dest, value)
        else:
            content = retry_estale(source.read_bytes, f'read:{source}')
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as stream:
                stream.write(content)
        operation['destination_sha256'] = sha(dest)
    save(report_path, report)
    return report


def _attempt_normalize(value, plan):
    """Relocate only one container checkout/dependency root, never data paths."""
    base = str(Path(plan['runtime_repo']).parent)
    if isinstance(value, str):
        return '$CHECKOUT/' + value[len(base)+1:] if value.startswith(base+'/') else value
    if isinstance(value, list):
        return [_attempt_normalize(v, plan) for v in value]
    if isinstance(value, dict):
        return {_attempt_normalize(k, plan): _attempt_normalize(v, plan) for k,v in value.items()}
    return value


def compatible_subset_plans(new, old, allowed_code_changes=(), *, benchmarks=None):
    """Authorized reuse records versions without config/hash/allowlist gates.

    Case selection, valid-result accounting and fresh-output ownership remain
    enforced by the copier. Source changes do not require updating a validator.
    """
    if Path(new['output']).resolve() == Path(old['output']).resolve():
        raise ValueError('completed-case reuse requires a fresh output')
    return _code_changes(new, old)


def _native_subset_contract(contract, old, new, benchmark, cases, manifest, *, allowed_code_changes=()):
    """Reuse native contract rebinding, changing only validated shard identities."""
    value = copy.deepcopy(contract)
    if contract['dataset'] != benchmark:
        raise ValueError('native completed dataset changed')
    before = Path(old['runtime_repo'])
    after = Path(new['runtime_repo'])
    target_set = Path(new['config'].get('vlnce_eval_set', after/'vln_demo/batch/eval_sets/english-val-unseen-v1')).resolve()
    value['eval_set'] = str(target_set)
    value['manifest_lock_sha256'] = sha(target_set/'lock.json')
    for name, expected in contract['source_sha256'].items():
        actual = sha(after/name)
        value['source_sha256'][name] = actual
    value.update(expected_episodes=len(cases), case_pool_sha256=digest(cases),
        suite_cases_sha256=hashlib.sha256(
            (json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()).hexdigest())
    return value


def _inherited_seed_paths(plan, stage, rank, directory):
    """Prove a CPU-restored shard never started services and is still audited."""
    receipt_path = directory/'result.json'
    if not _exists(receipt_path):
        return None
    receipt = load(receipt_path)
    if not receipt.get('no_services_started'):
        return None
    report_path = Path(plan['output'])/'reuse-audit.json'
    if (_exists(directory/'launch') or receipt.get('plan_sha256') != plan['plan_sha256']
            or receipt.get('stage') != stage['index'] or receipt.get('rank') != rank
            or receipt.get('evaluation_valid') is not True or receipt.get('exit_code') != 0
            or receipt.get('completed_case_reuse') != dict(path=str(report_path), sha256=sha(report_path))):
        raise ValueError('inherited completed shard identity or ownership changed')
    report = load(report_path)
    cases = partition(load(stage['manifest'])['cases'], rank, plan['config']['nodes'])
    from closedloop_evaluation_contract import stage_repeats
    expected = len(cases)*stage_repeats(plan['config'])
    if (receipt.get('logical_ids') != [case['logical_id'] for case in cases]
            or receipt.get('selected') != len(cases)
            or receipt.get('result', {}).get('completed') != expected
            or receipt.get('result', {}).get('reused_completed_attempts') != expected):
        raise ValueError('inherited completed shard population differs')
    if (report.get('schema') not in ('unified_completed_subset_seed_v1', 'unified_completed_sources_seed_v2')
            or report.get('new_plan_sha256') != plan['plan_sha256']
            or not report.get('source_writers_cleaned') or not report.get('original_artifacts_preserved')):
        raise ValueError('inherited completed shard audit differs')
    operations = [op for op in report['files'] if Path(op['destination']).is_relative_to(directory)]
    if not operations:
        raise ValueError('inherited completed shard has no audited files')
    for op in operations:
        if sha(op['source']) != op['source_sha256'] or sha(op['destination']) != op['destination_sha256']:
            raise ValueError('inherited completed shard source or destination changed')
    return {op['destination'] for op in operations}


def seed_subset_checkpoint(new_plan, reference, *, dry_run=False):
    """Repartition complete, same-checkpoint K4 rows; preserve source artifacts.

    Uses the existing collector resume checks. No start/completion/node result
    is copied; the fresh suite recomputes every node and global denominator.
    """
    from closedloop_evaluation_contract import stage_repeats
    from action_head_suite_stage import subset_manifest
    old_path = Path(reference['path'])
    if sha(old_path) != reference['sha256']:
        raise ValueError('completed source plan hash changed')
    old = load(old_path)
    if old['source_provenance']['git_runtime']['commit'] != reference['commit']:
        raise ValueError('completed source commit differs')
    output = Path(new_plan['output'])
    report_path, intent = output/'reuse-audit.json', output/'repair-seed-intent.json'
    if _exists(report_path):
        report = load(report_path)
        if report['new_plan_sha256'] != new_plan['plan_sha256'] or report['source_plan'] != reference:
            raise ValueError('existing completed seed belongs to different plans')
        for operation in report['files']:
            if (sha(operation['source']) != operation['source_sha256']
                    or sha(operation['destination']) != operation['destination_sha256']):
                raise ValueError('completed seed source or destination changed')
        return report
    if _exists(intent):
        raise ValueError('interrupted completed seed requires inspection')
    selected = reference.get('benchmarks')
    if selected is not None:
        from closedloop_evaluation_contract import BENCHMARKS
        if not selected or selected != [b for b in BENCHMARKS if b in selected]:
            raise ValueError('reuse benchmarks must be distinct and in canonical order')
    changes = compatible_subset_plans(new_plan, old, reference.get('allowed_code_changes', ()), benchmarks=selected)
    stages = {s['benchmark']:s for s in new_plan['stages']}
    if (not set(selected or stages).issubset(stages)
            or not set(selected or stages).issubset({s['benchmark'] for s in old['stages']})
            or (selected is None and set(stages) != {s['benchmark'] for s in old['stages']})):
        raise ValueError('completed source benchmark set differs')
    operations, seen, node_counts, skipped_shards, skipped_attempts = [], set(), {}, [], []
    repeats = stage_repeats(new_plan['config'])
    def add(source, destination, replacement):
        if _exists(destination):
            raise ValueError('completed seed destination exists: '+str(destination))
        operations.append(dict(source=str(source), source_sha256=sha(source),
            destination=str(destination), replacement=replacement))
    for prior in old['stages']:
        benchmark = prior['benchmark']
        if selected is not None and benchmark not in selected:
            continue
        stage = stages[benchmark]
        if benchmark not in ('sage-3d','r2r','rxr') or prior['checkpoint'] != stage['checkpoint']:
            raise ValueError('unsupported or changed completed checkpoint stage')
        old_manifest, manifest = load(prior['manifest']), load(stage['manifest'])
        lookup = {c['logical_id']:c for c in manifest['cases']}
        if len(lookup) != len(manifest['cases']) or len({c['logical_id'] for c in old_manifest['cases']}) != len(old_manifest['cases']):
            raise ValueError('duplicate manifest identity')
        if any(lookup.get(c['logical_id']) != c for c in old_manifest['cases']):
            raise ValueError('completed source cases are not an exact target subset')
        owners = {c['logical_id']:rank for rank in range(new_plan['config']['nodes'])
                  for c in partition(manifest['cases'],rank,new_plan['config']['nodes'])}
        ckpt = next(c for c in old['config']['checkpoints'] if c['name']==prior['checkpoint'])
        contracts = {}
        for rank in range(new_plan['config']['nodes']):
            cases = partition(manifest['cases'],rank,new_plan['config']['nodes'])
            contracts[rank] = (cases,subset_manifest(manifest,cases))
        for rank in range(old['config']['nodes']):
            source = stage_root(old,prior)/f'node_{rank}'
            completion_path=source/'launch/completion.json'
            inherited_paths = None
            if not _exists(completion_path):
                inherited_paths = _inherited_seed_paths(old, prior, rank, source)
                if inherited_paths is None:
                    if reference.get('skip_unfinished_shards'):
                        continue
                    raise ValueError('completed source writer has no cleanup or inherited seed receipt')
            completion = load(completion_path) if inherited_paths is None else None
            if completion is not None and (not completion.get('owned_process_groups_cleaned') or not completion.get('runtime_cleaned')):
                # A cleanup timeout remains immutable. A later ownership-only
                # audit can establish that its writers finally exited, so valid
                # cases survive without pretending the failed stage succeeded.
                release_path=reference.get('cleanup_recoveries',{}).get(str(completion_path))
                release=load(release_path) if release_path else {}
                released=(release.get('schema')=='owned_service_cleanup_release_v1'
                    and release.get('source_completion')==str(completion_path)
                    and release.get('owned_process_groups_cleaned') is True
                    and release.get('runtime_cleaned') is True
                    and release.get('native_workspaces_cleaned') is True
                    and release.get('evaluation_success_claimed') is False)
                if not released:
                    if reference.get('skip_unfinished_shards'):
                        continue  # Never read a case while its previous writer owns it.
                    raise ValueError('completed source writer has not released ownership')
            local_cases = partition(old_manifest['cases'],rank,old['config']['nodes'])
            if not _exists(source/'cases.json') and inherited_paths is not None and benchmark == 'sage-3d':
                pass  # CPU-restored SAGE identity is bound by its receipt/manifest.
            elif load(source/'cases.json')['cases'] != local_cases:
                raise ValueError('source node partition changed')
            native = benchmark in ('r2r','rxr')
            pattern = f'evaluation/{benchmark}/**/rollouts/*.json' if native else 'evaluation/worker_*/rollouts/**/*.json'
            paths = _glob(source,pattern)
            contract_path = source/'evaluation'/benchmark/'contract.json'
            if native and not _exists(contract_path):
                # A service can fail during startup, after cleanup was recorded
                # but before it published a contract or any usable attempt.
                valid_attempts = any((row := load(path)).get('evaluation_valid')
                                     and not row.get('infra_error') for path in paths)
                if reference.get('skip_unfinished_shards') and inherited_paths is None and not valid_attempts:
                    skipped_shards.append(dict(benchmark=benchmark,rank=rank,
                        reason='cleaned_empty_shard_without_native_contract'))
                    continue
                raise ValueError('native attempts require the original source contract: '+str(contract_path))
            contract = load(contract_path) if native else None
            if native and (contract['case_pool_sha256'] != digest(local_cases)
                    or contract['suite_cases_sha256'] != sha(source/'cases.json')
                    or contract['max_tool_calls'] != new_plan['config']['max_tool_calls'][benchmark]
                    or contract.get('repeats',1) != repeats
                    or contract['checkpoint']['action_head_dir'] != ckpt['identity']['action_head_dir']):
                raise ValueError('source native contract differs')
            expected = {c['logical_id']:c for c in local_cases}
            for path in paths:
                if inherited_paths is not None and str(path) not in inherited_paths:
                    raise ValueError('inherited completed shard contains an unaudited attempt')
                row = load(path)
                if not row.get('evaluation_valid') or row.get('infra_error'):
                    continue
                case = expected.get(row.get('logical_id'))
                repeat = row.get('repeat_index')
                key = (benchmark,row.get('logical_id'),repeat)
                # Historical SAGE collectors published raw rows before a
                # batch-level suite stamp. An interrupted cleaned shard can
                # therefore contain a valid navigation outcome that is NOT
                # a reusable suite result. Preserve it and re-evaluate it.
                # Never excuse a mismatching/partially present identity or
                # an inherited row whose audit already promised finality.
                if (not native and inherited_paths is None
                        and reference.get('skip_unfinished_shards')
                        and case and 'case_sha256' not in row and 'suite_plan_sha256' not in row
                        and row.get('checkpoint') == ckpt['identity']
                        and all(row.get(k) == case.get(k) for k in ('scene','episode_id','instruction'))
                        and type(repeat) is int and 1 <= repeat <= repeats
                        and row.get('repeats_per_case') == repeats
                        and type(row.get('calls_used')) is int
                        and 0 <= row['calls_used'] <= new_plan['config']['max_tool_calls'][benchmark]):
                    skipped_attempts.append(dict(path=str(path), sha256=sha(path),
                        benchmark=benchmark, rank=rank, logical_id=case['logical_id'],
                        repeat_index=repeat, reason='cleaned_sage_attempt_not_suite_finalized'))
                    continue
                if (not case or row.get('case_sha256') != digest(case) or type(repeat) is not int
                        or not 1<=repeat<=repeats or row.get('repeats_per_case') != repeats or key in seen
                        or type(row.get('calls_used')) is not int
                        or not 0<=row['calls_used']<=new_plan['config']['max_tool_calls'][benchmark]):
                    raise ValueError('completed case/repeat/budget identity differs: '+str(path))
                if any(row.get(k) != case.get(k) for k in ('scene','episode_id','instruction')):
                    raise ValueError('completed source instruction/reset differs')
                archive = row.get('canonical_context_archive')
                if archive and sha(archive['path']) != archive['sha256']:
                    raise ValueError('completed canonical history changed')
                if any(a.get('exact_parent_prefix') is not True for a in row.get('continuous_context_audit',[])):
                    raise ValueError('completed native prefix audit failed')
                target_rank = owners[case['logical_id']]
                dest = stage_root(new_plan,stage)/f'node_{target_rank}'
                replacement = dict(reuse_source=dict(path=str(path),sha256=sha(path),
                    plan_sha256=old['plan_sha256'],rank=rank,repeat_index=repeat))
                if native:
                    from v6_vlnce_eval import resumable
                    if not resumable(row,case,digest(contract),repeat):
                        raise ValueError('source native row is not resumable')
                    target_contract = _native_subset_contract(contract,old,new_plan,benchmark,*contracts[target_rank],
                        allowed_code_changes=reference.get('allowed_code_changes', ()))
                    replacement['contract_sha256'] = digest(target_contract)
                    target_contract_path = dest/'evaluation'/benchmark/'contract.json'
                    previous = node_counts.get((benchmark,target_rank))
                    if previous and previous['contract'] != target_contract:
                        raise ValueError('source native contracts disagree')
                    if previous is None:
                        add(source/'evaluation'/benchmark/'contract.json',target_contract_path,target_contract)
                        add(stage['manifest'],dest/'cases.json',contracts[target_rank][1])
                    node_counts.setdefault((benchmark,target_rank),dict(contract=target_contract,reused=0))['reused']+=1
                else:
                    if row.get('suite_plan_sha256') != old['plan_sha256'] or row.get('checkpoint') != ckpt['identity']:
                        raise ValueError('completed SAGE plan/checkpoint differs')
                    replacement.update(suite_plan_sha256=new_plan['plan_sha256'],suite_rank=target_rank)
                    node_counts.setdefault((benchmark,target_rank),dict(reused=0))['reused']+=1
                add(path,dest/path.relative_to(source),replacement)
                seen.add(key)
    report = dict(schema='unified_completed_subset_seed_v1',source_plan=reference,
        old_plan_sha256=old['plan_sha256'],new_plan_sha256=new_plan['plan_sha256'],
        code_changes=changes,reused_nodes=0,reused_cases=len(seen),
        skipped_shards=skipped_shards, skipped_attempts=skipped_attempts,
        by_stage_node=[dict(benchmark=b,rank=r,reused_attempts=v['reused']) for (b,r),v in node_counts.items()],
        source_writers_cleaned=True,original_artifacts_preserved=True,files=operations,dry_run=dry_run)
    if dry_run:
        return report
    save(intent,dict(source_plan=reference,new_plan_sha256=new_plan['plan_sha256']))
    _apply_subset_operations(operations)
    save(report_path,report)
    return report


def _apply_subset_operations(operations):
    """Apply only fully audited data operations; never touch a source file."""
    for operation in operations:
        source,dest=Path(operation['source']),Path(operation['destination'])
        if sha(source) != operation['source_sha256']:
            raise ValueError('completed source changed after audit')
        value=load(source)
        # Contracts/manifests are whole replacement documents; rows retain all
        # original model decisions/metrics and only gain audited placement data.
        if 'evaluation_valid' in value:
            value.update(operation['replacement'])
        else:
            value=operation['replacement']
        save(dest,value)
        operation['destination_sha256']=sha(dest)


def restore_completed_node_receipts(plan):
    """Validate fully seeded shards on CPU; the normal global scorer still runs.

    Never copy an old success/failure receipt or fabricate a service launch.
    Partial shards retain the collector's normal resume path.
    """
    from closedloop_evaluation_contract import stage_repeats
    report_path = Path(plan['output'])/'reuse-audit.json'
    report = load(report_path)
    if (report.get('schema') not in ('unified_completed_subset_seed_v1', 'unified_completed_sources_seed_v2')
            or report.get('new_plan_sha256') != plan['plan_sha256']
            or not report.get('source_writers_cleaned') or not report.get('original_artifacts_preserved')):
        raise ValueError('complete shard requires a bound completed-case audit')
    repeats = stage_repeats(plan['config'])
    restored = 0
    for stage in plan['stages']:
        manifest = load(stage['manifest'])
        ckpt = next(c for c in plan['config']['checkpoints'] if c['name']==stage['checkpoint'])
        for rank in range(plan['config']['nodes']):
            cases = partition(manifest['cases'], rank, plan['config']['nodes'])
            counts = [r['reused_attempts'] for r in report['by_stage_node']
                      if r['benchmark']==stage['benchmark'] and r['rank']==rank]
            if not cases or sum(counts) != len(cases)*repeats:
                continue
            directory = stage_root(plan, stage)/f'node_{rank}'
            if _exists(directory/'result.json') or _exists(directory/'launch'):
                raise ValueError('complete shard destination already has a business receipt')
            operations = [op for op in report['files'] if Path(op['destination']).is_relative_to(directory)]
            for op in operations:
                if sha(op['source']) != op['source_sha256'] or sha(op['destination']) != op['destination_sha256']:
                    raise ValueError('complete shard seed source or destination changed')
            benchmark = stage['benchmark']
            native = benchmark in ('r2r', 'rxr')
            pattern = f'evaluation/{benchmark}/**/rollouts/*.json' if native else 'evaluation/worker_*/rollouts/**/*.json'
            paths = _glob(directory, pattern)
            audited_paths = {op['destination'] for op in operations}
            expected = {case['logical_id']:case for case in cases}
            found = set()
            contract = load(directory/'evaluation'/benchmark/'contract.json') if native else None
            for path in paths:
                row = load(path)
                case = expected.get(row.get('logical_id'))
                repeat = row.get('repeat_index')
                if (str(path) not in audited_paths or not case or not row.get('evaluation_valid')
                        or row.get('infra_error') or type(repeat) is not int or not 1<=repeat<=repeats
                        or row.get('repeats_per_case') != repeats or row.get('case_sha256') != digest(case)
                        or (case['logical_id'], repeat) in found):
                    raise ValueError('complete shard has invalid, duplicate or unaudited attempts')
                if native:
                    from v6_vlnce_eval import resumable
                    if not resumable(row, case, digest(contract), repeat):
                        raise ValueError('complete native attempt is not resumable')
                elif (row.get('suite_plan_sha256') != plan['plan_sha256']
                        or row.get('checkpoint') != ckpt['identity'] or row.get('suite_rank') != rank):
                    raise ValueError('complete SAGE attempt identity changed')
                found.add((case['logical_id'],repeat))
            if found != {(case['logical_id'],r) for case in cases for r in range(1,repeats+1)}:
                raise ValueError('complete shard is missing audited attempts')
            result = dict(evaluation_valid=True, completed=len(found), reused_completed_attempts=len(found))
            save(directory/'evaluation_result.json', result)
            save(directory/'result.json', dict(plan_sha256=plan['plan_sha256'],stage=stage['index'],rank=rank,
                logical_ids=[case['logical_id'] for case in cases], selected=len(cases),
                evaluation_valid=True, exit_code=0, result=result, no_services_started=True,
                completed_case_reuse=dict(path=str(report_path),sha256=sha(report_path)),
                global_scoring_pending=True))
            restored += 1
    return restored


def seed_completed_case_sources(new_plan, reference):
    """Combine disjoint benchmark cohorts, e.g. parallel source preflights."""
    if 'sources' not in reference:
        return seed_subset_checkpoint(new_plan, reference)
    sources=reference['sources']
    if not isinstance(sources,list) or not sources:
        raise ValueError('completed sources must be a nonempty list')
    scopes=[b for source in sources for b in source.get('benchmarks',[])]
    if len(scopes)!=len(set(scopes)) or any(not source.get('benchmarks') for source in sources):
        raise ValueError('completed source benchmark scopes must be explicit and disjoint')
    output=Path(new_plan['output']);report_path=output/'reuse-audit.json';intent=output/'repair-seed-intent.json'
    if _exists(report_path):
        report=load(report_path)
        if report['source_plans']!=sources or report['new_plan_sha256']!=new_plan['plan_sha256']:
            raise ValueError('existing combined seed belongs to different plans')
        for operation in report['files']:
            if sha(operation['source'])!=operation['source_sha256'] or sha(operation['destination'])!=operation['destination_sha256']:
                raise ValueError('combined completed seed changed')
        return report
    if _exists(intent):
        raise ValueError('interrupted combined seed requires inspection')
    parts=[seed_subset_checkpoint(new_plan,source,dry_run=True) for source in sources]
    operations=[operation for part in parts for operation in part['files']]
    if len({op['destination'] for op in operations})!=len(operations):
        raise ValueError('completed sources have overlapping destinations')
    report=dict(schema='unified_completed_sources_seed_v2',source_plans=sources,
        new_plan_sha256=new_plan['plan_sha256'],reused_nodes=0,
        reused_cases=sum(part['reused_cases'] for part in parts),
        skipped_attempts=[row for part in parts for row in part.get('skipped_attempts',[])],
        skipped_shards=[row for part in parts for row in part.get('skipped_shards',[])],
        by_stage_node=[row for part in parts for row in part['by_stage_node']],
        code_changes=[row for part in parts for row in part['code_changes']],
        source_writers_cleaned=True,original_artifacts_preserved=True,files=operations)
    save(intent,dict(source_plans=sources,new_plan_sha256=new_plan['plan_sha256']))
    _apply_subset_operations(operations)
    save(report_path,report)
    return report
