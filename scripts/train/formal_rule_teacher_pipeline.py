#!/usr/bin/env python3
"""Compose existing frozen formal collectors/exporter/trainer; never manufacture gates.

prepare requires independently accepted quality AND measured efficiency evidence.
Operational acceptance records measured useful output without claiming a matched
baseline comparison; the original comparative acceptance remains available.
launch uses the existing detached six-stage supervisor. Existing brain processes
are never adopted or stopped: their endpoints/GPUs must be free before launch.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import build_opener, ProxyHandler

ART = Path('/data/zerovln/artifacts')
SELF = Path(__file__).resolve()
CONFIGURATION = dict(admission_mode='relaxed', grounding_mode='map_open', match_policy='target_or_direction')
POPULATION = dict(sage=21638, r2r=10819, rxr=10819)
GPU_PAIRS = ('0,1', '4,5', '8,9', '12,13')


def activate(root):
    root = Path(root).resolve()
    sys.path[:0] = [str(root/'scripts/train'), str(root/'scripts/eval'), str(root/'verl')]
    import run_rule_teacher_local_pipeline as runner
    import sft.build_rule_teacher_policy_review as policy
    if runner.REPO != root:
        raise ValueError('one frozen source root per process required')
    return runner, policy


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def checked(item):
    if not Path(item['path']).is_absolute() or sha(item['path']) != item['sha256']:
        raise ValueError('artifact hash changed: '+item['path'])
    return load(item['path'])


def pipeline_adapter():
    """Use this frozen entry's sibling, even after activating another source."""
    name = 'formal_pipeline_runner'
    path = SELF.with_name(name+'.py')
    existing = sys.modules.get(name)
    if existing is not None and Path(existing.__file__).resolve() == path:
        return existing
    descriptor = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(descriptor)
    sys.modules[name] = module
    descriptor.loader.exec_module(module)
    return module


def assert_fields(value, expected):
    for key, target in expected.items():
        actual = value
        for part in key.split('.'):
            actual = actual[part]
        if type(actual) is not type(target) or actual != target:
            raise ValueError(f'contract mismatch {key}: {actual!r} != {target!r}')


def report_number(report, key):
    for part in key.split('.'):
        report = report[part]
    if type(report) not in (int, float) or not math.isfinite(report) or report <= 0:
        raise ValueError('efficiency evidence requires a positive finite measured rate')
    return report


STARTUP_MANIFEST_REPLACEMENTS = (("        manifests = {}\n        for scene in sorted({row['scene'] for row in rows}):\n", "        manifests = {}\n        missing = {}\n        for scene in sorted({row['scene'] for row in rows}):\n"), ('            manifests[scene] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())\n', "            try:\n                manifest_bytes = path.read_bytes()\n            except FileNotFoundError:\n                missing[scene] = dict(path=str(path), reason='native_bank_manifest_missing')\n                continue\n            manifests[scene] = dict(path=str(path), sha256=hashlib.sha256(manifest_bytes).hexdigest())\n"), ("        config['runtime_config']['sage_navmesh_manifests'] = manifests\n", "        config['runtime_config']['sage_navmesh_manifests'] = manifests\n        config['runtime_config']['sage_navmesh_missing'] = missing\n"))


def startup_measurement_policy(source, gate, current_policy_sha):
    """Transfer existing measurement evidence only across this exact startup fix.

    Measurements retain their original policy SHA. Geometry, scheduler, admission,
    export and all other collector code must remain byte-identical.
    """
    migration = gate.get('startup_manifest_repair')
    if migration is None:
        return current_policy_sha, []
    proof = checked(migration)
    assert_fields(proof, dict(schema='rule_teacher_missing_manifest_startup_repair_v1',
        accepted=True, target_policy_sha256=current_policy_sha))
    _, policy = activate(source)
    old_root = Path(proof['measurement_source_root']).resolve()
    before = policy.policy_identity(old_root, CONFIGURATION)
    after = policy.policy_identity(source, CONFIGURATION)
    old_sha = policy.digest(before)
    if old_sha != proof['measurement_policy_sha256'] or old_sha == current_policy_sha:
        raise ValueError('startup repair measurement policy mismatch')
    changed = [name for name in policy.POLICY_FILES if before['files'][name] != after['files'][name]]
    pilot = 'scripts/eval/rule_teacher_pilot.py'
    if changed != [pilot]:
        raise ValueError('startup repair must not change another policy dependency')
    expected = (old_root/pilot).read_text()
    for original, replacement in STARTUP_MANIFEST_REPLACEMENTS:
        if expected.count(original) != 1:
            raise ValueError('startup repair original block is ambiguous')
        expected = expected.replace(original, replacement, 1)
    if expected != (Path(source)/pilot).read_text():
        raise ValueError('startup repair differs from the allowed manifest inventory change')
    tests = checked(proof['regression'])
    assert_fields(tests, dict(ok=True, target_policy_sha256=current_policy_sha,
        mixed_manifest_cohort_preserved=True, nonmissing_errors_propagate=True,
        existing_initial_failure_ledger_verified=True))
    prior_quality = checked(proof['prior_quality_gate'])
    assert_fields(prior_quality, dict(accepted=True, policy_sha256=old_sha))
    prior_report = checked(prior_quality['validation_report'])
    assert_fields(prior_report, dict(accepted=True, policy_sha256=old_sha))
    evidence = [migration, proof['regression'], proof['prior_quality_gate'],
        prior_quality['validation_report'], *prior_report['evidence']]
    evidence += [binding(old_root/name) for name in policy.POLICY_FILES]
    return old_sha, evidence

def operational_efficiency(gate, policy_sha, measurement_policy_sha=None):
    """Accept an assessed measured rate, without inventing baseline parity."""
    assert_fields(gate, dict(schema='rule_teacher_operational_efficiency_validation_v1',
        accepted=True, scope='formal_local16', policy_sha256=policy_sha,
        node_gpu_count=16, v6_throughput_parity_proven=False))
    measurement_policy_sha = measurement_policy_sha or policy_sha
    readiness = checked(gate['formal_profile'])
    assert_fields(readiness, dict(mode='full_population',
        original_counts=POPULATION, original_input_objects_unchanged=True,
        no_resampling=True, all_frozen_hashes_verified=True))
    workers = readiness['worker_configs']
    expected = {(source, worker) for source in POPULATION for worker in range(8)}
    if (len(workers) != len(expected) or
            {(w['source'], w['worker']) for w in workers} != expected):
        raise ValueError('operational acceptance requires all 24 formal workers')
    for worker in workers:
        assert_fields(worker, dict(same_depth_scheduler=True,
            parallel_initialization=True, http64=True, actor8=True))
    measurements = gate['measurements']
    if not isinstance(measurements, dict) or set(measurements) != set(POPULATION):
        raise ValueError('measured useful output required for all three sources')
    evidence, rates = [gate['formal_profile']], {}
    for source, item in measurements.items():
        assert_fields(item, dict(accepted=True))
        report = checked(item['measurement'])
        assert_fields(report, dict(schema='rule_teacher_operational_throughput_report_v1',
            source=source, policy_sha256=measurement_policy_sha, node_gpu_count=16,
            unit='admitted_complete_four_window_anchors_per_reserved_node_hour',
            resource_scope='shared_local16', includes_initialization=True,
            includes_policy_admission_and_export=True, original_data=True,
            v6_throughput_parity_proven=False))
        anchors, windows = report['admitted_anchors'], report['verified_windows']
        if (type(anchors) is not int or anchors < 0 or
                type(windows) is not int or windows != 4*anchors):
            raise ValueError('operational measurement requires actual complete four-window anchors')
        elapsed = report_number(report, 'collection_wall_seconds') + report_number(report, 'export_wall_seconds')
        measured = report['anchors_per_reserved_node_hour']
        if type(measured) not in (int, float) or not math.isfinite(measured) or measured < 0:
            raise ValueError('operational measurement requires a finite nonnegative rate')
        if not math.isclose(measured, anchors*3600/elapsed, rel_tol=1e-9):
            raise ValueError('operational rate differs from admitted output and wall time')
        # A bound loader is required in addition to the assessor's report.
        loader = checked(report['loader_verification'])
        assert_fields(loader, dict(ok=True, visual_input_mode='live_frozen_visual_v1',
            persisted_visual_features=False, zero_placeholder_features=False))
        if (type(loader['anchors']) is not int or type(loader['windows']) is not int or
                loader['anchors'] < anchors or loader['windows'] != 4*loader['anchors']):
            raise ValueError('operational measurement exceeds verified loader output')
        manifest = report['manifest']
        if not Path(manifest['path']).is_absolute() or sha(manifest['path']) != manifest['sha256']:
            raise ValueError('operational training manifest changed')
        with Path(manifest['path']).open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        groups = {}
        for row in rows:
            index = row['fast_window_index']
            if type(index) is not int or index not in range(4):
                raise ValueError('operational manifest has an invalid fast window')
            groups.setdefault(row['slow_anchor_id'], []).append(index)
        if (len(rows) != loader['windows'] or len(groups) != loader['anchors'] or
                any(sorted(indices) != list(range(4)) for indices in groups.values())):
            raise ValueError('operational loader and complete window manifest disagree')
        source_rows = [row for row in rows if row['source_dataset'] == source]
        if (len(source_rows) != windows or
                len({row['slow_anchor_id'] for row in source_rows}) != anchors):
            raise ValueError('operational source counts differ from actual training manifest')
        if not report.get('evidence'):
            raise ValueError('operational measurement requires bound collection evidence')
        evidence += [item['measurement'], report['loader_verification'], manifest, *report['evidence']]
        rates[source] = dict(measurement=measured, wall_seconds=elapsed,
            admitted_anchors=anchors, verified_windows=windows,
            resource_scope=report['resource_scope'], baseline_comparison=False)
    if sum(item['admitted_anchors'] for item in rates.values()) < 1:
        raise ValueError('operational acceptance requires at least one admitted anchor in total')
    return dict(evidence=evidence, rates=rates,
        acceptance_mode='assessed_operational_throughput', v6_throughput_parity_proven=False)


def validate_gates(source, quality, efficiency):
    _, policy = activate(source)
    identity = policy.policy_identity(source, CONFIGURATION)
    policy_sha = policy.digest(identity)
    review = dict(schema=policy.REVIEW_SCHEMA, review_type=policy.REVIEW_TYPE,
        human_verified=False, semantic_reviewed=False, policy_source_root=str(source),
        policy=identity, policy_sha256=policy_sha, policy_gate=quality)
    proof = policy.validate_policy_review(review)
    gate = checked(efficiency)
    measurement_policy_sha, migration_evidence = startup_measurement_policy(source, gate, policy_sha)
    if migration_evidence and checked(proof['validation_report']).get('startup_manifest_repair') != gate['startup_manifest_repair']:
        raise ValueError('quality and efficiency startup repair proofs differ')
    if gate.get('schema') == 'rule_teacher_operational_efficiency_validation_v1':
        operational = operational_efficiency(gate, policy_sha, measurement_policy_sha)
        evidence = [quality, efficiency, proof['validation_report'], *operational['evidence'], *migration_evidence]
        evidence += checked(proof['validation_report'])['evidence']
        for item in evidence:
            if sha(item['path']) != item['sha256']:
                raise ValueError('gate evidence changed')
        return dict(operational, policy_sha256=policy_sha, measurement_policy_sha256=measurement_policy_sha,
            startup_only_evidence_transfer=bool(migration_evidence), evidence=evidence)
    assert_fields(gate, dict(schema='rule_teacher_efficiency_validation_v1', accepted=True,
        scope='formal_local16', policy_sha256=policy_sha, node_gpu_count=16))
    comparisons = gate['comparisons']
    if not isinstance(comparisons, dict) or set(comparisons) != set(POPULATION):
        raise ValueError('accepted efficiency comparison is required for all three sources')
    evidence = [quality, efficiency, proof['validation_report']]
    evidence += checked(proof['validation_report'])['evidence']
    rates = {}
    for source_name, item in comparisons.items():
        assert_fields(item, dict(accepted=True, same_configuration=True,
            unit='admitted_complete_four_window_anchors_per_node_hour'))
        baseline, current = checked(item['baseline']), checked(item['measurement'])
        # Explicit reviewer acceptance alone is not a measured throughput result.
        old = report_number(baseline, item['baseline_value_key'])
        new = report_number(current, item['measurement_value_key'])
        if new <= old:
            raise ValueError('efficiency threshold not passed: '+source_name)
        if current.get('policy_sha256') != policy_sha:
            raise ValueError('measurement is not bound to this policy')
        evidence += [item['baseline'], item['measurement']]
        rates[source_name] = dict(baseline=old, measurement=new, ratio=new/old)
    for item in evidence:
        if sha(item['path']) != item['sha256']:
            raise ValueError('gate evidence changed')
    return dict(policy_sha256=policy_sha, evidence=evidence, rates=rates)


def ensure_new_artifact(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ART) or path == ART or path.exists():
        raise ValueError('requires a new directory on the authorized artifact disk')
    return path


def receipt(path, **equals):
    return dict(path=str(path), equals=equals)


def command(spec, mode, *extra):
    return [spec['python'], '-B', str(SELF), mode, '--spec', spec['spec_path'], *map(str, extra)]


def build_stages(spec):
    root = Path(spec['output_root'])
    collection = Path(spec['collection_plan']['path']).parent
    training = root/'training'
    proof = spec['gates']
    preconditions = [dict(proof[name], equals={'accepted': True}) for name in ('quality', 'efficiency')]
    stages = [dict(kind='collect', commands=[dict(name='three-source-collection',
        command=[spec['python'], '-B', str(SELF.parent/'formal_pipeline_runner.py'),
                 '--plan', str(root/'pipeline-plan.json'), '--run-collection'])], preconditions=preconditions,
        required_receipts=[receipt(collection/'completion.json', status='complete', runtime_cleaned=True),
            receipt(collection/'construction/formal-construction-report.json', complete=True, mode='full_population')])]
    stages.append(dict(kind='validate', idle_gpu_ids=list(range(16)), commands=[dict(name='rule-policy-admission', command=command(spec, 'validate'))],
        required_receipts=[receipt(root/'receipts/validate.json', ok=True, review_type='rule_policy_admission',
            human_verified=False, semantic_reviewed=False)]))
    stages.append(dict(kind='export', idle_gpu_ids=list(range(16)), commands=[dict(name=f'export-input-{lane:02d}',
        command=command(spec, 'export-lane', '--lane', lane), environment={'CUDA_VISIBLE_DEVICES': ''})
        for lane in range(16)], required_receipts=[receipt(root/f'receipts/export-{lane:02d}.json', ok=True,
            lane=lane, visual_input_mode='live_frozen_visual_v1', persisted_visual_features=False) for lane in range(16)]))
    stages.append(dict(kind='assemble', idle_gpu_ids=list(range(16)), commands=[dict(name='assemble-formal-training', command=command(spec, 'assemble'))],
        required_receipts=[receipt(root/'receipts/assemble.json', ok=True, formal_population_mode=True),
            receipt(training/'local_launch_plan.json', schema='rule_teacher_local16_launch_plan_v1',
                global_anchor_batch=64, frozen_frontend='live_query_and_fast_visual')]))
    for kind in ('initialize', 'train'):
        stages.append(dict(kind=kind, initialization_on_cpu=(kind == 'initialize'), commands=[dict(name=f'v6-{kind}', command=command(spec, kind))],
            required_receipts=[receipt(root/f'receipts/{kind}.json', ok=True, returncode=0, kind=kind)]))
    return stages


def prepare_services(spec, managed_plan_path):
    """Reuse current v6 TP2/NLP service preparation; the phase adapter owns workers."""
    from supervise_rule_teacher_collection import CACHE_KEYS
    managed = load(managed_plan_path)
    services = copy.deepcopy(managed['services'])
    names = {entry['name'] for entry in services}
    if len(services) != 5 or names != {'nlp', 'brain-0', 'brain-1', 'brain-2', 'brain-3'}:
        raise ValueError('four owned v6 brains and pinned NLP required')
    root = Path(spec['output_root'])/'brains'
    runtime = Path(spec['runtime'])/'owned-services'
    for entry in services:
        env = {k: str(v) for k, v in entry.get('environment', {}).items()
               if k not in CACHE_KEYS and k != 'RULE_COLLECTION_RUNTIME_ROOT'}
        env['RULE_COLLECTION_RUNTIME_ROOT'] = str(runtime)
        entry['environment'] = env
        entry['log'] = str(root/(entry['name']+'.log'))
    spec['service_urls'] = [entry['health_url'] for entry in services]
    save(root/'launch.json', dict(schema='rule_teacher_owned_formal_services_v2',
        runtime=str(runtime), source_plan=managed['source_plan'],
        environment={'RULE_COLLECTION_RUNTIME_ROOT': str(runtime.parent)},
        services=services, service_start_batch_size=4,
        collectors=[dict(name='ready-sentinel', command=command(spec, 'brain-ready'),
            log=str(root/'ready-sentinel.log'))]))
    return dict(name='owned-four-brains-and-nlp', command=[spec['python'], '-B',
        str(Path(spec['source_root'])/'scripts/train/supervise_rule_teacher_collection.py'),
        '--plan', str(root/'launch.json')], stop_before='validate', startup_timeout=1900,
        ready_receipts=[receipt(root/'ready.json', ready=True)])


def prepare(args):
    adapter = pipeline_adapter()
    source = args.source_root.resolve()
    runner, _ = activate(source)
    gates = dict(quality=binding(args.quality_gate), efficiency=binding(args.efficiency_gate))
    proof = validate_gates(source, **gates)  # Before directory creation or any subprocess.
    mode = getattr(args, 'training_authorization_mode', 'user_check')
    authorization = (binding(args.authorization_record) if mode == 'assistant_after_acceptance'
        and getattr(args, 'authorization_record', None) is not None else None)
    if mode not in ('user_check', 'assistant_after_acceptance') or (mode == 'assistant_after_acceptance' and authorization is None):
        raise ValueError('automatic training requires a frozen existing user authorization record')
    root = ensure_new_artifact(args.output)
    spec = dict(schema='formal_rule_teacher_orchestration_v5', source_root=str(source),
        output_root=str(root), spec_path=str(root/'spec.json'), python=sys.executable,
        runtime=str(runner.collection_runtime_root()/root.name/'pipeline'), gates=gates,
        gate_proof=proof, policy_configuration=CONFIGURATION, population=binding(args.population),
        backbone=args.backbone, fast_backbone=args.fast_backbone, training_epochs=10,
        training_control=dict(mode=mode, authorization_record=authorization))
    frozen = {str(path): sha(path) for path in (SELF, adapter.SELF, args.sage_template.resolve(), args.native_template.resolve(),
        args.services_template.resolve(), args.population.resolve())}
    frozen.update({p['path']: p['sha256'] for p in proof['evidence']})
    if authorization is not None:
        frozen[authorization['path']] = authorization['sha256']
    # Collection freezes every Python dependency. These non-Python trainer/config
    # dependencies must also remain immutable through later dynamic stages.
    for path in list(source.rglob('*.sh')) + list((source/'configs/action_head').glob('*.json')):
        frozen[str(path)] = sha(path)
    frozen.update({str(p): sha(p) for p in source.rglob('*.py') if '.runtime' not in p.parts})
    import prepare_rule_teacher_formal_local16_collection as collection
    spec['brain_urls'] = list(collection.BRAINS)
    collection.prepare(root/'collection', args.population.resolve(), brains=spec['brain_urls'],
        sage_template=args.sage_template.resolve(), native_template=args.native_template.resolve(),
        services_template=args.services_template.resolve(), native_scene_cache_size=2)
    service = prepare_services(spec, root/'collection/managed-services/launch.json')
    plan_path = root/'collection/collection-plan.json'
    collection_plan = load(plan_path)
    spec['collection_plan'] = binding(plan_path)
    spec['partitions'] = [dict(name=f'{w["source"]}-worker{w["worker"]}', source=w['source'],
        collection=w['collection'], review=str(root/'reviews'/f'{w["source"]}-worker{w["worker"]}.json'),
        output=str(root/'exports'/f'{w["source"]}-worker{w["worker"]}'), lane=i % 16)
        for i, w in enumerate(collection_plan['workers'])]
    if len(spec['partitions']) != 24:
        raise ValueError('expected eight frozen partitions per source')
    save(root/'spec.json', spec)
    # Freeze all generated plans and selections, not generated runtime artifacts.
    for path in root.rglob('*'):
        if path.is_file():
            frozen[str(path)] = sha(path)
    source_plan = load(root/'collection/source-plan.json')
    for section in ('source_sha256', 'code_sha256', 'input_sha256'):
        frozen.update(source_plan[section])
    model_environment = {k: v for k, v in load(args.services_template)['environment'].items() if k in
        ('LD_LIBRARY_PATH', 'CUDA_TOOLKIT_ROOT', 'CUDA_SDK', 'CUDA_SDK_VER', 'CUDA_PATH', 'CUDA_HOME', 'PPU_SDK')}
    plan = dict(schema='rule_teacher_local_pipeline_v1', output_root=str(root), runtime=spec['runtime'],
        orchestration=dict(source_root=str(source), runner_adapter=str(adapter.SELF),
            stop_grace_seconds=adapter.STOP_GRACE_SECONDS,
            collection_stop_grace_seconds=adapter.COLLECTION_STOP_GRACE_SECONDS),
        input_sha256=frozen, environment=dict(PYTHONPATH=f'{source}/verl:{source}/vln_demo/src',
            RULE_COLLECTION_RUNTIME_ROOT=str(runner.collection_runtime_root()),
            OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', **model_environment),
        services=[service], stages=build_stages(spec), poll_seconds=5,
        gpu_reservation=load(root/'collection/phase-plan.json')['gpu_reservation'])
    save(root/'pipeline-plan.json', plan)
    runner.validate_plan(plan, root/'pipeline-plan.json')
    return dict(plan=str(root/'pipeline-plan.json'), sha256=sha(root/'pipeline-plan.json'), launched=False,
        external_brains_stopped=False, gate_proof=proof)


def check_brains_released(spec, *, check_gpu_idle=True):
    root = Path(spec['output_root'])/'brains'
    completion = load(root/'completion.json')
    assert_fields(completion, dict(status='interrupted', runtime_cleaned=True))
    for url in [*spec['brain_urls'], *spec.get('service_urls', [])]:
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', int(url.rsplit(':', 1)[1].split('/', 1)[0]))) == 0:
                raise RuntimeError('owned brain endpoint remains occupied: '+url)
    if check_gpu_idle:
        runner, _ = activate(spec['source_root'])
        runner.assert_gpus_idle(list(range(16)))


def check_reserved_validation_entry(spec):
    # The parent checks GPU idleness before its own reservation starts. Repeating
    # that query here would reject the correctly owned occupier. Keep the brain
    # completion and port checks live, and bind the earlier check to this spec.
    check_brains_released(spec, check_gpu_idle=False)
    record = load(Path(spec['output_root'])/'receipts/pre-reservation-idle.json')
    assert_fields(record, dict(schema='formal_pre_reservation_idle_v1',
        stage='validate', gpu_ids=list(range(16)), idle=True))
    if record['spec'] != binding(spec['spec_path']):
        raise ValueError('pre-reservation observation belongs to another spec')
    checked(record['brain_completion'])


def run_command(argv, spec, *, environment=None):
    subprocess.run(argv, cwd=spec['source_root'], env=environment, check=True)


def data_inventory(output):
    """Pin generated tensors/condition images without a huge supervisor receipt."""
    path = output/'data-artifacts.jsonl'
    with path.open('x') as stream:
        for folder in ('samples', 'condition_images', 'fast_images'):
            for item in sorted((output/folder).rglob('*')):
                if item.is_file():
                    stream.write(json.dumps(binding(item))+'\n')
    return binding(path)


def verify_inventory(item):
    if sha(item['path']) != item['sha256']:
        raise ValueError('export data inventory changed')
    with Path(item['path']).open() as stream:
        for line in stream:
            artifact = json.loads(line)
            if sha(artifact['path']) != artifact['sha256']:
                raise ValueError('exported training data changed: '+artifact['path'])


def validate_collection(spec):
    runner, policy = activate(spec['source_root'])
    validate_gates(spec['source_root'], **spec['gates'])
    check_reserved_validation_entry(spec)
    plan = checked(spec['collection_plan'])
    construction = load(plan['construction_report'])
    assert_fields(construction, dict(complete=True, mode='full_population'))
    from refilter_rule_teacher_anchors import main as refilter
    dedup_root = Path(spec['output_root'])/'anchor-dedup'
    argv = ['--output', str(dedup_root)]
    for partition in spec['partitions']:
        argv += ['--collection', partition['source']+'='+partition['collection']]
    refilter(argv)
    ledger = binding(dedup_root/'dedup-report.json')
    height_root = Path(spec['output_root'])/'source-height'
    height_argv = [spec['python'], '-B',
        str(Path(spec['source_root'])/'scripts/eval/audit_sage_source_height.py'),
        '--output', str(height_root)]
    for partition in spec['partitions']:
        if partition['source'] == 'sage':
            height_argv += ['--collection', partition['collection']]
    run_command(height_argv, spec)
    height = binding(height_root/'audit.json')
    checked(height)
    reviews = []
    for partition in spec['partitions']:
        policy.main(['--collection', partition['collection'], '--policy-source-root', spec['source_root'],
            '--policy-gate', spec['gates']['quality']['path'], '--policy-gate-sha256', spec['gates']['quality']['sha256'],
            '--policy-configuration', json.dumps(CONFIGURATION), '--output', partition['review'],
            '--dedup-ledger', ledger['path'], '--dedup-ledger-sha256', ledger['sha256'],
            '--source-height-report', height['path'], '--source-height-report-sha256', height['sha256']])
        review = load(partition['review'])
        reviews.append(dict(name=partition['name'], source=partition['source'], review=binding(partition['review']),
            admitted=len(review['segments']), excluded=len(review['exclusions'])))
    return dict(review_type='rule_policy_admission', human_verified=False, semantic_reviewed=False,
        construction=binding(plan['construction_report']), reviews=reviews, dedup_ledger=ledger, source_height_audit=height)


def export_lane(spec, lane):
    runner, _ = activate(spec['source_root'])
    if not 0 <= lane < 16 or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('formal input export must not load GPU feature encoders')
    validation = load(Path(spec['output_root'])/'receipts/validate.json')
    assert_fields(validation, dict(ok=True, human_verified=False, semantic_reviewed=False))
    height = validation['source_height_audit']
    checked(height)
    by_name = {item['name']: item for item in validation['reviews']}
    outputs = []
    for partition in spec['partitions']:
        if partition['lane'] != lane:
            continue
        admission = by_name[partition['name']]
        review = checked(admission['review'])
        if not review['segments']:
            outputs.append(dict(name=partition['name'], source=partition['source'], status='no_admitted_anchors',
                review=admission['review'], admitted=0))
            continue
        output = Path(partition['output'])
        run_command([spec['python'], '-B', str(Path(spec['source_root'])/'verl/sft/export_rule_teacher_training.py'),
            '--review', partition['review'], '--output', str(output), '--backbone', spec['fast_backbone'],
            '--source-height-report', height['path'], '--source-height-report-sha256', height['sha256'],
            '--threads', '1', '--device', 'cpu', '--dtype', 'float32', '--visual-mode', 'live'], spec)
        loader = load(output/'loader_verification.json')
        assert_fields(loader, dict(ok=True, visual_input_mode='live_frozen_visual_v1', persisted_visual_features=False,
            weights='not_loaded_live_training', zero_placeholder_features=False))
        if loader['anchors'] <= 0 or loader['windows'] != 4*loader['anchors']:
            raise ValueError('export is not nonempty complete four-window anchors')
        outputs.append(dict(name=partition['name'], source=partition['source'], status='exported',
            review=admission['review'], anchors=loader['anchors'], windows=loader['windows'],
            data_inventory=data_inventory(output),
            artifacts=[binding(output/name) for name in ('manifest.jsonl', 'provenance.json', 'loader_verification.json')]))
    return dict(lane=lane, visual_input_mode='live_frozen_visual_v1', persisted_visual_features=False, outputs=outputs,
        parallelism='sixteen CPU input lanes; assigned partitions sequential',
        encoder_cache='none; same frozen tower runs inside each training forward')


def runner_environment(spec, name, environment):
    runner, _ = activate(spec['source_root'])
    return runner.runtime_environment(Path(spec['runtime']), name, environment)


def assemble(spec):
    root = Path(spec['output_root'])
    validated = load(root/'receipts/validate.json')
    assert_fields(validated, dict(ok=True, review_type='rule_policy_admission'))
    ledger = validated['dedup_ledger']
    checked(ledger)
    height = validated['source_height_audit']
    checked(height)
    by_source = {source: [] for source in POPULATION}
    proofs, inventories = [], []
    for lane in range(16):
        path = root/f'receipts/export-{lane:02d}.json'
        result = load(path)
        assert_fields(result, dict(ok=True, lane=lane))
        proofs.append(binding(path))
        for item in result['outputs']:
            if item['status'] == 'exported':
                verify_inventory(item['data_inventory'])
                inventories.append(item['data_inventory'])
                for artifact in item['artifacts']:
                    if sha(artifact['path']) != artifact['sha256']:
                        raise ValueError('export artifact changed')
                manifest = next(a['path'] for a in item['artifacts'] if Path(a['path']).name == 'manifest.jsonl')
                by_source[item['source']].append(manifest)
    if not any(by_source.values()):
        raise ValueError('training requires at least one actual qualified complete anchor')
    # Empty source manifests preserve source accounting without inventing anchors.
    empty_inputs = []
    for source, manifests in by_source.items():
        if not manifests:
            path = root/'empty-source-inputs'/(source+'.jsonl')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x'):
                pass
            manifests.append(str(path))
            empty_inputs.append(binding(path))
    argv = [spec['python'], '-B', str(Path(spec['source_root'])/'scripts/train/prepare_rule_teacher_local16_training.py'),
        '--output', str(root/'training/data'), '--formal-population-report', spec['population']['path'],
        '--construction-report', checked(spec['collection_plan'])['construction_report'], '--epochs', '10', '--live-fast-visual',
        '--dedup-ledger', ledger['path'], '--dedup-ledger-sha256', ledger['sha256'],
        '--source-height-report', height['path'], '--source-height-report-sha256', height['sha256']]
    for source, manifests in by_source.items():
        for manifest in manifests:
            argv += ['--'+source, manifest]
    run_command(argv, spec)
    training = root/'training'
    config_path = training/'experiment_config.json'
    config = load(config_path)
    config['shared'].update(slow_model=spec['backbone'], fast_backbone=spec['fast_backbone'])
    save(config_path, config)
    import prepare_rule_teacher_local16_training as assembler
    launch = assembler.local_launch_plan(config_path, live_fast_visual=True)
    # Use the runner's existing cache binder, including rank-local cache paths.
    bound_env = runner_environment(spec, 'training', launch['environment'])
    launch['environment'] = bound_env
    launch['cleanup_runtime_after_exit'] = bound_env['LOCAL_RUNTIME_ROOT']
    save(training/'local_launch_plan.json', launch)
    assembly = load(training/'data/assembly.json')
    assert_fields(assembly, dict(formal_population_mode=True, global_anchor_batch=64, epochs=10,
        fast_calls_per_slow_anchor=4, full_episode_success_filter=False, fast_visual_input_mode='live_frozen_visual_v1'))
    return dict(formal_population_mode=True, max_steps=assembly['max_steps'],
        dedup_ledger=ledger, source_height_audit=height, empty_source_manifests=empty_inputs,
        export_receipts=proofs, data_inventories=inventories, launch_plan=binding(training/'local_launch_plan.json'),
        artifacts=[binding(training/name) for name in ('data/assembly.json', 'data/manifest.jsonl', 'experiment_config.json')])


def nonempty_bindings(root, names):
    result = []
    for name in names:
        path = root/name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError('missing or empty real weight artifact: '+str(path))
        result.append(binding(path))
    return result


def verify_training_product(kind, launch, maximum):
    env = launch['environment']
    config_sha = launch['experiment_config_sha256']
    if sha(launch['experiment_config']) != config_sha:
        raise ValueError('assembled training configuration changed')
    if kind == 'initialize':
        root = Path(env['HEAD_CHECKPOINT']).parent
        report_path = root/'scale_report.json'
        report = load(report_path)
        assert_fields(report, dict(schema='action_head_scale_initialization_v1', scale='D16',
            experiment_config_sha256=config_sha, max_steps=maximum, reused_base_head_weights=False,
            velocity_output_weight_nonzero=0, velocity_output_bias_nonzero=0))
        weights = nonempty_bindings(root, ['head/fast_slow_action_head.pt',
            'head/fast_slow_action_head_config.json', 'query/dualvln_query_embeddings.pt'])
    else:
        root = Path(env['OUTPUT_DIR'])/f'global_step_{maximum:06d}'
        report_path = root/'report.json'
        report = load(report_path)
        assert_fields(report, dict(schema='action_head_verl_sft_report_v2', ok=True,
            action_head_as_tool_version='v6', global_step=maximum, local_step=maximum, step_offset=0,
            world_size=16, anchors_per_gpu=4, global_batch_size=64, configured_fsdp_size=1,
            frozen_query_backend='query_only', training_experiment_config_sha256=config_sha,
            frozen_qwen_exported=False, scale_name='D16',
            fast_visual_input_mode='live_frozen_visual_v1', persisted_visual_features=False,
            fast_visual_backbone=env['ACTION_HEAD_FAST_BACKBONE'], fast_visual_hidden_size=2560))
        report_number(report, 'query_parameter_delta_l2')
        report_number(report, 'action_head_parameter_delta_l2')
        if (set(report['qwen_audit']) != {'requires_grad_count', 'optimizer_overlap_count', 'non_none_grad_count'}
                or any(type(value) is not int or value != 0 for value in report['qwen_audit'].values())):
            raise ValueError('trainable/frozen parameter audit failed')
        weights = nonempty_bindings(root, ['action_head/fast_slow_action_head.pt',
            'action_head/fast_slow_action_head_config.json', 'dualvln_query/dualvln_query_embeddings.pt',
            'training_experiment.json'])
        if sha(root/'training_experiment.json') != config_sha:
            raise ValueError('checkpoint embeds another training configuration')
    return dict(report=binding(report_path), weights=weights, max_steps=maximum,
        full_optimizer_resume_claimed=False)


def live_process_group_members():
    group, own = os.getpgrp(), os.getpid()
    result = []
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            pid = int(path.parent.name)
            fields = path.read_text().rsplit(')', 1)[1].split()
            if pid != own and int(fields[2]) == group and fields[0] != 'Z':
                result.append(pid)
        except (OSError, ValueError, IndexError):
            continue
    return result


def verify_assembled_inputs(assembly):
    assert_fields(assembly, dict(ok=True, formal_population_mode=True))
    for artifact in assembly['artifacts']:
        if binding(artifact['path']) != artifact:
            raise ValueError('assembled training inputs changed')
    for inventory in assembly['data_inventories']:
        verify_inventory(inventory)
    for key in ('dedup_ledger', 'source_height_audit'):
        if key in assembly:
            checked(assembly[key])
    for item in assembly.get('empty_source_manifests', []):
        if binding(item['path']) != item or Path(item['path']).stat().st_size != 0:
            raise ValueError('empty source accounting changed')


def assistant_training_acceptance(spec, kind):
    """Apply the already authorized assistant acceptance, never impersonate a user check."""
    control = spec['training_control']
    if control['mode'] != 'assistant_after_acceptance':
        raise ValueError('automatic training mode was not explicitly frozen')
    record = control['authorization_record']
    if record is None or binding(record['path']) != record:
        raise ValueError('existing user authorization record changed or is missing')
    proof = validate_gates(spec['source_root'], **spec['gates'])
    root = Path(spec['output_root']).resolve()
    if not root.is_relative_to(ART):
        raise ValueError('acceptance receipt must be on the artifact disk')
    assembly_binding = binding(root/'receipts/assemble.json')
    assembly = checked(assembly_binding)
    verify_assembled_inputs(assembly)
    value = dict(schema='formal_collection_assistant_acceptance_v1', accepted=True,
        authorization_mode='assistant_after_quality_and_efficiency_acceptance',
        human_verified=False, user_checked_current_collection=False,
        authorization_record=record, spec=binding(spec['spec_path']), assembly=assembly_binding,
        gates=spec['gates'], policy_sha256=proof['policy_sha256'])
    path = root/'receipts/collection-assistant-acceptance.json'
    if path.exists():
        if load(path) != value:
            raise ValueError('assistant acceptance belongs to changed inputs or gates')
    elif kind == 'initialize':
        save(path, value)
    else:
        raise ValueError('training requires the preceding assistant acceptance')
    return binding(path)


def training_acceptance(spec, kind):
    if spec.get('training_control', {}).get('mode', 'user_check') == 'assistant_after_acceptance':
        return dict(collection_assistant_acceptance=assistant_training_acceptance(spec, kind))
    if kind == 'initialize':
        await_collection_approval(spec)
    else:
        checked_collection_approval(spec)
    return dict(collection_user_approval=binding(Path(spec['output_root'])/'receipts/collection-user-approval.json'))


def checked_collection_approval(spec):
    """The user's collection check binds the assembled data and this exact plan."""
    root = Path(spec['output_root'])
    approval = load(root/'receipts/collection-user-approval.json')
    assert_fields(approval, dict(schema='formal_collection_user_approval_v1', accepted=True))
    for key, path in [('spec', Path(spec['spec_path'])), ('assembly', root/'receipts/assemble.json')]:
        if approval[key] != binding(path):
            raise ValueError('collection approval belongs to another '+key)
    if binding(approval['user_record']['path']) != approval['user_record']:
        raise ValueError('user collection-check record changed')
    return approval


def record_collection_approval(spec, user_record, expected_spec_sha256, expected_assembly_sha256):
    """Record an explicit user decision against already assembled, unchanged data."""
    root = Path(spec['output_root']).resolve()
    path = root/'receipts/collection-user-approval.json'
    if not root.is_relative_to(ART) or path.exists():
        raise ValueError('fresh approval receipt on the data disk required')
    plan, assembly_binding = binding(spec['spec_path']), binding(root/'receipts/assemble.json')
    if plan['sha256'] != expected_spec_sha256 or assembly_binding['sha256'] != expected_assembly_sha256:
        raise ValueError('collection changed since the user checked it')
    assembly = checked(assembly_binding)
    verify_assembled_inputs(assembly)
    value = dict(schema='formal_collection_user_approval_v1', accepted=True,
        spec=plan, assembly=assembly_binding, user_record=binding(user_record), recorded_unix=time.time())
    save(path, value)
    return binding(path)


def await_collection_approval(spec):
    root = Path(spec['output_root'])
    save(root/'receipts/collection-review-pending.json', dict(
        schema='formal_collection_review_pending_v1', status='awaiting_user_collection_check',
        spec=binding(spec['spec_path']), assembly=binding(root/'receipts/assemble.json'),
        training_started=False))
    while not (root/'receipts/collection-user-approval.json').exists():
        time.sleep(5)
    return checked_collection_approval(spec)


def configured_training(spec, kind):
    runner, _ = activate(spec['source_root'])
    root = Path(spec['output_root'])
    assembly_path = root/'receipts/assemble.json'
    assembly = load(assembly_path)
    assert_fields(assembly, dict(ok=True, formal_population_mode=True))
    acceptance = training_acceptance(spec, kind)
    for item in assembly['artifacts']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('assembled training inputs changed')
    launch = checked(assembly['launch_plan'])
    maximum = assembly['max_steps']
    if type(maximum) is not int or maximum <= 0:
        raise ValueError('positive assembled final step required')
    if kind == 'train':
        for inventory in assembly['data_inventories']:
            verify_inventory(inventory)
        initialized = load(root/'receipts/initialize.json')
        assert_fields(initialized, dict(ok=True, returncode=0, kind='initialize'))
        for item in [initialized['report'], *initialized['weights']]:
            if sha(item['path']) != item['sha256']:
                raise ValueError('initialization artifact changed')
    target = (Path(launch['environment']['HEAD_CHECKPOINT']).parent if kind == 'initialize'
              else Path(launch['environment']['OUTPUT_DIR']))
    if target.exists():
        raise ValueError('refusing implicit initialization/training retry: '+str(target))
    env = runner.runtime_environment(Path(spec['runtime']), kind+'-child', launch['environment'])
    if kind == 'initialize':
        # The original initializer constructs CPU tensors; all GPUs remain owned
        # by the stage reservation until the training transition.
        env['CUDA_VISIBLE_DEVICES'] = ''
    run_command(launch[kind+'_argv'], spec, environment=env)
    # The detached runner owns this process group. A zero parent exit must not
    # masquerade as completion while surviving trainer ranks still write files.
    if live_process_group_members():
        raise RuntimeError('owned training process group still contains live children')
    return dict(kind=kind, returncode=0, **acceptance, assembly=binding(assembly_path),
        launch_plan=assembly['launch_plan'], **verify_training_product(kind, launch, maximum))


def brain_ready(spec):
    opener = build_opener(ProxyHandler({}))
    for url in spec['service_urls']:
        with opener.open(url, timeout=10) as response:
            data = json.load(response)
            if response.status != 200 or (url.endswith('/v1/models') and not data.get('data')):
                raise ValueError('owned formal service not ready: '+url)
    save(Path(spec['output_root'])/'brains/ready.json', dict(ready=True, urls=spec['service_urls'],
        readiness='four v6 model endpoints and owned NLP; generation quality separately gated'))
    while True:
        time.sleep(30)


def launch(spec):
    adapter = pipeline_adapter()
    path = Path(spec['output_root'])/'pipeline-plan.json'
    runner = adapter.runner_for_plan(path)
    validate_gates(spec['source_root'], **spec['gates'])
    for url in [*spec['brain_urls'], *spec.get('service_urls', [])]:
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', int(url.rsplit(':', 1)[1].split('/', 1)[0]))) == 0:
                raise RuntimeError('existing brain must be handled separately by verified owner: '+url)
    runner.assert_gpus_idle(list(range(16)))
    return runner.launch(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='mode', required=True)
    prepare_parser = subs.add_parser('prepare')
    for field in ('source-root', 'output', 'population', 'quality-gate', 'efficiency-gate', 'sage-template', 'native-template', 'services-template'):
        prepare_parser.add_argument('--'+field, required=True, type=Path)
    prepare_parser.add_argument('--backbone', default='/opt/external/user-e7a01947/Model/Qwen3.6-35B-A3B')
    prepare_parser.add_argument('--training-authorization-mode', choices=('assistant_after_acceptance', 'user_check'), default='assistant_after_acceptance')
    prepare_parser.add_argument('--authorization-record', type=Path, help='immutable record of the existing user authorization; required for automatic training')
    prepare_parser.add_argument('--fast-backbone', default='/opt/external/user-e7a01947/Model/Qwen3.5-4B')
    for mode in ('launch', 'validate', 'export-lane', 'assemble', 'initialize', 'train', 'brain-ready'):
        item = subs.add_parser(mode)
        item.add_argument('--spec', required=True, type=Path)
        if mode == 'export-lane':
            item.add_argument('--lane', required=True, type=int)
    approval_parser = subs.add_parser('approve-collection')
    approval_parser.add_argument('--spec', required=True, type=Path)
    approval_parser.add_argument('--user-record', required=True, type=Path)
    approval_parser.add_argument('--expected-spec-sha256', required=True)
    approval_parser.add_argument('--expected-assembly-sha256', required=True)
    args = parser.parse_args(argv)
    if args.mode == 'prepare':
        print(json.dumps(prepare(args)))
        return
    spec = load(args.spec)
    if args.mode == 'approve-collection':
        print(json.dumps(record_collection_approval(spec, args.user_record,
            args.expected_spec_sha256, args.expected_assembly_sha256)))
        return
    activate(spec['source_root'])
    if args.mode == 'launch':
        print(json.dumps(launch(spec)))
        return
    if args.mode == 'brain-ready':
        brain_ready(spec)
        return
    mode = f'export-{args.lane:02d}' if args.mode == 'export-lane' else args.mode
    path = Path(spec['output_root'])/f'receipts/{mode}.json'
    if path.exists():
        raise ValueError('stage receipt already exists; use a new attempt')
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'bridge received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    started = time.time()
    try:
        result = (validate_collection(spec) if args.mode == 'validate' else
            export_lane(spec, args.lane) if args.mode == 'export-lane' else
            assemble(spec) if args.mode == 'assemble' else configured_training(spec, args.mode))
        save(path, dict(ok=True, started_unix=started, finished_unix=time.time(), **result))
    except BaseException as exc:
        save(path, dict(ok=False, started_unix=started, finished_unix=time.time(), error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
