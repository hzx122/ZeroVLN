#!/usr/bin/env python3
"""Freeze scene-coherent formal workers and aggregate every construction outcome.

Collection uses the existing pilots and supervisor. The optional phase runner is
an adapter over the existing durable Pipeline, with eight workers per source.
Preparation and aggregation themselves do not render or query the model.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

from prepare_rule_teacher_local16_collection import option
from prepare_rule_teacher_three_source import ART, REPO, rank, save, save_rows, sha256, verify_plan
from prepare_rule_teacher_quality_attempt import native_cpu_environment, native_cpu_command, NATIVE_DATASETS
from supervise_rule_teacher_collection import CACHE_KEYS, collection_runtime_root
from run_rule_teacher_local_pipeline import (
    Pipeline, utc, runtime_environment, process_identity, read_status_json, verify_hashes,
)

SOURCES = ('sage', 'r2r', 'rxr')
POPULATION = dict(sage=21638, r2r=10819, rxr=10819)
DEFAULT_POPULATION = ART/'rule-teacher-r6-formal-local16-20261003-r1/population/formal-population-report.json'
DEFAULT_SAGE_TEMPLATE = ART/'rule-teacher-local16-20261002-r6/sage-shared-throughput/worker0/launch.json'
DEFAULT_NATIVE_TEMPLATE = ART/'rule-teacher-quality-attempt-20261002-r5/launch.json'
V6_QUERY_CHECKPOINT = Path('/data/zerovln/action_head/integration/qwen36_four_query_5001_migration_20260824/query/dualvln_query_embeddings.pt')
BRAINS = tuple(f'http://127.0.0.1:{29200 + 100*i}' for i in range(4))
RENDER_GPUS = [2, 3, 6, 7, 10, 11, 14, 15]


def identity(row):
    return row['dataset'], row['scene'], str(row['task_id'])


def sage_native_inputs(template, rows, population_sha):
    """Freeze native-floor inputs without filtering any selected instruction.

    A supported original case is only a service bootstrap. Missing seeds and
    unresolved starts stay in the unchanged instruction population.
    """
    collector = next(e for e in template['collectors'] if e['name']=='sage-collector')
    root = collector.get('environment', {}).get('SAGE_NAVMESH_ROOT')
    metadata = template.get('sage_native_floor_bank')
    if not root and metadata is None:
        return {}, None
    if not root or not metadata:
        raise ValueError('native SAGE floor bank requires explicit hashed bindings')
    bank = Path(root)
    if not bank.is_absolute():
        raise ValueError('native SAGE floor bank must be absolute')
    frozen = {}
    def checked(binding):
        path = Path(binding['path'])
        if not path.is_absolute() or sha256(path)!=binding['sha256']:
            raise ValueError('native SAGE floor input changed')
        frozen[str(path)] = binding['sha256']
        return path
    index_path = checked(metadata['index'])
    coverage_path = checked(metadata['coverage'])
    ledger_path = checked(metadata['ledger'])
    if index_path.parent.resolve()!=bank.parent.resolve():
        raise ValueError('native SAGE bank index belongs to another directory')
    index = json.loads(index_path.read_text())
    coverage = json.loads(coverage_path.read_text())
    selected_scenes = {row['scene'] for row in rows}
    if (index['population_sha256']!=population_sha or
            coverage['population_sha256']!=population_sha or
            coverage['selected_instructions']!=len(rows) or
            coverage['selected_scenes']!=len(selected_scenes) or
            coverage['raw_siblings_verified'] is not True):
        raise ValueError('native SAGE bank does not cover the frozen population')
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines() if line]
    if Counter((row['scene'],str(row['task_id'])) for row in ledger)!=Counter(
            (row['scene'],str(row['task_id'])) for row in rows):
        raise ValueError('native SAGE floor ledger changed selected identities')
    entries = {entry['scene']:entry for entry in index['scenes']}
    if (len(entries)!=len(index['scenes']) or
            set(entries)|set(index['missing_scene_ids'])!=selected_scenes or
            set(entries)&set(index['missing_scene_ids']) or
            coverage['native_bank_scenes']!=len(entries)):
        raise ValueError('native SAGE seed and missing-seed coverage changed')
    originals = {(row['scene'],str(row['task_id'])):row for row in rows}
    for row in ledger:
        original = originals[(row['scene'],str(row['task_id']))]
        if row['original_status']!=original['status']:
            raise ValueError('native SAGE floor ledger changed an original failure')
        if row['scene'] in entries and row.get('manifest')!=entries[row['scene']]['manifest']:
            raise ValueError('native SAGE floor ledger and seed bindings differ')
    for scene, entry in entries.items():
        manifest_path = checked(entry['manifest'])
        link = bank/scene.removeprefix('sage-')/'manifest.json'
        if link.resolve()!=manifest_path.resolve():
            raise ValueError('native SAGE bank link differs from its frozen index')
        frozen[str(link)] = entry['manifest']['sha256']
        manifest = json.loads(manifest_path.read_text())
        if (manifest['schema']!='sage-usd-navmesh-v1' or
                str(manifest['scene'])!=scene.removeprefix('sage-') or
                manifest['collision_sidedness']!='authored' or
                abs(manifest['settings']['agent_radius']-.07)>1e-9):
            raise ValueError('native SAGE navigation contract changed')
        for key in ('usd','navmesh','structure'):
            checked(dict(path=manifest[key+'_path'],sha256=manifest[key+'_sha256']))
        if manifest.get('original_manifest_path'):
            checked(dict(path=manifest['original_manifest_path'],
                         sha256=manifest['original_manifest_sha256']))
    python = Path(metadata['python'])
    if not python.is_absolute() or not python.is_file():
        raise ValueError('native SAGE requires an explicit existing Python interpreter')
    frozen[str(python)] = sha256(python)
    for entry in template['services']:
        if entry['name'].startswith('sage-gym-'):
            other = entry.get('environment', {}).get('SAGE_NAVMESH_ROOT', root)
            if Path(other).resolve()!=bank.resolve():
                raise ValueError('teacher and Gym native SAGE banks differ')
    eligible = {(row['scene'],str(row['task_id'])) for row in ledger
                if row.get('start_reason')=='source_floor_resolved' and
                row.get('original_route_status')=='same_floor_original_route'}
    return frozen, dict(python=str(python),root=str(bank),bootstrap_identities=eligible,
                        route_ledger=dict(metadata['ledger']))


def native_sage_command(command, python):
    command = list(command)
    index = next((i for i,arg in enumerate(command) if Path(arg).name in
                  ('python','python3','python3.12')), None)
    if index is None:
        raise ValueError('native SAGE command has no supported Python executable')
    command[index] = python
    return native_cpu_command(command)


def load_population(path):
    report = json.loads(path.read_text())
    if report.get('schema') != 'formal_rule_teacher_population_v1':
        raise ValueError('requires the frozen formal population')
    scope = report.get('same_floor_scope') or {}
    if scope.get('enabled') is not True:
        raise ValueError('formal collection requires a same-floor-admitted SAGE population')
    scope_path = Path(scope['route_audit'])
    if sha256(scope_path) != scope['route_audit_sha256']:
        raise ValueError('same-floor audit changed')
    routes = {}
    for line in scope_path.read_text().splitlines():
        if not line:
            continue
        audit = json.loads(line)
        key = audit['scene'], audit['original_trajectory_id']
        if key in routes:
            raise ValueError('duplicate same-floor route audit')
        routes[key] = audit
    result = {}
    for source in SOURCES:
        meta = report['sources'][source]
        manifest = Path(meta['manifest'])
        if sha256(manifest) != meta['manifest_sha256']:
            raise ValueError('population manifest changed: '+source)
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
        if len(rows) != meta['population_count'] or len(rows) != POPULATION[source]:
            raise ValueError('formal population count changed: '+source)
        if len({identity(row) for row in rows}) != len(rows):
            raise ValueError('duplicate frozen population identity')
        for row in rows:
            if source == 'sage':
                binding = row.get('same_floor_scope') or {}
                audit = routes.get((row['scene'], binding.get('original_trajectory_id')))
                if (binding.get('status') != 'retained' or audit is None or audit['status'] != 'retained'
                        or any(binding.get(k) != audit.get(k) for k in
                               ('source_path','source_sha256','labels_path','labels_sha256'))):
                    raise ValueError('SAGE case not bound to a retained same-floor route')
            if row['dataset'] != source or row['status'] not in ('pending_collection', 'source_unexecutable'):
                raise ValueError('unexpected source/status in frozen population')
            if row['status'] == 'source_unexecutable' and not row.get('source_validation_errors'):
                raise ValueError('source failure needs the original diagnostic')
        result[source] = rows
    return report, result


def select_validation(rows, limit, seed):
    """Freeze equal scene quotas before execution, plus known source failures.

    At 1024/source this uses 32 SAGE scenes x 32 instructions or 16 native
    scenes x 64 instructions. Eight workers therefore get exactly 128 cases
    from at most four scenes. Eligibility depends only on frozen input counts.
    """
    usable = [row for row in rows if row['status'] == 'pending_collection']
    failures = [row for row in rows if row['status'] == 'source_unexecutable']
    if limit is None:
        return usable, failures
    if limit < 8 or limit > len(usable) or limit % 8:
        raise ValueError('validation limit must fit each source and be divisible by eight workers')
    groups = defaultdict(list)
    for row in usable:
        groups[row['scene']].append(row)
    quota = math.gcd(limit // 8, 32 if usable[0]['dataset'] == 'sage' else 64)
    scenes = sorted((s for s, group in groups.items() if len(group) >= quota), key=lambda s: rank(s, seed))
    scene_count = limit // quota
    if len(scenes) < scene_count:
        raise ValueError('insufficient scenes for the fixed validation scene quota')
    chosen = []
    for scene in scenes[:scene_count]:
        ordered = sorted(groups[scene], key=lambda row: rank(':'.join(identity(row)), seed))
        # Within each chosen scene retain one of every available instruction
        # type before seeded filling. This uses no construction outcomes.
        first = {}
        for row in ordered:
            first.setdefault(row.get('instruction_type', 'native'), row)
        selected = list(first.values())
        if len(selected) > quota:
            raise ValueError('scene quota cannot retain its instruction type strata')
        keys = {identity(row) for row in selected}
        selected += [row for row in ordered if identity(row) not in keys][:quota-len(selected)]
        chosen.extend(selected)
    if ({row.get('instruction_type', 'native') for row in chosen}
            != {row.get('instruction_type', 'native') for row in usable}):
        raise ValueError('fixed validation scenes do not cover every frozen instruction type')
    return chosen, failures


def balance_scenes(rows, seed, workers=8):
    groups = defaultdict(list)
    for row in rows:
        groups[row['scene']].append(row)
    if len(groups) < workers:
        raise ValueError('eight scene-independent workers require at least eight selected scenes')
    buckets = [[] for _ in range(workers)]
    for scene in sorted(groups, key=lambda s: (-len(groups[s]), rank(s, seed))):
        worker = min(range(workers), key=lambda i: (len(buckets[i]), i))
        buckets[worker].extend(sorted(groups[scene], key=lambda row: rank(':'.join(identity(row)), seed)))
    return buckets


def set_option(argv, key, value):
    if key in argv:
        option(argv, key, value)
    else:
        argv.extend([key, str(value)])


def rebase_entry(entry, old_repo):
    result = copy.deepcopy(entry)
    result['command'] = [str(v).replace(str(old_repo), str(REPO)) for v in result['command']]
    result['environment'] = {k: str(v).replace(str(old_repo), str(REPO))
                             for k, v in result.get('environment', {}).items()
                             if k not in CACHE_KEYS and k != 'RULE_COLLECTION_RUNTIME_ROOT'}
    result.pop('native_gym_workspace', None)
    return result


def owned_runtime_environment(runtime):
    # An explicit override is already validated by collection_runtime_root.
    # Keep the existing local default implicit: the override accepts only
    # container-local roots, while local default runs use REPO.parent/.runtime.
    return ({'RULE_COLLECTION_RUNTIME_ROOT': str(runtime)}
            if os.environ.get('RULE_COLLECTION_RUNTIME_ROOT') else {})


def native_selection(source, input_path):
    """Reuse the pilot's exact start/geometry resolution and selection format."""
    sys.path[:0] = [str(REPO/'scripts/eval'), str(REPO/'vln_demo/src'), str(REPO/'verl')]
    import mp3d_rule_teacher_pilot as native
    args = native.parser().parse_args(['--backend', source, '--split', 'train',
        '--cases', str(input_path), '--output', str(input_path.parent/'unused'),
        '--render-url', 'http://127.0.0.1:7200', '--gym-url', 'http://127.0.0.1:21000'])
    dataset = native.MP3DInstructionSet(source, split='train', source=args.dataset_source)
    rows = native.source_cases(args, dataset)
    return dict(schema='mp3d_original_instruction_start_pilot_v1', backend=source,
                split='train', scope='Explicit original native starts; no reference-route/goal target labels', cases=rows)


def worker_plan(root, source, index, rows, template, source_plan, brains, *, actors=64,
                native_scene_cache_size=1, native_scene_refill=False, sage_native=None):
    if isinstance(actors, bool) or not isinstance(actors, int) or not 1 <= actors <= 64:
        raise ValueError('actors must be an integer in 1..64')
    if (isinstance(native_scene_cache_size, bool) or not isinstance(native_scene_cache_size, int)
            or native_scene_cache_size < 1):
        raise ValueError('native scene cache size must be a positive integer')
    wr = root/source/f'worker{index}'
    out = wr/'collections'/source/'shard_000'
    runtime = collection_runtime_root()/root.name/source/f'worker{index}'
    collector = next(e for e in template['collectors'] if e['name'] == source+'-collector')
    old_repo = next(Path(arg).parents[2] for arg in collector['command'] if arg.endswith('_teacher_pilot.py'))
    render_name = 'sage-render' if source == 'sage' else 'rxr-render'
    render = rebase_entry(next(e for e in template['services'] if e['name'] == render_name), old_repo)
    actor = rebase_entry(next(e for e in template['services'] if e['name'].startswith(source+'-gym-')), old_repo)
    collector = rebase_entry(collector, old_repo)
    render_url = f'http://127.0.0.1:{7200+index}'
    lane, gpu = index//2, (index//2)*4 + 2 + index%2
    minimal = [dict(row, backend=source, task=str(row['task_id'])) for row in rows]
    if source == 'sage' and sage_native is not None:
        for row in minimal:
            row['native_route_scope'] = dict(sage_native['route_ledger'])
    # Native input and final selection differ intentionally: the original pilot
    # resolves private reset/geometry metadata before emitting selection.json.
    input_path = wr/'inputs.json'
    save(input_path, dict(cases=minimal))
    if source == 'sage':
        selection = dict(schema='formal_sage_original_train_v1', split='train',
                         scope='Frozen formal population, original train instructions, no outcome filtering', cases=minimal)
    else:
        selection = native_selection(source, input_path)
    save(out/'selection.json', selection)
    env = collector['environment']
    env.update(PYTHONPATH=f'{REPO}:{REPO}/verl:{REPO}/vln_demo/src',
        CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='8',
        **owned_runtime_environment(runtime),
        RULE_TEACHER_ADMISSION='relaxed', RULE_TEACHER_GROUNDING='map_open',
        RULE_TEACHER_MATCH_POLICY='target_or_direction', RULE_TEACHER_EXACT_FOUR_FAST_CALLS='1',
        RULE_NATIVE_STAIR_SEARCH_POLICY='verified_proposals')
    if source == 'sage':
        env.update(SAGE_SPLIT='train', SAGE_RENDER_URL=render_url, SAGE_UNLOAD_ON_CLOSE='0')
        if sage_native is not None:
            env.update(native_cpu_environment())
            env.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1',
                       SAGE_NAVMESH_ROOT=sage_native['root'], SAGE_REQUIRE_NATIVE_ROUTE_SCOPE='1')
            collector['command'] = native_sage_command(collector['command'], sage_native['python'])
        render['environment'].update(SAGE_SPLIT='train', SAGE_RENDER_URL=render_url)
        set_option(render['command'], '--gpus', gpu)
    else:
        render['environment'].update(native_cpu_environment())
        render['environment']['PYTHONPATH'] = env['PYTHONPATH']
        set_option(render['command'], '--scene-cache-size', native_scene_cache_size)
        env.update(VLN_HABITAT_RENDER_URL=render_url, VLN_RENDER_MODE='batch', VLN_RENDER_TIMEOUT='600',
                   VLN_NATIVE_EPISODE_CACHE='1')
    render.update(name=source+'-render', log=str(wr/'renderer.log'), port=7200+index,
                  health_url=render_url+('/ping' if source == 'sage' else '/health'), startup_group='render')
    set_option(render['command'], '--port', 7200+index)
    services, urls = [render], []
    bootstrap = rows
    if source=='sage' and sage_native is not None:
        bootstrap = [row for row in rows if (row['scene'],str(row['task_id'])) in
                     sage_native['bootstrap_identities']]
        if not bootstrap:
            raise ValueError('native SAGE worker has no supported original service bootstrap')
    for slot in range(actors):
        name, port = f'{source}-gym-{slot:02}', 21000+100*index+slot
        url = f'http://127.0.0.1:{port}'
        urls.append(url)
        entry = copy.deepcopy(actor)
        actor_env = dict(env)
        # Threaded text retrieval belongs to collectors, not every Gym actor.
        actor_env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        if source == 'sage':
            # Initialize service metadata only; the requested original case
            # performs its own reset and receives the existing typed rejection.
            actor_env['VLN_GYM_DEFER_INITIAL_RESET'] = '1'
        if source=='sage' and sage_native is not None:
            actor_env.update(native_cpu_environment(), SAGE_NAVMESH_ROOT=sage_native['root'])
            entry['command'] = native_sage_command(entry['command'], sage_native['python'])
        if source != 'sage':
            actor_env.update(native_cpu_environment())
            entry['native_gym_workspace'] = dict(cwd=str(runtime/name/'work'),
                config_source=str(REPO/'vln_demo/src/vln_demo.yaml'),
                config_sha256=sha256(REPO/'vln_demo/src/vln_demo.yaml'),
                data_target=str(NATIVE_DATASETS.parent.resolve()))
            set_option(entry['command'], '--output_path', wr/'services'/name)
            entry['command'] = native_cpu_command(entry['command'])
        entry.update(name=name, port=port, log=str(wr/f'{name}.log'), health_url=url+'/health',
                     startup_group='actors', startup_timeout=600, environment=actor_env)
        initial = bootstrap[slot % len(bootstrap)] if source == 'sage' else rows[0]
        for key, value in {'--port': port, '--scene': initial['scene'],
                '--episode_id': initial['task_id'], '--split': 'train'}.items():
            set_option(entry['command'], key, value)
        services.append(entry)
    for entry in services:
        entry['environment'].update(owned_runtime_environment(runtime))
    collector.update(log=str(wr/'collector.log'), environment=env)
    argv = collector['command']
    # Reuse the validated scheduler overlap. All tools still wait for the
    # complete same-depth wave; only CPU validation overlaps other HTTP calls.
    overlap_flag = '--overlap-candidate-validation'
    if actors >= 2 and overlap_flag not in argv:
        argv.append(overlap_flag)
    elif actors == 1 and overlap_flag in argv:
        argv.remove(overlap_flag)
    for key, value in {'--output': out, '--limit': len(rows), '--depth-batch-size': actors,
            '--gym-url': urls[0], '--gym-urls': json.dumps(urls), '--render-url': render_url,
            '--brain-url': brains[lane], '--http-concurrency': 64, '--actor-workers': min(8, actors),
            '--candidate-strategy': 'fixed', '--sampling-scheduler': 'v6_depth'}.items():
        set_option(argv, key, value)
    for flag in ('--rolling-actors', '--rolling-scene-refill'):
        if flag in argv: argv.remove(flag)
    if source == 'sage':
        set_option(argv, '--sage-split', 'train')
    else:
        set_option(argv, '--rolling-scene-limit', native_scene_cache_size)
        if '--rolling-scene-refill' in argv:
            argv.remove('--rolling-scene-refill')
        # Every instruction participates at depth 1; scene cache bounds
        # residency only and must never postpone or rolling-refill instructions.
        set_option(argv, '--cases', input_path)
        set_option(argv, '--split', 'train')
    launch = dict(schema='formal_local16_worker_v1', runtime=str(runtime), source_plan=str(source_plan),
        environment=owned_runtime_environment(runtime.parent),
        services=services, collectors=[collector], service_start_batch_size=8,
        allocation=dict(source=source, worker=index, lane=lane, visual_gpu=gpu if source == 'sage' else None,
            actors=actors, http_concurrency=64, actor_workers=min(8, actors), case_count=len(rows),
            rolling_scene_limit=0 if source == 'sage' else native_scene_cache_size,
            native_scene_cache_size=None if source == 'sage' else native_scene_cache_size,
            rolling_scene_refill=False, sampling_scheduler="v6_depth", rejected_depth_retries=1,
            scenes=len({r['scene'] for r in rows}), initial_scenes=len({r['scene'] for r in rows}),
            can_refill=False, instruction_sessions=len(rows)))
    save(wr/'launch.json', launch)
    return dict(source=source, worker=index, launch=str(wr/'launch.json'), collection=str(out),
                selection=str(out/'selection.json'), input=str(input_path), count=len(rows),
                identities=[list(identity(row)) for row in rows], allocation=launch['allocation'])


def prepare(root, population=DEFAULT_POPULATION, *, validation_limit=None, seed=20261003,
            sage_template=DEFAULT_SAGE_TEMPLATE, native_template=DEFAULT_NATIVE_TEMPLATE, brains=BRAINS,
            native_scene_cache_size=1, native_scene_refill=False, services_template=None):
    root, population = root.resolve(), population.resolve()
    if root.exists() or not root.is_relative_to(ART) or len(brains) != 4:
        raise ValueError('requires a new artifact directory and exactly four brain endpoints')
    report, populations = load_population(population)
    templates = {'sage': json.loads(sage_template.read_text()),
                 'native': json.loads(native_template.read_text())}
    floor_inputs, sage_native = sage_native_inputs(templates['sage'], populations['sage'],
                                                  report['sources']['sage']['manifest_sha256'])
    source_path = root/'source-plan.json'
    workers, failure_rows, selected_counts = [], {}, {}
    for source in SOURCES:
        selected, failures = select_validation(populations[source], validation_limit, seed)
        selected_counts[source] = len(selected)+len(failures)
        failure_rows[source] = failures
        for index, rows in enumerate(balance_scenes(selected, seed)):
            workers.append(worker_plan(root, source, index, rows,
                templates['sage' if source == 'sage' else 'native'], source_path, brains,
                native_scene_cache_size=native_scene_cache_size, native_scene_refill=native_scene_refill,
                sage_native=sage_native if source=='sage' else None))
    failures_path = root/'source-unexecutable.jsonl'
    save_rows(failures_path, [row for source in SOURCES for row in failure_rows[source]])
    managed_path = None
    if services_template is not None:
        managed_path = prepare_managed_services(root, workers, services_template, brains, source_path)
    frozen_inputs = {str(p): sha256(p) for p in (population, sage_template, native_template,
                     failures_path, REPO/'vln_demo/src/vln_demo.yaml',
                     REPO/'third_party/Fine-Grained-R2R-NOTICE.txt')}
    frozen_inputs.update(floor_inputs)
    if managed_path is not None:
        frozen_inputs.update({str(services_template.resolve()): sha256(services_template),
                              str(managed_path): sha256(managed_path)})
        managed = json.loads(managed_path.read_text())
        for service in managed['services']:
            query_checkpoint = service.get('environment', {}).get('ACTION_HEAD_QUERY_CHECKPOINT')
            if query_checkpoint:
                frozen_inputs[query_checkpoint] = sha256(query_checkpoint)
            for arg in service['command']:
                path = Path(arg)
                if path.is_absolute() and path.is_file() and path.suffix == '.py':
                    frozen_inputs[str(path)] = sha256(path)
            logging_path = service.get('environment', {}).get('VLLM_LOGGING_CONFIG_PATH')
            if logging_path:
                frozen_inputs[logging_path] = sha256(logging_path)
        nlp = next(service for service in managed['services'] if service['name'] == 'nlp')
        model = Path(nlp['command'][nlp['command'].index('--model')+1])
        for path in model.rglob('*'):
            if path.is_file():
                frozen_inputs[str(path)] = sha256(path)
    frozen_inputs[report['same_floor_scope']['route_audit']] = report['same_floor_scope']['route_audit_sha256']
    frozen_inputs.update({str(Path(s['manifest'])): s['manifest_sha256'] for s in report['sources'].values()})
    for worker in workers:
        frozen_inputs.update({worker[key]: sha256(worker[key]) for key in ('selection', 'input')})
    code = {str(p): sha256(p) for p in REPO.rglob('*.py') if '.runtime' not in p.parts}
    source = dict(schema='formal_rule_teacher_collection_source_v1', population=str(population),
        population_sha256=sha256(population), mode='validation' if validation_limit else 'full_population',
        seed=seed, validation_executable_limit_per_source=validation_limit, selected_counts=selected_counts,
        native_scene_cache_size=native_scene_cache_size, native_scene_refill=bool(native_scene_refill),
        source_unexecutable=sum(map(len, failure_rows.values())), source_sha256=report.get('source_sha256', {}),
        input_sha256=frozen_inputs, code_sha256=code,
        grouping='whole selected scene assigned once by descending size to least-loaded worker; scene contiguous',
        validation_selection='seed-ranked eligible scenes; equal 32 SAGE / 64 native quota at 1024/source; instruction-type strata within each scene',
        original_instructions=True, training_started=False)
    save(source_path, source)
    plan = dict(schema='formal_rule_teacher_collection_v1', output_root=str(root), source_plan=str(source_path),
        population=str(population), population_sha256=sha256(population), mode=source['mode'],
        workers=workers, source_unexecutable=str(failures_path), selected_counts=selected_counts,
        construction_report=str(root/'construction/formal-construction-report.json'))
    plan_path = root/'collection-plan.json'
    save(plan_path, plan)
    stages = []
    python = templates['sage']['collectors'][0]['command'][0]
    for backend in SOURCES:
        subset = [worker for worker in workers if worker['source'] == backend]
        stages.append(dict(kind='collect-'+backend, commands=[dict(name=f'{backend}-worker{w["worker"]}',
            command=[python, '-B', str(REPO/'scripts/train/supervise_rule_teacher_collection.py'), '--plan', w['launch']]) for w in subset],
            required_receipts=[dict(path=str(Path(w['launch']).parent/'completion.json'),
                equals={'status': 'complete', 'runtime_cleaned': True}) for w in subset],
            idle_gpu_ids=[] if backend == 'sage' else list(RENDER_GPUS)))
    aggregate_command = [python, '-B', str(Path(__file__).resolve()), 'aggregate', '--plan', str(plan_path)]
    stages.append(dict(kind='construction', commands=[dict(name='construction-report', command=aggregate_command)],
        required_receipts=[dict(path=plan['construction_report'], equals={'complete': True})],
        idle_gpu_ids=list(RENDER_GPUS)))
    phase_plan = dict(schema='formal_rule_teacher_collection_phases_v1', output_root=str(root),
        runtime=str(collection_runtime_root()/root.name/'phase-runner'),
        environment=owned_runtime_environment(collection_runtime_root()/root.name), stages=stages,
        input_sha256={str(source_path): sha256(source_path), str(plan_path): sha256(plan_path),
            **{w['launch']: sha256(w['launch']) for w in workers}, **code},
        services=[], poll_seconds=5, collection_plan=str(plan_path),
        gpu_reservation=dict(python='/usr/local/bin/python3.12',
            script=str(REPO/'verl/scripts/train/action_head_sft_data/gpu_occupier.py'), startup_timeout=120))
    save(root/'phase-plan.json', phase_plan)
    save(root/'preparation.json', dict(mode=source['mode'], population_counts=POPULATION,
        selected_counts=selected_counts, source_unexecutable=source['source_unexecutable'],
        workers=[w['allocation'] for w in workers],
        construction_command=aggregate_command,
        collect_command=[python, '-B', str(Path(__file__).resolve()), 'run-phases', '--plan', str(root/'phase-plan.json')],
        collection_executed=False, phase_order=list(SOURCES),
        ports_reused_serially=True, validation_scope='fixed seeded scene-quota subset plus all known source failures',
        scene_cache_note=('Every selected instruction initializes before the first same-depth batch; '
            'renderer cache limits residency without postponing instructions or rolling refill. '
            'Scene assignments and original case identities are unchanged.')))
    if managed_path is not None:
        preparation = json.loads((root/'preparation.json').read_text())
        preparation['managed_collect_command'] = [python, '-B',
            str(REPO/'scripts/train/supervise_rule_teacher_collection.py'), '--plan', str(managed_path)]
        preparation['managed_service_launch'] = str(managed_path)
        preparation['managed_brain_gpus'] = [[4*i, 4*i+1] for i in range(4)]
        preparation['managed_nlp_url'] = 'http://127.0.0.1:19100'
        save(root/'preparation.json', preparation)
    return plan


def prepare_managed_services(root, workers, template_path, brains, source_path):
    """Wrap the existing phase runner in the existing HTTP service supervisor.

    The outer supervisor owns four v6 four-query TP2 brains and one pinned NLP
    service. The unchanged phase runner owns only its eight per-source workers.
    Thus neither process needs to adopt or signal an external service.
    """
    if tuple(brains) != BRAINS:
        raise ValueError('managed services require the four fixed loopback brain endpoints')
    template = json.loads(template_path.read_text())
    selected = {name: [entry for entry in template['services'] if entry['name'] == name]
                for name in ('brain', 'nlp')}
    if any(len(entries) != 1 for entries in selected.values()):
        raise ValueError('managed service template requires exactly one brain and one NLP service')
    old_repo = next(Path(arg).parents[2] for entry in template['collectors']
                    for arg in entry['command'] if arg.endswith('_teacher_pilot.py'))
    folder = root/'managed-services'
    runtime = collection_runtime_root()/root.name/'managed-services'
    nlp = rebase_entry(selected['nlp'][0], old_repo)
    nlp_url = 'http://127.0.0.1:19100'
    set_option(nlp['command'], '--port', 19100)
    nlp.update(port=19100, health_url=nlp_url+'/health', log=str(folder/'nlp.log'))
    nlp['environment']['CUDA_VISIBLE_DEVICES'] = ''
    services = [nlp]
    for lane, url in enumerate(brains):
        entry = rebase_entry(selected['brain'][0], old_repo)
        argv = entry['command']
        if ('vllm.entrypoints.openai.api_server' not in argv
                or argv[argv.index('--tensor-parallel-size')+1] != '2'):
            raise ValueError('managed brains require the ordinary-text vLLM API and TP2')
        # Match the original v6 open-loop engine configuration, rather than
        # the prior MTP3/eager/max-seqs64 experiment template.
        for flag in ('--speculative-config', '--max-num-seqs', '--scheduler-cls'):
            if flag in argv:
                position = argv.index(flag)
                del argv[position:position+2]
        for flag in ('--async-scheduling', '--enforce-eager'):
            if flag in argv: argv.remove(flag)
        if '--no-async-scheduling' not in argv:
            argv.append('--no-async-scheduling')
        set_option(argv, '--gpu-memory-utilization', .7)
        set_option(argv, '--max-model-len', 32768)
        set_option(argv, '--max-num-batched-tokens', 8192)
        set_option(argv, '--worker-cls', 'action_head.serve.qwen_query_worker.ActionHeadQueryWorker')
        set_option(argv, '--logits-processors',
                   'action_head.serve.dualvln_query_runtime:DualVlnQueryLogitsProcessor')
        if '--enable-prefix-caching' not in argv:
            argv.append('--enable-prefix-caching')
        port = 29200+100*lane
        gpus = f'{4*lane},{4*lane+1}'
        set_option(argv, '--port', port)
        env = entry['environment']
        env['PYTHONPATH'] = ':'.join(str(REPO/name) for name in (
            'verl/action_head/serve/bootstrap', 'verl', 'vln_demo/vllm')) + (
            ':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        env.pop('VLLM_SCHEDULER_CLS', None)
        env.update(CUDA_VISIBLE_DEVICES=gpus, VLLM_GPUS=gpus, VLLM_PORT=str(port),
                   VLLM_MAX_NUM_SEQS='0', VLLM_MTP_TOKENS='0', VLLM_ENFORCE_EAGER='0',
                   VLLM_GPU_MEMORY_UTILIZATION='0.7', VLLM_MAX_NUM_BATCHED_TOKENS='8192',
                   ACTION_HEAD_ENABLE_QUERY_TAIL='1',
                   ACTION_HEAD_QUERY_CHECKPOINT=str(V6_QUERY_CHECKPOINT),
                   ACTION_HEAD_QUERY_CACHE=str(runtime/f'query-cache/brain-{lane}'),
                   ACTION_HEAD_QUERY_TOKEN_ID='248060', ACTION_HEAD_QUERY_EOS_TOKEN_ID='248046',
                   ACTION_HEAD_TOOL_END_TOKEN_IDS='248059',
                   ACTION_HEAD_RUN_TOOL_TOKEN_IDS='5917,7681,12756',
                   VLLM_RPC_BASE_PATH='@formal-'+rank(str(root), lane)[:16])
        if env.get('VLLM_LOGGING_CONFIG_PATH'):
            logging = folder/'brain-logging.json'
            save(logging, json.loads(Path(env['VLLM_LOGGING_CONFIG_PATH']).read_text()))
            env['VLLM_LOGGING_CONFIG_PATH'] = str(logging)
        entry.update(name=f'brain-{lane}', port=port, health_url=url+'/v1/models',
                     log=str(folder/f'brain-{lane}.log'), startup_group='brains')
        services.append(entry)
    for entry in services:
        entry['environment'].update(owned_runtime_environment(runtime))
    # Bind every source to the pinned service before any worker hashes freeze.
    for worker in workers:
        path = Path(worker['launch'])
        launch = json.loads(path.read_text())
        for entry in launch['collectors']:
            entry['environment']['RULE_NLP_URL'] = nlp_url
            entry['environment']['RULE_TEACHER_QUERY_CACHE'] = str(
                runtime/f"query-cache/brain-{worker['allocation']['lane']}")
        save(path, launch)
    python = template['collectors'][0]['command'][0]
    plan = dict(schema='formal_local16_managed_services_v1', runtime=str(runtime),
        source_plan=str(source_path),
        environment=owned_runtime_environment(runtime.parent), services=services,
        service_start_batch_size=4, collectors=[dict(name='formal-phases',
            command=[python, '-B', str(Path(__file__).resolve()), 'run-phases',
                     '--plan', str(root/'phase-plan.json')], environment={},
            log=str(folder/'phases.log'))])
    path = folder/'launch.json'
    save(path, plan)
    return path


def aggregate(plan_path):
    plan = json.loads(plan_path.read_text())
    source = json.loads(Path(plan['source_plan']).read_text())
    verify_plan(source)
    if sha256(plan['population']) != plan['population_sha256']:
        raise ValueError('population report changed')
    _, populations = load_population(Path(plan['population']))
    lookup = {identity(row): row for rows in populations.values() for row in rows}
    terminal, missing, seen = defaultdict(list), [], set()
    for line in Path(plan['source_unexecutable']).read_text().splitlines():
        row = json.loads(line)
        key = identity(row)
        if lookup[key] != row or row['status'] != 'source_unexecutable':
            raise ValueError('source failure differs from frozen population')
        terminal[row['dataset']].append(dict(dataset=row['dataset'], scene=row['scene'], task_id=row['task_id'],
            status='source_unexecutable', reason='; '.join(map(str, row['source_validation_errors'])), started=False))
        seen.add(key)
    for worker in plan['workers']:
        selection = json.loads(Path(worker['selection']).read_text())['cases']
        if len(selection) != len(worker['identities']):
            raise ValueError('worker selection size changed')
        for index, expected in enumerate(worker['identities']):
            key = tuple(expected)
            if key in seen or key not in lookup:
                raise ValueError('duplicate or foreign worker identity')
            seen.add(key)
            row = selection[index]
            if (row['scene'], str(row['task'])) != (key[1], key[2]):
                raise ValueError('selection identity changed')
            case = Path(worker['collection'])/f'case_{index:03d}'
            result_path = case/'result.json'
            basic = dict(dataset=key[0], scene=key[1], task_id=key[2], case=str(case))
            if not result_path.is_file():
                missing.append(dict(basic, status='interrupted' if case.exists() else 'unstarted'))
                continue
            result = json.loads(result_path.read_text())
            if (result.get('scene'), str(result.get('episode'))) != (key[1], key[2]):
                raise ValueError('result identity mismatch: '+str(case))
            if result.get('status') not in ('rollout_ended', 'rule_abstention', 'infrastructure_error'):
                raise ValueError('unknown result terminal status: '+str(case))
            failed = result['status'] != 'rollout_ended' or bool(result.get('infra_error'))
            terminal[key[0]].append(dict(basic, status='failed' if failed else 'completed', started=True,
                reason=str(result.get('infra_error') or result.get('reason') or result.get('why') or result['status']) if failed else '',
                result=str(result_path), result_sha256=sha256(result_path),
                terminal_success=bool(result.get('success')), accepted_ah=int(result.get('accepted_ah', 0)),
                calls_used=result.get('calls_used'), wall_seconds=result.get('wall_seconds')))
    if plan['mode'] == 'full_population' and seen != set(lookup):
        raise ValueError('full plan no longer covers exact frozen population')
    report_path = Path(plan['construction_report'])
    if report_path.exists() and json.loads(report_path.read_text()).get('complete'):
        raise ValueError('completed construction report is immutable')
    sources = {}
    for backend in SOURCES:
        path = report_path.parent/(backend+'.jsonl')
        save_rows(path, terminal[backend])
        sources[backend] = dict(manifest=str(path), manifest_sha256=sha256(path),
            population_count=POPULATION[backend], selected_count=plan['selected_counts'][backend],
            terminal_count=len(terminal[backend]), outcomes=dict(Counter(r['status'] for r in terminal[backend])),
            failure_reasons=dict(Counter(r['reason'] for r in terminal[backend] if r['status'] != 'completed')),
            accepted_ah=sum(r.get('accepted_ah', 0) for r in terminal[backend]),
            terminal_successes=sum(r.get('terminal_success', False) for r in terminal[backend]))
    save(report_path.parent/'missing.json', missing)
    report = dict(schema='formal_rule_teacher_construction_v1', mode=plan['mode'], complete=not missing,
        population=plan['population'], population_sha256=plan['population_sha256'],
        collection_plan=str(plan_path), collection_plan_sha256=sha256(plan_path),
        sources=sources, missing_count=len(missing), missing_report=str(report_path.parent/'missing.json'),
        finished_utc=utc(), navigation_success_required=False, per_anchor_admission_pending=True)
    save(report_path, report)
    return report


class CollectionPhases(Pipeline):
    """Reuse Pipeline lifecycle with collection-specific serial stage validation."""
    def __init__(self, plan, path):
        self.output, self.runtime = Path(plan['output_root']).resolve(), Path(plan['runtime']).resolve()
        allowed = collection_runtime_root()
        if (plan.get('schema') != 'formal_rule_teacher_collection_phases_v1'
                or self.output != path.resolve().parent or not self.output.is_relative_to(ART)
                or not self.runtime.is_relative_to(allowed) or self.runtime == allowed
                or [s['kind'] for s in plan['stages']] != ['collect-'+s for s in SOURCES]+['construction']
                or plan.get('services') or not plan.get('input_sha256')):
            raise ValueError('invalid formal collection phase boundary')
        for stage in plan['stages']:
            expected = 1 if stage['kind'] == 'construction' else 8
            expected_idle = [] if stage['kind'] == 'collect-sage' else RENDER_GPUS
            if (len(stage['commands']) != expected or not stage['required_receipts']
                    or stage.get('idle_gpu_ids', []) != expected_idle):
                raise ValueError('collection phases need eight owned workers then a gated construction report')
        if not plan.get('gpu_reservation'):
            raise ValueError('native and construction phases require the existing GPU reservation helper')
        self.plan, self.path, self.plan_sha256 = plan, path, sha256(path)
        self.owned, self.results = [], []
        self.phase, self.status, self.reservation = 'preflight', 'starting', None

    def run(self):
        code = super().run()
        if code:
            # Failed supervisors may leave genuine unstarted/interrupted work.
            # Persist that evidence without inventing terminal failure results.
            try:
                aggregate(Path(self.plan['collection_plan']))
            except Exception as exc:
                save(self.output/'construction-aggregation-error.json', dict(error=repr(exc), at_utc=utc()))
        return code


def run_phases(path):
    plan = json.loads(path.read_text())
    runner = CollectionPhases(plan, path)
    with (runner.output/'phase-runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((runner.output/name).exists() for name in ('status.json', 'completion.json')):
            raise ValueError('collection already started; inspect receipts instead of automatic replay')
        if runner.runtime.exists():
            token = os.environ.get('RULE_TEACHER_PIPELINE_LAUNCH_TOKEN')
            if (not token or read_status_json(runner.output/'runtime-owner.json')
                    != dict(token=token, plan_sha256=runner.plan_sha256)):
                raise ValueError('existing runtime does not belong to this phase launch')
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'collection phase runner received {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        return runner.run()


def launch_phases(path):
    """Detach only this phase adapter; reuse the established launch ownership."""
    plan = json.loads(path.read_text())
    runner = CollectionPhases(plan, path)
    verify_hashes(plan['input_sha256'])
    with (runner.output/'phase-launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (any((runner.output/name).exists() for name in ('launch.json', 'status.json', 'completion.json'))
                or runner.runtime.exists()):
            raise ValueError('refusing duplicate or implicit recovery phase launch')
        token = uuid.uuid4().hex
        save(runner.output/'runtime-owner.json', dict(token=token, plan_sha256=runner.plan_sha256))
        env = runtime_environment(runner.runtime, 'phase-runner', plan.get('environment', {}))
        env['RULE_TEACHER_PIPELINE_LAUNCH_TOKEN'] = token
        command = [sys.executable, '-B', str(Path(__file__).resolve()), 'run-phases', '--plan', str(path.resolve())]
        with (runner.output/'phase-runner.log').open('a') as stream:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = dict(command=command, identity=process_identity(process.pid),
            plan_sha256=runner.plan_sha256, started_utc=utc())
        save(runner.output/'launch.json', receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('--root', type=Path, required=True)
    prepare_parser.add_argument('--population', type=Path, default=DEFAULT_POPULATION)
    prepare_parser.add_argument('--validation-limit-per-source', type=int)
    prepare_parser.add_argument('--seed', type=int, default=20261003)
    prepare_parser.add_argument('--sage-template', type=Path, default=DEFAULT_SAGE_TEMPLATE)
    prepare_parser.add_argument('--native-template', type=Path, default=DEFAULT_NATIVE_TEMPLATE)
    prepare_parser.add_argument('--brain-urls', nargs=4, default=BRAINS)
    prepare_parser.add_argument('--native-scene-cache-size', type=int, default=1)
    prepare_parser.add_argument('--native-scene-refill', action='store_true')
    prepare_parser.add_argument('--services-template', type=Path,
        help='own four TP2/MTP3 brains and pinned NLP via the existing collection supervisor')
    for mode in ('aggregate', 'run-phases', 'launch-phases'):
        command = sub.add_parser(mode)
        command.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        value = prepare(args.root, args.population, validation_limit=args.validation_limit_per_source,
            seed=args.seed, sage_template=args.sage_template, native_template=args.native_template, brains=args.brain_urls,
            native_scene_cache_size=args.native_scene_cache_size, native_scene_refill=args.native_scene_refill,
            services_template=args.services_template)
        print(json.dumps(dict(root=str(args.root), mode=value['mode'], selected_counts=value['selected_counts'])))
    elif args.mode == 'aggregate':
        result = aggregate(args.plan)
        print(json.dumps(dict(complete=result['complete'], missing_count=result['missing_count'])))
        raise SystemExit(0 if result['complete'] else 1)
    elif args.mode == 'launch-phases':
        print(json.dumps(launch_phases(args.plan)))
    else:
        raise SystemExit(run_phases(args.plan))


if __name__ == '__main__':
    main()
