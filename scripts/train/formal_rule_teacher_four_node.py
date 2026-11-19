#!/usr/bin/env python3
"""Four-node data-only adapter over the frozen formal collection/export pipeline.

The sampler, native input resolution, rules, export, and process cleanup stay in
their existing frozen implementations. Only scene ownership and stage barriers
are added here. No model initialization or optimizer stage is reachable.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import fcntl
import importlib.util
import json
import os
import errno
from pathlib import Path
import signal
import shutil
import sys
import time

SELF = Path(__file__).resolve()
SOURCES = ('sage', 'r2r', 'rxr')
COUNTS = dict(sage=21638, r2r=10819, rxr=10819)
DATA_STAGES = ('collect', 'validate', 'export', 'assemble')
DISTRIBUTED_KEYS = ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE',
    'MASTER_ADDR', 'MASTER_PORT', 'GROUP_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE', 'NODE_RANK')


def selected_sources(config):
    values = config.get('collection_sources', list(SOURCES))
    if (not isinstance(values, list) or not values or len(set(values)) != len(values)
            or any(value not in SOURCES for value in values)):
        raise ValueError('collection_sources must select unique original dataset sources')
    return tuple(source for source in SOURCES if source in values)


def native_gym_slots(config, source):
    values = config.get('native_gym_slots', {})
    if (not isinstance(values, dict) or any(k not in ('r2r', 'rxr') for k in values)
            or any(type(v) is not int or v not in (8, 16, 32, 64) for v in values.values())):
        raise ValueError('native_gym_slots must bind 8/16/32/64 physical servers per native worker')
    return values.get(source, 64)


def bind_native_gym_slots(launch, config, source):
    """Bound physical HTTP hosts while retaining every independent instruction session.

    Candidate count, HTTP/actor concurrency and depth-batch metadata stay fixed.
    """
    slots = native_gym_slots(config, source)
    if source == 'sage' or slots == 64:
        return launch
    gyms = [e for e in launch['services'] if e['name'].startswith(source+'-gym-')]
    if len(gyms) != 64:
        raise ValueError('slot override requires the original 64-server native template')
    keep = {e['name'] for e in gyms[:slots]}
    launch['services'] = [e for e in launch['services']
                          if not e['name'].startswith(source+'-gym-') or e['name'] in keep]
    for entry in launch['collectors']:
        argv = entry['command']
        index = argv.index('--gym-urls')+1
        urls = json.loads(argv[index])
        if len(urls) != 64:
            raise ValueError('native template URL count differs from physical Gym servers')
        argv[index] = json.dumps(urls[:slots])
    launch['allocation']['actors'] = slots
    launch['allocation']['physical_gym_servers'] = slots
    return launch


def native_renderer_bindings(config, sha):
    overlay = config.get('native_renderer_overlay')
    if overlay is None:
        return {}
    root = Path(overlay['root']).resolve(strict=True)
    files = overlay['files']
    required = {str(root/'remote_habitat'/name) for name in
                ('__init__.py', 'server.py', 'worker.py', 'client.py', 'policy.py')}
    if not isinstance(files, dict) or set(files) != required:
        raise ValueError('native renderer overlay must bind its complete private package')
    for path, expected in files.items():
        if not Path(path).resolve(strict=True).is_relative_to(root) or sha(path) != expected:
            raise ValueError('native renderer overlay changed: '+path)
    return dict(files)


def bind_native_renderer_overlay(launch, config, source):
    overlay = config.get('native_renderer_overlay')
    if source == 'sage' or overlay is None:
        return
    services = [s for s in launch['services'] if s['name'] == source+'-render']
    if len(services) != 1:
        raise ValueError('native renderer overlay requires one owned render service')
    env = services[0]['environment']
    previous = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = overlay['root'] + (os.pathsep+previous if previous else '')
    launch['native_renderer_overlay'] = copy.deepcopy(overlay)


def native_gym_bindings(config, sha):
    overlay = config.get('native_gym_overlay')
    if overlay is None:
        return {}
    root = Path(overlay['root']).resolve(strict=True)
    package = root/'vln_gym'
    files = overlay['files']
    actual = {str(p) for p in package.rglob('*') if p.is_file()}
    if (not isinstance(files, dict) or set(files) != actual
            or not {str(package/name) for name in ('__init__.py', 'env.py', 'server.py',
                                                  'vlnce_action_head.py')} <= actual):
        raise ValueError('native Gym overlay must bind its complete private package')
    for path, expected in files.items():
        if not Path(path).resolve(strict=True).is_relative_to(root) or sha(path) != expected:
            raise ValueError('native Gym overlay changed: '+path)
    return dict(files)


def bind_native_gym_overlay(launch, config, source):
    overlay = config.get('native_gym_overlay')
    if source == 'sage' or overlay is None:
        return
    gyms = [s for s in launch['services'] if s['name'].startswith(source+'-gym-')]
    if not gyms:
        raise ValueError('native Gym overlay requires owned Gym services')
    for service in gyms:
        env = service['environment']
        previous = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = overlay['root'] + (os.pathsep+previous if previous else '')
    launch['native_gym_overlay'] = copy.deepcopy(overlay)


def sage_source_reference_bindings(config, sha):
    """Bind the source-only official map; old frozen configs remain unchanged."""
    binding = config.get('sage_source_object_map')
    if binding is None:
        return {}
    path = Path(binding['path']).resolve(strict=True)
    if sha(path) != binding['sha256']:
        raise ValueError('SAGE source object map changed')
    manifest = load(path)
    if manifest.get('schema') != 'sage-official-source-object-map-v1':
        raise ValueError('invalid SAGE source object map schema')
    result = {str(path): binding['sha256']}
    for scene in manifest['scenes'].values():
        file = (path.parent/scene['file']).resolve(strict=True)
        if not file.is_relative_to(path.parent) or sha(file) != scene['sha256']:
            raise ValueError('SAGE source object scene binding changed')
        result[str(file)] = scene['sha256']
    return result


def sage_source_reference_environment(config):
    binding = config.get('sage_source_object_map')
    return (dict(SAGE_SOURCE_OBJECT_MAP=str(Path(binding['path']).resolve()),
                 SAGE_SOURCE_OBJECT_MAP_SHA256=binding['sha256'],
                 SAGE_REQUIRE_SOURCE_OBJECT_MAP='1') if binding is not None else {})


def bind_sage_source_objects(launch, config, source):
    if source != 'sage' or config.get('sage_source_object_map') is None:
        return
    environment = sage_source_reference_environment(config)
    launch.setdefault('environment', {}).update(environment)
    for service in launch['services']:
        if service['name'].startswith('sage-gym-'):
            service.setdefault('environment', {}).update(environment)
    launch['sage_source_object_map'] = copy.deepcopy(config['sage_source_object_map'])


def stage_kinds(config, node):
    nodes = config.get('collection_only_nodes', [])
    if not isinstance(nodes, list) or any(type(n) is not int or n not in range(4) for n in nodes):
        raise ValueError('collection_only_nodes must name physical node ranks')
    return ('collect', 'validate') if node in nodes else DATA_STAGES


def gpu_permutation(config, node):
    raw = config.get('node_gpu_permutations', {}).get(str(node), {})
    mapping = {int(k): v for k, v in raw.items()}
    if any(k not in range(16) or type(v) is not int or v not in range(16) for k, v in mapping.items()):
        raise ValueError('GPU permutation must name physical cards 0..15')
    if len({mapping.get(i, i) for i in range(16)}) != 16:
        raise ValueError('GPU mapping must be a permutation, without shared cards')
    return mapping


def render_gpu_overrides(config, node):
    mapping = gpu_permutation(config, node)
    model_cards = {mapping.get(i, i) for i in (0, 1, 4, 5, 8, 9, 12, 13)}
    render_cards = set(range(16))-model_cards
    raw = config.get('node_render_gpu_overrides', {}).get(str(node), {})
    overrides = {int(k): v for k, v in raw.items()}
    if any(k not in render_cards or type(v) is not int or v not in render_cards for k, v in overrides.items()):
        raise ValueError('render overrides must stay separate from the model cards')
    return overrides


def unused_collection_gpu_ids(config, node):
    mapping = gpu_permutation(config, node)
    overrides = render_gpu_overrides(config, node)
    models = {mapping.get(i, i) for i in (0, 1, 4, 5, 8, 9, 12, 13)}
    renders = {overrides.get(mapping.get(i, i), mapping.get(i, i)) for i in (2, 3, 6, 7, 10, 11, 14, 15)}
    return sorted(set(range(16))-models-renders)


def remap_gpus(value, mapping):
    """Change physical allocation only, keeping model and sampling arguments."""
    def ids(text):
        return ','.join(str(mapping.get(int(i), int(i))) for i in text.split(',')) if text else text
    if isinstance(value, dict):
        result = {k: remap_gpus(v, mapping) for k, v in value.items()}
        for key in ('visual_gpu',):
            if key in result:
                result[key] = mapping.get(value[key], value[key])
        for key in ('CUDA_VISIBLE_DEVICES', 'VLLM_GPUS'):
            if key in result:
                result[key] = ids(value[key])
        if 'idle_gpu_ids' in result:
            result['idle_gpu_ids'] = sorted(mapping.get(i, i) for i in value['idle_gpu_ids'])
        return result
    if isinstance(value, list):
        result = [remap_gpus(v, mapping) for v in value]
        for index, item in enumerate(value[:-1]):
            if item in ('--gpus', '--gpu', '--gpu-id', '--gpu_id'):
                result[index+1] = ids(value[index+1])
        return result
    return value


def load(path):
    for attempt in range(4):
        try:
            return json.loads(Path(path).read_text())
        except OSError as exc:
            if exc.errno != errno.ESTALE or attempt == 3:
                raise
            time.sleep(.2)


def modules(config):
    path = Path(config['composition'])
    name = 'formal_rule_teacher_pipeline'
    descriptor = importlib.util.spec_from_file_location(name, path)
    composition = importlib.util.module_from_spec(descriptor)
    sys.modules[name] = composition
    descriptor.loader.exec_module(composition)
    adapter = composition.pipeline_adapter()
    runner = adapter.configure(config['source_root'])
    import prepare_rule_teacher_formal_local16_collection as collection
    return composition, adapter, runner, collection


def scene_parts(identities):
    groups = defaultdict(list)
    for index, identity in enumerate(identities):
        groups[identity[1]].append(index)
    if len(groups) < 4:
        raise ValueError('each original worker must cover at least four scenes')
    parts = [[] for _ in range(4)]
    for scene in sorted(groups, key=lambda s: (-len(groups[s]), s)):
        node = min(range(4), key=lambda n: (len(parts[n]), n))
        parts[node].extend(groups[scene])
    return [sorted(part) for part in parts]


def rebase(value, old, new, old_runtime, new_runtime):
    if isinstance(value, str):
        return value.replace(str(old), str(new)).replace(old_runtime, new_runtime)
    if isinstance(value, list):
        return [rebase(v, old, new, old_runtime, new_runtime) for v in value]
    if isinstance(value, dict):
        return {rebase(k, old, new, old_runtime, new_runtime):
                rebase(v, old, new, old_runtime, new_runtime) for k, v in value.items()}
    return value


def copy_service_inputs(original, rebased, old, new, sha):
    """Package template-local service inputs before freezing the launch plan."""
    bindings = {}
    for before, after in zip(original['services'], rebased['services'], strict=True):
        for key, value in before.get('environment', {}).items():
            if key in ('VLLM_LOGGING_CONFIG_PATH', 'ACTION_HEAD_QUERY_CHECKPOINT') and not Path(value).is_file():
                raise ValueError('missing service input '+key+': '+value)
        pairs = list(zip(before['command'], after['command'], strict=True))
        pairs += [(value, after['environment'][key]) for key, value in before.get('environment', {}).items()]
        for source, target in pairs:
            path = Path(source)
            if not path.is_absolute() or not path.is_relative_to(old) or not path.is_file():
                continue
            destination = Path(target)
            if not destination.is_relative_to(new):
                raise ValueError('service input escaped new node directory')
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                with path.open('rb') as src, destination.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
            if sha(path) != sha(destination):
                raise ValueError('copied service input differs: '+target)
            bindings[target] = sha(destination)
    return bindings


def verify_service_inputs(launch, frozen):
    for service in launch['services']:
        for key in ('VLLM_LOGGING_CONFIG_PATH', 'ACTION_HEAD_QUERY_CHECKPOINT'):
            value = service.get('environment', {}).get(key)
            if value and (not Path(value).is_file() or value not in frozen):
                raise ValueError('missing or unfrozen service input '+key+': '+value)


def prepare(config_path):
    config = load(config_path)
    dataset_sources = selected_sources(config)
    native_gym_slots(config, 'rxr')
    c, adapter, runner, col = modules(config)
    renderer_bindings = {**native_renderer_bindings(config, c.sha), **native_gym_bindings(config, c.sha),
                         **sage_source_reference_bindings(config, c.sha)}
    root = Path(config['output_root'])
    old = Path(config['template_root'])
    base = load(old/'spec.json')
    c.checked(base['population'])
    col.verify_plan(load(old/'collection/source-plan.json'))
    gate = c.validate_gates(config['source_root'], **base['gates'])
    population, rows = col.load_population(Path(base['population']['path']))
    full = {col.identity(row): row for source in dataset_sources for row in rows[source]}
    old_plan = load(old/'collection/collection-plan.json')
    old_plan['workers'] = [w for w in old_plan['workers'] if w['source'] in dataset_sources]
    splits = [scene_parts(w['identities']) for w in old_plan['workers']]
    failures = [row for s in Path(old_plan['source_unexecutable']).read_text().splitlines() if s
                for row in [json.loads(s)] if row['dataset'] in dataset_sources]
    scene_owner = {}
    for w, parts in zip(old_plan['workers'], splits):
        for node, indices in enumerate(parts):
            for i in indices:
                key = (w['source'], w['identities'][i][1])
                if key in scene_owner and scene_owner[key] != node:
                    raise ValueError('scene assigned to more than one node')
                scene_owner[key] = node
    failure_parts = [[] for _ in range(4)]
    for row in failures:
        node = scene_owner.get((row['dataset'], row['scene']))
        if node is None:
            node = min(range(4), key=lambda n: (len(failure_parts[n]), n))
        failure_parts[node].append(row)
    snapshot = root/'source'/SELF.name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    with snapshot.open('x') as stream:
        stream.write(SELF.read_text())
    wrapper = str(snapshot)
    python = base['python']
    old_runtime = load(old/'collection/phase-plan.json')['environment']['RULE_COLLECTION_RUNTIME_ROOT']
    old_runtime = old_runtime.removesuffix('/collection')
    global_root = root/'global'
    nodes, all_workers, partitions, seen = [], [], [], set()
    for node in range(4):
        nr = root/f'node{node}'
        if nr.exists():
            raise ValueError('new node attempt directory required')
        runtime = config['runtime_root']+f'/node{node}'
        rb = lambda value: rebase(value, old, nr, old_runtime, runtime)
        mapping = gpu_permutation(config, node)
        render_overrides = render_gpu_overrides(config, node)
        workers = []
        for w, parts in zip(old_plan['workers'], splits):
            indices = parts[node]
            keys = [tuple(w['identities'][i]) for i in indices]
            if not keys or seen.intersection(keys):
                raise ValueError('empty or overlapping instruction partition')
            seen.update(keys)
            selection = load(w['selection'])
            inputs = load(w['input'])
            for i in indices:
                selected = selection['cases'][i]
                if (selected['scene'], str(selected['task'])) != tuple(w['identities'][i])[1:]:
                    raise ValueError('original selection identity mismatch')
            nw = rb(w)
            nw['identities'] = [list(key) for key in keys]
            nw['count'] = len(keys)
            allocation = nw['allocation']
            allocation.update(remap_gpus(allocation, mapping))
            if nw['source'] == 'sage':
                allocation.update(remap_gpus(allocation, render_overrides))
            allocation.update(case_count=len(keys), instruction_sessions=len(keys),
                scenes=len({key[1] for key in keys}), initial_scenes=len({key[1] for key in keys}))
            selection = rb(selection)
            selection['cases'] = [selection['cases'][i] for i in indices]
            inputs = rb(inputs)
            inputs['cases'] = [inputs['cases'][i] for i in indices]
            c.save(nw['selection'], selection)
            c.save(nw['input'], inputs)
            launch = remap_gpus(rb(load(w['launch'])), mapping)
            if nw['source'] == 'sage':
                launch = remap_gpus(launch, render_overrides)
            launch['allocation'] = allocation
            bind_native_gym_slots(launch, config, nw['source'])
            bind_native_renderer_overlay(launch, config, nw['source'])
            bind_native_gym_overlay(launch, config, nw['source'])
            bind_sage_source_objects(launch, config, nw['source'])
            for entry in launch['collectors']:
                argv = entry['command']
                argv[argv.index('--limit')+1] = str(len(keys))
            c.save(nw['launch'], launch)
            workers.append(nw)
            all_workers.append(nw)
            name = f'node{node}-{w["source"]}-worker{w["worker"]}'
            partitions.append(dict(name=name, source=w['source'], node=node,
                collection=nw['collection'], review=str(global_root/'reviews'/(name+'.json')),
                output=str(nr/'exports'/f'{w["source"]}-worker{w["worker"]}'), lane=len(workers[:-1]) % 16))
        for row in failure_parts[node]:
            key = col.identity(row)
            if key in seen or full.get(key) != row:
                raise ValueError('duplicate or changed original source failure')
            seen.add(key)
        fp = nr/'collection/source-unexecutable.jsonl'
        col.save_rows(fp, failure_parts[node])
        counts = {source: sum(w['count'] for w in workers if w['source'] == source)
                  +sum(r['dataset'] == source for r in failure_parts[node]) for source in SOURCES}
        source = rb(load(old/'collection/source-plan.json'))
        # Reuse unchanged external frozen dependencies; bind only newly generated
        # files after rebasing, rather than copying stale hashes of old plans.
        source['input_sha256'] = {p: h for p, h in load(old/'collection/source-plan.json')['input_sha256'].items()
                                  if not Path(p).is_relative_to(old)}
        source['input_sha256'][str(fp)] = c.sha(fp)
        source['input_sha256'].update(renderer_bindings)
        for w in workers:
            source['input_sha256'].update({w[k]: c.sha(w[k]) for k in ('selection', 'input')})
        source['selected_counts'] = counts
        source['source_unexecutable'] = len(failure_parts[node])
        source['four_node_scope'] = dict(node=node, original_counts=COUNTS,
            partition_only=True, selected_sources=list(dataset_sources), wrapper=c.binding(snapshot))
        c.save(nr/'collection/source-plan.json', source)
        cp = rb(old_plan)
        cp.update(workers=workers, selected_counts=counts, four_node_scope=source['four_node_scope'])
        c.save(nr/'collection/collection-plan.json', cp)
        spec = rb(base)
        spec.update(partitions=[p for p in partitions if p['node'] == node],
                    collection_plan=c.binding(nr/'collection/collection-plan.json'),
                    four_node_scope=source['four_node_scope'], training_control=dict(mode='data_only'))
        c.save(nr/'spec.json', spec)
        phase = remap_gpus(rb(load(old/'collection/phase-plan.json')), mapping)
        phase['environment'].update(sage_source_reference_environment(config))
        phase['stages'] = [s for s in phase['stages']
                           if s['kind'] in ['collect-'+source for source in dataset_sources]+['construction']]
        phase['stages'][-1]['commands'][0]['command'] = [python, '-B', wrapper, '--config', str(config_path),
            '--node', str(node), '--mode', 'aggregate-node']
        frozen = {**source['input_sha256'], **source['code_sha256'], **source['source_sha256'],
                  str(nr/'collection/source-plan.json'): c.sha(nr/'collection/source-plan.json'),
                  str(nr/'collection/collection-plan.json'): c.sha(nr/'collection/collection-plan.json'),
                  str(snapshot): c.sha(snapshot), str(config_path): c.sha(config_path)}
        frozen.update({w['launch']: c.sha(w['launch']) for w in workers})
        phase['input_sha256'] = frozen
        c.save(nr/'collection/phase-plan.json', phase)
        # The ready sentinel remains the frozen implementation. It does not
        # collect cases or enter the original six-stage training runner.
        original_brains = load(old/'brains/launch.json')
        brains = remap_gpus(rb(original_brains), mapping)
        service_inputs = copy_service_inputs(original_brains, brains, old, nr, c.sha)
        c.save(nr/'brains/launch.json', brains)
        pipeline = rb(load(old/'pipeline-plan.json'))
        pipeline.setdefault('environment', {}).update(sage_source_reference_environment(config))
        pipeline['input_sha256'] = {**frozen, **service_inputs,
            str(nr/'spec.json'): c.sha(nr/'spec.json'),
            str(nr/'collection/phase-plan.json'): c.sha(nr/'collection/phase-plan.json'),
            str(nr/'brains/launch.json'): c.sha(nr/'brains/launch.json')}
        pipeline['input_sha256'].update({p: h for p, h in load(old/'pipeline-plan.json')['input_sha256'].items()
                                        if not Path(p).is_relative_to(old)})
        def cmd(mode, *extra):
            return [python, '-B', wrapper, '--config', str(config_path), '--node', str(node), '--mode', mode, *extra]
        stages = pipeline['stages'][:4]
        stages[0]['commands'] = [dict(name='three-source-collection', command=cmd('run-collection'))]
        stages[1]['commands'] = [dict(name='global-admission-barrier', command=cmd('validate-node'))]
        stages[1]['required_receipts'] = [c.receipt(nr/'receipts/validate.json', ok=True)]
        stages[2]['commands'] = [dict(name=f'export-input-{lane:02d}', command=cmd('export-node', '--lane', str(lane)),
            environment={'CUDA_VISIBLE_DEVICES': ''}) for lane in range(16)]
        stages[3]['commands'] = [dict(name='global-assembly-barrier', command=cmd('assemble-node'))]
        stages[3]['required_receipts'] = [c.receipt(nr/'receipts/assemble.json', ok=True, data_only=True)]
        if node in config.get('collection_only_nodes', []):
            # A failed node can recollect in a new attempt while peers continue
            # writing their original outputs. This receipt proves collection
            # and service cleanup only; it never certifies training readiness.
            stages = stages[:2]
            stages[1]['commands'] = [dict(name='collection-recovery-completion', command=cmd('finish-collection'))]
            stages[1]['required_receipts'] = [c.receipt(nr/'receipts/collection-ready.json',
                ok=True, collection_only=True, training_ready=False)]
        pipeline['stages'] = stages
        c.save(nr/'pipeline-plan.json', pipeline)
        nodes.append(dict(node=node, root=str(nr), counts=counts, stage_kinds=list(stage_kinds(config, node)),
            plan=c.binding(nr/'pipeline-plan.json')))
    expected_counts = {source: COUNTS[source] for source in dataset_sources}
    if seen != set(full) or Counter(key[0] for key in seen) != Counter(expected_counts):
        raise ValueError('four-node union must equal the complete original population')
    # Global collection aggregation still uses the unchanged full-population
    # validator; source files and all 43,276 original objects remain bound.
    global_root.mkdir(parents=True, exist_ok=True)
    gsource = copy.deepcopy(load(old/'collection/source-plan.json'))
    gsource['input_sha256'][str(snapshot)] = c.sha(snapshot)
    c.save(global_root/'collection-source-plan.json', gsource)
    gf = global_root/'source-unexecutable.jsonl'
    col.save_rows(gf, failures)
    gp = dict(old_plan, output_root=str(global_root), workers=all_workers,
        source_plan=str(global_root/'collection-source-plan.json'), source_unexecutable=str(gf),
        construction_report=str(global_root/'construction/formal-construction-report.json'))
    c.save(global_root/'collection-plan.json', gp)
    gs = rebase(base, old, global_root, old_runtime, config['runtime_root']+'/global')
    gs.update(partitions=partitions, collection_plan=c.binding(global_root/'collection-plan.json'),
        training_control=dict(mode='data_only'))
    c.save(global_root/'spec.json', gs)
    proof = dict(schema='formal_rule_teacher_four_node_preparation_v1', ok=True,
        original_counts=COUNTS, exact_original_union=True, duplicate_instructions=0,
        selected_sources=list(dataset_sources), selected_counts=expected_counts,
        source_failures_preserved=len(failures), partitions=len(partitions), nodes=nodes,
        policy_sha256=gate['policy_sha256'], wrapper=c.binding(snapshot),
        sampler_modified=False, training_stages_present=False)
    c.save(root/'preparation.json', proof)
    verify(config_path)
    return proof


def verify(config_path):
    config = load(config_path)
    selected = selected_sources(config)
    c, _, runner, col = modules(config)
    root = Path(config['output_root'])
    proof = load(root/'preparation.json')
    if c.binding(proof['wrapper']['path']) != proof['wrapper']:
        raise ValueError('frozen four-node adapter changed')
    if (proof['original_counts'] != COUNTS or proof['partitions'] != 32*len(selected)
            or tuple(proof.get('selected_sources', SOURCES)) != selected):
        raise ValueError('four-node scope mismatch')
    _, populations = col.load_population(Path(load(root/'global/spec.json')['population']['path']))
    wanted = {col.identity(r) for source in selected for r in populations[source]}
    actual = set()
    runner.KINDS = DATA_STAGES
    original_verify = runner.verify_hashes
    verified = {}
    def verify_once(values):
        for path, expected in values.items():
            if path in verified and verified[path] != expected:
                raise ValueError('conflicting frozen hash across nodes: '+path)
        original_verify({path: expected for path, expected in values.items() if path not in verified})
        verified.update(values)
    # All four nodes bind the same large original assets. Read each unique
    # immutable input once during this single preflight, still rejecting any
    # node that binds different bytes. Production stage checks stay unchanged.
    runner.verify_hashes = verify_once
    try:
        return verify_nodes(proof, wanted, runner, c, col, actual, config)
    finally:
        runner.verify_hashes = original_verify


def verify_nodes(proof, wanted, runner, c, col, actual, config=None):
    native_renderer_bindings(config or {}, c.sha)
    gym_bindings = native_gym_bindings(config or {}, c.sha)
    source_bindings = sage_source_reference_bindings(config or {}, c.sha)
    for item in proof['nodes']:
        plan = c.checked(item['plan'])
        if any(plan['input_sha256'].get(path) != digest for path, digest in gym_bindings.items()):
            raise ValueError('Gym overlay is not bound by the pipeline')
        if any(plan['input_sha256'].get(path) != digest for path, digest in source_bindings.items()):
            raise ValueError('SAGE source object map is not bound by the pipeline')
        source_env = sage_source_reference_environment(config or {})
        if any(plan.get('environment', {}).get(key) != value for key,value in source_env.items()):
            raise ValueError('SAGE source object map missing from validation environment')
        verify_service_inputs(load(Path(item['root'])/'brains/launch.json'), plan['input_sha256'])
        expected = tuple(item.get('stage_kinds', DATA_STAGES))
        if expected not in (DATA_STAGES, ('collect', 'validate')):
            raise ValueError('unsupported data recovery stages')
        runner.KINDS = expected
        runner.validate_plan(plan, Path(item['plan']['path']))
        phase_path = Path(item['root'])/'collection/phase-plan.json'
        # Exercise the actual child constructor as well as the outer plan.
        # It validates a different schema and output/runtime boundary.
        previous_render = col.RENDER_GPUS
        previous_sources = col.SOURCES
        try:
            col.SOURCES = selected_sources(config or {})
            col.RENDER_GPUS = sorted(gpu_permutation(config or {}, item['node']).get(i, i)
                                    for i in (2, 3, 6, 7, 10, 11, 14, 15))
            col.CollectionPhases(load(phase_path), phase_path)
        finally:
            col.RENDER_GPUS = previous_render
            col.SOURCES = previous_sources
        if tuple(s['kind'] for s in plan['stages']) != expected:
            raise ValueError('only the four data stages may run')
        cp = load(Path(item['root'])/'collection/collection-plan.json')
        selected = selected_sources(config or {})
        if (Counter(w['source'] for w in cp['workers']) != Counter({s:8 for s in selected})
                or any(sorted(w['worker'] for w in cp['workers'] if w['source'] == s)
                       != list(range(8)) for s in selected)):
            raise ValueError('each selected source needs its eight original workers per node')
        keys = [tuple(k) for w in cp['workers'] for k in w['identities']]
        keys += [col.identity(json.loads(s)) for s in Path(cp['source_unexecutable']).read_text().splitlines() if s]
        if len(set(keys)) != len(keys) or actual.intersection(keys):
            raise ValueError('overlapping node input')
        actual.update(keys)
        for w in cp['workers']:
            launch = load(w['launch'])
            if w['source'] == 'sage' and source_env:
                environments = [launch.get('environment', {})]+[
                    service.get('environment', {}) for service in launch['services']
                    if service['name'].startswith('sage-gym-')]
                if (launch.get('sage_source_object_map') != config['sage_source_object_map']
                        or any(any(env.get(key) != value for key,value in source_env.items())
                               for env in environments)):
                    raise ValueError('SAGE teacher/Gym source reference bindings differ')
            argv = launch['collectors'][0]['command']
            a = w['allocation']
            if (a['sampling_scheduler'] != 'v6_depth' or a['rejected_depth_retries'] != 1
                    or a['http_concurrency'] != 64 or a['actor_workers'] != 8 or a['can_refill']
                    or argv[argv.index('--candidates')+1] != '8'
                    or int(argv[argv.index('--limit')+1]) != len(w['identities'])):
                raise ValueError('frozen v6 sampling contract changed')
            if w['source'] in ('r2r', 'rxr'):
                slots = native_gym_slots(config or {}, w['source'])
                gyms = [e for e in launch['services'] if e['name'].startswith(w['source']+'-gym-')]
                urls = json.loads(argv[argv.index('--gym-urls')+1])
                if len(gyms) != slots or len(urls) != slots or a['actors'] != slots:
                    raise ValueError('native physical Gym server ownership differs from configuration')
                gym_overlay = (config or {}).get('native_gym_overlay')
                if gym_overlay is not None and (launch.get('native_gym_overlay') != gym_overlay or any(
                        service['environment'].get('PYTHONPATH', '').split(os.pathsep)[0] != gym_overlay['root']
                        for service in gyms)):
                    raise ValueError('native Gym does not load the bound private package')
                overlay = (config or {}).get('native_renderer_overlay')
                if overlay is not None:
                    services = [s for s in launch['services'] if s['name'] == w['source']+'-render']
                    if (launch.get('native_renderer_overlay') != overlay or len(services) != 1
                            or services[0]['environment'].get('PYTHONPATH', '').split(os.pathsep)[0]
                            != overlay['root']):
                        raise ValueError('native renderer does not load the bound isolated package')
    if actual != wanted:
        raise ValueError('node inputs differ from complete original population')
    return dict(ok=True, instructions=len(actual), partitions=proof['partitions'],
        selected_sources=list(selected_sources(config or {})),
        nodes=[n['counts'] for n in proof['nodes']], training_started=False)


def rank_from_environment(env, hostname):
    # PyTorch DLC assigns one Pod per node, not one process per physical GPU.
    candidates = []
    for key in ('NODE_RANK', 'RANK', 'GROUP_RANK'):
        if key in env:
            candidates.append(int(env[key]))
    if hostname.endswith('-master-0'):
        candidates.append(0)
    elif '-worker-' in hostname:
        candidates.append(int(hostname.rsplit('-worker-', 1)[1])+1)
    if not candidates or len(set(candidates)) != 1 or not 0 <= candidates[0] < 4:
        raise ValueError('cannot unambiguously bind DLC Pod to node rank: '+str(candidates))
    return candidates[0]


def local_service_environment(env):
    """Each brain/actor is node-local; DLC's job ranks only select ownership."""
    return {key: value for key, value in env.items() if key not in DISTRIBUTED_KEYS}


def wait_for(path, root):
    while not path.exists():
        for node in range(4):
            failed = root/f'node{node}/entry-failure.json'
            if failed.exists():
                raise RuntimeError('another node failed: '+str(failed))
            completion = root/f'node{node}/completion.json'
            if completion.exists() and load(completion)['status'] != 'complete':
                raise RuntimeError('another node pipeline failed: '+str(completion))
        time.sleep(5)
    result = load(path)
    if result.get('ok') is False:
        raise RuntimeError('failed stage receipt: '+str(path))
    return result


def aggregate_node(config_path, node):
    config = load(config_path)
    c, _, _, col = modules(config)
    nr = Path(config['output_root'])/f'node{node}'
    cp = load(nr/'collection/collection-plan.json')
    original_load = col.load_population
    def scoped(path):
        previous = col.POPULATION
        try:
            col.POPULATION = COUNTS
            report, rows = original_load(path)
        finally:
            col.POPULATION = previous
        keys = {tuple(k) for w in cp['workers'] for k in w['identities']}
        keys.update(col.identity(json.loads(s)) for s in Path(cp['source_unexecutable']).read_text().splitlines() if s)
        return report, {source: [r for r in rows[source] if col.identity(r) in keys] for source in SOURCES}
    _, subset = scoped(Path(cp['population']))
    col.load_population = scoped
    col.POPULATION = {s: len(subset[s]) for s in SOURCES}
    result = col.aggregate(nr/'collection/collection-plan.json')
    if not result['complete']:
        raise ValueError('node partition contains unfinished original instructions')
    return result


def phase(config_path, node, mode, lane=None):
    config = load(config_path)
    c, adapter, runner, col = modules(config)
    root = Path(config['output_root'])
    nr = root/f'node{node}'
    spec = load(nr/'spec.json')
    gs = load(root/'global/spec.json')
    if mode == 'run-collection':
        previous_render = col.RENDER_GPUS
        previous_sources = col.SOURCES
        original_phases = col.CollectionPhases
        try:
            col.SOURCES = selected_sources(config)
            col.RENDER_GPUS = sorted(gpu_permutation(config, node).get(i, i)
                                    for i in (2, 3, 6, 7, 10, 11, 14, 15))
            spare = unused_collection_gpu_ids(config, node)
            if spare:
                class CollectionWithUnusedCards(original_phases):
                    def reserve(self, stage):
                        if stage['kind'] == 'collect-sage':
                            stage = dict(stage, idle_gpu_ids=spare)
                        return super().reserve(stage)
                col.CollectionPhases = CollectionWithUnusedCards
            code = adapter.run_collection(nr/'collection/phase-plan.json', runner)
        finally:
            col.RENDER_GPUS = previous_render
            col.SOURCES = previous_sources
            col.CollectionPhases = original_phases
        if code:
            raise RuntimeError('node collection failed')
        return
    if mode == 'aggregate-node':
        aggregate_node(config_path, node)
        return
    if mode == 'finish-collection':
        if node not in config.get('collection_only_nodes', []):
            raise ValueError('collection-only completion was not authorized by the frozen plan')
        c.check_reserved_validation_entry(spec)
        construction = c.binding(nr/'collection/construction/formal-construction-report.json')
        report = c.checked(construction)
        if not report.get('complete'):
            raise ValueError('recovery collection has unfinished instructions')
        c.save(nr/'receipts/collection-ready.json', dict(ok=True, collection_only=True, training_ready=False,
            node=node, construction=construction,
            pre_reservation=c.binding(nr/'receipts/pre-reservation-idle.json')))
        return
    if mode == 'validate-node':
        c.check_reserved_validation_entry(spec)
        c.save(nr/'receipts/node-ready.json', dict(ok=True, node=node,
            construction=c.binding(nr/'collection/construction/formal-construction-report.json'),
            pre_reservation=c.binding(nr/'receipts/pre-reservation-idle.json')))
        if node == 0:
            for index in range(4):
                ready = wait_for(root/f'node{index}/receipts/node-ready.json', root)
                c.checked(ready['construction'])
                c.checked(ready['pre_reservation'])
            col.aggregate(root/'global/collection-plan.json')
            original_check = c.check_reserved_validation_entry
            # The unchanged admission implementation operates on all 96 parts.
            # Each node has already proved its own release before reservation.
            c.check_reserved_validation_entry = lambda _: original_check(spec)
            result = c.validate_collection(gs)
            c.save(root/'global/receipts/validate.json', dict(ok=True, **result))
        validation = wait_for(root/'global/receipts/validate.json', root)
        c.save(nr/'receipts/validate.json', dict(validation, global_receipt=c.binding(root/'global/receipts/validate.json')))
        return
    if mode == 'export-node':
        local = dict(gs, partitions=[p for p in gs['partitions'] if p['node'] == node])
        result = c.export_lane(local, lane)
        c.save(nr/f'receipts/export-{lane:02d}.json', dict(ok=True, **result))
        return
    if mode == 'assemble-node':
        if node == 0:
            for lane in range(16):
                results = [wait_for(root/f'node{n}/receipts/export-{lane:02d}.json', root) for n in range(4)]
                merged = dict(results[0], outputs=[item for r in results for item in r['outputs']])
                c.save(root/f'global/receipts/export-{lane:02d}.json', merged)
            result = c.assemble(gs)
            c.save(root/'global/receipts/assemble.json', dict(ok=True, data_only=True, **result))
        result = wait_for(root/'global/receipts/assemble.json', root)
        c.save(nr/'receipts/assemble.json', dict(ok=True, data_only=True,
            global_receipt=c.binding(root/'global/receipts/assemble.json')))
        return
    raise ValueError('unsupported data phase')


def entry(config_path):
    config = load(config_path)
    node = rank_from_environment(os.environ, os.uname().nodename)
    service_env = local_service_environment(os.environ)
    os.environ.clear()
    os.environ.update(service_env)
    root = Path(config['output_root'])
    nr = root/f'node{node}'
    c, _, runner, _ = modules(config)
    with (nr/'entry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((nr/p).exists() for p in ('entry-launch.json', 'status.json', 'completion.json')):
            raise ValueError('attempt already launched; automatic replay is forbidden')
        c.save(nr/'entry-launch.json', dict(node=node, hostname=os.uname().nodename,
            identity=runner.process_identity(os.getpid()), config=c.binding(config_path), started_utc=runner.utc()))
        try:
            c.checked(load(root/'preparation.json')['nodes'][node]['plan'])
            runner.KINDS = stage_kinds(config, node)
            code = runner.execute(nr/'pipeline-plan.json')
            if code:
                raise RuntimeError('data pipeline failed; inspect node completion')
        except BaseException as exc:
            c.save(nr/'entry-failure.json', dict(ok=False, node=node, error=repr(exc), at_utc=runner.utc()))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=('prepare', 'verify', 'entry', 'run-collection',
        'aggregate-node', 'finish-collection', 'validate-node', 'export-node', 'assemble-node'))
    parser.add_argument('--node', type=int, choices=range(4))
    parser.add_argument('--lane', type=int, choices=range(16))
    args = parser.parse_args()
    if args.mode == 'prepare':
        proof = prepare(args.config.resolve())
        print(json.dumps({k: proof[k] for k in ('ok', 'original_counts', 'partitions', 'policy_sha256')}))
    elif args.mode == 'verify':
        print(json.dumps(verify(args.config.resolve())))
    elif args.mode == 'entry':
        entry(args.config.resolve())
    elif args.node is None or (args.mode == 'export-node' and args.lane is None):
        parser.error('node is mandatory for phases, and lane for export')
    else:
        phase(args.config.resolve(), args.node, args.mode, args.lane)


if __name__ == '__main__':
    main()
