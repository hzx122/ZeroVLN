"""Bind existing rule collectors/supervisor to a clean Git v8 workspace.

No new collection engine: reuse original population, worker_plan and resident
owner. Smoke and full runs differ only in selected instruction counts/shards.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from prepare_rule_teacher_formal_local16_collection import (
    load_population, sage_native_inputs, worker_plan, set_option, NATIVE_DATASETS,
    balance_scenes)
from prepare_rule_teacher_three_source import ART, REPO, sha256, rank
from rule_teacher_training_handoff import save, load

TEMPLATES = ART/'rule-teacher-r124-current-main-formal43276-retry-export-fixed-preparation-20261006-r1'
PRIOR = ART/'rule-teacher-r166-mfm-sequential-three-source-20261009-r1/instruction-scene-audit-20261010-r1'
POPULATION = ART/'rule-teacher-formal-same-floor-population-20261005-r2/population/formal-population-report.json'
MODEL = Path('/opt/external/user-e7a01947/Model/Qwen3.6-35B-A3B')
SAGE_BRAIN_GPUS = ('0,1', '4,6', '8,9', '12,13')
SAGE_RENDER_GPUS = (2, 3, 7, 7, 10, 11, 14, 15)
NATIVE_BRAIN_GPUS = SAGE_BRAIN_GPUS + ('2,3', '7,10', '11,14')


def collection_layout(source, collectors):
    """Restore eight SAGE renderers; dedicate one TP2 brain per native lane."""
    allowed = (1, 4, 8) if source == 'sage' else (1, 4, 7)
    if collectors not in allowed:
        raise ValueError('use eight SAGE or seven native formal collectors (one/four for diagnostics)')
    count = (collectors+1)//2 if source == 'sage' else collectors
    return dict(brain_gpus=(SAGE_BRAIN_GPUS if source == 'sage' else NATIVE_BRAIN_GPUS)[:count],
                collector_lanes=[i//2 if source == 'sage' else i for i in range(collectors)])


def source_partition(rows, source, rank, shards, collectors, *, smoke=False):
    """Keep each complete scene on one node, for every source and scale."""
    # Balance nodes first; concatenating fixed sets of globally balanced
    # collector buckets can concentrate all large scenes on the same node.
    return balance_scenes(rows, 20261003, workers=shards)[rank]


def smoke_rows(rows, limit, collectors):
    """Select before generation, representing every lane without splitting scenes."""
    if limit < collectors:
        raise ValueError('smoke must include at least one case per collector')
    parts = balance_scenes(rows, 20261003, workers=collectors)
    selected = []
    for index, part in enumerate(parts):
        quota = limit//collectors + int(index < limit % collectors)
        if len(part) < quota:
            raise ValueError('insufficient smoke cases for a collector')
        selected.extend(part[:quota])
    return selected


def prepare(args):
    sys.path.insert(0, str(REPO/'scripts/eval'))
    from unified_closedloop_eval import git_runtime
    provenance = git_runtime(require_container=True)
    if args.output.exists() or not args.output.is_relative_to(ART):
        raise ValueError('new data-disk attempt required')
    runtime = Path('/tmp/v8c')/args.output.name
    os.environ['RULE_COLLECTION_RUNTIME_ROOT'] = str(runtime)
    report, populations = load_population(POPULATION)
    template = load(TEMPLATES/('sage-native-template.json' if args.source=='sage' else 'native-template.json'))
    floor_inputs, sage_native = ({}, None)
    if args.source == 'sage':
        floor_inputs, sage_native = sage_native_inputs(template, populations['sage'],
            report['sources']['sage']['manifest_sha256'])
    population = populations[args.source]
    resume = getattr(args, 'resume_manifest', None)
    if resume:
        from prepare_completed_rule_case_resume import v8_pending_population
        population = v8_pending_population(population, args.source, load(resume))
    if resume and 'scope_node' in load(resume):
        if load(resume)['scope_node'] != args.rank or args.shards != 4:
            raise ValueError('resume population belongs to a different node')
        rows = population
    else:
        rows = source_partition(population, args.source, args.rank,
            args.shards, args.collectors, smoke=bool(args.limit))
    if args.limit:
        # Before any generation/outcome, choose admitted source starts only for
        # smoke; full population still contains the original typed failures.
        if sage_native:
            rows = [r for r in rows if (r['scene'],str(r['task_id'])) in sage_native['bootstrap_identities']]
        rows = smoke_rows(rows, args.limit, args.collectors)
    if not rows:
        raise ValueError('empty source partition')
    source_path = args.output/'source-plan.json'
    workers, plans = prepare_workers(args, rows, template, source_path, sage_native)
    plan = copy.deepcopy(plans[0])
    plan['services'] = [e for child in plans for e in child['services']]
    plan['collectors'] = [e for child in plans for e in child['collectors']]
    plan['service_start_batch_size'] = 16
    plan['allocations'] = [w['allocation'] for w in workers]
    plan.pop('allocation', None)
    services_template = load(PRIOR/'repair-full-recollection-r2/node0/supervisor/launch.json')
    lanes = sorted({w['allocation']['lane'] for w in workers})
    common = [copy.deepcopy(next(e for e in services_template['services'] if e['name']=='nlp'))]
    brain_template = next(e for e in services_template['services'] if e['name']=='brain-0')
    for lane in lanes:
        brain = copy.deepcopy(brain_template)
        brain['name'] = f'brain-{lane}'
        common.append(brain)
    old = str(Path(common[0]['command'][2]).parents[2])
    for e in common:
        e['command'] = [v.replace(old,str(REPO)) for v in e['command']]
        e['log'] = str(args.output/(e['name']+'.log'))
    plan['services'] = common + plan['services']
    plan['runtime'] = str(runtime/'services')
    plan['environment']['ACTION_HEAD_SUITE_PLAN'] = str(args.output/'launch.json')
    navigation = REPO.parent/'navigation/navigation'
    if not (navigation/'src').is_dir():
        raise ValueError('pinned navigation dependency checkout missing')
    for name,target in [('vln_demo.yaml',REPO/'vln_demo/src/vln_demo.yaml'),('data',NATIVE_DATASETS.parent.resolve())]:
        path = REPO/name
        if not path.exists(): path.symlink_to(target, target_is_directory=target.is_dir())
    query_caches = {lane: runtime/f'query/brain{lane}' for lane in lanes}
    for cache in query_caches.values(): cache.mkdir(parents=True, exist_ok=True)
    collector_lanes = {e['name']: w['allocation']['lane']
        for w, child in zip(workers,plans) for e in child['services']+child['collectors']}
    extension_proof = None
    source_map = PRIOR/'source-object-map-r2/manifest.json'
    for e in plan['services']+plan['collectors']:
        env = e['environment']
        lane = collector_lanes.get(e['name'], int(e['name'].split('-')[-1]) if e['name'].startswith('brain-') else lanes[0])
        query_cache = query_caches[lane]
        env.update(PYTHONPATH=f'{REPO}:{REPO}/verl:{REPO}/vln_demo/src',
            PYTHONDONTWRITEBYTECODE='1', RULE_COLLECTION_RUNTIME_ROOT=str(runtime),
            RYNNBRAIN_NAV_DIR=str(navigation), RULE_NLP_URL='http://127.0.0.1:19100',
            RULE_TEACHER_QUERY_CACHE=str(query_cache), RULE_TEACHER_MODEL_CONFIG=str(MODEL/'config.json'),
            RULE_TEACHER_CONTEXT_VERSION='v8', RULE_TEACHER_ROUTE_PREFIX_STEPS='16',
            RULE_TEACHER_PARALLEL_INITIALIZE='1',
            SAGE_SOURCE_OBJECT_MAP=str(source_map), SAGE_SOURCE_OBJECT_MAP_SHA256=sha256(source_map),
            SAGE_REQUIRE_SOURCE_OBJECT_MAP='1')
        env['RULE_TEACHER_V8_PREFIX_CACHE'] = '1' if getattr(args, 'prefix_cache', False) else '0'
        e.pop('native_gym_workspace',None)
        if '-gym-' in e['name']:
            # The collector owns the 100-tool-call horizon. Waypoint density
            # must not introduce a second, hidden episode truncation budget.
            set_option(e['command'], '--max_steps', '0')
        # Every child starts in this checkout; data and runtime caches remain
        # separately bound by the existing supervisor.
        if e['name'].startswith('sage-render'):
            from prepare_renderer_extensions import install
            extension_root = runtime/'renderer-extensions'
            if extension_proof is None:
                extension_proof = install(
                    ART.parent/'runtime_inputs/abot-renderer-ppu-cpython312-v1', extension_root)
            pos=next(i for i,v in enumerate(e['command']) if v.endswith('/render_sim_dynamic.py'))
            e['command'][pos]=str(REPO/'third_party/abot_renderer/render_sim_dynamic.py')
            env['SAGE_RENDERER_EXTENSION_ROOT'] = str(extension_root)
            env['PYTHONPATH'] += (f':{REPO}/vln_demo/render_libs:{REPO}/third_party/abot_renderer'
                f':{REPO}/third_party/abot_renderer/submodules/diff-plane-rasterization'
                f':{extension_root}')
            # Use the renderer's actual interpreter before starting any GPU
            # service; imports do not allocate a renderer or load scenes.
            probe = subprocess.check_output([e['command'][0], '-B', '-c',
                'import json, torch, diff_plane_rasterization, simple_knn._C; '
                'from gaussian_renderer.pgsr import render; '
                'print(json.dumps(dict(wrapper=diff_plane_rasterization.__file__, '
                'extension=diff_plane_rasterization._C.__file__, knn=simple_knn._C.__file__)))'],
                cwd=REPO, env={**os.environ, **env}, text=True)
            save(args.output/'renderer-dependency-preflight.json',
                dict(binary=extension_proof, imports=json.loads(probe.strip().splitlines()[-1])))
        e['command'] = ['env', *sum((['-u',k] for k in (
            'RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT',
            'GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE')), []), *e['command']]
    for brain in common[1:]:
        lane = int(brain['name'].split('-')[-1])
        configure_brain(brain, runtime, query_caches[lane], lane, args.gpu_memory_utilization,
                        gpu_pair=collection_layout(args.source, args.collectors)['brain_gpus'][lane],
                        prefix_cache=getattr(args, 'prefix_cache', False))
        if getattr(args, 'prefix_cache', False):
            brain['environment']['ACTION_HEAD_R3_AUDIT_DIR'] = str(args.output/'native_r3_audit'/brain['name'])
    for collector in plan['collectors']:
        configure_collector(collector, args)
    # Bind selected input data, actual Git files and immutable navigation commit.
    code = subprocess.check_output(['git','ls-files'],cwd=REPO,text=True).splitlines()
    bindings = {str(REPO/name):sha256(REPO/name) for name in code
                if name.endswith(('.py','.sh','.yaml','.json')) and (REPO/name).is_file()}
    input_bindings = {path: sha256(path) for w in workers for path in (w['selection'], w['input'])}
    if resume:
        input_bindings[str(resume)] = sha256(resume)
    source = dict(schema='rule_teacher_v8_git_collection_v1', git_runtime=provenance,
        source_sha256={},code_sha256=bindings,
        input_sha256={**floor_inputs,str(source_map):sha256(source_map), **input_bindings,
            str(REPO/'vln_demo/src/vln_demo.yaml'):sha256(REPO/'vln_demo/src/vln_demo.yaml')},
        source=args.source, count=len(rows), rank=args.rank, shards=args.shards,
        scene_partitioning='whole_scene_balanced',
        layout=collection_layout(args.source, args.collectors),
        tool_budget=args.max_calls, fast_calls=4, points_per_fast_call=16, workers=workers,
        depth_batch_size=args.depth_batch_size, http_concurrency=args.http_concurrency,
        gpu_memory_utilization=args.gpu_memory_utilization,
        prefix_cache=bool(getattr(args, 'prefix_cache', False)))
    if resume:
        source['resume_manifest'] = str(resume)
        source['pending_source_population'] = len(population)
    save(source_path,source)
    plan['independent_collectors'] = True
    save(args.output/'launch.json',plan)
    return source


def prepare_workers(args, rows, template, source_path, sage_native):
    workers, plans = [], []
    layout = collection_layout(args.source, args.collectors)
    brains = [f'http://127.0.0.1:{29200+100*lane}' for lane in range(len(layout['brain_gpus']))]
    # Reuse the same native scene grouping and cache capacity as the actual
    # r166 formal collectors. Neither is a model sampling/concurrency setting.
    parts = balance_scenes(rows, 20261003, workers=args.collectors)
    for collector_index in range(args.collectors):
        index = collector_index
        part = parts[collector_index]
        if not part:
            raise ValueError('empty collector partition')
        worker = worker_plan(args.output, args.source, index, part, template, source_path,
            brains, actors=args.actors, sage_native=sage_native,
            native_scene_cache_size=2)
        child = load(worker['launch'])
        # Keep the original four-TP2 layout, including the validated GPU5
        # isolation. Native renderers are CPU services.
        lane = layout['collector_lanes'][collector_index]
        worker['allocation'].update(lane=lane, depth_batch_size=args.depth_batch_size,
                                    http_concurrency=args.http_concurrency)
        child['allocation'].update(worker['allocation'])
        set_option(child['collectors'][0]['command'], '--brain-url', brains[lane])
        if args.source == 'sage':
            gpu = SAGE_RENDER_GPUS[collector_index]
            set_option(child['services'][0]['command'], '--gpus', gpu)
            worker['allocation']['visual_gpu'] = gpu
            child['allocation']['visual_gpu'] = gpu
        for entry in child['services']+child['collectors']:
            entry['name'] += f'-worker{index}'
        workers.append(worker)
        plans.append(child)
    return workers, plans


def configure_brain(brain, runtime, query_cache, lane, memory, *, gpu_pair=None, prefix_cache=False):
    (runtime/f'brain{lane}').mkdir(parents=True, exist_ok=True)
    port=29200+100*lane
    set_option(brain['command'],'--port',port)
    brain.update(port=port,health_url=f'http://127.0.0.1:{port}/v1/models')
    env=brain['environment']
    env.update(VLLM_PORT=str(port),PYTHONPATH=f'{REPO}/verl/action_head/serve/bootstrap:{REPO}/verl:{REPO}/vln_demo/vllm',
        ACTION_HEAD_QUERY_CACHE=str(query_cache), ACTION_HEAD_CONTINUOUS_CONTEXT='1',
        ACTION_HEAD_R3_CAPTURE_MODE='native',
        ACTION_HEAD_CAPTURE_ROUTING='1',ACTION_HEAD_CAPTURE_FULL_ROUTING='1',
        ACTION_HEAD_BULK_ROUTE_CAPTURE='1',
        VLLM_ENABLE_PREFIX_CACHING='1' if prefix_cache else '0',
        ACTION_HEAD_V8_PREFIX_CACHE='1' if prefix_cache else '0',
        VLLM_MAX_MODEL_LEN='131072', VLLM_MAX_IMAGES='202',
        # Reuse the formal collector's short abstract IPC namespace: runtime
        # names must not consume Linux's 107-byte Unix socket address budget.
        VLLM_RPC_BASE_PATH='@formal-'+rank(str(runtime), lane)[:16],
        VLLM_GPU_MEMORY_UTILIZATION=str(memory),
        CUDA_VISIBLE_DEVICES=gpu_pair or SAGE_BRAIN_GPUS[lane],
        VLLM_GPUS=gpu_pair or SAGE_BRAIN_GPUS[lane])
    env.pop('VLLM_LOGGING_CONFIG_PATH',None)
    command=brain['command']
    command[:] = [arg for arg in command if arg not in (
        '--enable-prefix-caching', '--no-enable-prefix-caching', '--enable-return-routed-experts')]
    command.extend(['--enable-prefix-caching' if prefix_cache else '--no-enable-prefix-caching',
                    '--enable-return-routed-experts'])
    if prefix_cache:
        set_option(command, '--mamba-cache-mode', 'align')
    # User explicitly selected .9 memory. Retain automatic engine concurrency
    # and the validated graph mode from the formal service.
    for key,value in {'--gpu-memory-utilization':str(memory), '--max-model-len':'131072',
                      '--limit-mm-per-prompt':'{"image":202}'}.items(): set_option(command,key,value)


def configure_collector(collector, args):
    for key,value in {'--action-head-as-tool-version':'v8','--route-prefix-steps':'16',
                      '--max-calls':str(args.max_calls),'--candidates':'2',
                      '--depth-batch-size':str(args.depth_batch_size),
                      '--http-concurrency':str(args.http_concurrency)}.items():
        set_option(collector['command'],key,value)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',choices=['sage','r2r','rxr'],required=True)
    p.add_argument('--rank',type=int,default=0)
    p.add_argument('--shards',type=int,default=1)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--actors',type=int,default=3, help='physical session servers per collector')
    p.add_argument('--collectors',type=int,default=1)
    p.add_argument('--depth-batch-size',type=int,default=256)
    p.add_argument('--http-concurrency',type=int,default=128)
    p.add_argument('--resume-manifest',type=Path)
    p.add_argument('--gpu-memory-utilization',type=float,default=.9)
    p.add_argument('--prefix-cache', action=argparse.BooleanOptionalAction, default=False,
                   help='reuse certified episode prefixes with the existing native R3 align implementation')
    p.add_argument('--max-calls',type=int,default=100,
                   help='tool-call budget; use a small value only for end-to-end smoke')
    a=p.parse_args()
    if not 0<=a.rank<a.shards: p.error('invalid partition')
    if not 1<=a.depth_batch_size<=256 or not 1<=a.http_concurrency<=512:
        p.error('invalid client concurrency')
    if not 0<a.gpu_memory_utilization<1: p.error('invalid GPU memory utilization')
    result=prepare(a)
    print(json.dumps({k:result[k] for k in ('git_runtime','source','count','rank','shards')}))
