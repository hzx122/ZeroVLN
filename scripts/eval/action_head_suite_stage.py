#!/usr/bin/env python3
"""Launch/score one suite stage using existing owned-process supervision."""
from __future__ import annotations

import argparse
import importlib.metadata
from concurrent.futures import ThreadPoolExecutor
import json
import os
import signal
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import time
import traceback

from action_head_benchmark_suite import (REPO, ROOT, ART, EVAL, SWEEP, GENERAL,
    atomic_json, sha256_file, digest, partition, stage_root, artifact_root)
from closedloop_evaluation_contract import V8_VERSION, stage_repeats

GV = Path(os.environ.get('RYNNBRAIN_VENV', sys.prefix))
NAV = Path(os.environ.get('ACTION_HEAD_NAVIGATION_REPO', str(REPO / 'third_party/RynnBrain/navigation')))
MESA = Path(os.environ.get('MESA_ROOT', '/usr'))
BOOT = REPO / 'verl/scripts/train/action_head_bootstrap'
SUPERVISOR = REPO / 'scripts/train/supervise_rule_teacher_collection.py'


def verify_frozen(plan):
    for section in ('source_sha256', 'code_sha256', 'input_sha256'):
        for path, expected in plan[section].items():
            if sha256_file(path) != expected:
                raise ValueError('frozen suite input/code changed: ' + path)
    # The payloads are bound by file metadata, not claimed as content hashes.
    # Head/query, model config/index/tokenizer and dataset/code use SHA256.
    for path, expected in plan.get('model_weight_metadata', {}).items():
        stat = Path(path).stat()
        if dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) != expected:
            raise ValueError('frozen HF model weight file changed: ' + path)


def checkpoint(plan, stage):
    return next(row for row in plan['config']['checkpoints'] if row['name'] == stage['checkpoint'])


def subset_manifest(original, cases):
    result = {**original, 'case_count': len(cases), 'cases': cases, 'case_pool_sha256': digest(cases)}
    if 'quotas' in result:
        result['quotas'] = {stage: sum(row['stage'] == stage for row in cases)
                            for stage in sorted({row['stage'] for row in cases})}
    return result


def reusable_compile_cache(plan, ckpt, runtime_base):
    """Version-bound, node-local compiler inputs; never store query state here."""
    versions = {}
    for name in ('torch', 'vllm', 'triton'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'unavailable'
    key = dict(commit=plan['source_provenance']['git_runtime']['commit'],
        model_config=sha256_file(Path(ckpt['brain']) / 'config.json'),
        protocol=plan['closedloop_contract'], python=sys.version, packages=versions,
        sdk=os.environ.get('PPU_SDK', '/usr/local/PPU_SDK'))
    return runtime_base / 'compile-cache' / digest(key)


def launch_plan(plan, stage, rank, plan_path):
    directory = stage_root(plan, stage) / f'node_{rank}'
    directory.mkdir(parents=True, exist_ok=True)
    attempt = directory / 'launch'
    if (attempt / 'plan.json').exists():
        raise ValueError('existing stage launch is immutable; inspect its receipts or prepare a new suite output')
    attempt.mkdir(exist_ok=True)
    ckpt, config, benchmark = checkpoint(plan, stage), plan['config'], stage['benchmark']
    dual = ckpt.get('policy') == 'dualvln'
    primitive = ckpt.get('policy') == 'primitive'
    manifest = json.loads(Path(stage['manifest']).read_text())
    selected = partition(manifest['cases'], rank, config['nodes'])
    node_manifest = directory / 'cases.json'
    atomic_json(node_manifest, subset_manifest(manifest, selected))
    runtime_base = Path(os.environ.get('ACTION_HEAD_SUITE_RUNTIME_BASE', str(ROOT / '.runtime/ahs')))
    runtime = runtime_base / (digest([plan['plan_sha256'], stage['index'], rank])[:12])
    from closedloop_evaluation_contract import VERSION
    mature = config.get('closedloop_protocol') in (VERSION, V8_VERSION)
    layout = None
    service_cfg = None
    if mature and benchmark not in GENERAL and not dual and not primitive:
        from closedloop_evaluation_contract import protocol
        contract = protocol(config['closedloop_protocol'],
            compaction=config.get('context_compaction'), max_tool_calls=config.get('max_tool_calls'))
        layout = contract['layouts'][benchmark]
        service_cfg = contract['services']
    gyms = layout['gym_actors'] if layout else config['gym_actors']
    renderer_port, gym_base, brain_base = 7085, 19100, 29600
    brain_replicas = 8 if benchmark in GENERAL or primitive else (0 if dual else min((layout or {}).get('brain_replicas', service_cfg.get('brain_replicas', 6)) if service_cfg else 6, gyms))
    brain_urls = [f'http://127.0.0.1:{brain_base+i*100}' for i in range(brain_replicas)]
    gym_urls = [f'http://127.0.0.1:{gym_base+i}' for i in range(gyms)]
    # Match the established query/FM layout; SAGE reserves GPU15 for rendering.
    fm_gpus = [12, 13, 14] if benchmark == 'sage-3d' else [12, 13, 14, 15]
    if layout:
        fm_gpus = layout['fm_gpus']
    if not dual and not primitive and benchmark not in GENERAL:
        from closedloop_evaluation_contract import fm_devices
        fm_gpus = fm_devices(config, fm_gpus)
    environment = dict(PYTHONPATH=':'.join(map(str, [EVAL, REPO / 'scripts/train', ROOT / 'VLMEvalKit', REPO, REPO / 'verl',
        REPO / 'vln_demo/src', NAV / 'src', SWEEP])), PYTHONUNBUFFERED='1',
        PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        PATH=os.environ.get('PATH', ''),
        LD_LIBRARY_PATH=f'{GV}/lib:{REPO}/vln_demo/lib:' + os.environ.get('LD_LIBRARY_PATH', ''),
        HTTP_PROXY='', HTTPS_PROXY='', ALL_PROXY='', http_proxy='', https_proxy='', all_proxy='',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', COMPANION_BRAIN_MODE='sft',
        ACTION_HEAD_QUERY_TEXT_BUDGET='256', ACTION_HEAD_INVALID_TOOL_CALL_POLICY='feedback',
        ACTION_HEAD_V43_PROMPT_REVISION='legacy', VLN_OBSERVATION_CONTRACT='agent_v1',
        CUDA_VISIBLE_DEVICES='', SAGE_BENCH_PROTOCOL_ROOT='', VLNCE_NAVMESH_ROOT='')
    if config.get('closedloop_protocol'):
        from closedloop_evaluation_contract import protocol, verify_normalized
        verify_normalized(config)
        environment.update(protocol(config['closedloop_protocol'],
            compaction=config.get('context_compaction'),
            max_tool_calls=config.get('max_tool_calls'))['service_environment'])
        if environment.get('ACTION_HEAD_EVAL_CONTEXT_COMPACTION'):
            environment['ACTION_HEAD_EVAL_BRAIN_CONFIG'] = str(Path(ckpt['brain'])/'config.json')
            environment['ACTION_HEAD_EVAL_CONTEXT_ARCHIVE'] = str(directory/'canonical-contexts')
    services = []
    if primitive:
        environment.update(VLN_PRIMITIVE_MAX_DISTANCE_M=str(ckpt['max_distance_m']),
            VLN_PRIMITIVE_MAX_CALLS_PER_TURN='1')
    frozen = {**plan, 'input_sha256': {**plan['input_sha256'], str(node_manifest): sha256_file(node_manifest)}}

    def service(name, command, port, health, extra=None, group=None):
        row = dict(name=name, command=list(map(str, command)), port=port, health_url=health,
                   startup_timeout=1800, log=str(attempt / (name + '.log')),
                   startup_group=group or name, environment=extra or {})
        services.append(row)
        return row

    if benchmark not in GENERAL:
        profile = ckpt['observation_profiles'][benchmark]
        environment['VLN_OBSERVATION_PROFILE'] = json.dumps(profile)
        first = selected[0]
        if benchmark == 'sage-3d':
            protocol_root = Path(config['sage_protocol_root']).resolve(strict=True)
            environment.update(SAGE_BENCH_PROTOCOL_ROOT=str(protocol_root), SAGE_SPLIT='official_sage_bench_v1',
                SAGE_RENDER_URL=f'http://127.0.0.1:{renderer_port}', SAGE_COLLISION='usd_rooms',
                SAGE_FURNITURE_COLLISION='none', SAGE_ROBOT_RADIUS='0.07',
                SAGE_CAMERA_HEIGHT=str(profile['height_m']), SAGE_HFOV=str(profile['hfov_deg']),
                SAGE_EPS_PER_SCENE='0', SAGE_EPISODES_PER_SCENE='0',
                SAGE_USD_COLLISION_MASK_ROOT=os.environ.get('SAGE_USD_COLLISION_MASK_ROOT', str(ART.parent / 'sage_usd_rooms_masks')))
            environment.update(ckpt['environment'])
            environment['VLN_GYM_DEFER_INITIAL_RESET'] = '1'
            cpu_renderer = bool(layout and layout.get('renderer_device') == 'cpu')
            render_source = (REPO / 'third_party/abot_renderer' if cpu_renderer else
                Path(os.environ.get('ACTION_HEAD_RENDER_REPO', str(ROOT / 'agentic_sim/repo/ABot-Navigation'))) / 'render_server')
            environment['PYTHONPATH'] += (f':{REPO}/vln_demo/render_libs:'
                f'{render_source}/submodules/diff-plane-rasterization:'
                f'{render_source}/submodules/simple-knn:{render_source}')
            if cpu_renderer:
                environment.update(SAGE_RENDER_DEVICE='cpu', SAGE_LAZY_EPISODES='1',
                    SAGE_CPU_RENDER_PARALLELISM=str(layout['render_parallelism']))
                service('renderer_0', [GV / 'bin/python', '-B', render_source / 'render_sim_dynamic.py',
                    '--device', 'cpu', '--cpu-render-parallelism', layout['render_parallelism'],
                    '--scenes_root', Path(os.environ.get('SAGE_SCENES_ROOT', str(ART.parent / 'sage_gs_root'))),
                    '--port', renderer_port, '--gpus', '0', '--sh_degree', '3',
                    '--max_scenes_per_gpu', layout['renderer_cache_scenes']],
                    renderer_port, f'http://127.0.0.1:{renderer_port}/ping',
                    {'CUDA_VISIBLE_DEVICES': '', 'SAGE_CPU_RENDER_CACHE': str(runtime / 'renderer_0/cpu-compositor')}, 'renderers')
            else:
                render_gpus = layout['render_gpus'] if layout else [15]
                render_urls = []
                for index, gpu in enumerate(render_gpus):
                    port = renderer_port + 1 + index if layout else renderer_port
                    url = f'http://127.0.0.1:{port}'
                    render_urls.append(url)
                    service(f'renderer_{index}' if layout else 'renderer',
                        [GV / 'bin/python', BOOT / 'single_gpu_renderer.py',
                        render_source / 'render_sim_dynamic.py', '--scenes_root',
                        Path(os.environ.get('SAGE_SCENES_ROOT', str(ART.parent / 'sage_gs_root'))), '--port', port, '--gpus', str(gpu),
                        '--sh_degree', '3', '--max_scenes_per_gpu',
                        str(layout['renderer_cache_scenes']) if layout else ('8' if primitive else '32')],
                        port, url + '/ping', {'CUDA_VISIBLE_DEVICES': str(gpu)}, 'renderers')
                if layout:
                    # Reuse bootstrap's SHA256 scene router and scheduler mapping.
                    service('renderer_router', [GV / 'bin/python', BOOT / 'service_router.py',
                        '--port', renderer_port, '--mode', 'scene', '--upstreams', ','.join(render_urls)],
                        renderer_port, f'http://127.0.0.1:{renderer_port}/ping')
                    environment['SAGE_LAZY_EPISODES'] = '1'
            for index in range(gyms):
                case = selected[index % len(selected)]
                service(f'gym_{index}', [GV / 'bin/python', '-B', '-u', '-m', 'vln_gym.server',
                    '--backend', 'sage', '--port', gym_base + index, '--max_steps', '0',
                    '--scene', case['scene'], '--episode_id', case['episode_id']], gym_base + index,
                    gym_urls[index] + '/health', group='gyms')
        else:
            import yaml
            base_config = REPO / 'vln_demo/src/vln_demo.yaml'
            native = yaml.safe_load(base_config.read_text())
            if dual:
                native['habitat']['simulator'].update(turn_angle=15, forward_step_size=.25)
            sensors = native['habitat']['simulator']['agents']['main_agent']['sim_sensors']
            for sensor in sensors.values():
                # Habitat's structured SimulatorSensorConfig requires an int.
                sensor.update(width=profile['width'], height=profile['height'], hfov=int(profile['hfov_deg']),
                              position=[0., profile['height_m'], 0.],
                              orientation=[profile['pitch_deg'] * 3.141592653589793 / 180., 0., 0.])
            yaml_path = directory / 'vln_demo.yaml'
            yaml_path.write_text(yaml.safe_dump(native, sort_keys=False))
            frozen['input_sha256'][str(yaml_path)] = sha256_file(yaml_path)
            environment.update(VLN_HABITAT_RENDER_URL=f'http://127.0.0.1:{renderer_port}',
                VLN_RENDER_MODE='batch', RXR_LANGUAGES='en-US,en-IN', RXR_ROLES='guide', RXR_EPISODES_PER_SCENE='0',
                RYNNBRAIN_NAV_DIR=str(NAV), RYNNVLN_LANGUAGE='en', RYNNVLN_HFOV=str(profile['hfov_deg']),
                RYNNVLN_TURN_ANGLE='15' if dual else '30', RYNNVLN_ALLOW_SLIDING='1',
                __EGL_VENDOR_LIBRARY_FILENAMES=str(MESA / '50_mesa.json'), LIBGL_DRIVERS_PATH=str(MESA / 'dri'))
            environment['LD_LIBRARY_PATH'] = str(MESA / 'lib') + ':' + environment['LD_LIBRARY_PATH']
            if not os.environ.get('MESA_ROOT'):
                environment.pop('__EGL_VENDOR_LIBRARY_FILENAMES', None)
                environment.pop('LIBGL_DRIVERS_PATH', None)
            environment['RYNNBRAIN_VENV'] = str(GV)
            environment['VLNCE_NAVMESH_ROOT'] = config.get('navmesh_root', '')
            service('renderer', ['bash', REPO / 'vln_demo/src/remote_habitat/start.sh',
                '--port', renderer_port, '--host', '127.0.0.1',
                *(['--scene-cache-size', str(layout['renderer_cache_scenes']),
                    '--render-parallelism', str(layout['render_parallelism']),
                    '--shared-scene-cache', '--scene-cache-max-gib', str(layout['native_cache_max_gib']),
                    '--scene-load-reserve-gib', str(layout['scene_load_reserve_gib'])] if layout else [])], renderer_port,
                f'http://127.0.0.1:{renderer_port}/health')
            for index in range(gyms):
                case = selected[index % len(selected)]
                row = service(f'gym_{index}', [GV / 'bin/python', '-B', '-u', '-m', 'vln_gym.vlnce_action_head',
                    '--backend', benchmark, '--port', gym_base + index, '--split', 'val_unseen', '--max_steps', '0',
                    '--output_path', directory / 'gym' / str(index), '--scene', case['scene'],
                    '--episode_id', case['episode_id']], gym_base + index, gym_urls[index] + '/health', group='gyms')
                row['native_gym_workspace'] = dict(inside_checkout=True,
                    cwd=str(REPO / '.runtime/native-gym' / runtime.name / row['name'] / 'work'),
                    config_source=str(yaml_path), config_sha256=sha256_file(yaml_path),
                    data_target=str((NAV / 'data').resolve()))
    for index in range(brain_replicas):
        extra = dict(CUDA_VISIBLE_DEVICES=f'{2*index},{2*index+1}')
        if config.get('closedloop_protocol') == V8_VERSION:
            # The service bootstrap installs scheduler hooks and requires its
            # query cache. Gym and actor processes must not run these hooks.
            extra['PYTHONPATH'] = f'{REPO}/verl/action_head/serve/bootstrap:' + environment['PYTHONPATH']
        repeat_index = config.get('evaluation_repeat_index', 1)
        from closedloop_evaluation_contract import LEGACY_CONTINUATION
        if repeat_index > 1 or (config.get('closedloop_protocol') and config['closedloop_protocol'] != LEGACY_CONTINUATION):
            extra['VLLM_SERVER_SEED'] = str(20261013 + repeat_index * 1000 + rank * 10 + index)
        if benchmark in GENERAL or primitive:
            # Keep each replica's internal TCPStore allocation out of the
            # ephemeral client range and away from the other replicas.
            # The HTTP listener is set separately by --port below.
            extra['VLLM_PORT'] = str(31000 + index * 100)
            command = ['vllm', 'serve', ckpt['brain'], '--served-model-name', 'default', '--port', brain_base + index * 100,
                       '--tensor-parallel-size', '2', '--dtype', 'bfloat16', '--max-model-len', str(ckpt.get('max_model_len', 128000 if dual else 131072)), '--gpu-memory-utilization', '.60' if primitive and benchmark == 'sage-3d' and index == 7 else '.92',
                       '--max-num-seqs', '64', '--max-num-batched-tokens', '32768', '--allowed-local-media-path', '/']
        else:
            command = ['bash', REPO / 'vln_demo/vllm/vllm-action-head-query.sh']
            extra.update(VLLM_MODEL=ckpt['brain'], VLLM_GPUS=f'{2*index},{2*index+1}',
                VLLM_DTYPE='bfloat16',
                VLLM_PORT=str(brain_base+index*100),
                VLLM_GPU_MEMORY_UTILIZATION=str(service_cfg.get('gpu_memory_utilization', 0.92)) if service_cfg else '.92',
                VLLM_MAX_NUM_SEQS=str(service_cfg.get('max_num_seqs', 64)) if service_cfg else '64',
                VLLM_MAX_NUM_BATCHED_TOKENS=str(service_cfg.get('max_num_batched_tokens', 32768)) if service_cfg else '32768',
                ACTION_HEAD_QUERY_CHECKPOINT=str(Path(ckpt['path']) / 'dualvln_query/dualvln_query_embeddings.pt'),
                ACTION_HEAD_QUERY_CACHE=str(runtime / 'queries'),
                ACTION_HEAD_COMPILE_CACHE_ROOT=str(runtime / f'brain_{index}/compile'),
                VLLM_RPC_BASE_PATH=f'@ahs-{plan["plan_sha256"][:6]}-{rank}-{index}')
        row = service(f'brain_{index}', command, brain_base+index*100, brain_urls[index] + '/health', extra, 'brains')
        if mature:
            row['reusable_compile_cache'] = str(reusable_compile_cache(plan, ckpt, runtime_base) / f'brain_{index}')
    collector_python = sys.executable
    if dual and benchmark not in GENERAL:
        from dualvln_benchmark_suite import DUAL_PYTHON
        collector_python = str(DUAL_PYTHON)
    collector = dict(name='evaluation', command=[collector_python, '-B', str(Path(__file__).resolve()),
        '--plan', str(plan_path), '--stage', str(stage['index']), '--rank', str(rank), '--collector'],
        log=str(attempt / 'evaluation.log'), environment=dict(SUITE_QUERY_CACHE=str(runtime / 'queries'),
        SUITE_GYM_URLS=','.join(gym_urls), SUITE_BRAIN_URLS=','.join(brain_urls),
        SUITE_FM_GPUS=','.join(map(str, fm_gpus)),
        SUITE_RENDERER_REPLICAS='1' if layout and layout.get('renderer_device') == 'cpu' else '3',
        SUITE_MAX_ACTIVE_SCENES=str(gyms), SUITE_MAX_ACTIVE_CASES=str(gyms),
        SUITE_FM_MAX_BATCH_SIZE=str(layout['max_batch_size'] if layout else 8),
        SUITE_FM_BATCH_WAIT_MS=str(layout['batch_wait_ms'] if layout else 12)))
    source_plan = attempt / 'inputs.json'
    atomic_json(source_plan, frozen)
    if layout and layout.get('renderer_device') == 'cpu':
        # Start real GPU model loading before CPU Gym/renderer initialization.
        services.sort(key=lambda row: 0 if row['name'].startswith('brain_') else 1)
    launch = dict(runtime=str(runtime), source_plan=str(source_plan), environment=environment,
                  services=services, collectors=[collector], service_start_batch_size=8,
                  service_log_spool=bool(mature),
                  unset_environment=['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE',
                    'MASTER_ADDR', 'MASTER_PORT', 'GROUP_RANK', 'NODE_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE'])
    atomic_json(attempt / 'plan.json', launch)
    return attempt / 'plan.json'


def collect(plan, stage, rank):
    directory = stage_root(plan, stage) / f'node_{rank}'
    manifest = json.loads(Path(stage['manifest']).read_text())
    cases = partition(manifest['cases'], rank, plan['config']['nodes'])
    ckpt = checkpoint(plan, stage)
    if stage['benchmark'] in GENERAL:
        from action_head_suite_general import setup, infer
        setup(Path(manifest['metadata']['data_root']), plan['config'])
        (directory / 'evaluation').mkdir(exist_ok=True)
        result = infer(plan, stage, cases, directory / 'evaluation', os.environ['SUITE_BRAIN_URLS'].split(','))
        atomic_json(directory / 'evaluation_result.json', result)
        return 0 if result['evaluation_valid'] else 4
    # Actual rollout resets validate the observed camera profile. Do not reset
    # every idle Gym here: this is a serial startup probe, not evaluation work.
    gym_urls = os.environ['SUITE_GYM_URLS'].split(',')
    if ckpt.get('policy') == 'primitive':
        from primitive_benchmark_suite import run_pool
        result = run_pool(plan, stage, cases, directory / 'evaluation', gym_urls,
            os.environ['SUITE_BRAIN_URLS'].split(','))
        atomic_json(directory / 'evaluation_result.json', result)
        return 0 if result['evaluation_valid'] else 4
    if ckpt.get('policy') == 'dualvln':
        from dualvln_benchmark_suite import run_pool
        result = run_pool(plan, stage, cases, directory / 'evaluation', gym_urls, 'http://127.0.0.1:7085')
        atomic_json(directory / 'evaluation_result.json', result)
        return 0 if result['evaluation_valid'] else 4
    brain_urls = os.environ['SUITE_BRAIN_URLS']
    if stage['benchmark'] in ('r2r', 'rxr'):
        from v6_vlnce_eval import DEFAULT_SET
        command = [sys.executable, '-B', str(EVAL / 'v6_vlnce_eval.py'), 'run-pool',
            '--dataset', stage['benchmark'], '--case-manifest', str(directory / 'cases.json'),
            '--eval-set', plan['config'].get('vlnce_eval_set', str(DEFAULT_SET)),
            '--checkpoint-dir', ckpt['path'], '--backbone', ckpt['backbone'],
            '--output', str(directory / 'evaluation'), '--max-turns', '0', '--max-tool-calls',
            str(plan['config']['max_tool_calls'][stage['benchmark']]), '--scoring-mode', 'merged_batch',
            '--bench-urls', ','.join(gym_urls), '--vllm-urls', brain_urls, '--fm-gpus', os.environ['SUITE_FM_GPUS'],
            '--query-cache', os.environ['SUITE_QUERY_CACHE'],
            '--max-batch-size', os.environ['SUITE_FM_MAX_BATCH_SIZE'],
            '--batch-wait-ms', os.environ['SUITE_FM_BATCH_WAIT_MS']]
        command += ['--repeats',str(stage_repeats(plan['config']))]
        code = subprocess.call(command)
        summary = directory / 'evaluation' / stage['benchmark'] / 'summary.json'
        result = json.loads(summary.read_text()) if summary.exists() else {'evaluation_valid': False}
    else:
        gpus = os.environ['SUITE_FM_GPUS'].split(',')
        workers = []
        for index, gpu in enumerate(gpus):
            worker_cases = cases[index::len(gpus)]
            if not worker_cases:
                continue
            worker = directory / 'evaluation' / f'worker_{index}'
            worker.mkdir(parents=True, exist_ok=True)
            path = worker / 'cases.json'
            atomic_json(path, subset_manifest(manifest, worker_cases))
            urls = gym_urls[index::len(gpus)]
            command = [sys.executable, '-B', str(EVAL / 'action_head_suite_closedloop.py'), '--plan',
                str(Path(plan['output']) / 'plan.json'), '--stage', str(stage['index']), '--rank', str(rank),
                '--manifest', str(path), '--output', str(worker), '--gym-urls', ','.join(urls),
                '--brain-urls', brain_urls, '--query-cache', os.environ['SUITE_QUERY_CACHE']]
            log = (worker / 'worker.log').open('a')
            child = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                SUITE_FM_WORKERS=str(sum(1 for i in range(len(gpus)) if cases[i::len(gpus)]))),
                stdout=log, stderr=subprocess.STDOUT)
            workers.append((child, log))
        code = 0
        for child, log in workers:
            code = child.wait() or code
            log.close()
        rows = [json.loads(p.read_text()) for p in directory.glob('evaluation/worker_*/rollouts/**/*.json')]
        valid = code == 0 and len(rows) == len(cases)*stage_repeats(plan['config']) and all(r.get('evaluation_valid') for r in rows)
        result = {'evaluation_valid': valid, 'completed': len(rows),
                  'invalid_cases': [r['logical_id'] for r in rows if not r.get('evaluation_valid')]}
    atomic_json(directory / 'evaluation_result.json', result)
    return code or (0 if result.get('evaluation_valid') else 4)


def aggregate_stage(plan, stage):
    directory = stage_root(plan, stage)
    receipts = [json.loads((directory / f'node_{r}/result.json').read_text()) for r in range(plan['config']['nodes'])]
    manifest = json.loads(Path(stage['manifest']).read_text())
    output = directory / 'merged'
    output.mkdir(exist_ok=True)
    result = dict(plan_sha256=plan['plan_sha256'], benchmark=stage['benchmark'], checkpoint=stage['checkpoint'],
                  expected=manifest['case_count'], scope=manifest['scope'], evaluation_valid=False, nodes=receipts)
    repeats = stage_repeats(plan['config'])
    try:
        for rank, receipt in enumerate(receipts):
            expected = [row['logical_id'] for row in partition(manifest['cases'], rank, plan['config']['nodes'])]
            if (receipt['plan_sha256'] != plan['plan_sha256'] or receipt['stage'] != stage['index']
                or receipt['rank'] != rank or receipt.get('logical_ids') != expected or not receipt['evaluation_valid']):
                raise ValueError('invalid, missing, overlapping or changed data-shard receipt')
        if stage['benchmark'] in GENERAL:
            from action_head_suite_general import setup, aggregate
            setup(Path(manifest['metadata']['data_root']), plan['config'])
            result.update(aggregate(plan, stage, receipts, output))
        elif checkpoint(plan, stage).get('policy') == 'primitive':
            from primitive_benchmark_suite import contract
            from dualvln_benchmark_suite import aggregate_rows
            rows = [json.loads(p.read_text()) for rank in range(plan['config']['nodes'])
                    for p in (directory / f'node_{rank}/evaluation/rollouts').glob('*.json')]
            result.update(aggregate_rows(plan, stage, rows, manifest['cases'], contract(plan, stage)))
        elif checkpoint(plan, stage).get('policy') == 'dualvln':
            from dualvln_benchmark_suite import aggregate_rows
            rows = [json.loads(p.read_text()) for rank in range(plan['config']['nodes'])
                    for p in (directory / f'node_{rank}/evaluation/rollouts').glob('*.json')]
            result.update(aggregate_rows(plan, stage, rows, manifest['cases']))
        elif stage['benchmark'] in ('r2r', 'rxr'):
            from v6_vlnce_eval import (summarize, read_result, result_path, digest as native_digest,
                                      resumable, build_contract, DEFAULT_SET, repeat_output,summarize_repeats)
            rounds, hashes = [{} for _ in range(repeats)], set()
            for rank in range(plan['config']['nodes']):
                contract_path = directory / f'node_{rank}/evaluation' / stage['benchmark'] / 'contract.json'
                local_cases = partition(manifest['cases'], rank, plan['config']['nodes'])
                if not local_cases:
                    continue
                contract = json.loads(contract_path.read_text())
                ckpt = checkpoint(plan, stage)
                args = SimpleNamespace(checkpoint_dir=ckpt['path'], backbone=ckpt['backbone'],
                    dataset=stage['benchmark'], eval_set=plan['config'].get('vlnce_eval_set', str(DEFAULT_SET)),
                    case_manifest=str(directory / f'node_{rank}/cases.json'), max_turns=0,
                    max_tool_calls=plan['config']['max_tool_calls'][stage['benchmark']], scoring_mode='merged_batch',repeats=repeats)
                overrides = dict(SAGE_BENCH_PROTOCOL_ROOT='',
                    VLNCE_NAVMESH_ROOT=plan['config'].get('navmesh_root', ''),
                    VLN_OBSERVATION_PROFILE=json.dumps(ckpt['observation_profiles'][stage['benchmark']]),
                    RULE_TEACHER_CONTEXT_VERSION=('v8' if plan['config'].get('closedloop_protocol') == V8_VERSION else ''))
                previous = {key: os.environ.get(key) for key in overrides}
                try:
                    os.environ.update(overrides)
                    expected_contract = build_contract(args, local_cases)
                finally:
                    for key, value in previous.items():
                        if value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = value
                if contract != expected_contract:
                    raise ValueError('VLN node contract differs from frozen camera/navmesh/data/checkpoint/protocol/budget')
                for case in partition(manifest['cases'], rank, plan['config']['nodes']):
                    for repeat in range(1,repeats+1):
                        root = repeat_output(directory/f'node_{rank}/evaluation'/stage['benchmark'],repeat,repeats)
                        row = read_result(result_path(root,case))
                        if (not resumable(row,case,native_digest(contract),repeat)
                                or row.get('repeats_per_case',1) != repeats):
                            raise ValueError('VLN case/repeat identity changed')
                        if case['logical_id'] in rounds[repeat-1]:
                            raise ValueError('duplicate VLN case/repeat across nodes')
                        rounds[repeat-1][case['logical_id']] = row
                        hashes.add(row['contract_sha256'])
            # Node contracts bind different local subsets; each was validated by
            # its existing evaluator. Rebind only after explicit source checks.
            contract_hash = digest([plan['plan_sha256'], stage['index']])
            for rows in rounds:
                for row in rows.values():
                    row['contract_sha256'] = contract_hash
            result.update(summarize_repeats(manifest['cases'],rounds,contract_hash) if repeats>1
                          else summarize(manifest['cases'],rounds[0],contract_hash))
            if repeats == 1 and plan['config'].get('pooled_attempts'):
                from rule_teacher_repeat_companion import paired_metrics
                result['paired_metrics'] = paired_metrics(rounds,list(rounds[0]),1)
        else:
            sys.path[:0] = [str(SWEEP), str(REPO / 'verl')]
            import closedloop_eval as CE
            ckpt = checkpoint(plan, stage)
            expected = {row['logical_id']: row for row in manifest['cases']}
            found, validated = set(), []
            rounds = [{} for _ in range(repeats)]
            for source in directory.glob('node_*/evaluation/worker_*/rollouts/**/*.json'):
                row = json.loads(source.read_text())
                case = expected.get(row.get('logical_id'))
                if not case or row.get('suite_plan_sha256') != plan['plan_sha256'] or row.get('case_sha256') != digest(case):
                    raise ValueError('SAGE rollout differs from frozen suite identity')
                repeat = row.get('repeat_index',1)
                key = (case['logical_id'],repeat)
                if key in found or not 1<=repeat<=repeats or row.get('repeats_per_case',1)!=repeats:
                    raise ValueError('duplicate or incompatible SAGE repeated rollout')
                found.add(key)
                validated.append(row)
                rounds[repeat-1][case['logical_id']] = {k:row[k] for k in ('evaluation_valid','success','spl')}
            if found != {(key,i) for key in expected for i in range(1,repeats+1)}:
                raise ValueError('missing SAGE rollouts')
            # Official scorer identity is evaluated in the same environment.
            os.environ['SAGE_BENCH_PROTOCOL_ROOT'] = plan['config']['sage_protocol_root']
            os.environ.update(ckpt['environment'])
            os.environ['ACTION_HEAD_INVALID_TOOL_CALL_POLICY'] = 'feedback'
            result.update(CE.aggregate(output, Path(stage['manifest']), ckpt['identity'],
                                      action_head_as_tool_version=('v8' if plan['config'].get('closedloop_protocol') == V8_VERSION else 'v6'),
                                      repeats=repeats, report_pass_at_k=repeats>1,
                                      validated_rows=validated))
            from rule_teacher_repeat_companion import paired_metrics
            result['paired_metrics'] = paired_metrics(rounds,list(expected),repeats)
    except Exception as exc:
        result.update(evaluation_valid=False, error=repr(exc), traceback=traceback.format_exc())
    result.update(plan_sha256=plan['plan_sha256'], benchmark=stage['benchmark'], checkpoint=stage['checkpoint'])
    atomic_json(directory / 'aggregate.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--stage', type=int, required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--collector', action='store_true')
    args = parser.parse_args()
    from action_head_suite_runtime import install_control_io
    install_control_io(args.plan.parent, sys.modules[__name__])
    plan = json.loads(args.plan.read_text())
    stage = plan['stages'][args.stage]
    if args.collector:
        return collect(plan, stage, args.rank)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'stage received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    verify_frozen(plan)
    directory = stage_root(plan, stage) / f'node_{args.rank}'
    directory.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(stage['manifest']).read_text())
    cases = partition(manifest['cases'], args.rank, plan['config']['nodes'])
    receipt = dict(plan_sha256=plan['plan_sha256'], stage=stage['index'], rank=args.rank,
        logical_ids=[row['logical_id'] for row in cases], selected=len(cases), evaluation_valid=False)
    try:
        if not cases:
            receipt.update(evaluation_valid=True, exit_code=0, result={'completed': 0, 'evaluation_valid': True})
        else:
            ckpt = checkpoint(plan, stage)
            if ckpt.get('policy') == 'dualvln' and stage['benchmark'] not in GENERAL:
                from dualvln_benchmark_suite import DUAL_PYTHON
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='', DS_ACCELERATOR='cpu',
                           PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
                from supervise_rule_teacher_collection import CACHE_KEYS, collection_runtime_root
                cache = collection_runtime_root() / ('dual-env-' + plan['plan_sha256'][:12])
                for key in CACHE_KEYS:
                    path = cache / key.lower()
                    path.mkdir(parents=True, exist_ok=True)
                    env[key] = str(path)
                subprocess.run([str(DUAL_PYTHON), '-B', str(EVAL / 'dualvln_benchmark_suite.py'),
                    '--check-environment', '--model-path', ckpt['path'], '--depth-path', ckpt['depth_path'],
                    '--output', str(directory / 'dualvln_environment.json')], env=env, check=True)
            launch = launch_plan(plan, stage, args.rank, args.plan.resolve())
            supervisor = subprocess.Popen([sys.executable, '-B', str(SUPERVISOR), '--plan', str(launch)])
            stat = Path(f'/proc/{supervisor.pid}/stat').read_text().rsplit(')', 1)[1].split()
            atomic_json(launch.parent / 'supervisor-launch.json', dict(pid=supervisor.pid,
                start_ticks=stat[19], argv=[sys.executable, '-B', str(SUPERVISOR), '--plan', str(launch)]))
            try:
                code = supervisor.wait()
            finally:
                if supervisor.poll() is None:
                    supervisor.terminate()
                    supervisor.wait()
            result_path = directory / 'evaluation_result.json'
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            completion_path = launch.parent / 'completion.json'
            completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
            receipt.update(exit_code=code, result=result, completion=str(completion_path),
                evaluation_valid=code == 0 and result.get('evaluation_valid') is True
                and completion.get('status') == 'complete' and completion.get('owned_process_groups_cleaned') is True)
    except Exception as exc:
        receipt.update(error=repr(exc), traceback=traceback.format_exc(), exit_code=1)
    atomic_json(directory / 'result.json', receipt)
    return 0 if receipt['evaluation_valid'] else 4


if __name__ == '__main__':
    raise SystemExit(main())
