#!/usr/bin/env python3
"""Benchmark-major checkpoint evaluation with frozen data and node barriers.

Uses the existing SAGE/VLN-CE/VLMEvalKit evaluators and collection supervisor.
No distributed model training: each node loads the same checkpoint and owns a
disjoint slice of the same benchmark. The selected cases never depend on ckpt.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO.parent
ART = Path(os.environ.get('RL_ARTIFACT_ROOT', '/data/closed_loop_3dgs_rl'))
EVAL = REPO / 'scripts/eval'
SWEEP = REPO / 'verl/scripts/eval/qwen36_4query_ckpt_sweep'
sys.path[:0] = [str(SWEEP), str(REPO / 'vln_demo/src'), str(REPO / 'scripts/train')]
from common import atomic_json, sha256_file, checkpoint_identity, load_subset
from watch_v6_rl_local_credit import process_identity, same_process_alive
from vln_gym.observation_profile import normalize_profile

GENERAL = ('CV-Bench-3D', 'CV-Bench-2D', 'DA-2K', 'ERQA', 'EmbSpatialBench',
           'RoboSpatialHome', 'MMSIBench_wo_circular', 'RefSpatial-Bench-Location',
           'RefSpatial-Bench-Placement', 'RefSpatial-Bench-Unseen', '3DSRBench',
           'ViewSpatialBench')
ORDER = ('sage-3d', 'r2r', 'rxr', *GENERAL)
SCHEMA = 'action_head_benchmark_major_suite_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def artifact_root(path):
    root = Path(path).resolve()
    if not root.is_relative_to(ART) or root == ART:
        raise ValueError(f'output must be an experiment under {ART}')
    return root


def select_cases(cases, limit, seed, benchmark):
    """Stable hash permutation, nested for any sample count, independent of ckpt."""
    if type(limit) is not int or limit < 0 or limit > len(cases):
        raise ValueError(f'{benchmark}: requested {limit} of {len(cases)} available cases')
    ids = [str(row['logical_id']) for row in cases]
    if len(set(ids)) != len(ids):
        raise ValueError(f'{benchmark}: duplicate source identities')
    ordered = sorted(cases, key=lambda row: (digest([seed, benchmark, row['logical_id']]),
                                            str(row['logical_id'])))
    return ordered[:limit] if limit else ordered


def partition(cases, rank, nodes):
    if type(nodes) is not int or nodes < 1 or type(rank) is not int or not 0 <= rank < nodes:
        raise ValueError('suite requires positive node count and rank in [0, nodes)')
    return cases[rank::nodes]


def model_weight_metadata(directory):
    root = Path(directory)
    index = root / 'model.safetensors.index.json'
    if index.exists():
        names = set(json.loads(index.read_text())['weight_map'].values())
    else:
        names = {p.name for pattern in ('*.safetensors', 'pytorch_model*.bin') for p in root.glob(pattern)}
    if not names or any(Path(name).name != name for name in names):
        raise ValueError('incomplete HF model weights: ' + str(root))
    result = {}
    for name in sorted(names):
        p = root / name
        stat = p.stat()
        if stat.st_size == 0:
            raise ValueError('empty model shard: ' + str(p))
        result[str(p)] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return result


def normalize_config(config):
    config = json.loads(json.dumps(config))
    nodes = config.get('nodes', 1)
    if type(nodes) is not int or nodes < 1:
        raise ValueError('nodes must be a positive integer, with 16 GPUs per node')
    names = config.get('benchmarks', list(ORDER))
    if not names or names != [name for name in ORDER if name in names]:
        raise ValueError('benchmarks must be unique and in SAGE → R2R → RxR → general12 order')
    limits = config.setdefault('sample_counts', {})
    if set(limits) - set(ORDER) or any(type(n) is not int or n < 0 for n in limits.values()):
        raise ValueError('sample_counts requires benchmark names and nonnegative integers; 0=full')
    checkpoints = config.get('checkpoints')
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError('at least one checkpoint entry is required')
    seen = set()
    for ckpt in checkpoints:
        name = ckpt['name']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in seen:
            raise ValueError('checkpoint names must be unique safe path components')
        seen.add(name)
        ckpt['path'] = str(Path(ckpt['path']).resolve(strict=True))
        if ckpt.get('policy', 'action_head') == 'dualvln':
            from dualvln_benchmark_suite import checkpoint_metadata
            ckpt['brain'] = str(Path(ckpt['brain']).resolve(strict=True))
            ckpt['depth_path'] = str(Path(ckpt['depth_path']).resolve(strict=True))
            identity = checkpoint_metadata(ckpt['path'], ckpt['brain'])
            report = {}
        elif ckpt.get('policy') == 'brain_only':
            if any(b not in GENERAL for b in names):
                raise ValueError('brain_only checkpoints support general benchmarks only')
            ckpt['brain'] = str(Path(ckpt.get('brain', ckpt['path'])).resolve(strict=True))
            if ckpt['brain'] != ckpt['path']:
                raise ValueError('brain_only must evaluate its specified model directory')
            identity = dict(policy='brain_only', checkpoint_dir=ckpt['path'], qwen_dir=ckpt['brain'])
            report = {}
        elif ckpt.get('policy') == 'primitive':
            ckpt['brain'] = str(Path(ckpt.get('brain', ckpt['path'])).resolve(strict=True))
            if ckpt['brain'] != ckpt['path']:
                raise ValueError('primitive baseline must evaluate its specified brain')
            identity = dict(policy='primitive', checkpoint_dir=ckpt['path'], qwen_dir=ckpt['brain'])
            ckpt['max_distance_m'] = float(ckpt.get('max_distance_m', 5.))
            if not .05 <= ckpt['max_distance_m'] <= 10.:
                raise ValueError('primitive movement range must be 0.05–10m')
            report = {}
        elif ckpt.get('policy', 'action_head') == 'action_head':
            identity = checkpoint_identity(ckpt['path'])
            report = json.loads((Path(ckpt['path']) / 'report.json').read_text())
        else:
            raise ValueError('unknown checkpoint policy')
        if identity.get('condition_mode') == 'v5_native':
            raise ValueError('suite requires query-conditioned action head as tool checkpoints; v5_native uses a separate evaluator')
        ckpt['identity'] = identity
        ckpt['brain'] = str(Path(ckpt.get('brain') or identity.get('qwen_dir') or
            config.get('default_brain', '')).resolve(strict=True))
        if not (Path(ckpt['brain']) / 'config.json').is_file():
            raise ValueError('brain must be a complete HF export, not an action head directory')
        if identity.get('qwen_dir') and ckpt['brain'] != identity['qwen_dir']:
            raise ValueError('joint RL evaluation must use its own exported trained brain')
        ckpt['backbone'] = str(Path(ckpt.get('backbone') or config['default_backbone']).resolve(strict=True))
        profiles = ckpt.get('observation_profiles') or report.get('observation_profiles') or config.get('observation_profiles', {})
        ckpt['observation_profiles'] = {b: normalize_profile(profiles[b]) for b in names if b not in GENERAL}
        for b in ('r2r', 'rxr'):
            if b in ckpt['observation_profiles'] and not ckpt['observation_profiles'][b]['hfov_deg'].is_integer():
                raise ValueError('native Habitat requires integer hfov_deg; fractional FOV is supported for SAGE')
        # Metadata source is frozen too; profiles remain explicit when historical
        # checkpoints predate camera metadata. Do not infer eye height from GT Z.
        source = ckpt.get('training_config') or config.get('training_config')
        if not source:
            raise ValueError('training_config or training service launcher required to trace camera settings')
        ckpt['training_config'] = str(Path(source).resolve(strict=True))
        ckpt['training_sources'] = [str(Path(p).resolve(strict=True))
                                    for p in ckpt.get('training_sources', [])]
        ckpt['environment'] = {**config.get('environment', {}), **ckpt.get('environment', {})}
        if 'max_model_len' in ckpt:
            if type(ckpt['max_model_len']) is not int or ckpt['max_model_len'] <= 0:
                raise ValueError('max_model_len must be a positive integer')
        allowed = {'SAGE_GEOMETRY_MODE', 'SAGE_ROBOT_RADIUS', 'SAGE_COLLISION', 'SAGE_FURNITURE_COLLISION',
                   'SAGE_NAVMESH_ROOT', 'SAGE_TONEMAP', 'SAGE_TONEMAP_EXP', 'SAGE_TONEMAP_GAMMA',
                   'SAGE_RGB_OUTSIDE_CLEANUP', 'SAGE_USD_BOUNDARY_CLIP', 'SAGE_USD_STATIC_BOUNDARY'}
        if set(ckpt['environment']) - allowed:
            raise ValueError('unsupported training environment override; camera uses observation_profiles')
        if 'sage-3d' in names and ckpt['environment'].get('SAGE_GEOMETRY_MODE') not in ('physical', 'legacy_xy'):
            raise ValueError('SAGE geometry mode must be explicitly bound to the checkpoint training environment')
    config['nodes'], config['benchmarks'] = nodes, names
    config.setdefault('seed', 20261008)
    config.setdefault('max_tool_calls', {'sage-3d': 200, 'r2r': 100, 'rxr': 100})
    if any(type(n) is not int or n < 1 for n in config['max_tool_calls'].values()):
        raise ValueError('tool budgets must be positive integers')
    config.setdefault('gym_actors', 32)
    if config['gym_actors'] not in (4, 8, 16, 32, 64):
        raise ValueError('gym_actors must be 4, 8, 16, 32 or 64')
    return config


@contextmanager
def preparation_cache():
    # Share the existing supervisor's complete cache-key list. Preparation can
    # import plotting/HF/dataset libraries before any GPU service is launched.
    sys.path.insert(0, str(REPO / 'scripts/train'))
    from supervise_rule_teacher_collection import CACHE_KEYS
    from supervise_rule_teacher_collection import collection_runtime_root
    base = collection_runtime_root()
    base.mkdir(parents=True, exist_ok=True)
    previous = {key: os.environ.get(key) for key in CACHE_KEYS}
    with tempfile.TemporaryDirectory(prefix='ahs-prepare-', dir=base) as folder:
        try:
            for key in CACHE_KEYS:
                path = Path(folder) / key.lower()
                path.mkdir()
                os.environ[key] = str(path)
            tempfile.tempdir = None
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            tempfile.tempdir = None


def prepare(config_path, output):
    with preparation_cache():
        return _prepare(config_path, output)


def _prepare(config_path, output):
    output = artifact_root(output)
    config = normalize_config(json.loads(Path(config_path).read_text()))
    if (output / 'plan.json').exists():
        raise ValueError('plan already exists; use run/status for existing output or a new directory')
    output.mkdir(parents=True, exist_ok=True)
    frozen = {str(Path(config_path).resolve()): sha256_file(config_path)}
    populations, stages = {}, []
    for benchmark in config['benchmarks']:
        if benchmark == 'sage-3d':
            source = Path(config['sage_manifest']).resolve(strict=True)
            metadata, cases = load_subset(source)
            cases = [{**row, 'logical_id': digest([benchmark, row['stage'], row['scene'],
                      str(row['episode_id'])])[:20]} for row in cases]
            frozen[str(source)] = sha256_file(source)
        elif benchmark in ('r2r', 'rxr'):
            from v6_vlnce_eval import load_cases, DEFAULT_SET
            eval_set = Path(config.get('vlnce_eval_set', str(DEFAULT_SET)))
            cases = load_cases(eval_set, benchmark)
            metadata = {'eval_set': str(eval_set.resolve())}
            frozen[str(eval_set.resolve() / 'lock.json')] = sha256_file(eval_set / 'lock.json')
        else:
            # Reuse VLMEvalKit's dataset construction to account for filtering,
            # image aliases and benchmark-specific indexes before sampling.
            from action_head_suite_general import prepare_population
            metadata, cases = prepare_population(benchmark, output / 'inputs/general', config)
            frozen.update(metadata['input_sha256'])
        print(json.dumps({'benchmark': benchmark, 'source_count': len(cases),
                          'phase': 'population_loaded'}), flush=True)
        selected = select_cases(cases, config['sample_counts'].get(benchmark, 0),
                                config['seed'], benchmark)
        if config.get('closedloop_protocol') and len(selected) < config['nodes']:
            raise ValueError('selected cases must cover every node; reduce nodes or increase subset: '+benchmark)
        manifest = {'schema': SCHEMA, 'benchmark': benchmark, 'seed': config['seed'],
                    'source_count': len(cases), 'case_count': len(selected), 'cases': selected,
                    'metadata': metadata, 'case_pool_sha256': digest(selected),
                    'scope': 'full' if len(selected) == len(cases) else 'subset'}
        if benchmark == 'sage-3d':
            manifest['quotas'] = {stage: sum(x['stage'] == stage for x in selected)
                                  for stage in sorted({x['stage'] for x in selected})}
        path = output / 'manifests' / (benchmark + '.json')
        atomic_json(path, manifest)
        frozen[str(path)] = sha256_file(path)
        populations[benchmark] = {k: manifest[k] for k in ('source_count', 'case_count', 'scope', 'case_pool_sha256')}
        for ckpt in config['checkpoints']:
            index = len(stages)
            stage = {'index': index, 'benchmark': benchmark, 'checkpoint': ckpt['name'],
                     'manifest': str(path), 'case_pool_sha256': digest(selected),
                     'node_counts': [len(partition(selected, rank, config['nodes']))
                                     for rank in range(config['nodes'])]}
            stages.append(stage)
    model_weights = {}
    for ckpt in config['checkpoints']:
        files = ('config.json',) if ckpt.get('policy') in ('primitive', 'brain_only') else ('download_verification.json', 'config.json', 'model.safetensors.index.json') if ckpt.get('policy') == 'dualvln' else (
            'report.json', 'action_head/fast_slow_action_head_config.json',
            'action_head/fast_slow_action_head.pt', 'dualvln_query/dualvln_query_embeddings.pt')
        for name in files:
            p = Path(ckpt['path']) / name
            frozen[str(p)] = sha256_file(p)
        frozen[ckpt['training_config']] = sha256_file(ckpt['training_config'])
        for path in ckpt['training_sources']:
            frozen[path] = sha256_file(path)
        if ckpt.get('policy') == 'dualvln':
            for p in Path(ckpt['depth_path']).iterdir():
                if p.is_file():
                    frozen[str(p)] = sha256_file(p)
        for role in ('brain', 'backbone'):
            model_weights.update(model_weight_metadata(ckpt[role]))
            for p in Path(ckpt[role]).iterdir():
                if p.is_file() and p.suffix in ('.json', '.jinja', '.model', '.tiktoken'):
                    frozen[str(p)] = sha256_file(p)
    if 'sage-3d' in config['benchmarks']:
        bundle = Path(config['sage_protocol_root']).resolve(strict=True)
        for p in (bundle / 'protocol.json', bundle / 'episodes.json',
                  bundle / 'official_semantic_maps_provenance.json'):
            frozen[str(p)] = sha256_file(p)
    if config.get('navmesh_root'):
        from vln_gym.vlnce_action_head import navmesh_override_identity
        navmesh = navmesh_override_identity(config['navmesh_root'])
        for p in [Path(navmesh['root']) / 'manifest.json',
                  *[Path(navmesh['root']) / scene / (scene + '.navmesh') for scene in navmesh['assets']]]:
            frozen[str(p)] = sha256_file(p)
    from v6_vlnce_eval import SOURCE_FILES
    code = [REPO / name for name in SOURCE_FILES]
    code.extend(EVAL / name for name in ('action_head_benchmark_suite.py', 'action_head_suite_stage.py',
                                        'action_head_suite_general.py', 'action_head_suite_closedloop.py',
                                        'submit_action_head_suite.py'))
    if any(c.get('policy') == 'dualvln' for c in config['checkpoints']):
        code.extend(EVAL / name for name in ('dualvln_benchmark_suite.py', 'navigation_baseline_policies.py',
            'navigation_baselines.py', 'export_dualvln_brain.py', 'dualvln_eval_requirements.txt'))
    if any(c.get('policy') == 'primitive' for c in config['checkpoints']):
        code.extend(EVAL / name for name in ('primitive_benchmark_suite.py', 'dualvln_benchmark_suite.py',
            'navigation_baselines.py', 'navigation_baseline_policies.py'))
        code += [SWEEP / 'native_tool_closedloop_eval.py', REPO / 'vln_demo/src/prompts.py']
    code += [REPO / 'vln_demo/src/vln_gym/observation_profile.py', REPO / 'vln_demo/src/sage/env.py',
             EVAL / 'run_action_head_suite.sh', REPO / 'scripts/train/supervise_rule_teacher_collection.py',
             REPO / 'scripts/serve/vllm/with_local_runtime.sh', REPO / 'scripts/serve/vllm/vllm-action-head-query.sh']
    code.extend((REPO / 'verl/action_head').rglob('*.py'))
    code += [REPO / 'scripts/train/watch_v6_rl_local_credit.py',
             REPO / 'scripts/train/prepare_rule_teacher_three_source.py',
             REPO / 'scripts/train/dlc_control_io_retry.py',
             EVAL / 'action_head_suite_runtime.py']
    if any(name in GENERAL for name in config['benchmarks']):
        code.extend((ROOT / 'VLMEvalKit/vlmeval').rglob('*.py'))
    plan = {'schema': SCHEMA, 'output': str(output), 'config': config, 'stages': stages,
            'populations': populations, 'input_sha256': frozen,
            'source_sha256': {}, 'model_weight_metadata': model_weights,
            'code_sha256': {str(p): sha256_file(p) for p in code}}
    plan['plan_sha256'] = digest(plan)
    atomic_json(output / 'plan.json', plan)
    return plan


def verify(plan):
    from action_head_suite_stage import verify_frozen
    if digest({k: v for k, v in plan.items() if k != 'plan_sha256'}) != plan['plan_sha256']:
        raise ValueError('suite plan identity changed')
    verify_frozen(plan)
    if plan['config'].get('closedloop_protocol'):
        from closedloop_evaluation_contract import contract_for, verify_normalized
        verify_normalized(plan['config'])
        if plan.get('closedloop_contract') != contract_for(plan['config']):
            raise ValueError('unified closed-loop contract is missing or changed')


def node_identity(nodes, explicit=None):
    if explicit is not None:
        rank = explicit
    elif nodes == 1:
        rank = 0
    else:
        candidates = [os.environ[k] for k in ('RANK', 'GROUP_RANK', 'NODE_RANK') if k in os.environ]
        if not candidates or len(set(candidates)) != 1:
            raise ValueError('4-node suite needs an unambiguous RANK/GROUP_RANK/NODE_RANK')
        rank = int(candidates[0])
    partition([], rank, nodes)
    return rank


def stage_root(plan, stage):
    return Path(plan['output']) / 'stages' / f"{stage['index']:02d}_{stage['benchmark']}" / stage['checkpoint']


def emit_stage_progress(directory, stage, rank, state):
    """Expose real supervisor/process and data progress to the DLC log monitor.

    Child output remains in its original per-service files. A timestamp alone
    is not a health claim; stale/missing receipts and process failures stay
    visible. The existing supervisor still owns failure/cleanup decisions.
    """
    from supervise_rule_teacher_collection import collection_status, read_status_json
    from action_head_suite_runtime import install_control_io
    install_control_io(Path(directory).parents[3], sys.modules['supervise_rule_teacher_collection'])
    read_status_json = sys.modules['supervise_rule_teacher_collection'].read_status_json
    directory = Path(directory)
    report = dict(event='suite_progress', observed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  rank=rank, stage=stage, state=state)
    launch = directory / 'launch' / 'plan.json'
    if launch.exists():
        observation = collection_status(launch)
        report['supervision'] = {key: observation[key] for key in
            ('recorded_status', 'heartbeat_age_seconds', 'heartbeat_stale', 'supervisor',
             'completion_status', 'processes', 'omitted_processes', 'receipt_warnings')}
    else:
        report['supervision'] = dict(recorded_status='awaiting_stage_launch')
    workers = []
    for path in sorted((directory / 'evaluation').glob('worker_*/parallel_progress.json')):
        progress = read_status_json(path)
        workers.append(dict(worker=path.parent.name, age_seconds=max(0, time.time() - path.stat().st_mtime),
            executed=progress.get('executed'), resumed=progress.get('resumed'),
            elapsed_s=progress.get('elapsed_s'), batching=progress.get('batching'),
            scheduler=progress.get('scheduler')))
    report['workers'] = workers
    # Always flush: otherwise buffered parent output can trigger AllLogHang.
    print(json.dumps(report, ensure_ascii=False, separators=(',', ':')), flush=True)
    return report


def run(plan_path, rank=None):
    from action_head_suite_runtime import install_control_io
    install_control_io(Path(plan_path).parent, sys.modules[__name__])
    plan = json.loads(Path(plan_path).read_text())
    if plan.get('runtime_repo') and Path(plan['runtime_repo']).resolve() != REPO:
        raise ValueError('suite must run from its frozen execution repository')
    nodes = plan['config']['nodes']
    rank = node_identity(nodes, rank)
    output = artifact_root(plan['output'])
    node = output / 'nodes' / str(rank)
    node.mkdir(parents=True, exist_ok=True)
    identity = dict(process_identity(os.getpid()), host=socket.gethostname())
    with (node / 'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from action_head_suite_runtime import SuiteRuntime
        with SuiteRuntime(REPO, node, plan_path) as runtime:
            verify(plan)
            # An orphaned stage supervisor can outlive the coordinator. Its owned
            # services must be accounted for before another writer touches the run.
            for stage in plan['stages']:
                previous = stage_root(plan, stage) / f'node_{rank}' / 'launch'
                if (previous / 'plan.json').exists():
                    completion = previous / 'completion.json'
                    if not completion.exists() or not json.loads(completion.read_text()).get('owned_process_groups_cleaned'):
                        raise ValueError(f'previous stage ownership not released; inspect {previous}/status.json and supervisor-launch.json')
                    if not (previous.parent / 'result.json').exists():
                        raise ValueError('previous stage has no result receipt; preserve it and prepare a new output: ' + str(previous))
            for stage in plan['stages']:
                directory = stage_root(plan, stage)
                receipt = directory / f'node_{rank}' / 'result.json'
                release = directory / 'aggregate.json'
                if not receipt.exists():
                    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                        ACTION_HEAD_SUITE_PLAN=str(Path(plan_path).resolve()))
                    cmd = [sys.executable, '-B', str(EVAL / 'action_head_suite_stage.py'),
                           '--plan', str(Path(plan_path).resolve()), '--stage', str(stage['index']),
                           '--rank', str(rank)]
                    receipt.parent.mkdir(parents=True, exist_ok=True)
                    with (receipt.parent / 'stage.log').open('a') as log:
                        runtime.phase('evaluating', 'active', stage)
                        child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                        next_log = 0
                        try:
                            while child.poll() is None:
                                atomic_json(node / 'status.json', dict(plan_sha256=plan['plan_sha256'],
                                    stage=stage, state='running', pid=os.getpid(), child_pid=child.pid,
                                    identity=identity, updated=time.time()))
                                if time.monotonic() >= next_log:
                                    emit_stage_progress(receipt.parent, stage, rank, 'running')
                                    next_log = time.monotonic() + 30
                                time.sleep(2)
                        finally:
                            if child.poll() is None:
                                child.terminate()
                                child.wait()
                    if not receipt.exists():
                        atomic_json(receipt, dict(plan_sha256=plan['plan_sha256'], rank=rank,
                            stage=stage['index'], evaluation_valid=False, exit_code=child.returncode,
                            error='stage failed before producing a result; inspect stage.log'))
                runtime.phase('stage_barrier_and_scoring', 'idle', stage)
                # Every checkpoint finishes on all nodes before the next checkpoint
                # or benchmark can start. Failure is preserved and later stages run.
                next_log = 0
                while not release.exists():
                    receipts = [directory / f'node_{r}' / 'result.json' for r in range(nodes)]
                    if rank == 0 and all(p.exists() for p in receipts):
                        from action_head_suite_stage import aggregate_stage
                        aggregate_stage(plan, stage)
                        break
                    atomic_json(node / 'status.json', dict(plan_sha256=plan['plan_sha256'],
                        stage=stage, state='waiting_for_stage_barrier', pid=os.getpid(), identity=identity, updated=time.time()))
                    if time.monotonic() >= next_log:
                        emit_stage_progress(receipt.parent, stage, rank, 'waiting_for_stage_barrier')
                        next_log = time.monotonic() + 30
                    runtime.reservation.check()
                    time.sleep(2)
                result = json.loads(release.read_text())
                if result['plan_sha256'] != plan['plan_sha256']:
                    raise ValueError('stage barrier belongs to another suite plan')
                print(json.dumps(dict(event='suite_stage_complete', rank=rank, stage=stage,
                                      evaluation_valid=result['evaluation_valid']), ensure_ascii=False), flush=True)
            results = [json.loads((stage_root(plan, s) / 'aggregate.json').read_text()) for s in plan['stages']]
            complete = all(row['evaluation_valid'] for row in results)
            atomic_json(node / 'completion.json', dict(plan_sha256=plan['plan_sha256'],
                        evaluation_valid=complete, stage_count=len(results), finished=time.time()))
            atomic_json(node / 'status.json', dict(plan_sha256=plan['plan_sha256'],
                        state='complete', evaluation_valid=complete, pid=os.getpid(), identity=identity, updated=time.time()))
            if rank == 0:
                atomic_json(output / 'summary.json', dict(schema=SCHEMA, plan_sha256=plan['plan_sha256'],
                    evaluation_valid=complete, benchmark_order=plan['config']['benchmarks'], stages=results))
            return 0 if complete else 4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'status', 'launch-local'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--rank', type=int)
    args = parser.parse_args()
    if args.command != 'status':
        source = args.config if args.command == 'prepare' else args.plan
        if source:
            value = json.loads(source.read_text())
            config = value if args.command == 'prepare' else value['config']
            if any(b in config.get('benchmarks', list(ORDER)) for b in ('sage-3d','r2r','rxr')):
                parser.error('new navigation evaluation must use scripts/eval/unified_closedloop_eval.py; this CLI is historical/internal')
    if args.command == 'prepare':
        if not args.config or not args.output:
            parser.error('prepare requires --config and --output')
        plan = prepare(args.config, args.output)
        print(json.dumps({'plan': str(args.output / 'plan.json'), 'populations': plan['populations'],
                          'stages': plan['stages'], 'plan_sha256': plan['plan_sha256']}, indent=2))
        return 0
    if not args.plan:
        parser.error('--plan required')
    if args.command == 'launch-local':
        plan = json.loads(args.plan.read_text())
        verify(plan)
        if plan['config']['nodes'] != 1:
            parser.error('launch-local owns one physical 16-GPU node; use platform submit for 4 nodes')
        root = artifact_root(plan['output'])
        with (root / 'local-launch.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            previous = root / 'local-launch.json'
            if previous.exists():
                record = json.loads(previous.read_text())
                stat = Path(f'/proc/{record["pid"]}/stat')
                if stat.exists():
                    fields = stat.read_text().rsplit(')', 1)[1].split()
                    if fields[0] != 'Z' and fields[19] == record['start_ticks']:
                        raise ValueError('suite is already running; use status, do not duplicate writers')
            from supervise_rule_teacher_collection import collection_runtime_root
            runtime = collection_runtime_root() / ('ahs-' + plan['plan_sha256'][:8])
            command = ['bash', str(REPO / 'scripts/serve/vllm/with_local_runtime.sh'),
                       'bash', str(EVAL / 'run_action_head_suite.sh')]
            env = dict(os.environ, ACTION_HEAD_LOCAL_RUNTIME_DIR=str(runtime),
                ACTION_HEAD_SUITE_RUNTIME_BASE=str(runtime / 'stages'), ACTION_HEAD_SUITE_PLAN=str(args.plan.resolve()))
            with (root / 'local-supervisor.log').open('a') as log:
                child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            stat = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()
            record = dict(pid=child.pid, start_ticks=stat[19], plan_sha256=plan['plan_sha256'],
                          command=command, plan=str(args.plan.resolve()), log=str(root / 'local-supervisor.log'))
            atomic_json(previous, record)
            print(json.dumps(record))
        return 0
    if args.command == 'run':
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'suite received signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        return run(args.plan, args.rank)
    plan = json.loads(args.plan.read_text())
    root = Path(plan['output'])
    statuses = {}
    for r in range(plan['config']['nodes']):
        status, completion = root / 'nodes' / str(r) / 'status.json', root / 'nodes' / str(r) / 'completion.json'
        row = json.loads(status.read_text()) if status.exists() else {}
        identity = row.get('identity') or {}
        if identity.get('host') == socket.gethostname():
            row['identity_alive'] = same_process_alive(identity)
        if completion.exists():
            row.update(state='complete', completion=json.loads(completion.read_text()))
        statuses[str(r)] = row or None
    launch = root / 'local-launch.json'
    local = json.loads(launch.read_text()) if launch.exists() else None
    if local:
        local['identity_alive'] = same_process_alive(local)
    summary = root / 'summary.json'
    result = json.loads(summary.read_text()) if summary.exists() else None
    if result:
        result = dict(evaluation_valid=result['evaluation_valid'], stages=[
            {k: row[k] for k in ('benchmark', 'checkpoint', 'scope', 'evaluation_valid',
                                'expected', 'completed', 'completed_runs', 'valid_runs', 'error') if k in row}
            for row in result['stages']])
    print(json.dumps(dict(nodes=statuses, local=local, summary=str(summary),
                         result=result), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
