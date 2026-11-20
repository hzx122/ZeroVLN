#!/usr/bin/env python3
"""Attach verified four-node v6 training to a completed data-only pipeline.

Reuses the existing formal composition's artifact and process checks, the
existing v6 trainer/initializer, official platform terminal, and idle controller.
No collection plans or source inputs are changed by this follow-on supervisor.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import threading
from types import SimpleNamespace
from dlc_control_io_retry import retry_estale, retry_atomic_json

USER_ROOT = Path('/workspace')
ART = Path('/data/zerovln/artifacts')
TERMINAL = ('Failed', 'Stopped', 'Deleted', 'Succeeded', 'Cancelled')

# Retain the exact native LayerNorm shape that reproduced node0/GPU5's fault,
# then exercise finite BF16/FP32 gradients before allowing distributed training.
GPU_HEALTH_PROGRAM = '''import torch,json,os
assert torch.cuda.device_count()==1
torch.set_num_threads(1)
for dtype in [torch.float32,torch.bfloat16]:
 x=torch.randn((4096,1152),device='cuda',dtype=dtype,requires_grad=True)
 w=torch.ones(1152,device='cuda',dtype=dtype,requires_grad=True)
 b=torch.zeros_like(w,requires_grad=True)
 for i in range(24):
  with torch.no_grad(): y=torch.nn.functional.layer_norm(x,(1152,),w,b,1e-6)
 torch.cuda.synchronize();assert torch.isfinite(y).all().item()
 y=torch.nn.functional.layer_norm(x,(1152,),w,b,1e-6)
 y.float().square().mean().backward()
 torch.cuda.synchronize()
 for t in (x,w,b): assert t.grad is not None and torch.isfinite(t.grad).all().item()
 a=torch.randn((512,512),device='cuda',dtype=dtype,requires_grad=True)
 c=torch.randn((512,512),device='cuda',dtype=dtype,requires_grad=True)
 z=a@c;assert torch.isfinite(z).all().item()
 z.float().square().mean().backward();torch.cuda.synchronize()
 for t in (a,c): assert t.grad is not None and torch.isfinite(t.grad).all().item()
print(json.dumps(dict(gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),ok=True,
 contract='native_ln4096x1152_fp32_bf16_forward_backward_gemm_v1',
 device=torch.cuda.get_device_name(0))),flush=True)
'''


def load(path):
    return json.loads(retry_estale(Path(path).read_text, str(path)))


def save(path, value):
    path = Path(path)
    def write():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name+f'.tmp.{os.getpid()}')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
        os.replace(temporary, path)
    retry_atomic_json(write, path)


def sha(path):
    def digest():
        result = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b''):
                result.update(chunk)
        return result.hexdigest()
    return retry_estale(digest, str(path))


def bind(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def checked(record):
    if sha(record['path']) != record['sha256']:
        raise ValueError('changed bound input: '+record['path'])
    return load(record['path'])


def data_ready(data_root):
    """A successful assembler alone is insufficient: all owners must finish."""
    root = Path(data_root)
    for node in range(4):
        nr = root/f'node{node}'
        if (nr/'node-owner/formal-failure.json').exists():
            raise RuntimeError(f'data owner {node} failed; automatic training forbidden')
        for name in ('completion.json', 'node-owner/formal-status.json'):
            path = nr/name
            if path.exists():
                state = load(path)
                if state.get('status', state.get('state')) in ('failed', 'blocked'):
                    raise RuntimeError(f'data node {node} failed: {path}')
        if not (nr/'node-owner/formal-result.json').exists():
            return False
        result = load(nr/'node-owner/formal-result.json')
        completion = load(nr/'completion.json')
        status = load(nr/'node-owner/formal-status.json')
        if status.get('state') != 'complete':
            return False  # The owner's finally block has not acknowledged idle yet.
        if not (result.get('completed') is True and result.get('returncode') == 0
                and completion.get('status') == 'complete'
                and completion.get('runtime_cleaned') is True
                and not completion.get('cleanup_errors')
                and status.get('idle_requested') is True
                and not status.get('cleanup_error')):
            raise RuntimeError(f'data node {node} did not finish cleanly')
    receipt = root/'global/receipts/assemble.json'
    if not receipt.exists():
        return False
    result = load(receipt)
    if not (result.get('ok') is True and result.get('formal_population_mode') is True):
        raise RuntimeError('global data assembly did not pass')
    return True


def handoff_data_ready(config):
    if not config.get('published_inputs'):
        return data_ready(config['data_root'])
    publication = load(config['published_inputs'])
    if publication.get('complete') is not True or set(publication['sources']) != {'r2r', 'rxr', 'sage'}:
        raise ValueError('three-source publication incomplete')
    for name, entry in publication['sources'].items():
        if not (Path(config['published_index'])/name/'summary.json').exists():
            return False
        receipt = checked(entry['completion'])
        index = load(Path(config['published_index'])/name/'summary.json')
        if (not all(receipt.get(k) is True for k in ('complete', 'formal_training_ready', 'loader_verified_all_shards'))
                or index.get('complete') is not True or index['source_sha256'] != entry['manifest']['sha256']
                or index['anchors'] != entry['counts']['accepted_anchors']):
            raise ValueError('published source/index mismatch: '+name)
    return True


def prepare_published_training(config):
    root = Path(config['root'])
    publication = load(config['published_inputs'])
    anchors = sum(e['counts']['accepted_anchors'] for e in publication['sources'].values())
    stride = float(config.get('checkpoint_stride_epochs', 0.5))
    epoch, steps = half_epoch_steps(anchors, stride=stride)
    base = checked(config['v6_template'])
    version = config.get('action_head_as_tool_version', 'v6')
    weights = config.get('source_sampling_ratio', {'r2r':1, 'rxr':1, 'sage':2})
    manifest = publication['sources']['r2r']['manifest']['path']
    base['shared'].update(manifest=manifest, output_root=str(root/'checkpoints'),
        fast_backbone=config['fast_backbone'], published_source_index=config['published_index'],
        source_sampling_ratio=weights, action_head_as_tool_version=version)
    base['initialization']['output_root'] = str(root/'initialization')
    base['training'].update(total_epochs=5, save_freq=1, val_freq=0, val_subset=0,
                           learning_rate=float(config.get('learning_rate', base['training']['learning_rate'])))
    base['scales']['D16'].update(max_steps=5*epoch, run_name=config['run_name'])
    cfg = root/'experiment_config.json'
    save(cfg, base)
    source = config['training_source']
    init = root/'initialization/d16'
    env = dict(ACTION_HEAD_CODE_ROOT=source, REPO=source, MANIFEST=manifest,
        ACTION_HEAD_AS_TOOL_VERSION=version,
        ACTION_HEAD_PUBLISHED_INDEX=config['published_index'],
        ACTION_HEAD_SOURCE_RATIO=':'.join(str(weights[n]) for n in ('r2r','rxr','sage')),
        PYTHON=config['python'], SLOW_MODEL=base['shared']['slow_model'],
        LEARNING_RATE=str(base['training']['learning_rate']),
        ACTION_HEAD_FAST_BACKBONE=config['fast_backbone'], ACTION_HEAD_FAST_VISUAL_MODE='live',
        ACTION_HEAD_SCALE_NAME='D16',
        NNODES='4', NPROC_PER_NODE='16', ACTION_HEAD_ANCHORS_PER_GPU='1',
        ACTION_HEAD_TOTAL_EPOCHS='5', MAX_STEPS=str(epoch*5), SAVE_FREQ='1', VAL_FREQ='0', VAL_SUBSET='0',
        ACTION_HEAD_CHECKPOINT_STEPS=','.join(map(str,steps)), ACTION_HEAD_STEP_OFFSET='0',
        ACTION_HEAD_RESUME_DATA_STEPS='0', ACTION_HEAD_WARM_START_CHECKPOINT='',
        ACTION_HEAD_EXPERIMENT_CONFIG=str(cfg), ACTION_HEAD_EXPERIMENT_CONFIG_SHA256=sha(cfg),
        HEAD_CHECKPOINT=str(init/'head'), QUERY_CHECKPOINT=str(init/'query'),
        OUTPUT_DIR=str(root/'checkpoints'/config['run_name']), TENSORBOARD_DIR=str(root/'tensorboard'),
        RUN_NAME=config['run_name'], PYTHONPATH=source+'/verl:'+source+'/vln_demo/src',
        ACTION_HEAD_TRAIN_PROFILE='suffix_v1', ACTION_HEAD_LR_WARMUP_STEPS='0',
        ACTION_HEAD_CLIP_GRAD='1.0', ACTION_HEAD_NOISE_SEED_POLICY='anchor_v1',
        ACTION_HEAD_FROZEN_QUERY_BACKEND='query_only', FSDP_SIZE='1', ACTION_HEAD_FLA_L2NORM_DYNAMIC_NB='1',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        PPU_SDK='/usr/local/PPU_SDK', CUDA_PATH='/usr/local/PPU_SDK/CUDA_SDK',
        PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    if config.get('prefix_reuse'):
        if version != 'v8' or weights != {'sage':1, 'r2r':1, 'rxr':2}:
            raise ValueError('v8 prefix training requires named SAGE:R2R:RxR=1:1:2')
        env.update(ACTION_HEAD_PREFIX_REUSE='1', ACTION_HEAD_EPISODE_SAMPLER='1',
                   ACTION_HEAD_PREFIX_CACHE_GIB=str(config.get('prefix_cache_gib', 4)),
                   ACTION_HEAD_PREFIX_CACHE_TOKENS='16384')
    if config.get('fm_prefix_reuse'):
        if config.get('prefix_reuse') or version != 'v8' or weights != {'sage':1, 'r2r':1, 'rxr':2}:
            raise ValueError('v8 FM prefix training replaces MoE prefix reuse at SAGE:R2R:RxR=1:1:2')
        # v8 keeps full history (up to ~96k tokens / ~200 images per anchor);
        # without expandable segments the frozen-MoE activations fragment and
        # OOM even on the uncached path.
        env.update(ACTION_HEAD_FM_PREFIX_REUSE='1', ACTION_HEAD_EPISODE_SAMPLER='1',
                   ACTION_HEAD_FROZEN_QUERY_BACKEND='legacy_suffix',
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    if config.get('v8_label_profile'):
        if version != 'v8' or config['v8_label_profile'] not in ('fine16', 'coarse4', 'exec4'):
            raise ValueError('v8 label profile must be fine16, coarse4 or exec4')
        env['ACTION_HEAD_V8_LABEL_PROFILE'] = config['v8_label_profile']
    if config.get('local_replica_load'):
        env['ACTION_HEAD_LOCAL_REPLICA_LOAD'] = '1'
    # Explicit, frozen communication overrides do not alter training semantics.
    communication = config.get('communication_environment', {})
    if any(not key.startswith(('NCCL_', 'GLOO_')) for key in communication):
        raise ValueError('communication override contains a non-communication setting')
    env.update({key: str(value) for key, value in communication.items()})
    if config.get('git_runtime'):
        env['ACTION_HEAD_VERL_REPO'] = config['git_runtime']['verl_checkout']
    launch = dict(experiment_config=str(cfg), experiment_config_sha256=sha(cfg), environment=env,
        initialize_argv=[config['python'],'-B',source+'/verl/sft/initialize_action_head_scale.py',
                         '--experiment-config',str(cfg),'--scale','D16'],
        train_argv=['bash',source+'/scripts/train/action_head_dualvln_sft/entry.sh',
                    '--action-head-as-tool-version',version])
    resume = config.get('warm_start')
    if resume:
        report = checked(resume['report'])
        offset = int(report['global_step'])
        if not 0 < offset < epoch*5 or offset % epoch:
            raise ValueError('published warm start requires a completed epoch boundary')
        if report.get('ok') is not True or report.get('global_batch_size') != 64:
            raise ValueError('invalid warm start checkpoint')
        for record in resume['artifacts']:
            if bind(record['path']) != record:
                raise ValueError('warm start checkpoint changed')
        checkpoint = Path(resume['report']['path']).parent
        env.update(HEAD_CHECKPOINT=str(checkpoint/'action_head'),
            QUERY_CHECKPOINT=str(checkpoint/'dualvln_query'),
            MAX_STEPS=str(epoch*5-offset), ACTION_HEAD_STEP_OFFSET=str(offset),
            ACTION_HEAD_RESUME_DATA_STEPS=str(offset),
            ACTION_HEAD_WARM_START_CHECKPOINT=str(checkpoint),
            ACTION_HEAD_CHECKPOINT_STEPS=','.join(str(s-offset) for s in steps if s>offset))
        steps = [s for s in steps if s > offset]
    save(root/'launch-plan.json',launch)
    save(root/'schedule.json',dict(ok=True,anchors=anchors,global_batch=64,steps_per_epoch=epoch,
        training_layout=training_layout(config),epochs=5,checkpoint_steps=steps,
        checkpoints=len(steps),max_steps=5*epoch,exposures_per_epoch=epoch*64,
        exposure_per_source={name:epoch*64*weights[name]//sum(weights.values()) for name in weights},
        named_source_sampling_ratio=weights,
        source_sampling='exact_1_1_2_per_global_batch_source_without_replacement_cycles',
        source_publication=bind(config['published_inputs']),config=bind(cfg),
        half_epoch_rounding='first completed update at or after the boundary'))
    return launch


def half_epoch_steps(anchors, batch=64, epochs=5, stride=0.5):
    if anchors <= 0:
        raise ValueError('no training anchors')
    denom = round(1.0 / float(stride))
    if denom < 1:
        raise ValueError('checkpoint_stride_epochs must be <= 1.0')
    epoch = (anchors+batch-1)//batch
    # Round to the first completed update at/after each stride boundary.
    return epoch, sorted({(k*epoch + denom - 1) // denom for k in range(1, epochs*denom + 1)})


def training_layout(config):
    """Explicit fallback only; never silently change the frozen v6 topology."""
    mode = config.get('training_layout', 'v6_four_node_16')
    if mode == 'v6_four_node_16':
        devices = [list(range(16)) for _ in range(4)]
    elif mode == 'isolate_node0_gpu5':
        devices = [[0, 1, 2, 3, 4, 6, 7, 8]] + [list(range(8)) for _ in range(3)]
    else:
        raise ValueError('unknown explicit training layout: '+str(mode))
    local_size = len(devices[0])
    world = 4*local_size
    return dict(mode=mode, devices=devices, nproc_per_node=local_size,
        world_size=world, anchors_per_gpu=64//world, global_batch=64,
        idle_devices=[[gpu for gpu in range(16) if gpu not in selected] for selected in devices])


def reserve_training_idle_gpus(config, rank, runner, env):
    """Reuse the collection's reservation and PID-guarded cleanup helpers."""
    ids = training_layout(config)['idle_devices'][rank]
    if not ids:
        return None
    runner.assert_gpus_idle(ids)
    output = Path(config['root'])/f'node{rank}'/'unused-gpu-reservation'
    output.mkdir(parents=True, exist_ok=False)
    identity = None
    reservation = runner.WaitingGpuReservation(dict(waiting_gpu_reservation=dict(
        enabled=True, gpu_count=len(ids), python=config['python'], startup_timeout=120,
        script=config['training_source']+'/verl/scripts/train/action_head_sft_data/gpu_occupier.py')),
        output, lambda process: runner.stop_owned(process, identity))
    original_check = reservation.check

    def check():
        nonlocal identity
        if reservation.process is not None and identity is None:
            identity = runner.process_identity(reservation.process.pid)
        original_check()

    reservation.check = check
    with runner.temporary_environment(dict(env, CUDA_VISIBLE_DEVICES=','.join(map(str, ids)))):
        reservation.start()
    save(output/'physical-gpus.json', dict(ids=ids, identity=identity,
        meaning='idle reservation only; excluded from training ranks'))
    return reservation


def training_gpu_preflight(config, rank, env):
    """Every selected physical device must pass before joining the barrier."""
    devices = training_layout(config)['devices'][rank]
    directory = Path(config['root'])/f'node{rank}'/'gpu-health'
    directory.mkdir(parents=True, exist_ok=False)

    def probe(gpu):
        local_env = dict(env, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_LAUNCH_BLOCKING='1',
                         OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        logfile = directory/f'gpu-{gpu}.log'
        try:
            result = subprocess.run([config['python'], '-B', '-c', GPU_HEALTH_PROGRAM],
                env=local_env, capture_output=True, text=True, timeout=120)
            logfile.write_text(result.stdout+'\n'+result.stderr)
            proof = json.loads(result.stdout.strip().splitlines()[-1]) if result.stdout.strip() else {}
            ok = (result.returncode == 0 and proof.get('ok') is True and proof.get('gpu') == gpu
                  and proof.get('contract') == 'native_ln4096x1152_fp32_bf16_forward_backward_gemm_v1')
            return dict(gpu=gpu, ok=ok, returncode=result.returncode, proof=proof, log=str(logfile))
        except subprocess.TimeoutExpired as exc:
            # subprocess.run has killed and waited for the timed-out child.
            chunks = [part.decode(errors='replace') if isinstance(part, bytes) else (part or '')
                      for part in (exc.stdout, exc.stderr)]
            logfile.write_text('\n'.join(chunks)+'\nGPU preflight timed out\n')
            return dict(gpu=gpu, ok=False, error='timeout', log=str(logfile))
        except (OSError, ValueError, IndexError) as exc:
            return dict(gpu=gpu, ok=False, error=repr(exc), log=str(logfile))

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        rows = list(pool.map(probe, devices))
    proof = dict(ok=all(row['ok'] for row in rows), node=rank, hostname=os.uname().nodename,
                 checked_at=time.time(), gpu_count=len(devices), devices=rows,
                 layout=training_layout(config),
                 probe_sha256=hashlib.sha256(GPU_HEALTH_PROGRAM.encode()).hexdigest())
    save(directory.parent/'gpu-health.json', proof)
    if not proof['ok']:
        raise RuntimeError('GPU training preflight failed: '+str([row['gpu'] for row in rows if not row['ok']]))
    return proof


def composition(config):
    descriptor = importlib.util.spec_from_file_location('formal_rule_teacher_pipeline', config['composition'])
    module = importlib.util.module_from_spec(descriptor)
    sys.modules[descriptor.name] = module
    descriptor.loader.exec_module(module)
    module.activate(config['collection_source'])
    runner = module.pipeline_adapter().configure(config['collection_source'])
    return module, runner


def verify_frozen(config):
    if config.get('git_runtime'):
        root = Path(config['training_source']).resolve()
        expected = config['git_runtime']
        if root != Path(__file__).resolve().parents[2] or Path.cwd().resolve() != root:
            raise ValueError('handoff must run from the container Git checkout')
        actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        origin = subprocess.check_output(['git', '-C', str(root), 'remote', 'get-url', 'origin'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
        if actual != expected['commit'] or origin != expected['repository'] or dirty:
            raise ValueError('container Git revision/origin/cleanliness differs')
        if Path(config['collection_source']).resolve() != root or not Path(config['composition']).resolve().is_relative_to(root):
            raise ValueError('business dependencies must use the same local checkout')
    for item in config['bindings']:
        if bind(item['path']) != item:
            raise ValueError('follow-on source/config binding changed: '+item['path'])


def audit_fast_windows(manifest):
    """Stream real NPZ labels and compare them with each measured route frame."""
    import numpy as np
    from sft.action_head_data_contract import RULE_FULL_ROUTE_CONTRACT, validate_v6_training_row
    from sft.export_rule_teacher_training import full_route_actions, fixed_unclipped_normalization
    transform, _ = fixed_unclipped_normalization()
    scale, offset = np.asarray(transform.scale), np.asarray(transform.offset)
    counts, lengths = Counter(), {}
    group, previous = [], None
    anchors = 0
    min_step, max_step = float('inf'), 0.
    min_gap, max_gap = float('inf'), 0.

    def verify(rows):
        nonlocal anchors, min_step, max_step, min_gap, max_gap
        if len(rows) != 4 or [r['fast_window_index'] for r in rows] != list(range(4)):
            raise ValueError('slow anchor does not contain exactly four ordered fast windows')
        proof = rows[0]['teacher_execution_provenance']
        execution_path = proof['execution_path']
        if sha(execution_path) != proof['execution_sha256']:
            raise ValueError('execution evidence changed')
        state = load(execution_path)['state']
        if state['route_chunks'] != 4 or not state['full_route_complete']:
            raise ValueError('formal anchor lacks four complete executed chunks')
        lengths[round(float(state['nominal_step_m']), 6)] = lengths.get(round(float(state['nominal_step_m']), 6), 0)+1
        origins = []
        for row, window in zip(rows, state['training_windows'], strict=True):
            with np.load(row['sample'], allow_pickle=False) as sample:
                target, mask = sample['target_actions'], sample['action_mask']
                if target.shape != (16, 3) or mask.shape != (16,) or not mask.all():
                    raise ValueError('incomplete label window')
                physical = target.astype(np.float64)*scale+offset
            expected, _ = full_route_actions(state, window)
            if not np.allclose(physical, expected, atol=1e-6, rtol=1e-6):
                raise ValueError('labels do not preserve actual planner displacement')
            pose = row['source_pose_xyyaw']
            expected_pose = [*window['world_pose'][:2], np.deg2rad(window['world_pose'][2])]
            if not np.allclose(pose, expected_pose, atol=1e-6):
                raise ValueError('fast image/label start pose mismatch')
            distances = np.linalg.norm(physical[:4, :2], axis=1)
            min_step, max_step = min(min_step, float(distances.min())), max(max_step, float(distances.max()))
            origins.append(pose[:2])
        gaps = np.linalg.norm(np.diff(np.asarray(origins), axis=0), axis=1)
        min_gap, max_gap = min(min_gap, float(gaps.min())), max(max_gap, float(gaps.max()))
        counts[rows[0]['source_dataset']] += 1
        anchors += 1

    with Path(manifest).open() as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            validate_v6_training_row(row)
            if (row['split'] != 'train' or row['data_contract'] != RULE_FULL_ROUTE_CONTRACT
                    or row.get('visual_input_mode') != 'live_frozen_visual_v1'):
                raise ValueError('unexpected formal training row')
            key = row['slow_anchor_id']
            if previous is not None and key != previous:
                verify(group)
                group = []
            previous = key
            group.append(row)
    if group:
        verify(group)
    if not anchors or set(counts) != {'sage', 'r2r', 'rxr'}:
        raise ValueError('three-source formal data is incomplete')
    return dict(ok=True, anchors=anchors, windows=4*anchors, per_source=dict(counts),
        nominal_step_distinct=len(lengths), nominal_step_min=min(lengths), nominal_step_max=max(lengths),
        executed_step_distance_min=min_step, executed_step_distance_max=max_step,
        fast_origin_distance_min=min_gap, fast_origin_distance_max=max_gap,
        label_verification='all NPZ labels round-trip to actual planner metre/radian deltas',
        grouping='slow_anchor_id + fast_window_index; no spatial resampling',
        terminal_holds_preserved=True, fixed_distance_forcing=False)


def prepare(config, c):
    if config.get('published_inputs'):
        return prepare_published_training(config)
    root, data = Path(config['root']), Path(config['data_root'])
    layout = training_layout(config)
    receipt = data/'global/receipts/assemble.json'
    assembled = load(receipt)
    c.verify_assembled_inputs(assembled)
    original = load(data/'global/training/data/assembly.json')
    manifest = data/'global/training/data/manifest.jsonl'
    audit = audit_fast_windows(manifest)
    if audit['anchors'] != original['anchors']['train']:
        raise ValueError('audited anchor count differs from loader assembly')
    epoch, steps = half_epoch_steps(audit['anchors'])
    base = checked(config['v6_template'])
    base['shared'].update(manifest=str(manifest), output_root=str(root/'checkpoints'),
        fast_backbone=config['fast_backbone'], validation_manifest=str(data/'global/training/data/validation.jsonl'))
    base['initialization']['output_root'] = str(root/'initialization')
    base['training'].update(nnodes=4, nproc_per_node=layout['nproc_per_node'], anchors_per_gpu=layout['anchors_per_gpu'],
        total_epochs=5, save_freq=1, val_freq=0, val_subset=0)
    base['scales']['D16'].update(max_steps=5*epoch, run_name=config['run_name'])
    cfg = root/'experiment_config.json'
    save(cfg, base)
    launch = load(data/'global/training/local_launch_plan.json')
    env = launch['environment']
    init = root/'initialization/d16'
    output = root/'checkpoints'/config['run_name']
    env.update(ACTION_HEAD_CODE_ROOT=config['training_source'], REPO=config['training_source'],
        MANIFEST=str(manifest), SLOW_MODEL=base['shared']['slow_model'],
        LEARNING_RATE=str(base['training']['learning_rate']),
        ACTION_HEAD_FAST_BACKBONE=config['fast_backbone'], ACTION_HEAD_FAST_VISUAL_MODE='live',
        ACTION_HEAD_SCALE_NAME='D16',
        NNODES='4', NPROC_PER_NODE=str(layout['nproc_per_node']), ACTION_HEAD_ANCHORS_PER_GPU=str(layout['anchors_per_gpu']),
        ACTION_HEAD_TOTAL_EPOCHS='5', MAX_STEPS=str(epoch*5), SAVE_FREQ='1', VAL_FREQ='0', VAL_SUBSET='0',
        ACTION_HEAD_CHECKPOINT_STEPS=','.join(map(str, steps)), ACTION_HEAD_STEP_OFFSET='0',
        ACTION_HEAD_RESUME_DATA_STEPS='0', ACTION_HEAD_WARM_START_CHECKPOINT='',
        ACTION_HEAD_EXPERIMENT_CONFIG=str(cfg), ACTION_HEAD_EXPERIMENT_CONFIG_SHA256=sha(cfg),
        HEAD_CHECKPOINT=str(init/'head'), QUERY_CHECKPOINT=str(init/'query'),
        OUTPUT_DIR=str(output), TENSORBOARD_DIR=str(root/'tensorboard'), RUN_NAME=config['run_name'],
        PYTHONPATH=f"{config['training_source']}/verl:{config['collection_source']}/vln_demo/src")
    launch.update(experiment_config=str(cfg), experiment_config_sha256=sha(cfg), environment=env,
        initialize_argv=[config['python'], '-B', config['training_source']+'/verl/sft/initialize_action_head_scale.py',
                         '--experiment-config', str(cfg), '--scale', 'D16'],
        train_argv=['bash', config['training_source']+'/scripts/train/action_head_dualvln_sft/entry.sh',
                    '--action-head-as-tool-version', 'v6'])
    save(root/'fast-window-audit.json', audit)
    save(root/'launch-plan.json', launch)
    save(root/'schedule.json', dict(ok=True, anchors=audit['anchors'], global_batch=64, steps_per_epoch=epoch,
        training_layout=layout,
        epochs=5, checkpoint_steps=steps, checkpoints=len(steps), max_steps=5*epoch,
        half_epoch_rounding='first completed update at or after the boundary',
        assembled_receipt=bind(receipt), manifest=bind(manifest), config=bind(cfg)))
    return launch


def abort_if_failed(root):
    paths = [root/'failure.json', *[root/f'node{n}/failure.json' for n in range(4)]]
    if any(path.exists() for path in paths):
        raise RuntimeError('a follow-on stage failed; no automatic replay')


def wait_file(path, root, state_root, phase):
    while not Path(path).exists():
        abort_if_failed(root)
        save(state_root/'status.json', dict(state=phase, heartbeat=time.time(), pid=os.getpid()))
        time.sleep(5)
    return load(path)


def node_heartbeat(config, owner, nr, runner):
    """Bind real node and resident identities; emit through original stdout."""
    from dlc_workspace_stdout_heartbeat import task_heartbeat
    current = runner.process_identity(os.getpid())
    controller = owner.get('controller_identity')
    if controller is None:
        start = load(Path(owner['debug_output'])/'debug-start.json')
        controller = runner.process_identity(start['pid'])
    if not runner.same_process_alive(controller):
        raise RuntimeError('resident controller identity is not live')
    stream = open(f'/proc/{controller["pid"]}/fd/1', 'w', buffering=1)
    stop = threading.Event()
    def progress():
        state = load(nr/'status.json')
        child = state.get('identity')
        logfile = nr/'training.log'
        observed = dict(state=state, child_live=runner.same_process_alive(child) if child else None)
        if logfile.exists():
            stat = logfile.stat()
            observed['training_log'] = dict(bytes=stat.st_size, modified=stat.st_mtime)
        return dict(heartbeat=state['heartbeat'], stage=state['state'],
            mode='training' if state['state']=='training' else 'waiting_or_initialization',
            occupier_pid=load(Path(owner['debug_output'])/'debug-status.json').get('occupier_pid'),
            progress=observed)
    args = SimpleNamespace(controller_pid=current['pid'], controller_start_ticks=current['start_ticks'], output=nr)
    thread = threading.Thread(target=task_heartbeat, args=(args,), kwargs=dict(stop_event=stop,
        read_status=progress, write_receipt=lambda value:save(nr/'stdout-heartbeat.json',value),
        emit=lambda value:print(json.dumps(dict(value, training_node=nr.name)),file=stream,flush=True)),daemon=True)
    thread.start()
    return stop, stream


def node(config, rank):
    root, data = Path(config['root']), Path(config['data_root'])
    nr = root/f'node{rank}'
    owner = (config['resident_owners'][rank] if config.get('published_inputs') else
             load(data/f'node{rank}/node-owner/config.json'))
    if os.uname().nodename != owner['hostname']:
        raise ValueError('wrong target pod')
    c, runner = composition(config)
    lock = (nr/'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (nr/'start.json').exists():
        raise ValueError('training owner already launched; refusing replay')
    verify_frozen(config)
    save(nr/'start.json', dict(identity=runner.process_identity(os.getpid()), hostname=os.uname().nodename,
        started_at=time.time(), config=bind(config['_path'])))
    control = [config['python'], '-B', owner['debug_control']]
    runtime = Path(config['runtime'])/f'node{rank}'
    child = identity = unused_reservation = None
    activated = False
    env = runner.runtime_environment(runtime, 'training', {})
    save(nr/'status.json',dict(state='preparing_training',heartbeat=time.time(),identity=None))
    heartbeat_stop, heartbeat_stream = node_heartbeat(config, owner, nr, runner)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    try:
        if not handoff_data_ready(config):
            raise RuntimeError('collection owners are not all complete')
        # Never overwrite the live collector's lease. It has finished above.
        idle = load(Path(owner['debug_output'])/'debug-status.json')
        if idle['mode'] != 'idle' or time.time()-idle['heartbeat'] > 30:
            raise RuntimeError('data owner has not handed back its idle controller')
        if rank == 0:
            save(nr/'status.json',dict(state='cpu_initialization_with_idle_reservation',heartbeat=time.time(),identity=None))
            launch = prepare(config, c)
            if config.get('warm_start'):
                proof = dict(warm_start=config['warm_start'], optimizer_state_restored=False)
                checked(config['warm_start']['report'])
                for record in config['warm_start']['artifacts']:
                    if bind(record['path']) != record:
                        raise ValueError('warm start checkpoint changed before initialization')
            else:
                init_env = runner.runtime_environment(runtime, 'initialization', launch['environment'])
                init_env['CUDA_VISIBLE_DEVICES'] = ''  # CPU initializer; keep idle GPU reservation.
                with (nr/'initialization.log').open('a') as log:
                    subprocess.run(launch['initialize_argv'], cwd=config['training_source'], env=init_env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
                proof = c.verify_training_product('initialize', launch, load(root/'schedule.json')['max_steps'])
                if load(Path(launch['environment']['HEAD_CHECKPOINT']).parent/'scale_report.json')['preserved_base_query_slots'] != 0:
                    raise ValueError('v6 requires all D16 queries freshly initialized')
            address = socket.gethostbyname(socket.gethostname())
            if address.startswith('127.'):
                raise ValueError('master address must be reachable from all four pods')
            with socket.socket() as probe:
                probe.bind((address, config['master_port']))
            save(root/'initialized.json', dict(ok=True, proof=proof, master_addr=address, master_port=config['master_port']))
        initialized = wait_file(root/'initialized.json', root, nr, 'waiting_initialization_with_idle_gpu_reservation')
        launch = load(root/'launch-plan.json')
        env = runner.runtime_environment(runtime, 'training', launch['environment'])
        for key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE','PET_NODE_RANK'):
            env.pop(key, None)
        layout = training_layout(config)
        if (launch['environment']['NPROC_PER_NODE'] != str(layout['nproc_per_node'])
                or launch['environment']['ACTION_HEAD_ANCHORS_PER_GPU'] != str(layout['anchors_per_gpu'])):
            raise ValueError('launch plan and explicit training layout differ')
        env.update(NNODES='4', NODE_RANK=str(rank), MASTER_ADDR=initialized['master_addr'],
            MASTER_PORT=str(initialized['master_port']), CUDA_VISIBLE_DEVICES=','.join(map(str, layout['devices'][rank])))
        subprocess.run(control+['active', '--output', owner['debug_output'], '--lease-seconds', '600'], env=env, check=True)
        activated = True
        runner.assert_gpus_idle(list(range(16)))
        training_gpu_preflight(config, rank, env)
        unused_reservation = reserve_training_idle_gpus(config, rank, runner, env)
        save(nr/'ready.json', dict(ok=True, hostname=os.uname().nodename, config=bind(root/'experiment_config.json')))
        for peer in range(4):
            wait_file(root/f'node{peer}/ready.json', root, nr, 'waiting_four_training_nodes')
        if unused_reservation is not None:
            unused_reservation.check()
        with (nr/'training.log').open('a') as log:
            child = subprocess.Popen(launch['train_argv'], cwd=config['training_source'], env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        identity = runner.process_identity(child.pid)
        save(nr/'training-start.json', dict(identity=identity, command=launch['train_argv'],
            nnodes=4, node_rank=rank, training_layout=layout, master_addr=env['MASTER_ADDR'], started_at=time.time(), config=bind(root/'experiment_config.json')))
        renewed = time.monotonic()
        while child.poll() is None:
            abort_if_failed(root)
            if unused_reservation is not None:
                unused_reservation.check()
            save(nr/'status.json', dict(state='training', heartbeat=time.time(), identity=identity,
                process_identity_live=runner.same_process_alive(identity)))
            if time.monotonic()-renewed >= 300:
                subprocess.run(control+['active', '--output', owner['debug_output'], '--lease-seconds','600'], env=env, check=True)
                renewed = time.monotonic()
            time.sleep(5)
        if child.returncode:
            raise RuntimeError(f'training rank {rank} exited {child.returncode}')
        members = []
        for proc in Path('/proc').glob('[0-9]*/stat'):
            try:
                fields = proc.read_text().rsplit(')', 1)[1].split()
                if fields[0] != 'Z' and int(fields[2]) == child.pid:
                    members.append(int(proc.parent.name))
            except (OSError, ValueError, IndexError):
                continue
        if members:
            raise RuntimeError(f'training process group still has live children: {members}')
        save(nr/'result.json', dict(ok=True, returncode=0, finished_at=time.time()))
    except BaseException as exc:
        save(nr/'failure.json', dict(error=repr(exc), failed_at=time.time()))
        raise
    finally:
        if child is not None and child.poll() is None:
            runner.stop_owned(child, identity)
        if unused_reservation is not None:
            unused_reservation.stop('training_finished_or_failed')
        if activated:
            result = subprocess.run(control+['idle','--output',owner['debug_output']], env=env, capture_output=True, text=True)
            idle_ready = False
            deadline = time.monotonic()+180
            while result.returncode == 0 and time.monotonic() < deadline:
                idle = load(Path(owner['debug_output'])/'debug-status.json')
                if idle.get('mode') == 'idle' and idle.get('transition') == 'ready' and idle.get('occupier_pid'):
                    idle_ready = True
                    break
                time.sleep(5)
            save(nr/'idle-return.json', dict(ok=idle_ready, ack=result.stdout[-2000:], error=result.stderr[-1000:]))
            if not idle_ready:
                save(nr/'failure.json', dict(error='idle controller restoration failed', failed_at=time.time()))
        shutil.rmtree(runtime, ignore_errors=True)
        heartbeat_stop.set()
        # The daemon owns this stream until interpreter exit; do not close it
        # while its final flush may still be in flight.


def platform(*args):
    result = subprocess.run(['bash','-c','source "$1"; shift; exec "$@"','training-handoff',
        str(USER_ROOT/'env_train_platform.sh'), str(USER_ROOT/'bin/train-platform'), *map(str,args)],
        capture_output=True, text=True, timeout=55)
    if result.returncode:
        raise RuntimeError('official platform command failed: '+result.stderr[-1000:])
    payload = json.loads(result.stdout)
    if payload.get('http_status') != 200 or payload['response']['code'] != 2000:
        raise RuntimeError('official platform query failed')
    return payload['response']['data']


def snapshot(config, state, dlc_state):
    save(Path(config['root'])/'training-status-snapshot.json', dict(job_id=config['job_id'],
        status=state, dlc_status=dlc_state, scope='data_then_training_execution',
        observed_at=time.time(), source='independent_verified_training_handoff_supervisor'))


def checkpoint_proof(root, launch, step, visual_provenance=None):
    cp = root/'checkpoints'/Path(launch['environment']['OUTPUT_DIR']).name/f'global_step_{step:06d}'
    report = load(cp/'report.json')
    env = launch['environment']
    nodes, local_size, anchors = (int(env[k]) for k in ('NNODES', 'NPROC_PER_NODE', 'ACTION_HEAD_ANCHORS_PER_GPU'))
    if nodes != 4 or (local_size, anchors) not in ((16, 1), (8, 2)):
        raise ValueError('unsupported checkpoint training layout')
    offset = int(env.get('ACTION_HEAD_STEP_OFFSET', '0'))
    version = env.get('ACTION_HEAD_AS_TOOL_VERSION', 'v6')
    expected = dict(ok=True, action_head_as_tool_version=version, global_step=step, local_step=step-offset,
        step_offset=offset, world_size=nodes*local_size, anchors_per_gpu=anchors, global_batch_size=64, configured_fsdp_size=1,
        frozen_query_backend=('legacy_suffix' if env.get('ACTION_HEAD_FM_PREFIX_REUSE') == '1' else 'query_only'),
        training_experiment_config_sha256=launch['experiment_config_sha256'],
        scale_name='D16', frozen_qwen_exported=False)
    if any(report.get(key) != value for key,value in expected.items()):
        raise ValueError('checkpoint differs from v6 four-node training contract')
    visual = dict(fast_visual_input_mode='live_frozen_visual_v1',
        persisted_visual_features=False, fast_visual_backbone=env['ACTION_HEAD_FAST_BACKBONE'])
    # Older frozen exporters omit these fields. Never manufacture a checkpoint
    # report: admit omissions only with separately bound live-pipeline evidence.
    missing = [key for key in visual if key not in report]
    if any(key in report and report[key] != value for key,value in visual.items()):
        raise ValueError('checkpoint differs from live visual contract')
    if missing:
        if visual_provenance is None:
            raise ValueError('checkpoint missing visual metadata and bound pipeline provenance')
        provenance = checked(visual_provenance)
        if (provenance.get('training_root') != str(root.resolve())
                or provenance.get('visual_contract') != visual
                or provenance.get('launch_plan') != bind(root/'launch-plan.json')
                or provenance.get('experiment_config_sha256') != launch['experiment_config_sha256']):
            raise ValueError('checkpoint visual provenance binding differs')
        for item in provenance['evidence']:
            if bind(item['path']) != item:
                raise ValueError('checkpoint visual provenance evidence changed: '+item['path'])
    if any(report['qwen_audit'].values()) or not all(report[key] > 0 for key in ('query_parameter_delta_l2','action_head_parameter_delta_l2')):
        raise ValueError('frozen/query/head update audit failed')
    if sha(cp/'training_experiment.json') != launch['experiment_config_sha256']:
        raise ValueError('checkpoint config mismatch')
    files = ['action_head/fast_slow_action_head.pt','action_head/fast_slow_action_head_config.json',
             'dualvln_query/dualvln_query_embeddings.pt','report.json','training_experiment.json']
    return dict(step=step, artifacts=[bind(cp/name) for name in files],
        visual_metadata_source='checkpoint_report' if not missing else 'bound_frozen_live_pipeline',
        missing_report_fields=missing, visual_provenance=visual_provenance if missing else None)


def validate_platform_detail(detail, config):
    """Missing observations are unknown; positive identity/terminal evidence is fatal."""
    expected = dict(jobId=config['job_id'], createdBy='anonymous',
        quotaId=config.get('quota_id','quota-anonymous'), podCount=4, priority=8,
        spaceId=config.get('workspace','150120'))
    for key, value in expected.items():
        actual = detail.get(key)
        if actual is not None and str(actual) != str(value):
            raise ValueError('bound training workspace identity/resources changed: '+key)
    if detail.get('status') in TERMINAL:
        raise RuntimeError('four-node workspace ended: '+detail['status'])
    if any(detail.get(key) is None for key in expected) or not detail.get('status'):
        return False
    pods = detail.get('podInfo')
    if not isinstance(pods, list) or not pods:
        return False
    workers = [p for p in pods if isinstance(p, dict) and p.get('type') != 'aimaster']
    for pod in workers:
        name, start = pod.get('podName'), pod.get('startTime')
        if name and start and (name not in config['pod_start_times']
                              or config['pod_start_times'][name] != start):
            raise RuntimeError('bound pods were replaced; verify process identities before recovery')
    return (len(workers) == 4 and all(p.get('startTime') and p.get('status') for p in workers)
            and {p.get('podName') for p in workers} == set(config['pod_start_times']))


def coordinator(config):
    root = Path(config['root'])
    lock = (root/'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    continuation = config.get('coordinator_continuation')
    if not continuation and (root/'coordinator-start.json').exists():
        raise ValueError('coordinator already started; inspect original identity before any recovery')
    c, runner = composition(config)
    verify_frozen(config)
    if continuation:
        generation = Path(continuation['output'])
        prior = checked(continuation['previous_start'])
        if (not generation.is_relative_to(root) or (generation/'start.json').exists()
                or runner.same_process_alive(prior['identity'])
                or not checked(continuation['dispatched']).get('ok')
                or not load(root/'initialized.json').get('ok')):
            raise ValueError('coordinator continuation requires dead prior owner and dispatched training')
        abort_if_failed(root)
        for rank in range(4):
            if (root/f'node{rank}/failure.json').exists() or not (root/f'node{rank}/training-start.json').exists():
                raise ValueError('coordinator continuation cannot launch or recover training nodes')
        start_path = generation/'start.json'
    else:
        start_path = root/'coordinator-start.json'
    save(start_path, dict(identity=runner.process_identity(os.getpid()), started_at=time.time(), config=bind(config['_path'])))
    save(root/'active-coordinator.json', bind(start_path))
    dispatched = bool(continuation or config.get('nodes_prelaunched'))
    published = {}
    query_failures = 0
    try:
        while True:
            abort_if_failed(root)
            try:
                detail = platform('detail','--task-id',config['task_id'])
            except (RuntimeError, subprocess.TimeoutExpired, ValueError, KeyError):
                detail = None
            # Do not propagate absent/partial platform observations into a
            # failure broadcast which kills otherwise healthy GPU workers.
            observed = detail is not None and validate_platform_detail(detail, config)
            if not observed:
                query_failures += 1
                save(root/'status.json', dict(state='platform_query_retry', failures=query_failures,
                    heartbeat=time.time(), pid=os.getpid(), platform_observation_available=False))
                print(json.dumps(dict(event='platform_observation_unavailable',
                    failures=query_failures, dispatched=dispatched, time=time.time())), flush=True)
                if not dispatched:
                    time.sleep(30)
                    continue
            else:
                query_failures = 0
                snapshot(config, 'Running', detail['status'])
            if observed and not dispatched and handoff_data_ready(config):
                pods = {p['podName'] for p in detail['podInfo'] if p.get('type') != 'aimaster' and p['status'] == 'Running'}
                if pods != set(config['pods']):
                    raise RuntimeError('four bound pods are not all Running')
                evidence = (dict(source_publication=bind(config['published_inputs']),
                    indexes=[bind(Path(config['published_index'])/name/'summary.json') for name in ('r2r','rxr','sage')])
                    if config.get('published_inputs') else dict(
                    completions=[bind(Path(config['data_root'])/f'node{n}/completion.json') for n in range(4)],
                    assembly=bind(Path(config['data_root'])/'global/receipts/assemble.json')))
                save(root/'data-handoff.json', dict(ok=True, checked_at=time.time(), **evidence))
                for rank, pod in enumerate(config['pods']):
                    receipt = root/f'platform/dispatch-node{rank}.json'
                    # Record intention BEFORE the remote call. Timeout never triggers replay.
                    save(root/f'platform/dispatch-intent-node{rank}.json', dict(pod=pod, attempted_at=time.time()))
                    result = subprocess.run(['bash','-c','source "$1"; shift; exec "$@"','training-dispatch',
                        str(USER_ROOT/'env_train_platform.sh'),str(USER_ROOT/'bin/train-platform'),'exec',
                        '--workspace',str(config.get('workspace','150120')),
                        '--job-id',config['job_id'],'--pod',pod,'--ws-proxy-env','--timeout','35',
                        '--command-file',str(root/f'node{rank}/launch.sh'),'--receipt',str(receipt)],
                        capture_output=True,text=True,timeout=55)
                    if result.returncode or not receipt.exists() or load(receipt).get('remote_exit_code') != 0:
                        raise RuntimeError(f'dispatch {rank} unsuccessful/unknown; inspect receipt, do not replay')
                dispatched = True
                save(root/'dispatched.json', dict(ok=True, time=time.time(), pods=config['pods']))
            if (root/'initialized.json').exists():
                schedule, launch = load(root/'schedule.json'), load(root/'launch-plan.json')
                output = Path(launch['environment']['OUTPUT_DIR'])
                for step in schedule['checkpoint_steps']:
                    if step not in published and (output/f'global_step_{step:06d}/report.json').exists():
                        published[step] = checkpoint_proof(root, launch, step, config.get('checkpoint_visual_provenance'))
                steps = sorted(published)
                save(root/'checkpoint-index.json', dict(ok=True, checkpoint_root=str(output), steps=steps,
                    checkpoints=[str(output/f'global_step_{step:06d}') for step in steps], proofs=[published[s] for s in steps]))
                if all((root/f'node{n}/result.json').exists() and (root/f'node{n}/idle-return.json').exists() for n in range(4)):
                    if steps != schedule['checkpoint_steps']:
                        raise ValueError('training ended without all half-epoch checkpoints')
                    for n in range(4):
                        if not load(root/f'node{n}/result.json')['ok'] or not load(root/f'node{n}/idle-return.json')['ok']:
                            raise ValueError('a training node failed or did not return to idle')
                    save(root/'complete.json', dict(ok=True, completed_at=time.time(), schedule=schedule, checkpoints=steps))
                    snapshot(config, 'Succeeded', detail['status'] if observed else 'Unknown')
                    return
            save(root/'status.json', dict(state='waiting_data' if not dispatched else 'training_handoff',
                heartbeat=time.time(), pid=os.getpid(), dispatched=dispatched, published_checkpoints=sorted(published),
                platform_observation_available=observed, platform_query_failures=query_failures))
            time.sleep(30)
    except BaseException as exc:
        save(root/'failure.json', dict(error=repr(exc), failed_at=time.time()))
        snapshot(config, 'Failed', 'see official task detail')
        raise


def companion(config):
    """Reserve idle evaluation GPUs until the first real checkpoint is ready."""
    root = Path(config['root'])
    output = Path(config['companion_root'])
    sys.path.insert(0, config['training_source']+'/scripts/eval')
    from v6_rl_checkpoint_companion import WaitingGpuReservation
    from watch_checkpoints import stop_process_group
    reservation = WaitingGpuReservation(dict(waiting_gpu_reservation=dict(enabled=True,
        gpu_count=16, python=config['python'], startup_timeout=120,
        script=config['training_source']+'/verl/scripts/train/action_head_sft_data/gpu_occupier.py')),
        output, stop_process_group)
    verify_frozen(config)
    child = None
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    try:
        reservation.start()
        while True:
            abort_if_failed(root)
            reservation.check()
            path = root/'checkpoint-index.json'
            if path.exists() and load(path).get('steps'):
                break
            state = dict(state='waiting_first_checkpoint_with_idle_gpu_reservation',
                heartbeat=time.time(), pid=os.getpid())
            save(output/'status.json', state)
            print(json.dumps(dict(event='companion_liveness', **state)), flush=True)
            reservation.wait(30)
        schedule = load(root/'schedule.json')
        os.environ['EXPECTED_CHECKPOINT_COUNT'] = str(schedule['checkpoints'])
        reservation.stop('first_verified_checkpoint_ready')
        child = subprocess.Popen(['bash',config['companion_entry']], start_new_session=True)
        next_console_heartbeat = 0.
        while child.poll() is None:
            state = dict(state='evaluating', heartbeat=time.time(), pid=os.getpid(), child_pid=child.pid)
            save(output/'status.json', state)
            if time.monotonic() >= next_console_heartbeat:
                print(json.dumps(dict(event='companion_liveness', **state)), flush=True)
                next_console_heartbeat = time.monotonic()+30
            time.sleep(5)
        if child.returncode or not (output/'evaluation/SWEEP_COMPLETE.json').exists():
            raise RuntimeError(f'companion failed: exit={child.returncode}')
        save(output/'complete.json', dict(ok=True, completed_at=time.time()))
    except BaseException as exc:
        save(output/'failure.json', dict(error=repr(exc), failed_at=time.time()))
        raise
    finally:
        if child is not None and child.poll() is None:
            stop_process_group(child)
        reservation.stop('companion_job_exit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=('coordinator','node','companion'), required=True)
    parser.add_argument('--node', type=int, choices=range(4))
    args = parser.parse_args()
    config = load(args.config)
    config['_path'] = str(args.config)
    if not all(Path(config[key]).resolve().is_relative_to(ART) for key in ('root','data_root')):
        raise ValueError('follow-on outputs/source snapshot must be on authorized data disk')
    if config.get('git_runtime'):
        verify_frozen(config)
    elif not Path(config['training_source']).resolve().is_relative_to(ART):
        raise ValueError('local source requires an explicit verified Git runtime')
    if args.mode == 'node':
        if args.node is None:
            parser.error('--node is required')
        node(config, args.node)
    elif args.mode == 'companion':
        companion(config)
    else:
        coordinator(config)


if __name__ == '__main__':
    main()
