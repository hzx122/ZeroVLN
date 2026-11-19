"""Native-record, streamed B1 parity protocol for zero-wave compaction.

The zero advantages below are an explicit controlled intervention on fixed
archived inputs, not relabelled rollout rewards or evidence of policy gain.
"""
from __future__ import annotations

import torch
import hashlib
import os
from pathlib import Path

from action_head_rl.packing_audit import collective_signature, packed_signature

SCHEMA = 'v6-native-zero-wave-replay-v2'
REPEATABILITY_SCHEMA = 'v6-native-zero-wave-repeatability-v2'
SCENARIOS = ('zero_only', 'zero_first', 'zero_last',
             'single_zero_only', 'single_zero_first', 'single_zero_last')
VARIANTS = ('original', 'compact')


def single_token_prefix_probe(record):
    """Exercise the single-label graph with a REAL recorded prompt and next ID.

    This is an audit-only transformation of a longer capture, NOT a claim that
    the request stopped after one token. Never invent scheduler completion
    provenance or pass this through the native capture certification path.
    """
    from action_head_rl.decision_records import SINGLE_TOKEN_TARGET_MODE
    from action_head_rl.text_policy import validate_response_mask
    if record.get('record_kind') != 'text_policy' or record.get('text_single_token_target_id') is not None:
        raise ValueError('prefix probe requires an original multi-token text capture')
    ids = torch.as_tensor(record['all_input_ids'])
    prompt = int(record['text_prompt_length'])
    validate_response_mask(ids, record['text_response_mask'], prompt_length=prompt,
                           response_end=record['text_response_end'])
    if not bool(record['text_response_mask'][prompt]):
        raise ValueError('first response ID is not an original sampled target')
    routes, positions = record['all_routed_experts'], record['all_positions']
    if len(routes) != len(ids) or not torch.equal(positions.cpu(), torch.arange(len(ids))):
        raise ValueError('prefix probe requires aligned native routes and positions')
    mask = torch.zeros(prompt, dtype=torch.bool)
    mask[-1] = True
    return dict(record, condition_id=record['condition_id']+'/audit-single-prefix',
        all_input_ids=ids[:prompt], all_routed_experts=routes[:prompt], all_positions=positions[:prompt],
        token_count=prompt, text_prompt_length=prompt, text_response_end=prompt, text_response_mask=mask,
        text_single_token_target_id=int(ids[prompt]), text_target_mode=SINGLE_TOKEN_TARGET_MODE,
        advantage=0., _audit_single_prefix=dict(original_condition_id=record['condition_id'],
            original_prompt_length=prompt, sampled_target_id=int(ids[prompt]),
            native_single_token_completion=False))


def replay_streams(fixtures, *, world=12, repeatability_only=False):
    from action_head_rl.trainer import _zero_advantage_training_records
    from action_head_rl.worker import shard_option_record_groups, _condition_family
    indices = [fixture['rollout_index'] for fixture in fixtures]
    if world < 1 or len(fixtures) < 8 or indices != sorted(set(indices)):
        raise ValueError('zero-wave audit requires all fixed captures in original order, at least eight')
    if {fixture['credit_mode'] for fixture in fixtures} != {'episode_grpo'}:
        raise ValueError('zero-wave replay requires original episode GRPO fixtures')
    original = []
    for fixture in fixtures:
        for record in fixture['rows']:
            row = {key: value for key, value in record.items()
                   if not key.startswith(('_condition_pack_', '_joint_'))}
            original.append(row)
    if not all(any(row['advantage'] != 0 and _condition_family(row) == family
                   for row in original) for family in ('fm', 'text')):
        raise ValueError('fixed captures must contain nonzero FM and text signals')
    zero = [dict(row, advantage=0.) for row in original]
    single_zero = [single_token_prefix_probe(row) if _condition_family(row) == 'text'
                   else dict(row, advantage=0.) for row in original]

    def shard(rows, compact):
        retained, mass, report = _zero_advantage_training_records(rows, compact_zero_waves=compact)
        ranks, _ = shard_option_record_groups(retained, world,
            balance_by_token_flops=True, condition_batch_size=1,
            max_padded_tokens=0, condition_token_bucket_size=0,
            align_text_families=bool(report.get('zero_wave_compacted')))
        return ranks, mass, report

    ordinary = shard(original, False)
    baseline_zero, compact_zero = shard(zero, False), shard(zero, True)
    baseline_single, compact_single = shard(single_zero, False), shard(single_zero, True)
    scenarios, summary = {}, {}
    for name in SCENARIOS:
        scenarios[name], summary[name] = {}, {}
        order = name.removeprefix('single_')
        pair = (baseline_single, compact_single) if name.startswith('single_') else (baseline_zero, compact_zero)
        for variant, zero_wave in zip(VARIANTS, pair):
            waves = ([zero_wave] if order == 'zero_only' else
                     [zero_wave, ordinary] if order == 'zero_first' else [ordinary, zero_wave])
            scenarios[name][variant] = [[wave[0][rank] for wave in waves] for rank in range(world)]
            summary[name][variant] = dict(logical_mass=sum(wave[1] for wave in waves),
                wave_reports=[wave[2] for wave in waves],
                rank_record_counts=[[len(rows) for rows in wave[0]] for wave in waves])
    if repeatability_only:
        # A/A negative control plus A/B measurement under the same lifecycle.
        # Diagnosis only: this reduced scope cannot authorize integration.
        original = scenarios['zero_first']['original']
        schedule = summary['zero_first']['original']
        return ({'zero_first_repeat': {'original': original, 'repeat': original,
                                      'compact': scenarios['zero_first']['compact']}},
                {'zero_first_repeat': {'original': schedule, 'repeat': schedule,
                                      'compact': summary['zero_first']['compact']}})
    return scenarios, summary


def pack_production_wave(rows, processor):
    """Match worker._backward_record_batches B1 (not the B2 packing helper)."""
    from action_head_rl.worker import consecutive_record_groups, record_group_to_tensordict
    from verl.utils.tensordict_utils import maybe_fix_3d_position_ids
    result = []
    for group in consecutive_record_groups([dict(row, logical_batch_valid=1.) for row in rows], True):
        data = record_group_to_tensordict(group, processor, 'cpu')
        maybe_fix_3d_position_ids(data)
        result.append(data)
    return result


def signatures(waves, processor, *, cache=None):
    result = []
    for rows in waves:
        # A scenario reuses the identical immutable wave in different positions.
        # Cache only its layout, never its tensor payload or a different input.
        key = tuple(id(row) for row in rows)
        if cache is None or key not in cache:
            packed = pack_production_wave(rows, processor)
            signature = dict(tensors=packed_signature(packed), collective=collective_signature(packed))
            if cache is not None:
                cache[key] = signature
        else:
            signature = cache[key]
        result.append(signature)
    return result


def cpu_preflight(scenarios, source):
    from transformers import AutoProcessor
    from action_head_rl.joint_audit import temporary_environment
    processor = AutoProcessor.from_pretrained(source['binding']['qwen'], local_files_only=True)
    result = {}
    cache = {}
    with temporary_environment(ACTION_HEAD_TEXT_POLICY='1'):
        for name, variants in scenarios.items():
            result[name] = {}
            for variant, ranks in variants.items():
                checked = [signatures(waves, processor, cache=cache) for waves in ranks]
                if any([wave['collective'] for wave in rank] !=
                       [wave['collective'] for wave in checked[0]] for rank in checked):
                    raise ValueError(f'zero-wave collective shapes differ: {name}/{variant}')
                result[name][variant] = checked
    return result


def model_state_receipt(engine, *, gradients=False):
    """Hash every local shard; keep no second CPU copy of Adam moments."""
    from action_head_rl.checkpoint import training_state_digest
    from action_head_rl.packing_audit_worker import local
    if gradients:
        result = {}
        for name, parameter in engine.module.named_parameters():
            value = None if parameter.grad is None else local(parameter.grad).detach().cpu()
            result[name] = dict(present=value is not None, sha256=training_state_digest(value),
                finite=value is None or bool(torch.isfinite(value).all()),
                zero=value is None or not bool(torch.count_nonzero(value)))
        return result
    return dict(
        parameters={name: training_state_digest(parameter) for name, parameter in engine.module.named_parameters()},
        buffers={name: training_state_digest(buffer) for name, buffer in engine.module.named_buffers()},
        training_modes={name: module.training for name, module in engine.module.named_modules()},
        optimizer={name: training_state_digest(engine.optimizer.state.get(parameter, {}))
                   for name, parameter in engine.module.named_parameters()},
        groups=training_state_digest([{key: value for key, value in group.items() if key != 'params'}
                                     for group in engine.optimizer.param_groups]),
        scheduler=training_state_digest(engine.lr_scheduler.state_dict()),
        rng=training_state_digest(engine.checkpoint_manager.get_rng_state()))


def compare_receipts(baseline, candidate):
    """Deliberately exact: a failure needs investigation, never a silent tolerance change."""
    if set(baseline) != set(candidate):
        raise ValueError('receipt fields differ')
    return {key: baseline[key] == candidate[key] for key in baseline}


def prepare_plan(*, fixture_dirs, source_checkpoint, runtime_plan, output,
                 after_process_records=(), timeout_seconds=7200, repeatability_only=False):
    from action_head_rl.joint_audit import ARTIFACTS, read_json, sha256, validate_source
    from action_head_rl.packing_audit import load_fixture, replay_worker_environment
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(ARTIFACTS) or output == ARTIFACTS:
        raise ValueError('audit requires a new artifact output directory')
    if not after_process_records or timeout_seconds <= 0:
        raise ValueError('bind preceding process identities and a positive deadline')
    source = validate_source(source_checkpoint)
    fixtures = [load_fixture(path, source_binding=source['binding']) for path in fixture_dirs]
    scenarios, schedule = replay_streams(fixtures, repeatability_only=repeatability_only)
    runtime_path = Path(runtime_plan).resolve()
    runtime = read_json(runtime_path)
    environment = dict(runtime.get('environment', runtime.get('env', {})))
    old_cache = environment['ACTION_HEAD_LOCAL_RUNTIME_ROOT']
    cache = '/tmp/ah_zero/' + hashlib.sha256(str(output).encode()).hexdigest()[:12]
    environment = {key: value.replace(old_cache, cache) for key, value in environment.items()}
    repo = Path(__file__).resolve().parents[2]
    upstream = repo.parent/'upstream'
    overlay = repo.parent/'fast_kernel_overlay'
    if not (upstream/'verl').is_dir() or not overlay.is_dir():
        raise ValueError('freeze the complete upstream and overlay beside source before preparation')
    environment.update(replay_worker_environment(source, token_cap=0, token_bucket=0))
    environment.update(CUDA_VISIBLE_DEVICES=','.join(map(str, range(12))),
        ACTION_HEAD_COMPACT_ZERO_ADVANTAGE_WAVES='1',
        FAST_KERNEL_OVERLAY=str(overlay), VERL_SFT_CONFIG_DIR=str(upstream/'verl/trainer/config'),
        PYTHONPATH=':'.join(map(str, (overlay, repo/'verl', upstream, repo/'vln_demo/src'))))
    contract = source['contract']
    learner = dict(slow_model=source['binding']['qwen'], learner_parameter_dtype='fp32',
        gradient_checkpointing=contract['gradient_checkpointing'], fsdp_size=contract['fsdp_size'],
        fsdp_forward_prefetch=contract['fsdp_forward_prefetch'], learning_rate=contract['action_head_learning_rate'],
        weight_decay=contract['optimizer_weight_decay'], opt_steps=22)
    prerequisites = []
    for path in map(Path, after_process_records):
        value = read_json(path)
        if type(value.get('pid')) is not int or not str(value.get('start_ticks', '')).isdigit():
            raise ValueError('preceding process record lacks PID/start_ticks')
        prerequisites.append(dict(path=str(path.resolve()), sha256=sha256(path),
            identity=dict(pid=value['pid'], start_ticks=str(value['start_ticks']))))
    files = [runtime_path]
    for directory in (repo, upstream, overlay, repo.parent/'autotune_seed'):
        files.extend(path for path in directory.rglob('*') if path.is_file()
                     and '__pycache__' not in path.parts and path.suffix != '.pyc')
    qwen_bytes = read_json(Path(source['binding']['qwen'])/'model.safetensors.index.json')['metadata']['total_size']
    return dict(schema=REPEATABILITY_SCHEMA if repeatability_only else SCHEMA,
        output=str(output), source=source, world_size=12,
        physical_gpus=list(range(12)), renderer_gpu=15, scenarios=list(scenarios),
        variants=list(next(iter(scenarios.values()))),
        equality='bitwise local gradients, parameters, all optimizer tensors/groups, scheduler and RNG',
        advantage_intervention='all advantages set to zero on copies for zero waves; original targets retained for mixed waves',
        single_token_probe='prompt-only native IDs/routes/positions plus the actually sampled first label from each longer capture; audit graph coverage, NOT a certified one-token completion',
        fixtures=[{key: value for key, value in fixture.items() if key != 'rows'} for fixture in fixtures],
        schedule=schedule, cpu_preflight=cpu_preflight(scenarios, source), learner=learner,
        environment=environment, prerequisites=prerequisites, timeout_seconds=timeout_seconds,
        cpu_snapshot_estimate_bytes=int(qwen_bytes*(4.1 if repeatability_only else 3.1)),
        cpu_required_headroom_bytes=int(qwen_bytes*(4.1 if repeatability_only else 3.1))+128*1024**3,
        sources_sha256={str(path): sha256(path) for path in sorted(set(files))},
        scope=('A/A and A/B gradient-error diagnosis only; no compaction acceptance'
               if repeatability_only else
               'Controlled native replay only; no rollout, published weights, training checkpoint or learning-gain claim'))


def validate_plan(plan, *, preflight=True):
    from action_head_rl.joint_audit import sha256, validate_source, read_json
    from action_head_rl.packing_audit import load_fixture, validate_replay_environment
    repeatability = plan.get('schema') == REPEATABILITY_SCHEMA
    if (plan.get('schema') not in (SCHEMA, REPEATABILITY_SCHEMA) or plan.get('world_size') != 12
            or plan.get('scenarios') != (['zero_first_repeat'] if repeatability else list(SCENARIOS))
            or plan.get('variants') != (['original', 'repeat', 'compact'] if repeatability else list(VARIANTS))
            or plan['environment'].get('CUDA_VISIBLE_DEVICES') != ','.join(map(str, range(12)))
            or plan['environment'].get('ACTION_HEAD_COMPACT_ZERO_ADVANTAGE_WAVES') != '1'):
        raise ValueError('zero-wave protocol changed')
    validate_replay_environment(plan['environment'], plan['source'], token_cap=0, token_bucket=0)
    if 'fixture_collection' in plan:
        collection = plan['fixture_collection']
        actual = sorted(read_json(path)['rollout_index'] for path in
                        Path(collection['root']).glob('step_000017_rollout_*/complete.json'))
        if actual != collection['complete_indices']:
            raise ValueError('fixed native capture completion set changed')
    for path, expected in plan['sources_sha256'].items():
        if sha256(path) != expected:
            raise ValueError(f'zero-wave audit source changed: {path}')
    for row in plan['prerequisites']:
        if sha256(row['path']) != row['sha256']:
            raise ValueError('preceding process record changed')
    source = validate_source(plan['source']['checkpoint'])
    if source != plan['source']:
        raise ValueError('source16 state binding changed')
    fixtures = [load_fixture(row['directory'], source_binding=source['binding']) for row in plan['fixtures']]
    if [{key: value for key, value in fixture.items() if key != 'rows'} for fixture in fixtures] != plan['fixtures']:
        raise ValueError('native fixture metadata changed')
    scenarios, schedule = replay_streams(fixtures, repeatability_only=repeatability)
    if schedule != plan['schedule'] or (preflight and cpu_preflight(scenarios, source) != plan['cpu_preflight']):
        raise ValueError('zero-wave CPU schedule/signatures changed')
    return scenarios


def execute_plan(path, *, expected_sha256, cpu_receipt_sha256):
    """Fresh interpreter; never touches a live learner or the renderer."""
    import time
    from types import SimpleNamespace
    from action_head_rl.joint_audit import (read_json, sha256, write_new_json,
        resolve_rpc, process_alive, wall_clock_deadline)
    from action_head_rl.packing_audit import execution_gate
    path = Path(path).resolve()
    if sha256(path) != expected_sha256:
        raise ValueError('zero-wave plan hash changed')
    plan = read_json(path)
    output = Path(plan['output'])
    receipt = output/'execute_cpu_preflight.json'
    if (sha256(receipt) != cpu_receipt_sha256 or read_json(receipt) != dict(
            schema=plan['schema'], plan_sha256=expected_sha256, cpu_validation='passed')):
        raise ValueError('zero-wave GPU entry requires a hash-bound CPU receipt')
    if any(os.environ.get(key) != value for key, value in plan['environment'].items()):
        raise ValueError('start a fresh interpreter with the frozen plan environment')
    scenarios = validate_plan(plan, preflight=False)
    if torch.cuda.is_initialized():
        raise RuntimeError('driver initialized CUDA before the idle gate')
    gate = execution_gate(plan)
    write_new_json(output/'attempt.json', dict(plan_sha256=expected_sha256, pid=os.getpid(),
        start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19], gate=gate))
    execution_gate(plan)
    import ray
    from action_head_rl.zero_wave_audit_worker import ZeroWaveAuditWorker
    from action_head_rl.trainer import learner_config
    from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
    cache = Path(plan['environment']['ACTION_HEAD_LOCAL_RUNTIME_ROOT'])
    if not cache.is_relative_to('/tmp/ah_zero'):
        raise ValueError('zero-wave cache escaped local private directory')
    cache.mkdir(parents=True, exist_ok=True)
    learner, identities, result = None, [], None
    deadline = time.monotonic()+plan['timeout_seconds']
    def resolve(value):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('zero-wave replay RPC deadline expired')
        return resolve_rpc(value, timeout_seconds=remaining)
    try:
        with wall_clock_deadline(plan['timeout_seconds'], 'zero-wave Ray bootstrap'):
            ray.init(address='local', num_gpus=12, include_dashboard=False, _temp_dir=str(cache/'ray'),
                runtime_env={'env_vars': plan['environment']})
            learner = RayWorkerGroup(resource_pool=RayResourcePool(process_on_nodes=[12], max_colocate_count=4),
                ray_cls_with_init=RayClassWithInitArgs(ray.remote(max_restarts=0, max_task_retries=0)(ZeroWaveAuditWorker),
                    config=learner_config(SimpleNamespace(**plan['learner']))), device_name='cuda')
        identities = resolve(learner.audit_process_identity())
        write_new_json(output/'worker_processes.json', dict(plan_sha256=expected_sha256, ranks=identities))
        resolve(learner.execute_all_async('reset'))
        resolve(learner.execute_all_async('load_checkpoint',
            local_path=str(Path(plan['source']['checkpoint'])/'trainer_state'), del_local_after_load=False))
        fast = resolve(learner.execute_all_async('audit_qwen_fast_path'))
        rank_scenarios = [{name: {variant: ranks[rank] for variant, ranks in variants.items()}
                           for name, variants in scenarios.items()} for rank in range(12)]
        ready = resolve(learner.prepare_zero_replay(scenarios=rank_scenarios))
        write_new_json(output/'worker_ready_shapes.json', dict(plan_sha256=expected_sha256, ranks=ready))
        if (len(ready) != 12 or {row['rank'] for row in ready} != set(range(12)) or any(
                row['signatures'] != {name: {variant: ranks[row['rank']] for variant, ranks in variants.items()}
                                     for name, variants in plan['cpu_preflight'].items()} for row in ready)):
            raise RuntimeError('worker shapes differ from actual CPU signatures')
        ranks = resolve(learner.run_zero_replay(plan=dict(plan, plan_sha256=expected_sha256)))
        if len(ranks) != 12 or {row['rank'] for row in ranks} != set(range(12)):
            raise RuntimeError('zero-wave replay did not return all ranks')
        result = dict(schema=plan['schema'], plan_sha256=expected_sha256, fast_path=fast, ranks=ranks,
            status='passed' if all(row['passed'] for row in ranks) else 'failed', scope=plan['scope'])
    except Exception as exc:
        write_new_json(output/'failure.json', dict(type=type(exc).__name__, error=str(exc), automatic_retry=False))
        raise
    finally:
        errors = []
        if learner is not None:
            for actor in learner._workers:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception as exc:
                    errors.append(str(exc))
        if ray.is_initialized():
            try:
                ray.shutdown()
            except Exception as exc:
                errors.append(str(exc))
        remaining = [row for row in identities if process_alive(row)]
        until = time.monotonic()+10
        while remaining and time.monotonic() < until:
            time.sleep(.2)
            remaining = [row for row in identities if process_alive(row)]
        teardown = dict(remaining_owned_processes=remaining, errors=errors, gpu_idle_not_claimed=True)
        write_new_json(output/'teardown.json', teardown)
    if result is not None:
        result['teardown'] = teardown
        if remaining or errors:
            result['status'] = 'failed_teardown'
        write_new_json(output/'result.json', result)
        return result
