"""Replay a complete real D16 shard set through production learners.

The probe reuses the production learner config, Ray worker group and logical
batch implementation. Head-only never updates weights. Full performs two real
recompute updates, then freezes weights and optimizer for matched residency and
repeated-recompute observations. Neither mode saves a checkpoint or tests resume.
Use --validate-inputs for CPU preflight; GPU runs require exactly ten devices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace


REPO = Path('/workspace/ZeroVLN')
VERL_REPO = Path('/workspace/verl')
BASE = Path('/data/zerovln/action_head/rl/'
            'd16_resume_acceptance_20260911/optimization')
DEFAULT_INPUT = BASE / 'd16-local-full-piold-reuse-20260912-r2_runtime/piold_audit'
SFT = Path('/data/zerovln/action_head/sft_ckpt/'
           'action-head-v3-official-gvln-d16-randomhead-10ep-4node-gb64-20260908-r1/'
           'global_step_010000')
SLOW_MODEL = Path('/opt/external/user-e7a01947/Model/Qwen3.6-35B-A3B')
WORLD_SIZE = 10


def load_shard(path: Path):
    import torch

    # These are trusted, locally produced experiment payloads, not user uploads.
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        handle.seek(0)
        records = torch.load(handle, map_location='cpu', weights_only=False)
    return records, digest


def shard_summary(records):
    import torch
    from action_head_rl.worker import consecutive_record_groups, _validate_shared_option

    if not isinstance(records, list) or not records:
        raise ValueError('payload must contain a nonempty record list')
    required = {'condition_id', 'slow_request', 'slow_response', 'all_routed_experts',
                'all_positions', 'all_input_ids', 'query_routed_experts',
                'reference_visual_tokens', 'current_visual_tokens', 'states',
                'successors', 'timesteps', 'signed_dt', 'noise_scales', 'var_floor',
                'old_logprob', 'advantage'}
    valid, nonzero_advantages = 0.0, 0
    for record in records:
        if not isinstance(record, dict) or required - record.keys():
            raise ValueError('record is missing the real rollout/replay fields')
        if not record['condition_id']:
            raise ValueError('each replay record needs its original condition_id')
        validity = float(record.get('sample_valid', 1.0))
        if validity not in (0.0, 1.0) or not math.isfinite(float(record['advantage'])):
            raise ValueError('invalid sample validity or advantage')
        old = torch.as_tensor(record['old_logprob']).reshape(-1)
        if old.numel() != 1 or not torch.isfinite(old).all():
            raise ValueError('each chunk needs one finite source old_logprob')
        valid += validity
        nonzero_advantages += int(validity > 0 and float(record['advantage']) != 0.0)
    groups = consecutive_record_groups(records, reuse=True)
    identities = [group[0]['condition_id'] for group in groups]
    if len(identities) != len(set(identities)):
        raise ValueError('nonconsecutive duplicate condition IDs could break FSDP alignment')
    for group in groups:
        if len(group) > 1:
            _validate_shared_option(group)
    return {'records': len(records), 'logical_valid': valid,
            'nonzero_advantage_chunks': nonzero_advantages,
            'group_lengths': [len(group) for group in groups],
            'valid_condition_ids': sorted({row['condition_id'] for row in records
                                           if row.get('sample_valid', 1.0)})}


def inspect_payloads(directory: Path, world_size: int = WORLD_SIZE):
    if world_size != WORLD_SIZE:
        raise ValueError('this probe requires all ten original ranks; resharding is not implemented')
    expected = {f'records_rank_{rank:03d}.pt' for rank in range(world_size)}
    actual = {path.name for path in directory.glob('records_rank_*.pt')}
    if actual != expected:
        raise ValueError(f'incomplete or different-world payload set: '
                         f'missing={sorted(expected - actual)}, extra={sorted(actual - expected)}')
    rows, seen_conditions = [], set()
    for rank in range(world_size):
        path = directory / f'records_rank_{rank:03d}.pt'
        records, digest = load_shard(path)
        row = shard_summary(records)
        duplicates = seen_conditions.intersection(row['valid_condition_ids'])
        if duplicates:
            raise ValueError(f'valid options are split across ranks: {sorted(duplicates)}')
        seen_conditions.update(row['valid_condition_ids'])
        row.update(rank=rank, path=str(path), sha256=digest, bytes=path.stat().st_size)
        rows.append(row)
        del records
    if any(row['group_lengths'] != rows[0]['group_lengths'] for row in rows):
        raise ValueError('rank option/chunk loop counts are not aligned for FSDP collectives')
    logical_valid = sum(row['logical_valid'] for row in rows)
    if logical_valid <= 0:
        raise ValueError('replay has no valid chunks')
    return {'world_size': world_size, 'logical_valid': logical_valid,
            'nonzero_advantage_chunks': sum(row['nonzero_advantage_chunks'] for row in rows),
            'padded_records': sum(row['records'] for row in rows), 'shards': rows}


def compare_tensors(left, right, *, gradient=False):
    """Match real outputs and local FSDP gradient shards without all-gathering."""
    import torch

    if left is None or right is None:
        return {'passed': left is None and right is None,
                'missing_baseline': left is None, 'missing_candidate': right is None}
    if left.shape != right.shape or left.dtype != right.dtype:
        return {'passed': False, 'baseline_shape': list(left.shape),
                'candidate_shape': list(right.shape), 'baseline_dtype': str(left.dtype),
                'candidate_dtype': str(right.dtype)}
    left, right = left.float(), right.float()
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    difference = left - right
    relative = float(difference.norm() / left.norm().clamp_min(1e-12))
    maximum = float(difference.abs().max()) if difference.numel() else 0.0
    passed = finite and (relative <= 1e-3 if gradient else
                         torch.allclose(left, right, atol=1e-5, rtol=1e-5))
    return {'passed': bool(passed), 'finite': finite, 'max_abs': maximum,
            'relative_l2': relative, 'exact': bool(torch.equal(left, right))}


def prepare_worker_payload(worker, specification):
    import torch

    rank = torch.distributed.get_rank()
    if torch.distributed.get_world_size() != WORLD_SIZE:
        raise ValueError('worker world differs from complete replay shard set')
    expected = specification['inputs']['shards'][rank]
    records, digest = load_shard(Path(expected['path']))
    actual = shard_summary(records)
    invalid = digest != expected['sha256'] or any(
        actual[key] != expected[key] for key in ('records', 'logical_valid', 'group_lengths'))
    invalid_tensor = torch.tensor(int(invalid), device=torch.cuda.current_device())
    torch.distributed.all_reduce(invalid_tensor, op=torch.distributed.ReduceOp.MAX)
    if invalid_tensor.item():
        raise RuntimeError('payload changed after driver preflight')
    logical_valid = torch.tensor(actual['logical_valid'], device=torch.cuda.current_device())
    torch.distributed.all_reduce(logical_valid)
    if float(logical_valid) != specification['inputs']['logical_valid']:
        raise RuntimeError('global valid chunk denominator changed')
    logical_valid = float(logical_valid)
    root = Path(specification['output_dir']) / f'rank_{rank:03d}'
    root.mkdir(parents=True, exist_ok=False)
    return rank, records, logical_valid, root, digest


def run_worker_probe(worker, specification):
    """Run through VERL's existing execute_with_func_generator entry point."""
    if specification.get('strict_determinism', False):
        if specification.get('train_scope') != 'full':
            raise ValueError('strict determinism diagnostic requires full train scope')
        helpers = runpy.run_path(str(REPO / 'scripts/eval/action_head_d16_determinism_audit.py'))
        with helpers['strict_determinism_audit'](worker, specification['output_dir']):
            return run_full_worker_probe(worker, specification)
    if specification.get('train_scope', 'head_only') == 'full':
        return run_full_worker_probe(worker, specification)
    import torch
    from torch.distributed.tensor import DTensor
    from action_head_rl.checkpoint import training_state_digest

    engine = worker.engine
    if engine.action_head_train_scope.value != 'head_only':
        raise ValueError('residency comparison requires head_only')
    if any(parameter.requires_grad for name in ('qwen', 'query_encoder')
           for parameter in getattr(engine.module, name).parameters()):
        raise ValueError('brain and query must be frozen')
    rank, records, logical_valid, root, digest = prepare_worker_payload(worker, specification)

    parameters = dict(engine.module.named_parameters())
    trainable = {name: parameter for name, parameter in parameters.items() if parameter.requires_grad}
    local = lambda value: value.to_local() if isinstance(value, DTensor) else value
    initial_weights = {name: local(parameter.detach()).cpu().clone()
                       for name, parameter in trainable.items()}
    initial_gradients = {name: parameter.grad.detach().clone()
                         for name, parameter in parameters.items() if parameter.grad is not None}
    rng = engine.checkpoint_manager.get_rng_state()
    torch.save(rng, root / 'initial_rng.pt')
    optimizer_digest = training_state_digest(engine.optimizer.state_dict())
    scheduler_digest = training_state_digest(engine.lr_scheduler.state_dict())
    original_optimizer = engine.optimizer_step
    original_scheduler = engine.lr_scheduler_step
    original_forward = engine.forward_backward_batch
    original_query = engine.module.query_forward.forward_full
    environment = {key: os.environ.get(key) for key in (
        'ACTION_HEAD_FROZEN_WEIGHT_RESIDENCY', 'ACTION_HEAD_SINGLE_FORWARD_AUDIT_DIR',
        'ACTION_HEAD_PIOLD_AUDIT_DIR', 'ACTION_HEAD_OPTION_CONDITION_REUSE',
        'ACTION_HEAD_OLD_LOGPROB_MODE')}
    state, passes, captures = {}, [], {}

    def observe_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        if state['capture']:
            state['losses'].append(result['loss'].detach().cpu().clone())
            state['logprobs'].append(result['model_output']['new_logprob'].detach().cpu().clone())
        return result

    def observe_query(*args, **kwargs):
        result = original_query(*args, **kwargs)
        if state['capture']:
            state['query_hidden'].append(result.detach().cpu().clone())
        return result

    def no_optimizer_step():
        torch.cuda.synchronize()
        state['before_optimizer_wall_seconds'] = time.perf_counter() - state['started']
        state['max_memory_allocated_gb'] = torch.cuda.max_memory_allocated() / 1024 ** 3
        state['optimizer_callback_calls'] += 1
        if state['capture']:
            state['gradients'] = {
                name: None if parameter.grad is None else local(parameter.grad.detach()).clone().cpu()
                for name, parameter in trainable.items()}
        return 0.0

    def no_scheduler_step():
        state['scheduler_callback_calls'] += 1
        return float(engine.optimizer.param_groups[0]['lr'])

    def run_pass(variant, phase, iteration, capture=False):
        state.clear()
        state.update(capture=capture, losses=[], logprobs=[], query_hidden=[],
                     optimizer_callback_calls=0, scheduler_callback_calls=0)
        os.environ['ACTION_HEAD_FROZEN_WEIGHT_RESIDENCY'] = str(int(variant == 'candidate'))
        engine.checkpoint_manager.load_rng_state(rng)
        torch.distributed.barrier()
        torch.cuda.synchronize()
        state['started'] = time.perf_counter()
        metrics = worker.train_record_batches(records, logical_valid)
        torch.cuda.synchronize()
        wall = time.perf_counter() - state['started']
        if state['optimizer_callback_calls'] != 1 or state['scheduler_callback_calls'] != 1:
            raise RuntimeError('production logical batch did not invoke exactly one intercepted update')
        row = {'variant': variant, 'phase': phase, 'iteration': iteration,
               'train_call_wall_seconds': wall,
               'before_optimizer_wall_seconds': state['before_optimizer_wall_seconds'],
               'forward_host_seconds': metrics['forward_seconds'],
               'backward_host_seconds': metrics['backward_seconds'],
               'max_memory_allocated_gb': state['max_memory_allocated_gb'],
               'conditions': metrics['condition_forwards'], 'records': len(records),
               'logical_valid': logical_valid, 'optimizer_steps': 0, 'scheduler_steps': 0,
               'rng_after': training_state_digest(engine.checkpoint_manager.get_rng_state())}
        if capture:
            captures[variant] = {key: state[key] for key in (
                'gradients', 'losses', 'logprobs', 'query_hidden')}
            captures[variant]['rng_after'] = row['rng_after']
        passes.append(row)
        with (root / 'passes.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        print(json.dumps({'event': 'learner_replay_pass', 'rank': rank, **row}), flush=True)
        torch.distributed.barrier()

    report = {'rank': rank, 'status': 'running', 'optimizer_steps': 0, 'scheduler_steps': 0,
              'resume_verified': False, 'payload_sha256': digest, 'passes': passes,
              'learning_rates': optimizer_learning_rate_report(engine, specification)}
    try:
        require_requested_learning_rates(report['learning_rates'])
        os.environ.update(ACTION_HEAD_OPTION_CONDITION_REUSE='1',
                          ACTION_HEAD_OLD_LOGPROB_MODE='single_forward')
        os.environ.pop('ACTION_HEAD_SINGLE_FORWARD_AUDIT_DIR', None)
        os.environ.pop('ACTION_HEAD_PIOLD_AUDIT_DIR', None)
        engine.optimizer_step = no_optimizer_step
        engine.lr_scheduler_step = no_scheduler_step
        engine.forward_backward_batch = observe_forward
        engine.module.query_forward.forward_full = observe_query
        for iteration in range(specification['warmup_pairs']):
            for variant in ('baseline', 'candidate'):
                run_pass(variant, 'warmup', iteration)
        for iteration in range(specification['measure_pairs']):
            order = ('baseline', 'candidate') if iteration % 2 == 0 else ('candidate', 'baseline')
            for variant in order:
                run_pass(variant, 'measure', iteration)
        for variant in ('baseline', 'candidate'):
            run_pass(variant, 'audit', 0, capture=True)
            torch.save(captures[variant], root / f'{variant}_audit.pt')

        baseline, candidate = captures['baseline'], captures['candidate']
        comparisons = {
            'gradients': {name: compare_tensors(baseline['gradients'][name],
                                               candidate['gradients'][name], gradient=True)
                          for name in trainable},
        }
        for key in ('losses', 'logprobs', 'query_hidden'):
            left, right = baseline[key], candidate[key]
            if len(left) != len(right) or not left:
                comparisons[key] = [{'passed': False, 'baseline_count': len(left),
                                     'candidate_count': len(right)}]
            else:
                comparisons[key] = [compare_tensors(a, b) for a, b in zip(left, right)]
        unchanged_weights = all(torch.equal(initial_weights[name], local(parameter.detach()).cpu())
                                for name, parameter in trainable.items())
        unchanged_optimizer = optimizer_digest == training_state_digest(engine.optimizer.state_dict())
        unchanged_scheduler = scheduler_digest == training_state_digest(engine.lr_scheduler.state_dict())
        rng_match = baseline['rng_after'] == candidate['rng_after']
        finite_nonzero_gradient = any(value is not None and bool(torch.isfinite(value).all())
                                      and bool(value.count_nonzero())
                                      for value in baseline['gradients'].values())
        passed = all(row['passed'] for row in comparisons['gradients'].values()) and all(
            row['passed'] for key in ('losses', 'logprobs', 'query_hidden') for row in comparisons[key])
        passed = passed and unchanged_weights and unchanged_optimizer and unchanged_scheduler and rng_match
        report.update(status='pass' if passed else 'fail', comparisons=comparisons,
                      trainable_weights_unchanged=unchanged_weights,
                      optimizer_state_unchanged=unchanged_optimizer,
                      scheduler_state_unchanged=unchanged_scheduler, rng_match=rng_match,
                      has_nonzero_gradient=finite_nonzero_gradient,
                      gradient_metadata={name: {'global_shape': list(parameter.shape),
                          'local_shape': list(local(parameter).shape), 'dtype': str(parameter.dtype),
                          'placements': [str(item) for item in parameter.placements]
                          if isinstance(parameter, DTensor) else []}
                          for name, parameter in trainable.items()})
    except Exception as error:
        report.update(status='error', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        engine.optimizer_step, engine.lr_scheduler_step = original_optimizer, original_scheduler
        engine.forward_backward_batch = original_forward
        engine.module.query_forward.forward_full = original_query
        for name, parameter in parameters.items():
            parameter.grad = initial_gradients.get(name)
        engine.checkpoint_manager.load_rng_state(rng)
        for key, value in environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        report['rng_restored'] = training_state_digest(rng) == training_state_digest(
            engine.checkpoint_manager.get_rng_state())
        (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return {'rank': rank, 'status': report['status'], 'report_path': str(root / 'report.json')}


def parameter_family(name):
    if name.startswith(('qwen.', 'query_forward.model.')):
        return 'qwen'
    if name.startswith(('query.', 'query_encoder.', 'query_forward.query_encoder.')):
        return 'query'
    return 'head' if name.startswith(('head.', 'action_head.')) else 'other'


def optimizer_learning_rate_report(engine, specification):
    requested = {'learning_rate': float(specification.get('learning_rate', 1e-6)),
                 'vlm_learning_rate': float(specification.get('vlm_learning_rate', 1e-6))}
    families = {id(parameter): parameter_family(name)
                for name, parameter in engine.module.named_parameters()}
    groups = []
    for index, group in enumerate(engine.optimizer.param_groups):
        group_families = sorted({families.get(id(parameter), 'other') for parameter in group['params']})
        # The production engine names the two actual Adam groups; classification
        # also supports the nameless optimizer used by the CPU Head harness.
        is_vlm = group.get('name') == 'slow_vlm' or group_families == ['qwen']
        expected = requested['vlm_learning_rate' if is_vlm else 'learning_rate']
        actual = float(group['lr'])
        groups.append({'index': index, 'name': group.get('name'), 'lr': actual,
            'initial_lr': float(group.get('initial_lr', actual)),
            'parameter_families': group_families, 'parameter_tensors': len(group['params']),
            'requested_lr': expected, 'matches_requested': math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0)})
    return {'requested': requested, 'actual_optimizer_groups': groups,
            'matches_requested': bool(groups) and all(row['matches_requested'] for row in groups)}


def require_requested_learning_rates(report):
    import torch

    invalid = torch.tensor(int(not report['matches_requested']), device=torch.cuda.current_device())
    torch.distributed.all_reduce(invalid, op=torch.distributed.ReduceOp.MAX)
    if invalid.item():
        raise RuntimeError('actual optimizer learning rate differs from requested probe configuration')


def tensor_change_statistics(before, after):
    """Exact element counts and FP64 chunked norms of the actual local shards."""
    import torch

    if before.shape != after.shape or before.dtype != after.dtype:
        raise ValueError('local tensor shape/dtype changed during an optimizer update')
    before, after = before.reshape(-1), after.reshape(-1)
    changed = nonzero = 0
    delta_sq = parameter_sq = maximum = 0.0
    finite = True
    for start in range(0, before.numel(), 1 << 20):
        left, right = before[start:start + (1 << 20)], after[start:start + (1 << 20)]
        changed += int(torch.count_nonzero(left != right))
        nonzero += int(torch.count_nonzero(left))
        left, right = left.double(), right.double()
        finite = finite and bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        delta = right - left
        delta_sq += float(delta.square().sum())
        parameter_sq += float(left.square().sum())
        maximum = max(maximum, float(delta.abs().max()))
    return {'elements': before.numel(), 'changed_elements': changed,
            'changed_fraction': changed / max(1, before.numel()),
            'before_nonzero_elements': nonzero, 'finite': finite,
            'delta_squared_sum': delta_sq, 'parameter_squared_sum': parameter_sq,
            'update_l2': math.sqrt(delta_sq),
            'relative_update_l2': math.sqrt(delta_sq) / max(math.sqrt(parameter_sq), 1e-30),
            'max_abs_update': maximum}


def aggregate_update_statistics(rows):
    totals = {}
    for name, row in rows.items():
        family = parameter_family(name)
        total = totals.setdefault(family, {'elements': 0, 'changed_elements': 0,
            'before_nonzero_elements': 0, 'delta_squared_sum': 0.0,
            'parameter_squared_sum': 0.0, 'max_abs_update': 0.0, 'finite': True})
        for key in ('elements', 'changed_elements', 'before_nonzero_elements',
                    'delta_squared_sum', 'parameter_squared_sum'):
            total[key] += row[key]
        total['max_abs_update'] = max(total['max_abs_update'], row['max_abs_update'])
        total['finite'] = total['finite'] and row['finite']
    for row in totals.values():
        row['changed_fraction'] = row['changed_elements'] / max(1, row['elements'])
        row['update_l2'] = math.sqrt(row['delta_squared_sum'])
        row['relative_update_l2'] = row['update_l2'] / max(math.sqrt(row['parameter_squared_sum']), 1e-30)
    return totals


def optimizer_state_summary(engine, parameters):
    """Inspect the actual optimizer; a nonzero gradient is not an update audit."""
    import torch
    from torch.distributed.tensor import DTensor

    rows, bytes_by_dtype = {}, {}
    for name, parameter in parameters.items():
        values = engine.optimizer.state.get(parameter, {})
        tensors = {}
        for key, value in values.items():
            if not torch.is_tensor(value):
                continue
            value = value.to_local() if isinstance(value, DTensor) else value
            dtype = str(value.dtype)
            nbytes = value.numel() * value.element_size()
            bytes_by_dtype[dtype] = bytes_by_dtype.get(dtype, 0) + nbytes
            tensors[key] = {'dtype': dtype, 'shape': list(value.shape), 'bytes': nbytes}
        step = values.get('step')
        if isinstance(step, DTensor):
            step = step.to_local()
        rows[name] = {'parameter_dtype': str(parameter.dtype), 'state': tensors,
                      'step': float(step.item()) if torch.is_tensor(step) else step}
    return {'class': f'{type(engine.optimizer).__module__}.{type(engine.optimizer).__name__}',
            'groups': [{key: value for key, value in group.items() if key != 'params'}
                       for group in engine.optimizer.param_groups],
            'state_bytes_by_dtype': bytes_by_dtype, 'parameters': rows}


def full_comparison_status(repeat_passed, candidate_passed, state_preserved):
    if not state_preserved:
        return 'fail'
    if not repeat_passed:
        return 'inconclusive'
    return 'pass' if candidate_passed else 'fail'


def full_collective_gate(rows):
    baseline, repeat, candidate = (rows[name]['collectives'] for name in ('baseline', 'repeat', 'candidate'))
    ag, rs = 'all_gather_into_tensor', 'reduce_scatter_tensor'
    return (baseline[rs] > 0 and repeat[rs] == candidate[rs] == baseline[rs]
            and 0 < candidate[ag] < baseline[ag] and repeat[ag] == baseline[ag])


def full_gradient_coverage_report(statistics, queries, expected_conditions, *, device):
    """Require global gradient coverage; an empty FSDP shard is locally N/A."""
    import torch

    family_names = ('qwen', 'query', 'head', 'other')
    count_names = ('parameter_elements', 'gradient_elements', 'nonzero_elements',
                   'nonfinite_ranks')
    local_counts = {name: dict.fromkeys(count_names, 0) for name in family_names}
    for name, row in statistics.items():
        counts = local_counts[parameter_family(name)]
        counts['parameter_elements'] += row['local_parameter_elements']
        counts['gradient_elements'] += row['local_gradient_elements']
        counts['nonzero_elements'] += row['nonzero_elements']
        counts['nonfinite_ranks'] = int(bool(counts['nonfinite_ranks']) or not row['finite'])
    expected = set(range(expected_conditions))
    observed = {row['forward_index'] for row in queries}
    with_gradient = {row['forward_index'] for row in queries if row['gradient_events']}
    complete_events = all(
        (not row['requires_grad'] or bool(row['gradient_events']))
        and [event['event_index'] for event in row['gradient_events']]
            == list(range(len(row['gradient_events']))) for row in queries)
    finite_events = all(bool(torch.isfinite(event['gradient']).all())
                        for row in queries for event in row['gradient_events'])
    boundary_complete = bool(expected) and observed == with_gradient == expected and complete_events and finite_events
    values = [local_counts[family][name] for family in family_names for name in count_names]
    values.append(int(not boundary_complete))
    counts = torch.tensor(values, dtype=torch.int64, device=device)
    # Fixed-size SUM includes empty ranks and gives every report the same global
    # decision. This runs after timing and outside the counted AG/RS passes.
    torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
    values = counts.cpu().tolist()
    global_counts = {family: dict(zip(count_names, values[index * len(count_names):
        (index + 1) * len(count_names)])) for index, family in enumerate(family_names)}
    passed = (values[-1] == 0
        and all(row['nonfinite_ranks'] == 0 for row in global_counts.values())
        and all(global_counts[family]['nonzero_elements'] > 0 for family in ('qwen', 'query', 'head')))
    return {'passed': passed,
        'local_parameter_elements': {name: row['parameter_elements'] for name, row in local_counts.items()},
        'local_gradient_elements': {name: row['gradient_elements'] for name, row in local_counts.items()},
        'local_nonzero_elements': {name: row['nonzero_elements'] for name, row in local_counts.items()},
        'local_family_status': {name: 'not_applicable_empty_shard' if row['parameter_elements'] == 0
            else 'nonempty_shard' for name, row in local_counts.items()},
        'local_boundary_gradient_coverage': boundary_complete,
        'global_boundary_incomplete_ranks': values[-1], 'global_counts': global_counts}


def run_full_worker_probe(worker, specification):
    """Warm real Adam twice, then isolate repeatability and Full residency."""
    import torch
    from torch.distributed.tensor import DTensor
    from action_head_rl.checkpoint import training_state_digest
    from action_head_rl.worker import consecutive_record_groups
    from action_head_rl import residency
    from contextlib import contextmanager

    engine = worker.engine
    if engine.action_head_train_scope.value != 'full':
        raise ValueError('Full replay requires Full learners')
    parameters = dict(engine.module.named_parameters())
    if not parameters or any(not parameter.requires_grad or parameter.dtype != torch.bfloat16
                             for parameter in parameters.values()):
        raise ValueError('Full replay requires the actual trainable BF16 parameter shards')
    if not isinstance(engine.optimizer, torch.optim.AdamW):
        raise ValueError('Full replay must retain the configured real torch.optim.AdamW')
    rank, records, logical_valid, root, digest = prepare_worker_payload(worker, specification)
    local = lambda value: value.to_local() if isinstance(value, DTensor) else value
    # This is the already initialized, cast, and sharded policy. Never compare
    # an FP32 SFT export with BF16 runtime weights and call that an RL update.
    weights = {name: local(parameter.detach()).cpu().clone() for name, parameter in parameters.items()}
    initial_gradients = {name: parameter.grad.detach().clone()
                        for name, parameter in parameters.items() if parameter.grad is not None}
    rng = engine.checkpoint_manager.get_rng_state()
    torch.save(rng, root / 'initial_rng.pt')
    original_optimizer, original_scheduler = engine.optimizer_step, engine.lr_scheduler_step
    original_forward = engine.forward_backward_batch
    original_query = engine.module.query_forward.forward_full
    original_residency = residency.full_weight_residency
    original_collectives = {name: getattr(torch.distributed, name) for name in
                            ('all_gather_into_tensor', 'reduce_scatter_tensor')}
    env_keys = ('ACTION_HEAD_FROZEN_WEIGHT_RESIDENCY', 'ACTION_HEAD_FULL_WEIGHT_RESIDENCY_GIB',
                'ACTION_HEAD_SINGLE_FORWARD_AUDIT_DIR', 'ACTION_HEAD_PIOLD_AUDIT_DIR',
                'ACTION_HEAD_OPTION_CONDITION_REUSE', 'ACTION_HEAD_OLD_LOGPROB_MODE',
                'ACTION_HEAD_MULTI_CONDITION_BATCH_SIZE')
    environment = {key: os.environ.get(key) for key in env_keys}
    state, passes, updates, captures, reference_gradients = {}, [], [], {}, {}
    optimizer_digest = scheduler_digest = None
    groups = consecutive_record_groups(records, reuse=True)
    report = {'rank': rank, 'status': 'running', 'train_scope': 'full',
              'optimizer_steps': 0, 'scheduler_steps': 0, 'resume_verified': False,
              'payload_sha256': digest, 'passes': passes, 'real_updates': updates,
              'weight_reference': 'actual initialized BF16 FSDP local shards, before real optimizer step',
              'full_weight_residency_gib': specification['full_weight_residency_gib'],
              'learning_rates': optimizer_learning_rate_report(engine, specification),
              'gradient_artifact_policy': 'one baseline local gradient clone in CPU RAM; '
                  'stream other gradients into per-parameter comparisons; no full weight/gradient dumps'}

    @contextmanager
    def observe_residency(target):
        with original_residency(target) as entry:
            if entry is not None and 'residency_reports' in state:
                state['residency_reports'].append(entry)
            yield entry

    def counted_collective(name):
        def call(*args, **kwargs):
            if 'collectives' in state:
                state['collectives'][name] += 1
            return original_collectives[name](*args, **kwargs)
        return call

    def observe_query(*args, **kwargs):
        value = original_query(*args, **kwargs)
        if not state.get('capture') or state.get('current_forward') is None:
            return value
        active = state['current_forward']
        row = {'forward_index': active['index'], 'query_call_index': active['query_calls'],
               'condition_id': groups[active['index']][0]['condition_id'],
               'grad_enabled': torch.is_grad_enabled(), 'requires_grad': value.requires_grad,
               'query_hidden': value.detach().clone(), 'gradient_events': []}
        active['query_calls'] += 1
        state['queries'].append(row)
        if value.requires_grad:
            def observe_gradient(gradient):
                row['gradient_events'].append({'event_index': len(row['gradient_events']),
                                               'gradient': gradient.detach().clone()})
            value.register_hook(observe_gradient)
        return value

    def observe_forward(*args, **kwargs):
        if not state.get('capture'):
            return original_forward(*args, **kwargs)
        active = {'index': len(state['losses']), 'query_calls': 0}
        state['current_forward'] = active
        try:
            result = original_forward(*args, **kwargs)
            state['losses'].append(result['loss'].detach().clone())
            state['logprobs'].append(result['model_output']['new_logprob'].detach().clone())
            state['actual_modes'].append(os.environ['ACTION_HEAD_OLD_LOGPROB_MODE'])
            return result
        finally:
            state['current_forward'] = None

    def observe_optimizer():
        torch.cuda.synchronize()
        state['before_optimizer_wall_seconds'] = time.perf_counter() - state['started']
        state['optimizer_callback_calls'] += 1
        if state['real_update']:
            state['parameters_with_grad'] = [name for name, parameter in parameters.items()
                                             if parameter.grad is not None]
            started = time.perf_counter()
            grad_norm = original_optimizer()
            torch.cuda.synchronize()
            state['optimizer_wall_seconds'] = time.perf_counter() - started
            report['optimizer_steps'] += 1
            return grad_norm
        if state['capture']:
            comparisons = {}
            nonzero = False
            families = {}
            tensor_gradient_statistics = {}
            for name, parameter in parameters.items():
                value = None if parameter.grad is None else local(parameter.grad.detach()).cpu()
                parameter_elements = local(parameter.detach()).numel()
                family = families.setdefault(parameter_family(name), {
                    'tensors_with_grad': 0, 'elements_with_grad': 0, 'nonzero_elements': 0, 'finite': True,
                    'local_parameter_elements': 0, 'local_gradient_elements': 0})
                family['local_parameter_elements'] += parameter_elements
                if value is not None:
                    finite = bool(torch.isfinite(value).all())
                    count = int(value.count_nonzero())
                    family['tensors_with_grad'] += 1
                    family['elements_with_grad'] += value.numel()
                    family['local_gradient_elements'] += value.numel()
                    family['nonzero_elements'] += count
                    family['finite'] = family['finite'] and finite
                    nonzero = nonzero or (finite and count > 0)
                    tensor_gradient_statistics[name] = {
                        'elements': value.numel(), 'nonzero_elements': count, 'finite': finite,
                        'family': parameter_family(name)}
                else:
                    tensor_gradient_statistics[name] = {
                        'elements': 0, 'nonzero_elements': 0, 'finite': True,
                        'family': parameter_family(name), 'gradient_is_none': True}
                tensor_gradient_statistics[name].update(
                    local_parameter_elements=parameter_elements,
                    local_gradient_elements=0 if value is None else value.numel(),
                    coverage_status='not_applicable_empty_shard' if parameter_elements == 0 else
                        ('gradient_missing' if value is None else 'nonempty_gradient'))
                if state['variant'] == 'baseline':
                    reference_gradients[name] = None if value is None else value.clone()
                else:
                    before = reference_gradients[name]
                    if before is None or value is None:
                        comparisons[name] = compare_tensors(before, value, gradient=True)
                    elif before.shape != value.shape or before.dtype != value.dtype:
                        comparisons[name] = {'passed': False, 'shape_or_dtype_changed': True}
                    else:
                        delta = tensor_change_statistics(before, value)
                        comparisons[name] = {'passed': delta['finite'] and delta['relative_update_l2'] <= 1e-3,
                            'finite': delta['finite'], 'exact': delta['changed_elements'] == 0,
                            'relative_l2': delta['relative_update_l2'], 'max_abs': delta['max_abs_update']}
            state['gradient_comparisons'] = comparisons
            state['has_nonzero_gradient'] = nonzero
            state['gradient_families'] = families
            if state['variant'] == 'baseline':
                report['baseline_tensor_gradient_statistics'] = tensor_gradient_statistics
        return 0.0

    def observe_scheduler():
        state['scheduler_callback_calls'] += 1
        if state['real_update']:
            report['scheduler_steps'] += 1
            return original_scheduler()
        return float(engine.optimizer.param_groups[0]['lr'])

    def freeze_piold(label):
        state.clear()
        state['capture'] = False
        worker.clear_staged_records()
        try:
            worker.stage_record_batches(records)
            started = time.perf_counter()
            inference_rng = training_state_digest(engine.checkpoint_manager.get_rng_state())
            metrics = worker.compute_staged_logprobs()
            torch.cuda.synchronize()
            records[:] = [dict(row) for row in worker._staged_action_head_records]
            values = torch.cat([torch.as_tensor(row['old_logprob']).reshape(-1) for row in records]).cpu()
            if len(values) != len(records) or not bool(torch.isfinite(values).all()):
                raise ValueError('invalid refreshed proximal probabilities')
            row = {'label': label, 'wall_seconds': time.perf_counter() - started,
                   'rng_unchanged': inference_rng == training_state_digest(engine.checkpoint_manager.get_rng_state()),
                   'old_logprob_digest': training_state_digest(values), 'metrics': metrics}
            torch.save({'old_logprobs': values, 'condition_ids': [record['condition_id'] for record in records],
                        'sample_valid': [record.get('sample_valid', 1.0) for record in records]},
                       root / f'piold_{label}.pt')
            report.setdefault('piold_refreshes', []).append(row)
            invalid = torch.tensor(int(not row['rng_unchanged']), device=torch.cuda.current_device())
            torch.distributed.all_reduce(invalid, op=torch.distributed.ReduceOp.MAX)
            if invalid.item():
                raise RuntimeError('proximal inference changed RNG on at least one learner rank')
        finally:
            worker.clear_staged_records()

    def run_pass(variant, phase, iteration, *, capture=False, real_update=False):
        state.clear()
        state.update(variant=variant, capture=capture, real_update=real_update,
                     losses=[], logprobs=[], queries=[], actual_modes=[], current_forward=None,
                     optimizer_callback_calls=0, scheduler_callback_calls=0,
                     residency_reports=[], collectives={name: 0 for name in original_collectives})
        os.environ['ACTION_HEAD_FULL_WEIGHT_RESIDENCY_GIB'] = (
            str(specification['full_weight_residency_gib']) if variant == 'candidate' else '0')
        if not real_update:
            engine.checkpoint_manager.load_rng_state(rng)
        torch.distributed.barrier()
        torch.cuda.synchronize()
        state['started'] = time.perf_counter()
        metrics = worker.train_record_batches(records, logical_valid)
        torch.cuda.synchronize()
        row = {'variant': variant, 'phase': phase, 'iteration': iteration,
               'train_call_wall_seconds': time.perf_counter() - state['started'],
               'before_optimizer_wall_seconds': state['before_optimizer_wall_seconds'],
               'optimizer_wall_seconds': state.get('optimizer_wall_seconds', 0.0),
               'forward_host_seconds': metrics['forward_seconds'],
               'backward_host_seconds': metrics['backward_seconds'],
               'max_memory_allocated_gb': torch.cuda.max_memory_allocated() / 1024 ** 3,
               'conditions': metrics['condition_forwards'], 'records': len(records), 'logical_valid': logical_valid,
               'optimizer_steps': int(real_update), 'scheduler_steps': int(real_update),
               'collectives': dict(state['collectives']), 'residency_reports': state['residency_reports'],
               'actual_old_logprob_mode': os.environ['ACTION_HEAD_OLD_LOGPROB_MODE'],
               'rng_after': training_state_digest(engine.checkpoint_manager.get_rng_state())}
        for key in ('step_peak_memory_reserved_gb', 'post_update_memory_allocated_gb',
                    'post_update_memory_reserved_gb', 'post_update_device_used_gb', 'grad_norm'):
            if key in metrics:
                row[key] = metrics[key]
        if state['optimizer_callback_calls'] != 1 or state['scheduler_callback_calls'] != 1:
            raise RuntimeError('production batch did not call optimizer/scheduler exactly once')
        residency_ran = len(state['residency_reports']) == (1 if variant == 'candidate' else 0)
        invalid = torch.tensor(int(not residency_ran), device=torch.cuda.current_device())
        torch.distributed.all_reduce(invalid, op=torch.distributed.ReduceOp.MAX)
        if invalid.item():
            raise RuntimeError('Full residency context was not invoked with the requested budget')
        # Auditing never contributes to the reported train-call measurement.
        check_started = time.perf_counter()
        if real_update:
            changes = {}
            for name, parameter in parameters.items():
                current = local(parameter.detach()).cpu()
                changes[name] = tensor_change_statistics(weights[name], current)
                weights[name].copy_(current)
            actual_optimizer = optimizer_state_summary(engine, parameters)
            expected = state['parameters_with_grad']
            initialized = bool(expected) and all(
                {'exp_avg', 'exp_avg_sq'} <= actual_optimizer['parameters'][name]['state'].keys()
                and actual_optimizer['parameters'][name]['step'] == iteration + 1 for name in expected)
            update = {'step': iteration + 1, 'parameters': changes,
                      'families': aggregate_update_statistics(changes),
                      'optimizer_state': actual_optimizer, 'adam_state_initialized': initialized,
                      'parameters_with_grad': expected, 'grad_norm': metrics.get('grad_norm')}
            updates.append(update)
            (root / f'update_{iteration + 1:03d}.json').write_text(json.dumps(update, indent=2) + '\n')
            invalid = torch.tensor(int(not initialized), device=torch.cuda.current_device())
            torch.distributed.all_reduce(invalid, op=torch.distributed.ReduceOp.MAX)
            if invalid.item():
                raise RuntimeError('real Full optimizer step did not create/advance expected Adam states')
        else:
            unchanged_weights = all(torch.equal(weights[name], local(parameter.detach()).cpu())
                                    for name, parameter in parameters.items())
            row.update(trainable_weights_unchanged=unchanged_weights,
                       optimizer_state_unchanged=optimizer_digest == training_state_digest(engine.optimizer.state_dict()),
                       scheduler_state_unchanged=scheduler_digest == training_state_digest(engine.lr_scheduler.state_dict()))
        if capture:
            for query in state['queries']:
                query['query_hidden'] = query['query_hidden'].cpu()
                for event in query['gradient_events']:
                    event['gradient'] = event['gradient'].cpu()
            captures[variant] = {'losses': [value.cpu() for value in state['losses']],
                'logprobs': [value.cpu() for value in state['logprobs']], 'queries': state['queries'],
                'gradient_comparisons': state['gradient_comparisons'], 'actual_modes': state['actual_modes'],
                'has_nonzero_gradient': state['has_nonzero_gradient'],
                'gradient_families': state['gradient_families'], 'rng_after': row['rng_after']}
            torch.save(captures[variant], root / f'{variant}_boundaries.pt')
        row['post_measurement_audit_seconds'] = time.perf_counter() - check_started
        passes.append(row)
        with (root / 'passes.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        print(json.dumps({'event': 'full_learner_replay_pass', 'rank': rank, **row}), flush=True)
        torch.distributed.barrier()

    def compare_capture(variant):
        before, after = captures['baseline'], captures[variant]
        comparisons = {'gradients': after['gradient_comparisons']}
        for key in ('losses', 'logprobs'):
            left, right = before[key], after[key]
            comparisons[key] = ([compare_tensors(a, b) for a, b in zip(left, right)]
                if left and len(left) == len(right) else [{'passed': False, 'length_mismatch': True}])
        query_rows = []
        left = {(row['forward_index'], row['query_call_index']): row for row in before['queries']}
        right = {(row['forward_index'], row['query_call_index']): row for row in after['queries']}
        for key in sorted(left.keys() | right.keys()):
            a, b = left.get(key), right.get(key)
            row = {'forward_index': key[0], 'query_call_index': key[1], 'passed': False}
            if a is not None and b is not None:
                row['query_hidden'] = compare_tensors(a['query_hidden'], b['query_hidden'])
                row['query_gradients'] = [compare_tensors(x['gradient'], y['gradient'], gradient=True)
                    for x, y in zip(a['gradient_events'], b['gradient_events'])]
                row['gradient_event_counts'] = [len(a['gradient_events']), len(b['gradient_events'])]
                row['passed'] = (row['query_hidden']['passed']
                    and len(a['gradient_events']) == len(b['gradient_events'])
                    and a['requires_grad'] == b['requires_grad']
                    and all(value['passed'] for value in row['query_gradients']))
            query_rows.append(row)
        comparisons['query_boundaries'] = query_rows
        passed = all(row['passed'] for row in comparisons['gradients'].values()) and all(
            row['passed'] for key in ('losses', 'logprobs', 'query_boundaries') for row in comparisons[key])
        passed = passed and bool(query_rows) and before['rng_after'] == after['rng_after']
        return bool(passed), comparisons

    try:
        require_requested_learning_rates(report['learning_rates'])
        os.environ.update(ACTION_HEAD_OPTION_CONDITION_REUSE='1', ACTION_HEAD_OLD_LOGPROB_MODE='recompute',
                          ACTION_HEAD_FROZEN_WEIGHT_RESIDENCY='0', ACTION_HEAD_FULL_WEIGHT_RESIDENCY_GIB='0',
                          ACTION_HEAD_MULTI_CONDITION_BATCH_SIZE=str(
                              specification.get('multi_condition_batch_size', 1)))
        for key in ('ACTION_HEAD_SINGLE_FORWARD_AUDIT_DIR', 'ACTION_HEAD_PIOLD_AUDIT_DIR'):
            os.environ.pop(key, None)
        engine.optimizer_step, engine.lr_scheduler_step = observe_optimizer, observe_scheduler
        engine.forward_backward_batch, engine.module.query_forward.forward_full = observe_forward, observe_query
        residency.full_weight_residency = observe_residency
        for name in original_collectives:
            setattr(torch.distributed, name, counted_collective(name))
        for iteration in range(2):
            freeze_piold(f'update_{iteration + 1:03d}')
            run_pass('baseline', 'real_update', iteration, real_update=True)
        freeze_piold('fixed_anchor')
        rng = engine.checkpoint_manager.get_rng_state()
        torch.save(rng, root / 'fixed_anchor_rng.pt')
        optimizer_digest = training_state_digest(engine.optimizer.state_dict())
        scheduler_digest = training_state_digest(engine.lr_scheduler.state_dict())
        report.update(fixed_optimizer_digest=optimizer_digest, fixed_scheduler_digest=scheduler_digest)
        for iteration in range(specification['warmup_pairs']):
            for variant in ('baseline', 'candidate'):
                run_pass(variant, 'warmup', iteration)
        for iteration in range(specification['measure_pairs']):
            order = ('baseline', 'candidate') if iteration % 2 == 0 else ('candidate', 'baseline')
            for variant in order:
                run_pass(variant, 'measure', iteration)
        for variant in ('baseline', 'repeat', 'candidate'):
            run_pass(variant, 'audit', 0, capture=True)
        repeat_passed, repeat_comparisons = compare_capture('repeat')
        candidate_passed, candidate_comparisons = compare_capture('candidate')
        audit_rows = {row['variant']: row for row in passes if row['phase'] == 'audit'}
        collective_passed = full_collective_gate(audit_rows)
        candidate_passed = candidate_passed and collective_passed
        preserved = all(row[key] for row in passes if row['phase'] != 'real_update'
                        for key in ('trainable_weights_unchanged', 'optimizer_state_unchanged', 'scheduler_state_unchanged'))
        nonzero = captures['baseline']['has_nonzero_gradient']
        gradient_families = captures['baseline']['gradient_families']
        coverage = full_gradient_coverage_report(report['baseline_tensor_gradient_statistics'],
            captures['baseline']['queries'], len(groups), device=torch.cuda.current_device())
        full_gradient_coverage = coverage['passed']
        report.update(status=full_comparison_status(repeat_passed, candidate_passed, preserved and full_gradient_coverage),
            repeat_control_passed=repeat_passed,
            collective_count_gate_passed=collective_passed,
            residency_numerical_gate_passed=candidate_passed,
            candidate_accepted=repeat_passed and candidate_passed and preserved and full_gradient_coverage,
            repeat_comparisons=repeat_comparisons, candidate_comparisons=candidate_comparisons,
            state_preserved_during_fixed_passes=preserved, has_nonzero_gradient=nonzero,
            baseline_gradient_families=gradient_families, full_gradient_coverage=full_gradient_coverage,
            full_gradient_coverage_report=coverage,
            gradient_metadata={name: {'global_shape': list(parameter.shape),
                'local_shape': list(local(parameter).shape), 'dtype': str(parameter.dtype),
                'placements': [str(item) for item in parameter.placements] if isinstance(parameter, DTensor) else []}
                for name, parameter in parameters.items()})
    except Exception as error:
        report.update(status='error', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        engine.optimizer_step, engine.lr_scheduler_step = original_optimizer, original_scheduler
        engine.forward_backward_batch, engine.module.query_forward.forward_full = original_forward, original_query
        residency.full_weight_residency = original_residency
        for name, function in original_collectives.items():
            setattr(torch.distributed, name, function)
        for name, parameter in parameters.items():
            parameter.grad = initial_gradients.get(name)
        worker.clear_staged_records()
        engine.checkpoint_manager.load_rng_state(rng)
        for key, value in environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        report['rng_restored'] = training_state_digest(rng) == training_state_digest(engine.checkpoint_manager.get_rng_state())
        (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return {'rank': rank, 'status': report['status'], 'report_path': str(root / 'report.json')}


def replay_learner_config(args):
    from action_head_rl.trainer import learner_config

    config = learner_config(SimpleNamespace(slow_model=str(SLOW_MODEL), gradient_checkpointing=True,
        learning_rate=args.learning_rate, opt_steps=2 if args.train_scope == 'full' else 1))
    if args.strict_determinism:
        helpers = runpy.run_path(str(REPO / 'scripts/eval/action_head_d16_determinism_audit.py'))
        config = helpers['configure_strict_determinism'](config)
    return config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expr-name', required=True)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--world-size', type=int, default=WORLD_SIZE)
    parser.add_argument('--gpus', default=os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    parser.add_argument('--warmup-pairs', type=int, default=1)
    parser.add_argument('--measure-pairs', type=int, default=2)
    parser.add_argument('--train-scope', choices=('head_only', 'full'), default='head_only')
    parser.add_argument('--learning-rate', type=float, default=1e-6)
    parser.add_argument('--vlm-learning-rate', type=float, default=1e-6)
    parser.add_argument('--strict-determinism', action='store_true')
    parser.add_argument('--full-weight-residency-gib', type=float, default=40.0)
    parser.add_argument('--multi-condition-batch-size', type=int, default=1)
    parser.add_argument('--validate-inputs', action='store_true')
    args = parser.parse_args(argv)
    if Path(args.expr_name).name != args.expr_name or args.expr_name in ('', '.', '..'):
        parser.error('expr-name must be a single directory name')
    if args.world_size != WORLD_SIZE:
        parser.error('all ten original ranks are required; do not select partial source ranks')
    if args.warmup_pairs < 1 or args.measure_pairs < 2:
        parser.error('at least one warmup pair and two measured pairs are required')
    if not math.isfinite(args.full_weight_residency_gib) or args.full_weight_residency_gib <= 0:
        parser.error('full-weight-residency-gib must be finite and positive')
    if args.multi_condition_batch_size <= 0:
        parser.error('multi-condition-batch-size must be positive')
    if any(not math.isfinite(value) or value <= 0 for value in (args.learning_rate, args.vlm_learning_rate)):
        parser.error('learning rates must be finite and positive')
    if args.strict_determinism and args.train_scope != 'full':
        parser.error('strict-determinism requires full train scope')
    if not args.validate_inputs:
        devices = args.gpus.split(',')
        if len(devices) != WORLD_SIZE or len(set(devices)) != WORLD_SIZE or any(
                not value.isdigit() or not 0 <= int(value) < 16 for value in devices):
            parser.error('GPU execution requires exactly ten distinct local GPU IDs in 0..15')
    args.input_dir = args.input_dir.resolve()
    if not args.input_dir.is_relative_to(BASE):
        parser.error('input-dir must be under the authorized optimization root')
    args.output_dir = BASE / args.expr_name
    return args


def main(argv=None):
    args = parse_args(argv)
    sys.path[:0] = [str(REPO), str(REPO / 'verl'), str(VERL_REPO), str(REPO / 'vln_demo/src')]
    # The workspace also contains a different importable ``scripts`` package.
    # Reuse the sibling helpers by absolute path without namespace ambiguity.
    helpers = runpy.run_path(str(REPO / 'scripts/eval/action_head_d16_native_r3_probe.py'))
    runtime_environment, write_json = helpers['runtime_environment'], helpers['write_json']

    args.output_dir.mkdir(parents=True, exist_ok=False)
    runtime_parent = Path('/tmp/action_head_rl')
    runtime_parent.mkdir(parents=True, exist_ok=True)
    runtime = Path(tempfile.mkdtemp(prefix='lrp-', dir=runtime_parent))
    old_environment = dict(os.environ)
    learner = pool = ray = None
    old_mode = 'recompute' if args.train_scope == 'full' else 'single_forward'
    report = {'status': 'running', 'train_scope': args.train_scope, 'world_size': WORLD_SIZE,
              'strict_determinism': args.strict_determinism,
              'requested_learning_rates': {'learning_rate': args.learning_rate, 'vlm_learning_rate': args.vlm_learning_rate},
              'old_logprob_mode': old_mode, 'source_old_logprobs_used': False,
              'full_weight_residency_gib': args.full_weight_residency_gib if args.train_scope == 'full' else 0.0,
              'multi_condition_batch_size': args.multi_condition_batch_size,
              'slow_model': str(SLOW_MODEL), 'sft_checkpoint': str(SFT),
              'output_dir': str(args.output_dir), 'runtime_root': str(runtime),
              'optimizer_steps': 0, 'scheduler_steps': 0, 'resume_verified': False,
              'timing_scope': 'steady matched learner calls intercept optimizer/scheduler; '
                              'Full real updates are reported separately and reuse a fixed rollout payload; '
                              'forward/backward components are host timers, not CUDA kernel durations'}
    try:
        os.environ.update(runtime_environment(runtime))
        os.environ.update(CUDA_VISIBLE_DEVICES='' if args.validate_inputs else args.gpus,
                          PYTHONPATH=os.pathsep.join(sys.path[:4]),
                          ACTION_HEAD_LOCAL_RUNTIME_ROOT=str(runtime),
                          ACTION_HEAD_HEAD_CHECKPOINT=str(SFT / 'action_head'),
                          ACTION_HEAD_QUERY_CHECKPOINT=str(SFT / 'dualvln_query/dualvln_query_embeddings.pt'),
                          ACTION_HEAD_SLOW_MODEL=str(SLOW_MODEL), ACTION_HEAD_TRAIN_SCOPE=args.train_scope,
                          ACTION_HEAD_ROUTER_REPLAY='R3', ACTION_HEAD_CAPTURE_FULL_ROUTING='1',
                          ACTION_HEAD_OPTION_CONDITION_REUSE='1', ACTION_HEAD_OLD_LOGPROB_MODE=old_mode,
                          ACTION_HEAD_FROZEN_WEIGHT_RESIDENCY='0', ACTION_HEAD_LR=str(args.learning_rate),
                          ACTION_HEAD_VLM_LR=str(args.vlm_learning_rate),
                          ACTION_HEAD_FULL_WEIGHT_RESIDENCY_GIB='0')
        os.environ['ACTION_HEAD_MULTI_CONDITION_BATCH_SIZE'] = str(
            args.multi_condition_batch_size)
        for key in ('ACTION_HEAD_SINGLE_FORWARD_AUDIT_DIR', 'ACTION_HEAD_PIOLD_AUDIT_DIR'):
            os.environ.pop(key, None)
        inputs = inspect_payloads(args.input_dir)
        report['inputs'] = inputs
        write_json(args.output_dir / 'inputs.json', inputs)
        if args.validate_inputs:
            report['status'] = 'cpu_inputs_pass'
            return report
        if not inputs['nonzero_advantage_chunks']:
            raise ValueError('source has only zero advantages; it cannot validate useful gradient parity')
        import ray as ray_module
        import torch
        from action_head_rl.trainer import _get_worker_result
        from action_head_rl.worker import ActionHeadGRPOWorker
        from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

        ray = ray_module
        if torch.cuda.device_count() != WORLD_SIZE:
            raise ValueError('visible CUDA device count does not match ten-rank replay')
        cache_keys = set(runtime_environment(runtime))
        runtime_env = {'env_vars': {key: value for key, value in os.environ.items()
                                   if key in cache_keys or key.startswith('ACTION_HEAD_')
                                   or key in ('PYTHONPATH', 'LD_LIBRARY_PATH')}}
        ray.init(address='local', runtime_env=runtime_env, include_dashboard=False,
                 _temp_dir=str(runtime / 'ray'), num_gpus=WORLD_SIZE)
        config = replay_learner_config(args)
        pool = RayResourcePool(process_on_nodes=[WORLD_SIZE], name_prefix=f'learner-replay-{args.expr_name}')
        learner = RayWorkerGroup(resource_pool=pool, ray_cls_with_init=RayClassWithInitArgs(
            ray.remote(ActionHeadGRPOWorker), config=config), device_name='cuda')
        learner.set_loss_fn(loss_fn=lambda **_: None)
        _get_worker_result(learner.reset())
        specification = {'inputs': inputs, 'output_dir': str(args.output_dir),
                         'warmup_pairs': args.warmup_pairs, 'measure_pairs': args.measure_pairs,
                         'train_scope': args.train_scope,
                         'strict_determinism': args.strict_determinism,
                         'learning_rate': args.learning_rate, 'vlm_learning_rate': args.vlm_learning_rate,
                         'full_weight_residency_gib': args.full_weight_residency_gib}
        specification['multi_condition_batch_size'] = args.multi_condition_batch_size
        completed = ray.get([actor.execute_with_func_generator.remote(
            func=run_worker_probe, specification=specification) for actor in learner.workers])
        reports = [json.loads(Path(row['report_path']).read_text()) for row in completed]
        report['actual_learning_rates_by_rank'] = [
            {'rank': row['rank'], **row['learning_rates']} for row in reports]
        measured = {variant: [] for variant in ('baseline', 'candidate')}
        for variant in measured:
            for iteration in range(args.measure_pairs):
                measured[variant].append(max(row['train_call_wall_seconds'] for rank_report in reports
                    for row in rank_report['passes'] if row['phase'] == 'measure'
                    and row['variant'] == variant and row['iteration'] == iteration))
        baseline = statistics.median(measured['baseline'])
        candidate = statistics.median(measured['candidate'])
        nonzero = any(row['has_nonzero_gradient'] for row in reports)
        passed = all(row['status'] == 'pass' and row['rng_restored'] for row in reports) and nonzero
        status = 'pass' if passed else 'fail'
        if args.train_scope == 'full':
            status = full_comparison_status(
                all(row['repeat_control_passed'] for row in reports),
                all(row['residency_numerical_gate_passed'] for row in reports),
                nonzero and all(row['state_preserved_during_fixed_passes'] and row['rng_restored']
                                and row['full_gradient_coverage'] for row in reports))
        report.update(status=status, ranks=completed,
                      has_nonzero_gradient=nonzero, measured_rank_max_seconds=measured,
                      baseline_median_seconds=baseline, candidate_median_seconds=candidate,
                      learner_speedup=baseline / candidate,
                      rollout_seconds=None, end_to_end_rl_step_seconds=None)
        if args.train_scope == 'full':
            update_totals = []
            for step in (1, 2):
                local_rows = {f'{parameter_family(name)}.rank_{rank_report["rank"]}.{name}': row
                    for rank_report in reports for update in rank_report['real_updates'] if update['step'] == step
                    for name, row in update['parameters'].items()}
                update_totals.append({'step': step, 'families': aggregate_update_statistics(local_rows),
                    'aggregation': 'sum of actual local FSDP shards; placements and shapes are in each rank report'})
            report.update(optimizer_steps=2, scheduler_steps=2,
                candidate_accepted=passed, real_update_totals=update_totals,
                repeat_control_passed=all(row['repeat_control_passed'] for row in reports),
                fixed_state_preserved=all(row['state_preserved_during_fixed_passes'] for row in reports),
                real_update_rank_max_seconds=[max(row['train_call_wall_seconds'] for rank_report in reports
                    for row in rank_report['passes'] if row['phase'] == 'real_update' and row['iteration'] == step)
                    for step in range(2)])
        if status == 'inconclusive':
            return report
        if not passed:
            raise RuntimeError('matched learner parity failed or every source advantage/gradient was zero')
        return report
    except (Exception, KeyboardInterrupt) as error:
        report.update(status=('interrupted' if isinstance(error, KeyboardInterrupt) else
                             'error' if report['status'] == 'running' else report['status']),
                      error=f'{type(error).__name__}: {error}')
        raise
    finally:
        cleanup_errors = []
        if ray is not None:
            if learner is not None:
                for actor in learner.workers:
                    try:
                        ray.kill(actor, no_restart=True)
                    except Exception as error:
                        cleanup_errors.append(f'actor cleanup: {error}')
            if pool is not None and pool.pgs:
                for placement in pool.pgs:
                    try:
                        ray.util.remove_placement_group(placement)
                    except Exception as error:
                        cleanup_errors.append(f'placement cleanup: {error}')
            try:
                ray.shutdown()
            except Exception as error:
                cleanup_errors.append(f'Ray shutdown: {error}')
        # Preserve Ray/worker logs on the data disk before removing local cache.
        ray_logs = runtime / 'ray/session_latest/logs'
        try:
            if ray_logs.is_dir():
                shutil.copytree(ray_logs, args.output_dir / 'ray_logs', dirs_exist_ok=True)
        except Exception as error:
            cleanup_errors.append(f'Ray log preservation: {error}')
        try:
            shutil.rmtree(runtime)
        except Exception as error:
            cleanup_errors.append(f'local runtime cleanup: {error}')
        report['runtime_cleaned'] = not runtime.exists()
        if cleanup_errors:
            report.update(cleanup_errors=cleanup_errors, status='cleanup_error')
        os.environ.clear()
        os.environ.update(old_environment)
        write_json(args.output_dir / 'report.json', report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if cleanup_errors:
            raise RuntimeError('learner replay cleanup failed; see report.json')


if __name__ == '__main__':
    main()
