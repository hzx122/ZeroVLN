"""One real-model, zero-update parity gate before optimized local training."""
from contextlib import contextmanager, nullcontext
import json
import math
import os
from pathlib import Path
import time
import torch
from tensordict import TensorDict
from sft.verl_action_head_engine import unpack_anchor_batch


@contextmanager
def joint_mode(enabled):
    # The independent FM reference has no selective joint backward graph.
    # These optimizations depend on that graph; disable them only while the
    # reference runs, then restore the candidate configuration exactly.
    keys = ('ACTION_HEAD_JOINT_TEXT_FM_FORWARD', 'ACTION_HEAD_FM_SUFFIX_VJP',
            'ACTION_HEAD_FM_CORE_SUFFIX_VJP', 'ACTION_HEAD_DUAL_SHARED_RECOMPUTE')
    saved = {key: os.environ.get(key) for key in keys}
    os.environ[keys[0]] = str(int(enabled))
    if not enabled:
        for key in keys[1:]:
            os.environ[key] = '0'
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def separate_payloads(data, query_count):
    payload = unpack_anchor_batch(data)
    fm = {k: v for k, v in payload.items() if not k.startswith('joint_')}
    keys = ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw',
            'mm_token_type_ids', 'logical_batch_valid')
    text = {k: payload[k] for k in keys if k in payload}
    batch = payload['input_ids'].shape[0]
    for key in ('all_routed_experts', 'all_positions'):
        value = payload[key]
        text[key] = value.reshape(batch, payload['input_ids'].shape[1] + query_count, *value.shape[1:])[:, :-query_count].flatten(0, 1)
    for source, target in [('joint_text_response_mask', 'text_response_mask'),
                           ('joint_text_temperature', 'text_temperature'),
                           ('joint_text_loss_coef', 'text_loss_coef'),
                           ('joint_text_advantage', 'advantage'),
                           ('joint_text_sample_valid', 'sample_valid'),
                           ('joint_text_loss_weight', 'loss_weight')]:
        text[target] = payload[source]
    pack = lambda values: TensorDict({k: v.unsqueeze(0) for k, v in values.items()}, batch_size=[1])
    return pack(fm), pack(text)


def parameter_roles(module):
    """Use module ownership, not display names (qwen is a property alias)."""
    groups = {
        'qwen': {id(p) for p in module.qwen.parameters()},
        'query': {id(p) for p in module.query_encoder.parameters()},
        'head': {id(p) for p in module.action_head.parameters()},
    }
    if any(not group for group in groups.values()):
        raise ValueError('gradient parity requires three nonempty parameter groups')
    owners = {}
    for role, group in groups.items():
        for identity in group:
            if identity in owners:
                raise ValueError('gradient parity parameter groups overlap')
            owners[identity] = role
    if set(owners) != {id(p) for p in module.parameters()}:
        raise ValueError('gradient parity parameter groups do not cover the model')
    return owners


def role_passed(role, relative_l2_limit=.01):
    return (math.isfinite(role['reference_norm']) and role['reference_norm'] > 0
            and role['relative_l2'] <= relative_l2_limit and role['cosine'] >= .999
            and role['global_none_mismatches'] == 0)


def repeatability_limit(control):
    """Bound shape-matched BF16 error against measured repeat noise.

    A noisy/invalid reference cannot grant an unbounded tolerance. The absolute
    cap is 5%, cosine stays >= .999, and None/zero-reference checks never relax.
    """
    if not role_passed(control, .05):
        return None
    return min(.05, max(.01, 2. * control['relative_l2']))


def _compare_gradients(params, owners, reference):
    roles = {k: dict(sq_ref=0., sq_new=0., dot=0., sq_error=0., max_abs=0.,
                    none_mismatches=0, parameters=0) for k in ('qwen', 'query', 'head')}
    for name, p in params:
        role = roles[owners[id(p)]]
        role['parameters'] += 1
        old, g = reference[name], p.grad
        if old is None or g is None:
            role['none_mismatches'] += int((old is None) != (g is None))
            continue
        old = old.to(device=g.device, dtype=torch.float32)
        new = (g.to_local() if hasattr(g, 'to_local') else g).detach().float()
        delta = new - old
        role['sq_ref'] += float(old.square().sum())
        role['sq_new'] += float(new.square().sum())
        role['dot'] += float((old * new).sum())
        role['sq_error'] += float(delta.square().sum())
        if delta.numel():
            role['max_abs'] = max(role['max_abs'], float(delta.abs().max()))
    for role in roles.values():
        values = torch.tensor([role[k] for k in
            ('sq_ref', 'sq_new', 'dot', 'sq_error', 'none_mismatches')],
            device=params[0][1].device, dtype=torch.float64)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(values)
        ref, new, dot, err, missing = values.tolist()
        role.update(relative_l2=math.sqrt(err / max(ref, 1e-30)),
                    cosine=dot / max(math.sqrt(ref * new), 1e-30) if ref and new else float(ref == new),
                    global_none_mismatches=int(missing), reference_norm=math.sqrt(ref))
    return roles


@contextmanager
def checkpoint_mode(qwen, enabled):
    """Audit-only HF switch; restore flags and checkpoint callable exactly."""
    saved = [(module, module.gradient_checkpointing,
              hasattr(module, '_gradient_checkpointing_func'),
              getattr(module, '_gradient_checkpointing_func', None))
             for module in qwen.modules() if hasattr(module, 'gradient_checkpointing')]
    if not saved:
        raise ValueError('native stacked audit requires HF checkpoint-capable modules')
    try:
        if enabled:
            qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        else:
            qwen.gradient_checkpointing_disable()
        if bool(qwen.is_gradient_checkpointing) != enabled:
            raise RuntimeError('HF checkpoint mode switch did not take effect')
        yield
    finally:
        for module, flag, present, function in saved:
            module.gradient_checkpointing = flag
            if present:
                module._gradient_checkpointing_func = function
            elif hasattr(module, '_gradient_checkpointing_func'):
                del module._gradient_checkpointing_func


@contextmanager
def partial_checkpoint_mode(backbone, last_layers_off=0, *, selection='last'):
    """Keep the prefix checkpointed and retain selected decoder activations."""
    if last_layers_off == 0:
        yield []
        return
    layers=list(backbone.layers)
    if type(last_layers_off) is not int or not 0 < last_layers_off < len(layers):
        raise ValueError('partial checkpoint selection must leave checkpointed decoder layers')
    if selection=='last':eligible=list(range(len(layers)))
    elif selection=='full_attention':
        eligible=[i for i,layer in enumerate(layers) if layer.layer_type=='full_attention']
    else:raise ValueError('unknown partial checkpoint layer selection')
    if last_layers_off>len(eligible):
        raise ValueError('partial checkpoint selection exceeds eligible layer count')
    selected=eligible[-last_layers_off:]
    saved=[]
    for index in selected:
        layer=layers[index]
        if not hasattr(layer,'gradient_checkpointing') or not layer.gradient_checkpointing:
            raise ValueError('partial checkpoint mode requires enabled native decoder checkpoints')
        saved.append((layer,layer.gradient_checkpointing))
    try:
        for layer,_ in saved:layer.gradient_checkpointing=False
        yield selected
    finally:
        for layer,flag in saved:layer.gradient_checkpointing=flag


def audit_stacked_rule_payloads(engine, baseline, stacked, output_dir, *,
                               candidate_checkpoint_layers_off=0,
                               candidate_checkpoint_selection='last',
                               verify_native_adam=False, warm_native_adam=False,
                               candidate_optimizer_cpu_during_backward=False,
                               candidate_full_weight_residency_gib=8.0,
                               baseline_checkpoint_layers_off=0,
                               baseline_full_weight_residency_gib=0.0,
                               candidate_extra_context=None):
    """Compare B1/GC-on/0 with the configured candidate and residency budget.

    Inputs must already pass the matched-record CPU gate. No throughput
    measurements or per-knob variants are performed. Adam and full-step memory
    remain separate final integration requirements by default. The opt-in Adam
    replay takes one native update per variant, checks the existing packing
    audit's parameter-delta tolerances, and restores the source state afterward.
    """
    from action_head_rl.joint_audit import forbid_updates, temporary_environment
    from action_head_rl.packing_audit_worker import LocalTrainingSnapshot, local
    from action_head_rl.residency import full_weight_residency
    if not math.isfinite(candidate_full_weight_residency_gib) or candidate_full_weight_residency_gib < 0:
        raise ValueError('candidate residency budget must be finite and nonnegative')
    if not math.isfinite(baseline_full_weight_residency_gib) or baseline_full_weight_residency_gib < 0:
        raise ValueError('baseline residency budget must be finite and nonnegative')
    if warm_native_adam and not verify_native_adam:
        raise ValueError('warm Adam audit requires native Adam verification')
    if candidate_optimizer_cpu_during_backward and not (verify_native_adam and warm_native_adam):
        raise ValueError('optimizer CPU audit requires nonempty warmed native Adam state')
    if any(os.environ.get(key) != '1' for key in (
            'ACTION_HEAD_RULE_CREDIT', 'ACTION_HEAD_TEXT_POLICY',
            'ACTION_HEAD_JOINT_TEXT_FM_FORWARD')):
        raise ValueError('stacked parity requires the shared rule-credit text/FM objective')
    if not baseline or not stacked:
        raise ValueError('stacked audit requires two nonempty payload streams')
    params = list(engine.module.named_parameters())
    if any(p.grad is not None for _, p in params):
        raise RuntimeError('stacked audit must precede gradient accumulation')
    owners = parameter_roles(engine.module)
    candidate_checkpointing = bool(engine.module.qwen.is_gradient_checkpointing)
    if candidate_checkpoint_layers_off and not candidate_checkpointing:
        raise ValueError('partial checkpoint candidate requires global checkpointing enabled')
    snapshot = LocalTrainingSnapshot(engine)
    entry_snapshot = snapshot
    report = dict(protocol='v3_b1_vs_complete_stacked_zero_update_v1', optimizer_updates=0,
                  rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                  speedup_measured=False, candidate_gradient_checkpointing=candidate_checkpointing,
                  candidate_full_weight_residency_gib=candidate_full_weight_residency_gib,
                  full_step_memory_verified=False, adam_parity_verified=False)
    report['candidate_checkpoint_layers_off']=candidate_checkpoint_layers_off
    report['candidate_checkpoint_selection']=candidate_checkpoint_selection
    report['baseline_checkpoint_layers_off'] = baseline_checkpoint_layers_off
    report['baseline_full_weight_residency_gib'] = baseline_full_weight_residency_gib
    report['candidate_extra_context'] = candidate_extra_context is not None
    report['net_optimizer_updates'] = 0
    report['warm_native_adam'] = warm_native_adam
    report['candidate_optimizer_cpu_during_backward'] = candidate_optimizer_cpu_during_backward
    from action_head_rl.runtime_progress import LearnerProgress
    progress = LearnerProgress(Path(output_dir) / 'progress', report['rank'])
    if verify_native_adam:
        from action_head_rl.packing_audit import THRESHOLDS
        from action_head_rl.packing_audit_worker import (
            compare, one_update_counts, optimizer_counts, summarize_checks)
        report['protocol'] = 'v3_b1_vs_stacked_native_adam_restored_v1'
        report['adam_thresholds'] = dict(THRESHOLDS)
        report['adam_variants'] = {}
        source_counts = optimizer_counts(engine)
        source_epoch = engine.lr_scheduler.last_epoch
        entry_counts, entry_epoch = source_counts.copy(), source_epoch

    def native_update(label, reference_delta=None, *, capture_delta=True):
        presence = {name: p.grad is not None for name, p in params}
        before = optimizer_counts(engine)
        if before != source_counts or engine.lr_scheduler.last_epoch != source_epoch:
            raise RuntimeError('native Adam replay did not restore source counters')
        norm = float(engine.optimizer_step())
        engine.lr_scheduler_step()
        report['optimizer_updates'] += int(math.isfinite(norm))
        deltas, checks = {}, {}
        for name, p in (params if capture_delta else []):
            delta = local(p).detach().cpu() - snapshot.parameters[name]
            if reference_delta is None:
                deltas[name] = delta.clone()
            else:
                checks[name] = compare(reference_delta[name], delta,
                    atol=THRESHOLDS['delta_atol'], rtol=THRESHOLDS['delta_rtol'])
        row = dict(grad_norm=norm if math.isfinite(norm) else None,
                   optimizer_counters_correct=one_update_counts(
                       source_counts, optimizer_counts(engine), presence),
                   scheduler_correct=engine.lr_scheduler.last_epoch == source_epoch + 1,
                   parameter_delta_checks=checks,
                   parameter_delta_summary=summarize_checks(checks) if checks else {})
        row['passed'] = (math.isfinite(norm) and row['optimizer_counters_correct']
                         and row['scheduler_correct']
                         and all(value['passed'] for value in checks.values()))
        report['adam_variants'][label] = row
        return deltas

    @contextmanager
    def run(payloads, *, optimized, label):
        progress.completed(label + '_started', force=True,
                           completed=0, total=len(payloads))
        snapshot.restore(engine)
        engine.module.set_requires_gradient_sync(True, recurse=True)
        if hasattr(engine.module, 'set_is_last_backward'):
            engine.module.set_is_last_backward(True)
        losses = []
        optimizer_context = nullcontext()
        if optimized and candidate_optimizer_cpu_during_backward:
            from action_head_rl.residency import optimizer_cpu_during_backward
            optimizer_context = optimizer_cpu_during_backward(engine, enabled=True)
        with optimizer_context as optimizer_report:
            with temporary_environment(ACTION_HEAD_FULL_WEIGHT_RESIDENCY_GIB=(
                    f'{candidate_full_weight_residency_gib if optimized else baseline_full_weight_residency_gib:g}')), \
                checkpoint_mode(engine.module.qwen, candidate_checkpointing if optimized else True), \
                (partial_checkpoint_mode(engine.module.query_forward._backbone(),
                    candidate_checkpoint_layers_off if optimized else baseline_checkpoint_layers_off,
                    selection=candidate_checkpoint_selection)
                 if (candidate_checkpoint_layers_off if optimized else baseline_checkpoint_layers_off)
                 else nullcontext()), \
                (candidate_extra_context() if optimized and candidate_extra_context is not None else nullcontext()), \
                full_weight_residency(engine):
                for index, payload in enumerate(payloads):
                    data = payload.to(next(engine.module.parameters()).device)
                    losses.append(float(engine.forward_backward_batch(data, None)['loss']))
                    del data
                    progress.completed(label + '_backward_completed',
                        force=index + 1 == len(payloads),
                        completed=index + 1, total=len(payloads))
            # Weight shards must be restored before native Adam loads moments.
            # Keep optimizer residency active through the caller's native update.
            if optimizer_report is not None:
                report['candidate_optimizer_residency'] = optimizer_report
            yield sum(losses), sum(abs(value) for value in losses)

    try:
        with (nullcontext([]) if verify_native_adam else forbid_updates(engine)) as attempts:
            if warm_native_adam:
                with run(baseline, optimized=False, label='warm_source'):
                    native_update('warm_source', capture_delta=False)
                engine.optimizer_zero_grad()
                if not report['adam_variants']['warm_source']['passed']:
                    raise RuntimeError('common native Adam warm update failed')
                snapshot = LocalTrainingSnapshot(engine)
                source_counts = optimizer_counts(engine)
                source_epoch = engine.lr_scheduler.last_epoch
            with run(baseline, optimized=False, label='baseline') as (reference_loss, loss_scale):
                reference = {name: None if p.grad is None else local(p.grad).detach().cpu().clone()
                             for name, p in params}
                reference_delta = native_update('baseline') if verify_native_adam else None
            with run(baseline, optimized=False, label='repeat_baseline'):
                repeat_roles = _compare_gradients(params, owners, reference)
                if verify_native_adam:
                    native_update('repeat_baseline', reference_delta)
            with run(stacked, optimized=True, label='candidate') as (candidate_loss, _):
                roles = _compare_gradients(params, owners, reference)
                if verify_native_adam:
                    native_update('candidate', reference_delta)
            if verify_native_adam:
                report['adam_parity_verified'] = all(
                    row['passed'] for row in report['adam_variants'].values())
                if candidate_optimizer_cpu_during_backward:
                    residency = report['candidate_optimizer_residency']
                    report['warm_state_transfers_verified'] = (
                        residency['entry_state_local_bytes'] > 0
                        and residency['entry_state_tensor_count'] > 0
                        and residency['optimizer_to_gpu_calls'] == 1
                        and residency['optimizer_step_calls'] == 1 and residency['closed'])
                    report['adam_parity_verified'] &= report['warm_state_transfers_verified']
            limits = {key: repeatability_limit(value) for key, value in repeat_roles.items()}
            loss_passed = (math.isfinite(reference_loss) and math.isfinite(candidate_loss)
                           and abs(reference_loss - candidate_loss) <= 1e-6 + .01 * loss_scale)
            report.update(reference_loss=reference_loss, candidate_loss=candidate_loss,
                          loss_passed=loss_passed, roles=roles, repeat_roles=repeat_roles,
                          relative_l2_limits=limits, forbidden_call_attempts=list(attempts))
            report['passed'] = loss_passed and not attempts and all(
                limits[key] is not None and role_passed(value, limits[key]) for key, value in roles.items())
            if verify_native_adam:
                report['passed'] = report['passed'] and report['adam_parity_verified']
                del reference_delta
    finally:
        progress.completed('restore_entry_source_started', force=True)
        entry_snapshot.restore(engine)
        # The engine normally synchronizes every backward in this recipe.
        engine.module.set_requires_gradient_sync(True, recurse=True)
        if hasattr(engine.module, 'set_is_last_backward'):
            engine.module.set_is_last_backward(True)
        if verify_native_adam:
            from action_head_rl.checkpoint import training_state_digest
            report['source_restored'] = (
                optimizer_counts(engine) == entry_counts
                and engine.lr_scheduler.last_epoch == entry_epoch
                and training_state_digest(engine.checkpoint_manager.get_rng_state()) == entry_snapshot.rng_digest
                and all(torch.equal(local(p).detach().cpu(), entry_snapshot.parameters[name])
                        for name, p in params))
            if not report['source_restored']:
                raise RuntimeError('native Adam audit changed protected source state')
        progress.completed('restore_entry_source_completed', force=True)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    from action_head_rl.joint_audit import write_new_json
    write_new_json(out / f'stacked_rank_{report["rank"]}.json', report)
    if not report['passed']:
        raise RuntimeError(f'stacked rule gradient/loss parity failed: {report}')
    return report


def audit_native_optimizer_roundtrip(engine, payloads, output_dir, *, progress=None):
    """Isolate native Adam state movement with identical real local gradients.

    Independent-backward parameter deltas remain a separate gate. This checks
    native clipping, Adam, scheduler and moments without conflating backward
    nondeterminism with CPU/GPU optimizer state movement; thresholds are unchanged.
    """
    from action_head_rl.checkpoint import training_state_digest
    from action_head_rl.joint_audit import temporary_environment, write_new_json
    from action_head_rl.packing_audit import THRESHOLDS
    from action_head_rl.packing_audit_worker import (
        LocalTrainingSnapshot, local, optimizer_counts, one_update_counts, compare)
    from action_head_rl.residency import optimizer_cpu_during_backward
    if not payloads or any(p.dtype != torch.float32 for p in engine.module.parameters()):
        raise ValueError('optimizer roundtrip requires real payloads and Full FP32 parameters')
    params = list(engine.module.named_parameters())
    if any(p.grad is not None for _, p in params):
        raise ValueError('optimizer roundtrip must precede gradient accumulation')
    entry = LocalTrainingSnapshot(engine)
    entry_counts = optimizer_counts(engine)
    report = dict(protocol='native_adam_identical_gradient_cpu_roundtrip_v1',
                  rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                  optimizer_updates=0, net_optimizer_updates=0, passed=False,
                  independent_backward_delta_verified=False, speedup_measured=False,
                  adam_thresholds=dict(THRESHOLDS), variants={})
    def phase(name):
        if progress is not None:
            progress(name)
    def update(label, source, expected_delta=None, expected_moments=None):
        before = optimizer_counts(engine)
        presence = {name: p.grad is not None for name, p in params}
        epoch = engine.lr_scheduler.last_epoch
        norm = float(engine.optimizer_step())
        engine.lr_scheduler_step()
        report['optimizer_updates'] += 1
        deltas, checks, moments = {}, {}, {}
        for name, p in params:
            delta = local(p).detach().cpu() - source.parameters[name]
            if expected_delta is None:
                deltas[name] = delta.clone()
            else:
                checks[name] = compare(expected_delta[name], delta,
                    atol=THRESHOLDS['delta_atol'], rtol=THRESHOLDS['delta_rtol'])
            moments[name] = training_state_digest(engine.optimizer.state.get(p, {}))
        row = dict(grad_norm=norm if math.isfinite(norm) else None,
                   optimizer_counters_correct=one_update_counts(before, optimizer_counts(engine), presence),
                   scheduler_correct=engine.lr_scheduler.last_epoch == epoch + 1,
                   parameter_delta_checks=checks,
                   moments_equal=expected_moments is None or moments == expected_moments)
        row['passed'] = (math.isfinite(norm) and row['optimizer_counters_correct']
                         and row['scheduler_correct'] and row['moments_equal']
                         and all(value['passed'] for value in checks.values()))
        report['variants'][label] = row
        return deltas, moments
    try:
        phase('capture_real_gradients')
        engine.module.set_requires_gradient_sync(True, recurse=True)
        if hasattr(engine.module, 'set_is_last_backward'):
            engine.module.set_is_last_backward(True)
        with temporary_environment(ACTION_HEAD_FULL_WEIGHT_RESIDENCY_GIB='0'), \
                checkpoint_mode(engine.module.qwen, True):
            for payload in payloads:
                data = payload.to(next(engine.module.parameters()).device)
                result = engine.forward_backward_batch(data, None)
                if not math.isfinite(float(result['loss'])):
                    raise RuntimeError('optimizer roundtrip captured a nonfinite real loss')
                del data
        gradients = {name: None if p.grad is None else local(p.grad).detach().cpu().clone()
                     for name, p in params}
        if not any(value is not None and value.numel() for value in gradients.values()):
            raise ValueError('optimizer roundtrip captured no real gradients')
        phase('common_native_warm_update')
        update('warm_source', entry)
        if not report['variants']['warm_source']['passed']:
            raise RuntimeError('optimizer roundtrip common native warm update failed')
        engine.optimizer_zero_grad()
        warm = LocalTrainingSnapshot(engine)
        def restore_gradients():
            warm.restore(engine)
            for name, p in params:
                saved = gradients[name]
                if saved is not None:
                    p.grad = torch.empty_like(p)
                    local(p.grad).copy_(saved)
                    if not torch.equal(local(p.grad).detach().cpu(), saved):
                        raise RuntimeError('fixed-gradient replay changed gradient values')
        phase('identical_gradient_native_baseline')
        restore_gradients()
        reference, moments = update('baseline', warm)
        phase('identical_gradient_native_repeat')
        restore_gradients()
        update('repeat_baseline', warm, reference, moments)
        phase('identical_gradient_cpu_roundtrip')
        restore_gradients()
        with optimizer_cpu_during_backward(engine, enabled=True) as residency:
            update('cpu_roundtrip', warm, reference, moments)
        report['optimizer_residency'] = residency
        report['same_gradients_verified'] = True
        report['warm_state_transfers_verified'] = (
            residency['entry_state_local_bytes'] > 0 and residency['entry_state_tensor_count'] > 0
            and residency['optimizer_step_calls'] == 1
            and residency['optimizer_to_gpu_calls'] == 1 and residency['closed'])
        report['passed'] = (report['warm_state_transfers_verified']
                            and all(row['passed'] for row in report['variants'].values()))
    finally:
        phase('restore_entry_source')
        entry.restore(engine)
        report['source_restored'] = (
            optimizer_counts(engine) == entry_counts
            and engine.lr_scheduler.last_epoch == entry.scheduler['last_epoch']
            and training_state_digest(engine.checkpoint_manager.get_rng_state()) == entry.rng_digest
            and all(torch.equal(local(p).detach().cpu(), entry.parameters[name]) for name, p in params))
        if not report['source_restored']:
            raise RuntimeError('optimizer roundtrip changed protected source state')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_new_json(out / f'optimizer_roundtrip_rank_{report["rank"]}.json', report)
    if not report['passed']:
        raise RuntimeError('identical-gradient native optimizer roundtrip failed')
    return report


def dispatch_stacked_rule_replay(learner, groups, *, world, token_balanced,
                                 output_dir, resolve_waves, resolve_result,
                                 candidate_batch_size, candidate_max_padded_tokens):
    """Replay the first real streamed wave before any production backward.

    Payloads stay in the object store. Both CPU preparations finish on all
    ranks before the single combined GPU gate is released. Candidate packing
    follows the actual streamed training recipe. The caller must
    already have drained preceding work; this helper never stages an update.
    """
    from action_head_rl.joint_scheduling import shard_joint_policy_groups
    common = dict(world=world, logical_valid=1.,
                  balance_by_token_flops=token_balanced)
    baseline, _ = shard_joint_policy_groups(groups, condition_batch_size=1,
                                            max_padded_tokens=0, **common)
    stacked, _ = shard_joint_policy_groups(groups, condition_batch_size=candidate_batch_size,
                                           max_padded_tokens=candidate_max_padded_tokens, **common)
    baseline, stacked = resolve_waves(baseline), resolve_waves(stacked)
    reports = resolve_result(learner.prepare_stacked_rule_replay(
        baseline, stacked, [1.] * world))
    validate_stacked_rule_reports(reports, world)
    result = resolve_result(learner.run_stacked_rule_replay(output_dir, reports))
    if (len(result) != world or {row['rank'] for row in result} != set(range(world))
            or any(not row['passed'] or row['optimizer_updates'] != 0 for row in result)):
        raise RuntimeError('stacked replay did not pass on every rank without updates')
    return result


def prepare_stacked_rule_payloads(baseline_records, stacked_records, *, processor,
                                 positions, logical_valid):
    """CPU-only matched replay gate; ignore only the two packing annotations.

    The driver must shard the same descriptors twice, using B1 and the actual
    candidate recipe, and complete both all-rank schedule/signature barriers before
    calling the numerical gate. Inputs here are resolved rank-local records.
    """
    from action_head_rl.checkpoint import training_state_digest
    from action_head_rl.joint_policy import consecutive_joint_groups, consecutive_joint_batches
    from action_head_rl.worker import joint_condition_batch_to_tensordict
    if not math.isfinite(logical_valid) or logical_valid <= 0:
        raise ValueError('matched replay requires positive original episode mass')
    ignored = {'_condition_pack_batch', '_condition_pack_token_width'}
    def semantic_digest(records):
        if not records or any(any(key.startswith('_rollout_payload') or
                                 key == '_reuse_staged_option_payload' for key in row)
                              for row in records):
            raise ValueError('matched replay requires nonempty resolved records')
        return training_state_digest([{k: v for k, v in row.items() if k not in ignored}
                                      for row in records])
    digest = semantic_digest(baseline_records)
    if digest != semantic_digest(stacked_records):
        raise ValueError('baseline and stacked replay changed records, targets, weights or rank assignment')
    def prepare(records):
        groups = consecutive_joint_groups(records)
        batches = consecutive_joint_batches(groups)
        payloads = [joint_condition_batch_to_tensordict(
            [[dict(row, logical_batch_valid=logical_valid) for row in group] for group in batch],
            processor, positions) for batch in batches]
        return payloads, batches
    baseline, baseline_batches = prepare(baseline_records)
    stacked, stacked_batches = prepare(stacked_records)
    if any(len(batch) != 1 for batch in baseline_batches):
        raise ValueError('baseline replay must use B1')
    if any(len(batch) > 4 for batch in stacked_batches):
        raise ValueError('stacked replay exceeds the frozen B4 recipe')
    return baseline, stacked, dict(record_sha256=digest, logical_valid=logical_valid,
        baseline_forwards=len(baseline), stacked_forwards=len(stacked))


def validate_stacked_rule_reports(reports, world):
    """Driver/worker barrier: each profile has one collective sequence on all ranks."""
    if len(reports) != world or {row['rank'] for row in reports} != set(range(world)):
        raise ValueError('stacked replay requires complete unique rank preflight reports')
    reference = reports[0]
    if any(row['logical_valid'] != reference['logical_valid'] or
           row['collective_signatures'] != reference['collective_signatures']
           for row in reports):
        raise ValueError('stacked replay has divergent rank collective shapes or episode mass')
    for name in ('baseline', 'stacked'):
        if not reference['collective_signatures'].get(name):
            raise ValueError('stacked replay has an empty collective sequence')


def audit_first_shared_batch(engine, data, output_dir):
    """Zero-update independent-forward vs shared-graph gate at equal shape.

    This does NOT claim bitwise parity with the old shorter BF16 text sequence.
    The independent native-FP32 short/long causal check is a separate prerequisite
    of preparing an optimized launch. Equal shape here isolates sharing/backward
    ownership from the fast kernels' sequence-shape rounding changes.
    """
    started = time.perf_counter()
    params = list(engine.module.named_parameters())
    owners = parameter_roles(engine.module)
    if any(p.grad is not None for _, p in params):
        raise RuntimeError('rule parity gate must precede the first gradient accumulation')
    rank = torch.distributed.get_rank()
    rng = engine.checkpoint_manager.get_rng_state()
    fm, _ = separate_payloads(data, engine.module.query_encoder.QUERY_COUNT)
    text = data.clone()
    text['advantage'] = torch.zeros_like(text['advantage'])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(data.cpu(), out / f'payload_rank_{rank}.pt')

    def independent():
        with joint_mode(False):
            action = engine.forward_backward_batch(fm, None)
        leaked = torch.tensor(int(any(p.grad is not None for p in engine.module.qwen.parameters())), device='cuda')
        torch.distributed.all_reduce(leaked)
        if leaked.item():
            raise RuntimeError('reference rule action boundary leaked Qwen gradients')
        with joint_mode(True):
            decision = engine.forward_backward_batch(text, None, rule_audit_text_only=True)
        return float(action['loss']) + float(decision['loss'])

    try:
        separate_loss = independent()
        reference = {name: None if p.grad is None else p.grad.to_local().detach().cpu().clone()
                     for name, p in params}
        engine.optimizer_zero_grad()
        engine.checkpoint_manager.load_rng_state(rng)
        independent()
        repeat_roles = _compare_gradients(params, owners, reference)
        engine.optimizer_zero_grad()
        engine.checkpoint_manager.load_rng_state(rng)
        with joint_mode(True):
            candidate = engine.forward_backward_batch(data, None)
        roles = _compare_gradients(params, owners, reference)
        limits = {key: repeatability_limit(value) for key, value in repeat_roles.items()}
        passed = all(limits[key] is not None and role_passed(value, limits[key])
                     for key, value in roles.items())
        report = dict(rank=rank, passed=passed, optimizer_updates=0, roles=roles,
            protocol='independent_equal_shape_forward_with_repeat_control_v2',
            repeat_roles=repeat_roles, separate_loss=separate_loss,
            shared_loss=float(candidate['loss']), seconds=time.perf_counter() - started,
            thresholds=dict(relative_l2_by_role=limits, relative_l2_floor=.01,
                            repeat_multiplier=2., relative_l2_cap=.05,
                            cosine=.999, none_mismatches=0),
            limitation='BF16 old shorter-text numerical identity is not asserted')
        (out / f'rank_{rank}.json').write_text(json.dumps(report, indent=2))
        if not passed:
            raise RuntimeError(f'rule shared gradient parity failed on rank {rank}: {roles}')
        return report
    finally:
        engine.optimizer_zero_grad()
        engine.checkpoint_manager.load_rng_state(rng)
