"""Actual FSDP replay worker. Never publishes or saves the audited update."""
from __future__ import annotations

import copy
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor

from action_head_rl.checkpoint import training_state_digest
from action_head_rl.joint_audit import tensor_difference, write_new_json
from action_head_rl.packing_audit import BATCH_SIZES, pack_stream, packed_signature, collective_signature
from action_head_rl.worker import ActionHeadGRPOWorker
from verl.single_controller.base.decorator import Dispatch, register


def local(value):
    return value.to_local() if isinstance(value, DTensor) else value


def clone_cpu(value):
    if torch.is_tensor(value):
        return local(value).detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_cpu(item) for item in value)
    return copy.deepcopy(value)


def restore_tree(target, saved):
    if torch.is_tensor(saved):
        if not torch.is_tensor(target) or local(target).shape != saved.shape or local(target).dtype != saved.dtype:
            raise ValueError("cached optimizer tensor changed layout")
        local(target).copy_(saved)
        return target
    if isinstance(saved, dict):
        if not isinstance(target, dict) or set(target) != set(saved):
            raise ValueError("cached optimizer state changed fields")
        for key in saved:
            target[key] = restore_tree(target[key], saved[key])
        return target
    if isinstance(saved, (list, tuple)):
        if type(target) is not type(saved) or len(target) != len(saved):
            raise ValueError("cached optimizer state changed sequence")
        values = [restore_tree(a, b) for a, b in zip(target, saved)]
        return tuple(values) if isinstance(saved, tuple) else values
    return copy.deepcopy(saved)


class LocalTrainingSnapshot:
    """CPU copies of source shards; never another full model on a GPU."""
    def __init__(self, engine):
        self.parameters = {name: clone_cpu(p) for name, p in engine.module.named_parameters()}
        # Nonpersistent buffers (for example rotary frequencies) also affect a
        # repeated forward, although they are absent from checkpoint state_dict.
        self.buffers = {name: clone_cpu(b) for name, b in engine.module.named_buffers()}
        self.training_modes = {name: module.training for name, module in engine.module.named_modules()}
        self.optimizer = {p: clone_cpu(state) for p, state in engine.optimizer.state.items()}
        self.groups = [{k: clone_cpu(v) for k, v in group.items() if k != 'params'}
                       for group in engine.optimizer.param_groups]
        self.scheduler = clone_cpu(engine.lr_scheduler.state_dict())
        self.rng = engine.checkpoint_manager.get_rng_state()
        self.rng_digest = training_state_digest(self.rng)

    @torch.no_grad()
    def restore(self, engine, *, optimizer=True):
        engine.optimizer_zero_grad()
        parameters = dict(engine.module.named_parameters())
        if parameters.keys() != self.parameters.keys():
            raise ValueError("audit parameter identities changed")
        for name, p in parameters.items():
            if local(p).shape != self.parameters[name].shape:
                raise ValueError("audit parameter shard layout changed")
            local(p).copy_(self.parameters[name])
        buffers = dict(engine.module.named_buffers())
        modules = dict(engine.module.named_modules())
        if buffers.keys() != self.buffers.keys() or modules.keys() != self.training_modes.keys():
            raise ValueError("audit buffer/module identities changed")
        for name, buffer in buffers.items():
            saved = self.buffers[name]
            if local(buffer).shape != saved.shape or local(buffer).dtype != saved.dtype:
                raise ValueError("audit buffer layout changed; cannot claim identical replay state")
            local(buffer).copy_(saved)
        for name, module in modules.items():
            # Assign individually: recursive train() would overwrite child modes.
            module.training = self.training_modes[name]
        if optimizer:
            for p in list(engine.optimizer.state):
                if p not in self.optimizer:
                    del engine.optimizer.state[p]
            for p, state in self.optimizer.items():
                restore_tree(engine.optimizer.state[p], state)
            if len(self.groups) != len(engine.optimizer.param_groups):
                raise ValueError("optimizer group layout changed")
            for group, state in zip(engine.optimizer.param_groups, self.groups):
                for key, value in state.items():
                    group[key] = clone_cpu(value)
            engine.lr_scheduler.load_state_dict(clone_cpu(self.scheduler))
        engine.checkpoint_manager.load_rng_state(self.rng)


def optimizer_counts(engine):
    return {name: float(engine.optimizer.state[p]['step']) if 'step' in engine.optimizer.state.get(p, {}) else None
            for name, p in engine.module.named_parameters()}


def one_update_counts(before, after, gradients_present):
    if set(before) != set(after) or set(before) != set(gradients_present):
        return False
    for name, previous in before.items():
        expected = (0. if previous is None else previous) + 1 if gradients_present[name] else previous
        if after[name] != expected:
            return False
    return True


def summarize_checks(checks):
    """Full local-shard sufficient statistics; driver sums before ratios."""
    result = {}
    for branch, prefix in [('all', ''), ('qwen', 'query_forward.model.'),
                           ('query', 'query_forward.query_encoder.'), ('fm', 'action_head.')]:
        selected = [row for name, row in checks.items() if not prefix or name.startswith(prefix)]
        base = sum(row.get('baseline_squared', 0.) for row in selected)
        error = sum(row.get('error_squared', 0.) for row in selected)
        current = sum(row.get('candidate_squared', 0.) for row in selected)
        dot = (base + current - error) / 2
        result[branch] = dict(baseline_squared=base, candidate_squared=current,
            error_squared=error, dot_product=dot,
            relative_l2=math.sqrt(error/max(base, 1e-30)),
            cosine=dot/math.sqrt(base*current) if base*current > 0 else None,
            norm_ratio=math.sqrt(current/base) if base > 0 else None,
            missing_gradient_count=sum(bool(row.get('missing_gradient')) for row in selected))
    return result


def compare(left, right, *, atol, rtol):
    result = tensor_difference(left, right, atol=atol, rtol=rtol)
    if right is not None and not result.get('nonfinite'):
        value = right.detach().cpu().double()
        result['candidate_squared'] = float(value.square().sum())
    return result


class StackedRuleReplayAuditWorker(ActionHeadGRPOWorker):
    """Final B1 versus stacked v3 numerical gate; never times individual knobs.

    This separate worker does not alter any existing training startup. The
    driver must inspect both prepare reports on every rank before run_replay.
    """
    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def prepare_stacked_rule_replay(self, baseline_records, stacked_records, logical_valid):
        from action_head_rl.rule_shared_audit import prepare_stacked_rule_payloads
        from action_head_rl.worker import _cpu_model_positions, joint_payload_signature
        from action_head_rl.joint_policy import consecutive_joint_groups, consecutive_joint_batches
        if self.engine.optimizer.state:
            raise ValueError('stacked v3 replay requires fresh Adam at step0')
        if any(p.is_floating_point() and p.dtype != torch.float32
               for p in self.engine.module.parameters()):
            raise ValueError('stacked v3 replay requires FP32 master parameters')
        self._stacked_rule_replay = None
        self._stacked_rule_preflight_report = None
        positions = lambda payload: _cpu_model_positions(
            self.engine.module.query_forward.model.model, payload)
        baseline, stacked, report = prepare_stacked_rule_payloads(
            baseline_records, stacked_records, processor=self.model_config.processor,
            positions=positions, logical_valid=logical_valid)
        backbone = self.engine.module.query_forward._backbone()
        profiles = {}
        for name, records, payloads in (('baseline', baseline_records, baseline),
                                        ('stacked', stacked_records, stacked)):
            batches = consecutive_joint_batches(consecutive_joint_groups(records))
            profiles[name] = [joint_payload_signature(data,
                family=batch[0][0]['_joint_family'],
                width=int(batch[0][0]['_condition_pack_token_width']),
                fm_chunks=sum(row.get('record_kind') != 'text_policy' for row in batch[0]),
                batch_size=len(batch), query_count=self.engine.module.query_encoder.QUERY_COUNT,
                layer_count=len(backbone.layers), top_k=int(backbone.config.num_experts_per_tok))
                for batch, data in zip(batches, payloads)]
        self._stacked_rule_replay = (baseline, stacked)
        self._stacked_rule_preflight_report = dict(rank=self.rank, **report,
                                                  collective_signatures=profiles)
        return self._stacked_rule_preflight_report

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def run_stacked_rule_replay(self, output_dir, prepared_reports):
        from action_head_rl.rule_shared_audit import audit_stacked_rule_payloads, validate_stacked_rule_reports
        if not getattr(self, '_stacked_rule_replay', None):
            raise RuntimeError('CPU matched replay preflight must precede the GPU gate')
        validate_stacked_rule_reports(prepared_reports, torch.distributed.get_world_size())
        if next(row for row in prepared_reports if row['rank'] == self.rank) != self._stacked_rule_preflight_report:
            raise ValueError('GPU gate report does not match this worker preflight')
        baseline, stacked = self._stacked_rule_replay
        try:
            context = (nullcontext() if getattr(self, '_streamed_train_context', None) is not None
                       else self.engine.train_mode(disable_auto_offload=False))
            with context:
                report = audit_stacked_rule_payloads(self.engine, baseline, stacked, output_dir,
                    candidate_checkpoint_layers_off=int(os.environ.get(
                        'ACTION_HEAD_FULL_ATTENTION_CHECKPOINT_LAYERS_OFF', '0')),
                    candidate_full_weight_residency_gib=float(os.environ.get(
                        'ACTION_HEAD_FULL_WEIGHT_RESIDENCY_GIB', '8')),
                    candidate_checkpoint_selection='full_attention')
            # The diagnostic is warmup, not part of the production step peak.
            torch.cuda.reset_peak_memory_stats()
            return report
        finally:
            self._stacked_rule_replay = None
            self._stacked_rule_preflight_report = None


class PackingReplayAuditWorker(ActionHeadGRPOWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def packing_process_identity(self):
        return dict(rank=self.rank, pid=os.getpid(),
                    start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19])

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def prepare_packing_replay(self, variants, plan):
        if not self.engine.optimizer.state:
            raise ValueError("packing replay must restore the source16 Adam states")
        if any(p.is_floating_point() and p.dtype != torch.float32 for p in self.engine.module.parameters()):
            raise ValueError("packing replay requires FP32 master parameters")
        if self.engine.module.query_encoder.QUERY_COUNT != 16:
            raise ValueError("packing replay requires the actual sixteen-query policy")
        if ({value for value in optimizer_counts(self.engine).values() if value is not None} != {16.}
                or self.engine.lr_scheduler.last_epoch != 16):
            raise ValueError('packing replay did not restore source16 Adam/scheduler counters')
        self._packing_replay_cpu = {size: pack_stream(rows, self.model_config.processor, int(size), plan['token_cap'])
                                    for size, rows in variants.items()}
        return dict(rank=self.rank, model_class=type(self.engine.module).__name__,
                    trainable_parameter_count=sum(p.numel() for p in self.engine.module.parameters() if p.requires_grad),
                    signatures={size: dict(tensors=packed_signature(data), collective=collective_signature(data))
                                               for size, data in self._packing_replay_cpu.items()})

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def run_packing_replay(self, plan):
        engine = self.engine
        parameters = dict(engine.module.named_parameters())
        limits = plan['thresholds']
        report = dict(rank=self.rank, variants={}, timings={}, optimizer_updates=0)
        # CPU-heavy copies/checks are all outside synchronized timing regions.
        started = time.perf_counter()
        snapshot = LocalTrainingSnapshot(engine)
        report['snapshot_seconds'] = time.perf_counter() - started
        source_counts = optimizer_counts(engine)
        if {value for value in source_counts.values() if value is not None} != {16.} or engine.lr_scheduler.last_epoch != 16:
            raise ValueError('numeric replay must start at actual source16 Adam/scheduler counters')
        report['source_optimizer_steps'] = sorted({value for value in source_counts.values() if value is not None})
        report['source_scheduler_last_epoch'] = engine.lr_scheduler.last_epoch
        baseline_grad, baseline_delta = None, None
        os.environ['ACTION_HEAD_JOINT_TEXT_FM_FORWARD'] = '0'
        # Keep alternate layouts on CPU. Each timed run transfers only its own
        # inputs before the timer, so B8's peak does not include B1/B2/B4 copies.
        packed = self._packing_replay_cpu
        try:
            with engine.train_mode(disable_auto_offload=False):
                engine.module.set_requires_gradient_sync(True, recurse=True)
                if hasattr(engine.module, 'set_is_last_backward'):
                    engine.module.set_is_last_backward(True)

                def backward(size):
                    engine.optimizer_zero_grad()
                    engine.checkpoint_manager.load_rng_state(snapshot.rng)
                    inputs = [data.to(self.device_name) for data in packed[size]]
                    torch.cuda.synchronize()
                    torch.distributed.barrier()
                    torch.cuda.reset_peak_memory_stats()
                    baseline_allocated = torch.cuda.memory_allocated()/1024**3
                    baseline_reserved = torch.cuda.memory_reserved()/1024**3
                    start = time.perf_counter()
                    losses = []
                    for data in inputs:
                        result = engine.forward_backward_batch(data, loss_function=lambda **_: None, forward_only=False)
                        losses.append(float(result['loss']))
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - start
                    free, total = torch.cuda.mem_get_info()
                    total_loss = sum(losses)
                    return dict(seconds=seconds, loss=total_loss if math.isfinite(total_loss) else None, forwards=len(packed[size]),
                        finite_loss=all(math.isfinite(x) for x in losses),
                        baseline_allocated_gib=baseline_allocated, baseline_reserved_gib=baseline_reserved,
                        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                        post_backward_device_used_gib=(total-free)/1024**3)

                for batch in BATCH_SIZES:
                    size = str(batch)
                    restore_start = time.perf_counter()
                    snapshot.restore(engine)
                    if optimizer_counts(engine) != source_counts:
                        raise RuntimeError("source optimizer counters were not restored")
                    if training_state_digest(engine.checkpoint_manager.get_rng_state()) != snapshot.rng_digest:
                        raise RuntimeError("source RNG was not restored")
                    numeric = backward(size)
                    numeric['restore_and_cold_seconds'] = time.perf_counter() - restore_start
                    presence = {name: p.grad is not None for name, p in parameters.items()}
                    if batch == 1:
                        baseline_grad = {name: None if p.grad is None else clone_cpu(p.grad) for name, p in parameters.items()}
                    checks = {name: compare(baseline_grad[name], None if p.grad is None else local(p.grad),
                        atol=limits['gradient_atol'], rtol=limits['gradient_rtol']) for name, p in parameters.items()}
                    torch.cuda.synchronize()
                    update_start = time.perf_counter()
                    norm = float(engine.optimizer_step())
                    engine.lr_scheduler_step()
                    torch.cuda.synchronize()
                    numeric.update(adam_scheduler_seconds=time.perf_counter()-update_start, grad_norm=norm if math.isfinite(norm) else None,
                        optimizer_steps=int(math.isfinite(norm)),
                        optimizer_counters_correct=one_update_counts(source_counts, optimizer_counts(engine), presence),
                        scheduler_correct=engine.lr_scheduler.last_epoch == report['source_scheduler_last_epoch'] + 1)
                    report['optimizer_updates'] += int(math.isfinite(norm))
                    delta_checks = {}
                    if batch == 1:
                        baseline_delta = {}
                    for name, p in parameters.items():
                        difference = local(p).detach().cpu() - snapshot.parameters[name]
                        if batch == 1:
                            baseline_delta[name] = difference.clone()
                        delta_checks[name] = compare(baseline_delta[name], difference,
                            atol=limits['delta_atol'], rtol=limits['delta_rtol'])
                    numeric.update(gradient_checks=checks, parameter_delta_checks=delta_checks,
                        gradient_summary=summarize_checks(checks), parameter_delta_summary=summarize_checks(delta_checks))
                    numeric['passed'] = (numeric['finite_loss'] and math.isfinite(norm)
                        and numeric['optimizer_counters_correct'] and numeric['scheduler_correct']
                        and all(row['passed'] for table in (checks, delta_checks) for row in table.values()))
                    report['variants'][size] = numeric
                    write_new_json(Path(plan['output'])/'rank_reports'/f'rank{self.rank:02d}_B{size}_numeric.json',
                        dict(plan_sha256=plan['plan_sha256'], rank=self.rank, batch_size=batch,
                             scope='partial per-rank evidence, not a global pass',
                             source_checkpoint=plan.get('source', {}).get('checkpoint'),
                             source_optimizer_steps=report['source_optimizer_steps'],
                             source_scheduler_last_epoch=report['source_scheduler_last_epoch'],
                             restored_source_rng_sha256=snapshot.rng_digest,
                             logical_denominator=32., **numeric))
                del baseline_grad, baseline_delta
                snapshot.restore(engine)
                for _ in range(plan['warmup']):
                    for batch in BATCH_SIZES:
                        backward(str(batch))
                # Alternate complete sweeps; there is no optimizer update in
                # these steady-state measurements and no change in targets.
                for repeat in range(plan['repeats']):
                    order = BATCH_SIZES if repeat % 2 == 0 else tuple(reversed(BATCH_SIZES))
                    report['timings'][str(repeat)] = {}
                    for batch in order:
                        timing = backward(str(batch))
                        report['timings'][str(repeat)][str(batch)] = timing
                        write_new_json(Path(plan['output'])/'rank_reports'/f'rank{self.rank:02d}_B{batch}_repeat{repeat}.json',
                            dict(plan_sha256=plan['plan_sha256'], rank=self.rank, batch_size=batch, repeat=repeat,
                                 scope='partial per-rank timing, not a global pass', **timing))
                snapshot.restore(engine)
                report['source_restored'] = (optimizer_counts(engine) == source_counts
                    and engine.lr_scheduler.last_epoch == report['source_scheduler_last_epoch']
                    and training_state_digest(engine.checkpoint_manager.get_rng_state()) == snapshot.rng_digest)
                report['passed'] = report['source_restored'] and all(row['passed'] for row in report['variants'].values())
        finally:
            self._packing_replay_cpu = None
            engine.optimizer_zero_grad()
        return report
