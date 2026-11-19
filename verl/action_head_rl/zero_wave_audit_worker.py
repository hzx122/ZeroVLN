"""Run original/compacted waves through the actual streamed optimizer lifecycle."""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import torch

from action_head_rl.joint_audit import write_new_json
from action_head_rl.packing_audit_worker import (
    LocalTrainingSnapshot, optimizer_counts, one_update_counts, clone_cpu, local, compare, summarize_checks,
)
from action_head_rl.worker import ActionHeadGRPOWorker
from action_head_rl.zero_wave_audit import (
    SCENARIOS, VARIANTS, signatures, model_state_receipt, compare_receipts,
)
from verl.single_controller.base.decorator import Dispatch, register


class ZeroWaveAuditWorker(ActionHeadGRPOWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def audit_process_identity(self):
        return dict(rank=self.rank, pid=os.getpid(),
                    start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19])

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def prepare_zero_replay(self, scenarios):
        if (not self.engine.optimizer.state
                or {x for x in optimizer_counts(self.engine).values() if x is not None} != {16.}
                or self.engine.lr_scheduler.last_epoch != 16
                or self.engine.module.query_encoder.QUERY_COUNT != 16
                or any(p.dtype != torch.float32 for p in self.engine.module.parameters() if p.is_floating_point())):
            raise ValueError('zero-wave replay requires actual FP32 source16 full optimizer/query/scheduler')
        self._zero_replay = scenarios
        cache = {}
        return dict(rank=self.rank, signatures={name: {
            variant: signatures(waves, self.model_config.processor, cache=cache) for variant, waves in variants.items()}
            for name, variants in scenarios.items()})

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def run_zero_replay(self, plan):
        engine = self.engine
        snapshot = LocalTrainingSnapshot(engine)
        source = model_state_receipt(engine)
        source_counts = optimizer_counts(engine)
        report = dict(rank=self.rank, scenarios={}, passed=False)
        scenarios = plan.get('scenarios', list(SCENARIOS))
        variants = plan.get('variants', list(VARIANTS))
        if (scenarios, variants) not in ((list(SCENARIOS), list(VARIANTS)),
                                         (['zero_first_repeat'], ['original', 'repeat', 'compact'])):
            raise ValueError('unsupported zero-wave replay protocol')
        repeatability = variants == ['original', 'repeat', 'compact']
        reference_gradients = {}
        try:
            for scenario in scenarios:
                paired = {}
                for variant in variants:
                    snapshot.restore(engine)
                    if model_state_receipt(engine) != source:
                        raise RuntimeError('full source state failed restoration before replay')
                    waves = self._zero_replay[scenario][variant]
                    expected = plan['schedule'][scenario][variant]
                    self.begin_streamed_train(expected_accumulations=len(waves))
                    engine.module.set_requires_gradient_sync(True, recurse=True)
                    if hasattr(engine.module, 'set_is_last_backward'):
                        engine.module.set_is_last_backward(True)
                    torch.cuda.synchronize()
                    torch.distributed.barrier()
                    start = time.perf_counter()
                    wave_metrics = [self.accumulate_streamed_records(rows) for rows in waves]
                    torch.cuda.synchronize()
                    backward_seconds = time.perf_counter() - start
                    # Capture after the production finalizer applies its denominator,
                    # immediately before clipping/Adam. Restore even on exceptions.
                    step = engine.optimizer_step
                    gradients = {}
                    gradient_checks = {}
                    def capture_step():
                        gradients.update(model_state_receipt(engine, gradients=True))
                        if repeatability:
                            for name, parameter in engine.module.named_parameters():
                                if variant == 'original':
                                    reference_gradients[name] = clone_cpu(parameter.grad)
                                else:
                                    gradient_checks[name] = compare(reference_gradients[name],
                                        local(parameter.grad), atol=0., rtol=0.)
                        return step()
                    engine.optimizer_step = capture_step
                    try:
                        update = self.finish_streamed_train(expected['logical_mass'])
                    finally:
                        engine.optimizer_step = step
                    after = model_state_receipt(engine)
                    presence = {name: row['present'] for name, row in gradients.items()}
                    valid = (bool(gradients) and any(presence.values())
                        and all(row['finite'] for row in gradients.values())
                        and one_update_counts(source_counts, optimizer_counts(engine), presence)
                        and engine.lr_scheduler.last_epoch == 17 and after['rng'] == source['rng']
                        and update is not None and update['optimizer_steps'] == 1
                        and math.isfinite(update['grad_norm']))
                    if scenario.endswith('zero_only'):
                        valid = valid and all(row['zero'] for row in gradients.values())
                    paired[variant] = dict(gradients=gradients, **after)
                    row = dict(passed=valid, backward_seconds=backward_seconds,
                        timing_scope='single replay including CPU record processing; not steady-state or rollout throughput',
                        update=update, waves=wave_metrics, state=paired[variant])
                    if gradient_checks:
                        row['gradient_repeatability'] = dict(
                            per_parameter=gradient_checks, branches=summarize_checks(gradient_checks),
                            diagnostic_only=True, acceptance_tolerance_changed=False)
                    report['scenarios'].setdefault(scenario, {})[variant] = row
                    write_new_json(Path(plan['output'])/'rank_reports'/f'rank{self.rank:02d}_{scenario}_{variant}.json',
                        dict(plan_sha256=plan['plan_sha256'], rank=self.rank, **row))
                equality = compare_receipts(paired[variants[0]], paired[variants[1]])
                if repeatability:
                    equality = {f'{variant}/{key}': value for variant in variants[1:]
                                for key, value in compare_receipts(paired['original'], paired[variant]).items()}
                report['scenarios'][scenario]['exact_equality'] = equality
                report['scenarios'][scenario]['passed'] = (all(equality.values()) and all(
                    report['scenarios'][scenario][variant]['passed'] for variant in variants))
            snapshot.restore(engine)
            report['source_restored'] = model_state_receipt(engine) == source
            report['passed'] = report['source_restored'] and all(row['passed'] for row in report['scenarios'].values())
        finally:
            self.abort_streamed_train()
            self._zero_replay = None
            engine.optimizer_zero_grad()
        return report
