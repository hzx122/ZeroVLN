#!/usr/bin/env python3
"""Thin adapter for the frozen six-stage runner's nested-supervisor cleanup.

All launch locking, process identity checks, stage execution and receipts remain
in the existing runner. Its child supervisors need their own 20-second cleanup
deadline to expire before the parent may escalate, so the outer grace is 60s.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SELF = Path(__file__).resolve()
# Activation adds the collector source to sys.path. Its orchestration files may
# be an older version; detached children must import their own frozen sibling.
_composition_path = SELF.with_name('formal_rule_teacher_pipeline.py')
composition = sys.modules.get('formal_rule_teacher_pipeline')
if composition is None or Path(composition.__file__).resolve() != _composition_path:
    _descriptor = importlib.util.spec_from_file_location('formal_rule_teacher_pipeline', _composition_path)
    composition = importlib.util.module_from_spec(_descriptor)
    sys.modules['formal_rule_teacher_pipeline'] = composition
    _descriptor.loader.exec_module(composition)

STOP_GRACE_SECONDS = 60
COLLECTION_STOP_GRACE_SECONDS = 45


def configure(source_root):
    runner, _ = composition.activate(source_root)
    if not getattr(runner, '_formal_nested_cleanup_adapter', False):
        original_stop = runner.stop_process_group
        runner._formal_base_stop_process_group = original_stop

        def stop_with_nested_grace(process, timeout=20):
            return original_stop(process, timeout=max(timeout, STOP_GRACE_SECONDS))

        # stop_owned itself is unchanged, including its PID/start-ticks guard.
        runner.stop_process_group = stop_with_nested_grace
        runner._formal_nested_cleanup_adapter = True
    if not getattr(runner, '_formal_cpu_reservation_adapter', False):
        original_pipeline = runner.Pipeline

        class PipelineWithCpuReservations(original_pipeline):
            def reserve(self, stage):
                if stage['kind'] == 'validate':
                    spec_path = self.output/'spec.json'
                    spec = composition.load(spec_path)
                    composition.check_brains_released(spec)
                    composition.save(self.output/'receipts/pre-reservation-idle.json', dict(
                        schema='formal_pre_reservation_idle_v1', stage='validate',
                        gpu_ids=list(range(16)), idle=True, spec=composition.binding(spec_path),
                        brain_completion=composition.binding(self.output/'brains/completion.json')))
                if stage['kind'] == 'initialize' and stage.get('initialization_on_cpu'):
                    stage = dict(stage, idle_gpu_ids=list(range(16)))
                return super().reserve(stage)

        runner.Pipeline = PipelineWithCpuReservations
        runner._formal_cpu_reservation_adapter = True
    # The existing launch() derives its child entrypoint from this value. REPO
    # remains the original frozen repository; cwd and runtime boundaries do not
    # change. The detached child enters this adapter and uses the same execute().
    runner.__file__ = str(SELF)
    return runner


def run_collection(path, runner):
    """Reuse frozen phase execution, allowing its eight workers to clean together.

Each worker is itself a supervisor whose children have independent sessions.
Sending TERM twice can interrupt that supervisor's finally block. Broadcast once
to identity-verified owned workers, then wait against one common deadline.
"""
    import prepare_rule_teacher_formal_local16_collection as collection
    original_class = collection.CollectionPhases
    original_stop_owned = runner.stop_owned

    class CollectionWithOwnedCleanup(original_class):
        def release_reservation(self, reason):
            if reason == 'pipeline_exit' and self.cleanup_deadline is None:
                self.cleanup_deadline = time.monotonic()+COLLECTION_STOP_GRACE_SECONDS
                # Unknown/reused identities remain un-signalled; stop_owned will
                # record the failure while other verified children are cleaned.
                valid = [(p, identity) for _, p, identity in self.owned
                    if p.poll() is None and identity and runner.same_process_alive(identity)]
                for process, _ in valid:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        self.signalled.add(process.pid)
                    except ProcessLookupError:
                        pass
            return super().release_reservation(reason)

        def stop_owned(self, process, identity):
            if self.cleanup_deadline is None:
                return original_stop_owned(process, identity)
            if process.poll() is None:
                if identity is None or not runner.same_process_alive(identity):
                    raise RuntimeError(f'owned process identity changed; refusing signal: {process.pid}')
                remaining = max(.1, self.cleanup_deadline-time.monotonic())
                if process.pid not in self.signalled:
                    # Reservations retain their existing stop helper and receipt
                    # handling, using the same shared cleanup deadline.
                    runner._formal_base_stop_process_group(process, timeout=remaining)
                else:
                    try:
                        process.wait(timeout=remaining)
                    except subprocess.TimeoutExpired:
                        if not runner.same_process_alive(identity):
                            raise RuntimeError(f'owned identity changed before escalation: {process.pid}')
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
            if process.poll() is None:
                raise RuntimeError(f'owned process did not stop: {process.pid}')

        def run(self):
            self.cleanup_deadline, self.signalled = None, set()
            runner.stop_owned = self.stop_owned
            try:
                return super().run()
            finally:
                runner.stop_owned = original_stop_owned
                composition.save(self.output/'owned-cleanup.json', dict(
                    schema='formal_collection_owned_cleanup_v1', broadcast_pids=sorted(self.signalled),
                    shared_grace_seconds=COLLECTION_STOP_GRACE_SECONDS,
                    processes=[dict(identity=identity, returncode=p.poll(),
                        live=bool(identity and runner.same_process_alive(identity)))
                        for _, p, identity in self.owned]))

    collection.CollectionPhases = CollectionWithOwnedCleanup
    try:
        return collection.run_phases(path)
    finally:
        collection.CollectionPhases = original_class


def runner_for_plan(path):
    plan = composition.load(path)
    descriptor = plan['orchestration']
    if (descriptor['runner_adapter'] != str(SELF)
            or descriptor['stop_grace_seconds'] != STOP_GRACE_SECONDS
            or descriptor['collection_stop_grace_seconds'] != COLLECTION_STOP_GRACE_SECONDS
            or plan['input_sha256'].get(str(SELF)) != composition.sha(SELF)
            or plan['input_sha256'].get(str(composition.SELF)) != composition.sha(composition.SELF)):
        raise ValueError('formal runner adapter is not frozen in this plan')
    runner = configure(descriptor['source_root'])
    runner.validate_plan(plan, path)
    return runner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--run-collection', action='store_true')
    mode.add_argument('--status', action='store_true')
    args = parser.parse_args(argv)
    runner = runner_for_plan(args.plan.resolve())
    if args.status:
        result = runner.read_status_json(args.plan.parent/'status.json')
        identity = result.get('supervisor')
        result['supervisor_alive'] = bool(identity and runner.same_process_alive(identity))
        pending = args.plan.parent/'receipts/collection-review-pending.json'
        if pending.exists() and not (pending.parent/'collection-user-approval.json').exists():
            result['collection_review'] = composition.load(pending)
        print(composition.json.dumps(result, ensure_ascii=False))
        return 0
    if args.run_collection:
        return run_collection(args.plan.resolve().parent/'collection/phase-plan.json', runner)
    return runner.execute(args.plan.resolve())


if __name__ == '__main__':
    sys.exit(main())
