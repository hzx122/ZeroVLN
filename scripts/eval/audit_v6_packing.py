#!/usr/bin/env python3
"""Prepare native B1/2/4/8 replay on CPU; execute separately on idle GPU0..11."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--validate-plan', type=Path)
    group.add_argument('--execute-plan', type=Path)
    parser.add_argument('--fixture-root', type=Path)
    parser.add_argument('--allow-two-pair-shape-audit', action='store_true')
    parser.add_argument('--source-checkpoint', type=Path)
    parser.add_argument('--runtime-plan', type=Path)
    parser.add_argument('--after-process-record', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--token-cap', type=int, default=40960)
    parser.add_argument('--token-bucket', type=int, default=1024)
    parser.add_argument('--thresholds', type=Path)
    parser.add_argument('--timeout-seconds', type=float, default=7200)
    parser.add_argument('--expected-plan-sha256')
    parser.add_argument('--runtime-configured', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--cpu-receipt-sha256', help=argparse.SUPPRESS)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo/'verl'))
    if args.execute_plan:
        # Configure devices/overlay in a NEW interpreter before importing
        # Torch/VERL; CPU prepare never contaminates CUDA availability caches.
        plan = json.loads(args.execute_plan.read_text())
        if not args.runtime_configured:
            if not args.expected_plan_sha256:
                parser.error('execute requires the reviewed expected plan SHA')
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            for path in reversed(plan['environment']['PYTHONPATH'].split(':')):
                sys.path.insert(0, path)
            from action_head_rl.packing_audit import validate_plan, SCHEMA
            from action_head_rl.joint_audit import sha256, write_new_json
            if sha256(args.execute_plan) != args.expected_plan_sha256:
                parser.error('reviewed plan SHA differs')
            validate_plan(plan)
            receipt = Path(plan['output'])/'execute_cpu_preflight.json'
            write_new_json(receipt, dict(schema=SCHEMA, plan_sha256=args.expected_plan_sha256,
                                        cpu_validation='passed'))
            env = {k: v for k, v in os.environ.items() if not k.startswith('ACTION_HEAD_')}
            env.update(plan['environment'])
            os.execvpe(sys.executable, [sys.executable, str(Path(__file__).resolve()),
                *sys.argv[1:], '--runtime-configured', '--cpu-receipt-sha256', sha256(receipt)], env)
        if any(os.environ.get(k) != v for k, v in plan['environment'].items()):
            parser.error('execute runtime does not match the prepared environment')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
        plan = json.loads(args.validate_plan.read_text()) if args.validate_plan else None
    if args.runtime_plan:
        sys.path.insert(1, json.loads(args.runtime_plan.read_text())['snapshot_upstream'])
    elif plan:
        for path in reversed(plan['environment']['PYTHONPATH'].split(':')):
            sys.path.insert(0, path)
    from action_head_rl.packing_audit import prepare_plan, validate_plan, execute_plan
    from action_head_rl.joint_audit import sha256, write_new_json
    if args.prepare:
        if not all((args.fixture_root, args.source_checkpoint, args.runtime_plan, args.output)):
            parser.error('prepare requires fixture root, source checkpoint, runtime plan and new output')
        complete = sorted(args.fixture_root.glob('step_000017_rollout_*/complete.json'))
        # Predeclared inclusion rule: all completed captures from selected32,
        # sorted by original index, never chosen by rewards/advantages.
        indices = [json.loads(path.read_text())['rollout_index'] for path in complete]
        if len(complete) > 32 or indices != sorted(set(indices)):
            parser.error('capture set must contain unique original selected32 indices')
        if len(complete) < 8 and not (args.allow_two_pair_shape_audit and indices == [0, 1]):
            parser.error('pending: fewer than eight original complete captures; no replacements selected')
        plan = prepare_plan(fixture_dirs=[path.parent for path in complete], source_checkpoint=args.source_checkpoint,
            runtime_plan=args.runtime_plan, output=args.output, after_process_records=args.after_process_record,
            token_cap=args.token_cap, token_bucket=args.token_bucket, timeout_seconds=args.timeout_seconds,
            thresholds=json.loads(args.thresholds.read_text()) if args.thresholds else None)
        plan['fixture_collection'] = dict(root=str(args.fixture_root.resolve()), complete_indices=indices,
                                        rule='all complete selected32 captures in original index order')
        path = args.output/'plan.json'
        write_new_json(path, plan)
        print(json.dumps(dict(plan=str(path), sha256=sha256(path), gpu_executed=False,
                             effective_real_conditions=plan['effective_real_conditions'], schedule=plan['schedule'])))
    elif args.validate_plan:
        validate_plan(plan)
        print(json.dumps(dict(plan=str(args.validate_plan), sha256=sha256(args.validate_plan),
                              cpu_validation='passed', gpu_executed=False)))
    else:
        if not args.expected_plan_sha256:
            parser.error('execute requires the reviewed expected plan SHA')
        result = execute_plan(args.execute_plan, expected_sha256=args.expected_plan_sha256,
                              cpu_receipt_sha256=args.cpu_receipt_sha256)
        print(json.dumps({k: v for k, v in result.items() if k != 'ranks'}))
        if result['status'] != 'passed':
            raise SystemExit(2)


if __name__ == '__main__':
    main()
