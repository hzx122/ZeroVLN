#!/usr/bin/env python3
"""Plan/validate a CPU replay audit, or explicitly execute its reviewed GPU plan.

No mode launches training, rollout, vLLM, or rendering services. GPU execution
is a fresh twelve-rank FSDP forward/backward audit with optimizer updates banned.
"""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "verl"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--validate-plan", type=Path)
    mode.add_argument("--execute-plan", type=Path)
    parser.add_argument("--fixture-dir", type=Path, action="append")
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--runtime-plan", type=Path)
    parser.add_argument("--after-process-record", action="append", default=[], metavar="LABEL=JSON")
    parser.add_argument("--prerequisite-profile", default="after_inputfix_and_credit",
                        choices=("after_inputfix_and_credit", "after_inputfix_before_credit"))
    parser.add_argument("--pending-credit-run-dir", type=Path)
    parser.add_argument("--pending-credit-launch-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rpc-timeout-seconds", type=float, default=3600)
    args = parser.parse_args()
    if args.runtime_plan:
        runtime = json.loads(args.runtime_plan.read_text())
        sys.path.insert(1, runtime["snapshot_upstream"])
    elif args.validate_plan or args.execute_plan:
        recorded = json.loads((args.validate_plan or args.execute_plan).read_text())
        for path in reversed(recorded["environment"]["PYTHONPATH"].split(":")):
            sys.path.insert(0, path)
    from action_head_rl.joint_audit import (
        prepare_plan, validate_plan, execute_plan, sha256, read_json, write_new_json)
    if args.prepare:
        if not all((args.fixture_dir, args.source_checkpoint, args.runtime_plan, args.output)):
            parser.error("--prepare requires fixtures, source checkpoint, runtime plan and output")
        identities = {}
        for item in args.after_process_record:
            label, path = item.split("=", 1)
            if label in identities:
                parser.error("duplicate prerequisite process label")
            identities[label] = path
        plan = prepare_plan(fixture_dirs=args.fixture_dir, source_checkpoint=args.source_checkpoint,
            runtime_plan=args.runtime_plan, prerequisites=identities, output=args.output,
            prerequisite_profile=args.prerequisite_profile,
            pending_credit_run_dir=args.pending_credit_run_dir,
            pending_credit_launch_dir=args.pending_credit_launch_dir,
            warmup=args.warmup, repeats=args.repeats,
            rpc_timeout_seconds=args.rpc_timeout_seconds,
            thresholds=read_json(args.thresholds) if args.thresholds else None)
        path = args.output / "plan.json"
        write_new_json(path, plan)
        print(json.dumps({"plan": str(path), "sha256": sha256(path), "cpu_preflight": plan["cpu_preflight"],
                          "gpu_executed": False}))
    elif args.validate_plan:
        validate_plan(read_json(args.validate_plan))
        print(json.dumps({"plan": str(args.validate_plan), "sha256": sha256(args.validate_plan),
                          "cpu_validation": "passed", "gpu_executed": False}))
    else:
        if not args.expected_plan_sha256:
            parser.error("--execute-plan requires the previously reviewed --expected-plan-sha256")
        result = execute_plan(args.execute_plan, expected_sha256=args.expected_plan_sha256)
        print(json.dumps({key: value for key, value in result.items() if key != "ranks"}))
        if result["status"] != "passed":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
