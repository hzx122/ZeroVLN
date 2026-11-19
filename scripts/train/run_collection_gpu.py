"""Run an explicit collection service/collector plan without a DLC resident."""
import argparse
import json
from pathlib import Path
import sys
from gpu_runtime import REPO, environment, execute, load, save


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True,help='existing supervisor launch JSON with services/collectors/source_plan')
    p.add_argument('--show-config',action='store_true')
    a=p.parse_args(argv)
    plan=load(a.plan)
    if a.show_config:
        print(json.dumps(plan,indent=2));return 0
    # Existing supervisor owns service readiness, PID/start-time checks and cleanup.
    if not {'services','collectors','source_plan','runtime'} <= set(plan):
        p.error('use a prepared collection launch plan with services/collectors/source_plan/runtime')
    root=a.plan.resolve().parent
    env=environment(root,dict(GPU_DIRECT_RUN='1', RULE_COLLECTION_RUNTIME_ROOT=str(Path(plan['runtime']).parent)))
    resolved=root/'gpu-launch.json'
    if resolved.exists():
        p.error('existing GPU launch attempt; use a fresh plan directory')
    save(resolved,plan)
    command=[sys.executable,'-B',str(REPO/'scripts/train/supervise_rule_teacher_collection.py'),
             '--plan',str(resolved)]
    return execute(command,env,root/'launch/gpu-collection.json')


if __name__ == '__main__':
    sys.exit(main())
