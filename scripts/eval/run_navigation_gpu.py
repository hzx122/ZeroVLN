"""Direct unified evaluation and navigation console on an existing GPU host."""
import argparse
import json
import os
from pathlib import Path
import sys
from gpu_runtime import REPO, artifact_root, environment, execute, load, save


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'run', 'console', 'smoke', 'show-config'])
    p.add_argument('--config', type=Path, required=True, help='unified selection JSON')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--port', type=int, choices=range(5000,5006), default=5002)
    p.add_argument('--navigation-repo', default=os.environ.get('RYNNBRAIN_NAV_DIR'))
    p.add_argument('--smoke-gpus', default='0,1', help='smoke only: two physical devices for TP2; FM shares the first')
    a = p.parse_args(argv)
    config = load(a.config, strict=a.mode != 'show-config')
    if a.mode == 'show-config':
        print(json.dumps(config,indent=2)); return 0
    if not a.navigation_repo:
        p.error('--navigation-repo or RYNNBRAIN_NAV_DIR is required')
    root = artifact_root(a.output_dir)
    env = environment(root, dict(GPU_DIRECT_RUN='1',
        ACTION_HEAD_RENDER_REPO=str(REPO/'third_party/abot_renderer'),
        ACTION_HEAD_NAVIGATION_REPO=a.navigation_repo, RYNNBRAIN_NAV_DIR=a.navigation_repo))
    os.environ.update(env)
    os.chdir(REPO)
    from unified_closedloop_eval import prepare
    if a.mode in ('prepare','console','smoke'):
        root.mkdir(parents=True,exist_ok=False)
        save(root/'selection.json',config)
        if a.mode == 'prepare':
            prepare(root/'selection.json',root/'run','full')
            return 0
        if a.mode == 'smoke':
            try:
                devices = [int(x.strip()) for x in a.smoke_gpus.split(',')]
                assert len(devices)==2 and len(set(devices))==2 and min(devices)>=0
            except (ValueError,AssertionError):
                p.error('--smoke-gpus must name two distinct nonnegative device indices')
            from unified_closedloop_eval import verify_dependencies
            from unified_closedloop_console import prepare_console
            verify_dependencies(config['benchmarks'])
            plan = prepare_console(root/'selection.json',root/'smoke',a.port,smoke=True,smoke_gpus=devices)
            command=[sys.executable,'-B',str(REPO/'scripts/train/supervise_rule_teacher_collection.py'),
                     '--plan',str(plan)]
        else:
            save(root/'dependencies.json',dict(navigation_repo=a.navigation_repo))
            command=[sys.executable,'-B',str(REPO/'scripts/eval/unified_closedloop_eval.py'),
                'console','--config',str(root/'selection.json'),'--dependency-spec',str(root/'dependencies.json'),
                '--output',str(root/'console'),'--port',str(a.port)]
    else:
        if not 0 <= a.rank < config['nodes']:
            p.error('rank is outside selection.nodes')
        saved = load(root/'selection.json')
        if config != saved:
            p.error('run config differs from prepared selection')
        command=[sys.executable,'-B',str(REPO/'scripts/eval/unified_closedloop_eval.py'),
            'run','--run',str(root/'run/run.json'),'--rank',str(a.rank)]
    return execute(command,env,root/'launch'/f'{a.mode}-node{a.rank}.json')


if __name__ == '__main__':
    sys.exit(main())
