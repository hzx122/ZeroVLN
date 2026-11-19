"""Bind unified watchers to existing idle DLC residents, one per node."""
import argparse
import fcntl
import importlib
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'scripts/train'), str(REPO/'scripts/eval')]
from rule_teacher_training_handoff import load, save
from watch_v6_rl_local_credit import process_identity, same_process_alive


def resident_binding(rows, actual, nodes):
    if (len(rows) != nodes
            or any(row['git'] != actual or row['rank'] != i for i,row in enumerate(rows))
            or len({row['owner']['hostname'] for row in rows}) != nodes):
        raise ValueError('resident node Git/rank/hostname differs')
    return [row['owner'] for row in rows]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--debug-output', type=Path, required=True)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--retire-config', type=Path)
    p.add_argument('--retire-rank', type=int)
    p.add_argument('--drain-step', type=int)
    args = p.parse_args()
    from unified_closedloop_eval import git_runtime, prepare_watch, watch
    from unified_companion_task import dependencies
    spec = load(args.spec)
    actual = git_runtime(require_container=True)
    if actual['commit'] != spec['commit'] or not 0 <= args.rank < spec['nodes']:
        raise ValueError('companion Git/spec/rank mismatch')
    if args.retire_config:
        if args.drain_step is None or args.drain_step < 0:
            raise ValueError('layout replacement requires an explicit checkpoint drain boundary')
        previous = load(args.retire_config)
        previous_rank = args.rank if args.retire_rank is None else args.retire_rank
        if not 0 <= previous_rank < len(previous['resident_owners']):
            raise ValueError('previous companion rank is outside its owner set')
        if previous['resident_owners'][previous_rank]['hostname'] != os.uname().nodename:
            raise ValueError('previous companion node ownership differs')
        previous_start = load(Path(previous['root'])/f'watchers/node{previous_rank}/start.json')
        summary = Path(previous['root'])/'repeat-summary'/f'global_step_{args.drain_step:06d}.json'
        from dlc_control_io_retry import bounded, retry_estale
        deadline = time.monotonic() + 4 * 3600
        while True:
            with bounded(time.monotonic()+15):
                available = retry_estale(summary.exists, str(summary))
                result = load(summary) if available else None
            if result is not None:
                if result.get('evaluation_valid') is not True or result.get('step') != args.drain_step:
                    raise ValueError('checkpoint drain boundary has an invalid summary')
                break
            if not same_process_alive(previous_start['identity']):
                raise RuntimeError('previous companion exited before completing the drain checkpoint')
            if time.monotonic() >= deadline:
                raise TimeoutError('previous checkpoint did not finish before the layout replacement deadline')
            print(dict(event='waiting_previous_checkpoint_before_layout_switch',
                rank=args.rank, step=args.drain_step, business_progress_reported=False), flush=True)
            time.sleep(20)
        from rule_teacher_openloop_companion import retire_evaluation_owner
        retire_evaluation_owner(args.retire_config, previous_rank,
            Path(spec['root'])/'platform'/f'previous-owner-retired-node{previous_rank}.json',
            reason=f'user requested mature evaluation layout; checkpoint {args.drain_step} completed')
    state = load(args.debug_output/'debug-status.json')
    owner = dict(hostname=os.uname().nodename, debug_output=str(args.debug_output),
        debug_control=str(REPO/'scripts/train/run_dlc_debug_reservation.py'),
        controller_identity=process_identity(state['pid']))
    if (state['hostname'] != owner['hostname'] or state['mode'] != 'idle'
            or state['transition'] != 'ready' or time.time()-state['heartbeat'] > 45
            or not same_process_alive(owner['controller_identity'])):
        raise ValueError('resident is not freshly verified idle/ready')
    from rule_teacher_openloop_companion import waiting_reservation
    waiting_reservation(owner)
    forbidden = {'rule_teacher_openloop_companion.py', 'rule_teacher_repeat_companion.py',
        'supervise_rule_teacher_collection.py', 'v6_rl_dlc_worker.py',
        'run_rule_teacher_repair_owner.py'}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc/'cmdline').read_bytes().split(b'\0')
            if any(Path(x.decode(errors='replace')).name in forbidden for x in argv if x):
                raise ValueError('existing business owner: '+proc.name)
            if any(x.endswith(b'/unified_closedloop_eval.py') for x in argv) and b'watch' in argv:
                raise ValueError('existing unified watcher: '+proc.name)
        except (FileNotFoundError, ProcessLookupError):
            pass
    root = Path(spec['root'])
    if not root.is_relative_to('/data/zerovln/artifacts'):
        raise ValueError('companion output must be on the authorized data disk')
    if args.rank == 0 and (root/'run').exists():
        raise ValueError('new output required; preserve prior owner receipts')
    node = root/'bootstrap'/f'node{args.rank}'
    node.mkdir(parents=True, exist_ok=True)
    lock = (node/'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (node/'identity.json').exists():
        raise ValueError('existing resident attempt must not be replayed')
    dependencies(spec)
    modules = {name: importlib.import_module(name).__file__ for name in (
        'unified_closedloop_eval', 'closedloop_evaluation_contract', 'rule_teacher_companion_repair',
        'action_head.evaluation_context', 'action_head.context_compaction', 'action_head.protocol_v8',
        'action_head_rl.rollout', 'action_head.runtime', 'vln_gym.env',
        'action_head.serve.native_r3_capture', 'action_head.serve.continuous_context')}
    if not all(Path(path).resolve().is_relative_to(REPO) for path in modules.values()):
        raise ValueError('companion module source escaped the container checkout')
    record = dict(git=actual, owner=owner, rank=args.rank,
        identity=process_identity(os.getpid()), cwd=os.getcwd(), modules=modules, time=time.time())
    save(node/'identity.json', record)
    save(root/'launch'/f'resident-identity-node{args.rank}.json', record)
    if spec['nodes'] == 1:
        save(root/'launch/resident-identity.json', record)
    from dlc_control_io_retry import retry_estale
    deadline = time.monotonic()+1800
    ready = [root/'bootstrap'/f'node{i}/identity.json' for i in range(spec['nodes'])]
    while not all(retry_estale(path.exists, str(path)) for path in ready):
        if time.monotonic() >= deadline:
            raise TimeoutError('not all resident nodes verified the local checkout')
        time.sleep(2)
    if spec.get('variants'):
        from unified_companion_queue import run_queue
        rows = [load(path) for path in ready]
        owners = resident_binding(rows, actual, spec['nodes'])
        code = run_queue(args.spec, spec, args.rank, owners)
        save(node/'business-exit.json', dict(returncode=code, time=time.time()))
        return code
    if args.rank == 0:
        rows = [load(path) for path in ready]
        binding = load(spec['binding'])
        binding['resident_owners'] = resident_binding(rows, actual, spec['nodes'])
        save(root/'resolved-binding.json', binding)
        prepare_watch(Path(spec['selection']), root/'resolved-binding.json', root/'run', spec['purpose'])
    config = root/'run/watcher-config.json'
    while not retry_estale(config.exists, str(config)):
        if time.monotonic() >= deadline:
            raise TimeoutError('resident companion plan preparation timed out')
        time.sleep(2)
    code = watch(config, args.rank) or 0
    save(node/'business-exit.json', dict(returncode=code, time=time.time()))
    if spec['nodes'] == 1:
        save(root/'launch/business-exit.json', dict(returncode=code, time=time.time()))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
