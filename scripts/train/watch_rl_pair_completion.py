"""Stop only this submitted RL/evaluation pair after verified successful work.

Run as an independent workstation supervisor. Failed jobs retain their original
GPU standby; missing/invalid state never authorizes a platform stop.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import align_three_source_observations as io
from watch_v6_rl_local_credit import process_identity

CLI = '/workspace/bin/train-platform'
RESIDENT_JOBS = {'job-anonymous', 'job-anonymous'}
DONE_STATES = {'terminal_verified', 'resident_retained_verified'}
TERMINAL = {'Stopped', 'Failed', 'Succeeded', 'Completed', 'Cancelled'}


def read_if_present(path):
    return io.load(path) if io.exists(path) else None


def training_complete(item):
    root = Path(item['root'])
    for rank in range(item['nodes']):
        node = root/'git_runtime'/f'node_{rank:02d}'
        receipt = read_if_present(node/'business_exit.json')
        if not receipt:
            return False
        checkout = io.load(node/'checkout.json')
        if (receipt.get('rank') != rank or receipt.get('phase') != 'succeeded'
                or receipt.get('business_returncode') != 0
                or checkout['code']['commit'] != item['commit']):
            raise ValueError('training receipt failed or belongs to another version')
    marker = root/'coordination/training_done.json'
    if not io.exists(marker) or io.load(marker).get('status') != 'succeeded':
        return False
    step = item['final_step']
    if int(io.read_bytes(root/'train/latest_checkpointed_iteration.txt')) < step:
        return False
    report = io.load(root/'train'/f'global_step_{step:06d}'/'report.json')
    if report.get('step') != step or report.get('qwen_exported') is not True:
        raise ValueError('final checkpoint export is inconsistent')
    return True


def evaluation_complete(item):
    root = Path(item['root'])/'run'
    config = read_if_present(root/'watcher-config.json')
    if config is None:
        return False
    repeats = item.get('evaluation_repeats', 1)
    counts = item.get('case_counts', dict.fromkeys(('sage-3d', 'r2r', 'rxr'), 100))
    if config['steps'] != item['steps'] or config['evaluation_repeats'] != repeats:
        raise ValueError('evaluation schedule differs')
    base = io.load(Path(config['base_plan']))
    if base['source_provenance']['git_runtime']['commit'] != item['commit']:
        raise ValueError('evaluation version differs')
    for rank in range(item['nodes']):
        result = read_if_present(root/f'watchers/node{rank}/completion.json')
        if not result:
            return False
        if (result.get('complete') is not True or result.get('evaluation_valid') is not True
                or result.get('completed_steps') != item['steps']):
            raise ValueError('evaluation did not finish all valid steps')
    for step in item['steps']:
        result = read_if_present(root/'repeat-summary'/f'global_step_{step:06d}.json')
        if not result:
            return False
        if (result.get('evaluation_valid') is not True or result.get('step') != step
                or result.get('repeats') != repeats
                or set(result.get('benchmarks', {})) != {'sage-3d', 'r2r', 'rxr'}):
            raise ValueError('invalid evaluation summary')
        for benchmark, metrics in result['benchmarks'].items():
            if (metrics.get('evaluation_valid') is not True or metrics.get('cases') != counts[benchmark]
                    or metrics.get('valid_attempts') != counts[benchmark] * repeats
                    or (repeats == 1 and 'pass_at_4' in metrics)):
                raise ValueError('evaluation coverage/repeat count differs')
    return True


def call_cli(arguments):
    result = subprocess.run([CLI, *arguments], capture_output=True, text=True,
                            check=True, timeout=45)
    envelope = json.loads(result.stdout)
    if not 200 <= envelope['http_status'] < 300 or envelope['response']['code'] != 2000:
        raise RuntimeError('official platform request failed')
    return envelope


def checked_detail(item):
    envelope = call_cli(['detail', '--task-id', item['task_id']])
    detail = envelope['response']['data']
    for key, wanted in dict(taskId=item['task_id'], jobId=item['job_id'],
            createdBy='anonymous', quotaId=item.get('quota_id', 'quota-anonymous'), platType=2,
            parsedProductDir=item['product_dir'], name=item['name']).items():
        if str(detail.get(key)) != str(wanted):
            raise ValueError('platform identity differs: '+key)
    if str(detail.get('spaceId', detail.get('quota', {}).get('spaceId'))) != str(item.get('space_id', '150120')):
        raise ValueError('platform workspace differs')
    return envelope


def poll(plan, state):
    for kind, checker in [('training', training_complete), ('evaluation', evaluation_complete)]:
        if kind not in plan:
            continue
        if state.get(kind) in DONE_STATES:
            continue
        item = plan[kind]
        if not checker(item):
            state[kind] = 'waiting_verified_completion'
            continue
        if item.get('job_id') in RESIDENT_JOBS:
            # These two user-designated resident jobs must never be stopped.
            # Their pinned node launchers restore the owned GPU reservation.
            root = Path(item['root'])/'platform/resident-retention'
            root.mkdir(parents=True, exist_ok=True)
            detail = checked_detail(item)
            io.write_json(root/'completion-policy.json',dict(time=time.time(),
                job_id=item['job_id'],policy='retain_DLC_restore_owned_GPU_reservation',
                platform_stop_allowed=False,detail=detail))
            state[kind] = 'resident_retained_verified'
            continue
        root = Path(item['root'])/'platform/auto-stop'
        root.mkdir(parents=True, exist_ok=True)
        detail = checked_detail(item)
        io.write_json(root/'detail-before.json', detail)
        if detail['response']['data']['status'] not in TERMINAL:
            if state.get(kind) == 'stop_accepted_pending_terminal':
                continue
            # Recheck after any previous timeout, before issuing a retry.
            io.write_json(root/'stop-intent.json', dict(time=time.time(), item=item))
            response = call_cli(['stop', '--json', json.dumps({'taskIds': [item['task_id']]})])
            io.write_json(root/'stop-response.json', response)
            state[kind] = 'stop_accepted_pending_terminal'
        else:
            io.write_json(root/'terminal.json', detail)
            state[kind] = 'terminal_verified'
    return all(state.get(kind) in DONE_STATES for kind in ('training', 'evaluation') if kind in plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = io.load(args.plan)
    if not any(kind in plan for kind in ('training', 'evaluation')):
        raise ValueError('completion plan requires a training or evaluation task')
    root = Path(plan.get('training', plan.get('evaluation'))['root'])/'platform/pair-supervisor'
    root.mkdir(parents=True, exist_ok=True)
    with (root/'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        io.write_json(root/'owner.json', dict(identity=process_identity(os.getpid()),
            plan=str(args.plan.resolve()), plan_sha256=io.digest(io.read_bytes(args.plan)),
            source=str(Path(__file__).resolve()), started=time.time()))
        state = {}
        while True:
            try:
                done = poll(plan, state)
                state.pop('error', None)
            except (OSError, ValueError, KeyError, subprocess.SubprocessError, RuntimeError) as exc:
                done = False
                state['error'] = repr(exc)
            state['heartbeat'] = time.time()
            io.write_json(root/'status.json', state)
            print(json.dumps(state), flush=True)
            if done:
                io.write_json(root/'completion.json', dict(complete=True, **state))
                return 0
            time.sleep(30)


if __name__ == '__main__':
    raise SystemExit(main())
