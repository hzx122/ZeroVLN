"""One-shot recovery of failed companions, retaining old attempts and step zero.

Only control/bootstrap is orchestrated here. All navigation and scoring reuse
start_unified_companion_resident and the unified watcher from pinned Git.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'scripts/train'),str(REPO/'scripts/eval')]
import align_three_source_observations as io
import watch_rl_pair_completion as lifecycle
from watch_v6_rl_local_credit import process_identity,same_process_alive
from prepare_git_workspace import entry
from resident_pair_handoff import dispatch,new_node_started,bound


def valid_anchor(item):
    if item.get('bootstrap_failure'):
        from resume_rl_companion import validate_bootstrap_failure
        return validate_bootstrap_failure(item)
    if item.get('authorized_handoff'):
        # Explicit repair authorization also covers a partially evaluated
        # checkpoint/full population. Actual process ownership and cleanup
        # are checked by retirement; only valid cases are seeded afterwards.
        # Historical spec/config hashes are provenance, not recovery gates.
        return True
    root=Path(item['old_root'])
    if item.get('resumed_failure'):
        # A failed full evaluation can start at any checkpoint and have only
        # partial valid cases. Step-zero coverage is not a recovery prerequisite.
        failed=Path(item['resumed_failure']['path'])
        if not failed.is_relative_to(root/'run/watchers'):
            raise ValueError('failure receipt is outside the previous attempt')
        if not io.load(failed).get('error'):
            raise ValueError('resumed companion lacks original failure')
        return True
    anchor_step = (io.load(root/'run/watcher-config.json').get('steps', [0])[0]
                   if item.get('authorized_handoff') else 0)
    if type(anchor_step) is not int or anchor_step < 0:
        raise ValueError('invalid authorized continuation anchor')
    path=root/'run/repeat-summary'/f'global_step_{anchor_step:06d}.json'
    if not io.exists(path):return False
    d=io.load(path)
    if d.get('step')!=anchor_step or d.get('evaluation_valid') is not True or d.get('repeats')!=1:
        raise ValueError('old step zero is invalid')
    for b in ['sage-3d','r2r','rxr']:
        m=d['benchmarks'][b]
        if m.get('cases')!=100 or m.get('valid_attempts')!=100 or m.get('evaluation_valid') is not True:
            raise ValueError('old step zero coverage differs')
    failure=root/'run/watchers/node0/failure.json'
    if not io.exists(failure):
        if not item.get('handoff_after_step_zero'):return False
        # Explicit preventive migration: the complete valid anchor and the
        # known legacy spec must both match before waiting on owned idle GPUs.
        spec=io.load(root/'spec.json')
        if spec['commit']!=item.get('legacy_validator_commit'):
            raise ValueError('preventive handoff legacy version differs')
        return True
    f=io.load(failure)
    if f.get('completed_steps')!=[0] or 'RL checkpoint report is not complete' not in f.get('error',''):
        raise ValueError('unexpected old failure; do not replace automatically')
    return True


def suspend_retired_owner(identity):
    """An already-cleaned launcher may exit between inspection and pidfd_open."""
    fd = None
    try:
        fd = os.pidfd_open(identity['pid'])
        if not same_process_alive(identity):
            return False
        signal.pidfd_send_signal(fd, signal.SIGSTOP)
    except ProcessLookupError:
        if same_process_alive(identity):
            raise
        return False
    finally:
        if fd is not None:
            os.close(fd)
    deadline = time.monotonic() + 5
    while True:
        actual = process_identity(identity['pid'])
        if actual is None or actual['start_ticks'] != identity['start_ticks']:
            return False
        if actual['state'] in ('T', 't'):
            return True
        if time.monotonic() > deadline:
            raise TimeoutError('old owner did not stop')
        time.sleep(.05)


def quiesce(item,rank):
    """Suspend only the bound idle old owner; retain its controller and shell.

    The legacy resident-watch launcher owns its controller and falls back to
    another controller if it exits. Keep that launcher stopped until DLC exit,
    avoiding a duplicate holder. Its independent parent stdout heartbeat stays.
    """
    root=Path(item['root']);old=Path(item['old_root'])
    config_path=old/'run/watcher-config.json';config=io.load(config_path)
    owner=config['resident_owners'][rank]
    start=io.load(old/f'run/watchers/node{rank}/start.json')
    identity=start['identity']
    if start['config']!=bound(config_path) or owner['hostname']!=os.uname().nodename:
        raise ValueError('old watcher binding differs')
    from rule_teacher_openloop_companion import waiting_reservation
    waiting_reservation(owner)
    if not same_process_alive(identity):
        io.write_json(root/'platform'/f'old-owner-node{rank}.json',dict(already_exited=True,identity=identity));return owner
    argv=(Path('/proc')/str(identity['pid'])/'cmdline').read_bytes().split(b'\0')
    if str(old/'spec.json').encode() not in argv or b'resident-watch' not in argv:
        raise ValueError('old process is not the bound resident launcher')
    # No evaluation service may remain a live child; only its existing sleep
    # and reservation controller are allowed. GPU occupation remains intact.
    state=io.load(Path(owner['debug_output'])/'debug-status.json')
    children=set()
    for task in (Path('/proc')/str(identity['pid'])/'task').iterdir():
        children.update(map(int,(task/'children').read_text().split()))
    allowed={state['pid'],state['sleep_pid']}
    if children-allowed:raise ValueError('old owner still has business children')
    receipt=root/'platform'/f'old-owner-node{rank}.json'
    if io.exists(receipt):raise ValueError('quiesce already attempted; inspect receipt')
    io.write_json(receipt.with_suffix('.intent.json'),dict(identity=identity,config=bound(config_path),owner=owner,time=time.time()))
    stopped=suspend_retired_owner(identity)
    waiting_reservation(owner)
    io.write_json(receipt,dict(identity=identity,old_owner_stopped=stopped,
        already_exited=not stopped,controller_retained=True,owner=owner,time=time.time()))
    return owner


def node(item_path,rank):
    item=io.load(item_path);root=Path(item['root'])
    from unified_closedloop_eval import git_runtime
    if git_runtime(True)['commit']!=item['commit'] or not valid_anchor(item):raise ValueError('source/old attempt differs')
    import action_head_benchmark_suite as suite
    source=Path(suite.checkpoint_identity.__code__.co_filename).resolve()
    if not source.is_relative_to(REPO):raise ValueError('checkpoint validator escaped local checkout')
    binding=io.load(root/'binding.json')
    ckpt=Path(binding['checkpoint_root'])/'global_step_000002'
    if item.get('handoff_after_step_zero') and not io.exists(ckpt/'report.json'):
        # Model publication may lag step-zero evaluation. Validate its already
        # completed initialization while retaining the normal tracker gate for2.
        ckpt=Path(binding['checkpoint_paths']['0'])
    proof=suite.checkpoint_identity(ckpt)
    io.write_json(root/'platform'/f'checkpoint-preflight-node{rank}.json',dict(checkpoint=proof,validator=str(source),commit=item['commit'],cwd=str(REPO),time=time.time()))
    owner=quiesce(item,rank)
    command=[sys.executable,'-B',str(REPO/'scripts/eval/start_unified_companion_resident.py'),
        '--spec',str(root/'spec.json'),'--rank',str(rank),'--debug-output',owner['debug_output']]
    subprocess.run(command,check=True,cwd=REPO)


def launch_node(item,item_path,rank):
    root=Path(item['root']);intent=root/'platform'/f'dispatch-intent-node{rank}.json'
    with intent.open('x') as stream:json.dump(dict(rank=rank,time=time.time(),automatic_retry=False),stream)
    prefix,sep,_=entry(item['repository'],item['commit'],root).partition('sleep 99999999 &\n')
    if not sep:raise ValueError('Git bootstrap boundary differs')
    prefix=prefix.replace('set -uo pipefail\n','set -uo pipefail\nexport LIBRARY_PATH=${LIBRARY_PATH:-}\nsource /usr/local/PPU_SDK/envsetup.sh cuda\n',1)
    local=Path('/root/rl-code')/root.name/'repo'
    body=prefix+'export ACTION_HEAD_TEACHER_STOP_TRAINING=0\n'+shlex.join([
        '/usr/local/bin/python3.12','-B',str(local/'scripts/eval/recover_rl_companions.py'),
        '--mode','node','--plan',str(item_path),'--rank',str(rank)])+'\n'
    command=root/'platform'/f'bootstrap-node{rank}.sh';io.write_bytes(command,body.encode())
    subprocess.run(['bash','-n',str(command)],check=True)
    code="""import subprocess,json
from pathlib import Path
with Path(%r).open('x') as receipt:
 with open('/proc/1/fd/1','w',buffering=1) as raw:
  child=subprocess.Popen(['bash',%r],stdin=subprocess.DEVNULL,stdout=raw,stderr=subprocess.STDOUT,start_new_session=True)
 ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
 json.dump(dict(pid=child.pid,start_ticks=ticks),receipt)
print(json.dumps(dict(pid=child.pid)))
"""%(str(root/'platform'/f'dispatch-node{rank}.json'),str(command))
    result=subprocess.run([lifecycle.CLI,'exec','--job-id',item['job_id'],'--workspace',item['space_id'],
        '--pod',item['residents'][rank]['hostname'],'--ws-proxy-env','--timeout','45','--','python3','-c',code],capture_output=True,text=True,timeout=55)
    io.write_json(root/'platform'/f'exec-node{rank}.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,automatic_retry=False))


def poll(item,path):
    root=Path(item['root'])
    if lifecycle.evaluation_complete(item):
        if item.get('old_root') and not valid_anchor(item):raise ValueError('old valid anchor changed')
        detail=lifecycle.checked_detail(item)
        if detail['response']['data']['status'] in lifecycle.TERMINAL:
            io.write_json(root/'platform/terminal.json',detail);return 'terminal_verified'
        if not io.exists(root/'platform/stop-intent.json'):
            io.write_json(root/'platform/stop-intent.json',dict(time=time.time(),task_id=item['task_id']))
            response=lifecycle.call_cli(['stop','--json',json.dumps({'taskIds':[item['task_id']]})])
            io.write_json(root/'platform/stop-response.json',response)
        return 'success_stop_pending_terminal'
    if item.get('handoff_only'):return 'waiting_scheduled_handoff_and_completion'
    if all(new_node_started({'evaluation':item},'evaluation',r) for r in range(4)):return 'new_watchers_started'
    if any(io.exists(root/'platform'/f'dispatch-intent-node{r}.json') for r in range(4)):
        return 'dispatch_entered_no_automatic_replay'
    if not valid_anchor(item):return 'waiting_old_step_zero_and_expected_failure'
    old=io.load(Path(item['old_root'])/'run/watcher-config.json')
    for owner in old['resident_owners']:
        s=io.load(Path(owner['debug_output'])/'debug-status.json')
        if s.get('mode')!='idle' or s.get('transition')!='ready' or time.time()-s['heartbeat']>45:return 'waiting_owned_idle'
    detail=lifecycle.checked_detail(item)['response']['data']
    if detail['status']!='Running' or detail.get('restartRecord') or detail['podCount']!=4:raise ValueError('resident DLC changed')
    item=dict(item,residents=[dict(rank=i,hostname=o['hostname']) for i,o in enumerate(old['resident_owners'])])
    item_path=root/'platform/dispatch-plan.json';io.write_json(item_path,item)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda r:launch_node(item,item_path,r),range(4)))
    return 'dispatched_once'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['watch','node'],required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--rank',type=int);args=p.parse_args()
    plan=io.load(args.plan)
    if args.mode=='node':return node(args.plan,args.rank)
    directory=Path(plan['root'])/'supervisor';directory.mkdir(exist_ok=True)
    with (directory/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        io.write_json(directory/'owner.json',dict(identity=process_identity(os.getpid()),plan=bound(args.plan),time=time.time()))
        while True:
            states={}
            for name,item in plan['items'].items():
                try:states[name]=poll(item,args.plan)
                except (OSError,ValueError,KeyError,RuntimeError,subprocess.SubprocessError) as e:states[name]=dict(error=repr(e),automatic_retry=False)
            io.write_json(directory/'status.json',dict(items=states,heartbeat=time.time()))
            print(json.dumps(states),flush=True)
            if all(v=='terminal_verified' for v in states.values()):return
            time.sleep(30)

if __name__=='__main__':main()
