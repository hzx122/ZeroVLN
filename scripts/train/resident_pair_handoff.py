"""One-shot handoff after exact old evaluation queues finish and reserve GPUs.

The independent DSW supervisor only dispatches pinned container-local entries.
It never retires active evaluations, signals unknown GPU PIDs or retries a
possibly started business. Per-container entry revalidates process ownership.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import align_three_source_observations as io
from watch_v6_rl_local_credit import process_identity, same_process_alive
from prepare_git_workspace import entry
import watch_rl_pair_completion as lifecycle

REPO=Path(__file__).resolve().parents[2]


def bound(path):
    return dict(path=str(path),sha256=io.digest(io.read_bytes(path)))


def queue_finished(binding):
    """Idle between cases/checkpoints, missing state and failure are not release."""
    path=Path(binding['path']);root=path.parent
    if bound(path)!=binding:raise ValueError('old queue specification changed: '+str(path))
    if not io.exists(root/'queue.json'):return False
    queue=io.load(root/'queue.json')
    if queue['spec']!=binding:raise ValueError('old queue has another spec binding')
    jobs=queue['jobs'];expected=[{k:j[k] for k in ('variant','step')} for j in jobs]
    if not jobs:raise ValueError('empty evaluation queue')
    for rank in range(4):
        completion=root/f'queue-nodes/{rank}/completion.json'
        exited=root/f'bootstrap/node{rank}/business-exit.json'
        if not io.exists(completion) or not io.exists(exited):return False
        c=io.load(completion);e=io.load(exited)
        if c.get('evaluation_valid') is not True or c.get('completed')!=expected or e.get('returncode')!=0:
            raise ValueError('previous evaluation did not finish successfully: '+str(root))
    for job in jobs:
        summary=Path(job['config']).parent/'repeat-summary'/f"global_step_{job['step']:06d}.json"
        if not io.exists(summary):return False
        s=io.load(summary)
        if s.get('evaluation_valid') is not True or s.get('step')!=job['step']:
            raise ValueError('old checkpoint summary is invalid')
    return True


def old_owner(node):
    path=Path(node['old_queue'])/f"bootstrap/node{node['old_rank']}/identity.json"
    record=io.load(path)
    owner=record['owner']
    if (record['rank']!=node['old_rank'] or owner['hostname']!=node['hostname']
            or owner['debug_output']!=node['debug_output']
            or owner['controller_identity']['pid']!=node['controller_identity']['pid']
            or owner['controller_identity']['start_ticks']!=node['controller_identity']['start_ticks']):
        raise ValueError('old queue owner/controller binding differs')
    return record


def idle_snapshot(node):
    """Disk evidence is only the scheduling gate; live checks follow in Pod."""
    record=old_owner(node)
    state=io.load(Path(node['debug_output'])/'debug-status.json')
    if (state.get('hostname')!=node['hostname'] or state.get('pid')!=node['controller_identity']['pid']):
        raise ValueError('resident controller changed')
    if (state.get('mode')!='idle' or state.get('transition')!='ready'
            or time.time()-state.get('heartbeat',0)>45 or not state.get('occupier_pid')):
        return False
    ready_path=Path(node['debug_output'])/'waiting_gpu_ready.json'
    if not io.exists(ready_path):return False
    ready=io.load(ready_path)
    return ready.get('pid')==state['occupier_pid'] and ready.get('gpus')==list(range(16))


def ready(plan,kind):
    item=plan[kind]
    if not all(queue_finished(q) for q in item['old_queues']):return False
    return all(idle_snapshot(node) for node in item['residents'])


def bootstrap(plan,path,kind,rank):
    root=Path(plan[kind]['root']);local=Path('/root/rl-code')/root.name
    prefix,separator,_=entry(plan['repository'],plan[kind]['commit'],root).partition('sleep 99999999 &\n')
    if not separator:raise ValueError('Git bootstrap boundary changed')
    prefix=prefix.replace('set -uo pipefail\n', 'set -uo pipefail\nexport LIBRARY_PATH=${LIBRARY_PATH:-}\nsource /usr/local/PPU_SDK/envsetup.sh cuda\n',1)
    command=['/usr/local/bin/python3.12','-B',str(local/'repo/scripts/train/resident_pair_handoff.py'),
        '--mode','node','--plan',str(path),'--kind',kind,'--rank',str(rank)]
    return prefix+'''export NCCL_IB_DISABLE=0 NCCL_NET=IB NCCL_NET_PLUGIN=none
export NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4
unset NCCL_PF_HCA
export ACTION_HEAD_TEACHER_STOP_TRAINING=0
'''+shlex.join(command)+'\n'


def dispatch(plan,path,kind,rank):
    item=plan[kind];root=Path(item['root']);node=item['residents'][rank]
    directory=Path(plan['training']['root'])/'platform/handoff'
    io.bounded(lambda: directory.mkdir(parents=True, exist_ok=True), 'handoff_directory:'+str(directory))
    intent=directory/f'{kind}-node{rank}-intent.json'
    if io.exists(intent):return  # Ambiguous exec is never replayed.
    body=bootstrap(plan,path,kind,rank)
    command_path=root/'platform'/f'bootstrap-node{rank}.sh'
    io.write_bytes(command_path,body.encode())
    subprocess.run(['bash','-n',str(command_path)],check=True)
    # Detached launch-control script, not a source-code snapshot. All business
    # functions are fetched from Git by the verified bootstrap above.
    code='''import json,os,subprocess,time
from pathlib import Path
receipt=Path(%r)
with receipt.open('x') as stream:
 with open('/proc/1/fd/1','w',buffering=1) as raw:
  child=subprocess.Popen(['bash',%r],stdin=subprocess.DEVNULL,stdout=raw,stderr=subprocess.STDOUT,start_new_session=True)
 ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
 json.dump(dict(pid=child.pid,start_ticks=ticks,time=time.time()),stream)
print(json.dumps(dict(dispatched=True,pid=child.pid)))
''' % (str(root/'platform'/f'dispatch-node{rank}.json'),str(command_path))
    io.write_json(intent,dict(kind=kind,rank=rank,time=time.time(),node=node,automatic_retry=False))
    try:
        p=subprocess.run([lifecycle.CLI,'exec','--job-id',item['job_id'],'--workspace',item['space_id'],
            '--pod',node['hostname'],'--ws-proxy-env','--timeout','45','--','python3','-c',code],
            capture_output=True,text=True,timeout=55)
        io.write_json(directory/f'{kind}-node{rank}-exec.json',dict(returncode=p.returncode,
            stdout=p.stdout,stderr=p.stderr,time=time.time(),automatic_retry=False))
    except subprocess.TimeoutExpired as exc:
        io.write_json(directory/f'{kind}-node{rank}-exec.json',dict(error=str(exc),
            dispatch_may_have_started=True,automatic_retry=False,time=time.time()))


def new_node_started(plan,kind,rank):
    root=Path(plan[kind]['root'])
    path=(root/'launch'/f'owner_{rank:02d}.json' if kind=='training'
          else root/f'bootstrap/node{rank}/identity.json')
    if not io.exists(path):return False
    state=io.load(path)
    if kind=='training':
        if state['commit']!=plan[kind]['commit']:raise ValueError('new training source differs')
    elif state['git']['commit']!=plan[kind]['commit']:raise ValueError('new evaluation source differs')
    return True


def poll(plan,path):
    states={}
    for kind in ('training','evaluation'):
        if all(new_node_started(plan,kind,r) for r in range(plan[kind]['nodes'])):
            states[kind]='dispatched_new_business';continue
        directory=Path(plan['training']['root'])/'platform/handoff'
        if any(io.exists(directory/f'{kind}-node{r}-intent.json') for r in range(plan[kind]['nodes'])):
            states[kind]='handoff_entered_no_automatic_replay';continue
        if not ready(plan,kind):
            states[kind]='waiting_old_queue_success_and_owned_gpu_reservation';continue
        detail=lifecycle.checked_detail(plan[kind])['response']['data']
        if detail['status']!='Running' or detail.get('restartRecord') or detail['podCount']!=plan[kind]['nodes']:
            raise ValueError('resident DLC state changed; no handoff')
        with ThreadPoolExecutor(max_workers=plan[kind]['nodes']) as pool:
            list(pool.map(lambda r:dispatch(plan,path,kind,r),range(plan[kind]['nodes'])))
        states[kind]='handoff_dispatched_once'
    return states


def node_main(plan,path,kind,rank,probe=False):
    item=plan[kind];node=item['residents'][rank]
    local=Path('/root/rl-code')/Path(item['root']).name
    if (REPO!=local/'repo' or Path.cwd()!=REPO or os.uname().nodename!=node['hostname']
            or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=item['commit']):
        raise ValueError('new node source/cwd/hostname differs')
    if not all(queue_finished(q) for q in item['old_queues']) or not idle_snapshot(node):
        raise ValueError('old queues not complete or this node is not idle')
    record=old_owner(node)
    if same_process_alive(record['identity']):raise ValueError('old evaluation process is still alive')
    if not same_process_alive(node['controller_identity']):raise ValueError('resident controller PID/start changed')
    from rule_teacher_openloop_companion import waiting_reservation
    resident=waiting_reservation(record['owner'])
    if probe:
        print(json.dumps(dict(probe=True,owned_idle=True,old_evaluation_dead=True,rank=rank)));return
    if kind=='training':
        command=[sys.executable,'-B',str(REPO/'scripts/train/start_megatron_history_resident.py'),
            '--previous-root',plan['previous_training_root'],'--output',item['root'],
            '--root',str(local),'--environment',str(local/'environment-local'),
            '--resident-node',node['debug_output'],'--node-rank',str(rank),'--commit',item['commit'],
            '--from-resident-standby','--detach','--task-spec',str(Path(item['root'])/'launch/git_task.json')]
    else:
        from verify_idle_gpu_ownership import verify
        verify(Path(node['debug_output']), REPO/'scripts/train/run_dlc_debug_reservation.py',
            Path(item['root'])/'platform'/f'idle-ownership-node{rank}.json')
        command=[sys.executable,'-B',str(REPO/'scripts/eval/start_unified_companion_resident.py'),
            '--spec',str(Path(item['root'])/'spec.json'),'--rank',str(rank),'--debug-output',node['debug_output']]
    io.write_json(Path(item['root'])/'platform'/f'node{rank}-verified-before-handoff.json',
        dict(rank=rank,hostname=os.uname().nodename,commit=item['commit'],cwd=str(REPO),
             controller=node['controller_identity'],old_evaluation_dead=True,resident=resident,time=time.time()))
    subprocess.run(command,check=True,cwd=REPO)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['watch','check','node'],required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--kind',choices=['training','evaluation'])
    p.add_argument('--rank',type=int)
    p.add_argument('--probe',action='store_true')
    args=p.parse_args();plan=io.load(args.plan)
    if args.mode=='node':return node_main(plan,args.plan,args.kind,args.rank,args.probe)
    directory=Path(plan['training']['root'])/'platform/handoff';directory.mkdir(parents=True,exist_ok=True)
    if args.mode=='check':
        print(json.dumps({k:dict(ready=ready(plan,k)) for k in ('training','evaluation')}));return
    with (directory/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        io.write_json(directory/'owner.json',dict(identity=process_identity(os.getpid()),
            plan=bound(args.plan),source=str(Path(__file__).resolve()),time=time.time()))
        while True:
            try:
                state=poll(plan,args.plan)
            except (OSError,ValueError,KeyError,RuntimeError,subprocess.SubprocessError) as exc:
                state=dict(error=repr(exc),automatic_retry=False)
            state['heartbeat']=time.time();io.write_json(directory/'status.json',state)
            print(json.dumps(state),flush=True)
            if all(state.get(k)=='dispatched_new_business' for k in ('training','evaluation')):
                io.write_json(directory/'completion.json',dict(dispatched=True,**state));return
            time.sleep(30)

if __name__=='__main__':main()
