"""Replace RL business attempts without stopping/recreating their DLC jobs.

Composes existing Git bootstrap, verified worker signals, watcher retirement,
resident GPU handoff, and success-only DLC completion. Queued tasks keep their
original requests and are handed over only after their real owners exist.
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
import threading
import time

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'scripts/train'),str(REPO/'scripts/eval')]
import align_three_source_observations as io
import watch_rl_pair_completion as life
import resident_pair_handoff as handoff
from resident_rl_attempt import proc, exact_live, signal_pid
from prepare_git_workspace import entry
from run_git_task import checkout


def require(ok,message):
    if not ok:raise ValueError(message)


def wait_for(predicate,seconds,label):
    deadline=time.monotonic()+seconds
    while True:
        value=predicate()
        if value:return value
        if time.monotonic()>deadline:raise TimeoutError(label)
        time.sleep(2)


def node_paths(item,rank):
    root=Path(item['root'])/'platform/restart'
    root.mkdir(parents=True,exist_ok=True)
    return root,root/f'node{rank}'


def require_all(item,name):
    root,_=node_paths(item,0)
    return all(io.exists(root/f'node{rank}-{name}.json') for rank in range(item['nodes']))


def ranked_item(item,rank):
    return dict(item,old_root=item.get('old_roots',{}).get(str(rank),item['old_root']))


def old_training_ready(item):
    return all((io.exists(Path(ranked_item(item,r)['old_root'])/f'coordination/node_{r:02d}.json') or
                io.exists(Path(ranked_item(item,r)['old_root'])/f'git_runtime/node_{r:02d}/business_exit.json')) and
               io.exists(Path(ranked_item(item,r)['old_root'])/f'git_runtime/node_{r:02d}/checkout.json')
               for r in range(item['nodes']))


def old_evaluation_ready(item):
    old=Path(item['old_root'])
    return io.exists(old/'run/watcher-config.json') and all(
        io.exists(old/f'run/watchers/node{r}/start.json') for r in range(item['nodes']))


def training_identity(item,rank):
    old=Path(item['old_root']);node=old/f'coordination/node_{rank:02d}.json'
    state=io.load(node) if io.exists(node) else None
    proof=io.load(old/f'git_runtime/node_{rank:02d}/checkout.json')
    require(os.uname().nodename==proof['hostname'] and (state is None or state['hostname']==proof['hostname']),
            'old training hostname differs')
    worker=proc(state['controller']['pid']) if state else None;owner=proc(proof['pid'])
    require(owner and str(old) in ' '.join(owner['argv']) and
            any(x.endswith('/run_git_task.py') for x in owner['argv']),'old Git supervisor differs')
    if state is None:
        failed=io.load(old/f'git_runtime/node_{rank:02d}/business_exit.json')
        require(failed['phase']=='failed' and failed['business_pid'] is None and failed['pid']==owner['pid']
                and failed['hostname']==proof['hostname'],'no verified failure before business launch')
    if worker:
        require(worker['start_ticks']==str(state['controller']['start_ticks']) and
            worker['ppid']==owner['pid'] and
            str(Path(proof['code']['path'])/'scripts/train/v6_rl_dlc_worker.py') in worker['argv'],
            'old worker PID/parent/source differs')
    else:
        require(io.exists(old/f'git_runtime/node_{rank:02d}/business_exit.json'),
                'old worker vanished without an exit receipt')
    return worker,owner


def stop_training(item,rank):
    old=Path(item['old_root']);root,_=node_paths(item,rank)
    worker,owner=training_identity(item,rank)
    intent=root/f'node{rank}-stop-intent.json'
    require(not io.exists(intent),'training stop already attempted; inspect receipts')
    io.write_json(intent,dict(worker=worker,owner=owner,time=time.time(),reason='user requested new initialization'))
    state=io.load(old/f'coordination/node_{rank:02d}.json') if worker else None
    if worker and state['phase'] not in ('succeeded','failed'):
        signal_pid(worker,{1,os.getpid(),owner['pid']})
    def exited():
        path=old/f'git_runtime/node_{rank:02d}/business_exit.json'
        if not io.exists(path) or (worker and exact_live(worker)):return False
        result=io.load(path)
        require(result['hostname']==os.uname().nodename and result['rank']==rank,'old exit identity differs')
        ready=Path('/root/rl-code')/old.name/'standby_ready.json'
        if not io.exists(ready):return False
        holder=proc(io.load(ready)['pid'])
        return result if holder and holder['ppid']==owner['pid'] else False
    result=wait_for(exited,900,'old training cleanup/standby has not completed')
    io.write_json(root/f'node{rank}-stopped.json',dict(exit=result,time=time.time()))


def old_model_processes(roots,proc_root=Path('/proc')):
    """Only exact retired experiment identities, never host smi PID numbers."""
    roots={str(Path(x)):str(Path('/root/rl-code')/Path(x).name/'repo') for x in roots}
    result=[]
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():continue
        try:
            actual=proc(int(directory.name))
            if not actual:continue
            title=actual['argv'][0].strip()
            if title not in ('ray::SimpleStorageUnit','VLLM::EngineCore','VLLM::Worker_TP0','VLLM::Worker_TP1'):continue
            env=dict(x.split(b'=',1) for x in (directory/'environ').read_bytes().split(b'\0') if b'=' in x)
            output=env.get(b'OUTPUT_DIR',b'').decode()
            progress=env.get(b'ACTION_HEAD_LEARNER_PROGRESS_DIR',b'').decode()
            matches=[root for root,repo in roots.items() if env.get(b'REPO',b'').decode()==repo and
                     progress==root+'/train/learner_progress' and output in (root,root+'-controller') and
                     os.readlink(directory/'cwd')==repo]
            if matches:result.append(actual)
        except (FileNotFoundError,ProcessLookupError):continue
    return result


def clean_old_storage(item,rank,receipt,*,resume_unstarted=False):
    from start_megatron_history_resident import remaining_rl_ray_workers
    old=Path(item['old_root'])
    require(require_all(item,'stopped'),'all old workers must have completed cleanup')
    if resume_unstarted:
        from start_megatron_history_resident import validate_unstarted_handoff
        validate_unstarted_handoff(Path(item['root']),rank,item['commit'])
    require((resume_unstarted or not io.exists(Path(item['root'])/'launch'/f'intent_{rank:02d}.json')) and
            not io.exists(Path(item['root'])/'launch'/f'owner_{rank:02d}.json'),
            'business handoff has already begun; do not replay it')
    roots=list(dict.fromkeys([str(old),*item.get('cleanup_roots',[])]))
    for source in roots:
        nodepath=Path(source)/f'coordination/node_{rank:02d}.json'
        if io.exists(nodepath):
            node=io.load(nodepath)
            require(not exact_live(node['controller'],argv=False),'old worker is still alive')
    identities=old_model_processes(roots)
    allowed={x['pid'] for x in identities}
    require(all(x['pid'] in allowed for x in remaining_rl_ray_workers()),
            "remaining Ray process is not this run's orphan CPU storage")
    io.write_json(receipt.with_suffix('.intent.json'),dict(time=time.time(),identities=identities,roots=roots))
    for actual in identities:
        if exact_live(actual):signal_pid(actual,{1,os.getpid()})
    deadline=time.monotonic()+20
    while any(exact_live(p) for p in identities) and time.monotonic()<deadline:time.sleep(.5)
    for actual in identities:
        if exact_live(actual):
            fd=os.pidfd_open(actual['pid'])
            try:
                require(exact_live(actual),'orphan identity changed before KILL')
                signal.pidfd_send_signal(fd,signal.SIGKILL)
            finally:os.close(fd)
    wait_for(lambda:not old_model_processes(roots) and not remaining_rl_ray_workers(),20,'orphan model processes did not exit')
    io.write_json(receipt,dict(time=time.time(),identities=identities,remaining=[]))


def start_training(item,rank,*,recovery=False):
    local=Path('/root/rl-code')/Path(item['root']).name
    launcher=REPO/'scripts/train/start_megatron_history_resident.py' if recovery else local/'repo/scripts/train/start_megatron_history_resident.py'
    command=['/usr/local/bin/python3.12','-B',str(launcher),
        '--previous-root',item['old_root'],'--output',item['root'],'--root',str(local),
        '--environment',str(local/'environment-local'),'--node-rank',str(rank),
        '--commit',item['commit'],'--task-spec',str(Path(item['root'])/'launch/git_task.json')]
    if recovery:
        command+=['--launcher-commit',subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()]
        if io.exists(Path(item['root'])/'launch'/f'intent_{rank:02d}.json'):
            command+=['--resume-unstarted-handoff']
    env=dict(os.environ,NODE_RANK=str(rank),ACTION_HEAD_RESIDENT_ENTRY='1',OUTPUT_DIR=item['root'])
    env['PYTHONPATH']=':'.join(str(local/'repo'/p) for p in ['scripts/train','scripts/eval','verl','vln_demo/src'])
    subprocess.run(command,cwd=local/'repo',env=env,check=True)


def recover_prelaunch(plan,key,rank):
    item=ranked_item(plan['items'][key],rank);root,_=node_paths(item,rank)
    require(subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==plan['control_commit'],
            'recovery controller commit differs')
    require(item['kind']=='training' and os.uname().nodename==item['hosts'][rank],'wrong recovery target')
    require(io.exists(root/f'node{rank}-failure.json'),'no recorded preparation failure')
    old_attempt=io.load(root/f'node{rank}-start.json')['identity']
    require(not exact_live(old_attempt,argv=False),'previous restart controller is still alive')
    require(not io.exists(Path(item['root'])/f'git_runtime/node_{rank:02d}') and
            not io.exists(Path(item['root'])/f'launch/owner_{rank:02d}.json'),
            'new business already exists; recovery may not replay it')
    resumed=io.exists(Path(item['root'])/f'launch/intent_{rank:02d}.json')
    if resumed:
        from start_megatron_history_resident import validate_unstarted_handoff
        validate_unstarted_handoff(Path(item['root']),rank,item['commit'])
    receipt=root/f'node{rank}-prelaunch-recovery.json'
    with receipt.open('x') as stream:json.dump(dict(time=time.time(),identity=proc(os.getpid()),business_replay=False),stream)
    stop=threading.Event()
    def heartbeat():
        while not stop.is_set():
            print(json.dumps(dict(event='prelaunch_recovery',hostname=os.uname().nodename,
                pid=os.getpid(),time=time.time(),stage='clean_owned_retired_processes_then_first_business_launch')),flush=True)
            stop.wait(20)
    threading.Thread(target=heartbeat,daemon=True).start()
    try:
        clean_old_storage(item,rank,root/f'node{rank}-recovery-storage-cleanup.json',resume_unstarted=resumed)
        start_training(item,rank,recovery=True)
        io.write_json(root/f'node{rank}-prelaunch-recovered.json',dict(time=time.time()))
    except BaseException as exc:
        io.write_json(root/f'node{rank}-prelaunch-recovery-failure.json',dict(time=time.time(),error=repr(exc)))
        raise
    finally:stop.set()


def stop_evaluation(item,rank):
    from rule_teacher_openloop_companion import retire_evaluation_owner,waiting_reservation
    old=Path(item['old_root']);root,_=node_paths(item,rank)
    config_path=old/'run/watcher-config.json';config=io.load(config_path)
    owner=config['resident_owners'][rank]
    require(owner['hostname']==os.uname().nodename,'old eval hostname differs')
    start=io.load(old/f'run/watchers/node{rank}/start.json')
    identity=start['identity'];actual=proc(identity['pid'])
    if actual and b'resident-watch' in [x.encode() for x in actual['argv']]:
        # This launcher retains its controller after watch() catches TERM.
        # Quiesce it only after its own cleanup has restored the bound holder.
        require(exact_live(identity,argv=False),'legacy watcher PID changed')
        require(start['config']==handoff.bound(config_path),'old watcher config changed')
        intent=root/f'node{rank}-stop-intent.json'
        require(not io.exists(intent),'legacy retirement already attempted')
        io.write_json(intent,dict(identity=actual,owner=owner,time=time.time()))
        exit_path=old/'nodes'/os.uname().nodename/'business-exit.json'
        if not io.exists(exit_path):signal_pid(actual,{1,os.getpid(),owner['controller_identity']['pid']})
        wait_for(lambda:io.exists(exit_path) or legacy_terminated(old,rank),600,'legacy watcher cleanup not complete')
        wait_for(lambda: _idle(owner),300,'legacy GPU holder not ready')
        from recover_rl_companions import quiesce
        quiesce(dict(root=item['root'],old_root=item['old_root']),rank)
    elif actual:
        retire_evaluation_owner(config_path,rank,root/f'node{rank}-retired.json',
                                reason='user requested new initialization and fresh companion')
    else:
        # Terminal evidence must predate replacement; vanished active owners
        # are not enough to prove that their children were cleaned up.
        require(io.exists(old/f'bootstrap/node{rank}/business-exit.json') or
                io.exists(old/f'run/watchers/node{rank}/failure.json') or
                io.exists(old/f'run/watchers/node{rank}/completion.json'),
                'evaluation vanished without terminal receipt')
    waiting_reservation(owner)
    io.write_json(root/f'node{rank}-stopped.json',dict(owner=owner,time=time.time()))
    return owner


def legacy_terminated(old,rank):
    failure=Path(old)/f'run/watchers/node{rank}/failure.json'
    return io.exists(failure) and io.load(failure).get('error')=="KeyboardInterrupt('companion termination requested')"


def start_evaluation(plan,item,rank,owner):
    local=Path('/root/rl-code')/(Path(item['root']).name+'-controller')/'repo'
    checkout(plan['repository'],item['commit'],local)
    command=[sys.executable,'-B',str(local/'scripts/eval/start_unified_companion_resident.py'),
        '--spec',str(Path(item['root'])/'spec.json'),'--rank',str(rank),'--debug-output',owner['debug_output']]
    subprocess.run(command,cwd=local,env=dict(os.environ,ACTION_HEAD_TEACHER_STOP_TRAINING='0'),check=True)


def recover_evaluation_retirement(plan,key,rank):
    item=plan['items'][key];root,_=node_paths(item,rank);old=Path(item['old_root'])
    require(item['kind']=='evaluation' and item['hosts'][rank]==os.uname().nodename,'wrong evaluation target')
    require(not io.exists(Path(item['root'])/f'bootstrap/node{rank}/identity.json'),'new evaluation already started')
    intent=io.load(root/f'node{rank}-stop-intent.json')
    require(legacy_terminated(old,rank),'old watcher has no requested termination receipt')
    require(io.load(old/f'run/watchers/node{rank}/failure.json')['time']>=intent['time'],'stale old termination')
    config=io.load(old/'run/watcher-config.json');owner=config['resident_owners'][rank]
    from rule_teacher_openloop_companion import waiting_reservation
    waiting_reservation(owner)
    controller=io.load(root/f'node{rank}-start.json')['identity']
    receipt=root/f'node{rank}-retirement-recovery.json'
    with receipt.open('x') as f:json.dump(dict(time=time.time(),old_controller=controller),f)
    if exact_live(controller):
        actual=proc(controller['pid'])
        require('--mode' in actual['argv'] and 'node' in actual['argv'] and
                any(x.endswith('/restart_rl_in_place.py') for x in actual['argv']),'unexpected controller')
        signal_pid(actual,{1,os.getpid()})
        wait_for(lambda:not exact_live(controller),20,'old handoff controller still alive')
    from recover_rl_companions import quiesce
    quiesce(item,rank)
    io.write_json(root/f'node{rank}-stopped.json',dict(owner=owner,time=time.time()))
    wait_for(lambda:require_all(item,'stopped'),300,'evaluation node retirement incomplete')
    start_evaluation(plan,item,rank,owner)
    io.write_json(root/f'node{rank}-retirement-recovered.json',dict(time=time.time()))


def _idle(owner):
    state=io.load(Path(owner['debug_output'])/'debug-status.json')
    return (state.get('mode')=='idle' and state.get('transition')=='ready'
            and state.get('occupier_pid') and time.time()-state['heartbeat']<45)


def run_node(plan,key,rank):
    item=ranked_item(plan['items'][key],rank);root,node=node_paths(item,rank)
    require(not REPO.is_relative_to('/data/storage') and not REPO.is_relative_to('/data/storage'),
            'restart implementation must be a container-local Git checkout')
    require(subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==plan['control_commit'],
            'restart controller commit differs')
    require(os.uname().nodename==item['hosts'][rank],'restart hostname/rank differs')
    lock=(root/f'node{rank}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=root/f'node{rank}-start.json';require(not io.exists(started),'node attempt already started')
    io.write_json(started,dict(identity=proc(os.getpid()),time=time.time(),item=key,rank=rank))
    status=dict(stage='preparing_pinned_checkout',rank=rank,item=key)
    stop=threading.Event()
    def heartbeat():
        while not stop.is_set():
            print(json.dumps(dict(event='in_place_restart',time=time.time(),hostname=os.uname().nodename,**status)),flush=True)
            stop.wait(20)
    threading.Thread(target=heartbeat,daemon=True).start()
    try:
        if item['kind']=='training':
            local=Path('/root/rl-code')/Path(item['root']).name
            checkout(plan['repository'],item['commit'],local/'repo')
            training_identity(item,rank)
        else:
            require(item['commit']==plan['control_commit'],'eval source must use the controller checkout')
        io.write_json(root/f'node{rank}-prepared.json',dict(time=time.time(),commit=item['commit']))
        status['stage']='waiting_all_node_preparations'
        wait_for(lambda:require_all(item,'prepared'),1800,'not all nodes prepared')
        status['stage']='retiring_old_business'
        owner=None
        if item['kind']=='training':stop_training(item,rank)
        else:owner=stop_evaluation(item,rank)
        status['stage']='waiting_all_node_cleanup'
        wait_for(lambda:require_all(item,'stopped'),1200,'not all old nodes cleaned up')
        status['stage']='starting_new_business'
        if item['kind']=='training':
            clean_old_storage(item,rank,root/f'node{rank}-storage-cleanup.json')
            start_training(item,rank)
        else:
            start_evaluation(plan,item,rank,owner)
        io.write_json(root/f'node{rank}-returned.json',dict(time=time.time(),returncode=0))
    except BaseException as exc:
        status.update(stage='restart_failed_no_automatic_retry',error=repr(exc))
        io.write_json(root/f'node{rank}-failure.json',dict(time=time.time(),**status))
        # Old Git owner/resident controller remains responsible for occupation.
        raise
    finally:stop.set()


def dispatch(plan,path,key,rank,recovery=False,evaluation_recovery=False):
    item=plan['items'][key];root,_=node_paths(item,rank)
    prefix_name=f'node{rank}'+('-retirement-recovery'+plan.get('retirement_recovery_tag','') if evaluation_recovery else '-prelaunch-recovery' if recovery else '')
    intent=root/f'{prefix_name}-dispatch-intent.json'
    require(not io.exists(intent),'dispatch already attempted')
    control_root=Path(item['root'])/'control'
    # Give each task a separate container checkout; spec.root remains the
    # isolated experiment root and all business modules stay container-local.
    control_root=io.ART/(Path(item['root']).name+'-controller'+('-retirement-recovery'+plan.get('retirement_recovery_tag','') if evaluation_recovery else '-prelaunch-recovery' if recovery else ''))
    control_root.mkdir(exist_ok=True)
    prefix,sep,_=entry(plan['repository'],plan['control_commit'],control_root).partition('sleep 99999999 &\n')
    require(sep,'Git bootstrap boundary changed')
    prefix=prefix.replace('set -uo pipefail\n','set -uo pipefail\nexport LIBRARY_PATH=${LIBRARY_PATH:-}\nsource /usr/local/PPU_SDK/envsetup.sh cuda\n',1)
    local=Path('/root/rl-code')/control_root.name/'repo'
    body=prefix+'''export NCCL_IB_DISABLE=0 NCCL_NET=IB NCCL_NET_PLUGIN=none
export NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4 ACTION_HEAD_RESIDENT_ENTRY=1
unset NCCL_PF_HCA
export ACTION_HEAD_TEACHER_STOP_TRAINING=0
'''+shlex.join(['/usr/local/bin/python3.12','-B',str(local/'scripts/train/restart_rl_in_place.py'),
        '--mode','recover-evaluation-retirement' if evaluation_recovery else 'recover-prelaunch' if recovery else 'node','--plan',str(path),'--item',key,'--rank',str(rank)])+'\n'
    command=root/f'{prefix_name}-bootstrap.sh';io.write_bytes(command,body.encode())
    subprocess.run(['bash','-n',str(command)],check=True)
    code="""import json,subprocess,time
from pathlib import Path
with Path(%r).open('x') as receipt:
 with open('/proc/1/fd/1','w',buffering=1) as raw:
  child=subprocess.Popen(['bash',%r],stdin=subprocess.DEVNULL,stdout=raw,stderr=subprocess.STDOUT,start_new_session=True)
 ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
 json.dump(dict(pid=child.pid,start_ticks=ticks,time=time.time()),receipt)
print(json.dumps(dict(pid=child.pid)))
"""%(str(root/f'{prefix_name}-dispatch.json'),str(command))
    io.write_json(intent,dict(item=key,rank=rank,time=time.time(),automatic_retry=False))
    try:
        result=subprocess.run([life.CLI,'exec','--job-id',item['job_id'],'--workspace',item['space_id'],
            '--pod',item['hosts'][rank],'--ws-proxy-env','--timeout','45','--','python3','-c',code],
            capture_output=True,text=True,timeout=55)
        io.write_json(root/f'{prefix_name}-exec.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,time=time.time()))
    except subprocess.TimeoutExpired as exc:
        io.write_json(root/f'{prefix_name}-exec.json',dict(error=str(exc),command_may_have_started=True,time=time.time()))


def started(item):
    root=Path(item['root'])
    paths=[root/(f'git_runtime/node_{r:02d}/checkout.json' if item['kind']=='training'
                 else f'bootstrap/node{r}/identity.json') for r in range(item['nodes'])]
    if not all(io.exists(p) for p in paths):return False
    for p in paths:
        d=io.load(p);actual=d['code']['commit'] if item['kind']=='training' else d['git']['commit']
        require(actual==item['commit'],'new business source differs')
    return True


def poll_item(plan,path,key):
    item=plan['items'][key];root,_=node_paths(item,0)
    if started(item):return 'new_business_checkout_verified'
    if any(io.exists(root/f'node{r}-failure.json') for r in range(item['nodes'])):return 'restart_failed_inspect_receipts'
    if any(io.exists(root/f'node{r}-dispatch-intent.json') for r in range(item['nodes'])):return 'handoff_dispatched_no_automatic_replay'
    detail=life.checked_detail(item)['response']['data']
    io.write_json(root/'latest-platform-detail.json',dict(time=time.time(),detail=detail))
    if detail['status']!='Running':return 'waiting_platform_'+detail['status']
    require(not detail.get('restartRecord') and detail['podCount']==item['nodes'],'platform identity changed')
    ready=old_training_ready(item) if item['kind']=='training' else old_evaluation_ready(item)
    if not ready:return 'waiting_original_owner_initialization'
    hosts=([io.load(Path(ranked_item(item,r)['old_root'])/f'git_runtime/node_{r:02d}/checkout.json')['hostname'] for r in range(item['nodes'])]
           if item['kind']=='training' else [x['hostname'] for x in io.load(Path(item['old_root'])/'run/watcher-config.json')['resident_owners']])
    require(hosts==item['hosts'] and all(h in detail['pods'] for h in hosts),'Pod topology changed')
    with ThreadPoolExecutor(max_workers=item['nodes']) as pool:
        list(pool.map(lambda rank:dispatch(plan,path,key,rank),range(item['nodes'])))
    return 'handoff_dispatched_once'


def watch(plan,path):
    root=Path(plan.get('supervisor_root',str(Path(plan['root'])/'supervisor')));root.mkdir(parents=True,exist_ok=True)
    lock=(root/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(not io.exists(root/'owner.json'),'supervisor already started')
    io.write_json(root/'owner.json',dict(identity=proc(os.getpid()),plan=handoff.bound(path),time=time.time()))
    state={};completion={key:{} for key in plan['pairs']}
    while True:
        for key,item in plan['items'].items():
            if item.get('old_queues'):continue
            try:state[key]=poll_item(plan,path,key)
            except Exception as exc:state[key]=dict(error=repr(exc),automatic_retry=False)
        if plan.get('handoff_plan'):
            try:
                hp=Path(plan['handoff_plan']);state['hnc_handoff']=handoff.poll(io.load(hp),hp)
            except Exception as exc:state['hnc_handoff']=dict(error=repr(exc))
        for key,pair in plan['pairs'].items():
            try:
                life.poll({kind:plan['items'][name] for kind,name in pair.items()},completion[key])
            except Exception as exc:completion[key]['error']=repr(exc)
        state['completion']=completion;state['heartbeat']=time.time()
        io.write_json(root/'status.json',state)
        print(json.dumps(state),flush=True)
        if completion:
            if all(all(s.get(k) in life.DONE_STATES for k in ['training','evaluation']) for s in completion.values()):return
        elif all(state.get(key)=='new_business_checkout_verified' for key in plan['items']):
            return
        time.sleep(30)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['watch','node','recover-prelaunch','recover-evaluation-retirement','status'],required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--item');p.add_argument('--rank',type=int)
    args=p.parse_args();plan=io.load(args.plan)
    if args.mode=='recover-prelaunch':return recover_prelaunch(plan,args.item,args.rank)
    if args.mode=='recover-evaluation-retirement':return recover_evaluation_retirement(plan,args.item,args.rank)
    if args.mode=='node':return run_node(plan,args.item,args.rank)
    if args.mode=='watch':return watch(plan,args.plan)
    print(json.dumps(io.load(Path(plan.get('supervisor_root',str(Path(plan['root'])/'supervisor')))/'status.json')))


if __name__=='__main__':main()
