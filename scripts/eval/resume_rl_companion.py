"""Resume an explicitly bound failed companion in a fresh local Git attempt.

Reuses resident retirement/bootstrap, completed-case seeding, unified scoring
and success-only DLC completion. Never edits previous failure/results.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'scripts/train'),str(REPO/'scripts/eval')]
import align_three_source_observations as io
from resident_pair_handoff import bound
from watch_v6_rl_local_credit import same_process_alive


def current_reservation(owner):
    """Resolve only the same node/output's independently restarted idle holder.

    Called after the old evaluation owner has exited/retired. Never substitute
    an unrelated PID or edit the previous attempt's resident ownership record.
    """
    from resident_rl_attempt import proc
    from rule_teacher_openloop_companion import waiting_reservation
    root=Path(owner['debug_output'])
    start=io.load(root/'debug-start.json');state=io.load(root/'debug-status.json')
    actual=proc(start['pid'])
    expected=[owner['debug_control'],'run','--output',str(root),'--sleep-pid',str(start['sleep_pid'])]
    if (owner['hostname']!=os.uname().nodename or start['hostname']!=owner['hostname']
            or start['source']!=owner['debug_control'] or not actual
            or actual['argv'][-len(expected):]!=expected
            or state.get('pid')!=actual['pid'] or state.get('sleep_pid')!=start['sleep_pid']
            or time.time()-state.get('heartbeat',0)>60
            or os.readlink(f'/proc/{actual["pid"]}/cwd')!=str(Path(owner['debug_control']).parents[2])):
        raise ValueError('replacement reservation process does not belong to this node/output')
    resolved=dict(owner,controller_identity=actual)
    waiting_reservation(resolved)
    return resolved


def quiesce_reservation(owner, receipt):
    """Release owned GPU work, then suspend its controller without parent respawn."""
    from resident_rl_attempt import proc,exact_live
    from dlc_reservation_request import request_mode
    identity=proc(owner['controller_identity']['pid'])
    if not identity or not same_process_alive(owner['controller_identity']):
        raise ValueError('reservation changed before retirement')
    state=request_mode(owner['debug_output'],'active',lease_seconds=600,write=io.write_json)
    if (not exact_live(identity) or state.get('pid')!=identity['pid']
            or state.get('mode')!='active' or state.get('transition')!='ready'
            or state.get('occupier_pid') is not None):
        raise RuntimeError('owned occupier release was not acknowledged')
    # Keep the controller a live child of the original platform launcher, which
    # otherwise restarts its OLD code. The new attempt owns a separate controller.
    io.write_json(receipt,dict(identity=identity,owner=owner,state='release_acknowledged',
        acknowledgement=state,time=time.time()))
    fd=os.pidfd_open(identity['pid'])
    try:
        if not exact_live(identity):raise ValueError('reservation identity changed before suspension')
        signal.pidfd_send_signal(fd,signal.SIGSTOP)
    finally:os.close(fd)
    deadline=time.monotonic()+10
    while True:
        actual=proc(identity['pid'])
        if not exact_live(identity):raise RuntimeError('reservation exited during suspension')
        if actual['state'] in ('T','t'):break
        if time.monotonic()>deadline:raise TimeoutError('reservation suspension not acknowledged')
        time.sleep(.1)
    io.write_json(receipt,dict(identity=identity,owner=owner,state='suspended_without_gpu_work',
        acknowledgement=state,time=time.time()))
    return identity


def recover_logs(old, rank, output):
    """Preserve retained stdout; only append after byte-exact prefix proof."""
    records=[]
    for file in (old/'run/evaluation').glob(f'*/stages/*/global_step_*/node_{rank}/launch/completion.json'):
        completion=io.load(file)
        if not completion.get('owned_process_groups_cleaned'):
            raise ValueError('previous services not cleaned: '+str(file))
        for record in completion.get('log_archives',[]):
            if not record.get('local_spool_retained'):continue
            local=Path(record['local_spool']);archive=Path(record['archive'])
            if (not local.is_relative_to('/tmp') or not archive.is_relative_to(old)
                    or record.get('error')):
                raise ValueError('retained log scope/error requires inspection')
            data=io.read_bytes(local)
            before=io.read_bytes(archive) if io.exists(archive) else b''
            reclaimed=int(record.get('reclaimed_bytes',0))
            if (not 0 <= reclaimed <= len(before) <= len(data)
                    or data[reclaimed:len(before)]!=before[reclaimed:]):
                raise ValueError('retained archive prefix differs: '+str(archive))
            # Reuse production idempotent positional writes; no original writer
            # remains after retire_evaluation_owner and process-group cleanup.
            from service_log_spool import LogSpool
            spool=object.__new__(LogSpool)
            spool.destination=archive;spool.path=local;spool.offset=len(before);spool.base=0
            while spool.offset<len(data):
                io.bounded(spool._copy_once,'retained_service_log:'+str(archive))
            recovered=io.read_bytes(archive)
            if recovered!=before+data[len(before):]:raise ValueError('retained log recovery differs')
            records.append(dict(source=str(local),archive=str(archive),bytes=len(data),
                appended=len(data)-len(before),sha256=io.digest(recovered),source_retained=True))
    io.write_json(output,dict(rank=rank,logs=records,time=time.time()))
    return records


def completed_sources(old, binding, allowed_changes):
    """Keep an older partial source if the latest attempt failed before plan publication."""
    sources=json.loads(json.dumps(binding.get('completed_case_sources', {})))
    for reference in sources.values():
        refs=reference.get('sources',[reference])
        for ref in refs:
            ref['allowed_code_changes']=sorted(set(ref.get('allowed_code_changes',[]))|set(allowed_changes))
    for previous in sorted((old/'run/evaluation').glob('global_step_*/plan.json')):
        plan=io.load(previous);step=int(previous.parent.name.rsplit('_',1)[1])
        if step not in binding['steps']:continue
        sources[f'global_step_{step:06d}']=dict(path=str(previous),sha256=io.digest(io.read_bytes(previous)),
            commit=plan['source_provenance']['git_runtime']['commit'],
            allowed_code_changes=allowed_changes,skip_unfinished_shards=True)
    return sources


def node(path, rank):
    item=io.load(path);old=Path(item['old_root']);root=Path(item['root'])
    from unified_closedloop_eval import git_runtime
    if git_runtime(True)['commit']!=item['commit']:raise ValueError('wrong local Git commit')
    if item.get('bootstrap_failure'):
        validate_bootstrap_failure(item)
        from resident_rl_attempt import proc
        start=io.load(old/'nodes'/os.uname().nodename/'bootstrap-start.json')
        actual=proc(start['identity']['pid'])
        old_repo=Path('/root/rl-code')/old.name/'repo'
        if (not actual or actual['start_ticks']!=str(start['identity']['start_ticks'])
                or str(old_repo/'scripts/eval/unified_closedloop_eval.py') not in actual['argv']
                or 'resident-watch' not in actual['argv'] or str(old/'spec.json') not in actual['argv']
                or os.readlink(f'/proc/{actual["pid"]}/cwd')!=str(old_repo)):
            raise ValueError('failed bootstrap process identity changed')
        owner=item['residents'][rank]
        if owner['hostname']!=os.uname().nodename:raise ValueError('wrong bootstrap resident')
        io.write_json(root/'platform'/f'old-clean-node{rank}.json',dict(
            identity=actual, business_exited=True, idle_bootstrap_retained=True,time=time.time()))
        # The original launcher only waits on its independent idle controller.
        # Reuse that reservation; never start a second holder or replay its attempt.
        from start_unified_companion_resident import main
        sys.argv=['start_unified_companion_resident','--spec',str(root/'spec.json'),
            '--rank',str(rank),'--debug-output',owner['debug_output']]
        return main()
    previous=Path(item.get('retire_root',item['old_root']))
    config=previous/'run/watcher-config.json'
    if bound(config)!=item['old_config']:raise ValueError('previous watcher config changed')
    c=io.load(config);owner=c['resident_owners'][rank]
    start=io.load(previous/f'run/watchers/node{rank}/start.json')
    if (start['config']!=bound(config) or start['hostname']!=os.uname().nodename
            or owner['hostname']!=os.uname().nodename):raise ValueError('old node identity differs')
    from rule_teacher_openloop_companion import retire_evaluation_owner,waiting_reservation
    if same_process_alive(start['identity']):
        argv=Path(f"/proc/{start['identity']['pid']}/cmdline").read_bytes().split(b'\0')
        if b'resident-watch' in argv:
            # Reuse the production retirement path: an active original
            # launcher must clean its suite before its idle quiesce gate.
            from restart_rl_in_place import stop_evaluation
            stop_evaluation(dict(item,old_root=str(previous)),rank)
        else:
            retire_evaluation_owner(config,rank,root/'platform'/f'retired-node{rank}.json',
                reason='authorized companion control/log archive repair continuation')
    else:
        if item.get('replace_reservation'):
            if not io.exists(previous/f'run/watchers/node{rank}/failure.json'):
                raise ValueError('missing failure receipt for vanished companion')
            owner=current_reservation(owner)
        waiting_reservation(owner)
        io.write_json(root/'platform'/f'retired-node{rank}.json',dict(identity=start['identity'],already_exited=True,time=time.time()))
    recover_logs(old,rank,root/'platform'/f'retained-logs-node{rank}.json')
    if item.get('cleanup_legacy_routing_shm'):
        # This is explicit retired-owner cache cleanup, not a startup probe.
        waiting_reservation(owner)
        from action_head.serve.routing_shm import cleanup_legacy_orphans
        released=cleanup_legacy_orphans()
        io.write_json(root/'platform'/f'legacy-routing-shm-node{rank}.json',dict(
            rank=rank,hostname=os.uname().nodename,time=time.time(),**released))
    io.write_json(root/'platform'/f'old-clean-node{rank}.json',dict(hostname=os.uname().nodename,rank=rank,time=time.time()))
    deadline=time.monotonic()+600
    while not all(io.exists(root/'platform'/f'old-clean-node{i}.json') for i in range(4)):
        if time.monotonic()>deadline:raise TimeoutError('old companion cleanup incomplete')
        time.sleep(2)
    if rank==0:
        binding=io.load(root/'binding.json')
        sources=completed_sources(old,binding,item['allowed_code_changes'])
        binding['completed_case_sources']=sources
        io.write_json(root/'binding.json',binding)
        io.write_json(root/'platform/reuse-sources-ready.json',dict(sources=list(sources),time=time.time()))
    while not io.exists(root/'platform/reuse-sources-ready.json'):
        if time.monotonic()>deadline:raise TimeoutError('reuse binding preparation incomplete')
        time.sleep(2)
    if item.get('replace_reservation'):
        quiesce_reservation(owner,root/'platform'/f'reservation-retired-node{rank}.json')
        from unified_companion_task import run
        return run(root/'spec.json',rank)
    from start_unified_companion_resident import main
    sys.argv=['start_unified_companion_resident','--spec',str(root/'spec.json'),
        '--rank',str(rank),'--debug-output',owner['debug_output']]
    return main()


def dispatch(path,rank):
    item=io.load(path)
    from prepare_git_workspace import entry as git_entry
    from watch_rl_pair_completion import CLI
    import shlex
    root=Path(item['root']);intent=root/'platform'/f'dispatch-intent-node{rank}.json'
    prefix,sep,_=git_entry(item['repository'],item['commit'],root).partition('sleep 99999999 &\n')
    if not sep:raise ValueError('Git bootstrap boundary changed')
    with intent.open('x') as f:json.dump(dict(rank=rank,time=time.time()),f)
    prefix=prefix.replace('set -uo pipefail\n','set -uo pipefail\nexport LIBRARY_PATH=${LIBRARY_PATH:-}\nsource /usr/local/PPU_SDK/envsetup.sh cuda\n',1)
    local=Path('/root/rl-code')/root.name/'repo'
    command=root/'platform'/f'bootstrap-node{rank}.sh'
    body=prefix+'export ACTION_HEAD_TEACHER_STOP_TRAINING=0 ACTION_HEAD_STRICT_STOP_GRPO=0\n'+shlex.join([
        '/usr/local/bin/python3.12','-B',str(local/'scripts/eval/resume_rl_companion.py'),
        '--mode','node','--plan',str(path),'--rank',str(rank)])+'\n'
    io.write_bytes(command,body.encode());subprocess.run(['bash','-n',str(command)],check=True)
    code="""import subprocess,json
from pathlib import Path
with Path(%r).open('x') as receipt:
 with open('/proc/1/fd/1','w',buffering=1) as raw:
  child=subprocess.Popen(['bash',%r],stdin=subprocess.DEVNULL,stdout=raw,stderr=subprocess.STDOUT,start_new_session=True)
 ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
 json.dump(dict(pid=child.pid,start_ticks=ticks),receipt)
print(json.dumps(dict(pid=child.pid)))
"""%(str(root/'platform'/f'dispatch-node{rank}.json'),str(command))
    result=subprocess.run([CLI,'exec','--job-id',item['job_id'],'--workspace',item['space_id'],
        '--pod',item['residents'][rank]['hostname'],'--ws-proxy-env','--timeout','45',
        '--','python3','-B','-c',code],capture_output=True,text=True,timeout=55)
    io.write_json(root/'platform'/f'exec-node{rank}.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,automatic_retry=False))
    return dict(rank=rank,returncode=result.returncode)


def validate_bootstrap_failure(item):
    old=Path(item['old_root'])
    if io.exists(old/'run/watcher-config.json') or list((old/'run/evaluation').glob('*/plan.json')):
        raise ValueError('bootstrap recovery cannot replace an existing watcher or evaluation')
    references=item['bootstrap_failure']
    expected={old/'spec.json'}
    for rank,owner in enumerate(item['residents']):
        expected.update([old/f'bootstrap/node{rank}.json',old/f'bootstrap/failure-node{rank}.json',
            old/'nodes'/owner['hostname']/'business-exit.json',
            old/'nodes'/owner['hostname']/'bootstrap-start.json'])
    if len(item['residents'])!=4 or len({o['hostname'] for o in item['residents']})!=4:
        raise ValueError('four distinct bootstrap residents required')
    if {Path(ref['path']) for ref in references}!=expected or len(references)!=len(expected):
        raise ValueError('incomplete bootstrap failure binding')
    for reference in references:
        if bound(Path(reference['path']))!=reference:raise ValueError('bootstrap failure proof changed')
    for rank,owner in enumerate(item['residents']):
        row=io.load(old/f'bootstrap/node{rank}.json')
        failure=io.load(old/f'bootstrap/failure-node{rank}.json')
        exited=io.load(old/'nodes'/owner['hostname']/'business-exit.json')
        if (row['rank']!=rank or row['owner']!=owner or not row['ready']
                or row['git']['commit']!=io.load(old/'spec.json')['commit']
                or failure['rank']!=rank or not failure.get('error')
                or exited.get('returncode')!=1 or not exited.get('error')):
            raise ValueError('bootstrap failure/owner differs')
    return True


def prepare_bootstrap(previous_plan, root, commit):
    """Recover a failed pre-watcher bootstrap without allocating new DLC nodes."""
    from watch_rl_pair_completion import checked_detail
    from formal_rl_pair_submission import companion_checkpoint
    item=io.load(previous_plan);old=Path(item['root']);root=Path(root)
    if root==old or not root.resolve().is_relative_to(io.ART) or io.exists(root):
        raise ValueError('use a fresh authorized companion attempt')
    detail=checked_detail(item)['response']['data']
    if item.get('kind')!='evaluation' or item.get('nodes')!=4 or detail['status']!='Running' or detail.get('restartRecord'):
        raise ValueError('failed bootstrap allocation identity changed')
    spec=io.load(old/'spec.json');residents=[io.load(old/f'bootstrap/node{r}.json')['owner'] for r in range(4)]
    paths=[old/'spec.json']
    for rank,owner in enumerate(residents):
        paths.extend([old/f'bootstrap/node{rank}.json',old/f'bootstrap/failure-node{rank}.json',
            old/'nodes'/owner['hostname']/'business-exit.json',old/'nodes'/owner['hostname']/'bootstrap-start.json'])
    revision=subprocess.check_output(['git','-C',str(REPO),'rev-parse',commit+'^{commit}'],text=True).strip()
    result=dict(item,root=str(root),old_root=str(old),commit=revision,
        repository=io.load(old/'bootstrap/node0.json')['git']['repository'],
        residents=residents,bootstrap_failure=[bound(p) for p in paths],handoff_only=True)
    validate_bootstrap_failure(result)
    selection=io.load(old/'selection.json');binding=io.load(old/'binding.json')
    selection['checkpoints']=[companion_checkpoint(c['path'],c['training_config'],
        binding['slow_model'],name=c['name']) for c in selection['checkpoints']]
    root.mkdir();(root/'platform').mkdir()
    spec.update(root=str(root),commit=revision,selection=str(root/'selection.json'),binding=str(root/'binding.json'))
    for name,value in [('spec.json',spec),('selection.json',selection),('binding.json',binding)]:io.write_json(root/name,value)
    path=root/'platform/repair-plan.json';io.write_json(path,result)
    io.write_json(root/'platform/monitor-plan.json',dict(root=str(root),items={'companion':result}))
    return path


def prepare(previous_plan, root, commit, *, retire_plan=None, allow_running_handoff=False,
            replace_reservation=False):
    """Prepare config-only continuation; all business code remains in Git."""
    from watch_rl_pair_completion import checked_detail
    item=io.load(previous_plan);previous=Path(item['root']);root=Path(root)
    if (root==previous or not root.resolve().is_relative_to(io.ART) or io.exists(root)):
        raise ValueError('use a fresh authorized companion attempt')
    detail=checked_detail(item)['response']['data']
    if detail['status']!='Running' or detail.get('restartRecord'):
        raise ValueError('bound DLC identity/state changed')
    failures=sorted((previous/'run/watchers').glob('node*/failure.json'))
    if not failures and not allow_running_handoff:raise ValueError('no original companion failure to continue')
    spec=io.load(previous/'spec.json');selection=io.load(previous/'selection.json')
    binding=io.load(previous/'binding.json');config=io.load(previous/'run/watcher-config.json')
    repository=item.get('repository')
    if not repository:
        proof_path=previous/'bootstrap/node0/identity.json'
        if not io.exists(proof_path):proof_path=previous/'bootstrap/node0.json'
        proof=io.load(proof_path)['git']
        if proof['commit']!=spec['commit']:raise ValueError('original Git source proof differs')
        repository=proof['repository']
    retire_item=io.load(retire_plan) if retire_plan else item
    retire=Path(retire_item['root'])
    if retire_plan and (retire_item['job_id']!=item['job_id']
            or retire_item['space_id']!=item['space_id']
            or retire_item.get('old_root')!=str(previous)):
        raise ValueError('retirement must belong to the same failed companion continuation')
    retire_config=io.load(retire/'run/watcher-config.json') if retire_plan else config
    def controller_identities(value):
        identities=json.loads(json.dumps(value['resident_owners']))
        for owner in identities:
            owner.pop('debug_control',None)
            # R/S is a scheduler snapshot, not part of process identity.
            owner.get('controller_identity',{}).pop('state',None)
        return identities
    if controller_identities(retire_config)!=controller_identities(config):
        raise ValueError('resident controller identities changed')
    revision=subprocess.check_output(['git','-C',str(REPO),'rev-parse',commit+'^{commit}'],text=True).strip()
    changes=subprocess.check_output(['git','-C',str(REPO),'diff','--name-only','-z',spec['commit'],revision]).decode().strip('\0').split('\0')
    root.mkdir();(root/'platform').mkdir()
    spec.update(root=str(root),commit=revision,selection=str(root/'selection.json'),binding=str(root/'binding.json'))
    for name,value in [('spec.json',spec),('selection.json',selection),('binding.json',binding)]:
        io.write_json(root/name,value)
    result=dict(item,root=str(root),old_root=str(previous),retire_root=str(retire),commit=revision,repository=repository,
        old_config=bound(retire/'run/watcher-config.json'),residents=config['resident_owners'],
        allowed_code_changes=[p for p in changes if p],handoff_only=True)
    if replace_reservation:result['replace_reservation']=True
    # Authorization/failure proofs belong to the immediate predecessor only.
    result.pop('resumed_failure',None)
    result.pop('authorized_handoff',None)
    if failures:result['resumed_failure']=bound(failures[0])
    if allow_running_handoff:
        result['authorized_handoff']=dict(spec=bound(previous/'spec.json'),
            config=bound(previous/'run/watcher-config.json'),reason='user authorized updating running companions')
    path=root/'platform/repair-plan.json';io.write_json(path,result)
    io.write_json(root/'platform/monitor-plan.json',dict(root=str(root),items={'companion':result}))
    return path


def prepare_batch(previous_plan, root, commit, *, allow_running_handoff=False):
    """Bind one fresh continuation for each explicitly selected evaluation DLC."""
    if not allow_running_handoff:raise ValueError('batch handoff requires explicit running-companion authorization')
    items=io.load(previous_plan)['items'];root=Path(root)
    if (not items or any(i.get('kind')!='evaluation' or i.get('nodes')!=4 for i in items.values())
            or len({i['job_id'] for i in items.values()})!=len(items)):
        raise ValueError('batch must contain distinct four-node evaluation DLCs only')
    if not root.resolve().is_relative_to(io.ART) or io.exists(root):raise ValueError('use a fresh authorized batch root')
    root.mkdir();(root/'platform').mkdir();prepared={}
    for name,item in items.items():
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in name):
            raise ValueError('invalid companion key')
        previous=root/'platform'/('previous-'+name+'.json');io.write_json(previous,item)
        target=root.parent/(root.name+'-'+name)
        path=prepare(previous,target,commit,allow_running_handoff=True)
        prepared[name]=io.load(path)
    path=root/'platform/batch-plan.json';io.write_json(path,dict(root=str(root),items=prepared))
    return path


def dispatch_batch(path):
    from recover_rl_companions import valid_anchor
    from watch_rl_pair_completion import checked_detail
    plan=io.load(path)
    for item in plan['items'].values():
        if any(key not in item for key in ('repository','commit','job_id','space_id','residents')):
            raise ValueError('incomplete prepared runtime identity; no nodes dispatched')
    for item in plan['items'].values():
        detail=checked_detail(item)['response']['data']
        if detail['status']!='Running' or detail.get('restartRecord') or not valid_anchor(item):
            raise ValueError('batch source allocation or completed anchor changed')
    targets=[(name,Path(item['root'])/'platform/repair-plan.json',rank)
             for name,item in plan['items'].items() for rank in range(4)]
    def start(target):
        name,item_path,rank=target
        return dict(item=name,**dispatch(item_path,rank))
    with ThreadPoolExecutor(max_workers=8) as pool:return list(pool.map(start,targets))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['node','dispatch','prepare','prepare-bootstrap','prepare-batch','dispatch-batch'],required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--rank',type=int)
    p.add_argument('--output',type=Path);p.add_argument('--commit');p.add_argument('--retire-plan',type=Path)
    p.add_argument('--allow-running-handoff',action='store_true')
    p.add_argument('--replace-reservation',action='store_true',
        help='retire the owned old controller and start the pinned Git occupier in a fresh attempt')
    args=p.parse_args()
    if args.mode in ('prepare','prepare-bootstrap','prepare-batch'):
        if not args.output or not args.commit:p.error('prepare requires --output and --commit')
        if args.mode=='prepare-bootstrap':
            print(prepare_bootstrap(args.plan,args.output,args.commit))
        elif args.mode=='prepare-batch':
            print(prepare_batch(args.plan,args.output,args.commit,allow_running_handoff=args.allow_running_handoff))
        else:print(prepare(args.plan,args.output,args.commit,retire_plan=args.retire_plan,
                          allow_running_handoff=args.allow_running_handoff,
                          replace_reservation=args.replace_reservation))
        return
    if args.mode=='dispatch-batch':print(json.dumps(dispatch_batch(args.plan)));return
    if args.mode=='node':return node(args.plan,args.rank)
    item=io.load(args.plan)
    import watch_rl_pair_completion as L
    detail=L.checked_detail(item)['response']['data']
    if detail['status']!='Running' or detail.get('restartRecord'):raise ValueError('DLC changed')
    from recover_rl_companions import valid_anchor
    if not valid_anchor(item):raise ValueError('no bound completed anchor and authorized continuation')
    with ThreadPoolExecutor(max_workers=4) as pool:print(json.dumps(list(pool.map(lambda i:dispatch(args.plan,i),range(4)))))


if __name__=='__main__':main()
