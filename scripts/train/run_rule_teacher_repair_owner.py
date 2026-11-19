"""Own a repair validation/collection without changing the DLC workspace entry.

Reuse its bounded active/idle lease, original stdout, occupier and independent
collection supervisor. A child error is recorded and returns GPUs to idle;
it never propagates to the platform's resident entry.
"""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback


def module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value
    spec.loader.exec_module(value);return value


def shared_parent_live(parent):
    """A recycled PID must never authorize borrowing another owner's services."""
    from watch_v6_rl_local_credit import process_identity
    current=process_identity(parent['pid'])
    return bool(current and current['state']!='Z'
        and str(current['start_ticks'])==str(parent['start_ticks']))


def validate_shared_lane(config, plan):
    parent=config.get('shared_node_parent')
    if parent is None:return
    if not shared_parent_live(parent):raise ValueError('shared node owner identity changed')
    gpus=config['spare_gpus']
    if not gpus or len(gpus)!=len(set(gpus)):
        raise ValueError('shared recovery requires an explicit distinct GPU subset')
    if len(plan['services'])!=1 or len(plan['collectors'])!=1:
        raise ValueError('shared recovery owns exactly one brain and collector')
    brain=plan['services'][0]
    if (not brain['name'].startswith('brain-')
            or brain['environment']['CUDA_VISIBLE_DEVICES']!=','.join(map(str,gpus))):
        raise ValueError('recovery reservation differs from its brain GPU subset')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    config=json.loads(parser.parse_args().config.read_text())
    root=Path(config['output']);root.mkdir(parents=True,exist_ok=True)
    source=Path(config['source_root'])
    sys.path[:0]=[str(source/'scripts/train'),str(source/'scripts/eval'),str(source/'verl'),str(source/'vln_demo/src')]
    from prepare_rule_teacher_three_source import verify_plan,sha256
    import v6_rl_checkpoint_companion as companion
    from watch_checkpoints import stop_process_group
    io=module(config['io_retry'],'_repair_owner_io')
    # Child status/completion live beside owner/, inside the same attempt.
    # Restrict retry to that complete attempt, not just the owner's receipts.
    io.install(companion,root.parent)
    save=companion.atomic_json
    def identity(pid):
        text=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return text[19],text[0]
    assert os.uname().nodename==config['hostname']
    for pid,expected in config['resident_identities'].items():
        ticks,state=identity(pid);assert ticks==expected and state!='Z'
    assert sha256(config['control'])==config['control_sha256']
    assert sha256(config['io_retry'])==config['io_retry_sha256']
    with (root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'start.json').exists():raise ValueError('repair attempt already started')
        stage={'stage':'cpu_preflight','updated':time.time(),'child_pid':None}
        stopped=threading.Event()
        original=os.open(f"/proc/{config['stdout_pid']}/fd/1",os.O_WRONLY|os.O_APPEND)
        def heartbeat():
            while not stopped.is_set():
                value=dict(event='sage_repair_owner',hostname=config['hostname'],time=time.time(),
                    **stage,observation_age_seconds=round(time.time()-stage['updated'],1),
                    meaning='actual repair owner phase; not a completed quality gate')
                try:os.write(original,(json.dumps(value)+'\n').encode())
                except OSError:pass
                stopped.wait(20)
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        control=[sys.executable,'-B',config['control']]
        child=None;reservation=None;error=None;quality_gate_validated=False
        def lease():
            subprocess.run(control+['active','--output',config['debug_output'],'--lease-seconds','600'],check=True)
        def terminate(*_):raise KeyboardInterrupt('repair owner interrupted')
        signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
        save(root/'start.json',dict(pid=os.getpid(),start_ticks=identity(os.getpid())[0],
            hostname=config['hostname'],config=config,started_at=time.time()))
        try:
            plan=json.loads(Path(config['launch_plan']).read_text())
            validate_shared_lane(config,plan)
            shared=config.get('shared_node_parent')
            verify_plan(json.loads(Path(plan['source_plan']).read_text()))
            if config.get('required_quality_gate'):
                # Full repair collection must bind the repaired policy's real
                # small-batch evidence, never inherit the old producer gate.
                from sft.build_rule_teacher_policy_review import (
                    policy_identity, digest, validate_policy_review, file_binding)
                policy=policy_identity(source,dict(admission_mode='relaxed',
                    grounding_mode='map_open',match_policy='target_or_direction'))
                proof=validate_policy_review(dict(schema='rule_teacher_policy_admission_v1',
                    review_type='rule_policy_admission',human_verified=False,
                    semantic_reviewed=False,policy_source_root=str(source),policy=policy,
                    policy_sha256=digest(policy),policy_gate=file_binding(config['required_quality_gate'])))
                save(root/'quality-gate-verification.json',proof)
                quality_gate_validated=True
            stage.update(stage='releasing_owned_idle_reservation',updated=time.time())
            if not shared:lease()
            # The workspace CLI only stops its own occupier. Never kill a
            # remaining CUDA client or interpret a failed query as idle.
            probe=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                                 check=True,capture_output=True,text=True,timeout=15)
            if not shared and probe.stdout.strip():raise RuntimeError('unowned CUDA client remains after idle release')
            spare=config['spare_gpus']
            reserve_config=dict(waiting_gpu_reservation=dict(enabled=True,gpu_count=len(spare),
                python=sys.executable,script=str(source/'verl/scripts/train/action_head_sft_data/gpu_occupier.py'),
                startup_timeout=120))
            reservation=companion.WaitingGpuReservation(reserve_config,root,stop_process_group)
            previous=os.environ.get('CUDA_VISIBLE_DEVICES');os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,spare))
            try:
                if not shared:reservation.start()
            finally:
                if previous is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
                else:os.environ['CUDA_VISIBLE_DEVICES']=previous
            command=[sys.executable,'-B',str(source/'scripts/train/supervise_rule_teacher_collection.py'),
                     '--plan',config['launch_plan']]
            environment=os.environ.copy();environment['RULE_COLLECTION_RUNTIME_ROOT']=config['runtime_root']
            with (root/'collection-supervisor.log').open('xb') as log:
                child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                       start_new_session=True,env=environment)
            stage.update(stage='repair_validation_startup',updated=time.time(),child_pid=child.pid)
            save(root/'child.json',dict(pid=child.pid,start_ticks=identity(child.pid)[0],command=command))
            renewed=time.monotonic()
            while child.poll() is None:
                if shared:
                    if not shared_parent_live(shared):
                        raise RuntimeError('borrowed node services owner exited; preserve this attempt for continuation')
                else:reservation.check()
                path=Path(config['launch_plan']).parent/'status.json'
                status=json.loads(path.read_text()) if path.exists() else {}
                stage.update(stage='repair_validation_'+status.get('status','starting'),updated=time.time())
                save(root/'status.json',dict(**stage,heartbeat=time.time(),child_returncode=None,
                    spare_gpu_occupier_pid=reservation.process.pid if reservation.process else None))
                if not shared and time.monotonic()-renewed>240:lease();renewed=time.monotonic()
                time.sleep(5)
            receipt=json.loads((Path(config['launch_plan']).parent/'completion.json').read_text())
            if child.returncode or receipt['status']!='complete' or not receipt['runtime_cleaned'] or not receipt['owned_process_groups_cleaned']:
                raise RuntimeError('repair collection failed:'+str(receipt))
            stage.update(stage='collection_complete_pending_quality_review',updated=time.time())
            save(root/'collection-result.json',receipt)
        except BaseException as exc:
            error=repr(exc);save(root/'failure.json',dict(error=error,time=time.time(),
                stage=dict(stage),traceback=traceback.format_exc()))
        finally:
            cleanup_errors=[]
            shared=config.get('shared_node_parent')
            try:
                if child is not None and child.poll() is None:stop_process_group(child, timeout=90)
            except BaseException as exc:cleanup_errors.append(repr(exc))
            if shared and reservation is not None and shared_parent_live(shared):
                # The original owner still owns all shared Gym/NLP services and
                # its other collectors. Reserve only this lane after it exits.
                previous=os.environ.get('CUDA_VISIBLE_DEVICES')
                os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,config['spare_gpus']))
                try:
                    reservation.start()
                    save(root/'lane-result.json',dict(error=error,
                        child_returncode=child.returncode if child else None,
                        waiting_for_shared_owner=True,occupier_pid=reservation.process.pid))
                    stage.update(stage='lane_finished_reserved_waiting_for_shared_owner',updated=time.time())
                    while shared_parent_live(shared):
                        reservation.check()
                        time.sleep(5)
                except BaseException as exc:cleanup_errors.append(repr(exc))
                finally:
                    if previous is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
                    else:os.environ['CUDA_VISIBLE_DEVICES']=previous
            try:
                if reservation is not None:reservation.stop('repair_owner_finished')
            except BaseException as exc:cleanup_errors.append(repr(exc))
            # A shared-lane failure must not revoke the healthy node owner's
            # active lease. Its existing controller defers idle until GPU work
            # and this lane's subset reservation have both ended.
            idle=(subprocess.CompletedProcess([],0,stdout='shared owner retains node lease',stderr='')
                if shared and shared_parent_live(shared) else
                subprocess.run(control+['idle','--output',config['debug_output']],capture_output=True,text=True))
            save(root/'completion.json',dict(collection_complete=error is None,error=error,
                cleanup_errors=cleanup_errors,
                child_returncode=child.returncode if child else None,idle_requested=idle.returncode==0,
                idle_ack=idle.stdout[-2000:],idle_error=idle.stderr[-1000:],quality_gate_accepted=False,
                repaired_policy_gate_verified=quality_gate_validated,finished_at=time.time()))
            stopped.set();thread.join(timeout=2);os.close(original)
        if error or idle.returncode:raise SystemExit(1)


if __name__=='__main__':main()
