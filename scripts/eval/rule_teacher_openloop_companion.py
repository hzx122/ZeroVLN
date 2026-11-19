"""Watch committed SFT checkpoints and reuse the frozen complete benchmark suite."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'scripts/train'), str(REPO/'scripts/eval'), str(REPO/'verl'), str(REPO/'vln_demo/src')]
from rule_teacher_training_handoff import load, save, sha, bind
from watch_v6_rl_local_credit import process_identity, same_process_alive
from dlc_workspace_stdout_heartbeat import task_heartbeat
from supervise_rule_teacher_collection import CACHE_KEYS


def suite_environment(runtime):
    runtime = Path(runtime)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        RULE_COLLECTION_RUNTIME_ROOT='/tmp/ahs',ACTION_HEAD_SUITE_RUNTIME_BASE='/tmp/ahs',
        CUDA_VISIBLE_DEVICES=','.join(map(str,range(16))),
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for key in CACHE_KEYS:
        path = runtime/key.lower()
        path.mkdir(parents=True,exist_ok=True)
        env[key] = str(path)
    return env


def waiting_reservation(owner):
    """A live controller alone is insufficient: its GPU worker must be live."""
    resident = load(Path(owner['debug_output'])/'debug-status.json')
    occupant = process_identity(resident.get('occupier_pid')) if resident.get('occupier_pid') else None
    if not (same_process_alive(owner['controller_identity'])
            and resident.get('mode') == 'idle' and resident.get('transition') == 'ready'
            and resident.get('gpu_count') == 16 and occupant
            and occupant['state'] not in ('Z', 'X')):
        raise RuntimeError('waiting companion lost resident GPU reservation')
    return resident


def stop_child(child, identity):
    if child is None or child.poll() is not None:
        return
    if not same_process_alive(identity):
        raise RuntimeError('evaluation child identity changed')
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=60)
    except subprocess.TimeoutExpired:
        if same_process_alive(identity):
            os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=20)


def retire_waiting_owner(config_path, rank, receipt_path):
    """Retire only the bound idle companion; its controller remains resident."""
    config = load(config_path)
    node = Path(config['root'])/f'watchers/node{rank}'
    start = load(node/'start.json')
    state = load(node/'status.json')
    owner = config['resident_owners'][rank]
    identity = start['identity']
    if (start['config'] != bind(config_path) or os.uname().nodename != owner['hostname']
            or start['hostname'] != owner['hostname'] or state.get('child') is not None
            or state['state'] not in ('waiting_checkpoint_with_resident_gpu_reservation', 'upstream_failed_with_idle_gpu_reservation')
            or not same_process_alive(identity)
            or identity['pid'] == owner['controller_identity']['pid']):
        raise ValueError('companion is not the verified waiting owner')
    argv = Path(f"/proc/{identity['pid']}/cmdline").read_bytes().split(b'\0')
    if str(config_path).encode() not in argv:
        raise ValueError('companion command/config binding changed')
    before = waiting_reservation(owner)
    receipt_path = Path(receipt_path)
    if receipt_path.exists() or receipt_path.with_suffix('.intent.json').exists():
        raise ValueError('retirement already attempted; inspect receipt instead of replaying')
    fd = os.pidfd_open(identity['pid'])
    suspended = False
    try:
        if not same_process_alive(identity):
            raise ValueError('companion identity changed before signal')
        # The frozen old owner has no retirement-request hook. Stop only its
        # thread group briefly so checkpoint publication cannot race this check.
        # The independent platform controller/heartbeat keeps running.
        signal.pidfd_send_signal(fd, signal.SIGSTOP)
        suspended = True
        deadline = time.monotonic()+5
        while process_identity(identity['pid'])['state'] not in ('T', 't'):
            if time.monotonic() >= deadline:
                raise TimeoutError('owner did not enter the retirement inspection stop')
            time.sleep(.02)
        state = load(node/'status.json')
        children = []
        for task in Path(f"/proc/{identity['pid']}/task").iterdir():
            children.extend((task/'children').read_text().split())
        if (state.get('child') is not None or children
                or state['state'] not in ('waiting_checkpoint_with_resident_gpu_reservation', 'upstream_failed_with_idle_gpu_reservation')):
            raise ValueError('owner left waiting phase; resume it without retirement')
        before = waiting_reservation(owner)
        save(receipt_path.with_suffix('.intent.json'), dict(config=bind(config_path), rank=rank,
            identity=identity, controller=owner['controller_identity'], state=state,
            resident=before, time=time.time(), reason='authorized repair continuation in new attempt'))
        signal.pidfd_send_signal(fd, signal.SIGTERM)
    finally:
        if suspended:
            signal.pidfd_send_signal(fd, signal.SIGCONT)
        os.close(fd)
    deadline = time.monotonic()+120
    while same_process_alive(identity):
        if time.monotonic() >= deadline:
            raise TimeoutError('waiting owner has not finished its cleanup; do not launch a replacement')
        time.sleep(1)
    resident = waiting_reservation(owner)
    result = dict(stopped_identity=identity, controller_alive=True, resident=resident,
                  idle_restored=True, rank=rank, time=time.time())
    save(receipt_path, result)
    return result


def checkpoint_published(config, step, checkpoint):
    """Read the existing trainer's publication marker; never sample a partial export."""
    if config.get('checkpoint_publication', 'sft_index') == 'rl_tracker':
        if step == 0:
            from action_head_benchmark_suite import checkpoint_identity
            checkpoint_identity(checkpoint)
            return True
        tracker = Path(config['checkpoint_root'])/'latest_checkpointed_iteration.txt'
        from dlc_control_io_retry import retry_estale
        return (retry_estale(tracker.exists, str(tracker)) and
                int(retry_estale(tracker.read_text, str(tracker))) >= step)
    index = Path(config['checkpoint_index'])
    published = load(index) if index.exists() else {}
    if step not in published.get('steps', []):
        return False
    if str(Path(published['checkpoint_root']).resolve()) != config['checkpoint_root']:
        raise ValueError('published checkpoint root differs')
    return True


def checkpoint_plan(config, checkpoint, output):
    """Reuse the exact selected populations; validate every new head/query pair."""
    import action_head_benchmark_suite as suite
    evaluation_name=config.get('evaluation_checkpoint_name',checkpoint.name)
    if (output/'plan.json').exists():
        # A new repair attempt can prepare/validate reusable results on CPU
        # before the four resident owners start. Never replay an old owner.
        existing = load(output/'plan.json')
        suite.verify(existing)
        ckpts = existing['config']['checkpoints']
        if (existing['output'] != str(output) or len(ckpts) != 1
                or ckpts[0]['path'] != str(checkpoint)
                or existing.get('runtime_repo') != str(REPO)
                or existing.get('source_sha256') != load(config['base_plan'])['source_sha256']):
            raise ValueError('prepared checkpoint plan differs from companion binding')
        if (config.get('reuse_companion_root') or config.get('completed_case_sources', {}).get(evaluation_name)) and not (output/'reuse-audit.json').exists():
            raise ValueError('repair plan has no completed reuse audit')
        return existing
    base = load(config['base_plan'])
    for path, expected in base['input_sha256'].items():
        if sha(path) != expected:
            raise ValueError('frozen evaluation input changed: '+path)
    cfg = copy.deepcopy(base['config_template'])
    if config.get('evaluation_repeats', 1) > 1:
        cfg['evaluation_repeat_index'] = config['evaluation_repeat_index']
        cfg['evaluation_repeats'] = config['evaluation_repeats']
    brain = config['slow_model']
    if cfg.get('closedloop_protocol'):
        # The provenance of the checkpoint, never the companion purpose,
        # decides whether a trained brain export belongs to this model.
        brain = suite.checkpoint_identity(checkpoint).get('qwen_dir') or brain
    checkpoint_name = config.get('evaluation_checkpoint_name', checkpoint.name)
    cfg['checkpoints'] = [dict(name=checkpoint_name,path=str(checkpoint),
        brain=brain,training_config=config['training_config'])]
    cfg = suite.normalize_config(cfg)
    ckpt = cfg['checkpoints'][0]
    input_sha = dict(base['input_sha256'])
    for name in ('report.json','action_head/fast_slow_action_head_config.json',
                 'action_head/fast_slow_action_head.pt','dualvln_query/dualvln_query_embeddings.pt'):
        path = checkpoint/name
        input_sha[str(path)] = sha(path)
    input_sha[config['training_config']] = sha(config['training_config'])
    metadata = {}
    for role in ('brain','backbone'):
        metadata.update(suite.model_weight_metadata(ckpt[role]))
        for path in Path(ckpt[role]).iterdir():
            if path.is_file() and path.suffix in ('.json','.jinja','.model','.tiktoken'):
                input_sha[str(path)] = sha(path)
    stages = [dict(row,checkpoint=checkpoint_name) for row in base['stages']]
    plan = dict(schema=suite.SCHEMA,output=str(output),config=cfg,stages=stages,
        populations=base['populations'],input_sha256=input_sha,
        code_sha256=base['code_sha256'],source_sha256=base['source_sha256'],
        runtime_repo=str(REPO),model_weight_metadata=metadata,
        source_provenance=base['source_provenance'])
    if cfg.get('closedloop_protocol'):
        from unified_closedloop_eval import seal_plan
        seal_plan(plan, config.get('purpose', 'companion'))
    plan['plan_sha256'] = suite.digest({k: v for k, v in plan.items() if k != 'plan_sha256'})
    suite.verify(plan)
    output.mkdir(parents=True,exist_ok=False)
    completed_source = config.get('completed_case_sources', {}).get(evaluation_name)
    if completed_source:
        if config.get('reuse_companion_root'):
            raise ValueError('only one audited reuse source may own a fresh checkpoint')
        from rule_teacher_companion_repair import seed_completed_case_sources, restore_completed_node_receipts
        seed_completed_case_sources(plan, completed_source)
        restore_completed_node_receipts(plan)
    if config.get('reuse_companion_root'):
        from rule_teacher_companion_repair import seed_checkpoint
        previous = Path(config['reuse_companion_root'])/'evaluation'/checkpoint.name/'plan.json'
        seed_checkpoint(plan, previous, allowed_code_changes=config['repair_code_allowlist'],
                        rerun_benchmarks=config.get('repair_rerun_benchmarks', []),
                        navigation_only=config.get('repair_navigation_only', False))
    # Publish only after seeding has completed: other nodes wait on this file.
    save(output/'plan.json',plan)
    return plan


def evaluation_owner_argv_matches(argv, config_path):
    if str(config_path).encode() in argv:
        return True
    if (b'--plan' in argv and argv.index(b'--plan')+1<len(argv)
            and any(x.endswith(b'/resume_rl_companion.py') for x in argv)
            and b'node' in argv):
        item=load(Path(os.fsdecode(argv[argv.index(b'--plan')+1])))
        return Path(item['root'])/'run/watcher-config.json'==Path(config_path)
    if (b'run' in argv and b'--root' in argv and argv.index(b'--root')+1<len(argv)
            and any(x.endswith(b'/resume_rloo_local_harness.py') for x in argv)):
        spec_path = Path(os.fsdecode(argv[argv.index(b'--root')+1]))/'spec.json'
    elif (b'--spec' in argv and argv.index(b'--spec') + 1 < len(argv)
            and any(x.endswith((b'/start_unified_companion_resident.py',b'/unified_companion_task.py',b'/unified_closedloop_eval.py')) for x in argv)):
        spec_path = Path(os.fsdecode(argv[argv.index(b'--spec')+1]))
    else:
        return False
    spec = load(spec_path)
    if Path(spec['root'])/'run/watcher-config.json' == Path(config_path):
        return True
    if spec.get('variants') and Path(config_path).is_relative_to(Path(spec['root'])/'jobs'):
        record = load(config_path)
        queue = load(Path(spec['root'])/'queue.json')
        spec_binding = bind(spec_path)
        return (record.get('queue_spec') == spec_binding and queue['spec'] == spec_binding
            and any(row['config'] == str(config_path) for row in queue['jobs']))
    return False


def retire_evaluation_owner(config_path, rank, receipt_path, reason='authorized scope correction'):
    """Authorized scope correction; let the bound owner clean up its own suite."""
    config=load(config_path); node=Path(config['root'])/f'watchers/node{rank}'
    start=load(node/'start.json'); owner=config['resident_owners'][rank]
    identity=start['identity']; receipt_path=Path(receipt_path)
    if (start['config']!=bind(config_path) or start['hostname']!=owner['hostname']
            or os.uname().nodename!=owner['hostname']
            or not same_process_alive(identity)
            or not same_process_alive(owner['controller_identity'])
            or identity['pid']==owner['controller_identity']['pid']):
        raise ValueError('evaluation retirement identity/config/controller differs')
    argv=Path(f"/proc/{identity['pid']}/cmdline").read_bytes().split(b'\0')
    # Resident starters call watch in-process; their immutable start receipt
    # binds the config, while argv names the enclosing experiment spec.
    if not evaluation_owner_argv_matches(argv, config_path):
        raise ValueError('evaluation owner command binding differs')
    intent=receipt_path.with_suffix('.intent.json')
    if receipt_path.exists() or intent.exists():
        raise ValueError('retirement was already attempted; inspect instead of replaying')
    save(intent,dict(config=bind(config_path),rank=rank,identity=identity,
        controller=owner['controller_identity'],state=load(node/'status.json'),
        time=time.time(),reason=reason))
    fd=os.pidfd_open(identity['pid'])
    try:
        if not same_process_alive(identity):
            raise ValueError('evaluation owner identity changed before termination')
        signal.pidfd_send_signal(fd,signal.SIGTERM)
    finally:
        os.close(fd)
    deadline=time.monotonic()+180
    while same_process_alive(identity):
        if time.monotonic()>=deadline:
            raise TimeoutError('evaluation cleanup still running; do not start replacement')
        time.sleep(1)
    resident=waiting_reservation(owner)
    result=dict(stopped_identity=identity,controller_alive=True,idle_restored=True,
        resident=resident,rank=rank,time=time.time(),reason=reason)
    save(receipt_path,result)
    return result


def run(config_path, rank):
    config=load(config_path)
    root=Path(config['root']);node=root/f'watchers/node{rank}'
    node.mkdir(parents=True,exist_ok=True)
    lock=(node/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (node/'start.json').exists():
        raise ValueError('companion owner already started; inspect original attempt')
    owner=config['resident_owners'][rank]
    if os.uname().nodename!=owner['hostname'] or not same_process_alive(owner['controller_identity']):
        raise ValueError('evaluation resident identity differs')
    me=process_identity(os.getpid())
    save(node/'start.json',dict(identity=me,hostname=os.uname().nodename,config=bind(config_path)))
    save(node/'status.json',dict(state='waiting_checkpoint_with_resident_gpu_reservation',
        heartbeat=time.time(),completed_steps=[],child=None))
    stream=open(f'/proc/{owner["controller_identity"]["pid"]}/fd/1','w',buffering=1)
    stop=threading.Event()
    def progress():
        state=load(node/'status.json')
        resident=load(Path(owner['debug_output'])/'debug-status.json')
        observed=dict(completed_steps=state['completed_steps'],checkpoint=state.get('checkpoint'),
            child=state.get('child'),child_live=same_process_alive(state['child']) if state.get('child') else None)
        if state.get('checkpoint') and state.get('child'):
            stage_status=Path(state.get('suite_output', str(root/'evaluation'/f"global_step_{state['checkpoint']:06d}")))/f'nodes/{rank}/status.json'
            if stage_status.exists():
                observed['suite']=load(stage_status)
        return dict(heartbeat=state['heartbeat'],stage=state['state'],mode=resident['mode'],
            occupier_pid=resident.get('occupier_pid'),progress=observed)
    args=SimpleNamespace(controller_pid=me['pid'],controller_start_ticks=me['start_ticks'],output=node)
    heartbeat=threading.Thread(target=task_heartbeat,args=(args,),kwargs=dict(stop_event=stop,
        read_status=progress,write_receipt=lambda value:save(node/'stdout-heartbeat.json',value),
        emit=lambda value:print(json.dumps(dict(value,evaluation_node=rank)),file=stream,flush=True)),daemon=True)
    heartbeat.start()
    control=[config['python'],'-B',owner['debug_control']]
    last_active_ack = None
    def mode(value, *, renew=False):
        nonlocal last_active_ack
        from dlc_reservation_request import request_mode
        if not same_process_alive(owner['controller_identity']):
            raise RuntimeError('reservation controller identity changed')
        renewal = (dict(acknowledgement=last_active_ack, controller=owner['controller_identity'])
                   if renew else None)
        resident=request_mode(owner['debug_output'],value,write=save,renewal=renewal)
        if (value=='active' and (resident.get('transition')!='ready'
                or resident.get('occupier_pid') is not None
                or resident.get('pid')!=owner['controller_identity']['pid'])):
            raise RuntimeError('active GPU release not verified')
        if value == 'active':
            last_active_ack = resident
        if value=='idle':
            deadline=time.monotonic()+180
            while time.monotonic()<deadline:
                resident=load(Path(owner['debug_output'])/'debug-status.json')
                if resident.get('mode')=='idle' and resident.get('transition')=='ready' and resident.get('occupier_pid'):
                    return resident
                time.sleep(2)
            raise TimeoutError('resident GPU reservation did not become ready')
    def check_queue_failure():
        if config.get('queue_root'):
            from unified_companion_queue import failed_nodes
            if failed_nodes(Path(config['queue_root']), len(config['resident_owners'])):
                raise RuntimeError('another queue owner failed; clean this owned evaluation')
    child=None;child_identity=None;completed=[]
    def interrupted(*_):raise KeyboardInterrupt('companion termination requested')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        from rule_teacher_repeat_companion import work_items, verify_historical, summarize_checkpoint
        results = []
        for step, repeat, unit, checkpoint, output, historical in work_items(config):
            tag = unit['evaluation_checkpoint_name'] + (f'-repeat{repeat}' if config.get('evaluation_repeats', 1) > 1 else '')
            if historical:
                verify_historical(config, step, historical)
                save(node/f'{tag}-reused.json', dict(source=historical, checkpoint=step, repeat=repeat))
                continue
            while True:
                check_queue_failure()
                if checkpoint_published(unit, step, checkpoint):
                    break
                resident=waiting_reservation(owner)
                upstream_failed=(Path(unit['training_root'])/'failure.json').exists()
                save(node/'status.json',dict(state='upstream_failed_with_idle_gpu_reservation' if upstream_failed else
                    'waiting_checkpoint_with_resident_gpu_reservation',heartbeat=time.time(),
                    completed_steps=completed,checkpoint=step,repeat=repeat,suite_output=str(output),child=None))
                time.sleep(10)
            if rank==0:
                checkpoint_plan(unit,checkpoint,output)
            while not (output/'plan.json').exists():
                check_queue_failure()
                save(node/'status.json',dict(state='waiting_frozen_evaluation_plan_with_resident_gpu_reservation',
                    heartbeat=time.time(),completed_steps=completed,checkpoint=step,repeat=repeat,suite_output=str(output),child=None))
                time.sleep(5)
            activated=False
            lease=None
            try:
                mode('active');activated=True
                from dlc_reservation_request import ActiveLeaseRenewer
                lease=ActiveLeaseRenewer(lambda: mode('active', renew=True),
                    lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
                env=suite_environment(Path('/tmp/ahs')/root.name/f'node{rank}/{tag}')
                with (node/f'{tag}.log').open('a') as log:
                    unified = unit.get('closedloop_protocol') or load(config['base_plan'])['config_template'].get('closedloop_protocol')
                    entry = 'unified_closedloop_eval.py' if unified else 'action_head_benchmark_suite.py'
                    command = 'run-round' if unified else 'run'
                    child=subprocess.Popen([config['python'],'-B',str(REPO/'scripts/eval'/entry),
                        command,'--plan',str(output/'plan.json'),'--rank',str(rank)],env=env,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                child_identity=process_identity(child.pid)
                while child.poll() is None:
                    check_queue_failure()
                    save(node/'status.json',dict(state='evaluating_complete_benchmark_suite',heartbeat=time.time(),
                        completed_steps=completed,checkpoint=step,repeat=repeat,suite_output=str(output),child=child_identity))
                    lease.check()
                    time.sleep(5)
                result=load(output/f'nodes/{rank}/completion.json') if (output/f'nodes/{rank}/completion.json').exists() else {}
                save(node/f'{tag}-result.json',dict(returncode=child.returncode,
                    evaluation_valid=result.get('evaluation_valid') is True,completion=result,finished_at=time.time()))
                results.append(load(node/f'{tag}-result.json'))
                if config.get('pooled_attempts') or repeat == config.get('evaluation_repeats', 1):
                    completed.append(step)
            finally:
                try:
                    stop_child(child, child_identity)
                finally:
                    if lease is not None:
                        lease.close()
                    if activated:
                        resident=mode('idle')
                        save(node/f'{tag}-idle-return.json',dict(ok=True,resident=resident))
            child=None;child_identity=None
            if rank == 0 and (config.get('pooled_attempts') or repeat == config.get('evaluation_repeats', 1)):
                while summarize_checkpoint(config, step) is None:
                    check_queue_failure()
                    waiting_reservation(owner)
                    save(node/'status.json', dict(state='waiting_repeat_aggregation', heartbeat=time.time(),
                        completed_steps=completed, checkpoint=step, repeat=repeat, child=None))
                    time.sleep(5)
        save(node/'completion.json',dict(complete=True,evaluation_valid=all(r['evaluation_valid'] for r in results),
            completed_steps=completed,finished_at=time.time(),resident=mode('idle')))
    except BaseException as error:
        save(node/'failure.json',dict(error=repr(error),completed_steps=completed,time=time.time()))
        raise
    finally:
        try:
            stop_child(child, child_identity)
        finally:
            try:
                mode('idle')
            finally:
                stop.set()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--rank',type=int,required=True,choices=range(4))
    parser.add_argument('--retire-evaluation-owner',type=Path)
    parser.add_argument('--retire-waiting-owner', type=Path,
                        help='write retirement proof after stopping only the verified waiting owner')
    args=parser.parse_args()
    if args.retire_evaluation_owner:
        print(json.dumps(retire_evaluation_owner(args.config,args.rank,args.retire_evaluation_owner)),flush=True)
    elif args.retire_waiting_owner:
        print(json.dumps(retire_waiting_owner(args.config,args.rank,args.retire_waiting_owner)),flush=True)
    else:
        run(args.config,args.rank)
