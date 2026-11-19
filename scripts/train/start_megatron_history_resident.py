"""Launch a fresh offline Megatron run on a completed resident Git task.

Run this from a pushed container-local checkout. The previous business must
have exited. Reuse the existing reservation controller and Git-task supervisor;
never signal host PIDs reported by PPU smi, or restart a DLC job.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import align_three_source_observations as io
from resident_rl_attempt import proc, exact_live, signal_pid
from run_git_task import checkout
from verify_idle_gpu_ownership import clients, verify as verify_resident_ownership

REPO = Path(__file__).resolve().parents[2]


def require_resident_entry(owner):
    """Do not terminate the last foreground command of an old DLC shell."""
    if owner.get('ppid') == 1:
        return  # Independently detached supervisor, not a shell's wait target.
    environment = (Path('/proc') / str(owner['pid']) / 'environ').read_bytes().split(b'\0')
    if b'ACTION_HEAD_RESIDENT_ENTRY=1' not in environment:
        raise RuntimeError('old foreground supervisor lacks a persistent platform entry; '
                           'handoff refused before releasing GPUs')


def previous_reservation(finished, oldspec, oldspec_path, *, from_resident,
                         resident_node, proof_path):
    """Require a terminal business and prove the holder before handing off."""
    if finished['phase'] not in ('succeeded', 'failed') or finished['business_returncode'] is None:
        raise RuntimeError('previous business has no real exit receipt')
    old = proc(finished['pid'])
    if proc(finished['business_pid']) is not None:
        raise RuntimeError('old business is still alive')
    if from_resident:
        if old is not None:
            raise RuntimeError('resident handoff requires the previous supervisor to have exited')
        proof = verify_resident_ownership(resident_node,
            REPO/'scripts/train/run_dlc_debug_reservation.py', proof_path)
        holder = proc(proof['occupier']['pid'])
        if not holder or holder['start_ticks'] != proof['occupier']['start_ticks']:
            raise RuntimeError('resident holder identity changed')
    else:
        expected = [str(Path(oldspec['checkout_root'])/'repo/scripts/train/run_git_task.py'),
                    '--spec', str(oldspec_path)]
        if not old or old['argv'][-3:] != expected:
            raise RuntimeError('old supervisor identity changed')
        ready = io.load(Path(oldspec['checkout_root'])/'standby_ready.json')
        holder = proc(ready['pid'])
        if not holder or holder['ppid'] != old['pid'] or 'gpu_occupier.py' not in ' '.join(holder['argv']):
            raise RuntimeError('previous reservation identity is unverified')
    return old, holder


ORPHAN_CONTEXT_LIMIT_MIB = 24 << 10


def container_gpu_users(proc_root=Path('/proc')):
    """PIDs in this container that hold an open PPU device file."""
    users = set()
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if any(os.readlink(fd).startswith('/dev/alixpu_ppu') for fd in (directory/'fd').iterdir()):
                users.add(int(directory.name))
        except OSError:
            continue
    return users


def orphan_contexts(baseline, hostname, holder_pid, *, users=container_gpu_users):
    """Driver contexts whose processes no longer exist in this container.

    A failed CUDA IPC publication left such contexts on the PPU driver after
    every importer exited. Tolerate exactly the audited set: no container
    process other than the verified holder may own a GPU, and each orphan
    must stay within the measured memory bound.
    """
    if baseline['hostname'] != hostname:
        raise RuntimeError('orphan GPU baseline belongs to another node')
    if users() != {holder_pid}:
        raise RuntimeError('a container process other than the holder owns a GPU')
    orphans = {(row['gpu'], int(row['pid'])) for row in baseline['orphans']}
    if any(row['mib'] > ORPHAN_CONTEXT_LIMIT_MIB for row in baseline['orphans']):
        raise RuntimeError('orphan GPU context exceeds the audited memory bound')
    return orphans


def split_clients(current, orphans):
    """Return client rows outside the audited orphan set (which may shrink)."""
    return [row for row in current if (row['gpu'], int(row['pid'])) not in orphans]


def remaining_rl_ray_workers(proc_root=Path('/proc')):
    """Find CPU or GPU Ray workers left by this project's business attempts.

    GPU file-descriptor checks miss orphan CPU-only TQ storage. Supervisors,
    stdout heartbeats and standby holders are intentionally ordinary Python
    processes, and stay alive throughout this read-only audit.
    """
    remaining = []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            title = (directory/'cmdline').read_bytes().split(b'\0', 1)[0]
            if not title.startswith(b'ray::'):
                continue
            environment = dict(item.split(b'=', 1) for item in
                (directory/'environ').read_bytes().split(b'\0') if b'=' in item)
            progress = environment.get(b'ACTION_HEAD_LEARNER_PROGRESS_DIR', b'').decode()
            repo = environment.get(b'REPO', b'').decode()
            if (not progress.startswith(str(io.ART) + '/')
                    or not progress.endswith('/train/learner_progress')
                    or not repo.startswith('/root/rl-code/')):
                continue
            fields = (directory/'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] in ('Z', 'X'):
                continue
            remaining.append(dict(pid=int(directory.name), start_ticks=fields[19],
                cwd=os.readlink(directory/'cwd'), repo=repo, progress=progress,
                title=title.decode().strip()))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return remaining


def validate_unstarted_handoff(output, rank, commit):
    """Resume only a recorded GPU guard failure before releasing its old holder."""
    launch=output/'launch'
    if io.exists(launch/f'owner_{rank:02d}.json') or io.exists(output/f'git_runtime/node_{rank:02d}'):
        raise RuntimeError('business was already launched; cannot resume handoff')
    intent=io.load(launch/f'intent_{rank:02d}.json')
    failure=io.load(launch/f'handoff_failure_{rank:02d}.json')
    if (failure['error'] != "RuntimeError('additional GPU clients; handoff refused')"
            or failure['time'] < intent['time'] or intent['commit'] != commit):
        raise RuntimeError('not the bound pre-release GPU guard failure')
    if intent.get('controller') is not None or not intent.get('old'):
        raise RuntimeError('recovery supports only the still-live Git standby owner')
    if not exact_live(intent['old']) or not exact_live(intent['holder']):
        raise RuntimeError('original standby owner/holder identity changed')
    return intent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--environment', type=Path, required=True)
    p.add_argument('--cache-root', type=Path)
    p.add_argument('--resident-node', type=Path,
                   help='Existing resident controller; omit for an ordinary Git-supervised task')
    p.add_argument('--node-rank', type=int, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--launcher-commit', help='Pinned separate Git controller revision; business remains --commit')
    p.add_argument('--resume-unstarted-handoff', action='store_true')
    p.add_argument('--detach', action='store_true',
                   help='Run the verified handoff independently of the terminal connection')
    p.add_argument('--from-resident-standby', action='store_true',
                   help='Use verified resident occupation after the previous task supervisor exited')
    p.add_argument('--orphan-gpu-baseline', type=Path,
                   help='Audited driver contexts left without any container process; tolerated exactly')
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument('--arguments-file', type=Path)
    target.add_argument('--task-spec', type=Path,
                        help='Run a prepared Git RL task with the same verified resident handoff')
    args = p.parse_args()
    if args.from_resident_standby and args.resident_node is None:
        p.error('--from-resident-standby requires --resident-node')
    if REPO.is_relative_to('/data/storage') or REPO.is_relative_to('/data/storage'):
        raise ValueError('launcher must run from a container-local Git checkout')
    revision = subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    if revision != (args.launcher_commit or args.commit):
        raise ValueError('launcher checkout must equal the explicitly requested commit')
    for path in [args.root,args.environment, args.cache_root or args.root/'cache']:
        if not path.is_absolute() or path.resolve().is_relative_to('/data/storage') or path.resolve().is_relative_to('/data/storage'):
            raise ValueError('code, environment and caches must be container local')
    for path in [args.previous_root,args.output,args.arguments_file or args.task_spec,
                 *([args.resident_node] if args.resident_node else []),
                 *([args.orphan_gpu_baseline] if args.orphan_gpu_baseline else [])]:
        if not path.resolve().is_relative_to(io.ART):
            raise ValueError('control and artifact paths must be on the authorized artifact disk')
    extra = io.load(args.arguments_file) if args.arguments_file else []
    prepared_spec = io.load(args.task_spec) if args.task_spec else None
    if prepared_spec is not None and (prepared_spec.get('kind') not in ('rl', 'diagnostic')
            or prepared_spec.get('commit') != args.commit
            or prepared_spec.get('checkout_root') != str(args.root)
            or prepared_spec.get('output_root') != str(args.output)):
        raise ValueError('prepared RL task differs from the bound checkout/output/commit')
    if not isinstance(extra,list) or not all(isinstance(x,str) for x in extra):
        raise ValueError('arguments-file must be a JSON argv list')
    if any(x in ('--output','--node-rank') or x.startswith(('--output=','--node-rank=')) for x in extra):
        raise ValueError('launcher owns output and node-rank')
    if args.output.resolve()==args.previous_root.resolve() or (args.output/'git_runtime'/f'node_{args.node_rank:02d}').exists():
        raise ValueError('use a fresh output, never reuse a previous attempt')
    node_rank=args.node_rank
    if node_rank < 0:
        raise ValueError('node-rank must be nonnegative')
    launch=args.output/'launch';launch.mkdir(parents=True,exist_ok=True)
    if args.detach:
        receipt = launch/f'preparation_owner_{node_rank:02d}.json'
        # Exclusive creation prevents an observation timeout from replaying a
        # possibly active handoff. The child still takes the existing lock.
        with receipt.open('x') as proof:
            with open('/proc/1/fd/1', 'w', buffering=1) as raw:
                child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                    *[arg for arg in sys.argv[1:] if arg != '--detach']],
                    cwd=REPO, stdin=subprocess.DEVNULL, stdout=raw,
                    stderr=subprocess.STDOUT, start_new_session=True)
            json.dump(dict(identity=proc(child.pid), commit=args.commit,
                           phase='preparing_or_handoff', time=time.time()), proof)
            proof.write('\n')
        print(json.dumps(dict(preparation_pid=child.pid, receipt=str(receipt))), flush=True)
        return
    with (launch/f'node_{node_rank:02d}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        intent=launch/f'intent_{node_rank:02d}.json'
        resumed=None
        if args.resume_unstarted_handoff:
            resumed=validate_unstarted_handoff(args.output,node_rank,args.commit)
            with (launch/f'unstarted_handoff_recovery_{node_rank:02d}.json').open('x') as proof:
                json.dump(dict(time=time.time(),identity=proc(os.getpid()),original_intent=resumed),proof)
        elif io.exists(intent) or io.exists(launch/f'owner_{node_rank:02d}.json'):
            raise RuntimeError('attempt already entered handoff; inspect its receipts, never replay')
        finished=io.load(args.previous_root/'git_runtime'/f'node_{node_rank:02d}'/'business_exit.json')
        oldspec_path=args.previous_root/'launch'/f'git_task_node_{node_rank:02d}.json'
        if not io.exists(oldspec_path):
            oldspec_path=args.previous_root/'launch'/'git_task.json'
        oldspec=io.load(oldspec_path)
        old, holder = previous_reservation(finished, oldspec, oldspec_path,
            from_resident=args.from_resident_standby, resident_node=args.resident_node,
            proof_path=launch/f'resident_ownership_{node_rank:02d}.json')
        if resumed and any(resumed[name]['pid'] != actual['pid'] or resumed[name]['start_ticks'] != actual['start_ticks']
                           for name,actual in [('old',old),('holder',holder)]):
            raise RuntimeError('standby identity changed during recovery')
        remaining = remaining_rl_ray_workers()
        io.write_json(launch/f'prelaunch_cpu_worker_audit_{node_rank:02d}.json',
            dict(time=time.time(), hostname=os.uname().nodename, remaining=remaining))
        if remaining:
            raise RuntimeError('previous RL Ray workers remain, including CPU-only storage; '
                               'holder preserved; inspect prelaunch_cpu_worker_audit before handoff')
        if old is not None:
            require_resident_entry(old)
        controller=None
        if args.resident_node is not None:
            resident=io.load(args.resident_node/'debug-status.json')
            controller=proc(resident['pid'])
            if not controller or str(args.resident_node) not in controller['argv'] or resident['hostname'] != os.uname().nodename:
                raise RuntimeError('resident controller identity changed')
        args.root.mkdir(parents=True,exist_ok=True)
        checkout('https://git.example.invalid/anonymous/closed_loop_3dgs_rl.git',args.commit,args.root/'repo')
        environment=args.root/'environment'
        if not environment.exists():environment.symlink_to(args.environment.resolve(),target_is_directory=True)
        if environment.resolve() != args.environment.resolve():raise ValueError('environment binding differs')
        env=dict(os.environ,OUTPUT_DIR=str(args.output),NODE_RANK=str(node_rank),CUDA_VISIBLE_DEVICES=','.join(map(str,range(16))),
            CUDA_DEVICE_MAX_CONNECTIONS='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
        for key in ['TMPDIR','XDG_CACHE_HOME','HF_HOME','HF_HUB_CACHE','HF_MODULES_CACHE','HF_DATASETS_CACHE',
                    'TRANSFORMERS_CACHE','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','CUDA_CACHE_PATH',
                    'PYTORCH_KERNEL_CACHE_PATH','TORCH_EXTENSIONS_DIR','PIP_CACHE_DIR']:
            directory=(args.cache_root or args.root/'cache')/key.lower()
            directory.mkdir(parents=True,exist_ok=True);env[key]=str(directory)
        # The RL recipe intentionally replaces generic cache variables with
        # per-attempt paths. Carry this explicit compiler-cache choice through
        # that layer; keep temporary files, IPC and policy transports isolated.
        env['ACTION_HEAD_TRAINING_PROCESS_COMPILE_CACHE_ROOT'] = str(
            (args.cache_root or args.root/'cache')/'process_compile')
        for key in ['RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE']:
            env.pop(key,None)
        for key in list(env):
            if key.lower().endswith('_proxy'):env.pop(key)
        spec=dict(repository='https://git.example.invalid/anonymous/closed_loop_3dgs_rl.git',commit=args.commit,
            checkout_root=str(args.root),output_root=str(args.output),kind='diagnostic',
            entry='scripts/train/train_megatron_history.py',arguments=extra+['--output',str(args.output/'train'),'--node-rank',str(node_rank)],
            node_count=int(extra[extra.index('--nnodes')+1]) if '--nnodes' in extra else 4,
            dependencies={'verl':dict(repository='https://git.example.invalid/local-user-03/verl.git',commit='b213f118f8df230bc97f68cdc56043623c6babdb')},
            standby_after_success=True,standby_after_failure=True)
        if prepared_spec is not None:
            spec = dict(prepared_spec, standby_after_success=True, standby_after_failure=True)
            if spec.get('kind') == 'rl' and io.load(spec['launch_plan'])['config'].get('text_backend') == 'megatron':
                preparation = args.output/'preparation'/f'node_{node_rank:02d}'
                # Keep the previous verified holder throughout CPU setup.
                subprocess.run([sys.executable, '-B', str(args.root/'repo/scripts/train/prepare_megatron_history.py'),
                    '--root', str(environment), '--output', str(preparation)], env=env, check=True)
                spec['megatron_environment_ready'] = str(preparation/'ready.json')
        spec_path=launch/f'git_task_node_{node_rank:02d}.json'
        io.write_json(spec_path,spec)
        if resumed is None:
            io.write_json(intent,dict(old=old,holder=holder,controller=controller,commit=args.commit,time=time.time()))
        command=[str(args.environment/'venv/bin/python'),'-B',str(args.root/'repo/scripts/train/run_git_task.py'),'--spec',str(spec_path)]
        child=None
        def mode(value):
            if controller is None:
                return
            if not exact_live(controller):raise RuntimeError('resident controller was replaced')
            subprocess.run([sys.executable,'-B',str(REPO/'scripts/train/run_dlc_debug_reservation.py'),value,
                '--output',str(args.resident_node),'--lease-seconds','7200'],check=True,timeout=200)
        try:
            orphans=set()
            if args.orphan_gpu_baseline is not None:
                baseline=io.load(args.orphan_gpu_baseline)
                orphans=orphan_contexts(baseline, os.uname().nodename, holder['pid'])
                io.write_json(launch/f'orphan_gpu_contexts_{node_rank:02d}.json',dict(
                    baseline=str(args.orphan_gpu_baseline),orphans=sorted(orphans),time=time.time()))
            before=split_clients(clients(), orphans)
            if len(before)!=16 or len({r['gpu'] for r in before})!=16 or len({r['pid'] for r in before})!=1:
                raise RuntimeError('additional GPU clients; handoff refused')
            if not exact_live(holder):
                raise RuntimeError('verified holder identity changed during preparation')
            mode('active')
            if old is not None:
                signal_pid(old,{1,os.getpid()})
            until=time.monotonic()+90
            while (old is not None and exact_live(old)) or exact_live(holder):
                if time.monotonic()>until:raise RuntimeError('old standby did not drain')
                time.sleep(1)
            if split_clients(clients(), orphans):raise RuntimeError('GPU clients remain after owned release')
            with open('/proc/1/fd/1','w',buffering=1) as raw:
                child=subprocess.Popen(command,cwd=args.root/'repo',env=env,stdin=subprocess.DEVNULL,
                    stdout=raw,stderr=subprocess.STDOUT,start_new_session=True)
            io.write_json(launch/f'owner_{node_rank:02d}.json',dict(identity=proc(child.pid),command=command,
                commit=args.commit,hostname=os.uname().nodename,time=time.time()))
            print(json.dumps(dict(node_rank=node_rank,owner_pid=child.pid,output=str(args.output))),flush=True)
        except BaseException as exc:
            io.write_json(launch/f'handoff_failure_{node_rank:02d}.json',dict(error=repr(exc),time=time.time(),automatic_retry=False))
            if child is None and (old is None or not exact_live(old)):
                if controller is not None:
                    mode('idle')
                else:
                    # The old Git owner is gone. Preserve the failed handoff
                    # and occupy without replaying either training attempt.
                    reservation=subprocess.Popen([sys.executable,'-B',str(REPO/'verl/scripts/train/action_head_sft_data/gpu_occupier.py'),
                        '--gpus',','.join(map(str,range(16))),'--require-gpu-count','16',
                        '--ready-file',str(args.root/'handoff_failure_standby.json')],env=env)
                    while reservation.poll() is None:
                        print(json.dumps(dict(event='handoff_failed_standby',time=time.time(),
                            hostname=os.uname().nodename,occupier_pid=reservation.pid,
                            ready=(args.root/'handoff_failure_standby.json').exists())),flush=True)
                        time.sleep(20)
            raise


if __name__=='__main__':
    main()
