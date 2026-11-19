"""Container bootstrap for the shared checkpoint watcher and resident GPU owner."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'scripts/train'), str(REPO/'scripts/eval')]
from rule_teacher_training_handoff import load, save, sha
from watch_v6_rl_local_credit import process_identity, same_process_alive
from dlc_control_io_retry import retry_estale


def dependencies(spec):
    if os.environ.get('GPU_DIRECT_RUN') == '1':
        navigation = Path(spec['navigation_repo']).resolve()
        os.environ['ACTION_HEAD_NAVIGATION_REPO'] = str(navigation)
        os.environ['RYNNBRAIN_NAV_DIR'] = str(navigation)
        os.environ['ACTION_HEAD_RENDER_REPO'] = str(REPO/'third_party/abot_renderer')
        from unified_closedloop_eval import verify_dependencies
        verify_dependencies(['sage-3d', 'r2r', 'rxr'])
        return
    """Git checkouts plus explicitly version-bound compiled renderer inputs."""
    local = REPO.parent
    for name, item in spec['dependencies'].items():
        if name == 'ABot-Navigation':
            continue  # CPU renderer is versioned in the main repository.
        root = local/name
        if not root.exists():
            subprocess.run(['git', 'clone', '--quiet', '--no-checkout', '--no-hardlinks',
                item['mirror'], str(root)], check=True)
            subprocess.run(['git', '-C', str(root), 'remote', 'set-url', 'origin',
                item['repository']], check=True)
            # Finish the initial worktree before the reused cleanliness check.
            subprocess.run(['git', '-C', str(root), 'checkout', '--quiet',
                '--detach', item['commit']], check=True)
        # The Git mirror clone already contains the pinned object. Do not
        # require a redundant public-network fetch to execute that checkout.
        if (subprocess.check_output(['git','-C',str(root),'remote','get-url','origin'],text=True).strip()
                != item['repository'] or
                subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
                != item['commit']):
            raise ValueError('dependency Git identity differs: '+name)
    render = REPO/'third_party/abot_renderer'
    nav = local/'RynnBrain/navigation'
    data = nav/'data'
    if not data.exists():
        data.symlink_to(spec['navigation_data'], target_is_directory=True)
    os.environ['ACTION_HEAD_RENDER_REPO'] = str(render)
    os.environ['ACTION_HEAD_NAVIGATION_REPO'] = str(nav)
    from unified_closedloop_eval import verify_dependencies
    verify_dependencies(['sage-3d','r2r','rxr'])


def run(spec_path, rank):
    from unified_closedloop_eval import git_runtime, prepare_watch, watch
    spec = load(spec_path)
    root = Path(spec['root'])
    if not root.is_relative_to('/data/zerovln/artifacts'):
        raise ValueError('companion artifacts must be on the authorized data disk')
    actual = git_runtime(True)
    if actual['commit'] != spec['commit'] or not 0 <= rank < spec['nodes']:
        raise ValueError('companion Git/rank differs from its task specification')
    node = root/'nodes'/os.uname().nodename
    node.mkdir(parents=True, exist_ok=True)
    lock = (node/'bootstrap.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (node/'bootstrap-start.json').exists():
        raise ValueError('existing attempt must be inspected, not automatically replayed')
    save(node/'bootstrap-start.json', dict(rank=rank,git=actual,
        identity=process_identity(os.getpid()),time=time.time()))
    sleeper = subprocess.Popen(['sleep','99999999'])
    control = REPO/'scripts/train/run_dlc_debug_reservation.py'
    controller = subprocess.Popen([sys.executable,'-B',str(control),'run',
        '--output',str(node/'node-debug'),'--sleep-pid',str(sleeper.pid)])
    owner = dict(hostname=os.uname().nodename,debug_control=str(control),
        debug_output=str(node/'node-debug'),controller_identity=process_identity(controller.pid))
    try:
        deadline = time.monotonic()+180
        while True:
            if controller.poll() is not None:
                raise RuntimeError('resident controller exited during initialization')
            status = node/'node-debug/debug-status.json'
            if retry_estale(status.exists,str(status)):
                state = load(status)
                if state.get('mode')=='idle' and state.get('transition')=='ready':
                    break
            if time.monotonic() >= deadline:
                raise TimeoutError('resident reservation initialization timed out')
            time.sleep(2)
        dependencies(spec)
        save(root/f'bootstrap/node{rank}.json',dict(owner=owner,git=actual,rank=rank,
            ready=True,time=time.time()))
        ready = [root/f'bootstrap/node{i}.json' for i in range(spec['nodes'])]
        deadline = time.monotonic()+1800
        while not all(retry_estale(p.exists,str(p)) for p in ready):
            if any(retry_estale((root/f'bootstrap/failure-node{i}.json').exists,
                    str(root/f'bootstrap/failure-node{i}.json')) for i in range(spec['nodes'])):
                raise RuntimeError('another companion node failed bootstrap; retain idle reservation')
            if time.monotonic() >= deadline:
                raise TimeoutError('not all companion nodes completed bootstrap')
            time.sleep(5)
        config_path = root/'run/watcher-config.json'
        if rank == 0:
            rows = [load(p) for p in ready]
            if any(row['git'] != actual for row in rows):
                raise ValueError('companion node Git checkout differs')
            binding = load(spec['binding'])
            binding['resident_owners'] = [row['owner'] for row in rows]
            save(root/'resolved-binding.json',binding)
            prepare_watch(Path(spec['selection']),root/'resolved-binding.json',
                root/'run',spec['purpose'])
        while not retry_estale(config_path.exists,str(config_path)):
            if time.monotonic() >= deadline:
                raise TimeoutError('companion plan preparation timed out')
            time.sleep(5)
        code = watch(config_path,rank)
        save(node/'business-exit.json',dict(returncode=code,time=time.time()))
    except Exception as exc:
        traceback.print_exc()
        save(node/'business-exit.json',dict(returncode=1,error=f'{type(exc).__name__}: {exc}',time=time.time()))
        save(root/f'bootstrap/failure-node{rank}.json',dict(rank=rank,error=str(exc),time=time.time()))
    # Keep the platform process and idle controller after completion or failure.
    # The watcher owns its own service cleanup and restores this reservation.
    while same_process_alive(owner['controller_identity']):
        time.sleep(20)
    raise RuntimeError('resident controller exited; inspect the original failure receipt')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--rank',type=int,required=True)
    args=p.parse_args()
    return run(args.spec,args.rank)


if __name__=='__main__':
    raise SystemExit(main())
