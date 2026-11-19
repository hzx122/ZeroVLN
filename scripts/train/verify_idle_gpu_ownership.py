"""Bind host GPU PIDs to an owned container occupier through a release cycle.

Reuses the resident controller's active/idle transitions. Unknown GPU clients
are never stopped. A previously proven, still-live process can be read-only
verified without another transition.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def identity(pid):
    row = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
    if row[0] == 'Z':
        raise RuntimeError('owned process is a zombie')
    return dict(pid=pid, start_ticks=row[19])


def clients():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True, timeout=20)
    return [dict(gpu=line.split(',')[0].strip(), pid=int(line.split(',')[1]))
            for line in output.splitlines() if line.strip()]


def matches(proof, state, current):
    return (proof.get('hostname') == state['hostname']
        and proof.get('controller') == identity(state['pid'])
        and proof.get('occupier') == identity(state['occupier_pid'])
        and len(current) == 16 and len({row['gpu'] for row in current}) == 16
        and {row['gpu'] for row in current} == set(proof.get('gpu_uuids', []))
        and {row['pid'] for row in current} == set(proof.get('host_pids', [])))


def wait_idle(node):
    deadline = time.monotonic()+150
    while time.monotonic() < deadline:
        state = json.loads((node/'debug-status.json').read_text())
        if state['mode'] == 'idle' and state['transition'] == 'ready' and time.time()-state['heartbeat'] < 30:
            ready = json.loads((node/'waiting_gpu_ready.json').read_text())
            if ready['pid'] == state['occupier_pid'] and ready['gpus'] == list(range(16)):
                return state
        time.sleep(2)
    raise TimeoutError('idle occupation not ready')


def verify(node, controller, proof_path):
    node, proof_path = Path(node), Path(proof_path)
    state = wait_idle(node)
    current = clients()
    if proof_path.exists():
        proof = json.loads(proof_path.read_text())
        if matches(proof, state, current):
            return proof
    for pid, needle in ((state['pid'], 'run_dlc_debug_reservation.py'),
                        (state['occupier_pid'], 'gpu_occupier.py')):
        assert needle in (Path('/proc')/str(pid)/'cmdline').read_text()
    before_controller, before_occupier = identity(state['pid']), identity(state['occupier_pid'])
    assert len(current) == 16 and len({r['gpu'] for r in current}) == 16
    assert len({r['pid'] for r in current}) == 1, 'unknown additional GPU clients'
    released = False
    def transition(mode):
        subprocess.run([sys.executable, '-B', str(controller), mode, '--output', str(node),
                        '--lease-seconds', '300'], check=True, capture_output=True, text=True, timeout=160)
    try:
        released = True
        transition('active')
        deadline = time.monotonic()+30
        while clients():
            if time.monotonic() > deadline:
                raise RuntimeError('GPU clients remain after owned release; ownership unproven')
            time.sleep(1)
        assert identity(state['pid']) == before_controller
    finally:
        if released:
            transition('idle')
    after = wait_idle(node)
    current = clients()
    assert identity(after['pid']) == before_controller
    assert len(current) == 16 and len({r['gpu'] for r in current}) == 16
    assert len({r['pid'] for r in current}) == 1, 'unknown clients after reoccupation'
    assert identity(after['occupier_pid']) != before_occupier
    proof = dict(hostname=after['hostname'], controller=before_controller,
        occupier=identity(after['occupier_pid']), host_pids=sorted({r['pid'] for r in current}),
        gpu_uuids=sorted({r['gpu'] for r in current}), verified_at=time.time(),
        method='owned_release_zero_clients_then_owned_reoccupation', gpu_count=16)
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    proof_path.write_text(json.dumps(proof, indent=2)+'\n')
    return proof


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--node', type=Path, required=True)
    p.add_argument('--controller', type=Path, required=True)
    p.add_argument('--proof', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(verify(a.node, a.controller, a.proof)))
