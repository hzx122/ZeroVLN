"""Reuse the DLC heartbeat and owned GPU reservation around suite phases."""
import json
import importlib.util
import hashlib
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

_REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_REPO / 'scripts/train'),
               str(_REPO / 'verl/scripts/eval/qwen36_4query_ckpt_sweep')]
from dlc_workspace_stdout_heartbeat import identity, workspace_heartbeat
from run_dlc_debug_reservation import transition_reservation
from v6_rl_checkpoint_companion import WaitingGpuReservation
from watch_checkpoints import stop_process_group
from common import atomic_json


def install_control_io(root, controller):
    """Reopen this suite's control receipts with its frozen retry helper."""
    root = Path(root).absolute()
    if getattr(controller, '_suite_control_io_root', None) == str(root):
        return
    helper_path = _REPO / 'scripts/train/dlc_control_io_retry.py'
    spec = importlib.util.spec_from_file_location('_suite_frozen_control_io', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    def owns(path):
        return Path(path).absolute().is_relative_to(root)
    def retry(operation, path, name):
        label = json.dumps(dict(path=str(path), operation=name, pid=os.getpid(),
                                host=__import__('socket').gethostname()))
        return helper.retry_estale(operation, label)
    for name in ('read_text', 'exists', 'stat'):
        original = getattr(Path, name)
        def wrapped(path, *args, _original=original, _name=name, **kwargs):
            if not owns(path):
                return _original(path, *args, **kwargs)
            return retry(lambda: _original(path, *args, **kwargs), path, _name)
        setattr(Path, name, wrapped)
    original_glob = Path.glob
    def glob(path, *args, **kwargs):
        if not owns(path):
            return original_glob(path, *args, **kwargs)
        return iter(retry(lambda: list(original_glob(path, *args, **kwargs)), path, 'glob'))
    Path.glob = glob
    # Preserve the supervisor's bounded receipt size, but don't swallow ESTALE
    # or malformed/permission errors as an empty, apparently absent receipt.
    if hasattr(controller, 'read_status_json'):
        def read_status(path):
            def read():
                with Path(path).open('rb') as stream:
                    body = stream.read(2 * 1024 * 1024 + 1)
                if len(body) > 2 * 1024 * 1024:
                    raise ValueError('oversized control receipt: ' + str(path))
                value = json.loads(body)
                if not isinstance(value, dict):
                    raise ValueError('control receipt must be an object: ' + str(path))
                return value
            try:
                return retry(read, path, 'open_read_json')
            except FileNotFoundError:
                return {}
        controller.read_status_json = read_status
    original_atomic = atomic_json
    for module in (controller, sys.modules[__name__], sys.modules.get('common'),
                   sys.modules.get('v6_rl_checkpoint_companion')):
        if module is None:
            continue
        for name in ('atomic_json', 'save'):
            original = getattr(module, name, None)
            if original is None:
                continue
            # The old supervisor save uses .partial; reuse the PID-bound
            # atomic writer so the existing lost-temp guard stays exact.
            writer = original_atomic if name == 'save' else original
            def write(path, *args, _original=writer, **kwargs):
                if not owns(path):
                    return _original(path, *args, **kwargs)
                return helper.retry_atomic_json(lambda: _original(path, *args, **kwargs), path)
            setattr(module, name, write)
    controller._suite_control_io_root = str(root)
    print(json.dumps(dict(event='suite_control_io_installed', root=str(root),
        helper=str(helper_path), sha256=hashlib.sha256(helper_path.read_bytes()).hexdigest(),
        pid=os.getpid())), flush=True)


class SuiteRuntime:
    def __init__(self, repo, node, plan_path):
        self.node = node
        self.phase_file = node / 'runtime-phase.json'
        self.mode = None
        self.heartbeat = None
        current = identity(os.getpid())
        self.controller = {key: current[key] for key in ('pid', 'start_ticks')}
        self.reservation = WaitingGpuReservation(dict(waiting_gpu_reservation=dict(
            enabled=os.environ.get('GPU_DIRECT_RUN') != '1', gpu_count=16, python=sys.executable,
            script=str(repo / 'verl/scripts/train/action_head_sft_data/gpu_occupier.py'),
            startup_timeout=120)), node / 'reservation', stop_process_group)
        self.plan_path = plan_path
        self.previous_sigterm = None

    def __enter__(self):
        self.previous_sigterm = signal.getsignal(signal.SIGTERM)
        def interrupted(*_):
            raise KeyboardInterrupt('suite termination requested; clean owned GPU reservation')
        signal.signal(signal.SIGTERM,interrupted)
        self.node.joinpath('reservation').mkdir(exist_ok=True)
        self.heartbeat = subprocess.Popen([sys.executable, '-B', __file__,
            str(self.node), str(self.controller['pid']), self.controller['start_ticks']],
            stdin=subprocess.DEVNULL)
        try:
            self.phase('verifying_frozen_inputs', 'idle')
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def phase(self, phase, mode, stage=None):
        # Publish the transition before a potentially slow occupier startup.
        value = dict(heartbeat=time.time(), mode='transitioning', phase=phase,
                     stage=stage, controller=self.controller, occupier_pid=None)
        atomic_json(self.phase_file, value)
        self.mode, transition = transition_reservation(self.reservation, mode, self.mode)
        self.reservation.check()
        value.update(mode=self.mode, transition=transition, heartbeat=time.time(),
                     occupier_pid=self.reservation.process.pid if self.reservation.process else None)
        atomic_json(self.phase_file, value)

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is not None:
                # Only reserve when real owned services have already released
                # CUDA; the existing transition never kills other GPU clients.
                self.phase('failed', 'idle')
        finally:
            try:
                self.reservation.stop('suite_controller_exit')
            finally:
                if self.heartbeat is not None:
                    self.heartbeat.terminate()
                    self.heartbeat.wait(timeout=10)
                if self.previous_sigterm is not None:
                    signal.signal(signal.SIGTERM,self.previous_sigterm)


def heartbeat(node, controller):
    install_control_io(node.parents[1], sys.modules[__name__])
    latest = {}
    def read():
        phase = json.loads((node / 'runtime-phase.json').read_text())
        if phase['controller'] != controller:
            raise ValueError('phase belongs to another controller generation')
        # Active phases get their freshness from the real coordinator receipt.
        status_path = node / 'status.json'
        if phase['mode'] == 'active' and status_path.exists():
            status = json.loads(status_path.read_text())
            phase['heartbeat'] = status['updated']
            phase['state'] = status['state']
            phase['child'] = identity(status['child_pid']) if status.get('child_pid') else None
        if phase.get('occupier_pid'):
            phase['occupier'] = identity(phase['occupier_pid'])
        latest.clear()
        latest.update(phase)
        return phase
    def emit(event):
        event.update(phase=latest.get('phase'), stage=latest.get('stage'),
                     state=latest.get('state'), child=latest.get('child'),
                     occupier=latest.get('occupier'))
        print(json.dumps(event, ensure_ascii=False), flush=True)
    workspace_heartbeat(node, node, controller, interval=20,
        read_status=read, write_receipt=lambda event: atomic_json(node / 'stdout-heartbeat.json', event), emit=emit)


if __name__ == '__main__':
    heartbeat(Path(sys.argv[1]), dict(pid=int(sys.argv[2]), start_ticks=sys.argv[3]))
