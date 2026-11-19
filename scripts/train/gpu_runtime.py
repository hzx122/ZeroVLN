"""Platform-independent process supervision and configuration for GPU jobs."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from dlc_control_io_retry import bounded, retry_estale, retry_atomic_json

REPO = Path(__file__).resolve().parents[2]
CACHES = ('TMPDIR', 'XDG_CACHE_HOME', 'HF_HOME', 'TRITON_CACHE_DIR',
          'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH', 'PYTORCH_KERNEL_CACHE_PATH',
          'TORCH_EXTENSIONS_DIR', 'MPLCONFIGDIR', 'NUMBA_CACHE_DIR')


def expand(value, strict=True):
    if isinstance(value, dict):
        return {key: expand(item, strict) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item, strict) for item in value]
    if not isinstance(value, str):
        return value
    result = os.path.expandvars(value)
    if strict and re.search(r'\$\{?[A-Za-z_]', result):
        raise ValueError('Set the environment variable in: ' + value)
    return result


def load(path, strict=True):
    path = Path(path)
    with bounded(time.monotonic() + 60):
        return expand(json.loads(retry_estale(path.read_text, str(path))), strict)


def save(path, value):
    path = Path(path)
    if path.suffix in ('.yaml', '.yml'):
        from omegaconf import OmegaConf
        serialized = OmegaConf.to_yaml(OmegaConf.create(value))
    else:
        serialized = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    def write():
        temporary.write_text(serialized)
        temporary.replace(path)
    with bounded(time.monotonic() + 60):
        retry_atomic_json(write, path)


def artifact_root(path):
    root = Path(path).resolve()
    base = Path(os.environ.get('RL_ARTIFACT_ROOT', '/data/closed_loop_3dgs_rl')).resolve()
    if root == base or not root.is_relative_to(base):
        raise ValueError('Use a new experiment directory under RL_ARTIFACT_ROOT: '+str(base))
    return root


def environment(output, overrides=None):
    runtime = Path(os.environ.get('GPU_CACHE_ROOT', '/tmp/closed-loop-gpu')) / hashlib.sha256(str(output).encode()).hexdigest()[:12]
    env = dict(os.environ)
    for name in CACHES:
        env[name] = str(runtime / name.lower())
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        RULE_COLLECTION_RUNTIME_ROOT=str(runtime / 'collection'),
        PYTHONPATH=os.pathsep.join([str(REPO/p) for p in
            ('scripts/train', 'scripts/eval', 'verl', 'vln_demo/src')] + [env.get('PYTHONPATH', '')]))
    env.update({key: str(value) for key, value in (overrides or {}).items()})
    return env


def execute(command, env, receipt, *, cwd=REPO):
    """Preserve real stdout/exit status; signal only this invocation's group."""
    for key in CACHES:
        if env.get(key):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
    receipt = Path(receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.with_suffix('.started.json').open('x') as stream:
        json.dump(dict(supervisor_pid=os.getpid(), command=command, started_at=time.time()), stream)
    try:
        child = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True)
    except OSError as error:
        save(receipt, dict(phase='failed', command=command, error=str(error),
                           returncode=127, finished_at=time.time()))
        return 127
    state = dict(pid=child.pid, command=command, started_at=time.time(), phase='running')
    save(receipt, state)
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(20):
            print(json.dumps(dict(event='process_alive', pid=child.pid, time=time.time(),
                note='process liveness, not training progress')), flush=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    def forward(signum, frame):
        if child.poll() is None:
            os.killpg(child.pid, signum)
    previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        code = child.wait()
    finally:
        stop.set()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    save(receipt, dict(state, phase='succeeded' if code == 0 else 'failed',
                       returncode=code, finished_at=time.time()))
    return code if code >= 0 else 128 - code
