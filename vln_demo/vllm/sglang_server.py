"""Supervise the selected TP8 shell in the independent 16-GPU service."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.request import ProxyHandler, build_opener


def command(config):
    return ['bash', str(Path(__file__).resolve().parents[2] / 'scripts/serve/vllm' / config['launcher'])]


def checked_info(info, config):
    expected = dict(served_model_name=config['model_id'], tp_size=config['tp_size'],
                    context_length=config['context_length'], disable_radix_cache=False,
                    speculative_algorithm=None, enable_cache_report=True)
    for key, value in expected.items():
        if key not in info or info[key] != value:
            raise ValueError(f'effective SGLang setting mismatch: {key}')


def run(config, run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    env = dict(os.environ)
    local = tempfile.TemporaryDirectory(prefix='vln-sglang-', dir='/tmp')
    for key in ('TMPDIR', 'XDG_CACHE_HOME', 'HF_HOME', 'TRITON_CACHE_DIR',
                'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH', 'PYTORCH_KERNEL_CACHE_PATH',
                'TORCH_EXTENSIONS_DIR', 'FLASHINFER_WORKSPACE_BASE', 'SGLANG_CACHE_DIR'):
        path = Path(local.name)/key.lower()
        path.mkdir()
        env[key] = str(path)
    env['NO_PROXY'] = env['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0,::1,' + env.get('NO_PROXY', '')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['VLN_SGLANG_LOCAL_CACHE'] = local.name
    env['VLN_SGLANG_PYTHON'] = sys.executable
    endpoint = f'http://127.0.0.1:{config["port"]}'
    opener = build_opener(ProxyHandler({}))
    def get(path):
        with opener.open(endpoint+path, timeout=10) as response:
            return json.load(response)
    def status(state, **extra):
        record = dict(state=state, updated_at=datetime.now(timezone.utc).isoformat(),
                      hostname=socket.gethostname(), model_id=config['model_id'],
                      cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES'),
                      port=config['port'], tp_size=config['tp_size'], **extra)
        temporary = run_dir/'.status.json.tmp'
        temporary.write_text(json.dumps(record, indent=2)+'\n')
        temporary.replace(run_dir/'status.json')
        print('MODEL_HEALTH '+json.dumps(record), flush=True)
    process = None
    try:
        with (run_dir/'server.log').open('ab', buffering=0) as log:
            process = subprocess.Popen(command(config), env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            started = time.monotonic()
            ready = False
            while not stop.is_set():
                if process.poll() is not None:
                    raise RuntimeError(f'SGLang exited with {process.returncode}; see server.log')
                try:
                    models = get('/v1/models')
                    if config['model_id'] not in [m['id'] for m in models.get('data', [])]:
                        raise ValueError('served model identity mismatch')
                    info = get('/server_info')
                    checked_info(info, config)
                    if not ready:
                        (run_dir/'server-info.json').write_text(json.dumps(info, indent=2)+'\n')
                    ready = True
                    status('ready', pid=process.pid, radix_cache=True, mtp=False)
                except OSError as exc:
                    # Request stalls under load are reported, not used to kill a live server.
                    status('degraded' if ready else 'starting', pid=process.pid, error=str(exc))
                    if not ready and time.monotonic()-started > 3600:
                        raise TimeoutError('SGLang startup exceeded one hour') from exc
                stop.wait(30)
    except Exception as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
        local.cleanup()
    status('stopped')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    profiles = json.loads(Path(__file__).with_name('sglang_profiles.json').read_text())
    parser.add_argument('--profile', required=True, choices=tuple(profiles))
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = profiles[args.profile]
    if args.dry_run:
        print(json.dumps(command(config), indent=2))
    else:
        if args.run_dir is None:
            parser.error('--run-dir is required')
        run(config, args.run_dir)


if __name__ == '__main__':
    main()
