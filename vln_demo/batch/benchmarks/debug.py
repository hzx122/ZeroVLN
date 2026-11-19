"""Persistent DLC development container, with key-only SSH and periodic status."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import threading
import time


def serve(config):
    output = Path(config['run_dir'])
    output.mkdir(parents=True, exist_ok=True)
    public_key = Path(config['ssh_authorized_keys']).read_text().strip()
    if not public_key.startswith('ssh-ed25519 ') or '\n' in public_key:
        raise ValueError('exactly one Ed25519 public key required')
    local_parent = Path('/root/vln-batch-debug')
    local_parent.mkdir(mode=0o700, exist_ok=True)
    local = Path(tempfile.mkdtemp(prefix='session-', dir=local_parent))
    auth = local / 'authorized_keys'
    auth.write_text(public_key + '\n')
    auth.chmod(0o600)
    host_key = local / 'host_ed25519'
    subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(host_key)], check=True)
    port = int(config.get('ssh_port', 22222))
    ssh_config = local / 'sshd_config'
    ssh_config.write_text('\n'.join([
        f'Port {port}', 'ListenAddress 0.0.0.0', f'HostKey {host_key}',
        f'AuthorizedKeysFile {auth}', f'PidFile {local / "sshd.pid"}',
        'PermitRootLogin prohibit-password', 'AllowUsers root', 'PubkeyAuthentication yes',
        'PasswordAuthentication no', 'KbdInteractiveAuthentication no', 'UsePAM no',
        'StrictModes yes', 'UseDNS no', 'X11Forwarding no', 'AllowAgentForwarding no',
        'AllowTcpForwarding no', 'Subsystem sftp internal-sftp', 'LogLevel VERBOSE', '']))
    Path('/run/sshd').mkdir(mode=0o755, exist_ok=True)
    subprocess.run(['/usr/sbin/sshd', '-t', '-f', str(ssh_config)], check=True)
    stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.set())
    with (output / 'sshd.log').open('a') as log:
        process = subprocess.Popen(['/usr/sbin/sshd', '-D', '-e', '-f', str(ssh_config)],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        connection = dict(hostname=socket.gethostname(), port=port,
                          host_public_key=host_key.with_suffix('.pub').read_text().strip(),
                          authentication='public-key-only', pid=process.pid)
        (output / 'debug-connection.json').write_text(json.dumps(connection, indent=2)+'\n')
        started = time.monotonic()
        try:
            while not stopped.is_set():
                if process.poll() is not None:
                    raise RuntimeError(f'sshd exited {process.returncode}')
                try:
                    with socket.create_connection(('127.0.0.1',port), timeout=2) as client:
                        listening = client.recv(100).startswith(b'SSH-')
                except OSError:
                    listening = False
                heartbeat = dict(state='debug_ready' if listening else 'starting', hostname=socket.gethostname(),
                                 ssh_port=port, uptime_seconds=int(time.monotonic()-started), time=time.time())
                print('DLC_DEBUG_HEALTH ' + json.dumps(heartbeat), flush=True)
                (output / 'debug-status.json').write_text(json.dumps(heartbeat)+'\n')
                stopped.wait(30)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
