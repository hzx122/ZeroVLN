"""Keep the local 5000 console available and record native-process exits.

Run with the console launch manifest. The supervisor does not import Torch or
touch render, gym, or the brain service. Stop the supervisor before a planned
console shutdown; it leaves the current console process running on SIGTERM.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.request import urlopen


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ')[1][0] != 'Z'
    except (OSError, IndexError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    output = manifest.parent
    (output / 'supervisor.pid').write_text(f'{os.getpid()}\n')
    stop = False

    def terminate(_signal, _frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    records = json.loads(manifest.read_text())
    entry = next(item for item in records if item['service'] == 'console')
    pid = entry['pid']
    child = None
    last_status = {}
    log_path = output / 'supervisor.log'
    try:
        while not stop:
            try:
                with urlopen('http://127.0.0.1:5000/status', timeout=1) as reply:
                    last_status = json.load(reply)
            except (OSError, ValueError):
                pass
            if child is not None:
                exit_code = child.poll()
                running = exit_code is None
            else:
                exit_code = 'unknown: process started before supervisor'
                running = alive(pid)
            if running:
                time.sleep(3)
                continue
            with log_path.open('a') as log:
                log.write(f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} '
                          f'console pid={pid} exited status={exit_code}; '
                          f'last_status={last_status.get("status")}; '
                          f'driver={last_status.get("driver")}\n')
            if stop:
                break
            command = list(entry['command'])
            for flag, key in (('--scene', 'scene'), ('--episode_id', 'episode_id')):
                if flag in command and last_status.get(key):
                    command[command.index(flag) + 1] = last_status[key]
            env = os.environ.copy()
            env.update(entry['environment'])
            if last_status.get('driver'):
                env['DEMO_DEFAULT_DRIVER'] = last_status['driver']
            with open(entry['log'], 'ab') as output_log:
                output_log.write(f'\nSupervisor restarting console after exit {exit_code}.\n'.encode())
                child = subprocess.Popen(command, cwd=entry['cwd'], env=env,
                                         stdout=output_log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            pid = child.pid
            entry.update(pid=pid, command=command)
            manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
            (output / 'console.pid').write_text(f'{pid}\n')
            time.sleep(3)
    finally:
        (output / 'supervisor.pid').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
