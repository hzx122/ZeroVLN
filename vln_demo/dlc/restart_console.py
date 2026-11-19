#!/usr/bin/env python3
"""Request a PID-bound restart of port 5000 from the running supervisor."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import uuid


def restart(run_dir, timeout=300):
    status = json.loads((Path(run_dir) / 'status.json').read_text())
    if status['state'] != 'ready':
        raise ValueError('service must be ready before console maintenance')
    attempt = Path(status['run_dir'])
    pid = status['processes']['console_5000']['pid']
    request = {'hostname': status['hostname'], 'expected_pid': pid}
    with (attempt / '.console-maintenance.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        destination = attempt / 'restart-console.request.json'
        if destination.exists():
            raise ValueError('console maintenance request already pending')
        temp = attempt / ('.console-request-' + uuid.uuid4().hex)
        temp.write_text(json.dumps(request))
        os.replace(temp, destination)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = attempt / 'restart-console.result.json'
            if result.exists():
                receipt = json.loads(result.read_text())
                if receipt.get('old_pid') == pid and receipt.get('state') == 'ready':
                    return receipt
                if receipt.get('state') == 'rejected' and not destination.exists():
                    raise RuntimeError(str(receipt))
            time.sleep(1)
        raise TimeoutError('maintenance has not completed; inspect request/result and task before retrying')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=300)
    args = parser.parse_args()
    print(json.dumps(restart(args.run_dir, args.timeout), ensure_ascii=False, indent=2))
