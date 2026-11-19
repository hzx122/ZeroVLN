#!/usr/bin/env python3
"""Publish credential-free DLC status snapshots from the submitting workstation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import atomic_json


def read_snapshot(path, job_id, max_age=180):
    try:
        data = json.loads(Path(path).read_text())
        if (data['job_id'] == job_id and 0 <= time.time()-data['observed_at'] <= max_age
                and data['status'] != 'Unknown'):
            return data
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def ensure_publisher(output_root, upstream_job, evaluation_job):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root/'upstream_bridge_config.json', dict(
        upstream_job=upstream_job, evaluation_job=evaluation_job))
    env = {k:v for k,v in os.environ.items()
           if k.lower() not in ('http_proxy', 'https_proxy', 'all_proxy')}
    env.update(PYTHONDONTWRITEBYTECODE='1')
    env.pop('UPSTREAM_STATUS_SNAPSHOT', None)
    log = root/'logs/upstream_status_bridge.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as sink:
        subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                          '--output-root', str(root)], env=env,
                         stdin=subprocess.DEVNULL, stdout=sink, stderr=subprocess.STDOUT,
                         start_new_session=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args()
    root = args.output_root
    lock = (root/'upstream_bridge.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return  # Existing publisher reads updated binding on its next poll.
    sys.path[:0] = [str(Path(__file__).resolve().parents[4]/'verl/dlc'),
                    '/workspace/dlc_cli']
    from _dlc_common import get_dlc_client
    from alibabacloud_pai_dlc20201203.models import GetJobRequest
    client = get_dlc_client()
    atomic_json(root/'upstream_bridge_process.json', {'pid': os.getpid(), 'started_at': time.time()})
    while True:
        try:
            config = json.loads((root/'upstream_bridge_config.json').read_text())
            job_id = config['upstream_job']
            body = client.get_job(job_id, GetJobRequest()).body.to_map()
            atomic_json(root/'upstream_status_snapshot.json', dict(
                job_id=job_id, display_name=body.get('DisplayName'), status=body['Status'],
                reason_code=body.get('ReasonCode'), reason_message=body.get('ReasonMessage'),
                observed_at=time.time(), source='credentialed_workstation'))
            status = client.get_job(config['evaluation_job'], GetJobRequest()).body.status
            if status in ('Stopped', 'Failed', 'Succeeded', 'Completed', 'Cancelled'):
                return
        except Exception as exc:
            atomic_json(root/'upstream_bridge_error.json', {'time': time.time(), 'error': type(exc).__name__})
        time.sleep(30)


if __name__ == '__main__':
    main()
