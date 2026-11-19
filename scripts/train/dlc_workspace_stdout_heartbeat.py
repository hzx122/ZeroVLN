#!/usr/bin/env python3
"""Expose actual workspace liveness in the existing DLC stdout stream.

The controller owns stdout's tee pipe. Bind its PID/start ticks before opening
that stream; never reuse a PID after controller or container replacement.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,start_ticks=fields[19],state=fields[0])
    except (OSError,IndexError):
        return None


def same(record):
    current=identity(record['pid'])
    return bool(current and current['start_ticks']==record['start_ticks'] and current['state']!='Z')


def write(path,value):
    stage=path.with_name(path.name+'.writing')
    stage.write_text(json.dumps(value,ensure_ascii=False)+'\n');os.replace(stage,path)


class HeartbeatDiskWorker:
    """One daemon owns all potentially blocking shared-filesystem operations.

    Latest-value slots stay bounded even when a read or write never returns.
    The stdout loop never joins this worker or waits for a filesystem lock.
    """
    def __init__(self, read_status, write_receipt, interval):
        self.read_status, self.write_receipt = read_status, write_receipt
        self.interval = interval
        self.snapshot = None
        self.read_error = self.write_error = None
        self.latest_event = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, name='heartbeat-disk-io', daemon=True)

    def run(self):
        while not self.stop.is_set():
            try:
                status = self.read_status()
                if not isinstance(status, dict) or 'heartbeat' not in status or 'mode' not in status:
                    raise ValueError('invalid workspace status')
                if not math.isfinite(float(status['heartbeat'])):
                    raise ValueError('invalid workspace heartbeat time')
                self.snapshot = (dict(status), time.time())
                self.read_error = None
            except Exception as exc:
                self.read_error = repr(exc)
            event = self.latest_event
            if event is not None:
                try:
                    self.write_receipt(event)
                    self.write_error = None
                except Exception as exc:
                    self.write_error = repr(exc)
            self.stop.wait(self.interval)


def workspace_heartbeat(debug_output, output, controller, *, interval=20.,
                        stop_event=None, read_status=None, write_receipt=None, emit=None):
    """Keep platform stdout live while truthfully flagging stale disk status."""
    stop = stop_event if stop_event is not None else threading.Event()
    disk = HeartbeatDiskWorker(
        read_status if read_status is not None else
            lambda: json.loads((debug_output/'debug-status.json').read_text()),
        write_receipt if write_receipt is not None else
            lambda event: write(output/'status.json', event), interval)
    if emit is None:
        emit = lambda event: print(json.dumps(event, ensure_ascii=False), flush=True)
    disk.thread.start()
    try:
        while not stop.is_set():
            actual = identity(controller['pid'])
            if not (actual and actual['start_ticks'] == controller['start_ticks']
                    and actual['state'] != 'Z'):
                raise RuntimeError('bound DLC controller exited; stop heartbeat')
            now = time.time()
            event = dict(event='workspace_liveness', time=now,
                hostname=os.uname().nodename, controller_identity_live=True,
                controller=actual, pid=os.getpid(), gpu_mode='unknown',
                status_unavailable=True,
                meaning='actual process liveness; stale disk status is not progress or success')
            snapshot = disk.snapshot
            if snapshot is not None:
                status, read_at = snapshot
                age = max(0., now-float(status['heartbeat']))
                stale = age > 60. or now-read_at > 60.
                event.update(controller_heartbeat_age_seconds=round(age, 1),
                    status_read_age_seconds=round(max(0., now-read_at), 1),
                    status_unavailable=stale, status_stale=stale,
                    reported_gpu_mode=status['mode'],
                    gpu_mode='unknown' if stale else status['mode'],
                    occupier_pid=None if stale else status.get('occupier_pid'))
            if disk.read_error:
                event.update(status_unavailable=True, status_read_error=disk.read_error,
                             gpu_mode='unknown', occupier_pid=None)
            if disk.write_error:
                event['receipt_write_error'] = disk.write_error
            event['controller_io_wait'] = actual['state'] == 'D'
            event['no_progress_warning'] = event['status_unavailable'] or event['controller_io_wait']
            emit(event)
            disk.latest_event = event
            stop.wait(interval)
    finally:
        # A daemon stuck in kernel I/O must not prevent this process exiting.
        disk.stop.set()


def task_heartbeat(args, *, interval=20., stop_event=None, read_status=None,
                   write_receipt=None, emit=None):
    """Keep bound process stdout independent of shared-disk errors and stalls."""
    controller = dict(pid=args.controller_pid, start_ticks=args.controller_start_ticks)
    if not same(controller):
        raise RuntimeError('bound task controller is not live')

    def read_progress():
        stage = json.loads(args.phase_file.read_text())
        if stage['controller'] != controller or stage['hostname'] != os.uname().nodename:
            raise RuntimeError('task phase receipt identity differs')
        progress = {}
        if args.progress_root:
            for p in sorted((args.progress_root/'data/parts').glob('*/heartbeat.json')):
                row = json.loads(p.read_text())
                progress[p.parent.name] = dict(processed=row['processed'], time=row['time'])
            # Never recurse through the potentially millions of rendered frames.
            for name in ('s2','s1','checks/dlc/smoke/s2','checks/dlc/smoke/s1'):
                directory = args.progress_root/name
                paths = list(directory.glob('checkpoint-*/trainer_state.json'))
                paths += [directory/'trainer_state.json',directory/'training_complete.json']
                for p in paths:
                    if p.exists():
                        progress[str(p.relative_to(args.progress_root))] = dict(
                            global_step=json.loads(p.read_text()).get('global_step'), modified=p.stat().st_mtime)
        return dict(heartbeat=time.time(), mode=stage['gpu_mode'],
                    stage=stage['stage'], occupier_pid=stage.get('occupier_pid'),
                    progress=progress)

    stop = stop_event if stop_event is not None else threading.Event()
    disk = HeartbeatDiskWorker(read_status or read_progress,
        write_receipt or (lambda event: write(args.output/'status.json', event)), interval)
    emit = emit or (lambda event: print(json.dumps(event, ensure_ascii=False), flush=True))
    last_progress, changed_at = None, time.monotonic()
    disk.thread.start()
    try:
        while not stop.is_set() and same(controller):
            now = time.time()
            event = dict(event='dualvln_liveness', time=now, hostname=os.uname().nodename,
                controller=controller, controller_identity_live=True, pid=os.getpid(),
                stage='unknown', gpu_mode='unknown', occupier_pid=None,
                status_unavailable=True,
                meaning='liveness only; stale disk data is not progress or optimizer success')
            snapshot = disk.snapshot
            if snapshot is not None:
                status, read_at = snapshot
                age = max(0., now-read_at)
                stale = age > 60. or bool(disk.read_error)
                event.update(status_read_age_seconds=round(age, 1), status_stale=stale,
                    status_unavailable=stale, reported_stage=status['stage'],
                    progress=status['progress'])
                if not stale:
                    event.update(stage=status['stage'], gpu_mode=status['mode'],
                                 occupier_pid=status.get('occupier_pid'))
                    signature = (status['stage'], json.dumps(status['progress'], sort_keys=True))
                    if signature != last_progress:
                        last_progress, changed_at = signature, time.monotonic()
            if disk.read_error:
                event['status_read_error'] = disk.read_error
            if disk.write_error:
                event['receipt_write_error'] = disk.write_error
            pid = event['occupier_pid']
            occupier = identity(pid) if pid else None
            event['occupier_live'] = bool(occupier and occupier['state'] != 'Z')
            elapsed = time.monotonic()-changed_at
            event.update(seconds_without_recorded_progress=round(elapsed),
                         no_progress_warning=event['status_unavailable'] or elapsed > 600)
            emit(event)
            disk.latest_event = event
            stop.wait(interval)
    finally:
        disk.stop.set()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--debug-output',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--controller-pid',type=int)
    parser.add_argument('--controller-start-ticks')
    parser.add_argument('--phase-file',type=Path)
    parser.add_argument('--progress-root',type=Path)
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--container-stdout',action='store_true',
        help='Write directly to container PID 1 stdout, after checking the controller identity')
    args=parser.parse_args();root=args.output
    root.mkdir(parents=True,exist_ok=True)
    if args.controller_pid and args.debug_output:
        if not args.controller_start_ticks or args.phase_file or args.launch:
            parser.error('bound workspace heartbeat requires PID/start ticks only')
        controller = dict(pid=args.controller_pid, start_ticks=args.controller_start_ticks)
        if not same(controller):
            raise RuntimeError('workspace controller identity is not live')
        workspace_heartbeat(args.debug_output, root, controller)
        return
    if args.controller_pid:
        if not args.controller_start_ticks or not args.phase_file or args.debug_output or args.launch:
            parser.error('task heartbeat requires PID/start ticks/phase file and inherited stdout')
        task_heartbeat(args)
        return
    if not args.debug_output:
        parser.error('debug heartbeat requires --debug-output')
    if args.launch:
        record=root/'launch.json'
        if record.exists():
            old=json.loads(record.read_text())
            if same(old): print(json.dumps(dict(already_live=True,**old)));return
            raise RuntimeError('old heartbeat receipt exists; use a new generation directory')
        controller=json.loads((args.debug_output/'debug-start.json').read_text())
        actual=identity(controller['pid'])
        if not actual or controller['hostname']!=os.uname().nodename:
            raise RuntimeError('controller is not local/live')
        cmd=Path(f'/proc/{controller["pid"]}/cmdline').read_text().replace('\x00',' ')
        if 'run_dlc_debug_reservation.py run' not in cmd or str(args.debug_output) not in cmd:
            raise RuntimeError('controller process identity differs')
        stdout_pid = 1 if args.container_stdout else controller['pid']
        with open(f'/proc/{stdout_pid}/fd/1','w',buffering=1) as log:
            child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),
                '--debug-output',str(args.debug_output),'--output',str(root)],
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        record_value=identity(child.pid);record_value.update(controller_identity=actual,
            stdout_pid=stdout_pid,started_at=time.time())
        write(record,record_value);print(json.dumps(record_value));return
    lock=(root/'heartbeat.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    controller=identity(json.loads((args.debug_output/'debug-start.json').read_text())['pid'])
    workspace_heartbeat(args.debug_output, root, controller)


if __name__=='__main__': main()
