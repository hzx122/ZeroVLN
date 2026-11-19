"""Preflight and reap only this collection's services, reusing eval cleanup."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "eval/qwen36_4query_ckpt_sweep"))
from watch_checkpoints import stop_vllm  # noqa: E402
from bootstrap_loop import atomic_json  # noqa: E402


def gpu_snapshot(ids: str) -> dict:
    try:
        result = subprocess.run([
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"], capture_output=True, text=True)
    except FileNotFoundError:
        ppu = Path("/usr/local/PPU_SDK/ppu-smi/bin/ppu-smi")
        if not ppu.is_file():
            return {"returncode": 127, "csv": "", "error": "GPU monitor unavailable"}
        result = subprocess.run([str(ppu)], capture_output=True, text=True)
        return {"returncode": result.returncode, "csv": "", "raw": result.stdout,
                "error": result.stderr.strip(), "format": "ppu-smi-table"}
    # This PPU-SMI build rejects comma-separated --id lists. Query is read-only;
    # filter the result rather than interpreting its 'No devices' as success.
    selected = set(ids.split(","))
    rows = [line for line in result.stdout.splitlines()
            if line.split(",")[0].strip() in selected]
    return {"returncode": result.returncode, "csv": "\n".join(rows),
            "error": result.stderr.strip()}


def preflight(ports: list[int], gpu_ids: str, output: Path) -> None:
    for port in sorted(set(ports)):
        with socket.socket() as sock:
            # Match the HTTP servers: a previous completed run can leave
            # TIME_WAIT connections, which do not mean a listener owns the port.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
    snapshot = gpu_snapshot(gpu_ids)
    if snapshot["returncode"] or len(snapshot["csv"].splitlines()) != len(gpu_ids.split(",")):
        raise RuntimeError(f"cannot inspect requested GPUs: {snapshot}")
    for line in snapshot["csv"].splitlines():
        fields = [v.strip() for v in line.split(",")]
        if float(fields[2]) > 1024 or float(fields[1]) > 0:
            raise RuntimeError(f"requested GPU is already in use: {line}")
    atomic_json(output, {"ok": True, "ports": ports, "gpu_ids": gpu_ids,
                         "gpu": snapshot, "time": time.time()})


def stop_services(pids: list[int], pid_files: list[Path], output: Path,
                  gpu_ids: str, grace_seconds: float = 10) -> None:
    if grace_seconds < 0:
        raise ValueError("grace_seconds must be nonnegative")
    # Capture descendants *before* their launcher exits. Some multiprocessing
    # children create a fresh group and are not covered by killpg alone.
    pids = set(pids)
    for path in pid_files:
        if path.exists():
            pids.add(int(path.read_text().strip()))
    if any(pid <= 1 or pid == os.getpid() for pid in pids):
        raise ValueError("refusing invalid service PID")
    owned = {}
    for pid in pids:
        try:
            parent = psutil.Process(pid)
            for process in [parent, *parent.children(recursive=True)]:
                owned[process.pid] = process
        except psutil.NoSuchProcess:
            pass
    started = time.monotonic()
    files = []
    for pid in pids:
        path = output.parent / f"cleanup_{pid}.pid"
        path.write_text(str(pid))
        files.append(path)
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    # psutil signals verify process identity, avoiding signals to reused PIDs.
    for process in owned.values():
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    # Collection work is already complete. Reap TERM-ignoring owned workers
    # before the legacy group cleanup's 60-second grace period, while retaining
    # its additional process-group coverage for descendants not in the snapshot.
    _, alive = psutil.wait_procs(list(owned.values()), timeout=grace_seconds)
    alive = [p for p in alive if p.status() != psutil.STATUS_ZOMBIE]
    for process in alive:
        process.kill()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(stop_vllm, files))
    _, alive = psutil.wait_procs(alive, timeout=10)
    survivors = [p.pid for p in alive if p.status() != psutil.STATUS_ZOMBIE]
    payload = {"ok": not survivors, "tracked_pids": sorted(owned),
               "surviving_pids": survivors, "seconds": time.monotonic() - started,
               "grace_seconds": grace_seconds,
               "gpu_ids": gpu_ids, "gpu": gpu_snapshot(gpu_ids), "time": time.time()}
    atomic_json(output, payload)
    if survivors:
        raise RuntimeError(f"collection services survived shutdown: {survivors}")
    for path in pid_files:
        path.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("preflight", "stop"))
    p.add_argument("--ports", default="")
    p.add_argument("--gpu-ids", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--pid", type=int, action="append", default=[])
    p.add_argument("--pid-file", type=Path, action="append", default=[])
    a = p.parse_args()
    if a.command == "preflight":
        preflight([int(v) for v in a.ports.split(",") if v], a.gpu_ids, a.output)
    else:
        stop_services(a.pid, a.pid_file, a.output, a.gpu_ids)


if __name__ == "__main__":
    main()
