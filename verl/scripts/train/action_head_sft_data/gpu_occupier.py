#!/usr/bin/env python3
"""GPU standby workload for explicitly owned idle slots.

Use --gpus for selected slots, --auto for every visible GPU, or the default
4*lane+{2,3} renderer/visual slots. Do not infer ownership from GPU utilization:
our own workload would otherwise trigger repeated self-release.

Render-aware mode aggregates rendering_<port>[_<pid>_<request>] flags and pauses
while any request for a mapped port is active. The monitor polls every 20 ms;
each synchronized workload iteration checks the busy/termination events.

Preallocated elementwise kernels avoid the PPU FP32 GEMM illegal-address failure
observed during companion checkpoint waits. Device errors remain fatal and are
reported through the existing strict worker-liveness and readiness protocol.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import threading
import time

import torch


def _occupy_loop(gpu, stop, busy=None, ready=None):
    """Bounded elementwise work, avoiding standby FP32 BLAS kernels.

    Preallocate 192 MiB per GPU. Synchronize every iteration so CUDA work cannot
    queue indefinitely and render/termination signals can take effect promptly.
    """
    try:
        torch.cuda.set_device(gpu)
        a = torch.zeros(4096, 4096, device="cuda")
        b = torch.ones_like(a)
        c = torch.empty_like(a)
        while not stop.is_set():
            if busy is not None and busy.is_set():
                time.sleep(0.005)
                continue
            for _ in range(8):
                if stop.is_set() or (busy is not None and busy.is_set()):break
                torch.add(a, b, out=c)
                torch.mul(c, 0.5, out=a)  # a converges to 1; no overflow/allocation.
            torch.cuda.synchronize(gpu)
            if ready is not None:
                ready.set()
    except Exception as exc:
        print(f"[gpu-occupier] gpu={gpu} thread stopped: {type(exc).__name__}: {exc}", flush=True)
    finally:
        try:
            torch.cuda.synchronize(gpu)
            torch.cuda.empty_cache()
        except Exception:
            pass


def _busy_render_ports(busy_dir):
    """Read legacy port flags and unique request flags in one directory scan."""
    try:
        names = os.listdir(busy_dir)
    except OSError:
        return set()
    ports = set()
    for name in names:
        if name.startswith("rendering_"):
            port = name[len("rendering_"):].split("_", 1)[0]
            if port.isdigit():
                ports.add(port)
    return ports


def _busy_monitor(gpu_ports, busy, stop, busy_dir, poll=0.02):
    """轮询 request/legacy flag，按 renderer 端口和 GPU 聚合 set/clear busy。

    gpu_ports: {gpu: set(ports)}。共享 renderer 的任一请求未完成，都保持 busy。
    """
    while not stop.is_set():
        present = _busy_render_ports(busy_dir)
        for gpu, ports in gpu_ports.items():
            if any(str(p) in present for p in ports):
                busy[gpu].set()
            else:
                busy[gpu].clear()
        time.sleep(poll)


def _parse_render_port_gpu(spec):
    """P:G1|G2|...,P:G,... → {gpu: set(ports)}。一个 port 可映射多 GPU（rs3d 4GPU/server，
    用 | 分隔同端口的多 GPU，逗号分隔不同端口）。"""
    gpu_ports = {}
    for item in spec.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        port_s, gpus_s = item.split(":", 1)
        port = port_s.strip()
        for g in gpus_s.split("|"):
            g = g.strip()
            if g.lstrip("-").isdigit():
                gpu_ports.setdefault(int(g), set()).add(port)
    return gpu_ports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="", help="占卡 GPU 槽位(逗号分隔)；默认空=4*lane+{2,3}")
    ap.add_argument("--auto", action="store_true", help="自动: 占所有 4*lane+{2,3} 槽位")
    ap.add_argument("--require-gpu-count", type=int, default=0,
                    help="严格校验可见及占用卡数；0 保持原有行为")
    ap.add_argument("--ready-file", type=Path,
                    help="全部 GPU 完成首次运算后原子发布就绪信息")
    ap.add_argument("--startup-timeout", type=float, default=90)
    ap.add_argument("--render-busy-dir", default="",
                    help="渲染感知模式: 按端口聚合 rendering_<port>[_<request>] flag 决定是否暂停 逐元素计算")
    ap.add_argument("--render-port-gpu", default="",
                    help="render server 端口→GPU 映射, 形如 7060:0,7061:1,7062:2|3|4 (同端口多 GPU 用 | 分隔)")
    args = ap.parse_args()

    n = torch.cuda.device_count()
    if args.require_gpu_count and n != args.require_gpu_count:
        raise RuntimeError(f"expected {args.require_gpu_count} visible GPUs, got {n}")
    busy = {}          # gpu -> threading.Event（render-aware 时用）
    gpu_ports = {}     # gpu -> set(ports)（render-aware 时用）

    if args.render_busy_dir and args.render_port_gpu:
        gpu_ports = _parse_render_port_gpu(args.render_port_gpu)
        gpus = sorted(gpu_ports.keys())
        if not gpus:
            print("[gpu-occupier] render-aware: empty port-gpu map; exit", flush=True)
            return
        for g in gpus:
            busy[g] = threading.Event()
        print(f"[gpu-occupier] render-aware: n_gpu={n} occupy={gpus} "
              f"busy_dir={args.render_busy_dir} ports/gpu="
              f"{ {g: sorted(p) for g, p in gpu_ports.items()} }", flush=True)
    else:
        if args.gpus:
            gpus = [int(x) for x in args.gpus.split(",") if x.strip().isdigit()]
        elif args.auto:
            gpus = list(range(n))
        else:
            gpus = []
            for lane in range(n // 4):
                for off in (2, 3):
                    g = lane * 4 + off
                    if g < n:
                        gpus.append(g)
        gpus = sorted(g for g in gpus if 0 <= g < n)
        if not gpus:
            print("[gpu-occupier] no target gpu; exit", flush=True)
            return
        print(f"[gpu-occupier] continuous: n_gpu={n} occupy={gpus} "
              f"(vLLM slots 4*lane+{{0,1}} untouched)", flush=True)

    if args.require_gpu_count and len(gpus) != args.require_gpu_count:
        raise RuntimeError(f"expected {args.require_gpu_count} occupied GPUs, got {gpus}")
    strict = bool(args.require_gpu_count or args.ready_file)
    ready = {gpu: threading.Event() for gpu in gpus}
    stops = {}
    for gpu in gpus:
        stop = threading.Event()
        ev = busy.get(gpu)  # render-aware 时为 Event，continuous 时 None
        t = threading.Thread(target=_occupy_loop,
                             args=(gpu, stop, ev, ready[gpu] if strict else None), daemon=True)
        t.start()
        stops[gpu] = (t, stop)
        print(f"[gpu-occupier] occupy gpu={gpu} mode={'render-aware' if ev is not None else 'continuous'}",
              flush=True)

    monitor_stop = None
    if gpu_ports:
        monitor_stop = threading.Event()
        mt = threading.Thread(target=_busy_monitor,
                              args=(gpu_ports, busy, monitor_stop, args.render_busy_dir),
                              daemon=True)
        mt.start()
        print(f"[gpu-occupier] busy-monitor started, poll=20ms", flush=True)

    finished = threading.Event()

    def _shutdown(signum, frame):
        for _, stop in stops.values():
            stop.set()
        if monitor_stop is not None:
            monitor_stop.set()
        finished.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    deadline = time.monotonic() + args.startup_timeout
    published = False
    try:
        while not finished.wait(0.5 if strict else 5.0):
            if not strict:
                continue
            failed = [gpu for gpu, (t, _) in stops.items() if not t.is_alive()]
            if failed or (not published and time.monotonic() >= deadline):
                raise RuntimeError(f"GPU occupier failed or timed out: failed={failed}, "
                                   f"ready={[g for g, ev in ready.items() if ev.is_set()]}")
            if not published and all(ev.is_set() for ev in ready.values()):
                if args.ready_file:
                    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
                    temporary = args.ready_file.with_suffix('.tmp')
                    temporary.write_text(json.dumps({'pid': os.getpid(), 'gpu_count': n,
                                                     'gpus': gpus, 'time': time.time()}))
                    os.replace(temporary, args.ready_file)
                published = True
                print(f"[gpu-occupier] ready: all {len(gpus)} GPU workers validated", flush=True)
    finally:
        _shutdown(None, None)
        # Keep PyTorch alive until CUDA work and every worker have drained.
        # Daemon threads running during interpreter teardown caused SIGABRT.
        for thread, _ in stops.values():
            thread.join()
        if monitor_stop is not None:
            mt.join()
    print("[gpu-occupier] shutting down", flush=True)


if __name__ == "__main__":
    main()
