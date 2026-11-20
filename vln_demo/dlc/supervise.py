#!/usr/bin/env python3
"""Own and supervise the complete single-node 16-GPU VLN service.

Only the Python standard library is used. entry.sh validates the runtime and
device mapping before invoking this file. A ready status means HTTP/process
health; it does not assert that an inference or navigation trial has succeeded.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import ipaddress
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
import uuid


REPO = Path(__file__).resolve().parents[2]
RENDER_ROOTS = {
    7036: ("RS3D_ROOT", "/workspace/agentic_sim/3drs_gs_root"),
    7037: ("ABOTN_ROOT", "/workspace/agentic_sim/abotn_gs_root"),
    7038: ("SAGE_ROOT", "/data/zerovln/SAGE-3D/sage_gs_root"),
    7039: ("MALL_ROOT", "/opt/external/user-9656ce61/ppu_sim/3drs_gs_root"),
}
MODEL_IDS = {9038: "qwen3.8-flash-next", 28900: "Qwen3.6-35B"}
EXTRA_MODEL_IDS = {9000: "qwen3.8-27b"}
PORTS = (5000, 7040, *RENDER_ROOTS, *MODEL_IDS)
PID_NAMES = tuple(f"render_{port}" for port in RENDER_ROOTS) + ("console_5000",)


def discover_host_ips():
    """Keep all local IPv4 interfaces; prefer the browser-facing DLC network."""
    candidates = set()
    try:
        result = subprocess.run(["hostname", "-I"], capture_output=True, text=True,
                                check=True, timeout=5)
        candidates.update(result.stdout.split())
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        candidates.update(item[4][0] for item in socket.getaddrinfo(
            socket.gethostname(), None, socket.AF_INET))
    except OSError:
        pass
    addresses = []
    for value in candidates:
        try:
            ip = ipaddress.ip_address(value)
        except ValueError:
            continue
        if ip.version == 4 and not (ip.is_loopback or ip.is_unspecified or ip.is_link_local):
            addresses.append(str(ip))
    return sorted(addresses, key=lambda value: (not value.startswith("10.230."),
                                               int(ipaddress.ip_address(value))))


def standalone_sim_env(env):
    """The CPU simulator is not a rank in the platform's training job."""
    result = dict(env)
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
                "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "ROLE_RANK",
                "ROLE_WORLD_SIZE"):
        result.pop(key, None)
    return result


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def proc_info(pid):
    """Linux process identity, robust to spaces/parentheses in comm."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"state": fields[0], "pgid": int(fields[2]), "start_ticks": int(fields[19])}
    except (OSError, ValueError, IndexError):
        return None


def atomic_json(path, value):
    fd, name = tempfile.mkstemp(prefix=".status-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(value, out, ensure_ascii=False, indent=2)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def check_ports_free(ports=PORTS):
    """Check wildcard binds, including services listening on a non-loopback IP."""
    held = []
    try:
        for port in ports:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            held.append(listener)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                listener.bind(("0.0.0.0", port))
            except OSError as exc:
                raise RuntimeError(f"port {port} is already occupied; no services started") from exc
    finally:
        for listener in held:
            listener.close()


def http_get(port, path, as_json=True):
    # Loopback probes must never inherit a public HTTP proxy.
    opener = build_opener(ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{port}{path}", timeout=3) as response:
        body = response.read(8 * 1024 * 1024).decode("utf-8")
    return json.loads(body) if as_json else body


def probe_component(name, port, root=None, *, required_backends=(), expected_model=None):
    result = {"ok": False, "checked_at": utc_now(), "url": f"http://127.0.0.1:{port}"}
    try:
        if name in ("habitat_render", "objectnav_render"):
            health = http_get(port, "/health")
            if health.get("protocol") != 1 or health.get("stateless") is not True or not health.get("ok"):
                raise ValueError("invalid Habitat renderer health")
        elif name.startswith("render_"):
            if http_get(port, "/ping", as_json=False).strip() != "pong":
                raise ValueError("/ping did not return pong")
            actual = http_get(port, "/status").get("scenes_root")
            if not isinstance(actual, str) or os.path.realpath(actual) != os.path.realpath(root):
                raise ValueError(f"scenes_root mismatch: expected {root!r}, got {actual!r}")
            result["scenes_root"] = actual
        elif name in ("next", "model_35b", "model_27b"):
            models = http_get(port, "/v1/models")
            ids = [item.get("id") for item in models.get("data", []) if isinstance(item, dict)]
            expected = expected_model or {**MODEL_IDS, **EXTRA_MODEL_IDS}[port]
            if expected not in ids:
                raise ValueError(f"expected model {expected!r}; served ids={ids!r}")
            result["model_id"] = expected
        elif name == "lightnav":
            completed = subprocess.run(["/usr/local/bin/python3", str(REPO / "vln_demo/dlc/lightnav_probe.py")],
                                       capture_output=True, text=True, timeout=10, check=True)
            result.update(json.loads(completed.stdout))
        elif name == "mujoco":
            health = http_get(port, "/api/health")
            if not isinstance(health, dict) or health.get("ok") is False:
                raise ValueError(f"invalid MuJoCo health: {health}")
        elif name == "console_5000":
            responses = {}
            for path in ("/config.json", "/status"):
                responses[path] = http_get(port, path)
                if not isinstance(responses[path], dict):
                    raise ValueError(f"{path} did not return a JSON object")
            if required_backends:
                # webui.config exposes only mounted, nonempty asset sets as
                # {key, label, desc, scenes: [[scene_id, label], ...]}.
                assets = responses["/config.json"].get("assets")
                if not isinstance(assets, list):
                    raise ValueError("/config.json did not return an assets list")
                mounted = {item.get("key"): item for item in assets
                           if isinstance(item, dict) and isinstance(item.get("key"), str)}
                result["verified_backends"] = []
                result["backend_scene_counts"] = {}
                for backend in required_backends:
                    if backend not in mounted:
                        raise ValueError(f"required backend {backend!r} is missing from console assets")
                    scenes = mounted[backend].get("scenes")
                    if not isinstance(scenes, list) or not scenes:
                        raise ValueError(f"required backend {backend!r} has no loaded scenes")
                    result["verified_backends"].append(backend)
                    result["backend_scene_counts"][backend] = len(scenes)
        else:
            raise ValueError(f"unknown component {name}")
        result["ok"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return name, result


@dataclass
class OwnedProcess:
    name: str
    pid: int
    pgid: int
    start_ticks: int
    process: object = None


class Supervisor:
    def __init__(self, run_dir, sleep_python, *, env=None, startup_timeout=1800,
                 probe_interval=15, failure_threshold=3, stop_timeout=20):
        self.run_dir = Path(run_dir).resolve()
        self.logs = self.run_dir / "logs"
        self.pids = self.run_dir / "pids"
        self.env = dict(os.environ if env is None else env)
        self.agent_profile = self.env.get('VLN_AGENT_MODEL_PROFILE', 'next')
        self.agent_spec = None
        if self.agent_profile != 'next':
            self.agent_spec = json.loads((REPO/'vln_demo/vllm/sglang_profiles.json').read_text())[self.agent_profile]
        self.extra_services = self.env.get("VLN_EXTRA_SERVICES") == "1"
        self.last_health_log = None
        self.required_backends = tuple(dict.fromkeys(
            item.strip() for item in self.env.get("VLN_REQUIRED_BACKENDS", "").split(",")
            if item.strip()))
        self.token = uuid.uuid4().hex
        self.started_ticks = proc_info(os.getpid())["start_ticks"]
        self.env.update({
            "VLN_SUPERVISOR_TOKEN": self.token,
            "VLN_LOG_DIR": str(self.logs), "VLN_PID_DIR": str(self.pids),
            "VLN_STRICT_STARTUP": "1",
            "CONSOLE_PORT": "5000", "RS3D_PORT": "7036", "ABOTN_PORT": "7037",
            "SAGE_PORT": "7038", "MALL_PORT": "7039", "RS3D_GPU": "0",
            "ABOTN_GPU": "1", "SAGE_GPU": "2", "MALL_GPU": "3",
            "VLLM_GPUS": "12,13", "VLLM_PORT": "28900",
        })
        # These default roots are for validation only. Exporting them would
        # overwrite the console's distinct dataset-root defaults of the same name.
        self.roots = {port: self.env.get(key, default) for port, (key, default) in RENDER_ROOTS.items()}
        self.sleep_python = sleep_python
        self.startup_timeout = startup_timeout
        self.probe_interval = probe_interval
        self.failure_threshold = failure_threshold
        self.stop_timeout = max(stop_timeout, 45) if self.agent_spec else stop_timeout
        self.stop_event = threading.Event()
        self.stop_signal = None
        self.owned = {}
        self.launchers = {}
        self.log_handles = []
        self.lock = None
        self.failures = {}
        self.status = {
            "state": "starting", "started_at": utc_now(), "run_dir": str(self.run_dir),
            "run_root": self.env.get("VLN_RUN_ROOT", str(self.run_dir)),
            "supervisor_pid": os.getpid(), "hostname": socket.gethostname(),
            "readiness_scope": "process and HTTP health only; inference/navigation not validated",
            "gpu_layout": {"render": [0, 1, 2, 3], "next": list(range(4, 12)),
                           "model_35b": [12, 13], "spare": [14, 15], "sleep": list(range(16))},
            "components": {}, "processes": {},
            "component_labels": {"objectnav_render": "Habitat-gs 渲染"},
        }
        self.status['agent_profile'] = self.agent_profile
        if self.agent_spec:
            self.status['agent_model'] = self.agent_spec
            env_key = {'qwen36-35b':'VLN_SGLANG_35B_URL', 'qwen38-27b':'VLN_SGLANG_27B_URL',
                       'qwen35-122b':'VLN_SGLANG_122B_URL'}[self.agent_profile]
            self.env.update(VLN_PI_PROVIDER='sglang-local', VLN_PI_MODEL=self.agent_spec['model_id'])
            self.env[env_key] = f"http://127.0.0.1:{self.agent_spec['port']}/v1"
        if self.extra_services:
            self.status["gpu_layout"].update(lightnav=[14], objectnav_render=[15], spare=[])
            self.env.update(LIGHTNAV_SERVER_URL="ws://127.0.0.1:8050",
                            LIGHTNAV_MUJOCO_URL="http://127.0.0.1:18088", OBJECTNAV_GPU="15",
                            OBJECTNAV_RENDER_URL="http://127.0.0.1:7041", VLN_DISABLE_LOCAL_27B="1")
        if self.required_backends:
            self.status["required_backends"] = list(self.required_backends)
            self.status["readiness_scope"] = (
                "process and HTTP health, including required mounted console backends; "
                "inference/navigation not validated")
        addresses = discover_host_ips()
        self.status["host_ips"] = addresses
        self.status["console_urls"] = [f"http://{ip}:5000" for ip in addresses]

    def prepare(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.lock = (self.run_dir / ".supervisor.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another supervisor owns {self.run_dir}") from exc
        self.logs.mkdir(exist_ok=True)
        self.pids.mkdir(exist_ok=True)
        if any(self.pids.iterdir()):
            raise RuntimeError(f"PID directory is not empty; use a fresh VLN_RUN_DIR: {self.pids}")

    def write_status(self, state=None, error=None):
        if state is not None:
            self.status["state"] = state
        if error is not None:
            self.status["error"] = str(error)
        self.status["updated_at"] = utc_now()
        self.status["processes"] = {
            name: {"pid": item.pid, "pgid": item.pgid,
                   "returncode": item.process.poll() if item.process is not None else None,
                   "alive": self.is_alive(item)}
            for name, item in self.owned.items()
        }
        atomic_json(self.run_dir / "status.json", self.status)
        self.log_health()

    def log_health(self):
        tick = time.monotonic()
        if self.last_health_log is not None and tick - self.last_health_log < 30:
            return
        self.last_health_log = tick
        # The platform detects hangs from container logs, not our CPFS files.
        # Emit measured health even during startup; never claim readiness here.
        print("VLN_HEALTH " + json.dumps({
            "at": self.status["updated_at"], "state": self.status["state"],
            "components": {name: {key: value[key] for key in ("ok", "error") if key in value}
                           for name, value in self.status["components"].items()},
            "processes": {name: value["alive"] for name, value in self.status["processes"].items()},
        }), flush=True)

    def signal_stop(self, signum, _frame=None):
        self.stop_signal = signum
        self.stop_event.set()

    def spawn(self, name, command, *, env=None, cwd=None):
        handle = (self.logs / f"{name}.log").open("ab", buffering=0)
        self.log_handles.append(handle)
        process = subprocess.Popen(command, cwd=REPO if cwd is None else cwd, env=self.env if env is None else env,
                                   stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        # Popen returns after the child has completed setsid/exec. It may already
        # be a zombie; retaining the group still allows cleanup of descendants.
        info = proc_info(process.pid)
        if info is None:
            process.poll()
            raise RuntimeError(f"cannot identify launched process {name} (pid {process.pid})")
        owned = OwnedProcess(name, process.pid, process.pid, info["start_ticks"], process)
        self.owned[name] = owned
        self.launchers[name] = process
        (self.pids / f"{name}.pid").write_text(f"{process.pid}\n")
        print(f"[supervisor] started {name} pid={process.pid} log={handle.name}", flush=True)
        return process

    def process_has_token(self, pid):
        try:
            values = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            return f"VLN_SUPERVISOR_TOKEN={self.token}".encode() in values
        except OSError:
            return False

    def discover_detached(self):
        errors = []
        for name in PID_NAMES:
            if name in self.owned:
                continue
            try:
                value = (self.pids / f"{name}.pid").read_text().strip()
            except FileNotFoundError:
                continue
            if not value:
                continue  # start_all may still be writing its new PID file.
            if not value.isdigit() or int(value) <= 1:
                errors.append(f"invalid owned PID file: {name}")
                continue
            pid = int(value)
            info = proc_info(pid)
            if info is None or info["state"] == "Z":
                # Remember the group even when its leader died before the first
                # observation. Cleanup still requires a live member's token.
                self.owned[name] = OwnedProcess(name, pid, pid, info["start_ticks"] if info else -1)
                errors.append(f"{name} exited during startup (pid {pid})")
                continue
            if info["pgid"] != pid:
                # Before setsid/exec, /proc/environ can still describe the parent.
                continue
            if not self.process_has_token(pid):
                errors.append(f"refusing unrelated PID in {name}.pid: {pid}")
                continue
            self.owned[name] = OwnedProcess(name, pid, pid, info["start_ticks"])
        if errors:
            raise RuntimeError("; ".join(errors))

    @staticmethod
    def is_alive(item):
        info = proc_info(item.pid)
        return bool(info and info["state"] != "Z" and info["start_ticks"] == item.start_ticks)

    def handle_console_restart(self):
        """Consume an explicit, PID-bound console-only maintenance request."""
        request_path = self.run_dir / "restart-console.request.json"
        if not request_path.exists():
            return
        try:
            request = json.loads(request_path.read_text())
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
        except (ValueError, TypeError) as exc:
            request_path.rename(self.run_dir / "restart-console.rejected.json")
            atomic_json(self.run_dir / "restart-console.result.json", {
                "state": "rejected", "error": str(exc), "at": utc_now()})
            return
        old = self.owned.get("console_5000")
        if (request.get("hostname") != socket.gethostname() or old is None
                or request.get("expected_pid") != old.pid or not self.is_alive(old)):
            request_path.rename(self.run_dir / "restart-console.rejected.json")
            atomic_json(self.run_dir / "restart-console.result.json", {
                "state": "rejected", "error": "task/PID mismatch", "at": utc_now()})
            return
        request_path.rename(self.run_dir / "restart-console.consumed.json")
        receipt = {"state": "restarting", "old_pid": old.pid, "started_at": utc_now()}
        result_path = self.run_dir / "restart-console.result.json"
        proc = Path(f"/proc/{old.pid}")
        command = [arg.decode() for arg in (proc / "cmdline").read_bytes().split(b"\0") if arg]
        if str(REPO / "vln_demo/src/vln_model.py") not in command:
            raise ValueError("console restart command is not this repository's console")
        env = dict(item.decode().split("=", 1) for item in (proc / "environ").read_bytes().split(b"\0") if b"=" in item)
        original_cwd = os.readlink(proc / "cwd")
        atomic_json(result_path, receipt)
        self.write_status("maintenance")
        os.killpg(old.pgid, signal.SIGTERM)
        deadline = time.monotonic() + self.stop_timeout
        while self.is_alive(old) and time.monotonic() < deadline:
            time.sleep(0.1)
        if self.is_alive(old):
            raise RuntimeError("old console did not stop during maintenance")
        # Only this component is replaced. Other launchers and identities remain owned.
        cwd = env.get("RYNNBRAIN_NAV_DIR", original_cwd)
        new = self.spawn("console_5000", command, env=env, cwd=cwd)
        receipt["new_pid"] = new.pid
        atomic_json(result_path, receipt)
        deadline = time.monotonic() + self.startup_timeout
        while not self.stop_event.is_set():
            if new.poll() is not None:
                raise RuntimeError("replacement console exited during maintenance")
            self.check_processes()
            probes = self.collect_probes()
            self.status["components"].update(probes)
            self.write_status("maintenance")
            if probes["console_5000"]["ok"]:
                self.failures["console_5000"] = 0
                receipt.update(state="ready", finished_at=utc_now())
                atomic_json(result_path, receipt)
                self.write_status("ready")
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("replacement console readiness timed out")
            self.stop_event.wait(5)

    def check_processes(self):
        self.handle_console_restart()
        for name, process in self.launchers.items():
            rc = process.poll()
            if name == "evaluation" and rc is not None:
                receipt_file = Path(self.env["VLN_WORKFLOW_PLAN"]).parent / "evaluation-ended.json"
                receipt = json.loads(receipt_file.read_text()) if receipt_file.exists() else {}
                if (rc != 0 or receipt.get("passed") is not True
                        or receipt.get("hostname") != socket.gethostname()
                        or not socket.gethostname().startswith(receipt.get("task", {}).get("job_id", "missing") + "-master-")):
                    raise RuntimeError(f"evaluation failed or completion receipt missing (exit {rc})")
                self.status["evaluation"] = receipt
                self.stop_event.set()
                continue
            if rc is not None and (name != "start_all" or rc != 0):
                raise RuntimeError(f"{name} exited unexpectedly with code {rc}")
        for name in PID_NAMES:
            item = self.owned.get(name)
            if item is not None and not self.is_alive(item):
                raise RuntimeError(f"{name} exited unexpectedly (pid {item.pid})")

    def collect_probes(self):
        specs = [(f"render_{port}", port, root) for port, root in self.roots.items()]
        specs.extend([("next", self.agent_spec["port"] if self.agent_spec else 9038, None), ("model_35b", 28900, None), ("console_5000", 5000, None)])
        specs.append(("habitat_render", 7040, None))
        if self.extra_services:
            specs.extend([("objectnav_render", 7041, None), ("lightnav", 8050, None), ("mujoco", 18088, None)])
        with ThreadPoolExecutor(max_workers=len(specs)) as pool:
            results = dict(pool.map(lambda spec: probe_component(
                *spec, required_backends=self.required_backends if spec[0] == "console_5000" else (),
                **({"expected_model": self.agent_spec["model_id"]} if spec[0] == "next" and self.agent_spec else {})), specs))
        if "sleep" in self.owned:
            results["sleep"] = {"ok": self.is_alive(self.owned["sleep"]), "checked_at": utc_now(),
                                "pid": self.owned["sleep"].pid, "visible_gpus": "all 16"}
        self.status["components"] = results
        return results

    def start_sleep(self):
        env = dict(self.env)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        self.spawn("sleep", [self.sleep_python, "-u", self.env.get("VLN_SLEEP_SCRIPT",
                   "/opt/external/user-d5121be6/sleep.py")], env=env)
        self.check_processes()
        self.status["components"]["sleep"] = {
            "ok": True, "checked_at": utc_now(), "pid": self.owned["sleep"].pid,
            "visible_gpus": "all 16",
        }

    def wait_until(self, seconds):
        deadline = time.monotonic() + seconds
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            self.discover_detached()
            self.check_processes()
            self.stop_event.wait(min(1, max(0, deadline - time.monotonic())))

    def monitor(self):
        deadline = time.monotonic() + self.startup_timeout
        ready = False
        while not self.stop_event.is_set():
            self.discover_detached()
            self.check_processes()
            probes = self.collect_probes()
            self.discover_detached()
            self.check_processes()
            if self.stop_event.is_set():
                break
            if not ready:
                complete = (self.launchers["start_all"].poll() == 0
                            and all(name in self.owned for name in PID_NAMES)
                            and all(value["ok"] for value in probes.values()))
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"startup timed out after {self.startup_timeout:g}s")
                if complete:
                    self.start_sleep()
                    if self.env.get("VLN_WORKFLOW_PLAN"):
                        self.spawn("evaluation", ["/usr/local/bin/python3", "-u",
                                   str(REPO / "vln_demo/batch/workflow.py"), "--server-worker",
                                   self.env["VLN_WORKFLOW_PLAN"]])
                    ready = True
                    self.status["ready_at"] = utc_now()
                    print("[supervisor] ready: all services healthy; sleep now monitors all 16 GPUs", flush=True)
                self.write_status("ready" if ready else "starting")
            else:
                for name, value in probes.items():
                    self.failures[name] = 0 if value["ok"] else self.failures.get(name, 0) + 1
                    value["consecutive_failures"] = self.failures[name]
                bad = [name for name, count in self.failures.items() if count >= self.failure_threshold]
                if bad:
                    raise RuntimeError(f"health check failed {self.failure_threshold} consecutive times: {', '.join(bad)}")
                self.write_status("ready")
            self.wait_until(self.probe_interval if ready else min(5, max(0, deadline - time.monotonic())))

    def owned_live_groups(self):
        """Validate group ownership, including descendants of an exited leader."""
        allowed = {item.pgid for item in self.owned.values()}
        found = set()
        for item in self.owned.values():
            if self.is_alive(item):
                found.add(item.pgid)
        missing = allowed - found
        if missing:
            for path in Path("/proc").iterdir():
                if not path.name.isdigit():
                    continue
                pid = int(path.name)
                info = proc_info(pid)
                if info and info["state"] != "Z" and info["pgid"] in missing and self.process_has_token(pid):
                    found.add(info["pgid"])
                    missing.discard(info["pgid"])
                    if not missing:
                        break
        return found

    def discover_unregistered_groups(self):
        """Cover a setsid child created just before its launcher wrote the PID.

        A unique inherited token proves task ownership. The start-time filter
        avoids reading environments of pre-existing workloads on the node.
        """
        known = {item.pgid for item in self.owned.values()}
        for path in Path("/proc").iterdir():
            if not path.name.isdigit():
                continue
            pid = int(path.name)
            info = proc_info(pid)
            if (not info or info["state"] == "Z" or info["pgid"] in known
                    or info["pgid"] <= 1 or info["pgid"] == os.getpgrp()
                    or info["start_ticks"] < self.started_ticks):
                continue
            if self.process_has_token(pid):
                name = f"descendant_{pid}"
                self.owned[name] = OwnedProcess(name, pid, info["pgid"], info["start_ticks"])
                known.add(info["pgid"])

    def cleanup(self):
        # Do not invoke `start_all.sh stop`: its port-based cleanup could target
        # another task. Fresh PID files + process identities establish ownership.
        errors = []
        deadline = time.monotonic() + self.stop_timeout
        signaled = set()
        while True:
            try:
                self.discover_detached()
            except RuntimeError as exc:
                if str(exc) not in errors:
                    errors.append(str(exc))
            self.discover_unregistered_groups()
            groups = self.owned_live_groups()
            for pgid in groups - signaled:
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                signaled.add(pgid)
            for process in self.launchers.values():
                process.poll()
            if not groups or time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        for pgid in self.owned_live_groups():
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for process in self.launchers.values():
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                errors.append(f"process {process.pid} survived SIGKILL")
        # Detached processes cannot be wait()ed by this parent. Allow delivery
        # before recording survivor evidence; zombies do not count as running.
        kill_deadline = time.monotonic() + 3
        survivors = self.owned_live_groups()
        while survivors and time.monotonic() < kill_deadline:
            time.sleep(0.1)
            survivors = self.owned_live_groups()
        self.status["cleanup"] = {"remaining_process_groups": sorted(survivors), "notes": errors}
        for handle in self.log_handles:
            handle.close()
        return not survivors

    def run(self):
        # Failure to acquire the run lock must not overwrite another task's status.
        self.prepare()
        old_handlers = {sig: signal.signal(sig, self.signal_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
        rc = 1
        state = "failed"
        error = None
        try:
            self.write_status("starting")
            ports = tuple(p for p in PORTS if p != 9038) + (self.agent_spec["port"],) if self.agent_spec else PORTS
            check_ports_free((*ports, 7041, 8050, 18088) if self.extra_services else ports)
            self.spawn("habitat_render", ["bash", str(REPO / "scripts/demo/remote_habitat/start.sh")],
                       env=standalone_sim_env(self.env))
            if self.agent_spec:
                model_env = dict(self.env, CUDA_VISIBLE_DEVICES="4,5,6,7,8,9,10,11")
                self.spawn("next", ["/usr/local/bin/python3", "-u", str(REPO/"vln_demo/vllm/sglang_server.py"),
                           "--profile", self.agent_profile, "--run-dir", str(Path(self.status["run_root"])/"agent-model")], env=model_env)
            else:
                self.spawn("next", ["bash", str(REPO / "scripts/serve/vllm/vllm38next.sh")])
            self.spawn("model_35b", ["bash", str(REPO / "scripts/serve/vllm/vllm-vln-35B.sh")])
            if self.extra_services:
                sim_env = standalone_sim_env(self.env)
                self.spawn("objectnav_render", ["bash", str(REPO / "scripts/demo/objectnav/start.sh")], env=sim_env)
                for component in ("lightnav", "mujoco"):
                    self.spawn(component, ["bash", str(REPO / "scripts/demo/dlc/start_extra.sh"), component], env=sim_env)
                # Register MuJoCo before console initialization builds its catalog.
                deadline = time.monotonic() + 300
                while not self.stop_event.is_set():
                    self.check_processes()
                    name, health = probe_component("mujoco", 18088)
                    self.status["components"][name] = health
                    self.write_status("starting")
                    if health["ok"]:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("MuJoCo did not become healthy before console startup")
                    self.stop_event.wait(3)
                if self.stop_event.is_set():
                    raise RuntimeError("startup interrupted")
            self.spawn("start_all", ["bash", str(REPO / "scripts/demo/start_all.sh"), "start"],
                       env=standalone_sim_env(self.env))
            self.monitor()
            state = "stopped"
            rc = 128 + self.stop_signal if self.stop_signal else 0
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"[supervisor] failed: {error}", file=sys.stderr, flush=True)
        finally:
            if not self.cleanup():
                state, rc = "failed", 1
                error = f"{error + '; ' if error else ''}owned processes survived cleanup"
            self.status["finished_at"] = utc_now()
            self.write_status(state, error)
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
            if self.lock is not None:
                self.lock.close()
        return rc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=os.environ.get("VLN_RUN_DIR"),
                        help="fresh persistent run directory (or VLN_RUN_DIR); stores status/logs/PIDs")
    parser.add_argument("--sleep-python", default=os.environ.get("VLN_SLEEP_PYTHON"),
                        help="prevalidated Python for sleep.py (or VLN_SLEEP_PYTHON)")
    parser.add_argument("--startup-timeout", type=float, default=os.environ.get("VLN_STARTUP_TIMEOUT", "1800"))
    parser.add_argument("--probe-interval", type=float, default=os.environ.get("VLN_PROBE_INTERVAL", "15"))
    parser.add_argument("--failure-threshold", type=int, default=os.environ.get("VLN_FAILURE_THRESHOLD", "3"))
    parser.add_argument("--stop-timeout", type=float, default=os.environ.get("VLN_STOP_TIMEOUT", "20"))
    args = parser.parse_args(argv)
    if not args.run_dir or not Path(args.run_dir).is_absolute():
        parser.error("an absolute --run-dir / VLN_RUN_DIR is required")
    if not args.sleep_python:
        parser.error("--sleep-python / VLN_SLEEP_PYTHON is required")
    if min(args.startup_timeout, args.probe_interval, args.failure_threshold, args.stop_timeout) <= 0:
        parser.error("timeouts, interval and failure threshold must be positive")
    try:
        return Supervisor(args.run_dir, args.sleep_python, startup_timeout=args.startup_timeout,
                          probe_interval=args.probe_interval, failure_threshold=args.failure_threshold,
                          stop_timeout=args.stop_timeout).run()
    except Exception as exc:
        print(f"[supervisor] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
