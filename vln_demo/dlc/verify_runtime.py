#!/usr/bin/env python3
"""Exercise one explicitly identified DLC stack and write an acceptance receipt.

This changes the *target DLC console*: one episode from Lepu and each required train
backend is loaded and turned left once. It never falls back to a local service.
Only the standard library is needed; CPU tests do not contact running services.
"""

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from supervise import MODEL_IDS, EXTRA_MODEL_IDS, RENDER_ROOTS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from objectnav.catalog_io import read_catalog
from objectnav.categories import BACKEND_CATEGORIES, expected_counts, select_category


EXPECTED_DATASETS = {
    "habitat_train": {"episodes": 10819, "scenes": 61},
    "rxr_train_guide": {"episodes": 60300, "scenes": 59},
    "rxr_train_follower": {"episodes": 58752, "scenes": 59},
    "lepu": {"episodes": 7381, "scenes": 1},
}
REQUIRED_PROCESSES = ("next", "model_35b", "console_5000", "sleep",
                      *(f"render_{port}" for port in RENDER_ROOTS))


def required_processes(status):
    return REQUIRED_PROCESSES + (("objectnav_render", "lightnav", "mujoco")
                                 if "objectnav_render" in status.get("gpu_layout", {}) else ())


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "status timestamp has no timezone")
    return parsed.timestamp()


def write_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def ready_status(status, *, at=None):
    require(status.get("state") == "ready", f"supervisor is {status.get('state')!r}")
    age = (time.time() if at is None else at) - timestamp(status["updated_at"])
    require(-10 <= age <= 120, f"supervisor status is stale or future-dated: age={age:.1f}s")
    # entry.sh execs the supervisor, so it can legitimately be container PID 1.
    require(type(status.get("supervisor_pid")) is int and status["supervisor_pid"] >= 1,
            "missing supervisor PID")
    for name in required_processes(status):
        require(status.get("components", {}).get(name, {}).get("ok") is True,
                f"component {name} is not healthy")
        require(status.get("processes", {}).get(name, {}).get("alive") is True,
                f"owned process {name} is not alive")
    return status


def target_host(status, job_id, run_root, requested=None):
    require(re.fullmatch(r"dlc[a-z0-9]+", job_id) is not None, "expected job ID must be a DLC job ID")
    hostname = status.get("hostname", "")
    require(hostname == job_id or hostname.startswith(job_id + "-"),
            f"status hostname {hostname!r} does not belong to expected job {job_id!r}")
    require(Path(status.get("run_root", "")).resolve() == run_root.resolve(),
            "status run_root does not match --run-dir")
    addresses = []
    for raw in status.get("host_ips", []):
        address = ipaddress.ip_address(raw)
        if address.version == 4 and not (address.is_loopback or address.is_unspecified
                                         or address.is_multicast or address.is_link_local):
            addresses.append(str(address))
    require(addresses, "DLC status has no usable non-loopback IPv4; no localhost fallback")
    if requested is not None:
        require(requested in addresses, "--host must be an IP advertised by this DLC status")
        addresses = [requested]
    # Protect against a copied/mislabelled status aimed at the current DSW.
    if socket.gethostname() != hostname:
        try:
            local = {row[4][0] for row in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
        except OSError:
            local = set()
        require(not any(address in local for address in addresses),
                "DLC address resolves to the current, differently named machine")
    return addresses[0]


def dataset_counts(config, expected):
    require(isinstance(config.get("assets"), list), "config has no assets list")
    require(isinstance(config.get("episodes"), dict), "config has no episodes mapping")
    assets = {item["key"]: item for item in config["assets"]}
    result = {}
    for backend, counts in expected.items():
        require(backend in assets, f"required backend {backend} is absent")
        scenes = [item[0] for item in assets[backend]["scenes"]]
        require(scenes and len(scenes) == len(set(scenes)), f"{backend}: empty or duplicate scenes")
        total, languages = 0, Counter()
        for scene in scenes:
            require(config.get("scene_of", {}).get(scene) == backend,
                    f"{scene}: wrong backend ownership")
            episodes = config["episodes"].get(scene)
            require(isinstance(episodes, list) and episodes, f"{scene}: no episodes")
            require(len(set(episodes)) == len(episodes), f"{scene}: duplicate episode IDs")
            total += len(episodes)
            by_episode = config.get("episode_languages", {}).get(scene, {})
            languages.update(by_episode.get(ep, "unspecified") for ep in episodes)
        actual = {"episodes": total, "scenes": len(scenes)}
        require(actual == counts, f"{backend}: expected {counts}, loaded {actual}")
        result[backend] = {**actual, "languages": dict(languages),
                           "sample_scene": scenes[0],
                           "sample_episode": config["episodes"][scenes[0]][0]}
    return result


def completion_text(response):
    require(isinstance(response, dict) and not response.get("error"), "model returned an error")
    choices = response.get("choices")
    require(isinstance(choices, list) and choices, "model returned no choices")
    choice = choices[0]
    require(choice.get("finish_reason") not in ("error", "content_filter"),
            "model generation failed or was filtered")
    content = choice.get("message", {}).get("content")
    if isinstance(content, list):
        content = "\n".join(item.get("text", "") for item in content if isinstance(item, dict))
    require(isinstance(content, str) and bool(content.strip()), "model returned no answer content")
    return content.strip()


def snapshot_data(payload):
    require(payload.get("ok") is True, "console snapshot is not ok")
    source = payload.get("source", {})
    require(isinstance(source.get("scene"), str) and source["scene"], "snapshot lacks scene")
    require(isinstance(source.get("episode_id"), str) and source["episode_id"], "snapshot lacks episode")
    require(type(source.get("seq")) is int and source["seq"] >= 0, "snapshot lacks frame sequence")
    require(all(type(payload.get(key)) is int and payload[key] > 0 for key in ("width", "height")),
            "snapshot has invalid dimensions")
    prefix = "data:image/jpeg;base64,"
    encoded = payload.get("image_data", "")
    require(encoded.startswith(prefix), "snapshot is not a JPEG data URI")
    body = base64.b64decode(encoded[len(prefix):], validate=True)
    require(len(body) > 100 and body.startswith(b"\xff\xd8\xff") and body.endswith(b"\xff\xd9"),
            "snapshot does not contain a complete JPEG")
    return body


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError(f"refusing redirect away from explicit DLC endpoint: HTTP {code}")


class Verifier:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.run_dir).resolve()
        self.output = Path(args.output_dir).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        require(not (self.output / "receipt.json").exists(), "use a fresh --output-dir; receipt already exists")
        self.status_path = self.root / "status.json"
        self.initial = None
        self.host = None
        self.opener = build_opener(ProxyHandler({}), NoRedirect())
        self.receipt = {"state": "running", "passed": False, "started_at": now(), "expected_job_id": args.expected_job_id,
                        "run_root": str(self.root), "checks": [],
                        "scope": "Dedicated DLC console: three train episode switches and single left turns; "
                                 "direct text/image model generation; HTTP and persistent process checks"}
        self.checkpoint()

    def checkpoint(self):
        self.receipt["updated_at"] = now()
        write_json(self.output / "receipt.json", self.receipt)

    def check(self, name, action):
        check = {"name": name, "started_at": now(), "passed": False}
        self.receipt["checks"].append(check)
        self.checkpoint()
        print(f"[acceptance] {name}", flush=True)
        try:
            check["evidence"] = action()
            check["passed"] = True
        except Exception as exc:
            check["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            check["finished_at"] = now()
            self.checkpoint()

    def read_status(self):
        resolved = self.status_path.resolve(strict=True)
        require(resolved.is_relative_to(self.root), "status pointer escapes requested run root")
        status = ready_status(json.loads(resolved.read_text()))
        host = target_host(status, self.args.expected_job_id, self.root, self.args.host)
        require(Path(status["run_dir"]).resolve() == resolved.parent, "status belongs to another attempt")
        if self.initial is not None:
            for key in ("run_dir", "hostname", "supervisor_pid", "started_at", "ready_at"):
                require(status.get(key) == self.initial.get(key), f"DLC attempt changed during acceptance: {key}")
            require(host == self.host, "DLC IP changed during acceptance")
        return status, host

    def http(self, port, path, payload=None, *, as_json=True):
        require(self.host is not None, "DLC target has not been validated")
        url = f"http://{self.host}:{port}{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        request = Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
        with self.opener.open(request, timeout=self.args.request_timeout) as response:
            require(response.status == 200, f"{url}: HTTP {response.status}")
            body = response.read(256 * 1024 * 1024 + 1)
        require(len(body) <= 256 * 1024 * 1024, f"{url}: response exceeds 256 MiB")
        return json.loads(body) if as_json else body.decode()

    def identify(self):
        self.initial, self.host = self.read_status()
        write_json(self.output / "status-initial.json", self.initial)
        self.receipt.update(hostname=self.initial["hostname"], host_ip=self.host,
                            attempt_dir=self.initial["run_dir"])
        return {"hostname": self.initial["hostname"], "host_ip": self.host,
                "status": "status-initial.json"}

    def endpoints(self):
        results = {}
        for port, (_, default_root) in RENDER_ROOTS.items():
            require(self.http(port, "/ping", as_json=False).strip() == "pong", f"render {port}: invalid ping")
            result = self.http(port, "/status")
            expected = self.initial["components"][f"render_{port}"].get("scenes_root", default_root)
            require(isinstance(result.get("scenes_root"), str)
                    and os.path.realpath(result["scenes_root"]) == os.path.realpath(expected),
                    f"render {port}: wrong scene root")
            results[f"render_{port}"] = {"scenes_root": result["scenes_root"], "status": result}
        model_ids = {**MODEL_IDS, **(EXTRA_MODEL_IDS if "model_27b" in self.initial.get("gpu_layout", {}) else {})}
        if self.initial.get("agent_model"):
            model_ids.pop(9038, None)
            spec = self.initial["agent_model"]
            model_ids[spec["port"]] = spec["model_id"]
        for port, model_id in model_ids.items():
            result = self.http(port, "/v1/models")
            ids = [item.get("id") for item in result.get("data", [])]
            require(model_id in ids, f"port {port}: expected {model_id}, got {ids}")
            results[str(port)] = {"expected_model": model_id, "served_models": result}
        write_json(self.output / "endpoints.json", results)
        return {"artifact": "endpoints.json"}

    def datasets(self):
        self.config = self.http(5000, "/config.json")
        write_json(self.output / "console-config.json", self.config)
        expected = EXPECTED_DATASETS
        if self.args.expected_datasets:
            expected = json.loads(Path(self.args.expected_datasets).read_text())
            require(set(expected) == set(EXPECTED_DATASETS),
                    "expected datasets must cover Lepu and all three train backends")
        self.counts = dataset_counts(self.config, expected)
        write_json(self.output / "datasets.json", self.counts)
        return {"counts": self.counts, "config": "console-config.json"}

    def gary_categories(self):
        source = json.loads((self.root.parent / "config-version/source-manifest.json").read_text())
        manifest = Path(source["objectnav_manifest"]["path"])
        require(hashlib.sha256(manifest.read_bytes()).hexdigest() == source["objectnav_manifest"]["sha256"],
                "Gary catalog changed since submission")
        rows = read_catalog(manifest)
        expected = expected_counts(rows)
        require(expected == source["gary_categories"], "Gary category counts changed since submission")
        assets = {asset["key"]: asset for asset in self.config["assets"]}
        require("objectnav" not in assets, "aggregate ObjectNav must not remain in the asset dropdown")
        counts = dataset_counts(self.config, expected)
        for key, category in BACKEND_CATEGORIES.items():
            require(assets[key]["label"] == key, "Gary asset label differs from its category name")
            actual = {(scene, str(episode)) for scene, _ in assets[key]["scenes"]
                      for episode in self.config["episodes"][scene]}
            wanted = {(row["scene"], str(row["episode_id"])) for row in select_category(rows, category)}
            require(actual == wanted, key + ": episodes leaked, missing or assigned to the wrong category")
        self.counts.update(counts)
        write_json(self.output / "gary-categories.json", counts)
        return {"counts": counts, "manifest": str(manifest), "exact_episode_membership": True}

    def command(self, action, **params):
        reply = self.http(5000, "/cmd?" + urlencode({"do": action, **params}), as_json=False)
        require(reply.strip() == "ok", f"console rejected {action}: {reply}")
        return {"action": action, "params": params, "reply": reply}

    def wait_console(self, predicate):
        deadline = time.monotonic() + self.args.request_timeout
        while True:
            status = self.http(5000, "/status")
            require(status.get("status") != "error", f"console error: {status.get('result')}")
            require(not status.get("locked"), "console is locked; refusing to override an active user")
            if predicate(status):
                return status
            require(time.monotonic() < deadline, f"console did not reach requested state: {status}")
            time.sleep(1)

    def save_snapshot(self, name, scene, episode):
        payload = self.http(5000, "/frame-snapshot")
        body = snapshot_data(payload)
        require(payload["source"]["scene"] == scene and payload["source"]["episode_id"] == episode,
                "snapshot belongs to another scene/episode")
        (self.output / (name + ".jpg")).write_bytes(body)
        metadata = {key: value for key, value in payload.items() if key != "image_data"}
        metadata.update(sha256=hashlib.sha256(body).hexdigest(), bytes=len(body), image=name + ".jpg")
        write_json(self.output / (name + ".json"), metadata)
        return payload, metadata

    def navigation(self, backend):
        selection = self.counts.get(backend)
        if selection is None:
            asset = next(a for a in self.config["assets"] if a["key"] == backend)
            scene = asset["scenes"][0][0]
            selection = {"sample_scene": scene, "sample_episode": self.config["episodes"][scene][0]}
        scene, episode = selection["sample_scene"], str(selection["sample_episode"])
        current = self.http(5000, "/status")
        require(current.get("status") in ("idle", "manual") and not current.get("locked"),
                "console is busy or locked; refusing to interrupt existing work")
        commands = [self.command("switch", scene=scene, episode_id=episode)]
        artifact = self.output / (backend + "-navigation.json")
        evidence = {"backend": backend, "commands": commands, "scene": scene, "episode_id": episode}
        write_json(artifact, evidence)
        before = self.wait_console(lambda s: s.get("scene") == scene and s.get("episode_id") == episode
                                   and s.get("backend") == backend and s.get("status") == "idle"
                                   and s.get("seq", -1) > current.get("seq", -1))
        _, before_image = self.save_snapshot(backend + "-before", scene, episode)
        evidence.update(before=before, before_observation=before_image)
        write_json(artifact, evidence)
        require(before_image["source"]["seq"] == before["seq"], "frame advanced before turn")
        commands.append(self.command("manual", a="a"))
        write_json(artifact, evidence)
        after = self.wait_console(lambda s: s.get("scene") == scene and s.get("episode_id") == episode
                                  and s.get("backend") == backend and s.get("status") == "manual"
                                  and s.get("seq", -1) > before["seq"])
        _, after_image = self.save_snapshot(backend + "-after", scene, episode)
        evidence.update(after=after, after_observation=after_image)
        write_json(artifact, evidence)
        require(after_image["source"]["seq"] == after["seq"], "frame advanced after turn")
        require(after.get("step", 0) > before.get("step", 0), "turn did not increment simulation step")
        turn = (float(after["map_heading_deg"]) - float(before["map_heading_deg"]) + 180) % 360 - 180
        require(math.isfinite(turn) and 1 <= abs(turn) <= 90, f"turn has no plausible heading change: {turn}")
        require(before_image["sha256"] != after_image["sha256"], "turn returned identical observation bytes")
        evidence["heading_change_degrees"] = turn
        write_json(artifact, evidence)
        return {"artifact": backend + "-navigation.json", "scene": scene, "episode_id": episode,
                "before_seq": before["seq"], "after_seq": after["seq"], "heading_change_degrees": turn}

    def generation(self, port, image=False):
        spec = (self.initial or {}).get("agent_model", {})
        model = spec["model_id"] if port == spec.get("port") else {**MODEL_IDS, **EXTRA_MODEL_IDS}[port]
        content = "Reply with one short sentence confirming that you can respond."
        frame_source = None
        if image:
            # A new atomic snapshot, not a stale frame saved before a turn.
            frame = self.http(5000, "/frame-snapshot")
            snapshot_data(frame)
            frame_source = frame["source"]
            content = [{"type": "text", "text": "Describe the visible scene in one short sentence."},
                       {"type": "image_url", "image_url": {"url": frame["image_data"]}}]
        request = {"model": model, "messages": [{"role": "user", "content": content}],
                   "max_tokens": 256, "temperature": 0, "stream": False,
                   "chat_template_kwargs": {"enable_thinking": False}}
        prefix = f"generation-{port}"
        write_json(self.output / (prefix + "-request.json"), request)
        response = self.http(port, "/v1/chat/completions", request)
        write_json(self.output / (prefix + "-response.json"), response)
        text = completion_text(response)
        return {"model": model, "response_text": text, "frame_source": frame_source,
                "request": prefix + "-request.json", "response": prefix + "-response.json"}

    def persistence(self):
        previous, _ = self.read_status()
        samples = [previous]
        started = time.monotonic()
        duration = 2100 if "model_27b" in self.initial.get("gpu_layout", {}) else 30
        while len(samples) < 3 or time.monotonic() - started < duration:
            time.sleep(15)
            deadline = time.monotonic() + 60
            while True:
                current, _ = self.read_status()
                if timestamp(current["updated_at"]) > timestamp(previous["updated_at"]):
                    break
                require(time.monotonic() < deadline, "supervisor status stopped advancing")
                time.sleep(1)
            for name in required_processes(current):
                require(current["processes"][name]["pid"] == previous["processes"][name]["pid"],
                        f"{name} restarted during persistence check")
            samples.append(current)
            previous = current
        elapsed = time.monotonic() - started
        write_json(self.output / "persistence.json", {"elapsed_seconds": elapsed, "samples": samples})
        return {"artifact": "persistence.json", "elapsed_seconds": elapsed, "samples": len(samples),
                "sleep_pid": samples[-1]["processes"]["sleep"]["pid"]}

    def lightnav(self):
        frame = self.http(5000, "/frame-snapshot")
        data = snapshot_data(frame)
        image = self.output / "lightnav-current-frame.jpg"
        image.write_bytes(data)
        result = subprocess.run(["/usr/local/bin/python3", str(Path(__file__).with_name("lightnav_probe.py")),
                                 "--host", self.host, "--image", str(image)],
                                capture_output=True, text=True, timeout=240, check=True)
        reply = json.loads(result.stdout)
        health = self.http(18088, "/api/health")
        require(health.get("ok") is True, "MuJoCo is not healthy")
        write_json(self.output / "lightnav-prediction.json", reply)
        return {"prediction": "lightnav-prediction.json", "mujoco_ok": True}

    def platform_health_logs(self):
        params = {"jobId": self.args.expected_job_id, "podName": self.initial["hostname"],
                  "startTime": int(time.time()) - 180, "endTime": int(time.time())}
        from submission import platform_command
        result = subprocess.run([*platform_command(), "logs", "--params", json.dumps(params)],
                                capture_output=True, text=True, timeout=90, check=True)
        payload = json.loads(result.stdout)
        require(200 <= payload.get("http_status", 0) < 300, "platform logs HTTP failure")
        response = payload["response"]
        require(response.get("code") == 2000, "platform logs business failure")
        lines = response.get("data", [])
        require(isinstance(lines, list), "invalid platform log response")
        matches = [line for line in lines if isinstance(line, str) and "VLN_HEALTH " in line]
        require(len(matches) >= 2, "platform did not ingest periodic health logs")
        write_json(self.output / "platform-health-logs.json", {"params": params, "lines": matches})
        return {"artifact": "platform-health-logs.json", "entries": len(matches)}

    def restore_console(self, original):
        current = self.http(5000, "/status")
        require(current.get("status") in ("idle", "manual") and not current.get("locked"),
                "console is busy or locked; refusing to restore over active work")
        scene, episode = original["scene"], str(original["episode_id"])
        self.command("switch", scene=scene, episode_id=episode)
        restored = self.wait_console(lambda s: s.get("scene") == scene
                                     and s.get("episode_id") == episode
                                     and s.get("status") == "idle"
                                     and s.get("seq", -1) > current.get("seq", -1))
        return {"scene": scene, "episode_id": episode, "seq": restored["seq"]}

    def run(self):
        original = None
        try:
            self.check("target_identity_and_ready", self.identify)
            self.check("four_render_roots_and_two_model_ids", self.endpoints)
            self.check("full_train_dataset_counts", self.datasets)
            extra = "objectnav_render" in (self.initial or {}).get("gpu_layout", {})
            if extra:
                self.check("gary_category_counts_and_membership", self.gary_categories)
            self.check("next_text_generation", lambda: self.generation((self.initial or {}).get("agent_model", {}).get("port", 9038)))
            original = self.http(5000, "/status")
            for backend in EXPECTED_DATASETS:
                self.check(backend + "_episode_and_turn", lambda backend=backend: self.navigation(backend))
            self.check("35b_current_frame_generation", lambda: self.generation(28900, image=True))
            if extra:
                for backend in BACKEND_CATEGORIES:
                    self.check(backend + "_episode_and_turn", lambda backend=backend: self.navigation(backend))
                self.check("lightnav_prediction_and_mujoco", self.lightnav)
            self.check("supervisor_and_sleep_persistence", self.persistence)
            if extra:
                self.check("platform_health_log_delivery", self.platform_health_logs)
            self.receipt.update(state="passed", passed=True)
        except Exception as exc:
            self.receipt.update(state="failed", passed=False, error=f"{type(exc).__name__}: {exc}")
        finally:
            if original is not None:
                try:
                    self.check("restore_original_console_scene", lambda: self.restore_console(original))
                except Exception as exc:
                    self.receipt.update(state="failed", passed=False,
                                        restore_error=f"{type(exc).__name__}: {exc}")
        self.receipt["finished_at"] = now()
        self.checkpoint()
        print(f"[acceptance] {self.receipt['state']}: {self.output / 'receipt.json'}", flush=True)
        return 0 if self.receipt["state"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="persistent output root for this DLC task")
    parser.add_argument("--output-dir", required=True, help="fresh directory for this acceptance run")
    parser.add_argument("--expected-job-id", required=True, help="exact DLC job ID owning the target hostname")
    parser.add_argument("--host", help="optional explicit IPv4, which must occur in this DLC status host_ips")
    parser.add_argument("--expected-datasets", help="JSON mapping the three train backend IDs to episodes/scenes counts")
    parser.add_argument("--request-timeout", type=float, default=300, help="HTTP and scene-switch timeout seconds")
    args = parser.parse_args()
    return Verifier(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
