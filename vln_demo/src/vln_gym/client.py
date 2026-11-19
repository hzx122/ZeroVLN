# -*- coding: utf-8 -*-
"""VLNRemoteClient — 经 HTTP 访问 vln_gym.server 的 gym-style 客户端。

verl rollout worker 里：
    env = VLNRemoteClient("http://bench-host:8899")
    obs, info = env.reset(episode_id="11")
    obs, r, terminated, truncated, info = env.step({"mode":"fm_trajectory",
                                                    "waypoints":wp,
                                                    "normalized":True})
当本地 gym env 用，不依赖 gymnasium（duck-typed）。

smoke：
    python3 -m vln_gym.client smoke --url http://127.0.0.1:8899
跑完四层动作（discrete / tool / fm_trajectory / stop），断言字段齐全。
"""
import os
import sys
import json
import base64
import argparse
import urllib.request
import urllib.error
import io

import numpy as np
from PIL import Image


def _post(url, body, timeout=120):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # 读 body，否则 500 只剩 status 看不到 server 的 error 消息
        raw = e.read().decode("utf-8", "replace")
        try:
            return json.loads(raw)
        except Exception:
            raise RuntimeError(f"POST {url} -> {e.code}: {raw}")


def _get(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def b64_to_rgb(s):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB"))


class VLNRemoteClient:
    """HTTP 代理的 gym env。所有方法转 HTTP，rgb base64 解回 np.ndarray。"""

    def __init__(self, url="http://127.0.0.1:8899", timeout=120):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.last_info = None
        self.action_space = self._get_json("/action_space")
        self.observation_space = self._get_json("/observation_space")
        self.spec = self._get_json("/spec")

    def _get_json(self, path):
        return json.loads(_get(self.url + path, self.timeout).decode("utf-8"))

    def health(self):
        try:
            return self._get_json("/health")
        except Exception:
            return {"ok": False}

    def reset(self, *, scene=None, episode_id=None, instruction=None, seed=None,
              options=None, backend=None):
        body = {}
        if scene is not None:
            body["scene"] = scene
        if episode_id is not None:
            body["episode_id"] = str(episode_id)
        if instruction is not None:
            body["instruction"] = instruction
        if seed is not None:
            body["seed"] = seed
        if backend is not None:
            # 切 backend 时 server 侧重建 env；不传则沿用 boot 时的 backend。
            body["backend"] = backend
        out = _post(self.url + "/reset", body, self.timeout)
        self.last_info = out.get("info", {})
        obs = out.get("obs", {})
        if isinstance(obs, dict) and obs.get("rgb_b64"):
            obs["rgb"] = b64_to_rgb(obs["rgb_b64"])
        if backend is not None and "error" not in out:
            # A backend switch rebuilds the server environment and its catalog.
            # Ordinary episode resets keep the cached spec without another GET.
            self.spec = self._get_json("/spec")
        return obs, out.get("info", {})

    def step(self, action):
        out = _post(self.url + "/step", {"action": action}, self.timeout)
        obs = out.get("obs", {})
        if isinstance(obs, dict) and obs.get("rgb_b64"):
            obs["rgb"] = b64_to_rgb(obs["rgb_b64"])
        self.last_info = out.get("info", {})
        return (obs, out.get("reward", 0.0), out.get("terminated", False),
                out.get("truncated", False), out.get("info", {}))

    def render(self, mode="rgb"):
        raw = _get(self.url + f"/render?mode={mode}", self.timeout)
        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))

    def close(self):
        try:
            _post(self.url + "/close", {}, self.timeout)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------
def _smoke(url):
    c = VLNRemoteClient(url)
    print("[smoke] health:", c.health())
    print("[smoke] scenes:", c.spec["scenes"][:5], "... total",
          len(c.spec["scenes"]))
    sc = c.spec["scenes"][0] if c.spec["scenes"] else "2azQ1b91cZZ"
    ep = c.spec["scene_eps"].get(sc, ["11"])[0]

    obs, info = c.reset(scene=sc, episode_id=ep)
    print(f"[smoke] reset {sc}/{ep}: step={info.get('steps_used')} "
          f"dist={info.get('distance_to_goal')} instr={info.get('instruction','')[:60]!r}")
    assert obs.get("rgb_b64"), "obs.rgb_b64 missing"
    rgb = b64_to_rgb(obs["rgb_b64"])
    print(f"[smoke]   rgb decoded shape={rgb.shape} dtype={rgb.dtype}")
    assert "distance_to_goal" in info and "success" in info, info

    # 1) discrete forward ×4
    for i in range(4):
        obs, r, term, trunc, info = c.step(1)
    print(f"[smoke] discrete forward×4: reward={r:.3f} dist={info.get('distance_to_goal')} "
          f"step={info.get('steps_used')} mode={info.get('mode')}")
    assert info.get("mode") == "discrete"

    # 2) tool move_forward 0.8m
    obs, r, term, trunc, info = c.step({"name": "move_forward",
                                        "arguments": {"distance": 0.8}})
    print(f"[smoke] tool move_forward(0.8): reward={r:.3f} dist={info.get('distance_to_goal')} "
          f"collided={info.get('collided_this_step')} mode={info.get('mode')}")
    assert info.get("mode") == "tool"

    # 3) fm_trajectory: 32×[0.1,0,0] = 直行 0.1m×32（归一化 false，直接给真实米/弧度）
    wps = [[0.1, 0.0, 0.0] for _ in range(32)]
    obs, r, term, trunc, info = c.step({"mode": "fm_trajectory",
                                        "waypoints": wps, "normalized": False})
    wl = info.get("waypoint_log") or []
    print(f"[smoke] fm_trajectory 32×[0.1,0,0]: reward={r:.3f} dist={info.get('distance_to_goal')} "
          f"waypoints={len(wl)} collided={info.get('collided_this_step')} mode={info.get('mode')}")
    assert info.get("mode") == "fm_trajectory"
    assert len(wl) == 32, f"waypoint_log len={len(wl)} != 32"

    # 4) fm 纯转向：32×[0,0,0.05]（左转 ~2.86°×32≈91°，近似原地转一圈）
    wps = [[0.0, 0.0, 0.05] for _ in range(32)]
    obs, r, term, trunc, info = c.step({"mode": "fm_trajectory",
                                        "waypoints": wps, "normalized": False})
    wl = info.get("waypoint_log") or []
    print(f"[smoke] fm_trajectory 32×[0,0,0.05](左转): waypoints={len(wl)} "
          f"yaw_deg={info.get('yaw_deg')}")

    # 5) stop
    obs, r, term, trunc, info = c.step({"name": "stop",
                                        "arguments": {"reason": "smoke done"}})
    print(f"[smoke] stop: terminated={term} episode_over={info.get('episode_over')} "
          f"success={info.get('success')}")
    assert term, "stop should terminate"

    c.close()
    print("[smoke] ALL PASS")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("smoke")
    s.add_argument("--url", default="http://127.0.0.1:8899")
    args = ap.parse_args()
    if args.cmd == "smoke":
        _smoke(args.url)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
