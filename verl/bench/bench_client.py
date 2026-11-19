# -*- coding: utf-8 -*-
"""bench_client — verl 侧纯 HTTP 客户端，访问 vln_gym bench 服务。

**解耦关键**：本文件零 vln_gym Python import。bench（vln_demo/src/vln_gym）
只通过 HTTP 契约暴露端点（GET /health /spec /action_space /observation_space
/render，POST /reset /step /close），verl 侧用 stdlib urllib 直连，rgb base64
就地解码。这样 vln_demo 可整体搬到另一台 GPU 服务器，verl 在另一台起，只靠
URL + JSON schema 交互，无 Python 路径耦合。

VLNRemoteClient 复刻自 vln_demo/src/vln_gym/client.py（保持协议一致，便于
bench 侧自测 client 与 verl 侧 client 行为同构），BenchClient 是它的语义
封装（分层 step_fm/step_tool/step_discrete）。

verl rollout worker / RL trainer 用法：
    bench = BenchClient("http://bench-host:8907")
    obs, info = bench.reset(scene="2azQ1b91cZZ", episode_id="11")
    obs, r, term, trunc, info = bench.step_fm(waypoints, normalized=False)
"""
import io
import json
import base64
import urllib.request
import urllib.error

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# HTTP 底座（stdlib only）
# ---------------------------------------------------------------------------
def _post(url, body, timeout=120):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return json.loads(raw)
        except Exception:
            raise RuntimeError(f"POST {url} -> {e.code}: {raw}")


def _get(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def b64_to_rgb(s):
    """base64 jpg -> (H, W, 3) uint8 ndarray。"""
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB"))


class VLNRemoteClient:
    """HTTP 代理的 gym env。所有方法转 HTTP，rgb base64 解回 np.ndarray。

    协议（与 bench server vln_gym.server 一致）：
      GET  /health  /spec  /action_space  /observation_space  /render?mode=
      POST /reset  {scene,episode_id,instruction,seed} -> {obs, info}
      POST /step   {action}                -> {obs,reward,terminated,truncated,info}
      POST /close  {}
    """

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
              options=None):
        body = {}
        if scene is not None:
            body["scene"] = scene
        if episode_id is not None:
            body["episode_id"] = str(episode_id)
        if instruction is not None:
            body["instruction"] = instruction
        if seed is not None:
            body["seed"] = seed
        out = _post(self.url + "/reset", body, self.timeout)
        self.last_info = out.get("info", {})
        obs = out.get("obs", {})
        if isinstance(obs, dict) and obs.get("rgb_b64"):
            obs["rgb"] = b64_to_rgb(obs["rgb_b64"])
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
# 语义封装（分层 step + obs image part）
# ---------------------------------------------------------------------------
def obs_image_part(obs):
    """bench obs.rgb -> 消息内的 image part（PIL 对象）。

    给 vlm_policy 喂 vLLM 用：内部 {"type":"image","image":<PIL>} 格式，
    发 vLLM 前由 vlm_policy.to_openai_messages 转 data url。
    """
    rgb = obs.get("rgb") if isinstance(obs, dict) else None
    if rgb is None:
        return None
    img = Image.fromarray(rgb).convert("RGB")
    w, h = img.size
    return {"type": "image", "image": img, "width": w, "height": h}


class BenchClient:
    """HTTP bench 的语义封装。hold 一个 VLNRemoteClient。"""

    def __init__(self, url="http://127.0.0.1:8907", timeout=180):
        self.cli = VLNRemoteClient(url, timeout=timeout)
        self.url = url

    # ---- 基础 ----
    def health(self):
        return self.cli.health()

    def spec(self):
        return self.cli.spec

    def reset(self, *, scene=None, episode_id=None, instruction=None, seed=None):
        return self.cli.reset(scene=scene, episode_id=episode_id,
                              instruction=instruction, seed=seed)

    def render(self, mode="rgb"):
        return self.cli.render(mode)

    # ---- 分层 step ----
    def step_tool(self, name, arguments=None):
        """tool 层：{name: move_forward|turn|move_backward|stop, arguments}。"""
        return self.cli.step({"name": name, "arguments": arguments or {}})

    def step_fm(self, waypoints, normalized=True, render_final_only=False):
        """fm_trajectory 层：32×[dx,dy,dyaw]。

        render_final_only=True：整条 chunk 全程不渲染，只在停下的真实位姿渲一帧
        （碰撞/stall 照常累积，见 vln_gym env fm 分支）。闭环 RL rollout 只需终点
        观测，用它把 render HTTP 从每 sub-step 一次降到每 chunk 一次。"""
        return self.cli.step({"mode": "fm_trajectory",
                              "waypoints": waypoints,
                              "normalized": normalized,
                              "render_final_only": bool(render_final_only)})

    def step_discrete(self, action):
        """discrete 层：0/1/2/3。"""
        return self.cli.step(int(action))

    def close(self):
        self.cli.close()


def tool_return_text(name, info, prev_info=None):
    """从 bench info 合成给模型的工具返回文本（不依赖 motion state）。

    vln_model.run_agent 用 agent_tools.move_observation/turn_observation（依赖
    累积 motion dict + 直持 env）；这里经 HTTP 只有 info，就地合成等价的、
    模型可读返回。
    """
    if prev_info is None:
        prev_info = {}
    moved = (info.get("steps_used", 0) - prev_info.get("steps_used", 0))
    gps = info.get("gps")
    yaw = info.get("yaw_deg")
    dist = info.get("distance_to_goal")
    collided = info.get("collided_this_step")
    blocked = info.get("blocked_this_step")
    parts = [f"executed {name}"]
    if name in ("move_forward", "move_backward"):
        parts.append(f"sim_steps={moved}")
    elif name == "turn":
        parts.append(f"sim_steps={moved}")
    if collided:
        parts.append("collided=true")
    if blocked:
        parts.append("blocked=true（实际移动小于请求，先转向再走）")
    if gps is not None:
        parts.append(f"gps={gps}")
    if yaw is not None:
        parts.append(f"yaw_deg={yaw:.1f}")
    if dist is not None:
        parts.append(f"distance_to_goal={dist:.2f}")
    return "；".join(parts)
