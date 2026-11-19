# -*- coding: utf-8 -*-
"""bench_client — 经 HTTP 访问 vln_gym.server 的语义客户端（verl 端访问层底座）。

复用 vln_gym.client.VLNRemoteClient（已实现 reset/step/render + rgb b64 解码），
补三个语义 step 方法（tool / fm_trajectory / discrete）+ 把 obs.rgb_b64 拼成
OpenAI 兼容 image_url 的辅助函数，给 vlm_policy 喂 vLLM 用。

设计：verl rollout worker 里把 bench 当本地 gym env 用，本层是它的 HTTP 代理 +
动作方言适配。不直持 habitat env（与 vln_model.run_agent 的直持 env 不同）——
所有运动执行都经 bench 的分层 step，bench 侧已内含 drive_forward/drive_turn 子步
+ 碰撞/stall，这里不重复造。
"""
import os
import io
import sys
import base64
import math

# vln_demo/src 在 sys.path 上，才能 import vln_gym.client / agent_tools / prompts
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from PIL import Image  # noqa: E402
import agent_tools as A  # noqa: E402
from vln_gym.client import VLNRemoteClient, b64_to_rgb  # noqa: E402


OBSERVATION_CONTRACT_AGENT_V1 = "agent_v1"
OBSERVATION_CONTRACT_LEGACY_ORACLE_V0 = "legacy_oracle_v0"
OBSERVATION_CONTRACTS = (
    OBSERVATION_CONTRACT_AGENT_V1,
    OBSERVATION_CONTRACT_LEGACY_ORACLE_V0,
)


def normalize_observation_contract(value=None):
    """Resolve and validate the model-visible HTTP observation contract."""
    contract = value or os.environ.get(
        "VLN_OBSERVATION_CONTRACT", OBSERVATION_CONTRACT_AGENT_V1)
    if contract not in OBSERVATION_CONTRACTS:
        raise ValueError(
            f"unknown observation contract {contract!r}; "
            f"expected one of {OBSERVATION_CONTRACTS}")
    return contract


def obs_image_part(obs):
    """bench obs.rgb -> 消息内的 image part（PIL 对象，与 vln_model._img_part 同形）。

    内部消息用 {"type":"image","image":<PIL>} 格式（不是 OpenAI image_url），
    由 vlm_policy.to_openai_messages 在发 vLLM 前转成 data url。这样 count_images /
    strip_old_images（都认 type=="image"）能正常工作，与 run_agent 完全一致。
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

    def reset(self, *, scene=None, episode_id=None, instruction=None, seed=None,
              backend=None):
        options = {"backend": backend} if backend is not None else {}
        return self.cli.reset(scene=scene, episode_id=episode_id,
                              instruction=instruction, seed=seed, **options)

    def render(self, mode="rgb"):
        return self.cli.render(mode)

    # ---- 分层 step ----
    def step_tool(self, name, arguments=None):
        """tool 层：{name: move_forward|turn|move_backward|stop, arguments}。"""
        return self.cli.step({"name": name, "arguments": arguments or {}})

    def step_fm(self, waypoints, normalized=True):
        """fm_trajectory 层：32×[dx,dy,dyaw]。"""
        return self.cli.step({"mode": "fm_trajectory",
                              "waypoints": waypoints,
                              "normalized": normalized})

    def step_discrete(self, action):
        """discrete 层：0/1/2/3。"""
        return self.cli.step(int(action))

    def close(self):
        self.cli.close()


def _legacy_oracle_tool_return_text(name, info, prev_info=None):
    """Pre-Gate-A observation, retained only for an explicit A/B baseline."""
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


def _distance(a, b):
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
        return None
    if len(a) < 2 or len(b) < 2:
        return None
    return math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))


def _fallback_tool_result(name, arguments, info, prev_info):
    """Compatibility for a pre-Gate-A bench server.

    New servers return the exact existing ``agent_tools.drive_*`` result in
    ``info.tool_result``.  This fallback derives only model-safe relative
    execution facts; absolute pose and goal distance never enter returned text.
    """
    arguments = arguments or {}
    prev_info = prev_info or {}
    n_steps = max(0, int(info.get("steps_used", 0) or 0)
                  - int(prev_info.get("steps_used", 0) or 0))
    if name in ("move_forward", "move_backward"):
        requested = float(arguments.get("distance", 0.0) or 0.0)
        moved = _distance(prev_info.get("gps"), info.get("gps"))
        moved = 0.0 if moved is None else round(moved, 3)
        blocked = bool(info.get("blocked_this_step"))
        return {"requested_m": requested, "moved_m": moved,
                "forward_m": -moved if name == "move_backward" else moved,
                "n_steps": n_steps, "collided": bool(info.get("collided_this_step")),
                "blocked": blocked, "reason": "blocked" if blocked else "ok",
                "sign": -1 if name == "move_backward" else 1}
    if name == "turn":
        before = prev_info.get("yaw_deg")
        after = info.get("yaw_deg")
        turned = 0.0
        if before is not None and after is not None:
            turned = abs(A.wrap180(float(after) - float(before)))
        return {"requested_deg": float(arguments.get("degrees", 0.0) or 0.0),
                "turned_deg": turned, "n_steps": n_steps, "reason": "ok"}
    return {}


def tool_return_text(name, info, prev_info=None, arguments=None, motion=None,
                     observation_contract=None):
    """Render a bench tool result under an explicit model-visible contract.

    ``agent_v1`` is the default and directly reuses the console observation
    renderers.  Oracle pose/goal fields stay available in ``info`` for scoring
    but are never interpolated into model messages.  ``legacy_oracle_v0`` is
    opt-in and exists only to quantify the historical leakage.
    """
    contract = normalize_observation_contract(observation_contract)
    if contract == OBSERVATION_CONTRACT_LEGACY_ORACLE_V0:
        return _legacy_oracle_tool_return_text(name, info, prev_info)

    if motion is None:
        motion = {"path_dist": 0.0, "yaw": 0.0, "steps": 0}
    result = dict(info.get("tool_result") or
                  _fallback_tool_result(name, arguments, info, prev_info))
    if name in ("move_forward", "move_backward"):
        result.setdefault("sign", -1 if name == "move_backward" else 1)
        signed = -float(result.get("moved_m", 0.0)) \
            if name == "move_backward" else float(result.get("moved_m", 0.0))
        motion["path_dist"] = float(motion.get("path_dist", 0.0)) + signed
        motion["steps"] = int(info.get("steps_used", motion.get("steps", 0)) or 0)
        text, _ = A.move_observation(result, motion, None)
        return text
    if name == "turn":
        direction = A.parse_direction((arguments or {}).get("direction"))
        if direction is None:
            direction = "left"
        signed = -float(result.get("turned_deg", 0.0)) \
            if direction == "left" else float(result.get("turned_deg", 0.0))
        motion["yaw"] = A.wrap180(float(motion.get("yaw", 0.0)) + signed)
        motion["steps"] = int(info.get("steps_used", motion.get("steps", 0)) or 0)
        text, _ = A.turn_observation(result, motion, None, direction)
        return text
    raise ValueError(f"unsupported model-facing tool result {name!r}")
