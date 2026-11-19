"""Agent-server client for the remote vln_demo render/environment service.

Only RGB, trajectory prefixes and non-oracle execution facts cross the server
boundary.  Qwen hidden states and action expert tensors never enter this client.
"""
from __future__ import annotations

import base64
import io
import json
import math
import time
import uuid
import urllib.error
import urllib.request
from typing import Any, Dict, Sequence

import numpy as np
from PIL import Image

from .runtime import OptionContext, PrefixExecution
from .protocol import validate_turn_arguments


def _world_pose(info: Dict[str, Any]):
    exact=info.get("oracle_world_pose")
    if isinstance(exact,(list,tuple)) and len(exact)==3:
        values=tuple(float(x) for x in exact)
        if all(math.isfinite(x) for x in values):return values
    xy, yaw = info.get("world_xy"), info.get("world_yaw_deg")
    if (not isinstance(xy, (list, tuple)) or len(xy) < 2
            or yaw is None):
        return None
    try:
        return (float(xy[0]), float(xy[1]), float(yaw))
    except (TypeError, ValueError):
        return None


def _option_ego_pose(start, end):
    if start is None or end is None:
        return None
    x0, y0, yaw0_deg = start
    x1, y1, yaw1_deg = end
    yaw0 = math.radians(yaw0_deg)
    dx, dy = x1 - x0, y1 - y0
    x_m = math.cos(yaw0) * dx + math.sin(yaw0) * dy
    y_m = -math.sin(yaw0) * dx + math.cos(yaw0) * dy
    yaw_deg = (yaw1_deg - yaw0_deg + 180.0) % 360.0 - 180.0
    return {"x_m": round(x_m, 3), "y_m": round(y_m, 3),
            "yaw_deg": round(yaw_deg, 1)}


def _decode_observation(value: Any) -> Dict[str, Any]:
    obs = dict(value or {})
    encoded = obs.pop("rgb_b64", None)
    if encoded:
        image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        obs["rgb"] = np.asarray(image)
    # Never carry pose/oracle fields into the action expert observation channel.
    return {key: obs[key] for key in ("rgb", "step", "instruction", "scene",
                                      "episode_id") if key in obs}


class SourceGeometryError(RuntimeError):
    """The original reset pose is unsupported; no substitute pose was used."""


class RemoteRenderClient:
    def __init__(self, url: str, timeout: float | None = None):
        self.url = url.rstrip("/")
        # Simulation queueing is not an episode failure. None/0 waits for the
        # response; socket disconnects and server errors still propagate once.
        self.timeout = None if timeout is None or timeout == 0 else float(timeout)
        if self.timeout is not None and (not math.isfinite(self.timeout) or self.timeout < 0):
            raise ValueError("simulation timeout must be finite and nonnegative, or None")
        # Evaluator metrics are retained for the console UI only.  They never
        # enter the sanitized observation or model-visible ToolResult.
        self.last_info: Dict[str, Any] = {}

    def _post(self, endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
        request_id = uuid.uuid4().hex
        started = time.monotonic()
        request = urllib.request.Request(
            self.url + endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-Request-ID": request_id},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw_detail = exc.read().decode("utf-8", "replace")
            detail = raw_detail[:500]
            if endpoint == '/reset' and exc.code == 422:
                try:
                    failure = json.loads(raw_detail)
                except ValueError:
                    failure = None
                if (isinstance(failure, dict) and
                        failure.get('error_kind') in ('unsupported_original_pose',
                            'unsupported_original_geometry')):
                    raise SourceGeometryError(str(failure.get('error') or
                        'original reset pose is unsupported')) from None
            raise RuntimeError(
                f"render server {endpoint} returned HTTP {exc.code}: {detail}") from None
        except (TimeoutError, urllib.error.URLError, ConnectionError) as exc:
            # Never replay /step: the server may already have moved the robot.
            raise RuntimeError(
                f"simulation request failed: endpoint={self.url + endpoint} "
                f"request_id={request_id} elapsed_s={time.monotonic()-started:.3f} "
                f"timeout_s={self.timeout} execution_state=unknown "
                f"cause={type(exc).__name__}: {exc}") from exc
        if result.get("error"):
            raise RuntimeError(f"render server {endpoint}: {result['error']}")
        return result

    def render_map(self, show_goal=False) -> Image.Image:
        """Fetch the model-safe map, or explicitly request the human GT display.

        This deliberately stays outside ``_decode_observation`` so the map is
        never exposed to the action expert or the slow model as an observation.
        """
        endpoint = "/map.jpg" if show_goal else "/vln-map.jpg"
        request = urllib.request.Request(self.url + endpoint, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(
                f"render server {endpoint} returned HTTP {exc.code}: {detail}") from None
        except urllib.error.URLError as exc:
            raise RuntimeError(f"render server {endpoint}: {exc.reason}") from None
        try:
            return Image.open(io.BytesIO(payload)).convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"render server {endpoint} returned an invalid image: {exc}") from None

    def reset(self, *, scene: str | None = None, episode_id: str | None = None,
              instruction: str | None = None, seed: int | None = None,
              backend: str | None = None):
        body = {key: value for key, value in {
            "scene": scene,
            "episode_id": episode_id,
            "instruction": instruction,
            "seed": seed,
            "backend": backend,
        }.items() if value is not None}
        result = self._post("/reset", body)
        self.last_info = dict(result.get("info") or {})
        return _decode_observation(result.get("obs")), dict(result.get("info") or {})

    def execute_prefix(self, waypoints: Sequence[Sequence[float]], *, execution_seconds_remaining=None):
        action = {
            "mode": "fm_trajectory",
            "waypoints": [[float(v) for v in waypoint] for waypoint in waypoints],
            "normalized": False,
            "render_final_only": True,
        }
        if execution_seconds_remaining is not None:
            action["execution_seconds_remaining"] = float(execution_seconds_remaining)
        result = self._post("/step", {"action": action})
        self.last_info = dict(result.get("info") or {})
        return (
            _decode_observation(result.get("obs")),
            float(result.get("reward", 0.0) or 0.0),
            bool(result.get("terminated")),
            bool(result.get("truncated")),
            dict(result.get("info") or {}),
        )

    def execute_turn(self, direction: str, degrees: float):
        """Execute one deterministic remote in-place turn.

        Only non-oracle execution facts are returned to the slow system.  The
        full environment info remains in ``last_info`` for evaluator/UI use.
        """
        arguments, error = validate_turn_arguments(
            {"direction": direction, "degrees": degrees}, "v4")
        if arguments is None:
            raise ValueError(error)
        direction, degrees = arguments["direction"], arguments["degrees"]
        result = self._post("/step", {"action": {
            "name": "turn",
            "arguments": {"direction": direction, "degrees": degrees},
        }})
        info = dict(result.get("info") or {})
        self.last_info = info
        tool = dict(info.get("tool_result") or {})
        model_result = {
            "schema": "turn_tool_result_v1",
            "tool": "turn",
            "ok": True,
            "direction": direction,
            "requested_deg": round(float(tool.get("requested_deg", degrees)), 1),
            "turned_deg": round(float(tool.get("turned_deg", degrees)), 1),
            "steps_used": int(tool.get("n_steps", 1) or 1),
        }
        return _decode_observation(result.get("obs")), model_result

    def execute_backoff(self, distance: float):
        """One bounded retreat; availability and call accounting belong to the caller."""
        from .backoff_recovery import backoff_distance
        from verl_access.bench_client import tool_return_text

        distance = backoff_distance(distance)
        previous = self.last_info
        result = self._post("/step", {"action": {
            "name": "move_backward", "arguments": {"distance": distance}}})
        info = dict(result.get("info") or {})
        self.last_info = info
        text = tool_return_text("move_backward", info, previous,
                                arguments={"distance": distance},
                                observation_contract="agent_v1")
        clean, _ = json.JSONDecoder().raw_decode(text)
        tool = dict(info.get("tool_result") or {})
        model_result = {key: clean[key] for key in
                        ("tool", "requested_m", "moved_m", "n_steps", "blocked")}
        model_result.update(schema="backoff_tool_result_v1", ok=True,
                            steps_used=int(clean["n_steps"]),
                            collision=bool(tool.get("collided", info.get("collided_this_step", False))),
                            return_reason=("episode_ended" if result.get("terminated") or result.get("truncated")
                                           else "blocked" if clean["blocked"] else "backoff_complete"))
        return _decode_observation(result.get("obs")), model_result

    def execute_recovery_turn(self, direction: str, degrees: float):
        """Execute one v4.1 recovery turn, splitting at the gym's 90-degree cap."""
        arguments, error = validate_turn_arguments(
            {"direction": direction, "degrees": degrees}, "v4.1")
        if arguments is None:
            raise ValueError(error)
        direction, degrees = arguments["direction"], arguments["degrees"]
        remaining, turned, steps = degrees, 0.0, 0
        observation = None
        while remaining > 1e-6:
            amount = min(90.0, remaining)
            # A legal 90.x-degree recovery must not leave an illegal <0.5
            # degree second call after already rotating the first 90 degrees.
            if 0.0 < remaining - amount < 0.5:
                amount = remaining / 2.0
            observation, result = self.execute_turn(direction, amount)
            turned += float(result["turned_deg"])
            steps += int(result["steps_used"])
            remaining -= amount
        return observation, {
            "schema": "turn_tool_result_v2", "tool": "turn", "ok": True,
            "direction": direction, "requested_deg": round(degrees, 1),
            "turned_deg": round(turned, 1), "steps_used": steps,
            "turn_reason": "previous_option_direction_mismatch",
            "turn_available_after_result": False,
        }

    def ui_metrics(self) -> Dict[str, Any]:
        """Return scoring fields for final UI reporting, never model input."""
        allowed = ("distance_to_goal", "success_radius", "success", "spl",
                   "official_object_success", "official_object_success_detail",
                   "path_length_m", "shortest_path_m")
        return {key: self.last_info[key] for key in allowed
                if key in self.last_info}


class RemotePrefixExecutor:
    """Adapt the render HTTP API to the Verl-owned option runtime."""

    def __init__(self, client: RemoteRenderClient):
        self.client = client

    def execute_prefix(self, trajectory_prefix: np.ndarray,
                       option: OptionContext) -> PrefixExecution:
        before_pose = _world_pose(getattr(self.client, "last_info", {}) or {})
        start_pose = option.policy_cache.setdefault(
            "_runtime_option_start_world_pose", before_pose)
        budget = option.policy_cache.get("execution_seconds_remaining")
        kwargs = {} if budget is None else {"execution_seconds_remaining": budget}
        observation, reward, terminated, truncated, info = self.client.execute_prefix(
            np.asarray(trajectory_prefix, dtype=np.float32).tolist(), **kwargs)
        logs = list(info.get("waypoint_log") or [])
        moved_m = sum(max(0.0, float(item.get("moved_m", 0.0))) for item in logs)
        blocked = bool(info.get("blocked_this_step")) or any(
            bool(item.get("blocked")) for item in logs)
        collision = bool(info.get("collided_this_step")) or any(
            bool(item.get("collided")) for item in logs)
        # Keep detail free of evaluator metrics such as distance/success/pose.
        detail = {
            "execution_budget_exhausted": bool(info.get("execution_budget_exhausted")),
            "waypoints_requested": len(trajectory_prefix),
            "waypoints_executed": len(logs),
            "render_final_only": True,
        }
        return PrefixExecution(
            observation=observation,
            moved_m=moved_m,
            sim_steps=len(logs),
            blocked=blocked,
            collision=collision,
            episode_ended=terminated or truncated,
            env_reward=float(reward),
            # Retain the per-request 3DGS/gym feedback for reward shaping and
            # audit, without exposing oracle metrics to either policy model.
            gym_feedback=dict(info),
            detail=detail,
            actual_end_pose=_option_ego_pose(start_pose, _world_pose(info)),
        )
