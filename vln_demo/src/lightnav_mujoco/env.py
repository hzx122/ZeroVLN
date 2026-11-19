"""Habitat-shaped facade over LightNav-0's standalone MuJoCo web server.

MuJoCo and CasADi stay in LightNav's Python 3.11 process.  The console talks
only to its documented HTTP/WebSocket API, so mounting this backend does not
mix either dependency stack into the main navigation process.

The upstream simulator is deliberately kinematic and ships one bundled
ProcTHOR scene without a goal annotation.  Metrics therefore expose odometry
and path length but keep distance/success/SPL as NaN.  The map pane is an
odometry trace, not an occupancy map.
"""
from __future__ import annotations

import io
import json
import math
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse, urlunparse

import numpy as np
from PIL import Image, ImageDraw

from rs3d.env import (
    Actuation,
    ActionSpec,
    Agent,
    AgentConfig,
    AgentState,
    Episode,
    Instruction,
    Quaternion,
)


STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2, 3
DEFAULT_URL = "http://127.0.0.1:18088"


def wrap_pi(value: float) -> float:
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"MuJoCo snapshot has invalid {name}") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"MuJoCo snapshot has non-finite {name}")
    return result


def pose_from_snapshot(snapshot: dict[str, Any]) -> tuple[float, float, float, float]:
    try:
        pose = snapshot["simulation"]["pose"]
        return tuple(_finite(pose[key], f"pose.{key}") for key in ("x", "y", "z", "yaw"))
    except (KeyError, TypeError) as exc:
        raise RuntimeError("MuJoCo snapshot has no simulation.pose") from exc


def agent_state_from_pose(pose: tuple[float, float, float, float]) -> AgentState:
    """Map MuJoCo +X-forward XY into Habitat's -Z-forward XZ convention."""
    x, y, z, yaw = pose
    position = np.asarray([-y, z, -x], dtype=np.float64)
    # A._yaw reads this quaternion as -yaw: left is negative/right positive,
    # the same sign used by every other console backend.
    rotation = Quaternion(math.cos(yaw / 2.0), 0.0, math.sin(yaw / 2.0), 0.0)
    return AgentState(position, rotation)


class RemoteMujocoClient:
    """Synchronous owner of the MuJoCo server's manual-control channel."""

    def __init__(self, base_url: str = DEFAULT_URL, *, timeout: float = 5.0,
                 linear_speed: float = 2.5,
                 angular_speed: float = math.radians(300.0),
                 connector=None):
        self.base_url = str(base_url).rstrip("/")
        parsed = urlparse(self.base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("LightNav MuJoCo URL must be http(s)://host:port")
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        self.ws_url = urlunparse((ws_scheme, parsed.netloc, "/ws", "", "", ""))
        self.timeout = float(timeout)
        self.linear_speed = float(linear_speed)
        self.angular_speed = float(angular_speed)
        if self.timeout <= 0 or self.linear_speed <= 0 or self.angular_speed <= 0:
            raise ValueError("timeout and control speeds must be positive")
        self._connector = connector
        self._ws = None
        self._manual = False
        self._condition = threading.Condition()
        self._latest: dict[str, Any] | None = None
        self._results: list[dict[str, Any]] = []
        self._reader_thread: threading.Thread | None = None
        self._reader_error: BaseException | None = None
        self._reader_stop = threading.Event()

    def health(self) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(
                    self.base_url + "/api/health", timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LightNav MuJoCo health unavailable: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise RuntimeError("LightNav MuJoCo health returned an invalid payload")
        pose_from_snapshot(payload)
        return payload

    def connect(self) -> None:
        if self._ws is not None:
            return
        connector = self._connector
        if connector is None:
            try:
                import websocket
            except ImportError as exc:
                raise RuntimeError(
                    "LightNav MuJoCo backend requires websocket-client"
                ) from exc
            connector = websocket.create_connection
        self._ws = connector(self.ws_url, timeout=self.timeout, http_proxy_host=None)
        initial = self._recv_json()
        if initial.get("type") != "snapshot":
            self.close()
            raise RuntimeError("LightNav MuJoCo WebSocket did not send an initial snapshot")
        data = initial.get("data")
        if not isinstance(data, dict):
            self.close()
            raise RuntimeError("LightNav MuJoCo initial snapshot has no object data")
        with self._condition:
            self._latest = data
            self._reader_error = None
            self._results.clear()
        self._reader_stop.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="lightnav-mujoco-ws", daemon=True)
        self._reader_thread.start()

    def reset(self) -> dict[str, Any]:
        self.connect()
        self._manual = False
        self._command("reset")
        self._command("acquire_control")
        self._manual = True
        self._twist(0.0, 0.0)
        return self._wait_for_camera()

    def camera(self) -> np.ndarray:
        return self.camera_with_jpeg()[0]

    def camera_with_jpeg(self) -> tuple[np.ndarray, bytes]:
        try:
            with urllib.request.urlopen(
                    self.base_url + "/api/camera.jpg", timeout=self.timeout) as response:
                raw = response.read()
            with Image.open(io.BytesIO(raw)) as opened:
                rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
            return rgb, raw
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeError(f"LightNav MuJoCo camera unavailable: {exc}") from exc

    def snapshot(self) -> dict[str, Any]:
        """Return the latest continuously consumed WebSocket runtime snapshot."""
        return self._snapshot()

    def drive(self, *, forward_m: float = 0.0, yaw_rad: float = 0.0) -> dict[str, Any]:
        if (forward_m == 0.0) == (yaw_rad == 0.0):
            raise ValueError("drive requires exactly one of forward_m or yaw_rad")
        self.connect()
        if not self._manual:
            self._command("acquire_control")
            self._manual = True
        start_snapshot = self._snapshot()
        start_pose = pose_from_snapshot(start_snapshot)
        start_stamp = start_snapshot["simulation"]["camera"].get("stamp_ns")
        if forward_m:
            target = abs(float(forward_m))
            linear = math.copysign(self.linear_speed, forward_m)
            angular = 0.0
            expected = target / self.linear_speed
        else:
            target = abs(float(yaw_rad))
            linear = 0.0
            angular = math.copysign(self.angular_speed, yaw_rad)
            expected = target / self.angular_speed

        # The bundled server renders MuJoCo in the same thread that advances
        # physics. EGL rendering on this host makes simulated time ~30-40x
        # slower than wall time, so a wall-time deadline derived directly from
        # velocity would fire after only a few 5 ms physics ticks.
        deadline = time.monotonic() + expected * 80.0 + self.timeout
        latest = start_snapshot
        actual = 0.0
        try:
            while time.monotonic() < deadline:
                # The upstream command watchdog expires at 0.35 s; this loop is
                # also the heartbeat, not just pose polling.
                self._twist(linear, angular)
                latest = self._wait_snapshot(timeout=0.08)
                pose = pose_from_snapshot(latest)
                if forward_m:
                    actual = math.hypot(pose[0] - start_pose[0], pose[1] - start_pose[1])
                else:
                    actual = abs(wrap_pi(pose[3] - start_pose[3]))
                if actual >= target:
                    break
            else:
                raise TimeoutError(
                    f"LightNav MuJoCo motion timed out ({actual:.3f}/{target:.3f})")
        finally:
            self._twist(0.0, 0.0)

        latest = self._wait_for_camera(after_stamp=start_stamp)
        end_pose = pose_from_snapshot(latest)
        if forward_m:
            actual = math.hypot(end_pose[0] - start_pose[0], end_pose[1] - start_pose[1])
        else:
            actual = abs(wrap_pi(end_pose[3] - start_pose[3]))
        return {
            "snapshot": latest,
            "requested": target,
            "actual": actual,
            # The upstream simulation integrates qpos kinematically and does
            # not expose contacts as a navigation collision signal.
            "collided": False,
            "blocked": actual < target * 0.9,
        }

    def stop(self) -> None:
        if self._ws is None:
            return
        self._twist(0.0, 0.0)

    def close(self) -> None:
        ws, self._ws = self._ws, None
        self._manual = False
        self._reader_stop.set()
        if ws is not None:
            try:
                ws.send(json.dumps({"type": "release_control"}))
            except Exception:
                pass
            try:
                ws.close()
            except Exception:
                pass
        thread, self._reader_thread = self._reader_thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def _wait_for_camera(self, after_stamp=None) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout
        latest = None
        while time.monotonic() < deadline:
            latest = self._wait_snapshot(timeout=min(0.1, deadline - time.monotonic()))
            camera = latest["simulation"].get("camera") or {}
            if camera.get("ready") and (after_stamp is None or camera.get("stamp_ns") != after_stamp):
                return latest
        raise TimeoutError("LightNav MuJoCo camera did not produce a fresh frame")

    def _command(self, kind: str) -> dict[str, Any]:
        self._ws.send(json.dumps({"type": kind}))
        deadline = time.monotonic() + self.timeout
        with self._condition:
            while time.monotonic() < deadline:
                self._raise_reader_error()
                if self._results:
                    message = self._results.pop(0)
                    if not message.get("ok"):
                        raise RuntimeError(
                            f"LightNav MuJoCo {kind} failed: "
                            f"{message.get('message', 'unknown error')}")
                    return message
                self._condition.wait(timeout=max(0.0, deadline - time.monotonic()))
        raise TimeoutError(f"LightNav MuJoCo {kind} did not return a result")

    def _twist(self, linear: float, angular: float) -> None:
        self._ws.send(json.dumps({"type": "twist", "linear": float(linear),
                                  "angular": float(angular)}))

    def _recv_json(self) -> dict[str, Any]:
        try:
            payload = json.loads(self._ws.recv())
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError("LightNav MuJoCo returned invalid WebSocket JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("LightNav MuJoCo WebSocket payload must be an object")
        return payload

    def _reader_loop(self) -> None:
        try:
            while not self._reader_stop.is_set():
                message = self._recv_json()
                kind = message.get("type")
                with self._condition:
                    if kind in ("snapshot", "runtime"):
                        data = message.get("data")
                        if isinstance(data, dict):
                            self._latest = data
                    elif kind == "command_result":
                        self._results.append(message)
                    self._condition.notify_all()
        except BaseException as exc:
            if not self._reader_stop.is_set():
                with self._condition:
                    self._reader_error = exc
                    self._condition.notify_all()

    def _raise_reader_error(self) -> None:
        if self._reader_error is not None:
            raise RuntimeError(
                f"LightNav MuJoCo WebSocket reader failed: {self._reader_error}")

    def _snapshot(self) -> dict[str, Any]:
        with self._condition:
            self._raise_reader_error()
            if self._latest is None:
                raise RuntimeError("LightNav MuJoCo has no runtime snapshot")
            return self._latest

    def _wait_snapshot(self, timeout: float) -> dict[str, Any]:
        with self._condition:
            before = self._latest
            self._condition.wait_for(
                lambda: self._latest is not before or self._reader_error is not None,
                timeout=max(0.0, timeout),
            )
            self._raise_reader_error()
            if self._latest is None:
                raise RuntimeError("LightNav MuJoCo has no runtime snapshot")
            return self._latest


class RemoteMujocoSim:
    def __init__(self, client: RemoteMujocoClient, *, forward_step: float = 0.25,
                 turn_angle: float = 30.0):
        self.client = client
        self.previous_step_collided = False
        self.agents = [Agent(AgentConfig(action_space={
            MOVE_FORWARD: ActionSpec("move_forward", Actuation(float(forward_step))),
            TURN_LEFT: ActionSpec("turn_left", Actuation(float(turn_angle))),
            TURN_RIGHT: ActionSpec("turn_right", Actuation(float(turn_angle))),
        }, height=0.20, radius=0.092, sensor_height=0.165))]

    def get_agent_state(self, agent_id: int = 0) -> AgentState:
        del agent_id
        return agent_state_from_pose(pose_from_snapshot(self.client.snapshot()))

    def act(self, action_id: int) -> dict[str, Any]:
        spec = self.agents[0].agent_config.action_space.get(int(action_id))
        if spec is None:
            raise KeyError(f"no action {action_id!r}")
        amount = float(spec.actuation.amount)
        if spec.name == "move_forward":
            result = self.client.drive(forward_m=amount)
        else:
            sign = 1.0 if spec.name == "turn_left" else -1.0
            result = self.client.drive(yaw_rad=sign * math.radians(amount))
        self.previous_step_collided = bool(result["collided"] or result["blocked"])
        return result

    def observe(self) -> dict[str, Any]:
        rgb, jpeg = self.client.camera_with_jpeg()
        return {"rgb": rgb, "_lightnav_jpeg": jpeg}


class LightNavMujocoEnv:
    """One-scene, goalless environment backed by LightNav's demo process."""

    def __init__(self, url: str = DEFAULT_URL, *, timeout: float = 5.0,
                 forward_step: float = 0.25, turn_angle: float = 30.0,
                 linear_speed: float = 2.5,
                 angular_speed: float = math.radians(300.0),
                 max_episode_steps: int = 0):
        self.client = RemoteMujocoClient(
            url, timeout=timeout, linear_speed=linear_speed,
            angular_speed=angular_speed)
        health = self.client.health()
        scene = health.get("scene") or {}
        remote_id = str(scene.get("id") or "procthor-val-2")
        self.scene = "lightnav-mujoco-" + remote_id
        self.scene_title = f"LightNav MuJoCo · {scene.get('name') or remote_id}"
        self.max_episode_steps = int(max_episode_steps)
        description = (f"LightNav-0 自带 MuJoCo 场景 {scene.get('name') or remote_id}；"
                       "远端 Python 3.11 进程负责渲染与运动学。")
        self.episodes = [Episode(
            episode_id="0",
            scene_id=f"remote/{self.scene}/scene.xml",
            instruction=Instruction("Move forward, then go to the trashcan on the right."),
            info={"scene_description": description},
        )]
        self.current_episode = self.episodes[0]
        self.sim = RemoteMujocoSim(
            self.client, forward_step=forward_step, turn_angle=turn_angle)
        self._stopped = False
        self._steps = 0
        self.collisions = 0
        self.trail: list[np.ndarray] = []

    def reset(self) -> dict[str, np.ndarray]:
        self.client.reset()
        self._stopped = False
        self._steps = 0
        self.collisions = 0
        self.trail = [self.sim.get_agent_state().position.copy()]
        return self.sim.observe()

    def step(self, action, **_kwargs) -> dict[str, np.ndarray]:
        if self.episode_over:
            raise AssertionError("episode is over; call reset()")
        aid = action if isinstance(action, int) else int(action.get("action"))
        if aid == STOP:
            self.client.stop()
            self._stopped = True
        else:
            self.sim.act(aid)
            if self.sim.previous_step_collided:
                self.collisions += 1
        self._steps += 1
        self.trail.append(self.sim.get_agent_state().position.copy())
        return self.sim.observe()

    @property
    def episode_over(self) -> bool:
        return self._stopped or (
            self.max_episode_steps > 0 and self._steps >= self.max_episode_steps)

    def get_metrics(self) -> dict[str, Any]:
        state = self.sim.get_agent_state()
        path = sum(float(np.linalg.norm(b - a))
                   for a, b in zip(self.trail, self.trail[1:]))
        return {
            "distance_to_goal": float("nan"),
            "success": float("nan"),
            "spl": float("nan"),
            "steps": self._steps,
            "collisions": self.collisions,
            "position": state.position.tolist(),
            "path_length_m": round(path, 3),
            "top_down_map": None,
        }

    def title_of(self, scene: str) -> str:
        return self.scene_title if scene == self.scene else ""

    def render_top_down(self, size: int = 512, show_goal: bool = False) -> np.ndarray:
        del show_goal
        image = Image.new("RGB", (size, size), (24, 27, 31))
        draw = ImageDraw.Draw(image)
        for p in range(0, size, size // 8):
            draw.line((p, 0, p, size), fill=(38, 42, 48))
            draw.line((0, p, size, p), fill=(38, 42, 48))
        if not self.trail:
            return np.asarray(image)
        origin = self.trail[0]
        scale = size / 8.0  # eight metres across, centred on reset pose

        def xy(point):
            return (size / 2 + (float(point[0]) - origin[0]) * scale,
                    size / 2 + (float(point[2]) - origin[2]) * scale)

        if len(self.trail) > 1:
            draw.line([xy(p) for p in self.trail], fill=(90, 190, 255), width=3)
        center = xy(self.trail[-1])
        draw.ellipse((center[0] - 7, center[1] - 7,
                      center[0] + 7, center[1] + 7), fill=(255, 220, 0))
        draw.text((12, 12), "odometry trace (8 m)", fill=(210, 215, 222))
        return np.asarray(image)

    def map_heading_degrees(self) -> float:
        return float(-math.degrees(pose_from_snapshot(self.client.snapshot())[3])) % 360.0

    def close(self) -> None:
        self.client.close()
