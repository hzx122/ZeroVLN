"""Minimal action expert Tool runtime with DualVLN-style fixed refresh.

The slow Agent calls ``run_action_head(target, where, plan)`` once.  This
runtime then owns the fast observe/predict/execute-prefix loop and does not call
the VLM again until it returns.  The first version deliberately has no learned
completion/replan/status head: the action expert predicts only one continuous
ego-relative trajectory and the runtime returns on fixed refresh or objective
execution facts such as blocked/invalid/error/interruption.

This module is environment- and model-agnostic.  Production Sage adapters can
reuse the existing trajectory executor; unit tests use deterministic fakes.
"""
from __future__ import annotations

import math
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol, Sequence

import numpy as np

from . import contract as A
from . import protocol as PV
from . import protocol_v2 as V2
from . import protocol_v3 as V3
from . import protocol_v4 as V4
from . import protocol_v4_1 as V41
from . import protocol_v4_2 as V42
from . import protocol_v4_3 as V43
from . import protocol_v4_4 as V44


@dataclass(frozen=True)
class ActionHeadRuntimeConfig:
    prefix_steps: int = 4
    fast_step_limit: int = 4
    max_distance_m: Optional[float] = 3.0
    max_sim_steps: Optional[int] = 64
    max_seconds: Optional[float] = 30.0
    no_progress_limit: int = 2
    min_executable_xy_m: float = 1e-3
    max_abs_xy_m: Optional[float] = 2.0
    max_abs_yaw_rad: Optional[float] = math.pi
    variable_scale_execution: bool = False

    def validate(self) -> None:
        integer_fields = {
            "prefix_steps": self.prefix_steps,
            "fast_step_limit": self.fast_step_limit,
            "no_progress_limit": self.no_progress_limit,
        }
        if self.max_sim_steps is not None:
            integer_fields['max_sim_steps'] = self.max_sim_steps
        bad_int = {k: v for k, v in integer_fields.items()
                   if not isinstance(v, int) or v <= 0}
        if bad_int:
            raise ValueError(f"runtime integer limits must be positive: {bad_int}")
        positive = {
            "max_distance_m": self.max_distance_m,
            "max_seconds": self.max_seconds,
            "max_abs_xy_m": self.max_abs_xy_m,
            "max_abs_yaw_rad": self.max_abs_yaw_rad,
        }
        bad = {k: v for k, v in positive.items()
               if v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0)}
        if bad:
            raise ValueError(f"runtime limits must be finite and positive: {bad}")
        if (not isinstance(self.min_executable_xy_m, (int, float))
                or not math.isfinite(self.min_executable_xy_m)
                or self.min_executable_xy_m < 0):
            raise ValueError('min_executable_xy_m must be finite and nonnegative')


@dataclass
class OptionContext:
    option_id: str
    instruction: str
    contract: Dict[str, Any]
    slow_condition: Any
    slow_reference_observation: Any
    current_observation: Any
    chunks: int = 0
    moved_m: float = 0.0
    sim_steps: int = 0
    blocked_count: int = 0
    no_progress_count: int = 0
    internal_log: list = field(default_factory=list)
    chunk_feedback: list = field(default_factory=list)
    # Policy-owned tensors that are safe to reuse only inside this option.
    # In particular, the slow reference visual tokens must not be recomputed
    # at every fast refresh, while current visual tokens must be.
    policy_cache: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PrefixExecution:
    observation: Any
    moved_m: float
    sim_steps: int
    blocked: bool = False
    collision: bool = False
    episode_ended: bool = False
    # Evaluator/training-only values returned by the real gym request.  They
    # are deliberately excluded from OptionRunResult.model_visible().
    env_reward: float = 0.0
    gym_feedback: Dict[str, Any] = field(default_factory=dict)
    detail: Dict[str, Any] = field(default_factory=dict)
    # v2 model-visible pose in the option-start ego frame. World pose stays in
    # gym_feedback/internal telemetry and must never be serialized to the model.
    actual_end_pose: Optional[Dict[str, float]] = None


@dataclass
class OptionRunResult:
    ok: bool
    return_reason: str
    observation: Any
    moved_m: float
    chunks: int
    sim_steps: int
    blocked: bool
    error: str = ""
    internal_log: Sequence[Dict[str, Any]] = ()
    chunk_feedback: Sequence[Dict[str, Any]] = ()

    def model_visible(self, totals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        totals = totals or {}
        payload = {
            "tool": "run_action_head",
            "ok": bool(self.ok),
            "return_reason": self.return_reason,
            "moved_m": round(float(self.moved_m), 3),
            "chunks": int(self.chunks),
            "blocked": bool(self.blocked),
            "steps_used": int(self.sim_steps),
        }
        if "total_forward_m" in totals:
            payload["total_forward_m"] = round(float(totals["total_forward_m"]), 2)
        if "net_yaw_deg" in totals:
            payload["net_yaw_deg"] = round(float(totals["net_yaw_deg"]), 1)
        if self.error:
            payload["error"] = self.error
        return payload

    def as_tool_result(self, totals: Optional[Dict[str, Any]] = None) -> A.ToolResult:
        raw = self.model_visible(totals)
        return A.ToolResult(
            ok=self.ok,
            model_text=A.compact_json(raw),
            raw=raw,
            with_image=self.observation is not None,
            error=self.error,
            observation=self.observation,
        )

    def model_visible_v2(self, option_id: str) -> Dict[str, Any]:
        last_pose = next((item.get("actual_end_pose")
                          for item in reversed(self.chunk_feedback)
                          if item.get("actual_end_pose") is not None), None)
        net_m = None if last_pose is None else math.hypot(
            float(last_pose["x_m"]), float(last_pose["y_m"]))
        payload = {
            "schema": V2.TOOL_RESULT_SCHEMA,
            "tool": "run_action_head",
            "option_id": option_id,
            "ok": bool(self.ok),
            "return_reason": self.return_reason,
            "completion": "unknown",
            "chunks": int(self.chunks),
            "path_m": round(float(self.moved_m), 3),
            "net_m": None if net_m is None else round(net_m, 3),
            "blocked": bool(self.blocked),
            "collision": any(bool(item.get("collision"))
                             for item in self.chunk_feedback),
            "steps_used": int(self.sim_steps),
        }
        if self.error:
            payload["error"] = self.error
        return payload

    def as_tool_result_v2(self, option_id: str) -> A.ToolResult:
        raw = self.model_visible_v2(option_id)
        frames = tuple({
            "option_id": option_id,
            "chunk": int(item["chunk"]),
            "actual_end_pose": item.get("actual_end_pose"),
            "path_m": round(float(item["path_m"]), 3),
            "blocked": bool(item["blocked"]),
            "collision": bool(item["collision"]),
            "no_progress": bool(item["no_progress"]),
            "observation": item.get("observation"),
            "final_slow_observation": index == len(self.chunk_feedback) - 1,
        } for index, item in enumerate(self.chunk_feedback))
        return A.ToolResult(
            ok=self.ok,
            model_text=A.compact_json(raw),
            raw=raw,
            with_image=bool(frames),
            error=self.error,
            observation=self.observation,
            trace_frames=frames,
        )

    def as_tool_result_v3(self, option_id: str,
                          arguments: Dict[str, Any]) -> A.ToolResult:
        raw = self.model_visible_v2(option_id)
        raw["schema"] = V3.TOOL_RESULT_SCHEMA
        last_pose = next((item.get("actual_end_pose")
                          for item in reversed(self.chunk_feedback)
                          if item.get("actual_end_pose") is not None), None)
        raw["direction_alignment"] = V3.direction_alignment(
            arguments, last_pose)
        frames = tuple({
            "option_id": option_id,
            "chunk": int(item["chunk"]),
            "actual_end_pose": item.get("actual_end_pose"),
            "path_m": round(float(item["path_m"]), 3),
            "blocked": bool(item["blocked"]),
            "collision": bool(item["collision"]),
            "no_progress": bool(item["no_progress"]),
            "observation": item.get("observation"),
            "final_slow_observation": index == len(self.chunk_feedback) - 1,
        } for index, item in enumerate(self.chunk_feedback))
        return A.ToolResult(
            ok=self.ok,
            model_text=A.compact_json(raw),
            raw=raw,
            with_image=bool(frames),
            error=self.error,
            observation=self.observation,
            trace_frames=frames,
        )

    def as_tool_result_v4(self, option_id: str,
                          arguments: Dict[str, Any]) -> A.ToolResult:
        result = self.as_tool_result_v3(option_id, arguments)
        result.raw["schema"] = V4.TOOL_RESULT_SCHEMA
        result.model_text = A.compact_json(result.raw)
        return result

    def as_tool_result_v41(self, option_id: str,
                           arguments: Dict[str, Any]) -> A.ToolResult:
        result = self.as_tool_result_v3(option_id, arguments)
        result.raw["schema"] = V41.TOOL_RESULT_SCHEMA
        result.model_text = A.compact_json(result.raw)
        return result

    def as_tool_result_v42(self, option_id: str,
                           arguments: Dict[str, Any]) -> A.ToolResult:
        result = self.as_tool_result_v3(option_id, arguments)
        result.raw["schema"] = V42.TOOL_RESULT_SCHEMA
        result.model_text = A.compact_json(result.raw)
        return result

    def as_tool_result_v43(self, option_id: str,
                           arguments: Dict[str, Any]) -> A.ToolResult:
        result = self.as_tool_result_v3(option_id, arguments)
        result.raw["schema"] = V43.TOOL_RESULT_SCHEMA
        result.model_text = A.compact_json(result.raw)
        return result

    def as_tool_result_v44(self, option_id: str,
                           arguments: Dict[str, Any]) -> A.ToolResult:
        result = self.as_tool_result_v3(option_id, arguments)
        result.raw["schema"] = V44.TOOL_RESULT_SCHEMA
        result.model_text = A.compact_json(result.raw)
        return result


class ActionHeadPolicy(Protocol):
    def predict(self, option: OptionContext) -> Any:
        """Return one trajectory with shape (T, 3), in metres/radians."""


class PrefixExecutor(Protocol):
    def execute_prefix(self, trajectory_prefix: np.ndarray,
                       option: OptionContext) -> PrefixExecution:
        """Execute the prefix using the real environment and return facts."""


class TrajectorySafetyRejection(ValueError):
    """A finite model prediction was rejected before executing its prefix."""

    def __init__(self, axis: str, maximum: float, limit: float,
                 trajectory: np.ndarray):
        super().__init__(f"Action Head trajectory exceeds configured {axis} safety bound "
                         f"(max_abs={maximum:.8g}, limit={limit:.8g}); "
                         "this predicted trajectory was not executed")
        self.audit = dict(axis=axis, max_abs=maximum, limit=limit,
                          rejected_trajectory=trajectory.tolist())


class ActionHeadToolRuntime:
    def __init__(self, policy: ActionHeadPolicy, executor: PrefixExecutor,
                 config: ActionHeadRuntimeConfig,
                 interrupted: Optional[Callable[[], bool]] = None,
                 progress_callback: Optional[
                     Callable[[OptionContext, PrefixExecution], None]] = None,
                 monotonic: Callable[[], float] = time.monotonic):
        config.validate()
        self.policy = policy
        self.executor = executor
        self.config = config
        self.interrupted = interrupted or (lambda: False)
        self.progress_callback = progress_callback
        self.monotonic = monotonic

    def _trajectory(self, prediction: Any) -> np.ndarray:
        if hasattr(prediction, "detach"):
            prediction = prediction.detach().float().cpu().numpy()
        trajectory = np.asarray(prediction, dtype=np.float32)
        if trajectory.ndim == 3 and trajectory.shape[0] == 1:
            trajectory = trajectory[0]
        if trajectory.ndim != 2 or trajectory.shape[1] != 3 or trajectory.shape[0] == 0:
            raise ValueError(
                f"Action Head trajectory must have shape (T,3), got {trajectory.shape}")
        if self.config.variable_scale_execution:
            trajectory = trajectory[:self.config.prefix_steps]
        if not np.isfinite(trajectory).all():
            raise ValueError("Action Head trajectory contains NaN or Inf")
        max_xy = np.abs(trajectory[:, :2]).max(initial=0.0)
        max_yaw = np.abs(trajectory[:, 2]).max(initial=0.0)
        if (not self.config.variable_scale_execution and self.config.max_abs_xy_m is not None
                and max_xy > self.config.max_abs_xy_m):
            raise TrajectorySafetyRejection("xy", float(max_xy), self.config.max_abs_xy_m, trajectory)
        if (not self.config.variable_scale_execution and self.config.max_abs_yaw_rad is not None
                and max_yaw > self.config.max_abs_yaw_rad):
            raise TrajectorySafetyRejection("yaw", float(max_yaw), self.config.max_abs_yaw_rad, trajectory)
        return trajectory

    def _result(self, option: OptionContext, ok: bool, reason: str,
                error: str = "") -> OptionRunResult:
        return OptionRunResult(
            ok=ok,
            return_reason=reason,
            observation=option.current_observation,
            moved_m=option.moved_m,
            chunks=option.chunks,
            sim_steps=option.sim_steps,
            blocked=option.blocked_count > 0,
            error=error,
            internal_log=tuple(option.internal_log),
            chunk_feedback=tuple(option.chunk_feedback),
        )

    def run(self, option: OptionContext) -> OptionRunResult:
        start = self.monotonic()
        while True:
            if self.interrupted():
                return self._result(option, False, "interrupted")
            if option.chunks >= self.config.fast_step_limit:
                return self._result(option, True, "fast_step_budget_reached")
            if (not self.config.variable_scale_execution and self.config.max_distance_m is not None
                    and option.moved_m >= self.config.max_distance_m):
                return self._result(option, True, "distance_budget_reached")
            if self.config.max_sim_steps is not None and option.sim_steps >= self.config.max_sim_steps:
                return self._result(option, True, "sim_step_budget_reached")
            if self.config.max_seconds is not None and self.monotonic() - start >= self.config.max_seconds:
                return self._result(option, True, "time_budget_reached")

            try:
                predict_started = self.monotonic()
                trajectory = self._trajectory(self.policy.predict(option))
                predict_ms = (self.monotonic() - predict_started) * 1000
                tool_ready = option.policy_cache.get("tool_ready_perf")
                trajectory_ready_ms = ((time.perf_counter() - tool_ready) * 1000
                                       if tool_ready else None)
            except TrajectorySafetyRejection as exc:
                # A boundedness failure is model output feedback, not a lost
                # inference/environment service. Do not clip, wrap or execute
                # it; preserve any earlier chunks and spend the normal call.
                option.internal_log.append(dict(event="trajectory_safety_rejection",
                    chunk=option.chunks + 1, **exc.audit))
                return self._result(option, False, "rejected", str(exc))
            except Exception as exc:
                return self._result(option, False, "error", str(exc))

            prefix = trajectory[:self.config.prefix_steps]
            if np.linalg.norm(prefix[:, :2], axis=1).max(initial=0.0) \
                    < self.config.min_executable_xy_m \
                    and np.abs(prefix[:, 2]).max(initial=0.0) \
                    < self.config.min_executable_xy_m:
                return self._result(option, True, "no_executable_motion")

            try:
                execute_started = self.monotonic()
                if self.config.variable_scale_execution and self.config.max_seconds is not None:
                    remaining = self.config.max_seconds - (self.monotonic() - start)
                    if remaining <= 0:
                        return self._result(option, True, "time_budget_reached")
                    option.policy_cache["execution_seconds_remaining"] = remaining
                else:
                    option.policy_cache.pop("execution_seconds_remaining", None)
                executed = self.executor.execute_prefix(prefix, option)
                execute_ms = (self.monotonic() - execute_started) * 1000
            except Exception as exc:
                return self._result(option, False, "error", str(exc))

            option.chunks += 1
            option.moved_m += max(0.0, float(executed.moved_m))
            option.sim_steps += max(0, int(executed.sim_steps))
            option.current_observation = executed.observation
            option.blocked_count += int(bool(executed.blocked))
            made_progress = float(executed.moved_m) >= self.config.min_executable_xy_m
            option.no_progress_count = 0 if made_progress else option.no_progress_count + 1
            if option.policy_cache.get("continuous_context") and option.chunk_feedback:
                option.chunk_feedback[-1].pop("observation", None)
            option.chunk_feedback.append({
                "chunk": option.chunks,
                "actual_end_pose": executed.actual_end_pose,
                "path_m": float(executed.moved_m),
                "blocked": bool(executed.blocked),
                "collision": bool(executed.collision),
                "no_progress": not made_progress,
                "observation": executed.observation,
            })
            option.internal_log.append({
                "chunk": option.chunks,
                "predict_wall_ms": predict_ms,
                "trajectory_ready_after_tool_ms": trajectory_ready_ms,
                "executor_version": "v7" if self.config.variable_scale_execution else "v6",
                "execute_and_render_wall_ms": execute_ms,
                "trajectory": trajectory.tolist(),
                "executed_prefix": prefix.tolist(),
                "moved_m": float(executed.moved_m),
                "sim_steps": int(executed.sim_steps),
                "blocked": bool(executed.blocked),
                "collision": bool(executed.collision),
                "env_reward": float(executed.env_reward),
                "gym_feedback": dict(executed.gym_feedback),
                "detail": dict(executed.detail),
            })
            if self.progress_callback is not None:
                try:
                    self.progress_callback(option, executed)
                except Exception as exc:
                    # UI/telemetry must never change controller execution or
                    # turn a valid physical prefix into a failed option.
                    option.internal_log[-1]["progress_callback_error"] = str(exc)

            if executed.detail.get("execution_budget_exhausted"):
                return self._result(option, True, "time_budget_reached")
            if executed.episode_ended:
                return self._result(option, True, "episode_ended")
            if executed.blocked or option.no_progress_count >= self.config.no_progress_limit:
                return self._result(option, True, "blocked")


class RunActionHeadExecutor:
    """Bridge from the shared Agent tool registry to the fast runtime."""

    def __init__(self, runtime: ActionHeadToolRuntime,
                 slow_condition_provider: Callable[..., Any]):
        self.runtime = runtime
        self.slow_condition_provider = slow_condition_provider

    def end_episode(self):
        cleanup = getattr(self.slow_condition_provider, "end_episode", None)
        return cleanup() if cleanup is not None else None

    def execute(self, *, option_id: str, instruction: str, arguments: Dict[str, Any],
                current_observation: Any, slow_request_snapshot: Any,
                slow_response_snapshot: Any = None,
                action_head_as_tool_version: str = "v1") -> A.ToolResult:
        version = PV.normalize_version(action_head_as_tool_version)
        protocol = PV.protocol_module(version)
        contract, error = A.validate_action_head_arguments(arguments)
        if contract is None:
            raw = {
                "tool": "run_action_head",
                "ok": False,
                "return_reason": "rejected",
                "error": error,
            }
            if protocol is not None:
                raw.update({
                    "schema": protocol.TOOL_RESULT_SCHEMA,
                    "option_id": option_id,
                    "completion": "unknown",
                    "chunks": 0,
                    "path_m": 0.0,
                    "net_m": 0.0,
                    "blocked": False,
                    "collision": False,
                    "steps_used": 0,
                })
                if version in ("v3", "v4", "v4.1", "v4.2", "v4.3", "v4.4", "v4.5", "v4.6", "v4.7", "v4.8", "v4.9", "v4.10", "v4.11", "v6", "v8"):
                    # The contract was rejected: never inspect its malformed
                    # plan again while constructing the error feedback.
                    raw["direction_alignment"] = V3.direction_alignment(
                        {}, None)
            return A.ToolResult(False, A.compact_json(raw), raw=raw, error=error)
        try:
            condition_started = time.perf_counter()
            slow_condition = self.slow_condition_provider(
                request=slow_request_snapshot,
                response=slow_response_snapshot,
                contract=contract,
            )
            condition_ms = (time.perf_counter() - condition_started) * 1000
        except Exception as exc:
            raw = {
                "tool": "run_action_head",
                "ok": False,
                "return_reason": "error",
                "error": str(exc),
            }
            if protocol is not None:
                raw.update({
                    "schema": protocol.TOOL_RESULT_SCHEMA,
                    "option_id": option_id,
                    "completion": "unknown",
                    "chunks": 0,
                    "path_m": 0.0,
                    "net_m": 0.0,
                    "blocked": False,
                    "collision": False,
                    "steps_used": 0,
                })
                if version in ("v3", "v4", "v4.1", "v4.2", "v4.3", "v4.4", "v4.5", "v4.6", "v4.7", "v4.8", "v4.9", "v4.10", "v4.11", "v6", "v8"):
                    raw["direction_alignment"] = V3.direction_alignment(
                        contract, None)
            return A.ToolResult(False, A.compact_json(raw), raw=raw, error=str(exc))
        option = OptionContext(
            option_id=option_id,
            instruction=instruction,
            contract=contract,
            slow_condition=slow_condition,
            slow_reference_observation=current_observation,
            current_observation=current_observation,
        )
        option.internal_log.append({
            "event": "condition_prepare", "wall_ms": condition_ms,
            "request_id": getattr(slow_condition, "request_id", ""),
            "metadata": getattr(slow_condition, "metadata", {}),
            "slow_inference": (slow_response_snapshot or {}).get("timing", {}),
        })
        option.policy_cache["tool_ready_perf"] = (slow_response_snapshot or {}).get("timing", {}).get("tool_ready_perf")
        option.policy_cache["continuous_context"] = version == "v8"
        result = self.runtime.run(option)
        # Optional evaluator-only trace.  It is deliberately gated by an env
        # variable and never enters ToolResult/model history.  This lets a
        # real console rollout audit each high-level tool contract against the
        # exact online action expert predictions and executed prefixes.
        trace_root = os.environ.get("ACTION_HEAD_EVAL_TRACE_DIR", "").strip()
        if trace_root:
            try:
                from pathlib import Path
                path = Path(trace_root)
                path.mkdir(parents=True, exist_ok=True)
                safe_id = "".join(
                    value if value.isalnum() or value in "-_." else "-"
                    for value in option_id)
                target = path / f"{time.time_ns()}_{safe_id}.json"
                target.write_text(json.dumps({
                    "schema": "action_head_online_option_trace_v1",
                    "option_id": option_id,
                    "driver": (slow_response_snapshot or {}).get("driver"),
                    "scene": (slow_response_snapshot or {}).get("scene"),
                    "episode_id": (slow_response_snapshot or {}).get("episode_id"),
                    "instruction": instruction,
                    "contract": contract,
                    "ok": result.ok,
                    "return_reason": result.return_reason,
                    "moved_m": result.moved_m,
                    "chunks": result.chunks,
                    "sim_steps": result.sim_steps,
                    "blocked": result.blocked,
                    "error": result.error,
                    "internal_log": list(result.internal_log),
                }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except Exception:
                # Evaluation telemetry must never alter controller execution.
                pass
        if version in ("v4.1", "v4.9", "v4.10"):
            tool_result = result.as_tool_result_v41(option_id, contract)
            if version in ("v4.9", "v4.10"):
                tool_result.raw["schema"] = PV.protocol_module(version).TOOL_RESULT_SCHEMA
                tool_result.model_text = A.compact_json(tool_result.raw)
            return tool_result
        if version == "v4.2":
            return result.as_tool_result_v42(option_id, contract)
        if version == "v4.3":
            return result.as_tool_result_v43(option_id, contract)
        if version in ("v4.5", "v4.6", "v4.7", "v4.8", "v4.11", "v6", "v8"):
            tool_result = result.as_tool_result_v43(option_id, contract)
            tool_result.raw["schema"] = PV.protocol_module(version).TOOL_RESULT_SCHEMA
            tool_result.model_text = A.compact_json(tool_result.raw)
            if version == "v8":
                for frame in tool_result.trace_frames:
                    if not frame["final_slow_observation"]:
                        frame.pop("observation", None)
            return tool_result
        if version == "v4.4":
            return result.as_tool_result_v44(option_id, contract)
        if version == "v4":
            return result.as_tool_result_v4(option_id, contract)
        if version == "v3":
            return result.as_tool_result_v3(option_id, contract)
        if version == "v2":
            return result.as_tool_result_v2(option_id)
        return result.as_tool_result()
