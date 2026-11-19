"""Reuse the production action expert Tool loop for stochastic GRPO rollout."""
from __future__ import annotations

import copy
import asyncio
import hashlib
import urllib.error
import urllib.request
import importlib.util
import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

from action_head.policy import TorchActionHeadPolicy, TrajectoryTransform
from action_head.query_provider import VllmQueryHidden, VllmQueryStateProvider
from action_head_rl.artifact import load_query_routing_artifact
from action_head_rl.config import TrainScope
from action_head_rl.sde import ReverseSDEConfig, sample_reverse_sde


@dataclass(frozen=True)
class RewardConfig:
    """Verifiable terminal and option-level process reward weights."""

    terminal_success_reward: float = 1.0
    direction_match_reward: float = 0.05
    collision_penalty: float = 0.05
    discount_gamma: float = 1.0
    success_contract: str = "legacy_object_arrival"
    process_reward_cap: float | None = None

    def validate(self) -> None:
        values = {
            "terminal_success_reward": self.terminal_success_reward,
            "direction_match_reward": self.direction_match_reward,
            "collision_penalty": self.collision_penalty,
            "discount_gamma": self.discount_gamma,
        }
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError(f"reward weights must be finite: {values}")
        if any(float(values[key]) < 0 for key in values if key != "discount_gamma"):
            raise ValueError(f"reward magnitudes must be non-negative: {values}")
        if not 0.0 <= float(self.discount_gamma) <= 1.0:
            raise ValueError("discount_gamma must be in [0, 1]")
        if self.success_contract not in ("legacy_object_arrival", "native_success", "native_success_and_stop"):
            raise ValueError("unknown RL terminal success contract")
        if self.process_reward_cap is not None:
            if not math.isfinite(self.process_reward_cap) or self.process_reward_cap < 0:
                raise ValueError("process reward cap must be finite and nonnegative")
            if self.discount_gamma != 1.0:
                raise ValueError("bounded v6 process rewards require gamma=1")


def _option_tool_results(result: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Index the actual model-visible run_action_head return for every option."""
    indexed: dict[str, dict[str, Any]] = {}
    for turn in result.get("turns") or []:
        for call in turn.get("calls") or []:
            raw = call.get("result") or {}
            option_id = str(raw.get("option_id") or "")
            if call.get("name") == "run_action_head" and option_id:
                indexed[option_id] = raw
    return indexed


def compute_episode_reward(result: Mapping[str, Any], records: list[dict[str, Any]],
                           config: RewardConfig) -> tuple[float, dict[str, Any]]:
    """Build episode return and diagnostic per-option process rewards.

    Every action expert option is one decision.  Its tool result supplies a
    direction-consistency label and one collision bit.  Official object arrival
    is a terminal-only reward. With gamma=1 the episode return is exactly the
    process sum plus terminal reward. Episode GRPO normalizes that scalar;
    per-option return-to-go remains available for the archived credit mode.
    """
    config.validate()
    option_order: list[str] = []
    records_by_option: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        option_id = str(record.get("option_id") or "")
        if not option_id:
            continue
        if option_id not in records_by_option:
            option_order.append(option_id)
            records_by_option[option_id] = []
        records_by_option[option_id].append(record)
    tool_results = _option_tool_results(result)
    terminal_success = bool(result.get("official_object_success_at_end",
                                       result.get("success", False)))
    if config.success_contract != "legacy_object_arrival":
        terminal_success = bool(result.get("success", False))
        if config.success_contract == "native_success_and_stop":
            terminal_success = terminal_success and bool(result.get("agent_stopped", False))
    credits = []
    for option_id in option_order:
        raw = tool_results.get(option_id) or {}
        match = (raw.get("direction_alignment") or {}).get("match")
        collision = (bool(raw.get("collision")) if "collision" in raw else
                     any(bool(row.get("collision"))
                         for row in records_by_option[option_id]))
        # Direction labels come from deterministic language/endpoint rules and
        # can be unknown or noisy.  Reward verified agreement, but do not turn
        # disagreement into a potentially incorrect negative training signal.
        direction_reward = config.direction_match_reward if match is True else 0.0
        process_reward = direction_reward - config.collision_penalty * float(collision)
        credits.append({
            "option_id": option_id,
            "option_index": len(credits),
            "direction_match": match if isinstance(match, bool) else None,
            "collision": collision,
            "direction_reward": float(direction_reward),
            "collision_reward": -config.collision_penalty * float(collision),
            "process_reward": float(process_reward),
            "terminal_reward": 0.0,
        })
    if config.process_reward_cap is not None:
        magnitude = sum(abs(item["process_reward"]) for item in credits)
        scale = min(1.0, config.process_reward_cap / magnitude) if magnitude else 1.0
        for item in credits:
            item["unscaled_process_reward"] = item["process_reward"]
            item["unscaled_direction_reward"] = item["direction_reward"]
            item["unscaled_collision_reward"] = item["collision_reward"]
            item["process_reward"] *= scale
            item["direction_reward"] *= scale
            item["collision_reward"] *= scale
    terminal_reward = config.terminal_success_reward * float(terminal_success)
    if credits:
        credits[-1]["terminal_reward"] = float(terminal_reward)
    running = float(terminal_reward if not credits else 0.0)
    for credit in reversed(credits):
        running = (float(credit["process_reward"])
                   + float(credit["terminal_reward"])
                   + config.discount_gamma * running)
        credit["return_to_go"] = float(running)
    episode_return = (float(credits[0]["return_to_go"])
                      if credits else float(terminal_reward))
    return episode_return, {
        "schema": "action-head-multiturn-credit-v1",
        "success_contract": result.get("success_contract", "unknown"),
        "terminal_success": terminal_success,
        "terminal_success_reward": float(terminal_reward),
        "discount_gamma": float(config.discount_gamma),
        "option_count": len(credits),
        "direction_matches": sum(x["direction_match"] is True for x in credits),
        "direction_mismatches": sum(x["direction_match"] is False for x in credits),
        "direction_unknown": sum(x["direction_match"] is None for x in credits),
        "collision_options": sum(bool(x["collision"]) for x in credits),
        "process_reward_sum": float(sum(x["process_reward"] for x in credits)),
        "episode_return": episode_return,
        "option_credits": credits,
        **({"rl_success_contract": config.success_contract,
            "process_reward_cap": config.process_reward_cap}
           if config.success_contract != "legacy_object_arrival" or config.process_reward_cap is not None else {}),
    }


@dataclass(frozen=True)
class RLQueryHidden(VllmQueryHidden):
    query_routed_experts: np.ndarray | None = None
    query_positions: np.ndarray | None = None
    all_routed_experts: np.ndarray | torch.Tensor | None = None
    all_positions: np.ndarray | None = None
    all_input_ids: np.ndarray | None = None
    slow_request: Any = None
    slow_response: Any = None


class RoutingQueryStateProvider(VllmQueryStateProvider):
    """Consume query hidden and the R3 routing decision from one vLLM request."""

    def __init__(self, cache_root: str, *, hidden_size: int, layer_count: int,
                 top_k: int, train_scope: TrainScope, wait_seconds: float = 30.0,
                 query_count: int = 4, continuous_context: bool = False):
        self.continuous_context = bool(continuous_context)
        self.remote_cache = cache_root.rstrip('/') if cache_root.startswith(('http://', 'https://')) else ''
        local_cache = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'action_head_query_downloads')
        os.makedirs(local_cache, exist_ok=True)
        super().__init__(local_cache if self.remote_cache else cache_root,
                         hidden_size=hidden_size, query_count=query_count,
                         wait_seconds=wait_seconds, cleanup=False)
        self.layer_count = layer_count
        self.top_k = top_k
        self.train_scope = train_scope

    def _request_artifact_path(self, response: Any) -> tuple[str, str, str]:
        """Fetch a shared atomic artifact, including a v6 decision-only one."""
        if not isinstance(response, Mapping):
            raise ValueError("missing slow response snapshot")
        request_id = self._safe_request_id(response.get("query_request_id"))
        path = self._path(request_id)
        source_path = path
        if self.remote_cache:
            url = f'{self.remote_cache}/{request_id}.safetensors'
            deadline = time.monotonic() + self.wait_seconds
            last_error = None
            while time.monotonic() < deadline:
                try:
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    with opener.open(url, timeout=min(10, self.wait_seconds)) as response_stream:
                        if (self.continuous_context and response_stream.headers.get(
                                'X-Action-Head-Artifact-Lifetime') != 'continuous-parent-v1'):
                            raise RuntimeError('v8 artifact server lacks continuous-parent retention')
                        with open(path + '.tmp', 'wb') as output:
                            output.write(response_stream.read())
                    os.replace(path + '.tmp', path)
                    break
                except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
                    last_error = exc
                    time.sleep(0.01)
            else:
                raise RuntimeError(f'latent-query artifact was not ready: {last_error}')
            source_path = url
        return request_id, path, source_path

    def __call__(self, *, request: Any, response: Any,
                 contract: Mapping[str, Any]) -> RLQueryHidden:
        del contract
        request_id, path, source_path = self._request_artifact_path(response)
        # Reuse the parent's retry behavior, but parse the richer artifact only
        # after it has become atomically visible.
        self._load_when_ready(path)
        artifact = load_query_routing_artifact(path)
        artifact.validate(
            hidden_size=self.hidden_size,
            layer_count=self.layer_count,
            top_k=self.top_k,
            require_full=True, query_count=self.query_count,
        )
        hidden = artifact.query_hidden.numpy()
        digest = hashlib.sha256(hidden.tobytes()).hexdigest()[:16]
        try:
            return RLQueryHidden(
                query_hidden=np.ascontiguousarray(hidden),
                request_id=f"{request_id}:{digest}",
                source_path=source_path,
                query_routed_experts=artifact.query_routed_experts.numpy(),
                query_positions=artifact.query_positions.numpy(),
                all_routed_experts=(None if artifact.all_routed_experts is None
                                    else artifact.all_routed_experts.numpy()),
                all_positions=(None if artifact.all_positions is None
                               else artifact.all_positions.numpy()),
                all_input_ids=(None if artifact.all_input_ids is None
                               else artifact.all_input_ids.numpy()),
                slow_request=copy.deepcopy(request),
                slow_response=copy.deepcopy(response),
            )
        finally:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


class DecisionRoutingQueryStateProvider(RoutingQueryStateProvider):
    """v6 opt-in: capture every answer and reuse AH artifacts for execution."""

    def begin_episode(self, policy_version: int) -> None:
        self.decision_records = []
        self._captured_query = None
        self.policy_version = int(policy_version)
        self._retained_request_ids = set()
        self._prefix_in_flight = False
        if getattr(self, "v8_prefix_cache", False):
            import uuid
            self._continuous_cache_salt = uuid.uuid4().hex

    def _release_artifact(self, request_id):
        request_id = self._safe_request_id(request_id)
        if self.remote_cache:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            request = urllib.request.Request(
                f'{self.remote_cache}/{request_id}.safetensors', method='DELETE')
            with opener.open(request, timeout=self.wait_seconds) as response:
                if response.status != 204:
                    raise RuntimeError('v8 artifact release was not acknowledged')
        else:
            Path(self._path(request_id)).unlink(missing_ok=True)
        self._retained_request_ids.discard(request_id)

    def end_episode(self):
        if not self.continuous_context:
            return
        if self._prefix_in_flight:
            # A timed-out HTTP call can still be queued in vLLM. Its TP workers
            # may not have opened the parent yet. Preserve that local artifact
            # instead of killing the engine when the late request is admitted.
            logging.getLogger(__name__).warning(
                'v8 parent artifacts retained for an unconfirmed request: %s',
                sorted(self._retained_request_ids))
            return
        for request_id in tuple(self._retained_request_ids):
            self._release_artifact(request_id)

    def continuous_prefix_ids(self):
        cache = (dict(continuous_cache_salt=self._continuous_cache_salt)
                 if getattr(self, "v8_prefix_cache", False) else {})
        if not self.decision_records:
            return cache
        last = self.decision_records[-1]
        self._prefix_in_flight = True
        return dict(**cache, continuous_prefix_ids=last["text_history_ids"].tolist(),
                    continuous_parent_request=last["slow_response"]["query_request_id"])

    def capture_decision(self, *, request, response, decision_index):
        from safetensors.torch import load_file
        from action_head_rl.decision_records import build_text_record
        from action_head_rl.artifact import (own_query_capture_tensors,
                                             query_routing_artifact_from_tensors)

        if self.continuous_context:
            # The response arrived only after all TP workers completed the
            # child; no worker can still need to register its parent from disk.
            self._prefix_in_flight = False
            self._retained_request_ids.add(
                self._safe_request_id(response.get("query_request_id")))
        request_id, path, source_path = self._request_artifact_path(response)
        deadline = time.monotonic() + self.wait_seconds
        while True:
            try:
                tensors = own_query_capture_tensors(load_file(path, device="cpu"))
                break
            except FileNotFoundError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"v6 decision artifact was not ready: {request_id}")
                time.sleep(0.01)
        try:
            self._captured_query = None
            self.decision_records.append(build_text_record(
                tensors, request=request, response=response,
                decision_index=decision_index, policy_version=self.policy_version))
            if "query_hidden" in tensors:
                artifact = query_routing_artifact_from_tensors(tensors)
                artifact.validate(hidden_size=self.hidden_size, layer_count=self.layer_count,
                                  top_k=self.top_k, require_full=True, query_count=self.query_count)
                hidden = artifact.query_hidden.numpy()
                digest = hashlib.sha256(hidden.tobytes()).hexdigest()[:16]
                self._captured_query = RLQueryHidden(
                    query_hidden=np.ascontiguousarray(hidden),
                    request_id=f"{request_id}:{digest}", source_path=source_path,
                    query_routed_experts=artifact.query_routed_experts.numpy(),
                    query_positions=artifact.query_positions.numpy(),
                    all_routed_experts=artifact.all_routed_experts,
                    all_positions=artifact.all_positions.numpy(),
                    all_input_ids=artifact.all_input_ids.numpy(),
                    slow_request=copy.deepcopy(request), slow_response=copy.deepcopy(response))
        finally:
            if self.remote_cache or not self.continuous_context:
                Path(path).unlink(missing_ok=True)
        if self.continuous_context:
            for previous in tuple(self._retained_request_ids - {request_id}):
                self._release_artifact(previous)

    def __call__(self, *, request, response, contract):
        del request, contract
        cached = self._captured_query
        request_id = self._safe_request_id(response.get("query_request_id"))
        if cached is None or cached.request_id.split(":", 1)[0] != request_id:
            raise ValueError("v6 AH request has no matching captured query artifact")
        self._captured_query = None
        return cached


class StochasticActionHeadPolicy(TorchActionHeadPolicy):
    """Production policy adapter with reverse-SDE exploration and audit data."""

    def __init__(self, *args, sde_config: ReverseSDEConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.sde_config = sde_config
        self.records: list[dict[str, Any]] = []

    @torch.no_grad()
    def predict(self, option):
        bundle = option.slow_condition
        if not isinstance(bundle, RLQueryHidden):
            raise TypeError("RL rollout requires an RLQueryHidden artifact")
        cfg = self.model.config
        queries = self._tensor(bundle.query_hidden, "query hidden", cfg.qwen_hidden_size)
        reference_rgb = self._rgb(option.slow_reference_observation)
        current_rgb = self._rgb(option.current_observation)
        reference = option.policy_cache.get("reference_visual")
        if reference is None:
            reference = self._visual(reference_rgb)
            option.policy_cache["reference_visual"] = reference
        current = self._visual(current_rgb)
        condition = self.model.encode_condition(
            queries, reference[0], current[0], reference[1], current[1])
        generator = torch.Generator(device=self.device)
        seed = int(self.seed or 0) + len(self.records)
        generator.manual_seed(seed)
        sampled = sample_reverse_sde(
            self.model.velocity, condition, self.sde_config, generator=generator)
        # These values are immutable for the lifetime of one option. Keep a
        # single CPU object graph and let every chunk reference it. Preserving
        # object identity also lets Ray serialize the large tensors once per
        # option instead of once per chunk.
        shared = option.policy_cache.get("rl_option_payload")
        if shared is None:
            shared = {
                "slow_request": bundle.slow_request,
                "slow_response": bundle.slow_response,
                "query_routed_experts": torch.from_numpy(
                    bundle.query_routed_experts),
                "query_positions": torch.from_numpy(bundle.query_positions),
                "all_routed_experts": (
                    None if bundle.all_routed_experts is None
                    else torch.as_tensor(bundle.all_routed_experts)),
                "all_positions": (
                    None if bundle.all_positions is None
                    else torch.from_numpy(bundle.all_positions)),
                "all_input_ids": (
                    None if bundle.all_input_ids is None
                    else torch.from_numpy(bundle.all_input_ids)),
                "reference_visual_tokens": (
                    reference[0].detach().float().cpu()),
                "reference_visual_mask": reference[1].detach().cpu(),
                "timesteps": sampled["timesteps"].detach().float().cpu(),
                "signed_dt": float(sampled["signed_dt"]),
                "noise_scales": sampled["noise_scales"].detach().float().cpu(),
                "var_floor": self.sde_config.var_floor,
            }
            option.policy_cache["rl_option_payload"] = shared
        self.records.append({
            "option_id": option.option_id,
            "chunk_index": option.chunks,
            "execution_status": "predicted",
            **shared,
            "current_visual_tokens": current[0].detach().float().cpu(),
            "current_visual_mask": current[1].detach().cpu(),
            "states": sampled["states"].detach().float().cpu(),
            "successors": sampled["successors"].detach().float().cpu(),
            "old_logprob": sampled["old_logprob"].detach().float().cpu(),
            "seed": seed,
            "collision": False,
            "env_reward": 0.0,
            "gym_feedback": {},
        })
        trajectory = self.trajectory_transform.apply(sampled["trajectory"])
        return trajectory[0].float().cpu().numpy()

    def mark_execution(self, option, execution) -> None:
        for record in reversed(self.records):
            if (record["option_id"] == option.option_id
                    and record["chunk_index"] == option.chunks - 1):
                record["collision"] = bool(execution.collision)
                record["blocked"] = bool(execution.blocked)
                record["moved_m"] = float(execution.moved_m)
                record["env_reward"] = float(execution.env_reward)
                record["gym_feedback"] = copy.deepcopy(execution.gym_feedback)
                record["execution_status"] = "executed"
                return


def colocated_fast_policy_runtime_env(cuda_visible_device: str) -> dict[str, Any]:
    """Bind a borrowed GPU before Ray imports Torch or deserializes the actor.

    A zero-GPU Ray actor otherwise receives an empty CUDA_VISIBLE_DEVICES.
    Changing it in __init__ is too late if an import has queried the driver.
    """
    if not cuda_visible_device or ',' in cuda_visible_device:
        raise ValueError('colocated fast policy requires one explicit GPU')
    return {'env_vars': {
        'CUDA_VISIBLE_DEVICES': cuda_visible_device,
        'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES': '1',
    }}


class FastPolicyWorker:
    """One batched action expert shared by many CPU episode workers.

    This is the project-specific equivalent of verl-vla's separate rollout
    replica. Incoming replans are coalesced for a short bounded window, while
    every sample retains its original per-episode RNG stream and SDE record.
    """

    def __init__(self, *, checkpoint: str, backbone: str,
                 max_batch_size: int = 8, batch_wait_ms: float = 12.0,
                 sde_steps: int = 3, sde_noise: float = 0.08,
                 policy_version: int = 0, pipeline_batches: bool = False,
                 max_pending_requests: int = 0, device: str = "cuda:0",
                 cpu_threads: int = 1, cuda_visible_device: str | None = None):
        if cuda_visible_device is not None:
            # This actor borrows an explicitly assigned learner GPU only
            # during rollout; Ray must not allocate a second GPU resource.
            colocated_fast_policy_runtime_env(cuda_visible_device)
            if os.environ.get('CUDA_VISIBLE_DEVICES') != cuda_visible_device:
                raise ValueError('colocated fast policy GPU must be bound in Ray runtime_env before import')
            # Production imports may eagerly initialize the already-bound
            # device. This is safe: unlike the former constructor, we never
            # change visibility after that initialization.
            if torch.cuda.is_initialized() and (
                    torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0):
                raise ValueError('colocated fast policy CUDA context must use the single bound GPU')
            if torch.device(device).type != 'cuda':
                raise ValueError('colocated fast policy requires CUDA')
        from action_head.policy import Qwen35VisualTokenEncoder
        from models.qwen36_fast_slow_action_head import Qwen36FastSlowActionHead

        self._initialize_batching(
            max_batch_size=max_batch_size, batch_wait_ms=batch_wait_ms,
            pipeline_batches=pipeline_batches,
            max_pending_requests=max_pending_requests)
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("fast policy supports only CPU or CUDA")
        if cpu_threads <= 0:
            raise ValueError("CPU thread count must be positive")
        if self.device.type == "cpu":
            torch.set_num_threads(cpu_threads)
        # Preserve BF16 model/SDE arithmetic; CPU changes placement, not precision.
        self.dtype = torch.bfloat16
        self.model = Qwen36FastSlowActionHead.from_pretrained(
            checkpoint, map_location="cpu").to(
                device=self.device, dtype=self.dtype).eval()
        self.visual_encoder = Qwen35VisualTokenEncoder.from_pretrained(
            backbone, device=self.device, dtype=self.dtype)
        self.sde_config = ReverseSDEConfig(
            horizon=self.model.config.horizon,
            action_dim=self.model.config.action_dim,
            num_steps=sde_steps, noise_scale=sde_noise)
        self._reference_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._batch_count = 0
        self._request_count = 0
        self._max_observed_batch = 0
        self._forward_seconds = 0.0
        self.policy_version = int(policy_version)

        low = [-0.10273411870002747, -0.10309220850467682,
               -0.5668500065803528]
        high = [0.1031249612569809, 0.10309189558029175,
                0.5747644901275635]
        self.trajectory_transform = TrajectoryTransform.from_quantiles(
            low, high, flip_y=True, flip_yaw=True)

    def _initialize_batching(self, *, max_batch_size: int, batch_wait_ms: float,
                             pipeline_batches: bool,
                             max_pending_requests: int = 0) -> None:
        if max_batch_size <= 0 or batch_wait_ms < 0 or max_pending_requests < 0:
            raise ValueError("invalid fast-policy batching settings")
        self.max_batch_size = int(max_batch_size)
        self.batch_wait_s = float(batch_wait_ms) / 1000.0
        self.pipeline_batches = bool(pipeline_batches)
        self.max_pending_requests = int(max_pending_requests) or max(
            256, 2 * self.max_batch_size)
        if self.max_pending_requests < self.max_batch_size:
            raise ValueError("max_pending_requests must cover one full batch")
        self._pending: list[
            tuple[dict[str, Any], asyncio.Future, float]] = []
        self._flush_task: asyncio.Task | None = None
        self._batch_ready = asyncio.Event()
        self._request_capacity = asyncio.Semaphore(self.max_pending_requests)
        self._gpu_lock = asyncio.Lock()
        # Reuse DynamicBatchingPolicy's single inference-thread design. Its
        # OptionContext / deterministic FM interface cannot serve RL SDE records.
        # Exactly one executor submission is active, so its internal queue does
        # not become a second unbounded inference queue.
        self._gpu_executor = (ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="action-head-rl-fast-policy")
            if self.pipeline_batches else None)
        self._reloading = False
        self._offloaded = False
        self._active_predict_calls = 0
        self._admitted_requests = 0
        self._peak_pending_requests = 0
        self._queue_wait_seconds = 0.0
        self._max_queue_wait_seconds = 0.0
        self._queue_wait_requests = 0
        self._batch_size_histogram: dict[int, int] = {}

    async def predict(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._offloaded:
            raise RuntimeError('cannot predict while fast policy is offloaded')
        if self._reloading:
            raise RuntimeError("cannot predict while fast policy is reloading")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        queued_at = time.perf_counter()
        self._active_predict_calls += 1
        try:
            if self.pipeline_batches:
                # Backpressure stays on the actor event loop; requests already
                # admitted (including the active batch) always remain bounded.
                await self._request_capacity.acquire()
                self._admitted_requests += 1
                self._peak_pending_requests = max(
                    self._peak_pending_requests, self._admitted_requests)
            self._pending.append((request, future, queued_at))
            self._batch_ready.set()
            if self._flush_task is None:
                flush = (self._flush_pipeline if self.pipeline_batches
                         else self._flush_after_window)
                self._flush_task = loop.create_task(flush())
            return await future
        finally:
            self._active_predict_calls -= 1

    def _record_batch_dispatch(self, items) -> None:
        started = time.perf_counter()
        waits = [started - queued_at for _, _, queued_at in items]
        self._queue_wait_seconds += sum(waits)
        self._queue_wait_requests += len(items)
        self._max_queue_wait_seconds = max(
            self._max_queue_wait_seconds, max(waits, default=0.0))
        self._batch_size_histogram[len(items)] = (
            self._batch_size_histogram.get(len(items), 0) + 1)

    def _finish_batch(self, items, *, results=None, error=None) -> None:
        if error is None and len(results) != len(items):
            error = RuntimeError("fast-policy returned an incomplete batch")
        for index, (_, future, _) in enumerate(items):
            if not future.done():
                if error is None:
                    future.set_result(results[index])
                else:
                    future.set_exception(error)
            if self.pipeline_batches:
                # Cancellation of one caller must not release its slot while
                # its request is still queued or executing on the GPU.
                self._admitted_requests -= 1
                self._request_capacity.release()

    async def _flush_after_window(self) -> None:
        if self.batch_wait_s:
            await asyncio.sleep(self.batch_wait_s)
        items = self._pending[:self.max_batch_size]
        del self._pending[:len(items)]
        self._flush_task = None
        try:
            self._record_batch_dispatch(items)
            results = self._predict_batch([request for request, _, _ in items])
            self._finish_batch(items, results=results)
        except Exception as exc:
            self._finish_batch(items, error=exc)
        if self._pending and self._flush_task is None:
            self._flush_task = asyncio.get_running_loop().create_task(
                self._flush_after_window())

    async def _flush_pipeline(self) -> None:
        """Accept the next batch while one dedicated thread owns the GPU.

        Like DynamicBatchingPolicy, a partial batch waits only until its first
        request's deadline, and a full batch never pays the coalescing window.
        Unlike submitting one future per batch, this keeps GPU work and cache
        mutations strictly serialized without buffering work in the executor.
        """
        items = []
        try:
            while self._pending:
                deadline = self._pending[0][2] + self.batch_wait_s
                while len(self._pending) < self.max_batch_size:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    self._batch_ready.clear()
                    try:
                        await asyncio.wait_for(self._batch_ready.wait(), remaining)
                    except asyncio.TimeoutError:
                        break
                items = self._pending[:self.max_batch_size]
                del self._pending[:len(items)]
                try:
                    async with self._gpu_lock:
                        self._record_batch_dispatch(items)
                        results = await asyncio.get_running_loop().run_in_executor(
                            self._gpu_executor, self._predict_batch,
                            [request for request, _, _ in items])
                    self._finish_batch(items, results=results)
                except Exception as exc:
                    self._finish_batch(items, error=exc)
                items = []
        finally:
            self._flush_task = None
            if items or self._pending:
                # Event-loop shutdown or cancellation must not strand callers.
                self._finish_batch(
                    items + self._pending,
                    error=RuntimeError("fast-policy batch dispatcher stopped"))
                self._pending.clear()

    def _tensor(self, value: Any, hidden_size: int) -> torch.Tensor:
        # Visual/query arrays can come from read-only mmap or PIL buffers.
        # Materialize a writable host copy before wrapping it with torch;
        # although inference never mutates the input, torch otherwise warns
        # that a future write would have undefined behaviour.
        if isinstance(value, np.ndarray) and not value.flags.writeable:
            value = value.copy()
        tensor = torch.as_tensor(value, device=self.device, dtype=self.dtype)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3 or tensor.shape[0] != 1 \
                or tensor.shape[-1] != hidden_size:
            raise ValueError(f"invalid fast-policy tensor shape {tuple(tensor.shape)}")
        return tensor.contiguous()

    def _normalized_visual(self, value: Any) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, mask = value if isinstance(value, tuple) else (value, None)
        tokens = self._tensor(tokens, self.model.config.visual_hidden_size)
        if mask is None:
            mask = torch.ones(tokens.shape[:2], device=self.device, dtype=torch.bool)
        else:
            mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
        return tokens, mask

    def _pad_visuals(self, values):
        width = max(int(tokens.shape[1]) for tokens, _ in values)
        result = torch.zeros(
            (len(values), width, values[0][0].shape[-1]),
            device=self.device, dtype=self.dtype)
        mask = torch.zeros(
            (len(values), width), device=self.device, dtype=torch.bool)
        for index, (tokens, token_mask) in enumerate(values):
            length = int(tokens.shape[1])
            result[index, :length] = tokens[0]
            mask[index, :length] = token_mask[0]
        return result, mask

    @torch.no_grad()
    def _predict_batch(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        started = time.perf_counter()
        stale = [request.get("policy_version") for request in requests
                 if int(request.get("policy_version", -1)) != self.policy_version]
        if stale:
            raise RuntimeError(
                f"fast-policy version mismatch: worker={self.policy_version}, "
                f"requests={stale[:8]}")
        cfg = self.model.config
        missing = [request for request in requests
                   if request["cache_key"] not in self._reference_cache]
        images = [request["reference_rgb"] for request in missing]
        images.extend(request["current_rgb"] for request in requests)
        encoded = self.visual_encoder.encode_batch(images)
        for index, request in enumerate(missing):
            self._reference_cache[request["cache_key"]] = \
                self._normalized_visual(encoded[index])
        currents = [self._normalized_visual(value)
                    for value in encoded[len(missing):]]
        references = [self._reference_cache[request["cache_key"]]
                      for request in requests]
        reference_tokens, reference_mask = self._pad_visuals(references)
        current_tokens, current_mask = self._pad_visuals(currents)
        queries = torch.cat([
            self._tensor(request["query_hidden"], cfg.qwen_hidden_size)
            for request in requests
        ], dim=0)
        condition = self.model.encode_condition(
            queries, reference_tokens, current_tokens,
            reference_mask, current_mask)

        initial_noises = []
        per_request_steps = []
        for request in requests:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(request["action_seed"]))
            shape = (1, cfg.horizon, cfg.action_dim)
            initial_noises.append(torch.randn(
                shape, device=self.device, dtype=self.dtype,
                generator=generator))
            per_request_steps.append(torch.stack([
                torch.randn(shape, device=self.device, dtype=self.dtype,
                            generator=generator)
                for _ in range(self.sde_config.num_steps)
            ], dim=0))
        step_noises = torch.cat(per_request_steps, dim=1)
        sampled = sample_reverse_sde(
            self.model.velocity, condition, self.sde_config,
            initial_noise=torch.cat(initial_noises, dim=0),
            step_noises=step_noises)
        trajectories = self.trajectory_transform.apply(
            sampled["trajectory"]).float().cpu().numpy()
        elapsed = time.perf_counter() - started
        self._batch_count += 1
        self._request_count += len(requests)
        self._max_observed_batch = max(self._max_observed_batch, len(requests))
        self._forward_seconds += elapsed
        max_memory = self._peak_cuda_memory_gb()
        results = []
        for index, request in enumerate(requests):
            reference = references[index]
            results.append({
                "trajectory": trajectories[index],
                "reference_visual_tokens": (
                    reference[0].detach().float().cpu()
                    if request.get("include_reference") else None),
                "reference_visual_mask": (
                    reference[1].detach().cpu()
                    if request.get("include_reference") else None),
                "current_visual_tokens": currents[index][0].detach().float().cpu(),
                "current_visual_mask": currents[index][1].detach().cpu(),
                "states": sampled["states"][index:index + 1].detach().float().cpu(),
                "successors": sampled["successors"][index:index + 1].detach().float().cpu(),
                "old_logprob": sampled["old_logprob"][index:index + 1].detach().float().cpu(),
                "timesteps": sampled["timesteps"].detach().float().cpu(),
                "signed_dt": float(sampled["signed_dt"]),
                "noise_scales": sampled["noise_scales"].detach().float().cpu(),
                "var_floor": self.sde_config.var_floor,
                "inference_batch_size": len(requests),
                "forward_seconds": elapsed,
                "max_memory_allocated_gb": max_memory,
                "policy_version": self.policy_version,
            })
        return results

    async def clear_episode(self, episode_key: str) -> None:
        if self.pipeline_batches:
            async with self._gpu_lock:
                await asyncio.get_running_loop().run_in_executor(
                    self._gpu_executor, self._clear_episode, episode_key)
        else:
            self._clear_episode(episode_key)

    def _clear_episode(self, episode_key: str) -> None:
        prefix = episode_key + "/"
        for key in [key for key in self._reference_cache if key.startswith(prefix)]:
            del self._reference_cache[key]

    async def reload_action_head(self, checkpoint: str,
                                 policy_version: int) -> None:
        if (self._pending or self._flush_task is not None
                or self._active_predict_calls or self._reloading):
            raise RuntimeError("cannot reload fast policy while requests are pending")
        # Set before the first await: a predict RPC may arrive while loading CPU
        # weights or waiting for a cache-clear operation to finish.
        self._reloading = True
        try:
            if self.pipeline_batches:
                async with self._gpu_lock:
                    await asyncio.get_running_loop().run_in_executor(
                        self._gpu_executor, self._reload_action_head,
                        checkpoint, policy_version)
            else:
                self._reload_action_head(checkpoint, policy_version)
        finally:
            self._reloading = False

    async def set_rollout_resident(self, resident: bool) -> dict[str, Any]:
        """Move the complete fast policy at a quiescent rollout/update boundary.

        Keep parameters and per-episode RNG unchanged. Drop only the frozen
        vision cache; a resumed option supplies its original reference frame.
        Never overlap GPU model moves with predictions or checkpoint reloads.
        """
        if (self._pending or self._flush_task is not None
                or self._active_predict_calls or self._reloading):
            raise RuntimeError('cannot move fast policy while requests are pending')
        if self.device.type != 'cuda':
            raise ValueError('residency transitions require a CUDA fast policy')
        self._reloading = True
        try:
            if self.pipeline_batches:
                async with self._gpu_lock:
                    await asyncio.get_running_loop().run_in_executor(
                        self._gpu_executor, self._set_rollout_resident, resident)
            else:
                self._set_rollout_resident(resident)
        finally:
            self._reloading = False
        return dict(resident=not self._offloaded, policy_version=self.policy_version)

    def _set_rollout_resident(self, resident: bool) -> None:
        target = self.device if resident else torch.device('cpu')
        self._reference_cache.clear()
        self.model.to(device=target)
        self.visual_encoder.vision_model.to(device=target)
        self.visual_encoder.device = target
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        self._offloaded = not resident

    def _reload_action_head(self, checkpoint: str, policy_version: int) -> None:
        from models.qwen36_fast_slow_action_head import Qwen36FastSlowActionHead

        updated = Qwen36FastSlowActionHead.from_pretrained(
            checkpoint, map_location="cpu").to(
                device='cpu' if self._offloaded else self.device, dtype=self.dtype).eval()
        if updated.config != self.model.config:
            raise ValueError("published Action Head config changed during training")
        self.model.load_state_dict(updated.state_dict(), strict=True)
        self.policy_version = int(policy_version)
        self._reference_cache.clear()

    def _peak_cuda_memory_gb(self) -> float:
        if self.device.type == "cpu":
            return 0.0
        return torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

    async def stats(self) -> dict[str, Any]:
        return {
            "requests": self._request_count,
            "batches": self._batch_count,
            "mean_batch_size": (
                self._request_count / self._batch_count if self._batch_count else 0.0),
            "max_batch_size": self._max_observed_batch,
            "forward_seconds": self._forward_seconds,
            "reference_cache_entries": len(self._reference_cache),
            "max_memory_allocated_gb": self._peak_cuda_memory_gb(),
            "device": str(self.device),
            "cuda_visible_devices": os.environ.get('CUDA_VISIBLE_DEVICES'),
            "resident": not self._offloaded,
            "policy_version": self.policy_version,
            "pipeline_batches": self.pipeline_batches,
            "max_pending_requests": self.max_pending_requests,
            "pending_requests": self._admitted_requests,
            "peak_pending_requests": self._peak_pending_requests,
            "batch_size_histogram": dict(self._batch_size_histogram),
            "queue_wait_seconds": self._queue_wait_seconds,
            "mean_queue_wait_seconds": (
                self._queue_wait_seconds / self._queue_wait_requests
                if self._queue_wait_requests else 0.0),
            "max_queue_wait_seconds": self._max_queue_wait_seconds,
            "queue_wait_requests": self._queue_wait_requests,
        }


class RemoteStochasticActionHeadPolicy:
    """CPU-side policy proxy retaining the exact rollout record contract."""

    def __init__(self, worker: Any, *, checkpoint: str, worker_id: int = 0,
                 policy_version: int = 0):
        from models.qwen36_fast_slow_action_head import FastSlowActionHeadConfig

        config_path = Path(checkpoint) / "fast_slow_action_head_config.json"
        config = FastSlowActionHeadConfig.from_dict(json.loads(
            config_path.read_text(encoding="utf-8")))
        self.worker = worker
        self.worker_id = int(worker_id)
        self.model = SimpleNamespace(config=config)
        self.records: list[dict[str, Any]] = []
        self.seed = 0
        self.episode_key = ""
        self._episode_generation = 0
        self.actor_max_memory_gb = 0.0
        self.policy_version = int(policy_version)

    def begin_episode(self, seed: int) -> None:
        self.records.clear()
        self.seed = int(seed)
        self._episode_generation += 1
        self.episode_key = (
            f"episode-worker={self.worker_id}/generation={self._episode_generation}/"
            f"seed={self.seed}")
        self.actor_max_memory_gb = 0.0

    def end_episode(self) -> None:
        import ray

        ray.get(self.worker.clear_episode.remote(self.episode_key))

    def predict(self, option) -> np.ndarray:
        import ray

        bundle = option.slow_condition
        if not isinstance(bundle, RLQueryHidden):
            raise TypeError("RL rollout requires an RLQueryHidden artifact")
        cache_key = f"{self.episode_key}/{option.option_id}"
        reference = option.policy_cache.get("remote_reference_visual")
        action_seed = int(self.seed or 0) + len(self.records)
        response = ray.get(self.worker.predict.remote({
            "cache_key": cache_key,
            "query_hidden": bundle.query_hidden,
            "reference_rgb": TorchActionHeadPolicy._rgb(
                option.slow_reference_observation),
            "current_rgb": TorchActionHeadPolicy._rgb(option.current_observation),
            "action_seed": action_seed,
            "include_reference": reference is None,
            "policy_version": self.policy_version,
        }))
        if int(response.get("policy_version", -1)) != self.policy_version:
            raise RuntimeError(
                "fast-policy response version changed during synchronous rollout")
        if reference is None:
            reference = (
                response["reference_visual_tokens"],
                response["reference_visual_mask"],
            )
            if reference[0] is None or reference[1] is None:
                raise RuntimeError("fast policy omitted initial reference visual")
            option.policy_cache["remote_reference_visual"] = reference
        shared = option.policy_cache.get("rl_option_payload")
        if shared is None:
            shared = {
                "slow_request": bundle.slow_request,
                "slow_response": bundle.slow_response,
                "query_routed_experts": torch.from_numpy(bundle.query_routed_experts),
                "query_positions": torch.from_numpy(bundle.query_positions),
                "all_routed_experts": (
                    None if bundle.all_routed_experts is None
                    else torch.as_tensor(bundle.all_routed_experts)),
                "all_positions": (
                    None if bundle.all_positions is None
                    else torch.from_numpy(bundle.all_positions)),
                "all_input_ids": (
                    None if bundle.all_input_ids is None
                    else torch.from_numpy(bundle.all_input_ids)),
                "reference_visual_tokens": reference[0],
                "reference_visual_mask": reference[1],
                "timesteps": response["timesteps"],
                "signed_dt": response["signed_dt"],
                "noise_scales": response["noise_scales"],
                "var_floor": response["var_floor"],
            }
            option.policy_cache["rl_option_payload"] = shared
        self.records.append({
            "option_id": option.option_id,
            "chunk_index": option.chunks,
            "execution_status": "predicted",
            **shared,
            "current_visual_tokens": response["current_visual_tokens"],
            "current_visual_mask": response["current_visual_mask"],
            "states": response["states"],
            "successors": response["successors"],
            "old_logprob": response["old_logprob"],
            "seed": action_seed,
            "collision": False,
            "env_reward": 0.0,
            "gym_feedback": {},
            "inference_batch_size": response["inference_batch_size"],
            "fast_policy_forward_seconds": response["forward_seconds"],
            "policy_version": self.policy_version,
        })
        self.actor_max_memory_gb = max(
            self.actor_max_memory_gb,
            float(response.get("max_memory_allocated_gb", 0.0) or 0.0))
        return np.asarray(response["trajectory"], dtype=np.float32)

    def mark_execution(self, option, execution) -> None:
        for record in reversed(self.records):
            if (record["option_id"] == option.option_id
                    and record["chunk_index"] == option.chunks - 1):
                record["collision"] = bool(execution.collision)
                record["blocked"] = bool(execution.blocked)
                record["moved_m"] = float(execution.moved_m)
                record["env_reward"] = float(execution.env_reward)
                record["gym_feedback"] = copy.deepcopy(execution.gym_feedback)
                record["execution_status"] = "executed"
                return


def _evaluator_module(repo: Path):
    path = repo / "verl/scripts/eval/action_head_fm_p0_p2/curriculum_action_head_eval.py"
    spec = importlib.util.spec_from_file_location("_action_head_eval_for_rl", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class ActionHeadRolloutWorker:
    """One Ray actor's reusable vLLM→Tool→HTTP-environment rollout state."""

    def __init__(self, *, repo: str, checkpoint: str, backbone: str,
                 slow_model: str, query_cache: str, bench_url: str, vllm_url: str,
                 train_scope: str = "head_query", seed: int = 0,
                 harness_version: str = "v1",
                 max_turns: int = 8, max_tool_calls: int = 8,
                 one_tool_call_per_turn: bool = False,
                 max_completion_tokens: int = 0,
                 prefix_steps: int = 4, fast_step_limit: int = 4,
                 sde_steps: int = 3, sde_noise: float = 0.08,
                 terminal_success_reward: float = 1.0,
                 direction_match_reward: float = 0.05,
                 collision_penalty: float = 0.05,
                 discount_gamma: float = 1.0,
                 timeout: int = 600, fast_policy_worker: Any = None,
                 worker_id: int = 0, policy_version: int = 0,
                 zero_copy_payload: bool = False,
                 text_policy: bool = False, text_temperature: float = 1.0,
                 success_contract: str = "legacy_object_arrival",
                 process_reward_cap: float | None = None, v8_prefix_cache: bool = False,
                 record_behavior_logprobs: bool = False):
        from action_head.runtime import ActionHeadRuntimeConfig

        self.repo = Path(repo)
        self.scope = TrainScope(train_scope)
        if text_policy and (harness_version not in ("v6", "v8") or self.scope is not TrainScope.FULL):
            raise ValueError("joint text policy requires v6/full")
        self.text_policy = bool(text_policy)
        self.policy_version = int(policy_version)
        self.eval_module = _evaluator_module(self.repo)
        args = SimpleNamespace(
            checkpoint=checkpoint, backbone=backbone, query_cache=query_cache,
            bench_url=bench_url, vllm_url=vllm_url, timeout=timeout,
            policy_seed=seed, max_turns=max_turns,
            action_head_as_tool_version=harness_version,
            max_tool_calls=max_tool_calls, keep_last_images=8, mode="instruct",
            one_tool_call_per_turn=one_tool_call_per_turn,
            max_completion_tokens=max_completion_tokens,
            record_decisions=self.text_policy, text_temperature=text_temperature,
            record_behavior_logprobs=record_behavior_logprobs,
            shared_action_head_policy=(
                None if fast_policy_worker is None else
                RemoteStochasticActionHeadPolicy(
                    fast_policy_worker, checkpoint=checkpoint,
                    worker_id=worker_id, policy_version=policy_version)),
        )
        evaluator = self.eval_module.ActionHeadEvaluator(args)
        old_policy = evaluator.runtime.tool_executor.runtime.policy
        if fast_policy_worker is None:
            stochastic = StochasticActionHeadPolicy(
                old_policy.model, old_policy.visual_encoder,
                device=old_policy.device, dtype=old_policy.dtype,
                trajectory_transform=old_policy.trajectory_transform, seed=seed,
                sde_config=ReverseSDEConfig(
                    horizon=old_policy.model.config.horizon,
                    action_dim=old_policy.model.config.action_dim,
                    num_steps=sde_steps, noise_scale=sde_noise),
            )
        else:
            stochastic = old_policy
        qwen_config = json.loads((Path(slow_model) / "config.json").read_text())
        text = qwen_config.get("text_config", qwen_config)
        provider_class = DecisionRoutingQueryStateProvider if self.text_policy else RoutingQueryStateProvider
        provider = provider_class(
            query_cache,
            hidden_size=old_policy.model.config.qwen_hidden_size,
            layer_count=int(text["num_hidden_layers"]),
            top_k=int(text["num_experts_per_tok"]),
            train_scope=self.scope,
            query_count=old_policy.model.config.query_num_queries,
            continuous_context=harness_version == "v8",
        )
        option_runtime = evaluator.runtime.tool_executor.runtime
        provider.v8_prefix_cache = bool(v8_prefix_cache)
        option_runtime.policy = stochastic
        # The evaluator owns execution semantics (including v7's unbounded
        # coordinates/yaw). Override only the two requested rollout budgets.
        option_runtime.config = replace(option_runtime.config,
            prefix_steps=prefix_steps, fast_step_limit=fast_step_limit)
        option_runtime.config.validate()
        option_runtime.progress_callback = stochastic.mark_execution
        evaluator.runtime.tool_executor.slow_condition_provider = provider
        if harness_version == "v8":
            evaluator.continuous_prefix_provider = provider.continuous_prefix_ids
        self.query_provider = provider
        if self.text_policy:
            evaluator.decision_observer = provider.capture_decision
        self.evaluator = evaluator
        self.policy = stochastic
        self.reward_config = RewardConfig(
            terminal_success_reward=terminal_success_reward,
            direction_match_reward=direction_match_reward,
            collision_penalty=collision_penalty,
            discount_gamma=discount_gamma,
            success_contract=success_contract, process_reward_cap=process_reward_cap,
        )
        self.reward_config.validate()
        self.zero_copy_payload = bool(zero_copy_payload)

    def reload_action_head(self, checkpoint: str,
                           policy_version: int | None = None) -> None:
        """Install a newly published head without rebuilding the visual tower."""
        if policy_version is not None:
            self.policy_version = int(policy_version)
        if isinstance(self.policy, RemoteStochasticActionHeadPolicy):
            if policy_version is None:
                raise ValueError("remote fast policy reload requires policy_version")
            self.policy.policy_version = int(policy_version)
            return
        from models.qwen36_fast_slow_action_head import Qwen36FastSlowActionHead

        updated = Qwen36FastSlowActionHead.from_pretrained(
            checkpoint, map_location="cpu").to(
                device=self.policy.device, dtype=self.policy.dtype).eval()
        self.policy.model = updated

    def evaluate(self, case: dict, seed: int, policy_version: int) -> dict:
        """Return only evaluation evidence from the current joint policy."""
        if self.policy_version != int(policy_version):
            raise RuntimeError("evaluation actor has a stale policy version")
        zero_copy = self.zero_copy_payload
        try:
            # No learner consumes heldout records: skip needless Plasma puts
            # while retaining the identical sampler and native R3 protocol.
            self.zero_copy_payload = False
            row = self.rollout(case, seed)
        finally:
            self.zero_copy_payload = zero_copy
        # Training tensors/ObjectRefs must not leave this actor during heldout
        # evaluation. The same request, action sampler and reward still run.
        row.pop("chunks", None)
        row.pop("decisions", None)
        row["policy_version"] = self.policy_version
        self.policy.records.clear()
        if self.text_policy:
            self.query_provider.decision_records.clear()
        return row

    def _begin_rollout(self, seed: int) -> None:
        if self.text_policy:
            self.query_provider.begin_episode(self.policy_version)
        if isinstance(self.policy, RemoteStochasticActionHeadPolicy):
            self.policy.begin_episode(seed)
        else:
            self.policy.records.clear()
            self.policy.seed = int(seed)

    def _end_rollout(self) -> None:
        try:
            if self.text_policy:
                self.query_provider.end_episode()
        finally:
            if isinstance(self.policy, RemoteStochasticActionHeadPolicy):
                self.policy.end_episode()

    def begin_partial_rollout(self, case: dict, seed: int, *, transfer_queue_destination=None) -> dict:
        """Reset once and retain the existing evaluator at its first request.

        Actor RPC serialization makes a returned boundary quiescent: no slow
        request or FM tool is running. The group scheduler may retain this
        cursor across a weight update without replaying environment actions.
        """
        if getattr(self, '_partial_rollout', None) is not None:
            raise RuntimeError('previous partial rollout must finish or be discarded')
        if (not self.text_policy or not self.query_provider.continuous_context
                or not isinstance(self.policy, RemoteStochasticActionHeadPolicy)):
            raise ValueError('partial rollout requires v8 text and the shared FM policy')
        if transfer_queue_destination is not None:
            import transfer_queue as tq
            tq.init()
        self._begin_rollout(seed)
        state = dict(case=case, seed=seed, started=time.perf_counter(),
                     steps=self.evaluator.rollout_steps(case, seed), versions=[],
                     transfer_queue_destination=transfer_queue_destination)
        self._partial_rollout = state
        try:
            state['query'] = next(state['steps'])
        except StopIteration as done:
            return self._finish_partial_rollout(done.value)
        except BaseException:
            self.discard_partial_rollout()
            raise
        # Building the next request reserves its R3 parent, but no HTTP request
        # has yet been sent. A discarded idle cursor can release that parent.
        self.query_provider._prefix_in_flight = False
        return dict(done=False, decisions=0)

    def advance_partial_rollout(self, policy_version: int) -> dict:
        """Execute exactly one decision, including its complete FM tool call."""
        state = getattr(self, '_partial_rollout', None)
        if state is None:
            raise RuntimeError('no partial rollout to advance')
        if self.policy_version != int(policy_version) or self.policy.policy_version != int(policy_version):
            raise ValueError('partial rollout must use the acknowledged published policy')
        if state['versions'] and int(policy_version) < state['versions'][-1]:
            raise ValueError('partial rollout policy version cannot move backwards')
        provider = self.query_provider
        query = state['query']
        kwargs = dict(query.kwargs)
        if provider.policy_version != int(policy_version):
            provider.policy_version = int(policy_version)
            if getattr(provider, 'v8_prefix_cache', False):
                import uuid
                provider._continuous_cache_salt = uuid.uuid4().hex
                kwargs['continuous_cache_salt'] = provider._continuous_cache_salt
        # Preserve canonical tokens and native historical R3 parent. Only KV
        # cache identity changes with weights; history and budgets never reset.
        provider._prefix_in_flight = bool(kwargs.get('continuous_parent_request'))
        state['versions'].append(int(policy_version))
        try:
            try:
                reply = query.execute(self.eval_module.chat_query, kwargs=kwargs)
            except Exception as exc:
                state['query'] = state['steps'].throw(exc)
            else:
                state['query'] = state['steps'].send(reply)
        except StopIteration as done:
            return self._finish_partial_rollout(done.value)
        except BaseException:
            self.discard_partial_rollout()
            raise
        provider._prefix_in_flight = False
        return dict(done=False, decisions=len(state['versions']),
                    min_policy_version=min(state['versions']),
                    max_policy_version=max(state['versions']), actor_pid=os.getpid())

    def _finish_partial_rollout(self, result: dict) -> dict:
        state = self._partial_rollout
        try:
            self._end_rollout()
            output = self._finish_rollout(state['case'], state['seed'], result,
                                          time.perf_counter() - state['started'])
            output['partial_rollout'] = dict(decision_policy_versions=state['versions'],
                min_policy_version=min(state['versions']) if state['versions'] else self.policy_version,
                max_policy_version=max(state['versions']) if state['versions'] else self.policy_version)
            if state['transfer_queue_destination'] is not None:
                from action_head_rl.partial_groups import store_partial_rollout
                receipt = store_partial_rollout(output, state['transfer_queue_destination'])
                # The queue now owns the payload. Keeping the completed lists
                # here would pin all old images/routes until this actor resets.
                self.policy.records = []
                self.query_provider.decision_records = []
                self.query_provider._captured_query = None
            else:
                return dict(done=True, rollout=output)
        finally:
            state['steps'].close()
            self._partial_rollout = None
        # Only queued completion reaches here: the native put has acknowledged
        # ownership, and no live navigation cursor consumes these records.
        # Clear this frame too, before gc/malloc_trim; retaining ``output``
        # would make the apparent cleanup leave the large tensors alive.
        del output, result, state
        from action_head_rl.cpu_memory import release_cpu_memory
        release_started = time.perf_counter()
        receipt['cpu_memory'] = release_cpu_memory()
        receipt['cpu_memory']['release_seconds'] = time.perf_counter() - release_started
        return receipt

    def discard_partial_rollout(self) -> dict:
        """Release a stale/failed cursor without manufacturing a terminal reward."""
        state = getattr(self, '_partial_rollout', None)
        if state is None:
            raise RuntimeError('no partial rollout to discard')
        try:
            state['steps'].close()
            self._end_rollout()
        finally:
            self._partial_rollout = None
        return dict(discarded=True, decisions=len(state['versions']))

    def rollout(self, case: dict, seed: int, *, fixture_capture: dict | None = None) -> dict:
        """Run one independent, complete multi-turn slow/fast-system episode."""
        if getattr(self, '_partial_rollout', None) is not None:
            raise RuntimeError('cannot replace an active partial rollout')
        self._begin_rollout(seed)
        t0 = time.perf_counter()
        try:
            result = self.evaluator.rollout(case, seed)
        finally:
            self._end_rollout()
        return self._finish_rollout(case, seed, result, time.perf_counter() - t0,
                                    fixture_capture=fixture_capture)

    def _finish_rollout(self, case, seed, result, episode_seconds, *, fixture_capture=None):
        # Reuse the per-turn Qwen wall clock already recorded by the
        # evaluator (curriculum_action_head_eval.py).  This is the slow-system
        # request wall time (including service queueing). Partial episodes also
        # include scheduler/update pauses, reported separately from FM/env work.
        try:
            slow_qwen_seconds = sum(
                float(turn.get("dt_s", 0.0) or 0.0)
                for turn in (result.get("turns") or []))
        except Exception:
            slow_qwen_seconds = 0.0
        rollout_queue_wait_seconds = sum(float(turn.get('queue_wait_s', 0.0) or 0.0)
                                        for turn in (result.get('turns') or []))
        if isinstance(self.policy, RemoteStochasticActionHeadPolicy):
            actor_max_memory_gb = self.policy.actor_max_memory_gb
        else:
            try:
                actor_max_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            except Exception:
                actor_max_memory_gb = 0.0
        records = self.policy.records
        boundary_credit = (result.get("final_credit") or {}).get("schema") == "three-source-boundary-stop-v1"
        if ((boundary_credit or any(row.get("execution_status") == "predicted" for row in records))
                and result.get("evaluation_valid") is True and not result.get("infra_error")):
            from action_head_rl.final_credit import executed_fm_records
            records, execution_audit = executed_fm_records(result, records)
            if boundary_credit:
                result["final_credit"]["fm_execution_audit"] = execution_audit
            else:
                result["fm_execution_audit"] = execution_audit
        reward, reward_components = compute_episode_reward(
            result, records, self.reward_config)
        decisions = self.query_provider.decision_records if self.text_policy else []
        fixture_audit = None
        if fixture_capture is not None:
            from action_head_rl.joint_fixture import capture_raw_fixture
            fixture_audit = capture_raw_fixture(
                fixture_capture, records, decisions, result, case=case, seed=seed,
                policy_version=self.policy_version)
        if self.text_policy and os.environ.get("ACTION_HEAD_JOINT_TEXT_FM_FORWARD", "0") == "1":
            from action_head_rl.joint_policy import annotate_rollout_decisions
            annotate_rollout_decisions(records, decisions, result)
        if self.zero_copy_payload:
            import ray
            shared_keys = (
                "slow_request", "slow_response", "query_routed_experts",
                "query_positions", "all_routed_experts", "all_positions",
                "all_input_ids", "reference_visual_tokens", "reference_visual_mask",
                "timesteps", "noise_scales", "signed_dt", "var_floor",
            )
            chunk_keys = (
                "current_visual_tokens", "current_visual_mask", "states", "successors",
            )
            packed_records = []
            for option_id in dict.fromkeys(str(row.get("option_id")) for row in records):
                option_records = [row for row in records if str(row.get("option_id")) == option_id]
                first = option_records[0]
                packed = {
                    "shared": {key: first[key] for key in shared_keys if key in first},
                    "chunks": [{key: row[key] for key in chunk_keys if key in row}
                               for row in option_records],
                }
                payload_ref = ray.put(packed)
                for index, row in enumerate(option_records):
                    metadata = {key: value for key, value in row.items()
                                if key not in shared_keys and key not in chunk_keys}
                    metadata["_rollout_payload_ref"] = payload_ref
                    metadata["_rollout_payload_index"] = index
                    metadata["token_count"] = int(
                        torch.as_tensor(first["all_input_ids"]).numel())
                    packed_records.append(metadata)
            records = packed_records
            # Reuse the same shared/chunks ObjectRef transport for every
            # high-level decision, including pure STOP episodes. Keep only
            # scalar provenance on the driver so long text/R3 contexts do not
            # bypass the existing zero-copy optimization.
            packed_decisions = []
            for row in decisions:
                shared = {key: value for key, value in row.items()
                          if key in ("slow_request", "slow_response")
                          or torch.is_tensor(value)}
                payload_ref = ray.put({"shared": shared, "chunks": [{}]})
                metadata = {key: value for key, value in row.items() if key not in shared}
                metadata["_rollout_payload_ref"] = payload_ref
                metadata["_rollout_payload_index"] = 0
                metadata["token_count"] = int(torch.as_tensor(row["all_input_ids"]).numel())
                packed_decisions.append(metadata)
            decisions = packed_decisions
        return {
            "case": case,
            "seed": int(seed),
            "reward": reward,
            "reward_components": reward_components,
            "result": result,
            "chunks": records,
            **({"fixture_audit": fixture_audit} if fixture_audit is not None else {}),
            **({"decisions": decisions} if self.text_policy else {}),
            "episode_seconds": episode_seconds,
            "slow_qwen_seconds": slow_qwen_seconds,
            "rollout_queue_wait_seconds": rollout_queue_wait_seconds,
            "actor_max_memory_gb": actor_max_memory_gb,
        }
