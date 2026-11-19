#!/usr/bin/env python3
"""Evaluate the trained DualVLN query + FM action expert on the frozen 262 cases."""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
import math
import os
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parents[4]
SRC = REPO / "vln_demo" / "src"
VERL = REPO / "verl"
for item in (str(SRC), str(VERL)):
    if item not in sys.path:
        sys.path.insert(0, item)

import agent_tools as A  # noqa: E402
import prompts as P  # noqa: E402
from sage.datasets import SageSet  # noqa: E402
from verl_access.bench_client import BenchClient, obs_image_part  # noqa: E402
from verl_access.vlm_policy import to_openai_messages  # noqa: E402
from action_head import vllm_bridge as AHV  # noqa: E402
from action_head import protocol as ACTION_HEAD_PROTOCOL  # noqa: E402
from action_head.agent_runtime import build_action_head_agent_runtime  # noqa: E402
from action_head.policy import TrajectoryTransform  # noqa: E402

SPLIT = "action_head_curriculum"
MODEL = "Qwen3.6-35B"
EXPECTED = {"p0": 37, "p1": 57, "p2": 168}


class SlowQuery(NamedTuple):
    """One slow-model request; resuming its rollout executes tools serially."""

    args: tuple
    kwargs: dict
    timing: dict | None = None

    def execute(self, query_fn, *, kwargs=None):
        """Time the request itself, excluding a suspended rollout cursor."""
        started = time.perf_counter()
        if self.timing is not None:
            self.timing['queue_wait_s'] = started - self.timing['queued_at']
        try:
            return query_fn(*self.args, **(self.kwargs if kwargs is None else kwargs))
        finally:
            if self.timing is not None:
                self.timing['request_s'] = time.perf_counter() - started


def success_weighted_spl(success: bool, path_length_m: float,
                         shortest_path_m: float) -> float:
    """Compute SPL using whichever success contract is active."""
    path = max(float(path_length_m), 0.0)
    shortest = max(float(shortest_path_m), 0.0)
    return float(bool(success)) * shortest / max(path, shortest, 1e-12)


def terminal_world_pose(info: dict):
    """Evaluator-only pose; never inserted into the model's tool feedback."""
    pose = info.get("oracle_world_pose")
    if pose is None and info.get("world_xy") is not None and info.get("world_yaw_deg") is not None:
        pose = [*info["world_xy"], info["world_yaw_deg"]]
    if pose is None or len(pose) != 3:
        return None
    values = [float(value) for value in pose]
    return values if all(math.isfinite(value) for value in values) else None


def stage_of(task) -> str:
    text = f"{task.instruction_type} {task.task_id}".lower()
    match = re.search(r"(?:curriculum/|__)(p[012])(?:_|\b)", text)
    if not match:
        raise ValueError(f"cannot recover stage for {task.task_id}")
    return match.group(1)


def load_cases() -> list[dict]:
    ds = SageSet(SPLIT)
    rows = []
    for scene in ds.list_scenes():
        for task in ds.tasks(scene, 0):
            rows.append({
                "scene": scene,
                "episode_id": task.task_id,
                "instruction": task.instruction,
                "instruction_type": task.instruction_type,
                "stage": stage_of(task),
                "ref_len": float(task.ref_len),
            })
    rows.sort(key=lambda x: (x["stage"], x["scene"], x["episode_id"]))
    counts = Counter(x["stage"] for x in rows)
    if dict(counts) != EXPECTED:
        raise RuntimeError(f"frozen set mismatch: got={dict(counts)} expected={EXPECTED}")
    return rows


def scene_shards(cases: list[dict], count: int) -> list[list[dict]]:
    grouped = {}
    for case in cases:
        grouped.setdefault(case["scene"], []).append(case)
    bins = [[] for _ in range(count)]
    totals = [0.0] * count
    groups = sorted(grouped.values(),
                    key=lambda xs: -sum(max(1.0, x["ref_len"]) for x in xs))
    for group in groups:
        idx = min(range(count), key=lambda i: (totals[i], i))
        bins[idx].extend(group)
        totals[idx] += sum(max(1.0, x["ref_len"]) for x in group)
    return bins


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def result_path(output: Path, case: dict) -> Path:
    return output / "rollouts" / case["stage"] / safe(case["scene"]) / (safe(case["episode_id"]) + ".json")


def atomic_json(path: Path, value: dict) -> None:
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.tmp.', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def deterministic_seed(base: int, case: dict) -> int:
    key = f"{case['scene']}:{case['episode_id']}".encode()
    return base + int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % 1_000_000_000


def chat_query(vllm_url: str, messages: list[dict], mode: str, seed: int,
               timeout: int, query_count: int = 4,
               max_completion_tokens: int = 0, record_decision: bool = False,
               text_temperature: float = 1.0, legal_action_policy=None, continuous_prefix_ids=None, continuous_parent_request=None, continuous_cache_salt=None,
               record_behavior_logprobs: bool = False) -> tuple[str, str, dict, list[dict]]:
    params, thinking = P.sampling_params(mode, None)
    if max_completion_tokens > 0:
        params["max_tokens"] = int(max_completion_tokens)
    payload = {
        "model": MODEL,
        "messages": to_openai_messages(messages),
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {
            "enable_thinking": thinking,
            "preserve_thinking": True,
        },
        "seed": int(seed),
    }
    # Privileged executors need the same brain/history loop without latent
    # query capture. Learned heads retain their existing nonzero query count.
    request_id = AHV.request_queries(payload, query_count) if query_count else ""
    payload.update(params)
    if continuous_cache_salt is not None:
        from action_head.serve.continuous_context import CACHE_KEY
        payload["cache_salt"] = continuous_cache_salt
        payload["vllm_xargs"][CACHE_KEY] = continuous_cache_salt
    if record_decision:
        if not query_count or not math.isfinite(text_temperature) or text_temperature <= 0:
            raise ValueError("v6 text training requires marked requests and positive temperature")
        payload["vllm_xargs"]["action_head_record_decision"] = True
        # Exact untruncated softmax behavior for the one-update text GPG loss.
        # Query-tail forcing remains unchanged and its tokens are never trained.
        payload.update(temperature=float(text_temperature), top_p=1.0, top_k=-1,
                       min_p=0.0, repetition_penalty=1.0,
                       presence_penalty=0.0, frequency_penalty=0.0)
    if continuous_prefix_ids is not None:
        from action_head.serve.continuous_context import PREFIX_KEY, PARENT_KEY
        payload["vllm_xargs"][PREFIX_KEY] = continuous_prefix_ids
        payload["vllm_xargs"][PARENT_KEY] = continuous_parent_request
    if legal_action_policy is not None:
        from action_head_rl.executable_policy import encode_request
        payload["vllm_xargs"]["action_head_executable_policy"] = encode_request(legal_action_policy)
        payload["chat_template_kwargs"]["enable_thinking"] = False
    if record_behavior_logprobs:
        if not record_decision or text_temperature != 1. or legal_action_policy is not None:
            raise ValueError('partial behavior capture requires untruncated temperature-one v3 text sampling')
        payload.update(logprobs=True, top_logprobs=0, return_tokens_as_token_ids=True)
    request = urllib.request.Request(
        vllm_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    chunks, usage, behavior = [], {}, []
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"vLLM HTTP {exc.code}: {body}") from None
    try:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            obj = json.loads(body)
            if obj.get("usage"):
                usage.update(obj["usage"])
            choice = (obj.get("choices") or [{}])[0]
            if record_behavior_logprobs:
                from action_head_rl.behavior_logprobs import sampled_logprobs
                behavior.extend(sampled_logprobs(choice))
            delta = choice.get("delta") or {}
            piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
            if piece:
                chunks.append(piece)
    finally:
        response.close()
    text = "".join(chunks).replace(AHV.QUERY_TOKEN_TEXT, "")
    if record_behavior_logprobs:
        from action_head_rl.behavior_logprobs import MODE
        if not behavior or len(behavior) != usage.get('completion_tokens'):
            raise ValueError('incomplete behavior logprobs in streamed completion')
        usage['behavior_logprobs'] = dict(mode=MODE, token_ids=[row[0] for row in behavior],
                                         logprobs=[row[1] for row in behavior])
    # Preserve the exact JPEG-backed OpenAI messages sent to vLLM so the RL
    # actor can reconstruct precisely the same multimodal prefix.
    return text, request_id, usage, payload["messages"]


class ActionHeadEvaluator:
    def __init__(self, args):
        import torch

        low = [-0.10273411870002747, -0.10309220850467682,
               -0.5668500065803528]
        high = [0.1031249612569809, 0.10309189558029175,
                0.5747644901275635]
        transform = TrajectoryTransform.from_quantiles(
            low, high, flip_y=True, flip_yaw=True)
        self.args = args
        self.action_head_as_tool_version = ACTION_HEAD_PROTOCOL.normalize_version(
            getattr(args, "action_head_as_tool_version", "v1"))
        self.protocol = ACTION_HEAD_PROTOCOL.protocol_module(
            self.action_head_as_tool_version)
        self.bench = BenchClient(args.bench_url, timeout=args.timeout)
        from action_head.runtime import ActionHeadRuntimeConfig
        # The closed-loop contract owns how many predicted points run per
        # fast window; the default remains the historical receding prefix.
        prefix_steps = int(os.environ.get("ACTION_HEAD_EVAL_PREFIX_STEPS", "4"))
        self.runtime = build_action_head_agent_runtime(
            option_config=ActionHeadRuntimeConfig(variable_scale_execution=True,
                prefix_steps=prefix_steps, max_seconds=None),
            checkpoint=args.checkpoint,
            backbone_model=args.backbone,
            query_cache_root=args.query_cache,
            gym_url=args.bench_url,
            device="cuda:0", dtype=torch.bfloat16,
            visual_device="cuda:0", visual_dtype=torch.bfloat16,
            seed=args.policy_seed,
            trajectory_transform=transform,
            shared_policy=getattr(args, "shared_action_head_policy", None),
            shared_condition_provider=getattr(args, "shared_condition_provider", None),
            context_encoder_model=getattr(args, "context_encoder_model", None),
            context_aggregator_checkpoint=getattr(args, "context_aggregator_checkpoint", None),
        )
        self.profile = A.get_tool_profile(
            "action_head",
            action_head_as_tool_version=self.action_head_as_tool_version)

    def initial_messages(self, instruction: str, obs: dict) -> list[dict]:
        image = obs_image_part(obs)
        if image is None:
            raise RuntimeError("reset did not return RGB")
        if self.protocol is not None:
            image = self.protocol.tagged_image_part(image, kind="slow")
            return [
                {"role": "system", "content": [{"type": "text", "text":
                    P.agent_system_prompt(
                        tool_specs=self.profile.tool_specs,
                        action_head_as_tool_version=self.action_head_as_tool_version)}]},
                {"role": "user", "content": [
                    image,
                    {"type": "text", "text": self.protocol.current_view_text() + "\n\n" +
                     self.protocol.task_text(instruction)},
                ]},
            ]
        return [
            {"role": "system", "content": [{"type": "text", "text":
                P.agent_system_prompt(tool_specs=self.profile.tool_specs)}]},
            {"role": "user", "content": [
                image,
                {"type": "text", "text": P.agent_current_view_text() + "\n\n" +
                 P.agent_task_text(instruction)},
            ]},
        ]

    def rollout(self, case: dict, seed: int) -> dict:
        """Synchronous compatibility driver for the resumable rollout."""
        steps = self.rollout_steps(case, seed)
        try:
            query = next(steps)
            while True:
                try:
                    reply = query.execute(chat_query)
                except Exception as exc:
                    query = steps.throw(exc)
                else:
                    query = steps.send(reply)
        except StopIteration as done:
            return done.value
        finally:
            steps.close()

    def rollout_steps(self, case: dict, seed: int):
        """Yield at model requests, retaining this episode's runtime and locals.

        Each simultaneous generator must own a distinct Gym actor. The caller
        must resume generators serially: only model HTTP may run concurrently.
        """
        obs, initial_info = self.runtime.reset_environment(
            scene=case["scene"], episode_id=case["episode_id"],
            instruction=case["instruction"], seed=seed,
            **({"backend": case["backend"]} if "backend" in case else {}))
        from action_head_rl.stop_arrival_contract import start_contract
        boundary_contract = start_contract(self.args, initial_info)
        from sage import benchmark_protocol as benchmark
        benchmark_positions = []
        rule_enabled = os.environ.get("ACTION_HEAD_RULE_CREDIT", "0") == "1"
        rule_current = copy.deepcopy(initial_info.get("rule_credit_state"))
        def benchmark_decision(record):
            nonlocal rule_current
            if rule_enabled:
                info = self.runtime.render_client.last_info or initial_info
                after = copy.deepcopy(info.get("rule_credit_state"))
                record["rule_credit"]["after"] = after
                record["rule_credit"]["rejected"] = bool(record.get("tool_call_error") or any(call.get("ok") is False for call in record.get("calls", [])))
                rule_current = after
            if not benchmark.enabled():
                return False
            info = self.runtime.render_client.last_info or initial_info
            detail = info.get("official_object_success_detail") or {}
            record["benchmark_decision_info"] = detail
            benchmark_positions.append(detail["official_position"])
            return False if boundary_contract is not None else benchmark.auto_stop(info)
        msgs = self.initial_messages(case["instruction"], obs)
        calls_used = 0
        turns = []
        agent_stopped = False
        recovery_episode_ended = False
        why = ""
        infra_error = ""
        episode_abort_reason = ""
        episode_abort_detail = ""
        recovery_turn_available = False
        consecutive_turn_calls = 0
        trajectory_map = (self.protocol.new_trajectory_map_state()
                          if self.action_head_as_tool_version in ("v6", "v8") else None)
        invalid_call_policy = A.invalid_tool_call_policy()
        review_factory = getattr(self.protocol, "STOP_REVIEW_FACTORY", None)
        stop_review = review_factory() if review_factory else None
        # Full RL uses one external budget: one slow-model decision may execute
        # at most one tool call, so max_tool_calls also bounds the decision loop.
        # Legacy evaluation/head-only callers keep the historical multi-call
        # per turn behavior through max_turns.
        one_tool_call_per_turn = bool(
            getattr(self.args, "one_tool_call_per_turn", False))
        turn_limit = (self.args.max_tool_calls if one_tool_call_per_turn
                      else self.args.max_turns)
        for turn in range(turn_limit):
            if calls_used >= self.args.max_tool_calls:
                why = "tool_budget_exhausted"
                break
            if self.action_head_as_tool_version == "v8":
                self.protocol.append_map_snapshot(msgs, trajectory_map)
            canonical_image_preparer = getattr(self, 'canonical_image_preparer', None)
            if canonical_image_preparer is not None:
                canonical_image_preparer(msgs)
            sent = (self.protocol.strip_context_images(
                        msgs, keep_slow=self.args.keep_last_images)
                    if self.protocol is not None
                    else A.strip_old_images(msgs, self.args.keep_last_images))
            if trajectory_map is not None and self.action_head_as_tool_version != "v8":
                sent = self.protocol.inject_trajectory_map(sent, trajectory_map)
            started = time.perf_counter()
            query_timing = {'queued_at': started}
            try:
                context_preparer = getattr(self, 'evaluation_context_preparer', None)
                if context_preparer is not None:
                    sent = context_preparer(sent, decision_index=turn + 1)
                decision_kwargs = ({
                    "record_decision": True,
                    "text_temperature": float(getattr(self.args, "text_temperature", 1.0)),
                    "record_behavior_logprobs": bool(getattr(self.args, 'record_behavior_logprobs', False)),
                } if bool(getattr(self.args, "record_decisions", False)) else {})
                if self.action_head_as_tool_version == "v8" and hasattr(self, "continuous_prefix_provider"):
                    decision_kwargs.update(self.continuous_prefix_provider())
                from action_head_rl.executable_policy import enabled as executable_enabled
                if executable_enabled():
                    from action_head_rl.executable_policy import request_policy
                    decision_kwargs["legal_action_policy"] = request_policy(consecutive_turn_calls)
                text, request_id, usage, slow_messages = yield SlowQuery(
                    (self.args.vllm_url, sent, self.args.mode, seed + turn,
                     self.args.timeout),
                    dict(query_count=self.runtime.query_count,
                         max_completion_tokens=int(getattr(
                             self.args, "max_completion_tokens", 0)),
                         **decision_kwargs), query_timing)
            except Exception as exc:
                # Candidate adapters may deliberately abstain before returning
                # a reply. Reuse the runtime's abort metadata so this rule
                # outcome remains distinct from a model/service failure.
                if getattr(exc, "continue_at_next_depth", False):
                    # A rejected depth spends a decision slot, never a physical
                    # action. Preserve the measured state and image/map history.
                    calls_used += 1
                    feedback = dict(getattr(exc, "feedback", {}))
                    feedback.update(action_executed=False, state_changed=False)
                    observer = getattr(self, 'decision_observer', None)
                    if observer is not None:
                        observer(request=getattr(exc, 'request_snapshot', {}),
                            response=dict(rejected_depth=True, rejection_feedback=feedback),
                            decision_index=turn + 1)
                    turns.append(dict(turn=turn + 1, calls=[], rejected_depth=True,
                        rejection_feedback=feedback,
                        dt_s=round(query_timing.get('request_s', time.perf_counter()-started), 3),
                        queue_wait_s=round(query_timing.get('queue_wait_s', 0.0), 3)))
                    rejected_text = getattr(exc, 'rejected_assistant_text', None)
                    if rejected_text is not None:
                        _, rejected_visible = A.split_thinking(rejected_text)
                        msgs.append({"role": "assistant", "content": [{"type": "text",
                            "text": A.assistant_history_content(rejected_text, rejected_visible,
                                "keep_thinking") or rejected_text}]})
                    msgs.append({"role": "user", "content": [{"type": "text",
                        "text": "The proposed action was rejected and was not executed. "
                                "You remain at the same actual state. Choose a supported action. "
                                + json.dumps(feedback, ensure_ascii=False)}]})
                    completed = getattr(self, 'completed_depth_observer', None)
                    if completed is not None:
                        progress = completed(turns[-1])
                        if progress.get('terminate'):
                            episode_abort_reason = why = progress['terminal_reason']
                            episode_abort_detail = 'consecutive_no_progress=' + str(progress['consecutive_no_progress'])
                            break
                    continue
                episode_abort_reason = str(getattr(exc, "episode_abort_reason", "") or "")
                if episode_abort_reason:
                    episode_abort_detail = str(getattr(exc, "episode_abort_detail", "") or str(exc))
                    why = episode_abort_reason
                else:
                    infra_error = str(exc)
                    why = "model_error"
                break
            # Legacy external drivers can still send replies directly. RL and
            # the synchronous driver execute the timed request above.
            elapsed = query_timing.get('request_s', time.perf_counter() - started)
            thinking, visible = A.split_thinking(text)
            observer = getattr(self, "decision_observer", None)
            if observer is not None:
                try:
                    observer(
                        request=AHV.semantic_request_snapshot(
                            messages=slow_messages, instruction=case["instruction"]),
                        response=AHV.response_snapshot(
                            text=text, visible=visible, usage=usage, query_request_id=request_id),
                        decision_index=turn + 1)
                except Exception as exc:
                    infra_error, why = str(exc), "decision_capture_error"
                    break
            parsed = A.parse_tool_calls(visible)
            valid_names = self.profile.tools_by_name
            calls = [call for call in parsed if call["name"] in valid_names]
            from action_head.console_events import Calls, emit, model_event
            record = {
                "turn": turn + 1,
                "assistant": str(visible or "").strip(),
                "reasoning": str(thinking or "").strip(),
                "request_id": request_id,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "dt_s": round(elapsed, 3),
                "queue_wait_s": round(query_timing.get('queue_wait_s', 0.0), 3),
                "calls": Calls(),
            }
            emit(model_event(record))
            if rule_enabled:
                record["rule_credit"] = {"before": copy.deepcopy(rule_current)}
            if bool(getattr(self.args, "capture_slow_messages", False)):
                # Bootstrap collection needs the exact multimodal prefix that
                # produced the correction ToolCall.  Keep this opt-in so the
                # established evaluation artifact remains byte-for-byte small.
                record["slow_messages"] = slow_messages
            msgs.append({"role": "assistant", "content": [{"type": "text",
                "text": A.assistant_history_content(text, visible, "keep_thinking") or text}]})
            format_error = A.tool_call_batch_error(visible, parsed, valid_names)
            if not calls or ((invalid_call_policy == "feedback" or boundary_contract is not None) and format_error):
                if invalid_call_policy == "terminate":
                    why = "no_valid_tool_call"
                    turns.append(record)
                    if boundary_contract is not None:
                        boundary_contract.observe_record(record, self.runtime.render_client.last_info or initial_info)
                    break
                # An invalid invocation spends one existing decision/tool slot.
                # Keep it outside calls: no named tool or action was executed.
                calls_used += 1
                raw = format_error
                record["tool_call_error"] = raw
                turns.append(record)
                if boundary_contract is not None:
                    boundary_contract.observe_record(record, self.runtime.render_client.last_info or initial_info)
                if benchmark_decision(record):
                    why = "official_goal_radius"
                    break
                if calls_used >= self.args.max_tool_calls:
                    why = "tool_budget_exhausted"
                    break
                msgs.append(A.tool_call_feedback_message(
                    raw, self.args.max_tool_calls - calls_used, self.protocol))
                completed = getattr(self, 'completed_depth_observer', None)
                if completed is not None:
                    progress = completed(record)
                    if progress.get('terminate'):
                        episode_abort_reason = why = progress['terminal_reason']
                        episode_abort_detail = 'consecutive_no_progress=' + str(progress['consecutive_no_progress'])
                        break
                continue
            results = []
            per_turn_limit = (1 if one_tool_call_per_turn or
                              self.action_head_as_tool_version in ("v4", "v4.1", "v4.2", "v4.3", "v4.4", "v4.5", "v4.6", "v4.7", "v4.8", "v4.9", "v4.10", "v4.11", "v6", "v8")
                              else P.MAX_CALLS_PER_TURN)
            for call in calls[:min(per_turn_limit,
                                   self.args.max_tool_calls - calls_used)]:
                calls_used += 1
                name, arguments = call["name"], call["arguments"]
                stop_rejection = (A.rejected_stop_result(arguments)
                                  if name == "stop" and (invalid_call_policy == "feedback" or boundary_contract is not None) else None)
                if stop_rejection is not None:
                    record["calls"].append({"name": name, "arguments": arguments,
                                            "ok": False, "result": stop_rejection})
                    results.append(A.ToolResult(False, A.compact_json(stop_rejection),
                                                raw=stop_rejection))
                    continue
                if name == "stop":
                    review = (stop_review.consider(
                        instruction=case["instruction"],
                        remaining_calls=self.args.max_tool_calls - calls_used,
                        remaining_decisions=turn_limit - turn - 1)
                        if stop_review is not None else None)
                    if review is not None:
                        result = A.ToolResult(False, A.compact_json(review), raw=review)
                        record["calls"].append({"name": name, "arguments": arguments,
                                                "ok": False, "result": review})
                        results.append(result)
                        continue
                    agent_stopped = True
                    stop_obs, reward, terminated, truncated, stop_info = \
                        self.bench.step_tool("stop", arguments)
                    if stop_obs.get("rgb") is not None:
                        obs = stop_obs
                    record["calls"].append({
                        "name": name, "arguments": arguments,
                        "reward": reward, "terminated": terminated,
                        "truncated": truncated, "score_info": stop_info,
                    })
                    why = "agent_stop"
                    break
                if name == "turn" and self.action_head_as_tool_version in ("v4", "v4.1", "v4.2", "v4.3", "v4.4", "v4.5", "v4.6", "v4.7", "v4.8", "v4.9", "v4.10", "v4.11", "v6", "v8"):
                    # Privileged audit only: never add these poses to ToolResult.
                    turn_before_info = copy.deepcopy(getattr(
                        getattr(self.runtime, "render_client", None), "last_info", {}) or {})
                    turn_args, note = ACTION_HEAD_PROTOCOL.validate_turn_arguments(
                        arguments, self.action_head_as_tool_version)
                    direction = turn_args["direction"] if turn_args else None
                    degrees = turn_args["degrees"] if turn_args else None
                    if turn_args is None:
                        raw = ACTION_HEAD_PROTOCOL.rejected_turn_result(note)
                        result = A.ToolResult(
                            False, A.compact_json(raw), raw=raw,
                            error=raw["error"])
                    elif (self.action_head_as_tool_version in ("v4.1", "v4.9", "v4.10")
                          and not recovery_turn_available):
                        raw = {
                            "schema": "turn_tool_result_v2", "tool": "turn",
                            "ok": False, "return_reason": "turn_not_available",
                            "error": ("turn is available only once after the immediately "
                                      "preceding run_action_head direction mismatch"),
                            "steps_used": 0,
                            "required_next_tool": "run_action_head_or_stop",
                            "turn_available_after_result": False,
                        }
                        result = A.ToolResult(
                            False, A.compact_json(raw), raw=raw,
                            error=raw["error"])
                    else:
                        # Invalid arguments/availability rejections must not
                        # mutate the streak; every request still costs budget.
                        consecutive_turn_calls, turn_limit_rejection = (
                            A.enforce_v43_consecutive_turn_limit(
                                self.action_head_as_tool_version, name,
                                consecutive_turn_calls))
                        if turn_limit_rejection is not None:
                            result = A.ToolResult(
                                False, A.compact_json(turn_limit_rejection),
                                raw=turn_limit_rejection,
                                error=turn_limit_rejection["error"])
                            record["calls"].append({
                                "name": name, "arguments": arguments,
                                "ok": False, "result": turn_limit_rejection,
                            })
                            results.append(result)
                            continue
                        try:
                            if self.action_head_as_tool_version in ("v4.1", "v4.9", "v4.10"):
                                turn_obs, raw = self.runtime.execute_recovery_turn(
                                    direction=direction, degrees=degrees)
                                recovery_turn_available = False
                            else:
                                turn_obs, raw = self.runtime.execute_turn(
                                    direction=direction, degrees=degrees)
                            obs = turn_obs or obs
                            result = A.ToolResult(
                                True, A.compact_json(raw), raw=raw,
                                with_image=turn_obs is not None,
                                observation=turn_obs)
                        except Exception as exc:
                            raw = {
                                "schema": "turn_tool_result_v1", "tool": "turn",
                                "ok": False, "return_reason": "error",
                                "direction": direction,
                                "requested_deg": degrees, "steps_used": 0,
                                "error": str(exc),
                            }
                            result = A.ToolResult(
                                False, A.compact_json(raw), raw=raw,
                                error=str(exc))
                    record["calls"].append({
                        "name": name, "arguments": arguments,
                        "ok": result.ok, "result": result.raw,
                        "execution_state_trace": {
                            "before_world_pose": {key: turn_before_info.get(key)
                                for key in ("world_xy", "world_yaw_deg")},
                            "after_world_pose": {key: copy.deepcopy((getattr(
                                getattr(self.runtime, "render_client", None), "last_info", {}) or {}).get(key))
                                for key in ("world_xy", "world_yaw_deg")},
                        },
                    })
                    results.append(result)
                    if trajectory_map is not None and result.ok:
                        trajectory_map.update_turn(result.raw)
                    if not result.ok and result.raw.get("return_reason") == "error":
                        infra_error = result.error or "turn infrastructure error"
                        why = "turn_error"
                        break
                    continue
                if name == "move_backward" and self.action_head_as_tool_version in ("v4.9", "v4.10"):
                    from action_head.backoff_recovery import validate_backoff
                    distance, raw = validate_backoff(msgs, arguments)
                    if raw is None:
                        try:
                            back_obs, raw = self.runtime.execute_backoff(distance=distance)
                            obs = back_obs or obs
                            recovery_turn_available = False
                            recovery_episode_ended = raw.get("return_reason") == "episode_ended"
                            result = A.ToolResult(True, A.compact_json(raw), raw=raw,
                                                  with_image=back_obs is not None,
                                                  observation=back_obs)
                        except Exception as exc:
                            raw = dict(schema="backoff_tool_result_v1", tool=name,
                                       ok=False, return_reason="error", error=str(exc))
                            result = A.ToolResult(False, A.compact_json(raw), raw=raw,
                                                  error=str(exc))
                            infra_error, why = str(exc), "backoff_error"
                    else:
                        result = A.ToolResult(False, A.compact_json(raw), raw=raw)
                    record["calls"].append(dict(name=name, arguments=arguments,
                                                ok=result.ok, result=result.raw))
                    results.append(result)
                    if recovery_episode_ended:
                        why = "episode_ended"
                    if infra_error or recovery_episode_ended:
                        break
                    continue
                result = self.runtime.execute_tool(
                    option_id=f"turn-{turn + 1}-call-{calls_used}",
                    instruction=case["instruction"], arguments=arguments,
                    current_observation=obs,
                    slow_request_snapshot=AHV.semantic_request_snapshot(
                        messages=slow_messages, instruction=case["instruction"]),
                    slow_response_snapshot=AHV.response_snapshot(
                        text=text, visible=visible, usage=usage,
                        query_request_id=request_id),
                    action_head_as_tool_version=self.action_head_as_tool_version,
                )
                if result.ok or result.trace_frames:
                    # Earlier executed chunks still break a turn streak when
                    # a later prediction is rejected by the safety guard.
                    consecutive_turn_calls, _ = A.enforce_v43_consecutive_turn_limit(
                        self.action_head_as_tool_version, name, consecutive_turn_calls)
                if result.observation is not None:
                    obs = result.observation
                if trajectory_map is not None and (result.ok or result.trace_frames):
                    trajectory_map.update_action_head(result.trace_frames)
                record["calls"].append({
                    "name": name, "arguments": arguments,
                    "ok": result.ok, "result": result.raw,
                })
                if self.action_head_as_tool_version in ("v4.1", "v4.9", "v4.10"):
                    recovery_turn_available = (
                        ((result.raw or {}).get("direction_alignment") or {})
                        .get("match") is False)
                results.append(result)
                # A collection runtime may deliberately end an unsupported
                # branch. Keep this private control signal off model_text/raw,
                # retain the original invocation, then use normal final scoring.
                episode_abort_reason = str(getattr(result, "episode_abort_reason", "") or "")
                if episode_abort_reason:
                    episode_abort_detail = str(getattr(result, "episode_abort_detail", "") or "")
                    record["episode_abort_reason"] = episode_abort_reason
                    record["episode_abort_detail"] = episode_abort_detail
                    why = episode_abort_reason
                    break
                if not result.ok and (result.raw or {}).get("return_reason") == "error":
                    infra_error = str((result.raw or {}).get("error") or "action head error")
                    why = "action_head_error"
                    break
            turns.append(record)
            completed = getattr(self, 'completed_depth_observer', None)
            if completed is not None and not (episode_abort_reason or infra_error or agent_stopped or recovery_episode_ended):
                progress = completed(record)
                if progress.get('terminate'):
                    episode_abort_reason = why = progress['terminal_reason']
                    episode_abort_detail = 'consecutive_no_progress=' + str(progress['consecutive_no_progress'])
            if boundary_contract is not None:
                boundary_contract.observe_record(record, self.runtime.render_client.last_info or initial_info)
            if episode_abort_reason:
                benchmark_decision(record)
                break
            if not infra_error and benchmark_decision(record) and not agent_stopped:
                why = "official_goal_radius"
                break
            if agent_stopped or infra_error or recovery_episode_ended:
                break
            if calls_used >= self.args.max_tool_calls:
                why = "tool_budget_exhausted"
                break
            if not results:
                why = "no_action_head_result"
                break
            budget_left = self.args.max_tool_calls - calls_used
            for index, result in enumerate(results):
                if self.protocol is not None:
                    body = (self.protocol.append_tool_budget(result.model_text, budget_left)
                            if index == len(results) - 1 else result.model_text)
                    content = [{"type": "text", "text": "[Tool result]\n" + body}]
                    frames = tuple(getattr(result, "trace_frames", ()) or ())
                    for frame in frames:
                        content.append({"type": "text", "text": self.protocol.chunk_text(
                            option_id=frame["option_id"], chunk=frame["chunk"],
                            actual_end_pose=frame.get("actual_end_pose"),
                            path_m=frame["path_m"], blocked=frame["blocked"],
                            collision=frame["collision"],
                            no_progress=frame["no_progress"])})
                        frame_obs = frame.get("observation")
                        frame_image = obs_image_part(frame_obs) if frame_obs is not None else None
                        if (frame_image is not None and
                                (self.action_head_as_tool_version != 'v8' or frame['final_slow_observation'])):
                            content.append(self.protocol.tagged_image_part(
                                frame_image,
                                kind=("final_slow" if frame["final_slow_observation"]
                                      else "fast_intermediate"),
                                option_id=frame["option_id"], chunk=frame["chunk"]))
                    if not frames and result.with_image and index == len(results) - 1:
                        image = obs_image_part(obs)
                        if image is not None:
                            content.append(self.protocol.tagged_image_part(
                                image, kind="slow"))
                else:
                    body = A.append_tool_budget(result.model_text, budget_left) \
                        if index == len(results) - 1 else result.model_text
                    content = [{"type": "text", "text": "[工具返回]\n" + body}]
                    if result.with_image and index == len(results) - 1:
                        image = obs_image_part(obs)
                        if image is not None:
                            content.insert(0, image)
                msgs.append({"role": "user", "content": content})
        else:
            why = "turn_budget_exhausted"

        # Obtain evaluator-only terminal metrics without crediting a model that
        # never called stop.  This STOP is outside the model-visible trajectory.
        if agent_stopped:
            score_info = turns[-1]["calls"][-1].get("score_info") or {}
        else:
            try:
                _, _, _, _, score_info = self.bench.step_tool(
                    "stop", {"reason": "evaluator terminal scoring only"})
            except Exception as exc:
                score_info = {}
                infra_error = infra_error or f"terminal scoring failed: {exc}"
        legacy_point_goal_success_at_end = bool(score_info.get("success", False))
        official_detail = score_info.get("official_object_success_detail") or {}
        official_signal_present = "official_object_success" in score_info
        official_available = bool(official_detail.get("available", False))
        official_success_at_end = bool(score_info.get(
            "official_object_success", False))
        # SAGE training/evaluation uses the target-specific official expanded
        # object bbox.  Non-SAGE backends retain their native terminal metric.
        terminal_success = (official_success_at_end if official_signal_present
                            else legacy_point_goal_success_at_end)
        path_length_m = float(score_info.get("path_length_m", 0.0) or 0.0)
        shortest_path_m = float(score_info.get("shortest_path_m", 0.0) or 0.0)
        if boundary_contract is None and benchmark.enabled() and official_detail.get('schema') != 'sage-designated-target-navmesh-v2':
            path_length_m, shortest_path_m = benchmark.trajectory_metrics(
                case["episode_id"], benchmark_positions)
        legacy_spl = float(score_info.get("spl", 0.0) or 0.0)
        official_spl = success_weighted_spl(
            terminal_success, path_length_m, shortest_path_m)
        if benchmark.enabled() and shortest_path_m == 0:
            official_spl = float(terminal_success)
        if official_signal_present and not official_available:
            infra_error = infra_error or (
                "official object success unavailable: "
                + str(official_detail.get("reason") or "unknown"))
        result = {
            "schema": "qwen36_action_head_curriculum_rollout_v1",
            **case,
            "model": MODEL,
            "mode": self.args.mode,
            "tool_profile": "action_head",
            "action_head_as_tool_version": self.action_head_as_tool_version,
            **(self.protocol.protocol_identity()
               if self.protocol is not None else {}),
            "checkpoint": self.args.checkpoint,
            "seed": seed,
            "evaluation_valid": not bool(infra_error),
            "invalid_tool_call_policy": invalid_call_policy,
            "success": bool(terminal_success and not infra_error),
            "success_contract": (official_detail.get("schema")
                                 if official_signal_present else score_info.get("success_contract", "backend_native")),
            "native_goal_success_detail": score_info.get("native_goal_success_detail"),
            "agent_stopped": agent_stopped,
            "oracle_success_at_end": legacy_point_goal_success_at_end,
            "legacy_point_goal_success_at_end": legacy_point_goal_success_at_end,
            "official_object_success_at_end": official_success_at_end,
            "official_object_success_detail": official_detail,
            "spl": official_spl if official_signal_present else legacy_spl,
            "legacy_point_goal_spl": legacy_spl,
            "distance_to_goal": score_info.get("distance_to_goal"),
            "final_pose_world_xy_yaw_deg": terminal_world_pose(score_info),
            "path_length_m": path_length_m,
            "shortest_path_m": shortest_path_m,
            "collisions": int(score_info.get("collisions", 0) or 0),
            "steps_used": score_info.get("steps_used"),
            "calls_used": calls_used,
            "why": why,
            "episode_abort_reason": episode_abort_reason,
            "episode_abort_detail": episode_abort_detail,
            "actual_progress": turns[-1].get('actual_progress') if turns else None,
            "infra_error": infra_error,
            "initial_info": initial_info,
            "turns": turns,
        }
        context_observer = getattr(self, 'evaluation_context_observer', None)
        if context_observer is not None:
            result['canonical_context_archive'] = context_observer(msgs, case=case, seed=seed)

        return (boundary_contract.finalize(result, score_info)
                if boundary_contract is not None else result)


def run(args) -> int:
    cases = scene_shards(load_cases(), args.shard_count)[args.shard_index]
    if args.limit_cases:
        cases = cases[:args.limit_cases]
    evaluator = ActionHeadEvaluator(args)
    output = Path(args.output)
    for index, case in enumerate(cases, 1):
        path = result_path(output, case)
        if path.exists():
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                A.assert_tool_call_policy_matches(saved)
                if saved.get("evaluation_valid"):
                    print(f"[{index}/{len(cases)}] resume {case['episode_id']}", flush=True)
                    continue
            except (OSError, ValueError):
                pass
        seed = deterministic_seed(args.seed_base, case)
        last = None
        for attempt in range(1, args.retries + 1):
            try:
                last = evaluator.rollout(case, seed)
            except Exception as exc:
                last = {
                    "schema": "qwen36_action_head_curriculum_rollout_v1",
                    **case, "evaluation_valid": False, "success": False,
                    "seed": seed, "attempt": attempt, "infra_error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            last["attempt"] = attempt
            if last.get("evaluation_valid"):
                break
            time.sleep(min(5 * attempt, 15))
        atomic_json(path, last)
        print(f"[{index}/{len(cases)}] {case['stage']} {case['scene']}:"
              f"{case['episode_id']} valid={last.get('evaluation_valid')} "
              f"success={last.get('success')} dist={last.get('distance_to_goal')} "
              f"why={last.get('why')}", flush=True)
    return 0


def aggregate(args) -> None:
    output = Path(args.output)
    rows = []
    for case in load_cases():
        path = result_path(output, case)
        if path.exists():
            try:
                rows.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                pass
    summary = {"schema": "qwen36_action_head_curriculum_summary_v1",
               "case_count": 262, "completed": len(rows), "stages": {}}
    summary["invalid_tool_call_policy"] = A.invalid_tool_call_policy()
    summary["invalid_tool_call_policy_mismatches"] = sum(
        row.get("invalid_tool_call_policy", "terminate") != A.invalid_tool_call_policy()
        for row in rows)
    summary["evaluation_valid"] = bool(rows) and len(rows) == 262 and all(
        row.get("evaluation_valid") for row in rows) and not summary["invalid_tool_call_policy_mismatches"]
    for stage in ("p0", "p1", "p2"):
        selected = [x for x in rows if x.get("stage") == stage]
        valid = [x for x in selected if x.get("evaluation_valid")]
        success = sum(bool(x.get("success")) for x in valid)
        distances = [float(x["distance_to_goal"]) for x in valid
                     if x.get("distance_to_goal") is not None]
        summary["stages"][stage.upper()] = {
            "expected": EXPECTED[stage], "completed": len(selected),
            "valid": len(valid), "successes": success,
            "success_rate": success / len(valid) if valid else None,
            "mean_distance_to_goal": sum(distances) / len(distances) if distances else None,
            "agent_stops": sum(bool(x.get("agent_stopped")) for x in valid),
            "oracle_success_at_end": sum(bool(x.get("oracle_success_at_end")) for x in valid),
        }
    valid = [x for x in rows if x.get("evaluation_valid")]
    success = sum(bool(x.get("success")) for x in valid)
    summary["overall"] = {
        "expected": 262, "completed": len(rows), "valid": len(valid),
        "successes": success,
        "success_rate": success / len(valid) if valid else None,
        "agent_stops": sum(bool(x.get("agent_stopped")) for x in valid),
        "oracle_success_at_end": sum(bool(x.get("oracle_success_at_end")) for x in valid),
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    runp = sub.add_parser("run")
    runp.add_argument("--checkpoint", required=True)
    runp.add_argument("--backbone", required=True)
    runp.add_argument("--query-cache", required=True)
    runp.add_argument("--bench-url", required=True)
    runp.add_argument("--vllm-url", required=True)
    runp.add_argument("--output", required=True)
    runp.add_argument("--shard-index", type=int, required=True)
    runp.add_argument("--shard-count", type=int, required=True)
    runp.add_argument("--mode", choices=("think", "instruct"), default="think")
    runp.add_argument("--max-turns", type=int, default=30)
    runp.add_argument("--max-tool-calls", type=int, default=30)
    runp.add_argument("--keep-last-images", type=int, default=8)
    runp.add_argument("--seed-base", type=int, default=20260827)
    runp.add_argument("--policy-seed", type=int, default=20260827)
    runp.add_argument("--timeout", type=int, default=600)
    runp.add_argument("--retries", type=int, default=3)
    runp.add_argument("--limit-cases", type=int, default=0)
    runp.add_argument("--action-head-as-tool-version", choices=("v1", "v2", "v3", "v4", "v4.1", "v4.2", "v4.3", "v4.4", "v4.5", "v4.6", "v4.7", "v4.8", "v4.9", "v4.10", "v4.11", "v6", "v8"), default="v1")
    aggp = sub.add_parser("aggregate")
    aggp.add_argument("--output", required=True)
    return ap


def main() -> None:
    args = parser().parse_args()
    if args.command == "aggregate":
        aggregate(args)
    else:
        raise SystemExit(run(args))


if __name__ == "__main__":
    main()
