#!/usr/bin/env python3
"""Closed-loop evaluation on the immutable balanced 50-case subset."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO), str(REPO / "vln_demo" / "src"), str(REPO / "verl")]

import agent_tools as A  # noqa: E402
from action_head import vllm_bridge as AHV  # noqa: E402
from action_head import protocol as ACTION_HEAD_PROTOCOL  # noqa: E402
from verl_access.vlm_policy import to_openai_messages  # noqa: E402
from verl_access.context_budget import (  # noqa: E402
    budget_chat_request, ContextBudgetServerRejected, is_context_overflow)
from verl_access.serving_error import (  # noqa: E402
    ServingGenerationError, TRANSPORT_ERRORS, serving_error)
from vln_gym import fm_action as FMA  # noqa: E402
from sage.goal_region import official_success_contract_identity, success_contract_matches  # noqa: E402

from common import (atomic_json, checkpoint_identity, contract_direction,
                    delta_endpoint, direction_from_endpoint, mean, rate,
                    safe, sha256_file, wrap_degrees, load_subset)  # noqa: E402

BASE_PATH = REPO / "verl/scripts/eval/action_head_fm_p0_p2/curriculum_action_head_eval.py"
SPEC = importlib.util.spec_from_file_location("_base_curriculum_action_head_eval", BASE_PATH)
E = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(E)
ACTIVE_QUERY_COUNT = AHV.DEFAULT_QUERY_COUNT


def brain_query_count(query_count=None):
    # Rule runtimes generate labels without using query hidden rows, but the
    # sampling request must still match the four-query v6 data collector.
    if (os.environ.get('RULE_TEACHER_SAMPLING_PROFILE') == 'v6_depth'
            and os.environ.get('COMPANION_BRAIN_MODE', 'sft') == 'sft'):
        return AHV.DEFAULT_QUERY_COUNT
    return ACTIVE_QUERY_COUNT if query_count is None else query_count


def brain_sampling_contract(query_count=None):
    mode = os.environ.get("COMPANION_BRAIN_MODE", "sft")
    text_budget = int(os.environ.get("ACTION_HEAD_QUERY_TEXT_BUDGET", "0"))
    if text_budget < 0 or (text_budget and mode != "sft"):
        raise ValueError("query text budget must be nonnegative and is supported in sft mode only")
    if mode == "think":
        params, thinking = E.P.sampling_params("think")
        return {"enable_thinking": thinking, **params,
                "preserve_thinking": True, "omitted_parameters": ["seed"],
                "matches_sft_bundle_generation": False,
                "preset": "vln_demo/src/prompts.py:MODES.think"}
    if mode != "sft":
        raise ValueError(f"unsupported COMPANION_BRAIN_MODE: {mode}")
    contract = {"enable_thinking": False, "temperature": 0.7, "top_p": 0.95,
            "max_tokens": 256,
            "omitted_parameters": ["seed", "top_k", "min_p", "presence_penalty"],
            "matches_sft_bundle_generation": True}
    if os.environ.get('RULE_TEACHER_SAMPLING_PROFILE') == 'v6_depth':
        # Verified original v6 job exports TEMPERATURE=1.0. Its beam payload
        # reserves exactly DEFAULT_QUERY_COUNT tokens on top of 256.
        contract['temperature'] = 1.0
        contract['max_tokens'] = 256 + AHV.DEFAULT_QUERY_COUNT
        contract['query_count'] = AHV.DEFAULT_QUERY_COUNT
    if text_budget:
        count = brain_query_count(query_count)
        reserve = count + 1 if count else 0
        contract.update(max_tokens=text_budget + reserve,
                        text_max_tokens=text_budget, query_tail_reserved_tokens=reserve,
                        query_budget_protocol="text_plus_query_tail_v1",
                        matches_sft_bundle_generation=False)
    return contract


def training_matched_payload(messages, query_count=None):
    """Encode once, before budgeting and same-state candidate sampling."""
    count = brain_query_count(query_count)
    contract = brain_sampling_contract(count)
    template = {"enable_thinking": contract["enable_thinking"]}
    if (os.environ.get('RULE_TEACHER_SAMPLING_PROFILE') != 'v6_depth'
            or os.environ.get('COMPANION_BRAIN_MODE', 'sft') != 'sft'):
        template["preserve_thinking"] = True
    return {
        "model": E.MODEL,
        "messages": to_openai_messages(messages),
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": template,
        **{key: contract[key] for key in (
            "temperature", "top_p", "max_tokens", "top_k", "min_p",
            "presence_penalty", "repetition_penalty") if key in contract},
    }


def prepare_training_matched_query(
        vllm_url, messages, mode, seed, timeout, query_count=None,
        max_completion_tokens=0):
    """Collector preflight; preserve the existing sampling/output contract."""
    del mode, seed, max_completion_tokens
    started = time.monotonic()
    payload = training_matched_payload(messages, query_count)
    encoded = time.monotonic()
    prepared, audit = budget_chat_request(
        vllm_url, payload, timeout, tokenize_image_cache="budget",
        allow_history_trim=os.environ.get('RULE_TEACHER_CONTEXT_VERSION') != 'v8')
    audit["preparation_client_timing"] = dict(schema="chat_preparation_client_timing_v1",
        message_encode_seconds=encoded - started,
        budget_seconds=time.monotonic() - encoded,
        elapsed_seconds=time.monotonic() - started)
    return prepared, audit


def training_matched_chat_query(
        vllm_url, messages, mode, seed, timeout, query_count=None,
        max_completion_tokens=0, record_decision=False, text_temperature=1.0,
        legal_action_policy=None, continuous_prefix_ids=None, continuous_parent_request=None,
        continuous_cache_salt=None,
        *, prepared_payload=None, timings=None):
    """Default to SFT sampling; optionally reuse the exact budgeted request."""
    # The shared evaluator forwards compatibility keywords; neither changes
    # the frozen SFT generation contract. Query IDs remain candidate-specific.
    started = time.monotonic()
    if timings is not None:
        timings.update(schema="chat_client_timing_v1",
            scope="client wall time; HTTP and first content include queue, network and model work",
            completed=False)
    del mode, seed, max_completion_tokens
    count = brain_query_count(query_count)
    payload = (training_matched_payload(messages, count) if prepared_payload is None
               else copy.deepcopy(prepared_payload))
    request_id = AHV.request_queries(payload, count) if count else ""
    continuous = os.environ.get("RULE_TEACHER_CONTEXT_VERSION") == "v8"
    if continuous:
        if not count:
            raise ValueError("v8 collection requires native query capture")
        payload["vllm_xargs"]["action_head_record_decision"] = True
        if continuous_cache_salt is not None:
            from action_head.serve.continuous_context import CACHE_KEY
            payload['vllm_xargs'][CACHE_KEY] = continuous_cache_salt
            payload['cache_salt'] = continuous_cache_salt
        if continuous_prefix_ids is not None:
            from action_head.serve.continuous_context import PREFIX_KEY, PARENT_KEY
            payload["vllm_xargs"][PREFIX_KEY] = continuous_prefix_ids
            payload["vllm_xargs"][PARENT_KEY] = continuous_parent_request
    if record_decision:
        if not count or not math.isfinite(text_temperature) or text_temperature <= 0:
            raise ValueError("text recording requires marked queries and positive temperature")
        payload["vllm_xargs"]["action_head_record_decision"] = True
        payload.update(temperature=float(text_temperature), top_p=1.0, top_k=-1,
                       min_p=0.0, repetition_penalty=1.0,
                       presence_penalty=0.0, frequency_penalty=0.0)
    if legal_action_policy is not None:
        from action_head_rl.executable_policy import encode_request
        payload.setdefault("vllm_xargs", {})["action_head_executable_policy"] = encode_request(legal_action_policy)
        payload["chat_template_kwargs"]["enable_thinking"] = False
    encoded = time.monotonic()
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        vllm_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    http_started = time.monotonic()
    if timings is not None:
        timings.update(payload_prepare_seconds=encoded - started,
            serialize_seconds=http_started - encoded, request_bytes=len(body),
            http_socket_timeout_seconds=timeout)
    chunks, reasoning_chunks, usage = [], [], {}
    finish_reason = None
    stream_error = None
    stream_done = False
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read(4096).decode("utf-8", "replace")
        except TRANSPORT_ERRORS as read_error:
            raise serving_error(payload, 'generation_http_body', read_error) from read_error
        finally:
            exc.close()
        if is_context_overflow(exc.code, detail):
            raise ContextBudgetServerRejected(f"vLLM HTTP {exc.code}: {detail[:500]}") from None
        raise serving_error(payload, 'generation_open', exc) from exc
    except TRANSPORT_ERRORS as exc:
        raise serving_error(payload, 'generation_open', exc) from exc
    finally:
        if timings is not None:
            timings.update(http_open_seconds=time.monotonic() - http_started,
                elapsed_seconds=time.monotonic() - started)
    stream_started = time.monotonic()
    first_event = first_content = None
    try:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                stream_done = True
                break
            if first_event is None:
                first_event = time.monotonic() - http_started
            try:
                obj = json.loads(body)
            except json.JSONDecodeError as exc:
                raise serving_error(payload, 'generation_stream_json', exc,
                    failure_kind='protocol', finish_reason=finish_reason, usage=usage,
                    emitted_text_chars=sum(map(len, chunks)) + sum(map(len, reasoning_chunks))) from exc
            if obj.get("error"):
                stream_error = str(obj["error"])[:1000]
            if obj.get("usage"):
                usage.update(obj["usage"])
            choice = (obj.get("choices") or [{}])[0]
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if reasoning:
                reasoning_chunks.append(reasoning)
            piece = delta.get("content") or ""
            if piece:
                chunks.append(piece)
            if first_content is None and (reasoning or piece):
                first_content = time.monotonic() - http_started
    except TRANSPORT_ERRORS as exc:
        raise serving_error(payload, 'generation_stream', exc,
            finish_reason=finish_reason, usage=usage,
            emitted_text_chars=sum(map(len, chunks)) + sum(map(len, reasoning_chunks)),
            stream_error=stream_error) from exc
    finally:
        response.close()
        query_cache = os.environ.get('RULE_TEACHER_QUERY_CACHE')
        if query_cache and request_id and not continuous and os.environ.get('RULE_TEACHER_SAMPLING_PROFILE') == 'v6_depth':
            # The v6 worker publishes the query rows before sampling EOS.
            # Rule labels use actual execution; release this request's unused
            # hidden rows after HTTP completion, keeping the serving protocol.
            (Path(query_cache)/(request_id+'.safetensors')).unlink(missing_ok=True)
        if timings is not None:
            timings.update(stream_seconds=time.monotonic() - stream_started,
                http_to_first_event_seconds=first_event,
                http_to_first_content_seconds=first_content,
                elapsed_seconds=time.monotonic() - started)
    emitted_chars = sum(map(len, chunks)) + sum(map(len, reasoning_chunks))
    if stream_error or finish_reason == "error" or (
            type(usage.get("completion_tokens")) is int
            and usage["completion_tokens"] == 0 and emitted_chars == 0):
        raise ServingGenerationError(dict(
            finish_reason=finish_reason, usage=usage, emitted_text_chars=emitted_chars,
            stream_error=stream_error,
            request_sha256=hashlib.sha256(json.dumps(payload, sort_keys=True,
                ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()))
    if not stream_done:
        raise serving_error(payload, 'generation_incomplete_stream',
            failure_kind='incomplete_stream', finish_reason=finish_reason,
            usage=usage, emitted_text_chars=emitted_chars)
    if finish_reason is not None:
        usage["finish_reason"] = finish_reason
    text, query_tail_count = AHV.strip_query_tail("".join(chunks), query_count=count)
    if reasoning_chunks:
        # Keep reasoning out of tool parsing while retaining it in history.
        _, visible = A.split_thinking(text)
        text = "<think>" + "".join(reasoning_chunks) + "</think>\n" + visible
    if timings is not None:
        timings.update(completed=True, query_tail_count=query_tail_count,
                       elapsed_seconds=time.monotonic() - started)
    # ActionHeadEvaluator also needs the exact JPEG-backed OpenAI messages
    # sent to vLLM to build the slow-request snapshot consumed by the tool.
    return text, request_id, usage, payload["messages"]


E.chat_query = training_matched_chat_query


def actual_endpoint(before: dict, after: dict) -> dict | None:
    before_xy, after_xy = before.get("world_xy"), after.get("world_xy")
    before_yaw, after_yaw = before.get("world_yaw_deg"), after.get("world_yaw_deg")
    if before_xy is None or after_xy is None or before_yaw is None or after_yaw is None:
        return None
    x, y, yaw = FMA.world_to_chunk_pose(
        after_xy, math.radians(float(after_yaw)), before_xy,
        math.radians(float(before_yaw)))
    return {
        "x_m": float(x), "y_m": float(y),
        "yaw_deg": float(math.degrees(yaw)),
        "path_m": math.hypot(float(x), float(y)),
    }


def traced_prefix_result(value):
    """Drop env reward while accepting both render-client API generations."""
    if len(value) == 5:
        observation, _reward, terminated, truncated, info = value
    elif len(value) == 4:
        observation, terminated, truncated, info = value
    else:
        raise RuntimeError(
            f"execute_prefix returned {len(value)} values; expected 4 or 5")
    return observation, terminated, truncated, info


class TracedEvaluator(E.ActionHeadEvaluator):
    """Add evaluator-only predicted/requested/actual trajectory traces."""

    def __init__(self, args):
        super().__init__(args)
        self.evaluation_context = None
        if os.environ.get('ACTION_HEAD_EVAL_BRAIN_CONFIG') and (
                getattr(args, 'action_head_as_tool_version', 'v6') == 'v8'
                or os.environ.get('ACTION_HEAD_EVAL_CONTEXT_COMPACTION') == 'complete_turn_stride'):
            from action_head.evaluation_context import install
            self.evaluation_context = install(self)
        self.option_traces = []
        self.prefix_traces = []
        self.prediction_traces = []
        self.query_traces = []
        original_query_provider = self.runtime.tool_executor.slow_condition_provider

        def trace_query_provider(**kwargs):
            value = original_query_provider(**kwargs)
            if hasattr(value, 'tokens'):
                trace = {'condition_mode': 'v5_native', 'request_id_with_digest': value.request_id,
                         'shape': list(value.tokens.shape[1:]), **copy.deepcopy(value.metadata)}
            else:
                trace = {
                    "request_id_with_digest": value.request_id,
                    "source_path": value.source_path,
                    "hidden_convention": value.hidden_convention,
                    "query_count": value.query_count,
                    "shape": list(value.query_hidden.shape),
                }
                archive = str(getattr(args, "query_archive", "") or "").strip()
                if archive:
                    archive_dir = Path(archive)
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    safe_request = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.request_id)
                    target = archive_dir / f"{safe_request}.npy"
                    np.save(target, np.asarray(value.query_hidden, dtype=np.float32))
                    trace["archive_path"] = str(target.resolve())
            self.query_traces.append(trace)
            return value

        self.runtime.tool_executor.slow_condition_provider = trace_query_provider
        option_runtime = self.runtime.tool_executor.runtime
        original_predict = option_runtime.policy.predict

        def trace_predict(option):
            value = original_predict(option)
            self.prediction_traces.append(copy.deepcopy(value.tolist()))
            return value

        option_runtime.policy.predict = trace_predict
        original_prefix = self.runtime.render_client.execute_prefix

        def trace_prefix(waypoints, **execution_kwargs):
            before_info = copy.deepcopy(self.runtime.render_client.last_info)
            prefix_result = original_prefix(waypoints, **execution_kwargs)
            obs, terminated, truncated, info = traced_prefix_result(
                prefix_result)
            after_info = copy.deepcopy(info)
            requested = delta_endpoint(waypoints)
            actual = actual_endpoint(before_info, after_info)
            self.prefix_traces.append({
                "requested_prefix": copy.deepcopy(waypoints),
                "requested_endpoint": requested,
                "actual_endpoint": actual or requested,
                "actual_endpoint_source": "world_pose" if actual else "requested_prefix_fallback",
                "before_world_pose": {
                    "world_xy": before_info.get("world_xy"),
                    "world_yaw_deg": before_info.get("world_yaw_deg"),
                },
                "after_world_pose": {
                    "world_xy": after_info.get("world_xy"),
                    "world_yaw_deg": after_info.get("world_yaw_deg"),
                },
                "terminated": terminated,
                "truncated": truncated,
                "waypoint_log": copy.deepcopy(info.get("waypoint_log") or []),
                "blocked": bool(info.get("blocked_this_step")),
                "collided": bool(info.get("collided_this_step")),
            })
            # Tracing is observational; preserve the caller-facing contract.
            return prefix_result

        self.runtime.render_client.execute_prefix = trace_prefix
        original_tool = self.runtime.execute_tool

        def trace_tool(**kwargs):
            prefix_start = len(self.prefix_traces)
            pred_start = len(self.prediction_traces)
            query_start = len(self.query_traces)
            before_info = copy.deepcopy(self.runtime.render_client.last_info)
            result = original_tool(**kwargs)
            after_info = copy.deepcopy(self.runtime.render_client.last_info)
            prefixes = copy.deepcopy(self.prefix_traces[prefix_start:])
            predictions = copy.deepcopy(self.prediction_traces[pred_start:])
            for prefix, prediction in zip(prefixes, predictions):
                prefix["predicted_trajectory"] = prediction
                prefix["predicted_endpoint"] = delta_endpoint(prediction)
                prefix["predicted_direction"] = direction_from_endpoint(
                    prefix["predicted_endpoint"]["x_m"],
                    prefix["predicted_endpoint"]["y_m"],
                    prefix["predicted_endpoint"]["yaw_deg"])
                endpoint = prefix["actual_endpoint"]
                prefix["actual_direction"] = direction_from_endpoint(
                    endpoint["x_m"], endpoint["y_m"], endpoint["yaw_deg"])
            option_endpoint = actual_endpoint(before_info, after_info)
            if option_endpoint is None:
                command = {"x_m": 0.0, "y_m": 0.0, "yaw_deg": 0.0, "path_m": 0.0}
                yaw = 0.0
                for prefix in prefixes:
                    endpoint = prefix["requested_endpoint"]
                    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
                    command["x_m"] += c * endpoint["x_m"] - s * endpoint["y_m"]
                    command["y_m"] += s * endpoint["x_m"] + c * endpoint["y_m"]
                    command["path_m"] += endpoint["path_m"]
                    yaw += endpoint["yaw_deg"]
                command["yaw_deg"] = wrap_degrees(yaw)
                option_endpoint = command
                endpoint_source = "requested_prefix_fallback"
            else:
                endpoint_source = "world_pose"
            self.option_traces.append({
                "option_id": kwargs["option_id"],
                "before_world_pose": {key: copy.deepcopy(before_info.get(key))
                    for key in ("world_xy", "world_yaw_deg")},
                "after_world_pose": {key: copy.deepcopy(after_info.get(key))
                    for key in ("world_xy", "world_yaw_deg")},
                "arguments": copy.deepcopy(kwargs["arguments"]),
                "contract_direction": contract_direction(kwargs["arguments"]),
                "option_actual_endpoint": option_endpoint,
                "option_endpoint_source": endpoint_source,
                "option_actual_direction": direction_from_endpoint(
                    option_endpoint["x_m"], option_endpoint["y_m"],
                    option_endpoint["yaw_deg"]),
                ("condition_artifact" if getattr(args, "context_encoder_model", None) else "query_artifact"): copy.deepcopy(
                    self.query_traces[query_start] if len(self.query_traces) > query_start else None),
                "prefixes": prefixes,
            })
            return result

        self.runtime.execute_tool = trace_tool

    def rollout(self, case, seed):
        self.option_traces.clear()
        self.prefix_traces.clear()
        self.prediction_traces.clear()
        self.query_traces.clear()
        context = self.evaluation_context
        if context is not None:
            context.begin_episode()
        try:
            row = super().rollout(case, seed)
            if context is not None:
                row['continuous_context_audit'] = copy.deepcopy(context.audit)
                row['context_compactions'] = copy.deepcopy(context.compactions)
        finally:
            if context is not None and context.provider is not None:
                context.provider.end_episode()
        traces = iter(self.option_traces)
        for turn in row.get("turns", []):
            for call in turn.get("calls", []):
                if call.get("name") != "run_action_head":
                    continue
                trace = next(traces, None)
                if trace is not None:
                    call["action_trace"] = trace
        row["action_head_runtime_config"] = asdict(self.runtime.tool_executor.runtime.config)
        row["thinking_contract"] = brain_sampling_contract()
        return row


def result_path(output: Path, case: dict, repeat_index: int) -> Path:
    return (output / "rollouts" / f"repeat_{repeat_index:02d}" /
            case["stage"] / safe(case["scene"]) /
            (safe(case["episode_id"]) + ".json"))


def legacy_result_path(output: Path, case: dict) -> Path:
    """Read pre-repeat K=1 artifacts without using them for K>1."""
    return (output / "rollouts" / case["stage"] / safe(case["scene"]) /
            (safe(case["episode_id"]) + ".json"))


def deterministic_env_seed(case: dict) -> int:
    key = f"{case['scene']}:{case['episode_id']}".encode()
    return 20260828 + int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def all_calls(row: dict) -> list[dict]:
    return [call for turn in row.get("turns", []) for call in turn.get("calls", [])
            if call.get("name") == "run_action_head"]


def turn_usage_block(rows: list[dict]) -> dict:
    valid = [row for row in rows if row.get("evaluation_valid")]
    turn_calls = 0
    successful_turns = 0
    rollouts_with_turn = 0
    mismatch_followups = {"turn": 0, "run_action_head": 0, "stop": 0,
                          "none": 0, "other": 0}
    with_turn_success, without_turn_success = [], []
    for row in valid:
        calls = [call for turn in row.get("turns", [])
                 for call in turn.get("calls", [])]
        turns = [call for call in calls if call.get("name") == "turn"]
        turn_calls += len(turns)
        successful_turns += sum(bool((call.get("result") or {}).get("ok"))
                                for call in turns)
        if turns:
            rollouts_with_turn += 1
            with_turn_success.append(bool(row.get("success")))
        else:
            without_turn_success.append(bool(row.get("success")))
        for index, call in enumerate(calls):
            if call.get("name") != "run_action_head":
                continue
            alignment = ((call.get("result") or {}).get(
                "direction_alignment") or {})
            if alignment.get("match") is not False:
                continue
            next_name = (calls[index + 1].get("name")
                         if index + 1 < len(calls) else "none")
            key = next_name if next_name in mismatch_followups else "other"
            mismatch_followups[key] += 1
    return {
        "valid_rollouts": len(valid),
        "turn_calls": turn_calls,
        "successful_turn_calls": successful_turns,
        "rollouts_with_turn": rollouts_with_turn,
        "rollout_turn_rate": (rollouts_with_turn / len(valid) if valid else None),
        "success_rate_with_turn": rate(with_turn_success),
        "success_rate_without_turn": rate(without_turn_success),
        "post_direction_mismatch_next_tool": mismatch_followups,
    }


def metric_block(rows: list[dict]) -> dict:
    round_flags, option_flags = [], []
    determinate_options = 0
    total_options = 0
    valid = [row for row in rows if row.get("evaluation_valid")]
    for row in valid:
        for call in all_calls(row):
            total_options += 1
            trace = call.get("action_trace") or {}
            expected = trace.get("contract_direction", "unspecified")
            if expected not in ("straight", "left", "right"):
                continue
            determinate_options += 1
            option_direction = trace.get("option_actual_direction")
            if option_direction != "degenerate":
                option_flags.append(option_direction == expected)
            for prefix in trace.get("prefixes", []):
                actual_direction = prefix.get("actual_direction")
                if actual_direction != "degenerate":
                    round_flags.append(actual_direction == expected)
    def numeric(key: str) -> list[float]:
        values = []
        for row in valid:
            try:
                value = float(row.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        return values

    all_model_calls = [
        call for row in valid for turn in row.get("turns", [])
        for call in turn.get("calls", [])]
    tool_usage = {
        name: sum(call.get("name") == name for call in all_model_calls)
        for name in ("run_action_head", "turn", "move_backward", "stop")
    }
    termination_reasons: dict[str, int] = {}
    for row in valid:
        reason = str(row.get("why") or "unknown")
        termination_reasons[reason] = termination_reasons.get(reason, 0) + 1
    collisions = numeric("collisions")
    return {
        "case_count": len({
            (str(row.get("scene")), str(row.get("episode_id"))) for row in rows}),
        "rollout_count": len(rows),
        "valid_rollouts": len(valid),
        "round_trajectory_consistency": rate(round_flags),
        "stitched_option_trajectory_consistency": rate(option_flags),
        "success_rate": rate(row.get("success", False) for row in valid),
        "spl": mean(float(row.get("spl", 0.0) or 0.0) for row in valid),
        "legacy_point_goal_success_rate": rate(
            row.get("legacy_point_goal_success_at_end", False) for row in valid),
        "legacy_point_goal_spl": mean(
            float(row.get("legacy_point_goal_spl", 0.0) or 0.0) for row in valid),
        "agent_stop_rate": rate(row.get("agent_stopped", False) for row in valid),
        "mean_final_point_goal_distance_m": mean(numeric("distance_to_goal")),
        "mean_path_length_m": mean(numeric("path_length_m")),
        "mean_shortest_path_m": mean(numeric("shortest_path_m")),
        "mean_steps_used": mean(numeric("steps_used")),
        "mean_tool_calls": mean(numeric("calls_used")),
        "mean_slow_turns": mean(
            len(row.get("turns", [])) for row in valid),
        "mean_collisions": mean(collisions),
        "collision_rollout_rate": rate(value > 0 for value in collisions),
        "tool_usage": tool_usage,
        "termination_reasons": termination_reasons,
        "invalid_tool_call_attempts": sum(
            bool(turn.get("tool_call_error"))
            for row in valid for turn in row.get("turns", [])) + sum(
                row.get("why") == "no_valid_tool_call" for row in valid),
        "invalid_tool_call_rollout_rate": rate(
            any(turn.get("tool_call_error") for turn in row.get("turns", []))
            or row.get("why") == "no_valid_tool_call" for row in valid),
        "determinate_options": determinate_options,
        "total_options": total_options,
    }


def direction_alignment_block(rows: list[dict]) -> dict:
    """Measure v3 mismatch visibility and the immediately following option."""
    comparable, matched = 0, 0
    followups, silent_followups, rematched = 0, [], []
    forbidden = re.compile(
        r"\b(correction|recovery|deviation|failure|previous execution)\b",
        re.IGNORECASE)
    for row in rows:
        if not row.get("evaluation_valid"):
            continue
        calls = all_calls(row)
        for index, call in enumerate(calls):
            alignment = ((call.get("result") or {}).get(
                "direction_alignment") or {})
            if alignment.get("match") is None:
                continue
            comparable += 1
            matched += int(bool(alignment["match"]))
            if alignment["match"] or index + 1 >= len(calls):
                continue
            followups += 1
            next_call = calls[index + 1]
            arguments = json.dumps(
                next_call.get("arguments") or {}, ensure_ascii=False)
            silent_followups.append(not bool(forbidden.search(arguments)))
            next_alignment = ((next_call.get("result") or {}).get(
                "direction_alignment") or {})
            if next_alignment.get("match") is not None:
                rematched.append(bool(next_alignment["match"]))
    return {
        "comparable_options": comparable,
        "initial_match_rate": {
            "value": matched / comparable if comparable else None,
            "numerator": matched, "denominator": comparable,
        },
        "mismatch_with_followup": followups,
        "silent_followup_rate": rate(silent_followups),
        "next_option_rematch_rate": rate(rematched),
    }


def pass_at_k_block(rows: list[dict], cases: list[dict], repeats: int) -> dict:
    """A case passes iff at least one of its K valid rollout samples succeeds."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (str(row.get("scene")), str(row.get("episode_id")))
        grouped.setdefault(key, []).append(row)
    flags = []
    incomplete = 0
    for case in cases:
        key = (str(case["scene"]), str(case["episode_id"]))
        valid = [row for row in grouped.get(key, []) if row.get("evaluation_valid")]
        if len(valid) != repeats:
            incomplete += 1
            continue
        flags.append(any(bool(row.get("success")) for row in valid))
    result = rate(flags)
    return {
        **result,
        "k": repeats,
        "expected_cases": len(cases),
        "complete_cases": len(flags),
        "incomplete_cases": incomplete,
    }


def aggregate(output: Path, subset_path: Path, identity: dict,
              allow_partial: bool = False,
              action_head_as_tool_version: str = "v1",
              repeats: int = 4, report_pass_at_k: bool = True,
              validated_rows: list[dict] | None = None) -> dict:
    subset, cases = load_subset(subset_path)
    rows = []
    if validated_rows is not None:
        # The suite already checked shard, manifest and checkpoint identities.
        # Score those same rows without copying/re-reading their full traces.
        indexed = {(r['logical_id'],r['repeat_index']):r for r in validated_rows}
        expected = {(c['logical_id'],i) for c in cases for i in range(1,repeats+1)}
        if len(indexed) != len(validated_rows) or set(indexed) != expected:
            raise ValueError('validated scoring rows have missing, duplicate or unknown attempts')
        rows = [indexed[(c['logical_id'],i)] for c in cases for i in range(1,repeats+1)]
    else:
        for case in cases:
            for repeat_index in range(1, repeats + 1):
                path = result_path(output, case, repeat_index)
                if repeats == 1 and not path.exists():
                    path = legacy_result_path(output, case)
                if path.exists():
                    try:
                        rows.append(json.loads(path.read_text(encoding="utf-8")))
                    except Exception:
                        pass
    stage_names = list((subset.get("quotas") or {}).keys())
    stages = {stage.upper(): metric_block(
        [row for row in rows if row.get("stage") == stage])
        for stage in stage_names}
    macro = {}
    for key in ("round_trajectory_consistency", "stitched_option_trajectory_consistency",
                "success_rate"):
        macro[key] = mean(block[key]["value"] for block in stages.values()
                          if block[key]["value"] is not None)
    macro["spl"] = mean(block["spl"] for block in stages.values()
                        if block["spl"] is not None)
    stage_pass_at_k = {
        stage.upper(): pass_at_k_block(
            [row for row in rows if row.get("stage") == stage],
            [case for case in cases if case.get("stage") == stage], repeats)
        for stage in stage_names
    }
    pass_macro = mean(
        block["value"] for block in stage_pass_at_k.values()
        if block["value"] is not None)
    overall = metric_block(rows)
    overall_pass_at_k = pass_at_k_block(rows, cases, repeats)
    pass_at_1 = {
        "k": 1,
        "definition": (
            "empirical per-rollout success over all valid repeated samples; "
            "the pass@1 estimator"),
        "stages": {
            stage.upper(): stages[stage.upper()]["success_rate"]
            for stage in stage_names
        },
        "overall_micro": overall["success_rate"],
    }
    expected_runs = len(cases) * repeats
    valid_runs = sum(bool(row.get("evaluation_valid")) for row in rows)
    success_contract = official_success_contract_identity()
    success_contract_mismatches = sum(
        bool(row.get("evaluation_valid")) and (
            row.get("success_contract") != success_contract["object_success_schema"]
            or (row.get("official_object_success_detail") or {}).get(
                "contract_sha256") != success_contract["contract_sha256"]
            or float((row.get("official_object_success_detail") or {}).get(
                "expansion_radius_m", float("nan"))) !=
            success_contract["expansion_radius_m"])
        for row in rows)
    protocol = ACTION_HEAD_PROTOCOL.protocol_module(action_head_as_tool_version)
    invalid_call_policy = A.invalid_tool_call_policy()
    invalid_call_policy_mismatches = sum(
        row.get("invalid_tool_call_policy", "terminate") != invalid_call_policy
        for row in rows)
    summary = {
        "schema": "qwen36_4query_closedloop_summary_v2",
        "action_head_as_tool_version": action_head_as_tool_version,
        **(protocol.protocol_identity() if protocol is not None else {}),
        "checkpoint": identity,
        "subset_manifest": str(subset_path.resolve()),
        "subset_manifest_sha256": sha256_file(subset_path),
        "expected_cases": len(cases),
        "repeats_per_case": repeats,
        "expected_runs": expected_runs,
        "completed_runs": len(rows),
        "valid_runs": valid_runs,
        "invalid_tool_call_policy": invalid_call_policy,
        "invalid_tool_call_policy_mismatches": invalid_call_policy_mismatches,
        "completed_cases": overall_pass_at_k["complete_cases"],
        "evaluation_valid": bool(rows) and all(
            row.get("evaluation_valid") for row in rows) and (
                allow_partial or len(rows) == expected_runs) and not (
                    success_contract_mismatches or invalid_call_policy_mismatches),
        "allow_partial": bool(allow_partial),
        "success_contract": success_contract,
        "success_contract_mismatches": success_contract_mismatches,
        "brain_sampling": {**next((row["thinking_contract"] for row in rows
                                   if row.get("thinking_contract")),
                                  brain_sampling_contract()), "seed": "omitted"},
        "stages": stages,
        "macro_average": macro,
        "overall_micro": overall,
        "mean_at_k": {
            "k": repeats,
            "definition": "metrics averaged over all valid repeated rollouts",
            "stages": stages,
            "macro_average": macro,
            "overall_micro": overall,
        },
        "pass_at_1": pass_at_1,
        "pass_at_k": {
            "k": repeats,
            "definition": "a case passes when any of its K valid rollouts succeeds",
            "stages": stage_pass_at_k,
            "macro_average": pass_macro,
            "overall_micro": overall_pass_at_k,
        },
        "primary_metrics": [
            "success_rate",
            "pass_at_1",
            "pass_at_k",
            "spl",
            "mean_final_point_goal_distance_m",
            "mean_path_length_m",
            "mean_tool_calls",
            "agent_stop_rate",
            "mean_collisions",
            "collision_rollout_rate",
            "round_trajectory_consistency",
            "stitched_option_trajectory_consistency",
        ],
        "subset_protocol": subset,
    }
    if action_head_as_tool_version in ("v3", "v4", "v4.1", "v4.2", "v4.3", "v4.4", "v4.5", "v4.6", "v4.7", "v4.8", "v4.9", "v4.10", "v4.11", "v6", "v8"):
        summary["direction_alignment"] = direction_alignment_block(rows)
    if action_head_as_tool_version in ("v4", "v4.1", "v4.2", "v4.3", "v4.4", "v4.5", "v4.6", "v4.7", "v4.8", "v4.9", "v4.10", "v4.11", "v6", "v8"):
        summary["turn_usage"] = turn_usage_block(rows)
    if not report_pass_at_k:
        if repeats != 1:
            raise ValueError("pass@k can only be omitted for one rollout per case")
        for key in ("pass_at_1", "pass_at_k"):
            summary.pop(key, None)
        summary["primary_metrics"] = [metric for metric in summary["primary_metrics"]
                                      if metric not in ("pass_at_1", "pass_at_k")]
    atomic_json(output / "summary.json", summary)
    return summary


def evaluate_repeat(evaluator, case, repeat_index, run_index, total_runs,
                    args, output, identity):
    started = time.monotonic()
    path = result_path(output, case, repeat_index)
    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            A.assert_tool_call_policy_matches(saved)
            if saved.get("evaluation_valid"):
                if str(case.get('scene', '')).startswith('sage-') and not success_contract_matches(saved):
                    raise RuntimeError('saved evaluation uses different scoring or changed geometry; use a fresh output directory')
                print(f"[closedloop] resume {run_index}/{total_runs} "
                      f"{case['episode_id']} repeat={repeat_index}", flush=True)
                return False
        except (OSError, ValueError):
            pass
    last = None
    for attempt in range(1, args.retries + 1):
        try:
            last = evaluator.rollout(case, deterministic_env_seed(case))
        except Exception as exc:
            last = {
                "schema": "qwen36_4query_closedloop_rollout_v1", **case,
                "evaluation_valid": False, "success": False,
                "infra_error": repr(exc), "traceback": traceback.format_exc(),
            }
        last["attempt"] = attempt
        last["checkpoint"] = identity
        last["repeat_index"] = repeat_index
        last["repeats_per_case"] = args.repeats
        if last.get("evaluation_valid"):
            break
        time.sleep(min(5 * attempt, 15))
    last["evaluation_elapsed_s"] = time.monotonic() - started
    atomic_json(path, last)
    print(f"[closedloop] {run_index}/{total_runs} {case['stage']} "
          f"{case['episode_id']} repeat={repeat_index}/{args.repeats} "
          f"valid={last.get('evaluation_valid')} success={last.get('success')} "
          f"why={last.get('why')}", flush=True)
    return True


def parallel_rollouts(args, namespace, cases, output, identity):
    from action_head.policy import DynamicBatchingPolicy, PolicyProxy
    sys.path.insert(0, str(REPO / "verl/scripts/train/action_head_bootstrap"))
    from case_scheduler import SceneCaseScheduler

    urls = [url.strip().rstrip('/') for url in args.bench_urls.split(',') if url.strip()]
    if not urls or len(urls) != len(set(urls)):
        raise ValueError("parallel actors require distinct Gym URLs")
    paths = [result_path(output, case, repeat) for case in cases
             for repeat in range(1, args.repeats + 1)]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate case identity or colliding rollout path")
    # Construct the checkpoint-selected visual encoder/transform once through
    # the same evaluator factory used by serial evaluation.
    native_provider = None
    if getattr(namespace, "context_encoder_model", None):
        from native_companion import SharedNativeProvider
        native_provider = SharedNativeProvider(namespace.context_encoder_model,
                                               namespace.context_aggregator_checkpoint)
        namespace.shared_condition_provider = native_provider
    base = E.ActionHeadEvaluator(namespace)
    policy = DynamicBatchingPolicy(base.runtime.tool_executor.runtime.policy,
                                  max_batch_size=args.max_batch_size,
                                  batch_wait_ms=args.batch_wait_ms)
    tasks = [{**case, '_repeat_index': repeat, '_run_index': index * args.repeats + repeat}
             for index, case in enumerate(cases)
             for repeat in range(1, args.repeats + 1)]
    # Several FM workers can share the same scene-routed renderers; split the
    # per-renderer admission budget so their sum keeps the original 48/64.
    shards = max(1, int(os.environ.get('SUITE_FM_WORKERS', '1')))
    scheduler = SceneCaseScheduler(tasks,
        replicas=int(os.environ.get('SUITE_RENDERER_REPLICAS', '3')),
        max_scenes=max(1, int(os.environ.get('SUITE_MAX_ACTIVE_SCENES', '48')) // shards),
        max_cases=max(1, int(os.environ.get('SUITE_MAX_ACTIVE_CASES', '64')) // shards))
    started = time.monotonic()
    lock = threading.Lock()
    counters = {'executed': 0, 'resumed': 0}

    def actor(url):
        try:
            local = SimpleNamespace(**vars(namespace))
            local.bench_url = url
            if getattr(namespace, 'vllm_urls', None):
                local.vllm_url = namespace.vllm_urls[urls.index(url) % len(namespace.vllm_urls)]
            local.shared_action_head_policy = PolicyProxy(policy)
            evaluator = TracedEvaluator(local)
            while (task := scheduler.acquire()) is not None:
                try:
                    case = {key: value for key, value in task.items()
                            if key not in ('_repeat_index', '_run_index')}
                    executed = evaluate_repeat(
                        evaluator, case, task['_repeat_index'], task['_run_index'],
                        len(tasks), args, output, identity)
                    with lock:
                        counters['executed' if executed else 'resumed'] += 1
                        atomic_json(output / 'parallel_progress.json', {
                            **counters, 'elapsed_s': time.monotonic() - started,
                            'batching': policy.stats(), 'scheduler': scheduler.stats()})
                finally:
                    scheduler.release(task)
        except BaseException:
            scheduler.cancel()
            raise

    try:
        with ThreadPoolExecutor(max_workers=len(urls)) as pool:
            futures = [pool.submit(actor, url) for url in urls]
            for future in futures:
                future.result()
    finally:
        policy.close()
        if native_provider is not None:
            native_provider.close()
        atomic_json(output / 'parallel_timing.json', {
            **counters, 'actors': len(urls), 'elapsed_s': time.monotonic() - started,
            'batching': policy.stats(), 'scheduler': scheduler.stats(),
            'native_condition_calls': native_provider.calls if native_provider else 0,
            'checkpoint': identity})


def parallel_run_contract(args, identity):
    """Semantic identity shared by fresh runs and audited serial migrations."""
    contract = {"checkpoint": identity,
            "subset_sha256": sha256_file(Path(args.subset_manifest)),
            "version": args.action_head_as_tool_version,
            "backbone": str(Path(args.backbone).resolve()),
            "repeats": args.repeats, "max_turns": args.max_turns,
            "max_tool_calls": args.max_tool_calls,
            "keep_last_images": args.keep_last_images,
            "success_contract": official_success_contract_identity()}
    # Preserve the identity of existing SFT runs, but never resume them as think.
    sampling = brain_sampling_contract()
    if sampling["enable_thinking"]:
        contract["brain_sampling"] = sampling
    if A.invalid_tool_call_policy() != "terminate":
        contract["invalid_tool_call_policy"] = A.invalid_tool_call_policy()
    if args.no_pass_at_k:
        contract["report_pass_at_k"] = False
    return contract


def main() -> None:
    global ACTIVE_QUERY_COUNT
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    runp = sub.add_parser("run")
    runp.add_argument("--checkpoint-dir", required=True)
    runp.add_argument("--subset-manifest", required=True)
    runp.add_argument("--backbone", required=True)
    runp.add_argument("--query-cache", required=True)
    runp.add_argument("--bench-urls", default="")
    runp.add_argument("--max-batch-size", type=int, default=32)
    runp.add_argument("--batch-wait-ms", type=float, default=50)
    runp.add_argument("--bench-url", required=True)
    runp.add_argument("--vllm-url", required=True)
    runp.add_argument("--output", required=True)
    runp.add_argument("--max-turns", type=int, default=30)
    runp.add_argument("--max-tool-calls", type=int, default=30)
    runp.add_argument("--keep-last-images", type=int, default=8)
    runp.add_argument("--timeout", type=int, default=600)
    runp.add_argument("--retries", type=int, default=3)
    runp.add_argument("--repeats", type=int, default=4)
    runp.add_argument("--no-pass-at-k", action="store_true")
    runp.add_argument("--limit-cases", type=int, default=0)
    runp.add_argument("--allow-partial", action="store_true")
    runp.add_argument("--action-head-as-tool-version", choices=("v1", "v2", "v3", "v4", "v4.1", "v4.2", "v4.3", "v4.4", "v4.5", "v4.6", "v4.7", "v4.8", "v4.9", "v4.10", "v4.11", "v6"), default="v1")
    aggp = sub.add_parser("aggregate")
    aggp.add_argument("--checkpoint-dir", required=True)
    aggp.add_argument("--subset-manifest", required=True)
    aggp.add_argument("--output", required=True)
    aggp.add_argument("--allow-partial", action="store_true")
    aggp.add_argument("--repeats", type=int, default=4)
    aggp.add_argument("--no-pass-at-k", action="store_true")
    aggp.add_argument("--action-head-as-tool-version", choices=ACTION_HEAD_PROTOCOL.SUPPORTED_VERSIONS, default="v1")
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.no_pass_at_k and args.repeats != 1:
        parser.error("--no-pass-at-k requires --repeats 1")
    identity = checkpoint_identity(args.checkpoint_dir)
    config_path = (Path(identity["action_head_dir"])
                   / "fast_slow_action_head_config.json")
    ACTIVE_QUERY_COUNT = int(json.loads(
        config_path.read_text(encoding="utf-8"))["query_num_queries"])
    output = Path(args.output)
    if args.command == "aggregate":
        print(json.dumps(aggregate(output, Path(args.subset_manifest), identity,
                                   allow_partial=args.allow_partial,
                                   action_head_as_tool_version=args.action_head_as_tool_version,
                                   repeats=args.repeats,
                                   report_pass_at_k=not args.no_pass_at_k),
                         ensure_ascii=False, indent=2))
        return

    if args.bench_urls:
        if args.max_batch_size <= 0 or args.batch_wait_ms < 0 or args.retries <= 0:
            parser.error("invalid parallel evaluation settings")
        config = json.loads(config_path.read_text())
        if identity.get("condition_mode") != "v5_native" and config.get("condition_source", "query") not in ("query", "query_hidden"):
            parser.error("bootstrap query layout requires a query-conditioned checkpoint")
        contract = parallel_run_contract(args, identity)
        contract_path = output / "parallel_contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("parallel output contract mismatch; use a new output directory")
        if not contract_path.exists() and any((output / "rollouts").glob("**/*.json")):
            raise ValueError("cannot adopt unaudited serial outputs into a parallel run")
        atomic_json(contract_path, contract)
    _, cases = load_subset(Path(args.subset_manifest))
    if args.limit_cases:
        cases = cases[:args.limit_cases]
    namespace = SimpleNamespace(
        checkpoint=identity["action_head_dir"], backbone=args.backbone,
        query_cache=args.query_cache, bench_url=args.bench_url,
        vllm_url=args.vllm_url, timeout=args.timeout, policy_seed=20260827,
        mode="think" if brain_sampling_contract()["enable_thinking"] else "instruct",
        max_turns=args.max_turns,
        max_tool_calls=args.max_tool_calls, keep_last_images=args.keep_last_images,
        action_head_as_tool_version=args.action_head_as_tool_version)
    if identity.get("condition_mode") == "v5_native":
        report = json.loads((Path(args.checkpoint_dir)/"report.json").read_text())
        namespace.context_encoder_model = report["frontend_model"]
        namespace.context_aggregator_checkpoint = identity["aggregator_path"]
    if args.bench_urls:
        parallel_rollouts(args, namespace, cases, output, identity)
    else:
        evaluator = TracedEvaluator(namespace)
        total_runs = len(cases) * args.repeats
        for case_index, case in enumerate(cases, 1):
            for repeat_index in range(1, args.repeats + 1):
                evaluate_repeat(evaluator, case, repeat_index,
                                (case_index - 1) * args.repeats + repeat_index,
                                total_runs, args, output, identity)
    summary = aggregate(output, Path(args.subset_manifest), identity,
                        allow_partial=args.allow_partial,
                        action_head_as_tool_version=args.action_head_as_tool_version,
                        repeats=args.repeats,
                        report_pass_at_k=not args.no_pass_at_k)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary["evaluation_valid"]:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
