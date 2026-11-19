#!/usr/bin/env python3
"""Bounded no-training smoke across vLLM hidden, action expert, and Sage gym."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import time
import urllib.request
from pathlib import Path

from PIL import Image
import torch

from action_head import contract as A
from action_head.hidden_provider import VllmHiddenStateProvider, VllmTokenizerClient
from action_head.policy import (
    Qwen35VisualTokenEncoder,
    TorchActionHeadPolicy,
    TrajectoryTransform,
)
from action_head.render_client import RemotePrefixExecutor, RemoteRenderClient
from action_head.runtime import ActionHeadRuntimeConfig, ActionHeadToolRuntime, OptionContext
from action_head.vllm_bridge import semantic_request_snapshot
from models.qwen36_fast_slow_action_head import Qwen36FastSlowActionHead


def post_json(url, payload, timeout):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def data_url(rgb):
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def transfer_metadata(response, choice, message):
    for source in (response, choice, message):
        value = source.get("kv_transfer_params") if isinstance(source, dict) else None
        if isinstance(value, dict):
            return value
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-url", default="http://127.0.0.1:28899")
    parser.add_argument("--model", default="Qwen3.6-35B")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--gym-url", default="http://127.0.0.1:8912")
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trajectory-mode", choices=("safe_random", "zyt_base"),
                        default="safe_random")
    parser.add_argument("--structured-output", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    gym = RemoteRenderClient(args.gym_url, timeout=args.timeout)
    observation, reset_info = gym.reset(
        scene="840483", episode_id="demo_840483_0_0", seed=20260824)
    instruction = str(observation.get("instruction") or
                      "Navigate to the target object using the visible open route.")
    contract = {
        "target": "the destination described by the instruction",
        "where": "along the visible route in the observation history",
        "plan": ["follow the visible free-space route", "approach the destination"],
    }
    exact = '<tool_call>{"name":"run_action_head","arguments":' + \
        json.dumps(contract, separators=(",", ":")) + "}</tool_call>"
    messages = [
        {"role": "system", "content": "Use the requested navigation tool."},
        {"role": "user", "content": [
            *[{"type": "image_url", "image_url": {
                "url": data_url(observation["rgb"])}} for _ in range(8)],
            {"type": "text", "text": instruction},
        ]},
    ]
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 96,
        "chat_template_kwargs": {"enable_thinking": False},
        "kv_transfer_params": {"include_output_tokens": True},
    }
    payload["messages"][0]["content"] = (
        "Copy the following text exactly, with no explanation:\n" + exact)
    if args.structured_output:
        payload["structured_outputs"] = {"choice": [exact]}
    before = time.monotonic()
    response = post_json(args.vllm_url.rstrip("/") + "/v1/chat/completions",
                         payload, args.timeout)
    slow_before_s = time.monotonic() - before
    choice = response["choices"][0]
    message = choice["message"]
    text = (message.get("reasoning_content") or "") + \
        (message.get("content") or "")
    match = A.TOOL_CALL_RE.search(text)
    if match is None:
        raise RuntimeError(f"slow VLM did not emit run_action_head: {text!r}")
    call = json.loads(A.strip_code_fence(match.group(1)))
    parsed_contract, error = A.validate_action_head_arguments(call.get("arguments"))
    if call.get("name") != "run_action_head" or parsed_contract is None:
        raise RuntimeError(error or f"unexpected tool call: {call}")
    snapshot = {
        "text": text,
        "usage": response.get("usage") or {},
        "kv_transfer_params": transfer_metadata(response, choice, message),
    }
    print(json.dumps({
        "stage": "slow_tool_call",
        "text": text,
        "reasoning_content": message.get("reasoning_content"),
        "usage": snapshot["usage"],
        "transfer": snapshot["kv_transfer_params"],
    }, ensure_ascii=False), flush=True)
    provider = VllmHiddenStateProvider(
        args.cache_root,
        VllmTokenizerClient(args.vllm_url, args.model, timeout=args.timeout),
        wait_seconds=30.0,
        cleanup=True,
    )
    bundle = provider(
        request=semantic_request_snapshot(
            messages=messages, instruction=instruction),
        response=snapshot,
        contract=parsed_contract,
    )

    dtype = torch.bfloat16
    vision = Qwen35VisualTokenEncoder.from_pretrained(
        args.backbone, device=args.device, dtype=dtype)
    head = Qwen36FastSlowActionHead.from_pretrained(
        args.checkpoint, map_location="cpu")
    if args.trajectory_mode == "zyt_base":
        transform = TrajectoryTransform.from_quantiles(
            [-0.10273411870002747, -0.10309220850467682, -0.5668500065803528],
            [0.1031249612569809, 0.10309189558029175, 0.5747644901275635],
            flip_y=True, flip_yaw=True)
    else:
        transform = TrajectoryTransform(scale=(0.02, 0.02, 0.02))
    policy = TorchActionHeadPolicy(
        head, vision, device=args.device, dtype=dtype,
        trajectory_transform=transform,
        seed=20260824,
    )
    option = OptionContext(
        option_id="sage-e2e-smoke",
        instruction=instruction,
        contract=parsed_contract,
        slow_condition=bundle,
        slow_reference_observation=observation,
        current_observation=observation,
    )
    runtime = ActionHeadToolRuntime(
        policy,
        RemotePrefixExecutor(gym),
        ActionHeadRuntimeConfig(
            prefix_steps=2, fast_step_limit=2, max_distance_m=0.5,
            max_sim_steps=8, max_seconds=180.0, max_abs_xy_m=0.2,
            max_abs_yaw_rad=1.2),
    )
    before = time.monotonic()
    option_result = runtime.run(option)
    option_s = time.monotonic() - before
    visible = option_result.model_visible()

    # Match run_agent's eight-image history contract: adding the post-option
    # observation evicts the oldest image rather than creating a ninth image.
    prior_user_content = list(messages[1]["content"])
    prior_images = [part for part in prior_user_content
                    if part.get("type") == "image_url"][-7:]
    prior_text = [part for part in prior_user_content
                  if part.get("type") == "text"]
    after_messages = [messages[0], {
        "role": "user", "content": prior_images + prior_text,
    }] + [
        {"role": "assistant", "content": text},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {
                "url": data_url(option_result.observation["rgb"])}},
            {"type": "text", "text": "[Tool result]\n" +
                json.dumps(visible, separators=(",", ":"))},
        ]},
    ]
    stop_text = '<tool_call>{"name":"stop","arguments":{"reason":"smoke complete"}}</tool_call>'
    followup = post_json(args.vllm_url.rstrip("/") + "/v1/chat/completions", {
        "model": args.model,
        "messages": after_messages,
        "temperature": 0.0,
        "max_tokens": 64,
        "structured_outputs": {"choice": [stop_text]},
    }, args.timeout)
    traces = [entry for entry in option_result.internal_log
              if entry.get("event") == "policy_predict"]
    executions = [entry for entry in option_result.internal_log
                  if "executed_prefix" in entry]
    report = {
        "ok": bool(option_result.ok and len(traces) >= 2 and len(executions) >= 2),
        "scope": "end-to-end wiring with constrained slow output",
        "trajectory_mode": args.trajectory_mode,
        "slow_vlm_calls_total": 2,
        "slow_vlm_calls_associated_with_option": 1,
        "additional_slow_vlm_calls_during_fast_loop": 0,
        "slow_before_latency_s": round(slow_before_s, 6),
        "option_latency_s": round(option_s, 6),
        "hidden_request_id": bundle.request_id,
        "semantic_hidden_shape": list(bundle.semantic_hidden.shape),
        "contract_hidden_shape": list(bundle.contract_hidden.shape),
        "contract_field_ids": sorted(set(bundle.field_ids.tolist())),
        "action_head_calls": len(traces),
        "executed_chunks": len(executions),
        "policy_trace": traces,
        "execution_trace": executions,
        "model_visible_result": visible,
        "followup_text": followup["choices"][0]["message"].get("content") or "",
        "reset_scene": reset_info.get("scene"),
    }
    forbidden = ("distance_to_goal", "world_xy", "world_yaw_deg",
                 "goal_coordinate", "semantic_region", "planner")
    encoded_visible = json.dumps(visible)
    report["model_visible_forbidden_fields"] = [
        key for key in forbidden if key in encoded_visible]
    report["second_prediction_uses_post_prefix_rgb"] = (
        len(traces) >= 2 and len(executions) >= 1 and
        traces[0]["current_rgb_sha256"] != traces[1]["current_rgb_sha256"])
    report["ok"] = bool(
        report["ok"] and report["second_prediction_uses_post_prefix_rgb"]
        and not report["model_visible_forbidden_fields"])
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
