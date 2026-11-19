#!/usr/bin/env python3
"""End-to-end probe for real generated-token hidden states from vLLM."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

from action_head import contract as A
from action_head.hidden_provider import VllmHiddenStateProvider, VllmTokenizerClient
from action_head.vllm_bridge import semantic_request_snapshot


def post_json(url, payload, timeout):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def transfer_metadata(response, choice, message):
    for source in (response, choice, message):
        value = source.get("kv_transfer_params") if isinstance(source, dict) else None
        if isinstance(value, dict):
            return value
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:28899")
    parser.add_argument("--model", default="Qwen3.6-35B")
    parser.add_argument(
        "--cache-root",
        default="/workspace/action_head_hidden_cache")
    parser.add_argument(
        "--report",
        default="/workspace/discussion_logs/"
                "vllm_generated_hidden_probe_20260821.json")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--image", action="append", default=[],
        help="Local image path; repeat this option to probe multimodal history.")
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()

    expected = {
        "target": "the chair",
        "where": "on the right side of the current view",
        "plan": ["align with the open route", "approach the chair"],
    }
    exact = (
        '<tool_call>{"name":"run_action_head","arguments":'
        + json.dumps(expected, ensure_ascii=False, separators=(",", ":"))
        + "}</tool_call>"
    )
    user_text = "Copy this byte-for-byte, with no explanation:\n" + exact
    if args.image:
        user_content = [
            {
                "type": "image_url",
                "image_url": {"url": "file://" + os.path.abspath(path)},
            }
            for path in args.image
        ]
        user_content.append({"type": "text", "text": user_text})
    else:
        user_content = user_text
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "Return only the exact requested text."},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "stream": False,
        # Probe-only determinism: isolate hidden-state transport correctness
        # from the model's willingness to obey a byte-for-byte copy request.
        "structured_outputs": {"choice": [exact]},
        "kv_transfer_params": {"include_output_tokens": True},
    }
    started = time.monotonic()
    api_root = args.base_url.rstrip("/")
    if api_root.endswith("/v1"):
        api_root = api_root[:-3]
    response = post_json(
        api_root + "/v1/chat/completions", payload, args.timeout)
    latency = time.monotonic() - started
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") or ""
    transfer = transfer_metadata(response, choice, message)
    snapshot = {
        "text": text,
        "usage": response.get("usage") or {},
        "kv_transfer_params": transfer,
    }

    contract = None
    for match in A.TOOL_CALL_RE.finditer(text):
        try:
            call = json.loads(A.strip_code_fence(match.group(1)))
        except Exception:
            continue
        value, error = A.validate_action_head_arguments(call.get("arguments"))
        if call.get("name") == "run_action_head" and value is not None and not error:
            contract = value
            break
    if contract is None:
        raise RuntimeError(f"model did not emit a valid tool call: {text!r}")

    tokenizer = VllmTokenizerClient(args.base_url, args.model, timeout=args.timeout)
    provider = VllmHiddenStateProvider(
        args.cache_root, tokenizer, cleanup=args.cleanup, wait_seconds=30.0)
    selected = provider(
        request=semantic_request_snapshot(
            messages=payload["messages"], instruction=user_text),
        response=snapshot,
        contract=contract,
    )
    report = {
        "ok": True,
        "model": args.model,
        "image_count": len(args.image),
        "latency_s": round(latency, 6),
        "response_text": text,
        "finish_reason": choice.get("finish_reason"),
        "usage": snapshot["usage"],
        "artifact_path": selected.source_path,
        "artifact_token_count": selected.source_token_count,
        "prompt_token_count": selected.prompt_token_count,
        "completion_token_count": selected.completion_token_count,
        "contract_token_count": int(selected.token_ids.shape[0]),
        "contract_hidden_shape": list(selected.hidden.shape),
        "contract_hidden_dtype": str(selected.hidden.dtype),
        "contract_hidden_finite": bool(
            __import__("numpy").isfinite(selected.hidden).all()),
        "contract_text": selected.contract_text,
        "boundary_expansion": selected.boundary_expansion,
        "generated_tokens_present": (
            selected.source_token_count >=
            selected.prompt_token_count + selected.completion_token_count - 1
        ),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
