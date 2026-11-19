#!/usr/bin/env python3
"""A/B ordinary requests on native and query-enabled vLLM endpoints."""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


def run(url: str, model: str, prompt: str, timeout: float, *, payload: dict | None = None):
    payload = payload if payload is not None else {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "max_tokens": 128,
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    first = None
    text, completion_tokens, observed_usage = [], 0, {}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            value = json.loads(line[5:])
            usage = value.get("usage") or {}
            observed_usage.update(usage)
            completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
            choice = (value.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
            if piece:
                if first is None:
                    first = time.perf_counter()
                text.append(piece)
    end = time.perf_counter()
    return {
        "text": "".join(text),
        "ttft_s": (first or end) - start,
        "total_s": end - start,
        "completion_tokens": completion_tokens,
        "usage": observed_usage,
        "tokens_per_s": completion_tokens / max(end - (first or end), 1e-9),
    }


def median(rows, key):
    return statistics.median(row[key] for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-url", required=True)
    parser.add_argument("--query-url", required=True)
    parser.add_argument("--model", default="Qwen3.6-35B")
    parser.add_argument("--prompt", default="Briefly describe a safe indoor route.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-regression", type=float, default=0.05)
    args = parser.parse_args()

    for _ in range(args.warmup):
        run(args.native_url, args.model, args.prompt, args.timeout)
        run(args.query_url, args.model, args.prompt, args.timeout)
    native, query = [], []
    for _ in range(args.rounds):
        native.append(run(args.native_url, args.model, args.prompt, args.timeout))
        query.append(run(args.query_url, args.model, args.prompt, args.timeout))
    report = {
        "ordinary_request_only": True,
        "rounds": args.rounds,
        "native": {key: median(native, key) for key in
                   ("ttft_s", "total_s", "tokens_per_s")},
        "query_enabled": {key: median(query, key) for key in
                          ("ttft_s", "total_s", "tokens_per_s")},
        "all_outputs_equal": all(
            left["text"] == right["text"] for left, right in zip(native, query)),
    }
    report["ttft_regression"] = (
        report["query_enabled"]["ttft_s"] / report["native"]["ttft_s"] - 1.0)
    report["throughput_regression"] = 1.0 - (
        report["query_enabled"]["tokens_per_s"]
        / report["native"]["tokens_per_s"])
    report["ok"] = bool(
        report["all_outputs_equal"]
        and report["ttft_regression"] <= args.max_regression
        and report["throughput_regression"] <= args.max_regression)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
