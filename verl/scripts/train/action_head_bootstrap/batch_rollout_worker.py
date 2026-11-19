#!/usr/bin/env python3
"""Concurrent v3 rollout worker sharing one dynamically batched action expert."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
from case_scheduler import SceneCaseScheduler

REPO = Path(__file__).resolve().parents[4]
EVAL_DIR = REPO / "verl/scripts/eval/qwen36_4query_ckpt_sweep"
sys.path[:0] = [str(REPO), str(REPO / "verl"), str(REPO / "vln_demo/src"),
                str(REPO / "verl/sft"), str(EVAL_DIR)]

import closedloop_eval as CE  # noqa: E402
import collect_action_head_recovery as CR  # noqa: E402
from action_head.policy import (  # noqa: E402
    DynamicBatchingPolicy, PolicyProxy, Qwen35VisualTokenEncoder, TorchActionHeadPolicy,
    TrajectoryTransform)


def load_cases(manifest: Path, max_cases: int, seed: int) -> list[dict]:
    """Use exactly the constructor's ordering before distributed sharding."""
    return CR.load_training_cases(manifest, seed, max_cases)


def shard_cases(cases: list[dict], worker_index: int, worker_count: int,
                mode: str) -> list[dict]:
    if not 0 <= worker_index < worker_count:
        raise ValueError("worker-index must be in [0, worker-count)")
    if mode == "strided":
        return cases[worker_index::worker_count]
    if mode != "contiguous":
        raise ValueError(f"unknown shard mode: {mode}")
    begin = len(cases) * worker_index // worker_count
    end = len(cases) * (worker_index + 1) // worker_count
    return cases[begin:end]


def env_seed(case: dict) -> int:
    raw = f"{case['scene']}:{case['episode_id']}".encode()
    return 20260908 + int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--backbone", required=True, type=Path)
    parser.add_argument("--query-cache", required=True, type=Path)
    parser.add_argument("--vllm-url", required=True)
    parser.add_argument("--bench-urls", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--worker-index", required=True, type=int)
    parser.add_argument("--worker-count", required=True, type=int)
    parser.add_argument("--worker-shard-mode", choices=("contiguous", "strided"),
                        default="contiguous")
    parser.add_argument("--max-cases", type=int, default=56)
    parser.add_argument("--case-seed", type=int, default=20260903)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-tool-calls", type=int, default=12)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--batch-wait-ms", type=float, default=12.0)
    parser.add_argument("--deduplicate-first-visual", action="store_true")
    parser.add_argument("--archive-query-hidden", action="store_true",
                        help="Keep v6 query tensors only for optional golden-route slow resampling.")
    parser.add_argument("--case-scheduling", choices=("static", "scene"), default="static")
    parser.add_argument("--renderer-count", type=int, default=3)
    parser.add_argument("--max-active-scenes-per-renderer", type=int, default=48)
    parser.add_argument("--max-active-cases-per-renderer", type=int, default=64)
    parser.add_argument("--action-head-as-tool-version", choices=("v3", "v4.3", "v6"),
                        default="v3")
    args = parser.parse_args()
    bench_urls = [value for value in args.bench_urls.split(",") if value]
    if not bench_urls:
        parser.error("--bench-urls is empty")
    config = json.loads((args.checkpoint_dir / "action_head" /
                         "fast_slow_action_head_config.json").read_text(
                             encoding="utf-8"))
    CE.ACTIVE_QUERY_COUNT = int(config["query_num_queries"])

    low = [-0.10273411870002747, -0.10309220850467682, -0.5668500065803528]
    high = [0.1031249612569809, 0.10309189558029175, 0.5747644901275635]
    transform = TrajectoryTransform.from_quantiles(low, high, flip_y=True,
                                                   flip_yaw=True)
    visual = Qwen35VisualTokenEncoder.from_pretrained(
        args.backbone, device="cuda:0", dtype=torch.bfloat16)
    base_policy = TorchActionHeadPolicy.from_pretrained(
        args.checkpoint_dir / "action_head", visual, device="cuda:0",
        dtype=torch.bfloat16, seed=20260827, trajectory_transform=transform,
        deduplicate_first_visual=args.deduplicate_first_visual)
    policy = DynamicBatchingPolicy(
        base_policy, max_batch_size=args.max_batch_size,
        batch_wait_ms=args.batch_wait_ms)

    cases = load_cases(args.manifest, args.max_cases, args.case_seed)
    try:
        cases = shard_cases(
            cases, args.worker_index, args.worker_count, args.worker_shard_mode)
    except ValueError as exc:
        parser.error(str(exc))
    scheduler = (SceneCaseScheduler(
        cases, replicas=args.renderer_count,
        max_scenes=args.max_active_scenes_per_renderer,
        max_cases=args.max_active_cases_per_renderer)
        if args.case_scheduling == "scene" else None)
    output = args.output_dir / f"worker_{args.worker_index:02d}"
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    counters = {"completed": 0, "valid": 0, "success": 0, "tool_calls": 0}
    lock = threading.Lock()
    actor_errors = []

    def run_actor(actor_index: int, actor_cases: list[dict], bench_url: str):
        namespace = SimpleNamespace(
            checkpoint=str(args.checkpoint_dir / "action_head"),
            backbone=str(args.backbone), query_cache=str(args.query_cache),
            bench_url=bench_url, vllm_url=args.vllm_url, timeout=600,
            policy_seed=20260827, mode="instruct", max_turns=args.max_turns,
            max_tool_calls=args.max_tool_calls, keep_last_images=8,
            action_head_as_tool_version=args.action_head_as_tool_version,
            capture_slow_messages=True,
            query_archive=(str(output / "query_hidden") if
                           args.action_head_as_tool_version != "v6" or args.archive_query_hidden
                           else None),
            shared_action_head_policy=PolicyProxy(policy))
        evaluator = CE.TracedEvaluator(namespace)
        if scheduler is not None:
            actor_cases = iter(scheduler.acquire, None)
        for case in actor_cases:
            try:
                case_started = time.monotonic()
                try:
                    row = evaluator.rollout(case, env_seed(case))
                    row = CR.externalize_rollout(
                        row, output / "slow_context_images" /
                        f"{case['scene']}_{case['episode_id']}")
                    if (args.action_head_as_tool_version == "v6" and
                            not args.archive_query_hidden):
                        for turn in row.get("turns", []):
                            for call in turn.get("calls", []):
                                (call.get("action_trace") or {}).pop("query_artifact", None)
                except Exception as exc:
                    row = {**case, "evaluation_valid": False, "success": False,
                           "infra_error": repr(exc), "traceback": traceback.format_exc()}
                row["benchmark"] = {
                    "worker_index": args.worker_index, "actor_index": actor_index,
                    "elapsed_s": time.monotonic() - case_started,
                }
                name = hashlib.sha256(
                    f"{case['scene']}:{case['episode_id']}".encode()).hexdigest()[:16]
                atomic_json(output / "rollouts" / f"{name}.json", row)
                calls = sum(len(turn.get("calls") or []) for turn in row.get("turns", []))
                with lock:
                    counters["completed"] += 1
                    counters["valid"] += int(bool(row.get("evaluation_valid")))
                    counters["success"] += int(bool(row.get("success")))
                    counters["tool_calls"] += calls
                print(f"[worker {args.worker_index} actor {actor_index}] "
                      f"{case['scene']}:{case['episode_id']} valid={row.get('evaluation_valid')} "
                      f"success={row.get('success')} elapsed={time.monotonic()-case_started:.2f}s",
                      flush=True)
            finally:
                if scheduler is not None:
                    scheduler.release(case)

    def actor(*values):
        try:
            run_actor(*values)
        except Exception:
            with lock:
                actor_errors.append(traceback.format_exc())
            if scheduler is not None:
                scheduler.cancel()

    threads = []
    for actor_index, bench_url in enumerate(bench_urls):
        thread = threading.Thread(
            target=actor,
            args=(actor_index, cases[actor_index::len(bench_urls)], bench_url),
            name=f"rollout-actor-{actor_index}")
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    elapsed = time.monotonic() - started
    policy_stats = policy.stats()
    policy.close()
    summary = {
        "schema": "action_head_v3_traced_rollout_worker_v2",
        "action_head_as_tool_version": args.action_head_as_tool_version,
        "worker_index": args.worker_index, "worker_count": args.worker_count,
        "worker_shard_mode": args.worker_shard_mode,
        "actors": len(bench_urls), "elapsed_s": elapsed, **counters,
        "expected_cases": len(cases), "actor_errors": actor_errors,
        "case_scheduling": args.case_scheduling,
        "scheduler": scheduler.stats() if scheduler is not None else None,
        "trajectories_per_hour": counters["completed"] * 3600 / elapsed,
        "action_head": policy_stats,
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if actor_errors or counters["valid"] != counters["completed"] or counters["completed"] != len(cases):
        raise SystemExit(4)


if __name__ == "__main__":
    main()
