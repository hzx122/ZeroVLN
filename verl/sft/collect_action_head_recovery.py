#!/usr/bin/env python3
"""Collect one round of v2 action expert recovery supervision.

The frozen slow policy is rolled out with the current action expert on training
episodes.  A single privileged VLM judge labels each post-option ToolCall as
one of: no mismatch, a valid correction aligned with an oracle recovery route,
or a wrong/missed correction.  Only valid correction calls become R1--R4 fast
windows; selection diagnostics remain audit-only and are never model-visible.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(REPO), str(REPO / "verl"), str(REPO / "verl" / "sft"),
    str(REPO / "vln_demo" / "src"),
    str(REPO / "verl/scripts/eval/qwen36_4query_ckpt_sweep"),
]

from action_head import protocol_v2 as V2  # noqa: E402
from build_abotn_sft_data import (  # noqa: E402
    _flip_to_base, normalize_waypoints, relative_future, render_frame)
from build_sage_sft_data import (  # noqa: E402
    CAMERA_HEIGHT, SAGE_CAM_INTRINSICS, _arc_resample_partial)
from sage.datasets import SageSet  # noqa: E402
from sage.occmap import OccMap  # noqa: E402
from sage.planner import (  # noqa: E402
    CenteredAStarConfig, PathSmoothingConfig, plan_path, smooth_path)
import closedloop_eval as CE  # noqa: E402

JUDGE_CHOICES = ("RECOVERY_ACCEPT", "NO_MISMATCH", "CORRECTION_WRONG")
ROBOT_RADIUS_M = 0.07
OFFSETS_M = (0.0, 0.4, 0.8, 1.2)


def post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")
    temporary.replace(path)


def externalize_messages(messages: list[dict], image_dir: Path) -> list[dict]:
    """Replace inline JPEG URLs by content-addressed file URLs."""
    result = copy.deepcopy(messages)
    image_dir.mkdir(parents=True, exist_ok=True)
    for message in result:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue
            url = str((item.get("image_url") or {}).get("url") or "")
            if not url.startswith("data:image/") or ";base64," not in url:
                continue
            encoded = url.split(";base64,", 1)[1]
            raw = base64.b64decode(encoded)
            suffix = ".png" if url.startswith("data:image/png") else ".jpg"
            path = image_dir / (hashlib.sha256(raw).hexdigest() + suffix)
            if not path.exists():
                path.write_bytes(raw)
            item["image_url"]["url"] = "file://" + str(path.resolve())
    return result


def externalize_rollout(row: dict, image_dir: Path) -> dict:
    result = copy.deepcopy(row)
    for turn_index, turn in enumerate(result.get("turns", []), 1):
        if turn.get("slow_messages"):
            turn["slow_messages"] = externalize_messages(
                turn["slow_messages"], image_dir / f"turn_{turn_index:02d}")
    return result


def load_training_cases(manifest: Path, seed: int, max_cases: int) -> list[dict]:
    by_task = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != "train":
            continue
        key = (row["source_scene"], row["source_task_id"])
        by_task.setdefault(key, {
            "stage": "p0", "scene": key[0], "episode_id": key[1],
            "instruction": row["instruction"],
        })
    grouped = {}
    for case in by_task.values():
        grouped.setdefault(case["scene"], []).append(case)
    rng = random.Random(seed)
    scenes = list(grouped)
    rng.shuffle(scenes)
    cases = []
    for scene in scenes:
        rng.shuffle(grouped[scene])
        cases.extend(grouped[scene])
    return cases[:max_cases] if max_cases else cases


def task_lookup(cases: list[dict], sage_split: str = "action_head_train_p0_unified") \
        -> tuple[SageSet, dict[tuple[str, str], object]]:
    dataset = SageSet(split=sage_split)
    wanted = {}
    by_scene = {}
    for case in cases:
        by_scene.setdefault(case["scene"], set()).add(case["episode_id"])
    for scene, ids in by_scene.items():
        for task in dataset.tasks(scene):
            if task.task_id in ids:
                wanted[(scene, task.task_id)] = task
    missing = [(c["scene"], c["episode_id"]) for c in cases
               if (c["scene"], c["episode_id"]) not in wanted]
    if missing:
        raise RuntimeError(f"training tasks missing from SageSet: {missing[:5]}")
    return dataset, wanted


def _route_yaws(points: list[tuple[float, float]], start_yaw: float) -> list[float]:
    yaws = [float(start_yaw)]
    for index in range(1, len(points)):
        x0, y0 = points[index - 1]
        x1, y1 = points[index]
        yaws.append(math.atan2(y1 - y0, x1 - x0))
    return list(np.unwrap(np.asarray(yaws, dtype=np.float64)))


def oracle_route(dataset: SageSet, scene: str, start_xy: tuple[float, float],
                 start_yaw: float, goal_xy: tuple[float, float]):
    occ = OccMap(dataset.scene_dir(scene), collision_mode="usd_rooms",
                 furniture_mode="none")
    raw = plan_path(
        occ, start_xy, goal_xy, ROBOT_RADIUS_M, mode="ecm",
        centered_config=CenteredAStarConfig())
    if raw is None or len(raw) < 2:
        return None, "plan_failed"
    try:
        points, smoothing = smooth_path(
            occ, raw, ROBOT_RADIUS_M, PathSmoothingConfig())
    except Exception as exc:
        return None, f"smooth_failed:{type(exc).__name__}:{exc}"
    if len(points) < 2:
        return None, "smoothed_route_too_short"
    # Match the golden builder's final physical checks without importing its
    # argv-driven module: dense swept occupancy plus exact try_step replay.
    for left, right in zip(points, points[1:]):
        segment = math.hypot(right[0] - left[0], right[1] - left[1])
        for index in range(max(2, int(segment / 0.05) + 1) + 1):
            ratio = index / float(max(2, int(segment / 0.05) + 1))
            if not occ.is_free(left[0] + (right[0] - left[0]) * ratio,
                               left[1] + (right[1] - left[1]) * ratio,
                               ROBOT_RADIUS_M):
                return None, "swept_collision"
    points_np = np.asarray(points, dtype=np.float64)
    segments = np.linalg.norm(np.diff(points_np[:, :2], axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    replay_targets = np.linspace(0.0, float(cumulative[-1]),
                                 max(2, int(math.ceil(cumulative[-1] / 0.1))))
    replay = np.column_stack((
        np.interp(replay_targets, cumulative, points_np[:, 0]),
        np.interp(replay_targets, cumulative, points_np[:, 1])))
    actual_x, actual_y = map(float, replay[0])
    replay_collisions = 0
    for index in range(1, len(replay)):
        target_x, target_y = map(float, replay[index])
        requested = math.hypot(target_x - actual_x, target_y - actual_y)
        next_x, next_y, hit = occ.try_step(
            actual_x, actual_y, target_x, target_y, ROBOT_RADIUS_M, True)
        replay_collisions += int(bool(hit))
        moved = math.hypot(next_x - actual_x, next_y - actual_y)
        if requested > 1e-9 and moved < requested * 0.95:
            return None, f"execution_replay_blocked:{index}"
        actual_x, actual_y = float(next_x), float(next_y)
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    yaws = _route_yaws(points, start_yaw)
    arc = np.concatenate(([0.0], np.cumsum([
        math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
        for i in range(len(xs) - 1)])))
    return {"xs": xs, "ys": ys, "yaws": yaws, "arc": arc,
            "length_m": float(arc[-1]), "smoothing": smoothing,
            "execution_replay": {"steps": len(replay) - 1,
                                 "collisions": replay_collisions}}, None


def pose_at(route: dict, offset_m: float) -> tuple[float, float, float]:
    at = min(float(offset_m), float(route["arc"][-1]))
    return (float(np.interp(at, route["arc"], route["xs"])),
            float(np.interp(at, route["arc"], route["ys"])),
            float(np.interp(at, route["arc"], route["yaws"])))


def route_suffix(route: dict, offset_m: float):
    x0, y0, yaw0 = pose_at(route, offset_m)
    keep = int(np.searchsorted(route["arc"], offset_m, side="right"))
    xs = [x0] + route["xs"][keep:]
    ys = [y0] + route["ys"][keep:]
    yaws = [yaw0] + route["yaws"][keep:]
    if len(xs) < 2:
        return None
    sampled = _arc_resample_partial(
        xs, ys, yaws, 0, step_m=0.1, n=16, min_future_m=0.1)
    if sampled is None:
        return None
    fx, fy, fyaw, mask, remaining = sampled
    sim_delta = relative_future(fx, fy, fyaw, x0, y0, yaw0)
    target = normalize_waypoints(_flip_to_base(sim_delta, has_yaw=True))
    return (x0, y0, yaw0), target, mask, remaining


def render_pose(render_url: str, scene: str, pose, target: Path,
                preloaded: set[str]) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(render_frame(
            render_url, scene, pose[0], pose[1], CAMERA_HEIGHT, pose[2],
            preloaded, SAGE_CAM_INTRINSICS))
    return target.resolve()


def correction_judge(vllm_url: str, model: str, timeout: float,
                     instruction: str, previous: dict, current: dict,
                     evidence_images: list[Path]) -> tuple[str, str]:
    prompt = (
        "Classify this post-option navigation decision. RECOVERY_ACCEPT means "
        "(1) the actual previous execution did not fulfill the previous semantic "
        "ToolCall, and (2) the new ToolCall is a correction consistent with the "
        "oracle recovery-route images in chronological order. A step-budget return, "
        "completion=unknown, or ordinary partial progress is not by itself a mismatch. "
        "An unchanged ToolCall is not automatically NO_MISMATCH: when the fast system "
        "collided, was blocked, made no progress, or moved against the requested "
        "direction, repeating the still-correct semantic intent can be a valid recovery "
        "decision because its hidden query and current observation have changed. A "
        "mismatch needs "
        "concrete contradictory execution evidence such as wrong direction, blockage, "
        "collision, no progress, overshoot, or failure of the stated semantic stage. "
        "The new ToolCall must remain consistent with the oracle route from the actual "
        "post-execution state. NO_MISMATCH means this is not a recovery anchor. CORRECTION_WRONG "
        "means a mismatch exists but the new ToolCall is "
        "absent, unsafe, or inconsistent with the oracle route. Judge semantics, "
        "not exact wording. Return exactly one label.\nInstruction: " + instruction +
        "\nPrevious ToolCall: " + json.dumps(previous.get("arguments"), ensure_ascii=False) +
        "\nActual ToolResult: " + json.dumps(previous.get("result"), ensure_ascii=False) +
        "\nActual execution trace: " + json.dumps({
            "option_actual_endpoint": (previous.get("action_trace") or {}).get(
                "option_actual_endpoint"),
            "option_actual_direction": (previous.get("action_trace") or {}).get(
                "option_actual_direction"),
            "chunks": [{
                "actual_endpoint": item.get("actual_endpoint"),
                "actual_direction": item.get("actual_direction"),
                "blocked": item.get("blocked"), "collided": item.get("collided"),
            } for item in (previous.get("action_trace") or {}).get("prefixes", [])],
        }, ensure_ascii=False) +
        "\nNew ToolCall: " + json.dumps(current.get("arguments"), ensure_ascii=False))
    messages = [
        {"role": "system", "content": "Return exactly one allowed label."},
        {"role": "user", "content": [
            *[{"type": "image_url", "image_url": {"url": "file://" + str(path)}}
              for path in evidence_images],
            {"type": "text", "text": prompt},
        ]},
    ]
    payload = {
        "model": model, "messages": messages, "temperature": 0.0,
        "max_tokens": 16, "chat_template_kwargs": {"enable_thinking": False},
        "structured_outputs": {"choice": list(JUDGE_CHOICES)},
    }
    response = post_json(vllm_url.rstrip("/") + "/v1/chat/completions", payload, timeout)
    message = response["choices"][0]["message"]
    answer = ((message.get("reasoning_content") or "") +
              (message.get("content") or "")).strip().upper()
    label = next((choice for choice in JUDGE_CHOICES
                  if re.search(rf"\b{choice}\b", answer)), "CORRECTION_WRONG")
    return label, answer


def flatten_calls(row: dict) -> list[dict]:
    flattened = []
    for turn in row.get("turns", []):
        calls = turn.get("calls", [])
        for call in calls:
            flattened.append({**call, "turn": turn})
        if not calls:
            # Preserve an explicit post-option failure to produce any valid
            # tool call.  It can be audited as a missed correction, but never
            # becomes fast-system supervision.
            flattened.append({"name": "none", "arguments": {}, "turn": turn})
    return flattened


def has_concrete_execution_mismatch(previous: dict) -> bool:
    """Evidence that the fast execution, rather than wording, failed its option."""
    result = previous.get("result") or {}
    trace = previous.get("action_trace") or {}
    prefixes = trace.get("prefixes") or []
    if bool(result.get("blocked")) or bool(result.get("collision")):
        return True
    if any(bool(item.get("blocked")) or bool(item.get("collided"))
           or bool(item.get("no_progress")) for item in prefixes):
        return True
    # Older traces did not persist an explicit no_progress flag.  Recover it
    # from requested-vs-actual translational progress while leaving legitimate
    # in-place turns alone.
    for item in prefixes:
        requested_path = float((item.get("requested_endpoint") or {}).get(
            "path_m") or 0.0)
        actual_path = float((item.get("actual_endpoint") or {}).get(
            "path_m") or 0.0)
        if requested_path >= 0.1 and actual_path < min(0.02, requested_path * 0.1):
            return True
    requested = str(trace.get("contract_direction") or "").lower()
    actual = str(trace.get("option_actual_direction") or "").lower()
    directional = {"left", "right", "straight"}
    return requested in directional and actual in directional and requested != actual


def correction_mode(previous: dict, current: dict) -> str:
    """Audit-only distinction; never expose this privileged label to the model."""
    text = json.dumps(current.get("arguments") or {}, ensure_ascii=False).lower()
    if re.search(r"\b(back up|back away|reverse|retreat|step away|move away)\b|"
                 r"后退|倒退|退后", text):
        return "explicit_slow_backtrack"
    if re.search(r"\b(turn|rotate|pivot|reorient|adjust (the )?(heading|direction))\b|"
                 r"转向|转身|旋转|调整方向", text):
        return "explicit_slow_reorientation"
    if current.get("arguments") == previous.get("arguments"):
        return "same_intent_fast_recovery"
    return "changed_call_without_explicit_escape_motion"


def visual_tokens(evaluator, image_path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    policy = evaluator.runtime.tool_executor.runtime.policy
    tokens, _mask = policy._visual(rgb)
    return tokens[0].float().cpu().numpy().astype(np.float32)


def build_candidate(*, evaluator, output: Path, case: dict, task, dataset,
                    previous: dict, current: dict, candidate_index: int,
                    vllm_url: str, model: str, timeout: float,
                    render_url: str, preloaded: set[str]) -> tuple[list[dict], dict]:
    trace = current.get("action_trace") or {}
    previous_prefixes = (previous.get("action_trace") or {}).get("prefixes") or []
    prefixes = trace.get("prefixes") or []
    # The recovery state is the actual end of the failed previous option.  It
    # is also the start of the new call, but deriving it from the previous
    # trace lets us audit stop/no-call as missed corrections.
    before = ((previous_prefixes[-1].get("after_world_pose")
               if previous_prefixes else None)
              or (prefixes[0].get("before_world_pose") if prefixes else None)
              or {})
    world_xy, yaw_deg = before.get("world_xy"), before.get("world_yaw_deg")
    audit = {"candidate_index": candidate_index,
             "previous_arguments": previous.get("arguments"),
             "previous_result": previous.get("result"),
             "correction_arguments": current.get("arguments"),
             "correction_mode": correction_mode(previous, current)}
    if not previous_prefixes:
        return [], {**audit, "accepted": False,
                    "judge_label": "NO_MISMATCH",
                    "reason": "no_fast_execution_to_correct"}
    if (current.get("name") == "run_action_head"
            and current.get("arguments") == previous.get("arguments")
            and not has_concrete_execution_mismatch(previous)):
        return [], {**audit, "accepted": False,
                    "judge_label": "NO_MISMATCH",
                    "reason": "unchanged_tool_call_without_execution_mismatch"}
    if world_xy is None or yaw_deg is None:
        return [], {**audit, "accepted": False, "reason": "missing_actual_start_pose"}
    route, error = oracle_route(
        dataset, case["scene"], (float(world_xy[0]), float(world_xy[1])),
        math.radians(float(yaw_deg)), tuple(task.goal_xy))
    if route is None:
        return [], {**audit, "accepted": False, "reason": error}
    if route["length_m"] < OFFSETS_M[-1] + 0.1:
        return [], {**audit, "accepted": False,
                    "reason": "oracle_route_too_short_for_r1_r4",
                    "oracle_length_m": route["length_m"]}
    anchor = hashlib.sha256(
        f"{case['scene']}:{case['episode_id']}:{candidate_index}".encode()).hexdigest()[:16]
    frame_dir = output / "frames" / anchor
    poses = [pose_at(route, offset) for offset in OFFSETS_M]
    paths = [render_pose(render_url, case["scene"], pose,
                         frame_dir / f"r{index}.jpg", preloaded)
             for index, pose in enumerate(poses, 1)]
    label, raw_judge = correction_judge(
        vllm_url, model, timeout, case["instruction"], previous, current, paths)
    if label == "RECOVERY_ACCEPT" and current.get("name") != "run_action_head":
        label = "CORRECTION_WRONG"
        raw_judge += " [collector: non-action decision cannot supervise recovery]"
    audit.update({"judge_label": label, "judge_raw": raw_judge,
                  "oracle_length_m": route["length_m"],
                  "oracle_smoothing": route["smoothing"],
                  "oracle_execution_replay": route["execution_replay"],
                  "world_pose": [world_xy[0], world_xy[1], yaw_deg],
                  "evidence_images": [str(path) for path in paths]})
    if label != "RECOVERY_ACCEPT":
        return [], {**audit, "accepted": False, "reason": label.lower()}
    query_path = ((trace.get("query_artifact") or {}).get("archive_path"))
    slow_messages = current["turn"].get("slow_messages")
    slow_completion = str(current["turn"].get("assistant") or "").strip()
    if not query_path or not slow_messages or not slow_completion.endswith("</tool_call>"):
        return [], {**audit, "accepted": False,
                    "reason": "missing_training_prefix_or_query_hidden"}
    query_hidden = np.load(query_path).astype(np.float32)
    persisted_messages = externalize_messages(
        slow_messages, output / "slow_context_images" / anchor)
    reference = visual_tokens(evaluator, paths[0])
    rows = []
    for index, (offset, image_path) in enumerate(zip(OFFSETS_M, paths), 1):
        suffix = route_suffix(route, offset)
        if suffix is None:
            return [], {**audit, "accepted": False,
                        "reason": f"r{index}_target_unavailable"}
        pose, target, mask, remaining = suffix
        current_visual = visual_tokens(evaluator, image_path)
        sample_path = output / "samples" / f"{anchor}_r{index}.npz"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            sample_path, query_hidden=query_hidden,
            reference_visual_tokens=reference,
            current_visual_tokens=current_visual,
            target_actions=target.astype(np.float32),
            action_mask=np.asarray(mask, dtype=np.bool_))
        rows.append({
            "schema": V2.BUNDLE_SCHEMA, **V2.protocol_identity(),
            "sample": str(sample_path.resolve()), "split": "train",
            "slow_anchor_id": f"recovery:{anchor}",
            "option_sequence_index": None,
            "slow_anchor_arc_m": None, "slow_anchor_remaining_arc_m": route["length_m"],
            "slow_coverage_m": 1.6, "option_coverage_truncated": False,
            "fast_window_index": index - 1, "option_replan_index": index - 1,
            "fast_window_arc_m": offset,
            "fast_interval_from_previous_m": 0.0 if index == 1 else 0.4,
            "fast_sampling_seed": None, "fast_offset_from_slow_m": offset,
            "fast_remaining_arc_m": remaining,
            "source_scene": case["scene"], "source_task_id": case["episode_id"],
            "source_point_index": None,
            "source_pose_xyyaw": [pose[0], pose[1], pose[2]],
            "instruction": case["instruction"],
            "contract": current["arguments"],
            "slow_messages": persisted_messages,
            "slow_completion": slow_completion,
            "query_tail_count": 0, "pass_rank": 1,
            "decision_filter": "actual_mismatch_and_oracle_alignment_v2",
            "query_request_id": (trace.get("query_artifact") or {}).get(
                "request_id_with_digest"),
            "reference_image": str(paths[0]), "current_image": str(image_path),
            "valid_action_steps": int(np.asarray(mask).sum()),
            "bootstrap_provenance": {
                "type": "online_recovery", "round": 1,
                "candidate_index": candidate_index, "r_index": index,
                "offset_m": offset,
                "correction_mode": audit["correction_mode"],
            },
        })
    return rows, {**audit, "accepted": True, "reason": "recovery_accept",
                  "slow_anchor_id": f"recovery:{anchor}", "samples": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--bench-url", default="http://127.0.0.1:8938")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:28899")
    parser.add_argument("--render-url", default="http://127.0.0.1:7038")
    parser.add_argument("--slow-model-name", default="Qwen3.6-35B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-recovery-anchors", type=int, default=300)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-tool-calls", type=int, default=30)
    parser.add_argument("--keep-last-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--allow-empty", action="store_true",
                        help="Smoke-only: validate rollout infrastructure without requiring an accepted anchor.")
    parser.add_argument("--action-head-as-tool-version", choices=("v2",), default="v2")
    args = parser.parse_args()
    if args.max_cases < 0 or args.max_recovery_anchors <= 0:
        parser.error("max-cases must be >=0 and max-recovery-anchors must be >0")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base_manifest = Path(args.base_manifest).resolve()
    cases = load_training_cases(base_manifest, args.seed, args.max_cases)
    dataset, tasks = task_lookup(cases)
    identity = CE.checkpoint_identity(args.checkpoint_dir)
    namespace = SimpleNamespace(
        checkpoint=identity["action_head_dir"], backbone=args.backbone,
        query_cache=args.query_cache, bench_url=args.bench_url,
        vllm_url=args.vllm_url, timeout=args.timeout, policy_seed=20260827,
        mode="instruct", max_turns=args.max_turns,
        max_tool_calls=args.max_tool_calls, keep_last_images=args.keep_last_images,
        action_head_as_tool_version="v2", capture_slow_messages=True,
        query_archive=str(output / "query_hidden"))
    evaluator = CE.TracedEvaluator(namespace)
    preloaded: set[str] = set()
    manifest_rows, audit_rows = [], []
    for case_index, case in enumerate(cases, 1):
        if len(manifest_rows) // 4 >= args.max_recovery_anchors:
            break
        rollout_path = output / "rollouts" / (
            hashlib.sha256(f"{case['scene']}:{case['episode_id']}".encode()).hexdigest()[:16]
            + ".json")
        accepted_this_case = False
        try:
            row = evaluator.rollout(case, CE.deterministic_env_seed(case))
            row = externalize_rollout(
                row, output / "rollout_context_images" / rollout_path.stem)
            atomic_json(rollout_path, row)
            calls = flatten_calls(row)
            for candidate_index in range(1, len(calls)):
                previous, current = calls[candidate_index - 1], calls[candidate_index]
                if previous.get("name") != "run_action_head":
                    continue
                samples, audit = build_candidate(
                    evaluator=evaluator, output=output, case=case,
                    task=tasks[(case["scene"], case["episode_id"])], dataset=dataset,
                    previous=previous, current=current,
                    candidate_index=candidate_index,
                    vllm_url=args.vllm_url, model=args.slow_model_name,
                    timeout=args.timeout, render_url=args.render_url,
                    preloaded=preloaded)
                audit_rows.append({**audit, "scene": case["scene"],
                                   "episode_id": case["episode_id"]})
                if samples:
                    manifest_rows.extend(samples)
                    accepted_this_case = True
                    break
            print(f"[collect] {case_index}/{len(cases)} {case['episode_id']} "
                  f"valid={row.get('evaluation_valid')} accepted={accepted_this_case} "
                  f"anchors={len(manifest_rows) // 4}", flush=True)
        except Exception as exc:
            audit_rows.append({
                "scene": case["scene"], "episode_id": case["episode_id"],
                "accepted": False, "reason": "collector_exception",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
            print(f"[collect] ERROR {case_index}/{len(cases)} {exc}", flush=True)
        # A long single-node collection should remain auditable/salvageable if
        # platform infrastructure fails before the final DONE marker.
        if accepted_this_case or case_index % 10 == 0:
            atomic_jsonl(output / "manifest.partial.jsonl", manifest_rows)
            atomic_jsonl(output / "recovery_audit.partial.jsonl", audit_rows)
    manifest_path = output / "manifest.jsonl"
    atomic_jsonl(manifest_path, manifest_rows)
    audit_path = output / "recovery_audit.jsonl"
    atomic_jsonl(audit_path, audit_rows)
    counts = {choice: sum(row.get("judge_label") == choice for row in audit_rows)
              for choice in JUDGE_CHOICES}
    correction_modes = {
        mode: sum(row.get("accepted") is True and row.get("correction_mode") == mode
                  for row in audit_rows)
        for mode in sorted({str(row.get("correction_mode")) for row in audit_rows
                            if row.get("correction_mode")})
    }
    provenance = {
        "schema": "action_head_bootstrap_recovery_v1",
        "action_head_as_tool_version": "v2", **V2.protocol_identity(),
        "bootstrap_round": 1, "base_manifest": str(base_manifest),
        "base_manifest_sha256": sha256_file(base_manifest),
        "checkpoint": identity, "selection": {
            "definition": "slow ToolCall after actual fast execution mismatch, aligned with oracle recovery",
            "judge_labels": list(JUDGE_CHOICES), "counts": counts,
            "accepted_correction_modes": correction_modes,
            "one_anchor_per_episode": True,
        },
        "oracle": {"planner": "ecm_centered_astar", "smoothing": "collision_aware",
                   "collision": "usd_rooms", "furniture": "none",
                   "robot_radius_m": ROBOT_RADIUS_M,
                   "r_offsets_m": list(OFFSETS_M)},
        "cases_attempted": len({(row.get("scene"), row.get("episode_id"))
                                for row in audit_rows}),
        "anchors_accepted": len(manifest_rows) // 4,
        "samples": len(manifest_rows),
        "model_visible_additions": [],
    }
    atomic_json(output / "provenance.json", provenance)
    (output / "DONE").write_text(json.dumps({
        "ok": True, "anchors": len(manifest_rows) // 4,
        "samples": len(manifest_rows), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")
    }) + "\n", encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False, indent=2), flush=True)
    if not manifest_rows and not args.allow_empty:
        raise SystemExit("no valid recovery samples collected")


if __name__ == "__main__":
    main()
