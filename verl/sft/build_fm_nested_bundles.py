#!/usr/bin/env python3
"""Materialize nested slow/fast FM-head SFT bundles.

One slow decision/query latent is generated at each outer anchor and reused by
all fast windows inside its coverage interval.  Candidate decisions pass hard
schema/safety checks and can additionally be checked by a VLM against expert
future observations.  The latter is an offline teacher filter, not evidence
that the online slow policy is always correct.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.request
import zipfile
from collections import OrderedDict, defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from numpy.lib import format as npy_format

from action_head import contract as A
from action_head import protocol as ACTION_HEAD_PROTOCOL
from action_head.policy import Qwen35VisualTokenEncoder
from action_head.query_provider import VllmQueryStateProvider
from action_head.vllm_bridge import (
    DEFAULT_QUERY_COUNT,
    request_four_queries,
    strip_query_tail,
)


def post_json(url, payload, timeout):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def save_npz(path, *, compresslevel=1, **arrays):
    """Write a NumPy-compatible NPZ with explicit fast DEFLATE level."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED,
                compresslevel=int(compresslevel), allowZip64=True) as archive:
            for name, value in arrays.items():
                with archive.open(name + ".npy", "w", force_zip64=True) as handle:
                    npy_format.write_array(
                        handle, np.asanyarray(value), allow_pickle=False)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def instruction_from_record(record):
    text = record["conversations"][0]["value"]
    marker = " anonymous Instruction: "
    if marker not in text:
        raise ValueError("source record has no anonymous instruction marker")
    return text.split(marker, 1)[1].split("\nPredict", 1)[0].strip()


def resolve_images(row, frame_dir):
    return [str((frame_dir / relative).resolve()) for relative in row["images"]]


def parse_tool_call_text(text, *, action_head_as_tool_version="v1"):
    version = ACTION_HEAD_PROTOCOL.normalize_version(action_head_as_tool_version)
    protocol = ACTION_HEAD_PROTOCOL.protocol_module(version)
    match = A.TOOL_CALL_RE.search(text)
    if match is None:
        return None, text, "missing_tool_call"
    try:
        call = json.loads(A.strip_code_fence(match.group(1)))
    except json.JSONDecodeError as exc:
        return None, text, f"invalid_json:{exc.msg}"
    name = call.get("name")
    if name == "stop" and protocol is not None:
        arguments, error = protocol.validate_stop_arguments(call.get("arguments"))
        if arguments is None:
            return None, text, f"invalid_stop_contract:{error}"
        return {"name": name, "arguments": arguments}, text, "hard_checks_passed"
    if name != "run_action_head":
        return None, text, f"unsupported_tool:{name}"
    contract, error = A.validate_action_head_arguments(call.get("arguments"))
    if contract is None:
        return None, text, f"invalid_action_head_contract:{error}"
    flattened = " ".join([
        contract["target"], contract["where"], *contract["plan"]]).lower()
    forbidden = ["move_forward", "move forward 0.", "turn 15", "world coordinate",
                 "gps", "stop immediately"]
    # Preserve the v1/v2 teacher filter exactly. v3 explicitly permits freely
    # worded backing/turn-around options, which are semantic plans rather than
    # primitive metric actions.
    if version not in ("v3", "v4.3", "v4.11", "v6"):
        forbidden.extend(("go backward", "backtrack"))
    hit = next((value for value in forbidden if value in flattened), None)
    if hit:
        return None, text, f"forbidden_low_level_or_reverse:{hit}"
    return {"name": name, "arguments": contract}, text, "hard_checks_passed"


def parse_tool_call(response, *, action_head_as_tool_version="v1"):
    choice = response["choices"][0]["message"]
    text = (choice.get("reasoning_content") or "") + (choice.get("content") or "")
    return parse_tool_call_text(
        text, action_head_as_tool_version=action_head_as_tool_version)


def correct_terminal_stop(item, success_radius_m=1.0):
    """Conservatively accept only golden anchors provably inside success radius."""
    remaining = float(item.get(
        "slow_anchor_remaining_arc_m",
        (item["slow_anchor_row"].get("meta_data") or {}).get(
            "remaining_arc_m", float("inf"))))
    return remaining <= float(success_radius_m) + 1e-6


def slow_decision_row(*, anchor_id, item, instruction, messages, completion,
                      decision, pass_rank, decision_filter, protocol,
                      source="generated"):
    meta = item["slow_anchor_row"].get("meta_data") or {}
    return {
        "schema": protocol.SLOW_DECISION_SCHEMA,
        **protocol.protocol_identity(),
        "split": item["split"],
        "slow_anchor_id": anchor_id,
        "option_sequence_index": item.get("option_sequence_index"),
        "source_scene": meta["source_scene"],
        "source_task_id": meta["source_task_id"],
        "source_point_index": meta["source_point_index"],
        "slow_anchor_remaining_arc_m": item.get(
            "slow_anchor_remaining_arc_m", meta.get("remaining_arc_m")),
        "instruction": instruction,
        "slow_messages": messages,
        "slow_completion": completion,
        "decision_type": decision["name"],
        "arguments": decision["arguments"],
        "pass_rank": pass_rank,
        "decision_filter": decision_filter,
        "source": source,
    }


def build_messages(instruction, image_paths, *,
                   action_head_as_tool_version="v1", history=None):
    version = ACTION_HEAD_PROTOCOL.normalize_version(action_head_as_tool_version)
    protocol = ACTION_HEAD_PROTOCOL.protocol_module(version)
    if protocol is not None:
        source = (protocol.initial_messages(instruction, image_paths)
                  if history is None else history)
        return protocol.strip_context_images(source, keep_slow=8)
    example = (
        '<tool_call>{"name":"run_action_head","arguments":'
        '{"target":"the visible destination","where":"ahead along the open route",'
        '"plan":["align with the open route","approach the destination"]}}'
        '</tool_call>'
    )
    return [
        {"role": "system", "content": (
            "You are the slow navigation system. From the instruction and RGB history, "
            "choose one coherent local option for the next fixed Action Head execution cycle. "
            "Return exactly one run_action_head tool call in the XML-wrapped JSON format "
            "shown below and no other text. Arguments must contain target "
            "and where strings plus a plan array of 1 to 3 semantic stages. Do not emit "
            "primitive actions, metric coordinates, GPS, or a stop decision.\nValid example:\n"
            + example)},
        {"role": "user", "content": [
            *[{"type": "image_url", "image_url": {"url": "file://" + path}}
              for path in image_paths],
            {"type": "text", "text": instruction},
        ]},
    ]


def _source_pose(row):
    pose = (row.get("meta_data") or {}).get("source_pose_xyyaw")
    if not isinstance(pose, (list, tuple)) or len(pose) != 3:
        raise ValueError("v2 golden row is missing source_pose_xyyaw")
    return tuple(float(value) for value in pose)


def _option_pose(start, end):
    x0, y0, yaw0 = start
    x1, y1, yaw1 = end
    dx, dy = x1 - x0, y1 - y0
    x_m = math.cos(yaw0) * dx + math.sin(yaw0) * dy
    y_m = -math.sin(yaw0) * dx + math.cos(yaw0) * dy
    yaw_deg = math.degrees((yaw1 - yaw0 + math.pi) % (2 * math.pi) - math.pi)
    return {"x_m": round(x_m, 3), "y_m": round(y_m, 3),
            "yaw_deg": round(yaw_deg, 1)}


def golden_tool_result(anchor_id, items, frame_dir, *, protocol, contract):
    """Build causal, non-oracle teacher execution feedback for one option."""
    ordered = sorted(items, key=lambda item: item["fast_window_arc_m"])
    anchor_row = ordered[0]["slow_anchor_row"]
    start_pose = _source_pose(anchor_row)
    option_end = ordered[-1]["option_end_row"]
    option_pose = _option_pose(start_pose, _source_pose(option_end))
    frames, steps_used = [], 0
    for chunk, item in enumerate(ordered, 1):
        start_row, end_row = item["fast_row"], item["chunk_end_row"]
        start_arc = float(item["fast_window_arc_m"])
        end_arc = float((end_row.get("meta_data") or {}).get(
            "source_arc_m", item["option_end_arc_m"]))
        path_m = max(0.0, end_arc - start_arc)
        start_index = int((start_row.get("meta_data") or {})["source_point_index"])
        end_index = int((end_row.get("meta_data") or {})["source_point_index"])
        steps_used += max(0, end_index - start_index)
        frames.append({
            "chunk": chunk,
            "actual_end_pose": _option_pose(start_pose, _source_pose(end_row)),
            "path_m": round(path_m, 3),
            "blocked": False,
            "collision": False,
            "no_progress": path_m < 1e-3,
            "image_path": resolve_images(end_row, frame_dir)[-1],
        })
    path_m = max(0.0, float(ordered[-1]["option_end_arc_m"])
                 - float(ordered[0]["slow_anchor_arc_m"]))
    summary = {
        "schema": protocol.TOOL_RESULT_SCHEMA,
        "tool": "run_action_head",
        "option_id": anchor_id,
        "ok": True,
        "return_reason": ("fast_step_budget_reached" if len(frames) >= 4
                          else "episode_ended"),
        "completion": "unknown",
        "chunks": len(frames),
        "path_m": round(path_m, 3),
        "net_m": round(math.hypot(option_pose["x_m"], option_pose["y_m"]), 3),
        "blocked": False,
        "collision": False,
        "steps_used": steps_used,
    }
    if protocol.ACTION_HEAD_AS_TOOL_VERSION in ("v3", "v4.3", "v4.11", "v6"):
        summary["direction_alignment"] = protocol.direction_alignment(
            contract, option_pose)
    return summary, frames


def golden_tool_result_v2(anchor_id, items, frame_dir):
    """Compatibility wrapper retained for v2 tests and external callers."""
    return golden_tool_result(
        anchor_id, items, frame_dir,
        protocol=ACTION_HEAD_PROTOCOL.protocol_module("v2"), contract={})


def vlm_validate(url, model, timeout, instruction, contract, evidence_images):
    prompt = (
        "Judge whether the candidate local navigation option is consistent with the "
        "instruction and the expert future RGB observations. Reject wrong targets, wrong "
        "direction, premature stopping, reversal, or plans unsupported by the route. "
        "Return exactly PASS or FAIL.\nInstruction: " + instruction +
        "\nCandidate: " + json.dumps(contract, ensure_ascii=False))
    messages = [
        {"role": "system", "content": "Return exactly PASS or FAIL."},
        {"role": "user", "content": [
            *[{"type": "image_url", "image_url": {"url": "file://" + path}}
              for path in evidence_images],
            {"type": "text", "text": prompt},
        ]},
    ]
    payload = {"model": model, "messages": messages, "temperature": 0.0,
               "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False},
               "structured_outputs": {"choice": ["PASS", "FAIL"]}}
    response = post_json(url.rstrip("/") + "/v1/chat/completions", payload, timeout)
    message = response["choices"][0]["message"]
    answer = ((message.get("reasoning_content") or "") +
              (message.get("content") or "")).strip().upper()
    return bool(re.search(r"\bPASS\b", answer)), answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nested-plan", required=True)
    parser.add_argument("--frame-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone", required=True,
                        help="Fast visual backbone, currently Qwen3.5-4B.")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:28899")
    parser.add_argument("--model", default="Qwen3.6-35B")
    parser.add_argument("--slow-hidden-size", type=int, default=2048,
                        help="Four-query hidden width of the slow model, not the fast visual backbone.")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--visual-devices", default="",
        help="Comma-separated visual worker devices. Empty keeps the legacy "
             "single worker on --device; e.g. cuda:0,cuda:1 enables two "
             "independent visual consumers while vLLM continues serving.")
    parser.add_argument("--pass-k", type=int, default=4)
    parser.add_argument("--decision-filter", choices=("hard", "vlm", "none"), default="vlm")
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--beam-seed", type=int, default=20260922)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-anchors", type=int, default=0)
    parser.add_argument("--task-shard-index", type=int, default=0,
                        help="Stable trajectory partition handled by this process.")
    parser.add_argument("--task-shard-count", type=int, default=1,
                        help="Number of independent trajectory partitions.")
    parser.add_argument("--visual-cache-size", type=int, default=128)
    parser.add_argument(
        "--visual-prefetch", type=int, default=4,
        help="Maximum accepted anchors queued to visual workers while later "
             "vLLM requests remain in flight.")
    parser.add_argument(
        "--pipeline-mode", choices=("legacy", "stream"), default="stream",
        help="legacy waits for the full option layer before post-processing; "
             "stream overlaps visual/npz work with later vLLM requests.")
    parser.add_argument("--npz-compress-level", type=int, default=1,
                        help="DEFLATE level 0-9; level 1 is much faster than "
                             "numpy.savez_compressed's implicit level 6.")
    parser.add_argument("--action-head-as-tool-version", choices=("v1", "v2", "v3", "v4.3", "v4.11", "v6"),
                        default="v1")
    parser.add_argument("--render-url", default="http://127.0.0.1:7038")
    parser.add_argument("--stop-success-radius-m", type=float, default=1.0)
    parser.add_argument(
        "--resume", dest="resume", action="store_true", default=None,
        help="Resume from output/layer_checkpoint.json if present. Default "
             "auto-detects: resume when a checkpoint exists, else fresh start "
             "(clears samples + jsonl). Pass --no-resume to force a fresh run.")
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Force a fresh start: delete existing samples/manifest/audit/"
             "decisions in the output dir and ignore any checkpoint.")
    args = parser.parse_args()
    if args.action_head_as_tool_version == "v6":
        if args.beam_width <= 0:
            raise ValueError("--beam-width must be positive")
        args.decision_filter = "none"
    if args.pass_k <= 0:
        raise ValueError("--pass-k must be positive")
    if args.stop_success_radius_m <= 0:
        raise ValueError("--stop-success-radius-m must be positive")
    if args.visual_prefetch <= 0:
        raise ValueError("--visual-prefetch must be positive")
    if (args.task_shard_count <= 0 or args.task_shard_index < 0
            or args.task_shard_index >= args.task_shard_count):
        raise ValueError("task shard must satisfy 0 <= index < count")
    if not 0 <= args.npz_compress_level <= 9:
        raise ValueError("--npz-compress-level must be in [0,9]")

    plan_path = Path(args.nested_plan).resolve()
    frame_dir = Path(args.frame_dir).resolve()
    plan_text = plan_path.read_text()
    plan_sha256 = hashlib.sha256(plan_text.encode()).hexdigest()
    plan = [json.loads(line) for line in plan_text.splitlines() if line.strip()]
    del plan_text
    requested_version = ACTION_HEAD_PROTOCOL.normalize_version(
        args.action_head_as_tool_version)
    protocol = ACTION_HEAD_PROTOCOL.protocol_module(requested_version)
    if protocol is not None:
        expected_plan_schema = f"fm_nested_sft_plan_{requested_version}"
        mismatches = [
            (row.get("slow_anchor_id"), row.get("action_head_as_tool_version"),
             row.get("schema"))
            for row in plan
            if row.get("action_head_as_tool_version") != requested_version
            or row.get("schema") != expected_plan_schema
        ]
        if mismatches:
            raise ValueError(
                f"{requested_version} bundle construction requires an "
                f"{expected_plan_schema} input; "
                f"first mismatches: {mismatches[:3]}")
    grouped = defaultdict(list)
    for row in plan:
        grouped[row["slow_anchor_id"]].append(row)
    groups = sorted(grouped.items())
    if args.max_anchors:
        groups = groups[:args.max_anchors]
    if not groups:
        raise RuntimeError("nested plan contains no anchors")

    output = Path(args.output_dir)
    samples_dir = output / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    visual_devices = [value.strip() for value in args.visual_devices.split(",")
                      if value.strip()] or [args.device]
    if len(set(visual_devices)) != len(visual_devices):
        raise ValueError("--visual-devices must not contain duplicates")
    visions = [Qwen35VisualTokenEncoder.from_pretrained(
        args.backbone, device=device, dtype=torch.bfloat16)
        for device in visual_devices]
    if args.slow_hidden_size <= 0:
        raise ValueError("--slow-hidden-size must be positive")
    provider = VllmQueryStateProvider(
        args.cache_root, hidden_size=args.slow_hidden_size,
        cleanup=True, wait_seconds=30.0)

    class _VisualRunner:
        """One single-threaded encoder/cache bound to one physical GPU."""

        def __init__(self, encoder):
            self.encoder = encoder
            self.cache = OrderedDict()

        def encode_many(self, paths):
            started = time.perf_counter()
            values = [None] * len(paths)
            hits = 0
            missing_paths = []
            missing_indices = []
            missing_rgbs = []
            for index, path in enumerate(paths):
                cached = self.cache.pop(path, None)
                if cached is not None:
                    self.cache[path] = cached
                    values[index] = cached
                    hits += 1
                    continue
                missing_paths.append(path)
                missing_indices.append(index)
                missing_rgbs.append(np.asarray(Image.open(path).convert("RGB")))
            if missing_rgbs:
                encoded = self.encoder.encode_batch(missing_rgbs)
                if len(encoded) != len(missing_rgbs):
                    raise RuntimeError("visual batch returned the wrong number of frames")
                encoded = [value.float().cpu().numpy() for value in encoded]
            else:
                encoded = []
            for index, path, value in zip(missing_indices, missing_paths, encoded):
                self.cache[path] = value
                while len(self.cache) > args.visual_cache_size:
                    self.cache.popitem(last=False)
                values[index] = value
            return values, time.perf_counter() - started, hits

    visual_runners = [_VisualRunner(vision) for vision in visions]

    version = requested_version
    print(json.dumps({
        "action_head_as_tool_version": version,
        **(protocol.protocol_identity() if protocol is not None else {}),
    }, ensure_ascii=False), flush=True)
    # ---- checkpoint resume (per option_idx layer) -------------------------
    # The build is killed often (Lingjun SM% patrol, OOM, preemption). Without
    # a checkpoint the whole shard restarts from anchor 0: manifest/audit are
    # only written at the very end in the original code, so a mid-run kill
    # orphans every npz it produced (no manifest row points at them). We now
    # append manifest/audit/decisions per anchor and write a layer checkpoint
    # after every option_idx layer; on restart we truncate outputs back to the
    # last fully-completed layer and resume. histories are small (file://
    # image paths, not base64), so the checkpoint stays MB-scale even with
    # thousands of tasks. tk tuples are joined with \x1f for JSON keys.
    manifest_path = output / "manifest.jsonl"
    audit_path = output / "anchor_audit.jsonl"
    decisions_path = output / "slow_decisions.jsonl"
    checkpoint_path = output / "layer_checkpoint.json"

    resume = args.resume
    if resume is None:
        resume = checkpoint_path.exists()
    if resume and not checkpoint_path.exists():
        print("[build] --resume requested but no checkpoint; fresh start", flush=True)
        resume = False
    beams, map_states = {}, {}
    beam_contract = {"width": args.beam_width, "pass_k": args.pass_k,
                     "sampling": "fixed_anchor_budget_balanced_parents_v1",
                     "requests_per_anchor": args.beam_width * args.pass_k,
                     "seed": args.beam_seed, "selection": "parent_stratified_random",
                     "direction_filter": False, "turn_policy": "reject_unexecuted"}
    if resume:
        ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if version == "v6":
            protocol.assert_protocol_identity(ckpt)
            if ckpt.get("beam_contract") != beam_contract:
                raise ValueError("v6 beam settings differ from the checkpoint")
            if ckpt.get("nested_plan_sha256") != plan_sha256:
                raise ValueError("v6 resume plan changed")
            beams = {tuple(json.loads(k)): [tuple(x) for x in v]
                     for k, v in ckpt["beams"].items()}
            map_states = {tuple(json.loads(k)): v for k, v in ckpt["map_states"].items()}
        start_option_idx = int(ckpt["option_idx_completed"]) + 1
        sample_index = int(ckpt["sample_index"])
        group_index_resume = int(ckpt["group_index"])
        histories = {tuple(json.loads(k)): v
                     for k, v in ckpt.get("histories", {}).items()}
        last_option_end = {tuple(json.loads(k)): v
                           for k, v in ckpt.get("last_option_end", {}).items()}
        broken_tasks = {tuple(json.loads(s)) for s in ckpt.get("broken_tasks", [])}
        terminated_tasks = {tuple(json.loads(s))
                            for s in ckpt.get("terminated_tasks", [])}
        # Truncate incremental outputs to the checkpoint boundary so a
        # partial-layer kill (some manifest rows / npz written, layer not
        # checkpointed yet) is discarded before resuming.
        for path, n in ((manifest_path, int(ckpt["manifest_count"])),
                        (audit_path, int(ckpt["audit_count"])),
                        (decisions_path, int(ckpt["decisions_count"]))):
            if path.exists() and n >= 0:
                lines = path.read_text(encoding="utf-8").splitlines()
                if len(lines) > n:
                    path.write_text("\n".join(lines[:n]) + ("\n" if n else ""),
                                    encoding="utf-8")
        removed = 0
        for p in samples_dir.glob("sample_*.npz"):
            try:
                idx = int(p.stem.split("_")[-1])
            except ValueError:
                continue
            if idx >= sample_index:
                p.unlink(missing_ok=True)
                removed += 1
        print(f"[build] RESUME from option_idx={start_option_idx} "
              f"sample_index={sample_index} group_index={group_index_resume} "
              f"histories={len(histories)} broken={len(broken_tasks)} "
              f"terminated={len(terminated_tasks)} removed_orphan_npz={removed}",
              flush=True)
        manifest_fp = manifest_path.open("a", encoding="utf-8")
        audit_fp = audit_path.open("a", encoding="utf-8")
        decisions_fp = (decisions_path.open("a", encoding="utf-8")
                        if protocol is not None else None)
    else:
        # Fresh start: clear prior samples + manifests so stale output from a
        # pre-checkpoint build (v1-style orphan npz) does not contaminate.
        for p in samples_dir.glob("sample_*.npz"):
            p.unlink(missing_ok=True)
        start_option_idx = 0
        sample_index = 0
        group_index_resume = 0
        histories, last_option_end = {}, {}
        broken_tasks, terminated_tasks = set(), set()
        manifest_fp = manifest_path.open("w", encoding="utf-8")
        audit_fp = audit_path.open("w", encoding="utf-8")
        decisions_fp = (decisions_path.open("w", encoding="utf-8")
                        if protocol is not None else None)
        print("[build] fresh start (cleared samples + manifest/audit/decisions)",
              flush=True)
    target_horizons = set(ckpt.get('target_horizons', [])) if resume else set()
    if resume and not target_horizons and int(ckpt['manifest_count']):
        with manifest_path.open() as stream:
            first_row = json.loads(next(line for line in stream if line.strip()))
        with np.load(first_row['sample']) as sample:
            target_horizons.add(int(sample['target_actions'].shape[0]))

    # Line counts for the three append-mode jsonl files, tracked incrementally
    # so the layer checkpoint can record exact truncation boundaries without
    # re-scanning MB-scale files on every layer.
    if resume:
        n_manifest = int(ckpt["manifest_count"])
        n_audit = int(ckpt["audit_count"])
        n_decisions = int(ckpt["decisions_count"])
    else:
        n_manifest = n_audit = n_decisions = 0
    # Per-stage processing profiling: cumulative seconds spent in each stage
    # of the accepted-anchor inner loop. Printed every 100 accepts so a
    # resumed run reveals the bottleneck (provider IO vs visual vs npz).
    prof = {"provider": 0.0, "visual": 0.0, "npz": 0.0,
            "manifest": 0.0, "golden": 0.0, "n": 0}
    # v3 batched pipeline: per-option-layer + pass@N concurrent HTTP to the
    # vLLM server (continuous batching + prefix cache) + direction-alignment
    # hard check (no vllm self-judge). Same-trajectory options stay causal
    # (option N context depends on option N-1 ToolCall+ToolResult), so we batch
    # across trajectories within one option layer.
    PASS = max(1, args.pass_k)
    from concurrent.futures import ThreadPoolExecutor
    visual_executors = [ThreadPoolExecutor(max_workers=1,
                                           thread_name_prefix=f"visual-{index}")
                        for index in range(len(visual_runners))]
    visual_round_robin = 0
    pending_visual = deque()
    by_task = {}
    for anchor_id, items in groups:
        items.sort(key=lambda it: it["fast_window_arc_m"])
        am = items[0]["slow_anchor_row"].get("meta_data") or {}
        tk = (am.get("source_scene"), am.get("source_task_id"))
        by_task.setdefault(tk, []).append((anchor_id, items))
    if args.task_shard_count > 1:
        def _task_partition(task_key):
            payload = json.dumps(task_key, ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8")
            return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") \
                % args.task_shard_count
        by_task = {task_key: anchors for task_key, anchors in by_task.items()
                   if _task_partition(task_key) == args.task_shard_index}
        print(f"[build] task partition {args.task_shard_index}/"
              f"{args.task_shard_count}: tasks={len(by_task)}", flush=True)
        if not by_task:
            raise RuntimeError("selected task partition contains no trajectories")
    for _tk in by_task:
        by_task[_tk].sort(key=lambda ai: ai[1][0]["slow_anchor_arc_m"])

    def _finish_visual(record):
        """Commit one accepted anchor after its asynchronous visual encoding."""
        nonlocal sample_index, n_manifest, n_audit
        values, visual_seconds, cache_hits = record["future"].result()
        prof["visual"] += visual_seconds
        prof["visual_cache_hits"] = prof.get("visual_cache_hits", 0) + cache_hits
        reference, currents = values[0], values[1:]
        items = record["items"]
        if len(currents) != len(items):
            raise RuntimeError("visual worker returned the wrong number of current frames")
        for item, current, current_path in zip(items, currents,
                                               record["current_paths"]):
            fast_row = item["fast_row"]
            target = np.asarray(fast_row["meta_data"]["gt_waypoints"], np.float32)
            mask = np.asarray(fast_row["meta_data"]["waypoint_mask"], np.bool_)
            target_horizons.add(int(target.shape[0]))
            sample_path = samples_dir / f"sample_{sample_index:07d}.npz"
            _tn = time.perf_counter()
            save_npz(
                sample_path,
                compresslevel=args.npz_compress_level,
                query_hidden=record["bundle"].query_hidden.astype(np.float32),
                reference_visual_tokens=reference.astype(np.float32),
                current_visual_tokens=current.astype(np.float32),
                target_actions=target, action_mask=mask)
            prof["npz"] += time.perf_counter() - _tn
            meta = fast_row["meta_data"]
            _tm = time.perf_counter()
            manifest_fp.write(json.dumps({
                "schema": (protocol.BUNDLE_SCHEMA if protocol is not None
                           else "fm_nested_bundle_v2"),
                **(protocol.protocol_identity() if protocol is not None else {}),
                "sample": str(sample_path.resolve()),
                "split": item["split"], "slow_anchor_id": record["anchor_id"],
                **({key: record[key] for key in ("golden_anchor_id", "branch_id",
                    "parent_branch_id", "branch_count", "trajectory_map_path")}
                   if version == "v6" else {}),
                "option_sequence_index": item.get("option_sequence_index"),
                "slow_anchor_arc_m": item["slow_anchor_arc_m"],
                "slow_anchor_remaining_arc_m": item.get(
                    "slow_anchor_remaining_arc_m",
                    record["anchor_row"]["meta_data"].get("remaining_arc_m")),
                "slow_coverage_m": item.get("slow_coverage_m"),
                "option_coverage_truncated": item.get("option_coverage_truncated"),
                "fast_window_index": item["fast_window_index"],
                "option_replan_index": item["fast_window_index"],
                "fast_window_arc_m": item["fast_window_arc_m"],
                "fast_interval_from_previous_m": item.get(
                    "fast_interval_from_previous_m"),
                "fast_sampling_seed": item.get("fast_sampling_seed"),
                "fast_offset_from_slow_m": item["fast_offset_from_slow_m"],
                "fast_remaining_arc_m": item.get(
                    "fast_remaining_arc_m", meta.get("remaining_arc_m")),
                "source_scene": meta["source_scene"],
                "source_task_id": meta["source_task_id"],
                "source_point_index": meta["source_point_index"],
                "source_pose_xyyaw": meta.get("source_pose_xyyaw"),
                "instruction": record["instruction"],
                "contract": record["contract"],
                "slow_messages": record["messages"],
                "slow_completion": record["clean_text"],
                "query_tail_count": record["query_tail_count"],
                "pass_rank": record["rank"],
                "decision_filter": args.decision_filter,
                "query_request_id": record["bundle"].request_id,
                "reference_image": record["anchor_images"][-1],
                "current_image": current_path,
                "valid_action_steps": int(mask.sum()),
            }, ensure_ascii=False) + "\n")
            prof["manifest"] += time.perf_counter() - _tm
            n_manifest += 1
            sample_index += 1
        if protocol is not None:
            conversation = (copy.deepcopy(record["history"])
                            if record["history"] is not None else
                            protocol.initial_messages(record["instruction"],
                                                      record["anchor_images"]))
            _tg = time.perf_counter()
            summary, frames = golden_tool_result(
                record["anchor_id"], items, frame_dir, protocol=protocol,
                contract=record["contract"])
            prof["golden"] += time.perf_counter() - _tg
            protocol.append_tool_exchange(
                conversation,
                completion=(record["normalized_completion"] if version in ("v3", "v4.3", "v4.11", "v6")
                            else record["clean_text"]),
                summary=summary, frames=frames)
            histories[record["tk"]] = conversation
            last_option_end[record["tk"]] = frames[-1]["image_path"]
        if version != "v6":
            audit_fp.write(json.dumps(
                {"slow_anchor_id": record["anchor_id"], "accepted": True,
                 "accepted_rank": record["rank"], "contract": record["contract"],
                 "attempts": record["attempts"]}, ensure_ascii=False) + "\n")
            audit_fp.flush()
            n_audit += 1
        prof["n"] += 1
        if prof["n"] % 100 == 0:
            _tot = (prof["provider"] + prof["visual"] + prof["npz"]
                    + prof["manifest"] + prof["golden"])
            print(f"[prof] n={prof['n']} tot={_tot:.1f}s "
                  f"provider={prof['provider']:.1f} visual_worker={prof['visual']:.1f} "
                  f"visual_cache_hits={prof['visual_cache_hits']} "
                  f"npz={prof['npz']:.1f} manifest={prof['manifest']:.1f} "
                  f"golden={prof['golden']:.1f} "
                  f"({_tot/max(prof['n'],1):.2f}s/anchor)", flush=True)
        print(f"[{record['group_index']}/{len(groups)}] accept "
              f"{record['anchor_id']} rank={record['rank']} windows={len(items)}",
              flush=True)

    def queue_record(record):
        nonlocal visual_round_robin
        started = time.perf_counter()
        record['bundle'] = provider(
            request={'messages': record['messages'], 'instruction': record['instruction']},
            response={'text': record['text'], 'usage': record['response'].get('usage') or {},
                      'query_request_id': record['request_id']}, contract=record['contract'])
        prof['provider'] += time.perf_counter() - started
        paths = [resolve_images(item['fast_row'], frame_dir)[-1] for item in record['items']]
        index = visual_round_robin % len(visual_runners)
        visual_round_robin += 1
        record['current_paths'] = paths
        record['future'] = visual_executors[index].submit(
            visual_runners[index].encode_many, [record['anchor_images'][-1], *paths])
        pending_visual.append(record)
        if len(pending_visual) >= max(args.visual_prefetch, len(visual_runners)):
            _finish_visual(pending_visual.popleft())

    max_opts = max((len(v) for v in by_task.values()), default=0)
    group_index = group_index_resume
    for option_idx in range(start_option_idx, max_opts):
        if version == "v6":
            from sft.golden_context_beam import expand_layer
            next_beams, next_maps = {}, {}
            layer_counts = {}
            packets = expand_layer(
                legacy=sys.modules[__name__], args=args, by_task=by_task,
                option_idx=option_idx, frame_dir=frame_dir, output=output,
                histories=histories, last_option_end=last_option_end,
                beams=beams, map_states=map_states, broken_tasks=broken_tasks,
                terminated_tasks=terminated_tasks)
            for records, audits, terminals, packet_beams, packet_maps in packets:
                next_beams.update(packet_beams)
                next_maps.update(packet_maps)
                for audit in audits:
                    for key, count in audit['counts'].items():
                        layer_counts[key] = layer_counts.get(key, 0) + count
                    audit_fp.write(json.dumps(audit, ensure_ascii=False) + "\n")
                    n_audit += 1
                for terminal in terminals:
                    decisions_fp.write(json.dumps(terminal, ensure_ascii=False) + "\n")
                    n_decisions += 1
                for record in records:
                    group_index += 1
                    record['group_index'] = group_index
                    record['text'] = record['clean_text']  # query request identity owns the latent
                    decision_row = slow_decision_row(
                        anchor_id=record['anchor_id'], item=record['items'][0],
                        instruction=record['instruction'], messages=record['messages'],
                        completion=record['normalized_completion'], decision=record['decision'],
                        pass_rank=record['rank'], protocol=protocol, decision_filter='none')
                    decision_row.update({key: record[key] for key in
                        ('golden_anchor_id', 'branch_id', 'parent_branch_id')})
                    decisions_fp.write(json.dumps(decision_row, ensure_ascii=False) + "\n")
                    n_decisions += 1
                    queue_record(record)
            while pending_visual:
                _finish_visual(pending_visual.popleft())
            beams = next_beams
            map_states = next_maps
            live_keys = {key for branches in beams.values() for key in branches}
            histories = {key: value for key, value in histories.items() if key in live_keys}
            last_option_end = {key: value for key, value in last_option_end.items() if key in live_keys}
            print(f"[beam] layer={option_idx} tasks={len(beams)} branches={len(live_keys)} "
                  f"counts={json.dumps(layer_counts, sort_keys=True)}", flush=True)
        else:
            active = []
            for tk, anchors in by_task.items():
                if option_idx >= len(anchors) or tk in broken_tasks or tk in terminated_tasks:
                    continue
                anchor_id, items = anchors[option_idx]
                anchor_row = items[0]["slow_anchor_row"]
                anchor_images = resolve_images(anchor_row, frame_dir)
                if protocol is not None and tk in last_option_end:
                    if Path(last_option_end[tk]).resolve() != Path(anchor_images[-1]).resolve():
                        raise ValueError(
                            f"{version} causal chain discontinuity for {anchor_id}: "
                            f"preceding {last_option_end[tk]} != current {anchor_images[-1]}")
                active.append((tk, anchor_id, items, instruction_from_record(anchor_row),
                               anchor_images, anchor_row,
                               histories.get(tk) if protocol is not None else None))
            if not active:
                continue
            jobs = []
            for (tk, anchor_id, items, instruction, anchor_images, anchor_row, history) in active:
                messages = build_messages(instruction, anchor_images,
                                           action_head_as_tool_version=version, history=history)
                for k in range(PASS):
                    payload = {"model": args.model, "messages": messages,
                               "temperature": args.temperature, "top_p": 0.95,
                               # vLLM counts the four internal query carriers in
                               # max_tokens. Preserve the 256-token semantic tool
                               # budget and reserve room for the complete tail.
                               "max_tokens": 256 + DEFAULT_QUERY_COUNT,
                               "chat_template_kwargs": {"enable_thinking": False}}
                    rid = request_four_queries(payload)
                    jobs.append((tk, k, rid, payload, messages))
            def _call(j):
                tk, k, rid, payload, messages = j
                try:
                    resp = post_json(args.vllm_url.rstrip("/") + "/v1/chat/completions",
                                     payload, args.timeout)
                    return (tk, k, rid, messages, resp, None, time.perf_counter())
                except Exception as exc:
                    return (tk, k, rid, messages, None,
                            f"{type(exc).__name__}: {exc}", time.perf_counter())
            # `jobs` is grouped by anchor, PASS adjacent requests per anchor.
            # ThreadPoolExecutor.map yields in that same order but keeps later HTTP
            # calls running in worker threads. Consume one anchor at a time instead
            # of materializing list(...): while visual/npz handles an early anchor,
            # vLLM continues serving the remainder of this option layer.
            layer_started = time.perf_counter()
            http_executor = ThreadPoolExecutor(
                max_workers=min(len(jobs), max(1, PASS * 8)),
                thread_name_prefix="slow-vllm")
            mapped_results = http_executor.map(_call, jobs)
            result_iter = iter(list(mapped_results) if args.pipeline_mode == "legacy"
                               else mapped_results)
            last_http_finished = layer_started
            for (tk, anchor_id, items, instruction, anchor_images, anchor_row, history) in active:
                group_index += 1
                tr = sorted([next(result_iter) for _ in range(PASS)], key=lambda r: r[1])
                last_http_finished = max(last_http_finished,
                                         *(result[6] for result in tr))
                attempts = []
                chosen = None
                for (tk2, k, rid, messages, resp, err, _finished_at) in tr:
                    rank = k + 1
                    if err or resp is None:
                        attempts.append({"rank": rank, "request_id": rid, "decision": None,
                                         "contract": None, "hard_reason": "request_error",
                                         "judge": "not_run", "accepted": False, "text": "",
                                         "error": err or "no_response"})
                        continue
                    decision, text, hard_reason = parse_tool_call(
                        resp, action_head_as_tool_version=version)
                    passed, judge = decision is not None, "not_run"
                    if passed:
                        try:
                            strip_query_tail(text)
                        except ValueError as exc:
                            attempts.append({"rank": rank, "request_id": rid,
                                             "decision": decision, "contract": None,
                                             "hard_reason": f"invalid_query_tail:{exc}",
                                             "judge": "not_run", "accepted": False,
                                             "text": text})
                            continue
                    if passed and decision["name"] == "stop":
                        passed = correct_terminal_stop(items[0], args.stop_success_radius_m)
                        judge = ("golden_success_radius_pass" if passed else
                                 "golden_success_radius_fail")
                        if not passed:
                            hard_reason = "premature_stop"
                    contract = (decision["arguments"] if decision is not None and
                                decision["name"] == "run_action_head" else None)
                    if (passed and protocol is not None
                            and protocol.ACTION_HEAD_AS_TOOL_VERSION in ("v3", "v4.3", "v4.11", "v6")
                            and decision["name"] == "run_action_head" and contract is not None):
                        ordered = sorted(items, key=lambda it: it["fast_window_arc_m"])
                        option_pose = _option_pose(
                            _source_pose(ordered[0]["slow_anchor_row"]),
                            _source_pose(ordered[-1]["option_end_row"]))
                        align = protocol.direction_alignment(contract, option_pose)
                        if align.get("match") is False:
                            attempts.append({"rank": rank, "request_id": rid,
                                             "decision": decision, "contract": contract,
                                             "hard_reason": "direction_mismatch",
                                             "judge": "direction_alignment_fail",
                                             "accepted": False, "text": text,
                                             "direction_alignment": align})
                            continue
                    attempts.append({"rank": rank, "request_id": rid, "decision": decision,
                                     "contract": contract, "hard_reason": hard_reason,
                                     "judge": judge, "accepted": passed, "text": text})
                    if passed:
                        chosen = (decision, text, rid, messages, resp, rank)
                        break
                for (tk2, k, rid, messages, resp, err, _finished_at) in tr:
                    if chosen is None or rid != chosen[2]:
                        try:
                            os.unlink(str(Path(args.cache_root) / f"{rid}.safetensors"))
                        except FileNotFoundError:
                            pass
                if chosen is None:
                    audit_fp.write(json.dumps(
                        {"slow_anchor_id": anchor_id, "accepted": False,
                         "attempts": attempts}, ensure_ascii=False) + "\n")
                    audit_fp.flush()
                    n_audit += 1
                    print(f"[{group_index}/{len(groups)}] reject {anchor_id}", flush=True)
                    if protocol is not None:
                        broken_tasks.add(tk)
                    continue

                decision, text, request_id, messages, response, rank = chosen
                clean_text, query_tail_count = strip_query_tail(text)
                normalized_completion = (protocol.format_tool_call(
                    decision["name"], decision["arguments"])
                    if protocol is not None else clean_text)
                if protocol is not None:
                    decisions_fp.write(json.dumps(slow_decision_row(
                        anchor_id=anchor_id, item=items[0], instruction=instruction,
                        messages=messages, completion=normalized_completion,
                        decision=decision, pass_rank=rank, protocol=protocol,
                        decision_filter=("golden_success_radius" if
                                         decision["name"] == "stop" else
                                         args.decision_filter)), ensure_ascii=False) + "\n")
                    n_decisions += 1
                if decision["name"] == "stop":
                    terminated_tasks.add(tk)
                    audit_fp.write(json.dumps(
                        {"slow_anchor_id": anchor_id, "accepted": True,
                         "accepted_rank": rank, "decision": decision,
                         "terminal": True, "attempts": attempts},
                        ensure_ascii=False) + "\n")
                    audit_fp.flush()
                    n_audit += 1
                    print(f"[{group_index}/{len(groups)}] accept terminal {anchor_id} "
                          f"rank={rank}", flush=True)
                    continue
                contract = decision["arguments"]
                snapshot = {"text": text, "usage": response.get("usage") or {},
                            "query_request_id": request_id}
                _t0 = time.perf_counter()
                bundle = provider(
                    request={"messages": messages, "instruction": instruction},
                    response=snapshot, contract=contract)
                prof["provider"] += time.perf_counter() - _t0
                current_paths = [resolve_images(item["fast_row"], frame_dir)[-1]
                                 for item in items]
                runner_index = visual_round_robin % len(visual_runners)
                visual_round_robin += 1
                future = visual_executors[runner_index].submit(
                    visual_runners[runner_index].encode_many,
                    [anchor_images[-1], *current_paths])
                pending_visual.append({
                    "future": future, "group_index": group_index, "tk": tk,
                    "anchor_id": anchor_id, "items": items,
                    "instruction": instruction, "anchor_images": anchor_images,
                    "anchor_row": anchor_row, "history": history,
                    "messages": messages, "bundle": bundle, "contract": contract,
                    "clean_text": clean_text, "query_tail_count": query_tail_count,
                    "normalized_completion": normalized_completion,
                    "rank": rank, "attempts": attempts,
                    "current_paths": current_paths,
                })
                # Bound queued image arrays/cache pressure. Waiting here does not
                # idle vLLM: the HTTP executor has already submitted later anchors
                # and continues driving the server in parallel with this commit.
                if len(pending_visual) >= max(args.visual_prefetch,
                                              len(visual_runners)):
                    _finish_visual(pending_visual.popleft())

            http_executor.shutdown(wait=True)
            while pending_visual:
                _finish_visual(pending_visual.popleft())
            layer_finished = time.perf_counter()
            print(f"[pipeline] option_idx={option_idx} mode={args.pipeline_mode} "
                  f"visual_workers={len(visual_runners)} "
                  f"layer_wall={layer_finished-layer_started:.1f}s "
                  f"last_http={last_http_finished-layer_started:.1f}s "
                  f"post_http_tail={layer_finished-last_http_finished:.1f}s",
                  flush=True)

        # ---- layer checkpoint: persist in-memory state after a full option
        # layer so a later kill resumes from here. Truncate-on-resume discards
        # any partial-layer output written after this point. tk tuples are
        # JSON-encoded to preserve element types (source_task_id may be int).
        manifest_fp.flush()
        if decisions_fp is not None:
            decisions_fp.flush()
        _ckpt_tmp = checkpoint_path.with_suffix(".json.tmp")
        _ckpt_tmp.write_text(json.dumps({
            **(protocol.protocol_identity() if protocol is not None else {}),
            **({"beam_contract": beam_contract,
                "nested_plan_sha256": plan_sha256,
                "beams": {json.dumps(k): v for k, v in beams.items()},
                "map_states": {json.dumps(k): v for k, v in map_states.items()}}
               if version == "v6" else {}),
            "option_idx_completed": option_idx,
            "target_horizons": sorted(target_horizons),
            "sample_index": sample_index,
            "group_index": group_index,
            "manifest_count": n_manifest,
            "audit_count": n_audit,
            "decisions_count": n_decisions,
            "broken_tasks": [json.dumps(tk, ensure_ascii=False) for tk in broken_tasks],
            "terminated_tasks": [json.dumps(tk, ensure_ascii=False) for tk in terminated_tasks],
            "histories": {json.dumps(tk, ensure_ascii=False): v for tk, v in histories.items()},
            "last_option_end": {json.dumps(tk, ensure_ascii=False): v for tk, v in last_option_end.items()},
        }, ensure_ascii=False), encoding="utf-8")
        os.replace(_ckpt_tmp, checkpoint_path)
        print(f"[build] checkpoint option_idx={option_idx} "
              f"sample_index={sample_index} manifest={n_manifest} "
              f"audit={n_audit} decisions={n_decisions} "
              f"active_tasks={len(histories)}", flush=True)

    for executor in visual_executors:
        executor.shutdown(wait=True)

    # Finalize: flush + close the incremental jsonl handles (data was written
    # per anchor above; no in-memory lists to dump). Validate counts.
    manifest_fp.flush()
    audit_fp.flush()
    if decisions_fp is not None:
        decisions_fp.flush()
    if n_manifest == 0 and n_decisions == 0:
        raise RuntimeError(
            "all slow anchors were rejected; inspect anchor_audit.jsonl")
    if n_manifest > 0 and len(target_horizons) != 1:
        raise RuntimeError(
            f"bundle contains inconsistent action horizons: {target_horizons}")
    # Close the incremental handles before re-reading the audit file.
    manifest_fp.close()
    audit_fp.close()
    if decisions_fp is not None:
        decisions_fp.close()
    _audit_accept = 0
    _audit_terminal = 0
    if audit_path.exists():
        with audit_path.open(encoding="utf-8") as _af:
            for _line in _af:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _r = json.loads(_line)
                except json.JSONDecodeError:
                    continue
                if _r.get("accepted"):
                    _audit_accept += 1
                if _r.get("terminal"):
                    _audit_terminal += 1
    provenance = {
        "schema": ("fm_nested_bundle_provenance_v4" if version in ("v3", "v4.3", "v4.11", "v6") else
                   "fm_nested_bundle_provenance_v3" if version == "v2"
                   else "fm_nested_bundle_provenance_v2"),
        "action_head_as_tool_version": version,
        **(protocol.protocol_identity() if protocol is not None else {}),
        "nested_plan": str(plan_path),
        "nested_plan_sha256": plan_sha256,
        "backbone": str(Path(args.backbone).resolve()), "slow_model": args.model,
        "slow_hidden_size": args.slow_hidden_size,
        "npz_compress_level": args.npz_compress_level,
        "task_partition": {"index": args.task_shard_index,
                           "count": args.task_shard_count},
        "pass_k": args.pass_k, "decision_filter": args.decision_filter,
        "beam_contract": beam_contract if version == "v6" else None,
        "trajectory_map_source": "completed_golden_tool_results_only" if version == "v6" else None,
        "judge_role": ("none" if version == "v6" else "golden-direction-consistency" if version in ("v4.3", "v4.11") else "same-model-self-judge"),
        "teacher_turns": ({"policy": "reject candidate without changing pose or history",
                           "supervision": "unchanged golden trajectory"}
                          if version in ("v4.3", "v4.11", "v6") else None),
        "stop_label_gate": {
            "type": "remaining_geodesic_arc_lte_success_radius",
            "success_radius_m": args.stop_success_radius_m,
        } if protocol is not None else None,
        "anchors_attempted": n_audit,
        # Recompute acceptance stats from the durable audit file so a resumed
        # run (whose accepts happened in a prior partial run) reports the
        # full-run totals, not just the post-resume increments.
        "anchors_accepted": _audit_accept,
        "slow_decisions": n_decisions,
        "terminal_decisions": _audit_terminal,
        "samples": n_manifest,
        "camera": {"height_m": 1.2, "hfov_deg": 115.0,
                   "width": 640, "height": 480},
        "trajectory": ({"representation": "normalized single-step ego delta",
                       "action_dim": 3, "horizon": target_horizons.pop(),
                       "padding": "endpoint-repeat with loss mask",
                       "base_convention": "x-forward,y-right,yaw-clockwise"}
                       if n_manifest > 0 else None),
        "boundary": ("VLM PASS is a same-model offline self-consistency filter; it "
                     "does not prove correctness or license evaluation leakage."),
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
