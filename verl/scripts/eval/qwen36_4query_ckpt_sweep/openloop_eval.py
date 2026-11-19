#!/usr/bin/env python3
"""Faithful open-loop validation for one joint query/FM-head checkpoint."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO), str(REPO / "verl")]

from action_head.policy import TrajectoryTransform  # noqa: E402
from action_head import protocol as ACTION_HEAD_PROTOCOL  # noqa: E402
from models.dualvln_query_encoder import DualVlnQueryEncoder  # noqa: E402
from models.qwen36_fast_slow_action_head import Qwen36FastSlowActionHead  # noqa: E402
from sft.joint_fm_head_sft import FrozenQwenQueryForward, build_slow_inputs  # noqa: E402

from common import (atomic_json, checkpoint_identity, contract_direction,
                    fixed_anchor_subset, mean, sha256_file)  # noqa: E402

from openloop_metrics import (METRIC_VERSIONS, aggregate_windows,
                              denormalize_fp32, trajectory_metrics)

Q01 = [-0.10273411870002747, -0.10309220850467682, -0.5668500065803528]
Q99 = [0.1031249612569809, 0.10309189558029175, 0.5747644901275635]
NOISE_BASE = 20260828


def stable_seed(row: dict, draw: int) -> int:
    key = f"{row['source_scene']}:{row['source_task_id']}:{row['slow_anchor_id']}:{row['fast_window_index']}"
    offset = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")
    return NOISE_BASE + offset + draw


def load_validation(manifest: Path, limit_rows: int, max_anchors: int,
                    anchor_sample_seed: int) -> tuple[list[dict], int, int]:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    rows = [row for row in rows if row.get("split") == "validation"]
    source_window_count = len(rows)
    source_anchor_count = len({row["slow_anchor_id"] for row in rows})
    rows, _anchors = fixed_anchor_subset(
        rows, max_anchors=max_anchors, seed=anchor_sample_seed)
    if limit_rows:
        rows = rows[:limit_rows]
    if not rows:
        raise RuntimeError("validation split is empty")
    return rows, source_window_count, source_anchor_count


def aggregate_draws(records: list[dict]) -> dict:
    keys = ("prefix_ade_m", "prefix_fde_m", "full_ade_m", "full_fde_m",
            "endpoint_yaw_error_deg")
    return {key: mean(record[key] for record in records) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--slow-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--metric-version", choices=METRIC_VERSIONS,
                        default="legacy_bf16_v1",
                        help="Use fp32_v2 for new searches; legacy preserves historical scores.")
    parser.add_argument("--num-random", type=int, default=8)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--max-anchors", type=int, default=0)
    parser.add_argument("--anchor-sample-seed", type=int, default=20260828)
    parser.add_argument(
        "--action-head-as-tool-version",
        choices=("v1", "v2", "v3", "v4", "v4.1", "v4.2", "v4.3",
                 "v4.4", "v4.5", "v4.6", "v4.7", "v4.8", "v4.9", "v4.10", "v4.11", "v6"),
        default="v1")
    args = parser.parse_args()
    if args.num_random <= 0 or args.limit_rows < 0 or args.max_anchors < 0:
        parser.error("num-random must be positive; limits must be non-negative")

    manifest = Path(args.manifest)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    previous_summary = output / "summary.json"
    if previous_summary.exists():
        previous_version = json.loads(previous_summary.read_text()).get(
            "metric_version", "legacy_bf16_v1")
        if previous_version != args.metric_version:
            raise ValueError("Metric version differs; use a separate evaluation output directory")
    identity = checkpoint_identity(args.checkpoint_dir)
    native = identity.get("condition_mode") == "v5_native"
    rows, source_window_count, source_anchor_count = load_validation(
        manifest, args.limit_rows, args.max_anchors, args.anchor_sample_seed)
    protocol = ACTION_HEAD_PROTOCOL.protocol_module(args.action_head_as_tool_version)
    if protocol is not None:
        for index, row in enumerate(rows):
            protocol.assert_protocol_identity(row)
            if row.get("schema") != protocol.BUNDLE_SCHEMA:
                raise ValueError(
                    f"validation row {index} is not {protocol.BUNDLE_SCHEMA}")
    anchor_count = len({row["slow_anchor_id"] for row in rows})
    atomic_json(output / "protocol.json", {
        "schema": "qwen36_4query_openloop_protocol_v1",
        "metric_version": args.metric_version,
        "action_head_as_tool_version": args.action_head_as_tool_version,
        **(protocol.protocol_identity() if protocol is not None else {}),
        "checkpoint": identity,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "split": "validation",
        "window_count": len(rows),
        "slow_anchor_count": anchor_count,
        "source_window_count": source_window_count,
        "source_slow_anchor_count": source_anchor_count,
        "max_anchors": args.max_anchors,
        "anchor_sample_seed": args.anchor_sample_seed,
        "anchor_sampling": "sha256(seed:slow_anchor_id), keep all owned windows",
        "condition_encoding": ("frozen_native_frontend_then_trained_aggregator" if native
                               else "forward_full(current_checkpoint_query)"),
        "visual_inputs": "saved reference/current visual tokens from SFT bundle",
        "random_draws": args.num_random,
        "noise_seed_policy": "sample-specific stable seeds shared by all checkpoints",
        "zero_noise_role": "diagnostic only, excluded from primary aggregate",
    })

    dtype = torch.bfloat16
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.slow_model, trust_remote_code=True)
    if native:
        from action_head.context_encoder import ContextAggregator, NativeQwenFrontend
        frontend = NativeQwenFrontend(args.slow_model, device=device, dtype=dtype).eval()
        aggregator = ContextAggregator().to(device=device, dtype=dtype).eval()
        aggregator.load_state_dict(torch.load(identity["aggregator_path"],
                                             map_location=device, weights_only=True))
    else:
        qwen = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            args.slow_model, dtype=dtype, device_map="auto", low_cpu_mem_usage=True,
            attn_implementation="sdpa", trust_remote_code=True).eval()
        query = DualVlnQueryEncoder.from_pretrained(identity["query_dir"]).to(
            device=device, dtype=dtype).eval()
        hidden_size = int(qwen.config.text_config.hidden_size)
        if query.hidden_size != hidden_size:
            raise ValueError(f"query hidden size {query.hidden_size} != Qwen {hidden_size}")
        frozen = FrozenQwenQueryForward(qwen, query)
    head = Qwen36FastSlowActionHead.from_pretrained(identity["action_head_dir"]).to(
        device=device, dtype=dtype).eval()
    transform = TrajectoryTransform.from_quantiles(Q01, Q99, flip_y=True, flip_yaw=True)

    def metric_actions(value):
        if args.metric_version == "fp32_v2":
            return denormalize_fp32(value.float().cpu().numpy(), transform)
        return transform.apply(value).float().cpu().numpy()

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["slow_anchor_id"]].append(row)
    per_window_path = output / "per_window.jsonl"
    with per_window_path.open("w", encoding="utf-8") as sink:
        completed = 0
        window_summaries = []
        for anchor_index, anchor_id in enumerate(sorted(grouped), 1):
            anchor_rows = grouped[anchor_id]
            exemplar = anchor_rows[0]
            slow_inputs = build_slow_inputs(
                processor, exemplar["slow_messages"], exemplar["slow_completion"], device)
            with torch.no_grad():
                if native:
                    if slow_inputs['image_grid_thw'].shape[0] > 11:
                        raise ValueError('native evaluation exceeds the shared 8+3 image window')
                    query_hidden = aggregator(frontend(slow_inputs),
                                              slow_inputs['attention_mask'].bool(),
                                              slow_inputs['input_ids'].eq(frontend.image_id))
                    condition = dict(query_hidden_states=None, slow_condition_tokens=query_hidden)
                else:
                    query_hidden = frozen.forward_full(slow_inputs)
                    condition = dict(query_hidden_states=query_hidden)
            for row in anchor_rows:
                bundle = np.load(row["sample"])

                def tensor(name: str) -> torch.Tensor:
                    return torch.from_numpy(bundle[name]).to(
                        device=device, dtype=dtype).unsqueeze(0).contiguous()

                reference = tensor("reference_visual_tokens")
                current = tensor("current_visual_tokens")
                target_norm = torch.from_numpy(bundle["target_actions"]).to(
                    device=device, dtype=(torch.float32 if args.metric_version == "fp32_v2" else dtype))
                gt = metric_actions(target_norm.unsqueeze(0))[0]
                ref_mask = torch.ones(reference.shape[:2], device=device, dtype=torch.bool)
                cur_mask = torch.ones(current.shape[:2], device=device, dtype=torch.bool)
                draws = []
                first_prediction = None
                for draw in range(args.num_random):
                    seed = stable_seed(row, draw)
                    generator = torch.Generator(device=device).manual_seed(seed)
                    with torch.no_grad():
                        pred_norm = head.sample(
                            **condition,
                            reference_visual_tokens=reference,
                            reference_visual_mask=ref_mask,
                            current_visual_tokens=current,
                            current_visual_mask=cur_mask,
                            generator=generator)
                    pred = metric_actions(pred_norm)[0]
                    if first_prediction is None:
                        first_prediction = pred.tolist()
                    draws.append({"draw": draw, "seed": seed,
                                  **trajectory_metrics(pred, gt, row["valid_action_steps"])})
                zero = torch.zeros((1, head.config.horizon, head.config.action_dim),
                                   device=device, dtype=dtype)
                with torch.no_grad():
                    zero_norm = head.sample(
                        **condition,
                        reference_visual_tokens=reference,
                        reference_visual_mask=ref_mask,
                        current_visual_tokens=current,
                        current_visual_mask=cur_mask,
                        initial_noise=zero)
                zero_pred = metric_actions(zero_norm)[0]
                item = {
                    "metric_version": args.metric_version,
                    "scene": row["source_scene"],
                    "task_id": row["source_task_id"],
                    "slow_anchor_id": anchor_id,
                    "fast_window_index": int(row["fast_window_index"]),
                    "valid_action_steps": int(row["valid_action_steps"]),
                    "contract_direction": contract_direction(row["contract"]),
                    "primary": aggregate_draws(draws),
                    "draws": draws,
                    "zero_noise": trajectory_metrics(
                        zero_pred, gt, row["valid_action_steps"]),
                    "trajectory_example": {
                        "seed": draws[0]["seed"],
                        "prediction": first_prediction,
                        "ground_truth": gt.tolist(),
                    },
                }
                sink.write(json.dumps(item, ensure_ascii=False) + "\n")
                sink.flush()
                window_summaries.append({key: item[key] for key in ("scene", "task_id", "primary")})
                bundle.close()
                completed += 1
                print(f"[openloop] anchor={anchor_index}/{anchor_count} "
                      f"window={completed}/{len(rows)} full_ade={item['primary']['full_ade_m']:.4f}",
                      flush=True)
            del query_hidden, slow_inputs
            gc.collect()

    aggregates = aggregate_windows(window_summaries)
    summary = {
        "schema": "qwen36_4query_openloop_summary_v1",
        "metric_version": args.metric_version,
        "task_count": aggregates["task_count"],
        "scene_count": aggregates["scene_count"],
        "task_macro_mean": aggregates["task_macro_mean"],
        "scene_macro_mean": aggregates["scene_macro_mean"],
        "checkpoint": identity,
        "action_head_as_tool_version": args.action_head_as_tool_version,
        **(protocol.protocol_identity() if protocol is not None else {}),
        "evaluation_valid": True,
        "split": "validation",
        "window_count": len(rows),
        "slow_anchor_count": anchor_count,
        "primary_random_mean": aggregates["window_mean"],
        "primary_metrics": ["prefix_ade_m", "prefix_fde_m", "full_ade_m", "full_fde_m"],
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
