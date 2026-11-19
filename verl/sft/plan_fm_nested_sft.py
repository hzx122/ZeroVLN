#!/usr/bin/env python3
"""Plan the slow-anchor/fast-window nested sampling for FM-head SFT.

This stage is deliberately model- and renderer-free.  It consumes the raw
``sage_sft.jsonl`` produced by ``build_sage_sft_data.py`` and emits a compact
plan.  Scene splits are inherited from the audited 1537-trajectory manifest;
no route or language paraphrase may cross train/validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from action_head import protocol as ACTION_HEAD_PROTOCOL


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def arc_of(row: dict, point_spacing_m: float) -> float:
    meta = row.get("meta_data") or {}
    if "source_arc_m" in meta:
        return float(meta["source_arc_m"])
    return float(meta["source_point_index"]) * point_spacing_m


def spaced(rows: list[dict], stride_m: float, point_spacing_m: float) -> list[dict]:
    """Select the available row nearest each metric stride target.

    Raw Sage rows can already be subsampled (the current pipeline uses stride
    2), so taking the first row after a target aliases a 0.4 m deployment
    refresh into roughly 0.6 m on 0.1 m source paths. Nearest-grid selection
    keeps the rendered observation closest to the fixed four-by-0.1 m prefix.
    """
    ordered = sorted(rows, key=lambda item: arc_of(item, point_spacing_m))
    if not ordered:
        return []
    arcs = [arc_of(row, point_spacing_m) for row in ordered]
    target = arcs[0]
    last_index = -1
    result = []
    while target <= arcs[-1] + 1e-6:
        index = min(
            range(last_index + 1, len(ordered)),
            key=lambda value: (abs(arcs[value] - target), arcs[value]),
            default=-1,
        )
        if index < 0:
            break
        result.append(ordered[index])
        last_index = index
        target += stride_m
    return result


def spaced_random(rows: list[dict], stride_min_m: float, stride_max_m: float,
                  point_spacing_m: float, seed: int) -> list[dict]:
    """Nearest-grid sampling with deterministic random metric intervals."""
    if stride_min_m <= 0 or stride_max_m < stride_min_m:
        raise ValueError("random stride bounds must satisfy 0 < min <= max")
    ordered = sorted(rows, key=lambda item: arc_of(item, point_spacing_m))
    if not ordered:
        return []
    arcs = [arc_of(row, point_spacing_m) for row in ordered]
    rng = random.Random(seed)
    target = arcs[0]
    last_index = -1
    result = []
    while target <= arcs[-1] + 1e-6:
        index = min(
            range(last_index + 1, len(ordered)),
            key=lambda value: (abs(arcs[value] - target), arcs[value]),
            default=-1,
        )
        if index < 0:
            break
        result.append(ordered[index])
        last_index = index
        target += rng.uniform(stride_min_m, stride_max_m)
    return result


def load_scene_split(path: Path) -> dict[str, str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    splits = manifest["splits"]
    result = {scene: "train" for scene in splits["train_scenes"]}
    for scene in splits["validation_scenes"]:
        if scene in result:
            raise ValueError(f"scene appears in both splits: {scene}")
        result[scene] = "validation"
    return result


def build_plan(rows: list[dict], scene_split: dict[str, str], *,
               slow_stride_m: float, slow_coverage_m: float,
               fast_stride_m: float, point_spacing_m: float,
               min_valid_action_steps: int,
               fast_stride_min_m: float | None = None,
               fast_stride_max_m: float | None = None,
               sampling_seed: int = 20260827,
               action_head_as_tool_version: str = "v1") -> list[dict]:
    version = ACTION_HEAD_PROTOCOL.normalize_version(action_head_as_tool_version)
    modern = version in ("v2", "v3", "v4.3", "v6")
    if modern and abs(slow_stride_m - slow_coverage_m) > 1e-6:
        raise ValueError(
            f"{version} requires slow_stride_m == slow_coverage_m so each next slow "
            "observation is the preceding option's actual final frame")
    by_task: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        meta = row.get("meta_data") or {}
        scene, task = meta.get("source_scene"), meta.get("source_task_id")
        if not scene or not task:
            raise ValueError("raw row is missing source_scene/source_task_id")
        if scene not in scene_split:
            raise ValueError(f"raw scene is absent from split manifest: {scene}")
        if int(meta.get("valid_action_steps", sum(meta["waypoint_mask"]))) \
                < min_valid_action_steps:
            continue
        by_task[(scene, task)].append(row)

    plan = []
    for (scene, task), task_rows in sorted(by_task.items()):
        task_seed = int.from_bytes(hashlib.sha256(
            f"{sampling_seed}:{scene}:{task}".encode()).digest()[:8], "big")
        if fast_stride_min_m is None and fast_stride_max_m is None:
            fast_rows = spaced(task_rows, fast_stride_m, point_spacing_m)
        else:
            fast_rows = spaced_random(
                task_rows,
                fast_stride_min_m if fast_stride_min_m is not None else fast_stride_m,
                fast_stride_max_m if fast_stride_max_m is not None else fast_stride_m,
                point_spacing_m, task_seed)
        anchors = spaced(task_rows, slow_stride_m, point_spacing_m)
        for anchor_index, anchor in enumerate(anchors):
            anchor_arc = arc_of(anchor, point_spacing_m)
            anchor_meta = anchor.get("meta_data") or {}
            anchor_remaining = float(anchor_meta.get("remaining_arc_m", 0.0))
            inner = [row for row in fast_rows
                     if anchor_arc - 1e-6 <= arc_of(row, point_spacing_m)
                     <= anchor_arc + slow_coverage_m + 1e-6]
            option_end_row = min(
                task_rows,
                key=lambda row: abs(
                    arc_of(row, point_spacing_m)
                    - min(anchor_arc + slow_coverage_m,
                          max(arc_of(item, point_spacing_m)
                              for item in task_rows))),
            )
            option_end_arc = arc_of(option_end_row, point_spacing_m)
            if modern:
                if option_end_arc <= anchor_arc + 1e-6:
                    continue
                inner = [row for row in inner
                         if arc_of(row, point_spacing_m) < option_end_arc - 1e-6]
                if not inner or abs(arc_of(inner[0], point_spacing_m) - anchor_arc) > 1e-6:
                    inner.insert(0, anchor)
                # Production executes at most four fast prefixes per option.
                inner = inner[:4]
            if not inner:
                continue
            anchor_id = f"{scene}:{task}:slow{anchor_index:03d}"
            for inner_index, row in enumerate(inner):
                plan.append({
                    "schema": (f"fm_nested_sft_plan_{version}" if modern
                               else "fm_nested_sft_plan_v1"),
                    "action_head_as_tool_version": version,
                    "split": scene_split[scene],
                    "slow_anchor_id": anchor_id,
                    "option_sequence_index": anchor_index,
                    "slow_anchor_arc_m": anchor_arc,
                    "slow_anchor_remaining_arc_m": anchor_remaining,
                    "slow_coverage_m": slow_coverage_m,
                    "option_coverage_truncated": anchor_remaining + 1e-6 < slow_coverage_m,
                    "slow_anchor_row": anchor,
                    "fast_window_index": inner_index,
                    "fast_window_arc_m": arc_of(row, point_spacing_m),
                    "fast_interval_from_previous_m": (
                        None if inner_index == 0 else
                        arc_of(row, point_spacing_m)
                        - arc_of(inner[inner_index - 1], point_spacing_m)),
                    "fast_sampling_seed": sampling_seed,
                    "fast_offset_from_slow_m": arc_of(row, point_spacing_m) - anchor_arc,
                    "fast_remaining_arc_m": float(
                        (row.get("meta_data") or {}).get("remaining_arc_m", 0.0)),
                    "fast_row": row,
                    **({
                        "chunk_end_row": (
                            inner[inner_index + 1]
                            if inner_index + 1 < len(inner) else option_end_row),
                        "option_end_row": option_end_row,
                        "option_end_arc_m": option_end_arc,
                    } if modern else {}),
                })
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-jsonl", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--slow-stride-m", type=float, default=0.8)
    parser.add_argument("--slow-coverage-m", type=float, default=1.6)
    parser.add_argument("--fast-stride-m", type=float, default=0.4)
    parser.add_argument("--fast-stride-min-m", type=float, default=None)
    parser.add_argument("--fast-stride-max-m", type=float, default=None)
    parser.add_argument("--sampling-seed", type=int, default=20260827)
    parser.add_argument("--point-spacing-m", type=float, default=0.1,
                        help="Fallback only for legacy raw rows without source_arc_m.")
    parser.add_argument("--min-valid-action-steps", type=int, default=4)
    parser.add_argument("--max-anchors", type=int, default=0)
    parser.add_argument("--action-head-as-tool-version", choices=("v1", "v2", "v3", "v4.3", "v6"),
                        default="v1")
    args = parser.parse_args()
    for name in ("slow_stride_m", "slow_coverage_m", "fast_stride_m",
                 "point_spacing_m"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if (args.fast_stride_min_m is None) != (args.fast_stride_max_m is None):
        raise ValueError("set both --fast-stride-min-m and --fast-stride-max-m")

    raw_path = Path(args.raw_jsonl).resolve()
    split_path = Path(args.split_manifest).resolve()
    rows = read_jsonl(raw_path)
    plan = build_plan(
        rows, load_scene_split(split_path),
        slow_stride_m=args.slow_stride_m,
        slow_coverage_m=args.slow_coverage_m,
        fast_stride_m=args.fast_stride_m,
        point_spacing_m=args.point_spacing_m,
        min_valid_action_steps=args.min_valid_action_steps,
        fast_stride_min_m=args.fast_stride_min_m,
        fast_stride_max_m=args.fast_stride_max_m,
        sampling_seed=args.sampling_seed,
        action_head_as_tool_version=args.action_head_as_tool_version,
    )
    if args.max_anchors:
        keep = []
        allowed = set()
        for row in plan:
            anchor = row["slow_anchor_id"]
            if anchor not in allowed and len(allowed) >= args.max_anchors:
                continue
            allowed.add(anchor)
            keep.append(row)
        plan = keep
    if not plan:
        raise RuntimeError("nested sampling produced no examples")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                              for row in plan), encoding="utf-8")
    anchors = {row["slow_anchor_id"] for row in plan}
    summary = {
        "schema": (f"fm_nested_sft_plan_summary_{args.action_head_as_tool_version}"
                   if args.action_head_as_tool_version in ("v2", "v3", "v4.3", "v6")
                   else "fm_nested_sft_plan_summary_v1"),
        "action_head_as_tool_version": args.action_head_as_tool_version,
        "raw_jsonl": str(raw_path),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "split_manifest": str(split_path),
        "split_manifest_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "slow_stride_m": args.slow_stride_m,
        "slow_coverage_m": args.slow_coverage_m,
        "fast_stride_m": args.fast_stride_m,
        "fast_stride_min_m": args.fast_stride_min_m,
        "fast_stride_max_m": args.fast_stride_max_m,
        "sampling_seed": args.sampling_seed,
        "spacing_selection": ("deterministic_random_nearest_available_to_metric_grid"
                              if args.fast_stride_min_m is not None else
                              "nearest_available_to_metric_grid"),
        "min_valid_action_steps": args.min_valid_action_steps,
        "anchor_count": len(anchors),
        "sample_count": len(plan),
        "train_samples": sum(row["split"] == "train" for row in plan),
        "validation_samples": sum(row["split"] == "validation" for row in plan),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
