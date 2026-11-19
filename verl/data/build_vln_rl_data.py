#!/usr/bin/env python3
"""Build verl-native RL jsonl for VLN goal-point GRPO (Phase 1).

Source: date_mfm VLN annotation jsonl (e.g. exp13-vln_ce_ar_fm_full_merge.jsonl).
Only the AR branch (metadata.train_branch == "ar" with a concrete goal_point) is
kept: the policy is RL-ed to emit ``goal point: [x, y]`` in text. The full
continuous trajectory lives in the FM head and is handled in Phase 2.

Each output record:
  {
    "data_source": "vln_goal_point",
    "images": [{"image": "file://...jpg", "min_pixels":..., "max_pixels":...}, ...x8],
    "prompt": [{"role":"system",...}, {"role":"user","content":"<image>x8 ... Inter instruction ..."}],
    "reward_model": {"style":"rule","ground_truth":{"goal_point":[x,y]}},
    "extra_info": {"goal_point":[x,y],"resolution":[w,h],"arrive":bool,"traj_type":str,"token":str,"index":int}
  }

Media stays lazy: images are written as file:// descriptors, decoded later by
verl's vision preprocessing. Mirrors VLA-RLVR/tools/data_pipeline/vl_unified_to_verl.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "You are an embodied navigation agent. Given the instruction and the historical "
    "observations, predict the goal point the agent should reach.\n"
    "Coordinates live on a fixed 0-1000 canvas (x to the right, y downward).\n"
    "Answer with EXACTLY one line and nothing else, in the form:\n"
    "goal point: [x, y]\n"
    "Do not add any explanation, reasoning, or extra text."
)


def _file_uri(path: str) -> str:
    return path if path.startswith("file://") else f"file://{path}"


def _images(record: dict[str, Any], *, min_pixels: int, max_pixels: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in record.get("image") or []:
        path = item.get("path") if isinstance(item, dict) else str(item)
        if not path:
            raise ValueError(f"image item without path: {item!r}")
        out.append({"image": _file_uri(path), "min_pixels": min_pixels, "max_pixels": max_pixels})
    return out


def _human_prompt(record: dict[str, Any]) -> str | None:
    for turn in record.get("conversations") or []:
        if str(turn.get("from") or turn.get("role")).lower() in {"human", "user"}:
            return str(turn.get("value") or turn.get("content") or "")
    return None


def _goal_point(goal: Any) -> list[int] | None:
    """Return [x, y] ints, or None if absent / 'none' / unparsable."""
    if not (isinstance(goal, (list, tuple)) and len(goal) == 2):
        return None
    try:
        x, y = int(float(goal[0])), int(float(goal[1]))
    except (TypeError, ValueError):
        return None  # e.g. ["none", "none"]
    return [x, y]


def _resolution(record: dict[str, Any]) -> list[int] | None:
    for item in record.get("image") or []:
        if isinstance(item, dict) and item.get("resolution"):
            return list(item["resolution"])
    return None


def convert_record(record: dict[str, Any], index: int, args: argparse.Namespace) -> dict[str, Any] | None:
    md = record.get("metadata") or {}
    if md.get("train_branch") != "ar":
        return None
    goal = _goal_point(md.get("goal_point"))
    if goal is None:
        return None  # no supervision signal for goal-point RL

    images = _images(record, min_pixels=args.image_min_pixels, max_pixels=args.image_max_pixels)
    if args.require_media and not images:
        return None

    human = _human_prompt(record)
    if not human:
        return None
    n_ph = human.count("<image>")
    if n_ph != len(images):
        return None  # placeholder / image count mismatch -> skip to keep verl vision happy

    token = str(record.get("id") or index)
    return {
        "data_source": "vln_goal_point",
        "images": images,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": human},
        ],
        "reward_model": {"style": "rule", "ground_truth": {"goal_point": goal}},
        "extra_info": {
            "index": index,
            "token": token,
            "goal_point": goal,
            "resolution": _resolution(record),
            "arrive": bool(md.get("arrive")),
            "traj_type": str(md.get("traj_type") or ""),
            "asset_id": str(md.get("asset_id") or ""),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--require-media", action="store_true")
    parser.add_argument("--image-min-pixels", type=int, default=28 * 28 * 64)
    parser.add_argument("--image-max-pixels", type=int, default=28 * 28 * 256)
    args = parser.parse_args()

    stats = {"read": 0, "written": 0, "skipped": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            if args.limit is not None and stats["written"] >= args.limit:
                break
            if not line.strip():
                continue
            stats["read"] += 1
            rec = convert_record(json.loads(line), index=stats["read"] - 1, args=args)
            if rec is None:
                stats["skipped"] += 1
                continue
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["written"] += 1

    print(json.dumps({"input": str(args.input), "output": str(args.output), "stats": stats}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
