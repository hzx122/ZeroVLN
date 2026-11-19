#!/usr/bin/env python3
"""Build the immutable balanced P0/P1/P2 50-case closed-loop subset."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO / "vln_demo" / "src"), str(REPO / "verl")]

from sage.datasets import SageSet  # noqa: E402

from common import STAGE_QUOTAS, atomic_json, sha256_file  # noqa: E402


def stage_of(task) -> str:
    text = f"{task.instruction_type} {task.task_id}".lower()
    for stage in ("p0", "p1", "p2"):
        if f"curriculum/{stage}" in text or f"__{stage}_" in text:
            return stage
    raise ValueError(f"cannot recover stage for {task.task_id}")


def load_cases(dataset_split: str, stage: str = "") -> list[dict]:
    dataset = SageSet(dataset_split)
    rows = []
    for scene in dataset.list_scenes():
        for task in dataset.tasks(scene, 0):
            rows.append({
                "scene": scene,
                "episode_id": task.task_id,
                "instruction": task.instruction,
                "instruction_type": task.instruction_type,
                "stage": stage or stage_of(task),
                "ref_len": float(task.ref_len),
            })
    rows.sort(key=lambda row: (row["stage"], row["scene"], row["episode_id"]))
    if dataset_split == "action_head_curriculum" and Counter(
            row["stage"] for row in rows) != Counter({"p0": 37, "p1": 57, "p2": 168}):
        raise RuntimeError("frozen 262-case dataset does not match the audited stage counts")
    return rows


def stable_stage_sample(rows: list[dict], stage: str, count: int, seed: int) -> list[dict]:
    bucket = [row for row in rows if row["stage"] == stage]
    stage_seed = seed + int.from_bytes(hashlib.sha256(stage.encode()).digest()[:4], "big")
    random.Random(stage_seed).shuffle(bucket)
    return sorted(bucket[:count], key=lambda row: (row["scene"], row["episode_id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--check-existing", action="store_true")
    parser.add_argument("--dataset-split", default="action_head_curriculum")
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--stage", default="")
    args = parser.parse_args()
    rows = load_cases(args.dataset_split, args.stage)
    chosen = []
    if args.all_cases:
        chosen = rows
    else:
        for stage, count in STAGE_QUOTAS.items():
            chosen.extend(stable_stage_sample(rows, stage, count, args.seed))
    chosen.sort(key=lambda row: (row["stage"], row["scene"], row["episode_id"]))
    payload = {
        "schema": "qwen36_action_head_closedloop_subset_v1",
        "selection": ("all cases in registered dataset split" if args.all_cases else
                      "fixed stratified random sample within each curriculum stage"),
        "seed": args.seed,
        "dataset_split": args.dataset_split,
        "quotas": (dict(Counter(row["stage"] for row in chosen))
                   if args.all_cases else STAGE_QUOTAS),
        "case_count": len(chosen),
        "source_manifest": str(Path(args.source_manifest).resolve()),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "cases": chosen,
    }
    target = Path(args.output)
    if args.check_existing and target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"existing fixed subset differs from deterministic rebuild: {target}")
    else:
        atomic_json(target, payload)
    print(json.dumps({"output": str(target), "case_count": len(chosen),
                      "stage_counts": dict(Counter(row["stage"] for row in chosen)),
                      "sha256": sha256_file(target)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
