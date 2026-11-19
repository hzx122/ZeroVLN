#!/usr/bin/env python3
"""Rescore v4.3 rollouts with exact official semantic-map item IDs.

This deliberately bypasses the local labels.json fallback/index reconstruction.
It calls the pinned upstream ObjectBasedSuccessEvaluator on semantic maps built
by the pinned upstream semantic_map_builder, with no smart-target fallback.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path("/workspace/ZeroVLN")
OFFICIAL = REPO.parent / "SAGE-3D_Official/Code/benchmark/environment_evaluation"
sys.path[:0] = [str(OFFICIAL), str(REPO / "vln_demo/src")]

import object_based_success as obs  # noqa: E402
from sage.datasets import SageSet  # noqa: E402
from sage.semantic_map_transform import cached_reverse_mapping  # noqa: E402

EVAL_ROOT = Path(
    "/data/zerovln/artifacts/"
    "action-head-v43-official-gvln-d16-allrandomquery-randomhead-10ep-4node-"
    "gb64-20260911-r3/eval/test100-ckpt1000-openclosed-k4-tool50-official-"
    "object-1m-v1"
)
OUTPUT = EVAL_ROOT / "analysis/evaluation_target_audit_20260916"
MAP_ROOT = OUTPUT / "official_semantic_maps"
DATASET = SageSet("official_gvln_action_head_p0_v3test100")

ORIGINAL_REVERSE = obs.reverse_position_mapping
cached_reverse = cached_reverse_mapping(ORIGINAL_REVERSE)


def scene_map(scene: str) -> Path:
    hits = list(MAP_ROOT.glob(f"*_{scene.removeprefix('sage-')}_Complete.json"))
    if len(hits) != 1:
        raise RuntimeError(f"expected one map for {scene}, got {hits}")
    return hits[0]


def score(evaluator, xy, target_id: str) -> bool:
    # SageSet geometry is the rot180 of the official Isaac frame.
    official_position = [-float(xy[0]), -float(xy[1]), 0.5]
    return bool(evaluator.is_position_in_object_area(
        official_position, target_id, expansion_radius=1.0))


def pass4(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["episode_id"]].append(row)
    complete = {key: value for key, value in grouped.items() if len(value) == 4}
    return {
        "success_cases": sum(any(item[field] for item in value)
                             for value in complete.values()),
        "complete_cases": len(complete),
        "value": (sum(any(item[field] for item in value)
                      for value in complete.values()) / len(complete)
                  if complete else None),
    }


def aggregate(rows, field):
    return {
        "valid_rollouts": len(rows),
        "successes": sum(x[field] for x in rows),
        "sr": sum(x[field] for x in rows) / len(rows),
        "pass4": pass4(rows, field),
    }


def main() -> None:
    obs.reverse_position_mapping = cached_reverse
    subset = json.loads((Path(DATASET.vln_dir) / "closedloop_subset_all100.json").read_text())
    cases = {item["episode_id"]: item for item in subset["cases"]}
    evaluators = {scene: obs.ObjectBasedSuccessEvaluator(
        str(scene_map(scene)), verbose=False) for scene in {x["scene"] for x in cases.values()}}
    tasks = {}
    golden = []
    for episode_id, case in cases.items():
        task = {item.task_id: item for item in DATASET.tasks(case["scene"])}[episode_id]
        tasks[episode_id] = task
        evaluator = evaluators[case["scene"]]
        golden.append({
            "episode_id": episode_id, "scene": case["scene"],
            "target_object_id": task.goal_label,
            "instruction": task.instruction,
            "golden_endpoint_x": task.goal_xy[0], "golden_endpoint_y": task.goal_xy[1],
            "golden_endpoint_success": score(evaluator, task.goal_xy, task.goal_label),
            "target_exists_in_official_map": evaluator.get_object_bbox(task.goal_label) is not None,
        })

    rows = []
    per_step_rows = defaultdict(list)
    for step_dir in sorted(EVAL_ROOT.glob("global_step_*")):
        if not (step_dir / "DONE").is_file():
            continue
        step = int(step_dir.name.rsplit("_", 1)[1])
        for path in (step_dir / "closedloop/rollouts").glob("**/*.json"):
            row = json.loads(path.read_text(encoding="utf-8"))
            if not row.get("evaluation_valid"):
                continue
            episode_id = row["episode_id"]
            task = tasks[episode_id]
            detail = row.get("official_object_success_detail") or {}
            xy = detail.get("position_xy")
            if xy is None:
                raise RuntimeError(f"missing terminal position: {path}")
            corrected = score(evaluators[row["scene"]], xy, task.goal_label)
            record = {
                "step": step, "episode_id": episode_id, "scene": row["scene"],
                "repeat_index": row["repeat_index"], "instruction": row["instruction"],
                "target_object_id": task.goal_label,
                "final_x": xy[0], "final_y": xy[1],
                "current_success": bool(row.get("success")),
                "corrected_official_map_success": corrected,
                "changed": bool(row.get("success")) != corrected,
                "rollout_path": str(path),
            }
            rows.append(record); per_step_rows[step].append(record)

    golden_fail_ids = {x["episode_id"] for x in golden
                       if not x["golden_endpoint_success"]}
    # Confirmed by geometry and final-frame inspection: the frozen target is
    # farther from the start than another notebook despite "closest to you".
    confirmed_annotation_conflicts = {"official_839902_92_0"}

    per_step = {}
    for step, items in sorted(per_step_rows.items()):
        per_step[str(step)] = {
            "valid_rollouts": len(items),
            "current_successes": sum(x["current_success"] for x in items),
            "current_sr": sum(x["current_success"] for x in items) / len(items),
            "corrected_successes": sum(x["corrected_official_map_success"] for x in items),
            "corrected_sr": sum(x["corrected_official_map_success"] for x in items) / len(items),
            "false_positive_under_current": sum(
                x["current_success"] and not x["corrected_official_map_success"] for x in items),
            "false_negative_under_current": sum(
                not x["current_success"] and x["corrected_official_map_success"] for x in items),
            "current_pass4": pass4(items, "current_success"),
            "corrected_pass4": pass4(items, "corrected_official_map_success"),
            "qc92_reference_compatible": aggregate([
                x for x in items if x["episode_id"] not in golden_fail_ids],
                "corrected_official_map_success"),
            "qc91_reference_compatible_without_confirmed_conflict": aggregate([
                x for x in items if x["episode_id"] not in golden_fail_ids
                and x["episode_id"] not in confirmed_annotation_conflicts],
                "corrected_official_map_success"),
        }
    summary = {
        "schema": "v43-exact-official-semantic-map-rescore-v1",
        "completed_steps": sorted(per_step_rows), "valid_rollouts": len(rows),
        "current_successes": sum(x["current_success"] for x in rows),
        "corrected_successes": sum(x["corrected_official_map_success"] for x in rows),
        "false_positive_under_current": sum(
            x["current_success"] and not x["corrected_official_map_success"] for x in rows),
        "false_negative_under_current": sum(
            not x["current_success"] and x["corrected_official_map_success"] for x in rows),
        "golden_endpoint_successes": sum(x["golden_endpoint_success"] for x in golden),
        "golden_endpoint_failures": sum(not x["golden_endpoint_success"] for x in golden),
        "targets_missing_from_official_maps": sum(
            not x["target_exists_in_official_map"] for x in golden),
        "golden_endpoint_failure_episode_ids": sorted(golden_fail_ids),
        "confirmed_annotation_conflict_episode_ids": sorted(
            confirmed_annotation_conflicts),
        "per_step": per_step,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("rescored_rollouts.csv", rows), ("golden_endpoint_audit.csv", golden)):
        with (OUTPUT / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    (OUTPUT / "official_map_rescore_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
