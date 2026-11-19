"""Two-stage physical QC for official SAGE-3D GVLN training trajectories.

Stage 1 calls `filter_reachable.filter_scene` against the *current* SageSet
coordinate transform: start and goal must fit a 0.07m robot on the `usd_rooms`
v4 mask and be connected.  It deliberately does not reuse historical allow-
lists: an August coordinate-frame fix made the old 2026-08-14 allow-list stale.
Stage 2 reuses `audit_action_head_curriculum.replay`: every unique surviving
official path is resampled at 0.1m and replayed with `OccMap.try_step`.

The collision contract is intentionally identical to the frozen action expert
P0-P2 set (29 scenes / 262 physical trajectories): `usd_rooms`, furniture
embedded in USD (`none` extra furniture mask), radius 0.07m, camera 1.2m,
HFOV 115 degrees.  Text is never a quality gate.  Only `Scenario_Driven` and
`Add_Object` instructions are removed on request; all remaining text is copied
verbatim beside the physically admitted reference path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from .datasets import SAGE_ROOT, SageSet

REPO = Path("/workspace/ZeroVLN")
DEFAULT_OUTPUT = Path(SAGE_ROOT) / "official_gvln_physical_clean_v2"
MASK_ROOT = Path(SAGE_ROOT) / "sage_usd_rooms_masks"
EXCLUDED_TYPES = {"scenario driven", "add object"}
CONTRACT = {
    "collision": "usd_rooms",
    "furniture": "none",
    "robot_radius_m": 0.07,
    "replay_step_m": 0.10,
    "camera_height_m": 1.2,
    "hfov_deg": 115,
}


def _normal_type(value: object) -> str:
    return " ".join(str(value or "").replace("_", " ").replace("-", " ").split()).lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_by_id(payload: dict) -> dict[str, dict]:
    result = {}
    for scene in payload.get("scenes", []):
        for sample in scene.get("samples", []):
            result[str(sample.get("trajectory_id"))] = sample
    return result


def _physical_contract() -> None:
    os.environ["SAGE_COLLISION"] = "usd_rooms"
    os.environ["SAGE_FURNITURE_COLLISION"] = "none"
    os.environ["SAGE_ROBOT_RADIUS"] = "0.07"
    os.environ["SAGE_CAMERA_HEIGHT"] = "1.2"
    os.environ["SAGE_HFOV"] = "115"
    os.environ["SAGE_USD_COLLISION_MASK_ROOT"] = str(MASK_ROOT)


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refuse to overwrite existing output: {output}")
    _physical_contract()
    # Reuse the same audited dense replay implementation used by the fixed
    # P0-P2 set rather than maintaining another collision loop.
    sys.path.insert(0, str(REPO / "vln_demo"))
    from audit_action_head_curriculum import replay
    from .occmap import OccMap
    from .filter_reachable import filter_scene

    stage = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    if stage.exists():
        raise FileExistsError(f"staging path already exists: {stage}")
    traj_root = stage / "Trajectory_demo"
    audit_rows, scene_rows = [], []
    instruction_counts, removed_counts = Counter(), Counter()
    total_stage1_tasks = total_stage1_routes = total_stage2_routes = kept_routes = kept_tasks = 0
    failed_routes = 0
    ds = SageSet(split="train")
    try:
        for number, scene in enumerate(ds.list_scenes(), 1):
            bare = ds.bare(scene)
            tasks = ds.tasks(scene, 0)
            occ = OccMap(ds.scene_dir(scene), collision_mode="usd_rooms", furniture_mode="none")
            stage1_tasks, stage1 = filter_scene(occ, tasks, CONTRACT["robot_radius_m"])
            wanted = {task.task_id for task in stage1_tasks}
            task_by_id = {task.task_id: task for task in tasks}
            grouped: dict[str, list[str]] = {}
            for task_id in wanted:
                route_id = task_id.rsplit("_", 1)[0]
                grouped.setdefault(route_id, []).append(task_id)
            total_stage1_tasks += len(wanted)
            total_stage1_routes += len(grouped)
            raw_path = Path(ds.traj_file(scene))
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            samples = _sample_by_id(payload)
            clean_samples = []
            scene_kept_routes = scene_kept_tasks = 0
            for route_id, task_ids in sorted(grouped.items()):
                task = task_by_id.get(task_ids[0])
                sample = samples.get(route_id)
                if task is None or sample is None:
                    audit_rows.append({"scene": bare, "trajectory_id": route_id,
                                       "ok": False, "reason": "source_task_mismatch"})
                    failed_routes += 1
                    continue
                total_stage2_routes += 1
                result = replay(occ, task.waypoints, CONTRACT["robot_radius_m"],
                                CONTRACT["replay_step_m"])
                audit_rows.append({"scene": bare, "trajectory_id": route_id, **result})
                if not result["ok"]:
                    failed_routes += 1
                    continue
                selected = []
                for index, instruction in enumerate(sample.get("instructions") or []):
                    task_id = f"{route_id}_{index}"
                    if task_id not in wanted:
                        continue
                    kind = _normal_type(instruction.get("instruction_type"))
                    if kind in EXCLUDED_TYPES:
                        removed_counts[str(instruction.get("instruction_type") or "")] += 1
                        continue
                    selected.append(instruction)
                    instruction_counts[str(instruction.get("instruction_type") or "")] += 1
                if not selected:
                    continue
                copied = dict(sample)
                copied["instructions"] = selected
                clean_samples.append(copied)
                kept_routes += 1
                scene_kept_routes += 1
                kept_tasks += len(selected)
                scene_kept_tasks += len(selected)
            if clean_samples:
                destination = traj_root / bare
                destination.mkdir(parents=True, exist_ok=True)
                (destination / raw_path.name).write_text(
                    json.dumps({"scenes": [{"samples": clean_samples}]}, ensure_ascii=False) + "\n",
                    encoding="utf-8")
            scene_rows.append({"scene": bare, "stage1_tasks": len(wanted),
                               "stage1_routes": len(grouped),
                               "stage1_reject": stage1,
                               "kept_routes": scene_kept_routes,
                               "kept_tasks": scene_kept_tasks})
            print(f"[clean] {number}/{len(ds.list_scenes())} {bare}: "
                  f"stage1={len(wanted)} routes={scene_kept_routes} tasks={scene_kept_tasks}", flush=True)
            ds._tasks.clear()  # full train parsing is large; retain only one scene at a time.
        audit = {
            "schema": "sage_official_gvln_two_stage_physical_qc_v1",
            "source_split": "train",
            "physical_contract": CONTRACT,
            "stage1": {"criterion": "current rot180 SageSet: start/goal free and connected on usd_rooms",
                       "historical_allowlist": None,
                       "stale_allowlist_not_used": str(Path(SAGE_ROOT) / "episode_filter/kept_train_r0.07.jsonl")},
            "stage2": {"criterion": "0.1m dense replay via audit_action_head_curriculum.replay / OccMap.try_step",
                       "routes_checked": total_stage2_routes, "failed_count": failed_routes,
                       "rows": audit_rows},
            "text_policy": {"quality_gate": False,
                            "excluded_instruction_types": ["Scenario_Driven", "Add_Object"],
                            "removed_counts": dict(removed_counts),
                            "kept_instruction_counts": dict(instruction_counts)},
            "counts": {"stage1_tasks": total_stage1_tasks, "stage1_routes": total_stage1_routes,
                       "kept_routes": kept_routes,
                       "kept_tasks": kept_tasks, "scene_count": sum(x["kept_routes"] > 0 for x in scene_rows)},
            "scenes": scene_rows,
        }
        stage.mkdir(parents=True, exist_ok=True)
        (stage / "physical_qc_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (stage / "manifest.json").write_text(
            json.dumps({k: audit[k] for k in ("schema", "source_split", "physical_contract", "stage1", "text_policy", "counts")},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        os.rename(stage, output)
        return audit
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"output": str(args.output), **result["counts"],
                      "stage2_failed": result["stage2"]["failed_count"],
                      "removed": result["text_policy"]["removed_counts"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
