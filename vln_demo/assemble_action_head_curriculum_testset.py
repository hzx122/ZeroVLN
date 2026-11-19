#!/usr/bin/env python3
"""Materialize the read-only P0/P1/P2 curriculum as one Sage test split."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_ROOT = Path("/data/zerovln/action_head/curriculum")
STAGES = ("p0", "p1", "p2")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":")).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path,
                        default=DEFAULT_ROOT / "testset" / "Trajectory_demo")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    roots = {stage: args.root / stage / "Trajectory_demo" for stage in STAGES}
    missing = [str(path) for path in roots.values() if not path.is_dir()]
    if missing:
        raise SystemExit("missing curriculum inputs: " + ", ".join(missing))
    output = args.output
    if output.exists():
        if not args.force:
            raise SystemExit("output already exists; pass --force to rebuild: " + str(output))
        shutil.rmtree(output)
    grouped: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for stage, root in roots.items():
        for source in sorted(root.glob("*/*.json")):
            grouped[source.parent.name].append((stage, source))
    if not grouped:
        raise SystemExit("no source JSON found")
    source_summary, counts = [], Counter()
    for scene, entries in sorted(grouped.items()):
        samples = []
        source_pipelines = {}
        for stage, source in entries:
            document = json.loads(source.read_text(encoding="utf-8"))
            source_pipelines[stage] = document.get("pipeline", {})
            blocks = document.get("scenes") or []
            for block in blocks:
                for sample in block.get("samples") or []:
                    copied = json.loads(json.dumps(sample))
                    original_id = str(copied.get("trajectory_id") or "")
                    if not original_id:
                        raise ValueError("missing trajectory_id: " + str(source))
                    copied["trajectory_id"] = original_id + "__" + stage
                    copied["curriculum_stage"] = stage.upper()
                    for instruction in copied.get("instructions") or []:
                        instruction["instruction_type"] = "Curriculum/" + stage.upper()
                        instruction["level"] = "action_head_" + stage
                    samples.append(copied)
                    counts[stage] += 1
        payload = {
            "pipeline": {
                "name": "sage_action_head_curriculum_testset",
                "version": "2026-08-24-action-head-curriculum-v1",
                "canonical": False,
                "config": {
                    "config_sha256": digest({
                        "sources": {stage: (pipe.get("config") or {}).get("config_sha256")
                                    for stage, pipe in source_pipelines.items()},
                        "schema": "p0-p2-merged-v1",
                    }),
                    "source_stages": sorted(source_pipelines),
                },
                "source_pipelines": source_pipelines,
            },
            "scenes": [{"scene_id": "demo_" + scene, "samples": samples}],
        }
        target = output / scene / (scene + "_curriculum.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
        source_summary.append({"scene": scene, "trajectory_count": len(samples),
                               "source_stages": sorted(source_pipelines)})
    manifest = {
        "schema": "sage_action_head_curriculum_testset_v1",
        "split_unit": "physical_trajectory_variant",
        "source_roots": {stage: str(root) for stage, root in roots.items()},
        "output_root": str(output),
        "stage_counts": dict(counts),
        "trajectory_count": sum(counts.values()),
        "scene_count": len(source_summary),
        "scenes": source_summary,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
