#!/usr/bin/env python3
"""Mix a complete golden set with one same-protocol bootstrap round."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np

import sys
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "verl"))
from action_head import protocol_v2 as V2  # noqa: E402
from action_head import protocol_v3 as V3  # noqa: E402

REQUIRED = {
    "sample", "split", "slow_anchor_id", "source_scene", "source_task_id",
    "slow_messages", "slow_completion", "reference_image", "current_image",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path, source: str, protocol) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    for index, row in enumerate(rows):
        missing = REQUIRED - row.keys()
        if missing:
            raise ValueError(f"{source} row {index} missing {sorted(missing)}")
        protocol.assert_protocol_identity(row)
        if row.get("schema") != protocol.BUNDLE_SCHEMA:
            raise ValueError(f"{source} row {index} has schema={row.get('schema')}")
        sample = Path(row["sample"])
        if not sample.is_file():
            raise FileNotFoundError(sample)
        with np.load(sample) as payload:
            if tuple(payload["target_actions"].shape) != (16, 3) or \
                    tuple(payload["action_mask"].shape) != (16,):
                raise ValueError(f"invalid tensor contract: {sample}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-manifest", required=True, type=Path)
    parser.add_argument("--recovery-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--action-head-as-tool-version", choices=("v2", "v3"),
                        required=True)
    parser.add_argument("--bootstrap-round", type=int)
    parser.add_argument("--allow-empty-recovery", action="store_true")
    parser.add_argument("--golden-sampling-unit",
                        choices=("slow_anchor_all_windows", "manifest_row"),
                        default="slow_anchor_all_windows")
    args = parser.parse_args()
    golden_path = args.golden_manifest.resolve()
    recovery_path = args.recovery_manifest.resolve()
    protocol = {"v2": V2, "v3": V3}[args.action_head_as_tool_version]
    golden = load(golden_path, "golden", protocol)
    recovery = load(recovery_path, "recovery", protocol)
    if not recovery and not args.allow_empty_recovery:
        raise ValueError("recovery manifest is empty")
    if any(row.get("split") != "train" for row in recovery):
        raise ValueError("bootstrap recovery rows must all be train split")
    golden_ids = {row["slow_anchor_id"] for row in golden}
    recovery_ids = {row["slow_anchor_id"] for row in recovery}
    overlap = golden_ids & recovery_ids
    if overlap:
        raise ValueError(f"golden/recovery anchor ids overlap: {sorted(overlap)[:5]}")

    rows = golden + recovery
    # The trainer groups windows by slow anchor and uses DistributedSampler.
    # Shuffle here for portable consumers; no source weights or oversampling
    # are introduced, matching aggregate-dataset DAgger semantics.
    random.Random(args.seed).shuffle(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "manifest.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")
    os.replace(temporary, output)
    report = {
        "schema": "action_head_bootstrap_mix_v1",
        "action_head_as_tool_version": args.action_head_as_tool_version,
        "bootstrap_round": args.bootstrap_round,
        **protocol.protocol_identity(),
        "mix_policy": "full_golden_plus_all_recovery_no_oversampling",
        "sampling_unit": args.golden_sampling_unit,
        "seed": args.seed,
        "sources": {
            "golden": {"manifest": str(golden_path), "sha256": sha256(golden_path),
                       "samples": len(golden), "anchors": len(golden_ids)},
            "recovery": {"manifest": str(recovery_path), "sha256": sha256(recovery_path),
                         "samples": len(recovery), "anchors": len(recovery_ids)},
        },
        "output_manifest": str(output.resolve()), "output_sha256": sha256(output),
        "samples": len(rows), "anchors": len(golden_ids | recovery_ids),
        "split_samples": dict(Counter(row["split"] for row in rows)),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "DONE").write_text(
        json.dumps({"ok": True, "samples": len(rows)}) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
