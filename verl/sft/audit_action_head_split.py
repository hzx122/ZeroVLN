#!/usr/bin/env python3
"""Audit train/validation leakage in an action expert nested manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from action_head import protocol as ACTION_HEAD_PROTOCOL


def _bytes_sha256(path: str, cache: dict[str, str]) -> str:
    if path not in cache:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        cache[path] = digest.hexdigest()
    return cache[path]


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.strip().encode("utf-8")).hexdigest()


def _trajectory_sha256(sample: str) -> str:
    with np.load(sample) as data:
        digest = hashlib.sha256()
        for key in ("target_actions", "action_mask"):
            value = np.ascontiguousarray(data[key])
            digest.update(key.encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(value.shape).encode("ascii"))
            digest.update(value.tobytes())
        return digest.hexdigest()


def audit_manifest(path: Path, action_head_as_tool_version: str = "v1") -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]
    protocol = ACTION_HEAD_PROTOCOL.protocol_module(action_head_as_tool_version)
    if protocol is not None:
        for index, row in enumerate(rows):
            protocol.assert_protocol_identity(row)
            if row.get("schema") != protocol.BUNDLE_SCHEMA:
                raise ValueError(f"row {index} has wrong versioned bundle schema")
    splits = {name: [row for row in rows if row.get("split") == name]
              for name in ("train", "validation")}
    if not all(splits.values()):
        raise ValueError("manifest must contain non-empty train and validation splits")
    file_hash_cache: dict[str, str] = {}

    def values(split: str, key: str) -> set[str]:
        return {str(row[key]) for row in splits[split] if row.get(key) is not None}

    overlap = {}
    for key in ("source_scene", "source_task_id", "slow_anchor_id"):
        overlap[key] = sorted(values("train", key) & values("validation", key))
    for image_key in ("reference_image", "current_image"):
        hashed = {
            split: {_bytes_sha256(row[image_key], file_hash_cache)
                    for row in split_rows}
            for split, split_rows in splits.items()
        }
        overlap[f"{image_key}_sha256"] = sorted(
            hashed["train"] & hashed["validation"])
    instruction_hashes = {
        split: {_text_sha256(str(row["instruction"])) for row in split_rows}
        for split, split_rows in splits.items()
    }
    trajectory_hashes = {
        split: {_trajectory_sha256(row["sample"]) for row in split_rows}
        for split, split_rows in splits.items()
    }
    overlap["instruction_sha256"] = sorted(
        instruction_hashes["train"] & instruction_hashes["validation"])
    overlap["trajectory_window_sha256"] = sorted(
        trajectory_hashes["train"] & trajectory_hashes["validation"])
    blocking_keys = (
        "source_scene", "source_task_id", "slow_anchor_id",
        "reference_image_sha256", "current_image_sha256",
    )
    return {
        "schema": "action_head_split_audit_v1",
        "action_head_as_tool_version": action_head_as_tool_version,
        **(protocol.protocol_identity() if protocol is not None else {}),
        "manifest": str(path.resolve()),
        "manifest_sha256": _bytes_sha256(str(path), file_hash_cache),
        "counts": dict(Counter(row.get("split") for row in rows)),
        "unique_scenes": {key: len(values(key, "source_scene")) for key in splits},
        "cross_split_overlap_count": {key: len(value) for key, value in overlap.items()},
        "cross_split_overlap": overlap,
        "blocking_keys": list(blocking_keys),
        "notes": {
            "instruction_sha256": "reported only; generic navigation templates may repeat across disjoint scenes",
            "trajectory_window_sha256": "reported only; identical short delta windows may occur on distinct physical paths",
        },
        "scene_disjoint_ok": all(not overlap[key] for key in blocking_keys),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--action-head-as-tool-version", choices=("v1", "v2", "v3", "v4.3", "v6"), default="v1")
    args = parser.parse_args()
    report = audit_manifest(args.manifest, args.action_head_as_tool_version)
    output = args.output or args.manifest.parent / "split_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["scene_disjoint_ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
