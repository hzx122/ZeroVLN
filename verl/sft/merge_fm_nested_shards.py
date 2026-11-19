#!/usr/bin/env python3
"""Merge independently materialized FM nested-data shards and verify training schema."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from action_head import protocol as ACTION_HEAD_PROTOCOL


REQUIRED = {
    "sample", "split", "slow_anchor_id", "source_scene", "source_task_id",
    "slow_messages", "slow_completion", "reference_image", "current_image",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--action-head-as-tool-version", choices=("v1", "v2", "v3", "v4.3", "v6"), default="v1")
    args = parser.parse_args()
    if args.shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    if args.action_head_as_tool_version == 'v6':
        from sft.stream_fm_nested_shards import merge_v6
        merge_v6(args.data_root, args.shard_count)
        return
    protocol = ACTION_HEAD_PROTOCOL.protocol_module(args.action_head_as_tool_version)

    rows, slow_decisions, sources = [], [], []
    beam_contract = None
    target_shapes, valid_steps = Counter(), Counter()
    for index in range(args.shard_count):
        shard = args.data_root / "shards" / f"shard_{index:03d}"
        manifest = shard / "bundles" / "manifest.jsonl"
        provenance = shard / "bundles" / "provenance.json"
        timing = shard / "timing.json"
        plan_summary = shard / "nested_plan.summary.json"
        if not manifest.is_file() or not provenance.is_file():
            raise FileNotFoundError(f"incomplete shard {index}: {shard}")
        shard_rows = [json.loads(line) for line in manifest.read_text().splitlines()
                      if line.strip()]
        slow_path = shard / "bundles" / "slow_decisions.jsonl"
        shard_slow = ([json.loads(line) for line in slow_path.read_text().splitlines()
                       if line.strip()] if slow_path.is_file() else [])
        provenance_value = json.loads(provenance.read_text())
        if protocol is not None:
            protocol.assert_protocol_identity(provenance_value)
            if args.action_head_as_tool_version == 'v6':
                current_contract = provenance_value.get('beam_contract')
                if not current_contract or current_contract.get('direction_filter') is not False:
                    raise ValueError('v6 shard is not no-direction-filter beam data')
                if beam_contract is not None and beam_contract != current_contract:
                    raise ValueError('v6 shards have different beam contracts')
                beam_contract = current_contract
            expected = ("fm_nested_bundle_provenance_v4"
                        if args.action_head_as_tool_version in ("v3", "v4.3", "v6")
                        else "fm_nested_bundle_provenance_v3")
            if provenance_value.get("schema") != expected:
                raise ValueError(
                    f"shard {index} has wrong {args.action_head_as_tool_version} "
                    "provenance schema")
        for row in shard_rows:
            missing = REQUIRED - row.keys()
            if missing:
                raise ValueError(f"shard {index} row is missing {sorted(missing)}")
            if protocol is not None:
                protocol.assert_protocol_identity(row)
                if args.action_head_as_tool_version == 'v6':
                    from sft.golden_context_beam import validate_context
                    validate_context(row['slow_messages'], row['trajectory_map_path'])
                if row.get("schema") != protocol.BUNDLE_SCHEMA:
                    raise ValueError(
                        f"shard {index} has wrong versioned bundle row")
            sample = Path(row["sample"])
            if not sample.is_file():
                raise FileNotFoundError(sample)
            with np.load(sample) as payload:
                shape = tuple(payload["target_actions"].shape)
                mask = payload["action_mask"]
                if shape != (16, 3) or tuple(mask.shape) != (16,):
                    raise ValueError(
                        f"training sample contract mismatch: {sample} {shape}/{mask.shape}")
                target_shapes[str(shape)] += 1
                valid_steps[int(mask.sum())] += 1
        rows.extend(shard_rows)
        for row in shard_slow:
            if protocol is None:
                raise ValueError(f"shard {index} has versioned slow decisions in a v1 merge")
            protocol.assert_protocol_identity(row)
            if row.get("schema") != protocol.SLOW_DECISION_SCHEMA:
                raise ValueError(f"shard {index} has invalid slow-decision schema")
            if row.get("decision_type") not in ("run_action_head", "stop"):
                raise ValueError(f"shard {index} has invalid slow decision type")
        slow_decisions.extend(shard_slow)
        sources.append({
            "shard_index": index,
            "manifest": str(manifest.resolve()),
            "manifest_sha256": sha256(manifest),
            "samples": len(shard_rows),
            "slow_decisions": len(shard_slow),
            "provenance": str(provenance.resolve()),
            "provenance_sha256": sha256(provenance),
            "timing": json.loads(timing.read_text()) if timing.is_file() else None,
            "plan_summary": json.loads(plan_summary.read_text()),
        })
    anchor_ids = [row["slow_anchor_id"] for row in rows]
    context_audit = None
    if args.action_head_as_tool_version in ("v3", "v4.3", "v6"):
        anchor_rows = {}
        for row in rows:
            anchor_rows.setdefault(row["slow_anchor_id"], row)
        expected_prior = {
            anchor for anchor, row in anchor_rows.items()
            if int(row.get("option_sequence_index") or 0) > 0
        }
        def has_prior_alignment(row):
            for message in row.get("slow_messages") or []:
                if message.get("role") != "user":
                    continue
                content = message.get("content") or []
                if not isinstance(content, list):
                    continue
                for part in content:
                    text = part.get("text", "") if isinstance(part, dict) else ""
                    if (text.startswith("[Tool result]")
                            and '"direction_alignment"' in text):
                        return True
            return False

        with_prior_result = {
            anchor for anchor, row in anchor_rows.items()
            if has_prior_alignment(row)
        }
        missing = sorted(expected_prior - with_prior_result)
        if missing:
            raise ValueError(
                "v3 causal contexts lost preceding direction_alignment: "
                f"{missing[:10]}")
        context_audit = {
            "unique_anchor_contexts": len(anchor_rows),
            "prior_tool_result_expected": len(expected_prior),
            "prior_direction_alignment_present": len(with_prior_result),
            "missing_prior_direction_alignment": len(missing),
        }
    # One anchor legitimately owns multiple windows; it must belong to one shard only.
    anchor_shards: dict[str, set[int]] = {}
    for source in sources:
        manifest_rows = [json.loads(line) for line in Path(source["manifest"]).read_text().splitlines()
                         if line.strip()]
        for row in manifest_rows:
            anchor_shards.setdefault(row["slow_anchor_id"], set()).add(source["shard_index"])
    crossed = sorted(key for key, value in anchor_shards.items() if len(value) != 1)
    if crossed:
        raise ValueError(f"anchors cross shards: {crossed[:10]}")

    rows.sort(key=lambda row: (
        row["split"], row["source_scene"], row["source_task_id"],
        row["slow_anchor_id"], float(row["fast_window_arc_m"])))
    sampling_keys = (
        "slow_stride_m", "slow_coverage_m", "fast_stride_min_m",
        "fast_stride_max_m", "sampling_seed", "spacing_selection",
    )
    sampling = {}
    for key in sampling_keys:
        values = {json.dumps(source["plan_summary"].get(key), sort_keys=True)
                  for source in sources}
        if len(values) != 1:
            raise ValueError(f"inconsistent shard sampling field {key}: {values}")
        sampling[key] = sources[0]["plan_summary"].get(key)
    sampling.update({"action_step_m": 0.1, "action_horizon_steps": 16})
    output = args.data_root / "manifest.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                 for row in rows), encoding="utf-8")
    os.replace(temporary, output)
    if protocol is not None and slow_decisions:
        slow_decisions.sort(key=lambda row: (
            row["split"], row["source_scene"], row["source_task_id"],
            int(row.get("option_sequence_index") or 0)))
        slow_output = args.data_root / "slow_decisions.jsonl"
        slow_temporary = slow_output.with_suffix(".jsonl.tmp")
        slow_temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n"
                    for row in slow_decisions), encoding="utf-8")
        os.replace(slow_temporary, slow_output)
    report = {
        "schema": "fm_nested_shard_merge_v1",
        "action_head_as_tool_version": args.action_head_as_tool_version,
        **(protocol.protocol_identity() if protocol is not None else {}),
        "manifest": str(output.resolve()),
        "manifest_sha256": sha256(output),
        "shard_count": args.shard_count,
        "beam_contract": beam_contract,
        "shards": sources,
        "samples": len(rows),
        "slow_decisions": len(slow_decisions),
        "terminal_decisions": sum(
            row.get("decision_type") == "stop" for row in slow_decisions),
        "anchors": len(set(anchor_ids)),
        "scenes": len({row["source_scene"] for row in rows}),
        "split_samples": dict(Counter(row["split"] for row in rows)),
        "target_shapes": dict(target_shapes),
        "valid_action_steps": dict(sorted(valid_steps.items())),
        "training_ready": bool(rows and set(target_shapes) == {"(16, 3)"}),
        "sampling": sampling,
        "v3_causal_context_audit": context_audit,
    }
    (args.data_root / "provenance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
