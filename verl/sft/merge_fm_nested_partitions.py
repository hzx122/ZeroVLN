#!/usr/bin/env python3
"""Merge independent trajectory partitions of one FM nested-data shard."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                 for row in rows), encoding="utf-8")
    os.replace(temporary, path)


def merge_partitions(shard_root: Path, partition_count: int) -> dict:
    if partition_count <= 1:
        raise ValueError("partition_count must exceed one")
    roots = [shard_root / f"bundles_part_{index:02d}"
             for index in range(partition_count)]
    provenance = []
    manifests, audits, decisions = [], [], []
    for index, root in enumerate(roots):
        current = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
        if current.get("task_partition") != {"index": index,
                                               "count": partition_count}:
            raise ValueError(f"wrong task partition provenance: {root}")
        if current.get('action_head_as_tool_version') == 'v6':
            from action_head import protocol_v6
            protocol_v6.assert_protocol_identity(current)
            if provenance and current.get('beam_contract') != provenance[0].get('beam_contract'):
                raise ValueError('v6 partitions have different beam contracts')
        provenance.append(current)
        manifests.extend(_read_jsonl(root / "manifest.jsonl"))
        audits.extend(_read_jsonl(root / "anchor_audit.jsonl"))
        decision_path = root / "slow_decisions.jsonl"
        if decision_path.is_file():
            decisions.extend(_read_jsonl(decision_path))

    manifests.sort(key=lambda row: (
        str(row.get("source_scene")), str(row.get("source_task_id")),
        int(row.get("option_sequence_index") or 0),
        float(row.get("fast_window_arc_m") or 0.0)))
    audits.sort(key=lambda row: str(row.get("slow_anchor_id")))
    decisions.sort(key=lambda row: (
        str(row.get("source_scene")), str(row.get("source_task_id")),
        int(row.get("option_sequence_index") or 0)))

    output = shard_root / "bundles"
    output.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(output / "manifest.jsonl", manifests)
    _atomic_jsonl(output / "anchor_audit.jsonl", audits)
    if decisions:
        _atomic_jsonl(output / "slow_decisions.jsonl", decisions)

    merged = dict(provenance[0])
    merged["task_partition"] = {"index": "merged", "count": partition_count}
    merged["partition_sources"] = [str(root.resolve()) for root in roots]
    for key in ("anchors_attempted", "anchors_accepted", "slow_decisions",
                "terminal_decisions", "samples"):
        merged[key] = sum(int(value.get(key, 0)) for value in provenance)
    provenance_path = output / "provenance.json"
    temporary = provenance_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, provenance_path)
    print(json.dumps({"output": str(output.resolve()),
                      "partitions": partition_count,
                      "samples": len(manifests),
                      "anchors": len(audits)}, ensure_ascii=False), flush=True)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--partition-count", type=int, default=2)
    args = parser.parse_args()
    merge_partitions(args.shard_root.resolve(), args.partition_count)


if __name__ == "__main__":
    main()
