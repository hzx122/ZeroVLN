#!/usr/bin/env python3
"""Merge disjoint v3 correction-builder lanes at whole-anchor granularity."""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

from correction_sampling import balanced_groups, is_dual, type_stats, CONTRACT


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                 for row in rows), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lanes", type=int)
    parser.add_argument("--input-dir", action="append", type=Path, default=[])
    parser.add_argument("--max-correction-anchors", required=True, type=int)
    parser.add_argument("--bootstrap-round", required=True, type=int)
    args = parser.parse_args()
    if (args.lanes is None) == (not args.input_dir):
        parser.error("specify exactly one of --lanes or one-or-more --input-dir")
    if args.lanes is not None and args.lanes <= 0:
        parser.error("--lanes must be positive")
    all_rows, audits, lane_reports = [], [], []
    inputs = (list(args.input_dir) if args.input_dir else
              [args.output_dir / "lanes" / f"lane_{index:02d}"
               for index in range(args.lanes)])
    for lane in inputs:
        all_rows.extend(read_jsonl(lane / "manifest.jsonl"))
        audits.extend(read_jsonl(lane / "correction_audit.jsonl"))
        lane_reports.append(json.loads((lane / "provenance.json").read_text()))
    versions = {report["action_head_as_tool_version"] for report in lane_reports}
    if len(versions) != 1 or not versions <= {"v3", "v4.3", "v6"}:
        raise ValueError(f"inconsistent correction protocols: {versions}")
    version = versions.pop()
    if any(row.get("action_head_as_tool_version") != version for row in all_rows):
        raise ValueError("manifest protocol does not match lane reports")
    grouped: dict[str, list[dict]] = collections.OrderedDict()
    for row in all_rows:
        grouped.setdefault(row["slow_anchor_id"], []).append(row)
    if version == 'v6':
        # The local-teacher contract is independent of the legacy v4.3 gate.
        # Keep whole anchors here; the v6 mixture applies the round/type budget.
        contracts = {'collision_relabel': 'v6_collision_task_oracle_v1',
                     'direction_mismatch_relabel': 'v6_direction_mismatch_task_oracle_v1',
                     'golden_route_relabel': 'v6_corrected_golden_slow_anchor_v1',
                     'execution_relabel': 'v6_local_teacher_checked_v1',
                     'recovery': 'v6_local_teacher_checked_v1'}
        if any(row.get('data_contract') != 'v6_online_corrections_v1'
               or row.get('decision_filter') != contracts.get(row.get('sample_type'))
               for row in all_rows):
            raise ValueError('unchecked v6 correction in merge input')
        filters = {row['decision_filter'] for row in all_rows}
        if len(filters) > 1 and not filters <= {
                'v6_collision_task_oracle_v1', 'v6_direction_mismatch_task_oracle_v1',
                'v6_corrected_golden_slow_anchor_v1'}:
            raise ValueError('v6 correction lanes mix incompatible selection contracts')
        dual = True
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'verl'))
        from sft.v6_bootstrap_mixture import select_anchors
        root_groups = [group for group in grouped.values()
                       if group[0].get('sample_type') != 'golden_route_relabel']
        extra_groups = [group for group in grouped.values()
                        if group[0].get('sample_type') == 'golden_route_relabel']
        selected_groups = select_anchors(root_groups, len(all_rows), args.bootstrap_round)
        selected_groups = selected_groups[:args.max_correction_anchors]
        remaining = args.max_correction_anchors - len(selected_groups)
        if remaining:
            selected_groups.extend(select_anchors(extra_groups, len(all_rows),
                args.bootstrap_round)[:remaining])
    else:
        dual = is_dual(all_rows)
        selected_groups = balanced_groups(all_rows, seed=args.bootstrap_round)[:args.max_correction_anchors]
    selected_ids = [group[0]["slow_anchor_id"] for group in selected_groups]
    rows = [row for group in selected_groups for row in group]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "manifest.jsonl", rows)
    write_jsonl(args.output_dir / "correction_audit.jsonl", audits)
    if version == 'v6':
        for name in ('high_level_audit.jsonl', 'high_level_manifest.jsonl'):
            write_jsonl(args.output_dir / name,
                        [row for lane in inputs for row in read_jsonl(lane / name)])
    report = {
        "schema": "action_head_v3_bootstrap_correction_merge_v1",
        "action_head_as_tool_version": version,
        "bootstrap_round": args.bootstrap_round,
        "selection_unit": "complete_condition_anchor" if dual else "complete_noncollision_option_endpoint",
        "oracle": (("task-goal physical oracle" if rows and rows[0].get('sample_type') in
                    {'collision_relabel', 'direction_mismatch_relabel', 'golden_route_relabel'}
                    else "locally checked v6 teacher") if version == 'v6' else
                   "A* from condition AH start to task goal" if dual else "A* from actual option endpoint to task goal"),
        "selection_contract": ("v6_physical_failure_pair_v1" if version == 'v6' and len({r['decision_filter'] for r in rows}) > 1 else
                               rows[0]['decision_filter'] if version == 'v6' and rows else
                               "v6_local_teacher_checked_v1" if version == 'v6' else
                               CONTRACT if dual else "legacy"),
        "candidate_type_counts": type_stats(all_rows),
        "selected_type_counts": type_stats(rows),
        "collision_recovery_included": version == 'v6' and any(
            row.get('sample_type') == 'recovery' for row in rows),
        "lanes": args.lanes,
        "input_count": len(inputs),
        "inputs": [str(path.resolve()) for path in inputs],
        "candidate_anchors_built": len(grouped),
        "anchors_selected": len(selected_ids),
        "samples_selected": len(rows),
        "fast_windows": "all available among R1-R4",
        "lane_reports": lane_reports,
    }
    if version == 'v6':
        report['training_ready'] = bool(rows)
    (args.output_dir / "provenance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "DONE").write_text(
        json.dumps({"ok": True, "anchors": len(selected_ids),
                    "samples": len(rows)}) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
