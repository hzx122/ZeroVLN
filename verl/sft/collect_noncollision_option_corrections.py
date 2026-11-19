#!/usr/bin/env python3
"""Build corrective supervision, preserving legacy v3 and dual-anchor v4.3.

v4.3 independently labels the failed call's start and the next AH's actual
start, optionally after successful, state-continuous turns. Each episode
contributes at most one anchor of each type. Privileged traces are audit-only.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

import collect_action_head_recovery as CR
from action_head import protocol_v3 as V3
from action_head import protocol as PROTOCOL
from sft.build_fm_nested_bundles import save_npz


def saved_rollout(source: Path, case: dict) -> Path:
    stem = hashlib.sha256(
        f"{case['scene']}:{case['episode_id']}".encode()).hexdigest()[:16]
    return source / "rollouts" / f"{stem}.json"


def execution_failed(call: dict) -> bool:
    result = call.get("result") or {}
    prefixes = (call.get("action_trace") or {}).get("prefixes") or []
    return (bool(result.get("collision")) or bool(result.get("blocked")) or
            any(bool(row.get("collided")) or bool(row.get("blocked"))
                for row in prefixes))


def selection_contract(protocol, teacher_audit=None, sample_type=None) -> str:
    if protocol.ACTION_HEAD_AS_TOOL_VERSION == "v6":
        from collect_v6_option_corrections import (
            SELECTION_CONTRACT, COLLISION_CONTRACT, DIRECTION_CONTRACT, GOLDEN_EXTRA_CONTRACT)
        if (teacher_audit or {}).get("mode") == "physical_oracle":
            return (GOLDEN_EXTRA_CONTRACT if sample_type == "golden_route_relabel" else
                    DIRECTION_CONTRACT if sample_type == "direction_mismatch_relabel"
                    else COLLISION_CONTRACT)
        return SELECTION_CONTRACT
    return ("v43_dual_anchor_turn_aligned_v1"
            if protocol.ACTION_HEAD_AS_TOOL_VERSION == "v4.3"
            else "noncollision_stitched_option_mismatch_v2")


def planned_direction(arguments: dict, protocol) -> str:
    if protocol.ACTION_HEAD_AS_TOOL_VERSION in {"v4.3", "v6"}:
        return protocol.planned_option_direction(arguments)
    return CR.CE.contract_direction(arguments)


def endpoint_direction(endpoint: dict, protocol) -> str:
    if protocol.ACTION_HEAD_AS_TOOL_VERSION in {"v4.3", "v6"}:
        return protocol.actual_option_direction(endpoint)
    return CR.CE.direction_from_endpoint(
        float(endpoint.get("x_m") or 0.0), float(endpoint.get("y_m") or 0.0),
        float(endpoint.get("yaw_deg") or 0.0))


def stitched_direction(call: dict, protocol=V3) -> tuple[str, dict | None]:
    endpoint = (call.get("action_trace") or {}).get("option_actual_endpoint")
    if not endpoint:
        return "degenerate", None
    canonical = dict(endpoint)
    canonical["yaw_deg"] = CR.CE.wrap_degrees(float(endpoint.get("yaw_deg") or 0.0))
    return endpoint_direction(canonical, protocol), canonical


def oracle_prefix_direction(route: dict, protocol=V3) -> tuple[str, dict]:
    start = CR.pose_at(route, 0.0)
    end = CR.pose_at(route, min(0.4, float(route["length_m"])))
    x, y, yaw = CR.CE.FMA.world_to_chunk_pose(
        (end[0], end[1]), end[2], (start[0], start[1]), start[2])
    endpoint = {"x_m": float(x), "y_m": float(y),
                "yaw_deg": CR.CE.wrap_degrees(math.degrees(float(yaw))),
                "path_m": min(0.4, float(route["length_m"]))}
    return endpoint_direction(endpoint, protocol), endpoint


def episode_candidates(calls: list[dict], counters, protocol=V3):
    """Yield adjacent executable failure/correction pairs, never bridging turns."""
    for index in range(1, len(calls)):
        previous, current = calls[index - 1], calls[index]
        if previous.get("name") != "run_action_head" or current.get("name") != "run_action_head":
            continue
        if execution_failed(previous):
            counters["excluded_collision_or_blocked"] += 1
            continue
        expected = planned_direction(previous.get("arguments") or {}, protocol)
        actual, endpoint = stitched_direction(previous, protocol)
        if expected not in {"left", "right", "straight"}:
            counters["previous_direction_unspecified"] += 1
            continue
        if actual in {"degenerate", "unknown"} or actual == expected:
            counters["previous_option_not_mismatch"] += 1
            continue
        yield index, previous, current, expected, actual, endpoint


def build_episode_rows(*, calls, counters, case, source_path, protocol=V3, **kwargs):
    """Accept at most one anchor; semantic rejection may try the next candidate.

    Constructor exceptions still propagate to the existing episode error audit;
    they are not treated as a semantic rejection or retried here.
    """
    if protocol.ACTION_HEAD_AS_TOOL_VERSION == "v6":
        from collect_v6_option_corrections import build_episode_rows as build_v6
        return build_v6(calls=calls, counters=counters, case=case,
            source_path=source_path, protocol=protocol, **kwargs)
    if protocol.ACTION_HEAD_AS_TOOL_VERSION == "v4.3":
        return build_dual_episode_rows(calls=calls, counters=counters, case=case,
            source_path=source_path, protocol=protocol, **kwargs)
    audits = []
    for index, previous, current, expected, actual, endpoint in episode_candidates(calls, counters, protocol):
        if not audits:
            counters["candidate_episodes"] += 1
        counters["candidate_attempts"] += 1
        samples, audit = build_rows(case=case, previous=previous, current=current,
            candidate_index=index, protocol=protocol, **kwargs)
        audit.update({"scene": case["scene"], "episode_id": case["episode_id"],
            "previous_contract_direction": expected, "previous_actual_direction": actual,
            "previous_actual_endpoint": endpoint, "source_rollout": str(source_path),
            "selection_contract": selection_contract(protocol)})
        audits.append(audit)
        if samples:
            counters["anchors_accepted"] += 1
            return samples, audits
        if protocol.ACTION_HEAD_AS_TOOL_VERSION != "v4.3":
            break
    return [], audits


def world_pose(value):
    """Require finite physical poses; never substitute commanded endpoints."""
    try:
        xy = value["world_xy"]
        pose = (float(xy[0]), float(xy[1]), float(value["world_yaw_deg"]))
        return pose if all(math.isfinite(v) for v in pose) else None
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def call_pose(call, before=True):
    trace = call.get("action_trace") or {}
    key = "before_world_pose" if before else "after_world_pose"
    pose = world_pose(trace.get(key))
    prefixes = trace.get("prefixes") or []
    if pose is None and prefixes:
        pose = world_pose(prefixes[0 if before else -1].get(key))
    return pose


def same_pose(left, right):
    return (left is not None and right is not None
            and math.hypot(left[0] - right[0], left[1] - right[1]) <= 1e-3
            and abs(CR.CE.wrap_degrees(left[2] - right[2])) <= 0.1)


def next_recovery_call(calls, trigger_index, protocol_version="v4.3"):
    """Only bridge successful in-place turns; enforce every physical boundary."""
    pose = call_pose(calls[trigger_index], before=False)
    turns = []
    if pose is None:
        return None, turns, "missing_trigger_end_pose"
    for index in range(trigger_index + 1, len(calls)):
        call = calls[index]
        if call.get("name") == "run_action_head":
            if not same_pose(pose, call_pose(call)):
                return None, turns, "recovery_state_discontinuity"
            return index, turns, None
        if call.get("name") != "turn":
            return None, turns, "recovery_interrupted_by_other_decision"
        result = call.get("result") or {}
        arguments, _ = PROTOCOL.validate_turn_arguments(call.get("arguments") or {}, protocol_version)
        if (arguments is None or call.get("ok") is not True or result.get("ok") is not True
                or result.get("return_reason") in {"rejected", "error", "episode_ended"}
                or execution_failed(call)):
            return None, turns, "recovery_turn_not_successful"
        trace = call.get("execution_state_trace") or {}
        before, after = world_pose(trace.get("before_world_pose")), world_pose(trace.get("after_world_pose"))
        if not same_pose(pose, before) or after is None:
            return None, turns, "recovery_turn_state_missing_or_discontinuous"
        if math.hypot(after[0] - before[0], after[1] - before[1]) > 1e-3:
            return None, turns, "recovery_turn_translated"
        turns.append({"call_index": index, "arguments": call.get("arguments"),
                      "before_world_pose": list(before), "after_world_pose": list(after)})
        pose = after
    return None, turns, "no_following_action_head"


def build_dual_episode_rows(*, calls, counters, case, source_path, protocol,
                            max_anchors=2, **kwargs):
    rows, audits, accepted, used_conditions = [], [], set(), set()
    episode_counted, counted_types = False, set()
    for trigger_index, trigger in enumerate(calls):
        if len(accepted) >= max_anchors:
            break
        if trigger.get("name") != "run_action_head":
            continue
        if execution_failed(trigger):
            counters["excluded_collision_or_blocked"] += 1
            continue
        expected = planned_direction(trigger.get("arguments") or {}, protocol)
        actual, endpoint = stitched_direction(trigger, protocol)
        if expected not in {"left", "right", "straight"}:
            counters["previous_direction_unspecified"] += 1
            continue
        if actual in {"degenerate", "unknown"} or actual == expected:
            counters["previous_option_not_mismatch"] += 1
            continue
        if (call_pose(trigger) is None or call_pose(trigger, before=False) is None
                or (trigger.get("action_trace") or {}).get("option_endpoint_source") != "world_pose"):
            counters["missing_physical_trigger_trace"] += 1
            continue
        if not episode_counted:
            counters["candidate_episodes"] += 1
            episode_counted = True
        recovery_index, turns, recovery_error = next_recovery_call(calls, trigger_index)
        for kind, condition_index in (("execution_relabel", trigger_index), ("recovery", recovery_index)):
            if len(accepted) >= max_anchors:
                break
            if kind in accepted:
                continue
            if kind not in counted_types:
                counters[f"{kind}_candidate_episodes"] += 1
                counted_types.add(kind)
            audit = {"scene": case["scene"], "episode_id": case["episode_id"],
                "sample_type": kind, "trigger_call_index": trigger_index,
                "condition_call_index": condition_index,
                "previous_contract_direction": expected, "previous_actual_direction": actual,
                "previous_actual_endpoint": endpoint, "source_rollout": str(source_path),
                "intervening_turns": turns if kind == "recovery" else [],
                "selection_contract": selection_contract(protocol)}
            if condition_index is None or condition_index in used_conditions:
                audits.append({**audit, "accepted": False,
                    "reason": recovery_error if condition_index is None else "duplicate_condition_anchor"})
                continue
            counters["candidate_attempts"] += 1
            counters[f"{kind}_candidate_attempts"] += 1
            samples, result = build_rows(case=case, previous=trigger,
                current=calls[condition_index], candidate_index=condition_index,
                sample_type=kind, trigger_index=trigger_index,
                intervening_turns=audit["intervening_turns"], protocol=protocol, **kwargs)
            audits.append({**result, **audit})
            if samples:
                rows.extend(samples)
                accepted.add(kind)
                used_conditions.add(condition_index)
        # Exceptions deliberately propagate: an infrastructure failure is not
        # an oracle rejection and must not be disguised as a later acceptance.
    if rows:
        counters["episodes_accepted"] += 1
        counters["anchors_accepted"] += len(accepted)
        for kind in accepted:
            counters[f"{kind}_anchors_accepted"] += 1
            counters[f"{kind}_windows_accepted"] += sum(row["sample_type"] == kind for row in rows)
        if any(a.get("accepted") and a["sample_type"] == "recovery" and a["intervening_turns"] for a in audits):
            counters["recovery_cross_turn_anchors_accepted"] += 1
    return rows, audits


def available_windows(route: dict) -> list[tuple[float, tuple]]:
    """Build the consecutive fixed replan windows that have valid targets."""
    windows = []
    route_length_m = float(route["length_m"])
    for offset in CR.OFFSETS_M:
        if offset >= route_length_m - 1e-6:
            break
        suffix = CR.route_suffix(route, offset)
        if suffix is None or not np.asarray(suffix[2]).any():
            break
        windows.append((offset, suffix))
    return windows


def visual_tokens_batch(evaluator, image_paths: list[Path]) -> list[np.ndarray]:
    """Reuse the validated batched vision path from full-data construction."""
    policy = evaluator.runtime.tool_executor.runtime.policy
    encoder = getattr(policy.visual_encoder, "encode_batch", None)
    if encoder is None:
        return [CR.visual_tokens(evaluator, path) for path in image_paths]
    rgbs = [np.asarray(Image.open(path).convert("RGB")) for path in image_paths]
    encoded = encoder(rgbs)
    if len(encoded) != len(image_paths):
        raise RuntimeError("visual batch returned the wrong number of frames")
    return [value.float().cpu().numpy().astype(np.float32) for value in encoded]


def build_rows(*, evaluator, output: Path, case: dict, task, dataset,
               previous: dict, current: dict, candidate_index: int,
               render_url: str, preloaded: set[str],
               bootstrap_round: int,
               collection_iteration: int,
               npz_compress_level: int, protocol=V3, sample_type=None,
               trigger_index=None, intervening_turns=None,
               prepared_route=None, teacher_audit=None) -> tuple[list[dict], dict]:
    is_v6 = protocol.ACTION_HEAD_AS_TOOL_VERSION == "v6"
    if is_v6 and (prepared_route is None or not teacher_audit or
                  teacher_audit.get("approved") is not True):
        raise ValueError("v6 requires a locally approved, replay-checked route")
    if is_v6:
        from collect_v6_option_corrections import condition_binding, route_digest
        if (teacher_audit.get("mode") not in {"semantic_judge", "reviewed", "physical_oracle"}
                or teacher_audit.get("condition_sha256") != condition_binding(
                    case, current, candidate_index, call_pose(current))
                or teacher_audit.get("route_sha256") != route_digest(prepared_route)):
            raise ValueError("v6 teacher approval is not bound to this condition and route")
    if not is_v6 and (prepared_route is not None or teacher_audit is not None):
        raise ValueError("prepared local teachers are v6-only")
    previous_trace = previous.get("action_trace") or {}
    current_trace = current.get("action_trace") or {}
    previous_prefixes = previous_trace.get("prefixes") or []
    audit = {
        "candidate_index": candidate_index,
        "previous_arguments": previous.get("arguments"),
        "correction_arguments": current.get("arguments"),
        "previous_result": previous.get("result"),
    }
    if not previous_prefixes and not is_v6:
        return [], {**audit, "accepted": False, "reason": "missing_previous_execution"}
    if sample_type is not None:
        if sample_type not in {"execution_relabel", "recovery", "collision_relabel",
                               "direction_mismatch_relabel", "golden_route_relabel"}:
            raise ValueError(f"unsupported sample type: {sample_type}")
        if current.get("ok") is False and sample_type != "collision_relabel":
            return [], {**audit, "accepted": False, "reason": "condition_action_head_not_executed"}
        start = call_pose(current)
        if start is None:
            return [], {**audit, "accepted": False, "reason": "missing_condition_start_pose"}
        end_pose = {"world_xy": start[:2], "world_yaw_deg": start[2]}
    else:
        end_pose = previous_prefixes[-1].get("after_world_pose") or {}
    world_xy, yaw_deg = end_pose.get("world_xy"), end_pose.get("world_yaw_deg")
    if world_xy is None or yaw_deg is None:
        return [], {**audit, "accepted": False, "reason": "missing_actual_end_pose"}

    if is_v6:
        route, error = prepared_route, None
    else:
        route, error = CR.oracle_route(
            dataset, case["scene"], (float(world_xy[0]), float(world_xy[1])),
            math.radians(float(yaw_deg)), tuple(task.goal_xy))
    if route is None:
        return [], {**audit, "accepted": False, "reason": error}
    windows = available_windows(route)
    if not windows:
        return [], {**audit, "accepted": False,
                    "reason": "oracle_route_has_no_actionable_window",
                    "oracle_length_m": route["length_m"]}
    correction_direction = planned_direction(current.get("arguments") or {}, protocol)
    oracle_direction, oracle_endpoint = oracle_prefix_direction(route, protocol)
    audit.update({
        "correction_direction": correction_direction,
        "oracle_prefix_direction": oracle_direction,
        "oracle_prefix_endpoint": oracle_endpoint,
        "oracle_length_m": route["length_m"],
        "oracle_smoothing": route["smoothing"],
        "oracle_execution_replay": route["execution_replay"],
        "correction_start_world_pose": [world_xy[0], world_xy[1], yaw_deg],
    })
    if not is_v6 and correction_direction not in {"left", "right", "straight"}:
        return [], {**audit, "accepted": False,
                    "reason": "correction_toolcall_direction_unspecified"}
    if not is_v6 and oracle_direction != correction_direction:
        return [], {**audit, "accepted": False,
                    "reason": "correction_toolcall_oracle_direction_mismatch"}

    query_path = (current_trace.get("query_artifact") or {}).get("archive_path")
    slow_turn = current.get("turn") or {}
    slow_messages = slow_turn.get("slow_messages")
    slow_completion = str(slow_turn.get("assistant") or "").strip()
    if not is_v6 and (not query_path or not Path(query_path).is_file()):
        return [], {**audit, "accepted": False, "reason": "missing_query_hidden"}
    if not slow_messages or not slow_completion.endswith("</tool_call>"):
        return [], {**audit, "accepted": False, "reason": "missing_slow_prefix"}

    anchor_kind = ("golden-route" if sample_type == "golden_route_relabel" else
                   "direction-mismatch" if sample_type == "direction_mismatch_relabel"
                   else "collision" if sample_type == "collision_relabel" else "noncollision")
    anchor_key = (f"{anchor_kind}:r{bootstrap_round}:i{collection_iteration}:{case['scene']}:"
                  f"{case['episode_id']}:{candidate_index}")
    if sample_type is not None:
        anchor_key += f":{sample_type}"
    anchor = hashlib.sha256(anchor_key.encode()).hexdigest()[:16]
    frame_dir = output / "frames" / anchor
    poses = [suffix[0] for _, suffix in windows]
    images = [CR.render_pose(render_url, case["scene"], pose,
                             frame_dir / f"r{index}.jpg", preloaded)
              for index, pose in enumerate(poses, 1)]
    # v6 joint query/head training reconstructs query states from the saved
    # slow messages and completion; only legacy head-only paths need this tensor.
    query_hidden = None if is_v6 else np.load(query_path).astype(np.float32)
    persisted_messages = CR.externalize_messages(
        slow_messages, output / "slow_context_images" / anchor)
    if is_v6:
        from collect_v6_option_corrections import trajectory_map_path, validate_condition
        validate_condition(current)
        trajectory_map_path(persisted_messages)
    # R1 is both the reference and first current frame. Encode every distinct
    # frame once in a single vision forward instead of issuing 2..5 tiny
    # forwards per anchor.
    visual_tokens = visual_tokens_batch(evaluator, images)
    reference = visual_tokens[0]
    rows = []
    previous_offset = 0.0
    for index, ((offset, suffix), image_path, current_visual) in enumerate(
            zip(windows, images, visual_tokens), 1):
        pose, target, mask, remaining = suffix
        sample_path = output / "samples" / f"{anchor}_r{index}.npz"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        save_npz(
            sample_path, compresslevel=npz_compress_level,
            **({"query_hidden": query_hidden} if query_hidden is not None else {}),
            reference_visual_tokens=reference,
            current_visual_tokens=current_visual,
            target_actions=target.astype(np.float32),
            action_mask=np.asarray(mask, dtype=np.bool_))
        rows.append({
            "schema": protocol.BUNDLE_SCHEMA, **protocol.protocol_identity(),
            "sample": str(sample_path.resolve()), "split": "train",
            "slow_anchor_id": f"{anchor_kind}-correction:{anchor}",
            "option_sequence_index": None, "slow_anchor_arc_m": None,
            "slow_anchor_remaining_arc_m": route["length_m"],
            "slow_coverage_m": min(1.6, float(route["length_m"])),
            "option_coverage_truncated": route["length_m"] < 1.6,
            "fast_window_index": index - 1, "option_replan_index": index - 1,
            "fast_window_arc_m": offset,
            "fast_interval_from_previous_m": (
                0.0 if index == 1 else offset - previous_offset),
            "fast_sampling_seed": None, "fast_offset_from_slow_m": offset,
            "fast_remaining_arc_m": remaining,
            "source_scene": case["scene"], "source_task_id": case["episode_id"],
            "source_point_index": None,
            "source_pose_xyyaw": [pose[0], pose[1], pose[2]],
            "instruction": case["instruction"],
            "contract": current["arguments"],
            "slow_messages": persisted_messages,
            "slow_completion": slow_completion,
            "pass_rank": 1,
            "decision_filter": selection_contract(protocol, teacher_audit, sample_type),
            **({} if is_v6 else {
                "query_tail_count": int(query_hidden.shape[0]),
                "query_request_id": (current_trace.get("query_artifact") or {}).get(
                    "request_id_with_digest")}),
            "reference_image": str(images[0]), "current_image": str(image_path),
            "valid_action_steps": int(np.asarray(mask).sum()),
            "bootstrap_provenance": {
                "type": "v6_online_option_correction" if is_v6 else "noncollision_option_correction",
                "round": bootstrap_round,
                "collection_iteration": collection_iteration,
                "candidate_index": candidate_index, "r_index": index,
                "offset_m": offset,
                "selection_contract": selection_contract(protocol, teacher_audit, sample_type),
                **({"sample_type": sample_type, "trigger_call_index": trigger_index,
                    "condition_call_index": candidate_index,
                    "condition_start_world_pose": [world_xy[0], world_xy[1], yaw_deg],
                    "intervening_turns": intervening_turns or [],
                    "oracle_contract": ((("golden_route_suffix_replay_v1" if
                                         sample_type == "golden_route_relabel" else
                                         "direction_mismatch_task_goal_replay_v1" if
                                         sample_type == "direction_mismatch_relabel" else
                                         "collision_task_goal_replay_v1") if
                                        teacher_audit and teacher_audit.get("mode") == "physical_oracle"
                                        else "local_teacher_checked_route_v1") if is_v6
                                        else "task_goal_astar_plan_prefix_direction_v1")}
                   if sample_type is not None else {}),
            },
            **({"sample_type": sample_type} if sample_type is not None else {}),
        })
        previous_offset = offset
    return rows, {**audit, "accepted": True, "reason": "accepted",
                  "slow_anchor_id": f"{anchor_kind}-correction:{anchor}",
                  "samples": len(rows), "evidence_images": [str(x) for x in images]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-recovery-dir", required=True, type=Path)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--bench-url", default="http://127.0.0.1:8938")
    parser.add_argument("--render-url", default="http://127.0.0.1:7038")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-correction-anchors", type=int, default=300)
    parser.add_argument("--case-shard-index", type=int, default=0)
    parser.add_argument("--case-shard-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--action-head-as-tool-version", choices=("v3", "v4.3", "v6"), default="v3")
    parser.add_argument("--v6-teacher-mode", choices=("audit_only", "semantic_judge", "reviewed", "physical_oracle"),
                        default="audit_only")
    parser.add_argument("--v6-anchor-mode", choices=("collision-only", "failure-pair"),
                        default="collision-only")
    parser.add_argument("--v6-judge-url")
    parser.add_argument("--v6-judge-model")
    parser.add_argument("--v6-reviewed-corrections", type=Path)
    parser.add_argument("--v6-golden-extra-max", type=int, default=0)
    parser.add_argument("--v6-golden-vllm-url")
    parser.add_argument("--v6-golden-vllm-model")
    parser.add_argument("--bootstrap-round", type=int, default=1)
    parser.add_argument("--collection-iteration", type=int, default=1)
    parser.add_argument("--sage-split", default="action_head_train_p0_unified")
    parser.add_argument("--npz-compress-level", type=int, default=1)
    args = parser.parse_args()
    protocol = PROTOCOL.protocol_module(args.action_head_as_tool_version)
    is_v6 = args.action_head_as_tool_version == "v6"
    v6_kwargs = {}
    if is_v6:
        import collect_v6_option_corrections as V6
        if not 0 <= args.v6_golden_extra_max <= 3:
            parser.error("v6 golden extra max must be in [0,3]")
        if args.v6_golden_extra_max and (args.v6_teacher_mode != "physical_oracle" or
                                         not args.v6_golden_vllm_url or
                                         not args.v6_golden_vllm_model):
            parser.error("v6 golden extras require physical_oracle and a vLLM URL/model")
        if args.v6_teacher_mode == "semantic_judge" and not (args.v6_judge_url and args.v6_judge_model):
            parser.error("v6 semantic_judge requires --v6-judge-url and --v6-judge-model")
        if args.v6_teacher_mode == "reviewed" and not args.v6_reviewed_corrections:
            parser.error("v6 reviewed mode requires --v6-reviewed-corrections")
        reviews = V6.load_reviews(args.v6_reviewed_corrections)
        v6_kwargs = dict(teacher_mode=args.v6_teacher_mode, anchor_mode=args.v6_anchor_mode,
            reviews=reviews,
            judge_url=args.v6_judge_url, judge_model=args.v6_judge_model, timeout=args.timeout,
            golden_extra_max=args.v6_golden_extra_max, golden_extra_seed=args.seed,
            golden_vllm_url=args.v6_golden_vllm_url,
            golden_vllm_model=args.v6_golden_vllm_model,
            golden_query_cache=args.query_cache)
    elif (args.v6_teacher_mode != "audit_only" or args.v6_judge_url or args.v6_judge_model
          or args.v6_reviewed_corrections):
        parser.error("v6 teacher options require --action-head-as-tool-version v6")
    source, output = args.source_recovery_dir.resolve(), args.output_dir.resolve()
    base_manifest = args.base_manifest.resolve()
    if args.max_cases < 0 or args.max_correction_anchors <= 0:
        parser.error("invalid limits")
    if args.bootstrap_round <= 0 or args.collection_iteration <= 0:
        parser.error("--bootstrap-round and --collection-iteration must be positive")
    if not 0 <= args.npz_compress_level <= 9:
        parser.error("--npz-compress-level must be in [0,9]")
    if args.case_shard_count <= 0 or not 0 <= args.case_shard_index < args.case_shard_count:
        parser.error("invalid case shard")
    for path in (source / "rollouts", base_manifest):
        if not path.exists():
            parser.error(f"required input missing: {path}")
    output.mkdir(parents=True, exist_ok=True)

    cases = CR.load_training_cases(base_manifest, args.seed, args.max_cases)
    cases = cases[args.case_shard_index::args.case_shard_count]
    needs_fm_teacher = not is_v6 or args.v6_teacher_mode != "audit_only"
    dataset, tasks = CR.task_lookup(cases, args.sage_split) if needs_fm_teacher else (None, {})
    identity = CR.CE.checkpoint_identity(args.checkpoint_dir)
    evaluator = CR.CE.TracedEvaluator(SimpleNamespace(
        checkpoint=identity["action_head_dir"], backbone=args.backbone,
        query_cache=args.query_cache, bench_url=args.bench_url,
        vllm_url="http://127.0.0.1:1", timeout=args.timeout,
        policy_seed=20260827, mode="instruct", max_turns=30,
        max_tool_calls=30, keep_last_images=8,
        action_head_as_tool_version=args.action_head_as_tool_version, capture_slow_messages=True,
        query_archive=(None if is_v6 else str(source / "query_hidden")))) if needs_fm_teacher else None

    rows: list[dict] = []
    audits: list[dict] = []
    high_level_audits: list[dict] = []
    high_level_rows: list[dict] = []
    preloaded: set[str] = set()
    counters = collections.Counter()
    accepted_anchors = 0
    for case_index, case in enumerate(cases, 1):
        if accepted_anchors >= args.max_correction_anchors:
            break
        counters["cases_scanned"] += 1
        samples: list[dict] = []
        path = saved_rollout(source, case)
        if not path.is_file():
            counters["missing_rollout"] += 1
            continue
        try:
            rollout = json.loads(path.read_text(encoding="utf-8"))
            if is_v6:
                protocol.assert_protocol_identity(rollout)
            calls = CR.flatten_calls(rollout)
            if is_v6 and args.v6_teacher_mode != "physical_oracle":
                decision_audits, decision_rows = V6.high_level_records(calls, case, reviews, output,
                    initial_pose=world_pose(rollout.get("initial_info")))
                high_level_audits.extend(decision_audits)
                high_level_rows.extend(decision_rows)
            samples, episode_audits = build_episode_rows(
                calls=calls, counters=counters, source_path=path,
                evaluator=evaluator, output=output, case=case,
                task=tasks.get((case["scene"], case["episode_id"])), dataset=dataset,
                render_url=args.render_url, preloaded=preloaded,
                bootstrap_round=args.bootstrap_round,
                collection_iteration=args.collection_iteration,
                npz_compress_level=args.npz_compress_level, protocol=protocol, **v6_kwargs,
                **({"max_anchors": (args.max_correction_anchors - accepted_anchors
                    if is_v6 and args.v6_teacher_mode == "physical_oracle" else
                    min(2, args.max_correction_anchors - accepted_anchors))}
                   if args.action_head_as_tool_version in {"v4.3", "v6"} else {}))
            audits.extend(episode_audits)
            rows.extend(samples)
            if samples:
                accepted_anchors += len({row["slow_anchor_id"] for row in samples})
            print(f"[noncollision] {case_index}/{len(cases)} {case['episode_id']} "
                  f"attempts={len(episode_audits)} accepted={bool(samples)} "
                  f"anchors={accepted_anchors} samples={len(rows)}", flush=True)
        except Exception as exc:
            audits.append({
                "scene": case["scene"], "episode_id": case["episode_id"],
                "accepted": False, "reason": "constructor_exception",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(), "source_rollout": str(path),
            })
        if audits and (len(audits) % 10 == 0 or samples):
            CR.atomic_jsonl(output / "manifest.partial.jsonl", rows)
            CR.atomic_jsonl(output / "correction_audit.partial.jsonl", audits)

    manifest = output / "manifest.jsonl"
    audit_path = output / "correction_audit.jsonl"
    CR.atomic_jsonl(manifest, rows)
    CR.atomic_jsonl(audit_path, audits)
    if is_v6:
        CR.atomic_jsonl(output / "high_level_audit.jsonl", high_level_audits)
        CR.atomic_jsonl(output / "high_level_manifest.jsonl", high_level_rows)
    reason_counts = collections.Counter(row.get("reason") for row in audits)
    report = {
        "schema": "action_head_noncollision_option_correction_v2",
        **protocol.protocol_identity(),
        "bootstrap_round": args.bootstrap_round,
        "collection_iteration": args.collection_iteration,
        "selection_contract": selection_contract(protocol),
        "selection": {
            "unit": "complete_slow_toolcall_option",
            "candidate": "noncollision stitched actual direction differs from ToolCall A",
            "correction_start": ("each condition AH's actual before_world_pose"
                if args.action_head_as_tool_version == "v4.3" else "actual endpoint after ToolCall A"),
            "label": "A* route aligned with conditioning ToolCall",
            "one_anchor_per_episode": args.action_head_as_tool_version != "v4.3",
            "max_anchors_per_type_per_episode": 1,
            "sample_types": (["execution_relabel", "recovery"]
                if args.action_head_as_tool_version == "v4.3" else ["legacy_recovery"]),
            "candidate_policy": ("first_accepted" if args.action_head_as_tool_version == "v4.3"
                                 else "first_candidate_only"),
            "adjacent_action_head_only": args.action_head_as_tool_version != "v4.3",
            "turn_policy": "bridge only successful state-continuous in-place turns; never label turn",
            "direction_classifier": ("online_protocol" if args.action_head_as_tool_version == "v4.3"
                                     else "legacy_contract_direction"),
            "fast_windows": "available R1-R4 points at 0.0/0.4/0.8/1.2m",
            "collision_recovery_included": False,
        },
        "source": {"rollout_dir": str(source / "rollouts"),
                   "base_manifest": str(base_manifest),
                   "base_manifest_sha256": CR.sha256_file(base_manifest)},
        "checkpoint": identity, "counters": dict(counters),
        "audit_reason_counts": dict(reason_counts),
        "anchors_accepted": accepted_anchors, "samples": len(rows),
        "case_shard_index": args.case_shard_index,
        "case_shard_count": args.case_shard_count,
        "visual_batching": True,
        "npz_compress_level": args.npz_compress_level,
        "model_visible_additions": [],
    }
    if is_v6:
        report.update(schema="action_head_v6_online_correction_v1", data_contract=V6.DATA_CONTRACT,
            selection_contract=(V6.PHYSICAL_PAIR_CONTRACT if args.v6_teacher_mode == "physical_oracle"
                                and args.v6_anchor_mode == "failure-pair" else
                                V6.COLLISION_CONTRACT if args.v6_teacher_mode == "physical_oracle"
                                else selection_contract(protocol)),
            teacher_mode=args.v6_teacher_mode, training_ready=bool(rows),
            anchor_mode=args.v6_anchor_mode,
            golden_extra_max=args.v6_golden_extra_max,
            high_level_labels=len(high_level_rows), high_level_decisions=len(high_level_audits),
            reviewed_corrections=(str(args.v6_reviewed_corrections.resolve())
                                  if args.v6_reviewed_corrections else None),
            reviewed_corrections_sha256=(CR.sha256_file(args.v6_reviewed_corrections)
                                        if args.v6_reviewed_corrections else None))
        if args.v6_teacher_mode == "physical_oracle":
            report["selection"].update(candidate=("all v4.3-style direction mismatches and observed collisions"
                if args.v6_anchor_mode == "failure-pair" else "all observed action-head collisions"),
                correction_start="each selected AH's actual before_world_pose",
                label="task-goal oracle route with collision-free replay and plan/prefix direction alignment",
                one_anchor_per_episode=False, max_anchors_per_episode=None,
                sample_types=((["direction_mismatch_relabel", "collision_relabel"]
                    if args.v6_anchor_mode == "failure-pair" else ["collision_relabel"]) +
                    (["golden_route_relabel"] if args.v6_golden_extra_max else [])),
                candidate_policy="all_eligible_in_rollout_order",
                adjacent_action_head_only=False, direction_classifier="none",
                collision_recovery_included=False, high_level_labels="none",
                golden_extra_selection="one to three seeded random 1.6m slow boundaries per accepted route; new causal model query and prefix direction check" if args.v6_golden_extra_max else "disabled")
        else:
            report["selection"].update(candidate="observed collision/blocked/no_progress/direction mismatch or reviewed overshoot",
                correction_start="each condition AH's actual before_world_pose",
                label="replay-safe route checked against the whole conditioning local ToolCall",
                one_anchor_per_episode=False, sample_types=["execution_relabel", "recovery"],
                candidate_policy="first_accepted_per_type", adjacent_action_head_only=False,
                direction_classifier="online_protocol", collision_recovery_included=True,
                high_level_labels="explicit reviewed state-bound tool calls only")
    CR.atomic_json(output / "provenance.json", report)
    (output / "DONE").write_text(json.dumps({
        "ok": True, "anchors": accepted_anchors, "samples": len(rows),
        **({"training_ready": bool(rows), "teacher_mode": args.v6_teacher_mode} if is_v6 else {}),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
