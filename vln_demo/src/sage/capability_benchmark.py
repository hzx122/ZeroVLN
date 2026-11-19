"""Pure geometry contracts and diagnostic metrics for the SAGE capability bench.

The module deliberately contains no renderer, simulator, or model dependency.  A
case builder records oracle geometry in a manifest; the rollout evaluator then
uses these functions to align the agent trace to the golden path and report the
five requested capabilities without collapsing every failure into SR.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


CAPABILITIES = ("perception", "grounding", "planning", "recovery", "execution")
SCHEMA_VERSION = "sage-capability-v2"
V3_SCHEMA_VERSION = "sage-capability-v3-dev"
LEGACY_SUCCESS_CONTRACT = "legacy_point_goal"
SEMANTIC_SUCCESS_CONTRACT = "semantic_terminal_region_v1"


def wrap_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def path_length(points: Sequence[Sequence[float]]) -> float:
    return sum(math.hypot(float(b[0]) - float(a[0]),
                          float(b[1]) - float(a[1]))
               for a, b in zip(points, points[1:]))


def _segments(points: Sequence[Sequence[float]]):
    arc = 0.0
    for i, (a, b) in enumerate(zip(points, points[1:])):
        ax, ay, bx, by = float(a[0]), float(a[1]), float(b[0]), float(b[1])
        length = math.hypot(bx - ax, by - ay)
        if length > 1e-9:
            yield i, ax, ay, bx, by, length, arc
        arc += length


def project_to_path(xy: Sequence[float], points: Sequence[Sequence[float]]) -> dict:
    """Nearest point, lateral error, and normalized arc progress on a polyline."""
    if len(points) < 2:
        return {"distance_m": float("inf"), "arc_m": 0.0, "progress": 0.0,
                "segment": -1, "nearest_xy": None, "heading_deg": None}
    px, py = float(xy[0]), float(xy[1])
    total = path_length(points)
    best = None
    for i, ax, ay, bx, by, length, arc0 in _segments(points):
        vx, vy = bx - ax, by - ay
        t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) /
                              max(length * length, 1e-12)))
        nx, ny = ax + t * vx, ay + t * vy
        dist = math.hypot(px - nx, py - ny)
        candidate = (dist, arc0 + t * length, i, nx, ny,
                     math.degrees(math.atan2(vy, vx)))
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return {"distance_m": best[0], "arc_m": best[1],
            "progress": best[1] / max(total, 1e-9), "segment": best[2],
            "nearest_xy": [best[3], best[4]], "heading_deg": best[5]}


def point_at_arc(points: Sequence[Sequence[float]], target_arc_m: float) -> Optional[list]:
    """Interpolate a point on a polyline at an absolute arc length."""
    if not points:
        return None
    target = max(0.0, float(target_arc_m))
    for _, ax, ay, bx, by, length, arc0 in _segments(points):
        if target <= arc0 + length:
            t = max(0.0, min(1.0, (target - arc0) / length))
            return [ax + t * (bx - ax), ay + t * (by - ay)]
    return [float(points[-1][0]), float(points[-1][1])]


def plan_recovery_intervention(occ, golden: Sequence[Sequence[float]],
                               xy: Sequence[float], yaw_deg: float, config: dict,
                               return_path_fn=None) -> Optional[dict]:
    """Choose a safe, observable perturbation from the *actual* trigger pose.

    This contains no scene/object special cases.  A candidate must be physically
    executable, leave the golden corridor by a configured amount, and retain a
    short collision-free route back to a point ahead on the golden path.
    ``return_path_fn`` is injectable so the geometry contract is unit-testable.
    """
    aligned = project_to_path(xy, golden)
    if aligned["distance_m"] > float(config.get("max_pre_lateral_error_m", .6)):
        return None
    progress = aligned["progress"]
    if not (float(config.get("trigger_progress", .4)) <= progress <=
            float(config.get("max_trigger_progress", .75))):
        return None
    radius = float(config.get("robot_radius_m", .07))
    min_move = float(config.get("min_forced_move_m", .5))
    min_lateral = float(config.get("min_post_lateral_error_m", .35))
    max_lateral = float(config.get("max_post_lateral_error_m", 1.2))
    rejoin_ahead = float(config.get("rejoin_ahead_m", 1.0))
    max_return = float(config.get("max_return_path_m", 3.0))
    angles = [float(x) for x in config.get("candidate_turn_degrees",
                                           (-90, 90, -60, 60, -45, 45))]
    distances = [float(x) for x in config.get("candidate_move_distances_m", (.8, .6))]
    rejoin = point_at_arc(golden, aligned["arc_m"] + rejoin_ahead)
    candidates = []
    for angle in angles:
        heading = math.radians(float(yaw_deg) + angle)
        for distance in distances:
            tx = float(xy[0]) + distance * math.cos(heading)
            ty = float(xy[1]) + distance * math.sin(heading)
            nx, ny, hit = occ.try_step(float(xy[0]), float(xy[1]), tx, ty,
                                       radius, True)
            moved = math.hypot(nx - float(xy[0]), ny - float(xy[1]))
            post = project_to_path((nx, ny), golden)
            if hit or moved < min_move or not (min_lateral <= post["distance_m"] <= max_lateral):
                continue
            return_path = (return_path_fn((nx, ny), rejoin)
                           if return_path_fn is not None else [(nx, ny), rejoin])
            if not return_path or path_length(return_path) > max_return:
                continue
            score = (abs(post["distance_m"] - .5 * (min_lateral + max_lateral)),
                     path_length(return_path), abs(angle), -moved)
            candidates.append((score, angle, distance, nx, ny, moved,
                               post["distance_m"], return_path))
    if not candidates:
        return None
    _, angle, distance, nx, ny, moved, lateral, return_path = min(candidates)
    return {
        "sequence": [
            {"name": "turn", "arguments": {
                "direction": "left" if angle > 0 else "right",
                "degrees": abs(angle)}},
            {"name": "move_forward", "arguments": {"distance": distance}},
        ],
        "expected_wrong_xy": [float(nx), float(ny)],
        "expected_lateral_error_m": float(lateral),
        "expected_move_m": float(moved),
        "rejoin_xy": rejoin,
        "return_path_length_m": path_length(return_path),
        "trigger_progress": progress,
        "pre_lateral_error_m": aligned["distance_m"],
    }


def _meaningful_turns(points: Sequence[Sequence[float]], threshold_deg=30.0) -> int:
    headings = [math.degrees(math.atan2(float(b[1]) - float(a[1]),
                                        float(b[0]) - float(a[0])))
                for a, b in zip(points, points[1:])
                if math.hypot(float(b[0]) - float(a[0]),
                              float(b[1]) - float(a[1])) > 1e-6]
    return sum(abs(wrap_deg(b - a)) >= threshold_deg
               for a, b in zip(headings, headings[1:]))


def trajectory_features(points: Sequence[Sequence[float]], route: dict,
                        clearances_m: Optional[Iterable[float]] = None) -> dict:
    clearances = [float(x) for x in (clearances_m or [])]
    turns = route.get("turns") or []
    return {
        "path_length_m": round(path_length(points), 4),
        "turn_count": len([x for x in turns if x != "straight"]),
        "geometric_turn_count": _meaningful_turns(points),
        "door_count": int(route.get("doors_crossed") or 0),
        "room_count": len(route.get("cross_rooms") or []),
        "landmark_count": len(route.get("landmarks") or []),
        "min_clearance_m": min(clearances) if clearances else None,
        "p10_clearance_m": (sorted(clearances)[max(0, int(len(clearances) * .1) - 1)]
                             if clearances else None),
    }


def select_grounding_pair(records: Sequence[dict], contract: dict):
    """Select the closest balanced pair from valid routes sharing one start."""
    candidates = []
    for i, left in enumerate(records):
        for right in records[i + 1:]:
            if left["target_object_id"] == right["target_object_id"]:
                continue
            if str(left.get("target_term", "")).strip().lower() == \
                    str(right.get("target_term", "")).strip().lower():
                continue
            length_a, length_b = float(left["path_length_m"]), float(right["path_length_m"])
            gap = abs(length_a - length_b)
            ratio = max(length_a, length_b) / max(min(length_a, length_b), 1e-9)
            turn_gap = abs(int(left["turn_count"]) - int(right["turn_count"]))
            door_gap = abs(int(left["door_count"]) - int(right["door_count"]))
            separation = math.hypot(float(left["goal_xy"][0]) - float(right["goal_xy"][0]),
                                    float(left["goal_xy"][1]) - float(right["goal_xy"][1]))
            if (gap > float(contract["max_length_gap_m"])
                    or ratio > float(contract["max_length_ratio"])
                    or turn_gap > int(contract["max_turn_gap"])
                    or door_gap > int(contract["max_door_gap"])
                    or separation < float(contract["min_goal_separation_m"])):
                continue
            score = (gap / max(float(contract["max_length_gap_m"]), 1e-9)
                     + (ratio - 1.0) + .25 * turn_gap + .25 * door_gap)
            candidates.append((score, -separation, left, right))
    if not candidates:
        return None
    _, _, left, right = min(candidates, key=lambda item: item[:2])
    return left, right


def eligible_dimensions(features: dict, sample: dict) -> Dict[str, dict]:
    """Return explicit pass/reasons for each capability's construction contract."""
    length = float(features["path_length_m"])
    turns = int(features["turn_count"])
    doors = int(features["door_count"])
    landmarks = int(features["landmark_count"])
    clearance = features.get("min_clearance_m")
    visibility = sample.get("goal_constraints", {}).get("goal_visibility", {})
    visible = visibility.get("status") == "verified"
    grounding = sample.get("capability_grounding") or {}
    contrast = grounding.get("contrast_set_id") and len(grounding.get("candidates", [])) >= 2
    execution_geometry = (2.0 <= length <= 6.0 and turns <= 1 and doors <= 1 and
                          clearance is not None and clearance <= 0.22 and
                          turns >= 1)
    checks = {
        "perception": (2.0 <= length <= 5.0 and turns <= 1 and doors <= 1 and visible,
                       "requires 2-5m, <=1 turn, <=1 door, verified target view"),
        "grounding": (bool(contrast),
                      "requires shared-start contrast_set with >=2 grounded candidates"),
        "planning": (8.0 <= length <= 15.0 and (turns + doors) >= 3 and landmarks >= 1,
                     "requires 8-15m, >=3 turn/door decisions, >=1 ordered landmark"),
        "recovery": (8.0 <= length <= 15.0 and (turns + doors) >= 3,
                     "requires an 8-15m multi-decision route plus a recoverable perturbation; "
                     "landmark evidence is intentionally not required"),
        "execution": (bool(execution_geometry) and visible,
                      "requires 2-6m, <=1 semantic turn/door, min clearance <=0.22m, "
                      "one route-level corner, verified target view"),
    }
    return {name: {"eligible": bool(ok), "reason": reason}
            for name, (ok, reason) in checks.items()}


def _trace_steps(rollout: dict, golden: Sequence[Sequence[float]]) -> List[dict]:
    rows = []
    initial = rollout.get("initial_info") or {}
    # SAGE is z-up. Its compatibility GPS is habitat-shaped [x,-y] and its
    # compass uses a different zero; golden paths are in geometry world XY.
    # Prefer explicit world fields whenever the backend provides them.
    prev_xy = initial.get("world_xy") or initial.get("gps")
    prev_yaw = initial.get("world_yaw_deg", initial.get("yaw_deg"))
    for turn in rollout.get("turns", []):
        for call in turn.get("calls", []):
            info = call.get("info") or {}
            xy = info.get("world_xy") or info.get("gps")
            yaw = info.get("world_yaw_deg", info.get("yaw_deg"))
            if xy is None:
                continue
            aligned = project_to_path(xy, golden)
            before = project_to_path(prev_xy, golden) if prev_xy is not None else None
            progress_delta = (aligned["arc_m"] - before["arc_m"]) if before else None
            heading_error = (abs(wrap_deg(float(yaw) - aligned["heading_deg"]))
                             if yaw is not None and aligned["heading_deg"] is not None else None)
            name = call.get("name")
            macro_correct = None
            if name in ("move_forward", "move_backward") and progress_delta is not None:
                signed = progress_delta if name == "move_forward" else -progress_delta
                macro_correct = signed >= -0.05 and aligned["distance_m"] <= 1.0
            elif name == "turn" and heading_error is not None and prev_yaw is not None:
                before_err = abs(wrap_deg(float(prev_yaw) - aligned["heading_deg"]))
                macro_correct = heading_error <= before_err + 5.0
            intervention = info.get("benchmark_intervention")
            if intervention:
                # The returned pose includes an exogenous perturbation and no
                # longer measures whether the agent's requested macro was right.
                macro_correct = None
            rows.append({"turn": turn.get("turn"), "name": name,
                         "args": call.get("args") or {}, "xy": xy, "yaw_deg": yaw,
                         "progress_delta_m": progress_delta, "heading_error_deg": heading_error,
                         "macro_correct": macro_correct,
                         "collided": bool(info.get("collided_this_step")),
                         "blocked": bool(info.get("blocked_this_step")), **aligned})
            prev_xy, prev_yaw = xy, yaw
    return rows


def score_rollout(case: dict, rollout: dict,
                  semantic_terminal: Optional[dict] = None) -> dict:
    """Score one trace with capability-specific observables and honest nulls."""
    golden = case.get("golden_path_xy") or []
    rows = _trace_steps(rollout, golden)
    final = rollout.get("final") or {}
    assistants = "\n".join(str(t.get("assistant") or "")
                           for t in rollout.get("turns", [])).lower()
    target_terms = [str(x).lower() for x in case.get("target_terms", []) if x]
    mentioned = bool(target_terms) and any(term in assistants for term in target_terms)
    correct = [r for r in rows if r.get("macro_correct") is True]
    adherence = ([r["distance_m"] for r in rows] if rows else [])
    success_contract = str((case.get("evaluation_contract") or {}).get(
        "primary_success") or LEGACY_SUCCESS_CONTRACT)
    if success_contract not in (LEGACY_SUCCESS_CONTRACT, SEMANTIC_SUCCESS_CONTRACT):
        raise ValueError("unknown primary success contract: " + success_contract)
    base = {
        "schema_version": case.get("schema_version", SCHEMA_VERSION),
        "case_id": case.get("case_id"), "dimension": case.get("dimension"),
        "success_contract": success_contract,
        "evaluation_valid": True,
        "success": bool(final.get("success")), "spl": final.get("spl"),
        "final_distance_m": final.get("distance_to_goal"),
        "tool_calls": rollout.get("calls_used", sum(len(t.get("calls", []))
                                                      for t in rollout.get("turns", []))),
        "route_adherence_mean_m": (sum(adherence) / len(adherence) if adherence else None),
        "route_adherence_max_m": (max(adherence) if adherence else None),
        "macro_correct_rate": ((sum(r.get("macro_correct") is True for r in rows) /
                                sum(r.get("macro_correct") is not None for r in rows))
                               if any(r.get("macro_correct") is not None for r in rows)
                               else None),
    }
    dim = case.get("dimension")
    if dim == "perception":
        base.update({"oracle_visible": case.get("oracle", {}).get("target_visible"),
                     "target_mentioned_in_reasoning": mentioned,
                     "diagnosis": ("perception_or_grounding_failure" if not mentioned and not base["success"]
                                   else "passed" if base["success"] else "navigation_after_recognition_failure")})
    elif dim == "grounding":
        final_xy = final.get("world_xy") or final.get("gps")
        candidates = case.get("oracle", {}).get("candidate_goal_xy") or {}
        dists = ({k: math.hypot(float(final_xy[0]) - float(v[0]),
                                float(final_xy[1]) - float(v[1]))
                  for k, v in candidates.items()} if final_xy is not None else {})
        chosen = min(dists, key=dists.get) if dists else None
        base.update({"nearest_candidate": chosen, "candidate_distances_m": dists,
                     "grounding_correct": chosen == case.get("target_object_id")})
    elif dim == "planning":
        milestones = case.get("oracle", {}).get("milestone_arcs_m") or []
        max_arc = max((r["arc_m"] for r in rows), default=0.0)
        base.update({"milestones_reached": sum(max_arc + 0.4 >= float(x) for x in milestones),
                     "milestones_total": len(milestones), "max_progress": max(
                         (r["progress"] for r in rows), default=0.0)})
    elif dim == "recovery":
        event = rollout.get("intervention") or {}
        event_call = event.get("after_call_index")
        after = rows[event_call:] if isinstance(event_call, int) else []
        recovered_at = next((i for i, r in enumerate(after)
                             if r["distance_m"] <= float(case.get("recovery", {}).get(
                                 "return_corridor_m", 0.6))), None)
        event_turn = next((int(t.get("turn") or 0) for t in rollout.get("turns", [])
                           if any(c.get("info", {}).get("benchmark_intervention")
                                  for c in t.get("calls", []))), None)
        post_reasoning = "\n".join(str(t.get("reasoning") or "")
                                  for t in rollout.get("turns", [])
                                  if event_turn is not None and int(t.get("turn") or 0) > event_turn)
        reason_evidence = (any(x in post_reasoning.lower() for x in
                               ("wrong", "mistake", "deviat", "return", "错", "偏", "返回", "掉头"))
                           if post_reasoning.strip() else None)
        intervention_valid = (bool(event.get("applied"))
                              and .35 <= float(event.get("post_lateral_error_m") or 0.0)
                              <= float(case.get("recovery", {}).get(
                                  "max_post_lateral_error_m", 1.2))
                              and not bool(event.get("blocked"))
                              and not bool(event.get("collided")))
        base.update({"intervention_applied": bool(event.get("applied")),
                     "intervention_valid": intervention_valid,
                     "post_intervention_lateral_error_m": event.get("post_lateral_error_m"),
                     "deviation_detected_in_reasoning": reason_evidence,
                     "recovered_to_corridor": recovered_at is not None,
                     "recovery_calls": recovered_at,
                     "pre_intervention_failure": not bool(event.get("applied"))})
        base["evaluation_valid"] = intervention_valid
        if not intervention_valid:
            base["invalid_reason"] = ("intervention_not_applied"
                                      if not event.get("applied")
                                      else "unsafe_or_unobservable_intervention")
    elif dim == "execution":
        base.update({"correct_macro_calls": len(correct),
                     "collision_on_correct_macro_rate": (sum(r["collided"] for r in correct) /
                         len(correct) if correct else None),
                     "blocked_on_correct_macro_rate": (sum(r["blocked"] for r in correct) /
                         len(correct) if correct else None),
                     "stop_error": not any(c.get("name") == "stop" for t in rollout.get("turns", [])
                                           for c in t.get("calls", []))})
        if not correct and success_contract == LEGACY_SUCCESS_CONTRACT:
            base["evaluation_valid"] = False
            base["invalid_reason"] = "no_correct_macro_calls"
    if semantic_terminal is not None:
        semantic_primary = semantic_terminal.get("primary") or {}
        base["terminal_metrics"] = {
            "legacy_point_goal": {
                "success": base["success"],
                "spl": base["spl"],
                "final_distance_m": base["final_distance_m"],
            },
            "semantic_region": {
                **semantic_terminal,
                "success": (semantic_primary.get("success")
                            if semantic_terminal.get("available") else None),
                "event": (semantic_primary.get("event")
                          if semantic_terminal.get("available") else None),
                "evaluation_valid": (semantic_primary.get("evaluation_valid")
                                     if semantic_terminal.get("available") else False),
            },
        }
        if success_contract == SEMANTIC_SUCCESS_CONTRACT:
            if not semantic_terminal.get("available"):
                base["success"] = False
                base["evaluation_valid"] = False
                base["invalid_reason"] = "semantic_terminal_unavailable"
            else:
                base["success"] = bool(semantic_primary.get("success"))
                base["evaluation_valid"] = (
                    bool(base["evaluation_valid"])
                    and bool(semantic_primary.get("evaluation_valid")))
                if not semantic_primary.get("evaluation_valid"):
                    base["invalid_reason"] = "ambiguous_semantic_terminal_region"
            # The simulator's SPL is tied to its legacy point goal and can be
            # zero for a valid semantic-region success. Preserve it above as a
            # diagnostic, but do not present it as the v3 primary-contract SPL.
            base["legacy_spl"] = base["spl"]
            base["spl"] = None
    elif success_contract == SEMANTIC_SUCCESS_CONTRACT:
        base["success"] = False
        base["evaluation_valid"] = False
        base["invalid_reason"] = "semantic_terminal_not_evaluated"
    return base


def aggregate_scores(scores: Sequence[dict]) -> dict:
    schemas = sorted({str(row.get("schema_version") or SCHEMA_VERSION)
                      for row in scores})
    contracts = sorted({str(row.get("success_contract") or LEGACY_SUCCESS_CONTRACT)
                        for row in scores})
    out = {"schema_version": (schemas[0] if len(schemas) == 1
                              else "mixed" if schemas else SCHEMA_VERSION),
           "success_contract": (contracts[0] if len(contracts) == 1
                                else "mixed" if contracts else LEGACY_SUCCESS_CONTRACT),
           "overall": {"cases": len(scores)}}
    for dim in CAPABILITIES:
        rows = [x for x in scores if x.get("dimension") == dim]
        valid = [x for x in rows if x.get("evaluation_valid", True)]
        spl_rows = [x for x in valid if x.get("spl") is not None]
        out[dim] = {"attempted_cases": len(rows), "valid_cases": len(valid),
                    "invalid_cases": len(rows) - len(valid),
                    "success_rate": (sum(bool(x.get("success")) for x in valid) / len(valid)
                                     if valid else None),
                    "mean_spl": (sum(float(x["spl"]) for x in spl_rows) / len(spl_rows)
                                 if spl_rows else None)}
    return out
