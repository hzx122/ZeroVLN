"""Opt-in three-source credit at observable decision boundaries.

Navigation learns reaching the goal once. STOP learns the current binary
decision separately. Path preference compares only successful trajectories.
Privileged boundary metadata is consumed here, never added to model inputs.
"""
from __future__ import annotations

from collections import Counter
import math
import os
from typing import Mapping

VERSION = "v4_boundary_stop_spl"
SCHEMA = "v4-boundary-stop-spl-credit-v1"
EVIDENCE_SCHEMA = "three-source-boundary-stop-v1"
MAX_CHUNKS_PER_CALL = 4
NORMAL_UNEXECUTED_REASONS = frozenset(("time_budget_reached", "no_executable_motion"))


def ablated(name):
    values = set(filter(None, os.environ.get('ACTION_HEAD_FINAL_ABLATIONS', '').split(',')))
    known = {'no_stop_supervision', 'no_success_path', 'no_chunk_collision',
             'joint_tool_choice', 'unbalanced_stop', 'no_first_arrival', 'fixed_budget_stop'}
    if values - known:
        raise ValueError('unknown v4 ablations: ' + str(sorted(values - known)))
    if {'fixed_budget_stop', 'unbalanced_stop'} <= values:
        raise ValueError('choose exactly one STOP normalization variant')
    return name in values


def _normal_unexecuted_call(call):
    raw = call.get("result") or {}
    return (call.get("name") == "run_action_head" and call.get("ok") is True
            and raw.get("ok") is True and raw.get("tool") == "run_action_head"
            and str(raw.get("schema", "")).startswith("action_head_tool_result_v")
            and raw.get("return_reason") in NORMAL_UNEXECUTED_REASONS)


def _normal_zero_chunk_call(call):
    raw = call.get("result") or {}
    return (_normal_unexecuted_call(call)
            and type(raw.get("chunks")) is int and raw["chunks"] == 0
            and all(raw.get(key) == 0 for key in ("steps_used", "path_m"))
            and raw.get("net_m") in (None, 0.)
            and raw.get("blocked") is False and raw.get("collision") is False)


def executed_fm_records(result, records):
    """Join sampled records to execution acknowledgments before learner packing.

    A prediction can finish after the option deadline, or contain no executable
    motion. Those normal tails are not environment actions. Keep their compact
    audit evidence, but never give their FM samples a training gradient.
    """
    from .rule_credit import _protocol_rejection_reason
    calls = {}
    for turn in result.get("turns") or []:
        for call in turn.get("calls") or []:
            if call.get("name") != "run_action_head":
                continue
            raw = call.get("result") or {}
            option = str(raw.get("option_id") or "")
            if not option or option in calls:
                raise ValueError("missing/duplicate FM execution option")
            calls[option] = (turn, call)
    grouped = {}
    for record in records:
        option = str(record.get("option_id") or "")
        if option not in calls:
            raise ValueError("unmatched FM prediction option")
        grouped.setdefault(option, []).append(record)
    skipped, empty = [], []
    for option, (turn, call) in calls.items():
        raw = call.get("result") or {}
        count = raw.get("chunks")
        if type(count) is not int or not 0 <= count <= MAX_CHUNKS_PER_CALL:
            raise ValueError("invalid executed FM chunk count")
        rows = grouped.get(option, [])
        if any(r.get("execution_status") not in ("predicted", "executed") for r in rows):
            raise ValueError("FM prediction lacks explicit execution acknowledgment")
        executed = [r for r in rows if r["execution_status"] == "executed"]
        pending = [r for r in rows if r["execution_status"] == "predicted"]
        if ([r.get("chunk_index") for r in executed] != list(range(count))
                or len(rows) != count + len(pending)
                or [r.get("chunk_index") for r in rows] != list(range(len(rows)))):
            raise ValueError(f"missing/duplicate executed FM records: option={option!r}")
        if pending:
            if (len(pending) != 1 or pending[0].get("chunk_index") != count
                    or not _normal_unexecuted_call(call)):
                raise ValueError(f"unverified unexecuted FM prediction: option={option!r}")
            skipped.append(dict(option_id=option, chunk_index=count,
                                return_reason=raw["return_reason"]))
        elif raw.get("return_reason") == "no_executable_motion":
            raise ValueError("no-motion result lacks its predicted FM record")
        if not count:
            if not (_normal_zero_chunk_call(call) or _protocol_rejection_reason(turn)):
                raise ValueError("zero executed FM chunks lack normal completion/rejection evidence")
            if _normal_zero_chunk_call(call):
                empty.append(option)
    selected = [record for record in records if record["execution_status"] == "executed"]
    return selected, dict(schema="executed-fm-records-v1", executed_chunks=len(selected),
                          unexecuted_predictions=skipped, normal_zero_chunk_options=empty)


def contract(*, selective_backward=False):
    from .rule_credit import ALPHA, INFO_RATIO, COLLISION
    return dict(
        schema=SCHEMA, mode="rule_call_loo", version=VERSION,
        alpha=ALPHA, information_ratio=INFO_RATIO, collision_coefficient=COLLISION,
        information_cap_entities=5, gamma=1., critic_parameters=0,
        evidence_schema=EVIDENCE_SCHEMA,
        navigation_reward="v3_shaping_until_first_decision_boundary_arrival",
        first_arrival_reward=1., first_arrival_limit="once_per_episode",
        initial_arrival_bonus=False, arrival_evidence="current_goal_at_decision_boundary",
        baseline="same_case_depth_leave_one_out_absorbing_zero",
        terminal_success="accepted_model_stop_and_native_terminal_success",
        terminal_reward_to_navigation=False,
        stop_objective="binary_stop_vs_sum_legal_continue",
        stop_label="before_current_goal_membership",
        stop_normalization=("fixed_tool_budget_per_valid_decision" if ablated('fixed_budget_stop')
            else "same_case_group_unbalanced_episode_mass" if ablated('unbalanced_stop')
            else "same_case_group_present_class_balanced_episode_mass"),
        path_objective="successful_subset_native_spl_leave_one_out",
        path_credit="separate_channel_successful_translating_arrival_prefix_only",
        minimum_path_successes=2,
        rejected_text_advantage="min_navigation_advantage_zero",
        text_normalization="fixed_tool_budget",
        fm_normalization="fixed_tool_budget_times_max_chunks",
        max_chunks_per_call=MAX_CHUNKS_PER_CALL,
        action_credit="clipped_navigation_minus_executed_chunk_collision_v1",
        action_advantage_clip=[-1., 1.],
        action_qwen_gradient=("selective_action_leaves_shared_graph_v1"
                              if selective_backward else "discard_fm_sharded_gradients_v1"))


def _number(value, label, *, nullable=False):
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid final credit {label}")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid final credit {label}")
    return float(value)


def _state(value, identity):
    if not isinstance(value, Mapping):
        raise ValueError("missing final credit boundary state")
    if type(value.get("goal_reached")) is not bool:
        raise ValueError("current boundary goal membership is required")
    if value.get("goal_contract_sha256") != identity or not identity:
        raise ValueError("goal contract changed inside final credit episode")
    for name in ("path_length_m", "shortest_path_m", "collisions", "steps_used"):
        _number(value.get(name), name)
    _number(value.get("goal_distance_m"), "goal_distance_m", nullable=True)
    if type(value.get("information_available")) is not bool:
        raise ValueError("missing information availability")
    if not isinstance(value.get("relevant_entities"), (list, tuple)):
        raise ValueError("missing information entities")
    return value


def score_episode(rollout, *, step, max_tool_calls=50):
    from .rule_credit import ALPHA, INFO_RATIO, _protocol_rejection_reason, _agent_stopped
    if type(step) is not int or step <= 0 or type(max_tool_calls) is not int or max_tool_calls <= 0:
        raise ValueError("invalid final credit step/budget")
    result = rollout["result"]
    if result.get("evaluation_valid") is not True or result.get("infra_error"):
        raise ValueError("invalid rollout cannot receive final credit")
    evidence = result.get("final_credit") or {}
    if evidence.get("schema") != EVIDENCE_SCHEMA:
        raise ValueError("missing versioned final credit evidence")
    identity = evidence.get("goal_contract_sha256")
    initial = _state(evidence.get("initial_state"), identity)
    final = _state(evidence.get("final_state"), identity)
    stopped = _agent_stopped(result)
    if type(evidence.get("model_stop_accepted")) is not bool or stopped != evidence["model_stop_accepted"]:
        raise ValueError("inconsistent accepted model STOP evidence")
    if stopped and evidence.get("stop_origin") != "model_legal":
        raise ValueError("successful STOP must originate from the model")
    if any(type(evidence.get(k)) is not bool for k in ("native_success", "terminal_native_success")):
        raise ValueError("missing native terminal success evidence")
    strict_success = stopped and evidence["terminal_native_success"]
    if (type(result.get("success")) is not bool or evidence["native_success"] != strict_success
            or result.get("success") != strict_success):
        raise ValueError("final success must require an accepted model STOP")
    if strict_success and not final["goal_reached"]:
        raise ValueError("native STOP success contradicts final goal membership")
    spl = _number(evidence.get("native_spl"), "native_spl")
    if spl > 1.:
        raise ValueError("native SPL outside [0, 1]")
    for key in ("path_length_m", "shortest_path_m"):
        if not math.isclose(_number(evidence.get(key), key), final[key], abs_tol=1e-6):
            raise ValueError("final physical path evidence mismatch")
    turns = result.get("turns") or []
    if not turns or len(turns) > max_tool_calls:
        raise ValueError("empty or over-budget final credit decisions")
    if result.get("calls_used") != len(turns):
        raise ValueError("final credit requires one budget slot per decision")
    interactions = {}
    for chunk in rollout["chunks"]:
        option = str(chunk.get("option_id") or "")
        if not option or chunk.get("policy_version") != step - 1:
            raise ValueError("invalid final credit FM option/policy version")
        interactions.setdefault(option, []).append(chunk)
    states = [initial]
    indexed, used_options, seen_options, decisions = [], set(), set(), set()
    previous = initial
    for turn in turns:
        index = turn.get("turn")
        if type(index) is not int or index != len(decisions) + 1:
            raise ValueError("duplicate/invalid final credit decision")
        decisions.add(index)
        calls = turn.get("calls") or []
        if len(calls) > 1:
            raise ValueError("final credit requires one tool per decision")
        boundary = turn.get("final_credit") or {}
        before = _state(boundary.get("before"), identity)
        after = _state(boundary.get("after"), identity)
        if before != previous:
            raise ValueError("discontinuous final credit boundary chain")
        for key in ("path_length_m", "collisions", "steps_used"):
            if after[key] + 1e-6 < before[key]:
                raise ValueError("decreasing physical execution counter")
        if after.get("map_sha256") != initial.get("map_sha256"):
            raise ValueError("map changed within final credit episode")
        if after["shortest_path_m"] != initial["shortest_path_m"]:
            raise ValueError("shortest path changed within final credit episode")
        rejected_reason = _protocol_rejection_reason(turn)
        rejected = rejected_reason is not None
        if type(boundary.get("rejected")) is not bool or boundary["rejected"] != rejected:
            raise ValueError("final credit rejection lacks protocol evidence")
        call = calls[0] if calls else {}
        name = call.get("name", "rejected")
        if name not in ("run_action_head", "turn", "stop", "rejected"):
            raise ValueError("unsupported final credit tool")
        if not calls and not rejected:
            raise ValueError("text-only decision must be a protocol rejection")
        accepted_stop = name == "stop" and not rejected
        if (type(boundary.get("model_stop_accepted")) is not bool
                or boundary["model_stop_accepted"] != accepted_stop):
            raise ValueError("per-decision model STOP evidence mismatch")
        if accepted_stop and (turn is not turns[-1] or not stopped
                              or boundary.get("stop_origin") != "model_legal"):
            raise ValueError("accepted model STOP must terminate the episode")
        option, chunks = None, []
        if name == "run_action_head" and not rejected:
            raw = call.get("result") or {}
            option = str(raw.get("option_id") or "")
            chunks = interactions.get(option, [])
            zero = _normal_zero_chunk_call(call)
            if (not option or option in seen_options or not 0 <= len(chunks) <= MAX_CHUNKS_PER_CALL
                    or (not chunks and not zero)
                    or raw.get("chunks") != len(chunks)
                    or [c.get("chunk_index") for c in chunks] != list(range(len(chunks)))):
                raise ValueError("incomplete/duplicate final credit FM chunks: "
                                 f"decision={index} option={option!r} "
                                 f"reported_chunks={raw.get('chunks')!r} "
                                 f"record_indices={[c.get('chunk_index') for c in chunks]!r} "
                                 f"return_reason={raw.get('return_reason')!r}")
            seen_options.add(option)
            if chunks:
                used_options.add(option)
            elif (before["goal_reached"] != after["goal_reached"]
                  or any(before[k] != after[k] for k in ("path_length_m", "collisions", "steps_used"))):
                raise ValueError("zero-chunk tool changed physical state")
            if any(type(c.get("collision")) is not bool for c in chunks):
                raise ValueError("chunk collision requires explicit execution evidence")
            if any(c["collision"] for c in chunks) != (after["collisions"] > before["collisions"]):
                raise ValueError("call/chunk collision evidence disagrees")
        if rejected and (before["goal_reached"] != after["goal_reached"]
                         or any(before[k] != after[k] for k in ("path_length_m", "collisions", "steps_used"))):
            raise ValueError("rejected tool changed physical state")
        indexed.append((turn, before, after, name, option, chunks, rejected_reason))
        states.append(after)
        previous = after
    if previous != final or used_options != set(interactions):
        raise ValueError("final state or FM option coverage mismatch")
    expected = [d.get("decision_index") for d in rollout.get("decisions", [])]
    if len(expected) != len(set(expected)) or set(expected) != decisions:
        raise ValueError("final credit text decision coverage mismatch")
    distance_available = all(s.get("goal_distance_m") is not None for s in states)
    information_available = all(s["information_available"] for s in states)
    normalizer = max(initial["goal_distance_m"], 1.) if distance_available else 1.
    seen, discoveries = set(initial["relevant_entities"]), 0
    arrived = initial["goal_reached"]
    arrival_index = 0 if arrived else None
    arrival_path = initial["path_length_m"] if arrived else None
    scored = []
    for turn, before, after, name, option, chunks, rejected_reason in indexed:
        rejected = rejected_reason is not None
        active = name in ("run_action_head", "turn") and not rejected
        prefix = not arrived
        dp = 0.
        if prefix and active and distance_available:
            dp = (normalizer / (normalizer + after["goal_distance_m"])
                  - normalizer / (normalizer + before["goal_distance_m"]))
        if name in ("turn", "stop") and not rejected:
            if before["goal_reached"] != after["goal_reached"] or abs(after["path_length_m"] - before["path_length_m"]) > 1e-6:
                raise ValueError("non-translating tool changed goal membership/path")
            if name == "turn" and abs(dp) > 1e-6:
                raise ValueError("turn changed goal distance")
            dp = 0.
        new = set(after["relevant_entities"]) - seen
        di = ((min(5, discoveries + len(new)) - min(5, discoveries)) / 5.
              if prefix and active and information_available else 0.)
        if prefix and active:
            discoveries += len(new)
        seen.update(after["relevant_entities"])
        arrival = float(prefix and active and after["goal_reached"])
        if arrival:
            arrived, arrival_index, arrival_path = True, turn["turn"], after["path_length_m"]
        row = dict(call_id=f"decision-{turn['turn']}", decision_index=turn["turn"],
                   tool=name, option_id=option, chunk_count=len(chunks),
                   navigation_prefix=prefix, navigation_active=active,
                   path_credit_eligible=(prefix and name == "run_action_head" and not rejected
                                         and after["path_length_m"] > before["path_length_m"]),
                   protocol_rejected=rejected, protocol_rejection_reason=rejected_reason,
                   collision=after["collisions"] > before["collisions"],
                   distance_reward=ALPHA * dp, information_reward=ALPHA * INFO_RATIO * di,
                   first_arrival_reward=0. if ablated('no_first_arrival') else arrival,
                   process_reward=ALPHA * (dp + INFO_RATIO * di) + (0. if ablated('no_first_arrival') else arrival),
                   stop_target=int(before["goal_reached"]), stop_choice=int(name == "stop"),
                   stop_supervision_valid=not rejected)
        scored.append(row)
    if (evidence.get("first_arrival_decision_index") != arrival_index
            or evidence.get("first_arrival_path_length_m") != arrival_path):
        raise ValueError("first arrival must match current decision-boundary membership")
    total = 0.
    for row in reversed(scored):
        total += row["process_reward"]
        row["return_to_go"] = total
    return dict(schema=SCHEMA, calls=scored, native_success=strict_success,
                active_stop_success=strict_success, agent_stopped=stopped,
                terminal_reward=float(strict_success), shaped_return=total,
                boundary_arrived=arrived, first_arrival_decision_index=arrival_index,
                first_arrival_path_length_m=arrival_path, native_spl=spl if strict_success else 0.,
                physical_path_m=final["path_length_m"], shortest_path_m=final["shortest_path_m"],
                distance_available=distance_available, information_available=information_available,
                collision_chunks=sum(c["collision"] for c in rollout["chunks"]),
                max_tool_calls=max_tool_calls, rule_state_seconds=0.)


def assign_group(rollouts, *, step, max_tool_calls=50):
    if len(rollouts) < 2:
        raise ValueError("final credit needs at least two same-case trajectories")
    identities = {(r["result"].get("scene"), r["result"].get("episode_id"),
                   (r["result"].get("final_credit") or {}).get("goal_contract_sha256")) for r in rollouts}
    seeds = [r.get("seed") for r in rollouts]
    if (len(identities) != 1 or any(v in (None, "") for identity in identities for v in identity)
            or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError("final credit requires one goal and distinct independent seeds")
    scores = [score_episode(r, step=step, max_tool_calls=max_tool_calls) for r in rollouts]
    success_scores = [s for s in scores if s["native_success"]]
    classes = Counter(c["stop_target"] for s in scores for c in s["calls"] if c["stop_supervision_valid"])
    for index, score in enumerate(scores):
        path_adv = 0.
        if not ablated('no_success_path') and score["native_success"] and len(success_scores) >= 2:
            path_adv = score["native_spl"] - sum(s["native_spl"] for s in success_scores if s is not score) / (len(success_scores) - 1)
        text, options, stop_targets, stop_weights, path_text, path_options = {}, {}, {}, {}, {}, {}
        for depth, row in enumerate(score["calls"]):
            baseline = sum(s["calls"][depth]["return_to_go"] if depth < len(s["calls"]) else 0.
                           for j, s in enumerate(scores) if j != index) / (len(scores) - 1)
            raw = row["return_to_go"] - baseline
            # STOP has its own goal-conditioned loss. Suffix continuation gets
            # no past arrival reward, even if other samples ended earlier.
            effective = raw if row["navigation_prefix"] and row["navigation_active"] else 0.
            if row["protocol_rejected"]:
                effective = min(raw, 0.)
            decision = row["decision_index"]
            text[decision] = effective
            row.update(baseline=baseline, raw_text_advantage=raw, text_advantage=effective)
            stop_targets[decision] = row["stop_target"]
            stop_weights[decision] = (len(scores) / (sum(classes.values()) if ablated('unbalanced_stop')
                                     else len(classes) * classes[row["stop_target"]])
                                      if row["stop_supervision_valid"] else 0.)
            if ablated('fixed_budget_stop'):
                # Preserve the observed arrival prior. A single positive must
                # not receive the combined mass of hundreds of negatives.
                stop_weights[decision] = (1. / max_tool_calls
                    if row['stop_supervision_valid'] else 0.)
            if ablated('no_stop_supervision'):
                stop_weights[decision] = 0.
            path_text[decision] = path_adv if row["path_credit_eligible"] else 0.
            if row["chunk_count"]:
                option = row["option_id"]
                options[option] = max(-1., min(1., effective))
                path_options[option] = path_adv if row["path_credit_eligible"] else 0.
                row["action_advantage"] = options[option]
        score.update(text_advantages=text, option_advantages=options,
                     stop_targets=stop_targets, stop_weights=stop_weights,
                     path_advantage=path_adv, path_text_advantages=path_text,
                     path_option_advantages=path_options)
    return scores


def apply_records(records, credit):
    from .text_policy import is_text_record
    from .rule_credit import COLLISION
    if credit.get("schema") != SCHEMA:
        raise ValueError("wrong final credit schema")
    budget = credit["max_tool_calls"]
    for record in records:
        if is_text_record(record):
            index = int(record["decision_index"])
            record.update(advantage=credit["text_advantages"][index], loss_weight=1. / budget,
                          text_stop_target=credit["stop_targets"][index],
                          text_stop_choice=next(c["stop_choice"] for c in credit["calls"] if c["decision_index"] == index),
                          text_stop_weight=credit["stop_weights"][index],
                          text_path_advantage=credit["path_text_advantages"][index])
        else:
            option = str(record["option_id"])
            if record.get("execution_status") != "executed":
                raise ValueError("only acknowledged executed FM chunks may receive credit")
            if type(record.get("collision")) is not bool:
                raise ValueError("FM record lacks executed chunk collision")
            collision_cost = 0. if ablated('no_chunk_collision') else COLLISION * float(record["collision"])
            record.update(advantage=credit["option_advantages"][option] - collision_cost,
                          navigation_advantage=credit["option_advantages"][option],
                          option_collision_cost=collision_cost,
                          loss_weight=1. / (budget * MAX_CHUNKS_PER_CALL),
                          action_path_advantage=credit["path_option_advantages"][option])
        record["final_credit_schema"] = SCHEMA


def summarize(scores):
    if any(s.get("schema") != SCHEMA for s in scores):
        raise ValueError("cannot mix final credit with legacy credit")
    calls = [c for s in scores for c in s["calls"]]
    return dict(schema=SCHEMA, mode="rule_call_loo", episodes=len(scores), calls=len(calls),
                active_stop_success_episodes=sum(s["native_success"] for s in scores),
                agent_stop_episodes=sum(s["agent_stopped"] for s in scores),
                boundary_arrival_episodes=sum(s["boundary_arrived"] for s in scores),
                first_arrival_reward=sum(c["first_arrival_reward"] for c in calls),
                distance_reward=sum(c["distance_reward"] for c in calls),
                information_reward=sum(c["information_reward"] for c in calls),
                protocol_rejection_calls=sum(c["protocol_rejected"] for c in calls),
                collision_chunks=sum(s.get("collision_chunks", 0) for s in scores),
                stop_positive_decisions=sum(c["stop_supervision_valid"] and c["stop_target"] == 1 for c in calls),
                stop_negative_decisions=sum(c["stop_supervision_valid"] and c["stop_target"] == 0 for c in calls),
                success_path_nonzero_episodes=sum(s["path_advantage"] != 0. for s in scores),
                terminal_reward=sum(s["terminal_reward"] for s in scores))
