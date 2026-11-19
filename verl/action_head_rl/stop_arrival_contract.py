"""Opt-in evidence for decision-boundary arrival and accepted model STOP.

Private reward/scoring metadata only: no goals, distances or labels are added
to the model's observation. A terminal scoring STOP never becomes model credit.
"""
from __future__ import annotations

import copy
import math
import os
from collections.abc import Mapping


SCHEMA = "three-source-boundary-stop-v1"
ENVIRONMENT_KEY = "ACTION_HEAD_FINAL_CONTRACT"
STATE_SCHEMA = "three-source-decision-boundary-state-v1"


def _number(value, name, *, nullable=False):
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("missing or nonnumeric boundary " + name)
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("invalid boundary " + name)
    return value


def boundary_state(info):
    """Read current goal membership, never infer it from past microsteps."""
    state = info.get("rule_credit_state")
    if not isinstance(state, Mapping):
        raise ValueError("boundary STOP requires private rule-credit state")
    schema = state.get("schema")
    goal_hash = state.get("goal_contract_sha256")
    if not goal_hash or not state.get("map_sha256"):
        raise ValueError("boundary goal/map is not version bound")
    distance = _number(state.get("goal_distance_m"), "goal_distance_m", nullable=True)
    if schema == "mp3d-rule-credit-state-v1":
        detail = info.get("native_goal_success_detail") or {}
        reached = detail.get("inside_goal_range")
        if (detail.get("schema") != "habitat_vlnce_native_stop_3m_v1"
                or detail.get("available") is not True
                or detail.get("goal_contract_sha256") != goal_hash
                or distance is None or type(reached) is not bool
                or reached != (distance < 3.)):
            raise ValueError("native decision-boundary goal evidence is inconsistent")
    elif schema in ("sage-rule-credit-state-v1", "sage-rule-credit-state-v2"):
        detail = info.get("official_object_success_detail") or {}
        reached = detail.get("success")
        detail_hash = detail.get("target_contract_sha256", detail.get("contract_sha256"))
        if (detail.get("available") is not True or detail_hash != goal_hash
                or type(reached) is not bool
                or info.get("official_object_success") is not reached):
            raise ValueError("SAGE decision-boundary goal evidence is inconsistent")
    else:
        raise ValueError("unsupported boundary state schema: " + str(schema))
    micro = state.get("ever_goal_reached")
    if type(micro) is not bool:
        raise ValueError("microstep arrival diagnostic must be explicitly available")
    information = state.get("information_available")
    entities = state.get("relevant_entities")
    if type(information) is not bool or not isinstance(entities, list):
        raise ValueError("boundary information evidence is missing")
    return dict(schema=STATE_SCHEMA, source_schema=schema,
        goal_reached=reached, goal_distance_m=distance,
        path_length_m=_number(info.get("path_length_m"), "path_length_m"),
        shortest_path_m=_number(info.get("shortest_path_m"), "shortest_path_m"),
        collisions=_number(state.get("collisions"), "collisions"),
        steps_used=_number(info.get("steps_used"), "steps_used"),
        goal_contract_sha256=goal_hash, map_sha256=state["map_sha256"],
        relevant_entities=copy.deepcopy(entities), information_available=information,
        ever_goal_reached_microsteps=micro,
        arrival_evidence="current_goal_membership_at_slow_decision_boundary")


def _model_stop(record):
    calls = record.get("calls") or []
    if len(calls) > 1:
        raise ValueError("boundary STOP contract requires one tool per decision")
    call = calls[0] if calls else {}
    if call.get("name") != "stop":
        return False
    if call.get("ok") is False or not isinstance(call.get("score_info"), Mapping):
        return False
    from agent_tools import rejected_stop_result
    if rejected_stop_result(call.get("arguments")) is not None:
        raise ValueError("invalid model STOP was executed by the evaluator")
    # Gym can report both terminated and truncated when an explicit STOP uses
    # its last environment step. The accepted model action remains decisive.
    if record.get("tool_call_error") or not call.get("terminated"):
        raise ValueError("accepted model STOP lacks unambiguous terminal evidence")
    return True


class StopArrivalContract:
    def __init__(self, initial_info):
        self.initial = boundary_state(initial_info)
        self.current = copy.deepcopy(self.initial)
        self.records = []
        self.first_arrival_decision_index = 0 if self.initial["goal_reached"] else None
        self.first_arrival_path_length_m = (self.initial["path_length_m"]
                                           if self.initial["goal_reached"] else None)

    def observe_record(self, record, info):
        """Called exactly once after every sampled decision, including rejects."""
        if self.records and self.records[-1]["final_credit"]["model_stop_accepted"]:
            raise ValueError("decision followed an accepted terminal STOP")
        if int(record["turn"]) != len(self.records) + 1:
            raise ValueError("decision-boundary state chain has a missing decision")
        accepted = _model_stop(record)
        calls = record.get("calls") or []
        # bench.step_tool(STOP) can bypass the render client's cached last_info.
        # The accepted STOP's direct HTTP response is authoritative here.
        if accepted:
            info = calls[0]["score_info"]
        after = boundary_state(info)
        for key in ("source_schema", "goal_contract_sha256", "map_sha256", "shortest_path_m"):
            if after[key] != self.initial[key]:
                raise ValueError("boundary identity changed inside episode: " + key)
        for key in ("path_length_m", "collisions", "steps_used"):
            if after[key] + 1e-6 < self.current[key]:
                raise ValueError("boundary counter decreased: " + key)
        # A failed service and a physically blocked but valid AH execution are
        # not model protocol rejections. Share the learner's exact evidence
        # predicate so the same transition cannot receive conflicting labels.
        from .rule_credit import _protocol_rejection_reason
        rejection_reason = _protocol_rejection_reason(record)
        rejected = rejection_reason is not None
        from agent_tools import parse_tool_calls
        visible = record.get("assistant") or ""
        attempted_stop = any(call.get("name") == "stop"
                             for call in [*calls, *parse_tool_calls(visible)])
        origin = ("model_legal" if accepted else "model_invalid" if attempted_stop
                  else "natural_text" if not calls and "<tool_call>" not in visible
                  else "invalid_tool_call" if not calls else "none")
        record["final_credit"] = dict(before=copy.deepcopy(self.current), after=after,
            rejected=rejected, protocol_rejection_reason=rejection_reason,
            model_stop_accepted=accepted, stop_origin=origin,
            stop_environment_truncated=bool(accepted and calls[0].get("truncated")))
        self.current = copy.deepcopy(after)
        if self.first_arrival_decision_index is None and after["goal_reached"]:
            self.first_arrival_decision_index = int(record["turn"])
            self.first_arrival_path_length_m = after["path_length_m"]
        self.records.append(record)

    def finalize(self, result, score_info):
        accepted = bool(self.records and self.records[-1]["final_credit"]["model_stop_accepted"])
        if bool(result.get("agent_stopped")) != accepted or (accepted and result.get("why") != "agent_stop"):
            raise ValueError("terminal model STOP provenance disagrees with the executed call")
        native = bool(score_info.get("official_object_success", False)
                      if "official_object_success" in score_info else score_info.get("success", False))
        success = bool(accepted and self.current["goal_reached"] and native
                       and result.get("evaluation_valid") and not result.get("infra_error"))
        path, shortest = self.current["path_length_m"], self.current["shortest_path_m"]
        if self.current["source_schema"] == "mp3d-rule-credit-state-v1":
            spl = _number(score_info.get("spl"), "native_spl") if success else 0.
            spl_source = "habitat_native_spl"
        else:
            # Same SAGE success-weighted environment path/reference denominator
            # as the r7 evaluator; do not substitute sparse decision chords.
            spl = (shortest / max(path, shortest, 1e-12)) if success else 0.
            if success and shortest == 0. and self.current["source_schema"] == "sage-rule-credit-state-v2":
                spl = 1.
            spl_source = "sage_success_weighted_environment_path_and_shortest"
        if spl > 1. + 1e-6:
            raise ValueError("native SPL outside [0, 1]")
        diagnostic = None if accepted else dict(origin="wrapper_diagnostic",
            reason="evaluator terminal scoring only", native_success=native,
            native_spl=score_info.get("spl"), score_info=copy.deepcopy(score_info))
        result["final_credit"] = dict(schema=SCHEMA,
            initial_state=copy.deepcopy(self.initial), final_state=copy.deepcopy(self.current),
            goal_contract_sha256=self.initial["goal_contract_sha256"],
            native_success=success, terminal_native_success=native,
            model_stop_accepted=accepted,
            stop_origin="model_legal" if accepted else "wrapper_diagnostic",
            termination_reason=result.get("why"), native_spl=spl,
            path_length_m=path, physical_path_m=path, shortest_path_m=shortest,
            path_source="environment_actual_native_path_length_m",
            spl_source=spl_source,
            first_arrival_decision_index=self.first_arrival_decision_index,
            first_arrival_path_length_m=self.first_arrival_path_length_m,
            wrapper_stop_diagnostic=diagnostic)
        result.update(success=success, spl=spl, path_length_m=path, shortest_path_m=shortest)
        return result


def start_contract(args, initial_info):
    selected = os.environ.get(ENVIRONMENT_KEY, "")
    if not selected:
        return None
    if selected != SCHEMA:
        raise ValueError("unknown final RL observation/termination contract: " + selected)
    if not getattr(args, "one_tool_call_per_turn", False):
        raise ValueError("final RL contract requires one_tool_call_per_turn")
    return StopArrivalContract(initial_info)
