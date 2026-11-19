"""CPU-only, opt-in credit at executed tool-call boundaries.

This module does not change model inputs, rewards, or the training driver.  A
critic fitted through batch n may only score batch n+1 or later.  The critic is
privileged training-side metadata, never a policy conditioning feature.

The task remains undiscounted native terminal success at the configured tool
budget.  GAE(lambda < 1) with an approximate critic is a biased estimator; the
existing sequence/token-mean policy losses introduce additional weighting.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "v6-tool-credit-logistic-v1"
FEATURE_SCHEMA = "official-expanded-bbox-call-state-v1"
FEATURE_NAMES = (
    "log1p_bbox_distance_m", "inside_official_bbox", "tool_budget_fraction",
    "log1p_path_length_m", "collision_step_fraction", "log1p_shortest_path_m",
)
SUCCESS_SCHEMA = "sage-official-expanded-object-bbox-v1"
PHYSICAL_SUCCESS_SCHEMA = "sage-designated-target-navmesh-v2"
NATIVE_SUCCESS_SCHEMA = "habitat_vlnce_native_stop_3m_v1"
TOOL_CREDIT_MODES = (
    "previous_value_gae", "previous_value_mc", "previous_value_mc_loo",
)


def credit_estimator_spec(mode: str) -> dict[str, Any]:
    """Return fixed estimator fields for the training/resume contract.

    The default GAE fields exactly match its existing contract; adding the
    other modes must not silently revise old GAE checkpoints. Critic state,
    fitting, terminal rules, and record loss weights are common to all modes.
    """
    if mode not in TOOL_CREDIT_MODES:
        raise ValueError(f"unsupported tool-credit mode: {mode!r}")
    spec = {"mode": mode, "lambda": 0.95 if mode == "previous_value_gae" else 1.0,
            "normalization": "none"}
    if mode != "previous_value_gae":
        hybrid = mode == "previous_value_mc_loo"
        spec.update(baseline_kind=("previous_state_value_and_leave_one_out" if hybrid
                                   else "previous_state_value"),
                    state_baseline_weight=0.5 if hybrid else 1.0,
                    loo_baseline_weight=0.5 if hybrid else 0.0)
    return spec


def credit_reward_normalization(mode: str) -> str:
    """Name the estimator in the reward contract without changing its reward."""
    credit_estimator_spec(mode)  # Reject unknown modes, never fall back to GAE.
    return {
        "previous_value_gae": "privileged_previous_value_call_gae",
        "previous_value_mc": "privileged_previous_value_call_mc",
        "previous_value_mc_loo": "privileged_previous_value_call_mc_loo",
    }[mode]


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _finite(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"missing/non-numeric {name}") from exc
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise ValueError(f"invalid {name}: {number}")
    return number


def _success_detail(info):
    if info.get('success_contract') == NATIVE_SUCCESS_SCHEMA:
        return info.get('native_goal_success_detail')
    return info.get('official_object_success_detail')


def state_features(info: Mapping[str, Any], *, calls_used: int,
                   max_tool_calls: int) -> tuple[float, ...]:
    """Read authoritative official geometry; never substitute point-goal data."""
    if not 0 <= calls_used <= max_tool_calls or max_tool_calls <= 0:
        raise ValueError("invalid tool budget state")
    detail = _success_detail(info)
    if not isinstance(detail, Mapping) or detail.get("schema") not in (SUCCESS_SCHEMA, PHYSICAL_SUCCESS_SCHEMA, NATIVE_SUCCESS_SCHEMA):
        raise ValueError("missing supported official success detail")
    if detail.get("available") is not True or not detail.get("contract_sha256"):
        raise ValueError("official success detail is unavailable/unversioned")
    if detail['schema'] == NATIVE_SUCCESS_SCHEMA:
        state = info.get('rule_credit_state') or {}
        distance = _finite(detail.get('goal_distance_m'), 'native goal geodesic', nonnegative=True)
        inside = distance < 3.
        stopped, success = detail.get('native_stop_called'), detail.get('success')
        if (detail.get('radius_m') != 3. or type(stopped) is not bool or type(success) is not bool
                or detail.get('inside_goal_range') is not inside or success != (stopped and inside)
                or info.get('success') is not success
                or state.get('schema') != 'mp3d-rule-credit-state-v1'
                or state.get('goal_distance_m') != distance
                or any(not detail.get(k) or detail[k] != state.get(k)
                       for k in ('map_sha256', 'goal_contract_sha256'))):
            raise ValueError('inconsistent/unversioned native STOP/geodesic state')
    elif detail['schema'] == PHYSICAL_SUCCESS_SCHEMA:
        # No XY distance or inferred success: these are measured on the
        # executed support layer by the environment.
        distance = _finite(info.get('distance_to_goal'), '3-D geodesic distance', nonnegative=True)
        inside = detail.get('success')
        if (not isinstance(inside, bool) or inside != (distance == 0.)
                or not detail.get('target_contract_sha256')
                or not detail.get('navmesh_manifest_sha256')):
            raise ValueError('inconsistent/unversioned physical target state')
    else:
        point, box = detail.get("position_xy"), detail.get("expanded_bbox_xyxy")
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("official position_xy must contain two coordinates")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("official expanded bbox must contain four coordinates")
        x, y = (_finite(v, "official position") for v in point)
        x0, y0, x1, y1 = (_finite(v, "official bbox") for v in box)
        if x0 > x1 or y0 > y1:
            raise ValueError("official bbox bounds are inverted")
        distance = math.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))
        inside = distance == 0.0
        if not isinstance(detail.get("success"), bool) or detail["success"] != inside:
            raise ValueError("official success flag contradicts official geometry")
    if "official_object_success" in info and info["official_object_success"] != inside:
        raise ValueError("official success fields disagree")
    path = _finite(info.get("path_length_m"), "path_length_m", nonnegative=True)
    collisions = _finite(info.get("collisions"), "collisions", nonnegative=True)
    steps = _finite(info.get("steps_used"), "steps_used", nonnegative=True)
    shortest = _finite(info.get("shortest_path_m"), "shortest_path_m", nonnegative=True)
    return (math.log1p(distance), float(inside), calls_used / max_tool_calls,
            math.log1p(path), collisions / max(steps, 1.0), math.log1p(shortest))


@dataclass(frozen=True)
class CallTransition:
    call_id: str
    decision_index: int
    tool_name: str
    option_id: str | None
    chunk_count: int
    before: tuple[float, ...]
    after: tuple[float, ...]
    reward: float = 0.0
    done: bool = False


@dataclass(frozen=True)
class CallLedger:
    step: int
    policy_version: int
    episode_id: str
    scene: str
    max_tool_calls: int
    native_success: bool
    calls: tuple[CallTransition, ...]
    provenance_sha256: str


def build_call_ledger(result: Mapping[str, Any], interactions: Sequence[Mapping], *,
                      step: int, policy_version: int, max_tool_calls: int = 50) -> CallLedger:
    """Join production turns to chunk feedback without inspecting policy tensors.

    Every executed AH call has one transition, regardless of its chunk count.
    Invalid non-calling responses get a text-only budget-consuming transition. Several calls in a
    single response retain one decision_index, allowing one text loss only.
    """
    if step <= 0 or policy_version != step - 1:
        raise ValueError("call credit requires the synchronous rollout policy version")
    if result.get("evaluation_valid") is not True or result.get("infra_error"):
        raise ValueError("invalid/infrastructure-failed rollout cannot receive credit")
    if result.get("success_contract") not in (SUCCESS_SCHEMA, PHYSICAL_SUCCESS_SCHEMA, NATIVE_SUCCESS_SCHEMA) or not isinstance(result.get("success"), bool):
        raise ValueError("call credit requires explicit native official terminal success")
    why = result.get("why")
    if why not in ("agent_stop", "tool_budget_exhausted"):
        raise ValueError(f"unsupported terminal reason: {why!r}")
    by_option: dict[str, list[Mapping]] = {}
    for row in interactions:
        option = str(row.get("option_id") or "")
        if not option:
            raise ValueError("FM chunk lacks option_id")
        by_option.setdefault(option, []).append(row)
    initial = result.get("initial_info")
    if not isinstance(initial, Mapping):
        raise ValueError("rollout lacks authoritative initial_info")
    schema = result['success_contract']
    def checked_features(info, **kwargs):
        features = state_features(info, **kwargs)
        if _success_detail(info)['schema'] != schema:
            raise ValueError('call ledger cannot mix geometry contracts')
        return features
    current = initial
    used = 0
    calls: list[CallTransition] = []
    consumed: set[str] = set()
    decision_indices: set[int] = set()
    accepted_stop = False
    for turn in result.get("turns") or []:
        decision = int(turn["turn"])
        if decision <= 0 or decision in decision_indices:
            raise ValueError("duplicate/invalid high-level decision index")
        decision_indices.add(decision)
        for ordinal, call in enumerate(turn.get("calls") or [None]):
            before = checked_features(current, calls_used=used, max_tool_calls=max_tool_calls)
            name, option, count = "text_only", None, 0
            after_info = current
            if call is None:
                rejection = turn.get("tool_call_error") or {}
                if not (rejection.get("schema") == "tool_call_rejection_v1"
                        and rejection.get("ok") is False
                        and rejection.get("return_reason") == "rejected"):
                    raise ValueError("text-only decision lacks verified budget-consuming rejection")
                used += 1
            else:
                used += 1
                name = str(call.get("name") or "")
                raw = call.get("result") or {}
                if name == "run_action_head":
                    option = str(raw.get("option_id") or "")
                    if not option or option in consumed:
                        raise ValueError("missing/duplicate AH call option_id")
                    consumed.add(option)
                    chunks = by_option.get(option, [])
                    count = len(chunks)
                    if count:
                        if [row.get("chunk_index") for row in chunks] != list(range(count)):
                            raise ValueError("AH chunks must be complete and ordered")
                        if raw.get("chunks") != count:
                            raise ValueError("tool result/chunk record count mismatch")
                        after_info = chunks[-1].get("gym_feedback")
                        if not isinstance(after_info, Mapping):
                            raise ValueError("AH call lacks authoritative final gym feedback")
                    else:
                        from .final_credit import _normal_zero_chunk_call
                        rejected = (call.get("ok") is False and raw.get("return_reason") == "rejected"
                                    and all(raw.get(key) == 0 for key in (
                                        "chunks", "steps_used", "path_m", "net_m")))
                        if not (_normal_zero_chunk_call(call) or rejected):
                            raise ValueError("zero-chunk AH call lacks verified no-op evidence")
                elif name == "turn":
                    rejected = (call.get("ok") is False and raw.get("steps_used") == 0
                                and raw.get("return_reason") in ("rejected", "consecutive_turn_limit_reached"))
                    trace = call.get("execution_state_trace") or {}
                    poses = [trace.get(key) or {} for key in ("before_world_pose", "after_world_pose")]
                    points = [p.get("world_xy") for p in poses]
                    actual = _success_detail(current)["position_xy"]
                    if not rejected and (call.get("ok") is not True or any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in points)
                            or any(abs(_finite(p[k], "turn pose") - actual[k]) > 0.002
                                   for p in points for k in (0, 1))):
                        raise ValueError("turn is not a verified position-preserving execution")
                    # Rotation changes pose orientation, absent from this first
                    # critic. Use measured step count, not a guessed distance.
                    after_info = dict(current)
                    after_info["steps_used"] = _finite(current.get("steps_used"), "steps_used") + _finite(
                        raw.get("steps_used"), "turn steps_used", nonnegative=True)
                elif name == "stop":
                    rejected = (call.get("ok") is False
                                and raw.get("schema") == "tool_call_rejection_v1"
                                and raw.get("tool") == "stop" and raw.get("ok") is False
                                and raw.get("return_reason") == "rejected"
                                and raw.get("error_code") == "invalid_tool_arguments"
                                and raw.get("action_executed") is False
                                and raw.get("steps_used") == 0 and raw.get("budget_cost") == 1)
                    if not rejected:
                        after_info = call.get("score_info")
                        if not isinstance(after_info, Mapping):
                            raise ValueError("STOP lacks authoritative score_info or verified rejection")
                        accepted_stop = True
                else:
                    raise ValueError(f"unsupported tool in credit ledger: {name!r}")
            after = checked_features(after_info, calls_used=used, max_tool_calls=max_tool_calls)
            calls.append(CallTransition(f"decision-{decision}/call-{ordinal}", decision, name,
                                        option, count, before, after))
            current = after_info
    if not calls or set(by_option) - consumed:
        raise ValueError("unmatched FM records or empty decision ledger")
    if used != result.get("calls_used") or used > max_tool_calls:
        raise ValueError("actual tool-call budget does not match ledger")
    if why == "tool_budget_exhausted" and used != max_tool_calls:
        raise ValueError("budget terminal before the configured tool budget")
    if why == "agent_stop" and (calls[-1].tool_name != "stop" or not accepted_stop):
        raise ValueError("agent_stop terminal lacks final STOP transition")
    if accepted_stop and why != "agent_stop":
        raise ValueError("accepted STOP contradicts terminal reason")
    native_success = bool(result["success"])
    if bool(calls[-1].after[1]) != native_success:
        raise ValueError("native terminal success contradicts final official state")
    calls[-1] = replace(calls[-1], reward=float(native_success), done=True)
    provenance = {"step": step, "policy_version": policy_version,
                  "episode_id": result["episode_id"], "scene": result["scene"],
                  "max_tool_calls": max_tool_calls, "calls": [asdict(c) for c in calls]}
    return CallLedger(step, policy_version, str(result["episode_id"]), str(result["scene"]),
                      max_tool_calls, native_success, tuple(calls), _digest(provenance))


@dataclass(frozen=True)
class LogisticStateCritic:
    """Immutable snapshot; fitting returns a new snapshot with resumable Adam."""
    coefficients: tuple[float, ...] = (0.0,) * (len(FEATURE_NAMES) + 1)
    adam_m: tuple[float, ...] = (0.0,) * (len(FEATURE_NAMES) + 1)
    adam_v: tuple[float, ...] = (0.0,) * (len(FEATURE_NAMES) + 1)
    optimizer_step: int = 0
    fitted_through_step: int = 0
    max_tool_calls: int = 50
    training_sha256: str = ""
    learning_rate: float = 0.03
    l2: float = 0.001

    def values(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
            raise ValueError("critic features have wrong shape/nonfinite values")
        w = np.asarray(self.coefficients)
        z = x @ w[1:] + w[0]
        return np.exp(-np.logaddexp(0.0, -z))

    def fit_mc(self, ledgers: Sequence[CallLedger], *, updates: int = 20) -> "LogisticStateCritic":
        if not ledgers or updates <= 0:
            raise ValueError("critic update needs completed rollout ledgers and positive updates")
        if any(not row.calls[-1].done for row in ledgers):
            raise ValueError("MC critic requires completed terminal outcomes")
        if any(row.max_tool_calls != self.max_tool_calls for row in ledgers):
            raise ValueError("critic training tool-budget contract mismatch")
        x, y, mass = [], [], []
        for row in ledgers:
            for call in row.calls:
                x.append((1.0, *call.before))
                y.append(float(row.native_success))
                mass.append(1.0 / len(row.calls))
        x, y, mass = np.asarray(x), np.asarray(y), np.asarray(mass)
        mass = mass / mass.sum()
        w, m, v = (np.asarray(a, dtype=np.float64).copy()
                   for a in (self.coefficients, self.adam_m, self.adam_v))
        iteration = self.optimizer_step
        for _ in range(updates):
            iteration += 1
            prediction = np.exp(-np.logaddexp(0.0, -(x @ w)))
            gradient = x.T @ ((prediction - y) * mass)
            gradient[1:] += self.l2 * w[1:]
            m = 0.9 * m + 0.1 * gradient
            v = 0.999 * v + 0.001 * gradient ** 2
            w -= self.learning_rate * (m / (1 - 0.9 ** iteration)) / (
                np.sqrt(v / (1 - 0.999 ** iteration)) + 1e-8)
        if not all(np.isfinite(a).all() for a in (w, m, v)):
            raise ValueError("nonfinite critic/optimizer state")
        return replace(self, coefficients=tuple(w), adam_m=tuple(m), adam_v=tuple(v),
                       optimizer_step=iteration,
                       fitted_through_step=max(self.fitted_through_step, *(r.step for r in ledgers)),
                       training_sha256=_digest({"previous": self.training_sha256,
                                                "ledgers": [r.provenance_sha256 for r in ledgers]}))

    def state_dict(self) -> dict[str, Any]:
        body = {"schema": SCHEMA, "feature_schema": FEATURE_SCHEMA,
                "feature_names": list(FEATURE_NAMES), **asdict(self)}
        return {**body, "sha256": _digest(body)}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "LogisticStateCritic":
        body = dict(state)
        sha = body.pop("sha256", None)
        if sha != _digest(body) or body.pop("schema", None) != SCHEMA:
            raise ValueError("critic state hash/schema mismatch")
        if body.pop("feature_schema", None) != FEATURE_SCHEMA or body.pop("feature_names", None) != list(FEATURE_NAMES):
            raise ValueError("critic feature schema mismatch")
        for key in ("coefficients", "adam_m", "adam_v"):
            values = tuple(body[key])
            if len(values) != len(FEATURE_NAMES) + 1 or not all(math.isfinite(v) for v in values):
                raise ValueError(f"invalid critic {key}")
            body[key] = values
        if (body["optimizer_step"] < 0 or body["fitted_through_step"] < 0
                or body["max_tool_calls"] <= 0 or not math.isfinite(body["learning_rate"])
                or not math.isfinite(body["l2"]) or body["learning_rate"] <= 0 or body["l2"] < 0):
            raise ValueError("invalid critic optimizer/version metadata")
        return cls(**body)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(self.state_dict(), indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "LogisticStateCritic":
        return cls.from_state_dict(json.loads(Path(path).read_text()))


def rebase_critic_for_new_run(critic: LogisticStateCritic, *, source_actor: Mapping[str, Any],
                            target_run: str) -> LogisticStateCritic:
    """Reset the actor-batch clock, preserving the critic's real Adam clock.

    This starts a new experiment. It is never used for an exact resume, and
    does not mutate the committed source critic or reset Adam bias correction.
    """
    if not source_actor or not target_run or critic.fitted_through_step <= 0:
        raise ValueError("critic warm-start requires a committed actor and a new run identity")
    lineage = _digest({"kind": "policy_critic_new_experiment_v1",
                       "source_actor": dict(source_actor),
                       "source_critic_sha256": critic.state_dict()["sha256"],
                       "target_run": str(target_run)})
    return replace(critic, fitted_through_step=0, training_sha256=lineage)


def gae_returns(rewards: Sequence[float], values: Sequence[float], dones: Sequence[bool], *,
                gamma: float = 1.0, lam: float = 0.95) -> tuple[float, ...]:
    """One recursion per call; final next-state value is always zero."""
    if gamma != 1.0 or not 0 <= lam <= 1:
        raise ValueError("tool-credit v1 preserves gamma=1 and requires lambda in [0,1]")
    if len(rewards) == 0 or len(rewards) != len(values) or len(values) != len(dones):
        raise ValueError("unaligned/empty GAE sequence")
    if not dones[-1] or any(dones[:-1]):
        raise ValueError("GAE requires one complete episode, not bootstrapped truncation")
    if not all(math.isfinite(float(v)) for v in (*rewards, *values)):
        raise ValueError("nonfinite GAE inputs")
    output = [0.0] * len(values)
    following = 0.0
    for index in reversed(range(len(values))):
        next_value = 0.0 if dones[index] else float(values[index + 1])
        delta = float(rewards[index]) + next_value - float(values[index])
        following = delta + (0.0 if dones[index] else lam * following)
        output[index] = following
    return tuple(output)


def assign_call_credit(ledger: CallLedger, critic: LogisticStateCritic, *,
                       gamma: float = 1.0, lam: float | None = None,
                       mode: str = "previous_value_gae",
                       loo_baseline: float | None = None) -> dict[str, Any]:
    """Assign frozen-critic credit, leaving policy records/loss weights alone.

    MC uses the complete terminal return minus the pre-call state baseline.
    MC_LOO averages that baseline with the other same-case episode outcomes.
    The caller must verify their group identity/independent seeds and exclude
    this episode before supplying ``loo_baseline``; it is never inferred from
    a call depth or fitted to this episode's own outcome here.
    """
    spec = credit_estimator_spec(mode)
    gamma = _finite(gamma, "gamma")
    if gamma != 1.0:
        raise ValueError("tool-credit estimators require gamma=1")
    lam = spec["lambda"] if lam is None else _finite(lam, "lambda")
    if lam != spec["lambda"]:
        raise ValueError(f"{mode} requires lambda={spec['lambda']}")
    if mode == "previous_value_mc_loo":
        loo_baseline = _finite(loo_baseline, "leave-one-out baseline")
        if not 0.0 <= loo_baseline <= 1.0:
            raise ValueError("leave-one-out baseline must be in [0,1]")
    elif loo_baseline is not None:
        raise ValueError("leave-one-out baseline is only valid for previous_value_mc_loo")
    if critic.fitted_through_step >= ledger.step:
        raise ValueError("critic must be frozen from previous completed batches")
    if critic.max_tool_calls != ledger.max_tool_calls:
        raise ValueError("critic inference tool-budget contract mismatch")
    if mode != "previous_value_gae" and (
            not ledger.calls or any(call.reward != 0.0 for call in ledger.calls[:-1])
            or ledger.calls[-1].reward != float(ledger.native_success)):
        raise ValueError("MC call credit requires the native terminal-only reward")
    values = critic.values([call.before for call in ledger.calls])
    advantages = gae_returns([c.reward for c in ledger.calls], values,
                             [c.done for c in ledger.calls], gamma=gamma, lam=lam)
    if mode == "previous_value_mc_loo":
        # At lambda=1 GAE telescopes to R - V(before). Replace its baseline
        # with the fixed half-state/half-LOO mixture, with no group whitening.
        advantages = tuple(float(advantage + 0.5 * (value - loo_baseline))
                           for advantage, value in zip(advantages, values))
    decisions, options = {}, {}
    for call, advantage in zip(ledger.calls, advantages):
        # A multi-call assistant response is sampled once. Its text loss uses
        # the return from its first call, never one copy per tool or FM chunk.
        decisions.setdefault(call.decision_index, advantage)
        if call.option_id is not None and call.chunk_count:
            options[call.option_id] = advantage
    return {"schema": SCHEMA, "step": ledger.step, "policy_version": ledger.policy_version,
            "ledger_sha256": ledger.provenance_sha256,
            "critic_sha256": critic.state_dict()["sha256"],
            "critic_fitted_through_step": critic.fitted_through_step,
            "gamma": gamma, **spec,
            **({"loo_baseline": loo_baseline} if mode == "previous_value_mc_loo" else {}),
            "text_advantages": decisions, "option_advantages": options,
            "call_advantages": dict(zip((c.call_id for c in ledger.calls), advantages)),
            "call_values": dict(zip((c.call_id for c in ledger.calls), values.tolist()))}
