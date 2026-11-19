"""Fail-closed configuration contract for Qwen3.6 MoE action expert RL."""
from __future__ import annotations

from enum import Enum


from action_head.serve.compile_cache import configure_process_compile_cache


class TrainScope(str, Enum):
    FULL = "full"
    HEAD_QUERY = "head_query"
    HEAD_ONLY = "head_only"


class RouterReplayMode(str, Enum):
    R3 = "R3"


def validate_training_contract(
    *,
    train_scope: str,
    router_replay_mode: str,
    capture_full_routes: bool,
    rollout_backend: str,
) -> tuple[TrainScope, RouterReplayMode]:
    """Validate invariants that would otherwise silently bias policy gradients.

    Qwen3.6-35B-A3B is MoE.  Rollout and actor recomputation must execute the
    same experts, so disabling replay is intentionally not supported.  Updating
    the Qwen backbone additionally requires routes for every slow-policy token;
    query-only routes are sufficient only when Qwen remains frozen.
    """
    try:
        scope = TrainScope(train_scope)
    except ValueError as exc:
        raise ValueError("train_scope must be 'full', 'head_query' or 'head_only'") from exc
    try:
        replay = RouterReplayMode(router_replay_mode)
    except ValueError as exc:
        raise ValueError("Qwen3.6 Action Head RL requires router_replay_mode='R3'") from exc
    if rollout_backend != "vllm":
        raise ValueError("the validated Action Head rollout backend is vllm")
    if not capture_full_routes:
        raise ValueError(
            "Qwen3.6 training requires full-token vLLM route capture; four "
            "query routes alone cannot reproduce the prefix state consumed "
            "by those queries, even when the backbone is frozen"
        )
    return scope, replay
