"""HF Qwen3.6 MoE route replay used by continuous-action recomputation."""
from __future__ import annotations

import types
from contextlib import contextmanager
from typing import Iterator

import torch


def validate_routes(routes: torch.Tensor, *, token_count: int,
                    layer_count: int, top_k: int) -> None:
    expected = (token_count, layer_count, top_k)
    if tuple(routes.shape) != expected:
        raise ValueError(f"routed_experts must have shape {expected}, got {tuple(routes.shape)}")
    if routes.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError("routed_experts must contain integer expert ids")
    if torch.any(routes < 0):
        raise ValueError("routed_experts contains a negative expert id")


@contextmanager
def replay_qwen35_moe_routes(
    backbone: torch.nn.Module,
    routes: torch.Tensor,
    *,
    token_offset: int = 0,
    route_selector=None,
) -> Iterator[None]:
    """Force selected experts while preserving gradients through router scores.

    ``routes`` is ``[token, layer, top_k]`` in causal execution order.  A
    token offset allows query-only R3 replay against the last four rows of a
    full prefix+query forward.  The implementation is promoted from the
    already validated ``query_q3_path_diagnostic.py`` probe.
    """
    layers = list(backbone.layers)
    top_k = int(backbone.config.num_experts_per_tok)
    validate_routes(
        routes, token_count=routes.shape[0], layer_count=len(layers), top_k=top_k)
    originals = []
    for layer_index, layer in enumerate(layers):
        gate = layer.mlp.gate
        original = gate.forward
        originals.append((gate, original))

        def replay(self, hidden_states, _original=original, _layer=layer_index):
            router_logits, router_scores, selected_experts = _original(hidden_states)
            flat = selected_experts.reshape(-1, selected_experts.shape[-1])
            start = token_offset if token_offset >= 0 else flat.shape[0] + token_offset
            active_routes = routes if route_selector is None else route_selector()
            if active_routes is None:
                return router_logits, router_scores, selected_experts
            end = start + active_routes.shape[0]
            if start < 0 or end > flat.shape[0]:
                raise RuntimeError(
                    f"route replay rows [{start}:{end}] exceed {flat.shape[0]} tokens")
            target = active_routes[:, _layer].to(
                device=router_logits.device, dtype=flat.dtype)
            # Qwen's first return changed from probabilities to raw logits
            # across Transformers versions. Keep that native return unchanged,
            # and derive replay probabilities from the router's actual linear
            # projection. Never infer the contract from a tensor's values.
            # Retain the full token shape, including prefix rows, so suffix
            # replay uses the same router GEMM shape/autocast as native Qwen.
            logits = torch.nn.functional.linear(
                hidden_states.reshape(-1, self.weight.shape[-1]), self.weight)
            probabilities = torch.nn.functional.softmax(logits, dtype=torch.float, dim=-1)
            scores = probabilities[start:end].gather(-1, target)
            # Preserve the original full-softmax -> gather -> normalize order.
            scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            flat = flat.clone()
            flat[start:end] = target
            score_flat = router_scores.reshape(-1, router_scores.shape[-1]).clone()
            score_flat[start:end] = scores.to(score_flat.dtype)
            return (
                router_logits,
                score_flat.reshape_as(router_scores),
                flat.reshape_as(selected_experts),
            )

        gate.forward = types.MethodType(replay, gate)
    try:
        yield
    finally:
        for gate, original in originals:
            gate.forward = original
