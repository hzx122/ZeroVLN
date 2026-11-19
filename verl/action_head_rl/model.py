"""Route-replayed Qwen query recomputation and trainable-scope selection."""
from __future__ import annotations

from contextlib import nullcontext

import torch

from action_head_rl.config import TrainScope
from action_head_rl.router_replay import replay_qwen35_moe_routes
from action_head_rl.text_policy import selected_token_logprobs
from sft.joint_fm_head_sft import FrozenQwenQueryForward


def set_train_scope(module: torch.nn.Module, scope: TrainScope) -> dict[str, int]:
    """Apply the supported optimization scopes and return an audit."""
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    for parameter in module.query_encoder.parameters():
        parameter.requires_grad_(scope is not TrainScope.HEAD_ONLY)
    for parameter in module.action_head.parameters():
        parameter.requires_grad_(True)
    if scope is TrainScope.FULL:
        for parameter in module.qwen.parameters():
            parameter.requires_grad_(True)
    return {
        "all": sum(parameter.numel() for parameter in module.parameters()),
        "trainable": sum(parameter.numel() for parameter in module.parameters()
                         if parameter.requires_grad),
        "qwen_trainable": sum(parameter.numel() for parameter in module.qwen.parameters()
                              if parameter.requires_grad),
        "query_trainable": sum(parameter.numel() for parameter in
                                module.query_encoder.parameters()
                                if parameter.requires_grad),
        "head_trainable": sum(parameter.numel() for parameter in
                               module.action_head.parameters()
                               if parameter.requires_grad),
    }


def recompute_query_hidden(
    query_forward: FrozenQwenQueryForward,
    slow_inputs: dict[str, torch.Tensor],
    *,
    scope: TrainScope,
    query_routes: torch.Tensor,
    all_routes: torch.Tensor | None = None,
    all_positions: torch.Tensor | None = None,
    routes_already_installed: bool = False,
) -> torch.Tensor:
    """Recompute the full prefix and queries under rollout routing."""
    backbone = query_forward._backbone()
    del query_routes
    if all_routes is None:
        raise ValueError("R3 recomputation requires all-token routes")
    token_rows = slow_inputs.get("input_ids", slow_inputs["attention_mask"])
    batch_size = int(token_rows.shape[0])
    expected_tokens = int(token_rows.numel()) \
        + batch_size * query_forward.query_encoder.QUERY_COUNT
    if all_routes.shape[0] != expected_tokens:
        raise ValueError(
            f"R3 route/token mismatch: rollout={all_routes.shape[0]} "
            f"recompute={expected_tokens}")
    if all_positions is None or all_positions.numel() != expected_tokens:
        raise ValueError("R3 all_positions must align one-to-one with all routes")
    cache = getattr(query_forward, '_fm_prefix_cache', None)
    if cache is not None:
        if not routes_already_installed or scope is TrainScope.HEAD_ONLY:
            raise ValueError('FM prefix cache requires scoped selective query replay')
        return cache.forward(query_forward, slow_inputs, all_routes)
    routes, offset = all_routes, 0
    context = (nullcontext() if routes_already_installed else
               replay_qwen35_moe_routes(backbone, routes, token_offset=offset))
    # HF gradient checkpointing marks embedding outputs as requiring gradients
    # even when every backbone parameter is frozen. Pure Head training needs
    # only the resulting condition, not a backward pass through that backbone.
    # Query training and full training must retain the differentiable graph.
    grad_context = torch.no_grad() if scope is TrainScope.HEAD_ONLY else nullcontext()
    with context, grad_context:
        return query_forward.forward_full(slow_inputs)


def recompute_text_logprobs(
    query_forward: FrozenQwenQueryForward,
    slow_inputs: dict[str, torch.Tensor],
    *,
    response_mask: torch.Tensor,
    all_routes: torch.Tensor,
    all_positions: torch.Tensor,
    temperature: float = 1.0,
    routes_already_installed: bool = False,
    single_token_target_ids: torch.Tensor | None = None,
    executable_metadata: torch.Tensor | None = None,
    trajectory_targets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Replay a sampled answer without appending trainable query carriers.

    This runs inside the joint module's root FSDP forward, so the untied LM
    head is unsharded and receives the same optimizer step as the FM head.
    """
    ids = slow_inputs["input_ids"]
    if all_routes is None or all_routes.shape[0] != ids.numel():
        raise ValueError("text R3 routes must align with the exact input ids")
    if all_positions is None or all_positions.numel() != ids.numel():
        raise ValueError("text R3 positions must align with the exact input ids")
    if response_mask.shape != ids.shape:
        raise ValueError("text response mask must align with exact input ids")
    backbone = query_forward._backbone()
    context = nullcontext() if routes_already_installed else replay_qwen35_moe_routes(
        backbone, all_routes, token_offset=0)
    with context:
        embeds, mask, positions = query_forward.prefix_embeds(slow_inputs)
        output = backbone(inputs_embeds=embeds, attention_mask=mask,
                          position_ids=positions, use_cache=False,
                          output_hidden_states=False, return_dict=True)
        hidden = output.last_hidden_state
        lm_head = query_forward.model.get_output_embeddings()
        if lm_head is None:
            raise ValueError("text policy requires Qwen's causal language-model head")
        return selected_token_logprobs(
            hidden, lm_head, ids, response_mask, temperature,
            trajectory_targets=trajectory_targets,
            single_token_target_ids=single_token_target_ids,
            executable_metadata=executable_metadata)


def recompute_joint_outputs(
    query_forward, slow_inputs, *, response_mask, all_routes, all_positions,
    temperature=1.0, routes_already_installed=False, executable_metadata=None,
):
    """Experimental common causal prefix; only selected answer positions use LM head.

    CPU algebra tests do not establish GDN kernel/RNG parity on the target GPU.
    The full prefix plus query routes stay installed through the single backward.
    """
    from sft.joint_fm_head_sft import _append_positions

    ids = slow_inputs["input_ids"]
    count = query_forward.query_encoder.QUERY_COUNT
    expected = ids.numel() + ids.shape[0] * count
    if response_mask.shape != ids.shape:
        raise ValueError("joint forward requires an aligned text mask")
    if all_routes is None or all_routes.shape[0] != expected:
        raise ValueError("joint R3 routes must cover prefix and query tail")
    if all_positions is None or all_positions.numel() != expected:
        raise ValueError("joint R3 positions must cover prefix and query tail")
    backbone = query_forward._backbone()
    context = nullcontext() if routes_already_installed else replay_qwen35_moe_routes(
        backbone, all_routes, token_offset=0)
    with context:
        embeds, mask, positions = query_forward.prefix_embeds(slow_inputs)
        inputs, full_mask = query_forward.query_encoder.append(embeds, mask)
        output = backbone(
            inputs_embeds=inputs, attention_mask=full_mask,
            position_ids=_append_positions(positions, count), use_cache=False,
            output_hidden_states=False, return_dict=True)
        hidden = output.last_hidden_state
        queries = hidden[:, -count:].to(query_forward.query_encoder.query_embeddings.device)
        lm_head = query_forward.model.get_output_embeddings()
        if lm_head is None:
            raise ValueError("joint forward requires the causal language-model head")
        text = selected_token_logprobs(hidden[:, :-count], lm_head, ids,
                                       response_mask, temperature,
                                       executable_metadata=executable_metadata)
        return queries, text
