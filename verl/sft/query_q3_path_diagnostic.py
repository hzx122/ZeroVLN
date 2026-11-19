#!/usr/bin/env python3
"""Compare Qwen3.6 four-query full, chunk-cache and sequential-cache paths."""
from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from models.dualvln_query_encoder import DualVlnQueryEncoder
from sft.joint_fm_head_sft import (
    FrozenQwenQueryForward,
    _append_positions,
    _query_positions,
    build_slow_inputs,
)


def _routes(outputs, token_slice: slice, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    logits = outputs.router_logits
    if logits is None:
        raise RuntimeError("Qwen did not return router logits")
    ids, margins = [], []
    for layer_logits in logits:
        values, indices = torch.topk(layer_logits.float(), top_k + 1, dim=-1)
        ids.append(indices[token_slice, :top_k].cpu())
        margins.append((values[token_slice, top_k - 1] - values[token_slice, top_k]).cpu())
    # [query, layer, top_k] and [query, layer]
    return torch.stack(ids, dim=1), torch.stack(margins, dim=1)


def _hidden(outputs, token_slice: slice) -> torch.Tensor:
    if outputs.hidden_states is None:
        raise RuntimeError("Qwen did not return layer hidden states")
    # Include embedding output at layer 0 and all decoder outputs thereafter.
    return torch.stack([
        value[:, token_slice, :].squeeze(0).detach().cpu().to(torch.bfloat16)
        for value in outputs.hidden_states
    ], dim=1)


@torch.inference_mode()
def _full(frozen: FrozenQwenQueryForward, inputs: dict[str, torch.Tensor],
          forced_query_routes: torch.Tensor | None = None):
    embeds, mask, positions = frozen.prefix_embeds(inputs)
    combined, combined_mask = frozen.query_encoder.append(embeds, mask)
    originals = []
    if forced_query_routes is not None:
        count = frozen.query_encoder.QUERY_COUNT
        if tuple(forced_query_routes.shape[:2]) != (
                count, len(frozen._backbone().layers)):
            raise ValueError(
                "forced query routes must have shape "
                f"[{count},{len(frozen._backbone().layers)},top_k], got "
                f"{tuple(forced_query_routes.shape)}")
        for layer_index, layer in enumerate(frozen._backbone().layers):
            gate = layer.mlp.gate
            original = gate.forward
            originals.append((gate, original))

            def replay_query_rows(self, hidden_states, _original=original,
                                  _layer=layer_index, _count=count):
                router_logits, router_scores, selected_experts = _original(
                    hidden_states)
                target = forced_query_routes[:, _layer].to(
                    device=router_logits.device, dtype=selected_experts.dtype)
                if target.shape != selected_experts[-_count:].shape:
                    raise RuntimeError(
                        "forced query route shape does not match router output: "
                        f"{tuple(target.shape)} vs "
                        f"{tuple(selected_experts[-_count:].shape)}")
                target_scores = router_logits[-_count:].gather(-1, target)
                target_scores = target_scores / target_scores.sum(
                    dim=-1, keepdim=True)
                selected_experts = selected_experts.clone()
                router_scores = router_scores.clone()
                selected_experts[-_count:] = target
                router_scores[-_count:] = target_scores.to(router_scores.dtype)
                return router_logits, router_scores, selected_experts

            gate.forward = types.MethodType(replay_query_rows, gate)
    try:
        outputs = frozen._backbone()(
            inputs_embeds=combined,
            attention_mask=combined_mask,
            position_ids=_append_positions(
                positions, frozen.query_encoder.QUERY_COUNT),
            use_cache=False,
            output_hidden_states=True,
            output_router_logits=True,
            return_dict=True,
        )
    finally:
        for gate, original in originals:
            gate.forward = original
    count = frozen.query_encoder.QUERY_COUNT
    routes, margins = _routes(outputs, slice(-count, None),
                               frozen._backbone().config.num_experts_per_tok)
    # output_router_logits contains the current router probabilities, whose
    # natural top-k can differ from the indices actually dispatched by replay.
    # Report the executed routes, not a second top-k over those probabilities.
    if forced_query_routes is not None:
        routes = forced_query_routes.detach().cpu()
    return _hidden(outputs, slice(-count, None)), routes, margins, positions


@torch.inference_mode()
def _cached(frozen: FrozenQwenQueryForward, inputs: dict[str, torch.Tensor],
            sequential: bool, forced_routes: torch.Tensor | None = None):
    embeds, mask, positions = frozen.prefix_embeds(inputs)
    prefix = frozen._backbone()(
        inputs_embeds=embeds,
        attention_mask=mask,
        position_ids=positions,
        use_cache=True,
        return_dict=True,
    )
    past = prefix.past_key_values
    queries = frozen.query_encoder.query_embeddings.to(
        device=embeds.device, dtype=embeds.dtype).unsqueeze(0)
    query_positions = _query_positions(positions, frozen.query_encoder.QUERY_COUNT)
    top_k = frozen._backbone().config.num_experts_per_tok
    if not sequential:
        query_mask = torch.ones((1, 4), device=mask.device, dtype=mask.dtype)
        outputs = frozen._backbone()(
            inputs_embeds=queries,
            attention_mask=torch.cat((mask, query_mask), dim=1),
            position_ids=query_positions,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
            output_router_logits=True,
            return_dict=True,
        )
        routes, margins = _routes(outputs, slice(None), top_k)
        return _hidden(outputs, slice(None)), routes, margins, query_positions

    hidden_steps, route_steps, margin_steps = [], [], []
    originals = []
    replay_state = {"slot": 0}
    if forced_routes is not None:
        for layer_index, layer in enumerate(frozen._backbone().layers):
            gate = layer.mlp.gate
            original = gate.forward
            originals.append((gate, original))

            def replay_forward(self, hidden_states, _original=original,
                               _layer=layer_index):
                router_logits, _, _ = _original(hidden_states)
                target = forced_routes[replay_state["slot"], _layer].to(
                    router_logits.device).reshape(1, -1)
                scores = router_logits.gather(-1, target)
                scores = scores / scores.sum(dim=-1, keepdim=True)
                return router_logits, scores.to(router_logits.dtype), target

            gate.forward = types.MethodType(replay_forward, gate)
    try:
        for slot in range(4):
            replay_state["slot"] = slot
            step_mask = torch.ones((1, slot + 1), device=mask.device, dtype=mask.dtype)
            outputs = frozen._backbone()(
                inputs_embeds=queries[:, slot:slot + 1],
                attention_mask=torch.cat((mask, step_mask), dim=1),
                position_ids=query_positions[..., slot:slot + 1],
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                output_router_logits=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            hidden_steps.append(_hidden(outputs, slice(None)).squeeze(0))
            routes, margins = _routes(outputs, slice(None), top_k)
            route_steps.append(
                forced_routes[slot].cpu() if forced_routes is not None
                else routes.squeeze(0))
            margin_steps.append(margins.squeeze(0))
    finally:
        for gate, original in originals:
            gate.forward = original
    return (
        torch.stack(hidden_steps),
        torch.stack(route_steps),
        torch.stack(margin_steps),
        query_positions,
    )


def _pair(left: torch.Tensor, right: torch.Tensor) -> list[dict[str, float]]:
    result = []
    for slot in range(4):
        x, y = left[slot, -1].float(), right[slot, -1].float()
        result.append({
            "cosine": float(torch.nn.functional.cosine_similarity(x, y, dim=0)),
            "relative_l2": float((x - y).norm() / x.norm()),
        })
    return result


def _route_compare(left: torch.Tensor, right: torch.Tensor) -> list[dict]:
    result = []
    for slot in range(4):
        mismatched_layers = []
        for layer in range(left.shape[1]):
            if set(left[slot, layer].tolist()) != set(right[slot, layer].tolist()):
                mismatched_layers.append(layer)
        result.append({
            "mismatched_layer_count": len(mismatched_layers),
            "first_mismatched_layer": mismatched_layers[0] if mismatched_layers else None,
            "mismatched_layers": mismatched_layers,
        })
    return result


def _q3_layerwise(left: torch.Tensor, right: torch.Tensor) -> list[dict]:
    result = []
    for layer in range(left.shape[1]):
        x, y = left[3, layer].float(), right[3, layer].float()
        result.append({
            "hidden_index": layer,
            "cosine": float(torch.nn.functional.cosine_similarity(x, y, dim=0)),
            "relative_l2": float((x - y).norm() / x.norm()),
        })
    return result


def _margin_summary(margins: torch.Tensor) -> list[dict[str, float | int]]:
    result = []
    for slot in range(4):
        values = margins[slot].float()
        result.append({
            "minimum": float(values.min()),
            "median": float(values.median()),
            "mean": float(values.mean()),
            "layers_below_1e-5": int((values < 1e-5).sum()),
            "layers_below_1e-4": int((values < 1e-4).sum()),
        })
    return result


@torch.inference_mode()
def _phase_identity_probe(frozen: FrozenQwenQueryForward,
                          inputs: dict[str, torch.Tensor]) -> dict:
    """Separate physical query position effects from embedding identity effects."""
    parameter = frozen.query_encoder.query_embeddings
    original = parameter.detach().clone()
    variants = {
        "original": [0, 1, 2, 3],
        "swap_adjacent": [1, 0, 3, 2],
        "rotate_left": [1, 2, 3, 0],
        "repeat_q0": [0, 0, 0, 0],
        "repeat_q1": [1, 1, 1, 1],
        "repeat_q2": [2, 2, 2, 2],
        "repeat_q3": [3, 3, 3, 3],
    }
    report = {}
    try:
        for name, indices in variants.items():
            parameter.copy_(original[indices])
            full = _full(frozen, inputs)
            sequential = _cached(frozen, inputs, sequential=True)
            report[name] = {
                "embedding_indices_by_position": indices,
                "full_vs_sequential": _pair(full[0], sequential[0]),
                "routes_full_vs_sequential": _route_compare(
                    full[1], sequential[1]),
                "full_router_margin": _margin_summary(full[2]),
                "sequential_router_margin": _margin_summary(sequential[2]),
            }
    finally:
        parameter.copy_(original)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--query-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vllm-capture", default="")
    parser.add_argument("--phase-identity-probe", action="store_true")
    args = parser.parse_args()

    from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

    row = json.loads(next(line for line in Path(args.manifest).read_text().splitlines()
                          if line.strip()))
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, attn_implementation="sdpa").eval()
    device = model.get_input_embeddings().weight.device
    query = DualVlnQueryEncoder.from_pretrained(args.query_checkpoint).to(
        device=device, dtype=torch.bfloat16)
    frozen = FrozenQwenQueryForward(model, query)
    inputs = build_slow_inputs(
        processor, row["slow_messages"], row["slow_completion"], device)

    full = _full(frozen, inputs)
    chunk = _cached(frozen, inputs, sequential=False)
    sequential = _cached(frozen, inputs, sequential=True)
    sequential_replay = _cached(
        frozen, inputs, sequential=True, forced_routes=full[1])
    vllm = load_file(args.vllm_capture) if args.vllm_capture else None
    vllm_replay = (_cached(
        frozen, inputs, sequential=True,
        forced_routes=vllm["query_routed_experts"])
        if vllm is not None else None)
    full_vllm_replay = (_full(
        frozen, inputs, forced_query_routes=vllm["query_routed_experts"])
        if vllm is not None else None)
    names = ["full", "chunk", "sequential", "sequential_replay"]
    values = [full, chunk, sequential, sequential_replay]
    if vllm_replay is not None:
        names.extend(("vllm_route_replay", "full_vllm_route_replay"))
        values.extend((vllm_replay, full_vllm_replay))
    tensors = {}
    for name, (hidden, routes, margins, positions) in zip(names, values):
        tensors[f"{name}_layer_hidden"] = hidden
        tensors[f"{name}_routed_experts"] = routes.to(torch.int32)
        tensors[f"{name}_router_margins"] = margins
        tensors[f"{name}_query_positions"] = positions.detach().cpu()

    report = {
        "schema": "action_head_q3_path_diagnostic_v2",
        "model": str(Path(args.model).resolve()),
        "query_checkpoint": str(Path(args.query_checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "full_vs_chunk": _pair(full[0], chunk[0]),
        "full_vs_sequential": _pair(full[0], sequential[0]),
        "full_vs_sequential_replay": _pair(full[0], sequential_replay[0]),
        "routes_full_vs_chunk": _route_compare(full[1], chunk[1]),
        "routes_full_vs_sequential": _route_compare(full[1], sequential[1]),
        "routes_full_vs_sequential_replay": _route_compare(
            full[1], sequential_replay[1]),
        "q3_full_vs_sequential_layerwise": _q3_layerwise(
            full[0], sequential[0]),
        "q3_full_vs_sequential_replay_layerwise": _q3_layerwise(
            full[0], sequential_replay[0]),
    }
    if vllm is not None:
        report["full_vs_vllm"] = _pair(full[0], vllm["query_hidden"][:, None, :])
        report["routes_full_vs_vllm"] = _route_compare(
            full[1], vllm["query_routed_experts"])
        report["sequential_vs_vllm"] = _pair(
            sequential[0], vllm["query_hidden"][:, None, :])
        report["routes_sequential_vs_vllm"] = _route_compare(
            sequential[1], vllm["query_routed_experts"])
        report["vllm_route_replay_vs_vllm"] = _pair(
            vllm_replay[0], vllm["query_hidden"][:, None, :])
        report["full_vs_vllm_route_replay"] = _pair(full[0], vllm_replay[0])
        report["full_vllm_route_replay_vs_vllm"] = _pair(
            full_vllm_replay[0], vllm["query_hidden"][:, None, :])
        report["full_vs_full_vllm_route_replay"] = _pair(
            full[0], full_vllm_replay[0])
        report["routes_full_vllm_replay_vs_vllm"] = _route_compare(
            full_vllm_replay[1], vllm["query_routed_experts"])
        report["vllm_query_positions"] = vllm["query_positions"].tolist()
    if args.phase_identity_probe:
        report["phase_identity_probe"] = _phase_identity_probe(frozen, inputs)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output / "diagnostic.safetensors"))
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
