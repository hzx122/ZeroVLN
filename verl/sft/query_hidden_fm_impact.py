#!/usr/bin/env python3
"""Measure the FM-head impact of swapping FSDP query hidden for vLLM hidden."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from models.qwen36_fast_slow_action_head import Qwen36FastSlowActionHead


def _pair(left: torch.Tensor, right: torch.Tensor, *, dtype=torch.float32) -> dict[str, float]:
    x, y = left.to(dtype=dtype).reshape(-1), right.to(dtype=dtype).reshape(-1)
    return {
        "cosine": float(torch.nn.functional.cosine_similarity(x, y, dim=0)),
        "relative_l2": float((x - y).norm() / x.norm()),
        "max_abs": float((x - y).abs().max()),
    }


def _metrics(prediction: torch.Tensor, target: torch.Tensor,
             mask: torch.Tensor) -> dict[str, float]:
    distance = (prediction.float() - target.float()).norm(dim=-1)[0]
    valid = mask[0].bool()
    last = int(valid.sum()) - 1
    return {
        "ade": float(distance[valid].mean()),
        "fde": float(distance[last]),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--fsdp-capture", required=True)
    parser.add_argument("--vllm-capture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    head = Qwen36FastSlowActionHead.from_pretrained(
        args.head, map_location="cpu").to(device=device, dtype=dtype).eval()
    sample = np.load(args.sample)
    fsdp_payload = torch.load(args.fsdp_capture, map_location="cpu",
                              weights_only=False)
    fsdp = fsdp_payload["query_hidden"].to(device=device, dtype=dtype)
    vllm = load_file(args.vllm_capture)["query_hidden"].unsqueeze(0).to(
        device=device, dtype=dtype)
    if fsdp.shape != vllm.shape or tuple(fsdp.shape[1:]) != (4, 2048):
        raise ValueError(
            f"query captures must both be [1,4,2048], got {fsdp.shape}/{vllm.shape}")

    common = {
        "reference_visual_tokens": torch.from_numpy(
            sample["reference_visual_tokens"]).unsqueeze(0).to(device, dtype),
        "current_visual_tokens": torch.from_numpy(
            sample["current_visual_tokens"]).unsqueeze(0).to(device, dtype),
        "reference_visual_mask": torch.ones(
            (1, sample["reference_visual_tokens"].shape[0]),
            device=device, dtype=torch.bool),
        "current_visual_mask": torch.ones(
            (1, sample["current_visual_tokens"].shape[0]),
            device=device, dtype=torch.bool),
    }
    target = torch.from_numpy(sample["target_actions"]).unsqueeze(0).to(
        device=device, dtype=dtype)
    mask = torch.from_numpy(sample["action_mask"]).unsqueeze(0).to(device)
    zero = torch.zeros_like(target)

    def evaluate(query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        inputs = {"query_hidden_states": query, **common}
        condition = head.encode_condition(**inputs)
        prediction = head.sample(initial_noise=zero, **inputs)
        generator = torch.Generator(device=device).manual_seed(20260827)
        loss = head.flow_matching_loss(condition, target, mask, generator)
        return condition, prediction, float(loss.float())

    fsdp_condition, fsdp_prediction, fsdp_loss = evaluate(fsdp)
    vllm_condition, vllm_prediction, vllm_loss = evaluate(vllm)
    slot_replacements = []
    for slot in range(4):
        mixed = fsdp.clone()
        mixed[:, slot] = vllm[:, slot]
        condition, prediction, loss = evaluate(mixed)
        slot_replacements.append({
            "slot": slot,
            "condition_vs_fsdp": _pair(fsdp_condition, condition),
            "trajectory_vs_fsdp": _pair(fsdp_prediction, prediction),
            "trajectory_vs_vllm": _pair(vllm_prediction, prediction),
            "metrics": _metrics(prediction, target, mask),
            "flow_loss": loss,
        })

    report = {
        "schema": "action_head_query_hidden_fm_impact_v1",
        "head": str(Path(args.head).resolve()),
        "sample": str(Path(args.sample).resolve()),
        "fsdp_capture": str(Path(args.fsdp_capture).resolve()),
        "vllm_capture": str(Path(args.vllm_capture).resolve()),
        "query_hidden": _pair(fsdp, vllm),
        "condition": _pair(fsdp_condition, vllm_condition),
        "trajectory": _pair(fsdp_prediction, vllm_prediction),
        "fsdp_metrics": _metrics(fsdp_prediction, target, mask),
        "vllm_metrics": _metrics(vllm_prediction, target, mask),
        "fsdp_flow_loss": fsdp_loss,
        "vllm_flow_loss": vllm_loss,
        "slot_replacements": slot_replacements,
        "noise": "zero ODE; flow loss seed=20260827",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
