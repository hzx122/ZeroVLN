#!/usr/bin/env python3
"""Real Qwen3.5/3.6 vision tower + random fast/slow action expert smoke."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from action_head.query_provider import VllmQueryHidden
from action_head.policy import Qwen35VisualTokenEncoder, TorchActionHeadPolicy
from action_head.runtime import OptionContext
from models.qwen36_fast_slow_action_head import (
    FastSlowActionHeadConfig,
    Qwen36FastSlowActionHead,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    from transformers import AutoConfig

    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    qwen_hidden_size = int(config.text_config.hidden_size)
    visual_hidden_size = int(config.vision_config.out_hidden_size)
    head_config = FastSlowActionHeadConfig(
        backbone_model_type=str(config.model_type),
        qwen_hidden_size=qwen_hidden_size,
        visual_hidden_size=visual_hidden_size,
        fusion_dim=256,
        action_dim=3,
        horizon=args.horizon,
        query_num_queries=4,
        visual_num_queries=4,
        visual_layers=1,
        fusion_layers=1,
        fusion_heads=8,
        flow_model_dim=256,
        flow_layers=2,
        flow_heads=8,
        inference_steps=3,
    )
    dtype = torch.bfloat16
    vision = Qwen35VisualTokenEncoder.from_pretrained(
        args.model_dir, device=args.device, dtype=dtype)
    model = Qwen36FastSlowActionHead(head_config).to(
        device=args.device, dtype=dtype)
    policy = TorchActionHeadPolicy(
        model, vision, device=args.device, dtype=dtype, seed=args.seed)

    queries = np.random.default_rng(args.seed).standard_normal(
        (4, qwen_hidden_size)).astype(np.float32)
    bundle = VllmQueryHidden(
        query_hidden=queries,
        request_id="synthetic-condition-wiring-only",
        source_path="synthetic",
    )
    reference = np.full((480, 640, 3), (80, 110, 140), dtype=np.uint8)
    current = np.full((480, 640, 3), (90, 120, 150), dtype=np.uint8)
    option = OptionContext(
        option_id="qwen36-35b-wiring",
        instruction="smoke instruction",
        contract={"target": "x", "where": "y", "plan": ["z"]},
        slow_condition=bundle,
        slow_reference_observation={"rgb": reference},
        current_observation={"rgb": current},
    )
    prediction = policy.predict(option)
    if prediction.shape != (args.horizon, 3) or not np.isfinite(prediction).all():
        raise AssertionError(f"invalid prediction: {prediction.shape}")

    output = Path(args.output_dir)
    model.save_pretrained(output / "action_head")
    report = {
        "ok": True,
        "scope": "wiring only; random hidden and random Action Head",
        "model_dir": str(Path(args.model_dir).resolve()),
        "model_type": str(config.model_type),
        "qwen_hidden_size": qwen_hidden_size,
        "visual_hidden_size": visual_hidden_size,
        "prediction_shape": list(prediction.shape),
        "prediction_finite": bool(np.isfinite(prediction).all()),
        "reference_cached": "reference_visual" in option.policy_cache,
        "head_config": head_config.to_dict(),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
