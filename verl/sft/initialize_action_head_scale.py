#!/usr/bin/env python3
"""Build reproducible initialization artifacts for a config-defined v2 scale sweep."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from models.dualvln_query_encoder import DualVlnQueryEncoder
from models.qwen36_fast_slow_action_head import (
    FastSlowActionHeadConfig,
    Qwen36FastSlowActionHead,
)
from sft.action_head_scale_config import load_scale_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--scale", required=True)
    args = parser.parse_args()
    experiment = load_scale_experiment(args.experiment_config)
    spec = experiment.scale(args.scale)

    output = experiment.initialization_dir(spec.name)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    # This entry point initializes a new head. Explicit checkpoint loading for
    # warm-start/resume belongs to the training entry point, never this script.
    base_config = FastSlowActionHeadConfig.from_dict(json.loads(
        (Path(experiment.base_head_checkpoint)
         / Qwen36FastSlowActionHead.CONFIG_NAME).read_text()))
    config = replace(base_config, **spec.head_overrides)
    config.validate()
    torch.manual_seed(experiment.seed)
    head = Qwen36FastSlowActionHead(config)
    head.save_pretrained(output / "head")

    base_query = DualVlnQueryEncoder.from_pretrained(
        experiment.base_query_checkpoint)
    if base_query.hidden_size != config.qwen_hidden_size:
        raise ValueError("base query and head hidden sizes disagree")
    query = DualVlnQueryEncoder(
        base_query.hidden_size, query_count=config.query_num_queries)
    with torch.no_grad():
        available = min(base_query.query_count, query.query_count)
        preserved = (available if experiment.preserve_base_query_slots is None
                     else experiment.preserve_base_query_slots)
        if preserved > available:
            raise ValueError(
                "preserve_base_query_slots exceeds the available base/target query slots: "
                f"requested={preserved}, available={available}")
        if preserved:
            query.query_embeddings[:preserved].copy_(
                base_query.query_embeddings[:preserved])
        if query.query_count > preserved:
            generator = torch.Generator(device="cpu").manual_seed(
                experiment.extra_query_seed)
            query.query_embeddings[preserved:].normal_(
                mean=0.0, std=0.02, generator=generator)
    query.save_pretrained(output / "query")

    head_numel = sum(parameter.numel() for parameter in head.parameters())
    query_numel = sum(parameter.numel() for parameter in query.parameters())
    report = {
        "schema": "action_head_scale_initialization_v1",
        "scale": spec.name,
        "experiment_config": str(experiment.source),
        "experiment_config_sha256": experiment.sha256,
        "seed": experiment.seed,
        "extra_query_seed": experiment.extra_query_seed,
        "base_head": str(Path(experiment.base_head_checkpoint).resolve()),
        "base_query": str(Path(experiment.base_query_checkpoint).resolve()),
        "head_config": config.to_dict(),
        "head_parameter_count": head_numel,
        "query_parameter_count": query_numel,
        "trainable_parameter_count": head_numel + query_numel,
        "preserved_base_query_slots": preserved,
        "random_initialized_query_slots": query.query_count - preserved,
        "reused_base_head_weights": False,
        "max_steps": spec.max_steps,
        "velocity_output_weight_nonzero": int(
            torch.count_nonzero(head.velocity.output.weight).item()),
        "velocity_output_bias_nonzero": int(
            torch.count_nonzero(head.velocity.output.bias).item()),
    }
    if (report["velocity_output_weight_nonzero"]
            or report["velocity_output_bias_nonzero"]):
        raise RuntimeError("new scaled head did not preserve zero-output initialization")
    (output / "scale_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
