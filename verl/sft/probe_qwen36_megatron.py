#!/usr/bin/env python3
"""Distributed capability probe for the action expert Megatron learner.

This intentionally stops after loading the frozen Qwen checkpoint.  It proves
that the selected MCore/Megatron bridge, model architecture, and TP/PP/EP mesh
work in the PPU image before the action expert loss is attached.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time

import torch
import torch.distributed as dist


def expose_transformer_engine_pytorch() -> None:
    """Bridge PPU's lazy TE namespace to the MCore 0.16 import contract."""
    # The image advertises a ``flash-attn-3`` distribution but only ships its
    # binary extension, not ``flash_attn_3.flash_attn_interface``.  TE treats
    # package metadata as proof that the Python API exists, so make this one
    # incomplete optional backend look absent while TE initializes.
    original_version = importlib.metadata.version

    def compatible_version(distribution_name: str) -> str:
        if distribution_name == "flash-attn-3":
            raise importlib.metadata.PackageNotFoundError(distribution_name)
        return original_version(distribution_name)

    importlib.metadata.version = compatible_version
    import transformer_engine as transformer_engine
    try:
        if not hasattr(transformer_engine, "pytorch"):
            import transformer_engine.pytorch as transformer_engine_pytorch
            transformer_engine.pytorch = transformer_engine_pytorch
    finally:
        importlib.metadata.version = original_version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-weights", action="store_true")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    expose_transformer_engine_pytorch()
    world = dist.get_world_size()
    model_parallel = args.tp * args.pp
    if world % model_parallel:
        raise ValueError(
            f"world_size={world} must be divisible by TP*PP={model_parallel}")
    from transformers import AutoConfig
    import megatron.core
    import mbridge
    from verl.models.mcore.mbridge import AutoBridge

    hf_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    started = time.perf_counter()
    bridge = AutoBridge.from_config(
        hf_config, dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16)
    bridge.set_extra_args(
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=args.pp,
        expert_model_parallel_size=args.ep,
        expert_tensor_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=args.tp > 1,
        variable_seq_lengths=True,
        moe_token_dispatcher_type="alltoall",
        moe_router_load_balancing_type="none",
    )
    report = {
        "schema": "action_head_megatron_probe_v1",
        "rank": dist.get_rank(),
        "local_rank": local_rank,
        "world_size": world,
        "architecture": list(getattr(hf_config, "architectures", []) or []),
        "model_type": hf_config.model_type,
        "bridge_type": type(bridge).__name__,
        "megatron_core_file": megatron.core.__file__,
        "mbridge_file": mbridge.__file__,
        "transformers_version": importlib.metadata.version("transformers"),
        "tp": args.tp,
        "pp": args.pp,
        "ep": args.ep,
        "dp": world // model_parallel,
        "device": torch.cuda.get_device_name(local_rank),
        "bridge_seconds": time.perf_counter() - started,
        "load_weights_requested": args.load_weights,
    }
    if args.load_weights:
        from verl.trainer.config import CheckpointConfig
        from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig
        from verl.workers.engine import EngineRegistry

        model_config = HFModelConfig(
            path=args.model, load_tokenizer=False, trust_remote_code=True)
        engine_config = McoreEngineConfig(
            forward_only=True,
            use_mbridge=True,
            vanilla_mbridge=True,
            dtype=args.dtype,
            tensor_model_parallel_size=args.tp,
            pipeline_model_parallel_size=args.pp,
            expert_model_parallel_size=args.ep,
            expert_tensor_parallel_size=1,
            context_parallel_size=1,
            sequence_parallel=args.tp > 1,
            param_offload=False,
            grad_offload=False,
            optimizer_offload=False,
            use_remove_padding=False,
        )
        engine = EngineRegistry.new(
            model_type="language_model", backend="megatron",
            model_config=model_config, engine_config=engine_config,
            optimizer_config=McoreOptimizerConfig(
                lr=3e-5, lr_decay_steps=1, total_training_steps=1),
            checkpoint_config=CheckpointConfig())
        load_started = time.perf_counter()
        engine.initialize()
        torch.cuda.synchronize()
        parameters = [parameter for chunk in engine.module
                      for parameter in chunk.parameters()]
        report.update({
            "load_seconds": time.perf_counter() - load_started,
            "local_parameter_tensors": len(parameters),
            "local_parameter_numel": sum(p.numel() for p in parameters),
            "local_trainable_numel": sum(p.numel() for p in parameters
                                           if p.requires_grad),
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        })
    gathered = [None for _ in range(world)] if dist.get_rank() == 0 else None
    dist.gather_object(report, gathered, dst=0)
    if dist.get_rank() == 0:
        print(json.dumps({"ranks": gathered}, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
