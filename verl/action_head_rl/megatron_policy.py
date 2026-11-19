"""Publish Megatron's HF-named inference weights for the existing vLLM service.

These are disposable inference weights. Exact resume remains the native
Megatron checkpoint (FP32 master weights, optimizer, scheduler and RNG).
"""
from __future__ import annotations

import json
import asyncio
import contextlib
import hashlib
import functools
import inspect
import os
import sys
import threading
from pathlib import Path
import time

import torch

from action_head_rl.checkpoint import hf_qwen_export_key, validate_hf_qwen_export_keys


def reference_shapes(model_path):
    """Read safetensor headers only, never load a second full HF model."""
    from safetensors import safe_open

    path = Path(model_path)
    index = path / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        weight_map = None
        shards = ["model.safetensors"]
    shapes = {}
    for shard in shards:
        with safe_open(path / shard, framework="pt", device="cpu") as source:
            for key in source.keys():
                if weight_map is not None and weight_map.get(key) != shard:
                    raise ValueError("source HF index/shard disagree: " + key)
                name = hf_qwen_export_key(key)
                if name in shapes:
                    raise ValueError("duplicate source HF key: " + name)
                shapes[name] = tuple(source.get_slice(key).get_shape())
    if weight_map is not None and len(shapes) != len(weight_map):
        raise ValueError("source HF index contains missing weights")
    validate_hf_qwen_export_keys(shapes)
    return shapes


def validate_export(state, shapes):
    validate_hf_qwen_export_keys(state)
    if state.keys() != shapes.keys():
        raise ValueError(f"Megatron HF export key mismatch: missing={sorted(shapes.keys()-state.keys())[:8]}, "
                         f"unexpected={sorted(state.keys()-shapes.keys())[:8]}")
    for name, tensor in state.items():
        if tuple(tensor.shape) != shapes[name]:
            raise ValueError(f"Megatron HF export shape mismatch: {name} {tuple(tensor.shape)} != {shapes[name]}")


def save_rollout_weights(state, *, model_path, destination, processor, step,
                         max_shard_size="5GB", mtp_disabled=False):
    """Reuse HF's serializer without model-class name conversion hooks."""
    from huggingface_hub import save_torch_state_dict
    from transformers import AutoConfig, GenerationConfig

    final = Path(destination)
    temporary = final.with_name(final.name + ".publishing")
    if final.exists() or temporary.exists():
        raise FileExistsError("refuse to overwrite a policy publication: " + str(final))
    shapes = reference_shapes(model_path)
    omitted = sorted(key for key in shapes if mtp_disabled and key.startswith("mtp."))
    shapes = {key: value for key, value in shapes.items() if key not in omitted}
    validate_export(state, shapes)
    temporary.mkdir(parents=True)
    # Bridge already returned canonical HF keys; no meta-model or another
    # model loader is needed just to serialize its complete converted state.
    save_torch_state_dict(state, str(temporary), max_shard_size=max_shard_size,
                          safe_serialization=True, force_contiguous=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if mtp_disabled:
        # agentic-nav HFModelConfig disables these when mtp.enable=False.
        # Persist the effective configuration, not the source's unused MTP head.
        for node in (config, getattr(config, "text_config", None)):
            for key in ("num_nextn_predict_layers", "mtp_num_hidden_layers"):
                if node is not None and hasattr(node, key):
                    setattr(node, key, 0)
    config.save_pretrained(temporary)
    if (Path(model_path) / "generation_config.json").exists():
        GenerationConfig.from_pretrained(model_path).save_pretrained(temporary)
    processor.save_pretrained(temporary)
    if reference_shapes(temporary) != shapes:
        raise RuntimeError("serialized policy layout changed during HF export")
    report = dict(schema="v8_megatron_qwen_rollout_transport_v1", step=int(step),
        source=str(model_path), weights=len(state), total_bytes=sum(t.numel()*t.element_size() for t in state.values()),
        publication_kind="rollout_transport", exact_resume=False, time=time.time(),
        mtp_disabled=mtp_disabled, omitted_inactive_source_weights=omitted)
    (temporary / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, final)
    return report


@contextlib.contextmanager
def ipc_exportable_allocations():
    """Allocate CUDA IPC buckets outside expandable segments.

    On the PPU stack, sharing an expandable-segment block fails with a CUDA
    driver error (probe_native_ipc_allocator.py). Follow VERL's colocated
    sync: disable only for the transfer, then restore the learner setting.
    """
    from verl.utils.device import set_expandable_segments
    enabled = torch.cuda.is_initialized() and torch.cuda.memory._snapshot().get(
        'allocator_settings', {}).get('expandable_segments')
    if not enabled:
        yield
        return
    set_expandable_segments(False)
    torch.cuda.empty_cache()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        set_expandable_segments(True)


_BRIDGE_EXPORT_LOCK = threading.RLock()


@contextlib.contextmanager
def completed_bridge_collectives():
    """Finish each Bridge collective, including multiple calls before a yield.

    Patch only the conversion module's torch reference, never global torch.
    The proxy is scoped to the iterator advance and its owning thread; the
    caller/IPC sender and other threads retain their normal collective APIs.
    Explicit Work.wait completes the host work as well as the device stream.
    Failed collectives propagate once; they are never replayed.
    """
    module = sys.modules.get('megatron.bridge.models.conversion.param_mapping')
    if module is None or not torch.cuda.is_initialized():
        yield
        return
    owner = threading.get_ident()
    with _BRIDGE_EXPORT_LOCK:
        original = module.torch

        class DistributedProxy:
            def __getattr__(self, name):
                function = getattr(original.distributed, name)
                if name not in ('all_gather', 'broadcast', 'all_reduce', 'scatter', 'gather'):
                    return function

                @functools.wraps(function)
                def completed(*args, **kwargs):
                    if threading.get_ident() != owner:
                        return function(*args, **kwargs)
                    bound = inspect.signature(function).bind(*args, **kwargs)
                    requested_async = bound.arguments.get('async_op', False)
                    bound.arguments['async_op'] = True
                    work = function(*bound.args, **bound.kwargs)
                    work.wait()
                    torch.cuda.synchronize()
                    return work if requested_async else None

                return completed

        class TorchProxy:
            distributed = DistributedProxy()

            def __getattr__(self, name):
                return getattr(original, name)

        module.torch = TorchProxy()
        try:
            yield
        finally:
            module.torch = original


def completed_export_tensors(tensors):
    """Drain conversion and consumer work before advancing the Bridge iterator.

    A large IPC byte bucket can otherwise enqueue hundreds of small PP/TP
    collectives. On PPU this can exhaust the finite nccl_call pool before
    the sender fills its bucket and synchronizes. Do not retry collectives:
    a failed communicator must fail the publication and preserve its receipt.
    """
    cuda = torch.cuda.is_initialized()
    if cuda:
        torch.cuda.synchronize()
    iterator = iter(tensors)
    while True:
        try:
            with completed_bridge_collectives():
                item = next(iterator)
        except StopIteration:
            return
        yield item
        # The caller has now copied this tensor into its IPC bucket (and
        # refreshed the FM mirror). Finish those streams before conversion
        # allocates communication work for the next tensor.
        if cuda:
            torch.cuda.synchronize()


def publish_megatron_qwen(engine, **kwargs):
    """Standalone, synchronous publication (outside Ray's actor event loop)."""
    return asyncio.run(publish_megatron_qwen_async(engine, **kwargs))


async def publish_megatron_qwen_async(engine, *, model_path, destination, processor, step,
                                    tensor_consumer=None, native_sender=None):
    """Collective Bridge conversion; rank zero writes, all ranks see success."""
    dist = torch.distributed
    rank = dist.get_rank()
    if engine.model_config.mtp.enable:
        raise ValueError("v8 policy publication requires MTP disabled in training and rollout")
    tensors, peft = engine.get_per_tensor_param()
    if peft is not None:
        raise ValueError("v8 full-policy publication cannot publish only a LoRA adapter")
    state = {}
    seen = set()
    digest = hashlib.sha256()
    total_bytes = 0
    from action_head_rl.direct_weight_sync import record_metadata
    shapes = {key: shape for key, shape in reference_shapes(model_path).items()
              if not key.startswith('mtp.')}
    def converted_weights():
        nonlocal total_bytes
        for name, tensor in completed_export_tensors(tensors):
            name = hf_qwen_export_key(name)
            if name in seen or name not in shapes or tuple(tensor.shape) != shapes[name]:
                raise ValueError('invalid Bridge export key/shape: ' + name)
            seen.add(name)
            if tensor_consumer is not None:
                tensor_consumer(name, tensor)
            if rank == 0 and destination is not None:
                state[name] = tensor.detach().cpu()
            total_bytes += tensor.nbytes
            record_metadata(digest, name, tensor)
            yield name, tensor
        if seen != shapes.keys():
            raise RuntimeError('incomplete native Bridge policy export')
    if native_sender is None:
        if destination is None:
            raise ValueError('policy publication needs disk or native receivers')
        for _, tensor in converted_weights():
            del tensor
    else:
        with ipc_exportable_allocations():
            await native_sender.async_send_weights(converted_weights())
    result = [None]
    if rank == 0:
        try:
            report = (save_rollout_weights(state, model_path=model_path,
                destination=destination, processor=processor, step=step, mtp_disabled=True)
                if destination is not None else dict(step=int(step), exact_resume=False,
                    publication_kind='native_colocated_transport'))
            report['collective_completion'] = 'per_bridge_collective_work_wait_and_device_sync'
            report['transport'] = dict(tensor_count=len(seen), tensor_bytes=total_bytes,
                                       metadata_sha256=digest.hexdigest())
            result[0] = dict(ok=True, report=report)
        except Exception as exc:
            # Peers must not continue to vLLM reload after rank-zero I/O failed.
            result[0] = dict(ok=False, error=f"{type(exc).__name__}: {exc}")
    del state
    dist.broadcast_object_list(result, src=0)
    if not result[0]["ok"]:
        raise RuntimeError("Megatron policy publication failed: " + result[0]["error"])
    return result[0]["report"]
