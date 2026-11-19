"""Opt-in BF16 transport preparation; no collectives or parameter mutation.

Casting commutes with concatenation, not with a Partial reduction.  Only the
known CUDA Shard/Replicate layout is eligible.  This is not a PPU validation.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard


def validate_bf16_gather_before_export(*, enabled: bool, transport_only: bool,
                                      qwen_export_dtype: str) -> None:
    if enabled and (not transport_only or qwen_export_dtype != "bf16"):
        raise ValueError("BF16 gather-before-export requires BF16 transport-only publication")


def qwen_transport_gather_info(value: torch.Tensor) -> dict:
    """Classify using metadata only, with no local shape/value/device-index test."""
    distributed = isinstance(value, DTensor)
    placements = tuple(value.placements) if distributed else ()
    mesh_metadata = None
    if distributed:
        local = value.to_local()
        if local.dtype != value.dtype or local.device.type != value.device.type:
            raise ValueError("DTensor and local shard dtype/device metadata disagree")
        mesh = value.device_mesh
        if mesh.mesh.device.type != "cpu":
            raise ValueError("DTensor mesh membership metadata must be on CPU")
        mesh_metadata = {
            "device_type": mesh.device_type,
            "shape": list(mesh.mesh.shape),
            "members": mesh.mesh.tolist(),
        }
    placement_metadata = []
    for placement in placements:
        entry = {"type": f"{type(placement).__module__}.{type(placement).__qualname__}"}
        if hasattr(placement, "dim"):
            entry["dim"] = int(placement.dim)
        placement_metadata.append(entry)
    if not distributed:
        reason = "not_dtensor"
    elif value.device.type != "cuda":
        reason = "not_cuda"
    elif not value.is_floating_point():
        reason = "not_floating"
    elif any(type(placement) not in (Shard, Replicate) for placement in placements):
        reason = "unsupported_placement"
    elif not any(type(placement) is Shard for placement in placements):
        reason = "no_shard"
    else:
        reason = "eligible"
    eligible = reason == "eligible"
    return {
        "kind": "dtensor" if distributed else "tensor",
        "input_dtype": str(value.dtype),
        "gather_dtype": str(torch.bfloat16 if eligible else value.dtype),
        "device_type": value.device.type,
        "global_shape": list(value.shape),
        "placements": placement_metadata,
        "mesh": mesh_metadata,
        "eligible": eligible,
        "reason": reason,
    }


def prepare_qwen_transport_gather(value: torch.Tensor, *, enabled: bool,
                                  transport_only: bool, qwen_export_dtype: str):
    """Return a detached, disposable gather input and its small audit record."""
    validate_bf16_gather_before_export(
        enabled=enabled, transport_only=transport_only,
        qwen_export_dtype=qwen_export_dtype)
    detached = value.detach()
    if not enabled:
        return detached, None
    info = qwen_transport_gather_info(value)
    if info["eligible"]:
        detached = detached.to(dtype=torch.bfloat16)
    return detached, info


def qwen_transport_gather_consensus(items) -> dict:
    """Build a fixed-size all-rank message; metadata errors also participate.

    The caller must exchange this message on every rank before any Qwen
    gather.  No tensor values, local shard shapes, or per-key lists are sent.
    """
    try:
        digest = hashlib.sha256()
        inputs, gathers, fallbacks = Counter(), Counter(), Counter()
        count = eligible = 0
        for name, value in items:
            info = qwen_transport_gather_info(value)
            digest.update(json.dumps([name, info], sort_keys=True,
                                     separators=(",", ":")).encode())
            digest.update(b"\n")
            count += 1
            eligible += int(info["eligible"])
            inputs[info["input_dtype"]] += 1
            gathers[info["gather_dtype"]] += 1
            if not info["eligible"]:
                fallbacks[info["reason"]] += 1
        return {"ok": True, "sha256": digest.hexdigest(), "audit": {
            "enabled": True, "tensor_count": count, "eligible_count": eligible,
            "input_dtype_counts": dict(inputs), "gather_dtype_counts": dict(gathers),
            "fallback_counts": dict(fallbacks),
        }}
    except Exception as exc:
        # Avoid a rank-local raise while peers are about to enter a collective.
        return {"ok": False, "error_type": type(exc).__name__}


def validate_qwen_transport_gather_consensus(messages) -> dict:
    """All ranks see the same messages and reject before the first Qwen gather."""
    if not messages or any(not isinstance(item, dict) or item.get("ok") is not True
                           for item in messages):
        raise RuntimeError("BF16 transport gather metadata failed on one or more ranks")
    first = messages[0]
    if any(item != first for item in messages[1:]):
        raise RuntimeError("BF16 transport gather metadata differs across ranks")
    return {**first["audit"], "metadata_sha256": first["sha256"],
            "verified_ranks": len(messages)}
