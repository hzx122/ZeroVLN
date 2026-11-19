"""Opt-in FSDP2 weight residency for one logical batch.

The optimizer must run outside this context: FSDP's original parameter shards
must be registered again before an update. No model, routing, or dtype changes
are made. Head-only retains all groups; Full retains budgeted decoder groups.
Private FSDP metadata is read to preserve per-module settings exactly;
parameter transitions use the existing FSDPModule setters and reshard method.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
import time

import torch
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule


_GIB = 1024 ** 3


@dataclass
class _Snapshot:
    module: FSDPModule
    state: object
    group: object
    post_forward_mesh: object
    backward: bool
    auto_forward: bool


def _padded_group_bytes(group):
    return sum(
        math.prod(parameter.padded_sharded_param_size)
        * group.mesh_info.shard_mesh_size * 2
        for parameter in group.fsdp_params)


def _preflight(engine):
    scope = getattr(engine.action_head_train_scope, "value", engine.action_head_train_scope)
    if scope != "head_only":
        raise ValueError("frozen weight residency requires head_only")
    if engine.engine_config.strategy != "fsdp2" or not isinstance(engine.module, FSDPModule):
        raise ValueError("frozen weight residency requires an FSDP2 root")
    module = engine.module
    for name in ("qwen", "query_encoder"):
        parameters = list(getattr(module, name).parameters())
        if not parameters or any(parameter.requires_grad for parameter in parameters):
            raise ValueError(f"frozen weight residency requires frozen {name} parameters")
        if name == "qwen" and any(parameter.dtype != torch.bfloat16 for parameter in parameters):
            raise ValueError("frozen weight residency requires BF16 Qwen storage")

    snapshots = []
    padded_weight_bytes = 0
    largest_group_bytes = 0
    for child in module.modules():
        if not isinstance(child, FSDPModule):
            continue
        state = child._get_fsdp_state()
        group = state._fsdp_param_group
        if group is None:
            continue
        if group.device.type != "cuda" or isinstance(group.offload_policy, CPUOffloadPolicy):
            raise ValueError("frozen weight residency requires CUDA parameters without CPU offload")
        if group.mp_policy.param_dtype != torch.bfloat16:
            raise ValueError("frozen weight residency requires BF16 FSDP parameter dtype")
        snapshots.append(_Snapshot(
            child, state, group, group.post_forward_mesh_info,
            group.reshard_after_backward, state._auto_reshard_after_forward))
        # FSDP's full parameter is a view into the padded all-gather output.
        # For example, 256 experts across 28 ranks occupy 280 experts' storage.
        # Use the actual local padded shape (also handles nonzero shard dims),
        # without invoking all_gather_inputs or allocating/converting tensors.
        group_bytes = _padded_group_bytes(group)
        padded_weight_bytes += group_bytes
        largest_group_bytes = max(largest_group_bytes, group_bytes)
    if not snapshots:
        raise ValueError("frozen weight residency found no FSDP2 parameter groups")
    parameters = list(module.parameters())  # DTensor.numel() is the global size.
    if any(parameter.dtype not in (torch.bfloat16, torch.float32) for parameter in parameters):
        raise ValueError("frozen weight residency supports only BF16/FP32 parameter storage")
    full_weight_bytes = sum(parameter.numel() * 2 for parameter in parameters)
    if padded_weight_bytes < full_weight_bytes:
        raise ValueError("frozen weight residency FSDP storage metadata does not cover all parameters")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    reserved_bytes = torch.cuda.memory_reserved()
    allocated_bytes = torch.cuda.memory_allocated()
    reclaimable_bytes = max(0, reserved_bytes - allocated_bytes)
    # Keep current shards/optimizer allocations, allow one extra largest-group
    # collective buffer, and leave space for activations, gradients and kernels.
    reserve_bytes = max(8 * _GIB, int(total_bytes * 0.1))
    required_bytes = padded_weight_bytes + largest_group_bytes + reserve_bytes
    available_bytes = free_bytes + reclaimable_bytes
    report = {
        "modules": len(snapshots), "full_weight_bytes": full_weight_bytes,
        "padded_weight_bytes": padded_weight_bytes,
        "padding_bytes": padded_weight_bytes - full_weight_bytes,
        "largest_group_bytes": largest_group_bytes, "reserve_bytes": reserve_bytes,
        "free_bytes": free_bytes, "reclaimable_bytes": reclaimable_bytes,
        "allocated_bytes": allocated_bytes, "reserved_bytes": reserved_bytes,
        "total_bytes": total_bytes,
        "available_bytes": available_bytes, "required_bytes": required_bytes,
        "parameter_dtype": "bfloat16", "train_scope": scope,
    }
    if required_bytes > available_bytes:
        raise ValueError(
            "frozen weight residency memory preflight failed: "
            f"required={required_bytes / _GIB:.2f} GiB "
            f"available={available_bytes / _GIB:.2f} GiB; {report}")
    return snapshots, report


def _check_all_ranks(engine, error, *, label="frozen weight residency"):
    # A rank with insufficient free memory must not leave its peers entering
    # the subsequent weight all-gathers alone.
    if torch.distributed.is_initialized():
        failed = torch.tensor(int(error is not None), device=torch.cuda.current_device())
        torch.distributed.all_reduce(
            failed, op=torch.distributed.ReduceOp.MAX,
            group=engine.get_data_parallel_group())
        if failed.item() and error is None:
            raise RuntimeError(f"{label} preflight failed on another rank")
    if error is not None:
        raise error


def _restore(snapshots):
    errors = []
    for saved in reversed(snapshots):
        # True also forces reshard if a failed forward left its training state
        # at FORWARD. reshard() itself is deliberately non-recursive.
        for operation in (
            lambda: saved.module.set_reshard_after_forward(True, recurse=False),
            lambda: saved.module.reshard(),
            lambda: saved.module.set_reshard_after_backward(saved.backward, recurse=False),
            lambda: saved.module.set_reshard_after_forward(
                saved.post_forward_mesh is not None, recurse=False),
        ):
            try:
                operation()
            except Exception as error:
                errors.append(error)
        # Preserve partial-reshard mesh and the lazy-init auto flag, neither of
        # which is exposed by the public boolean setters.
        saved.group.post_forward_mesh_info = saved.post_forward_mesh
        saved.state._auto_reshard_after_forward = saved.auto_forward
    if errors:
        raise RuntimeError("failed to restore FSDP shards after frozen weight residency") from errors[0]


@contextmanager
def frozen_weight_residency(engine):
    """Keep weights through all option forwards/backwards, then restore shards."""
    if os.environ.get("ACTION_HEAD_FROZEN_WEIGHT_RESIDENCY", "0") != "1":
        yield None
        return
    snapshots, report, error = [], None, None
    try:
        snapshots, report = _preflight(engine)
    except Exception as caught:
        error = caught
    _check_all_ranks(engine, error)
    started = time.perf_counter()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    print({"event": "frozen_weight_residency_enter", "rank": rank, **report}, flush=True)
    try:
        for saved in snapshots:
            saved.module.set_reshard_after_forward(False, recurse=False)
            # Root post-backward also visits no_grad frozen groups. Disabling
            # only the forward flag would still reshard them after each option.
            saved.module.set_reshard_after_backward(False, recurse=False)
        yield report
    finally:
        _restore(snapshots)
        report["seconds"] = time.perf_counter() - started
        print({"event": "frozen_weight_residency_exit", "rank": rank,
               "shards_restored": True, "seconds": report["seconds"]}, flush=True)


def _future_full_optimizer_bytes(optimizer):
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("Full residency requires torch.optim.AdamW for its memory preflight")
    future_adam_bytes = future_gradient_bytes = 0
    seen = set()
    for group in optimizer.param_groups:
        if group.get("amsgrad", False):
            raise ValueError("Full residency does not support AMSGrad")
        for parameter in group["params"]:
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if parameter.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError("Full residency supports BF16/FP32 master parameters")
            # The optimizer owns original DTensor shards, not global tensors.
            local = getattr(parameter, "_local_tensor", parameter)
            size = local.numel() * parameter.element_size()
            state = optimizer.state.get(parameter, {})  # Do not create lazy state.
            for name in ("exp_avg", "exp_avg_sq"):
                value = state.get(name)
                if value is None:
                    future_adam_bytes += size
                elif value.dtype != parameter.dtype:
                    raise ValueError("Full residency requires Adam moments matching master parameter dtype")
            if parameter.grad is None:
                future_gradient_bytes += size
    return future_adam_bytes, future_gradient_bytes


def _full_preflight(engine, budget_gib):
    scope = getattr(engine.action_head_train_scope, "value", engine.action_head_train_scope)
    if scope != "full":
        raise ValueError("Full weight residency requires full training scope")
    if engine.engine_config.strategy != "fsdp2" or not isinstance(engine.module, FSDPModule):
        raise ValueError("Full weight residency requires an FSDP2 root")
    if getattr(engine.engine_config, "optimizer_offload", False):
        raise ValueError("Full weight residency requires GPU optimizer state")
    parameters = list(engine.module.parameters())
    master_dtypes = {parameter.dtype for parameter in parameters}
    if (not parameters or len(master_dtypes) != 1
            or not master_dtypes <= {torch.bfloat16, torch.float32}
            or any(not parameter.requires_grad for parameter in parameters)):
        raise ValueError("Full weight residency requires trainable uniform BF16/FP32 master parameters")

    snapshots, by_module = [], {}
    largest_group_bytes = largest_reduce_bytes = 0
    for child in engine.module.modules():
        if not isinstance(child, FSDPModule):
            continue
        state = child._get_fsdp_state()
        group = state._fsdp_param_group
        if group is None:
            continue
        if group.device.type != "cuda" or isinstance(group.offload_policy, CPUOffloadPolicy):
            raise ValueError("Full weight residency requires CUDA parameters without CPU offload")
        if (group.mp_policy.param_dtype != torch.bfloat16
                or group.mp_policy.reduce_dtype != torch.float32):
            raise ValueError("Full residency requires BF16 parameters and FP32 reduction")
        if not group.reduce_grads or not group.all_reduce_grads:
            raise ValueError("Full residency requires gradient synchronization on every backward")
        saved = _Snapshot(child, state, group, group.post_forward_mesh_info,
                          group.reshard_after_backward, state._auto_reshard_after_forward)
        snapshots.append(saved)
        by_module[id(child)] = saved
        size = _padded_group_bytes(group)
        largest_group_bytes = max(largest_group_bytes, size)
        # Padded FP32 reduce input plus the local FP32 reduce output.
        largest_reduce_bytes = max(
            largest_reduce_bytes, size * 2 + size * 2 // group.mesh_info.shard_mesh_size)

    candidates = []
    for index, layer in enumerate(engine.module.query_forward._backbone().layers):
        saved = by_module.get(id(layer))
        if saved is None:
            raise ValueError("Full residency requires individually sharded decoder layers")
        candidates.append((index, _padded_group_bytes(saved.group), saved))
    budget_bytes = int(budget_gib * _GIB)
    selected, plan, selected_bytes = [], [], 0
    for index, size, saved in sorted(candidates, key=lambda item: (-item[1], item[0])):
        if selected_bytes + size > budget_bytes:
            continue
        selected.append(saved)
        selected_bytes += size
        plan.append({"layer": index, "padded_bytes": size,
                     "shard_world_size": saved.group.mesh_info.shard_mesh_size,
                     "parameter_shapes": [list(p._orig_size) for p in saved.group.fsdp_params]})
    if not selected:
        raise ValueError("Full residency budget does not fit any decoder FSDP group")
    selection_hash = hashlib.sha256(json.dumps(
        {"budget_bytes": budget_bytes, "groups": plan},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    future_adam_bytes, future_gradient_bytes = _future_full_optimizer_bytes(engine.optimizer)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    allocated_bytes, reserved_bytes = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    reclaimable_bytes = max(0, reserved_bytes - allocated_bytes)
    activation_reserve_bytes = max(8 * _GIB, int(total_bytes * 0.1))
    # Preserve room for unselected groups' all-gathers as well as every group's
    # FP32 RS input/output. Future state protects the first, Adam-cold step.
    collective_reserve_bytes = largest_group_bytes + largest_reduce_bytes
    required_bytes = (selected_bytes + future_adam_bytes + future_gradient_bytes
                      + collective_reserve_bytes + activation_reserve_bytes)
    available_bytes = free_bytes + reclaimable_bytes
    report = {
        "train_scope": scope, "budget_bytes": budget_bytes,
        "master_parameter_dtype": str(parameters[0].dtype),
        "adam_moment_dtype": str(parameters[0].dtype),
        "resident_compute_dtype": "torch.bfloat16",
        "selected_layers": [entry["layer"] for entry in plan],
        "selected_groups": len(selected), "selected_padded_bytes": selected_bytes,
        "selection_hash": selection_hash, "gradient_sync": True,
        "future_adam_bytes": future_adam_bytes, "future_gradient_bytes": future_gradient_bytes,
        "largest_group_bytes": largest_group_bytes, "largest_fp32_reduce_bytes": largest_reduce_bytes,
        "collective_reserve_bytes": collective_reserve_bytes,
        "activation_reserve_bytes": activation_reserve_bytes,
        "free_bytes": free_bytes, "reclaimable_bytes": reclaimable_bytes,
        "allocated_bytes": allocated_bytes, "reserved_bytes": reserved_bytes,
        "device_used_bytes": total_bytes - free_bytes, "total_bytes": total_bytes,
        "available_bytes": available_bytes, "required_bytes": required_bytes,
    }
    if required_bytes > available_bytes:
        raise ValueError(f"Full residency memory preflight failed: {report}")
    return snapshots, selected, report


def _full_rank_consensus(engine, report):
    if torch.distributed.is_initialized():
        group = engine.get_data_parallel_group()
        rows = [None] * torch.distributed.get_world_size(group)
        torch.distributed.all_gather_object(rows, report, group=group)
    else:
        rows = [report]
    if len({row["selection_hash"] for row in rows}) != 1:
        raise RuntimeError("Full residency selection hash differs across ranks")
    return [{key: row[key] for key in (
        "rank", "selection_hash", "allocated_bytes", "reserved_bytes", "device_used_bytes",
        "free_bytes", "available_bytes", "required_bytes", "future_adam_bytes",
    )} for row in rows]


@contextmanager
def full_weight_residency(engine):
    """Budgeted Full decoder residency; keep every backward's reduce-scatter.

    The caller must distribute one enabled/disabled setting to the whole DP
    group. Disabled mode remains a no-op; all enabled ranks check the exact
    selection collectively before any residency flags change.
    """
    snapshots, selected, report, error = [], [], None, None
    try:
        budget = float(os.environ.get("ACTION_HEAD_FULL_WEIGHT_RESIDENCY_GIB", "0"))
        if not math.isfinite(budget) or budget < 0:
            raise ValueError("Full residency budget must be finite and non-negative")
    except Exception as caught:
        error = caught
    if error is None and budget == 0:
        yield None
        return
    if error is None:
        try:
            snapshots, selected, report = _full_preflight(engine, budget)
        except Exception as caught:
            error = caught
    _check_all_ranks(engine, error, label="Full residency")
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    report["rank"] = rank
    report["all_rank_memory"] = _full_rank_consensus(engine, report)
    started = time.perf_counter()
    print({"event": "full_weight_residency_enter", **report}, flush=True)
    try:
        for saved in selected:
            saved.module.set_reshard_after_forward(False, recurse=False)
            saved.module.set_reshard_after_backward(False, recurse=False)
        yield report
    finally:
        # Include unselected groups so an interrupted forward cannot leave the
        # optimizer inspecting a partially unsharded model.
        _restore(snapshots)
        if any(not saved.group.reduce_grads or not saved.group.all_reduce_grads
               for saved in snapshots):
            raise RuntimeError("Full residency gradient synchronization changed inside the context")
        report["seconds"] = time.perf_counter() - started
        print({"event": "full_weight_residency_exit", "rank": rank,
               "shards_restored": True, "gradient_sync": True,
               "seconds": report["seconds"]}, flush=True)


@contextmanager
def optimizer_cpu_during_backward(engine, *, enabled=None):
    """Keep native Adam state on CPU except during its unchanged GPU step.

    Reuses the native FSDP optimizer transfer helpers. Parameters, gradients,
    clipping, learning-rate scheduling and RNG are untouched. The caller must
    close weight-residency contexts before optimizer_step. Disabled mode never
    inspects the engine; this primitive is not enabled by current launch plans.
    """
    if enabled is None:
        enabled = os.environ.get('ACTION_HEAD_OPTIMIZER_CPU_DURING_BACKWARD', '0') == '1'
    if not enabled:
        yield None
        return
    error = None
    try:
        scope = getattr(engine.action_head_train_scope, 'value', engine.action_head_train_scope)
        if scope != 'full' or engine.engine_config.strategy != 'fsdp2':
            raise ValueError('optimizer CPU residency requires Full FSDP2 training')
        if not isinstance(engine.module, FSDPModule):
            raise ValueError('optimizer CPU residency requires an FSDP2 root')
        if (getattr(engine.engine_config, 'optimizer_offload', False)
                or getattr(engine.engine_config, 'param_offload', False)):
            raise ValueError('manual optimizer residency cannot mix with automatic offload')
        if any(p.dtype != torch.float32 for p in engine.module.parameters() if p.requires_grad):
            raise ValueError('optimizer CPU residency requires unchanged FP32 master parameters')
        if engine.optimizer is None:
            raise ValueError('optimizer CPU residency requires the native optimizer')
    except Exception as caught:
        error = caught
    _check_all_ranks(engine, error, label='optimizer CPU residency')
    from verl.utils.fsdp_utils import load_fsdp_optimizer, offload_fsdp_optimizer

    report = dict(optimizer_step_calls=0, optimizer_to_cpu_calls=0,
                  optimizer_to_gpu_calls=0, optimizer_to_cpu_seconds=0.0,
                  optimizer_to_gpu_seconds=0.0, closed=False,
                  model_moved=False, gradients_moved=False, native_adam=True)
    state_tensors = [value for state in engine.optimizer.state.values()
                     for value in state.values() if torch.is_tensor(value)]
    report['entry_state_tensor_count'] = len(state_tensors)
    report['entry_state_local_bytes'] = sum(
        (value.to_local() if hasattr(value, 'to_local') else value).numel()
        * value.element_size() for value in state_tensors)
    # Do not retain aliases to the original GPU moments during CPU residency.
    del state_tensors

    def move(device):
        started = time.perf_counter()
        # FSDPEngine.to documents manual control, but BaseEngine.to asserts
        # model=True for optimizer transfers. Its existing lower-level helpers
        # move only moments without violating that model/gradient contract.
        if device == 'cpu':
            offload_fsdp_optimizer(engine.optimizer)
        else:
            load_fsdp_optimizer(engine.optimizer, device)
        # The upstream transfer helper uses non_blocking=True. Complete D2H
        # before a checkpoint/snapshot reads CPU moments, and time H2D fully.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        direction = 'cpu' if device == 'cpu' else 'gpu'
        report[f'optimizer_to_{direction}_calls'] += 1
        report[f'optimizer_to_{direction}_seconds'] += time.perf_counter() - started

    def move_checked(device):
        error = None
        try:
            move(device)
        except Exception as caught:
            error = caught
            if device != 'cpu':
                # A partially loaded rank must release its moments before the
                # all-rank failure barrier; peers must not enter grad clipping.
                try:
                    move('cpu')
                except Exception as cleanup_error:
                    caught.add_note(f'optimizer transfer cleanup failed: {cleanup_error}')
        _check_all_ranks(engine, error, label=f'optimizer transfer to {device}')

    move_checked('cpu')
    had_own_step = 'optimizer_step' in vars(engine)
    original_own_step = vars(engine).get('optimizer_step')
    original_step = engine.optimizer_step

    def native_step(*args, **kwargs):
        try:
            move_checked('cuda')
            report['optimizer_step_calls'] += 1
            return original_step(*args, **kwargs)
        finally:
            move_checked('cpu')

    engine.optimizer_step = native_step
    try:
        yield report
    finally:
        # Restore the exact instance/class binding even if backward, transfer,
        # native Adam, or final CPU cleanup fails. State stays on CPU for the
        # following wave/update; no unique optimizer state is discarded.
        try:
            move_checked('cpu')
        finally:
            if had_own_step:
                engine.optimizer_step = original_own_step
            else:
                del engine.optimizer_step
            report['closed'] = True
