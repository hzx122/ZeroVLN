#!/usr/bin/env python3
"""action expert SFT entry that reuses VERL's native SFT/TrainingWorker stack."""
from __future__ import annotations

import json
import hashlib
import math
import os
import random
import shutil
import sys
from datetime import timedelta
from pathlib import Path

def configure_rank_compile_cache(cache_root: str) -> None:
    """Isolate every compiler output before importing Torch/Triton consumers."""
    # RANK is global inside torchrun; LOCAL_RANK alone collides across nodes.
    cache_rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    rank_cache = Path(cache_root) / f"rank_{cache_rank}"
    directories = {
        "TRITON_CACHE_DIR": rank_cache / "triton",
        "TORCHINDUCTOR_CACHE_DIR": rank_cache / "inductor",
        "CUDA_CACHE_PATH": rank_cache / "cuda",
        "PYTORCH_KERNEL_CACHE_PATH": rank_cache / "kernels",
        "TORCH_EXTENSIONS_DIR": rank_cache / "extensions",
        "RTC_CACHE_PATH": rank_cache / "rtc",
        "PPU_LIB_RTC_CACHE_PATH": rank_cache / "ppu-rtc",
        "NUMBA_CACHE_DIR": rank_cache / "numba",
        "DG_CACHE_DIR": rank_cache / "deep-gemm",
        "DG_JIT_CACHE_DIR": rank_cache / "deep-gemm-jit",
    }
    # Keep temporary Unix-socket paths short, while avoiding shared LLVM output
    # locations between nodes/ranks and between independent cache roots.
    temp_base = Path(os.environ.get("TMPDIR", "/workspace/.bootstrap-tmp"))
    cache_id = hashlib.sha256(str(Path(cache_root).resolve()).encode()).hexdigest()[:8]
    directories["TMPDIR"] = temp_base / f"t{cache_id}" / cache_rank
    for key, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    import tempfile
    tempfile.tempdir = None # Discard any directory cached before rank isolation.


if cache_root := os.environ.get("ACTION_HEAD_LOCAL_RANK_CACHE_ROOT"):
    configure_rank_compile_cache(cache_root)

# Importing this hook has no Torch/FLA side effects. Apply it after binding
# LOCAL_RANK below: importing FLA can initialize the current CUDA device.
from sft.fla_l2norm_dynamic_nb import maybe_install_l2norm_dynamic_nb_patch

FLA_L2NORM_PATCH_AUDIT = {
    "enabled": os.environ.get("ACTION_HEAD_FLA_L2NORM_DYNAMIC_NB", "0") == "1",
    "status": ("pending" if os.environ.get("ACTION_HEAD_FLA_L2NORM_DYNAMIC_NB", "0") == "1"
               else "disabled"),
}

import hydra
import torch
from omegaconf import OmegaConf
from tensordict.tensorclass import NonTensorData
from torch.distributed.tensor import DTensor
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader

from sft.action_head_joint_dataset import ActionHeadAnchorCollator, ActionHeadAnchorDataset, iter_manifest_rows, load_anchor_splits
from action_head import protocol as ACTION_HEAD_PROTOCOL
from verl.trainer.sft_trainer import SFTTrainer
from verl.utils import tensordict_utils as tu
from verl.utils.distributed import destroy_global_process_group
from verl.utils.device import auto_set_device
from verl.workers.engine_workers import TrainingWorker
from sft.distributed_tensor_stats import unique_shard_weight


def configure_training_determinism() -> dict:
    """Apply the explicit process-wide policy before initializing CUDA."""
    value = os.environ.get("ACTION_HEAD_DETERMINISTIC_ALGORITHMS", "0")
    if value not in ("0", "1"):
        raise ValueError("ACTION_HEAD_DETERMINISTIC_ALGORITHMS must be 0 or 1")
    if value == "1":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=False)
    return {
        "requested": value == "1",
        "enabled": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


class IndexedAnchorDataset(Dataset):
    def __init__(self, dataset: ActionHeadAnchorDataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        valid = index >= 0
        return valid, self.dataset[index if valid else 0]


class PaddedShardSampler(Sampler[int]):
    """Shard without silently counting DistributedSampler padding copies."""

    def __init__(self, size: int, rank: int, world_size: int, *,
                 shuffle: bool, seed: int, batch_size: int = 1,
                 resume_steps: int = 0):
        self.size, self.rank, self.world_size = size, rank, world_size
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.samples_per_rank = (
            math.ceil(size / (world_size * self.batch_size)) * self.batch_size)
        if resume_steps < 0:
            raise ValueError("resume_steps must be non-negative")
        self.resume_epoch, self.resume_step_in_epoch = divmod(
            resume_steps, self.samples_per_rank // self.batch_size)

    def __len__(self):
        return self.samples_per_rank

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        indices = list(range(self.size))
        if self.shuffle:
            random.Random(self.seed + self.resume_epoch + self.epoch).shuffle(indices)
        indices.extend([-1] * (self.samples_per_rank * self.world_size - self.size))
        local_indices = indices[self.rank::self.world_size]
        if self.epoch == 0:
            local_indices = local_indices[self.resume_step_in_epoch * self.batch_size:]
        return iter(local_indices)


class PackedAnchorCollator:
    """Pack heterogeneous anchor tensors behind one TensorDict batch axis."""

    def __init__(self, processor, seed: int, dataset_size: int):
        if os.environ.get('ACTION_HEAD_NATIVE_CACHE'):
            from sft.native_feature_dataset import NativeFeatureCollator
            self.inner = NativeFeatureCollator(processor.tokenizer.pad_token_id, seed)
        else:
            self.inner = ActionHeadAnchorCollator(processor, seed, dataset_size)

    def __call__(self, features):
        if not features:
            raise ValueError("Action Head received an empty anchor batch")
        valid, groups = zip(*features)
        batch = self.inner(list(groups))
        batch["anchor_valid"] = torch.tensor(valid, dtype=torch.float32)
        return {key: value.unsqueeze(0) if torch.is_tensor(value) else value
                for key, value in batch.items()}


def _full_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.full_tensor() if isinstance(value, DTensor) else value


def _action_head_key(name: str) -> str | None:
    if name.startswith("action_head."):
        return name.removeprefix("action_head.")
    marker = ".action_head."
    return name.split(marker, 1)[1] if marker in name else None


def _native_condition_key(name: str) -> str | None:
    marker = "context_aggregator."
    return name.split(marker, 1)[1] if marker in name else None


def finalize_validation_report(report: dict, initial: dict | None,
                               final: dict | None, *, configured: bool) -> dict:
    """Keep checkpoint audits mandatory when a train-only manifest has no val."""
    validation_ok = not configured or bool(
        initial and final and all(
            math.isfinite(float(result[key]))
            for result in (initial, final)
            for key in ("flow_loss", "ade", "fde")))
    return {
        **report,
        "initial_validation": initial,
        "final_validation": final,
        "validation_metric_weighting": "equal across held-out trajectory windows",
        "validation_status": (
            "not_configured" if not configured else "passed" if validation_ok else "failed"),
        "ok": bool(report.get("ok") and validation_ok),
    }


class SmallActionHeadCheckpointHandler:
    """Publish query/head tensors only; never gather the frozen Qwen state."""

    def __init__(self, engine, output_dir: str, action_head_as_tool_version: str):
        self.engine = engine
        self.output_dir = Path(output_dir)
        self.action_head_as_tool_version = ACTION_HEAD_PROTOCOL.normalize_version(
            action_head_as_tool_version)
        self.step_offset = int(os.environ.get("ACTION_HEAD_STEP_OFFSET", "0"))
        if self.step_offset < 0:
            raise ValueError("ACTION_HEAD_STEP_OFFSET must be non-negative")
        self.initial = {}
        for name, parameter in self.engine.module.named_parameters():
            if ("query_encoder.query_embeddings" in name
                    or _native_condition_key(name) is not None
                    or _action_head_key(name) is not None):
                local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
                self.initial[name] = local.detach().float().cpu().clone()

    def load_checkpoint(self):
        # Formal resume will be added only after the one-step VERL/FSDP gate.
        return 0

    def save_checkpoint(self, step: int):
        local_step = int(step)
        step = self.step_offset + local_step
        selected = {}
        query_delta_sq = torch.zeros((), device=torch.cuda.current_device())
        head_delta_sq = torch.zeros((), device=torch.cuda.current_device())
        for name, parameter in self.engine.module.named_parameters():
            if ("query_encoder.query_embeddings" in name
                    or _native_condition_key(name) is not None
                    or _action_head_key(name) is not None):
                local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
                before = self.initial[name].to(device=local.device)
                delta_sq = (local.detach().float() - before).pow(2).sum() * unique_shard_weight(
                    parameter, torch.distributed.get_world_size())
                if "query_encoder.query_embeddings" in name or _native_condition_key(name) is not None:
                    query_delta_sq += delta_sq
                else:
                    head_delta_sq += delta_sq
                # full_tensor is collective for FSDP2 DTensors, so all ranks call it.
                selected[name] = _full_tensor(parameter.detach()).float().cpu()
        torch.distributed.all_reduce(query_delta_sq, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(head_delta_sq, op=torch.distributed.ReduceOp.SUM)
        optimizer_ids = {
            id(parameter) for group in self.engine.optimizer.param_groups
            for parameter in group["params"]
        }
        qwen_parameters = [
            parameter for name, parameter in self.engine.module.named_parameters()
            if "query_encoder.query_embeddings" not in name
            and _native_condition_key(name) is None
            and _action_head_key(name) is None
        ]
        qwen_audit = {
            "requires_grad_count": sum(p.requires_grad for p in qwen_parameters),
            "optimizer_overlap_count": sum(id(p) in optimizer_ids for p in qwen_parameters),
            "non_none_grad_count": sum(p.grad is not None for p in qwen_parameters),
        }
        rank = torch.distributed.get_rank()
        if rank != 0:
            return
        if os.environ.get("ACTION_HEAD_CONDITION_MODE") == "v5_native":
            from sft.native_context_checkpoint import publish_native_checkpoint
            publish_native_checkpoint(
                self.output_dir, step, selected, qwen_audit,
                float(query_delta_sq.sqrt().cpu()), float(head_delta_sq.sqrt().cpu()),
                self.action_head_as_tool_version, self.engine)
            return
        query_items = [value for name, value in selected.items()
                       if "query_encoder.query_embeddings" in name]
        if len(query_items) != 1:
            raise RuntimeError(f"expected one query tensor, found {len(query_items)}")
        query_tensor = query_items[0]
        query_count = int(query_tensor.shape[0])
        head_state = {}
        for name, value in selected.items():
            key = _action_head_key(name)
            if key is not None:
                head_state[key] = value
        if not head_state:
            raise RuntimeError("FSDP checkpoint contains no Action Head tensors")

        # Each save lands in a step-tagged snapshot directory so periodic
        # save_freq checkpoints do not clobber one another (or the final
        # snapshot). Mirrors verl's global_step_{N} convention.
        snapshot_dir = self.output_dir / f"global_step_{int(step):06d}"
        temporary = snapshot_dir.with_name(snapshot_dir.name + ".publishing")
        if temporary.exists():
            shutil.rmtree(temporary)
        (temporary / "dualvln_query").mkdir(parents=True)
        (temporary / "action_head").mkdir(parents=True)
        torch.save({
            "query_embeddings": query_tensor,
            "query_count": query_count,
            "hidden_size": int(query_tensor.shape[1]),
            "hidden_convention": "final_norm",
        }, temporary / "dualvln_query" / "dualvln_query_embeddings.pt")
        source_head = Path(os.environ["ACTION_HEAD_HEAD_CHECKPOINT"])
        source_config = json.loads(
            (source_head / "fast_slow_action_head_config.json").read_text())
        if int(source_config["query_num_queries"]) != query_count:
            raise RuntimeError(
                "checkpoint export query/head count mismatch: "
                f"{query_count} != {source_config['query_num_queries']}")
        shutil.copy2(source_head / "fast_slow_action_head_config.json",
                     temporary / "action_head" / "fast_slow_action_head_config.json")
        torch.save(head_state, temporary / "action_head" / "fast_slow_action_head.pt")
        experiment_config = os.environ.get("ACTION_HEAD_EXPERIMENT_CONFIG", "")
        experiment_config_sha256 = os.environ.get(
            "ACTION_HEAD_EXPERIMENT_CONFIG_SHA256", "")
        if experiment_config:
            config_source = Path(experiment_config)
            if not config_source.is_file():
                raise FileNotFoundError(
                    f"training experiment config disappeared: {config_source}")
            shutil.copy2(config_source, temporary / "training_experiment.json")
        protocol = ACTION_HEAD_PROTOCOL.protocol_module(
            self.action_head_as_tool_version)
        report = {
            "schema": "action_head_verl_sft_report_v2",
            "backend": "verl_sft_training_worker_fsdp2",
            "global_step": step,
            "local_step": local_step,
            "step_offset": self.step_offset,
            "save_step_basis": os.environ.get("ACTION_HEAD_SAVE_STEP_BASIS", "local"),
            "save_freq": int(os.environ.get("ACTION_HEAD_SAVE_FREQ", "-1")),
            "checkpoint_steps": os.environ.get("ACTION_HEAD_CHECKPOINT_STEPS", ""),
            "optimizer_state_restored": False,
            "noise_seed_policy": os.environ.get("ACTION_HEAD_NOISE_SEED_POLICY", "legacy"),
            "lr_warmup_steps": int(os.environ.get("ACTION_HEAD_LR_WARMUP_STEPS", "0")),
            "clip_grad": float(os.environ.get("ACTION_HEAD_CLIP_GRAD", "1")),
            "data_resume_steps": int(os.environ.get("ACTION_HEAD_RESUME_DATA_STEPS", "0")),
            "warm_start_checkpoint": os.environ.get(
                "ACTION_HEAD_WARM_START_CHECKPOINT", ""),
            "parameter_delta_reference": "warm_start_boundary",
            "world_size": torch.distributed.get_world_size(),
            "anchors_per_gpu": int(os.environ.get("ACTION_HEAD_ANCHORS_PER_GPU", "1")),
            "global_batch_size": self.engine.get_data_parallel_size() * int(
                os.environ.get("ACTION_HEAD_ANCHORS_PER_GPU", "1")),
            "device_mesh_shape": list(self.engine.device_mesh.mesh.shape),
            "configured_fsdp_size": int(self.engine.engine_config.fsdp_size),
            "reshard_after_forward": bool(self.engine.engine_config.reshard_after_forward),
            "fsdp_resident_max_tokens": int(os.environ.get("ACTION_HEAD_FSDP_RESIDENT_MAX_TOKENS", "0")),
            "compile_cache_paths": {key: os.environ.get(key) for key in (
                "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH",
                "PYTORCH_KERNEL_CACHE_PATH", "TORCH_EXTENSIONS_DIR", "TMPDIR")},
            "fla_l2norm_dynamic_nb": FLA_L2NORM_PATCH_AUDIT,
            "fm_batch_windows": os.environ.get("ACTION_HEAD_FM_BATCH_WINDOWS", "0") == "1",
            "train_profile": os.environ.get("ACTION_HEAD_TRAIN_PROFILE", "custom"),
            "cpu_prefetch": os.environ.get("ACTION_HEAD_CPU_PREFETCH", "0") == "1",
            "query_suffix_backward": os.environ.get("ACTION_HEAD_QUERY_SUFFIX_BACKWARD", "0") == "1",
            "frozen_query_backend": getattr(
                getattr(self.engine.module, "query_forward", None), "frozen_query_backend", None),
            "frozen_query_backend_report": getattr(
                getattr(self.engine.module, "query_forward", None), "backend_report", None),
            "frozen_query_trainable_contract": getattr(self.engine, "frozen_query_trainable_contract", None),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "anchor_microbatch": os.environ.get("ACTION_HEAD_ANCHOR_MICROBATCH", "0") == "1",
            "slow_batch_mode": os.environ.get("ACTION_HEAD_SLOW_BATCH_MODE", "sequential"),
            "pack_anchors": int(os.environ.get("ACTION_HEAD_PACK_ANCHORS", "2")),
            "pack_max_tokens": int(os.environ.get("ACTION_HEAD_PACK_MAX_TOKENS", "12000")),
            "pack_execution": os.environ.get("ACTION_HEAD_PACK_EXECUTION", "layerwise"),
            "pack_conv_backend": os.environ.get("ACTION_HEAD_PACK_CONV_BACKEND", "segmented"),
            "gc_skip_every": int(os.environ.get("ACTION_HEAD_GC_SKIP_EVERY", "0")),
            "gc_max_tokens": int(os.environ.get("ACTION_HEAD_GC_MAX_TOKENS", "5120")),
            "query_count": query_count,
            "query_parameter_count": int(query_tensor.numel()),
            "action_head_parameter_count": int(sum(
                value.numel() for value in head_state.values())),
            "trainable_parameter_count": int(
                query_tensor.numel()
                + sum(value.numel() for value in head_state.values())),
            "trainable_parameter_names": sorted(selected),
            "frozen_qwen_exported": False,
            "fast_visual_input_mode": getattr(
                self.engine.module, "fast_visual_input_mode", "preencoded"),
            "persisted_visual_features": getattr(
                self.engine.module, "fast_visual_input_mode", "preencoded") != "live_frozen_visual_v1",
            "fast_visual_backbone": getattr(self.engine.module, "fast_visual_backbone", ""),
            "query_parameter_delta_l2": float(query_delta_sq.sqrt().cpu()),
            "action_head_parameter_delta_l2": float(head_delta_sq.sqrt().cpu()),
            "qwen_audit": qwen_audit,
            "action_head_as_tool_version": self.action_head_as_tool_version,
            "scale_name": os.environ.get("ACTION_HEAD_SCALE_NAME", ""),
            "training_experiment_config_sha256": experiment_config_sha256,
            **(protocol.protocol_identity() if protocol is not None else {}),
        }
        report["ok"] = bool(
            report["query_parameter_delta_l2"] > 0
            and report["action_head_parameter_delta_l2"] > 0
            and all(value == 0 for value in qwen_audit.values()))
        (temporary / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        os.replace(temporary, snapshot_dir)
        # Mirror the parameter-delta diagnostics into TensorBoard so the
        # snapshot-by-snapshot drift of the latent query / FM head is visible
        # alongside the train/val curves. query_delta_sq/head_delta_sq were
        # already all_reduced (SUM) above, so rank-0 holds the global value.
        tb_dir = os.environ.get("TENSORBOARD_DIR")
        if tb_dir:
            from torch.utils.tensorboard import SummaryWriter
            run_name = os.path.basename(os.environ.get("ACTION_HEAD_OUTPUT_DIR", "run"))
            writer = SummaryWriter(os.path.join(tb_dir, run_name))
            try:
                writer.add_scalar("train/query_delta_l2",
                                  float(query_delta_sq.sqrt().cpu()), int(step))
                writer.add_scalar("train/head_delta_l2",
                                  float(head_delta_sq.sqrt().cpu()), int(step))
                writer.flush()
            finally:
                writer.close()


class _ValidatingCheckpointHandler:
    """Decouple periodic snapshots from fixed validation passes.

    VERL has one periodic hook named ``save_freq``. The launcher sets that
    hook to gcd(ACTION_HEAD_SAVE_FREQ, ACTION_HEAD_VAL_FREQ), and this wrapper
    decides whether each invocation should save, validate, or do both.
    """

    def __init__(self, inner, trainer):
        self._inner = inner
        self._trainer = trainer
        requested = os.environ.get("ACTION_HEAD_CHECKPOINT_STEPS", "")
        self._save_steps = None
        if requested:
            steps = tuple(int(value) for value in requested.split(","))
            final = int(trainer.total_training_steps)
            if (steps != tuple(sorted(set(steps))) or not steps
                    or steps[0] <= 0 or steps[-1] != final):
                raise ValueError("checkpoint steps must be sorted, unique, positive and end at final step")
            self._save_steps = frozenset(steps)

    def load_checkpoint(self):
        return self._inner.load_checkpoint()

    def should_trigger_checkpoint(self, step: int) -> bool:
        """Select save/validation steps before VERL synchronizes its cache."""
        if self._save_steps is not None:
            val_freq = int(os.environ.get("ACTION_HEAD_VAL_FREQ", "0"))
            return step in self._save_steps or (val_freq > 0 and step % val_freq == 0)
        if os.environ.get("ACTION_HEAD_SAVE_STEP_BASIS", "local") != "global":
            return self._trainer.save_freq > 0 and step % self._trainer.save_freq == 0
        save_freq = int(os.environ.get("ACTION_HEAD_SAVE_FREQ", "-1"))
        val_freq = int(os.environ.get("ACTION_HEAD_VAL_FREQ", "0"))
        return ((save_freq > 0 and self._trainer.logical_step(step) % save_freq == 0)
                or (val_freq > 0 and step % val_freq == 0))

    def save_checkpoint(self, step: int):
        trainer = self._trainer
        step = int(step)
        final_step = int(trainer.total_training_steps)
        save_freq = int(os.environ.get("ACTION_HEAD_SAVE_FREQ", "-1"))
        val_freq = int(os.environ.get("ACTION_HEAD_VAL_FREQ", "0"))
        basis = os.environ.get("ACTION_HEAD_SAVE_STEP_BASIS", "local")
        if basis not in ("local", "global"):
            raise ValueError("ACTION_HEAD_SAVE_STEP_BASIS must be local or global")
        save_step = trainer.logical_step(step) if basis == "global" else step
        should_save = step >= final_step or (save_freq > 0 and save_step % save_freq == 0)
        if self._save_steps is not None:
            should_save = step >= final_step or step in self._save_steps
        if should_save:
            # Collective on all ranks; rank 0 publishes the snapshot.
            self._inner.save_checkpoint(step)
        # The final-step validation is run by ActionHeadVerlSFTTrainer.fit()
        # after super().fit() returns; skip it here to avoid a double pass.
        if step >= final_step or getattr(trainer, "_validation_loader", None) is None:
            return
        # VAL_FREQ=0 preserves the historical behavior: validate whenever a
        # periodic snapshot is written. A positive value decouples the two.
        should_validate = (
            (val_freq > 0 and step % val_freq == 0)
            or (val_freq <= 0 and should_save)
        )
        if should_validate:
            logical_step = trainer.logical_step(step)
            result = trainer._fixed_validation(f"step_{logical_step}")
            trainer._log_validation_tb(logical_step, result)


class ActionHeadVerlSFTTrainer(SFTTrainer):
    """Thin specialization of the mature VERL SFT trainer."""

    @staticmethod
    def logical_step(local_step: int) -> int:
        """Map this process' update count onto the experiment-wide step axis."""
        return int(os.environ.get("ACTION_HEAD_STEP_OFFSET", "0")) + int(local_step)

    def _build_dataset(self):
        manifest = os.environ["ACTION_HEAD_MANIFEST"]
        version = ACTION_HEAD_PROTOCOL.normalize_version(str(self.config.action_head_as_tool_version))
        protocol = ACTION_HEAD_PROTOCOL.protocol_module(version)
        from sft.action_head_data_contract import (
            BEAM_SAMPLING, manifest_sampling_contract, validate_v6_training_row)
        sampling = manifest_sampling_contract(manifest) if version in ('v6', 'v8') else None

        def validate_row(row, index):
            if protocol is not None:
                protocol.assert_protocol_identity(row)
                if version == 'v6':
                    validate_v6_training_row(row)
                elif version == 'v8':
                    from sft.v8_rule_training_data import validate_training_row
                    validate_training_row(row)
                if row.get("schema") != protocol.BUNDLE_SCHEMA:
                    raise ValueError(f"manifest row {index} is not {protocol.BUNDLE_SCHEMA}")

        if os.environ.get('ACTION_HEAD_NATIVE_CACHE'):
            # Preserve the existing source protocol audit for the optional
            # legacy cached frontend; this branch is not used by v4.3 tuning.
            count = 0
            for index, row in enumerate(iter_manifest_rows(manifest)):
                validate_row(row, index)
                count += 1
            if not count:
                raise ValueError(f"{version} training manifest is empty")
            from sft.native_feature_dataset import NativeFeatureDataset
            self.train_dataset = IndexedAnchorDataset(NativeFeatureDataset(
                os.environ['ACTION_HEAD_NATIVE_CACHE'], os.environ['SLOW_MODEL'],
                source_manifest=manifest))
            self.val_dataset = None
            return

        if os.environ.get('ACTION_HEAD_PUBLISHED_INDEX'):
            from sft.index_case_branches import IndexedPublishedDataset
            if version not in ('v6', 'v8') or sampling == BEAM_SAMPLING:
                raise ValueError('published full-route index requires independent v6/v8 anchors')
            published = IndexedPublishedDataset(os.environ['ACTION_HEAD_PUBLISHED_INDEX'])
            if published.protocol_version != version:
                raise ValueError('published index and configured prompt protocol differ')
            self.train_dataset = IndexedAnchorDataset(published)
            groups = {}
        elif version == 'v6' and sampling == BEAM_SAMPLING:
            # Full validation and path selection run once before submission.
            # Each rank mmaps a compact index instead of decoding the full pool.
            from sft.index_case_branches import IndexedCaseBranchDataset
            self.train_dataset = IndexedAnchorDataset(
                IndexedCaseBranchDataset(manifest, seed=int(self.config.trainer.seed)))
            groups = {}
        else:
            groups = load_anchor_splits(manifest, validate_row)
            if not groups:
                raise ValueError(f"{version} training manifest is empty")
            self.train_dataset = IndexedAnchorDataset(
                ActionHeadAnchorDataset(manifest, "train", seed=int(self.config.trainer.seed),
                                        groups=groups.get("train", [])))
        selection = self.train_dataset.dataset.branch_selection
        if selection is not None and torch.distributed.get_rank() == 0:
            output = Path(os.environ['ACTION_HEAD_OUTPUT_DIR'])
            output.mkdir(parents=True, exist_ok=True)
            (output / 'training_branch_selection.json').write_text(
                json.dumps(selection, ensure_ascii=False, indent=2) + '\n')
            print(json.dumps({'event': 'training_branch_selection', **{
                key: value for key, value in selection.items() if key != 'selections'}},
                ensure_ascii=False), flush=True)
        try:
            # Optional deterministic subset for cheap, frequent inline val.
            # Same seed every snapshot => identical subset => comparable curve.
            val_limit = int(os.environ.get("VAL_SUBSET", "0"))
            val_seed = int(self.config.trainer.seed) + 1_000_000
            self.val_dataset = IndexedAnchorDataset(
                ActionHeadAnchorDataset(manifest, "validation",
                                        limit=val_limit, seed=val_seed,
                                        groups=groups.get("validation", [])))
        except ValueError:
            self.val_dataset = None

    def _build_engine(self):
        from verl.workers.engine_workers import TrainingWorkerConfig

        # Importing the external module registers action_head/{fsdp,fsdp2}.
        __import__("sft.verl_action_head_engine")
        config = TrainingWorkerConfig(
            model_type="action_head",
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
            profiler_config=self.profiler_config,
        )
        self.training_client = TrainingWorker(config=config)
        self.training_client.set_loss_fn(loss_fn=lambda **_: None)
        self.engine = self.training_client.engine

    def _build_dataloader(self):
        dp_rank = self.engine.get_data_parallel_rank()
        dp_size = self.engine.get_data_parallel_size()
        train_anchor_batch = int(os.environ.get("ACTION_HEAD_ANCHORS_PER_GPU", "1"))
        val_anchor_batch = int(os.environ.get("ACTION_HEAD_VAL_ANCHORS_PER_GPU", "1"))
        if train_anchor_batch <= 0 or val_anchor_batch <= 0:
            raise ValueError("anchors per GPU must be positive")
        self.global_batch_size = dp_size * train_anchor_batch
        self.train_batch_size_per_dp = train_anchor_batch
        resume_data_steps = int(os.environ.get("ACTION_HEAD_RESUME_DATA_STEPS", "0"))
        if os.environ.get('ACTION_HEAD_SOURCE_RATIO'):
            from sft.index_case_branches import BalancedSourceShardSampler
            ratio = os.environ['ACTION_HEAD_SOURCE_RATIO']
            if not os.environ.get('ACTION_HEAD_PUBLISHED_INDEX'):
                raise ValueError('balanced sampling requires a published index')
            if os.environ.get('ACTION_HEAD_EPISODE_SAMPLER', '0') == '1':
                from sft.index_case_branches import EpisodeSourceShardSampler
                self.train_sampler = EpisodeSourceShardSampler(
                    self.train_dataset.dataset.source_counts, self.train_dataset.dataset.source_episodes,
                    dp_rank, dp_size, weights=[int(x) for x in ratio.split(':')],
                    seed=int(self.config.trainer.seed), batch_size=train_anchor_batch,
                    resume_steps=resume_data_steps)
            else:
                if ratio != '1:1:2':
                    raise ValueError('legacy balanced sampler requires R2R:RxR:SAGE=1:1:2')
                self.train_sampler = BalancedSourceShardSampler(
                    self.train_dataset.dataset.source_counts, dp_rank, dp_size,
                    seed=int(self.config.trainer.seed), batch_size=train_anchor_batch,
                    resume_steps=resume_data_steps)
        else:
            self.train_sampler = PaddedShardSampler(
                len(self.train_dataset), dp_rank, dp_size, shuffle=True,
                seed=int(self.config.trainer.seed), batch_size=train_anchor_batch,
                resume_steps=resume_data_steps)
        self.collate_fn = PackedAnchorCollator(
            self.model_config.processor, int(self.config.trainer.seed),
            len(self.train_dataset))
        self.collate_fn.inner.calls = resume_data_steps * train_anchor_batch
        if dp_rank == 0:
            print(json.dumps({"event": "action_head_runtime_audit",
                "data_resume_steps": resume_data_steps,
                "optimizer_state_restored": False,
                "noise_seed_policy": os.environ.get("ACTION_HEAD_NOISE_SEED_POLICY", "legacy"),
                "lr_warmup_steps": int(os.environ.get("ACTION_HEAD_LR_WARMUP_STEPS", "0")),
                "clip_grad": float(os.environ.get("ACTION_HEAD_CLIP_GRAD", "1")),
                "fla_l2norm_dynamic_nb": FLA_L2NORM_PATCH_AUDIT,
                "fm_batch_windows": os.environ.get("ACTION_HEAD_FM_BATCH_WINDOWS", "0") == "1",
                "train_profile": os.environ.get("ACTION_HEAD_TRAIN_PROFILE", "custom"),
                "cpu_prefetch": os.environ.get("ACTION_HEAD_CPU_PREFETCH", "0") == "1",
                "query_suffix_backward": os.environ.get("ACTION_HEAD_QUERY_SUFFIX_BACKWARD", "0") == "1",
                "requested_frozen_query_backend": os.environ.get(
                    "ACTION_HEAD_FROZEN_QUERY_BACKEND", "legacy_suffix"),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "anchor_microbatch": os.environ.get("ACTION_HEAD_ANCHOR_MICROBATCH", "0") == "1",
                "slow_batch_mode": os.environ.get("ACTION_HEAD_SLOW_BATCH_MODE", "sequential"),
                "pack_anchors": int(os.environ.get("ACTION_HEAD_PACK_ANCHORS", "2")),
                "pack_max_tokens": int(os.environ.get("ACTION_HEAD_PACK_MAX_TOKENS", "12000")),
                "pack_execution": os.environ.get("ACTION_HEAD_PACK_EXECUTION", "layerwise"),
                "pack_conv_backend": os.environ.get("ACTION_HEAD_PACK_CONV_BACKEND", "segmented"),
                "reshard_after_forward": bool(self.engine.engine_config.reshard_after_forward),
                "fsdp_resident_max_tokens": int(os.environ.get("ACTION_HEAD_FSDP_RESIDENT_MAX_TOKENS", "0")),
                "gc_skip_every": int(os.environ.get("ACTION_HEAD_GC_SKIP_EVERY", "0")),
                "gc_max_tokens": int(os.environ.get("ACTION_HEAD_GC_MAX_TOKENS", "5120")),
                "cache_scope": os.environ.get("ACTION_HEAD_COMPILE_CACHE_SCOPE", "unspecified"),
                "cache_paths": {key: os.environ.get(key) for key in (
                    "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH",
                    "PYTORCH_KERNEL_CACHE_PATH", "TORCH_EXTENSIONS_DIR", "TMPDIR")}}), flush=True)
        self.train_dataloader = StatefulDataLoader(
            self.train_dataset, batch_size=train_anchor_batch, sampler=self.train_sampler,
            collate_fn=self.collate_fn, num_workers=0,
            pin_memory=os.environ.get('ACTION_HEAD_PIN_MEMORY') == '1',
            drop_last=False)
        if os.environ.get("ACTION_HEAD_CPU_PREFETCH", "0") == "1":
            from sft.anchor_prefetch import AnchorPrefetchLoader
            self.train_dataloader = AnchorPrefetchLoader(self.train_dataloader)
        if self.val_dataset is None:
            self.val_dataloader = None
        else:
            self.val_sampler = PaddedShardSampler(
                len(self.val_dataset), dp_rank, dp_size, shuffle=False,
                seed=int(self.config.trainer.seed), batch_size=val_anchor_batch)
            self.val_collate_fn = PackedAnchorCollator(
                self.model_config.processor,
                int(self.config.trainer.seed) + 1_000_000,
                len(self.val_dataset))
            self.val_dataloader = StatefulDataLoader(
                self.val_dataset, batch_size=val_anchor_batch, sampler=self.val_sampler,
                collate_fn=self.val_collate_fn,
                num_workers=0, pin_memory=False, drop_last=False)

    def _get_batch_seqlens(self, data):
        mask = data["attention_mask"]
        return mask.reshape(mask.shape[0], -1).sum(dim=-1).tolist()

    def _build_ckpt_handler(self):
        inner = SmallActionHeadCheckpointHandler(
            self.engine, os.environ["ACTION_HEAD_OUTPUT_DIR"],
            str(self.config.action_head_as_tool_version))
        # Wrap so each periodic save also runs a fixed validation pass and
        # mirrors val/{flow_loss,ade,fde} into TensorBoard, producing a val
        # curve across the 200-step snapshots (not only step 0 / final).
        self.ckpt_handler = _ValidatingCheckpointHandler(inner, self)

    @staticmethod
    def _sum_metric(value) -> float:
        if torch.is_tensor(value):
            return float(value.detach().float().sum().cpu())
        if isinstance(value, (list, tuple)):
            return sum(ActionHeadVerlSFTTrainer._sum_metric(item) for item in value)
        return float(value)

    def _log_validation_tb(self, step: int, result: dict | None) -> None:
        """Mirror fixed-validation flow_loss/ade/fde into the verl Tracking TB dir.

        The base SFTTrainer only writes per-step train/* scalars via Tracking;
        the held-out validation metrics come from _fixed_validation and would
        otherwise live only in report.json. Reuse torch's SummaryWriter against
        the same TENSORBOARD_DIR so train/val curves share one board.
        """
        if result is None or torch.distributed.get_rank() != 0:
            return
        tb_dir = os.environ.get("TENSORBOARD_DIR")
        if not tb_dir:
            return
        from torch.utils.tensorboard import SummaryWriter
        run_name = os.path.basename(os.environ.get("ACTION_HEAD_OUTPUT_DIR", "run"))
        writer = SummaryWriter(os.path.join(tb_dir, run_name))
        try:
            for key in ("flow_loss", "ade", "fde"):
                if key in result:
                    writer.add_scalar(f"val/{key}", float(result[key]), int(step))
            writer.flush()
        finally:
            writer.close()

    def _fixed_validation(self, checkpoint: str) -> dict | None:
        # Uses the persisted validation loader so periodic saves (called from
        # inside super().fit(), where self.val_dataloader is None) can still
        # run a fixed validation pass without re-enabling the lossy base loop.
        loader = getattr(self, "_validation_loader", None)
        if loader is None:
            return None
        if hasattr(self.train_dataloader, "wait_pending"):
            self.train_dataloader.wait_pending()
        # Reset the independent validation collator so initial/final flow
        # losses use exactly the same per-anchor/per-window noise draws.
        self.val_collate_fn.inner.calls = 0
        meta_info = {
            "use_remove_padding": self.config.model.use_remove_padding,
            "use_dynamic_bsz": self.config.data.use_dynamic_bsz,
            "max_token_len_per_gpu": self.config.data.max_token_len_per_gpu,
            "micro_batch_size_per_gpu": self.config.data.micro_batch_size_per_gpu,
            "temperature": 1.0,
            "global_batch_size": self.global_batch_size,
            "pad_mode": self.config.data.pad_mode,
            "pad_token_id": self.model_config.tokenizer.pad_token_id,
        }
        totals = {key: 0.0 for key in (
            "flow_loss_sum", "flow_windows", "ade_sum", "fde_sum", "metric_windows")}
        for val_data in loader:
            data = tu.get_tensordict(tensor_dict=val_data, non_tensor_dict=meta_info)
            seqlens = self._get_batch_seqlens(data)
            tu.assign_non_tensor(
                data, global_token_num=NonTensorData(seqlens))
            output = self.training_client.infer_batch(data)
            if self.engine.is_mp_src_rank_with_outputs():
                metrics = tu.get(output, "metrics")
                for key in totals:
                    totals[key] += self._sum_metric(metrics[key])
        torch.distributed.barrier()
        if torch.distributed.get_rank() != 0:
            return None
        if totals["flow_windows"] <= 0 or totals["metric_windows"] <= 0:
            raise RuntimeError("fixed validation produced no valid windows")
        if totals["flow_windows"] != totals["metric_windows"]:
            raise RuntimeError(f"validation window count mismatch: {totals}")
        result = {
            "checkpoint": checkpoint,
            "flow_loss": totals["flow_loss_sum"] / totals["flow_windows"],
            "ade": totals["ade_sum"] / totals["metric_windows"],
            "fde": totals["fde_sum"] / totals["metric_windows"],
            "windows": int(totals["metric_windows"]),
            "anchors": len(self.val_dataset),
            "noise_policy": "fixed_per_anchor_random_initial_noise_across_checkpoints",
            "decode": "random-initial-noise (per-anchor seed) ODE with configured inference_steps; matches online sampling",
            "metric_space": "normalized single-step ego action [dx, dy, dyaw]; not meters",
        }
        print(json.dumps({"fixed_validation": result}, ensure_ascii=False), flush=True)
        return result

    def fit(self):
        if os.environ.get("ACTION_HEAD_LOCAL_PROFILE"):
            from sft.local_training_profile import run_profile
            try:
                return run_profile(self)
            finally:
                if hasattr(self.train_dataloader, "close"):
                    self.train_dataloader.close()
        validation_loader = self.val_dataloader
        # Persist so _ValidatingCheckpointHandler (active during super().fit(),
        # where self.val_dataloader is None) can run periodic fixed validation.
        self._validation_loader = validation_loader
        initial = self._fixed_validation("initial") if validation_loader is not None else None
        self._log_validation_tb(step=self.logical_step(0), result=initial)
        # The base trainer only averages per-rank batch losses and cannot
        # aggregate window-weighted ADE/FDE. Run its mature train loop without
        # its lossy validation branch, then evaluate through the method above.
        self.val_dataloader = None
        try:
            super().fit()
        finally:
            if hasattr(self.train_dataloader, "close"):
                self.train_dataloader.close()
        self.val_dataloader = validation_loader
        final = self._fixed_validation("final") if validation_loader is not None else None
        final_step = self.logical_step(self.total_training_steps)
        self._log_validation_tb(step=final_step, result=final)
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            report_path = (
                Path(os.environ["ACTION_HEAD_OUTPUT_DIR"])
                / f"global_step_{final_step:06d}"
                / "report.json")
            report = json.loads(report_path.read_text())
            report = finalize_validation_report(
                report, initial, final, configured=validation_loader is not None)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        torch.distributed.barrier()


def main() -> None:
    config_dir = os.environ.get(
        "VERL_SFT_CONFIG_DIR",
        "/workspace/verl/verl/trainer/config")
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        config = hydra.compose(
            config_name="sft_trainer_engine", overrides=sys.argv[1:])
    auto_set_device(config)
    if torch.distributed.get_rank() == 0:
        print(OmegaConf.to_yaml(config, resolve=True))
    gate = os.environ.get("ACTION_HEAD_GATE", "train")
    version = ACTION_HEAD_PROTOCOL.normalize_version(
        str(config.action_head_as_tool_version))
    # Dataset construction validates every source row while grouping both
    # splits, before model initialization. Avoid decoding a multi-GB manifest
    # separately for the protocol audit, training and optional validation.
    if gate == "import":
        __import__("sft.verl_action_head_engine")
        if torch.distributed.get_rank() == 0:
            print(json.dumps({
                "ok": True,
                "gate": "verl_distributed_import",
                "world_size": torch.distributed.get_world_size(),
            }))
        destroy_global_process_group()
        return
    trainer = ActionHeadVerlSFTTrainer(config=config)
    if gate == "load":
        trainable = sum(parameter.numel() for parameter in
                        trainer.engine.module.parameters()
                        if parameter.requires_grad)
        if torch.distributed.get_rank() == 0:
            print(json.dumps({
                "ok": trainable > 0,
                "gate": "verl_fsdp2_model_load",
                "world_size": torch.distributed.get_world_size(),
                "trainable_local_numel": trainable,
            }))
        destroy_global_process_group()
        return
    trainer.fit()
    destroy_global_process_group()


if __name__ == "__main__":
    deterministic_audit = configure_training_determinism()
    if os.environ.get("RANK", "0") == "0":
        print(json.dumps({"event": "action_head_determinism", **deterministic_audit}), flush=True)
    # Bind NCCL to LOCAL_RANK while creating the process group. The shared
    # verl helper selects the CUDA device only afterwards; its first bare
    # barrier can therefore guess from global rank and hang on partial-GPU
    # DLC pods (for example the 2-GPU-per-node multinode smoke).
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    FLA_L2NORM_PATCH_AUDIT = maybe_install_l2norm_dynamic_nb_patch()
    torch.distributed.init_process_group(
        "nccl",
        timeout=timedelta(seconds=36000),
        init_method=os.environ.get("DIST_INIT_METHOD"),
        device_id=torch.device("cuda", local_rank),
    )
    main()
