"""Resume helpers for action expert Ray GRPO checkpoints.

The directory/tracker convention intentionally matches VERL's diffusion
trainer.  action expert model weights stay in the existing lightweight export;
``trainer_state`` contains VERL's sharded optimizer, scheduler and RNG state.
"""
from __future__ import annotations

import json
import hashlib
import inspect
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path


_STEP_RE = re.compile(r"global_step_(\d+)")


def hf_qwen_export_key(name: str) -> str:
    """Restore the upstream HF visual-tower path after joint-policy wrapping."""
    prefix = "model.language_model.visual."
    if name.startswith(prefix):
        return "model.visual." + name[len(prefix):]
    return name


def validate_hf_qwen_export_keys(keys) -> None:
    """Fail if a serialized Qwen-VL state is not vLLM-loadable."""
    keys = tuple(keys)
    wrapped = "model.language_model.visual."
    if any(name.startswith(wrapped) for name in keys):
        raise ValueError("Qwen export still contains wrapped visual-tower keys")
    if not any(name.startswith("model.visual.") for name in keys):
        raise ValueError("Qwen export is missing the visual tower")


def normalize_saved_hf_qwen_visual_keys(qwen_dir: Path) -> int:
    """Atomically repair HF save hooks that re-nest the Qwen visual tower."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    qwen_dir = Path(qwen_dir)
    index_path = qwen_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    wrapped = "model.language_model.visual."
    affected = {
        shard for name, shard in weight_map.items() if name.startswith(wrapped)
    }
    for shard_name in affected:
        shard = qwen_dir / shard_name
        temporary = shard.with_name(shard.name + ".visual-remap.tmp")
        with safe_open(shard, framework="pt", device="cpu") as source:
            metadata = source.metadata()
            tensors = {
                hf_qwen_export_key(name): source.get_tensor(name)
                for name in source.keys()
            }
        save_file(tensors, temporary, metadata=metadata)
        os.replace(temporary, shard)
    normalized = {hf_qwen_export_key(name): shard
                  for name, shard in weight_map.items()}
    if len(normalized) != len(weight_map):
        raise ValueError("duplicate Qwen keys after serialized visual remap")
    validate_hf_qwen_export_keys(normalized)
    index["weight_map"] = normalized
    temporary_index = index_path.with_suffix(".json.visual-remap.tmp")
    temporary_index.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_index, index_path)
    return sum(name.startswith(wrapped) for name in weight_map)


def save_hf_qwen_export(model, qwen_dir: Path, state_dict: dict,
                        *, max_shard_size: str = "5GB") -> dict:
    """Avoid rewriting a visual shard after HF reverses its canonical names.

    This optimization is limited to Qwen3.5 MoE and a proven name-only
    conversion. Probe the installed conversion on tensor references, without
    copying their data: after visual normalization every key must still point
    to the original tensor. Other architectures/converters and older HF
    versions retain the existing save-and-repair path. Save failures propagate.
    The caller must release its state dictionary, then run the serialized-key
    normalizer, so legacy repair does not retain the full CPU model alongside
    its temporary shard.
    """
    export_state = state_dict
    save_options = {}
    reason = "other_architecture"
    classes = {cls.__name__ for cls in type(model).__mro__}
    if ("Qwen3_5MoeForConditionalGeneration" in classes
            and getattr(getattr(model, "config", None), "model_type", None)
            == "qwen3_5_moe"):
        parameter = inspect.signature(model.save_pretrained).parameters.get(
            "save_original_format")
        if parameter is None:
            reason = "save_original_format_not_explicitly_supported"
        else:
            # These imports are only needed for the supported HF API. Older
            # versions may expose the save argument but not these utilities.
            try:
                from transformers.conversion_mapping import get_model_conversion_mapping
                from transformers.core_model_loading import (
                    WeightRenaming, revert_weight_conversion,
                )
            except ImportError:
                reason = "conversion_introspection_unavailable"
            else:
                conversions = getattr(model, "_weight_conversions", None)
                if conversions is None:
                    conversions = get_model_conversion_mapping(model, add_legacy=False)
                if any(type(item) is not WeightRenaming for item in conversions):
                    reason = "non_renaming_weight_conversion"
                else:
                    converted = revert_weight_conversion(model, dict(state_dict))
                    canonical = {}
                    for name, tensor in converted.items():
                        name = hf_qwen_export_key(name)
                        if name in canonical:
                            raise ValueError("duplicate Qwen keys after in-memory visual remap")
                        canonical[name] = tensor
                    if (canonical.keys() == state_dict.keys()
                            and all(tensor is state_dict[name]
                                    for name, tensor in canonical.items())):
                        validate_hf_qwen_export_keys(canonical)
                        # Keep the reverse-conversion iteration order: shard
                        # membership stays the same as legacy save + repair.
                        export_state = canonical
                        save_options["save_original_format"] = False
                        reason = "canonical_tensor_identity_verified"
                    else:
                        reason = "conversion_changes_other_keys_or_tensors"

    model.save_pretrained(qwen_dir, state_dict=export_state,
                          safe_serialization=True, max_shard_size=max_shard_size,
                          **save_options)
    return {
        "serialization_mode": ("canonical_in_memory" if save_options
                               else "legacy_visual_remap"),
        "serialization_reason": reason,
    }


@dataclass(frozen=True)
class ResumeCheckpoint:
    path: Path
    step: int
    head: Path
    query: Path
    qwen: Path | None
    trainer_state: Path | None

    @property
    def exact(self) -> bool:
        return self.trainer_state is not None


@dataclass(frozen=True)
class PolicyWarmStart:
    """Joint weights and bound critic, deliberately without actor trainer state."""
    path: Path
    step: int
    head: Path
    query: Path
    qwen: Path
    source_contract: dict
    critic: object
    lineage: dict


def checkpoint_step(path: Path) -> int:
    match = _STEP_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"resume path must end in global_step_<N>: {path}")
    return int(match.group(1))


def _complete_trainer_state(path: Path, world_size: int) -> bool:
    joint = path / "joint_state.json"
    if joint.is_file():
        manifest = json.loads(joint.read_text())
        if (manifest.get("schema") != "v8_megatron_fm_joint_v1"
                or manifest.get("step") != checkpoint_step(path.parent)):
            raise ValueError("joint checkpoint schema/step mismatch")
        if not _complete_trainer_state(path / "fm", world_size):
            return False
        from torch.distributed.checkpoint import FileSystemReader
        text = path / "text"
        contents = json.loads((text / "ckpt_contents.json").read_text())
        if (contents.get("world_size") != world_size
                or contents.get("global_step") != manifest["step"]
                or contents.get("schema_version") != 2):
            raise ValueError("Megatron checkpoint topology/step/layout mismatch")
        for key in ("model", "optimizer", "lr_scheduler", "rng_state"):
            relative = Path(contents["contents"][key]["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("invalid Megatron checkpoint relative path")
            directory = text / relative
            metadata = FileSystemReader(directory).read_metadata()
            if not metadata.state_dict_metadata or not metadata.storage_data:
                return False
            for item in metadata.storage_data.values():
                shard = directory / item.relative_path
                if not shard.is_file() or shard.stat().st_size < item.offset + item.length:
                    return False
        return True
    config_path = path / "fsdp_config.json"
    if not config_path.is_file():
        return False
    try:
        saved_world = int(json.loads(config_path.read_text())["world_size"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if saved_world != world_size:
        raise ValueError(
            f"resume learner world size mismatch: checkpoint={saved_world}, "
            f"requested={world_size}")
    for prefix in ("optim", "extra_state"):
        expected = {
            f"{prefix}_world_size_{world_size}_rank_{rank}.pt"
            for rank in range(world_size)
        }
        present = {item.name for item in path.glob(f"{prefix}_world_size_{world_size}_rank_*.pt")}
        if present != expected:
            return False
    return True


def load_checkpoint_tool_critic(path: Path):
    """Validate the critic committed with this actor step, never a latest file."""
    from action_head_rl.tool_credit import LogisticStateCritic
    path = Path(path)
    step = checkpoint_step(path)
    report = json.loads((path / "report.json").read_text())
    metadata = report.get("tool_credit_critic")
    if (not isinstance(metadata, dict)
            or metadata.get("file") != "tool_credit_critic.json"
            or metadata.get("fitted_through_step") != step):
        raise ValueError("checkpoint lacks an aligned committed tool-credit critic")
    critic = LogisticStateCritic.load(path / metadata["file"])
    if (critic.fitted_through_step != step
            or critic.state_dict()["sha256"] != metadata.get("sha256")):
        raise ValueError("checkpoint critic step/hash mismatch")
    return critic


def resolve_resume_checkpoint(
    *, mode: str, output_dir: Path, resume_from_path: str,
    train_scope: str, learner_world_size: int,
    allow_weights_only: bool,
) -> ResumeCheckpoint | None:
    """Resolve and validate a VERL-style resume checkpoint."""
    if mode == "disable":
        if resume_from_path:
            raise ValueError("resume-from-path requires resume-mode=resume_path")
        return None
    if mode == "auto":
        value = find_latest_ckpt_path(
            str(output_dir), directory_format="global_step_{:06d}")
        if value is None:
            return None
        path = Path(value)
    elif mode == "resume_path":
        if not resume_from_path:
            raise ValueError("resume-mode=resume_path requires --resume-from-path")
        path = Path(resume_from_path).resolve()
    else:
        raise ValueError(f"unsupported resume mode: {mode}")

    if not path.is_dir():
        raise FileNotFoundError(path)
    step = checkpoint_step(path)
    report_path = path / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text())
    if report.get("publication_kind") == "rollout_transport":
        raise ValueError("rollout transport exports cannot be resumed, including weights-only resume")
    if int(report.get("step", -1)) != step:
        raise ValueError(f"resume report step mismatch: {report_path}")
    if report.get("train_scope") != train_scope:
        raise ValueError(
            f"resume train_scope mismatch: checkpoint={report.get('train_scope')}, "
            f"requested={train_scope}")

    head = path / "action_head"
    query = path / "dualvln_query"
    if not head.is_dir():
        raise FileNotFoundError(head)
    if not (query / "dualvln_query_embeddings.pt").is_file():
        raise FileNotFoundError(query / "dualvln_query_embeddings.pt")
    qwen = path / "qwen" if bool(report.get("qwen_exported")) else None
    if train_scope == "full" and (qwen is None or not qwen.is_dir()):
        raise ValueError("full-scope resume requires an exported qwen directory")

    state = path / "trainer_state"
    exact = state if state.is_dir() and _complete_trainer_state(
        state, learner_world_size) else None
    if exact is None and not allow_weights_only:
        raise ValueError(
            f"checkpoint has no complete optimizer/scheduler/RNG state: {state}; "
            "pass --allow-weights-only-resume for a legacy checkpoint")
    if "tool_credit_critic" in report:
        if exact is None:
            raise ValueError("tool-credit resume requires complete optimizer/scheduler/RNG state")
        load_checkpoint_tool_critic(path)
    return ResumeCheckpoint(
        path=path, step=step, head=head, query=query, qwen=qwen,
        trainer_state=exact)


def _validate_fp32_policy_payload(checkpoint: ResumeCheckpoint) -> str:
    """Inspect tensor headers, not 140 GB of Qwen data; return index hash."""
    import torch
    from safetensors import safe_open

    index_path = checkpoint.qwen / "model.safetensors.index.json"
    if not (checkpoint.qwen / "config.json").is_file():
        raise FileNotFoundError(checkpoint.qwen / "config.json")
    index_bytes = index_path.read_bytes()
    mapping = json.loads(index_bytes).get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("warm-start requires a complete indexed Qwen export")
    validate_hf_qwen_export_keys(mapping)
    shards = set(mapping.values())
    if any(not isinstance(name, str) or Path(name).name != name for name in shards):
        raise ValueError("invalid Qwen shard path")
    floats = 0
    for name in sorted(shards):
        with safe_open(checkpoint.qwen / name, framework="pt", device="cpu") as saved:
            expected = {key for key, shard in mapping.items() if shard == name}
            if set(saved.keys()) != expected:
                raise ValueError("Qwen index and shard keys differ")
            for key in saved.keys():
                dtype = saved.get_slice(key).get_dtype()
                if dtype.startswith(("F", "BF")):
                    if dtype != "F32":
                        raise ValueError("policy warm-start requires full FP32 Qwen weights")
                    floats += 1
    if not floats:
        raise ValueError("Qwen export has no FP32 tensors")

    def tensor_dtypes(value):
        if isinstance(value, torch.Tensor):
            return [value.dtype] if value.is_floating_point() else []
        if isinstance(value, dict):
            return [dtype for item in value.values() for dtype in tensor_dtypes(item)]
        if isinstance(value, (tuple, list)):
            return [dtype for item in value for dtype in tensor_dtypes(item)]
        return []

    for path in (checkpoint.head / "fast_slow_action_head.pt",
                 checkpoint.query / "dualvln_query_embeddings.pt"):
        dtypes = tensor_dtypes(torch.load(path, map_location="cpu", weights_only=True, mmap=True))
        if not dtypes or any(dtype != torch.float32 for dtype in dtypes):
            raise ValueError(f"policy warm-start requires full FP32 head/query weights: {path}")
    if not (checkpoint.head / "fast_slow_action_head_config.json").is_file():
        raise FileNotFoundError(checkpoint.head / "fast_slow_action_head_config.json")
    return hashlib.sha256(index_bytes).hexdigest()


def resolve_policy_warm_start(checkpoint_path: str, *, output_dir: Path) -> PolicyWarmStart:
    """Validate the source in its own topology, then start a new actor run.

    Exact-resume validation remains unchanged. Source optimizer/RNG files are
    checked for completeness but are never exposed to the target actor loader.
    """
    from action_head_rl.tool_credit import TOOL_CREDIT_MODES, rebase_critic_for_new_run

    path = Path(checkpoint_path)
    if not path.is_absolute():
        raise ValueError("policy warm-start checkpoint must be an absolute path")
    path = path.resolve()
    if output_dir.exists():
        raise ValueError("policy warm-start requires a new output directory")
    source_bytes = (path.parent / "contract.json").read_bytes()
    source = json.loads(source_bytes)
    if (source.get("harness_version") != "v6" or source.get("train_scope") != "full"
            or source.get("learner_parameter_dtype") != "fp32"
            or source.get("tool_credit", {}).get("mode") not in TOOL_CREDIT_MODES):
        raise ValueError("policy warm-start requires a full FP32 v6 previous-value credit source contract")
    source_world = source.get("learner_world_size")
    if not isinstance(source_world, int) or source_world <= 0:
        raise ValueError("warm-start source lacks its learner world size")
    checkpoint = resolve_resume_checkpoint(
        mode="resume_path", output_dir=output_dir, resume_from_path=str(path),
        train_scope="full", learner_world_size=source_world, allow_weights_only=False)
    for prefix in ("optim", "extra_state"):
        if any((checkpoint.trainer_state / f"{prefix}_world_size_{source_world}_rank_{rank}.pt").stat().st_size == 0
               for rank in range(source_world)):
            raise ValueError("warm-start source trainer state contains an empty file")
    report_bytes = (path / "report.json").read_bytes()
    report = json.loads(report_bytes)
    if (checkpoint.step <= 0 or report.get("publication_kind") != "checkpoint"
            or report.get("qwen_export_dtype") != "preserve"):
        raise ValueError("policy warm-start requires a complete FP32 checkpoint publication")
    tracker = path.parent / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or int(tracker.read_text().strip()) < checkpoint.step:
        raise ValueError("policy warm-start source has not been committed by its checkpoint tracker")
    critic = load_checkpoint_tool_critic(path)
    credit = source["tool_credit"]
    if (critic.max_tool_calls != credit.get("max_tool_calls")
            or critic.learning_rate != credit.get("critic_learning_rate")
            or critic.l2 != credit.get("critic_l2")):
        raise ValueError("warm-start critic state differs from its source contract")
    qwen_index_sha = _validate_fp32_policy_payload(checkpoint)
    source_actor = {
        "checkpoint": str(path), "step": checkpoint.step,
        "learner_world_size": source_world,
        "contract_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "qwen_index_sha256": qwen_index_sha,
    }
    rebased = rebase_critic_for_new_run(critic, source_actor=source_actor,
                                      target_run=str(output_dir.resolve()))
    lineage = {
        "kind": "joint_policy_and_critic_warm_start",
        "at_experiment_start": True, "source_actor": source_actor,
        "target_run": str(output_dir.resolve()), "start_step": 0,
        "actor_optimizer": "new", "actor_scheduler": "new", "actor_rng": "new",
        "exact_trainer_state": False,
        "critic_optimizer": "inherited_with_unchanged_real_adam_counter",
        "source_critic_sha256": critic.state_dict()["sha256"],
        "source_critic_fitted_through_step": critic.fitted_through_step,
        "critic_optimizer_step": critic.optimizer_step,
        "rebased_critic_sha256": rebased.state_dict()["sha256"],
        "rebased_critic_fitted_through_step": 0,
        "critic_training_lineage_sha256": rebased.training_sha256,
    }
    return PolicyWarmStart(path, checkpoint.step, checkpoint.head, checkpoint.query,
                           checkpoint.qwen, source, rebased, lineage)


def validate_policy_warm_start_contract(warm_start: PolicyWarmStart, requested: dict) -> None:
    """Allow new topology/data/batch size, preserve the learned objective."""
    keys = (
        "schema", "harness_version", "train_scope", "v6_training", "tool_credit", "reward", "geometry_contract",
        "max_tool_calls", "max_completion_tokens", "tool_budget_semantics", "fast_step_limit",
        "sde_steps", "sde_noise", "router_replay", "capture_full_routes", "rollout_semantics",
        "group_case_sharing", "group_episode_sharing", "slow_system_calls", "fast_system_execution",
        "terminal_success_owner", "old_logprob_mode", "optimizer_steps_per_logical_batch",
        "learner_parameter_dtype", "action_head_learning_rate", "vlm_learning_rate", "optimizer_weight_decay",
    )
    changed = {key: {"source": warm_start.source_contract.get(key), "requested": requested.get(key)}
               for key in keys if warm_start.source_contract.get(key) != requested.get(key)}
    # A new actor optimizer/RNG may use a different implementation of the same
    # text/FM loss. The trainer records both forward modes in warm-start lineage;
    # exact resume still checks this flag strictly below.
    if changed:
        raise ValueError(f"policy warm-start objective contract mismatch: {json.dumps(changed, sort_keys=True)}")


def validate_resume_contract(checkpoint: ResumeCheckpoint,
                             requested: dict, *,
                             allow_tool_credit_fork: bool = False,
                             allow_rule_credit_transition: bool = False,
                             allow_rollout_topology_change: bool = False) -> dict | None:
    """Fail closed if resume would change the rollout/update semantics."""
    source_path = checkpoint.path.parent / "contract.json"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source = json.loads(source_path.read_text())
    requested = dict(requested)
    # Contracts written before the 2026-09-09 performance work used the
    # two-forward anchor, gradient checkpointing, and one GPU per rollout
    # actor.  Make those implicit legacy values explicit so unchanged resumes
    # remain valid while optimized runs still fail closed on topology changes.
    source.setdefault("old_logprob_mode", "recompute")
    source.setdefault("gradient_checkpointing", True)
    source.setdefault("learner_parameter_dtype", "bf16")
    source.setdefault("option_condition_reuse", False)
    source.setdefault("piold_option_reuse", False)
    source.setdefault("frozen_weight_residency", False)
    source.setdefault("defer_query_capture", False)
    source.setdefault("defer_gradient_sync", False)
    source.setdefault("fsdp_size", -1)
    source.setdefault("fsdp_forward_prefetch", False)
    source.setdefault("prepare_streamed_wave", False)
    source.setdefault("vllm_tensor_parallel_size", 2)
    source.setdefault("vllm_replicas", 1)
    source.setdefault("colocated_vllm_replicas", 0)
    source.setdefault("colocated_vllm_mode", "disabled")
    # These knobs were not persisted by older contracts. Preserve compatibility
    # once, while every newly written contract checks them exactly.
    source.setdefault("vllm_max_num_seqs", requested.get("vllm_max_num_seqs", 0))
    source.setdefault("vllm_max_num_batched_tokens", requested.get(
        "vllm_max_num_batched_tokens", 0))
    source.setdefault("full_weight_residency_gib", 0.0)
    source.setdefault("optimizer_cpu_during_backward", False)
    source.setdefault("optimizer_weight_decay", 0.01)
    source.setdefault("async_rollout", False)
    source.setdefault("rollout_gpu_fraction", 1.0)
    source.setdefault(
        "rollout_reserved_gpus", source.get("rollout_actor_gpus"))
    source.setdefault("episode_worker_device", "rollout_actor_gpu")
    source.setdefault("fast_policy_workers", 0)
    source.setdefault("fast_policy_device", "cuda")
    source.setdefault("text_before_fm", False)
    source.setdefault("fast_policy_max_batch_size", 8)
    source.setdefault("fast_policy_batch_wait_ms", 12.0)
    source.setdefault("fast_policy_weight_sync", "per_actor_reload")
    requested.setdefault("old_logprob_mode", "recompute")
    requested.setdefault("gradient_checkpointing", True)
    requested.setdefault("learner_parameter_dtype", "bf16")
    requested.setdefault("option_condition_reuse", False)
    requested.setdefault("piold_option_reuse", False)
    requested.setdefault("frozen_weight_residency", False)
    requested.setdefault("defer_query_capture", False)
    requested.setdefault("defer_gradient_sync", False)
    requested.setdefault("fsdp_size", -1)
    requested.setdefault("fsdp_forward_prefetch", False)
    requested.setdefault("prepare_streamed_wave", False)
    requested.setdefault("vllm_tensor_parallel_size", 2)
    requested.setdefault("vllm_replicas", 1)
    requested.setdefault("colocated_vllm_replicas", 0)
    requested.setdefault("colocated_vllm_mode", "disabled")
    requested.setdefault("vllm_max_num_seqs", 0)
    requested.setdefault("vllm_max_num_batched_tokens", 0)
    requested.setdefault("full_weight_residency_gib", 0.0)
    requested.setdefault("optimizer_cpu_during_backward", False)
    requested.setdefault("optimizer_weight_decay", 0.01)
    requested.setdefault("async_rollout", False)
    requested.setdefault("rollout_gpu_fraction", 1.0)
    requested.setdefault(
        "rollout_reserved_gpus", requested.get("rollout_actor_gpus"))
    requested.setdefault("episode_worker_device", "rollout_actor_gpu")
    requested.setdefault("fast_policy_workers", 0)
    requested.setdefault("fast_policy_device", "cuda")
    requested.setdefault("text_before_fm", False)
    requested.setdefault("fast_policy_max_batch_size", 8)
    requested.setdefault("fast_policy_batch_wait_ms", 12.0)
    requested.setdefault("fast_policy_weight_sync", "per_actor_reload")
    source.setdefault("tool_budget_semantics", "legacy_multi_call_turns")
    requested.setdefault("tool_budget_semantics", "legacy_multi_call_turns")
    source.setdefault("fast_step_limit", source.get("fm_chunks_per_episode", 4))
    requested.setdefault("fast_step_limit", requested.get("fm_chunks_per_episode", 4))
    source.setdefault("max_completion_tokens", 0)
    requested.setdefault("max_completion_tokens", 0)
    source.setdefault("joint_text_fm_forward", False)
    requested.setdefault("joint_text_fm_forward", False)
    # Audit the exact-ID input fix separately, preserving old actor/Adam state.
    # multimodal_replay_revision is intentionally not a strict-equality key.
    keys = (
        "text_backend", "megatron_parallelism", "group_partial_rollout",
        "case_sampling", "case_sampling_seed", "case_source_groups",
        "dataset_split", "case_order_sha256", "harness_version", "v6_training", "online_evaluation",
        "context_protocol", "text_training_layout",
        "tool_credit", "policy_warm_start", "joint_text_fm_forward", "geometry_contract",
        "max_turns", "max_tool_calls", "tool_budget_semantics",
        "fast_step_limit", "max_completion_tokens", "sde_steps", "sde_noise",
        "schema", "train_scope", "router_replay", "capture_full_routes",
        "smoke_case_count", "num_groups_per_step", "group_size",
        "episodes_per_step", "learner_world_size", "learner_gpus_layout",
        "rollout_semantics", "group_case_sharing", "group_episode_sharing",
        "slow_system_calls", "fast_system_execution", "terminal_success_owner",
        "fm_chunks_per_episode",
        "learner_microbatch_per_rank", "optimizer_steps_per_logical_batch",
        "learner_dispatch", "logical_batch_transport", "reward",
        "old_logprob_mode", "gradient_checkpointing", "learner_parameter_dtype",
        "option_condition_reuse", "piold_option_reuse", "frozen_weight_residency",
        "defer_query_capture", "defer_gradient_sync", "async_rollout",
        "fsdp_size", "fsdp_forward_prefetch", "prepare_streamed_wave",
        "vllm_tensor_parallel_size", "vllm_replicas",
        "colocated_vllm_replicas", "colocated_vllm_mode",
        "vllm_max_num_seqs", "vllm_max_num_batched_tokens",
        "full_weight_residency_gib",
        "optimizer_cpu_during_backward",
        "action_head_learning_rate", "vlm_learning_rate", "optimizer_weight_decay",
        "rollout_gpu_fraction", "rollout_reserved_gpus",
        "episode_worker_device", "fast_policy_workers", "fast_policy_device", "text_before_fm",
        "fast_policy_max_batch_size", "fast_policy_batch_wait_ms",
        "fast_policy_weight_sync",
    )
    changed = {
        key: {"checkpoint": source.get(key), "requested": requested.get(key)}
        for key in keys if source.get(key) != requested.get(key)
    }
    topology_audit = None
    if allow_rollout_topology_change:
        # Rebuild serving from fully restored learners, without restoring
        # partial environments/queues. Only redistribute the same GPU pool.
        topology_keys = ('vllm_tensor_parallel_size', 'colocated_vllm_replicas')
        stable = ('learner_world_size', 'learner_gpus_layout', 'megatron_parallelism',
                  'rollout_actor_count', 'vllm_query_tail', 'router_replay',
                  'group_partial_rollout', 'vllm_gpu_memory_utilization')
        def valid_layout(value):
            tp, replicas = (value.get(key) for key in topology_keys)
            layout = value.get('learner_gpus_layout')
            return (type(tp) is int and tp in (2, 4, 8) and type(replicas) is int
                    and replicas > 0 and isinstance(layout, list) and layout
                    and all(n == 16 for n in layout)
                    and replicas * tp == value.get('learner_world_size') == sum(layout)
                    and value.get('vllm_replicas') == 0)
        if (not checkpoint.exact or allow_tool_credit_fork or allow_rule_credit_transition
                or source.get('harness_version') != 'v8' or requested.get('harness_version') != 'v8'
                or source.get('text_backend') != 'megatron' or requested.get('text_backend') != 'megatron'
                or not source.get('group_partial_rollout') or not source.get('vllm_query_tail')
                or source.get('router_replay') != 'R3'
                or any(source.get(key) != requested.get(key) for key in stable)
                or not all(valid_layout(value) for value in (source, requested))):
            raise ValueError('rollout topology change requires exact v8 Megatron resume with the same learner and node-local GPU pool')
        topology_audit = dict(source_checkpoint=str(checkpoint.path),
            source_contract_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            changes={key: changed[key] for key in topology_keys if key in changed},
            restored_model_optimizer_scheduler_rng=True,
            live_rollout_state_restored=False, cross_topology_bitwise_equivalence=False)
        for key in topology_keys:
            changed.pop(key, None)
    if 'geometry_contract' in changed:
        old_geometry = dict(source.get('geometry_contract') or {})
        new_geometry = dict(requested.get('geometry_contract') or {})
        # Moving unchanged scoring configuration into a local Git checkout is
        # provenance relocation. Keep every scoring parameter and content hash
        # strict; only the file location can differ when its digest is bound.
        digest = old_geometry.get('contract_sha256')
        if isinstance(digest, str) and len(digest) == 64 and digest == new_geometry.get('contract_sha256'):
            old_geometry.pop('contract_path', None)
            new_geometry.pop('contract_path', None)
            if old_geometry == new_geometry:
                changed.pop('geometry_contract')
    if allow_rule_credit_transition:
        from action_head_rl.rule_credit import contract as rule_credit_contract
        if allow_tool_credit_fork:
            raise ValueError("rule-credit transition cannot mix with a learned tool-credit fork")
        selective_backward = bool(source.get("joint_text_fm_forward", False))
        expected_source = rule_credit_contract(
            selective_backward=selective_backward, version="v3_first_arrival")
        allowed_targets = [rule_credit_contract(
            selective_backward=selective_backward, version=version)
            for version in ("v3_reject_guard", "v3_reject_guard_stop")]
        if (not checkpoint.exact or source.get("harness_version") != "v6"
                or source.get("train_scope") != "full"
                or not (source.get("v6_training") or {}).get("text_policy")
                or source.get("tool_credit") != expected_source
                or requested.get("tool_credit") not in allowed_targets):
            raise ValueError(
                "rule-credit transition requires a complete v3_first_arrival checkpoint "
                "and a canonical v3_reject_guard or v3_reject_guard_stop target")
        # This is an explicit objective experiment with all actor/Adam/RNG
        # state restored. Only the fully validated rule-credit spec may change;
        # the rest of the data, rollout, update and topology contract remains.
        changed.pop("tool_credit", None)
    if allow_tool_credit_fork:
        from action_head_rl.tool_credit import (
            TOOL_CREDIT_MODES, credit_estimator_spec, credit_reward_normalization,
        )
        # This is an explicit algorithm experiment with preserved actor/Adam
        # state, not a claim that the old training objective was resumed.
        # Every unrelated data, reward, precision and topology check remains.
        old_reward = source.get("reward") or {}
        new_reward = requested.get("reward") or {}
        tool_credit = requested.get("tool_credit") or {}
        if (not checkpoint.exact or source.get("tool_credit") is not None
                or source.get("harness_version") != "v6"
                or source.get("train_scope") != "full"
                or not (source.get("v6_training") or {}).get("text_policy")
                or tool_credit.get("mode") not in TOOL_CREDIT_MODES
                or old_reward.get("normalization") != "same_case_trajectory_outcome_grpo"
                or old_reward.get("terminal_success_reward") != 1.0
                or old_reward.get("direction_match_reward") != 0.0
                or old_reward.get("collision_penalty") != 0.0
                or old_reward.get("discount_gamma") != 1.0):
            raise ValueError("tool-credit fork requires a complete pure-terminal v6 episode-GRPO checkpoint")
        if any(tool_credit.get(key) != value for key, value in
               credit_estimator_spec(tool_credit["mode"]).items()):
            raise ValueError("tool-credit fork has an inconsistent estimator contract")
        replacements = {
            "normalization": credit_reward_normalization(tool_credit["mode"]),
            "advantage_broadcast": "per_call_shared_by_its_text_and_fm_chunks",
        }
        expected_reward = {**old_reward, **replacements}
        if new_reward != expected_reward:
            raise ValueError("tool-credit fork may only change the declared credit assignment, not rewards")
        changed.pop("tool_credit", None)
        changed.pop("reward", None)
    if changed:
        raise ValueError(
            "resume contract mismatch: "
            + json.dumps(changed, ensure_ascii=False, sort_keys=True))
    return topology_audit


def validate_optimizer_learning_rates(optimizer, scheduler, *,
                                     action_head_lr: float, vlm_lr: float,
                                     train_scope: str) -> dict:
    """Check the loaded constant-LR state rather than only launch arguments.

    Optimizer.load_state_dict restores its saved learning rates. Reject a
    stale or inconsistent state before rollout starts; do not silently rewrite
    the optimizer or scheduler and call that an exact resume.
    """
    expected = {"action_head_query": float(action_head_lr)}
    if train_scope == "full":
        expected["slow_vlm"] = float(vlm_lr)
    if any(not math.isfinite(value) or value <= 0 for value in expected.values()):
        raise ValueError("learning rates must be finite and positive")
    names = [group.get("name") for group in optimizer.param_groups]
    if len(names) != len(expected) or set(names) != set(expected):
        raise ValueError(f"optimizer learning-rate group mismatch: {names}, expected {list(expected)}")
    rows = []
    for group in optimizer.param_groups:
        name = group["name"]
        row = {"name": name, "expected_lr": expected[name], "lr": float(group["lr"])}
        if row["lr"] != expected[name]:
            raise ValueError(f"optimizer learning-rate mismatch: {row}")
        if "initial_lr" in group:
            row["initial_lr"] = float(group["initial_lr"])
            if row["initial_lr"] != expected[name]:
                raise ValueError(f"optimizer initial learning-rate mismatch: {row}")
        rows.append(row)
    wanted = [expected[name] for name in names]
    state = scheduler.state_dict()
    scheduler_rates = {}
    for key in ("base_lrs", "_last_lr"):
        if key in state:
            actual = [float(value) for value in state[key]]
            scheduler_rates[key] = actual
            if actual != wanted:
                raise ValueError(f"scheduler learning-rate mismatch for {key}: {actual}, expected {wanted}")
    return {"passed": True, "groups": rows, "scheduler": scheduler_rates}


def write_latest_checkpoint(output_dir: Path, step: int) -> None:
    """Atomically publish the latest fully saved training checkpoint."""
    final = output_dir / "latest_checkpointed_iteration.txt"
    temporary = final.with_suffix(".txt.tmp")
    temporary.write_text(str(int(step)))
    os.replace(temporary, final)


def truncate_jsonl_after_step(path: Path, step: int) -> None:
    """Drop uncommitted rows after the checkpoint selected for resume."""
    if not path.exists():
        return
    kept = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if int(row.get("step", -1)) <= step:
            kept.append(line)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(value + "\n" for value in kept))
    os.replace(temporary, path)


def prune_committed_checkpoints(output_dir: Path, *, keep_last: int,
                                train_scope: str, learner_world_size: int) -> list[str]:
    """Retain complete exports plus VERL state, only after tracker commit.

    VERL's worker retention covers trainer_state alone. The driver also owns
    the HF export, so validate complete checkpoints with the shared resolver
    before removing old directories from this run. Preserve incomplete and
    uncommitted directories for recovery/diagnosis, and never follow symlinks.
    """
    if keep_last < 0:
        raise ValueError("keep_last must be nonnegative")
    if keep_last == 0:
        return []
    root = Path(output_dir).resolve()
    latest = resolve_resume_checkpoint(
        mode="auto", output_dir=root, resume_from_path="",
        train_scope=train_scope, learner_world_size=learner_world_size,
        allow_weights_only=False)
    if latest is None:
        return []
    if latest.path.is_symlink() or latest.path.parent != root:
        raise ValueError("checkpoint retention requires a local committed checkpoint")
    complete = []
    for path in root.glob("global_step_*"):
        if path.is_symlink() or not path.is_dir() or not _STEP_RE.fullmatch(path.name):
            continue
        if checkpoint_step(path) > latest.step:
            continue
        try:
            checkpoint = resolve_resume_checkpoint(
                mode="resume_path", output_dir=root, resume_from_path=str(path),
                train_scope=train_scope, learner_world_size=learner_world_size,
                allow_weights_only=False)
        except (ValueError, OSError):
            continue
        complete.append(checkpoint)
    complete.sort(key=lambda checkpoint: checkpoint.step)
    removed = []
    for checkpoint in complete[:-keep_last]:
        if checkpoint.path.parent != root or checkpoint.path == latest.path:
            raise ValueError("checkpoint retention escaped the committed run")
        shutil.rmtree(checkpoint.path)
        removed.append(str(checkpoint.path))
    return removed


def training_state_digest(value) -> str:
    """Hash nested local tensors without serializing device IDs or object aliases."""
    import numpy as np
    import torch
    from torch.distributed.tensor import DTensor

    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, DTensor):
            item = item.to_local()
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            digest.update(str((str(item.dtype), tuple(item.shape))).encode())
            digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((str(item.dtype), item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda x: (type(x).__name__, repr(x))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(str((type(item).__name__, len(item))).encode())
            for child in item:
                visit(child)
        else:
            digest.update(str((type(item).__name__, repr(item))).encode())
    visit(value)
    return digest.hexdigest()


def wait_for_companion_retention(output_dir, *, next_step, keep_last, sleep=None):
    """Backpressure before export so the last two never race an active reader.

    Only opt-in paired runs wait. Reports are read with the existing ESTALE
    helper; absent/invalid results never authorize deleting evaluation inputs.
    """
    evaluation = os.environ.get('ACTION_HEAD_RETENTION_EVAL_ROOT')
    if not evaluation or keep_last <= 0:
        return
    import time
    import importlib.util
    source = Path(__file__).resolve().parents[2]/'scripts/train/align_three_source_observations.py'
    spec = importlib.util.spec_from_file_location('_rloo_retention_io', source)
    io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(io)
    exists = io.exists
    read_text = lambda path: io.read_bytes(path).decode('utf-8')
    pause = sleep or time.sleep
    root = Path(output_dir)
    tracker = root/'latest_checkpointed_iteration.txt'
    latest = int(read_text(tracker).strip()) if exists(tracker) else 0
    candidates = io.bounded(lambda: list(root.glob('global_step_*')), 'retention_list:'+str(root))
    paths = sorted((p for p in candidates
                    if not p.is_symlink() and p.is_dir() and _STEP_RE.fullmatch(p.name)
                    and checkpoint_step(p) <= latest), key=checkpoint_step)
    retiring = paths[:max(0, len(paths) + 1 - keep_last)]
    for checkpoint in retiring:
        step = checkpoint_step(checkpoint)
        if step % 4:
            continue
        report = Path(evaluation)/'repeat-summary'/f'global_step_{step:06d}.json'
        while True:
            if exists(report):
                result = json.loads(read_text(report))
                if result.get('evaluation_valid') is True and result.get('step') == step:
                    break
            print(json.dumps(dict(event='checkpoint_retention_waiting_for_companion',
                checkpoint=str(checkpoint), evaluation_report=str(report), next_step=next_step,
                pid=os.getpid(), time=time.time(), business_progress=False)), flush=True)
            pause(20)
