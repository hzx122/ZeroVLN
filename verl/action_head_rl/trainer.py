#!/usr/bin/env python3
"""Ray-native closed-loop RL driver for the ZeroVLN brain and action expert."""
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from contextlib import closing, nullcontext
from pathlib import Path

import hydra
import ray
import torch

from action_head_rl.config import TrainScope, validate_training_contract
from action_head_rl.local_policy_transport import (
    LocalPolicyTransport, cleanup_local_transport, publication_root,
)
from action_head_rl.checkpoint import (
    resolve_resume_checkpoint, validate_resume_contract,
    resolve_policy_warm_start, validate_policy_warm_start_contract,
    truncate_jsonl_after_step, write_latest_checkpoint,
    prune_committed_checkpoints,
)
from action_head_rl.rollout import (
    ActionHeadRolloutWorker, FastPolicyWorker, RewardConfig,
)
from action_head_rl.scheduling import iter_grouped_rollouts
from action_head_rl.rollout_prefetch import BackgroundGroupedRollouts
from action_head_rl.streaming import (
    BoundedLearnerStream, stage_aligned_wave, validate_streamed_optimizer_result,
)
from action_head_rl.runtime_progress import resolve_ray_results
from action_head_rl.sde import grpo_advantages
from action_head_rl.sde import OLD_LOGPROB_MODES
from action_head_rl.text_policy import is_text_record
from action_head.protocol import protocol_module
from action_head_rl.tool_credit import (
    TOOL_CREDIT_MODES, credit_estimator_spec, credit_reward_normalization,
)
from action_head_rl.vllm_service import VllmService
from action_head_rl.worker import (
    ActionHeadGRPOWorker, OPTION_SHARED_KEYS, FM_REPLAY_INPUT_REVISION,
    shard_option_record_groups, shard_record_batches, _condition_family,
)
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.engine_workers import TrainingWorkerConfig
from sft.qwen35_padding_mask import PADDING_MASK_REVISION


class _DirectVllmReplica:
    """Minimal adapter that lets the upstream VERL manager own our services."""

    def __init__(self, workers):
        self.workers = list(workers)

    async def abort_all_requests(self):
        return None

    async def sleep(self):
        # Follow VERL's free_cache_engine lifecycle.  Building the temporary
        # NCCL group while a 35B vLLM model and KV cache are both resident can
        # exhaust the rollout GPU before the first bucket is transferred.
        ray.get([worker.sleep.remote(level=1) for worker in self.workers])

    async def wake_up(self):
        ray.get([worker.resume_after_direct_weight_sync.remote()
                 for worker in self.workers])

    async def resume_generation(self):
        return None


@ray.remote(num_cpus=0.1)
def _resolve_rank_rollout_wave(records, preserve_staged_reuse: bool = False):
    """Resolve nested rollout refs in Plasma before entering FSDP.

    Ray only resolves top-level ObjectRefs automatically.  Resolving nested
    refs inside learner actors lets ready ranks enter collectives while another
    rank is still fetching its payload.  This CPU task materializes one rank's
    wave in the object store; the driver waits only for readiness and never
    receives the tensor payload.
    """
    cache = {}
    resolved = []
    for source in records:
        record = dict(source)
        reuse_staged = bool(record.get("_reuse_staged_option_payload", False))
        payload_ref = record.pop("_rollout_payload_ref", None)
        if payload_ref is not None:
            key = payload_ref.hex()
            packed = cache.get(key)
            if packed is None:
                packed = ray.get(payload_ref)
                cache[key] = packed
            chunk_index = int(record.pop("_rollout_payload_index"))
            if not (preserve_staged_reuse and reuse_staged):
                record.update(packed["shared"])
            record.update(packed["chunks"][chunk_index])
        # Return complete records.  Payload de-duplication inside learner
        # staging remains an optimization, not a correctness dependency.
        if not preserve_staged_reuse:
            record.pop("_reuse_staged_option_payload", None)
        resolved.append(record)
    return resolved


def _resolve_rank_waves(waves, world: int):
    """Resolve all ranks before dispatch, without fetching tensors into driver."""
    if any("_rollout_payload_ref" in row for wave in waves for row in wave):
        resolved_refs = [_resolve_rank_rollout_wave.remote(wave) for wave in waves]
        ready, _ = ray.wait(
            resolved_refs, num_returns=world, timeout=300.0,
            fetch_local=False)
        if len(ready) != world:
            raise TimeoutError(
                f"rollout payload barrier resolved {len(ready)}/{world} ranks")
        return resolved_refs
    else:
        return waves


def _stage_rank_waves(learner, waves, world: int, *, text_placements=None):
    """Stage a synchronized rank wave without routing tensors via driver."""
    if text_placements is not None and not any(
            "_rollout_payload_ref" in row for wave in waves for row in wave):
        from action_head_rl.megatron_joint import text_transport_dp_waves
        from verl.single_controller.base.decorator import dispatch_nd_compute
        logical = text_transport_dp_waves(waves, text_placements)
        if logical is not None:
            mapping, dp_waves = logical
            args, kwargs = dispatch_nd_compute(mapping, len(dp_waves), learner, dp_waves)
            # Native dispatch shares ObjectRefs across TP/PP/CP. Keep the physical
            # DP_COMPUTE collector: staging must acknowledge ALL ranks, not only
            # the model-parallel source ranks used by native result collection.
            return learner.stage_record_batches(*args, **kwargs)
    return learner.stage_record_batches(_resolve_rank_waves(waves, world))


def _zero_advantage_training_records(records: list[dict], *,
                                    compact_zero_waves: bool = False) -> tuple[list[dict], float, dict]:
    """Drop exact zero-gradient records before global FSDP sharding.

    Episode normalization is measured on the original wave, including zero
    advantage episodes and STOP-only text episodes. The remaining per-record
    weights are untouched. Keep the original wave when filtering would remove
    every record or an entire forward branch: an absent gradient is different
    from a zero gradient for Adam's accumulated moments and weight decay.

    The experimental compaction path retains one REAL zero-loss graph for
    every original forward family (FM, text, single-token text). It does not
    skip the accumulation or synthesize gradients. Full-model gradient/RNG/
    Adam parity must be validated before enabling this in a production recipe.
    """
    def loss_mass(rows):
        if os.environ.get("ACTION_HEAD_FINAL_CONTRACT") == "three-source-boundary-stop-v1":
            from action_head_rl.joint_policy import episode_text_mass
            return episode_text_mass(rows)
        text_rows = [row for row in rows if is_text_record(row)]
        return sum(float(row.get("sample_valid", 1.0))
                   * float(row.get("loss_weight", 1.0))
                   for row in (text_rows or rows))

    original_mass = loss_mass(records)
    if not math.isfinite(original_mass) or original_mass <= 0:
        raise ValueError("zero-advantage filtering requires positive finite original loss mass")
    active = []
    for row in records:
        advantage = float(row["advantage"])
        if not math.isfinite(advantage):
            raise ValueError("zero-advantage filtering requires finite advantages")
        from action_head_rl.joint_policy import final_record_has_auxiliary_loss
        auxiliary_active = final_record_has_auxiliary_loss(row)
        if advantage != 0.0 or auxiliary_active:
            active.append(row)
    all_zero = not active
    missing_branch = bool(active) and (
        {is_text_record(row) for row in active}
        != {is_text_record(row) for row in records})
    retained = records if all_zero or missing_branch else active
    compacted = bool(compact_zero_waves and all_zero)
    representatives = {}
    if compacted:
        for row in records:
            representatives.setdefault(_condition_family(row), row)
        retained = list(representatives.values())
    return retained, original_mass, {
        "enabled": True,
        "original_records": len(records),
        "nonzero_advantage_records": len(active),
        "zero_advantage_records": len(records) - len(active),
        "skipped_records": len(records) - len(retained),
        "retained_records": len(retained),
        "original_loss_mass": original_mass,
        "retained_loss_mass": loss_mass(retained),
        "logical_denominator": original_mass,
        "all_zero_fallback": all_zero and not compacted,
        "missing_branch_fallback": missing_branch,
        **({"zero_wave_compacted": compacted,
            "representative_conditions": [dict(family=family,
                condition_id=row.get("condition_id"), token_count=row.get("token_count"))
                for family, row in representatives.items()]}
           if compact_zero_waves else {}),
    }


def _tool_credit_contract(max_tool_calls: int, fit_updates: int = 20, *,
                          mode: str = "previous_value_gae") -> dict:
    from action_head_rl.tool_credit import SCHEMA, FEATURE_SCHEMA, FEATURE_NAMES
    return {
        "schema": SCHEMA,
        "feature_schema": FEATURE_SCHEMA, "feature_names": list(FEATURE_NAMES),
        "gamma": 1.0, **credit_estimator_spec(mode),
        "critic_fit_rule": "previous_frozen_mc_episode_equal",
        "critic_fit_updates": fit_updates, "critic_learning_rate": 0.03,
        "critic_l2": 0.001, "max_tool_calls": max_tool_calls,
        "terminal_bootstrap": False,
    }


def _load_tool_credit_critic(args, resume):
    """Load only the explicit fork seed or this exact checkpoint's critic."""
    if args.tool_credit_mode in ("episode_grpo", "rule_call_loo"):
        if args.allow_tool_credit_fork or args.tool_credit_initial_state:
            raise ValueError("tool-credit seed/fork requires a previous-value credit mode")
        return None
    credit_estimator_spec(args.tool_credit_mode)
    from action_head_rl.checkpoint import load_checkpoint_tool_critic
    from action_head_rl.tool_credit import LogisticStateCritic
    if resume is None or not resume.exact:
        raise ValueError("tool credit requires a complete actor/optimizer/RNG checkpoint")
    if args.allow_tool_credit_fork:
        if resume.step != 16 or not args.tool_credit_initial_state:
            raise ValueError("tool-credit fork requires checkpoint step 16 and an explicit C16 seed")
        critic = LogisticStateCritic.load(args.tool_credit_initial_state)
    else:
        if args.tool_credit_initial_state:
            raise ValueError("tool-credit initial state is only valid for an explicit algorithm fork")
        critic = load_checkpoint_tool_critic(resume.path)
    if (critic.fitted_through_step != resume.step
            or critic.max_tool_calls != args.max_tool_calls
            or critic.learning_rate != 0.03 or critic.l2 != 0.001):
        raise ValueError("critic step/tool budget/optimizer contract differs from actor resume")
    return critic


def _tool_credit_for_rollout(rollout, critic, *, step: int, max_tool_calls: int,
                           mode: str = "previous_value_gae", loo_baseline=None):
    from action_head_rl.tool_credit import build_call_ledger, assign_call_credit
    ledger = build_call_ledger(rollout["result"], rollout["chunks"], step=step,
                               policy_version=step - 1, max_tool_calls=max_tool_calls)
    if float(rollout["reward"]) != float(ledger.native_success):
        raise ValueError("tool credit must preserve the native terminal-only rollout reward")
    credit = assign_call_credit(ledger, critic, gamma=1.0, mode=mode,
                                loo_baseline=loo_baseline)
    decisions = rollout.get("decisions") or []
    decision_ids = [int(row["decision_index"]) for row in decisions]
    if (len(decision_ids) != len(set(decision_ids))
            or set(decision_ids) != set(credit["text_advantages"])):
        raise ValueError("tool-credit ledger and text decisions do not match")
    if {str(row["option_id"]) for row in rollout["chunks"]} != set(credit["option_advantages"]):
        raise ValueError("tool-credit ledger and FM options do not match")
    return ledger, credit


def _tool_credit_group_baselines(rollouts: list[dict], *, mode: str):
    """Compute an outcome baseline from other independent same-case episodes.

    Excluding this episode is essential: its terminal reward may not enter its
    own baseline. Seeds and task identity come from completed rollout records,
    never from guessed group positions or a subset of successful trajectories.
    """
    credit_estimator_spec(mode)
    if mode != "previous_value_mc_loo":
        return [None] * len(rollouts)
    if len(rollouts) < 2:
        raise ValueError("MC/LOO credit requires at least two same-case rollouts")
    identities, seeds, rewards = set(), set(), []
    for row in rollouts:
        result = row.get("result") or {}
        identity = (result.get("scene"), result.get("episode_id"))
        seed = row.get("seed")
        if (any(value in (None, "") for value in identity)
                or not isinstance(seed, int) or isinstance(seed, bool) or seed in seeds
                or result.get("evaluation_valid") is not True or result.get("infra_error")
                or not isinstance(result.get("success"), bool)
                or result.get("success_contract") not in ("sage-official-expanded-object-bbox-v1", "sage-designated-target-navmesh-v2")):
            raise ValueError("MC/LOO requires valid independent native same-case rollouts")
        reward = float(row["reward"])
        if not math.isfinite(reward) or reward != float(result["success"]):
            raise ValueError("MC/LOO baseline requires native binary terminal outcomes")
        identities.add(identity)
        seeds.add(seed)
        rewards.append(reward)
    if len({row["result"]["success_contract"] for row in rollouts}) != 1:
        raise ValueError("MC/LOO baseline cannot mix geometry contracts")
    if len(identities) != 1:
        raise ValueError("MC/LOO baseline cannot mix different cases")
    total = sum(rewards)
    return [(total - reward) / (len(rewards) - 1) for reward in rewards]


def _apply_tool_credit_records(records: list[dict], credit: dict) -> None:
    """Change advantages only; retain episode loss mass and policy tensors."""
    for record in records:
        record["advantage"] = (
            credit["text_advantages"][int(record["decision_index"])]
            if is_text_record(record) else
            credit["option_advantages"][str(record["option_id"])])


def _advance_tool_critic(critic, ledgers, rank_metrics, *, step: int, world: int, fit_updates: int):
    # VERL may skip Adam on a nonfinite gradient without raising. Do not
    # commit a newer critic unless all ranks completed a finite actor update.
    if (len(rank_metrics) != world or any(
            not isinstance(row, dict) or row.get("optimizer_steps") != 1
            or not math.isfinite(float(row.get("grad_norm", float("nan"))))
            for row in rank_metrics)):
        raise RuntimeError("critic update requires a finite successful actor update on every rank")
    if (critic.fitted_through_step != step - 1 or not ledgers
            or any(row.step != step or row.policy_version != step - 1 for row in ledgers)):
        raise ValueError("critic fit must follow this actor batch and its previous frozen critic")
    return critic.fit_mc(ledgers, updates=fit_updates)


def resume_audit_compare_label(output: Path, step: int, world_size: int,
                               resumed: bool) -> tuple[str, dict]:
    """Select strict comparison only when every saved audit sidecar exists."""
    label = f"saved_{step:06d}"
    if not resumed:
        return "", {"mode": "initial_load", "compare_label": ""}
    root = Path(output) / "resume_audit"
    expected = {f"{label}_rank{rank}.json" for rank in range(world_size)}
    present = {path.name for path in root.glob(f"{label}_rank*.json")}
    missing = sorted(expected - present)
    if not missing:
        return label, {
            "mode": "strict_saved_vs_loaded",
            "compare_label": label,
            "baseline_files": world_size,
        }
    return "", {
        "mode": "record_loaded_without_legacy_baseline",
        "compare_label": "",
        "baseline_files": len(expected & present),
        "missing_baseline_files": len(missing),
        "reason": "checkpoint_predates_resume_hash_audit",
    }


def _query_file(path: str) -> Path:
    value = Path(path)
    if value.is_dir():
        value = value / "dualvln_query_embeddings.pt"
    if not value.is_file():
        raise FileNotFoundError(value)
    return value


def load_smoke_cases(limit: int, split: str = "action_head_train_p0_unified") -> list[dict]:
    """Read Sage P0 episodes directly; never convert open-loop SFT records."""
    from sage.datasets import SageSet

    if limit < 0:
        raise ValueError("smoke-cases must be nonnegative; zero selects the full split")
    rows = []
    dataset = SageSet(split)
    for scene in dataset.list_scenes():
        for task in dataset.tasks(scene, 0):
            rows.append({
                "scene": scene,
                "episode_id": task.task_id,
                "instruction": task.instruction,
                "instruction_type": task.instruction_type,
                "target_object_id": task.goal_label,
                "stage": "p0",
                "ref_len": float(task.ref_len),
            })
            if limit and len(rows) >= limit:
                return rows
    if not rows:
        raise RuntimeError(f"{split} contains no episodes")
    return rows


def load_case_manifest(path: str) -> list[dict]:
    """Load an explicit ordered episode list without scene-prefix truncation."""
    rows, identities = [], set()
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if (not isinstance(row, dict) or row.get("scene") in (None, "")
                or row.get("episode_id") in (None, "")):
            raise ValueError(f"case manifest line {line_number} requires scene and episode_id")
        identity = (str(row.get("dataset", "sage")), str(row["scene"]), str(row["episode_id"]))
        if identity in identities:
            raise ValueError(f"case manifest contains duplicate episode {identity}")
        identities.add(identity)
        rows.append(row)
    if not rows:
        raise ValueError("case manifest contains no episodes")
    return rows


def evaluation_episode_specs(cases: list[dict], repeats: int,
                             seed: int) -> list[tuple[int, int, dict, int]]:
    """Keep the same case/repetition seeds across every policy checkpoint."""
    return [(index, repeat, case, seed + index * repeats + repeat)
            for index, case in enumerate(cases) for repeat in range(repeats)]


def run_online_evaluation(workers, cases: list[dict], *, step: int,
                          repeats: int, seed: int, output: Path) -> dict:
    """Evaluate the already synchronized joint policy; never stage its records."""
    started = time.perf_counter()
    successes, returns, lengths = [], [], []
    specifications = evaluation_episode_specs(cases, repeats, seed)
    grouped = iter_grouped_rollouts(
        specifications, workers, group_size=repeats,
        max_in_flight_groups=max(1, math.ceil(len(workers) / repeats)),
        submit=lambda actor, spec: actor.evaluate.remote(spec[2], spec[3], step),
        wait_one=lambda refs: ray.wait(list(refs), num_returns=1)[0][0],
        resolve=ray.get, scheduler="completion")
    with (output / "evaluation_rollouts.jsonl").open("a", encoding="utf-8") as handle:
        for case_index, _, rollouts in grouped:
            # Preserve failed probes as evidence, but never count them as
            # navigation failures or publish an incomplete evaluation score.
            for repeat, row in enumerate(rollouts):
                handle.write(json.dumps({"step": step, "case_index": case_index,
                                         "repeat": repeat, **row}, ensure_ascii=False) + "\n")
            handle.flush()
            require_valid_rollouts(rollouts, step=step)
            if any(int(row.get("policy_version", -1)) != step for row in rollouts):
                raise RuntimeError("evaluation policy-version mismatch")
            case_success = []
            for repeat, row in enumerate(rollouts):
                success = bool(row["reward_components"]["terminal_success"])
                case_success.append(success)
                returns.append(float(row["reward"]))
                lengths.append(len((row.get("result") or {}).get("turns") or []))
            successes.append(case_success)
            print(json.dumps({"evaluation_progress": len(successes),
                              "evaluation_cases": len(cases), "step": step}), flush=True)
    summary = {
        "step": step, "policy_version": step, "cases": len(cases),
        "repeats": repeats, "episodes": len(returns), "seed": seed,
        "sr": sum(sum(row) for row in successes) / len(returns),
        "pass_at_k": sum(any(row) for row in successes) / len(successes),
        "reward_mean": sum(returns) / len(returns),
        "turns_mean": sum(lengths) / len(lengths),
        "seconds": time.perf_counter() - started,
        "case_order_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
        "sampling": "same_stochastic_joint_policy_as_training",
    }
    with (output / "evaluation_metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(json.dumps({"online_evaluation": summary}, ensure_ascii=False), flush=True)
    return summary


def learner_config(args):
    config_dir = os.environ.get(
        "VERL_SFT_CONFIG_DIR",
        "/workspace/verl/verl/trainer/config")
    overrides = [
        f"model.path={args.slow_model}",
        "model.external_lib=action_head_rl.engine",
        "model.use_remove_padding=false",
        f"model.enable_gradient_checkpointing={str(args.gradient_checkpointing).lower()}",
        "+model.override_config.attn_implementation=sdpa",
        "engine=fsdp", "engine.strategy=fsdp2",
        f"engine.model_dtype={getattr(args, 'learner_parameter_dtype', 'bf16')}",
        "++engine.mixed_precision={param_dtype:bf16,reduce_dtype:fp32,buffer_dtype:fp32}",
        f"engine.fsdp_size={getattr(args, 'fsdp_size', -1)}",
        "engine.forward_prefetch="
        f"{str(getattr(args, 'fsdp_forward_prefetch', False)).lower()}",
        "optim=fsdp", f"optim.lr={args.learning_rate}",
        f"optim.weight_decay={getattr(args, 'weight_decay', 0.01)}",
        "optim.lr_scheduler_type=constant",
        "data.micro_batch_size_per_gpu=1", "data.use_dynamic_bsz=false",
        # Model weights use the existing compact action expert/query publication.
        # Reuse VERL checkpointing for optimizer, scheduler and per-rank RNG.
        "checkpoint.save_contents=[optimizer,extra]",
        "checkpoint.load_contents=[optimizer,extra]",
    ]
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        config = hydra.compose(config_name="sft_trainer_engine", overrides=overrides)
    model = omega_conf_to_dataclass(config.model)
    engine = omega_conf_to_dataclass(config.engine)
    optimizer = omega_conf_to_dataclass(config.optim)
    checkpoint = omega_conf_to_dataclass(config.checkpoint)
    profiler = omega_conf_to_dataclass(config.profiler)
    optimizer.total_training_steps = args.opt_steps
    engine.micro_batch_size_per_gpu = 1
    engine.use_dynamic_bsz = False
    return TrainingWorkerConfig(
        model_type="action_head_grpo", model_config=model,
        engine_config=engine, optimizer_config=optimizer,
        checkpoint_config=checkpoint, profiler_config=profiler)


def _first_result(values):
    for value in values:
        if value:
            return value
    raise RuntimeError("rank zero did not return a policy publication")


def _publication_export_options(*, checkpoint_due: bool,
                                rollout_transport_dtype: str,
                                bf16_gather_before_export: bool = False) -> dict:
    """Keep the experimental gather path out of every durable checkpoint."""
    options = {
        "transport_only": not checkpoint_due,
        "qwen_export_dtype": "preserve" if checkpoint_due else rollout_transport_dtype,
    }
    if bf16_gather_before_export and not checkpoint_due:
        if rollout_transport_dtype != "bf16":
            raise ValueError("BF16 gather before export requires BF16 rollout transport")
        options["bf16_gather_before_export"] = True
    return options


# Healthy native publication takes 12-85 s (r5); vLLM RPCs use 1800 s.
NATIVE_PUBLICATION_TIMEOUT_SECONDS = 1800


def _get_worker_result(value, *, timeout=None, phase="learner RPC"):
    return resolve_ray_results(value, ray_api=ray, timeout=timeout, phase=phase)


def _enqueue_streamed_group(
        learner, records, world: int, stage_records_per_rpc: int,
        token_balanced: bool, condition_batch_size: int,
        max_condition_tokens: int, condition_token_bucket_size: int = 0, *,
        skip_zero_advantage_records: bool = False,
        compact_zero_advantage_waves: bool = False,
        zero_advantage_reports: list[dict] | None = None,
        joint_text_fm_forward: bool = False,
        joint_reports: list[dict] | None = None, prepare_only: bool = False,
        stacked_replay_gate: dict | None = None):
    """Stage completed groups with final record advantages, then accumulate.

    Per-call credit is assigned with the previous frozen critic before this
    boundary. Stream waves change scheduling, not the logical-batch objective.
    """
    original_mass = None
    if joint_text_fm_forward:
        from action_head_rl.joint_policy import episode_text_mass
        original_mass = episode_text_mass(records)
    if compact_zero_advantage_waves and not skip_zero_advantage_records:
        raise ValueError("zero-wave compaction requires zero-advantage filtering")
    if skip_zero_advantage_records:
        # Filter before padding and preserve the original episode mass. All-zero
        # or missing-branch waves retain their graphs, including Adam's zero
        # gradients; they still count as one ordinary streamed accumulation.
        records, original_mass, filter_report = _zero_advantage_training_records(
            records, compact_zero_waves=compact_zero_advantage_waves)
    preflight = None
    if joint_text_fm_forward:
        from action_head_rl.joint_policy import build_joint_policy_groups, validate_joint_preflight_reports
        from action_head_rl.joint_scheduling import shard_joint_policy_groups
        groups, pairing = build_joint_policy_groups(records)
        # Streaming accumulates unnormalized gradients; finish applies the one
        # original episode denominator after every wave has acknowledged.
        rank_records, schedule = shard_joint_policy_groups(
            groups, world, logical_valid=1.0, balance_by_token_flops=token_balanced,
            condition_batch_size=condition_batch_size, max_padded_tokens=max_condition_tokens)
        logical_valid = original_mass
        def preflight():
            reports = _get_worker_result(learner.validate_staged_joint_records([1.0] * world),
                                        timeout=1200, phase="joint CPU preflight")
            validate_joint_preflight_reports(reports, schedule['rounds'], world=world, logical_valid=1.0)
            if joint_reports is not None:
                joint_reports.append(dict(pairing=pairing, scheduling=schedule, all_rank_preflight_passed=True))
    else:
        rank_records, logical_valid = shard_option_record_groups(
            records, world, balance_by_token_flops=token_balanced,
            condition_batch_size=condition_batch_size,
            max_padded_tokens=max_condition_tokens,
            condition_token_bucket_size=condition_token_bucket_size,
            **({"align_text_families": True}
               if original_mass is not None and filter_report.get("zero_wave_compacted") else {}))
    counts = [len(shard) for shard in rank_records]
    if len(set(counts)) != 1:
        raise RuntimeError(f"streamed rank records are not aligned: {counts}")
    def dispatch():
        # The controller has already drained the preceding all-rank backward.
        # Complete every staging acknowledgement before releasing any rank into
        # FSDP; async actors and resolver dependencies do not provide FIFO order.
        if stacked_replay_gate is not None and not stacked_replay_gate['passed']:
            from action_head_rl.rule_shared_audit import dispatch_stacked_rule_replay
            gate_started = time.perf_counter()
            # This RPC includes a reference backward, a repeatability backward,
            # and the candidate backward over the entire first online wave.
            # The ordinary one-wave backward deadline is not its work budget.
            gate_timeout = 3 * 3600
            print(json.dumps({'stacked_replay_gate_started': dict(
                output_dir=stacked_replay_gate['output_dir'],
                timeout_seconds=gate_timeout, backward_passes=3)}), flush=True)
            results = dispatch_stacked_rule_replay(
                learner, groups, world=world, token_balanced=token_balanced,
                output_dir=stacked_replay_gate['output_dir'],
                candidate_batch_size=condition_batch_size,
                candidate_max_padded_tokens=max_condition_tokens,
                resolve_waves=lambda waves: _resolve_rank_waves(waves, world),
                resolve_result=lambda refs: _get_worker_result(
                    refs, timeout=gate_timeout, phase='combined v3 replay gate'))
            stacked_replay_gate.update(passed=True, seconds=time.perf_counter()-gate_started)
            print(json.dumps({'stacked_replay_gate': stacked_replay_gate,
                              'rank_results': results}), flush=True)
        update = stage_aligned_wave(
            rank_records, stage_records_per_rpc=stage_records_per_rpc,
            shared_keys=OPTION_SHARED_KEYS, clear=learner.clear_staged_records,
            stage=lambda waves: _stage_rank_waves(learner, waves, world),
            accumulate=learner.accumulate_staged_records,
            resolve=lambda refs: _get_worker_result(refs, timeout=300, phase="clear/stage acknowledgement"),
            preflight=preflight)
        if original_mass is not None:
            if zero_advantage_reports is not None and skip_zero_advantage_records:
                zero_advantage_reports.append(filter_report)
        return update, original_mass if original_mass is not None else logical_valid, counts
    return dispatch if prepare_only else dispatch()


def select_colocated_placements(
        placements: list[dict], replica_count: int, tp_size: int = 1) -> list[dict]:
    """Select disjoint, node-local TP groups from learner GPUs, round-robin."""
    if tp_size <= 0:
        raise ValueError("tensor parallel size must be positive")
    if replica_count < 0:
        raise ValueError("colocated replica count must be nonnegative")
    ordered = sorted(placements, key=lambda row: int(row["rank"]))
    by_node: dict[str, list[dict]] = {}
    identities = set()
    for row in ordered:
        identity = (str(row["node_id"]), str(row["cuda_visible_device"]))
        if identity in identities:
            raise ValueError(f"duplicate learner GPU placement: {identity}")
        identities.add(identity)
        by_node.setdefault(str(row["node_id"]), []).append(row)
    selected = []
    nodes = sorted(by_node)
    while len(selected) < replica_count:
        progressed = False
        for node in nodes:
            if len(by_node[node]) >= tp_size and len(selected) < replica_count:
                group = by_node[node][:tp_size]
                del by_node[node][:tp_size]
                selected.append(dict(
                    group[0], ranks=[int(row["rank"]) for row in group],
                    cuda_visible_device=",".join(
                        str(row["cuda_visible_device"]) for row in group)))
                progressed = True
        if not progressed:
            raise ValueError(
                f"requested {replica_count} colocated replicas from "
                f"{len(placements)} learner GPUs")
    return selected


def require_valid_rollouts(rollouts: list[dict], *, step: int) -> None:
    """Refuse to optimize on infrastructure failures.

    An invalid rollout is not a negative navigation sample.  Continuing would
    silently assign it a task reward and corrupt both GRPO group normalization
    and the reported success denominator.
    """
    # Production SAGE rows must match the active geometry and immutable assets.
    # This catches old remote Gym servers as well as stale saved rollouts.
    from sage.goal_region import success_contract_matches
    for row in rollouts:
        result = row.get("result") or {}
        if (str(result.get("scene", "")).startswith("sage-")
                and result.get("evaluation_valid") is True
                and not success_contract_matches(result)):
            raise RuntimeError(f"step {step}: SAGE rollout geometry contract mismatch")
    invalid = [row for row in rollouts
               if not bool((row.get("result") or {}).get("evaluation_valid"))]
    if not invalid:
        return
    reasons: dict[str, int] = {}
    for row in invalid:
        reason = str((row.get("result") or {}).get("infra_error") or "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1
    summary = sorted(reasons.items(), key=lambda item: (-item[1], item[0]))[:5]
    raise RuntimeError(
        f"step {step} produced {len(invalid)}/{len(rollouts)} invalid rollouts; "
        f"refusing to train on infrastructure failures: {summary}")


def group_independent_episodes(flat_rollouts: list[dict], *,
                               num_groups: int,
                               group_size: int) -> list[list[dict]]:
    """Restore GxK results where every item is an independent full episode."""
    expected = num_groups * group_size
    if len(flat_rollouts) != expected:
        raise RuntimeError(
            f"rollout workers returned {len(flat_rollouts)} episodes, "
            f"expected {num_groups}x{group_size}={expected}")
    return [
        flat_rollouts[start:start + group_size]
        for start in range(0, expected, group_size)
    ]


def independent_episode_specs(step_cases: list[dict], *, step: int,
                              group_size: int) -> list[tuple[int, int, dict, int]]:
    """Expand every group case into K independently seeded episode calls."""
    return [
        (
            group_index,
            sample_index,
            case,
            20260831 + step * 100000 + group_index * group_size + sample_index,
        )
        for group_index, case in enumerate(step_cases)
        for sample_index in range(group_size)
    ]


def group_option_advantages(rollouts: list[dict]) -> list[dict[str, float]]:
    """Normalize return-to-go across same-case samples at each option depth.

    Later tool calls condition on different histories and trajectories may end
    at different lengths.  Comparing only the same option depth retains GRPO's
    same-task baseline while providing a separate advantage for every decision.
    """
    outputs: list[dict[str, float]] = [{} for _ in rollouts]
    credits = [list((row.get("reward_components") or {}).get(
        "option_credits") or []) for row in rollouts]
    max_depth = max((len(items) for items in credits), default=0)
    for depth in range(max_depth):
        present = [(index, items[depth]) for index, items in enumerate(credits)
                   if depth < len(items)]
        if len(present) < 2:
            normalized = [0.0] * len(present)
        else:
            values = torch.tensor(
                [float(item["return_to_go"]) for _, item in present])
            normalized = grpo_advantages(values, len(present)).tolist()
        for (rollout_index, item), advantage in zip(present, normalized):
            outputs[rollout_index][str(item["option_id"])] = float(advantage)
    return outputs


def group_trajectory_advantages(rollouts: list[dict]) -> list[float]:
    """Apply standard outcome GRPO to complete same-case trajectories.

    ``rollout['reward']`` already contains the discounted sum of all verifiable
    option process rewards plus the official terminal-success reward. GRPO
    standardizes that single outcome across the K same-case trajectories and
    broadcasts one scalar advantage to every trainable action in a trajectory.
    """
    if not rollouts:
        return []
    rewards = torch.tensor([float(row["reward"]) for row in rollouts])
    if rewards.numel() < 2:
        return [0.0] * len(rollouts)
    return [float(value) for value in grpo_advantages(
        rewards, int(rewards.numel())).tolist()]


def _driver_memory_gb() -> dict[str, float]:
    """Read Linux process RSS/high-water memory without another dependency."""
    values: dict[str, float] = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            key, _, raw = line.partition(":")
            if key in ("VmRSS", "VmHWM"):
                values[key] = float(raw.strip().split()[0]) / (1024 ** 2)
    except (OSError, ValueError, IndexError):
        pass
    return {
        "driver_rss_gb": values.get("VmRSS", 0.0),
        "driver_peak_rss_gb": values.get("VmHWM", 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--slow-model", required=True)
    parser.add_argument("--fast-backbone", required=True)
    parser.add_argument("--head-checkpoint", required=True)
    parser.add_argument("--query-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bench-urls", required=True)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--learner-cpus-per-rank", type=int, default=10)
    parser.add_argument("--learner-gpus-per-node", type=int, default=13)
    parser.add_argument("--learner-gpus-layout", default="",
                        help="optional comma-separated per-node learner ranks")
    parser.add_argument("--fsdp-size", type=int, default=-1,
                        help="FSDP shard group size; -1 shards across the learner world")
    parser.add_argument("--prepare-streamed-wave", action=argparse.BooleanOptionalAction,
                        default=False, help="CPU prepare the next wave before draining the preceding learner wave")
    parser.add_argument("--fsdp-forward-prefetch", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--train-scope", choices=[x.value for x in TrainScope],
                        default=TrainScope.HEAD_QUERY.value)
    parser.add_argument(
        "--learner-parameter-dtype", choices=("bf16", "fp32"), default="bf16",
        help="original parameter/Adam storage dtype; fp32 requires full scope; compute stays BF16")
    parser.add_argument(
        "--rollout-transport-dtype", choices=("preserve", "bf16"), default="preserve",
        help="Qwen export dtype between checkpoints; committed checkpoints always preserve learner precision")
    parser.add_argument(
        "--bf16-gather-before-export", action="store_true",
        help="experimental: cast eligible CUDA DTensor shards before ordinary BF16 transport gathers; checkpoints keep the original path")
    parser.add_argument(
        "--local-policy-transport", action="store_true",
        help="opt in to this single-host local Ray run's private /tmp BF16 transport; exact checkpoints stay in output")
    parser.add_argument("--router-replay", default="R3")
    parser.add_argument("--opt-steps", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--num-groups-per-step", type=int, default=2)
    parser.add_argument("--rollout-actors", type=int, default=1)
    parser.add_argument("--rollout-gpu-fraction", type=float, default=1.0)
    parser.add_argument(
        "--fast-policy-workers", type=int, default=0,
        help=("Batched policy replicas; zero retains the legacy "
              "one-model-per-rollout-actor path"))
    parser.add_argument("--text-before-fm", action="store_true",
                        help="accumulate text then FM gradients at the same weights; keep one joint optimizer step")
    parser.add_argument("--learner-collectives-only", action="store_true",
                        help="Diagnostic: validate actual Ray learner communication and exit before model loading")
    parser.add_argument("--text-backend", choices=("fsdp", "megatron"), default="fsdp")
    for parallel, default in (("tp", 2), ("pp", 2), ("cp", 8), ("ep", 16), ("etp", 1)):
        parser.add_argument("--megatron-" + parallel, type=int, default=default)
    parser.add_argument("--megatron-dynamic-bsz", action="store_true",
                        help="pack whole text trajectories using native Megatron token-budget batching")
    parser.add_argument("--megatron-max-token-len-per-gpu", type=int, default=16384)
    parser.add_argument("--megatron-cpu-prefetch", action="store_true",
                        help="prepare at most one next text microbatch on CPU during GPU work")
    parser.add_argument('--fm-prefix-reuse', action='store_true',
                        help='keep each FM trajectory on one learner and reuse its fixed-policy causal history')
    parser.add_argument("--trajectory-transfer-queue", action="store_true",
                        help="native TQ condition storage and independent bounded learner pulls")
    parser.add_argument('--group-partial-rollout', action='store_true',
                        help='native whole-group warmup1/stale4 partial rollout with behavior correction')
    parser.add_argument('--cpu-group-preparation', action='store_true',
                        help='prepare complete v3 groups on CPU workers; driver schedules metadata only')
    parser.add_argument('--overlap-cpu-group-preparation', action='store_true',
                        help='prepare each settled K group during sampling, then replace acknowledged raw payloads')
    parser.add_argument('--cpu-group-preparation-tasks-per-node', type=int, choices=(1, 2),
                        default=int(os.environ.get('ACTION_HEAD_CPU_GROUP_PREP_TASKS_PER_NODE', '1')),
                        help='bounded CPU preparation concurrency per node, with headroom for the second task')
    parser.add_argument("--fast-policy-device", choices=("cuda", "cpu", "colocated_cuda"), default="cuda")
    parser.add_argument("--fast-policy-cpus", type=int, default=1)
    parser.add_argument("--fast-policy-max-batch-size", type=int, default=8)
    parser.add_argument("--fast-policy-batch-wait-ms", type=float, default=12.0)
    parser.add_argument(
        "--rollout-groups-in-flight", type=int, default=2,
        help="bound completed rollout payloads retained in the Ray object store")
    parser.add_argument("--smoke-cases", type=int, default=2)
    parser.add_argument("--case-manifest", default="",
                        help="ordered training JSONL; uses all listed cases instead of smoke-cases")
    parser.add_argument("--eval-case-manifest", default="",
                        help="fixed disjoint heldout JSONL for step-zero and periodic evaluation")
    parser.add_argument("--eval-bench-urls", default="",
                        help="independent heldout bench URLs; one CPU actor per URL")
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--eval-repeats", type=int, default=1)
    parser.add_argument("--eval-seed", type=int, default=20260926)
    parser.add_argument("--eval-max-tool-calls", type=int, default=0,
                        help="zero inherits training tool budget")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--vlm-learning-rate", type=float, default=0.0,
        help=("LR for Qwen parameters in train-scope=full; zero reuses "
              "--learning-rate"))
    parser.add_argument(
        "--old-logprob-mode", choices=OLD_LOGPROB_MODES,
        default="recompute",
        help=("single_forward removes the redundant proximal pass when each "
              "logical batch has exactly one optimizer update"))
    parser.add_argument(
        "--skip-zero-advantage-records", action=argparse.BooleanOptionalAction,
        default=False,
        help="skip exact zero-advantage records before sharding while preserving original episode normalization")
    parser.add_argument("--joint-fixture-dir", default="", help="optional bounded CPU raw text/FM fixture export")
    parser.add_argument("--update-replay-dir", default="",
                        help="optional first-update CPU shards for matched backend audits; no rollout replay")
    parser.add_argument("--update-backend-audit-input", default="",
                        help="zero-update FSDP/Megatron comparison on a complete initial-policy archive")
    parser.add_argument("--joint-fixture-count", type=int, default=2)
    parser.add_argument("--joint-fixture-step", type=int, default=0, help="0 means first update of this run")
    parser.add_argument("--joint-text-fm-forward", action=argparse.BooleanOptionalAction,
                        default=False, help="experimental common-prefix forward; requires separate GPU parity approval")
    parser.add_argument("--tool-credit-mode", choices=("episode_grpo", "rule_call_loo", *TOOL_CREDIT_MODES),
                        default="episode_grpo")
    parser.add_argument("--rule-credit-version", choices=("v2", "v3_first_arrival", "v3_terminal_local", "v3_reject_guard", "v3_reject_guard_stop", "v4_boundary_stop_spl"), default="v2",
                        help="explicit reward revision; exact resume rejects a different revision")
    parser.add_argument("--allow-rule-credit-transition", action="store_true",
                        help="explicit full-state v3_first_arrival to rejection-guard experiment; all unrelated contract checks remain strict")
    parser.add_argument("--tool-credit-initial-state", default="",
                        help="explicit C16 critic seed for a one-time algorithm fork")
    parser.add_argument("--tool-credit-fit-updates", type=int, default=20,
                        help="CPU critic Adam iterations per completed actor batch; fixed by the resume contract")
    parser.add_argument("--allow-tool-credit-fork", action="store_true",
                        help="retain actor/Adam/RNG while explicitly changing episode credit to call credit")
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction,
        default=True)
    parser.add_argument('--fm-suffix-vjp', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--fm-core-suffix-vjp', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--dual-shared-recompute', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--fast-causal-attention', action=argparse.BooleanOptionalAction, default=False,
                        help='B1 full learner: reuse native attention primals and require mask-free causal backward')
    parser.add_argument('--full-attention-checkpoint-layers-off', type=int, default=0,
                        help='retain activations in the last N native full-attention decoder layers')
    parser.add_argument('--optimizer-cpu-during-backward', action='store_true',
                        help='keep native FP32 Adam moments on CPU during Full streamed backward; load only for its GPU update')
    parser.add_argument(
        "--option-condition-reuse", action=argparse.BooleanOptionalAction,
        default=False,
        help=("reuse one differentiable Qwen/R3 condition for consecutive "
              "chunks from the same option; rollout/reward/update logic is unchanged"))
    parser.add_argument(
        "--piold-option-reuse", action=argparse.BooleanOptionalAction, default=False,
        help="recompute fixed pi_old on staged option groups before the optimizer update")
    parser.add_argument(
        "--frozen-weight-residency", action=argparse.BooleanOptionalAction, default=False,
        help="keep Head-only FSDP weights gathered during accumulation; reshard before optimizer step")
    parser.add_argument(
        "--full-weight-residency-gib", type=float, default=0.0,
        help="Full-only selected FSDP weight residency budget in GiB; zero disables it")
    parser.add_argument(
        "--r3-capture-mode", choices=("legacy", "native_audit", "native"),
        default=os.environ.get("ACTION_HEAD_R3_CAPTURE_MODE", "legacy"),
        help="full-route collection source; R3 replay and prefix-cache policy remain unchanged")
    parser.add_argument("--r3-audit-dir", default=os.environ.get("ACTION_HEAD_R3_AUDIT_DIR", ""))
    parser.add_argument(
        "--defer-query-capture", action=argparse.BooleanOptionalAction,
        default=False,
        help="defer legacy query-hidden D2H copies until tool-call completion")
    parser.add_argument(
        "--stage-records-per-rpc", type=int, default=1,
        help="number of aligned rank-local records transported per Ray staging call")
    parser.add_argument(
        "--stream-groups-per-update", type=int, default=1,
        help="completed GRPO groups coalesced into one streamed FSDP accumulation")
    parser.add_argument(
        "--token-balanced-sharding", action=argparse.BooleanOptionalAction,
        default=False,
        help=("Full-only: reuse VERL's transformer-FLOPs sequence partitioner "
              "to balance complete option conditions across learner ranks"))
    parser.add_argument(
        "--multi-condition-batch-size", type=int, default=1,
        help="Full-only number of independent option conditions per Qwen learner forward")
    parser.add_argument(
        "--max-condition-tokens", type=int, default=0,
        help="Full-only dynamic batch cap on condition_count times padded prefix tokens")
    parser.add_argument(
        "--condition-token-bucket-size", type=int, default=0,
        help=("Full-only token multiple for aligned padded widths; bounds "
              "shape-dependent compiler variants, zero disables"))
    parser.add_argument(
        "--zero-copy-rollout-payload", action=argparse.BooleanOptionalAction,
        default=False,
        help="Full-only keep tensor-rich rollout payloads in Ray Plasma until learner staging")
    parser.add_argument(
        "--direct-vllm-weight-sync", action=argparse.BooleanOptionalAction,
        default=False,
        help="Full-only VERL NCCL to relay plus CUDA IPC publication; skips per-step HF export")
    parser.add_argument("--native-colocated-weight-sync", action=argparse.BooleanOptionalAction,
                        default=False, help="Megatron native per-GPU bucketed IPC publication")
    parser.add_argument("--weight-sync-bucket-mb", type=int, default=2048)
    parser.add_argument(
        "--overlap-rollout-learner", action=argparse.BooleanOptionalAction,
        default=False,
        help="Full-only stream completed GRPO groups into backward before rollout ends")
    parser.add_argument(
        "--rollout-background-prefetch", action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep completion-scheduled actors supplied during streamed learner waits")
    parser.add_argument("--v8-prefix-cache", action=argparse.BooleanOptionalAction, default=None,
                        help="Reuse per-episode hybrid KV/GDN state with certified native R3 rows (default on for v8)")
    parser.add_argument("--v8-cuda-graphs", action="store_true",
                        help="Experimental graph execution with tensor-buffer historical route replay")
    parser.add_argument(
        "--compact-zero-advantage-waves", action=argparse.BooleanOptionalAction,
        default=False,
        help="Experimental B1 streamed GRPO: retain one real graph per all-zero forward family")
    parser.add_argument(
        "--defer-gradient-sync", action=argparse.BooleanOptionalAction, default=False,
        help="sync FSDP2 gradients only on the final microbatch (head_only only)")
    parser.add_argument("--sde-steps", type=int, default=3)
    parser.add_argument("--sde-noise", type=float, default=0.08)
    parser.add_argument("--terminal-success-reward", type=float, default=1.0)
    parser.add_argument("--direction-match-reward", type=float, default=0.05)
    parser.add_argument("--collision-penalty", type=float, default=0.05)
    parser.add_argument("--discount-gamma", type=float, default=1.0)
    parser.add_argument(
        "--episode-grpo", action="store_true",
        help="use complete-episode GRPO and one slow decision per tool call in either scope")
    parser.add_argument("--max-turns", type=int, default=2)
    parser.add_argument("--max-tool-calls", type=int, default=2)
    parser.add_argument("--fm-chunks-per-episode", type=int, default=8)
    parser.add_argument("--vllm-port", type=int, default=28899)
    parser.add_argument("--vllm-tp-size", type=int, default=2)
    parser.add_argument("--vllm-replicas", type=int, default=1)
    parser.add_argument(
        "--colocated-vllm-replicas", type=int, default=0,
        help=("Full-only node-local TP replicas that time-share learner GPUs through "
              "FSDP CPU offload and vLLM sleep/wake"))
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--resume-audit", action="store_true")
    parser.add_argument("--stop-after-step", type=int, default=0)
    parser.add_argument("--keep-last-checkpoints", type=int, default=0)
    parser.add_argument(
        "--checkpoint-interval", type=int, default=1,
        help="commit a resumable model/optimizer/RNG checkpoint every N steps")
    parser.add_argument("--case-shuffle-seed", type=int, default=None)
    parser.add_argument("--dataset-split", default="action_head_train_p0_unified")
    parser.add_argument("--case-sampling", choices=("sequential", "uniform_per_step", "stratified_per_step"), default="sequential")
    parser.add_argument("--case-source-groups", type=json.loads, default=None)
    parser.add_argument("--case-sampling-seed", type=int, default=20261010)
    parser.add_argument("--harness-version", choices=("v1", "v3", "v6", "v8"), default="v1")
    parser.add_argument("--text-policy", action="store_true",
                        help="v6/v8: on-policy text GPG plus continuous FM policy loss")
    parser.add_argument("--text-loss-coef", type=float, default=0.1)
    parser.add_argument("--text-temperature", type=float, default=1.0)
    parser.add_argument("--success-contract", choices=("legacy_object_arrival", "native_success", "native_success_and_stop"),
                        default="legacy_object_arrival")
    parser.add_argument("--process-reward-cap", type=float, default=None)
    parser.add_argument("--resume-mode", choices=("disable", "auto", "resume_path"),
                        default="disable")
    parser.add_argument("--resume-from-path", default="")
    parser.add_argument("--policy-warm-start-checkpoint", default="",
                        help="new actor optimizer/scheduler/RNG from a complete bound GAE policy+critic checkpoint")
    parser.add_argument("--allow-weights-only-resume", action="store_true")
    parser.add_argument("--allow-rollout-topology-change", action="store_true",
                        help="explicit full-state resume with a different node-local vLLM TP/replica layout")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--rollout-scheduler", choices=("wave", "completion"), default="wave")
    parser.add_argument("--fast-policy-pipeline", action="store_true")
    parser.add_argument("--bulk-route-capture", action="store_true")
    parser.add_argument("--vllm-max-num-seqs", type=int, default=0)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--rollout-benchmark", action="store_true")
    parser.add_argument("--benchmark-variants", default="baseline16,bulk16,pipeline64")
    args = parser.parse_args()
    os.environ["ACTION_HEAD_RULE_TERMINAL_SUCCESS_REWARD"] = str(args.terminal_success_reward)
    if args.allow_rollout_topology_change and (
            args.resume_mode != 'resume_path' or not args.resume_from_path
            or args.allow_weights_only_resume or args.policy_warm_start_checkpoint):
        raise ValueError('rollout topology change requires an explicit full-state resume path')
    if args.v8_prefix_cache is None:
        args.v8_prefix_cache = args.harness_version == "v8"
    if args.policy_warm_start_checkpoint and (
            args.resume_mode != "disable" or args.resume_from_path
            or args.allow_weights_only_resume or args.allow_tool_credit_fork
            or args.allow_rule_credit_transition
            or args.tool_credit_initial_state or args.tool_credit_mode not in TOOL_CREDIT_MODES):
        raise ValueError("policy warm-start requires previous-value credit and cannot mix resume/fork/independent critic seed")
    if args.r3_capture_mode not in ("legacy", "native_audit", "native"):
        parser.error("ACTION_HEAD_R3_CAPTURE_MODE must be legacy, native_audit or native")
    if args.rollout_benchmark and args.validate_only:
        parser.error(
            "--validate-only cannot be combined with --rollout-benchmark; "
            "use the DLC submitter --dry-run for benchmark resource preflight")
    if args.rollout_benchmark:
        from action_head_rl.benchmark import run_benchmark
        run_benchmark(args)
        return

    scope, _ = validate_training_contract(
        train_scope=args.train_scope, router_replay_mode=args.router_replay,
        capture_full_routes=True,
        rollout_backend="vllm")
    if args.learner_parameter_dtype == "fp32" and scope is not TrainScope.FULL:
        raise ValueError("FP32 learner parameters require train_scope=full")
    if args.rollout_transport_dtype != "preserve" and scope is not TrainScope.FULL:
        raise ValueError("BF16 rollout transport requires train_scope=full")
    if args.bf16_gather_before_export and (
            scope is not TrainScope.FULL or args.learner_parameter_dtype != "fp32"
            or args.rollout_transport_dtype != "bf16" or args.direct_vllm_weight_sync):
        raise ValueError(
            "BF16 gather before export requires full scope, FP32 learner parameters, "
            "BF16 rollout transport and file-based weight sync")
    if args.harness_version == "v8":
        from action_head_rl.trajectory_text import validate_v8_options
        validate_v8_options(vars(args))
    if args.text_policy:
        if args.harness_version not in ("v6", "v8") or scope is not TrainScope.FULL:
            raise ValueError("--text-policy requires --harness-version v6 --train-scope full")
        if args.r3_capture_mode != "native" or args.old_logprob_mode != "single_forward":
            raise ValueError("v6 text GPG requires native R3 and single_forward")
        if args.multi_condition_batch_size not in (1, 2, 4, 8) or args.piold_option_reuse:
            raise ValueError("v6 text GPG requires condition batch 1/2/4/8 and no pi-old replay")
        if args.multi_condition_batch_size > 1 and not (args.tool_credit_mode == "episode_grpo" or (args.tool_credit_mode == "rule_call_loo" and args.joint_text_fm_forward)):
            raise ValueError("v6 text packing currently requires episode_grpo")
        if not args.option_condition_reuse:
            raise ValueError("v6 text GPG requires option-condition-reuse for homogeneous rank waves")
        if (not math.isfinite(args.text_loss_coef) or args.text_loss_coef <= 0
                or not math.isfinite(args.text_temperature) or args.text_temperature <= 0):
            raise ValueError("text coefficient and sampling temperature must be positive and finite")
        if args.success_contract == "legacy_object_arrival":
            raise ValueError("v6 text policy requires an explicit native success contract")
    from action_head_rl.joint_fixture import validate_fixture_options
    validate_fixture_options(args.joint_fixture_dir, args.joint_fixture_count, args.joint_fixture_step)
    validate_fixture_options(args.update_replay_dir, 1, 0)
    if args.update_replay_dir and args.overlap_rollout_learner:
        raise ValueError("update replay capture requires full-step staging")
    if args.update_backend_audit_input:
        validate_fixture_options(args.update_backend_audit_input, 1, 0)
        if args.text_backend != 'megatron' or args.overlap_rollout_learner:
            raise ValueError('backend audit requires the full-step Megatron layout')
    if args.joint_fixture_dir and (not args.text_policy or args.overlap_rollout_learner):
        raise ValueError("fixture export requires non-streamed text training")
    if args.joint_text_fm_forward and (
            not args.text_policy or scope is not TrainScope.FULL
            or args.r3_capture_mode != "native" or args.old_logprob_mode != "single_forward"
            or not args.option_condition_reuse or args.multi_condition_batch_size not in (1, 2, 4, 8)
            or (args.multi_condition_batch_size > 1 and args.tool_credit_mode != "rule_call_loo")
            or (args.overlap_rollout_learner and args.tool_credit_mode != "rule_call_loo") or args.condition_token_bucket_size):
        raise ValueError("joint text/FM requires Full native text policy, single_forward, B1/2/4/8 option reuse and unbucketed scheduling")
    # The explicit comparison flag extends Full's episode credit and corrected
    # tool budget to Head-only. Archived Head-only presets keep their contract.
    episode_grpo = args.episode_grpo or scope is TrainScope.FULL
    rule_credit_enabled = args.tool_credit_mode == "rule_call_loo"
    if args.rule_credit_version != "v2" and not rule_credit_enabled:
        raise ValueError("rule-credit-version requires rule_call_loo")
    if args.allow_rule_credit_transition and (
            not rule_credit_enabled or args.resume_mode != "resume_path"
            or not args.resume_from_path or args.allow_weights_only_resume
            or args.allow_tool_credit_fork or args.tool_credit_initial_state
            or args.rule_credit_version not in ("v3_reject_guard", "v3_reject_guard_stop")):
        raise ValueError("rule-credit transition requires an explicit full-state resume_path and a rejection-guard target")
    synchronous_rule_update = (args.text_before_fm and args.colocated_vllm_replicas > 0
                               and args.fast_policy_device in ("cpu", "colocated_cuda")
                               and not args.overlap_rollout_learner)
    if args.megatron_dynamic_bsz and args.text_backend != 'megatron':
        raise ValueError('Megatron dynamic batching requires the Megatron text backend')
    if args.megatron_cpu_prefetch and args.text_backend != 'megatron':
        raise ValueError('text CPU prefetch requires the Megatron text backend')
    if args.fm_prefix_reuse and (args.text_backend != 'megatron' or args.harness_version != 'v8'
            or not args.option_condition_reuse or args.multi_condition_batch_size != 1
            or args.joint_text_fm_forward or args.old_logprob_mode != 'single_forward'):
        raise ValueError('FM prefix reuse requires v8 Megatron, separate FM, option reuse and single-forward logprobs')
    if args.trajectory_transfer_queue and (args.text_backend != 'megatron'
            or args.harness_version != 'v8' or args.overlap_rollout_learner):
        raise ValueError('trajectory TransferQueue requires full-step v8 Megatron training')
    if args.cpu_group_preparation and (not args.group_partial_rollout
            or args.rule_credit_version not in ('v3_first_arrival', 'v3_terminal_local')):
        raise ValueError('CPU group preparation requires partial v3 whole-group credit')
    if args.overlap_cpu_group_preparation and not args.cpu_group_preparation:
        raise ValueError('CPU preparation overlap requires CPU group preparation')
    if (args.cpu_group_preparation_tasks_per_node not in (1, 2)
            or (args.cpu_group_preparation_tasks_per_node != 1
                and not args.overlap_cpu_group_preparation)):
        raise ValueError('two CPU preparation tasks per node require CPU preparation overlap')
    if args.group_partial_rollout and (not args.trajectory_transfer_queue or not args.fast_policy_workers
            or not args.text_policy or not args.episode_grpo
            or args.text_temperature != 1. or args.old_logprob_mode != 'single_forward'
            or args.tool_credit_mode != 'rule_call_loo'
            or args.rollout_background_prefetch or args.joint_fixture_dir):
        raise ValueError('group partial requires v8/Megatron TQ, shared FM, temperature1, v3 single update, '
                         'and decision-boundary admission; no legacy prefetch or fixture')
    if args.text_backend == "megatron":
        if args.megatron_max_token_len_per_gpu <= 0:
            raise ValueError('Megatron per-GPU token budget must be positive')
        if (args.harness_version != "v8" or not synchronous_rule_update
                or args.old_logprob_mode != "single_forward" or args.joint_text_fm_forward
                or args.resume_audit
                or args.policy_warm_start_checkpoint
                or args.bf16_gather_before_export or args.direct_vllm_weight_sync):
            raise ValueError("Megatron v8 requires full-step colocation, native separate replay, "
                             "single_forward and unchanged record mass")
    if rule_credit_enabled and (
            args.group_size < 2
            or (args.multi_condition_batch_size > 1 and not args.joint_text_fm_forward)
            or (not args.overlap_rollout_learner and not synchronous_rule_update)
            or args.compact_zero_advantage_waves or args.joint_fixture_dir):
        raise ValueError("rule credit requires independent group samples and either synchronized "
                         "streaming or a full-step colocated text-then-FM update")
    if args.prepare_streamed_wave and not args.overlap_rollout_learner:
        raise ValueError("prepare-streamed-wave requires streamed training")
    if args.tool_credit_fit_updates <= 0:
        raise ValueError("tool-credit-fit-updates must be positive")
    if args.tool_credit_mode != "episode_grpo" and (
            not args.text_policy or args.harness_version not in ("v6", "v8")
            or scope is not TrainScope.FULL
            or args.direction_match_reward != 0.0
            or args.collision_penalty != 0.0 or args.discount_gamma != 1.0
            or args.success_contract != ("native_success_and_stop"
                if args.rule_credit_version == "v4_boundary_stop_spl" else "native_success")
            or (args.max_tool_calls != 50
                and not (args.harness_version == "v8" and rule_credit_enabled))):
        raise ValueError(f"{args.tool_credit_mode} requires Full text training with native reward; only v8 rule credit supports shorter budgets")
    if args.tool_credit_mode == "previous_value_mc_loo" and args.group_size < 2:
        raise ValueError("MC/LOO credit requires group-size >= 2")
    if args.skip_zero_advantage_records and (
            not episode_grpo or scope is not TrainScope.FULL
            or args.old_logprob_mode != "single_forward"
            or not args.option_condition_reuse
            or ((args.multi_condition_batch_size != 1 or args.overlap_rollout_learner)
                and (not args.text_policy or args.harness_version not in ("v6", "v8")
                     or args.r3_capture_mode != "native"
                     or args.tool_credit_mode not in ("episode_grpo", "rule_call_loo")
                     or args.multi_condition_batch_size not in (1, 2, 4, 8)))):
        raise ValueError(
            "zero-advantage filtering requires Full episode GRPO, single_forward, "
            "option reuse; packing/streaming additionally requires native v6 "
            "text episode-GRPO with condition batch 1/2/4/8")
    unified_tool_budget = episode_grpo
    if args.compact_zero_advantage_waves and (
            not args.skip_zero_advantage_records or not args.overlap_rollout_learner
            or not args.text_policy or args.harness_version not in ("v6", "v8")
            or args.multi_condition_batch_size != 1):
        raise ValueError("zero-wave compaction requires filtered B1 streamed native v6 text GRPO")
    rollout_max_turns = (args.max_tool_calls if unified_tool_budget
                         else args.max_turns)
    rollout_fast_step_limit = (4 if unified_tool_budget
                               else args.fm_chunks_per_episode)
    # Production data tops out at 176 completion tokens.  A 512-token Full
    # cap preserves ample margin while preventing one malformed generation
    # from consuming the context/latency budget of a 30-decision episode.
    # Final executable JSON has bounded strings/plans; preserve its full legal support.
    rollout_max_completion_tokens = ((2304 if args.rule_credit_version == "v4_boundary_stop_spl"
                                      else 512) if unified_tool_budget else 0)
    fla_l2norm_dynamic_nb = os.environ.get(
        "ACTION_HEAD_FLA_L2NORM_DYNAMIC_NB", "0")
    if fla_l2norm_dynamic_nb not in ("0", "1"):
        raise ValueError("ACTION_HEAD_FLA_L2NORM_DYNAMIC_NB must be 0 or 1")
    fla_l2norm_dynamic_nb_enabled = (
        scope is TrainScope.FULL and fla_l2norm_dynamic_nb == "1")
    fla_unused_shape_dynamic = os.environ.get(
        "ACTION_HEAD_FLA_UNUSED_SHAPE_DYNAMIC", "0")
    if fla_unused_shape_dynamic not in ("0", "1"):
        raise ValueError("ACTION_HEAD_FLA_UNUSED_SHAPE_DYNAMIC must be 0 or 1")
    fla_unused_shape_dynamic_enabled = (
        scope is TrainScope.FULL and fla_unused_shape_dynamic == "1")
    triton_autotune_seed = os.environ.get("ACTION_HEAD_TRITON_AUTOTUNE_SEED", "")
    if triton_autotune_seed:
        seed_manifest = Path(triton_autotune_seed) / "manifest.json"
        if not seed_manifest.is_file():
            raise FileNotFoundError(f"missing Triton autotune seed manifest: {seed_manifest}")
        # Full learner ranks are the only consumers; prevent an inherited
        # experimental seed from entering the finalized Head-only runtime.
        if scope is not TrainScope.FULL:
            triton_autotune_seed = ""
    RewardConfig(
        terminal_success_reward=args.terminal_success_reward,
        direction_match_reward=args.direction_match_reward,
        collision_penalty=args.collision_penalty,
        discount_gamma=args.discount_gamma,
        success_contract=args.success_contract, process_reward_cap=args.process_reward_cap,
    ).validate()
    if args.nnodes <= 0 or args.rollout_actors <= 0 or args.num_groups_per_step <= 0:
        raise ValueError(
            "nnodes, rollout-actors, and num-groups-per-step must be positive")
    if not 0.0 < args.rollout_gpu_fraction <= 1.0:
        raise ValueError("rollout-gpu-fraction must be in (0, 1]")
    if args.text_before_fm and (not args.text_policy or not args.option_condition_reuse
                                or args.overlap_rollout_learner or args.joint_text_fm_forward):
        raise ValueError("text-before-fm requires full-step separate text/FM with option reuse")
    if args.learner_cpus_per_rank <= 0:
        raise ValueError("learner-cpus-per-rank must be positive")
    if args.fast_policy_cpus <= 0:
        raise ValueError("fast-policy-cpus must be positive")
    if args.fast_policy_device == "cpu" and not args.fast_policy_workers:
        raise ValueError("CPU fast policy requires shared fast-policy-workers")
    if args.fast_policy_device == 'colocated_cuda' and (
            not args.fast_policy_workers or not args.colocated_vllm_replicas
            or args.overlap_rollout_learner or args.rollout_background_prefetch
            or args.direct_vllm_weight_sync):
        raise ValueError('colocated CUDA fast policy requires quiescent colocated rollout/update phases')
    if args.fast_policy_workers < 0 or args.fast_policy_max_batch_size <= 0 \
            or args.fast_policy_batch_wait_ms < 0:
        raise ValueError("invalid fast-policy worker or batching settings")
    if (not math.isfinite(args.learning_rate) or not math.isfinite(args.vlm_learning_rate)
            or args.learning_rate <= 0 or args.vlm_learning_rate < 0):
        raise ValueError("learning rates must be finite and positive (VLM may be zero=inherit)")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight-decay must be finite and nonnegative")
    if args.rollout_groups_in_flight <= 0:
        raise ValueError("rollout-groups-in-flight must be positive")
    if (args.vllm_tp_size <= 0 or args.vllm_replicas < 0
            or args.vllm_replicas + args.colocated_vllm_replicas <= 0):
        raise ValueError("positive TP size and at least one vLLM replica are required")
    if args.colocated_vllm_replicas < 0:
        raise ValueError("colocated-vllm-replicas must be nonnegative")
    if args.colocated_vllm_replicas and scope is not TrainScope.FULL:
        raise ValueError("colocated vLLM replicas are supported only for full training")
    if args.stage_records_per_rpc <= 0:
        raise ValueError("stage-records-per-rpc must be positive")
    if args.stream_groups_per_update <= 0:
        raise ValueError("stream-groups-per-update must be positive")
    if args.rollout_background_prefetch and (
            not args.overlap_rollout_learner or args.rollout_scheduler != "completion"):
        raise ValueError("background rollout prefetch requires streamed training and completion scheduling")
    if args.v8_prefix_cache and (args.harness_version != "v8" or scope is not TrainScope.FULL):
        raise ValueError("prefix state reuse requires v8 full RL")
    if args.v8_cuda_graphs and (args.harness_version != "v8" or scope is not TrainScope.FULL
                               or args.r3_capture_mode != "native"):
        raise ValueError("v8 CUDA graphs require v8 full RL and native R3 capture")
    if args.token_balanced_sharding and scope is not TrainScope.FULL:
        raise ValueError("token-balanced-sharding is supported only for full training")
    if args.rule_credit_version == "v4_boundary_stop_spl" and args.multi_condition_batch_size != 1:
        raise ValueError("final executable policy requires B1 replay")
    if args.multi_condition_batch_size <= 0:
        raise ValueError("multi-condition-batch-size must be positive")
    if args.max_condition_tokens < 0:
        raise ValueError("max-condition-tokens must be nonnegative")
    if args.condition_token_bucket_size < 0:
        raise ValueError("condition-token-bucket-size must be nonnegative")
    if args.multi_condition_batch_size > 1 and scope is not TrainScope.FULL:
        raise ValueError("multi-condition batching is supported only for full training")
    if args.condition_token_bucket_size and scope is not TrainScope.FULL:
        raise ValueError("condition token bucketing is supported only for full training")
    if args.zero_copy_rollout_payload and scope is not TrainScope.FULL:
        raise ValueError("zero-copy rollout payloads are supported only for full training")
    if args.direct_vllm_weight_sync and scope is not TrainScope.FULL:
        raise ValueError("direct vLLM weight sync is supported only for full training")
    if args.direct_vllm_weight_sync and (
            args.vllm_tp_size not in (1, 2) or args.colocated_vllm_replicas):
        raise ValueError("direct vLLM weight sync requires static TP1/TP2 replicas")
    if args.native_colocated_weight_sync and (args.text_backend != 'megatron'
            or scope is not TrainScope.FULL or args.vllm_replicas
            or not args.colocated_vllm_replicas or args.direct_vllm_weight_sync
            or args.local_policy_transport):
        raise ValueError('native colocated publication requires full Megatron and only colocated receivers')
    if args.weight_sync_bucket_mb <= 0:
        raise ValueError("weight-sync-bucket-mb must be positive")
    if args.overlap_rollout_learner and (
            scope is not TrainScope.FULL
            or args.old_logprob_mode != "single_forward"
            or not args.option_condition_reuse or args.colocated_vllm_replicas):
        raise ValueError(
            "rollout/learner overlap requires Full, single_forward, option reuse and disjoint rollout GPUs")
    if args.piold_option_reuse and (
            not args.option_condition_reuse or args.old_logprob_mode != "recompute"):
        raise ValueError("piold-option-reuse requires option-condition-reuse and recompute mode")
    if not math.isfinite(args.full_weight_residency_gib) or args.full_weight_residency_gib < 0:
        raise ValueError("full-weight-residency-gib must be finite and nonnegative")
    if args.full_weight_residency_gib > 0 and args.frozen_weight_residency:
        raise ValueError("Full and Head frozen weight residency are mutually exclusive")
    if args.full_weight_residency_gib > 0 and scope is not TrainScope.FULL:
        raise ValueError("positive full-weight-residency-gib requires train_scope=full")
    if args.full_weight_residency_gib > 0 and args.defer_gradient_sync:
        raise ValueError(
            "full-weight-residency requires gradient synchronization on every backward")
    # Full streamed accumulation reduce-scatters every backward. Residency
    # restores original shards at each wave boundary, before the single Adam
    # update. The runtime also checks every selected FSDP group's sync flags.
    if args.frozen_weight_residency and scope is not TrainScope.HEAD_ONLY:
        raise ValueError("frozen-weight-residency is supported only for head_only")
    if args.defer_gradient_sync and scope is not TrainScope.HEAD_ONLY:
        raise ValueError("defer-gradient-sync is supported only for head_only")
    if any((args.fast_causal_attention, args.fm_suffix_vjp,
            args.fm_core_suffix_vjp, args.dual_shared_recompute)):
        raise ValueError("retired experimental backward kernels are not available on main")
    if not 0 <= args.full_attention_checkpoint_layers_off < 10:
        raise ValueError('full-attention checkpoint selection must be in [0,9]')
    if args.full_attention_checkpoint_layers_off and (
            scope is not TrainScope.FULL or not args.gradient_checkpointing):
        raise ValueError('partial checkpoints require Full scope with global checkpointing enabled')
    if args.optimizer_cpu_during_backward and (
            scope is not TrainScope.FULL or not args.overlap_rollout_learner
            or args.learner_parameter_dtype != 'fp32'):
        raise ValueError('optimizer CPU residency requires Full FP32 streamed training')
    learner_layout = (
        [int(value) for value in args.learner_gpus_layout.split(",") if value]
        if args.learner_gpus_layout else
        [args.learner_gpus_per_node] * args.nnodes)
    if len(learner_layout) != args.nnodes or any(
            value < 1 or value > 16 for value in learner_layout):
        raise ValueError(
            "learner-gpus-layout must contain one value in [1,16] per node")
    learner_world = sum(learner_layout)
    stacked_replay_dir = os.environ.get('ACTION_HEAD_STACKED_RULE_REPLAY_DIR')
    stacked_replay_gate = (dict(output_dir=stacked_replay_dir, passed=False)
                           if stacked_replay_dir else None)
    packed_gate_recipe = (args.multi_condition_batch_size == 4
                          and args.max_condition_tokens == 40960
                          and args.full_attention_checkpoint_layers_off == 0)
    partial_gate_recipe = (args.multi_condition_batch_size == 1
                           and args.max_condition_tokens == 0
                           and args.full_attention_checkpoint_layers_off in (4, 6, 8))
    fast_causal_gate_recipe = (args.fast_causal_attention
                        and args.multi_condition_batch_size == 1 and args.max_condition_tokens == 0
                        and args.full_attention_checkpoint_layers_off == 0
                        and args.full_weight_residency_gib == 0 and args.stage_records_per_rpc == 4
                        and not args.optimizer_cpu_during_backward)
    if stacked_replay_gate is not None and (
            not rule_credit_enabled or args.rule_credit_version not in ('v3_first_arrival', 'v4_boundary_stop_spl')
            or not args.joint_text_fm_forward or (not fast_causal_gate_recipe and not args.prepare_streamed_wave)
            or args.ray_address != 'local' or learner_layout != [12]
            or not (packed_gate_recipe or partial_gate_recipe or fast_causal_gate_recipe)
            or (not fast_causal_gate_recipe and (args.stage_records_per_rpc != 16 or args.full_weight_residency_gib not in (4., 8.)))
            or not args.gradient_checkpointing or args.learner_parameter_dtype != 'fp32'):
        raise ValueError('stacked replay gate requires the complete local12 v3 performance candidate')
    if args.colocated_vllm_replicas * args.vllm_tp_size > learner_world:
        raise ValueError("colocated vLLM GPU count cannot exceed learner world size")
    if args.fsdp_size != -1 and (
            args.fsdp_size <= 1 or learner_world % args.fsdp_size != 0):
        raise ValueError("fsdp-size must be -1 or a divisor >1 of learner world size")
    # DLC workers are fixed 16-GPU nodes.  Local smoke may intentionally expose
    # only GPU 0..7; report and validate the actual visible budget there.
    local_visible_gpus = torch.cuda.device_count() if args.ray_address == "local" else None
    gpu_budget_is_static = bool(args.validate_only and local_visible_gpus == 0)
    total_gpus = (args.nnodes * 16 if args.ray_address != "local" or gpu_budget_is_static
                  else local_visible_gpus)
    rollout_reserved_gpus = (
        float(args.fast_policy_workers) * (args.fast_policy_device == "cuda")
        if args.fast_policy_workers else
        args.rollout_actors * args.rollout_gpu_fraction)
    vllm_reserved_gpus = args.vllm_tp_size * args.vllm_replicas
    requested_gpus = learner_world + vllm_reserved_gpus + rollout_reserved_gpus
    if not 1 <= args.learner_gpus_per_node <= 16:
        raise ValueError("learner-gpus-per-node must be in [1, 16]")
    if requested_gpus > total_gpus:
        raise ValueError(
            f"GPU allocation exceeds Ray workers: learner={learner_world} + "
            f"vLLM={args.vllm_replicas}xTP{args.vllm_tp_size}="
            f"{vllm_reserved_gpus} + rollout={args.rollout_actors}x"
            f"{args.rollout_gpu_fraction}={rollout_reserved_gpus:g} "
            f"> total={total_gpus}")
    if args.group_size < 2:
        raise ValueError("GRPO group_size must be at least two")
    if args.max_tool_calls <= 0:
        raise ValueError("max-tool-calls must be positive")
    if not unified_tool_budget and args.max_turns <= 0:
        raise ValueError("max-turns must be positive")
    if not unified_tool_budget and args.fm_chunks_per_episode <= 0:
        raise ValueError("fm-chunks-per-episode must be positive")
    output = Path(args.output_dir).resolve()
    local_transport = None
    if args.local_policy_transport:
        local_transport = LocalPolicyTransport(
            output=output, runtime_root=os.environ.get("ACTION_HEAD_LOCAL_RUNTIME_ROOT", ""),
            nnodes=args.nnodes, ray_address=args.ray_address,
            rollout_transport_dtype=args.rollout_transport_dtype)
        # Read-only gate before Ray/GPU startup. Creation is inside the main
        # cleanup scope, with another capacity check immediately beforehand.
        local_transport.preflight()
    r3_capture = {
        "mode": args.r3_capture_mode,
        "audit_dir": (str(Path(args.r3_audit_dir).resolve()) if args.r3_audit_dir
                      else str(output / "native_r3_audit") if args.r3_capture_mode != "legacy" else ""),
        "prefix_caching": args.v8_prefix_cache,
    }
    resume = resolve_resume_checkpoint(
        mode=args.resume_mode, output_dir=output,
        resume_from_path=args.resume_from_path,
        train_scope=scope.value, learner_world_size=learner_world,
        allow_weights_only=args.allow_weights_only_resume)
    start_step = resume.step if resume is not None else 0
    partial_resume = None
    if resume is not None and args.text_backend == 'megatron':
        if resume.exact and not (resume.trainer_state / 'joint_state.json').is_file():
            raise ValueError('Megatron resume requires complete joint FM/text training state')
        if args.group_partial_rollout:
            from action_head_rl.partial_groups import resume_admission_state
            partial_resume = resume_admission_state(resume.path)
    if stacked_replay_gate is not None and (resume is not None or args.policy_warm_start_checkpoint):
        raise ValueError('stacked replay gate requires fresh Adam at RL step0')
    if start_step >= args.opt_steps:
        raise ValueError(
            f"resume step {start_step} must be below opt-steps {args.opt_steps}")
    if resume is not None:
        args.head_checkpoint = str(resume.head)
        args.query_checkpoint = str(resume.query)
        if resume.qwen is not None:
            args.slow_model = str(resume.qwen)
    policy_warm_start = (resolve_policy_warm_start(
        args.policy_warm_start_checkpoint, output_dir=output)
        if args.policy_warm_start_checkpoint else None)
    if policy_warm_start is not None:
        query_inputs = {policy_warm_start.query,
                        policy_warm_start.query / "dualvln_query_embeddings.pt"}
        if (Path(args.head_checkpoint).resolve() != policy_warm_start.head
                or Path(args.query_checkpoint).resolve() not in query_inputs
                or Path(args.slow_model).resolve() != policy_warm_start.qwen):
            raise ValueError("policy warm-start Qwen/query/FM arguments must all match the source checkpoint")
        args.head_checkpoint = str(policy_warm_start.head)
        args.query_checkpoint = str(policy_warm_start.query)
        args.slow_model = str(policy_warm_start.qwen)
        tool_critic = policy_warm_start.critic
    else:
        tool_critic = _load_tool_credit_critic(args, resume)

    if args.joint_fixture_dir and args.joint_fixture_step not in (0, start_step + 1):
        raise ValueError("bounded fixture export is restricted to this run's first update")
    required = [args.repo, args.slow_model, args.fast_backbone,
                args.head_checkpoint, _query_file(args.query_checkpoint)]
    for path in required:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    cases = (load_case_manifest(args.case_manifest) if args.case_manifest else
             load_smoke_cases(args.smoke_cases, args.dataset_split))
    eval_cases = load_case_manifest(args.eval_case_manifest) if args.eval_case_manifest else []
    eval_benches = [value.strip() for value in args.eval_bench_urls.split(",") if value.strip()]
    if args.eval_interval < 0 or args.eval_repeats < 1 or args.eval_max_tool_calls < 0:
        raise ValueError("evaluation interval/budget must be nonnegative and repeats positive")
    if bool(eval_cases) != bool(args.eval_interval) or bool(eval_cases) != bool(eval_benches):
        raise ValueError("online evaluation requires eval-case-manifest, eval-bench-urls and eval-interval")
    if eval_cases:
        if not args.fast_policy_workers:
            raise ValueError("online evaluation requires shared fast-policy-workers for CPU eval actors")
        identities = {(str(row["scene"]), str(row["episode_id"])) for row in cases}
        overlap = identities & {(str(row["scene"]), str(row["episode_id"])) for row in eval_cases}
        if overlap:
            raise ValueError(f"training and heldout cases overlap: {sorted(overlap)[:5]}")
        train_benches = {value.strip() for value in args.bench_urls.split(",") if value.strip()}
        if len(set(eval_benches)) != len(eval_benches) or train_benches & set(eval_benches):
            raise ValueError("evaluation bench URLs must be unique and separate from training")
    if args.keep_last_checkpoints < 0:
        raise ValueError("keep-last-checkpoints must be nonnegative")
    if args.checkpoint_interval <= 0:
        raise ValueError("checkpoint-interval must be positive")
    if args.case_shuffle_seed is not None:
        random.Random(args.case_shuffle_seed).shuffle(cases)
    if len(cases) < args.num_groups_per_step:
        raise ValueError(
            f"need at least {args.num_groups_per_step} distinct cases, "
            f"but smoke-cases loaded {len(cases)}")
    if args.case_sampling == "stratified_per_step":
        from action_head_rl.three_source_cases import sample_step_cases
        if args.case_source_groups is None:
            raise ValueError("stratified sampling requires explicit case-source-groups")
        sample_step_cases(cases, step=1, count=args.num_groups_per_step,
            seed=args.case_sampling_seed, source_counts=args.case_source_groups)
    elif args.case_source_groups is not None:
        raise ValueError("case-source-groups requires stratified_per_step sampling")
    from sage.physical_navigation import training_geometry_contract
    geometry_contract = training_geometry_contract(cases)
    contract = {
        **({"geometry_contract": geometry_contract} if geometry_contract else {}),
        "schema": "action_head_ray_grpo_contract",
        "train_scope": scope.value,
        "router_replay": "R3",
        "capture_full_routes": True,
        "ray_job": True,
        "vllm_query_tail": True,
        "smoke_case_count": len(cases),
        "dataset_split": args.dataset_split,
        "case_sampling": args.case_sampling,
        "case_sampling_seed": args.case_sampling_seed,
        **({"case_source_groups": args.case_source_groups} if args.case_source_groups else {}),
        **({"case_manifest": str(Path(args.case_manifest).resolve())} if args.case_manifest else {}),
        **({"online_evaluation": {
            "case_manifest": str(Path(args.eval_case_manifest).resolve()),
            "case_order_sha256": hashlib.sha256(json.dumps(eval_cases, sort_keys=True).encode()).hexdigest(),
            "case_count": len(eval_cases), "interval": args.eval_interval,
            "repeats": args.eval_repeats, "seed": args.eval_seed,
            "max_tool_calls": args.eval_max_tool_calls or args.max_tool_calls,
            "step_zero": True, "final_step": True,
            "sampling": "same_stochastic_joint_policy_as_training",
            "policy_components": ["qwen", "dualvln_query", "action_head"],
            "heldout_bench_urls": eval_benches,
        }} if eval_cases else {}),
        "case_order_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
        "harness_version": args.harness_version,
        "multimodal_replay_revision": FM_REPLAY_INPUT_REVISION,
        "linear_attention_padding_revision": PADDING_MASK_REVISION if scope is TrainScope.FULL else "not_installed",
        **({"text_training_layout": "whole_trajectory_v1",
            "context_protocol": protocol_module("v8").protocol_identity()}
           if args.harness_version == "v8" else {}),
        "text_single_token_boundary_revision": "single_sampled_token_from_last_prompt_v1",
        **({"v6_training": {
            "text_policy": args.text_policy, "text_objective": "on_policy_gpg" if args.text_policy else None,
            "text_loss_coef": args.text_loss_coef if args.text_policy else 0.0,
            "text_temperature": args.text_temperature,
            "success_contract": args.success_contract, "process_reward_cap": args.process_reward_cap,
            "text_normalization": ("per_decision_episode_mean_whole_trajectory"
                if args.harness_version == "v8" else "one_record_per_decision_episode_mean"),
            "text_final_token": "unforwarded_final_token_excluded",
            "sampling": {"top_p": 1.0, "top_k": -1, "penalties": "disabled"},
        }} if args.harness_version in ("v6", "v8") else {}),
        "max_tool_calls": args.max_tool_calls,
        "max_completion_tokens": rollout_max_completion_tokens,
        "tool_budget_semantics": (
            "one_slow_decision_one_tool_call" if unified_tool_budget
            else "legacy_multi_call_turns"),
        "sde_steps": args.sde_steps,
        "sde_noise": args.sde_noise,
        "num_groups_per_step": args.num_groups_per_step,
        "group_size": args.group_size,
        "episodes_per_step": args.num_groups_per_step * args.group_size,
        "rollout_semantics": "independent_full_episode",
        "group_case_sharing": "same_case",
        "group_episode_sharing": "none",
        "slow_system_calls": "per_episode_multi_turn",
        "fast_system_execution": "per_tool_call",
        "fast_step_limit": rollout_fast_step_limit,
        "terminal_success_owner": (
            ("designated_supported_target_at_episode_end" if geometry_contract and geometry_contract["mode"] == "physical" else
             "official_target_bbox_at_episode_end")
            if args.success_contract == "legacy_object_arrival"
            else args.success_contract),
        "learner_world_size": learner_world,
        "fast_causal_attention": args.fast_causal_attention,
        "fm_suffix_vjp": args.fm_suffix_vjp,
        "fm_core_suffix_vjp": args.fm_core_suffix_vjp,
        "dual_shared_recompute": args.dual_shared_recompute,
        "full_attention_checkpoint_layers_off": args.full_attention_checkpoint_layers_off,
        "optimizer_cpu_during_backward": args.optimizer_cpu_during_backward,
        "learner_gpus_layout": learner_layout,
        "fsdp_size": args.fsdp_size,
        "fsdp_forward_prefetch": args.fsdp_forward_prefetch,
        "prepare_streamed_wave": args.prepare_streamed_wave,
        "ray_worker_gpus": total_gpus,
        "rollout_actor_gpus": rollout_reserved_gpus,
        "rollout_actor_count": args.rollout_actors,
        "rollout_gpu_fraction": args.rollout_gpu_fraction,
        "rollout_reserved_gpus": rollout_reserved_gpus,
        "episode_worker_device": (
            "cpu" if args.fast_policy_workers else "rollout_actor_gpu"),
        "fast_policy_workers": args.fast_policy_workers,
        "learner_cpus_per_rank": args.learner_cpus_per_rank,
        "text_before_fm": args.text_before_fm,
        "text_policy_training": os.environ.get('ACTION_HEAD_TRAIN_TEXT_POLICY', '1') != '0',
        "action_head_training": os.environ.get('ACTION_HEAD_TRAIN_ACTION_HEAD', '1') != '0',
        "text_backend": args.text_backend,
        "learner_collectives_only": args.learner_collectives_only,
        "megatron_parallelism": {key: getattr(args, 'megatron_' + key)
                                  for key in ('tp', 'pp', 'cp', 'ep', 'etp')},
        "fast_policy_device": args.fast_policy_device,
        "fast_policy_cpus": args.fast_policy_cpus,
        "fast_policy_max_batch_size": args.fast_policy_max_batch_size,
        "fast_policy_batch_wait_ms": args.fast_policy_batch_wait_ms,
        "fast_policy_weight_sync": (
            "checkpoint_reload" if args.fast_policy_workers else "per_actor_reload"),
        "rollout_groups_in_flight": args.rollout_groups_in_flight,
        "rollout_scheduler": args.rollout_scheduler,
        "megatron_dynamic_bsz": args.megatron_dynamic_bsz,
        "megatron_cpu_prefetch": args.megatron_cpu_prefetch,
        "fm_prefix_reuse": args.fm_prefix_reuse,
        "megatron_max_token_len_per_gpu": args.megatron_max_token_len_per_gpu,
        "trajectory_transfer_queue": args.trajectory_transfer_queue,
        "cpu_group_preparation": args.cpu_group_preparation,
        "overlap_cpu_group_preparation": args.overlap_cpu_group_preparation,
        "cpu_group_preparation_tasks_per_node": args.cpu_group_preparation_tasks_per_node,
        "cpu_group_preparation_memory_reserve_fraction": .25,
        'group_partial_rollout': (dict(warmup_batches=1, max_off_policy_threshold=4,
            stale_strategy='drop', correction='token_tis', correction_threshold=2.)
            if args.group_partial_rollout else None),
        "rollout_background_prefetch": args.rollout_background_prefetch,
        "v8_prefix_cache": args.v8_prefix_cache,
        "v8_cuda_graphs": args.v8_cuda_graphs,
        "fast_policy_pipeline": args.fast_policy_pipeline,
        "bulk_route_capture": args.bulk_route_capture,
        "r3_capture": r3_capture,
        "vllm_tensor_parallel_size": args.vllm_tp_size,
        "vllm_replicas": args.vllm_replicas,
        "colocated_vllm_replicas": args.colocated_vllm_replicas,
        "colocated_vllm_mode": (
            "fsdp_cpu_offload_vllm_sleep_level1"
            if args.colocated_vllm_replicas else "disabled"),
        "rollout_vllm_replicas_total": (
            args.vllm_replicas + args.colocated_vllm_replicas),
        "vllm_gpus": vllm_reserved_gpus,
        "vllm_colocated_gpus": args.colocated_vllm_replicas * args.vllm_tp_size,
        "vllm_max_num_seqs": args.vllm_max_num_seqs,
        "vllm_max_num_batched_tokens": args.vllm_max_num_batched_tokens,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "unallocated_gpus": total_gpus - requested_gpus,
        "gpu_budget_validation": ("static_16_per_node_no_visible_gpu" if gpu_budget_is_static
                                  else "local_visible_devices" if args.ray_address == "local"
                                  else "dlc_16_per_node"),
        "learner_microbatch_per_rank": 1,
        "learner_parameter_dtype": args.learner_parameter_dtype,
        "rollout_transport_dtype": args.rollout_transport_dtype,
        "bf16_gather_before_export": args.bf16_gather_before_export,
        "skip_zero_advantage_records": args.skip_zero_advantage_records,
        "compact_zero_advantage_waves": args.compact_zero_advantage_waves,
        "update_replay_capture": {"enabled": bool(args.update_replay_dir),
                                  "directory": args.update_replay_dir, "updates": 1},
        "joint_fixture_capture": {"enabled": bool(args.joint_fixture_dir),
                                  "directory": args.joint_fixture_dir,
                                  "count": args.joint_fixture_count,
                                  "step": args.joint_fixture_step or start_step + 1},
        "joint_text_fm_forward": args.joint_text_fm_forward,
        "joint_text_fm_forward_validation": "experimental_cpu_only_gpu_parity_pending",
        "optimizer_steps_per_logical_batch": 1,
        "old_logprob_mode": args.old_logprob_mode,
        "gradient_checkpointing": args.gradient_checkpointing,
        "option_condition_reuse": args.option_condition_reuse,
        "piold_option_reuse": args.piold_option_reuse,
        "frozen_weight_residency": args.frozen_weight_residency,
        "defer_query_capture": args.defer_query_capture,
        "full_weight_residency_gib": args.full_weight_residency_gib,
        "stage_records_per_rpc": args.stage_records_per_rpc,
        "stream_groups_per_update": args.stream_groups_per_update,
        "token_balanced_sharding": args.token_balanced_sharding,
        "multi_condition_batch_size": args.multi_condition_batch_size,
        "max_condition_tokens": args.max_condition_tokens,
        "condition_token_bucket_size": args.condition_token_bucket_size,
        "fla_l2norm_dynamic_nb": fla_l2norm_dynamic_nb_enabled,
        "fla_unused_shape_dynamic": fla_unused_shape_dynamic_enabled,
        "triton_autotune_seed": triton_autotune_seed,
        "zero_copy_rollout_payload": args.zero_copy_rollout_payload,
        "direct_vllm_weight_sync": args.direct_vllm_weight_sync,
        "native_colocated_weight_sync": args.native_colocated_weight_sync,
        "weight_sync_bucket_mb": args.weight_sync_bucket_mb,
        "overlap_rollout_learner": args.overlap_rollout_learner,
        "defer_gradient_sync": args.defer_gradient_sync,
        "action_head_learning_rate": args.learning_rate,
        "optimizer_weight_decay": args.weight_decay,
        "vlm_learning_rate": (
            args.vlm_learning_rate if args.vlm_learning_rate > 0
            else args.learning_rate),
        "vlm_trainable": scope is TrainScope.FULL,
        "async_rollout": False,
        "learner_dispatch": ("megatron_text_native_nd_fm_rank_local"
            if args.text_backend == "megatron" else "dp_compute_rank_local"),
        "logical_batch_transport": "rank_sharded_record_streams",
        "reward": {
            "schema": "action-head-multiturn-credit-v1",
            "success_contract": (
                (geometry_contract["object_success_schema"] if geometry_contract else "sage-official-expanded-object-bbox-v1")
                if args.success_contract == "legacy_object_arrival"
                else args.success_contract),
            **({"process_reward_cap": args.process_reward_cap}
               if args.process_reward_cap is not None else {}),
            "terminal_success_reward": args.terminal_success_reward,
            "direction_match_reward": args.direction_match_reward,
            "collision_penalty": args.collision_penalty,
            "discount_gamma": args.discount_gamma,
            "normalization": (
                "same_case_trajectory_outcome_grpo"
                if episode_grpo else
                "same_case_same_option_depth_grpo"),
            **({
                "process_reward_assignment":
                    "discounted_process_sum_with_terminal_outcome",
                "advantage_broadcast":
                    "one_scalar_per_trajectory_all_action_chunks",
                "loss_aggregation": "trajectory_mean_chunk_mean",
            } if episode_grpo else {}),
        },
        "resume": {
            "mode": args.resume_mode,
            "source": str(resume.path) if resume is not None else "",
            "start_step": start_step,
            "exact_trainer_state": bool(resume and resume.exact),
            **({'partial_admission': partial_resume} if partial_resume else {}),
            **({"same_training_objective": not args.allow_tool_credit_fork,
                "critic_sha256": tool_critic.state_dict()["sha256"],
                "critic_fitted_through_step": tool_critic.fitted_through_step}
               if tool_critic is not None else {}),
        },
        "checkpoint_interval": args.checkpoint_interval,
    }
    if tool_critic is not None:
        contract["tool_credit"] = _tool_credit_contract(
            args.max_tool_calls, args.tool_credit_fit_updates, mode=args.tool_credit_mode)
        contract["reward"].update(
            normalization=credit_reward_normalization(args.tool_credit_mode),
            advantage_broadcast="per_call_shared_by_its_text_and_fm_chunks")
    if rule_credit_enabled:
        from action_head_rl.rule_credit import contract as rule_contract
        contract["tool_credit"] = rule_contract(selective_backward=True,
                                               version=args.rule_credit_version)
        contract["reward"].update(normalization="call_depth_loo_no_std", advantage_broadcast="text_call_return_fm_call_advantage_minus_collision", rule_process_reward=True)
        if args.rule_credit_version == 'v3_terminal_local':
            contract["reward"].update(
                process_reward_assignment="return_to_go_including_last_call_terminal_reward",
                advantage_broadcast="text_call_return_fm_call_advantage_minus_collision")
        if contract['tool_credit'].get('sequence_rloo'):
            contract['reward'].update(normalization='sequence_loo_no_std',
                process_reward_assignment='unchanged_immediate_rewards_summed_once',
                advantage_broadcast='text_sequence_advantage_fm_sequence_advantage_minus_local_collision')
        contract["resume"]["same_training_objective"] = not args.allow_rule_credit_transition
    if policy_warm_start is not None:
        validate_policy_warm_start_contract(policy_warm_start, contract)
        contract["policy_warm_start"] = {
            **policy_warm_start.lineage,
            "source_joint_text_fm_forward": bool(
                policy_warm_start.source_contract.get("joint_text_fm_forward", False)),
            "target_joint_text_fm_forward": bool(args.joint_text_fm_forward),
            "source_multimodal_replay_revision": policy_warm_start.source_contract.get(
                "multimodal_replay_revision", "legacy_unknown"),
            "target_multimodal_replay_revision": FM_REPLAY_INPUT_REVISION,
        }
    elif resume is not None:
        # Carry the experiment origin forward, without rebasing its critic again.
        source_contract = json.loads((resume.path.parent / "contract.json").read_text())
        source_revision = source_contract.get("multimodal_replay_revision", "legacy_unknown")
        contract["resume"].update(
            source_multimodal_replay_revision=source_revision,
            multimodal_replay_revision_changed=source_revision != FM_REPLAY_INPUT_REVISION,
            source_linear_attention_padding_revision=source_contract.get(
                "linear_attention_padding_revision", "legacy_batch_gt1_mask"),
            linear_attention_padding_revision_changed=source_contract.get(
                "linear_attention_padding_revision", "legacy_batch_gt1_mask") != contract["linear_attention_padding_revision"])
        if "policy_warm_start" in source_contract:
            contract["policy_warm_start"] = source_contract["policy_warm_start"]
        if args.allow_rule_credit_transition:
            contract["rule_credit_transition"] = {
                "source_checkpoint": str(resume.path),
                "source_contract_sha256": hashlib.sha256(
                    (resume.path.parent / "contract.json").read_bytes()).hexdigest(),
                "source_schema": source_contract.get("tool_credit", {}).get("schema"),
                "target_schema": contract["tool_credit"]["schema"],
                "restored_model_optimizer_scheduler_rng": bool(resume.exact),
                "preserved_global_step": start_step,
                "allowed_contract_changes": ["tool_credit"],
            }
        elif "rule_credit_transition" in source_contract:
            contract["rule_credit_transition"] = source_contract["rule_credit_transition"]
    if not unified_tool_budget:
        contract.update(
            max_turns=args.max_turns,
            fm_chunks_per_episode=args.fm_chunks_per_episode)
    if resume is not None:
        topology_audit = validate_resume_contract(resume, contract,
                                 allow_tool_credit_fork=args.allow_tool_credit_fork,
                                 allow_rule_credit_transition=args.allow_rule_credit_transition,
                                 allow_rollout_topology_change=args.allow_rollout_topology_change)
        if topology_audit is not None:
            contract['resume']['rollout_topology_change'] = topology_audit
    print(json.dumps(contract, ensure_ascii=False, indent=2), flush=True)
    if args.validate_only:
        return

    if output.exists():
        if resume is None or resume.path.parent.resolve() != output:
            raise FileExistsError(output)
        later = [path for path in output.glob("global_step_*")
                 if path.is_dir() and int(path.name.rsplit("_", 1)[-1]) > start_step]
        if later and args.resume_mode != "auto":
            raise ValueError(
                f"output contains checkpoints newer than resume step {start_step}: {later}")
    else:
        output.mkdir(parents=True)
    contract_path = output / "contract.json"
    if not contract_path.exists():
        contract_path.write_text(
            json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    if policy_warm_start is not None:
        # Audit C'0 only. Exact resumes load their own committed checkpoint critic.
        tool_critic.save(output / "initial_tool_credit_critic.json")
    with (output / "resume_history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({**contract["resume"], "time": time.time(),
                                 "r3_capture": r3_capture},
                                ensure_ascii=False) + "\n")
    stable_query = output / "current_query.pt"
    shutil.copy2(_query_file(args.query_checkpoint), stable_query)

    os.environ.update({
        "ACTION_HEAD_HEAD_CHECKPOINT": str(Path(args.head_checkpoint).resolve()),
        "ACTION_HEAD_RULE_CREDIT": "1" if rule_credit_enabled else "0",
        "ACTION_HEAD_FINAL_CONTRACT": ("three-source-boundary-stop-v1"
            if args.rule_credit_version == "v4_boundary_stop_spl" else ""),
        "ACTION_HEAD_QUERY_CHECKPOINT": str(_query_file(args.query_checkpoint).resolve()),
        "ACTION_HEAD_TRAIN_SCOPE": scope.value,
        "ACTION_HEAD_ROUTER_REPLAY": "R3",
        "ACTION_HEAD_CAPTURE_FULL_ROUTING": "1",
    })
    runtime_env = {"env_vars": {
        key: value for key, value in {
            **{key: os.environ.get(key, "") for key in (
                "ACTION_HEAD_RAY_CACHE_ROOT", "XDG_CACHE_HOME", "HF_HOME",
                "ACTION_HEAD_MEGATRON_ENV", "CUDA_DEVICE_MAX_CONNECTIONS",
                "ACTION_HEAD_TRAIN_ACTION_HEAD", "ACTION_HEAD_TRAIN_TEXT_POLICY",
                "ACTION_HEAD_SINGLE_FORWARD_AUDIT_DIR", "ACTION_HEAD_PIOLD_AUDIT_DIR",
                "ACTION_HEAD_RULE_SHARED_AUDIT_DIR", "ACTION_HEAD_LEARNER_PROGRESS_DIR",
                "ACTION_HEAD_RULE_DISTANCE_COEFFICIENT", "ACTION_HEAD_RULE_INFORMATION_COEFFICIENT",
                "ACTION_HEAD_RULE_TERMINAL_SUCCESS_REWARD", "ACTION_HEAD_SEQUENCE_RLOO",
                "ACTION_HEAD_RULE_FIRST_ARRIVAL_REWARD",
                "ACTION_HEAD_QUEUE_PAYLOAD_CODEC", "ACTION_HEAD_QUEUE_MMAP_DIR",
                "ACTION_HEAD_BACKWARD_REPLAY_DIR",
                "ACTION_HEAD_CAUSAL_SDPA_UNPADDING",
                "ACTION_HEAD_STACKED_RULE_REPLAY_DIR",
                "ACTION_HEAD_COMPILE_CACHE_ROOT",
                "VLLM_RPC_BASE_PATH",
                "ACTION_HEAD_VLLM_WEIGHT_SYNC",
                "ACTION_HEAD_PROCESS_COMPILE_CACHE_ROOT",
                "ACTION_HEAD_PACKING_AUDIT",
                "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
                "PYTORCH_KERNEL_CACHE_PATH", "TORCH_EXTENSIONS_DIR", "TMPDIR",
                "ACTION_HEAD_REQUIRE_QWEN_FAST_PATH")},
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "ACTION_HEAD_HEAD_CHECKPOINT": os.environ["ACTION_HEAD_HEAD_CHECKPOINT"],
            "ACTION_HEAD_UPDATE_REPLAY_DIR": args.update_replay_dir,
            "ACTION_HEAD_RULE_CREDIT": "1" if rule_credit_enabled else "0",
        "ACTION_HEAD_FINAL_CONTRACT": ("three-source-boundary-stop-v1"
            if args.rule_credit_version == "v4_boundary_stop_spl" else ""),
            "ACTION_HEAD_QUERY_CHECKPOINT": os.environ["ACTION_HEAD_QUERY_CHECKPOINT"],
            "ACTION_HEAD_TRAIN_SCOPE": scope.value,
            "ACTION_HEAD_TEXT_POLICY": "1" if args.text_policy else "0",
            "ACTION_HEAD_JOINT_TEXT_FM_FORWARD": "1" if args.joint_text_fm_forward else "0",
            "ACTION_HEAD_COMPACT_ZERO_ADVANTAGE_WAVES": "1" if args.compact_zero_advantage_waves else "0",
            "ACTION_HEAD_ROUTER_REPLAY": "R3",
            "ACTION_HEAD_CAPTURE_FULL_ROUTING": "1",
            # Full owns this experimental patch.  Force zero for every other
            # scope even if the submitter inherited an unrelated shell value.
            "ACTION_HEAD_FLA_L2NORM_DYNAMIC_NB": (
                "1" if fla_l2norm_dynamic_nb_enabled else "0"),
            "ACTION_HEAD_FLA_UNUSED_SHAPE_DYNAMIC": (
                "1" if fla_unused_shape_dynamic_enabled else "0"),
            "ACTION_HEAD_TRITON_AUTOTUNE_SEED": triton_autotune_seed,
            "ACTION_HEAD_R3_CAPTURE_MODE": r3_capture["mode"],
            "ACTION_HEAD_R3_AUDIT_DIR": r3_capture["audit_dir"],
            "ACTION_HEAD_R3_AUDIT_SPOOL": "1",
            "ACTION_HEAD_DEFER_QUERY_CAPTURE": (
                "1" if args.defer_query_capture else "0"),
            "ACTION_HEAD_OLD_LOGPROB_MODE": args.old_logprob_mode,
            "ACTION_HEAD_OPTION_CONDITION_REUSE": (
                "1" if args.option_condition_reuse else "0"),
            "ACTION_HEAD_FM_PREFIX_REUSE": str(int(args.fm_prefix_reuse)),
            "ACTION_HEAD_FROZEN_WEIGHT_RESIDENCY": (
                "1" if args.frozen_weight_residency else "0"),
            "ACTION_HEAD_FULL_WEIGHT_RESIDENCY_GIB": str(args.full_weight_residency_gib),
            "ACTION_HEAD_FAST_CAUSAL_ATTENTION": str(int(args.fast_causal_attention)),
            "ACTION_HEAD_FM_SUFFIX_VJP": str(int(args.fm_suffix_vjp)),
            "ACTION_HEAD_FM_CORE_SUFFIX_VJP": str(int(args.fm_core_suffix_vjp)),
            "ACTION_HEAD_DUAL_SHARED_RECOMPUTE": str(int(args.dual_shared_recompute)),
            "ACTION_HEAD_FULL_ATTENTION_CHECKPOINT_LAYERS_OFF": str(args.full_attention_checkpoint_layers_off),
            "ACTION_HEAD_OPTIMIZER_CPU_DURING_BACKWARD": str(int(args.optimizer_cpu_during_backward)),
            "ACTION_HEAD_VLM_LR": str(args.vlm_learning_rate),
            "ACTION_HEAD_MEGATRON_CONFIG": json.dumps({key: getattr(args, 'megatron_' + key)
                for key in ('tp', 'pp', 'cp', 'ep', 'etp', 'dynamic_bsz', 'max_token_len_per_gpu', 'cpu_prefetch')}),
            "ACTION_HEAD_DEFER_GRADIENT_SYNC": (
                "1" if args.defer_gradient_sync else "0"),
            "ACTION_HEAD_MULTI_CONDITION_BATCH_SIZE": str(
                args.multi_condition_batch_size),
            "ACTION_HEAD_MAX_CONDITION_TOKENS": str(args.max_condition_tokens),
        }.items() if value
    }}
    if args.ray_address == "local":
        local_options = {}
        if os.environ.get("ACTION_HEAD_LOCAL_RAY_TEMP_DIR"):
            local_options["_temp_dir"] = os.environ["ACTION_HEAD_LOCAL_RAY_TEMP_DIR"]
        # CPFS stores logs/artifacts but cannot bind Unix-domain sockets.
        # Restrict the node-local exception to transient IPC socket files.
        from ray._private.node import Node
        original_init_temp = Node._init_temp
        def initialize_local_ipc(node):
            original_init_temp(node)
            ipc_root = os.environ.get("ACTION_HEAD_LOCAL_RAY_IPC_DIR")
            if ipc_root:
                node._sockets_dir = str(Path(ipc_root) / node._session_name)
                Path(node._sockets_dir).mkdir(parents=True, exist_ok=True)
        try:
            Node._init_temp = initialize_local_ipc
            ray.init(runtime_env=runtime_env, include_dashboard=False, **local_options)
        finally:
            Node._init_temp = original_init_temp
    else:
        ray.init(address=args.ray_address, runtime_env=runtime_env)
    # Reserve the node-specific FSDP placement groups before global service
    # actors can fragment GPUs across Ray worker nodes.
    worker_config = learner_config(args)
    learner_cpus_per_rank = args.learner_cpus_per_rank
    if args.ray_address == "local":
        # The production nodes have enough host CPUs for VERL's default ten
        # CPUs per learner rank.  A 16-GPU DSW node exposes 128 CPUs, so a
        # 13-rank matched gate needs to leave room for vLLM, FastPolicy and
        # rollout actors.  This changes only Ray scheduling, never math.
        service_cpus = (args.vllm_replicas + args.colocated_vllm_replicas
                        + args.fast_policy_workers * args.fast_policy_cpus
                        + args.rollout_actors + len(eval_benches))
        available_cpus = int(ray.cluster_resources().get("CPU", os.cpu_count() or 1))
        learner_cpus_per_rank = min(
            learner_cpus_per_rank, max(1, (available_cpus - service_cpus) // learner_world))
        print(json.dumps({
            "local_learner_cpus_per_rank": learner_cpus_per_rank,
            "local_ray_cpus": available_cpus,
            "reserved_service_cpus": service_cpus,
        }), flush=True)
    pool = RayResourcePool(
        process_on_nodes=learner_layout,
        max_colocate_count=learner_cpus_per_rank)
    learner_class = ActionHeadGRPOWorker
    if args.text_backend == "megatron":
        from action_head_rl.megatron_worker import ActionHeadMegatronWorker
        learner_class = ActionHeadMegatronWorker
    if stacked_replay_gate is not None:
        from action_head_rl.packing_audit_worker import StackedRuleReplayAuditWorker
        learner_class = StackedRuleReplayAuditWorker
    if args.update_backend_audit_input:
        if resume is not None or policy_warm_start is not None:
            raise ValueError('initial-policy audit cannot load a later policy')
        from action_head_rl.joint_audit import read_json
        source = read_json(Path(args.update_backend_audit_input).parent / 'launch/submit_plan.json')
        if (source['config']['slow_model'] != args.slow_model
                or str((Path(source['initial_checkpoint']) / 'action_head').resolve()) != os.environ['ACTION_HEAD_HEAD_CHECKPOINT']
                or str(_query_file(str(Path(source['initial_checkpoint']) / 'dualvln_query')).resolve()) != os.environ['ACTION_HEAD_QUERY_CHECKPOINT']):
            raise ValueError('backend audit source initialization differs')
    learner = RayWorkerGroup(
        resource_pool=pool,
        ray_cls_with_init=RayClassWithInitArgs(
            ray.remote(learner_class), config=worker_config),
        # Actor environment is installed before torch/CUDA initialization.
        # FM replay has variable-length prefixes; expandable segments reduce
        # allocator fragmentation. Keep this local to learners, not vLLM.
        # Both aliases agree so a newer PyTorch alias cannot mask this setting.
        worker_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "PYTORCH_ALLOC_CONF": "expandable_segments:True"},
        device_name="cuda")
    if args.learner_collectives_only or args.text_backend == "megatron":
        # Complete this RPC on EVERY rank before any reset enters model/engine
        # initialization. Per-rank checks inside reset do not enforce that
        # driver-level phase boundary against early model allocation.
        audits = _get_worker_result(learner.audit_initial_collectives(str(output)))
        (output / "learner_collectives_summary.json").write_text(
            json.dumps(dict(ranks=audits, training_steps=0,
                            diagnostic_only=args.learner_collectives_only), indent=2) + "\n")
        print(json.dumps(dict(event="ray_learner_collectives_passed", ranks=len(audits),
                              training_steps=0)), flush=True)
        if args.learner_collectives_only:
            ray.shutdown()
            return
    learner.set_loss_fn(loss_fn=lambda **_: None)
    learner.reset()
    if args.update_backend_audit_input:
        reports = _get_worker_result(learner.audit_update_backends(
            args.update_backend_audit_input, str(output / 'backend_audit')))
        (output / 'backend_audit_summary.json').write_text(json.dumps(dict(
            ranks=reports, optimizer_updates=0,
            passed=all(r['comparison']['passed'] for r in reports)), indent=2) + '\n')
        ray.shutdown()
        if not all(r['comparison']['passed'] for r in reports):
            raise RuntimeError('same-input backend gradient comparison failed; inspect saved evidence')
        return
    megatron_placements = (_get_worker_result(learner.megatron_placement())
                          if args.text_backend == "megatron" else None)
    if os.environ.get("ACTION_HEAD_REQUIRE_QWEN_FAST_PATH") == "1":
        fast_path_audit = _get_worker_result(learner.audit_qwen_fast_path())
        print(json.dumps({"qwen_fast_path_audit": fast_path_audit},
                         ensure_ascii=False), flush=True)
    if resume is not None and resume.exact:
        loaded = _get_worker_result(learner.load_checkpoint(
            str(resume.trainer_state), del_local_after_load=False))
        if args.text_backend == 'megatron':
            if len(loaded) != learner_world or any(row['step'] != start_step for row in loaded):
                raise RuntimeError('incomplete Megatron/FM checkpoint load acknowledgments')
            (output / 'resume_loaded_ranks.json').write_text(json.dumps(loaded, indent=2) + '\n')
        print(json.dumps({"resume_loaded": str(resume.path),
                          "step": start_step, "exact": True}), flush=True)
    elif resume is not None:
        print(json.dumps({"resume_loaded": str(resume.path),
                          "step": start_step, "exact": False,
                          "warning": "optimizer/scheduler/RNG reset"}), flush=True)
    elif policy_warm_start is not None:
        print(json.dumps({"policy_warm_start_loaded": contract["policy_warm_start"],
                          "step": 0, "exact": False,
                          "actor_optimizer_scheduler_rng": "new"}), flush=True)

    learning_rate_audit = _get_worker_result(learner.audit_learning_rates(
        contract["action_head_learning_rate"], contract["vlm_learning_rate"], scope.value))
    with (output / "learning_rate_audit.jsonl").open("a") as handle:
        handle.write(json.dumps({"step": start_step, "resumed": bool(resume),
                                 "ranks": learning_rate_audit}, ensure_ascii=False) + "\n")

    if args.resume_audit:
        compare_label, resume_audit_preflight = resume_audit_compare_label(
            output, start_step, learner_world, resume is not None)
        print(json.dumps({"resume_audit_preflight": resume_audit_preflight},
                         ensure_ascii=False), flush=True)
        _get_worker_result(learner.audit_training_state(
            str(output), f"loaded_{start_step:06d}",
            compare_label))

    colocated_placements = []
    colocated_ranks = []
    initial_colocated_offload = []
    learner_colocation_offloaded = False
    if args.colocated_vllm_replicas:
        placements = _get_worker_result(learner.colocated_placement())
        colocated_placements = select_colocated_placements(
            placements, args.colocated_vllm_replicas, args.vllm_tp_size)
        colocated_ranks = [rank for item in colocated_placements
                           for rank in item["ranks"]]
        initial_colocated_offload = _get_worker_result(
            learner.offload_for_colocated_rollout(colocated_ranks))
        learner_colocation_offloaded = True

    query_cache = Path(os.environ['ACTION_HEAD_LOCAL_RUNTIME_ROOT']) / 'query_cache'
    service_cls = ray.remote(num_gpus=args.vllm_tp_size)(VllmService)
    static_services = [
        service_cls.remote(
            repo=args.repo, model=args.slow_model,
            query_checkpoint=str(stable_query),
            query_cache=str(query_cache / f"replica_{index:02d}"),
            log_file=str(output / f"logs/vllm_{index:02d}.log"),
            port=args.vllm_port + index * 10,
            continuous_context=args.harness_version == "v8",
            v8_prefix_cache=args.v8_prefix_cache,
            v8_cuda_graphs=args.v8_cuda_graphs,
            max_images=2 * args.max_tool_calls + 2 if args.harness_version == "v8" else 64,
            tensor_parallel_size=args.vllm_tp_size,
            capture_full_routes=True,
            bulk_route_capture=args.bulk_route_capture,
            max_num_seqs=args.vllm_max_num_seqs,
            max_num_batched_tokens=args.vllm_max_num_batched_tokens,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            enable_sleep_mode=args.direct_vllm_weight_sync,
            replica_rank=index,
            direct_weight_sync=args.direct_vllm_weight_sync)
        for index in range(args.vllm_replicas)
    ]
    colocated_services = []
    if colocated_placements:
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        colocated_cls = ray.remote(num_cpus=1, num_gpus=0)(VllmService)
        for offset, placement in enumerate(colocated_placements):
            index = args.vllm_replicas + offset
            colocated_services.append(colocated_cls.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=placement["node_id"], soft=False)).remote(
                repo=args.repo, model=args.slow_model,
                query_checkpoint=str(stable_query),
                query_cache=str(query_cache / f"replica_{index:02d}"),
                log_file=str(output / f"logs/vllm_{index:02d}_colocated.log"),
                port=args.vllm_port + index * 10,
                tensor_parallel_size=args.vllm_tp_size,
                cuda_visible_devices=placement["cuda_visible_device"],
                continuous_context=args.harness_version == "v8",
                v8_prefix_cache=args.v8_prefix_cache,
                v8_cuda_graphs=args.v8_cuda_graphs,
                max_images=2 * args.max_tool_calls + 2 if args.harness_version == "v8" else 64,
                capture_full_routes=True,
                bulk_route_capture=args.bulk_route_capture,
                max_num_seqs=args.vllm_max_num_seqs,
                max_num_batched_tokens=args.vllm_max_num_batched_tokens,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                enable_sleep_mode=True,
                replica_rank=index,
                native_colocated_weight_sync=args.native_colocated_weight_sync))
    services = static_services + colocated_services
    vllm_urls = ray.get([service.start.remote() for service in services])
    query_artifact_urls = ray.get(
        [service.artifact_url.remote() for service in services])
    native_weight_targets = None
    if args.native_colocated_weight_sync:
        from action_head_rl.native_colocated_weights import map_native_targets
        native_weight_targets = map_native_targets(colocated_placements,
            ray.get([service.native_weight_target.remote() for service in colocated_services]),
            learner_world)
    direct_weight_manager = None
    if args.direct_vllm_weight_sync:
        from verl.checkpoint_engine import CheckpointEngineManager
        from verl.workers.config import CheckpointEngineConfig

        _get_worker_result(learner.init_direct_weight_sync(
            args.weight_sync_bucket_mb))
        ray.get([service.init_direct_weight_sync.remote(
            args.weight_sync_bucket_mb) for service in services])
        direct_weight_manager = CheckpointEngineManager(
            config=CheckpointEngineConfig(
                backend="nccl",
                update_weights_bucket_megabytes=args.weight_sync_bucket_mb,
                engine_kwargs={"nccl": {
                    "group_name": "action_head_qwen_direct",
                    "rebuild_group": False,
                }}),
            trainer=learner,
            replicas=[_DirectVllmReplica(services)],
        )

    fast_policy_workers = []
    if args.fast_policy_workers:
        fast_policy_cls = ray.remote(
            num_gpus=int(args.fast_policy_device == "cuda"),
            num_cpus=args.fast_policy_cpus,
            max_concurrency=max(32, args.fast_policy_max_batch_size * 4),
        )(FastPolicyWorker)
        fast_options = [{} for _ in range(args.fast_policy_workers)]
        fast_devices = [None] * args.fast_policy_workers
        if args.fast_policy_device == 'colocated_cuda':
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
            from action_head_rl.rollout import colocated_fast_policy_runtime_env
            fast_placements = select_colocated_placements(
                [row for row in placements if int(row['rank']) in colocated_ranks],
                args.fast_policy_workers, 1)
            fast_options = [dict(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=row['node_id'], soft=False),
                runtime_env=colocated_fast_policy_runtime_env(row['cuda_visible_device']))
                for row in fast_placements]
            fast_devices = [row['cuda_visible_device'] for row in fast_placements]
        if args.fast_policy_device == "cpu" and colocated_placements:
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
            nodes = sorted({row["node_id"] for row in colocated_placements})
            fast_options = [dict(scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=nodes[i % len(nodes)], soft=False))
                for i in range(args.fast_policy_workers)]
        fast_policy_workers = [
            fast_policy_cls.options(**options).remote(
                checkpoint=args.head_checkpoint,
                backbone=args.fast_backbone,
                max_batch_size=args.fast_policy_max_batch_size,
                batch_wait_ms=args.fast_policy_batch_wait_ms,
                sde_steps=args.sde_steps,
                sde_noise=args.sde_noise,
                policy_version=start_step,
                pipeline_batches=args.fast_policy_pipeline,
                device="cpu" if args.fast_policy_device == "cpu" else "cuda:0",
                cpu_threads=args.fast_policy_cpus,
                cuda_visible_device=visible_device,
            )
            for options, visible_device in zip(fast_options, fast_devices, strict=True)
        ]
        rollout_cls = ray.remote(num_cpus=1)(ActionHeadRolloutWorker)
    else:
        rollout_cls = ray.remote(
            num_gpus=args.rollout_gpu_fraction)(ActionHeadRolloutWorker)
    benches = [value.strip() for value in args.bench_urls.split(",") if value.strip()]
    if len(benches) < args.rollout_actors:
        raise ValueError(
            f"rollout-actors={args.rollout_actors} requires at least that many "
            f"independent bench URLs, got {len(benches)}")
    if len(set(benches[:args.rollout_actors])) != args.rollout_actors:
        raise ValueError("each rollout actor must use a distinct stateful bench URL")
    def create_rollout_worker(index, url, *, evaluation=False):
        tool_budget = (args.eval_max_tool_calls or args.max_tool_calls) if evaluation else args.max_tool_calls
        return rollout_cls.remote(
            repo=args.repo, checkpoint=args.head_checkpoint,
            backbone=args.fast_backbone, slow_model=args.slow_model,
            query_cache=query_artifact_urls[index % len(query_artifact_urls)], bench_url=url,
            vllm_url=vllm_urls[index % len(vllm_urls)], train_scope=scope.value,
            harness_version=args.harness_version,
            text_policy=args.text_policy, text_temperature=args.text_temperature,
            record_behavior_logprobs=args.group_partial_rollout and not evaluation,
            success_contract=args.success_contract, process_reward_cap=args.process_reward_cap,
            seed=20260831 + index * 10000,
            max_turns=tool_budget if unified_tool_budget else rollout_max_turns,
            max_tool_calls=tool_budget,
            one_tool_call_per_turn=unified_tool_budget,
            max_completion_tokens=rollout_max_completion_tokens,
            fast_step_limit=rollout_fast_step_limit,
            sde_steps=args.sde_steps, sde_noise=args.sde_noise,
            terminal_success_reward=args.terminal_success_reward,
            direction_match_reward=args.direction_match_reward,
            collision_penalty=args.collision_penalty,
            discount_gamma=args.discount_gamma,
            fast_policy_worker=(
                fast_policy_workers[index % len(fast_policy_workers)]
                if fast_policy_workers else None),
            worker_id=index, policy_version=start_step,
            v8_prefix_cache=args.v8_prefix_cache,
            zero_copy_payload=args.zero_copy_rollout_payload)
    rollout_workers = [
        create_rollout_worker(index, url)
        for index, url in enumerate(benches[:args.rollout_actors])
    ]
    eval_workers = [create_rollout_worker(index + args.rollout_actors, url, evaluation=True)
                    for index, url in enumerate(eval_benches)]
    policy_consumers = rollout_workers + eval_workers
    if not rollout_workers:
        raise ValueError("bench-urls contains no URL")

    audits = _get_worker_result(learner.trainable_audit())
    print(json.dumps({"trainable_audit": audits}, ensure_ascii=False), flush=True)
    world = learner_world

    metrics_path = output / "metrics.jsonl"
    rollouts_path = output / "rollouts.jsonl"
    if resume is not None and resume.path.parent.resolve() == output:
        # A crash can happen after rollout/metrics append but before the atomic
        # latest-checkpoint tracker advances.  Make replay from the committed
        # step idempotent, like restoring a StatefulDataLoader position.
        truncate_jsonl_after_step(metrics_path, start_step)
        truncate_jsonl_after_step(rollouts_path, start_step)
        if eval_workers:
            for name in ("evaluation_metrics.jsonl", "evaluation_rollouts.jsonl"):
                truncate_jsonl_after_step(output / name, start_step - 1)
    # Multi-node policy consumers need a shared transport path.  It is kept
    # outside the resumable global_step namespace and overwritten/cleaned as
    # soon as the next policy has been loaded.  Durable checkpoints remain in
    # OUTPUT_DIR/global_step_* only at the requested cadence.
    policy_transport = output / ".policy_transport"
    if local_transport is None and policy_transport.exists():
        shutil.rmtree(policy_transport)
    fixture_source = None
    if args.joint_fixture_dir:
        from action_head_rl.joint_fixture import actor_source_binding
        try:
            fixture_source = actor_source_binding(
                (resume.path if resume is not None else
                 policy_warm_start.path if policy_warm_start is not None else None), qwen=args.slow_model,
                head=args.head_checkpoint, query=args.query_checkpoint)
        except Exception as exc:
            fixture_source = {"error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps({"fixture_source_audit": fixture_source}), flush=True)
    streamed_train_active = False
    learner_stream = None
    trajectory_queue_created = False
    partial_producer = None
    native_receivers = None
    try:
        if args.trajectory_transfer_queue:
            from action_head_rl.queue_records import initialize_record_queue
            initialize_record_queue(storage_units=32)
            trajectory_queue_created = True
        if args.group_partial_rollout:
            from action_head_rl.partial_groups import PartialGroupRollouts, PartialCaseGroups, iter_leased_rollout_groups
            def sample_partial_cases(block, policy_version):
                from action_head_rl.three_source_cases import sample_step_cases, case_identity
                if args.case_sampling in ('uniform_per_step', 'stratified_per_step'):
                    admitted = sample_step_cases(cases, step=block, count=args.num_groups_per_step,
                        seed=args.case_sampling_seed, source_counts=args.case_source_groups)
                else:
                    start = (block - 1) * args.num_groups_per_step
                    admitted = [cases[(start + index) % len(cases)] for index in range(args.num_groups_per_step)]
                with (output / 'partial_admissions.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(dict(block=block, admission_policy_version=policy_version,
                        mode=args.case_sampling, identities=[case_identity(row) for row in admitted])) + '\n')
                return admitted
            preparation = None
            if args.overlap_cpu_group_preparation:
                from action_head_rl.cpu_group_preparation import GroupPreparationPipeline
                def preparation_log(row):
                    with (output / 'cpu_group_pipeline.jsonl').open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(row, allow_nan=False) + '\n')
                preparation = GroupPreparationPipeline(options=dict(max_tool_calls=args.max_tool_calls,
                    reward_version=args.rule_credit_version, text_loss_coef=args.text_loss_coef),
                    log=preparation_log, concurrency=args.nnodes * args.cpu_group_preparation_tasks_per_node,
                    tasks_per_node=args.cpu_group_preparation_tasks_per_node,
                    # Sampling is followed by 128 concurrent learner pulls.
                    # The former 15% reserve allowed staging to cross Ray's
                    # 95% threshold before any GPU backward could start.
                    available_fraction=.25)
            partial_producer = PartialGroupRollouts(rollout_workers, group_size=args.group_size,
                batch_groups=args.num_groups_per_step,
                next_groups=PartialCaseGroups(sample_partial_cases, independent_episode_specs,
                    group_size=args.group_size,
                    start_block=partial_resume['last_admitted_block'] if partial_resume else 0),
                num_warmup_batches=1, max_off_policy_threshold=4, preparation=preparation)
        if local_transport is not None:
            local_transport.create()
            print(json.dumps({"local_policy_transport": local_transport.owner}), flush=True)
        if eval_workers:
            run_online_evaluation(eval_workers, eval_cases, step=start_step,
                                  repeats=args.eval_repeats, seed=args.eval_seed, output=output)
        for step in range(start_step + 1, args.opt_steps + 1):
            step_t0 = time.perf_counter()
            tool_credit_ledgers = []
            tool_credit_diagnostics = []
            tool_credit_summary = None
            joint_forward_audit = None
            if tool_critic is not None and tool_critic.fitted_through_step != step - 1:
                raise ValueError("each batch must use the previous completed batch's frozen critic")
            colocated_sleep_seconds = 0.0
            colocated_restore_seconds = 0.0
            colocated_offload_seconds = 0.0
            colocated_wake_reload_seconds = 0.0
            direct_weight_sync_seconds = 0.0
            rollout_t0 = time.perf_counter()
            partial_batch, partial_metrics = None, None
            if partial_producer is not None:
                partial_batch, partial_metrics = partial_producer.sample(step - 1)
                step_cases = partial_producer.batch_cases(partial_batch)
            elif args.case_sampling in ("uniform_per_step", "stratified_per_step"):
                from action_head_rl.three_source_cases import sample_step_cases, case_identity
                step_cases = sample_step_cases(cases, step=step,
                    count=args.num_groups_per_step, seed=args.case_sampling_seed,
                    source_counts=args.case_source_groups)
                with (output / "sampled_cases.jsonl").open("a", encoding="utf-8") as case_log:
                    case_log.write(json.dumps(dict(step=step, seed=args.case_sampling_seed,
                        mode=args.case_sampling, source_groups=args.case_source_groups,
                        identities=[case_identity(row) for row in step_cases])) + "\n")
            else:
                case_start = (step - 1) * args.num_groups_per_step
                step_cases = [cases[(case_start + index) % len(cases)]
                              for index in range(args.num_groups_per_step)]
            episode_specs = independent_episode_specs(
                step_cases, step=step, group_size=args.group_size)
            records = []
            prepared_queue_leases = []
            reward_groups = []
            episode_seconds_list = []
            slow_qwen_seconds_list = []
            rollout_queue_wait_seconds_list = []
            actor_max_memory_gb = 0.0
            episode_count = 0
            slow_system_calls = 0
            reward_components_list = []
            learner_stream = None
            text_decision_count = 0
            action_chunk_count = 0
            training_record_count = 0
            pending_stream_records = []
            pending_stream_groups = 0
            streamed_zero_advantage_reports = []
            streamed_joint_reports = []
            if args.overlap_rollout_learner:
                expected_waves = math.ceil(
                    args.num_groups_per_step / args.stream_groups_per_update)
                prepare_wave = lambda rows: _enqueue_streamed_group(
                        learner, rows, world, args.stage_records_per_rpc,
                        args.token_balanced_sharding, args.multi_condition_batch_size,
                        args.max_condition_tokens, args.condition_token_bucket_size,
                        skip_zero_advantage_records=args.skip_zero_advantage_records,
                        compact_zero_advantage_waves=args.compact_zero_advantage_waves,
                        zero_advantage_reports=streamed_zero_advantage_reports,
                        joint_text_fm_forward=args.joint_text_fm_forward,
                        joint_reports=streamed_joint_reports, prepare_only=True,
                        stacked_replay_gate=stacked_replay_gate)
                learner_stream = BoundedLearnerStream(
                    world=world, expected_waves=expected_waves,
                    prepare=prepare_wave if args.prepare_streamed_wave else None,
                    submit=(lambda prepared: prepared()) if args.prepare_streamed_wave
                           else (lambda rows: prepare_wave(rows)()),
                    resolve=lambda refs: _get_worker_result(
                        refs, timeout=3600, phase=f"step {step} streamed backward"))
                # Mark active before the RPC so a partially failed begin is
                # still covered by abort; no optimizer update happens here.
                streamed_train_active = True
                _get_worker_result(learner.begin_streamed_train(
                    expected_accumulations=expected_waves), timeout=300,
                    phase=f"step {step} begin streamed train")
            # Preserve the G x K episode specs and ordered group consumption,
            # while assigning each newly free actor another complete episode.
            # The admission bound includes groups waiting behind a straggler.
            def submit_training_rollout(actor, spec):
                if args.joint_fixture_dir:
                    from action_head_rl.joint_fixture import fixture_capture_spec
                    capture = fixture_capture_spec(
                        args.joint_fixture_dir, count=args.joint_fixture_count,
                        capture_step=args.joint_fixture_step, first_step=start_step + 1,
                        step=step, rollout_index=spec[0] * args.group_size + spec[1])
                    if capture is not None:
                        capture['actor_source'] = fixture_source
                        return actor.rollout.remote(spec[2], spec[3], fixture_capture=capture)
                return actor.rollout.remote(spec[2], spec[3])

            grouped_rollouts = iter_grouped_rollouts(
                episode_specs, rollout_workers, group_size=args.group_size,
                max_in_flight_groups=args.rollout_groups_in_flight,
                submit=submit_training_rollout,
                wait_one=lambda refs: ray.wait(list(refs), num_returns=1)[0][0],
                resolve=ray.get,
                scheduler=getattr(args, "rollout_scheduler", "completion"),
            )
            if partial_batch is not None:
                grouped_rollouts = iter_leased_rollout_groups(partial_batch, group_size=args.group_size)
                if args.cpu_group_preparation:
                    from action_head_rl.cpu_group_preparation import iter_prepared_groups
                    grouped_rollouts = iter_prepared_groups(partial_batch, group_size=args.group_size,
                        producer=partial_producer, step=step, max_tool_calls=args.max_tool_calls,
                        reward_version=args.rule_credit_version, text_loss_coef=args.text_loss_coef)
            rollout_prefetch = None
            if args.rollout_background_prefetch:
                rollout_prefetch = BackgroundGroupedRollouts(
                    episode_specs, rollout_workers, group_size=args.group_size,
                    max_in_flight_groups=args.rollout_groups_in_flight,
                    submit=submit_training_rollout,
                    wait_one=lambda refs: next(iter(ray.wait(
                        list(refs), num_returns=1, timeout=.1)[0]), None),
                    resolve=ray.get)
            rollout_context = rollout_prefetch or (
                closing(grouped_rollouts) if args.cpu_group_preparation else nullcontext(grouped_rollouts))
            with rollout_context as grouped_rollouts, rollouts_path.open("a", encoding="utf-8") as handle:
                for group_index, group_source, rollouts in grouped_rollouts:
                    if args.cpu_group_preparation:
                        prepared_queue_leases.append(group_source)
                    group_record_start = len(records)
                    require_valid_rollouts(rollouts, step=step)
                    rewards = torch.tensor(
                        [float(row["reward"]) for row in rollouts])
                    trajectory_advantages = (
                        group_trajectory_advantages(rollouts)
                        if episode_grpo else None)
                    option_advantages = (
                        None if episode_grpo else
                        group_option_advantages(rollouts))
                    reward_groups.append(rewards)
                    loo_baselines = (_tool_credit_group_baselines(rollouts, mode=args.tool_credit_mode)
                                     if tool_critic is not None else [None] * len(rollouts))
                    rule_group = None
                    if args.cpu_group_preparation:
                        rule_group = [row['_prepared_credit'] for row in rollouts]
                    elif rule_credit_enabled:
                        from action_head_rl.rule_credit import assign_group
                        try:
                            rule_group = assign_group(rollouts, step=step, max_tool_calls=args.max_tool_calls,
                                                      version=args.rule_credit_version)
                        except Exception as error:
                            # Scoring precedes the ordinary rollout log. Preserve
                            # native call evidence before Ray objects are released.
                            failure_dir = output / "rule_credit_failures"
                            failure_dir.mkdir(parents=True, exist_ok=True)
                            failure_path = failure_dir / f"step_{step:04d}_group_{group_index:04d}.json"
                            failure = {"step": step, "group_index": group_index,
                                       "error": f"{type(error).__name__}: {error}",
                                       "rollouts": [{key: row.get(key) for key in
                                           ("seed", "case", "reward", "result")}
                                           for row in rollouts]}
                            failure_path.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n")
                            print(json.dumps({"rule_credit_failure": str(failure_path),
                                              "error": failure["error"]}), flush=True)
                            raise
                    for sample_index, rollout in enumerate(rollouts):
                        rollout_index = group_index * args.group_size + sample_index
                        episode_record_start = len(records)
                        call_credit = None
                        if rule_group is not None:
                            call_credit = rule_group[sample_index]
                            tool_credit_diagnostics.append(call_credit)
                        if tool_critic is not None:
                            ledger, call_credit = _tool_credit_for_rollout(
                                rollout, tool_critic, step=step, max_tool_calls=args.max_tool_calls,
                                mode=args.tool_credit_mode, loo_baseline=loo_baselines[sample_index])
                            tool_credit_ledgers.append(ledger)
                            tool_credit_diagnostics.append(call_credit)
                        trajectory_advantage = (
                            float(trajectory_advantages[sample_index])
                            if trajectory_advantages is not None else None)
                        from action_head_rl.rollout_records import attach_episode_records
                        if args.cpu_group_preparation:
                            records.extend(rollout['_prepared_records'])
                        else:
                            records.extend(attach_episode_records(
                                rollout, step=step, rollout_index=rollout_index,
                                policy_version=step - 1, trajectory_advantage=trajectory_advantage,
                                option_advantages=(option_advantages[sample_index]
                                    if option_advantages is not None else None),
                                text_policy=args.text_policy, text_loss_coef=args.text_loss_coef,
                                check_fm_version=bool(args.fast_policy_workers),
                                allow_mixed_policy=args.group_partial_rollout))
                        if args.joint_text_fm_forward:
                            for record in records[episode_record_start:]:
                                record["joint_episode_id"] = f"step={step}/rollout={rollout_index}/policy={step-1}"
                        if call_credit is not None and not args.cpu_group_preparation:
                            if rule_credit_enabled:
                                from action_head_rl.rule_credit import apply_records
                                apply_records(records[episode_record_start:], call_credit)
                            else:
                                _apply_tool_credit_records(records[episode_record_start:], call_credit)
                        if args.harness_version == "v8" and not args.cpu_group_preparation:
                            from action_head_rl.trajectory_text import merge_trajectory_text
                            records[episode_record_start:] = merge_trajectory_text(records[episode_record_start:],
                                allow_mixed_policy=args.group_partial_rollout)
                        fixture_audit = rollout.get("fixture_audit")
                        if fixture_audit is not None:
                            from action_head_rl.joint_fixture import finalize_fixture_targets
                            fixture_audit = finalize_fixture_targets(
                                fixture_audit, records[episode_record_start:], step=step,
                                policy_version=step - 1, credit_mode=args.tool_credit_mode,
                                logical_denominator=args.num_groups_per_step * args.group_size,
                                critic_sha256=(call_credit.get("critic_sha256") if call_credit else None))
                            print(json.dumps({"step": step, "fixture_audit": fixture_audit}), flush=True)
                        interactions = [{
                            "option_id": record.get("option_id"),
                            "chunk_index": record.get("chunk_index"),
                            "env_reward": record.get("env_reward", 0.0),
                            "collision": bool(record.get("collision")),
                            "blocked": bool(record.get("blocked")),
                            "moved_m": record.get("moved_m", 0.0),
                            "advantage": record.get("advantage", 0.0),
                            **({'policy_version': record['policy_version'],
                                'update_policy_version': record['update_policy_version']}
                               if args.group_partial_rollout else {}),
                            "option_credit": record.get("option_credit", {}),
                            "gym_feedback": record.get("gym_feedback", {}),
                        } for record in rollout["chunks"]]
                        handle.write(json.dumps({
                            "step": step,
                            "group_index": group_index,
                            "sample_index": sample_index,
                            "seed": rollout.get("seed"),
                            "case": rollout["case"],
                            "reward": rollout["reward"],
                            "training_reward": rollout["reward"],
                            "reward_components": rollout["reward_components"],
                            "result": rollout["result"],
                            **({'partial_rollout': rollout['partial_rollout']}
                               if args.group_partial_rollout else {}),
                            "interactions": interactions,
                            **({"fixture_audit": fixture_audit} if fixture_audit is not None else {}),
                            **({"tool_credit": call_credit} if call_credit is not None else {}),
                            **({"high_level_decisions": len(rollout.get("decisions") or [])}
                               if args.text_policy else {}),
                        }, ensure_ascii=False) + "\n")
                        episode_seconds_list.append(float(
                            rollout.get("episode_seconds", 0.0) or 0.0))
                        slow_qwen_seconds_list.append(float(
                            rollout.get("slow_qwen_seconds", 0.0) or 0.0))
                        rollout_queue_wait_seconds_list.append(float(
                            rollout.get("rollout_queue_wait_seconds", 0.0) or 0.0))
                        slow_system_calls += len(
                            (rollout.get("result") or {}).get("turns") or [])
                        reward_components_list.append(
                            rollout["reward_components"])
                        actor_max_memory_gb = max(
                            actor_max_memory_gb, float(
                                rollout.get("actor_max_memory_gb", 0.0) or 0.0))
                        episode_count += 1
                    if args.overlap_rollout_learner:
                        group_records = records[group_record_start:]
                        text_decision_count += sum(is_text_record(row) for row in group_records)
                        training_record_count += len(group_records)
                        action_chunk_count = training_record_count - text_decision_count
                        pending_stream_records.extend(group_records)
                        pending_stream_groups += 1
                        # Retain only the pending completed groups. The old
                        # records list kept every tensor-rich episode for the
                        # whole step even after its backward had completed.
                        records.clear()
                        del group_records
                        if (pending_stream_groups >= args.stream_groups_per_update
                                or episode_count == len(episode_specs)):
                            learner_stream.submit(pending_stream_records)
                            pending_stream_records = []
                            pending_stream_groups = 0
                        # Loop locals otherwise retain the last episode's
                        # Plasma references across learner accumulation.
                        del rollout
                        record = None
                    del rollouts
                    if ((group_index + 1) % args.rollout_groups_in_flight == 0
                            or episode_count == len(episode_specs)):
                        gc.collect()
                        print(json.dumps({
                            "rollout_progress": episode_count,
                            "rollout_total": len(episode_specs),
                            "records": (training_record_count if args.overlap_rollout_learner else len(records)),
                            **_driver_memory_gb(),
                        }), flush=True)
            rollout_seconds = time.perf_counter() - rollout_t0
            fast_policy_offload_seconds = 0.0
            if args.fast_policy_device == 'colocated_cuda':
                transition_t0 = time.perf_counter()
                ray.get([worker.set_rollout_resident.remote(False) for worker in fast_policy_workers])
                fast_policy_offload_seconds = time.perf_counter() - transition_t0
            if colocated_services:
                transition_t0 = time.perf_counter()
                # Megatron always publishes a complete updated Qwen before
                # waking these paused replicas. Match reference vLLM>=0.18:
                # discard obsolete inference weights instead of retaining a
                # full CPU backup per replica throughout the learner phase.
                ray.get([service.sleep.remote(level=2 if args.text_backend == "megatron" else 1)
                         for service in colocated_services])
                colocated_sleep_seconds = time.perf_counter() - transition_t0
                transition_t0 = time.perf_counter()
                _get_worker_result(learner.restore_after_colocated_rollout(
                    colocated_ranks))
                learner_colocation_offloaded = False
                colocated_restore_seconds = time.perf_counter() - transition_t0
            fast_policy_stats = (
                ray.get([worker.stats.remote() for worker in fast_policy_workers])
                if fast_policy_workers else [])
            if not (training_record_count if args.overlap_rollout_learner else len(records)):
                raise RuntimeError("rollout produced no trainable policy records")
            zero_advantage_filter = None
            staging_seconds = None  # Streaming staging overlaps rollout/backward.
            if args.overlap_rollout_learner:
                tail_t0 = time.perf_counter()
                streamed_rank_metrics = learner_stream.complete(
                    expected_logical_valid=len(episode_specs) if args.text_policy else None)
                logical_valid = learner_stream.logical_valid
                rank_record_counts = learner_stream.rank_record_counts
                learner_rank_metrics = _get_worker_result(
                    learner.finish_streamed_train(logical_valid=logical_valid),
                    timeout=600, phase=f"step {step} optimizer finalization")
                validate_streamed_optimizer_result(
                    learner_rank_metrics, world=world, logical_valid=logical_valid)
                streamed_train_active = False
                updates = _first_result(learner_rank_metrics)
                if args.skip_zero_advantage_records:
                    zero_advantage_filter = {
                        "enabled": True, "scope": "streamed_waves",
                        "logical_denominator": logical_valid,
                        **{key: sum(report[key] for report in streamed_zero_advantage_reports)
                           for key in ("original_records", "retained_records", "skipped_records",
                                       "original_loss_mass", "retained_loss_mass")},
                        "waves": streamed_zero_advantage_reports,
                    }
                # Preserve the historical tail-only meaning of train_seconds.
                # Per-wave times below describe overlapped learner work.
                train_seconds = time.perf_counter() - tail_t0
                piold_seconds = 0.0
                piold_updates = []
                batches = []
            else:
                text_decision_count = sum(is_text_record(row) for row in records)
                action_chunk_count = len(records) - text_decision_count
                training_record_count = len(records)
                original_logical_valid = None
                if args.joint_text_fm_forward:
                    from action_head_rl.joint_policy import episode_text_mass
                    original_logical_valid = episode_text_mass(records)
                if args.skip_zero_advantage_records:
                    records, original_logical_valid, zero_advantage_filter = (
                        _zero_advantage_training_records(records))
                    print(json.dumps({"step": step,
                                      "zero_advantage_filter": zero_advantage_filter}), flush=True)
                batches = []
                piold_t0 = time.perf_counter()
                for start in range(0, len(records), world):
                    batch = records[start:start + world]
                    while len(batch) < world:
                        padding = dict(records[0])
                        padding["sample_valid"] = 0.0
                        padding["advantage"] = 0.0
                        batch.append(padding)
                    if args.old_logprob_mode == "recompute" and not args.piold_option_reuse:
                        proximal = _get_worker_result(
                            learner.compute_record_logprobs(batch))
                        if len(proximal) != world:
                            raise RuntimeError(
                                f"actor returned {len(proximal)} proximal logprobs "
                                f"for world={world}")
                        for record, old_logprob in zip(batch, proximal):
                            record["old_logprob"] = torch.tensor(
                                [float(old_logprob)], dtype=torch.float32)
                    batches.append(batch)
                try:
                    piold_seconds = time.perf_counter() - piold_t0
                except Exception:
                    piold_seconds = 0.0
                # Freeze pi_old for the complete rollout group before the first
                # optimizer update.  Recomputing it batch-by-batch after earlier
                # batches have stepped makes every ratio start at one and is not
                # PPO's fixed behavior-policy denominator.
                # The complete G x K rollout wave is one logical batch.  All
                # microbatches accumulate into one optimizer/scheduler step, so
                # changing world size or chunk count does not change update count.
                queued_staging_t0 = None
                queue_lease = None
                if args.trajectory_transfer_queue:
                    from action_head_rl.queue_records import publish_record_groups, record_locator, release_record_groups, stage_queued_records
                    queued_staging_t0 = time.perf_counter()
                    if not args.cpu_group_preparation:
                        queue_lease = publish_record_groups(records)
                if args.text_backend == "megatron":
                    from action_head_rl.megatron_joint import shard_joint_records
                    from functools import partial
                    text_scheduling = {}
                    if args.cpu_group_preparation:
                        from action_head_rl.cpu_group_preparation import shard_text_descriptors
                        text_scheduling['text_sharder'] = shard_text_descriptors
                    rank_records, logical_valid = shard_joint_records(records, megatron_placements,
                        fm_sharder=partial(shard_option_record_groups,
                            balance_by_token_flops=args.token_balanced_sharding,
                            condition_batch_size=args.multi_condition_batch_size,
                            max_padded_tokens=args.max_condition_tokens,
                            condition_token_bucket_size=args.condition_token_bucket_size,
                            trajectory_affinity=args.fm_prefix_reuse),
                        **text_scheduling)
                elif args.joint_text_fm_forward:
                    from action_head_rl.joint_policy import build_joint_policy_groups
                    from action_head_rl.joint_scheduling import shard_joint_policy_groups
                    joint_groups, pairing = build_joint_policy_groups(records)
                    rank_records, scheduling = shard_joint_policy_groups(
                        joint_groups, world, logical_valid=original_logical_valid,
                        balance_by_token_flops=args.token_balanced_sharding,
                        condition_batch_size=args.multi_condition_batch_size,
                        max_padded_tokens=args.max_condition_tokens)
                    logical_valid = original_logical_valid
                    joint_forward_audit = {"pairing": pairing, "scheduling": scheduling}
                    del joint_groups
                elif args.option_condition_reuse:
                    rank_records, logical_valid = shard_option_record_groups(
                        records, world,
                        balance_by_token_flops=args.token_balanced_sharding,
                        condition_batch_size=args.multi_condition_batch_size,
                        max_padded_tokens=args.max_condition_tokens,
                        condition_token_bucket_size=args.condition_token_bucket_size,
                        text_before_fm=args.text_before_fm)
                else:
                    rank_records, logical_valid = shard_record_batches(batches, world)
                if original_logical_valid is not None:
                    logical_valid = original_logical_valid
                rank_record_counts = [len(shard) for shard in rank_records]
                if len(set(rank_record_counts)) != 1:
                    raise RuntimeError(
                        f"rank-local train streams are not aligned: {rank_record_counts}")
                # Transport one aligned rank-local record at a time. A single
                # DP_COMPUTE call remains small enough for the object store; each
                # learner retains its shard in process memory. Training starts only
                # after every pi-old value has been fixed and all waves are staged.
                staging_t0 = queued_staging_t0 if queued_staging_t0 is not None else time.perf_counter()
                _get_worker_result(learner.clear_staged_records())
                if args.trajectory_transfer_queue:
                    rank_records = [[record_locator(row) for row in shard] for shard in rank_records]
                    # The store now owns the heavy payloads; retain only small
                    # rank-local schedules while all learners pull independently.
                    del batches, records
                    batch = rollout = record = None
                    gc.collect()
                    acknowledgments = _get_worker_result(stage_queued_records(
                        learner, rank_records, records_per_fetch=args.stage_records_per_rpc))
                    if acknowledgments != rank_record_counts:
                        raise RuntimeError('queued learner staging did not acknowledge every scheduled record')
                    _get_worker_result(learner.release_staged_queue_files())
                    for lease in prepared_queue_leases if args.cpu_group_preparation else [queue_lease]:
                        release_record_groups(lease)
                    if partial_batch is not None:
                        partial_producer.release(partial_batch)
                        ownership = partial_producer.audit_released_batch()
                        with (output / 'cpu_memory_release.jsonl').open('a') as memory_log:
                            memory_log.write(json.dumps(dict(step=step, phase='queue_released',
                                **ownership, preparations=[lease.get('cpu_memory')
                                    for lease in prepared_queue_leases])) + '\n')
                    del rank_records
                else:
                    staged_condition = [None] * world
                    for record_start in range(
                            0, rank_record_counts[0], args.stage_records_per_rpc):
                        staged_wave = [[] for _ in range(world)]
                        record_stop = min(
                            record_start + args.stage_records_per_rpc,
                            rank_record_counts[0])
                        for record_index in range(record_start, record_stop):
                            for rank in range(world):
                                source_record = rank_records[rank][record_index]
                                condition_id = source_record.get("condition_id")
                                if condition_id and condition_id == staged_condition[rank]:
                                    record = dict(source_record)
                                    for key in OPTION_SHARED_KEYS:
                                        record.pop(key, None)
                                    record["_reuse_staged_option_payload"] = True
                                else:
                                    record = source_record
                                    staged_condition[rank] = condition_id
                                staged_wave[rank].append(record)
                        _get_worker_result(_stage_rank_waves(
                            learner, staged_wave, world,
                            text_placements=(megatron_placements
                                if args.text_backend == "megatron" else None)))
                        del staged_wave
                    del rank_records, batches, records
                gc.collect()
                staging_seconds = time.perf_counter() - staging_t0
                piold_updates = []
                if args.old_logprob_mode == "recompute" and args.piold_option_reuse:
                    piold_t0 = time.perf_counter()
                    piold_updates = _get_worker_result(learner.compute_staged_logprobs())
                    piold_seconds = time.perf_counter() - piold_t0
                if args.joint_text_fm_forward:
                    # No rank may enter FSDP until every local pair has passed
                    # actual CPU processor/MRoPE/R3 equality and global schedule checks.
                    preflight_t0 = time.perf_counter()
                    preflight = _get_worker_result(learner.validate_staged_joint_records(
                        [logical_valid] * world))
                    from action_head_rl.joint_policy import validate_joint_preflight_reports
                    validate_joint_preflight_reports(
                        preflight, joint_forward_audit["scheduling"]["rounds"],
                        world=world, logical_valid=logical_valid)
                    joint_forward_audit["all_rank_preflight_passed"] = True
                    joint_forward_audit["preflight_seconds"] = time.perf_counter() - preflight_t0
                train_t0 = time.perf_counter()
                learner_rank_metrics = _get_worker_result(
                    learner.train_staged_records([logical_valid] * world))
                updates = _first_result(learner_rank_metrics)
                try:
                    train_seconds = time.perf_counter() - train_t0
                except Exception:
                    train_seconds = 0.0
            # Preserve the learner result before publication: a failed weight
            # transfer must not erase the only timing/update evidence. This is
            # not a checkpoint and cannot authorize a resume or new rollout.
            learner_receipt = {
                "step": step, "time": time.time(), "stage": "learner_returned",
                "episodes": episode_count,
                "logical_denominator": float(logical_valid),
                "all_ranks_finite_once": len(learner_rank_metrics) == world and all(
                    row.get("optimizer_steps") == 1
                    and math.isfinite(float(row.get("grad_norm", float("nan"))))
                    for row in learner_rank_metrics),
                "elapsed_before_publication_seconds": time.perf_counter() - step_t0,
                "rollout_seconds": rollout_seconds, "train_seconds": train_seconds,
                "staging_seconds": staging_seconds,
                "train_seconds_scope": "post_rollout_tail" if args.overlap_rollout_learner else "serial_train",
                "learner_rank_metrics": learner_rank_metrics,
                "streamed_rank_metrics": streamed_rank_metrics if args.overlap_rollout_learner else [],
                "checkpoint_committed": False, "policy_sync_complete": False,
            }
            with (output / "training_updates.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(learner_receipt, ensure_ascii=False) + "\n")
            print(json.dumps({key: value for key, value in learner_receipt.items()
                              if key not in ("learner_rank_metrics", "streamed_rank_metrics")}), flush=True)
            if tool_critic is not None:
                # Actor update succeeded. Only now may this batch's terminal
                # outcomes enter the immutable critic used by the next batch.
                previous_critic_hash = tool_critic.state_dict()["sha256"]
                tool_critic = _advance_tool_critic(
                    tool_critic, tool_credit_ledgers, learner_rank_metrics,
                    step=step, world=world, fit_updates=args.tool_credit_fit_updates)
                critic_directory = output / "tool_credit_states"
                critic_directory.mkdir(exist_ok=True)
                tool_critic.save(critic_directory / f"step_{step:06d}.json")
                advantages = [value for credit in tool_credit_diagnostics
                              for value in credit["call_advantages"].values()]
                tool_credit_summary = {
                    "mode": args.tool_credit_mode,
                    "critic_used_sha256": previous_critic_hash,
                    "critic_saved_sha256": tool_critic.state_dict()["sha256"],
                    "critic_fitted_through_step": tool_critic.fitted_through_step,
                    "episodes": len(tool_credit_ledgers), "calls": len(advantages),
                    "positive_calls": sum(value > 0 for value in advantages),
                    "negative_calls": sum(value < 0 for value in advantages),
                    "zero_calls": sum(value == 0 for value in advantages),
                    "advantage_mean": sum(advantages) / len(advantages),
                    "advantage_min": min(advantages), "advantage_max": max(advantages),
                    "logical_denominator": float(logical_valid),
                }
            if rule_credit_enabled:
                from action_head_rl.rule_credit import summarize
                tool_credit_summary = summarize(tool_credit_diagnostics)
            stopping = step >= args.opt_steps or bool(
                args.stop_after_step and step >= args.stop_after_step)
            evaluation_due = bool(eval_workers and (stopping or step % args.eval_interval == 0))
            checkpoint_due = bool(
                stopping or step % args.checkpoint_interval == 0)
            if checkpoint_due and args.keep_last_checkpoints:
                from action_head_rl.checkpoint import wait_for_companion_retention
                wait_for_companion_retention(output, next_step=step, keep_last=args.keep_last_checkpoints)
            publish_root = publication_root(
                output=output, shared_transport=policy_transport, local=local_transport,
                step=step, checkpoint_due=checkpoint_due)
            publish_t0 = time.perf_counter()
            native_receivers = None
            native_publication_options = {}
            if native_weight_targets is not None and (not stopping or evaluation_due):
                prepared = _get_worker_result(learner.prepare_native_publication())
                if sorted(prepared) != list(range(world)):
                    raise RuntimeError('native publication requires every learner to release update buffers')
                # Receivers wake weights only, keeping KV and requests asleep.
                # Launch every receiver before Bridge enters collective export.
                native_receivers = [service.receive_native_weights.remote() for service in colocated_services]
                native_publication_options = dict(native_targets=native_weight_targets,
                    native_bucket_mb=args.weight_sync_bucket_mb)
            publication_refs = learner.publish_policy(
                str(publish_root), step,
                scope is TrainScope.FULL and (
                    checkpoint_due or not (args.direct_vllm_weight_sync or native_receivers is not None)),
                **_publication_export_options(
                    checkpoint_due=checkpoint_due,
                    rollout_transport_dtype=args.rollout_transport_dtype,
                    bf16_gather_before_export=args.bf16_gather_before_export),
                **native_publication_options)
            if native_receivers is None:
                publication = _first_result(_get_worker_result(publication_refs))
            else:
                # A failed receiver leaves its sender blocked in ZMQ recv.
                # Resolve both sides together so the first error from either
                # one fails the step instead of waiting on the learners.
                joint = _get_worker_result(
                    dict(learners=publication_refs, receivers=native_receivers),
                    timeout=NATIVE_PUBLICATION_TIMEOUT_SECONDS,
                    phase=f"native policy publication step {step}")
                publication = _first_result(joint['learners'])
                acknowledgments = joint['receivers']
                expected = publication['native_transport']
                if any(any(any(row.get(key) != value for key, value in expected.items())
                           for row in replies) for replies in acknowledgments):
                    raise RuntimeError('native receivers differ from the complete published Qwen stream')
            if local_transport is not None and not checkpoint_due:
                local_transport.publication_complete(step)
            if not checkpoint_due:
                # The query-tail hook watches the durable stable filename.
                # Keep that small hot-reload artifact current even though the
                # large per-step policy export is only transport state.
                stable_query_tmp = stable_query.with_suffix(".pt.tmp")
                shutil.copy2(publication["query"], stable_query_tmp)
                os.replace(stable_query_tmp, stable_query)
            try:
                publish_seconds = time.perf_counter() - publish_t0
            except Exception:
                publish_seconds = 0.0
            trainer_state = None
            checkpoint_seconds = 0.0
            if checkpoint_due:
                trainer_state = (
                    output / f"global_step_{step:06d}" / "trainer_state")
                checkpoint_t0 = time.perf_counter()
                _get_worker_result(learner.save_checkpoint(
                    str(trainer_state), global_step=step,
                    max_ckpt_to_keep=None))
                if partial_producer is not None:
                    from action_head_rl.partial_groups import save_admission_state
                    save_admission_state(trainer_state.parent, step=step,
                                         block=partial_producer.next_groups.block)
                try:
                    checkpoint_seconds = time.perf_counter() - checkpoint_t0
                except Exception:
                    checkpoint_seconds = 0.0
                if args.resume_audit:
                    _get_worker_result(learner.audit_training_state(
                        str(output), f"saved_{step:06d}"))
                report_path = (
                    output / f"global_step_{step:06d}" / "report.json")
                report = json.loads(report_path.read_text())
                report["trainer_state"] = {
                    "format": ("verl_megatron_fm_joint_v1" if args.text_backend == "megatron"
                               else "verl_fsdp_optimizer_extra"),
                    "path": str(trainer_state),
                    "world_size": learner_world,
                }
                if tool_critic is not None:
                    critic_path = report_path.parent / "tool_credit_critic.json"
                    tool_critic.save(critic_path)
                    report["tool_credit_critic"] = {
                        "file": critic_path.name,
                        "sha256": tool_critic.state_dict()["sha256"],
                        "fitted_through_step": step,
                    }
                report_path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            reload_t0 = time.perf_counter()
            fast_policy_wake_seconds = 0.0
            # No following rollout in this process: resume will load the export.
            if not stopping or evaluation_due:
                if local_transport is not None and not checkpoint_due:
                    local_transport.begin_reload(step)
                if fast_policy_workers:
                    ray.get([worker.reload_action_head.remote(
                                 publication["action_head"], step)
                             for worker in fast_policy_workers])
                    ray.get([actor.reload_action_head.remote(
                                 publication["action_head"], step)
                             for actor in policy_consumers])
                else:
                    ray.get([actor.reload_action_head.remote(publication["action_head"], step)
                             for actor in policy_consumers])
                if scope is TrainScope.FULL:
                    if native_receivers is not None:
                        transition_t0 = time.perf_counter()
                        _get_worker_result(learner.offload_for_colocated_rollout(colocated_ranks))
                        learner_colocation_offloaded = True
                        colocated_offload_seconds = time.perf_counter() - transition_t0
                        transition_t0 = time.perf_counter()
                        colocated_urls = ray.get([service.finish_native_weights.remote(
                            publication['native_transport'], step) for service in colocated_services])
                        colocated_wake_reload_seconds = time.perf_counter() - transition_t0
                        vllm_urls = colocated_urls
                    elif direct_weight_manager is not None:
                        direct_sync_t0 = time.perf_counter()
                        direct_weight_manager.update_weights(global_steps=step)
                        direct_weight_sync_seconds = (
                            time.perf_counter() - direct_sync_t0)
                    elif colocated_services:
                        transition_t0 = time.perf_counter()
                        _get_worker_result(learner.offload_for_colocated_rollout(
                            colocated_ranks))
                        learner_colocation_offloaded = True
                        colocated_offload_seconds = (
                            time.perf_counter() - transition_t0)
                        transition_t0 = time.perf_counter()
                        # Static and colocated services read the same immutable
                        # export on disjoint GPUs. Reload all replicas in one
                        # wave so adding rollout capacity does not add a second
                        # serial model-load interval between learner steps.
                        static_refs = [
                            service.restart.remote(publication["qwen"])
                            for service in static_services]
                        colocated_refs = [
                            service.wake_reload_resume.remote(publication["qwen"])
                            for service in colocated_services]
                        reloaded_urls = ray.get(static_refs + colocated_refs)
                        static_urls = reloaded_urls[:len(static_refs)]
                        colocated_urls = reloaded_urls[len(static_refs):]
                        colocated_wake_reload_seconds = (
                            time.perf_counter() - transition_t0)
                        vllm_urls = static_urls + colocated_urls
                    else:
                        static_urls = ray.get([
                            service.restart.remote(publication["qwen"])
                            for service in static_services])
                        vllm_urls = static_urls
                if args.fast_policy_device == 'colocated_cuda':
                    transition_t0 = time.perf_counter()
                    ray.get([worker.set_rollout_resident.remote(True) for worker in fast_policy_workers])
                    fast_policy_wake_seconds = time.perf_counter() - transition_t0
                if not checkpoint_due:
                    if local_transport is not None:
                        local_transport.consumers_complete(step)
                    else:
                        shutil.rmtree(
                            policy_transport / f"global_step_{step:06d}",
                            ignore_errors=True)
            try:
                reload_seconds = (time.perf_counter() - reload_t0
                                  if not stopping or evaluation_due else 0.0)
            except Exception:
                reload_seconds = 0.0
            evaluation_summary = (run_online_evaluation(
                eval_workers, eval_cases, step=step, repeats=args.eval_repeats,
                seed=args.eval_seed, output=output) if evaluation_due else None)
            try:
                step_seconds = time.perf_counter() - step_t0
            except Exception:
                step_seconds = 0.0
            timing = {
                "fast_policy_offload_seconds": fast_policy_offload_seconds,
                "fast_policy_wake_seconds": fast_policy_wake_seconds,
                "step_seconds": step_seconds,
                "rollout_seconds": rollout_seconds,
                "piold_seconds": piold_seconds,
                "staging_seconds": staging_seconds,
                "piold_option_reuse": args.piold_option_reuse,
                "piold_rank_metrics": piold_updates,
                "learner_rank_metrics": learner_rank_metrics,
                "train_seconds": train_seconds,
                "rollout_learner_overlap": args.overlap_rollout_learner,
                **({"rollout_prefetch": rollout_prefetch.stats()} if rollout_prefetch else {}),
                "streamed_rank_metrics": (
                    streamed_rank_metrics if args.overlap_rollout_learner else []),
                **({"streamed_schedule": {
                    "revision": "all_rank_ack_one_inflight_v1",
                    "max_in_flight_learner_waves": 1,
                    "waves": learner_stream.submitted,
                    "logical_denominator": logical_valid,
                    "driver_wait_seconds": learner_stream.wait_seconds,
                    # Submission includes filtering, sharding, CPU transport
                    # and all-rank staging ACKs. It excludes the prior wave's
                    # result wait; neither field is GPU compute time.
                    "driver_submission_seconds": learner_stream.submission_seconds,
                    "driver_preparation_seconds": learner_stream.preparation_seconds,
                    "wave_driver_timings": learner_stream.driver_wave_timings,
                    "driver_pending_group_bound": args.stream_groups_per_update,
                    "rollout_admitted_group_bound": args.rollout_groups_in_flight,
                }} if args.overlap_rollout_learner else {}),
                "publish_seconds": publish_seconds,
                "checkpoint_seconds": checkpoint_seconds,
                "reload_seconds": reload_seconds,
                "evaluation_seconds": (evaluation_summary or {}).get("seconds", 0.0),
                "direct_weight_sync_seconds": direct_weight_sync_seconds,
                "colocated_sleep_seconds": colocated_sleep_seconds,
                "colocated_restore_seconds": colocated_restore_seconds,
                "colocated_offload_seconds": colocated_offload_seconds,
                "colocated_wake_reload_seconds": colocated_wake_reload_seconds,
                "episode_seconds_mean": (
                    sum(episode_seconds_list) / len(episode_seconds_list)
                    if episode_seconds_list else 0.0),
                "episode_seconds_max": (
                    max(episode_seconds_list) if episode_seconds_list else 0.0),
                "slow_qwen_seconds_sum": sum(slow_qwen_seconds_list),
                "rollout_queue_wait_seconds_sum": sum(rollout_queue_wait_seconds_list),
                "slow_qwen_seconds_mean": (
                    sum(slow_qwen_seconds_list) / len(slow_qwen_seconds_list)
                    if slow_qwen_seconds_list else 0.0),
            }
            if args.overlap_rollout_learner:
                final_memory = _get_worker_result(learner.step_memory_report(), timeout=300,
                                                  phase=f"step {step} full-step memory report")
                if len(final_memory) != world or {row['rank'] for row in final_memory} != set(range(world)):
                    raise RuntimeError("full-step memory report requires every learner rank")
                by_rank = {row['rank']: row for row in final_memory}
                learner_rank_metrics = [dict(row, **by_rank[row['rank']]) for row in learner_rank_metrics]
            memory = {
                **{f"learner_all_ranks_max_{key}": max(
                    (float(row[key]) for row in learner_rank_metrics
                     if isinstance(row, dict) and row.get(key) is not None), default=None)
                   for key in ("step_peak_memory_allocated_gb", "step_peak_memory_reserved_gb",
                               "post_update_memory_allocated_gb", "post_update_device_used_gb")},
                "learner_max_memory_allocated_gb": (
                    updates.get("max_memory_allocated_gb") if isinstance(
                        updates, dict) else None),
                "learner_memory_allocated_gb": (
                    updates.get("memory_allocated_gb") if isinstance(
                        updates, dict) else None),
                "learner_forward_seconds": (
                    updates.get("forward_seconds") if isinstance(
                        updates, dict) else None),
                "learner_backward_seconds": (
                    updates.get("backward_seconds") if isinstance(
                        updates, dict) else None),
                "actor_max_memory_gb": actor_max_memory_gb,
                **_driver_memory_gb(),
            }
            event = {
                "step": step,
                "cases": step_cases,
                "num_groups": args.num_groups_per_step,
                "group_size": args.group_size,
                "episodes": episode_count,
                "rollout_semantics": ('group_partial_decision_boundary' if partial_producer is not None
                                      else "independent_full_episode"),
                **({'partial_rollout': dict(buffer=partial_metrics, cumulative=partial_producer.stats)}
                   if partial_producer is not None else {}),
                "slow_system_calls": slow_system_calls,
                "reward_mean": float(torch.cat(reward_groups).mean()),
                "reward_std": float(torch.cat(reward_groups).std()),
                "group_rewards": [
                    {
                        "case": case,
                        "mean": float(rewards.mean()),
                        "std": float(rewards.std()),
                        "values": rewards.tolist(),
                    }
                    for case, rewards in zip(step_cases, reward_groups)
                ],
                "reward_components": reward_components_list,
                "chunks": action_chunk_count,
                **({"text_decisions": text_decision_count,
                    "training_records": training_record_count}
                   if args.text_policy else {}),
                "loss_weight": float(logical_valid),
                **({"joint_forward": joint_forward_audit} if joint_forward_audit is not None else {}),
                **({"streamed_joint_forward": streamed_joint_reports} if streamed_joint_reports else {}),
                **({"tool_credit": tool_credit_summary} if tool_credit_summary is not None else {}),
                **({"zero_advantage_filter": zero_advantage_filter}
                   if zero_advantage_filter is not None else {}),
                "rank_record_counts": rank_record_counts,
                "updates": updates,
                "checkpoint": publication,
                "checkpoint_committed": checkpoint_due,
                **({"evaluation": evaluation_summary} if evaluation_summary is not None else {}),
                "timing": timing,
                "memory": memory,
                "fast_policy": fast_policy_stats,
                "time": time.time(),
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            # Match VERL's checkpoint convention: publish the tracker only
            # after model, optimizer/extra state and step metrics are durable.
            if checkpoint_due:
                write_latest_checkpoint(output, step)
                prune_committed_checkpoints(
                    output, keep_last=args.keep_last_checkpoints,
                    train_scope=scope.value, learner_world_size=learner_world)
            if args.stop_after_step and step >= args.stop_after_step:
                print(json.dumps({"stopped_at_committed_step": step}), flush=True)
                break
            print(json.dumps(event, ensure_ascii=False), flush=True)
    finally:
        # A failed stop RPC must not bypass cleanup of our acknowledged cache.
        # Unacknowledged remote I/O is retained even when service.stop fails.
        with cleanup_local_transport(local_transport):
            active_error, partial_close_error = sys.exc_info()[1], None
            failed_native_sync = active_error is not None and native_receivers is not None
            if failed_native_sync:
                import traceback
                traceback.print_exception(active_error)
            if partial_producer is not None:
                try:
                    partial_producer.close()
                except Exception as error:
                    partial_close_error = error
                    print(json.dumps({'partial_rollout_cleanup_error': str(error)}), flush=True)
            if streamed_train_active:
                # Do not race abort with an unfinished async accumulation.
                # A failed collective still requires Ray's own failure handling;
                # no partial stream is ever sent to finish/optimizer_step.
                try:
                    if learner_stream is not None:
                        learner_stream.drain()
                except Exception as error:
                    print(json.dumps({"streamed_train_drain_error": str(error)}), flush=True)
                try:
                    _get_worker_result(learner.abort_streamed_train(), timeout=30,
                                       phase="abort partial streamed train")
                except Exception as error:
                    print(json.dumps({"streamed_train_cleanup_error": str(error)}),
                          flush=True)
            if colocated_services and not failed_native_sync:
                if not learner_colocation_offloaded:
                    try:
                        _get_worker_result(learner.offload_for_colocated_rollout(
                            colocated_ranks))
                        learner_colocation_offloaded = True
                    except Exception as error:
                        print(json.dumps({"colocated_cleanup_offload_error": str(error)}),
                              flush=True)
                if learner_colocation_offloaded:
                    ray.get([service.wake_for_shutdown.remote()
                             for service in colocated_services])
            if failed_native_sync:
                from action_head_rl.native_colocated_weights import stop_failed_native_services
                stop_failed_native_services(services)
            else:
                ray.get([service.stop.remote() for service in services])
            if trajectory_queue_created:
                import transfer_queue as tq
                tq.close()
            if local_transport is None:
                shutil.rmtree(policy_transport, ignore_errors=True)
            if partial_close_error is not None and active_error is None:
                raise RuntimeError('partial rollout cleanup failed') from partial_close_error
    done_root = os.environ.get("HC_DIR", "")
    expr_name = os.environ.get("EXPR_NAME", "")
    if done_root and expr_name:
        done = Path(done_root) / f"trainer_done_{expr_name}"
        done.write_text(
            json.dumps({"status": "succeeded", "output_dir": str(output)}) + "\n")


if __name__ == "__main__":
    main()
