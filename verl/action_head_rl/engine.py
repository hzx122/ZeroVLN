"""VERL FSDP2 engine for route-replayed continuous-action GRPO updates."""
from __future__ import annotations

import os
import json
import time
from contextlib import nullcontext

import torch
from tensordict import TensorDict

from action_head_rl.config import TrainScope, validate_training_contract, configure_process_compile_cache
from action_head_rl.model import set_train_scope
from action_head_rl.router_replay import replay_qwen35_moe_routes
from action_head_rl.sde import select_proximal_old_logprob
from action_head_rl.text_policy import text_gpg_loss
from sft.fla_l2norm_dynamic_nb import maybe_install_l2norm_dynamic_nb_patch
from sft.fla_unused_shape_dynamic import maybe_install_fla_unused_shape_patch
from sft.qwen35_padding_mask import install_single_batch_padding_fix
from sft.qwen35_sdpa_mask import install_unpadded_causal_sdpa
from sft.verl_action_head_engine import ActionHeadFSDPEngine, unpack_anchor_batch
from verl.workers.engine.base import EngineRegistry


def initialize_learner_collectives(*, probe_bytes=64 * 1024 * 1024):
    """Allocate and check the weight-broadcast communicator before model loading.

    NCCL initializes lazily. Waiting until FSDP broadcasts the full state can
    make RDMA queue allocation compete with all ranks' unsharded CPU weights.
    Reuse the default process group used by fsdp2_load_full_state_dict; this
    allocates no model tensors and consumes no random numbers. Scalar messages
    alone can leave bulk-transfer connections uninitialized: the actual PPU
    transport has passed the scalar check then failed CQ creation for the
    initial weight broadcast. Exercise both sizes before loading weights.
    """
    dist = torch.distributed
    if not dist.is_initialized():
        raise RuntimeError("learner process group must exist before model initialization")
    if type(probe_bytes) is not int or probe_bytes < 0 or probe_bytes % 4:
        raise ValueError('collective probe bytes must be a nonnegative multiple of four')
    rank, world = dist.get_rank(), dist.get_world_size()
    value = torch.full((1,), float(rank + 1), device=torch.cuda.current_device())
    dist.all_reduce(value)
    if value.item() != world * (world + 1) / 2:
        raise RuntimeError("learner collective initialization returned an incorrect sum")
    value.fill_(7.0 if rank == 0 else 0.0)
    dist.broadcast(value, src=0)
    if value.item() != 7.0:
        raise RuntimeError("learner weight-broadcast initialization returned an incorrect value")
    if probe_bytes:
        print(json.dumps(dict(event='learner_bulk_collectives_start', rank=rank,
                              bytes=probe_bytes, before_model_load=True)), flush=True)
        bulk = torch.full((probe_bytes // 4,), float(rank + 1), dtype=torch.float32,
                          device=torch.cuda.current_device())
        dist.all_reduce(bulk)
        low, high = torch.aminmax(bulk)
        expected = world * (world + 1) / 2
        if low.item() != expected or high.item() != expected:
            raise RuntimeError('learner bulk all-reduce returned incorrect values')
        bulk.fill_(7.0 if rank == 0 else 0.0)
        dist.broadcast(bulk, src=0)
        low, high = torch.aminmax(bulk)
        if low.item() != 7.0 or high.item() != 7.0:
            raise RuntimeError('learner bulk weight-broadcast returned incorrect values')
        del bulk, low, high
    audit = dict(event="learner_collectives_ready", rank=rank, world_size=world,
                 before_model_load=True, bulk_probe_bytes=probe_bytes,
                 seconds_since_epoch=time.time())
    print(json.dumps(audit), flush=True)
    return audit


@EngineRegistry.register(
    model_type="action_head_grpo", backend=["fsdp", "fsdp2"], device="cuda")
class ActionHeadGRPOFSDPEngine(ActionHeadFSDPEngine):
    """Keep VERL's FSDP lifecycle and replace only the continuous policy loss."""

    def initialize(self):
        # The pinned upstream uses asynchronous D2H. FSDP's immediate CPU
        # re-padding reads incomplete shards before that copy has finished.
        # Install before initialize() can perform its first model offload.
        from sft.fsdp_model_transfer import install_blocking_fsdp2_offload
        install_blocking_fsdp2_offload()
        self.collective_initialization_audit = initialize_learner_collectives()
        return super().initialize()

    def _build_module(self):
        self.compile_cache_audit = configure_process_compile_cache(
            f"learner_world_{os.environ.get('WORLD_SIZE', 'unknown')}_"
            f"rank_{os.environ.get('RANK', 'unknown')}")
        # The finalized Head-only path must never import or modify FLA.  Full
        # candidates opt in after TrainingWorker has bound the local CUDA
        # device and before the first Qwen forward compiles l2norm kernels.
        if os.environ.get("ACTION_HEAD_TRAIN_SCOPE") == TrainScope.FULL.value:
            self.padding_mask_patch_audit = install_single_batch_padding_fix()
            self.sdpa_mask_patch_audit = (install_unpadded_causal_sdpa()
                if os.environ.get('ACTION_HEAD_CAUSAL_SDPA_UNPADDING', '1') == '1'
                else dict(enabled=False, status='explicitly_disabled'))
            self.fla_l2norm_patch_audit = maybe_install_l2norm_dynamic_nb_patch()
            self.fla_unused_shape_patch_audit = maybe_install_fla_unused_shape_patch()
        else:
            self.padding_mask_patch_audit = {
                "enabled": False, "status": "full_only_guard"}
            self.sdpa_mask_patch_audit = {
                "enabled": False, "status": "full_only_guard"}
            self.fla_l2norm_patch_audit = {
                "enabled": False, "status": "full_only_guard"}
            self.fla_unused_shape_patch_audit = {
                "enabled": False, "status": "full_only_guard"}
        module = super()._build_module()
        scope, _ = validate_training_contract(
            train_scope=os.environ.get("ACTION_HEAD_TRAIN_SCOPE", "head_query"),
            router_replay_mode=os.environ.get("ACTION_HEAD_ROUTER_REPLAY", "R3"),
            capture_full_routes=(
                os.environ.get("ACTION_HEAD_CAPTURE_FULL_ROUTING", "0") == "1"),
            rollout_backend="vllm",
        )
        self.action_head_train_scope = scope
        if os.environ.get("ACTION_HEAD_TEXT_POLICY", "0") == "1" and scope is not TrainScope.FULL:
            raise ValueError("text policy gradients require full train scope")
        self.trainable_audit = set_train_scope(module, scope)
        if scope is TrainScope.FULL:
            # FSDP2 requires every original floating-point parameter managed
            # by one fully-sharded root to have a uniform dtype.  The HF Qwen
            # checkpoint is BF16 while the separately trained query/head
            # checkpoints are FP32.  In head_query mode the frozen Qwen is not
            # part of the trainable shard; in full mode align the complete
            # joint policy to Qwen's storage dtype before FSDP wraps it.
            qwen_dtype = next(
                parameter.dtype for parameter in module.qwen.parameters()
                if parameter.is_floating_point())
            module.to(dtype=qwen_dtype)
            trainable_dtypes = {
                parameter.dtype for parameter in module.parameters()
                if parameter.requires_grad and parameter.is_floating_point()
            }
            if trainable_dtypes != {qwen_dtype}:
                raise RuntimeError(
                    "full-scope FSDP parameters must have one dtype, got "
                    f"{trainable_dtypes}")
        if os.environ.get("ACTION_HEAD_OPTION_CONDITION_REUSE", "0") == "1":
            stochastic_dropout = [
                (name, float(layer.p)) for name, layer in module.qwen.named_modules()
                if isinstance(layer, torch.nn.Dropout) and float(layer.p) != 0.0
            ]
            text_config = getattr(module.qwen.config, "text_config", module.qwen.config)
            attention_dropout = float(
                getattr(text_config, "attention_dropout", 0.0) or 0.0)
            if stochastic_dropout or attention_dropout != 0.0:
                raise ValueError(
                    "option condition reuse requires deterministic frozen Qwen; "
                    f"dropout_modules={stochastic_dropout[:8]}, "
                    f"attention_dropout={attention_dropout}")
        if (os.environ.get("ACTION_HEAD_JOINT_TEXT_FM_FORWARD", "0") == "1"
                or os.environ.get("ACTION_HEAD_COMPACT_ZERO_ADVANTAGE_WAVES", "0") == "1"):
            stochastic = [(name, layer.p) for name, layer in module.named_modules()
                          if isinstance(layer, torch.nn.Dropout) and layer.p]
            if stochastic:
                mode = ("joint forward" if os.environ.get("ACTION_HEAD_JOINT_TEXT_FM_FORWARD", "0") == "1"
                        else "zero-wave compaction")
                raise ValueError(f"{mode} requires zero dropout in both branches: {stochastic}")
        return module

    def _build_fsdp_module(self, module):
        wrapped = super()._build_fsdp_module(module)
        if self.engine_config.strategy == "fsdp2" and self.engine_config.forward_prefetch:
            from torch.distributed.fsdp import FSDPModule
            layers = list(wrapped.query_forward._backbone().layers)
            if not layers or any(not isinstance(layer, FSDPModule) for layer in layers):
                raise ValueError("explicit FSDP2 prefetch requires individually sharded Qwen layers")
            for index, layer in enumerate(layers):
                # Depth one only: earlier CPU launch, same overlap depth as implicit prefetch.
                # Preserve default backward order for the selective action/text backward graph.
                layer.set_modules_to_forward_prefetch(layers[index + 1:index + 2])
            self.fsdp2_prefetch_audit = dict(depth=1, decoder_layers=len(layers),
                                           backward="implicit", gpu_verified=False)
        return wrapped

    def _build_optimizer(self, module):
        """Allow a conservative VLM LR without changing the GRPO objective."""
        from verl.workers.config.optimizer import build_optimizer

        qwen_ids = {id(parameter) for parameter in module.qwen.parameters()}
        vlm, action = [], []
        for parameter in module.parameters():
            if not parameter.requires_grad:
                continue
            (vlm if id(parameter) in qwen_ids else action).append(parameter)
        if not action:
            raise RuntimeError("Action Head/query optimizer group is empty")
        base_lr = float(self.optimizer_config.lr)
        vlm_lr = float(os.environ.get("ACTION_HEAD_VLM_LR", "0") or 0.0)
        groups = [{"params": action, "lr": base_lr, "name": "action_head_query"}]
        if vlm:
            groups.append({
                "params": vlm,
                "lr": vlm_lr if vlm_lr > 0 else base_lr,
                "name": "slow_vlm",
            })
        return build_optimizer(groups, self.optimizer_config)

    def forward_backward_batch(self, data: TensorDict, loss_function,
                               forward_only: bool = False, *,
                               rule_audit_text_only: bool = False):
        del loss_function
        payload = unpack_anchor_batch(data.to(torch.cuda.current_device()))
        if "text_response_mask" in payload:
            if rule_audit_text_only:
                raise ValueError("text-only audit requires a rule shared training payload")
            return self._forward_backward_text(payload, forward_only=forward_only)
        if "joint_text_response_mask" in payload and (
                os.environ.get("ACTION_HEAD_JOINT_TEXT_FM_FORWARD", "0") != "1"
                or os.environ.get("ACTION_HEAD_TEXT_POLICY", "0") != "1"
                or os.environ.get("ACTION_HEAD_OLD_LOGPROB_MODE", "recompute") != "single_forward"
                or self.action_head_train_scope is not TrainScope.FULL):
            raise ValueError("joint text/FM forward requires opt-in Full single-forward text training")
        started = time.perf_counter()
        slow_keys = {
            "input_ids", "attention_mask", "pixel_values", "image_grid_thw",
            "mm_token_type_ids",
        }
        slow = {key: payload[key] for key in slow_keys if key in payload}
        context = torch.no_grad() if forward_only else nullcontext()
        prefix_cache = getattr(self.module.query_forward, '_fm_prefix_cache', None)
        route_context = prefix_cache.replay(self.module.query_forward._backbone(),
            payload['all_routed_experts']) if prefix_cache is not None else replay_qwen35_moe_routes(
            self.module.query_forward._backbone(),
            payload["all_routed_experts"], token_offset=0)
        device = payload["input_ids"].device
        isolate = os.environ.get("ACTION_HEAD_RULE_CREDIT", "0") == "1"
        # Rule-credit FM owns query/head leaves even when text uses a separate
        # trajectory graph. Joint forward is a layout choice, not permission
        # to compute (and subsequently discard) Qwen parameter derivatives.
        selective = isolate
        if rule_audit_text_only and (forward_only or not selective or
                                    "joint_text_response_mask" not in payload):
            raise ValueError("text-only audit requires a rule shared training payload")
        if isolate and "joint_text_response_mask" in payload and not selective:
            raise ValueError("rule shared forward requires selective action backward")
        with route_context, context, torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prepared = self.module(
                **slow,
                reference_visual_tokens=payload["reference_visual_tokens"],
                current_visual_tokens=payload["current_visual_tokens"],
                reference_visual_mask=payload.get("reference_visual_mask"),
                current_visual_mask=payload.get("current_visual_mask"),
                grpo_query_routes=payload["query_routed_experts"],
                grpo_all_routes=payload.get("all_routed_experts"),
                grpo_all_positions=payload.get("all_positions"),
                grpo_train_scope=self.action_head_train_scope.value,
                grpo_states=payload["states"],
                grpo_successors=payload["successors"],
                grpo_timesteps=payload["timesteps"],
                grpo_signed_dt=float(payload["signed_dt"].reshape(-1)[0].item()),
                grpo_noise_scales=payload["noise_scales"],
                grpo_var_floor=float(payload["var_floor"].reshape(-1)[0].item()),
                grpo_chunk_anchor_index=payload.get("grpo_chunk_anchor_index"),
                grpo_routes_already_installed=True,
                grpo_joint_text_response_mask=payload.get("joint_text_response_mask"),
                grpo_executable_policy=payload.get("joint_text_executable_policy"),
                grpo_text_temperature=float(payload.get(
                    "joint_text_temperature", torch.tensor([1.0])).reshape(-1)[0]),
            )
            try:
                forward_seconds = time.perf_counter() - started
            except Exception:
                forward_seconds = 0.0
            new_logprob = prepared["new_logprob"]
            old_logprob_mode = os.environ.get(
                "ACTION_HEAD_OLD_LOGPROB_MODE", "recompute")
            old_logprob = select_proximal_old_logprob(
                new_logprob, payload["old_logprob"], old_logprob_mode)
            advantage = payload["advantage"].reshape_as(new_logprob)
            valid = payload.get("sample_valid", torch.ones_like(advantage)).reshape_as(
                advantage).to(advantage)
            loss_weight = payload.get(
                "loss_weight", torch.ones_like(advantage)).reshape_as(
                    advantage).to(advantage)
            weighted_valid = valid * loss_weight
            clip = float(os.environ.get("ACTION_HEAD_GRPO_CLIP", "0.2"))
            log_ratio = (new_logprob.float() - old_logprob.float()).clamp(-20, 20)
            ratio = log_ratio.exp()
            unclipped = ratio * advantage.float()
            clipped = ratio.clamp(1.0 - clip, 1.0 + clip) * advantage.float()
            global_valid = weighted_valid.detach().sum().float()
            group = self.get_data_parallel_group()
            if group is not None:
                torch.distributed.all_reduce(
                    global_valid, op=torch.distributed.ReduceOp.SUM, group=group)
                dp_size = torch.distributed.get_world_size(group)
            else:
                dp_size = 1
            logical_valid = payload.get("logical_batch_valid")
            if logical_valid is not None:
                # Every rank receives the same logical-batch denominator.  Do
                # not all-reduce it: it already counts all ranks/microbatches.
                global_valid = logical_valid.reshape(-1)[0].float()
            if 'partial_policy_correction' in payload:
                from action_head_rl.behavior_logprobs import correction_weights
                if old_logprob_mode != 'single_forward':
                    raise ValueError('partial FM correction requires the single-update v3 objective')
                correction, _ = correction_weights(new_logprob,
                    payload['old_logprob'].reshape_as(new_logprob), valid.bool())
                # Preserve the original episode denominator. IS changes only
                # the numerator, and must never overwrite the stored behavior.
                weighted_valid = weighted_valid * correction
            # FSDP averages gradients across DP ranks. Multiplying each local
            # loss by dp/global_valid makes accumulated microbatches equal the
            # global weighted sample mean, including final-batch padding.
            loss = -(torch.minimum(unclipped, clipped) * weighted_valid).sum() \
                * dp_size / global_valid.clamp_min(1.0)
            if "action_path_advantage" in payload:
                from action_head_rl.executable_policy_learning import path_policy_loss
                loss = loss + path_policy_loss(
                    new_logprob, old_logprob, payload["action_path_advantage"], weighted_valid,
                    clip=clip, global_valid=global_valid, dp_size=dp_size)
            fm_loss = loss
            joint_loss = None
            if "joint_text_response_mask" in payload:
                if os.environ.get("ACTION_HEAD_JOINT_TEXT_FM_FORWARD", "0") != "1":
                    raise ValueError("joint text/FM forward is explicitly opt-in")
                joint_loss = text_gpg_loss(
                    prepared["text_logprobs"], payload["joint_text_response_mask"],
                    payload["joint_text_advantage"],
                    sample_valid=payload["joint_text_sample_valid"],
                    loss_weight=payload["joint_text_loss_weight"],
                    coefficient=float(payload["joint_text_loss_coef"].reshape(-1)[0]),
                    global_weight=global_valid.clamp_min(1.0), dp_size=dp_size,
                    stop_target=payload.get("joint_text_stop_target"),
                    stop_weight=payload.get("joint_text_stop_weight"),
                    path_advantage=payload.get("joint_text_path_advantage"))
                loss = fm_loss + joint_loss
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite Action Head GRPO loss")
            backward_seconds = 0.0
            if not forward_only:
                if selective:
                    # The first traversal requests only action leaves: no Qwen
                    # weight gradient is computed or reduce-scattered. Retain the
                    # shared checkpoint graph for the ordinary text backward.
                    if not rule_audit_text_only:
                        torch.autograd.backward(fm_loss, inputs=prepared["action_parameters"],
                                                retain_graph=joint_loss is not None)
                    if joint_loss is not None:
                        joint_loss.backward()
                else:
                    loss.backward()
                try:
                    backward_seconds = time.perf_counter() - started - forward_seconds
                except Exception:
                    backward_seconds = 0.0
        # Peak is reset once per logical batch in worker.train_record_batches,
        # so max_memory_allocated() here is the monotonic step peak so far.
        try:
            max_memory_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        except Exception:
            max_memory_allocated_gb = 0.0
        try:
            memory_allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        except Exception:
            memory_allocated_gb = 0.0
        metrics = {
            "policy_loss": loss.detach().float().cpu(),
            "ratio_mean": ratio.detach().mean().float().cpu(),
            "approx_kl": ((ratio - 1.0) - log_ratio).detach().mean().float().cpu(),
            "clip_fraction": ((ratio - 1.0).abs() > clip).detach().float().mean().cpu(),
            "old_logprob_single_forward": float(old_logprob_mode == "single_forward"),
            "action_qwen_gradient_isolated": float(isolate),
            "action_selective_backward": float(selective),
            "shared_text_action_forward": float(joint_loss is not None),
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "max_memory_allocated_gb": max_memory_allocated_gb,
            "memory_allocated_gb": memory_allocated_gb,
        }
        if joint_loss is not None:
            metrics.update({"fm_policy_loss": fm_loss.detach().float().cpu(),
                            "text_policy_loss": joint_loss.detach().float().cpu(),
                            "joint_forward": 1.0})
        return {
            "model_output": {"new_logprob": new_logprob.detach().float().cpu()},
            "loss": loss.detach().float().cpu(),
            "metrics": metrics,
        }

    def _forward_backward_text(self, payload, *, forward_only=False):
        """Accumulate one language decision into the existing joint optimizer."""
        if os.environ.get("ACTION_HEAD_TEXT_POLICY", "0") != "1":
            raise ValueError("text policy gradients are opt-in")
        if self.action_head_train_scope is not TrainScope.FULL:
            raise ValueError("text policy gradients require full train scope")
        if os.environ.get("ACTION_HEAD_OLD_LOGPROB_MODE", "recompute") != "single_forward":
            raise ValueError("one-pass text GPG requires single_forward policy updates")
        started = time.perf_counter()
        slow_keys = {"input_ids", "attention_mask", "pixel_values",
                     "image_grid_thw", "mm_token_type_ids"}
        slow = {key: payload[key] for key in slow_keys if key in payload}
        context = torch.no_grad() if forward_only else nullcontext()
        device = payload["input_ids"].device
        routes = replay_qwen35_moe_routes(
            self.module.query_forward._backbone(), payload["all_routed_experts"],
            token_offset=0)
        with routes, context, torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda"):
            prepared = self.module(
                **slow, grpo_all_routes=payload["all_routed_experts"],
                grpo_all_positions=payload["all_positions"],
                grpo_train_scope="full", grpo_routes_already_installed=True,
                grpo_text_response_mask=payload["text_response_mask"],
                grpo_executable_policy=payload.get("text_executable_policy"),
                grpo_text_trajectory_targets=payload.get("text_trajectory_targets"),
                grpo_text_single_token_target_ids=payload.get("text_single_token_target_ids"),
                grpo_text_temperature=float(payload["text_temperature"].reshape(-1)[0]))
            logprobs = prepared["text_logprobs"]
            valid = payload.get("sample_valid", torch.ones_like(payload["advantage"]))
            weight = payload.get("loss_weight", torch.ones_like(valid))
            denominator = (valid * weight).detach().sum().float()
            group = self.get_data_parallel_group()
            dp_size = 1
            if group is not None:
                torch.distributed.all_reduce(denominator, group=group)
                dp_size = torch.distributed.get_world_size(group)
            if "logical_batch_valid" in payload:
                denominator = payload["logical_batch_valid"].reshape(-1)[0]
            if "text_trajectory_weights" in payload:
                from action_head_rl.trajectory_text import trajectory_text_loss
                loss = trajectory_text_loss(logprobs, payload["text_trajectory_weights"],
                    sample_valid=valid,
                    coefficient=float(payload["text_loss_coef"].reshape(-1)[0]),
                    global_weight=denominator.clamp_min(1.0), dp_size=dp_size)
            else:
                loss = text_gpg_loss(
                    logprobs, payload["text_response_mask"], payload["advantage"],
                    sample_valid=valid, loss_weight=weight,
                    coefficient=float(payload["text_loss_coef"].reshape(-1)[0]),
                    global_weight=denominator.clamp_min(1.0), dp_size=dp_size,
                    stop_target=payload.get("text_stop_target"),
                    stop_weight=payload.get("text_stop_weight"),
                    path_advantage=payload.get("text_path_advantage"))
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite text GPG loss")
            forward_seconds = time.perf_counter() - started
            if not forward_only:
                loss.backward()
            backward_seconds = time.perf_counter() - started - forward_seconds
        detached = loss.detach().float().cpu()
        metrics = {
            "policy_loss": detached, "text_policy_loss": detached,
            "text_response_tokens": float(payload["text_response_mask"].sum()),
            "text_decisions": float((valid * payload.get("text_trajectory_decisions", torch.ones_like(valid))).sum()),
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "max_memory_allocated_gb": (torch.cuda.max_memory_allocated() / 1024 ** 3
                                         if device.type == "cuda" else 0.0),
            "memory_allocated_gb": (torch.cuda.memory_allocated() / 1024 ** 3
                                    if device.type == "cuda" else 0.0),
        }
        return {"model_output": {"text_logprobs": logprobs.detach().float().cpu()},
                "loss": detached, "metrics": metrics}
