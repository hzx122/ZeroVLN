"""v8 prediction-slot adapter for the pinned agentic-nav Megatron engine.

The engine owns model loading, CP/TP/EP, recomputation, backward scheduling and
the optimizer. This adapter only changes sampled targets and the v3 loss.
Imports of Megatron remain lazy so CPU input validation works before installing
the separate CP debug environment.
"""
from __future__ import annotations

from functools import partial
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import importlib.util
import inspect
import math

import torch

from action_head_rl.text_policy import is_text_record
from action_head_rl.trajectory_text import merge_trajectory_text, validate_trajectory_targets


UPSTREAM_COMMIT = "b213f118f8df230bc97f68cdc56043623c6babdb"
REFERENCE_PACKAGES = {
    "megatron-core": "0.19.2", "megatron-bridge": "0.6.2",
    "flash-linear-attention": "0.5.2",
}


def create_text_engine(*, model_path, learning_rate, total_steps, weight_decay,
                       tp=2, pp=2, cp=8, ep=16, etp=1, v8_optimizer=True,
                       dynamic_bsz=False, max_token_len_per_gpu=16384, cpu_prefetch=False):
    """One engine recipe shared by the accepted replay and the RL worker."""
    from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig
    from verl.workers.config.engine import EngineRouterReplayConfig
    from verl.trainer.config import CheckpointConfig
    optimizer_options = (dict(weight_decay=weight_decay, betas=(0.9, 0.999),
        override_optimizer_config=dict(adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8,
                                       decoupled_weight_decay=True)) if v8_optimizer else {})
    return make_engine_class(v8_optimizer=v8_optimizer, cpu_prefetch=cpu_prefetch)(
        model_config=HFModelConfig(path=model_path, trust_remote_code=True),
        engine_config=McoreEngineConfig(tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp, context_parallel_size=cp,
            expert_model_parallel_size=ep, expert_tensor_parallel_size=etp,
            sequence_parallel=tp > 1, use_remove_padding=True, use_dist_checkpointing=False,
            use_dynamic_bsz=dynamic_bsz, micro_batch_size_per_gpu=1,
            max_token_len_per_gpu=max_token_len_per_gpu,
            router_replay=EngineRouterReplayConfig(mode="R3"),
            override_transformer_config=dict(recompute_granularity="full", recompute_method="uniform",
                recompute_num_layers=1, vision_dp_when_cp=True, moe_grouped_gemm=True,
                moe_permute_fusion=True, moe_router_load_balancing_type="none",
                moe_router_fusion=False, moe_aux_loss_coeff=0.0, moe_z_loss_coeff=0.0)),
        optimizer_config=McoreOptimizerConfig(lr=learning_rate, lr_decay_style="constant",
            total_training_steps=total_steps, lr_decay_steps=total_steps,
            **optimizer_options), checkpoint_config=CheckpointConfig())


def dependency_report():
    # The upstream lock is a CUDA reference, not a PPU version requirement.
    # Preserve the vendor torch/TE stack and gate on the APIs we actually use.
    result = {"packages": {}, "api_checks": {}, "ready": False}
    for package, reference in REFERENCE_PACKAGES.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        result["packages"][package] = dict(installed=installed, upstream_reference=reference)
    try:
        bridge_available = importlib.util.find_spec("megatron.bridge") is not None
    except ModuleNotFoundError:
        bridge_available = False
    if not bridge_available:
        result["api_checks"]["bridge"] = dict(ready=False, reason="megatron.bridge is absent")
        return result
    try:
        from sft.probe_qwen36_megatron import expose_transformer_engine_pytorch
        expose_transformer_engine_pytorch()
        from megatron.bridge import AutoBridge
        # Model registration alone leaves Bridge's training dependencies lazy.
        # The engine imports this module when constructing DDP, before forward.
        from megatron.bridge.training.config import DistributedDataParallelConfig
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import model as vlm
        from megatron.core.ssm import gated_delta_net
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        from fla.ops.cp.context import build_cp_context
        checks = {
            "bridge": callable(getattr(AutoBridge, "from_hf_pretrained", None)),
            "bridge_training_config": inspect.isclass(DistributedDataParallelConfig),
            "packed_vlm": hasattr(vlm, "_is_packed_input_pre_sharded") and hasattr(vlm, "Qwen3VLModel"),
            "gdn_packed": "packed_seq_params" in inspect.signature(gated_delta_net.GatedDeltaNet.forward).parameters,
            "fla_cp": callable(build_cp_context) and "cp_context" in inspect.signature(chunk_gated_delta_rule).parameters,
        }
        # The old 0.16 source explicitly advertises its unimplemented CP path.
        source = inspect.getsource(gated_delta_net)
        checks["gdn_cp_not_explicitly_unimplemented"] = "TODO: Implement GatedDeltaNetContextParallel" not in source
        result["api_checks"] = {key: {"ready": value} for key, value in checks.items()}
        result["ready"] = all(checks.values())
        result["scope"] = "API import preflight only; real GDN/TE CP4 forward/backward still required"
    except (ImportError, AttributeError, RuntimeError) as exc:
        result["api_checks"]["import"] = dict(ready=False, reason=f"{type(exc).__name__}: {exc}")
    return result


def prepare_record(record):
    """Accept a native v8 record or one historical condition, never join windows."""
    if not is_text_record(record) or float(record.get("sample_valid", 1.0)) <= 0:
        raise ValueError("CP debug input must be a valid sampled text record")
    clean = {key: value for key, value in record.items()
             if not key.startswith(("_joint_", "_condition_pack_"))}
    if "text_trajectory_targets" in clean:
        layout = clean.get("_v8_cp_debug_input_layout", "v8_whole_trajectory")
    else:
        clean = merge_trajectory_text([clean])[-1]
        layout = "legacy_single_condition"
        clean["_v8_cp_debug_input_layout"] = layout
    ids = clean["all_input_ids"]
    validate_trajectory_targets(ids, clean["text_response_mask"],
                                clean["text_trajectory_targets"], clean["text_trajectory_weights"])
    if 'update_policy_version' in clean:
        behavior = clean.get('text_trajectory_behavior_logprobs')
        versions = clean.get('text_trajectory_policy_versions')
        mask = clean['text_response_mask']
        if (behavior is None or versions is None or behavior.shape != ids.shape or versions.shape != ids.shape
                or not torch.isfinite(behavior).all() or (behavior[mask] > 0).any()
                or (versions[mask] < 0).any() or (versions[mask] > clean['update_policy_version']).any()
                or (versions[~mask] != -1).any() or behavior[~mask].ne(0).any()):
            raise ValueError('partial trajectory lacks aligned behavior probabilities/source versions')
    if clean["all_routed_experts"].shape[0] != ids.numel():
        raise ValueError("CP replay requires exact all-token routes, including the final row")
    return clean, layout


def prediction_slot_loss(logprobs, weights, *, coefficient, logical_weight, dp_size, sample_valid=1.0):
    """The existing v8 sum, with no extra token/trajectory averaging."""
    from action_head_rl.trajectory_text import trajectory_text_loss

    return trajectory_text_loss(logprobs.reshape(1, -1), weights.reshape(1, -1),
        sample_valid=torch.as_tensor([sample_valid], device=logprobs.device),
        coefficient=coefficient, global_weight=logical_weight, dp_size=dp_size)


def prediction_slot_statistics(logits, trajectory_targets, temperature, prediction_mask, *, logprob_fn):
    """Keep the original FP32 temperature operation on sampled prediction rows."""
    result = torch.zeros_like(trajectory_targets, dtype=torch.float32)
    # CP shards without predictions still need a backward graph edge.
    result = result + logits[..., 0].float().sum() * 0.0
    if prediction_mask.any():
        selected = logits[prediction_mask].float() / temperature[prediction_mask].float().unsqueeze(-1)
        result[prediction_mask] = logprob_fn(selected, trajectory_targets[prediction_mask])
    return {"log_probs": result}


def hf_aligned_vision_positions(vision, grid_thw):
    """Preserve HF's FP32 interpolation arithmetic before BF16 vision blocks.

    Bridge casts interpolation weights to the embedding dtype and adds four
    rounded products sequentially. HF 5.12 retains FP32 weights and sums before
    the final cast. On archived v8 images this alone changes BF16 vision outputs
    by 2.8--5.4%, despite the two paths agreeing in FP32.
    """
    from transformers.vision_utils import get_vision_bilinear_indices_and_weights
    indices, weights = get_vision_bilinear_indices_and_weights(grid_thw,
        num_grid_per_side=vision.num_grid_per_side,
        spatial_merge_size=vision.config.spatial_merge_size)
    return (vision.pos_embed(indices) * weights[:, :, None]).sum(0).to(vision.pos_embed.weight.dtype)


def align_vision_forward(model, *, align_merger=True):
    """Match HF position arithmetic and merger GELU on this vision instance.

    HF's transformer MLP uses tanh GELU, but its patch merger uses exact GELU.
    Bridge shares one activation setting between the two; keep its transformer
    activation and change only the merger. The opt-out is for component probes.
    """
    from types import MethodType
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import Qwen3VLVisionModel
    vision = getattr(model, 'vision_model', None)
    if vision is None:
        return
    if not isinstance(vision, Qwen3VLVisionModel):
        raise TypeError('v8 vision interpolation requires the pinned Qwen3VL vision module')
    vision.fast_pos_embed_interpolate = MethodType(hf_aligned_vision_positions, vision)
    if align_merger:
        vision.merger.activation_func = torch.nn.functional.gelu


def make_engine_class(*, v8_optimizer=False, cpu_prefetch=False):
    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead
    from verl.models.mcore.model_forward import gptmodel_forward_model_engine
    from verl.models.mcore.util import accepts_packed_thd_vlm_inputs, preprocess_thd_engine
    from verl.utils.megatron.router_replay_patch import RouterReplayAction
    from verl.utils.megatron.router_replay_utils import set_model_router_replay_action, set_router_replay_data
    from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits
    from verl.utils.megatron_utils import unwrap_model
    from verl.utils import tensordict_utils as tu

    class V8TrajectoryMegatronEngine(MegatronEngineWithLMHead):
        def forward_backward_batch(self, *args, **kwargs):
            if not cpu_prefetch:
                return super().forward_backward_batch(*args, **kwargs)
            if len(self.module) != 1:
                raise ValueError('bounded text CPU prefetch requires one local pipeline chunk')
            prepare = partial(prepare_text_microbatch,
                              prepare=getattr(self, 'prepare_text_payload', None))
            with TextMicrobatchPrefetch(prepare) as prefetch:
                self._text_cpu_prefetch = prefetch
                try:
                    return super().forward_backward_batch(*args, **kwargs)
                finally:
                    self._text_cpu_prefetch = None

        def initialize(self):
            result = super().initialize()
            for chunk in self.module:
                align_vision_forward(unwrap_model(chunk))
            return result

        def _build_optimizer(self):
            if not v8_optimizer:
                return super()._build_optimizer()
            from megatron.core.optimizer import get_megatron_optimizer
            from verl.utils.megatron.optimizer import init_megatron_optim_config
            from verl.workers.engine.megatron.transformer_impl import register_megatron_training_hooks

            config = init_megatron_optim_config(self.optimizer_config,
                use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
                fp16=self.param_dtype == torch.float16,
                bf16=self.param_dtype == torch.bfloat16)
            # FSDP's v8 AdamW decays every trainable parameter, including bias
            # and norm vectors. MCore's None selects bias/norm exclusions.
            # An explicit empty map preserves the existing v8 parameter groups.
            optimizer = get_megatron_optimizer(config=config, model_chunks=self.module,
                                                config_overrides={})
            register_megatron_training_hooks(self.module, optimizer)
            return optimizer

        def forward_step(self, batch_iter, model, logits_processor_func, postprocess_micro_batch_func):
            del logits_processor_func
            prefetch = getattr(self, '_text_cpu_prefetch', None)
            batch = (finalize_text_microbatch(prefetch.take(batch_iter)) if prefetch is not None else
                     materialize_text_microbatch(next(batch_iter),
                         prepare=getattr(self, 'prepare_text_payload', None)))
            batch = batch.to(torch.cuda.current_device())
            unwrapped = unwrap_model(model)
            if not accepts_packed_thd_vlm_inputs(model):
                raise RuntimeError("CP debug requires Bridge >=0.6 Qwen3.5 packed VLM/mRoPE support")
            if self.engine_config.pad_to_length or self.engine_config.dynamic_context_parallel:
                raise ValueError("v8 initial CP debug requires static CP and no bucket padding")
            ids = batch["input_ids"]
            routes = batch["routed_experts"]
            if not torch.equal(ids.offsets(), routes.offsets()):
                raise ValueError("v8 native R3 must cover every input row; no synthetic final route")
            replay_mask = torch.nested.nested_tensor_from_jagged(
                torch.ones_like(ids.values(), dtype=torch.bool), offsets=ids.offsets())
            set_model_router_replay_action(unwrapped, RouterReplayAction.REPLAY_FORWARD)
            set_router_replay_data(routes, None, self.tf_config,
                getattr(unwrapped, "vp_stage", 0), replay_mask=replay_mask, model=unwrapped)

            output = gptmodel_forward_model_engine(model, ids,
                microbatch_multimodal(batch),
                logits_processor=partial(prediction_slot_statistics,
                    logprob_fn=vocab_parallel_log_probs_from_logits),
                # Unlike upstream's "label", these already address prediction
                # rows and must not be rolled left a second time.
                logits_processor_args={"trajectory_targets": batch["v8_targets"],
                                       "temperature": batch["v8_temperature"],
                                       "prediction_mask": batch["loss_mask"]},
                vision_model=True, pad_token_id=self.model_config.tokenizer.pad_token_id,
                data_format="thd", cp_layout="zigzag", position_ids=batch["position_ids"])
            set_model_router_replay_action(unwrapped, RouterReplayAction.REPLAY_BACKWARD)
            return output, partial(postprocess_micro_batch_func, data=batch)

        def postprocess_micro_batch_func(self, output, data, forward_only, loss_function, local_cp_size=None):
            result = super().postprocess_micro_batch_func(output, data, forward_only, loss_function, local_cp_size)
            if len(result) == 3:
                # Upstream static CP uses floor(real_tokens / CP). For a v8
                # length not divisible by CP that loses real tokens in the
                # final gradient denominator. Count the actual packed rows.
                ids = data["input_ids"]
                ones = torch.nested.nested_tensor_from_jagged(
                    torch.ones_like(ids.values(), dtype=torch.int32), offsets=ids.offsets())
                local = preprocess_thd_engine(ones, pre_process=True, cp_layout="zigzag")[0]
                result = result[0], local.sum().to(torch.int), result[2]
            return result

    return V8TrajectoryMegatronEngine


class TextMicrobatchPrefetch:
    """Overlap one CPU image preparation with the native GPU pipeline.

    The native iterator still owns ordering and packing. Only its caller
    advances it; one thread serializes the processor/mRoPE calls. At most one
    future owns prepared pixels beyond the current pipeline work. Never put
    expanded pixels back into the retained native microbatch list.
    """
    def __init__(self, prepare):
        self.prepare = prepare
        self.source = None
        self.future = None
        self.pool = None
        self.closed = False

    def __enter__(self):
        if self.pool is not None:
            raise RuntimeError('text prefetch context cannot be reused')
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='text-cpu-prepare')
        return self

    def _submit(self):
        try:
            batch = next(self.source)
        except StopIteration:
            return
        else:
            self.future = self.pool.submit(self.prepare, batch)

    def take(self, source):
        if self.pool is None or self.closed:
            raise RuntimeError('text prefetch requires its context manager')
        if self.source is None:
            self.source = source
            self._submit()
        elif source is not self.source:
            raise ValueError('text prefetch cannot mix native pipeline iterators')
        if self.future is None:
            raise StopIteration
        future, self.future = self.future, None
        batch = future.result()
        del future
        self._submit()
        return batch

    def __exit__(self, exc_type, exc, traceback):
        future, self.future = self.future, None
        try:
            if future is not None and not future.cancel() and exc_type is None:
                # Do not hide a producer failure on a short consumer path.
                future.result()
        finally:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.closed = True
            self.source = None
            self.prepare = None


def _microbatch_image_rows(batch):
    """Keep each trajectory's images aligned through native batch reordering."""
    from tensordict.tensorclass import NonTensorStack
    from verl.utils import tensordict_utils as tu

    value = tu.get_non_tensor_data(batch, "v8_multimodal", {})
    if isinstance(value, NonTensorStack):
        value = value.tolist()
    elif isinstance(value, dict):
        value = [value]
    if (not isinstance(value, list) or len(value) != len(batch)
            or not all(isinstance(row, dict) for row in value)):
        raise ValueError("v8 multimodal payload must match every trajectory in the microbatch")
    return value


def microbatch_multimodal(batch):
    """Flatten images in the same order as packed THD trajectory tokens."""
    rows = _microbatch_image_rows(batch)
    if len(rows) == 1:
        return rows[0]
    if any('_deferred_text_record' in row for row in rows):
        raise ValueError('materialize deferred images before forwarding a packed microbatch')
    return {key: torch.cat([row[key] for row in rows if key in row], dim=0)
            for key in ('pixel_values', 'image_grid_thw') if any(key in row for row in rows)}


def materialize_text_microbatch(batch, *, prepare):
    """Expand images only for this pipeline microbatch, never the full DP batch.

    Do not mutate the iterator's batch: native Megatron retains its microbatch
    list until the whole update ends. A shallow copy keeps expanded pixels
    alive only with the in-flight forward/backward closure.
    """
    return finalize_text_microbatch(prepare_text_microbatch(batch, prepare=prepare))


def prepare_text_microbatch(batch, *, prepare):
    """Prepare ordinary CPU tensors only; safe to overlap model FX tracing.

    Nested tensor construction consults a process-global FX tracing flag, even
    in another thread. Keep it and TensorDict mutation on the consumer thread.
    """
    rows = _microbatch_image_rows(batch)
    deferred = ['_deferred_text_record' in row for row in rows]
    if not any(deferred):
        return batch, None, None
    if not all(deferred):
        raise ValueError('one microbatch cannot mix deferred and eager image payloads')
    if prepare is None:
        raise RuntimeError('deferred text requires the original processor and model mRoPE')
    images, positions = [], []
    for row in rows:
        record = row['_deferred_text_record']
        payload, mrope = prepare(record)
        positions.append(position_values(record['all_input_ids'], mrope))
        images.append({key: payload[key] for key in ('pixel_values', 'image_grid_thw') if key in payload})
    return batch, images, torch.cat(positions, dim=1)


def finalize_text_microbatch(prepared):
    """Build nested metadata on the model thread, outside its FX trace."""
    from verl.utils import tensordict_utils as tu
    batch, images, positions = prepared
    if images is None:
        return batch
    result = batch.clone(recurse=False)
    # The logical batch and native dynamic scheduler otherwise concatenate
    # all R3 tensors twice, in addition to the original record storage on
    # every TP/PP/CP rank. Only materialize the current pipeline microbatch.
    result['routed_experts'] = tu.nested_tensor_from_tensor_list(
        [row['_deferred_text_record']['all_routed_experts']
         for row in _microbatch_image_rows(batch)], ragged_idx=1)
    result['position_ids'] = torch.nested.nested_tensor_from_jagged(
        positions, offsets=batch['input_ids'].offsets(), jagged_dim=2)
    tu.assign_non_tensor_stack(result, 'v8_multimodal', images)
    return result


def shard_text_trajectories(records, *, dp_size, policy_version):
    """Assign each complete trajectory once to a dense-DP replica.

    TP/PP/CP ranks in a replica receive the same shard. Padding repeats a short
    real input with zero loss, solely to keep all replicas' PP schedules equal.
    The caller must still supply the original logical denominator, which is
    not derivable from retained record count (e.g. zero-credit filtering).
    """
    if type(dp_size) is not int or dp_size < 1:
        raise ValueError("dense DP size must be a positive integer")
    rows = [row for row in records if is_text_record(row)]
    if not rows:
        raise ValueError("full-step text update has no trajectories")
    for row in rows:
        if row.get('update_policy_version', row.get("policy_version")) != policy_version:
            raise ValueError("full-step text update mixes policy versions")
        if "text_trajectory_targets" not in row or row.get("_v8_cp_debug_input_layout") == "legacy_single_condition":
            raise ValueError("full-step update requires merged native v8 trajectories")
        prepare_record(row)
    if len({float(row["text_loss_coef"]) for row in rows}) != 1:
        raise ValueError("one logical update must use a common text coefficient")
    return schedule_text_rows(rows, dp_size=dp_size,
                              lengths=[row['all_input_ids'].numel() for row in rows])


def schedule_text_rows(rows, *, dp_size, lengths):
    """Shared pure scheduling for validated tensors or CPU-worker descriptors."""
    if (type(dp_size) is not int or dp_size < 1 or not rows or len(rows) != len(lengths)
            or any(type(length) is not int or length < 1 for length in lengths)):
        raise ValueError('text scheduling needs positive lengths and a dense DP size')
    # Longest-first scheduling balances work without changing loss weights.
    shards, loads = [[] for _ in range(dp_size)], [0] * dp_size
    for index in sorted(range(len(rows)), key=lambda index: -lengths[index]):
        rank = min(range(dp_size), key=lambda rank: (loads[rank], rank))
        shards[rank].append((rows[index], 1.0))
        loads[rank] += lengths[index]
    width = max(map(len, shards))
    padding = rows[min(range(len(rows)), key=lambda index: lengths[index])]
    for shard in shards:
        shard.extend((padding, 0.0) for _ in range(width - len(shard)))
    return shards


def build_logical_batch(items, payloads=None, positions=None, *, logical_weight,
                        dynamic_bsz=False, max_token_len_per_gpu=16384):
    """One engine invocation for ALL local trajectories, one finalization.

    Calling forward_backward_batch repeatedly on accumulated MCore buffers
    would finalize/reduce/normalize old gradients repeatedly. Instead use its
    native pipeline microbatch schedule with one whole batch.
    """
    from verl.utils import tensordict_utils as tu

    deferred = payloads is None and positions is None
    if deferred:
        payloads, positions = [None] * len(items), [None] * len(items)
    if (not items or payloads is None or positions is None
            or len(items) != len(payloads) or len(items) != len(positions)):
        raise ValueError("one payload and mRoPE tensor required per trajectory")
    if len({float(row["text_loss_coef"]) for row, _ in items}) != 1:
        raise ValueError("mixed text coefficients in one logical batch")
    batches = []
    for (record, valid), payload, mrope in zip(items, payloads, positions, strict=True):
        batch = build_batch(record, payload, mrope, logical_weight=logical_weight,
                            sample_valid=valid, deferred=deferred)
        # Per-example objects must be a stack, not broadcast batch metadata.
        images = tu.get_non_tensor_data(batch, "v8_multimodal", {})
        tu.assign_non_tensor_stack(batch, "v8_multimodal", [images])
        batches.append(batch)
    batch = tu.concat_tensordict(batches)
    if type(dynamic_bsz) is not bool or type(max_token_len_per_gpu) is not int or max_token_len_per_gpu <= 0:
        raise ValueError('dynamic batching requires a boolean and positive per-GPU token budget')
    tu.assign_non_tensor(batch, use_dynamic_bsz=dynamic_bsz,
                         max_token_len_per_gpu=max_token_len_per_gpu)
    return batch


def position_values(ids, mrope):
    n = ids.numel()
    if mrope is None or tuple(mrope.shape) != (3, 1, n):
        raise ValueError("Qwen native mRoPE must have shape (3, 1, exact_tokens)")
    return torch.cat((torch.arange(n).unsqueeze(0), mrope[:, 0]), dim=0)


def packed_position_ids(ids, mrope):
    n = ids.numel()
    positions = position_values(ids, mrope)
    return torch.nested.nested_tensor_from_jagged(
        positions, offsets=torch.tensor([0, n], dtype=torch.long), jagged_dim=2)


def build_batch(record, payload, mrope, *, logical_weight, sample_valid, deferred=False):
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu

    if not math.isfinite(logical_weight) or logical_weight <= 0:
        raise ValueError("explicit original logical loss denominator must be positive")
    ids = record["all_input_ids"]
    n = ids.numel()
    offsets = torch.tensor([0, n], dtype=torch.long)
    def nested(value):
        return torch.nested.nested_tensor_from_jagged(value, offsets=offsets)
    batch = TensorDict(dict(input_ids=nested(ids),
        loss_mask=nested(record["text_response_mask"]),
        v8_targets=nested(record["text_trajectory_targets"]),
        v8_weights=nested(record["text_trajectory_weights"]),
        v8_temperature=nested(torch.full((n,), float(record["text_temperature"]))),
        v8_sample_valid=torch.tensor([sample_valid], dtype=torch.float32)), batch_size=[1])
    if 'update_policy_version' in record:
        batch['v8_behavior_logprobs'] = nested(record['text_trajectory_behavior_logprobs'])
    if deferred:
        if payload is not None or mrope is not None:
            raise ValueError('deferred batch cannot retain pre-expanded pixels or positions')
        images = {'_deferred_text_record': record}
    else:
        batch['routed_experts'] = nested(record["all_routed_experts"])
        batch['position_ids'] = packed_position_ids(ids, mrope)
        images = {k: payload[k] for k in ('pixel_values', 'image_grid_thw') if k in payload}
    tu.assign_non_tensor(batch, v8_multimodal=images,
        v8_logical_weight=float(logical_weight), v8_coefficient=float(record["text_loss_coef"]),
        enable_routing_replay=True, use_dynamic_bsz=False, micro_batch_size_per_gpu=1)
    return batch


def loss_function(*, model_output, data, dp_group):
    from verl.utils import tensordict_utils as tu
    dp = torch.distributed.get_world_size(dp_group) if dp_group is not None else 1
    # Dynamic microbatches can pack real and zero-loss padding trajectories.
    # Expand validity over that trajectory's rows, never broadcast row zero.
    valid = torch.repeat_interleave(data['v8_sample_valid'], data['input_ids'].offsets().diff())
    weights = data['v8_weights'].values() * valid
    correction_metrics = {}
    if 'v8_behavior_logprobs' in data:
        from action_head_rl.behavior_logprobs import correction_weights
        correction, correction_metrics = correction_weights(model_output['log_probs'].values(),
            data['v8_behavior_logprobs'].values(), data['loss_mask'].values() & valid.bool())
        weights = weights * correction
    loss = prediction_slot_loss(model_output["log_probs"].values(), weights,
        sample_valid=1., dp_size=dp,
        coefficient=tu.get_non_tensor_data(data, "v8_coefficient", None),
        logical_weight=tu.get_non_tensor_data(data, "v8_logical_weight", None))
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite v8 CP text loss")
    return loss, {"v8_text_loss": float(loss.detach()), "text_forward_count": len(data),
                  **{'v8_partial/' + key: value for key, value in correction_metrics.items()}}
