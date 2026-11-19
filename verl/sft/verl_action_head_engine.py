"""VERL FSDP engine for frozen-Qwen latent-query/FM-head SFT.

This module is loaded through VERL's ``model.external_lib`` hook.  It keeps
VERL's proven model loading, FSDP wrapping, optimizer, scheduler and worker
orchestration, while replacing only the model-specific forward/loss boundary.
"""
from __future__ import annotations

import os
import time
from contextlib import nullcontext

import torch
from tensordict import TensorDict

from models.dualvln_query_encoder import DualVlnQueryEncoder
from models.qwen36_dualvln_action_head import Qwen36DualVlnActionHead
from models.qwen36_fast_slow_action_head import Qwen36FastSlowActionHead
from sft.distributed_tensor_stats import unique_shard_weight
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


def unpack_anchor_batch(data: TensorDict | dict) -> dict[str, torch.Tensor]:
    """Remove the one-anchor dataloader dimension added for TensorDict."""
    ignored = {
        "global_token_num", "update_lr_scheduler", "use_remove_padding",
        "use_dynamic_bsz", "max_token_len_per_gpu",
        "micro_batch_size_per_gpu", "use_fused_kernels",
    }
    result = {}
    for key in data.keys():
        if key in ignored:
            continue
        value = data[key]
        if torch.is_tensor(value):
            if value.ndim == 0 or value.shape[0] != 1:
                raise ValueError(
                    f"Action Head batch tensor {key!r} must have leading anchor dimension 1; "
                    f"got {tuple(value.shape)}")
            result[key] = value.squeeze(0)
    return result


def iter_anchor_microbatches(payload, group_size=1):
    """Reuse the B1 multimodal slice and preserve each window's RNG identity."""
    window_anchor = payload['window_anchor_index'].reshape(-1)
    slow_keys = {'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'mm_token_type_ids'}
    anchor_keys = {'noise_seed', 'anchor_valid', 'slow_image_count_per_anchor', 'slow_pixel_count_per_anchor'}
    window_keys = {'reference_visual_tokens', 'current_visual_tokens', 'reference_visual_mask',
                   'current_visual_mask', 'target_actions', 'action_mask', 'window_index_within_anchor'}
    from sft.live_fast_visual import FAST_INPUTS, slice_fast_inputs
    unknown = set(payload) - slow_keys - anchor_keys - window_keys - FAST_INPUTS - {'window_anchor_index'}
    if unknown:
        raise ValueError(f'unsupported anchor microbatch fields: {sorted(unknown)}')
    slow = {key: value for key, value in payload.items() if key in slow_keys}
    if group_size < 1:
        raise ValueError('anchor group size must be positive')
    for index in range(0, payload['input_ids'].shape[0], group_size):
        end = min(index + group_size, payload['input_ids'].shape[0])
        selected = (window_anchor >= index) & (window_anchor < end)
        if end == index + 1:
            batch = Qwen36DualVlnActionHead._slice_slow_anchor(
                slow, index, payload['slow_image_count_per_anchor'].reshape(-1),
                payload['slow_pixel_count_per_anchor'].reshape(-1))
        else:
            batch = {key: value[index:end] for key, value in slow.items()
                     if key in ('input_ids', 'attention_mask', 'mm_token_type_ids')}
            for key, count_key in (('pixel_values', 'slow_pixel_count_per_anchor'),
                                   ('image_grid_thw', 'slow_image_count_per_anchor')):
                counts = payload[count_key].reshape(-1)
                start_row, end_row = int(counts[:index].sum()), int(counts[:end].sum())
                if key in slow:
                    batch[key] = slow[key][start_row:end_row]
        batch.update({key: value[index:end] for key, value in payload.items() if key in anchor_keys})
        batch.update({key: value[selected] for key, value in payload.items() if key in window_keys})
        if FAST_INPUTS & payload.keys():
            if not FAST_INPUTS <= payload.keys():
                raise ValueError('incomplete live fast image microbatch')
            batch.update(slice_fast_inputs(payload, selected))
        batch['window_anchor_index'] = window_anchor[selected] - index
        yield batch


def forward_backward_anchor_microbatches(model, payload, forward_only=False):
    """One global window denominator and one eventual optimizer update.

    Mirrors VERL's microbatch loop with our anchor/window collator contract.
    Only detached health metrics survive each backward, freeing its graph.
    """
    window_anchor = payload['window_anchor_index'].reshape(-1)
    denominator = payload['anchor_valid'].reshape(-1).index_select(0, window_anchor).sum().float()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(denominator)
    totals = {}
    total_loss = denominator.new_zeros(())
    velocity_sum = denominator.new_zeros(())
    cond_sum = denominator.new_zeros(())
    physical_windows = 0
    group_size = 1
    if os.environ.get('ACTION_HEAD_SLOW_BATCH_MODE') == 'packed':
        from sft.selective_checkpointing import shared_token_budget
        group_size = int(os.environ.get('ACTION_HEAD_PACK_ANCHORS', '2'))
        if group_size < 1:
            raise ValueError('ACTION_HEAD_PACK_ANCHORS must be positive')
        lengths = payload['attention_mask'].sum(-1) + model.query_encoder.query_count
        largest = max(int(lengths[i:i + group_size].sum()) for i in range(0, len(lengths), group_size))
        largest = shared_token_budget(largest, lengths.device)
        limit = int(os.environ.get('ACTION_HEAD_PACK_MAX_TOKENS', '12000'))
        if largest > limit:
            group_size = 1  # Same decision on every FSDP rank; never drop data.
        if int(os.environ.get('RANK', '0')) == 0:
            print({'event': 'action_head_packed_budget', 'max_tokens': largest,
                   'limit': limit, 'group_size': group_size}, flush=True)
    for batch in iter_anchor_microbatches(payload, group_size):
        # Integer count survives FSDP mixed-precision input casting exactly.
        output = model(**batch, compute_metrics=forward_only, global_window_count=denominator.to(torch.int64))
        if not torch.isfinite(output['loss']):
            raise RuntimeError('nonfinite anchor microbatch loss')
        if not forward_only:
            output['loss'].backward()
        total_loss += output['loss'].detach()
        velocity_sum += output['local_velocity_mean'].detach() * output['local_window_count'].detach()
        count = batch['target_actions'].shape[0]
        cond_sum += output['local_cond_norm_mean'].detach() * count
        physical_windows += count
        for key in ('local_loss_sum', 'local_window_count', 'local_ade_sum',
                    'local_fde_sum', 'local_metric_window_count'):
            if key in output:
                totals[key] = totals.get(key, 0) + output[key].detach()
        del output
    totals.update(loss=total_loss, local_velocity_mean=velocity_sum / totals['local_window_count'].clamp_min(1),
                  local_cond_norm_mean=cond_sum / max(1, physical_windows))
    return totals


def validate_frozen_query_trainable_parameters(module) -> dict:
    """Guard the frozen-prefix derivative contract after wrapping/adapters."""
    expected = list(module.trainable_parameters())
    actual = [parameter for parameter in module.parameters() if parameter.requires_grad]
    expected_ids = {id(parameter) for parameter in expected}
    actual_ids = {id(parameter) for parameter in actual}
    backbone = list(module.qwen.parameters())
    if (not expected_ids or expected_ids != actual_ids
            or actual_ids.intersection(id(parameter) for parameter in backbone)):
        raise ValueError("query_only requires exactly the original query/head trainable parameters and a frozen backbone")
    return {
        "trainable_parameter_tensors": len(actual),
        "trainable_numel": sum(parameter.numel() for parameter in actual),
        "frozen_backbone_parameter_tensors": len(backbone),
    }


@EngineRegistry.register(
    model_type="action_head", backend=["fsdp", "fsdp2"], device="cuda")
class ActionHeadFSDPEngine(FSDPEngine):
    """FSDP/FSDP2 engine that trains only query embeddings and the FM head."""

    def _validate_frozen_query_config(self):
        # The persistent FM history cache relies on the same frozen-Qwen B1 contract.
        if (os.environ.get('ACTION_HEAD_FROZEN_QUERY_BACKEND', 'legacy_suffix') != 'query_only'
                and not (os.environ.get('ACTION_HEAD_FM_PREFIX_REUSE', '0') == '1'
                         and os.environ.get('ACTION_HEAD_ROUTER_REPLAY') != 'R3')):
            return
        if (self.model_config.enable_gradient_checkpointing or self._is_lora or self._qat_enabled
                or self.engine_config.strategy != 'fsdp2' or self.ulysses_sequence_parallel_size != 1):
            raise ValueError('query_only requires fsdp2, no gradient checkpointing, LoRA or QAT, and sequence parallel size 1')
        if (os.environ.get('ACTION_HEAD_SLOW_BATCH_MODE', 'sequential') != 'sequential'
                or os.environ.get('ACTION_HEAD_ANCHOR_MICROBATCH', '0') != '1'
                or os.environ.get('ACTION_HEAD_CONDITION_MODE') == 'v5_native'):
            raise ValueError('query_only requires frozen Qwen and sequential B1 anchor microbatches')

    def _build_fsdp_module(self, module):
        if os.environ.get('ACTION_HEAD_LOCAL_REPLICA_LOAD', '0') == '1':
            from sft.local_replica_fsdp import build_with_local_replica_loader
            return build_with_local_replica_loader(self, module, FSDPEngine._build_fsdp_module)
        if os.environ.get('ACTION_HEAD_NATIVE_DDP') == '1':
            if os.environ.get('ACTION_HEAD_CONDITION_MODE') != 'v5_native':
                raise ValueError('native DDP is limited to v5')
            from torch.nn.parallel import DistributedDataParallel
            module.to(torch.cuda.current_device())
            return DistributedDataParallel(module, device_ids=[torch.cuda.current_device()],
                                           broadcast_buffers=False, gradient_as_bucket_view=True)
        return super()._build_fsdp_module(module)

    def _build_module(self):
        self._validate_frozen_query_config()
        if os.environ.get("ACTION_HEAD_CONDITION_MODE") == "v5_native":
            from models.native_context_action_head import NativeContextTrainModel
            return NativeContextTrainModel.from_environment()
        # Reuse VERL's exact HF/Qwen model loader and kernel patching.  The
        # parent loader dispatches on model_type, so expose the underlying
        # language model only for that call.
        original_type = self.model_config.model_type
        self.model_config.model_type = "language_model"
        try:
            qwen = super()._build_module()
        finally:
            self.model_config.model_type = original_type

        query_path = os.environ.get("ACTION_HEAD_QUERY_CHECKPOINT", "")
        head_path = os.environ.get("ACTION_HEAD_HEAD_CHECKPOINT", "")
        if not query_path or not head_path:
            raise RuntimeError(
                "ACTION_HEAD_QUERY_CHECKPOINT and ACTION_HEAD_HEAD_CHECKPOINT are required")
        hidden_size = int(qwen.config.text_config.hidden_size)
        query = DualVlnQueryEncoder.from_pretrained(query_path)
        if query.hidden_size != hidden_size:
            raise ValueError(
                f"query hidden size {query.hidden_size} != Qwen hidden size {hidden_size}")
        head = Qwen36FastSlowActionHead.from_pretrained(head_path)
        if head.config.qwen_hidden_size != hidden_size:
            raise ValueError(
                f"FM head hidden size {head.config.qwen_hidden_size} != Qwen hidden size {hidden_size}")
        if query.query_count != head.config.query_num_queries:
            raise ValueError(
                f"query checkpoint count {query.query_count} != FM head count "
                f"{head.config.query_num_queries}")
        fast_visual, fast_backbone = None, ""
        if os.environ.get("ACTION_HEAD_FAST_VISUAL_MODE") == "live":
            from sft.live_fast_visual import load_frozen_fast_tower
            fast_backbone = os.environ.get("ACTION_HEAD_FAST_BACKBONE", "")
            fast_visual = load_frozen_fast_tower(fast_backbone, head.config.visual_hidden_size,
                device="cpu", dtype=torch.bfloat16)
        module = Qwen36DualVlnActionHead(qwen, query, head,
            fast_visual=fast_visual, fast_visual_backbone=fast_backbone)
        # Several VERL helpers inspect ``config`` on the root module.
        module.config = qwen.config
        # FSDP2 selects transformer wrap targets from the root module.  Keep
        # Qwen's native text/vision block classes visible through our wrapper.
        module._no_split_modules = list(qwen._no_split_modules)
        return module

    def _build_optimizer(self, module):
        from verl.workers.config.optimizer import build_optimizer

        if (os.environ.get('ACTION_HEAD_FROZEN_QUERY_BACKEND', 'legacy_suffix') == 'query_only'
                or (os.environ.get('ACTION_HEAD_FM_PREFIX_REUSE', '0') == '1'
                    and os.environ.get('ACTION_HEAD_ROUTER_REPLAY') != 'R3')):
            self.frozen_query_trainable_contract = validate_frozen_query_trainable_parameters(module)
        trainable = [parameter for parameter in module.parameters()
                     if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("Action Head engine found no trainable parameters")
        return build_optimizer(trainable, self.optimizer_config)

    def forward_backward_batch(self, data: TensorDict, loss_function,
                               forward_only: bool = False):
        del loss_function  # The model owns the Wan-style FM objective.
        payload = unpack_anchor_batch(data.to(
            torch.cuda.current_device(), non_blocking=os.environ.get('ACTION_HEAD_PIN_MEMORY') == '1'))
        resident_limit = int(os.environ.get('ACTION_HEAD_FSDP_RESIDENT_MAX_TOKENS', '0'))
        if resident_limit:
            if resident_limit < 0 or self.engine_config.reshard_after_forward:
                raise ValueError('bounded residency requires a positive token limit and reshard_after_forward=false')
            from torch.distributed.fsdp import FSDPModule
            from sft.selective_checkpointing import shared_token_budget
            if not isinstance(self.module, FSDPModule) or not isinstance(self.module, Qwen36DualVlnActionHead):
                raise ValueError('bounded residency requires the FSDP2 Qwen/query training model')
            # Match the existing checkpoint guard's conservative padded budget.
            # All ranks must choose the same collective/reshard schedule.
            tokens = payload['input_ids'].numel() + payload['input_ids'].shape[0] * self.module.query_encoder.query_count
            tokens = shared_token_budget(tokens, payload['input_ids'].device)
            reshard = tokens > resident_limit
            if getattr(self, '_bounded_reshard', None) != reshard:
                for module in self.module.modules():
                    if isinstance(module, FSDPModule) and module is not self.module:
                        module.set_reshard_after_forward(reshard, recurse=False)
                self._bounded_reshard = reshard
            if self.rank == 0:
                print({'event': 'action_head_bounded_residency', 'global_max_tokens': tokens,
                       'max_tokens': resident_limit, 'reshard_after_forward': reshard}, flush=True)
        context = torch.no_grad() if forward_only else nullcontext()
        started = time.perf_counter()
        # Measure each validation/train batch independently. This is more
        # actionable for batch-size planning than the static post-FSDP number.
        torch.cuda.reset_peak_memory_stats()
        print({"event": "action_head_forward_start", "rank": self.rank,
               "forward_only": forward_only}, flush=True)
        with context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            microbatch = os.environ.get('ACTION_HEAD_ANCHOR_MICROBATCH', '0') == '1'
            if microbatch:
                output = forward_backward_anchor_microbatches(self.module, payload, forward_only)
            else:
                output = self.module(**payload, compute_metrics=forward_only)
            loss = output["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite Action Head loss: {loss.detach()}")
            if not forward_only and not microbatch:
                print({"event": "action_head_backward_start", "rank": self.rank,
                       "forward_seconds": time.perf_counter() - started}, flush=True)
                loss.backward()
        gib = float(1024 ** 3)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print({
            "event": "action_head_step_compute_done",
            "rank": self.rank,
            "forward_only": forward_only,
            "seconds": time.perf_counter() - started,
            "memory_allocated_gb": torch.cuda.memory_allocated() / gib,
            "memory_reserved_gb": torch.cuda.memory_reserved() / gib,
            "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / gib,
            "max_memory_reserved_gb": torch.cuda.max_memory_reserved() / gib,
            "device_used_gb": (total_bytes - free_bytes) / gib,
            "device_total_gb": total_bytes / gib,
        }, flush=True)
        local_windows = output["local_window_count"].detach().float().cpu()
        if forward_only:
            # Validation metrics consumed by _fixed_validation's _sum_metric
            # (which handles tensors/lists). Not routed through the base loop's
            # tracking, so the transferqueue list-wrapping is harmless here.
            metrics = {
                "flow_loss_sum": output["local_loss_sum"].detach().float().cpu(),
                "flow_windows": local_windows,
                "ade_sum": output["local_ade_sum"].detach().float().cpu(),
                "fde_sum": output["local_fde_sum"].detach().float().cpu(),
                "metric_windows": output["local_metric_window_count"].detach().float().cpu(),
            }
        else:
            # Live TB metrics written DIRECTLY to a SummaryWriter here, bypassing
            # the base SFTTrainer tracking path: that path serializes the engine
            # return across the transferqueue boundary, which wraps every metric
            # value into a python list, and _TensorboardAdapter.add_scalar then
            # rejects lists. Writing here (rank-0, post all_reduce) keeps the
            # per-step diagnostics in live TB. The metrics dict returned to the
            # base loop is empty, so the base loop only logs its own
            # train/global_tokens / train/total_tokens (floats, console-safe).
            extra = self._extra_train_metrics(output, loss)
            self._log_train_metrics_tb(extra)
            metrics = {}
        return {
            "model_output": {},
            "loss": loss.detach().float().cpu(),
            "metrics": metrics,
        }

    def optimizer_step(self):
        norm = super().optimizer_step()
        if (os.environ.get('ACTION_HEAD_HPARAM_METRICS') == '1'
                and torch.distributed.get_rank() == 0
                and getattr(self, '_tb_writer', None) is not None):
            step = self._train_step
            self._tb_writer.add_scalar('train/grad_norm', norm, step)
            self._tb_writer.add_scalar('train/clip_triggered', float(norm > self.optimizer_config.clip_grad), step)
            self._tb_writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], step)
            self._tb_writer.flush()
        return norm

    def _log_train_metrics_tb(self, extra: dict) -> None:
        """Rank-0 only: write the per-step FM diagnostics to a live SummaryWriter.

        Lazily creates a persistent writer against TENSORBOARD_DIR and tracks a
        step counter (the engine is not told the global step by the base loop;
        for a fresh, non-resume run the counter matches global_step 1:1).
        """
        # Rank 0 writes (after _extra_train_metrics already ran all_reduce on
        # every rank, so the collectives are complete; no deadlock). For this
        # DP-only engine global rank 0 == DP rank 0.
        if torch.distributed.get_rank() != 0:
            return
        if not hasattr(self, "_tb_writer"):
            tb_dir = os.environ.get("TENSORBOARD_DIR")
            if not tb_dir:
                self._tb_writer = None
            else:
                from torch.utils.tensorboard import SummaryWriter
                # Per-run subdir (experiment_name = basename(OUTPUT_DIR)) so a
                # shared TENSORBOARD_DIR root lists runs by name, not ".".
                run_name = os.path.basename(os.environ.get("ACTION_HEAD_OUTPUT_DIR", "run"))
                self._tb_writer = SummaryWriter(os.path.join(tb_dir, run_name))
        if self._tb_writer is None:
            return
        self._train_step = getattr(
            self, "_train_step",
            int(os.environ.get("ACTION_HEAD_STEP_OFFSET", "0"))) + 1
        step = self._train_step
        for key, value in extra.items():
            scalar = float(value.item()) if torch.is_tensor(value) else float(value)
            self._tb_writer.add_scalar(key, scalar, step)
        self._tb_writer.flush()

    def _extra_train_metrics(self, output: dict, loss: torch.Tensor) -> dict[str, torch.Tensor]:
        """Global FM diagnostics derived from the existing forward output.

        All values are reduced across the data-parallel group so rank-0 logs a
        global, not per-shard, value. Nothing here runs a new forward or
        backward; it only reduces tensors the model already computed. Returned
        tensors are written to TensorBoard directly by _log_train_metrics_tb
        (bypassing the list-wrapping transferqueue path).
        """
        from torch.distributed.tensor import DTensor
        if os.environ.get('ACTION_HEAD_NATIVE_LIGHT_METRICS') == '1':
            # One collective and one host copy; loss/window counts stay exact.
            values = torch.stack([output['local_loss_sum'], output['local_window_count'],
                                  output['local_velocity_mean'], output['local_cond_norm_mean']]).detach().float()
            torch.distributed.all_reduce(values)
            values = values.cpu()
            world = torch.distributed.get_world_size()
            return {'train/loss': values[0]/values[1].clamp_min(1),
                    'train/flow_loss': values[0]/values[1].clamp_min(1),
                    'train/valid_windows': values[1],
                    'train/action_magnitude': values[2]/world,
                    'train/cond_norm': values[3]/world}
        # Reduce on-GPU: the DP group backend is the PPU/CUDA device, so a CPU
        # tensor has "No backend type associated with device type cpu". Keep
        # tensors on-device through all_reduce, then .cpu()/.item() for logging.
        group = self.get_data_parallel_group()

        def _sum(t: torch.Tensor) -> torch.Tensor:
            t = t.detach().float()
            if group is not None:
                torch.distributed.all_reduce(
                    t, op=torch.distributed.ReduceOp.SUM, group=group)
            return t.cpu()

        def _avg(t: torch.Tensor) -> torch.Tensor:
            t = t.detach().float()
            if group is not None:
                torch.distributed.all_reduce(
                    t, op=torch.distributed.ReduceOp.AVG, group=group)
            return t.cpu()

        loss_sum = _sum(output["local_loss_sum"])
        win_cnt = _sum(output["local_window_count"])
        flow_loss = loss_sum / win_cnt.clamp_min(1.0)          # per-window FM MSE
        velocity_mean = _avg(output["local_velocity_mean"])
        cond_norm = _avg(output["local_cond_norm_mean"])
        loss_avg = _avg(loss)                                  # backward loss, global mean

        # params_norm: global L2 of every trainable parameter (values); and
        # num_zeros: count of gradient elements that are exactly zero (a
        # sentinel for a degenerate / unused latent query). Both reduce over
        # already-materialized params/grads; no forward.
        device = torch.cuda.current_device()
        pnorm_sq = torch.zeros((), dtype=torch.float64, device=device)
        nzero = torch.zeros((), dtype=torch.float64, device=device)
        world_size = torch.distributed.get_world_size(group) if group is not None else 1
        for parameter in self.module.parameters():
            if not parameter.requires_grad:
                continue
            local = (parameter.to_local() if isinstance(parameter, DTensor)
                     else parameter)
            weight = unique_shard_weight(parameter, world_size)
            pnorm_sq += local.detach().float().double().pow(2).sum() * weight
            grad = parameter.grad
            if grad is None:
                nzero += local.numel() * weight
                continue
            local_grad = (grad.to_local() if isinstance(grad, DTensor) else grad)
            nzero += (local_grad.detach().float() == 0).sum().double() * weight
        if group is not None:
            torch.distributed.all_reduce(
                pnorm_sq, op=torch.distributed.ReduceOp.SUM, group=group)
            torch.distributed.all_reduce(
                nzero, op=torch.distributed.ReduceOp.SUM, group=group)
        # All values are 0-dim tensors, written to TB directly by the caller.
        return {
            "train/loss": loss_avg,
            "train/flow_loss": flow_loss,
            "train/valid_windows": win_cnt,
            "train/action_magnitude": velocity_mean,
            "train/cond_norm": cond_norm,
            "train/params_norm": pnorm_sq.sqrt().cpu(),
            "train/num_zeros": nzero.cpu(),
        }
