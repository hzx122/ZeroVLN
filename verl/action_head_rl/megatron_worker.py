"""Megatron text updates with the existing selective FSDP query/FM replay.

Both engines share the same GPU pool in phases. The FSDP Qwen copy only
conditions FM/query derivatives; its parameters never receive optimizer steps.
"""
from __future__ import annotations

import json
import gc
import os
from pathlib import Path
import shutil
import time

import torch

from action_head_rl.worker import ActionHeadGRPOWorker, text_record_to_tensordict, _cpu_model_positions
from action_head_rl.text_policy import is_text_record
from action_head_rl.megatron_joint import reduced_fm_norm, step_fm, refresh_fm_condition
from action_head_rl.megatron_trajectory_text import create_text_engine, build_logical_batch, loss_function
from action_head_rl.joint_optimizer import step_text_with_fm_norm, combined_clip
from action_head_rl.workload_metrics import sum_workload_metrics
from verl.single_controller.base.decorator import Dispatch, register


class ActionHeadMegatronWorker(ActionHeadGRPOWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset(self):
        import megatron.core, megatron.bridge
        environment = Path(os.environ['ACTION_HEAD_MEGATRON_ENV']).resolve()
        for module in (megatron.core, megatron.bridge):
            if not Path(module.__file__).resolve().is_relative_to(environment):
                raise RuntimeError('Megatron resolved outside the bound container checkout')
        super().reset()
        self._train_action_head = os.environ.get('ACTION_HEAD_TRAIN_ACTION_HEAD', '1') != '0'
        self._train_text_policy = os.environ.get('ACTION_HEAD_TRAIN_TEXT_POLICY', '1') != '0'
        if not self._train_action_head:
            self.engine.module.query_encoder.requires_grad_(False)
            self.engine.module.action_head.requires_grad_(False)
        self.engine.to(device='cpu', model=True, optimizer=True, grad=True)
        torch.cuda.empty_cache()
        from sft.probe_qwen36_megatron import expose_transformer_engine_pytorch
        expose_transformer_engine_pytorch()
        options = json.loads(os.environ['ACTION_HEAD_MEGATRON_CONFIG'])
        self.text_engine = create_text_engine(model_path=self.model_config.path,
            learning_rate=float(os.environ['ACTION_HEAD_VLM_LR']),
            total_steps=self.optimizer_config.total_training_steps,
            weight_decay=self.optimizer_config.weight_decay, **options)
        self.text_engine.initialize()
        self.text_engine.prepare_text_payload = self._prepare_text_payload
        self.text_engine.checkpoint_mananager.use_dist_checkpointing = True
        self.text_engine.to(device='cpu', model=True, optimizer=True, grad=True)
        torch.cuda.empty_cache()
        self._fm_condition_ready = True

    def _prepare_text_payload(self, record):
        from sft.verl_action_head_engine import unpack_anchor_batch
        payload = unpack_anchor_batch(text_record_to_tensordict(
            record, self.model_config.processor, 'cpu'))
        return payload, _cpu_model_positions(self.engine.module.qwen.model, payload)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def megatron_placement(self):
        return dict(rank=torch.distributed.get_rank(),
            text_dp_rank=self.text_engine.get_data_parallel_rank(),
            text_dp_size=self.text_engine.get_data_parallel_size())

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def audit_update_backends(self, input_directory, output_directory):
        from action_head_rl.backend_audit import audit_backends
        return audit_backends(self, input_directory, output_directory)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def audit_learning_rates(self, action_head_lr, vlm_lr, train_scope):
        report = super().audit_learning_rates(action_head_lr, vlm_lr, train_scope)
        groups = self.text_engine.optimizer.param_groups
        if any(float(g['lr']) != float(vlm_lr) or float(g.get('wd_mult', 1.)) != 1.
               or float(g['weight_decay']) != float(self.optimizer_config.weight_decay) for g in groups):
            raise ValueError('Megatron AdamW groups differ from the original v8 optimizer')
        report['text_backend'] = 'megatron'
        report['text_learning_rate'] = float(vlm_lr)
        return report

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def offload_for_colocated_rollout(self, selected_ranks):
        if set(selected_ranks) != set(range(torch.distributed.get_world_size())):
            raise ValueError('Megatron architecture requires the complete shared GPU pool')
        # MCore offload resizes DDP grad storage to zero while retaining its
        # logical shape. reset() already offloads it, so zero_grad_buffer here
        # would launch a kernel against freed storage. Restore allocates the
        # buffer again; train_record_batches clears it before every backward.
        self.text_engine.to(device='cpu', model=True, optimizer=True, grad=True)
        return super().offload_for_colocated_rollout(selected_ranks)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def train_record_batches(self, records, logical_valid):
        if not self._fm_condition_ready:
            raise RuntimeError('previous updated Qwen was not published to the FM condition mirror')
        mixed = any('update_policy_version' in r for r in records)
        source_versions = sorted({r['policy_version'] for r in records})
        if mixed and (not all('update_policy_version' in r for r in records)
                or any(not 0 <= r['policy_version'] <= r['update_policy_version'] for r in records)):
            raise ValueError('partial update requires explicit target and actual source versions on every record')
        versions = {r.get('update_policy_version', r['policy_version']) for r in records}
        if len(versions) != 1 or logical_valid <= 0:
            raise ValueError('joint update requires a single rollout policy and original denominator')
        version = next(iter(versions))
        text = [r for r in records if is_text_record(r)]
        fm = ([r for r in records if not is_text_record(r)]
              if self._train_action_head else [])
        if not text:
            raise ValueError('each Megatron replica needs an aligned text batch')
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        if self._train_action_head:
            self.engine.optimizer_zero_grad()
        # Build action derivatives before changing their Qwen condition. Keep
        # them resident; native MCore CPU offload discards text grad buffers.
        outputs, groups, batches = [], [], []
        if fm:
            with self.engine.train_mode(disable_auto_offload=True, zero_grad_on_exit=False):
                outputs, groups, batches = self._backward_record_batches(fm, logical_valid)
        fm_norm = reduced_fm_norm(self.engine) if fm else 0.
        fm_seconds = time.perf_counter() - started
        record_count, condition_forwards = len(records), len(batches)
        condition_reuse_chunks = len(fm) - len(groups)
        # Preserve existing shape-only counters before releasing FM payloads.
        # They distinguish real prefix work from cross-rank padding without
        # adding device synchronization or changing the training denominator.
        fm_workload = sum_workload_metrics(output['metrics'] for output in outputs)
        # FM backward is complete. Its parameter gradients remain resident,
        # but neither text nor the joint optimizer needs FM replay inputs.
        # Mutate the consumed list so the outer staged-call frame also drops
        # its references; clearing only `fm` leaves the entire shard alive.
        records[:] = text
        self._staged_option_payloads = {}
        self._staged_rollout_payloads = {}
        self._joint_preflight_data = None
        del outputs, groups, batches, fm
        gc.collect()
        self._completed_learner_work('fm_gradient_ready', force=True, policy_version=version)
        text_started = time.perf_counter()
        if self._train_text_policy:
            batch = build_logical_batch([(r, float(r.get('sample_valid', 1.))) for r in text],
                logical_weight=logical_valid,
                dynamic_bsz=self.text_engine.engine_config.use_dynamic_bsz,
                max_token_len_per_gpu=self.text_engine.engine_config.max_token_len_per_gpu)
            self._completed_learner_work('text_batch_prepared', force=True,
                                         policy_version=version, trajectories=len(text))
            self.text_engine.to(device=self.device_name, model=True, optimizer=True, grad=True)
            with self.text_engine.train_mode(disable_auto_offload=True):
                self.text_engine.optimizer_zero_grad()
                self._completed_learner_work('text_backward_started', force=True, policy_version=version)
                self.text_engine.forward_backward_batch(batch, loss_function=loss_function)
                self._completed_learner_work('text_backward_completed', force=True, policy_version=version)
                joint = step_text_with_fm_norm(self.text_engine, fm_grad_norm=fm_norm,
                    max_norm=self.optimizer_config.clip_grad,
                    rollout_policy_version=version, fm_gradient_policy_version=version)
        else:
            norm, factor = combined_clip(0., fm_norm, self.optimizer_config.clip_grad)
            joint = dict(text_grad_norm=0., fm_grad_norm=fm_norm, grad_norm=norm,
                         clip_factor=factor, text_optimizer_steps=0, policy_version=version)
        if self._train_action_head:
            step_fm(self.engine, clip_factor=joint['clip_factor'])
        self._fm_condition_ready = False
        torch.cuda.synchronize()
        self._completed_learner_work('joint_update_completed', force=True, policy_version=version)
        return dict(rank=torch.distributed.get_rank(), optimizer_steps=1,
            text_policy_training=self._train_text_policy,
            action_head_training=self._train_action_head,
            action_head_optimizer_steps=int(self._train_action_head),
            partial_behavior_correction=mixed, rollout_source_versions=source_versions,
            logical_valid_chunks=logical_valid, microbatches=record_count,
            fm_backward_seconds=fm_seconds, text_update_seconds=time.perf_counter()-text_started,
            condition_forwards=condition_forwards, condition_reuse_chunks=condition_reuse_chunks,
            fm_workload=fm_workload,
            lr=float(self.engine.optimizer.param_groups[0]['lr']),
            step_peak_memory_allocated_gb=torch.cuda.max_memory_allocated()/1024**3,
            **joint)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def prepare_native_publication(self):
        # Native Bridge reloads model weights with load_grad=False for export.
        # Release optimizer + DDP grad storage before the colocated vLLM weights
        # wake up; keep the FM module on-device for the same-stream mirror refresh.
        self.text_engine.to(device='cpu', model=True, optimizer=True, grad=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return torch.distributed.get_rank()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def publish_policy(self, output_dir, step, export_qwen, *, transport_only=False,
                       qwen_export_dtype='preserve', bf16_gather_before_export=False,
                       native_targets=None, native_bucket_mb=4096):
        if (not export_qwen and native_targets is None) or bf16_gather_before_export:
            raise ValueError('Megatron uses its native Bridge HF publication')
        from action_head_rl.megatron_policy import publish_megatron_qwen_async
        from action_head_rl.checkpoint import hf_qwen_export_key
        root = Path(output_dir)
        stage_root = root / f'.megatron-publish-{step:06d}'
        staged = stage_root / f'global_step_{step:06d}'
        final = root / staged.name
        # Parent publishes only query/head under a private staging root. No
        # global_step directory is visible to companions before Qwen finishes.
        small = super().publish_policy(str(stage_root), step, False, transport_only=transport_only)
        torch.distributed.barrier()
        expected = {hf_qwen_export_key(k) for k, _ in self.engine.module.qwen.named_parameters()}
        seen = set()
        def consume(name, value):
            refresh_fm_condition(self.engine.module, name, value)
            seen.add(name)
        sender = None
        if native_targets is not None:
            if len(native_targets) != torch.distributed.get_world_size():
                raise ValueError('native publication requires every physical rank')
            from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender
            target = native_targets[torch.distributed.get_rank()]
            # Never overwrite a failed round's socket; fresh attempt owns it.
            if Path(target['handle'].removeprefix('ipc://')).exists():
                raise FileExistsError(target['handle'])
            sender = BucketedWeightSender(zmq_handle=target['handle'],
                bucket_size_mb=native_bucket_mb, use_shm=target['use_shm'])
        report = await publish_megatron_qwen_async(self.text_engine, model_path=self.model_config.path,
            destination=staged/'qwen' if export_qwen else None, processor=self.model_config.processor,
            step=step, tensor_consumer=consume, native_sender=sender)
        if seen != expected:
            raise RuntimeError('incomplete refresh of FM Qwen condition')
        self._fm_condition_ready = True
        if torch.distributed.get_rank() == 0:
            metadata = json.loads((staged/'report.json').read_text())
            metadata.update(qwen_exported=bool(export_qwen), qwen_export_dtype='bf16',
                text_backend='megatron', hf_publication=report)
            (staged/'report.json').write_text(json.dumps(metadata, indent=2)+'\n')
            if final.exists():
                raise FileExistsError('refuse existing combined policy: '+str(final))
            os.replace(staged, final)
            query = root/'current_query.pt'
            temporary = query.with_suffix('.pt.tmp')
            shutil.copy2(final/'dualvln_query/dualvln_query_embeddings.pt', temporary)
            os.replace(temporary, query)
            return dict(checkpoint=str(final), action_head=str(final/'action_head'),
                query=str(query), qwen=str(final/'qwen') if export_qwen else None,
                native_transport=report['transport'] if native_targets is not None else None)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        if hdfs_path:
            raise ValueError('joint state uses the configured shared artifact disk')
        self.engine.save_checkpoint(str(Path(local_path)/'fm'), global_step=global_step,
                                    max_ckpt_to_keep=max_ckpt_to_keep)
        self.text_engine.save_checkpoint(str(Path(local_path)/'text'), global_step=global_step,
                                         max_ckpt_to_keep=max_ckpt_to_keep)
        if torch.distributed.get_rank() == 0:
            (Path(local_path)/'joint_state.json').write_text(json.dumps(dict(
                schema='v8_megatron_fm_joint_v1', step=global_step))+'\n')

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        if hdfs_path or del_local_after_load:
            raise ValueError('joint state must remain on the configured artifact disk')
        manifest = json.loads((Path(local_path)/'joint_state.json').read_text())
        if manifest['schema'] != 'v8_megatron_fm_joint_v1':
            raise ValueError('not a joint Megatron/FM checkpoint')
        self.engine.load_checkpoint(str(Path(local_path)/'fm'), del_local_after_load=False)
        self.engine.to(device='cpu', model=True, optimizer=True, grad=True)
        # Upstream defaults to ignoring the saved scheduler. A continuation
        # must restore it together with Adam rather than restart its counter.
        self.text_engine.checkpoint_mananager.use_checkpoint_opt_param_scheduler = True
        self.text_engine.to(device=self.device_name, model=True, optimizer=True, grad=True)
        from action_head_rl.megatron_checkpoint import load_text_checkpoint
        text_checkpoint = load_text_checkpoint(self.text_engine, Path(local_path)/'text',
            trainable=os.environ.get('ACTION_HEAD_TRAIN_TEXT_POLICY', '1') != '0')
        scheduler = self.text_engine.lr_scheduler.state_dict()
        fm_steps = sorted({int(value['step'].item() if torch.is_tensor(value['step']) else value['step'])
                           for value in self.engine.optimizer.state.values() if 'step' in value})
        self.text_engine.to(device='cpu', model=True, optimizer=True, grad=True)
        torch.cuda.empty_cache()
        self._fm_condition_ready = True
        return dict(rank=torch.distributed.get_rank(), step=manifest['step'],
                    fm_optimizer_steps=fm_steps, text_scheduler=scheduler,
                    text_checkpoint=text_checkpoint,
                    fm_scheduler=self.engine.lr_scheduler.state_dict(),
                    restored=['model', 'optimizer', 'scheduler', 'rng'],
                    allocator_settings=torch.cuda.memory._snapshot().get('allocator_settings', {}))
