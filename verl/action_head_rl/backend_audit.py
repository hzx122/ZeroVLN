"""Zero-update comparison of the two real v8 text backward engines.

Inputs come from complete staged-update archives. Megatron master-gradient
shards are assembled before Bridge conversion; stale regions of main_grad
are never interpreted as reduced gradients. No model parameter is replaced.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import math
import os
from pathlib import Path

import torch

from action_head_rl.checkpoint import hf_qwen_export_key
from action_head_rl.joint_fixture import _sha, validate_fixture_options
from action_head_rl.joint_optimizer import _optimizers
from action_head_rl.text_policy import is_text_record


@contextmanager
def capture_text_input(module, *, key, batch_first, indices, metadata, destination):
    """Observe the first text input and its raw adjoint; never alter either."""
    from action_head_rl.joint_audit import write_new_json
    captured, handles = {}, []
    def before(module, args, kwargs):
        value = kwargs.get(key)
        if captured or value is None:
            return
        if value.ndim != 3 or not value.requires_grad:
            raise ValueError('text boundary audit requires a differentiable single-example input')
        if value.shape[0 if batch_first else 1] != 1:
            raise ValueError('text boundary audit requires microbatch size one')
        flat = value[0] if batch_first else value[:, 0]
        if flat.shape[0] != indices.numel():
            raise ValueError('text boundary shape differs from native CP/SP index layout')
        captured.update(metadata=metadata, indices=indices.cpu().clone(),
                        value=flat.detach().cpu().clone())
        def observe_gradient(gradient):
            flat_gradient = gradient[0] if batch_first else gradient[:, 0]
            if 'gradient' in captured:
                raise ValueError('first text input unexpectedly traversed backward twice')
            captured['gradient'] = flat_gradient.detach().cpu().clone()
            # Returning None preserves the original autograd gradient.
        handles.append(value.register_hook(observe_gradient))
    handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
    try:
        yield
        if 'gradient' not in captured:
            raise ValueError('text input/gradient boundary was not observed')
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            torch.save(captured, stream)
        write_new_json(destination.with_suffix('.json'), dict(
            status='complete', metadata=metadata, rows=indices.numel(),
            tensor_file=str(destination), sha256=_sha(destination)))
    finally:
        for handle in handles:
            handle.remove()


def parameter_family(name):
    if name.startswith('model.visual.'):
        return 'vision'
    for fragment, label in (('.mlp.experts.', 'moe_experts'),
                            ('.mlp.shared_expert', 'shared_expert'),
                            ('.mlp.gate.', 'router'), ('.linear_attn.', 'gdn'),
                            ('.self_attn.', 'attention')):
        if fragment in name:
            return label
    return 'embeddings_head_and_norms'


def compare_text_boundaries(reference, candidates):
    """Restore native CP/SP rows for diagnosis, without changing the parity gate.

    Raw input adjoints precede backend-specific parameter reduction. Report a
    fitted scale separately; it must never replace the unscaled gradient gate.
    """
    from action_head_rl.joint_audit import tensor_difference
    metadata = reference['metadata']
    length = metadata['real_tokens']
    if length <= 0 or not torch.equal(reference['indices'], torch.arange(length)):
        raise ValueError('reference boundary must contain the complete ordered sequence')
    shape = reference['value'].shape
    if len(shape) != 2 or shape[0] != length:
        raise ValueError('invalid reference boundary shape')
    assembled = {key: torch.empty_like(reference[key]) for key in ('value', 'gradient')}
    seen = torch.zeros(length, dtype=torch.bool)
    partitions = set()
    padding_rows = 0
    for row in candidates:
        info = row['metadata']
        if any(info[key] != metadata[key] for key in ('condition_id', 'real_tokens', 'text_dp_rank')):
            raise ValueError('boundary captures describe different trajectory/DP inputs')
        partition = (info['cp_rank'], info['tp_rank'])
        if partition in partitions:
            raise ValueError('duplicate CP/SP partition')
        partitions.add(partition)
        indices = row['indices']
        if indices.ndim != 1 or indices.dtype != torch.long or bool(((indices < -1) | (indices >= length)).any()):
            raise ValueError('invalid native CP/SP token indices')
        valid = indices >= 0
        selected = indices[valid]
        if selected.unique().numel() != selected.numel() or bool(seen[selected].any()):
            raise ValueError('duplicate original token in CP/SP boundary')
        for key in assembled:
            value = row[key]
            if value.shape != (indices.numel(), shape[1]) or value.dtype != reference[key].dtype:
                raise ValueError('boundary tensor shape/dtype mismatch')
            if not torch.isfinite(value).all():
                raise ValueError('nonfinite boundary tensor')
            assembled[key][selected] = value[valid]
        seen[selected] = True
        padding_rows += int((~valid).sum())
    if not bool(seen.all()):
        raise ValueError('incomplete CP/SP boundary token coverage')
    result = dict(condition_id=metadata['condition_id'], real_tokens=length,
                  partitions=len(partitions), padding_rows=padding_rows,
                  scope='raw boundary diagnosis only; not a parameter-gradient parity verdict')
    for key, candidate in assembled.items():
        old = reference[key]
        if old.shape != shape or not torch.isfinite(old).all():
            raise ValueError('invalid reference boundary tensor')
        report = tensor_difference(old, candidate, atol=0., rtol=0.)
        report['exactly_equal'] = report.pop('passed')
        a, b = old.double(), candidate.double()
        aa, bb, ab = float(a.square().sum()), float(b.square().sum()), float((a*b).sum())
        report.update(reference_norm=math.sqrt(aa), candidate_norm=math.sqrt(bb),
                      cosine=ab / max(math.sqrt(aa*bb), 1e-30))
        if key == 'gradient':
            scale = ab / bb if bb else 0.
            report.update(diagnostic_candidate_to_reference_scale=scale,
                diagnostic_scaled_relative_l2=float((a-scale*b).norm()) / max(math.sqrt(aa), 1e-30))
        result[key] = report
    return result


def gradient_mapping(mapping):
    """Use the Jacobian of the pinned Qwen mappings, excluding norm offsets."""
    name = type(mapping).__name__
    allowed = {'AutoMapping', 'ReplicatedMapping', 'QKVMapping', 'ConcatenatedQKVMapping',
               'GDNConv1dMapping', 'GDNLinearMappingSeparate', 'GatedMLPMapping',
               'FusedExpertMapping', 'FusedGatedExpertMapping'}
    if name == 'RMSNorm2ZeroCenteredRMSNormMapping':
        result = copy.copy(mapping)
        # Its only extra operation is HF gamma = MCore gamma + 1. Applying
        # that weight offset to gradients is wrong; subtracting it afterwards
        # also erases tiny FP32 derivatives. The AutoMapping parent keeps the
        # exact TP/PP ownership/permutation without ever adding the constant.
        result.megatron_to_hf = super(type(mapping), result).megatron_to_hf
        return result
    if name not in allowed:
        raise ValueError('unaudited parameter-to-gradient mapping: ' + name)
    return mapping


def load_update_shard(root, rank, world_size, dp_rank, dp_size):
    validate_fixture_options(root, 1, 0)
    directory = Path(root) / f'rank_{rank:03d}'
    from action_head_rl.joint_audit import read_json
    report = read_json(directory / 'complete.json')
    path = directory / 'records.pt'
    if (report['schema'] != 'v8-complete-update-replay-v1'
            or report['rank'] != rank or report['world_size'] != world_size
            or report['policy_version'] != 0
            or report['layout'] != dict(text_backend='megatron',
                text_dp_rank=dp_rank, text_dp_size=dp_size)
            or report['input_sha256'] != _sha(path)):
        raise ValueError('backend audit requires the unchanged initial-policy update shard')
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if saved['logical_weight'] != report['logical_weight']:
        raise ValueError('original update denominator changed')
    if len(saved['records']) != report['records']:
        raise ValueError('update archive lost records')
    return saved, report


def reduced_gradient_tasks(engine):
    """Build independent Bridge tasks from prepared optimizer FP32 shards.

    The ranges and DP groups are the native distributed optimizer's own
    ownership map. Zero padding is explicit. All-reduce sums disjoint shards,
    not replicated full gradients, so this introduces no averaging factor.
    """
    dist = torch.distributed
    gradients = {}
    for optimizer in _optimizers(engine.optimizer):
        if getattr(optimizer, 'is_stub_optimizer', False):
            continue
        if (not optimizer.ddp_config.use_distributed_optimizer
                or getattr(optimizer.ddp_config, 'num_distributed_optimizer_instances', 1) != 1
                or getattr(optimizer.config, 'use_precision_aware_optimizer', False)):
            raise ValueError('audit supports ordinary single-instance distributed Adam only')
        buffers = {(i, j): torch.zeros_like(bucket.grad_data, dtype=torch.float32)
                   for i, buffer in enumerate(optimizer.buffers)
                   for j, bucket in enumerate(buffer.buckets)}
        pairs = [(optimizer.model_float16_groups, optimizer.shard_fp32_from_float16_groups),
                 (optimizer.model_fp32_groups, optimizer.shard_fp32_groups)]
        for model_groups, main_groups in pairs:
            for model_group, main_group in zip(model_groups, main_groups, strict=True):
                for model, main in zip(model_group, main_group, strict=True):
                    if main.grad is None or main.grad.dtype != torch.float32:
                        raise ValueError('prepare_grads must produce FP32 master gradients first')
                    index, _, bucket = optimizer.model_param_gbuf_map[model]
                    span = optimizer._get_model_param_range_map(model)['gbuf_world_in_bucket']
                    buffers[index, bucket][span.start:span.end].copy_(main.grad.reshape(-1))
        for (i, j), full in buffers.items():
            dist.all_reduce(full, group=optimizer.data_parallel_group)
            bucket = optimizer.buffers[i].buckets[j]
            for parameter in bucket.params_list:
                start, end = bucket.param_to_index[parameter]
                if id(parameter) in gradients:
                    raise ValueError('duplicate native gradient ownership')
                gradients[id(parameter)] = full[start:end].view(parameter.shape)
    tasks = []
    for task in engine.bridge.get_conversion_tasks(engine.module):
        if getattr(task, 'export_hook', None) is not None:
            raise ValueError('gradient audit cannot apply arbitrary weight export hooks')
        gradient = None
        if task.param_weight is not None:
            if id(task.param_weight) not in gradients:
                raise ValueError('Bridge parameter missing from reduced master gradients: '
                                 + task.global_param_name)
            gradient = gradients[id(task.param_weight)]
        # Bridge's WeightConversionTask is a frozen dataclass. Construct new
        # task descriptors without changing the weight publication directory.
        tasks.append(replace(task, mapping=gradient_mapping(task.mapping),
                             param_weight=gradient, weight_dtype=torch.float32))
    return tasks


def snapshot_fsdp_text(module, *, gradients=True):
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
    saved = {}
    for name, parameter in module.named_parameters():
        name = hf_qwen_export_key(name)
        gradient = parameter.grad if gradients else parameter
        if not isinstance(parameter, DTensor) or parameter.device_mesh.size() != torch.distributed.get_world_size():
            raise ValueError('reference audit requires the full-world FSDP layout')
        shape, offset = compute_local_shape_and_global_offset(
            parameter.shape, parameter.device_mesh, parameter.placements)
        local = None if gradient is None else gradient.to_local().detach().float().cpu().clone()
        if local is not None and tuple(local.shape) != tuple(shape):
            raise ValueError('FSDP gradient slice shape changed')
        saved[name] = dict(value=local, shape=tuple(shape), offset=tuple(offset))
    return saved


def compare_hf_gradients(engine, reference, *, gradients=True, parameter_report=None):
    # Parameter comparison must retain the real weight mappings, including
    # norm offsets. Only gradient comparison uses their Jacobians.
    options = dict(conversion_tasks=reduced_gradient_tasks(engine)) if gradients else {}
    iterator = engine.bridge.export_hf_weights(engine.module,
        cpu=False, show_progress=False, merge_adapter_weights=False, **options)
    seen, names, statistics = set(), [], []
    for name, full in iterator:
        name = hf_qwen_export_key(name)
        if name not in reference or name in seen:
            raise ValueError('gradient conversion keys differ from the FSDP model: ' + name)
        seen.add(name)
        row = reference[name]
        slices = tuple(slice(offset, offset + size) for offset, size in zip(row['offset'], row['shape']))
        candidate = full[slices].float()
        if row['value'] is None:
            # A missing gradient and a zero tensor have different Adam state
            # semantics even when both cause zero change on the first step.
            stats = torch.zeros(6, dtype=torch.float64, device=full.device)
            stats[4] = 1
            names.append(name)
            statistics.append(stats)
            continue
        old = row['value'].to(candidate.device)
        delta = candidate - old
        names.append(name)
        statistics.append(torch.stack([old.double().square().sum(), candidate.double().square().sum(),
            (old.double() * candidate.double()).sum(), delta.double().square().sum(),
            torch.zeros((), dtype=torch.float64, device=full.device),
            (old != candidate).double().sum()]))
    if seen != set(reference):
        raise ValueError('incomplete gradient export')
    table = torch.stack(statistics)
    torch.distributed.all_reduce(table)
    old, new, dot, error, missing, unequal = table.sum(0).tolist()
    details = [dict(name=name, reference_norm=math.sqrt(a), candidate_norm=math.sqrt(b),
                    error_norm=math.sqrt(e), relative_l2=math.sqrt(e/max(a,1e-30)),
                    cosine=d/max(math.sqrt(a*b),1e-30), none_mismatches=int(n),
                    value_mismatches=int(u))
               for name, (a,b,d,e,n,u) in zip(names, table.tolist())]
    families = {}
    for name, row in zip(names, table.tolist()):
        totals = families.setdefault(parameter_family(name), [0.] * 6)
        for index, value in enumerate(row):
            totals[index] += value
    by_family = {name: dict(reference_norm=math.sqrt(a), candidate_norm=math.sqrt(b),
        relative_l2=math.sqrt(e / max(a, 1e-30)),
        cosine=d / max(math.sqrt(a*b), 1e-30), none_mismatches=int(n))
        for name, (a,b,d,e,n,u) in families.items()}
    if parameter_report is not None and torch.distributed.get_rank() == 0:
        from action_head_rl.joint_audit import write_new_json
        write_new_json(parameter_report, dict(parameters=details, by_family=by_family))
    relative = math.sqrt(error / max(old, 1e-30))
    cosine = dot / max(math.sqrt(old * new), 1e-30)
    return dict(reference_norm=math.sqrt(old), candidate_norm=math.sqrt(new),
        relative_l2=relative, cosine=cosine, none_mismatches=int(missing),
        largest_errors=sorted(details, key=lambda row: row['error_norm'], reverse=True)[:20],
        parameters=len(seen), value_mismatches=int(unequal),
        by_family=by_family,
        passed=(old > 0 and relative <= .01 and cosine >= .999 and not int(missing)
                and (gradients or not int(unequal))))


def audit_backends(worker, root, output):
    """Use both production backward paths at fixed weights; never step Adam."""
    from action_head_rl.joint_audit import forbid_updates, write_new_json
    from action_head_rl.megatron_trajectory_text import build_logical_batch, loss_function
    from action_head_rl.worker import text_record_to_tensordict, _cpu_model_positions
    from sft.verl_action_head_engine import unpack_anchor_batch
    dist = torch.distributed
    rank, world = dist.get_rank(), dist.get_world_size()
    engine = worker.text_engine
    saved, binding = load_update_shard(root, rank, world,
        engine.get_data_parallel_rank(), engine.get_data_parallel_size())
    text = [r for r in saved['records'] if is_text_record(r)]
    logical = saved['logical_weight']
    copies = world // engine.get_data_parallel_size()
    if not text or world % engine.get_data_parallel_size():
        raise ValueError('unaligned dense-DP transport')
    from action_head_rl.joint_audit import read_json
    target_dp = read_json(Path(root) / 'rank_000/complete.json')['layout']['text_dp_rank']
    boundary_metadata = dict(condition_id=text[0]['condition_id'],
        real_tokens=text[0]['all_input_ids'].numel(), rank=rank,
        text_dp_rank=engine.get_data_parallel_rank(),
        note='raw input adjoint before backend-specific final parameter reduction/normalization')
    # The archive repeats each dense-DP shard over TP/PP/CP ranks. FSDP
    # treats those ranks as independent replicas; cancel exactly that count.
    # reset() explicitly offloads both engines. The production rollout path
    # restores FSDP (including vision buffers) before backward; this direct
    # replay skips rollout and must perform that same restore explicitly.
    worker.engine.to(device=worker.device_name, model=True, optimizer=True, grad=True)
    initial = snapshot_fsdp_text(worker.engine.module.qwen, gradients=False)
    worker.engine.to(device='cpu', model=True, optimizer=True, grad=True)
    engine.to(device=worker.device_name, model=True, optimizer=True, grad=True)
    with torch.no_grad():
        initial_comparison = compare_hf_gradients(engine, initial, gradients=False)
    initial_comparison['runtime_cache'] = dict(
        initializer=getattr(worker.engine, 'compile_cache_audit', None),
        environment={key: os.environ.get(key) for key in (
            'ACTION_HEAD_PROCESS_COMPILE_CACHE_ROOT', 'TRITON_CACHE_DIR',
            'TORCHINDUCTOR_CACHE_DIR')})
    write_new_json(Path(output) / f'initial_weights_rank_{rank:03d}.json', initial_comparison)
    del initial
    if not initial_comparison['passed']:
        raise RuntimeError('initial HF/FSDP and Megatron weights differ; inspect initial_weights evidence')
    engine.to(device='cpu', model=True, optimizer=True, grad=True)
    worker.engine.to(device=worker.device_name, model=True, optimizer=True, grad=True)
    reference_boundary = (capture_text_input(worker.engine.module.qwen.model.language_model,
        key='inputs_embeds', batch_first=True,
        indices=torch.arange(text[0]['all_input_ids'].numel()),
        metadata=boundary_metadata, destination=Path(output)/'boundary_reference.pt')
        if rank == 0 else nullcontext())
    with forbid_updates(worker.engine), worker.engine.train_mode(disable_auto_offload=True), reference_boundary:
        worker.engine.optimizer_zero_grad()
        reference_outputs, _, _ = worker._backward_record_batches(text, logical * copies)
        reference = snapshot_fsdp_text(worker.engine.module.qwen)
        payloads = [unpack_anchor_batch(text_record_to_tensordict(r, worker.model_config.processor, 'cpu')) for r in text]
        positions = [_cpu_model_positions(worker.engine.module.qwen.model, p) for p in payloads]
    worker.engine.to(device='cpu', model=True, optimizer=True, grad=True)
    torch.cuda.empty_cache()
    batch = build_logical_batch([(r, float(r.get('sample_valid', 1.))) for r in text],
                               payloads, positions, logical_weight=logical)
    engine.to(device=worker.device_name, model=True, optimizer=True, grad=True)
    from verl.utils.megatron_utils import unwrap_model
    native_models = unwrap_model(engine.module)
    if len(native_models) != 1:
        raise ValueError('boundary audit currently requires the existing non-interleaved pipeline')
    native = native_models[0]
    candidate_boundary = nullcontext()
    if native.pre_process and engine.get_data_parallel_rank() == target_dp:
        from verl.models.mcore.util import preprocess_thd_engine
        n = text[0]['all_input_ids'].numel()
        # Use the actual backend's CP partition, including padding. Positive
        # indices distinguish token zero from native zero-filled padding.
        original_indices = torch.nested.nested_tensor(
            [torch.arange(1, n+1, device=worker.device_name)], layout=torch.jagged)
        cp_indices = preprocess_thd_engine(original_indices, pre_process=True, cp_layout='zigzag')[0].reshape(-1)-1
        indices = cp_indices.chunk(native.pg_collection.tp.size())[native.pg_collection.tp.rank()]
        candidate_boundary = capture_text_input(native.language_model,
            key='decoder_input', batch_first=False, indices=indices,
            metadata={**boundary_metadata, 'cp_rank':native.pg_collection.cp.rank(),
                      'tp_rank':native.pg_collection.tp.rank()},
            destination=Path(output)/f'boundary_candidate_rank_{rank:03d}.pt')
    forward_rows = []
    def observed_loss(**kwargs):
        data = kwargs['data']
        ids = data['input_ids'].values().detach().cpu()
        # Match by exact tokens, not pipeline order or a CP rank's ownership.
        matches = [i for i, row in enumerate(text) if torch.equal(ids, row['all_input_ids'].cpu())]
        if len(matches) != 1:
            raise ValueError('forward audit cannot uniquely identify the original trajectory')
        index = matches[0]
        mask = text[index]['text_response_mask'].bool().reshape(-1)
        old = reference_outputs[index]['model_output']['text_logprobs'].reshape(-1)[mask]
        new = kwargs['model_output']['log_probs'].values().detach().float().cpu()[mask]
        difference = new - old
        forward_rows.append(dict(condition_id=text[index]['condition_id'],
            sampled_tokens=int(mask.sum()), reference_logprobs=old.tolist(),
            candidate_logprobs=new.tolist(), max_abs=float(difference.abs().max()),
            mean_abs=float(difference.abs().mean())))
        return loss_function(**kwargs)
    with forbid_updates(engine), engine.train_mode(disable_auto_offload=True), candidate_boundary:
        engine.optimizer_zero_grad()
        engine.forward_backward_batch(batch, loss_function=observed_loss)
        write_new_json(Path(output) / f'forward_rank_{rank:03d}.json',
                       dict(rank=rank, trajectories=forward_rows))
        if engine.optimizer.prepare_grads():
            raise RuntimeError('non-finite candidate gradients')
        native_norm = float(engine.optimizer.get_grad_norm())
        comparison = compare_hf_gradients(engine, reference,
            parameter_report=Path(output)/'gradient_parameters.json')
        comparison['native_candidate_norm'] = native_norm
        comparison['export_norm_matches'] = math.isclose(
            native_norm, comparison['candidate_norm'], rel_tol=1e-4, abs_tol=1e-8)
        comparison['passed'] = comparison['passed'] and comparison['export_norm_matches']
    report = dict(rank=rank, world_size=world, binding=binding,
                  logical_weight=logical, reference_transport_copies=copies,
                  optimizer_updates=0, comparison=comparison)
    write_new_json(Path(output) / f'rank_{rank:03d}.json', report)
    return report
