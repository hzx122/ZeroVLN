"""Opt-in fixed-weight profiling through the production SFT engine.

Uses the real collator, model, diagnostics and optimizer. LR=0 keeps weights
fixed while materializing Adam state. Never writes a training checkpoint.
Synchronized phase timings exclude CPU gradient copies and cold repetitions.
The distributed case compares gradients without any optimizer updates.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import torch
from torch.distributed.tensor import DTensor

from sft.action_head_joint_dataset import ActionHeadAnchorCollator
from sft.query_hidden_fm_impact import _pair


def gradient_snapshot(model, queries, loss):
    values = {'query_gradient': [], 'head_gradient': []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        grad = parameter.grad
        if grad is None:
            raise RuntimeError(f'missing trainable gradient: {name}')
        grad = grad.to_local() if isinstance(grad, DTensor) else grad
        key = 'head_gradient' if 'action_head.' in name else 'query_gradient'
        values[key].append(grad.detach().float().cpu().reshape(-1))
    return {**{key: torch.cat(value) for key, value in values.items()},
            'query_hidden': torch.cat(queries).float().cpu(),
            'loss': loss.detach().float().cpu().reshape(-1)}


def run_distributed_profile(trainer):
    """Two real GBs, fixed weights: compare gradients before choosing a run.

    Uses the production distributed sampler/collator and FSDP2 runtime API.
    This is a compute diagnostic, never a substitute for TB training throughput.
    Writes each rank's results after every completed measurement so an OOM in
    the final memory-intensive candidate does not erase prior evidence.
    """
    from torch.distributed.fsdp import FSDPModule
    from sft.verl_action_head_engine import unpack_anchor_batch
    dist = torch.distributed
    rank = dist.get_rank()
    engine, model = trainer.engine, trainer.engine.module
    if any(parameter.requires_grad for parameter in model.qwen.parameters()):
        raise RuntimeError('distributed profile requires a frozen Qwen backbone')
    output = Path(os.environ['OUTPUT_DIR']) / f'profile_rank_{rank:03d}.json'
    rows = []
    original_slow = model._forward_slow_anchors
    trace_enabled = False
    layer_values, layer_references, trace_handles = {}, {}, []
    if os.environ.get('ACTION_HEAD_PROFILE_PACK_LAYERS') == '1':
        from sft.packed_query_forward import _LAYOUT
        def capture_layer(name):
            def hook(module, inputs, output):
                if not trace_enabled:
                    return
                value = output[0] if isinstance(output, tuple) else output
                layout = _LAYOUT.get()
                count = model.query_encoder.query_count
                value = value[:, -count:] if layout is None else value.index_select(1, layout.query_indices)
                layer_values.setdefault(name, []).append(value.reshape(-1, count, value.shape[-1]).detach().float().cpu())
            return hook
        for index, layer in enumerate(model.qwen.model.language_model.layers):
            for name, module in ((f'{index:02d}_attention', getattr(layer, 'linear_attn', getattr(layer, 'self_attn', None))),
                                 (f'{index:02d}_layer', layer)):
                trace_handles.append(module.register_forward_hook(capture_layer(name)))
    original_gc = [layer.gradient_checkpointing for layer in model.qwen.model.language_model.layers]
    original_env = {key: os.environ.get(key) for key in (
        'ACTION_HEAD_GC_SKIP_EVERY', 'ACTION_HEAD_GC_MAX_TOKENS', 'ACTION_HEAD_FM_BATCH_WINDOWS',
        'ACTION_HEAD_SLOW_BATCH_MODE')}
    queries = []

    def capture(*args, **kwargs):
        value = original_slow(*args, **kwargs)
        queries.append(value.detach())
        return value

    model._forward_slow_anchors = capture
    # Keep FM batching fixed across candidates: its gate already exists.
    os.environ['ACTION_HEAD_FM_BATCH_WINDOWS'] = '1'
    os.environ['ACTION_HEAD_GC_MAX_TOKENS'] = str(int(os.environ.get('ACTION_HEAD_GC_MAX_TOKENS', '32768')))
    iterator = iter(trainer.train_dataloader)
    from contextlib import ExitStack
    suffix_case = os.environ['ACTION_HEAD_LOCAL_PROFILE'] == 'suffix-distributed'
    mode_stack = ExitStack()
    try:
        batches = [next(iterator) for _ in range(2)]
        references, suffix_references = {}, {}
        with engine.train_mode():
            # Finish both low-memory batches before testing full residency.
            modes = ('reference', 'repeat', 'microbatch', 'suffix_microbatch') if suffix_case else (
                'reference', 'repeat', 'gc2', 'resident')
            if suffix_case and os.environ.get('ACTION_HEAD_PROFILE_FUSED_MOE') == '1':
                modes += ('suffix_veomni_microbatch',)
            if suffix_case and os.environ.get('ACTION_HEAD_PROFILE_PACKED') == '1':
                modes = ('reference', 'suffix_microbatch', 'packed_microbatch')
            for mode in modes:
                mode_stack.close()
                is_suffix = mode in ('suffix_microbatch', 'suffix_veomni_microbatch', 'packed_microbatch')
                os.environ['ACTION_HEAD_SLOW_BATCH_MODE'] = 'packed' if mode == 'packed_microbatch' else 'sequential'
                is_microbatch = is_suffix or mode == 'microbatch'
                if is_suffix:
                    from sft.frozen_expert_grouped_mm import query_suffix_backward
                    mode_stack.enter_context(query_suffix_backward(model.query_forward._backbone(), model.query_encoder.query_count,
                        expert_backward='veomni' if mode == 'suffix_veomni_microbatch' else 'hf'))
                for batch_index, cpu_batch in enumerate(batches):
                    # The loader returns the project's leading packed axis.
                    payload = unpack_anchor_batch({k: v.cuda() for k, v in cpu_batch.items()})
                    os.environ['ACTION_HEAD_GC_SKIP_EVERY'] = '2' if mode == 'gc2' else '0'
                    for layer, enabled in zip(model.qwen.model.language_model.layers, original_gc):
                        layer.gradient_checkpointing = enabled if not is_suffix else False
                    for module in model.modules():
                        if isinstance(module, FSDPModule) and module is not model:
                            module.set_reshard_after_forward(mode != 'resident', recurse=False)
                    for repetition in range(2):
                        engine.optimizer_zero_grad()
                        queries.clear()
                        layer_values.clear()
                        trace_enabled = bool(trace_handles) and repetition == 0 and mode in ('suffix_microbatch', 'packed_microbatch')
                        dist.barrier()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        print(json.dumps({'distributed_profile_start': mode, 'rank': rank,
                                          'batch': batch_index, 'repetition': repetition}), flush=True)
                        start = time.perf_counter()
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            if is_microbatch:
                                from sft.verl_action_head_engine import forward_backward_anchor_microbatches
                                result = forward_backward_anchor_microbatches(model, payload)
                            else:
                                result = model(**payload, compute_metrics=False)
                            loss = result['loss']
                            if not is_microbatch:
                                loss.backward()
                        torch.cuda.synchronize()
                        seconds = time.perf_counter() - start
                        if not torch.isfinite(loss):
                            raise RuntimeError('nonfinite distributed profile loss')
                        row = {'rank': rank, 'batch': batch_index, 'mode': mode,
                               'repetition': repetition, 'compute_seconds': seconds,
                               'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                               'tokens': payload['attention_mask'].sum(-1).tolist(),
                               'checkpointed_text_layers': sum(layer.gradient_checkpointing
                                   for layer in model.qwen.model.language_model.layers),
                               'reshard_after_forward': mode != 'resident',
                               'world_size': dist.get_world_size(), 'time_utc': time.time()}
                        if trace_enabled:
                            captured = {name: torch.cat(values) for name, values in layer_values.items()}
                            if mode == 'suffix_microbatch':
                                layer_references[batch_index] = captured
                            else:
                                row['layer_query_parity_vs_suffix'] = {
                                    name: _pair(value, captured[name], dtype=torch.float64)
                                    for name, value in layer_references[batch_index].items()}
                            row['cold_measurement_includes_layer_trace'] = True
                        trace_enabled = False
                        if repetition:
                            current = gradient_snapshot(model, queries, loss)
                            if mode == 'reference':
                                references[batch_index] = current
                            else:
                                row['parity'] = {key: _pair(value, current[key], dtype=torch.float64)
                                                 for key, value in references[batch_index].items()}
                            if mode == 'suffix_microbatch':
                                suffix_references[batch_index] = current
                            elif mode in ('suffix_veomni_microbatch', 'packed_microbatch'):
                                row['parity_vs_suffix'] = {key: _pair(value, current[key], dtype=torch.float64)
                                    for key, value in suffix_references[batch_index].items()}
                            if mode in ('suffix_microbatch', 'packed_microbatch'):
                                # Query embeddings are sharded once across this FSDP world.
                                base = (references if mode == 'suffix_microbatch' else suffix_references)[batch_index]
                                left, right = base['query_gradient'].double(), current['query_gradient'].double()
                                stats = torch.tensor([(left - right).square().sum(), left.square().sum(),
                                                      right.square().sum(), (left * right).sum()],
                                                     device='cuda', dtype=torch.float64)
                                dist.all_reduce(stats)
                                row['full_query_gradient_vs_' + ('reference' if mode == 'suffix_microbatch' else 'suffix')] = {
                                    'relative_l2': float((stats[0] / stats[1].clamp_min(1e-30)).sqrt()),
                                    'cosine': float(stats[3] / (stats[1] * stats[2]).clamp_min(1e-30).sqrt())}
                            row['gradient_parity_scope'] = 'local DTensor shard, not full gradient vector'
                        rows.append(row)
                        output.write_text(json.dumps({'scope': 'fixed weights; no optimizer updates; two real distributed batches',
                            'frozen_qwen_trainable_parameters': 0,
                            'rows': rows}, indent=2) + '\n')
                        del result, loss
                    del payload
    finally:
        for handle in trace_handles:
            handle.remove()
        mode_stack.close()
        engine.optimizer_zero_grad()
        model._forward_slow_anchors = original_slow
        for layer, enabled in zip(model.qwen.model.language_model.layers, original_gc):
            layer.gradient_checkpointing = enabled
        for module in model.modules():
            if isinstance(module, FSDPModule) and module is not model:
                module.set_reshard_after_forward(bool(engine.engine_config.reshard_after_forward), recurse=False)
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_prefix_profile(trainer):
    """Diagnose the existing cached chunk against full causal training.

    Recompute every prefix on every repetition. No persisted features or
    optimizer updates. Layer probes are separate from synchronized timings.
    """
    if torch.distributed.get_world_size() != 1:
        raise ValueError('prefix diagnostic requires one rank')
    engine, model = trainer.engine, trainer.engine.module
    from contextlib import ExitStack
    suffix_only = os.environ['ACTION_HEAD_LOCAL_PROFILE'] == 'suffix'
    mode_stack = ExitStack()
    dataset = trainer.train_dataset.dataset
    collator = ActionHeadAnchorCollator(trainer.model_config.processor, 1, len(dataset))
    forward = model.query_forward
    layers = list(model.qwen.model.language_model.layers)
    original_gc = [layer.gradient_checkpointing for layer in layers]
    checkpoint_modules = [(module, module.gradient_checkpointing)
                          for module in model.qwen.modules() if hasattr(module, 'gradient_checkpointing')]
    full, prefix = forward.forward_full, forward.prefix_embeds
    slow = model._forward_slow_anchors
    queries, rows, comparisons, layer_checks = [], [], [], []
    prefix_dtype = None
    path = Path(os.environ['OUTPUT_DIR']) / 'prefix_profile.json'

    def capture_prefix(*args, **kwargs):
        nonlocal prefix_dtype
        result = prefix(*args, **kwargs)
        prefix_dtype = result[0].dtype
        return result

    def capture_slow(*args, **kwargs):
        value = slow(*args, **kwargs)
        queries.append(value.detach())
        return value

    def cached(inputs, *, matched_dtype, fp32_state=False, aligned=False):
        from contextlib import nullcontext
        from unittest.mock import patch
        from transformers.cache_utils import LinearAttentionLayer
        original_initialize = LinearAttentionLayer.lazy_initialization
        def initialize(layer, conv_states=None, recurrent_states=None):
            original_initialize(layer, conv_states=conv_states, recurrent_states=recurrent_states)
            # HF 5.5.4 initializes recurrent state using the conv state's BF16
            # dtype, losing the FP32 state returned by the FLA chunk kernel.
            if recurrent_states is not None:
                layer.recurrent_states = torch.zeros_like(recurrent_states)
        forward._cached_key = None
        forward._cached_prefix = None
        try:
            context = patch.object(LinearAttentionLayer, 'lazy_initialization', initialize) if fp32_state else nullcontext()
            with context:
                if aligned:
                    from sft.joint_fm_head_sft import _append_positions
                    with torch.no_grad():
                        embeds, mask, positions = forward.prefix_embeds(inputs)
                        boundary = (embeds.shape[1] // 64) * 64
                        if boundary == 0:
                            raise ValueError('aligned diagnostic needs at least 64 prefix tokens')
                        state = forward._backbone()(
                            inputs_embeds=embeds[:, :boundary], attention_mask=mask[:, :boundary],
                            position_ids=None if positions is None else positions[..., :boundary],
                            use_cache=True, return_dict=True).past_key_values
                    joined, joined_mask = forward.query_encoder.append(embeds, mask)
                    joined_positions = _append_positions(positions, forward.query_encoder.query_count)
                    output = forward._backbone()(
                        inputs_embeds=joined[:, boundary:], attention_mask=joined_mask,
                        position_ids=None if joined_positions is None else joined_positions[..., boundary:],
                        past_key_values=state, use_cache=False, return_dict=True)
                    return output.last_hidden_state[:, -forward.query_encoder.query_count:]
                forward.ensure_prefix_cache(inputs, 'current_batch_only')
            return forward.forward_cached_chunk(query_dtype=prefix_dtype if matched_dtype else None)
        finally:
            forward._cached_key = None
            forward._cached_prefix = None

    def save():
        path.write_text(json.dumps({'scope': 'fixed weights; real-time prefix every iteration; no optimizer update',
            'rows': rows, 'comparisons': comparisons, 'layer_checks': layer_checks}, indent=2) + '\n')

    forward.prefix_embeds = capture_prefix
    model._forward_slow_anchors = capture_slow
    try:
        with engine.train_mode():
            for offset in range(min(2, len(dataset))):
                cpu_batch = collator([dataset[offset]])
                payload = {key: value.cuda() for key, value in cpu_batch.items()}
                reference = None
                modes = ('full', 'suffix_gc', 'suffix_no_gc') if suffix_only else (
                    'full', 'cached_original', 'cached_matched_dtype', 'cached_fp32_state', 'cached_aligned')
                for mode in modes:
                    mode_stack.close()
                    if mode.startswith('suffix'):
                        from sft.frozen_expert_grouped_mm import query_suffix_backward
                        mode_stack.enter_context(query_suffix_backward(forward._backbone(), model.query_encoder.query_count))
                    forward.forward_full = full if mode == 'full' or mode.startswith('suffix') else (
                        lambda inputs, match=mode != 'cached_original', fp32=mode in ('cached_fp32_state', 'cached_aligned'),
                               align=mode == 'cached_aligned':
                            cached(inputs, matched_dtype=match, fp32_state=fp32, aligned=align))
                    for module, enabled in checkpoint_modules:
                        module.gradient_checkpointing = enabled if mode in ('full', 'suffix_gc') else False
                    previous = None
                    for repetition in range(3):
                        engine.optimizer_zero_grad()
                        queries.clear()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        print(json.dumps({'prefix_profile_start': mode, 'offset': offset,
                                          'repetition': repetition}), flush=True)
                        started = time.perf_counter()
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            result = model(**payload, compute_metrics=False)
                            torch.cuda.synchronize()
                            forward_seconds = time.perf_counter() - started
                            backward_start = time.perf_counter()
                            result['loss'].backward()
                            torch.cuda.synchronize()
                        backward_seconds = time.perf_counter() - backward_start
                        current = gradient_snapshot(model, queries, result['loss'])
                        row = {'offset': offset, 'mode': mode, 'repetition': repetition,
                               'tokens': payload['attention_mask'].sum(-1).tolist(),
                               'forward_seconds': forward_seconds, 'backward_seconds': backward_seconds,
                               'compute_seconds': forward_seconds + backward_seconds,
                               'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                               'prefix_dtype': str(prefix_dtype), 'loss': result['loss'].item()}
                        rows.append(row)
                        if repetition == 2:
                            comparisons.append({'offset': offset, 'mode': mode,
                                'repeat': {k: _pair(previous[k], v, dtype=torch.float64) for k, v in current.items()},
                                'vs_full': None if reference is None else {
                                    k: _pair(reference[k], v, dtype=torch.float64) for k, v in current.items()}})
                            if mode == 'full':
                                reference = current
                        previous = current
                        del result
                        save()
                        print(json.dumps(row), flush=True)
                mode_stack.close()
                if suffix_only:
                    del payload
                    continue
                # Per-layer output diagnosis, outside the performance samples.
                captured = {}
                probe_mode = 'full'
                hooks = []
                def hook(index):
                    def capture(module, args, output):
                        output = output[0] if isinstance(output, tuple) else output
                        key = probe_mode
                        if probe_mode == 'cached':
                            key = 'prefix' if output.shape[1] > model.query_encoder.query_count else 'query'
                        captured.setdefault(key, {})[index] = output[:, -model.query_encoder.query_count:].detach().float().cpu()
                    return capture
                try:
                    for index, layer in enumerate(layers):
                        hooks.append(layer.register_forward_hook(hook(index)))
                        layer.gradient_checkpointing = False
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        forward.forward_full = full
                        model(**payload, compute_metrics=False)
                        probe_mode = 'cached'
                        forward.forward_full = lambda inputs: cached(inputs, matched_dtype=True, fp32_state=True)
                        model(**payload, compute_metrics=False)
                    layer_checks.append({'offset': offset, 'query': {
                        str(i): _pair(captured['full'][i], captured['query'][i], dtype=torch.float64)
                        for i in range(len(layers))}})
                    save()
                finally:
                    for handle in hooks:
                        handle.remove()
                del payload
    finally:
        mode_stack.close()
        engine.optimizer_zero_grad()
        forward.forward_full = full
        forward.prefix_embeds = prefix
        model._forward_slow_anchors = slow
        forward._cached_key = None
        forward._cached_prefix = None
        for layer, enabled in zip(layers, original_gc):
            layer.gradient_checkpointing = enabled
        for module, enabled in checkpoint_modules:
            module.gradient_checkpointing = enabled


def fm_prefix_chain_indices(dataset, episodes_per_source, anchors_per_episode, seed=20261019, tail=False):
    """Deterministic consecutive anchors of whole episodes, in decision order.

    tail=True takes the last anchors of the longest episodes (longest contexts).
    """
    import random
    selected, start = [], 0
    for name, count, episodes in zip(('r2r', 'rxr', 'sage'), dataset.source_counts,
                                     dataset.source_episodes):
        eligible = [e for e in episodes if e['end'] - e['start'] >= anchors_per_episode]
        if tail:
            chosen = sorted(eligible, key=lambda e: e['end'] - e['start'])[-episodes_per_source:]
        else:
            chosen = random.Random(f'{seed}:{name}').sample(eligible, episodes_per_source)
        for episode in chosen:
            first = episode['end'] - anchors_per_episode if tail else episode['start']
            selected.append(dict(source=name, identity=episode['identity'], indices=[
                start + index for index in range(first, first + anchors_per_episode)]))
        start += count
    return selected


def run_fm_prefix_chain_profile(trainer):
    """Replay same-episode anchor chains through the production engine.

    Fixed weights, no optimizer update. Run once per backend; the comparison
    pairs the saved per-anchor loss, query hidden and query/head gradients.
    """
    if torch.distributed.get_world_size() != 1:
        raise ValueError('FM prefix chain profile requires one rank')
    engine, model = trainer.engine, trainer.engine.module
    dataset = trainer.train_dataset.dataset
    collator = ActionHeadAnchorCollator(trainer.model_config.processor, 1, len(dataset))
    chains = fm_prefix_chain_indices(dataset, int(os.environ.get('ACTION_HEAD_FM_PROFILE_EPISODES', '2')),
                                     int(os.environ.get('ACTION_HEAD_FM_PROFILE_ANCHORS', '5')),
                                     tail=os.environ.get('ACTION_HEAD_FM_PROFILE_SELECT') == 'tail')
    output = Path(os.environ['OUTPUT_DIR'])
    queries, rows = [], []
    slow = model._forward_slow_anchors
    cache = getattr(model.query_forward, '_fm_prefix_cache', None)

    def capture_slow(*args, **kwargs):
        value = slow(*args, **kwargs)
        queries.append(value.detach())
        return value

    def uncached_full(payload):
        # Same backend without the history cache, twice, through the root
        # module so FSDP unshards parameters; separates cache error from
        # backend/kernel differences and run-to-run noise.
        forward = model.query_forward
        persistent = getattr(forward, '_persistent_fm_prefix', False)
        forward._persistent_fm_prefix = False
        try:
            values = []
            for _ in range(2):
                queries.clear()
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    model(**payload, compute_metrics=False)
                values.append(torch.cat(queries).float().cpu())
            return values
        finally:
            forward._persistent_fm_prefix = persistent

    def layer_check(payload):
        # Per-layer query-row outputs: cached branch vs the uncached forward.
        from sft.query_hidden_fm_impact import _pair
        forward, captured, handles = model.query_forward, {}, []
        layers = list(forward._backbone().layers)
        count = forward.query_encoder.query_count
        def hook(index):
            def capture(module, args, output):
                output = output[0] if isinstance(output, tuple) else output
                captured.setdefault(mode, {})[index] = output[:, -count:].detach().float().cpu()
            return capture
        persistent = forward._persistent_fm_prefix
        try:
            handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(layers)]
            for mode in ('cached', 'full'):
                forward._persistent_fm_prefix = mode == 'cached'
                if mode == 'cached':
                    forward._fm_prefix_cache.clear()
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    model(**payload, compute_metrics=False)
        finally:
            forward._persistent_fm_prefix = persistent
            forward._fm_prefix_cache.clear()
            for handle in handles:
                handle.remove()
        types = [getattr(layer, 'layer_type', type(getattr(layer, 'linear_attn', getattr(layer, 'self_attn', None))).__name__)
                 for layer in layers]
        return [dict(layer=i, type=types[i], **_pair(captured['full'][i], captured['cached'][i], dtype=torch.float64))
                for i in range(len(layers))]

    model._forward_slow_anchors = capture_slow
    try:
        with engine.train_mode():
            if os.environ.get('ACTION_HEAD_FM_PROFILE_LAYER_CHECK') == '1':
                checks = []
                for chain in chains[:2]:
                    payload = {key: value.cuda() for key, value in collator([dataset[chain['indices'][0]]]).items()}
                    checks.append(dict(index=chain['indices'][0], tokens=int(payload['attention_mask'].sum()),
                                       layers=layer_check(payload)))
                    (output / 'fm_prefix_layer_check.json').write_text(json.dumps(checks, indent=2) + '\n')
                    print(json.dumps(dict(event='fm_prefix_layer_check', **checks[-1])), flush=True)
                return
            for chain_index, chain in enumerate(chains):
                for position, index in enumerate(chain['indices']):
                    payload = {key: value.cuda() for key, value in collator([dataset[index]]).items()}
                    before = None if cache is None else dict(cache.stats)
                    engine.optimizer_zero_grad()
                    queries.clear()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        result = model(**payload, compute_metrics=False)
                        torch.cuda.synchronize()
                        forward_seconds = time.perf_counter() - started
                        result['loss'].backward()
                        torch.cuda.synchronize()
                    row = dict(chain=chain_index, position=position, index=index, source=chain['source'],
                               identity=chain['identity'], tokens=int(payload['attention_mask'].sum()),
                               forward_seconds=forward_seconds,
                               compute_seconds=time.perf_counter() - started,
                               peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                               loss=float(result['loss'].item()),
                               backend=model.query_forward.frozen_query_backend)
                    if cache is not None:
                        row['fm_prefix'] = {k: cache.stats[k] - before[k] for k in cache.stats}
                    snapshot = gradient_snapshot(model, queries, result['loss'])
                    if os.environ.get('ACTION_HEAD_FM_PROFILE_FULL_CHECK') == '1':
                        snapshot['full_query_hidden'], snapshot['full_repeat_query_hidden'] = uncached_full(payload)
                    torch.save(snapshot, output / f'anchor_{chain_index:02d}_{position:02d}.pt')
                    rows.append(row)
                    print(json.dumps(dict(event='fm_prefix_chain_profile', **row)), flush=True)
                    (output / 'fm_prefix_chain_profile.json').write_text(json.dumps(dict(
                        scope='fixed weights; consecutive same-episode anchors; no optimizer update',
                        chains=chains, rows=rows), indent=2) + '\n')
                    del result, payload
    finally:
        engine.optimizer_zero_grad()
        model._forward_slow_anchors = slow


def compare_fm_prefix_chain_profiles(reference, candidate):
    """Pair two chain profiles anchor by anchor (float64 statistics)."""
    from sft.query_hidden_fm_impact import _pair
    reference, candidate = Path(reference), Path(candidate)
    left = json.loads((reference / 'fm_prefix_chain_profile.json').read_text())
    right = json.loads((candidate / 'fm_prefix_chain_profile.json').read_text())
    if left['chains'] != right['chains'] or len(left['rows']) != len(right['rows']):
        raise ValueError('chain profiles used different anchors')
    pairs = []
    for a, b in zip(left['rows'], right['rows']):
        name = f"anchor_{a['chain']:02d}_{a['position']:02d}.pt"
        x, y = torch.load(reference / name), torch.load(candidate / name)
        pairs.append(dict(chain=a['chain'], position=a['position'], source=a['source'], tokens=a['tokens'],
            reference_seconds=a['compute_seconds'], candidate_seconds=b['compute_seconds'],
            fm_prefix=b.get('fm_prefix'),
            **{key: _pair(x[key], y[key], dtype=torch.float64) for key in x}))
    return pairs


def run_profile(trainer):
    if os.environ['ACTION_HEAD_LOCAL_PROFILE'] == 'fm_prefix_chain':
        return run_fm_prefix_chain_profile(trainer)
    if os.environ['ACTION_HEAD_LOCAL_PROFILE'] in ('prefix', 'suffix'):
        return run_prefix_profile(trainer)
    if os.environ['ACTION_HEAD_LOCAL_PROFILE'] in ('distributed', 'suffix-distributed'):
        return run_distributed_profile(trainer)
    if torch.distributed.get_world_size() != 1:
        raise ValueError('local profile requires exactly one rank')
    case = os.environ['ACTION_HEAD_LOCAL_PROFILE']
    if case == 'sweep':
        parent = Path(os.environ['OUTPUT_DIR'])
        try:
            for child in ('gc2', 'b2', 'b4'):
                child_dir = parent / child
                child_dir.mkdir(exist_ok=False)
                os.environ['ACTION_HEAD_LOCAL_PROFILE'] = child
                os.environ['OUTPUT_DIR'] = str(child_dir)
                run_profile(trainer)
                import gc
                gc.collect()
                torch.cuda.empty_cache()
        finally:
            os.environ['ACTION_HEAD_LOCAL_PROFILE'] = case
            os.environ['OUTPUT_DIR'] = str(parent)
        return
    if case not in ('b1', 'b2', 'b4', 'gc4', 'gc2', 'guard', 'bmm'):
        raise ValueError(case)
    engine = trainer.engine
    model = engine.module
    dataset = trainer.train_dataset.dataset
    collator = ActionHeadAnchorCollator(trainer.model_config.processor, 1, len(dataset))
    batch_size = int(case[1:]) if case in ('b1', 'b2', 'b4') else 1
    modes = ['sequential'] if case == 'b1' else ['sequential', 'candidate']
    output_dir = Path(os.environ['OUTPUT_DIR'])
    rows, comparisons = [], []
    original_lr = [group['lr'] for group in engine.optimizer.param_groups]
    for group in engine.optimizer.param_groups:
        group['lr'] = 0.0
    layers = list(model.qwen.model.language_model.layers)
    original_gc = [layer.gradient_checkpointing for layer in layers]
    if not all(original_gc):
        raise ValueError('profile reference requires checkpointing on every text layer')
    original_slow = model._forward_slow_anchors
    original_prefix = model.query_forward.prefix_embeds
    phases = {}
    queries = []
    from transformers.integrations import moe
    original_grouped_mm = moe._grouped_mm
    grouped_stats = {}
    original_env = {key: os.environ.get(key) for key in (
        'ACTION_HEAD_SLOW_BATCH_MODE', 'ACTION_HEAD_GC_SKIP_EVERY')}

    def grouped_mm(inputs, weights, *, offs):
        from sft.frozen_expert_grouped_mm import padded_grouped_mm
        return padded_grouped_mm(inputs.to(weights.dtype), weights, offs=offs, stats=grouped_stats)

    def measured(name, fn, *args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        torch.cuda.synchronize()
        phases[name] = phases.get(name, 0.0) + time.perf_counter() - start
        return result

    def slow(*args, **kwargs):
        result = measured('slow_forward', original_slow, *args, **kwargs)
        queries.append(result.detach())
        return result

    def prefix(*args, **kwargs):
        return measured('vision_and_prefix', original_prefix, *args, **kwargs)

    def snapshot(loss):
        return gradient_snapshot(model, queries, loss)

    def compare(left, right):
        return {key: _pair(left[key], right[key], dtype=torch.float64) for key in left}

    model._forward_slow_anchors = slow
    model.query_forward.prefix_embeds = prefix
    try:
        with engine.train_mode():
            for offset in range(0, len(dataset), batch_size):
                count = min(batch_size, len(dataset) - offset)
                start = time.perf_counter()
                cpu_batch = collator([dataset[index] for index in range(offset, offset + count)])
                input_seconds = time.perf_counter() - start
                reference = None
                for mode in modes:
                    if case == 'bmm':
                        moe._grouped_mm = grouped_mm if mode == 'candidate' else original_grouped_mm
                    if case == 'guard':
                        os.environ['ACTION_HEAD_GC_SKIP_EVERY'] = '2' if mode == 'candidate' else '0'
                    os.environ['ACTION_HEAD_SLOW_BATCH_MODE'] = (
                        'vectorized' if mode == 'candidate' and case in ('b2', 'b4') else 'sequential')
                    for index, layer in enumerate(layers):
                        layer.gradient_checkpointing = original_gc[index] and not (
                            mode == 'candidate' and case.startswith('gc') and index % int(case[2:]) == 0)
                    previous = None
                    for repetition in range(3):
                        print(json.dumps({'profile_start': case, 'offset': offset,
                                          'mode': mode, 'repetition': repetition}), flush=True)
                        engine.optimizer_zero_grad()
                        phases.clear()
                        queries.clear()
                        torch.cuda.reset_peak_memory_stats()
                        payload = measured('h2d', lambda: {k: v.cuda() for k, v in cpu_batch.items()})
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            output = measured('forward', model, **payload, compute_metrics=False)
                            loss = output['loss']
                            measured('backward', loss.backward)
                        if not torch.isfinite(loss):
                            raise RuntimeError('nonfinite profile loss')
                        measured('metrics', engine._extra_train_metrics, output, loss)
                        # Capture pre-clipping gradients, outside timing; repeat-vs-repeat
                        # is the numerical reference for this vendor's kernels.
                        current = snapshot(loss) if repetition else None
                        measured('optimizer', engine.optimizer_step)
                        row = {'case': case, 'offset': offset, 'anchors': count,
                               'mode': mode, 'repetition': repetition,
                               'tokens': cpu_batch['attention_mask'].sum(-1).tolist(),
                               'cpu_input_seconds': input_seconds, 'phases': dict(phases),
                               'loss': float(loss.detach()),
                               'checkpointed_text_layers': sum(layer.gradient_checkpointing for layer in layers),
                               'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
                               'time_utc': time.time()}
                        row['step_seconds'] = sum(phases[k] for k in ('h2d', 'forward', 'backward', 'metrics', 'optimizer'))
                        rows.append(row)
                        with (output_dir / 'profile.jsonl').open('a') as stream:
                            stream.write(json.dumps(row) + '\n')
                        print(json.dumps(row), flush=True)
                        if repetition == 2:
                            comparisons.append({'offset': offset, 'mode': mode,
                                                'repeat': compare(previous, current),
                                                'candidate_vs_reference': compare(reference, current) if reference else None})
                            if mode == 'sequential':
                                reference = current
                        previous = current
                        del output, loss, payload
                (output_dir / 'comparisons.json').write_text(json.dumps(comparisons, indent=2) + '\n')
        report = {'scope': f'fixed-weight LR=0; production engine; {len(dataset)} real anchors; synchronized phases; repetition 0 is warmup',
                  'case': case, 'rows': rows, 'comparisons': comparisons,
                  'grouped_stats': grouped_stats,
                  'frozen_qwen_trainable_parameters': sum(p.numel() for p in model.qwen.parameters() if p.requires_grad)}
        (output_dir / 'profile.json').write_text(json.dumps(report, indent=2) + '\n')
        if case in ('b1', 'gc4'):
            # Separate trace: profiler overhead is excluded from all timings above.
            from torch.profiler import ProfilerActivity, profile, record_function, supported_activities
            activities = [ProfilerActivity.CPU]
            if ProfilerActivity.CUDA in supported_activities():
                activities.append(ProfilerActivity.CUDA)
            payload = {k: v.cuda() for k, v in cpu_batch.items()}
            with engine.train_mode(), profile(activities=activities, record_shapes=True) as trace:
                engine.optimizer_zero_grad()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    with record_function('sft_forward'):
                        output = model(**payload, compute_metrics=False)
                    with record_function('sft_backward'):
                        output['loss'].backward()
                with record_function('sft_metrics'):
                    engine._extra_train_metrics(output, output['loss'])
                with record_function('sft_optimizer'):
                    engine.optimizer_step()
                torch.cuda.synchronize()
            trace.export_chrome_trace(str(output_dir / 'trace.json'))
            # Vendor profiler's recursive key_averages is prohibitively slow on
            # this graph. Preserve the raw trace for offline event aggregation.
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        moe._grouped_mm = original_grouped_mm
        model._forward_slow_anchors = original_slow
        model.query_forward.prefix_embeds = original_prefix
        for layer, value in zip(layers, original_gc):
            layer.gradient_checkpointing = value
        for group, lr in zip(engine.optimizer.param_groups, original_lr):
            group['lr'] = lr
