"""Independent native-Qwen context frontend, shared by live policy and benchmarks."""
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
import time
import copy
from collections import OrderedDict

import torch
from torch import nn
from safetensors import safe_open
from models.qwen36_fast_slow_action_head import _QueryPooler
from .policy import Qwen35VisualTokenEncoder, TorchActionHeadPolicy


class ContextAggregator(nn.Module):
    """Independent linear-cost pooling; positions retain serialized history order."""
    def __init__(self, hidden=2048, dim=384, slots=16):
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden)
        self.projection = nn.Linear(hidden, dim)
        self.modality = nn.Embedding(2, dim)
        self.pool = _QueryPooler(dim, slots, 6)

    def forward(self, tokens, valid, is_image):
        value = self.projection(self.input_norm(tokens))
        half = value.shape[-1] // 2
        scales = torch.exp(torch.arange(half, device=value.device).float()
                           * (-math.log(10000.0) / max(half-1, 1)))
        # Left padding must not shift a context's original token positions.
        positions = (valid.long().cumsum(-1) - 1).clamp_min(0).float()
        phase = positions[..., None] * scales
        position = torch.cat((phase.sin(), phase.cos()), -1).to(value.dtype)
        value = value + position + self.modality(is_image.long())
        return self.pool(value, valid)


def native_text_embedding(model_dir, device, dtype):
    model_dir = Path(model_dir)
    weight_map = json.loads((model_dir/'model.safetensors.index.json').read_text())['weight_map']
    key = 'model.language_model.embed_tokens.weight'
    with safe_open(model_dir/weight_map[key], framework='pt', device='cpu') as handle:
        return nn.Embedding.from_pretrained(handle.get_tensor(key), freeze=True).to(device, dtype)


class CachedNativeFrontend(nn.Module):
    """Frozen text lookup plus previously encoded, version-bound image tokens."""
    def __init__(self, model_dir, device='cpu', dtype=torch.bfloat16):
        super().__init__()
        self.embedding = native_text_embedding(model_dir, device, dtype)
        self.image_id = int(json.loads((Path(model_dir)/'config.json').read_text())['image_token_id'])

    def forward(self, inputs):
        tokens = self.embedding(inputs['input_ids'])
        visual = inputs['cached_visual_tokens']
        mask = inputs['input_ids'].eq(self.image_id)
        return tokens.masked_scatter(mask.unsqueeze(-1), visual.to(tokens.dtype))


class NativeQwenFrontend(nn.Module):
    """Native pre-language embeddings, shared by component and RPC benchmarks."""
    def __init__(self, model_dir, device='cuda:0', dtype=torch.bfloat16):
        super().__init__()
        model_dir = Path(model_dir)
        self.vision = Qwen35VisualTokenEncoder.from_pretrained(model_dir, device=device, dtype=dtype)
        # The policy adapter is not an nn.Module. Register its actual tower so
        # training device moves, FSDP and frozen-parameter audits include it.
        self.vision_model = self.vision.vision_model
        self.processor = self.vision.processor
        self.embedding = native_text_embedding(model_dir, device, dtype)
        self.image_id = int(json.loads((model_dir/'config.json').read_text())['image_token_id'])
        self.dtype = dtype

    def forward(self, inputs):
        tokens = self.embedding(inputs['input_ids'])
        visual = self.vision.vision_model(inputs['pixel_values'].to(self.dtype),
            grid_thw=inputs['image_grid_thw'], return_dict=True).pooler_output
        image_mask = inputs['input_ids'].eq(self.image_id)
        if image_mask.sum().item() != visual.shape[0]:
            raise ValueError('image placeholder count differs from native merged vision output')
        return tokens.masked_scatter(image_mask.unsqueeze(-1), visual.to(tokens.dtype))


@dataclass(frozen=True)
class NativeContextCondition:
    tokens: torch.Tensor
    request_id: str
    metadata: dict


class _CachedGridProcessor:
    """Reuse the official text/placeholder processing with already known grids."""
    def __init__(self, original, grids):
        self.original, self.grids = original, grids

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, images=None, **kwargs):
        if len(images) != len(self.grids):
            raise ValueError('cached grid/image ordering mismatch')
        return {'image_grid_thw': self.grids}


class NativeImageFeatureCache:
    """Per-encoder content-addressed LRU; retains merged features, not pixels.

    Encoding stays independent for each image. No language hidden states or
    position-dependent aggregate is cached. Eviction only triggers re-encoding.
    """
    def __init__(self, max_bytes=512 * 1024**2):
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = 0
        self.processor_signature = None

    def clear(self):
        self.entries.clear()
        self.bytes = 0

    @torch.inference_mode()
    def encode(self, frontend, messages, completion, device):
        from sft.joint_fm_head_sft import _slow_template_messages, build_online_prefix_text
        def sync():
            if torch.device(device).type == 'cuda':
                torch.cuda.synchronize(device)
        start = time.perf_counter()
        processor = frontend.processor
        signature = json.dumps(processor.image_processor.to_dict(), sort_keys=True, default=str)
        invalidated = self.processor_signature is not None and self.processor_signature != signature
        if invalidated:
            self.clear()
        self.processor_signature = signature
        template, images = _slow_template_messages(messages, copy_native_images=False)
        prompt = build_online_prefix_text(processor, template, completion)
        templated = time.perf_counter()
        keys, local, missing = [], {}, {}
        hits = 0
        for image in images:
            digest = hashlib.sha256(f'{image.mode}:{image.size}:'.encode() + image.tobytes()).hexdigest()
            keys.append(digest)
            if digest in self.entries:
                hits += 1
                local[digest] = self.entries[digest]
                self.entries.move_to_end(digest)
            elif digest not in missing:
                missing[digest] = image
        active_keys = set(keys)
        retired = 0
        for key in list(self.entries):
            if key not in active_keys:
                _, dropped = self.entries.pop(key)
                self.bytes -= dropped.numel() * dropped.element_size()
                del dropped
                retired += 1
        looked_up = time.perf_counter()
        new_inputs = (processor.image_processor(images=list(missing.values()), return_tensors='pt')
                      if missing else None)
        if missing:
            for key, grid in zip(missing, new_inputs['image_grid_thw']):
                local[key] = (grid.clone(), None)
        preprocessed = time.perf_counter()
        # Native Qwen processor expands the exact same placeholders/token IDs;
        # its image processor is replaced only on a private shallow copy.
        grids = torch.stack([local[key][0] for key in keys])
        text_processor = copy.copy(processor)
        text_processor.image_processor = _CachedGridProcessor(processor.image_processor, grids)
        inputs = text_processor(text=[prompt], images=images, padding=True, return_tensors='pt')
        tokenized = time.perf_counter()
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        if new_inputs is not None:
            pixels = new_inputs['pixel_values'].to(device=device, dtype=frontend.dtype)
            new_grids = new_inputs['image_grid_thw'].to(device)
        sync()
        transferred = time.perf_counter()
        if missing:
            values = frontend.vision.vision_model(pixels, grid_thw=new_grids, return_dict=True).pooler_output
            counts = (new_inputs['image_grid_thw'].prod(-1) // processor.image_processor.merge_size**2).tolist()
            if sum(counts) != values.shape[0]:
                raise ValueError('new-image token count mismatch')
            pieces = values.split(counts)
        sync()
        encoded = time.perf_counter()
        evictions = 0
        if missing:
            for key, value in zip(missing, pieces):
                entry = (local[key][0], value.detach().clone())
                local[key] = entry
                size = entry[1].numel() * entry[1].element_size()
                if size <= self.max_bytes:
                    self.entries[key] = entry
                    self.bytes += size
                    while self.bytes > self.max_bytes:
                        _, (_, dropped) = self.entries.popitem(last=False)
                        self.bytes -= dropped.numel() * dropped.element_size()
                        evictions += 1
        sync()
        written = time.perf_counter()
        # local pins every needed frame for this request even if LRU evicts it.
        visual = torch.cat([local[key][1] for key in keys], dim=0)
        tokens = frontend.embedding(inputs['input_ids'])
        mask = inputs['input_ids'].eq(frontend.image_id)
        if mask.sum().item() != visual.shape[0]:
            raise ValueError('cached native image placeholder count mismatch')
        hidden = tokens.masked_scatter(mask.unsqueeze(-1), visual.to(tokens.dtype))
        sync()
        assembled = time.perf_counter()
        metadata = {
            'hits': hits, 'misses': len(missing), 'image_occurrences': len(keys),
            'duplicate_occurrences': len(keys) - len(set(keys)), 'entries': len(self.entries),
            'retained_feature_bytes': self.bytes, 'capacity_bytes': self.max_bytes,
            'evictions': evictions, 'retired_entries': retired,
            'processor_invalidated': invalidated,
            'template_ms': (templated-start)*1000,
            'lookup_and_content_hash_ms': (looked_up-templated)*1000,
            'new_image_preprocess_ms': (preprocessed-looked_up)*1000,
            'text_tokenize_ms': (tokenized-preprocessed)*1000,
            'transfer_ms': (transferred-tokenized)*1000,
            'new_image_encode_ms': (encoded-transferred)*1000,
            'write_ms': (written-encoded)*1000,
            'read_and_assemble_ms': (assembled-written)*1000,
            'prepare_ms': (transferred-start)*1000,
        }
        return inputs, hidden, metadata


class NativeContextProvider:
    """Encode the actual live request and tool completion once per option."""
    query_count = 0
    requires_query_tail = False

    def __init__(self, model_dir, *, device, dtype, seed, aggregator_checkpoint, image_cache=False,
                 frontend=None):
        self.device, self.dtype = device, dtype
        self.frontend = frontend if frontend is not None else NativeQwenFrontend(model_dir, device, dtype)
        self.image_cache = NativeImageFeatureCache() if image_cache else None
        self._reset_pending = False
        with torch.random.fork_rng(devices=[torch.device(device)]):
            torch.manual_seed(seed)
            self.aggregator = ContextAggregator().to(device, dtype).eval()
        path = Path(aggregator_checkpoint)
        root = Path('/data/zerovln')
        if not path.resolve().is_relative_to(root):
            raise ValueError('aggregator checkpoint must be on the authorized data disk')
        if path.exists():
            self.aggregator.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.aggregator.state_dict(), path)
        self.aggregator.requires_grad_(False)

    def reset(self):
        # reset_environment runs outside the device-owner thread. Defer tensor
        # release to the next encode on its owner thread.
        self._reset_pending = True

    def end_episode(self):
        """Release episode features on the same thread that encoded them."""
        if self.image_cache is None:
            return None
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        before = torch.cuda.memory_allocated(self.device)
        entries = len(self.image_cache.entries)
        retained = self.image_cache.bytes
        self.image_cache.clear()
        self._reset_pending = False
        torch.cuda.synchronize(self.device)
        return {
            'entries_before': entries, 'entries_after': len(self.image_cache.entries),
            'retained_feature_bytes_before': retained,
            'retained_feature_bytes_after': self.image_cache.bytes,
            'gpu_allocated_before_bytes': before,
            'gpu_allocated_after_bytes': torch.cuda.memory_allocated(self.device),
            'gpu_reserved_after_bytes': torch.cuda.memory_reserved(self.device),
            'release_ms': (time.perf_counter() - started) * 1000,
        }

    @torch.inference_mode()
    def __call__(self, *, request, response, contract):
        from sft.joint_fm_head_sft import build_slow_inputs
        del contract
        if not request or not request.get('messages') or not response or not response.get('text'):
            raise ValueError('v5 requires the complete live history and current tool completion')
        if response.get('query_request_id'):
            raise ValueError('v5 must not generate or consume brain query states')
        torch.cuda.synchronize(self.device)
        if self.image_cache is not None:
            torch.cuda.reset_peak_memory_stats(self.device)
            allocated_before = torch.cuda.memory_allocated(self.device)
        started = time.perf_counter()
        cache_metadata = None
        if self.image_cache is not None:
            if self._reset_pending:
                self.image_cache.clear()
                self._reset_pending = False
            inputs, hidden, cache_metadata = self.image_cache.encode(
                self.frontend, request['messages'], response['text'], self.device)
            cpu_ms = cache_metadata['prepare_ms']
        else:
            inputs = build_slow_inputs(self.frontend.processor, request['messages'], response['text'], self.device)
            torch.cuda.synchronize(self.device)
            cpu_ms = (time.perf_counter()-started)*1000
            hidden = self.frontend(inputs)
        torch.cuda.synchronize(self.device)
        aggregate_started = time.perf_counter()
        pooled = self.aggregator(hidden, inputs['attention_mask'].bool(),
                                 inputs['input_ids'].eq(self.frontend.image_id))
        if not torch.isfinite(pooled).all():
            raise ValueError('nonfinite native condition')
        digest = hashlib.sha256(inputs['input_ids'].cpu().numpy().tobytes()).hexdigest()
        torch.cuda.synchronize(self.device)
        finished = time.perf_counter()
        if cache_metadata is not None:
            cache_metadata['aggregate_and_validation_ms'] = (finished-aggregate_started)*1000
            cache_metadata['gpu_allocated_before_bytes'] = allocated_before
            cache_metadata['gpu_allocated_after_bytes'] = torch.cuda.memory_allocated(self.device)
            cache_metadata['gpu_peak_during_condition_bytes'] = torch.cuda.max_memory_allocated(self.device)
            cache_metadata['grid_cpu_bytes'] = sum(grid.numel()*grid.element_size()
                for grid, _ in self.image_cache.entries.values())
        metadata = {'encoder': 'qwen36_native_pre_language', 'trained_aggregator': False,
                    'messages': len(request['messages']), 'images': len(inputs['image_grid_thw']),
                    'sequence_tokens': hidden.shape[1], 'hidden_shape': list(hidden.shape),
                    'condition_shape': list(pooled.shape), 'input_ids_sha256': digest,
                    'all_supplied_history_included': True, 'query_tail_requested': False,
                    'cpu_prepare_transfer_ms': cpu_ms,
                    'encode_and_aggregate_ms': (finished-started)*1000-cpu_ms,
                    'image_cache': cache_metadata}
        return NativeContextCondition(pooled, 'v5-'+digest[:16], metadata)


class NativeContextActionHeadPolicy(TorchActionHeadPolicy):
    """Reuse live reference/current vision and FM, replacing only slow input."""
    def _slow_condition_inputs(self, options):
        bundles = [option.slow_condition for option in options]
        if any(not isinstance(value, NativeContextCondition) for value in bundles):
            raise TypeError('v5 requires NativeContextCondition')
        tokens = torch.cat([value.tokens for value in bundles], dim=0)
        return {'query_hidden_states': None, 'slow_condition_tokens': tokens}
