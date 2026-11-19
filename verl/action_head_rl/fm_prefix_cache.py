"""Update-local causal prefix reuse for selective query/FM differentiation.

Canonical caches contain ordinary history only, never learned query carriers.
Every query branch gets a private cache and retains its query derivative.
This is opt-in: a caller must scope it to one fixed Qwen policy/update.
Open-loop SFT keeps one per-rank cache for the whole run because its Qwen
weights never change; it uses native routing and commits rendered history only.
"""
from contextlib import contextmanager
import copy
import os
import types

import torch

from action_head_rl.router_replay import replay_qwen35_moe_routes
from sft.joint_fm_head_sft import _append_positions


def _preserve_recurrent_state(self, recurrent_states, **kwargs):
    # HF cache initialization may select the BF16 conv dtype for an FP32 FLA
    # recurrent state. Keep the kernel's state precision across chunk borders.
    if torch.is_grad_enabled():
        raise RuntimeError('canonical recurrent cache cannot acquire a graph')
    self.recurrent_states = recurrent_states.detach().clone()
    self.is_recurrent_states_initialized = True
    return self.recurrent_states


def new_prefix_cache(backbone):
    from transformers.cache_utils import DynamicCache, LinearAttentionLayer
    cache = DynamicCache(config=backbone.config)
    cache._action_head_commit_prefix = True
    for layer in cache.layers:
        if isinstance(layer, LinearAttentionLayer):
            layer.update_recurrent_state = types.MethodType(_preserve_recurrent_state, layer)
    return cache


class FmImagePrefixCache:
    """One trajectory's exact CPU pixels/features, never a global LRU.

    The caller fixes model weights and releases this object before any update.
    A byte cap bounds retained tensors even for unexpectedly long trajectories.
    Every nonempty image request still executes ONE vision forward, including
    complete hits, to preserve the existing FSDP collective order on all ranks.
    """

    def __init__(self, max_bytes=1024 ** 3):
        self.max_bytes = max_bytes
        self.clear()
        self.stats = dict(image_calls=0, image_reused=0, image_encoded=0,
                          image_alignment_encoded=0, image_cap_skips=0)

    def clear(self):
        self.pixels = self.grid = self.embeddings = None

    @property
    def retained_bytes(self):
        return sum(t.numel() * t.element_size() for t in
                   (self.pixels, self.grid, self.embeddings) if t is not None)

    def features(self, multimodal, pixels, grid):
        if torch.is_grad_enabled():
            raise RuntimeError('FM image reuse requires a frozen no-grad prefix')
        if pixels.ndim != 2 or grid.ndim != 2 or grid.shape[1] != 3 or not len(grid):
            raise ValueError('FM image reuse requires native flattened image patches and THW grids')
        sizes = grid.prod(-1).tolist()
        if any(n <= 0 for n in sizes) or sum(sizes) != pixels.shape[0]:
            raise ValueError('image grids do not cover the processed pixel rows')
        # Keep matching inputs and cached features in host memory. The GPU
        # owns only this call's inputs/result, not a persistent image cache.
        # These synchronous copies are intentional and included in the probe's
        # wall time; CPU storage is not claimed to provide zero-cost reuse.
        host_pixels = pixels.detach().to('cpu')
        host_grid = grid.detach().to('cpu')
        old_images = 0 if self.grid is None else len(self.grid)
        old_rows = 0 if self.pixels is None else self.pixels.shape[0]
        hit = (0 < old_images <= len(grid) and old_rows <= len(pixels)
               and self.pixels.dtype == host_pixels.dtype
               and torch.equal(self.grid, host_grid[:old_images])
               and torch.equal(self.pixels, host_pixels[:old_rows]))
        if not hit:
            self.clear()
            old_images = old_rows = 0
        # Qwen vision attends independently inside each image. Encode only
        # newly appended images; a full hit runs one disposable first image so
        # FSDP vision blocks participate in the same collectives as other ranks.
        alignment = old_images == len(grid)
        new_pixels = pixels[:sizes[0]] if alignment else pixels[old_rows:]
        new_grid = grid[:1] if alignment else grid[old_images:]
        output = multimodal.get_image_features(new_pixels, new_grid, return_dict=True)
        if alignment:
            result = self.embeddings.to(device=pixels.device)
        else:
            fresh = torch.cat(output.pooler_output, dim=0)
            result = torch.cat((self.embeddings.to(device=fresh.device), fresh), dim=0) if hit else fresh
        del output
        self.stats['image_calls'] += 1
        self.stats['image_reused'] += old_images
        self.stats['image_encoded'] += len(new_grid)
        self.stats['image_alignment_encoded'] += int(alignment)
        size = sum(t.numel() * t.element_size() for t in (pixels, grid, result))
        if size > self.max_bytes:
            self.clear()
            self.stats['image_cap_skips'] += 1
        elif not alignment:
            # Own only the latest prefix, with no autograd or input-storage
            # aliases that could keep a whole staged batch alive.
            self.clear()
            self.pixels = host_pixels.clone()
            self.grid = host_grid.clone()
            self.embeddings = result.detach().to(device='cpu', copy=True)
        return result


class FmPrefixCache:
    """One bounded, exact-input prefix, released before the policy changes."""

    def __init__(self, *, aligned_forwards=False, history_token_id=None):
        self.aligned_forwards = aligned_forwards
        # Chat templates re-render the latest assistant turn once it becomes
        # history, so only tokens before its final role marker are shared.
        self.history_token_id = history_token_id
        # Preserve main's persistent native-route SFT cache; image reuse is
        # the existing R21 opt-in path for routed RL conditions only.
        self.images = (FmImagePrefixCache() if history_token_id is None and
                       os.environ.get('ACTION_HEAD_FM_VISION_PREFIX_REUSE', '1') == '1' else None)
        self.clear()
        self.routes = None
        self.stats = dict(calls=0, hits=0, reused_tokens=0, prefix_tokens=0, query_tokens=0)

    def clear(self):
        if self.images is not None:
            self.images.clear()
        self._clear_history()

    def _clear_history(self):
        self.cache = None
        self.embeds = self.positions = self.prefix_routes = None

    @contextmanager
    def replay(self, backbone, routes):
        self.routes = routes
        with replay_qwen35_moe_routes(backbone, routes,
                                      route_selector=lambda: self.routes):
            yield

    def forward(self, query_forward, inputs, routes):
        if inputs['input_ids'].shape[0] != 1:
            raise ValueError('FM prefix reuse requires one condition per forward')
        mask = inputs['attention_mask']
        padding = mask.shape[1] - int(mask.sum())
        if not torch.equal(mask.bool(),
                           (torch.arange(mask.shape[1], device=mask.device) >= padding)[None]):
            raise ValueError('FM prefix reuse requires a real suffix after contiguous left padding')
        if padding:
            inputs = {key: (value[:, padding:] if key in
                ('input_ids', 'attention_mask', 'mm_token_type_ids') else value)
                for key, value in inputs.items()}
            routes = None if routes is None else routes[padding:]
        native = routes is None
        backbone = query_forward._backbone()
        if any(getattr(m, 'gradient_checkpointing', False) for m in backbone.modules()):
            raise ValueError('cached FM branch requires scoped checkpoint disabling')
        count = query_forward.query_encoder.QUERY_COUNT
        with torch.no_grad():
            if self.images is None or native:
                embeds, mask, positions = query_forward.prefix_embeds(inputs)
            else:
                embeds, mask, positions = query_forward.prefix_embeds(inputs, image_cache=self.images)
            n = embeds.shape[1]
            if positions is None or (not native and routes.shape[0] != n + count):
                raise ValueError('cached FM needs exact mRoPE and prefix/query routes')
            # Keep recurrent chunk boundaries equal to the full FLA path.
            boundary = n // 64 * 64
            if self.history_token_id is not None:
                markers = (inputs['input_ids'][0] == self.history_token_id).nonzero()
                if not len(markers):
                    raise ValueError('FM history boundary marker is absent')
                boundary = min(boundary, int(markers[-1]) // 64 * 64)
            if not boundary:
                raise ValueError('FM cached prefix needs at least 64 tokens')
            old = 0 if self.embeds is None else self.embeds.shape[1]
            hit = (0 < old <= boundary
                   and torch.equal(self.embeds, embeds[:, :old])
                   and torch.equal(self.positions, positions[..., :old])
                   and (native or torch.equal(self.prefix_routes, routes[:old])))
            # FSDP ranks must each run a prefix forward and a query forward.
            # Repeated zero-valid padding cannot skip a collective on one rank.
            if not hit or (self.aligned_forwards and boundary == old):
                self._clear_history()
                hit = False
                old = 0
                self.cache = new_prefix_cache(backbone)
            if boundary > old:
                self.routes = None if native else routes[old:boundary]
                output = backbone(inputs_embeds=embeds[:, old:boundary],
                    attention_mask=mask[:, :boundary], position_ids=positions[..., old:boundary],
                    past_key_values=self.cache, use_cache=True, return_dict=True)
                if output.past_key_values is None:
                    raise RuntimeError('FM prefix forward failed to return a cache')
                self.cache = output.past_key_values
                del output
            self.embeds = embeds[:, :boundary].detach().clone()
            self.positions = positions[..., :boundary].detach().clone()
            self.prefix_routes = None if native else routes[:boundary].detach().clone()
            # Attention appends K/V; recurrent layers may write in-place. A
            # private branch prevents query tokens contaminating later options.
            branch = copy.deepcopy(self.cache)
            branch._action_head_commit_prefix = False
        joined, joined_mask = query_forward.query_encoder.append(embeds, mask)
        self.routes = None if native else routes[boundary:]
        output = backbone(inputs_embeds=joined[:, boundary:], attention_mask=joined_mask,
            position_ids=_append_positions(positions, count)[..., boundary:],
            past_key_values=branch, use_cache=False, return_dict=True)
        self.stats['calls'] += 1
        self.stats['hits'] += int(hit)
        self.stats['reused_tokens'] += old if hit else 0
        self.stats['prefix_tokens'] += boundary - old
        self.stats['query_tokens'] += n - boundary + count
        return output.last_hidden_state[:, -count:].to(query_forward.query_encoder.query_embeddings.device)


@contextmanager
def fm_prefix_reuse(query_forward, *, aligned_forwards=False):
    """Caller keeps Qwen fixed; restore checkpointing and release all cache."""
    if (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
            and not aligned_forwards):
        raise ValueError('distributed FM cache needs aligned prefix-forward scheduling first')
    if getattr(query_forward, '_fm_prefix_cache', None) is not None:
        raise ValueError('FM cache scopes cannot overlap')
    states = [(m, m.gradient_checkpointing) for m in query_forward.model.modules()
              if hasattr(m, 'gradient_checkpointing')]
    cache = FmPrefixCache(aligned_forwards=aligned_forwards)
    try:
        for module, _ in states:
            module.gradient_checkpointing = False
        query_forward._fm_prefix_cache = cache
        yield cache
    finally:
        query_forward._fm_prefix_cache = None
        for module, enabled in states:
            module.gradient_checkpointing = enabled
        cache.clear()
        cache.routes = None


def install_persistent_fm_prefix_cache(query_forward, history_token_id):
    """Run-long cache for frozen-Qwen open-loop SFT with unsharded replicas.

    Native routing is valid only because every Qwen parameter is frozen; the
    learned queries never enter the canonical cache, so optimizer steps do not
    invalidate it. Sharded FSDP would need aligned forwards, which SFT lacks.
    """
    if any(parameter.requires_grad for parameter in query_forward.model.parameters()):
        raise ValueError('persistent FM prefix cache requires a frozen Qwen backbone')
    if os.environ.get('FSDP_SIZE', '1') != '1':
        raise ValueError('persistent FM prefix cache requires FSDP_SIZE=1 replicas')
    if getattr(query_forward, '_fm_prefix_cache', None) is not None:
        raise ValueError('FM cache scopes cannot overlap')
    query_forward._fm_prefix_cache = FmPrefixCache(history_token_id=int(history_token_id))
    query_forward._persistent_fm_prefix = True
    return query_forward._fm_prefix_cache
