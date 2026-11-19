"""Restore sampled token history after chat-template/multimodal processing.

Qwen's chat template normalizes reasoning/whitespace. Re-rendering messages
therefore cannot certify a continuous token stream. Reuse the preceding native
capture and tokenize only the newly appended tool observations in effect.
"""
from dataclasses import replace
from functools import wraps

PREFIX_KEY = "action_head_continuous_prefix_ids"
PARENT_KEY = "action_head_continuous_parent_request"
CACHE_KEY = "action_head_continuous_cache_salt"
CACHE_PARENT_END_KEY = "action_head_cache_parent_end"


def install_parent_cache_limit_hook(manager_class=None):
    """Keep parallel candidates within their authoritative parent's prefix.

    Siblings/retries have identical prompts, but only the selected parent's
    native artifact certifies cached routes. Reuse vLLM's own cache coordinator
    with that natural answer boundary; first-turn candidates have no parent.
    Requests without this collection-only metadata use vLLM unchanged.
    """
    if manager_class is None:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        manager_class = KVCacheManager
    original = manager_class.get_computed_blocks
    if getattr(original, '_v8_parent_cache_limit', False):
        return

    @wraps(original)
    def get_computed_blocks(self, request):
        extra = getattr(request.sampling_params, 'extra_args', None) or {}
        if CACHE_PARENT_END_KEY not in extra:
            return original(self, request)
        limit = extra[CACHE_PARENT_END_KEY]
        if type(limit) is not int or limit < 0 or not extra.get(CACHE_KEY):
            raise ValueError('cached collection requires a certified parent boundary and episode salt')
        if limit and not extra.get(PARENT_KEY):
            raise ValueError('cached collection prefix has no selected parent')
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return original(self, request)
        blocks, count = (self.coordinator.find_longest_cache_hit(
            request.block_hashes, min(limit, request.num_tokens - 1)) if limit else (None, 0))
        if count > limit:
            raise ValueError('cache coordinator crossed the selected parent boundary')
        if self.log_stats:
            self.prefix_cache_stats.record(num_tokens=request.num_tokens,
                num_hits=count, preempted=request.num_preemptions > 0)
        return (self.create_kv_cache_blocks(blocks) if blocks is not None
                else self.empty_kv_cache_blocks), count

    get_computed_blocks._v8_parent_cache_limit = True
    manager_class.get_computed_blocks = get_computed_blocks


def validate_cached_routes(request, routes, cache_root):
    """Certify cached SHM rows against the authoritative parent, before export.

    vLLM owns both attention KV and GDN align-mode state. Its route reader uses
    the same attention-block slots, including cache hits. Never accept a hit
    from another episode or past the parent's natural answer/query boundary.
    """
    import re
    import numpy as np
    import torch
    from pathlib import Path
    from safetensors import safe_open

    extra = request.sampling_params.extra_args or {}
    salt = extra.get(CACHE_KEY)
    if not isinstance(salt, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", salt):
        raise ValueError("v8 prefix cache requires an episode cache salt")
    if getattr(request, "cache_salt", None) != salt:
        raise ValueError("v8 request/cache salt mismatch")
    cached = max(0, int(request.num_cached_tokens))
    if not cached:
        return {"cached_parent_tokens_verified": 0, "prefix_cache_enabled": True}
    parent = extra.get(PARENT_KEY)
    if not isinstance(parent, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", parent):
        raise ValueError("v8 cache hit requires an authoritative parent")
    with safe_open(str(Path(cache_root) / (parent + ".safetensors")), framework="pt", device="cpu") as f:
        end = int(f.get_tensor("decision_response_end").item())
        old_salt = bytes(f.get_tensor(CACHE_KEY).tolist()).decode("ascii")
        if old_salt != salt or cached > end or cached > len(request.prompt_token_ids):
            raise ValueError("v8 cache hit crosses an episode or natural response boundary")
        ids = f.get_slice("all_input_ids")[:cached]
        old_routes = f.get_slice("all_routed_experts")[:cached]
    if not torch.equal(ids, torch.tensor(request.prompt_token_ids[:cached])):
        raise ValueError("v8 cached token IDs differ from the parent")
    if (not isinstance(routes, np.ndarray)
            or routes[:cached].shape != tuple(old_routes.shape)
            or not np.array_equal(routes[:cached], old_routes.numpy())):
        raise ValueError("v8 cached MoE routes differ from the parent")
    return {"cached_parent_tokens_verified": cached, "prefix_cache_enabled": True,
            "parent_response_end": end}


def merge_prefix_token_ids(ids, prefix, *, assistant_marker, end_token):
    """Shared native-prefix reconstruction for serving and budget accounting."""
    if (not prefix or not ids or any(type(x) is not int or x < 0 for x in prefix)):
        raise ValueError("v8 requires a nonempty authoritative token prefix")
    starts = [i for i in range(len(ids) - len(assistant_marker) + 1)
              if ids[i:i + len(assistant_marker)] == assistant_marker]
    # The final marker opens the new generation; the previous one is the
    # completed sampled response being restored, not a textual sentinel.
    if len(starts) < 2:
        raise ValueError("v8 continuation lacks a completed assistant boundary")
    cut = next((i for i in range(starts[-2], starts[-1]) if ids[i] == end_token), None)
    if cut is None:
        raise ValueError("v8 continuation lacks an assistant end boundary")
    prefix = list(prefix)
    if prefix[-1] == end_token:
        prefix.pop()  # Keep exactly one canonical conversation separator.
    return prefix + ids[cut:]


def restore_prefix(request, prefix, *, assistant_marker, end_token, image_token, max_length):
    merged = merge_prefix_token_ids(request.prompt_token_ids, prefix,
        assistant_marker=assistant_marker, end_token=end_token)
    if len(merged) + request.sampling_params.max_tokens > max_length:
        raise ValueError("v8 context budget exhausted; compression is not enabled")
    # Rebase each visual placeholder after text length changes. Pixel data and
    # image order are unchanged; never retokenize or resize a historical image.
    positions = []
    for i, value in enumerate(merged):
        if value == image_token and (i == 0 or merged[i - 1] != image_token):
            end = i + 1
            while end < len(merged) and merged[end] == image_token:
                end += 1
            positions.append((i, end - i))
    features = request.mm_features or []
    if len(positions) != len(features):
        raise ValueError("v8 image placeholders changed across continuous history")
    rebased = []
    for feature, (offset, length) in zip(features, positions, strict=True):
        if feature.modality != "image" or feature.mm_position.length != length:
            raise ValueError("v8 historical image shape or modality changed")
        rebased.append(replace(feature, mm_position=replace(feature.mm_position, offset=offset)))
    request.prompt_token_ids = merged
    request.mm_features = rebased or None
    return request


def install_continuous_context_hook():
    from vllm.v1.engine.input_processor import InputProcessor
    original = InputProcessor.process_inputs
    if getattr(original, "_v8_continuous", False):
        return
    original_init = InputProcessor.__init__

    @wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # The renderer shares its fast tokenizer with asynchronous request
        # preprocessing. HF encode() changes truncation/padding settings, so
        # calling it from process_inputs can mutate a Rust tokenizer already
        # borrowed by another request. Resolve the constant boundaries before
        # this InputProcessor is exposed to requests, never lazily on first use.
        tokenizer = self.tokenizer
        self._v8_continuous_tokens = (
            tuple(tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)),
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
            tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        )

    @wraps(original)
    def process(self, *args, **kwargs):
        request = original(self, *args, **kwargs)
        params = request.sampling_params
        extra = getattr(params, "extra_args", None) or {}
        prefix = extra.pop(PREFIX_KEY, None)
        if prefix is not None:
            if not extra.get(PARENT_KEY):
                raise ValueError("v8 continuation requires native parent provenance")
            assistant_marker, end_token, image_token = self._v8_continuous_tokens
            restore_prefix(request, prefix,
                assistant_marker=list(assistant_marker),
                end_token=end_token, image_token=image_token,
                max_length=self.model_config.max_model_len)
        return request

    process._v8_continuous = True
    InputProcessor.__init__ = initialize
    InputProcessor.process_inputs = process


def replay_prefix_routes(weights, ids, logits, segments, layer_index):
    """Keep old expert choices; recompute their weights from current logits."""
    import torch
    for start, end, routes in segments:
        chosen = routes[:, layer_index].long()
        if chosen.shape != ids[start:end].shape or end > len(ids):
            raise ValueError("v8 scheduled prefix routes do not align with MoE rows")
        probabilities = torch.softmax(logits[start:end].float(), dim=-1)
        selected = probabilities.gather(-1, chosen)
        selected = selected / selected.sum(-1, keepdim=True)
        ids[start:end] = chosen.to(ids.dtype)
        weights[start:end] = selected.to(weights.dtype)
    return weights, ids


class PrefixRouteBuffers:
    """Stable-address replay inputs; prepare outside the captured model graph.

    The mask, rather than a Python list captured on the first forward, selects
    historical rows. New observations, generated tokens and padding retain the
    ordinary router outputs. Keep the same softmax/gather/renormalize as eager.
    """
    def __init__(self, capacity, layers, top_k, device):
        import torch
        if any(type(x) is not int or x <= 0 for x in (capacity, layers, top_k)):
            raise ValueError('route buffers require positive integer dimensions')
        self.routes = torch.zeros((capacity, layers, top_k), dtype=torch.int64, device=device)
        self.mask = torch.zeros((capacity, 1), dtype=torch.bool, device=device)

    def clear(self):
        self.mask.zero_()

    def put(self, start, end, routes):
        if (not 0 <= start <= end <= len(self.mask)
                or tuple(routes.shape) != (end-start, *self.routes.shape[1:])):
            raise ValueError('scheduled prefix exceeds route buffer shape/capacity')
        self.routes[start:end].copy_(routes)
        self.mask[start:end].fill_(True)

    def apply(self, weights, ids, logits, layer):
        import torch
        chosen = self.routes[:ids.shape[0], layer]
        mask = self.mask[:ids.shape[0]]
        probabilities = torch.softmax(logits.float(), dim=-1)
        selected = probabilities.gather(-1, chosen)
        selected = selected / selected.sum(-1, keepdim=True)
        return (torch.where(mask, selected.to(weights.dtype), weights),
                torch.where(mask, chosen.to(ids.dtype), ids))


class ContinuousRouting:
    """Reuse the existing scheduler row mapping and native R3 artifacts.

    Prefill can choose different experts from the earlier decode due to numeric
    rounding. Pin only previously computed prefix choices; new observations,
    new sampled text and FM query carriers still use the ordinary router.
    """
    def __init__(self, model, cache_root, *, graph_capacity=0, top_k=0):
        import re
        from pathlib import Path
        self.root = Path(cache_root)
        self.device = next(model.parameters()).device
        self.requests = {}
        self.segments = []
        self.buffers = None
        layers = set()
        for name, module in model.named_modules():
            router = getattr(module, "router", None)
            match = re.search(r"layers\.(\d+)\.mlp\.experts$", name)
            if router is None or match is None:
                continue
            if getattr(module, "is_sequence_parallel", False):
                raise ValueError("v8 prefix route replay does not support sequence-parallel MoE")
            layer = int(match.group(1))
            if getattr(router, "scoring_func", None) != "softmax" or not router.renormalize:
                raise ValueError("v8 continuous routing requires normalized softmax experts")
            original = router._compute_routing

            def compute(hidden_states, router_logits, indices_type, original=original, layer=layer):
                weights, ids = original(hidden_states, router_logits, indices_type)
                if self.buffers is not None:
                    return self.buffers.apply(weights, ids, router_logits, layer)
                return replay_prefix_routes(weights, ids, router_logits, self.segments, layer)

            router._compute_routing = compute
            layers.add(layer)
        if not layers or layers != set(range(max(layers) + 1)):
            raise ValueError("v8 failed to bind every Qwen MoE layer")
        self.layer_count = len(layers)
        if graph_capacity:
            self.buffers = PrefixRouteBuffers(graph_capacity, self.layer_count, top_k, self.device)

    def register(self, request, extra):
        import re
        import torch
        from safetensors import safe_open
        parent = extra.get(PARENT_KEY)
        if parent is None:
            return
        if not isinstance(parent, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", parent):
            raise ValueError("v8 parent must be a safe native request ID")
        with safe_open(str(self.root / (parent + ".safetensors")), framework="pt", device="cpu") as f:
            end = int(f.get_tensor("decision_response_end").item())
            ids = f.get_tensor("all_input_ids")[:end]
            routes = f.get_tensor("all_routed_experts")[:end].clone()
        if (not torch.equal(ids, torch.tensor(request.prompt_token_ids[:end]))
                or routes.ndim != 3 or routes.shape[1] != self.layer_count):
            raise ValueError("v8 parent routes do not match the restored prefix")
        self.requests[extra["action_head_query_request_id"]] = routes

    def prepare(self, rows):
        self.clear()
        for start, end, request_id, token_start in rows:
            routes = self.requests.get(request_id)
            if routes is None or token_start >= len(routes):
                continue
            count = min(end - start, len(routes) - token_start)
            source = routes[token_start:token_start + count]
            if self.buffers is not None:
                self.buffers.put(start, start + count, source)
            else:
                self.segments.append((start, start + count, source.to(self.device)))

    def clear(self):
        self.segments = []
        if self.buffers is not None:
            self.buffers.clear()

    def discard(self, request_id):
        self.requests.pop(request_id, None)
