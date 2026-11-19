"""Bounded, exact-input memoization of frozen slow-backbone MoE prefixes.

Only prefix rows of frozen tokenwise MoE blocks are retained. Attention/GDN
still run their validated full causal kernels; no recurrent/KV approximation
is introduced. Trainable queries are recomputed on every optimizer step.
"""
import os
import json

import torch


class FrozenPrefixReuse:
    def __init__(self, modules, *, max_bytes=4 << 30, max_tokens=16384):
        if not modules or max_bytes <= 0 or max_tokens <= 0:
            raise ValueError('prefix reuse needs frozen MoE modules and a positive memory bound')
        if any(p.requires_grad for m in modules for p in m.parameters()):
            raise ValueError('prefix memoization requires frozen weights')
        self.entries = {}
        self.per_module_bytes = max_bytes // len(modules)
        self.max_tokens = max_tokens
        self.calls = self.reused_rows = self.computed_rows = self.retained_bytes = 0
        self.enabled = True
        self.verified = set()
        self.native_fallbacks = {}
        self.max_relative_error = 0.

    def run(self, key, original, inputs, query_count):
        if not self.enabled:
            return original(inputs)
        if inputs.ndim != 3 or inputs.shape[0] != 1 or inputs.shape[1] < query_count:
            raise ValueError('prefix reuse requires one unpadded anchor and its query suffix')
        prefix = inputs.shape[1] - query_count
        if key in self.native_fallbacks:
            self.calls += 1
            self.computed_rows += inputs.shape[1]
            return original(inputs)
        reuse = 0
        entry = self.entries.pop(key, None)
        if entry is not None:
            old_input, old_output = entry
            self.retained_bytes -= sum(x.numel()*x.element_size() for x in entry)
            if old_input.device == inputs.device and old_input.dtype == inputs.dtype:
                common = min(prefix, old_input.shape[1])
                # Actual hidden values include image embeddings, positions and
                # all previous layers. Token IDs or episode lineage alone are
                # insufficient proof that a multimodal prefix can be reused.
                equal = (inputs[:, :common].detach() == old_input[:, :common]).all(-1).reshape(-1)
                changed = torch.nonzero(~equal, as_tuple=False)
                reuse = int(changed[0, 0]) if changed.numel() else common
        if reuse:
            output = torch.cat((old_output[:, :reuse], original(inputs[:, reuse:].contiguous())), dim=1)
            if key not in self.verified:
                with torch.no_grad():
                    reference = original(inputs)
                    difference = (output.detach().float()-reference.float())
                    relative = float(difference.square().mean().sqrt()
                                     / reference.float().square().mean().sqrt().clamp_min(1e-8))
                    finite = bool(torch.isfinite(difference).all())
                if not finite:
                    raise RuntimeError('nonfinite frozen prefix candidate/reference')
                if relative > .005:
                    # GEMM/router dispatch can depend on token batch shape.
                    # Preserve the original differentiable path for this
                    # module, rather than accepting an invalid memoization or
                    # relaxing the numerical gate. All other layers retain
                    # independent admission and their own bounded cache.
                    self.native_fallbacks[key] = relative
                    self.calls += 1
                    self.computed_rows += inputs.shape[1]
                    print(json.dumps(dict(event='slow_prefix_native_fallback',
                        rank=int(os.environ.get('RANK', '0')), module=key,
                        relative_error=relative, tolerance=.005,
                        candidate_discarded=True, query_gradient_path='original_full_forward')), flush=True)
                    return original(inputs)
                self.max_relative_error = max(self.max_relative_error, relative)
                self.verified.add(key)
        else:
            output = original(inputs)
        entry = old_input = old_output = None
        # Store owned detached storage, never a view retaining a query graph.
        row_bytes = (inputs.shape[-1]*inputs.element_size()
                     + output.shape[-1]*output.element_size())
        retain = min(prefix, self.max_tokens, self.per_module_bytes // row_bytes)
        if retain:
            value = (inputs[:, :retain].detach().clone(), output[:, :retain].detach().clone())
            self.entries[key] = value
            self.retained_bytes += sum(x.numel()*x.element_size() for x in value)
        self.calls += 1
        self.reused_rows += reuse
        self.computed_rows += inputs.shape[1] - reuse
        return output

    def clear(self):
        self.entries.clear()
        self.retained_bytes = 0

    def report(self):
        return dict(scope='frozen_tokenwise_moe_prefix', calls=self.calls,
                    reused_rows=self.reused_rows, computed_rows=self.computed_rows,
                    retained_bytes=self.retained_bytes, query_outputs_cached=False,
                    parity_verified_modules=len(self.verified), max_relative_error=self.max_relative_error,
                    native_fallback_modules=len(self.native_fallbacks),
                    native_fallback_errors=dict(self.native_fallbacks),
                    attention_gdn_cache=False, exact_hidden_input_match=True)


def install_prefix_reuse(backbone):
    from sft.frozen_query_pointwise import FrozenQuerySparseMoeBlock
    modules = [m for m in backbone.modules() if isinstance(m, FrozenQuerySparseMoeBlock)]
    cache = FrozenPrefixReuse(modules,
        max_bytes=int(float(os.environ.get('ACTION_HEAD_PREFIX_CACHE_GIB', '4')) * (1 << 30)),
        max_tokens=int(os.environ.get('ACTION_HEAD_PREFIX_CACHE_TOKENS', '16384')))
    for index, module in enumerate(modules):
        module._prefix_reuse = cache
        module._prefix_reuse_key = index
    return cache
