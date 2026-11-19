"""Process-local compatibility patches for the action expert vLLM service."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from math import ceil, lcm


def _is_stdlib_resource_tracker(argv=None) -> bool:
    """Recognize only CPython's dedicated IPC cleanup helper.

    Spawned model workers must still install every patch. The resource tracker
    only reads the stdlib resource-registration pipe and never runs model code.
    Its inherited PYTHONPATH otherwise imports vLLM/Torch during Python startup.
    """
    import re
    import sys

    argv = getattr(sys, "orig_argv", ()) if argv is None else argv
    if "-c" not in argv or "-m" in argv:
        return False
    index = argv.index("-c")
    if len(argv) != index + 2:
        return False
    return re.fullmatch(
        r"from multiprocessing\.resource_tracker import main;main\([0-9]+\)",
        argv[index + 1].strip(),
    ) is not None


def _route_buffer_factor(vllm_config) -> int:
    """Recover the KV-group factor dropped by vLLM 0.18's route buffer.

    ``KVCacheConfig.num_blocks`` is the number of blocks in each shared HMA
    pool.  vLLM 0.18 sizes the routed-expert buffer with
    ``num_blocks // num_groups`` even though attention slot mappings use the
    full per-pool block-id range.

    The number of groups is not merely the number of attention *types*.
    vLLM's hybrid cache grouping splits every type into chunks whose size is
    the smallest type population (unless populations differ by less than
    1.5x).  Qwen3.6 has 30 linear-attention and 10 full-attention layers, so
    it produces four groups (three linear plus one full), not two.  Mirror
    that grouping rule here because init_buffer/attach_buffer receive only the
    already-divided token capacity, not KVCacheConfig itself.
    """
    text = vllm_config.model_config.hf_text_config
    layer_types = getattr(text, "layer_types", None) or []
    counts = list(Counter(layer_types).values())
    if not counts:
        return 1
    group_size = min(counts)
    if max(counts) < group_size * 1.5:
        group_size = max(counts)
    return sum(ceil(count / group_size) for count in counts)


def _route_buffer_tokens(max_num_kv_tokens, vllm_config) -> tuple[int, int]:
    """Recover the hybrid pool and cover vLLM's truncated block remainder.

    vLLM computes ``max_num_kv_tokens`` from ``num_blocks // num_groups``.
    Multiplying that quotient by the group count restores all complete groups,
    but it cannot restore the remainder discarded by integer division. Those
    last physical blocks still have valid block IDs and can appear in an
    attention slot mapping. One scheduler batch is a conservative upper bound
    for that small remainder on the supported vLLM configuration and is
    available identically in the worker and EngineCore processes.
    """
    factor = _route_buffer_factor(vllm_config)
    guard_tokens = int(vllm_config.scheduler_config.max_num_batched_tokens)
    return int(max_num_kv_tokens) * factor + guard_tokens, factor


def _patch_routed_experts_buffer_capacity() -> None:
    """Size routed-expert shared memory for the complete hybrid KV pool."""
    from vllm.model_executor.layers.fused_moe import routed_experts_capturer as module
    from action_head.serve.routing_shm import configure_namespace

    configure_namespace(module)

    capturer_original = module.RoutedExpertsCapturer.init_buffer
    if getattr(capturer_original, "_action_head_patched", False):
        return
    reader_original = module.RoutedExpertsReader.attach_buffer

    def init_buffer(self, max_num_batched_tokens, max_num_kv_tokens, vllm_config):
        safe_tokens, factor = _route_buffer_tokens(
            max_num_kv_tokens, vllm_config)
        print(
            f"[action-head] routed-experts buffer tokens "
            f"{max_num_kv_tokens} -> {safe_tokens} "
            f"(factor={factor}, guard={int(max_num_batched_tokens)})",
            flush=True,
        )
        return capturer_original(
            self, max_num_batched_tokens, safe_tokens, vllm_config)

    def attach_buffer(self, max_num_kv_tokens, vllm_config):
        safe_tokens, _ = _route_buffer_tokens(max_num_kv_tokens, vllm_config)
        return reader_original(
            self, safe_tokens, vllm_config)

    init_buffer._action_head_patched = True
    attach_buffer._action_head_patched = True
    module.RoutedExpertsCapturer.init_buffer = init_buffer
    module.RoutedExpertsReader.attach_buffer = attach_buffer


def _patch_padded_kv_page_unification() -> None:
    """Make vLLM 0.18 hybrid KV pages valid on the PPU build.

    The PPU cache specs expose a pre-padded ``page_size_bytes``.  Upstream uses
    that padded value to scale an attention block and then retains the old
    padding, so its resulting page is not the requested common size.  Merely
    changing the padding is also invalid: the runner later views the complete
    physical allocation as the logical attention tensor.

    For attention-like specs, derive bytes/token from the *real* page and
    change block_size so both real and physical pages equal one common size.
    State-space cache specs have their own strided layout and only receive the
    physical padding expected by vLLM's hybrid cache manager.
    """
    import vllm.v1.core.kv_cache_utils as utils

    original = utils.unify_kv_cache_spec_page_size
    if getattr(original, "_action_head_patched", False):
        return

    def describe(stage, specs):
        for name, spec in specs.items():
            fields = {
                "type": type(spec).__name__,
                "block": getattr(spec, "block_size", None),
                "real": getattr(spec, "real_page_size_bytes", None),
                "page": spec.page_size_bytes,
                "padded": getattr(spec, "page_size_padded", None),
            }
            print(f"[action-head] kv-spec {stage} {name}: {fields}", flush=True)

    def compatible(specs):
        # extract_hidden_states adds an auxiliary cache-only attention layer.
        # Qwen3.5/3.6's hybrid-cache setup gives it the globally enlarged
        # attention block size, making this auxiliary cache four times larger
        # than every real model cache.  It must not dictate the common HMA page:
        # choose the block size that fits the largest *model* cache page first.
        model_specs = {
            name: spec for name, spec in specs.items()
            if not name.startswith("cache_only_layers.")
        }
        if model_specs:
            model_attention_blocks = [
                spec.block_size for spec in model_specs.values()
                if getattr(spec, "real_page_size_bytes", None) is not None
            ]
            normalized = dict(specs)
            for name, spec in specs.items():
                if not name.startswith("cache_only_layers."):
                    continue
                block_size = getattr(spec, "block_size", None)
                real_page = getattr(spec, "real_page_size_bytes", None)
                if not block_size or not real_page or real_page % block_size:
                    continue
                bytes_per_token = real_page // block_size
                if not model_attention_blocks:
                    continue
                # The auxiliary cache inherits the hybrid manager's enlarged
                # block (528 for Qwen3.5-4B, 1056 for Qwen3.6-35B).  Reset it
                # to the model attention block before unifying physical pages.
                # This avoids letting a cache-only layer dictate the layout.
                target_block = min(model_attention_blocks)
                changes = {"block_size": target_block}
                if hasattr(spec, "page_size_padded"):
                    changes["page_size_padded"] = None
                candidate = replace(spec, **changes)
                normalized[name] = candidate
                print(
                    f"[action-head] normalized auxiliary hidden cache "
                    f"{name}: block {block_size} -> "
                    f"{candidate.block_size}, page {real_page} -> "
                    f"{candidate.page_size_bytes}",
                    flush=True,
                )
            specs = normalized

        describe("before", specs)
        try:
            result = original(specs)
            describe("upstream", result)
            return result
        except (AssertionError, NotImplementedError):
            pass

        minimum_common_page = max(spec.page_size_bytes for spec in specs.values())
        attention_bytes_per_token = []
        for spec in specs.values():
            block_size = getattr(spec, "block_size", None)
            real_page = getattr(spec, "real_page_size_bytes", None)
            if block_size and real_page:
                if real_page % block_size:
                    raise RuntimeError(
                        f"KV real page {real_page} is not divisible by block "
                        f"size {block_size} for {type(spec).__name__}"
                    )
                attention_bytes_per_token.append(real_page // block_size)

        # Attention kernels in this vLLM build operate on 16-token virtual
        # blocks.  The physical manager block may be larger, but it must split
        # cleanly into that kernel block.  State-space pages can be padded and
        # therefore do not need to divide the common page.
        attention_page_alignment = 1
        for bytes_per_token in attention_bytes_per_token:
            attention_page_alignment = lcm(
                attention_page_alignment, bytes_per_token * 16)
        common_page = (
            (minimum_common_page + attention_page_alignment - 1)
            // attention_page_alignment
            * attention_page_alignment
        )

        print(
            f"[action-head] using PPU hybrid KV common page {common_page}",
            flush=True,
        )
        updated = {}
        for name, spec in specs.items():
            block_size = getattr(spec, "block_size", None)
            real_page = getattr(spec, "real_page_size_bytes", None)
            if block_size and real_page:
                bytes_per_token = real_page // block_size
                new_block_size = common_page // bytes_per_token
                changes = {"block_size": new_block_size}
                if hasattr(spec, "page_size_padded"):
                    changes["page_size_padded"] = common_page
                candidate = replace(spec, **changes)
                if (
                    candidate.real_page_size_bytes != common_page
                    or candidate.page_size_bytes != common_page
                ):
                    raise RuntimeError(
                        f"Cannot form a dense common KV page for {name}: "
                        f"real={candidate.real_page_size_bytes}, "
                        f"physical={candidate.page_size_bytes}, "
                        f"expected={common_page}"
                    )
            else:
                if not hasattr(spec, "page_size_padded"):
                    raise RuntimeError(
                        f"Cannot pad hybrid KV spec {name} ({type(spec).__name__})"
                    )
                candidate = replace(spec, page_size_padded=common_page)
                if candidate.page_size_bytes != common_page:
                    raise RuntimeError(
                        f"Cannot pad hybrid KV spec {name} to {common_page}"
                    )
            updated[name] = candidate
        describe("compatible", updated)
        return updated

    compatible._action_head_patched = True
    utils.unify_kv_cache_spec_page_size = compatible


def _patch_extract_hidden_states_token_shape() -> None:
    """Backport extract-hidden fixes missing from the PPU vLLM 0.18 build.

    Besides the one-token shape fix from PR #38610, the 0.18 proposer ignores
    its ``slot_mappings`` argument.  That is unsafe with hybrid KV caches: the
    common attention metadata carries group 0's mapping, while the auxiliary
    hidden cache can have a different physical block size.  Repeated requests
    eventually write beyond the hidden-cache allocation and trigger a delayed
    device-side assert.  Select the mapping for the cache-only layer before
    entering the upstream proposer.
    """
    from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer

    original = ExtractHiddenStatesProposer.propose
    if getattr(original, "_action_head_patched", False):
        return

    def propose_one_token(self, *args, **kwargs):
        common = kwargs.get("common_attn_metadata")
        slot_mappings = kwargs.get("slot_mappings")
        if common is not None and slot_mappings is not None:
            if isinstance(slot_mappings, list):
                if len(slot_mappings) != 1:
                    raise RuntimeError(
                        "extract_hidden_states does not support DBO slot mappings"
                    )
                slot_mappings = slot_mappings[0]
            if not self.attn_layer_names:
                raise RuntimeError("extract_hidden_states has no cache-only layer")
            hidden_layer = self.attn_layer_names[0]
            try:
                hidden_slot_mapping = slot_mappings[hidden_layer]
            except (KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"missing slot mapping for hidden layer {hidden_layer}"
                ) from exc
            kwargs["common_attn_metadata"] = common.replace(
                slot_mapping=hidden_slot_mapping
            )
        sampled_token_ids = original(self, *args, **kwargs)
        # The rejection sampler returns [batch, max_spec_len + 1] after the
        # first decode step.  This proposer has exactly one draft token.
        return sampled_token_ids[:, :1]

    propose_one_token._action_head_patched = True
    ExtractHiddenStatesProposer.propose = propose_one_token


def _install_runtime_patches() -> None:
    from action_head.serve.compile_cache import configure_process_compile_cache
    configure_process_compile_cache()

    _patch_padded_kv_page_unification()
    _patch_extract_hidden_states_token_shape()
    _patch_routed_experts_buffer_capacity()

    # Keep the existing service path untouched unless native R3 is opted in.
    import os
    if os.environ.get("ACTION_HEAD_R3_CAPTURE_MODE", "legacy") != "legacy":
        from action_head.serve.native_r3_capture import install_native_r3_scheduler_hook
        install_native_r3_scheduler_hook()

    if os.environ.get("ACTION_HEAD_CONTINUOUS_CONTEXT") == "1":
        from action_head.serve.continuous_context import install_continuous_context_hook
        install_continuous_context_hook()


if not _is_stdlib_resource_tracker():
    _install_runtime_patches()
