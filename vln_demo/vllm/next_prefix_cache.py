"""Non-speculative align-cache compatibility for the Next PPU image.

PLE has one convolution state; GDN has convolution and recurrent states.
vLLM 0.24's copy-buffer allocator incorrectly requires their full MambaSpecs
to match. Its non-speculative copy path already uses each tensor's own size
and each group's block IDs. Only their scheduling geometry must agree.
"""

import logging
from dataclasses import replace


def install():
    import torch
    from vllm.model_executor.layers.mamba.mamba_utils import (
        get_conv_copy_spec,
        get_temporal_copy_spec,
    )
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec
    from vllm.v1.worker.mamba_utils import MambaCopyBuffers

    if getattr(MambaCopyBuffers, "_vln_next_prefix_cache", False):
        return
    original_create = MambaCopyBuffers.create.__func__
    original_unify = kv_cache_utils.unify_kv_cache_spec_page_size
    expected_states = {
        MambaAttentionBackendEnum.SHORT_CONV: 1,
        MambaAttentionBackendEnum.GDN_ATTN: 2,
    }

    def create(cls, max_num_reqs, kv_cache_config, copy_funcs, make_buffer):
        groups = [
            (gid, group)
            for gid, group in enumerate(kv_cache_config.kv_cache_groups)
            if isinstance(group.kv_cache_spec, MambaSpec)
        ]
        specs = [group.kv_cache_spec for _, group in groups]
        if {spec.mamba_type for spec in specs} != set(expected_states):
            return original_create(
                cls, max_num_reqs, kv_cache_config, copy_funcs, make_buffer
            )

        if len({spec.block_size for spec in specs}) != 1:
            raise ValueError("Next PLE/GDN cache groups need equal token block sizes")
        if any(spec.mamba_cache_mode != "align" for spec in specs):
            raise ValueError("Next prefix-cache compatibility requires align mode")
        if any(spec.num_speculative_blocks != 0 for spec in specs):
            raise ValueError("Next prefix-cache compatibility does not enable MTP")
        if copy_funcs != (get_conv_copy_spec, get_temporal_copy_spec):
            raise ValueError("Unexpected Next convolution/recurrent copy functions")
        for spec in specs:
            count = expected_states[spec.mamba_type]
            if len(spec.shapes) != count or len(spec.dtypes) != count:
                raise ValueError("Unexpected Next PLE/GDN state count")

        # Allocate one descriptor per actual state, including PLE's single
        # state. The existing copier reads sizes from the layer tensors,
        # rather than from the representative scheduling spec below.
        entries = max_num_reqs * sum(
            len(group.layer_names) * len(group.kv_cache_spec.shapes)
            for _, group in groups
        )
        logging.getLogger("vllm.vln_next_prefix_cache").info(
            "Next prefix cache: %d PLE/GDN groups, block_size=%d, %d copy entries",
            len(groups), specs[0].block_size, entries,
        )
        return cls(
            src_ptrs=make_buffer(entries, dtype=torch.uint64),
            dst_ptrs=make_buffer(entries, dtype=torch.uint64),
            sizes=make_buffer(entries, dtype=torch.int32),
            mamba_group_ids=[gid for gid, _ in groups],
            mamba_spec=specs[0],
        )

    MambaCopyBuffers.create = classmethod(create)

    def unify_pages(kv_cache_spec):
        specs = list(kv_cache_spec.values())
        mamba_types = {spec.mamba_type for spec in specs if isinstance(spec, MambaSpec)}
        if mamba_types == set(expected_states) and all(
            spec.mamba_cache_mode == "align"
            for spec in specs if isinstance(spec, MambaSpec)
        ):
            # The generic unifier enlarges compressed QSA's *token* block
            # (e.g. 400 -> 12800, depending on page size) to fill a page.
            # The align scheduler only saves Mamba states at its smaller
            # prefill boundaries, so no common checkpoint can be matched.
            # QSA indexes by physical block stride: padding its page instead
            # keeps compressed storage rows and logical token boundaries
            # correct, at the cost of some KV capacity.
            page_bytes = max(spec.page_size_bytes for spec in specs)
            kv_cache_spec = dict(kv_cache_spec)
            for name, spec in kv_cache_spec.items():
                if isinstance(spec, MLAAttentionSpec) and spec.compress_ratio > 1:
                    if not spec.indexes_kv_by_block_stride:
                        raise ValueError("Next QSA padding requires block-stride indexing")
                    kv_cache_spec[name] = replace(spec, page_size_padded=page_bytes)
            logging.getLogger("vllm.vln_next_prefix_cache").info(
                "Next QSA prefix cache: padded compressed pages to %d bytes; "
                "preserved logical token block sizes", page_bytes,
            )
        return original_unify(kv_cache_spec)

    kv_cache_utils.unify_kv_cache_spec_page_size = unify_pages
    MambaCopyBuffers._vln_next_prefix_cache = True
