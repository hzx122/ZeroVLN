"""Run with Python in the public image; all state-copy checks use CPU tensors."""

import ctypes
from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch

from next_prefix_cache import install
from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_copy_spec,
    get_temporal_copy_spec,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum as Backend
from vllm.v1.core import kv_cache_utils
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, MLAAttentionSpec
from vllm.v1.worker import mamba_utils


install()
COPY_FUNCS = (get_conv_copy_spec, get_temporal_copy_spec)


def make_buffer(n, dtype):
    return SimpleNamespace(np=torch.zeros(n, dtype=dtype).numpy())


def config(ple=None, gdn=None):
    ple = ple or MambaSpec(
        block_size=128, shapes=((4, 5),), dtypes=(torch.float16,),
        mamba_type=Backend.SHORT_CONV, mamba_cache_mode="align",
    )
    gdn = gdn or MambaSpec(
        block_size=128, shapes=((2, 3), (2, 2, 3)),
        dtypes=(torch.float16, torch.float32),
        mamba_type=Backend.GDN_ATTN, mamba_cache_mode="align",
    )
    return SimpleNamespace(kv_cache_groups=[
        SimpleNamespace(kv_cache_spec=object(), layer_names=["attention"]),
        SimpleNamespace(kv_cache_spec=ple, layer_names=["ple"]),
        SimpleNamespace(kv_cache_spec=gdn, layer_names=["gdn"]),
    ])


class PrefixCacheTests(unittest.TestCase):
    def test_heterogeneous_copy_preserves_group_ids_sizes_and_page_padding(self):
        cfg = config()
        bufs = mamba_utils.MambaCopyBuffers.create(2, cfg, COPY_FUNCS, make_buffer)
        self.assertEqual(bufs.mamba_group_ids, [1, 2])
        self.assertEqual(bufs.src_ptrs.np.size, 6)
        context = {}
        tensors = []
        for name, group, src, dst in [
            ("ple", cfg.kv_cache_groups[1], 2, 3),
            ("gdn", cfg.kv_cache_groups[2], 1, 2),
        ]:
            states = []
            for shape, dtype in zip(group.kv_cache_spec.shapes, group.kv_cache_spec.dtypes):
                raw = torch.full((4 * 64,), -11, dtype=dtype)
                strides = torch.empty(shape).stride()
                state = raw.as_strided((4, *shape), (64, *strides))
                state[src].fill_(len(tensors) + 5)
                before = raw.clone()
                states.append(state)
                tensors.append((raw, state, before, src, dst))
            context[name] = SimpleNamespace(kv_cache=states)
        mamba_utils.collect_mamba_copy_meta(
            bufs, cfg, COPY_FUNCS, bufs.mamba_group_ids, 0, 1, 0,
            SimpleNamespace(block_ids=[[], [2, 3], [1, 2]]), context,
        )
        self.assertEqual(bufs.offset, 3)
        self.assertEqual(bufs.sizes.np[:3].tolist(), [40, 12, 48])
        for src, dst, size in zip(bufs.src_ptrs.np[:3], bufs.dst_ptrs.np[:3], bufs.sizes.np[:3]):
            ctypes.memmove(int(dst), int(src), int(size))
        for raw, state, before, src, dst in tensors:
            self.assertTrue(torch.equal(state[src], state[dst]))
            expected = before.clone()
            begin = dst * 64
            expected[begin:begin + state[dst].numel()] = state[src].flatten()
            self.assertTrue(torch.equal(raw, expected), "copy overwrote another state/page")

    def test_uniform_groups_keep_original_allocator(self):
        cfg = config()
        cfg.kv_cache_groups[1].kv_cache_spec = cfg.kv_cache_groups[2].kv_cache_spec
        bufs = mamba_utils.MambaCopyBuffers.create(2, cfg, COPY_FUNCS, make_buffer)
        self.assertEqual(bufs.src_ptrs.np.size, 8)

    def test_incompatible_block_boundaries_are_rejected(self):
        cfg = config()
        cfg.kv_cache_groups[1].kv_cache_spec = replace(cfg.kv_cache_groups[1].kv_cache_spec, block_size=256)
        with self.assertRaisesRegex(ValueError, "equal token block sizes"):
            mamba_utils.MambaCopyBuffers.create(1, cfg, COPY_FUNCS, make_buffer)

    def test_speculative_state_layout_is_not_silently_enabled(self):
        cfg = config()
        cfg.kv_cache_groups[1].kv_cache_spec = replace(cfg.kv_cache_groups[1].kv_cache_spec, num_speculative_blocks=3)
        with self.assertRaisesRegex(ValueError, "does not enable MTP"):
            mamba_utils.MambaCopyBuffers.create(1, cfg, COPY_FUNCS, make_buffer)

    def test_wrong_state_count_is_rejected(self):
        cfg = config()
        cfg.kv_cache_groups[1].kv_cache_spec = replace(cfg.kv_cache_groups[1].kv_cache_spec, shapes=((4, 5), (2, 3)))
        with self.assertRaisesRegex(ValueError, "state count"):
            mamba_utils.MambaCopyBuffers.create(1, cfg, COPY_FUNCS, make_buffer)

    def test_wrong_cache_mode_is_rejected(self):
        cfg = config()
        cfg.kv_cache_groups[1].kv_cache_spec = replace(cfg.kv_cache_groups[1].kv_cache_spec, mamba_cache_mode="all")
        with self.assertRaisesRegex(ValueError, "align mode"):
            mamba_utils.MambaCopyBuffers.create(1, cfg, COPY_FUNCS, make_buffer)

    def test_original_strict_get_mamba_groups_is_unchanged(self):
        with self.assertRaises(AssertionError):
            mamba_utils.get_mamba_groups(config())

    def test_qsa_page_padding_preserves_shared_checkpoint_boundaries(self):
        cfg = config()
        specs = {
            "ple": replace(cfg.kv_cache_groups[1].kv_cache_spec, block_size=400, page_size_padded=819200),
            "gdn": replace(cfg.kv_cache_groups[2].kv_cache_spec, block_size=400, page_size_padded=819200),
            "attention": FullAttentionSpec(
                block_size=400, num_kv_heads=4, head_size=128, dtype=torch.float16,
            ),
            "qsa.raw": FullAttentionSpec(
                block_size=400, num_kv_heads=1, head_size=140,
                head_size_v=0, dtype=torch.float16, indexes_kv_by_block_stride=True,
            ),
            "qsa.compressed": MLAAttentionSpec(
                block_size=400, num_kv_heads=1, head_size=128,
                dtype=torch.float16, compress_ratio=4,
                indexes_kv_by_block_stride=True,
            ),
        }
        result = kv_cache_utils.unify_kv_cache_spec_page_size(specs)
        self.assertEqual({spec.block_size for spec in result.values()}, {400})
        self.assertEqual({spec.page_size_bytes for spec in result.values()}, {819200})
        self.assertEqual(result["qsa.compressed"].storage_block_size, 100)
        self.assertIsNone(specs["qsa.compressed"].page_size_padded)

    def test_other_models_keep_native_page_unification(self):
        specs = {
            "attention": FullAttentionSpec(
                block_size=400, num_kv_heads=4, head_size=128, dtype=torch.float16,
            ),
            "compressed": MLAAttentionSpec(
                block_size=400, num_kv_heads=1, head_size=128,
                dtype=torch.float16, compress_ratio=4,
                indexes_kv_by_block_stride=True,
            ),
        }
        result = kv_cache_utils.unify_kv_cache_spec_page_size(specs)
        self.assertEqual(result["compressed"].block_size, 12800)


if __name__ == "__main__":
    unittest.main(argv=["test_next_prefix_cache"], verbosity=2)
