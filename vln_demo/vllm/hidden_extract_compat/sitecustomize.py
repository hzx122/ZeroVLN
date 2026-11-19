"""Scoped vLLM 0.18.0 VLM hidden-extraction compatibility shim.

Upstream vLLM PR #38987 fixes ExtractHiddenStatesConfig flattening a composite
VLM's ``text_config`` into a dict.  The PPU image currently contains 0.18.0
without that patch.  Restore the original PretrainedConfig object after the
upstream constructor.  This module is loaded only by vllm-vln-35B-hidden.sh;
the normal servers are unaffected.
"""
import os
import sys


def _install_vllm_vlm_hidden_fix():
    try:
        from vllm.transformers_utils.configs.extract_hidden_states import (
            ExtractHiddenStatesConfig,
        )
    except Exception:
        return

    original = ExtractHiddenStatesConfig.__init__
    if getattr(original, "_action_head_vlm_fix", False):
        return

    def fixed_init(self, model, **kwargs):
        original(self, model, **kwargs)
        text_config = model.get_text_config()
        if text_config is not model:
            self.text_config = text_config

    fixed_init._action_head_vlm_fix = True
    ExtractHiddenStatesConfig.__init__ = fixed_init


# vLLM launches fresh Python subprocesses to inspect the model registry.  They
# inherit PYTHONPATH, but must not recursively import vLLM from sitecustomize.
# The patched config object is constructed only in the top-level CLI process.
if os.path.basename(sys.argv[0]) == "vllm":
    _install_vllm_vlm_hidden_fix()
