#!/usr/bin/env python3
"""Launch the Verl-side vLLM with the upstream VLM extraction fix scoped in.

Equivalent to merged upstream vLLM PR #38987.  A wrapper is used instead of
sitecustomize so vLLM's model-registry and engine subprocesses do not import the
whole package during Python startup.
"""
import os
import traceback


def _write_diagnostic(exc_type, exc, tb) -> None:
    path = os.environ.get("ACTION_HEAD_VLLM_DIAGNOSTIC")
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            traceback.print_exception(exc_type, exc, tb, file=handle)


def run() -> None:
    # Keep every vLLM import inside the real CLI entry point.  Python spawn
    # reloads this file as ``__mp_main__`` for registry/engine subprocesses;
    # top-level imports here would recursively initialize vLLM in each child.
    from vllm.transformers_utils.configs.extract_hidden_states import (
        ExtractHiddenStatesConfig,
    )

    original_init = ExtractHiddenStatesConfig.__init__

    def fixed_init(self, model, **kwargs):
        original_init(self, model, **kwargs)
        text_config = model.get_text_config()
        if text_config is not model:
            self.text_config = text_config

    ExtractHiddenStatesConfig.__init__ = fixed_init

    # vLLM 0.18's Qwen3.5-VL wrapper falls through to the generic
    # EagleModelMixin setter.  That setter assumes ``parent_ref.model`` itself
    # is an Eagle model and asserts during model load.  Qwen3.5 actually keeps
    # the extraction-capable language model one level deeper.  Upstream's
    # Qwen3.5 implementation now sets ``model.aux_hidden_state_layers``
    # directly; backport that narrowly for this launcher.
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ForCausalLMBase,
        Qwen3_5ForConditionalGeneration,
    )

    def set_causal_aux_layers(self, layers):
        self.model.aux_hidden_state_layers = tuple(layers)

    def set_vl_aux_layers(self, layers):
        language_model = self.language_model
        setter = getattr(language_model, "set_aux_hidden_state_layers", None)
        if setter is not None:
            setter(tuple(layers))
        else:
            language_model.model.aux_hidden_state_layers = tuple(layers)

    Qwen3_5ForCausalLMBase.set_aux_hidden_state_layers = set_causal_aux_layers
    Qwen3_5ForConditionalGeneration.set_aux_hidden_state_layers = set_vl_aux_layers

    from vllm.entrypoints.cli.main import main

    try:
        main()
    except BaseException as exc:
        _write_diagnostic(type(exc), exc, exc.__traceback__)
        raise


if __name__ == "__main__":
    run()
