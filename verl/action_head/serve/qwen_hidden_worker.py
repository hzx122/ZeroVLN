"""vLLM worker that installs the scoped Qwen3.5/3.6 VL hidden-state fix."""
from __future__ import annotations

import os
from pathlib import Path

from vllm.v1.worker.gpu_worker import Worker


def install_qwen_vl_aux_hidden_fix() -> None:
    # This runs inside every TP worker, immediately before model construction.
    # vLLM's multiprocessing uses spawn, so a parent-process monkeypatch alone
    # is not inherited by workers.
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ForCausalLMBase,
        Qwen3_5ForConditionalGeneration,
    )

    def install_final_output_adapter(text_model):
        final_layer_id = len(text_model.layers)
        if final_layer_id not in text_model.aux_hidden_state_layers:
            return
        if getattr(text_model, "_action_head_final_output_adapter", False):
            return
        original_norm_forward = text_model.norm.forward

        def capture_final_pre_norm(hidden_states, residual=None):
            text_model._action_head_final_pre_norm = (
                hidden_states + residual if residual is not None else hidden_states
            )
            return original_norm_forward(hidden_states, residual)

        text_model.norm.forward = capture_final_pre_norm
        text_model._action_head_final_output_adapter = True

    def normalize_final_output(original_forward, text_model_getter):
        def forward(self, *args, **kwargs):
            text_model = text_model_getter(self)
            if not getattr(text_model, "_action_head_final_output_adapter", False):
                return original_forward(self, *args, **kwargs)
            text_model._action_head_final_pre_norm = None
            output = original_forward(self, *args, **kwargs)
            final_pre_norm = text_model._action_head_final_pre_norm
            if final_pre_norm is None:
                raise RuntimeError("final-output hidden state was not captured")
            if isinstance(output, tuple):
                hidden_states, aux_hidden_states = output
                return hidden_states, [*aux_hidden_states, final_pre_norm]
            return output, [final_pre_norm]

        return forward

    def set_causal_aux_layers(self, layers):
        self.model.aux_hidden_state_layers = tuple(layers)
        install_final_output_adapter(self.model)

    def set_vl_aux_layers(self, layers):
        language_model = self.language_model
        setter = getattr(language_model, "set_aux_hidden_state_layers", None)
        if setter is not None:
            setter(tuple(layers))
        else:
            language_model.model.aux_hidden_state_layers = tuple(layers)
        install_final_output_adapter(language_model.model)

    original_causal_forward = Qwen3_5ForCausalLMBase.forward
    original_vl_forward = Qwen3_5ForConditionalGeneration.forward
    Qwen3_5ForCausalLMBase.forward = normalize_final_output(
        original_causal_forward, lambda model: model.model
    )
    Qwen3_5ForConditionalGeneration.forward = normalize_final_output(
        original_vl_forward, lambda model: model.language_model.model
    )
    Qwen3_5ForCausalLMBase.set_aux_hidden_state_layers = set_causal_aux_layers
    Qwen3_5ForConditionalGeneration.set_aux_hidden_state_layers = set_vl_aux_layers


def isolate_triton_cache(cache_root: str | None, local_rank: int) -> None:
    """Keep TP ranks from concurrently replacing the same CPFS cache files."""
    if cache_root:
        rank_cache = Path(cache_root) / "triton" / f"rank_{local_rank}"
        rank_cache.mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(rank_cache)


class ActionHeadHiddenWorker(Worker):
    def load_model(self) -> None:
        # TP ranks must not write the same Triton temporary/cache files on
        # CPFS concurrently; that produced stale file handles during profiling.
        isolate_triton_cache(os.environ.get("ACTION_HEAD_COMPILE_CACHE"), self.local_rank)
        install_qwen_vl_aux_hidden_fix()
        return super().load_model()
