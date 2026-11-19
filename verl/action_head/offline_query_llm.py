"""Offline vLLM LLM loader for action expert query-tail batch generation.

Replaces the server (HTTP) path with an in-process LLM so the data pipeline can
batch-generate slow-system ToolCalls across all anchors of one option layer in a
single ``llm.generate()`` call. Query-tail hidden export still flows through
``ActionHeadQueryWorker`` + ``install_query_model_hooks`` (safetensors), so the
existing ``VllmQueryStateProvider`` reads the same artifacts unchanged.

Mirrors ``vllm-vln-35B.sh``: TP2, limit_mm_per_prompt={"image":64}, prefix cache,
no async scheduling (query tail needs synchronous row mapping), worker_cls +
logits_processors + bootstrap PYTHONPATH + ACTION_HEAD_* env.
"""
import os
import sys
import types
from typing import Any

# PPU build: acext native lib (libacext.so) is absent, so `import acext` raises
# FileNotFoundError. vLLM's sitecustomize -> extract_hidden_states -> `from
# acext import fusedmoe_wrapper` chain crashes sitecustomize at import time,
# which skips _patch_routed_experts_buffer_capacity (MoE buffer, Qwen3.6 needs
# it). Mock acext so sitecustomize loads fully; check_acext below disables real
# acext use anyway (VLLM_*_USE_ACEXT=0), so the mock is never called.
if 'acext' not in sys.modules:
    _mock = types.ModuleType('acext')
    _mock.fusedmoe_wrapper = lambda *a, **k: None
    _mock.get_fusedmoe_status_wrapper = lambda *a, **k: None
    _mock.int8_gemm = lambda *a, **k: None
    _mock.gemm_config_select = lambda *a, **k: None
    _mock.pack_int4s = lambda *a, **k: None
    _mock.preprocess_weights_for_mixed_dgemm = lambda *a, **k: None
    _mock.get_version = lambda *a, **k: "mock"
    sys.modules['acext'] = _mock

from vllm import LLM, SamplingParams


def configure_query_tail_env(*, query_checkpoint: str, query_cache: str,
                             compile_cache_root: str = "/tmp/action_head_vllm_cache",
                             repo: str = "/workspace/ZeroVLN") -> None:
    """Set the ACTION_HEAD_* env that ActionHeadQueryWorker / hooks read.

    Must run before vLLM imports torch / builds the worker, so call before LLM().
    """
    os.environ["ACTION_HEAD_ENABLE_QUERY_TAIL"] = "1"
    os.environ["ACTION_HEAD_QUERY_CHECKPOINT"] = query_checkpoint
    os.environ["ACTION_HEAD_QUERY_CACHE"] = query_cache
    os.environ.setdefault("ACTION_HEAD_QUERY_TOKEN_ID", "248060")
    os.environ.setdefault("ACTION_HEAD_QUERY_EOS_TOKEN_ID", "248046")
    os.environ.setdefault("ACTION_HEAD_TOOL_END_TOKEN_IDS", "248059")
    os.environ.setdefault("ACTION_HEAD_RUN_TOOL_TOKEN_IDS", "5917,7681,12756")
    os.environ.setdefault("ACTION_HEAD_COMPILE_CACHE_ROOT", compile_cache_root)
    os.makedirs(f"{compile_cache_root}/torchinductor", exist_ok=True)
    os.makedirs(f"{compile_cache_root}/triton", exist_ok=True)
    os.makedirs(f"{compile_cache_root}/xdg", exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"{compile_cache_root}/torchinductor")
    os.environ.setdefault("TRITON_CACHE_DIR", f"{compile_cache_root}/triton")
    os.environ.setdefault("XDG_CACHE_HOME", f"{compile_cache_root}/xdg")
    # vLLM 0.18 compatibility patches must load in engine + worker subprocesses.
    boot = f"{repo}/verl/action_head/serve/bootstrap:{repo}/verl"
    os.environ["PYTHONPATH"] = f"{boot}:{os.environ.get('PYTHONPATH', '')}"


def build_offline_llm(*, model: str, gpus: str, query_checkpoint: str,
                      query_cache: str, max_model_len: int = 131072,
                      gpu_memory_utilization: float = 0.9,
                      enable_prefix_caching: bool = True) -> LLM:
    """Load a vLLM LLM in-process with the action expert query-tail worker.

    Equivalent to ``vllm-vln-35B.sh`` server flags, minus the HTTP serving.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    # PPU NOTE: server (vllm serve) sets PJRT_DEVICE=CUDA so vLLM's current_platform
    # is CUDA and check_acext() takes the device_capability>80 early-return (skipping
    # the acext import that throws FileNotFoundError on this PPU build). Without this
    # the offline LLM() crashes in ppu.check_acext.
    os.environ.setdefault("PJRT_DEVICE", "CUDA")
    configure_query_tail_env(query_checkpoint=query_checkpoint, query_cache=query_cache)
    # PPU build: libacext.so is absent under site-packages/lib, and check_acext()
    # only catches ImportError (not FileNotFoundError) so the offline LLM() crashes
    # in ppu.__post_init__. The server (vllm serve) skips acext via the
    # device_capability>80 early-return; mirror that here by patching check_acext
    # to disable acext without importing it.
    import vllm.config.ppu as _ppu
    from vllm.config import envs as _vllm_envs
    def _skip_acext(cls):
        _vllm_envs.VLLM_MOE_USE_ACEXT = 0
        _vllm_envs.VLLM_DENSE_USE_ACEXT = 0
    _ppu.PPUConfig.check_acext = classmethod(_skip_acext)
    return LLM(
        model=model,
        tensor_parallel_size=2,
        data_parallel_size=1,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 64},
        allowed_local_media_path="/",
        served_model_name="Qwen3.6-35B",
        worker_cls="action_head.serve.qwen_query_worker.ActionHeadQueryWorker",
        logits_processors=[
            "action_head.serve.dualvln_query_runtime:DualVlnQueryLogitsProcessor"],
        enable_prefix_caching=enable_prefix_caching,
        async_scheduling=False,  # query tail needs synchronous row mapping
        disable_log_stats=True,
    )


def batch_chat(llm: LLM, messages_batch: list[list[dict]], *,
               request_ids: list[str], temperature: float = 0.7,
               top_p: float = 0.95, max_tokens: int = 256,
               enable_thinking: bool = False) -> list[dict[str, Any]]:
    """One ``llm.chat()`` over a batch of slow-system contexts (OpenAI messages).

    Per-prompt ``SamplingParams.extra_args`` carries the query-tail ``request_id``
    so ``ActionHeadQueryWorker`` exports each request's 4-query hidden to its own
    safetensors. ``VllmQueryStateProvider`` reads them back unchanged. Messages
    use the same OpenAI format (image_url ``file://`` paths) as the server path,
    so no caller change is needed beyond switching from post_json to batch_chat.
    """
    if len(messages_batch) != len(request_ids):
        raise ValueError("messages_batch and request_ids length mismatch")
    sps = [
        SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
            extra_args={
                "action_head_query_count": 4,
                "action_head_query_request_id": rid,
            },
        )
        for rid in request_ids
    ]
    outputs = llm.chat(
        messages_batch, sps, use_tqdm=False,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )
    results = []
    for i, out in enumerate(outputs):
        text = out.outputs[0].text if out.outputs else ""
        usage = {}
        if hasattr(out, "metrics") and out.metrics and hasattr(out.metrics, "items"):
            usage = dict(out.metrics)
        results.append({
            "request_id": request_ids[i],
            "text": text,
            "usage": usage,
        })
    return results
