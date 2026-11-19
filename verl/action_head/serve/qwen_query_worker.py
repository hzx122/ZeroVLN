"""vLLM worker for the configurable-query action expert side channel."""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import numpy as np

from vllm.v1.worker.gpu_worker import Worker


def _aligned_query_rows(
    req_ids: Sequence[str],
    num_computed_tokens: Sequence[int],
    num_scheduled_tokens: Sequence[int],
    requests: Mapping[str, str],
) -> list[tuple[int, int, str, int]]:
    """Map marked requests to the flattened rows used by GPUModelRunner.

    ``GPUModelRunner._prepare_inputs`` flattens tokens in persistent-batch
    order, which can differ from ``SchedulerOutput.num_scheduled_tokens``
    insertion order after vLLM compacts or reorders a concurrent batch.
    """
    if not (len(req_ids) == len(num_computed_tokens)
            == len(num_scheduled_tokens)):
        raise ValueError("query-row inputs must have the same length")
    rows = []
    start = 0
    for req_id, token_start, count in zip(
            req_ids, num_computed_tokens, num_scheduled_tokens):
        end = start + int(count)
        external_request_id = requests.get(req_id)
        if external_request_id is not None:
            rows.append((start, end, external_request_id, int(token_start)))
        start = end
    return rows


class ActionHeadQueryWorker(Worker):
    def load_model(self) -> None:
        from action_head.serve.compile_cache import configure_process_compile_cache
        configure_process_compile_cache(f"vllm_tp_rank_{self.local_rank}")
        if os.environ.get("ACTION_HEAD_ISOLATE_TP_CACHE") == "1":
            from action_head.serve.qwen_hidden_worker import isolate_triton_cache
            isolate_triton_cache(
                os.environ.get("ACTION_HEAD_COMPILE_CACHE_ROOT"), self.local_rank)
        result = super().load_model()
        from action_head.serve.dualvln_query_runtime import (
            install_query_model_hooks,
            set_worker_query_rows,
        )

        checkpoint = os.environ.get("ACTION_HEAD_QUERY_CHECKPOINT", "")
        cache_root = os.environ.get(
            "ACTION_HEAD_QUERY_CACHE", "/dev/shm/action_head_queries")
        if not checkpoint:
            raise RuntimeError("ACTION_HEAD_QUERY_CHECKPOINT is required")
        install_query_model_hooks(self.model_runner, checkpoint, cache_root)
        self._continuous_routing = None
        if os.environ.get("ACTION_HEAD_CONTINUOUS_CONTEXT") == "1":
            from action_head.serve.continuous_context import ContinuousRouting
            graph = os.environ.get('ACTION_HEAD_V8_CUDA_GRAPHS') == '1'
            if graph and os.environ.get('ACTION_HEAD_R3_CAPTURE_MODE') != 'native':
                raise ValueError('v8 graph replay requires native full R3 capture')
            self._continuous_routing = ContinuousRouting(
                self.model_runner.model, cache_root,
                graph_capacity=self.model_runner.max_num_tokens if graph else 0,
                top_k=self.model_config.hf_text_config.num_experts_per_tok if graph else 0)


        # Build row metadata only after GPUModelRunner._update_states() has
        # compacted/reordered its persistent batch.  The old worker-side code
        # used SchedulerOutput dict order before that update, which associates
        # rows with the wrong request under concurrent rollout.
        original_prepare_inputs = self.model_runner._prepare_inputs

        def prepare_inputs(scheduler_output, num_scheduled_tokens):
            requests = getattr(
                self.model_runner, "_action_head_query_requests", {})
            num_reqs = self.model_runner.input_batch.num_reqs
            rows = _aligned_query_rows(
                self.model_runner.input_batch.req_ids[:num_reqs],
                self.model_runner.input_batch.num_computed_tokens_cpu[:num_reqs],
                np.asarray(num_scheduled_tokens)[:num_reqs],
                requests,
            )
            set_worker_query_rows(rows)
            if self._continuous_routing is not None:
                self._continuous_routing.prepare(rows)
            return original_prepare_inputs(
                scheduler_output, num_scheduled_tokens)

        self.model_runner._prepare_inputs = prepare_inputs
        return result

    def execute_model(self, scheduler_output):
        """Bridge request identity from the scheduler into each TP worker.

        Logits processors execute in EngineCore, not in worker processes under
        vLLM V1.  New-request sampling parameters are therefore cached here and
        converted to flattened scheduled-token row ranges for the model hook.
        """
        from action_head.serve.dualvln_query_runtime import set_worker_query_rows

        if not hasattr(self, "_action_head_query_requests"):
            self._action_head_query_requests = {}
        requests = self._action_head_query_requests
        self.model_runner._action_head_query_requests = requests
        for request in scheduler_output.scheduled_new_reqs:
            params = request.sampling_params
            extra = getattr(params, "extra_args", None) or {}
            count = extra.get("action_head_query_count")
            if isinstance(count, int) and count > 0:
                requests[request.req_id] = extra["action_head_query_request_id"]
                if self._continuous_routing is not None:
                    self._continuous_routing.register(request, extra)

        try:
            return super().execute_model(scheduler_output)
        finally:
            set_worker_query_rows([])
            if self._continuous_routing is not None:
                self._continuous_routing.clear()
            for request_id in scheduler_output.finished_req_ids:
                external_request_id = requests.pop(request_id, None)
                if self._continuous_routing is not None:
                    self._continuous_routing.discard(external_request_id)
                discard = getattr(
                    self.model_runner, "_action_head_discard_query", None)
                if external_request_id is not None and discard is not None:
                    discard(external_request_id)
