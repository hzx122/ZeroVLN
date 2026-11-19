"""Process-local backport of vLLM PR 38907 for the pinned PPU 0.18 runtime.

Use --scheduler-cls vllm_mm_scheduler.PostExecutionEncoderScheduler together
with --no-async-scheduling. No worker code, cache capacity, image input, target
model computation, sampling parameter, or installed vLLM file is changed.
Upstream: https://github.com/vllm-project/vllm/pull/38907
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import textwrap

from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler


PINNED_SCHEDULER_SHA256 = 'c6a24aa35f101dfc96d2b19e35a1f57c5a85a99419951224dd44dd08ddf28104'
logger = init_logger(__name__)


class PostExecutionEncoderScheduler(Scheduler):
    """Keep encoder references until synchronous model output is processed.

    The pinned scheduler advances progress before a model step and immediately
    makes those encoder entries evictable. Another request can then evict an
    embedding the current step's MTP proposal still reads. Deferring reference
    release until update_from_output matches the upstream ordering fix. The
    original bounded EncoderCacheManager still controls eviction and capacity.
    """

    def __init__(self, vllm_config, *args, **kwargs):
        source = Path(inspect.getfile(Scheduler))
        if hashlib.sha256(source.read_bytes()).hexdigest() != PINNED_SCHEDULER_SHA256:
            raise RuntimeError('encoder ordering backport requires the hash-pinned PPU scheduler')
        if vllm_config.scheduler_config.async_scheduling is not False:
            raise ValueError('encoder ordering backport requires --no-async-scheduling')
        self._submitting_step = False
        super().__init__(vllm_config, *args, **kwargs)
        logger.info('Post-execution encoder reference release enabled; scheduler SHA256=%s',
                    PINNED_SCHEDULER_SHA256)

    def _update_after_schedule(self, scheduler_output):
        self._submitting_step = True
        try:
            return super()._update_after_schedule(scheduler_output)
        finally:
            self._submitting_step = False

    def _free_encoder_inputs(self, request):
        if not self._submitting_step:
            return super()._free_encoder_inputs(request)

    def update_from_output(self, scheduler_output, model_runner_output):
        result = super().update_from_output(scheduler_output, model_runner_output)
        # The target forward and MTP draft proposal have both completed here.
        # The parent has also applied speculative rejection and removed finished
        # requests; its ordinary request-finalization cache cleanup is retained.
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(req_id)
            if request is not None and not request.is_finished() and request.has_encoder_inputs:
                super()._free_encoder_inputs(request)
        return result


def _encoder_window_backport():
    """Reuse the pinned method with the same shifted encoder window throughout.

    The original overlap check includes the MTP lookahead, but its embedding
    count excludes it. At an image boundary the latter sees zero embeddings,
    so the drafter reads an image that was never scheduled. Only this end
    expression is replaced; the original allocation, budget rollback, sparse
    masks, cache references and connector handling are reused verbatim.
    """
    if hashlib.sha256(Path(inspect.getfile(Scheduler)).read_bytes()).hexdigest() != PINNED_SCHEDULER_SHA256:
        raise RuntimeError('encoder window backport requires the hash-pinned PPU scheduler')
    original = Scheduler._try_schedule_encoder_inputs
    source = textwrap.dedent(inspect.getsource(original))
    old = 'num_encoder_tokens, num_computed_tokens + num_new_tokens - start_pos'
    new = ('num_encoder_tokens, num_computed_tokens + num_new_tokens '
           '+ shift_computed_tokens - start_pos')
    if source.count(old) != 1:
        raise RuntimeError('pinned encoder window expression did not match exactly once')
    namespace = dict(original.__globals__)
    exec(compile(source.replace(old, new), __file__, 'exec'), namespace)
    return namespace[original.__name__]


class EncoderBoundaryScheduler(PostExecutionEncoderScheduler):
    """Also encode the image needed by MTP across a chunked-prefill boundary."""

    _try_schedule_encoder_inputs = _encoder_window_backport()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info('MTP encoder window lookahead enabled; original cache budgets retained')
