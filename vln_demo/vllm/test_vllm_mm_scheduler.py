"""Exercise the actual pinned scheduler/cache manager at the unsafe boundary."""
from types import SimpleNamespace

import pytest

import vllm_mm_scheduler as backport
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager


class Request:
    def __init__(self, identity, *, start=0, length=20, image='image-a'):
        self.request_id = identity
        self.mm_features = [SimpleNamespace(identifier=image,
            mm_position=SimpleNamespace(offset=start, length=length))]
        self.num_computed_tokens = 0
        self.num_tokens = 100
        self.num_output_placeholders = 0
        self.use_structured_output = False
        self.has_encoder_inputs = True
        self.finished = False

    def get_num_encoder_embeds(self, index):
        return self.mm_features[index].mm_position.length

    def is_finished(self):
        return self.finished


def boundary(cls):
    scheduler = object.__new__(cls)
    scheduler._submitting_step = False
    scheduler.is_encoder_decoder = False
    scheduler.encoder_cache_manager = EncoderCacheManager(20)
    request = Request('request-a')
    scheduler.requests = {request.request_id: request}
    scheduler.finished_req_ids = set()
    scheduler.encoder_cache_manager.allocate(request, 0)
    output = SimpleNamespace(num_scheduled_tokens={request.request_id: 20},
                             has_structured_output_requests=False)
    return scheduler, request, output


def test_original_scheduler_reproduces_eviction_before_execution():
    scheduler, request, output = boundary(backport.Scheduler)
    scheduler._update_after_schedule(output)
    next_request = Request('request-b', image='image-b')
    assert scheduler.encoder_cache_manager.can_allocate(next_request, 0, 20, 0)
    assert scheduler.encoder_cache_manager.get_freed_mm_hashes() == ['image-a']
    assert request.num_computed_tokens == 20


def test_backport_retains_embedding_during_submission_then_releases_bounded_cache(monkeypatch):
    scheduler, request, output = boundary(backport.PostExecutionEncoderScheduler)
    scheduler._update_after_schedule(output)
    manager = scheduler.encoder_cache_manager
    next_request = Request('request-b', image='image-b')
    assert not manager.can_allocate(next_request, 0, 20, 0)
    assert manager.cached['image-a'] == {'request-a'} and manager.get_freed_mm_hashes() == []
    # The parent output handling is isolated; allocation/free and submission use
    # the real installed implementations. Assert release follows parent output.
    result = object()
    def processed(self, supplied, model_output):
        assert manager.cached['image-a'] == {'request-a'}
        return result
    monkeypatch.setattr(backport.Scheduler, 'update_from_output', processed)
    assert scheduler.update_from_output(output, None) is result
    assert manager.can_allocate(next_request, 0, 20, 0)
    assert manager.get_freed_mm_hashes() == ['image-a']
    manager.allocate(next_request, 0)
    assert manager.num_free_slots == 0 and manager.num_freeable_slots == 0
    assert list(manager.cached) == ['image-b']


@pytest.mark.parametrize('progress,freed', [(19, False), (20, True)])
def test_actual_progress_after_speculative_rejection_controls_release(monkeypatch, progress, freed):
    scheduler, request, output = boundary(backport.PostExecutionEncoderScheduler)
    scheduler._update_after_schedule(output)
    def processed(self, supplied, model_output):
        request.num_computed_tokens = progress
        return 'parent-output'
    monkeypatch.setattr(backport.Scheduler, 'update_from_output', processed)
    assert scheduler.update_from_output(output, None) == 'parent-output'
    assert (scheduler.encoder_cache_manager.cached['image-a'] == set()) is freed


def test_aborted_request_is_not_resurrected(monkeypatch):
    scheduler, request, output = boundary(backport.PostExecutionEncoderScheduler)
    scheduler._update_after_schedule(output)
    def processed(self, supplied, model_output):
        scheduler.encoder_cache_manager.free(request)
        scheduler.requests.pop(request.request_id)
        return 'aborted'
    monkeypatch.setattr(backport.Scheduler, 'update_from_output', processed)
    assert scheduler.update_from_output(output, None) == 'aborted'
    assert not scheduler.requests and scheduler.encoder_cache_manager.num_freeable_slots == 20


def test_other_request_reference_prevents_shared_image_eviction(monkeypatch):
    scheduler, request, output = boundary(backport.PostExecutionEncoderScheduler)
    peer = Request('peer', image='image-a')
    manager = scheduler.encoder_cache_manager
    assert manager.check_and_update_cache(peer, 0)
    scheduler.requests['peer'] = peer
    scheduler._update_after_schedule(output)
    monkeypatch.setattr(backport.Scheduler, 'update_from_output', lambda *args: None)
    scheduler.update_from_output(output, None)
    assert manager.cached['image-a'] == {'peer'}
    assert not manager.can_allocate(Request('third', image='image-b'), 0, 20, 0)
    manager.free(peer)
    assert manager.can_allocate(Request('third', image='image-b'), 0, 20, 0)


def test_constructor_rejects_async_before_starting_parent():
    config = SimpleNamespace(scheduler_config=SimpleNamespace(async_scheduling=True))
    with pytest.raises(ValueError, match='no-async'):
        backport.PostExecutionEncoderScheduler(config)


def test_constructor_refuses_changed_installed_runtime(monkeypatch):
    monkeypatch.setattr(backport, 'PINNED_SCHEDULER_SHA256', 'changed')
    config = SimpleNamespace(scheduler_config=SimpleNamespace(async_scheduling=False))
    with pytest.raises(RuntimeError, match='hash-pinned'):
        backport.PostExecutionEncoderScheduler(config)


def encoder_boundary(cls, *, budget=300, sparse=None):
    from vllm.multimodal.inputs import PlaceholderRange
    scheduler = object.__new__(cls)
    scheduler.is_encoder_decoder = False
    scheduler.scheduler_config = SimpleNamespace(disable_chunked_mm_input=False)
    scheduler.ec_connector = None
    scheduler.encoder_cache_manager = EncoderCacheManager(budget)
    request = Request('boundary', start=5395, length=300)
    request.mm_features[0].mm_position = PlaceholderRange(5395, 300, is_embed=sparse)
    return scheduler, request


def test_installed_method_reproduces_actual_missing_image_at_5360_plus_35():
    scheduler, request = encoder_boundary(backport.Scheduler)
    # The actual SAGE r3 dump: next image starts at 5395. The original method
    # schedules no encoder even though the shifted MTP range needs its first
    # embedding; it neither ran out of cache nor freed this identifier.
    assert scheduler._try_schedule_encoder_inputs(request, 5360, 35, 300, 1) == ([], 35, 300, [])


def test_shifted_window_encodes_the_actual_boundary_without_changing_target_tokens():
    scheduler, request = encoder_boundary(backport.EncoderBoundaryScheduler)
    assert scheduler._try_schedule_encoder_inputs(request, 5360, 35, 300, 1) == ([0], 35, 0, [])


def test_non_mtp_range_does_not_encode_a_future_image():
    scheduler, request = encoder_boundary(backport.EncoderBoundaryScheduler)
    assert scheduler._try_schedule_encoder_inputs(request, 5360, 35, 300, 0) == ([], 35, 300, [])


@pytest.mark.parametrize('slots,compute', [(0, 300), (300, 0)])
def test_shifted_boundary_preserves_cache_and_compute_budget_rollback(slots, compute):
    scheduler, request = encoder_boundary(backport.EncoderBoundaryScheduler, budget=slots)
    assert scheduler._try_schedule_encoder_inputs(request, 5360, 35, compute, 1) == ([], 34, compute, [])


def test_existing_boundary_image_is_referenced_without_encoding_twice():
    scheduler, request = encoder_boundary(backport.EncoderBoundaryScheduler)
    scheduler.encoder_cache_manager.allocate(request, 0)
    assert scheduler._try_schedule_encoder_inputs(request, 5360, 35, 300, 1) == ([], 35, 300, [])
    assert scheduler.encoder_cache_manager.cached['image-a'] == {'boundary'}


@pytest.mark.parametrize('first_embedding,expected', [(False, []), (True, [0])])
def test_sparse_placeholder_uses_shifted_embedding_mask(first_embedding, expected):
    import torch
    sparse = torch.ones(300, dtype=torch.bool)
    sparse[0] = first_embedding
    scheduler, request = encoder_boundary(backport.EncoderBoundaryScheduler, sparse=sparse)
    actual = scheduler._try_schedule_encoder_inputs(request, 5360, 35, 300, 1)
    assert actual[0] == expected and actual[1] == 35


def test_encoder_window_backport_refuses_a_changed_method(monkeypatch):
    monkeypatch.setattr(backport.Scheduler, '_try_schedule_encoder_inputs', lambda *args: None)
    with pytest.raises(RuntimeError, match='exactly once'):
        backport._encoder_window_backport()
