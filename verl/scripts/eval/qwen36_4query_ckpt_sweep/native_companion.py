"""Shared read-only v5 weights; each request owns its full bounded context."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class SharedNativeProvider:
    query_count = 0
    requires_query_tail = False

    def __init__(self, model, aggregator, *, provider_factory=None):
        if not Path(aggregator).is_file():
            raise FileNotFoundError(f'trained aggregator required: {aggregator}')
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='native-condition')
        self.calls = 0
        def load():
            import torch
            from action_head.context_encoder import NativeContextProvider
            factory = provider_factory or NativeContextProvider
            return factory(model, device='cuda:0', dtype=torch.bfloat16, seed=20260908,
                           aggregator_checkpoint=aggregator, image_cache=False)
        try:
            self._provider = self._pool.submit(load).result()
        except BaseException:
            self._pool.shutdown(wait=True)
            raise
        # No episode history or feature cache is retained in this provider.
        from common import sha256_file
        self.aggregator_sha256 = sha256_file(aggregator)

    def __call__(self, *, request, response, contract):
        messages = request.get('messages', []) if request else []
        images = sum(part.get('type') in ('image', 'image_url')
                     for message in messages for part in message.get('content', [])
                     if isinstance(part, dict))
        if not 1 <= images <= 11:
            raise ValueError('native companion input must obey the brain 8+3 window')
        if not response or response.get('query_request_id'):
            raise ValueError('native companion must not request or consume queries')
        def encode():
            value = self._provider(request=request, response=response, contract=contract)
            if value.metadata['images'] != images or list(value.tokens.shape) != [1, 16, 384]:
                raise ValueError('native condition shape/image-count mismatch')
            value.metadata.update(trained_aggregator=True, aggregator_sha256=self.aggregator_sha256,
                                  brain_images=images, episode_feature_cache=False)
            self.calls += 1
            return value
        return self._pool.submit(encode).result()

    def reset(self):
        pass  # All context is request-local; there is no shared episode state.

    def close(self):
        self._pool.submit(lambda: setattr(self, '_provider', None)).result()
        self._pool.shutdown(wait=True)


def smoke():
    import argparse
    import json
    import torch
    from common import atomic_json, checkpoint_identity
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    identity = checkpoint_identity(a.checkpoint)
    report = json.loads((Path(a.checkpoint)/'report.json').read_text())
    rows = {}
    with Path(a.manifest).open() as source:
        for line in source:
            row = json.loads(line)
            rows.setdefault(row['slow_anchor_id'], row)
            if len(rows) == 2:
                break
    provider = SharedNativeProvider(report['frontend_model'], identity['aggregator_path'])
    def encode(row):
        value = provider(request={'messages': row['slow_messages']},
                         response={'text': row['slow_completion'], 'query_request_id': ''}, contract={})
        assert torch.isfinite(value.tokens).all() and value.metadata['trained_aggregator']
        return dict(anchor=row['slow_anchor_id'], shape=list(value.tokens.shape), metadata=value.metadata)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(encode, rows.values()))
        atomic_json(a.output, dict(ok=True, checkpoint=identity, requests=results,
                                  scope='two real full contexts, trained aggregator, shared owner thread; no navigation rollout'))
    finally:
        provider.close()


if __name__ == '__main__':
    smoke()
