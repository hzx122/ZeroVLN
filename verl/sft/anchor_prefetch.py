"""One CPU batch of lookahead without copying the stateful anchor collator.

The repository's other Prefetcher is tied to mmengine and CUDA-stream data
movement; SFT needs CPU image/token preprocessing overlap instead. Keep the
existing StatefulDataLoader, sampler, and collator in this process and call
them serially in one thread. This wrapper deliberately does not export loader
state: production SFT currently resumes via its explicit data-step cursor.
"""
from concurrent.futures import ThreadPoolExecutor


class AnchorPrefetchLoader:
    def __init__(self, loader):
        self.loader = loader
        self._iterator = None
        self._pending = None

    def __len__(self):
        return len(self.loader)

    def wait_pending(self):
        # Validation uses the same processor. Finish outstanding preprocessing
        # before it starts; no new work is scheduled until training resumes.
        if self._pending is not None:
            self._pending.result()

    def close(self):
        if self._iterator is not None:
            self._iterator.close()
            self._iterator = None

    def __iter__(self):
        self.close()
        self._iterator = self._iterate()
        return self._iterator

    def _iterate(self):
        iterator = iter(self.loader)
        end = object()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="anchor-input") as pool:
            try:
                self._pending = pool.submit(next, iterator, end)
                while True:
                    batch = self._pending.result()
                    if batch is end:
                        break
                    self._pending = pool.submit(next, iterator, end)
                    yield batch
            finally:
                self._pending = None
