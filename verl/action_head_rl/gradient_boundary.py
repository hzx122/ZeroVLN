"""Correctness-first action/text isolation for fully synchronized FSDP2.

Action backward still computes backbone derivatives needed by queries. Its
Qwen parameter gradients are discarded, preserving pre-existing text gradients
by reference (no full weight or gradient clone). This is not a backward speedup.
"""
from contextlib import contextmanager

@contextmanager
def preserve_qwen_gradients(qwen):
    parameters=list(qwen.parameters())
    saved=[p.grad for p in parameters]
    for p in parameters:
        p.grad=None
    try:
        yield
    finally:
        # Requires a completed, reduce-scattered backward, never FSDP no_sync.
        for p,grad in zip(parameters,saved):
            p.grad=grad
