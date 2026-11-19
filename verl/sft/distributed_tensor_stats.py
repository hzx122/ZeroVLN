"""Weights for summing local parameter statistics across a DP process group."""
import math

from torch.distributed.tensor import DTensor


def unique_shard_weight(parameter, world_size: int) -> float:
    """Count each logical parameter element once under FSDP, HSDP, or DDP.

    Callers SUM local squared norms/counts over the complete DP group. HSDP
    shards have replicas along the non-sharded mesh axes. A plain parameter
    is replicated on every rank. This concerns diagnostics, not gradient loss
    scaling or clipping, which remain owned by FSDP/DTensor.
    """
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    shards = 1
    if isinstance(parameter, DTensor):
        if any(p.is_partial() for p in parameter.placements):
            raise ValueError("parameter statistics require resolved DTensor placements")
        shards = math.prod(parameter.device_mesh.size(axis)
                           for axis, placement in enumerate(parameter.placements)
                           if placement.is_shard())
    if world_size % shards:
        raise ValueError("statistics group must contain all parameter shards")
    return shards / world_size
