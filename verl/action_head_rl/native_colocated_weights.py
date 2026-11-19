"""Reference bucketed transport contracts for one sender per colocated GPU."""
from pathlib import Path
import os
import json


def stop_failed_native_services(services, *, timeout=30):
    """Do not queue a wake behind a receiver whose sender already failed.

    The resident supervisor reaps the owned process tree after trainer exit.
    Preserve the original publication error even if a receiver cannot stop.
    """
    import ray
    try:
        ray.get([service.stop.remote() for service in services], timeout=timeout)
    except Exception as error:
        print(json.dumps({'native_sync_cleanup_error': str(error),
                          'supervisor_cleanup_required': True}), flush=True)


def native_ipc_handle(root, rank):
    path = Path(root) / f'native-weight-{rank}.sock'
    if not path.is_absolute() or len(os.fsencode(path)) > 100 or type(rank) is not int or rank < 0:
        raise ValueError('native weight IPC requires a short private local path and TP rank')
    return 'ipc://' + str(path)


def map_native_targets(placements, targets, world):
    if len(placements) != len(targets):
        raise ValueError('one native receiver per colocated replica required')
    result = [None] * world
    endpoints = set()
    for placement, target in zip(placements, targets, strict=True):
        ranks = placement['ranks']
        if (target['cuda_visible_devices'] != placement['cuda_visible_device']
                or target['node_id'] != placement['node_id'] or target['tp_size'] != len(ranks)):
            raise ValueError('native receiver does not own the learner physical GPUs')
        for local_rank, rank in enumerate(ranks):
            handle = native_ipc_handle(target['root'], local_rank)
            endpoint = (target['node_id'], handle)
            if not 0 <= rank < world or result[rank] is not None or endpoint in endpoints:
                raise ValueError('duplicate native sender or IPC endpoint')
            endpoints.add(endpoint)
            result[rank] = dict(handle=handle, use_shm=target['use_shm'])
    if any(value is None for value in result):
        raise ValueError('every Megatron rank must have a colocated receiver')
    return result
