"""Load verified full replicas locally when FSDP shard size is exactly one.

The original VERL FSDP2 wrapping and PyTorch local tensor copy remain in use. All
ranks must hold the same complete state (checked byte for byte) before the
loader may omit the redundant rank-0 broadcast of the frozen 35B backbone.
"""
import hashlib
import json
import os
from pathlib import Path
from types import FunctionType

import torch
import torch.distributed as dist


def state_digest(state, details=None):
    digest = hashlib.sha256()
    size = 0
    for name, tensor in state.items():
        value = tensor.to_local() if hasattr(tensor, 'to_local') else tensor
        if not torch.is_tensor(value) or value.is_meta:
            raise ValueError('local replica loading requires every tensor to be materialized')
        entry = hashlib.sha256(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8)
        # Bound host copies even for a multi-GB routed-expert weight tensor.
        for start in range(0, raw.numel(), 64 << 20):
            array = raw[start:start+(64 << 20)].cpu().numpy()
            entry.update(memoryview(array))
        digest.update(entry.digest())
        if details is not None:
            details[name] = dict(sha256=entry.hexdigest(), shape=list(value.shape),
                                 dtype=str(value.dtype), bytes=raw.numel())
        size += raw.numel()
    return digest.hexdigest(), size


def require_same_digest(digest, device):
    encoded = torch.tensor(list(bytes.fromhex(digest)), dtype=torch.uint8, device=device)
    gathered = [torch.empty_like(encoded) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, encoded)
    values = [bytes(t.cpu().tolist()).hex() for t in gathered]
    if len(set(values)) != 1:
        raise ValueError('full initialization state differs across ranks: '+str(values))


def copy_full_state_locally(model, full_state):
    targets = model.state_dict()
    if list(targets) != list(full_state):
        raise ValueError('local initialization state keys/order differ')
    pairs = []
    for name, target in targets.items():
        local = target.to_local() if hasattr(target, 'to_local') else target
        source = full_state[name]
        if (local.is_meta or source.is_meta or local.shape != source.shape
                or local.dtype != source.dtype):
            raise ValueError('local initialization layout differs: '+name)
        pairs.append((local, source))
    # FSDP2 has already allocated each full local parameter. Copy into that
    # storage in place, preserving Parameter/DTensor identities and avoiding a
    # second 70GB GPU state dict. DCP's full-state distribute_tensor still
    # broadcasts over Replicate mesh dimensions even with broadcast=False.
    with torch.no_grad():
        for target, source in pairs:
            target.copy_(source)


def load_local_replica(model, full_state, device_mesh=None, cpu_offload=None):
    if (device_mesh is None or device_mesh.shape[-1] != 1 or cpu_offload is not None
            or os.environ.get('FSDP_SIZE') != '1'):
        raise ValueError('local full-state load requires resident FSDP2 shard size one')
    source_details = {}
    before, size = state_digest(full_state, source_details)
    from rule_teacher_training_handoff import save
    audit = Path(os.environ['ACTION_HEAD_OUTPUT_DIR'])/'initialization-audit'
    save(audit/f'source-rank{dist.get_rank()}.json', dict(sha256=before, bytes=size, tensors=source_details))
    device = torch.device('cuda', torch.cuda.current_device())
    print(json.dumps(dict(event='local_replica_state_hashed', rank=dist.get_rank(),
                          sha256=before, bytes=size)), flush=True)
    require_same_digest(before, device)
    copy_full_state_locally(model, full_state)
    after, loaded_size = state_digest(model.state_dict())
    if after != before or loaded_size != size:
        raise ValueError('local FSDP2 load changed initialization bytes/layout')
    # Nonpersistent rotary buffers are absent from state_dict. Verify them
    # independently instead of silently omitting their initialization contract.
    buffers, _ = state_digest(dict(model.named_buffers()))
    require_same_digest(buffers, device)
    report = dict(ok=True, rank=dist.get_rank(), world_size=dist.get_world_size(),
        source_sha256=before, loaded_sha256=after, state_bytes=size,
        buffers_sha256=buffers, broadcast_from_rank0=False, fsdp_size=1,
        loader=__file__, torch_version=torch.__version__)
    save(audit/f'rank{dist.get_rank()}.json', report)
    print(json.dumps(dict(event='local_replica_state_loaded', **report)), flush=True)


def build_with_local_replica_loader(engine, module, original):
    if engine.engine_config.strategy != 'fsdp2' or engine.engine_config.fsdp_size != 1:
        raise ValueError('local replica load cannot change a sharded topology')
    if not isinstance(original, FunctionType) or 'fsdp2_load_full_state_dict' not in original.__globals__:
        raise ValueError('unexpected VERL FSDP construction implementation')
    namespace = dict(original.__globals__)
    namespace['fsdp2_load_full_state_dict'] = load_local_replica
    bound = FunctionType(original.__code__, namespace, original.__name__,
                         original.__defaults__, original.__closure__)
    bound.__kwdefaults__ = original.__kwdefaults__
    return bound(engine, module)
