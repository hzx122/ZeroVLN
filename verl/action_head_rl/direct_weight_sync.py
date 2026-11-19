"""Bounded NCCL-to-TP CUDA IPC fanout for BF16 Qwen transport.

One NCCL receiver feeds every TP worker. Buffers are private to this service;
all TP acknowledgements precede buffer reuse. This is transport, not a
checkpoint, and does not mutate learner parameters or optimizer state.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
import zmq


PROTOCOL = 'action_head_qwen_ipc_v1'


def ipc_handle(root: str, rank: int) -> str:
    path = Path(root) / f'weights-rank-{rank}.sock'
    if not path.is_absolute() or len(os.fsencode(path)) > 100 or rank < 0:
        raise ValueError('direct IPC needs a short absolute private runtime path')
    return 'ipc://' + str(path)


def record_metadata(digest, name, tensor):
    digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)],
                             separators=(',', ':')).encode() + b'\n')


def check_transport_tensor(name, tensor, seen, bucket_size):
    if not isinstance(name, str) or not name or name in seen:
        raise ValueError(f'invalid/duplicate direct weight key: {name!r}')
    if name.startswith('model.language_model.visual.'):
        raise ValueError('direct weights require canonical HF visual keys')
    if tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
        raise ValueError('direct Qwen floating transport must be BF16')
    if tensor.nbytes > bucket_size or tensor.nbytes == 0:
        raise ValueError(f'weight {name} ({tensor.nbytes} bytes) does not fit bucket {bucket_size}')
    seen.add(name)


def _socket(kind, handle, timeout, *, bind):
    sock = zmq.Context.instance().socket(kind)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.SNDTIMEO, int(timeout * 1000))
    sock.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
    try:
        (sock.bind if bind else sock.connect)(handle)
    except BaseException:
        sock.close(linger=0)
        raise
    return sock


class TensorParallelWeightSender:
    """Copy a single reusable source stream to one IPC buffer per TP device."""

    def __init__(self, *, root, tensor_parallel_size, bucket_size_mb, timeout):
        if tensor_parallel_size not in (1, 2) or bucket_size_mb <= 0 or timeout <= 0:
            raise ValueError('direct sync supports TP1/TP2, positive bucket and timeout')
        self.root, self.world = str(root), int(tensor_parallel_size)
        self.bucket_size = int(bucket_size_mb) << 20
        self.timeout = timeout
        self.sockets, self.buffers = [], []
        self.handles = []

    def _open(self):
        from torch.multiprocessing.reductions import reduce_tensor
        root = Path(self.root)
        if not root.is_dir() or root.is_symlink():
            raise ValueError('direct IPC root must be an existing private directory')
        if torch.cuda.device_count() != self.world:
            raise RuntimeError('relay GPU assignment differs from TP world')
        for rank in range(self.world):
            # Never unlink someone else's endpoint. Prior successful rounds
            # remove their own socket; a stale failed round needs teardown.
            handle = ipc_handle(self.root, rank)
            if Path(handle[6:]).exists():
                raise FileExistsError(handle[6:])
            sock = _socket(zmq.REQ, handle, self.timeout, bind=True)
            self.sockets.append(sock)
            self.handles.append(handle)
            with torch.cuda.device(rank):
                buffer = torch.empty(self.bucket_size, dtype=torch.uint8,
                                     device=f'cuda:{rank}')
                self.buffers.append(buffer)
                sock.send_pyobj({'protocol': PROTOCOL, 'rank': rank,
                                 'handle': reduce_tensor(buffer)})
                if sock.recv() != b'':
                    raise RuntimeError('invalid direct IPC buffer acknowledgement')

    def _copy(self, offset, weight):
        raw = weight.detach().contiguous().view(-1).view(torch.uint8)
        # Synchronous per-device copy also fences NCCL-source reuse before the
        # generator advances to its next bucket. Do not remove this fence.
        for rank, buffer in enumerate(self.buffers):
            with torch.cuda.device(rank):
                buffer[offset:offset + weight.nbytes].copy_(raw, non_blocking=False)
                torch.cuda.synchronize(rank)

    def _flush(self, metadata, last):
        for sock in self.sockets:
            sock.send_pyobj({'bucket_meta': metadata, 'is_last': last})
        for sock in self.sockets:
            if sock.recv() != b'':
                raise RuntimeError('invalid direct IPC bucket acknowledgement')

    def _close(self, *, release_buffers=True):
        for sock in self.sockets:
            sock.close(linger=0)
        for handle in self.handles:
            Path(handle[6:]).unlink(missing_ok=True)
        self.sockets.clear()
        self.handles.clear()
        if not release_buffers:
            return
        count = len(self.buffers)
        self.buffers.clear()
        # Each per-device buffer is released only after all bucket ACKs, or
        # on fatal failure while generation remains paused; no retry occurs.
        for rank in range(count):
            with torch.cuda.device(rank):
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

    async def async_send_weights(self, weights):
        digest, seen, offset, metadata = hashlib.sha256(), set(), 0, {}
        total_bytes = 0
        try:
            self._open()
            async for name, weight in weights:
                check_transport_tensor(name, weight, seen, self.bucket_size)
                if weight.device.type != 'cuda':
                    raise ValueError('NCCL relay weights must remain on CUDA')
                if offset + weight.nbytes > self.bucket_size:
                    self._flush(metadata, False)
                    offset, metadata = 0, {}
                self._copy(offset, weight)
                metadata[name] = {'shape': list(weight.shape), 'dtype': weight.dtype,
                                  'offset': offset}
                record_metadata(digest, name, weight)
                offset += weight.nbytes
                total_bytes += weight.nbytes
            if not seen:
                raise ValueError('empty direct Qwen publication')
            self._flush(metadata, True)
            result = {'tensor_count': len(seen), 'tensor_bytes': total_bytes,
                      'metadata_sha256': digest.hexdigest()}
        except BaseException:
            # Preserve CUDA IPC allocations until the owner has stopped all
            # consumer processes. An HTTP timeout does not stop their kernels.
            self._close(release_buffers=False)
            raise
        self._close()
        return result


class TensorParallelWeightReceiver:
    def __init__(self, *, root, device, rank, timeout):
        self.root, self.device, self.rank = root, device, rank
        self.timeout = timeout
        self.stats = None

    def iter_weights(self):
        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import rebuild_ipc
        sock = _socket(zmq.REP, ipc_handle(self.root, self.rank), self.timeout, bind=False)
        buffer = None
        seen, digest, total_bytes = set(), hashlib.sha256(), 0
        try:
            header = sock.recv_pyobj()
            if header.get('protocol') != PROTOCOL or header.get('rank') != self.rank:
                raise RuntimeError('direct IPC protocol/rank mismatch')
            buffer = rebuild_ipc(header['handle'], self.device.index)
            if buffer.device != self.device or buffer.dtype != torch.uint8:
                raise RuntimeError('direct IPC buffer mapped to wrong device/dtype')
            sock.send(b'')
            while True:
                message = sock.recv_pyobj()
                for name, meta in message['bucket_meta'].items():
                    size = torch.Size(meta['shape']).numel() * meta['dtype'].itemsize
                    offset = meta['offset']
                    if offset < 0 or offset + size > buffer.numel():
                        raise ValueError('direct IPC metadata exceeds buffer')
                    view = buffer[offset:offset + size].view(meta['dtype']).view(meta['shape'])
                    check_transport_tensor(name, view, seen, buffer.numel())
                    record_metadata(digest, name, view)
                    total_bytes += view.nbytes
                    # Official layerwise reload may retain a tensor until the
                    # remaining parameters of that layer arrive in a later
                    # bucket. Never let those references alias reusable IPC.
                    yield name, view.clone()
                torch.cuda.synchronize(self.device)
                sock.send(b'')
                if message['is_last']:
                    break
            if not seen:
                raise ValueError('empty direct Qwen publication')
            self.stats = {'tensor_count': len(seen), 'tensor_bytes': total_bytes,
                          'metadata_sha256': digest.hexdigest()}
        finally:
            sock.close(linger=0)
            del buffer


class DirectWeightSyncExtension:
    """Use the same layerwise Qwen loader as the verified file reload path."""

    def update_weights_from_ipc(self, *, timeout=1800, expected_query_sha256):
        if self.device is None or self.device.type != 'cuda':
            raise RuntimeError('direct Qwen reload requires CUDA TP workers')
        torch.cuda.set_device(self.device)
        config = self.model_runner.model_config
        if config.quantization is not None or config.dtype != torch.bfloat16:
            raise RuntimeError('direct sync supports the unquantized BF16 Qwen policy')
        root = os.environ['ACTION_HEAD_DIRECT_WEIGHT_SYNC_ROOT']
        receiver = TensorParallelWeightReceiver(
            root=root, device=self.device, rank=int(self.local_rank), timeout=timeout)
        model = self.model_runner.get_model()
        expected = {name for name, _ in model.named_parameters()}
        original = model.load_weights
        loaded = set()

        def checked_load(weights):
            result = original(weights)
            if not isinstance(result, set):
                raise RuntimeError('Qwen loader did not report loaded parameter names')
            loaded.update(result)
            missing = expected - loaded
            if missing:
                raise RuntimeError(f'direct Qwen loader missed parameters: {sorted(missing)[:8]}')
            return result

        model.load_weights = checked_load
        iterator = receiver.iter_weights()
        try:
            self.model_runner.reload_weights(weights_iterator=iterator,
                                             is_checkpoint_format=True)
            torch.cuda.synchronize(self.device)
            if receiver.stats is None:
                raise RuntimeError('direct Qwen loader did not consume the complete stream')
            query_reload = getattr(self.model_runner.model,
                                   '_action_head_reload_query_checkpoint', None)
            if not callable(query_reload):
                raise RuntimeError('direct Qwen reload requires the live query hot-reload hook')
            query = query_reload()
            if (query.get('checkpoint_sha256') != expected_query_sha256
                    or query.get('pending_query_phases') != 0):
                raise RuntimeError('direct query publication differs from expected checkpoint')
        finally:
            model.load_weights = original
            iterator.close()
        return {'protocol': PROTOCOL, 'tp_rank': int(self.local_rank),
                'pid': os.getpid(), 'device': str(self.device),
                'loaded_parameter_count': len(loaded),
                'loader': 'reload_weights_layerwise', 'query': query, **receiver.stats}
