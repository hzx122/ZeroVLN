"""Lossless, storage-preserving payloads for native TransferQueue.

TQ's recursive msgpack encoder sends a buffer for *each tensor occurrence*.
It does not preserve the sharing between chunks or tensor views. Torch's
archive format does, including offsets/strides; no dtype or value is changed.
Only payloads from this job's own queue are decoded here.
"""
import io
import os
import logging
import time

import torch


def load_mapped_payload(path, fetch):
    """Materialize an immutable queue archive once per host, then map privately.

    Tensor-parallel/context-parallel readers share file-backed CPU pages, not
    separate decoded tensor allocations. MAP_PRIVATE preserves per-rank writes.
    The caller retains the file until ALL learners acknowledge staging.
    """
    import fcntl
    import mmap
    from pathlib import Path
    import shutil
    import pyarrow as pa

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            value = fetch()
            temporary = path.with_suffix(f'.tmp.{os.getpid()}')
            try:
                with temporary.open('wb') as output:
                    if not isinstance(value, bytes):
                        torch.save(value, output)
                    elif value.startswith(_MAGIC):
                        output.write(memoryview(value)[len(_MAGIC):])
                    elif value.startswith(_ZSTD_MAGIC):
                        with pa.CompressedInputStream(pa.BufferReader(
                                memoryview(value)[len(_ZSTD_MAGIC):]), 'zstd') as stream:
                            shutil.copyfileobj(stream, output, length=8 * 1024 * 1024)
                    else:
                        raise ValueError('unrecognized training queue payload')
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            del value
    with torch.serialization.set_default_mmap_options(mmap.MAP_PRIVATE):
        return torch.load(path, map_location='cpu', weights_only=False, mmap=True)


_MAGIC = b'v8-tq-torch-storage-v1\0'
_ZSTD_MAGIC = b'v8-tq-torch-zstd-v1\0'
CODEC_ENV = 'ACTION_HEAD_QUEUE_PAYLOAD_CODEC'


class _CountingWriter:
    def __init__(self, stream):
        self.stream,self.bytes=stream,0
    def write(self, data):
        size=self.stream.write(data)
        self.bytes+=len(data)
        return size
    def flush(self):return self.stream.flush()


def pack_payload(value, *, metrics=None):
    codec=os.environ.get(CODEC_ENV, 'torch')
    if codec == 'zstd':
        import pyarrow as pa
        # Stream directly into compressed storage. Never retain an additional
        # full-size uncompressed archive on the rollout/preparation producer.
        archive=pa.BufferOutputStream()
        archive.write(_ZSTD_MAGIC)
        with pa.CompressedOutputStream(archive, 'zstd') as stream:
            counted=_CountingWriter(stream)
            torch.save(value, counted)
        value=archive.getvalue().to_pybytes()
        if metrics is not None:metrics.update(codec=codec,stored_bytes=len(value),uncompressed_bytes=counted.bytes+len(_MAGIC))
        return value
    if codec != 'torch':raise ValueError('unsupported training queue payload codec')
    archive = io.BytesIO()
    # PyTorch's stream archive offsets are relative to its own first write.
    # Writing the existing header first preserves the wire format while
    # avoiding a second, potentially multi-GiB bytes concatenation.
    archive.write(_MAGIC)
    torch.save(value, archive)
    value=archive.getvalue()
    if metrics is not None:metrics.update(codec=codec,stored_bytes=len(value),uncompressed_bytes=len(value))
    return value


def unpack_payload(value):
    # Old saved fixtures/queues remain readable; new producers always pack.
    if not isinstance(value, bytes):
        return value
    return _unpack_buffer(memoryview(value))


def _unpack_buffer(value):
    import pyarrow as pa
    if bytes(value[:len(_MAGIC)]) == _MAGIC:
        raw=value[len(_MAGIC):]
    elif bytes(value[:len(_ZSTD_MAGIC)]) == _ZSTD_MAGIC:
        with pa.CompressedInputStream(pa.BufferReader(value[len(_ZSTD_MAGIC):]), 'zstd') as stream:
            raw=stream.read_buffer()
    else:
        raise ValueError('unrecognized training queue payload')
    # Both encodings contain the identical torch archive: aliases, views,
    # strides, dtype, token/image/routing values and Python metadata survive.
    with pa.BufferReader(raw) as archive:
        return torch.load(archive, map_location='cpu', weights_only=False)


# Keep both PUT and GET messages bounded. Tensor byte chunks use native TQ
# buffer frames instead of embedding the entire archive in a pickle envelope.
_TRANSFER_CHUNK_BYTES = 32 * 1024 * 1024
_CHUNKED = 0x7638747131


def put_payload_fields(*, key, partition_id, fields, tag=None, replace=False):
    """Store large archives in fields of the SAME key; publish locators last.

    No write retries: a failed transfer remains a real producer failure. Native
    group completion is acknowledged only after every chunk and locator is put.
    Clearing the original key also clears all its chunks.
    """
    import transfer_queue as tq
    pending = {}
    for field, payload in fields.items():
        previous_count = 0
        if replace:
            from tensordict.tensorclass import NonTensorData
            previous = tq.kv_batch_get(keys=[key], partition_id=partition_id,
                                       select_fields=[field])[field][0]
            previous = previous.data if isinstance(previous, NonTensorData) else previous
            if torch.is_tensor(previous) and previous.numel() == 3:
                previous_count = int(previous.flatten()[1].item())
        count = (len(payload) + _TRANSFER_CHUNK_BYTES - 1) // _TRANSFER_CHUNK_BYTES
        for index in range(count):
            start = index * _TRANSFER_CHUNK_BYTES
            tq.kv_put(key=key, partition_id=partition_id,
                      fields={f'{field}__archive_chunk_{index}':
                              torch.frombuffer(bytearray(memoryview(payload)[
                                  start:start + _TRANSFER_CHUNK_BYTES]), dtype=torch.uint8)})
        # Prepared groups replace large raw rollouts with a small stub before
        # the lease is consumed. Release every old trailing chunk at that point.
        for index in range(count, previous_count):
            tq.kv_put(key=key, partition_id=partition_id,
                      fields={f'{field}__archive_chunk_{index}': torch.empty(0, dtype=torch.uint8)})
        pending[field] = torch.tensor([_CHUNKED, count, len(payload)], dtype=torch.int64)
    tq.kv_put(key=key, partition_id=partition_id, fields=pending, tag=tag)


def unpack_queue_payload(value, *, key, partition_id, field):
    """Read chunk fields individually so GET cannot recreate a giant message."""
    from tensordict.tensorclass import NonTensorData
    import transfer_queue as tq
    value = value.data if isinstance(value, NonTensorData) else value
    if not (torch.is_tensor(value) and value.dtype == torch.int64
            and value.numel() == 3 and value.flatten()[0].item() == _CHUNKED):
        return unpack_payload(value)
    _, count, size = value.flatten().tolist()
    archive = io.BytesIO()
    for index in range(count):
        name = f'{field}__archive_chunk_{index}'
        # A published archive was observed returning an empty TQ field selection.
        # Reopen only this read; never replay a put, clear, or whole rollout.
        # The published locator must remain identical throughout recovery.
        for attempt in range(4):
            selected = tq.kv_batch_get(keys=[key], partition_id=partition_id,
                                       select_fields=[name])
            if name in selected:
                chunk = selected[name][0]
                if attempt:
                    logging.getLogger(__name__).warning(
                        "queue archive field recovered partition=%s key=%s field=%s attempt=%s",
                        partition_id, key, name, attempt + 1)
                break
            logging.getLogger(__name__).warning(
                'queue archive field unavailable partition=%s key=%s field=%s '
                'attempt=%s/4 pid=%s; no payload progress',
                partition_id, key, name, attempt + 1, os.getpid())
            current = tq.kv_batch_get(keys=[key], partition_id=partition_id,
                                       select_fields=[field])[field][0]
            current = current.data if isinstance(current, NonTensorData) else current
            if not torch.is_tensor(current) or not torch.equal(current, value):
                raise RuntimeError(f'queue archive locator changed: {partition_id}/{key}/{field}')
            if attempt == 3:
                raise KeyError(f'queue archive field remains unavailable: {partition_id}/{key}/{name}')
            time.sleep(.25 * 2 ** attempt)
        chunk = chunk.data if isinstance(chunk, NonTensorData) else chunk
        archive.write(memoryview(chunk.numpy()) if torch.is_tensor(chunk) else chunk)
    if archive.tell() != size:
        raise ValueError('incomplete training queue archive')
    return _unpack_buffer(archive.getbuffer())
