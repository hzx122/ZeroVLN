"""Native TransferQueue storage for prepared, already credited records.

Store each complete condition once, preserving shared chunk payloads. Scheduling
still uses the established full-record sharder; only its wire representation
changes to small keys. Learners fetch independently without per-wave barriers.
"""
from __future__ import annotations

from collections import OrderedDict
import uuid
import re

from action_head_rl.queue_payload import pack_payload, unpack_payload


REFERENCE = '_transfer_queue_record'
# These are the only changes made by the established sharder/padding path.
OVERRIDES = ('condition_id', 'sample_valid', 'advantage',
             '_condition_pack_batch', '_condition_pack_token_width')
IDENTITY = ('record_kind', 'policy_version', 'update_policy_version', 'loss_weight', 'token_count')


def stage_queued_records(learner, rank_records, records_per_fetch):
    """VERL DP_COMPUTE dispatches every argument, including scalar options."""
    if len(rank_records) != learner.world_size:
        raise ValueError('queued staging needs one schedule per physical learner')
    return learner.stage_queued_record_batches(
        rank_records, records_per_fetch=[records_per_fetch] * learner.world_size)


def initialize_record_queue(*, storage_units=32):
    import ray
    import transfer_queue as tq
    from omegaconf import OmegaConf
    if tq.__version__ != '0.1.10':
        raise ValueError('prepared trajectory transport requires reference TransferQueue 0.1.10')
    try:
        ray.get_actor('TransferQueueController', namespace='transfer_queue')
    except ValueError:
        pass
    else:
        raise RuntimeError('refuse to reuse another live TransferQueue controller')
    return tq.init(OmegaConf.create(dict(backend=dict(storage_backend='SimpleStorage',
        SimpleStorage=dict(num_data_storage_units=storage_units, total_storage_size=100000)))))


def publish_record_groups(records, *, partition='v8_records', conditions_per_put=4):
    """Publish bounded batches and attach immutable locators to original rows.

    Caller owns the lease until every rank's staging acknowledgment. The driver
    must release its local heavy records after compacting the sharded streams.
    """
    import transfer_queue as tq
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    from action_head_rl.worker import canonicalize_option_payload
    from action_head_rl.text_policy import is_text_record

    if not records or conditions_per_put < 1:
        raise ValueError('record queue requires data and positive bounded puts')
    groups = OrderedDict()
    for row in records:
        if REFERENCE in row or not row.get('condition_id'):
            raise ValueError('publish unqueued records with a condition identity exactly once')
        groups.setdefault(row['condition_id'], []).append(row)
    prefix = uuid.uuid4().hex
    keys = [f'{prefix}_{i}' for i in range(len(groups))]
    payloads = list(groups.values())
    for rows in payloads:
        if any(is_text_record(row) for row in rows):
            if len(rows) != 1:
                raise ValueError('one complete text trajectory per queue condition required')
        elif 'slow_request' in rows[0]:
            canonicalize_option_payload(rows)
    serialized_bytes = 0
    for start in range(0, len(keys), conditions_per_put):
        selected = payloads[start:start + conditions_per_put]
        batch = TensorDict({}, batch_size=[len(selected)])
        # Copy only dictionaries so attaching the locator AFTER put cannot
        # mutate a locally retained payload (e.g. an in-process transport).
        packed = [pack_payload([dict(row) for row in rows]) for rows in selected]
        serialized_bytes += sum(map(len, packed))
        tu.assign_non_tensor_stack(batch, 'records', packed)
        tq.kv_batch_put(keys=keys[start:start + conditions_per_put], partition_id=partition, fields=batch)
        del packed, batch
    for key, rows in zip(keys, payloads, strict=True):
        for index, row in enumerate(rows):
            row[REFERENCE] = dict(partition=partition, key=key, index=index,
                                  condition_id=row['condition_id'])
    return dict(partition=partition, keys=keys, records=len(records), serialized_bytes=serialized_bytes)


def record_locator(row):
    if REFERENCE not in row:
        raise ValueError('all scheduled records must have an acknowledged queue locator')
    return {key: row[key] for key in (REFERENCE, *IDENTITY, *OVERRIDES) if key in row}


def bind_prepared_locator(row, *, step, rollout_index):
    """Bind an early prepared group to native selection without moving tensors.

    Only the update version and the numeric scheduling prefix may change.
    Source policy, behavior probabilities, suffixes and credit stay immutable.
    The learner independently checks this binding against the stored payload.
    """
    ref = dict(row[REFERENCE])
    if 'preparation_binding' in ref:
        raise ValueError('prepared record may be bound only once')
    original = dict(row)
    original['condition_id'] = ref['condition_id']
    binding = dict(step=step, rollout_index=rollout_index)
    bound = _bind_prepared_record(original, binding)
    ref['preparation_binding'] = binding
    bound[REFERENCE] = ref
    return bound


def _bind_prepared_record(original, binding):
    match = re.fullmatch(r'step=(\d+)/rollout=(\d+)/(option=.+|text=\d+/whole-trajectory)',
                         original['condition_id'])
    if (not match or set(binding) != {'step', 'rollout_index'}
            or type(binding['step']) is not int or type(binding['rollout_index']) is not int
            or binding['step'] < int(match[1]) or binding['rollout_index'] < 0
            or original.get('update_policy_version') != int(match[1]) - 1):
        raise ValueError('invalid prepared update binding')
    return dict(original, update_policy_version=binding['step'] - 1,
                condition_id=f"step={binding['step']}/rollout={binding['rollout_index']}/{match[3]}")


def resolve_record_locators(locators, cache, *, mapped_directory=None, mapped_paths=None):
    """Fetch each option once and preserve existing padding/packing metadata."""
    import transfer_queue as tq
    from tensordict.tensorclass import NonTensorData
    if not locators or any(REFERENCE not in row for row in locators):
        raise ValueError('queued staging requires only queue locators')
    partition = locators[0][REFERENCE]['partition']
    if any(row[REFERENCE]['partition'] != partition for row in locators):
        raise ValueError('a queue staging wave must belong to one partition')
    missing = list(dict.fromkeys(row[REFERENCE]['key'] for row in locators
                                if (partition, row[REFERENCE]['key']) not in cache))
    if missing and mapped_directory is not None:
        import hashlib
        from pathlib import Path
        from action_head_rl.queue_payload import load_mapped_payload
        for key in missing:
            name = hashlib.sha256((partition + '\0' + key).encode()).hexdigest()
            path = Path(mapped_directory) / (name + '.pt')
            def fetch():
                value = tq.kv_batch_get(keys=[key], partition_id=partition,
                                       select_fields=['records'])['records'][0]
                return value.data if isinstance(value, NonTensorData) else value
            cache[partition, key] = load_mapped_payload(path, fetch)
            mapped_paths.add(path)
    elif missing:
        fields = tq.kv_batch_get(keys=missing, partition_id=partition, select_fields=['records'])
        for index, key in enumerate(missing):
            rows = fields['records'][index]
            cache[partition, key] = unpack_payload(rows.data if isinstance(rows, NonTensorData) else rows)
    resolved = []
    for locator in locators:
        ref = locator[REFERENCE]
        rows = cache[partition, ref['key']]
        if type(ref['index']) is not int or not 0 <= ref['index'] < len(rows):
            raise ValueError('queued record index out of range')
        original = rows[ref['index']]
        if original['condition_id'] != ref['condition_id']:
            raise ValueError('queued condition identity mismatch')
        if 'preparation_binding' in ref:
            original = _bind_prepared_record(original, ref['preparation_binding'])
        if any(locator.get(key) != original.get(key) for key in IDENTITY):
            raise ValueError('queued record credit/version identity changed')
        valid = float(locator.get('sample_valid', 1.))
        if valid not in (0., 1.) or (valid == 1. and any(
                locator.get(key) != original.get(key) for key in ('condition_id', 'advantage'))):
            raise ValueError('queued live record credit/identity cannot be overridden')
        if valid == 0. and locator.get('advantage') not in (0., original.get('advantage')):
            raise ValueError('queued padding must preserve or zero original advantage')
        if set(locator) - {REFERENCE, *IDENTITY, *OVERRIDES}:
            raise ValueError('unexpected override on queued record')
        record = dict(original)
        for key in OVERRIDES:
            if key in locator:
                record[key] = locator[key]
        resolved.append(record)
    return resolved


def release_record_groups(lease):
    import transfer_queue as tq
    tq.kv_clear(keys=lease['keys'], partition_id=lease['partition'])
