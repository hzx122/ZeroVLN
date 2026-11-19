"""Decision-boundary rollout producer for the native VERL async replay buffer.

Only complete K-sample groups are trainable. Unfinished navigation episodes
remain on their original stateful actors across a colocated update. Heavy
rollout data goes directly from those actors to TransferQueue; this driver
holds keys and small lifecycle metadata, never rollout tensors.
"""
from __future__ import annotations

from collections import deque
import json
import importlib.util
from pathlib import Path
import time
import uuid

from action_head_rl.queue_payload import pack_payload, put_payload_fields, unpack_queue_payload


class PartialCaseGroups:
    """Consume the existing seeded case schedule in complete admission blocks."""

    def __init__(self, sample_cases, episode_specs, *, group_size, start_block=0):
        self.sample_cases, self.episode_specs = sample_cases, episode_specs
        self.group_size, self.block, self.pending = group_size, start_block, deque()

    def __call__(self, count, policy_version):
        groups = []
        while len(groups) < count:
            if not self.pending:
                self.block += 1
                cases = self.sample_cases(self.block, policy_version)
                specs = self.episode_specs(cases, step=self.block, group_size=self.group_size)
                if not specs or len(specs) != len(cases) * self.group_size:
                    raise ValueError('partial admission requires complete seeded groups')
                self.pending.extend([[(row[2], row[3]) for row in specs[start:start + self.group_size]]
                                     for start in range(0, len(specs), self.group_size)])
            groups.append(self.pending.popleft())
        return groups


def admission_io():
    path = Path(__file__).resolve().parents[2] / 'scripts/train/align_three_source_observations.py'
    spec = importlib.util.spec_from_file_location('_partial_admission_io', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_admission_state(checkpoint, *, step, block):
    admission_io().write_json(Path(checkpoint) / 'rollout_admission.json', dict(
        schema='partial_admission_boundary_v1', step=step,
        last_admitted_block=block, pending_rollouts='discard_and_resample'))


def resume_admission_state(checkpoint):
    """Resume at an admission boundary; live simulators/queue are not durable."""
    path = Path(checkpoint)
    step = int(path.name.rsplit('_', 1)[1])
    saved = path / 'rollout_admission.json'
    io = admission_io()
    if io.exists(saved):
        state = io.load(saved)
        if state.get('step') != step or state.get('schema') != 'partial_admission_boundary_v1':
            raise ValueError('partial admission checkpoint mismatch')
        block, source = state['last_admitted_block'], str(saved)
    else:
        # Legacy checkpoints saved no cursor. Admissions under policy < N
        # preceded update N; policy N admissions belong to the failed next step.
        source = str(path.parent / 'partial_admissions.jsonl')
        rows = [json.loads(line) for line in io.read_bytes(source).decode().splitlines() if line.strip()]
        blocks = [row['block'] for row in rows if row['admission_policy_version'] < step]
        if not blocks:
            raise ValueError('cannot reconstruct legacy partial admission boundary')
        block = max(blocks)
    if type(block) is not int or block < 0:
        raise ValueError('invalid partial admission block')
    return dict(last_admitted_block=block, source=source,
                rollout_state_restored=False, pending_rollouts='discard_and_resample')


def leased_group_keys(batch, *, group_size):
    groups = {}
    for key in batch.keys:
        groups.setdefault(key.split('_')[0], []).append(key)
    for index, (uid, keys) in enumerate(groups.items()):
        expected = [f'{uid}_{session}_0' for session in range(group_size)]
        if len(keys) != group_size or set(keys) != set(expected):
            raise ValueError('partial batch contains an incomplete or duplicate group')
        yield index, uid, expected


def read_rollout_group(keys, partition, *, field='rollout'):
    import transfer_queue as tq
    values = tq.kv_batch_get(keys=keys, partition_id=partition, select_fields=[field])[field]
    rows = [unpack_queue_payload(row, key=key, partition_id=partition, field=field)
            for key, row in zip(keys, values, strict=True)]
    if any(row['case'] != rows[0]['case'] for row in rows):
        raise ValueError('partial group changed its case')
    return rows


def iter_leased_rollout_groups(batch, *, group_size):
    """Read one complete K-group at a time; preserve native buffer selection."""
    for index, uid, keys in leased_group_keys(batch, group_size=group_size):
        yield index, uid, read_rollout_group(keys, batch.partition_id)


def store_partial_rollout(output, destination):
    """Persist on the producing actor before acknowledging session completion."""
    import transfer_queue as tq

    versions = output['partial_rollout']
    valid = output['result'].get('evaluation_valid') is True and not output['result'].get('infra_error')
    tag = dict(status='success' if valid else 'failure',
               global_steps=destination['global_steps'],
               expected_num_sessions=destination['group_size'],
               min_global_steps=versions['min_policy_version'],
               max_global_steps=versions['max_policy_version'])
    payload_stats={}
    payload = pack_payload(output,metrics=payload_stats)
    fields = {'rollout': payload}
    if destination.get('credit_metadata'):
        from action_head_rl.rollout_records import credit_metadata
        fields['credit_metadata'] = pack_payload(credit_metadata(output))
    put_payload_fields(key=destination['key'], partition_id=destination['partition_id'],
                       fields=fields, tag=tag)
    return dict(done=True, key=destination['key'], valid=valid,
                payload_bytes=len(payload),payload_memory_bytes=payload_stats['uncompressed_bytes'],
                payload_codec=payload_stats['codec'],
                min_policy_version=versions['min_policy_version'],
                max_policy_version=versions['max_policy_version'])


class PartialGroupRollouts:
    """Own admission, quiescence and leases; reuse native selection/stale drop.

    ``next_groups(count, policy_version)`` supplies whole groups of (case, seed)
    pairs. It owns the existing case sampler and its cursor. ``sample`` adds a
    normal batch, plus one extra batch on first use (the reference warmup=1).
    Native ReplayBufferAsync refills dropped terminal groups one-for-one.

    A returned batch is a lease: release it only AFTER every learner has
    acknowledged its data. No outstanding tool/model RPC remains when sample
    returns, so callers may then sleep vLLM and change both policies. The next
    sample resumes retained actors only after the caller acknowledges weights.
    """

    def __init__(self, workers, *, group_size, batch_groups, next_groups,
                 num_warmup_batches=1, max_off_policy_threshold=4,
                 wait_one=None, resolve=None, preparation=None):
        import ray
        import transfer_queue as tq
        from omegaconf import OmegaConf
        from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync

        if (not workers or type(group_size) is not int or group_size < 1
                or type(batch_groups) is not int or batch_groups < 1
                or type(num_warmup_batches) is not int or num_warmup_batches < 0):
            raise ValueError('positive group/batch sizes, workers and nonnegative warmup required')
        self.workers, self.group_size, self.batch_groups = workers, group_size, batch_groups
        self.next_groups, self.warmup = next_groups, num_warmup_batches
        self.wait_one = wait_one or (lambda refs: next(iter(ray.wait(list(refs), num_returns=1,
            timeout=1. if preparation is not None else None)[0]), None))
        self.resolve = resolve or ray.get
        self.tq = tq
        self.preparation = preparation
        self.partition = 'train'
        self.groups, self.leased = {}, set()
        self.prepared_raw_keys = set()
        self.waiting, self.idle = deque(), deque(range(len(workers)))
        self.cursors, self.pending = {}, {}
        self.policy_version = None
        self.started = self.running = self.closed = False
        self.stats = dict(admitted_groups=0, completed_groups=0, completed_sessions=0,
                          resumed_decisions=0, pause_seconds=0.,
                          cpu_preparation_drain_seconds=0.,
                          native_metadata_syncs=0, native_metadata_seconds=0.)

        producer = self

        class ProducerBuffer(ReplayBufferAsync):
            def _sync_metadata_from_transfer_queue(self):
                started = time.perf_counter()
                try:
                    return super()._sync_metadata_from_transfer_queue()
                finally:
                    producer.stats['native_metadata_syncs'] += 1
                    producer.stats['native_metadata_seconds'] += time.perf_counter() - started

            def _wait_for_next_poll(self, partition_id, last_debug_time):
                # Drive existing actor RPCs instead of sleeping while no
                # producer thread exists. Intermediate decisions do not make
                # a group sampleable: avoid scanning every prepared-record
                # key after each one. Wake native selection at the FIRST
                # whole-group transition, or periodically for native checks.
                producer._poll_until_group_change()
                return last_debug_time

        self.buffer = ProducerBuffer(
            trainer_mode='colocate_async', trainer_config=OmegaConf.create({}),
            max_off_policy_threshold=max_off_policy_threshold,
            max_off_policy_strategy='drop', sampler_kwargs=OmegaConf.create({}),
            refill_fn=self._refill)

    def _refill(self, count):
        # Native eviction already removed terminal groups. No sibling can
        # write after that transition; discard only the driver's small state.
        present = self.tq.kv_list().get(self.partition, {})
        for uid, state in list(self.groups.items()):
            if uid not in present and uid not in self.leased:
                if state['settled'] != self.group_size:
                    raise RuntimeError('native queue removed an unfinished navigation group')
                if self.preparation is not None:
                    self.preparation.discard(uid)
                del self.groups[uid]
        supplied = list(self.next_groups(count, self.policy_version))
        if len(supplied) != count:
            raise ValueError('case sampler must supply exactly the requested whole groups')
        for group in supplied:
            if len(group) != self.group_size or len({seed for _, seed in group}) != self.group_size:
                raise ValueError('partial groups require K independent seeds')
            if any(case != group[0][0] for case, _ in group):
                raise ValueError('all partial group sessions must use the same case')
        for group in supplied:
            uid = uuid.uuid4().hex
            self.tq.kv_put(key=uid, partition_id=self.partition,
                tag=dict(is_prompt=True, status='pending', global_steps=self.policy_version))
            self.groups[uid] = dict(settled=0, failed=False, raw_bytes=0, global_steps=self.policy_version,
                                    case=group[0][0])
            for session, (case, seed) in enumerate(group):
                self.waiting.append((uid, session, case, seed))
            self.stats['admitted_groups'] += 1
        if self.running:
            self._dispatch()

    def _track(self, handle, worker):
        if handle is None or handle in self.pending:
            raise RuntimeError('partial rollout returned an invalid/duplicate RPC handle')
        self.pending[handle] = worker

    def _dispatch(self):
        if not self.running:
            return
        allowed_uid = self.preparation.dispatch_uid(self) if self.preparation is not None else None
        busy = set(self.pending.values())
        for worker in self.cursors:
            uid = self.cursors[worker]['key'].split('_')[0]
            if worker not in busy and (allowed_uid is None or uid == allowed_uid):
                self._track(self.workers[worker].advance_partial_rollout.remote(self.policy_version), worker)
                self.stats['resumed_decisions'] += 1
        while self.idle and self.waiting:
            if allowed_uid is not None:
                entry = next((row for row in self.waiting if row[0] == allowed_uid), None)
                if entry is None:
                    break
                self.waiting.remove(entry)
            else:
                entry = self.waiting.popleft()
            worker = self.idle.popleft()
            uid, session, case, seed = entry
            destination = dict(key=f'{uid}_{session}_0', partition_id=self.partition,
                               global_steps=self.groups[uid]['global_steps'], group_size=self.group_size)
            if self.preparation is not None:
                destination['credit_metadata'] = True
            self.cursors[worker] = destination
            self.tq.kv_put(key=uid, partition_id=self.partition, tag={'status': 'running'})
            self._track(self.workers[worker].begin_partial_rollout.remote(
                case, seed, transfer_queue_destination=destination), worker)

    def _poll_until_group_change(self):
        completed = self.stats['completed_groups']
        deadline = time.monotonic() + 1.
        while self.stats['completed_groups'] == completed:
            if not self._poll() or time.monotonic() >= deadline:
                return

    def _poll(self):
        if self.preparation is not None:
            self.preparation.pump(self)
        cpu_pending = self.preparation.handles() if self.preparation is not None else ()
        if not self.pending and not cpu_pending:
            raise RuntimeError('partial queue cannot make progress without an active rollout')
        handle = self.wait_one(tuple(self.pending) + cpu_pending)
        if handle is None:
            return False  # Keep native checks fresh during slow model/tool RPCs.
        if handle in cpu_pending:
            self.preparation.complete(handle, self)
            self._dispatch()
            return True
        if handle not in self.pending:
            raise RuntimeError('wait returned an unknown partial rollout handle')
        worker = self.pending.pop(handle)
        result = self.resolve(handle)  # A failed RPC is a real failure, not a synthetic trajectory.
        if result['done']:
            destination = self.cursors[worker]
            if result.get('key') != destination['key'] or 'rollout' in result:
                raise RuntimeError('partial actor must acknowledge stored data without returning payloads')
            uid = destination['key'].split('_')[0]
            group = self.groups[uid]
            group['settled'] += 1
            group['raw_bytes'] += result.get('payload_bytes', 0)
            # Admission must budget the decoded working set, not compressed
            # wire size. Otherwise a good compression ratio admits too many
            # concurrent preparations and can recreate host memory pressure.
            expanded=result.get('payload_memory_bytes',result.get('payload_bytes',0))
            group['raw_memory_bytes']=group.get('raw_memory_bytes',0)+expanded
            group['max_raw_episode_bytes'] = max(group.get('max_raw_episode_bytes', 0),expanded)
            group['failed'] |= not result['valid']
            if 'cpu_memory' in result:
                group.setdefault('actor_cpu_releases', []).append(
                    dict(worker=worker, **result['cpu_memory']))
            self.stats['completed_sessions'] += 1
            if group['settled'] == self.group_size:
                self.tq.kv_put(key=uid, partition_id=self.partition,
                    tag={'status': 'failure' if group['failed'] else 'finished'})
                self.stats['completed_groups'] += 1
                if self.preparation is not None:
                    self.preparation._event('group_complete', uid, valid=not group['failed'],
                        raw_bytes=group['raw_bytes'],raw_memory_bytes=group.get('raw_memory_bytes',0), **(
                            {'actor_cpu_releases': group['actor_cpu_releases']}
                            if 'actor_cpu_releases' in group else {}))
            del self.cursors[worker]
            self.idle.append(worker)
        else:
            # Reuse the small acknowledgment already returned at a decision
            # boundary. Do not fetch/serialize live trajectories to audit them.
            self.cursors[worker]['residency'] = {name: result[name] for name in
                ('decisions', 'min_policy_version', 'max_policy_version', 'actor_pid')
                if name in result}
        self._dispatch()
        if self.preparation is not None:
            self.preparation.pump(self)
        return True

    def pause(self):
        self.running = False
        started = time.monotonic()
        while self.pending:
            self._poll()
        self.stats['pause_seconds'] += time.monotonic() - started

    def sample(self, policy_version):
        if self.closed or self.running or self.leased:
            raise RuntimeError('release the previous batch and acknowledge policies before sampling')
        if type(policy_version) is not int or policy_version < 0:
            raise ValueError('policy version must be a nonnegative integer')
        if self.policy_version is not None and policy_version < self.policy_version:
            raise ValueError('policy version cannot move backwards')
        if self.preparation is not None:
            self.preparation.begin_sampling()
        self.policy_version = policy_version
        self._refill(self.batch_groups * (1 + (self.warmup if not self.started else 0)))
        self.started = self.running = True
        try:
            if self.preparation is not None:
                # A full prepared carryover can be selected immediately. The
                # first node probes must resolve before dispatch, so new actors
                # start under a fresh memory decision even in that fast path.
                self.preparation.refresh_before_dispatch(self)
            self._dispatch()
            batch, metrics = self.buffer.sample(
                global_steps=policy_version, partition_id=self.partition, batch_size=self.batch_groups)
            self.leased = {key.split('_')[0] for key in batch.keys}
            if len(self.leased) != self.batch_groups or len(batch.keys) != self.batch_groups * self.group_size:
                raise RuntimeError('native buffer returned incomplete navigation groups')
            self.pause()
            if self.preparation is not None:
                self.preparation._event('sampling_paused', '', policy_version=policy_version,
                    pending_rollout_rpcs=len(self.pending),
                    pending_preparations=len(self.preparation.pending),
                    ready_groups=len(self.preparation.ready))
                drain_started = time.perf_counter()
                self.preparation.finish(self)
                drained_seconds = time.perf_counter() - drain_started
                self.stats['cpu_preparation_drain_seconds'] += drained_seconds
                self.preparation._event('preparation_drained', '', policy_version=policy_version,
                    seconds=drained_seconds, pending_rollout_rpcs=len(self.pending),
                    pending_preparations=len(self.preparation.pending),
                    ready_groups=len(self.preparation.ready))
            return batch, metrics
        except BaseException:
            self.running = False
            raise

    def release(self, batch):
        if self.running or self.pending or batch.partition_id != self.partition:
            raise RuntimeError('release requires a quiescent owned batch')
        uids = {key.split('_')[0] for key in batch.keys}
        expected = {f'{uid}_{session}_0' for uid in self.leased for session in range(self.group_size)}
        if not self.leased or uids != self.leased or set(batch.keys) != expected or len(batch.keys) != len(expected):
            raise ValueError('release must match the complete outstanding batch')
        remaining = [key for key in batch.keys if key not in self.prepared_raw_keys]
        if remaining:
            self.tq.kv_clear(keys=remaining, partition_id=self.partition)
        for uid in self.leased:
            if self.preparation is not None:
                self.preparation.discard(uid)
            del self.groups[uid]
        self.leased.clear()
        self.prepared_raw_keys.clear()

    def acknowledge_prepared_group(self, uid):
        """Replace heavy source fields only; retain native selection metadata.

        Unlike a leased-batch clear this is legal during sampling: all K
        siblings are already settled, and the pipeline owns their prepared
        lease. Unfinished groups and actors cannot enter this transition.
        """
        state = self.groups.get(uid)
        if (self.preparation is None or uid not in self.preparation.ready
                or state is None or state['settled'] != self.group_size
                or state['failed'] or state.get('raw_replaced')):
            raise ValueError('raw replacement requires one acknowledged complete group')
        stub = pack_payload(dict(prepared_group=uid))
        for i in range(self.group_size):
            put_payload_fields(key=f'{uid}_{i}_0', partition_id=self.partition, fields={'rollout': stub}, replace=True)
        state['raw_replaced'] = True

    def release_prepared_raw_group(self, batch, keys):
        """Transfer raw payload ownership only after a complete prepared put ack."""
        expected = {f'{uid}_{session}_0' for uid in self.leased for session in range(self.group_size)}
        if (self.running or self.pending or batch.partition_id != self.partition
                or set(batch.keys) != expected or len(keys) != self.group_size
                or len(set(keys)) != self.group_size or not set(keys) <= expected
                or len({key.split('_')[0] for key in keys}) != 1
                or self.prepared_raw_keys.intersection(keys)):
            raise ValueError('raw release requires one complete unreleased owned group')
        self.tq.kv_clear(keys=keys, partition_id=self.partition)
        self.prepared_raw_keys.update(keys)

    def batch_cases(self, batch):
        uids = list(dict.fromkeys(key.split('_')[0] for key in batch.keys))
        if set(uids) != self.leased or batch.partition_id != self.partition:
            raise ValueError('case list requires the outstanding partial batch')
        return [self.groups[uid]['case'] for uid in uids]

    def audit_released_batch(self):
        """Check the queue owns only bounded, unfinished/surplus warmup data."""
        if self.running or self.pending or self.leased or self.prepared_raw_keys:
            raise RuntimeError('queue memory audit requires a released, paused batch')
        present = self.tq.kv_list()
        prepared = self.preparation.owned_keys() if self.preparation is not None else set()
        if set(present.get('v8_records', {})) != prepared:
            raise RuntimeError('unowned prepared payloads survived learner staging acknowledgments')
        if self.preparation is not None and not set(self.preparation.ready) <= set(self.groups):
            raise RuntimeError('prepared queue lost its native replay group')
        allowed = {key for uid in self.groups for key in
                   [uid, *(f'{uid}_{i}_0' for i in range(self.group_size))]}
        if set(present.get(self.partition, {})) - allowed:
            raise RuntimeError('consumed rollout payloads survived queue release')
        if len(self.groups) > self.batch_groups * self.warmup:
            raise RuntimeError('partial replay retained more than its configured warmup bound')
        cursors = []
        for worker, cursor in self.cursors.items():
            uid = cursor['key'].split('_')[0]
            group = self.groups[uid]
            cursors.append(dict(worker=worker, key=cursor['key'],
                admission_policy_version=group['global_steps'],
                backend=group['case'].get('backend'), **cursor.get('residency', {})))
        return dict(retained_groups=len(self.groups), warmup_group_bound=self.batch_groups*self.warmup,
                    raw_queue_keys=len(present.get(self.partition, {})), prepared_queue_keys=len(prepared),
                    unfinished_cursors=cursors, waiting_sessions=len(self.waiting))

    def close(self):
        """Drain current decisions, discard idle cursors and clear owned keys only."""
        self.pause()
        if self.preparation is not None:
            self.preparation.finish(self)
        for worker in tuple(self.cursors):
            self.resolve(self.workers[worker].discard_partial_rollout.remote())
            del self.cursors[worker]
        keys = [key for uid in self.groups for key in
                [uid, *(f'{uid}_{session}_0' for session in range(self.group_size))]]
        if keys:
            self.tq.kv_clear(keys=keys, partition_id=self.partition)
        if self.preparation is not None:
            for uid in tuple(self.groups):
                self.preparation.discard(uid)
        self.groups.clear()
        self.leased.clear()
        self.waiting.clear()
        self.closed = True
