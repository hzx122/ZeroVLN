"""CPU group credit and metadata-only scheduling over acknowledged TQ storage."""
from collections import deque
import json
import time

from action_head_rl.queue_records import record_locator, publish_record_groups, REFERENCE
from action_head_rl.text_policy import is_text_record


def host_memory():
    from pathlib import Path
    values = {parts[0].rstrip(':'): int(parts[1]) * 1024
              for line in Path('/proc/meminfo').read_text().splitlines()
              if (parts := line.split()) and parts[0] in ('MemTotal:', 'MemAvailable:')}
    return dict(total=values['MemTotal'], available=values['MemAvailable'], time=time.time())


class GroupPreparationPipeline:
    """Bounded CPU conversion per node, independent of native group selection.

    The driver owns only JSON descriptors and queue leases. Completed surplus
    groups remain paired with their native replay UID, including stale drops.
    There is no background thread mutating replay-buffer state.
    """

    def __init__(self, *, options, concurrency=8, slots=None, submit=None,
                 resolve=None, log=None, memory_guard=True, available_fraction=.15,
                 tasks_per_node=1):
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError('CPU preparation requires a positive concurrency bound')
        if type(tasks_per_node) is not int or tasks_per_node not in (1, 2):
            raise ValueError('CPU preparation supports one or two tasks per node')
        if slots is None:
            slots = sorted(node['NodeID'] for node in ray.nodes()
                           if node['Alive'] and node['Resources'].get('CPU', 0) >= 1)
        slots = list(slots)[:concurrency]
        if not slots or len(set(slots)) != len(slots):
            raise ValueError('CPU pipeline needs distinct live node slots')
        if submit is None:
            remote = ray.remote(num_cpus=1, num_gpus=0, max_retries=0)(prepare_group)
            def submit(node, keys, partition, step):
                placement = NodeAffinitySchedulingStrategy(node, soft=False)
                return remote.options(scheduling_strategy=placement).remote(
                    keys, partition, step=step, group_index=0, streaming=True, **options)
        self.submit, self.resolve = submit, resolve or ray.get
        # Fill every physical node before using its second slot. Keep memory
        # probes distinct even when the execution slots repeat a node.
        self.free = deque((slots * tasks_per_node)[:concurrency])
        self.pending, self.ready, self.claimed = {}, {}, set()
        self.pending_episode_bytes, self.tasks_per_node = {}, tasks_per_node
        self.log = log or (lambda row: None)
        if not 0 < available_fraction < 1:
            raise ValueError('CPU memory reserve must be a fraction between zero and one')
        self.slots, self.probes, self.memory = slots, {}, {}
        self.memory_guard, self.available_fraction = memory_guard, available_fraction
        self.next_probe, self.pressure_uid, self.pressure_state = 0., None, None
        memory_remote = ray.remote(num_cpus=0, num_gpus=0, max_retries=0)(host_memory)
        self.probe = lambda node: memory_remote.options(scheduling_strategy=
            NodeAffinitySchedulingStrategy(node, soft=False)).remote()

    def _event(self, event, uid, **fields):
        self.log(dict(event=event, uid=uid, time=time.time(), **fields))

    def begin_sampling(self):
        if self.pending or self.probes:
            raise RuntimeError('CPU preparation must quiesce before the next policy')
        # GPU training/offload can change host residency substantially. Never
        # use the previous step's sampling snapshot to admit a new wave.
        self.memory.clear()
        self.next_probe, self.pressure_uid, self.pressure_state = 0., None, None

    def _probe_memory(self):
        if self.memory_guard and not self.probes and time.monotonic() >= self.next_probe:
            self.probes = {self.probe(node): node for node in self.slots}
            self.next_probe = time.monotonic() + 2.

    def refresh_before_dispatch(self, producer):
        self._probe_memory()
        while self.probes:
            handle = producer.wait_one(tuple(self.probes))
            if handle is not None:
                self.complete(handle, producer)

    def pump(self, producer):
        self._probe_memory()
        # Oldest admission first only schedules CPU work; it never selects the
        # training batch. Ineligible terminal groups are left to native drop.
        candidates = sorted(producer.groups, key=lambda uid: producer.groups[uid]['global_steps'])
        for uid in candidates:
            if not self.free:
                break
            # Once native selection pauses rollout, only the selected batch
            # needs new CPU work. Keep surplus raw leases for the next sample;
            # already submitted tasks still drain normally before GPU training.
            if not producer.running and producer.leased and uid not in producer.leased:
                continue
            state = producer.groups[uid]
            if (uid in self.claimed or state['settled'] != producer.group_size or state['failed']
                    or producer.policy_version - state['global_steps'] + 1 >
                       producer.buffer.max_off_policy_threshold):
                continue
            node = next((node for node in self.free if self._slot_available(node, state)), None)
            if node is None:
                continue
            self.free.remove(node)
            keys = [f'{uid}_{i}_0' for i in range(producer.group_size)]
            handle = self.submit(node, keys, producer.partition, producer.policy_version + 1)
            if handle in self.pending:
                raise RuntimeError('CPU preparation returned a duplicate handle')
            self.pending[handle] = (uid, node)
            self.pending_episode_bytes[handle] = state.get('max_raw_episode_bytes', 0)
            self.claimed.add(uid)
            self._event('prepare_started', uid, node=node, raw_bytes=state.get('raw_bytes', 0),
                        node_active_tasks=sum(owner == node for _, owner in self.pending.values()),
                        tasks_per_node=self.tasks_per_node)

    def _slot_available(self, node, state):
        running = [handle for handle, (_, owner) in self.pending.items() if owner == node]
        # The original first slot must remain available to free completed raw
        # groups under pressure. Only extra concurrency needs fresh headroom.
        if not running or not self.memory_guard:
            return True
        sample = self.memory.get(node)
        incoming = state.get('max_raw_episode_bytes', 0)
        active = [self.pending_episode_bytes[handle] for handle in running]
        if (sample is None or time.time() - sample['time'] > 4.
                or incoming <= 0 or any(size <= 0 for size in active)):
            return False
        # Each task streams one sibling at a time. Budget four times its
        # largest serialized episode for decode, records and publication, on
        # top of the existing host reserve; do not spend the reserve itself.
        working = 4 * (incoming + sum(active))
        return sample['available'] >= sample['total'] * self.available_fraction + working

    def complete(self, handle, producer):
        if handle in self.probes:
            node = self.probes[handle]
            sample = self.resolve(handle)
            if not 0 <= sample['available'] <= sample['total'] or sample['total'] <= 0:
                raise ValueError('CPU memory probe returned invalid memory bounds')
            self.memory[node] = sample
            del self.probes[handle]
            return
        uid, node = self.pending[handle]
        lease, summaries = self.resolve(handle)
        # A successful result proves all puts and validation completed. Never
        # release the source on an RPC failure or a partially published lease.
        if (not lease.get('keys') or len(summaries) != producer.group_size
                or any('_prepared_records' not in row for row in summaries)):
            raise ValueError('CPU preparation did not acknowledge a complete group')
        json.dumps((lease, summaries), allow_nan=False)
        self.ready[uid] = (lease, summaries)
        self._event('prepared_ack', uid, node=node, prepared_bytes=lease.get('serialized_bytes', 0),
            **({'episode_cpu_releases': lease['episode_cpu_releases']}
               if 'episode_cpu_releases' in lease else {}))
        producer.acknowledge_prepared_group(uid)
        self._event('raw_replaced', uid, node=node)
        del self.pending[handle]
        del self.pending_episode_bytes[handle]
        self.free.append(node)

    def finish(self, producer):
        """Quiesce CPU work too before GPU training/policy changes."""
        self.pump(producer)
        while self.pending:
            handle = producer.wait_one(self.handles())
            if handle is not None:
                self.complete(handle, producer)
            self.pump(producer)
        while self.probes:
            handle = producer.wait_one(tuple(self.probes))
            if handle is not None:
                self.complete(handle, producer)

    def handles(self):
        return (*self.pending, *self.probes)

    def dispatch_uid(self, producer):
        """Under host pressure, reserve forward progress for one whole K group.

        Holding every incomplete actor would deadlock before a group can be
        credited/released. Finish one existing group instead of admitting or
        advancing all 512 sessions. No trajectory/call is dropped or truncated.
        """
        if not self.memory_guard:
            return None
        if len(self.memory) != len(self.slots):
            return ''  # Wait for the initial asynchronous probes.
        pressure = any(row['available'] < row['total'] * self.available_fraction
                       for row in self.memory.values())
        if pressure != self.pressure_state:
            self._event('memory_backpressure', '', active=pressure,
                reserve_fraction=self.available_fraction, nodes=self.memory,
                prepared_bytes=sum(lease.get('serialized_bytes', 0) for lease, _ in self.ready.values()),
                retained_raw_bytes=sum(state.get('raw_bytes', 0) for state in producer.groups.values()
                                       if not state.get('raw_replaced')))
            self.pressure_state = pressure
        if not pressure:
            self.pressure_uid = None
            return None
        if (self.pressure_uid in producer.groups
                and producer.groups[self.pressure_uid]['settled'] < producer.group_size):
            return self.pressure_uid
        active = {row['key'].split('_')[0] for row in producer.cursors.values()}
        candidates = active or {row[0] for row in producer.waiting}
        self.pressure_uid = min(candidates, default='', key=lambda uid:
            (-producer.groups[uid]['settled'], producer.groups[uid]['global_steps']))
        return self.pressure_uid

    def take(self, uid, *, step, group_index, group_size):
        from action_head_rl.queue_records import bind_prepared_locator
        lease, summaries = self.ready[uid]
        # Preserve the exact final selected order and update version. No
        # tensors are repacked, copied or returned to the driver here.
        result = []
        for sample, row in enumerate(summaries):
            bound = dict(row)
            bound['_prepared_records'] = [bind_prepared_locator(record, step=step,
                rollout_index=group_index * group_size + sample) for record in row['_prepared_records']]
            bound['chunks'] = [dict(chunk, update_policy_version=step - 1) for chunk in row['chunks']]
            result.append(bound)
        del self.ready[uid]
        self._event('prepared_selected', uid, step=step, group_index=group_index)
        return lease, result

    def discard(self, uid):
        from action_head_rl.queue_records import release_record_groups
        if any(owner == uid for owner, _ in self.pending.values()):
            raise RuntimeError('cannot discard a group during CPU conversion')
        if uid in self.ready:
            lease, _ = self.ready[uid]
            release_record_groups(lease)
            del self.ready[uid]
            self._event('prepared_discarded', uid)
        self.claimed.discard(uid)

    def owned_keys(self):
        return {key for lease, _ in self.ready.values() for key in lease['keys']}


def describe_records(records):
    from action_head_rl.megatron_trajectory_text import shard_text_trajectories
    from action_head_rl.worker import record_input_length
    rows = []
    for record in records:
        if is_text_record(record):
            # Run the original full native/R3/version validation on the CPU
            # producer. Learners validate the resolved payload again.
            shard_text_trajectories([record], dp_size=1,
                policy_version=record.get('update_policy_version', record['policy_version']))
        row = record_locator(record)
        row['_queue_input_numel'] = record_input_length(record)
        for key in ('text_loss_coef', 'text_stop_weight', 'text_path_advantage', 'action_path_advantage'):
            if key in record:
                row[key] = float(record[key])
        rows.append(row)
    json.dumps(rows, allow_nan=False)
    return rows


def shard_text_descriptors(rows, *, dp_size, policy_version):
    from action_head_rl.megatron_trajectory_text import schedule_text_rows
    if (not rows or any(REFERENCE not in row or not is_text_record(row)
            or row.get('update_policy_version', row.get('policy_version')) != policy_version for row in rows)
            or len({row['text_loss_coef'] for row in rows}) != 1):
        raise ValueError('prepared text metadata changed its version or coefficient')
    return schedule_text_rows(rows, dp_size=dp_size, lengths=[row['_queue_input_numel'] for row in rows])


def prepare_group(keys, partition, *, step, group_index, max_tool_calls,
                  reward_version, text_loss_coef, streaming=False):
    from action_head_rl.cpu_memory import release_cpu_memory
    try:
        # Keep the tensor-owning frame separate: trimming inside that frame's
        # finally would run before its local rollout/episode references die.
        prepare = _prepare_group_streamed if streaming else _prepare_group
        lease, summaries = prepare(keys, partition, step=step, group_index=group_index,
            max_tool_calls=max_tool_calls, reward_version=reward_version,
            text_loss_coef=text_loss_coef)
    finally:
        memory = release_cpu_memory()
    lease['cpu_memory'] = memory
    return lease, summaries


def _prepare_group_streamed(keys, partition, *, step, group_index, max_tool_calls,
                            reward_version, text_loss_coef):
    import transfer_queue as tq
    from action_head_rl.partial_groups import read_rollout_group
    from action_head_rl.rule_credit import assign_group
    from action_head_rl.trainer import require_valid_rollouts
    from action_head_rl.cpu_memory import release_cpu_memory
    if reward_version not in ('v3_first_arrival', 'v3_terminal_local'):
        raise ValueError('streamed group preparation requires v3 whole-group credit')
    tq.init()
    metadata = read_rollout_group(keys, partition, field='credit_metadata')
    require_valid_rollouts(metadata, step=step)
    scores = assign_group(metadata, step=step, max_tool_calls=max_tool_calls, version=reward_version)
    lease = dict(partition='v8_records', keys=[], records=0, serialized_bytes=0)
    summaries = []
    for sample, (key, expected, credit) in enumerate(zip(keys, metadata, scores, strict=True)):
        # A separate frame releases all tensors before reading the next sibling.
        published, summary = _prepare_episode(key, partition, expected=expected, credit=credit,
            step=step, rollout_index=group_index * len(keys) + sample, text_loss_coef=text_loss_coef)
        lease['keys'].extend(published['keys'])
        lease['records'] += published['records']
        lease['serialized_bytes'] += published['serialized_bytes']
        summaries.append(summary)
        # torch.save/load can keep storage in unreachable local Pickler
        # classes. The acknowledged episode's tensor-owning frame is gone;
        # reclaim it before the next sibling, not only after all K episodes.
        started = time.perf_counter()
        memory = release_cpu_memory()
        memory['release_seconds'] = time.perf_counter() - started
        lease.setdefault('episode_cpu_releases', []).append(dict(sample=sample, **memory))
    json.dumps(summaries, allow_nan=False)
    return lease, summaries


def _prepare_episode(key, partition, *, expected, credit, step, rollout_index, text_loss_coef):
    from action_head_rl.partial_groups import read_rollout_group
    from action_head_rl.rollout_records import credit_metadata, prepare_v3_episode
    rollout, = read_rollout_group([key], partition)
    if credit_metadata(rollout) != expected:
        raise ValueError('rollout payload differs from its acknowledged group-credit metadata')
    records = prepare_v3_episode(rollout, credit=credit, step=step,
        rollout_index=rollout_index, policy_version=step-1,
        text_loss_coef=text_loss_coef, allow_mixed_policy=True)
    lease = publish_record_groups(records)
    return lease, _summary(rollout, records, credit)


def _summary(rollout, records, credit):
    summary = {key: rollout[key] for key in ('seed', 'case', 'reward', 'reward_components',
        'result', 'partial_rollout', 'episode_seconds', 'slow_qwen_seconds',
        'rollout_queue_wait_seconds', 'actor_max_memory_gb') if key in rollout}
    summary['decisions'] = [None] * len(rollout['decisions'])
    summary['chunks'] = [{key: row[key] for key in ('option_id', 'chunk_index', 'env_reward',
        'collision', 'blocked', 'moved_m', 'advantage', 'policy_version', 'update_policy_version',
        'option_credit', 'gym_feedback') if key in row} for row in rollout['chunks']]
    summary['_prepared_records'] = describe_records(records)
    summary['_prepared_credit'] = credit
    return summary


def _prepare_group(keys, partition, *, step, group_index, max_tool_calls,
                   reward_version, text_loss_coef):
    import transfer_queue as tq
    from action_head_rl.partial_groups import read_rollout_group
    from action_head_rl.rollout_records import prepare_v3_group
    from action_head_rl.trainer import require_valid_rollouts
    tq.init()
    rollouts = read_rollout_group(keys, partition)
    require_valid_rollouts(rollouts, step=step)
    episodes, credits = prepare_v3_group(rollouts, step=step, group_index=group_index,
        policy_version=step-1, max_tool_calls=max_tool_calls, reward_version=reward_version,
        text_loss_coef=text_loss_coef, allow_mixed_policy=True)
    lease = publish_record_groups([row for episode in episodes for row in episode])
    summaries = []
    for rollout, records, credit in zip(rollouts, episodes, credits, strict=True):
        summaries.append(_summary(rollout, records, credit))
    # Never return a torch payload accidentally through Ray's driver object store.
    json.dumps(summaries, allow_nan=False)
    return lease, summaries


def iter_prepared_groups(batch, *, group_size, producer, concurrency=8, **options):
    """Bounded CPU workers preserve buffer/group order and exact credit rules."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from action_head_rl.partial_groups import leased_group_keys
    if producer.preparation is not None:
        for index, uid, keys in leased_group_keys(batch, group_size=group_size):
            lease, summaries = producer.preparation.take(uid, step=options['step'],
                group_index=index, group_size=group_size)
            producer.release_prepared_raw_group(batch, keys)
            yield index, lease, summaries
        return
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError('CPU preparation requires a positive concurrency bound')
    remote = ray.remote(num_cpus=1, num_gpus=0, max_retries=0)(prepare_group)
    nodes = sorted(node['NodeID'] for node in ray.nodes()
                   if node['Alive'] and node['Resources'].get('CPU', 0) >= 1)
    if not nodes:
        raise RuntimeError('CPU preparation has no live CPU node')
    # At most one heavy preparation per node. The old num_cpus=1 declaration
    # let Ray place several 50+ GiB preparations together on the head node.
    slots = nodes[:min(concurrency, len(nodes))]
    source = iter(leased_group_keys(batch, group_size=group_size))
    pending = deque()
    def submit():
        group = next(source, None)
        if group is None:
            return
        index, uid, keys = group
        placement = NodeAffinitySchedulingStrategy(slots[index % len(slots)], soft=False)
        pending.append((index, keys, remote.options(scheduling_strategy=placement).remote(keys, batch.partition_id,
                        group_index=index, **options)))
    try:
        for _ in slots:
            submit()
        while pending:
            index, keys, future = pending.popleft()
            lease, summaries = ray.get(future)
            # The prepared queue now owns all exact training tensors. Raw data is
            # released group by group instead of keeping two complete step copies.
            producer.release_prepared_raw_group(batch, keys)
            submit()
            yield index, lease, summaries
    finally:
        for _, _, future in pending:
            ray.cancel(future, force=True)
