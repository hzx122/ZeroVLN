"""Freeze complete case paths once; load only one anchor's metadata per access."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
import random
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from action_head import protocol_v6 as P
from sft.action_head_joint_dataset import select_case_branch_paths
from sft.joint_fm_head_sft import JointBundleDataset
from sft.merge_fm_nested_shards import sha256


def index_root(manifest, seed):
    return Path(manifest).parent / f'training_branches_seed{seed}'


def manifest_identity(manifest):
    stat = manifest.stat()
    return dict(path=str(manifest.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def prepare_index(manifest, seed):
    manifest = Path(manifest)
    provenance_path = manifest.parent / 'provenance.json'
    provenance = json.loads(provenance_path.read_text())
    P.assert_protocol_identity(provenance)
    if not provenance.get('training_ready'):
        raise ValueError('source merge is not training-ready')
    root = index_root(manifest, seed)
    if (root / 'summary.json').exists():
        return IndexedCaseBranchDataset(manifest, seed).branch_selection
    root.mkdir(parents=True, exist_ok=True)
    initial = manifest_identity(manifest)
    digest, counters = hashlib.sha256(), Counter()
    identity_keys = ('source_scene', 'source_task_id', 'branch_id', 'parent_branch_id',
                     'golden_anchor_id', 'slow_anchor_id', 'option_sequence_index')

    def rows():
        with manifest.open('rb') as source:
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                digest.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                P.assert_protocol_identity(row)
                if row.get('split') != 'train' or row.get('decision_filter') != 'none':
                    raise ValueError('v6 branch index requires train-only no-filter source')
                compact = {key: row[key] for key in identity_keys}
                compact['offset'] = offset
                counters['source_windows'] += 1
                yield compact

    previous_case = None
    binary_temp = root / 'offsets.i64.tmp'
    selections_temp = root / 'selections.jsonl.tmp'
    with binary_temp.open('wb') as target, selections_temp.open('w') as selections:
        for case, items in itertools.groupby(rows(), lambda r: (r['source_scene'], r['source_task_id'])):
            if previous_case is not None and case <= previous_case:
                raise ValueError('source manifest must be ordered by case')
            previous_case = case
            groups = {}
            for row in items:
                groups.setdefault(row['slow_anchor_id'], []).append(row)
            selected, report = select_case_branch_paths(list(groups.values()), seed=seed)
            for group in selected:
                if not 1 <= len(group) <= 4:
                    raise ValueError('expected 1–4 fast windows per slow anchor')
                record = [len(group)] + [row['offset'] for row in group]
                target.write(np.asarray(record + [-1] * (5 - len(record)), dtype='<i8').tobytes())
                counters['fast_window_exposures'] += len(group)
            for key in ('cases', 'selected_paths', 'anchor_exposures', 'unique_anchor_contexts'):
                counters[key] += report[key]
            counters['single_branch_cases'] += report['selected_paths'] == 1
            selections.write(json.dumps(report['selections'][0], ensure_ascii=False) + '\n')
            if counters['cases'] % 4096 == 0:
                print(json.dumps(dict(event='branch_index', **counters)), flush=True)
    if not counters['anchor_exposures']:
        raise ValueError('empty branch selection')
    if digest.hexdigest() != provenance['manifest_sha256'] or manifest_identity(manifest) != initial:
        raise ValueError('manifest changed or does not match validated merge')
    if counters['source_windows'] != provenance['samples']:
        raise ValueError('source count disagrees with validated merge')
    report = dict(schema='case_branch_offset_index_v1',
                  policy='fixed_two_complete_case_branches_v1', seed=seed,
                  requested_branches_per_case=2, shared_prefix_policy='one_exposure_per_selected_path',
                  **counters, source=initial, source_sha256=digest.hexdigest(),
                  provenance_sha256=sha256(provenance_path), index_sha256=sha256(binary_temp),
                  selections_sha256=sha256(selections_temp),
                  selections_file=str(root / 'selections.jsonl'))
    os.replace(binary_temp, root / 'offsets.i64')
    os.replace(selections_temp, root / 'selections.jsonl')
    temporary = root / 'summary.json.tmp'
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, root / 'summary.json')
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


class IndexedCaseBranchDataset(Dataset):
    def __init__(self, manifest, seed):
        self.manifest = Path(manifest)
        root = index_root(manifest, seed)
        report = json.loads((root / 'summary.json').read_text())
        if report.get('schema') != 'case_branch_offset_index_v1' or report['seed'] != seed:
            raise ValueError('branch index schema/seed mismatch')
        if report['source'] != manifest_identity(self.manifest):
            raise ValueError('branch index source has changed')
        provenance_path = self.manifest.parent / 'provenance.json'
        if report['provenance_sha256'] != sha256(provenance_path):
            raise ValueError('branch index provenance has changed')
        if report['index_sha256'] != sha256(root / 'offsets.i64'):
            raise ValueError('branch offsets have changed')
        if report['selections_sha256'] != sha256(root / 'selections.jsonl'):
            raise ValueError('branch selections have changed')
        count = report['anchor_exposures']
        if (root / 'offsets.i64').stat().st_size != count * 5 * 8:
            raise ValueError('branch index size mismatch')
        self.offsets = np.memmap(root / 'offsets.i64', mode='r', dtype='<i8', shape=(count, 5))
        self.branch_selection = report

    def __len__(self):
        return len(self.offsets)

    def read_group(self, index):
        record = self.offsets[index]
        count = int(record[0])
        if not 1 <= count <= 4:
            raise ValueError('invalid anchor window count')
        group = []
        with self.manifest.open('rb') as source:
            for offset in record[1:count + 1]:
                source.seek(int(offset))
                group.append(json.loads(source.readline()))
        if len({row['slow_anchor_id'] for row in group}) != 1:
            raise ValueError('indexed anchor contains different contexts')
        return group

    def __getitem__(self, index):
        dataset = JointBundleDataset(self.read_group(index))
        return [dataset[item] for item in range(len(dataset))]


def prepare_published_source(inputs_path, output, source_name):
    """Index all accepted four-window groups; retain immutable source bytes."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/train'))
    from dlc_control_io_retry import retry_estale
    inputs_path, output = Path(inputs_path), Path(output)
    publication = json.loads(retry_estale(inputs_path.read_text, str(inputs_path)))
    entry = publication['sources'][source_name]
    completion = Path(entry['completion']['path'])
    if sha256(completion) != entry['completion']['sha256']:
        raise ValueError('published completion changed')
    receipt = json.loads(retry_estale(completion.read_text, str(completion)))
    if not all(receipt.get(k) is True for k in ('complete', 'formal_training_ready', 'loader_verified_all_shards')):
        raise ValueError('published source lacks formal loader acceptance')
    output.mkdir(parents=True, exist_ok=False)
    manifest = Path(entry['manifest']['path'])
    initial = manifest_identity(manifest)
    binary = output / 'offsets.i64.pending'
    digest = hashlib.sha256()
    seen, group, offsets, count, windows = set(), [], [], 0, 0
    protocol_version = None
    episodes = []

    def finish(target):
        nonlocal count
        if (len(group) != 4 or [r['fast_window_index'] for r in group] != list(range(4))
                or any(r['slow_anchor_id'] != group[0]['slow_anchor_id'] for r in group)):
            raise ValueError('expected exactly four consecutive windows per anchor')
        key = group[0]['slow_anchor_id']
        if key in seen:
            raise ValueError('duplicate accepted anchor identity')
        seen.add(key)
        identity = [group[0].get('source_dataset'), str(group[0].get('source_scene')),
                    str(group[0].get('source_task_id'))]
        if not episodes or episodes[-1]['identity'] != identity:
            episodes.append(dict(identity=identity, start=count, end=count))
        episodes[-1]['end'] = count + 1
        target.write(np.asarray([4, *offsets], dtype='<i8').tobytes())
        count += 1
        if count % 4096 == 0:
            print(json.dumps(dict(event='published_source_index', source=source_name,
                                  anchors=count, windows=windows)), flush=True)

    with manifest.open('rb') as stream, binary.open('xb') as target:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            digest.update(line)
            row = json.loads(line)
            version = row.get('action_head_as_tool_version')
            if protocol_version is not None and version != protocol_version:
                raise ValueError('published index mixes prompt protocols')
            protocol_version = version
            if version == 'v8':
                from sft.v8_rule_training_data import validate_training_row, SCHEMA
                validate_training_row(row)
                contract = SCHEMA
            else:
                P.assert_protocol_identity(row)
                contract = 'v6_rule_teacher_full_target_route_v1'
            if (row['split'] != 'train' or row['source_dataset'] != source_name
                    or row['data_contract'] != contract
                    or row.get('visual_input_mode') != 'live_frozen_visual_v1'):
                raise ValueError('published full-route source contract mismatch')
            if group and row['slow_anchor_id'] != group[0]['slow_anchor_id']:
                finish(target)
                group, offsets = [], []
            group.append(row)
            offsets.append(offset)
            windows += 1
        if group:
            finish(target)
    if (digest.hexdigest() != entry['manifest']['sha256']
            or initial != manifest_identity(manifest)
            or count != entry['counts']['accepted_anchors']
            or windows != entry['counts']['fast_windows']):
        raise ValueError('published manifest hash/count/identity mismatch')
    os.replace(binary, output / 'offsets.i64')
    episode_path = output / 'episode_ranges.json'
    episode_path.write_text(json.dumps(episodes)+'\n')
    report = dict(schema='published_full_route_offset_index_v1', complete=True,
        source_name=source_name, source=initial, source_sha256=digest.hexdigest(),
        completion=entry['completion'], anchors=count, windows=windows,
        protocol_version=protocol_version,
        episode_ranges=dict(path=str(episode_path), sha256=sha256(episode_path)),
        index_sha256=sha256(output / 'offsets.i64'),
        inherited_loader_acceptance=True, source_data_rewritten=False)
    tmp = output / 'summary.json.pending'
    tmp.write_text(json.dumps(report, indent=2)+'\n')
    os.replace(tmp, output / 'summary.json')
    return report


class IndexedPublishedDataset(Dataset):
    """Reuse v6 offset reads without materializing multi-GB history per rank."""
    branch_selection = None

    def __init__(self, root):
        self.sources, self.source_counts, self.ends = [], [], []
        self.source_episodes = []
        self.protocol_version = None
        total = 0
        for name in ('r2r', 'rxr', 'sage'):
            folder = Path(root) / name
            report = json.loads((folder / 'summary.json').read_text())
            if report['schema'] != 'published_full_route_offset_index_v1' or not report['complete']:
                raise ValueError('published index incomplete')
            version = report.get('protocol_version', 'v6')
            if self.protocol_version is not None and self.protocol_version != version:
                raise ValueError('published sources mix prompt protocols')
            self.protocol_version = version
            manifest = Path(report['source']['path'])
            if manifest_identity(manifest) != report['source']:
                raise ValueError('published manifest identity changed')
            if sha256(Path(report['completion']['path'])) != report['completion']['sha256']:
                raise ValueError('published completion changed')
            binary = folder / 'offsets.i64'
            if (sha256(binary) != report['index_sha256']
                    or binary.stat().st_size != report['anchors'] * 40):
                raise ValueError('published offsets changed')
            # Reuse the tested historical read_group/getitem byte-offset implementation.
            dataset = IndexedCaseBranchDataset.__new__(IndexedCaseBranchDataset)
            dataset.manifest = manifest
            dataset.offsets = np.memmap(binary, mode='r', dtype='<i8', shape=(report['anchors'], 5))
            self.sources.append(dataset)
            self.source_counts.append(report['anchors'])
            episode = report.get('episode_ranges')
            if episode and sha256(Path(episode['path'])) != episode['sha256']:
                raise ValueError('published episode ranges changed')
            self.source_episodes.append(json.loads(Path(episode['path']).read_text()) if episode else None)
            total += report['anchors']
            self.ends.append(total)

    def __len__(self):
        return self.ends[-1]

    def read_group(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = 0
        for dataset, end in zip(self.sources, self.ends):
            if index < end:
                return dataset.read_group(index - start)
            start = end

    def __getitem__(self, index):
        dataset = JointBundleDataset(self.read_group(index))
        return [dataset[item] for item in range(len(dataset))]


class BalancedSourceShardSampler:
    """Exact 1:1:2 per global batch; traverse each source before repeating."""
    def __init__(self, counts, rank, world_size, *, seed, batch_size=1, resume_steps=0):
        import math
        if len(counts) != 3 or min(counts) <= 0 or not 0 <= rank < world_size:
            raise ValueError('three nonempty sources and a valid rank required')
        self.counts = list(counts)
        self.rank, self.world_size, self.seed = rank, world_size, int(seed)
        self.batch_size = batch_size
        self.global_batch = world_size * batch_size
        if self.global_batch % 4:
            raise ValueError('balanced global batch must be divisible by four')
        self.steps = math.ceil(sum(counts) / self.global_batch)
        self.size = self.steps * self.global_batch
        self.samples_per_rank = self.steps * batch_size
        self.resume_epoch, self.resume_step = divmod(resume_steps, self.steps)
        if resume_steps < 0:
            raise ValueError('negative resume steps')
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.samples_per_rank

    def __iter__(self):
        import random
        epoch = self.resume_epoch + self.epoch
        quotas = [self.size // 4, self.size // 4, self.size // 2]
        pools, base = [], 0
        for source, (count, quota) in enumerate(zip(self.counts, quotas)):
            absolute = epoch * quota
            remaining, chosen = quota, []
            while remaining:
                cycle, offset = divmod(absolute, count)
                permutation = list(range(base, base + count))
                random.Random(f'{self.seed}:{source}:{cycle}').shuffle(permutation)
                take = min(remaining, count - offset)
                chosen.extend(permutation[offset:offset+take])
                absolute += take
                remaining -= take
            pools.append(chosen)
            base += count
        per_batch = [self.global_batch // 4, self.global_batch // 4, self.global_batch // 2]
        skip = self.resume_step if self.epoch == 0 else 0
        shuffle = random.Random(f'{self.seed}:global:{epoch}')
        for step in range(self.steps):
            batch = []
            for pool, quota in zip(pools, per_batch):
                batch.extend(pool[step*quota:(step+1)*quota])
            shuffle.shuffle(batch)
            if step >= skip:
                yield from batch[self.rank::self.world_size]


class EpisodeSourceShardSampler:
    """Keep ordered episode anchors on one rank, with exact named-source quotas.

    Ranks are assigned to R2R/RxR/SAGE in proportion to explicit weights.
    Whole episodes are shuffled and greedily balanced over their source ranks.
    Each rank cycles its own complete episode list without changing GB64.
    """
    def __init__(self, counts, episodes, rank, world_size, *, seed, weights,
                 batch_size=1, resume_steps=0):
        import math
        if batch_size != 1 or len(weights) != 3 or min(weights) <= 0 or world_size % sum(weights):
            raise ValueError('episode sampler requires B1 and integral source rank quotas')
        if not 0 <= rank < world_size or len(counts) != 3 or min(counts) <= 0:
            raise ValueError('invalid source counts or rank')
        self.steps = math.ceil(sum(counts) / world_size)
        self.resume_steps = resume_steps
        self.epoch = 0
        self.rank, self.world_size = rank, world_size
        self.order = None
        rank_base = anchor_base = 0
        for source, (count, groups, weight) in enumerate(zip(counts, episodes, weights)):
            slots = world_size*weight//sum(weights)
            if groups is None or not groups or groups[0]['start'] != 0 or groups[-1]['end'] != count:
                raise ValueError('episode sampler requires complete immutable episode ranges')
            if any(g['start'] >= g['end'] or (i and groups[i-1]['end'] != g['start'])
                   for i, g in enumerate(groups)):
                raise ValueError('episode ranges must exactly partition the source')
            shuffled = list(groups)
            random.Random(f'{seed}:episodes:{source}').shuffle(shuffled)
            lanes, sizes = [[] for _ in range(slots)], [0]*slots
            for group in shuffled:
                lane = min(range(slots), key=lambda i: sizes[i])
                lanes[lane].extend(range(anchor_base+group['start'], anchor_base+group['end']))
                sizes[lane] += group['end']-group['start']
            position = rank % sum(weights)
            if rank_base <= position < rank_base+weight:
                self.order = lanes[(rank//sum(weights))*weight + position-rank_base]
            rank_base += weight
            anchor_base += count
        if not self.order:
            raise ValueError('not enough episodes for the requested source ranks')

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.steps

    def __iter__(self):
        offset = self.resume_steps + self.epoch*self.steps
        skip = self.resume_steps % self.steps if self.epoch == 0 else 0
        start = offset if self.epoch == 0 else (self.resume_steps//self.steps+self.epoch)*self.steps
        for index in range(self.steps-skip):
            yield self.order[(start+index) % len(self.order)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--published-inputs')
    parser.add_argument('--output')
    parser.add_argument('--source', choices=('r2r', 'rxr', 'sage'))
    args = parser.parse_args()
    if args.published_inputs:
        if not args.output or not args.source:
            parser.error('published index requires --output and --source')
        prepare_published_source(args.published_inputs, args.output, args.source)
    elif args.manifest:
        prepare_index(args.manifest, args.seed)
    else:
        parser.error('--manifest or --published-inputs required')
