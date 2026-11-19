#!/usr/bin/env python3
"""Freeze a balanced R2R/RxR English training population before rendering."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random

REPO = Path(__file__).resolve().parents[2]
SOURCE = Path('/opt/navigation-runtime/VLMEvalKit/3rdparty/RynnBrain/navigation/data/datasets')
SCENES = Path('/data/shared/groups/mfm/common/anno/vlmevalkit/RynnBrain/data/scene_datasets/mp3d')
ART = Path('/data/zerovln/artifacts')
DEFAULT_ROOT = ART / 'action-head-v6-r2r-rxr-balanced-20260928-r2'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_episodes(path):
    with gzip.open(path, 'rt') as f:
        return json.load(f)['episodes']


def scene(row):
    return Path(row['scene_id']).stem


def balanced_cases(r2r, rxr, seed):
    def ordered(rows):
        result = sorted(rows, key=lambda r: (scene(r), str(r['episode_id'])))
        keys = [(scene(r), str(r['episode_id'])) for r in result]
        if len(set(keys)) != len(keys):
            raise ValueError('duplicate episode identity')
        return result
    r2r = ordered(r2r)
    rxr = ordered([r for r in rxr if r['instruction']['language'] in ('en-US', 'en-IN')])
    if not r2r or len(rxr) < len(r2r):
        raise ValueError('insufficient English RxR episodes for sampling without replacement')
    selected = sorted(random.Random(seed).sample(rxr, len(r2r)),
                      key=lambda r: (scene(r), str(r['episode_id'])))
    return {'r2r': r2r, 'rxr': selected}, len(rxr)


def freeze(root, seed=20260928, shards=16, eligibility=None):
    root = Path(root).resolve()
    if not root.is_relative_to(ART) or shards <= 0:
        raise ValueError('root must be on the artifact disk; shards must be positive')
    sources = {d: SOURCE / d / 'train' / ('train.json.gz' if d == 'r2r' else 'train_guide.json.gz')
               for d in ('r2r', 'rxr')}
    populations = {d: read_episodes(sources[d]) for d in ('r2r','rxr')}
    eligibility_record = None
    if eligibility is not None:
        eligibility = Path(eligibility)
        audit = json.loads(eligibility.read_text())
        if audit['schema'] != 'v6_vlnce_native_reference_eligibility_v1':
            raise ValueError('invalid eligibility audit')
        for source in sources.values():
            if audit['source_sha256'][str(source)] != sha256(source):
                raise ValueError('eligibility source changed')
        valid = {(r['dataset'],r['scene'],r['episode_id']) for r in audit['eligible']}
        if any(('r2r',scene(r),str(r['episode_id'])) not in valid for r in populations['r2r']):
            raise ValueError('this recipe requires every R2R training episode to be eligible')
        populations['rxr'] = [r for r in populations['rxr'] if ('rxr',scene(r),str(r['episode_id'])) in valid]
        eligibility_record = dict(path=str(eligibility.resolve()), sha256=sha256(eligibility),
                                  eligible_counts=audit['eligible_counts'], rejected_counts=audit['rejected_counts'])
    cases, available = balanced_cases(populations['r2r'], populations['rxr'], seed)
    heldout = set()
    for d in ('r2r', 'rxr'):
        p = SOURCE / d / 'val_unseen' / ('val_unseen.json.gz' if d == 'r2r' else 'val_unseen_guide.json.gz')
        sources[d + '_val_unseen'] = p
        heldout.update(scene(r) for r in read_episodes(p))
    train_scenes = {scene(r) for rows in cases.values() for r in rows}
    if train_scenes & heldout:
        raise ValueError(f'train/val_unseen scene leakage: {train_scenes & heldout}')
    missing = [str(SCENES / s / (s + ext)) for s in sorted(train_scenes | heldout)
               for ext in ('.glb', '.navmesh') if not (SCENES / s / (s + ext)).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} scene assets absent: {missing[:8]}')
    # Balance each dataset across shards. A case stays intact within one shard.
    rows = []
    for d, episodes in cases.items():
        for i, e in enumerate(episodes):
            rows.append(dict(dataset=d, scene=scene(e), episode_id=str(e['episode_id']),
                task_id=f"{d}_{e['episode_id']}", shard=i % shards,
                instruction=e['instruction']['instruction_text'],
                language=e['instruction'].get('language', 'en'),
                start_position=e['start_position'], start_rotation=e['start_rotation'],
                reference_path=e['reference_path'], goals=e['goals'],
                trajectory_id=str(e.get('trajectory_id', ''))))
    content = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows)
    report = dict(schema='v6_vlnce_balanced_cases_v1', seed=seed, shards=shards,
        sampling_unit='instruction_episode', replacement=False,
        counts={d: len(es) for d, es in cases.items()}, rxr_english_available=available,
        train_scenes=sorted(train_scenes), validation_scenes=sorted(heldout),
        scene_overlap=[], source_sha256={str(p): sha256(p) for p in sources.values()},
        cases_sha256=hashlib.sha256(content.encode()).hexdigest(),
        languages=dict(Counter(r['language'] for r in rows if r['dataset'] == 'rxr')),
        trajectory_policy='navmesh geodesic through ordered reference waypoints, preserving start and goal',
        eligibility=eligibility_record,
        artifact_root=str(root), scene_root=str(SCENES))
    target = root / 'dataset'
    target.mkdir(parents=True, exist_ok=True)
    files = {'train_cases.jsonl': content, 'lock.json': json.dumps(report, indent=2) + '\n',
             'split.json': json.dumps({'splits': {'train_scenes': sorted(train_scenes),
                                                  'validation_scenes': sorted(heldout)}}, indent=2) + '\n'}
    for name, value in files.items():
        p = target / name
        if p.exists() and p.read_text() != value:
            raise FileExistsError(f'refusing different frozen input: {p}')
        if not p.exists():
            p.write_text(value)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--seed', type=int, default=20260928)
    p.add_argument('--shards', type=int, default=16)
    p.add_argument('--eligibility', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(freeze(args.root, args.seed, args.shards, args.eligibility), indent=2))


if __name__ == '__main__':
    main()
