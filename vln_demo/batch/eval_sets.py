"""Versioned nested English validation manifests; never consult evaluation outcomes."""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random

LEVELS = ('100', '200', '500', '1000', 'full')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_path(value):
    """Resolve portable dataset paths against the configured navigation tree."""
    path = Path(value)
    if path.is_absolute():
        return path
    root = os.environ.get('RYNNBRAIN_NAV_DIR') or os.environ.get('ACTION_HEAD_NAVIGATION_REPO')
    if not root:
        raise ValueError('Set RYNNBRAIN_NAV_DIR to resolve evaluation dataset path: ' + value)
    if '..' in path.parts:
        raise ValueError('evaluation dataset path must stay inside the navigation tree')
    return Path(root) / path


def nested_order(rows, anchor, seed):
    """Keep the original 100; balance scenes and prefer unseen routes thereafter."""
    lookup = {r['logical_id']: r for r in rows}
    if len(lookup) != len(rows) or len({r['logical_id'] for r in anchor}) != len(anchor):
        raise ValueError('duplicate episode identity')
    if any(lookup.get(r['logical_id']) != r for r in anchor):
        raise ValueError('anchor differs from source episodes')
    rng = random.Random(seed)
    selected = list(anchor)
    seen = {r['logical_id'] for r in selected}
    routes = Counter((r['scene'], r['trajectory_id']) for r in selected)
    counts = Counter(r['scene'] for r in selected)
    pools = defaultdict(list)
    for row in sorted(rows, key=lambda r: r['logical_id']):
        if row['logical_id'] not in seen:
            pools[row['scene']].append(row)
    for pool in pools.values():
        rng.shuffle(pool)
    scene_order = sorted(pools)
    rng.shuffle(scene_order)
    while pools:
        # First represent every route once, then second annotations, etc.
        wave = min(routes[(r['scene'], r['trajectory_id'])] for pool in pools.values() for r in pool)
        eligible = [s for s in scene_order if s in pools and any(
            routes[(s, r['trajectory_id'])] == wave for r in pools[s])]
        scene = min(eligible, key=lambda s: counts[s])
        index = next(i for i, r in enumerate(pools[scene]) if routes[(scene, r['trajectory_id'])] == wave)
        row = pools[scene].pop(index)
        if not pools[scene]:
            del pools[scene]
        selected.append(row)
        counts[scene] += 1
        routes[(scene, row['trajectory_id'])] += 1
    return selected


def write_ladder(directory, ordered, provenance):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    all_rows = [row for name in ('r2r', 'rxr') for row in ordered[name]]
    (directory / 'episodes.jsonl').write_text(''.join(
        json.dumps(r, sort_keys=True, ensure_ascii=False) + '\n' for r in all_rows))
    stages = {}
    for level in LEVELS:
        lists = [ordered[name][:int(level)] if level != 'full' else ordered[name] for name in ('r2r', 'rxr')]
        ids = [rows[i]['logical_id'] for i in range(max(map(len, lists))) for rows in lists if i < len(rows)]
        filename = level + '.json'
        (directory / filename).write_text(json.dumps(ids, indent=2) + '\n')
        stages[level] = dict(file=filename, counts={name: len(rows) for name, rows in zip(('r2r', 'rxr'), lists)})
    lock = dict(version='english-val-unseen-nested-v1', stages=stages, **provenance)
    lock['sha256'] = {p.name: digest(p) for p in sorted(directory.iterdir())}
    (directory / 'lock.json').write_text(json.dumps(lock, ensure_ascii=False, indent=2) + '\n')
    load_ladder(directory)
    for path in directory.iterdir():
        path.chmod(0o444)
    return lock


def load_ladder(directory, verify_sources=False):
    directory = Path(directory)
    lock = json.loads((directory / 'lock.json').read_text())
    for name, expected in lock['sha256'].items():
        if Path(name).name != name or digest(directory / name) != expected:
            raise ValueError('evaluation manifest hash mismatch: ' + name)
    if verify_sources:
        for source in lock['sources'].values():
            if digest(source_path(source['path'])) != source['sha256']:
                raise ValueError('evaluation source changed: ' + source['path'])
    rows = [json.loads(line) for line in (directory / 'episodes.jsonl').read_text().splitlines()]
    if verify_sources:
        rows = [{**row, 'source_file': str(source_path(row['source_file']))}
                if 'source_file' in row else row for row in rows]
    lookup = {r['logical_id']: r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError('duplicate identity in episode pool')
    stages, previous = {}, set()
    for level in LEVELS:
        ids = json.loads((directory / lock['stages'][level]['file']).read_text())
        if len(set(ids)) != len(ids) or not previous <= set(ids) <= lookup.keys():
            raise ValueError('stages must be unique nested subsets')
        selected = [lookup[i] for i in ids]
        if dict(Counter(r['dataset'] for r in selected)) != lock['stages'][level]['counts']:
            raise ValueError('stage counts differ from lock')
        if any(r['language'] not in ('en', 'en-IN', 'en-US') or r['split'] != 'val_unseen'
               or r['role'] != 'guide' for r in selected):
            raise ValueError('stage contains non-English/non-guide/non-validation episode')
        stages[level] = selected
        previous = set(ids)
    if previous != lookup.keys():
        raise ValueError('full stage omits episodes')
    return stages
