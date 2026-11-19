"""Read large v6 beam pools through the existing validated offset index."""
from __future__ import annotations

import json
from pathlib import Path
import random

INDEX_THRESHOLD_BYTES = 64 * 1024 * 1024


def existing_index(path, seed=1):
    path = Path(path)
    if not (path.parent / f'training_branches_seed{seed}/summary.json').is_file():
        if path.stat().st_size > INDEX_THRESHOLD_BYTES:
            raise ValueError('large v6 pool requires the existing validated branch offset index; '
                             f'prepare with python -m sft.index_case_branches --manifest {path} --seed {seed}')
        return None
    from sft.index_case_branches import IndexedCaseBranchDataset
    return IndexedCaseBranchDataset(path, seed)


def source_identity(path, seed=1):
    from sft.v6_bootstrap_mixture import sha256
    index = existing_index(path, seed)
    stat = Path(path).stat()
    return dict(path=str(Path(path).resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                sha256=index.branch_selection['source_sha256'] if index else sha256(path))


def golden_candidates(path, window_budget, seed, index_seed=1):
    """Replay independent unique contexts from the original trained branch pool."""
    from sft.v6_bootstrap_mixture import read_rows
    index = existing_index(path, index_seed)
    if index is None:
        return read_rows(path), 'small_manifest_all_anchors'
    import numpy as np
    # Shared path prefixes have equal row offsets: deduplicate before sampling.
    _, unique = np.unique(index.offsets[:, 1], return_index=True)
    candidates = random.Random(seed).sample(list(map(int, unique)), min(window_budget, len(unique)))

    def rows():
        windows = 0
        for item in candidates:
            group = index.read_group(item)
            if windows + len(group) <= window_budget:
                yield from group
                windows += len(group)
            if windows == window_budget:
                break
    return rows(), 'validated_training_branch_index_unique_anchors'


def training_cases(path, index_seed=1):
    from sft.v6_bootstrap_mixture import read_rows
    index = existing_index(path, index_seed)
    if index is None:
        source = read_rows(path)
    else:
        def indexed_rows():
            cursor = 0
            with Path(index.branch_selection['selections_file']).open() as selections, Path(path).open('rb') as data:
                for line in selections:
                    selection = json.loads(line)
                    data.seek(int(index.offsets[cursor, 1]))
                    yield json.loads(data.readline())
                    cursor += sum(len(p['slow_anchor_ids']) for p in selection['paths'])
            if cursor != len(index):
                raise ValueError('case selection and anchor index disagree')
        source = indexed_rows()
    seen = set()
    for row in source:
        if row.get('split') != 'train':
            raise ValueError('collection source includes non-training rows')
        case = (row['source_scene'], row['source_task_id'])
        if case in seen:
            continue
        seen.add(case)
        yield dict(schema='v6_bootstrap_case_pool_v1', split='train',
                   source_scene=case[0], source_task_id=case[1], instruction=row['instruction'])
