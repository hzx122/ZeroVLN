"""Bounded v6 replay manifests, with whole anchors and no silent duplication."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random

from sft.action_head_data_contract import (
    ANCHOR_SAMPLING, ONLINE_CONTRACT, validate_v6_training_row)


def read_rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def rows_with_offsets(path):
    with Path(path).open('rb') as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if line.strip():
                yield json.loads(line), offset


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def select_anchors(groups, budget, seed, *, age_balanced=True):
    """Approximate equal *window* exposure per round/type without cutting anchors.

    Smaller remaining buckets can donate unused capacity. A budget smaller than
    every remaining anchor leaves explicit slack instead of truncating labels.
    """
    buckets = defaultdict(list)
    for group in groups:
        row = group[0]
        key = (int(row.get('bootstrap_source_round', 0)) if age_balanced else 0,
               row.get('sample_type', 'golden'))
        buckets[key].append(group)
    rng = random.Random(seed)
    for key in sorted(buckets):
        buckets[key].sort(key=lambda group: group[0]['slow_anchor_id'])
        rng.shuffle(buckets[key])
    exposure = Counter()
    selected, remaining = [], budget
    while buckets and remaining:
        key = min(buckets, key=lambda key: (exposure[key], key))
        group = buckets[key].pop()
        if not buckets[key]:
            del buckets[key]
        if len(group) <= remaining:
            selected.append(group)
            exposure[key] += len(group)
            remaining -= len(group)
    return selected


def mix(*, corrections, output_dir, validation_manifest, window_budget,
        mode='mixed', golden_manifest=None, golden_fraction=0.0,
        age_balanced=True, seed=1, require_full_budget=False, anchor_budget=None):
    if mode not in ('A-only', 'B-only', 'mixed', 'collision-only', 'failure-pair'):
        raise ValueError('unknown correction mixture mode')
    if not 0 <= golden_fraction < 1:
        raise ValueError('golden_fraction must be in [0, 1)')
    if anchor_budget is None:
        if window_budget <= 0:
            raise ValueError('positive window budget required')
    elif anchor_budget <= 0 or window_budget != 0 or golden_fraction:
        raise ValueError('anchor budget requires no window budget or golden replay')
    if golden_fraction and golden_manifest is None:
        raise ValueError('golden replay requires golden_manifest')
    if not corrections:
        raise ValueError('at least one correction manifest is required')
    from action_head import protocol_v6 as protocol
    validation_scenes, validation_cases = set(), set()
    validation_count = 0
    if Path(validation_manifest).suffix == '.json':
        # Closed-loop case identity does not depend on the prompt protocol.
        for case in json.loads(Path(validation_manifest).read_text())['cases']:
            validation_scenes.add(case['scene'])
            validation_cases.add((case['scene'], case['episode_id']))
            validation_count += 1
    else:
        for row in read_rows(validation_manifest):
            protocol.assert_protocol_identity(row)
            if row.get('split') != 'validation':
                raise ValueError('selection manifest must contain only held-out validation rows')
            validation_scenes.add(row['source_scene'])
            validation_cases.add((row['source_scene'], row['source_task_id']))
            validation_count += 1
    if not validation_count:
        raise ValueError('empty selection manifest')
    groups, identities, sources = {}, {}, []
    source_specs = [('correction', int(round_index), Path(path))
                    for round_index, path in corrections]
    if golden_fraction:
        source_specs.append(('golden', 0, Path(golden_manifest)))
    for kind, round_index, path in source_specs:
        if kind == 'correction' and round_index <= 0:
            raise ValueError('correction source round must be positive')
        count = 0
        from sft.v6_bootstrap_sources import golden_candidates, source_identity
        if kind == 'golden':
            candidates, source_sampling = golden_candidates(path, int(window_budget * golden_fraction), seed+1)
            items = ((row, None) for row in candidates)
        else:
            items, source_sampling = rows_with_offsets(path), 'all_validated_anchors'
        for row, offset in items:
            if row.get('split') != 'train':
                if kind == 'golden':
                    continue
                raise ValueError('corrections may contain only training rows')
            validate_v6_training_row(row)
            if kind == 'correction' and row.get('data_contract') != ONLINE_CONTRACT:
                raise ValueError('correction source does not have the online data contract')
            if kind == 'golden' and row.get('data_contract') == ONLINE_CONTRACT:
                raise ValueError('golden replay source is an online correction')
            case = (row['source_scene'], row['source_task_id'])
            if case in validation_cases or row['source_scene'] in validation_scenes:
                raise ValueError(f'train/selection scene overlap: {case}')
            count += 1
            permitted = {
                'A-only': {'execution_relabel'}, 'B-only': {'recovery'},
                'collision-only': {'collision_relabel', 'golden_route_relabel'},
                'failure-pair': {'direction_mismatch_relabel', 'collision_relabel',
                                 'golden_route_relabel'},
                'mixed': {'execution_relabel', 'recovery',
                          'direction_mismatch_relabel', 'collision_relabel',
                          'golden_route_relabel'}}[mode]
            if kind == 'correction' and row['sample_type'] not in permitted:
                continue
            anchor = row['slow_anchor_id']
            key = (anchor, row['fast_window_index'])
            # Repeated input paths are rejected; accidental duplicate replay
            # must never change source weights silently.
            if key in identities:
                raise ValueError(f'duplicate training window: {key}')
            identities[key] = row if offset is None else (path, offset)
            # Keep only compact sampling metadata for cumulative corrections;
            # slow contexts and tensors are decoded for selected windows only.
            marked = {**{field: row.get(field) for field in (
                      'slow_anchor_id', 'fast_window_index', 'source_scene', 'source_task_id', 'sample_type')},
                      'sampling_contract': ANCHOR_SAMPLING,
                      'bootstrap_source_kind': kind, 'bootstrap_source_round': round_index}
            group = groups.setdefault(anchor, [])
            if group and any(group[0].get(field) != marked.get(field) for field in (
                    'source_scene', 'source_task_id', 'bootstrap_source_kind',
                    'bootstrap_source_round', 'sample_type')):
                raise ValueError(f'mixed anchor identity: {anchor}')
            group.append(marked)
        sources.append(dict(kind=kind, round=round_index, manifest=str(path.resolve()),
                            sha256=source_identity(path)['sha256'] if kind == 'golden' else sha256(path),
                            candidate_train_windows=count, sampling=source_sampling))
    correction_groups = [g for g in groups.values() if g[0]['bootstrap_source_kind'] == 'correction']
    golden_groups = [g for g in groups.values() if g[0]['bootstrap_source_kind'] == 'golden']
    golden_budget = int(window_budget * golden_fraction)
    correction_budget = (sum(len(group) for group in correction_groups)
                         if anchor_budget is not None else window_budget - golden_budget)
    selected_corrections = select_anchors(correction_groups, correction_budget,
                                          seed, age_balanced=age_balanced)
    if anchor_budget is not None:
        selected_corrections = selected_corrections[:anchor_budget]
    if not selected_corrections:
        raise ValueError('no approved correction anchors fit the budget; stop instead of training an empty pool')
    selected = selected_corrections + select_anchors(golden_groups, golden_budget, seed + 1)
    random.Random(seed).shuffle(selected)
    rows = []
    for group in selected:
        hydrated = []
        for compact in sorted(group, key=lambda row: row['fast_window_index']):
            location = identities[(compact['slow_anchor_id'], compact['fast_window_index'])]
            if isinstance(location, tuple):
                path, offset = location
                with path.open('rb') as source:
                    source.seek(offset)
                    original = json.loads(source.readline())
            else:
                original = location
            # Do not inject sample_type=None into golden provenance.
            marked = {**original, **{key: value for key, value in compact.items() if value is not None}}
            if hydrated and any(marked.get(field) != hydrated[0].get(field)
                                for field in ('slow_messages', 'slow_completion')):
                raise ValueError('selected anchor mixes slow conditioning')
            hydrated.append(marked)
        rows.extend(hydrated)
    if require_full_budget and anchor_budget is not None and len(selected) != anchor_budget:
        raise ValueError(f'slow-anchor budget underfilled: {len(selected)}/{anchor_budget}')
    if require_full_budget and anchor_budget is None and len(rows) != window_budget:
        raise ValueError(f'whole-anchor budget underfilled: {len(rows)}/{window_budget}')
    report = dict(schema='v6_bootstrap_mixture_v1', **protocol.protocol_identity(),
                  sampling_contract=ANCHOR_SAMPLING, mode=mode, seed=seed,
                  window_budget=window_budget, anchor_budget=anchor_budget,
                  windows=len(rows), anchors=len(selected),
                  unused_budget=(0 if anchor_budget is not None else window_budget-len(rows)),
                  unused_anchor_budget=(None if anchor_budget is None else anchor_budget-len(selected)),
                  golden_window_budget=golden_budget,
                  age_balanced=age_balanced, oversampling=False, partial_anchors=False,
                  balance_unit='windows_per_round_and_type', sources=sources,
                  validation_manifest=str(Path(validation_manifest).resolve()),
                  validation_sha256=sha256(validation_manifest),
                  selected_by_source=dict(Counter(row['bootstrap_source_kind'] for row in rows)),
                  selected_by_round=dict(Counter(str(row['bootstrap_source_round']) for row in rows)),
                  selected_by_type=dict(Counter(row.get('sample_type', 'golden') for row in rows)))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / 'manifest.jsonl'
    temporary = output.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    os.replace(temporary, output)
    report.update(output_manifest=str(output.resolve()), output_sha256=sha256(output))
    write_json(output_dir / 'provenance.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--correction', action='append', required=True,
                        help='ROUND:MANIFEST, one per collected round')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--validation-manifest', required=True, type=Path)
    parser.add_argument('--window-budget', required=True, type=int)
    parser.add_argument('--mode', choices=('A-only', 'B-only', 'mixed', 'collision-only', 'failure-pair'), default='mixed')
    parser.add_argument('--golden-manifest', type=Path)
    parser.add_argument('--golden-fraction', type=float, default=0)
    parser.add_argument('--no-age-balance', action='store_true')
    parser.add_argument('--require-full-budget', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    corrections = [(int(value.split(':', 1)[0]), Path(value.split(':', 1)[1]))
                   for value in args.correction]
    print(json.dumps(mix(corrections=corrections, output_dir=args.output_dir,
                        validation_manifest=args.validation_manifest,
                        window_budget=args.window_budget, mode=args.mode,
                        golden_manifest=args.golden_manifest, golden_fraction=args.golden_fraction,
                        age_balanced=not args.no_age_balance, seed=args.seed,
                        require_full_budget=args.require_full_budget), indent=2))


if __name__ == '__main__':
    main()
