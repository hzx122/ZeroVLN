"""Bounded-memory v6 merge, retaining the full per-NPZ validation gate."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import heapq
import itertools
import json
import os
from pathlib import Path

import numpy as np

from action_head import protocol_v6 as P
from sft.golden_context_beam import validate_context
from sft.merge_fm_nested_shards import REQUIRED, sha256


def row_key(row, decisions=False):
    case = (row['split'], row['source_scene'], row['source_task_id'])
    return case + ((int(row.get('option_sequence_index') or 0),) if decisions else
                   (row['slow_anchor_id'], float(row['fast_window_arc_m'])))


def sorted_case_rows(path, *, decisions=False):
    """Existing partition merge sorts by case; only sort one case at a time."""
    previous, buffered = None, []
    with path.open() as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row_key(row, decisions)[:3]
            if previous is not None and key != previous:
                if key < previous:
                    raise ValueError(f'input cases are not sorted: {path}')
                yield from sorted(buffered, key=lambda r: row_key(r, decisions))
                buffered = []
            previous = key
            buffered.append(row)
    yield from sorted(buffered, key=lambda r: row_key(r, decisions))


def validate_sample(row):
    missing = REQUIRED - row.keys()
    if missing:
        raise ValueError(f'missing sample fields: {sorted(missing)}')
    P.assert_protocol_identity(row)
    if row.get('schema') != P.BUNDLE_SCHEMA or row.get('decision_filter') != 'none':
        raise ValueError('invalid v6 bundle schema or direction gate')
    validate_context(row['slow_messages'], row['trajectory_map_path'])
    with np.load(row['sample']) as data:
        shape = tuple(data['target_actions'].shape)
        mask = data['action_mask']
        if shape != (16, 3) or mask.shape != (16,):
            raise ValueError(f'invalid target/mask: {row["sample"]}')
        return str(shape), int(mask.sum())


def has_prior_alignment(row):
    return any(part.get('text', '').startswith('[Tool result]') and
               '"direction_alignment"' in part.get('text', '')
               for message in row.get('slow_messages', []) if message.get('role') == 'user'
               for part in message.get('content', []) if isinstance(part, dict))


def merge_v6(data_root, shard_count, workers=16):
    data_root = Path(data_root)
    sources, manifests, decisions = [], [], []
    contract = None
    for index in range(shard_count):
        shard = data_root / 'shards' / f'shard_{index:03d}'
        manifest = shard / 'bundles/manifest.jsonl'
        provenance = shard / 'bundles/provenance.json'
        value = json.loads(provenance.read_text())
        P.assert_protocol_identity(value)
        current = value.get('beam_contract')
        if not current or current.get('direction_filter') is not False:
            raise ValueError('v6 shard is not a no-direction-filter beam')
        if contract is not None and current != contract:
            raise ValueError('inconsistent beam contract')
        if value.get('schema') != 'fm_nested_bundle_provenance_v4':
            raise ValueError('invalid shard provenance schema')
        contract = current
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        manifests.append(manifest)
        slow = shard / 'bundles/slow_decisions.jsonl'
        if slow.exists():
            decisions.append(slow)
        sources.append(dict(shard_index=index, manifest=str(manifest.resolve()),
                            samples=value['samples'], slow_decisions=value['slow_decisions'],
                            provenance=str(provenance.resolve()), provenance_sha256=sha256(provenance),
                            timing=json.loads((shard / 'timing.json').read_text()) if (shard / 'timing.json').exists() else None,
                            plan_summary=json.loads((shard / 'nested_plan.summary.json').read_text())))
    sampling = {}
    for key in ('slow_stride_m', 'slow_coverage_m', 'fast_stride_min_m',
                'fast_stride_max_m', 'sampling_seed', 'spacing_selection'):
        values = {json.dumps(s['plan_summary'].get(key), sort_keys=True) for s in sources}
        if len(values) != 1:
            raise ValueError(f'inconsistent sampling field: {key}')
        sampling[key] = sources[0]['plan_summary'].get(key)
    sampling.update(action_step_m=.1, action_horizon_steps=16)
    owners, scenes = {}, set()
    counts, shapes, steps = Counter(), Counter(), Counter()
    expected_prior = with_prior = sample_count = 0
    source_counts = Counter()

    def source_rows(index, path):
        for row in sorted_case_rows(path):
            owner = owners.setdefault(row['slow_anchor_id'], index)
            if owner != index:
                raise ValueError(f'anchor crosses shards: {row["slow_anchor_id"]}')
            source_counts[index] += 1
            yield row

    merged = heapq.merge(*(source_rows(i, p) for i, p in enumerate(manifests)), key=row_key)
    output = data_root / 'manifest.jsonl'
    temporary = output.with_suffix('.jsonl.tmp')
    last_anchor = None
    with ThreadPoolExecutor(max_workers=workers) as pool, temporary.open('w') as target:
        while batch := list(itertools.islice(merged, 256)):
            checked = list(pool.map(validate_sample, batch))
            for row, (shape, valid) in zip(batch, checked):
                if row['slow_anchor_id'] != last_anchor:
                    prior = int(row.get('option_sequence_index') or 0) > 0
                    present = has_prior_alignment(row)
                    if prior and not present:
                        raise ValueError('v6 causal context lost prior tool result')
                    expected_prior += prior
                    with_prior += present
                    last_anchor = row['slow_anchor_id']
                target.write(json.dumps(row, ensure_ascii=False) + '\n')
                counts[row['split']] += 1
                shapes[shape] += 1
                steps[valid] += 1
                scenes.add(row['source_scene'])
                sample_count += 1
            if sample_count % 65536 == 0:
                print(json.dumps(dict(event='stream_merge', validated_samples=sample_count)), flush=True)
    for i, source in enumerate(sources):
        if source_counts[i] != source['samples']:
            raise ValueError(f'shard count disagrees with provenance: {i}')
        source['manifest_sha256'] = sha256(manifests[i])
    slow_count = terminal_count = 0
    slow_output = data_root / 'slow_decisions.jsonl'
    slow_temp = slow_output.with_suffix('.jsonl.tmp')
    with slow_temp.open('w') as target:
        merged_slow = heapq.merge(*(sorted_case_rows(p, decisions=True) for p in decisions),
                                 key=lambda row: row_key(row, True))
        for row in merged_slow:
            P.assert_protocol_identity(row)
            if row.get('schema') != P.SLOW_DECISION_SCHEMA or row.get('decision_type') not in ('run_action_head', 'stop'):
                raise ValueError('invalid slow decision')
            target.write(json.dumps(row, ensure_ascii=False) + '\n')
            slow_count += 1
            terminal_count += row['decision_type'] == 'stop'
    if slow_count != sum(s['slow_decisions'] for s in sources):
        raise ValueError('slow decision count disagrees with provenance')
    report = dict(schema='fm_nested_shard_merge_v1', **P.protocol_identity(),
                  manifest=str(output.resolve()), manifest_sha256=sha256(temporary),
                  shard_count=shard_count, beam_contract=contract, shards=sources,
                  samples=sample_count, slow_decisions=slow_count, terminal_decisions=terminal_count,
                  anchors=len(owners), scenes=len(scenes), split_samples=dict(counts),
                  target_shapes=dict(shapes), valid_action_steps=dict(sorted(steps.items())),
                  training_ready=bool(sample_count and set(shapes) == {'(16, 3)'}),
                  sampling=sampling, merge_mode='case_stream_npz_validation_v1',
                  v3_causal_context_audit=dict(unique_anchor_contexts=len(owners),
                      prior_tool_result_expected=expected_prior,
                      prior_direction_alignment_present=with_prior, missing_prior_direction_alignment=0))
    os.replace(temporary, output)
    os.replace(slow_temp, slow_output)
    report_temp = data_root / 'provenance.json.tmp'
    report_temp.write_text(json.dumps(report, indent=2) + '\n')
    os.replace(report_temp, data_root / 'provenance.json')
    print(json.dumps({k: v for k, v in report.items() if k != 'shards'}), flush=True)
    return report
