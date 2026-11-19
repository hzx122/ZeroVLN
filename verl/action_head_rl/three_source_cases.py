"""Reuse the frozen semantic-grounding task population for online RL.

The pool contains task identities/instructions, never teacher calls or SFT
features. Each step draws independently; a resumed step has the same draw.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random


def case_identity(row):
    return (row.get('dataset', 'sage'), str(row['scene']), str(row['episode_id']))


def sample_step_cases(cases, *, step, count, seed, source_counts=None):
    if step < 1 or not 1 <= count <= len(cases):
        raise ValueError('positive step/count and enough distinct pool cases required')
    identities = [case_identity(row) for row in cases]
    if len(set(identities)) != len(identities):
        raise ValueError('random sampling requires unique backend/scene/episode identities')
    digest = hashlib.sha256(f'three-source-rl-v1:{seed}:{step}'.encode()).digest()
    rng = random.Random(int.from_bytes(digest, 'big'))
    if source_counts is None:
        return rng.sample(cases, count)
    if (not isinstance(source_counts, dict)
            or set(source_counts) != {'sage', 'r2r', 'rxr'}
            or any(type(value) is not int or value <= 0 for value in source_counts.values())
            or sum(source_counts.values()) != count):
        raise ValueError('source_counts must contain positive integer sage/r2r/rxr quotas summing to count')
    selected = []
    for backend in ('sage', 'r2r', 'rxr'):
        population = [row for row in cases if row.get('dataset', 'sage') == backend]
        if len(population) < source_counts[backend]:
            raise ValueError(f'not enough distinct {backend} cases for source quota')
        selected.extend(rng.sample(population, source_counts[backend]))
    rng.shuffle(selected)
    return selected


def prepare_population(report_path, output):
    """Project all executable source tasks; retain known source failures.

    Reuses the population already frozen for the three-source open-loop job.
    No successful-teacher-collection or export-based filtering is performed.
    """
    report_path, output = Path(report_path).resolve(), Path(output).resolve()
    allowed = Path('/data/zerovln/artifacts')
    if not output.is_relative_to(allowed) or output.exists():
        raise ValueError('requires a fresh output directory on the artifact disk')
    raw_report = report_path.read_bytes()
    report = json.loads(raw_report)
    if set(report['sources']) != {'sage', 'r2r', 'rxr'}:
        raise ValueError('requires the exact three-source frozen population')
    if report.get('same_floor_scope', {}).get('enabled') is not True:
        raise ValueError('SAGE same-floor population audit is required')
    cases, failures, bindings, original_counts = [], [], [], {}
    for backend in ('sage', 'r2r', 'rxr'):
        source = report['sources'][backend]
        path = Path(source['manifest'])
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != source['manifest_sha256']:
            raise ValueError(f'open-loop source population changed: {path}')
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if len(rows) != source['population_count']:
            raise ValueError('source population count changed')
        bindings.append(dict(path=str(path), sha256=digest))
        original_counts[backend] = len(rows)
        for row in rows:
            if row['dataset'] != backend:
                raise ValueError('source backend mismatch')
            if row.get('status') == 'source_unexecutable':
                failures.append(row)
                continue
            if row.get('source_validation_errors') or not row.get('instruction', '').strip():
                raise ValueError('unrecorded invalid source task')
            if backend == 'sage' and row.get('same_floor_scope', {}).get('status') != 'retained':
                raise ValueError('SAGE task was not admitted by frozen same-floor audit')
            if backend == 'rxr' and row.get('language') not in ('en-US', 'en-IN'):
                raise ValueError('RxR population must retain its frozen English selection')
            cases.append(dict(dataset=backend, backend=backend, scene=row['scene'],
                episode_id=str(row['task_id']), instruction=row['instruction'],
                instruction_type=row.get('instruction_type', backend), stage='p0',
                source_provenance=row['source_provenance']))
    identities = [case_identity(row) for row in cases]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate task identity in merged source pool')
    cases.sort(key=case_identity)
    content = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n' for row in cases)
    output.mkdir(parents=True)
    (output/'train_cases.jsonl').write_text(content)
    (output/'source_unexecutable.jsonl').write_text(''.join(
        json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n' for row in failures))
    receipt = dict(schema='three_source_grounding_rl_pool_v1',
        population_report=str(report_path), population_report_sha256=hashlib.sha256(raw_report).hexdigest(),
        original_counts=original_counts, eligible_counts=dict(Counter(row['dataset'] for row in cases)),
        source_unexecutable_counts=dict(Counter(row['dataset'] for row in failures)),
        source_bindings=bindings, train_cases_sha256=hashlib.sha256(content.encode()).hexdigest(),
        sampling='uniform independent per-step draw without replacement within the step; backend ratio follows pool size',
        filtering='only frozen source_unexecutable failures; no filtering by teacher success/export availability',
        teacher_labels_consumed=False, policy_inputs='original instruction and current online observations')
    (output/'provenance.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    return receipt
