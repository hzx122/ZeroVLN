"""Four independent suite runs on immutable cases, with logical-ID paired metrics."""
from pathlib import Path
import math
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'train'))
from rule_teacher_training_handoff import load, save, bind


def work_items(config):
    repeats = config.get('evaluation_repeats', 1)
    if type(repeats) is not int or repeats < 1:
        raise ValueError('companion requires a positive number of complete attempts')
    for step in config['steps']:
        source = config.get('checkpoint_sources', {}).get(str(step), config)
        for repeat in range(1, (1 if config.get('pooled_attempts') else repeats) + 1):
            unit = dict(config, **{k: source[k] for k in
                ('training_root', 'training_config', 'checkpoint_root', 'checkpoint_index')})
            unit['evaluation_repeat_index'] = repeat
            name = f'global_step_{step:06d}'
            checkpoint = Path(config.get('checkpoint_paths', {}).get(str(step),
                str(Path(unit['checkpoint_root']) / name)))
            unit['evaluation_checkpoint_name'] = name
            base = Path(config['root']) / 'evaluation'
            output = (base / f'repeat_{repeat:02d}' / name
                      if repeats > 1 else base / name)
            historical = config.get('historical_attempts', {}).get(f'{step}:{repeat}')
            if historical is None and repeat == 1:
                historical = config.get('historical_first_attempts', {}).get(str(step))
            yield step, repeat, unit, checkpoint, output, historical


def paired_metrics(rounds, expected_ids, expected_repeats=None):
    """Reject partial/duplicate attempts; pair by instruction ID, not scene/episode."""
    k = len(rounds)
    if k < 1 or (expected_repeats is not None and k != expected_repeats) or len(expected_ids) != len(set(expected_ids)):
        raise ValueError('need all requested rounds and unique case identities')
    for rows in rounds:
        if set(rows) != set(expected_ids):
            raise ValueError('repeat case coverage differs')
        for row in rows.values():
            if row.get('evaluation_valid') is not True or type(row.get('success')) is not bool:
                raise ValueError('invalid attempts cannot become navigation failures')
            if not math.isfinite(row['spl']) or not 0 <= row['spl'] <= 1:
                raise ValueError('invalid SPL')
    n = len(expected_ids)
    if not n:
        raise ValueError('empty evaluation')
    wins = sum(row['success'] for rows in rounds for row in rows.values())
    any_wins = sum(any(rows[key]['success'] for rows in rounds) for key in expected_ids)
    result = dict(evaluation_valid=True, cases=n, valid_attempts=k*n,
        pass_at_1=dict(value=wins/(k*n), numerator=wins, denominator=k*n),
        spl=sum(row['spl'] for rows in rounds for row in rows.values())/(k*n),
        per_round_sr=[sum(row['success'] for row in rows.values())/n for rows in rounds])
    if k > 1:
        result[f'pass_at_{k}'] = dict(value=any_wins/n, numerator=any_wins, denominator=n, k=k)
    if all('oracle_success' in row for rows in rounds for row in rows.values()):
        result['osr'] = sum(row['oracle_success'] for rows in rounds for row in rows.values())/(k*n)
    return result


def verify_historical(config, step, reference):
    path = Path(reference['path'])
    if bind(path) != reference:
        raise ValueError('historical suite summary changed')
    summary = load(path)
    if summary.get('evaluation_valid') is not True:
        raise ValueError('historical first attempt is incomplete')
    old = load(path.parent/'plan.json')
    base = load(config['base_plan'])
    from closedloop_evaluation_contract import LEGACY_CONTINUATION
    if base.get('config', {}).get('closedloop_protocol') == LEGACY_CONTINUATION:
        # Explicit user-authorized continuation: same cases and benchmark settings;
        # ordering/source_count may differ when loading the original fixed subset.
        for key in ('max_tool_calls', 'observation_profiles', 'environment', 'seed', 'gym_actors'):
            if old['config'].get(key) != base['config'].get(key):
                raise ValueError('historical continuation setting differs: ' + key)
        for stage in base['stages']:
            previous = next(s for s in old['stages'] if s['benchmark'] == stage['benchmark'])
            current_cases = {c['logical_id']: c for c in load(stage['manifest'])['cases']}
            previous_cases = {c['logical_id']: c for c in load(previous['manifest'])['cases']}
            if current_cases != previous_cases:
                raise ValueError('historical continuation fixed cases differ')
    elif old['populations'] != base['populations'] or [s['benchmark'] for s in old['stages']] != ['sage-3d','r2r','rxr']:
        raise ValueError('historical case population/protocol differs')
    if old['config']['checkpoints'][0]['name'] != f'global_step_{step:06d}':
        raise ValueError('historical checkpoint differs')
    return path.parent


def summarize_checkpoint(config, step):
    root = Path(config['root'])
    outputs = []
    for s, repeat, unit, checkpoint, output, historical in work_items(config):
        if s != step:
            continue
        output = verify_historical(config, step, historical) if historical else output
        if not (output/'summary.json').exists():
            return None
        if load(output/'summary.json').get('evaluation_valid') is not True:
            raise ValueError('a repeat suite is invalid')
        outputs.append(output)
    k = config.get('evaluation_repeats', 1)
    runs = 1 if config.get('pooled_attempts') else k
    if len(outputs) != runs or len(set(outputs)) != runs:
        raise ValueError('attempt output directories must be distinct')
    base = load(config['base_plan'])
    common = summarize_outputs(base, outputs, f'global_step_{step:06d}')
    result = dict(evaluation_valid=True, step=step, repeats=k,
        definition=f'pass@1: mean of all {k} trials; pass@{k}: any success per logical case',
        benchmarks=common['benchmarks'], first_attempt_reused=str(step) in config.get('historical_first_attempts', {}),
        attempts=list(map(str, outputs)), evidence=common['evidence'])
    save(root/'repeat-summary'/f'global_step_{step:06d}.json', result)
    return result


def summarize_outputs(base, outputs, checkpoint_name):
    """Shared paired scoring for full, RL-companion and open-loop evaluation."""
    k = base.get('config', {}).get('evaluation_repeats', 4)
    if base.get('config', {}).get('pooled_attempts'):
        if len(outputs) != 1:
            raise ValueError('pooled evaluation requires one service lifecycle per checkpoint')
        output = Path(outputs[0])
        if load(output/'summary.json').get('evaluation_valid') is not True:
            raise ValueError('incomplete pooled suite')
        plan = load(output/'plan.json')
        metrics, evidence = {}, []
        for stage in plan['stages']:
            if stage['checkpoint'] != checkpoint_name:
                continue
            path = output/'stages'/f"{stage['index']:02d}_{stage['benchmark']}"/checkpoint_name/'aggregate.json'
            aggregate = load(path)
            n = len(load(stage['manifest'])['cases'])
            paired = aggregate.get('paired_metrics', {})
            if (aggregate.get('evaluation_valid') is not True or aggregate.get('expected') != n
                    or paired.get('evaluation_valid') is not True or paired.get('cases') != n
                    or paired.get('valid_attempts') != n*k
                    or paired.get(f'pass_at_{k}', {}).get('k', 1) != k):
                raise ValueError('pooled stage is missing or invalid repeated attempts')
            metrics[stage['benchmark']] = paired
            evidence.append(bind(path))
        if not metrics:
            raise ValueError('pooled checkpoint has no stages')
        return dict(evaluation_valid=True, checkpoint=checkpoint_name, repeats=k,
            benchmarks=metrics, evidence=evidence, attempts=list(map(str,outputs)))
    if len(outputs) != k or len(set(map(str, outputs))) != k:
        raise ValueError('requested number of distinct attempt directories required')
    outputs = list(map(Path, outputs))
    for output in outputs:
        if load(output/'summary.json').get('evaluation_valid') is not True:
            raise ValueError('incomplete/invalid suite cannot enter paired metrics')
    metrics = {}
    evidence = []
    stages = [s for s in base['stages'] if s.get('checkpoint') == checkpoint_name]
    if not stages:
        stages = base['stages']  # Legacy companion base templates have placeholder checkpoint names.
    for stage in stages:
        bench = stage['benchmark']
        cases = load(stage['manifest'])['cases']
        expected = {c['logical_id']: c for c in cases}
        rounds = []
        identity = None
        for output in outputs:
            plan = load(output/'plan.json')
            ckpt = next(c['identity'] for c in plan['config']['checkpoints'] if c['name'] == checkpoint_name)
            if identity is not None and identity != ckpt:
                raise ValueError('checkpoint identity differs across attempts')
            identity = ckpt
            directory = output/'stages'/f"{stage['index']:02d}_{bench}"/checkpoint_name
            aggregate = load(directory/'aggregate.json')
            if aggregate.get('evaluation_valid') is not True or aggregate.get('expected') != len(cases):
                raise ValueError('invalid per-benchmark aggregate')
            pattern = ('node_*/evaluation/worker_*/rollouts/**/*.json' if bench == 'sage-3d'
                       else f'node_*/evaluation/{bench}/rollouts/*.json')
            rows = {}
            for path in sorted(directory.glob(pattern)):
                row = load(path)
                key = row.get('logical_id')
                if key not in expected or key in rows:
                    raise ValueError('unknown or duplicate rollout')
                if any(row.get(k) != v for k,v in expected[key].items()):
                    raise ValueError('rollout manifest identity changed')
                selected = {k: row[k] for k in ('evaluation_valid','success','spl')}
                # Native OSR is stored in the native_metrics block.
                native = row.get('native_metrics', {})
                if 'oracle_success' in row:
                    selected['oracle_success'] = row['oracle_success']
                elif 'oracle_success' in native:
                    selected['oracle_success'] = native['oracle_success']
                rows[key] = selected
            rounds.append(rows)
            evidence.append(bind(directory/'aggregate.json'))
        metrics[bench] = paired_metrics(rounds, list(expected), k)
    return dict(evaluation_valid=True, checkpoint=checkpoint_name, repeats=k,
        benchmarks=metrics, evidence=evidence, attempts=list(map(str, outputs)))
