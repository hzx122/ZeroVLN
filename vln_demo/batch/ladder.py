"""Run nested benchmarks incrementally, retaining one archive per valid episode."""
from collections import Counter
import fcntl
import json
from pathlib import Path
import subprocess

from eval_sets import digest, load_ladder
from recovery import load_results
from workspace import REPO, verified_result
from navigation_metrics import finite_result
from version_audit import observe_runtime, record_runtime_change, runtime_difference


def reusable(result):
    return result.get('state') == 'finished' and result.get('reason') == 'stop' and finite_result(result)


def resolved(result):
    return reusable(result) or (result.get('state') == 'failed'
        and result.get('reason') == 'invalid_geometry'
        and result.get('geometry_validation') == 'initial_goal_unreachable')


def report(rows, results):
    datasets = {}
    for dataset in ('r2r', 'rxr'):
        selected = [results[r['logical_id']] for r in rows if r['dataset'] == dataset]
        if not selected:
            continue
        valid = [r for r in selected if reusable(r)]
        budget_exhausted = sum(r.get('reason') == 'timeout' for r in selected)
        invalid_episodes = sum(r.get('reason') in ('invalid_geometry', 'invalid_metrics')
                               or (r.get('reason') == 'stop' and not finite_result(r)) for r in selected)
        datasets[dataset] = dict(episodes=len(selected), completed=len(valid),
            incomplete=len(selected)-len(valid), time_budget_exhausted=budget_exhausted,
            invalid_episodes=invalid_episodes,
            infrastructure_failures=len(selected)-len(valid)-budget_exhausted-invalid_episodes,
            score_final=len(valid) == len(selected),
            successes=sum(bool(r.get('navigation_success')) for r in valid),
            success_rate=sum(bool(r.get('navigation_success')) for r in valid)/len(selected),
            valid_success_rate=(sum(bool(r.get('navigation_success')) for r in valid)/len(valid) if valid else None),
            reasons=dict(Counter(r.get('reason') for r in selected)))
        metric_names = {key for r in valid for key in r.get('metrics', {})}
        datasets[dataset]['completed_metrics_mean'] = {}
        for key in sorted(metric_names):
            values = [r['metrics'][key] for r in valid
                      if isinstance(r.get('metrics', {}).get(key), (float, int))]
            if values:
                datasets[dataset]['completed_metrics_mean'][key] = dict(mean=sum(values)/len(values), n=len(values))
    return datasets


def fingerprint(directory):
    files = [REPO/'third_party/pi-agent/.pi/workspaces/SpatialOnboarding/AGENTS.md',
             REPO/'third_party/pi-agent/.pi/extensions/vln-console.ts',
             REPO/'vln_demo/config/pi/SpatialOnboarding/settings.json',
             REPO/'vln_demo/config/pi/project-models.ts', Path(directory)/'lock.json']
    files.append(REPO/'vln_demo/vllm/sglang_profiles.json')
    files += sorted((REPO/'vln_demo/batch').glob('*.py'))
    files += sorted((REPO/'vln_demo/batch').glob('*.mjs'))
    files += sorted((REPO/'vln_demo/src').rglob('*.py'))
    files += sorted((REPO/'vln_demo/src').glob('*.yaml'))
    hashes = {}
    for path in files:
        try:
            hashes[str(path)] = digest(path)
        except OSError as exc:
            hashes[str(path)] = {'read_error': f'{type(exc).__name__}: {exc}'}
    try:
        pi_commit = subprocess.check_output(['git', '-C', str(REPO/'third_party/pi-agent'),
                                             'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        pi_commit = {'read_error': f'{type(exc).__name__}: {exc}'}
    return dict(files=hashes, pi_commit=pi_commit)


def check_resume_runtime(old, new):
    """Keep service/settings checks; source version changes are recorded only."""
    if old['services'] != new['services']:
        raise ValueError('resume services changed')
    for key in ('workers', 'render_mode'):
        if old['settings'].get(key) != new['settings'].get(key):
            raise ValueError('resume setting changed: ' + key)
    if new['settings'].get('timeout') not in (0, old['settings'].get('timeout')):
        raise ValueError('resume may only preserve or disable the wall-clock limit')
    return runtime_difference(old['runtime'], new['runtime'])['files']


def resume_results(source, current, rows):
    source = Path(source).resolve()
    old = json.loads((source/'ladder-runtime.json').read_text())
    changes = check_resume_runtime(old, current)
    lookup = {r['logical_id']: r for r in rows}
    results = {}
    maximum = current['settings'].get('max_tool_calls')
    def fits_budget(result):
        if not maximum:
            return True
        try:
            report = json.loads((Path(result['archive'])/'workspace/.runtime/agent-result.json').read_text())
            calls = report['toolCalls']
            return (type(calls) is int and calls <= maximum and
                    (report.get('termination') == 'normal' or
                     (report.get('termination') == 'tool_limit' and calls == maximum)))
        except (OSError, ValueError, KeyError):
            return False
    # A subsequent restart also inherits verified references from its parent.
    cache = source/'reused-results.json'
    if cache.exists():
        for result in json.loads(cache.read_text()).values():
            row = lookup.get(result['logical_id'])
            if row is None:
                continue
            try:
                checked = verified_result(result['archive'])
                if reusable(checked) and all(checked.get(k) == v for k, v in row.items()) and fits_budget(result):
                    results[row['logical_id']] = dict(checked, archive=result['archive'])
            except (OSError, ValueError, KeyError) as exc:
                print('VLN_REUSE_REJECTED ' + str(exc), flush=True)
    for level in ('100', '200', '500', '1000', 'full'):
        run_dir = source/level/'episodes-run'
        # Metadata only shortlists candidates. load_results verifies every byte
        # before acceptance, including sessions, images and the loaded prompt.
        candidates = set()
        for path in (run_dir/'episodes').glob('*/*/*/*/*/attempt-*/result.json'):
            try:
                result = json.loads(path.read_text())
                if reusable(result):
                    candidates.add(result['logical_id'])
            except (OSError, ValueError, KeyError):
                pass
        selected = [lookup[i] for i in candidates if i in lookup and i not in results]
        if selected:
            print('VLN_REUSE_VERIFY ' + json.dumps(dict(stage=level, candidates=len(selected))), flush=True)
            saved = load_results(run_dir, selected, normal_only=True)
            results.update({i: r for i, r in saved.items() if fits_budget(r)})
    return results, dict(source=str(source), count=len(results), runtime_changes=changes,
                         version_changes=runtime_difference(old['runtime'], current['runtime']),
                         old_timeout=old['settings'].get('timeout'), new_timeout=current['settings'].get('timeout'))


def run(config, services):
    from launch import runner_command
    directory = Path(config['evaluation_ladder'])
    all_stages = load_ladder(directory, verify_sources=True)
    size = config.get('evaluation_size', 'ladder')
    stages = {level: [r for r in rows if r['dataset'] in config.get('datasets', 'r2r,rxr').split(',')]
              for level, rows in all_stages.items() if size == 'ladder' or level == size}
    if not stages or any(not rows for rows in stages.values()):
        raise ValueError('empty evaluation selection')
    root = Path(config['run_dir'])
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.ladder.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = dict(runtime=fingerprint(directory), services=services,
                       settings={k: config.get(k) for k in ('workers', 'timeout', 'render_mode', 'max_tool_calls', 'agent_profile', 'datasets', 'evaluation_size')},
                       resume_ladder=config.get('resume_ladder'))
        provenance = root/'ladder-runtime.json'
        if provenance.exists():
            previous = json.loads(provenance.read_text())
            if any(previous.get(key) != current.get(key) for key in ('services', 'settings', 'resume_ladder')):
                raise ValueError('ladder services or evaluation settings changed')
            observed = observe_runtime(root, previous['runtime'], current['runtime'], phase='restart')
        else:
            provenance.write_text(json.dumps(current, indent=2)+'\n')
            observed = observe_runtime(root, current['runtime'], current['runtime'], phase='start')
        results = {}
        if config.get('resume_ladder'):
            source = Path(config['resume_ladder']).resolve()
            if source == root.resolve():
                raise ValueError('resume_ladder must refer to a different run')
            # Refuse a source still being scheduled. Its committed archives are
            # append-only, so references stay valid after this lock is released.
            with (source/'.ladder.lock').open('a') as source_lock:
                fcntl.flock(source_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                results, receipt = resume_results(source, current, all_stages['full'])
            (root/'reused-results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n')
            (root/'reuse.json').write_text(json.dumps(receipt, indent=2)+'\n')
            record_runtime_change(root, json.loads((source/'ladder-runtime.json').read_text())['runtime'],
                                  current['runtime'], phase='resume_from')
            print('VLN_LADDER_REUSED ' + json.dumps(receipt), flush=True)
        for level, rows in stages.items():
            observed = observe_runtime(root, observed, fingerprint(directory), phase=level+':before')
            stage = root/level
            stage.mkdir(exist_ok=True)
            manifest = stage/'input.jsonl'
            if not manifest.exists():
                missing = [r for r in rows if not resolved(results.get(r['logical_id'], {}))]
                manifest.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in missing))
            missing = [json.loads(line) for line in manifest.read_text().splitlines()]
            stage_config = {**config, 'run_dir': str(stage/'episodes-run'), 'manifest': str(manifest)}
            stage_config.pop('resume_from', None)
            print('VLN_LADDER_STAGE ' + json.dumps(dict(stage=level, total=len(rows), new_or_retry=len(missing))), flush=True)
            if missing:
                subprocess.run(runner_command(stage_config, services), check=True)
                results.update(load_results(stage/'episodes-run', missing))
            observed = observe_runtime(root, observed, fingerprint(directory), phase=level+':after')
            if any(r['logical_id'] not in results for r in rows):
                raise RuntimeError('stage has unverified/missing archives; refusing expansion')
            summary = dict(stage=level, datasets=report(rows, results),
                new_or_retry=len(missing), reused=len(rows)-len(missing),
                archives={r['logical_id']: results[r['logical_id']]['archive'] for r in rows})
            (stage/'report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
            (root/'progress.json').write_text(json.dumps(summary, ensure_ascii=False)+'\n')
            print('VLN_LADDER_RESULT ' + json.dumps(summary['datasets']), flush=True)
            # Poor navigation is a measurement, not a reason to select other samples.
            # Widespread infrastructure failure, however, must not trigger a larger run.
            if any(d['infrastructure_failures']/d['episodes'] >= .2 for d in summary['datasets'].values()):
                raise RuntimeError('at least 20% infrastructure failures; inspect this stage before expansion')
        (root/'_SUCCESS.json').write_text(json.dumps(dict(stages=list(stages), episodes=len(results)))+'\n')
