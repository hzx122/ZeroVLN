#!/usr/bin/env python3
"""Freeze instruction/start-only populations; dispatch the existing pilots.

The collector/controller remains in rule_teacher_pilot.py and
mp3d_rule_teacher_pilot.py. This entry only prepares source populations,
dispatches official SAGE instruction synthesis, and retains every failure in
the population ledger. It never labels a call from an episode route or goal.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
ART = Path(os.environ.get('RL_ARTIFACT_ROOT', '/data/closed_loop_3dgs_rl'))
SAGE_SPLIT = 'official_gvln_action_head_p0_v3full827train'
SEED = 20260930


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    # Shared by stage supervisors even when no ACTION_HEAD_SUITE_PLAN is set.
    # Reuse the target/PID-bound atomic writer and exact ESTALE/ENOENT retry.
    from align_three_source_observations import write_json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    partial.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n'
                              for row in rows))
    partial.replace(path)


def rank(value, seed):
    return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()


def limit_sage_pairs(population, shards, maximum, seed):
    """Freeze a bounded original-identity cohort before synthesis or rollout."""
    pairs = [pair for shard in shards for pair in shard]
    if maximum is None:
        return population, shards
    if not 1 <= maximum <= len(pairs):
        raise ValueError('SAGE pair limit must be positive and cannot exceed available distinct pairs')
    selected = {pair['pair_id'] for pair in sorted(
        pairs, key=lambda pair: rank(pair['pair_id'], seed))[:maximum]}
    return ([row for row in population if row['source_provenance']['pair_id'] in selected],
            [[pair for pair in shard if pair['pair_id'] in selected] for shard in shards])


def prepare(args):
    from prepare_v6_vlnce import read_episodes, balanced_cases, SOURCE
    sys.path.insert(0, str(REPO / 'vln_demo/src'))
    from sage.datasets import SageSet
    from mp3d_rule_teacher import MP3DInstructionSet

    root = args.root.resolve()
    if not root.is_relative_to(ART) or root.exists():
        raise ValueError('prepare requires a new directory on the authorized artifact disk')
    if (args.shards < 1 or args.sage_per_pair not in range(1, 6)
            or args.mp3d_per_source < 1 or args.max_calls < 1):
        raise ValueError('positive population/shards required; SAGE per-pair must be 1..5')
    populations = {'sage': [], 'r2r': [], 'rxr': []}
    inputs = {d: [[] for _ in range(args.shards)] for d in populations}
    source_hashes, counts = {}, Counter()

    dataset = SageSet(args.sage_split)
    for scene in sorted(dataset.list_scenes()):
        source = Path(dataset.traj_file(scene)).resolve()
        source_hashes[str(source)] = sha256(source)
        tasks = dataset.tasks(scene)
        counts['sage_original_instruction_tasks'] += len(tasks)
        groups = {}
        for task in tasks:
            # Endpoint identities only define the upstream synthesis unit.
            # They are absent from the population and collector manifests.
            key = (task.start_label, task.goal_label) if task.start_label and task.goal_label else ('unsupported', task.task_id)
            previous = groups.get(key)
            if previous is None or rank(scene + ':' + task.task_id, args.seed) < rank(scene + ':' + previous.task_id, args.seed):
                groups[key] = task
        for key, task in sorted(groups.items()):
            pair_id = rank(scene + ':' + json.dumps(key), args.seed)
            shard = int(pair_id[:16], 16) % args.shards
            unsupported = not (task.start_label and task.goal_label)
            pair = dict(scene=scene, task=task.task_id, source=str(source), pair_id=pair_id)
            inputs['sage'][shard].append(pair)
            counts['sage_distinct_endpoint_pairs'] += 1
            counts['sage_unsupported_endpoint_pairs'] += unsupported
            for slot in range(args.sage_per_pair):
                populations['sage'].append(dict(dataset='sage', scene=scene,
                    task_id=f'{task.task_id}:generated:{slot}', source_task_id=task.task_id,
                    instruction=None, start=task.start, shard=shard,
                    status='unsupported_endpoint_identity' if unsupported else 'pending_instruction_synthesis',
                    source_provenance=dict(path=str(source), sha256=source_hashes[str(source)],
                                           pair_id=pair_id, generated_slot=slot)))
        counts['sage_available_scenes'] += 1
        # The existing reader caches dense routes for evaluation. The formal
        # manifest stores no route; release that per-scene private read cache.
        dataset._tasks.clear()

    populations['sage'], inputs['sage'] = limit_sage_pairs(
        populations['sage'], inputs['sage'], args.sage_pairs, args.seed)
    counts['sage_selected_endpoint_pairs'] = sum(len(shard) for shard in inputs['sage'])

    source_paths = {d: (SOURCE / d / 'train' /
                       ('train.json.gz' if d == 'r2r' else 'train_guide.json.gz')).resolve()
                    for d in ('r2r', 'rxr')}
    raw = {d: read_episodes(path) for d, path in source_paths.items()}
    selected, english_available = balanced_cases(raw['r2r'], raw['rxr'], args.seed)
    counts['rxr_english_available'] = english_available
    counts['r2r_available'] = len(selected['r2r'])
    if args.mp3d_per_source > len(selected['r2r']):
        raise ValueError('requested MP3D population exceeds R2R train; no replacement allowed')
    for backend in ('r2r', 'rxr'):
        source = source_paths[backend]
        source_hashes[str(source)] = sha256(source)
        native = MP3DInstructionSet(backend, source=source)
        tasks = {(scene, task.task_id): task for scene in native.scenes for task in native.tasks(scene)}
        rows = selected[backend]
        if args.mp3d_per_source < len(rows):
            rows = sorted(rows, key=lambda row: rank(str(row['scene_id']) + ':' + str(row['episode_id']), args.seed))[:args.mp3d_per_source]
        rows = sorted(rows, key=lambda row: (Path(row['scene_id']).stem, str(row['episode_id'])))
        for index, row in enumerate(rows):
            scene, episode = Path(row['scene_id']).stem, str(row['episode_id'])
            task = tasks[scene, episode]
            shard = index % args.shards
            population = dict(dataset=backend, scene=scene, task_id=episode,
                instruction=task.instruction, start=task.start,
                native_start_position=task.native_start_position,
                native_start_rotation=task.native_start_rotation,
                shard=shard, status='pending_collection',
                source_provenance=dict(path=str(source), sha256=source_hashes[str(source)]))
            populations[backend].append(population)
            inputs[backend][shard].append(dict(backend=backend, scene=scene, task=episode,
                                               instruction=task.instruction))
        counts[backend + '_selected'] = len(rows)

    code_paths = [Path(__file__), REPO / 'scripts/train/prepare_v6_vlnce.py',
        REPO / 'scripts/eval/synthesize_rule_teacher_instructions.py',
        REPO / 'scripts/eval/rule_teacher_pilot.py', REPO / 'scripts/eval/mp3d_rule_teacher_pilot.py',
        REPO / 'vln_demo/src/sage/datasets.py', REPO / 'vln_demo/src/mp3d_rule_teacher.py',
        REPO / 'vln_demo/src/mp3d_teacher_semantics.py', REPO / 'vln_demo/src/rule_teacher.py',
        REPO / 'vln_demo/src/rule_match_policy.py', REPO / 'vln_demo/src/rule_route_execution.py',
        REPO / 'verl/action_head/runtime.py',
        REPO / 'vln_demo/src/privileged_teacher.py']
    code_hashes = {str(path): sha256(path) for path in code_paths}
    root.mkdir(parents=True)
    input_hashes = {}
    for backend, population in populations.items():
        target = root / 'population' / f'{backend}.jsonl'
        save_rows(target, population)
        input_hashes[str(target)] = sha256(target)
        for shard, rows in enumerate(inputs[backend]):
            target = root / 'inputs' / f'shard_{shard:03d}' / f'{backend}_cases.json'
            save(target, dict(scope='Frozen original train identities; instruction/start only; every outcome retained', cases=rows))
            input_hashes[str(target)] = sha256(target)
    for source in code_paths:
        target = root / 'source_snapshot' / source.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    plan = dict(schema='rule_teacher_three_source_population_v1', root=str(root),
        seed=args.seed, logical_shards=args.shards, sage_split=args.sage_split,
        sage_per_pair=args.sage_per_pair, sage_pairs=args.sage_pairs,
        mp3d_per_source=args.mp3d_per_source,
        population_counts={d: len(rows) for d, rows in populations.items()}, counts=dict(counts),
        max_calls=args.max_calls, candidates=args.candidates,
        fast_calls_per_slow_anchor=4,
        grounding_policy=dict(match_policy='target_or_direction', admission='relaxed', grounding='map_open'),
        source_sha256=source_hashes, code_sha256=code_hashes, input_sha256=input_hashes,
        sampling='Original training-matched SFT request contract; 8 candidates are archived, then first rule-supported candidate is executed',
        selection='All distinct available SAGE train endpoint pairs by default, or a fixed identity-hash cohort with --sage-pairs; all R2R 10819 and fixed without-replacement English RxR 10819 by default, or --mp3d-per-source. Cohorts are frozen before synthesis/collection. No teacher outcome or terminal success filter.',
        supervision='Only original call + actual state + independent mesh/semantic/depth evidence; route and goal remain evaluator-private. Private teacher assets never enter brain requests.',
        failure_policy='Synthesis failures, overlong requests, unsupported calls and infrastructure errors remain in the frozen population/ledger; no silent resampling or replacement.',
        service_defaults=dict(sage_gym='http://127.0.0.1:18974',sage_render='http://127.0.0.1:7074',
            r2r_gym='http://127.0.0.1:18983',rxr_gym='http://127.0.0.1:18984',
            mp3d_render='http://127.0.0.1:7043',brain='http://127.0.0.1:28970',nlp='http://127.0.0.1:19074'),
        worker_layout='16 logical shards; physical workers/GPU allocation and remote image/environment remain to be validated. One mutable Gym per active collector; one brain may serve several independent collectors.',
        runtime_cache='Require --runtime-dir on the worker container local filesystem; never use the artifact disk as a writable cache.',
        validation_status='Prepared population/dispatch only; formal services, throughput, all-source acceptance and remote dry-run are not certified.')
    save(root / 'plan.json', plan)
    print(json.dumps({k: plan[k] for k in ('schema','root','population_counts','counts','validation_status')}, ensure_ascii=False, indent=2))


def verify_plan(plan):
    for section in ('source_sha256', 'code_sha256', 'input_sha256'):
        for source, digest in plan[section].items():
            if sha256(source) != digest:
                raise RuntimeError('frozen input/code changed: ' + source)


def invoke(command, log, environment):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                   env=environment, cwd=REPO, check=False)
    return dict(command=command, log=str(log), returncode=completed.returncode)


def collect(args):
    root = args.root.resolve()
    if not root.is_relative_to(ART):
        raise ValueError('root must be on the authorized artifact disk')
    plan = json.loads((root / 'plan.json').read_text())
    verify_plan(plan)
    if plan.get('fast_calls_per_slow_anchor') != 4:
        raise ValueError('prepare a new population plan with four complete fast calls per slow anchor')
    policy = plan.get('grounding_policy')
    if policy != dict(match_policy='target_or_direction', admission='relaxed', grounding='map_open'):
        raise ValueError('prepare a new population plan with the explicit relaxed target-or-direction policy')
    if not 0 <= args.shard < plan['logical_shards']:
        raise ValueError('shard outside frozen plan')
    runtime = args.runtime_dir.resolve()
    if runtime.is_relative_to(Path('/data/storage')) or runtime.is_relative_to(Path('/data/shared/groups')):
        raise ValueError('runtime cache must be worker-local, not a shared data disk')
    runtime.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    for key in ('SAGE_BENCH_PROTOCOL_ROOT','ACTION_HEAD_RULE_CREDIT','ACTION_HEAD_QUERY_TEXT_BUDGET'):
        environment.pop(key, None)
    environment['RULE_NLP_URL'] = args.nlp_url
    environment['RULE_TEACHER_SCOPE'] = 'target_direction'
    environment['RULE_TEACHER_EXACT_FOUR_FAST_CALLS'] = '1'
    environment['RULE_TEACHER_MATCH_POLICY'] = policy['match_policy']
    environment['RULE_TEACHER_ADMISSION'] = policy['admission']
    environment['RULE_TEACHER_GROUNDING'] = policy['grounding']
    for key in ('TMPDIR','XDG_CACHE_HOME','HF_HOME','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR',
                'CUDA_CACHE_PATH','PYTORCH_KERNEL_CACHE_PATH','TORCH_EXTENSIONS_DIR','MPLCONFIGDIR','NUMBA_CACHE_DIR'):
        path = runtime / key.lower()
        path.mkdir(exist_ok=True)
        environment[key] = str(path)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    backend, shard = args.backend, f'shard_{args.shard:03d}'
    source = root / 'inputs' / shard / f'{backend}_cases.json'
    rows = json.loads(source.read_text())['cases']
    output = root / 'collections' / backend / shard
    output.mkdir(parents=True, exist_ok=True)
    ledger = dict(schema='rule_teacher_three_source_shard_ledger_v1',backend=backend,shard=args.shard,
        fast_calls_per_slow_anchor=4,
        grounding_policy=policy,
        population_units=len(rows) * (plan['sage_per_pair'] if backend == 'sage' else 1),
        outcome_filter=False, status='running', synthesis=[], collector=None)
    save(output / 'population_ledger.json', ledger)
    if backend == 'sage':
        generated = []
        for index, row in enumerate(rows):
            synthesis = root / 'synthesis' / shard / f'pair_{index:06d}'
            selection = synthesis / 'selection.json'
            if not selection.exists():
                pair_input = root / 'synthesis_inputs' / shard / f'pair_{index:06d}.json'
                save(pair_input, dict(cases=[row]))
                # The official synthesis deliberately refuses an existing
                # incomplete output directory: failed units stay failed.
                if synthesis.exists():
                    result = dict(returncode=-1, error='existing incomplete synthesis; retained failure, no resampling')
                else:
                    command = [sys.executable,'-B',str(REPO / 'scripts/eval/synthesize_rule_teacher_instructions.py'),
                        '--selection',str(pair_input),'--output',str(synthesis),'--pairs','1',
                        '--per-pair',str(plan['sage_per_pair']),'--api-base',args.brain_url.rstrip('/') + '/v1']
                    result = invoke(command, root / 'logs' / shard / f'synthesis_{index:06d}.log', environment)
            else:
                result = dict(returncode=0, resumed_complete=True)
            if selection.exists() and result['returncode'] == 0:
                document = json.loads(selection.read_text())
                if document.get('complete') is not True or len(document['cases']) != plan['sage_per_pair']:
                    raise RuntimeError('completed synthesis population cardinality changed')
                generated.extend(document['cases'])
                result.update(status='generated', generated=len(document['cases']))
            else:
                result.update(status='generation_error', missing_slots=plan['sage_per_pair'])
            ledger['synthesis'].append(dict(pair_index=index,source_pair=row,**result))
            save(output / 'population_ledger.json', ledger)
            verify_plan(plan)
        save(output / 'selection.json', dict(scope='Official generated SAGE training instructions; all frozen synthesis errors retained in population_ledger.json; no terminal-success selection',cases=generated))
        command = [sys.executable,'-B',str(REPO / 'scripts/eval/rule_teacher_pilot.py'),
            '--output',str(output),'--limit',str(max(1,len(generated)))] if generated else None
    else:
        command = [sys.executable,'-B',str(REPO / 'scripts/eval/mp3d_rule_teacher_pilot.py'),
            '--backend',backend,'--split','train','--cases',str(source),'--output',str(output),
            '--limit',str(max(1,len(rows)))] if rows else None
    if command:
        defaults = plan['service_defaults']
        command += ['--gym-url',args.gym_url or defaults[backend + '_gym'],
            '--render-url',args.render_url or defaults['sage_render' if backend == 'sage' else 'mp3d_render'],
            '--brain-url',args.brain_url,'--max-calls',str(plan['max_calls']),
            '--candidates',str(plan['candidates'])]
        ledger['collector'] = invoke(command,root / 'logs' / shard / f'{backend}_collector.log',environment)
    verify_plan(plan)
    summary = output / 'summary.json'
    status = ('collector_finished' if ledger['collector']['returncode'] == 0 else 'collector_error') if command else 'no_generated_cases'
    ledger.update(status=status,
                  summary=str(summary) if summary.exists() else None)
    if summary.exists():
        ledger['collector_counts'] = {key: value for key, value in json.loads(summary.read_text()).items() if key != 'cases'}
    save(output / 'population_ledger.json',ledger)
    print(json.dumps(ledger,ensure_ascii=False,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation',required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--root',type=Path,required=True)
    prep.add_argument('--seed',type=int,default=SEED)
    prep.add_argument('--shards',type=int,default=16)
    prep.add_argument('--sage-split',default=SAGE_SPLIT)
    prep.add_argument('--sage-per-pair',type=int,default=3)
    prep.add_argument('--sage-pairs',type=int,
                      help='optional fixed distinct-pair cohort before synthesis; default all pairs')
    prep.add_argument('--mp3d-per-source',type=int,default=10819)
    prep.add_argument('--max-calls',type=int,default=200)
    prep.add_argument('--candidates',type=int,choices=(1,8),default=8)
    worker = commands.add_parser('collect-shard')
    worker.add_argument('--root',type=Path,required=True)
    worker.add_argument('--backend',choices=('sage','r2r','rxr'),required=True)
    worker.add_argument('--shard',type=int,required=True)
    worker.add_argument('--runtime-dir',type=Path,required=True)
    worker.add_argument('--gym-url')
    worker.add_argument('--render-url')
    worker.add_argument('--brain-url',default='http://127.0.0.1:28970')
    worker.add_argument('--nlp-url',default='http://127.0.0.1:19074')
    args=parser.parse_args()
    (prepare if args.operation == 'prepare' else collect)(args)


if __name__ == '__main__':
    main()
