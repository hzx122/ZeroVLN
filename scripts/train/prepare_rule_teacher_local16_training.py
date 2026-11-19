#!/usr/bin/env python3
"""Assemble reviewed full-route exports for the existing v6 local trainer.

The exporter already constructs image features and fast windows. This adapter
only validates and combines those outputs, holds out complete source tasks,
and publishes atomic slow-anchor manifests plus a scale configuration.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from sft.action_head_data_contract import (
    RULE_FULL_ROUTE_CONTRACT, validate_v6_training_row, original_task_group,
)
from sft.action_head_joint_dataset import iter_manifest_rows, load_anchor_splits
from sft.action_head_scale_config import load_scale_experiment
from sft.collect_action_head_recovery import atomic_json, atomic_jsonl, sha256_file
from sft.export_rule_teacher_training import (
    training_task_identity, original_task_identity, preceding_executions,
)
from sft.rule_teacher_progress import execution_progress

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = Path('/data/zerovln/artifacts')
SOURCES = ('sage', 'r2r', 'rxr')
FORMAL_POPULATION = {'sage': 21638, 'r2r': 10819, 'rxr': 10819}


def local_launch_plan(config, *, live_fast_visual=False):
    """Derive the local invocation from the assembled formal configuration."""
    experiment = load_scale_experiment(config)
    arm = experiment.scale('D16')
    if (experiment.action_head_as_tool_version != 'v6' or experiment.nnodes != 1
            or experiment.nproc_per_node != 16 or experiment.anchors_per_gpu not in (1, 4)):
        raise ValueError('local run requires v6, one node, sixteen ranks, APG1 or formal APG4')
    runtime = REPO.parent/'.runtime'/config.parent.name
    output = experiment.output_dir('D16')
    initialization = experiment.initialization_dir('D16')
    env = dict(REPO=str(REPO), RUN_NAME=arm.run_name, NPROC_PER_NODE='16', NNODES='1', NODE_RANK='0', MASTER_ADDR='127.0.0.1',
        CUDA_VISIBLE_DEVICES=','.join(map(str, range(16))), GATE='train',
        MANIFEST=experiment.manifest, SLOW_MODEL=experiment.slow_model,
        HEAD_CHECKPOINT=str(initialization/'head'), QUERY_CHECKPOINT=str(initialization/'query'),
        OUTPUT_DIR=str(output), TENSORBOARD_DIR=str(output/'tensorboard'),
        MAX_STEPS=str(arm.training_steps), LEARNING_RATE=str(experiment.learning_rate),
        SAVE_FREQ=str(experiment.save_freq), VAL_FREQ=str(experiment.val_freq),
        VAL_SUBSET=str(experiment.val_subset), ACTION_HEAD_TOTAL_EPOCHS=str(experiment.total_epochs),
        ACTION_HEAD_ANCHORS_PER_GPU=str(experiment.anchors_per_gpu), ACTION_HEAD_VAL_ANCHORS_PER_GPU='1',
        ACTION_HEAD_EXPERIMENT_CONFIG=str(experiment.source),
        ACTION_HEAD_EXPERIMENT_CONFIG_SHA256=experiment.sha256, ACTION_HEAD_SCALE_NAME='D16',
        ACTION_HEAD_TRAIN_PROFILE='suffix_v1', ACTION_HEAD_NOISE_SEED_POLICY='anchor_v1',
        ACTION_HEAD_FROZEN_QUERY_BACKEND='query_only', FSDP_SIZE='1',
        ACTION_HEAD_FLA_L2NORM_DYNAMIC_NB='1', ACTION_HEAD_NATIVE_PREPARE='0',
        ACTION_HEAD_NATIVE_CACHE='', ACTION_HEAD_CONDITION_MODE='', ACTION_HEAD_NATIVE_DDP='0',
        ACTION_HEAD_LR_WARMUP_STEPS='0', ACTION_HEAD_CLIP_GRAD='1.0',
        ACTION_HEAD_ANCHOR_MICROBATCH='1', ACTION_HEAD_SLOW_BATCH_MODE='sequential',
        GRADIENT_CHECKPOINTING='false', FSDP_RESHARD_AFTER_FORWARD='true',
        ACTION_HEAD_DETERMINISTIC_ALGORITHMS='0', OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=f'{REPO}/verl:{REPO}/vln_demo/src',
        LD_LIBRARY_PATH='/opt/accl-p:/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/lib:/usr/local/lib',
        LOCAL_RUNTIME_ROOT=str(runtime), ACTION_HEAD_LOCAL_RANK_CACHE_ROOT=str(runtime/'ranks'))
    if live_fast_visual:
        env['ACTION_HEAD_FAST_VISUAL_MODE'] = 'live'
        env['ACTION_HEAD_FAST_BACKBONE'] = json.loads(config.read_text())['shared'].get(
            'fast_backbone', '/opt/external/user-e7a01947/Model/Qwen3.5-4B')
    for name, folder in dict(TMPDIR='tmp', XDG_CACHE_HOME='xdg', HF_HOME='huggingface',
            TORCHINDUCTOR_CACHE_DIR='inductor', TRITON_CACHE_DIR='triton', CUDA_CACHE_PATH='cuda',
            PYTORCH_KERNEL_CACHE_PATH='kernels', TORCH_EXTENSIONS_DIR='extensions',
            RTC_CACHE_PATH='rtc', PPU_LIB_RTC_CACHE_PATH='ppu-rtc', NUMBA_CACHE_DIR='numba',
            MPLCONFIGDIR='matplotlib', DG_CACHE_DIR='deep-gemm', DG_JIT_CACHE_DIR='deep-gemm-jit').items():
        env[name] = str(runtime/folder)
    return dict(schema='rule_teacher_local16_launch_plan_v1', status='prepared_not_launched',
        experiment_config=str(experiment.source), experiment_config_sha256=experiment.sha256,
        initialize_argv=['/usr/local/bin/python3.12', '-B',
            str(REPO/'verl/sft/initialize_action_head_scale.py'),
            '--experiment-config', str(experiment.source), '--scale', 'D16'],
        train_argv=['bash', str(REPO/'scripts/train/action_head_dualvln_sft_local.sh'),
            '--action-head-as-tool-version', 'v6'], environment=env,
        initialize_log=str(config.parent/'logs/initialization.log'),
        training_log=str(config.parent/'logs/training.log'),
        global_anchor_batch=16*experiment.anchors_per_gpu, fast_calls_per_slow_anchor=4,
        frozen_frontend=('live_query_and_fast_visual' if live_fast_visual else 'live_query_only'), cleanup_runtime_after_exit=str(runtime))


def identity(row):
    return (row['source_dataset'], row['source_scene'], str(row['source_task_id']))


def group_identity(row):
    return (row['source_dataset'], row['original_task_group'])


def restore_anchor_provenance(row):
    """Legacy rows need real archived evidence, never a guessed generated-ID prefix."""
    provenance = row['teacher_execution_provenance']
    path = Path(provenance['execution_path']).resolve()
    try:
        if sha256_file(path) != provenance['execution_sha256']:
            raise ValueError('execution_hash_mismatch')
        execution = json.loads(path.read_text())
        case_path = path.parents[2]/'case.json'
        case = json.loads(case_path.read_text())
        if (provenance.get('case_sha256') is not None and
                sha256_file(case_path) != provenance['case_sha256']):
            raise ValueError('case_hash_mismatch')
        task_id, generated = training_task_identity(case, execution)
        if (str(task_id) != str(row['source_task_id']) or
                execution['scene'] != row['source_scene'] or
                case['instruction'] != row['instruction']):
            raise ValueError('archived_instruction_identity_mismatch')
        proof = original_task_identity(case, execution, row['source_dataset'])
        group = original_task_group(proof)
        if (row.get('original_task_group', group) != group or
                provenance.get('original_task_identity', proof) != proof):
            raise ValueError('archived_original_task_group_mismatch')
        progress = execution_progress(execution, preceding_executions(path))
        return dict(original_task_group=group, original_task_identity=proof,
            actual_progress=progress, source_reset_episode_id=execution['episode'],
            case_path=str(case_path), case_sha256=sha256_file(case_path)), str(case_path.parent)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f'original_task_provenance_unverified_reexport_required: {path}: {exc}') from exc


def population_coverage(population_path, construction_path):
    """Require every frozen input to have a terminal record, including failures.

    A terminal failure is coverage, not a training example. No outcome or
    per-task anchor quota is imposed on the admitted four-window exports.
    """
    population_path, construction_path = Path(population_path), Path(construction_path)
    population = json.loads(population_path.read_text())
    construction = json.loads(construction_path.read_text())
    if population.get('schema') != 'formal_rule_teacher_population_v1':
        raise ValueError('unsupported formal population schema')
    if construction.get('schema') != 'formal_rule_teacher_construction_v1':
        raise ValueError('unsupported formal construction schema')
    counts, failures = {}, []
    def read_records(record, source):
        path = Path(record['manifest'])
        if sha256_file(path) != record['manifest_sha256']:
            raise ValueError(f'{source}: coverage manifest hash mismatch')
        indexed = {}
        for row in iter_manifest_rows(str(path)):
            if row['dataset'] != source:
                raise ValueError('coverage dataset mismatch')
            key = (str(row['scene']), str(row['task_id']))
            if key in indexed:
                raise ValueError(f'{source}: duplicate population/terminal identity: {key}')
            indexed[key] = row
        return indexed
    for source in SOURCES:
        source_population = population['sources'][source]
        wanted = FORMAL_POPULATION[source]
        original = read_records(source_population, source)
        terminal = read_records(construction['sources'][source], source)
        if source_population['population_count'] != wanted or len(original) != wanted:
            raise ValueError(f'{source}: incomplete formal input population')
        if terminal.keys() != original.keys():
            raise ValueError(f'{source}: construction must cover the exact complete population')
        outcomes = Counter()
        for key, row in terminal.items():
            status = row['status']
            if status not in ('completed', 'failed', 'source_unexecutable'):
                raise ValueError(f'{source}: construction record is not terminal: {key}')
            outcomes[status] += 1
            if status != 'completed':
                if not isinstance(row.get('reason'), str) or not row['reason'].strip():
                    raise ValueError('all construction failures require a reason')
                failures.append(dict(dataset=source, scene=key[0], task_id=key[1],
                    status=status, reason=row['reason']))
        counts[source] = dict(population_count=wanted, terminal_count=len(terminal),
            coverage=1.0, outcomes=dict(outcomes),
            failure_reasons=dict(Counter(row['reason'] for row in failures if row['dataset'] == source)))
    return dict(population_report=str(population_path.resolve()),
        population_report_sha256=sha256_file(population_path),
        construction_report=str(construction_path.resolve()),
        construction_report_sha256=sha256_file(construction_path),
        sources=counts, failures=failures, complete=True)


def assemble(manifests, output, expected=30, validation_per_source=3, epochs=10, seed=20261001,
             select_from_pool=False, check_only=False, formal_population_report=None, construction_report=None, live_fast_visual=False,
             dedup_ledger=None, dedup_ledger_sha256=None,
             source_height_report=None, source_height_report_sha256=None):
    output = output.resolve()
    if not output.is_relative_to(ARTIFACTS):
        raise ValueError('output must be under the authorized data-disk artifacts root')
    if output.exists():
        raise FileExistsError(f'refusing existing output: {output}')
    from sft.rule_teacher_source_height_filter import load_source_height_filter
    height_filter = load_source_height_filter(source_height_report, source_height_report_sha256)
    formal = formal_population_report is not None
    if formal and (construction_report is None or select_from_pool or check_only or epochs != 10):
        raise ValueError('formal assembly requires construction coverage, ten epochs, and no small-pool mode')
    if formal and height_filter is None:
        raise ValueError('formal assembly requires a hash-bound SAGE source height audit')
    coverage = population_coverage(formal_population_report, construction_report) if formal else None
    from sft.rule_teacher_anchor_dedup import checked_decisions, keep_execution
    if bool(dedup_ledger) != bool(dedup_ledger_sha256):
        raise ValueError('dedup ledger and its frozen sha256 are required together')
    dedup = checked_decisions(dedup_ledger, dedup_ledger_sha256) if dedup_ledger else None
    population_members = set()
    if formal:
        population = json.loads(Path(formal_population_report).read_text())
        for source, record in population['sources'].items():
            population_members.update((source, str(row['scene']), str(row['task_id']))
                for row in iter_manifest_rows(record['manifest']))
        validation_per_source = 0
        seed = 20260908
    rows, inputs, anchors = [], [], defaultdict(list)
    for source in SOURCES:
        for manifest in manifests[source]:
            manifest = manifest.resolve()
            inputs.append(dict(source=source, manifest=str(manifest), sha256=sha256_file(manifest)))
            for original in iter_manifest_rows(str(manifest)):
                if original.get('source_dataset') not in (None, source):
                    raise ValueError(f'declared source_dataset disagrees with input source: {manifest}')
                validate_v6_training_row(original)
                if live_fast_visual and original.get('visual_input_mode') != 'live_frozen_visual_v1':
                    raise ValueError('live formal training rejects preencoded fast visual features')
                if original['data_contract'] != RULE_FULL_ROUTE_CONTRACT:
                    raise ValueError('historical truncated-route labels cannot enter this run')
                row = dict(original, source_dataset=source)
                if not Path(row['sample']).is_file():
                    raise FileNotFoundError(row['sample'])
                rows.append(row)
                anchors[row['slow_anchor_id']].append(row)
    episodes, original_groups = defaultdict(list), defaultdict(list)
    instruction_by_identity = {}
    exclusions = []
    for anchor, windows in anchors.items():
        first = windows[0]
        if any(identity(row) != identity(first) or row['instruction'] != first['instruction']
               for row in windows):
            raise ValueError(f'mixed source identity in anchor {anchor}')
        indices = sorted(row['fast_window_index'] for row in windows)
        chunks = first['teacher_execution_provenance']['route_chunks']
        if chunks != 4 or any(row['teacher_execution_provenance']['route_chunks'] != 4 for row in windows):
            raise ValueError(f'formal slow anchor requires exactly four complete fast calls: {anchor}')
        if indices != list(range(chunks)):
            raise ValueError(f'duplicate or missing full-route windows: {anchor}')
        previous_instruction = instruction_by_identity.setdefault(identity(first), first['instruction'])
        if previous_instruction != first['instruction']:
            raise ValueError(f'multiple instructions share one source task; assign unique generated task identities: {identity(first)}')
        restored, case_dir = restore_anchor_provenance(first)
        if formal and (first['source_dataset'], str(first['source_scene']),
                str(restored['source_reset_episode_id'])) not in population_members:
            raise ValueError('training anchor is outside the frozen formal input population')
        for row in windows:
            if row['teacher_execution_provenance'] != first['teacher_execution_provenance']:
                raise ValueError(f'mixed execution provenance in anchor {anchor}')
        for row in windows:
            row['original_task_group'] = restored['original_task_group']
            row['teacher_execution_provenance'].update({key: value for key, value in restored.items()
                                                      if key != 'original_task_group'})
            validate_v6_training_row(row)
        group = group_identity(first)
        # A retry is a separate instruction episode, even if its wording ID is the same.
        episode = (*identity(first), case_dir)
        original_groups[group].append(windows)
        if height_filter is not None:
            reason = height_filter.rejection(
                first['teacher_execution_provenance']['execution_path'],
                first['teacher_execution_provenance']['execution_sha256'], first['source_dataset'])
            if reason:
                exclusions.append(dict(slow_anchor_id=anchor, original_task_group=list(group),
                    reason=reason, fast_windows_removed=len(windows)))
                continue
        if dedup is not None and not keep_execution(
                first['teacher_execution_provenance']['execution_path'],
                first['teacher_execution_provenance']['execution_sha256'], dedup):
            exclusions.append(dict(slow_anchor_id=anchor, original_task_group=list(group),
                reason='redundant_middle_of_exact_hold_run', fast_windows_removed=len(windows)))
            continue
        if not restored['actual_progress']['admissible_geometry']:
            exclusions.append(dict(slow_anchor_id=anchor, original_task_group=list(group),
                reason=restored['actual_progress']['reason']))
            continue
        episodes[episode].append(windows)
    by_source = {source: sorted(key for key in original_groups if key[0] == source) for source in SOURCES}
    pool_counts = {source: len(keys) for source, keys in by_source.items()}
    eligible_counts = {}
    group_episodes = defaultdict(list)
    for episode, calls in episodes.items():
        if len({group[0]['instruction'] for group in calls}) != 1:
            raise ValueError(f'multiple instructions share one source task; assign unique generated task identities: {episode}')
        group_episodes[group_identity(calls[0][0])].append(episode)
    for source, keys in by_source.items():
        if not (formal or select_from_pool or check_only) and len(keys) != expected:
            raise ValueError(f'{source}: expected {expected} independent original task groups, got {len(keys)}')
        eligible = []
        for key in keys:
            calls = original_groups[key]
            anchor_counts = [len({
                (group[0]['teacher_execution_provenance']['condition_sha256'],
                 group[0]['teacher_execution_provenance']['execution_sha256'])
                for group in episodes[episode]}) for episode in group_episodes[key]]
            anchor_count = max(anchor_counts, default=0)
            if anchor_count < (1 if formal else 2):
                if not (formal or select_from_pool or check_only):
                    raise ValueError(f'episode lacks two reviewed distinct executed calls: {key}')
                exclusions.append(dict(original_task_group=list(key),
                    reason=('no_admissible_complete_anchor' if formal else 'fewer_than_two_reviewed_executed_anchors'),
                    reviewed_anchors=len(calls), best_instruction_episode_reviewed_anchors=anchor_count))
                continue
            eligible.append(key)
        eligible_counts[source] = len(eligible)
        if formal or check_only:
            by_source[source] = eligible
            continue
        if select_from_pool:
            if len(eligible) < expected:
                raise ValueError(f'{source}: expected {expected} eligible original task groups, '
                                 f'got {len(eligible)} from pool of {len(keys)}')
            ordered = sorted(eligible, key=lambda key: hashlib.sha256(
                json.dumps([seed, 'cohort', *key], ensure_ascii=False).encode()).hexdigest())
            by_source[source] = sorted(ordered[:expected])
            exclusions.extend(dict(original_task_group=list(key), reason='eligible_pool_task_not_selected',
                reviewed_anchors=len(original_groups[key])) for key in ordered[expected:])
    if check_only:
        report = dict(schema='reviewed_original_task_admission_check_v1', status='checked_not_launched',
            inputs=inputs, source_height_audit=height_filter.binding if height_filter else None,
            expected_original_tasks_per_source=expected,
            input_original_task_groups=pool_counts, eligible_original_task_groups=eligible_counts,
            deficit_original_task_groups={source: max(0, expected - eligible_counts[source]) for source in SOURCES},
            quota_met=all(eligible_counts[source] >= expected for source in SOURCES),
            reviewed_complete_anchors=len(anchors),
            admitted_geometry_anchors=sum(len(calls) for calls in episodes.values()),
            eligible_groups={source: [key[1] for key in keys] for source, keys in by_source.items()},
            exclusions=exclusions,
            task_quota_unit='original_task_group_with_one_instruction_episode_having_two_distinct_reviewed_executed_anchors_including_holds',
            split_policy='whole_original_task_group_holdout_all_instruction_variants',
            formal_training_started=False, feature_reencoding=False)
        output.mkdir(parents=True)
        atomic_json(output/'admission-check.json', report)
        return report
    selected = {key for keys in by_source.values() for key in keys}
    admitted_anchors = {windows[0]['slow_anchor_id'] for calls in episodes.values() for windows in calls}
    rows = [row for row in rows if group_identity(row) in selected and
            row['slow_anchor_id'] in admitted_anchors]
    validation = set()
    for source, keys in by_source.items():
        ordered = sorted(keys, key=lambda key: hashlib.sha256(
            json.dumps([seed, *key], ensure_ascii=False).encode()).hexdigest())
        validation.update(ordered[:validation_per_source])
    for row in rows:
        row['split'] = 'validation' if group_identity(row) in validation else 'train'
    rows.sort(key=lambda row: (identity(row), row['option_sequence_index'], row['fast_window_index']))
    output.mkdir(parents=True)
    manifest = output/'manifest.jsonl'
    atomic_jsonl(manifest, rows)
    groups = load_anchor_splits(str(manifest), lambda row, index: validate_v6_training_row(row))
    shard_rows = [[] for _ in range(16)]
    for index, group in enumerate(groups['train']):
        shard_rows[index % 16].extend(group)
    for rank, shard in enumerate(shard_rows):
        atomic_jsonl(output/'shards'/f'train_{rank:02d}.jsonl', shard)
    atomic_jsonl(output/'validation.jsonl', [row for row in rows if row['split'] == 'validation'])
    counts = {source: dict(episodes=len(by_source[source]),
        training_episodes=sum(key not in validation for key in by_source[source]),
        validation_episodes=sum(key in validation for key in by_source[source]),
        original_task_groups=len(by_source[source]),
        instruction_episodes=sum(len(group_episodes[key]) for key in by_source[source]),
        calls=len({row['slow_anchor_id'] for row in rows if row['source_dataset'] == source}),
        windows=sum(row['source_dataset'] == source for row in rows))
        for source in SOURCES}
    if not groups['train']:
        raise ValueError('no qualified complete slow anchors for training')
    global_batch = 64 if formal else 16
    steps_per_epoch = math.ceil(len(groups['train'])/global_batch)
    report = dict(schema='reviewed_full_route_three_source_local16_v1',
        inputs=inputs, counts=counts, seed=seed, manifest=str(manifest),
        manifest_sha256=sha256_file(manifest), windows=len(rows),
        anchors={split: len(items) for split, items in groups.items()},
        epochs=epochs, steps_per_epoch=steps_per_epoch, max_steps=steps_per_epoch*epochs,
        formal_population_mode=formal, population_coverage=coverage,
        fast_visual_input_mode=('live_frozen_visual_v1' if live_fast_visual else 'preencoded'),
        global_anchor_batch=global_batch,
        split_policy='whole_original_task_group_holdout_all_instruction_variants',
        task_quota_unit='original_task_group_with_one_instruction_episode_having_two_distinct_reviewed_executed_anchors_including_holds',
        sharding_unit='complete_slow_anchor_all_fast_windows',
        logical_training_shards=16,
        fast_calls_per_slow_anchor=4,
        runtime_sampling='existing PaddedShardSampler reshuffles complete anchors each epoch; invalid tail slots have zero weight',
        shard_windows=[len(items) for items in shard_rows],
        data_contracts=dict(Counter(row['data_contract'] for row in rows)),
        pool_selection=dict(enabled=select_from_pool, input_episodes=pool_counts,
            eligible_episodes=eligible_counts, count_unit='original_task_group',
            input_original_task_groups=pool_counts, eligible_original_task_groups=eligible_counts,
            exclusions=exclusions,
            selection_policy='seeded_original_task_group_hash_no_outcome_score'),
        full_episode_success_filter=False, assistant_review_is_human_verification=False,
        source_height_audit=height_filter.binding if height_filter else None,
        dedup_ledger=(dict(path=str(Path(dedup_ledger).resolve()), sha256=dedup_ledger_sha256)
                      if dedup_ledger else None))
    if formal:
        report['task_quota_unit'] = 'all_population_inputs_accounted_actual_qualified_four_window_anchors_only'
        report['split_policy'] = 'all_qualified_anchors_train_v6_validation_disabled'
        atomic_json(output/'construction-failures.json', coverage['failures'])
    atomic_json(output/'assembly.json', report)
    base = json.loads((REPO/'configs/action_head/v6_official_gvln_d16_beam4_10epochs.json').read_text())
    training = output.parent
    base['shared'].update(manifest=str(manifest), output_root=str(training/'checkpoints'),
        validation_manifest=str(output/'validation.jsonl'))
    base['initialization'].update(output_root=str(training/'initialization'))
    if not formal:
        base['initialization'].update(seed=seed, extra_query_seed=seed)
    base['training'].update(nnodes=1, nproc_per_node=16, total_epochs=epochs,
        anchors_per_gpu=4 if formal else 1,
        save_freq=1000 if formal else steps_per_epoch, val_freq=0 if formal else steps_per_epoch, val_subset=0)
    base['scales']['D16'].update(max_steps=report['max_steps'], run_name='v6-three-source-full-route-d16-local16')
    atomic_json(training/'experiment_config.json', base)
    atomic_json(training/'local_launch_plan.json', local_launch_plan(training/'experiment_config.json', live_fast_visual=live_fast_visual))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for source in SOURCES:
        parser.add_argument('--'+source, type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--formal-population-report', type=Path)
    parser.add_argument('--construction-report', type=Path)
    parser.add_argument('--source-height-report', type=Path)
    parser.add_argument('--source-height-report-sha256')
    parser.add_argument('--dedup-ledger', type=Path)
    parser.add_argument('--dedup-ledger-sha256')
    parser.add_argument('--live-fast-visual', action='store_true', help='require image-only fast inputs and encode with the same frozen tower during training')
    parser.add_argument('--expected', type=int, default=30)
    parser.add_argument('--validation-per-source', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--select-from-pool', action='store_true',
                        help='select complete continuous tasks from a larger reviewed pool; record exclusions')
    parser.add_argument('--check-only', action='store_true',
                        help='write admission counts and quota deficits without assembling or launching training')
    args = parser.parse_args()
    if not args.formal_population_report and (args.expected < 2 or not 0 < args.validation_per_source < args.expected or args.epochs < 1):
        parser.error('positive epochs and a smaller positive whole-episode validation set required')
    if not args.check_only and any(not getattr(args, source) for source in SOURCES):
        parser.error('normal assembly requires at least one manifest for every source')
    print(json.dumps(assemble({source: getattr(args, source) for source in SOURCES},
        args.output, args.expected, args.validation_per_source, args.epochs,
        select_from_pool=args.select_from_pool, check_only=args.check_only,
        formal_population_report=args.formal_population_report, construction_report=args.construction_report, live_fast_visual=args.live_fast_visual,
        dedup_ledger=args.dedup_ledger, dedup_ledger_sha256=args.dedup_ledger_sha256,
        source_height_report=args.source_height_report,
        source_height_report_sha256=args.source_height_report_sha256), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
