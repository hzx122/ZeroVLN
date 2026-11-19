"""Prepare a separate bounded retry/continuation of a frozen collection.

Reuse the pilots and owned-service supervisor. Old cases are never overwritten
or copied into a new attempt. Normal navigation failures are retained as final;
only unfinished units and one previous input-context failure are retried. An
explicit, single infrastructure recovery can replace proven empty generations.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys

from prepare_rule_teacher_three_source import ART, REPO, save, sha256, verify_plan
from prepare_v6_vlnce import SOURCE as NATIVE_DATASETS
from supervise_rule_teacher_collection import collection_status
from prepare_rule_teacher_local16_collection import option


def context_failure(result):
    error = str(result.get('infra_error', ''))
    return (result.get('why') == 'model_error' and 'Input length (' in error
            and "exceeds model's maximum context length" in error)


def empty_generation_evidence(case, result):
    """Recognize this incident from complete terminal evidence, never its label alone."""
    if result.get('why') != 'rule_abstention':
        return None
    decisions = sorted((path for path in (case / 'candidates').glob('decision_*')
                        if path.is_dir() and path.name.removeprefix('decision_').isdigit()),
                       key=lambda path: int(path.name.removeprefix('decision_')))
    if not decisions:
        return None
    terminal = decisions[-1]
    selection_path = terminal / 'selection.json'
    if not selection_path.is_file():
        return None
    selection = json.loads(selection_path.read_text())
    if (selection.get('sampled') != 8 or selection.get('all_rejected') is not True
            or 'selected_index' not in selection or selection['selected_index'] is not None):
        return None
    candidates = [terminal / f'candidate_{index:02d}.json' for index in range(8)]
    actual = {path.name for path in terminal.glob('candidate_*.json')
              if path.stem.removeprefix('candidate_').isdigit()}
    if actual != {path.name for path in candidates}:
        return None
    for index, path in enumerate(candidates):
        row = json.loads(path.read_text())
        usage = row.get('usage')
        tokens = usage.get('completion_tokens') if isinstance(usage, dict) else None
        if (type(row.get('index')) is not int or row['index'] != index
                or row.get('original_text') != '' or row.get('calls') != []
                or type(tokens) is not int or tokens != 0 or row.get('selected') is not False):
            return None
    files = [case / 'result.json', selection_path, *candidates]
    for optional in (case / 'case.json', terminal / 'context_budget.json'):
        if optional.is_file():
            files.append(optional)
    return dict(kind='terminal_eight_empty_zero_token_candidates',
                terminal_decision=str(terminal), candidate_count=8,
                evidence_sha256={str(path): sha256(path) for path in files})


def population(previous_output, *, recover_empty_generation=False, continue_unstarted=False):
    selection = json.loads((previous_output / 'selection.json').read_text())
    pending, ledger = [], []
    for index, row in enumerate(selection['cases']):
        previous_case = previous_output / f'case_{index:03d}'
        result_path = previous_case / 'result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        evidence = (empty_generation_evidence(previous_case, result)
                    if recover_empty_generation and result is not None else None)
        if continue_unstarted:
            status = ('retained_previous_result' if result is not None else
                      'quarantined_started_without_result' if previous_case.exists() else 'unstarted')
        elif result is None:
            status = 'interrupted_retry' if previous_case.exists() else 'unstarted'
        elif evidence is not None:
            status = 'empty_generation_retry_once'
        elif not recover_empty_generation and context_failure(result):
            status = 'context_budget_retry_once'
        else:
            status = 'retained_previous_result'
        entry = dict(previous_index=index, previous_case=str(previous_case),
                     scene=row['scene'], task=str(row['task']), status=status,
                     previous_result_sha256=sha256(result_path) if result is not None else None)
        if evidence is not None:
            entry['infrastructure_recovery'] = evidence
        if status not in ('retained_previous_result', 'quarantined_started_without_result'):
            entry['new_index'] = len(pending)
            pending.append(row)
        ledger.append(entry)
    return dict(selection, cases=pending), ledger


def continuation_lineage(previous_plan, backend, output, selection, ledger):
    """Carry frozen r2 ancestry and counters without spending another retry."""
    mapping_path = Path(previous_plan['attempt_mapping']) / (backend + '.json')
    if previous_plan['input_sha256'].get(str(mapping_path)) != sha256(mapping_path):
        raise ValueError('previous attempt mapping is not frozen')
    mapping = json.loads(mapping_path.read_text())
    parent_selection = Path(mapping['previous_selection'])
    if (mapping['previous_selection_sha256'] != sha256(parent_selection)
            or previous_plan['input_sha256'].get(str(parent_selection)) != sha256(parent_selection)):
        raise ValueError('previous selection is not frozen')
    parent_rows = json.loads(parent_selection.read_text())['cases']
    current_rows = json.loads((output / 'selection.json').read_text())['cases']
    ancestors = [row for row in mapping['units'] if 'new_index' in row]
    indexed = {row['new_index']: row for row in ancestors}
    if len(indexed) != len(ancestors) or set(indexed) != set(range(len(current_rows))):
        raise ValueError('previous attempt lineage indices are incomplete or duplicated')
    original = previous_plan['original_populations'][backend]
    if original != mapping['original_population']:
        raise ValueError('previous attempt changed original population accounting')
    successful = {}
    # Every older mapping and its result evidence is already frozen by the
    # previous plan. A successful original task excludes all wording variants.
    history = [(ledger, current_rows)]
    for path in previous_plan['input_sha256']:
        candidate = Path(path)
        if candidate.name == backend + '.json' and candidate.parent.name == 'attempts':
            older = json.loads(candidate.read_text())
            frozen_selection = Path(older['previous_selection'])
            if previous_plan['input_sha256'].get(str(frozen_selection)) != sha256(frozen_selection):
                raise ValueError('historical success selection is not frozen')
            history.append((older['units'], json.loads(frozen_selection.read_text())['cases']))
    for units, rows in history:
        for unit in units:
            result_path = Path(unit['previous_case']) / 'result.json'
            expected = unit.get('previous_result_sha256')
            if expected is not None:
                if sha256(result_path) != expected:
                    raise ValueError('previous result changed during continuation')
                result = json.loads(result_path.read_text())
                if result.get('success') is True:
                    group, identity = continuation_identity(backend, rows[unit['previous_index']])
                    successful.setdefault(group, []).append(dict(original_task_identity=identity,
                        result_path=str(result_path), result_sha256=expected))
    pending = []
    for entry in ledger:
        ancestor = indexed[entry['previous_index']]
        row = current_rows[entry['previous_index']]
        if row != parent_rows[ancestor['previous_index']]:
            raise ValueError('previous selection differs from its frozen lineage')
        required = ('original_index', 'original_case', 'original_result_sha256',
                    'context_retries_used', 'empty_generation_recoveries_used')
        if any(key not in ancestor for key in required):
            raise ValueError('continuation requires explicit original lineage and retry counters')
        entry.update({key: ancestor[key] for key in required})
        entry['previous_attempt_status'] = ancestor['status']
        group, identity = continuation_identity(backend, row)
        entry.update(original_task_group=group, original_task_identity=identity)
        entry.pop('new_index', None)
        if entry['status'] == 'unstarted':
            if group in successful:
                entry['status'] = 'excluded_previously_successful_original_task'
                entry['previous_success_evidence'] = successful[group]
            else:
                entry['new_index'] = len(pending)
                pending.append(row)
        # Preserve executed prefixes by reference and freeze their machine
        # evidence. No image data or case files are copied into the new output.
        if Path(entry['previous_case']).exists():
            entry['preserved_evidence_sha256'] = {
                str(path): sha256(path) for path in Path(entry['previous_case']).rglob('*.json')}
    selection['cases'] = pending
    return original


def continuation_identity(backend, row):
    """Reuse the training/admission identity, including verified SAGE endpoints."""
    repo = Path(__file__).resolve().parents[2]
    for path in (repo / 'verl', repo / 'vln_demo/src'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from sft.export_rule_teacher_training import original_task_identity
    from sft.action_head_data_contract import original_task_group
    identity = original_task_identity(dict(row, backend=backend),
        dict(scene=row['scene'], episode=str(row['task'])), backend)
    return original_task_group(identity), identity


def recovery_lineage(previous_plan, original_plan, backend, output, ledger):
    """Join by frozen indices, preserving SAGE wording variants and retry budgets."""
    mapping_path = Path(previous_plan['attempt_mapping']) / (backend + '.json')
    if previous_plan['input_sha256'].get(str(mapping_path)) != sha256(mapping_path):
        raise ValueError('previous attempt mapping is not frozen')
    mapping = json.loads(mapping_path.read_text())
    previous_rows = json.loads((output / 'selection.json').read_text())['cases']
    original_selection = Path(mapping['previous_selection'])
    if (mapping['previous_selection_sha256'] != sha256(original_selection)
            or previous_plan['input_sha256'].get(str(original_selection)) != sha256(original_selection)):
        raise ValueError('original selection is not frozen in the previous attempt')
    original_rows = json.loads(original_selection.read_text())['cases']
    ancestors = [row for row in mapping['units'] if 'new_index' in row]
    by_index = {row['new_index']: row for row in ancestors}
    if len(by_index) != len(ancestors) or set(by_index) != set(range(len(previous_rows))):
        raise ValueError('previous attempt lineage indices are incomplete or duplicated')
    original = original_population(original_plan, backend, original_selection.parent, len(mapping['units']))
    if original != mapping['original_population'] or original != previous_plan['original_populations'][backend]:
        raise ValueError('previous attempt changed the original population accounting')
    for entry in ledger:
        ancestor = by_index[entry['previous_index']]
        original_index = ancestor['previous_index']
        if previous_rows[entry['previous_index']] != original_rows[original_index]:
            raise ValueError('previous attempt selection differs from its original lineage')
        entry.update(original_index=original_index, original_case=ancestor['previous_case'],
                     original_result_sha256=ancestor['previous_result_sha256'],
                     previous_attempt_status=ancestor['status'],
                     context_retries_used=int(ancestor['status'] == 'context_budget_retry_once'),
                     empty_generation_recoveries_used=int(entry['status'] == 'empty_generation_retry_once'))
    return original


def previous_output(command):
    def argument(name):
        return command[command.index(name) + 1]
    if '--output' in command:
        return Path(argument('--output'))
    return Path(argument('--root')) / 'collections' / argument('--backend') / ('shard_' + argument('--shard').zfill(3))


def original_population(plan, backend, output, selection_units):
    """Account for the frozen pool, including instructions that failed synthesis."""
    source = Path(plan['root']) / 'inputs' / output.name / (backend + '_cases.json')
    expected = plan['input_sha256'].get(str(source))
    if expected is None or sha256(source) != expected:
        raise ValueError('original population input is not frozen: ' + str(source))
    rows = json.loads(source.read_text())['cases']
    units = len(rows) * (plan['sage_per_pair'] if backend == 'sage' else 1)
    ledger_path = output / 'population_ledger.json'
    if not ledger_path.exists():
        ledger_path = Path(plan['root']) / 'collections' / backend / output.name / 'population_ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else None
    if ledger is not None and (ledger['backend'] != backend or ledger['population_units'] != units):
        raise ValueError('original population ledger disagrees with frozen input')
    generated_pairs = failed_pairs = generated_units = failed_units = 0
    if backend == 'sage':
        if ledger is None:
            raise ValueError('SAGE original synthesis population ledger is required')
        synthesis = ledger['synthesis']
        if len(synthesis) != len(rows):
            raise ValueError('SAGE original synthesis population is incomplete')
        for index, (record, row) in enumerate(zip(synthesis, rows)):
            if record['pair_index'] != index or record['source_pair'] != row:
                raise ValueError('SAGE synthesis identity differs from frozen original pair')
            if record['status'] == 'generated' and record['generated'] == plan['sage_per_pair']:
                generated_pairs += 1
                generated_units += record['generated']
            elif record['status'] == 'generation_error' and record['missing_slots'] == plan['sage_per_pair']:
                failed_pairs += 1
                failed_units += record['missing_slots']
            else:
                raise ValueError('SAGE original synthesis disposition is incomplete')
        if generated_units != selection_units or generated_units + failed_units != units:
            raise ValueError('SAGE generated selection and synthesis population disagree')
    elif selection_units != units:
        raise ValueError('native selection differs from the complete original population')
    return dict(frozen_population_input=str(source), frozen_population_input_sha256=expected,
        cohort_population_units=plan['population_counts'][backend], original_population_units=units,
        selection_units=selection_units, generated_synthesis_pairs=generated_pairs,
        generated_synthesis_units=generated_units, failed_synthesis_pairs=failed_pairs,
        failed_synthesis_units=failed_units,
        population_ledger=str(ledger_path) if ledger is not None else None,
        population_ledger_sha256=sha256(ledger_path) if ledger is not None else None,
        population_ledger_status='frozen_existing' if ledger is not None else 'direct_pilot_no_population_ledger',
        synthesis_retry=False)


NATIVE_DISTRIBUTED_ENV_KEYS = (
    'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR',
    'MASTER_PORT', 'GROUP_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE',
)


def native_cpu_command(command):
    """Apply the existing run_v6_vlnce_dlc.sh standalone Gym environment rule."""
    return ['env', *[part for key in NATIVE_DISTRIBUTED_ENV_KEYS for part in ('-u', key)], *command]


def native_cpu_environment():
    """Explicit settings from the native Gym and remote_habitat launchers."""
    mesa = ART.parent / 'runtime_inputs/habitat-mesa-25.2.8-ubuntu24.04-amd64'
    return dict(CUDA_VISIBLE_DEVICES='', EGL_PLATFORM='surfaceless',
        LIBGL_ALWAYS_SOFTWARE='1', MESA_LOADER_DRIVER_OVERRIDE='llvmpipe',
        GALLIUM_DRIVER='llvmpipe', HABITAT_SIM_GPU_DEVICE_ID='-1',
        LP_NUM_THREADS='4', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', MAGNUM_LOG='quiet', HABITAT_SIM_LOG='quiet',
        LIBGL_DRIVERS_PATH=str(mesa / 'dri'),
        __EGL_VENDOR_LIBRARY_FILENAMES=str(mesa / '50_mesa.json'),
        RYNNBRAIN_NAV_DIR=str(NATIVE_DATASETS.parent.parent),
        RYNNVLN_LANGUAGE='en', RYNNVLN_TURN_ANGLE='30', RYNNVLN_HFOV='120',
        RYNNVLN_ALLOW_SLIDING='1', MODEL_PATH='/dummy/vln_gym_no_vllm')


def configure_depth_batch(launch, sources, root, *, size, port_base):
    """Allocate independent Gym actors; retain each source's renderer lease.

    Reuse the SAGE service command and the native vlnce_action_head entrypoint.
    No previously running Gym is reset or repurposed by this batch.
    """
    if not 2 <= size <= 8:
        raise ValueError('depth batch size must be in 2..8')
    if not 1024 <= port_base <= 65535 - size * len(sources):
        raise ValueError('invalid Gym port range')
    sage_template = next((copy.deepcopy(entry) for entry in launch['services']
                          if entry['name'] == 'sage-gym'), None)
    if 'sage' in sources and sage_template is None:
        raise ValueError('depth batching requires the existing SAGE Gym template')
    launch['services'] = [entry for entry in launch['services']
                          if entry['name'] != 'sage-gym']
    existing_ports = {entry['port'] for entry in launch['services']}
    allocated, bindings = set(), {}
    for source_index, collector in enumerate(launch['collectors']):
        backend = collector['name'].removesuffix('-collector')
        rows = sources[backend]['selection']['cases']
        command = collector['command']
        # The frozen pool may be larger than a bounded validation run.
        if '--limit' in command:
            rows = rows[:int(command[command.index('--limit') + 1])]
        render_url = command[command.index('--render-url') + 1]
        python = command[0]
        urls = []
        slots = min(size, max(Counter(row['scene'] for row in rows).values()))
        for slot in range(slots):
            port = port_base + source_index * size + slot
            if port in existing_ports or port in allocated:
                raise ValueError('Gym actor port overlaps a planned service')
            allocated.add(port)
            name = f'{backend}-gym-{slot:02d}'
            row = rows[0]
            environment = copy.deepcopy(collector.get('environment', {}))
            environment.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            if backend == 'sage':
                entry = copy.deepcopy(sage_template)
                actor_command = entry['command']
                for flag, value in (('--port', str(port)), ('--scene', row['scene']),
                                    ('--episode_id', row['task'])):
                    actor_command[actor_command.index(flag) + 1] = str(value)
                entry['environment'].update(environment, SAGE_RENDER_URL=render_url)
            else:
                environment.update(native_cpu_environment())
                environment.update(VLN_HABITAT_RENDER_URL=render_url,
                    VLN_RENDER_MODE='batch', VLN_RENDER_TIMEOUT='600')
                entry = dict(command=[python, '-B', '-u', '-m', 'vln_gym.vlnce_action_head',
                    '--backend', backend, '--port', str(port), '--split', 'train',
                    '--max_steps', '0', '--output_path', str(root / 'services' / name),
                    '--scene', row['scene'].removeprefix('rxr-'),
                    '--episode_id', str(row['task'])], environment=environment)
                # Match vln_gym_server.sh / run_v6_vlnce_dlc.sh: Hydra's
                # config and relative data paths belong to each Gym's cwd.
                config = REPO / 'vln_demo/src/vln_demo.yaml'
                entry['native_gym_workspace'] = dict(
                    cwd=str(Path(launch['runtime']) / name / 'work'),
                    config_source=str(config), config_sha256=sha256(config),
                    data_target=str(NATIVE_DATASETS.parent.resolve()))
            entry.update(name=name, port=port, log=str(root / (name + '.log')),
                         health_url=f'http://127.0.0.1:{port}/health')
            launch['services'].append(entry)
            urls.append(f'http://127.0.0.1:{port}')
        command[command.index('--gym-url') + 1] = urls[0]
        for key, value in {'--gym-urls': json.dumps(urls), '--depth-batch-size': size,
                '--candidates': 8, '--candidate-strategy': 'fixed',
                '--sampling-scheduler': 'v6_depth'}.items():
            option(command, key, value)
        for flag in ('--rolling-actors', '--rolling-scene-refill'):
            if flag in command: command.remove(flag)
        collector.setdefault('environment', {}).update(CUDA_VISIBLE_DEVICES='',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        bindings[backend] = dict(gym_urls=urls, renderer=render_url,
            independent_episode_states=True, cohort_order='same_scene_original_order',
            actor_count=slots)
    # Native rendering is software EGL, matching the already verified R2R
    # renderer. It must not initialize the GPUs reserved for the brain.
    for entry in launch['services']:
        if entry['name'] == 'rxr-render':
            entry.setdefault('environment', {}).update(native_cpu_environment())
    return dict(schema='rule_teacher_depth_batch_v1', batch_size=size,
                scheduling='v6_same_depth_batch',
                per_case_order_preserved=True, whole_depth_barrier=True,
                candidate_strategy='fixed', candidate_waves=[8, 8], rejected_depth_retries=1,
                maximum_candidates=16, bindings=bindings,
                owned_gym_ports=sorted(allocated))


def prepare(previous_launch, root, *, preview=False, recover_empty_generation=False,
            continue_unstarted=False, brain_url=None, depth_batch_size=1,
            gym_port_base=19100, validation_limit_per_source=0):
    if not 1 <= depth_batch_size <= 8:
        raise ValueError('depth batch size must be in 1..8')
    if type(validation_limit_per_source) is not int or validation_limit_per_source < 0:
        raise ValueError('validation limit must be a nonnegative integer')
    old = json.loads(previous_launch.read_text())
    previous_plan = json.loads(Path(old['source_plan']).read_text())
    if continue_unstarted and recover_empty_generation:
        raise ValueError('continuation cannot request any recovery')
    if continue_unstarted and (not previous_plan.get('empty_generation_recovery')
                               or previous_plan.get('unstarted_continuation')):
        raise ValueError('unstarted continuation is one explicit transition after empty-generation recovery')
    if previous_plan.get('empty_generation_recovery') and not continue_unstarted:
        raise ValueError('empty-generation recovery is bounded to one transition; cannot recover its output')
    if recover_empty_generation and not previous_plan.get('attempt_policy'):
        raise ValueError('empty-generation recovery requires the first isolated attempt')
    if previous_plan.get('attempt_policy') and not (recover_empty_generation or continue_unstarted):
        raise ValueError('automatic retries are bounded to one transition from the original cohort')
    verify_plan(previous_plan)
    original_plan = (json.loads(Path(previous_plan['previous_cohort_plan']).read_text())
                     if recover_empty_generation or continue_unstarted else previous_plan)
    if original_plan.get('attempt_policy'):
        raise ValueError('recovery must retain a direct lineage to the original cohort')
    if recover_empty_generation or continue_unstarted:
        verify_plan(original_plan)
    sources = {}
    for collector in old['collectors']:
        backend = collector['name'].removesuffix('-collector')
        source = previous_output(collector['command'])
        selection, ledger = population(source, recover_empty_generation=recover_empty_generation,
                                       continue_unstarted=continue_unstarted)
        original = (continuation_lineage(previous_plan, backend, source, selection, ledger)
                    if continue_unstarted else recovery_lineage(previous_plan, original_plan, backend, source, ledger)
                    if recover_empty_generation else original_population(original_plan, backend, source, len(ledger)))
        sources[backend] = dict(source=source, selection=selection, ledger=ledger,
                                collector=collector, original_population=original)
    counts = {backend: dict(Counter(row['status'] for row in source['ledger']))
              for backend, source in sources.items()}
    validation_scope = None
    if validation_limit_per_source:
        available = {backend: len(source['selection']['cases']) for backend, source in sources.items()}
        scheduled = {backend: min(validation_limit_per_source, count) for backend, count in available.items()}
        validation_scope = dict(purpose='pre-production throughput and complete-trajectory acceptance',
            limit_per_source=validation_limit_per_source, selection='original_order_prefix',
            available_units=available, scheduled_units=scheduled,
            deferred_units={backend: count - scheduled[backend] for backend, count in available.items()},
            failures_retained=True, original_max_calls_preserved=True,
            formal_collection_requires_user_acceptance=True)
    if preview:
        return dict(preview=True, counts=counts, mutates_previous=False,
                    validation_scope=validation_scope,
                    original_populations={backend: item['original_population'] for backend, item in sources.items()})
    if not (previous_launch.parent / 'completion.json').exists():
        raise ValueError('stop the verified owned supervisor and wait for its completion receipt first')
    status = collection_status(previous_launch)
    if status['supervisor']['live'] or any(row['live'] for row in status['processes']):
        raise ValueError('previous supervisor or an owned process is still live')
    root = root.resolve()
    if not root.is_relative_to(ART) or root.exists():
        raise ValueError('requires a fresh directory on the artifact disk')
    root.mkdir(parents=True)
    previous_repo = str(Path(old['collectors'][0]['command'][2]).parents[2])

    def rebase(value):
        if isinstance(value, str):
            return value.replace(previous_repo, str(REPO))
        if isinstance(value, list):
            return [rebase(item) for item in value]
        if isinstance(value, dict):
            return {key: rebase(item) for key, item in value.items()}
        return value

    launch = rebase(copy.deepcopy(old))
    launch['runtime'] = str(REPO.parent / '.runtime' / root.name)
    launch['source_plan'] = str(root / 'plan.json')
    launch['previous_launch'] = str(previous_launch)
    launch['collectors'] = []
    inputs = {str(previous_launch): sha256(previous_launch), old['source_plan']: sha256(old['source_plan'])}
    if continue_unstarted:
        completion = previous_launch.parent / 'completion.json'
        inputs[str(completion)] = sha256(completion)
    for entry in launch['services']:
        entry['log'] = str(root / (entry['name'] + '.log'))
    for backend, source in sources.items():
        selection, ledger = source['selection'], source['ledger']
        original = source['original_population']
        previous_selection = source['source'] / 'selection.json'
        inputs[str(previous_selection)] = sha256(previous_selection)
        if original['population_ledger'] is not None:
            inputs[original['population_ledger']] = original['population_ledger_sha256']
        for entry in ledger:
            if entry['previous_result_sha256'] is not None:
                inputs[str(Path(entry['previous_case']) / 'result.json')] = entry['previous_result_sha256']
            inputs.update(entry.get('infrastructure_recovery', {}).get('evidence_sha256', {}))
            inputs.update(entry.get('preserved_evidence_sha256', {}))
        mapping = root / 'attempts' / (backend + '.json')
        save(mapping, dict(previous_selection=str(previous_selection),
                           previous_selection_sha256=inputs[str(previous_selection)],
                           original_population=original, units=ledger))
        inputs[str(mapping)] = sha256(mapping)
        if not selection['cases']:
            continue
        output = root / 'collections' / backend / 'shard_000'
        selected = output / 'selection.json' if backend == 'sage' else root / 'inputs' / (backend + '.json')
        save(selected, selection)
        inputs[str(selected)] = sha256(selected)
        collector = rebase(copy.deepcopy(source['collector']))
        python = collector['command'][0]
        script = 'rule_teacher_pilot.py' if backend == 'sage' else 'mp3d_rule_teacher_pilot.py'
        command = [python, '-B', str(REPO / 'scripts/eval' / script), '--output', str(output),
                   '--limit', str(min(validation_limit_per_source, len(selection['cases']))
                                  if validation_limit_per_source else len(selection['cases'])),
                   '--max-calls', str(original_plan['max_calls']),
                   '--candidates', '8', '--candidate-strategy', 'fixed', '--sampling-scheduler', 'v6_depth']
        if backend != 'sage':
            command += ['--backend', backend, '--split', 'train', '--cases', str(selected)]
        old_command = source['collector']['command']
        for flag in ('--gym-url', '--render-url', '--brain-url'):
            if flag == '--brain-url' and brain_url is not None:
                value = brain_url
            elif flag in old_command:
                value = old_command[old_command.index(flag) + 1]
            elif flag == '--brain-url':
                value = original_plan['service_defaults']['brain']
            else:
                raise ValueError('explicit service URL missing: ' + flag)
            command += [flag, value]
        collector.update(command=command, log=str(root / (backend + '-collector.log')))
        launch['collectors'].append(collector)
    depth_batch = (configure_depth_batch(launch, sources, root,
                       size=depth_batch_size, port_base=gym_port_base)
                   if depth_batch_size > 1 else None)
    for entry in launch['services']:
        workspace = entry.get('native_gym_workspace')
        if workspace is not None:
            inputs[workspace['config_source']] = workspace['config_sha256']
    code = {str(path): sha256(path) for folder in ('scripts', 'verl', 'vln_demo/src')
            for path in (REPO / folder).rglob('*.py') if '__pycache__' not in path.parts}
    plan = dict(previous_plan, root=str(root), code_sha256=code,
                input_sha256={**previous_plan['input_sha256'], **inputs},
                previous_cohort_plan=(previous_plan['previous_cohort_plan'] if recover_empty_generation or continue_unstarted else old['source_plan']),
                previous_launch=str(previous_launch),
                attempt_policy='separate_directory; unfinished continuation and one bounded context-error retry; no navigation-outcome resampling',
                attempt_counts=counts, attempt_mapping=str(root / 'attempts'),
                original_populations={backend: item['original_population'] for backend, item in sources.items()},
                quality_policy='plausible local target OR direction; measured progress; original task group counts and splits')
    if recover_empty_generation:
        plan.update(previous_attempt_plan=old['source_plan'],
            attempt_policy='separate_directory; one explicit empty-generation infrastructure recovery plus unfinished units; no further context retries or navigation-outcome resampling',
            empty_generation_recovery=dict(schema='rule_teacher_empty_generation_recovery_v1',
                recoveries_used=1, context_failure_retries_allowed=False,
                evidence_groups=sum(value.get('empty_generation_retry_once', 0) for value in counts.values()),
                terminal_evidence='eight archived empty texts, empty calls, integer zero completion tokens, no selected candidate',
                old_prefixes_preserved=True, resumes_simulator_state=False))
    if continue_unstarted:
        plan.update(previous_attempt_plan=old['source_plan'],
            attempt_policy='separate_directory; only cases whose previous directory never existed; no retries or resampling',
            unstarted_continuation=dict(schema='rule_teacher_unstarted_continuation_v1',
                transitions_used=1, started_cases_excluded=True, retries_allowed=False,
                prior_successful_original_tasks_excluded=True,
                never_started_before_identity_filter=sum(
                    value.get('unstarted', 0) + value.get('excluded_previously_successful_original_task', 0)
                    for value in counts.values()),
                historical_success_units_excluded=sum(
                    value.get('excluded_previously_successful_original_task', 0) for value in counts.values()),
                scheduled_units=sum(value.get('unstarted', 0) for value in counts.values()),
                old_prefixes_preserved=True, resumes_simulator_state=False))
    if brain_url is not None:
        plan['service_defaults'] = dict(plan['service_defaults'], brain=brain_url)
    if depth_batch is not None:
        plan['depth_batch'] = depth_batch
    if validation_scope is not None:
        plan['validation_scope'] = validation_scope
        if continue_unstarted:
            plan['unstarted_continuation']['available_units'] = plan['unstarted_continuation']['scheduled_units']
            plan['unstarted_continuation']['scheduled_units'] = sum(validation_scope['scheduled_units'].values())
    save(root / 'plan.json', plan)
    save(root / 'launch.json', launch)
    return dict(preview=False, launch=str(root / 'launch.json'), counts=counts,
                code_files=len(code), mutates_previous=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-launch', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--recover-empty-generation', action='store_true',
                        help='one explicit infrastructure recovery from the first isolated attempt only')
    parser.add_argument('--continue-unstarted', action='store_true',
                        help='one explicit continuation after recovery; exclude every existing case directory')
    parser.add_argument('--brain-url', help='explicit brain endpoint for the new attempt')
    parser.add_argument('--depth-batch-size', type=int, default=1)
    parser.add_argument('--gym-port-base', type=int, default=19100)
    parser.add_argument('--validation-limit-per-source', type=int, default=0,
                        help='run only each source original-order prefix for acceptance; 0 uses the full pool')
    args = parser.parse_args()
    print(json.dumps(prepare(args.previous_launch, args.root, preview=args.preview,
                            recover_empty_generation=args.recover_empty_generation,
                            continue_unstarted=args.continue_unstarted,
                            brain_url=args.brain_url, depth_batch_size=args.depth_batch_size,
                            gym_port_base=args.gym_port_base,
                            validation_limit_per_source=args.validation_limit_per_source), ensure_ascii=False))


if __name__ == '__main__':
    main()
