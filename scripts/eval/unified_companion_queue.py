"""Alternate model variants at checkpoint boundaries on one resident node set.

Each job calls the existing unified watcher unchanged in scoring/model scope.
The queue owns scheduling, barriers and failure propagation only.
"""
import copy
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from types import SimpleNamespace

from rule_teacher_training_handoff import load, save, bind
from dlc_control_io_retry import retry_estale
from dlc_workspace_stdout_heartbeat import task_heartbeat
from watch_v6_rl_local_credit import process_identity


def job_order(variants):
    names = [row['name'] for row in variants]
    if (not names or len(names) != len(set(names))
            or any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name) for name in names)):
        raise ValueError('queue variant names must be distinct safe identifiers')
    bindings = [load(row['binding']) for row in variants]
    for binding in bindings:
        steps = binding['steps']
        if (not steps or len(set(steps)) != len(steps)
                or any(type(step) is not int or step < 0 for step in steps)):
            raise ValueError('queue requires distinct nonnegative real checkpoint steps')
    for step in sorted({step for binding in bindings for step in binding['steps']}):
        for variant, binding in zip(variants, bindings):
            if step in binding['steps']:
                yield variant, step


def prepare_queue(spec_path, spec, owners):
    from unified_closedloop_eval import prepare_watch
    root = Path(spec['root'])
    variants = spec['variants']
    ordered = list(job_order(variants))
    configs = {}
    contracts = []
    for variant in variants:
        name = variant['name']
        selected = load(variant['selection'])
        if selected['nodes'] != spec['nodes']:
            raise ValueError('queue selection node count differs from resident set')
        base = root/'variants'/name
        binding = load(variant['binding'])
        binding['resident_owners'] = owners
        save(base/'resolved-binding.json', binding)
        configs[name] = prepare_watch(Path(variant['selection']), base/'resolved-binding.json',
            base/'run', spec['purpose'])
        contracts.append(load(base/'run/base-plan.json')['closedloop_contract'])
    if any(contract != contracts[0] for contract in contracts):
        raise ValueError('variants must share one evaluation contract')
    jobs = []
    for variant, step in ordered:
        name = variant['name']
        directory = root/'jobs'/name/f'global_step_{step:06d}'
        config = copy.deepcopy(configs[name])
        config.update(root=str(directory), steps=[step], queue_root=str(root),
            queue_spec=bind(spec_path), evaluation_variant=name)
        path = directory/'watcher-config.json'
        save(path, config)
        jobs.append(dict(variant=name, step=step, config=str(path)))
    result = dict(schema='unified_companion_checkpoint_queue_v1', jobs=jobs,
        contract=contracts[0], spec=bind(spec_path), nodes=spec['nodes'])
    save(root/'queue.json', result)
    return result


def failed_nodes(root, nodes):
    return [str(root/f'queue-failure-node{rank}.json') for rank in range(nodes)
            if retry_estale((root/f'queue-failure-node{rank}.json').exists,
                str(root/f'queue-failure-node{rank}.json'))]


def run_queue(spec_path, spec, rank, owners):
    from unified_closedloop_eval import watch
    from rule_teacher_openloop_companion import waiting_reservation
    root, owner = Path(spec['root']), owners[rank]
    node = root/'queue-nodes'/str(rank)
    node.mkdir(parents=True, exist_ok=True)
    state_path = node/'status.json'
    completed = []
    save(state_path, dict(stage='preparing_variant_queue', completed=completed))
    me = process_identity(os.getpid())
    stream = open(f'/proc/{owner["controller_identity"]["pid"]}/fd/1', 'w', buffering=1)
    stop = threading.Event()
    def progress():
        state = load(state_path)
        resident = load(Path(owner['debug_output'])/'debug-status.json')
        return dict(heartbeat=time.time(), stage=state['stage'], mode=resident['mode'],
            occupier_pid=resident.get('occupier_pid'), progress=state)
    args = SimpleNamespace(controller_pid=me['pid'], controller_start_ticks=me['start_ticks'], output=node)
    heartbeat = threading.Thread(target=task_heartbeat, args=(args,), daemon=True,
        kwargs=dict(stop_event=stop, read_status=progress,
            write_receipt=lambda event: save(node/'stdout-heartbeat.json', event),
            emit=lambda event: print(json.dumps(dict(event, queue_node=rank)), file=stream, flush=True)))
    heartbeat.start()
    try:
        if rank == 0:
            prepare_queue(spec_path, spec, owners)
        deadline = time.monotonic()+1800
        while not retry_estale((root/'queue.json').exists, str(root/'queue.json')):
            if failed_nodes(root, spec['nodes']):
                raise RuntimeError('another queue owner failed preparation')
            waiting_reservation(owner)
            if time.monotonic() >= deadline:
                raise TimeoutError('queue preparation timed out')
            time.sleep(2)
        queue = load(root/'queue.json')
        if queue['spec'] != bind(spec_path):
            raise ValueError('queue specification changed')
        for job in queue['jobs']:
            if failed_nodes(root, spec['nodes']):
                raise RuntimeError('another queue owner failed; preserve all attempts')
            config_path = Path(job['config'])
            save(state_path, dict(stage='running_unified_watch', job=job, completed=completed))
            code = watch(config_path, rank) or 0
            directory = config_path.parent
            result = load(directory/f'watchers/node{rank}/completion.json')
            if code or result.get('evaluation_valid') is not True:
                raise RuntimeError('queue job failed unified evaluation')
            save(directory/f'queue-node{rank}.json', dict(job=job, identity=me, evaluation_valid=True))
            releases = [directory/f'queue-node{node_rank}.json' for node_rank in range(spec['nodes'])]
            while not all(retry_estale(path.exists, str(path)) for path in releases):
                if failed_nodes(root, spec['nodes']):
                    raise RuntimeError('another queue owner failed at checkpoint barrier')
                waiting_reservation(owner)
                save(state_path, dict(stage='waiting_all_nodes_before_variant_switch', job=job, completed=completed))
                time.sleep(2)
            if any(load(path).get('evaluation_valid') is not True or load(path)['job'] != job for path in releases):
                raise ValueError('queue release identity differs')
            completed.append(dict(variant=job['variant'], step=job['step']))
            save(state_path, dict(stage='checkpoint_complete_idle', completed=completed))
        save(node/'completion.json', dict(evaluation_valid=True, completed=completed, time=time.time()))
        return 0
    except BaseException as exc:
        save(root/f'queue-failure-node{rank}.json', dict(error=repr(exc), identity=me,
            completed=completed, time=time.time()))
        raise
    finally:
        stop.set()
