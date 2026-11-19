"""Prepare eight v6-sized online workers using the existing pilots/supervisor.

SAGE throughput phase: four scene strata per renderer, sixteen original train
instructions per scene, selected before any calls. Native sources run separately.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

from prepare_rule_teacher_three_source import ART, REPO, SAGE_SPLIT, rank, save, sha256
from supervise_rule_teacher_collection import CACHE_KEYS


def option(command, name, value):
    if name in command:
        command[command.index(name) + 1] = str(value)
    else:
        command.extend([name, str(value)])


def prepare(root):
    if root.exists() or not root.resolve().is_relative_to(ART):
        raise ValueError('requires a new artifact directory')
    sys.path.insert(0, str(REPO / 'vln_demo/src'))
    from sage.datasets import SageSet
    dataset = SageSet(SAGE_SPLIT)
    strata = []
    inputs = {}
    for scene in sorted(dataset.list_scenes(), key=lambda s: rank(s, 20261002)):
        tasks = sorted(dataset.tasks(scene), key=lambda t: rank(scene + ':' + t.task_id, 20261002))
        if len(tasks) >= 16:
            source = str(Path(dataset.traj_file(scene)).resolve())
            inputs[source] = sha256(source)
            strata.append([dict(scene=scene, task=t.task_id, source=source) for t in tasks[:16]])
        dataset._tasks.clear()
        if len(strata) == 32:
            break
    if len(strata) != 32:
        raise ValueError('not enough original train scene strata')
    old = json.loads((ART / 'rule-teacher-quality-attempt-20261002-r5/launch.json').read_text())
    render = next(e for e in old['services'] if e['name'] == 'sage-render')
    actor = next(e for e in old['services'] if e['name'].startswith('sage-gym-'))
    collector = next(e for e in old['collectors'] if e['name'] == 'sage-collector')
    def environment(original):
        return {k: v.replace('rule-teacher-quality-20261002-r5', REPO.name)
                for k, v in original.items() if k not in CACHE_KEYS}
    root.mkdir(parents=True)
    code = {str(p): sha256(p) for p in REPO.rglob('*.py') if '.runtime' not in p.parts}
    save(root / 'source-plan.json', dict(schema='local16_original_train_v1',
         source_sha256=inputs, code_sha256=code, input_sha256={},
         selection='hash-ranked scenes with >=16 instructions; first16 hash-ranked tasks; no outcome filtering',
         seed=20261002, original_instructions=True))
    workers = []
    for worker in range(8):
        wr = root / f'worker{worker}'
        out = wr / 'collections/sage/shard_000'
        rows = [row for stratum in strata[worker*4:worker*4+4] for row in stratum]
        save(out / 'selection.json', dict(scope='SAGE original train, fixed scene strata before rollout', cases=rows))
        lane = worker // 2
        gpu = lane*4 + 2 + worker%2
        render_port = 7200 + worker
        actor_base = 21000 + worker*100
        env = environment(collector['environment'])
        env['SAGE_RENDER_URL'] = f'http://127.0.0.1:{render_port}'
        runtime = REPO.parent / '.runtime' / root.name / f'worker{worker}'
        rs = copy.deepcopy(render)
        rs.update(name='sage-render', log=str(wr/'sage-render.log'), port=render_port,
                  health_url=f'http://127.0.0.1:{render_port}/ping', startup_group='render')
        rs['environment'] = environment(render['environment'])
        rs['environment']['SAGE_RENDER_URL'] = env['SAGE_RENDER_URL']
        option(rs['command'], '--port', render_port)
        option(rs['command'], '--gpus', gpu)
        services = [rs]
        urls = []
        for slot in range(64):
            port = actor_base + slot
            url = f'http://127.0.0.1:{port}'
            urls.append(url)
            entry = copy.deepcopy(actor)
            entry.update(name=f'sage-gym-{slot:02}', log=str(wr/f'sage-gym-{slot:02}.log'),
                         port=port, health_url=url+'/health', startup_group='actors',
                         startup_timeout=600, environment=dict(env))
            option(entry['command'], '--port', port)
            option(entry['command'], '--scene', rows[slot]['scene'])
            option(entry['command'], '--episode_id', rows[slot]['task'])
            services.append(entry)
        col = copy.deepcopy(collector)
        col.update(log=str(wr/'collector.log'), environment=env)
        col['command'] = [v.replace('rule-teacher-quality-20261002-r5', REPO.name) for v in col['command']]
        for key, value in {'--output':out, '--limit':64, '--depth-batch-size':64,
                           '--gym-url':urls[0], '--gym-urls':json.dumps(urls),
                           '--render-url':env['SAGE_RENDER_URL'],
                           '--brain-url':f'http://127.0.0.1:{29200+100*lane}'}.items():
            option(col['command'], key, value)
        for flag in ('--rolling-actors', '--rolling-scene-refill'):
            if flag in col['command']: col['command'].remove(flag)
        for key, value in {'--sampling-scheduler': 'v6_depth', '--candidate-strategy': 'fixed',
                '--candidates': 8, '--http-concurrency': 64, '--actor-workers': 8}.items():
            option(col['command'], key, value)
        plan = dict(schema='local16_online_collection_v1', runtime=str(runtime),
                    source_plan=str(root/'source-plan.json'), environment={}, services=services,
                    collectors=[col], service_start_batch_size=8,
                    allocation=dict(worker=worker, lane=lane, visual_gpu=gpu, actors=64,
                                    http_concurrency=64, actor_workers=8),
                    formal_training_started=False)
        save(wr/'launch.json', plan)
        workers.append(str(wr/'launch.json'))
    save(root/'preparation.json', dict(workers=workers, cases=512, scenes=32, brain_lanes=4,
         gpu_count=16, phase='SAGE online throughput', native_sources_pending=True,
         preserved=['all instructions at the same depth', 'one rejected retry', 'zero/tiny movement labels'],
         differences_from_v6=['online rule execution instead of cached golden bundles',
                             'rule labels do not consume query hidden rows',
                             'original instruction scene-stratified diagnostic population']))
    print(json.dumps(dict(root=str(root), workers=len(workers), cases=512)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    prepare(parser.parse_args().root)
