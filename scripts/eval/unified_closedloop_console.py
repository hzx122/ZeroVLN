"""Interactive case selection around the production suite, without a second policy loop.

Only service concurrency is reduced: one brain and one Gym per benchmark. Camera,
target geometry, prompts, context compaction, execution and scoring come from the
unified suite. Every Start writes a new attempt and invokes its actual collector.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.parse import urlparse, parse_qs

from action_head_benchmark_suite import REPO, atomic_json, digest, stage_root
from rule_teacher_training_handoff import load


def remap(value, ports):
    """Rewrite endpoint/argv ports, never case IDs or checkpoint metadata."""
    if isinstance(value, dict):
        return {k: remap(v, ports) for k, v in value.items()}
    if isinstance(value, list):
        return [remap(v, ports) for v in value]
    if isinstance(value, int):
        return ports.get(value, value)
    if isinstance(value, str):
        if value.isdigit() and int(value) in ports:
            return str(ports[int(value)])
        for old, new in ports.items():
            value = value.replace('127.0.0.1:' + str(old), '127.0.0.1:' + str(new))
        return value
    return value


def prepare_console(config_path, output, port, smoke=False, smoke_gpus=(0,1)):
    from unified_closedloop_eval import prepare
    from action_head_suite_stage import launch_plan
    from supervise_rule_teacher_collection import collection_runtime_root
    selected = load(config_path)
    selected['nodes'] = 1
    # One user-selected episode is one evaluation attempt; no hidden repetitions.
    selected['evaluation_repeats'] = 1
    selected['sample_counts'] = dict.fromkeys(selected['benchmarks'], 1 if smoke else 0)
    output.mkdir(parents=True, exist_ok=False)
    selection = output/'selection.json'
    atomic_json(selection, selected)
    run = prepare(selection, output/'suite', 'full')
    plan_path = Path(run['rounds'][0])
    plan = load(plan_path)
    runtime = collection_runtime_root()/output.name
    os.environ['ACTION_HEAD_SUITE_RUNTIME_BASE'] = str(runtime/'stages')
    services, bindings = [], {}
    inputs = copy.deepcopy(plan)
    for index, stage in enumerate(plan['stages']):
        bench = stage['benchmark']
        original = load(launch_plan(plan, stage, 0, plan_path))
        inputs['input_sha256'].update(load(original['source_plan'])['input_sha256'])
        ports = {7085: 17885+index, 19100: 17900+index, 29600: 29880}
        env = remap(original['environment'], ports)
        collector_env = remap(original['collectors'][0]['environment'], ports)
        collector_env.update(SUITE_GYM_URLS=f'http://127.0.0.1:{17900+index}',
            SUITE_BRAIN_URLS='http://127.0.0.1:29880', SUITE_FM_GPUS='0',
            SUITE_MAX_ACTIVE_SCENES='1', SUITE_MAX_ACTIVE_CASES='1',
            SUITE_QUERY_CACHE=str(runtime/'queries'))
        if smoke:
            collector_env['SUITE_FM_GPUS'] = str(smoke_gpus[0])
        bindings[bench] = dict(stage=stage, environment={**env, **collector_env},
            gym=f'http://127.0.0.1:{17900+index}', manifest=stage['manifest'])
        for entry in original['services']:
            name = entry['name']
            if not (name in ('renderer', 'renderer_0', 'gym_0') or (index == 0 and name == 'brain_0')):
                continue
            row = remap(copy.deepcopy(entry), ports)
            row['name'] = bench.replace('-', '_')+'_'+name
            row['environment'] = {**env, **row.get('environment', {})}
            row['log'] = str(output/'services'/(row['name']+'.log'))
            if row.get('reusable_compile_cache'):
                key = Path(row['reusable_compile_cache']).parent.name
                row['reusable_compile_cache'] = str(collection_runtime_root()/'compile-cache'/key/'brain_0')
            if name == 'brain_0':
                row['environment'].update(ACTION_HEAD_QUERY_CACHE=str(runtime/'queries'),
                    ACTION_HEAD_COMPILE_CACHE_ROOT=str(runtime/'brain-compile'))
                if smoke:
                    row['environment'].update(VLLM_GPUS=','.join(map(str,smoke_gpus)),
                        VLLM_MAX_NUM_SEQS='4', VLLM_MAX_NUM_BATCHED_TOKENS='4096')
            if 'native_gym_workspace' in row:
                row['native_gym_workspace']['cwd'] = str(REPO/'.runtime/native-gym'/runtime.name/row['name']/'work')
            row['startup_group'] = 'brains' if name.startswith('brain') else ('renderers' if name.startswith('renderer') else 'gyms')
            services.append(row)
    services.sort(key=lambda row: {'brains': 0, 'renderers': 1, 'gyms': 2}[row['startup_group']])
    atomic_json(output/'inputs.json', inputs)
    settings = dict(plan=str(plan_path), output=str(output), port=port,
        checkpoint=plan['config']['checkpoints'][0]['path'], bindings=bindings,
        protocol=plan['source_provenance']['protocol'])
    atomic_json(output/'console.json', settings)
    supervisor = dict(runtime=str(runtime), source_plan=str(output/'inputs.json'),
        environment=dict(PYTHONPATH=':'.join(map(str,[REPO/'scripts/eval', REPO/'scripts/train',REPO/'verl',REPO/'vln_demo/src',REPO])),
            PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'), services=services,
        collectors=[dict(name='console', command=[sys.executable,'-B',str(Path(__file__).resolve()),
            '--settings',str(output/'console.json')],log=str(output/'console.log'))],
        independent_collectors=True, service_log_spool=True, service_start_batch_size=4)
    if smoke:
        supervisor['collectors'][0]['name'] = 'minimal_evaluation'
        supervisor['collectors'][0]['command'].append('--smoke')
    atomic_json(output/'launch.json', supervisor)
    return output/'launch.json'


def launch(config, output, port, dependency_spec):
    from unified_closedloop_eval import git_runtime
    from unified_companion_task import dependencies
    git_runtime(require_container=True)
    dependencies(load(dependency_spec))
    path = prepare_console(config, output, port)
    return subprocess.call([sys.executable,'-B',str(REPO/'scripts/train/supervise_rule_teacher_collection.py'),
                            '--plan',str(path)], cwd=REPO)


class Console:
    def __init__(self, settings):
        self.settings = settings
        self.root = Path(settings['output'])
        self.plan = load(settings['plan'])
        self.lock = threading.Lock()
        self.busy = False
        self.child = None
        self.cancelled = False
        self.state = dict(phase='ready', attempt=None, error=None)
        self.cases = {b:load(v['manifest'])['cases'] for b,v in settings['bindings'].items()}
        self.selected = next(iter(self.cases))

    def act(self, body):
        bench, logical = body['benchmark'], body['logical_id']
        case = next(c for c in self.cases[bench] if c['logical_id'] == logical)
        operation = body['operation']
        if operation not in ('load','start'):
            raise ValueError('unknown operation')
        with self.lock:
            if self.busy:
                raise ValueError('当前回合仍在运行，请等待结束后切换')
            self.busy = True
            self.cancelled = False
            self.selected = bench
            self.state = dict(phase='loading' if operation=='load' else 'running',
                benchmark=bench, case=case, attempt=None, error=None)
        threading.Thread(target=self.work,args=(bench,case,operation),daemon=True).start()

    def stop(self):
        import signal
        self.cancelled = True
        child = self.child
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)


    def work(self, bench, case, operation):
        from action_head_suite_stage import subset_manifest
        try:
            binding = self.settings['bindings'][bench]
            if operation == 'load':
                request = Request(binding['gym']+'/reset', data=json.dumps(dict(scene=case['scene'],
                    episode_id=case['episode_id'])).encode(), headers={'Content-Type':'application/json'})
                with urlopen(request, timeout=600) as response:
                    result = json.load(response)
                if result.get('error') or result.get('ok') is False:
                    raise RuntimeError(str(result))
                self.state.update(phase='loaded', info=result.get('info'))
                return
            directory = self.root/'attempts'/str(time.time_ns())
            directory.mkdir(parents=True)
            self.state['attempt'] = str(directory)
            plan = copy.deepcopy(self.plan)
            plan['output'] = str(directory)
            stage = copy.deepcopy(binding['stage'])
            manifest = directory/'cases.json'
            atomic_json(manifest,subset_manifest(load(binding['manifest']),[case]))
            stage['manifest'] = str(manifest)
            plan['stages'][stage['index']] = stage
            atomic_json(directory/'plan.json',plan)
            node = stage_root(plan, stage)/'node_0'
            node.mkdir(parents=True)
            atomic_json(node/'cases.json',load(manifest))
            env = {**os.environ, **binding['environment']}
            env['ACTION_HEAD_EVAL_CONTEXT_ARCHIVE'] = str(directory/'canonical-contexts')
            env['ACTION_HEAD_CONSOLE_EVENTS'] = str(directory/'model-events.jsonl')
            command = [sys.executable,'-B',str(REPO/'scripts/eval/action_head_suite_stage.py'),
                '--plan',str(directory/'plan.json'),'--stage',str(stage['index']),'--rank','0','--collector']
            with (directory/'collector.log').open('ab') as log:
                child = subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT, start_new_session=True)
                self.child = child
                if self.cancelled:
                    self.stop()
                self.state['pid'] = child.pid
                code = child.wait()
            result_path = node/'evaluation_result.json'
            result = load(result_path) if result_path.exists() else None
            atomic_json(directory/'exit.json',dict(returncode=code,result=result))
            self.state.update(phase='stopped' if self.cancelled else ('complete' if code==0 else 'failed'),returncode=code,result=result)
        except Exception as exc:
            self.state.update(phase='failed',error=repr(exc))
        finally:
            self.child = None
            self.busy = False
            atomic_json(self.root/'console-state.json',self.state)


def serve(settings):
    # Keep the established console, panels, HTTP streams and event log.
    import vln_model
    from unified_console_adapter import serve as serve_existing
    serve_existing(vln_model, settings)


def run_smoke(settings):
    """Run one real case per selected benchmark through the native collector."""
    console = Console(settings)
    results = []
    for bench, cases in console.cases.items():
        if len(cases) != 1:
            raise ValueError('smoke evaluation requires exactly one prepared case per benchmark')
        console.work(bench, cases[0], 'start')
        state = copy.deepcopy(console.state)
        valid = (state.get('phase') == 'complete' and state.get('returncode') == 0
                 and (state.get('result') or {}).get('evaluation_valid') is True)
        results.append(dict(benchmark=bench, logical_id=cases[0]['logical_id'],
                            valid=valid, state=state))
        atomic_json(console.root/'smoke-summary.json', dict(
            complete=len(results)==len(console.cases), ok=all(r['valid'] for r in results),
            results=results, scope='Runtime smoke only; not reported benchmark performance.'))
        print(json.dumps(dict(event='smoke_case_finished',benchmark=bench,valid=valid)),flush=True)
        if not valid:
            return 1
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--settings',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.smoke:
        raise SystemExit(run_smoke(load(args.settings)))
    serve(load(args.settings))
