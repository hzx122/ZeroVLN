"""Connect the established VLN console to the production unified evaluator.

The console retains its panels, streams, terminal and event log. Dataset and
inference settings are owned by the unified selection, not legacy UI defaults.
"""
import io
import json
import math
import os
from pathlib import Path
import threading
import time
from urllib.request import urlopen
from urllib.parse import urlparse

from PIL import Image

from unified_closedloop_console import Console

LABELS = {'sage-3d': 'SAGE-3D', 'r2r': 'R2R', 'rxr': 'RxR'}


def catalog(cases):
    """Namespace scene IDs: R2R and RxR share MP3D IDs, but not episodes."""
    scenes, lookup, assets = {}, {}, []
    for bench, rows in cases.items():
        ids = set()
        for case in rows:
            key = bench + ':' + case['scene']
            ids.add(key)
            scenes.setdefault(key, []).append(str(case['episode_id']))
            lookup[key, str(case['episode_id'])] = bench, case
        assets.append((bench, LABELS[bench], '统一闭环评测', sorted(ids)))
    return scenes, lookup, assets


def serve(m, settings):
    console = Console(settings)
    scenes, lookup, assets = catalog(console.cases)
    first_scene = os.environ.get('UNIFIED_CONSOLE_INITIAL_SCENE') or next(iter(scenes))
    current = [lookup[first_scene, scenes[first_scene][0]]]
    checkpoint = dict(path=settings['checkpoint'], step=96, version='v8',
                      experiment=Path(settings['checkpoint']).parents[1].name,
                      group=str(Path(settings['checkpoint']).parent))
    m.CFG.update(driver='action_head', model='distance96', maxtools=200,
                 nact=1, action_head_as_tool_version='v8')
    m.set_state(driver='action_head', model='distance96', action_head_as_tool_version='v8')
    m.P.MODELS['distance96'] = ('Distance ×5 · step 96', 'http://127.0.0.1:29880',
                              'Qwen3.5-4B', '统一评测 checkpoint')
    m.manual_head_status = lambda: dict(state='selected', active=checkpoint,
        phase='统一评测将在开始回合时加载所选小脑', error=None)
    def readiness():
        try:
            with urlopen('http://127.0.0.1:29880/v1/models', timeout=3) as r:
                ready = bool(json.load(r).get('data'))
            return dict(ready=ready, label='Distance ×5 · step 96')
        except OSError as exc:
            return dict(ready=False, message=str(exc))
    m.brain_readiness = readiness

    def select(bench, case, operation):
        console.act(dict(benchmark=bench, logical_id=case['logical_id'], operation=operation))
        current[:] = [(bench, case)]
        m.set_state(scene=bench+':'+case['scene'], scene_title=case['scene'],
            scene_name=case['scene'], episode_id=str(case['episode_id']),
            backend=bench, backend_label=LABELS[bench],
            instruction=case.get('instruction', case.get('instruction_text', '')),
            status='loading' if operation == 'load' else 'running', result='')
        m.push_event('sys', text=f'{LABELS[bench]} · {case["scene"]} · {case["episode_id"]} · {operation}')

    class Handler(m.Handler):
        def _cmd(self, q, agent_control=False):
            do = q.get('do', [''])[0]
            try:
                if do in ('lock', 'unlock', 'clearlog'):
                    if do == 'clearlog': m.clear_events()
                    else: m.set_state(locked=do == 'lock')
                    return 'ok'
                if m.STATE['locked'] and not agent_control:
                    return 'locked'
                if do == 'stop':
                    console.stop()
                    return 'ok'
                if do == 'switch':
                    key = (q['scene'][0], q['episode_id'][0])
                    select(*lookup[key], 'load')
                    return 'ok'
                if do in ('reset', 'start'):
                    bench, case = current[0]
                    instruction = q.get('instruction', [''])[0].strip()
                    if instruction and instruction != m.STATE['instruction'].strip():
                        raise ValueError('统一评测使用数据集原始指令')
                    select(bench, case, 'start' if do == 'start' else 'load')
                    return 'ok'
                if do == 'load_action_head' and q.get('path', [''])[0] == checkpoint['path']:
                    return 'ok'
                # Do not silently accept settings which the production collector ignores.
                raise ValueError('本次使用固定 step 96 和统一评测配置；此操作未启用')
            except (ValueError, KeyError) as exc:
                m.push_event('err', text=str(exc))
                return str(exc)

        def _prompt(self, q):
            from action_head import protocol
            p = protocol.protocol_module('v8')
            system = p.build_system_prompt()
            return dict(label='v8 统一闭环评测', text=system+'\n\n'+m.STATE['instruction'],
                        sys_text=system, nact=1, maxtools=200, nhist=1, sys_custom=False)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == '/action-head/catalog':
                return self._send(json.dumps(dict(ok=True, checkpoints=[checkpoint], rejected=[], warnings=[])), 'application/json')
            if path == '/traj_cases':
                return self._send(json.dumps(dict(cases=[])), 'application/json')
            return super().do_GET()

        def do_POST(self):
            if urlparse(self.path).path.startswith('/terminal'):
                return super().do_POST()
            length = int(self.headers.get('Content-Length', '0'))
            self.rfile.read(min(length, 1024*1024))
            self._send(json.dumps(dict(ok=False, error='统一评测配置固定')), 'application/json', 409)

    config = m.webui.config(list(scenes), first_scene, scenes,
        m.P.template_list(), m.CFG['tpl'], [('instruct', '统一评测', '固定采样配置')], 'instruct',
        [('action_head', 'FM Head · v8 统一评测', 'Distance ×5 step 96')], 'action_head',
        [('distance96', 'Distance ×5 · step 96', settings['checkpoint'], False)], 'distance96',
        200, 200, 200, assets, scene_titles={s:s.split(':',1)[1] for s in scenes}, nact=1)
    server = m.ThreadingHTTPServer(('0.0.0.0', settings['port']), Handler)
    server.daemon_threads = True
    server.web_config = config

    from action_head.console_events import Tail, replay
    from rule_teacher_training_handoff import load
    history = os.environ.get('UNIFIED_CONSOLE_HISTORY_ROOT')
    if history:
        history = Path(history).resolve()
        if not history.is_relative_to(Path(os.environ.get('RL_ARTIFACT_ROOT', '/data/closed_loop_3dgs_rl')).resolve()):
            raise ValueError('history must be on the authorized data disk')
        for path in sorted(history.glob('attempts/*/stages/**/rollouts/**/*.json')):
            row = load(path)
            m.push_event('sys', text='恢复已完成回合日志：'+str(path))
            for event in replay(row):
                kind = event.pop('t')
                m.push_event(kind, **event)

    def update():
        event_tail = Tail()
        previous_phase, previous_log, offset = None, None, 0
        while not m.SHUTDOWN.is_set():
            try:
                state = dict(console.state)
                phase = state.get('phase')
                changed_phase = phase != previous_phase
                if changed_phase:
                    m.set_state(status={'loading':'loading', 'running':'running', 'loaded':'idle',
                        'complete':'done', 'failed':'error', 'stopped':'stopped'}.get(phase,'idle'),
                        result=state.get('error') or '')
                    m.push_event('sys' if phase != 'failed' else 'err',
                        text=f'统一评测：{phase}' + (f' · {state["error"]}' if state.get('error') else ''))
                    previous_phase = phase
                if state.get('attempt'):
                    for event in event_tail.read(Path(state['attempt'])/'model-events.jsonl'):
                        kind = event.pop('t')
                        m.push_event(kind, **event)
                    log = Path(state['attempt'])/'collector.log'
                    if log != previous_log: offset, previous_log = 0, log
                    if log.exists():
                        with log.open() as stream:
                            stream.seek(offset)
                            for line in stream:
                                m.push_event('sys', text=line.rstrip())
                            offset = stream.tell()
                if changed_phase and phase in ('complete', 'failed', 'stopped'):
                    # A zero process exit is not itself a navigation success.
                    m.push_event('done', text=json.dumps(dict(phase=phase,
                        returncode=state.get('returncode'), result=state.get('result'),
                        error=state.get('error')), ensure_ascii=False))
                if phase in ('loading', 'ready'):
                    time.sleep(.3)
                    continue
                bench, case = current[0]
                base = settings['bindings'][bench]['gym']
                with urlopen(base+'/telemetry', timeout=10) as response:
                    telemetry = json.load(response)
                if not telemetry['ready']: continue
                images = []
                for path in ('frame.jpg','map.jpg','vln-map.jpg'):
                    with urlopen(base+'/'+path, timeout=20) as response: images.append(response.read())
                metrics = telemetry.get('metrics', {})
                distance = float(metrics.get('distance_to_goal', float('nan')))
                m.publish_frame(Image.open(io.BytesIO(images[0])).convert('RGB'), images[0],
                    int(telemetry['steps_used']), distance, map_jpg=images[1], tool_map_jpg=images[2])
                evaluation = m.goal_evaluation(metrics)
                if bench != 'sage-3d':
                    evaluation.update(criterion='原生 geodesic < 3m，主动 STOP',
                        success=bool(metrics.get('success', False)),
                        label='成功' if metrics.get('success') else '未成功')
                m.set_state(evaluation=evaluation)
            except Exception as exc:
                if str(exc) != getattr(update, 'error', None):
                    m.push_event('err', text=f'可视化刷新：{exc}')
                    update.error = str(exc)
            time.sleep(.5)
    select(*current[0], 'load')
    threading.Thread(target=update, daemon=True).start()
    print(f'Existing VLN console on {settings["port"]}; checkpoint={settings["checkpoint"]}', flush=True)
    try:
        server.serve_forever()
    finally:
        m.SHUTDOWN.set()
        console.stop()
