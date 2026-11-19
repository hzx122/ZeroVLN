"""Private, one-episode console adapter. No terminal or configuration writes."""
import hmac
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlparse, parse_qs
import importlib.util

_spec = importlib.util.spec_from_file_location('vln_navigation_metrics', Path(__file__).with_name('navigation_metrics.py'))
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
scalar_metrics, initial_geometry_error = _metrics.scalar_metrics, _metrics.initial_geometry_error


def install(module):
    token = os.environ.get('VLN_BATCH_TOKEN')
    if not token:
        return lambda env: None
    os.environ.setdefault('VLN_RENDER_MODE', 'batch')
    output = Path(os.environ['VLN_BATCH_OUTPUT'])
    output.mkdir(parents=True, exist_ok=True)
    finish = threading.Event()
    result = {}
    geometry = {}
    original_get, original_post = module.Handler.do_GET, module.Handler.do_POST
    controller_url = os.environ['VLN_BATCH_CONTROLLER_MODEL_URL'].rstrip('/').removesuffix('/v1')
    label, _, model_id, desc = module.P.MODELS['qwen35b']
    module.P.MODELS['qwen35b'] = label, controller_url, model_id, desc

    def reply(handler, payload, code=200):
        handler._send(json.dumps(payload), 'application/json', code)

    def authorized(handler):
        if not hmac.compare_digest(handler.headers.get('X-VLN-Batch-Token', ''), token):
            reply(handler, {'error': 'unauthorized'}, 403)
            return False
        return True

    def get(handler):
        if not authorized(handler):
            return
        parsed = urlparse(handler.path)
        if parsed.path == '/batch-config':
            sampling, thinking = module.P.sampling_params(module.CFG['mode'], module.CFG['temp'])
            reply(handler, dict(config=module.CFG, model_endpoint=module.P.model_endpoint(module.CFG['model']),
                sampling=sampling, enable_thinking=thinking,
                geometry=geometry,
                prompt=handler._prompt({})))
            return
        if parsed.path == '/batch-result':
            reply(handler, result or {'state': 'stop_requested' if finish.is_set() else 'pending'})
            return
        if parsed.path not in ('/status', '/frame.jpg', '/frame-snapshot', '/vln-map-snapshot', '/cmd'):
            reply(handler, {'error': 'endpoint unavailable in isolated episode'}, 403)
            return
        if parsed.path == '/cmd':
            query = parse_qs(parsed.query)
            if result or query.get('do', [''])[0] not in ('start', 'point_move', 'stop'):
                reply(handler, {'error': 'episode reset/switch/settings unavailable'}, 403)
                return
            with (output / 'commands.jsonl').open('a') as stream:
                stream.write(json.dumps({'time': time.time(), 'query': query}, ensure_ascii=False) + '\n')
        return original_get(handler)

    def post(handler):
        if not authorized(handler):
            return
        path = urlparse(handler.path).path
        if path == '/batch-finish':
            length = int(handler.headers.get('Content-Length', '0'))
            if not 0 <= length <= 1024:
                reply(handler, {'error': 'oversized request'}, 400)
                return
            handler.rfile.read(length)
            if module.STATE['status'] in ('running', 'loading'):
                reply(handler, {'error': 'wait for current action to finish'}, 409)
                return
            finish.set()
            reply(handler, {'state': 'stop_requested'})
            return
        if path != '/react-reference' or result:
            reply(handler, {'error': 'endpoint unavailable in isolated episode'}, 403)
            return
        return original_post(handler)

    module.Handler.do_GET, module.Handler.do_POST = get, post
    original_refresh = module.refresh

    def refresh(env, obs, *args, **kwargs):
        value = original_refresh(env, obs, *args, **kwargs)
        state = env.sim.get_agent_state()
        with (output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps({'time': time.time(), 'position': state.position.tolist(),
                'rotation_wxyz': [state.rotation.w, state.rotation.x, state.rotation.y, state.rotation.z],
                'step': module.STATE.get('step'), 'seq': module.STATE.get('seq')}) + '\n')
        return value
    module.refresh = refresh
    original_load = module.load_episode
    def load(env, ep, args):
        env.seed(int(os.environ.get('VLN_BATCH_SEED', '0')))
        obs = original_load(env, ep, args)
        if getattr(env.sim, 'config', None) is not None:
            agent = env.sim.config.agents[0]
            geometry.update(agent_height=float(agent.height), agent_radius=float(agent.radius))
            mesh = env.sim.pathfinder.nav_mesh_settings
            geometry.update(navmesh_height=float(mesh.agent_height), navmesh_radius=float(mesh.agent_radius))
        metrics = env.get_metrics()
        if initial_geometry_error(metrics):
            values, invalid = scalar_metrics(metrics)
            result.update(state='failed', reason='invalid_geometry', explicit_stop=False,
                          navigation_success=False, metrics=values, invalid_metrics=invalid,
                          geometry_validation='initial_goal_unreachable', steps=0)
            (output / 'evaluator.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
            module.set_state(status='error', result='初始位置到目标不可达；本回合不计作模型导航失败')
        return obs
    module.load_episode = load

    def tick(env):
        if result:
            return
        if not finish.is_set():
            return
        env.step(module.STOP)
        metrics = env.get_metrics()
        values, invalid = scalar_metrics(metrics)
        result.update(state='finished', reason='stop', explicit_stop=True,
                      navigation_success=bool(values.get('success')),
                      metrics=values, steps=env._elapsed_steps,
                      protocol='external_reads_and_network_allowed')
        if invalid:
            result.update(state='failed', reason='invalid_metrics', invalid_metrics=invalid,
                          navigation_success=False)
        (output / 'evaluator.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        module.set_state(status='finished')
    return tick
