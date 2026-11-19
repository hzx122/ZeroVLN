"""Run isolated episodes against model/render services in this same DLC Pod."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.request import ProxyHandler, build_opener

HERE = Path(__file__).resolve().parent


def get(url):
    with build_opener(ProxyHandler({})).open(url, timeout=3) as response:
        return json.load(response)


def identity_matches(job_id, binding, host, health, status_file=None):
    if health.get('job_id') is not None:
        return health['job_id'] == job_id and health.get('hostname') == binding['pod_id']
    # Compatibility for an already-running renderer from before /health exposed
    # identity. Require a fresh, independently supervised CPFS heartbeat.
    if not status_file:
        return False
    try:
        status = json.loads(Path(status_file).read_text())
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(status['updated_at'])).total_seconds()
        return (0 <= age < 120 and status['hostname'] == binding['pod_id']
                and host in binding['hosts'] and status['state'] in ('starting', 'ready')
                and status['processes']['habitat_render']['alive'])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def resolve_services(job_id, timeout, binding, status_file=None, datasets='r2r,rxr', require_next=True):
    if (not binding or binding.get('job_id') != job_id
            or not binding.get('pod_id', '').startswith(job_id + '-master-')
            or not binding.get('hosts')):
        raise ValueError('missing/mismatched server binding; resolve the owning Pod before starting')
    deadline = time.monotonic() + timeout
    last = 'not ready'
    while time.monotonic() < deadline:
        for host in binding['hosts']:
            try:
                port = 7041 if datasets == 'objectnav' else 7040
                health = get(f'http://{host}:{port}/health')
                if datasets == 'objectnav' and health.get('backend') != 'objectnav':
                    last = 'ObjectNav renderer required'
                    continue
                if not identity_matches(job_id, binding, host, health, status_file):
                    last = 'server identity or supervisor heartbeat mismatch'
                    continue
                models = get(f'http://{host}:9038/v1/models') if require_next else {'data': []}
                controller_models = get(f'http://{host}:28900/v1/models')
                if health.get('ok') and health.get('stateless') and (not require_next or any(
                        model['id']=='qwen3.8-flash-next' for model in models.get('data', []))) and any(
                        model['id']=='Qwen3.6-35B' for model in controller_models.get('data', [])):
                    return dict(job_id=job_id, pod_id=binding['pod_id'], host=host,
                                render_url=f'http://{host}:{port}', model_url=f'http://{host}:9038/v1',
                                controller_model_url=f'http://{host}:28900/v1')
            except (OSError, ValueError) as exc:
                last = f'{type(exc).__name__}: {exc}'
        print(json.dumps({'waiting_for_server': job_id, 'reason': last}), flush=True)
        time.sleep(15)
    raise TimeoutError(f'selected services were not ready after {timeout}s; last state {last}')


def runner_command(config, services):
    command = [sys.executable, '-u', str(HERE / 'runner.py'), '--run-dir', config['run_dir'],
               '--model-url', services['model_url'], '--render-url', services['render_url'],
               '--controller-model-url', services['controller_model_url']]
    for key in ('datasets', 'split', 'role', 'languages', 'limit_per_dataset', 'workers',
                'timeout', 'manifest', 'render_mode', 'resume_from', 'max_tool_calls', 'agent_profile'):
        if key in config:
            command += ['--'+key.replace('_','-'), str(config[key])]
    return command


def resolve_agent_service(config, timeout):
    from agent_models import model_choice
    spec = model_choice(config['agent_profile'])
    binding = config['agent_server_binding']
    job = config['agent_server_job_id']
    if binding.get('job_id') != job or not binding.get('pod_id', '').startswith(job+'-master-'):
        raise ValueError('agent model job/Pod binding mismatch')
    deadline = time.monotonic()+timeout
    last = 'not ready'
    while time.monotonic() < deadline:
        for host in binding['hosts']:
            try:
                status = json.loads(Path(config['agent_server_status_file']).read_text())
                age = (datetime.now(timezone.utc)-datetime.fromisoformat(status['updated_at'])).total_seconds()
                if not (status['hostname'] == binding['pod_id'] and 0 <= age < 120
                        and status['state'] == 'ready' and status['model_id'] == spec['model_id']
                        and status['port'] == spec['port'] and status.get('radix_cache') is True
                        and status.get('mtp') is False):
                    raise ValueError('agent model identity/cache/heartbeat mismatch')
                url = f'http://{host}:{spec["port"]}'
                models = get(url+'/v1/models')
                info = get(url+'/server_info')
                if spec['model_id'] not in [m['id'] for m in models.get('data', [])]:
                    raise ValueError('wrong model at selected endpoint')
                if (info.get('disable_radix_cache') is not False or info.get('speculative_algorithm') is not None
                        or info.get('tp_size') != 8 or info.get('context_length') != spec['context_length']):
                    raise ValueError('effective model cache/MTP/TP/context mismatch')
                return dict(model_url=url+'/v1', agent_job_id=job, agent_pod_id=binding['pod_id'],
                            agent_profile=config['agent_profile'], agent_model_id=spec['model_id'])
            except (OSError, ValueError, KeyError) as exc:
                last = str(exc)
        print(json.dumps({'waiting_for_agent_model': job, 'reason': last}), flush=True)
        time.sleep(5)
    raise TimeoutError('selected agent model unavailable: '+last)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get('mode') == 'debug':
        # Legacy running debug pods may restart this shared path. Not a submission mode.
        from benchmarks.debug import serve
        serve(config)
        return
    from agent_models import model_choice
    choice = model_choice(config.get('agent_profile', 'next'))
    services = resolve_services(config['server_job_id'], config.get('server_timeout', 2400),
                                config.get('server_binding'), config.get('server_status_file'),
                                config.get('datasets', 'r2r,rxr'),
                                require_next=config.get('agent_profile', 'next') == 'next' or choice.get('gateway', False))
    if choice.get('gateway'):
        services.update(model_url='', agent_profile=config['agent_profile'], agent_model_id=choice['model_id'])
    elif config.get('agent_profile', 'next') != 'next':
        from agent_models import validate_acceptance
        validate_acceptance(config)
        services.update(resolve_agent_service(config, config.get('server_timeout', 2400)))
    (args.config.parent.parent / 'server-binding.json').write_text(json.dumps(services, indent=2)+'\n')
    # Real kernel write + child-process + network probe, inside this exact DLC container.
    with tempfile.TemporaryDirectory(prefix='vln-sandbox-probe-') as tmp:
        work = Path(tmp) / 'workspace'
        work.mkdir()
        (work / 'AGENTS.md').write_text('sandbox preflight\n')
        outside = Path(tmp) / 'outside'
        code = ('const fs=require("node:fs");fs.writeFileSync("inside","ok");'
                f'try{{fs.writeFileSync({json.dumps(str(outside))},"bad");process.exit(20)}}'
                'catch(e){if(e.code!=="EROFS"&&e.code!=="EACCES")throw e;}'
                f'fetch({json.dumps(services["render_url"]+"/health")}).then(r=>r.json())'
                '.then(x=>{if(!x.ok)process.exit(21);console.log("sandbox-preflight-ok")});')
        subprocess.run([sys.executable, '-B', str(HERE / 'sandbox.py'), '--workspace', str(work), '--',
                        '/opt/external/user-d5121be6/tools/node/bin/node', '-e', code], check=True, timeout=60)
        if outside.exists() or (work / 'inside').read_text() != 'ok':
            raise RuntimeError('sandbox write boundary failed')
    if config.get('evaluation_ladder'):
        from ladder import run
        return run(config, services)
    command = runner_command(config, services)
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
