"""Named pi-agent choices; model/controller/render ports share one DLC job."""
import json
from pathlib import Path


def model_choice(profile='next'):
    if profile == 'next':
        return dict(provider='vllm-next-local', model_id='qwen3.8-flash-next', port=9038)
    if profile.startswith('anonymous-data/'):
        import re
        model_id = profile.split('/', 1)[1]
        source = (Path(__file__).resolve().parents[1]/'config/pi/project-models.ts').read_text()
        gateway = source.split('pi.registerProvider("vllm-next-local"', 1)[0]
        if model_id not in re.findall(r'id: "([^"]+)"', gateway):
            raise ValueError('gateway model is not registered in project-models.ts: '+model_id)
        return dict(provider='anonymous-data', model_id=model_id, gateway=True)
    profiles = json.loads((Path(__file__).resolve().parents[1]/'vllm/sglang_profiles.json').read_text())
    if profile not in profiles:
        raise ValueError('unknown agent model profile: '+profile)
    return dict(profiles[profile], provider='sglang-local')


def validate_agent_config(config):
    profile = config.get('agent_profile', 'next')
    spec = model_choice(profile)
    if spec.get('gateway'):
        if any(config.get(k) for k in ('agent_server_job_id', 'agent_server_status_file')):
            raise ValueError('gateway models use the existing project provider configuration')
        return
    if profile != 'next':
        if config.get('server_job_id') and config.get('agent_server_job_id') != config['server_job_id']:
            raise ValueError('agent model and renderer must belong to the same independent DLC job')
        if not str(config.get('agent_server_job_id', '')).startswith('dlc'):
            raise ValueError('selected model requires its own agent_server_job_id')
        if not config.get('agent_server_status_file') or not config.get('agent_acceptance_file'):
            raise ValueError('selected model requires status and acceptance receipt files')
    elif any(config.get(k) for k in ('agent_server_job_id', 'agent_server_status_file')):
        raise ValueError('Next uses the existing resident server binding')


def validate_acceptance(config):
    spec = model_choice(config['agent_profile'])
    binding = config['agent_server_binding']
    receipt = json.loads(Path(config['agent_acceptance_file']).read_text())
    urls = {f'http://{host}:{spec["port"]}' for host in binding['hosts']}
    if not (receipt.get('passed') is True and receipt.get('hostname') == binding['pod_id']
            and receipt.get('model_id') == spec['model_id'] and receipt.get('url', '').rstrip('/') in urls
            and all(receipt.get(key) is True for key in ('text', 'image', 'tools'))
            and max(receipt.get('cached_tokens', [0])) > 0):
        raise ValueError('selected model requires matching real inference/cache acceptance')
