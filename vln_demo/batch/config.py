"""Validate a local batch running inside its model/render DLC Pod."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULTS = dict(datasets='r2r,rxr', split='val_unseen', role='guide', languages='*',
                limit_per_dataset=0, workers=32, render_mode='batch', timeout=0, max_tool_calls=100)


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError('config must be a JSON object')
    unknown = set(config) - set(DEFAULTS) - {'server_job_id', 'server_timeout', 'server_status_file', 'manifest', 'resume_from', 'evaluation_ladder', 'resume_ladder', 'agent_profile', 'agent_server_job_id', 'agent_server_status_file', 'agent_acceptance_file', 'evaluation_size'}
    if unknown:
        raise ValueError('unsupported config keys: ' + ', '.join(sorted(unknown)))
    config = {**DEFAULTS, **config}
    from agent_models import validate_agent_config
    validate_agent_config(config)
    if not isinstance(config.get('server_job_id'), str) or not config['server_job_id'].startswith('dlc'):
        raise ValueError('explicit resident server_job_id required')
    for key in ('workers', 'server_timeout', 'max_tool_calls'):
        value = config.get(key, 2400)
        if type(value) is not int or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(config['timeout']) is not int or config['timeout'] < 0:
        raise ValueError('timeout must be >= 0; 0 disables the episode wall clock')
    if config.get('resume_ladder') and not config.get('evaluation_ladder'):
        raise ValueError('resume_ladder requires evaluation_ladder')
    if type(config['limit_per_dataset']) is not int or config['limit_per_dataset'] < 0:
        raise ValueError('limit_per_dataset must be >= 0; 0 means all episodes')
    if config['workers'] > 32:
        raise ValueError('workers must not exceed 32')
    if config.get('evaluation_size', 'ladder') not in ('ladder', '100', '200', '500', '1000', 'full'):
        raise ValueError('unknown fixed evaluation size')
    if config.get('evaluation_size') and not config.get('evaluation_ladder'):
        raise ValueError('evaluation_size requires locked evaluation_ladder')
    if config.get('evaluation_ladder'):
        if config.get('manifest') or config.get('resume_from'):
            raise ValueError('evaluation_ladder cannot be combined with manifest/resume_from')
        from eval_sets import load_ladder
        load_ladder(config['evaluation_ladder'], verify_sources=True)
    if config['datasets'] not in ('r2r', 'rxr', 'r2r,rxr', 'rxr,r2r', 'objectnav'):
        raise ValueError('datasets must select r2r/rxr or objectnav')
    if config['datasets'] == 'objectnav':
        if not config.get('manifest') or config.get('evaluation_ladder'):
            raise ValueError('objectnav requires an explicit versioned manifest')
        from objectnav_batch import load_manifest
        load_manifest(config['manifest'])
    if config['split'] not in ('train', 'val_seen', 'val_unseen'):
        raise ValueError('evaluation uses val_seen/val_unseen; data generation uses train')
    if config['role'] not in ('guide', 'follower') or config['render_mode'] not in ('batch', 'display'):
        raise ValueError('invalid role or render_mode')
    return config
