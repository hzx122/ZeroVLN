"""Validate the TP8 agent slot of an independent 16-GPU service."""
import json
from pathlib import Path


def model_profile_files(repo, profile):
    config = json.loads((repo/'vln_demo/vllm/sglang_profiles.json').read_text())[profile]
    if config['tp_size'] != 8:
        raise ValueError('agent model must use TP8')
    model = Path(config['model_path'])
    architecture = json.loads((model/'config.json').read_text())['architectures'][0]
    if not architecture.startswith('Qwen3_5'):
        raise ValueError('unexpected model architecture')
    index = json.loads((model/'model.safetensors.index.json').read_text())
    if any(not (model/shard).is_file() for shard in set(index['weight_map'].values())):
        raise ValueError('model checkpoint is incomplete')
    return ('vln_demo/dlc/model_service.py', 'vln_demo/vllm/sglang_profiles.json', 'vln_demo/vllm/sglang_server.py',
            'vln_demo/vllm/'+config['launcher'])
