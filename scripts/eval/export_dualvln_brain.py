#!/usr/bin/env python3
"""Export the unchanged Qwen2.5-VL tensors from an official DualVLN checkpoint.

The navigation checkpoint stays intact. Only S1 modules are omitted from the
separate, standard HF brain used by the existing VLMEvalKit/vLLM evaluator.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'verl/scripts/eval/qwen36_4query_ckpt_sweep'))
from common import atomic_json, sha256_file

PREFIXES = ('model.layers.', 'model.embed_tokens.', 'model.norm.', 'visual.', 'lm_head.')


def export(source, output):
    from safetensors import safe_open
    from safetensors.torch import save_file
    source, output = Path(source).resolve(), Path(output).resolve()
    allowed = Path('/data/zerovln/artifacts')
    if not output.is_relative_to(allowed):
        raise ValueError('brain export must be an own experiment artifact')
    proof = json.loads((source / 'download_verification.json').read_text())
    if not proof.get('ok') or proof['model'] != 'InternRobotics/InternVLA-N1-DualVLN':
        raise ValueError('verified official DualVLN weights required')
    output.mkdir(parents=True, exist_ok=False)
    index = json.loads((source / 'model.safetensors.index.json').read_text())
    selected = {k: v for k, v in index['weight_map'].items() if k.startswith(PREFIXES)}
    if not {'model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight'} <= set(selected):
        raise ValueError('incomplete standard Qwen2.5-VL brain')
    source_hashes = {row['file']: row['sha256'] for row in proof['files']}
    exported, total_bytes = {}, 0
    for shard in sorted(set(selected.values())):
        path = source / shard
        if sha256_file(path) != source_hashes[shard]:
            raise ValueError('official checkpoint shard changed: ' + shard)
        with safe_open(str(path), framework='pt', device='cpu') as weights:
            tensors = {k: weights.get_tensor(k) for k, filename in selected.items() if filename == shard}
        partial = output / (shard + '.partial')
        save_file(tensors, str(partial), metadata={'format': 'pt'})
        partial.replace(output / shard)
        # Safetensors is lossless. Compare every exported tensor on CPU against
        # the corresponding official tensor before declaring the export usable.
        import torch
        with safe_open(str(output / shard), framework='pt', device='cpu') as written:
            for key, tensor in tensors.items():
                if not torch.equal(tensor, written.get_tensor(key)):
                    raise ValueError('export changed an official tensor: ' + key)
                total_bytes += tensor.numel() * tensor.element_size()
        exported[shard] = sha256_file(output / shard)
        del tensors
        print(json.dumps({'phase': 'brain_shard_verified', 'shard': shard}), flush=True)
    for row in proof['files']:
        name = row['file']
        if name.endswith(('.json', '.txt', '.model')) and name not in ('config.json', 'model.safetensors.index.json'):
            if sha256_file(source / name) != row['sha256']:
                raise ValueError('official processor/tokenizer changed: ' + name)
            shutil.copy2(source / name, output / name)
    config = json.loads((source / 'config.json').read_text())
    config.update(architectures=['Qwen2_5_VLForConditionalGeneration'], model_type='qwen2_5_vl', use_cache=True)
    for name in ('system1', 'n_query', 'model_cfg'):
        config.pop(name, None)
    atomic_json(output / 'config.json', config)
    atomic_json(output / 'model.safetensors.index.json', dict(metadata={'total_size': total_bytes}, weight_map=selected))
    result = dict(schema='dualvln_unchanged_brain_export_v1', ok=True, source=str(source),
        source_revision=proof['revision'], source_proof_sha256=sha256_file(source / 'download_verification.json'),
        selected_tensors=len(selected), removed_tensors=len(index['weight_map']) - len(selected),
        tensor_values='exact_cpu_equality', exported_shard_sha256=exported)
    atomic_json(output / 'brain_export.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output)), flush=True)
