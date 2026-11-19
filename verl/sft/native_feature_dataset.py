"""Version-bound native features; shard image/fast tensors without duplicating text embeddings."""
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import Dataset

from sft.action_head_joint_dataset import load_anchor_groups
from sft.joint_fm_head_sft import JointBundleDataset, joint_collate


def frontend_fingerprint(model_dir):
    root = Path(model_dir)
    index = root/'model.safetensors.index.json'
    weights = json.loads(index.read_text())['weight_map']
    names = sorted({v for k,v in weights.items()
                    if k.startswith('model.visual.') or k == 'model.language_model.embed_tokens.weight'})
    config = {name:hashlib.sha256((root/name).read_bytes()).hexdigest()
              for name in ('config.json','preprocessor_config.json','tokenizer_config.json','tokenizer.json')}
    signature = dict(model=str(root.resolve()), config=config,
        weights={name:dict(size=(root/name).stat().st_size, mtime_ns=(root/name).stat().st_mtime_ns)
                 for name in names},
        implementation=hashlib.sha256((Path(__file__).parents[1]/'action_head/context_encoder.py').read_bytes()).hexdigest())
    return hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()


def prepare_shards(manifest, output, model_dir, device, rank, world, shard_anchors=16):
    """Reuse the tested online cache frontend, including native placeholder expansion."""
    from action_head.context_encoder import NativeQwenFrontend, NativeImageFeatureCache
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups = load_anchor_groups(str(manifest), 'train')
    frontend = NativeQwenFrontend(model_dir, device=device).eval().requires_grad_(False)
    cache = NativeImageFeatureCache()
    signature = frontend_fingerprint(model_dir)
    index_rows = []
    for shard in range(rank, (len(groups)+shard_anchors-1)//shard_anchors, world):
        tensors, rows = {}, []
        target = output/f'shard_{shard:05d}.safetensors'
        if target.exists(): raise FileExistsError(target)

        def put(value):
            value = value.detach().cpu().contiguous().clone()
            key = hashlib.sha256(str((value.dtype,tuple(value.shape))).encode()+value.view(torch.uint8).numpy().tobytes()).hexdigest()
            tensors.setdefault(key, value)
            return key

        for group in groups[shard*shard_anchors:(shard+1)*shard_anchors]:
            row = group[0]
            inputs, hidden, meta = cache.encode(frontend, row['slow_messages'], row['slow_completion'], device)
            count = inputs['image_grid_thw'].shape[0]
            if not 1 <= count <= 11: raise ValueError('cached input exceeds 8+3')
            ids = inputs['input_ids'].cpu().squeeze(0)
            visual = hidden[inputs['input_ids'].eq(frontend.image_id)]
            counts = (inputs['image_grid_thw'].prod(-1)//frontend.processor.image_processor.merge_size**2).tolist()
            images = [put(v) for v in visual.split(counts)]
            dataset = JointBundleDataset(group)
            batch = joint_collate([dataset[i] for i in range(len(dataset))])
            fast = {k:[put(v) for v in batch[k]] for k in
                    ('reference_visual_tokens','current_visual_tokens','target_actions','action_mask')}
            item = dict(anchor_id=row['slow_anchor_id'], split='train', shard=str(target),
                        input_ids=put(ids), images=images, image_count=count,
                        token_count=ids.numel(), window_count=len(group), image_token_id=frontend.image_id,
                        source_samples=[r['sample'] for r in group], **fast)
            rows.append(item)
            del hidden, visual, batch, inputs
        pending = target.with_suffix('.pending')
        save_file(tensors, str(pending), metadata={'frontend_fingerprint':signature})
        pending.replace(target)
        target.with_suffix('.json').write_text(json.dumps(rows)+'\n')
        index_rows.extend(rows)
        print(json.dumps(dict(event='native_cache_shard',rank=rank,shard=shard,anchors=len(rows),bytes=target.stat().st_size)),flush=True)
    cache.clear()
    (output/f'rank_{rank:02d}.json').write_text(json.dumps(dict(
        frontend_fingerprint=signature, source_manifest=str(manifest), anchors=len(index_rows),
        image_token_id=frontend.image_id, pad_token_id=frontend.processor.tokenizer.pad_token_id,
        rank=rank, world=world))+'\n')


def merge_shards(output, model_dir, world):
    output = Path(output)
    audits = [json.loads((output/f'rank_{rank:02d}.json').read_text()) for rank in range(world)]
    signature = frontend_fingerprint(model_dir)
    assert all(a['frontend_fingerprint']==signature for a in audits)
    rows = [r for p in sorted(output.glob('shard_*.json')) for r in json.loads(p.read_text())]
    rows.sort(key=lambda r:r['anchor_id'])
    assert len(rows)==sum(a['anchors'] for a in audits)
    assert len({r['anchor_id'] for r in rows})==len(rows)
    (output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    source = Path(audits[0]['source_manifest'])
    report = dict(ok=True, schema='v5_native_features_v1', frontend_fingerprint=signature,
                  model_dir=str(model_dir), source_manifest=str(source),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  anchors=len(rows), windows=sum(r['window_count'] for r in rows),
                  max_images=max(r['image_count'] for r in rows),
                  max_tokens=max(r['token_count'] for r in rows),
                  pad_token_id=audits[0]['pad_token_id'], image_token_id=audits[0]['image_token_id'],
                  bytes=sum(p.stat().st_size for p in output.glob('shard_*.safetensors')))
    (output/'ready.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


class NativeFeatureDataset(Dataset):
    def __init__(self, root, model_dir, split='train', source_manifest=None):
        self.root = Path(root)
        self.ready = json.loads((self.root/'ready.json').read_text())
        if not self.ready['ok'] or self.ready['frontend_fingerprint'] != frontend_fingerprint(model_dir):
            raise ValueError('native feature cache/model/processor version mismatch')
        if source_manifest is not None:
            digest = hashlib.sha256(Path(source_manifest).read_bytes()).hexdigest()
            if digest != self.ready['source_sha256']:
                raise ValueError('native feature cache/source manifest mismatch')
        self.rows = [json.loads(s) for s in (self.root/'manifest.jsonl').read_text().splitlines()]
        self.rows = [r for r in self.rows if r['split']==split]
        if not self.rows: raise ValueError(f'no {split} features')

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with safe_open(row['shard'], framework='pt', device='cpu') as f:
            item = {k:torch.stack([f.get_tensor(key) for key in row[k]]) for k in
                    ('reference_visual_tokens','current_visual_tokens','target_actions','action_mask')}
            item.update(input_ids=f.get_tensor(row['input_ids']),
                        cached_visual_tokens=torch.cat([f.get_tensor(key) for key in row['images']]),
                        image_count=row['image_count'], image_token_id=row['image_token_id'])
        return item


class NativeFeatureCollator:
    def __init__(self, pad_token_id, seed):
        self.pad_token_id, self.seed, self.calls = pad_token_id, seed, 0

    def __call__(self, items):
        width = max(i['input_ids'].numel() for i in items)
        counts = [i['target_actions'].shape[0] for i in items]
        ids = torch.stack([F.pad(i['input_ids'],(width-i['input_ids'].numel(),0),value=self.pad_token_id) for i in items])
        masks = torch.stack([F.pad(torch.ones_like(i['input_ids']),(width-i['input_ids'].numel(),0)) for i in items])
        visual = torch.cat([i['cached_visual_tokens'] for i in items])
        if sum(i['input_ids'].eq(i['image_token_id']).sum().item() for i in items) != visual.shape[0]:
            raise ValueError('cached image placeholder count mismatch')
        batch = {k:torch.cat([i[k] for i in items]) for k in
                 ('reference_visual_tokens','current_visual_tokens','target_actions','action_mask')}
        batch.update(input_ids=ids,attention_mask=masks,cached_visual_tokens=visual,
                     slow_image_count_per_anchor=torch.tensor([i['image_count'] for i in items]),
                     slow_pixel_count_per_anchor=torch.zeros(len(items),dtype=torch.long),
                     window_anchor_index=torch.repeat_interleave(torch.arange(len(items)),torch.tensor(counts)),
                     window_index_within_anchor=torch.cat([torch.arange(n) for n in counts]),
                     noise_seed=torch.tensor([self.seed+(self.calls+i)*1000 for i in range(len(items))]),
                     anchor_valid=torch.ones(len(items)))
        for prefix in ('reference_visual','current_visual'):
            batch[prefix+'_mask'] = torch.ones(batch[prefix+'_tokens'].shape[:2],dtype=torch.bool)
        self.calls += len(items)
        return batch


def main():
    """Run the existing sharded encoder and require full coverage before training."""
    import argparse
    import os
    import subprocess
    import sys
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'worker', 'verify'])
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--world', type=int, default=16)
    p.add_argument('--expected-anchors', type=int, required=True)
    p.add_argument('--expected-windows', type=int, required=True)
    a = p.parse_args()
    if a.world != 16:
        raise ValueError('native cache preparation requires a complete 16-GPU node')
    if not a.output.resolve().is_relative_to(Path(
            '/data/zerovln')):
        raise ValueError('native features must be written to the authorized data disk')
    if a.mode == 'worker':
        rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(rank)
        prepare_shards(a.manifest, a.output, a.model, f'cuda:{rank}', rank, a.world)
        return
    if a.mode == 'prepare' and not (a.output/'ready.json').exists():
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   '--nproc_per_node=16', '-m', 'sft.native_feature_dataset',
                   'worker', *sys.argv[2:]]
        print(json.dumps(dict(event='native_full_cache_start', command=command)), flush=True)
        subprocess.run(command, check=True)
        merge_shards(a.output, a.model, a.world)
    dataset = NativeFeatureDataset(a.output, a.model, source_manifest=a.manifest)
    ready = dataset.ready
    actual = (len(dataset), sum(r['window_count'] for r in dataset.rows))
    expected = (a.expected_anchors, a.expected_windows)
    if actual != expected or (ready['anchors'], ready['windows']) != expected:
        raise ValueError(f'native cache coverage mismatch: {actual} != {expected}')
    if ready['max_images'] > 11:
        raise ValueError('native cache exceeds the 8+3 image window')
    print(json.dumps(dict(event='native_full_cache_ready', **ready)), flush=True)


if __name__ == '__main__':
    main()
