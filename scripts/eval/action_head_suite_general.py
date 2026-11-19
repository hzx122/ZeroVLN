"""Reuse the VLA-RLVR VLMEvalKit prompts, inference and native dataset scorers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import threading

from action_head_benchmark_suite import ROOT, GENERAL, atomic_json, digest, sha256_file

KIT = ROOT / 'VLMEvalKit'
SHARED = Path('/data/shared/groups/mfm/common/anno/vlmevalkit')
TSV_NAMES = {name: name + '.tsv' for name in GENERAL}
TSV_NAMES.update({'RefSpatial-Bench-' + name: 'RefSpatial_' + name + '.tsv'
                 for name in ('Location', 'Placement', 'Unseen')})


def setup(data_root, config):
    sys.path.insert(0, str(KIT))
    os.environ.update(LMUData=str(data_root), FORCE_REBUILD_TSV='0',
        VLMEVAL_LOCAL_MEDIA='0', VLMEVAL_TEXT_ACTION_COT_PROMPT='0',
        REFSPATIAL_ROOT=config.get('refspatial_root', '/data/datasets/data/ann/vlf_baidu/RefSpatial-Bench'),
        VIEWSPATIAL_ROOT=config.get('viewspatial_root', '/data/datasets/data/ann/vlf_baidu/ViewSpatial-Bench'),
        DA2K_ROOT=str(data_root / 'DA-2K'))


def build(benchmark, data_root, config):
    setup(data_root, config)
    from vlmeval.dataset import build_dataset
    dataset = build_dataset(benchmark)
    if dataset is None or dataset.data.empty:
        raise ValueError('benchmark has no usable samples: ' + benchmark)
    indexes = dataset.data['index'].astype(str).tolist()
    if len(indexes) != len(set(indexes)):
        raise ValueError('duplicate benchmark indexes: ' + benchmark)
    return dataset


def prepare_population(benchmark, data_root, config):
    data_root.mkdir(parents=True, exist_ok=True)
    setup(data_root, config)
    inputs = {}
    if benchmark == 'DA-2K':
        source = Path(config.get('da2k_root', str(SHARED)))
        # Existing public data has a ready TSV; copy it instead of rebuilding
        # into ~/.cache or modifying its shared source.
        tsv = source / 'da2k.tsv'
        if not tsv.exists():
            source = source / 'DA-2K'
            tsv = source / 'da2k.tsv'
        own = data_root / 'DA-2K'
        own.mkdir(exist_ok=True)
        if tsv.exists():
            shutil.copyfile(tsv, own / 'da2k.tsv')
            inputs[str(tsv.resolve())] = sha256_file(tsv)
        else:
            annotation = source / 'annotations.json'
            if not annotation.is_file() or not (source / 'images').is_dir():
                raise FileNotFoundError('DA-2K local annotations/images or ready da2k.tsv not found')
            from vlmeval.dataset.da2k import DA2K
            DA2K.__new__(DA2K)._build_tsv_from_annotations(str(source), str(own / 'da2k.tsv'), True)
            inputs[str(annotation.resolve())] = sha256_file(annotation)
        inputs[str(own / 'da2k.tsv')] = sha256_file(own / 'da2k.tsv')
    else:
        source = Path(config.get('general_data_root', str(SHARED))) / TSV_NAMES[benchmark]
        if not source.is_file():
            raise FileNotFoundError(source)
        target = data_root / source.name
        if not target.exists():
            shutil.copyfile(source, target)
        inputs[str(source.resolve())] = sha256_file(source)
        if sha256_file(target) != inputs[str(source.resolve())]:
            raise ValueError('copied benchmark TSV differs from source')
        inputs[str(target.resolve())] = sha256_file(target)
    dataset = build(benchmark, data_root, config)
    cases = [dict(index=str(index), logical_id=digest([benchmark, str(index)])[:20])
             for index in dataset.data['index']]
    return dict(data_root=str(data_root), kit=str(KIT), input_sha256=inputs,
                dataset_class=type(dataset).__name__, judge='exact_matching'), cases


def select_dataset(dataset, cases):
    indexes = [row['index'] for row in cases]
    table = dataset.data.copy()
    table['_suite_index'] = table['index'].astype(str)
    table = table.set_index('_suite_index', drop=True)
    if any(index not in table.index for index in indexes):
        raise ValueError('frozen sample no longer belongs to this benchmark')
    if dataset.dataset_name == '3DSRBench':
        dataset.suite_reference_qids = dataset.data['qid'].tolist()
    dataset.data = table.loc[indexes].reset_index(drop=True).copy()
    if dataset.dataset_name.startswith('RefSpatial-Bench-'):
        # These three datasets use the existing local RefSpatialDataset with
        # embedded masks, not the unrelated RefSpatialBench mask_path schema.
        for mask in dataset.data['mask']:
            if dataset._load_mask(mask) is None:
                raise ValueError('selected RefSpatial embedded mask cannot be decoded')
    return dataset


class APIReplicas:
    """Distribute requests while retaining the existing model prompt/API code."""
    is_api = True

    def __init__(self, urls, config):
        from vlmeval.api import LMDeployAPI
        sampling = dict(temperature=.7, top_p=.8, top_k=20, min_p=0., presence_penalty=1.5,
            repetition_penalty=1., max_tokens=32768, seed=3407,
            chat_template_kwargs={'enable_thinking': False}, retry=10, timeout=1800)
        sampling.update(config.get('general_sampling', {}))
        self.models = [LMDeployAPI(model='default', api_base=url.rstrip('/') + '/v1/chat/completions',
                                  key='', **sampling) for url in urls]
        if not self.models:
            raise ValueError('at least one real brain endpoint required')
        self.lock, self.counter = threading.Lock(), 0

    def __getattr__(self, name):
        return getattr(self.models[0], name)

    def set_dump_image(self, function):
        for model in self.models:
            model.set_dump_image(function)

    def generate(self, *args, **kwargs):
        with self.lock:
            model = self.models[self.counter % len(self.models)]
            self.counter += 1
        return model.generate(*args, **kwargs)


def infer(plan, stage, cases, output, urls):
    from vlmeval.inference import infer_data_job
    manifest = json.loads(Path(stage['manifest']).read_text())
    dataset = select_dataset(build(stage['benchmark'], Path(manifest['metadata']['data_root']),
                                   plan['config']), cases)
    dataset.img_root = str(output / 'media')
    model = APIReplicas(urls, plan['config'])
    infer_data_job(model, str(output), 'suite_brain', dataset,
                   api_nproc=plan['config'].get('general_api_workers', 128), retry_failed=True)
    from vlmeval.smp.file import get_pred_file_path
    from vlmeval.smp import load
    predictions = Path(get_pred_file_path(str(output), 'suite_brain', stage['benchmark'], use_env_format=True))
    frame = load(str(predictions))
    if set(frame['index'].astype(str)) != {row['index'] for row in cases}:
        raise ValueError('prediction indexes differ from the frozen node slice')
    valid = len(frame) == len(cases) and not any('Failed to obtain answer' in str(x) for x in frame['prediction'])
    return dict(evaluation_valid=valid, completed=len(frame), predictions=str(predictions),
                predictions_sha256=sha256_file(predictions))


def aggregate(plan, stage, receipts, output):
    import pandas as pd
    from vlmeval.smp import load, dump
    manifest = json.loads(Path(stage['manifest']).read_text())
    dataset = select_dataset(build(stage['benchmark'], Path(manifest['metadata']['data_root']),
                                   plan['config']), manifest['cases'])
    frames = []
    for receipt in receipts:
        result = receipt.get('result') or {}
        if not receipt.get('evaluation_valid'):
            raise ValueError('one or more benchmark data shards failed')
        if not result.get('predictions'):
            if receipt.get('selected') == 0:
                continue
            raise ValueError('missing predictions for a nonempty shard')
        path = Path(result['predictions'])
        if sha256_file(path) != result['predictions_sha256']:
            raise ValueError('benchmark predictions changed after node receipt')
        frames.append(load(str(path)))
    data = pd.concat(frames, ignore_index=True)
    ids = data['index'].astype(str).tolist()
    expected = {case['index'] for case in manifest['cases']}
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError('benchmark shards have overlap or missing predictions')
    predictions = output / 'predictions.xlsx'
    dump(data, str(predictions))
    # Score the complete selected dataset once. Averaging per-node scores
    # would bias unequal shards/categories and alter benchmark semantics.
    kwargs = dict(model='exact_matching')
    if stage['benchmark'] == 'RoboSpatialHome':
        dataset.img_root = str(output / 'media')
        for _, row in dataset.data.iterrows():
            dataset.dump_image(row)
    if stage['benchmark'] == '3DSRBench':
        kwargs['reference_qids'] = dataset.suite_reference_qids
    metrics = dataset.evaluate(str(predictions), **kwargs)
    serial = json.loads(metrics.to_json(orient='records')) if hasattr(metrics, 'to_json') else metrics
    atomic_json(output / 'metrics.json', serial)
    return dict(evaluation_valid=True, completed=len(data), metrics=serial,
                predictions=str(predictions), scope=manifest['scope'])
