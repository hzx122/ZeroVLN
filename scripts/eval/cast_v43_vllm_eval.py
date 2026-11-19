#!/usr/bin/env python3
"""Replay the frozen CAST paired experiment through actual vLLM query decode."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'verl'), str(REPO/'vln_demo/src'), str(REPO/'scripts/eval'),
               str(REPO/'verl/scripts/eval/qwen36_4query_ckpt_sweep')]
from common import atomic_json, checkpoint_identity, sha256_file
from v43_online_query_parity import SOURCE, MODEL, ART

DEFAULT_OUTPUT = ART/'cast-v43-paired-vllm-local-20260915-r1'


def prepare(output):
    report = json.loads((SOURCE/'final_summary.json').read_text())
    for suffix in ['', 'equal_diagonal_paraphrase']:
        source, target = SOURCE/suffix, output/suffix
        target.mkdir(parents=True, exist_ok=True)
        for name in ['accepted_pairs.json', 'data_report.json', 'manifest.jsonl']:
            shutil.copy2(source/name, target/name)
        for phase in ['eval_before', 'eval_after']:
            (target/phase).mkdir(exist_ok=True)
            shutil.copy2(source/phase/'cases.json', target/phase/'cases.json')
    atomic_json(output/'protocol.json', {
        'source_hf_experiment': str(SOURCE),
        'checkpoints': {'eval_before': report['source_checkpoint'], 'eval_after': report['checkpoint']},
        'backend': 'real vLLM TP2 autoregressive decode, fixed per-request tool completion, 16 query tail',
        'format': 'production auto/string, unchanged; this is the primary deployment-path retest',
        'cases': 6, 'train_states': 4, 'pilot_heldout_states': 2, 'draws_per_condition': 4,
        'online_requests_per_checkpoint': 48,
        'canonical_and_primary_paraphrase': 'unchanged from frozen original evaluation',
        'supplemental_paraphrase': 'the same post-hoc equal-diagonal template from the HF experiment',
        'gate': 'both requested sides have >=5cm signed y in first 4 points and static replay collision-free',
        'scope': 'tests fixed brain-instruction -> action; not free brain generation or rendered closed-loop SR',
    })


def capture(output):
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from action_head_rl.vllm_service import VllmService
    from action_head.vllm_bridge import request_queries, strip_query_tail
    from action_head_rl.artifact import load_query_routing_artifact
    from action_head_d16_weight_sync_probe import import_probe
    http = import_probe('paired_online_http', REPO/'tmp/query_parity_probe_runtime/capture_vllm.py')
    protocol = json.loads((output/'protocol.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = json.loads((Path(MODEL)/'config.json').read_text())['text_config']
    active = output/'active_query/dualvln_query_embeddings.pt'
    active.parent.mkdir(exist_ok=True)

    def publish(checkpoint):
        source = Path(checkpoint)/'dualvln_query/dualvln_query_embeddings.pt'
        staged = active.with_suffix('.next.pt')
        shutil.copyfile(source, staged)
        os.replace(staged, active)
        if sha256_file(source) != sha256_file(active):
            raise ValueError('published query checkpoint hash differs')

    publish(next(iter(protocol['checkpoints'].values())))
    has_supplement = protocol.get('supplemental', True)
    port = protocol.get('port', 28916)
    total_requests = 0
    service = VllmService(repo=str(REPO), model=MODEL, query_checkpoint=str(active),
        query_cache=str(output/'online_artifacts'), log_file=str(output/'logs/vllm.log'),
        port=port, capture_full_routes=True, bulk_route_capture=True,
        tensor_parallel_size=protocol.get('tp', 2), cuda_visible_devices=protocol.get('devices', '0,1'), startup_timeout=1800,
        max_num_seqs=4, max_num_batched_tokens=8192, gpu_memory_utilization=.6)
    try:
        service.start()
        for phase, checkpoint in protocol['checkpoints'].items():
            publish(checkpoint)
            identity = checkpoint_identity(checkpoint)
            primary = json.loads((output/phase/'cases.json').read_text())['cases']
            supplement = (json.loads((output/'equal_diagonal_paraphrase'/phase/'cases.json').read_text())['cases']
                          if has_supplement else primary)
            records = []
            for ci, (case, extra_case) in enumerate(zip(primary, supplement, strict=True)):
                if case['row'] != extra_case['row']:
                    raise ValueError('primary/supplement context differs')
                variants = dict(case['variants'])
                for name in (['left_paraphrase', 'right_paraphrase'] if has_supplement else []):
                    variants['supp_'+name] = extra_case['variants'][name]
                saved = {}
                for name, completion in variants.items():
                    ids = tokenizer.encode(completion, add_special_tokens=False)
                    if tokenizer.decode(ids) != completion:
                        raise ValueError('fixed instruction token roundtrip differs')
                    payload = {'model': 'Qwen3.6-35B', 'messages': case['row']['slow_messages'],
                        'temperature': 0.0, 'max_tokens': len(ids)+20,
                        'chat_template_kwargs': {'enable_thinking': False},
                        'vllm_xargs': {'action_head_forced_completion_ids': ids}}
                    request_id = request_queries(payload, 16)
                    response = http.post_json(f'http://127.0.0.1:{port}/v1/chat/completions', payload, 1200)
                    clean, tail = strip_query_tail(response['choices'][0]['message']['content'], query_count=16)
                    if clean != completion or tail != 16:
                        atomic_json(output/phase/f'failed_{ci}_{name}.json', response)
                        raise RuntimeError(f'fixed instruction/tail mismatch: {phase}/{ci}/{name}')
                    path = output/'online_artifacts'/f'{request_id}.safetensors'
                    deadline = time.monotonic()+30
                    while not path.exists() and time.monotonic() < deadline: time.sleep(.1)
                    artifact = load_query_routing_artifact(path)
                    artifact.validate(query_count=16, require_full=True, hidden_size=config['hidden_size'],
                        layer_count=config['num_hidden_layers'], top_k=config['num_experts_per_tok'])
                    if artifact.all_input_ids[-16-len(ids):-16].tolist() != ids:
                        raise ValueError('worker did not execute the specified instruction token ids')
                    saved[name] = artifact.query_hidden.unsqueeze(0).numpy()
                    records.append({'case_index': ci, 'variant': name, 'request_id': request_id,
                        'artifact': str(path), 'completion': clean, 'tail': tail,
                        'input_tokens': artifact.all_input_ids.numel(), 'forced_token_ids_verified': True})
                    atomic_json(output/phase/'online_requests.json', {'checkpoint': identity, 'requests': records})
                    print(json.dumps({'phase':phase,'case':ci,'variant':name,'captured':len(records)}),flush=True)
                np.savez_compressed(output/phase/f'case_{ci}_queries.npz',
                                    **{name:saved[name] for name in case['variants']})
                if has_supplement:
                    np.savez_compressed(output/'equal_diagonal_paraphrase'/phase/f'case_{ci}_queries.npz',
                        **{name:saved.get('supp_'+name, saved[name]) for name in extra_case['variants']})
            total_requests += len(records)
            for suffix in (['', 'equal_diagonal_paraphrase'] if has_supplement else ['']):
                evaluation = output/suffix/phase
                atomic_json(evaluation/'online_capture_completed.json', {'ok':True,'checkpoint':identity,
                    'cases_sha256':sha256_file(evaluation/'cases.json'),'online_requests':str(output/phase/'online_requests.json')})
    finally:
        service.stop()
    atomic_json(output/'capture_completed.json', {'ok':True, 'requests':total_requests})


def evaluate(output):
    from cast_v43_prefix_pilot import infer, compare
    if not json.loads((output/'capture_completed.json').read_text())['ok']:
        raise RuntimeError('vLLM capture incomplete')
    protocol = json.loads((output/'protocol.json').read_text())
    for suffix in ['', 'equal_diagonal_paraphrase']:
        for phase, checkpoint in protocol['checkpoints'].items():
            infer(output/suffix, phase, checkpoint, query_backend='vllm')
        compare(output/suffix)
    atomic_json(output/'completed.json', {'ok':True,'backend':'vllm', 'cases':6,'requests':96})


def summarize(output):
    import numpy as np
    import torch
    from action_head_rl.artifact import load_query_routing_artifact
    if not json.loads((output/'completed.json').read_text())['ok']:
        raise RuntimeError('online evaluation incomplete')
    result = {'source_hf': str(SOURCE), 'online': str(output), 'comparisons': {},
              'same_cases_and_seeds': True, 'identical_online_instruction_repeats': True}
    records = [json.loads((output/phase/'online_requests.json').read_text())['requests']
               for phase in ['eval_before','eval_after']]
    changed = 0
    for before, after in zip(*records, strict=True):
        if (before['case_index'],before['variant'],before['completion']) != (after['case_index'],after['variant'],after['completion']):
            raise ValueError('online before/after request intervention differs')
        a,b = [load_query_routing_artifact(r['artifact']) for r in [before,after]]
        if not torch.equal(a.all_input_ids,b.all_input_ids):
            raise ValueError('online before/after actual input tokens differ')
        changed += int(not torch.equal(a.query_hidden,b.query_hidden))
    if not changed: raise ValueError('no online query changed after checkpoint update')
    result['online_before_after_input_tokens_equal'] = True
    result['requests_with_changed_query_after_checkpoint_update'] = changed
    for suffix in ['', 'equal_diagonal_paraphrase']:
        sets = {}
        for phase in ['eval_before', 'eval_after']:
            hf, online = SOURCE/suffix/phase, output/suffix/phase
            hc, oc = [json.loads((p/'cases.json').read_text()) for p in [hf,online]]
            if hc != oc: raise ValueError('HF/vLLM cases differ')
            for ci, case in enumerate(hc['cases']):
                ha, oa = [json.loads((p/f'case_{ci}_actions.json').read_text())['variants'] for p in [hf,online]]
                if {k:[d['seed'] for d in v] for k,v in ha.items()} != {k:[d['seed'] for d in v] for k,v in oa.items()}:
                    raise ValueError('HF/vLLM seeds differ')
                with np.load(online/f'case_{ci}_queries.npz') as q:
                    if not np.array_equal(q['original'],q['original_repeat']):
                        raise ValueError('identical online query requests differ')
            sets[phase] = {}
            for backend, path in [('hf',hf), ('vllm',online)]:
                score = json.loads((path/'prefix_score.json').read_text())
                if score['gate_defined_before_inference'] != json.loads((hf/'prefix_score.json').read_text())['gate_defined_before_inference']:
                    raise ValueError('scoring thresholds differ')
                summary = {'aggregate':score['aggregate'], 'direction_signs':{}}
                for split in ['train','validation']:
                    for variant in ['', '_paraphrase']:
                        values=[d for c in score['cases'] if c['split']==split
                            for side in ['left','right'] for d in c['conditions'][side+variant]]
                        summary['direction_signs'][split+variant] = {
                            'correct':sum(d['signed_prefix_y_m']>0 for d in values), 'total':len(values),
                            'collision_free':sum(d['static_replay_free'] for d in values),
                            'minimum_signed_y_m':min(d['signed_prefix_y_m'] for d in values)}
                sets[phase][backend] = summary
        result['comparisons'][suffix or 'primary'] = sets
    atomic_json(output/'hf_vllm_comparison.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare','capture','evaluate','summarize'])
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(ART): parser.error('output must be on the project data disk')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'logs').mkdir(exist_ok=True)
    {'prepare':prepare,'capture':capture,'evaluate':evaluate,'summarize':summarize}[args.mode](args.output)
