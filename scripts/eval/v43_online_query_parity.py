#!/usr/bin/env python3
"""Small production-vLLM capture followed by HF replay of the exact tool text."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'verl'), str(REPO / 'vln_demo/src'),
               str(REPO / 'scripts/eval'), str(REPO / 'verl/scripts/eval/qwen36_4query_ckpt_sweep')]
from common import atomic_json
from action_head_d16_weight_sync_probe import import_probe

ART = Path('/data/zerovln/artifacts')
SOURCE = ART / 'cast-v43-prefix-formal-local-20260915-r1'
CHECKPOINT = SOURCE / 'checkpoints/global_step_019706'
MODEL = '/opt/external/user-e7a01947/Model/Qwen3.6-35B-A3B'
DEFAULT_OUTPUT = ART / 'v43-query-vllm-hf-parity-20260915-r1'


def capture(output):
    from action_head_rl.vllm_service import VllmService
    from action_head.vllm_bridge import request_queries, strip_query_tail
    from action_head_rl.artifact import load_query_routing_artifact
    http = import_probe('parity_http', REPO / 'tmp/query_parity_probe_runtime/capture_vllm.py')
    source = json.loads((SOURCE / 'eval_after/cases.json').read_text())['cases']
    model_config = json.loads((Path(MODEL) / 'config.json').read_text())['text_config']
    indices = [0, 2, 5, 0]
    atomic_json(output / 'protocol.json', {
        'source_checkpoint': str(CHECKPOINT), 'slow_model': MODEL, 'query_count': 16,
        'case_indices': indices, 'last_request_repeats_first': True,
        'completion_policy': 'vLLM generates freely; HF replays the exact returned tool text',
        'thresholds': {'query_cosine_min': .99, 'query_relative_l2_max': .15,
                       'per_slot_cosine_min': .99, 'prefix_ade_m_max': .01, 'prefix_fde_m_max': .02},
        'threshold_origin': 'query thresholds reused from archived parity/compare.py; action thresholds diagnostic only',
        'scope': 'single-request scheduling, TP2; no closed-loop success claim',
    })
    service = VllmService(repo=str(REPO), model=MODEL,
        query_checkpoint=str(CHECKPOINT / 'dualvln_query/dualvln_query_embeddings.pt'),
        query_cache=str(output / 'online_artifacts'), log_file=str(output / 'logs/vllm.log'),
        port=28916, capture_full_routes=True, bulk_route_capture=True,
        max_num_seqs=4, max_num_batched_tokens=8192, gpu_memory_utilization=.6,
        tensor_parallel_size=2, cuda_visible_devices='0,1', startup_timeout=1800)
    records = []
    try:
        service.start()
        for ri, ci in enumerate(indices):
            row = source[ci]['row']
            payload = {'model': 'Qwen3.6-35B', 'messages': row['slow_messages'],
                       'temperature': 0.0, 'max_tokens': 512,
                       'chat_template_kwargs': {'enable_thinking': False}}
            request_id = request_queries(payload, 16)
            response = http.post_json('http://127.0.0.1:28916/v1/chat/completions', payload, 600)
            clean, tail = strip_query_tail(response['choices'][0]['message']['content'], query_count=16)
            record = {'case_index': ci, 'request_index': ri, 'row': row,
                      'request_id': request_id, 'payload': payload, 'response': response,
                      'completion': clean, 'query_tail_count': tail}
            atomic_json(output / f'request_{ri}.json', record)
            if tail != 16:
                raise RuntimeError(f'request {ri} produced no complete 16-query tool call: {clean}')
            path = output / 'online_artifacts' / (request_id + '.safetensors')
            deadline = time.monotonic() + 30
            while not path.exists() and time.monotonic() < deadline:
                time.sleep(.1)
            artifact = load_query_routing_artifact(path)
            artifact.validate(query_count=16, require_full=True,
                hidden_size=model_config['hidden_size'], layer_count=model_config['num_hidden_layers'],
                top_k=model_config['num_experts_per_tok'])
            record['artifact'] = str(path)
            records.append(record)
            atomic_json(output / 'captured.json', {'records': records})
            print(json.dumps({'captured': ri, 'case_index': ci, 'tail': tail,
                              'tokens': artifact.all_input_ids.numel(), 'completion': clean}), flush=True)
    finally:
        service.stop()
    atomic_json(output / 'capture_completed.json', {'ok': True, 'requests': len(records)})


def replay(output, exact_online_prefix=False):
    import numpy as np
    import torch
    from scipy.optimize import linear_sum_assignment
    from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration
    from models.dualvln_query_encoder import DualVlnQueryEncoder
    from models.qwen36_fast_slow_action_head import Qwen36FastSlowActionHead
    from sft.joint_fm_head_sft import FrozenQwenQueryForward, build_slow_inputs
    from sft.fla_l2norm_dynamic_nb import maybe_install_l2norm_dynamic_nb_patch
    from action_head_rl.artifact import load_query_routing_artifact
    from action_head.policy import TrajectoryTransform
    from openloop_eval import Q01, Q99, trajectory_metrics
    metrics = import_probe('archived_query_metrics', REPO / 'tmp/query_parity_probe_runtime/compare.py').metrics
    if not json.loads((output / 'capture_completed.json').read_text())['ok']:
        raise RuntimeError('online capture is incomplete')
    maybe_install_l2norm_dynamic_nb_patch()
    torch.set_grad_enabled(False)
    torch.manual_seed(20260915)
    device = torch.device('cuda:0'); dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(MODEL)
    qwen = Qwen3_5MoeForConditionalGeneration.from_pretrained(MODEL, dtype=dtype,
        device_map='auto', low_cpu_mem_usage=True, attn_implementation='sdpa').eval()
    query = DualVlnQueryEncoder.from_pretrained(CHECKPOINT / 'dualvln_query').to(device=device, dtype=dtype).eval()
    frozen = FrozenQwenQueryForward(qwen, query)
    head = Qwen36FastSlowActionHead.from_pretrained(CHECKPOINT / 'action_head').to(device=device, dtype=dtype).eval()
    transform = TrajectoryTransform.from_quantiles(Q01, Q99, flip_y=True, flip_yaw=True)
    records = json.loads((output / 'captured.json').read_text())['records']; results = []
    output_prefix = 'exact_' if exact_online_prefix else ''
    for record in records:
        ri = record['request_index']; row = record['row']
        online = load_query_routing_artifact(record['artifact'])
        inputs = build_slow_inputs(processor, row['slow_messages'], record['completion'], device)
        if exact_online_prefix:
            # Preserve the same image tensors/order, but replay captured online
            # text/image placement instead of assuming chat renderers agree.
            ids = online.all_input_ids[:-16].unsqueeze(0).to(device)
            image_id = qwen.config.image_token_id
            if int((ids == image_id).sum()) != int((inputs['input_ids'] == image_id).sum()):
                raise RuntimeError('online/HF image token counts differ')
            inputs['input_ids'] = ids
            inputs['attention_mask'] = torch.ones_like(ids)
            if 'mm_token_type_ids' in inputs:
                inputs['mm_token_type_ids'] = (ids == image_id).long()
        expected_ids = inputs['input_ids'][0].cpu()
        prefix_equal = torch.equal(expected_ids, online.all_input_ids[:-16])
        positions_equal = torch.equal(online.query_positions, torch.arange(len(expected_ids), len(expected_ids)+16))
        hf = frozen.forward_full(inputs)[0].float().cpu()
        value = online.query_hidden
        similarity = torch.nn.functional.normalize(hf, dim=-1) @ torch.nn.functional.normalize(value, dim=-1).T
        _, assignment = linear_sum_assignment(-similarity.numpy())
        overall = metrics(value, hf)
        slot = [metrics(value[i], hf[i]) for i in range(16)]
        with np.load(row['sample']) as sample:
            reference = torch.from_numpy(sample['reference_visual_tokens']).unsqueeze(0).to(device=device,dtype=dtype)
            current = torch.from_numpy(sample['current_visual_tokens']).unsqueeze(0).to(device=device,dtype=dtype)
        kwargs = {'reference_visual_tokens':reference,'current_visual_tokens':current,
                  'reference_visual_mask':torch.ones(reference.shape[:2],device=device,dtype=torch.bool),
                  'current_visual_mask':torch.ones(current.shape[:2],device=device,dtype=torch.bool)}
        trajectories = {}; states = {}
        for name, q in [('hf',hf),('vllm',value)]:
            q = q.unsqueeze(0).to(device=device,dtype=dtype)
            states[name+'_adapter'] = head.slow_query_adapter(q).float().cpu()
            states[name+'_fusion'] = head.encode_condition(q,**kwargs).float().cpu()
            predictions = []
            for draw in range(4):
                gen = torch.Generator(device=device).manual_seed(20260914+record['case_index']*100+draw)
                noise = torch.randn((1,head.config.horizon,head.config.action_dim),device=device,dtype=dtype,generator=gen)
                predictions.append(transform.apply(head.sample(query_hidden_states=q,initial_noise=noise,**kwargs))[0].float().cpu().numpy())
            trajectories[name] = np.stack(predictions)
        action_metrics = [trajectory_metrics(a,b,16) for a,b in zip(trajectories['vllm'],trajectories['hf'])]
        action_mean = {key:float(np.mean([m[key] for m in action_metrics])) for key in action_metrics[0]}
        result = {'request_index':ri,'case_index':record['case_index'],'scene':row['source_scene'],
                  'exact_online_prefix':exact_online_prefix,
                  'prefix_token_ids_equal':prefix_equal,'query_positions_equal':positions_equal,
                  'prefix_lengths':[len(expected_ids),len(online.all_input_ids)-16],
                  'query':overall,'per_slot':slot,'best_slot_assignment':assignment.tolist(),
                  'adapter':metrics(states['vllm_adapter'],states['hf_adapter']),
                  'fusion':metrics(states['vllm_fusion'],states['hf_fusion']),
                  'actions_mean':action_mean,'actions_per_draw':action_metrics}
        result['query_gate'] = bool(prefix_equal and positions_equal and overall['cosine']>=.99
            and overall['relative_l2']<=.15 and min(x['cosine'] for x in slot)>=.99
            and assignment.tolist()==list(range(16)))
        result['action_gate'] = bool(all(x['prefix_ade_m']<=.01 and x['prefix_fde_m']<=.02 for x in action_metrics))
        np.savez_compressed(output/f'{output_prefix}comparison_{ri}.npz', hf_query=hf.numpy(),vllm_query=value.numpy(),
                            **{name+'_actions':values for name,values in trajectories.items()})
        results.append(result); atomic_json(output/f'{output_prefix}comparison.json',{'requests':results})
        print(json.dumps(result,ensure_ascii=False),flush=True)
    atomic_json(output/f'{output_prefix}completed.json',{'ok':True,'query_gate':all(x['query_gate'] for x in results),
        'action_gate':all(x['action_gate'] for x in results),'requests':len(results)})


def audit_template(output):
    import torch
    from transformers import AutoProcessor
    from vllm.config import ModelConfig
    from vllm.entrypoints.chat_utils import parse_chat_messages
    from sft.joint_fm_head_sft import build_slow_inputs, _slow_template_messages
    from action_head_rl.artifact import load_query_routing_artifact
    cfg = ModelConfig(model=MODEL, max_model_len=8192,
                      allowed_local_media_path='/data/zerovln')
    processor = AutoProcessor.from_pretrained(MODEL)
    rows = []
    for record in json.loads((output/'captured.json').read_text())['records'][:3]:
        messages = record['row']['slow_messages']
        _, images = _slow_template_messages(messages)
        hf = build_slow_inputs(processor, messages, record['completion'], torch.device('cpu'))['input_ids'][0]
        online = load_query_routing_artifact(record['artifact']).all_input_ids[:-16]
        result = {'request': record['request_index']}
        for fmt in ['string', 'openai']:
            conversation, _, _ = parse_chat_messages(messages, cfg, fmt)
            prompt = processor.apply_chat_template(conversation, tokenize=False,
                add_generation_prompt=True, enable_thinking=False) + record['completion']
            ids = processor(text=[prompt], images=images, return_tensors='pt')['input_ids'][0]
            result[fmt] = {'length': len(ids), 'equals_current_hf': torch.equal(ids, hf),
                           'equals_captured_vllm': torch.equal(ids, online)}
        rows.append(result)
    atomic_json(output/'template_audit.json', rows)
    print(json.dumps(rows), flush=True)


def summarize(output):
    import numpy as np
    captured = json.loads((output/'captured.json').read_text())['records']
    summary = {'independent_states': 3, 'requests': 4,
               'repeat_completion_equal': captured[0]['completion'] == captured[3]['completion']}
    for prefix in ['', 'exact_']:
        rows = json.loads((output/f'{prefix}comparison.json').read_text())['requests']
        unique = rows[:3]
        with np.load(output/f'{prefix}comparison_0.npz') as first, np.load(output/f'{prefix}comparison_3.npz') as repeat:
            repeat_equal = {k: bool(np.array_equal(first[k], repeat[k])) for k in first.files}
        summary[prefix or 'original'] = {
            'query_gate': all(r['query_gate'] for r in rows),
            'action_gate': all(r['action_gate'] for r in rows),
            'prefix_ids_equal': all(r['prefix_token_ids_equal'] for r in rows),
            'query_cosine_range': [min(r['query']['cosine'] for r in unique), max(r['query']['cosine'] for r in unique)],
            'query_relative_l2_range': [min(r['query']['relative_l2'] for r in unique), max(r['query']['relative_l2'] for r in unique)],
            'per_slot_cosine_min': min(s['cosine'] for r in unique for s in r['per_slot']),
            'identity_slot_assignment': all(r['best_slot_assignment'] == list(range(16)) for r in rows),
            'prefix_ade_mean_m': float(np.mean([r['actions_mean']['prefix_ade_m'] for r in unique])),
            'prefix_ade_max_draw_m': max(d['prefix_ade_m'] for r in unique for d in r['actions_per_draw']),
            'prefix_fde_max_draw_m': max(d['prefix_fde_m'] for r in unique for d in r['actions_per_draw']),
            'full_ade_mean_m': float(np.mean([r['actions_mean']['full_ade_m'] for r in unique])),
            'repeat_arrays_bitwise_equal': repeat_equal,
        }
    audit = output/'template_audit.json'
    if audit.exists(): summary['template_audit'] = json.loads(audit.read_text())
    atomic_json(output/'final_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['capture','replay','audit-template','summarize'])
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--exact-online-prefix',action='store_true');args=parser.parse_args()
    if not args.output.resolve().is_relative_to(ART):parser.error('output must be in project artifact root')
    args.output.mkdir(exist_ok=True,parents=True)
    if args.mode == 'capture': capture(args.output)
    elif args.mode == 'audit-template': audit_template(args.output)
    elif args.mode == 'summarize': summarize(args.output)
    else: replay(args.output, args.exact_online_prefix)
