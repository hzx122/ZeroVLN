#!/usr/bin/env python3
"""Small, paired current-instruction intervention; reuse faithful SFT/eval forward.

This is a sensitivity diagnostic, not a CAST success benchmark: counterfactual
branches have no certified trajectory labels and are not used for training.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
ARTIFACTS = Path('/data/zerovln/artifacts')
EXPR = 'v43-step19450-instruction-control-diagnostic-20260914-r1'
TRAIN = 'action-head-v43-official-gvln-d16-l2normfix-from1000-20260912-r5'
CHECKPOINT = ARTIFACTS / TRAIN / 'checkpoints' / TRAIN / 'global_step_019450'
SLOW_MODEL = '/opt/external/user-e7a01947/Model/Qwen3.6-35B-A3B'
SOURCE = ARTIFACTS / 'sage_full_status_analysis_20260914/cast_case_audit/sampled_anchors.json'
EVAL = REPO / 'verl/scripts/eval/qwen36_4query_ckpt_sweep'
sys.path[:0] = [str(REPO), str(REPO / 'verl'), str(EVAL)]


def write_json(path, value):
    from common import atomic_json
    atomic_json(Path(path), value)


def prepare(output):
    from action_head import protocol_v4_3 as protocol
    rows = json.loads(SOURCE.read_text())['anchors']
    cases = []
    for i in (3, 4, 11):
        row = rows[i]
        protocol.assert_protocol_identity(row)
        original = row['slow_completion']
        call = json.loads(original.split('<tool_call>')[1].split('</tool_call>')[0])
        variants = {'original': original, 'original_repeat': original}
        for side in ('left', 'right'):
            for paraphrase in (False, True):
                branch = copy.deepcopy(call)
                first = (f'begin by moving diagonally forward to the {side}' if paraphrase
                         else f'first move forward and slightly to the {side}')
                branch['arguments']['plan'] = [first, 'then continue toward ' + call['arguments']['target']]
                variants[side + ('_paraphrase' if paraphrase else '')] = (
                    '<tool_call>\n' + json.dumps(branch, ensure_ascii=False, separators=(',', ':'))
                    + '\n</tool_call>')
        cases.append({'source_index': i, 'row': row, 'variants': variants})
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'cases.json', {
        'source': str(SOURCE), 'cases': cases,
        'scope': '3 visually inspected training-source anchors; first fast window only; '
                 'same task/history/images/target/where, current plan intervention only. '
                 'Short-prefix steering plausibility inspected visually, not certified by replay. '
                 'No counterfactual ground truth, held-out generalization, or closed-loop SR claims.',
    })


def submit(output):
    sys.path[:0] = [str(WORKSPACE / 'dlc_cli'), str(REPO / 'verl/dlc')]
    # Tea SDK ignores this environment's no_proxy for the local credential URI.
    # Restrict the workaround to this submission process, never persist credentials.
    for key in list(os.environ):
        if key.lower() in ('http_proxy', 'https_proxy', 'all_proxy'):
            os.environ.pop(key)
    from _dlc_common import get_dlc_client, shell_quote, user_vpc, IMAGE, PRIORITY, WORKSPACE as PROFILE
    from workspace_config import resolve_workspace, build_data_sources
    from alibabacloud_pai_dlc20201203.models import CreateJobRequest, CreateJobRequestDataSources, JobSpec, ResourceConfig
    q = shell_quote
    command = f'''#!/usr/bin/env bash
set -euo pipefail
mkdir -p {q(str(output / 'logs'))}
exec >{q(str(output / 'logs/run.log'))} 2>&1
RUNTIME_ROOT=/tmp/{EXPR}/$(hostname)
export TMPDIR="$RUNTIME_ROOT/tmp" XDG_CACHE_HOME="$RUNTIME_ROOT/xdg" HF_HOME="$RUNTIME_ROOT/hf"
export TRITON_CACHE_DIR="$RUNTIME_ROOT/triton" TORCHINDUCTOR_CACHE_DIR="$RUNTIME_ROOT/inductor"
export CUDA_CACHE_PATH="$RUNTIME_ROOT/cuda" PYTORCH_KERNEL_CACHE_PATH="$RUNTIME_ROOT/kernels"
export TORCH_EXTENSIONS_DIR="$RUNTIME_ROOT/extensions" MPLCONFIGDIR="$RUNTIME_ROOT/mpl" NUMBA_CACHE_DIR="$RUNTIME_ROOT/numba"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$PYTORCH_KERNEL_CACHE_PATH" "$TORCH_EXTENSIONS_DIR" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PPU_SDK=/usr/local/PPU_SDK PPU_HOME=/usr/local/PPU_SDK
export CUDA_PATH=/usr/local/PPU_SDK/CUDA_SDK CUDA_HOME=/usr/local/PPU_SDK/CUDA_SDK
export LD_LIBRARY_PATH="/opt/accl-p:/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/lib:/opt/navigation-runtime/venv/lib:/usr/local/lib:${{LD_LIBRARY_PATH:-}}"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH={q(str(REPO))}:{q(str(REPO / 'verl'))}:{q(str(REPO / 'vln_demo/src'))}
cd {q(str(REPO))}
exec /usr/local/bin/python3.12 {q(str(Path(__file__).resolve()))} --run --output {q(str(output))} --checkpoint {q(str(CHECKPOINT))}
'''
    (output / 'launch.sh').write_text(command)
    profile = resolve_workspace(PROFILE)
    request = CreateJobRequest(
        display_name=EXPR, job_type='PyTorchJob', user_command=command,
        data_sources=build_data_sources(profile, CreateJobRequestDataSources),
        resource_id=profile.resource_id, workspace_id=profile.workspace_id,
        user_vpc=user_vpc(), priority=PRIORITY,
        job_specs=[JobSpec(type='Worker', pod_count=1, image=IMAGE,
                           resource_config=ResourceConfig(cpu='24', memory='224Gi', shared_memory='224Gi', gpu='2'))])
    job_id = get_dlc_client().create_job(request).body.job_id
    write_json(output / 'submission.json', {'job_id': job_id, 'expr': EXPR, 'output': str(output), 'checkpoint': str(CHECKPOINT)})
    print(job_id, flush=True)
    with (REPO / 'docs/实验记录.md').open('a') as f:
        f.write(f'\n\n## 2026-09-14 v4.3 当前指令到 query/action head 三用例诊断启动\n\n'
                f'- EXPR_NAME：`{EXPR}`；DLC：`{job_id}`；单节点 2 GPU。\n'
                f'- 启动：`python3 {Path(__file__).resolve()} --submit --output {output}`。\n'
                f'- 实际容器命令：`bash {output / "launch.sh"}`；配置：`{output / "cases.json"}`。\n'
                f'- 模型：`{CHECKPOINT}`；冻结大脑：`{SLOW_MODEL}`。\n'
                f'- 数据：`{SOURCE}` 的第 3/4/11 个 anchor，固定上下文与视觉，仅改变当前 plan。\n'
                f'- 产出：`{output}`；日志：`{output / "logs/run.log"}`；TensorBoard：无（纯推理）。\n'
                '- 同种子四次采样；原指令重复、左右反事实、同义改写、零 query 和跨场景 query 控制。'
                '保存 query/adapter/fusion 和动作；没有反事实 GT，不作为 CAST 成功率或泛化结论。\n')


def run(output, query_backend="hf"):
    (output / 'completed.json').unlink(missing_ok=True)
    import numpy as np
    import torch
    from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration
    from models.dualvln_query_encoder import DualVlnQueryEncoder
    from models.qwen36_fast_slow_action_head import Qwen36FastSlowActionHead
    from sft.joint_fm_head_sft import FrozenQwenQueryForward, build_slow_inputs
    from action_head.policy import TrajectoryTransform
    from openloop_eval import Q01, Q99, trajectory_metrics
    from common import checkpoint_identity
    torch.set_grad_enabled(False)
    torch.manual_seed(20260914)
    identity = checkpoint_identity(CHECKPOINT)
    data = json.loads((output / 'cases.json').read_text())
    write_json(output / 'protocol.json', {'checkpoint': identity, 'slow_model': SLOW_MODEL,
        'draws': 4, 'dtype': 'bfloat16', 'query': ('recompute forward_full for each completion' if query_backend == 'hf'
                  else 'vLLM online autoregressive forced completion plus 16-query tail'),
        'coordinate': 'transformed deltas in fixed chunk-start frame, cumulatively summed; y/yaw left positive',
        'scope': data['scope']})
    device = torch.device('cuda:0'); dtype = torch.bfloat16
    queries = []
    if query_backend == 'hf':
        print('loading Qwen', flush=True)
        processor = AutoProcessor.from_pretrained(SLOW_MODEL, trust_remote_code=True)
        qwen = Qwen3_5MoeForConditionalGeneration.from_pretrained(SLOW_MODEL, dtype=dtype,
            device_map='auto', low_cpu_mem_usage=True, attn_implementation='sdpa', trust_remote_code=True).eval()
        query = DualVlnQueryEncoder.from_pretrained(identity['query_dir']).to(device=device, dtype=dtype).eval()
        frozen = FrozenQwenQueryForward(qwen, query)
    elif query_backend == 'vllm':
        import hashlib
        capture = json.loads((output / 'online_capture_completed.json').read_text())
        if (not capture['ok'] or capture['checkpoint'] != identity
                or capture['cases_sha256'] != hashlib.sha256((output / 'cases.json').read_bytes()).hexdigest()):
            raise ValueError('online query capture does not match checkpoint/evaluation inputs')
    else:
        raise ValueError(f'unknown query backend: {query_backend}')
    head = Qwen36FastSlowActionHead.from_pretrained(identity['action_head_dir']).to(device=device, dtype=dtype).eval()
    transform = TrajectoryTransform.from_quantiles(Q01, Q99, flip_y=True, flip_yaw=True)
    for ci, case in enumerate(data['cases']):
        if query_backend == 'vllm':
            with np.load(output / f'case_{ci}_queries.npz') as arrays:
                saved = dict(arrays)
            if set(saved) != set(case['variants']) or any(
                    v.shape != (1, 16, 2048) or not np.isfinite(v).all() for v in saved.values()):
                raise ValueError(f'invalid online query variants for case {ci}')
        else:
            saved = {}
            for name, completion in case['variants'].items():
                messages = case.get('variant_messages', {}).get(name, case['row']['slow_messages'])
                inputs = build_slow_inputs(processor, messages, completion, device)
                value = frozen.forward_full(inputs)
                saved[name] = value.float().cpu().numpy()
                print(f'query {ci} {name} tokens={inputs["input_ids"].shape[-1]} norm={value.float().norm().item():.4f}', flush=True)
                del inputs, value
            np.savez_compressed(output / f'case_{ci}_queries.npz', **saved)
        queries.append(saved)
    for ci, case in enumerate(data['cases']):
        row = case['row']
        with np.load(row['sample']) as bundle:
            ref = torch.from_numpy(bundle['reference_visual_tokens']).to(device=device, dtype=dtype).unsqueeze(0)
            cur = torch.from_numpy(bundle['current_visual_tokens']).to(device=device, dtype=dtype).unsqueeze(0)
            gt = transform.apply(torch.from_numpy(bundle['target_actions']).to(device=device, dtype=dtype).unsqueeze(0))[0].float().cpu().numpy()
        condition_args = dict(reference_visual_tokens=ref, current_visual_tokens=cur,
            reference_visual_mask=torch.ones(ref.shape[:2], device=device, dtype=torch.bool),
            current_visual_mask=torch.ones(cur.shape[:2], device=device, dtype=torch.bool))
        variants = dict(queries[ci])
        if data.get('include_query_controls', True):
            variants['zero_query'] = np.zeros_like(variants['original'])
            variants['other_scene_query'] = queries[(ci + 1) % len(queries)]['original']
            # Deliberate off-manifold sensitivity control, never a semantic repair.
            variants['right_delta_x4'] = variants['left'] + 4 * (variants['right'] - variants['left'])
        saved = {'ground_truth': gt}
        records = {}
        for name, value in variants.items():
            qvalue = torch.from_numpy(value).to(device=device, dtype=dtype)
            adapted = head.slow_query_adapter(qvalue)
            fused = head.encode_condition(query_hidden_states=qvalue, **condition_args)
            saved[name + '_adapter'] = adapted.float().cpu().numpy()
            saved[name + '_fusion'] = fused.float().cpu().numpy()
            predictions = []
            draws = []
            for draw in range(4):
                seed = 20260914 + ci * 100 + draw
                generator = torch.Generator(device=device).manual_seed(seed)
                noise = torch.randn((1, head.config.horizon, head.config.action_dim), device=device, dtype=dtype, generator=generator)
                pred = transform.apply(head.sample(query_hidden_states=qvalue, initial_noise=noise, **condition_args))[0].float().cpu().numpy()
                predictions.append(pred)
                draws.append({'seed': seed, 'prefix4_delta_sum': pred[:4].sum(0).tolist(),
                              'full_delta_sum': pred.sum(0).tolist(),
                              'original_gt_metrics': trajectory_metrics(pred, gt, row['valid_action_steps'])})
            saved[name + '_actions'] = np.stack(predictions)
            records[name] = draws
            print(f'head {ci} {name} prefix4={np.stack(predictions)[:, :4].sum(1).mean(0).tolist()}', flush=True)
        np.savez_compressed(output / f'case_{ci}_head.npz', **saved)
        write_json(output / f'case_{ci}_actions.json', {'anchor': row['slow_anchor_id'], 'variants': records})
    write_json(output / 'completed.json', {'ok': True, 'cases': len(queries), 'draws': 4})
    print('diagnostic completed', flush=True)


def analyze(output):
    import itertools
    import numpy as np
    if not (output / 'completed.json').exists():
        raise RuntimeError('inference is incomplete')
    data = json.loads((output / 'cases.json').read_text())
    pairs = [('original', 'original_repeat'), ('left', 'right'),
             ('left_paraphrase', 'right_paraphrase'), ('left', 'left_paraphrase'),
             ('right', 'right_paraphrase'), ('original', 'left'), ('original', 'right'),
             ('original', 'zero_query'), ('original', 'other_scene_query'),
             ('left', 'right_delta_x4')]

    def representation(a, b):
        a, b = a.astype('float64'), b.astype('float64')
        norm = np.linalg.norm(a)
        per_token = 1 - (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12)
        return {'relative_l2': float(np.linalg.norm(a-b) / max(norm, 1e-12)),
                'cosine_distance': float(1 - np.sum(a*b) / max(norm*np.linalg.norm(b), 1e-12)),
                'token_cosine_distance_min_mean_max': [float(per_token.min()), float(per_token.mean()), float(per_token.max())]}

    def actions(a, b):
        ap, bp = np.cumsum(a, axis=-2), np.cumsum(b, axis=-2)
        dist = np.linalg.norm(ap[..., :2] - bp[..., :2], axis=-1)
        return {'prefix4_ade_m': float(dist[..., :4].mean()),
                'prefix4_fde_m': float(dist[..., 3].mean()),
                'full_ade_m': float(dist.mean()), 'full_fde_m': float(dist[..., -1].mean()),
                'prefix4_yaw_abs_diff_deg': float(np.degrees(np.abs(ap[..., 3, 2] - bp[..., 3, 2])).mean())}

    result = {'cases': [], 'scope': data['scope'],
              'interpretation': 'No cosine threshold proves absent information. Cross-case contrast alignment '
              'only tests consistency of these particular instruction pairs. Signed output checks are '
              'short-prefix diagnostics without collision/semantic ground-truth certification.'}
    all_representations = []
    for ci, case in enumerate(data['cases']):
        q = dict(np.load(output / f'case_{ci}_queries.npz'))
        h = dict(np.load(output / f'case_{ci}_head.npz'))
        layers = {'query': q,
                  'adapter': {k[:-8]: v for k, v in h.items() if k.endswith('_adapter')},
                  'fusion': {k[:-7]: v for k, v in h.items() if k.endswith('_fusion')}}
        all_representations.append(layers)
        record = {'anchor': case['row']['slow_anchor_id'], 'split': case['row']['split'], 'pairs': {}, 'directions': {}}
        for a, b in pairs:
            value = {layer: representation(tensors[a], tensors[b]) for layer, tensors in layers.items() if a in tensors and b in tensors}
            value['actions'] = actions(h[a + '_actions'], h[b + '_actions'])
            record['pairs'][a + '__' + b] = value
        original = h['original_actions']
        noise_pairs = list(itertools.combinations(range(len(original)), 2))
        record['original_noise_variation'] = actions(
            np.stack([original[a] for a, _ in noise_pairs]), np.stack([original[b] for _, b in noise_pairs]))
        for name in ('original', 'left', 'right', 'left_paraphrase', 'right_paraphrase'):
            prefix = h[name + '_actions'][:, :4].sum(1)
            record['directions'][name] = {
                'prefix4_y_left_m': prefix[:, 1].tolist(),
                'prefix4_yaw_left_deg': np.degrees(prefix[:, 2]).tolist(),
                'prefix4_heading_deg': np.degrees(np.arctan2(prefix[:, 1], prefix[:, 0])).tolist()}
        for a, b in (('left', 'right'), ('left_paraphrase', 'right_paraphrase')):
            ay, by = h[a + '_actions'][:, :4, 1].sum(1), h[b + '_actions'][:, :4, 1].sum(1)
            record['pairs'][a + '__' + b]['direction_diagnostic'] = {
                'left_minus_right_y_m': (ay-by).tolist(),
                'left_more_left_draws': int((ay > by).sum()),
                'both_have_requested_sign_draws': int(((ay > 0) & (by < 0)).sum()),
                'draws': len(ay)}
        result['cases'].append(record)
    # Learn one contrast direction from the OTHER states, evaluate each held-out
    # state's independently phrased contrast. No intercept fitted on test cases.
    result['leave_one_state_out_contrast_alignment'] = {}
    for layer in ('query', 'adapter', 'fusion'):
        scores = []
        for heldout in range(len(all_representations)):
            train = []
            for i, layers in enumerate(all_representations):
                if i != heldout:
                    delta = (layers[layer]['left'] - layers[layer]['right']).astype('float64').ravel()
                    train.append(delta / max(np.linalg.norm(delta), 1e-12))
            direction = np.mean(train, axis=0)
            test = all_representations[heldout][layer]
            delta = (test['left_paraphrase'] - test['right_paraphrase']).astype('float64').ravel()
            scores.append(float(np.dot(delta, direction) / max(np.linalg.norm(delta) * np.linalg.norm(direction), 1e-12)))
        result['leave_one_state_out_contrast_alignment'][layer] = scores
    write_json(output / 'analysis.json', result)
    for case in result['cases']:
        lr = case['pairs']['left__right']
        print(json.dumps({'anchor': case['anchor'], 'query': lr['query'], 'adapter': lr['adapter'],
                          'fusion': lr['fusion'], 'actions': lr['actions'],
                          'direction': lr['direction_diagnostic']}, ensure_ascii=False))


def main():
    global CHECKPOINT
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--submit', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--analyze', action='store_true')
    parser.add_argument('--output', type=Path, default=ARTIFACTS / EXPR)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    CHECKPOINT = args.checkpoint.resolve()
    output = args.output.resolve()
    if not output.is_relative_to(ARTIFACTS):
        parser.error('output must be inside authorized project artifact root')
    if args.analyze:
        analyze(output)
    elif args.run:
        run(output)
    else:
        prepare(output)
        if args.submit:
            submit(output)


if __name__ == '__main__':
    main()
