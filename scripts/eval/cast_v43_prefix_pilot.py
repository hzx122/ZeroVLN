#!/usr/bin/env python3
"""CAST prompt/schema adapter for a bounded, same-state SAGE prefix pilot.

Reuses stored geometric proposals, SAGE exact replay, the official CAST prompt
assets, and the existing recovery-to-SFT label conversion. No atomic policy is
trained; no regenerated history or future observations enter model inputs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'verl'), str(REPO / 'verl/sft'),
               str(REPO / 'vln_demo/src'), str(REPO / 'scripts/eval'),
               str(REPO / 'verl/scripts/eval/qwen36_4query_ckpt_sweep')]
from common import atomic_json, sha256_file

ART = Path('/data/zerovln/artifacts')
DEFAULT_OUTPUT = ART / 'cast-v43-prefix-pilot-20260914-r2'
AUDIT = ART / 'sage_full_status_analysis_20260914/cast_case_audit/sampled_anchors.json'
PRESCREEN = ART / 'cast-v43-full-anchor-prescreen-20260914-r1'
CAST = REPO / 'third_party/CAST'
MODEL = '/opt/external/user-e7a01947/Model/Qwen3.6-35B-A3B'
TEACHER_MIN_SIDE_M = 0.075
EVAL_MIN_SIDE_M = 0.05


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def local_xy(points, pose):
    xy = np.asarray(points, dtype=float)[:, :2] - np.asarray(pose[:2])
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return xy @ np.asarray([[c, -s], [s, c]])


def prepare_geometric_pair(row, candidate, index, dataset, occ=None, strict_training_replay=False):
    """Recheck an existing prescreen proposal with the pilot's exact gates."""
    from cast_case_feasibility_audit import OccMap, _execution_replay_free, RADIUS
    from sage.capability_benchmark import point_at_arc
    if occ is None:
        occ = OccMap(dataset.scene_dir(row['source_scene']), collision_mode='usd_rooms', furniture_mode='none')
    branches = {}
    for b in candidate['branch_pair']:
        y = local_xy([b['prefix_xy']], row['source_pose_xyyaw'])[0, 1]
        if abs(y) < TEACHER_MIN_SIDE_M or b['path_length_m'] < 1.6:
            continue
        ok, replay = _execution_replay_free(occ, b['path_xy'], step_m=0.1, radius_m=RADIUS)
        if ok and replay['collisions'] == 0:
            if strict_training_replay:
                sampled = [row['source_pose_xyyaw'][:2]] + [point_at_arc(b['path_xy'], i*.1) for i in range(1,17)]
                fine_ok, fine_replay = _execution_replay_free(occ, sampled, step_m=.05, radius_m=RADIUS)
                if not fine_ok or fine_replay.get('collisions', 0):
                    continue
            side = 'left' if y > 0 else 'right'
            branches[side] = {**b, 'checked_replay': replay,
                'prefix_local_points': local_xy([point_at_arc(b['path_xy'], a)
                    for a in (0.,0.1,0.2,0.3,0.4)], row['source_pose_xyyaw']).tolist(),
                'prefix_local_xy': local_xy([b['prefix_xy']], row['source_pose_xyyaw'])[0].tolist()}
    if set(branches) != {'left', 'right'}:
        return None
    uid = 'cast-prefix-' + hashlib.sha256(row['slow_anchor_id'].encode()).hexdigest()[:12]
    return {'pair_id': uid, 'source_index': index, 'row': row, 'branches': branches,
            'occ_mask': occ.usd_mask_path, 'occ_version': occ._usd_version}


def prepare(output):
    from cast_case_feasibility_audit import SageSet, OccMap, _execution_replay_free, RADIUS
    from sage.capability_benchmark import point_at_arc
    rows = json.loads(AUDIT.read_text())['anchors']
    dataset = SageSet(split='official_gvln_action_head_p0_v3full827train')
    selected, excluded = [], []
    # Reuse the previously inspected cases. Exclude the outdoor/incomplete
    # render, and allow later gates to reject uncertain indoor grounding.
    for index in (0, 1, 3, 5, 6, 9, 10, 11):
        row = rows[index]
        scene = json.loads((PRESCREEN / 'scenes' / (row['source_scene'] + '.json')).read_text())
        candidate = next((v for v in scene['candidates'] if v['slow_anchor_id'] == row['slow_anchor_id']), None)
        if candidate is None:
            excluded.append({'index': index, 'reason': 'no_geometric_pair'}); continue
        pair = prepare_geometric_pair(row, candidate, index, dataset)
        if pair is None:
            excluded.append({'index': index, 'reason': 'not_opposite_valid_prefixes'}); continue
        selected.append(pair)
    output.mkdir(parents=True, exist_ok=True)
    assets = [CAST / 'cast/data/prompts' / name for name in ('cf_step_prompt.txt', 'filter_step_prompt.txt')]
    assets += [CAST / 'cast/data/schemas' / name for name in ('cf_step_schema.json', 'filter_step_schema.json')]
    atomic_json(output / 'prepared.json', {'pairs': selected, 'excluded': excluded,
        'source': str(AUDIT), 'source_sha256': sha256_file(AUDIT),
        'cast_commit': 'ec7a214e76167e0f844800ea91d6664863a3d9b1',
        'cast_assets_sha256': {str(f): sha256_file(f) for f in assets},
        'teacher_min_side_m': TEACHER_MIN_SIDE_M, 'eval_min_side_m': EVAL_MIN_SIDE_M,
        'language_adapter_version': 2,
        'scope': 'first fast window only; 8 previously inspected train-source scenes at most; '
                 'scene holdout is relative to this pilot fine-tuning, not original SFT; '
                 'CAST prompts/schemas reused, Google batch transport and learned atomic policy replaced; '
                 'static SAGE geometry replay, no new rendered rollout or closed-loop SR'})
    print(json.dumps({'prepared_pairs': len(selected), 'excluded': excluded}), flush=True)


def parse_json(text):
    text = text.rsplit('</think>', 1)[-1].strip()
    start = text.find('{')
    if start < 0:
        raise ValueError('no JSON object')
    return json.JSONDecoder().raw_decode(text[start:])[0]


def only_side(text, side):
    words = set(re.findall(r'\b(left|right)\b', text.lower()))
    return words == {side}


def generate(output, teacher_url=None, cache_only=False):
    import torch
    import jsonschema
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration
    from sft.collect_action_head_recovery import _route_yaws, route_suffix
    from sft.build_fm_nested_bundles import save_npz, parse_tool_call_text
    from action_head import protocol_v4_3 as V43
    from action_head.policy import TrajectoryTransform
    from openloop_eval import Q01, Q99
    from sft.joint_fm_head_sft import JointBundleDataset, joint_collate
    torch.set_grad_enabled(False)
    torch.manual_seed(20260914)
    from sft.fla_l2norm_dynamic_nb import maybe_install_l2norm_dynamic_nb_patch
    patch_audit = maybe_install_l2norm_dynamic_nb_patch()
    atomic_json(output / 'fla_patch.json', patch_audit)
    prepared = json.loads((output / 'prepared.json').read_text())
    if cache_only:
        teacher_url = 'cache-only'
    if teacher_url is None:
        processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
        model = Qwen3_5MoeForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16,
            device_map='auto', low_cpu_mem_usage=True, attn_implementation='sdpa', trust_remote_code=True).eval()
    else:
        from action_head_d16_weight_sync_probe import import_probe
        http = import_probe('cast_teacher_http', REPO/'tmp/query_parity_probe_runtime/capture_vllm.py')
    cf_template = (CAST / 'cast/data/prompts/cf_step_prompt.txt').read_text()
    filter_template = (CAST / 'cast/data/prompts/filter_step_prompt.txt').read_text()
    cf_schema = json.loads((CAST / 'cast/data/schemas/cf_step_schema.json').read_text())
    filter_schema = json.loads((CAST / 'cast/data/schemas/filter_step_schema.json').read_text())
    evidence = output / 'teacher'; evidence.mkdir(exist_ok=True)

    def ask(uid, image, prompt, schema):
        dest = evidence / (uid + '.json')
        full_prompt = prompt + '\nReturn only JSON, without reasoning outside JSON. Schema:\n' + json.dumps(schema)
        digest = hashlib.sha256((full_prompt + str(image) + sha256_file(Path(image))).encode()).hexdigest()
        if dest.exists():
            record = json.loads(dest.read_text())
            if record['prompt_sha256'] != digest: raise ValueError('teacher cache identity changed')
        else:
            if cache_only:
                raise RuntimeError('teacher response missing from recovery cache: ' + str(dest))
            if teacher_url is None:
                messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': full_prompt}]}]
                rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                inputs = processor(text=[rendered], images=[Image.open(image).convert('RGB')], return_tensors='pt').to('cuda:0')
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=384, use_cache=True)
                raw = processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            else:
                response = http.post_json(teacher_url, {'model':'Qwen3.6-35B',
                    'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'file://'+str(image)}},
                        {'type':'text','text':full_prompt}]}], 'temperature':0.0, 'max_tokens':384,
                    'chat_template_kwargs':{'enable_thinking':False}}, 1200)
                raw = response['choices'][0]['message']['content']
            record = {'prompt': full_prompt, 'prompt_sha256': digest, 'image': str(image), 'model': MODEL, 'raw': raw}
            atomic_json(dest, record)
        value = parse_json(record['raw']); jsonschema.validate(value, schema)
        return value

    accepted, audits = [], []
    quotas = prepared.get('split_quotas')
    for pair in prepared['pairs']:
        if quotas and sum(p['planned_split']==pair['planned_split'] for p in accepted) >= quotas[pair['planned_split']]:
            continue
        uid, row = pair['pair_id'], pair['row']
        try:
            prompt = cf_template.format(labels='not reclassified in this adapter; see original local plan',
                instructions=json.dumps({'global_task': row['instruction'], 'original_local_call': row['contract']}),
                curr_atomic_action='approach the original local target', base_actions='adjust left, adjust right', unique_id=uid)
            prompt += ('\nSAGE adaptation: propose at most one adjust left and one adjust right. Both must begin '
                'with translating diagonally forward on the named side of the CURRENT view; do not describe an in-place turn. '
                'The first motion must explicitly say left or right, not a later stage. Ground it in visible free space '
                'or a visible landmark. Keep the global task and the original target/where compatible: these are brief '
                'local route alternatives before continuing the original target. Do not change the global goal. '
                'If one side is not visually plausible, omit it. Do not put distances, coordinates, primitive codes, '
                'or future observations into instructions. '
                'IMPORTANT: each instruction MUST start exactly with "Move diagonally left" or "Move diagonally right" '
                'and then describe the visible free floor/corridor or a compatible landmark. This is a current '
                'ego-motion command, NOT a reference to the left/right part of an object. Avoid selecting a new target. '
                'The next unchanged stage will continue toward the original target. '
                'The official response schema below is authoritative.')
            cf = ask(uid + '-cf', row['current_image'], prompt, cf_schema)
            if cf['unique_id'] != uid: raise ValueError('CF id mismatch')
            proposals = {}
            for v in cf['counterfactual_action_instruction']:
                side = v['counterfactual_action'].replace('adjust ', '')
                instruction = v['counterfactual_instruction'].strip()
                if (side in ('left', 'right') and only_side(instruction, side)
                        and instruction.lower().startswith('move diagonally ' + side)):
                    if side in proposals: raise ValueError('duplicate side')
                    proposals[side] = instruction
            if set(proposals) != {'left', 'right'}: raise ValueError('teacher omitted a valid opposite pair')
            filtered = {}
            for side in ('left', 'right'):
                opposite = 'right' if side == 'left' else 'left'
                branch = pair['branches'][side]
                # Counterbalance ordering so the filter cannot always select item 0.
                names = ['left', 'right'] if side == 'left' else ['right', 'left']
                if pair['source_index'] % 2: names.reverse()
                originals = [proposals[n] for n in names]
                prompt = filter_template.format(labels=f'adjust {side}, with forward translation',
                    orig_instructions=json.dumps(originals), unique_id=uid + '-' + side)
                prompt += ('\nSAGE alignment audit: the actual first movement is shown by these ego-frame prefix points '
                    '(meters, x forward, y left): ' + json.dumps(branch['prefix_local_points']) +
                    '\nThis geometry passed the configured static collision replay. Image grounding still requires your judgment. '
                    'Treat the first short prefix as decisive: reject an instruction that asks for the opposite initial side, '
                    'even if the route later rejoins. Reject both if they contradict visible objects or the preserved target/where. '
                    'This filter judges ONLY the immediate first movement above, NOT the later route to the global goal. '
                    'The unchanged second stage is continue toward the original target. A temporary lateral movement '
                    'is not a change of final destination. '
                    'Keep only original strings in best. Give one short direction-preserving paraphrase in new when accepted. '
                    'The paraphrase MUST differ from every original string and retain explicit current left/right motion, '
                    'for example replacing Move diagonally left with Move forward and left. '
                    'Preserved global task and contract: ' + json.dumps({'task': row['instruction'], 'contract': row['contract']}))
                checked = ask(uid + '-filter-' + side, row['current_image'], prompt, filter_schema)
                if checked['unique_id'] != uid + '-' + side: raise ValueError('filter id mismatch')
                if set(checked['best']) != {proposals[side]}: raise ValueError('self/cross instruction filter failed: ' + side)
                alternatives = [s for s in checked['new'] if only_side(s, side) and s.strip() != proposals[side]]
                if not alternatives: raise ValueError('no direction-preserving paraphrase: ' + side)
                filtered[side] = {'instruction': proposals[side], 'paraphrase': alternatives[0]}
            accepted.append({**pair, 'language': filtered})
            audits.append({'pair_id': uid, 'accepted': True})
        except (ValueError, KeyError, jsonschema.ValidationError) as exc:
            audits.append({'pair_id': uid, 'accepted': False, 'reason': str(exc)})
        atomic_json(output / 'language_audit.json', audits)
        print(json.dumps(audits[-1], ensure_ascii=False), flush=True)
    required = prepared.get('minimum_pairs', 4)
    if len(accepted) < required:
        atomic_json(output / 'insufficient_pairs.json', {'accepted': len(accepted), 'required': required})
        raise RuntimeError('fewer than 4 language/geometry accepted pairs; no training manifest published')
    # Scene roles in expanded runs are frozen before semantic filtering.
    accepted.sort(key=lambda v: hashlib.sha256(v['row']['source_scene'].encode()).hexdigest())
    if prepared.get('fixed_split'):
        val_scenes = {p['row']['source_scene'] for p in accepted} if prepared['fixed_split']=='validation' else set()
        nval = len(accepted) if val_scenes else 0
    elif quotas:
        counts = {split:sum(p['planned_split']==split for p in accepted) for split in quotas}
        if counts != quotas:
            atomic_json(output/'insufficient_pairs.json', {'accepted':counts,'required':quotas})
            raise RuntimeError('expanded semantic filtering did not fill the frozen split quotas')
        val_scenes = {p['row']['source_scene'] for p in accepted if p['planned_split']=='validation'}
        nval = counts['validation']
    else:
        nval = max(1, len(accepted) // 3)
        val_scenes = {v['row']['source_scene'] for v in accepted[:nval]}
    samples = output / 'samples'; samples.mkdir(exist_ok=True)
    rows, eval_cases = [], []
    transform = TrajectoryTransform.from_quantiles(Q01, Q99, flip_y=True, flip_yaw=True)
    for pair in accepted:
        source = pair['row']; uid = pair['pair_id']
        split = 'validation' if source['source_scene'] in val_scenes else 'train'
        pair['pilot_split'] = split
        variants = {'original': source['slow_completion'], 'original_repeat': source['slow_completion']}
        with np.load(source['sample']) as saved:
            visuals = {k: saved[k] for k in ('reference_visual_tokens', 'current_visual_tokens')}
        originals = []
        for side in ('left', 'right'):
            branch = pair['branches'][side]; points = np.asarray(branch['path_xy'])
            route = {'xs': points[:,0].tolist(), 'ys': points[:,1].tolist(),
                     'yaws': _route_yaws(points.tolist(), source['source_pose_xyyaw'][2]),
                     'arc': np.r_[0., np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()]}
            route['length_m'] = float(route['arc'][-1])
            result = route_suffix(route, 0.0)
            if result is None: raise ValueError('no route suffix')
            pose, target, mask, remaining = result
            decoded = transform.apply(torch.from_numpy(target).unsqueeze(0))[0].numpy()
            prefix = decoded[:4,:2].sum(0)
            expected = branch['prefix_local_xy']
            if not np.allclose(prefix, expected, atol=2e-5): raise ValueError('coordinate/normalization roundtrip changed prefix')
            pair['branches'][side]['target_real'] = decoded.tolist()
            sample_path = samples / (uid + '-' + side + '.npz')
            save_npz(sample_path, **visuals, target_actions=target.astype(np.float32), action_mask=np.asarray(mask, dtype=bool))
            for paraphrase in (False, True):
                contract = copy.deepcopy(source['contract'])
                contract['plan'] = [pair['language'][side]['paraphrase' if paraphrase else 'instruction'],
                                    'continue toward ' + contract['target']]
                completion = V43.format_tool_call('run_action_head', contract)
                parsed, _, reason = parse_tool_call_text(completion, action_head_as_tool_version='v4.3')
                if parsed is None: raise ValueError(reason)
                variants[side + ('_paraphrase' if paraphrase else '')] = completion
                if paraphrase: continue  # Keep paraphrases out of pilot fine-tuning.
                row = copy.deepcopy(source)
                row.update(sample=str(sample_path), split=split, slow_anchor_id=uid + ':' + side,
                    contract=contract, slow_completion=completion, pass_rank=None, query_request_id=None,
                    decision_filter='cast_cf_schema_and_self_cross_filter_and_exact_sage_prefix_replay',
                    slow_coverage_m=1.6, slow_anchor_remaining_arc_m=remaining, fast_remaining_arc_m=remaining,
                    valid_action_steps=int(mask.sum()), cast_pair_id=uid,
                    cast_provenance={'source_anchor_id': source['slow_anchor_id'], 'side': side,
                        'teacher_kind': 'CAST prompt adapter + SAGE geometric teacher',
                        'query_hidden': 'not stored; must recompute from this completion',
                        'pair_manifest': str(output / 'accepted_pairs.json'), 'scope': 'first fast window only'})
                V43.assert_protocol_identity(row); rows.append(row)
            pair['branches'][side]['sample'] = str(sample_path)
            # Equal number of original and augmented anchor groups (50/50).
            retained = copy.deepcopy(source); retained['split'] = split
            retained['slow_anchor_id'] = uid + ':original-control-' + side
            retained['cast_pair_id'] = uid; originals.append(retained)
        rows.extend(originals)
        eval_cases.append({'pair_id': uid, 'pilot_split': split, 'row': source, 'variants': variants})
    # Exercise the actual training dataset/collator, including NPZ shape/mask.
    for row in rows:
        batch = joint_collate([JointBundleDataset([row])[0]])
        if batch['target_actions'].shape != (1,16,3): raise ValueError('wrong training target shape')
    atomic_json(output / 'accepted_pairs.json', accepted)
    write_jsonl(output / 'manifest.jsonl', rows)
    for name in ('eval_before', 'eval_after'):
        sub = output / name; sub.mkdir(exist_ok=True)
        atomic_json(sub / 'cases.json', {'source': str(output / 'accepted_pairs.json'), 'cases': eval_cases,
            'scope': prepared['scope'] + '; paraphrases excluded from fine-tuning; explicit pilot scene split'})
    atomic_json(output / 'data_report.json', {'ok': True, 'pairs': len(accepted), 'rows': len(rows),
        'train_pairs': len(accepted)-nval, 'validation_pairs': nval, 'validation_scenes': sorted(val_scenes),
        'manifest': str(output / 'manifest.jsonl'), 'manifest_sha256': sha256_file(output / 'manifest.jsonl'),
        'original_to_augmented': '1:1 by anchor groups; original controls duplicated with explicit distinct IDs',
        'paraphrases_in_training': False, 'query_recomputed_by_joint_sft': True,
        'teacher_min_side_m': TEACHER_MIN_SIDE_M, 'eval_min_side_m': EVAL_MIN_SIDE_M})
    print('data materialized', flush=True)


def score(output, evaluation_name):
    from cast_case_feasibility_audit import SageSet, OccMap, _execution_replay_free, RADIUS
    pairs = json.loads((output / 'accepted_pairs.json').read_text())
    evaluation = output / evaluation_name
    if not json.loads((evaluation / 'completed.json').read_text())['ok']:
        raise RuntimeError('incomplete inference')
    dataset = SageSet(split=json.loads((output/'data_report.json').read_text()).get('sage_split', 'official_gvln_action_head_p0_v3full827train'))
    results = []
    for index, pair in enumerate(pairs):
        head = np.load(evaluation / f'case_{index}_head.npz')
        source = pair['row']; pose = source['source_pose_xyyaw']
        occ = OccMap(dataset.scene_dir(source['source_scene']), collision_mode='usd_rooms', furniture_mode='none')
        yaw = pose[2]; c, s = math.cos(yaw), math.sin(yaw)
        targets = {}
        for side in ('left', 'right'):
            if 'target_real' in pair['branches'][side]:
                targets[side] = np.asarray(pair['branches'][side]['target_real']).cumsum(0)
            else:
                # Backward compatible with the initial materializer: reconstruct
                # through the same public checkpoint-space transform.
                import torch
                from action_head.policy import TrajectoryTransform
                from openloop_eval import Q01, Q99
                with np.load(pair['branches'][side]['sample']) as sample:
                    transform = TrajectoryTransform.from_quantiles(Q01, Q99, flip_y=True, flip_yaw=True)
                    targets[side] = transform.apply(torch.from_numpy(sample['target_actions']).unsqueeze(0))[0].numpy().cumsum(0)
        record = {'pair_id': pair['pair_id'], 'scene': source['source_scene'], 'split': pair['pilot_split'], 'conditions': {}}
        for suffix in ('', '_paraphrase'):
            pair_checks = []
            for draw in range(4):
                checks = []
                for side in ('left', 'right'):
                    name = side + suffix; opposite = 'right' if side == 'left' else 'left'
                    trajectory = head[name + '_actions'][draw].cumsum(0)
                    own = np.linalg.norm(trajectory[:4,:2] - targets[side][:4,:2], axis=1).mean()
                    swapped = np.linalg.norm(trajectory[:4,:2] - targets[opposite][:4,:2], axis=1).mean()
                    local = trajectory[:4,:2]
                    world = local @ np.asarray([[c, s], [-s, c]]) + np.asarray(pose[:2])
                    free, replay = _execution_replay_free(occ, np.vstack([pose[:2], world]), step_m=0.05, radius_m=RADIUS)
                    signed_y = float(trajectory[3,1]) * (1 if side == 'left' else -1)
                    values = {'prefix_own_ade_m': float(own), 'prefix_swapped_ade_m': float(swapped),
                              'matched_better': bool(own < swapped), 'signed_prefix_y_m': signed_y,
                              'side_margin_pass': signed_y >= EVAL_MIN_SIDE_M,
                              'static_replay_free': bool(free and replay['collisions'] == 0)}
                    record['conditions'].setdefault(name, []).append(values)
                    checks.append(values['side_margin_pass'] and values['static_replay_free'])
                pair_checks.append(all(checks))
            record['paired_prefix_gate' + suffix] = pair_checks
        result = json.loads((evaluation / f'case_{index}_actions.json').read_text())
        record['original_full_ade_m'] = float(np.mean([v['original_gt_metrics']['full_ade_m'] for v in result['variants']['original']]))
        results.append(record)
    aggregate = {}
    for split in ('train', 'validation'):
        subset = [r for r in results if r['split'] == split]
        aggregate[split] = {'pairs': len(subset)}
        for suffix in ('', '_paraphrase'):
            checks = [v for r in subset for v in r['paired_prefix_gate' + suffix]]
            values = [v for r in subset for side in ('left','right') for v in r['conditions'][side + suffix]]
            aggregate[split]['paired_prefix_gate' + suffix] = {'passed': sum(checks), 'total': len(checks)}
            aggregate[split]['mean_own_prefix_ade_m' + suffix] = float(np.mean([v['prefix_own_ade_m'] for v in values])) if values else None
            aggregate[split]['matched_better' + suffix] = {'passed': sum(v['matched_better'] for v in values), 'total': len(values)}
        aggregate[split]['original_full_ade_m'] = float(np.mean([v['original_full_ade_m'] for v in subset])) if subset else None
    report = {'evaluation': evaluation_name, 'aggregate': aggregate, 'cases': results,
              'gate_defined_before_inference': {'min_signed_y_m': EVAL_MIN_SIDE_M, 'prefix_steps': 4,
                  'both_sides_required': True, 'static_replay_free_required': True},
              'scope': json.loads((output/'data_report.json').read_text()).get('evaluation_scope', 'engineering prefix gate, not semantic task SR; static occupancy replay, not rendered closed loop; validation scenes held out only from this small fine-tuning pilot')}
    atomic_json(evaluation / 'prefix_score.json', report)
    print(json.dumps(aggregate, ensure_ascii=False), flush=True)


class PairGeometryRejection(ValueError):
    """Expected per-pair quality rejection, distinct from a broken data contract."""
    def __init__(self, side, reason, details):
        super().__init__(f'{side}: {reason}')
        self.audit = {'side': side, 'reason': reason, **details}


def validate_decoded_branch(branch, pose, side, occ, transform):
    from cast_case_feasibility_audit import _execution_replay_free, RADIUS
    from sage.capability_benchmark import point_at_arc
    import torch
    c, s = math.cos(pose[2]), math.sin(pose[2])
    with np.load(branch['sample']) as sample:
        real = transform.apply(torch.from_numpy(sample['target_actions']).unsqueeze(0))[0].numpy()
    if real.shape != (16,3) or not np.isfinite(real).all():
        raise ValueError('invalid decoded teacher label shape or nonfinite values')
    local = real[:,:2].cumsum(0)
    world = local @ np.asarray([[c, s], [-s, c]]) + np.asarray(pose[:2])
    expected = np.asarray([point_at_arc(branch['path_xy'], .1*i) for i in range(1,17)])
    max_error = float(np.max(np.abs(world-expected)))
    if not np.allclose(world, expected, atol=2e-5):
        raise ValueError('full target XY roundtrip mismatch')
    free, replay = _execution_replay_free(occ, np.vstack([pose[:2], world]), step_m=.05, radius_m=RADIUS)
    if not free or replay.get('collisions', 0):
        comparisons = {}
        for name, points, step in [('original_path_0p1m', branch['path_xy'], .1),
                                  ('original_path_0p05m', branch['path_xy'], .05),
                                  ('sampled_teacher_0p05m', np.vstack([pose[:2], expected]), .05)]:
            ok, detail = _execution_replay_free(occ, points, step_m=step, radius_m=RADIUS)
            comparisons[name] = {'free': bool(ok and not detail.get('collisions',0)), 'replay': detail}
        raise PairGeometryRejection(side, 'decoded_teacher_replay_failed',
            {'replay': replay, 'max_xy_roundtrip_error_m': max_error, 'comparisons': comparisons})
    if (1 if side=='left' else -1)*local[3,1] < TEACHER_MIN_SIDE_M:
        raise PairGeometryRejection(side, 'decoded_teacher_side_margin_failed', {'signed_y_m':float((1 if side=='left' else -1)*local[3,1])})
    branch['target_real'] = real.tolist()
    branch['decoded_replay'] = replay


def finalize(output, reject_invalid_pairs=False):
    """Narrow the first pilot to certified immediate direction semantics.

CAST proposals can include 'past/around' relations not certified by our static
prefix check. Preserve them as provenance, never silently train those claims.
The unchanged target/where carries grounding; the current stage is canonical
ego-direction text, with a separate held-out paraphrase template.
"""
    from action_head import protocol_v4_3 as V43
    from sft.build_fm_nested_bundles import parse_tool_call_text
    from sft.train_fm_head_sft import split_rows
    from cast_case_feasibility_audit import SageSet, OccMap, _execution_replay_free, RADIUS
    from sage.capability_benchmark import point_at_arc
    from action_head.policy import TrajectoryTransform
    from openloop_eval import Q01, Q99
    import torch
    if (output / 'eval_before/completed.json').exists():
        raise RuntimeError('cannot relabel a dataset after baseline evaluation')
    pairs = json.loads((output / 'accepted_pairs.json').read_text())
    by_id = {p['pair_id']: p for p in pairs}
    rows = [json.loads(s) for s in (output / 'manifest.jsonl').read_text().splitlines()]
    fixed_split = json.loads((output/'prepared.json').read_text()).get('fixed_split')
    if fixed_split:
        if any(r['split'] != fixed_split for r in rows): raise ValueError('fixed split mismatch')
    else:
        split_rows(rows)  # Existing scene-disjointness check.
    assert len({r['slow_anchor_id'] for r in rows}) == len(rows)
    dataset = SageSet(split=json.loads((output/'prepared.json').read_text()).get('sage_split', 'official_gvln_action_head_p0_v3full827train'))
    transform = TrajectoryTransform.from_quantiles(Q01, Q99, flip_y=True, flip_yaw=True)
    rejected = []
    for pair in pairs:
        pair['canonical_language'] = {s: {'instruction': f'Move diagonally {s} across the open floor',
            'paraphrase': f'Move forward and {s} across the open floor'} for s in ('left', 'right')}
        pair['contract_scope'] = 'direction-only immediate prefix; original CAST landmark relations kept as audit only'
        pose = pair['row']['source_pose_xyyaw']; c, s = math.cos(pose[2]), math.sin(pose[2])
        occ = OccMap(dataset.scene_dir(pair['row']['source_scene']), collision_mode='usd_rooms', furniture_mode='none')
        try:
            for side, branch in pair['branches'].items():
                validate_decoded_branch(branch, pose, side, occ, transform)
        except PairGeometryRejection as exc:
            if not reject_invalid_pairs:
                raise
            rejected.append({'pair_id':pair['pair_id'], 'source_anchor_id':pair['row']['slow_anchor_id'],
                             'scene':pair['row']['source_scene'], **exc.audit})
    rejected_ids = {v['pair_id'] for v in rejected}
    pairs = [p for p in pairs if p['pair_id'] not in rejected_ids]
    rows = [r for r in rows if r['cast_pair_id'] not in rejected_ids]
    by_id = {p['pair_id']:p for p in pairs}
    if reject_invalid_pairs:
        atomic_json(output/'geometry_quality_audit.json', {'policy':'reject whole pair; retain strict 0.05m decoded replay',
            'rejected_pairs':rejected,'retained_pairs':len(pairs), 'quality_policy_version':2})
    for row in rows:
        pair = by_id[row['cast_pair_id']]
        source = pair['row']
        for key in ('slow_messages','reference_image','current_image','source_pose_xyyaw','instruction'):
            if row[key] != source[key]: raise ValueError('same-state invariant failed: ' + key)
        if 'cast_provenance' not in row: continue
        side = row['cast_provenance']['side']
        row['contract']['plan'] = [pair['canonical_language'][side]['instruction'], 'continue toward ' + source['contract']['target']]
        row['slow_completion'] = V43.format_tool_call('run_action_head', row['contract'])
        parsed, _, _ = parse_tool_call_text(row['slow_completion'], action_head_as_tool_version='v4.3')
        if parsed is None or parsed['arguments'] != row['contract']: raise ValueError('completion/contract mismatch')
        row['cast_provenance']['contract_scope'] = pair['contract_scope']
        with np.load(row['sample']) as a, np.load(source['sample']) as b:
            for key in ('reference_visual_tokens','current_visual_tokens'):
                if not np.array_equal(a[key],b[key]): raise ValueError('visual input changed between paired branches')
    atomic_json(output / 'accepted_pairs.json', pairs)
    write_jsonl(output / 'manifest.jsonl', rows)
    for name in ('eval_before','eval_after'):
        path = output / name / 'cases.json'; document = json.loads(path.read_text())
        document['cases'] = [case for case in document['cases'] if case['pair_id'] in by_id]
        for case in document['cases']:
            pair = by_id[case['pair_id']]
            for side in ('left','right'):
                for paraphrase in (False,True):
                    contract = copy.deepcopy(pair['row']['contract'])
                    contract['plan'] = [pair['canonical_language'][side]['paraphrase' if paraphrase else 'instruction'],
                                        'continue toward ' + contract['target']]
                    case['variants'][side + ('_paraphrase' if paraphrase else '')] = V43.format_tool_call('run_action_head',contract)
        document['scope'] += '; canonical immediate-direction stages; no complex landmark-relation claim'
        atomic_json(path,document)
    report = json.loads((output / 'data_report.json').read_text())
    report.update(language_accepted_pairs=report.get('language_accepted_pairs', report.get('pairs',len(pairs)+len(rejected))),
                  pairs=len(pairs), rows=len(rows), geometry_rejected_pairs=len(rejected),
                  train_pairs=sum(p['pilot_split']=='train' for p in pairs),
                  validation_pairs=sum(p['pilot_split']=='validation' for p in pairs),
                  validation_scenes=sorted({p['row']['source_scene'] for p in pairs if p['pilot_split']=='validation'}),
                  manifest_sha256=sha256_file(output / 'manifest.jsonl'), finalized=True,
                  contract_scope='direction-only prefix pilot; target/where preserved; CAST relation text audit-only',
                  verified=['scene disjointness','unique branch IDs','same full context','same exact visuals',
                            'completion matches contract','full XY label roundtrip','decoded static teacher replay','opposite prefix margins'])
    atomic_json(output / 'data_report.json', report)
    print(json.dumps(report,ensure_ascii=False), flush=True)


def infer(output, evaluation_name, checkpoint, query_backend="hf"):
    report = json.loads((output / 'data_report.json').read_text())
    if not report.get('finalized') or report['manifest_sha256'] != sha256_file(output / 'manifest.jsonl'):
        raise RuntimeError('run finalize and freeze the data before inference')
    from sft.fla_l2norm_dynamic_nb import maybe_install_l2norm_dynamic_nb_patch
    maybe_install_l2norm_dynamic_nb_patch()
    import diagnose_v43_instruction_control as diagnostic
    if checkpoint: diagnostic.CHECKPOINT = Path(checkpoint)
    evaluation = output / evaluation_name
    diagnostic.run(evaluation, query_backend=query_backend)
    diagnostic.analyze(evaluation)
    score(output, evaluation_name)


def compare(output):
    """Compare completed evaluations without changing the frozen scoring rule."""
    before, after = [output / name for name in ('eval_before', 'eval_after')]
    cases = [json.loads((path / 'cases.json').read_text())['cases'] for path in (before, after)]
    if cases[0] != cases[1]:
        raise ValueError('before/after evaluation inputs differ')
    for index in range(len(cases[0])):
        actions = [json.loads((path / f'case_{index}_actions.json').read_text())
                   for path in (before, after)]
        seeds = [{name: [draw['seed'] for draw in draws]
                  for name, draws in document['variants'].items()} for document in actions]
        if seeds[0] != seeds[1]:
            raise ValueError(f'before/after sampling seeds differ for case {index}')
        for path in (before, after):
            with np.load(path / f'case_{index}_head.npz') as head:
                if not np.array_equal(head['original_actions'], head['original_repeat_actions']):
                    raise ValueError(f'identical-input control differs: {path}, case {index}')
    reports = []
    for path in (before, after):
        if not json.loads((path / 'completed.json').read_text())['ok']:
            raise ValueError(f'incomplete evaluation: {path}')
        reports.append(json.loads((path / 'prefix_score.json').read_text()))
    if reports[0]['gate_defined_before_inference'] != reports[1]['gate_defined_before_inference']:
        raise ValueError('before/after scoring rules differ')
    layers = {}
    for label, split, suffix in (
        ('1_train_canonical', 'train', ''),
        ('2_train_paraphrase', 'train', '_paraphrase'),
        ('3_heldout_canonical', 'validation', ''),
        ('3_heldout_paraphrase', 'validation', '_paraphrase'),
    ):
        layers[label] = {
            phase: {key: report['aggregate'][split][key + suffix] for key in
                    ('paired_prefix_gate', 'matched_better', 'mean_own_prefix_ade_m')}
            for phase, report in zip(('before', 'after'), reports)
        }
    retention = {split: {phase: report['aggregate'][split]['original_full_ade_m']
                         for phase, report in zip(('before', 'after'), reports)}
                 for split in ('train', 'validation')}
    result = {'layers': layers, 'original_full_ade_m': retention,
              'scope': reports[1]['scope'], 'same_evaluation_cases': True,
              'same_sampling_seeds': True, 'identical_input_controls_passed': True,
              'before': str(before.resolve()), 'after': str(after.resolve()),
              'per_case_scores': [str(path / 'prefix_score.json') for path in (before, after)]}
    atomic_json(output / 'three_layer_comparison.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'generate', 'finalize', 'infer', 'score', 'compare'])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--evaluation-name', choices=['eval_before', 'eval_after'], default='eval_before')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--query-backend', choices=['hf', 'vllm'], default='hf')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ART): parser.error('output outside authorized artifact root')
    if args.mode == 'compare': compare(args.output)
    elif args.mode == 'infer': infer(args.output, args.evaluation_name, args.checkpoint, args.query_backend)
    elif args.mode == 'score': score(args.output, args.evaluation_name)
    elif args.mode == 'finalize': finalize(args.output)
    else: (prepare if args.mode == 'prepare' else generate)(args.output)


if __name__ == '__main__': main()
