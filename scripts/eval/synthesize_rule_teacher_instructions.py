#!/usr/bin/env python3
"""Reuse SAGE-3D's official scene-text and instruction-generation stages.

Existing train endpoints supply feasible starts and endpoint identities only.
Reference trajectories are never supplied to the downstream rule teacher.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import traceback
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'vln_demo/src'))
from sage.datasets import SageSet
from sage.goal_region import label_index_from_items

ARTIFACTS = Path('/data/zerovln/artifacts')
TYPES = ('Area-based', 'Relative_Relationship', 'Scenario_Driven', 'Attribute-based', 'Add_Object')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def canonical_physical_map(items, official_objects, physical_module):
    """Join canonical training IDs to official objects by label and 3D bbox.

    Both indexes come from their existing converters. Follow each original
    labels.json row so identical bboxes still refer to distinct source
    instances; validate the official rounded bbox as well.
    """
    canonical, audit, used = {}, [], set()
    previous_canonical, previous_official = {}, {}
    for source_index in range(len(items)):
        prefix = items[:source_index + 1]
        indexed = label_index_from_items(prefix)
        converted, _, _ = physical_module.build_scene_entries(prefix)
        new_canonical = set(indexed) - set(previous_canonical)
        new_official = set(converted) - set(previous_official)
        previous_canonical, previous_official = indexed, converted
        if not new_canonical and not new_official:
            continue
        if len(new_canonical) != 1 or len(new_official) != 1:
            raise ValueError(f'official/canonical source-row coverage mismatch: {source_index}')
        object_id, official_id = next(iter(new_canonical)), next(iter(new_official))
        obj = indexed[object_id]
        x0, y0, x1, y1 = obj['bbox_xyxy']
        z0, z1 = obj['z_range']
        bounds = physical_module.Bounds(x0, y0, z0, x1, y1, z1).to_string()
        if (official_id.rsplit('_', 1)[0] != obj['label'] or
                official_objects.get(official_id) != bounds or official_id in used):
            raise ValueError(f'ambiguous or missing official object identity: {object_id}')
        used.add(official_id)
        canonical[object_id] = bounds
        audit.append(dict(canonical_id=object_id, official_id=official_id,
                          source_index=source_index, ins_id=obj['ins_id'], bounds=bounds))
    if used != set(official_objects):
        raise ValueError('official and canonical physical maps have different object coverage')
    return canonical, audit


def choose_instructions(instructions, start, end, count):
    """Require the requested distinct, real instructions for this exact pair."""
    chosen, seen = [], set()
    for kind in TYPES:
        for index, item in enumerate(instructions):
            if not isinstance(item, dict):
                continue
            text = item.get('generated_instruction')
            if (item.get('instruction_type') != kind or item.get('start') != start
                    or item.get('end') != end or not isinstance(text, str)
                    or not text.strip() or text.strip().lower() == 'default'):
                continue
            normalized_text = ' '.join(text.split()).casefold()
            if normalized_text in seen:
                continue
            chosen.append((index, item))
            seen.add(normalized_text)
            break
        if len(chosen) == count:
            return chosen
    raise ValueError(f'need {count} distinct official instructions; received {len(chosen)}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pairs', type=int, default=2)
    p.add_argument('--per-pair', type=int, default=3)
    p.add_argument('--official-root', type=Path, default=REPO.parent/'SAGE-3D_Official')
    p.add_argument('--api-base', default='http://127.0.0.1:28970/v1')
    p.add_argument('--model', default='Qwen3.6-35B')
    p.add_argument('--max-tokens', type=int, default=8192)
    p.add_argument('--timeout', type=int, default=240,
                   help='HTTP response timeout; recorded without changing official prompts/sampling')
    p.add_argument('--continue-on-error', action='store_true',
                   help='Retain failed pairs in the denominator and continue the frozen population')
    args = p.parse_args()
    if args.pairs < 1 or not 1 <= args.per_pair <= len(TYPES):
        p.error('pairs must be positive; per-pair must be between 1 and 5')
    if args.timeout < 1:
        p.error('timeout must be positive')
    rows = json.loads(args.selection.read_text())['cases'][:args.pairs]
    if len(rows) != args.pairs or len({(r['scene'], r['task']) for r in rows}) != args.pairs:
        p.error('selection must supply the requested distinct endpoint pairs')
    out = args.output.resolve()
    if not out.is_relative_to(ARTIFACTS):
        p.error('output must be on authorized artifact disk')
    out.mkdir(parents=True, exist_ok=False)
    official = args.official_root.resolve()
    paths = {
        'physical': official/'Code/data_pipeline/interiorgs_processing/physical_map_converter.py',
        'scene_text': official/'Code/data_pipeline/interiorgs_processing/scene_text_generator.py',
        'instruction': official/'Code/data_pipeline/trajectory_generation/vln_trajectory_generator.py',
    }
    mods = {name: module('rule_synthesis_'+name, path) for name, path in paths.items()}
    # The official helpers expose neither completion length nor Qwen's
    # thinking switch. Adapt their transport locally while keeping their
    # message construction, temperature and response parsing unchanged.
    import requests
    transport_index = 0
    def generation_post(url, **kwargs):
        nonlocal transport_index
        transport_index += 1
        request = kwargs['json']
        token_response = requests.post(args.api_base.rstrip('/').removesuffix('/v1') + '/tokenize',
            json=dict(model=args.model, messages=request['messages'], add_generation_prompt=True,
                      chat_template_kwargs={'enable_thinking': False}), timeout=30)
        token_response.raise_for_status()
        tokens = token_response.json()
        budget = min(args.max_tokens, int(tokens['max_model_len']) - int(tokens['count']) - 64)
        if budget <= 0:
            raise ValueError('full official prompt exceeds model context; no truncated scene accepted')
        audit = dict(input_tokens=tokens['count'], max_model_len=tokens['max_model_len'],
                     requested_max_tokens=args.max_tokens, effective_max_tokens=budget,
                     chat_template_kwargs={'enable_thinking': False})
        audit_path = out/'transport'/f'request_{transport_index:03d}.json'
        save(audit_path, audit)
        payload = dict(request, max_tokens=budget,
                       chat_template_kwargs={'enable_thinking': False})
        response = requests.post(url, **dict(kwargs, json=payload))
        audit['http_status'] = response.status_code
        if response.status_code == 200:
            audit['finish_reasons'] = [c.get('finish_reason') for c in response.json().get('choices', [])]
        save(audit_path, audit)
        if response.status_code == 200 and any(
                c.get('finish_reason') == 'length' for c in response.json().get('choices', [])):
            raise ValueError('generation truncated; do not use partial instructions')
        return response
    for name in ('scene_text', 'instruction'):
        mods[name].requests = SimpleNamespace(post=generation_post)
    prompt_paths = [official/'prompts/prompt_phy_to_sem.json',
                    official/'prompts/trajectory_generation/prompt_traj_to_instruction.json']
    for path in [*paths.values(), *prompt_paths, Path(__file__)]:
        dest = out/'source_snapshot'/path.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, dest)
    config = dict(official_sources={str(path): sha(path) for path in [*paths.values(), *prompt_paths]},
        adapter_sha256=sha(__file__),
        selection=str(args.selection.resolve()), selection_sha256=sha(args.selection),
        api_base=args.api_base, model=args.model, pairs=args.pairs, per_pair=args.per_pair,
        max_tokens=args.max_tokens, chat_template_kwargs={'enable_thinking': False},
        timeout=args.timeout,
        continue_on_error=args.continue_on_error,
        method='official physical map -> audited canonical training IDs -> official scene text -> official instruction prompt; existing train endpoint pairs')
    save(out/'run_config.json', config)
    dataset = SageSet('official_gvln_action_head_p0_v3full827train')
    selected, prepared = [], []
    for index, row in enumerate(rows):
        task = next(t for t in dataset.tasks(row['scene']) if t.task_id == row['task'])
        folder = out/f'pair_{index:03d}'
        labels = Path(dataset.scene_dir(row['scene']))/'labels.json'
        physical = folder/'scene.json'
        mods['physical'].convert_one_scene(labels, physical, overwrite=False)
        items = json.loads(labels.read_text())
        if isinstance(items, dict):
            items = items['labels']
        objects, identity_audit = canonical_physical_map(
            items, json.loads(physical.read_text()), mods['physical'])
        canonical_path = folder/'scene_canonical.json'
        save(canonical_path, objects)
        save(folder/'object_identity_audit.json', dict(labels_path=str(labels),
            labels_sha256=sha(labels), official_physical_sha256=sha(physical),
            canonical_physical_sha256=sha(canonical_path), objects=identity_audit))
        if task.start_label not in objects or task.goal_label not in objects:
            raise ValueError('source train endpoint IDs do not match official physical map')
        prepared.append((row, task, folder, canonical_path))
    # Complete identity preflight for the entire batch before any LLM call.
    failures = []

    def generate_pair(row, task, folder, canonical_path):
        scene_messages = mods['scene_text'].build_messages(
            mods['scene_text'].load_prompt_template(prompt_paths[0]), canonical_path.read_text())
        save(folder/'scene_request.json', scene_messages)
        scene_text = mods['scene_text'].call_chat_completion(args.api_base, 'local-no-auth',
            args.model, scene_messages, temperature=0.7, timeout=args.timeout)
        (folder/'scene_response.txt').write_text(scene_text)
        upstream = mods['instruction'].OpenAIClient('local-no-auth', args.api_base, args.model, args.timeout)

        class RecordingClient:
            def chat_completion(self, messages):
                save(folder/'instruction_request.json', messages)
                response = upstream.chat_completion(messages)
                (folder/'instruction_response.txt').write_text(response)
                return response

        instructions, valid = mods['instruction'].llm_generate_instructions(
            RecordingClient(), mods['instruction'].load_prompt_template(prompt_paths[1]),
            scene_text, task.start_label, task.goal_label, dataset.bare(row['scene']))
        generated = folder/'instructions.json'
        save(generated, instructions)
        if not valid:
            raise ValueError('official instruction generation failed; no fallback text accepted')
        chosen = choose_instructions(instructions, task.start_label, task.goal_label, args.per_pair)
        for item_index, item in chosen:
            selected.append(dict(scene=row['scene'], task=row['task'],
                instruction=item['generated_instruction'], instruction_provenance=dict(
                    schema='sage_official_generated_instruction_v1', path=str(generated),
                    sha256=sha(generated), index=item_index, start=task.start_label, end=task.goal_label,
                    instruction_type=item['instruction_type'], source_task=row['task'],
                    scene=row['scene'], source_selection=str(args.selection.resolve()),
                    object_identity_audit=str(folder/'object_identity_audit.json'))))
        save(out/'selection.partial.json', dict(complete=False, cases=selected))
        print(json.dumps(dict(event='generated', scene=row['scene'], instructions=len(instructions),
            selected=len(chosen), total_selected=len(selected))), flush=True)
    for row, task, folder, canonical_path in prepared:
        try:
            generate_pair(row, task, folder, canonical_path)
        except Exception as exc:
            failure = dict(scene=row['scene'], task=row['task'], error_type=type(exc).__name__,
                           error=str(exc), traceback=traceback.format_exc())
            save(folder/'generation_error.json', failure)
            failures.append(failure)
            save(out/'generation_errors.json', failures)
            if not args.continue_on_error:
                raise
            print(json.dumps(dict(event='generation_error', scene=row['scene'],
                task=row['task'], error_type=type(exc).__name__, error=str(exc))), flush=True)
    save(out/'selection.json', dict(complete=True, population_complete=not failures,
        scope='Synthetic instructions from official SAGE generation stages on fixed train endpoint pairs; errors retained in denominator; endpoint score is diagnostic, not an official benchmark evaluation',
        cases=selected, failed_pairs=failures))
    save(out/'summary.json', dict(complete=True, population_complete=not failures,
        pairs=args.pairs, successful_pairs=args.pairs-len(failures), failed_pairs=len(failures),
        instructions=len(selected), expected_instructions=args.pairs * args.per_pair,
        selection_sha256=sha(out/'selection.json')))


if __name__ == '__main__':
    main()
