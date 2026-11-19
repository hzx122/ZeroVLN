#!/usr/bin/env python3
"""Native R2R/RxR starts through the existing v6 call-grounded pilot harness.

This entrypoint imports the SAGE pilot's candidate and observation archive
logic. Native instructions/start poses and independent scene geometry replace
its SAGE-only selection; episode goals/reference paths never select labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/eval'), str(ROOT / 'vln_demo/src')]

import rule_teacher_pilot as pilot
from mp3d_rule_teacher import (
    MP3DInstructionSet, MP3DRuleTeacher, HOUSE_ROOT, SCENES_ROOT,
)


@contextmanager
def native_renderer_lease(render_url, *, lock_dir=None):
    """Serialize native collectors sharing the renderer's current scene.

    Independent Gym ports do not imply independent native renderer state. The
    existing nonblocking flock convention rejects a second consumer before
    either Gym reset or rendering, including localhost/127.0.0.1 aliases.
    """
    port = urlsplit(render_url).port or 80
    directory = (Path(lock_dir) if lock_dir is not None else
                 Path(os.environ['RULE_COLLECTION_RUNTIME_ROOT']) / 'native-leases'
                 if os.environ.get('RULE_COLLECTION_RUNTIME_ROOT') else ROOT / '.runtime-mp3d-r2')
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f'native-renderer-{port}.consumer.lock').open('a') as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'native renderer port {port} already has a collector; run R2R/RxR serially') from exc
        try:
            yield
        finally:
            fcntl.flock(lease, fcntl.LOCK_UN)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_cases(args, dataset):
    """Resolve explicit IDs against the original instruction/start-only API."""
    if args.cases:
        document = json.loads(args.cases.read_text())
        requested = document['cases'] if isinstance(document, dict) else document
    else:
        requested = [dict(scene=args.scene, task=args.episode)]
    if not isinstance(requested, list) or not requested:
        raise ValueError('cases must be a nonempty JSON list or {"cases": [...]}')
    rows, seen = [], set()
    source_digest = sha256(dataset.source)
    geometry_digests = {}
    for item in requested[:args.limit]:
        if item.get('backend', args.backend) != args.backend:
            raise ValueError('case backend differs from this pilot backend')
        scene = item['scene'].removeprefix('rxr-')
        if scene not in dataset.scenes:
            raise ValueError(f'scene absent from original {args.backend} split: {scene}')
        episode = str(item.get('task', item.get('episode_id', item.get('episode', ''))))
        task = next((task for task in dataset.tasks(scene) if task.task_id == episode), None)
        if task is None:
            raise ValueError(f'episode absent from original instruction source: {scene}/{episode}')
        wording = item.get('instruction', task.instruction)
        proof = None
        if wording != task.instruction or 'instruction_provenance' in item:
            proof = validate_generated_instruction(item, task, args.backend,
                                                  args.split, source_digest)
        if (scene, episode) in seen:
            raise ValueError(f'duplicate case identity: {scene}/{episode}')
        seen.add((scene, episode))
        if scene not in geometry_digests:
            geometry = {
                'house': args.house_root / scene / f'{scene}.house',
                'navmesh': args.scenes_root / scene / f'{scene}.navmesh',
                'glb': args.scenes_root / scene / f'{scene}.glb',
            }
            geometry_digests[scene] = {
                key: dict(path=str(path.resolve()), sha256=sha256(path))
                for key, path in geometry.items()
            }
        row = dict(backend=args.backend, scene=scene, task=episode,
            instruction=wording, start=task.start,
            native_start_position=task.native_start_position,
            native_start_rotation=task.native_start_rotation,
            instruction_source=dict(path=str(dataset.source.resolve()), sha256=source_digest),
            geometry_sources=geometry_digests[scene])
        if proof is not None:
            row['instruction_provenance'] = proof
        rows.append(row)
    return rows


def validate_generated_instruction(item, task, backend, split, source_digest):
    """Bind generated wording to a frozen train identity/geometry audit."""
    proof = item.get('instruction_provenance', {})
    if (split != 'train' or proof.get('split') != 'train' or
            proof.get('schema') != 'mp3d_official_generated_instruction_v1' or
            proof.get('backend') != backend or proof.get('source_task') != task.task_id or
            proof.get('scene') != item['scene'].removeprefix('rxr-') or
            proof.get('source_dataset_sha256') != source_digest):
        raise ValueError('generated native instruction train/source identity mismatch')
    generated, audit = Path(proof['path']), Path(proof['identity_audit'])
    if sha256(generated) != proof['sha256'] or sha256(audit) != proof['identity_audit_sha256']:
        raise ValueError('generated native instruction or private identity audit changed')
    entry = json.loads(generated.read_text())[proof['index']]
    identity = json.loads(audit.read_text())
    if (entry.get('generated_instruction') != item['instruction'] or
            entry.get('start') != proof['start'] or entry.get('end') != proof['end'] or
            entry.get('instruction_type') != proof['instruction_type'] or
            identity.get('backend') != backend or identity.get('scene') != proof['scene'] or
            identity.get('source_task') != task.task_id or identity.get('split') != 'train' or
            identity.get('source_dataset_sha256') != source_digest or
            identity.get('native_start_position') != task.native_start_position or
            identity.get('native_start_rotation') != task.native_start_rotation or
            identity.get('start_object') != proof['start'] or identity.get('end_object') != proof['end']):
        raise ValueError('generated native wording/start differs from frozen synthesis evidence')
    return proof


def case_for_row(row):
    case = dict(scene=row['scene'], episode_id=row['task'],
        instruction=row['instruction'], stage=f"{row['backend']}_native_train_rule_diagnostic",
        backend=row['backend'], start=row['start'],
        native_start_position=row['native_start_position'],
        native_start_rotation=row['native_start_rotation'],
        instruction_source=row['instruction_source'], geometry_sources=row['geometry_sources'],
        terminal_score_scope='native terminal diagnostic only; never action-head target supervision')
    if 'instruction_provenance' in row:
        case.update(instruction_provenance=row['instruction_provenance'],
            stage=f"{row['backend']}_native_train_generated_instruction",
            terminal_score_scope='unchanged source endpoint diagnostic; not official benchmark evaluation')
    return case


def code_sources():
    """Freeze the imported pilot and the native/shared executor/request chain."""
    names = (
        'scripts/eval/mp3d_rule_teacher_pilot.py', 'scripts/eval/rule_teacher_pilot.py',
        'scripts/eval/serve_rule_call_nlp.py',
        'vln_demo/src/mp3d_rule_teacher.py', 'vln_demo/src/mp3d_teacher_semantics.py',
        'vln_demo/src/rule_teacher.py', 'vln_demo/src/rule_semantics.py',
        'vln_demo/src/rule_map_grounding.py', 'vln_demo/src/rule_text_retrieval.py',
        'vln_demo/src/rule_match_policy.py',
        'vln_demo/src/rule_lexicon.py', 'vln_demo/src/rule_nlp_grounding.py',
        'vln_demo/src/rule_call_nlp.py', 'vln_demo/src/rule_plan_chunks.py',
        'vln_demo/src/privileged_teacher.py', 'vln_demo/src/rule_route_execution.py',
        'vln_demo/src/agent_tools.py',
        'vln_demo/src/prompts.py', 'vln_demo/src/verl_access/bench_client.py',
        'vln_demo/src/verl_access/vlm_policy.py',
        'vln_demo/src/verl_access/context_budget.py', 'vln_demo/src/verl_access/serving_error.py',
        'scripts/eval/rule_teacher_depth_scheduler.py',
        'vln_demo/src/vln_gym/env.py', 'vln_demo/src/vln_gym/server.py',
        'vln_demo/src/vln_gym/client.py', 'vln_demo/src/vln_gym/fm_action.py',
        'vln_demo/src/vln_gym/vlnce_action_head.py',
        'vln_demo/src/remote_habitat/client.py', 'vln_demo/src/remote_habitat/server.py',
        'vln_demo/src/remote_habitat/worker.py', 'vln_demo/src/remote_habitat/policy.py',
        'vln_demo/src/rxr.py',
        'vln_demo/src/sage/occmap.py', 'vln_demo/src/sage/planner.py',
        'vln_demo/src/sage/visibility_audit.py', 'vln_demo/src/sage/goal_region.py',
        'third_party/Fine-Grained-R2R-NOTICE.txt',
        'verl/action_head/render_client.py', 'verl/action_head/runtime.py',
        'verl/action_head/protocol.py', 'verl/action_head/protocol_v6.py', 'verl/action_head/protocol_v8.py',
        'verl/action_head/protocol_v4_11.py', 'verl/action_head/vllm_bridge.py',
        'verl/action_head/trajectory_map.py',
        'verl/scripts/eval/qwen36_4query_ckpt_sweep/closedloop_eval.py',
        'verl/scripts/eval/qwen36_4query_ckpt_sweep/common.py',
        'verl/scripts/eval/action_head_fm_p0_p2/curriculum_action_head_eval.py',
    )
    return [ROOT / name for name in names]


def freeze_config(output, config, sources):
    config['git_runtime'] = pilot.collection_git_provenance()
    config_path = output / 'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError('source/config changed; choose a new output directory')
    pilot.atomic_json(config_path, config)


def validate_bench(bench, args, row):
    spec = bench.spec()
    if spec.get('backend') != args.backend or spec.get('splits') != [args.split]:
        raise RuntimeError(f'Gym backend/split mismatch: {spec.get("backend")}/{spec.get("splits")}')
    catalogue = spec.get('scene_eps', {})
    matching_scenes = [scene for scene in catalogue
        if (scene.removeprefix('rxr-') if args.backend == 'rxr' else scene) == row['scene']]
    if len(matching_scenes) != 1 or row['task'] not in catalogue[matching_scenes[0]]:
        raise RuntimeError('original case identity absent from Gym episode catalogue')


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--backend', choices=('r2r', 'rxr'), required=True)
    result.add_argument('--split', default='train')
    result.add_argument('--scene')
    result.add_argument('--episode')
    result.add_argument('--cases', type=Path, help='JSON cases with scene + task/episode_id')
    result.add_argument('--dataset-source', type=Path)
    result.add_argument('--scenes-root', type=Path, default=SCENES_ROOT)
    result.add_argument('--house-root', type=Path, default=HOUSE_ROOT)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--gym-url', required=True)
    result.add_argument('--gym-urls', help='JSON list of independent Gym actor URLs')
    result.add_argument('--depth-batch-size', type=int, default=1)
    result.add_argument('--http-concurrency', type=int, choices=range(1,513), default=64)
    result.add_argument('--actor-workers', type=int, choices=range(1,9), default=1)
    result.add_argument('--overlap-candidate-validation', action='store_true')
    result.add_argument('--rolling-actors', action='store_true')
    result.add_argument('--rolling-scene-limit', type=int, default=1,
                        help='Maximum scenes per rolling group; default 1 matches native scene cache. 0 replays legacy scheduling.')
    result.add_argument('--rolling-scene-refill', action='store_true',
                        help='Refill across scene boundaries up to the active scene limit; match renderer cache capacity.')
    result.add_argument('--render-url', required=True)
    result.add_argument('--brain-url', default='http://127.0.0.1:28970')
    result.add_argument('--limit', type=int, default=1000000)
    result.add_argument('--max-calls', type=int, default=12)
    result.add_argument('--candidates', type=int, choices=(1, 2, 4, 8), default=8)
    result.add_argument('--candidate-strategy', choices=('adaptive', 'fixed'), default='fixed')
    result.add_argument('--sampling-scheduler', choices=('v6_depth','mixed_depth'), default='v6_depth')
    result.add_argument('--route-min-chunks', '--fast-step-limit', dest='fast_step_limit', type=int, default=4,metavar='CHUNKS',
                        help='Minimum refreshes over the complete route; each predicts 16 poses and executes 4')
    result.add_argument('--select-only', action='store_true')
    return result


def main():
    arg_parser = parser()
    pilot.collection_protocol_arguments(arg_parser)
    args = arg_parser.parse_args()
    pilot.configure_collection_protocol(args)
    try:
        gym_urls = pilot.actor_urls(args)
    except (ValueError, TypeError) as exc:
        arg_parser.error(str(exc))
    if args.overlap_candidate_validation and args.actor_workers == 1:
        arg_parser.error('overlap-candidate-validation requires at least two actor-workers')
    if args.rolling_scene_limit < 0:
        arg_parser.error('rolling-scene-limit must be nonnegative')
    if args.rolling_scene_refill and (not args.rolling_actors or args.rolling_scene_limit < 1):
        arg_parser.error('rolling-scene-refill requires rolling actors and a positive scene limit')
    if not args.output.resolve().is_relative_to(pilot.ART):
        arg_parser.error('output must be within the vln_verl artifact disk root')
    if args.limit < 1 or args.max_calls < 1 or args.fast_step_limit < 1:
        arg_parser.error('limit, max-calls and route-min-chunks / fast-step-limit must be positive')
    if bool(args.cases) == bool(args.scene or args.episode) or (
            not args.cases and not (args.scene and args.episode)):
        arg_parser.error('provide either --cases or both --scene and --episode')
    if os.environ.get('SAGE_BENCH_PROTOCOL_ROOT') or os.environ.get('ACTION_HEAD_RULE_CREDIT') == '1':
        raise RuntimeError('native pilot requires SAGE oracle autostop/rule-credit disabled')
    collection_started_epoch = time.time()
    collection_started_monotonic = time.monotonic()
    dataset = MP3DInstructionSet(args.backend, split=args.split, source=args.dataset_source)
    rows = source_cases(args, dataset)
    selection = dict(schema='mp3d_original_instruction_start_pilot_v1', backend=args.backend,
        split=args.split, scope='Explicit original native starts; no reference-route/goal target labels', cases=rows)
    args.output.mkdir(parents=True, exist_ok=True)
    selection_path = args.output / 'selection.json'
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise RuntimeError('selection changed; choose a new output directory')
    pilot.atomic_json(selection_path, selection)
    if args.select_only:
        print(json.dumps(selection, ensure_ascii=False, indent=2))
        return
    if args.sampling_scheduler == 'v6_depth':
        os.environ['RULE_TEACHER_SAMPLING_PROFILE'] = 'v6_depth'
    sampling = pilot.brain_sampling_contract(0)
    if not sampling['matches_sft_bundle_generation']:
        raise RuntimeError('pilot requires original SFT brain sampling')
    sources = code_sources()
    config = dict(source_sha256={str(path): sha256(path) for path in sources},
        harness=args.action_head_as_tool_version, brain='Qwen3.6-35B', brain_sampling=sampling,
        backend=args.backend, split=args.split, max_calls=args.max_calls,
        candidates=args.candidates, candidate_strategy=args.candidate_strategy, teacher='text_rules_only_native_mp3d',
        gym_url=args.gym_url, render_url=args.render_url, brain_url=args.brain_url,
        selection_scope=selection['scope'], selection_sha256=sha256(selection_path),
        nlp_url=os.environ.get('RULE_NLP_URL', ''),
        supervision_scope=os.environ.get('RULE_TEACHER_SCOPE', 'target_direction'),
        admission_mode=os.environ.get('RULE_TEACHER_ADMISSION', 'strict'),
        grounding_mode=os.environ.get('RULE_TEACHER_GROUNDING', 'lexical'),
        text_embedding_model=os.environ.get('RULE_TEXT_EMBEDDING_MODEL'))
    config['depth_batch_size'] = args.depth_batch_size
    config['sampling_scheduler'] = args.sampling_scheduler
    config['rejected_depth_retries'] = 1 if args.sampling_scheduler == 'v6_depth' else None
    config['http_concurrency'] = args.http_concurrency
    config['actor_workers'] = args.actor_workers
    config['overlap_candidate_validation'] = args.overlap_candidate_validation
    config['parallel_initialization'] = (args.sampling_scheduler == 'v6_depth' and
        os.environ.get('RULE_TEACHER_PARALLEL_INITIALIZE') == '1')
    config['rolling_actors'] = args.rolling_actors
    config['rolling_scene_limit'] = args.rolling_scene_limit
    config['rolling_scene_refill'] = args.rolling_scene_refill
    config['gym_urls'] = gym_urls
    config['context_budget'] = pilot.context_budget_policy()
    config['runtime_config'] = pilot.collection_runtime_config(args)
    freeze_config(args.output, config, sources)
    def build_case(row, destination, gym_url):
        teacher = MP3DRuleTeacher(backend=args.backend, split=args.split,
            dataset_source=dataset.source, scenes_root=args.scenes_root, house_root=args.house_root,
            gym_url=gym_url, vlm_url='DISABLED', render_url=args.render_url,
            artifact_dir=destination/'options', fast_step_limit=args.fast_step_limit,
            instruction_overrides={(row['scene'], row['task']): row['instruction']})
        return teacher, case_for_row(row)
    with native_renderer_lease(args.render_url):
        pilot.collect_cases(rows, args, build_case, scope=selection['scope'],
                            sources=sources, config=config, validate=validate_bench)
    pilot.atomic_json(args.output / 'collection-timing.json', dict(
        schema='native_collection_wall_v1', started_epoch=collection_started_epoch,
        finished_epoch=time.time(), wall_seconds=time.monotonic()-collection_started_monotonic,
        selected_cases=len(rows),
        scope='collector input validation, native initialization, sampling and execution; excludes service cold startup and module import'))


if __name__ == '__main__':
    main()
