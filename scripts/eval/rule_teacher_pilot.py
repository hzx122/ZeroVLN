#!/usr/bin/env python3
"""Run the existing v6 brain loop with a non-LLM rule-grounded executor."""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import shutil
import time
import traceback
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, deque
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'verl/scripts/eval/qwen36_4query_ckpt_sweep'),str(ROOT/'vln_demo/src'),str(ROOT/'verl')]
from rule_teacher import RuleTeacher, TeacherRejected, parse_contract
from closedloop_eval import (E, brain_sampling_contract, training_matched_chat_query,
                             prepare_training_matched_query, ServingGenerationError)
from verl_access.context_budget import ContextBudgetExhausted, context_budget_policy
ActionHeadEvaluator, atomic_json, deterministic_seed = E.ActionHeadEvaluator, E.atomic_json, E.deterministic_seed
from verl_access.bench_client import BenchClient
from action_head import protocol as PV
import agent_tools as A
from privileged_teacher import _world_pose
from sage.datasets import SageSet, _to_geometry_frame, yaw_from_wxyz
from sft.action_head_data_contract import generated_instruction_task_id

ART = Path('/data/zerovln/artifacts')


def collection_git_provenance():
    # Reuse the shared checkout validation; archive identity, never source copies.
    sys.path.insert(0, str(ROOT/'scripts/train'))
    from unified_closedloop_eval import git_runtime
    return git_runtime(require_container=True)



class CandidateSelectionRejected(TeacherRejected):
    """A rule abstention, using the evaluator's existing abort metadata."""
    episode_abort_reason = 'rule_abstention'

    def __init__(self, detail):
        super().__init__(detail)
        self.episode_abort_detail = detail


def raise_if_serving_error(destination):
    """Stop the collector after its failing case and summary have been saved."""
    receipt = next(destination.glob('candidates/decision_*/serving_error.json'), None)
    if receipt is not None:
        raise RuntimeError(f'serving generation failed; collector stopped: {receipt}')


class ReplayBenchClient(BenchClient):
    """Archive the real STOP observation through the unchanged bench path."""
    def __init__(self, *args, teacher, destination, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher, self.destination = teacher, destination

    def step_tool(self, name, arguments=None):
        before_rgb = self.teacher.current_rgb if name == 'stop' else None
        before_pose = _world_pose(self.teacher.render_client.last_info) if name == 'stop' else None
        result = super().step_tool(name, arguments)
        if name == 'stop':
            obs, reward, terminated, truncated, info = result
            folder = self.destination/'stops'/str(time.time_ns())
            folder.mkdir(parents=True, exist_ok=False)
            if before_rgb is not None:
                Image.fromarray(before_rgb).save(folder/'actual_chunk_0.png')
            after_rgb = obs.get('rgb')
            if after_rgb is not None:
                Image.fromarray(after_rgb).save(folder/'actual_chunk_1.png')
                self.teacher.current_rgb = after_rgb
            after_pose = _world_pose(info)
            pose_audit_available = before_pose is not None and after_pose is not None
            atomic_json(folder/'execution.json', dict(tool='stop', original_call=arguments or {},
                ok=True, accepted_for_training=False, reason='stop_executed_no_AH_label',
                reward=reward, terminated=terminated, truncated=truncated,
                fresh_observation_archived=after_rgb is not None,
                pose_audit_available=pose_audit_available,
                moved_m=(math.dist(before_pose[:2], after_pose[:2])
                         if pose_audit_available else None),
                state=dict(initial_pose=before_pose, actual_poses=[after_pose])))
        return result


@lru_cache(maxsize=2)
def _existing_instruction_source(path, digest, stat_identity):
    """Bounded read cache; file identity changes require hash verification again."""
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError('existing instruction source hash mismatch')
    return json.loads(content)


def verify_existing_instruction_record(proof, instruction, scene, source_task):
    """Read the exact existing official record; never synthesize an identity."""
    if (proof.get('schema') != 'sage_official_existing_instruction_v1'
            or proof.get('split') != 'train' or proof.get('scene') != scene
            or proof.get('source_task') != source_task):
        raise ValueError('existing instruction provenance identity mismatch')
    indexes = [proof.get(k) for k in ('scene_index', 'sample_index', 'instruction_index')]
    if any(type(i) is not int or i < 0 for i in indexes):
        raise ValueError('existing instruction index mismatch')
    path = Path(proof['path'])
    stat = path.stat()
    data = _existing_instruction_source(str(path), proof['sha256'],
        (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    try:
        source_scene = data['scenes'][indexes[0]]
        sample = source_scene['samples'][indexes[1]]
        item = sample['instructions'][indexes[2]]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError('existing instruction record absent') from exc
    digest = hashlib.sha256(json.dumps(item, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if (str(source_scene['scene_id']) != scene.removeprefix('sage-')
            or str(sample.get('trajectory_id')) != proof.get('trajectory_id')
            or source_task != f"{proof['trajectory_id']}_{indexes[2]}"
            or (item.get('generated_instruction') or '').strip() != instruction
            or item.get('instruction_type') != proof.get('instruction_type')
            or digest != proof.get('record_sha256')):
        raise ValueError('existing instruction record differs from frozen source')
    return sample, item


def validate_selection_split(selection, sage_split):
    if selection.get('split', sage_split) != sage_split:
        raise ValueError('selection and teacher SAGE split mismatch')
    if any(row.get('instruction_provenance', {}).get('schema') ==
           'sage_official_existing_instruction_v1' for row in selection['cases']) and sage_split != 'train':
        raise ValueError('existing official instructions require the native train split')


def case_for_row(row, task):
    """Bind existing or generated wording to the exact native train task."""
    case = dict(scene=row['scene'], episode_id=row['task'], instruction=task.instruction,
                stage='train_rule_diagnostic', ref_len=task.ref_len)
    if 'instruction' not in row:
        return case
    proof = row['instruction_provenance']
    if proof.get('schema') == 'sage_official_existing_instruction_v1':
        sample, item = verify_existing_instruction_record(proof, row['instruction'], row['scene'], row['task'])
        p0 = sample['points'][0]
        x, y, yaw = _to_geometry_frame(float(p0['position'][0]), float(p0['position'][1]),
                                       yaw_from_wxyz(p0['rotation']))
        start = dict(x=x, y=y, z=float(p0['position'][2]), roll=0., pitch=0., yaw=yaw)
        if (task.task_id != row['task'] or task.scene != row['scene'] or task.start != start
                or task.instruction != row['instruction'] or task.instruction_type != item['instruction_type']
                or task.start_label != (item.get('start') or '').strip()
                or task.goal_label != (item.get('end') or '').strip()):
            raise ValueError('existing official instruction and native task mismatch')
        case.update(backend='sage', stage='train_rule_existing_official_instruction',
                    existing_instruction_provenance=proof)
        if row.get('native_route_scope') is not None:
            case['native_route_scope'] = dict(row['native_route_scope'])
        return case
    path = Path(proof['path'])
    if (proof.get('schema') != 'sage_official_generated_instruction_v1' or
            proof.get('source_task') != row['task'] or proof.get('scene') != row['scene'] or
            proof.get('start') != task.start_label or proof.get('end') != task.goal_label or
            hashlib.sha256(path.read_bytes()).hexdigest() != proof['sha256']):
        raise ValueError('generated instruction provenance/endpoint mismatch')
    item = json.loads(path.read_text())[proof['index']]
    if (item.get('generated_instruction') != row['instruction'] or
            item.get('start') != task.start_label or item.get('end') != task.goal_label or
            item.get('instruction_type') != proof['instruction_type']):
        raise ValueError('generated instruction differs from saved official output')
    case.update(instruction=row['instruction'], stage='train_rule_synthetic_instruction',
                instruction_provenance=proof,
                generated_task_id=generated_instruction_task_id(
                    row['scene'], row['task'], row['instruction'], proof),
                terminal_score_scope='source endpoint diagnostic; not official benchmark evaluation')
    return case


def source_goal_arrival(teacher, case, pose):
    """Read-only source-task STOP admission; never a local AH target or prompt."""
    if case.get('backend') in ('r2r', 'rxr'):
        info = teacher.render_client.last_info
        distance, radius = info.get('distance_to_goal'), info.get('success_radius')
        if (not isinstance(distance, (int, float)) or not isinstance(radius, (int, float))
                or not math.isfinite(distance) or not math.isfinite(radius)
                or distance < 0 or radius <= 0):
            return dict(available=False, success=False, reason='source_goal_metric_unavailable')
        return dict(available=True, success=distance <= radius,
                    distance_to_goal=distance, success_radius=radius,
                    scope='unchanged_native_source_endpoint')
    from sage.goal_region import (load_label_index, official_object_arrival,
                                  object_height_relation)
    task = teacher.task(case['scene'], case['episode_id'])
    target = task.goal_label
    from sage.source_objects import load_source_label_index
    labels = load_source_label_index(teacher.dataset.scene_dir(case['scene']))
    if not target or target not in labels:
        return dict(available=False, success=False, reason='source_goal_object_unavailable')
    official = official_object_arrival(labels[target], pose[:2])
    occ, _, _ = teacher.scene(case['scene'])
    height = object_height_relation(labels[target], occ.ground_z(*pose[:2]))
    return dict(schema='sage-source-endpoint-height-guard-v1',
                available=height['available'],
                success=bool(official['success'] and height['compatible']),
                target_object_id=target, target_ins_id=labels[target].get('ins_id'),
                scope='unchanged_sage_source_endpoint',
                official_xy_arrival=official, height=height)


class CandidateQuery:
    """Unchanged sampling and rule admission, split for depth-wide scheduling.

    prepare/resolve run serially with the owning environment. sample performs
    only a model request and audit writes, and may run across independent states.
    Candidate indices always refer to generation order, including wave surplus.
    """
    def __init__(self, teacher, case, destination, count, strategy='adaptive'):
        if not 1 <= count <= 8 or strategy not in ('adaptive', 'fixed', 'v6_depth'):
            raise ValueError('candidate count must be 1..8 and strategy adaptive or fixed')
        self.teacher, self.case, self.destination = teacher, case, destination
        self.count, self.strategy, self.decision_index = count, strategy, 0
        self.maximum_candidates = count * (2 if strategy == "v6_depth" else 1)
        self.continuous = None
        self.cache_parent_end = 0
        if os.environ.get('RULE_TEACHER_CONTEXT_VERSION') == 'v8':
            from action_head_rl.rollout import DecisionRoutingQueryStateProvider
            model = json.loads(Path(os.environ['RULE_TEACHER_MODEL_CONFIG']).read_text())['text_config']
            self.continuous = DecisionRoutingQueryStateProvider(
                os.environ['RULE_TEACHER_QUERY_CACHE'], hidden_size=model['hidden_size'],
                layer_count=model['num_hidden_layers'], top_k=model['num_experts_per_tok'],
                train_scope=None, query_count=4, continuous_context=True)
            self.continuous.v8_prefix_cache = os.environ.get('RULE_TEACHER_V8_PREFIX_CACHE') == '1'
            self.continuous.begin_episode(0)

    def capture_continuation(self, ctx, reply):
        if self.continuous is None:
            return
        text, request_id, usage, messages = reply
        _, visible = A.split_thinking(text)
        response = E.AHV.response_snapshot(text=text, visible=visible,
            usage=usage, query_request_id=request_id)
        request = E.AHV.semantic_request_snapshot(messages=messages, instruction=self.case['instruction'])
        self.continuous.capture_decision(request=request, response=response,
            decision_index=self.decision_index)
        record = self.continuous.decision_records[-1]
        self.cache_parent_end = int(record['text_response_end'])
        ids = record['text_history_ids']
        previous = ctx.kwargs.get('continuous_prefix_ids')
        # The service canonicalizes exactly one final im_end separator.
        matches = previous is None or ids[:len(previous)].tolist() == previous
        atomic_json(ctx.folder/'continuous_context.json', dict(
            query_request_id=request_id,
            parent_request=ctx.kwargs.get('continuous_parent_request'),
            prompt_tokens=record['text_prompt_length'], history_tokens=len(ids),
            history_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
            exact_parent_prefix=matches,
            prefix_cache_enabled=bool(getattr(self.continuous, 'v8_prefix_cache', False)),
            cache_salt=ctx.kwargs.get('continuous_cache_salt'),
            image_count=sum(part.get('type') in ('image', 'image_url')
                for msg in messages if isinstance(msg.get('content'), list) for part in msg['content'])))
        if not matches:
            raise ValueError('v8 native parent token prefix changed')
        # Collection needs only the last native parent, not an RL minibatch of
        # every full routing tensor. The provider owns its artifact lifetime.
        self.continuous.decision_records[:] = [dict(
            text_history_ids=ids, slow_response=response)]
        self.continuous._captured_query = None
        for index in ctx.generated:
            row = json.loads((ctx.folder/f'candidate_{index:02d}.json').read_text())
            other = row.get('query_request_id')
            if other and other != request_id:
                self.continuous._release_artifact(other)

    def close(self):
        if self.continuous is not None:
            self.continuous.end_episode()

    def prepare(self, *args, **kwargs):
        if hasattr(self.teacher, "discard_candidate_plan"):
            self.teacher.discard_candidate_plan()
        self.decision_index += 1
        folder = self.destination/'candidates'/f'decision_{self.decision_index:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        try:
            payload, budget_audit = prepare_training_matched_query(*args, **kwargs)
        except Exception as exc:
            atomic_json(folder/'context_budget.json', dict(
                getattr(exc, 'audit', {}), status='preflight_failed',
                abort_reason=getattr(exc, 'episode_abort_reason', None), error=str(exc)[:1000]))
            if isinstance(exc, ServingGenerationError):
                atomic_json(folder/'serving_error.json', exc.audit)
            raise
        atomic_json(folder/'context_budget.json', budget_audit)
        if self.continuous is not None:
            kwargs.update(self.continuous.continuous_prefix_ids())
            if self.continuous.v8_prefix_cache:
                from action_head.serve.continuous_context import CACHE_PARENT_END_KEY
                payload.setdefault('vllm_xargs', {})[CACHE_PARENT_END_KEY] = self.cache_parent_end
        return SimpleNamespace(folder=folder, args=args,
            kwargs=dict(kwargs, prepared_payload=payload), budget_audit=budget_audit,
            request_sha256=hashlib.sha256(json.dumps(payload, sort_keys=True,
                ensure_ascii=False, separators=(',', ':')).encode()).hexdigest(),
            before=_world_pose(self.teacher.render_client.last_info), lock=Lock(),
            requested=set(), generated=set(), waves=[], selected=None, failed=False)

    def next_wave(self, ctx):
        if ctx.selected is not None or ctx.failed or len(ctx.requested) >= self.maximum_candidates:
            return []
        offset = sum(len(wave['indices']) for wave in ctx.waves)
        width = self.count if self.strategy in ('fixed', 'v6_depth') else (1, 1, 2, 4)[len(ctx.waves)]
        return [index for index in range(offset, min(offset + width, self.maximum_candidates))
                if index not in ctx.requested]

    def _sampling_audit(self, ctx):
        atomic_json(ctx.folder/'sampling.json', dict(strategy=self.strategy,
            maximum_candidates=self.maximum_candidates, request_sha256=ctx.request_sha256, wave_sizes=([self.count, self.count] if self.strategy == 'v6_depth' else [self.count] if self.strategy == 'fixed' else [1, 1, 2, 4]),
            requested=len(ctx.requested), generated=len(ctx.generated),
            requested_indices=sorted(ctx.requested), generated_indices=sorted(ctx.generated),
            waves=ctx.waves, serving_failed=ctx.failed))

    def _reply_audit(self, ctx, index, reply):
        text, request_id, usage, messages = reply
        _, visible = A.split_thinking(text)
        return dict(index=index, query_request_id=request_id, wave_index=len(ctx.waves), request_sha256=ctx.request_sha256, original_text=text,
            calls=A.parse_tool_calls(visible), usage=usage,
            request=E.AHV.semantic_request_snapshot(messages=messages, instruction=self.case['instruction']),
            real_pose=ctx.before, selected=False, status='pending_validation')

    def sample(self, ctx, index):
        started = time.monotonic()
        timings = dict(schema='candidate_client_timing_v1', completed=False)
        try:
            reply = self._sample(ctx, index, timings)
            timings['completed'] = True
            return reply
        finally:
            timings['elapsed_seconds'] = time.monotonic() - started
            atomic_json(ctx.folder/f'timing_candidate_{index:02d}.json', timings)

    def _sample(self, ctx, index, timings):
        dispatched = time.monotonic()
        with ctx.lock:
            if ctx.failed:
                raise RuntimeError('candidate dispatch attempted after serving failure')
            if index not in self.next_wave(ctx):
                raise ValueError('candidate index was duplicated or is outside the current wave')
            ctx.requested.add(index)
            self._sampling_audit(ctx)
        timings['dispatch_audit_seconds'] = time.monotonic() - dispatched
        generation = {}
        timings['generation'] = generation
        try:
            reply = training_matched_chat_query(*ctx.args, **ctx.kwargs, timings=generation)
        except ContextBudgetExhausted as exc:
            with ctx.lock:
                ctx.failed = True
                atomic_json(ctx.folder/'context_budget.json', dict(ctx.budget_audit,
                    status='sampling_context_rejected', abort_reason=exc.episode_abort_reason,
                    error=str(exc)[:1000]))
                self._sampling_audit(ctx)
            raise
        except ServingGenerationError as exc:
            atomic_json(ctx.folder/f'serving_error_candidate_{index:02d}.json', exc.audit)
            with ctx.lock:
                ctx.failed = True
                if not (ctx.folder/'serving_error.json').exists():
                    atomic_json(ctx.folder/'serving_error.json', exc.audit)
                self._sampling_audit(ctx)
            raise
        # Archive every genuine reply even if a different request in this wave
        # fails. Validation must not erase responses or turn errors into rejects.
        archive_started = time.monotonic()
        atomic_json(ctx.folder/f'candidate_{index:02d}.json', self._reply_audit(ctx, index, reply))
        timings['reply_snapshot_archive_seconds'] = time.monotonic() - archive_started
        accounting_started = time.monotonic()
        with ctx.lock:
            ctx.generated.add(index)
            self._sampling_audit(ctx)
        timings['completion_audit_seconds'] = time.monotonic() - accounting_started
        return reply

    def resolve(self, ctx, replies):
        if ctx.failed:
            raise RuntimeError('cannot validate a failed candidate wave')
        expected = set(ctx.generated) - {i for wave in ctx.waves for i in wave['indices']}
        if set(replies) != expected or ctx.requested != ctx.generated or not replies:
            raise ValueError('candidate replies do not match the generated wave')
        selected_reply = None
        for index, reply in sorted(replies.items()):
            audit = self._reply_audit(ctx, index, reply)
            if ctx.selected is not None:
                audit['status'] = 'not_checked_after_first_valid_candidate'
            else:
                try:
                    calls = audit['calls']
                    if len(calls) != 1:
                        raise TeacherRejected('candidate must contain exactly one tool call')
                    call = calls[0]
                    if call['name'] == 'run_action_head':
                        validator = getattr(self.teacher, 'validate_tool_arguments',
                                            A.validate_action_head_arguments)
                        arguments, error = validator(call['arguments'])
                        if arguments is None:
                            raise TeacherRejected(error)
                        prepare = getattr(self.teacher, 'prepare_candidate', self.teacher.prepare)
                        prepare(scene=self.case['scene'], episode=self.case['episode_id'],
                            pose=ctx.before, rgb=self.teacher.current_rgb, contract=arguments,
                            snapshot=audit['request'], output=ctx.folder/f'candidate_{index:02d}_binding')
                    elif call['name'] == 'stop':
                        invalid = A.rejected_stop_result(call['arguments'])
                        if invalid is not None:
                            raise TeacherRejected(invalid['error'])
                        arrival = source_goal_arrival(self.teacher, self.case, ctx.before)
                        audit['stop_admission'] = arrival
                        if not arrival.get('available'):
                            raise TeacherRejected('stop: source goal arrival unavailable')
                        if not arrival['success']:
                            raise TeacherRejected('stop: source task final goal not reached')
                    elif call['name'] != 'turn':
                        raise TeacherRejected('unsupported tool')
                    ctx.selected, selected_reply = index, reply
                    audit.update(status='selected', selected=True)
                except TeacherRejected as exc:
                    audit.update(status='rejected', reason=str(exc))
                except Exception as exc:
                    audit.update(status='validation_error', reason=str(exc))
                    atomic_json(ctx.folder/f'candidate_{index:02d}.json', audit)
                    raise
            atomic_json(ctx.folder/f'candidate_{index:02d}.json', audit)
        if _world_pose(self.teacher.render_client.last_info) != ctx.before:
            raise RuntimeError('candidate validation mutated the environment')
        ctx.waves.append(dict(wave_index=len(ctx.waves), indices=sorted(replies),
            generated=len(replies), selected_index=ctx.selected))
        self._sampling_audit(ctx)
        exhausted = len(ctx.requested) >= self.maximum_candidates
        if ctx.selected is not None or exhausted:
            atomic_json(ctx.folder/'selection.json', dict(selected_index=ctx.selected,
                sampled=len(ctx.generated), requested=len(ctx.requested), maximum_candidates=self.maximum_candidates,
                strategy=self.strategy, waves=ctx.waves,
                rule='first supported candidate; STOP requires source final goal arrival',
                all_rejected=ctx.selected is None))
            continuation = selected_reply if ctx.selected is not None else replies[max(replies)]
            self.capture_continuation(ctx, continuation)
            if ctx.selected is None:
                rejection = CandidateSelectionRejected('all sampled candidates rejected; no action executed')
                if self.strategy == 'v6_depth':
                    rejection.continue_at_next_depth = True
                    if self.continuous is not None:
                        rejection.rejected_assistant_text = continuation[0]
                    rejection.request_snapshot = self._reply_audit(ctx, max(replies), replies[max(replies)])['request']
                    rejection.feedback = dict(status='rule_rejected_no_action',
                        action_executed=False, attempts=len(ctx.waves),
                        reasons=[json.loads((ctx.folder/f'candidate_{index:02d}.json').read_text()).get('reason', '')
                                 for index in sorted(ctx.generated)])
                raise rejection
        return selected_reply

    def __call__(self, *args, **kwargs):
        ctx = self.prepare(*args, **kwargs)
        while indices := self.next_wave(ctx):
            with ThreadPoolExecutor(max_workers=len(indices)) as pool:
                replies = dict(zip(indices, pool.map(lambda index: self.sample(ctx, index), indices)))
            selected = self.resolve(ctx, replies)
            if selected is not None:
                return selected
        raise RuntimeError('candidate scheduling ended without a selection')


def candidate_query(teacher, case, destination, count, strategy='adaptive'):
    return CandidateQuery(teacher, case, destination, count, strategy)


def select_cases(path, limit, mode='historical', exclude=None, min_ref_len=0., navigation_only=False):
    if mode in ('unfiltered', 'object-endpoint'):
        from rule_lexicon import NAMES
        dataset=SageSet('official_gvln_action_head_p0_v3full827train')
        excluded={r['scene'] for r in json.loads(exclude.read_text())['cases']} if exclude else set()
        # Scene-disjoint diagnostic, chosen without parsing any brain calls or
        # looking at outcomes. One hash-selected original train episode/scene.
        key=lambda value: hashlib.sha256(('rule-teacher-holdout-20260930:'+value).encode()).hexdigest()
        scenes=sorted((s for s in dataset.list_scenes() if s not in excluded),key=key)
        rows=[]
        endpoint_families=('bed','chair','armchair','table','desk','coffee table',
                           'conference table','dining table','bedside table',
                           'cabinet','wardrobe','bookshelf','sofa')
        for scene in scenes:
            tasks=dataset.tasks(scene)
            tasks=[t for t in tasks if t.ref_len >= min_ref_len and
                   (not navigation_only or re.match(
                       r'^(?:(?:walk|go|head|navigate|proceed)\b|move\s+(?:from|to|towards?|forward|backward|left|right|through|into|along)\b)',
                       t.instruction.strip(), re.I))]
            if mode=='object-endpoint':
                tasks=[t for t in tasks if NAMES.get(
                    t.goal_label.rsplit('_',1)[0].replace('_',' ').lower()) in endpoint_families]
            if not tasks:continue
            task=min(tasks,key=lambda t:key(scene+':'+t.task_id))
            rows.append(dict(scene=scene,task=task.task_id,source=dataset.traj_file(scene),ref_len=task.ref_len))
            if len(rows)==limit:break
        atomic_json(path,dict(scope=('SAGE original train, scene-disjoint supported physical object endpoint stratum; one hash-selected episode per scene, source labels only, no teacher outcomes' if mode=='object-endpoint' else 'SAGE original train, scene-disjoint unfiltered diagnostic; one hash-selected episode per scene'),
            endpoint_families=endpoint_families if mode=='object-endpoint' else None,
            min_ref_len=min_ref_len,navigation_only=navigation_only,
            seed='rule-teacher-holdout-20260930',excluded_scenes=sorted(excluded),cases=rows))
        return
    pool = []
    source = ART/'action-head-sft-data-v6-beam4-official-gvln-20260922-r1/shards'
    for shard in (0,5,10,15):
        audit=source/f'shard_{shard:03d}/bundles/anchor_audit.jsonl'
        for line_no,line in enumerate(audit.open(),1):
            row=json.loads(line)
            if row['option_sequence_index'] != 0: continue
            for attempt in row['attempts']:
                d=attempt.get('decision') or {}
                if not attempt.get('accepted') or d.get('name')!='run_action_head':continue
                try:parse_contract(d['arguments'])
                except TeacherRejected:continue
                pool.append(dict(anchor=row['golden_anchor_id'],scene=row['source_scene'],task=row['source_task_id'],arguments=d['arguments'],source=str(audit),line=line_no));break
    # Historical language stratification only; no outcomes or new brain calls
    # are used to choose cases. Episode starts and instructions stay original.
    pool.sort(key=lambda r: hashlib.sha256(r['anchor'].encode()).hexdigest())
    seen = set(); selected = []
    for row in pool:
        if row['scene'] in seen: continue
        seen.add(row['scene']); selected.append(row)
        if len(selected) == limit: break
    atomic_json(path, dict(scope='SAGE train, historically simple object-call stratum, not representative mixture',
                           source=str(source),eligible_pool=len(pool),cases=selected))


def audit_initial_candidates(source, output, rows, *, gym_url='http://127.0.0.1:18974',
                             render_url='http://127.0.0.1:7074'):
    """Controlled rule comparison on genuine saved first-state calls.

    This resets original starts and runs the full private planning preflight;
    it does not query the brain, execute actions, or count as a live rollout.
    """
    records=[]
    teacher=RuleTeacher(gym_url=gym_url,vlm_url='DISABLED',
                        render_url=render_url,artifact_dir=output/'unused_options')
    for index,row in enumerate(rows):
        case=source/f'case_{index:03d}'
        files=sorted(case.glob('candidates/decision_001/candidate_??.json'))
        print(json.dumps(dict(event='preflight_start',case=index,candidates=len(files))),flush=True)
        obs,info=teacher.reset_environment(scene=row['scene'],episode_id=row['task'])
        pose=_world_pose(info)
        for f in files:
            saved=json.loads(f.read_text())
            if math.dist(pose[:2],saved['real_pose'][:2])>1e-5 or abs(pose[2]-saved['real_pose'][2])>1e-5:
                raise RuntimeError('frozen initial candidate pose does not match reset state')
            for call_index,call in enumerate(saved['calls']):
                if call.get('name')!='run_action_head':continue
                record=dict(source=str(f),source_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                            case=index,candidate=saved['index'],call_index=call_index,
                            original_call=call['arguments'],old_status=saved['status'],old_reason=saved.get('reason'))
                dest=output/f'case_{index:03d}'/f"candidate_{saved['index']:02d}_{call_index}"
                try:
                    state=teacher.prepare(scene=row['scene'],episode=row['task'],pose=pose,
                        rgb=obs['rgb'],contract=call['arguments'],snapshot=saved['request'],output=dest)
                    record.update(status='planned_not_executed',bound=state['bound']['id'])
                except TeacherRejected as exc:record.update(status='rule_abstention',reason=str(exc))
                except Exception as exc:record.update(status='infrastructure_error',reason=str(exc),traceback=traceback.format_exc())
                if _world_pose(teacher.render_client.last_info)!=pose:
                    raise RuntimeError('planning audit changed real environment pose')
                records.append(record)
                atomic_json(output/'preflight_summary.json',dict(records=records,
                    planned=sum(r['status']=='planned_not_executed' for r in records),
                    warning='Frozen first-state original calls; planning only, no new brain generation or execution.'))
        print(json.dumps(dict(event='preflight_done',case=index,planned=sum(r['case']==index and r['status']=='planned_not_executed' for r in records))),flush=True)


def actor_urls(args):
    """Require real independent Gym actors for simultaneous episode state."""
    from urllib.parse import urlsplit
    urls = json.loads(args.gym_urls) if args.gym_urls is not None else [args.gym_url]
    if (not isinstance(urls, list) or not urls or
            any(not isinstance(url, str) or not url for url in urls)):
        raise ValueError('--gym-urls must be a JSON list of nonempty URL strings')
    # Physical servers already multiplex independent instruction sessions.
    # Expanding v8's submission window does not require duplicating servers.
    maximum = 256 if os.environ.get('RULE_TEACHER_CONTEXT_VERSION') == 'v8' else 64
    if not 1 <= args.depth_batch_size <= maximum or len(urls) > 64:
        raise ValueError(f'depth-batch-size must be 1..{maximum}; Gym actor count 1..64')
    identities = []
    for url in urls:
        parsed = urlsplit(url)
        host = parsed.hostname
        if host in ('localhost', '127.0.0.1', '::1'):
            host = 'loopback'
        identities.append((parsed.scheme, host, parsed.port or 80, parsed.path.rstrip('/')))
    if len(set(identities)) != len(identities):
        raise ValueError('Gym actor URLs must be distinct, including loopback aliases')
    return urls


def case_cohorts(rows, size):
    """Same-scene cohorts retain original selection indices and input order."""
    if size == 1:
        return [[(index, row)] for index, row in enumerate(rows)]
    scenes = {}
    for index, row in enumerate(rows):
        scenes.setdefault(row['scene'], []).append((index, row))
    return [group[offset:offset + size] for group in scenes.values()
            for offset in range(0, len(group), size)]


def rolling_scene_cohorts(rows, scene_limit):
    """Retain rolling actor reuse within bounded groups of complete scenes."""
    if isinstance(scene_limit, bool) or not isinstance(scene_limit, int) or scene_limit < 0:
        raise ValueError('rolling-scene-limit must be a nonnegative integer')
    if not rows:
        return []
    if scene_limit == 0:
        return [list(enumerate(rows))]
    scenes = case_cohorts(rows, len(rows))
    return [sum(scenes[offset:offset + scene_limit], [])
            for offset in range(0, len(scenes), scene_limit)]


def collect_cases(rows, args, build_case, *, scope, sources, config, validate=None):
    """Reuse one archive/evaluator path for serial and independent-actor cohorts."""
    urls = actor_urls(args)
    depth_synchronous = getattr(args, 'sampling_scheduler', 'mixed_depth') == 'v6_depth'
    if depth_synchronous:
        os.environ['RULE_TEACHER_SAMPLING_PROFILE'] = 'v6_depth'
    rolling = bool(getattr(args, 'rolling_actors', False))
    if depth_synchronous and (rolling or getattr(args, 'rolling_scene_refill', False)):
        raise ValueError('v6 same-depth collection forbids rolling instruction refill')
    scene_limit = getattr(args, 'rolling_scene_limit', 0)
    scene_refill = bool(getattr(args, 'rolling_scene_refill', False))
    if scene_refill and (not rolling or scene_limit < 1):
        raise ValueError('rolling-scene-refill requires rolling actors and a positive scene limit')
    capacity = min(args.depth_batch_size, len(urls))
    if depth_synchronous:
        # Keep a private live state for every instruction. Physical HTTP actors
        # host multiple isolated sessions; no instruction is postponed or refilled.
        for url in urls:
            health = BenchClient(url).health()
            if health.get('session_contract') != 'independent_gym_sessions_v1':
                raise ValueError('v6 same-depth collection requires independent Gym sessions')
        urls = [urls[index % capacity].rstrip('/') + f'/sessions/case_{index:06d}'
                for index in range(len(rows))]
    cohorts = ([list(enumerate(rows))] if depth_synchronous or scene_refill else
               rolling_scene_cohorts(rows, scene_limit) if rolling else case_cohorts(rows, capacity))
    atomic_json(args.output/'cohort_manifest.json', dict(
        schema=('rule_teacher_v6_all_instruction_depth_v1' if depth_synchronous else
                'rule_teacher_scene_refill_v1' if scene_refill else
                'rule_teacher_scene_bounded_rolling_v1' if rolling and scene_limit else
                'rule_teacher_rolling_actors_v1' if rolling else 'rule_teacher_same_scene_cohorts_v1'), depth_batch_size=args.depth_batch_size,
        rolling_scene_limit=scene_limit, rolling_scene_refill=scene_refill, gym_urls=urls, cohorts=[dict(cohort_index=number, scene=None if rolling or depth_synchronous else group[0][1]['scene'],
            scenes=list(dict.fromkeys(row['scene'] for _, row in group)),
            cases=[dict(index=index, task=row['task'], gym_url=None if rolling else urls[slot])
                   for slot, (index, row) in enumerate(group)])
            for number, group in enumerate(cohorts)]))
    results, completed = [], set()
    for index, row in enumerate(rows):
        destination = args.output/f'case_{index:03d}'
        if (destination/'result.json').exists():
            results.append(json.loads((destination/'result.json').read_text()))
            completed.add(index)
            raise_if_serving_error(destination)

    def summary():
        results.sort(key=lambda record: record['index'])
        atomic_json(args.output/'summary.json', dict(cases=results,
            accepted_ah=sum(item['accepted_ah'] for item in results), completed=len(results),
            valid_episodes=sum(bool(item.get('evaluation_valid')) for item in results),
            terminal_successes=sum(bool(item.get('evaluation_valid') and item.get('success')) for item in results),
            cases_with_ah=sum(item['accepted_ah'] > 0 for item in results),
            max_ah_per_episode=max((item['accepted_ah'] for item in results), default=0),
            end_reasons=dict(Counter(item.get('why') or item['status'] for item in results)),
            warning=scope + '; rule-supported does not certify semantic accuracy.'))

    for cohort_index, group in enumerate(cohorts):
        pending = [(slot, index, row) for slot, (index, row) in enumerate(group) if index not in completed]
        if not pending:
            continue
        entries, states, session_clients = [], {}, {}
        backlog = iter(pending) if rolling else iter(())
        scene_backlog = {}
        if scene_refill:
            for item in pending:
                scene_backlog.setdefault(item[2]['scene'], deque()).append(item)
        actor_slots = {}
        def finish(index, rollout=None, error=None):
            destination, record, started = states[index]
            if rollout is not None:
                atomic_json(destination/'rollout.json', rollout)
                record.update(status='infrastructure_error' if rollout.get('infra_error') else 'rollout_ended',
                    why=rollout.get('why'), infra_error=rollout.get('infra_error'))
                record.update({key: rollout.get(key) for key in (
                    'evaluation_valid', 'success', 'success_contract', 'official_object_success_at_end',
                    'official_object_success_detail', 'agent_stopped', 'distance_to_goal',
                    'final_pose_world_xy_yaw_deg', 'path_length_m', 'calls_used',
                    'episode_abort_reason', 'episode_abort_detail', 'actual_progress')})
            elif isinstance(error, TeacherRejected):
                record.update(status='rule_abstention', reason=str(error))
            else:
                record.update(status='infrastructure_error', reason=str(error),
                    traceback=''.join(traceback.format_exception(error)))
            executions = [json.loads(path.read_text()) for path in
                          sorted((destination/'options').glob('*/execution.json'))]
            record.update(accepted_ah=sum(bool(item.get('accepted_for_training')) for item in executions),
                ah_attempts=len(executions), decisions=len(list(destination.glob('decision_*.json'))),
                wall_seconds=time.time() - started)
            changed = [str(path) for path in sources
                       if hashlib.sha256(path.read_bytes()).hexdigest() != config['source_sha256'][str(path)]]
            if changed:
                record.update(status='infrastructure_error', evaluation_valid=False,
                    infra_error='source changed during rollout', changed_sources=changed)
            atomic_json(destination/'result.json', record)
            results.append(record)
            completed.add(index)
            if depth_synchronous:
                client = session_clients.pop(index, None)
                if client is not None:
                    client.close()
            summary()
            print(json.dumps(record, ensure_ascii=False), flush=True)
            return record

        def initialize(slot, index, row):
            destination = args.output/f'case_{index:03d}'
            destination.mkdir(exist_ok=True)
            record = dict(index=index, scene=row['scene'], episode=row['task'],
                cohort_index=cohort_index, actor_slot=slot, gym_url=urls[slot])
            if row.get('backend'):
                record['backend'] = row['backend']
            states[index] = destination, record, time.time()
            print(json.dumps(dict(event='start', **record)), flush=True)
            teacher, case = build_case(row, destination, urls[slot])
            if os.environ.get('SAGE_NAVMESH_ROOT') and isinstance(teacher.dataset, SageSet):
                case['sage_floor_profile'] = 'sage-executed-floor-v1'
                def record_floor(proof, case=case, destination=destination):
                    case['sage_floor_contract'] = proof
                    atomic_json(destination/'case.json', case)
                teacher.floor_contract_observer = record_floor
                def record_native_scope(proof, case=case, destination=destination):
                    case['sage_native_route_admission'] = proof
                    atomic_json(destination/'case.json', case)
                teacher.native_route_admission_observer = record_native_scope
            evaluator = ActionHeadEvaluator.__new__(ActionHeadEvaluator)
            version = getattr(args, 'action_head_as_tool_version', 'v6')
            simulation_timeout = configure_collection_simulation_timeout(teacher, version)
            evaluator.args = SimpleNamespace(bench_url=urls[slot], vllm_url=args.brain_url,
                    mode='sft', timeout=None if version == 'v8' else 240,
                    max_turns=args.max_calls, max_tool_calls=args.max_calls,
                    keep_last_images=8, checkpoint='rule_teacher_no_head',
                    one_tool_call_per_turn=True, capture_slow_messages=True)
            evaluator.action_head_as_tool_version = version
            evaluator.protocol = PV.protocol_module(version)
            evaluator.bench = ReplayBenchClient(urls[slot], timeout=simulation_timeout,
                    teacher=teacher, destination=destination)
            if depth_synchronous:
                session_clients[index] = evaluator.bench
            if validate is not None:
                validate(evaluator.bench, args, row)
            evaluator.runtime = teacher
            evaluator.profile = A.get_tool_profile('action_head', action_head_as_tool_version=version)
            def observe(destination=destination, **kwargs):
                atomic_json(destination/f"decision_{kwargs['decision_index']:03d}.json", kwargs)
            evaluator.decision_observer = observe
            from sft.rule_teacher_progress import DepthProgressTracker
            progress_tracker = DepthProgressTracker(limit=3)
            seen_executions = set()
            def completed_depth(record, destination=destination, tracker=progress_tracker,
                                seen=seen_executions):
                paths = sorted((p for group in ('options', 'turns')
                                for p in (destination/group).glob('*/execution.json')
                                if p not in seen), key=lambda p: int(p.parent.name.split('_', 1)[0]))
                executions = [json.loads(path.read_text()) for path in paths]
                if not executions and any(call.get('ok') for call in record.get('calls', [])):
                    raise ValueError('successful_depth_missing_actual_execution_receipt')
                proof = tracker.complete_depth(record['turn'], executions)
                proof['execution_paths'] = [str(path) for path in paths]
                seen.update(paths)
                record['actual_progress'] = proof
                atomic_json(destination/'progress_status.json', proof)
                with (destination/'progress_events.jsonl').open('a') as stream:
                    stream.write(json.dumps(proof, ensure_ascii=False) + '\n')
                return proof
            evaluator.completed_depth_observer = completed_depth
            atomic_json(destination/'case.json', case)
            adapter = candidate_query(teacher, case, destination, args.candidates,
                'v6_depth' if depth_synchronous else args.candidate_strategy)
            return evaluator, case, adapter, deterministic_seed(20260930, case)

        def lazy_episode(slot, index, row, adapter):
            # An entry is started only when the scheduler first advances it.
            # A fatal reset of an earlier entry leaves later tasks unstarted.
            evaluator, case, delegate, seed = initialize(slot, index, row)
            for method in ('prepare', 'next_wave', 'sample', 'resolve'):
                setattr(adapter, method, getattr(delegate, method))
            try:
                return (yield from evaluator.rollout_steps(case, seed))
            finally:
                delegate.close()

        for slot, index, row in ([] if rolling else pending):
            if args.depth_batch_size == 1 and not depth_synchronous:
                try:
                    evaluator, case, adapter, seed = initialize(slot, index, row)
                    original_query = E.chat_query
                    try:
                        E.chat_query = adapter
                        finish(index, evaluator.rollout(case, seed))
                    finally:
                        E.chat_query = original_query
                except Exception as exc:
                    if index in completed:
                        raise
                    finish(index, error=exc)
                    if isinstance(exc, ServingGenerationError):
                        raise
                raise_if_serving_error(states[index][0])
            else:
                adapter = SimpleNamespace()
                entries.append((index, lazy_episode(slot, index, row, adapter), adapter))
        def refill(active_keys):
            occupied = {actor_slots[key] for key in active_keys}
            active_scenes = {rows[key]['scene'] for key in active_keys}
            for slot in range(capacity):
                if slot in occupied:
                    continue
                if scene_refill:
                    # Fill a resident scene first. Admit a new scene only when
                    # a scene's last active episode has released its cache slot.
                    scene = next((scene for scene, queue in scene_backlog.items()
                                  if queue and scene in active_scenes), None)
                    if scene is None and len(active_scenes) < scene_limit:
                        scene = next((scene for scene, queue in scene_backlog.items() if queue), None)
                    item = scene_backlog[scene].popleft() if scene is not None else None
                else:
                    item = next(backlog, None)
                if item is None:
                    return
                _, index, row = item
                active_scenes.add(row['scene'])
                actor_slots[index] = slot
                adapter = SimpleNamespace()
                yield index, lazy_episode(slot, index, row, adapter), adapter

        if entries or rolling:
            from rule_teacher_depth_scheduler import run_depth_cohort
            def on_depth(event):
                event = dict(event, cohort_index=cohort_index, time=time.time())
                if event.get('phase') == 'completed':
                    worker_module = sys.modules.get('native_compile_worker')
                    compile_pool = getattr(worker_module, '_POOL', None)
                    if compile_pool is not None and hasattr(compile_pool, 'snapshot_metrics'):
                        event['native_compile'] = compile_pool.snapshot_metrics()
                if scene_refill:
                    event['active_scenes'] = list(dict.fromkeys(rows[key]['scene'] for key in event.get('active_cases', [])))
                    event['rolling_scene_limit'] = scene_limit
                    event['rolling_scene_refill'] = True
                atomic_json(args.output/'depth_status.json', event)
                with (args.output/'depth_events.jsonl').open('a') as stream:
                    stream.write(json.dumps(event) + '\n')
            run_depth_cohort(entries, max_workers=min(getattr(args, "http_concurrency", 64),
                min(args.depth_batch_size, len(urls)) * args.candidates),
                on_result=lambda index, rollout: finish(index, rollout),
                on_error=lambda index, exc: finish(index, error=exc), on_depth=on_depth,
                refill=refill if rolling else None,
                actor_workers=getattr(args, 'actor_workers', 1),
                overlap_validation=getattr(args, 'overlap_candidate_validation', False),
                depth_synchronous=depth_synchronous,
                stop_on_serving_error=getattr(args, 'action_head_as_tool_version', 'v6') != 'v8',
                parallel_initialization=(depth_synchronous and
                    os.environ.get('RULE_TEACHER_PARALLEL_INITIALIZE') == '1'))
            for index, _, _ in entries:
                raise_if_serving_error(states[index][0])


def collection_protocol_arguments(parser):
    parser.add_argument('--action-head-as-tool-version', choices=('v6', 'v8'), default='v6')
    parser.add_argument('--route-prefix-steps', type=int, choices=(4, 16), default=4,
                        help='Subdivide each same physical refresh segment; does not extend the route')


def configure_collection_protocol(args):
    os.environ['RULE_TEACHER_ROUTE_PREFIX_STEPS'] = str(args.route_prefix_steps)
    os.environ['RULE_TEACHER_CONTEXT_VERSION'] = args.action_head_as_tool_version


def configure_collection_simulation_timeout(teacher, version):
    """Let v8 collection wait for its own simulator, without replaying a step.

    These clients are owned by the new episode. Their existing HTTP transports
    already accept None; leave shared defaults and legacy callers unchanged.
    This does not change the model worker's separate RPC timeout.
    """
    if version != 'v8':
        return 180
    for name in ('render_client', 'renderer'):
        client = getattr(teacher, name, None)
        if client is not None:
            client.timeout = None
    return None


def collection_runtime_config(args):
    steps = getattr(args, 'route_prefix_steps', 4)
    return dict(execution_contract='full_target_route_v1' if steps == 4 else 'full_target_route_v2_fine16',
        prefix_cache=os.environ.get('RULE_TEACHER_V8_PREFIX_CACHE') == '1',
        simulation_http_timeout_s=None if args.action_head_as_tool_version=='v8' else 180,
        minimum_chunks=args.fast_step_limit,
        exact_fast_calls=4 if os.environ.get('RULE_TEACHER_EXACT_FOUR_FAST_CALLS','0')=='1' else None,
        prediction_steps=16, prefix_steps=steps,
        budget='collector_tool_calls_only' if args.action_head_as_tool_version=='v8' else 'derived_from_complete_route')


def parser():
    p=argparse.ArgumentParser()
    collection_protocol_arguments(p)
    p.add_argument('--output',type=Path,default=ART/'rule-teacher-v6-local-20260930-r1/pilot')
    p.add_argument('--limit',type=int,default=12)
    p.add_argument('--max-calls',type=int,default=12)
    p.add_argument('--candidates',type=int,choices=(1,2,4,8),default=8)
    p.add_argument('--candidate-strategy',choices=('adaptive','fixed'),default='fixed')
    p.add_argument('--sampling-scheduler', choices=('v6_depth','mixed_depth'), default='v6_depth')
    p.add_argument('--route-min-chunks','--fast-step-limit',dest='fast_step_limit',type=int,default=4,metavar='CHUNKS',
                   help='Minimum refreshes over the complete route; each predicts 16 poses and executes 4')
    p.add_argument('--select-only',action='store_true')
    p.add_argument('--selection-mode',choices=('historical','unfiltered','object-endpoint'),default='historical')
    p.add_argument('--sage-split',default='official_gvln_action_head_p0_v3full827train',
                   help='Native teacher dataset split; formal existing five-type inputs require train')
    p.add_argument('--exclude-selection',type=Path)
    p.add_argument('--min-ref-len',type=float,default=0.,help='Source reference length stratum, before any model calls')
    p.add_argument('--navigation-only',action='store_true',help='Original instructions beginning with a navigation verb')
    p.add_argument('--audit-initial-candidates',type=Path)
    p.add_argument('--gym-url',default='http://127.0.0.1:18974')
    p.add_argument('--gym-urls',help='JSON list of independent Gym actor URLs')
    p.add_argument('--depth-batch-size',type=int,default=1)
    p.add_argument('--http-concurrency',type=int,choices=range(1,513),default=64)
    p.add_argument('--actor-workers',type=int,choices=range(1,9),default=1,
                   help='Persistent actor execution lanes; each episode stays on one thread.')
    p.add_argument('--overlap-candidate-validation', action='store_true',
                   help='Validate complete case replies while peer HTTP requests are still running.')
    p.add_argument('--rolling-actors', action='store_true',
                   help='Refill independent Gym slots within each scene group.')
    p.add_argument('--rolling-scene-limit', type=int, default=0,
                   help='Maximum scenes in a rolling group; 0 preserves unbounded legacy scheduling.')
    p.add_argument('--rolling-scene-refill', action='store_true',
                   help='Refill across scene boundaries while limiting active scenes to rolling-scene-limit.')
    p.add_argument('--render-url',default='http://127.0.0.1:7074')
    p.add_argument('--brain-url',default='http://127.0.0.1:28970')
    return p


def main():
    p=parser()
    args=p.parse_args()
    configure_collection_protocol(args)
    try: gym_urls=actor_urls(args)
    except (ValueError, TypeError) as exc: p.error(str(exc))
    if args.rolling_scene_limit < 0: p.error('rolling-scene-limit must be nonnegative')
    if args.rolling_scene_refill and (not args.rolling_actors or args.rolling_scene_limit < 1):
        p.error('rolling-scene-refill requires rolling actors and a positive scene limit')
    if args.overlap_candidate_validation and args.actor_workers == 1:
        p.error('overlap-candidate-validation requires at least two actor-workers')
    if args.min_ref_len < 0: p.error('min-ref-len must be nonnegative')
    if args.selection_mode=='historical' and (args.min_ref_len or args.navigation_only):
        p.error('reference length / navigation filters require an original-source selection mode')
    if args.fast_step_limit < 1: p.error('route-min-chunks / fast-step-limit must be positive')
    if not args.output.resolve().is_relative_to(ART): p.error('output must be on artifact disk')
    args.output.mkdir(parents=True,exist_ok=True)
    selection=args.output/'selection.json'
    if args.audit_initial_candidates and not selection.exists():
        shutil.copyfile(args.audit_initial_candidates/'selection.json',selection)
    if not selection.exists():
        if args.sage_split != 'official_gvln_action_head_p0_v3full827train':
            p.error('nondefault SAGE split requires a frozen selection.json')
        select_cases(selection,args.limit,args.selection_mode,args.exclude_selection,args.min_ref_len,args.navigation_only)
    selection_record=json.loads(selection.read_text())
    validate_selection_split(selection_record,args.sage_split)
    rows=selection_record['cases'][:args.limit]
    if args.select_only:
        print(json.dumps([dict(scene=r['scene'],task=r['task'],call=r.get('arguments')) for r in rows],ensure_ascii=False,indent=2));return
    if args.sampling_scheduler == 'v6_depth':
        os.environ['RULE_TEACHER_SAMPLING_PROFILE'] = 'v6_depth'
    sampling=brain_sampling_contract(0)
    if not sampling['matches_sft_bundle_generation']:
        raise RuntimeError('pilot requires the original SFT brain sampling contract')
    sources=[Path(__file__),*[ROOT/name for name in (
        'vln_demo/src/rule_teacher.py','vln_demo/src/rule_semantics.py','vln_demo/src/rule_lexicon.py',
        'vln_demo/src/rule_map_grounding.py','vln_demo/src/rule_text_retrieval.py','vln_demo/src/rule_match_policy.py',
        'vln_demo/src/rule_nlp_grounding.py','vln_demo/src/rule_call_nlp.py','vln_demo/src/rule_plan_chunks.py',
        'third_party/Fine-Grained-R2R-NOTICE.txt','scripts/eval/serve_rule_call_nlp.py',
        'vln_demo/src/privileged_teacher.py','vln_demo/src/rule_route_execution.py',
        'vln_demo/src/sage/env.py','vln_demo/src/sage/goal_region.py',
        'vln_demo/src/sage/navmesh_support.py','vln_demo/src/sage/navmesh_map.py',
        'vln_demo/src/sage/same_floor_scope.py',
        'vln_demo/src/mp3d_rule_teacher.py',
        'vln_demo/src/sage/planner.py','vln_demo/src/sage/occmap.py','vln_demo/src/agent_tools.py',
        'vln_demo/src/verl_access/context_budget.py','vln_demo/src/verl_access/serving_error.py',
        'scripts/eval/rule_teacher_depth_scheduler.py',
        'vln_demo/src/vln_gym/env.py','vln_demo/src/vln_gym/fm_action.py','verl/action_head/render_client.py',
        'verl/sft/action_head_data_contract.py',
        'verl/action_head/runtime.py','scripts/eval/run_rule_teacher_local.sh',
        'verl/action_head/protocol.py','verl/action_head/protocol_v6.py',
        'verl/action_head/protocol_v8.py','verl/action_head/trajectory_map.py',
        'verl/scripts/eval/qwen36_4query_ckpt_sweep/closedloop_eval.py',
        'verl/scripts/eval/action_head_fm_p0_p2/curriculum_action_head_eval.py')]]
    mesh_config=ROOT/'vln_demo/config/rule_mesh_semantics.json'
    if mesh_config.exists():sources.append(mesh_config)
    hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}
    scope=json.loads(selection.read_text())['scope']
    config=dict(source_sha256=hashes,harness=args.action_head_as_tool_version,brain='Qwen3.6-35B',brain_sampling=sampling,max_calls=args.max_calls,candidates=args.candidates,candidate_strategy=args.candidate_strategy,teacher='text_rules_only',args={'limit':args.limit},
                selection_scope=scope,selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest())
    config['depth_batch_size']=args.depth_batch_size
    config['sampling_scheduler']=args.sampling_scheduler
    config['parallel_initialization'] = (args.sampling_scheduler == 'v6_depth' and
        os.environ.get('RULE_TEACHER_PARALLEL_INITIALIZE') == '1')
    config['rejected_depth_retries']=1 if args.sampling_scheduler == 'v6_depth' else None
    config['consecutive_no_progress_limit'] = 3
    config['drop_repeated_fragments_at_export'] = True
    config['http_concurrency']=args.http_concurrency
    config['actor_workers']=args.actor_workers
    config['overlap_candidate_validation']=args.overlap_candidate_validation
    config['rolling_actors']=args.rolling_actors
    config['rolling_scene_limit']=args.rolling_scene_limit
    config['rolling_scene_refill']=args.rolling_scene_refill
    config['sage_split']=args.sage_split
    config['gym_urls']=gym_urls
    config['context_budget']=context_budget_policy()
    config['nlp_url']=os.environ.get('RULE_NLP_URL','')
    config['admission_mode']=os.environ.get('RULE_TEACHER_ADMISSION','strict')
    config['grounding_mode']=os.environ.get('RULE_TEACHER_GROUNDING','lexical')
    config['match_policy']=os.environ.get('RULE_TEACHER_MATCH_POLICY','target_and_direction')
    config['text_embedding_model']=os.environ.get('RULE_TEXT_EMBEDDING_MODEL')
    from rule_map_grounding import mesh_history_configuration
    history_config=mesh_history_configuration()
    config['mesh_semantic_history']=history_config['database']
    config['mesh_semantic_history_sha256']=history_config['sha256']
    config['supervision_scope']=os.environ.get('RULE_TEACHER_SCOPE','target_direction')
    config['runtime_config']=collection_runtime_config(args)
    if os.environ.get('SAGE_NAVMESH_ROOT'):
        from sage.navmesh_map import manifest_for
        manifests = {}
        missing = {}
        for scene in sorted({row['scene'] for row in rows}):
            path = manifest_for(os.environ['SAGE_NAVMESH_ROOT'], scene).resolve()
            try:
                manifest_bytes = path.read_bytes()
            except FileNotFoundError:
                missing[scene] = dict(path=str(path), reason='native_bank_manifest_missing')
                continue
            manifests[scene] = dict(path=str(path), sha256=hashlib.sha256(manifest_bytes).hexdigest())
        config['runtime_config']['sage_floor_profile'] = 'sage-executed-floor-v1'
        config['runtime_config']['sage_navmesh_manifests'] = manifests
        config['runtime_config']['sage_navmesh_missing'] = missing
    if (args.gym_url,args.render_url,args.brain_url) != (
            'http://127.0.0.1:18974','http://127.0.0.1:7074','http://127.0.0.1:28970'):
        config['service_urls']=dict(gym=args.gym_url,render=args.render_url,brain=args.brain_url)
    if args.audit_initial_candidates:config['audit_initial_candidates']=str(args.audit_initial_candidates.resolve())
    config['git_runtime'] = collection_git_provenance()
    config_path=args.output/'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError('source or configuration changed; use a new output directory, not mixed-version resume')
    atomic_json(config_path,config)
    if args.audit_initial_candidates:
        audit_initial_candidates(args.audit_initial_candidates,args.output,rows,
                                 gym_url=args.gym_url,render_url=args.render_url)
        return
    def build_case(row, destination, gym_url):
        teacher = RuleTeacher(gym_url=gym_url, vlm_url='DISABLED', render_url=args.render_url,
            artifact_dir=destination/'options', fast_step_limit=args.fast_step_limit, split=args.sage_split)
        teacher.native_route_scope = row.get('native_route_scope')
        teacher.native_route_provenance = row.get('instruction_provenance')
        return teacher, case_for_row(row, teacher.task(row['scene'], row['task']))
    collect_cases(rows, args, build_case, scope=scope, sources=sources, config=config)


if __name__=='__main__': main()
