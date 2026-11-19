"""Target-first admission using the shared mesh linker and physical planner.

Directions and referent relations rank target hypotheses. Only the fallback
requires direction agreement. Neither branch overrides physical reachability.
"""
import base64
import io
import json
import math
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

from privileged_teacher import TeacherRejected, opening_object, wrap
from rule_lexicon import NAMES, ROOMS
from rule_map_grounding import (OpenMapCandidates, MeshSemanticHistory,
    map_semantic_scene, nominal_mentions, normalize, open_object_spec)
from rule_route_execution import prepare_full_route
from sage.planner import _path_is_safe
from sage.route_extractor import _bbox_distance_xy
from sage.visibility_audit import project_bbox, depth_support


def direction_spec(contract):
    """Conservative ego fallback; object-relative left/right is not ego motion."""
    where=normalize(contract.get('where',''))
    location=normalize(contract.get('target',''))+' '+where
    location=re.sub(r'\b(?:left|right)\s+(?:side\s+)?of\s+(?!(?:the\s+)?(?:current|view|image)\b)(?:the\s+)?[^,;.]+','',location)
    first=re.split(r'[.;]|\bthen\b',normalize(' '.join(contract.get('plan',[]))),maxsplit=1)[0]
    head=re.split(r'\b(?:towards?|to(?!\s+(?:the\s+)?(?:left|right)\b)|near|through|into|on|with)\b',first,maxsplit=1)[0]
    # A first explicit turn is the current action. A later target/location
    # relation (e.g. door on the right after turning left) cannot reverse it.
    turn = re.match(r'^(?:please\s+)?turn\s+(?:slightly\s+|a little\s+)?(?:to\s+(?:the\s+|your\s+)?)?(left|right)\b', first)
    if turn:
        direction = turn.group(1)
        angle = 45. if re.search(r'\b(?:slightly|a little)\b', turn.group(0)) else 90.
        return dict(direction=direction, angle_deg=angle if direction=='left' else -angle,
                    source=first, evidence='first_explicit_turn')
    # Reuse the current-motion parser used by admission. Target/where sides
    # describe the referent; they cannot override an explicit first translation.
    # A following coordinated navigation action belongs to a later stage.
    from rule_teacher import parse_target_direction_contract
    current = re.split(r'\band\s+(?=(?:turn|move|walk|go|approach|reach|proceed|continue|enter|exit|leave|follow|pass|face|align|head|advance|navigate|cross|stop)\b)',
                       first, maxsplit=1)[0]
    motion = parse_target_direction_contract(
        dict(contract, plan=[current]), select_prefix=False,
        entity_spec={'family': 'direction'})
    if motion['forward'] or motion['backward']:
        direction = 'backward' if motion['backward'] else 'forward'
        angle = 180. if direction == 'backward' else 0.
        if direction == 'forward' and motion['move_side'] in ('left', 'right'):
            angle = 45. if motion['move_side'] == 'left' else -45.
        return dict(direction=direction, angle_deg=angle, source=current,
                    evidence='first_explicit_motion')
    directions=re.findall(r'\b(left|right|ahead|front|behind)\b',location)
    source=location
    if not directions:
        directions=re.findall(r'\b(left|right|forward|backward|backwards|ahead|behind)\b',head)
        source=head
    direction=directions[0] if directions else None
    angle={'left':90.,'right':-90.,'backward':180.,'backwards':180.,'behind':180.,
           'forward':0.,'ahead':0.,'front':0.}.get(direction)
    if set(directions)&{'ahead','front','forward'} and ('left' in directions)^('right' in directions):
        angle=45. if 'left' in directions else -45.
    return dict(direction=direction,angle_deg=angle,source=source)


def direction_agrees(bearing, direction):
    return direction['angle_deg'] is None or abs(math.degrees(wrap(math.radians(bearing-direction['angle_deg']))))<=65.


def explicit_navigation_stage(prefix):
    """Honor an actual first endpoint, not incidental route landmarks."""
    if prefix.get('mode')!='first_navigation_endpoint':return None
    action=normalize(prefix.get('endpoint_action',{}).get('lemma',prefix.get('endpoint_action',{}).get('text','')))
    relation=normalize(prefix.get('endpoint_relation',{}).get('text',''))
    lexical=prefix.get('endpoint_entity',{}).get('lexical') or {}
    if action in {'approach','reach'}:return action
    if lexical.get('kind')=='room' or lexical.get('canonical')=='doorway':
        if relation in {'into','through'}:return relation
        if action=='enter':return action
    return None


def unresolved_nominal_part(field, main, scene_labels):
    """Keep an unknown part attached to its physical parent, not a proxy label.

    For example, the parser's ``foot -> of -> bed`` is not a request for an
    independently retrievable foot/floor object. No generic object-part geometry
    is available here. Exact scene labels and already known object categories
    keep their existing handling; an unresolved part can still use ego motion.
    """
    if (not main or main['kind'] != 'object' or main.get('family') is not None
            or main['query'] in scene_labels):
        return None
    nominal_tokens = {token['i'] for token in main['tokens']}
    entities = {entity['id']: entity for entity in field.get('entities', [])}
    for relation in field.get('relations', []):
        if (normalize(relation.get('predicate', relation.get('text', ''))) != 'of'
                or relation.get('head_token') not in nominal_tokens):
            continue
        parents = [entities[oid] for oid in relation.get('objects', []) if oid in entities]
        if any((entity.get('lexical') or {}).get('kind') == 'object' for entity in parents):
            return dict(schema='unresolved_nominal_part_v1', nominal=main['text'],
                source_span=main['span'], relation=relation, parents=parents,
                reason='part-of geometry unresolved; independent label retrieval cannot establish the requested part')
    return None


def prepare_target_or_direction(teacher, *, scene, episode, pose, rgb, contract, graph, output, parse_language, snapshot=None):
    if graph is None:raise RuntimeError('target_or_direction requires RULE_NLP_URL')
    from rule_text_retrieval import TextRetriever
    from rule_teacher import parse_target_direction_contract,select_navigation_prefix
    from vln_gym.env import _jpg_b64
    occ,objects,openings=teacher.scene(scene)
    semantic=map_semantic_scene(teacher.semantic_scene(scene,objects,openings))
    if teacher._text_retriever is None:
        if not teacher.text_embedding_model:raise RuntimeError('target_or_direction requires text embedding model')
        teacher._text_retriever=TextRetriever(teacher.text_embedding_model)
    if teacher.mesh_semantic_history_path and (teacher._mesh_semantic_history is None or
            teacher._mesh_semantic_history.path!=str(Path(teacher.mesh_semantic_history_path).resolve())):
        teacher._mesh_semantic_history=MeshSemanticHistory(teacher.mesh_semantic_history_path)
    retrieval=OpenMapCandidates(semantic,teacher._text_retriever,
        history=teacher._mesh_semantic_history if teacher.mesh_semantic_history_path else None,scene=scene,broaden=True)
    semantic.candidate_matcher=retrieval.match
    warnings=[];direction=direction_spec(contract)
    original_contract=contract
    original_mentions=nominal_mentions(next(f for f in graph['fields'] if f['field']=='target'))
    original_main=original_mentions[0] if original_mentions else None
    stage=None
    try:
        local,prefix=select_navigation_prefix(contract,graph,open_vocabulary=True)
        stage=explicit_navigation_stage(prefix)
        if (original_main and original_main['kind']=='object'
                and original_main.get('family')!='doorway' and stage is None):
            # The explicit object is the requested endpoint. Intermediate
            # landmarks such as "pass the back of the sofa" or "go toward
            # the wall" guide the route, not the object's identity.
            prefix=dict(mode='explicit_object_target',original_target=contract['target'],
                proposed_navigation_prefix=prefix)
            local=contract
        if local!=contract:
            graph=parse_language(local)
            contract=local
            if stage:direction=direction_spec(local)
    except TeacherRejected as exc:
        prefix=dict(mode='explicit_call_target_fallback',reason=str(exc))
        warnings.append(dict(rule='navigation_prefix',reason=str(exc)))
    fields={f['field']:f for f in graph['fields']}
    mentions=nominal_mentions(fields['target'])
    main=mentions[0] if mentions else None
    nominal_part=unresolved_nominal_part(fields['target'],main,retrieval.labels)
    if nominal_part:
        # Preserve the original contract and its part/parent relation. The
        # existing direction branch remains available under target OR direction.
        main=dict(main,kind='region',family=None)
        warnings.append(dict(rule='unresolved_nominal_part',evidence=nominal_part))
    # Recover the full leading mesh noun compound when POS tagging splits it
    # (e.g. a noun followed by a word tagged as a verb). This uses this scene's
    # actual labels, never a scene-specific synonym table.
    leading=re.split(r'\b(?:on|near|with|from|behind|beside|in|at|through)\b',normalize(contract['target']),maxsplit=1)[0]
    literal=[label for label in retrieval.labels if re.search(r'(?<!\w)'+re.escape(label)+r'(?!\w)',leading)]
    if literal and nominal_part is None:
        label=max(literal,key=len)
        if main is None or (main['kind']=='object' and len(label)>len(main['query'])):
            main=dict(query=label,kind='object',family=NAMES.get(label),text=leading.strip())
    spec=dict(family=None,kind='approach',relations=[],entity_relation_alternatives=[],target_room=None)
    if main and main['kind']=='object' and main.get('family')!='doorway':
        spec['family']=retrieval.recall(main['query'])
        try:
            rich=open_object_spec(contract,graph,retrieval)
            if rich is not None:
                rich['family']=spec['family']
                spec=parse_target_direction_contract(contract,graph,select_prefix=False,entity_spec=rich)
        except TeacherRejected as exc:warnings.append(dict(rule='soft_language_constraint',reason=str(exc)))
    portal_constraints_failed=False
    if main and main.get('family')=='doorway':
        try:
            # A named destination is part of the door's identity, not a
            # ranking hint that any reachable opening can replace.
            spec=parse_target_direction_contract(contract,graph,select_prefix=False)
        except TeacherRejected as exc:
            portal_constraints_failed=True
            warnings.append(dict(rule='unresolved_portal_constraint',reason=str(exc)))
    rendered,depth=teacher.renderer.render_one(scene,teacher.camera_position(pose),math.radians(pose[2]),want_depth=True)
    comparable=np.asarray(Image.open(io.BytesIO(base64.b64decode(_jpg_b64(rendered)))).convert('RGB'),float)
    mae=float(np.abs(comparable-np.asarray(rgb,float)).mean())
    if mae>3:raise RuntimeError(f'current-state render mismatch {mae}')
    from rule_target_scope import TargetSearchScope
    search_scope = TargetSearchScope(teacher, scene=scene, episode=episode,
        pose=pose, rgb=rgb, depth=depth, snapshot=snapshot)
    evidence=dict(schema='rule-teacher-v1',scene=scene,episode=episode,original_call=original_contract,planning_call=contract,navigation_prefix=prefix,
        pose=pose,parsed=spec,admission_mode=teacher.admission_mode,grounding_mode=teacher.grounding_mode,
        match_policy='target_or_direction',direction=direction,render_rgb_mae=mae,
        candidate_audit=[],candidates=[],relaxations=warnings,open_map_retrieval=retrieval.audit,
        target_identity_verified=False)
    evidence['target_search_scope'] = search_scope.audit
    yaw=math.radians(pose[2]);candidates=[]
    current_room=semantic.room_at(pose[:2])
    local_room=ROOMS.get(main['query']) if stage and main and main['kind']=='room' else None
    already_in_room=bool(local_room and semantic.room_evidence(current_room,local_room,pose[:2]))
    stay_in_room=already_in_room and stage in {'into','enter'}
    evidence['first_navigation_stage']=dict(action=stage,current_room=current_room,
        requested_room=local_room,already_in_room=already_in_room,stay_in_room=stay_in_room)
    def add(oid,obj,match,program,kind='object',constraints=None):
        delta=np.asarray(obj['centre_xy'])-pose[:2]
        bearing=math.degrees(wrap(math.atan2(delta[1],delta[0])-yaw))
        projection=project_bbox(teacher.camera,teacher.camera_position(pose),yaw,obj)
        support=depth_support(depth,projection['bbox_px'],teacher.camera,teacher.camera_position(pose),yaw,obj,stride=8) if projection else dict(visible_rays=0,visible_fraction=0.)
        visibility = search_scope.admit(oid, obj, projection, support)
        if visibility is None:
            return
        candidates.append(dict(id=oid,label=obj['label'],ins_id=obj.get('ins_id'),lexical_match=match,
            bearing_deg=bearing,direction_agrees=direction_agrees(bearing,direction),
            surface_distance_m=_bbox_distance_xy(pose[:2],obj['bbox_xyxy']),projection=projection,depth=support,
            semantic_constraints=constraints,historical_projection=visibility['source']=='history',
            target_visibility=visibility,program=program,target_kind=kind))
    if spec['family'] and spec['family']!='doorway':
        for oid,obj in objects.items():
            if not search_scope.nearby(obj):continue
            match=retrieval.match(oid,obj,spec['family'])
            if match is None:continue
            constraints=semantic.candidate_evidence(oid,obj,spec,pose,NAMES,normalize)
            add(oid,obj,match,dict(status='ok',steps=[dict(kind=spec['kind'],object=oid,side='nearest',standoff_m=.3)]),constraints=constraints)
    elif main and (main.get('family')=='doorway' or main['kind']=='room'):
        room=ROOMS.get(main['query']) if main['kind']=='room' else None
        for oid,opening in openings.items():
            obj=opening_object(oid,opening)
            if not search_scope.nearby(obj):continue
            if portal_constraints_failed:continue
            constraints=None
            near,far,_,far_xy=semantic.portal_sides(opening,pose)
            if stage and (stay_in_room or (current_room is not None and near!=current_room)):
                continue
            if room:
                # "Through" the room we occupy may use its exit; "into"
                # that room continues locally instead of finding another one.
                constraints=semantic.room_evidence(near if already_in_room and stage=='through' else far,
                    room,pose[:2] if already_in_room and stage=='through' else far_xy)
                if constraints is None:continue
            if spec.get('family')=='doorway':
                constraints=semantic.candidate_evidence(oid,obj,spec,pose,NAMES,normalize)
                if constraints is None:continue
            add(oid,obj,dict(method='native_room_portal' if room else 'native_opening',score=1.),
                dict(status='ok',steps=[dict(kind='portal',opening=oid)]),kind='room' if room else 'opening',constraints=constraints)
    # These are ranking signals only. A reachable target survives an opposite
    # direction, failed relation hypothesis, or current occlusion.
    candidates.sort(key=lambda c:(-int(c['depth'].get('visible_rays',0)>0),
        -int(c['direction_agrees']),-int(c['semantic_constraints'] is not None),
        -c['lexical_match']['score'],c['surface_distance_m'],c['id']))
    evidence['candidates']=candidates
    state=None;bound=None
    for candidate in candidates:
        audit=dict(id=candidate['id'],stage='physical_planning');evidence['candidate_audit'].append(audit)
        try:
            trial=search_scope.compile_candidate(candidate['program'],occ,objects,openings)
            preview=prepare_full_route(trial,pose,occ,teacher.route_min_chunks,
                                       exact_chunks=getattr(teacher,'route_exact_chunks',None),
                                       prefix_steps=getattr(teacher,'route_prefix_steps',4))
            if (stage or spec.get('family')=='doorway') and candidate['target_kind'] in {'room','opening'}:
                near,far,_,_=semantic.portal_sides(openings[candidate['id']],pose)
                if any(semantic.room_at(p[:2]) not in {near,far,None} for p in preview):
                    raise TeacherRejected('first portal route crosses an unrelated room')
            if candidate['target_kind']=='object':
                distance=_bbox_distance_xy(preview[-1][:2],objects[candidate['id']]['bbox_xyxy'])
                if distance>.9:raise TeacherRejected('target stance farther than 0.9m')
                trial['target_surface_distance_m']=distance
            if (os.environ.get('RULE_TEACHER_CONTEXT_VERSION') != 'v8'
                    and trial['planned_path_m']>25):
                raise TeacherRejected('target route exceeds 25m local segment limit')
            audit['stage']='selected';state=trial;bound=candidate;break
        except TeacherRejected as exc:audit.update(stage='unreachable',reason=str(exc))
        except ValueError as exc:
            if str(exc)!='original centred route is unsafe':raise
            audit.update(stage='unreachable',reason=str(exc))
    if state is None:
        if direction['angle_deg'] is None:
            output.joinpath('binding.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
            raise TeacherRejected('rule: no reachable target and no explicit ego direction')
        angle=yaw+math.radians(direction['angle_deg']);axis=np.array([math.cos(angle),math.sin(angle)])
        failures=[]
        for distance in (2.,1.,.5,.25,.10,.05,.025,.01):
            goal=np.asarray(pose[:2])+distance*axis
            try:
                # Straight free-space fallback cannot silently detour in the
                # opposite direction or claim an arbitrary object identity.
                if not occ.seg_free(*pose[:2],*goal,.07):raise TeacherRejected('direction ray blocked')
                trial=teacher.compile(dict(status='ok',steps=[dict(kind='via',xy=goal.tolist())]),pose,occ,objects,openings)
                preview=prepare_full_route(trial,pose,occ,teacher.route_min_chunks,
                                           exact_chunks=getattr(teacher,'route_exact_chunks',None),
                                           prefix_steps=getattr(teacher,'route_prefix_steps',4))
                if portal_constraints_failed or any(spec.get(key) for key in ('source_room','destination_room','target_room')):
                    # The direction branch remains available, but may not
                    # cross into a room that contradicts the requested door.
                    for point in preview:
                        rid=semantic.room_at(point[:2])
                        if current_room is None:
                            if not any(w['rule']=='direction_room_unresolved' for w in warnings):
                                warnings.append(dict(rule='direction_room_unresolved',
                                    reason='room identity unavailable; only the physical ego direction is admitted'))
                            continue
                        # A missing initial room provides no room constraint.
                        # Once the current room is witnessed, however, None
                        # can also mean a shared door boundary or leaving its
                        # polygon; it must not authorize an unverified entry.
                        if rid is None:
                            raise TeacherRejected('portal direction fallback leaves the witnessed current room')
                        if rid!=current_room and (portal_constraints_failed or not any(
                                semantic.room_evidence(rid,spec[key],point[:2])
                                for key in ('destination_room','target_room') if spec.get(key))):
                            raise TeacherRejected('portal direction fallback enters an unrequested room')
                if stay_in_room and current_room is not None and any(
                        semantic.room_at(p[:2])!=current_room
                        for p in preview):
                    raise TeacherRejected('first into-room direction leaves the current requested room')
                if trial['planned_path_m']>distance*1.2+.05:raise TeacherRejected('direction fallback detour')
                for point in preview:
                    delta=np.asarray(point[:2])-pose[:2]
                    if np.linalg.norm(delta)>.1 and delta@axis/np.linalg.norm(delta)<math.cos(math.radians(30)):
                        raise TeacherRejected('direction fallback deviates from requested direction')
                state=trial;bound=dict(id=None,label='direction only',lexical_match=None,semantic_constraints=None,
                    target_kind='direction',projection=None,historical_projection=False);break
            except TeacherRejected as exc:failures.append(dict(distance_m=distance,reason=str(exc)))
            except ValueError as exc:
                if str(exc)!='original centred route is unsafe':raise
                failures.append(dict(distance_m=distance,reason=str(exc)))
        evidence['direction_fallback_rejections']=failures
        if state is None:
            output.joinpath('binding.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
            raise TeacherRejected('rule: no reachable target or collision-free directional endpoint')
    preview=state['route_target_world_poses'][1:]
    if not _path_is_safe(occ,np.asarray([pose[:2]]+[p[:2] for p in preview]),.07):
        raise TeacherRejected('rule: unsafe controller prediction')
    mode='direction' if bound['target_kind']=='direction' else 'target'
    evidence.update(match_branch=mode,reachable_selection=[bound['id']] if bound['id'] else [])
    state.update(admission_mode=teacher.admission_mode,match_policy='target_or_direction',match_branch=mode,
        relaxation_warnings=warnings,binding=evidence,bound=bound,initial_pose=pose,
        preview_world_xy_yaw_deg=preview,semantic_status='rule_supported_pending_human_review')
    output.joinpath('binding.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    for key in ('sage_floor_contract', 'sage_native_route_admission'):
        proof = getattr(teacher, '_' + key, None)
        if proof is not None:
            import copy
            state[key] = copy.deepcopy(proof)
    output.joinpath('accepted.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
    teacher.map_image(occ,pose,objects,path=[pose[:2]]+[p[:2] for p in preview]).save(output/'teacher_map.png')
    return state
