"""Trajectory planner with conservative text-only grounding and shared geometry/execution.

Objects, relational anchors, room connections and ordered portal/search phases
are grounded against explicit evidence. Unknown constraints cause abstention,
never a rewritten brain call, a teacher language model, or a golden fallback.
"""
from __future__ import annotations

import json
import copy
import hashlib
import shutil
import base64
import io
import math
import os
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from privileged_teacher import PrivilegedTeacher, TeacherRejected, RoutePolicy, preview_option, wrap
from action_head.runtime import ActionHeadToolRuntime, ActionHeadRuntimeConfig
from action_head.render_client import RemotePrefixExecutor, SourceGeometryError, _world_pose
from sage.visibility_audit import project_bbox, depth_support, ray_box_interval
from sage.route_extractor import _bbox_distance_xy
from sage.planner import _path_is_safe
from rule_route_execution import FullRoutePolicy, FullRouteRuntime, prepare_full_route, local_approach_feedback

from rule_lexicon import ALIASES, NAMES, NOUN, PLURAL_TARGETS
FORBIDDEN = re.compile(r'\b(pass|past|beyond|around|between|enter|exit|carry|retrieve|pick|along|behind|back|first|second|third|before|after|avoid|leftmost|rightmost)\b')
WORDS = set('the a an my me our your current view image visible directly straight ahead forward slightly left right side on in at front of to toward towards near nearby nearest closest object target it that this there now is located situated and then turn face facing align with move moving walk walking go going approach approaching continue continuing get closer reach reaching arrive arriving stop stopping beside next position myself until'.split())
WORDS.update(['through','proceed','open'])


def normalized(text):
    return re.sub(r'\s+', ' ', str(text).lower().replace('_', ' ').replace('-', ' ')).strip()


def parse_simple_contract(contract):
    target = normalized(contract.get('target', '')).replace('right in front of', 'directly in front of')
    where = normalized(contract.get('where', '')).replace('right in front of', 'directly in front of')
    plan = contract.get('plan')
    if not isinstance(plan, list) or not plan or not all(isinstance(x, str) for x in plan):
        raise TeacherRejected('rule: plan must be nonempty text list')
    plans = [normalized(x) for x in plan]
    found = list(NOUN.finditer(target))
    if len(found) != 1:
        raise TeacherRejected('rule: target is not a single supported object')
    if found[0].group() in PLURAL_TARGETS:
        raise TeacherRejected('rule: plural main target requires a group goal')
    family = NAMES[found[0].group()]
    traverse = any('through' in text for text in plans)
    if any('through' in text for text in [target,where,*plans]) and family != 'doorway':
        raise TeacherRejected('rule: occluded target beyond a doorway unsupported')
    if any('open' in text.split() for text in [target,where,*plans]) and family != 'doorway':
        raise TeacherRejected('rule: open modifier only supported for traversable openings')
    for text in [target, where, *plans]:
        if FORBIDDEN.search(text):
            raise TeacherRejected('rule: unsupported relation/action: ' + FORBIDDEN.search(text).group())
        for match in NOUN.finditer(text):
            if NAMES[match.group()] != family:
                raise TeacherRejected('rule: another referenced object needs relational grounding')
        remainder = normalized(NOUN.sub(' object ', text))
        unknown = set(re.findall(r"[a-z]+|\d+", remainder)) - WORDS
        if unknown:
            raise TeacherRejected('rule: unsupported words: ' + ','.join(sorted(unknown)))
        if re.search(r'(?:left|right|front) (?:side )?of (?:the )?object', remainder):
            raise TeacherRejected('rule: object-relative stance unsupported')
    sides = set(re.findall(r'\b(left|right)\b', target + ' ' + where))
    if len(sides) > 1:
        raise TeacherRejected('rule: inconsistent target sides')
    movement = any(re.search(r'\b(move|moving|walk|walking|go|going|approach|approaching|reach|reaching|get|arrive|arriving|proceed)\b', p) for p in plans)
    facing = any(re.search(r'\b(face|facing|align)\b', p) for p in plans)
    if not movement and not facing:
        raise TeacherRejected('rule: no supported motion primitive')
    turns = set(re.findall(r'\bturn (?:slightly )?(left|right)\b', ' '.join(plans)))
    if len(turns) > 1:
        raise TeacherRejected('rule: multi-turn sequence unsupported')
    move_sides = set()
    for text in plans:
        if re.search(r'\b(move|moving|walk|walking|go|going|approach|approaching)\b', text) and 'turn' not in text.split():
            move_sides.update(re.findall(r'\b(left|right)\b', text))
    if len(move_sides) > 1:
        raise TeacherRejected('rule: multi-side motion unsupported')
    # All accepted target adjectives and plan tokens have been checked above.
    return dict(family=family, side=next(iter(sides), None),
                ahead=bool(re.search(r'\b(ahead|front)\b', target + ' ' + where)),
                nearest=bool(re.search(r'\b(nearest|closest)\b', target)),
                kind='portal' if traverse else 'approach' if movement else 'face', turn=next(iter(turns), None),
                straight=any('straight' in p for p in plans),
                move_side=next(iter(move_sides), None),
                gentle_move=any('slightly' in p and 'turn' not in p.split() for p in plans),
                forward=any(re.search(r'\b(forward|straight)\b', p) for p in plans),
                directly_ahead=bool(re.search(r'\b(directly|straight)\b', target+' '+where)) and not sides)


def parse_contract(contract):
    try:
        return parse_simple_contract(contract)
    except TeacherRejected:
        from rule_semantics import parse_extended
        return parse_extended(contract, parse_simple_contract, NOUN, NAMES, normalized)


def select_navigation_prefix(contract, graph=None, *, open_vocabulary=False):
    """Choose the first explicit navigation endpoint, never jump to the last.

    Later obligations need not be executable. An unrecognized *earlier*
    endpoint cannot be skipped. The returned call is private compiler input.
    """
    from rule_lexicon import ROOMS
    from rule_nlp_grounding import EGO_PREFIX
    from rule_plan_chunks import segment_plan
    if graph is not None and graph.get('original_call')!=contract:
        raise TeacherRejected('rule NLP: original plan graph does not match the call')
    fields={field['field']:field for field in (graph or {}).get('fields',[])}
    original_target=normalized(contract.get('target',''))
    match=NOUN.search(original_target)
    original_family=NAMES[match.group()] if match else None
    original_mentions=[]
    if open_vocabulary and graph is not None:
        from rule_map_grounding import nominal_mentions
        original_mentions=nominal_mentions(fields.get('target',dict(text='',tokens=[])))
        if original_mentions and original_mentions[0]['kind']=='object':
            original_family='scene:'+original_mentions[0]['query']
    verbs=r'approach|reach|move|walk|go|enter|pass|face|turn|proceed|continue|head|navigate|cross|exit|leave|advance|follow|stop|align'
    route_context_only=object()
    def graph_endpoint(field,chunk,predicate):
        # Reuse the original dependency graph to find this action's nominal
        # endpoint. A later advcl's object ("passing the bed") is route
        # context, never the object of the earlier "through" relation.
        action=next((a for a in field.get('actions',[]) if a['root']==chunk.get('action_token')),None)
        if action is None:return None
        tokens={t['i']:t for t in field.get('tokens',[])}
        def descends(token,head):
            seen=set()
            while token in tokens and token not in seen:
                if token==head:return True
                seen.add(token);token=tokens[token]['head']
            return False
        subordinate=[a for a in field.get('actions',[]) if a.get('dependency')=='advcl'
            and chunk['span'][0]<=a['span'][0]<chunk['span'][1] and descends(a['root'],action['root'])]
        edges=[r for r in field.get('relations',[]) if r.get('head_token')==action['root']
            and normalized(r['text'])==predicate and chunk['span'][0]<=r['span'][0]<chunk['span'][1]]
        if not edges and normalized(action['text'])!=predicate:
            route_predicates=[r.get('head_token') for r in field.get('relations',[])
                if normalized(r['text'])==predicate and chunk['span'][0]<=r['span'][0]<chunk['span'][1]]
            route_predicates += [a['root'] for a in field.get('actions',[])
                if normalized(a['text'])==predicate and chunk['span'][0]<=a['span'][0]<chunk['span'][1]]
            if any(descends(root,a['root']) for root in route_predicates for a in subordinate):
                # The first lexical match may belong only to route wording.
                # Prefer an actual main-action endpoint in this same clause.
                direct=[r for r in field.get('relations',[]) if r.get('head_token')==action['root']
                    and normalized(r['text']) in {'toward','towards','through','into'}
                    and chunk['span'][0]<=r['span'][0]<chunk['span'][1]]
                if not direct:return route_context_only
                edges=[min(direct,key=lambda r:r['span'][0])]
        if edges:
            objects=[oid for edge in edges for oid in edge.get('objects',[])]
        elif normalized(action['text'])==predicate:
            objects=action.get('objects',[])
        else:return None  # Keep existing handling of e.g. "turn to face".
        if len(objects)!=1:
            raise TeacherRejected('rule NLP: first endpoint has unresolved syntax; cannot skip to final target')
        entity=next((e for e in field.get('entities',[]) if e['id']==objects[0]),None)
        if entity is None:
            raise TeacherRejected('rule NLP: missing first endpoint noun phrase; cannot skip to final target')
        start,nominal_end=entity['span'];end=chunk['span'][1]
        if not chunk['span'][0]<=start<nominal_end<=end or field['text'][start:nominal_end]!=entity['text']:
            raise TeacherRejected('rule NLP: first endpoint span does not match original plan')
        contexts=[a for a in subordinate if nominal_end<=a['span'][0]<end
            and not descends(a['root'],entity['root'])]
        context_start=min((a['span'][0] for a in contexts),default=end)
        # Retain adjectives and nominal complements such as "foot of the
        # bed" and "doorway on the left"; only separate the action advcl.
        phrase=field['text'][start:context_start].strip(' ,')
        evidence=dict(endpoint_entity=copy.deepcopy(entity),endpoint_action=copy.deepcopy(action),
            endpoint_source=dict(field=field['field'],span=[start,start+len(phrase)],text=phrase),
            route_context=([dict(field=field['field'],span=[context_start,end],
                text=field['text'][context_start:end],action_ids=[a['id'] for a in contexts])]
                if contexts else []))
        if edges:evidence['endpoint_relation']=copy.deepcopy(edges[0])
        return phrase,entity,evidence
    preceding=[]
    for index,text in enumerate(contract.get('plan',[])):
        if graph is not None:
            if f'plan[{index}]' not in fields:
                raise TeacherRejected('rule NLP: missing original plan field')
            clauses=segment_plan(fields[f'plan[{index}]'])['clauses']
        else:
            clauses=[dict(text=value) for value in re.split(r'[.;]|\b(?:and\s+)?then\b|\band\s+(?=(?:'+verbs+r')\b)',normalized(text))]
        for chunk in clauses:
            clause=normalized(chunk['text'])
            clause=clause.strip(' ,')
            if not clause: continue
            endpoint=re.search(r'\b(approach|reach|face|enter|towards?|through|into)\s+(.+)',clause)
            if endpoint is None:
                endpoint=re.search(r'\b(to)\s+(?!the (?:left|right)\b|(?:left|right)\b)(.+)',clause)
            if endpoint is None:
                # Route wording and other obligations are retained as context,
                # not independent gates for target + initial direction labels.
                preceding.append(clause)
                continue
            phrase=endpoint[2].strip(' ,')
            endpoint_evidence={};entity=None
            if graph is not None:
                selected=graph_endpoint(fields[f'plan[{index}]'],chunk,endpoint[1])
                if selected is route_context_only:
                    preceding.append(clause)
                    continue
                if selected is not None:phrase,entity,endpoint_evidence=selected
            phrase=normalized(phrase)
            if re.fullmatch(r'(?:it|there|the target|the object)',phrase):
                return copy.deepcopy(contract),dict(mode='original_target',source_plan_index=index,source_clause=clause,**endpoint_evidence)
            # A generic region's nominal complement/route landmark cannot
            # turn it into an object endpoint. Preserve it for the compiler,
            # which rejects it until an actual region binding is supported.
            region=bool(entity and (entity.get('lexical') or {}).get('kind')=='region')
            found=NOUN.search(phrase)
            room=phrase.removeprefix('the ') in ROOMS
            open_nominal=None
            if open_vocabulary and graph is not None:
                # Reuse exact plan offsets, repairing split adjacent nouns.
                plan_mentions=nominal_mentions(fields[f'plan[{index}]'])
                open_nominal=next((n for n in plan_mentions if n['span'][0]==endpoint_evidence.get('endpoint_source',{}).get('span',[None])[0]),None)
            if found is None and not room and not region and not open_nominal:
                raise TeacherRejected('rule: first navigation endpoint is unresolved; cannot skip to final target')
            if region:found=None
            family=None if region else NAMES[found.group()] if found else 'doorway'
            if open_nominal and open_nominal['kind']=='object' and open_nominal['family']!='doorway':
                family='scene:'+open_nominal['query']
            bare_reference=bool(found and re.fullmatch(r'(?:the |a |an )?'+re.escape(found.group()),phrase))
            phrase_sides=set(re.findall(r'\b(left|right)\b',phrase))
            original_location=original_target+' '+normalized(contract.get('where',''))
            original_sides=set(re.findall(r'\b(left|right)\b',original_location))
            # A noun plus ego direction ("the doorway ahead") abbreviates
            # this call's qualified referent, just as its bare noun does.
            # Keep the room/object identity in target/where. New identity
            # modifiers and incompatible ego directions are not abbreviations.
            direction_words=set(re.findall(r'[a-z]+',NOUN.sub(' ',phrase)))
            direction_reference=bool(found and len(list(NOUN.finditer(phrase)))==1
                and direction_words <= EGO_PREFIX | {'a','an','forward'}
                and (not phrase_sides or phrase_sides <= original_sides)
                and not (re.search(r'\b(ahead|front|forward)\b',phrase)
                    and re.search(r'\b(behind|back|backward)\b',original_location)))
            # An abbreviated reference to the same object does not erase the
            # original disambiguating text (e.g. "second bedroom"). Only
            # explicitly distinct same-family endpoints may be separated.
            short=lambda value: re.sub(r'^(?:the|a|an)\s+','',value)
            same_target=(family is not None and family==original_family and (bare_reference or direction_reference or
                short(original_target).startswith(short(phrase))))
            if open_nominal and original_mentions:
                identity=lambda node:tuple(t['lemma'].lower() for t in node['tokens'] if t['pos'] in {'ADJ','NOUN','PROPN'})
                head=lambda node:next((t['lemma'].lower() for t in reversed(node['tokens']) if t['pos'] in {'NOUN','PROPN'}),None)
                # A plan may abbreviate an attributed entity (white wardrobe
                # -> wardrobe). Preserve its original target and where.
                same_target |= (head(open_nominal)==head(original_mentions[0])
                    and set(identity(open_nominal))<=set(identity(original_mentions[0]))
                    and short(phrase)==short(open_nominal['text']))
            if same_target or phrase==original_target:
                local=copy.deepcopy(contract)
                local.update(plan=preceding+[clause],_navigation_stage_clause=clause)
                return local,dict(mode='original_target',source_plan_index=index,source_clause=clause,
                    source_span=chunk.get('span'),clause_parser='dependency' if graph is not None else 'lexical',
                    retained_target=contract.get('target'),retained_where=contract.get('where'),**endpoint_evidence)
            original_bare=bool(match and re.fullmatch(r'(?:the |a |an )?'+re.escape(match.group()),original_target))
            if family is not None and family==original_family and not original_bare and not (phrase_sides and original_sides and phrase_sides.isdisjoint(original_sides)):
                raise TeacherRejected('rule: same-family first endpoint has unresolved identity relative to final target')
            local=copy.deepcopy(contract)
            local_where=contract.get('where','') if family==original_family and original_bare else ''
            # In "wardrobe, through the doorway on the right", the where
            # clause describes the intermediate portal and must follow it.
            where=normalized(contract.get('where',''))
            # A location describing the intermediate object is still part of
            # its identity (e.g. pillow "on the bed to the left"). Selecting
            # that bed must not discard the left-side condition.
            if any(NAMES[noun.group()]==family for noun in NOUN.finditer(where)):
                local_where=contract.get('where','')
            if open_nominal and original_mentions and re.search(r'\b(?:view|image|screen)\b',where):
                # A visible support landmark shares the image location of the
                # item on it. Keep the direction when approaching that support.
                for reference in original_mentions[1:]:
                    before=original_target[original_mentions[0]['span'][1]:reference['span'][0]]
                    if reference['query']==open_nominal['query'] and re.fullmatch(r'\s*on\s*',before):
                        local_where=contract.get('where','')
            portal_description=re.search(r'\b(?:through|via)\s+((?:the |a |an )?(?:doorway|opening)\b.*)',where)
            if family=='doorway' and portal_description:
                local_where=portal_description[1]
                # Ego direction before "through" still locates this portal:
                # "straight ahead through the doorway" must not become an
                # unrestricted doorway after choosing the first endpoint.
                ego=where[:portal_description.start()].strip(' ,')
                if ego and set(re.findall(r'[a-z]+',ego)) <= EGO_PREFIX | {'forward'}:
                    local_where=ego+' '+local_where
                # An explicit room target identifies the far side of this
                # portal. Keep that identity when selecting its approach;
                # later search/manipulation stages remain outside this label.
                room_name=original_target.removeprefix('the ')
                room_alias=(room_name if room_name in ROOMS else next(
                    (name for name,value in ROOMS.items() if value==room_name),None))
                if room_alias:
                    local_where+=' leading to the '+room_alias
            local.update(target=phrase,where=local_where,plan=[clause],_navigation_stage_clause=clause)
            # A prior explicit turn/straight direction remains part of the
            # first segment, even if its endpoint appears in the next clause.
            local['plan']=preceding+[clause]
            return local,dict(mode='first_navigation_endpoint',source_plan_index=index,source_clause=clause,
                source_span=chunk.get('span'),clause_parser='dependency' if graph is not None else 'lexical',
                original_target=contract.get('target'),local_target=phrase,
                retained_where=local_where,
                end_policy='stop at or before this first endpoint; no later endpoint is executed',**endpoint_evidence)
    return copy.deepcopy(contract),dict(mode='original_target',source_plan_index=None)


def parse_target_direction_contract(contract, graph=None, *, select_prefix=True, entity_spec=None):
    """User-selected local target/direction scope; original call stays intact.

    Target and where retain referent qualifiers. Plan contributes the first
    explicit motion direction only, not future search/manipulation obligations.
    The internal primitive is a compiler input, never a replacement brain call.
    """
    from rule_lexicon import ROOMS
    from rule_nlp_grounding import lower_relations, NLPConstraintRejected
    if select_prefix:
        local,prefix=select_navigation_prefix(contract,graph)
        if graph is not None and graph.get('original_call')!=local:
            raise TeacherRejected('rule NLP: selected local endpoint needs its own syntax graph')
        spec=parse_target_direction_contract(local,graph,select_prefix=False)
        spec['navigation_prefix']=prefix
        return spec
    internal=copy.deepcopy(contract);extra={};lowering=[]
    if graph is not None and entity_spec is None:
        try:internal,extra,lowering=lower_relations(contract,graph,fields={'target','where'})
        except NLPConstraintRejected as exc:raise TeacherRejected('rule NLP: '+str(exc)) from exc
    target=normalized(internal.get('target',''));match=NOUN.search(target)
    room_target=target.removeprefix('the ') in ROOMS
    if not match and not room_target and entity_spec is None:
        raise TeacherRejected('rule: target has no supported object or room')
    family=entity_spec['family'] if entity_spec is not None else 'doorway' if room_target else NAMES[match.group()]
    first_motion='';source=None
    motion=re.compile(r'\b(turn|move|walk|go|approach|reach|proceed|continue|enter|exit|leave|follow|pass|face|align|head|advance|navigate|cross|stop)\b')
    for index,text in enumerate(contract.get('plan',[])):
        value=normalized(text);hit=motion.search(value)
        if hit:
            if re.search(r"\b(not|never|don't|avoid)\b",value[:hit.start()]):
                raise TeacherRejected('rule: negated initial motion needs explicit direction resolution')
            # A later "then turn right" does not constrain this segment's
            # initial left turn. Noun qualifiers are not movement directions.
            first_motion=re.split(r'[.;]|\bthen\b',value[hit.start():],maxsplit=1)[0]
            source=dict(plan_index=index,text=text);break
    # No explicit motion word leaves the basic direction unconstrained;
    # the current target can still be compiled without inventing a direction.
    def motion_head(value):
        return re.split(r'\b(?:towards?|approach|reach|face|enter|exit|leave|beside|near|past|along|around|between|on|at|by|with|through|into|to(?!\s+(?:the\s+)?(?:left|right)\b))\b',
                        value,maxsplit=1)[0]
    direction_head=motion_head(first_motion)
    from rule_call_nlp import REGIONS
    nominal=re.compile(NOUN.pattern+'|'+r'\b(?:'+'|'.join(map(re.escape,sorted(set(ROOMS)|set(REGIONS),key=len,reverse=True)))+r')\b')
    landmark=nominal.search(direction_head)
    if landmark:
        direction_head=direction_head[:landmark.start()]
    if re.search(r"\b(?:not|never|don't)\b[^.;]*\b(?:left|right|forward|backward|backwards|reverse|clockwise|counterclockwise)\b",direction_head):
        raise TeacherRejected('rule: negated basic direction needs explicit direction resolution')
    # Only the first explicit basic motion direction contributes a gate.
    # Numerical angles, adverbs and later route/direction stages stay in the
    # original call but are not additional acceptance requirements.
    turn=re.search(r'\bturn\b[^.;]*?\b(left|right|clockwise|counterclockwise|anticlockwise)\b',direction_head)
    turn_side=({'clockwise':'right','counterclockwise':'left','anticlockwise':'left'}.get(turn[1],turn[1]) if turn else None)
    side=re.search(r'\b(?:move|walk|go|proceed|continue|head|advance|follow)\b[^.;]*?\b(left|right)\b',direction_head)
    backward=bool(re.search(r'\b(back|backward|backwards|reverse)\b',direction_head))
    forward=bool(re.search(r'\bforward\b',direction_head))
    straight=bool(re.search(r'\bstraight\b',direction_head))
    face_only=bool(re.search(r'\b(turn|face|align)\b',first_motion)) and not re.search(
        r'\b(move|walk|go|approach|reach|proceed|continue|enter|head|advance)\b',' '.join(contract.get('plan',[])))
    stage_motion=contract.get('_navigation_stage_clause',first_motion)
    traverse=family=='doorway' and bool(re.search(r'\b(through|enter)\b',stage_motion))
    primitive=('face the ' if face_only else 'move through the ' if traverse or room_target else 'approach the ')+family
    # This synthetic phrase expresses the local control objective only. It is
    # recorded privately; binding, labels and brain history use original_call.
    internal['plan']=[primitive]
    spec=copy.deepcopy(entity_spec) if entity_spec is not None else parse_contract(internal)
    if entity_spec is not None:spec['kind']='face' if face_only else 'approach'
    if graph is not None and spec.get('relations'):
        raise TeacherRejected('rule NLP: target relation was not completely covered by the nominal grammar')
    spec.update(extra)
    spec.update(turn=turn_side,move_side=side[1] if side else None,
        forward=(forward or straight) and not backward,backward=backward,
        straight=False,gentle_move=False,gentle_turn=False,
        requested_straight=straight,requested_slightly='slightly' in direction_head,
        supervision_scope='target_direction',nlp_lowering=lowering,
        direction_source=source,plan_validation='current target + first basic motion direction only; route wording, later stages, numbers and degree adverbs are not acceptance gates')
    return spec


def validate_motion_prefix(spec, pose, preview, *, admission_mode="strict"):
    """Check each predicted displacement; relaxed mode checks net direction."""
    points = np.asarray(preview, dtype=float)
    yaw = math.radians(pose[2])
    xy = points[:, :2] - pose[:2]
    forward = xy @ np.asarray([math.cos(yaw), math.sin(yaw)])
    left = xy @ np.asarray([-math.sin(yaw), math.cos(yaw)])
    if spec['kind'] != 'face' and np.linalg.norm(xy[:16], axis=1).max() < .05:
        raise TeacherRejected('rule: requested approach has no translational progress')
    if admission_mode == 'relaxed':
        # A safe detour may initially move away; constrain net motion instead.
        forward, left = forward[-1:], left[-1:]
    if not spec['turn'] and spec['forward'] and forward.min() < -.05:
        raise TeacherRejected('rule: forward request would move backward')
    if spec.get('backward') and forward.max() > .05:
        raise TeacherRejected('rule: backward request would move forward')
    if spec['move_side']=='right' and left.max() > .05:
        raise TeacherRejected('rule: rightward request would move left')
    if spec['move_side']=='left' and left.min() < -.05:
        raise TeacherRejected('rule: leftward request would move right')
    if spec.get('gentle_turn'):
        angles=np.abs((points[:,2]-pose[2]+180)%360-180)
        if np.any(angles>45+1e-6):
            raise TeacherRejected('rule: slightly-turn request would require more than 45 degrees')
    if spec['gentle_move'] and admission_mode == 'strict':
        moving=np.linalg.norm(xy, axis=1)>.15
        if np.any(np.abs(np.arctan2(left[moving],forward[moving])) > math.radians(45)):
            raise TeacherRejected('rule: slightly-sideways request would require a sharp detour')


def opening_depth_evidence(depth, camera, pose, obj, projection, *, camera_xyz=None):
    """Require rays through the aperture, not just a carved collision mask."""
    camera_xyz=(*pose[:2],1.2) if camera_xyz is None else camera_xyz
    x0,y0,x1,y1=obj['bbox_xyxy']
    aperture=dict(bbox_xyxy=[x0+.2*(x1-x0),y0+.2*(y1-y0),x1-.2*(x1-x0),y1-.2*(y1-y0)],z_range=[.6,1.8])
    u0,v0,u1,v1=map(int,projection['bbox_px']);tested=clear=0
    for v in range(max(0,v0),min(camera.height,v1),8):
        for u in range(max(0,u0),min(camera.width,u1),8):
            interval=ray_box_interval(camera,camera_xyz,math.radians(pose[2]),u+.5,v+.5,aperture)
            if interval is None:continue
            value=float(depth[v,u]);tested+=1
            clear+=int(math.isfinite(value) and value>interval[1]+.15)
    return dict(tested_rays=tested,clear_rays=clear,clear_fraction=clear/max(1,tested),
                evidence='depth beyond inner aperture plane')


def portal_prefix(points, opening, pose):
    """Select the first ordered phase; later phases have their own intent."""
    center=np.asarray(opening['center']);axis=np.diff(np.asarray(opening['ends']),axis=0)[0]
    normal=np.asarray([-axis[1],axis[0]])/np.linalg.norm(axis)
    if np.dot(center-np.asarray(pose[:2]),normal)<0:normal=-normal
    for index,point in enumerate(points):
        if np.dot(np.asarray(point[:2])-center,normal)>=.5:
            return points[:index+1]
    return points


def directed_portal_program(opening, pose, normal=None):
    """Reuse the waypoint compiler, keeping only still-ahead portal phases."""
    center = np.asarray(opening['center'], float)
    axis = np.diff(np.asarray(opening['ends'], float), axis=0)[0]
    if normal is None:
        normal = np.asarray([-axis[1], axis[0]]) / np.linalg.norm(axis)
        if np.dot(center - np.asarray(pose[:2]), normal) < 0:
            normal = -normal
    else:
        normal = np.asarray(normal, float)
        if normal.shape != (2,) or not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal), 1.):
            raise TeacherRejected('rule: invalid retained portal direction')
        if abs(float(normal @ axis)) > 1e-6:
            raise TeacherRejected('rule: retained portal direction does not match opening')
    signed = float((np.asarray(pose[:2]) - center) @ normal)
    goals = [(center + offset * normal).tolist() for offset in (-.35, 0., .55)
             if offset > signed + .015]
    if not goals:
        raise TeacherRejected('rule: retained portal traversal already complete')
    return dict(status='ok', steps=[dict(kind='via', xy=goal) for goal in goals]), normal.tolist()


class RecordingRoutePolicy(RoutePolicy):
    def predict(self, option):
        trajectory = super().predict(option)
        state = option.slow_condition
        state.setdefault('training_windows', []).append(dict(
            world_pose=_world_pose(self.client.last_info),
            predicted_delta_xyyaw=trajectory.tolist(), proposed_prefix=trajectory[:4].tolist(), executed=False,
            chunk_index=option.chunks))
        return trajectory

    def commit(self, option, executed):
        super().commit(option, executed)
        state = option.slow_condition
        actual = _world_pose(self.client.last_info)
        state['training_windows'][-1].update(executed=True,
            executed_prefix=option.internal_log[-1]['executed_prefix'], actual_end_pose=actual)
        angle = abs(math.degrees(wrap(math.radians(actual[2] - state['expected_end'][2]))))
        state.setdefault('tracking_errors_deg', []).append(angle)
        if angle > 2:
            state['tracking_failure'] = f'yaw tracking mismatch {angle:.3f} deg'


class RuleTeacher(PrivilegedTeacher):
    def __init__(self, *, fast_step_limit=4, route_min_chunks=None, admission_mode=None, grounding_mode=None, text_embedding_model=None, match_policy=None, **kwargs):
        self.match_policy=match_policy or os.environ.get('RULE_TEACHER_MATCH_POLICY','target_and_direction')
        if self.match_policy not in {'target_and_direction','target_or_direction'}:
            raise ValueError('unknown rule teacher match policy')
        self.admission_mode = admission_mode or os.environ.get("RULE_TEACHER_ADMISSION", "strict")
        if self.admission_mode not in {"strict", "relaxed"}:
            raise ValueError("unknown rule teacher admission mode")
        self.grounding_mode = grounding_mode or os.environ.get("RULE_TEACHER_GROUNDING", "lexical")
        self.text_embedding_model=text_embedding_model or os.environ.get('RULE_TEXT_EMBEDDING_MODEL')
        self._text_retriever=None
        from rule_map_grounding import mesh_history_configuration
        self.mesh_semantic_history_path=mesh_history_configuration()['database']
        self._mesh_semantic_history=None
        self._map_label_cache={}
        if self.grounding_mode not in {"lexical", "map_nlp", "map_semantic", "map_open"}:
            raise ValueError("unknown rule teacher grounding mode")
        super().__init__(**kwargs)
        self.route_min_chunks = fast_step_limit if route_min_chunks is None else route_min_chunks
        self.route_exact_chunks = 4 if os.environ.get('RULE_TEACHER_EXACT_FOUR_FAST_CALLS', '0') == '1' else None
        self.route_prefix_steps = int(os.environ.get('RULE_TEACHER_ROUTE_PREFIX_STEPS', '4'))
        if self.route_prefix_steps not in (4, 16):
            raise ValueError('RULE_TEACHER_ROUTE_PREFIX_STEPS must be 4 or 16')
        if self.route_min_chunks < 1:
            raise ValueError('route_min_chunks must be positive')
        self.policy = FullRoutePolicy(self.render_client)
        self.runtime = FullRouteRuntime(self.policy, RemotePrefixExecutor(self.render_client))
        self.last_rejection = None
        self._candidate_plan = None
        self.current_rgb = None
        self.semantic_scenes = {}
        self.portal_progress = None
        self.nlp_url=os.environ.get('RULE_NLP_URL','').rstrip('/')
        self.supervision_scope=os.environ.get('RULE_TEACHER_SCOPE','target_direction')
        if self.supervision_scope not in {'target_direction','full_plan'}:
            raise ValueError('unknown rule teacher supervision scope')
        if self.nlp_url and not self.nlp_url.startswith('http://127.0.0.1:'):
            raise ValueError('rule NLP service must use an explicit loopback endpoint')

    def semantic_scene(self, scene, objects, openings):
        if scene not in self.semantic_scenes:
            from rule_semantics import SemanticScene
            structure=getattr(self, '_sage_active_structure', None)
            if structure is None:
                structure=json.loads((Path(self.dataset.scene_dir(scene))/'structure.json').read_text())
            self.semantic_scenes={scene:SemanticScene(structure,objects,openings,self.scene(scene)[0])}
        return self.semantic_scenes[scene]

    def connecting_portals(self, semantic, occ, openings, obj, spec, pose, depth):
        target_room=semantic.room_at(obj['centre_xy'])
        current_room=semantic.room_at(pose[:2])
        if target_room is None or current_room is None or target_room==current_room:return []
        matches=[]
        for oid,opening in openings.items():
            near,far,_,_=semantic.portal_sides(opening,pose)
            if near!=current_room or far!=target_room:continue
            center=np.asarray(opening['center']);ends=np.asarray(opening['ends'])
            if not occ.seg_free(*pose[:2],*(np.asarray(pose[:2])+.97*(center-pose[:2])),0):continue
            if not occ.is_free(*center,.07):continue
            bbox=dict(bbox_xyxy=[*ends.min(axis=0),*ends.max(axis=0)],z_range=[0,2.3])
            projection=project_bbox(self.camera,self.camera_position(pose),math.radians(pose[2]),bbox)
            if not projection or projection['projected_area_ratio']<.002:continue
            evidence=opening_depth_evidence(depth,self.camera,pose,bbox,projection,camera_xyz=self.camera_position(pose))
            if evidence['clear_rays']<8 or evidence['clear_fraction']<.35:continue
            if spec.get('source_room') and semantic.room_evidence(near,spec['source_room']) is None:continue
            matches.append(dict(id=oid,source_room=near,destination_room=far,projection=projection,depth=evidence))
        return matches

    def discard_candidate_plan(self):
        self._candidate_plan = None

    def _candidate_plan_identity(self, *, scene, episode, pose, rgb, contract, snapshot, **_):
        # A frozen source snapshot is verified by the supervisor. Keep its
        # planner bytes in the handoff proof as well; no code/data mutation is
        # supported within a running collector.
        source_hash = getattr(self, '_candidate_source_sha256', None)
        if source_hash is None:
            source = hashlib.sha256()
            for name in ('rule_teacher.py', 'rule_match_policy.py', 'rule_target_scope.py',
                         'rule_route_execution.py', 'privileged_teacher.py',
                         'mp3d_rule_teacher.py', 'native_stair_route.py',
                         'native_resource_search.py', 'native_compile_worker.py',
                         'native_surface_route.py', 'rule_plan_chunks.py',
                         'mp3d_teacher_semantics.py', 'sage/planner.py',
                         'verl_access/vlm_policy.py',
                         'sage/navmesh_support.py', 'sage/navmesh_map.py', 'sage/same_floor_scope.py'):
                source.update(name.encode())
                source.update((Path(__file__).parent/name).read_bytes())
            source.update(b'verl/action_head/contract.py')
            source.update((Path(__file__).resolve().parents[2]/
                           'verl/action_head/contract.py').read_bytes())
            source_hash = self._candidate_source_sha256 = source.hexdigest()
        image = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        config = {key: getattr(self, key, None) for key in (
            'match_policy', 'admission_mode', 'grounding_mode', 'text_embedding_model',
            'route_min_chunks', 'route_exact_chunks', 'route_prefix_steps', 'supervision_scope', 'nlp_url')}
        payload = dict(scene=scene, episode=episode, pose=np.asarray(pose).tolist(),
            contract=contract, snapshot=snapshot, config=config,
            portal_progress=getattr(self, 'portal_progress', None),
            native_geometry=self.render_client.last_info.get('oracle_habitat_geometry'),
            sage_floor_contract=getattr(self, '_sage_floor_contract', None),
            sage_native_route_admission=getattr(self, '_sage_native_route_admission', None),
            image_shape=list(image.shape), image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            source_sha256=source_hash)
        wire=json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(wire.encode()).hexdigest()

    def prepare_candidate(self, **kwargs):
        self.discard_candidate_plan()
        identity = self._candidate_plan_identity(**kwargs)
        state = self.prepare(**kwargs)
        output = Path(kwargs['output'])
        files = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in output.rglob('*') if p.is_file()}
        self._candidate_plan = dict(identity=identity, state=copy.deepcopy(state),
            output=output, files=files, source_sha256=self._candidate_source_sha256)
        return state

    def prepare_for_execution(self, **kwargs):
        pending = getattr(self, '_candidate_plan', None)
        # Consume before any operation: failure, mismatch and successful reuse
        # all require a new admission before another handoff.
        self.discard_candidate_plan()
        identity = self._candidate_plan_identity(**kwargs)
        if pending is not None and pending['identity'] == identity:
            source = pending['output']
            intact = all((source/name).is_file() and
                hashlib.sha256((source/name).read_bytes()).hexdigest() == digest
                for name, digest in pending['files'].items())
            if intact and 'accepted.json' in pending['files']:
                output = Path(kwargs['output'])
                output.mkdir(parents=True, exist_ok=True)
                for name in pending['files']:
                    target=output/name
                    if target.exists():
                        raise RuntimeError('candidate handoff would overwrite execution evidence')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source/name, target)
                (output/'candidate_plan_handoff.json').write_text(json.dumps(dict(
                    schema='rule_teacher_candidate_plan_handoff_v1',
                    candidate_output=str(source), identity_sha256=identity,
                    planner_source_sha256=pending['source_sha256'],
                    candidate_file_sha256=pending['files'], reused=True), indent=2))
                return copy.deepcopy(pending['state'])
        return self.prepare(**kwargs)

    def reset_environment(self, **kwargs):
        from sage.navmesh_support import SourceCaseGeometryUnsupported
        self.discard_candidate_plan()
        try:
            obs,info=super().reset_environment(**kwargs)
        except SourceGeometryError as error:
            raise TeacherRejected('SAGE source geometry unsupported: ' + str(error)) from error
        except SourceCaseGeometryUnsupported as error:
            raise TeacherRejected('SAGE source geometry unsupported: ' + str(error)) from error
        self.current_rgb=obs['rgb']
        self.portal_progress = None
        return obs,info

    def execute_turn(self, **kwargs):
        self.discard_candidate_plan()
        folder=self.artifact_dir.parent/'turns'/str(time.time_ns())
        folder.mkdir(parents=True,exist_ok=True)
        before=_world_pose(self.render_client.last_info)
        Image.fromarray(np.asarray(self.current_rgb,np.uint8)).save(folder/'actual_chunk_0.png')
        obs,feedback=super().execute_turn(**kwargs)
        self.current_rgb=obs['rgb']
        Image.fromarray(np.asarray(self.current_rgb,np.uint8)).save(folder/'actual_chunk_1.png')
        record=dict(tool='turn',original_call=kwargs,ok=True,accepted_for_training=False,
                    reason='turn_executed_no_AH_label',state=dict(initial_pose=before,actual_poses=[_world_pose(self.render_client.last_info)]))
        (folder/'execution.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
        return obs,feedback

    def ask(self, *args, **kwargs):
        raise RuntimeError('RuleTeacher must never invoke a teacher model')

    def validate_tool_arguments(self, arguments):
        # Later plan stages are preserved in the archive, but do not limit
        # admission or execution of the current target/direction segment.
        from action_head.contract import validate_action_head_arguments
        return validate_action_head_arguments(arguments, max_plan_stages=None)

    def execute_tool(self, **kwargs):
        self.last_rejection=None
        try:
            result = super().execute_tool(**kwargs)
        finally:
            self.discard_candidate_plan()
        if result.observation is not None:
            self.current_rgb=result.observation['rgb']
        latest = max(self.artifact_dir.glob('*/execution.json'),key=lambda p:p.parent.name)
        record=json.loads(latest.read_text())
        accepted = bool(result.ok and not result.raw.get('blocked') and not result.raw.get('collision')
                        and result.raw.get('chunks',0)>0 and (record.get('state') or {}).get('full_route_complete'))
        record['accepted_for_training']=accepted
        feedback = local_approach_feedback(record) if accepted else None
        if feedback is not None:
            from agent_tools import compact_json
            record['local_navigation'] = feedback
            result.raw['local_navigation'] = feedback
            result.model_text = compact_json(result.raw)
        latest.write_text(json.dumps(record,ensure_ascii=False,indent=2))
        # Candidate prepare() calls are read-only. Only a verified real
        # execution can establish or advance the retained doorway binding.
        state = record.get('state') or {}
        retain_native = getattr(self, 'retain_native_transition', None)
        if accepted and state.get('native_surface_contract') and retain_native is not None:
            retain_native(state, latest)
        if accepted and state.get('portal_binding'):
            progress = copy.deepcopy(state['portal_binding'])
            actual = _world_pose(self.render_client.last_info)
            progress.update(actual_pose=actual,
                signed_progress_m=float((np.asarray(actual[:2]) - progress['center']) @ np.asarray(progress['normal'])),
                execution_source=str(latest), call_sha256=record.get('call_sha256'))
            self.portal_progress = progress
        elif accepted:
            self.portal_progress = None
        if not accepted:
            # Keep this signal evaluator-only. The standard evaluator records
            # the actual failed call and still obtains official terminal scores.
            result.episode_abort_reason='rule_abstention'
            result.episode_abort_detail=record.get('error') or self.last_rejection or str(result.raw)
        return result

    def retained_portal(self, scene, episode, pose, local, spec, openings, original_contract=None):
        """Continue a witnessed crossing only near its original aperture.

        Identical referent text and the original room constraints are still
        required; a new target, turn away, or completed crossing cannot use
        historical visibility to bind an unseen doorway.
        """
        progress = self.portal_progress
        if not progress or spec['family'] != 'doorway' or spec['kind'] != 'portal':
            return None
        if progress['scene'] != scene or progress['episode'] != episode:
            return None
        if progress['id'] not in openings:
            return None
        if progress['referent'] != [normalized(local.get('target', '')), normalized(local.get('where', ''))]:
            return None
        original_contract = original_contract or local
        if progress['original_referent'] != [normalized(original_contract.get('target', '')),
                                             normalized(original_contract.get('where', ''))]:
            return None
        opening = openings[progress['id']]
        center = np.asarray(opening['center'], float)
        normal = np.asarray(progress['normal'], float)
        axis = np.diff(np.asarray(opening['ends'], float), axis=0)[0]
        if not np.allclose(center, progress['center']) or abs(float(normal @ axis)) > 1e-6:
            return None
        delta = np.asarray(pose[:2]) - center
        signed = float(delta @ normal)
        lateral = abs(float(delta @ (axis / np.linalg.norm(axis))))
        heading = np.asarray([math.cos(math.radians(pose[2])), math.sin(math.radians(pose[2]))])
        if not -.75 <= signed < .535 or lateral > np.linalg.norm(axis) / 2 + .1:
            return None
        if spec['turn'] or float(heading @ normal) < math.cos(math.radians(50)):
            return None
        return progress

    def prepare(self, *, scene, episode, pose, rgb, contract, snapshot, output):
        output = Path(output); output.mkdir(parents=True, exist_ok=True)
        try:
            self.last_rejection=None
            def parse_language(value):
                request=urllib.request.Request(self.nlp_url+'/parse',data=json.dumps(value).encode(),
                                               headers={'Content-Type':'application/json'})
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=30) as response:
                    return json.load(response)
            local=contract
            original_graph=parse_language(contract) if self.nlp_url else None
            if original_graph is not None:
                (output/'original_nlp_graph_private.json').write_text(json.dumps(original_graph,ensure_ascii=False,indent=2))
            prepare_native = getattr(self, 'prepare_native_navigation', None)
            if prepare_native is not None:
                native_state = prepare_native(scene=scene, episode=episode, pose=pose,
                    rgb=rgb, contract=contract, graph=original_graph, output=output)
                if native_state is not None:
                    return native_state
            validate_call = getattr(self, 'validate_first_navigation_motion', None)
            if validate_call is not None:
                validate_call(contract, original_graph)
            if self.match_policy=='target_or_direction':
                from rule_match_policy import prepare_target_or_direction
                return prepare_target_or_direction(self,scene=scene,episode=episode,pose=pose,
                    rgb=rgb,contract=contract,graph=original_graph,output=output,parse_language=parse_language,
                    snapshot=snapshot)
            relaxations = []
            grounding_changes=[]
            planning_contract=contract
            planning_graph=original_graph
            if self.grounding_mode in {'map_nlp','map_semantic','map_open'}:
                from rule_map_grounding import normalize_call
                planning_contract,grounding_changes=normalize_call(contract,original_graph)
                planning_graph=parse_language(planning_contract) if planning_contract!=contract else original_graph
                if self.grounding_mode=='map_semantic':
                    from rule_text_retrieval import TextRetriever
                    from rule_map_grounding import normalize_semantic_call
                    if self._text_retriever is None:
                        if not self.text_embedding_model:
                            raise RuntimeError('map_semantic requires RULE_TEXT_EMBEDDING_MODEL')
                        self._text_retriever=TextRetriever(self.text_embedding_model)
                    linked,changes=normalize_semantic_call(planning_contract,planning_graph,self._text_retriever)
                    grounding_changes.extend(changes)
                    if linked!=planning_contract:planning_graph=parse_language(linked)
                    planning_contract=linked
                (output/'map_language.json').write_text(json.dumps(dict(original_call=contract,
                    normalized_call=planning_contract,changes=grounding_changes),ensure_ascii=False,indent=2))
            try:
                local,prefix=(select_navigation_prefix(planning_contract,planning_graph,open_vocabulary=self.grounding_mode=='map_open') if self.supervision_scope=='target_direction'
                              else (planning_contract,None))
            except TeacherRejected as exc:
                if self.admission_mode != 'relaxed':
                    raise
                # The explicit tool target/where remain intact. Route-language
                # resolution is a quality warning in this opt-in experiment.
                local = copy.deepcopy(planning_contract)
                prefix = dict(mode='explicit_call_target_fallback', strict_rejection=str(exc))
                relaxations.append(dict(rule='navigation_prefix', reason=str(exc)))
            graph=(parse_language(local) if original_graph is not None and local!=contract else original_graph)
            if graph is not None:
                from rule_nlp_grounding import lower_relations,NLPConstraintRejected
                (output/'nlp_graph_private.json').write_text(json.dumps(graph,ensure_ascii=False,indent=2))
            open_retrieval=None
            if self.grounding_mode=='map_open':
                if graph is None:raise RuntimeError('map_open requires RULE_NLP_URL')
                if self.supervision_scope!='target_direction':raise RuntimeError('map_open currently requires target_direction scope')
                from rule_map_grounding import OpenMapCandidates,open_object_spec,map_semantic_scene
                from rule_text_retrieval import TextRetriever
                occ,objects,openings=self.scene(scene)
                semantic=map_semantic_scene(self.semantic_scene(scene,objects,openings))
                if self._text_retriever is None:
                    if not self.text_embedding_model:raise RuntimeError('map_open requires RULE_TEXT_EMBEDDING_MODEL')
                    self._text_retriever=TextRetriever(self.text_embedding_model)
                if self.mesh_semantic_history_path and (self._mesh_semantic_history is None
                        or self._mesh_semantic_history.path!=str(Path(self.mesh_semantic_history_path).resolve())):
                    from rule_map_grounding import MeshSemanticHistory
                    if self._mesh_semantic_history is not None:self._mesh_semantic_history.db.close()
                    self._mesh_semantic_history=MeshSemanticHistory(self.mesh_semantic_history_path)
                open_retrieval=OpenMapCandidates(semantic,self._text_retriever,
                    history=self._mesh_semantic_history if self.mesh_semantic_history_path else None,scene=scene)
                entity_spec=open_object_spec(local,graph,open_retrieval)
                semantic.candidate_matcher=open_retrieval.match
                (output/'open_map_retrieval.json').write_text(json.dumps(dict(
                    queries=open_retrieval.audit,model=self.text_embedding_model),ensure_ascii=False,indent=2))
            else:entity_spec=None
            if self.supervision_scope=='target_direction':
                spec=parse_target_direction_contract(local,graph,select_prefix=False,entity_spec=entity_spec)
                spec['navigation_prefix']=prefix
            elif graph is not None:
                try:internal,extra,lowering=lower_relations(planning_contract,graph)
                except NLPConstraintRejected as exc:raise TeacherRejected('rule NLP: '+str(exc)) from exc
                spec=parse_contract(internal)
                if spec.get('relations'):
                    raise TeacherRejected('rule NLP: relation was not completely covered by the nominal grammar')
                spec.update(extra)
                spec['nlp_lowering']=lowering
            else:
                spec = parse_contract(planning_contract)
            if open_retrieval is None:
                occ, objects, openings = self.scene(scene)
                semantic=self.semantic_scene(scene,objects,openings)
            if self.grounding_mode in {'map_nlp','map_semantic'}:
                from rule_map_grounding import map_semantic_scene, map_label_matches, normalize
                labels=None
                if self.grounding_mode=='map_semantic':
                    label_key=tuple(dict.fromkeys(normalize(obj['label']) for obj in objects.values()))
                    if label_key not in self._map_label_cache:
                        label_graph=parse_language(dict(target='',where='',plan=['the '+label for label in label_key]))
                        self._map_label_cache[label_key]=map_label_matches(objects,label_graph,self._text_retriever)
                    labels=self._map_label_cache[label_key]
                    (output/'map_label_retrieval.json').write_text(json.dumps(dict(model=self.text_embedding_model,
                        min_score=self._text_retriever.min_score,min_margin=self._text_retriever.min_margin,
                        labels=labels,retrieval_audit=self._text_retriever.audit),ensure_ascii=False,indent=2))
                semantic=map_semantic_scene(semantic,labels)
            rendered, depth = self.renderer.render_one(scene, self.camera_position(pose), math.radians(pose[2]), want_depth=True)
            # Gym observations use JPEG quality 85. Compare the same codec
            # and exact evaluator-only pose, not rounded UI telemetry.
            from vln_gym.env import _jpg_b64
            comparable=Image.open(io.BytesIO(base64.b64decode(_jpg_b64(rendered)))).convert('RGB')
            mae = float(np.mean(np.abs(np.asarray(comparable, float)-np.asarray(rgb, float))))
            if mae > 3:
                raise RuntimeError(f'current-state render mismatch {mae}')
            retained = self.retained_portal(scene, episode, pose, local, spec, openings, contract)
            candidates = []
            candidate_audit=[]
            yaw = math.radians(pose[2])
            if spec['family']=='doorway':
                search_objects={}
                for oid,opening in openings.items():
                    ends=np.asarray(opening['ends'])
                    center=np.asarray(opening['center'])
                    near=np.asarray(pose[:2])+.97*(center-pose[:2])
                    if not occ.seg_free(*pose[:2],*near,0): continue
                    if not occ.is_free(*center,.07): continue
                    from privileged_teacher import opening_object
                    search_objects[oid]=opening_object(oid,opening)
            else:
                search_objects=objects
            for oid, obj in search_objects.items():
                label = normalized(obj['label'])
                # Exact semantic aliases only; basin cabinet is not a basin.
                lexical_match=None
                if open_retrieval is not None:
                    lexical_match=open_retrieval.match(oid,obj,spec['family'])
                elif self.grounding_mode in {'map_nlp','map_semantic'}:
                    from rule_map_grounding import candidate_match
                    lexical_match=candidate_match(oid,obj,spec['family'],semantic)
                elif NAMES.get(label)==spec['family']:
                    lexical_match=dict(method='exact_alias',label=label,family=spec['family'],score=1.)
                audit=dict(id=oid,label=label,match=lexical_match,stage='category')
                candidate_audit.append(audit)
                if lexical_match is None:continue
                audit['stage']='direction' 
                delta = np.asarray(obj['centre_xy']) - pose[:2]
                bearing = math.degrees(wrap(math.atan2(delta[1], delta[0])-yaw))
                continuation = retained if retained and oid == retained['id'] else None
                # At the aperture, 'ahead' can refer to the witnessed remaining
                # crossing. Sided referents still use their current bearing.
                aperture_continuation = bool(continuation and not spec['side'] and
                    abs(float((np.asarray(pose[:2]) - continuation['center']) @ np.asarray(continuation['normal']))) <= .55)
                if aperture_continuation:
                    bearing = math.degrees(wrap(math.atan2(continuation['normal'][1], continuation['normal'][0]) - yaw))
                if spec['side'] == 'left' and bearing < 3: continue
                if spec['side'] == 'right' and bearing > -3: continue
                if spec['ahead'] and abs(bearing) > 50: continue
                if spec['directly_ahead'] and abs(bearing) > 20: continue
                audit['stage']='relation_or_room'
                constraints=semantic.candidate_evidence(oid,obj,spec,pose,NAMES,normalized,portal_progress=continuation)
                if constraints is None:continue
                via=[]
                if spec.get('via_portal'):
                    via=self.connecting_portals(semantic,occ,openings,obj,spec,pose,depth)
                    if len(via)!=1:continue
                audit['stage']='projection'
                projection = project_bbox(self.camera, self.camera_position(pose), yaw, obj)
                historical_projection = False
                if not projection or projection['projected_area_ratio'] < (.0005 if self.admission_mode == 'relaxed' else .002):
                    if not aperture_continuation: continue
                    projection = copy.deepcopy(continuation['projection'])
                    historical_projection = True
                audit['stage']='depth'
                support = depth_support(depth, projection['bbox_px'], self.camera, self.camera_position(pose), yaw, obj, stride=8) if spec['family']!='doorway' else opening_depth_evidence(depth,self.camera,pose,obj,projection,camera_xyz=self.camera_position(pose))
                if spec['family']=='doorway' and (support['clear_rays']<8 or support['clear_fraction']<.35):
                    if not aperture_continuation: continue
                    support = dict(historical_aperture=continuation['depth'],
                        evidence='previous real crossing; current near-aperture pose and collision-free route',
                        historical=True, execution_source=continuation['execution_source'])
                if spec['family']!='doorway' and not via:
                    min_rays, min_fraction = (1, .01) if self.admission_mode == 'relaxed' else (4, .05)
                    if support['visible_rays'] < min_rays or support['visible_fraction'] < min_fraction: continue
                audit['stage']='eligible'
                candidates.append(dict(lexical_match=lexical_match,id=oid, ins_id=obj['ins_id'], label=obj['label'],
                    bearing_deg=bearing, surface_distance_m=_bbox_distance_xy(pose[:2], obj['bbox_xyxy']),
                    projection=projection, depth=support,semantic_constraints=constraints,via_portals=via,
                    retained_portal=bool(continuation), historical_projection=historical_projection,
                    remaining_crossing_bearing=aperture_continuation))
            candidates.sort(key=lambda x: x['surface_distance_m'])
            evidence = dict(schema='rule-teacher-v1', scene=scene, episode=episode, original_call=contract,
                            pose=pose, parsed=spec, candidates=candidates, render_rgb_mae=mae,
                            admission_mode=self.admission_mode, grounding_mode=self.grounding_mode,
                            grounding_changes=grounding_changes,candidate_audit=candidate_audit,relaxations=relaxations)
            if open_retrieval is not None:evidence['open_map_retrieval']=open_retrieval.audit
            (output/'binding.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
            open_state=None
            if self.grounding_mode=='map_open' and spec['family']!='doorway':
                candidates.sort(key=lambda c:(0 if spec['nearest'] else -c['lexical_match']['score'],c['surface_distance_m']))
                failures=[];selected=[]
                for candidate in candidates:
                    try:
                        compiled=self.compile(dict(status='ok',steps=[dict(kind=spec['kind'],
                            object=candidate['id'],side='nearest',standoff_m=.3)]),pose,occ,objects,openings)
                        trial_preview=prepare_full_route(compiled,pose,occ,self.route_min_chunks,
                                                         exact_chunks=getattr(self,'route_exact_chunks',None),
                                                         prefix_steps=getattr(self,'route_prefix_steps',4))
                        validate_motion_prefix(spec,pose,
                            compiled.get('coarse_route_target_world_poses', [pose, *trial_preview])[1:],
                            admission_mode=self.admission_mode)
                        if spec['kind']=='approach':
                            bbox=objects[candidate['id']]['bbox_xyxy']
                            if _bbox_distance_xy(compiled['path'][-1],bbox)>.9:
                                raise TeacherRejected('rule: open-map stance too far from target')
                            if _bbox_distance_xy(trial_preview[-1][:2],bbox)>candidate['surface_distance_m']-.05:
                                raise TeacherRejected('rule: open-map approach makes no progress')
                        if not _path_is_safe(occ,np.asarray([pose[:2]]+[p[:2] for p in trial_preview]),.07):
                            raise TeacherRejected('rule: open-map preview is not collision-free')
                        selected=[candidate];open_state=compiled;break
                    except TeacherRejected as exc:failures.append(dict(id=candidate['id'],reason=str(exc)))
                evidence['stance_rejections']=failures
                evidence['reachable_selection']=[c['id'] for c in selected]
                if len(candidates)>1:
                    relaxations.append(dict(rule='open_map_instance_selection',
                        selection='first reachable direction-consistent instance in semantic then distance order',
                        candidate_ids=[c['id'] for c in candidates]))
                candidates=selected
                (output/'binding.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
            if self.grounding_mode in {'map_nlp','map_semantic'} and candidates and not spec['nearest']:
                best=max(c['lexical_match']['score'] for c in candidates)
                candidates=[c for c in candidates if c['lexical_match']['score']==best]
            if spec['nearest'] and len(candidates)>1 and candidates[1]['surface_distance_m']-candidates[0]['surface_distance_m']>.15:
                candidates = candidates[:1]
            if (self.admission_mode == 'relaxed' and len(candidates)>1
                    and candidates[1]['surface_distance_m']-candidates[0]['surface_distance_m']>.15):
                relaxations.append(dict(rule='unique_referent', selection='nearest matching visible instance',
                                        candidate_ids=[c['id'] for c in candidates]))
                candidates = candidates[:1]
            if len(candidates) != 1:
                raise TeacherRejected(f'rule: {len(candidates)} visible matching instances; need unique referent')
            bound = candidates[0]
            if self.admission_mode == 'relaxed' and spec['family']!='doorway':
                if (bound['projection']['projected_area_ratio'] < .002 or
                        bound['depth']['visible_rays'] < 4 or bound['depth']['visible_fraction'] < .05):
                    relaxations.append(dict(rule='visibility', evidence=bound['depth']))
            if spec['turn'] and ((spec['turn']=='left' and bound['bearing_deg'] < -3) or (spec['turn']=='right' and bound['bearing_deg'] > 3)):
                raise TeacherRejected('rule: turn direction contradicts bound target')
            portal_binding = None
            if spec['family']=='doorway':
                if spec['kind']=='portal':
                    continued = retained if retained and bound['id'] == retained['id'] else None
                    program, normal = directed_portal_program(openings[bound['id']], pose,
                        continued['normal'] if continued else None)
                    portal_binding = dict(id=bound['id'], scene=scene, episode=episode,
                        center=openings[bound['id']]['center'], normal=normal,
                        referent=[normalized(local.get('target', '')), normalized(local.get('where', ''))],
                        original_referent=[normalized(contract.get('target', '')), normalized(contract.get('where', ''))],
                        projection=(continued['projection'] if continued else bound['projection']),
                        depth=(continued['depth'] if continued else bound['depth']),
                        connection=bound['semantic_constraints']['portal_connection'],
                        continued_from=continued['execution_source'] if continued else None)
                elif spec['kind']=='approach':
                    opening=openings[bound['id']];center=np.asarray(opening['center'])
                    axis=np.diff(np.asarray(opening['ends']),axis=0)[0]
                    normal=np.asarray([-axis[1],axis[0]])/np.linalg.norm(axis)
                    if np.dot(center-np.asarray(pose[:2]),normal)<0:normal=-normal
                    program=dict(status='ok',steps=[dict(kind='via',xy=(center-.35*normal).tolist())])
                else:
                    raise TeacherRejected('rule: portal face-only not implemented')
            else:
                program = dict(status='ok', steps=[dict(kind=spec['kind'], object=bound['id'], side='nearest', standoff_m=.3)])
                if spec.get('via_portal'):
                    program['steps'].insert(0,dict(kind='portal',opening=bound['via_portals'][0]['id']))
            search_object=None
            search_view=None
            if spec.get('search_after_portal'):
                far_room=bound['semantic_constraints']['portal_connection']['destination_room']
                groups=[];search_objects={}
                for entity in spec['search_entities']:
                    members=[]
                    for oid,obj in sorted(objects.items()):
                        if NAMES.get(normalized(obj['label']))!=entity['family'] or semantic.room_at(obj['centre_xy'])!=far_room:
                            continue
                        entity_evidence=semantic.candidate_evidence(oid,obj,entity,pose,NAMES,normalized)
                        if entity_evidence is not None:
                            members.append(dict(id=oid,constraints=entity_evidence));search_objects[oid]=obj
                    if not members:
                        raise TeacherRejected('rule: search has no constrained '+entity['family']+' in connected room')
                    groups.append(dict(entity=entity,members=members))
                if os.environ.get('RULE_TEACHER_CONTEXT_VERSION') != 'v8' and len(search_objects)>32:
                    raise TeacherRejected('rule: search candidate set exceeds verified viewpoint budget')
                # A category search keeps all hypotheses. Select a reachable
                # informative view; do not claim a unique object was requested
                # or found, and never apply this relaxation to approach calls.
                views=[]; failures=[]
                for search_id,obj in sorted(search_objects.items()):
                    candidate_program=dict(status='ok',steps=[*program['steps'],
                        dict(kind='approach',object=search_id,side='nearest',standoff_m=.6)])
                    try:
                        candidate_state=self.compile(candidate_program,pose,occ,objects,openings)
                        path=np.asarray(candidate_state['path']);end=path[-1]
                        delta=np.asarray(obj['centre_xy'])-end;heading=math.atan2(delta[1],delta[0])
                        if semantic.room_at(end)!=far_room or _bbox_distance_xy(end,obj['bbox_xyxy'])>.85:
                            raise TeacherRejected('search stance outside connected room or too far from object')
                        future_rgb,future_depth=self.renderer.render_one(scene,self.camera_position(end),heading,want_depth=True)
                        projection=project_bbox(self.camera,self.camera_position(end),heading,obj)
                        support=(depth_support(future_depth,projection['bbox_px'],self.camera,self.camera_position(end),heading,obj,stride=8)
                                 if projection else dict(visible_rays=0,visible_fraction=0.))
                        if support['visible_rays']<4 or support['visible_fraction']<.05:
                            raise TeacherRejected('planned search viewpoint cannot see matching object')
                        jointly_visible=[]
                        for group in groups:
                            visible=[]
                            for member in group['members']:
                                other=search_objects[member['id']]
                                projected=project_bbox(self.camera,self.camera_position(end),heading,other)
                                supported=(depth_support(future_depth,projected['bbox_px'],self.camera,self.camera_position(end),heading,other,stride=8)
                                           if projected else dict(visible_rays=0,visible_fraction=0.))
                                if supported['visible_rays']>=4 and supported['visible_fraction']>=.05:
                                    visible.append(dict(id=member['id'],projection=projected,depth=supported))
                            if not visible:
                                raise TeacherRejected('search viewpoint does not expose every requested entity')
                            jointly_visible.append(dict(family=group['entity']['family'],visible=visible))
                        cost=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
                        views.append(dict(id=search_id,object=obj,state=candidate_state,rgb=future_rgb,
                                          projection=projection,depth=support,cost=cost,jointly_visible=jointly_visible))
                    except TeacherRejected as exc:
                        failures.append(dict(id=search_id,reason=str(exc)))
                if not views:
                    (output/'search_rejected.json').write_text(json.dumps(failures,indent=2))
                    raise TeacherRejected('rule: search has no reachable verified object viewpoint')
                search_view=min(views,key=lambda v:(v['cost'],v['id']))
                search_object=search_view['object'];state=search_view['state']
                state['search_target']=dict(id=search_view['id'],room_id=far_room,
                    candidate_ids=sorted(search_objects),candidate_sets=groups,unique_referent=False,
                    jointly_visible=search_view['jointly_visible'],
                    objective='one verified view of every constrained search entity; bounded prefix may end earlier',
                    selection='shortest reachable path among verified views',rejected_views=failures,
                    projection=search_view['projection'],depth=search_view['depth'],
                    evidence='privileged prospective visibility; not an observed discovery')
                Image.fromarray(np.asarray(search_view['rgb'],np.uint8)).save(output/'search_goal_private.png')
            else:
                state = open_state if open_state is not None else self.compile(program, pose, occ, objects, openings)
            if portal_binding:
                state['portal_binding'] = portal_binding
                state['face_xy'] = (np.asarray(portal_binding['center']) + 2 * np.asarray(portal_binding['normal'])).tolist()
            if spec['family']=='doorway' and spec['kind']=='approach':
                state['face_xy']=openings[bound['id']]['center']
            route = np.asarray(state['path'])
            first_opening=(bound['id'] if search_object is not None else
                           bound['via_portals'][0]['id'] if spec.get('via_portal') else None)
            intent_route=portal_prefix(route,openings[first_opening],pose) if first_opening else route
            distance = float(np.linalg.norm(np.diff(intent_route, axis=0), axis=1).sum())
            direct = float(np.linalg.norm(intent_route[-1]-intent_route[0]))
            if spec['straight'] and distance > direct*1.1+.05:
                raise TeacherRejected('rule: straight instruction would require a detour')
            if spec['kind']=='approach' and spec['family']!='doorway' and _bbox_distance_xy(route[-1],objects[bound['id']]['bbox_xyxy']) > (.9 if self.admission_mode == 'relaxed' else .65):
                raise TeacherRejected('rule: final stance too far from target surface')
            preview = prepare_full_route(state, pose, occ, self.route_min_chunks,
                                         exact_chunks=getattr(self,'route_exact_chunks',None),
                                         prefix_steps=getattr(self,'route_prefix_steps',4))
            semantic_preview=state.get('coarse_route_target_world_poses', [pose, *preview])[1:]
            intent_preview=portal_prefix(semantic_preview,openings[first_opening],pose) if first_opening else semantic_preview
            try:
                validate_motion_prefix(spec, pose, intent_preview)
            except TeacherRejected as exc:
                if self.admission_mode != 'relaxed': raise
                validate_motion_prefix(spec, pose, intent_preview, admission_mode='relaxed')
                relaxations.append(dict(rule='route_direction', reason=str(exc)))
            if spec['kind']=='approach' and spec['family']!='doorway' and not spec.get('via_portal'):
                bbox=objects[bound['id']]['bbox_xyxy']
                distances=[_bbox_distance_xy(point[:2],bbox) for point in preview]
                initial_distance=bound['surface_distance_m']
                if distances[-1] > initial_distance-.05 or (self.admission_mode == 'strict' and max(distances)>initial_distance+.05):
                    raise TeacherRejected('rule: approach prefix does not get closer to the target surface')
            if spec['straight'] and not spec['turn']:
                for point in intent_preview:
                    delta=np.asarray(point[:2])-pose[:2]
                    if np.linalg.norm(delta)>.15 and abs(math.degrees(wrap(math.atan2(delta[1],delta[0])-yaw)))>20:
                        raise TeacherRejected('rule: straight prefix deviates more than 20 degrees')
            if not _path_is_safe(occ, np.asarray([pose[:2]]+[p[:2] for p in preview]), .07):
                raise TeacherRejected('rule: controller prediction is not collision-free')
            if self.admission_mode == 'relaxed' and spec['kind']=='approach' and spec['family']!='doorway':
                if _bbox_distance_xy(route[-1],objects[bound['id']]['bbox_xyxy']) > .65:
                    relaxations.append(dict(rule='final_standoff', limit_m=.9))
                if not spec.get('via_portal') and max(distances)>initial_distance+.05:
                    relaxations.append(dict(rule='nonmonotonic_approach'))
            (output/'binding.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
            state.update(admission_mode=self.admission_mode, relaxation_warnings=relaxations,
                         binding=evidence, bound=bound, initial_pose=pose, preview_world_xy_yaw_deg=preview,
                         semantic_status='rule_supported_pending_human_review')
            if getattr(self, '_sage_floor_contract', None) is not None:
                state['sage_floor_contract'] = copy.deepcopy(self._sage_floor_contract)
            if getattr(self, '_sage_native_route_admission', None) is not None:
                state['sage_native_route_admission'] = copy.deepcopy(self._sage_native_route_admission)
            self.map_image(occ,pose,objects,path=[pose[:2]]+[p[:2] for p in preview]).save(output/'teacher_map.png')
            marked=Image.fromarray(np.asarray(rgb,np.uint8)); from PIL import ImageDraw
            draw=ImageDraw.Draw(marked)
            if not bound['historical_projection']:
                draw.rectangle(bound['projection']['bbox_px'],outline='yellow',width=3)
            draw.text((8,8),bound['id']+(' (retained crossing; prior view)' if bound['historical_projection'] else ''),fill='yellow')
            marked.save(output/'bound_object.png')
            (output/'accepted.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
            return state
        except TeacherRejected as exc:
            self.last_rejection = str(exc)
            (output/'rejected.json').write_text(json.dumps(dict(reason=str(exc),original_call=contract),ensure_ascii=False,indent=2))
            raise
