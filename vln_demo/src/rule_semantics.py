"""Explicit text constraints and privileged scene evidence for rule teachers.

This module never generates or edits a brain reply. Normalized phrases below
are an internal compiler representation; every removed phrase becomes a
constraint checked against the scene before an action can be executed.
"""
from __future__ import annotations

import copy
import math
import re

import numpy as np

from privileged_teacher import TeacherRejected
from sage.goal_region import _polygon_covers_point
from sage.route_extractor import FUNCTIONAL_AREA_ANCHORS, _room_label_map, _functional_area

from rule_lexicon import ROOMS, ROOM_PATTERN


def parse_search_entities(phrase, simple_parser, noun, names, normalize):
    """Reuse entity constraints, preserving search conjunctions and modifiers.

    Bare conjunctions are joint search goals. A relation-bearing expression
    remains one entity; ambiguous coordination is rejected by the same parser.
    """
    relational=bool(re.search(r'\b(near|next to|beside|with)\b',phrase))
    parts=[phrase] if relational else re.split(r'\s+and\s+',phrase)
    if not 1<=len(parts)<=3:
        raise TeacherRejected('rule: search supports up to three jointly visible entities')
    entities=[]
    for part in parts:
        match=noun.search(part)
        if not match:raise TeacherRejected('rule: search target has no supported entity')
        internal=dict(target=part,where='',plan=['approach the '+match.group()])
        try:spec=simple_parser(internal)
        except TeacherRejected:
            spec=parse_extended(internal,simple_parser,noun,names,normalize)
        if spec.get('search_after_portal') or spec.get('via_portal') or spec['kind']!='approach':
            raise TeacherRejected('rule: nested search/action in entity description')
        if spec['side'] or spec['ahead'] or spec['nearest']:
            raise TeacherRejected('rule: search entity viewpoint-relative descriptor unresolved')
        if spec['family']=='doorway':
            raise TeacherRejected('rule: search for openings needs a room-search policy')
        if any(x['family']==spec['family'] for x in entities):
            raise TeacherRejected('rule: repeated search family needs explicit cardinality')
        entities.append(spec)
    return entities


def parse_extended(contract, simple_parser, noun, names, normalize):
    """Compile relations/rooms/ordered portal approach into explicit guards.

    Unsupported attributes, manipulation, guest-room role, and
    object-relative orientation stay unsupported rather than being discarded.
    """
    original = copy.deepcopy(contract)
    target = normalize(contract.get('target', ''))
    where = normalize(contract.get('where', ''))
    plans = contract.get('plan')
    if not isinstance(plans, list) or not plans or not all(isinstance(x,str) for x in plans):
        raise TeacherRejected('rule: plan must be nonempty text list')
    plans = [normalize(x) for x in plans]
    guards = dict(relations=[], source_room=None, destination_room=None,
                  target_room=None, elongated=False, via_portal=False, search_after_portal=None,
                  exit_current_room=False, destination_connected=False, search_entities=[])

    def room(role, value):
        value = ROOMS[value]
        if guards[role] and guards[role] != value:
            raise TeacherRejected('rule: conflicting '+role+' constraints')
        guards[role] = value
        return ''

    # A search phase is an explicit navigation objective: enter the connected
    # room and reach a view of a matching object, retaining the candidate set.
    # This does not assert that an ambiguous approach referent is resolved.
    # The search is not dropped
    # just because the current option may run out of fast-step budget first.
    motion_plans=[]
    for text in plans:
        match=re.search(r'\b(?:look for|find|locate) (.+)$',text)
        if match:
            phrase=match[1]
            phrase=re.sub(r'\b(?:in|inside) (?:the )?('+ROOM_PATTERN+r')\b',
                          lambda m:room('destination_room',m[1]),phrase).strip()
            # "inside the room" refers to the just-entered room. Search is
            # subsequently required to follow that same verified portal.
            phrase=re.sub(r'\b(?:in|inside) (?:the )?room\b','',phrase).strip()
            entities=parse_search_entities(phrase,simple_parser,noun,names,normalize)
            if guards['search_entities'] and guards['search_entities']!=entities:
                raise TeacherRejected('rule: multiple ordered search phases unsupported')
            guards['search_entities']=entities
            guards['search_after_portal']=entities[0]['family']
            text=re.sub(r'\b(?:and|to)\s*$','',text[:match.start()]).strip()
        if text:motion_plans.append(text)
    if guards['search_after_portal']:
        if not any(re.search(r'\b(through|enter)\b',p) for p in motion_plans):
            raise TeacherRejected('rule: search phase requires an explicit portal entry')
        plans=motion_plans
    if target.removeprefix('the ') in ROOMS:
        if not any(('through' in p and ('doorway' in p or 'opening' in p))
                   or re.search(r'\benter (?:the )?'+ROOM_PATTERN+r'\b',p) for p in plans):
            raise TeacherRejected('rule: room target requires an explicit doorway route')
        room('destination_room',target.removeprefix('the '))
        target='the doorway'

    def clean(text, is_plan=False):
        text = text.replace('first person view', 'current view')
        text = re.sub(r'\b(?:the )?robot\b', 'me', text)
        def connected_room(match):
            guards['destination_connected']=True
            return 'move through the doorway' if match[0].startswith('enter') else ''
        text=re.sub(r'\b(?:enter|into|leading into|leading to|to) (?:the |a )?(?:next )?room\b',connected_room,text)
        def current_exit(match):
            guards['exit_current_room']=True
            return ''
        text = re.sub(r'\bleading out of (?:the )?(?:current )?(?:room|space)\b',current_exit,text)
        text = re.sub(r'\b(?:in the )?cent(?:er|re) of (?:the )?(?:current )?view\b', 'directly ahead', text)
        # Plural references preserve their family; the relation quantifier is
        # existential (near one or more mirrors), never an ordinal selector.
        text = re.sub(r'\b(?:leading )?out of (?:the )?('+ROOM_PATTERN+r')\b',
                      lambda m: room('source_room', m[1]), text)
        text = re.sub(r'\bleading (?:to|into) (?:the )?('+ROOM_PATTERN+r')\b',
                      lambda m: room('destination_room', m[1]), text)
        text = re.sub(r'\b(doorway|opening) (?:to|into) (?:the )?('+ROOM_PATTERN+r')\b',
                      lambda m: m[1]+room('destination_room',m[2]), text)
        text = re.sub(r'\b('+ROOM_PATTERN+r') doorway\b',
                      lambda m: room('destination_room',m[1])+' doorway', text)
        text = re.sub(r'\bin (?:the )?('+ROOM_PATTERN+r')\b',
                      lambda m: room('target_room',m[1]), text)
        if is_plan:
            text=re.sub(r'\b(?:exit|leave) (?:the )?(?:current )?room\b','move through the doorway',text)
            text = re.sub(r'\bexit (?:the )?('+ROOM_PATTERN+r')\b',
                          lambda m: room('source_room',m[1])+' move through the doorway', text)
            text = re.sub(r'\benter (?:the )?('+ROOM_PATTERN+r')\b',
                          lambda m: room('destination_room',m[1])+' move through the doorway', text)
            text = re.sub(r'\binto (?:the )?('+ROOM_PATTERN+r')\b',
                          lambda m: room('destination_room',m[1]), text)
            text = re.sub(r'\bpass through\b', 'move through', text)
            text = re.sub(r'\btowards? and through\b', 'through', text)
        # Relations are checked in geometry, not treated as neutral words.
        pattern = r'\b(near|next to|beside|with) (?:the |a |an )?('+noun.pattern[2:-2]+r')\b'
        def relation(match):
            family = names[match[2]]
            rec = dict(kind='supports' if match[1]=='with' else 'near', family=family)
            if match[2] in {'mirrors','stools','books','pillows','blankets'}:rec['min_count']=2
            if rec not in guards['relations']: guards['relations'].append(rec)
            return ''
        text = re.sub(pattern, relation, text)
        if re.search(r'\blong\b', text):
            guards['elongated'] = True
            text = re.sub(r'\blong\b', '', text)
        return normalize(text)

    internal = dict(target=clean(target), where=clean(where), plan=[clean(p, True) for p in plans])
    # "move toward the doorway to exit ..." refers to the same crossing.
    internal['plan'] = [re.sub(r'\bto move through', 'and move through', p) for p in internal['plan']]
    main = list(noun.finditer(internal['target']))
    if len(main) != 1:
        raise TeacherRejected('rule: extended target is not one explicit object')
    family = names[main[0].group()]
    explicit_portal=any('through' in p for p in internal['plan'])
    described_portal=bool(re.search(r'\bthrough (?:the )?(?:doorway|opening)\b',internal['where']))
    if family != 'doorway' and (explicit_portal or described_portal):
        # Only a verified doorway-then-approach sequence. No implicit turn or
        # search phase may disappear into the path planner.
        portal = re.compile(r'^(?:move|go|walk|proceed)(?: straight| forward)* through (?:the )?(?:doorway|opening)$')
        stages=[]
        for p in internal['plan']:
            stages.extend(re.split(r'\s+and\s+(?=approach\b)', p))
        if not explicit_portal:
            stages=['move through the doorway',*stages]
        if len(stages)<2 or not portal.fullmatch(stages[0]):
            raise TeacherRejected('rule: unsupported ordered doorway sequence')
        internal['plan'] = stages[1:]
        internal['where'] = re.sub(r'\bthrough (?:the )?(?:doorway|opening)\b','',internal['where']).strip()
        guards['via_portal'] = True
        spec = simple_parser(internal)
        if spec['kind'] != 'approach' or spec['turn']:
            raise TeacherRejected('rule: unsupported action after doorway')
        first = simple_parser(dict(target='the doorway',where='',plan=[stages[0]]))
        # The first stage's forward/straight constraint covers the entire
        # predicted window, conservatively rejecting a later sharp bend.
        for key in ('straight','forward','move_side','gentle_move'):
            spec[key]=first[key]
    else:
        spec = simple_parser(internal)
    if guards['relations'] and any(r['family']==family for r in guards['relations']):
        raise TeacherRejected('rule: same-family relational reference is ambiguous')
    if spec['kind']=='portal' and family!='doorway':
        raise TeacherRejected('rule: portal traversal needs an opening')
    if guards['search_after_portal'] and (family!='doorway' or spec['kind']!='portal'):
        raise TeacherRejected('rule: search follows a verified opening traversal only')
    if guards['exit_current_room'] and family!='doorway':
        raise TeacherRejected('rule: leaving current room needs an explicit opening target')
    if guards['destination_connected'] and family!='doorway':
        raise TeacherRejected('rule: generic next room needs an explicit opening target')
    spec.update(guards)
    assert contract==original, 'semantic compiler mutated original call'
    return spec


class SemanticScene:
    """Native room polygons plus high-specificity labelled functional anchors."""
    def __init__(self, structure, objects, openings, occ=None):
        self.rooms=[r['profile'] for r in structure.get('rooms',[]) if len(r.get('profile',[]))>=3]
        self.objects=objects
        self.openings=openings
        self.occ=occ
        self.room_labels=_room_label_map(occ) if occ is not None else None
        self.anchors={i:[] for i in range(len(self.rooms))}
        for oid,obj in objects.items():
            family=FUNCTIONAL_AREA_ANCHORS.get(obj['label'].replace(' ','_'))
            rid=self.room_at(obj['centre_xy'])
            if family and rid is not None:
                self.anchors[rid].append(dict(id=oid,label=obj['label'],area=family))

    def room_at(self, xy):
        hits=[i for i,vertices in enumerate(self.rooms) if _polygon_covers_point(vertices,*map(float,xy))]
        return hits[0] if len(hits)==1 else None

    def room_evidence(self, rid, requested, xy=None):
        if rid is None:return None
        anchors=self.anchors[rid]
        areas={a['area'] for a in anchors}
        public={'living area','dining area','kitchen area'}
        if requested=='common area':
            valid=bool(areas & public) and areas.issubset(public|{'study area'})
        elif requested=='sleeping area':
            # A desk does not stop a bedroom being a bedroom. Require an
            # actual bed, not just a generic cabinet/nightstand as evidence.
            valid=any(a['label'].replace(' ','_') in {'bed','double_bed','single_bed','bunk_bed'} for a in anchors)
        else:
            valid=bool(areas) and areas.issubset({requested})
        local=None
        if not valid and xy is not None and self.occ is not None:
            local=_functional_area(xy,self.occ,self.objects,self.room_labels)
            valid=local['status']=='verified' and local['area']==requested
        if not valid:return None
        return dict(room_id=rid,requested=requested,anchors=anchors,
                    local=local,evidence='native room polygon and specific semantic anchors; inferred function')

    def portal_sides(self, opening, pose, direction=None):
        center=np.asarray(opening['center']);axis=np.diff(np.asarray(opening['ends']),axis=0)[0]
        normal=np.array([-axis[1],axis[0]])/np.linalg.norm(axis)
        if direction is not None:normal=np.asarray(direction,float)
        elif np.dot(center-np.asarray(pose[:2]),normal)<0:normal=-normal
        near=center-.55*normal;far=center+.55*normal
        return self.room_at(near),self.room_at(far),near,far

    def candidate_evidence(self, oid, obj, spec, pose, names, normalize, portal_progress=None):
        if spec.get('entity_relation_alternatives'):
            feasible=[]
            for index,relations in enumerate(spec['entity_relation_alternatives']):
                branch={k:v for k,v in spec.items() if k!='entity_relation_alternatives'}
                branch['relations']=[*spec.get('relations',[]),*relations]
                result=self.candidate_evidence(oid,obj,branch,pose,names,normalize,portal_progress=portal_progress)
                if result is not None:feasible.append(dict(alternative=index,evidence=result))
            return dict(feasible_language_alternatives=feasible) if feasible else None
        evidence={}
        if spec.get('elongated'):
            x0,y0,x1,y1=obj['bbox_xyxy'];width,height=x1-x0,y1-y0
            if max(width,height)<1.5 or max(width,height)/max(.01,min(width,height))<2:return None
            evidence['elongated']=dict(extent=[width,height],min_length_m=1.5,min_aspect=2.)
        if spec.get('relations'):
            x0,y0,x1,y1=obj['bbox_xyxy'];relations=[]
            for relation in spec['relations']:
                matches=[]
                matcher=getattr(self,'candidate_matcher',None)
                candidate_ids=getattr(getattr(matcher,'__self__',None),'candidate_ids',None)
                references=((ref_id,self.objects[ref_id]) for ref_id in candidate_ids(relation['family'])) if candidate_ids else self.objects.items()
                for ref_id,ref in references:
                    if matcher is not None:
                        if not matcher(ref_id,ref,relation['family']):continue
                    elif names.get(normalize(ref['label']))!=relation['family']:continue
                    if ref_id==oid:continue
                    a,b,c,d=ref['bbox_xyxy']
                    distance=math.hypot(max(x0-c,a-x1,0.),max(y0-d,b-y1,0.))
                    # Same topological room avoids matching through a wall.
                    room=self.room_at(obj['centre_xy'])
                    kind=relation['kind']
                    geometrically_valid=False
                    if kind=='near':geometrically_valid=distance<=.75
                    elif kind in {'supports','supported_by'}:
                        overlap_x=min(x1,c)-max(x0,a);overlap_y=min(y1,d)-max(y0,b)
                        z_gap=(ref['z_range'][0]-obj['z_range'][1] if kind=='supports'
                               else obj['z_range'][0]-ref['z_range'][1])
                        geometrically_valid=(overlap_x>0 and overlap_y>0 and abs(z_gap)<=.25)
                    elif kind in {'left_of','right_of'}:
                        yaw=math.radians(pose[2]);left=np.array([-math.sin(yaw),math.cos(yaw)])
                        lateral=float((np.asarray(obj['centre_xy'])-ref['centre_xy'])@left)
                        geometrically_valid=lateral>.1 if kind=='left_of' else lateral<-.1
                    elif kind in {'behind','in_front_of'}:
                        yaw=math.radians(pose[2]);forward=np.array([math.cos(yaw),math.sin(yaw)])
                        longitudinal=float((np.asarray(obj['centre_xy'])-ref['centre_xy'])@forward)
                        geometrically_valid=longitudinal>.1 if kind=='behind' else longitudinal<-.1
                    if geometrically_valid and room is not None and self.room_at(ref['centre_xy'])==room:
                        reference_evidence=(self.candidate_evidence(ref_id,ref,relation['entity'],pose,names,normalize)
                                            if relation.get('entity') else {})
                        if reference_evidence is None:continue
                        matches.append(dict(id=ref_id,bbox_gap_m=distance,reference_constraints=reference_evidence,
                                            frame='current_camera' if kind in {'left_of','right_of','behind','in_front_of'} else 'world_geometry'))
                if len(matches)<relation.get('min_count',1):return None
                relations.append(dict(constraint=relation,references=matches))
            evidence['relations']=relations
        if oid in self.openings:
            retained = portal_progress if portal_progress and portal_progress['id'] == oid else None
            near,far,near_xy,far_xy=self.portal_sides(self.openings[oid],pose,retained['normal'] if retained else None)
            current=self.room_at(pose[:2])
            if near is None or far is None or near==far or current not in ({near,far} if retained else {near}):return None
            if retained and (near != retained['connection']['source_room'] or far != retained['connection']['destination_room']):return None
            evidence['portal_connection']=dict(source_room=near,destination_room=far,
                                                near_xy=near_xy.tolist(),far_xy=far_xy.tolist())
            for key,rid,xy in [('source_room',near,near_xy),('destination_room',far,far_xy),('target_room',far,far_xy)]:
                if spec.get(key):
                    ev=self.room_evidence(rid,spec[key],xy)
                    if ev is None:return None
                    evidence[key]=ev
        else:
            if spec.get('source_room') and not spec.get('via_portal'):return None
            for key in ['target_room','destination_room']:
                if spec.get(key):
                    ev=self.room_evidence(self.room_at(obj['centre_xy']),spec[key],obj['centre_xy'])
                    if ev is None:return None
                    evidence[key]=ev
        return evidence
