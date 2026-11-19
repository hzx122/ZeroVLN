"""Map candidate retrieval for the existing NLP/geometry grounding pipeline.

Entity-linking separates mention normalization, candidate recall and spatial
verification. These are candidate aliases, never a claim of object identity.
Original calls and NLP source spans remain in the audit record.
"""
import copy
import re
import numpy as np
from rule_lexicon import NAMES
from sage.goal_region import semantic_family


_KNOWN_COARSE_FAMILIES = frozenset(semantic_family(value) for value in NAMES.values())


def category_conflict(query, label):
    """Reject contradictory known classes, keeping open vocabulary unresolved.

    Reuse the existing proxy-family contract: cabinet/wardrobe and table/desk
    remain compatible. Similarity alone cannot turn a door into a window.
    This is a veto on text retrieval, not proof of an object's visual identity.
    """
    requested = semantic_family(NAMES.get(normalize(query), normalize(query)))
    actual = semantic_family(NAMES.get(normalize(label), normalize(label)))
    if requested in _KNOWN_COARSE_FAMILIES and actual in _KNOWN_COARSE_FAMILIES and requested != actual:
        return dict(requested_family=requested, candidate_family=actual)
    return None


def normalize(text):
    return re.sub(r'\s+', ' ', str(text).lower().replace('_', ' ').replace('-', ' ')).strip()


# Lexical equivalence only. Functional subtypes are verified per instance below.
MENTIONS = {
    'sleeping area': 'bedroom', 'kitchen area': 'kitchen',
    'lounge area': 'living room', 'lounge': 'living room',
    'bedroom area': 'bedroom', 'dining chairs': 'chairs', 'dining chair': 'chair',
    'open door': 'open doorway', 'door': 'doorway',
}


def normalize_call(contract, graph):
    """Apply noun lemmas/aliases using original offsets, preserving qualifiers."""
    result=copy.deepcopy(contract); changes=[]
    if graph is None:
        raise ValueError('map_nlp grounding requires RULE_NLP_URL')
    if graph.get('original_call') != contract:
        raise ValueError('NLP graph belongs to another call')
    for field in graph['fields']:
        text=field['text']; edits=[]
        for token in field['tokens']:
            lemma=token['lemma'].lower()
            if token['pos'] in {'NOUN','PROPN'} and lemma in NAMES and lemma != token['text'].lower():
                edits.append((*token['span'],lemma))
        for start,end,value in reversed(edits):text=text[:start]+value+text[end:]
        pattern=r'\b(?:'+'|'.join(map(re.escape,sorted(MENTIONS,key=len,reverse=True)))+r')\b'
        text=re.sub(pattern,lambda m:MENTIONS[m[0].lower()],text,flags=re.I)
        # Keep source vs destination roles explicit, including room qualifiers.
        text=re.sub(r'\bexit (?:of|from)\b','doorway leading out of',text,flags=re.I)
        text=re.sub(r'\b(?:entrance|entry) (?:to|of|into)\b','doorway leading to',text,flags=re.I)
        text=re.sub(r'\b(bedroom|bathroom|kitchen|living room) (?:entrance|entry)\b',r'doorway leading to the \1',text,flags=re.I)
        key=field['field']
        if text != field['text']:changes.append(dict(field=key,original=field['text'],normalized=text))
        if key.startswith('plan['):result['plan'][int(key[5:-1])]=text
        else:result[key]=text
    return result,changes


def bbox_gap(a,b):
    x0,y0,x1,y1=a['bbox_xyxy'];u0,v0,u1,v1=b['bbox_xyxy']
    return float(np.hypot(max(x0-u1,u0-x1,0),max(y0-v1,v0-y1,0)))


def normalize_semantic_call(contract, graph, resolver):
    """Link noun heads/compounds; keep attributes, counts and spatial relations."""
    if graph.get('original_call') != contract:
        raise ValueError('NLP graph belongs to another call')
    result=copy.deepcopy(contract); proposed=[]; changes=[]
    for field in graph['fields']:
        tokens={t['i']:t for t in field['tokens']}
        for entity in field.get('entities',[]):
            if entity.get('unresolved_reference'):continue
            if (entity.get('lexical') or {}).get('kind') in {'room','region'}:continue
            root=tokens[entity['root']]
            if root['pos'] not in {'NOUN','PROPN'}:continue
            parts=[root]+[t for t in tokens.values() if t['head']==root['i']
                         and t['dependency']=='compound' and t['pos'] in {'NOUN','PROPN'}]
            parts.sort(key=lambda t:t['span'][0])
            # A known object used as a modifier can encode a relation/type
            # constraint (basin cabinet, sofa bed), not a removable synonym.
            if any(normalize(t['lemma']) in NAMES for t in parts if t is not root):continue
            start,end=parts[0]['span'][0],parts[-1]['span'][1]
            phrase=normalize(field['text'][start:end])
            if phrase in NAMES:continue
            if normalize(' '.join(t['text'] for t in parts))!=phrase:continue
            proposed.append((field['field'],start,end,phrase))
    matches=resolver.resolve_many([p[3] for p in proposed])
    for field in graph['fields']:
        text=field['text']
        for key,start,end,phrase in sorted(proposed,key=lambda p:p[1],reverse=True):
            match=matches[phrase]
            if key!=field['field'] or match is None or match['family']=='doorway':continue
            text=text[:start]+match['family']+text[end:]
            changes.append(dict(field=key,span=[start,end],original=phrase,
                                normalized=match['family'],evidence=match))
        key=field['field']
        if key.startswith('plan['):result['plan'][int(key[5:-1])]=text
        else:result[key]=text
    return result,changes


def map_label_matches(objects, graph, resolver):
    """Infer label heads from syntax, with learned retrieval for unknown names."""
    labels=list(dict.fromkeys(normalize(obj['label']) for obj in objects.values()))
    fields={f['field']:f for f in graph['fields']}
    matches={};unknown=[]
    for i,label in enumerate(labels):
        field=fields[f'plan[{i}]']
        nodes=field.get('entities',[])
        lexical=nodes[0].get('lexical') if len(nodes)==1 else None
        if label in NAMES:
            matches[label]=dict(method='exact_alias',family=NAMES[label],score=1.,label=label)
        elif lexical and lexical['kind']=='object':
            matches[label]=dict(method='map_noun_head',family=lexical['canonical'],score=.90,
                                label=label,entity=nodes[0])
        else:unknown.append(label)
    matches.update(resolver.resolve_many(unknown))
    return matches


def candidate_match(oid,obj,family,semantic):
    """Recall actual map instances, verifying functional aliases with anchors."""
    label=normalize(obj['label']); actual=NAMES.get(label)
    if actual==family:return dict(method='exact_alias',label=label,family=family,score=1.)
    aliases={'multi person sofa':'sofa','single sofa':'armchair','dining chair':'chair',
             'office chair':'chair','tv cabinet':'cabinet','wall cabinet':'cabinet',
             'storage cabinet':'cabinet','side table':'side table','table combination':'table',
             'green plants':'plant','landscape plants':'plant'}
    actual=aliases.get(label,actual)
    if actual==family:return dict(method='map_label_alias',label=label,family=family,score=.95)
    learned=getattr(semantic,'label_matches',{}).get(label)
    if learned and actual is None and learned['family']==family:
        return {**learned, 'score':min(.79,learned['score']), 'retrieval_score':learned['score']}
    if learned and actual is None:actual=learned['family']
    if family=='table' and actual in {'table','side table','dining table','coffee table','bedside table','conference table'}:
        return dict(method='subtype',label=label,family=family,score=.85)
    if family=='chair' and actual=='armchair':
        return dict(method='subtype',label=label,family=family,score=.85)
    anchor_family=None
    if family=='bedside table' and actual in {'side table','table'}:anchor_family='bed'
    if family=='dining table' and actual in {'table','conference table'}:anchor_family='chair'
    if anchor_family:
        room=semantic.room_at(obj['centre_xy']);anchors=[]
        for ref_id,ref in semantic.objects.items():
            ref_label=normalize(ref['label'])
            ref_family=NAMES.get(ref_label)
            if ref_family is None:ref_family=(getattr(semantic,'label_matches',{}).get(ref_label) or {}).get('family')
            if ref_family!=anchor_family:continue
            if room is None or semantic.room_at(ref['centre_xy'])!=room:continue
            gap=bbox_gap(obj,ref)
            if gap<=1.5:anchors.append(dict(id=ref_id,gap_m=gap))
        if anchors:return dict(method='functional_anchor',label=label,family=family,score=.8,anchors=anchors)
    return None


from rule_semantics import SemanticScene
from sage.route_extractor import FUNCTIONAL_AREA_ANCHORS


class MapSemanticScene(SemanticScene):
    """A per-mode view of cached geometry, with map-aware entity semantics."""
    def __init__(self, base, label_matches=None):
        self.__dict__.update(base.__dict__)
        self.label_matches=label_matches or {}
        self.objects={oid:dict(obj,label=normalize(obj['label'])) for oid,obj in base.objects.items()}
        self.anchors={i:[] for i in range(len(self.rooms))}
        for oid,obj in self.objects.items():
            area=FUNCTIONAL_AREA_ANCHORS.get(obj['label'].replace(' ','_'))
            if area is None:
                family=(self.label_matches.get(obj['label']) or {}).get('family','')
                area=FUNCTIONAL_AREA_ANCHORS.get(family.replace(' ','_'))
            rid=self.room_at(obj['centre_xy'])
            if area and rid is not None:
                self.anchors[rid].append(dict(id=oid,label=obj['label'],area=area))
        self.candidate_matcher=lambda oid,obj,family: candidate_match(oid,obj,family,self)

    def room_evidence(self, rid, requested, xy=None):
        evidence=super().room_evidence(rid,requested,xy)
        if evidence is not None or rid is None:return evidence
        anchors=self.anchors[rid];areas={a['area'] for a in anchors}
        public={'living area','dining area','kitchen area'}
        # One architectural room can contain multiple public functions. Require
        # positive requested anchors; never rename a bedroom/bathroom to a lounge.
        if requested in public and requested in areas and areas.issubset(public|{'study area'}):
            return dict(room_id=rid,requested=requested,anchors=anchors,
                        evidence='requested function present in shared open-plan room',
                        shared_functions=sorted(areas))
        return None


def map_semantic_scene(base, label_matches=None):
    # Native adapters override topology, floor and authoritative room labels.
    # Preserve those methods instead of replacing the adapter by the SAGE view.
    if type(base) is SemanticScene:result=MapSemanticScene(base,label_matches)
    else:result=copy.copy(base)
    result.label_matches=label_matches or {}
    result.candidate_matcher=lambda oid,obj,family: candidate_match(oid,obj,family,result)
    return result


def nominal_mentions(field):
    """Recover whole nominals even when spaCy splits adjacent noun chunks.

    Source spans stay exact. A preposition/verb separates entities; adjacent
    nouns such as 'feather painting' must not become the query 'feather'.
    """
    from rule_lexicon import ROOMS
    from rule_call_nlp import REGIONS
    tokens=field['tokens']; mentions=[];run=[]
    def finish():
        if not run or not any(t['pos'] in {'NOUN','PROPN'} for t in run):return
        text=field['text'][run[0]['span'][0]:run[-1]['span'][1]]
        content=[t for t in run if t['pos'] not in {'DET','NUM'}]
        if not content:return
        # View direction, ordinal and appearance remain audit qualifiers;
        # identity compounds and adjectives remain retrieval inputs.
        query=normalize(' '.join(t['lemma'] if t['pos'] in {'NOUN','PROPN'} else t['text'] for t in content))
        query=re.sub(r'^(?:(?:nearest|closest|leftmost|rightmost|first|second|third|left|right|open)\s+)+','',query)
        kind='room' if query in ROOMS else 'region' if query in REGIONS or content[-1]['lemma'] in {'gap','space','corner','end','side','area','room'} else 'object'
        mentions.append(dict(text=text,query=query,span=[run[0]['span'][0],run[-1]['span'][1]],
            kind=kind,family=NAMES.get(query),tokens=copy.deepcopy(run)))
    for token in tokens:
        if token['pos'] in {'DET','NUM','ADJ','NOUN','PROPN'}:
            run.append(token)
        else:
            finish();run=[]
    finish()
    return mentions


def mesh_fingerprint(obj):
    """Bind annotations to native instance geometry, never an ordinal ID alone."""
    import hashlib
    import json
    values={k:obj.get(k) for k in ('label','ins_id','native_object_id','bbox_xyxy','z_range','profile')}
    values['label']=normalize(values['label'])
    return hashlib.sha256(json.dumps(values,sort_keys=True,default=list).encode()).hexdigest()


def mesh_history_configuration():
    """Freeze the same optional index consumed by the collection teacher."""
    import os,hashlib,json
    from pathlib import Path
    path=os.environ.get('RULE_MESH_SEMANTIC_HISTORY')
    config=Path(__file__).resolve().parents[1]/'config/rule_mesh_semantics.json'
    expected=None
    if 'RULE_MESH_SEMANTIC_HISTORY' not in os.environ and config.exists():
        configured=json.loads(config.read_text());path=configured['database']
        expected=configured.get('sha256')
    digest=None
    if path:
        with Path(path).open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if expected and digest!=expected:raise ValueError('configured mesh history checksum differs')
    return dict(database=path,sha256=digest)


class MeshSemanticHistory:
    """Read-only instance annotations. Historical edges remain recall evidence."""
    def __init__(self, path):
        import sqlite3
        from pathlib import Path
        self.path=str(Path(path).resolve())
        self.db=sqlite3.connect(Path(self.path).as_uri()+'?mode=ro',uri=True)
        if self.db.execute('SELECT value FROM metadata WHERE key="schema"').fetchone()!=('mesh-semantic-history-v1',):
            raise ValueError('unsupported mesh history schema')
        if self.db.execute('SELECT value FROM metadata WHERE key="complete"').fetchone()!=('true',):
            raise ValueError('mesh history index is incomplete')
        self.cache={};self.description_cache={}

    def instance_matches(self, scene, semantic, query):
        key=(scene,normalize(query))
        if key not in self.cache:
            rows=self.db.execute('''SELECT e.object_id,i.fingerprint,e.score,e.status,m.id
                FROM mentions m JOIN edges e ON e.mention_id=m.id
                JOIN instances i ON i.scene=m.scene AND i.object_id=e.object_id
                WHERE m.scene=? AND m.query=? ORDER BY e.score DESC''',key).fetchall()
            self.cache[key]=rows
        result={}
        for oid,fingerprint,score,status,mention_id in self.cache[key]:
            obj=semantic.objects.get(oid)
            if obj is None or mesh_fingerprint(obj)!=fingerprint:continue
            if oid not in result:
                result[oid]=dict(method='history_instance_recall',label=normalize(obj['label']),
                    score=float(score),history_status=status,history_mention_id=mention_id,
                    history_database=self.path,identity_verified=False)
        return result

    def descriptions(self, scene, semantic):
        """Distinct instance descriptions; duplicate replies add no confidence."""
        if scene not in self.description_cache:
            self.description_cache[scene]=self.db.execute("""SELECT DISTINCT
                m.query,e.object_id,i.fingerprint,m.status FROM mentions m
                JOIN edges e ON e.mention_id=m.id JOIN instances i
                ON i.scene=m.scene AND i.object_id=e.object_id
                WHERE m.scene=? AND m.field='target'
                AND m.status IN ('scene_unique_hypothesis','spatial_unique_hypothesis')
                AND json_extract(e.evidence_json,'$.method')!='scene_text_retrieval'
                AND json_extract(m.context_json,'$.constraint_compile_error') IS NULL""",(scene,)).fetchall()
        return [(query,oid,status) for query,oid,fp,status in self.description_cache[scene]
                if oid in semantic.objects and mesh_fingerprint(semantic.objects[oid])==fp]



class OpenMapCandidates:
    """Retrieve actual scene labels and optional broader semantic hypotheses."""
    def __init__(self, semantic, resolver, history=None, scene=None, *, broaden=False):
        self.semantic=semantic;self.resolver=resolver;self.matches={};self.audit={}
        self.history=history;self.scene=scene;self.instance_matches={};self.family_ids={}
        self.broaden=broaden
        self.labels=list(dict.fromkeys(normalize(o['label']) for o in semantic.objects.values()))

    def recall(self, query):
        query=normalize(query)
        if query in self.matches:return 'scene:'+query
        compact=lambda x: re.sub(r'[^a-z0-9]','',x)
        exact=[label for label in self.labels if compact(label)==compact(query)]
        scores={label:dict(method='scene_label_exact',label=label,score=1.) for label in exact}
        family=NAMES.get(query)
        if not exact and family:
            for oid,obj in self.semantic.objects.items():
                match=candidate_match(oid,obj,family,self.semantic)
                if match:scores[normalize(obj['label'])]=match
        # When lexical recall is available it protects semantic identity;
        # otherwise keep several scene labels for spatial disambiguation.
        ranked=[];category_rejections=[]
        if (not scores or self.broaden) and self.labels:
            values=self.resolver.scores([query],self.labels)[0]
            ranked=[dict(label=self.labels[i],score=float(values[i]),method='scene_text_retrieval')
                    for i in np.argsort(values)[::-1][:12 if self.broaden else 5]]
            for r in ranked:
                # An exact label has score 1 regardless of its scene location.
                # It must not raise the gate for other legitimate subtypes.
                if self.broaden and r['score'] < .76:continue
                # Uninformative taxonomy labels supply no category evidence.
                # Instance-specific historical descriptions remain eligible.
                if r['label'] in {'other','unknown','miscellaneous','unlabeled'}:continue
                conflict=category_conflict(query,r['label']) if self.broaden else None
                if conflict:
                    category_rejections.append(dict(**r,**conflict,reason='contradictory_known_category'))
                    continue
                scores.setdefault(r['label'],r)
        # Add instance-specific language without copying it to same-label peers.
        # Literal current labels keep precedence over historical hypotheses.
        history_matches=(self.history.instance_matches(self.scene,self.semantic,query)
                         if self.history is not None and (not exact or self.broaden) else {})
        if self.history is not None and (not exact or self.broaden) and self.resolver is not None:
            descriptions=self.history.descriptions(self.scene,self.semantic)
            texts=list(dict.fromkeys(q for q,_,_ in descriptions))
            if texts:
                values=self.resolver.scores([query],texts)[0]
                top={texts[i]:float(values[i]) for i in np.argsort(values)[::-1][:5]}
                for text,oid,status in descriptions:
                    if text not in top or (self.broaden and top[text]<.76):continue
                    label=normalize(self.semantic.objects[oid]['label'])
                    conflict=category_conflict(query,label) if self.broaden else None
                    if conflict and oid not in history_matches:
                        # Existing instance-local historical evidence remains
                        # eligible; a new loose description score cannot
                        # independently override a known conflicting class.
                        category_rejections.append(dict(method='history_description_retrieval',
                            label=label,object_id=oid,score=top[text],**conflict,
                            reason='contradictory_known_category'))
                        continue
                    if oid not in history_matches or top[text]>history_matches[oid]['score']:
                        history_matches[oid]=dict(method='history_description_retrieval',
                            label=normalize(self.semantic.objects[oid]['label']),score=top[text],
                            history_description=text,history_status=status,history_database=self.history.path,
                            identity_verified=False)
        if family and not self.broaden:
            history_matches={oid:match for oid,match in history_matches.items()
                             if candidate_match(oid,self.semantic.objects[oid],family,self.semantic)}
        self.instance_matches[query]=history_matches
        self.matches[query]=scores
        self.audit[query]=dict(query=query,candidates=list(scores.values()),
            scene_labels=len(self.labels),absolute_score_gate=self.broaden,
            semantic_score_floor=.76 if self.broaden else None,
            historical_instances=history_matches,
            category_rejections=category_rejections,
            identity_verified=False,exact_label_match=bool(exact),status='recall_hypotheses_require_geometry')
        return 'scene:'+query

    def match(self, oid, obj, family):
        if family.startswith('scene:'):
            match=self.matches.get(family[6:],{}).get(normalize(obj['label']))
            historical=self.instance_matches.get(family[6:],{}).get(oid)
            if historical and (match is None or historical['score']>match.get('score',0.)):
                return historical
            if match and match['method']=='functional_anchor':
                # Shared mesh labels do not share instance-local anchors.
                return candidate_match(oid,obj,NAMES[family[6:]],self.semantic)
            return match
        return candidate_match(oid,obj,family,self.semantic)

    def candidate_ids(self, family):
        """Reuse exactly the matcher's candidate set in relation evaluation."""
        if family not in self.family_ids:
            self.family_ids[family]=tuple(oid for oid,obj in self.semantic.objects.items()
                if self.match(oid,obj,family))
        return self.family_ids[family]


def open_object_spec(contract, graph, retrieval):
    """Compile scene-local entities and PP relations for the shared planner.

    Openings/rooms continue through the existing topology compiler. Derived
    regions explicitly abstain instead of binding one of their landmarks.
    """
    from privileged_teacher import TeacherRejected
    from rule_lexicon import ROOMS
    from rule_nlp_grounding import QUALIFIERS, relation_forests, NUMBERS
    fields={f['field']:f for f in graph['fields']}
    mentions={key:nominal_mentions(field) for key,field in fields.items()}
    main=next(iter(mentions.get('target',[])),None)
    if not main:raise TeacherRejected('rule: target has no physical noun phrase')
    for entity in fields['target'].get('entities',[]):
        if (entity['span'][0]<=main['span'][0] and entity['span'][1]>main['span'][1]
                and (entity.get('lexical') or {}).get('kind') in {'room','region'}):
            raise TeacherRejected('rule: compound room or region endpoint needs topology, not an object label')
    if main['kind']=='room' or main['family']=='doorway' or NAMES.get(main['tokens'][-1]['lemma'])=='doorway':return None
    if main['kind']=='region':raise TeacherRejected('rule: derived region needs a geometric goal, not a landmark object')
    if re.search(r"\b(?:not|never|avoid|except)\b",contract['target']+' '+contract['where'],re.I):
        raise TeacherRejected('rule: negated object constraint needs explicit resolution')
    family=retrieval.recall(main['query']);chains=[];qualifiers=[];room=None
    relation_words={**QUALIFIERS,'behind':'behind','in front of':'in_front_of','infront of':'in_front_of'}
    pattern=re.compile(r'\b('+ '|'.join(map(re.escape,sorted(relation_words,key=len,reverse=True)))+r')\s*$',re.I)
    ego=[]
    for key in ('target','where'):
        field=fields[key];text=field['text'];nodes=mentions[key];chain=[];last=main['span'][1] if key=='target' else 0
        ego_text=text
        for node in nodes:
            if key=='target' and node is main:continue
            if node['query']==main['query']:
                last=node['span'][1];continue
            if node['query'] in {'side','left','right','left side','right side','current view','view','image','front','current image','me'}:
                last=node['span'][1];continue
            if node['query'] in {'robot','agent','current camera','current position','position'} and re.search(
                    r'\b(?:front|left|right|behind|near)\b',text[:node['span'][0]],re.I):
                # Explicit observer references define the ego frame.
                last=node['span'][1];continue
            if node['kind']=='region' and re.search(r'\b(?:view|image|screen)\b',text,re.I):
                last=node['span'][1];continue
            prefix=text[last:node['span'][0]];hit=pattern.search(prefix)
            if node['kind']=='room':
                if re.search(r'\b(?:in|inside|within)\s*(?:the\s*)?$',prefix,re.I):
                    value=ROOMS[node['query']]
                    if room and room!=value:raise TeacherRejected('rule: conflicting target rooms')
                    room=value
                else:raise TeacherRejected('rule: room relation needs topology: '+prefix.strip())
                last=node['span'][1];continue
            if not hit:
                # A relation-bearing nominal must never silently disappear.
                raise TeacherRejected('rule: unresolved open-map relation: '+prefix.strip())
            reference=dict(family=retrieval.recall(node['query']),relations=[])
            count=max([NUMBERS.get(t['text'].lower(),int(t['text']) if t['text'].isdigit() else 1)
                       for t in node['tokens'] if t['pos']=='NUM'] or [1])
            chain.append(dict(kind=relation_words[hit[1].lower()],entity=reference,min_count=count,
                source=dict(field=key,text=text[last+hit.start():node['span'][1]],
                            span=[last+hit.start(),node['span'][1]])))
            # Relative left/right is not an ego direction.
            ego_text=ego_text.replace(text[last+hit.start():node['span'][1]],' ')
            last=node['span'][1]
        if chain:chains.append(relation_forests(chain))
        ego.append(ego_text)
        qualifiers.extend(dict(field=key,text=t['text']) for t in field['tokens']
                          if t['pos']=='ADJ' and t['text'].lower() not in {'left','right','nearest','closest','open'})
    alternatives=[[]]
    for variants in chains:
        if len(alternatives)*len(variants)>32:raise TeacherRejected('rule: too many open-map relation attachments')
        alternatives=[a+b for a in alternatives for b in variants]
    location=normalize(' '.join(ego));sides=set(re.findall(r'\b(left|right)\b',location))
    if len(sides)>1:raise TeacherRejected('rule: inconsistent target sides')
    return dict(family=family,side=next(iter(sides),None),ahead=bool(re.search(r'\b(ahead|front)\b',location)),
        directly_ahead=bool(re.search(r'\b(directly|straight)\b',location)) and not sides,
        nearest=bool(re.search(r'\b(nearest|closest)\b',contract['target'])),kind='approach',
        turn=None,move_side=None,forward=False,straight=False,gentle_move=False,gentle_turn=False,
        relations=[],entity_relation_alternatives=alternatives if chains else [],target_room=room,
        open_entity=main,open_qualifiers=qualifiers,
        qualifier_status='appearance retained as retrieval context; not visually verified')
