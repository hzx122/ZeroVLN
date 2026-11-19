"""Compile nominal relation chains into map-checkable alternative trees.

NLP noun spans/lemmas identify entities. A small explicit grammar records every
removed qualifier, retaining plausible PP attachments instead of blindly
trusting the dependency parser. Original calls are never modified.
"""
from __future__ import annotations

import copy
import itertools
import json
import re

from rule_lexicon import NAMES

QUALIFIERS={
    'to the left of':'left_of','to the right of':'right_of',
    'on the left of':'left_of','on the right of':'right_of',
    'left of':'left_of','right of':'right_of',
    'next to':'near','adjacent to':'near','beside':'near','near':'near','by':'near',
    'on top of':'supported_by','on':'supported_by','with':'supports',
}
PREFIX=re.compile(r'\s*('+ '|'.join(map(re.escape,sorted(QUALIFIERS,key=len,reverse=True)))+r')\s+',re.I)
NUMBERS={'one':1,'two':2,'three':3,'four':4,'five':5}
EGO_PREFIX=set('on to the left right side of current view image ahead straight directly slightly front me in and visible just'.split())


class NLPConstraintRejected(ValueError):pass


def reference_entity(node,field):
    lexical=node.get('lexical')
    if not lexical or lexical['kind']!='object' or lexical['canonical']=='doorway':
        raise NLPConstraintRejected('relation reference needs a mapped physical object')
    modifiers=node['modifiers'];count=1;elongated=False
    for modifier in modifiers:
        word=modifier['text'].lower()
        if word=='long':elongated=True
        elif modifier['dependency']=='nummod' and (word in NUMBERS or word.isdigit()):
            count=NUMBERS.get(word,int(word) if word.isdigit() else 0)
            if not 1<=count<=5:raise NLPConstraintRejected('reference cardinality outside verified range')
        else:raise NLPConstraintRejected('unresolved reference modifier: '+word)
    root=field['tokens'][node['root']]
    if root.get('morph',{}).get('Number')=='Plur':count=max(2,count)
    return dict(family=lexical['canonical'],relations=[],elongated=elongated),count


def relation_forests(chain,max_alternatives=32):
    """Enumerate right-branching and earlier-head nominal PP attachments.

    Map candidates are evaluated against the union of all feasible trees. A
    unique object under one selected parse is insufficient if another feasible
    parse selects a different object.
    """
    count=1
    for index in range(len(chain)):count*=index+1
    if count>max_alternatives:raise NLPConstraintRejected('too many plausible relation attachments')
    forests=[]
    for parents in itertools.product(*(range(i+1) for i in range(len(chain)))):
        nodes=[dict(relations=[]),*[copy.deepcopy(item['entity']) for item in chain]]
        for index,(parent,item) in enumerate(zip(parents,chain),1):
            nodes[parent]['relations'].append(dict(kind=item['kind'],family=nodes[index]['family'],
                entity=nodes[index],min_count=item['min_count'],source=item['source']))
        forests.append(nodes[0]['relations'])
    return forests


def lower_relations(contract,graph,fields=None):
    if graph.get('original_call')!=contract:
        raise NLPConstraintRejected('NLP graph does not belong to this original call')
    internal=copy.deepcopy(contract);alternatives=[[]];evidence=[]
    target=next((f for f in graph['fields'] if f['field']=='target'),None)
    if not target or not target['entities']:return internal,{},[]
    main=target['entities'][0].get('lexical')
    if not main or main['kind']!='object':return internal,{},[]
    family=main['canonical']
    for field in graph['fields']:
        if fields is not None and field['field'] not in fields:continue
        text=field['text'];nodes=field['entities'];changes=[]
        # Each occurrence of the main entity may carry a relation. A where
        # fragment can start with a qualifier whose subject is the target.
        starts=[(node['span'][1],node['id']) for node in nodes
                if node.get('lexical') and node['lexical']['kind']=='object'
                and node['lexical']['canonical']==family]
        if field['field']=='where':
            starts[:0]=[(match.start(),'implicit_target') for match in PREFIX.finditer(text)
                        if set(re.findall(r'[a-z]+',text[:match.start()].lower()))<=EGO_PREFIX]
        for start,head in starts:
            if any(lo<=start<hi for lo,hi,_ in changes):continue
            cursor=start;chain=[]
            while True:
                match=PREFIX.match(text,cursor)
                if not match:break
                node=next((n for n in nodes if n['span'][0]==match.end()),None)
                if node is None:break
                lexical=node.get('lexical')
                if not lexical or lexical['kind']!='object':break
                entity,count=reference_entity(node,field)
                if entity['family']==family:
                    raise NLPConstraintRejected('same-family relational reference is unresolved')
                end=node['span'][1]
                chain.append(dict(kind=QUALIFIERS[match[1].lower()],entity=entity,min_count=count,
                    source=dict(field=field['field'],span=[match.start(),end],text=text[match.start():end])))
                cursor=end
            if not chain:continue
            # In a strict chain grammar, an unconsumed relative clause or
            # coordinated modifier must stay in the lowered call and fail the
            # existing compiler; it must not silently vanish with a subtree.
            variants=relation_forests(chain)
            if len(alternatives)*len(variants)>32:
                raise NLPConstraintRejected('cross-field relation alternatives exceed budget')
            alternatives=[a+b for a in alternatives for b in variants]
            changes.append((start,cursor,''));evidence.append(dict(field=field['field'],head=head,
                span=[start,cursor],text=text[start:cursor],alternatives=len(variants)))
        for start,end,replacement in sorted(changes,reverse=True):text=text[:start]+replacement+text[end:]
        if field['field'].startswith('plan['):internal['plan'][int(field['field'][5:-1])]=text
        else:internal[field['field']]=text
    if not evidence:return internal,{},[]
    unique={json.dumps(a,sort_keys=True):a for a in alternatives}
    return internal,dict(entity_relation_alternatives=list(unique.values())),evidence
