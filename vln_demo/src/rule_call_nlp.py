"""Local dependency-based call analysis, with exact original source spans.

The small CPU parser proposes linguistic structure, not grounded truth. Every
modifier, negation, conjunction and unresolved reference is retained. Execution
must separately resolve the graph against map/history and verify trajectories.
No language generation, remote inference, golden goal or action labels.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from rule_lexicon import NAMES, ROOMS

REGIONS={'room':'room','hallway':'corridor','corridor':'corridor','hall':'hall',
         'entrance':'entrance','exit':'exit','corner':'corner','end':'end','area':'area'}
RELATIONS={'near':'near','beside':'near','with':'with', 'on':'on','under':'under',
           'behind':'behind','between':'between','through':'through','into':'into',
           'in':'in','inside':'in','toward':'toward','towards':'toward',
           'to':'to','from':'from','of':'of','at':'at','past':'past','along':'along'}


class CallLanguageParser:
    def __init__(self, model_path):
        import spacy
        from spacy.matcher import PhraseMatcher
        self.nlp=spacy.load(Path(model_path),exclude=['ner'])
        self.model=dict(name=self.nlp.meta['name'],version=self.nlp.meta['version'],spacy=spacy.__version__)
        self.matcher=PhraseMatcher(self.nlp.vocab,attr='LOWER')
        self.lemma_matcher=PhraseMatcher(self.nlp.vocab,attr='LEMMA')
        self.lexicon={alias:dict(kind='object',canonical=family) for alias,family in NAMES.items()}
        self.lexicon.update({alias:dict(kind='room',canonical=family) for alias,family in ROOMS.items()})
        self.lexicon.update({alias:dict(kind='region',canonical=family) for alias,family in REGIONS.items()})
        for alias,doc in zip(self.lexicon,self.nlp.pipe(self.lexicon)):
            self.matcher.add(alias,[self.nlp.make_doc(alias)])
            self.lemma_matcher.add(alias,[doc])

    def graph(self, doc, field):
        nodes=[];root_nodes={};matches=[*self.matcher(doc),*self.lemma_matcher(doc)]
        for number,chunk in enumerate(doc.noun_chunks):
            hits=[(end-start,start,end,self.nlp.vocab.strings[key]) for key,start,end in matches
                  if chunk.start<=start and end<=chunk.end and start<=chunk.root.i<end]
            match=max(hits,default=None)
            token_ids=set(range(match[1],match[2])) if match else {chunk.root.i}
            node=dict(id=f'{field}:entity:{number}',text=chunk.text,span=[chunk.start_char,chunk.end_char],
                      root=chunk.root.i,head=chunk.root.head.i,dependency=chunk.root.dep_,
                      lexical=self.lexicon[match[3]] if match else None,
                      head_lemma=chunk.root.lemma_.lower(),
                      modifiers=[dict(text=t.text,lemma=t.lemma_,dependency=t.dep_,span=[t.idx,t.idx+len(t)])
                                 for t in chunk if t.i not in token_ids and t.pos_ not in {'DET','PUNCT'}],
                      unresolved_reference=chunk.root.pos_=='PRON' or chunk.root.lemma_.lower() in {'room','area','end','corner','side'})
            nodes.append(node);root_nodes[chunk.root.i]=node['id']
        actions=[]
        for token in doc:
            if token.pos_!='VERB':continue
            action=dict(id=f'{field}:action:{token.i}',lemma=token.lemma_.lower(),text=token.text,
                        span=[token.idx,token.idx+len(token)],root=token.i,head=token.head.i,dependency=token.dep_,
                        modifiers=[dict(text=c.text,lemma=c.lemma_,dependency=c.dep_,span=[c.idx,c.idx+len(c)])
                                   for c in token.children if c.dep_ in {'advmod','prt','neg','mark'}],
                        objects=[root_nodes[c.i] for c in token.children if c.i in root_nodes and c.dep_ in {'dobj','pobj','attr','oprd'}])
            actions.append(action);root_nodes[token.i]=action['id']
        relations=[]
        for token in doc:
            if token.pos_!='ADP':continue
            children=[c for c in token.children if c.dep_ in {'pobj','pcomp'}]
            relations.append(dict(id=f'{field}:relation:{token.i}',text=token.text,
                predicate=RELATIONS.get(token.lemma_.lower()),head=root_nodes.get(token.head.i),
                objects=[root_nodes.get(c.i) for c in children],span=[token.idx,token.idx+len(token)],
                head_token=token.head.i,object_tokens=[c.i for c in children]))
        conjunctions=[dict(left=root_nodes.get(t.head.i),right=root_nodes.get(t.i),
                          conjunction=[c.text for c in t.head.children if c.dep_=='cc'],
                          source_tokens=[t.head.i,t.i]) for t in doc if t.dep_=='conj']
        tokens=[dict(i=t.i,text=t.text,lemma=t.lemma_,pos=t.pos_,dependency=t.dep_,head=t.head.i,
                     span=[t.idx,t.idx+len(t)],morph=t.morph.to_dict()) for t in doc]
        return dict(field=field,text=doc.text,entities=nodes,actions=actions,relations=relations,
                    conjunctions=conjunctions,tokens=tokens,
                    status='syntax_hypothesis_not_grounded')

    def parse(self, arguments):
        return self.parse_many([arguments])[0]

    def parse_many(self, calls):
        """Share spaCy batching for history ingestion; retain the same graph API."""
        fields=[];before=copy.deepcopy(calls)
        for index,arguments in enumerate(calls):
            for key in ['target','where']:
                if isinstance(arguments.get(key),str):fields.append((index,key,arguments[key]))
            plans=arguments.get('plan',[])
            if isinstance(plans,list):
                fields.extend((index,f'plan[{i}]',text) for i,text in enumerate(plans) if isinstance(text,str))
        graphs=[[] for _ in calls]
        for (index,name,_),doc in zip(fields,self.nlp.pipe([text for _,_,text in fields],batch_size=256)):
            graphs[index].append(self.graph(doc,name))
        assert before==calls
        return [dict(schema='call-language-graph-v1',original_call=call,
                    original_sha256=hashlib.sha256(json.dumps(call,sort_keys=True).encode()).hexdigest(),
                    model=self.model,fields=graph,
                    warning='Dependency attachment and lexical matches require map/history verification; never execute this graph directly.')
                for call,graph in zip(before,graphs)]

