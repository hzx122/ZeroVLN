"""Ordered, source-preserving clauses from an existing call-language graph.

Adapted from FGR2R's root/conj/parataxis dependency chunking method; see
third_party/Fine-Grained-R2R-NOTICE.txt for provenance and the MIT license.
Unlike its lemma output, every text here is an exact original substring.
These are syntax hypotheses, not grounding or permission to execute. In
particular, consumers must not discard an unknown first clause to use a later
known target. No NLP model, teacher, or environment is loaded by this module.
"""
from __future__ import annotations

import copy


NAVIGATION = frozenset(('approach reach move walk go enter pass face turn '
                        'proceed continue head navigate cross exit leave '
                        'advance follow stop align').split())
ROTATION = frozenset({'turn', 'face', 'align'})
CONNECTORS = frozenset({'and', 'then', 'next', 'afterwards', 'subsequently'})
TEMPORAL = frozenset({'before', 'after', 'while', 'until', 'unless'})
PUNCTUATION = frozenset({';', '.', '!', '?'})


def _word(token):
    return token['text'].lower()


def _lemma(token):
    # Directional "left" is often mistagged as the verb "leave" by the small
    # parser. Never create an extra action for that token.
    return _word(token) if _word(token) in {'left', 'right'} else token.get('lemma', token['text']).lower()


def _predicate(token, position, tokens):
    """Use dependency predicates, with a narrow repair for imperative POS errors."""
    word = _word(token)
    if word in {'left', 'right'}:
        return False
    if token.get('pos') == 'VERB':
        return True
    if word not in NAVIGATION:
        return False
    # A noun such as "the next turn" is not an imperative. Repair only at the
    # start or after a clause connector/punctuation (optionally an adverb).
    earlier = tokens[:position]
    while earlier and (earlier[-1].get('pos') == 'ADV' or _word(earlier[-1]) in {'please', 'not', 'never'}):
        earlier = earlier[:-1]
    return not earlier or _word(earlier[-1]) in CONNECTORS | PUNCTUATION


def _trim(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] == ','):
        end -= 1
    return start, end


def segment_plan(field):
    """Return ordered source clauses without choosing or skipping endpoints.

    Input is one ``rule_call_nlp`` field graph. ``action`` can be inherited for
    an explicit elliptical sequence ("approach A and then B"); its text is
    never fabricated. Noun/direction conjunctions remain one clause. A clause
    containing before/after or another temporal dependency has
    ``order_status='unknown_order'``; the caller must resolve or stop there,
    not silently assume written order. An earlier plain clause remains usable.
    """
    text = field['text']
    tokens = sorted(field.get('tokens', []), key=lambda t: t['span'][0])
    for token in tokens:
        start, end = token['span']
        if text[start:end] != token['text']:
            raise ValueError('plan token span does not match original text')
    predicates = {t['i'] for n, t in enumerate(tokens) if _predicate(t, n, tokens)}
    # Each boundary removes only a connector/punctuation span. Everything
    # else, including unsupported words and negation, stays in a clause.
    boundaries = []
    for n, token in enumerate(tokens):
        word = _word(token)
        if word in PUNCTUATION:
            boundaries.append((token['span'][0], token['span'][1], 'sentence', False))
            continue
        if word not in CONNECTORS:
            continue
        # "the next door", "the then current room", and noun coordination
        # are references, not action boundaries.
        if word == 'next' and token.get('dependency') not in {'advmod', 'cc'}:
            continue
        m = n + 1
        while m < len(tokens) and (_word(tokens[m]) in {'then', ','} or
                (tokens[m].get('pos') == 'ADV' and _word(tokens[m]) not in {'left', 'right'})):
            m += 1
        if m >= len(tokens):
            continue
        following = tokens[m]
        action_after = following['i'] in predicates
        if not action_after and following.get('pos') == 'AUX':
            k = m + 1
            while k < len(tokens) and (tokens[k].get('dependency') == 'neg' or _word(tokens[k]) in {'not', "n't", 'never'}):
                k += 1
            action_after = k < len(tokens) and tokens[k]['i'] in predicates
        then_present = word in {'then', 'afterwards', 'subsequently'} or any(
            _word(t) == 'then' for t in tokens[n + 1:m])
        # Explicit "then" can coordinate an omitted action and another NP.
        # Bare "and" does not turn table-and-chairs into two instructions.
        noun_after = following.get('pos') in {'DET', 'NOUN', 'PROPN', 'PRON'}
        elliptical = then_present and noun_after and not action_after
        if not (action_after or elliptical):
            continue
        # Don't split embedded complements merely because a verb was found:
        # and/then plus a genuine predicate is the root/conj/parataxis intent
        # of FGR2R. Explicit connectors also repair small-parser root errors.
        begin = token['span'][0]
        if n and _word(tokens[n - 1]) == 'and' and then_present:
            begin = tokens[n - 1]['span'][0]
        end = following['span'][0]
        boundaries.append((begin, end, 'then' if then_present else word, elliptical))
    # A new dependency root/parataxis without a conjunction also separates
    # imperatives; unlike FGR2R we do not output lemmatized/reconstructed text.
    for n, token in enumerate(tokens):
        if token['i'] not in predicates or token.get('dependency', '').lower() not in {'root', 'parataxis'}:
            continue
        previous = [t for t in tokens[:n] if t['i'] in predicates]
        if not previous:
            continue
        if any(b[0] >= previous[-1]['span'][1] and b[1] <= token['span'][0] for b in boundaries):
            continue
        if any(_word(t) in TEMPORAL for t in tokens[:n]):
            continue
        # An earlier subordinate or negated predicate cannot be thrown away;
        # keep it in this clause unless punctuation/sequence resolved it.
        if previous[-1].get('dependency', '').lower() not in {'root', 'parataxis', 'conj'} and not (
                _word(previous[-1]) in NAVIGATION and previous[-1].get('pos') != 'VERB'):
            continue
        boundaries.append((token['span'][0], token['span'][0], 'dependency', False))

    # Coalesce "and then" boundaries found from both connector tokens.
    merged = []
    for boundary in sorted(boundaries):
        if merged and boundary[0] < merged[-1][1]:
            old = merged[-1]
            merged[-1] = (old[0], max(old[1], boundary[1]),
                          'then' if 'then' in {old[2], boundary[2]} else old[2],
                          old[3] or boundary[3])
        else:
            merged.append(boundary)
    spans = []
    start = 0
    connector = None
    elliptical = False
    for begin, end, kind, next_elliptical in merged:
        a, b = _trim(text, start, begin)
        if a < b:
            spans.append((a, b, connector, elliptical))
        start, connector, elliptical = end, dict(text=text[begin:end], span=[begin, end], kind=kind), next_elliptical
    a, b = _trim(text, start, len(text))
    if a < b:
        spans.append((a, b, connector, elliptical))

    clauses = []
    inherited = None
    for start, end, connector, elliptical in spans:
        part = [t for t in tokens if start <= t['span'][0] < end]
        candidates = [t for t in part if t['i'] in predicates]
        action = candidates[0] if candidates else None
        lemma = _lemma(action) if action else inherited if elliptical else None
        if action:
            inherited = lemma
        temporal = [dict(text=t['text'], span=t['span']) for t in part if _word(t) in TEMPORAL]
        entities = [copy.deepcopy(e) for e in field.get('entities', [])
                    if start <= next((t['span'][0] for t in part if t['i'] == e['root']), -1) < end]
        clauses.append(dict(index=len(clauses), field=field.get('field', 'plan'),
            text=text[start:end], span=[start, end], connector_before=connector,
            action=lemma, action_token=action['i'] if action else None,
            kind='rotation' if lemma in ROTATION else 'navigation' if lemma in NAVIGATION else 'unknown',
            elliptical=bool(elliptical and action is None),
            negated=any(t.get('dependency') == 'neg' or _word(t) in {'not', "n't", 'never'} for t in part),
            order_status='unknown_order' if temporal else 'ordered', temporal_markers=temporal,
            entities=entities, tokens=copy.deepcopy(part)))
    return dict(schema='ordered-plan-clauses-v1', field=field.get('field', 'plan'), text=text,
                status='unknown_order' if any(c['order_status'] == 'unknown_order' for c in clauses) else 'ordered',
                clauses=clauses, warning='Preserve clause order; unknown or negated earlier clauses may not be skipped.')
