"""Opt-in bridge to frozen SAGE public-code scoring (never model input)."""
import hashlib
import importlib.util
import json
import math
import os
from functools import lru_cache
from pathlib import Path

BUNDLE_SCHEMA = 'sage-public-highlevel-ppu-v1'
SCHEMA = 'sage-designated-target-xy-v2'

def enabled():
    return bool(os.environ.get('SAGE_BENCH_PROTOCOL_ROOT'))

@lru_cache(maxsize=1)
def bundle():
    root = Path(os.environ['SAGE_BENCH_PROTOCOL_ROOT'])
    protocol_path = root / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    assert protocol['schema'] == BUNDLE_SCHEMA
    for filename, key in (('dataset/closedloop_subset.json', 'manifest_sha256'),
                          ('official_semantic_maps_provenance.json', 'maps_provenance_sha256')):
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == protocol[key]
    for filename, digest in protocol['upstream_sha256'].items():
        assert hashlib.sha256((root / 'upstream' / filename).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((root / 'episodes.json').read_bytes()).hexdigest() == protocol['episodes_sha256']
    spec = importlib.util.spec_from_file_location('_sage_frozen_object_success', root / 'upstream/object_based_success.py')
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    # Reuse the audited exact min/max reduction, avoiding rescanning every mask
    # pixel for every simulator micro-step. It calls the upstream transform.
    from .semantic_map_transform import cached_reverse_mapping
    scorer.reverse_position_mapping = cached_reverse_mapping(scorer.reverse_position_mapping)
    return root, protocol, json.loads((root / 'episodes.json').read_text()), scorer

@lru_cache(maxsize=1)
def identity():
    root, protocol, _, _ = bundle()
    bundle_hash = hashlib.sha256((root / 'protocol.json').read_bytes()).hexdigest()
    return dict(schema=SCHEMA, object_success_schema=SCHEMA,
                contract_path=str(root / 'protocol.json'),
                source_bundle_sha256=bundle_hash,
                contract_sha256=hashlib.sha256((SCHEMA + ':' + bundle_hash).encode()).hexdigest(),
                expansion_radius_m=1.0, requires_agent_stop=False)

@lru_cache(maxsize=44)
def evaluator(scene):
    root, protocol, _, scorer = bundle()
    maps = list((root / 'official_semantic_maps').glob(f"*_{scene.removeprefix('sage-')}_Complete.json"))
    assert len(maps) == 1, (scene, maps)
    provenance = json.loads((root / 'official_semantic_maps_provenance.json').read_text())
    assert hashlib.sha256(maps[0].read_bytes()).hexdigest() == provenance['maps'][maps[0].name]
    result = scorer.ObjectBasedSuccessEvaluator(str(maps[0]), verbose=False)
    assert result.semantic_map_data, scene
    return result

def score(scene, episode_id, xy):
    _, _, episodes, _ = bundle()
    ep = episodes[episode_id]
    pos = [-float(xy[0]), -float(xy[1]), float(ep['start_position'][2])]
    ev = evaluator(scene)
    target = str(ep.get('end_item') or '').strip()
    # Never invoke evaluate_success: its positional fallback can change targets.
    if not target or ev.get_object_bbox(target) is None:
        raise ValueError(f'missing designated SAGE target: {target!r}')
    designated_bbox(scene, episode_id)  # Validate the exact target's geometry.
    success = bool(ev.is_position_in_object_area(pos, target, 1.0))
    return dict(schema=SCHEMA, success=success, available=True,
                target_object_id=target, target_selection='episode.end_item',
                contract_sha256=identity()['contract_sha256'], expansion_radius_m=1.0,
                reference_distance_m=math.dist(pos, ep['goals'][0]['position']),
                official_position=pos, strict_target_success=success,
                metric_semantics='designated_target_xy_only')

def auto_stop(info):
    detail = info.get('official_object_success_detail') or {}
    if detail.get('schema') == 'sage-designated-target-navmesh-v2':
        if not detail.get('available'):
            raise RuntimeError('physical target scoring unavailable')
        return bool(detail['success'])
    if detail.get('schema') != SCHEMA or not detail.get('available'):
        raise RuntimeError('official benchmark scoring unavailable; refusing silent fallback')
    return bool(detail['success'])

def trajectory_metrics(episode_id, positions):
    """Public SPL samples position after decisions, with straight-line optimum."""
    ep = bundle()[2][episode_id]
    start, goal = ep['start_position'], ep['goals'][0]['position']
    points = [start] + positions
    length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
    optimum = math.dist(start, goal)
    return length, optimum


@lru_cache(maxsize=512)
def designated_bbox(scene, episode_id, expansion_radius=0.0):
    """Exact frozen-map target bbox, inverse-transformed into local SAGE XY."""
    import numpy as np
    _, _, episodes, scorer = bundle()
    ev = evaluator(scene)
    target = str(episodes[episode_id].get('end_item') or '').strip()
    obj = ev.get_object_bbox(target) if target else None
    if not obj or len(obj.get('bbox_m', [])) != 4:
        raise ValueError(f'missing or malformed designated target: {target!r}')
    reverse = scorer.reverse_position_mapping
    origin = np.asarray(reverse(0., 0., ev.semantic_map_data))
    columns = np.column_stack((np.asarray(reverse(-1.,0.,ev.semantic_map_data))-origin,
                               np.asarray(reverse(0.,-1.,ev.semantic_map_data))-origin))
    x0,y0,x1,y1 = map(float, obj['bbox_m'])
    if not np.isfinite([x0,y0,x1,y1]).all() or x0 > x1 or y0 > y1:
        raise ValueError(f'invalid designated target bbox: {target}')
    if not math.isfinite(expansion_radius) or expansion_radius < 0:
        raise ValueError('invalid target expansion radius')
    # Match is_position_in_object_area: expand in official semantic-map metres
    # BEFORE inverse mapping. Do not expand an already transformed local box.
    x0, y0, x1, y1 = (x0-expansion_radius, y0-expansion_radius,
                       x1+expansion_radius, y1+expansion_radius)
    corners = np.asarray([np.linalg.solve(columns, np.asarray(p)-origin)
                          for p in ((x0,y0),(x0,y1),(x1,y0),(x1,y1))])
    return target, [*corners.min(axis=0).tolist(), *corners.max(axis=0).tolist()]
