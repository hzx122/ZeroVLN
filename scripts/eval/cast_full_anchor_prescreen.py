"""Conservative all-anchor pre-screen; outputs proposals, never CAST labels.

Reuses the case audit's SAGE geometry and exact replay. Straight two-segment
proposals rejoin the original golden at 1.6m. This deliberately misses routes
requiring wider obstacle detours; no-candidate does not mean impossible.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cast_case_feasibility_audit import (
    SageSet, OccMap, CameraSpec, smooth_path, _path_is_safe,
    _execution_replay_free, load_label_index, project_bbox,
    visible_to_bbox_2d, length, RADIUS,
)
from sage.capability_benchmark import point_at_arc

DATA = Path('/data/zerovln')
SOURCE = DATA / 'vln_verl/artifacts/action-head-sft-data-v43-official-gvln-full-20260911-r2/manifest.jsonl'
OUTPUT = DATA / 'vln_verl/artifacts/cast-v43-full-anchor-prescreen-20260914-r1'
CONFIG = dict(radius_m=RADIUS, collision_mode='usd_rooms', furniture_mode='none',
              rejoin_arc_m=1.6, endpoint_distances_m=[0.8, 1.2, 1.6],
              endpoint_bearings_left_deg=[-45, -20, 0, 20, 45],
              max_rejoin_path_m=2.4, min_separation_at_0p4m=0.15,
              min_separation_at_1p2m=0.5, min_landmark_width_px=32,
              min_landmark_height_px=24, min_landmark_area_ratio=0.004,
              max_landmark_distance_m=5.0, replay_step_m=0.1)
COARSE = ('sofa', 'table', 'desk', 'wardrobe', 'cabinet', 'shelf', 'shelving',
          'storage rack', 'door', 'bed', 'painting', 'mirror', 'window', 'plant')


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def compact_index(source=None, output=None, split="train"):
    SOURCE = Path(source) if source is not None else globals()["SOURCE"]
    OUTPUT = Path(output) if output is not None else globals()["OUTPUT"]
    marker = OUTPUT / 'input_index.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        stat = SOURCE.stat()
        if saved['source_size'] != stat.st_size or saved['source_mtime_ns'] != stat.st_mtime_ns:
            raise RuntimeError('Source manifest changed; use a new output directory')
        return saved
    grouped = defaultdict(list)
    digest = hashlib.sha256()
    seen = set()
    rows = 0
    with SOURCE.open('rb') as stream:
        for rows, line in enumerate(stream, 1):
            digest.update(line)
            row = json.loads(line)
            if row['fast_window_index'] != 0:
                continue
            aid = row['slow_anchor_id']
            if aid in seen:
                raise RuntimeError(f'Duplicate first window: {aid}')
            seen.add(aid)
            if row['split'] != split or row['action_head_as_tool_version'] != 'v4.3':
                raise RuntimeError('Unexpected split or protocol')
            fields = ['slow_anchor_id', 'source_scene', 'source_task_id',
                      'source_point_index', 'source_pose_xyyaw', 'instruction',
                      'contract', 'current_image', 'reference_image', 'sample',
                      'slow_anchor_remaining_arc_m', 'valid_action_steps']
            compact = {k: row[k] for k in fields}
            compact['source_manifest_line'] = rows
            grouped[row['source_scene']].append(compact)
    for scene, anchors in grouped.items():
        dump(OUTPUT / 'inputs' / f'{scene}.json', anchors)
    stat = SOURCE.stat()
    saved = dict(source=str(SOURCE), source_sha256=digest.hexdigest(),
                 source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns,
                 windows=rows, anchors=len(seen), scenes=sorted(grouped))
    dump(marker, saved)
    return saved


def process_scene(scene, output=None, sage_split="official_gvln_action_head_p0_v3full827train"):
    OUTPUT = Path(output) if output is not None else globals()["OUTPUT"]
    anchors = json.loads((OUTPUT / 'inputs' / f'{scene}.json').read_text())
    dataset = SageSet(split=sage_split)
    sd = dataset.scene_dir(scene)
    occ = OccMap(sd, collision_mode='usd_rooms', furniture_mode='none')
    tasks = {t.task_id: t for t in dataset.tasks(scene)}
    labels = load_label_index(sd)
    coarse = [(oid, info) for oid, info in labels.items()
              if any(term in info['label'].lower().replace('_', ' ') for term in COARSE)]
    camera = CameraSpec(width=640, height=480, hfov_deg=115)
    counts = Counter(anchors=len(anchors))
    candidates = []
    for row in anchors:
        x, y, yaw = row['source_pose_xyyaw']
        start = [x, y]
        task = tasks.get(row['source_task_id'])
        if task is None:
            counts['missing_task'] += 1
            continue
        if not Path(row['current_image']).is_file() or not Path(row['reference_image']).is_file():
            counts['missing_image'] += 1
            continue
        if not occ.is_free(x, y, RADIUS):
            counts['invalid_start'] += 1
            continue
        suffix = [list(p[:2]) for p in task.waypoints[row['source_point_index']:]]
        if not suffix or math.dist(start, suffix[0]) > 0.03:
            counts['source_pose_mismatch'] += 1
            continue
        counts['asset_valid'] += 1
        if row['valid_action_steps'] < 16 or length(suffix) < 1.6:
            counts['short_remaining_route'] += 1
            continue
        rejoin = point_at_arc(suffix, 1.6)
        paths = []
        for distance, angle in itertools.product(CONFIG['endpoint_distances_m'], CONFIG['endpoint_bearings_left_deg']):
            heading = yaw + math.radians(angle)
            endpoint = [x + distance * math.cos(heading), y + distance * math.sin(heading)]
            if distance + math.dist(endpoint, rejoin) > CONFIG['max_rejoin_path_m']:
                continue
            if not occ.is_free(*endpoint, RADIUS) or not occ.seg_free(x, y, *endpoint, RADIUS):
                continue
            if not occ.seg_free(*endpoint, *rejoin, RADIUS):
                continue
            raw = [start, endpoint]
            if math.dist(endpoint, rejoin) > 1e-6:
                raw.append(rejoin)
            points, _ = smooth_path(occ, raw, RADIUS)
            if length(points) > CONFIG['max_rejoin_path_m'] or not _path_is_safe(occ, np.asarray(points), RADIUS):
                continue
            ok, replay = _execution_replay_free(occ, points, step_m=0.1, radius_m=RADIUS)
            if not ok or replay['collisions']:
                continue
            paths.append(dict(endpoint_distance_m=distance, endpoint_bearing_left_deg=angle,
                              path_xy=points, path_length_m=length(points), replay=replay,
                              prefix_xy=point_at_arc(points, 0.4), window_xy=point_at_arc(points, 1.2)))
        if paths:
            counts['has_local_path'] += 1
        pairs = []
        for a, b in itertools.combinations(paths, 2):
            near = math.dist(a['prefix_xy'], b['prefix_xy'])
            far = math.dist(a['window_xy'], b['window_xy'])
            if near >= CONFIG['min_separation_at_0p4m'] and far >= CONFIG['min_separation_at_1p2m']:
                pairs.append((far, near, a, b))
        if not pairs:
            counts['no_distinct_pair'] += 1
            continue
        far, near, a, b = max(pairs, key=lambda p: (p[0], p[1]))
        counts['geometric_pair'] += 1
        projected = []
        for oid, info in coarse:
            if math.dist(start, info['centre_xy']) > CONFIG['max_landmark_distance_m']:
                continue
            projection = project_bbox(camera, (x, y, 1.2), yaw, info)
            if projection is None or projection['width_px'] < CONFIG['min_landmark_width_px'] or projection['height_px'] < CONFIG['min_landmark_height_px'] or projection['projected_area_ratio'] < CONFIG['min_landmark_area_ratio']:
                continue
            projected.append(dict(object_id=oid, label=info['label'], projection=projection,
                                  los_2d=visible_to_bbox_2d(occ, start, info, RADIUS)))
        grounded = [o for o in projected if o['los_2d']]
        counts['pair_with_large_projected_landmark'] += bool(projected)
        counts['pair_with_landmark_los'] += bool(grounded)
        row.update(goal_label=task.goal_label, goal_xy=task.goal_xy,
                   rejoin_xy=rejoin, branch_pair=[a, b], separation_0p4m=near,
                   separation_1p2m=far, landmarks=projected,
                   priority_candidate=bool(grounded), accepted_training_pair=False)
        candidates.append(row)
    result = dict(scene=scene, counts=dict(counts), mask_path=occ.usd_mask_path,
                  mask_version=occ._usd_version, candidates=candidates)
    dump(OUTPUT / 'scenes' / f'{scene}.json', result)
    return dict(scene=scene, **dict(counts))


def main():
    for sub in ('inputs', 'scenes'):
        (OUTPUT / sub).mkdir(parents=True, exist_ok=True)
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    run = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), script=str(Path(__file__).resolve()),
               script_sha256=script_hash, config=CONFIG, workers=4)
    run_path = OUTPUT / 'run_manifest.json'
    if run_path.exists() and json.loads(run_path.read_text())['script_sha256'] != script_hash:
        raise RuntimeError('Script changed; use a new output directory')
    dump(run_path, run)
    index = compact_index()
    print(json.dumps(dict(event='indexed', windows=index['windows'], anchors=index['anchors'], scenes=len(index['scenes']))), flush=True)
    summaries = []
    pending = []
    for scene in index['scenes']:
        path = OUTPUT / 'scenes' / f'{scene}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            summaries.append(dict(scene=scene, **saved['counts']))
        else:
            pending.append(scene)
    errors = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(process_scene, s): s for s in pending}
        for future in as_completed(futures):
            try:
                summaries.append(future.result())
            except Exception as exc:
                errors.append(dict(scene=futures[future], error=repr(exc)))
            if (len(summaries) + len(errors)) % 20 == 0:
                progress = dict(done=len(summaries), total=len(index['scenes']), errors=len(errors),
                                geometric_pairs=sum(s.get('geometric_pair', 0) for s in summaries),
                                priority=sum(s.get('pair_with_landmark_los', 0) for s in summaries))
                dump(OUTPUT / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    totals = Counter()
    for s in summaries:
        totals.update({k: v for k, v in s.items() if k != 'scene'})
    scene_counts = {key: sum(s.get(key, 0) > 0 for s in summaries)
                    for key in ['asset_valid', 'geometric_pair', 'pair_with_large_projected_landmark', 'pair_with_landmark_los']}
    distribution = {str(n): sum(s.get('pair_with_landmark_los', 0) >= n for s in summaries)
                    for n in [1, 5, 10, 20, 50, 100]}
    output = dict(run=run, input=index, completed_scenes=len(summaries), errors=errors,
                  totals=dict(totals), scene_counts=scene_counts,
                  scenes_with_at_least_n_priority_anchors=distribution,
                  scenes=sorted(summaries, key=lambda s: (-s.get('pair_with_landmark_los', 0), s['scene'])),
                  accepted_training_pairs=0,
                  limitations=['Pre-screen only: no generated instructions or new RGB/depth renders.',
                               'Landmark projection and 2D LOS do not certify visual grounding or instruction compatibility.',
                               'A nearby landmark need not distinguish the two branches; semantic selection remains pending.',
                               'Rejoins existing golden at 1.6m; wider/curved detours and short terminal anchors are not exhaustively searched.',
                               'Continuation after rejoin relies on the existing golden data physical QC.',
                               'Counts refer to anchors, not deduplicated physical states or independent tasks.'])
    dump(OUTPUT / 'summary.json', output)
    print(json.dumps(dict(event='complete', totals=dict(totals), scene_counts=scene_counts, errors=errors)), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
