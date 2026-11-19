"""Offline training admission, separate from success scoring and navigation.

Extends the rule-teacher same_floor_task_audit standing-anchor/text guard.
Constant released path Z is never evidence of a floor. All instruction
siblings of a route share its admission; ambiguous multi-level routes are
quarantined rather than declared physically valid.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
import json
import hashlib
from functools import lru_cache
from types import SimpleNamespace

import numpy as np

from .goal_region import label_index_from_items, semantic_family
from .route_extractor import _bbox_distance_xy

SCHEMA = 'sage-rl-same-floor-scope-v1'
TYPES = ('Relative_Relationship', 'Add_Causality', 'Scenario_Driven',
         'Attribute-based', 'Area-based')
TYPE_ALIASES = {
    'Add_Object': 'Add_Causality', 'Added_Causality': 'Add_Causality',
    'Attribute-Based': 'Attribute-based', 'Scene_Driven': 'Scenario_Driven',
    'Situation_Driven': 'Scenario_Driven', 'Economic_Scenario_Driven': 'Scenario_Driven',
    'Self-narrative': 'Scenario_Driven',
}
ANCHOR_FAMILIES = {'chair', 'table', 'sofa', 'bed', 'single_bed', 'double_bed'}
_MOTION = re.compile(
    r"\b(?:go|walk|head|move|proceed|continue|travel|climb|ascend|descend)\s+"
    r"(?:(?:up|down)\s+(?:the\s+)?(?:stairs|staircase|steps)|upstairs|downstairs)\b")
_LOW = re.compile(r'\b(?:downstairs|(?:lower|ground|bottom)\s+(?:\w+\s+){0,2}(?:floor|level|storey|story))\b')
_HIGH = re.compile(r'\b(?:upstairs|(?:upper|top)\s+(?:\w+\s+){0,2}(?:floor|level|storey|story))\b')


@lru_cache(maxsize=2)
def _native_floor_ledger(path, expected_sha256, stat_identity):
    """Read the existing JSONL audit once per unchanged, frozen ledger.

    The dedup ledger reader has a different JSON schema and cannot consume
    this instruction-level audit. Hash and unique source identities are checked
    before sharing its read-only records among independently reset teachers.
    """
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError('native route floor ledger hash mismatch')
    records = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row['scene']), str(row['task_id']))
        if key in records:
            raise ValueError('duplicate native route floor identity')
        records[key] = row
    return records


def native_route_admission(binding, *, scene, task_id, provenance, nav, start_audit):
    """Bind whole-route floor evidence independently of reference walkability.

    A blocked GT reference does not reject an otherwise local valid call.
    A single native floor band proves floor scope independently; multilevel
    unresolved routes remain quarantined, without consulting released path Z.
    This proves collection scope only, never local semantics or GT success.
    """
    path = Path(binding['path'])
    if not path.is_absolute():
        raise ValueError('absolute native route floor ledger required')
    stat = path.stat()
    records = _native_floor_ledger(str(path), binding['sha256'],
        (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    row = records.get((str(scene), str(task_id)))
    if row is None:
        raise ValueError('native route floor identity missing')
    if row.get('source_provenance') != provenance:
        raise ValueError('native route floor original instruction mismatch')
    manifest = row.get('manifest') or {}
    if (manifest.get('sha256') != start_audit['manifest_sha256']
            or Path(manifest.get('path', '')).resolve() != nav.manifest_path.resolve()):
        raise ValueError('native route floor navigation seed mismatch')
    ground = float(start_audit['ground_z_m'])
    if not math.isfinite(ground):
        raise ValueError('finite independent native start floor required')
    original_status = row.get('original_route_status')
    proof = dict(schema='sage-native-whole-route-runtime-scope-v1', scene=str(scene),
        task_id=str(task_id), ledger=dict(path=str(path), sha256=binding['sha256']),
        original_route_status=original_status, original_route_reason=row.get('original_route_reason'),
        manifest=manifest, ground_z_m=ground, source_provenance=provenance,
        quality_approved=False, reference_walkability_is_quality_gate=False)
    if original_status == 'cross_floor_original_route':
        proof.update(status='excluded', reason='cross_floor_original_route')
    elif original_status == 'same_floor_original_route':
        recorded = row.get('ground_z_m')
        if recorded is None or not math.isfinite(float(recorded)) or abs(float(recorded)-ground) > .06:
            raise ValueError('native route floor differs from frozen whole-route proof')
        proof.update(status='retained', reason='complete_native_route_floor_measured')
    elif (original_status == 'unresolved' and
          row.get('original_route_reason') == 'native_reference_route_blocked'):
        triangles = np.asarray(nav.triangles, float)
        if (triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or
                not len(triangles) or not np.isfinite(triangles).all()):
            raise ValueError('finite complete native support triangles required')
        low, high = float(triangles[:, :, 2].min()), float(triangles[:, :, 2].max())
        single_band = max(abs(low-ground), abs(high-ground)) <= .06
        proof['independent_floor_band'] = dict(native_min_z_m=low, native_max_z_m=high,
            maximum_start_floor_deviation_m=.06, native_triangle_count=len(triangles),
            complete_native_mesh=True, single_floor_verified=single_band)
        proof.update(status='retained' if single_band else 'quarantined',
            reason='independent_native_single_floor_band' if single_band else 'multilevel_route_floor_unresolved')
    else:
        proof.update(status='quarantined', reason=row.get('original_route_reason') or 'native_route_floor_unresolved')
    return proof


def instruction_cross_floor(text: str) -> bool:
    text = str(text).lower()
    # Preserve stairs as landmarks and explicitly negated stair movement.
    motion = any(not re.search(r"(?:not|never|don't|avoid)\s+$",
                              text[max(0, m.start()-24):m.start()])
                 for m in _MOTION.finditer(text))
    opposite = bool(_LOW.search(text) and _HIGH.search(text))
    numbered = set(re.findall(
        r'\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+(?:floor|level)\b', text))
    numbered = {dict(first='1', second='2', third='3', fourth='4',
                     **{'1st': '1', '2nd': '2', '3rd': '3', '4th': '4'})[x]
                for x in numbered}
    directed = bool(re.search(
        r'\b(?:ascend|descend|climb)\b.{0,35}\b(?:stairs|staircase|floor|level)\b', text))
    return motion or opposite or len(numbered) > 1 or directed


def anchor_height(info: dict):
    if semantic_family(info.get('label', '')) not in ANCHOR_FAMILIES:
        return None
    try:
        low, high = map(float, info['z_range'])
    except (KeyError, TypeError, ValueError):
        return None
    return low if all(map(math.isfinite, (low, high))) and .1 <= high-low <= 1.8 else None


def floor_levels(objects: dict) -> list[float]:
    """Repeated furniture bases suggest levels, not staircase connectivity."""
    clusters = []
    for height in sorted(x for x in map(anchor_height, objects.values()) if x is not None):
        if not clusters or height-clusters[-1][0] > .35:
            clusters.append([height])
        else:
            clusters[-1].append(height)
    return [sum(group)/len(group) for group in clusters if len(group) >= 2]


def route_audit(task, instructions: list[dict], objects: dict,
                levels: list[float]) -> dict:
    counts = Counter()
    hits = []
    for index, ins in enumerate(instructions):
        kind = TYPE_ALIASES.get(ins.get('instruction_type'), ins.get('instruction_type'))
        if kind not in TYPES:
            raise ValueError(f'unsupported instruction type: {kind!r}')
        counts[kind] += 1
        if instruction_cross_floor(ins.get('generated_instruction', '')):
            hits.append({'index': index, 'instruction_type': ins['instruction_type'],
                         'instruction': ins['generated_instruction']})
    source = objects.get(task.start_label, {})
    target = objects.get(task.goal_label, {})
    source_height = anchor_height(source)
    target_height = anchor_height(target)
    # A plant is only a floor anchor when its base agrees with repeated furniture
    # bases; a plant on a desk and a ceiling fixture cannot invent a level.
    for info, which in ((source, 'source'), (target, 'target')):
        if semantic_family(info.get('label', '')) in {'plant', 'potted_plant'}:
            low, high = info.get('z_range', (float('nan'), float('nan')))
            if .1 <= high-low <= 1.8 and any(abs(low-z) <= .25 for z in levels):
                if which == 'source':
                    source_height = low
                else:
                    target_height = low
    nearby = (source_height is not None and source.get('bbox_xyxy') is not None
              and _bbox_distance_xy((task.start['x'], task.start['y']),
                                   source['bbox_xyxy']) <= 1.5)
    different = (nearby and target_height is not None
                 and abs(source_height-target_height) >= 2.)
    same = (nearby and target_height is not None
            and abs(source_height-target_height) <= .35)
    multilevel = bool(levels and max(levels)-min(levels) >= 2.)
    if hits or different:
        status, reason = 'excluded', ('cross_floor_instruction_sibling' if hits
                                     else 'cross_floor_standing_anchors')
    elif multilevel and not same:
        status, reason = 'quarantined', 'multilevel_route_floor_unresolved'
    else:
        status, reason = 'retained', ('matching_standing_anchors' if same
                                     else 'no_cross_floor_evidence_single_level_scene')
    return dict(schema=SCHEMA, status=status, reason=reason,
                instruction_counts=dict(counts), instruction_hits=hits,
                source_anchor_floor_m=source_height, target_anchor_floor_m=target_height,
                nearby_source_anchor=bool(nearby), scene_floor_levels_m=levels,
                physical_route_validated=False)


class SameFloorAdmission:
    """Bind materialized routes to original five-type instructions, fail closed."""
    def __init__(self, dataset, official_dataset):
        self.dataset = dataset
        self.official_dataset = official_dataset
        self.records = {}
        self.sources = {str(Path(__file__).resolve())}
        self._scene = None
        self._samples = {}
        self._objects = {}
        self._levels = []

    def __call__(self, scene, task):
        key = (scene, task.task_id)
        if key in self.records:
            return self.records[key]['status'] == 'retained'
        if scene != self._scene:
            path = Path(self.official_dataset.traj_file(scene) or '')
            labels = Path(self.dataset.scene_dir(scene)) / 'labels.json'
            doc = json.loads(path.read_text())
            blocks = doc['scenes']
            if len(blocks) != 1 or str(blocks[0]['scene_id']) != self.dataset.bare(scene):
                raise ValueError(f'official scene binding mismatch: {scene}')
            self._samples = {str(sample['trajectory_id']): sample for sample in blocks[0]['samples']}
            if len(self._samples) != len(blocks[0]['samples']):
                raise ValueError(f'duplicate official trajectory IDs: {scene}')
            self._objects = label_index_from_items(json.loads(labels.read_text()))
            self._levels = floor_levels(self._objects)
            from .source_objects import load_source_label_index
            self._objects = load_source_label_index(labels.parent, local=self._objects)
            self.sources.update((str(path.resolve()), str(labels.resolve())))
            self._scene = scene
        match = re.fullmatch(r'official_(\d+)_(\d+)_\d+', task.task_id)
        route_id = match[2] if match else task.task_id.rsplit('_', 1)[0]
        if match and match[1] != self.dataset.bare(scene):
            raise ValueError(f'route scene mismatch: {key}')
        sample = self._samples.get(route_id)
        if sample is None:
            raise ValueError(f'original route missing: {key}')
        instructions = sample.get('instructions') or []
        # Check IDs, endpoints AND text. Never join nearby routes by XY or silently
        # treat absent metadata as a same-floor decision.
        if not any(ins.get('generated_instruction') == task.instruction and
                   ins.get('instruction_type') == task.instruction_type and
                   (ins.get('start'), ins.get('end')) ==
                   (task.start_label, task.goal_label) for ins in instructions):
            raise ValueError(f'original instruction binding mismatch: {key}')
        audit = route_audit(task, instructions, self._objects, self._levels)
        # Some official siblings name different object instances on the same
        # released route. Check their own endpoints instead of assuming the RR
        # object's height applies to all five categories.
        for start, end in sorted({(ins['start'], ins['end']) for ins in instructions}):
            variant = SimpleNamespace(start=task.start, start_label=start, goal_label=end)
            geometry = route_audit(variant, [], self._objects, self._levels)
            severity = {'retained': 0, 'quarantined': 1, 'excluded': 2}
            if severity[geometry['status']] > severity[audit['status']]:
                audit.update(status=geometry['status'], reason=geometry['reason'],
                             sibling_geometry_endpoints=[start, end])
        audit.update(scene=scene, episode_id=task.task_id, original_trajectory_id=route_id)
        self.records[key] = audit
        return audit['status'] == 'retained'

    def report(self):
        rows = list(self.records.values())
        by_type = {kind: Counter() for kind in TYPES}
        for row in rows:
            for kind, count in row['instruction_counts'].items():
                by_type[kind][row['status']] += count
        return dict(schema=SCHEMA, enabled=True, instruction_types=list(TYPES),
                    type_aliases=TYPE_ALIASES, route_counts=dict(Counter(row['status'] for row in rows)),
                    instruction_variant_counts={k: dict(v) for k, v in by_type.items()},
                    reason_counts=dict(Counter(row['reason'] for row in rows)),
                    physical_route_validated=False,
                    policy='exclude all siblings of cross-floor routes; quarantine unresolved multilevel routes')
