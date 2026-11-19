"""Frozen official source references, separate from the teacher's local IDs.

Only native instruction start/end IDs use this index. Local candidates retain
all raw annotations and their legacy numbering. Rasterization is offline;
runtime reads small hash-bound scene files and never imports Shapely.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

SCHEMA = 'sage-official-source-object-map-v1'


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def source_object_identity():
    path = os.environ.get('SAGE_SOURCE_OBJECT_MAP')
    if not path:
        if os.environ.get('SAGE_REQUIRE_SOURCE_OBJECT_MAP') == '1':
            raise ValueError('required frozen SAGE source object map missing')
        return {'schema': 'sage-legacy-raw-source-reference-v1'}
    expected = os.environ.get('SAGE_SOURCE_OBJECT_MAP_SHA256')
    if not expected:
        raise ValueError('frozen SAGE source object map SHA256 required')
    _manifest(str(Path(path).resolve()), expected)
    return dict(schema=SCHEMA, path=str(Path(path).resolve()), sha256=expected,
                resolver_sha256=sha256(__file__))


def _manifest(path, expected):
    stat = Path(path).stat()
    return _checked_manifest(path, expected, (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))


@lru_cache(maxsize=4)
def _checked_manifest(path, expected, stat_identity):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError('SAGE source object map hash mismatch')
    result = json.loads(content)
    if result.get('schema') != SCHEMA:
        raise ValueError('unsupported SAGE source object map schema')
    return result


@lru_cache(maxsize=32)
def _scene_index(path, expected, scene_dir):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError('SAGE source object scene hash mismatch')
    record = json.loads(content)
    if record.get('schema') != SCHEMA or record['scene_dir'] != scene_dir:
        raise ValueError('SAGE source object scene binding mismatch')
    for name, digest in record['assets_sha256'].items():
        if sha256(Path(scene_dir) / name) != digest:
            raise ValueError('SAGE source object input changed: ' + name)
    return record['objects']


def load_source_label_index(scene_dir, *, local=None, identity=None):
    """Resolve native IDs. A frozen-map miss NEVER falls back to raw numbering.

    No-map mode is retained solely for explicitly old code/configuration and
    its historical receipts. Corrected formal attempts require the map flag.
    """
    identity = identity if identity is not None else source_object_identity()
    if identity['schema'] != SCHEMA:
        from .goal_region import load_label_index
        return local if local is not None else load_label_index(scene_dir)
    root = str(Path(scene_dir).resolve())
    manifest = _manifest(identity['path'], identity['sha256'])
    record = manifest['scenes'].get(root)
    if record is None:
        raise ValueError('scene absent from frozen SAGE source map: ' + root)
    path = Path(identity['path']).parent / record['file']
    return _scene_index(str(path), record['sha256'], root)


def require_source_references(task, labels):
    from .navmesh_support import SourceCaseGeometryUnsupported
    for role in ('start_label', 'goal_label'):
        key = str(getattr(task, role, '') or '').strip()
        if key not in labels:
            raise SourceCaseGeometryUnsupported('unresolved official source '+role+': '+key)


def compile_scene(scene_dir, upstream):
    """Reuse the audited upstream cell-centre enumeration, with bbox extrema.

    Masks are visited in bounded vector batches, never stored in the output.
    Connected structural entries participate in identity and mirror-origin
    controls but have no raw Z annotation, so native references to them remain
    explicitly quarantined rather than borrowing a nearby raw object's Z.
    """
    import numpy as np
    import shapely
    from PIL import Image
    from .goal_region import label_index_from_items
    from scipy.ndimage import find_objects

    root = Path(scene_dir).resolve()
    items = json.loads((root / 'labels.json').read_text())
    local = label_index_from_items(items)
    by_instance = {}
    for key, info in local.items():
        if info['ins_id'] in by_instance:
            raise ValueError('duplicate raw SAGE instance ID')
        by_instance[info['ins_id']] = (key, info)
    meta = json.loads((root / 'occupancy.json').read_text())
    with Image.open(root / 'occupancy.png') as image:
        occupancy = np.asarray(image.convert('L'))
    height, width = occupancy.shape
    scale = float(meta['scale'])
    x_min, y_min = meta['min'][:2]
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('invalid official occupancy scale')
    indexed, counters, objects, boxes, extents = {}, Counter(), {}, {}, []
    fmt = lambda x: float(f'{x:.2f}')
    for item in items:
        if 'bounding_box' not in item:
            continue
        vertices = np.array([[p['x'], p['y']] for p in item['bounding_box'][:4]])
        polygon = upstream.Polygon(vertices)
        low = np.clip(np.floor((vertices.min(axis=0) - [x_min, y_min]) / scale).astype(int),
                      [0, 0], [width-1, height-1])
        high = np.clip(np.floor((vertices.max(axis=0) - [x_min, y_min]) / scale).astype(int),
                       [0, 0], [width-1, height-1])
        nx, ny = high - low + 1
        limits = None
        for offset in range(0, int(nx*ny), 4096):
            positions = np.arange(offset, min(offset+4096, int(nx*ny)))
            cols, rows = low[0] + positions // ny, low[1] + positions % ny
            hit = shapely.covers(polygon, shapely.points(
                x_min + (cols+.5)*scale, y_min + (rows+.5)*scale))
            if hit.any():
                c, r = cols[hit], rows[hit]
                current = (int(c.min()), int(r.min()), int(c.max()), int(r.max()))
                limits = current if limits is None else (
                    min(limits[0], current[0]), min(limits[1], current[1]),
                    max(limits[2], current[2]), max(limits[3], current[3]))
        if limits is None:
            continue
        label = item['label']
        counters[label] += 1
        key = f'{upstream.normalize_label(label)}_{counters[label]}'
        instance = str(item.get('ins_id', ''))
        indexed[key] = instance
        c0, r0, c1, r1 = limits
        mc0, mr0, mc1, mr1 = width-1-c1, height-1-r1, width-1-c0, height-1-r0
        extents.append((mc0, mr0, mc1, mr1))
        boxes[key] = tuple(fmt(v) for v in (x_min+mc0*scale, y_min+mr0*scale,
                                          x_min+(mc1+1)*scale, y_min+(mr1+1)*scale))
        local_key, info = by_instance[instance]
        objects[key] = dict(info, source_object_id=key, local_object_id=local_key,
                            source_reference_schema=SCHEMA)
    pixels, pixel_counts = np.unique(occupancy, return_counts=True)
    candidates = [int(v) for v in pixels if 0 < v < 250]
    wall = candidates[int(np.argmax([pixel_counts[np.where(pixels == v)[0][0]]
        for v in candidates]))] if candidates else int(pixels[0])
    for label, mask, minimum, prefix in (
            ('wall', occupancy == wall, 1, 'wall'),
            ('Unable Area', occupancy == 0, 5, 'unable_area')):
        components, number = upstream.nd_label(np.flipud(mask), structure=np.ones((3, 3)))
        sizes = np.bincount(components.ravel())
        for component, bounds in enumerate(find_objects(components), 1):
            if bounds is not None and sizes[component] >= minimum:
                counters[label] += 1
                indexed[f'{upstream.normalize_label(label)}_{counters[label]}'] = f'{prefix}_{component}'
                rs, cs = bounds
                extents.append((cs.start, rs.start, cs.stop-1, rs.stop-1))
    if extents:
        extrema = np.asarray(extents)
        origin_x = fmt(x_min+(extrema[:, 0].min()+.5)*scale) + fmt(x_min+(extrema[:, 2].max()+.5)*scale)
        origin_y = fmt(y_min+(extrema[:, 1].min()+.5)*scale) + fmt(y_min+(extrema[:, 3].max()+.5)*scale)
        for key, bbox in boxes.items():
            objects[key]['official_map_bbox_xyxy'] = bbox
            objects[key]['official_bbox_xyxy'] = (origin_x-bbox[2], origin_y-bbox[3],
                                                  origin_x-bbox[0], origin_y-bbox[1])
    return dict(schema=SCHEMA, scene_dir=str(root), identities=indexed, objects=objects,
                unsupported={k: 'official_structural_component_without_raw_height'
                             for k in indexed if k not in objects},
                assets_sha256={name: sha256(root/name) for name in
                               ('labels.json', 'occupancy.json', 'occupancy.png')})
