"""Data-only ObjectNav adapter for the unchanged isolated pi-agent runner."""
import hashlib
import json
from pathlib import Path

CATEGORIES = ('habitatgs_insight', 'habitatgs_longrange')


def load_manifest(path):
    path = Path(path)
    data = path.read_bytes()
    lock = json.loads(path.with_name('lock.json').read_text())
    if hashlib.sha256(data).hexdigest() != lock['manifest_sha256']:
        raise ValueError('ObjectNav manifest hash mismatch')
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    if not rows or len(rows) != lock['episodes'] or len({r['logical_id'] for r in rows}) != len(rows):
        raise ValueError('ObjectNav manifest count/identity mismatch')
    for row in rows:
        if row['dataset'] != 'objectnav' or row['category'] not in CATEGORIES:
            raise ValueError('manifest must contain the two Gary Habitat-GS categories')
        for key in ('goal_position', 'start_position', 'start_rotation', 'asset_path', 'navmesh', 'camera', 'instruction', 'seed'):
            if key not in row:
                raise ValueError('ObjectNav manifest missing ' + key)
    return rows


def controller_environment(row, output, render_url):
    # A controller sees exactly this episode; rendering uses its original asset ID.
    path = Path(output) / 'episode.jsonl'
    path.write_text(json.dumps(row, ensure_ascii=False) + '\n')
    return dict(DEMO_BACKEND='gary-' + row['category'].replace('_', '-'),
                OBJECTNAV_MANIFEST=str(path), OBJECTNAV_EPISODE=row['logical_id'],
                OBJECTNAV_RENDER_URL=render_url)
