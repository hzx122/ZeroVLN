"""Private instruction-relevant discoveries, using SAGE's visibility gates.

Native .house instances are reward evidence only. No goals/reference route,
room visitation proxy, semantic oracle or annotations enter policy inputs.
"""
import hashlib
from functools import lru_cache
from pathlib import Path
import re

import numpy as np

from mp3d_teacher_semantics import DEFAULT_MP3D_ROOT, load_mp3d_house
from rule_lexicon import NAMES
from sage.goal_region import semantic_family
from sage.visibility_audit import visible_object_ids
from rs3d.gs_client import CameraSpec

CONTRACT = 'instruction_object_family_same_frame_depth_discovery_v1'


def family(label):
    label = re.sub(r'[_#\s]+', ' ', str(label).strip().lower())
    return semantic_family(NAMES.get(label, label))


def instruction_families(instruction, objects):
    """Whole noun phrases; alias/plural expansion, never substring matching."""
    names = {str(o['label']).lower().replace('_', ' '): family(o['label'])
             for o in objects.values() if o['label'] not in ('unknown', 'other')}
    names.update({name: family(canonical) for name, canonical in NAMES.items()})
    # Longest phrases win: 'floor lamp' must not create a floor discovery.
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(n) + ('' if n.endswith('s') else 's?')
                         for n in sorted(names, key=len, reverse=True)) + r')\b')
    matches = set()
    for match in pattern.finditer(instruction.lower()):
        noun = match.group()
        if noun not in names and noun.endswith('s'):
            noun = noun[:-1]
        if noun in names:
            matches.add(names[noun])
    return matches


@lru_cache(maxsize=8)
def _scene(path, size, mtime):
    data = Path(path).read_bytes()
    return load_mp3d_house(path), hashlib.sha256(data).hexdigest()


class InformationState:
    def __init__(self, episode):
        scene = Path(episode.scene_id).stem
        path = DEFAULT_MP3D_ROOT / scene / (scene + '.house')
        stat = path.stat()
        native, self.sha256 = _scene(str(path), stat.st_size, stat.st_mtime_ns)
        self.source = str(path)
        self.families = instruction_families(episode.instruction.instruction_text, native.objects)
        self.objects = {key: obj for key, obj in native.objects.items()
                        if family(obj['label']) in self.families}
        self.seen = set()

    def snapshot(self, gym, obs, geometry):
        if not isinstance(obs, dict) or 'depth' not in obs or 'rgb' not in obs:
            raise ValueError('information reward requires actual same-frame RGB and metric depth')
        specs = {s['uuid']: s for s in geometry['sensors']}
        if set(specs) != {'rgb', 'depth'}:
            raise ValueError('information reward requires RGB/depth camera identity')
        rgb, depth_spec = specs['rgb'], specs['depth']
        for key in ('resolution', 'hfov', 'position', 'orientation'):
            if rgb[key] != depth_spec[key]:
                raise ValueError('information reward RGB/depth camera mismatch')
        if (not np.allclose(rgb['orientation'], 0., atol=1e-6)
                or not np.allclose(np.asarray(rgb['position'])[[0, 2]], 0., atol=1e-6)):
            raise ValueError('information reward requires the formal upright camera')
        if gym.env.sim.sensor_suite.sensors['depth'].normalize_depth:
            raise ValueError('information reward requires metric depth, not normalized depth')
        height, width = rgb['resolution']
        camera = CameraSpec(width=width, height=height, hfov_deg=float(rgb['hfov']))
        depth = np.asarray(obs['depth'])
        if depth.shape == (height, width, 1):
            depth = depth[..., 0]
        if depth.shape != (height, width) or np.asarray(obs['rgb']).shape[:2] != (height, width):
            raise ValueError('information reward frame/camera resolution mismatch')
        # Habitat uses zero for rays that do not hit scene geometry. A valid
        # view of sky/sea can therefore be entirely zero. It proves no object
        # visibility; the shared SAGE depth gate already rejects zero rays.
        if not np.isfinite(depth).all() or (depth < 0).any():
            raise ValueError('information reward metric depth is unavailable: '
                             'nonfinite or negative metric depth')
        depth_evidence = dict(pixels=int(depth.size),
                              positive_pixels=int((depth > 0).sum()),
                              no_hit_pixels=int((depth == 0).sum()))
        x, y, yaw = gym._world_pose_rad()
        ground = geometry['position']
        if not np.allclose([x, y], [-ground[2], -ground[0]], atol=1e-6):
            raise ValueError('information reward pose differs from observed frame')
        camera_pos = (x, y, float(ground[1]) + float(rgb['position'][1]))
        # Habitat clips metric depth at its configured far plane; saturated
        # pixels cannot prove that an annotated box at that depth was seen.
        sensor = gym.env.sim.sensor_suite.sensors['depth']
        far = getattr(sensor, 'max_depth_value', None)
        if far is not None:
            depth = np.where(depth < float(far), depth, np.nan)
        visible = visible_object_ids(self.objects, depth, camera, camera_pos, yaw)
        self.seen.update(visible)
        return dict(information_available=True, relevant_entities=sorted(self.seen),
                    visible_entities=visible, information_contract=CONTRACT,
                    information_source=dict(path=self.source, sha256=self.sha256),
                    information_families=sorted(self.families),
                    information_depth=depth_evidence)
