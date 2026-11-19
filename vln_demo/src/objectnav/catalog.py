"""Normalize source trajectories, never use SAM overlays as observations.

Run with the preparation Python (pyarrow), not the Habitat console Python.
"""
import argparse
from collections import Counter
import hashlib
from functools import lru_cache
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path('/data/shared/groups/mfm/common/anno/gary.zheng')
DEFAULT = ROOT/'objectnav_sam_tracked_gt7m_20260920/episodes.jsonl'
SYNTH = ROOT/'navigation_synthesis_all_20260912'
HM3D = Path('/data/datasets/data/ann/vlf_baidu/HM3D')
MP3D = Path('/data/datasets/data/ann/vlf_baidu/Matterport3D/data/v1/tasks/mp3d')
C = np.array([[0.,0.,-1.],[-1.,0.,0.],[0.,1.,0.]])

def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]

@lru_cache(maxsize=4096)
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def yaw_quat(yaw):
    return [0., math.sin(yaw/2), 0., math.cos(yaw/2)]

class Sources:
    def __init__(self):
        self.jsonl = {}; self.meta = {}; self.tables = {}; self.original = {}
        self.hm_paths = {p.stem: p for p in HM3D.glob('*/*.basis.glb')}
    def raw(self, item):
        cat, scene, eid = item['category'], item['scene'], str(item['episode_id'])
        if cat == 'hm3d_insight':
            path = ROOT/'objectnav_insight_20k_cam025_relaxed_supplement_20260907-022739/trajectories.jsonl'
        elif cat == 'habitatgs_insight':
            camera = 'cam150' if '_cam150_' in eid else 'cam050'
            path = ROOT/f'habitatgs_objectnav_metricdepth_full_scenes2_20260909/scenes/{scene}/{camera}/trajectories.jsonl'
        else:
            path = ROOT/f'habitatgs_objectnav_longrange_1000_20260910/scenes/{scene}/trajectories.jsonl'
        if path not in self.jsonl:
            self.jsonl[path] = {str(r['trajectory_id']):r for r in read_jsonl(path)}
        row = self.jsonl[path][eid]; frames = row['frames']; target = row['selected_target']
        mesh = row.get('gs_ply', row.get('scene_path'))
        gs = cat.startswith('habitatgs')
        navmesh = row.get('navmesh') or str(Path(mesh).with_suffix('.navmesh'))
        # Use the captured initial frame, not a sampled start whose yaw can differ.
        return dict(asset_type='gs' if gs else 'mesh', asset_path=mesh, navmesh=navmesh,
            camera=row['render'], start_position=frames[0]['position'],
            start_rotation=yaw_quat(frames[0]['yaw']-math.pi),
            goal_position=target.get('nav_goal_position',target['target_position']),
            object_position=target.get('object_world_position'),
            reference_path=[f['position'] for f in frames],
            source_trajectory=str(path), source_trajectory_sha256=digest(path),
            source_trajectory_id=eid, scoring='selected_target_standoff', success_distance_m=0.5)
    def lerobot(self, item):
        import pyarrow.parquet as pq
        cat = item['category']; family = cat.split('_')[0]; eid = int(item['episode_id'])
        root = SYNTH/'hm3d_train/lerobot' if family=='hm3d' else SYNTH/f'_parallel_staging/pointgoal_mp3d_v1_train_scene100/scene={item["scene"]}'
        if root not in self.meta:
            self.meta[root] = {int(r['episode_index']):r for p in sorted(root.glob('meta/episodes/chunk-*/file-*.parquet')) for r in pq.read_table(p).to_pylist()}
        meta = self.meta[root][eid]
        if str(meta['scene']) != str(item['scene']):
            raise ValueError('LeRobot scene mismatch')
        path = root/f'data/chunk-{int(meta["data/chunk_index"]):03d}/file-{int(meta["data/file_index"]):03d}.parquet'
        if path not in self.tables:
            # Bound memory to a single data shard; the full dataset is not loaded.
            self.tables.clear()
            self.tables[path] = pq.read_table(path, columns=['episode_index','frame_index','observation.state','observation.state.color_sensor.intrinsics','observation.state.color_sensor.extrinsics']).to_pylist()
        rows = sorted((r for r in self.tables[path] if int(r['episode_index'])==eid),key=lambda r:r['frame_index'])
        if len(rows)!=meta['length'] or not rows: raise ValueError('incomplete trajectory')
        states = np.asarray([r['observation.state'] for r in rows]); positions = states[:,:3] @ C
        original_path = ROOT/f'habitat_objectnav_train_scene100_20260912/episodes/{family}_{"v2" if family=="hm3d" else "v1"}_train_scene100.json'
        if family not in self.original:
            self.original[family] = {}
            for candidate in json.loads(original_path.read_text())['episodes']:
                self.original[family].setdefault(Path(candidate['scene_id']).stem.removesuffix('.basis'), []).append(candidate)
        stem = str(item['scene']).removesuffix('.basis')
        matches = [r for r in self.original[family].get(stem,[]) if r['instruction_text']==item['instruction'] and np.linalg.norm(np.asarray(r['start_position'])-positions[0])<0.02]
        # Resolve repeated starts by matching heading. No index-based guessing.
        q = np.asarray(yaw_quat(float(states[0,6])))
        matches = [r for r in matches if abs(np.dot(q,np.asarray(r['start_rotation'])/np.linalg.norm(r['start_rotation'])))>0.999]
        if len(matches)!=1: raise ValueError(f'ambiguous original ObjectNav goal: {len(matches)} matches')
        original = matches[0]
        mesh = self.hm_paths[item['scene']] if family=='hm3d' else MP3D/stem/f'{stem}.glb'
        navmesh = SYNTH/f'navmesh_cache/{family}/{mesh.stem}.h1.50_r0.10_c0.05_mc0.20_ms45.navmesh'
        if family=='mp3d': navmesh=mesh.with_suffix('.navmesh')  # process_all.sh uses --prebaked-navmesh
        intr = rows[0]['observation.state.color_sensor.intrinsics']; width,height=640,480
        # Camera extrinsics are OpenGL camera-to-world, unlike the FLU observation.state.
        extr = np.asarray(rows[0]['observation.state.color_sensor.extrinsics']).reshape(4,4)
        camera_height = float(extr[1,3]-positions[0,1])
        if not 0.05<camera_height<3: raise ValueError(f'unsupported camera extrinsics: height={camera_height}')
        return dict(asset_type='mesh', asset_path=str(mesh), navmesh=str(navmesh),
            camera=dict(width=width,height=height,hfov_deg=math.degrees(2*math.atan(width/(2*intr[0]))),camera_height_m=camera_height),
            start_position=positions[0].tolist(),start_rotation=q.tolist(),goal_position=original['goal_position'],
            object_position=original['metadata']['object_position'],reference_path=positions.tolist(),
            source_trajectory=str(path),source_trajectory_sha256=digest(path),source_trajectory_id=original['episode_id'],
            source_goal=str(original_path), scoring='selected_object_goal_viewpoint',success_distance_m=0.5)
    def convert(self,item):
        data=self.lerobot(item) if item['category'].endswith('_train') else self.raw(item)
        for k in ['asset_path','navmesh']:
            if not Path(data[k]).is_file(): raise FileNotFoundError(data[k])
        identity=[item['category'],str(item['scene']),str(item['episode_id'])]
        logical=hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:20]
        scene=Path(item['scene']).stem.replace('.basis','')
        return dict(dataset='objectnav',split='train',role='guide',language='en' if item['instruction'].isascii() else 'zh',
            seed=0,logical_id=logical,episode_id=str(item['episode_id']),scene=f'objectnav-{item["category"]}-{scene}',
            category=item['category'],source_scene=item['scene'],instruction=item['instruction'],target_label=item['target_label'],
            **data)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=DEFAULT);p.add_argument('--output',type=Path,required=True);p.add_argument('--per-category',type=int,default=0)
    args=p.parse_args(); sources=Sources();counts=Counter();out=[];rejected=[];seen=set()
    for item in read_jsonl(args.source):
        cat=item['category']
        if args.per_category and counts[cat]>=args.per_category: continue
        try:
            row=sources.convert(item)
            if row['logical_id'] in seen: raise ValueError('duplicate category/scene/episode')
            seen.add(row['logical_id']);out.append(row);counts[cat]+=1
            if len(out)%100==0: print(json.dumps(dict(converted=len(out),rejected=len(rejected))),flush=True)
        except (ValueError,KeyError,OSError) as exc:
            rejected.append(dict(category=cat,scene=item['scene'],episode_id=item['episode_id'],error=str(exc)))
            # Pilot attempts a bounded number of candidates per category.
            if args.per_category and sum(r['category']==cat for r in rejected)>=3:
                counts[cat]=args.per_category
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        for row in out:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    report=dict(source=str(args.source),source_sha256=digest(args.source),manifest_sha256=digest(args.output),accepted=len(out),categories=dict(Counter(r['category'] for r in out)),rejected=rejected)
    args.output.with_suffix('.report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
