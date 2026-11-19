"""Private, CPU-only rule evidence. Never include this in observation/prompt data."""
from __future__ import annotations
import hashlib
import math
import os
import time
from collections import OrderedDict
import numpy as np

SCHEMA = 'sage-rule-credit-state-v1'

class RuleCreditState:
    navigation = None
    def __init__(self, env):
        from .goal_region import load_label_index
        from .planner import passable_grid
        from scipy.sparse import coo_matrix
        self.scene = env.current_episode.scene
        self.occ = env._occ_for(self.scene)
        self.labels = load_label_index(env.ds.scene_dir(self.scene))
        self.navigation = getattr(getattr(env, 'sim', None), 'navigation', None)
        if self.navigation is not None:
            from .navmesh_support import digest
            self.map_hash = digest(self.navigation.manifest_path)
            self.regions = {}
            self.seen = set()
            self.episode = None
            return
        self.free = passable_grid(self.occ, env.robot_radius)
        h, w = self.free.shape
        ids = np.arange(h*w).reshape(h, w)
        rows, cols, vals = [], [], []
        for dr, dc in ((1,0),(0,1),(1,1),(1,-1)):
            r0,r1=max(0,-dr),min(h,h-dr)
            c0,c1=max(0,-dc),min(w,w-dc)
            ok = self.free[r0:r1,c0:c1] & self.free[r0+dr:r1+dr,c0+dc:c1+dc]
            if dr and dc:
                ok &= self.free[r0:r1,c0+dc:c1+dc] & self.free[r0+dr:r1+dr,c0:c1]
            a,b=ids[r0:r1,c0:c1][ok],ids[r0+dr:r1+dr,c0+dc:c1+dc][ok]
            rows.extend((a,b)); cols.extend((b,a))
            vals.extend((np.full(len(a), math.hypot(dr,dc)*self.occ.scale),)*2)
        self.graph=coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(h*w,h*w)).tocsr()
        self.fields=OrderedDict()
        rr,cc=np.indices((h,w))
        self.xx=self.occ.lower[0]+cc*self.occ.scale
        self.yy=self.occ.lower[1]+(h-1-rr)*self.occ.scale
        self.map_hash=hashlib.sha256(self.free.tobytes()+str((self.occ.lower,self.occ.scale,env.robot_radius)).encode()).hexdigest()
        self.seen=set()
        self.episode=None

    def distance(self, xy, bbox):
        from scipy.sparse.csgraph import dijkstra
        box=tuple(float(x) for x in bbox)
        c,r=self.occ.to_pixel(*xy)
        if not (0<=r<self.free.shape[0] and 0<=c<self.free.shape[1] and self.free[r,c]):
            return None
        if box[0]<=xy[0]<=box[2] and box[1]<=xy[1]<=box[3]:
            return 0.0
        if box not in self.fields:
            targets=self.free & (self.xx>=box[0]) & (self.xx<=box[2]) & (self.yy>=box[1]) & (self.yy<=box[3])
            indices=np.flatnonzero(targets)
            if not len(indices):
                return None
            self.fields[box]=dijkstra(self.graph, directed=True, indices=indices,min_only=True).reshape(self.free.shape)
            while len(self.fields)>12:
                self.fields.popitem(last=False)
        self.fields.move_to_end(box)
        value=float(self.fields[box][r,c])
        return value if math.isfinite(value) else None

    def physical_region(self, key):
        from .physical_navigation import TargetRegion
        from .navmesh_support import FloorUnresolved
        if key not in self.regions:
            try:
                self.regions[key] = TargetRegion(self.navigation, key, self.labels[key])
            except FloorUnresolved:
                # Visible entities with unresolved support supply no distance
                # evidence. The task target is required at reset and cannot
                # take this optional-entity path.
                self.regions[key] = None
        return self.regions[key]

    def snapshot(self, env, obs, metrics):
        from .goal_region import semantic_family, official_object_arrival
        from .visibility_audit import project_bbox, depth_support, visible_to_bbox_2d
        started=time.perf_counter()
        ep=env.current_episode
        if ep.episode_id!=self.episode or env._steps==0:
            self.seen=set(); self.episode=ep.episode_id
            self.arrival_cursor=0; self.ever_goal_reached=False
        official=metrics.get('official_object_success_detail') or {}
        xy=tuple(env.sim.xy)
        target=official.get('target_object_id')
        from .source_objects import load_source_label_index
        source_labels = (load_source_label_index(env.ds.scene_dir(self.scene), local=self.labels)
                         if os.environ.get('SAGE_SOURCE_OBJECT_MAP') else self.labels)
        family=semantic_family(source_labels.get(target,{}).get('label',''))
        depth=(obs or {}).get('depth')
        visible=[]
        camera=env.client.camera
        depth_valid=depth is not None and np.asarray(depth).shape==(camera.height,camera.width)
        if depth_valid:
            pos=(*xy,env.sim.camera_z)
            from .visibility_audit import visible_object_ids
            visible = visible_object_ids(self.labels, depth, camera, pos, env.sim.yaw,
                candidate=(None if self.navigation is not None else
                           lambda obj: visible_to_bbox_2d(self.occ, xy, obj, 0.0)))
            self.seen.update(visible)
        distances={}
        for key in sorted(self.seen):
            box=official_object_arrival(self.labels[key],xy)['expanded_bbox_xyxy']
            if self.navigation is not None:
                region = self.physical_region(key)
                distances[key] = region.distance(self.navigation.position) if region else None
            else:
                distances[key]=self.distance(xy,box)
        box=official.get('expanded_bbox_xyxy')
        goal=(env._physical_target.distance(self.navigation.position) if self.navigation is not None
              else self.distance(xy,box) if official.get('available') and box else None)
        # Read every newly executed native endpoint, including points between
        # tool boundaries. A call that enters then leaves still owns its arrival.
        # No inferred geodesic-zero or interpolated straight-line crossing.
        arrival={}
        if official.get('available') and box and hasattr(env, 'trail'):
            trail=env.trail_xyz if self.navigation is not None else env.trail
            cursor=getattr(self,'arrival_cursor',0)
            if len(trail)<cursor:
                raise ValueError('native trajectory shrank without episode reset')
            reached=(any(env._physical_target.contains(point) for point in trail[cursor:])
                     if self.navigation is not None else
                     any(box[0]<=point[0]<=box[2] and box[1]<=point[1]<=box[3]
                         for point in trail[cursor:]))
            self.ever_goal_reached=getattr(self,'ever_goal_reached',False) or reached
            self.arrival_cursor=len(trail)
            arrival=dict(arrival_evidence=('supported_goal_membership_native_xyz_v2' if self.navigation is not None
                                                else 'official_goal_membership_native_trail_points_v1'),
                         ever_goal_reached=bool(self.ever_goal_reached))
        return dict(schema=('sage-rule-credit-state-v2' if self.navigation is not None else SCHEMA),
                    **({'ground_position': self.navigation.position.tolist()} if self.navigation is not None else {}),map_sha256=self.map_hash,goal_contract_sha256=official.get('target_contract_sha256',official.get('contract_sha256')),
                    target_object_id=target,goal_distance_m=goal,position_xy=list(xy),
                    visible_entities=visible,information_available=bool(depth_valid),
                    relevant_entities=[key for key in sorted(self.seen) if family and semantic_family(self.labels[key]['label'])==family],
                    anchored_entities={key:dict(label=self.labels[key]['label'],distance_m=distances[key]) for key in sorted(self.seen)},
                    collisions=int(env.collisions),seconds=time.perf_counter()-started,**arrival)


def snapshot(env, obs, metrics):
    started=time.perf_counter()
    current=getattr(env,'_rule_credit_state',None)
    if (current is None or current.scene!=env.current_episode.scene
            or current.navigation is not getattr(env.sim, 'navigation', None)):
        current=RuleCreditState(env); env._rule_credit_state=current
    result=current.snapshot(env,obs,metrics)
    result['seconds']=time.perf_counter()-started
    return result
