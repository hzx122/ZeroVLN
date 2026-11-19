"""Private ObjectNav state on source NavMesh; images from a stateless service.

Scoring is distance to the source-selected goal viewpoint/standoff, NOT the
Habitat category-wide ObjectNav benchmark. SAM/GT never enter observations.
"""
import io
import json
from .catalog_io import read_catalog
from .categories import select_category
import math
import os
from pathlib import Path
import threading
from types import SimpleNamespace
from urllib.request import Request,build_opener,ProxyHandler
import uuid
import numpy as np
from hsim.env import HabSceneEnv, SimFacade, Instruction

class RemoteSim(SimFacade):
    _remote_render_installed=True
    _remote_defer_depth=0
    _remote_deferred_steps=0
    _sensor_suite=SimpleNamespace(get_observations=lambda raw:raw)
    def __init__(self,sim,row,url):
        super().__init__(sim);self.row=row;self.url=url;self._last=None
        self.habitat_config=SimpleNamespace(scene=row['scene'])
    def get_agent_state(self,agent_id=0):
        import quaternion
        state=super().get_agent_state(agent_id)
        cam=state.position+quaternion.rotate_vectors(state.rotation,[0,self.row['camera']['camera_height_m'],0])
        state.sensor_states={k:SimpleNamespace(position=cam,rotation=state.rotation) for k in ['rgb','depth']}
        return state
    def get_sensor_observations(self):
        from habitat_sim.utils.common import quat_to_coeffs
        if self._remote_defer_depth and self._last is not None:
            self._remote_deferred_steps+=1
            return self._last
        state=self.get_agent_state();rid=uuid.uuid4().hex
        payload=dict(asset_id=self.row['logical_id'],request_id=rid,position=state.position.tolist(),rotation=quat_to_coeffs(state.rotation).tolist())
        req=Request(self.url+'/render',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        from .transport import render_bytes
        body=render_bytes(req,timeout=600)
        with np.load(io.BytesIO(body),allow_pickle=False) as z:
            if str(z['request_id'])!=rid:raise ValueError('render identity mismatch')
            self._last={k:z[k].copy() for k in ['rgb','depth']}
        return self._last

class ObjectNavEnv(HabSceneEnv):
    def __init__(self,manifest=None,url=None,category=None):
        import habitat_sim as hs
        from habitat_sim.utils.common import quat_from_coeffs,quat_rotate_vector
        self._hs=hs;self._quat_from_coeffs=quat_from_coeffs;self._quat_rotate=quat_rotate_vector
        self.rows=select_category(read_catalog(manifest or os.environ['OBJECTNAV_MANIFEST']),category)
        selected=os.environ.get('OBJECTNAV_EPISODE')
        if selected:self.rows=[r for r in self.rows if r['logical_id']==selected]
        if not self.rows:raise ValueError('empty ObjectNav catalog')
        self.episodes=[SimpleNamespace(episode_id=r['episode_id'],scene=r['scene'],scene_id=r['asset_path'],instruction=Instruction(r['instruction']),start_position=r['start_position'],start_rotation=r['start_rotation'],info={'scoring':r['scoring']},record=r) for r in self.rows]
        self.current_episode=self.episodes[0];self.url=(url or os.environ.get('OBJECTNAV_RENDER_URL','http://127.0.0.1:7041')).rstrip('/')
        self.scene=None;self.sim=None;self._lock=threading.Lock();self._map_cache={};self._map_inv={}
        self.max_episode_steps=0;self.want_depth=True;self._steps=0;self._stopped=False;self.trail=[];self.collisions=0
        self._use_scene(self.current_episode.scene)
    def seed(self,value):self._seed=int(value)
    @property
    def _elapsed_steps(self):return self._steps
    def _use_scene(self,scene):
        row=self.current_episode.record
        if self.sim is not None and self.sim.row['logical_id']==row['logical_id']:return
        if self.sim is not None:self.sim.close()
        hs=self._hs;cfg=hs.SimulatorConfiguration();cfg.scene_id='NONE';cfg.create_renderer=False;cfg.enable_physics=False
        a=hs.agent.AgentConfiguration();a.sensor_specifications=[]
        a.action_space={i:hs.agent.ActionSpec(name,hs.agent.ActuationSpec(amount=amount)) for i,name,amount in [(1,'move_forward',.25),(2,'turn_left',30.),(3,'turn_right',30.)]}
        sim=hs.Simulator(hs.Configuration(cfg,[a]))
        if not sim.pathfinder.load_nav_mesh(row['navmesh']):sim.close();raise ValueError('source navmesh failed to load')
        self.sim=RemoteSim(sim,row,self.url);self.scene=scene
        cam=row['camera'];self._geom=dict(width=cam['width'],height=cam['height'],hfov=cam['hfov_deg'],sensor_height=cam['camera_height_m'],forward_step=.25,turn_angle=30.)
        self._map_cache.clear();self._map_inv.clear()
    def _resolve_spawn(self,ep):pass
    def reset(self):
        obs=super().reset();self._initial_distance=self.distance();self._minimum_distance=self._initial_distance
        if not math.isfinite(self._initial_distance):raise ValueError('source start/goal disconnected on source navmesh')
        return obs
    def step(self,action,**kwargs):
        obs=super().step(action,**kwargs)
        self._minimum_distance=min(self._minimum_distance,self.distance())
        if self.sim._remote_defer_depth:return {'_render_deferred':True}
        return obs
    def distance(self):
        q=self._hs.ShortestPath();q.requested_start=self._position();q.requested_end=np.asarray(self.current_episode.record['goal_position'],dtype=np.float32)
        return float(q.geodesic_distance) if self.sim.pathfinder.find_path(q) else float('inf')
    def get_metrics(self):
        d=super().get_metrics();distance=self.distance();radius=self.current_episode.record['success_distance_m']
        success=self._stopped and distance<=radius;length=d['path_length_m'];initial=self._initial_distance
        d.update(distance_to_goal=distance,success=float(success),spl=float(success)*initial/max(initial,length,1e-8),oracle_success=float(self._minimum_distance<=radius),oracle_navigation_error=self._minimum_distance)
        return d
    def description_of(self,scene):return 'ObjectNav / source-selected goal viewpoint; not official category-wide scoring'
    def render_top_down(self, size=512, trail=True, show_goal=False):
        # Never annotate the cached raster or the model's independently rendered map.
        image = super().render_top_down(size=size, trail=trail, show_goal=False)
        if not show_goal:
            return image
        from PIL import Image, ImageDraw
        canvas = Image.fromarray(image).copy()
        _, to_px = self._top_down_base(size)
        x, y = to_px(self.current_episode.record['goal_position'])
        draw = ImageDraw.Draw(canvas)
        radius = 7
        draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                     fill=(255, 255, 255), outline=(40, 220, 80), width=3)
        return np.asarray(canvas)

    def title_of(self,scene):return scene
