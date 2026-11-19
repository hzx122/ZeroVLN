"""Stateless RGB-D service for registered ObjectNav assets; GPU GS, CPU mesh GL."""
import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from .catalog_io import read_catalog
import os
from pathlib import Path
import socket
import threading
import time
import numpy as np


def metric_depth(depth, near=.01, far=1000.):
    depth=np.asarray(depth,dtype=np.float32)
    out=near*far/np.maximum(far-depth*(far-near),1e-8)
    out[depth>=1.-1e-7]=np.inf
    return out

class Renderer:
    def __init__(self, manifest, capacity=2):
        self.rows={r['logical_id']:r for r in read_catalog(manifest)}
        self.cache=OrderedDict();self.capacity=capacity;self.requests=0
    def close(self):
        for sim, helper in self.cache.values(): sim.close()
        self.cache.clear()
    def render(self, payload):
        import habitat_sim as hs
        import magnum as mn
        from habitat_sim.utils.common import quat_from_coeffs
        row=self.rows[payload['asset_id']];cam=row['camera']
        position=np.asarray(payload['position'],dtype=np.float32);q=np.asarray(payload['rotation'],dtype=float)
        if position.shape!=(3,) or q.shape!=(4,) or not np.isfinite(position).all() or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-3:
            raise ValueError('invalid pose (rotation must be xyzw unit quaternion)')
        key=(row['asset_path'], json.dumps(cam,sort_keys=True))
        if key not in self.cache:
            if len(self.cache)>=self.capacity:
                _,(old,helper)=self.cache.popitem(last=False);old.close();del helper
            cfg=hs.SimulatorConfiguration();cfg.scene_id='NONE' if row['asset_type']=='gs' else row['asset_path']
            cfg.gpu_device_id=-1;cfg.enable_physics=False
            if hasattr(cfg,'gaussian_auto_play'):cfg.gaussian_auto_play=False
            specs=[]
            for uuid,kind in [('rgb',hs.SensorType.COLOR),('depth',hs.SensorType.DEPTH)]:
                s=hs.CameraSensorSpec();s.uuid=uuid;s.sensor_type=kind
                s.resolution=[cam['height'],cam['width']];s.hfov=cam['hfov_deg'];s.position=mn.Vector3(0.,cam['camera_height_m'],0.)
                s.near=.01;s.far=1000.;specs.append(s)
            a=hs.agent.AgentConfiguration();a.sensor_specifications=specs;a.action_space={}
            sim=hs.Simulator(hs.Configuration(cfg,[a]));helper=None
            try:
                if row['asset_type']=='gs':
                    helper=hs.RenderInstanceHelper(sim,use_xyzw_orientations=False)
                    helper.add_instance(asset_filepath=row['asset_path'],semantic_id=0,scale=mn.Vector3(1.))
                    helper.set_world_poses(np.zeros((1,3),dtype=np.float32),np.array([[1,0,0,0]],dtype=np.float32))
                self.cache[key]=(sim,helper)
            except BaseException:sim.close();raise
        sim,_=self.cache[key];self.cache.move_to_end(key)
        state=sim.get_agent(0).get_state();state.position=position;state.rotation=quat_from_coeffs(q)
        sim.get_agent(0).set_state(state,reset_sensors=True)
        obs=sim.get_sensor_observations();depth=np.asarray(obs['depth'],dtype=np.float32)
        if row['asset_type']=='gs':depth=metric_depth(depth)
        self.requests+=1
        return dict(rgb=np.asarray(obs['rgb'])[...,:3].copy(),depth=depth.copy())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--port',type=int,default=7041);ap.add_argument('--cache-size',type=int,default=2)
    args=ap.parse_args()
    if args.cache_size<1:ap.error('cache-size must be positive')
    renderer=Renderer(args.manifest,args.cache_size);pool=ThreadPoolExecutor(1);admission=threading.BoundedSemaphore(32)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path!='/health':self.send_error(404);return
            body=json.dumps(dict(ok=True,protocol=1,stateless=True,hostname=socket.gethostname(),backend='objectnav',job_id=socket.gethostname().split('-master-')[0] if '-master-' in socket.gethostname() else None,assets=len(renderer.rows),cached_scenes=len(renderer.cache),requests=renderer.requests,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),manifest=str(Path(args.manifest).resolve()))).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(body)
        def do_POST(self):
            if self.path!='/render':self.send_error(404);return
            if not admission.acquire(blocking=False):self.send_error(503,'render queue full');return
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<16384:raise ValueError('invalid body size')
                payload=json.loads(self.rfile.read(n));started=time.monotonic()
                obs=pool.submit(renderer.render,payload).result()
                out=io.BytesIO();np.savez_compressed(out,request_id=payload['request_id'],**obs)
                body=out.getvalue();self.send_response(200);self.send_header('Content-Type','application/octet-stream');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
                print(json.dumps(dict(event='render',asset_id=payload['asset_id'],seconds=round(time.monotonic()-started,3))),flush=True)
            except Exception as exc:
                self.send_error(400,str(exc));print(repr(exc),flush=True)
            finally:admission.release()
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('0.0.0.0',args.port),Handler)
    try:server.serve_forever()
    finally:pool.submit(renderer.close).result();pool.shutdown();server.server_close()
if __name__=='__main__':main()
