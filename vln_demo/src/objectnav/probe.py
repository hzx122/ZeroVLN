"""Actual render/navmesh/motion/STOP acceptance; only the selected test endpoint."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from .env import ObjectNavEnv

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--url',required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    env=ObjectNavEnv(args.manifest,args.url);results=[]
    try:
        for ep in env.episodes:
            env.current_episode=ep;t=time.monotonic();obs=env.reset();row=ep.record
            Image.fromarray(obs['rgb']).save(args.output/(row['logical_id']+'-initial.png'))
            depth=obs['depth'];finite=depth[np.isfinite(depth)&(depth>0)]
            if len(finite)<100 or np.std(obs['rgb'])<5:raise ValueError('invalid RGB-D observation')
            init=env._position().copy();d=env.distance();times=[]
            for action in [2,3,1]:
                start=time.monotonic();obs=env.step(action);times.append(time.monotonic()-start)
            travelled=float(np.linalg.norm(env._position()-init))
            env.step(0);far=env.get_metrics()
            # Scoring check at source goal, separate from model-policy success.
            env.reset();env.sim.set_agent_state(row['goal_position']);env.step(0);goal=env.get_metrics()
            if goal['success']!=1:raise ValueError('STOP at source goal must succeed')
            result=dict(category=row['category'],episode_id=ep.episode_id,seconds=time.monotonic()-t,initial_distance=d,moved_m=travelled,action_seconds=times,depth_median=float(np.median(finite)),source_goal_stop=goal['success'],start_stop_success=far['success'],scoring=row['scoring'])
            results.append(result);print(json.dumps(result),flush=True)
    finally:env.close();(args.output/'probe.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
