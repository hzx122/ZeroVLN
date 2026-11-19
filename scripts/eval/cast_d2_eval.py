#!/usr/bin/env python3
"""D2 same-state route contrast, reusing real vLLM capture and FM inference."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import cast_v43_prefix_pilot as pilot
from common import atomic_json, checkpoint_identity, sha256_file
from cast_v43_vllm_eval import capture

GATE = dict(own_ade_max_m=.5, minimum_margin_fraction_of_expert_separation=.1,
            both_branches_required=True, static_replay_step_m=.05, robot_radius_m=.07,
            horizon='common valid portion of the first 16-step fast window', draws=4)


def contrast(prediction, targets, branch, valid):
    points=np.asarray(prediction).cumsum(0)[:valid,:2]
    own=np.asarray(targets[branch]).cumsum(0)[:valid,:2]
    other=np.asarray(targets[1-branch]).cumsum(0)[:valid,:2]
    separation=float(np.linalg.norm(own-other,axis=1).mean())
    if separation<=1e-6:raise ValueError('no supervised contrast')
    ade=float(np.linalg.norm(points-own,axis=1).mean())
    wrong=float(np.linalg.norm(points-other,axis=1).mean())
    margin=wrong-ade
    return dict(own_ade_m=ade,swapped_ade_m=wrong,expert_separation_m=separation,
        normalized_matching_margin=margin/separation,
        route_match=bool(ade<=GATE['own_ade_max_m'] and
            margin>=GATE['minimum_margin_fraction_of_expert_separation']*separation))


def evaluate(data, output, checkpoint, wait_seconds=0):
    deadline=time.monotonic()+wait_seconds
    while not (data/'data_ready.json').exists():
        if time.monotonic()>=deadline:raise RuntimeError('D2 validation not frozen: '+str(data))
        time.sleep(30)
    ready=json.loads((data/'data_ready.json').read_text())
    source=data/'eval_after/cases.json';digest=sha256_file(source)
    if ready['split']!='validation' or digest!=ready['cases_sha256']:raise ValueError('D2 evaluation identity mismatch')
    identity=checkpoint_identity(checkpoint)
    if (output/'completed.json').exists():
        done=json.loads((output/'completed.json').read_text())
        if done['checkpoint']!=identity or done['cases_sha256']!=digest or done['gate']!=GATE:
            raise ValueError('completed D2 evaluation changed')
        return
    (output/'logs').mkdir(parents=True,exist_ok=True);(output/'eval_after').mkdir(exist_ok=True)
    shutil.copyfile(source,output/'eval_after/cases.json')
    atomic_json(output/'protocol.json',dict(checkpoints={'eval_after':str(checkpoint)},
        supplemental=False,port=28916,tp=2,devices='0,1',cases_sha256=digest,gate=GATE))
    os.environ['ACTION_HEAD_ALLOW_FORCED_COMPLETION']='1'
    capture(output)
    from sft.fla_l2norm_dynamic_nb import maybe_install_l2norm_dynamic_nb_patch
    maybe_install_l2norm_dynamic_nb_patch()
    import diagnose_v43_instruction_control as diagnostic
    diagnostic.CHECKPOINT=Path(checkpoint)
    diagnostic.run(output/'eval_after',query_backend='vllm')
    from cast_case_feasibility_audit import SageSet,OccMap,_execution_replay_free
    dataset=SageSet('official_gvln_action_head_p0_v3test100')
    cases=json.loads(source.read_text())['cases'];results=[];occs={}
    for ci,case in enumerate(cases):
        pair=case['pair'];row=case['row'];scene=row['source_scene']
        if row['split']!='validation':raise ValueError('training case entered evaluation')
        if scene not in occs:
            if len(occs)>=4:occs.pop(next(iter(occs)))
            occs[scene]=OccMap(dataset.scene_dir(scene),collision_mode='usd_rooms',furniture_mode='none')
        targets=[np.asarray(b['target_real']) for b in pair['branches']]
        masks=[]
        for branch in pair['branches']:
            with np.load(branch['sample']) as sample:masks.append(int(sample['action_mask'].sum()))
        valid=min(masks)
        # The expert itself must pass the matching gate, including pairs that separate late.
        if not all(contrast(targets[i],targets,i,valid)['route_match'] for i in (0,1)):
            raise ValueError('expert cannot pass the predeclared route gate')
        with np.load(output/'eval_after'/f'case_{ci}_queries.npz') as q:
            if not np.array_equal(q['original'],q['original_repeat']):raise ValueError('query repeat differs')
        with np.load(output/'eval_after'/f'case_{ci}_head.npz') as head:
            if not np.array_equal(head['original_actions'],head['original_repeat_actions']):raise ValueError('head repeat differs')
            poses=row['source_pose_xyyaw'];c,s=math.cos(poses[2]),math.sin(poses[2]);draws=[]
            for draw in range(4):
                branches=[]
                for bi,name in enumerate(('left','right')):
                    actions=head[name+'_actions'][draw]
                    values=contrast(actions,targets,bi,valid)
                    points=actions.cumsum(0)[:valid,:2]
                    world=points@np.asarray([[c,s],[-s,c]])+np.asarray(poses[:2])
                    free,replay=_execution_replay_free(occs[scene],np.vstack([poses[:2],world]),
                        step_m=.05,radius_m=.07)
                    values['static_replay_free']=bool(free and replay['collisions']==0)
                    branches.append(values)
                draws.append(dict(branches=branches,paired_pass=all(
                    b['route_match'] and b['static_replay_free'] for b in branches)))
            results.append(dict(pair_id=pair['pair_id'],kind=pair['kind'],scene=scene,draws=draws))
    aggregate={}
    for kind in ('all','object_target','object_side'):
        subset=[r for r in results if kind=='all' or r['kind']==kind]
        draws=[d for r in subset for d in r['draws']]
        branches=[b for d in draws for b in d['branches']]
        scene_rates=[np.mean([d['paired_pass'] for r in subset if r['scene']==scene for d in r['draws']])
                     for scene in sorted({r['scene'] for r in subset})]
        aggregate[kind]=dict(pairs=len(subset),draws=len(draws),paired_passed=sum(d['paired_pass'] for d in draws),
            scene_macro_pair_pass=float(np.mean(scene_rates)) if scene_rates else None,
            mean_own_ade_m=float(np.mean([b['own_ade_m'] for b in branches])) if branches else None,
            mean_matching_margin=float(np.mean([b['normalized_matching_margin'] for b in branches])) if branches else None)
    atomic_json(output/'scores.json',dict(cases=results,gate=GATE,aggregate=aggregate))
    atomic_json(output/'completed.json',dict(ok=True,checkpoint=identity,cases_sha256=digest,
        gate=GATE,aggregate=aggregate,paired_controls_verified=True,
        scope='held-out scenes; first-window target/relation route following, not semantic goal completion or closed-loop SR'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--wait-seconds',type=int,default=0);a=p.parse_args()
    if not a.output.resolve().is_relative_to(pilot.ART):p.error('output outside data root')
    evaluate(a.data,a.output,a.checkpoint,a.wait_seconds)
