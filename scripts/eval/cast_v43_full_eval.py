#!/usr/bin/env python3
"""Single-checkpoint held-out CAST pairs, actual vLLM query decode and head."""
import argparse
import json
import os
from pathlib import Path
import shutil
import numpy as np
import cast_v43_prefix_pilot as pilot
from common import atomic_json, checkpoint_identity, sha256_file
from cast_v43_vllm_eval import capture


def evaluate(data, output, checkpoint):
    output.mkdir(parents=True,exist_ok=True);(output/'logs').mkdir(exist_ok=True)
    identity=checkpoint_identity(checkpoint)
    source_sha=sha256_file(data/'eval_after/cases.json')
    if (output/'completed.json').exists():
        done=json.loads((output/'completed.json').read_text())
        if done['checkpoint']!=identity or done['cases_sha256']!=source_sha:raise ValueError('completed eval identity differs')
        return
    for name in ['accepted_pairs.json','manifest.jsonl','data_report.json']:
        shutil.copyfile(data/name,output/name)
    (output/'eval_after').mkdir(exist_ok=True)
    shutil.copyfile(data/'eval_after/cases.json',output/'eval_after/cases.json')
    atomic_json(output/'protocol.json',{'checkpoints':{'eval_after':str(checkpoint)},'supplemental':False,
        'port':28916,'tp':2,'devices':'0,1','cases_sha256':source_sha,
        'scope':'fixed tool instruction, real vLLM 16-query decode; static replay, not rendered closed-loop SR'})
    os.environ['ACTION_HEAD_ALLOW_FORCED_COMPLETION']='1'
    capture(output)
    pilot.infer(output,'eval_after',str(checkpoint),query_backend='vllm')
    score=json.loads((output/'eval_after/prefix_score.json').read_text())
    pairs=score['cases']
    if not pairs or any(p['split']!='validation' for p in pairs):raise ValueError('not a held-out paired test')
    for ci in range(len(pairs)):
        with np.load(output/'eval_after'/f'case_{ci}_queries.npz') as q:
            if not np.array_equal(q['original'],q['original_repeat']):raise ValueError('repeat query differs')
        with np.load(output/'eval_after'/f'case_{ci}_head.npz') as h:
            if not np.array_equal(h['original_actions'],h['original_repeat_actions']):raise ValueError('repeat head differs')
    scenes=sorted({p['scene'] for p in pairs})
    stats={}
    for suffix in ['', '_paraphrase']:
        scene_rates=[]
        for scene in scenes:
            bits=[v for p in pairs if p['scene']==scene for v in p['paired_prefix_gate'+suffix]]
            scene_rates.append(sum(bits)/len(bits))
        values=[v for p in pairs for side in ['left','right'] for v in p['conditions'][side+suffix]]
        stats['canonical' if not suffix else 'heldout_wording']={
            'scene_macro_pair_pass':sum(scene_rates)/len(scene_rates),
            'correct_direction_signs':sum(v['signed_prefix_y_m']>0 for v in values),'direction_draws':len(values)}
    atomic_json(output/'completed.json',{'ok':True,'checkpoint':identity,'cases_sha256':source_sha,
        'pairs':len(pairs),'scenes':len(scenes),'aggregate':score['aggregate']['validation'],'additional':stats,
        'paired_controls_verified':True,'scope':score['scope']})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args()
    if not a.output.resolve().is_relative_to(pilot.ART):p.error('output outside data root')
    evaluate(a.data,a.output,a.checkpoint)
