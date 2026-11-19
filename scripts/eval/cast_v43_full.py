#!/usr/bin/env python3
"""Full v4.3 CAST directional augmentation, using the verified pilot adapter.

All original golden windows are retained. CAST adds only same-state first
windows; unverified landmark-relation language stays in the teacher audit.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import cast_v43_prefix_pilot as pilot
from cast_v43_prefix_pilot import ART, REPO, PRESCREEN, CAST, atomic_json, sha256_file

ROOT = ART/'action-head-v43-cast-expanded-full-20260915-r1'
TRAIN_SOURCE = ART/'action-head-sft-data-v43-official-gvln-full-20260911-r2/manifest.jsonl'
VAL_SOURCE = ART/'action-head-sft-data-v43-official-gvln-test100-20260911-r1/manifest.jsonl'
TRAIN_SHA = '961eb93747fca96fb4c4d9078c0f94b0f159830e52ab7b37b481169a7b8fdef7'
VAL_SHA = '0be180ac6cfedd6abbac24f13fe5a84ef61b7e72692d9bb35c8b7436719ac645'
TRAIN_SPLIT = 'official_gvln_action_head_p0_v3full827train'
VAL_SPLIT = 'official_gvln_action_head_p0_v3test100'
TRAIN_TEMPLATES = ['Move diagonally {side} across the open floor',
    'Advance diagonally {side} over the clear floor',
    'Proceed forward on a diagonal to the {side} across the open floor']
EVAL_TEMPLATE = 'Take a diagonal path to the {side} across the open floor'
SCOPE = ('same-state immediate diagonal direction; first fast window only; CAST CF prompt/schema '
         'and RGB self/cross audit plus SAGE static replay; no learned atomic policy; '
         'complex landmark relations audit-only; validation from original test100, excluded from full training')


def load(path): return json.loads(Path(path).read_text())


def opposite(candidate):
    sides = {('left' if y>0 else 'right') for b in candidate['branch_pair']
        if abs(y:=pilot.local_xy([b['prefix_xy']],candidate['source_pose_xyyaw'])[0,1])>=pilot.TEACHER_MIN_SIDE_M
        and b['path_length_m']>=1.6}
    return sides == {'left','right'}


def prepare(root):
    import cast_full_anchor_prescreen as screen
    if (root/'prepared.json').exists():
        print('Full input index already frozen; resuming.', flush=True); return
    val_screen=root/'test100_prescreen'
    for sub in ['inputs','scenes']:(val_screen/sub).mkdir(parents=True,exist_ok=True)
    val_index=screen.compact_index(VAL_SOURCE,val_screen,split='validation')
    train_index=load(PRESCREEN/'input_index.json')
    if set(train_index['scenes']) & set(val_index['scenes']):raise ValueError('original train/test scenes overlap')
    if val_index['source_sha256']!=VAL_SHA:raise ValueError('test100 source hash differs')
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(screen.process_scene,s,val_screen,VAL_SPLIT)
                 for s in val_index['scenes'] if not (val_screen/'scenes'/f'{s}.json').exists()]
        for f in futures:f.result()
    counts={};units=[]
    for split,source,screen_root,index,expected in [('train',TRAIN_SOURCE,PRESCREEN,train_index,TRAIN_SHA),
            ('validation',VAL_SOURCE,val_screen,val_index,VAL_SHA)]:
        by_line={};ngeometry=0
        for scene in index['scenes']:
            result=load(screen_root/'scenes'/f'{scene}.json')
            for c in result['candidates']:
                ngeometry+=1
                if opposite(c):by_line[c['source_manifest_line']]=c
        digest=hashlib.sha256();seen=set();buffers=[[] for _ in range(16)];numbers=[0]*16
        def flush(shard):
            if not buffers[shard]:return
            unit=f'{split}-{shard:02d}-{numbers[shard]:05d}'
            dest=root/'inputs'/f'{unit}.json';dest.parent.mkdir(exist_ok=True)
            atomic_json(dest,{'split':split,'candidates':buffers[shard]})
            units.append({'unit':unit,'shard':shard,'split':split,'input':str(dest),'sha256':sha256_file(dest),'candidates':len(buffers[shard])})
            numbers[shard]+=1;buffers[shard]=[]
        with source.open('rb') as stream:
            for lineno,line in enumerate(stream,1):
                digest.update(line)
                if lineno not in by_line:continue
                c=by_line[lineno];row=json.loads(line)
                if row['slow_anchor_id']!=c['slow_anchor_id'] or row['split']!=split or row['fast_window_index']!=0:raise ValueError('index mismatch')
                if row['slow_anchor_id'] in seen:raise ValueError('duplicate source anchor')
                seen.add(row['slow_anchor_id'])
                shard=int(hashlib.sha256(row['source_scene'].encode()).hexdigest(),16)%16
                buffers[shard].append({'row':row,'candidate':c,'source_index':lineno})
                if len(buffers[shard])==32:flush(shard)
        for shard in range(16):flush(shard)
        if digest.hexdigest()!=expected:raise ValueError('full source changed: '+str(source))
        counts[split]={'source':str(source),'sha256':digest.hexdigest(),'windows':index['windows'],
            'anchors':index['anchors'],'scenes':len(index['scenes']),'geometric_candidates':ngeometry,'opposite_candidates':len(seen)}
        print(json.dumps({'event':'indexed','split':split,**counts[split]}),flush=True)
    assets=[CAST/'cast/data'/sub/name for sub,names in [('prompts',['cf_step_prompt.txt','filter_step_prompt.txt']),
        ('schemas',['cf_step_schema.json','filter_step_schema.json'])] for name in names]
    atomic_json(root/'prepared.json',{'schema':'cast_v43_full_v1','counts':counts,'units':units,'shards':16,
        'train_templates':TRAIN_TEMPLATES,'heldout_template':EVAL_TEMPLATE,'scope':SCOPE,
        'cast_commit':'ec7a214e76167e0f844800ea91d6664863a3d9b1',
        'cast_assets_sha256':{str(p):sha256_file(p) for p in assets}})


def finish_unit(dest, split, sage_split):
    from action_head import protocol_v4_3 as protocol
    pilot.finalize(dest, reject_invalid_pairs=True)
    original=[json.loads(line) for line in (dest/'manifest.jsonl').open()]
    rows=[]
    for row in original:
        if 'cast_provenance' not in row:continue # Full golden already contains these originals.
        for ti,template in enumerate(TRAIN_TEMPLATES if split=='train' else TRAIN_TEMPLATES[:1]):
            value=copy.deepcopy(row);value['slow_anchor_id']+=f':wording-{ti}'
            value['contract']['plan'][0]=template.format(side=value['cast_provenance']['side'])
            value['slow_completion']=protocol.format_tool_call('run_action_head',value['contract'])
            value['cast_provenance']['wording_family']=ti
            protocol.assert_protocol_identity(value);rows.append(value)
    pilot.write_jsonl(dest/'manifest.jsonl',rows)
    for phase in ['eval_before','eval_after']:
        document=load(dest/phase/'cases.json');document['scope']=SCOPE
        for case in document['cases']:
            for side in ['left','right']:
                contract=copy.deepcopy(case['row']['contract'])
                contract['plan']=[EVAL_TEMPLATE.format(side=side),'continue toward '+contract['target']]
                case['variants'][side+'_paraphrase']=protocol.format_tool_call('run_action_head',contract)
        atomic_json(dest/phase/'cases.json',document)
    report=load(dest/'data_report.json')
    report.update(rows=len(rows),manifest_sha256=sha256_file(dest/'manifest.jsonl'),sage_split=sage_split,
        evaluation_scope=SCOPE,original_to_augmented='CAST branches only; original golden retained once in full merge',
        paraphrases_in_training=split=='train',training_templates=TRAIN_TEMPLATES,heldout_template=EVAL_TEMPLATE)
    atomic_json(dest/'data_report.json',report)
    atomic_json(dest/'completed.json',{'ok':True,'pairs':report['pairs'],'rows':len(rows),
        'manifest_sha256':report['manifest_sha256'],'split':split,'quality_policy_version':2,
        'geometry_rejected_pairs':report.get('geometry_rejected_pairs',0)})


def worker(root, shard, limit_units=0, tp=4, devices=None, port=28916):
    import fcntl
    import signal
    def terminate(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, terminate)
    from cast_case_feasibility_audit import SageSet, OccMap
    from action_head_rl.vllm_service import VllmService
    plan=load(root/'prepared.json')
    units=sorted([u for u in plan['units'] if u['shard']==shard], key=lambda u:(u['split']!='validation',u['unit']))
    if limit_units:units=units[:limit_units]
    lock_path=root/f'shard_{shard:02d}.lock'
    with lock_path.open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        query=ART/'action-head-v43-official-gvln-d16-allrandomquery-randomhead-10ep-4node-gb64-20260911-r3/initialization/d16/query/dualvln_query_embeddings.pt'
        service=VllmService(repo=str(REPO),model=pilot.MODEL,query_checkpoint=str(query),
            query_cache=str(root/f'teacher_artifacts/shard_{shard:02d}'),log_file=str(root/f'logs/teacher_{shard:02d}.log'),
            port=port,capture_full_routes=False,tensor_parallel_size=tp,
            cuda_visible_devices=devices or ','.join(map(str,range(tp))),max_num_seqs=4,max_num_batched_tokens=8192,gpu_memory_utilization=.6)
        datasets={s:SageSet(split=n) for s,n in [('train',TRAIN_SPLIT),('validation',VAL_SPLIT)]}
        occs={};started=False
        try:
            for unit in units:
                dest=root/'units'/unit['unit'];dest.mkdir(parents=True,exist_ok=True)
                if (dest/'completed.json').exists():
                    if sha256_file(dest/'manifest.jsonl')!=load(dest/'completed.json')['manifest_sha256']:raise ValueError('completed unit changed')
                    continue
                if sha256_file(unit['input'])!=unit['sha256']:raise ValueError('input changed')
                source=load(unit['input']);split=source['split'];pairs=[];precheck_rejected=[]
                for entry in source['candidates']:
                    row=entry['row'];scene=row['source_scene']
                    if scene not in occs:
                        if len(occs)>=4:occs.clear()
                        occs[scene]=OccMap(datasets[split].scene_dir(scene),collision_mode='usd_rooms',furniture_mode='none')
                    pair=pilot.prepare_geometric_pair(row,entry['candidate'],entry['source_index'],datasets[split],occs[scene],strict_training_replay=True)
                    if pair is not None:pairs.append(pair)
                    else:precheck_rejected.append({'source_anchor_id':row['slow_anchor_id'],
                        'reason':'opposite_pair_training_geometry_precheck_failed'})
                sage_split=TRAIN_SPLIT if split=='train' else VAL_SPLIT
                atomic_json(dest/'prepared.json',{'pairs':pairs,'scope':SCOPE,'fixed_split':split,'sage_split':sage_split,
                    'minimum_pairs':0,'source_unit':unit,'cast_assets_sha256':plan['cast_assets_sha256'],
                    'quality_policy_version':2,'geometry_precheck_rejected':precheck_rejected})
                if not started:service.start();started=True
                pilot.generate(dest,teacher_url=f'http://127.0.0.1:{port}/v1/chat/completions')
                finish_unit(dest,split,sage_split)
                atomic_json(root/f'shard_{shard:02d}_progress.json',{'last_unit':unit['unit'],
                    'done':sum((root/'units'/u['unit']/'completed.json').exists() for u in units),'total':len(units)})
            if not limit_units:atomic_json(root/f'shard_{shard:02d}_completed.json',{'ok':True,'units':len(units)})
        finally:
            service.stop()
            rpc=os.environ.get('VLLM_RPC_BASE_PATH','')
            if rpc.startswith('@castfull-'):
                shutil.rmtree(REPO/rpc,ignore_errors=True)


def merge(root):
    from action_head import protocol_v4_3 as protocol
    plan=load(root/'prepared.json')
    if not all((root/f'shard_{s:02d}_completed.json').exists() for s in range(16)):raise RuntimeError('all 16 shards must finish')
    train=root/'mixed';validation=root/'cast_test100';train.mkdir(exist_ok=True);validation.mkdir(exist_ok=True)
    seen=set();scenes={'train':set(),'validation':set()};counts=Counter();pairs=[];cases=[]
    source_digest=hashlib.sha256();mixed_digest=hashlib.sha256()
    staging=train/'manifest.jsonl.staging'
    with staging.open('wb') as out:
        with TRAIN_SOURCE.open('rb') as source:
            for line in source:
                source_digest.update(line);row=json.loads(line);protocol.assert_protocol_identity(row)
                if row['split']!='train':raise ValueError('golden split differs')
                seen.add(row['slow_anchor_id']);scenes['train'].add(row['source_scene'])
                out.write(line);mixed_digest.update(line);counts['golden_windows']+=1
        if source_digest.hexdigest()!=TRAIN_SHA:raise ValueError('golden changed')
        counts['golden_anchors']=len(seen)
        with (validation/'manifest.jsonl').open('wb') as val:
            for unit in plan['units']:
                dest=root/'units'/unit['unit'];marker=load(dest/'completed.json')
                if not marker['ok'] or sha256_file(dest/'manifest.jsonl')!=marker['manifest_sha256']:raise ValueError('unit corrupted')
                counts[unit['split']+'_pairs']+=marker['pairs']
                for line in (dest/'manifest.jsonl').open('rb'):
                    row=json.loads(line);protocol.assert_protocol_identity(row)
                    if row['split']!=unit['split'] or 'cast_provenance' not in row:raise ValueError('invalid augmentation row')
                    if row['slow_anchor_id'] in seen:raise ValueError('duplicate CAST anchor')
                    seen.add(row['slow_anchor_id']);scenes[row['split']].add(row['source_scene'])
                    counts[row['split']+'_cast_anchors']+=1
                    if row['split']=='train':out.write(line);mixed_digest.update(line)
                    else:val.write(line)
                if unit['split']=='validation':
                    pairs.extend(load(dest/'accepted_pairs.json'));cases.extend(load(dest/'eval_after/cases.json')['cases'])
    if len(pairs)!=counts['validation_pairs'] or len(cases)!=len(pairs) or {c['pair_id'] for c in cases}!={p['pair_id'] for p in pairs}:raise ValueError('test pair/case cardinality mismatch')
    if scenes['train'] & scenes['validation']:raise ValueError('CAST test scene leakage')
    if counts['train_pairs']==0 or counts['validation_pairs']==0:raise ValueError('no accepted full train/test pairs')
    staging.replace(train/'manifest.jsonl')
    atomic_json(validation/'accepted_pairs.json',pairs)
    for phase in ['eval_before','eval_after']:
        (validation/phase).mkdir(exist_ok=True);atomic_json(validation/phase/'cases.json',{'cases':cases,'scope':SCOPE})
    atomic_json(validation/'data_report.json',{'ok':True,'finalized':True,'pairs':len(pairs),'sage_split':VAL_SPLIT,
        'manifest_sha256':sha256_file(validation/'manifest.jsonl'),'evaluation_scope':SCOPE})
    n=counts['golden_anchors']+counts['train_cast_anchors'];steps=math.ceil(n/64)*10
    config=load(REPO/'configs/action_head/v43_official_gvln_d16_allrandomquery_10ep_4node_20260911.json')
    expr=root.name
    config['shared'].update(manifest=str(train/'manifest.jsonl'),output_root=str(root/'checkpoints'))
    config['initialization']['output_root']=str(root/'initialization')
    config['scales']['D16'].update(max_steps=steps,run_name=expr,warm_start_checkpoint='',step_offset=0)
    atomic_json(root/'training_config.json',config)
    atomic_json(root/'training_ready.json',{'ok':True,'counts':dict(counts),'train_anchors':n,
        'train_windows':counts['golden_windows']+counts['train_cast_anchors'],'steps_per_epoch':math.ceil(n/64),'epochs':10,
        'max_steps':steps,'cast_anchor_fraction':counts['train_cast_anchors']/n,'manifest_sha256':mixed_digest.hexdigest(),
        'config_sha256':sha256_file(root/'training_config.json'),'scene_overlap':0,'scope':SCOPE})
    print(json.dumps(load(root/'training_ready.json')),flush=True)


def repair_materialized_units(root):
    """Recover stopped shards from complete teacher caches without any GPU call."""
    import fcntl
    recovered=[]
    for unit in load(root/'prepared.json')['units']:
        dest=root/'units'/unit['unit']
        if (dest/'completed.json').exists() or not (dest/'data_report.json').exists():continue
        with (root/f"shard_{unit['shard']:02d}.lock").open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            backup=root/'recovery_v2/before'/unit['unit']
            if not backup.exists():
                backup.mkdir(parents=True)
                files=['prepared.json','accepted_pairs.json','manifest.jsonl','data_report.json',
                       'language_audit.json','eval_before/cases.json','eval_after/cases.json']
                for relative in files:
                    source=dest/relative;target=backup/relative
                    if source.exists():
                        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
                        if sha256_file(source)!=sha256_file(target):raise ValueError('backup hash mismatch')
            pilot.generate(dest,cache_only=True)
            finish_unit(dest,unit['split'],TRAIN_SPLIT if unit['split']=='train' else VAL_SPLIT)
            recovered.append({'unit':unit['unit'],**load(dest/'completed.json')})
            atomic_json(root/'recovery_v2/repair_report.json',{'recovered':recovered,'teacher_requests':0})
            print(json.dumps(recovered[-1]),flush=True)
    print(json.dumps({'repaired_units':len(recovered),'rejected_pairs':sum(v['geometry_rejected_pairs'] for v in recovered)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','worker','merge','repair'])
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--shard',type=int,default=0)
    p.add_argument('--limit-units',type=int,default=0);p.add_argument('--tp',type=int,default=4)
    p.add_argument('--devices');p.add_argument('--port',type=int,default=28916);a=p.parse_args()
    if not a.root.resolve().is_relative_to(ART):p.error('output outside data root')
    a.root.mkdir(parents=True,exist_ok=True)
    if a.mode=='worker':worker(a.root,a.shard,a.limit_units,a.tp,a.devices,a.port)
    else:{'prepare':prepare,'merge':merge,'repair':repair_materialized_units}[a.mode](a.root)
