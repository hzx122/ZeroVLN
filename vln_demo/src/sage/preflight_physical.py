"""CPU admission audit using exactly the training/evaluation reset contract.

Does not resample, filter or rewrite the case manifest. Exit 2 if any original
case lacks unambiguous connected geometry. An imported seed is source evidence,
not a claim of source-mesh quality or successful navigation.
"""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import threading
import traceback

from .datasets import SageSet
from .goal_region import load_label_index
from .navmesh_support import digest
from .physical_navigation import DEFAULT_ROOT, SCHEMA, identity, load_navigation, resolve_task


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--split',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--radius',type=float,default=.07)
    p.add_argument('--import-index',type=Path)
    p.add_argument('--add-structure',action='store_true',help='Freeze USD architectural support when importing into a fresh geometry root')
    p.add_argument('--build-missing',action='store_true',help='Build missing authored USD/Recast seeds on CPU, never change cases')
    args=p.parse_args()
    allowed=Path('/data/zerovln/artifacts')
    output=args.output.resolve()
    if not output.is_relative_to(allowed) or output.exists():
        p.error('output must be a fresh directory under the project data artifacts root')
    output.mkdir(parents=True)
    def heartbeat():
        while True:
            value=dict(pid=os.getpid(),updated_epoch=time.time(),gpu=False)
            temp=output/'heartbeat.tmp'
            temp.write_text(json.dumps(value)+'\n')
            temp.replace(output/'heartbeat.json')
            time.sleep(20)
    threading.Thread(target=heartbeat,daemon=True).start()
    raw=args.cases.read_text()
    rows=json.loads(raw) if args.cases.suffix=='.json' else [json.loads(s) for s in raw.splitlines() if s.strip()]
    if isinstance(rows,dict): rows=rows['cases']
    groups=defaultdict(list)
    for row in rows: groups[row['scene']].append(row)
    seeds=json.loads(args.import_index.read_text())['scenes'] if args.import_index else {}
    root=Path(os.environ.get('SAGE_NAVMESH_ROOT',DEFAULT_ROOT)).resolve()
    if seeds and not root.is_relative_to(allowed):
        p.error('import destination must be under project data artifacts')
    started=time.time()
    code_identity=identity()
    ds=SageSet(args.split)
    counts=Counter(); reasons=Counter()
    with (output/'cases.jsonl').open('w') as stream:
        for scene,cases in groups.items():
            try:
                if scene in seeds:
                    source=Path(seeds[scene]['manifest'])
                    payload=source.read_bytes(); spec=json.loads(payload)
                    if spec.get('collision_sidedness') != 'authored':
                        raise ValueError('diagnostic winding seed cannot be imported as the default')
                    for receipt in seeds[scene].get('receipts',[]):
                        if Path(receipt['path'])==source and digest(source)!=receipt['sha256']:
                            raise ValueError('source manifest receipt changed')
                    for kind in ('usd','navmesh'):
                        if digest(spec[kind+'_path'])!=spec[kind+'_sha256']:
                            raise ValueError(kind+' source changed')
                    dest=root/scene.removeprefix('sage-')/'manifest.json'
                    if args.add_structure:
                        if dest.exists():
                            raise ValueError('structural enrichment requires a fresh geometry destination')
                        from .build_navmesh import write_structure
                        structural=output/'structure'
                        structural.mkdir(exist_ok=True)
                        spec.update(write_structure(spec['usd_path'],structural/(scene+'.npy')))
                        payload=(json.dumps(spec,indent=2)+'\n').encode()
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    if dest.exists() and dest.read_bytes()!=payload:
                        raise ValueError('refuse replacing an existing geometry manifest')
                    if not dest.exists():
                        with dest.open('xb') as f: f.write(payload)
                dest=root/scene.removeprefix('sage-')/'manifest.json'
                if args.build_missing and not dest.exists():
                    if not root.is_relative_to(allowed):
                        raise ValueError('build destination must be under project data artifacts')
                    seed=output/'built'/scene
                    command=[sys.executable,'-B','-m','sage.build_navmesh',
                        '--scene',scene.removeprefix('sage-'),'--radius',str(args.radius),
                        '--output',str(seed)]
                    with (output/(scene+'.build.log')).open('w') as log:
                        child=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                    if child.returncode:
                        raise RuntimeError(f'CPU mesh build failed ({child.returncode}); see {scene}.build.log')
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    with dest.open('xb') as f: f.write((seed/'manifest.json').read_bytes())
                nav=load_navigation(scene,args.radius)
                from .source_objects import load_source_label_index
                labels=load_source_label_index(ds.scene_dir(scene))
                tasks={t.task_id:t for t in ds.tasks(scene,0)}
                scene_error=None
            except Exception as exc:
                scene_error=f'{type(exc).__name__}: {exc}'
            for row in cases:
                result=dict(scene=scene,episode_id=row['episode_id'],valid=False)
                try:
                    if scene_error: raise ValueError(scene_error)
                    audit,target,distance=resolve_task(nav,tasks[row['episode_id']],labels)
                    result.update(valid=True,start_audit=audit,target_id=target.target_id,
                        target_contract_sha256=target.hash,shortest_path_m=distance,
                        target_support_height_m=[min(p[2] for p in target.patches),max(p[2] for p in target.patches)])
                except Exception as exc:
                    result['error']=f'{type(exc).__name__}: {exc}'
                    try:
                        reason=json.loads(str(exc)).get('reason',type(exc).__name__)
                    except (ValueError, AttributeError):
                        message=str(exc)
                        reason=('target_support_ambiguous_or_absent' if 'support layers' in message else
                                'target_disconnected' if 'no connected 3-D route' in message else
                                'missing_navmesh' if 'missing 3-D geometry' in message else type(exc).__name__)
                    result['reason']=reason
                    reasons[reason]+=1
                counts['valid' if result['valid'] else 'invalid']+=1
                stream.write(json.dumps(result)+'\n'); stream.flush()
            print(json.dumps(dict(scene=scene,counts=dict(counts))),flush=True)
            status=dict(pid=os.getpid(),last_scene=scene,counts=dict(counts),updated_epoch=time.time())
            (output/'status.json').write_text(json.dumps(status)+'\n')
    report=dict(schema=SCHEMA,cases=len(rows),scenes=len(groups),counts=dict(counts),
        cases_path=str(args.cases.resolve()),cases_sha256=digest(args.cases),
        navmesh_root=str(root),radius=args.radius,split=args.split,errors=dict(reasons),
        original_population_preserved=True,physical_rollout_validated=False,
        scoring_identity=code_identity,finished_scoring_identity=identity(),
        started_epoch=started,finished_epoch=time.time())
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return 2 if counts['invalid'] else 0


if __name__=='__main__':
    exit_code=1
    try:
        exit_code=main()
    except Exception:
        traceback.print_exc()
    finally:
        if '--output' in sys.argv:
            output=Path(sys.argv[sys.argv.index('--output')+1])
            # Only our fresh output may receive a completion receipt.
            heartbeat_path=output/'heartbeat.json'
            if heartbeat_path.is_file() and json.loads(heartbeat_path.read_text()).get('pid')==os.getpid():
                (output/'completion.json').write_text(json.dumps(dict(
                    pid=os.getpid(),exit_code=exit_code,finished_epoch=time.time()))+'\n')
    raise SystemExit(exit_code)
