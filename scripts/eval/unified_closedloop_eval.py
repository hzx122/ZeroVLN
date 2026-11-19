"""The public entry for full, RL-companion and open-loop navigation evaluation.

Purpose is provenance only. All modes use the same suite, rollout, executor,
scorers and four-attempt pairing; selection and checkpoint weights may differ.
"""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'scripts/train'))
from rule_teacher_training_handoff import load, save, sha
import action_head_benchmark_suite as suite
from closedloop_evaluation_contract import (VERSION, V8_VERSION, identity, selection_config,
    verify_normalized, PROTOCOL, contract_for, protocol, LEGACY_CONTINUATION)

PURPOSES = ('full', 'rl-companion', 'openloop-companion')
CORE = ('scripts/eval/unified_closedloop_eval.py',
    'scripts/eval/unified_companion_queue.py',
    'scripts/eval/rule_teacher_openloop_companion.py',
    'scripts/eval/rule_teacher_companion_repair.py',
    'scripts/eval/start_unified_companion_resident.py',
    'scripts/train/supervise_rule_teacher_collection.py',
    'scripts/train/service_log_spool.py',
    'third_party/abot_renderer/render_sim_dynamic.py', 'third_party/abot_renderer/render_sim.py',
    'third_party/abot_renderer/gaussian_renderer/cpu.py',
    'third_party/abot_renderer/gaussian_renderer/cpu_composite.cpp',
    'third_party/abot_renderer/scene/cpu_gaussian_model.py',
    'third_party/abot_renderer/scene/gaussian_data.py',
    'vln_demo/src/remote_habitat/scene_cache.py', 'vln_demo/src/remote_habitat/worker.py',
    'vln_demo/src/remote_habitat/server.py',
    'verl/action_head/serve/qwen_query_worker.py',
    'scripts/eval/closedloop_evaluation_contract.py',
    'scripts/eval/action_head_benchmark_suite.py',
    'scripts/eval/action_head_suite_stage.py',
    'scripts/eval/action_head_suite_closedloop.py',
    'scripts/eval/v6_vlnce_eval.py',
    'scripts/eval/rule_teacher_repeat_companion.py',
    'scripts/eval/abot_renderer_compat.py', 'scripts/eval/abot_renderer_compat.json',
    'scripts/eval/scene_lifetime.py',
    'verl/action_head/runtime.py', 'vln_demo/src/vln_gym/env.py',
    'verl/action_head/evaluation_context.py', 'verl/action_head_rl/rollout.py',
    'verl/action_head/protocol_v8.py', 'verl/action_head/serve/continuous_context.py',
    'verl/action_head/context_compaction.py',
    'vln_demo/batch/model_gateway.py',
    'vln_demo/src/verl_access/context_budget.py',
    'vln_demo/src/verl_access/vlm_policy.py',
    'verl/action_head/serve/native_r3_capture.py',
    'vln_demo/src/vln_gym/fm_action.py', 'vln_demo/src/vln_gym/observation_profile.py',
    'verl/scripts/eval/action_head_fm_p0_p2/curriculum_action_head_eval.py',
    'verl/scripts/eval/qwen36_4query_ckpt_sweep/closedloop_eval.py')


def git_runtime(require_container=False):
    if os.environ.get('GPU_DIRECT_RUN') == '1':
        if Path.cwd().resolve() != REPO:
            raise ValueError('run from the selected local checkout')
        commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
        return dict(repository=str(REPO), commit=commit, checkout=str(REPO))
    from run_git_task import clean_repository
    origin = subprocess.check_output(['git','-C',str(REPO),'remote','get-url','origin'],text=True).strip()
    clean_repository(origin)
    commit = subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(REPO),'status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('evaluation checkout has uncommitted tracked changes')
    if require_container and (not str(REPO).startswith('/root/rl-code/') or Path.cwd().resolve()!=REPO):
        raise ValueError('GPU evaluation must start from its container-local Git checkout')
    return dict(repository=origin,commit=commit,checkout=str(REPO))


def seal_plan(plan, purpose):
    verify_normalized(plan['config'])
    if plan['config']['closedloop_protocol'] == V8_VERSION:
        wanted = protocol(V8_VERSION)['services']['max_model_len']
        for ckpt in plan['config']['checkpoints']:
            model = load(Path(ckpt['brain'])/'config.json')
            text = model.get('text_config', model)
            if int(text.get('max_position_embeddings', 0)) < wanted:
                raise ValueError('v8 evaluation exceeds the model native window; no RoPE override')
    plan['closedloop_contract'] = contract_for(plan['config'])
    plan['purpose'] = purpose
    plan['runtime_repo'] = str(REPO)
    plan['source_provenance'] = dict(git_runtime=git_runtime(),
        core={name:sha(REPO/name) for name in CORE}, protocol=protocol(plan['config']['closedloop_protocol'], plan['config']['evaluation_repeats'], plan['config'].get('context_compaction'), plan['config'].get('max_tool_calls')),
        purpose_is_metadata_only=True)
    plan['code_sha256'].update({str(REPO/name):sha(REPO/name) for name in CORE})
    plan['plan_sha256'] = suite.digest({k:v for k,v in plan.items() if k!='plan_sha256'})
    return plan


def prepare(config_path, output, purpose):
    git_runtime()
    output = suite.artifact_root(output)
    if output.exists():
        raise ValueError('prepare requires a new output; preserve previous attempts')
    config = selection_config(load(config_path))
    config.update(evaluation_repeat_index=1)
    save(output/'selection.json', config)
    first = suite.prepare(output/'selection.json', output/'rounds/repeat_01')
    first['config_template'] = {k:v for k,v in first['config'].items() if k!='checkpoints'}
    seal_plan(first,purpose)
    save(Path(first['output'])/'plan.json',first)
    rounds=[]
    for repeat in range(1, (1 if config.get('pooled_attempts') else config['evaluation_repeats'])+1):
        plan=copy.deepcopy(first)
        plan['output']=str(output/f'rounds/repeat_{repeat:02d}')
        plan['config']['evaluation_repeat_index']=repeat
        plan['plan_sha256']=suite.digest({k:v for k,v in plan.items() if k!='plan_sha256'})
        path=Path(plan['output'])/'plan.json';save(path,plan);rounds.append(str(path))
    run=dict(schema=VERSION,output=str(output),purpose=purpose,
        contract=contract_for(config),rounds=rounds,
        git_runtime=first['source_provenance']['git_runtime'])
    save(output/'run.json',run)
    save(output/'base-plan.json',first)
    return run


def prepare_watch(config_path,binding_path,output,purpose):
    """Bind the existing resident watcher to the same prepared suite core."""
    config=load(binding_path)
    required=('training_root','training_config','checkpoint_root','checkpoint_index',
              'resident_owners','steps','slow_model','python')
    if any(key not in config for key in required):
        raise ValueError('watch binding misses checkpoint/resident fields')
    if (config.get('historical_first_attempts') or config.get('historical_attempts')) and load(config_path).get('continuation_protocol') != LEGACY_CONTINUATION:
        raise ValueError('unified evaluation starts fresh; old protocol is historical')
    prepare(config_path,output,purpose)
    config.update(root=str(output),base_plan=str(output/'base-plan.json'),
        runtime_repo=str(REPO),evaluation_repeats=load(output/'base-plan.json')['config']['evaluation_repeats'],purpose=purpose,
        closedloop_protocol=load(output/'base-plan.json')['config']['closedloop_protocol'],execution_protocol='v7',
        pooled_attempts=load(output/'base-plan.json')['config'].get('pooled_attempts',False))
    save(output/'watcher-config.json',config)
    return config


def run_round(plan_path,rank):
    actual=git_runtime(require_container=True)
    plan=load(plan_path);suite.verify(plan)
    if plan.get('closedloop_contract') != contract_for(plan.get('config', {})):
        raise ValueError('old evaluation plans are historical only; prepare unified evaluation')
    if actual != plan['source_provenance']['git_runtime']:
        raise ValueError('actual container checkout differs from prepared plan')
    verify_dependencies(plan['config']['benchmarks'])
    return suite.run(plan_path,rank)


def verify_dependencies(benchmarks):
    if os.environ.get('GPU_DIRECT_RUN') == '1':
        if 'sage-3d' in benchmarks and Path(os.environ.get('ACTION_HEAD_RENDER_REPO', '')).resolve() != REPO/'third_party/abot_renderer':
            raise ValueError('use the CPU renderer in this checkout')
        if any(b in benchmarks for b in ('r2r', 'rxr')):
            navigation = Path(os.environ.get('ACTION_HEAD_NAVIGATION_REPO', ''))
            if not navigation.is_absolute() or not (navigation/'data').is_dir():
                raise ValueError('set an absolute navigation checkout containing data/')
        return
    """Reject shared source fallbacks before releasing GPUs or starting services."""
    inputs=[]
    if 'sage-3d' in benchmarks:
        render = Path(os.environ.get('ACTION_HEAD_RENDER_REPO', ''))
        if render.resolve() != REPO/'third_party/abot_renderer':
            raise ValueError('SAGE CPU renderer must use the versioned container checkout')
        for name in ('render_sim_dynamic.py', 'gaussian_renderer/cpu.py',
                     'gaussian_renderer/cpu_composite.cpp', 'scene/cpu_gaussian_model.py'):
            if not (render/name).is_file():
                raise ValueError('CPU renderer source is missing: '+name)
    if any(b in benchmarks for b in ('r2r','rxr')):
        inputs.append(('ACTION_HEAD_NAVIGATION_REPO','40f19547941454835ff5e63c22994da74605cfea'))
    for key,expected in inputs:
        value=os.environ.get(key)
        if not value or not str(Path(value).resolve()).startswith('/root/rl-code/'):
            raise ValueError('container-local Git dependency required: '+key)
        root=Path(value).parent if key=='ACTION_HEAD_NAVIGATION_REPO' else Path(value)
        commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
        if commit!=expected:
            raise ValueError('evaluation dependency version differs: '+key)
        if subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],text=True).strip():
            raise ValueError('evaluation dependency source changed: '+key)
        if key=='ACTION_HEAD_NAVIGATION_REPO' and not (Path(value)/'data').is_dir():
            raise ValueError('native dataset directory is missing')
        if key=='ACTION_HEAD_RENDER_REPO':
            from abot_renderer_compat import CONTRACT,patched_source
            for file in CONTRACT['files']:patched_source(root/file,root)
            for sub,package in [('diff-plane-rasterization','diff_plane_rasterization'),('simple-knn','simple_knn')]:
                if not (root/'render_server/submodules'/sub/package/'_C.so').is_file():
                    raise ValueError('compiled renderer library is missing: '+package)


def summarize(run):
    from rule_teacher_repeat_companion import summarize_outputs
    plans=[load(p) for p in run['rounds']]
    for plan in plans:suite.verify(plan)
    results={}
    for ckpt in plans[0]['config']['checkpoints']:
        results[ckpt['name']]=summarize_outputs(plans[0],
            [Path(p['output']) for p in plans],ckpt['name'])
    result=dict(evaluation_valid=True,contract=run['contract'],checkpoints=results,
                purpose=run['purpose'],source_runs=run['rounds'])
    save(Path(run['output'])/'summary.json',result)
    return result


def run_all(run_path,rank):
    run=load(run_path)
    if run['contract'] != contract_for(load(run['rounds'][0])['config']):
        raise ValueError('evaluation run contract differs')
    for path in run['rounds']:
        code=run_round(Path(path),rank)
        if code:
            save(Path(run['output'])/f'failure-node{rank}.json',
                dict(round=path,exit_code=code,time=time.time()))
            return code
    if rank==0:summarize(run)
    return 0


def watch(config_path,rank):
    git_runtime(require_container=True)
    config=load(config_path);base=load(config['base_plan'])
    owners=config['resident_owners'];nodes=base['config_template']['nodes']
    if len(owners)!=nodes or len({o['hostname'] for o in owners})!=nodes or not 0<=rank<nodes:
        raise ValueError('watch requires one distinct bound resident per node and a valid rank')
    verify_normalized(dict(base['config_template'],checkpoints=base['config']['checkpoints']))
    if base.get('closedloop_contract') != contract_for(base['config']) or config.get('evaluation_repeats')!=base['config']['evaluation_repeats']:
        raise ValueError('watch requires a unified base plan with matching repeat count')
    if (config.get('historical_first_attempts') or config.get('historical_attempts')) and base['config'].get('closedloop_protocol') != LEGACY_CONTINUATION:
        raise ValueError('old protocol results cannot be reused by unified evaluation')
    verify_dependencies(base['config_template']['benchmarks'])
    from rule_teacher_openloop_companion import run
    return run(config_path,rank)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--config',type=Path,required=True)
    prep.add_argument('--output',type=Path,required=True);prep.add_argument('--purpose',choices=PURPOSES,required=True)
    binding=sub.add_parser('prepare-watch');binding.add_argument('--config',type=Path,required=True)
    binding.add_argument('--binding',type=Path,required=True);binding.add_argument('--output',type=Path,required=True)
    binding.add_argument('--purpose',choices=PURPOSES,required=True)
    start=sub.add_parser('run');start.add_argument('--run',type=Path,required=True);start.add_argument('--rank',type=int,required=True)
    round_=sub.add_parser('run-round',help='internal shared companion/suite backend')
    round_.add_argument('--plan',type=Path,required=True);round_.add_argument('--rank',type=int,required=True)
    follow=sub.add_parser('watch');follow.add_argument('--config',type=Path,required=True);follow.add_argument('--rank',type=int,required=True)
    task=sub.add_parser('resident-watch',help='bootstrap a submitted companion and use the shared watcher')
    task.add_argument('--spec',type=Path,required=True);task.add_argument('--rank',type=int,required=True)
    check=sub.add_parser('verify');check.add_argument('--run',type=Path,required=True)
    status=sub.add_parser('status');status.add_argument('--run',type=Path,required=True)
    console=sub.add_parser('console',help='interactive visualization using the same suite services and collectors')
    console.add_argument('--config',type=Path,required=True)
    console.add_argument('--output',type=Path,required=True)
    console.add_argument('--dependency-spec',type=Path,required=True)
    console.add_argument('--port',type=int,choices=range(5000,5006),default=5002)
    args=p.parse_args()
    if args.command=='prepare':print(prepare(args.config,args.output,args.purpose))
    elif args.command=='console':
        from unified_closedloop_console import launch
        return launch(args.config,args.output,args.port,args.dependency_spec)
    elif args.command=='prepare-watch':print(prepare_watch(args.config,args.binding,args.output,args.purpose))
    elif args.command=='run':return run_all(args.run,args.rank)
    elif args.command=='run-round':return run_round(args.plan,args.rank)
    elif args.command=='watch':return watch(args.config,args.rank)
    elif args.command=='resident-watch':
        from unified_companion_task import run
        return run(args.spec,args.rank)
    elif args.command=='verify':
        run=load(args.run)
        for path in run['rounds']:suite.verify(load(path))
        print(dict(verified=True,contract=run['contract'],rounds=len(run['rounds'])))
    else:
        run=load(args.run);print(dict(contract=run['contract'],summary=(Path(run['output'])/'summary.json').exists(),
            nodes=[dict(round=i+1,status=[{k:v for k,v in load(p).items() if k in ('state','updated','stage','evaluation_valid')}
                   for p in Path(load(path)['output']).glob('nodes/*/status.json')])
                   for i,path in enumerate(run['rounds'])]))
    return 0


if __name__=='__main__':raise SystemExit(main())
