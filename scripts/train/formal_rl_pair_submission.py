"""Preparation and idempotent official submission behind submit_formal_rl_pair.sh."""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import align_three_source_observations as io
from prepare_git_workspace import rl_entry, companion_entry
from watch_v6_rl_local_credit import process_identity, same_process_alive
import watch_rl_pair_completion as lifecycle
from rloo_variants import add_arguments as add_rloo_arguments, resolve as resolve_rloo

REPO = Path(__file__).resolve().parents[2]
BASE = Path('/workspace')
sys.path.insert(0, str(BASE/'dlc_cli'))
from workspace_config import resolve_workspace



def companion_steps(final_step, start_step=0):
    """Evaluate every four global RL steps, retaining initialization and endpoint."""
    return sorted({start_step, final_step, *range(((start_step + 3) // 4) * 4, final_step + 1, 4)})


def platform(arguments, receipt):
    """Preserve rejection/timeout evidence; never retry create in this call."""
    if receipt.exists():
        raise ValueError('new immutable API receipt required: '+str(receipt))
    result = subprocess.run([lifecycle.CLI, *arguments], capture_output=True,
                            text=True, timeout=60)
    envelope = json.loads(result.stdout)
    io.write_json(receipt, envelope)
    if result.returncode or not 200 <= envelope['http_status'] < 300 or envelope['response']['code'] != 2000:
        raise RuntimeError('official platform rejected the request; see '+str(receipt))
    return envelope['response']['data']


def companion_checkpoint(checkpoint, training_config, fallback_brain, *, name=None):
    """Select the exported joint policy whenever checkpoint provenance has one."""
    sys.path.insert(0, str(REPO/'scripts/eval'))
    from action_head_benchmark_suite import checkpoint_identity
    checkpoint = Path(checkpoint).resolve(strict=True)
    identity = io.bounded(lambda: checkpoint_identity(checkpoint),
                          'companion_checkpoint_identity:'+str(checkpoint))
    return dict(name=name or checkpoint.name, path=str(checkpoint),
                training_config=str(training_config),
                brain=identity.get('qwen_dir') or str(fallback_brain))


def prepared(args):
    train, evaluation = args.training_root, args.evaluation_root
    original = dict(io.load(args.reference/'launch/config.json'))
    variant, settings = resolve_rloo(args)
    settings.apply({'config': original})
    original['keep_last_checkpoints'] = 2
    evaluation_commit = getattr(args, 'evaluation_commit', None) or args.commit
    batch_size = getattr(args, 'batch_size', None)
    max_tool_calls = getattr(args, 'max_tool_calls', None)
    if batch_size is not None or max_tool_calls is not None:
        from copy import deepcopy
        from prepare_git_rl_task import configure
        requested = io.load(args.reference/'launch/submit_plan.json')
        requested['config'] = deepcopy(original)
        configure(requested, batch_size=batch_size, max_tool_calls=max_tool_calls)
        original = requested['config']
    if args.final_step is not None: original['opt_steps'] = args.final_step
    if args.terminal_success_reward is not None:
        original['terminal_success_reward'] = args.terminal_success_reward
    if not io.exists(train/'launch/git_task.json'):
        subprocess.run([sys.executable, '-B', str(REPO/'scripts/train/prepare_git_rl_task.py'),
            '--reference', str(args.reference), '--output', str(train), '--commit', args.commit,
            '--opt-steps', str(original['opt_steps']),
            *(['--batch-size', str(batch_size)] if batch_size is not None else []),
            *(['--max-tool-calls', str(max_tool_calls)] if max_tool_calls is not None else []),
            '--overlap-cpu-group-preparation', '--cpu-group-preparation-tasks-per-node', '2',
            '--sage-cpu-render-parallelism', '8', '--native-render-shared-cache',
            '--native-render-cache-max-gib', '128', '--native-render-scene-load-reserve-gib', '16',
            '--native-episode-cache'], check=True)
        plan = io.load(train/'launch/submit_plan.json')
        settings.apply(plan)
        plan['config']['keep_last_checkpoints'] = 2
        plan['rloo_variant'] = variant
        plan['checkpoint_retention_evaluation_root'] = str(evaluation/'run')
        io.write_json(train/'launch/config.json', plan['config'])
        ckpt = args.checkpoint
        source_root = ckpt.parent.parent.parent
        index = source_root/'checkpoint-index.json'
        report = io.load(ckpt/'report.json')
        assert report['global_step'] in io.load(index)['steps']
        files = ['action_head/fast_slow_action_head.pt', 'action_head/fast_slow_action_head_config.json',
            'dualvln_query/dualvln_query_embeddings.pt', 'report.json', 'training_experiment.json']
        initial = dict(checkpoint=str(ckpt), global_step=report['global_step'],
            publication_index=str(index), publication_index_sha256=io.digest(io.read_bytes(index)),
            artifacts=[dict(path=str(ckpt/name),sha256=io.digest(io.read_bytes(ckpt/name))) for name in files],
            optimizer_restored=False, slow_model=original['slow_model'], fast_backbone=original['fast_backbone'],
            description='published openloop head/all-query initialization; fresh RL Adam/scheduler/RNG')
        plan.update(initial_checkpoint=str(ckpt), initialization=initial['description'],
                    openloop_initialization=initial, workspace=getattr(args, 'workspace', 'pai2_vla_xpu'))
        if getattr(args, 'teacher_stop', False):
            from action_head_rl.teacher_stop import contract
            plan.update(teacher_stop_training=True, teacher_stop_contract=contract())
        io.write_json(train/'launch/submit_plan.json', plan)
        io.write_json(train/'launch/config-diff.json', dict(reference=str(args.reference/'launch/config.json'),
            config_equal=plan['config']==original, initialization=initial, full_resume=False))
    plan = io.load(train/'launch/submit_plan.json')
    if bool(plan.get('teacher_stop_training')) != bool(getattr(args, 'teacher_stop', False)):
        raise ValueError('algorithm selection differs; use a fresh name')
    settings.check_plan(plan)
    spec = io.load(train/'launch/git_task.json')
    if (plan['config'] != original or spec['commit'] != args.commit
            or plan['initial_checkpoint'] != str(args.checkpoint)
            or any(k in spec or k in plan for k in ('resume_from_path','policy_warm_start_checkpoint'))):
        raise ValueError('existing training does not match this exact formal baseline; use a fresh name')
    for artifact in plan['openloop_initialization']['artifacts']:
        if io.digest(io.read_bytes(artifact['path'])) != artifact['sha256']:
            raise ValueError('initial checkpoint artifact changed: '+artifact['path'])
    if not io.exists(evaluation/'spec.json'):
        evaluation.mkdir(exist_ok=False)
        selection = io.load(REPO/'configs/eval/unified_closedloop_eval.example.json')
        selection.update(nodes=4,evaluation_repeats=1,max_tool_calls=100)
        selection['checkpoints'] = [companion_checkpoint(args.checkpoint,
            train/'launch/config.json', original['slow_model'], name='global_step_000000')]
        io.write_json(evaluation/'selection.json', selection)
        binding = dict(training_root=str(train),training_config=str(train/'launch/config.json'),
            checkpoint_root=str(train/'train'),checkpoint_index=str(train/'train/latest_checkpointed_iteration.txt'),
            checkpoint_publication='rl_tracker', steps=companion_steps(original['opt_steps']),
            checkpoint_paths={'0':str(args.checkpoint)},slow_model=original['slow_model'],
            python='/usr/local/bin/python3.12',resident_owners=[],suite_gpu_preflight=False)
        io.write_json(evaluation/'binding.json',binding)
        io.write_json(evaluation/'spec.json',dict(root=str(evaluation),commit=evaluation_commit,nodes=4,
            purpose='rl-companion',selection=str(evaluation/'selection.json'),binding=str(evaluation/'binding.json'),
            navigation_data='/data/shared/groups/mfm/common/anno/vlmevalkit/RynnBrain/data',
            dependencies={'RynnBrain':dict(repository='https://github.com/alibaba-damo-academy/RynnBrain.git',
                commit='40f19547941454835ff5e63c22994da74605cfea',
                mirror='/opt/navigation-runtime/VLMEvalKit/3rdparty/RynnBrain')},renderer_binaries=[]))
    selected = io.load(evaluation/'selection.json')
    bound = io.load(evaluation/'binding.json')
    if (io.load(evaluation/'spec.json')['commit'] != evaluation_commit or selected['nodes'] != 4
            or selected['evaluation_repeats'] != 1 or selected['sample_counts'] != dict.fromkeys(['sage-3d','r2r','rxr'],100)
            or selected['max_tool_calls'] != 100 or bound['steps'] != companion_steps(original['opt_steps'])
            or bound['training_root'] != str(train) or bound['checkpoint_paths']['0'] != str(args.checkpoint)
            or original['nnodes'] != 8 or original['checkpoint_interval'] != 2):
        raise ValueError('formal pair differs from the validated 8+4-node, requested-step configuration')
    profile = resolve_workspace(getattr(args, 'workspace', 'pai2_vla_xpu'))
    exports = ('export NCCL_IB_DISABLE=0','export NCCL_NET=IB','export NCCL_NET_PLUGIN=none',
        'export NCCL_MIN_NCHANNELS=4','export NCCL_MAX_NCHANNELS=4','unset NCCL_PF_HCA')
    for root,nodes,phase,body,config,product in [
        (train,8,14,rl_entry(spec['repository'],args.commit,train,train/'launch/git_task.json',8,megatron=True,exports=exports),train/'launch/config.json',train/'train'),
        (evaluation,4,5,companion_entry(spec['repository'],evaluation_commit,evaluation,evaluation/'spec.json',4),evaluation/'selection.json',evaluation/'run')]:
        (root/'platform').mkdir(exist_ok=True)
        if io.exists(root/'platform/create.json'):
            continue  # Never hot-edit a submitted command or configuration.
        if phase==5:
            body='export ACTION_HEAD_TEACHER_STOP_TRAINING=0\n'+body
        command = 'bash -c '+shlex.quote(body)
        subprocess.run(['bash','-n'],input=body,text=True,check=True)
        subprocess.run(['/bin/sh','-n','-c',command],check=True)
        request = dict(name=root.name,desc=(f'RLOO配置：{plan["rloo_hyperparameters"]}；')+(('D5序列级RLOO：文本与动作头共用整轨迹优势，保留局部碰撞扣分和裁剪；' if getattr(args,'sequence_rloo',False) else '')+(f"信息发现奖励系数={args.information_reward_coefficient}；其余沿用参考D5配置。" if getattr(args,'information_reward_coefficient',None) is not None else '')+('首次到达教师STOP变体：50次自主调用+最多1次教师STOP，单位SFT权重1、教师段无TIS/LOO；' if getattr(args,'teacher_stop',False) else '')+(f"正式v8 RL：指定开环初始化、fresh Adam；B{original['num_groups_per_step']*original['group_size']}/K{original['group_size']}，训练{original['max_tool_calls']}次调用、{original['opt_steps']}步，每2步保存，成功后自动停止，失败占卡保留回执。" if phase==14 else
            '正式RL四节点伴评：起点、每4步及终点，三源各100例K1；统一100工具预算，报告pass@1、SPL及R2R/RxR OSR，等待占卡，全部评完自动停止。')),
            phaseCode=phase,module='vla',platType=2,operatedBy='anonymous',spaceId=str(profile.workspace_id),
            quotaId=profile.resource_id,podCount=nodes,priority=8,
            resourceConfig=dict(cpu=184,memory=1802240,gpuSource=dict(gpuType='',gpu=16)),
            imageUrl='registry.example.invalid/aliyun-wl-gpu02/ppu-training:verl-veomni-eval-lite-zhouyang-0518',
            dataSources=[dict(dataSourceId=s.data_source_id,dataSourceName='',mountPath=s.mount_path) for s in profile.data_sources],
            commitId=(args.commit if phase==14 else evaluation_commit),downloadCode=False,configPath=str(config),parsedProductDir=str(product),
            workDir='/root',userCommand=command)
        if getattr(args, 'retain_resources', False):
            request['desc'] = request['desc'].replace('成功后自动停止', '成功后继续占卡').replace('全部评完自动停止','全部评完继续占卡')
        io.write_json(root/'platform/create.json', request)
        (root/'platform/command.sh').write_text(body)
    return plan


def created_task(root):
    for filename in ('create-response.json','create-response-r2.json'):
        path = root/'platform'/filename
        if io.exists(path):
            envelope=io.load(path)
            if envelope.get('response',{}).get('code')==2000:
                # Actual API shape: data.tasks[], not data.experimentId.
                tasks=envelope['response']['data']['tasks']
                if len(tasks)!=1:
                    raise ValueError('unexpected create task count')
                return tasks[0]
    return None


def create_one(root):
    task = created_task(root)
    if task:
        return task
    intent = root/'platform/create-intent.json'
    if io.exists(intent):
        # A timed-out create may have succeeded: never repeat a mutation.
        raise RuntimeError('prior submission has no conclusive response; reconcile the official task list before retrying')
    request = io.load(root/'platform/create.json')
    found = platform(['list','--params',json.dumps(dict(createdBy='anonymous',query=request['name'],page=1,size=100))],
        root/'platform/list-before-create.json')
    if found.get('total',0)>len(found.get('list',[])):
        raise ValueError('task-name query was paginated; reconcile all matches before creating')
    if any(row['name']==request['name'] for row in found.get('list',[])):
        raise ValueError('an official task already uses this name; adopt its receipt instead of creating again')
    io.write_json(intent,dict(time=time.time(),request_sha256=io.digest(io.read_bytes(root/'platform/create.json'))))
    workspace = 'pai2_mfm_xpu' if request['quotaId']=='quota-anonymous' else 'pai2_vla_xpu'
    data = platform(['--submit-workspace',workspace,'create','--json-file',str(root/'platform/create.json')], root/'platform/create-response.json')
    if len(data['tasks']) != 1:
        raise ValueError('unexpected create task count')
    return data['tasks'][0]


def start_monitor(plan):
    root=Path(plan.get('training', plan.get('evaluation'))['root'])/'platform'
    path=root/'pair-completion-plan.json'
    if io.exists(path) and io.load(path)!=plan:
        raise ValueError('previous supervisor binding differs')
    if not io.exists(path):io.write_json(path,plan)
    owner=root/'pair-supervisor/owner.json'
    if io.exists(owner) and same_process_alive(io.load(owner)['identity']):
        return io.load(owner)['identity']
    with (root/'pair-supervisor.log').open('ab',buffering=0) as log:
        child=subprocess.Popen([sys.executable,'-B',str(REPO/'scripts/train/watch_rl_pair_completion.py'),
            '--plan',str(path)],cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    identity=process_identity(child.pid)
    io.write_json(root/f'pair-supervisor-launch-{child.pid}.json',dict(identity=identity,plan=str(path),source=str(REPO)))
    return identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['prepare','submit','status'],required=True)
    for key in ('training-root','evaluation-root','reference','checkpoint'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--workspace',choices=['pai2_vla_xpu','pai2_mfm_xpu'],default='pai2_vla_xpu')
    parser.add_argument('--teacher-stop',action='store_true')
    add_rloo_arguments(parser)
    parser.add_argument('--evaluation-commit')
    parser.add_argument('--retain-resources',action='store_true')
    parser.add_argument('--final-step',type=int)
    parser.add_argument('--batch-size',type=int)
    parser.add_argument('--max-tool-calls',type=int)
    args=parser.parse_args()
    variant, settings = resolve_rloo(args)
    args.sequence_rloo = settings.credit_level == 'sequence'
    print(json.dumps(dict(variant=variant, **asdict(settings)), ensure_ascii=False))
    if os.environ.get('TASK_API_USERNAME')!='anonymous':raise ValueError('wrong platform identity')
    for root in (args.training_root,args.evaluation_root):
        if not root.resolve().is_relative_to(io.ART) or root.resolve()==io.ART:raise ValueError('wrong output root')
    if args.mode!='status':prepared(args)
    if args.mode=='prepare':
        print(json.dumps(dict(prepared=True,submitted=False,training=str(args.training_root),evaluation=str(args.evaluation_root))))
        return
    if args.mode=='submit':
        submission_lock=(args.training_root/'platform/submission.lock').open('a')
        fcntl.flock(submission_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        subprocess.run(['git','-c','gc.auto=0','-C',str(REPO),'fetch','--no-tags','origin',args.commit],check=True)
        stamp=str(time.time_ns())
        output=args.training_root/'platform'
        quotas=platform(['quotas','--params','{"platType":2}'],output/f'quotas-{stamp}.json')
        profile=resolve_workspace(args.workspace)
        if not any(x['quotaId']==profile.resource_id and str(x['spaceId'])==str(profile.workspace_id) for x in quotas['list']):raise ValueError('quota mapping differs')
        for nodes in (8,4):
            permission=platform(['--submit-workspace',args.workspace,'quota-status','--space-id',str(profile.workspace_id),'--quota-id',profile.resource_id,'--pod-count',str(nodes)],output/f'permission-{nodes}-{stamp}.json')
            if permission['maxAllowedPriority']<8 or permission['willForceDowngrade']:raise ValueError('priority permission differs')
        train=create_one(args.training_root)
        # Associate the companion with the actual returned training experiment.
        if not created_task(args.evaluation_root):
            path=args.evaluation_root/'platform/create.json';request=io.load(path)
            request['experimentId']=str(train['experimentId']);io.write_json(path,request)
        create_one(args.evaluation_root)
    final_step=io.load(args.training_root/'launch/config.json')['opt_steps']
    plan={}
    for kind,root,nodes,extra in [('training',args.training_root,8,dict(final_step=final_step)),
            ('evaluation',args.evaluation_root,4,dict(steps=companion_steps(final_step)))]:
        task=created_task(root)
        if task is None:raise ValueError('no successful create receipt: '+str(root))
        request=io.load(root/'platform/create.json')
        item=dict(root=str(root),kind=kind,nodes=nodes,commit=(args.commit if kind=='training' else args.evaluation_commit or args.commit),task_id=str(task['taskId']),job_id=task['externalTaskId'],
            product_dir=request['parsedProductDir'],name=request['name'],quota_id=request['quotaId'],space_id=request['spaceId'],**extra)
        detail=lifecycle.checked_detail(item)['response']['data']
        print(json.dumps(dict(kind=kind,**{k:detail.get(k) for k in ('taskId','jobId','status','podCount','priority','quotaId')})))
        plan[kind]=item
        if args.mode=='submit':
            stamp=str(time.time_ns())
            need=platform(['experiment-info-need','--json',json.dumps(dict(taskId=item['task_id']))],root/f'platform/info-need-{stamp}.json')
            before=platform(['experiment-info-get','--task-id',item['task_id']],root/f'platform/info-before-{stamp}.json')
            if need['need'] and not before:
                types=platform(['iter-types'],root/f'platform/iter-types-{stamp}.json')
                chosen=next(x for x in types if x['value']==7 and not x['needApproval'])
                fill=dict(taskId=item['task_id'],jobId=item['job_id'],modelDesc=request['desc'],iterType=chosen['value'],
                    purposeDesc=('本次D5序列级RLOO：复用整轨迹shaped_return，K8同case留一基线，不除标准差；文本与head共享优势，head保留局部碰撞扣分和裁剪；其余D5奖励及损失归一化不变。' if args.sequence_rloo else '')+('本次算法变体为首次到达截断后补教师STOP，前缀原v3 RTG/LOO、教师单位权重SFT且不做TIS；自主调用50+最多1教师调用；评测无教师。' if args.teacher_stop else '')+'实验目的与目标：从固定开环checkpoint开展正式闭环RL基线及三源固定子集伴评，保持正式训练配置，建立后续组件消融的可追溯对照。预期产物与验收：形成指定步数训练checkpoint、每4步三源各100条K1有效评测、pass@1/SPL与R2R及RxR OSR；核对代码来源、实际更新、原始日志、退出回执和完成后自动停止。')
                path=root/f'platform/info-fill-{stamp}.json';io.write_json(path,fill)
                platform(['experiment-info-fill','--json-file',str(path)],root/f'platform/info-fill-response-{stamp}.json')
                platform(['experiment-info-get','--task-id',item['task_id']],root/f'platform/info-after-{stamp}.json')
    if args.mode=='submit':
        if args.retain_resources:
            io.write_json(args.training_root/'platform/retained-pair-plan.json',plan)
            print(json.dumps(dict(resources_retained=True, automatic_platform_stop=False)))
        else: print(json.dumps(dict(supervisor=start_monitor(plan))))


if __name__ == "__main__":
    raise SystemExit("DLC submission entry removed in the open-source edition; use the root GPU run scripts.")
