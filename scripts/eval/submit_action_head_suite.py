#!/usr/bin/env python3
"""Prepare or submit a 1/4-node evaluation through the official train-platform CLI.

No legacy DLC create SDK. Templates are live, owner-verified VLA task details;
quota and priority are rechecked for every submission. Default is prepare only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from action_head_benchmark_suite import REPO, ROOT, EVAL, GENERAL, atomic_json, sha256_file, verify

CLI = ROOT / 'bin/train-platform'
QUOTA = 'quota-anonymous'


def platform_call(args, output):
    # Activate the user's official wrapper without persisting credentials.
    command = 'source ' + shlex.quote(str(ROOT / 'env_train_platform.sh')) + '; exec ' + shlex.join([str(CLI), *args])
    process = subprocess.run(['bash', '-c', command], capture_output=True, text=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(process.stdout)
    if process.returncode:
        (output.with_suffix('.stderr.txt')).write_text(process.stderr)
        raise RuntimeError(f'train-platform failed ({process.returncode}); see {output}')
    result = json.loads(process.stdout)
    if not 200 <= result['http_status'] < 300 or result['response']['code'] != 2000:
        raise RuntimeError(f'train-platform rejected request; see {output}')
    return result['response']['data']


def verify_template(value):
    if (value.get('createdBy') != 'anonymous' or value.get('module') != 'vla'
        or int(value.get('platType', 0)) != 2 or value.get('quotaId') != QUOTA
        or str(value.get('spaceId')) != '150120'):
        raise ValueError('template must be the current user own pai2_vla_xpu / vla task')
    resource = value['resourceConfig']
    if isinstance(resource, str):
        resource = json.loads(resource)
    if resource['gpuSource']['gpu'] != 16:
        raise ValueError('template must have 16 GPUs per node')
    sys.path.insert(0, str(ROOT / 'dlc_cli'))
    from workspace_config import resolve_workspace
    profile = resolve_workspace('pai2_vla_xpu')
    mounts = {(row['dataSourceId'], row['mountPath'].rstrip('/')) for row in value['dataSources']}
    expected = {(row.data_source_id, row.mount_path.rstrip('/')) for row in profile.data_sources}
    if mounts != expected or profile.resource_id != QUOTA:
        raise ValueError('template mounts differ from explicit pai2_vla_xpu workspace profile')
    return resource


def command_body(plan_path):
    plan = json.loads(plan_path.read_text())
    runtime = Path(plan['runtime_repo'])
    body = (runtime / 'scripts/eval/run_action_head_suite.sh').read_text()
    return ('export ACTION_HEAD_SUITE_REPO=' + shlex.quote(str(runtime)) + '\n' +
            'export ACTION_HEAD_SUITE_PLAN=' + shlex.quote(str(plan_path.resolve())) + '\n' + body)


def source_commit(plan, destination):
    """Version the actual frozen entry/code, rather than borrowing another HEAD."""
    if destination.exists():
        raise ValueError('submission snapshot already exists; use a new platform attempt')
    destination.mkdir(parents=True)
    for name in plan['code_sha256']:
        path = Path(name)
        target = destination / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    atomic_json(destination / 'runtime-code-contract.json', plan['code_sha256'])
    subprocess.run(['git', 'init', '-q', str(destination)], check=True)
    subprocess.run(['git', '-C', str(destination), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(destination), '-c', 'user.name=anonymous',
                    '-c', 'user.email=anonymous@local', 'commit', '-q', '-m', 'Freeze action head benchmark suite'], check=True)
    return subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()


def prepare_request(plan_path, template, platform_root, experiment_id=None):
    plan = json.loads(plan_path.read_text())
    verify(plan)
    resource = verify_template(template)
    nodes = plan['config']['nodes']
    body = command_body(plan_path)
    command = 'bash -c ' + shlex.quote(body)
    platform_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(['bash', '-n'], input=body, text=True, check=True)
    subprocess.run(['/bin/sh', '-n', '-c', command], check=True)
    (platform_root / 'user-command.sh').write_text(body)
    runtime = Path(plan['runtime_repo'])
    if not runtime.is_relative_to(Path(plan['output'])) or runtime.is_relative_to(REPO):
        raise ValueError('submission must execute an independent data-disk source snapshot')
    source = runtime.parent
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
        raise ValueError('runtime snapshot changed after its source commit')
    label = plan['config']['checkpoints'][0]['name'] if len(plan['config']['checkpoints']) == 1 else 'suite'
    request = dict(name='anonymous_ah_' + label + '_' + str(nodes) + 'node_' + plan['plan_sha256'][:10],
        phaseCode=5, module='vla', platType=2, quotaId=QUOTA, spaceId='150120',
        resourceConfig=resource, imageUrl=template['imageUrl'], podCount=nodes,
        priority=8, operatedBy='anonymous', commitId=commit, configPath=str(plan_path.resolve()),
        parsedProductDir=plan['output'], workDir=str(runtime), dataSources=template['dataSources'],
        downloadCode=False, preemptible=False, parentModels=[], userCommand=command,
        desc='按benchmark外层与checkpoint内层顺序评测冻结清单；本次范围：'
             + '、'.join(plan['config']['benchmarks']) + '；'
             '固定子集跨模型一致，节点仅在数据上分片，闭环相机绑定训练配置并实测验收。'
             '保留逐回合错误和各阶段有效性，统一汇总原生指标，产物写数据盘，缓存容器本地。')
    if len(plan['config']['checkpoints']) == 1:
        ckpt = plan['config']['checkpoints'][0]
        if ckpt.get('policy') == 'brain_only':
            request['desc'] = ('DualVLN原始大模型backbone对照：' + ckpt.get('source_model', Path(ckpt['brain']).name) + '；'
                '仅评通用12benchmark全量27771题，使用同主实验题池、原生问答提示和原生评分器，'
                'K=1、输出32768token、温度0.7/top_p0.8，8个TP2副本、128并发请求。'
                '本任务评测原始Instruct权重；已完成的DualVLN导航微调后S2成绩另列。'
                '预期得到可追溯的完整12项原始backbone预测/指标，运行缓存容器本地，产物数据盘。')
        elif ckpt.get('policy') == 'primitive' and all(b in GENERAL for b in plan['config']['benchmarks']):
            request['desc'] = ('Qwen3.6-35B-A3B历史原生tool大脑的通用空间理解补评；'
                '本次仅全量通用12benchmark，使用与B45/v3/Dual相同题池和原生问答提示/评分器，'
                'K=1、输出32768token、温度0.7/top_p0.8，128并发请求、8个TP2副本；'
                '不重跑已有三个闭环，不输入导航动作工具或改变题目。'
                '预期获得12项完整预测与原生评分；正式产物数据盘，运行缓存容器本地。')
        elif ckpt.get('policy') == 'primitive':
            request['desc'] = ('Qwen3.6-35B-A3B历史大脑的原生ReAct工具基线，全量SAGE-3D、R2R、RxR；'
                f'单次前进/后退0.05–{ckpt["max_distance_m"]:g}m，system prompt、schema、执行器一致，'
                '仍以≤0.25m子步执行。慢决策预算200/100/100，一轮一个工具调用，K=1；'
                '同主实验样本池、相机和修复navmesh，保留原生SR/SPL、主动STOP诊断、逐调用轨迹及无效回合账本。'
                '预期获得完整有效主实验基线；运行缓存容器本地，正式产物数据盘。')
        elif ckpt.get('policy') == 'dualvln':
            request['desc'] = ('DualVLN官方权重全量三个闭环及通用12bench，闭环保留原生S2→S1，'
                '慢决策预算200/100/100、K=1；640×480/79度/1.25m相机、S1下视30度。'
                '通用评测使用该官方checkpoint逐张量相等的Qwen2.5-VL大脑导出。'
                '同主实验样本池与修复navmesh，预期获得完整有效SR/SPL和12bench原生评分；'
                '无效回合单列，不记作导航失败，缓存容器本地，产物数据盘。')
    experiment = experiment_id or template.get('experimentId')
    if experiment:
        request['experimentId'] = experiment
    atomic_json(platform_root / 'create.json', request)
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--template-task-id', required=True)
    parser.add_argument('--experiment-id')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--attempt', default='r1')
    args = parser.parse_args()
    if not args.attempt.isalnum():
        parser.error('attempt must be alphanumeric')
    plan = json.loads(args.plan.read_text())
    directory = Path(plan['output']) / 'platform' / args.attempt
    if directory.exists():
        raise ValueError('platform attempt already exists; inspect receipts, do not repeat create')
    directory.mkdir(parents=True)
    template = platform_call(['detail', '--task-id', args.template_task_id], directory / 'template-detail.json')
    verify_template(template)
    quotas = platform_call(['quotas', '--params', '{"platType":2}'], directory / 'quotas.json')
    candidates = [row for row in quotas['list'] if row.get('quotaId') == QUOTA]
    if (len(candidates) != 1 or candidates[0].get('displayName') != 'pai2_vla_xpu(vla)'
        or str(candidates[0].get('spaceId')) != '150120' or int(candidates[0].get('platType', 0)) != 2):
        raise ValueError('live quota mapping differs from the approved VLA target')
    permission = platform_call(['quota-status', '--space-id', '150120', '--quota-id', QUOTA,
                                '--pod-count', str(plan['config']['nodes'])], directory / 'priority.json')
    request = prepare_request(args.plan, template, directory, args.experiment_id)
    # The wrapper schema is preserved in receipts; use the live priority fields.
    if permission.get('willForceDowngrade') or int(permission.get('maxAllowedPriority', 0)) < request['priority']:
        raise ValueError('priority8 unavailable; no submission or silent downgrade')
    if not args.submit:
        print(json.dumps(dict(prepared=True, request=str(directory / 'create.json'),
                              nodes=request['podCount'], submitted=False), ensure_ascii=False))
        return
    # Required instructions are re-read immediately before each remote mutation.
    for path in (ROOT / 'AGENTS.md', REPO / 'AGENTS.md'):
        if not path.read_text():
            raise ValueError('missing mandatory submission instructions')
    verify(plan)
    atomic_json(directory / 'create-started.json', dict(plan_sha256=plan['plan_sha256']))
    result = platform_call(['create', '--json-file', str(directory / 'create.json')], directory / 'create-response.json')
    for task in result['tasks']:
        task_id, job_id = str(task['taskId']), str(task['externalTaskId'])
        detail = platform_call(['detail', '--task-id', task_id], directory / f'detail-{task_id}.json')
        verify_template(detail)
        if (str(detail['jobId']) != job_id or detail['podCount'] != request['podCount']
            or detail['userCommand'] != request['userCommand'] or detail['priority'] != request['priority']):
            raise ValueError('created task resources/command differ from requested contract')
        needed = platform_call(['experiment-info-need', '--json', json.dumps({'taskId': task_id})],
                               directory / f'experiment-need-{task_id}.json')
        platform_call(['experiment-info-get', '--task-id', task_id], directory / f'experiment-info-{task_id}.json')
        atomic_json(directory / f'accepted-{task_id}.json', dict(taskId=task_id, jobId=job_id,
                    status=detail['status'], experiment_info_need=needed))
        needs_info = needed if isinstance(needed, bool) else needed.get('need', needed.get('needFill', False))
        if needs_info:
            types = platform_call(['iter-types'], directory / f'iter-types-{task_id}.json')
            configured = plan['config'].get('experiment_info')
            if not configured:
                raise RuntimeError('task created, platform requires experiment declaration; configure a verified ordinary iterType/maxIter and fill this task, do not recreate')
            entries = types if isinstance(types, list) else types.get('list', [])
            selected = next((row for row in entries if row['value'] == configured['iterType']), None)
            if not selected or selected.get('needApproval') or configured['iterType'] == 6:
                raise ValueError('experiment declaration requires separate approval; task receipt retained')
            if (type(configured.get('maxIter')) is not int or configured['maxIter'] < 1
                or configured['maxIter'] > int(selected['maxIter'])):
                raise ValueError('experiment maxIter exceeds the live ordinary declaration bounds')
            info = dict(taskId=task_id, jobId=job_id, iterType=configured['iterType'], maxIter=configured['maxIter'],
                modelDesc=request['desc'], purposeDesc='实验目的与目标：比较训练后大脑与动作头在闭环导航和通用空间理解评测中的表现，'
                '保持数据子集、相机和评分口径可追溯。预期结果与验收：获得各benchmark与checkpoint的完整预测、原生指标与无效回合账本。')
            atomic_json(directory / f'experiment-fill-{task_id}.json', info)
            filled = platform_call(['experiment-info-fill', '--json-file', str(directory / f'experiment-fill-{task_id}.json')],
                                   directory / f'experiment-fill-response-{task_id}.json')
            if isinstance(filled, dict) and filled.get('needApproval'):
                raise ValueError('experiment declaration is pending approval; task receipt retained')
            saved = platform_call(['experiment-info-get', '--task-id', task_id], directory / f'experiment-filled-{task_id}.json')
            if not isinstance(saved, dict) or any(str(saved.get(key)) != str(info[key])
                for key in ('taskId', 'jobId', 'iterType', 'maxIter', 'modelDesc', 'purposeDesc')):
                raise ValueError('experiment declaration readback differs; inspect saved receipts, do not recreate')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit("DLC submission entry removed in the open-source edition; use the root GPU run scripts.")
