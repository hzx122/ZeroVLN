#!/usr/bin/env python3
"""Deploy, accept, evaluate and release owned DLC services. See workflow.md."""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import socket
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO/'vln_demo/dlc'))
from submission import platform_call, create_and_verify
from agent_models import model_choice

GPU_ROOT = Path('/data/zerovln/artifacts')
EVAL_ROOT = Path('/opt/external/user-d5121be6/SpatialOnboarding/artifacts')
TERMINAL = {'failed','stopped','cancelled','canceled','deleted','succeeded','completed','terminated','error'}
SUCCESS = {'succeeded','completed'}


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')


def detail(root, receipt):
    for attempt in range(3):
        try:
            data = platform_call(['detail','--task-id',receipt['task_id']], root/('detail-'+stamp()+'.json'))
            break
        except (OSError, ValueError, subprocess.TimeoutExpired):
            if attempt == 2:
                raise
            time.sleep(10 * (attempt+1))
    if data.get('createdBy') != 'local-user-03':
        raise ValueError('task owner mismatch')
    if str(data['taskId']) != receipt['task_id'] or data['jobId'] != receipt['job_id']:
        raise ValueError('platform task/job identity changed')
    return data


def create_once(directory):
    receipt_file = directory/'submission-receipt.json'
    if receipt_file.exists():
        return json.loads(receipt_file.read_text())
    # A timeout may still mean accepted. Never turn restart into duplicate create.
    if (directory/'create-response.json').exists():
        response = json.loads((directory/'create-response.json').read_text())
        tasks = response.get('response', {}).get('data', {}).get('tasks', [])
        if response.get('http_status', 0) not in range(200,300) or response.get('response', {}).get('code') != 2000 or len(tasks) != 1:
            raise RuntimeError('ambiguous previous create; reconcile with platform before resuming')
        receipt = dict(task_id=str(tasks[0]['taskId']),job_id=tasks[0]['externalTaskId'],submitted=True)
        save(receipt_file, receipt)
        return receipt
    # Platform anti-duplicate protection is account-scoped, even for different tasks.
    # Serialize workflow creates on this DSW; uncertain receipts still require reconciliation.
    with (EVAL_ROOT.parent/'.platform-create.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        lock.seek(0)
        previous = lock.read().strip()
        if previous:
            time.sleep(max(0, 10-(time.time()-float(previous))))
        try:
            return create_and_verify(directory, json.loads((directory/'create.json').read_text()))
        finally:
            lock.seek(0); lock.truncate(); lock.write(str(time.time())); lock.flush()


def experiment_info(root, receipt):
    destination = root/('experiment-'+receipt['task_id'])
    if destination.exists():
        return
    destination.mkdir()
    need = platform_call(['experiment-info-need','--json',json.dumps({'taskId':receipt['task_id']})],destination/'need.json')
    existing = platform_call(['experiment-info-get','--task-id',receipt['task_id']],destination/'info.json')
    if need.get('need') and not existing:
        types = platform_call(['iter-types'],destination/'iter-types.json')
        # No fake training iterations for inference/evaluation. Save a truthful draft.
        save(destination/'needs-service-category.json', dict(task_id=receipt['task_id'],
            purposeDesc='实验目的与目标：比较所选 pi-agent 模型在固定导航评测清单上的成功率与运行性能，复用标准仿真和底层控制配置；预期结果与验收：保留独立回合会话、图片、轨迹、指标和可核验归档，完成后回收本次服务。',
            reason='推理/导航评估没有训练迭代；需要核实服务档位，不自动发起自定义审批。', available=types))


def wait_ready(root, receipt, status_file):
    local_binding(receipt['job_id'])
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        try:
            status = json.loads(status_file.read_text())
            age = time.time()-datetime.fromisoformat(status['updated_at']).timestamp()
            if status['hostname'] != socket.gethostname():
                raise RuntimeError('supervisor belongs to another Pod')
            if status['state'] in ('failed', 'stopped'):
                raise RuntimeError('supervisor ended before acceptance')
            if status['state'] == 'ready' and 0 <= age < 120:
                return status
        except (OSError, ValueError, KeyError):
            pass
        save(root/'status.json', dict(state='waiting_for_service', task=receipt, updated_at=stamp()))
        time.sleep(10)
    raise TimeoutError('local service did not become ready')


def run_command(command, log, timeout):
    with log.open('x') as out:
        subprocess.run([str(c) for c in command], cwd=REPO, stdout=out, stderr=subprocess.STDOUT,
                       check=True, timeout=timeout, env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))


def validate_plan(plan):
    if (plan.get('deployment') != 'single-dlc' or plan.get('coordinator') != 'server'
            or any(key in plan for key in ('cpu_base', 'cpu_submission', 'renderer', 'render_acceptance'))):
        raise ValueError('single-dlc plans run evaluation inside their own 16-GPU Pod; create a new plan')


def local_binding(job_id):
    hostname = socket.gethostname()
    if not hostname.startswith(job_id + '-master-'):
        raise ValueError('evaluation must run inside its owned GPU Pod')
    return dict(job_id=job_id, pod_id=hostname, hosts=['127.0.0.1'])


def assert_independent_request(directory, plan):
    validate_plan(plan)
    request = json.loads((directory/'create.json').read_text())
    if request['resourceConfig']['gpuSource']['gpu'] != 16 or request['podCount'] != 1:
        raise ValueError('independent service must allocate one complete 16-GPU node')
    if request['workDir'] != str(REPO) or request['priority'] != 8:
        raise ValueError('service must use current repository and priority 8')
    profile = 'next' if model_choice(plan['model']).get('gateway') else plan['model']
    if ('export VLN_AGENT_MODEL_PROFILE='+profile+'\n') not in request['userCommand']:
        raise ValueError('service model profile mismatch')
    if str(REPO/'vln_demo/dlc/entry.sh') not in request['userCommand']:
        raise ValueError('full render/controller deployment entry is required')


def worker(plan_file):
    plan = json.loads(plan_file.read_text())
    validate_plan(plan)
    root = plan_file.parent
    if (root/'cancelled.json').exists():
        raise ValueError('this workflow was cancelled; use a new independent service plan')
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            save(root/'status.json',dict(state='resuming',updated_at=stamp(),hostname=socket.gethostname()))
            execute(root,plan)
        except Exception as exc:
            save(root/'status.json',dict(state='failed',error=f'{type(exc).__name__}: {exc}',updated_at=stamp()))
            raise


def prepare_service(root, plan):
    gpu = Path(plan['gpu_submission'])
    if not gpu.exists():
        service = load_module('service_submit', REPO/'vln_demo/dlc/submit.py')
        service.prepare(Path(plan['gpu_base']), 'VLM+渲染+控制台服务部署-'+root.name,
                        gpu, REPO, agent_profile=plan['model'] if not model_choice(plan['model']).get('gateway') else 'next',
                        workflow_plan=root/'plan.json')
    assert_independent_request(gpu, plan)
    return gpu


def server_worker(plan_file):
    plan = json.loads(plan_file.read_text())
    if plan.get('coordinator') != 'server':
        raise ValueError('server coordinator requires a server-owned plan')
    receipt_file = Path(plan['gpu_submission'])/'submission-receipt.json'
    # The Pod can start before the create request returns to its submitter.
    for _ in range(120):
        if receipt_file.exists():
            break
        time.sleep(5)
    receipt = json.loads(receipt_file.read_text())
    if not socket.gethostname().startswith(receipt['job_id']+'-master-'):
        raise ValueError('server coordinator must run inside its owned GPU Pod')
    save(plan_file.parent/'coordinator.json', dict(pid=os.getpid(), hostname=socket.gethostname(),
         mode='server', plan=str(plan_file), log=str(Path(os.environ['VLN_RUN_DIR'])/'logs/evaluation.log')))
    worker(plan_file)


def execute(root, plan):
    spec = model_choice(plan['model'])
    gpu = Path(plan['gpu_submission'])
    assert_independent_request(gpu, plan)
    receipt = json.loads((gpu/'submission-receipt.json').read_text())
    render = dict(receipt,run=str(gpu/'run'))
    status_file = gpu/'run/status.json'
    wait_ready(root,receipt,status_file)
    binding = local_binding(receipt['job_id'])
    render_binding = binding  # renderer/controller/agent model belong to this same Pod
    from launch import resolve_services
    services = resolve_services(render['job_id'],600,render_binding,str(Path(render['run'])/'status.json'),
                                plan['datasets'],require_next=spec.get('gateway', False) or plan['model']=='next')
    config = dict(datasets=plan['datasets'], workers=plan['workers'], timeout=0,max_tool_calls=100,
                  server_job_id=render['job_id'],server_status_file=str(Path(render['run'])/'status.json'),
                  agent_profile=plan['model'],languages='en,en-US,en-IN',limit_per_dataset=0)
    if plan.get('manifest'):
        config['manifest'] = plan['manifest']
    else:
        config.update(evaluation_ladder=str(HERE/'eval_sets/english-val-unseen-v1'),evaluation_size=plan['size'])
    acceptance = root/'acceptance'
    acceptance.mkdir(exist_ok=True)
    save(root/'status.json',dict(state='accepting',model=plan['model'],hostname=socket.gethostname(),updated_at=stamp()))
    if not spec.get('gateway') and plan['model'] != 'next':
        model_status = gpu/'run/agent-model/status.json'
        config.update(agent_server_job_id=receipt['job_id'],agent_server_status_file=str(model_status),
                      agent_server_binding=binding)
        from launch import resolve_agent_service
        services.update(resolve_agent_service(config,600))
        probe_dir = acceptance/'model'
        if not (probe_dir/'receipt.json').exists() or not json.loads((probe_dir/'receipt.json').read_text()).get('passed'):
            if probe_dir.exists():
                probe_dir.rename(acceptance/('model-failed-'+stamp()))
            run_command([sys.executable,REPO/'vln_demo/vllm/sglang_probe.py','--profile',plan['model'],
                         '--url',services['model_url'].removesuffix('/v1'),'--status-file',model_status,
                         '--output-dir',probe_dir],acceptance/('model-'+stamp()+'.log'),2400)
        config['agent_acceptance_file'] = str(probe_dir/'receipt.json')
        from agent_models import validate_acceptance
        validate_acceptance(config)
    check = acceptance/'stack'
    if not (check/'receipt.json').exists() or not json.loads((check/'receipt.json').read_text()).get('passed'):
        if check.exists(): check.rename(acceptance/('stack-failed-'+stamp()))
        run_command([sys.executable,REPO/'vln_demo/dlc/verify_runtime.py','--run-dir',render['run'],
                     '--output-dir',check,'--expected-job-id',render['job_id'],
                     '--host',services['host']],acceptance/('stack-'+stamp()+'.log'),7200)
    from follow_submission import verify_receipt
    verify_receipt(check/'receipt.json',render['job_id'],render['run'])
    # This test uses private controllers and real tools. It never switches the shared web episode.
    motion = acceptance/'motion'
    if plan['datasets'] != 'objectnav' and not (motion/'result.json').exists():
        if motion.exists(): motion.rename(acceptance/('motion-failed-'+stamp()))
        run_command([sys.executable,HERE/'benchmarks/check_concurrent_controllers.py',
                     '--render-url',services['render_url'],'--controller-model-url',services['controller_model_url'],
                     '--output',motion],acceptance/('motion-'+stamp()+'.log'),2400)
    # Verify chosen SDK/provider, including closed-source auth, before the local batch.
    if not (acceptance/'provider-passed.json').exists():
        raw = acceptance/('provider-'+stamp()+'.json')
        url = '' if spec.get('gateway') else services['model_url']
        run_command(['/opt/external/user-d5121be6/tools/node/bin/node',HERE/'probe_model.mjs',
                     spec['provider'],spec['model_id'],raw,url],acceptance/('provider-'+stamp()+'.log'),660)
        save(acceptance/'provider-passed.json',dict(passed=True,model=plan['model'],evidence=str(raw)))
    save(root/'services.json',services)
    config.update(server_binding=binding, run_dir=str(Path(plan['evaluation_dir'])/'episodes-run'))
    from config import validate_config
    # Runtime-only values are bound here, never supplied by a remote CPU task.
    validated = validate_config({k: v for k, v in config.items()
                                if k not in ('server_binding', 'agent_server_binding', 'run_dir')})
    config = dict(validated, **{k: config[k] for k in ('server_binding', 'run_dir')})
    if not spec.get('gateway') and plan['model'] != 'next':
        config['agent_server_binding'] = binding
    run_local_evaluation(root, plan, receipt, config, services)


def run_local_evaluation(root, plan, receipt, config, services):
    evaluation = Path(plan['evaluation_dir'])
    evaluation.mkdir(parents=True, exist_ok=True)
    config_dir = evaluation/'config-version'
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir/'batch.json'
    save(config_file, config)
    save(root/'evaluation.json', dict(task=receipt, directory=str(evaluation), mode='same-pod'))
    save(root/'status.json', dict(state='evaluating', task=receipt, services=services, updated_at=stamp()))
    passed = False
    error = None
    try:
        run_command([sys.executable, HERE/'launch.py', '--config', config_file],
                    evaluation/('launcher-'+stamp()+'.log'), None)
        passed = (Path(config['run_dir'])/'_SUCCESS.json').is_file()
        if not passed:
            raise RuntimeError('batch exited without verified archive completion marker')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save(root/'evaluation-ended.json', dict(passed=passed, error=error, task=receipt,
             hostname=socket.gethostname(), evaluation_dir=str(evaluation), updated_at=stamp()))
        # The owning supervisor verifies this receipt, cleans all children and exits.
        save(root/'status.json', dict(state='evaluation_complete' if passed else 'evaluation_failed',
             task=receipt, updated_at=stamp()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='next',help='next / qwen36-35b / qwen38-27b / qwen35-122b / anonymous-data/MODEL_ID')
    parser.add_argument('--datasets',choices=('r2r','rxr','r2r,rxr','objectnav'),default='r2r,rxr')
    parser.add_argument('--size',choices=('100','200','500','1000','full','ladder'),default='ladder')
    parser.add_argument('--manifest',type=Path,help='ObjectNav repository manifest')
    parser.add_argument('--workers',type=int,default=32)
    parser.add_argument('--gpu-base',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--submit',action='store_true',help='start persistent coordinator; default writes reviewable plan only')
    parser.add_argument('--resume',type=Path,help='resume saved plan without duplicate submissions')
    parser.add_argument('--server-worker',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.server_worker:
        server_worker(args.server_worker.resolve(strict=True))
        return
    if args.resume:
        plan_file=args.resume.resolve(strict=True)
    else:
        spec=model_choice(args.model)
        if not args.output_dir or not args.gpu_base or not 1 <= args.workers <= 32:
            parser.error('--output-dir, --gpu-base and workers 1..32 required')
        root=args.output_dir.resolve()
        if root.exists() or not str(root).startswith(('/data/storage/','/data/storage/')):
            parser.error('output-dir must be a fresh CPFS directory')
        if args.datasets=='objectnav':
            if not args.manifest or not args.manifest.resolve().is_relative_to(REPO):
                parser.error('ObjectNav requires a repository-owned manifest')
        elif args.manifest:
            parser.error('R2R/RxR use the repository locked size selections')
        if (EVAL_ROOT/root.name).exists() or (GPU_ROOT/root.name/'service').exists():
            parser.error('run name already used for service/evaluation artifacts; choose a unique output directory name')
        root.mkdir(parents=True)
        plan=dict(model=args.model,datasets=args.datasets,size=args.size,workers=args.workers,
                  gpu_base=str(args.gpu_base.resolve(strict=True)),
                  gpu_submission=str(GPU_ROOT/root.name/'service'),
                  evaluation_dir=str(EVAL_ROOT/root.name),deployment='single-dlc',coordinator='server',
                  manifest=str(args.manifest.resolve(strict=True)) if args.manifest else None, repo=str(REPO))
        plan_file=root/'plan.json'
        save(plan_file,plan)
    validate_plan(json.loads(plan_file.read_text()))
    if args.submit:
        plan = json.loads(plan_file.read_text())
        root = plan_file.parent
        gpu = prepare_service(root, plan)
        receipt = create_once(gpu)
        experiment_info(root, receipt)
        if (root/'evaluation-ended.json').exists():
            print(json.dumps(dict(task=receipt, evaluation=json.loads((root/'evaluation-ended.json').read_text()),
                                  platform_state=detail(root,receipt)['status'])))
        else:
            print(json.dumps(dict(submitted=True,coordinator='server',task=receipt,plan=str(plan_file))))
    else:
        print(json.dumps(dict(started=False,plan=str(plan_file))))


if __name__=='__main__':
    main()
