"""Preflight an existing resident, then launch the existing collection owner.

Run only from a pinned container-local Git checkout. The resident entry,
heartbeat, lease/idle controller and collection engine are reused unchanged.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare_rule_teacher_three_source import REPO, sha256, verify_plan
from rule_teacher_training_handoff import load, save


def identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    if fields[0] == 'Z':
        raise ValueError('process is a zombie')
    return fields[19]


def active_business_processes():
    """Share the resident's existing ownership gate with a deferred starter."""
    forbidden = ('run_rule_teacher_repair_owner.py', 'supervise_rule_teacher_collection.py',
                 'v6_rl_dlc_worker.py', 'rule_teacher_repeat_companion.py')
    result = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc/'cmdline').read_bytes().split(b'\0')
            if any(Path(arg.decode(errors='replace')).name in forbidden for arg in argv if arg):
                result.append(int(proc.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return result


ENGINE_PROBE = '''import json,sys
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.engine.arg_utils import AsyncEngineArgs
from action_head.serve.native_r3_capture import _validate_config
parser=make_arg_parser(FlexibleArgumentParser())
args=AsyncEngineArgs.from_cli_args(parser.parse_args(json.loads(sys.argv[1])))
config=args.create_engine_config()
_validate_config(config)
print(json.dumps(dict(enforce_eager=config.model_config.enforce_eager,
 gpu_memory_utilization=config.cache_config.gpu_memory_utilization,
 max_num_seqs=config.scheduler_config.max_num_seqs,
 max_model_len=config.model_config.max_model_len,
 prefix_cache=config.cache_config.enable_prefix_caching,
 mamba_cache_mode=config.cache_config.mamba_cache_mode,
 return_routes=config.model_config.enable_return_routed_experts,
 async_scheduling=config.scheduler_config.async_scheduling)))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--debug-output',type=Path,required=True)
    p.add_argument('--source',choices=('sage','r2r','rxr'),required=True)
    p.add_argument('--rank',type=int,default=0)
    p.add_argument('--shards',type=int,default=4)
    p.add_argument('--collectors',type=int,default=None)
    p.add_argument('--actors',type=int,default=None,
                   help='physical Gym servers per collector; default SAGE=64, R2R/RxR=16 (r166)')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--max-calls',type=int,default=100)
    p.add_argument('--http-concurrency',type=int,default=128)
    p.add_argument('--resume-manifest',type=Path)
    p.add_argument('--prefix-cache', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--stdout-pid',type=int,default=16)
    a=p.parse_args()
    if a.collectors is None:
        a.collectors = 8 if a.source == 'sage' else 7
    if a.actors is None:
        a.actors = 64 if a.source == 'sage' else 16
    sys.path[:0]=[str(REPO/'scripts/eval'),str(REPO/'verl'),str(REPO/'vln_demo/src')]
    from unified_closedloop_eval import git_runtime
    git=git_runtime(require_container=True)
    if a.output.exists():
        raise ValueError('fresh attempt required; inspect existing receipts instead of replay')
    status=load(a.debug_output/'debug-status.json')
    controller=int(status['pid']);ticks=identity(controller)
    if (status['hostname']!=os.uname().nodename or status['mode']!='idle'
            or status['transition']!='ready' or time.time()-status['heartbeat']>45):
        raise ValueError('resident must be freshly confirmed idle/ready')
    # Reject live business owners before even preparing another attempt.
    busy = active_business_processes()
    if busy:
        raise ValueError('existing business processes: '+str(busy))
    command=[sys.executable,'-B',str(REPO/'scripts/train/prepare_rule_teacher_v8_collection.py'),
        '--output',str(a.output),'--source',a.source,'--rank',str(a.rank),'--shards',str(a.shards),
        '--collectors',str(a.collectors),'--actors',str(a.actors),'--limit',str(a.limit),
        '--max-calls',str(a.max_calls),'--depth-batch-size','256','--http-concurrency',str(a.http_concurrency),
        '--gpu-memory-utilization','0.9']
    if a.resume_manifest:
        command += ['--resume-manifest', str(a.resume_manifest)]
    if a.prefix_cache:
        command += ['--prefix-cache']
    subprocess.run(command,cwd=REPO,check=True)
    plan=load(a.output/'launch.json');source=load(a.output/'source-plan.json')
    verify_plan(source)
    entries=plan['services']+plan['collectors']
    names=[e['name'] for e in entries]
    ports=[e['port'] for e in entries if 'port' in e]
    assert len(names)==len(set(names)) and len(ports)==len(set(ports))
    assert len(plan['collectors'])==a.collectors
    used=set()
    brains=[e for e in plan['services'] if e['name'].startswith('brain-')]
    for brain in brains:
        used.update(map(int,brain['environment']['CUDA_VISIBLE_DEVICES'].split(',')))
    brain_gpus = set(used)
    for entry in plan['services']:
        if entry['name'].startswith('sage-render'):
            gpu=int(entry['command'][entry['command'].index('--gpus')+1])
            # The verified legacy layout shares GPU7 between two renderers
            # while GPU5 is isolated. Renderers must never share a brain GPU.
            assert gpu not in brain_gpus
            used.add(gpu)
        if '-gym-' in entry['name']:
            assert entry['command'][entry['command'].index('--max_steps')+1]=='0'
    assert 5 not in used
    for entry in plan['collectors']:
        argv=entry['command']
        for key,value in {'--depth-batch-size':'256','--http-concurrency':str(a.http_concurrency),
                '--candidates':'2','--max-calls':str(a.max_calls),'--route-prefix-steps':'16',
                '--sampling-scheduler':'v6_depth'}.items():
            assert argv[argv.index(key)+1]==value
    # Parse every final collector command before allocating any GPU. Building
    # argv and testing the scheduler alone do not exercise CLI choices.
    import rule_teacher_pilot as pilot
    import mp3d_rule_teacher_pilot as native
    cli_proof=[]
    for entry in plan['collectors']:
        argv=entry['command']
        pos=next(i for i,v in enumerate(argv) if v.endswith('_teacher_pilot.py'))
        parser=pilot.parser() if argv[pos].endswith('/rule_teacher_pilot.py') else native.parser()
        if argv[pos].endswith('/mp3d_rule_teacher_pilot.py'):
            pilot.collection_protocol_arguments(parser)
        parsed=parser.parse_args(argv[pos+1:])
        pilot.configure_collection_protocol(parsed)
        pilot.actor_urls(parsed)
        cli_proof.append(dict(name=entry['name'],batch=parsed.depth_batch_size,
            http=parsed.http_concurrency,candidates=parsed.candidates,version=parsed.action_head_as_tool_version))
    save(a.output/'collector-cli-preflight.json',cli_proof)
    # Configuration parsing uses the actual server interpreter and environment.
    # One lane suffices for identical engine arguments; all GPU/port/cache
    # allocations are checked separately above and preserved in launch.json.
    brain=brains[0];argv=brain['command'];pos=argv.index('-m')
    environment={**os.environ,**brain['environment']}
    with (a.output/'vllm-config-preflight.log').open('xb') as stream:
        subprocess.run([argv[pos-1],'-B','-c',ENGINE_PROBE,json.dumps(argv[pos+2:])],
            cwd=REPO,env=environment,check=True,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
    lines=(a.output/'vllm-config-preflight.log').read_text().splitlines()
    engine=json.loads(next(line for line in reversed(lines) if line.startswith('{')))
    assert engine['gpu_memory_utilization']==.9 and not engine['enforce_eager']
    assert engine['prefix_cache'] == a.prefix_cache and engine['return_routes']
    if a.prefix_cache:
        assert engine['mamba_cache_mode'] == 'align'
    assert all(e['environment']['RULE_TEACHER_V8_PREFIX_CACHE'] == ('1' if a.prefix_cache else '0')
               for e in plan['collectors'])
    save(a.output/'vllm-config-preflight.json',engine)
    from action_head.protocol_v8 import PROMPT_SHA256
    assert PROMPT_SHA256=='1e557dd81f833870038ca3e10e7bba1ba7cb1f1c6b01e11bd8b092c661cce061'
    modules={name:importlib.import_module(name).__file__ for name in (
        'rule_teacher_pilot','rule_route_execution','rule_match_policy','action_head.runtime')}
    assert all(Path(path).is_relative_to(REPO) for path in modules.values())
    assert identity(controller)==ticks
    save(a.output/'preflight.json',dict(git=git,hostname=os.uname().nodename,
        controller_pid=controller,start_ticks=ticks,prompt_sha256=PROMPT_SHA256,modules=modules,
        collectors=a.collectors,used_gpus=sorted(used),spare_gpus=sorted(set(range(16))-used)))
    control=REPO/'scripts/train/run_dlc_debug_reservation.py'
    io=REPO/'scripts/train/dlc_control_io_retry.py'
    config=dict(output=str(a.output/'owner'),source_root=str(REPO),hostname=os.uname().nodename,
        stdout_pid=a.stdout_pid,resident_identities={str(controller):ticks},control=str(control),
        control_sha256=sha256(control),io_retry=str(io),io_retry_sha256=sha256(io),
        debug_output=str(a.debug_output),spare_gpus=sorted(set(range(16))-used),
        runtime_root=str(Path('/tmp/v8c')/a.output.name),launch_plan=str(a.output/'launch.json'))
    save(a.output/'owner-config.json',config)
    with (a.output/'owner-stdout.log').open('xb') as stream:
        child=subprocess.Popen([sys.executable,'-B',str(REPO/'scripts/train/run_rule_teacher_repair_owner.py'),
            '--config',str(a.output/'owner-config.json')],cwd=REPO,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(pid=child.pid,start_ticks=identity(child.pid),git=git,time=time.time(),output=str(a.output))
    save(a.output/'owner-launch.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__': main()
