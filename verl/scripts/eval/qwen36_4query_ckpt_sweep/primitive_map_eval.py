"""Repeated primitive ReAct evaluation using the existing rollout and scheduler."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import native_tool_closedloop_eval as N
from common import atomic_json, load_subset, sha256_file
from closedloop_eval import result_path, deterministic_env_seed, pass_at_k_block
from action_head import primitive_trajectory_map as TM
from sage.goal_region import official_success_contract_identity

sys.path.insert(0, str(N.REPO / 'verl/scripts/train/action_head_bootstrap'))
from case_scheduler import SceneCaseScheduler


def run_contract(args):
    prompt = TM.system_prompt(N.P.agent_system_prompt(tool_specs=N.A.get_tool_profile('primitive').tool_specs), args.trajectory_map)
    return dict(schema='primitive_map_test100_v1', **TM.identity(prompt, args.trajectory_map),
                model=N.E.MODEL, model_path='/opt/external/user-e7a01947/Model/Qwen3.6-35B-A3B',
                subset_sha256=sha256_file(args.subset_manifest), repeats=args.repeats,
                max_turns=args.max_turns, max_tool_calls=args.max_tool_calls,
                keep_last_images=args.keep_last_images, success_contract=official_success_contract_identity(),
                invalid_tool_call_policy=N.A.invalid_tool_call_policy(),
                brain_sampling=dict(enable_thinking=False, temperature=0.7, top_p=0.95,
                                    max_tokens=256, seed='omitted', query_tail=False),
                evaluator_sha256=sha256_file(__file__),
                native_evaluator_sha256=sha256_file(N.__file__),
                map_adapter_sha256=sha256_file(TM.__file__),
                agent_tools_sha256=sha256_file(N.A.__file__))


def aggregate(output, cases, contract):
    rows = []
    for case in cases:
        for repeat in range(1, contract['repeats'] + 1):
            p = result_path(output, case, repeat)
            if p.exists():
                row = json.loads(p.read_text())
                if row.get('run_contract') != contract:
                    raise ValueError(f'rollout contract mismatch: {p}')
                if (row.get('scene'), row.get('episode_id'), row.get('repeat_index')) != (case['scene'], case['episode_id'], repeat):
                    raise ValueError(f'rollout identity mismatch: {p}')
                rows.append(row)
    valid = [r for r in rows if r.get('evaluation_valid')]
    pk = pass_at_k_block(rows, cases, contract['repeats'])
    metrics = N.metric_block(rows)
    summary = dict(schema='primitive_map_test100_summary_v1', run_contract=contract,
                   expected_cases=len(cases), expected_runs=len(cases)*contract['repeats'],
                   completed_runs=len(rows), valid_runs=len(valid), completed_cases=pk['complete_cases'],
                   evaluation_valid=len(valid)==len(cases)*contract['repeats'] and pk['complete_cases']==len(cases),
                   overall_micro=metrics, pass_at_1=metrics['success_rate'], pass_at_k=pk)
    atomic_json(output/'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subset-manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--vllm-url', default='http://127.0.0.1:28899')
    p.add_argument('--bench-urls', default='')
    p.add_argument('--repeats', type=int, default=8)
    p.add_argument('--max-turns', type=int, default=50)
    p.add_argument('--max-tool-calls', type=int, default=50)
    p.add_argument('--keep-last-images', type=int, default=8)
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--retries', type=int, default=3)
    p.add_argument('--trajectory-map', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--aggregate-only', action='store_true')
    args = p.parse_args()
    if min(args.repeats,args.max_turns,args.max_tool_calls,args.keep_last_images,args.retries) < 1:
        p.error('positive budgets and repeats required')
    if N.A.invalid_tool_call_policy() != 'feedback':
        p.error('this protocol requires feedback/retry on invalid tool calls')
    _, cases = load_subset(Path(args.subset_manifest))
    if len(cases)!=100 or sha256_file(args.subset_manifest)!='2a4b975c72915634d4f3ef58da98f519b845f9c962f48a50be9f37624722aff9':
        p.error('requires the immutable official test100 used for v4.11/v6')
    output = Path(args.output)
    contract = run_contract(args)
    cp = output/'run_contract.json'
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise ValueError('resume protocol mismatch; use a new output directory')
    if not cp.exists() and any((output/'rollouts').glob('**/*.json')):
        raise ValueError('refusing unaudited existing rollouts')
    atomic_json(cp, contract)
    if args.preflight_only:
        print(json.dumps(contract,ensure_ascii=False,indent=2)); return
    if args.aggregate_only:
        result = aggregate(output,cases,contract)
        print(json.dumps(result,ensure_ascii=False,indent=2)); return
    urls = [x.strip() for x in args.bench_urls.split(',') if x.strip()]
    if not urls or len(urls)!=len(set(urls)):
        p.error('distinct Gym URLs required')
    tasks = [{**c,'_repeat':r} for c in cases for r in range(1,args.repeats+1)]
    scheduler = SceneCaseScheduler(tasks, replicas=3, max_scenes=48,max_cases=64)
    counters = dict(executed=0,resumed=0)
    lock = threading.Lock()
    started = time.monotonic()

    def actor(url):
        evaluator = N.PrimitiveEvaluator(SimpleNamespace(**vars(args),bench_url=url,success_metric='official'))
        try:
            while (task := scheduler.acquire()) is not None:
                try:
                    repeat=task['_repeat'];case={k:v for k,v in task.items() if k!='_repeat'}
                    path=result_path(output,case,repeat)
                    row=json.loads(path.read_text()) if path.exists() else None
                    if row and row.get('run_contract')!=contract:
                        raise ValueError(f'resume row mismatch: {path}')
                    cached=bool(row and row.get('evaluation_valid'))
                    if not cached:
                        for attempt in range(1,args.retries+1):
                            try:
                                row=evaluator.rollout(case,deterministic_env_seed(case))
                            except Exception as exc:
                                row=dict(**case,evaluation_valid=False,success=False,infra_error=repr(exc))
                            row.update(run_contract=contract,repeat_index=repeat,repeats_per_case=args.repeats,attempt=attempt)
                            if row.get('evaluation_valid'):break
                        atomic_json(path,row)
                    with lock:
                        counters['resumed' if cached else 'executed']+=1
                        atomic_json(output/'parallel_progress.json',dict(**counters,elapsed_s=time.monotonic()-started,scheduler=scheduler.stats()))
                    print(f'[primitive-map] {case["episode_id"]} repeat={repeat} cached={cached} valid={row.get("evaluation_valid")} success={row.get("success")}',flush=True)
                finally:
                    scheduler.release(task)
        except BaseException:
            scheduler.cancel();raise
        finally:
            evaluator.bench.close()
    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        futures=[pool.submit(actor,url) for url in urls]
        for f in futures:f.result()
    summary=aggregate(output,cases,contract)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    if not summary['evaluation_valid']:raise SystemExit(4)


if __name__=='__main__':
    main()
