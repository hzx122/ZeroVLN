"""Reuse the primitive ReAct evaluator on the main experiment's closed pools."""
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

from action_head_benchmark_suite import REPO, SWEEP, atomic_json, digest
from dualvln_benchmark_suite import aggregate_rows


def contract(plan, stage):
    ckpt = next(c for c in plan['config']['checkpoints'] if c['name'] == stage['checkpoint'])
    return dict(schema='primitive_main_eval_v1', plan_sha256=plan['plan_sha256'],
        benchmark=stage['benchmark'], checkpoint=ckpt['identity'], seed=plan['config']['seed'],
        camera=ckpt['observation_profiles'][stage['benchmark']],
        max_tool_calls=plan['config']['max_tool_calls'][stage['benchmark']],
        one_tool_call_per_turn=True, max_distance_m=ckpt['max_distance_m'],
        repeats=1, scoring_mode='merged_batch', trajectory_map=True,
        sampling=dict(temperature=.7, top_p=.95, max_tokens=256, enable_thinking=False),
        navmesh_root=plan['config'].get('navmesh_root'))


def evaluator(plan, stage, gym_url, brain_url):
    sys.path[:0] = [str(SWEEP), str(REPO / 'verl')]
    import native_tool_closedloop_eval as N
    from v6_vlnce_eval import validate_reset, score_native
    from navigation_baselines import validate_reset as sage_reset
    from vln_gym.observation_profile import assert_profile
    N.E.MODEL = 'default'
    ckpt = next(c for c in plan['config']['checkpoints'] if c['name'] == stage['checkpoint'])
    sage = stage['benchmark'] == 'sage-3d'
    budget = plan['config']['max_tool_calls'][stage['benchmark']]
    base = N.PrimitiveEvaluator(SimpleNamespace(bench_url=gym_url, vllm_url=brain_url, timeout=600,
        max_tool_calls=budget, max_turns=budget, keep_last_images=8, trajectory_map=True,
        one_tool_call_per_turn=True, stop_on_environment_termination=True,
        success_metric='official' if sage else 'native'))
    original_reset, original_step = base.bench.reset, base.bench.step_tool
    state = {}
    dataset = None
    if sage:
        from sage.datasets import SageSet
        dataset = SageSet('official_sage_bench_v1')

    def reset(**kwargs):
        obs, info = original_reset(**kwargs)
        case = state['case']
        assert_profile(info.get('observation_profile') or {}, ckpt['observation_profiles'][stage['benchmark']])
        if sage:
            task = next(t for t in dataset.tasks(case['scene']) if str(t.task_id) == str(case['episode_id']))
            sage_reset(case, task, obs, info)
        else:
            validate_reset(case, obs, info)
            if not math.isfinite(float(info.get('distance_to_goal', float('nan')))):
                raise ValueError('nonfinite native distance at reset, before model actions')
        state.update(positions=[], info=info)
        return obs, info

    def step(name, arguments=None):
        obs, reward, terminated, truncated, info = original_step(name, arguments)
        state['info'] = info
        if sage:
            from sage import benchmark_protocol
            state['positions'].append(info['official_object_success_detail']['official_position'])
            terminated = terminated or benchmark_protocol.auto_stop(info)
        return obs, reward, terminated, truncated, info

    base.bench.reset, base.bench.step_tool = reset, step

    def rollout(case, seed):
        state['case'] = case
        row = base.rollout(case, seed)
        row['slow_calls'] = len(row['turns'])
        if row['calls_used'] != row['slow_calls']:
            raise ValueError('primitive budget must count one call per slow decision')
        if sage:
            from sage import benchmark_protocol
            length, optimum = benchmark_protocol.trajectory_metrics(case['episode_id'], state['positions'])
            row.update(decision_positions=state['positions'], path_length_m=length, shortest_path_m=optimum,
                spl=float(row['success']) * (optimum / max(optimum, length) if optimum else 1.))
        else:
            row = score_native(row, state['info'], scoring_mode='merged_batch')
        return row
    return SimpleNamespace(rollout=rollout, close=base.bench.close)


def run_pool(plan, stage, cases, output, gym_urls, brain_urls):
    from v6_vlnce_eval import evaluate_attempts, read_result, result_path
    sys.path.insert(0, str(REPO / 'verl/scripts/train/action_head_bootstrap'))
    from case_scheduler import SceneCaseScheduler
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = contract(plan, stage)
    atomic_json(output / 'contract.json', protocol)
    scheduler = SceneCaseScheduler(cases, replicas=1, max_scenes=len(gym_urls), max_cases=len(gym_urls))
    lock, counts, started = threading.Lock(), dict(executed=0, valid=0, success=0), time.monotonic()
    def actor(index, gym_url):
        local = evaluator(plan, stage, gym_url, brain_urls[index % len(brain_urls)])
        try:
            while (case := scheduler.acquire()) is not None:
                try:
                    row = evaluate_attempts(local, case, protocol, output, worker_index=index, actor_index=index)
                    with lock:
                        counts['executed'] += 1
                        counts['valid'] += int(bool(row['evaluation_valid']))
                        counts['success'] += int(bool(row['evaluation_valid'] and row['success']))
                        atomic_json(output / 'worker_0/parallel_progress.json', dict(counts,
                            scheduler=scheduler.stats(), elapsed_s=time.monotonic()-started))
                    print(json.dumps(dict(event='primitive_case', logical_id=case['logical_id'],
                        valid=row['evaluation_valid'], success=row['success'], calls_used=row.get('calls_used'))), flush=True)
                finally:
                    scheduler.release(case)
        except BaseException:
            scheduler.cancel()
            raise
        finally:
            local.close()
    with ThreadPoolExecutor(max_workers=len(gym_urls)) as pool:
        futures = [pool.submit(actor, i, url) for i, url in enumerate(gym_urls)]
        for future in futures:
            future.result()
    rows = [read_result(result_path(output, c)) for c in cases]
    result = aggregate_rows(plan, stage, rows, cases, protocol)
    atomic_json(output / 'summary.json', result)
    return result
