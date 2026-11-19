"""Official DualVLN adapter for the existing frozen benchmark-major suite."""
from __future__ import annotations

import copy
import fcntl
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import traceback
import uuid
from urllib.request import Request, ProxyHandler, build_opener

import numpy as np
from PIL import Image

from action_head_benchmark_suite import REPO, ROOT, SWEEP, atomic_json, digest, sha256_file

DUAL_PYTHON = Path('/workspace/envs/dualvln_eval/bin/python')


def environment_worker(connection, model_path, depth_path):
    """Check the real spawn child's imports without loading the navigation weights."""
    try:
        import importlib.metadata
        if Path(sys.prefix) != DUAL_PYTHON.parent.parent:
            raise ValueError('DualVLN collector and spawn child must both use dualvln_eval')
        requirements = REPO / 'scripts/eval/dualvln_eval_requirements.txt'
        versions = {}
        for line in requirements.read_text().splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            name, expected = line.strip().split('==')
            actual = importlib.metadata.version(name)
            if actual != expected:
                raise ValueError(f'DualVLN dependency mismatch: {name}={actual}, expected {expected}')
            versions[name] = actual
        sys.path[:0] = [str(REPO / 'third_party/InternNav'),
                       str(REPO / 'third_party/scaling_on_scales')]
        import torch
        import transformers
        import diffusers
        # PPU xformers probes the GPU at import. This CPU-only check exercises
        # the policy's SDPA path; production workers retain their native imports.
        from diffusers.utils import import_utils
        import_utils._xformers_available = False
        sys.modules['xformers'] = None  # Depth-Anything also has an optional CPU fallback.
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ModelConfig
        from internnav.model.basemodel.internvla_n1.internvla_n1_policy import InternVLAN1Net
        from internnav.model.basemodel.internvla_n1.nextdit_crossattn_traj import NextDiTCrossAttn
        from internnav.model.encoder.depth_anything.depth_anything_v2.dpt import DepthAnythingV2
        for module in (transformers, diffusers):
            if not Path(module.__file__).is_relative_to(Path(sys.prefix)):
                raise ValueError('DualVLN dependency shadowed by parent sys.path: ' + module.__file__)
        config = InternVLAN1ModelConfig.from_pretrained(model_path, local_files_only=True)
        processor = transformers.AutoProcessor.from_pretrained(model_path, local_files_only=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(2, device='cpu')
        if not Path(depth_path, 'depth_anything_v2_metric_hypersim_vits.pth').is_file():
            raise ValueError('official local depth weights missing')
        if torch.cuda.is_initialized():
            raise ValueError('CPU environment check unexpectedly initialized CUDA')
        connection.send(dict(ok=True, python=sys.executable, prefix=sys.prefix,
            versions=versions, torch=torch.__version__, torch_file=torch.__file__,
            transformers_file=transformers.__file__, diffusers_file=diffusers.__file__,
            model_type=config.model_type, processor=type(processor).__name__,
            tokenizer=type(tokenizer).__name__, native_policy=InternVLAN1Net.__name__,
            trajectory_decoder=NextDiTCrossAttn.__name__, depth_encoder=DepthAnythingV2.__name__,
            requirements_sha256=sha256_file(requirements), cuda_initialized=False,
            cpu_probe_disabled_optional_xformers=True))
    except Exception as exc:
        connection.send(dict(ok=False, error=repr(exc), traceback=traceback.format_exc()))
    finally:
        connection.close()


def check_environment(model_path, depth_path, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('environment check requires CUDA_VISIBLE_DEVICES empty')
    context = multiprocessing.get_context('spawn')
    reader, writer = context.Pipe(duplex=False)
    child = context.Process(target=environment_worker, args=(writer, model_path, depth_path))
    child.start()
    writer.close()
    try:
        if not reader.poll(180):
            raise TimeoutError('DualVLN CPU spawn environment check timed out')
        report = reader.recv()
        child.join(timeout=10)
        report.update(child_exit_code=child.exitcode, checked_at=time.time())
        atomic_json(Path(output), report)
        if not report['ok'] or child.exitcode != 0:
            raise RuntimeError('DualVLN environment check failed: ' + str(output))
        print(json.dumps(dict(event='dualvln_environment_ready', **report)), flush=True)
        return report
    finally:
        reader.close()
        if child.is_alive():
            child.terminate()
            child.join(timeout=10)


def checkpoint_metadata(path, brain):
    path, brain = Path(path), Path(brain)
    proof = json.loads((path / 'download_verification.json').read_text())
    exported = json.loads((brain / 'brain_export.json').read_text())
    if (not proof.get('ok') or proof.get('model') != 'InternRobotics/InternVLA-N1-DualVLN'
            or not exported.get('ok') or exported.get('schema') != 'dualvln_unchanged_brain_export_v1'
            or Path(exported['source']).resolve() != path.resolve()
            or exported['source_proof_sha256'] != sha256_file(path / 'download_verification.json')):
        raise ValueError('official DualVLN requires its own verified unchanged brain export')
    return dict(policy='dualvln', checkpoint_dir=str(path), qwen_dir=str(brain),
        official_model=proof['model'], revision=proof['revision'],
        model_proof_sha256=sha256_file(path / 'download_verification.json'),
        brain_export_sha256=sha256_file(brain / 'brain_export.json'))


def contract(plan, stage):
    ckpt = next(c for c in plan['config']['checkpoints'] if c['name'] == stage['checkpoint'])
    return dict(schema='dualvln_main_eval_v1', plan_sha256=plan['plan_sha256'],
        benchmark=stage['benchmark'], checkpoint=ckpt['identity'], seed=plan['config']['seed'],
        camera=ckpt['observation_profiles'][stage['benchmark']],
        slow_call_budget=plan['config']['max_tool_calls'][stage['benchmark']],
        count_at='model.generate', s1_view='down', s1_first_empty_recovery='left',
        s2_max_new_tokens=128, repeats=1, scoring_mode='merged_batch',
        navmesh_root=plan['config'].get('navmesh_root'))


def native_down_rgb(case, info, render_url):
    """Render a current S1 camera; simulator pose never enters the policy."""
    geometry = info['oracle_habitat_geometry']
    sensors = copy.deepcopy(geometry['sensors'])
    for sensor in sensors:
        sensor['orientation'][0] = -math.pi / 6
    scene = case['scene'].removeprefix('rxr-')
    request_id = uuid.uuid4().hex
    payload = dict(request_id=request_id, scene=f'{scene}/{scene}.glb',
        position=geometry['position'], rotation=geometry['rotation_xyzw'], sensors=sensors)
    request = Request(render_url.rstrip('/') + '/render', data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    with build_opener(ProxyHandler({})).open(request, timeout=600) as response:
        body = response.read(32 * 1024 * 1024)
    with np.load(io.BytesIO(body), allow_pickle=False) as arrays:
        if str(arrays['request_id']) != request_id:
            raise ValueError('S1 camera rendering response identity mismatch')
        # Habitat COLOR observations may include alpha; the official policy
        # and its ResNet normalization consume three RGB channels.
        return np.asarray(Image.fromarray(arrays['rgb']).convert('RGB')).copy()


def rollout(plan, stage, case, client, policy, render_url, dataset=None):
    from navigation_baseline_policies import SlowBudgetExhausted
    from navigation_baselines import execute_native, look_down_rgb, validate_reset as sage_reset
    from v6_vlnce_eval import validate_reset as native_reset, score_native
    from vln_gym.observation_profile import assert_profile
    ckpt = next(c for c in plan['config']['checkpoints'] if c['name'] == stage['checkpoint'])
    sage = stage['benchmark'] == 'sage-3d'
    row = {**case, 'evaluation_valid': False, 'success': False, 'agent_stopped': False,
           'native_actions': 0, 'turns': [], 'repeat_index': 1, 'repeats_per_case': 1}
    seed = int(case.get('seed', plan['config']['seed']))
    obs, info = client.reset(scene=case['scene'], episode_id=case['episode_id'],
        instruction=case['instruction'], seed=seed)
    assert_profile(info.get('observation_profile') or {}, ckpt['observation_profiles'][stage['benchmark']])
    if sage:
        task = next(t for t in dataset.tasks(case['scene']) if str(t.task_id) == str(case['episode_id']))
        row['initial_rgb_sha256'] = sage_reset(case, task, obs, info)
    else:
        native_reset(case, obs, info)
        if not math.isfinite(float(info.get('distance_to_goal', float('nan')))):
            raise ValueError('nonfinite native distance at reset, before model actions')
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy.reset()
    positions, looking_down, done = [], False, False
    while not done:
        down = None

        def current_down():
            nonlocal down
            if down is None:
                down = (look_down_rgb(case['scene'], info, obs['rgb'], render_url, dataset) if sage
                        else native_down_rgb(case, info, render_url))
            return down

        try:
            prediction = policy.predict(current_down() if looking_down else obs['rgb'],
                case['instruction'], look_down=looking_down, fast_rgb_provider=current_down)
        except SlowBudgetExhausted:
            row['why'] = 'slow_call_budget_exhausted'
            break
        looking_down = False
        turn = dict(index=len(row['turns']), slow_calls=policy.slow_calls, **prediction, executions=[])
        row['turns'].append(turn)
        for index, action in enumerate(prediction['actions']):
            if action == 5:
                looking_down = True
                turn['executions'].append({'action': 5, 'camera_pitch_deg': -30})
                break
            # Habitat's task uses exact 25cm/15deg discrete actions. SAGE's
            # tool adapter is the already-tested equivalent for its simulator.
            obs, _, terminated, truncated, info = (execute_native(client, action) if sage
                                                  else client.step_discrete(action))
            row['native_actions'] += 1
            row['agent_stopped'] = action == 0
            turn['executions'].append({'action': action, 'info': info})
            done = bool(terminated or truncated or row['agent_stopped'])
            if done:
                row['why'] = 'agent_stop' if row['agent_stopped'] else 'environment_terminated'
                break
            if index < len(prediction['actions']) - 1:
                policy.observe(obs['rgb'])
        if sage:
            from sage import benchmark_protocol
            detail = info['official_object_success_detail']
            positions.append(detail['official_position'])
            if benchmark_protocol.auto_stop(info):
                row['why'], done = 'official_goal_radius', True
    row.update(evaluation_valid=True, slow_calls=policy.slow_calls,
               calls_used=policy.slow_calls, final_info=info)
    if sage:
        from sage import benchmark_protocol
        detail = info['official_object_success_detail']
        length, optimum = benchmark_protocol.trajectory_metrics(case['episode_id'], positions)
        row.update(success=bool(detail['success']), official_object_success_detail=detail,
            decision_positions=positions, path_length_m=length, shortest_path_m=optimum,
            spl=float(detail['success']) * (optimum / max(optimum, length) if optimum else 1.),
            distance_to_goal=info.get('distance_to_goal'), steps_used=info.get('steps_used'))
    else:
        if not row['agent_stopped']:
            _, _, _, _, info = client.step_tool('stop')
        row = score_native(row, info, scoring_mode='merged_batch')
    return row


def worker(plan, stage, cases, output, gym_url, gpu, index, scheduler, render_url):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    sys.path[:0] = [str(REPO / 'vln_demo/src'), str(SWEEP),
                   str(REPO / 'third_party/scaling_on_scales')]
    from navigation_baseline_policies import DualVLNPolicy
    from verl_access.bench_client import BenchClient
    ckpt = next(c for c in plan['config']['checkpoints'] if c['name'] == stage['checkpoint'])
    policy = DualVLNPolicy(ckpt['path'], ckpt['depth_path'], fast_view='down',
        first_empty_recovery='left', slow_call_budget=plan['config']['max_tool_calls'][stage['benchmark']])
    client = BenchClient(gym_url, timeout=600)
    dataset = None
    if stage['benchmark'] == 'sage-3d':
        from sage.datasets import SageSet
        dataset = SageSet('official_sage_bench_v1')
    lookup, contract_hash = {c['logical_id']: c for c in cases}, digest(contract(plan, stage))
    count = dict(executed=0, valid=0, success=0, infrastructure_failed=0)
    out = Path(output)
    started = time.monotonic()
    from v6_vlnce_eval import evaluate_attempts
    evaluator = SimpleNamespace(rollout=lambda case, seed: rollout(
        plan, stage, case, client, policy, render_url, dataset))

    def progress():
        atomic_json(out / 'workers' / f'worker_{index:02d}' / 'progress.json',
            dict(count, worker_index=index, gpu=gpu, elapsed_s=time.monotonic() - started))

    try:
        progress()
        while (case := scheduler.acquire()) is not None:
            try:
                if lookup.get(case['logical_id']) != case:
                    raise ValueError('scheduler returned a changed case')
                row = evaluate_attempts(evaluator, case, contract(plan, stage), out,
                    worker_index=index, actor_index=index)
                valid = bool(row.get('evaluation_valid') and not row.get('infra_error'))
                count['executed'] += 1
                count['valid'] += int(valid)
                count['success'] += int(valid and row['success'])
                count['infrastructure_failed'] += int(not valid)
                progress()
                print(json.dumps(dict(event='dualvln_case', logical_id=case['logical_id'],
                    worker_index=index, valid=valid, success=row['success'],
                    slow_calls=row.get('slow_calls'), attempts=row['attempt_number'])), flush=True)
            finally:
                scheduler.release(case)
    finally:
        client.close()


def aggregate_rows(plan, stage, rows, cases, evaluation_contract=None):
    contract_hash = digest(evaluation_contract or contract(plan, stage))
    lookup = {c['logical_id']: c for c in cases}
    if len(lookup) != len(cases) or len({r['logical_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate DualVLN case identity')
    for row in rows:
        case = lookup.get(row['logical_id'])
        if (case is None or row.get('case_sha256') != digest(case)
                or row.get('contract_sha256') != contract_hash
                or row.get('repeat_index') != 1 or any(row.get(k) != v for k, v in case.items())):
            raise ValueError('changed DualVLN case or evaluation contract')
    if stage['benchmark'] != 'sage-3d':
        from v6_vlnce_eval import summarize
        return summarize(cases, {r['logical_id']: r for r in rows}, contract_hash)
    valid = [r for r in rows if r.get('evaluation_valid') and not r.get('infra_error')]
    expected = {c['logical_id'] for c in cases}
    found = {r['logical_id'] for r in rows}
    def rate(key):
        hits = sum(bool(r.get(key)) for r in valid)
        return dict(value=hits / len(cases) if cases else None, numerator=hits, denominator=len(cases))
    metrics = dict(success_rate=rate('success'), spl=sum(float(r.get('spl', 0)) for r in valid) / len(cases) if cases else None,
        agent_stop_rate=rate('agent_stopped'), mean_tool_calls=sum(r['slow_calls'] for r in valid) / len(valid) if valid else None)
    return dict(evaluation_valid=found == expected and len(rows) == len(cases) == len(valid),
        completed_runs=len(rows), valid_runs=len(valid), expected_runs=len(cases),
        overall_micro=metrics, invalid_logical_ids=[r['logical_id'] for r in rows if r not in valid])


def run_pool(plan, stage, cases, output, gym_urls, render_url):
    from v6_vlnce_eval import PoolManager, pool_progress, read_result, result_path, resumable
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'contract.json', contract(plan, stage))
    count = min(len(cases), len(gym_urls), 15 if stage['benchmark'] == 'sage-3d' else 16)
    layouts = [dict(worker_index=i, gpu=str(i)) for i in range(count)]
    pending = [c for c in cases if not resumable(read_result(result_path(output, c)),
                                               c, digest(contract(plan, stage)))]
    with (output / '.pool.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if pending:
            if Path(sys.prefix) != DUAL_PYTHON.parent.parent:
                raise ValueError('launch the DualVLN collector with dualvln_eval; spawn inherits parent sys.path')
            multiprocessing.set_executable(str(DUAL_PYTHON))
            context = multiprocessing.get_context('spawn')
            manager = PoolManager(address=('127.0.0.1', 0), ctx=context)
            manager.start()
            scheduler = manager.SceneCaseScheduler(pending, replicas=1,
                max_scenes=count, max_cases=count)
            processes, started = [], time.monotonic()
            try:
                for layout in layouts:
                    i = layout['worker_index']
                    process = context.Process(target=worker, args=(plan, stage, cases, str(output),
                        gym_urls[i], layout['gpu'], i, scheduler, render_url), name=f'dualvln-{i}')
                    process.start()
                    processes.append(process)
                while True:
                    failed = [p for p in processes if p.exitcode not in (None, 0)]
                    if failed:
                        raise RuntimeError('DualVLN workers failed: ' + ', '.join(
                            f'{p.name}={p.exitcode}' for p in failed))
                    active = any(p.is_alive() for p in processes)
                    pool_progress(output, scheduler, layouts, len(cases), len(cases)-len(pending),
                        started, 'running' if active else 'finished')
                    if not active:
                        break
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline and any(p.is_alive() for p in processes):
                        if any(p.exitcode not in (None, 0) for p in processes):
                            break
                        time.sleep(1)
            finally:
                scheduler.cancel()
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                for process in processes:
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5)
                manager.shutdown()
        rows = [read_result(result_path(output, c)) for c in cases]
        result = aggregate_rows(plan, stage, [r for r in rows if r], cases)
        atomic_json(output / 'summary.json', result)
        return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CPU-only official DualVLN spawn/import acceptance')
    parser.add_argument('--check-environment', action='store_true', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--depth-path', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    check_environment(args.model_path, args.depth_path, args.output)
