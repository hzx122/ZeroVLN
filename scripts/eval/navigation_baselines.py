#!/usr/bin/env python3
"""Evaluate official navigation checkpoints on the existing SAGE test manifest."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time
import traceback

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
DATA = Path('/data/zerovln')
sys.path[:0] = [str(REPO / 'vln_demo/src'),
               str(REPO / 'verl/scripts/eval/qwen36_4query_ckpt_sweep')]
from common import atomic_json, load_subset, safe, sha256_file
from verl_access.bench_client import BenchClient
from navigation_baseline_policies import NaVILAPolicy, DualVLNPolicy


def execute_native(client, action):
    # Bench's default discrete turn is 30deg; upstream checkpoints use 15deg.
    if action == 0:
        return client.step_tool('stop')
    if action == 1:
        return client.step_tool('move_forward', {'distance': .25})
    if action in (2, 3):
        return client.step_tool('turn', {'direction': 'left' if action == 2 else 'right', 'degrees': 15})
    raise ValueError(f'unsupported native action: {action}')


def validate_reset(case, task, obs, info):
    """Check backend identity and physical start, not the echoed request IDs."""
    for key, expected in (('actual_scene', case['scene']),
                          ('actual_episode_id', case['episode_id'])):
        if info.get(key) != expected:
            raise ValueError(f'reset {key}: expected {expected!r}, got {info.get(key)!r}')
    xy = np.asarray(info['world_xy'], dtype=float)
    expected_xy = np.array([task.start['x'], task.start['y']])
    yaw_error = (math.radians(info['world_yaw_deg']) - task.start['yaw'] + math.pi) % (2*math.pi) - math.pi
    if np.linalg.norm(xy - expected_xy) > .002 or abs(yaw_error) > math.radians(.11):
        raise ValueError(f'reset start pose mismatch: {xy.tolist()}, yaw_error={yaw_error}')
    rgb = obs['rgb']
    if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
        raise ValueError(f'invalid reset RGB: {rgb.shape}, {rgb.dtype}')
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def look_down_rgb(scene, info, rgb, render_url, dataset):
    # Pose is used only by the simulator renderer, never passed to a policy.
    from sage.gs_client import XYZEulerClient
    from sage.occmap import OccMap
    from rs3d.gs_client import CameraSpec
    xy = info['world_xy']
    occ = OccMap(dataset.scene_dir(scene), collision_mode='usd_rooms', furniture_mode='none')
    profile = info.get('observation_profile') or dict(hfov_deg=115., height_m=1.2)
    client = XYZEulerClient(render_url, camera=CameraSpec(rgb.shape[1], rgb.shape[0], profile['hfov_deg']))
    frame, _ = client.render_one(scene, [*xy, occ.ground_z(*xy) + profile['height_m']],
                                math.radians(info['world_yaw_deg']), pitch=-math.pi/6)
    return np.asarray(frame)


def predict_step(policy, scene, info, rgb, instruction, looking_down, render_url, dataset):
    """Keep the horizontal history separate from the fast camera at this pose."""
    down = None

    def current_down():
        nonlocal down
        if down is None:
            down = look_down_rgb(scene, info, rgb, render_url, dataset)
        return down

    slow_rgb = current_down() if looking_down else rgb
    kwargs = {}
    if isinstance(policy, DualVLNPolicy) and policy.fast_view != 'legacy':
        kwargs['fast_rgb_provider'] = (current_down if policy.fast_view == 'down' else lambda: rgb)
    return policy.predict(slow_rgb, instruction, look_down=looking_down, **kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy', choices=('navila', 'dualvln'), required=True)
    p.add_argument('--model-path', type=Path, required=True)
    p.add_argument('--depth-path', type=Path, default=DATA / 'vln_verl/models/Depth-Anything-V2-Metric-Hypersim-Small')
    p.add_argument('--manifest', type=Path, default=DATA / 'SAGE-3D/official_gvln_action_head_p0_v3test100_20260908/closedloop_subset_all100.json')
    p.add_argument('--sage-split', default='official_gvln_action_head_p0_v3test100')
    p.add_argument('--map-root', type=Path, default=None)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gym-url', default='http://127.0.0.1:8968')
    p.add_argument('--render-url', default='http://127.0.0.1:7068')
    p.add_argument('--max-decisions', type=int, default=50)
    p.add_argument('--max-actions', type=int, default=500)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--seed', type=int, default=20260919)
    p.add_argument('--dualvln-fast-view', choices=('legacy', 'down', 'horizontal'), default='legacy')
    p.add_argument('--dualvln-first-empty-recovery', choices=('refresh', 'left'), default='refresh')
    args = p.parse_args()
    if not args.output.resolve().is_relative_to(DATA / 'vln_verl/artifacts'):
        p.error('output must be on the authorized data disk')
    if min(args.max_decisions, args.repeats) < 1 or min(args.max_actions, args.limit, args.offset) < 0:
        p.error('invalid evaluation budget')
    manifest, all_cases = load_subset(args.manifest)
    cases = all_cases[args.offset:][:args.limit or None]
    if not cases:
        p.error('empty case selection')
    proof = args.model_path / 'download_verification.json'
    if not proof.exists() and args.policy == 'dualvln':
        proof = args.model_path / 'training_complete.json'
    verification = json.loads(proof.read_text())
    if not verification.get('ok'):
        raise RuntimeError('checkpoint download is not verified')
    if verification.get('smoke_only') or verification.get('random_initialization_lineage'):
        raise RuntimeError('pipeline smoke checkpoints are not navigation baselines')
    import rescore_v43_with_official_semantic_maps as scoring
    scoring.obs.reverse_position_mapping = scoring.cached_reverse
    if args.map_root is not None:
        scoring.MAP_ROOT = args.map_root
    from sage import benchmark_protocol as benchmark
    if benchmark.enabled() and (args.max_decisions != 200 or args.max_actions != 0):
        raise ValueError("official protocol requires 200 decisions and no native action cap")
    from sage.datasets import SageSet
    dataset = SageSet(args.sage_split)
    dataset_manifest = Path(dataset.vln_dir).parent / 'manifest.json'
    if not dataset_manifest.exists():
        dataset_manifest = Path(dataset.vln_dir) / 'manifest.json'
    if not dataset_manifest.exists():
        raise RuntimeError(f'missing dataset manifest for split {args.sage_split}')
    evaluators = {scene: scoring.obs.ObjectBasedSuccessEvaluator(str(scoring.scene_map(scene)), verbose=False)
                  for scene in {case['scene'] for case in cases}}
    tasks = {scene: {task.task_id: task for task in dataset.tasks(scene)} for scene in evaluators}
    identity = {
        'schema': 'official-navigation-sage-v2-verified-reset', 'policy': args.policy,
        'model': verification.get('model', str(args.model_path)),
        'revision': verification.get('revision', verification.get('identity_sha256')),
        'model_proof_sha256': sha256_file(proof),
        'manifest_sha256': sha256_file(args.manifest),
        'sage_split': args.sage_split, 'dataset_manifest_sha256': sha256_file(dataset_manifest),
        'policy_code_sha256': sha256_file(Path(__file__).with_name('navigation_baseline_policies.py')),
        'evaluator_code_sha256': sha256_file(__file__),
        'semantic_maps': {scene: sha256_file(scoring.scene_map(scene)) for scene in evaluators},
        'case_ids': [case['episode_id'] for case in cases],
        'max_decisions': args.max_decisions, 'max_native_actions': args.max_actions,
        'repeats': args.repeats, 'seed': args.seed,
        'success': 'exact official semantic map target ID; bbox expansion 1m; terminal arrival',
        'policy_observation': 'RGB history and instruction only',
        'native_action_m': .25, 'native_turn_deg': 15,
        'physical_contract': {'collision': 'usd_rooms', 'furniture': 'none',
                              'robot_radius': .07, 'camera_height': 1.2, 'hfov': 115},
        'dualvln_adapter': 'official S2 and S1, partial_async budget8, native prefix4; serialized inference',
    }
    if benchmark.enabled():
        identity["benchmark_protocol"] = benchmark.identity()
        identity["success"] = benchmark.bundle()[1]["success"]
    if args.policy == 'dualvln':
        identity['dualvln_input_contract'] = dict(
            version=2, fast_view=args.dualvln_fast_view,
            first_empty_recovery=args.dualvln_first_empty_recovery,
            slow_history_view='horizontal', fast_anchor='fixed until next S2 pixel goal')
    args.output.mkdir(parents=True, exist_ok=True)
    identity_path = args.output / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError('refusing resume with a different evaluation identity')
    atomic_json(identity_path, identity)
    client = BenchClient(args.gym_url, timeout=300)
    client.health()
    policy = (NaVILAPolicy if args.policy == 'navila' else DualVLNPolicy)(
        model_path=str(args.model_path), depth_path=str(args.depth_path),
        fast_view=args.dualvln_fast_view, first_empty_recovery=args.dualvln_first_empty_recovery)
    import torch
    rows = []
    initial_frames = {}
    for case in cases:
        for repeat in range(args.repeats):
            path = args.output / 'rollouts' / f"{safe(case['episode_id'])}_r{repeat}.json"
            if path.exists():
                rows.append(json.loads(path.read_text()))
                continue
            seed = args.seed + repeat + int.from_bytes(hashlib.sha256(case['episode_id'].encode()).digest()[:3], 'big')
            torch.manual_seed(seed)
            np.random.seed(seed)
            row = {**case, 'repeat_index': repeat, 'seed': seed, 'evaluation_valid': False,
                   'success': False, 'stopped': False, 'native_actions': 0, 'turns': []}
            started = time.monotonic()
            try:
                obs, info = client.reset(scene=case['scene'], episode_id=case['episode_id'],
                                         instruction=case['instruction'], seed=seed)
                task = tasks[case['scene']][case['episode_id']]
                frame_hash = validate_reset(case, task, obs, info)
                previous_scene = initial_frames.setdefault(frame_hash, case['scene'])
                if previous_scene != case['scene']:
                    raise ValueError('identical reset RGB across different scenes')
                row['reset_info'] = info
                row['initial_rgb_sha256'] = frame_hash
                policy.reset()
                frame_dir = args.output / 'frames' / f"{safe(case['episode_id'])}_r{repeat}"
                frame_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(obs['rgb']).save(frame_dir / 'initial.png')
                decision_positions = []
                done, looking_down = False, False
                for decision in range(args.max_decisions):
                    prediction = predict_step(policy, case['scene'], info, obs['rgb'],
                                              case['instruction'], looking_down, args.render_url, dataset)
                    looking_down = False
                    turn = {'index': decision, **prediction, 'executions': []}
                    row['turns'].append(turn)
                    for index, action in enumerate(prediction['actions']):
                        if args.max_actions and row['native_actions'] >= args.max_actions:
                            done = True
                            break
                        row['native_actions'] += 1
                        if action == 5:
                            looking_down = True
                            turn['executions'].append({'action': 5, 'camera_pitch_deg': -30})
                            break
                        obs, reward, terminated, truncated, info = execute_native(client, action)
                        turn['executions'].append({'action': action, 'info': info})
                        row['stopped'] = action == 0
                        done = bool(terminated or truncated or row['stopped'])
                        if done:
                            break
                        if index < len(prediction['actions']) - 1:
                            policy.observe(obs['rgb'])
                    if benchmark.enabled():
                        detail = info.get("official_object_success_detail") or {}
                        decision_positions.append(detail["official_position"])
                        if benchmark.auto_stop(info):
                            row["official_auto_stop"] = True
                            done = True
                    if done:
                        break
                task = tasks[case['scene']][case['episode_id']]
                xy = info['world_xy']
                row.update(evaluation_valid=True, final_info=info,
                           target_object_id=task.goal_label,
                           success=scoring.score(evaluators[case['scene']], xy, task.goal_label))
                if benchmark.enabled():
                    detail = info['official_object_success_detail']
                    benchmark.auto_stop(info)  # validates protocol even for explicit STOP
                    row['success'] = bool(detail['success'])
                    row['strict_target_success'] = bool(detail['strict_target_success'])
                    row['official_object_success_detail'] = detail
                    row['decision_positions'] = decision_positions
                    length, optimum = benchmark.trajectory_metrics(case['episode_id'], decision_positions)
                    row['path_length_m'], row['shortest_path_m'] = length, optimum
                    row['spl'] = float(row['success']) * (optimum / max(optimum, length) if optimum else 1.0)
                    row['why'] = ('agent_stop' if row['stopped'] else 'official_goal_radius'
                                  if row.get('official_auto_stop') else 'decision_budget_exhausted')
                row['success_with_stop'] = row['success'] and row['stopped']
                Image.fromarray(obs['rgb']).save(frame_dir / 'final.png')
            except Exception:
                row['error'] = traceback.format_exc()
                print(row['error'], flush=True)
                if 'reset_info' not in row:
                    atomic_json(path, row)
                    raise RuntimeError('environment reset audit failed; aborting evaluation')
            row['wall_seconds'] = time.monotonic() - started
            atomic_json(path, row)
            rows.append(row)
            valid = [r for r in rows if r['evaluation_valid']]
            summary = {'policy': args.policy, 'selected_cases': len(cases),
                       'test_manifest_cases': len(all_cases), 'expected_rollouts': len(cases)*args.repeats,
                       'completed_rollouts': len(rows), 'valid_rollouts': len(valid),
                       'invalid_rollouts': len(rows)-len(valid), 'successes': sum(r['success'] for r in valid),
                       'sr_all_completed': sum(r['success'] for r in valid)/len(rows),
                       'full_test100': len(cases) == len(all_cases),
                       'complete': len(rows) == len(cases)*args.repeats}
            atomic_json(args.output / 'summary.json', summary)
            print(json.dumps(summary), flush=True)
    client.close()
    if any(not row['evaluation_valid'] for row in rows):
        raise SystemExit('evaluation contains invalid rollouts; inspect errors before reporting SR')


if __name__ == '__main__':
    main()
