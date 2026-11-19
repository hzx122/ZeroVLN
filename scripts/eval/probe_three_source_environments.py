"""Exercise real native CPU frames and cross-dataset RL environment resets."""
import argparse
import json
import math
from pathlib import Path
import time
from urllib.request import urlopen

from action_head.render_client import RemoteRenderClient
from action_head_rl.rule_credit import _state, score_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gym-url', required=True)
    parser.add_argument('--renderer-url', required=True)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-sage', action='store_true')
    args = parser.parse_args()
    allowed = Path('/data/zerovln')
    if not args.output.resolve().is_relative_to(allowed):
        parser.error('probe outputs must be on the authorized data disk')
    args.output.mkdir(parents=True, exist_ok=False)
    chosen = {}
    no_hit_case = None
    for line in args.pool.read_text().splitlines():
        row = json.loads(line)
        chosen.setdefault(row['backend'], row)
        if (row['backend'],row['scene'],str(row['episode_id'])) == ('rxr','b8cTxDM8gDG','26597'):
            no_hit_case = row
    client = RemoteRenderClient(args.gym_url, timeout=600)
    order = ['r2r', 'rxr', 'r2r']
    if args.include_sage:
        order = ['sage', *order, 'sage']
    selected = [chosen[backend] for backend in order]
    # This admitted first-batch case looks into unmeshed sky/sea. Exercise it
    # through the actual HTTP reset/turn/prefix path on every training node.
    if no_hit_case is not None:
        selected.insert(-1, no_hit_case)
    order = [row['backend'] for row in selected]
    reports = []
    for index, row in enumerate(selected):
        backend = row['backend']
        started = time.monotonic()
        obs, initial = client.reset(scene=row['scene'], episode_id=row['episode_id'],
            instruction=row['instruction'], backend=backend, seed=20261010 + index)
        reset_seconds = time.monotonic() - started
        assert obs['instruction'] == row['instruction']
        assert tuple(obs['rgb'].shape) == (480, 640, 3)
        assert _state(initial['rule_credit_state'])
        # Sanitized policy observations cannot carry evaluator-only evidence.
        assert not any(key.startswith('oracle') or key == 'rule_credit_state' for key in obs)
        started = time.monotonic()
        _, turn_result = client.execute_turn('left', 30.)
        turned = client.last_info
        turn_seconds = time.monotonic() - started
        difference = (turned['world_yaw_deg'] - initial['world_yaw_deg'] + 180.) % 360. - 180.
        assert abs(difference - 30.) < 2., difference
        started = time.monotonic()
        _, _, _, _, moved = client.execute_prefix([[i * .008, 0., 0.] for i in range(1, 33)])
        prefix_seconds = time.monotonic() - started
        assert _state(moved['rule_credit_state'])
        assert all(math.isfinite(float(v)) for v in moved['world_xy'])
        final = client._post('/step', {'action': 0})
        raw_report = dict(backend=backend, scene=row['scene'], episode_id=row['episode_id'],
            reset_seconds=reset_seconds, turn_seconds=turn_seconds, prefix_seconds=prefix_seconds,
            initial_info=initial, turn_info=turned, move_info=moved, terminal_info=final['info'])
        (args.output / f'{index:02d}_{backend}.json').write_text(json.dumps(raw_report, indent=2) + '\n')
        assert final['terminated'] is True
        assert _state(final['info']['rule_credit_state'])
        terminal = final['info']
        option = 'native-credit-probe-call'
        rollout_result = dict(episode_id=row['episode_id'], scene=row['scene'],
            evaluation_valid=True, infra_error='', success=bool(terminal.get('official_object_success', terminal['success'])),
            success_contract=(terminal.get('official_object_success_detail') or {}).get('schema', terminal.get('success_contract')),
            why='agent_stop', calls_used=3, initial_info=initial,
            turns=[dict(turn=1, calls=[dict(name='turn', ok=True, result=turn_result,
                execution_state_trace=dict(before_world_pose=dict(world_xy=initial['world_xy']),
                                           after_world_pose=dict(world_xy=turned['world_xy'])))],
                rule_credit=dict(before=initial['rule_credit_state'], after=turned['rule_credit_state'])),
                dict(turn=2, calls=[dict(name='run_action_head', ok=True,
                    result=dict(option_id=option, chunks=1))],
                    rule_credit=dict(before=turned['rule_credit_state'], after=moved['rule_credit_state'])),
                dict(turn=3, calls=[dict(name='stop', ok=True, score_info=terminal)],
                    rule_credit=dict(before=moved['rule_credit_state'], after=terminal['rule_credit_state']))])
        reward = score_episode(dict(result=rollout_result,
            chunks=[dict(option_id=option, chunk_index=0, gym_feedback=moved)]),
            step=1, version='v3_first_arrival')
        raw_report['full_call_credit_score'] = reward
        (args.output / f'{index:02d}_{backend}.json').write_text(json.dumps(raw_report, indent=2) + '\n')
        if backend != 'sage':
            for info in (initial, turned, moved):
                audit = info['sensor_audit']
                assert audit['depth_finite_fraction'] == 1.
                assert audit['depth_normalized'] is False
                assert audit['depth_dtype'] == 'float32'
                assert 0. <= audit['depth_min_m'] <= audit['depth_max_m'] <= 10.
                state = info['rule_credit_state']
                assert state['schema'] == 'mp3d-rule-credit-state-v1'
                assert state['information_available'] is True
                if state['information_depth']['positive_pixels'] == 0:
                    assert state['visible_entities'] == []
                assert info['success_contract'] == 'habitat_vlnce_native_stop_3m_v1'
        reports.append(raw_report)
        print(f'probe {index} {backend} passed reset={reset_seconds:.2f}s '
              f'turn={turn_seconds:.2f}s prefix={prefix_seconds:.2f}s', flush=True)
    with urlopen(args.renderer_url + '/health', timeout=5) as response:
        renderer = json.load(response)
    assert renderer['renderer'] == 'habitat-mesh-software-egl'
    (args.output / 'summary.json').write_text(json.dumps(dict(passed=True,
        order=order, cases=len(reports), cpu_renderer=renderer,
        full_rl_verified=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
