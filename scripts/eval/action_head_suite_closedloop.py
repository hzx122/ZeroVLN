"""Small suite adapter around the existing SAGE TracedEvaluator/parallel loop."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from action_head_benchmark_suite import REPO, SWEEP, atomic_json, digest


def run_sage(plan, stage, rank, manifest_path, output, gym_urls, brain_urls, query_cache):
    sys.path[:0] = [str(SWEEP), str(REPO / 'verl'), str(REPO / 'vln_demo/src')]
    import closedloop_eval as CE
    from vln_gym.observation_profile import requested_profile, assert_profile
    ckpt = next(x for x in plan['config']['checkpoints'] if x['name'] == stage['checkpoint'])
    config = json.loads((Path(ckpt['identity']['action_head_dir']) / 'fast_slow_action_head_config.json').read_text())
    CE.ACTIVE_QUERY_COUNT = int(config['query_num_queries'])
    CE.E.chat_query = CE.training_matched_chat_query
    original_type = CE.TracedEvaluator

    class SuiteEvaluator(original_type):
        def __init__(self, args):
            super().__init__(args)
            original_reset = self.runtime.reset_environment

            def reset(**kwargs):
                obs, info = original_reset(**kwargs)
                assert_profile(info.get('observation_profile') or {}, requested_profile())
                scene = info.get('actual_scene', info.get('scene'))
                episode = info.get('actual_episode_id', info.get('episode_id'))
                if scene != self.current_case['scene'] or str(episode) != str(self.current_case['episode_id']):
                    raise ValueError('SAGE reset identity differs from frozen suite case')
                return obs, info

            self.runtime.reset_environment = reset

        def rollout(self, case, seed):
            self.current_case = case
            try:
                row = super().rollout(case, seed)
                # Bind the suite identity before the collector's FIRST atomic
                # publication; a later batch-level loop can be interrupted.
                row.update(suite_plan_sha256=plan['plan_sha256'], case_sha256=digest(case),
                           observation_profile=requested_profile(), suite_rank=rank)
                return row
            finally:
                self.runtime.tool_executor.end_episode()

    CE.TracedEvaluator = SuiteEvaluator
    manifest = json.loads(Path(manifest_path).read_text())
    cases = manifest['cases']
    namespace = SimpleNamespace(checkpoint=ckpt['identity']['action_head_dir'],
        backbone=ckpt['backbone'], query_cache=query_cache, bench_url=gym_urls[0],
        vllm_url=brain_urls[0], vllm_urls=brain_urls, timeout=600, policy_seed=20260827,
        mode='instruct', max_turns=0, max_tool_calls=plan['config']['max_tool_calls']['sage-3d'],
        one_tool_call_per_turn=True, keep_last_images=8,
        action_head_as_tool_version=os.environ.get('RULE_TEACHER_CONTEXT_VERSION') or 'v6')
    args = SimpleNamespace(bench_urls=','.join(gym_urls),
                           max_batch_size=int(os.environ.get('SUITE_FM_MAX_BATCH_SIZE', '8')),
                           batch_wait_ms=int(os.environ.get('SUITE_FM_BATCH_WAIT_MS', '12')),
                           repeats=plan['config'].get('evaluation_repeats',1) if plan['config'].get('pooled_attempts') else 1,
                           retries=2)
    CE.parallel_rollouts(args, namespace, cases, Path(output), ckpt['identity'])
    for case in cases:
        for repeat in range(1,args.repeats+1):
            path = CE.result_path(Path(output), case, repeat)
            row = json.loads(path.read_text())
            row.update(suite_plan_sha256=plan['plan_sha256'], case_sha256=digest(case),
                       observation_profile=requested_profile(), suite_rank=rank)
            atomic_json(path, row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--stage', type=int, required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gym-urls', required=True)
    parser.add_argument('--brain-urls', required=True)
    parser.add_argument('--query-cache', required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    run_sage(plan, plan['stages'][args.stage], args.rank, args.manifest, args.output,
             args.gym_urls.split(','), args.brain_urls.split(','), args.query_cache)


if __name__ == '__main__':
    main()
