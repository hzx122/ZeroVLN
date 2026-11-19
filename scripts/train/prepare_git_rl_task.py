"""Prepare configuration for local Git training; never copy source or submit."""
import argparse
import json
from pathlib import Path

import align_three_source_observations as io
from rl_task_config import ARMS, OVERLAY, remap, rollout_gyms_per_node, native_scene_cache_settings, apply_navigation_pool


def configure(plan, *, algorithm=None, nodes=None, batch_size=None,
              opt_steps=None, checkpoint_interval=None, architecture=None, max_tool_calls=None,
              resource_mode=None, rollout_actors_per_node=None, vllm_tp_size=None,
              sage_cpu_render_parallelism=None, cpu_group_preparation_tasks_per_node=None,
              native_render_shared_cache=None, native_render_cache_max_gib=None,
              native_render_scene_load_reserve_gib=None, native_episode_cache=None):
    """Scale the existing per-node layout without inheriting stale v4 evidence."""
    config = plan['config']
    # A reference that is already colocated keeps its validated fast-policy
    # placement (e.g. colocated_cuda); only an explicit conversion resets it.
    inherited_fast_policy = (config.get('fast_policy_device')
        if resource_mode is None and plan.get('resource_mode') else None)
    resource_mode = resource_mode or plan.get('resource_mode')
    if native_episode_cache is not None:
        plan['native_episode_cache'] = native_episode_cache
    if type(plan.get('native_episode_cache', False)) is not bool:
        raise ValueError('native episode cache must be an explicit boolean')
    for name, value in (('native_render_shared_cache', native_render_shared_cache),
                        ('native_render_cache_max_gib', native_render_cache_max_gib),
                        ('native_render_scene_load_reserve_gib', native_render_scene_load_reserve_gib)):
        if value is not None:
            plan[name] = value
    native_scene_cache_settings(plan, resource_mode)
    if cpu_group_preparation_tasks_per_node is not None:
        plan['cpu_group_preparation_tasks_per_node'] = cpu_group_preparation_tasks_per_node
    preparation_tasks = plan.get('cpu_group_preparation_tasks_per_node', 1)
    if (type(preparation_tasks) is not int or preparation_tasks not in (1, 2)
            or (preparation_tasks != 1 and (resource_mode != 'cpu_colocated'
                or not config.get('cpu_group_preparation')
                or not config.get('overlap_cpu_group_preparation')))):
        raise ValueError('two CPU preparation tasks per node require colocated CPU overlap')
    if sage_cpu_render_parallelism is not None:
        plan['sage_cpu_render_parallelism'] = sage_cpu_render_parallelism
    parallelism = plan.get('sage_cpu_render_parallelism', 1)
    if (type(parallelism) is not int or not 1 <= parallelism <= 16
            or (resource_mode != 'cpu_colocated' and parallelism != 1)):
        raise ValueError('SAGE CPU render parallelism must be 1..16; GPU mode requires 1')
    if vllm_tp_size is not None:
        if resource_mode != 'cpu_colocated' or vllm_tp_size not in (2, 4, 8):
            raise ValueError('vLLM topology override requires node-local cpu_colocated TP2/4/8')
        config['vllm_tp_size'] = vllm_tp_size
    if algorithm is not None:
        config['rule_credit_version'] = algorithm
        config['success_contract'] = ('native_success_and_stop'
            if algorithm == 'v4_boundary_stop_spl' else 'native_success')
    if architecture == 'v8':
        if algorithm not in (None, 'v3_first_arrival', 'v3_terminal_local'):
            raise ValueError('v8 requires a v3 reward version')
        config.update(harness_version='v8', rule_credit_version=algorithm or 'v3_first_arrival',
            tool_credit_mode='rule_call_loo', success_contract='native_success',
            max_tool_calls=50, terminal_success_reward=1.0, direction_match_reward=0.0,
            collision_penalty=0.0, discount_gamma=1.0,
            joint_text_fm_forward=False, multi_condition_batch_size=1,
            zero_copy_rollout_payload=False, direct_vllm_weight_sync=False,
            compact_zero_advantage_waves=False, fast_causal_attention=False,
            fm_suffix_vjp=False, fm_core_suffix_vjp=False, dual_shared_recompute=False)
        plan['pipeline_version'] = 'v8'
        config.setdefault('v8_prefix_cache', True)
    elif architecture not in (None, 'v7'):
        raise ValueError('unsupported architecture')
    if max_tool_calls is not None:
        if config.get('harness_version') != 'v8' or type(max_tool_calls) is not int or max_tool_calls < 1:
            raise ValueError('v8 tool-call budget must be a positive integer')
        config['max_tool_calls'] = max_tool_calls
    if nodes is not None:
        if nodes not in (4, 8):
            raise ValueError('the formal layout supports four or eight complete nodes')
        config.update(nnodes=nodes, learner_gpus_layout=','.join(['12'] * nodes),
            vllm_replicas=nodes, fast_policy_workers=nodes, rollout_actors=16 * nodes)
    if batch_size is not None:
        if batch_size <= 0 or batch_size % config['group_size']:
            raise ValueError('episode batch must be a positive multiple of group_size')
        groups = batch_size // config['group_size']
        old_groups = config['num_groups_per_step']
        sources = config.get('case_source_groups', {})
        if sources and any(groups * value % old_groups for value in sources.values()):
            raise ValueError('batch size must preserve the exact source group proportions')
        config['case_source_groups'] = {k: groups * v // old_groups for k, v in sources.items()}
        config.update(num_groups_per_step=groups, rollout_groups_in_flight=groups)
    for key, value in (('opt_steps', opt_steps), ('checkpoint_interval', checkpoint_interval)):
        if value is not None:
            if value <= 0:
                raise ValueError(key + ' must be positive')
            config[key] = value
    if rollout_actors_per_node is not None:
        if type(rollout_actors_per_node) is not int:
            raise ValueError('rollout actors per node must be an integer')
        config['rollout_actors'] = rollout_actors_per_node * config['nnodes']
    rollout_gyms_per_node(config)
    plan.update(nodes=config['nnodes'], train_trajectories_per_step=
        config['num_groups_per_step'] * config['group_size'],
        resource_layout=dict(learner=12 * config['nnodes'], vllm=2 * config['nnodes'],
            fast_policy=config['nnodes'], renderer=config['nnodes'],
            total=16 * config['nnodes'], ray_visible_per_node=15))
    if resource_mode is not None:
        if resource_mode != 'cpu_colocated' or config.get('harness_version') != 'v8':
            raise ValueError('cpu_colocated resource mode requires the v8 algorithm recipe')
        n = config['nnodes']
        tp = config['vllm_tp_size']
        if 16 % tp:
            raise ValueError('colocated TP replicas must fit complete node-local groups')
        config.update(learner_gpus_layout=','.join(['16'] * n),
            learner_cpus_per_rank=4, vllm_replicas=0, colocated_vllm_replicas=16*n//tp,
            fast_policy_device='cpu', fast_policy_cpus=8, fast_policy_workers=n,
            overlap_rollout_learner=False, rollout_background_prefetch=False,
            prepare_streamed_wave=False, optimizer_cpu_during_backward=False,
            text_before_fm=True, text_backend='megatron',
            megatron_tp=2, megatron_pp=2, megatron_cp=8, megatron_ep=16, megatron_etp=1)
        if inherited_fast_policy is not None:
            config['fast_policy_device'] = inherited_fast_policy
        plan['resource_mode'] = resource_mode
        plan['resource_layout'] = dict(learner=16*n, vllm=16*n, shared_gpu_pool=True,
            fast_policy=0, renderer=0, total=16*n, ray_visible_per_node=16,
            order=['complete_rollout', 'old_policy_fm_backward', 'megatron_text_backward',
                   'joint_clip', 'text_optimizer_step', 'fm_optimizer_step'])
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--arm', default='h00')
    p.add_argument('--resource-mode', choices=['cpu_colocated'])
    p.add_argument('--vllm-tp-size', type=int, choices=[2, 4, 8])
    p.add_argument('--sage-cpu-render-parallelism', type=int,
                   help='CPU renderer request concurrency; otherwise preserve the reference plan')
    p.add_argument('--allow-rollout-topology-change', action='store_true')
    p.add_argument('--overlap-cpu-group-preparation', action='store_true')
    p.add_argument('--cpu-group-preparation-tasks-per-node', type=int, choices=[1, 2])
    p.add_argument('--native-render-shared-cache', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--native-render-cache-max-gib', type=int)
    p.add_argument('--native-render-scene-load-reserve-gib', type=int)
    p.add_argument('--native-episode-cache', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--architecture', choices=['v7', 'v8'])
    p.add_argument('--algorithm', choices=['v3_first_arrival', 'v3_terminal_local', 'v4_boundary_stop_spl'])
    p.add_argument('--nodes', type=int, choices=[4, 8])
    p.add_argument('--rollout-actors-per-node', type=int, choices=[16, 32, 64],
                   help='Independent Gym sessions per node; does not change the episode batch')
    p.add_argument('--max-tool-calls', type=int, help='Explicit positive v8 tool-call budget override')
    p.add_argument('--batch-size', type=int, help='Number of complete sampled episodes per update')
    p.add_argument('--opt-steps', type=int)
    p.add_argument('--checkpoint-interval', type=int)
    p.add_argument('--resume', type=Path)
    p.add_argument('--native-admission', type=Path, help='Versioned reachable or filtered formal training pool directory')
    p.add_argument('--navigation-git')
    args = p.parse_args()
    root = args.output.resolve()
    if root == io.ART or not root.is_relative_to(io.ART) or io.exists(root):
        raise ValueError('use a fresh experiment on the authorized artifact disk')
    before = io.load(args.reference / 'launch/submit_plan.json')
    plan = remap(before, args.reference, root)
    # Historical reference plans contain copied-source paths and stale
    # validation commands. Only carry data, initialization and resource inputs.
    plan = {k: plan[k] for k in ('schema', 'config', 'job_type', 'gpu_per_node',
        'cpu_per_node', 'memory_per_node', 'priority', 'workspace',
        'initial_checkpoint', 'initialization', 'openloop_initialization',
        'pipeline_version', 'resource_mode', 'sage_cpu_render_parallelism',
        'cpu_group_preparation_tasks_per_node', 'native_render_shared_cache',
        'native_render_cache_max_gib', 'native_render_scene_load_reserve_gib',
        'native_episode_cache', 'rule_reward_coefficients', 'sequence_rloo',
        'freeze_action_head', 'freeze_text_policy', 'first_arrival_reward',
        'rloo_variant', 'rloo_hyperparameters', 'rloo_ablation',
        'checkpoint_retention_evaluation_root') if k in plan}
    if args.overlap_cpu_group_preparation:
        if not plan['config'].get('cpu_group_preparation'):
            raise ValueError('CPU overlap requires the existing complete-group CPU preparation recipe')
        plan['config']['overlap_cpu_group_preparation'] = True
    configure(plan, algorithm=args.algorithm, nodes=args.nodes, batch_size=args.batch_size,
              opt_steps=args.opt_steps, checkpoint_interval=args.checkpoint_interval, architecture=args.architecture,
              max_tool_calls=args.max_tool_calls, resource_mode=args.resource_mode,
              rollout_actors_per_node=args.rollout_actors_per_node, vllm_tp_size=args.vllm_tp_size,
              sage_cpu_render_parallelism=args.sage_cpu_render_parallelism,
              cpu_group_preparation_tasks_per_node=args.cpu_group_preparation_tasks_per_node,
              native_render_shared_cache=args.native_render_shared_cache,
              native_render_cache_max_gib=args.native_render_cache_max_gib,
              native_render_scene_load_reserve_gib=args.native_render_scene_load_reserve_gib,
              native_episode_cache=args.native_episode_cache)
    plan['config'].pop('allow_rollout_topology_change', None)
    if args.allow_rollout_topology_change:
        if not args.resume or args.vllm_tp_size is None:
            raise ValueError('topology-change admission requires an explicit resume and vLLM TP override')
        plan['config']['allow_rollout_topology_change'] = True
    if (args.algorithm in ('v3_first_arrival', 'v3_terminal_local') or args.architecture == 'v8') and args.arm != 'h00':
        raise ValueError('v3 baseline cannot include v4 ablation components')
    plan['config']['case_manifest'] = before['config']['case_manifest']
    # Initialization is explicit for each attempt; never inherit a resume
    # hidden in the reference plan when the caller requests a fresh run.
    for key in ('resume_from_path', 'policy_warm_start_checkpoint', 'allow_rule_credit_transition'):
        plan.pop(key, None)
    admission = None
    if args.native_admission:
        report_path = args.native_admission.resolve() / 'provenance.json'
        report = io.load(report_path)
        cases = report_path.with_name('train_cases.jsonl')
        if (report['schema'] not in ('native_rl_start_goal_admission_v1', 'three_source_navigation_pool_v2')
                or io.digest(io.read_bytes(cases)) != report['train_cases_sha256']):
            raise ValueError('native admission pool changed')
        plan['config']['case_manifest'] = str(cases)
        admission = dict(path=str(report_path), sha256=io.digest(io.read_bytes(report_path)))
        if report['schema'] == 'three_source_navigation_pool_v2':
            admission = apply_navigation_pool(plan['config'], report_path.parent)
    plan['output_root'] = str(root)
    plan['display_name'] = root.name
    root.mkdir(); launch = root / 'launch'; launch.mkdir()
    io.write_json(launch / 'submit_plan.json', plan)
    io.write_json(launch / 'config.json', plan['config'])
    reference_spec = (io.load(args.reference / 'launch/git_task.json')
        if io.exists(args.reference / 'launch/git_task.json') else {})
    spec = dict(repository='https://git.example.invalid/anonymous/closed_loop_3dgs_rl.git',
        commit=args.commit, checkout_root='/root/rl-code/' + root.name,
        output_root=str(root), kind='rl', launch_plan=str(launch / 'submit_plan.json'),
        fast_kernel_overlay=str(OVERLAY), components=list(ARMS.get(args.arm, ())),
        execution_profile='v7_r11_collision' if args.arm == 'a10' else 'v7_v3_collision',
        standby_after_failure=True,
        dependencies={
            'verl': dict(repository='https://git.example.invalid/anonymous/verl.git',
                commit='1633e1f51545b16b4751f86481b9616fcd23cb64'),
            'navigation': dict(repository=args.navigation_git or reference_spec.get(
                'dependencies', {}).get('navigation', {}).get('repository',
                    'https://github.com/alibaba-damo-academy/RynnBrain.git'),
                commit='40f19547941454835ff5e63c22994da74605cfea', subdirectory='navigation')})
    if plan['config'].get('text_backend') == 'megatron':
        spec['dependencies']['verl'] = dict(repository='https://git.example.invalid/local-user-03/verl.git',
                                           commit='b213f118f8df230bc97f68cdc56043623c6babdb')
    mapping = args.reference / 'launch/sage_source_objects/manifest.json'
    if io.exists(mapping):
        spec['source_object_map'] = dict(path=str(mapping), sha256=io.digest(io.read_bytes(mapping)))
    elif 'source_object_map' in reference_spec:
        spec['source_object_map'] = reference_spec['source_object_map']
    if 'source_object_map' not in spec:
        raise ValueError('training needs the bound SAGE source object map')
    binding = spec['source_object_map']
    if io.digest(io.read_bytes(binding['path'])) != binding['sha256']:
        raise ValueError('SAGE source object map changed')
    if admission:
        spec['native_admission'] = admission
    elif 'native_admission' in reference_spec:
        spec['native_admission'] = reference_spec['native_admission']
    if args.resume:
        spec['resume_from_path'] = str(args.resume.resolve())
    io.write_json(launch / 'git_task.json', spec)
    import shlex
    entry = Path(spec['checkout_root'])/'repo/scripts/train/run_git_task.py'
    (launch/'entry.sh').write_text('#!/bin/bash\nset -euo pipefail\nexec python3 -B ' +
        shlex.quote(str(entry)) + ' --spec ' + shlex.quote(str(launch/'git_task.json')) + '\n')
    print(json.dumps(dict(spec=str(launch / 'git_task.json'), source_copied=False,
        code_commit=args.commit, remote_submission=False)))


if __name__ == '__main__':
    main()
