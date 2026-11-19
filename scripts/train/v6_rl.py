#!/usr/bin/env python3
"""Reviewable opt-in v6 joint-RL launcher; defaults to a CPU-only dry run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(os.environ.get('RL_ARTIFACT_ROOT', '/data/closed_loop_3dgs_rl'))
CONFIG = REPO / 'configs/action_head/v6_joint_rl.json'
TOOL_CREDIT_MODES = ('previous_value_gae', 'previous_value_mc', 'previous_value_mc_loo')


def build_plan(config: dict, *, head_checkpoint: str, query_checkpoint: str,
               bench_urls: str, output_dir: str, python: str = sys.executable,
               resume_from_path: str = '', allow_weights_only_resume: bool = False, tool_credit_mode: str | None = None,
               policy_warm_start_checkpoint: str = '',
               tool_credit_initial_state: str | None = None, allow_tool_credit_fork: bool = False,
               allow_rule_credit_transition: bool = False,
               tool_credit_fit_updates: int | None = None) -> dict:
    config = dict(config)
    if tool_credit_mode is not None:
        config['tool_credit_mode'] = tool_credit_mode
    if tool_credit_initial_state is not None:
        config['tool_credit_initial_state'] = tool_credit_initial_state
    if tool_credit_fit_updates is not None:
        config['tool_credit_fit_updates'] = tool_credit_fit_updates
    if config.get('schema') != 'v6_joint_rl_recipe_v1':
        raise ValueError('expected v6_joint_rl_recipe_v1')
    output = Path(output_dir).resolve()
    if not output.is_relative_to(ARTIFACTS) or output == ARTIFACTS:
        raise ValueError('v6 RL outputs must be an experiment directory on the data disk')
    if resume_from_path:
        if not Path(resume_from_path).is_absolute():
            raise ValueError('resume checkpoint must be an explicit absolute path')
        if output.exists():
            raise ValueError('exact resume requires a new output directory')
    if config.get('harness_version') == 'v8':
        config.setdefault('v8_prefix_cache', True)
        sys.path.insert(0, str(REPO / 'verl'))
        from action_head_rl.trajectory_text import validate_v8_options
        validate_v8_options(config)
    credit_mode = config.get('tool_credit_mode', 'episode_grpo')
    credit_seed = config.get('tool_credit_initial_state', '')
    if not isinstance(allow_rule_credit_transition, bool):
        raise ValueError('allow_rule_credit_transition must be an explicit boolean')
    if allow_rule_credit_transition and (
            not resume_from_path or policy_warm_start_checkpoint or allow_tool_credit_fork
            or credit_seed or credit_mode != 'rule_call_loo'
            or config.get('rule_credit_version') not in ('v3_reject_guard', 'v3_reject_guard_stop')):
        raise ValueError('rule-credit transition requires explicit full-state resume and a rejection-guard target')
    if policy_warm_start_checkpoint:
        if (resume_from_path or allow_tool_credit_fork or credit_seed
                or credit_mode not in TOOL_CREDIT_MODES):
            raise ValueError('policy warm-start requires previous-value credit and cannot mix resume/fork/independent critic seed')
        if not Path(policy_warm_start_checkpoint).is_absolute():
            raise ValueError('policy warm-start checkpoint must be an explicit absolute path')
        if output.exists():
            raise ValueError('policy warm-start requires a new output directory')
        source = Path(policy_warm_start_checkpoint).resolve()
        if (Path(head_checkpoint).resolve() != source / 'action_head'
                or Path(query_checkpoint).resolve() != source / 'dualvln_query'
                or Path(config['slow_model']).resolve() != source / 'qwen'):
            raise ValueError('policy warm-start Qwen/query/FM arguments must all match the source checkpoint')
    if credit_mode not in ('episode_grpo', 'rule_call_loo', *TOOL_CREDIT_MODES):
        raise ValueError('unsupported tool-credit mode')
    if credit_seed and not Path(credit_seed).is_absolute():
        raise ValueError('tool-credit initial state must be an explicit absolute path')
    if allow_tool_credit_fork and (
            not resume_from_path or credit_mode not in TOOL_CREDIT_MODES or not credit_seed):
        raise ValueError('tool-credit fork requires explicit resume, previous-value credit and initial critic state')
    if credit_mode in TOOL_CREDIT_MODES and not (resume_from_path or policy_warm_start_checkpoint):
        raise ValueError('tool credit requires an exact actor checkpoint resume or explicit policy warm-start')
    if credit_mode == 'previous_value_mc_loo' and config.get('group_size', 0) < 2:
        raise ValueError('MC/LOO credit requires group_size >= 2')
    if credit_seed and not allow_tool_credit_fork:
        raise ValueError('initial critic state requires explicit --allow-tool-credit-fork')
    if config.get('success_contract') not in ('native_success', 'native_success_and_stop'):
        raise ValueError('v6 requires an explicit native success contract')
    for key in ('slow_model', 'fast_backbone'):
        if not Path(config[key]).is_absolute():
            raise ValueError(f'{key} must be absolute')
    if not Path(head_checkpoint).is_absolute() or not Path(query_checkpoint).is_absolute():
        raise ValueError('head and query checkpoints must be explicit absolute paths')
    allowed = {
        'harness_version', 'slow_model', 'fast_backbone', 'dataset_split', 'nnodes', 'learner_gpus_layout',
        'vllm_replicas', 'vllm_tp_size', 'fast_policy_workers', 'rollout_actors',
        'colocated_vllm_replicas', 'fast_policy_device', 'fast_policy_cpus', 'learner_cpus_per_rank',
        'num_groups_per_step', 'group_size', 'smoke_cases', 'opt_steps', 'checkpoint_interval',
        'max_tool_calls', 'learning_rate', 'vlm_learning_rate', 'text_loss_coef',
        'text_temperature', 'terminal_success_reward', 'direction_match_reward',
        'collision_penalty', 'discount_gamma', 'process_reward_cap', 'success_contract',
        'ray_address', 'case_manifest', 'case_shuffle_seed', 'case_sampling',
        'case_sampling_seed', 'case_source_groups', 'eval_case_manifest',
        'eval_bench_urls', 'eval_interval', 'eval_repeats', 'eval_seed', 'eval_max_tool_calls',
        'weight_decay', 'rollout_scheduler', 'rollout_groups_in_flight',
        'fast_policy_max_batch_size', 'stage_records_per_rpc',
        'vllm_max_num_seqs', 'vllm_max_num_batched_tokens', 'vllm_gpu_memory_utilization',
        'keep_last_checkpoints',
        'learner_parameter_dtype', 'rollout_transport_dtype', 'fsdp_size', 'full_weight_residency_gib',
        'full_attention_checkpoint_layers_off',
        'tool_credit_mode', 'tool_credit_initial_state', 'tool_credit_fit_updates', 'rule_credit_version',
        'joint_fixture_dir', 'joint_fixture_count', 'joint_fixture_step', 'update_replay_dir',
        'update_backend_audit_input',
        'multi_condition_batch_size', 'max_condition_tokens', 'condition_token_bucket_size',
        'stream_groups_per_update', 'weight_sync_bucket_mb',
        'text_backend', 'megatron_tp', 'megatron_pp', 'megatron_cp', 'megatron_ep', 'megatron_etp',
        'megatron_max_token_len_per_gpu',
    }
    flags = {'overlap_cpu_group_preparation', 'fm_prefix_reuse', 'megatron_cpu_prefetch', 'cpu_group_preparation', 'native_colocated_weight_sync', 'group_partial_rollout', 'trajectory_transfer_queue', 'megatron_dynamic_bsz', 'learner_collectives_only', 'text_before_fm', 'fm_suffix_vjp', 'fm_core_suffix_vjp', 'dual_shared_recompute', 'fast_causal_attention', 'fast_policy_pipeline', 'token_balanced_sharding', 'zero_copy_rollout_payload',
             'skip_zero_advantage_records', 'joint_text_fm_forward', 'prepare_streamed_wave', 'local_policy_transport',
             'bf16_gather_before_export', 'overlap_rollout_learner', 'direct_vllm_weight_sync',
             'rollout_background_prefetch', 'compact_zero_advantage_waves', 'v8_prefix_cache', 'v8_cuda_graphs',
             'gradient_checkpointing', 'fsdp_forward_prefetch', 'optimizer_cpu_during_backward',
             'defer_query_capture', 'allow_rollout_topology_change'}
    unknown = set(config) - allowed - flags - {'schema'}
    if unknown:
        raise ValueError(f'unknown v6 RL recipe fields: {sorted(unknown)}')
    if type(config.get('allow_rollout_topology_change', False)) is not bool:
        raise ValueError('allow_rollout_topology_change must be an explicit boolean')
    if type(config.get('overlap_cpu_group_preparation', False)) is not bool:
        raise ValueError('overlap_cpu_group_preparation must be an explicit boolean')
    if config.get('overlap_cpu_group_preparation') and not config.get('cpu_group_preparation'):
        raise ValueError('CPU preparation overlap requires CPU group preparation')
    if config.get('allow_rollout_topology_change') and (
            not resume_from_path or policy_warm_start_checkpoint or allow_tool_credit_fork
            or allow_rule_credit_transition or config.get('harness_version') != 'v8'
            or config.get('text_backend') != 'megatron'):
        raise ValueError('rollout topology change requires explicit v8 Megatron full-state resume')
    if not isinstance(config.get('megatron_cpu_prefetch', False), bool):
        raise ValueError('megatron_cpu_prefetch must be a boolean')
    if config.get('megatron_cpu_prefetch') and config.get('text_backend') != 'megatron':
        raise ValueError('text CPU prefetch requires the Megatron text backend')
    if not isinstance(config.get('fm_prefix_reuse', False), bool):
        raise ValueError('fm_prefix_reuse must be a boolean')
    if config.get('fm_prefix_reuse') and config.get('text_backend') != 'megatron':
        raise ValueError('FM prefix reuse requires the Megatron text backend')
    if ('megatron_max_token_len_per_gpu' in config and
            (type(config['megatron_max_token_len_per_gpu']) is not int
             or config['megatron_max_token_len_per_gpu'] <= 0)):
        raise ValueError('megatron_max_token_len_per_gpu must be a positive integer')
    if not isinstance(config.get('rollout_background_prefetch', False), bool):
        raise ValueError('rollout_background_prefetch must be a boolean')
    if config.get('prepare_streamed_wave', False) and not config.get('overlap_rollout_learner', False):
        raise ValueError('prepare_streamed_wave requires streamed training')
    if config.get('rollout_background_prefetch', False) and (
            not config.get('overlap_rollout_learner', False)
            or config.get('rollout_scheduler') != 'completion'):
        raise ValueError('background rollout prefetch requires streamed training and completion scheduling')
    if not isinstance(config.get('compact_zero_advantage_waves', False), bool):
        raise ValueError('compact_zero_advantage_waves must be a boolean')
    if config.get('compact_zero_advantage_waves', False) and (
            not config.get('skip_zero_advantage_records', False)
            or not config.get('overlap_rollout_learner', False)
            or config.get('multi_condition_batch_size', 1) != 1
            or credit_mode != 'episode_grpo'):
        raise ValueError('zero-wave compaction requires filtered B1 streamed episode GRPO')
    for key in ('multi_condition_batch_size', 'stream_groups_per_update', 'weight_sync_bucket_mb'):
        if key in config and (type(config[key]) is not int or config[key] <= 0):
            raise ValueError(f'{key} must be a positive integer')
    for key in ('max_condition_tokens', 'condition_token_bucket_size'):
        if key in config and (type(config[key]) is not int or config[key] < 0):
            raise ValueError(f'{key} must be a nonnegative integer')
    if config.get('joint_text_fm_forward', False) and (
            config.get('multi_condition_batch_size', 1) not in (1, 2, 4, 8)
            or (config.get('multi_condition_batch_size', 1) > 1 and credit_mode != 'rule_call_loo')
            or (config.get('overlap_rollout_learner', False) and credit_mode != 'rule_call_loo')
            or config.get('condition_token_bucket_size', 0)):
        raise ValueError('joint text/FM forward requires B1/B2/B4/B8, unbucketed training; streaming requires rule credit')
    if (credit_mode != 'episode_grpo' and config.get('multi_condition_batch_size', 1) != 1
            and not (credit_mode == 'rule_call_loo' and config.get('joint_text_fm_forward', False))):
        raise ValueError('per-call credit currently requires B1 training')
    if (credit_mode not in ('episode_grpo', 'rule_call_loo') and config.get('overlap_rollout_learner', False)
            and config.get('skip_zero_advantage_records', False)):
        raise ValueError('streamed per-call credit does not support zero-advantage filtering')
    if config.get('direct_vllm_weight_sync', False) and (
            config.get('local_policy_transport', False)
            or config.get('bf16_gather_before_export', False)):
        raise ValueError('direct weight sync cannot use file-transport optimizations')
    if not isinstance(config.get('bf16_gather_before_export', False), bool):
        raise ValueError('bf16_gather_before_export must be a boolean')
    if config.get('bf16_gather_before_export', False) and (
            config.get('learner_parameter_dtype', 'bf16') != 'fp32'
            or config.get('rollout_transport_dtype', 'preserve') != 'bf16'):
        raise ValueError('BF16 gather before export requires FP32 learner and BF16 rollout transport')
    if config.get('local_policy_transport', False):
        if config.get('nnodes', 1) != 1 or config.get('ray_address') != 'local':
            raise ValueError('local policy transport requires nnodes=1 and ray_address=local')
        if config.get('rollout_transport_dtype', 'preserve') != 'bf16':
            raise ValueError('local policy transport requires BF16 rollout transport')
    command = [python, '-B', '-m', 'action_head_rl.trainer', '--repo', str(REPO),
               '--head-checkpoint', head_checkpoint, '--query-checkpoint', query_checkpoint,
               '--bench-urls', bench_urls, '--output-dir', str(output),
               '--harness-version', config.get('harness_version', 'v6'), '--train-scope', 'full', '--text-policy',
               '--episode-grpo', '--old-logprob-mode', 'single_forward',
               '--r3-capture-mode', 'native', '--option-condition-reuse',
               '--multi-condition-batch-size', str(config.get('multi_condition_batch_size', 1)), '--resume-mode',
               'resume_path' if resume_from_path else 'disable']
    if not config.get('joint_text_fm_forward', False):
        command.append('--no-joint-text-fm-forward')
    if resume_from_path:
        command.extend(['--resume-from-path', resume_from_path])
        if allow_weights_only_resume:
            command.append('--allow-weights-only-resume')
    if allow_tool_credit_fork:
        command.append('--allow-tool-credit-fork')
    if allow_rule_credit_transition:
        command.append('--allow-rule-credit-transition')
    if policy_warm_start_checkpoint:
        command.extend(['--policy-warm-start-checkpoint', policy_warm_start_checkpoint])
    for key in sorted((allowed & set(config)) - {'multi_condition_batch_size', 'harness_version'}):
        value = json.dumps(config[key], sort_keys=True) if key == 'case_source_groups' else str(config[key])
        command.extend(['--' + key.replace('_', '-'), value])
    for key in sorted(flags & set(config)):
        if not isinstance(config[key], bool):
            raise ValueError(f'{key} must be a boolean')
        if config[key]:
            command.append('--' + key.replace('_', '-'))
        elif key in ('gradient_checkpointing', 'fsdp_forward_prefetch', 'prepare_streamed_wave',
                     'v8_prefix_cache'):
            command.append('--no-' + key.replace('_', '-'))
    # Ray Unix-domain sockets have a short path limit; descriptive run names do not fit.
    runtime = Path('/tmp/action_head_rl') / ('l-' + hashlib.sha256(str(output).encode()).hexdigest()[:12])
    cache = runtime / 'cache'
    cache_keys = ('TMPDIR', 'XDG_CACHE_HOME', 'HF_HOME', 'TRITON_CACHE_DIR',
                  'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH', 'PYTORCH_KERNEL_CACHE_PATH',
                  'TORCH_EXTENSIONS_DIR')
    env = {key: str(cache / key.lower()) for key in cache_keys}
    env.update(PYTHONPATH=os.pathsep.join((str(REPO / 'verl'),
                    os.environ.get('VERL_SOURCE_ROOT', str(REPO / 'verl')),
                    str(REPO / 'vln_demo/src'), os.environ.get('PYTHONPATH', ''))),
               PYTHONDONTWRITEBYTECODE='1', ACTION_HEAD_TEXT_POLICY='1',
               ACTION_HEAD_FINAL_CONTRACT=('three-source-boundary-stop-v1'
                   if config.get('rule_credit_version') == 'v4_boundary_stop_spl' else ''),
               ACTION_HEAD_OLD_LOGPROB_MODE='single_forward',
               ACTION_HEAD_JOINT_TEXT_FM_FORWARD='1' if config.get('joint_text_fm_forward', False) else '0',
               ACTION_HEAD_COMPACT_ZERO_ADVANTAGE_WAVES='1' if config.get('compact_zero_advantage_waves', False) else '0',
               ACTION_HEAD_LOCAL_RUNTIME_ROOT=str(runtime),
               ACTION_HEAD_LOCAL_RAY_TEMP_DIR=str(runtime / 'ray'),
               ACTION_HEAD_LOCAL_RAY_IPC_DIR=str(runtime / 'ipc'),
               ACTION_HEAD_RAY_CACHE_ROOT=str(cache / 'ray'),
               ACTION_HEAD_COMPILE_CACHE_ROOT=str(cache / 'vllm'),
               ACTION_HEAD_PROCESS_COMPILE_CACHE_ROOT=str(cache / 'process_compile'),
               RTC_CACHE_PATH=str(cache / 'rtc'), PPU_LIB_RTC_CACHE_PATH=str(cache / 'rtc'),
               MPLCONFIGDIR=str(cache / 'matplotlib'), NUMBA_CACHE_DIR=str(cache / 'numba'),
               ACTION_HEAD_VLLM_WEIGHT_SYNC='inplace',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return {'schema': 'v6_joint_rl_launch_plan_v1', 'command': command, 'env': env,
            'output_dir': str(output), 'default_action': 'dry_run',
            'gpu_validated': False,
            'objective': 'episode_mean_fm_plus_weighted_episode_mean_text_gpg',
            'initialization': ('joint_policy_and_critic_warm_start_new_actor_optimizer_scheduler_rng'
                               if policy_warm_start_checkpoint else
                               'rule_credit_transition_with_exact_actor_optimizer_scheduler_rng'
                               if allow_rule_credit_transition else
                               'algorithm_fork_with_exact_actor_optimizer_scheduler_rng'
                               if allow_tool_credit_fork else
                               'weights_only_resume_with_fresh_optimizer_scheduler_rng'
                               if resume_from_path and allow_weights_only_resume else
                               'exact_checkpoint_resume_with_optimizer_scheduler_rng'
                               if resume_from_path else
                               'explicit_weight_warm_start_not_old_optimizer_resume'),
            **({'resume_from_path': resume_from_path} if resume_from_path else {}),
            **({'allow_rule_credit_transition': True, 'same_training_objective': False,
                'rule_credit_version': config['rule_credit_version']}
               if allow_rule_credit_transition else {}),
            **({'policy_warm_start_checkpoint': policy_warm_start_checkpoint,
                'resolved_policy_paths': {'qwen': config['slow_model'],
                                          'query': query_checkpoint, 'head': head_checkpoint,
                                          'critic': str(Path(policy_warm_start_checkpoint) / 'tool_credit_critic.json')}}
               if policy_warm_start_checkpoint else {})}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--head-checkpoint', required=True)
    parser.add_argument('--query-checkpoint', required=True)
    parser.add_argument('--bench-urls', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--resume-from-path', default='',
                        help='explicit complete checkpoint; resumes optimizer/scheduler/RNG into a new output directory')
    parser.add_argument('--allow-weights-only-resume', action='store_true')
    parser.add_argument('--policy-warm-start-checkpoint', default='',
                        help='explicit bound previous-value joint policy; new actor optimizer/scheduler/RNG in a new experiment')
    parser.add_argument('--tool-credit-mode', choices=('episode_grpo', 'rule_call_loo', *TOOL_CREDIT_MODES))
    parser.add_argument('--tool-credit-initial-state')
    parser.add_argument('--tool-credit-fit-updates', type=int)
    parser.add_argument('--allow-tool-credit-fork', action='store_true')
    parser.add_argument('--allow-rule-credit-transition', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true', help='explicitly execute the reviewed training command')
    mode.add_argument('--validate-only', action='store_true', help='run trainer preflight; never initialize Ray')
    mode.add_argument('--dry-run', action='store_true', help='default; print only, no model or service loading')
    args = parser.parse_args(argv)
    plan = build_plan(json.loads(args.config.read_text()), head_checkpoint=args.head_checkpoint,
                      query_checkpoint=args.query_checkpoint, bench_urls=args.bench_urls,
                      output_dir=args.output_dir, resume_from_path=args.resume_from_path,
                      allow_weights_only_resume=args.allow_weights_only_resume,
                      policy_warm_start_checkpoint=args.policy_warm_start_checkpoint,
                      tool_credit_mode=args.tool_credit_mode,
                      tool_credit_initial_state=args.tool_credit_initial_state,
                      allow_tool_credit_fork=args.allow_tool_credit_fork,
                      allow_rule_credit_transition=args.allow_rule_credit_transition,
                      tool_credit_fit_updates=args.tool_credit_fit_updates)
    if not args.run and not args.validate_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    command = plan['command'] + (['--validate-only'] if args.validate_only else [])
    env = {**os.environ, **plan['env']}
    if args.validate_only:
        env['CUDA_VISIBLE_DEVICES'] = ''
    else:
        if os.environ.get('CUDA_VISIBLE_DEVICES') == '':
            raise ValueError('CUDA_VISIBLE_DEVICES is empty; use --dry-run or --validate-only for CPU checks')
        for key in ('TMPDIR', 'XDG_CACHE_HOME', 'HF_HOME', 'TRITON_CACHE_DIR',
                    'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH', 'PYTORCH_KERNEL_CACHE_PATH',
                    'TORCH_EXTENSIONS_DIR'):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
    return subprocess.run(command, env=env, check=False, cwd=REPO).returncode


if __name__ == '__main__':
    raise SystemExit(main())
