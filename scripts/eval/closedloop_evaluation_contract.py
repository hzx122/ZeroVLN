"""One navigation evaluation protocol, independent of checkpoint provenance."""
import copy
import hashlib
import json

VERSION = 'unified-closedloop-coarse4-rl-layout100-20261021-v13'
V8_VERSION = 'unified-closedloop-v8-coarse4-rl-layout100-20261021-v13'
LEGACY_CONTINUATION = 'openloop-20261012-native100-continuation'
BENCHMARKS = ('sage-3d', 'r2r', 'rxr')
PROFILES = {
    'sage-3d': dict(width=640, height=480, hfov_deg=115, height_m=1.2, pitch_deg=0),
    'r2r': dict(width=640, height=480, hfov_deg=120, height_m=1.25, pitch_deg=0),
    'rxr': dict(width=640, height=480, hfov_deg=120, height_m=1.25, pitch_deg=0),
}
TOOL_BUDGETS = dict.fromkeys(BENCHMARKS, 100)
ENVIRONMENT = dict(SAGE_GEOMETRY_MODE='legacy_xy')
SERVICE_ENVIRONMENT = dict(ACTION_HEAD_EXECUTION_PROFILE='v7_v3_collision',
    ACTION_HEAD_EVAL_DISABLE_WALL_CLOCK='1',
    # Preserve the established evaluator's free tool generation and feedback.
    # Clear inherited RL grammar flags in both the actor and model processes.
    ACTION_HEAD_EXECUTABLE_POLICY='', ACTION_HEAD_FINAL_CONTRACT='',
    COMPANION_BRAIN_MODE='sft', RULE_TEACHER_SAMPLING_PROFILE='',
    ACTION_HEAD_QUERY_TEXT_BUDGET='256', VLLM_MAX_MODEL_LEN='131072',
    # v6-style coarse route labels: execute the first 4 of 16 points (user 2026-10-20).
    ACTION_HEAD_EVAL_PREFIX_STEPS='4',
    ACTION_HEAD_TOKENIZE_TIMEOUT_SECONDS='120', ACTION_HEAD_TOKENIZE_ATTEMPTS='2')
# Verified running RL R20: 630a9884fe023a21480917c05941d614d3ba2b13.
# Borrow its resource placement; evaluation keeps its deterministic FM sampler.
ROLLOUT_SOURCE_COMMIT = '630a9884fe023a21480917c05941d614d3ba2b13'
LAYOUTS = {bench: dict(gym_actors=64, render_gpus=[], fm_gpus=[0],
    brain_replicas=8, renderer_device='cpu', renderer_cache_scenes=32,
    render_parallelism=8 if bench == 'sage-3d' else 4,
    native_shared_cache=True, native_cache_max_gib=128, scene_load_reserve_gib=16,
    max_batch_size=8, batch_wait_ms=12) for bench in BENCHMARKS}

PROTOCOL = dict(version=VERSION, observation_profiles=PROFILES,
    max_tool_calls=TOOL_BUDGETS, environment=ENVIRONMENT,
    service_environment=SERVICE_ENVIRONMENT, seed=20261008,
    policy_seed=20260827, repeats=4, gym_actors=64, pooled_attempts=True,
    prompt=dict(version='v6', mode='instruct', keep_last_images=8,
                one_tool_call_per_turn=True, invalid_tool_call_policy='feedback'),
    executor=dict(profile='v7_v3_collision', variable_scale=True, prefix_steps=4,
                  fast_step_limit=4, max_sim_steps=64, max_seconds=None,
                  finite_and_structure_required=True),
    sampling=dict(enable_thinking=False,temperature=0.7,top_p=0.95,
        text_budget=256,query_reserve='checkpoint query count + 1',request_seed='omitted',
        server_seed='20262013 + rank*10 + replica; shared by all pooled attempts of a checkpoint',
        omitted_parameters=['top_k','min_p','presence_penalty']),
    layouts=LAYOUTS, rollout_source_commit=ROLLOUT_SOURCE_COMMIT,
    services=dict(brain_replicas=8,tensor_parallel=2,max_num_seqs=1024,
        max_num_batched_tokens=131072,max_model_len=131072,gpu_memory_utilization=0.60),
    scoring=dict(sage='existing official SAGE scorer',
                 r2r='existing native scorer', rxr='existing native scorer'),
    metric_definition=dict(pass_at_1='successes / all four valid attempts',
                           pass_at_4='cases with any success / fixed cases'))


def fm_devices(config, default):
    """Explicit FM placement; preserve the established layout when omitted."""
    gpus = config.get('serving', {}).get('fm_gpus', default)
    if (not isinstance(gpus, list) or not gpus
            or any(type(gpu) is not int or not 0 <= gpu < 16 for gpu in gpus)
            or len(set(gpus)) != len(gpus)):
        raise ValueError('fm_gpus must name distinct devices on the 16-GPU node')
    return list(gpus)


def tool_budgets(value=None):
    """Explicit positive call limits; omitted datasets retain the default."""
    if value is None:
        return copy.deepcopy(TOOL_BUDGETS)
    if type(value) is int:
        value = dict.fromkeys(BENCHMARKS, value)
    if (not isinstance(value, dict) or not value or set(value)-set(BENCHMARKS)
            or any(type(v) is not int or v < 1 for v in value.values())):
        raise ValueError('max_tool_calls must be a positive integer or benchmark-to-positive-integer mapping')
    return {**TOOL_BUDGETS, **value}


def protocol(version=VERSION, repeats=4, compaction=None, max_tool_calls=None):
    if type(repeats) is not int or repeats < 1:
        raise ValueError("evaluation_repeats must be a positive integer")
    value = copy.deepcopy(PROTOCOL)
    value["repeats"] = repeats
    if repeats != 4:
        value['metric_definition'] = dict(pass_at_1=f'successes / all {repeats} valid attempts')
        if repeats > 1:
            value['metric_definition'][f'pass_at_{repeats}'] = 'cases with any success / fixed cases'
    value['max_tool_calls'] = tool_budgets(max_tool_calls)
    if version == LEGACY_CONTINUATION:
        value['version'] = version
        value['max_tool_calls'] = {'sage-3d': 200, 'r2r': 100, 'rxr': 100}
        value['sampling']['server_seed'] = 'first round omitted; later 20261013 + repeat*1000 + rank*10 + replica'
    elif version in (VERSION, V8_VERSION):
        value['version'] = version
        settings = dict(frequency=4, recent=8, reserve_tokens=8192, window=131072)
        if compaction is not None:
            if not isinstance(compaction, dict) or set(compaction)-set(settings):
                raise ValueError('unknown context_compaction settings')
            settings.update(compaction)
        if (any(type(v) is not int or v <= 0 for v in settings.values())
                or not settings['reserve_tokens'] + 1024 < settings['window']
                or settings['window'] != value['services']['max_model_len']):
            raise ValueError('invalid context_compaction settings')
        value['prompt']['automatic_compression'] = dict(
            revision='v8_complete_turn_stride_v1', **settings)
        value['service_environment'].update(
            ACTION_HEAD_EVAL_CONTEXT_COMPACTION='complete_turn_stride',
            ACTION_HEAD_EVAL_CONTEXT_FREQUENCY=str(settings['frequency']),
            ACTION_HEAD_EVAL_CONTEXT_RECENT=str(settings['recent']),
            ACTION_HEAD_EVAL_CONTEXT_RESERVE=str(settings['reserve_tokens']),
            ACTION_HEAD_EVAL_CONTEXT_WINDOW=str(settings['window']))
        if version == V8_VERSION:
            value['prompt'].update(version='v8', keep_last_images=None,
                continuous_native_prefix=True)
            value['service_environment'].update(RULE_TEACHER_CONTEXT_VERSION='v8',
            ACTION_HEAD_CONTINUOUS_CONTEXT='1', ACTION_HEAD_R3_CAPTURE_MODE='native',
            ACTION_HEAD_CAPTURE_ROUTING='1', ACTION_HEAD_CAPTURE_FULL_ROUTING='1',
            ACTION_HEAD_BULK_ROUTE_CAPTURE='1', ACTION_HEAD_V8_PREFIX_CACHE='1',
            # Each observation can contain a map and a first-person image,
            # including the initial observation. Admission must cover the
            # entire call budget before token-based compaction is triggered.
            VLLM_ENABLE_PREFIX_CACHING='1',
            VLLM_MAX_IMAGES=str(2 * (max(value['max_tool_calls'].values()) + 1)),
            ACTION_HEAD_V8_CUDA_GRAPHS='1',
            VLLM_MAX_MODEL_LEN='131072')
    elif version != VERSION:
        raise ValueError('unknown closed-loop protocol: ' + str(version))
    return value


def contract_for(config):
    version = config.get('closedloop_protocol', VERSION)
    return dict(version=version, sha256=identity(version, config.get('evaluation_repeats', 1), config.get('context_compaction'), config.get('max_tool_calls')))


def identity(version=VERSION, repeats=4, compaction=None, max_tool_calls=None):
    return hashlib.sha256(json.dumps(protocol(version, repeats, compaction, max_tool_calls), sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def selection_config(value):
    """Reject per-route semantic overrides rather than silently accepting them."""
    config = copy.deepcopy(value)
    harness = config.pop('harness_version', 'v6')
    if harness not in ('v6', 'v8'):
        raise ValueError('harness_version must be v6 or v8')
    version = config.pop('continuation_protocol', V8_VERSION if harness == 'v8' else VERSION)
    if harness == 'v8' and version != V8_VERSION:
        raise ValueError('v8 cannot reuse historical continuation results')
    if version == LEGACY_CONTINUATION:
        raise ValueError('historical continuation cannot start a new closed-loop evaluation')
    selected = protocol(version, config.get('evaluation_repeats', 4), config.get('context_compaction'), config.get('max_tool_calls'))
    if version in (VERSION, V8_VERSION):
        config['context_compaction'] = {k:v for k,v in selected['prompt']['automatic_compression'].items() if k != 'revision'}
    config['evaluation_repeats'] = selected['repeats']
    # A stale per-run flag must not reintroduce the removed time cutoff.
    config['disable_action_head_wall_clock'] = True
    allowed={'evaluation_repeats','nodes','checkpoints','default_backbone','default_brain','training_config',
        'benchmarks','sample_counts','seed','max_tool_calls','gym_actors','environment','disable_action_head_wall_clock',
        'observation_profiles','sage_manifest','sage_protocol_root','navmesh_root','vlnce_eval_set','context_compaction'}
    if set(config)-allowed:
        raise ValueError('unsupported unified selection fields: '+str(sorted(set(config)-allowed)))
    names = config.get('benchmarks', list(BENCHMARKS))
    if not names or names != [b for b in BENCHMARKS if b in names]:
        raise ValueError('closed-loop evaluation supports ordered SAGE-3D/R2R/RxR only')
    config['max_tool_calls'] = selected['max_tool_calls']
    canonical = dict(observation_profiles=PROFILES,
                     environment=ENVIRONMENT, seed=PROTOCOL['seed'], gym_actors=64)
    for key, expected in canonical.items():
        if key in config and config[key] != expected:
            raise ValueError('shared closed-loop setting cannot differ: ' + key)
        config[key] = copy.deepcopy(expected)
    for checkpoint in config.get('checkpoints', []):
        if checkpoint.get('policy', 'action_head') != 'action_head':
            raise ValueError('unified action-head closed-loop entry requires action-head checkpoints')
        for key in ('observation_profiles', 'environment', 'max_model_len'):
            if key in checkpoint:
                raise ValueError('checkpoint may select weights, not override evaluation settings: ' + key)
    config['benchmarks'] = names
    config['closedloop_protocol'] = version
    config['pooled_attempts'] = True
    return config


def verify_normalized(config):
    if config.get('pooled_attempts') is not True:
        raise ValueError('unified evaluation requires pooled attempts')
    for key, expected in [('max_tool_calls', tool_budgets(config.get('max_tool_calls'))), ('seed', PROTOCOL['seed']),
                          ('gym_actors', 64), ('environment', ENVIRONMENT)]:
        if config.get(key) != expected:
            raise ValueError('unified closed-loop protocol mismatch: ' + key)
    for checkpoint in config['checkpoints']:
        for bench in config['benchmarks']:
            if checkpoint['observation_profiles'][bench] != PROFILES[bench]:
                raise ValueError('checkpoint camera differs from evaluation contract: ' + bench)
        if checkpoint['environment'] != ENVIRONMENT:
            raise ValueError('checkpoint environment differs from shared evaluation contract')


def stage_repeats(config):
    return config.get('evaluation_repeats', 1) if config.get('pooled_attempts') else 1
