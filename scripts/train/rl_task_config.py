"""Shared RL launch configuration without source packaging or submission imports."""
from pathlib import Path


OVERLAY = Path('/workspace/envs/qwen36_full_rl_fastkernels_native_py312/site-packages')

ARMS = {
    'a01': ('no_stop_supervision',),
    'a02': ('no_success_path',),
    'a03': ('no_chunk_collision',),
    'a04': ('joint_tool_choice',),
    'a05': ('unbalanced_stop',),
    'a06': ('no_first_arrival',),
    'a07': ('no_stop_supervision', 'joint_tool_choice'),
    'a08': ('no_success_path', 'no_chunk_collision'),
    'a09': ('no_first_arrival', 'no_success_path'),
    'stop_fixed_budget': ('fixed_budget_stop',),
}


def native_scene_cache_settings(plan, resource_mode=None):
    shared = plan.get('native_render_shared_cache', False)
    maximum = plan.get('native_render_cache_max_gib', 128)
    reserve = plan.get('native_render_scene_load_reserve_gib', 16)
    mode = resource_mode if resource_mode is not None else plan.get('resource_mode')
    if (type(shared) is not bool or type(maximum) is not int or type(reserve) is not int
            or not 1 <= reserve <= maximum <= 512
            or (shared and mode != 'cpu_colocated')):
        raise ValueError('native shared scene cache requires CPU colocation and a valid memory budget')
    return dict(shared=shared, max_gib=maximum, reserve_gib=reserve)


def rollout_gyms_per_node(config):
    """Keep every stateful rollout on its own Gym as admission scales."""
    nodes = config.get('nnodes', 4)
    actors = config.get('rollout_actors', 16 * nodes)
    if (type(nodes) is not int or nodes <= 0 or type(actors) is not int
            or actors % nodes or actors // nodes not in (16, 32, 64)):
        raise ValueError('rollout actors must provide 16, 32 or 64 independent gyms per node')
    return actors // nodes


def remap(value, old, new):
    if isinstance(value, str):
        return value.replace(str(old), str(new))
    if isinstance(value, list):
        return [remap(x, old, new) for x in value]
    if isinstance(value, dict):
        return {k: remap(x, old, new) for k, x in value.items()}
    return value


def apply_navigation_pool(config, directory):
    """Bind the admitted formal pool and exact 2:1:1 per-step group quotas."""
    import align_three_source_observations as io
    path = Path(directory).resolve() / 'provenance.json'
    raw = io.read_bytes(path)
    import json
    report = json.loads(raw)
    cases = path.with_name('train_cases.jsonl')
    if (report.get('schema') != 'three_source_navigation_pool_v2'
            or io.digest(io.read_bytes(cases)) != report['train_cases_sha256']
            or report.get('geometry_mode') != 'legacy_xy'):
        raise ValueError('formal navigation pool changed or uses another geometry')
    groups = config['num_groups_per_step']
    if type(groups) is not int or groups < 4 or groups % 4:
        raise ValueError('2:1:1 sampling requires group count divisible by four')
    quotas = dict(sage=groups//2, r2r=groups//4, rxr=groups//4)
    if any(report['eligible_counts'].get(k, 0) < n for k,n in quotas.items()):
        raise ValueError('filtered pool cannot supply distinct groups')
    config.update(case_manifest=str(cases), case_sampling='stratified_per_step',
                  case_source_groups=quotas)
    return dict(path=str(path), sha256=io.digest(raw))
