"""Execute an entire grounded ECM route; geometry determines action expert scale.

The legacy route executes four points per refresh. The optional fine route
subdivides those same segments into sixteen points, preserving refresh endpoints.
Physical and training audits keep targets separate from measured simulator poses.
"""
from __future__ import annotations

import math
import os

import numpy as np

from action_head.render_client import _world_pose
from action_head.runtime import ActionHeadRuntimeConfig, ActionHeadToolRuntime
from privileged_teacher import RoutePolicy, TeacherRejected, wrap
from sage.planner import _path_is_safe, _resample_polyline
from vln_gym.fm_action import decompose, world_to_chunk_pose


def _length(path):
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def local_approach_feedback(execution):
    """Describe a completed local approach using measured execution evidence.

    This does not advance a later clause, change admission, or remove holds.
    Endpoint completion is distinct from completing the whole navigation plan.
    """
    state = execution.get('state') or {}
    binding = state.get('binding') or {}
    steps = (state.get('program') or {}).get('steps', [])
    if (not execution.get('ok') or not state.get('full_route_complete')
            or state.get('tracking_failure') or len(steps) != 1
            or steps[0].get('kind') != 'approach'
            or binding.get('match_branch') != 'target'):
        return None
    from sft.rule_teacher_progress import measured_poses
    try:
        poses, _ = measured_poses(execution)
    except ValueError:
        return None
    if len(poses) != state.get('route_prefix_steps', 4)*state.get('route_chunks', 0)+1:
        return None
    held = all(pose == poses[0] for pose in poses)
    target = (binding.get('planning_call') or execution.get('original_call') or {}).get('target')
    return dict(scope='local_approach_endpoint', target=target,
        status='endpoint_already_reached' if held else 'local_approach_complete',
        actual_path_m=_length(np.asarray(poses)[:, :2]),
        remaining_plan_steps_executed=False,
        detail=('No motion was measured; the local approach endpoint was already reached. '
                if held else 'Only the local approach was executed. ')
               + 'Other steps in the requested plan were not executed by this call.')


def _faithful(original, sampled, tolerance=.025):
    """Bound the deviation of every original vertex from its resampled chord."""
    arcs = np.r_[0., np.cumsum(np.linalg.norm(np.diff(original, axis=0), axis=1))]
    bins = np.linspace(0., arcs[-1], len(sampled))
    for i, (a, b) in enumerate(zip(sampled[:-1], sampled[1:])):
        pts = original[(arcs >= bins[i]-1e-9) & (arcs <= bins[i+1]+1e-9)]
        if not len(pts):
            continue
        delta = b-a
        ratio = np.clip((pts-a) @ delta / max(float(delta @ delta), 1e-20), 0., 1.)
        if np.linalg.norm(pts-(a+ratio[:, None]*delta), axis=1).max() > tolerance:
            return False
    return True


def _fixed_count_route(original, occ, count, *, safe_route=None):
    """Keep necessary bends, then subdivide safe legs to the requested count.

    Reuse the planner's collision checks and resampler. Distances are allowed
    to vary: uniform arclength sampling can miss a narrow doorway corner.
    """
    if len(original) == 1 or _length(original) < 1e-8:
        return np.repeat(original[-1:], count, axis=0)
    selected = [0, len(original)-1]
    while True:
        split = None
        priority = -1.
        for left, right in zip(selected, selected[1:]):
            a, b = original[left], original[right]
            points = original[left:right+1]
            delta = b-a
            ratios = np.clip((points-a) @ delta / max(float(delta @ delta), 1e-20), 0., 1.)
            errors = np.linalg.norm(points-(a+ratios[:, None]*delta), axis=1)
            worst = int(np.argmax(errors))
            safe = (safe_route(np.asarray([a, b])) if safe_route is not None
                    else _path_is_safe(occ, np.asarray([a, b]), .07))
            if errors[worst] <= .025 and safe:
                continue
            if right-left <= 1:
                raise TeacherRejected('rule: original route leg is unsafe')
            index = left+worst if 0 < worst < right-left else (left+right)//2
            score = float(errors[worst]) + (0. if safe else 1.)
            if score > priority:
                split, priority = index, score
        if split is None:
            break
        if len(selected) == count:
            raise TeacherRejected('rule: complete safe route cannot fit four fast calls')
        selected = sorted([*selected, split])
    path = original[selected]
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    pieces = np.ones(len(lengths), dtype=int)
    while pieces.sum() < count-1:
        pieces[int(np.argmax(lengths/pieces))] += 1
    return np.vstack([path[:1], *[
        _resample_polyline(np.asarray([a, b]), count=int(n)+1)[1:]
        for a, b, n in zip(path[:-1], path[1:], pieces)]])


def _budgeted_terrain_targets(pathfinder, route, count, *, radius,
                              executor_spacing, route_resources, resource_limits,
                              diagnostics=None, verified_proposals_only=False):
    """Compress the ordered route with budgets on fine and executor steps.

    Subdividing an accepted leg changes substep locations on a stair. Keep
    the proven vertices instead and fill remaining slots with terminal holds.
    A final continuous replay checks resources using actual arriving heights.
    """
    from native_resource_search import resource_constrained_path
    from native_surface_route import follow_native_polyline

    planar = route[:, [0, 2]]
    neighbors = {}
    for left in range(len(route)-1):
        candidates = []
        for right in range(left+1, len(route)):
            a, b = planar[left], planar[right]
            delta = b-a
            points = planar[left:right+1]
            ratios = np.clip((points-a) @ delta / max(float(delta @ delta), 1e-20), 0., 1.)
            if np.max(np.linalg.norm(points-(a+ratios[:, None]*delta), axis=1)) <= .025:
                candidates.append(right)
        neighbors[left] = candidates

    def edge(left, right):
        fine = follow_native_polyline(pathfinder, route[[left, right]], radius=radius)
        coarse = follow_native_polyline(pathfinder, route[[left, right]],
                                       radius=radius, spacing=executor_spacing)
        if fine is None or coarse is None:
            return None
        return _length(fine), (*route_resources(fine), *route_resources(coarse), 1.)

    limits = tuple(resource_limits)
    def replay(indices):
        targets = route[indices].copy()
        targets = np.vstack([targets, np.repeat(targets[-1:], count-len(targets), axis=0)])
        proofs = []
        for spacing in (.025, executor_spacing):
            actual, trace = targets[0].copy(), [targets[0].copy()]
            for target in targets[1:]:
                leg = follow_native_polyline(pathfinder, [actual, target], radius=radius, spacing=spacing)
                if leg is None:
                    raise TeacherRejected('rule: continuous terrain replay failed physical steps')
                trace.extend(leg[1:])
                actual = leg[-1]
            used = tuple(float(v) for v in route_resources(np.asarray(trace)))
            if (len(used) != len(limits) or not np.isfinite(used).all()
                    or any(v < 0 or v > limit+1e-9 for v, limit in zip(used, limits))):
                raise TeacherRejected('rule: continuous terrain replay exceeds route height budgets')
            proofs.append(dict(spacing_m=spacing, resources=list(used), measured_points=len(trace)))
        return targets, proofs

    prepared = None
    prepared_indices = None
    rejections = []
    def accept(indices):
        nonlocal prepared, prepared_indices
        try:
            prepared = replay(indices)
        except TeacherRejected as error:
            rejections.append(str(error))
            return False
        prepared_indices = list(indices)
        return True

    search = {}
    indices = resource_constrained_path(neighbors, 0, len(route)-1, edge,
        resource_limits=(*limits, *limits, count-1),
        heuristic=lambda i: float(np.linalg.norm(route[-1]-route[i])), diagnostics=search,
        # Every allowed chord uses exactly one target slot. Reverse minima on
        # this optimistic graph prove the remaining slot count even before
        # physical edge checks; zero is a valid bound for length and resources.
        use_completion_bounds=True,
        completion_edge_bound=lambda left, right: (0., (0.,)*(2*len(limits))+(1.,)),
        use_shortest_certificate=True, feasible_path_accept=accept,
        verified_proposals_only=verified_proposals_only)
    if diagnostics is not None:
        diagnostics.update(search, continuous_candidate_rejections=rejections)
    if indices is None:
        if search.get('outcome') == 'verified_proposals_exhausted':
            raise TeacherRejected('rule: planner produced no verified four-window terrain proposal; feasibility unknown')
        raise TeacherRejected('rule: no budget-preserving terrain route fits four fast calls')
    targets, proofs = (prepared if prepared_indices == list(indices) else replay(indices))
    if diagnostics is not None:
        diagnostics['continuous_replays'] = proofs
    return targets


def _terrain_targets(pathfinder, route, count, *, radius, executor_spacing,
                     route_resources=None, resource_limits=None, diagnostics=None,
                     verified_proposals_only=False):
    """Preserve planar bends and derive all target heights from physical steps."""
    from native_surface_route import follow_native_polyline
    if (route_resources is None) != (resource_limits is None):
        raise ValueError('terrain route resource callback and limits must be supplied together')
    if route_resources is not None:
        return _budgeted_terrain_targets(pathfinder, route, count, radius=radius,
            executor_spacing=executor_spacing, route_resources=route_resources,
            resource_limits=resource_limits, diagnostics=diagnostics,
            verified_proposals_only=verified_proposals_only)
    planar = route[:, [0, 2]]

    def original_position(xz):
        matches = route[np.linalg.norm(planar-xz, axis=1) < 1e-6]
        if not len(matches) or np.ptp(matches[:, 1]) > .06:
            raise TeacherRejected('rule: ambiguous floor at a native route vertex')
        return matches[0]

    sampled = _fixed_count_route(planar, None, count, safe_route=lambda leg:
        follow_native_polyline(pathfinder, [original_position(p) for p in leg],
                               radius=radius) is not None)
    guessed = np.asarray([[p[0], route[np.argmin(np.linalg.norm(planar-p, axis=1)), 1], p[1]]
                          for p in sampled])
    guessed[0], guessed[-1] = route[0], route[-1]
    measured = follow_native_polyline(pathfinder, guessed, radius=radius)
    if measured is None:
        raise TeacherRejected('rule: sampled terrain path fails physical steps')
    targets, cursor = [], 0
    for p in sampled:
        hits = np.flatnonzero(np.linalg.norm(measured[cursor:, [0, 2]]-p, axis=1) < 1e-5)
        if not len(hits):
            raise TeacherRejected('rule: native target lacks ordered measured height')
        cursor += int(hits[0])
        targets.append(measured[cursor])
    targets = np.asarray(targets)
    actual = targets[0].copy()
    for target in targets[1:]:
        leg = follow_native_polyline(pathfinder, [actual, target],
                                     radius=radius, spacing=executor_spacing)
        if leg is None:
            raise TeacherRejected('rule: native target fails executor physical substeps')
        actual = leg[-1]
    return targets


def prepare_full_route(state, pose, occ, minimum_chunks=4, *, exact_chunks=None, prefix_steps=4):
    """Sample the complete route, retaining bends and its exact bound endpoint.

    Minimum chunks sets temporal resolution, never an execution budget. More
    chunks are added only when necessary to preserve geometric clearance/bends.
    Neither the old action expert quantiles nor a fixed step length enter here.
    """
    if minimum_chunks < 1:
        raise ValueError('minimum_chunks must be positive')
    if type(prefix_steps) is not int or prefix_steps not in (4, 16):
        raise ValueError('route prefix_steps must be 4 or 16')
    if exact_chunks is not None and exact_chunks != 4:
        raise ValueError('formal slow anchors require exactly four fast calls')
    native_path = state.get('path_native_xyz')
    if native_path is not None:
        from native_surface_route import follow_native_polyline, native_clearance_contract
        native_path = np.asarray(native_path, dtype=float)
        if (native_path.ndim != 2 or native_path.shape[1] != 3
                or not len(native_path) or not np.isfinite(native_path).all()):
            raise TeacherRejected('rule: invalid native surface route')
        if math.dist(occ.world_xy(native_path[0]), pose[:2]) > .035 or abs(native_path[0, 1]-occ.floor_height) > .06:
            raise TeacherRejected('rule: surface route does not start at actual native pose')
        clearance = native_clearance_contract(occ.pathfinder)
        radius = clearance['extra_clearance_m']
        # Prove each specified vertex height by physical motion. Stair height
        # between adjacent vertices need not follow a straight 3-D chord.
        safe_route = lambda points: all(follow_native_polyline(occ.pathfinder, leg,
            radius=radius) is not None for leg in
            ([points] if len(points) == 1 else zip(points[:-1], points[1:])))
        if not safe_route(native_path):
            raise TeacherRejected('rule: native surface route is not physically executable')
        chunks = exact_chunks or minimum_chunks
        from agent_tools import SUBSTEP_M
        constraints = state.get('native_stair_constraints')
        resources = None
        sampling = {}
        if constraints is not None:
            from native_stair_route import named_stair_metrics
            def resources(points):
                proof = named_stair_metrics(points, constraints['objects'], occ.world_xy,
                                            constraints['direction'])
                return proof['outside_height_change_m'], proof['contrary_height_change_m']
        sampled_native = _terrain_targets(occ.pathfinder, native_path, 4*chunks+1,
            radius=radius, executor_spacing=SUBSTEP_M, route_resources=resources,
            resource_limits=constraints['resource_limits'] if constraints is not None else None,
            diagnostics=sampling,
            verified_proposals_only=(constraints is not None and
                                     constraints.get('search_policy') == 'verified_proposals'))
        original = np.asarray([occ.world_xy(p) for p in native_path])
        sampled = np.asarray([occ.world_xy(p) for p in sampled_native])
        labels = np.asarray(state['path'], dtype=float)
        if labels.shape != original.shape or not np.allclose(labels, original, atol=1e-7, rtol=0):
            raise TeacherRejected('rule: XY labels differ from the verified native route')
        state['route_target_native_positions'] = sampled_native.tolist()
        state['native_surface_contract'] = dict(schema='habitat_surface_route_v1',
            world_frame=occ.world_frame, radius_m=radius, clearance=clearance, probe_spacing_m=.025,
            executor_substep_m=SUBSTEP_M,
            height_tolerance_m=.06, planned_native_path=native_path.tolist(),
            planned_native_path_m=_length(native_path), terrain_following=True)
        if constraints is not None:
            state['native_surface_contract'].update(stair_constraints=constraints,
                                                    sampling_proof=sampling)
    else:
        original = np.asarray(state['path'], dtype=float)
    if original.ndim != 2 or original.shape[1] != 2 or not len(original):
        raise ValueError('route must contain world XY points')
    if not np.isfinite(original).all() or math.dist(original[0], pose[:2]) > .035:
        raise ValueError('route does not start at actual pose')
    safe = (True if native_path is not None else
            occ.is_free(*original[0], .07) if len(original) == 1
            else _path_is_safe(occ, original, .07))
    if not safe:
        raise ValueError('original centred route is unsafe')
    length = _length(original)
    chunks = exact_chunks or minimum_chunks
    if native_path is not None:
        pass  # Keep the 3-D-verified subdivision, including every necessary bend.
    elif length > 1e-8 and exact_chunks is not None:
        sampled = _fixed_count_route(original, occ, 4*chunks+1)
    elif length > 1e-8:
        # Adaptive sampling changes temporal resolution, never the endpoint.
        for candidate in range(chunks, max(chunks+1, len(original)*2)):
            sampled = _resample_polyline(original, count=4*candidate+1)
            if _faithful(original, sampled) and _path_is_safe(occ, sampled, .07):
                chunks = candidate
                break
        else:
            # Exact original vertices are a safe geometric fallback. Terminal
            # holds align the final execution prefix to four waypoints.
            sampled = _resample_polyline(original, step_m=length/(4*minimum_chunks), preserve_vertices=True)
            chunks = math.ceil((len(sampled)-1)/4)
            sampled = np.vstack([sampled, np.repeat(sampled[-1:], 4*chunks+1-len(sampled), axis=0)])
    else:
        sampled = np.repeat(original[-1:], 4*chunks+1, axis=0)
    yaw = math.radians(pose[2])
    yaws = [yaw]
    # Reuse the executor's translation precision. A wholly unexecutable
    # numerical residual must not choose an arbitrary travel heading, while
    # the requested XY endpoint and four measured hold/turn windows remain.
    translate = any(segment['kind']=='forward' for segment in decompose(length,0.,0.))
    for a, b in zip(sampled[:-1], sampled[1:]):
        delta = b-a
        desired = math.atan2(delta[1], delta[0]) if translate and np.linalg.norm(delta) > 1e-8 else yaw
        yaw += wrap(desired-yaw)
        yaws.append(yaw)
    if state.get('face_xy') is not None:
        delta = np.asarray(state['face_xy'])-sampled[-1]
        if np.linalg.norm(delta) > 1e-8:
            facing = math.atan2(delta[1], delta[0])
            if not translate:
                yaws = np.linspace(yaws[0], yaws[0]+wrap(facing-yaws[0]), len(sampled)).tolist()
            else:
                yaws[-1] += wrap(facing-yaws[-1])
    targets = np.column_stack([sampled, np.degrees(yaws)]).tolist()
    coarse_targets = targets
    if prefix_steps == 16:
        # Subdivide each old leg independently: preserve every bend, each
        # four-leg refresh endpoint and total physical XY length. Never take
        # sixteen OLD points, which would advance the old route fourfold.
        refined = [targets[0]]
        native_refined = [sampled_native[0]] if native_path is not None else None
        for index, (start, end) in enumerate(zip(targets[:-1], targets[1:])):
            points = np.linspace(start, end, 5)
            if native_path is not None:
                terrain = _terrain_targets(occ.pathfinder, sampled_native[index:index+2], 5,
                    radius=radius, executor_spacing=SUBSTEP_M)
                points[:, :2] = [occ.world_xy(p) for p in terrain]
                native_refined.extend(terrain[1:])
            refined.extend(points[1:].tolist())
        targets = refined
        sampled = np.asarray(targets)[:, :2]
        if native_refined is not None:
            state['route_target_native_positions'] = np.asarray(native_refined).tolist()
            if constraints is not None:
                # Re-prove the refined route: fine simulator steps must not
                # inherit a stair-resource certificate for coarse execution.
                refined_proof = {}
                physical = follow_native_polyline(occ.pathfinder, native_refined,
                    radius=radius, spacing=SUBSTEP_M, diagnostics=refined_proof)
                if physical is None or any(value > limit+1e-8 for value, limit in
                        zip(resources(physical), constraints['resource_limits'])):
                    raise TeacherRejected('rule: refined terrain route fails physical resource proof')
                state['native_surface_contract']['fine_sampling_proof'] = dict(
                    resources=list(resources(physical)), diagnostics=refined_proof)
    state.update(execution_contract=('full_target_route_v2_fine16' if prefix_steps == 16
                                     else 'full_target_route_v1'), full_route_complete=False,
                 planned_path_m=length, sampled_path_m=_length(sampled), execution_path_m=_length(sampled), route_chunks=chunks,
                 nominal_step_m=length/(prefix_steps*chunks), route_progress=0,
                 route_prefix_steps=prefix_steps, coarse_route_target_world_poses=coarse_targets,
                 route_target_world_poses=targets, execution_world_poses=[list(pose)],
                 training_windows=[], actual_poses=[], tracking_errors_m=[], tracking_errors_deg=[])
    return targets[1:]


def _chunk_to_world(chunk, start):
    x, y, yaw = chunk
    a = math.radians(start[2])
    return [start[0]+math.cos(a)*x-math.sin(a)*y,
            start[1]+math.sin(a)*x+math.cos(a)*y, start[2]+math.degrees(yaw)]


class FullRoutePolicy(RoutePolicy):
    def predict(self, option):
        state = option.slow_condition
        if state.get('tracking_failure'):
            raise RuntimeError(state['tracking_failure'])
        pose = _world_pose(self.client.last_info)
        progress = state['route_progress']
        route = state['route_target_world_poses']
        targets = np.asarray([route[min(progress+i, len(route)-1)] for i in range(1, 17)])
        angles = np.unwrap(np.r_[math.radians(pose[2]), np.radians(targets[:, 2])])[1:]
        cumulative = [world_to_chunk_pose(p[:2], a, pose[:2], math.radians(pose[2]))
                      for p, a in zip(targets, angles)]
        trajectory = np.diff(np.vstack([np.zeros(3), cumulative]), axis=0)
        prefix_steps = state.get('route_prefix_steps', 4)
        end = min(progress+prefix_steps, len(route)-1)
        state.update(expected_end=route[end], pending_cursor=end)
        state['training_windows'].append(dict(world_pose=pose, chunk_index=option.chunks,
            predicted_delta_xyyaw=trajectory.tolist(), proposed_prefix=trajectory[:prefix_steps].tolist(),
            predicted_world_poses=targets.tolist(), route_progress_start=progress,
            route_progress_end=end, executed=False))
        return trajectory

    def commit(self, option, executed):
        state = option.slow_condition
        window = state['training_windows'][-1]
        prefix_steps = state.get('route_prefix_steps', 4)
        super().commit(option, executed)
        actual = _world_pose(self.client.last_info)
        logs = executed.gym_feedback.get('waypoint_log') or []
        if state.get('native_surface_contract'):
            try:
                positions = np.asarray([row.get('oracle_native_end_position', [])
                                        for row in logs], float)
                starts = np.asarray([row.get('oracle_native_start_position', [])
                                     for row in logs], float)
                geometry = self.client.last_info.get('oracle_habitat_geometry') or {}
                final = np.asarray(geometry.get('position', []), float)
            except (TypeError, ValueError):
                positions = starts = final = np.asarray([], float)
            expected = np.asarray(state['route_target_native_positions'][
                window['route_progress_start']+1:window['route_progress_end']+1], float)
            previous = np.asarray(state.get('execution_native_positions') or
                                  state['route_target_native_positions'][:1], float)[-1]
            valid = (positions.shape == starts.shape == expected.shape == (prefix_steps, 3)
                     and final.shape == (3,) and np.isfinite(positions).all()
                     and np.isfinite(starts).all() and np.isfinite(final).all())
            if valid:
                heights = np.abs(positions[:, 1]-expected[:, 1])
                continuity = np.vstack([previous, positions[:-1]])
                valid = (np.max(heights) <= .06
                         and np.max(np.linalg.norm(starts-continuity, axis=1)) <= .002
                         and np.linalg.norm(final-positions[-1]) <= .002)
                window.update(measured_native_positions=positions.tolist(),
                              measured_native_start_positions=starts.tolist(),
                              native_height_errors_m=heights.tolist())
                state.setdefault('execution_native_positions', [previous.tolist()]).extend(positions.tolist())
            if not valid:
                state['tracking_failure'] = 'native surface height or measured position evidence mismatch'
        # cur_chunk is the measured pose BEFORE each waypoint. Thus rows 1..3
        # give real endpoints 0..2; the actual final pose supplies endpoint 3.
        measured = [window['world_pose']]
        measured.extend(_chunk_to_world(row['cur_chunk'], window['world_pose']) for row in logs[1:])
        if logs:
            measured.append(actual)
        if state.get('native_surface_contract') and not state.get('tracking_failure'):
            frame = state['native_surface_contract']['world_frame']
            native_xy = (positions[:, [0, 2]] * [1., -1.] if frame == 'sage'
                         else positions[:, [2, 0]] * -1.)
            measured_xy = np.asarray(measured[1:])[:, :2]
            if (frame not in {'sage', 'mp3d'} or native_xy.shape != measured_xy.shape
                    or np.max(np.linalg.norm(native_xy-measured_xy, axis=1)) > .002):
                state['tracking_failure'] = 'native surface XY differs from measured waypoint evidence'
        state['execution_world_poses'].extend(measured[1:])
        window.update(executed=True, executed_prefix=option.internal_log[-1]['executed_prefix'],
                      actual_end_pose=actual, measured_world_poses=measured)
        errors = [abs(math.degrees(wrap(math.radians(p[2]-q[2]))))
                  for p, q in zip(measured[1:], window['predicted_world_poses'][:prefix_steps])]
        state['tracking_errors_deg'].extend(errors)
        if executed.blocked or executed.collision:
            state['tracking_failure'] = 'full route interrupted by collision or obstruction'
        elif len(logs) != prefix_steps or len(measured) != prefix_steps+1:
            state['tracking_failure'] = ('executor did not return four measured waypoint endpoints'
                if prefix_steps == 4 else 'executor did not return sixteen measured waypoint endpoints')
        elif max(errors, default=0.) >= 2.:
            state['tracking_failure'] = 'full route yaw tracking mismatch'
        elif any(math.dist(p[:2], q[:2]) >= .035
                 for p, q in zip(measured[1:], window['predicted_world_poses'][:prefix_steps])):
            state['tracking_failure'] = 'full route waypoint tracking mismatch'
        elif not state.get('tracking_failure'):
            state['route_progress'] = state['pending_cursor']


class FullRouteRuntime:
    """Reuse the shared runtime with bounds derived from this entire route."""
    def __init__(self, policy, executor):
        self.policy, self.executor = policy, executor

    def run(self, option):
        state = option.slow_condition
        chunks = state['route_chunks']
        prefix_steps = state.get('route_prefix_steps', 4)
        unbounded = os.environ.get('RULE_TEACHER_CONTEXT_VERSION') == 'v8'
        config = ActionHeadRuntimeConfig(prefix_steps=prefix_steps, fast_step_limit=chunks,
            # This finite teacher route may contain sub-mm steps or holds before
            # its final facing adjustment. Execute and measure every prefix;
            # commit still checks tracking, collisions and full-route progress.
            min_executable_xy_m=0.,
            max_distance_m=None if unbounded else max(1., state['execution_path_m']*1.05+1.),
            max_sim_steps=None if unbounded else prefix_steps*chunks+1,
            max_seconds=None if unbounded else 600,
            no_progress_limit=chunks+1,
            max_abs_xy_m=None if unbounded else max(1., state['planned_path_m']+.1),
            max_abs_yaw_rad=None if unbounded else math.pi+1e-5)
        result = ActionHeadToolRuntime(self.policy, self.executor, config,
            progress_callback=self.policy.commit).run(option)
        actual = _world_pose(self.policy.client.last_info)
        goal = state['route_target_world_poses'][-1]
        state['goal_error_m'] = math.dist(actual[:2], goal[:2])
        state['execution_path_m'] = _length(np.asarray(state['execution_world_poses'])[:, :2])
        state['goal_yaw_error_deg'] = abs(math.degrees(wrap(math.radians(actual[2]-goal[2]))))
        complete = (result.ok and not result.blocked and not state.get('tracking_failure')
                    and state['route_progress'] == prefix_steps*chunks
                    and state['goal_error_m'] < .035 and state['goal_yaw_error_deg'] < 2.)
        state['full_route_complete'] = bool(complete)
        if complete:
            result.return_reason = 'planned_route_complete'
        else:
            result.ok = False
            result.error = state.get('tracking_failure') or f'incomplete full route: {result.return_reason}'
            if result.return_reason == 'fast_step_budget_reached':
                result.return_reason = 'incomplete_planned_route'
        return result
