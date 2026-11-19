"""Call-grounded stair flights from scene annotations and native walkable mesh.

No episode endpoint or reference route enters this module. Planar FM commands
follow the measured surface; native Y is proof, never a commanded teleport.
"""
import base64
import hashlib
import io
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image

from native_surface_route import centered_native_route, follow_native_polyline, native_route_safe
from privileged_teacher import TeacherRejected, wrap
from rule_plan_chunks import segment_plan


STAIR_LABELS = {'stair', 'stairs', 'staircase', 'stairway', 'steps'}


def stair_search_policy(value=None):
    policy = os.environ.get('RULE_NATIVE_STAIR_SEARCH_POLICY', 'complete') if value is None else value
    if policy not in ('complete', 'verified_proposals'):
        raise ValueError('unknown native stair search policy')
    return policy


def stair_annotation_component(objects, object_id):
    """Join touching source boxes within one annotated native region.

    A shared XY projection or region alone is insufficient. Strict 3-D box
    overlap connects annotated parts; endpoints stay bound to the original
    object, and physical route proofs remain mandatory.
    """
    original = objects[object_id]
    region = original.get('native_region_id')
    if region is None:
        return {object_id: original}
    candidates = {oid: obj for oid, obj in objects.items()
                  if obj.get('native_region_id') == region
                  and obj['label'].strip().lower() in STAIR_LABELS}
    selected = {object_id: original}
    queue = [original]
    while queue:
        current = queue.pop()
        a, az = np.asarray(current['bbox_xyxy']), current['z_range']
        for oid, obj in candidates.items():
            if oid in selected:
                continue
            b, bz = np.asarray(obj['bbox_xyxy']), obj['z_range']
            if (np.all(np.maximum(a[:2], b[:2]) <= np.minimum(a[2:], b[2:]))
                    and max(az[0], bz[0]) <= min(az[1], bz[1])):
                selected[oid] = obj
                queue.append(obj)
    return selected


def named_stair_metrics(route, objects, world_xy, direction):
    route = np.asarray(route, float)
    xy = np.asarray([world_xy(p) for p in route])
    inside = np.zeros(len(route), dtype=bool)
    for obj in objects:
        bounds, heights = np.asarray(obj['bbox_xyxy']), obj['z_range']
        inside |= (np.all((xy >= bounds[:2]-.35) & (xy <= bounds[2:]+.35), axis=1)
                   & (route[:, 1] >= heights[0]-.30) & (route[:, 1] <= heights[1]+.30))
    changes = np.diff(route[:, 1])
    return dict(outside_height_change_m=float(np.abs(changes[~(inside[:-1] & inside[1:])]).sum()),
                contrary_height_change_m=float(np.maximum(-direction*changes, 0.).sum()),
                named_stair_height_gain_m=float(direction*changes[inside[:-1] & inside[1:]].sum()))


def first_stair_motion(contract, graph):
    if graph is None or graph.get('original_call') != contract:
        raise TeacherRejected('native: stair motion requires the original language graph')
    fields = {f['field']: f for f in graph['fields']}
    for i, text in enumerate(contract.get('plan', [])):
        if not text.strip():
            continue
        clauses = segment_plan(fields[f'plan[{i}]'])['clauses']
        if not clauses:
            continue
        # A rotation is a legitimate first local stage. Never silently skip it
        # to compile a later movement or declare its anchor a failed ascent.
        first = clauses[0]
        if first['kind'] == 'rotation' or first['negated']:
            return None
        if first['action'] not in {'go', 'move', 'walk', 'proceed', 'head', 'continue',
                                   'advance', 'navigate', 'follow', 'ascend', 'descend', 'climb'}:
            return None
        value = first['text'].lower()
        noun = r'(?:stair(?:s|case|way)?|steps)'
        if (re.search(r'\b(?:stair(?:s|case|way)?|landing)\b', value)
                and re.search(r'\b(?:second|third|fourth|fifth|\d+)\s+(?:stair|step|landing)', value)):
            raise TeacherRejected('native: counted stair endpoint requires tread identity')
        up = bool(re.search(r'\bup\s+(?:(?:the|a|an|these|those)\s+)?'+noun+r'\b|\bupstairs\b', value)
                  or re.search(r'\bascend\b[^.;]*\b'+noun+r'\b', value)
                  or (re.search(r'\bclimb\b[^.;]*\b'+noun+r'\b', value)
                      and not re.search(r'\bdown\b', value)))
        down = bool(re.search(r'\bdown\s+(?:(?:the|a|an|these|those)\s+)?'+noun+r'\b|\bdownstairs\b', value)
                    or re.search(r'\bdescend\b[^.;]*\b'+noun+r'\b', value))
        if not up and not down:
            # Keep the existing rejection for non-stair floor requests until
            # there is an independently grounded connector for that wording.
            return None
        if up == down or first.get('order_status') == 'unknown_order':
            raise TeacherRejected('native: ambiguous stair direction or clause order')
        return dict(direction=1 if up else -1, clause=first, plan_index=i,
                    scope='first_stair_flight')
    return None


def stair_surface_candidates(pathfinder, obj, world_xy, direction, *, radius=.07):
    """Select the far supported height in the annotated flight's local domain.

    Triangle interiors avoid navmesh edge vertices with zero clearance. The
    35 cm margin permits an adjacent landing, never a different scene target.
    Native height bounds exclude stacked walkable floors sharing the same XY.
    """
    vertices = np.asarray(pathfinder.build_navmesh_vertices(), float)
    indices = np.asarray(pathfinder.build_navmesh_vertex_indices(), int).reshape(-1, 3)
    triangles = vertices[indices]
    centers = triangles.mean(axis=1)
    xy = np.asarray([world_xy(p) for p in centers])
    bounds = np.asarray(obj['bbox_xyxy'], float)
    heights = np.asarray(obj['z_range'], float)
    mask = (np.all((xy >= bounds[:2]-.35) & (xy <= bounds[2:]+.35), axis=1)
            & (centers[:, 1] >= heights[0]-.30) & (centers[:, 1] <= heights[1]+.30))
    supported = []
    for point in centers[mask]:
        if (pathfinder.is_navigable(point.astype(np.float32), .06)
                and pathfinder.distance_to_closest_obstacle(point.astype(np.float32), 2.) > radius):
            supported.append(point)
    if not supported:
        return []
    supported = np.asarray(supported)
    extreme = (supported[:, 1].max() if direction > 0 else supported[:, 1].min())
    return supported[np.abs(supported[:, 1]-extreme) <= .06].tolist()


def compile_stair_surface(occ, obj, actual_start, motion, *, diagnostics=None,
                          annotation_component=None, feasible_route_accept=None,
                          verified_proposals_only=False):
    if verified_proposals_only and feasible_route_accept is None:
        raise ValueError('proposal-only stairs require the complete four-window verifier')
    import habitat_sim
    from native_surface_route import native_clearance_contract

    start = np.asarray(actual_start, float)
    radius = native_clearance_contract(occ.pathfinder)['extra_clearance_m']
    direction = motion['direction']
    ordered_endpoints = []
    reasons = []
    # All endpoints below share this unchanged pathfinder, annotation component
    # and resource proof. Release the cache when this flight compilation ends.
    edge_cost_cache = {}
    component = list(annotation_component or [obj])
    boxes = np.asarray([part['bbox_xyxy'] for part in component], float)
    # Production SAGE/MP3D world_xy is a signed permutation of native X/Z.
    # The physical verifier's .015 m XY error thus has the same world bound.
    # Unknown maps keep the universal zero bound rather than assume isometry.
    tracked_boxes_min = boxes[:, :2] - (.35 + .015)
    tracked_boxes_max = boxes[:, 2:] + (.35 + .015)
    world_tracking_proved = getattr(occ, 'world_frame', None) in ('sage', 'mp3d')
    bounds = np.r_[boxes[:, :2].min(axis=0), boxes[:, 2:].max(axis=0)]
    def metrics(points):
        return named_stair_metrics(points, component, occ.world_xy, direction)
    def resources(points):
        proof = metrics(points)
        return proof['outside_height_change_m'], proof['contrary_height_change_m']
    def completion_resources(points):
        # The edge verifier starts at points[0] and independently rejects an
        # endpoint mismatch above .025 m. Contrary total variation therefore
        # cannot be below this directed endpoint difference minus .025 m.
        outside = 0.
        if world_tracking_proved:
            xy = np.asarray([occ.world_xy(point) for point in (points[0], points[-1])])
            # Every measured position remains inside the commanded straight
            # segment's XY box plus tracking tolerance. A disjoint box proves
            # every physical step outside every named annotation, including
            # the existing .35 m annotation margin. Intersecting commands keep
            # the zero bound even when both endpoints individually lie outside.
            disjoint = np.any((xy.max(axis=0) < tracked_boxes_min)
                              | (xy.min(axis=0) > tracked_boxes_max), axis=1)
            if np.all(disjoint):
                outside = max(0., abs(points[-1, 1]-points[0, 1])-.025)
        return outside, max(0., -direction*(points[-1, 1]-points[0, 1])-.025)
    def admissible(points):
        proof = metrics(points)
        return (proof['outside_height_change_m'] <= .06
                and proof['contrary_height_change_m'] <= .06
                and proof['named_stair_height_gain_m'] > .12)
    def accept_feasible_route(points):
        return (admissible(points) and native_route_safe(occ.pathfinder, points, radius=radius)
                and feasible_route_accept(points))
    for endpoint in stair_surface_candidates(occ.pathfinder, obj, occ.world_xy, direction, radius=radius):
        endpoint = np.asarray(endpoint, float)
        # This is a semantic height proof beyond the mesh tolerance, not a
        # distance filter on valid turn, hold or tiny-movement FM anchors.
        if direction*(endpoint[1]-start[1]) <= .12:
            continue
        query = habitat_sim.ShortestPath()
        query.requested_start = start.astype(np.float32)
        query.requested_end = endpoint.astype(np.float32)
        if not occ.pathfinder.find_path(query):
            reasons.append('disconnected_native_surface')
            continue
        route = np.asarray(query.points, float)
        if len(route) < 2 or not np.isfinite(route).all():
            continue
        if np.linalg.norm(route[0]-start) > .025 or np.linalg.norm(route[-1]-endpoint) > .025:
            continue
        route[0] = start
        if float(query.geodesic_distance) > 25:
            reasons.append('flight_route_exceeds_local_scope')
            continue
        ordered_endpoints.append((float(query.geodesic_distance), endpoint, route))
    # The original implementation compiled every endpoint, then selected the
    # valid route with the smallest native geodesic distance (stable ties).
    # Visit precisely that order and stop at its first fully validated route.
    # This keeps the selection criterion without compiling losing endpoints.
    for _, endpoint, route in sorted(ordered_endpoints, key=lambda pair: pair[0]):
        funnel_evidence = {}
        route = follow_native_polyline(occ.pathfinder, route, radius=radius, diagnostics=funnel_evidence)
        if route is None or not admissible(route):
            # Keep the endpoint grounded in the same named stair. Search only
            # its local native domain and retain all subsequent flight guards.
            lower = np.minimum(start, [-bounds[3], obj['z_range'][0]-.3, -bounds[2]])-[1., .3, 1.]
            upper = np.maximum(start, [-bounds[1], obj['z_range'][1]+.3, -bounds[0]])+[1., .3, 1.]
            centered_evidence = {}
            route = centered_native_route(occ.pathfinder, start, endpoint,
                bounds=[lower, upper], radius=radius, diagnostics=centered_evidence,
                edge_resources=resources, resource_limits=(.06, .06), edge_cost_cache=edge_cost_cache,
                # Optimistic reverse minima prune only proven infeasibility;
                # avoid physically compiling every graph edge up front.
                use_completion_bounds=True, completion_resource_bound=completion_resources,
                feasible_route_accept=(None if feasible_route_accept is None else accept_feasible_route),
                verified_proposals_only=verified_proposals_only)
            if diagnostics is not None:
                diagnostics.append(dict(endpoint=endpoint.tolist(), funnel=funnel_evidence,
                                        centered=centered_evidence))
            if route is None:
                if centered_evidence.get('outcome') == 'verified_proposals_exhausted':
                    # The same nearest native endpoint is bound as before.
                    # Do not silently pick an easier endpoint or claim no
                    # physical route exists after an incomplete search.
                    raise TeacherRejected('native: planner produced no verified named stair proposal '
                                          'for the selected endpoint; feasibility unknown')
                reasons.append('native_physical_step_failed')
                continue
        # Keep the full measured path. The four-window sampler proves its
        # shortcuts at both spacings under cumulative budgets; greedy string
        # pulling can consume the same height budget several times.
        if not admissible(route):
            reasons.append('height_change_outside_named_flight')
            continue
        if not native_route_safe(occ.pathfinder, route, radius=radius):
            reasons.append('native_physical_step_failed')
            continue
        return route
    raise TeacherRejected('native: no physically executable named stair flight; '+','.join(sorted(set(reasons))))


def prepare_stair_state(occ, object_id, obj, actual_start, motion, component, pose,
                        minimum_chunks, exact_chunks, *, diagnostics=None, prefix_steps=4):
    """Original pure geometry boundary; no renderer, parser or simulator action."""
    from rule_route_execution import prepare_full_route
    from native_surface_route import native_clearance_contract, simplify_native_route
    radius = native_clearance_contract(occ.pathfinder)['extra_clearance_m']
    search_policy = stair_search_policy(motion.get('search_policy'))
    def observe(**details):
        event = dict(event='native_stair_route_contract', search_policy=search_policy, **details)
        if diagnostics is not None:
            diagnostics.append(event)
        if os.environ.get('RULE_NATIVE_SEARCH_PROGRESS') == '1':
            print(json.dumps(event), flush=True)
    def prepare_original(points):
        trial = dict(path_native_xyz=points.tolist(), path=[occ.world_xy(p) for p in points],
            native_stair_constraints=dict(direction=motion['direction'],
                annotation_component_ids=list(component),
                objects=[{key: part[key] for key in ('bbox_xyxy', 'z_range')}
                         for part in component.values()], resource_limits=[.06, .06],
                search_policy=search_policy),
            program=dict(status='ok', steps=[dict(kind='native_stair_flight', object=object_id,
                                                  direction=motion['direction'])]))
        prepare_full_route(trial, pose, occ, minimum_chunks, exact_chunks=exact_chunks,
                           prefix_steps=prefix_steps)
        return trial
    def budgets(points):
        proof = named_stair_metrics(points, component.values(), occ.world_xy, motion['direction'])
        return (proof['outside_height_change_m'] <= .06
                and proof['contrary_height_change_m'] <= .06)
    def prepare(points):
        try:
            return prepare_original(points)
        except TeacherRejected as original_error:
            if str(original_error) not in (
                    'rule: no budget-preserving terrain route fits four fast calls',
                    'rule: planner produced no verified four-window terrain proposal; feasibility unknown'):
                raise
            started = time.monotonic()
            # The existing helper checks each physical shortcut AND the full
            # continuously measured route against the original budgets. Never
            # sum independent per-leg allowances as a whole-route proof.
            simplified = simplify_native_route(occ.pathfinder, points, radius=radius,
                                               route_accept=budgets)
            if simplified is None or np.array_equal(simplified, points):
                observe(outcome='simplification_unavailable', reason=str(original_error),
                        seconds=time.monotonic()-started, original_points=len(points))
                raise
            proof = named_stair_metrics(simplified, component.values(), occ.world_xy,
                                        motion['direction'])
            if (proof['named_stair_height_gain_m'] <= .12
                    or np.linalg.norm(simplified[0]-points[0]) > .025
                    or np.linalg.norm(simplified[-1]-points[-1]) > .025):
                observe(outcome='simplification_contract_rejected', reason=str(original_error),
                        seconds=time.monotonic()-started, resources=proof)
                raise
            try:
                trial = prepare_original(simplified)
            except TeacherRejected as error:
                observe(outcome='simplification_four_windows_rejected', reason=str(error),
                        seconds=time.monotonic()-started, resources=proof)
                raise
            trial['native_surface_contract']['route_simplification'] = dict(
                original_path_native_xyz=points.tolist(), original_rejection=str(original_error),
                original_path_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
                verified_resources=proof, endpoints_preserved=True)
            observe(outcome='simplification_four_windows_verified', seconds=time.monotonic()-started,
                    original_points=len(points), simplified_points=len(simplified), resources=proof)
            return trial
    prepared = None
    prepared_points = None
    def accept(points):
        nonlocal prepared, prepared_points
        try:
            trial = prepare(points)
        except TeacherRejected as error:
            observe(outcome='route_rejected', reason=str(error), original_points=len(points))
            return False
        prepared, prepared_points = trial, points.copy()
        return True
    route = compile_stair_surface(occ, obj, actual_start, motion,
        diagnostics=diagnostics, annotation_component=component.values(),
        feasible_route_accept=accept, verified_proposals_only=search_policy == 'verified_proposals')
    return (prepared if prepared_points is not None and np.array_equal(route, prepared_points)
            else prepare(route))


def prepare_stair_route(teacher, *, scene, episode, pose, rgb, contract, graph, output, **_):
    motion = first_stair_motion(contract, graph)
    if motion is None:
        return None
    from mp3d_teacher_semantics import load_mp3d_scene
    from rule_match_policy import direction_spec, direction_agrees
    from rule_route_execution import prepare_full_route
    from sage.route_extractor import _bbox_distance_xy
    from sage.visibility_audit import project_bbox, depth_support
    from vln_gym.env import _jpg_b64

    occ = teacher.scene(scene)[0]
    actual = teacher.actual_geometry()
    source = load_mp3d_scene(scene.removeprefix('rxr-'), root=teacher.house_root)
    camera = teacher.camera_position(pose)
    yaw = math.radians(pose[2])
    rendered, depth = teacher.renderer.render_one(scene, camera, yaw, want_depth=True)
    comparable = np.asarray(Image.open(io.BytesIO(base64.b64decode(_jpg_b64(rendered)))).convert('RGB'), float)
    mae = float(np.abs(comparable-np.asarray(rgb, float)).mean())
    if mae > 3:
        raise RuntimeError(f'current-state stair render mismatch {mae}')
    direction = direction_spec(contract)
    candidates, audit = [], []
    for oid, obj in source.objects.items():
        if obj['label'].strip().lower() not in {'stair', 'stairs', 'staircase', 'stairway', 'steps'}:
            continue
        lower, upper = obj['z_range']
        if not lower-.30 <= actual['position'][1] <= upper+.30:
            continue
        delta = np.asarray(obj['centre_xy'])-pose[:2]
        bearing = math.degrees(wrap(math.atan2(delta[1], delta[0])-yaw))
        if not direction_agrees(bearing, direction):
            continue
        projection = project_bbox(teacher.camera, camera, yaw, obj)
        support = depth_support(depth, projection['bbox_px'], teacher.camera, camera, yaw, obj,
                                stride=8) if projection else dict(visible_rays=0, visible_fraction=0.)
        distance = _bbox_distance_xy(pose[:2], obj['bbox_xyxy'])
        if support['visible_rays'] < 1 and distance > .40:
            continue
        candidates.append(dict(id=oid, label=obj['label'], native_object_id=obj.get('native_object_id'),
                               bearing_deg=bearing, surface_distance_m=distance,
                               projection=projection, depth=support, target_kind='stair'))
    candidates.sort(key=lambda c: (-int(c['depth']['visible_rays'] > 0), c['surface_distance_m'], c['id']))
    state = bound = None
    for candidate in candidates:
        compiler_audit = []
        try:
            component = stair_annotation_component(source.objects, candidate['id'])
            args = (occ, candidate['id'], source.objects[candidate['id']],
                    actual['position'], motion, component, pose, teacher.route_min_chunks,
                    getattr(teacher, 'route_exact_chunks', None))
            prepare_state = getattr(teacher, 'prepare_stair_state', prepare_stair_state)
            trial = prepare_state(*args, diagnostics=compiler_audit,
                                  prefix_steps=getattr(teacher, 'route_prefix_steps', 4))
            state, bound = trial, candidate
            audit.append(dict(id=candidate['id'], stage='selected', compiler_audit=compiler_audit,
                              annotation_component=list(component)))
            break
        except TeacherRejected as exc:
            audit.append(dict(id=candidate['id'], stage='rejected', reason=str(exc),
                              compiler_audit=compiler_audit))
    evidence = dict(schema='rule-teacher-v1', scene=scene, episode=episode, original_call=contract,
                    planning_call=contract, pose=pose, match_policy=teacher.match_policy,
                    admission_mode=teacher.admission_mode, grounding_mode=teacher.grounding_mode,
                    parsed=dict(family='stairs', kind='native_stair_flight'),
                    native_stair_motion=motion, direction=direction, render_rgb_mae=mae,
                    candidates=candidates, candidate_audit=audit, relaxations=[],
                    match_branch='target', target_identity_verified=False)
    if state is None:
        output.joinpath('binding.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
        raise TeacherRejected('native: no call-grounded executable staircase')
    geometry_sources = {key: dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
                        for key, path in [('house', source.source_path), ('navmesh', occ.navmesh_path)]}
    state['native_surface_contract'].update(geometry_sources=geometry_sources,
                                           grounded_stair_object=bound['id'], direction=motion['direction'])
    evidence.update(reachable_selection=[bound['id']], geometry_sources=geometry_sources)
    preview = state['route_target_world_poses'][1:]
    state.update(admission_mode=teacher.admission_mode, match_policy=teacher.match_policy,
                 match_branch='target', binding=evidence, bound=bound, initial_pose=pose,
                 relaxation_warnings=[], preview_world_xy_yaw_deg=preview,
                 semantic_status='rule_supported_pending_human_review')
    output.joinpath('binding.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
    output.joinpath('accepted.json').write_text(json.dumps(state, ensure_ascii=False, indent=2))
    teacher.map_image(occ, pose, teacher.scene(scene)[1], path=[pose[:2]]+[p[:2] for p in preview]).save(output/'teacher_map.png')
    return state
