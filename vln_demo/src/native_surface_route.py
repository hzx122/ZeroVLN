"""Verify ordered Habitat paths by replaying terrain-following physical steps.

Inputs are scene navigation geometry and a call-grounded route, never episode
goals or reference trajectories. A connected shortest path is insufficient:
each horizontal command must also reach the specified surface without sliding.
"""
import math
from collections import defaultdict, OrderedDict

import numpy as np


class NativeProofCache:
    """Bounded exact query reuse for one immutable, owned PathFinder.

    Native queries consume float32 positions. Keys preserve those exact bytes
    and all support/clearance parameters. The final path remains uncached.
    """
    def __init__(self, pathfinder, maximum=65536):
        if not isinstance(maximum, int) or maximum < 1:
            raise ValueError('positive integer native proof cache capacity required')
        self.pathfinder, self.maximum = pathfinder, maximum
        self.points, self.steps = OrderedDict(), OrderedDict()
        self.point_hits = self.step_hits = 0
        self.point_misses = self.step_misses = 0

    def check_owner(self, pathfinder):
        if pathfinder is not self.pathfinder:
            raise ValueError('native geometry proof cache belongs to another PathFinder')

    def remember(self, table, key, value):
        table[key] = value
        if len(table) > self.maximum:
            table.popitem(last=False)

    def step(self, pathfinder, start, requested):
        self.check_owner(pathfinder)
        start, requested = np.asarray(start, np.float32), np.asarray(requested, np.float32)
        key = start.tobytes(), requested.tobytes()
        if key in self.steps:
            self.step_hits += 1
            return self.steps[key].copy()
        actual = np.asarray(pathfinder.try_step_no_sliding(start, requested), float)
        self.step_misses += 1
        self.remember(self.steps, key, actual.copy())
        return actual

    def metrics(self):
        return dict(point_hits=self.point_hits, step_hits=self.step_hits,
                    point_misses=self.point_misses, step_misses=self.step_misses,
                    point_entries=len(self.points), step_entries=len(self.steps), maximum=self.maximum)


def _native_point_safe(pathfinder, point, *, radius, height_tolerance=.06,
                       diagnostics=None, proof_cache=None):
    native = np.asarray(point, np.float32)
    key = native.tobytes(), float(radius), float(height_tolerance)
    if proof_cache is not None:
        proof_cache.check_owner(pathfinder)
    if proof_cache is not None and key in proof_cache.points:
        proof_cache.point_hits += 1
        reason, clearance = proof_cache.points[key]
    else:
        if proof_cache is not None:
            proof_cache.point_misses += 1
        clearance = None
        if not pathfinder.is_navigable(native, height_tolerance):
            reason = 'unsupported_surface'
        else:
            clearance = float(pathfinder.distance_to_closest_obstacle(native, 2.))
            reason = 'insufficient_clearance' if not math.isfinite(clearance) or clearance <= radius else None
        if proof_cache is not None:
            proof_cache.remember(proof_cache.points, key, (reason, clearance))
    if reason is not None and diagnostics is not None:
        diagnostics.update(reason=reason, point=np.asarray(point).tolist())
        if reason == 'insufficient_clearance':
            diagnostics.update(clearance_m=clearance, required_radius_m=radius)
    return reason is None


def native_clearance_contract(pathfinder):
    """Account for body clearance already baked into the loaded navmesh.

    Unknown or smaller meshes retain the existing additional 7 cm check.
    A verified 10 cm erosion permits positive residual clearance; zero-distance
    boundary points remain forbidden by native_route_safe.
    """
    settings = getattr(pathfinder, 'nav_mesh_settings', None)
    if settings is None:
        settings = getattr(pathfinder, 'navmesh_settings', None)
    radius = getattr(settings, 'agent_radius', None)
    try:
        radius = float(radius)
    except (TypeError, ValueError):
        radius = None
    verified = radius is not None and math.isfinite(radius) and radius >= .1-1e-8
    return dict(mesh_agent_radius_m=radius if radius is not None and math.isfinite(radius) else None,
                minimum_body_radius_m=.1, mesh_body_clearance_verified=verified,
                extra_clearance_m=0. if verified else .07)


def simplify_native_route(pathfinder, points, *, radius=.07, route_accept=None):
    """Reuse string pulling, with a physical proof for every retained shortcut."""
    from visual_navigation.planning import _string_pull
    points = np.asarray(points, float)
    def accepted(a, b):
        measured = follow_native_polyline(pathfinder, [a, b], radius=radius)
        return measured is not None and (route_accept is None or route_accept(measured))
    corners = _string_pull(points, accepted)
    measured = follow_native_polyline(pathfinder, corners, radius=radius)
    return measured if measured is not None and (route_accept is None or route_accept(measured)) else None


def centered_native_route(pathfinder, start, end, *, bounds, radius=.07,
                          max_length=25., diagnostics=None, edge_accept=None,
                          edge_resources=None, resource_limits=None, edge_cost_cache=None,
                          use_completion_bounds=False, use_physical_proof_cache=True,
                          completion_resource_bound=None, use_shortest_certificate=True,
                          use_lazy_resource_minima=True, feasible_route_accept=None,
                          verified_proposals_only=False):
    """Search shared 3-D triangle portals when a funnel touches an obstacle.

    The existing grid A* is a single-floor projection. Its grid cannot retain
    stacked floors, so this fallback searches the native triangles instead.
    All directed edges use the existing physical surface verifier; no surface
    snapping, vertical interpolation or relaxed clearance is introduced.
    The optional cache is owned by one read-only flight compilation. Exact
    coordinate bytes identify directed edges across endpoint graphs; different
    physical/semantic proof owners reset it. Never reuse it after mutating a
    pathfinder or the supplied proof's captured annotation data.
    """
    start, end, bounds = np.asarray(start, float), np.asarray(end, float), np.asarray(bounds, float)
    if (start.shape != (3,) or end.shape != (3,) or bounds.shape != (2, 3)
            or not np.isfinite(np.concatenate([start, end, bounds.ravel()])).all()
            or np.any(bounds[0] > bounds[1])):
        raise ValueError('finite native endpoints and 3-D search bounds required')
    if (edge_resources is None) != (resource_limits is None):
        raise ValueError('measured resource costs and limits must be supplied together')
    if edge_cost_cache is not None and edge_resources is not None:
        context = edge_cost_cache.get('context')
        if (context is None or context[0] is not pathfinder or context[1] != radius
                or context[2] is not edge_resources or context[3] is not edge_accept):
            edge_cost_cache.clear()
            edge_cost_cache.update(context=(pathfinder, radius, edge_resources, edge_accept), costs={})
        if use_physical_proof_cache and 'physical_proofs' not in edge_cost_cache:
            edge_cost_cache['physical_proofs'] = NativeProofCache(pathfinder)
        proof_cache = edge_cost_cache.get('physical_proofs') if use_physical_proof_cache else None
    else:
        proof_cache = NativeProofCache(pathfinder) if use_physical_proof_cache else None
    vertices = np.asarray(pathfinder.build_navmesh_vertices(), float)
    indices = np.asarray(pathfinder.build_navmesh_vertex_indices(), int).reshape(-1, 3)
    triangles = vertices[indices]
    centers = triangles.mean(axis=1)
    selected = np.all((centers >= bounds[0]) & (centers <= bounds[1]), axis=1)
    triangles, centers = triangles[selected], centers[selected]
    # Portal matching uses all three native coordinates. Equal XY on separate
    # floors never establishes adjacency. Exported duplicate vertices can have
    # float roundoff; every resulting connection is still physically proved.
    nodes = [start, end]
    neighbors = defaultdict(set)
    portals = defaultdict(list)
    for triangle, center in zip(triangles, centers):
        if not native_route_safe(pathfinder, [center], radius=radius, proof_cache=proof_cache):
            continue
        node = len(nodes)
        nodes.append(center)
        for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
            key = tuple(sorted((tuple(np.round(a, 5)), tuple(np.round(b, 5)))))
            portals[key].append((node, (a+b)/2))
    for entries in portals.values():
        if len(entries) < 2:
            continue
        point = entries[0][1]
        if not native_route_safe(pathfinder, [point], radius=radius, proof_cache=proof_cache):
            continue
        node = len(nodes)
        nodes.append(point)
        for center, _ in entries:
            neighbors[node].add(center)
            neighbors[center].add(node)
    # Exported polygons can have T junctions rather than matching full edges.
    # Sample their interiors too, and connect only close native 3-D neighbors.
    # Physical edge proofs below are authoritative even across such junctions.
    samples = {}
    for triangle in triangles:
        edge_lengths = np.linalg.norm((triangle-np.roll(triangle, -1, axis=0))[:, [0, 2]], axis=1)
        count = max(1, math.ceil(float(edge_lengths.max())/.12))
        for i in range(count+1):
            for j in range(count+1-i):
                point = triangle[0]+(i/count)*(triangle[1]-triangle[0])+(j/count)*(triangle[2]-triangle[0])
                if np.all((point >= bounds[0]) & (point <= bounds[1])):
                    samples.setdefault(tuple(np.round(point, 5)), point)
    for point in samples.values():
        if native_route_safe(pathfinder, [point], radius=radius, proof_cache=proof_cache):
            nodes.append(point)
    nodes = np.asarray(nodes)
    from scipy.spatial import cKDTree
    # A nearest-k cap can fill with coplanar samples and omit the next tread,
    # even though that tread is within the same local connection radius.
    # Retain every local 3-D neighbor; physical directed-edge proofs still
    # reject walls and stacked floors.
    nearest = cKDTree(nodes).query_ball_point(nodes, r=.35)
    for node, next_nodes in enumerate(nearest):
        for next_node in next_nodes:
            if np.linalg.norm(nodes[next_node]-nodes[node]) > 1e-8:
                neighbors[node].add(int(next_node))
    for endpoint in (0, 1):
        nearby = np.argsort(np.linalg.norm(nodes[2:]-nodes[endpoint], axis=1))[:24]+2
        for node in nearby:
            if np.linalg.norm(nodes[node]-nodes[endpoint]) <= 2.:
                neighbors[endpoint].add(int(node))
                neighbors[int(node)].add(endpoint)
    # A physically valid direct path remains the best option where available.
    neighbors[0].add(1)
    if edge_resources is not None:
        from native_resource_search import resource_constrained_path
        coordinate_keys = [point.tobytes() for point in nodes]
        cached_costs = None
        if edge_cost_cache is not None:
            cached_costs = edge_cost_cache['costs']
        def physical_cost(a, b):
            key = (coordinate_keys[a], coordinate_keys[b])
            if cached_costs is not None and key in cached_costs:
                return cached_costs[key]
            measured = follow_native_polyline(pathfinder, nodes[[a, b]], radius=radius, proof_cache=proof_cache)
            if measured is None or (edge_accept is not None and not edge_accept(measured)):
                result = None
            else:
                result = float(np.linalg.norm(nodes[b]-nodes[a])), edge_resources(measured)
            if cached_costs is not None:
                cached_costs[key] = result
            return result
        completion_costs = {}
        completion_endpoint_proofs = 0
        def completion_cost(a, b):
            nonlocal completion_endpoint_proofs
            key = (a, b)
            if key not in completion_costs:
                coordinate_key = (coordinate_keys[a], coordinate_keys[b])
                if b == 1:
                    # Every completed route must enter its actual endpoint.
                    # Prove only these incoming edges before reverse minima,
                    # rather than eagerly compiling the whole physical graph.
                    completion_endpoint_proofs += 1
                    completion_costs[key] = physical_cost(a, b)
                elif cached_costs is not None and coordinate_key in cached_costs:
                    completion_costs[key] = cached_costs[coordinate_key]
                else:
                    completion_costs[key] = (
                        float(np.linalg.norm(nodes[b]-nodes[a])),
                        completion_resource_bound(nodes[[a, b]]))
            return completion_costs[key]
        evidence = {}
        def accept_feasible_path(path):
            # Always replay the complete proposed path without the edge cache.
            # The caller additionally proves its complete execution contract.
            measured = follow_native_polyline(pathfinder, nodes[path], radius=radius)
            return measured is not None and feasible_route_accept(measured)
        path = resource_constrained_path(neighbors, 0, 1, physical_cost,
            resource_limits=resource_limits, max_length=max_length,
            use_completion_bounds=use_completion_bounds,
            heuristic=lambda node: float(np.linalg.norm(nodes[node]-end)), diagnostics=evidence,
            edge_length_bound=lambda a, b: float(np.linalg.norm(nodes[b]-nodes[a])),
            completion_edge_bound=(None if completion_resource_bound is None else completion_cost),
            use_shortest_certificate=use_shortest_certificate,
            use_lazy_resource_minima=use_lazy_resource_minima,
            feasible_path_accept=(None if feasible_route_accept is None else accept_feasible_path),
            verified_proposals_only=verified_proposals_only)
        measured = None if path is None else follow_native_polyline(pathfinder, nodes[path], radius=radius)
        if diagnostics is not None:
            diagnostics.update(evidence, nodes=len(nodes),
                completion_endpoint_proofs=completion_endpoint_proofs,
                physical_proof_cache=proof_cache.metrics() if proof_cache is not None else None,
                outcome=('verified' if measured is not None else
                         'whole_path_failed' if path is not None else evidence['outcome']))
        return measured
    from native_resource_search import shortest_physical_path
    edge_cache = {}
    examined = 0
    def physical_length(a, b):
        nonlocal examined
        key = (a, b)
        if key not in edge_cache:
            examined += 1
            edge_cache[key] = follow_native_polyline(pathfinder, nodes[[a, b]],
                radius=radius, proof_cache=proof_cache)
            if (edge_cache[key] is not None and edge_accept is not None
                    and not edge_accept(edge_cache[key])):
                edge_cache[key] = None
        return (None if edge_cache[key] is None else
                float(np.linalg.norm(nodes[b]-nodes[a])))
    route, costs = shortest_physical_path(neighbors, 0, 1, physical_length,
        max_length=max_length, heuristic=lambda node: float(np.linalg.norm(nodes[node]-end)),
        edge_length_bound=lambda a, b: float(np.linalg.norm(nodes[b]-nodes[a])))
    if route is not None:
        measured = follow_native_polyline(pathfinder, nodes[route], radius=radius)
        if diagnostics is not None:
            diagnostics.update(nodes=len(nodes), examined_edges=examined,
                               outcome='verified' if measured is not None else 'whole_path_failed')
        return measured
    if diagnostics is not None:
        reached = nodes[list(costs)]
        frontier = sorted((float(np.linalg.norm(nodes[b]-end)), a, b)
                          for (a, b), value in edge_cache.items()
                          if value is None and a in costs and b not in costs)[:12]
        failures = []
        for _, a, b in frontier:
            proof = {}
            follow_native_polyline(pathfinder, nodes[[a, b]], radius=radius, diagnostics=proof)
            snapped = np.asarray(pathfinder.snap_point(nodes[b].astype(np.float32)), float) if hasattr(pathfinder, 'snap_point') else None
            failures.append(dict(start=nodes[a].tolist(), target=nodes[b].tolist(),
                                 target_snapped=None if snapped is None else snapped.tolist(), failure=proof))
        diagnostics.update(nodes=len(nodes), examined_edges=examined, outcome='no_verified_route',
                           reached_nodes=len(costs),
                           reached_native_bounds=[reached.min(axis=0).tolist(), reached.max(axis=0).tolist()],
                           closest_reached_endpoint_m=float(np.linalg.norm(reached-end, axis=1).min()),
                           closest_endpoint_nodes=[dict(position=nodes[n].tolist(), reached=int(n) in costs)
                               for n in np.argsort(np.linalg.norm(nodes[2:]-end, axis=1))[:8]+2],
                           closest_frontier_failures=failures)
    return None


def follow_native_polyline(pathfinder, points, *, spacing=.025, radius=.07, diagnostics=None, proof_cache=None):
    """Measure the walkable surface along a native funnel's horizontal path.

    Funnel corners describe connectivity; interpolating their Y can cut through
    stair landings. Each command preserves current Y and obtains the next Y
    from the same no-sliding physical step used by the executor.
    """
    def rejected(reason, **evidence):
        if diagnostics is not None:
            diagnostics.update(reason=reason, **evidence)
        return None

    points = np.asarray(points, float)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or not np.isfinite(points).all()):
        return rejected('invalid_native_polyline')
    actual = points[0].copy()
    measured = [actual.copy()]
    for end in points[1:]:
        start = actual.copy()
        distance = float(np.linalg.norm((end-start)[[0, 2]]))
        if distance < 1e-8:
            if abs(end[1]-actual[1]) > .06:
                return rejected('vertical_teleport', actual=actual.tolist(), requested=end.tolist())
            continue
        for xy in np.linspace(start[[0, 2]], end[[0, 2]], max(1, math.ceil(distance/spacing))+1)[1:]:
            requested = np.asarray([xy[0], actual[1], xy[1]], np.float32)
            before = actual.copy()
            actual = (np.asarray(pathfinder.try_step_no_sliding(before.astype(np.float32), requested), float)
                      if proof_cache is None else proof_cache.step(pathfinder, before.astype(np.float32), requested))
            tolerance = min(.015, .05*float(np.linalg.norm(xy-before[[0, 2]]))+1e-5)
            if (actual.shape != (3,) or not np.isfinite(actual).all()
                    or np.linalg.norm(actual[[0, 2]]-xy) > tolerance):
                return rejected('horizontal_tracking', before=before.tolist(),
                                requested=requested.tolist(), actual=actual.tolist(),
                                tolerance_m=tolerance)
            measured.append(actual.copy())
    measured = np.asarray(measured)
    if np.linalg.norm(measured[-1]-points[-1]) > .025:
        return rejected('endpoint_surface_mismatch', actual=measured[-1].tolist(),
                        requested=points[-1].tolist())
    safety = {}
    # Heights between measured physical steps are terrain, not linear Y
    # labels. Replaying a coarse step's interpolated Y rejects valid treads.
    # The no-sliding step above proves motion; independently check support
    # and clearance at every measured position, including the original start.
    if not all(_native_point_safe(pathfinder, point, radius=radius, diagnostics=safety, proof_cache=proof_cache)
               for point in measured):
        return rejected('measured_surface_or_clearance_failed', safety=safety)
    return measured


def native_route_safe(pathfinder, points, *, radius=.07, spacing=.025,
                      height_tolerance=.06, diagnostics=None, proof_cache=None):
    def rejected(reason, **evidence):
        if diagnostics is not None:
            diagnostics.update(reason=reason, **evidence)
        return False

    points = np.asarray(points, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or not np.isfinite(points).all()):
        return rejected('invalid_native_polyline')
    actual = points[0].copy()

    def supported(point):
        return _native_point_safe(pathfinder, point, radius=radius, height_tolerance=height_tolerance,
                                  diagnostics=diagnostics, proof_cache=proof_cache)

    if not supported(actual):
        return False
    for start, end in zip(points[:-1], points[1:]):
        horizontal = float(np.linalg.norm((end-start)[[0, 2]]))
        # A planar action cannot change floors at an unchanged XY location.
        if horizontal < 1e-8 and abs(end[1]-start[1]) > 1e-5:
            return rejected('vertical_teleport', start=start.tolist(), end=end.tolist())
        count = max(1, math.ceil(horizontal/spacing))
        for expected in np.linspace(start, end, count+1)[1:]:
            requested = expected.copy()
            requested[1] = actual[1]
            before = actual.copy()
            actual = np.asarray(pathfinder.try_step_no_sliding(
                before.astype(np.float32), requested.astype(np.float32)), float)
            distance = float(np.linalg.norm((expected-before)[[0, 2]]))
            tolerance = min(.015, .05*distance+1e-5)
            if (actual.shape != (3,) or not np.isfinite(actual).all()
                    or np.linalg.norm((actual-expected)[[0, 2]]) > tolerance
                    or abs(actual[1]-expected[1]) > height_tolerance):
                return rejected('physical_tracking', before=before.tolist(),
                                requested=requested.tolist(), actual=actual.tolist(),
                                expected=expected.tolist(), tolerance_m=tolerance)
            if not supported(actual):
                return False
    return True
