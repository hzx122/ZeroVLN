"""Backend-aware path planning for visual targets."""

from __future__ import annotations

import heapq
import math

import numpy as np


PLANNER_CLEARANCE_M = 0.05


def _square_erode(free, radius_px):
    """Square-footprint passability, matching ABotN's collision oracle."""
    free = np.asarray(free, dtype=bool)
    if radius_px <= 0:
        return free.copy()
    blocked = (~free).astype(np.int32)
    summed = np.pad(blocked, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    height, width = free.shape
    y, x = np.ogrid[:height, :width]
    y0, y1 = np.maximum(0, y-radius_px), np.minimum(height, y+radius_px+1)
    x0, x1 = np.maximum(0, x-radius_px), np.minimum(width, x+radius_px+1)
    count = summed[y1, x1] - summed[y0, x1] - summed[y1, x0] + summed[y0, x0]
    full = ((y1-y0) == 2*radius_px+1) & ((x1-x0) == 2*radius_px+1)
    return (count == 0) & full


def _occupancy_passable(env, occupancy, footprint_radius):
    """Return the planner/collision footprint grid, cached per static map."""
    free_source = getattr(occupancy, "free", None)
    if free_source is None:
        return None
    free = np.asarray(free_source, dtype=bool)
    if free.ndim != 2:
        return None
    clearance = getattr(occupancy, "clearance_m", None)
    cache_key = (id(occupancy), id(free_source), id(clearance), free.shape,
                 float(footprint_radius))
    cache = getattr(env, "_vnt_passability_cache", None)
    if cache is not None and cache[0] == cache_key:
        return cache[1]

    if clearance is not None:
        passable = free & (np.asarray(clearance) > footprint_radius)
    else:
        if hasattr(occupancy, "radius_px"):
            footprint_px = occupancy.radius_px(footprint_radius)
        else:
            metres_per_pixel = float(getattr(occupancy, "mpp", 0.0) or 0.0)
            if metres_per_pixel <= 0:
                return None
            footprint_px = int(math.ceil(footprint_radius/metres_per_pixel))
        passable = _square_erode(free, footprint_px)
    try:
        env._vnt_passability_cache = (cache_key, passable)
    except (AttributeError, TypeError):
        pass
    return passable


def _nearest_true(mask, column, row, limit_px):
    height, width = mask.shape
    column, row = int(column), int(row)
    if 0 <= row < height and 0 <= column < width and mask[row, column]:
        return column, row
    for radius in range(1, max(1, int(limit_px))+1):
        x0, x1 = max(0, column-radius), min(width, column+radius+1)
        y0, y1 = max(0, row-radius), min(height, row+radius+1)
        ys, xs = np.nonzero(mask[y0:y1, x0:x1])
        if xs.size:
            xs, ys = xs+x0, ys+y0
            index = int(np.argmin((xs-column)**2 + (ys-row)**2))
            return int(xs[index]), int(ys[index])
    return None


_GRID_NEIGHBOURS = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
)


def _astar_grid(passable, start, goal):
    """Return an eight-connected shortest grid path in (column, row)."""
    if start == goal:
        return [start]
    height, width = passable.shape
    queue = [(0.0, 0.0, start)]
    cost = {start: 0.0}
    parent = {}
    while queue:
        _, current_cost, current = heapq.heappop(queue)
        if current_cost != cost.get(current):
            continue
        if current == goal:
            path = [current]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return list(reversed(path))
        x, y = current
        for dx, dy, step_cost in _GRID_NEIGHBOURS:
            nx, ny = x+dx, y+dy
            if not (0 <= nx < width and 0 <= ny < height and passable[ny, nx]):
                continue
            if dx and dy and not (passable[y, nx] and passable[ny, x]):
                continue
            neighbour = (nx, ny)
            candidate_cost = current_cost+step_cost
            if candidate_cost >= cost.get(neighbour, float("inf")):
                continue
            cost[neighbour] = candidate_cost
            parent[neighbour] = current
            heuristic = math.hypot(goal[0]-nx, goal[1]-ny)
            heapq.heappush(queue,
                           (candidate_cost+heuristic, candidate_cost, neighbour))
    return None


def _string_pull(points, line_free):
    if len(points) <= 2:
        return points
    output, anchor = [points[0]], 0
    while anchor < len(points)-1:
        farthest = anchor+1
        for candidate in range(len(points)-1, anchor, -1):
            if line_free(points[anchor], points[candidate]):
                farthest = candidate
                break
        output.append(points[farthest])
        anchor = farthest
    return output


def plan_occupancy(env, target_hab):
    simulator = env.sim
    occupancy = simulator.occ
    radius = float(getattr(env, "robot_radius", 0.10))+PLANNER_CLEARANCE_M
    passable = _occupancy_passable(env, occupancy, radius)
    if passable is None:
        raise ValueError("occupancy map 缺少可用的碰撞栅格")
    start_native = tuple(float(value) for value in simulator.xy)
    goal_native = (float(target_hab[0]), -float(target_hab[1]))
    start_column, start_row = occupancy.to_pixel(*start_native)
    goal_column, goal_row = occupancy.to_pixel(*goal_native)
    metres_per_pixel = float(
        getattr(occupancy, "mpp", getattr(occupancy, "scale", 0.05)))
    search_limit = max(40, int(math.ceil(2.0/max(metres_per_pixel, 1e-6))))
    start = _nearest_true(passable, start_column, start_row, search_limit)
    goal = _nearest_true(passable, goal_column, goal_row, search_limit)
    if start is None or goal is None:
        raise ValueError("视觉目标附近找不到可达导航目标")
    pixels = _astar_grid(passable, start, goal)
    if not pixels:
        raise ValueError("视觉目标与当前位置在 occupancy map 上不连通")
    native = [tuple(float(value) for value in occupancy.to_world(c, r))
              for c, r in pixels]
    native[0] = start_native
    native[-1] = tuple(float(value) for value in occupancy.to_world(*goal))
    native = _string_pull(
        native,
        lambda a, b: bool(occupancy.seg_free(a[0], a[1], b[0], b[1], radius)),
    )
    return [(x, -y) for x, y in native], "occupancy_astar"


def plan_habitat(env, target_hab):
    simulator = env.sim
    pathfinder = getattr(simulator, "pathfinder", None)
    if pathfinder is None or not getattr(pathfinder, "is_loaded", True):
        return None
    try:
        import habitat_sim
    except ImportError:
        return None
    state = simulator.get_agent_state()
    start = np.asarray(state.position, dtype=np.float32)
    goal = np.array([target_hab[0], start[1], target_hab[1]], dtype=np.float32)
    goal = np.asarray(pathfinder.snap_point(goal), dtype=np.float32)
    if not np.isfinite(goal).all():
        raise ValueError("视觉目标无法 snap 到 Habitat navmesh")
    query = habitat_sim.ShortestPath()
    query.requested_start = start
    query.requested_end = goal
    if not pathfinder.find_path(query):
        raise ValueError("视觉目标与当前位置在 Habitat navmesh 上不连通")
    return (
        [(float(point[0]), float(point[2])) for point in query.points],
        "shortest_path_follower",
    )


def plan_mall(env, target_hab):
    simulator = env.sim
    assets = getattr(simulator, "assets", None)
    if assets is None or not getattr(assets, "has_topo", False):
        return None
    start = (float(simulator.xz[0]), float(simulator.xz[1]))
    snapped = assets.snap(float(target_hab[0]), float(target_hab[1]), limit_m=2.0)
    if snapped is None:
        raise ValueError("视觉目标附近没有可达的商场导航目标")
    source, goal = assets.nearest_node(*start), assets.nearest_node(*snapped)
    if source is None or goal is None:
        raise ValueError("商场拓扑图无法定位视觉 waypoint")
    queue, cost, parent = [(0.0, source)], {source: 0.0}, {}
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != cost.get(node):
            continue
        if node == goal:
            break
        for neighbour, weight in assets._adj.get(node, ()):
            candidate = distance+float(weight)
            if candidate < cost.get(neighbour, float("inf")):
                cost[neighbour], parent[neighbour] = candidate, node
                heapq.heappush(queue, (candidate, neighbour))
    if goal not in cost:
        raise ValueError("视觉目标与当前位置在商场拓扑图上不连通")
    nodes = [goal]
    while nodes[-1] != source:
        nodes.append(parent[nodes[-1]])
    nodes.reverse()
    points = [start]+[tuple(map(float, assets._node_xy[node]))
                     for node in nodes[1:]]
    points.append(tuple(map(float, snapped)))
    return points, "topology_dijkstra"


def _mask_seg_free(assets, mask, a, b):
    """Check a segment against the same rounded pixels used by locnav motion."""
    p0, q0 = assets.to_pixel(*a)
    p1, q1 = assets.to_pixel(*b)
    count = max(1, int(math.ceil(math.hypot(p1-p0, q1-q0) * 2)))
    for t in np.linspace(0.0, 1.0, count + 1):
        p, q = int(round(p0 + (p1-p0)*t)), int(round(q0 + (q1-q0)*t))
        if not (0 <= q < mask.shape[0] and 0 <= p < mask.shape[1] and mask[q, p]):
            return False
    return True


def plan_locnav_mask(env, target_hab):
    """Lepu has a walk mask but no topology edges; preserve the other planners."""
    assets = getattr(env.sim, "assets", None)
    if getattr(assets, "navigation_planner", None) != "mask_astar":
        return None
    mask = np.asarray(assets.mask, dtype=bool)
    start = tuple(map(float, env.sim.xz))
    if not assets.walkable(*start):
        raise ValueError("当前位置不在可走域内")
    goal = assets.snap(float(target_hab[0]), float(target_hab[1]), limit_m=2.0)
    if goal is None:
        raise ValueError("视觉目标附近没有可走域")
    start_pixel = tuple(int(round(v)) for v in assets.to_pixel(*start))
    goal_pixel = tuple(int(round(v)) for v in assets.to_pixel(*goal))
    pixels = _astar_grid(mask, start_pixel, goal_pixel)
    if not pixels:
        raise ValueError("视觉目标与当前位置在可走域上不连通")
    points = [start] + [tuple(map(float, assets.to_world(c, r))) for c, r in pixels] + [goal]
    points = _string_pull(points, lambda a, b: _mask_seg_free(assets, mask, a, b))
    if not all(_mask_seg_free(assets, mask, a, b) for a, b in zip(points, points[1:])):
        raise ValueError("视觉路径穿过不可走区域")
    return points, "mask_astar"


def _triangle_contains_xz(triangle, point, epsilon=1e-7):
    p = np.asarray(point, dtype=float)
    a, b, c = np.asarray(triangle, dtype=float)[:, [0, 2]]
    v0, v1, v2 = b-a, c-a, p-a
    denominator = v0[0]*v1[1]-v1[0]*v0[1]
    if abs(denominator) < epsilon:
        return False
    u = (v2[0]*v1[1]-v1[0]*v2[1])/denominator
    v = (v0[0]*v2[1]-v2[0]*v0[1])/denominator
    return u >= -epsilon and v >= -epsilon and u+v <= 1.0+epsilon


def _nav_poly_at(navigation, point3):
    best = None
    for triangle_index in navigation.candidates(float(point3[0]), float(point3[2])):
        triangle = navigation.tris[int(triangle_index)]
        if not _triangle_contains_xz(triangle, (point3[0], point3[2])):
            continue
        height = float(np.mean(triangle[:, 1]))
        delta = abs(height-float(point3[1]))
        if best is None or delta < best[0]:
            best = (delta, int(navigation.tri_poly[int(triangle_index)]))
    return None if best is None else best[1]


def plan_rs3d(env, target_hab):
    simulator = env.sim
    navigation = getattr(simulator, "nav", None)
    if navigation is None or not hasattr(navigation, "tri_poly"):
        return None
    start3 = np.asarray(simulator.get_agent_state().position, dtype=float)
    proposed = np.array([target_hab[0], start3[1], target_hab[1]], dtype=float)
    goal3 = navigation.snap(proposed)
    if goal3 is None:
        goal3 = navigation.snap_nearest(proposed, radius=2.0)
    if goal3 is None:
        raise ValueError("视觉目标无法 snap 到 3DRS navmesh")
    start_poly = _nav_poly_at(navigation, start3)
    goal_poly = _nav_poly_at(navigation, goal3)
    if start_poly is None or goal_poly is None:
        raise ValueError("3DRS navmesh 无法定位视觉 waypoint 所属多边形")
    mesh = navigation.nav
    centres = [np.asarray(mesh.verts[polygon], dtype=float).mean(axis=0)
               if len(polygon) else np.zeros(3) for polygon in mesh.polys]
    queue = [(0.0, 0.0, start_poly)]
    cost, parent = {start_poly: 0.0}, {}
    while queue:
        _, current_cost, node = heapq.heappop(queue)
        if current_cost != cost.get(node):
            continue
        if node == goal_poly:
            break
        for neighbour in mesh.neighbours[node]:
            if (neighbour < 0 or neighbour >= len(mesh.polys)
                    or (mesh.walkable is not None and not mesh.walkable[neighbour])):
                continue
            weight = float(np.linalg.norm(
                (centres[neighbour]-centres[node])[[0, 2]]))
            candidate = current_cost+weight
            if candidate < cost.get(neighbour, float("inf")):
                cost[neighbour], parent[neighbour] = candidate, node
                heuristic = float(np.linalg.norm((centres[neighbour]-goal3)[[0, 2]]))
                heapq.heappush(queue, (candidate+heuristic, candidate, neighbour))
    if goal_poly not in cost:
        raise ValueError("视觉目标与当前位置在 3DRS navmesh 上不连通")
    polygons = [goal_poly]
    while polygons[-1] != start_poly:
        polygons.append(parent[polygons[-1]])
    polygons.reverse()
    points3 = [start3]+[centres[index] for index in polygons[1:-1]]+[goal3]

    def line_free(a, b):
        reached, hit = navigation.try_step(a, b, slide=False)
        return not hit and float(np.linalg.norm((reached-b)[[0, 2]])) < 0.03

    points3 = _string_pull(points3, line_free)
    return (
        [(float(point[0]), float(point[2])) for point in points3],
        "navmesh_astar",
    )


def plan_path(env, target_hab):
    """Plan to a visual target using the active backend's navigation map."""
    simulator = env.sim
    if (getattr(simulator, "occ", None) is not None
            and getattr(simulator, "xy", None) is not None):
        return plan_occupancy(env, target_hab)
    for planner in (plan_habitat, plan_mall, plan_locnav_mask, plan_rs3d):
        result = planner(env, target_hab)
        if result is not None:
            return result
    raise ValueError("当前 backend 还没有可用的 map-aware waypoint planner")
