"""Floor-aware display maps; navigation geometry and scoring stay untouched."""
import numpy as np


def agent_map_raster(raster):
    """Remove privileged Habitat annotations without changing metrics/source data.

    Habitat codes: 6=goal, 7=shortest path, 8=goal viewpoints, 9=goal AABB.
    Real visited trails use codes >=10; keep them and the observed start (4).
    """
    result = np.asarray(raster, dtype=np.uint8).copy()
    result[np.isin(result, (6, 7, 8, 9))] = 1
    return result


def floor_heights(vertices):
    """Find broad, nearly horizontal surfaces, rather than individual steps.

    Habitat supplies triangle-soup navmesh vertices. Weight by triangle area:
    counting vertices instead would favour densely triangulated stairs.
    """
    triangles = np.asarray(vertices, dtype=float).reshape(-1, 3, 3)
    if not len(triangles):
        return []
    area = np.abs(np.cross(triangles[:, 1] - triangles[:, 0],
                           triangles[:, 2] - triangles[:, 0])[:, 1]) / 2
    flat = (np.ptp(triangles[:, :, 1], axis=1) <= 0.2) & (area > 0)
    height, area = triangles[flat, :, 1].mean(axis=1), area[flat]
    if not len(area):
        return []
    bins = np.rint(height / 0.25).astype(int)
    weights = {int(b): float(area[bins == b].sum()) for b in np.unique(bins)}
    levels = []
    for b in sorted(weights, key=weights.get, reverse=True):
        if weights[b] < max(1.0, max(weights.values()) * 0.03):
            continue
        if any(abs(b * 0.25 - h) < 0.75 for h in levels):
            continue
        nearby = np.abs(height - b * 0.25) <= 0.25
        levels.append(float(np.average(height[nearby], weights=area[nearby])))
    return sorted(levels)


def select_floor(levels, height, previous=None):
    if not levels:
        return float(height)
    nearest = min(levels, key=lambda level: abs(level - height))
    # A small mezzanine may not form a broad histogram peak. Never project an
    # agent several storeys away onto one of the detected large floors.
    if abs(nearest - height) > 1.75:
        return float(height)
    # Prevent flicker when the agent moves back and forth on a staircase.
    if previous in levels and abs(previous - height) <= abs(nearest - height) + 0.3:
        return previous
    return nearest


def register():
    """Register lazily so non-Habitat backends need no Habitat dependency."""
    from habitat.core.registry import registry
    from habitat.tasks.nav.nav import TopDownMap
    from habitat.utils.visualizations import maps

    if registry.get_measure("FloorTopDownMap") is not None:
        return

    @registry.register_measure
    class FloorTopDownMap(TopDownMap):
        def get_original_map(self):
            self._levels = floor_heights(self._sim.pathfinder.build_navmesh_vertices())
            self._floor_height = select_floor(
                self._levels, float(self._sim.get_agent_state().position[1]))
            self._floor_cache = {}
            return self._new_floor_map()

        def _new_floor_map(self):
            raster = maps.get_topdown_map(
                self._sim.pathfinder, self._floor_height,
                map_resolution=self._map_resolution, draw_border=self._config.draw_border)
            self._fog_of_war_mask = (np.zeros_like(raster)
                                     if self._config.fog_of_war.draw else None)
            return raster

        def _is_on_same_floor(self, height, ref_floor_height=None, ceiling_height=2.0):
            reference = self._floor_height if ref_floor_height is None else ref_floor_height
            return abs(float(height) - reference) <= 0.5

        def update_metric(self, episode, action, *args, **kwargs):
            position = self._sim.get_agent_state().position
            height = select_floor(self._levels, float(position[1]), self._floor_height)
            if abs(height - self._floor_height) > 0.25:
                self._floor_cache[self._floor_height] = (
                    self._top_down_map, self._fog_of_war_mask)
                if len(self._floor_cache) > 8:
                    del self._floor_cache[next(iter(self._floor_cache))]
                self._floor_height = height
                cached = self._floor_cache.get(height)
                if cached is not None:
                    self._top_down_map, self._fog_of_war_mask = cached
                else:
                    self._top_down_map = self._new_floor_map()
                    if self._config.draw_source and self._is_on_same_floor(episode.start_position[1]):
                        self._draw_point(episode.start_position, maps.MAP_SOURCE_POINT_INDICATOR)
                # Do not join poses from different floors with a spurious line.
                self._previous_xy_location = [None] * len(self._previous_xy_location)
            super().update_metric(episode, action, *args, **kwargs)
