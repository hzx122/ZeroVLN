"""Full training episode catalogues, using the installed Habitat data loaders."""

import os

from rxr import SharedSimEnv


def load_episodes(family, r2r_config=None, role="guide"):
    if family == "rxr":
        import rxr

        return rxr.load_episodes(split="train", per_scene=0, languages="*",
                                 roles=role, prefix="rxr-train-%s-" % role)
    if family != "r2r":
        raise ValueError("Unknown VLN-CE dataset: %s" % family)

    from habitat.core.registry import registry
    from omegaconf import OmegaConf

    # Preserve the host's configured data root, while loading a separate split.
    cfg = OmegaConf.create(OmegaConf.to_container(r2r_config, resolve=True))
    cfg.split = "train"
    cfg.content_scenes = ["*"]
    episodes = registry.get_dataset(cfg.type)(cfg).episodes
    meshes = {path: os.path.isfile(path) for path in
              {ep.scene_id for ep in episodes}}
    have, missing = [], set()
    for ep in episodes:
        scene = os.path.basename(os.path.dirname(ep.scene_id))
        if not meshes[ep.scene_id]:
            missing.add(scene)
            continue
        ep.scene = "r2r-train-" + scene
        have.append(ep)
    return have, sorted(missing)


def reference_map(env, tdm, maps):
    """Browser-only GT overlay; never alter the map used by tools or metrics."""
    if maps is None or not isinstance(tdm, dict) or "map" not in tdm:
        return tdm
    episode = env.current_episode
    result = dict(tdm)
    original = tdm["map"]
    raster = original.copy()
    path = getattr(episode, "reference_path", None)
    goals = getattr(episode, "goals", ())
    def grid(point):
        return maps.to_grid(point[2], point[0], raster.shape[:2],
                            pathfinder=env.sim.pathfinder)
    if path:
        points = [episode.start_position, *path]
        if goals:
            points.append(goals[0].position)
        maps.draw_path(raster, [grid(p) for p in points],
                       color=maps.MAP_SHORTEST_PATH_COLOR, thickness=2)
    # Keep the actual walked trail visible over the reference route.
    raster[original >= 10] = original[original >= 10]
    raster[original == maps.MAP_SOURCE_POINT_INDICATOR] = maps.MAP_SOURCE_POINT_INDICATOR
    radius = max(1, min(raster.shape[:2]) // 256)
    for goal in goals:
        row, col = map(int, grid(goal.position))
        raster[max(0, row-radius):min(raster.shape[0], row+radius+1),
               max(0, col-radius):min(raster.shape[1], col+radius+1)] = maps.MAP_TARGET_POINT_INDICATOR
    result["map"] = raster
    return result
