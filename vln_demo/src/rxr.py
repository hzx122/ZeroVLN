"""RxR val_unseen, mounted on the R2R backend's simulator.

RxR (Room-Across-Room) is R2R's wordier sibling: same MP3D houses, same VLN-CE
episode skeleton, but the instructions are long, multilingual and written against
a human demonstration -- "walk down the stairs, turn right, go through the door
and stop on the carpet" rather than R2R's one-clause hop.  Two facts about it
shape this whole module.

*It needs its own loader.*  habitat's ``R2RVLN-v1`` cannot read RxR: the file
carries no ``instruction_vocab``, and every instruction has four extra fields
(``language``, ``annotator_id``, ``edit_distance``, ``timed_instruction``) that
habitat's ``InstructionData`` refuses.  RynnBrain already ships a loader for
exactly this format (``RxR-VLN-CE-v1``, registered by
``habitat_extensions.task``), so we configure that instead of writing a third
parser.  Everything downstream is unchanged: the episodes are still
``VLNEpisode``s with ``goals`` (radius 3 m, R2R's convention) and a
``reference_path``, so distance_to_goal / success / spl / oracle_* all score.

*It does not need its own simulator.*  RxR val_unseen uses exactly the same 11
MP3D scenes as R2R val_unseen -- set equality, verified, with every mesh already
on disk -- and habitat's ``current_episode`` setter clears
``_episode_from_iter_on_reset``, so an ``Env`` resets onto whatever episode you
hand it, dataset membership be damned.  ``SharedSimEnv`` is that observation
made concrete: the habitat backend's Env, with our episode list bolted on.  The
alternative, a second ``habitat_sim`` in the same process, costs another MP3D
mesh in memory and is not obviously safe either -- mounting ReplicaCAD alongside
the other sets is what turns every render into a red wash (README §1).

Scene ids get an ``rxr-`` prefix for the same reason ABotN prefixes its three
benchmarks: ``SCENE_BACKEND`` maps one scene id to one asset set, and R2R has
already claimed the bare MP3D ids.
"""

import os


# Keep the legacy evaluation view on guide trajectories. Training views load
# guide and follower trajectories separately, with the original instructions.
SPLIT = os.environ.get("RXR_SPLIT", "val_unseen")
ROLES = os.environ.get("RXR_ROLES", "guide")
DATA_PATH = os.environ.get(
    "RXR_DATA_PATH", "data/datasets/rxr/{split}/{split}_{role}.json.gz")
SCENES_DIR = os.environ.get("RXR_SCENES_DIR", "data/scene_datasets/")
# Compact English evaluation view: 3669 of val_unseen's 11006 episodes.
# This is a display default, independent of model capabilities. Training views
# explicitly pass ``*`` to keep English, Hindi and Telugu.
LANGUAGES = os.environ.get("RXR_LANGUAGES", "en-US,en-IN")
# Per scene, not in total. All 3669 would put 615 options in one scene's dropdown
# and add ~1.6 MB of instruction text to the page -- RxR instructions run 400+
# chars where R2R's run 120. 0 = keep every one.
EPISODES_PER_SCENE = int(os.environ.get("RXR_EPISODES_PER_SCENE", "30"))
PREFIX = "rxr-"


def _csv(spec):
    return [s.strip() for s in spec.split(",") if s.strip()]


def _mp3d_id(episode):
    return episode.scene_id.split("/")[-2]


def load_episodes(split=None, per_scene=None, languages=None, roles=None,
                  prefix=PREFIX):
    from vln_gym.native_episode_cache import enabled, cached_episodes
    if not enabled():
        return _load_episodes_uncached(split, per_scene, languages, roles, prefix)
    signature = dict(kind='rxr', split=split or SPLIT,
        per_scene=EPISODES_PER_SCENE if per_scene is None else per_scene,
        languages=languages or LANGUAGES, roles=roles or ROLES, prefix=prefix,
        data_path=DATA_PATH, scenes_dir=os.path.realpath(SCENES_DIR))
    role_list = _csv(signature['roles'])
    if '*' in role_list:
        role_list = ['guide', 'follower']
    sources = [DATA_PATH.format(split=signature['split'], role=r) for r in role_list]
    def load():
        episodes, missing = _load_episodes_uncached(split, per_scene, languages, roles, prefix)
        return missing, episodes
    missing, episodes, _ = cached_episodes(signature, sources, load)
    return episodes, missing


def _load_episodes_uncached(split=None, per_scene=None, languages=None, roles=None,
                            prefix=PREFIX):
    """RxR episodes, English by default, one flat list with prefixed scene ids.

    No ``Env`` is built here and none is needed: the dataset class only reads a
    json.gz.  ``habitat_extensions.task`` is what registers ``RxR-VLN-CE-v1``,
    and it lives in RynnBrain's ``src`` -- already on ``sys.path`` because the
    console puts it there for ``rynnvln_eval``.
    """
    import habitat_extensions.task                     # noqa: F401  (registers)
    from habitat.core.registry import registry

    split = split or SPLIT
    cfg = {
        "data_path": DATA_PATH,
        "split": split,
        "scenes_dir": SCENES_DIR,
        "roles": _csv(roles or ROLES),
        # the loader's own masks; we do the per-scene thinning ourselves below
        "languages": _csv(languages or LANGUAGES) or ["*"],
        "content_scenes": ["*"],
        "episodes_allowed": ["*"],
    }
    episodes = registry.get_dataset("RxR-VLN-CE-v1")(cfg).episodes

    # A scene whose mesh is absent would be a dropdown entry that throws on
    # switch. Nothing is missing today; this is so it stays a startup line rather
    # than a red error mid-demo if the MP3D copy is ever thinned.
    have, missing = [], set()
    meshes = {path: os.path.isfile(path) for path in
              {ep.scene_id for ep in episodes}}
    for ep in episodes:
        if meshes[ep.scene_id]:
            have.append(ep)
        else:
            missing.add(_mp3d_id(ep))

    by_scene = {}
    for ep in have:
        by_scene.setdefault(_mp3d_id(ep), []).append(ep)

    n = EPISODES_PER_SCENE if per_scene is None else per_scene
    out = []
    for scene, eps in sorted(by_scene.items()):
        eps.sort(key=lambda e: int(e.episode_id) if str(e.episode_id).isdigit()
                 else str(e.episode_id))
        if n and len(eps) > n:
            # Stride rather than head: episode ids run in trajectory order, so
            # the first 30 would be 30 instructions for a handful of nearby
            # paths. This spreads the sample over the whole scene.
            eps = eps[::len(eps) // n][:n]
        for ep in eps:
            # scene_of() prefers this over the mesh path precisely so two asset
            # sets can share a scene directory (see abotn/env.py).
            ep.scene = prefix + scene
            out.append(ep)
    return out, sorted(missing)


class SharedSimEnv:
    """One habitat ``Env``'s simulator, a second episode list.

    Everything the console asks of an env -- ``step`` / ``reset`` /
    ``episode_over`` / ``get_metrics`` / ``sim`` -- is forwarded to the Env this
    wraps; only ``episodes`` is ours.  Assigning ``current_episode`` forwards
    too, and that is the trick that makes the whole thing work: habitat's setter
    marks the episode as externally supplied, so the ``reset()`` right after it
    loads our RxR episode even though the Env's own dataset is R2R.

    The two asset sets therefore take turns on one simulator.  That is safe here
    only because the worker thread owns it and switching scenes is a
    ``reconfigure`` either way -- see the module docstring in vln_model.py.
    """

    def __init__(self, env, episodes):
        self._env = env
        from vln_gym.native_episode_cache import FrozenEpisodes
        self.episodes = episodes if isinstance(episodes, FrozenEpisodes) else list(episodes)

    def __getattr__(self, name):
        # Guard the forwarding target itself: without this, an attribute lookup
        # before __init__ finished (unpickling, a subclass, a typo) recurses
        # until the stack ends instead of raising AttributeError.
        if name == "_env":
            raise AttributeError(name)
        return getattr(self._env, name)

    def __setattr__(self, name, value):
        if name in ("_env", "episodes"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._env, name, value)
