# -*- coding: utf-8 -*-
"""Serving side of the web console.

The page itself lives in ``src/web/`` as ordinary HTML/CSS/JS files -- there is
no build step, the browser loads them as ES modules.  This module is only the
two things the page cannot get from a file:

* :func:`static` hands out ``src/web/`` over HTTP, and
* :func:`config` is the one JSON document the page boots from (which asset sets
  are mounted, which scenes and episodes they have, the dropdown registries).

It used to be a single 700-line module with the whole page in an r-string and a
dozen ``__TOKEN__`` substitutions.  The tokens are gone: the page fetches
``/config.json`` instead, so the markup is editable as markup and the server no
longer has an opinion about the layout.
"""
import os

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

CTYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
}


def static(rel):
    """``(bytes, content-type)`` for a path under ``src/web/``, or None.

    Only ever reads inside WEB_DIR: the realpath is checked against it, so a
    ``..`` or an absolute path in the URL cannot walk out of the tree.  The
    console listens on 0.0.0.0 by default, which makes this worth being strict
    about even for a single-user demo.

    ``web/test/`` is excluded.  It lives next to the page because that is where
    it belongs, but nothing in it is part of the page, and the served surface
    should be exactly what the browser needs.
    """
    rel = (rel or "").lstrip("/")
    if not rel:
        return None
    path = os.path.realpath(os.path.join(WEB_DIR, rel))
    root = os.path.realpath(WEB_DIR)
    if path != root and not path.startswith(root + os.sep):
        return None
    if path.startswith(os.path.join(root, "test") + os.sep):
        return None
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        body = f.read()
    ctype = CTYPES.get(os.path.splitext(path)[1].lower(),
                       "application/octet-stream")
    return body, ctype


def page():
    """The shell document. Everything else it needs it fetches itself."""
    got = static("index.html")
    if got is None:
        return "<h1>src/web/index.html 不见了</h1>"
    return got[0]


def config(scenes, cur_scene, scene_eps, templates, cur_tpl,
           modes=(), cur_mode="", drivers=(), cur_driver="",
           models=(), cur_model="", maxtools=10,
           maxtools_min=1, maxtools_max=500, asset_sets=(),
           scene_titles=None, hold_speed=1.0, hold_turn=45.0, nact=4,
           episode_dimensions=None, cur_dimension="", episode_languages=None):
    """Everything the page needs that only the running console knows.

    ``asset_sets`` is ``[(key, label, description, [scene, ...]), ...]`` in
    dropdown order -- one entry per mounted asset set.  The scene dropdown only
    ever lists one set's scenes; the set dropdown is what moves between them,
    because a mixed list of MP3D ids, HSSD numbers and 3DRS hashes is not
    something you can pick a scene out of.
    """
    titles = scene_titles or {}

    def label_of(s):
        # Title first: a 3DRS id is an opaque string, so a list of them tells you
        # nothing about which scene you are picking. The id still shows, because
        # it is what the log lines and the episode dropdown are keyed by.
        ttl = titles.get(s)
        n = len(scene_eps.get(s, ()))
        label = "%s · %s" % (ttl, s) if ttl else s
        return label + (" (%d)" % n if n > 1 else "")

    def model_item(entry):
        # Five fields are current; accepting the previous four-field tuple
        # keeps small external fixtures and embedders source-compatible.
        if len(entry) == 4:
            key, label, desc, is_fm = entry
            is_lightnav = False
        else:
            key, label, desc, is_fm, is_lightnav = entry
        return {"key": key, "label": label, "desc": desc,
                "fm": bool(is_fm), "lightnav": bool(is_lightnav)}

    sets = list(asset_sets) or [("all", "全部场景", "", list(scenes))]
    scene_set = {s: key for key, _, _, scs in sets for s in scs}
    ep_dims = episode_dimensions or {}
    canonical_dims = ["perception", "grounding", "planning", "recovery",
                      "execution"]
    present_dims = {dim for by_episode in ep_dims.values()
                    for dim in by_episode.values()}
    dimensions = [dim for dim in canonical_dims if dim in present_dims]
    dimensions += sorted(present_dims.difference(dimensions))
    dim_counts = {dim: sum(value == dim for by_episode in ep_dims.values()
                           for value in by_episode.values())
                  for dim in dimensions}

    return {
        "assets": [{"key": key, "label": lbl, "desc": desc,
                    "scenes": [[s, label_of(s)] for s in scs]}
                   for key, lbl, desc, scs in sets],
        "scene_of": scene_set,
        "episodes": scene_eps,
        "episode_languages": episode_languages or {},
        # A unified capability backend remains one dataset/server, while the UI
        # uses this metadata to present five separate switchable views.
        "episode_dimensions": ep_dims,
        "dimension_label": ("Course stage" if dimensions and all(dim in ("p0", "p1", "p2") for dim in dimensions) else "Capability"),
        "capability_dimensions": [
            {"key": dim, "label": dim.title(), "count": dim_counts[dim]}
            for dim in dimensions],
        "templates": [{"key": k, "label": l, "desc": d} for k, l, d in templates],
        "modes": [{"key": k, "label": l, "desc": d} for k, l, d in modes],
        "drivers": [{"key": k, "label": l, "desc": d} for k, l, d in drivers],
        "models": [model_item(entry) for entry in models],
        "cur": {"set": scene_set.get(cur_scene, sets[0][0]),
                "scene": cur_scene, "tpl": cur_tpl, "mode": cur_mode,
                "driver": cur_driver, "model": cur_model,
                "maxtools": int(maxtools), "nact": int(nact),
                "capability": cur_dimension},
        "maxtools": {"min": int(maxtools_min), "max": int(maxtools_max)},
        # the manual-control legend quotes the real speeds, so DEMO_HOLD_SPEED /
        # _TURN cannot leave it describing a robot that no longer exists
        "hold": {"speed": float(hold_speed), "turn": float(hold_turn)},
    }
