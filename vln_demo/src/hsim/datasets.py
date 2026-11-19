"""The habitat_sim asset sets available locally, and how to read each one.

A dataset object answers four questions for :class:`hsim.env.HabSceneEnv`:
``config_file`` (which scene_dataset_config to hand habitat_sim), ``episodes()``,
``title_of()`` and ``description_of()``.

Everything here is discovered from the download under ``agentic_sim/benchmark``;
nothing is invented.  Where an asset set ships no episodes (ReplicaCAD) the
spawn is sampled and the instruction is left empty, the same policy the 3DRS
backend follows -- rewriting a scene caption into a "navigation instruction"
would be making up the task.
"""

from __future__ import annotations

import gzip
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Sequence

from .env import Episode, Instruction

AGENTIC_SIM = os.environ.get(
    "AGENTIC_SIM_ROOT", "/workspace/agentic_sim")

# HSSD region labels -> what to call the room in the dropdown.
ROOM_ZH = {
    "living room": "客厅", "bedroom": "卧室", "bathroom": "卫生间",
    "kitchen": "厨房", "dining room": "餐厅", "hallway": "走廊",
    "entryway": "门厅", "laundryroom": "洗衣房", "office": "书房",
    "garage": "车库", "closet": "储物间", "stairs": "楼梯",
    "balcony": "阳台", "porch": "门廊", "other room": "其它房间",
    "workout room": "健身房", "utility room": "杂物间", "rec room": "娱乐室",
}


def _rooms(path: str) -> List[str]:
    """Region labels of one HSSD scene, ``.001`` duplicate suffixes dropped."""
    try:
        with open(path) as f:
            regs = (json.load(f) or {}).get("region_annotations") or []
    except Exception:
        return []
    out = []
    for r in regs:
        name = (r.get("label") or r.get("name") or "").strip().lower()
        out.append(name.split(".")[0])
    return [n for n in out if n]


def _room_title(rooms: Sequence[str], limit: int = 22) -> str:
    """"2卧2卫·客厅厨房" out of a scene's region labels."""
    if not rooms:
        return ""
    c = Counter(rooms)
    head = ""
    for label, unit in (("bedroom", "卧"), ("bathroom", "卫")):
        if c.get(label):
            head += "%d%s" % (c[label], unit)
    rest = [ROOM_ZH.get(r, r) for r in
            ("living room", "kitchen", "dining room", "office")
            if c.get(r)]
    title = head + ("·" if head and rest else "") + "".join(rest)
    if not title:
        title = "".join(ROOM_ZH.get(r, r) for r, _ in c.most_common(2))
    return title[:limit] + "…" if len(title) > limit else title


class PartnrHSSD:
    """PARTNR episodes over the HSSD scenes they were authored against.

    Only scenes that actually have episodes in the chosen split are exposed:
    the download carries 50 HSSD scenes but the val split visits 12 of them, and
    inventing spawns for the other 38 would put fabricated episodes next to real
    ones in the same dropdown.
    """

    name = "PARTNR / HSSD"
    goalless = True  # rearrangement propositions, not a path to score

    def __init__(self, root: Optional[str] = None, split: str = "val"):
        self.root = root or os.path.join(AGENTIC_SIM, "benchmark", "PARTNR")
        self.scene_root = os.path.join(self.root, "hssd-hab")
        self.split = split
        self.config_file = self._config()
        self._rooms: Dict[str, List[str]] = {}

    def _config(self) -> str:
        """habitat-sim 0.2.4-readable copy of the scene dataset config.

        The shipped config lists articulated objects under a ``.json`` path key
        (the newer ``*.ao_config.json`` convention); 0.2.4 only resolves ``.urdf``
        there and aborts the whole scene load when a handle comes back empty.
        The derived copy is written once, next to the original because
        habitat resolves those paths relative to the config's own directory, and
        it only adds a key -- the original file is left alone.
        """
        orig = os.path.join(self.scene_root,
                            "hssd-hab-partnr.scene_dataset_config.json")
        patched = os.path.join(self.scene_root,
                               "hssd-hab-partnr-h024.scene_dataset_config.json")
        if os.path.exists(patched):
            return patched
        with open(orig) as f:
            cfg = json.load(f)
        cfg["articulated_objects"] = {"paths": {".urdf": ["urdf/*/"]}}
        tmp = patched + ".tmp%d" % os.getpid()
        with open(tmp, "w") as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp, patched)
        return patched

    def episodes(self, scenes: Optional[Sequence[str]] = None,
                 episodes_per_scene: int = 5, seed: int = 0) -> List[Episode]:
        path = os.path.join(self.root, "partnr_episodes", "v0_0",
                            "%s.json.gz" % self.split)
        with gzip.open(path) as f:
            raw = (json.load(f) or {}).get("episodes", [])
        want = set(scenes or ())
        per: Dict[str, int] = {}
        out: List[Episode] = []
        for e in raw:
            sc = str(e.get("scene_id", "")).split("/")[-1].split(".")[0]
            if not sc or (want and sc not in want):
                continue
            if episodes_per_scene and per.get(sc, 0) >= episodes_per_scene:
                continue
            per[sc] = per.get(sc, 0) + 1
            out.append(Episode(
                episode_id=str(e.get("episode_id")),
                scene_id=os.path.join(self.scene_root, sc,
                                      "%s.scene_instance.json" % sc),
                instruction=Instruction(str(e.get("instruction") or "").strip()),
                start_position=e.get("start_position"),
                start_rotation=e.get("start_rotation"),
                info={"scene_description": self.description_of(sc)}))
        out.sort(key=lambda ep: (ep.scene, int(ep.episode_id)
                                if ep.episode_id.isdigit() else 0))
        return out

    def rooms_of(self, scene: str) -> List[str]:
        if scene not in self._rooms:
            self._rooms[scene] = _rooms(os.path.join(
                self.scene_root, "semantics", "scenes",
                "%s.semantic_config.json" % scene))
        return self._rooms[scene]

    def title_of(self, scene: str) -> str:
        return _room_title(self.rooms_of(scene))

    def description_of(self, scene: str) -> str:
        rooms = self.rooms_of(scene)
        if not rooms:
            return ""
        c = Counter(ROOM_ZH.get(r, r) for r in rooms)
        return "HSSD 户型，%d 个区域：%s" % (
            len(rooms), "、".join("%s×%d" % (k, v) if v > 1 else k
                                  for k, v in c.most_common()))


class ReplicaCADHAB:
    """The two hab2 benchmark stages: a furnished FRL apartment and a baked scene.

    Ships rearrangement episodes for a Fetch arm, not navigation ones, so this
    is free roam: one reproducible sampled spawn per scene and no instruction.
    """

    name = "Habitat-HAB / ReplicaCAD"
    goalless = True

    TITLES = {
        "Baked_sc1_staging_00_bench": "烘焙公寓（家具齐全）",
        "frl_apartment_stage_pvizplan_empty": "FRL 公寓（空房）",
    }

    def __init__(self, root: Optional[str] = None):
        self.root = root or os.path.join(
            AGENTIC_SIM, "benchmark", "Habitat-HAB", "data", "hab2_bench_assets")
        self.config_file = os.path.join(
            self.root, "hab2_bench.scene_dataset_config.json")

    def list_scenes(self) -> List[str]:
        d = os.path.join(self.root, "configs", "scenes")
        return sorted(f.split(".scene_instance.json")[0]
                      for f in os.listdir(d)
                      if f.endswith(".scene_instance.json"))

    def episodes(self, scenes: Optional[Sequence[str]] = None,
                 episodes_per_scene: int = 1, seed: int = 0) -> List[Episode]:
        """One sampled spawn per scene.

        ``episodes_per_scene`` is ignored: every extra "episode" here would be
        another spawn point this code invented, which is the same line the 3DRS
        backend draws -- two scenes are two scenes, not six episodes.
        """
        want = [s for s in (scenes or ()) if s] or self.list_scenes()
        return [Episode(episode_id="0",
                        scene_id=os.path.join(self.root, sc,
                                              "%s.scene_instance.json" % sc),
                        instruction=Instruction(""),
                        info={"scene_description": self.description_of(sc)})
                for sc in want]

    def title_of(self, scene: str) -> str:
        return self.TITLES.get(scene, "")

    def description_of(self, scene: str) -> str:
        return {
            "Baked_sc1_staging_00_bench":
                "ReplicaCAD 烘焙光照公寓，客厅、厨房、餐桌与家具齐全。",
            "frl_apartment_stage_pvizplan_empty":
                "FRL 公寓空房版本，只有房屋结构没有家具。",
        }.get(scene, "")


def available(partnr_root: Optional[str] = None, hab_root: Optional[str] = None):
    """Every asset set whose files are actually on disk, in dropdown order."""
    out = {}
    p = PartnrHSSD(partnr_root) if os.path.exists(os.path.join(
        partnr_root or os.path.join(AGENTIC_SIM, "benchmark", "PARTNR"),
        "hssd-hab")) else None
    if p is not None:
        out["partnr"] = p
    h = ReplicaCADHAB(hab_root)
    if os.path.exists(h.config_file):
        out["hab"] = h
    return out
