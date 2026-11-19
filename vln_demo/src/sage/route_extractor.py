"""Trajectory-grounded route extraction for SAGE-3D demo episodes.

The official SAGE-3D instruction pipeline feeds the LLM only a start/end object
id plus an LLM-fabricated scene text -- it never sees the A* path, so the
generated instruction can't reference the route the robot actually takes. This
module is the inverse: it turns an A* golden path (waypoints, geometry frame)
into a structured ``route`` dict -- rooms crossed, doors passed, turns made,
objects passed -- consumed by deterministic instruction templates. Grounding
becomes structural (the route only ever contains elements the path really
visited), rather than depending on unconstrained model generation.

Pure CPU, no LLM. Reuses ``OccMap`` (``to_pixel``/``to_world``) and the
usd_rooms topology the loader already mounts (``_usd_room_region`` /
``_usd_door_opening``). labels.json object bboxes are in the same geometry
frame as the waypoints and the occupancy grid, so no extra transform is needed
(see :mod:`sage.occmap` docstring + the ``_to_geometry_frame`` note in
:mod:`sage.datasets`).

Output ``route`` schema (JSON-serializable, fed to the verbalize prompt)::

    {
      "scene": "840483",
      "start": {"object": "bed_2", "room": "bedroom"},
      "goal":  {"object": "fridge_1", "room": "kitchen"},
      "cross_rooms": ["bedroom", "hallway", "kitchen"],   # dedup, order preserved
      "doors_crossed": 2,
      "turns": ["right", "straight", "left"],              # per simplified segment
      "nearby_objects": ["sofa_1", "table_3", ...],        # passed within 0.5 m, order
      "path_length_m": 8.7,
      "room_names": {"1": "bedroom", "3": "kitchen", ...}  # room_id -> friendly name
    }
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

# label (normalized snake_case) -> friendly room name. Several furniture labels
# map to the same room class; the dominant class among objects whose bbox centre
# falls in a room_id names that room. Anything unmapped falls back to "room N".
LABEL_TO_ROOM = {
    "bed": "bedroom", "wardrobe": "bedroom", "nightstand": "bedroom",
    "dresser": "bedroom", "pillow": "bedroom", "double_bed": "bedroom",
    "single_bed": "bedroom", "bunk_bed": "bedroom",
    "refrigerator": "kitchen", "fridge": "kitchen", "microwave": "kitchen",
    "microwave_oven": "kitchen", "stove": "kitchen", "oven": "kitchen",
    "range_hood": "kitchen", "kitchen_cabinet": "kitchen", "kitchen_counter": "kitchen",
    "kitchen_island": "kitchen", "cabinet": "kitchen",
    "sofa": "living room", "couch": "living room", "multi_person_sofa": "living room",
    "coffee_table": "living room", "tv": "living room", "television": "living room",
    "tv_cabinet": "living room", "fireplace": "living room",
    "dining_table": "dining room", "dining_chair": "dining room",
    "dining": "dining room",
    "bathtub": "bathroom", "toilet": "bathroom", "shower": "bathroom",
    "wash_basin": "bathroom", "bathroom_cabinet": "bathroom",
    "basin": "bathroom", "bidet": "bathroom",
    "desk": "study", "bookshelf": "study", "bookshelf": "study",
    "office_chair": "study", "computer_desk": "study",
}

# Room masks describe topology, not necessarily a single semantic room.  In
# open-plan scenes one connected component can contain a kitchen, living area
# and dining area.  Natural-language area names therefore come from *local*
# high-specificity anchors around the queried pose, never from whole-component
# majority voting.  Generic cabinets/chairs/lamps are deliberately absent.
FUNCTIONAL_AREA_ANCHORS = {
    "bed": "sleeping area", "double_bed": "sleeping area",
    "single_bed": "sleeping area", "bunk_bed": "sleeping area",
    "nightstand": "sleeping area",
    "dining_table": "dining area",
    "sofa": "living area", "couch": "living area",
    "multi_person_sofa": "living area", "coffee_table": "living area",
    "tv_cabinet": "living area",
    "refrigerator": "kitchen area", "fridge": "kitchen area",
    "microwave": "kitchen area", "microwave_oven": "kitchen area",
    "stove": "kitchen area", "oven": "kitchen area",
    "range_hood": "kitchen area", "kitchen_counter": "kitchen area",
    "kitchen_island": "kitchen area",
    "bathtub": "bathroom area", "toilet": "bathroom area",
    "shower": "bathroom area", "shower_head": "bathroom area",
    "wash_basin": "bathroom area", "basin": "bathroom area",
    "bidet": "bathroom area",
    "washing_machine": "laundry area",
    "desk": "study area", "computer_desk": "study area",
    "bookshelf": "study area",
}
AREA_ANCHOR_MAX_M = 2.5
AREA_SCORE_SCALE_M = 1.25
AREA_DIRECT_EVIDENCE_BONUS = 0.35
AREA_MIN_SCORE = 0.30
AREA_MIN_SCORE_MARGIN = 0.15
AREA_MIN_SCORE_RATIO = 1.30

# turn quantization
TURN_THRESH = math.radians(30.0)      # |Δyaw| above this = a real turn
NEARBY_M = 0.5                        # object is "passed" if path comes within this
SAMPLE_STEP_M = 0.1                   # dense resample along polyline for mask queries


def _normalize(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def _room_label_map(occ) -> Optional[np.ndarray]:
    """Per-room integer label map from the usd_rooms ``room_region`` bool mask.

    Connected components (8-connectivity) of ``room_region`` -- doorways are a
    separate mask, so two rooms sharing a door stay distinct components, which
    is exactly the "crossed into another room" semantics the route needs.
    Returns ``None`` when usd_rooms topology is unavailable (older artifact).
    """
    rr = getattr(occ, "_usd_room_region", None)
    if rr is None:
        return None
    try:
        from scipy.ndimage import label
    except Exception:
        return None
    struct = np.ones((3, 3), dtype=int)   # 8-connectivity
    lab, n = label(np.asarray(rr, dtype=bool), structure=struct)
    return lab


def _bbox_centre_xy(bb: list) -> Optional[Tuple[float, float]]:
    """Axis-aligned bbox centre (geometry frame) from an 8-corner labels.json
    bounding_box. Corners carry {x,y,z}; collapse to (min..max) on x/y."""
    if not bb:
        return None
    xs = [float(p["x"]) for p in bb]
    ys = [float(p["y"]) for p in bb]
    return (0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)))


def _bbox_xyxy(bb: list) -> Optional[Tuple[float, float, float, float]]:
    if not bb:
        return None
    xs = [float(p["x"]) for p in bb]
    ys = [float(p["y"]) for p in bb]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_distance_xy(xy: Tuple[float, float], bb) -> float:
    x, y = xy
    x0, y0, x1, y1 = bb
    return math.hypot(max(x0 - x, 0.0, x - x1),
                      max(y0 - y, 0.0, y - y1))


def _functional_area(xy: Tuple[float, float], occ,
                     label_index: Dict[str, dict],
                     room_lab: Optional[np.ndarray],
                     object_id: str = "") -> dict:
    """Ground a local functional area by confidence-weighted local evidence.

    All compatible anchors in the same topological room contribute, with
    distance decay and a bounded bonus when the queried object is itself a
    functional anchor. A winner must clear absolute, margin and ratio tests;
    otherwise the result is explicitly ambiguous. This avoids both global room
    majority errors and one-nearest-object accidents in open-plan layouts.
    """
    query_room = _room_at(room_lab, occ, xy[0], xy[1])
    evidence_by_area: Dict[str, List[dict]] = {}
    for oid, info in label_index.items():
        label = _normalize(info.get("label", ""))
        area = FUNCTIONAL_AREA_ANCHORS.get(label)
        bb = info.get("bbox_xyxy")
        if area is None or bb is None:
            continue
        dist = _bbox_distance_xy(xy, bb)
        direct = oid == object_id
        centre = info.get("centre_xy")
        anchor_room = (_room_at(room_lab, occ, centre[0], centre[1])
                       if centre is not None else 0)
        if not direct and query_room and anchor_room not in (0, query_room):
            continue
        if dist > AREA_ANCHOR_MAX_M and not direct:
            continue
        score = math.exp(-dist / AREA_SCORE_SCALE_M)
        if direct:
            score += AREA_DIRECT_EVIDENCE_BONUS
        rec = {"object_id": oid, "label": info.get("label", label),
               "area": area, "distance_m": round(float(dist), 3),
               "direct_goal_anchor": bool(direct), "score": round(score, 4)}
        evidence_by_area.setdefault(area, []).append(rec)
    if not evidence_by_area:
        return {"status": "ambiguous", "area": "", "reason": "no_local_anchor",
                "topology_room_id": int(query_room), "anchors": [],
                "area_scores": {}}
    area_scores = {
        area: sum(item["score"] for item in evidence)
        for area, evidence in evidence_by_area.items()
    }
    ranked = sorted(area_scores.items(), key=lambda item: item[1], reverse=True)
    winner_area, winner_score = ranked[0]
    runner_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = winner_score - runner_score
    ratio = winner_score / max(runner_score, 1e-9) if runner_score else float("inf")
    winner_evidence = sorted(evidence_by_area[winner_area],
                             key=lambda item: item["score"], reverse=True)
    all_evidence = sorted(
        (item for values in evidence_by_area.values() for item in values),
        key=lambda item: item["score"], reverse=True)
    confidence = {
        "winner_score": round(winner_score, 4),
        "runner_up_score": round(runner_score, 4),
        "score_margin": round(margin, 4),
        "score_ratio": None if not math.isfinite(ratio) else round(ratio, 4),
    }
    if (winner_score < AREA_MIN_SCORE
            or (runner_score > 0 and (margin < AREA_MIN_SCORE_MARGIN
                                      or ratio < AREA_MIN_SCORE_RATIO))):
        return {"status": "ambiguous", "area": "",
                "reason": "competing_area_evidence",
                "topology_room_id": int(query_room),
                "anchors": all_evidence[:4],
                "area_scores": {k: round(v, 4) for k, v in ranked},
                "confidence": confidence}
    return {"status": "verified", "area": winner_area,
            "reason": "local_evidence_consensus",
            "topology_room_id": int(query_room),
            "anchors": winner_evidence[:4],
            "area_scores": {k: round(v, 4) for k, v in ranked},
            "confidence": confidence}


# labels that never define a room's function (ceiling lights, wall decor,
# generic openings). Excluded from room-naming so a room with 8 downlights
# and 1 bed is still "bedroom", not "downlights area".
SKIP_FOR_NAMING = {
    "downlight", "chandelier", "light", "lamp", "ceiling_light", "strip_light",
    "floor_lamp", "table_lamp", "wall_lamp", "pendant_lamp", "lampshade",
    "painting", "decorative_painting", "ornament", "decoration", "vase",
    "plant", "flowers", "picture", "photo", "mirror", "window", "door",
    "interior_doors", "door_curtain", "curtain", "carpet", "rug",
    "downlights", "spotlight", "spotlights", "tuyere", "wall_panels",
    "other", "strip_light",
}

# These labels may be physically close to the route, but they are poor verbal
# landmarks: they are structural, repeated many times, or semantically empty.
LANDMARK_SKIP = SKIP_FOR_NAMING | {
    "ceiling", "floor", "wall", "baseboard", "air_outlet", "switch",
    "socket", "skirting", "unknown", "plaster_line", "ceiling_lamp",
    "wall_mounted_air_conditioner", "air_conditioner",
    "vertical_air_conditioning", "bath_heater",
}


def _room_names(room_lab: Optional[np.ndarray], occ,
                label_index: Dict[str, dict]) -> Dict[str, str]:
    """room_id (str) -> friendly name via dominant object class in that room.

    An object "belongs" to a room when its bbox centre pixel lands in that
    room_id's region. The room is named by the majority *mapped* room-class of
    its objects; if no class maps, fall back to the majority raw label + "
    area" (e.g. "wardrobe area"); rooms with no objects become "room N".
    Ceiling lights / wall decor are skipped (SKIP_FOR_NAMING) so they don't
    outvote real furniture.
    """
    if room_lab is None:
        return {}
    mapped: Dict[int, Dict[str, int]] = {}     # room_id -> {room_class: count}
    rawlab: Dict[int, Dict[str, int]] = {}      # room_id -> {raw_label_norm: count}
    for label_N, info in label_index.items():
        nl = _normalize(info.get("label", ""))
        if nl in SKIP_FOR_NAMING:
            continue
        cxy = info.get("centre_xy")
        if cxy is None:
            continue
        col, row = occ.to_pixel(cxy[0], cxy[1])
        if not (0 <= row < room_lab.shape[0] and 0 <= col < room_lab.shape[1]):
            continue
        rid = int(room_lab[row, col])
        if rid == 0:
            continue
        rclass = LABEL_TO_ROOM.get(nl)
        if rclass is not None:
            mapped.setdefault(rid, {})
            mapped[rid][rclass] = mapped[rid].get(rclass, 0) + 1
        else:
            rawlab.setdefault(rid, {})
            rawlab[rid][nl] = rawlab[rid].get(nl, 0) + 1
    out: Dict[str, str] = {}
    all_rids = set(mapped) | set(rawlab)
    for rid in all_rids:
        if rid in mapped and mapped[rid]:
            out[str(rid)] = max(mapped[rid].items(), key=lambda kv: kv[1])[0]
        elif rid in rawlab and rawlab[rid]:
            top = max(rawlab[rid].items(), key=lambda kv: kv[1])[0]
            out[str(rid)] = "%s area" % top.replace("_", " ")
    return out


def _bresenham_pixels(occ, x0: float, y0: float, x1: float, y1: float):
    """Yield (row, col) along the segment in the occupancy grid."""
    c0, r0 = occ.to_pixel(x0, y0)
    c1, r1 = occ.to_pixel(x1, y1)
    dr = abs(r1 - r0); dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    r, c = r0, c0
    H, W = occ.h, occ.w
    while True:
        if 0 <= r < H and 0 <= c < W:
            yield r, c
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr


def _doors_crossed(occ, waypoints) -> int:
    """Count distinct door_opening segments the polyline touches.

    A door is "crossed" when the path enters a door pixel. Multiple consecutive
    door pixels along the path count as one crossing (a single doorway), so we
    tally transitions False->True on the door mask along the rasterised path.
    """
    door = getattr(occ, "_usd_door_opening", None)
    if door is None:
        return 0
    crossed = 0
    in_door = False
    for i in range(len(waypoints) - 1):
        x0, y0 = waypoints[i][:2]
        x1, y1 = waypoints[i + 1][:2]
        for r, c in _bresenham_pixels(occ, x0, y0, x1, y1):
            on = bool(door[r, c])
            if on and not in_door:
                crossed += 1
            in_door = on
    return crossed


def _turns_from_waypoints(waypoints) -> List[str]:
    """Per interior waypoint, quantise the heading change to left/right/straight.

    Heading at waypoint i is the tangent towards i+1; the turn at i is the
    signed Δyaw from the previous segment. Only interior waypoints have a
    defined turn (first has no prior segment, last no next).
    """
    if len(waypoints) < 3:
        return []
    yaws = []
    for i in range(len(waypoints) - 1):
        dx = waypoints[i + 1][0] - waypoints[i][0]
        dy = waypoints[i + 1][1] - waypoints[i][1]
        yaws.append(math.atan2(dy, dx))
    turns = []
    for i in range(1, len(yaws)):
        d = yaws[i] - yaws[i - 1]
        d = math.atan2(math.sin(d), math.cos(d))      # wrap to [-pi, pi]
        if d > TURN_THRESH:
            turns.append("left")
        elif d < -TURN_THRESH:
            turns.append("right")
        else:
            turns.append("straight")
    # collapse consecutive duplicates so the verbalize prompt sees the
    # meaningful turn sequence (e.g. [left, straight, straight, right] ->
    # [left, straight, right]) -- straight segments between turns stay as one.
    dedup: List[str] = []
    for t in turns:
        if not dedup or dedup[-1] != t:
            dedup.append(t)
    # drop a leading/trailing straight (no turn happened before/after)
    if dedup and dedup[0] == "straight":
        dedup = dedup[1:]
    if dedup and dedup[-1] == "straight":
        dedup = dedup[:-1]
    return dedup


def _path_samples(waypoints) -> List[Tuple[float, float, float]]:
    """Dense ``(x, y, arc_m)`` samples along a polyline."""
    samples: List[Tuple[float, float, float]] = []
    arc = 0.0
    for i in range(len(waypoints) - 1):
        x0, y0 = waypoints[i][:2]
        x1, y1 = waypoints[i + 1][:2]
        seg = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(seg / SAMPLE_STEP_M) + 1)
        for k in range(n):
            t = k / float(n)
            samples.append((x0 + (x1 - x0) * t,
                            y0 + (y1 - y0) * t,
                            arc + seg * t))
        arc += seg
    samples.append((float(waypoints[-1][0]), float(waypoints[-1][1]), arc))
    return samples


def _point_bbox_distance(x: float, y: float, bb) -> float:
    x0, y0, x1, y1 = bb
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return math.hypot(dx, dy)


def _visible_to_bbox(occ, x: float, y: float, bb) -> bool:
    """LOS from a route sample to 8 cm before the nearest bbox boundary."""
    x0, y0, x1, y1 = bb
    tx = min(max(x, x0), x1)
    ty = min(max(y, y0), y1)
    dx, dy = tx - x, ty - y
    dist = math.hypot(dx, dy)
    if dist <= 1e-6:
        return True
    keep = max(0.0, dist - 0.08) / dist
    return bool(occ.seg_free(x, y, x + dx * keep, y + dy * keep, 0.0))


def _object_hits(occ, waypoints, label_index: Dict[str, dict]) -> list:
    """Route-local objects sorted by first/closest encounter arc length."""
    samples = _path_samples(waypoints)
    hits = []
    for label_N, info in label_index.items():
        bb = info.get("bbox_xyxy")
        if bb is None:
            continue
        best = min(samples, key=lambda p: _point_bbox_distance(p[0], p[1], bb))
        distance = _point_bbox_distance(best[0], best[1], bb)
        if distance <= NEARBY_M:
            hits.append({"object_id": label_N,
                         "label": info.get("label", label_N),
                         "arc_m": float(best[2]),
                         "min_path_distance_m": float(distance),
                         "sample_xy": (float(best[0]), float(best[1])),
                         "visible_from_path": _visible_to_bbox(
                             occ, float(best[0]), float(best[1]), bb)})
    hits.sort(key=lambda hit: (hit["arc_m"], hit["min_path_distance_m"]))
    return hits


def _nearby_objects(occ, waypoints, label_index: Dict[str, dict]) -> List[str]:
    return [hit["object_id"] for hit in _object_hits(occ, waypoints, label_index)]


def _landmarks(occ, waypoints, label_index: Dict[str, dict],
               room_lab: Optional[np.ndarray], room_names: Dict[str, str],
               start_object: str, goal_object: str,
               max_landmarks: int = 5, min_arc_gap_m: float = 0.75) -> list:
    """Ordered, visible, semantically useful objects passed by the route."""
    out = []
    used_labels = set()
    last_arc = -1e9
    samples = _path_samples(waypoints)
    total_arc = samples[-1][2] if samples else 0.0
    goal_label = _normalize(label_index.get(goal_object, {}).get("label", ""))
    for hit in _object_hits(occ, waypoints, label_index):
        object_id = hit["object_id"]
        label = _normalize(hit["label"])
        # Start/goal-local objects were not actually "passed".  Also suppress
        # same-class alternatives to the goal (e.g. chair_3 before chair_8),
        # because natural-language labels cannot distinguish those instances.
        if (hit["arc_m"] < min_arc_gap_m
                or hit["arc_m"] > total_arc - min_arc_gap_m
                or object_id in (start_object, goal_object)
                or label == goal_label or label in LANDMARK_SKIP):
            continue
        if not hit["visible_from_path"] or label in used_labels:
            continue
        if hit["arc_m"] - last_arc < min_arc_gap_m:
            continue
        sx, sy = hit.pop("sample_xy")
        room_id = _room_at(room_lab, occ, sx, sy)
        hit["room_id"] = int(room_id)
        hit["room"] = room_names.get(str(room_id),
                                      "room %d" % room_id if room_id else "an open area")
        hit["arc_m"] = round(hit["arc_m"], 3)
        hit["min_path_distance_m"] = round(hit["min_path_distance_m"], 3)
        out.append(hit)
        used_labels.add(label)
        last_arc = float(hit["arc_m"])
        if len(out) >= max_landmarks:
            break
    return out


def _room_at(room_lab: Optional[np.ndarray], occ, x: float, y: float) -> int:
    if room_lab is None:
        return 0
    col, row = occ.to_pixel(x, y)
    if not (0 <= row < room_lab.shape[0] and 0 <= col < room_lab.shape[1]):
        return 0
    return int(room_lab[row, col])


def object_room_ids(occ, room_lab: Optional[np.ndarray], info: dict) -> set[int]:
    """Topology rooms overlapped by an object's floor footprint.

    Sampling the footprint instead of only its centre handles furniture against
    walls and door-adjacent annotations without any category or scene rules.
    """
    if room_lab is None or info.get("bbox_xyxy") is None:
        return set()
    x0, y0, x1, y1 = info["bbox_xyxy"]
    xs = np.linspace(float(x0), float(x1), 5)
    ys = np.linspace(float(y0), float(y1), 5)
    rooms = {_room_at(room_lab, occ, float(x), float(y)) for x in xs for y in ys}
    return {room for room in rooms if room > 0}


def _cross_rooms(room_lab, occ, waypoints, room_names: Dict[str, str]) -> List[str]:
    """Ordered, deduped room names along the (densely resampled) path.

    Walks the rasterised polyline, reads the room_id at each pixel, keeps the
    nonzero ones in first-encounter order, maps to friendly names.
    """
    if room_lab is None:
        return []
    seq: List[int] = []
    for i in range(len(waypoints) - 1):
        x0, y0 = waypoints[i][:2]
        x1, y1 = waypoints[i + 1][:2]
        for r, c in _bresenham_pixels(occ, x0, y0, x1, y1):
            rid = int(room_lab[r, c])
            if rid != 0 and (not seq or seq[-1] != rid):
                seq.append(rid)
    out: List[str] = []
    for rid in seq:
        name = room_names.get(str(rid), "room %d" % rid)
        if not out or out[-1] != name:
            out.append(name)
    return out


def extract(waypoints: List[Tuple[float, float]], occ, label_index: Dict[str, dict],
            scene: str, start_object: str = "", goal_object: str = "",
            turn_waypoints: Optional[List[Tuple[float, float]]] = None) -> dict:
    """Build the route dict for one A* golden path.

    ``waypoints`` are geometry-frame ``(x, y)`` (z omitted -- navigation is 2D)
    from :func:`sage.planner.plan_path`. ``label_index`` maps ``label_N`` to
    ``{"label", "bbox_xyxy", "centre_xy", "ins_id"}`` (built by the driver from
    labels.json). ``start_object`` / ``goal_object`` are the snapped label_N the
    driver already chose, echoed back for the prompt.
    """
    room_lab = _room_label_map(occ)
    room_names = _room_names(room_lab, occ, label_index)

    s_room_id = _room_at(room_lab, occ, waypoints[0][0], waypoints[0][1])
    g_room_id = _room_at(room_lab, occ, waypoints[-1][0], waypoints[-1][1])
    start_area = _functional_area(waypoints[0][:2], occ, label_index, room_lab,
                                  start_object)
    goal_area = _functional_area(waypoints[-1][:2], occ, label_index, room_lab,
                                 goal_object)
    start_room = (start_area["area"] if start_area["status"] == "verified"
                  else "topological area %d" % s_room_id if s_room_id else "an open area")
    goal_room = (goal_area["area"] if goal_area["status"] == "verified"
                 else "topological area %d" % g_room_id if g_room_id else "an open area")

    cross = _cross_rooms(room_lab, occ, waypoints, room_names)
    doors = _doors_crossed(occ, waypoints)
    # Execution waypoints are dense and smooth, so a real 60-degree corner is
    # split into many <30-degree deltas. Semantic turns must come from the raw
    # ECM control polyline, while rooms/doors/landmarks follow the path executed.
    turns = _turns_from_waypoints(turn_waypoints or waypoints)
    nearby = _nearby_objects(occ, waypoints, label_index)
    landmarks = _landmarks(occ, waypoints, label_index, room_lab, room_names,
                           start_object, goal_object)

    length = 0.0
    for i in range(len(waypoints) - 1):
        length += math.hypot(waypoints[i + 1][0] - waypoints[i][0],
                             waypoints[i + 1][1] - waypoints[i][1])

    return {
        "scene": scene,
        "start": {"object": start_object, "room": start_room},
        "goal": {"object": goal_object, "room": goal_room},
        "cross_rooms": cross,
        "doors_crossed": doors,
        "turns": turns,
        "nearby_objects": nearby,
        "landmarks": landmarks,
        "path_length_m": round(length, 2),
        "room_names": room_names,
        "start_area_grounding": start_area,
        "goal_area_grounding": goal_area,
        # raw room ids for the grounding validator (names are LLM-facing aliases)
        "start_room_id": s_room_id,
        "goal_room_id": g_room_id,
    }
