"""MP3D semantic geometry for the request-grounded trajectory planner.

Only the scene's ASCII 1.1 ``.house`` annotations are read.  Episode goals,
reference routes and panorama sequences are deliberately outside this API.
The existing Habitat parser supplies the C/O/R field convention, but discards
S/V floor polygons; these are preserved here for ``SemanticScene`` reuse.
House XYZ is transformed to the action expert world as (Y, -X, Z).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import os
from pathlib import Path

import numpy as np


DEFAULT_MP3D_ROOT = Path(os.environ.get(
    "MP3D_HOUSE_ROOT", "/data/shared/groups/mfm/common/anno/vlmevalkit/mp3d"))

# Native labels from Habitat's Mp3dSemanticScene.cpp, including uncommon codes.
ROOM_CODES = {
    "a": "bathroom", "b": "bedroom", "c": "closet", "d": "dining room",
    "e": "entryway/foyer/lobby", "f": "familyroom/lounge", "g": "garage",
    "h": "hallway", "i": "library", "j": "laundryroom/mudroom",
    "k": "kitchen", "l": "living room", "m": "meetingroom/conferenceroom",
    "n": "lounge", "o": "office", "p": "porch/terrace/deck",
    "r": "rec/game", "s": "stairs", "t": "toilet", "u": "utilityroom/toolroom",
    "v": "tv", "w": "workout/gym/exercise", "x": "outdoor", "y": "balcony",
    "z": "other room", "B": "bar", "C": "classroom", "D": "dining booth",
    "S": "spa/sauna", "Z": "junk",
}

# Only unambiguous native room labels establish these existing teacher areas.
# Objects inside a labelled bedroom cannot relabel it as a bathroom.
NATIVE_FUNCTIONAL_AREAS = {
    "a": {"bathroom area"}, "b": {"sleeping area"},
    "d": {"dining area", "common area"},
    "k": {"kitchen area", "common area"},
    "l": {"living area", "common area"},
}


def house_to_world_xyz(points):
    """Transform one XYZ or an array of XYZ points without changing heights."""
    values = np.asarray(points, dtype=float)
    if values.ndim == 0 or values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError("house coordinates must be finite XYZ points")
    return np.stack((values[..., 1], -values[..., 0], values[..., 2]), axis=-1)


def _vector(tokens, offset):
    result = np.asarray(tokens[offset:offset + 3], dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("invalid XYZ field")
    return result


def _bounds_world(tokens, offset):
    lower, upper = _vector(tokens, offset), _vector(tokens, offset + 3)
    if np.any(upper < lower):
        raise ValueError("invalid bounding-box bounds")
    corners = house_to_world_xyz([lower, upper])
    return corners.min(axis=0).tolist(), corners.max(axis=0).tolist()


def native_size_evidence(oid, objects, spec, names, normalize, floor_height):
    """Resolve a relative size adjective using independent native OBBs.

    A unique object must exceed every same-family floor-supported peer in
    both upright horizontal dimensions. Singletons, ties, missing geometry
    and objects without a witnessed floor cannot establish relative size.
    """
    constraint = spec.get('native_size_constraint')
    if not constraint:
        return {}
    if constraint.get('adjective') != 'large' or floor_height is None:
        return None
    ratio = 1.25
    inventory = []
    for candidate_id, obj in objects.items():
        if names.get(normalize(obj['label'])) != spec['family']:
            continue
        obb = obj.get('obb_world', {})
        axes = np.asarray(obb.get('axes', []), float)
        half = np.asarray(obb.get('half_lengths', []), float)
        if axes.shape != (3, 3) or half.shape != (3,) or not np.isfinite(axes).all() or not np.isfinite(half).all():
            return None
        vertical = np.where(np.abs(axes[:, 2]) >= .98)[0]
        if len(vertical) != 1 or abs(float(obj['z_range'][0]) - float(floor_height)) > .25:
            return None
        horizontal = [i for i in range(3) if i != vertical[0]]
        if np.max(np.abs(axes[horizontal, 2])) > .02 or np.any(half <= 0):
            return None
        inventory.append(dict(id=candidate_id, native_object_id=obj.get('native_object_id'),
                              horizontal_lengths_m=sorted((2 * half[horizontal]).tolist())))
    candidate = next((item for item in inventory if item['id'] == oid), None)
    if candidate is None or len(inventory) < 2:
        return None
    lengths = np.asarray(candidate['horizontal_lengths_m'])
    peers = [item for item in inventory if item['id'] != oid]
    if any(np.any(lengths < ratio * np.asarray(peer['horizontal_lengths_m'])) for peer in peers):
        return None
    return dict(native_size=dict(**constraint, required_linear_ratio=ratio, candidate=candidate,
        compared_inventory=inventory, floor_height_m=float(floor_height),
        evidence='native upright OBB full horizontal lengths, bottom near current annotated/navmesh floor'))


@dataclass
class Mp3dTeacherMap:
    """Scene geometry; all instance and region IDs retain native provenance."""

    source_path: str
    objects: dict
    regions: dict
    structure: dict
    openings: dict
    metadata: dict

    def _room_indices_for_floor(self, floor_height, tolerance=.25):
        height, tolerance = float(floor_height), float(tolerance)
        if not np.isfinite(height) or not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("floor height and tolerance must be finite; tolerance >= 0")
        # A polygon spanning a large height range cannot establish membership
        # in one horizontal floor. Keep both bounds near the actual navmesh
        # floor, rather than using a room centre or a goal-selected level.
        return [i for i, room in enumerate(self.structure["rooms"])
                if max(abs(z - height) for z in room["floor_z_range"]) <= tolerance]

    def objects_for_floor(self, floor_height, tolerance=.25):
        """Filter by native region floor, retaining the raw .objects catalogue.

        An object's height/OBB alone does not imply floor membership (walls
        and stairs can span floors). Unknown native region membership stays
        excluded when a floor is supplied.
        """
        room_indices = self._room_indices_for_floor(floor_height, tolerance)
        region_ids = {self.structure["rooms"][i]["native_region_id"]
                      for i in room_indices}
        return {oid: obj for oid, obj in self.objects.items()
                if obj["native_region_id"] in region_ids}

    def native_room_evidence(self, room_id, requested):
        """Use the native region label only after a floor polygon was selected."""
        rooms = self.structure["rooms"]
        if room_id is None or not isinstance(room_id, (int, np.integer)):
            return None
        if not 0 <= room_id < len(rooms):
            return None
        room = rooms[room_id]
        code = room["native_room_code"]
        # Exact native labels and the existing compiler's conservative areas.
        accepted = {room["native_room_name"], *NATIVE_FUNCTIONAL_AREAS.get(code, ())}
        if requested not in accepted or code in {"z", "Z"}:
            return None
        return dict(
            room_id=int(room_id), requested=requested,
            native_region_id=room["native_region_id"], native_room_code=code,
            native_room_name=room["native_room_name"],
            native_surface_id=room["native_surface_id"],
            level_id=room["level_id"], floor_z_range=room["floor_z_range"],
            source_path=self.source_path,
            evidence="native MP3D region label and annotated floor polygon")

    def make_semantic_scene(self, occ=None, floor_height=None, floor_tolerance=.25):
        """Reuse the existing semantic guards while retaining native room labels.

        The actual geometry adapter's ``occ.floor_height`` selects the floor.
        Remaining XY overlaps on that floor stay ambiguous in room_at. With
        no supplied floor, all floors remain visible and overlap is rejected;
        no floor is picked using the episode goal or nearest room centre.
        """
        from rule_semantics import SemanticScene

        scene_map = self
        if floor_height is None and occ is not None:
            floor_height = getattr(occ, "floor_height", None)
        if floor_height is None:
            room_indices = list(range(len(self.structure["rooms"])))
            objects = self.objects
        else:
            room_indices = self._room_indices_for_floor(floor_height, floor_tolerance)
            objects = self.objects_for_floor(floor_height, floor_tolerance)
        structure = dict(self.structure, rooms=[self.structure["rooms"][i]
                                               for i in room_indices])

        class NativeMp3dSemanticScene(SemanticScene):
            def portal_sides(self, opening, pose, direction=None):
                near, far, near_xy, far_xy = super().portal_sides(opening, pose, direction)
                distance = opening.get('native_evidence', {}).get('crossing_offset_m', .55)
                if distance == .55:
                    return near, far, near_xy, far_xy
                center = np.asarray(opening['center'], float)
                normal = (np.asarray(far_xy) - center) / .55
                near_xy, far_xy = center - distance * normal, center + distance * normal
                return self.room_at(near_xy), self.room_at(far_xy), near_xy, far_xy

            def candidate_evidence(self, oid, obj, spec, pose, names, normalize, **kwargs):
                size = native_size_evidence(oid, self.objects, spec, names, normalize, floor_height)
                if size is None:
                    return None
                evidence = super().candidate_evidence(oid, obj, spec, pose, names, normalize, **kwargs)
                if evidence is not None:
                    evidence.update(size)
                if evidence is not None and spec.get('native_target_adjacent_room'):
                    source = self.room_at(pose[:2])
                    target = self.room_at(obj['centre_xy'])
                    if source is None or target is None or source == target or oid in self.openings:
                        return None
                    portals = []
                    for opening_id, opening in self.openings.items():
                        native = opening.get('native_evidence', {})
                        if (native.get('schema') != 'mp3d_native_doorframe_aperture_v1'
                                or floor_height is None
                                or abs(float(native.get('floor_height', float('inf'))) - floor_height) > .06):
                            continue
                        near, far, _, _ = self.portal_sides(opening, pose)
                        if near == source and far == target:
                            portals.append(opening_id)
                    if len(portals) != 1:
                        return None
                    evidence['native_target_adjacent_room'] = dict(
                        spec['native_target_adjacent_room'], source_room=source,
                        target_room=target, verified_portal_ids=portals,
                        evidence='original object room across a unique same-floor native verified aperture; object relations retained')
                if evidence is not None and oid in self.openings:
                    destination = spec.get('native_original_destination')
                    if destination:
                        far = evidence['portal_connection']['destination_room']
                        references = [dict(id=ref_id, native_object_id=ref.get('native_object_id'))
                                      for ref_id, ref in self.objects.items()
                                      if names.get(normalize(ref['label'])) == destination.get('reference_family')
                                      and self.room_at(ref['centre_xy']) == far
                                      and floor_height is not None
                                      and abs(float(ref['z_range'][0]) - floor_height) <= .25]
                        if destination.get('reference_family') and not references:
                            return None
                        evidence['native_original_destination'] = dict(destination, room_id=far,
                            references=references,
                            evidence='annotated far room and same-floor native object inventory; original target retained')
                    native = self.openings[oid].get('native_evidence')
                    if spec.get('native_double_door_group'):
                        group = (native or {}).get('double_door_group')
                        if not group or len(group.get('leaves', [])) < 2:
                            return None
                        evidence['native_double_door_group'] = group
                    if native is not None:
                        evidence['native_aperture'] = native
                return evidence

            def room_evidence(self, rid, requested, xy=None):
                if rid is None or not isinstance(rid, (int, np.integer)) or not (
                        0 <= rid < len(room_indices)):
                    return None
                evidence = scene_map.native_room_evidence(room_indices[rid], requested)
                if evidence is not None:
                    return dict(evidence, room_id=int(rid))
                # A known native label must not be overridden by a furniture
                # heuristic. Unknown room function can still use existing
                # explicit-anchor evidence; "junk" can never support a room.
                code = structure["rooms"][rid]["native_room_code"]
                if code != "z":
                    return None
                return super().room_evidence(rid, requested, xy)

        semantic = NativeMp3dSemanticScene(structure, objects, self.openings, occ)
        semantic.native_map = self
        semantic.native_room_indices = room_indices
        semantic.floor_height = floor_height
        return semantic


def load_mp3d_house(path):
    """Load native C/O/R/S/V records, preserving exact floor-vertex order.

    Object radii in O records are half lengths (Habitat constructs an OBB
    with sizes ``2 * radius``); AABBs include all eight oriented corners.
    P records may describe panoramas or portals. They supply no verified
    navigable opening in this adapter and are never used to make doorways.
    """
    source = Path(path)
    categories, regions, object_records, surfaces, vertices = {}, {}, {}, {}, {}
    record_counts = {}
    with source.open(encoding="utf-8") as handle:
        if handle.readline().strip() != "ASCII 1.1":
            raise ValueError(f"unsupported MP3D house header: {source}")
        for line_number, line in enumerate(handle, 2):
            tokens = line.split()
            if not tokens:
                continue
            kind = tokens[0]
            record_counts[kind] = record_counts.get(kind, 0) + 1
            try:
                if kind == "C":
                    index = int(tokens[1])
                    if index in categories:
                        raise ValueError("duplicate category ID")
                    categories[index] = dict(
                        raw_index=int(tokens[2]),
                        raw_name=tokens[3].replace("#", " "),
                        mpcat40_index=int(tokens[4]), mpcat40_name=tokens[5])
                elif kind == "R":
                    index = int(tokens[1])
                    if index in regions:
                        raise ValueError("duplicate region ID")
                    lower, upper = _bounds_world(tokens, 9)
                    code = tokens[5]
                    regions[index] = dict(
                        native_region_id=index, level_id=int(tokens[2]),
                        native_room_code=code,
                        native_room_name=ROOM_CODES.get(code, "unknown"),
                        centre_xyz=house_to_world_xyz(_vector(tokens, 6)).tolist(),
                        bbox_xyxy=[*lower[:2], *upper[:2]],
                        z_range=[lower[2], upper[2]], surface_ids=[])
                elif kind == "O":
                    index = int(tokens[1])
                    if index in object_records:
                        raise ValueError("duplicate object ID")
                    center = _vector(tokens, 4)
                    first, second = _vector(tokens, 7), _vector(tokens, 10)
                    radii = _vector(tokens, 13)
                    axes = np.asarray([first, second, np.cross(first, second)])
                    if np.any(radii < 0) or not np.allclose(
                            axes @ axes.T, np.eye(3), atol=2e-3, rtol=0):
                        raise ValueError("invalid object OBB axes or half lengths")
                    corners = house_to_world_xyz([
                        center + (np.asarray(signs) * radii) @ axes
                        for signs in product((-1, 1), repeat=3)])
                    lower, upper = corners.min(axis=0), corners.max(axis=0)
                    object_records[index] = dict(
                        label="unknown", ins_id=str(index), native_object_id=index,
                        native_region_id=int(tokens[2]), category_id=int(tokens[3]),
                        centre_xy=house_to_world_xyz(center)[:2].tolist(),
                        bbox_xyxy=[*lower[:2].tolist(), *upper[:2].tolist()],
                        z_range=[float(lower[2]), float(upper[2])],
                        obb_world=dict(
                            centre_xyz=house_to_world_xyz(center).tolist(),
                            axes=house_to_world_xyz(axes).tolist(),
                            half_lengths=radii.tolist(), corners=corners.tolist()))
                elif kind == "S":
                    index = int(tokens[1])
                    if index in surfaces:
                        raise ValueError("duplicate surface ID")
                    surfaces[index] = dict(
                        region_id=int(tokens[2]), label=tokens[4],
                        normal=house_to_world_xyz(_vector(tokens, 8)).tolist())
                elif kind == "V":
                    index = int(tokens[1])
                    if index in vertices:
                        raise ValueError("duplicate vertex ID")
                    vertices[index] = (int(tokens[2]),
                                       house_to_world_xyz(_vector(tokens, 4)))
            except (ValueError, IndexError) as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error

    objects = {}
    for index, obj in sorted(object_records.items()):
        category = categories.get(obj["category_id"])
        if obj["category_id"] >= 0 and category is None:
            raise ValueError(f"{source}: missing object category {obj['category_id']}")
        if obj["native_region_id"] >= 0 and obj["native_region_id"] not in regions:
            raise ValueError(f"{source}: missing object region {obj['native_region_id']}")
        obj["category"] = category
        obj["label"] = category["raw_name"] if category else "unknown"
        objects[f"object_{index}"] = obj

    surface_points = {}
    for _, (surface_id, position) in sorted(vertices.items()):
        if surface_id not in surfaces:
            raise ValueError(f"{source}: missing vertex surface {surface_id}")
        surface_points.setdefault(surface_id, []).append(position)
    rooms = []
    for surface_id, surface in sorted(surfaces.items()):
        region_id = surface["region_id"]
        if region_id not in regions:
            raise ValueError(f"{source}: missing surface region {region_id}")
        regions[region_id]["surface_ids"].append(surface_id)
        if surface["label"] != "F":
            continue
        points = np.asarray(surface_points.get(surface_id, []), dtype=float)
        if len(points) < 3:
            raise ValueError(f"{source}: floor {surface_id} needs at least three vertices")
        if np.allclose(points[0], points[-1], atol=1e-9, rtol=0):
            points = points[:-1]
        xy = points[:, :2]
        twice_area = float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1)
                                  - xy[:, 1] * np.roll(xy[:, 0], -1)))
        if len(points) < 3 or abs(twice_area) < 1e-8:
            raise ValueError(f"{source}: degenerate floor polygon {surface_id}")
        region = regions[region_id]
        rooms.append(dict(
            profile=xy.tolist(), native_region_id=region_id,
            native_surface_id=surface_id, native_room_code=region["native_room_code"],
            native_room_name=region["native_room_name"], level_id=region["level_id"],
            floor_z_range=[float(points[:, 2].min()), float(points[:, 2].max())]))

    return Mp3dTeacherMap(
        source_path=str(source.resolve()), objects=objects, regions=regions,
        structure=dict(rooms=rooms, holes=[], walls=[]), openings={},
        metadata=dict(schema="mp3d-house-teacher-v1", record_counts=record_counts,
                      categories=categories, coordinate_frame="world=(house_y,-house_x,house_z)",
                      opening_evidence="none; P records never establish navigable portals"))


def load_mp3d_scene(scene, root=DEFAULT_MP3D_ROOT):
    """Resolve one native scene without accepting a path as the scene ID."""
    scene = str(scene)
    if not scene or scene in {".", ".."} or "/" in scene or "\\" in scene:
        raise ValueError("MP3D scene must be a single scene identifier")
    return load_mp3d_house(Path(root) / scene / (scene + ".house"))
