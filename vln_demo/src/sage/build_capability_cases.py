"""Compile auditable capability cases from SAGE golden trajectories.

This is an experimental branch beside ``build_demo_episodes``.  It never edits
the canonical GVLN JSON.  It writes a manifest, sampled route frames, a contact
sheet/video, and optional Qwen-generated instructions whose claims are checked
against the structured route before they are admitted.
"""
from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import math
import os
import re
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_DEFAULT_SAGE_ROOT = "/data/zerovln/SAGE-3D"
os.environ.setdefault("SAGE_COLLISION", "usd_rooms")
os.environ.setdefault("SAGE_USD_COLLISION_MASK_ROOT",
                      os.path.join(_DEFAULT_SAGE_ROOT, "sage_usd_rooms_masks"))

from .capability_benchmark import (CAPABILITIES, SCHEMA_VERSION,
                                   SEMANTIC_SUCCESS_CONTRACT, V3_SCHEMA_VERSION,
                                   eligible_dimensions, project_to_path,
                                   trajectory_features)
from .datasets import _to_geometry_frame, yaw_from_wxyz
from .gs_client import XYZEulerClient
from .occmap import OccMap
from . import route_extractor
from .visibility_audit import depth_support, project_bbox
from rs3d.gs_client import CameraSpec


DEFAULT_ROOT = _DEFAULT_SAGE_ROOT
DEFAULT_OUT = "/data/zerovln/capability_benchmark/v1"

# Route-frame sampling is deliberately expressed in physical/angular units.
# Straight segments receive one frame per 0.8 m, while turns receive an extra
# frame per accumulated 15 degrees.  The frame cap is a hard preflight budget,
# not a target: Qwen3.6 serves a 131072-token context and vLLM accepts at most
# 64 images.  Reserving 32768 tokens for route JSON, chat framing and output,
# and charging a conservative 2048 tokens for each fixed 640x480 frame yields
# a 48-frame ceiling.  Every sampling branch, including landmark overflow,
# must stay within this ceiling.
FRAME_DISTANCE_STEP_M = 0.8
FRAME_TURN_STEP_RAD = math.radians(15.0)
QWEN36_CONTEXT_TOKENS = 131072
QWEN36_NON_IMAGE_RESERVE_TOKENS = 32768
QWEN36_IMAGE_TOKEN_BUDGET = 2048
QWEN36_VLLM_IMAGE_LIMIT = 64
QWEN36_CHAT_FRAMING_RESERVE_TOKENS = 4096
MAX_ROUTE_FRAMES = min(
    QWEN36_VLLM_IMAGE_LIMIT,
    (QWEN36_CONTEXT_TOKENS - QWEN36_NON_IMAGE_RESERVE_TOKENS)
    // QWEN36_IMAGE_TOKEN_BUDGET,
)


def _code_provenance() -> dict:
    """Record the exact implementation state without requiring a clean tree."""
    repo = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"repository": str(repo), "git_commit": commit, "git_dirty": dirty,
            "builder": str(Path(__file__).resolve())}


def _json_object(text: str) -> dict:
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text,
                  flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        value = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("model did not return JSON: " + repr(text[:500]))
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("model JSON must be an object")
    return value


def _preflight_content_budget(content: list, max_tokens: int) -> dict:
    """Fail before VLM submission if the conservative context budget is exceeded.

    UTF-8 byte count is used as an upper bound for prompt text tokens.  Images
    are fixed 640x480 route renders and charged at the deliberately generous
    ``QWEN36_IMAGE_TOKEN_BUDGET``.  This complements the sampler cap and also
    protects unusually large ROUTE_JSON prompts.
    """
    images = sum(1 for part in content if part.get("type") == "image_url")
    text_bytes = sum(len(str(part.get("text") or "").encode("utf-8"))
                     for part in content if part.get("type") == "text")
    non_image = text_bytes + int(max_tokens) + QWEN36_CHAT_FRAMING_RESERVE_TOKENS
    conservative_total = non_image + images * QWEN36_IMAGE_TOKEN_BUDGET
    if images > QWEN36_VLLM_IMAGE_LIMIT:
        raise ValueError("VLM image count exceeds vLLM per-prompt limit")
    if images > MAX_ROUTE_FRAMES:
        raise ValueError("VLM image count exceeds conservative route-frame budget")
    if non_image > QWEN36_NON_IMAGE_RESERVE_TOKENS:
        raise ValueError("VLM text/output exceeds reserved non-image context budget")
    if conservative_total > QWEN36_CONTEXT_TOKENS:
        raise ValueError("VLM request exceeds conservative Qwen3.6 context budget")
    return {"images": images, "text_bytes_upper_bound": text_bytes,
            "max_output_tokens": int(max_tokens),
            "conservative_total_tokens": conservative_total}


def _chat(url: str, model: str, content: list, max_tokens=4096) -> tuple[dict, str]:
    _preflight_content_budget(content, max_tokens)
    payload = {"model": model, "messages": [{"role": "user", "content": content}],
               "temperature": 0.2, "top_p": 0.8, "max_tokens": max_tokens,
               "response_format": {"type": "json_object"},
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        message = json.load(resp)["choices"][0]["message"]
        answer = (message.get("reasoning_content") or "") + (message.get("content") or "")
    return _json_object(answer), answer


def _image_part(image: Image.Image) -> dict:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "JPEG", quality=88)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + data}}


def _geometry_points(sample: dict):
    points, yaws = [], []
    for point in sample.get("points", []):
        pos = point["position"]
        x, y, yaw = _to_geometry_frame(float(pos[0]), float(pos[1]),
                                        yaw_from_wxyz(point["rotation"]))
        points.append([x, y])
        yaws.append(yaw)
    return points, yaws


def _scene_dir(root: str, bare: str) -> str:
    matches = [x for x in Path(root, "InteriorGS").iterdir()
               if x.is_dir() and x.name.split("_")[-1] == bare]
    if not matches:
        raise FileNotFoundError("InteriorGS scene not found: " + bare)
    return str(matches[0])


def _nearest_progress_index(progress: np.ndarray, target: float) -> int:
    """Return the path index closest to a monotonic progress target."""
    right = int(np.searchsorted(progress, target, side="left"))
    if right <= 0:
        return 0
    if right >= len(progress):
        return len(progress) - 1
    left = right - 1
    return left if target - progress[left] <= progress[right] - target else right


def _adaptive_sample_indices(points, yaws, min_frames: int,
                             mandatory_indices=(),
                             distance_step_m: float = FRAME_DISTANCE_STEP_M,
                             turn_step_rad: float = FRAME_TURN_STEP_RAD,
                             max_frames: int = MAX_ROUTE_FRAMES) -> list[int]:
    """Sample more densely where the camera accumulates a larger yaw change.

    The progress measure combines travelled arc length and wrapped yaw change::

        ds / distance_step_m + abs(wrap_pi(dyaw)) / turn_step_rad

    Uniform samples on that progress axis therefore retain straight-route
    coverage while allocating extra frames to turns.  Start/end and caller
    supplied semantic evidence frames are prioritized.  Endpoints are always
    retained; if prioritized landmarks alone exceed ``max_frames``, they are
    thinned uniformly so the context ceiling remains a hard invariant.
    """
    count = len(points)
    if count != len(yaws):
        raise ValueError("points and yaws must have equal length")
    if count == 0:
        return []
    if distance_step_m <= 0 or turn_step_rad <= 0:
        raise ValueError("sampling distance and turn steps must be positive")
    if max_frames < 2:
        raise ValueError("frame budget must leave room for start and end")

    mandatory = {0, count - 1}
    mandatory.update(int(i) for i in mandatory_indices if 0 <= int(i) < count)
    frame_cap = min(count, int(max_frames))
    if len(mandatory) > frame_cap:
        # Context safety outranks semantic density.  Keep endpoints, then thin
        # landmark evidence uniformly along the route instead of overflowing
        # the model window.  This is only a pathological fallback: normal
        # routes contain far fewer landmarks than the 48-frame hard budget.
        endpoints = {0, count - 1}
        landmark_candidates = sorted(mandatory - endpoints)
        slots = frame_cap - len(endpoints)
        if slots > 0:
            positions = np.linspace(0, len(landmark_candidates) - 1, slots)
            mandatory = endpoints | {
                landmark_candidates[int(round(pos))] for pos in positions}
        else:
            mandatory = endpoints
    if count <= frame_cap and count <= max(max(0, int(min_frames)), len(mandatory)):
        return list(range(count))

    increments = []
    for i in range(1, count):
        ds = math.hypot(float(points[i][0]) - float(points[i - 1][0]),
                        float(points[i][1]) - float(points[i - 1][1]))
        dyaw = float(yaws[i]) - float(yaws[i - 1])
        wrapped = math.atan2(math.sin(dyaw), math.cos(dyaw))
        increments.append(ds / distance_step_m + abs(wrapped) / turn_step_rad)
    progress = np.concatenate(([0.0], np.cumsum(np.asarray(increments, dtype=float))))

    desired = max(max(2, int(min_frames)), int(math.ceil(float(progress[-1]))) + 1)
    desired = min(frame_cap, desired)
    if progress[-1] <= 1e-12:
        soft = {int(round(x)) for x in np.linspace(0, count - 1, desired)}
    else:
        targets = np.linspace(0.0, float(progress[-1]), desired)
        soft = {_nearest_progress_index(progress, float(target)) for target in targets}

    selected = mandatory | soft
    if len(selected) <= frame_cap:
        return sorted(selected)

    # The soft set is already distributed uniformly on the importance axis.
    # Thin it uniformly in that ordering while preserving all mandatory frames.
    optional = sorted(selected - mandatory)
    slots = max(0, frame_cap - len(mandatory))
    if slots == 0:
        return sorted(mandatory)
    keep_positions = np.linspace(0, len(optional) - 1, slots)
    kept_optional = {optional[int(round(pos))] for pos in keep_positions}
    return sorted(mandatory | kept_optional)


def _render_route(client, scene, occ, points, yaws, out_dir: Path, nframes: int,
                  route=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    landmark_indices = []
    if route:
        arcs = [0.0]
        for a, b in zip(points, points[1:]):
            arcs.append(arcs[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        for landmark in route.get("landmarks", []):
            target_arc = float(landmark.get("arc_m") or 0.0)
            landmark_indices.append(
                min(range(len(arcs)), key=lambda i: abs(arcs[i] - target_arc)))
    indices = _adaptive_sample_indices(points, yaws, nframes,
                                       mandatory_indices=landmark_indices)
    frames = []
    for order, idx in enumerate(indices):
        x, y = points[idx]
        z = float(occ.ground_z(x, y)) + 1.2
        image, _ = client.render_one(scene, (x, y, z), yaws[idx])
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((5, 5, 110, 29), fill=(0, 0, 0))
        draw.text((10, 9), "route %02d/%02d" % (order + 1, len(indices)), fill="white")
        image.save(out_dir / ("frame_%02d.jpg" % order), quality=90)
        frames.append(image)
    if frames:
        thumb_w, thumb_h = 320, 240
        sheet = Image.new("RGB", (thumb_w * 4, thumb_h * math.ceil(len(frames) / 4)), "black")
        for i, image in enumerate(frames):
            sheet.paste(image.resize((thumb_w, thumb_h)), ((i % 4) * thumb_w,
                                                          (i // 4) * thumb_h))
        sheet.save(out_dir / "contact_sheet.jpg", quality=90)
        try:
            import imageio.v2 as imageio
            imageio.mimsave(out_dir / "route.mp4", [np.asarray(x) for x in frames], fps=2)
        except Exception as exc:
            (out_dir / "video_warning.txt").write_text(str(exc), encoding="utf-8")
    return indices, frames


def _arc_index(points, target_arc: float) -> int:
    arcs = [0.0]
    for a, b in zip(points, points[1:]):
        arcs.append(arcs[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    return min(range(len(arcs)), key=lambda i: abs(arcs[i] - float(target_arc)))


def _audit_route_landmarks(renderer, scene: str, occ, points, yaws, route: dict,
                           labels: dict, camera, args) -> tuple[dict, list]:
    """Keep only landmarks supported by topology, camera projection and depth."""
    audited = copy.deepcopy(route)
    accepted, rejected = [], []
    room_lab = route_extractor._room_label_map(occ)
    for landmark in audited.get("landmarks") or []:
        object_id = landmark.get("object_id")
        info = labels.get(object_id)
        reason = None
        evidence = {"version": "topology-frustum-depth-v1"}
        if info is None or info.get("bbox_xyxy") is None or info.get("z_range") is None:
            reason = "missing_3d_annotation"
        else:
            idx = _arc_index(points, float(landmark.get("arc_m") or 0.0))
            path_room = route_extractor._room_at(room_lab, occ, points[idx][0], points[idx][1])
            object_rooms = route_extractor.object_room_ids(occ, room_lab, info)
            evidence.update({"path_index": idx, "path_room_id": path_room,
                             "object_room_ids": sorted(object_rooms)})
            if path_room <= 0 or path_room not in object_rooms:
                reason = "topology_room_mismatch"
            else:
                camera_pos = (points[idx][0], points[idx][1],
                              float(occ.ground_z(*points[idx])) + args.camera_height)
                projected = project_bbox(camera, camera_pos, yaws[idx], info)
                evidence["projection"] = projected
                if (projected is None
                        or projected["width_px"] < args.landmark_min_width
                        or projected["height_px"] < args.landmark_min_height
                        or projected["projected_area_ratio"] < args.landmark_min_area):
                    reason = "outside_or_weak_camera_projection"
                elif not args.render:
                    reason = "depth_audit_not_run"
                else:
                    _, depth = renderer.render_one(
                        "sage-" + scene, camera_pos, yaws[idx], want_depth=True)
                    support = depth_support(
                        depth, projected["bbox_px"], camera, camera_pos, yaws[idx], info,
                        args.landmark_depth_stride, args.landmark_depth_tolerance)
                    evidence["depth"] = {k: round(v, 6) if isinstance(v, float) else v
                                         for k, v in support.items()}
                    if (support["visible_fraction"] < args.landmark_min_depth_fraction
                            or support["visible_area_ratio"] < args.landmark_min_depth_area):
                        reason = "insufficient_metric_depth_support"
        landmark["evidence"] = evidence
        if reason is None:
            accepted.append(landmark)
        else:
            rejected.append({"object_id": object_id, "reason": reason,
                             "evidence": evidence})
    audited["landmarks"] = accepted
    audited["landmark_audit"] = {"accepted": len(accepted),
                                  "rejected": rejected,
                                  "contract": "topology-frustum-depth-v1"}
    return audited, rejected


def _route_facts(route: dict, label_index: dict) -> dict:
    ids = [route.get("start", {}).get("object"), route.get("goal", {}).get("object")]
    ids += [x.get("object_id") for x in route.get("landmarks", [])]
    labels = {oid: label_index.get(oid, {}).get("label", oid) for oid in ids if oid}
    return {"scene": route.get("scene"), "start": route.get("start"),
            "goal": route.get("goal"), "cross_rooms": route.get("cross_rooms"),
            "doors_crossed": route.get("doors_crossed"), "turns": route.get("turns"),
            "landmarks": route.get("landmarks"), "path_length_m": route.get("path_length_m"),
            "labels": labels}


def _instruction_requirements(dimension: str, route: dict, sample: dict,
                              label_index: dict) -> dict:
    """Describe what must differ in the instruction, without inventing facts."""
    route_landmarks = [x.get("object_id") for x in route.get("landmarks", [])]
    route_turns = [x for x in (route.get("turns") or []) if x != "straight"]
    grounding = sample.get("capability_grounding") or {}
    contrast_ids = [x.get("target_object_id")
                    for x in (grounding.get("candidates") or [])
                    if x.get("target_object_id")]
    contrast_labels = {
        object_id: str(label_index.get(object_id, {}).get("label", object_id))
        for object_id in contrast_ids
    }
    policies = {
        "perception": (
            "Keep the route simple. Describe one or more stable visual properties of the "
            "goal that are actually visible in the frames; do not add route landmarks."),
        "grounding": (
            "Make the referring expression distinguish the required target from the "
            "contrast candidates using only visible spatial, regional, relational, or "
            "appearance evidence. The distinguishing claim must affect target choice."),
        "planning": (
            "Express every audited landmark and meaningful turn in route order so the "
            "instruction tests long-horizon route following."),
        "execution": (
            "Keep target semantics unambiguous and describe the short local maneuver, "
            "corner, doorway, or constrained approach only when visible in the frames."),
        "recovery": (
            "Describe the original long-horizon route exactly as for a clean run. Never "
            "mention the benchmark perturbation or recovery intervention."),
    }
    required_landmarks = route_landmarks if dimension in ("planning", "recovery") else []
    required_turns = route_turns if dimension in ("planning", "recovery", "execution") else []
    return {
        "dimension": dimension,
        "policy": policies[dimension],
        "required_landmark_ids": required_landmarks,
        "required_turns": required_turns,
        "contrast_candidate_ids": contrast_ids if dimension == "grounding" else [],
        "contrast_candidate_labels": contrast_labels if dimension == "grounding" else {},
        "require_visual_claim": dimension in ("perception", "grounding", "execution"),
    }


def _symbolic_instruction_gate(candidate: dict, route: dict,
                               label_index: dict, requirements=None) -> tuple[bool, list[str]]:
    errors = []
    requirements = requirements or {
        "required_landmark_ids": [
            x.get("object_id") for x in route.get("landmarks", [])],
        "required_turns": [
            x for x in (route.get("turns") or []) if x != "straight"],
        "require_visual_claim": False,
    }
    instruction = str(candidate.get("instruction") or "").strip()
    goal = route.get("goal", {}).get("object")
    landmarks = [x.get("object_id") for x in route.get("landmarks", [])]
    required_landmarks = requirements.get("required_landmark_ids") or []
    used = candidate.get("ordered_landmark_ids") or []
    if not instruction:
        errors.append("empty_instruction")
    if candidate.get("target_object_id") != goal:
        errors.append("wrong_target_object_id")
    lower_instruction = instruction.lower()
    goal_label = str(label_index.get(goal, {}).get("label", goal)).lower().replace("_", " ")
    if goal_label and goal_label not in lower_instruction:
        errors.append("target_label_missing_from_instruction")
    if not re.search(r"\bstop\b", lower_instruction):
        errors.append("missing_explicit_stop")
    if any(x not in landmarks for x in used):
        errors.append("invented_landmark")
    if used != required_landmarks:
        errors.append("landmark_set_or_order_mismatch")
    positions = [landmarks.index(x) for x in used if x in landmarks]
    if positions != sorted(positions):
        errors.append("landmark_order_mismatch")
    for object_id in used:
        label = str(label_index.get(object_id, {}).get("label", object_id))
        if label.lower().replace("_", " ") not in lower_instruction:
            errors.append("declared_landmark_missing_from_instruction:" + str(object_id))
    ordinal = candidate.get("door_ordinal")
    if ordinal is not None and (not isinstance(ordinal, int) or ordinal < 1 or
                                ordinal > int(route.get("doors_crossed") or 0)):
        errors.append("invalid_door_ordinal")
    claimed_turns = candidate.get("turns") or []
    if claimed_turns != (requirements.get("required_turns") or []):
        errors.append("turn_sequence_mismatch")
    claims = candidate.get("grounding_claims") or []
    if requirements.get("require_visual_claim") and not claims:
        errors.append("missing_visual_grounding_claim")
    forbidden = ("pick up", "carry", "bring", "retrieve", "place the")
    if any(x in instruction.lower() for x in forbidden):
        errors.append("non_navigation_task")
    return not errors, errors


def _generate_instruction(url, model, frames, route, label_index, dimension,
                          sample, verify=True, feedback=None):
    facts = _route_facts(route, label_index)
    requirements = _instruction_requirements(dimension, route, sample, label_index)
    required_landmarks = requirements["required_landmark_ids"]
    required_turns = requirements["required_turns"]
    required_labels = [facts["labels"].get(x, x) for x in required_landmarks]
    prompt = """You create one English indoor navigation instruction for CAPABILITY_DIMENSION from
an executed golden route. Ordered frames are visual evidence, not images shown to the evaluated
agent. Use only facts in ROUTE_JSON and entities visibly supported by those frames. The target must
be the route goal. A claim is useful only when it changes target choice or route execution; avoid
decorative verbosity and never invent colors, materials, room names, or spatial relations.
DIMENSION_POLICY is mandatory. Every instruction must explicitly tell the agent to STOP beside
or next to the target; merely saying approach, walk toward, or go to the target is not sufficient.
Return JSON only with keys: instruction, target_object_id,
ordered_landmark_ids (array), door_ordinal (integer or null), turns (array of left/right),
grounding_claims (array of short factual claims), instruction_factors (array chosen from
visual_attribute, spatial_relation, region, negative_constraint, ordered_landmark,
door_ordinal, turn_sequence, constrained_approach), confidence (0..1).
For auditability, target_object_id MUST equal
REQUIRED_TARGET, ordered_landmark_ids MUST equal REQUIRED_LANDMARK_IDS in exactly that order,
turns MUST equal REQUIRED_TURNS, and the instruction must explicitly contain every
REQUIRED_LANDMARK_LABEL plus the target label.
CAPABILITY_DIMENSION=""" + dimension + "\nDIMENSION_POLICY=" + requirements["policy"] + \
        "\nREQUIRED_TARGET=" + str(
        route.get("goal", {}).get("object")) + "\nREQUIRED_LANDMARK_IDS=" + json.dumps(
        required_landmarks) + "\nREQUIRED_LANDMARK_LABELS=" + json.dumps(
        required_labels) + "\nREQUIRED_TURNS=" + json.dumps(required_turns) + \
        "\nCONTRAST_CANDIDATE_IDS=" + json.dumps(
            requirements["contrast_candidate_ids"]) + \
        "\nCONTRAST_CANDIDATE_LABELS=" + json.dumps(
            requirements["contrast_candidate_labels"], ensure_ascii=False) + \
        "\nROUTE_JSON=" + json.dumps(facts, ensure_ascii=False)
    if feedback:
        prompt += "\nA previous candidate was rejected for: " + ", ".join(feedback) + \
                  ". Correct those errors and stay strictly inside ROUTE_JSON."
    content = [{"type": "text", "text": prompt}]
    for image in frames:
        content.append(_image_part(image))
    candidate, raw = _chat(url, model, content)
    symbolic_ok, errors = _symbolic_instruction_gate(
        candidate, route, label_index, requirements=requirements)
    verification = {"pass": symbolic_ok, "errors": errors, "mode": "symbolic"}
    if symbolic_ok and verify:
        vp = """Audit whether the instruction is fully grounded in the ordered route frames and
ROUTE_JSON and satisfies DIMENSION_POLICY. ROUTE_JSON is authoritative for turns, rooms, door
counts, and ordering; do not
second-guess those geometric facts from sparse frames. Reject unsupported visual attributes,
wrong object identity, a claimed visual landmark absent from the frames, or decorative claims
that do not help execute this capability case.
Return JSON only: {\"pass\":true|false,\"reasons\":[...]}.
DIMENSION_POLICY=""" + requirements["policy"] + "\nROUTE_JSON=" + json.dumps(
            facts, ensure_ascii=False) + "\nCANDIDATE=" + json.dumps(candidate)
        verify_content = [{"type": "text", "text": vp}] + [_image_part(x) for x in frames]
        judged, judge_raw = _chat(url, model, verify_content, max_tokens=2048)
        verification = {"pass": bool(judged.get("pass")),
                        "reasons": judged.get("reasons") or [], "mode": "symbolic+visual",
                        "raw": judge_raw}
    return candidate, {"raw": raw, "symbolic_errors": errors,
                       "verification": verification,
                       "accepted": symbolic_ok and bool(verification.get("pass"))}


def _fallback_instruction(route: dict, labels: dict, dimension="planning") -> str:
    goal_id = route["goal"]["object"]
    goal = str(labels.get(goal_id, {}).get("label", goal_id)).replace("_", " ").lower()
    landmarks = (route.get("landmarks") or []) if dimension in (
        "planning", "recovery") else []
    parts = []
    if landmarks:
        lid = landmarks[0]["object_id"]
        label = str(labels.get(lid, {}).get("label", lid)).replace("_", " ").lower()
        parts.append("Pass the %s" % label)
    if dimension in ("planning", "recovery") and route.get("doors_crossed", 0) >= 2:
        parts.append("take the second doorway on the route")
    turns = ([x for x in route.get("turns", []) if x != "straight"][:2]
             if dimension in ("planning", "recovery", "execution") else [])
    for turn in turns:
        parts.append("turn %s" % turn)
    parts.append("stop beside the %s" % goal)
    return ", then ".join(parts).capitalize() + "."


def _source_instruction(sample: dict) -> str | None:
    """Prefer the deterministic route-grounded Object-Goal low instruction."""
    instructions = sample.get("instructions") or []
    ranked = sorted(instructions, key=lambda item: (
        item.get("instruction_type") != "Object-Goal",
        item.get("level") != "low"))
    for item in ranked:
        text = str(item.get("generated_instruction") or item.get("instruction") or "").strip()
        if text:
            return text
    return None


def _console_sample(sample: dict, case: dict, route: dict,
                    target_label: str) -> dict:
    """Materialize one accepted case in the existing GVLN console format."""
    exported = copy.deepcopy(sample)
    exported["trajectory_id"] = "%s__%s" % (
        sample["trajectory_id"], case["dimension"])
    start_id = route.get("start", {}).get("object")
    exported["instructions"] = [{
        "instruction_type": "Capability/%s" % case["dimension"].title(),
        "level": "v3-dev",
        "generated_instruction": case["instruction"],
        "start": str(start_id or ""),
        "end": str(target_label or case.get("target_object_id") or ""),
    }]
    exported["capability_case"] = {
        "case_id": case["case_id"],
        "dimension": case["dimension"],
        "schema_version": case["schema_version"],
        "evaluation_contract": case["evaluation_contract"],
        "instruction_factors": case.get("instruction_factors") or [],
        "target_object_id": case.get("target_object_id"),
    }
    return exported


def _write_console_dataset(root: Path, samples_by_scene: dict,
                           append: bool = False) -> dict:
    """Write/merge accepted cases into one split consumed by SageSet."""
    written = 0
    root.mkdir(parents=True, exist_ok=True)
    for scene, incoming in sorted(samples_by_scene.items()):
        scene_dir = root / str(scene)
        scene_dir.mkdir(parents=True, exist_ok=True)
        path = scene_dir / ("%s_capability_v3.json" % scene)
        existing = []
        if append and path.is_file():
            doc = json.loads(path.read_text(encoding="utf-8"))
            for scene_doc in doc.get("scenes", []):
                existing.extend(scene_doc.get("samples") or [])
        merged = {
            str(item.get("trajectory_id")): item
            for item in existing
            if item.get("trajectory_id") is not None
        }
        for item in incoming:
            merged[str(item["trajectory_id"])] = item
        samples = [merged[key] for key in sorted(merged)]
        doc = {"scenes": [{"scene_name": str(scene), "samples": samples}]}
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        written += len(incoming)
    return {"root": str(root), "new_cases": written,
            "scenes": len(samples_by_scene)}


def _recovery_plan(occ, points, yaws):
    total = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))
    arcs = [0.0]
    for a, b in zip(points, points[1:]):
        arcs.append(arcs[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    candidates = [i for i, arc in enumerate(arcs[:-1]) if .4 <= arc / max(total, 1e-9) <= .7]
    best = None
    for i in candidates:
        x, y = points[i]
        for signed in (-90, 90, -60, 60, -45, 45):
            yaw = yaws[i] + math.radians(signed)
            tx, ty = x + .8 * math.cos(yaw), y + .8 * math.sin(yaw)
            nx, ny, hit = occ.try_step(x, y, tx, ty, .07, True)
            moved = math.hypot(nx - x, ny - y)
            lateral = project_to_path((nx, ny), points)["distance_m"]
            score = lateral + .25 * moved - .3 * int(bool(hit))
            if moved >= .55 and lateral >= .35 and (best is None or score > best[0]):
                direction = "left" if signed > 0 else "right"
                best = (score, {"trigger_progress": round(arcs[i] / total, 3),
                                "sequence": [{"name": "turn", "arguments": {
                                    "direction": direction, "degrees": abs(signed)}},
                                    {"name": "move_forward", "arguments": {"distance": .8}}],
                                "golden_pose_xy": [float(x), float(y)],
                                "expected_wrong_xy": [float(nx), float(ny)],
                                "expected_lateral_error_m": float(round(lateral, 3)),
                                "return_corridor_m": .6})
    if best is None:
        return None
    # The fixed construction probe proves that this route has recoverable side
    # space.  Evaluation never replays its scene-coordinate target blindly: it
    # replans from the agent's actual trigger pose using these generic bounds.
    return {
        "trigger_progress": 0.40,
        "max_trigger_progress": 0.75,
        "max_pre_lateral_error_m": 0.60,
        "min_post_lateral_error_m": 0.35,
        "max_post_lateral_error_m": 1.20,
        "min_forced_move_m": 0.50,
        "candidate_turn_degrees": [-90, 90, -60, 60, -45, 45],
        "candidate_move_distances_m": [0.8, 0.6],
        "robot_radius_m": 0.07,
        "rejoin_ahead_m": 1.0,
        "max_return_path_m": 3.0,
        "return_corridor_m": 0.6,
        "construction_probe": best[1],
    }


def compile_cases(args):
    source = Path(args.source)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    camera = CameraSpec(640, 480, args.hfov)
    renderer = XYZEulerClient(url=args.render_url, camera=camera)
    if args.render and not renderer.ping():
        raise RuntimeError("render server unavailable: " + args.render_url)
    manifest = {"schema_version": args.schema_version, "source": str(source),
                "evaluation_contract": {
                    "primary_success": (SEMANTIC_SUCCESS_CONTRACT
                                        if args.schema_version == V3_SCHEMA_VERSION
                                        else "legacy_point_goal"),
                    "primary_metric": "success_rate",
                    "diagnostics_only": [
                        "spl", "route_adherence", "collision", "blocked",
                        "milestone_coverage", "proxy_stop", "premature_stop"]},
                "provenance": _code_provenance(),
                "generator": {"vlm": args.vlm_model if args.vlm_instructions else None,
                              "instruction_scope": "per_case_dimension",
                              "symbolic_gate": True, "visual_gate": bool(args.verify_vlm),
                              "frame_sampling": {
                                  "mode": "arc_length_plus_wrapped_yaw_v1",
                                  "minimum_frames": int(args.frames),
                                  "distance_step_m": FRAME_DISTANCE_STEP_M,
                                  "turn_step_rad": FRAME_TURN_STEP_RAD,
                                  "max_route_frames": MAX_ROUTE_FRAMES,
                                  "context_budget": {
                                      "model_context_tokens": QWEN36_CONTEXT_TOKENS,
                                      "non_image_reserve_tokens":
                                          QWEN36_NON_IMAGE_RESERVE_TOKENS,
                                      "conservative_tokens_per_640x480_frame":
                                          QWEN36_IMAGE_TOKEN_BUDGET,
                                      "vllm_image_limit": QWEN36_VLLM_IMAGE_LIMIT,
                                      "chat_framing_reserve_tokens":
                                          QWEN36_CHAT_FRAMING_RESERVE_TOKENS,
                                      "preflight":
                                          "utf8_bytes+output+framing+images"},
                                  "prioritized": ["start", "end", "route_landmarks"],
                                  "overflow_policy":
                                      "keep_endpoints_evenly_thin_landmarks"}},
                "cases": [], "coverage": {x: {"accepted": 0, "rejected": []}
                                             for x in CAPABILITIES}}
    console_samples = {}
    vlm_generated = 0
    for path in sorted(source.glob("*/*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for scene_doc in doc.get("scenes", []):
            bare = str(scene_doc.get("scene_name") or path.parent.name)
            scene_dir = _scene_dir(args.sage_root, bare)
            occ = OccMap(scene_dir, collision_mode="usd_rooms", furniture_mode="none")
            label_path = Path(scene_dir, "labels.json")
            raw_labels = json.loads(label_path.read_text(encoding="utf-8"))
            counts, labels = {}, {}
            for item in raw_labels:
                name = str(item.get("label") or "unknown").strip().lower().replace(" ", "_")
                counts[name] = counts.get(name, 0) + 1
                info = dict(item)
                bbox = item.get("bounding_box") or []
                xs = [float(p["x"]) for p in bbox]
                ys = [float(p["y"]) for p in bbox]
                zs = [float(p.get("z", 0.0)) for p in bbox]
                if xs and ys and zs:
                    info.update({"bbox_xyxy": (min(xs), min(ys), max(xs), max(ys)),
                                 "centre_xy": (.5 * (min(xs) + max(xs)),
                                               .5 * (min(ys) + max(ys))),
                                 "z_range": (min(zs), max(zs))})
                labels["%s_%d" % (name, counts[name])] = info
            for sample in scene_doc.get("samples", []):
                points, yaws = _geometry_points(sample)
                clearances = [occ.clearance(x, y) for x, y in points]
                route, landmark_rejections = _audit_route_landmarks(
                    renderer, bare, occ, points, yaws,
                    sample.get("route_grounding") or {}, labels, camera, args)
                features = trajectory_features(points, route, clearances)
                eligibility = eligible_dimensions(features, sample)
                source_profile = sample.get("goal_constraints", {}).get(
                    "capability_profile", "canonical")
                if source_profile in CAPABILITIES:
                    for dimension in CAPABILITIES:
                        if dimension != source_profile:
                            eligibility[dimension] = {
                                "eligible": False,
                                "reason": "source was built for %s profile" % source_profile}
                frames, indices = [], []
                eligible = [d for d in CAPABILITIES if eligibility[d]["eligible"]]
                if eligible and args.render:
                    asset_dir = out / "assets" / bare / sample["trajectory_id"]
                    indices, frames = _render_route(renderer, "sage-" + bare, occ, points, yaws,
                                                    asset_dir, args.frames, route=route)
                target = route.get("goal", {}).get("object")
                target_label = str(labels.get(target, {}).get("label", target))
                common = {"scene": bare, "episode_id": sample["trajectory_id"] + "_0",
                          "source_trajectory_id": sample["trajectory_id"],
                          "schema_version": args.schema_version,
                          "evaluation_contract": dict(manifest["evaluation_contract"]),
                          "target_object_id": target,
                          "target_terms": [target_label.replace("_", " ").lower()],
                          "golden_path_xy": [[round(x, 5), round(y, 5)] for x, y in points],
                          "features": features, "route": route,
                          "landmark_rejections": landmark_rejections,
                          "assets": {"sampled_indices": indices,
                                     "dir": str(out / "assets" / bare /
                                                sample["trajectory_id"]) if indices else None},
                          "oracle": {"target_visible": sample.get("goal_constraints", {}).get(
                              "goal_visibility", {}).get("status") == "verified",
                                     "milestone_arcs_m": [x.get("arc_m") for x in
                                                           route.get("landmarks", [])],
                                     "candidate_goal_xy": {
                                         x["target_object_id"]: x["goal_xy"]
                                         for x in (sample.get("capability_grounding", {}).get(
                                             "candidates") or [])}}}
                recovery = _recovery_plan(occ, points, yaws) if eligibility["recovery"]["eligible"] else None
                if eligibility["recovery"]["eligible"] and recovery is None:
                    eligibility["recovery"] = {"eligible": False,
                                                "reason": "no traversable off-route perturbation"}
                for dim in CAPABILITIES:
                    if not eligibility[dim]["eligible"]:
                        manifest["coverage"][dim]["rejected"].append({
                            "trajectory_id": sample["trajectory_id"],
                            "reason": eligibility[dim]["reason"], "features": features})
                        continue
                    instruction = None
                    generation = None
                    if (args.vlm_instructions and frames and
                            (args.max_vlm_trajectories <= 0 or
                             vlm_generated < args.max_vlm_trajectories)):
                        try:
                            attempts, feedback = [], None
                            candidate = None
                            for _ in range(max(1, args.vlm_retries)):
                                candidate, attempt = _generate_instruction(
                                    args.vlm_url, args.vlm_model, frames, route, labels,
                                    dim, sample, args.verify_vlm, feedback=feedback)
                                attempts.append(attempt)
                                if attempt["accepted"]:
                                    instruction = candidate["instruction"]
                                    break
                                feedback = list(attempt.get("symbolic_errors") or [])
                                feedback += list(
                                    attempt.get("verification", {}).get("reasons") or [])
                            generation = {
                                "accepted": instruction is not None,
                                "dimension": dim,
                                "attempts": attempts,
                                "selected": candidate if instruction is not None else None}
                            vlm_generated += 1
                        except Exception as exc:
                            generation = {"accepted": False, "dimension": dim,
                                          "error": str(exc)}
                    if instruction is None:
                        if args.require_vlm_instructions:
                            manifest["coverage"][dim]["rejected"].append({
                                "trajectory_id": sample["trajectory_id"],
                                "reason": "dimension-specific VLM instruction rejected",
                                "features": features,
                                "instruction_generation": generation})
                            continue
                        instruction = (_source_instruction(sample)
                                       or _fallback_instruction(route, labels, dim))
                    case = dict(common)
                    case.update({"case_id": "%s__%s" % (sample["trajectory_id"], dim),
                                 "dimension": dim,
                                 "instruction": instruction,
                                 "instruction_generation": generation,
                                 "instruction_factors": (
                                     (generation.get("selected") or {}).get(
                                         "instruction_factors", [])
                                     if generation else [])})
                    if dim == "recovery":
                        case["recovery"] = recovery
                    manifest["cases"].append(case)
                    if args.console_output:
                        console_samples.setdefault(bare, []).append(
                            _console_sample(sample, case, route, target_label))
                    manifest["coverage"][dim]["accepted"] += 1
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    coverage = {k: v["accepted"] for k, v in manifest["coverage"].items()}
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    console = None
    if args.console_output:
        console = _write_console_dataset(
            Path(args.console_output), console_samples, append=args.console_append)
        manifest["console_dataset"] = console
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(out / "manifest.json"), "cases": len(manifest["cases"]),
                      "coverage": coverage, "console_dataset": console},
                     ensure_ascii=False, indent=2))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sage-root", default=os.environ.get("SAGE_ROOT", DEFAULT_ROOT))
    ap.add_argument("--source", default=os.path.join(DEFAULT_ROOT,
                    "SAGE-3D_VLN_Data/demo/Trajectory_demo"))
    ap.add_argument("--output", default=DEFAULT_OUT)
    ap.add_argument("--console-output", default=None,
                    help="optional unified GVLN Trajectory_demo directory for one console")
    ap.add_argument("--console-append", action="store_true",
                    help="merge accepted cases into an existing unified console directory")
    ap.add_argument("--schema-version", choices=(SCHEMA_VERSION, V3_SCHEMA_VERSION),
                    default=SCHEMA_VERSION,
                    help="v3-dev uses one semantic-terminal success contract for every family")
    ap.add_argument("--render-url", default="http://127.0.0.1:7038")
    ap.add_argument("--hfov", type=float, default=115.0)
    ap.add_argument("--camera-height", type=float, default=1.2)
    ap.add_argument("--frames", type=int, default=8,
                    help="minimum route frames; turns and long paths add adaptive frames")
    ap.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--vlm-instructions", action="store_true")
    ap.add_argument("--require-vlm-instructions", action="store_true",
                    help="reject a case instead of falling back when its dimension-specific "
                         "trajectory-frame instruction fails audit")
    ap.add_argument("--verify-vlm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--vlm-url", default="http://127.0.0.1:28899")
    ap.add_argument("--vlm-model", default="Qwen3.6-35B")
    ap.add_argument("--max-vlm-trajectories", type=int, default=0,
                    help="legacy flag name: maximum dimension-specific VLM generations; "
                         "0 means all eligible cases")
    ap.add_argument("--vlm-retries", type=int, default=3)
    ap.add_argument("--landmark-min-width", type=float, default=20.0)
    ap.add_argument("--landmark-min-height", type=float, default=20.0)
    ap.add_argument("--landmark-min-area", type=float, default=.001)
    ap.add_argument("--landmark-depth-stride", type=int, default=4)
    ap.add_argument("--landmark-depth-tolerance", type=float, default=.10)
    ap.add_argument("--landmark-min-depth-fraction", type=float, default=.12)
    ap.add_argument("--landmark-min-depth-area", type=float, default=.001)
    compile_cases(ap.parse_args())


if __name__ == "__main__":
    main()
