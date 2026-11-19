"""Canonical action expert v3 prompt, result, and memory contract.

This module is deliberately dependency-light so online inference, SFT data
construction, training replay, and evaluation can import the exact same bytes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from typing import Any, Iterable


ACTION_HEAD_AS_TOOL_VERSION = "v3"
PROMPT_VERSION = "slow_system_prompt_v3"
TOOL_RESULT_SCHEMA = "action_head_tool_result_v3"
SLOW_CONTEXT_VERSION = "slow_context_v3"
BUNDLE_SCHEMA = "fm_nested_bundle_v4"
SLOW_DECISION_SCHEMA = "slow_tool_decision_v3"
SUPPORTED_ACTION_HEAD_AS_TOOL_VERSIONS = ("v1", "v2", "v3")
DEFAULT_ACTION_HEAD_AS_TOOL_VERSION = "v1"
IMAGE_META_KEY = "_action_head_frame"


def normalize_version(value: str | None) -> str:
    version = str(value or DEFAULT_ACTION_HEAD_AS_TOOL_VERSION).strip().lower()
    if version not in SUPPORTED_ACTION_HEAD_AS_TOOL_VERSIONS:
        raise ValueError(
            "action_head_as_tool_version must be one of "
            f"{SUPPORTED_ACTION_HEAD_AS_TOOL_VERSIONS}, got {value!r}")
    return version


RUN_ACTION_HEAD_DESCRIPTION = (
    "Delegate one local option to the Action Head. Give a visible target, its "
    "location, and 1-3 freely worded semantic stages. State the intended motion "
    "clearly in plan, including any initial turn, backing up, or turn-around. The "
    "fast system repeatedly observes RGB, predicts a short trajectory, and executes "
    "its prefix. Do not provide primitive actions, coordinates, a future trajectory, "
    "goal distance, or a safety budget."
)

RUN_ACTION_HEAD_PARAMETERS = {
    "type": "object",
    "properties": {
        "target": {
            "type": "string",
            "description": "The local object or region to approach, pass, or reach.",
        },
        "where": {
            "type": "string",
            "description": (
                "The target's observable location and spatial relation in the "
                "current or retained views."
            ),
        },
        "plan": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {"type": "string"},
            "description": (
                "One to three ordered semantic motion stages. Describe how the "
                "option should turn or reorient, while remaining natural language."
            ),
        },
    },
    "required": ["target", "where", "plan"],
    "additionalProperties": False,
}

STOP_DESCRIPTION = (
    "End the task only when current visual evidence confirms that the instructed "
    "final destination has been reached."
)

STOP_PARAMETERS = {
    "type": "object",
    "properties": {
        "reason": {
            "type": "string",
            "description": "A short statement of the visible evidence for stopping.",
        },
    },
    "required": ["reason"],
    "additionalProperties": False,
}


def validate_stop_arguments(arguments: Any):
    """Validate stop separately from the run_action_head contract."""
    if not isinstance(arguments, dict):
        return None, "arguments must be an object"
    unknown = sorted(set(arguments) - {"reason"})
    if unknown:
        return None, f"unknown arguments: {', '.join(unknown)}"
    reason = str(arguments.get("reason") or "").strip()
    if not reason:
        return None, "reason must be a non-empty string"
    return {"reason": reason}, ""


def format_tool_call(name: str, arguments: dict[str, Any]) -> str:
    payload = {"name": str(name), "arguments": arguments}
    return "<tool_call>" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")) + "</tool_call>"

TOOL_CALL_INSTRUCTION = """For each function call, return a JSON object with function name and arguments within <tool_call></tool_call> XML tags:

<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

TOOLS = (
    {"name": "run_action_head", "description": RUN_ACTION_HEAD_DESCRIPTION,
     "parameters": RUN_ACTION_HEAD_PARAMETERS},
    {"name": "stop", "description": STOP_DESCRIPTION,
     "parameters": STOP_PARAMETERS},
)

POLICY_TEXT = """You are an indoor navigation robot. Complete the user's instruction by calling tools.

The latest slow observation is the current first-person view. Up to eight slow observations remain in their original conversation positions. During run_action_head, the fast system repeatedly observes live RGB, predicts a short trajectory, and executes its prefix. Its result pairs every chunk's actual end pose with the RGB observed there. All intermediate images from the latest call remain visible; older ones may become labeled placeholders.

Use only the instruction, visible images, history, and returned non-oracle facts. The result's direction_alignment compares the plan's intended direction with the complete option's actual direction. If match=false, treat the previous plan as not faithfully executed and replan from the final RGB and trace. Silently issue a normal target, where, and plan for the current state: never mention correction, recovery, deviation, failure, or the previous execution in those arguments. Do not mechanically invert or repeat a direction without current visual support.

Poses are relative to that option's start: x=forward meters, y=left meters, positive yaw=left degrees. path_m is traveled distance and net_m is start-to-end displacement. ok=true only means the tool ran. completion=unknown or a budget return never proves goal completion.

Call stop only when the current view confirms the final destination. If uncertain, continue. Never assume hidden goal coordinates, future routes, or evaluator data."""

CORRECTION_EXAMPLE = """Example:
<tool_call>
{"name":"run_action_head","arguments":{"target":"the doorway","where":"on the left side of the current view","plan":["turn left toward the doorway","approach the doorway"]}}
</tool_call>

Possible multimodal result (the image markers represent image parts):
{"schema":"action_head_tool_result_v3","tool":"run_action_head","option_id":"option-12","ok":true,"return_reason":"fast_step_budget_reached","completion":"unknown","chunks":2,"path_m":0.80,"net_m":0.78,"blocked":false,"collision":false,"steps_used":8,"direction_alignment":{"planned":"left","actual":"right","match":false}}
[option=option-12 chunk=1 actual_end_pose={"x_m":0.39,"y_m":-0.04,"yaw_deg":-5.0} path_m=0.40 blocked=false collision=false no_progress=false] <image:chunk=1>
[option=option-12 chunk=2 actual_end_pose={"x_m":0.77,"y_m":-0.13,"yaw_deg":-12.0} path_m=0.40 blocked=false collision=false no_progress=false] <image:chunk=2,final>

If the final image still places the doorway on the left, silently replan from that current view:
<tool_call>
{"name":"run_action_head","arguments":{"target":"the doorway","where":"on the left side of the current view","plan":["turn left until facing the doorway","approach the doorway"]}}
</tool_call>

When the final destination is visibly reached:
<tool_call>
{"name":"stop","arguments":{"reason":"The instructed destination is visible at the robot's current position."}}
</tool_call>"""


def build_system_prompt() -> str:
    # Keep v3 close to v2's prompt budget even though it also documents the
    # multimodal result contract and one correction example.
    tool_json = json.dumps(TOOLS, ensure_ascii=False, separators=(",", ":"))
    return (
        f"{POLICY_TEXT}\n\nYou have the following tools available.\n\n"
        f"Tool call format:\n{TOOL_CALL_INSTRUCTION}\n\nTool list:\n"
        f"<tools>\n{tool_json}\n</tools>\n\n{CORRECTION_EXAMPLE}"
    )


SYSTEM_PROMPT = build_system_prompt()
PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def current_view_text() -> str:
    return "The image above is your current first-person view. Choose the next tool call."


def task_text(instruction: str) -> str:
    return f"Navigation instruction: {instruction}"


def append_tool_budget(text: str, remaining: int) -> str:
    remaining = max(0, int(remaining))
    notice = ("[Tool-call budget exhausted. Do not call another tool.]"
              if remaining == 0 else
              f"[Tool-call budget: at most {remaining} calls remain.]")
    return f"{text.rstrip()}\n\n{notice}" if text.strip() else notice


def protocol_identity() -> dict[str, str]:
    return {
        "action_head_as_tool_version": ACTION_HEAD_AS_TOOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "tool_result_schema": TOOL_RESULT_SCHEMA,
        "slow_context_version": SLOW_CONTEXT_VERSION,
    }


def assert_protocol_identity(value: dict[str, Any]) -> None:
    expected = protocol_identity()
    mismatches = {
        key: {"expected": wanted, "actual": value.get(key)}
        for key, wanted in expected.items() if value.get(key) != wanted
    }
    if mismatches:
        raise ValueError(f"Action Head-as-Tool v3 identity mismatch: {mismatches}")


def wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def planned_option_direction(arguments: dict[str, Any]) -> str:
    """Project a free-language plan into an audit label without constraining it."""
    plan = " ".join(str(value) for value in arguments.get("plan", [])).lower()
    if re.search(r"\b(turn around|turn back|u-turn|u turn|face .*behind)\b", plan):
        return "turn_around"
    if re.search(r"\b(back up|move backward|go backward|step back|reverse)\b", plan):
        return "backward"
    left = bool(re.search(r"\b(turn|bear|veer|rotate|pivot)\s+(?:to the\s+)?left\b", plan))
    right = bool(re.search(r"\b(turn|bear|veer|rotate|pivot)\s+(?:to the\s+)?right\b", plan))
    if left != right:
        return "left" if left else "right"
    if left and right:
        return "unknown"
    if re.search(
            r"\b(move|go|walk|head|proceed|continue)\s+(?:straight|forward)\b",
            plan):
        return "straight"
    where = str(arguments.get("where") or "").lower()
    if re.search(r"\bbehind\b", where):
        return "turn_around"
    left = bool(re.search(r"\bleft\b", where))
    right = bool(re.search(r"\bright\b", where))
    if left != right:
        return "left" if left else "right"
    if re.search(
            r"\bstraight ahead\b|\bdirectly (?:in front|ahead)\b|\bon the wall ahead\b",
            where):
        return "straight"
    return "unknown"


def actual_option_direction(endpoint: dict[str, Any] | None) -> str:
    """Classify one complete option endpoint in its option-start frame."""
    if not endpoint:
        return "unknown"
    x_m = float(endpoint.get("x_m") or 0.0)
    y_m = float(endpoint.get("y_m") or 0.0)
    yaw_deg = wrap_degrees(float(endpoint.get("yaw_deg") or 0.0))
    if abs(yaw_deg) >= 120.0:
        return "turn_around"
    if x_m <= -0.05 and abs(yaw_deg) < 45.0:
        return "backward"
    if yaw_deg >= 3.0:
        return "left"
    if yaw_deg <= -3.0:
        return "right"
    if y_m >= 0.03:
        return "left"
    if y_m <= -0.03:
        return "right"
    if x_m >= 0.05:
        return "straight"
    return "unknown"


def direction_alignment(arguments: dict[str, Any],
                        endpoint: dict[str, Any] | None, *,
                        mode: str | None = None) -> dict[str, Any]:
    mode = mode or os.environ.get("ACTION_HEAD_DIRECTION_ALIGNMENT", "legacy")
    if mode == "conservative_v1":
        return conservative_direction_alignment(arguments, endpoint)
    if mode != "legacy":
        raise ValueError("unknown direction alignment mode: " + mode)
    planned = planned_option_direction(arguments)
    actual = actual_option_direction(endpoint)
    comparable = planned != "unknown" and actual != "unknown"
    return {"planned": planned, "actual": actual,
            "match": planned == actual if comparable else None}


def conservative_direction_alignment(arguments: dict[str, Any],
                                     endpoint: dict[str, Any] | None) -> dict[str, Any]:
    """Compare like quantities, abstaining on weak or compound evidence.

    Translation uses displacement in the option-start frame, not final yaw.
    Only explicit, single-stage rotation plans use yaw. This is a direction
    diagnostic, never a task-completion or collision/safety judgement.
    Thresholds are an initial conservative candidate, not fitted SR optima.
    """
    planned = planned_option_direction(arguments)
    result = {"planned": planned, "actual": "unknown", "match": None}
    if planned == "unknown" or not endpoint:
        return result
    try:
        x, y, yaw = (float(endpoint[k]) for k in ("x_m", "y_m", "yaw_deg"))
    except (KeyError, TypeError, ValueError):
        return result
    if not all(math.isfinite(v) for v in (x, y, yaw)):
        return result
    yaw = wrap_degrees(yaw)
    plan = " ".join(str(v) for v in arguments.get("plan", [])).lower()
    # An option endpoint cannot establish the order of intermediate maneuvers.
    rotation = bool(re.search(r"\b(turn|rotate|pivot|face|bear|veer)\b", plan))
    translation = bool(re.search(r"\b(move|go|walk|head|proceed|continue|step|approach|back up)\b", plan))
    if (re.search(r"\b(then|after|before|followed by)\b", plan)
            or (re.search(r"\bleft\b", plan) and re.search(r"\bright\b", plan))
            or (rotation and translation)):
        return result
    if rotation:
        if planned == "turn_around":
            if abs(yaw) >= 150:
                result.update(actual="turn_around", match=True)
            return result  # Partial rotation is not evidence of the opposite turn.
        if planned not in ("left", "right") or not 30 <= abs(yaw) < 150:
            return result
        actual = "left" if yaw > 0 else "right"
        result.update(actual=actual, match=actual == planned)
        return result
    if math.hypot(x, y) < .25:
        return result  # Keep blocked/no_progress facts in their existing fields.
    bearing = math.degrees(math.atan2(y, x))
    actual = ("straight" if abs(bearing) <= 30 else
              "backward" if abs(bearing) >= 150 else
              "left" if bearing > 0 else "right")
    result["actual"] = actual
    if planned in ("straight", "backward"):
        error = abs(wrap_degrees(bearing - (0 if planned == "straight" else 180)))
        if error <= 30:
            result["match"] = True
        elif error >= 60:
            result["match"] = False
    elif planned in ("left", "right") and abs(y) >= .20:
        # Ahead/behind can be intermediate progress toward a lateral target.
        # Require a clear side component before declaring agreement/opposition.
        if actual in ("left", "right"):
            result["match"] = actual == planned
    return result


def is_image_part(part: Any) -> bool:
    return isinstance(part, dict) and part.get("type") in ("image", "image_url")


def tagged_image_part(part: dict[str, Any], *, kind: str,
                      option_id: str | None = None,
                      chunk: int | None = None) -> dict[str, Any]:
    if kind not in ("slow", "fast_intermediate", "final_slow", "reference"):
        raise ValueError(f"unknown Action Head frame kind: {kind}")
    result = copy.deepcopy(part)
    result[IMAGE_META_KEY] = {
        "kind": kind,
        "option_id": option_id,
        "chunk": chunk,
    }
    return result


def _omitted_marker(meta: dict[str, Any]) -> str:
    option_id, chunk = meta.get("option_id"), meta.get("chunk")
    if meta.get("kind") == "fast_intermediate":
        return f"[image omitted: option={option_id}, chunk={chunk}]"
    if option_id is not None:
        return f"[slow observation omitted: option={option_id}, chunk={chunk}]"
    return "[slow observation omitted]"


def strip_context_images(messages: Iterable[dict[str, Any]],
                         keep_slow: int = 8) -> list[dict[str, Any]]:
    """Apply v3's independent slow and latest-fast image budgets.

    Untagged images are treated as slow observations for backwards-compatible
    first-turn construction. Metadata is removed from retained image parts
    before the request is sent to an OpenAI-compatible endpoint.
    """
    if keep_slow <= 0:
        raise ValueError("keep_slow must be positive")
    copied = copy.deepcopy(list(messages))
    images: list[tuple[int, int, dict[str, Any]]] = []
    for message_index, message in enumerate(copied):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part_index, part in enumerate(content):
            if not is_image_part(part):
                continue
            meta = dict(part.get(IMAGE_META_KEY) or {"kind": "slow"})
            images.append((message_index, part_index, meta))

    slow = [entry for entry in images
            if entry[2].get("kind") in ("slow", "final_slow", "reference")]
    keep = {(i, j) for i, j, _ in slow[-keep_slow:]}
    fast = [entry for entry in images
            if entry[2].get("kind") == "fast_intermediate"]
    option_images = [entry for entry in images
                     if entry[2].get("option_id") is not None]
    latest_option = (option_images[-1][2].get("option_id")
                     if option_images else None)
    keep.update((i, j) for i, j, meta in fast
                if meta.get("option_id") == latest_option)

    for message_index, message in enumerate(copied):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        replaced = []
        for part_index, part in enumerate(content):
            if not is_image_part(part):
                replaced.append(part)
                continue
            meta = dict(part.get(IMAGE_META_KEY) or {"kind": "slow"})
            if (message_index, part_index) not in keep:
                replaced.append({"type": "text", "text": _omitted_marker(meta)})
                continue
            clean = dict(part)
            clean.pop(IMAGE_META_KEY, None)
            replaced.append(clean)
        message["content"] = replaced
    return copied


def chunk_text(*, option_id: str, chunk: int,
               actual_end_pose: dict[str, Any] | None, path_m: float,
               blocked: bool, collision: bool, no_progress: bool) -> str:
    pose = actual_end_pose or {"available": False}
    return (
        f"[option={option_id} chunk={int(chunk)} actual_end_pose="
        f"{json.dumps(pose, ensure_ascii=False, separators=(',', ':'))} "
        f"path_m={float(path_m):.3f} blocked={str(bool(blocked)).lower()} "
        f"collision={str(bool(collision)).lower()} "
        f"no_progress={str(bool(no_progress)).lower()}]"
    )


def file_image_part(path: str, *, kind: str,
                    option_id: str | None = None,
                    chunk: int | None = None) -> dict[str, Any]:
    return tagged_image_part(
        {"type": "image_url", "image_url": {"url": "file://" + str(path)}},
        kind=kind, option_id=option_id, chunk=chunk)


def initial_messages(instruction: str,
                     slow_image_paths: Iterable[str]) -> list[dict[str, Any]]:
    # v3: 首次慢系统(option 0)无历史执行, 初始上下文只保留当前首帧,
    # 不放多帧历史; 后续 option 上下文由 append_tool_exchange 累积,
    # 并由 strip_context_images 按 8-慢观测预算压缩.
    images = [file_image_part(path, kind="slow")
              for path in list(slow_image_paths)[-1:]]
    if not images:
        raise ValueError("v3 initial messages require at least one slow observation")
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            *images,
            {"type": "text", "text": current_view_text() + "\n\n" +
             task_text(instruction)},
        ]},
    ]


def append_tool_exchange(messages: list[dict[str, Any]], *, completion: str,
                         summary: dict[str, Any],
                         frames: Iterable[dict[str, Any]]) -> None:
    """Append one completed teacher/runtime option to an unsanitized history."""
    messages.append({"role": "assistant", "content": [
        {"type": "text", "text": completion.strip()}]})
    frame_list = list(frames)
    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": "[Tool result]\n" + json.dumps(
            summary, ensure_ascii=False, separators=(",", ":")),
    }]
    for index, frame in enumerate(frame_list):
        content.append({"type": "text", "text": chunk_text(
            option_id=str(summary["option_id"]), chunk=int(frame["chunk"]),
            actual_end_pose=frame.get("actual_end_pose"),
            path_m=float(frame["path_m"]), blocked=bool(frame.get("blocked")),
            collision=bool(frame.get("collision")),
            no_progress=bool(frame.get("no_progress")))})
        content.append(file_image_part(
            str(frame["image_path"]),
            kind="final_slow" if index == len(frame_list) - 1
            else "fast_intermediate",
            option_id=str(summary["option_id"]), chunk=int(frame["chunk"])))
    messages.append({"role": "user", "content": content})
