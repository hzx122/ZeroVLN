"""Canonical action expert v2 prompt, result, and memory contract.

This module is deliberately dependency-light so online inference, SFT data
construction, training replay, and evaluation can import the exact same bytes.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Iterable


ACTION_HEAD_AS_TOOL_VERSION = "v2"
PROMPT_VERSION = "slow_system_prompt_v2"
TOOL_RESULT_SCHEMA = "action_head_tool_result_v2"
SLOW_CONTEXT_VERSION = "slow_context_v2"
BUNDLE_SCHEMA = "fm_nested_bundle_v3"
SLOW_DECISION_SCHEMA = "slow_tool_decision_v2"
SUPPORTED_ACTION_HEAD_AS_TOOL_VERSIONS = ("v1", "v2")
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
    "location, and 1-3 semantic stages. The fast system repeatedly observes RGB, "
    "predicts a short trajectory, and executes its prefix. Do not provide primitive "
    "actions, coordinates, a future trajectory, goal distance, or a safety budget."
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
            "description": "One to three ordered semantic motion stages.",
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

Use only the instruction, visible images, history, and returned non-oracle facts. Actual motion, blocked, collision, and no_progress override your earlier expectation. After every return, compare the call with the trace and final view. If execution deviated, correct the next target, where, or plan; never assume the fast system was right.

Poses are relative to that option's start: x=forward meters, y=left meters, positive yaw=left degrees. path_m is traveled distance and net_m is start-to-end displacement. ok=true only means the tool ran. completion=unknown or a budget return never proves goal completion.

Call stop only when the current view confirms the final destination. If uncertain, continue. Never assume hidden goal coordinates, future routes, or evaluator data."""

CORRECTION_EXAMPLE = """Example:
<tool_call>
{"name":"run_action_head","arguments":{"target":"the doorway ahead","where":"in the center of the current view","plan":["approach the doorway","pass through it"]}}
</tool_call>

Possible multimodal result (the image markers represent image parts):
{"schema":"action_head_tool_result_v2","tool":"run_action_head","option_id":"option-12","ok":true,"return_reason":"blocked","completion":"unknown","chunks":2,"path_m":0.49,"net_m":0.47,"blocked":true,"collision":true,"steps_used":8}
[option=option-12 chunk=1 actual_end_pose={"x_m":0.46,"y_m":0.01,"yaw_deg":2.0} path_m=0.46 blocked=false collision=false no_progress=false] <image:chunk=1>
[option=option-12 chunk=2 actual_end_pose={"x_m":0.47,"y_m":0.00,"yaw_deg":2.0} path_m=0.03 blocked=true collision=true no_progress=true] <image:chunk=2,final>

If the final image shows blockage and an open passage on the left, correct rather than repeat:
<tool_call>
{"name":"run_action_head","arguments":{"target":"the open passage left of the obstacle","where":"on the left side of the current view","plan":["turn toward the open passage","enter the passage"]}}
</tool_call>

When the final destination is visibly reached:
<tool_call>
{"name":"stop","arguments":{"reason":"The instructed destination is visible at the robot's current position."}}
</tool_call>"""


def build_system_prompt() -> str:
    # Keep v2 close to v1's prompt budget even though it also documents the
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
        raise ValueError(f"Action Head-as-Tool v2 identity mismatch: {mismatches}")


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
    """Apply v2's independent slow and latest-fast image budgets.

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
    images = [file_image_part(path, kind="slow") for path in slow_image_paths]
    if not images:
        raise ValueError("v2 initial messages require at least one slow observation")
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
