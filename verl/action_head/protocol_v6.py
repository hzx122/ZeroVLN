"""v4.11 ReAct semantics plus one latest actual-trajectory map per request."""
from __future__ import annotations

import hashlib

from . import protocol_v4_11 as _V411
from .protocol_v4_11 import *  # noqa: F401,F403
from .trajectory_map import (
    TRAJECTORY_MAP_RENDERER_SHA256,
    TRAJECTORY_MAP_SCHEMA,
    TrajectoryMapState,
    inject_trajectory_map,
)


ACTION_HEAD_AS_TOOL_VERSION = "v6"
PROMPT_VERSION = "slow_system_prompt_v6_trajectory_map"
TOOL_RESULT_SCHEMA = "action_head_tool_result_v6"
SLOW_CONTEXT_VERSION = "slow_context_v6_trajectory_map"
SLOW_DECISION_SCHEMA = "slow_tool_decision_v6"


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"v6 prompt source changed: expected one occurrence of {old!r}")
    return text.replace(old, new)


CONTEXT_AND_POLICY = _V411.CONTEXT_AND_POLICY
CONTEXT_AND_POLICY = _replace_once(
    CONTEXT_AND_POLICY,
    "- The latest returned first-person image is the current observation. Earlier images show scenes from previous positions or orientations. Use them to recognize where you came from, landmarks, and changes in the target's location, not as your current view.",
    "- The latest returned first-person image is the current observation. Earlier images show scenes from previous positions or orientations. Use them to recognize where you came from, landmarks, and changes in the target's location, not as your current view.\n"
    "- Each decision also includes one latest actual trajectory map immediately before the current first-person image. It is a start-relative top-down rendering of motion that the robot actually executed. Legend: the green S marks the episode start; the blue C marks the current position; S/C labels their overlap at the start. The blue line connects recorded actual chunk endpoints in execution order, and the dark blue arrow indicates the current heading. Orange squares mark recorded chunk-end positions where execution reported blocked; red crosses mark recorded chunk-end positions where execution reported collision. Refer to these historical execution markers when reviewing repeated blocked or collision attempts. Up is the starting forward direction and left is the starting left direction. The half-width label gives the distance in meters from the plot center to each side of the square plotting area; for example, half-width 4 m means an 8 m by 8 m plotting area. The displayed scale may expand as the trajectory grows."
)
CONTEXT_AND_POLICY = _replace_once(
    CONTEXT_AND_POLICY,
    "- You can also review your movement history: the actual chunk poses, path lengths, and start-to-end displacements returned by past run_action_head calls, together with actual turn angles, remain in chronological order in the conversation. Use these records with their images to determine where you have passed, which way you moved or turned, and whether you backtracked or repeated an attempt. These are actual execution records, not previously planned routes. They are segment poses and movement summaries, not a complete step-by-step coordinate sequence or a global map. Each run_action_head call uses its own starting pose as the reference; do not directly compare coordinates from different calls as if they shared one coordinate frame. Do not guess missing poses.",
    "- You can also review your movement history: the actual chunk poses, path lengths, and start-to-end displacements returned by past run_action_head calls, together with actual turn angles, remain in chronological order in the conversation. Use these records with their images to determine where you have passed, which way you moved or turned, and whether you backtracked or repeated an attempt. The trajectory map is a compact spatial rendering of this same actual execution history; use the textual results for precise per-call facts. These are actual execution records, not previously planned routes. The textual records are segment poses and movement summaries, and each run_action_head call uses its own starting pose as the reference; do not directly compare coordinates from different calls as if they shared one coordinate frame. Do not guess missing poses or treat the trajectory map as environment geometry."
)
CONTEXT_AND_POLICY = _replace_once(
    CONTEXT_AND_POLICY,
    "- At most 8 slow observations and 3 intermediate images from the latest run_action_head call are retained, for a total of at most 11 images. Slow observations include the starting view, views after turn, and views at the end of run_action_head; the current view counts as one of them. Intermediate images from the latest call are counted separately; images from older calls are not used to fill unused slots.",
    "- At most 8 slow observations and 3 intermediate images from the latest run_action_head call are retained. One latest trajectory map is added in its own slot, for a total of at most 12 images. Slow observations include the starting view, views after turn, and views at the end of run_action_head; the current view counts as one of them. Intermediate images from the latest call are counted separately; images from older calls are not used to fill unused slots. The trajectory map never replaces the current RGB."
)
CONTEXT_AND_POLICY = _replace_once(
    CONTEXT_AND_POLICY,
    "- When movement is needed, use run_action_head to specify a visually grounded local target: target names the object to approach or pass, where describes its location in the image, and plan gives the movement plan for this call. Describe what to do from the current state; do not put a narrative of the previous failure into plan.",
    "- When movement is needed, use run_action_head to specify a visually grounded local target: target names the object to approach or pass, where describes its location in the image, and plan gives the movement plan for this call. Ground the local target and nearby obstacles in the current or retained RGB observations. Refer to the trajectory map when it helps detect revisits, loops, backtracking, or lack of overall displacement. Describe what to do from the current state; do not put a narrative of the previous failure into plan."
)
CONTEXT_AND_POLICY = _replace_once(
    CONTEXT_AND_POLICY,
    "- direction_alignment is an auxiliary comparison of planned and actual execution direction, not a measure of task completion. When they disagree, use the latest image and actual movement records to check whether you have deviated from the target.",
    "- direction_alignment is an auxiliary comparison of planned and actual execution direction, not a measure of task completion. When they disagree, use the latest image, actual movement records, and trajectory map to check whether you have deviated from the target."
)
CONTEXT_AND_POLICY = _replace_once(
    CONTEXT_AND_POLICY,
    "Call stop only when the current image, interpreted together with the movement that has actually occurred, shows that you have reached the final location or spatial relationship specified by the instruction.",
    "Call stop only when the current image, interpreted together with the movement that has actually occurred, shows that you have reached the final location or spatial relationship specified by the instruction."
)

TOOLS = _V411.TOOLS
EXAMPLES = _V411.EXAMPLES.replace(_V411.TOOL_RESULT_SCHEMA, TOOL_RESULT_SCHEMA)


def build_system_prompt() -> str:
    return _V411.build_system_prompt().replace(
        _V411.CONTEXT_AND_POLICY, CONTEXT_AND_POLICY
    ).replace(_V411.TOOL_RESULT_SCHEMA, TOOL_RESULT_SCHEMA)


SYSTEM_PROMPT = build_system_prompt()
PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def protocol_identity():
    return dict(
        action_head_as_tool_version=ACTION_HEAD_AS_TOOL_VERSION,
        prompt_version=PROMPT_VERSION,
        prompt_sha256=PROMPT_SHA256,
        tool_result_schema=TOOL_RESULT_SCHEMA,
        slow_context_version=SLOW_CONTEXT_VERSION,
        trajectory_map_schema=TRAJECTORY_MAP_SCHEMA,
        trajectory_map_renderer_sha256=TRAJECTORY_MAP_RENDERER_SHA256,
    )


def assert_protocol_identity(value):
    mismatches = {key: {"expected": wanted, "actual": value.get(key)}
                  for key, wanted in protocol_identity().items()
                  if value.get(key) != wanted}
    if mismatches:
        raise ValueError(f"Action Head-as-Tool v6 identity mismatch: {mismatches}")


def initial_messages(instruction, slow_image_paths):
    messages = _V411.initial_messages(instruction, slow_image_paths)
    messages[0]["content"] = [{"type": "text", "text": SYSTEM_PROMPT}]
    return messages


def new_trajectory_map_state() -> TrajectoryMapState:
    return TrajectoryMapState()
