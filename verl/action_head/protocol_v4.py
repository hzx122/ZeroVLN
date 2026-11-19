"""Inference-only action expert v4 protocol.

v4 keeps the complete v3 action expert and memory contract, and adds one
deterministic in-place ``turn`` tool for the slow system.  It is deliberately
not a training/data protocol: the existing D16-v3 checkpoint is evaluated
unchanged so the experiment isolates the harness intervention.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from . import protocol_v3 as V3


ACTION_HEAD_AS_TOOL_VERSION = "v4"
PROMPT_VERSION = "slow_system_prompt_v4"
TOOL_RESULT_SCHEMA = "action_head_tool_result_v4"
SLOW_CONTEXT_VERSION = "slow_context_v4"
BUNDLE_SCHEMA = V3.BUNDLE_SCHEMA
SLOW_DECISION_SCHEMA = "slow_tool_decision_v4"
IMAGE_META_KEY = V3.IMAGE_META_KEY

RUN_ACTION_HEAD_DESCRIPTION = (
    "Delegate one local navigation option to the Action Head. Give a visible "
    "target, its location, and 1-3 freely worded semantic stages. Use this for "
    "motion involving translation; the fast system repeatedly observes live RGB, "
    "predicts a short trajectory, and executes its prefix. Do not provide primitive "
    "actions, coordinates, a future trajectory, goal distance, or a safety budget."
)
RUN_ACTION_HEAD_PARAMETERS = V3.RUN_ACTION_HEAD_PARAMETERS

TURN_DESCRIPTION = (
    "Rotate the robot in place to inspect a different direction. Use this when "
    "pure reorientation is needed before choosing a local navigation option. "
    "left is counter-clockwise and right is clockwise; this tool does not translate."
)
TURN_PARAMETERS = {
    "type": "object",
    "properties": {
        "direction": {
            "type": "string", "enum": ["left", "right"],
            "description": "Direction of the in-place rotation.",
        },
        "degrees": {
            "type": "number", "minimum": 0.5, "maximum": 90.0,
            "description": "Rotation angle in degrees, from 0.5 through 90.",
        },
    },
    "required": ["direction", "degrees"],
    "additionalProperties": False,
}

STOP_DESCRIPTION = V3.STOP_DESCRIPTION
STOP_PARAMETERS = V3.STOP_PARAMETERS
TOOLS = (
    {"name": "turn", "description": TURN_DESCRIPTION,
     "parameters": TURN_PARAMETERS},
    {"name": "run_action_head", "description": RUN_ACTION_HEAD_DESCRIPTION,
     "parameters": RUN_ACTION_HEAD_PARAMETERS},
    {"name": "stop", "description": STOP_DESCRIPTION,
     "parameters": STOP_PARAMETERS},
)

TOOL_CALL_INSTRUCTION = V3.TOOL_CALL_INSTRUCTION
POLICY_TEXT = """You are an indoor navigation robot. Complete the user's instruction by calling exactly one tool at a time.

The latest slow observation is the current first-person view. Up to eight slow observations remain in their original conversation positions. During run_action_head, the fast system repeatedly observes live RGB, predicts a short trajectory, and executes its prefix. Its result pairs every chunk's actual end pose with the RGB observed there. All intermediate images from the latest call remain visible; older ones may become labeled placeholders.

Use turn only for deliberate in-place reorientation: for example, to inspect a direction that is not currently visible or after an option moved in a direction inconsistent with its plan. After turn, inspect its returned current RGB before deciding again. Use run_action_head for every option involving translation. Do not describe correction or recovery inside run_action_head arguments; give the normal target, where, and plan for the current state.

Use only the instruction, visible images, history, and returned non-oracle facts. direction_alignment compares the plan's intended direction with the complete option's actual direction. If match=false, do not assume the prior plan was executed faithfully; decide from the final RGB and trace whether to turn, issue a new local option, or stop. Do not mechanically invert or repeat a direction without visual support.

Poses are relative to that option's start: x=forward meters, y=left meters, positive yaw=left degrees. path_m is traveled distance and net_m is start-to-end displacement. ok=true only means the tool ran. completion=unknown or a budget return never proves goal completion. Call stop only when the current view confirms the final destination."""

EXAMPLES = """Example: inspect a target believed to be behind the robot.
<tool_call>
{"name":"turn","arguments":{"direction":"left","degrees":90}}
</tool_call>

Possible result followed by the new current image:
{"schema":"turn_tool_result_v1","tool":"turn","ok":true,"direction":"left","requested_deg":90.0,"turned_deg":90.0,"steps_used":1}
<image:current-after-turn>

After inspecting that image, move only through a new local option:
<tool_call>
{"name":"run_action_head","arguments":{"target":"the doorway","where":"ahead on the left","plan":["turn slightly left toward the doorway","approach the doorway"]}}
</tool_call>

When the final destination is visibly reached:
<tool_call>
{"name":"stop","arguments":{"reason":"The instructed destination is visible at the robot's current position."}}
</tool_call>"""


def build_system_prompt() -> str:
    tool_json = json.dumps(TOOLS, ensure_ascii=False, separators=(",", ":"))
    return (
        f"{POLICY_TEXT}\n\nYou have the following tools available.\n\n"
        f"Tool call format:\n{TOOL_CALL_INSTRUCTION}\n\nTool list:\n"
        f"<tools>\n{tool_json}\n</tools>\n\n{EXAMPLES}"
    )


SYSTEM_PROMPT = build_system_prompt()
PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


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
    mismatches = {key: {"expected": wanted, "actual": value.get(key)}
                  for key, wanted in expected.items()
                  if value.get(key) != wanted}
    if mismatches:
        raise ValueError(f"Action Head-as-Tool v4 identity mismatch: {mismatches}")


# The result geometry, direction audit and multimodal memory behavior remain v3.
validate_stop_arguments = V3.validate_stop_arguments
format_tool_call = V3.format_tool_call
current_view_text = V3.current_view_text
task_text = V3.task_text
append_tool_budget = V3.append_tool_budget
wrap_degrees = V3.wrap_degrees
planned_option_direction = V3.planned_option_direction
actual_option_direction = V3.actual_option_direction
direction_alignment = V3.direction_alignment
is_image_part = V3.is_image_part
tagged_image_part = V3.tagged_image_part
strip_context_images = V3.strip_context_images
chunk_text = V3.chunk_text
file_image_part = V3.file_image_part
