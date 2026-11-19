"""Inference-only v4.1: one-shot recovery turn after direction mismatch."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from . import protocol_v4 as V4


ACTION_HEAD_AS_TOOL_VERSION = "v4.1"
PROMPT_VERSION = "slow_system_prompt_v4_1"
TOOL_RESULT_SCHEMA = "action_head_tool_result_v4_1"
SLOW_CONTEXT_VERSION = "slow_context_v4_1"
BUNDLE_SCHEMA = V4.BUNDLE_SCHEMA
SLOW_DECISION_SCHEMA = "slow_tool_decision_v4_1"
IMAGE_META_KEY = V4.IMAGE_META_KEY

RUN_ACTION_HEAD_DESCRIPTION = V4.RUN_ACTION_HEAD_DESCRIPTION
RUN_ACTION_HEAD_PARAMETERS = V4.RUN_ACTION_HEAD_PARAMETERS
TURN_DESCRIPTION = (
    "Perform one in-place recovery rotation after the immediately preceding "
    "run_action_head reported direction_alignment.match=false. At most one turn "
    "is available for that mismatch; after it, inspect the returned RGB and use "
    "run_action_head or stop. left is counter-clockwise and right is clockwise."
)
TURN_PARAMETERS = {
    "type": "object",
    "properties": {
        "direction": {"type": "string", "enum": ["left", "right"],
                      "description": "Direction of the in-place recovery rotation."},
        "degrees": {"type": "number", "minimum": 0.5, "maximum": 180.0,
                    "description": "Recovery rotation angle from 0.5 through 180 degrees."},
    },
    "required": ["direction", "degrees"],
    "additionalProperties": False,
}
STOP_DESCRIPTION = V4.STOP_DESCRIPTION
STOP_PARAMETERS = V4.STOP_PARAMETERS
TOOLS = (
    {"name": "turn", "description": TURN_DESCRIPTION, "parameters": TURN_PARAMETERS},
    {"name": "run_action_head", "description": RUN_ACTION_HEAD_DESCRIPTION,
     "parameters": RUN_ACTION_HEAD_PARAMETERS},
    {"name": "stop", "description": STOP_DESCRIPTION, "parameters": STOP_PARAMETERS},
)
TOOL_CALL_INSTRUCTION = V4.TOOL_CALL_INSTRUCTION

POLICY_TEXT = """You are an indoor navigation robot. Complete the user's instruction by calling exactly one tool at a time.

The latest slow observation is the current first-person view. Up to eight slow observations remain in their original conversation positions. During run_action_head, the fast system repeatedly observes live RGB, predicts a short trajectory, and executes its prefix. Its result pairs every chunk's actual end pose with the RGB observed there; older images may become labeled placeholders.

Use run_action_head for normal navigation and every action involving translation. The turn tool is a one-shot recovery tool, not a normal navigation or exploration tool. It is available only when the immediately preceding run_action_head result has direction_alignment.match=false. At most one turn may follow that mismatch. After turn, inspect the returned RGB and call run_action_head or stop; never call turn twice in a row. A turn can be up to 180 degrees.

Use only the instruction, visible images, history, and returned non-oracle facts. If direction_alignment.match=false, decide from the final RGB and trace whether the one recovery turn is useful; it is optional. Do not mechanically invert a direction without visual support. Do not describe correction, recovery, deviation, or failure inside run_action_head arguments; give a normal target, where, and plan for the current state.

Option poses are relative to that option's start: x=forward meters, y=left meters, positive yaw=left degrees. path_m is traveled distance and net_m is start-to-end displacement. ok=true only means the tool ran. completion=unknown or a budget return never proves goal completion. Call stop only when the current view confirms the final destination."""

EXAMPLES = """After a mismatched option result:
{"schema":"action_head_tool_result_v4_1","tool":"run_action_head","ok":true,"direction_alignment":{"planned":"left","actual":"right","match":false}}

One recovery turn is now available:
<tool_call>
{"name":"turn","arguments":{"direction":"left","degrees":90}}
</tool_call>

Possible result followed by the new current image:
{"schema":"turn_tool_result_v2","tool":"turn","ok":true,"direction":"left","requested_deg":90.0,"turned_deg":90.0,"steps_used":1,"turn_reason":"previous_option_direction_mismatch","turn_available_after_result":false}
<image:current-after-turn>

After inspecting it, return to local navigation:
<tool_call>
{"name":"run_action_head","arguments":{"target":"the doorway","where":"ahead on the left","plan":["turn slightly left toward the doorway","approach the doorway"]}}
</tool_call>"""


def build_system_prompt() -> str:
    tool_json = json.dumps(TOOLS, ensure_ascii=False, separators=(",", ":"))
    return (f"{POLICY_TEXT}\n\nYou have the following tools available.\n\n"
            f"Tool call format:\n{TOOL_CALL_INSTRUCTION}\n\nTool list:\n"
            f"<tools>\n{tool_json}\n</tools>\n\n{EXAMPLES}")


SYSTEM_PROMPT = build_system_prompt()
PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def protocol_identity() -> dict[str, str]:
    return {"action_head_as_tool_version": ACTION_HEAD_AS_TOOL_VERSION,
            "prompt_version": PROMPT_VERSION, "prompt_sha256": PROMPT_SHA256,
            "tool_result_schema": TOOL_RESULT_SCHEMA,
            "slow_context_version": SLOW_CONTEXT_VERSION}


def assert_protocol_identity(value: dict[str, Any]) -> None:
    expected = protocol_identity()
    mismatches = {key: {"expected": wanted, "actual": value.get(key)}
                  for key, wanted in expected.items() if value.get(key) != wanted}
    if mismatches:
        raise ValueError(f"Action Head-as-Tool v4.1 identity mismatch: {mismatches}")


validate_stop_arguments = V4.validate_stop_arguments
format_tool_call = V4.format_tool_call
current_view_text = V4.current_view_text
task_text = V4.task_text
append_tool_budget = V4.append_tool_budget
wrap_degrees = V4.wrap_degrees
planned_option_direction = V4.planned_option_direction
actual_option_direction = V4.actual_option_direction
direction_alignment = V4.direction_alignment
is_image_part = V4.is_image_part
tagged_image_part = V4.tagged_image_part
strip_context_images = V4.strip_context_images
chunk_text = V4.chunk_text
file_image_part = V4.file_image_part
