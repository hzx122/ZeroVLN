"""Inference-only action expert v4.2 prompt protocol.

v4.2 keeps v4 execution semantics and the frozen D16-v3 action expert.  It only
removes policy-heavy examples and clarifies the roles of the translation and
view-adjustment tools in the slow-system prompt.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from . import protocol_v4 as V4


ACTION_HEAD_AS_TOOL_VERSION = "v4.2"
PROMPT_VERSION = "slow_system_prompt_v4_2"
TOOL_RESULT_SCHEMA = "action_head_tool_result_v4_2"
SLOW_CONTEXT_VERSION = "slow_context_v4_2"
BUNDLE_SCHEMA = V4.BUNDLE_SCHEMA
SLOW_DECISION_SCHEMA = "slow_tool_decision_v4_2"
IMAGE_META_KEY = V4.IMAGE_META_KEY

RUN_ACTION_HEAD_DESCRIPTION = (
    "Delegate one local navigation option involving translation to the Action "
    "Head. Provide a target grounded in the current or retained observations, "
    "its observed spatial relation, and one to three freely worded semantic "
    "motion stages. The option may include reorientation as part of the "
    "movement. During execution, the Action Head repeatedly observes live RGB, "
    "predicts a short trajectory, and executes its prefix. Do not provide "
    "primitive actions, coordinates, a future trajectory, goal distance, or a "
    "safety budget."
)
RUN_ACTION_HEAD_PARAMETERS = V4.RUN_ACTION_HEAD_PARAMETERS

TURN_DESCRIPTION = (
    "Rotate the robot in place to adjust its viewing direction and obtain a new "
    "first-person observation. The robot does not translate during this tool "
    "call. left is counter-clockwise and right is clockwise. The tool returns "
    "the actual rotation and the latest first-person RGB."
)
TURN_PARAMETERS = V4.TURN_PARAMETERS
STOP_DESCRIPTION = V4.STOP_DESCRIPTION
STOP_PARAMETERS = V4.STOP_PARAMETERS

# Keep the descriptions in the same order in which the prompt discusses them.
TOOLS = (
    {"name": "run_action_head", "description": RUN_ACTION_HEAD_DESCRIPTION,
     "parameters": RUN_ACTION_HEAD_PARAMETERS},
    {"name": "turn", "description": TURN_DESCRIPTION,
     "parameters": TURN_PARAMETERS},
    {"name": "stop", "description": STOP_DESCRIPTION,
     "parameters": STOP_PARAMETERS},
)

TOOL_CALL_INSTRUCTION = V4.TOOL_CALL_INSTRUCTION

CONTEXT_AND_POLICY = """The latest slow observation is the robot's current first-person view. Up to eight earlier slow observations remain in their original conversation positions.

During run_action_head, the Action Head repeatedly observes live RGB, predicts a short trajectory, and executes its prefix. The tool result pairs every executed chunk's actual end pose with the RGB observed at that pose. Images from the latest run_action_head call remain visible. Images from older calls may be replaced by labeled placeholders.

After every run_action_head result, compare the requested option with the actual execution trace and the latest visual observation.

direction_alignment summarizes whether the complete option's actual direction matched its planned direction. It describes the previous execution and does not prescribe which tool to call next.

If the previous option was not executed faithfully, make a new decision from the current state. Depending on the current situation, you may call run_action_head or turn. Call stop only if the current visual observation confirms that the final destination has been reached.

Use run_action_head for a local navigation option involving translation. The Action Head may reorient the robot as part of that local option.

Use turn primarily to adjust the robot's viewing direction and obtain a new first-person observation without changing its position. After turn, inspect the returned RGB before deciding what to do next.

After any tool executes, inspect its returned execution result and latest RGB before making the next decision.

Option poses returned by run_action_head are relative to that option's start. x is forward in meters, y is left in meters, and positive yaw is left rotation in degrees. path_m is the traveled distance, and net_m is the start-to-end displacement.

ok=true only means that the tool executed. completion=unknown or a budget-related return does not prove that the navigation goal has been reached.

Call stop only when the current visual evidence confirms that the final destination described by the instruction has been reached. Otherwise, continue with an appropriate available tool."""

EXAMPLES = """The following examples demonstrate call syntax only. They do not prescribe when a tool should be used, a navigation strategy, or a sequence of actions.

run_action_head syntax:
<tool_call>
{"name":"run_action_head","arguments":{"target":"the doorway","where":"ahead on the left","plan":["turn slightly left toward the doorway","approach the doorway"]}}
</tool_call>

turn syntax:
<tool_call>
{"name":"turn","arguments":{"direction":"left","degrees":45}}
</tool_call>

stop syntax:
<tool_call>
{"name":"stop","arguments":{"reason":"The instructed destination is confirmed in the current view."}}
</tool_call>"""


def build_system_prompt() -> str:
    tool_json = json.dumps(TOOLS, ensure_ascii=False, separators=(",", ":"))
    return (
        "You are an indoor navigation robot. Complete the user's navigation "
        "instruction by calling exactly one tool at a time.\n\n"
        "You have the following tools available.\n\n"
        f"Tool list:\n<tools>\n{tool_json}\n</tools>\n\n"
        f"{CONTEXT_AND_POLICY}\n\n"
        f"Tool call format:\n{TOOL_CALL_INSTRUCTION}\n\n{EXAMPLES}"
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
        raise ValueError(f"Action Head-as-Tool v4.2 identity mismatch: {mismatches}")


# v4.2 changes prompt/tool descriptions only; geometry and memory stay on v4.
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
