"""Inference-only action expert v4.4 prompt protocol.

v4.4 keeps v4.3 execution semantics and adds explicit arrival, navigation-only,
and execution-feedback rules to the slow-system prompt.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from . import protocol_v4_3_legacy as V43


ACTION_HEAD_AS_TOOL_VERSION = "v4.4"
PROMPT_VERSION = "slow_system_prompt_v4_4"
TOOL_RESULT_SCHEMA = "action_head_tool_result_v4_4"
SLOW_CONTEXT_VERSION = "slow_context_v4_4"
BUNDLE_SCHEMA = V43.BUNDLE_SCHEMA
SLOW_DECISION_SCHEMA = "slow_tool_decision_v4_4"
IMAGE_META_KEY = V43.IMAGE_META_KEY
MAX_CONSECUTIVE_TURN_CALLS = V43.MAX_CONSECUTIVE_TURN_CALLS

RUN_ACTION_HEAD_DESCRIPTION = V43.RUN_ACTION_HEAD_DESCRIPTION
RUN_ACTION_HEAD_PARAMETERS = V43.RUN_ACTION_HEAD_PARAMETERS
TURN_DESCRIPTION = V43.TURN_DESCRIPTION
TURN_PARAMETERS = V43.TURN_PARAMETERS
STOP_DESCRIPTION = (
    "End the task only when the robot is physically close enough to the "
    "instructed final destination. Merely seeing or recognizing it in the "
    "current view is not sufficient evidence of arrival."
)
STOP_PARAMETERS = V43.STOP_PARAMETERS

TOOLS = (
    {"name": "run_action_head", "description": RUN_ACTION_HEAD_DESCRIPTION,
     "parameters": RUN_ACTION_HEAD_PARAMETERS},
    {"name": "turn", "description": TURN_DESCRIPTION,
     "parameters": TURN_PARAMETERS},
    {"name": "stop", "description": STOP_DESCRIPTION,
     "parameters": STOP_PARAMETERS},
)

TOOL_CALL_INSTRUCTION = V43.TOOL_CALL_INSTRUCTION

CRITICAL_RULES = """Critical navigation and arrival rules:

The robot must be physically close enough to the instructed final destination before calling stop. Seeing, recognizing, or facing the target only proves that it has been identified; it does not prove that the robot has reached it. A target visible across a room, at the far end of a corridor, high on a wall, or on the ceiling is not reached merely because it appears in the image. For a wall or ceiling target, navigate to its corresponding nearby floor position. If there is any uncertainty about whether the robot is sufficiently close, continue navigating and do not call stop.

The available tools control navigation only. They cannot pick up, carry, move, or place an object. For instructions such as moving an object from A to B, placing an object beside Y, or taking an object to B, interpret B or Y as the final navigation destination. Do not navigate back to the source object unless that is genuinely needed to identify the route. Never claim that the robot picked up, carried, moved, or placed an object.

Actual motion and the returned blocked, collision, no_progress, path_m, execution trace, and latest RGB are more reliable than the robot's earlier expectation. After every tool return, compare the previous target, where, and plan with the actual execution trace and final observation. If execution deviated, was blocked, collided, or made no progress, the next navigation option must revise its target, where, or plan from the current state. Do not assume that the Action Head executed the previous plan correctly, and do not repeat an unchanged failed option."""

CONTEXT_AND_POLICY = V43.CONTEXT_AND_POLICY.replace(
    "Call stop only if the current visual observation confirms that the final "
    "destination has been reached.",
    "Call stop only if the current visual observation shows that the robot is "
    "physically close enough to the final destination, not merely viewing it "
    "from a distance.",
).replace(
    "Call stop only when the current visual evidence confirms that the final "
    "destination described by the instruction has been reached.",
    "Call stop only when the current visual evidence shows that the robot is "
    "physically close enough to the final destination described by the instruction.",
)

EXAMPLES = """The following examples demonstrate call syntax only. They do not prescribe when a tool should be used, a navigation strategy, or a sequence of actions.

run_action_head syntax:
<tool_call>
{"name":"run_action_head","arguments":{"target":"the doorway","where":"ahead on the left","plan":["turn slightly left toward the doorway","approach the doorway"]}}
</tool_call>

turn syntax:
<tool_call>
{"name":"turn","arguments":{"direction":"left","degrees":45}}
</tool_call>

stop syntax, only after physically reaching the destination:
<tool_call>
{"name":"stop","arguments":{"reason":"The robot is now physically beside the instructed destination after approaching it; this is not based only on seeing it from a distance."}}
</tool_call>"""


def build_system_prompt() -> str:
    tool_json = json.dumps(TOOLS, ensure_ascii=False, separators=(",", ":"))
    return (
        "You are an indoor navigation robot. Complete the user's navigation "
        "instruction by calling exactly one tool at a time.\n\n"
        f"{CRITICAL_RULES}\n\n"
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
        raise ValueError(f"Action Head-as-Tool v4.4 identity mismatch: {mismatches}")


validate_stop_arguments = V43.validate_stop_arguments
format_tool_call = V43.format_tool_call
current_view_text = V43.current_view_text
task_text = V43.task_text
append_tool_budget = V43.append_tool_budget
wrap_degrees = V43.wrap_degrees
planned_option_direction = V43.planned_option_direction
actual_option_direction = V43.actual_option_direction
direction_alignment = V43.direction_alignment
is_image_part = V43.is_image_part
tagged_image_part = V43.tagged_image_part
strip_context_images = V43.strip_context_images
chunk_text = V43.chunk_text
file_image_part = V43.file_image_part
