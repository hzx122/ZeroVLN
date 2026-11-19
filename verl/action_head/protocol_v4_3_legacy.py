# Frozen before the 2026-09-20 v4.3 prompt replacement. Historical ablations
# import this snapshot so their prompt identities do not change transitively.
"""action expert v4.3 prompt protocol.

v4.3 keeps the v4.2 prompt and v4 execution semantics, while bounding view
adjustment to at most two consecutive ``turn`` calls.  The runtime enforces
the same limit and returns a model-visible rejection for further turn calls.
Open-loop teacher construction may use this prompt to sample the conditioning
``run_action_head`` call. A sampled ``turn`` is rejected as an unusable
candidate: it does not change the golden pose, images, trajectory, or history.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from . import protocol_v4_2 as V42


ACTION_HEAD_AS_TOOL_VERSION = "v4.3"
PROMPT_VERSION = "slow_system_prompt_v4_3"
TOOL_RESULT_SCHEMA = "action_head_tool_result_v4_3"
SLOW_CONTEXT_VERSION = "slow_context_v4_3"
BUNDLE_SCHEMA = V42.BUNDLE_SCHEMA
SLOW_DECISION_SCHEMA = "slow_tool_decision_v4_3"
IMAGE_META_KEY = V42.IMAGE_META_KEY
MAX_CONSECUTIVE_TURN_CALLS = 2

RUN_ACTION_HEAD_DESCRIPTION = V42.RUN_ACTION_HEAD_DESCRIPTION
RUN_ACTION_HEAD_PARAMETERS = V42.RUN_ACTION_HEAD_PARAMETERS
TURN_DESCRIPTION = (
    V42.TURN_DESCRIPTION + " At most two turn calls may be made consecutively."
)
TURN_PARAMETERS = V42.TURN_PARAMETERS
STOP_DESCRIPTION = V42.STOP_DESCRIPTION
STOP_PARAMETERS = V42.STOP_PARAMETERS

TOOLS = (
    {"name": "run_action_head", "description": RUN_ACTION_HEAD_DESCRIPTION,
     "parameters": RUN_ACTION_HEAD_PARAMETERS},
    {"name": "turn", "description": TURN_DESCRIPTION,
     "parameters": TURN_PARAMETERS},
    {"name": "stop", "description": STOP_DESCRIPTION,
     "parameters": STOP_PARAMETERS},
)

TOOL_CALL_INSTRUCTION = V42.TOOL_CALL_INSTRUCTION
CONTEXT_AND_POLICY = V42.CONTEXT_AND_POLICY + """

Do not call turn more than twice in a row. After two consecutive turn calls, the next tool must be run_action_head or stop. A run_action_head call resets the consecutive-turn count. A further turn call is rejected without rotating the robot and its ToolResult will ask you to choose run_action_head or stop."""
EXAMPLES = V42.EXAMPLES


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
        raise ValueError(f"Action Head-as-Tool v4.3 identity mismatch: {mismatches}")


validate_stop_arguments = V42.validate_stop_arguments
format_tool_call = V42.format_tool_call
current_view_text = V42.current_view_text
task_text = V42.task_text
append_tool_budget = V42.append_tool_budget
wrap_degrees = V42.wrap_degrees
planned_option_direction = V42.planned_option_direction
actual_option_direction = V42.actual_option_direction
direction_alignment = V42.direction_alignment
is_image_part = V42.is_image_part
tagged_image_part = V42.tagged_image_part
strip_context_images = V42.strip_context_images
chunk_text = V42.chunk_text
file_image_part = V42.file_image_part


def initial_messages(instruction, slow_image_paths):
    from . import protocol_v3 as V3
    messages = V3.initial_messages(instruction, slow_image_paths)
    messages[0]['content'] = [{'type': 'text', 'text': SYSTEM_PROMPT}]
    return messages


def append_tool_exchange(messages, *, completion, summary, frames):
    from . import protocol_v3 as V3
    return V3.append_tool_exchange(messages, completion=completion, summary=summary, frames=frames)
