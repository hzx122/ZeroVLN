"""action expert v4.3 r2 prompt implementation.

The 2026-09-20 r2 prompt replaces the original v4.3 prompt by user request.
It separates visible tool-call syntax from evidence-based arrival decisions.
Tool schemas, geometry, image handling, and the two-turn limit are unchanged.
Historical v4.3-based ablations use protocol_v4_3_legacy, not this revision.
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
PROMPT_VERSION = "slow_system_prompt_v4_3_r2"
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

TOOL_CALL_INSTRUCTION = """Your visible response must contain exactly one <tool_call>...</tool_call> block and nothing else: no explanation outside the block, Markdown fences, or Python-style calls. Inside the block, output one valid JSON object with exactly two fields, "name" and "arguments". Use only the registered tool names. "arguments" must always be a JSON object, never a string.

For stop, "arguments" must contain only "reason", a string. Give one short sentence identifying the final destination and the observed spatial evidence that the robot is already there; do not merely assert completion or invent evidence. Follow the complete valid examples below."""

CONTEXT_AND_POLICY = """The latest slow observation is the robot's current first-person view. Retain at most eight slow images including the current view, plus up to three intermediate images from the latest run_action_head option. Older images may be labeled placeholders; retained text remains history, not a current observation.

During run_action_head, the Action Head repeatedly observes live RGB, predicts a short trajectory, and executes its prefix. Returned chunk poses and RGB show the actual execution. Poses are relative to the option's start: x is forward in meters, y is left in meters, and positive yaw is left rotation in degrees. path_m is traveled distance; net_m is start-to-end displacement.

After each tool, assess its actual result and latest RGB before choosing the next action. direction_alignment compares intended and actual option direction; it is not an instruction to reverse direction. If execution was mismatched, blocked, or made no progress, replan from the current view and trace. Silently issue a normal local target, where, and plan: do not narrate correction, recovery, failure, or previous execution in those arguments. Do not mechanically repeat or reverse a direction without current visual support.

Stop only when you have identified the instructed FINAL destination, rather than an intermediate landmark or a similar object, and the current observation interpreted with recent execution supports that the robot is already at the location or spatial relation required by the instruction. Seeing or facing the destination from elsewhere is not arrival. Completing a local option, ok=true, completion=unknown, collision, blockage, or a budget notice alone is not evidence of arrival. Never invent distances, hidden goal coordinates, future routes, or evaluator facts.

If arrival is not established, use the remaining budget for a purposeful action. Use turn when changing the viewing direction can resolve missing visual information; inspect the returned RGB before deciding again. Use run_action_head when a visually supported approach involving translation is needed; that option may include reorientation. Keep the final destination fixed while choosing local waypoints. When arrival is established, stop promptly: do not require an extra movement or turn solely to confirm it.

Do not call turn more than twice in a row. A run_action_head call resets the consecutive-turn count. After two consecutive turns, stop only if the arrival condition is met; otherwise choose a visually supported run_action_head option. The turn limit never establishes arrival. A further turn call is rejected without rotating the robot."""

EXAMPLES = """Examples illustrate valid syntax and decisions under the stated observations, not a fixed action sequence. Apply the actual instruction and current evidence.

View adjustment: a different viewing direction is needed to identify the next local target, and a turn is available:
<tool_call>
{"name":"turn","arguments":{"direction":"left","degrees":45}}
</tool_call>

Visible but not reached: the instruction is to go beside the wooden table beneath the hanging lamp. The robot is still at the doorway; the table is across the room and the intervening floor is visibly clear:
<tool_call>
{"name":"run_action_head","arguments":{"target":"the wooden table","where":"across the room beneath the hanging lamp","plan":["approach the table along the visible clear floor"]}}
</tool_call>

Reached: for the same instruction, the current view and just-completed approach show that the robot is already beside the specified table beneath the lamp. No further approach is needed:
<tool_call>
{"name":"stop","arguments":{"reason":"The current view after approaching shows the robot beside the specified wooden table beneath the hanging lamp."}}
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
