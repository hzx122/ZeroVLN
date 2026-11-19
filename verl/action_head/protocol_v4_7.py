"""Prepared recovery ablation: v4.3 until repeated impaired translation.

Reuse v4.5's detector and policy, but put both only in a triggered request
copy; normal requests from the same input history match v4.3 exactly.
"""
import json
from . import protocol_v4_3_legacy as BASE
from .protocol_v4_5 import recovery_state, RECOVERY_POLICY

BUNDLE_SCHEMA = BASE.BUNDLE_SCHEMA
IMAGE_META_KEY = BASE.IMAGE_META_KEY
MAX_CONSECUTIVE_TURN_CALLS = BASE.MAX_CONSECUTIVE_TURN_CALLS
RUN_ACTION_HEAD_DESCRIPTION = BASE.RUN_ACTION_HEAD_DESCRIPTION
RUN_ACTION_HEAD_PARAMETERS = BASE.RUN_ACTION_HEAD_PARAMETERS
TURN_DESCRIPTION = BASE.TURN_DESCRIPTION
TURN_PARAMETERS = BASE.TURN_PARAMETERS
STOP_DESCRIPTION = BASE.STOP_DESCRIPTION
STOP_PARAMETERS = BASE.STOP_PARAMETERS
TOOLS = BASE.TOOLS
validate_stop_arguments = BASE.validate_stop_arguments
format_tool_call = BASE.format_tool_call
current_view_text = BASE.current_view_text
task_text = BASE.task_text
append_tool_budget = BASE.append_tool_budget
wrap_degrees = BASE.wrap_degrees
planned_option_direction = BASE.planned_option_direction
actual_option_direction = BASE.actual_option_direction
direction_alignment = BASE.direction_alignment
is_image_part = BASE.is_image_part
tagged_image_part = BASE.tagged_image_part
chunk_text = BASE.chunk_text
file_image_part = BASE.file_image_part
SYSTEM_PROMPT = BASE.SYSTEM_PROMPT
PROMPT_SHA256 = BASE.PROMPT_SHA256
build_system_prompt = BASE.build_system_prompt
ACTION_HEAD_AS_TOOL_VERSION = 'v4.7'
PROMPT_VERSION = 'slow_system_prompt_v4_7'
TOOL_RESULT_SCHEMA = 'action_head_tool_result_v4_7'
SLOW_CONTEXT_VERSION = 'slow_context_v4_7'
SLOW_DECISION_SCHEMA = 'slow_tool_decision_v4_7'


def protocol_identity():
    return dict(action_head_as_tool_version=ACTION_HEAD_AS_TOOL_VERSION,
                prompt_version=PROMPT_VERSION, prompt_sha256=PROMPT_SHA256,
                tool_result_schema=TOOL_RESULT_SCHEMA,
                slow_context_version=SLOW_CONTEXT_VERSION)


def assert_protocol_identity(value):
    if any(value.get(k) != v for k, v in protocol_identity().items()):
        raise ValueError('Action Head-as-Tool v4.7 identity mismatch')


def strip_context_images(messages, keep_slow=8):
    copied = BASE.strip_context_images(messages, keep_slow)
    state = recovery_state(copied)
    if not state['different_approach_needed']:
        return copied
    note = RECOVERY_POLICY + '\n\n[Recovery state]\n' + json.dumps(state, separators=(',', ':'))
    for message in reversed(copied):
        if message.get('role') == 'user' and isinstance(message.get('content'), list):
            message['content'].append(dict(type='text', text=note))
            break
    return copied
