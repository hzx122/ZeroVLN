"""v4.1 ablation with optional bounded retreat after actual stalls."""
import hashlib
import json
from . import protocol_v4_1 as BASE
from .backoff_recovery import backoff_state, MIN_BACKOFF_M, MAX_BACKOFF_M

BUNDLE_SCHEMA = BASE.BUNDLE_SCHEMA
IMAGE_META_KEY = BASE.IMAGE_META_KEY
RUN_ACTION_HEAD_DESCRIPTION = BASE.RUN_ACTION_HEAD_DESCRIPTION
RUN_ACTION_HEAD_PARAMETERS = BASE.RUN_ACTION_HEAD_PARAMETERS
TURN_DESCRIPTION = BASE.TURN_DESCRIPTION
TURN_PARAMETERS = BASE.TURN_PARAMETERS
STOP_DESCRIPTION = BASE.STOP_DESCRIPTION
STOP_PARAMETERS = BASE.STOP_PARAMETERS
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
TOOLS = BASE.TOOLS
ACTION_HEAD_AS_TOOL_VERSION = 'v4.9'
PROMPT_VERSION = 'slow_system_prompt_v4_9'
TOOL_RESULT_SCHEMA = 'action_head_tool_result_v4_9'
SLOW_CONTEXT_VERSION = 'slow_context_v4_9'
SLOW_DECISION_SCHEMA = 'slow_tool_decision_v4_9'

BACKOFF_DESCRIPTION = (
    'Optionally retreat a short distance without changing heading after two '
    'consecutive run_action_head options were blocked with net displacement '
    'below 0.15 meters each. At most one retreat is available until an Action '
    'Head option makes at least 0.3 meters of net progress. Inspect the new RGB '
    'before choosing the next local target. This is not normal navigation; '
    'run_action_head and stop remain available. This attempt costs one tool call.')
BACKOFF_PARAMETERS = dict(
    type='object', properties=dict(distance=dict(
        type='number', minimum=MIN_BACKOFF_M, maximum=MAX_BACKOFF_M,
        description='Requested backward distance in meters.')),
    required=['distance'], additionalProperties=False)
AVAILABLE_TOOLS = TOOLS + (dict(name='move_backward',
                                description=BACKOFF_DESCRIPTION,
                                parameters=BACKOFF_PARAMETERS),)
_translation_rule = 'Use run_action_head for normal navigation and every action involving translation.'
_replacement = ('Use run_action_head for normal navigation. A bounded move_backward '
                'is also available now because the last two Action Head options '
                'were blocked and each made less than 0.15 meters of net progress. '
                'Retreat is optional, not evidence that the final goal is unreached. '
                'If the current view confirms the final destination, stop remains '
                'appropriate. Otherwise choose whether a short retreat would help '
                'you select a reachable local target from a new view. After retreat, '
                'use run_action_head or stop; a previous turn entitlement expires.')
assert _translation_rule in BASE.SYSTEM_PROMPT
AVAILABLE_SYSTEM_PROMPT = BASE.SYSTEM_PROMPT.replace(_translation_rule, _replacement, 1).replace(
    json.dumps(BASE.TOOLS, ensure_ascii=False, separators=(',', ':')),
    json.dumps(AVAILABLE_TOOLS, ensure_ascii=False, separators=(',', ':')), 1)
AVAILABLE_PROMPT_SHA256 = hashlib.sha256(AVAILABLE_SYSTEM_PROMPT.encode()).hexdigest()


def protocol_identity():
    return dict(action_head_as_tool_version=ACTION_HEAD_AS_TOOL_VERSION,
                prompt_version=PROMPT_VERSION, prompt_sha256=PROMPT_SHA256,
                tool_result_schema=TOOL_RESULT_SCHEMA,
                slow_context_version=SLOW_CONTEXT_VERSION,
                backoff_available_prompt_sha256=AVAILABLE_PROMPT_SHA256)


def assert_protocol_identity(value):
    if any(value.get(k) != v for k, v in protocol_identity().items()):
        raise ValueError('Action Head-as-Tool v4.9 identity mismatch')


def strip_context_images(messages, keep_slow=8):
    copied = BASE.strip_context_images(messages, keep_slow)
    if not backoff_state(messages)['available']:
        return copied
    for message in copied:
        if message.get('role') != 'system':
            continue
        content = message.get('content')
        if isinstance(content, str):
            message['content'] = content.replace(SYSTEM_PROMPT, AVAILABLE_SYSTEM_PROMPT, 1)
        elif isinstance(content, list):
            for part in content:
                if part.get('type') == 'text':
                    part['text'] = part.get('text', '').replace(SYSTEM_PROMPT, AVAILABLE_SYSTEM_PROMPT, 1)
    return copied
