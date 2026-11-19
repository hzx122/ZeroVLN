"""Inference ablation: retain local recovery guidance after a v4.9 retreat.

Motion and stop rules stay v4.9.
"""
import hashlib
import json
from . import protocol_v4_9 as BASE
from .protocol_v4_9 import (
    BUNDLE_SCHEMA, IMAGE_META_KEY, RUN_ACTION_HEAD_DESCRIPTION,
    RUN_ACTION_HEAD_PARAMETERS, TURN_DESCRIPTION, TURN_PARAMETERS,
    STOP_DESCRIPTION, STOP_PARAMETERS, BACKOFF_DESCRIPTION, BACKOFF_PARAMETERS,
    TOOLS, AVAILABLE_TOOLS, SYSTEM_PROMPT, PROMPT_SHA256,
    AVAILABLE_SYSTEM_PROMPT, AVAILABLE_PROMPT_SHA256,
    validate_stop_arguments, format_tool_call, current_view_text, task_text,
    append_tool_budget, wrap_degrees, planned_option_direction,
    actual_option_direction, direction_alignment, is_image_part,
    tagged_image_part, chunk_text, file_image_part, build_system_prompt)
from .backoff_recovery import backoff_state
from .protocol_v4_5 import RECOVERY_POLICY, successful_runtime_results

ACTION_HEAD_AS_TOOL_VERSION = 'v4.10'
PROMPT_VERSION = 'slow_system_prompt_v4_10'
TOOL_RESULT_SCHEMA = 'action_head_tool_result_v4_10'
SLOW_CONTEXT_VERSION = 'slow_context_v4_10'
SLOW_DECISION_SCHEMA = 'slow_tool_decision_v4_10'
POST_BACKOFF_POLICY = (
    'A retreat was executed, but no Action Head option has since made at least '
    '0.3 meters of net progress. The previous blocked approach has not become '
    'traversable merely because the robot stepped backward. Another retreat '
    'is unavailable regardless of the requested distance; changing the distance '
    'argument does not restore that ability. Use the current image to decide '
    'whether you are already beside the correct final destination and should '
    'stop. If navigation is still needed, select a visible reachable local '
    'waypoint around the obstruction before returning to the final destination. '
    'Keep the final destination fixed, but make the next run_action_head target '
    'that local waypoint rather than repeating the blocked target and plan.\n\n'
    + RECOVERY_POLICY)
POST_BACKOFF_POLICY_SHA256 = hashlib.sha256(POST_BACKOFF_POLICY.encode()).hexdigest()


def protocol_identity():
    return dict(action_head_as_tool_version=ACTION_HEAD_AS_TOOL_VERSION,
                prompt_version=PROMPT_VERSION, prompt_sha256=PROMPT_SHA256,
                tool_result_schema=TOOL_RESULT_SCHEMA,
                slow_context_version=SLOW_CONTEXT_VERSION,
                backoff_available_prompt_sha256=AVAILABLE_PROMPT_SHA256,
                post_backoff_policy_sha256=POST_BACKOFF_POLICY_SHA256)


def assert_protocol_identity(value):
    if any(value.get(k) != v for k, v in protocol_identity().items()):
        raise ValueError('Action Head-as-Tool v4.10 identity mismatch')


def strip_context_images(messages, keep_slow=8):
    copied = BASE.strip_context_images(messages, keep_slow)
    if not backoff_state(messages)['used_since_progress']:
        return copied
    last_retreat = next(raw for raw in reversed(list(successful_runtime_results(messages)))
                        if raw.get('tool') == 'move_backward')
    facts = {k:last_retreat[k] for k in ('moved_m', 'blocked', 'steps_used') if k in last_retreat}
    note = '[Post-retreat recovery]\n' + POST_BACKOFF_POLICY + '\nLast retreat execution: ' + json.dumps(facts, separators=(',', ':'))
    for message in reversed(copied):
        if message.get('role') == 'user' and isinstance(message.get('content'), list):
            message['content'].append(dict(type='text', text=note))
            break
    return copied
