"""v4.5: recovery-only ablation over frozen v4.3, with factual decision memory.

No extra model calls, action overrides, goal oracle, or stop gate. The request
copy gets a bounded summary of already-visible tool results. History stays intact.
"""
from __future__ import annotations

import hashlib
import json
import math
from . import protocol_v4_3_legacy as BASE

# Reuse the complete v4.3 image, geometry and tool contracts.
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
ACTION_HEAD_AS_TOOL_VERSION = 'v4.5'
PROMPT_VERSION = 'slow_system_prompt_v4_5'
TOOL_RESULT_SCHEMA = 'action_head_tool_result_v4_5'
SLOW_CONTEXT_VERSION = 'slow_context_v4_5'
SLOW_DECISION_SCHEMA = 'slow_tool_decision_v4_5'
LOW_PROGRESS_M = 0.15
RECOVERY_POLICY = '''Recovery decisions:
A recovery_state summarizes recent actual motion, not distance to the goal. Keep the instructed final destination and its identifying relations fixed while changing local waypoints. When repeated attempts are blocked or barely translate, inspect the current image for a different open approach: first navigate toward visible free space or a doorway around the obstruction, then approach the destination. Use the local free-space waypoint as run_action_head target and describe only the next executable movement, not an entire route through an obstacle. A turn can reveal a route but cannot move the robot out of a blocked location. After turning, use the new view to choose a translation route rather than undoing the turn without new evidence. Do not infer arrival or failure merely from collision, no progress, or exhausting a direction. If already beside the correct destination, stop normally; otherwise use a different visually supported approach. Express the decision directly as one tool call.'''
SYSTEM_PROMPT = BASE.SYSTEM_PROMPT + '\n\n' + RECOVERY_POLICY
PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()


def build_system_prompt():
    return SYSTEM_PROMPT


def protocol_identity():
    return dict(action_head_as_tool_version=ACTION_HEAD_AS_TOOL_VERSION,
                prompt_version=PROMPT_VERSION, prompt_sha256=PROMPT_SHA256,
                tool_result_schema=TOOL_RESULT_SCHEMA,
                slow_context_version=SLOW_CONTEXT_VERSION)


def assert_protocol_identity(value):
    expected = protocol_identity()
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('Action Head-as-Tool v4.5 identity mismatch')


def successful_runtime_results(messages):
    """Yield parsed successful runtime feedback, never assistant claims."""
    for message in messages:
        if message.get('role') != 'user':
            continue
        for part in message.get('content', []):
            if not isinstance(part, dict) or part.get('type') != 'text':
                continue
            text = part.get('text', '')
            if not text.startswith('[Tool result]\n'):
                continue
            try:
                raw, _ = json.JSONDecoder().raw_decode(text.split('\n', 1)[1].lstrip())
            except (ValueError, TypeError):
                continue
            if not isinstance(raw, dict) or raw.get('ok') is not True:
                continue
            yield raw


def recovery_state(messages):
    """Whitelist only facts in runtime-authored tool messages, never assistant claims."""
    events = []
    for raw in successful_runtime_results(messages):
        if raw.get('tool') == 'run_action_head':
            net = raw.get('net_m')
            if not isinstance(net, (int, float)) or not math.isfinite(net):
                continue
            events.append(dict(tool='run_action_head', net_m=round(net, 3),
                               blocked=bool(raw.get('blocked')),
                               collision=bool(raw.get('collision'))))
        elif raw.get('tool') == 'turn':
            degrees = raw.get('turned_deg')
            if isinstance(degrees, (int, float)) and math.isfinite(degrees):
                events.append(dict(tool='turn', direction=raw.get('direction'),
                                   turned_deg=degrees))
    recent = events[-6:]
    stalled = 0
    # Turns do not reset evidence of repeated failed translation.
    for event in reversed(events):
        if event['tool'] == 'turn':
            continue
        if event['net_m'] < LOW_PROGRESS_M or event['blocked']:
            stalled += 1
        else:
            break
    return dict(recent_execution=recent, consecutive_impaired_options=stalled,
                different_approach_needed=stalled >= 2)


def strip_context_images(messages, keep_slow=8):
    copied = BASE.strip_context_images(messages, keep_slow)
    state = recovery_state(copied)
    if not state['recent_execution']:
        return copied
    note = '[Recovery state]\n' + json.dumps(state, separators=(',', ':'))
    if state['different_approach_needed']:
        note += ('\nRepeated impaired translation: choose a different visible local '
                 'approach to the same destination. Turning alone does not establish '
                 'translation progress; collision does not establish arrival.')
    # Modify only this request copy; no growing synthetic history or extra turn.
    for message in reversed(copied):
        if message.get('role') == 'user' and isinstance(message.get('content'), list):
            message['content'].append(dict(type='text', text=note))
            break
    return copied
