"""Prepare a fresh goal decision from existing model-visible observations.

Used by the isolated v4.8 candidate protocol. This changes
only the request copy, never the episode, tool state, budgets, or saved history.
"""
from copy import deepcopy
import json

from . import protocol_v4_3_legacy as BASE
from .protocol_v4_5 import recovery_state


def current_goal_context(messages, keep_slow=8):
    history = list(messages)
    fallback = BASE.strip_context_images(history, keep_slow)
    # Use the original user task, not a target inferred from assistant plans.
    task = None
    latest_result = None
    current_image = None
    for message in history:
        if message.get('role') != 'user' or not isinstance(message.get('content'), list):
            continue
        for part in message['content']:
            if part.get('type') == 'text':
                text = part.get('text', '')
                marker = 'Navigation instruction: '
                if task is None and marker in text and not text.startswith('[Tool result]\n'):
                    task = text[text.index(marker):]
                if text.startswith('[Tool result]\n'):
                    latest_result = text
            elif BASE.is_image_part(part):
                meta = part.get(BASE.IMAGE_META_KEY) or {'kind': 'slow'}
                if meta.get('kind') in ('slow', 'final_slow'):
                    current_image = part
    if task is None or latest_result is None or current_image is None:
        return fallback
    systems = [deepcopy(m) for m in history if m.get('role') == 'system']
    if not systems:
        return fallback
    # The existing detector whitelists actual runtime motion/turn facts. It
    # contributes recent events only, without adding its recovery policy.
    state = recovery_state(history)
    content = [deepcopy(current_image),
               {'type': 'text', 'text': BASE.current_view_text() + '\n\n' + task},
               {'type': 'text', 'text': latest_result},
               {'type': 'text', 'text': '[Recent executed motion facts]\n' +
                json.dumps(state['recent_execution'], separators=(',', ':'))}]
    return BASE.strip_context_images(systems + [{'role': 'user', 'content': content}], keep_slow)
