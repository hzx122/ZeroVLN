"""Append-only slow observations and immutable trajectory-map snapshots."""
from __future__ import annotations

import copy
import hashlib

from . import protocol_v6 as _V6
from .protocol_v6 import *  # noqa: F401,F403
from .protocol_v3 import IMAGE_META_KEY
from .trajectory_map import MAP_CONTEXT_TEXT, inject_trajectory_map

ACTION_HEAD_AS_TOOL_VERSION = "v8"
PROMPT_VERSION = "slow_system_prompt_v8_continuous"
TOOL_RESULT_SCHEMA = "action_head_tool_result_v8"
SLOW_CONTEXT_VERSION = "slow_context_v8_append_only"
SLOW_DECISION_SCHEMA = "slow_tool_decision_v8"

# Keep the established v6 policy verbatim except for statements contradicted
# by append-only slow history and final-frame-only tool observations.
CONTEXT_AND_POLICY = _V6.CONTEXT_AND_POLICY
for old, new in (
    ("one latest actual trajectory map immediately before the current first-person image",
     "one new actual trajectory-map snapshot alongside the latest tool results"),
    (", and corresponding RGB, in execution order.",
     ", in execution order, followed by the final chunk's RGB."),
    ("The last chunk's image shows the state at the end of the call; earlier images show intermediate states.",
     "The returned image shows the state at the end of the call."),
    ("At most 8 slow observations and 3 intermediate images from the latest run_action_head call are retained. One latest trajectory map is added in its own slot, for a total of at most 12 images.",
     "All slow observations and historical trajectory-map snapshots are retained in chronological order."),
    ("Intermediate images from the latest call are counted separately; images from older calls are not used to fill unused slots.",
     "Fast intermediate images are not returned."),
    ("Images outside the retention window are replaced in place with omission markers.",
     "No historical observations are removed or replaced with omission markers."),
    (" Omission markers are not visible images; do not invent their details.", ""),
):
    CONTEXT_AND_POLICY = _V6._replace_once(CONTEXT_AND_POLICY, old, new)
SYSTEM_PROMPT = _V6.SYSTEM_PROMPT.replace(_V6.CONTEXT_AND_POLICY, CONTEXT_AND_POLICY).replace(
    _V6.TOOL_RESULT_SCHEMA, TOOL_RESULT_SCHEMA)
EXAMPLES = _V6.EXAMPLES.replace(_V6.TOOL_RESULT_SCHEMA, TOOL_RESULT_SCHEMA)
PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()


def build_system_prompt():
    return SYSTEM_PROMPT


def protocol_identity():
    return dict(_V6.protocol_identity(), action_head_as_tool_version="v8",
                prompt_version=PROMPT_VERSION, prompt_sha256=PROMPT_SHA256,
                tool_result_schema=TOOL_RESULT_SCHEMA, slow_context_version=SLOW_CONTEXT_VERSION)


def assert_protocol_identity(value):
    if any(value.get(k) != v for k, v in protocol_identity().items()):
        raise ValueError("v8 continuous context identity mismatch")


def initial_messages(instruction, slow_image_paths):
    messages = _V6.initial_messages(instruction, slow_image_paths)
    messages[0]["content"] = [{"type": "text", "text": SYSTEM_PROMPT}]
    return messages


def strip_context_images(messages, keep_slow=None):
    """Remove private tags only; reject fast images instead of silently pruning."""
    result = copy.deepcopy(messages)
    for message in result:
        if not isinstance(message.get("content"), list):
            continue
        for part in message["content"]:
            meta = part.pop(IMAGE_META_KEY, {})
            if meta.get("kind") == "fast_intermediate":
                raise ValueError("v8 runtime leaked a fast intermediate frame")
    return result


def append_map_snapshot(messages, state):
    """Append once per decision to canonical history, never rewrite old messages."""
    # Only the unsent tool-result suffix can be modified. Already sampled
    # messages, including all previous snapshots, remain byte-for-byte intact.
    start = next((i + 1 for i in range(len(messages) - 1, -1, -1)
                  if messages[i]["role"] == "assistant"), 0)
    suffix = messages[start:]
    has_image = any(part.get("type") in ("image", "image_url")
                    for msg in suffix if isinstance(msg.get("content"), list)
                    for part in msg["content"])
    if has_image:
        messages[start:] = inject_trajectory_map(suffix, state)
    else:
        messages.append({"role": "user", "content": [
            {"type": "text", "text": MAP_CONTEXT_TEXT},
            {"type": "image", "image": state.render()},
        ]})
