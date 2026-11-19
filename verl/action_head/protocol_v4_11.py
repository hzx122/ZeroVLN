"""English ReAct harness; v4.3 motion semantics under a separate identity.

The collaboratively edited text is frozen in the former experimental ReAct
module. Only the example's result-schema label changes for this public version.
"""
import hashlib

from . import protocol_v4_3_react as _REACT
from .protocol_v4_3_react import *  # noqa: F401,F403

ACTION_HEAD_AS_TOOL_VERSION = "v4.11"
PROMPT_VERSION = "slow_system_prompt_v4_11"
TOOL_RESULT_SCHEMA = "action_head_tool_result_v4_11"
SLOW_CONTEXT_VERSION = "slow_context_v4_11"
SLOW_DECISION_SCHEMA = "slow_tool_decision_v4_11"
EXAMPLES = _REACT.EXAMPLES.replace(_REACT.TOOL_RESULT_SCHEMA, TOOL_RESULT_SCHEMA)


def build_system_prompt():
    return _REACT.build_system_prompt().replace(_REACT.TOOL_RESULT_SCHEMA, TOOL_RESULT_SCHEMA)


SYSTEM_PROMPT = build_system_prompt()
PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def protocol_identity():
    return dict(action_head_as_tool_version=ACTION_HEAD_AS_TOOL_VERSION,
                prompt_version=PROMPT_VERSION, prompt_sha256=PROMPT_SHA256,
                tool_result_schema=TOOL_RESULT_SCHEMA,
                slow_context_version=SLOW_CONTEXT_VERSION)


def assert_protocol_identity(value):
    mismatches = {key: {"expected": wanted, "actual": value.get(key)}
                  for key, wanted in protocol_identity().items()
                  if value.get(key) != wanted}
    if mismatches:
        raise ValueError(f"Action Head-as-Tool v4.11 identity mismatch: {mismatches}")


def initial_messages(instruction, slow_image_paths):
    messages = _REACT.initial_messages(instruction, slow_image_paths)
    messages[0]["content"] = [{"type": "text", "text": SYSTEM_PROMPT}]
    return messages
