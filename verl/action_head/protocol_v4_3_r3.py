"""R2 plus the v3 structural template and a complete execution/replan example.

Keep r2 policy, stop evidence requirements, schemas and execution unchanged.
This is an opt-in revision; historical r2 evaluations retain their exact bytes.
"""
import hashlib

from . import protocol_v3 as _V3
from . import protocol_v4_3_r2 as _R2
from .protocol_v4_3_r2 import *  # noqa: F401,F403


PROMPT_VERSION = "slow_system_prompt_v4_3_r3"
TOOL_CALL_INSTRUCTION = _V3.TOOL_CALL_INSTRUCTION + "\n\n" + _R2.TOOL_CALL_INSTRUCTION

# Reuse the complete v3 call/result/replan exchange, but retain r2's stronger
# arrival/stop examples instead of appending the old v3 stop sentence.
_V3_EXCHANGE, _separator, _old_stop = _V3.CORRECTION_EXAMPLE.partition(
    "\n\nWhen the final destination is visibly reached:")
assert _separator, "v3 correction example boundary changed"
WORKED_EXAMPLE = (
    "Worked execution and replanning example: these are two separate decisions, "
    "with a tool result and new observations between them. Emit only the next "
    "single tool call, not this entire exchange.\n\n"
    + _V3_EXCHANGE.replace("Example:", "First decision:", 1).replace(
        _V3.TOOL_RESULT_SCHEMA, TOOL_RESULT_SCHEMA)
)
EXAMPLES = WORKED_EXAMPLE + "\n\n" + _R2.EXAMPLES


def build_system_prompt():
    # Reuse the frozen r2 builder; only the two requested blocks differ.
    base = _R2.build_system_prompt()
    assert base.count(_R2.TOOL_CALL_INSTRUCTION) == base.count(_R2.EXAMPLES) == 1
    return base.replace(_R2.TOOL_CALL_INSTRUCTION, TOOL_CALL_INSTRUCTION, 1).replace(
        _R2.EXAMPLES, EXAMPLES, 1)


SYSTEM_PROMPT = build_system_prompt()
PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def protocol_identity():
    return {**_R2.protocol_identity(), "prompt_version": PROMPT_VERSION,
            "prompt_sha256": PROMPT_SHA256}


def assert_protocol_identity(value):
    mismatches = {key: {"expected": wanted, "actual": value.get(key)}
                  for key, wanted in protocol_identity().items()
                  if value.get(key) != wanted}
    if mismatches:
        raise ValueError(f"Action Head-as-Tool v4.3 r3 identity mismatch: {mismatches}")


def initial_messages(instruction, slow_image_paths):
    messages = _R2.initial_messages(instruction, slow_image_paths)
    messages[0]['content'] = [{'type': 'text', 'text': SYSTEM_PROMPT}]
    return messages
