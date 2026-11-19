"""Version routing for action expert without mutating frozen protocols."""
from __future__ import annotations

import math
from typing import Any


SUPPORTED_VERSIONS = ("v1", "v2", "v3", "v4", "v4.1", "v4.2", "v4.3", "v4.4", "v4.5", "v4.6", "v4.7", "v4.8", "v4.9", "v4.10", "v4.11", "v6", "v8")
DEFAULT_VERSION = "v1"


def normalize_version(value: str | None) -> str:
    version = str(value or DEFAULT_VERSION).strip().lower()
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(
            "action_head_as_tool_version must be one of "
            f"{SUPPORTED_VERSIONS}, got {value!r}")
    return version


def protocol_module(value: str) -> Any:
    version = normalize_version(value)
    if version == "v2":
        from . import protocol_v2
        return protocol_v2
    if version == "v3":
        from . import protocol_v3
        return protocol_v3
    if version == "v4":
        from . import protocol_v4
        return protocol_v4
    if version == "v4.1":
        from . import protocol_v4_1
        return protocol_v4_1
    if version == "v4.2":
        from . import protocol_v4_2
        return protocol_v4_2
    if version == "v4.3":
        from . import protocol_v4_3
        return protocol_v4_3
    if version == "v4.4":
        from . import protocol_v4_4
        return protocol_v4_4
    if version == "v4.5":
        from . import protocol_v4_5
        return protocol_v4_5
    if version == "v4.6":
        from . import protocol_v4_6
        return protocol_v4_6
    if version == "v4.8":
        from . import protocol_v4_8
        return protocol_v4_8
    if version == "v4.7":
        from . import protocol_v4_7
        return protocol_v4_7
    if version == "v4.9":
        from . import protocol_v4_9
        return protocol_v4_9
    if version == "v4.10":
        from . import protocol_v4_10
        return protocol_v4_10
    if version == "v4.11":
        from . import protocol_v4_11
        return protocol_v4_11
    if version == "v6":
        from . import protocol_v6
        return protocol_v6
    if version == "v8":
        from . import protocol_v8
        return protocol_v8
    return None


def validate_turn_arguments(arguments: Any, version: str):
    """Validate against the selected Harness schema, without coercion/clamping.

    This is a model-input check, not an execution exception handler. Callers
    must return a rejected ToolResult before invoking any environment action.
    """
    module = protocol_module(version)
    schema = getattr(module, "TURN_PARAMETERS", None)
    if schema is None:
        return None, f"turn is not available in {version}"
    if not isinstance(arguments, dict):
        return None, "arguments must be an object"
    properties = schema["properties"]
    unknown = set(arguments) - set(properties)
    if unknown:
        return None, "unknown arguments: " + ", ".join(sorted(map(str, unknown)))
    missing = [key for key in schema["required"] if key not in arguments]
    if missing:
        return None, "missing arguments: " + ", ".join(missing)
    direction = arguments["direction"]
    if not isinstance(direction, str) or direction not in properties["direction"]["enum"]:
        return None, "direction must be left or right"
    degrees = arguments["degrees"]
    if isinstance(degrees, bool) or not isinstance(degrees, (int, float)):
        return None, "degrees must be a finite number"
    try:
        degrees = float(degrees)
    except (OverflowError, ValueError):
        return None, "degrees must be a finite number"
    if not math.isfinite(degrees):
        return None, "degrees must be a finite number"
    limits = properties["degrees"]
    if not limits["minimum"] <= degrees <= limits["maximum"]:
        return None, f"degrees must be in [{limits['minimum']}, {limits['maximum']}]"
    return {"direction": direction, "degrees": degrees}, ""


def rejected_turn_result(error: str) -> dict:
    return {"schema": "turn_tool_result_v1", "tool": "turn", "ok": False,
            "return_reason": "rejected", "error": error, "steps_used": 0}
