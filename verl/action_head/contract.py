"""Shared action expert Tool wire contract, independent of the render server."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict


TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)

RUN_ACTION_HEAD_DESCRIPTION = (
    "把当前局部导航阶段交给 Action Head 连续执行。"
    "target 说明本次局部导航的目标；where 说明目标在当前或历史视觉中的位置；"
    "plan 给出 1到3 个按顺序执行的高层运动阶段。"
    "Action Head 会在工具内部多次读取新 RGB、预测连续局部轨迹并只执行轨迹前缀；"
    "到达固定 fast-step 刷新上限、受阻、无可执行运动或安全事件时返回。"
    "不要提供 primitive action、未来轨迹、世界坐标、goal distance 或安全预算。"
)

RUN_ACTION_HEAD_PARAMETERS = {
    "type": "object",
    "properties": {
        "target": {
            "type": "string",
            "description": "本次局部导航要接近、经过或到达的对象或区域。",
        },
        "where": {
            "type": "string",
            "description": "目标在当前或历史视觉中的可观察位置和空间关系。",
        },
        "plan": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {"type": "string"},
            "description": "1到3个按执行顺序排列的高层运动阶段。",
        },
    },
    "required": ["target", "where", "plan"],
    "additionalProperties": False,
}

RUN_ACTION_HEAD_EXAMPLE = {
    "name": "run_action_head",
    "arguments": {
        "target": "the chair in the living area",
        "where": "on the right side of the current view beyond the open floor",
        "plan": [
            "align with the open route toward the chair",
            "approach the chair",
        ],
    },
}

RUN_ACTION_HEAD_EXAMPLE_OBSERVATION = {
    "tool": "run_action_head",
    "ok": True,
    "return_reason": "fast_step_budget_reached",
    "moved_m": 1.4,
    "chunks": 4,
    "blocked": False,
    "total_forward_m": 3.2,
    "net_yaw_deg": 10.0,
    "steps_used": 16,
}


@dataclass
class ToolResult:
    ok: bool
    model_text: str
    raw: dict = field(default_factory=dict)
    with_image: bool = False
    error: str = ""
    observation: object = None
    # v2-only ordered runtime payload. Each item contains compact non-oracle
    # chunk facts plus the RGB observation at that actual executed endpoint.
    # v1 callers leave this empty and retain byte-for-byte message behavior.
    trace_frames: tuple = ()


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def strip_code_fence(text: str) -> str:
    text = (text or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def validate_action_head_arguments(arguments: Any, *, max_plan_stages: int | None = 3):
    if not isinstance(arguments, dict):
        return None, "arguments must be an object"
    unknown = sorted(set(arguments) - {"target", "where", "plan"})
    if unknown:
        return None, f"unknown arguments: {', '.join(unknown)}"
    target = str(arguments.get("target") or "").strip()
    where = str(arguments.get("where") or "").strip()
    plan = arguments.get("plan")
    if not target:
        return None, "target must be a non-empty string"
    if not where:
        return None, "where must be a non-empty string"
    if (not isinstance(plan, list) or not plan or
            (max_plan_stages is not None and len(plan) > max_plan_stages)):
        return None, (f"plan must contain 1 to {max_plan_stages} ordered stages"
                      if max_plan_stages is not None else "plan must contain ordered stages")
    stages = [str(stage or "").strip() for stage in plan]
    if any(not stage for stage in stages):
        return None, "plan stages must be non-empty strings"
    return {"target": target, "where": where, "plan": stages}, ""
