# -*- coding: utf-8 -*-
"""ReAct tool layer for the VLN demo: continuous metres / degrees.

Why this exists
---------------
The classic loop (vln_model.run_model) can only emit the four discrete habitat
actions, so "go forward 0.8 m" has to be spelled 前进,前进,前进 and rounds to
0.75 m. Here the model instead calls tools with real units -- move_forward(0.8),
turn("left", 47) -- and we drive the simulator to match.

Continuous motion works because habitat_sim's Agent.act() re-reads
``action_space[id].actuation.amount`` on *every* call, so mutating it before a
step is enough. Measured on a live env: amount=0.13 moves exactly 0.1300 m,
turn amount=17.0 rotates exactly 17.00 deg. No rounding anywhere.

Two things that are NOT obvious and are load-bearing:

1. A single big forward step tunnels through geometry. Measured from one pose, forward
   1 m in one step travelled 1.0000 m and reported collided=False; the same
   metre as 4x0.25 m travelled 0.9823 m and reported collided=True, ending 6 cm
   away -- the big step went through a wall and lied about it. So distance is
   always split into sub-steps of at most SUBSTEP_M. Visible large turns are
   split into small angular steps so the first-person stream rotates smoothly.
2. The actuation amount is shared with the manual W/A/D path and with
   run_model. It MUST be restored in a finally, or a mid-step exception leaves
   the manual key silently walking a metre per press.
3. There is no backward action in this sim -- the action space is exactly
   stop / move_forward / turn_left / turn_right, and registering a
   "move_backward" ActionSpec raises (habitat_sim's compiled registry does not
   know the name; the task action registry does not either, so env.step could
   not reach it anyway). What does work, measured: a NEGATIVE move_forward
   amount. amount=-0.25 moved exactly 0.25 m opposite the facing direction in
   one env step with the collision flag intact. That is what MAX_BACK_M drives.
   The alternative -- turn 180, walk, turn back -- also works and ends up in
   the same place with no heading drift, but burns 5 env steps per 0.25 m
   against a 500-step budget, so it is not used.

Parser / ToolSpec / system-prompt builder below are copied from
/opt/external/user-d5121be6/Agentic_MoBot/agentic (core/tool_call_parser.py,
tools/base.py, prompts/templates.py, prompts/builder.py, core/loop.py) so the
tool-call wire format matches what that harness already trains against. Kept
close to the originals on purpose -- diff against them when upstream moves.
"""
import json
from remote_habitat.policy import terminal_frame
import math
import os
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field, replace

_VERL_ROOT = str(Path(__file__).resolve().parents[2] / "verl")
if _VERL_ROOT not in sys.path:
    sys.path.insert(0, _VERL_ROOT)

from action_head import contract as AH
from action_head import protocol as AHP
from action_head import protocol_v2 as AHV2
from action_head import protocol_v3 as AHV3
from action_head import protocol_v4 as AHV4
from action_head import protocol_v4_1 as AHV41
from action_head import protocol_v4_2 as AHV42
from action_head import protocol_v4_3 as AHV43
from action_head import protocol_v4_4 as AHV44

# ----------------------------------------------------------------------------
# motion limits
# ----------------------------------------------------------------------------
MAX_FWD_M = float(os.environ.get('VLN_PRIMITIVE_MAX_DISTANCE_M', '1.0'))
if not math.isfinite(MAX_FWD_M) or not .05 <= MAX_FWD_M <= 10.:
    raise ValueError('VLN_PRIMITIVE_MAX_DISTANCE_M must be finite, 0.05–10m')
MAX_TURN_DEG = 90.0    # ... and at most 90 degrees per call
MIN_FWD_M = 0.05       # a smaller move still costs a full env step
MIN_TURN_DEG = 0.5     # below this, skip the step entirely
SUBSTEP_M = 0.25       # anti-tunnelling; == habitat forward_step_size. Do not raise.
TURN_RENDER_STEP_DEG = 5.0  # smooth first-person animation for model turns
STALL_FRAC = 0.20      # a sub-step yielding <20% of its intent means we are jammed
# Backing up gets the same 1 m ceiling as going forward -- the two directions
# are the same primitive (a negative move_forward amount), split into the same
# 0.25 m sub-steps, with the same collision flag and the same stall detection.
# An earlier 0.5 m cap here was reasoning about the camera facing forward, but
# that argues for telling the model to prefer turning around, not for a limit
# the model can't see the reason for -- asked to back up a metre it would report
# having backed up a metre in two calls, which is worse than just doing it.
MAX_BACK_M = MAX_FWD_M


# ----------------------------------------------------------------------------
# tool-call parsing  (copied from agentic/core/tool_call_parser.py)
# ----------------------------------------------------------------------------
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>",
                          re.IGNORECASE | re.DOTALL)


def strip_code_fence(text):
    text = (text or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines:
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def split_thinking(text, reasoning=None):
    """-> (reasoning, visible). Handles both <think> tags and a reasoning field."""
    if "</think>" not in (text or ""):
        return (reasoning, text or "") if reasoning else (None, text or "")
    before, after = (text or "").split("</think>", 1)
    before = before.replace("<think>", "").strip()
    return reasoning or before or None, after.strip()


# Upstream's two ways of carrying an assistant turn into history (agentic's
# --thinking-history flag), copied verbatim so the wire format matches:
#   keep_thinking  -- the raw reply, <think> block and all, goes into history.
#                     The model sees its own reasoning from previous turns.
#   strip_thinking -- only the part after </think>.
# Upstream defaults to strip; this demo defaults to keep (see THINKING_HISTORY
# in vln_model) because navigation reasoning is cumulative -- "the doorway is on
# my left" is worth not re-deriving every turn. The cost is real and measured;
# see that constant's comment.
#
# THIS FUNCTION IS NOT SUFFICIENT ON ITS OWN, which is the trap. Both Qwen
# templates here re-split every assistant message on </think> and emit the
# reasoning only when `preserve_thinking` is set or the turn comes after the
# last user query -- and in a tool loop the last message is always a tool
# return, so every assistant turn falls on the stripping side. Passing the raw
# reply in and calling it keep_thinking gets you strip_thinking with extra
# steps: measured 0 of 2 reasoning blocks surviving on both models. The caller
# must also send chat_template_kwargs={"preserve_thinking": True}; stream_chat
# does, driven off the same constant. (Upstream never passes it, so upstream's
# keep_thinking is inert against these two templates.)
THINKING_HISTORY_MODES = ("keep_thinking", "strip_thinking")


def assistant_history_content(raw_content, visible_content, thinking_history):
    if thinking_history == "keep_thinking":
        return raw_content or visible_content
    return visible_content


def parse_tool_calls(text):
    """Extract Tool calls; tolerate only surplus trailing closing braces.

    Small Qwen models occasionally emit a complete valid object followed by
    one extra ``}`` before ``</tool_call>``.  ``raw_decode`` lets us accept that
    narrow formatting error while rejecting arbitrary trailing text; the
    tool-specific schema validator still checks names and arguments later.
    """
    calls = []
    for match in TOOL_CALL_RE.finditer(text or ""):
        cleaned = strip_code_fence(match.group(1).strip())
        try:
            data, end = json.JSONDecoder().raw_decode(cleaned)
        except Exception:
            # Safe only for the observed Qwen formatting fault: exactly one
            # unclosed object, with an otherwise complete final inner object.
            if (cleaned.endswith("}") and
                    cleaned.count("{") == cleaned.count("}") + 1):
                try:
                    cleaned += "}"
                    data, end = json.JSONDecoder().raw_decode(cleaned)
                except Exception:
                    continue
            else:
                continue
        if cleaned[end:].strip().strip("}"):
            continue
        if not isinstance(data, dict):
            continue
        name = data.get("name")
        # Do not turn malformed false-y values (null, [], "", 0) into {}.
        arguments = data.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            continue
        calls.append({"name": name.strip(), "arguments": arguments,
                      "raw": match.group(0).strip()})
    return calls


def invalid_tool_call_policy():
    """All live loops recover by default; terminate is historical replay only."""
    value = os.environ.get("ACTION_HEAD_INVALID_TOOL_CALL_POLICY", "feedback")
    if value not in ("terminate", "feedback"):
        raise ValueError("ACTION_HEAD_INVALID_TOOL_CALL_POLICY must be terminate or feedback")
    return value


def assert_tool_call_policy_matches(saved):
    if saved.get("invalid_tool_call_policy", "terminate") != invalid_tool_call_policy():
        raise RuntimeError("invalid-tool-call policy changed; use a new output directory")


def rejected_tool_call_result(parsed_calls, tools_by_name):
    """Describe a failed invocation without guessing or executing an action."""
    unknown = [call["name"] for call in parsed_calls
               if call["name"] not in tools_by_name]
    error = (
        "Unknown tool name. " if unknown else
        "No valid tool call could be parsed. "
    ) + (
        "Return exactly one <tool_call>...</tool_call> block containing valid JSON "
        "with a registered string name and an arguments object. "
        "arguments must not be a string, list, or null. "
        "For stop, arguments must be an object containing a reason string. "
        "Choose the next tool from the current task and observation; "
        "correcting a format error is not evidence of arrival."
    )
    return {
        "schema": "tool_call_rejection_v1", "ok": False,
        "return_reason": "rejected", "error_code": "invalid_tool_call",
        "error": error, "available_tools": list(tools_by_name),
        "unknown_tools": unknown, "action_executed": False,
        "steps_used": 0, "budget_cost": 1,
    }


def tool_call_batch_error(text, parsed_calls, tools_by_name):
    """Do not silently skip a bad block and execute a later valid action."""
    blocks = list(TOOL_CALL_RE.finditer(text or ""))
    if (not parsed_calls or len(blocks) != len(parsed_calls)
            or (text or "").lower().count("<tool_call>") != len(blocks)
            or any(call["name"] not in tools_by_name for call in parsed_calls)):
        return rejected_tool_call_result(parsed_calls, tools_by_name)
    return None


def tool_call_feedback_message(raw, budget_left, protocol=None):
    """Use the same tool-result channel as successful calls, without new images."""
    append_budget = protocol.append_tool_budget if protocol is not None else append_tool_budget
    return {"role": "user", "content": [{"type": "text",
        "text": "[Tool result]\n" + append_budget(compact_json(raw), budget_left)}]}


def rejected_stop_result(arguments):
    """A syntactically valid but invalid stop must never terminate navigation."""
    if not isinstance(arguments, dict) or not isinstance(arguments.get("reason"), str):
        note = "stop.arguments.reason must be a non-empty string inside an arguments object"
    else:
        _, note = AHV3.validate_stop_arguments(arguments)
    if not note:
        return None
    return rejected_tool_arguments_result("stop", note)


def rejected_tool_arguments_result(name, note):
    return {"schema": "tool_call_rejection_v1", "tool": name, "ok": False,
            "return_reason": "rejected", "error_code": "invalid_tool_arguments",
            "error": note, "action_executed": False, "steps_used": 0,
            "budget_cost": 1}


def normalize_primitive_call(name, arguments):
    """Shared native-eval/online validation; retain the primitive clamp contract."""
    if name in ("move_forward", "move_backward"):
        maximum = MAX_BACK_M if name == "move_backward" else None
        value, note = clamp_distance(arguments.get("distance"), maximum)
        if value is None:
            return None, note
        return {"distance": value}, note
    if name == "turn":
        direction = parse_direction(arguments.get("direction"))
        degrees, note = clamp_degrees(arguments.get("degrees"))
        if direction is None or degrees is None:
            return None, note or "direction must be left or right"
        return {"direction": direction, "degrees": degrees}, note
    return arguments, ""


# ----------------------------------------------------------------------------
# tool spec / result  (copied from agentic/tools/base.py, minus the
# multi-image fields -- navigation only ever has "the current frame")
# ----------------------------------------------------------------------------
@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    example: dict
    example_description: str = ""
    example_observation: dict = None


@dataclass
class ToolResult:
    ok: bool
    model_text: str
    raw: dict = field(default_factory=dict)
    with_image: bool = False
    error: str = ""
    # Runtime-only payload. Never serialized into model_text/raw.
    observation: object = None
    trace_frames: tuple = ()


@dataclass(frozen=True)
class ToolProfile:
    """The only Agent-level ablation variable: which tools are visible.

    Message construction, image retention, sampling and the ReAct loop remain
    shared.  A profile owns no observation preprocessing and cannot inject
    extra model-visible state.
    """

    name: str
    tool_specs: tuple

    @property
    def tools_by_name(self):
        return {spec.name: spec for spec in self.tool_specs}


# ----------------------------------------------------------------------------
# system prompt  (copied from agentic/prompts/{templates,builder}.py; the
# image_indices injection is dropped -- there is only ever one live frame here
# and advertising the parameter just invites bogus arguments)
# ----------------------------------------------------------------------------
QWEN_TOOL_CALL_INSTRUCTION = """For each function call, return a JSON object with function name and arguments within <tool_call></tool_call> XML tags:

<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""


def tool_schema(spec):
    return {"name": spec.name, "description": spec.description.strip(),
            "parameters": json.loads(json.dumps(spec.parameters))}


def tool_example_block(spec):
    lines = [f"Example for {spec.name}:"]
    if spec.example_description.strip():
        lines.append(spec.example_description.strip())
    lines += ["<tool_call>", json.dumps(spec.example, ensure_ascii=False),
              "</tool_call>"]
    if spec.example_observation is not None:
        lines += ["The tool may return:",
                  json.dumps(spec.example_observation, ensure_ascii=False)]
    lines.append("")
    return lines


def build_system_prompt(tool_specs):
    if not tool_specs:
        return ""
    blocks = [
        "You have the following tools available.",
        "",
        "Tool call format:",
        QWEN_TOOL_CALL_INSTRUCTION,
        "",
        "Tool list:",
        "<tools>",
        json.dumps([tool_schema(s) for s in tool_specs],
                   ensure_ascii=False, indent=2),
        "</tools>",
        "",
        "Examples:",
    ]
    for spec in tool_specs:
        blocks.extend(tool_example_block(spec))
    return "\n".join(blocks).strip()


def compact_json(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def append_tool_budget(text, remaining):
    """Same shape as agentic's helper, localised -- everything else the model
    reads in this demo is Chinese, and the budget line is meant to be obeyed."""
    remaining = max(0, int(remaining))
    if remaining == 0:
        notice = "[工具调用次数已用尽：不能再调用工具，请直接给出结论。]"
    else:
        notice = f"[工具调用预算：最多还可以调用 {remaining} 次。]"
    return f"{text.rstrip()}\n\n{notice}" if text.strip() else notice


# ----------------------------------------------------------------------------
# the three VLN tools
# ----------------------------------------------------------------------------
MOVE_FORWARD = ToolSpec(
    name="move_forward",
    description=(
        f"沿机器人当前朝向前进指定距离（米）。一次最多 {MAX_FWD_M:g} 米，需要走更远就连续多次调用。"
        "返回实际走过的距离——被墙壁或家具挡住时实际距离会小于请求距离。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "distance": {
                "type": "number", "minimum": MIN_FWD_M, "maximum": MAX_FWD_M,
                "description": f"前进距离，单位米，范围 {MIN_FWD_M}~{MAX_FWD_M}。",
            },
        },
        "required": ["distance"],
    },
    example={"name": "move_forward", "arguments": {"distance": 0.8}},
    example_description="前进 0.8 米：",
    # steps_used, not steps_left: the ReAct driver runs with the env step budget
    # uncapped, so there is no remainder to count down. Keep these in sync with
    # _totals -- the example is part of the system prompt, so a stale field name
    # here teaches the model to expect a key it will never see.
    example_observation={"tool": "move_forward", "requested_m": 0.8,
                         "moved_m": 0.8, "n_steps": 4, "blocked": False,
                         "total_forward_m": 3.25, "net_yaw_deg": -30.0,
                         "steps_used": 24},
)

TURN = ToolSpec(
    name="turn",
    description=(
        "原地转向指定角度（度）。一次最多 90 度，需要转更多就连续多次调用。"
        "left 为逆时针（向左），right 为顺时针（向右）。转向不会移动位置。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["left", "right"],
                          "description": "转向方向：left 向左，right 向右。"},
            "degrees": {"type": "number", "minimum": MIN_TURN_DEG,
                        "maximum": MAX_TURN_DEG,
                        "description": f"转向角度，单位度，范围 {MIN_TURN_DEG}~{MAX_TURN_DEG}。"},
        },
        "required": ["direction", "degrees"],
    },
    example={"name": "turn", "arguments": {"direction": "left", "degrees": 45}},
    example_description="向左转 45 度：",
    example_observation={"tool": "turn", "direction": "left",
                         "requested_deg": 45.0, "turned_deg": 45.0,
                         "n_steps": 1, "total_forward_m": 3.25,
                         "net_yaw_deg": -75.0, "steps_used": 25},
)

MOVE_BACKWARD = ToolSpec(
    name="move_backward",
    description=(
        f"保持朝向不变，向后倒退指定距离（米）。一次最多 {MAX_BACK_M} 米。"
        "摄像头朝前，倒退时看不到身后，所以要去后面较远的地方请转身再前进；"
        "贴到墙上或卡住时可以先退一点再转向。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "distance": {
                "type": "number", "minimum": MIN_FWD_M, "maximum": MAX_BACK_M,
                "description": f"倒退距离，单位米，范围 {MIN_FWD_M}~{MAX_BACK_M}。",
            },
        },
        "required": ["distance"],
    },
    example={"name": "move_backward", "arguments": {"distance": 0.3}},
    example_description="贴墙卡住时先后退 0.3 米：",
    example_observation={"tool": "move_backward", "requested_m": 0.3,
                         "moved_m": 0.3, "n_steps": 2, "blocked": False,
                         "total_forward_m": 2.95, "net_yaw_deg": -30.0,
                         "steps_used": 27},
)

STOP = ToolSpec(
    name="stop",
    description=(
        "结束本次导航任务。只有当你确认已经到达指令描述的目标、或指令要求的距离/角度"
        "已经走够时才调用。调用后任务立即结束，不能再移动。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "一句话说明为什么现在可以停止。"},
        },
        "required": ["reason"],
    },
    example={"name": "stop", "arguments": {"reason": "已到达桌子旁边"}},
    example_description="任务完成时：",
)

RUN_ACTION_HEAD = ToolSpec(
    name="run_action_head",
    description=AH.RUN_ACTION_HEAD_DESCRIPTION,
    parameters=AH.RUN_ACTION_HEAD_PARAMETERS,
    example=AH.RUN_ACTION_HEAD_EXAMPLE,
    example_description="把右前方椅子的局部接近阶段交给 Action Head：",
    example_observation=AH.RUN_ACTION_HEAD_EXAMPLE_OBSERVATION,
)

MOVE_TO_POINT = ToolSpec(
    name="move_to_point",
    description=(
        "传入当前第一视角图像中的一个二维视觉目标点。"
        "工具读取与当前 RGB 同帧的 metric depth，将该像素反投影为局部三维 waypoint，"
        "再由地图规划器和跟随器执行移动。point 必须落在当前画面可见、可行走的地面上；"
        "接近物体时选择物体前方的地面，禁止点击物体、墙面或家具表面。"
        "坐标是 0~1000 的千分位：[x, y]，左上角为原点。"
        "坐标越界、画面已过期或该像素没有有效同帧深度时会拒绝。"
        "不要传视角编号、pixel 字段或距离；移动后会返回新的第一视角画面。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "point_2d": {
                "type": "array", "minItems": 2, "maxItems": 2,
                "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                "description": "当前画面中可见、可行走地面上的 [x, y] 千分位坐标，必须具有有效同帧 metric depth。",
            },
        },
        "required": ["point_2d"],
    },
    example={"name": "move_to_point", "arguments": {"point_2d": [530, 720]}},
    example_description="选择当前画面中可见、可行走地面上的一点并向它移动：",
    example_observation={"tool": "move_to_point", "depth_m": 2.4,
                         "moved_m": 2.15, "blocked": False, "steps_used": 10},
)

MOVE_TO_BBOX = ToolSpec(
    name="move_to_bbox",
    description=(
        "Uni-LaViRA 风格的视觉框导航。用 bbox_2d 框住当前第一视角图像中最合适的"
        "前进目标；工具使用框的底边中心（目标与地面接触/门槛位置）及其邻域的"
        "有效 metric depth 反投影为局部三维 waypoint，再由底层跟随器转向并前进。"
        "框应覆盖完整目标并让底边贴近其地面接触处。坐标是 0~1000 的"
        "千分位 [x1, y1, x2, y2]，左上角为原点。不要传视角编号或额外字段。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "bbox_2d": {
                "type": "array", "minItems": 4, "maxItems": 4,
                "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                "description": "当前单视角图像中的 [x1, y1, x2, y2] 千分位框。",
            },
        },
        "required": ["bbox_2d"],
    },
    example={"name": "move_to_bbox",
             "arguments": {"bbox_2d": [410, 480, 620, 850]}},
    example_description="框住前方门洞或目标区域并向它移动：",
    example_observation={"tool": "move_to_bbox", "depth_m": 2.8,
                         "moved_m": 2.5, "blocked": False, "steps_used": 12},
)

RUN_ACTION_HEAD_V2 = ToolSpec(
    name="run_action_head",
    description=AHV2.RUN_ACTION_HEAD_DESCRIPTION,
    parameters=AHV2.RUN_ACTION_HEAD_PARAMETERS,
    example={
        "name": "run_action_head",
        "arguments": {
            "target": "the doorway ahead",
            "where": "in the center of the current view",
            "plan": ["approach the doorway", "pass through it"],
        },
    },
    example_description="Delegate one local option to the Action Head:",
    example_observation={
        "schema": AHV2.TOOL_RESULT_SCHEMA,
        "tool": "run_action_head",
        "option_id": "option-12",
        "ok": True,
        "return_reason": "fast_step_budget_reached",
        "completion": "unknown",
        "chunks": 4,
        "path_m": 1.4,
        "net_m": 1.3,
        "blocked": False,
        "collision": False,
        "steps_used": 16,
    },
)

STOP_V2 = ToolSpec(
    name="stop",
    description=AHV2.STOP_DESCRIPTION,
    parameters=AHV2.STOP_PARAMETERS,
    example={"name": "stop", "arguments": {
        "reason": "The instructed destination is visible at the current position."}},
    example_description="Stop after visually confirming the final destination:",
)

RUN_ACTION_HEAD_V3 = ToolSpec(
    name="run_action_head",
    description=AHV3.RUN_ACTION_HEAD_DESCRIPTION,
    parameters=AHV3.RUN_ACTION_HEAD_PARAMETERS,
    example={
        "name": "run_action_head",
        "arguments": {
            "target": "the doorway",
            "where": "on the left side of the current view",
            "plan": ["turn left toward the doorway", "approach the doorway"],
        },
    },
    example_description="Delegate one local option to the Action Head:",
    example_observation={
        "schema": AHV3.TOOL_RESULT_SCHEMA,
        "tool": "run_action_head",
        "option_id": "option-12",
        "ok": True,
        "return_reason": "fast_step_budget_reached",
        "completion": "unknown",
        "chunks": 4,
        "path_m": 1.4,
        "net_m": 1.3,
        "blocked": False,
        "collision": False,
        "steps_used": 16,
        "direction_alignment": {
            "planned": "left", "actual": "left", "match": True},
    },
)

STOP_V3 = ToolSpec(
    name="stop",
    description=AHV3.STOP_DESCRIPTION,
    parameters=AHV3.STOP_PARAMETERS,
    example={"name": "stop", "arguments": {
        "reason": "The instructed destination is visible at the current position."}},
    example_description="Stop after visually confirming the final destination:",
)

TURN_V4 = ToolSpec(
    name="turn",
    description=AHV4.TURN_DESCRIPTION,
    parameters=AHV4.TURN_PARAMETERS,
    example={"name": "turn", "arguments": {
        "direction": "left", "degrees": 90}},
    example_description="Rotate left in place, then inspect the returned image:",
    example_observation={
        "schema": "turn_tool_result_v1", "tool": "turn", "ok": True,
        "direction": "left", "requested_deg": 90.0,
        "turned_deg": 90.0, "steps_used": 1,
    },
)

RUN_ACTION_HEAD_V4 = ToolSpec(
    name="run_action_head",
    description=AHV4.RUN_ACTION_HEAD_DESCRIPTION,
    parameters=AHV4.RUN_ACTION_HEAD_PARAMETERS,
    example=RUN_ACTION_HEAD_V3.example,
    example_description=RUN_ACTION_HEAD_V3.example_description,
    example_observation={
        **RUN_ACTION_HEAD_V3.example_observation,
        "schema": AHV4.TOOL_RESULT_SCHEMA,
    },
)

STOP_V4 = ToolSpec(
    name="stop",
    description=AHV4.STOP_DESCRIPTION,
    parameters=AHV4.STOP_PARAMETERS,
    example=STOP_V3.example,
    example_description=STOP_V3.example_description,
)

TURN_V41 = ToolSpec(
    name="turn", description=AHV41.TURN_DESCRIPTION,
    parameters=AHV41.TURN_PARAMETERS,
    example={"name": "turn", "arguments": {
        "direction": "left", "degrees": 90}},
    example_description="Use the one available recovery turn after a mismatch:",
    example_observation={
        "schema": "turn_tool_result_v2", "tool": "turn", "ok": True,
        "direction": "left", "requested_deg": 90.0, "turned_deg": 90.0,
        "steps_used": 1, "turn_reason": "previous_option_direction_mismatch",
        "turn_available_after_result": False,
    },
)
RUN_ACTION_HEAD_V41 = ToolSpec(
    name="run_action_head", description=AHV41.RUN_ACTION_HEAD_DESCRIPTION,
    parameters=AHV41.RUN_ACTION_HEAD_PARAMETERS,
    example=RUN_ACTION_HEAD_V4.example,
    example_description=RUN_ACTION_HEAD_V4.example_description,
    example_observation={**RUN_ACTION_HEAD_V4.example_observation,
                         "schema": AHV41.TOOL_RESULT_SCHEMA},
)
STOP_V41 = ToolSpec(
    name="stop", description=AHV41.STOP_DESCRIPTION,
    parameters=AHV41.STOP_PARAMETERS, example=STOP_V4.example,
    example_description=STOP_V4.example_description,
)

RUN_ACTION_HEAD_V42 = ToolSpec(
    name="run_action_head", description=AHV42.RUN_ACTION_HEAD_DESCRIPTION,
    parameters=AHV42.RUN_ACTION_HEAD_PARAMETERS,
    example=RUN_ACTION_HEAD_V4.example,
    example_description="Example call syntax:",
    example_observation={**RUN_ACTION_HEAD_V4.example_observation,
                         "schema": AHV42.TOOL_RESULT_SCHEMA},
)
TURN_V42 = ToolSpec(
    name="turn", description=AHV42.TURN_DESCRIPTION,
    parameters=AHV42.TURN_PARAMETERS,
    example={"name": "turn", "arguments": {
        "direction": "left", "degrees": 45}},
    example_description="Example call syntax:",
    example_observation={
        "schema": "turn_tool_result_v1", "tool": "turn", "ok": True,
        "direction": "left", "requested_deg": 45.0,
        "turned_deg": 45.0, "steps_used": 1,
    },
)
STOP_V42 = ToolSpec(
    name="stop", description=AHV42.STOP_DESCRIPTION,
    parameters=AHV42.STOP_PARAMETERS, example=STOP_V4.example,
    example_description="Example call syntax:",
)

RUN_ACTION_HEAD_V43 = ToolSpec(
    name="run_action_head", description=AHV43.RUN_ACTION_HEAD_DESCRIPTION,
    parameters=AHV43.RUN_ACTION_HEAD_PARAMETERS,
    example=RUN_ACTION_HEAD_V42.example,
    example_description=RUN_ACTION_HEAD_V42.example_description,
    example_observation={**RUN_ACTION_HEAD_V42.example_observation,
                         "schema": AHV43.TOOL_RESULT_SCHEMA},
)
TURN_V43 = ToolSpec(
    name="turn", description=AHV43.TURN_DESCRIPTION,
    parameters=AHV43.TURN_PARAMETERS,
    example=TURN_V42.example,
    example_description=TURN_V42.example_description,
    example_observation=TURN_V42.example_observation,
)
STOP_V43 = ToolSpec(
    name="stop", description=AHV43.STOP_DESCRIPTION,
    parameters=AHV43.STOP_PARAMETERS, example=STOP_V42.example,
    example_description=STOP_V42.example_description,
)

RUN_ACTION_HEAD_V44 = ToolSpec(
    name="run_action_head", description=AHV44.RUN_ACTION_HEAD_DESCRIPTION,
    parameters=AHV44.RUN_ACTION_HEAD_PARAMETERS,
    example=RUN_ACTION_HEAD_V43.example,
    example_description=RUN_ACTION_HEAD_V43.example_description,
    example_observation={**RUN_ACTION_HEAD_V43.example_observation,
                         "schema": AHV44.TOOL_RESULT_SCHEMA},
)
TURN_V44 = ToolSpec(
    name="turn", description=AHV44.TURN_DESCRIPTION,
    parameters=AHV44.TURN_PARAMETERS,
    example=TURN_V43.example,
    example_description=TURN_V43.example_description,
    example_observation=TURN_V43.example_observation,
)
STOP_V44 = ToolSpec(
    name="stop", description=AHV44.STOP_DESCRIPTION,
    parameters=AHV44.STOP_PARAMETERS,
    example={"name": "stop", "arguments": {
        "reason": ("The robot is physically beside the instructed destination "
                   "after approaching it, not merely viewing it from a distance.")}},
    example_description="Stop only after physically reaching the destination:",
)

PRIMITIVE_TOOL_SPECS = (MOVE_FORWARD, MOVE_BACKWARD, TURN, STOP)
ACTION_HEAD_TOOL_SPECS = (RUN_ACTION_HEAD, STOP)
EMBODIED_POINT_TOOL_SPECS = (MOVE_TO_POINT, MOVE_BACKWARD, TURN, STOP)
UNILAVIRA_BBOX_TOOL_SPECS = (MOVE_TO_BBOX, MOVE_BACKWARD, TURN, STOP)
ACTION_HEAD_V2_TOOL_SPECS = (RUN_ACTION_HEAD_V2, STOP_V2)
ACTION_HEAD_V3_TOOL_SPECS = (RUN_ACTION_HEAD_V3, STOP_V3)
ACTION_HEAD_V4_TOOL_SPECS = (TURN_V4, RUN_ACTION_HEAD_V4, STOP_V4)
ACTION_HEAD_V41_TOOL_SPECS = (TURN_V41, RUN_ACTION_HEAD_V41, STOP_V41)
ACTION_HEAD_V42_TOOL_SPECS = (RUN_ACTION_HEAD_V42, TURN_V42, STOP_V42)
ACTION_HEAD_V43_TOOL_SPECS = (RUN_ACTION_HEAD_V43, TURN_V43, STOP_V43)
ACTION_HEAD_V44_TOOL_SPECS = (RUN_ACTION_HEAD_V44, TURN_V44, STOP_V44)
TOOL_PROFILES = {
    "primitive": ToolProfile("primitive", PRIMITIVE_TOOL_SPECS),
    "action_head": ToolProfile("action_head", ACTION_HEAD_TOOL_SPECS),
    "embodied_point": ToolProfile("embodied_point", EMBODIED_POINT_TOOL_SPECS),
    "unilavira_bbox": ToolProfile("unilavira_bbox", UNILAVIRA_BBOX_TOOL_SPECS),
}

# Backward-compatible names for existing callers. Strict A/B code selects a
# ToolProfile explicitly; legacy callers continue to receive primitive tools.
TOOL_SPECS = list(PRIMITIVE_TOOL_SPECS)
TOOLS_BY_NAME = {s.name: s for s in TOOL_SPECS}


def get_tool_profile(name, action_head_as_tool_version="v1"):
    version = AHP.normalize_version(action_head_as_tool_version)
    if str(name) == "action_head" and version == "v2":
        return ToolProfile("action_head", ACTION_HEAD_V2_TOOL_SPECS)
    if str(name) == "action_head" and version == "v3":
        return ToolProfile("action_head", ACTION_HEAD_V3_TOOL_SPECS)
    if str(name) == "action_head" and version == "v4":
        return ToolProfile("action_head", ACTION_HEAD_V4_TOOL_SPECS)
    if str(name) == "action_head" and version == "v4.1":
        return ToolProfile("action_head", ACTION_HEAD_V41_TOOL_SPECS)
    if str(name) == "action_head" and version in ("v4.9", "v4.10"):
        protocol = AHP.protocol_module(version)
        specs = tuple(replace(spec, example_observation={
            **spec.example_observation, "schema": protocol.TOOL_RESULT_SCHEMA})
            if spec.name == "run_action_head" else spec
            for spec in ACTION_HEAD_V41_TOOL_SPECS)
        return ToolProfile("action_head", specs + (replace(
            MOVE_BACKWARD, description=protocol.BACKOFF_DESCRIPTION,
            parameters=protocol.BACKOFF_PARAMETERS),))
    if str(name) == "action_head" and version == "v4.2":
        return ToolProfile("action_head", ACTION_HEAD_V42_TOOL_SPECS)
    if str(name) == "action_head" and version == "v4.3":
        return ToolProfile("action_head", ACTION_HEAD_V43_TOOL_SPECS)
    if str(name) == "action_head" and version in ("v4.11", "v6", "v8"):
        protocol = AHP.protocol_module(version)
        definitions = {item["name"]: item for item in protocol.TOOLS}
        specs = tuple(replace(spec, description=definitions[spec.name]["description"],
                              parameters=definitions[spec.name]["parameters"],
                              example_observation={**spec.example_observation,
                                                   "schema": protocol.TOOL_RESULT_SCHEMA}
                              if spec.name == "run_action_head" else spec.example_observation)
                      for spec in ACTION_HEAD_V43_TOOL_SPECS)
        return ToolProfile("action_head", specs)
    if str(name) == "action_head" and version in ("v4.5", "v4.6", "v4.7", "v4.8"):
        specs = tuple(replace(spec, example_observation={
            **spec.example_observation, "schema": AHP.protocol_module(version).TOOL_RESULT_SCHEMA})
            if spec.name == "run_action_head" else spec
            for spec in ACTION_HEAD_V43_TOOL_SPECS)
        return ToolProfile("action_head", specs)
    if str(name) == "action_head" and version == "v4.4":
        return ToolProfile("action_head", ACTION_HEAD_V44_TOOL_SPECS)
    try:
        return TOOL_PROFILES[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"unknown tool profile {name!r}; expected one of "
            f"{sorted(TOOL_PROFILES)}") from exc


def validate_action_head_arguments(arguments):
    """Fail-closed validation for the deliberately small v1 contract.

    This is intentionally stricter than the tolerant primitive argument
    coercion.  Invalid semantic contracts must return to the slow Agent rather
    than trigger robot motion.
    """
    return AH.validate_action_head_arguments(arguments)


def enforce_v43_consecutive_turn_limit(version, tool_name, current_count):
    """Advance the accepted-call streak or return a model-visible rejection.

    Call for turn only after argument/availability validation, and for AH only
    after an ok result. Rejected invocations spend global tool budget but must
    not increment or reset this separate execution-state counter.
    """
    if str(tool_name) != "turn":
        return 0, None
    count = max(0, int(current_count))
    if AHP.normalize_version(version) not in ("v4.3", "v4.4", "v4.5", "v4.6", "v4.7", "v4.8", "v4.11", "v6", "v8"):
        return count, None
    if count < AHV43.MAX_CONSECUTIVE_TURN_CALLS:
        return count + 1, None
    raw = {
        "schema": "turn_tool_result_v1",
        "tool": "turn",
        "ok": False,
        "return_reason": "consecutive_turn_limit_reached",
        "error": (
            "The limit of two consecutive turn calls has been reached. "
            "The robot did not rotate. Call run_action_head or stop next."
        ),
        "steps_used": 0,
        "max_consecutive_turn_calls": AHV43.MAX_CONSECUTIVE_TURN_CALLS,
        "attempted_consecutive_turn_call": count + 1,
        "required_next_tool": "run_action_head_or_stop",
    }
    return count, raw


# ----------------------------------------------------------------------------
# argument coercion
# ----------------------------------------------------------------------------
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def as_float(value, default=None):
    """Tolerant number parsing: 0.8, "0.8", "0.8 米" all give 0.8."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    m = _NUM_RE.search(str(value or ""))
    return float(m.group(0)) if m else default


def clamp_distance(value, maximum=None):
    """-> (metres, note). note is non-empty when the request was out of range.

    `maximum` lets move_backward reuse this with its own lower ceiling.
    """
    maximum = MAX_FWD_M if maximum is None else maximum
    d = as_float(value)
    if d is None or not math.isfinite(d):
        return None, "distance 不是一个数字"
    if d <= 0:
        return None, "distance 必须大于 0"
    if d > maximum:
        return maximum, f"请求 {d:.2f} 米超过单次上限，已按 {maximum:.2f} 米执行"
    if d < MIN_FWD_M:
        return MIN_FWD_M, f"请求 {d:.3f} 米过小，已按 {MIN_FWD_M:.2f} 米执行"
    return d, ""


def clamp_degrees(value):
    a = as_float(value)
    if a is None or not math.isfinite(a):
        return None, "degrees 不是一个数字"
    if a < 0:
        return None, "degrees 必须为正，方向请用 direction 指定"
    if a > MAX_TURN_DEG:
        return MAX_TURN_DEG, f"请求 {a:.1f} 度超过单次上限，已按 {MAX_TURN_DEG:.0f} 度执行"
    return a, ""


def parse_direction(value):
    s = str(value or "").strip().lower()
    if s in ("left", "l", "左", "向左", "左转", "ccw"):
        return "left"
    if s in ("right", "r", "右", "向右", "右转", "cw"):
        return "right"
    return None


# ----------------------------------------------------------------------------
# simulator geometry helpers (no habitat import -- only duck-typed env access)
# ----------------------------------------------------------------------------
def _agent_cfg(env):
    return env.sim.agents[0].agent_config


def action_id(env, name):
    """habitat_sim action-space key for 'move_forward' / 'turn_left' / 'turn_right'."""
    for key, spec in _agent_cfg(env).action_space.items():
        if getattr(spec, "name", None) == name:
            return key
    raise KeyError(f"no action named {name!r} in the agent action space")


def actuation(env, name):
    return _agent_cfg(env).action_space[action_id(env, name)].actuation


def default_amounts(env):
    """{action name: configured amount} -- the values we must always restore to."""
    return {n: float(actuation(env, n).amount)
            for n in ("move_forward", "turn_left", "turn_right")}


def _state(env):
    return env.sim.get_agent_state()


def _xz(env):
    p = _state(env).position
    return (float(p[0]), float(p[2]))


def _yaw(env):
    """Heading in degrees, from the agent quaternion. Sign matches habitat's
    -Z-forward / Y-up convention; only differences of this are ever used."""
    q = _state(env).rotation
    w, x, y, z = (float(q.w), float(q.x), float(q.y), float(q.z))
    # rotate the forward vector (0, 0, -1) and read its heading in the XZ plane
    fx = -2.0 * (x * z + w * y)
    fz = -(1.0 - 2.0 * (x * x + y * y))
    return math.degrees(math.atan2(fx, -fz))


def wrap180(deg):
    return ((float(deg) + 180.0) % 360.0) - 180.0


def _dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _forward_progress(p0, p1, yaw0):
    """Net displacement projected onto the heading we started with.

    Sliding along a wall keeps the path length high while this collapses, which
    is exactly the difference the model needs in order to decide to turn.
    """
    dx, dz = p1[0] - p0[0], p1[1] - p0[1]
    r = math.radians(yaw0)
    return dx * math.sin(r) - dz * math.cos(r)


def relative_motion_pose(p0, p1, yaw0, yaw1):
    """Measured displacement in call-start forward/left axes (left-positive yaw).

    Unlike moved_m, this retains sideways displacement caused by wall sliding.
    Only relative odometry is returned, never world pose or target information.
    """
    dx, dz = p1[0] - p0[0], p1[1] - p0[1]
    r = math.radians(yaw0)
    return {"x_m": round(_forward_progress(p0, p1, yaw0), 3),
            "y_m": round(-dx * math.cos(r) - dz * math.sin(r), 3),
            "yaw_deg": round(-wrap180(yaw1 - yaw0), 2)}


# ----------------------------------------------------------------------------
# execution
# ----------------------------------------------------------------------------
@terminal_frame
def drive_forward(env, meters, on_step=None, abort=None, sign=1, min_m=None,
                  render=True):
    """Walk `meters` forward as equal sub-steps of at most SUBSTEP_M.

    on_step(obs, i) fires after every sub-step so the UI animates; abort() is
    polled between sub-steps. Returns (obs, info) where info reports *measured*
    motion -- never the requested value, see the module docstring.

    sign=-1 backs up instead, by setting a negative actuation amount (measured
    to move exactly the requested distance opposite the facing direction, with
    the collision flag still reported). Everything else -- sub-stepping, stall
    detection, the finally-restore -- is identical, which is the whole reason
    backward shares this function rather than getting its own copy.

    render=False passes through to env.step so intermediate sub-steps skip the
    render HTTP (closed-loop FM chunk executes the whole path unrendered, then
    renders once at the stopping pose). Motion is measured off the sim pose
    (_xz/env.sim.previous_step_collided), not off the returned obs, so stall and
    collision detection are unaffected; obs just carries no rgb.

    min_m overrides the MIN_FWD_M floor. That floor exists to stop the *model*
    from spending a tool call and a full env step on a move too small to matter;
    hold-to-drive has no tool budget and wants the opposite trade -- the smallest
    step the simulator will take, so the frame rate is bounded by env.step rather
    than by the step size. It is a parameter and not a lowered constant so the
    model-facing guard stays where it is.
    """
    lo = MIN_FWD_M if min_m is None else max(0.0, float(min_m))
    hi = MAX_FWD_M if sign > 0 else MAX_BACK_M
    meters = max(lo, min(hi, float(meters)))   # belt and braces
    aid = action_id(env, "move_forward")
    act = actuation(env, "move_forward")
    orig = act.amount
    n = max(1, math.ceil(meters / SUBSTEP_M - 1e-9))
    per = meters / n                       # uniform, always <= SUBSTEP_M
    p0, yaw0 = _xz(env), _yaw(env)
    moved, done, collided, reason, obs = 0.0, 0, False, "ok", None
    try:
        act.amount = per * sign
        for _ in range(n):
            if env.episode_over:           # Env.step asserts rather than erroring
                reason = "episode_over"
                break
            if abort is not None and abort():
                reason = "aborted"
                break
            before = _xz(env)
            obs = env.step(aid, render=render)
            step_d = _dist(before, _xz(env))
            moved += step_d
            done += 1
            if env.sim.previous_step_collided:
                collided = True
            if on_step is not None:
                on_step(obs, done)
            # Do NOT break on `collided` alone: with allow_sliding grazing a
            # corridor wall is normal and still makes progress (measured:
            # 4x0.25 m reported a collision and still travelled 0.98 m). What
            # actually means "stuck" is the pose refusing to change.
            if step_d < per * STALL_FRAC:
                reason = "blocked"
                break
    finally:
        act.amount = orig                  # never optional -- shared with manual drive
    p1 = _xz(env)
    # Signed along the *facing* axis, so backing up reads as a negative
    # forward_m rather than as mysterious progress.
    info = {
        "requested_m": round(meters, 3),
        "moved_m": round(moved, 3),
        "forward_m": round(_forward_progress(p0, p1, yaw0), 3),
        "n_steps": done,
        "collided": collided,
        "reason": reason,
        "blocked": reason == "blocked" or moved < meters * 0.95,
        "yaw_deg": round(wrap180(_yaw(env) - yaw0), 2),
        "sign": sign,
        "actual_end_pose": relative_motion_pose(p0, p1, yaw0, _yaw(env)),
    }
    return obs, info


def drive_backward(env, meters, on_step=None, abort=None, min_m=None):
    """Back up `meters`. Thin wrapper -- see drive_forward's sign parameter."""
    return drive_forward(env, meters, on_step, abort, sign=-1, min_m=min_m)


@terminal_frame
def drive_turn(env, direction, degrees, on_step=None, render=True, abort=None):
    """Turn in place, splitting visible turns into rendered angular sub-steps.

    Headless/chunk calls without an ``on_step`` callback stay a single action.
    Yaw is measured off the sim, not the returned observation.
    """
    name = "turn_left" if direction == "left" else "turn_right"
    aid = action_id(env, name)
    act = actuation(env, name)
    orig = act.amount
    yaw0 = _yaw(env)
    obs, done, reason = None, 0, "ok"
    if degrees < MIN_TURN_DEG:
        # too small to matter, and an env step is a scarce resource
        return None, {"requested_deg": round(degrees, 2), "turned_deg": 0.0,
                      "n_steps": 0, "reason": "too_small"}
    try:
        n = (max(1, math.ceil(float(degrees) / TURN_RENDER_STEP_DEG - 1e-9))
             if render and on_step is not None else 1)
        act.amount = float(degrees) / n
        for _ in range(n):
            if env.episode_over:
                reason = "episode_over"
                break
            if abort is not None and abort():
                reason = "aborted"
                break
            obs = env.step(aid, render=render)
            done += 1
            if on_step is not None:
                on_step(obs, done)
    finally:
        act.amount = orig
    return obs, {"requested_deg": round(degrees, 2),
                 "turned_deg": round(abs(wrap180(_yaw(env) - yaw0)), 2),
                 "n_steps": done, "reason": reason}


# ----------------------------------------------------------------------------
# observations handed back to the model
# ----------------------------------------------------------------------------
def move_observation(info, totals, steps_left, note=""):
    """Compact JSON + (only when something went wrong) a plain-language note.

    distance_to_goal is deliberately absent: it is the quantity success is
    judged on, so feeding it back would turn the run into an oracle-guided one
    and make it incomparable with the en_r1 eval baseline.
    """
    back = info.get("sign", 1) < 0
    obs = {"tool": "move_backward" if back else "move_forward",
           "requested_m": info["requested_m"], "moved_m": info["moved_m"],
           "n_steps": info["n_steps"], "blocked": bool(info["blocked"])}
    notes = [note] if note else []
    if info["blocked"]:
        obs["forward_m"] = info["forward_m"]
        obs["reason"] = "obstacle" if info["reason"] in ("blocked", "ok") else info["reason"]
        notes.append(
            f"后方受阻：请求后退 {info['requested_m']:.2f} 米，"
            f"实际只退了 {info['moved_m']:.2f} 米。请转向后再移动。"
            if back else
            f"前方受阻：请求前进 {info['requested_m']:.2f} 米，"
            f"实际只走了 {info['moved_m']:.2f} 米。请先转向再前进。")
    obs.update(_totals(totals, steps_left))
    return _render(obs, notes)


def turn_observation(info, totals, steps_left, direction, note=""):
    obs = {"tool": "turn", "direction": direction,
           "requested_deg": info["requested_deg"],
           "turned_deg": info["turned_deg"], "n_steps": info["n_steps"]}
    notes = [note] if note else []
    if info["reason"] == "too_small":
        notes.append("转向角度过小，未执行。")
    obs.update(_totals(totals, steps_left))
    return _render(obs, notes)


def error_observation(name, message, totals, steps_left):
    obs = {"tool": name, "ok": False, "error": message,
           "return_reason": "rejected", "error_code": "invalid_tool_arguments",
           "action_executed": False}
    obs.update(_totals(totals, steps_left))
    return _render(obs, [message])


def _totals(totals, steps_left):
    """steps_left=None means the env step budget is uncapped (the ReAct driver's
    case: vln_demo.yaml sets max_episode_steps: 0). Report steps_used instead of
    a made-up remainder -- a "steps_left" that counts down from a limit nothing
    enforces is worse than no field at all, because the model paces itself by it.
    The classic driver still passes a real number."""
    out = {"total_forward_m": round(
               totals.get("path_dist", totals.get("dist", 0.0)), 2),
           "net_yaw_deg": round(wrap180(totals.get("yaw", 0.0)), 1)}
    if steps_left is None:
        out["steps_used"] = int(totals.get("steps", 0))
    else:
        out["steps_left"] = int(steps_left)
    return out


def _render(obs, notes):
    text = compact_json(obs)
    if notes:
        text += "\n" + "\n".join(n for n in notes if n)
    return text, obs


# ----------------------------------------------------------------------------
# message plumbing
# ----------------------------------------------------------------------------
def strip_old_images(messages, keep_last=8):
    """Drop image parts from all but the most recent `keep_last` frames.

    A variant of agentic's sanitize_messages, which strips *every* image --
    navigation has to keep the recent ones or the model loses all visual
    context. Stripped parts leave a marker so the frame ordering stays legible.
    """
    idx = [(i, j)
           for i, msg in enumerate(messages)
           if isinstance(msg.get("content"), list)
           for j, part in enumerate(msg["content"])
           if isinstance(part, dict) and part.get("type") == "image"]
    if len(idx) <= keep_last:
        return messages
    drop = set(idx[:len(idx) - keep_last])
    out = []
    for i, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        new = []
        for j, part in enumerate(content):
            if (i, j) in drop:
                new.append({"type": "text", "text": "[历史画面已省略]"})
            else:
                new.append(part)
        out.append({**msg, "content": new})
    return out


def strip_context_images_v2(messages, keep_last_slow=8):
    """action expert v2's independent slow/latest-fast budget."""
    return AHV2.strip_context_images(messages, keep_slow=keep_last_slow)


def count_images(messages):
    return sum(1 for msg in messages
               if isinstance(msg.get("content"), list)
               for part in msg["content"]
               if isinstance(part, dict) and part.get("type") == "image")
