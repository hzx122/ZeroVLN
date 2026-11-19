# -*- coding: utf-8 -*-
"""Prompt templates + action parsing for the VLN demo.

Why this module exists
----------------------
The stock ``embodied_r1`` prompt (in rynnvln_eval.build_prompt_embodied) feeds
the model *images only* -- it never tells the model what it has already done.
For dataset episodes ("walk to the kitchen and stop by the fridge") that is
mostly fine, but for metric instructions ("go forward one meter") the model has
no way to know it already moved 1 m, so it keeps shuffling around forever.

Every template here except ``en_r1`` injects an explicit action-history block:
the executed action counts, the accumulated forward distance (0.25 m/step) and
the net heading change (30 deg/turn). That turns "walk one meter then stop"
into simple arithmetic the model can actually check.

Add a template by appending to TEMPLATES; the web UI picks it up automatically.
"""
import re
import os

FWD, LEFT, RIGHT, STOP = 1, 2, 3, 0

STEP_METERS = 0.25   # habitat forward step (see vln_demo.yaml)
TURN_DEGREES = 30.0  # habitat turn angle

ZH_ACTION = {FWD: "前进", LEFT: "左转", RIGHT: "右转", STOP: "停止"}
EN_ACTION = {
    STOP: "D. Stop",
    FWD: "A. Move forward 25 cm",
    LEFT: "B. Turn left 30 degrees",
    RIGHT: "C. Turn right 30 degrees",
}
# Order used when listing options to the model.
_ORDER = [FWD, LEFT, RIGHT, STOP]


# ----------------------------------------------------------------------------
# action history -> text
# ----------------------------------------------------------------------------
def action_stats(actions):
    """Aggregate an executed-action list into the numbers a model can reason on."""
    fwd = actions.count(FWD)
    left = actions.count(LEFT)
    right = actions.count(RIGHT)
    yaw = (right - left) * TURN_DEGREES
    yaw = ((yaw + 180.0) % 360.0) - 180.0  # normalise to (-180, 180]
    return {
        "n": len(actions), "fwd": fwd, "left": left, "right": right,
        "dist": fwd * STEP_METERS, "yaw": yaw,
    }


def history_block_zh(actions, tail=20):
    """Chinese action-history block. Empty string when nothing has run yet."""
    if not actions:
        return "【已执行动作】无（这是本次任务的第 1 步）"
    st = action_stats(actions)
    parts = []
    if st["fwd"]:
        parts.append(f"前进×{st['fwd']}")
    if st["left"]:
        parts.append(f"左转×{st['left']}")
    if st["right"]:
        parts.append(f"右转×{st['right']}")
    seq = "，".join(ZH_ACTION.get(a, str(a)) for a in actions[-tail:])
    more = f"（仅列出最近 {tail} 步）" if len(actions) > tail else ""
    turn_dir = "右" if st["yaw"] > 0 else ("左" if st["yaw"] < 0 else "无")
    return (
        f"【已执行动作】共 {st['n']} 步：{'、'.join(parts) if parts else '无'}\n"
        f"  · 累计前进距离 ≈ {st['dist']:.2f} 米（每次前进 {STEP_METERS} 米）\n"
        f"  · 相对本次任务起点的净转向 ≈ {abs(st['yaw']):.0f}° 向{turn_dir}"
        f"（每次转向 {TURN_DEGREES:.0f}°）\n"
        f"  · 动作序列（旧→新）：{seq}{more}"
    )


def history_block_en(actions, tail=20):
    if not actions:
        return "[Actions already taken] None (this is step 1 of the task)"
    st = action_stats(actions)
    parts = []
    if st["fwd"]:
        parts.append(f"forward x{st['fwd']}")
    if st["left"]:
        parts.append(f"turn-left x{st['left']}")
    if st["right"]:
        parts.append(f"turn-right x{st['right']}")
    names = {FWD: "forward", LEFT: "left", RIGHT: "right", STOP: "stop"}
    seq = ", ".join(names.get(a, str(a)) for a in actions[-tail:])
    more = f" (last {tail} only)" if len(actions) > tail else ""
    turn_dir = "right" if st["yaw"] > 0 else ("left" if st["yaw"] < 0 else "none")
    return (
        f"[Actions already taken] {st['n']} step(s): "
        f"{', '.join(parts) if parts else 'none'}\n"
        f"  - accumulated forward distance ~ {st['dist']:.2f} m "
        f"({STEP_METERS} m per forward step)\n"
        f"  - net heading change ~ {abs(st['yaw']):.0f} deg to the {turn_dir} "
        f"({TURN_DEGREES:.0f} deg per turn)\n"
        f"  - action sequence (oldest to newest): {seq}{more}"
    )


def _obs_block_zh(num_history):
    if num_history > 0:
        return (f"- 历史观测：{num_history} 张按时间顺序排列的图像（最早→最近）\n"
                f"- 当前观测：最后一张图像")
    return "- 历史观测：无\n- 当前观测：唯一一张图像"


def _zh_options(num_actions):
    opts = "\n".join(f"{ZH_ACTION[a]}（{_zh_desc(a)}）" for a in _ORDER)
    example = "，".join(["前进"] * max(1, num_actions - 1) + ["停止"])
    return opts, example


def _zh_desc(a):
    return {FWD: f"沿当前朝向前进 {STEP_METERS} 米",
            LEFT: f"原地左转 {TURN_DEGREES:.0f} 度",
            RIGHT: f"原地右转 {TURN_DEGREES:.0f} 度",
            STOP: "停止并结束本次任务"}[a]


# ----------------------------------------------------------------------------
# templates
# ----------------------------------------------------------------------------
def _t_en_r1(instruction, num_history, num_actions, actions):
    """Stock training-format prompt. No action history -- kept as the baseline."""
    hist = (f"{num_history} historical observation image(s) in temporal order "
            f"(if there are more than 16 frames, only 16 are selected evenly "
            f"from the whole history)") if num_history > 0 else "None"
    option_list = "\n".join(EN_ACTION[k] for k in sorted(EN_ACTION))
    ex = ["A. Move forward 25 cm", "B. Turn left 30 degrees",
          "C. Turn right 30 degrees", "D. Stop"]
    example_str = ", ".join(ex[i % 4] for i in range(num_actions))
    return f"""
Imagine you are a robot designed for navigation tasks. Your instruction is {instruction!r}.
You are provided with:
- Historical observations: {hist}
- Current observation: the last image

The images are given in order: the historical observations first (oldest to newest), and your current observation last.

Your task is to select the next {num_actions} actions based on this information.
Options:
{option_list}

Answer the formatting requirements:
1. must contain {num_actions} action predictions
2. use full option text
3. be separated by commas
4. example: "{example_str}"
""".strip()


def _t_en_r1_hist(instruction, num_history, num_actions, actions):
    """Stock prompt + action history. Closest to training, but self-aware."""
    base = _t_en_r1(instruction, num_history, num_actions, actions)
    hist = history_block_en(actions)
    marker = "Your task is to select"
    extra = (
        f"{hist}\n\n"
        f"Use the action history to judge progress. If the instruction states a "
        f"distance or an angle, compare it against the accumulated numbers above "
        f"and answer \"D. Stop\" as soon as the target is reached -- do not keep "
        f"moving after the goal is satisfied.\n\n"
    )
    return base.replace(marker, extra + marker, 1)


def _zh_output_rules(num_actions, example, extra=""):
    """Shared answer-format block.

    The loop re-queries the model after every batch, so the model must NOT
    pre-plan a stop: padding the list with 停止 makes it stop early (e.g. it
    answers 前进x3 + 停止 for a 1 m target and only travels 0.75 m). Stopping is
    a decision about the *past* -- emit it only once the history shows the goal
    is already met.
    """
    return f"""【输出格式】{extra}
最后一行以"答："开头，给出接下来最多 {num_actions} 个动作，用中文逗号分隔。
只能使用：前进、左转、右转、停止。
· 动作个数可以少于 {num_actions} 个，**不要用"停止"凑数**。
· 只有当上面的「已执行动作」已经表明任务完成时才输出"停止"，此时整行只写"答：停止"。
· 任务没完成就只给移动动作；这一批执行完后系统会带着新画面再问你一次。
例如：答：{example}"""


def _t_zh_nav(instruction, num_history, num_actions, actions):
    """General-purpose Chinese navigation prompt (default)."""
    opts, example = _zh_options(num_actions)
    return f"""你是一个室内导航机器人，正在第一人称视角下执行导航指令。

【任务指令】{instruction}

【你看到的画面】
{_obs_block_zh(num_history)}
图像按时间顺序给出：先是历史观测（由早到近），最后一张是你此刻看到的画面。

{history_block_zh(actions)}

【可选动作】
{opts}

【决策要求】
1. 先看当前画面：正前方是否可通行？是否已经到达指令描述的目标？
2. 结合历史画面和已执行动作，判断任务完成到什么程度，不要重复来回打转。
3. 目标已达成、或指令要求的距离/角度已经走够，就输出"停止"。
4. 前方被墙壁或家具挡住时，先转向再前进。

{_zh_output_rules(num_actions, example)}"""


def _t_zh_metric(instruction, num_history, num_actions, actions):
    """Chinese prompt tuned for metric commands: 'go forward 1 m', 'turn 90 deg'."""
    opts, example = _zh_options(num_actions)
    st = action_stats(actions)
    return f"""你是一个室内导航机器人。请严格按照指令中的**数值**执行动作，并在达标后立即停止。

【任务指令】{instruction}

【你看到的画面】
{_obs_block_zh(num_history)}
图像按时间顺序给出：先是历史观测（由早到近），最后一张是你此刻看到的画面。

{history_block_zh(actions)}

【单位换算表】
- 前进 1 次 = {STEP_METERS} 米；1 米 = 4 次前进；2 米 = 8 次；0.5 米 = 2 次
- 转向 1 次 = {TURN_DEGREES:.0f} 度；90 度 = 3 次转向；180 度 = 6 次；45 度 ≈ 1~2 次

【进度核对】（先算这一步，最多写 3 行）
- 指令要求前进多少米？记为 D。已前进 {st['dist']:.2f} 米。
  还需前进次数 = round((D - {st['dist']:.2f}) / {STEP_METERS})。
- 指令要求转多少度？记为 A。已净转 {abs(st['yaw']):.0f} 度。
  还需转向次数 = (A - {abs(st['yaw']):.0f}) / {TURN_DEGREES:.0f}。
- 还需次数 ≤ 0 → 已经走够了，输出"停止"。
- 还需次数 > {num_actions} → 本轮就输出 {num_actions} 个移动动作，一个"停止"都不要加，
  剩下的下一轮再走。

【可选动作】
{opts}

{_zh_output_rules(num_actions, example, "（计算过程不超过 3 行）")}"""


def _t_zh_cot(instruction, num_history, num_actions, actions):
    """Chinese prompt that asks for short explicit reasoning before the answer."""
    opts, example = _zh_options(num_actions)
    return f"""你是一个室内导航机器人，需要先分析再决策。

【任务指令】{instruction}

【你看到的画面】
{_obs_block_zh(num_history)}
图像按时间顺序给出：先是历史观测（由早到近），最后一张是你此刻看到的画面。

{history_block_zh(actions)}

【可选动作】
{opts}

【请按下面四步简要分析，每步一句话】
1. 场景：当前画面里有什么？我在房间的哪个位置？
2. 进度：对照指令和已执行动作，任务完成到哪一步了？
3. 障碍：正前方 1 米内是否可通行？需要先转向吗？
4. 决策：接下来怎么走？若已完成任务则停止。

{_zh_output_rules(num_actions, example, "（分析控制在 4 行以内）")}"""


def _t_zh_terse(instruction, num_history, num_actions, actions):
    """Minimal Chinese prompt -- fastest, least tokens, no reasoning requested."""
    opts, example = _zh_options(num_actions)
    return f"""室内导航机器人。指令：{instruction}

图像按时间顺序给出（最后一张是当前画面）。
{history_block_zh(actions)}

可选动作：
{opts}

只输出一行，不要任何解释：以"答："开头，跟最多 {num_actions} 个动作（中文逗号分隔）。
只有「已执行动作」显示任务已完成时才输出"停止"，否则只给移动动作、不要用"停止"凑数。
例如：答：{example}"""


# Only templates that earn their place stay here. _t_zh_cot / _t_zh_terse /
# _t_en_r1_hist above are kept but unregistered -- they were never validated
# against the model; add a line here to bring one back.
# en_r1 first and default: it is the format the model was actually trained and
# evaluated in, so it is what the classic driver should show unprompted -- the
# two Chinese templates are variations to compare against it, not the baseline.
TEMPLATES = {
    "en_r1":     ("English·原始训练格式", "默认。与评测完全一致的基线，无动作历史", _t_en_r1),
    "zh_nav":    ("中文·通用导航", "含动作历史，适合“走到桌子旁边”这类没有数值的指令", _t_zh_nav),
    "zh_metric": ("中文·量化执行", "带米/度换算表与进度核对，适合“往前走一米”这类带数值的指令", _t_zh_metric),
}
DEFAULT_TEMPLATE = "en_r1"


# ----------------------------------------------------------------------------
# drivers: how the model is allowed to move the robot
# ----------------------------------------------------------------------------
# This is a separate axis from TEMPLATES on purpose. A template is
# fn(instruction, num_history, num_actions, actions) -> one user string; the
# ReAct driver needs a system prompt plus tool schemas plus a multi-turn
# conversation, which does not fit that signature, and "动作数" would silently
# come to mean "max tool calls". Registering it as a 4th template would force
# run_model to branch on the key anyway.
DRIVERS = {
    "legacy": ("经典·多动作",
               "一次输出最多 N 个离散动作（前进 0.25 m / 转 30°），顺序执行后再推理。"
               "与原始评测同构。"),
    "react":  ("ReAct·工具调用",
               "模型以工具调用驱动：前进/后退给米数（各 ≤1 m）、"
               "转向给角度（≤90°），连续量，执行后把新画面作为工具返回喂回去。"),
    "embodied_point": ("ReAct·Embodied Point",
                        "保留 ReAct 与自主 stop；仅把 move_forward 换成单视角 "
                        "move_to_point(point_2d)，千分位点经同帧深度反投影后执行。"),
    "unilavira_bbox": ("ReAct·Uni-LaViRA BBox",
                        "保留 ReAct 与自主 stop；仅把 move_forward 换成单视角 "
                        "move_to_bbox(bbox_2d)，底边中心邻域深度反投影后动态重规划。"),
    "action_head": ("FM Head · 自选 checkpoint",
                    "从 artifacts 选择小脑 checkpoint，确认后加载；大脑由模型栏独立选择。"),
    "action_head_10ep": ("FM Head as Tool（10ep / 当前部署）",
                    "保留 Qwen3.6 ReAct 大脑，仅将 primitive tools 替换为 "
                    "run_action_head + stop；小脑加载当前部署指定的 10-epoch 系列 checkpoint，"
                    "并使用同一 snapshot 的四-query embedding。当前 checkpoint 以控制台启动参数为准。"),
    "action_head_planner": ("视觉指令 Planner（v4.3 / 实验）",
                    "保留 v4.3 大脑 tool call，视觉落地有序地面路点，使用特权碰撞地图规划局部路径；不读取任务目标坐标。"),
    "action_head_v43_7000": ("FM Head as Tool（v4.3 / step 7000）",
                    "加载 v4.3 official-GVLN D16 开环训练的逻辑 step 7000 checkpoint；"
                    "16-query embedding 与 FM Head 来自同一 snapshot，测试集固定为 held-out 100 cases。"),
    "action_head_v5_native": ("FM Head as Tool（v5 / 原生图文条件）",
                    "完整历史与当前tool call独立编码，不生成大脑query；聚合器未训练，当前用于真实在线链路测速。"),
    "action_head_v5_cached": ("FM Head as Tool（v5 / 历史图像缓存）",
                    "逐帧缓存原生视觉特征，每次聚合完整历史与当前tool call；聚合器未训练，用于在线测速。"),
    "action_head_2800": ("FM Head as Tool（训练 2800 steps）",
                    "保留 Qwen3.6 ReAct 大脑，仅将 primitive tools 替换为 "
                    "run_action_head + stop；小脑为 5000-step 长训第 2800 step snapshot 的 vision + flow-matching head，"
                    "与当前 28899 慢模型四-query embedding（同一 2800 snapshot）严格配套。"),
    "action_head_untrained": ("FM Head as Tool（未训练 / 训练起点）",
                    "保留同一 Qwen3.6 ReAct 大脑，仅切换到本次训练实际使用的未训练 FM head 初始权重。"),
    "fm":     ("FM·流匹配轨迹",
               "进程内加载带 FM 头的 Qwen3-VL 模型（不走 vLLM），每步从当前画面 + 零历史 "
               "input_waypoint 预测一条 32×3 的 ego (dx,dy,dyaw) 归一化轨迹 chunk，"
               "denorm 后逐 waypoint 用 fm_action 落到 sim（turn→fwd→turn 分解 + 子步防穿墙）。"
               "部署口径对齐 verl_nav FM-GRPO 的 ODE 单帧 cond。"),
    "lightnav": ("LightNav-0·轨迹",
                 "通过独立 WebSocket 服务调用 LightNav-0；服务端保留 SlowFast 图像历史，"
                 "每轮输出 10×3 ego waypoint。按其原生 Habitat 口径执行首个有效点："
                 "lateral 丢弃、前进≤0.25m、转向≤30°，并记录适配损失。"),
}
# Keep the primitive ReAct tool loop as the production default. Point and bbox
# supplied by pi-agent are first-frame visual references; they must not switch
# the lower-level controller to a depth-projected point policy.
DEFAULT_DRIVER = "react"
MAX_CALLS_PER_TURN = int(os.environ.get('VLN_PRIMITIVE_MAX_CALLS_PER_TURN', '4'))
if not 1 <= MAX_CALLS_PER_TURN <= 4:
    raise ValueError('VLN_PRIMITIVE_MAX_CALLS_PER_TURN must be 1–4')

# Total tool calls one ReAct run may make -- the only budget the model is told
# about (it rides on every batch's last tool return via append_tool_budget), so
# it is the one worth putting on the page instead of burying in a constant.
# Keep the default run short enough for interactive use. The page can still
# raise it when a longer navigation task needs more steps.
DEFAULT_MAX_TOOL_CALLS = 10
TOOL_BUDGET_MIN, TOOL_BUDGET_MAX = 1, 500


def driver_list():
    """[(key, label, desc)] for the web UI dropdown."""
    return [(k, v[0], v[1]) for k, v in DRIVERS.items()
            if k in ('action_head', 'action_head_planner') or not k.startswith('action_head')]


def canonical_driver(key):
    """Keep old bookmarks/API callers on the single checkpoint picker."""
    if key.startswith('action_head') and key != 'action_head_planner':
        return 'action_head'
    return key


# ----------------------------------------------------------------------------
# selectable backends
# ----------------------------------------------------------------------------
# Model services run separately and the UI switches endpoints without restarting
# them. Each wire name must match its launcher's --served-model-name; Next uses
# "qwen3.8-flash-next" on port 9038.
#
# General launchers: vllm-vln-35B.sh, vllm-vln-qwen35-4B.sh and vllm-vln-2B.sh.
# The four-query action expert is isolated in vllm-action expert-query.sh.
# The third field must match that script's --served-model-name exactly. The first
# is only ever shown on the page, which is why it can carry the full checkpoint
# name (Qwen3.6-35B-A3B) while the wire name stays the shorter one the server was
# actually started under -- widening the third field to match would 404.
MODELS = {
    "qwen38_flash_next": (
        "Qwen3.8-Flash-Next · vLLM",
        "http://127.0.0.1:9038",
        "qwen3.8-flash-next",
        "多模态后端（PPU 4-11，TP8/DP1，端口 9038）；"
        "支持思维链、图片和工具调用，262144 token 上下文。"),
    # 28899 is reserved by the action expert hidden-state experiment.  The
    # ordinary VLN endpoint must stay separate: that diagnostic server may use
    # a much smaller context window and a custom worker.
    "qwen35b": ("Qwen3.6-35B-A3B · vLLM", "http://127.0.0.1:28900", "Qwen3.6-35B",
                "默认导航后端（MoE，2 卡）。没在 VLN 上专门训过，靠通用能力导航。"),
    "qwen38_27b_sglang": (
        "Qwen3.8-27B · SGLang",
        "http://127.0.0.1:9000",
        os.environ.get("VLN_27B_MODEL_NAME", "/opt/external/user-d5121be6/model/Qwen3.8-27B"),
        "SGLang 部署的 Qwen3.8-27B（端口 9000）；支持独立 reasoning_content "
        "思维链流，用于和当前 vLLM Qwen3.6-35B 对照。"),
    "qwen35_4b": ("Qwen3.5-4B", "http://127.0.0.1:28901", "Qwen3.5-4B",
                   "通用多模态模型（4B dense，1 卡），原始 instruct 权重；"
                   "用于和 Qwen3.6-35B 做轻量基线对照。"),
    "mobot2b": ("MoBot_VLN-2B", "http://127.0.0.1:28898", "MoBot_VLN-2B",
                "在 VLN 上训过的小模型（2B dense，1 卡），"
                "final_mix_mt_me6/hf_iter_0001250。"),
}

# ---- VLN(FM) 模型：进程内加载带 FM 头的 Qwen3-VL，不走 vLLM ----
# 作为**模型选项**进下拉（不是驱动）；选中时**驱动锁死 FM·流匹配**，别的模型
# 不能选 FM 驱动。ckpt 来自 closed_loop 训练配置那份 fm_policy（0729 base 等）；
# env FM_CKPT 可覆盖任意一条。url/served_name 留空——FM 不走 vLLM。
FM_MODELS = {
    "vlnfm0729": ("VLN-FM·0729 base",
                  "/data/shared/groups/mfm/vl/exp/qwen3_vl_2b_fm/fm_baseline_fixed/"
                  "qwen3vl_2b_fm_0729-baseline-cocofix-0p5-align_tp2_16nodes_mbs8_"
                  "20260802_012647/checkpoint_tp2_hf_16k",
                  "进程内加载带 FM 头的 Qwen3-VL（closed_loop 训练配置 fm_policy + "
                  "0729 base），驱动锁死 FM·流匹配，不走 vLLM。env FM_CKPT 可覆盖。"),
}
FM_MODEL_KEYS = set(FM_MODELS)
# 把 FM 模型并进下拉（url/served_name 空）；保持原 vLLM 模型在前。
MODELS.update({k: (v[0], "", "", v[2]) for k, v in FM_MODELS.items()})

# ---- LightNav 独立策略：同样作为“模型”让用户明确选择 ----
# 它不走普通 chat-completions/vLLM endpoint，而是由 LightNav WebSocket
# 服务持有时序历史并直接输出 waypoint。把它放在模型下拉后锁定 lightnav
# 驱动，避免页面出现“选的是 Qwen，实际却绕过它跑 LightNav”的错觉。
LIGHTNAV_MODELS = {
    "lightnav0": (
        "LightNav-0",
        "独立视觉导航策略；通过 LightNav WebSocket 读取当前第一视角与自然语言指令，"
        "输出 ego waypoint，并在第一视角显示 APOS/OPOS。选中后驱动锁定 LightNav-0·轨迹。",
    ),
}
LIGHTNAV_MODEL_KEYS = set(LIGHTNAV_MODELS)
MODELS.update({k: (v[0], "", "", v[1]) for k, v in LIGHTNAV_MODELS.items()})

DEFAULT_MODEL = "qwen35b"
MODEL_CONTEXT_WINDOWS = {
    "qwen38_flash_next": 262144,
    "qwen35b": 131072,
    "qwen38_27b_sglang": int(os.environ.get("VLN_27B_CONTEXT_LENGTH", "262144")),
    "qwen35_4b": 131072,
    "mobot2b": 131072,
}


def is_fm_model(key):
    """True 当 key 是进程内 FM 模型（驱动须锁 fm，不走 vLLM）。"""
    return key in FM_MODEL_KEYS


def is_lightnav_model(key):
    """True when ``key`` is the independently served LightNav policy."""
    return key in LIGHTNAV_MODEL_KEYS


def fm_ckpt_for(key):
    """FM 模型 key → ckpt 绝对路径；非 FM 返回 None。"""
    ent = FM_MODELS.get(key)
    return ent[1] if ent else None


def allowed_drivers_for(key):
    """Return the driver set compatible with one selectable model."""
    if is_fm_model(key):
        return ["fm"]
    if is_lightnav_model(key):
        return ["lightnav"]
    return ["legacy", "react", "embodied_point", "unilavira_bbox"]


def model_list():
    """[(key, label, desc, is_fm, is_lightnav)] for the web UI."""
    return [(k, v[0], v[3], is_fm_model(k), is_lightnav_model(k))
            for k, v in MODELS.items()]


def model_endpoint(key):
    """(url, served_model_name) for `key`, falling back to the default.

    FM/LightNav 模型返回 ("", "")——dispatch 不会调用 chat-model endpoint。"""
    m = MODELS.get(key) or MODELS[DEFAULT_MODEL]
    return m[1], m[2]


# Endpoint capabilities, independent of parameter count. Other model endpoints
# keep their existing textual protocol unless explicitly enabled here.
MODEL_NATIVE_TOOL_CALLS = {"qwen38_flash_next"}


def model_context_window(key):
    """Declared server context window, used only for UI/console headroom."""
    return MODEL_CONTEXT_WINDOWS.get(key, 131072)


# A custom system prompt that drops the tool block leaves the model with no way
# to call anything -- the run then ends on the first reply with "没有工具调用".
# The UI prefills its editor with the default text (tools included) so the common
# path is safe, and warns on this marker's absence rather than silently
# re-appending the block: a prompt you did not write is worse than a warning.
TOOL_BLOCK_MARKER = "<tools>"


def prompt_has_tools(text):
    """Does this (custom) system prompt still carry the tool schemas?"""
    return TOOL_BLOCK_MARKER in (text or "")


def agent_visual_context(reference_kind=None):
    """Runtime image contract for the active ReAct reference mode."""
    if reference_kind == "upload":
        return """【本轮首轮视觉输入：额外参考图模式】
首轮 user message 一共包含两张图片，顺序固定：
1. 第 1 张是机器人当前的真实起点第一人称画面。紧随其后的文字说明这就是当前画面；
   导航、避障和第一次动作必须以它为准。
2. 第 2 张是用户提供的额外参考图。它不是机器人当前观测，也不是历史轨迹画面；
   只用它理解目标外观、物体类别或用户想强调的视觉信息。
导航指令放在第 2 张图片之后。后续工具返回的图片都是机器人移动后的实时第一人称画面，
不再重复参考图。不要把第 2 张参考图误当成可直接移动的场景。"""
    if reference_kind == "bbox":
        return """【本轮首轮视觉输入：红框指示模式】
首轮 user message 包含一张机器人当前的第一人称画面，用户指定的 bbox 已作为红色矩形框直接绘制在图片上。
紧随图片的文字会给出 bbox 的 0 到 1000 归一化整数坐标。红框内的物体或区域是本次局部移动的视觉目标；
请结合完整画面判断可行路径。后续工具返回的是移动后的实时第一人称画面，不带红框。"""
    if reference_kind == "point":
        return """【本轮首轮视觉输入：视觉点指示模式】
首轮 user message 包含一张机器人当前的第一人称画面，用户指定的 point 已作为红色十字圆环画在图片上。
该 point 必须落在当前画面可见、可行走的地面上，它表示本次要接近的地面位置；接近物体时应选择物体前方的地面，不要点击物体、墙面或家具表面。
请使用经典 ReAct 的转向、前进、后退工具，结合每一步返回的新画面逐步接近红色标记区域并自主避障。
后续工具返回移动后的实时第一人称画面，不带该红色标记。"""
    return ""


def shared_tool_agent_prompt(tool_specs):
    """Tool-agnostic navigation brief used by strict primitive/AH A/B.

    The rendered text is identical for both profiles up to the generated tool
    block. Tool-specific behavior belongs in ToolSpec descriptions, never in
    observation construction or the common navigation policy.
    """
    import agent_tools as A
    return f"""你是一个室内导航机器人，通过调用当前提供的工具完成用户给出的导航指令。

【你能看到什么】
每次工具执行完，系统会把工具返回的非 oracle 执行事实和机器人最新的第一人称画面交给你。
最后一张图永远是当前画面，较早的画面可能因为篇幅被省略。

【怎么决策】
· 只使用导航指令、当前可见画面、保留的历史和工具返回事实；不得假设隐藏的目标坐标或未来路线。
· 根据当前提供的工具说明选择下一步。工具执行完后你会收到新画面，再继续决策。
· 工具返回的实际运动和 blocked 信息优先于你之前预期的执行结果。

【什么时候停】
确认已经到达导航指令描述的最终目标时，调用 stop 结束完整任务。
不确定就继续使用当前可用工具，不要提前停止。任何一次回复里没有工具调用，也会被当作结束。

{A.build_system_prompt(tool_specs)}""".strip()


def agent_system_prompt(custom=None, reference_kind=None, tool_specs=None,
                        action_head_as_tool_version="v1"):
    """System prompt for the ReAct driver: navigation brief + tool schemas.

    The tool block is generated by agent_tools.build_system_prompt (copied from
    Agentic_MoBot) so the wire format matches what that harness uses. The brief
    above it is Chinese because every other prompt this model sees here is, and
    because the plain-language rules are what it actually follows -- it was
    never trained on JSON tool observations.

    ``custom`` (a prompt hand-edited in the web UI) replaces the base prompt,
    tool block included. The short runtime image contract is still appended
    when a reference is active because it describes the actual wire input, not
    a user-editable navigation policy.
    """
    from action_head import protocol as AHP
    version = AHP.normalize_version(action_head_as_tool_version)
    protocol = AHP.protocol_module(version)
    if protocol is not None:
        if custom and custom.strip():
            raise ValueError(
                f"action_head_as_tool_version={version} requires the canonical system prompt")
        if reference_kind is not None:
            raise ValueError(
                f"action_head_as_tool_version={version} does not allow a prompt-changing reference mode")
        return protocol.SYSTEM_PROMPT
    if custom and custom.strip():
        base = custom.strip()
    elif tool_specs is not None:
        base = shared_tool_agent_prompt(tuple(tool_specs))
    else:
        import agent_tools as A
        base = f"""你是一个室内导航机器人，通过调用工具来移动自己，完成用户给出的导航指令。

【你能看到什么】
每次工具执行完，系统会把机器人最新的第一人称画面作为工具返回结果给你。
最后一张图永远是当前画面。较早的画面可能因为篇幅被省略。

【怎么行动】
· 用 move_forward 前进，参数是**米**（一次最多 {A.MAX_FWD_M:.0f} 米）；
  用 turn 转向，参数是方向和**度数**（一次最多 {A.MAX_TURN_DEG:.0f} 度）。
  距离和角度都是连续值，不必凑整——需要走 0.8 米就写 0.8。
· 用 move_backward 后退，一次最多 {A.MAX_BACK_M:.0f} 米，朝向不变。
  摄像头朝前，后退时看不见身后：贴墙卡住了先退一点再转向；
  想去身后较远的地方还是**转身再前进**更稳。
· 一次回复里可以给出多个工具调用，它们会**按顺序执行**，最多 {MAX_CALLS_PER_TURN} 个。
  执行完你会拿到新画面，再决定下一步。
· 每个工具返回里都有实际执行的结果：`moved_m` 是**真实走过的距离**。
  被挡住时它会小于你请求的距离，且 `blocked` 为 true——这时不要硬顶，先转向再走。
· `total_forward_m` 和 `net_yaw_deg` 是本次任务累计的前进距离和净转向角度
  （后退会把 `total_forward_m` 减回去）。
  指令里带数值时（"往前走一米"），拿它们对照，够了就停。

【什么时候停】
确认已经到达指令描述的目标、或指令要求的距离/角度已经走够，就调用 stop 结束。
不确定就继续观察和移动，不要提前停。任何一次回复里没有工具调用，也会被当作结束。

{A.build_system_prompt(A.TOOL_SPECS)}"""
    visual = agent_visual_context(reference_kind)
    return base + ("\n\n" + visual if visual else "")


def agent_task_text(instruction):
    """Navigation instruction text placed after the first-turn visual context."""
    return f"【导航指令】{instruction}"


def agent_current_view_text():
    """Text that must immediately follow the current first-person image."""
    return "上面是你此刻的第一人称画面。请开始执行，调用工具移动。"



# ----------------------------------------------------------------------------
# thinking / non-thinking sampling modes
# ----------------------------------------------------------------------------
# Sampling presets lifted from curl-Instruct.sh / curl-Thinking.sh. The model is
# a reasoning model: with enable_thinking it emits a long <think> block before
# the answer, which is why the token budget has to be generous -- a truncated
# reply loses the "答：" line entirely and parses to zero actions.
MODES = {
    "think": {
        "label": "思考模式",
        "desc": "模型先推理再作答，更准但每步 10-20s。默认。",
        "enable_thinking": True,
        "max_tokens": 40960,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
    "instruct": {
        "label": "非思考模式",
        "desc": "跳过推理直接给动作，快 3-5 倍，复杂指令上略弱。",
        "enable_thinking": False,
        "max_tokens": 32768,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
}
DEFAULT_MODE = "think"


def mode_list():
    """[(key, label, desc)] for the web UI dropdown."""
    return [(k, v["label"], v["desc"]) for k, v in MODES.items()]


def sampling_params(mode, temperature=None):
    """Sampling kwargs for the vLLM payload.

    `temperature` overrides the preset when the user moved the slider; passing
    None keeps the mode's own value.
    """
    m = MODES.get(mode, MODES[DEFAULT_MODE])
    p = {k: m[k] for k in ("max_tokens", "temperature", "top_p", "top_k",
                           "min_p", "presence_penalty", "repetition_penalty")}
    if temperature is not None:
        p["temperature"] = float(temperature)
    return p, m["enable_thinking"]


def template_list():
    """[(key, label, desc)] for the web UI dropdown, in declaration order."""
    return [(k, v[0], v[1]) for k, v in TEMPLATES.items()]


def build_prompt(style, instruction, num_history, num_actions, actions):
    fn = TEMPLATES.get(style, TEMPLATES[DEFAULT_TEMPLATE])[2]
    return fn(instruction, num_history, num_actions, actions or [])


def build_conversation(style, instruction, hist_imgs, curr_img, num_actions, actions):
    """Images first (oldest to newest, current last), prompt text last.

    Same layout as rynnvln_eval.construction_conversation_embodied so the model
    sees the frame ordering it was trained on, whichever template is selected.
    """
    content = []
    for img in hist_imgs:
        w, h = img.size
        content.append({"type": "image", "image": img, "width": w, "height": h})
    w, h = curr_img.size
    content.append({"type": "image", "image": curr_img, "width": w, "height": h})
    content.append({"type": "text", "text": build_prompt(
        style, instruction, len(hist_imgs), num_actions, actions)})
    return [{"role": "user", "content": content}]


# ----------------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------------
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_SEG_SPLIT = re.compile(r"[,，、\n;；]")

# Checked in order; the first hit wins, so "停止" must precede the direction
# words (a segment like "前进到门口后停止" is a stop).
_KEYWORDS = [
    (STOP, ("停止", "停下", "停车", "结束", "stop")),
    (FWD, ("前进", "向前", "直行", "往前", "forward")),
    (LEFT, ("左转", "向左", "往左", "turn left", "left")),
    (RIGHT, ("右转", "向右", "往右", "turn right", "right")),
]
_LETTER = {"a": FWD, "b": LEFT, "c": RIGHT, "d": STOP}


def _seg_to_action(seg):
    s = seg.strip().lower()
    if not s:
        return None
    m = re.match(r"^\(?([abcd])[\.\):．）]", s) or re.match(r"^([abcd])$", s)
    if m:
        return _LETTER[m.group(1)]
    for act, words in _KEYWORDS:
        for w in words:
            if w in s:
                return act
    m3 = re.search(r"\b([ABCD])\b", seg)
    if m3:
        return _LETTER[m3.group(1).lower()]
    return None


def parse_actions(output, num_actions=4):
    """Extract the action list from a model reply.

    Accepts Chinese ("答：前进，前进，左转") and the English training format
    ("A. Move forward 25 cm, B. Turn left 30 degrees"). Reasoning before the
    answer is ignored: the answer line is preferred, otherwise the last line
    that parses to at least one action is used.
    """
    if not output:
        return []
    text = _THINK_RE.sub(" ", output)

    # 1. explicit answer marker wins (last occurrence)
    cand = None
    idx = max(text.rfind("答："), text.rfind("答:"), text.rfind("答 "))
    if idx >= 0:
        cand = text[idx + 1:].lstrip("：: ").split("\n")[0]

    lines = [l for l in text.strip().split("\n") if l.strip()]

    # 2. otherwise: the last line that yields any action
    if not cand:
        for line in reversed(lines):
            if any(_seg_to_action(s) is not None for s in _SEG_SPLIT.split(line)):
                cand = line
                break
    if not cand:
        return []

    seq = [a for a in (_seg_to_action(s) for s in _SEG_SPLIT.split(cand))
           if a is not None]

    # 3. one-action-per-line answers ("1. 前进\n2. 左转\n..."): the single line
    # above only gave us the final action, so walk back over the trailing run
    # of single-action lines and rebuild the whole list.
    if len(seq) == 1 and cand in lines:
        run, i = [], lines.index(cand)
        while i >= 0:
            got = [a for a in (_seg_to_action(s) for s in _SEG_SPLIT.split(lines[i]))
                   if a is not None]
            if len(got) != 1:
                break
            run.append(got[0])
            i -= 1
        if len(run) > 1:
            seq = list(reversed(run))
    if len(seq) > num_actions:
        seq = seq[:num_actions]
    # Nothing after a stop can be executed.
    if STOP in seq:
        seq = seq[:seq.index(STOP) + 1]
    return seq

if os.environ.get("VLN_DISABLE_LOCAL_27B") == "1":
    MODELS.pop("qwen38_27b_sglang", None)
