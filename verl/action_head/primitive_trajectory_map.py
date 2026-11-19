"""Primitive ReAct map adapter; reuses the v6 renderer and current-view injection."""
import hashlib
import math

from .trajectory_map import (TrajectoryMapState, inject_trajectory_map,
                             TRAJECTORY_MAP_SCHEMA, TRAJECTORY_MAP_RENDERER_SHA256)

MAP_BRIEF = """每次决策还会在当前第一人称画面之前提供一张最新的累计轨迹图。
图例：绿色 S 为起点，蓝色 C 为当前位置，重合时标为 S/C；蓝线按执行顺序连接实际移动终点，深蓝箭头为当前朝向；橙框表示执行返回受阻的位置，红叉表示执行返回碰撞的位置。
图中向上为起点的前方，向左为起点的左方。half-width 是图中心到每侧边缘的米数，例如 half-width 4 m 表示 8×8 米范围；随着轨迹延长，图幅可能扩大。
轨迹图与工具返回的 actual_end_pose 使用实际执行的相对位移和转角，而非请求的距离；actual_end_pose 的 x_m 向前、y_m 向左、yaw_deg 左转为正，以该次调用起点为参考。
可以参考轨迹图判断重访、绕圈、回退、反复受阻和整体位移，并结合最新第一人称画面重新决定下一步。"""


def system_prompt(base, enabled):
    if not enabled:
        return base
    marker = "【怎么决策】"
    if base.count(marker) != 1:
        raise ValueError('primitive base prompt changed; map placement requires audit')
    # Match the actual default feedback/retry runtime, not the obsolete stop rule.
    base = base.replace("任何一次回复里没有工具调用，也会被当作结束。",
                        "工具调用格式或参数错误时，会收到错误反馈；请在剩余预算内修正并继续调用工具。")
    return base.replace(marker, MAP_BRIEF + "\n\n" + marker)


def identity(prompt, enabled):
    return {'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'trajectory_map': bool(enabled),
            'trajectory_map_schema': TRAJECTORY_MAP_SCHEMA if enabled else None,
            'trajectory_map_renderer_sha256': TRAJECTORY_MAP_RENDERER_SHA256 if enabled else None,
            'primitive_odometry_revision': 'call_relative_forward_left_v1' if enabled else None}


def update_from_tool(state, name, arguments, result):
    """Return the exact whitelist also appended to the model-visible tool result."""
    if result.get('ok') is False or result.get('return_reason') in ('rejected', 'error'):
        return {}
    if name == 'turn':
        turned = float(result['turned_deg'])
        if not math.isfinite(turned) or turned < 0 or arguments['direction'] not in ('left', 'right'):
            raise ValueError('invalid measured primitive turn')
        state.update_turn(dict(ok=True, direction=arguments['direction'], turned_deg=turned))
        return {}
    if name not in ('move_forward', 'move_backward'):
        return {}
    pose = result.get('actual_end_pose')
    if not isinstance(pose, dict) or not all(math.isfinite(float(pose.get(k, float('nan'))))
                                            for k in ('x_m', 'y_m', 'yaw_deg')):
        raise ValueError('primitive map requires measured relative actual_end_pose; no requested-distance fallback')
    safe = {'actual_end_pose': {k: float(pose[k]) for k in ('x_m', 'y_m', 'yaw_deg')},
            'blocked': bool(result.get('blocked')), 'collision': bool(result.get('collided'))}
    state.update_action_head([safe])
    return safe
