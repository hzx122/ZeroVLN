# -*- coding: utf-8 -*-
"""FM VLN 轨迹动作的 (dx, dy, dyaw) → habitat forward/turn 分解。

参考模型（date_mfm/.../run_train_qwenvl_2b_fm_tp2_fm_ar_perct_0728-...sh）
输出 action_dim=3 / action_chunk=32 的 ego 系相对位移 (dx, dy, dyaw)，
32 个 waypoint 组成一条未来轨迹。训练时按 _CE_WP_STATS 的 q01/q99 归一化到
[-1, 1]（源：date_mfm/projects/MegaTE/vla/reference/SocialNav/src/dataset/
vln_ce_dataset.py:83 的 _CE_WP_STATS，含 125cm_0deg spline 重采样后的统计）。

habitat 的 agent 只有 forward / turn_left / turn_right（无 strafe 原语），
所以一条 ego (dx, dy, dyaw) 要分解成三段：
    turn1 = atan2(dy, dx)        # 转到指向 (dx, dy) 位移方向
    fwd   = hypot(dx, dy)         # 沿该方向前进
    turn2 = dyaw - turn1           # 补转到净 heading 变化 = dyaw
不变量：turn1 + turn2 == dyaw（heading 总变化正确），位移方向正确。
正向 (dy>0) = 左转（ccw），与 atan2 / 标准 math 约定一致。

这里的分解纯做几何 + 调度，真正动 sim 的是 agent_tools.drive_turn / drive_forward
（它们带子步防穿墙、collision/stall 检测、finally-restore actuation）——
本模块不重复造那些逻辑，只把 waypoint 翻译成一串 drive_* 调用。

ego 系约定（与 SocialNav 数据一致）：
    dx  > 0 前进, dy > 0 左移, dyaw > 0 左转（ccw）。
若 smoke 实测发现 habitat 的 turn_left 与本约定方向相反，改 LEFT_POSITIVE 即可。
"""
import math
import os
from typing import Any

import numpy as np

# ----------------------------------------------------------------------------
# CE 数据集 stats（内联，不跨仓 import date_mfm；源见模块 docstring）
# 归一化方式：[q01, q99] 线性映射到 [-1, 1]，clamp 到 [-2, 2]（同 SocialNav
# _normalize_waypoints）。这里只存 q01/q99，反归一化把 [-1,1] 映回原尺度。
# ----------------------------------------------------------------------------
_CE_WP_STATS = {
    "dx":   {"q01": -0.10273411870002747, "q99":  0.1031249612569809},
    "dy":   {"q01": -0.10309220850467682, "q99":  0.10309189558049175},
    "dyaw": {"q01": -0.5668500065803528,  "q99":  0.5747644901275635},
}

# ego 约定开关：True=正角度左转(ccw)。若 smoke 实测 habitat 左右反了改 False。
LEFT_POSITIVE = True

# 符号约定开关（P1 探针用，生产默认全 False = 与历史行为逐字一致）。
# 0729 base 训练用 Y右/yaw顺时针为正，当前 builder+executor 是 Y左/yaw逆时针为正；
# 探针判定是否需要把模型输出的 dy/dyaw 翻号后再消费。翻转点放消费边界
# （denormalize 之后、cumsum 之前），training 侧不经 fm_action 不受影响。
# 动态读 env（非模块级缓存）——符号探针需在同进程内切 4 组合。
# ABOTN_BASE_CONVENTION=1（默认）= SFT 数据已对齐 base Y右约定（relative_future 翻
# dy/dyaw + ego_relative_xy 翻 y），故执行器消费时须把模型 Y右 输出翻回 Y左 喂 abotn
# sim——此时 FLIP_DY/DYAW 默认 1。显式 ABOTN_FLIP_DY/DYAW 覆盖 BASE_CONV。
def _flip_dy():
    v = os.environ.get("ABOTN_FLIP_DY")
    if v is not None:
        return v == "1"
    return os.environ.get("ABOTN_BASE_CONVENTION", "1") == "1"


def _flip_dyaw():
    v = os.environ.get("ABOTN_FLIP_DYAW")
    if v is not None:
        return v == "1"
    return os.environ.get("ABOTN_BASE_CONVENTION", "1") == "1"

_RAD2DEG = 180.0 / math.pi


def denormalize(waypoints, normalized=True, stats=None):
    """把 [[dx,dy,dyaw], ...] 从 [-1,1] 反归一化回原尺度。

    normalized=False 时原样返回（caller 已经给真实米/弧度）。
    归一化是 [q01,q99]→[-1,1]，反归一化是 [-1,1]→[q01,q99]：
        v_real = q01 + (v_norm + 1) * 0.5 * (q99 - q01)
    """
    if not normalized:
        return [[float(x) for x in wp] for wp in waypoints]
    stats = stats or _CE_WP_STATS
    q01 = [stats["dx"]["q01"], stats["dy"]["q01"], stats["dyaw"]["q01"]]
    q99 = [stats["dx"]["q99"], stats["dy"]["q99"], stats["dyaw"]["q99"]]
    out = []
    for wp in waypoints:
        real = []
        for i, v in enumerate(wp):
            vn = max(-2.0, min(2.0, float(v)))  # clamp 同训练侧
            real.append(q01[i] + (vn + 1.0) * 0.5 * (q99[i] - q01[i]))
        # 符号翻转在反归一化之后（消费边界）：dy=通道1、dyaw=通道2
        if _flip_dy():
            real[1] = -real[1]
        if _flip_dyaw():
            real[2] = -real[2]
        out.append(real)
    return out


# ----------------------------------------------------------------------------
# chunk-frame 协议（P0 修复）
# 0729/SocialNav 的 32 个 delta 都在 chunk 起点 fixed ego frame，恢复轨迹需
# np.cumsum（官方 train_clean/data.py:denormalize_and_cumsum）。当前闭环执行器
# 对每条 delta 从"已转向后的当前 heading"再做一次 atan2(dy,dx)（execute_waypoint
# → decompose），转弯后第 2 步起 fixed-frame delta 被当 rolling-local delta 二次
# 旋转。修法：先 cumsum 成 chunk-frame 绝对轨迹，每步读真实 sim pose 换算成从
# 当前 pose 出发的 ego (dx,dy,dyaw) 再喂 execute_waypoint。
# ----------------------------------------------------------------------------
def chunk_frame_trajectory(real_deltas):
    """denormalized 单步 delta (N,3) → chunk 起点 frame 的绝对 (x,y,yaw) 轨迹。

    yaw 从 0 累积，与 SocialNav denormalize_and_cumsum 逐字同款。输入可为 list
    of list 或 ndarray，返回 (N,3) float64 ndarray。
    """
    arr = np.asarray(real_deltas, dtype=np.float64)
    return np.cumsum(arr, axis=0)


def world_to_chunk_pose(world_xy, world_yaw_rad, start_xy, start_yaw_rad):
    """world 系 pose → chunk 起点 frame 的 (x,y,yaw)。

    与 build_abotn_sft_data.relative_future 的 R(-yaw0) 同系，保证 SFT 标签 frame
    与执行 frame 一致：
        cx = cos(yaw0)*(wx-x0) + sin(yaw0)*(wy-y0)
        cy = -sin(yaw0)*(wx-x0) + cos(yaw0)*(wy-y0)
        cyaw = wyaw - yaw0
    """
    c, s = math.cos(start_yaw_rad), math.sin(start_yaw_rad)
    dx = float(world_xy[0]) - float(start_xy[0])
    dy = float(world_xy[1]) - float(start_xy[1])
    cx = c * dx + s * dy
    cy = -s * dx + c * dy
    cyaw = float(world_yaw_rad) - float(start_yaw_rad)
    return (cx, cy, cyaw)


def target_to_local_command(target_xyz, cur_xyz):
    """chunk-frame 绝对 target (x,y,yaw) → 从当前 chunk-frame pose 出发的
    **local-ego** (dx,dy,dyaw)，供 execute_waypoint 消费。

    关键：execute_waypoint.decompose 把 (dx,dy) 当**当前 heading 系**位移（turn1=
    atan2(dy,dx) 是相对当前 heading 的转角）。所以 chunk-frame 位移 (DX,DY)=
    (target-cur) 必须先旋到当前 heading 系（R(-cyaw)）再喂：
        dx = cos(cyaw)*DX + sin(cyaw)*DY
        dy = -sin(cyaw)*DX + cos(cyaw)*DY
        dyaw = wrap(target_yaw - cur_yaw)
    这样 turn1 后 heading 指向 target、前进 hypot 后正好到 target 点，再补转
    到 target heading——闭环还原 chunk-frame 轨迹。dyaw 归一到 (-π,π]。
    """
    cx, cy, cyaw = float(cur_xyz[0]), float(cur_xyz[1]), float(cur_xyz[2])
    DX = float(target_xyz[0]) - cx
    DY = float(target_xyz[1]) - cy
    c, s = math.cos(cyaw), math.sin(cyaw)
    dx = c * DX + s * DY
    dy = -s * DX + c * DY
    dyaw = (float(target_xyz[2]) - cyaw + math.pi) % (2 * math.pi) - math.pi
    return dx, dy, dyaw


def _turn_sign(deg):
    """正 deg → ('left', abs); 负 deg → ('right', abs)。受 LEFT_POSITIVE 控制。"""
    if not LEFT_POSITIVE:
        deg = -deg
    if deg >= 0:
        return "left", abs(deg)
    return "right", abs(deg)


def decompose(dx, dy, dyaw):
    """一条 ego (dx, dy, dyaw) → motion segments 列表。

    每段是 {"kind":"turn","direction","degrees"} 或 {"kind":"forward","meters"}。
    跳过零长度段（drive_turn 自己也会跳 <MIN_TURN_DEG，但这里先省掉）。
    """
    # Gym world poses are rounded to millimetres. At a repeated/terminal
    # waypoint that quantization leaves a sub-millimetre residual. Passing it
    # to drive_forward invokes its 5cm minimum and makes a stationary target
    # oscillate. Treat translation within 2mm as reached, preserving yaw.
    if math.hypot(dx, dy) <= .002:
        turn1 = 0.0
        fwd = 0.0
    else:
        turn1 = math.atan2(dy, dx)  # 弧度，正=左
        fwd = math.hypot(dx, dy)
    turn2 = float(dyaw) - turn1

    segs = []
    d1 = math.degrees(turn1)
    if abs(d1) > 1e-3:
        direction, degrees = _turn_sign(d1)
        segs.append({"kind": "turn", "direction": direction, "degrees": degrees})
    if fwd > 1e-6:
        segs.append({"kind": "forward", "meters": fwd})
    d2 = math.degrees(turn2)
    if abs(d2) > 1e-3:
        direction, degrees = _turn_sign(d2)
        segs.append({"kind": "turn", "direction": direction, "degrees": degrees})
    return segs


def execute_waypoint(env, dx, dy, dyaw, A, on_step=None, abort=None, render=True):
    """在 env 上执行一条 (dx,dy,dyaw)，复用 agent_tools 的 drive_*。

    返回 log dict：{dx,dy,dyaw, moved_m, collided, blocked, segments, terminated_mid}
    若中途 env.episode_over（撞到 success 终止），terminated_mid=True 并停。

    一个逻辑 waypoint 可超过用户原语的单次距离上限。只把 forward 调用拆成
    不超过 MAX_FWD_M 的物理段，仍复用 drive_forward 的碰撞、stall 和恢复逻辑；
    起点/终点的转向各执行一次，不改变 action expert 的 waypoint 尺度。

    render=False 时逐 sub-step 不渲染（透传给 drive_*→env.step），运动学/碰撞/stall
    照旧（读 sim pose，不读 obs）——闭环 chunk 全程 render=False 执行，末尾单独渲一帧。
    """
    segs = decompose(dx, dy, dyaw)
    log = {"dx": dx, "dy": dy, "dyaw": dyaw,
           "moved_m": 0.0, "collided": False, "blocked": False,
           "segments": segs, "terminated_mid": False, "aborted": False}
    r11 = os.environ.get('ACTION_HEAD_EXECUTION_PROFILE', 'v7_r11_collision') == 'v7_r11_collision'
    for seg in segs:
        if env.episode_over:
            log["terminated_mid"] = True
            break
        if abort is not None and abort():
            log["aborted"] = True
            break
        if seg["kind"] == "turn":
            _obs, tinfo = A.drive_turn(env, seg["direction"], seg["degrees"],
                                        on_step=on_step, render=render)
            if tinfo.get("reason") == "episode_over":
                log["terminated_mid"] = True
                break
            if tinfo.get("reason") == "aborted":
                log["aborted"] = True
                break
        else:  # forward
            remaining = float(seg["meters"])
            while remaining > 1e-9:
                if env.episode_over:
                    log["terminated_mid"] = True
                    break
                if abort is not None and abort():
                    log["aborted"] = True
                    break
                distance = min(remaining, float(getattr(A, "MAX_FWD_M", 1.0)))
                _obs, finfo = A.drive_forward(env, distance, on_step=on_step,
                                              abort=abort, render=render, min_m=0.0)
                log["moved_m"] += float(finfo.get("moved_m", 0.0))
                log["collided"] |= bool(finfo.get("collided"))
                log["blocked"] |= bool(finfo.get("blocked"))
                log["terminated_mid"] |= finfo.get("reason") == "episode_over"
                log["aborted"] |= finfo.get("reason") == "aborted"
                if log["blocked"] or log["terminated_mid"] or log["aborted"] or (r11 and log["collided"]):
                    break
                remaining -= distance
            # v3: a blocked forward may still complete its final turn.
            if log["terminated_mid"] or log["aborted"] or (r11 and (log["blocked"] or log["collided"])):
                break
    return log


# ----------------------------------------------------------------------------
# 单元自检：不依赖 habitat，只验分解几何
# ----------------------------------------------------------------------------
def _selfcheck():
    # 反归一化往返测试 denormalize 的纯数学（翻号开关置 0，隔离坐标约定）。
    os.environ["ABOTN_FLIP_DY"] = "0"
    os.environ["ABOTN_FLIP_DYAW"] = "0"
    fails = []
    # 纯前进
    s = decompose(0.1, 0, 0)
    assert len(s) == 1 and s[0]["kind"] == "forward" and \
        abs(s[0]["meters"] - 0.1) < 1e-9, s
    # 纯转向（dyaw>0 左转）
    s = decompose(0, 0, 0.5)
    assert len(s) == 1 and s[0]["kind"] == "turn" and \
        s[0]["direction"] == "left" and \
        abs(s[0]["degrees"] - math.degrees(0.5)) < 1e-6, s
    # 纯侧移 (0,0.1,0)：turn left 90 → fwd 0.1 → turn right 90，净 heading 0
    s = decompose(0, 0.1, 0)
    assert len(s) == 3, s
    assert s[0] == {"kind": "turn", "direction": "left", "degrees": 90.0}, s[0]
    assert s[1] == {"kind": "forward", "meters": 0.1}, s[1]
    assert s[2] == {"kind": "turn", "direction": "right", "degrees": 90.0}, s[2]
    # 不变量：turn1+turn2 == dyaw
    for dx, dy, dyaw in [(0.1, 0.05, 0.3), (-0.05, 0.1, -0.4), (0.0, 0.0, 0.0),
                          (0.1, -0.1, 0.0), (0.0, 0.1, 0.5)]:
        segs = decompose(dx, dy, dyaw)
        tsum = sum((seg["degrees"] if seg["kind"] == "turn" else 0.0) *
                   (1 if seg.get("direction") == "left" else -1)
                   for seg in segs)
        assert abs(tsum - math.degrees(dyaw)) < 1e-6, (dx, dy, dyaw, tsum)
    # 反归一化往返：真实值→[-1,1]→真实值 误差 < 1e-6
    import random
    for _ in range(1000):
        real = [random.uniform(q01, q99)
                for q01, q99 in [(_CE_WP_STATS["dx"]["q01"], _CE_WP_STATS["dx"]["q99"]),
                                 (_CE_WP_STATS["dy"]["q01"], _CE_WP_STATS["dy"]["q99"]),
                                 (_CE_WP_STATS["dyaw"]["q01"], _CE_WP_STATS["dyaw"]["q99"])]]
        normed = [2 * (r - q01) / (q99 - q01) - 1 for r, q01, q99 in
                  zip(real, [_CE_WP_STATS["dx"]["q01"], _CE_WP_STATS["dy"]["q01"], _CE_WP_STATS["dyaw"]["q01"]],
                      [_CE_WP_STATS["dx"]["q99"], _CE_WP_STATS["dy"]["q99"], _CE_WP_STATS["dyaw"]["q99"]])]
        back = denormalize([normed], normalized=True)[0]
        for a, b in zip(real, back):
            assert abs(a - b) < 1e-6, (real, back)
    # ---- chunk-frame 协议自检（P0）----
    # chunk_frame_trajectory == cumsum
    d = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, math.pi / 2], [1.0, 0.0, -math.pi / 2]],
                 dtype=np.float64)
    traj = chunk_frame_trajectory(d)
    assert np.allclose(traj, np.cumsum(d, axis=0)), traj
    # world_to_chunk_pose：start 即原点 → 不动；绕 yaw0 旋转一致
    cx, cy, cyaw = world_to_chunk_pose([1.0, 1.0], math.pi / 2, [0.0, 0.0], 0.0)
    assert abs(cx - 1.0) < 1e-12 and abs(cy - 1.0) < 1e-12 and abs(cyaw - math.pi/2) < 1e-12
    # 旋转 90° 的 world 点落到 chunk 系
    cx, cy, cyaw = world_to_chunk_pose([1.0, 0.0], math.pi / 2, [0.0, 0.0], math.pi / 2)
    assert abs(cx - 0.0) < 1e-12 and abs(cy + 1.0) < 1e-12, (cx, cy)  # (1,0)→(0,-1)
    # 理想运动学闭环：逐 target 用 target_to_local_command 派生 local 命令，
    # 按 execute_waypoint.decompose 的语义（turn1→fwd→补转）理想执行，
    # 断言每步终点 == chunk-frame target，整条还原 cumsum 轨迹。
    def _ideal_execute(cur, dx, dy, dyaw):
        x, y, yaw = cur
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            turn1 = 0.0
            fwd = 0.0
        else:
            turn1 = math.atan2(dy, dx)
            fwd = math.hypot(dx, dy)
        a = yaw + turn1
        x += math.cos(a) * fwd
        y += math.sin(a) * fwd
        yaw += dyaw
        return (x, y, yaw)
    rng = np.random.default_rng(0)
    for _ in range(500):
        deltas = rng.uniform(-0.1, 0.1, size=(8, 2)).tolist()
        deltas = [row + [rng.uniform(-0.3, 0.3)] for row in deltas]
        traj = chunk_frame_trajectory(deltas)
        cur = (0.0, 0.0, 0.0)
        for k in range(len(traj)):
            dx, dy, dyaw = target_to_local_command(traj[k], cur)
            cur = _ideal_execute(cur, dx, dy, dyaw)
            assert math.hypot(cur[0] - traj[k, 0], cur[1] - traj[k, 1]) < 1e-9, \
                (k, cur, traj[k])
            assert abs(cur[2] - traj[k, 2]) < 1e-9, (k, cur[2], traj[k, 2])
    print("[fm_action selfcheck] OK — decompose 几何 + 反归一化往返 + chunk-frame 闭环还原全过")


if __name__ == "__main__":
    _selfcheck()
