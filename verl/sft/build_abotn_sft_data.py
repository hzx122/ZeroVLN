#!/usr/bin/env python3
"""abotn domain-matched SFT 数据生成（带历史上下文 N=8，对齐闭环部署）。

用 abotn traj_*.json 的标注 reference path 作模仿 target，沿 path 滑窗取 **8 帧历史
窗口**（points [i-7..i]，i<7 用 point 0 补满，对齐 HistoryBuffer.reset 用首帧重复 N 份
的部署行为）→ 渲染 8 帧 + 算 8 个 input_waypoint（历史位姿 ego 系相对 (dx,dy)，复用
history_util.ego_relative_xy + fm_policy._norm_input_wp/_ZYT，与部署逐字同款）+ 32 步
gt future（弧长 resample 3.2m，每步 0.1m 匹配 _ZYT_WP_STATS）→ 归一化输出 MinimalDataset
jsonl。

**与旧版（单帧 + input_waypoints=[[0,0]]）的差异**：闭环 rollout（run_episode_ode_multiturn
/rollout_episode_sde_multiturn）已喂 8 帧历史 + 8 input_waypoint（history_util.HistoryBuffer
N=8 对齐 SocialNav）。旧 SFT 单帧零历史 → 部署分布外（covariate shift），闭环 success=0。
本版产 8 帧历史记录，SFT 分布 = 部署分布。

**prompt 对齐（关 auto-fill）**：human turn 显式写部署同款 prompt 串
`<image>×N <input_waypoint>×N anonymous Instruction: {instr}\nPredict the goal point and trajectory.`
（与 fm_policy._build_prompt_ids 逐字一致）。data.py:458 auto-fill 只在不含 `<image>` 且
不含 `<input_waypoint>` 时触发 → 本版含 `<image>` → auto-fill 跳过 → 用本串 →
replace_image_tokens 把 `<image>`→`<|vision_start|><|image_pad|><|vision_end|>`（同 fm_policy）。
旧版 bare instr → auto-fill 塞中文 wrapper + `<input_posK>`，与部署 `<input_waypoint>` 不一致
（success=0 的独立嫌疑，本版一并修掉）。

复用 SocialNav train.py + data.py 的 record 格式（零改 loss/forward/ckpt；data.py 原生吃
images:[8] + input_waypoints:[[x,y]×8]，N 从 len(images) 推断）。render 直接 urllib POST
/render_gs_xyz_euler（不 import gs_client，避免依赖链）。per-task 帧缓存：每个 path point
只渲一次，跨滑窗样本复用（8 帧窗口重叠 7 帧，缓存后渲染量 ≈ len(traj)/task 而非 8×samples）。

用法（render server 已起 abotn 7037 sh3，pong OK）：
  python3 build_abotn_sft_data.py --render-url http://127.0.0.1:7037 \
      --abotn-set poi --out-dir /data/storage/.../vln_verl/sft_data/abotn_poi
  # smoke：--max-tasks 1 --stride 8
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
import urllib.request
import urllib.error
import numpy as np

# 复用部署同款 ego 对齐 + 归一化（保证 SFT 分布 == 部署分布，零重造）。
# verl/workers 在 ../workers（本文件在 verl/sft/）。
_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKERS = os.path.dirname(_HERE) + os.sep + "workers"
if _WORKERS not in sys.path:
    sys.path.insert(0, _WORKERS)
from history_util import ego_relative_xy  # noqa: E402
from fm_policy import _norm_input_wp, _ZYT  # noqa: E402

# 坐标约定开关：1（默认）= 对齐 0729 base 的 Y右/yaw顺时针约定。base 训练时
# raw_waypoints._flip_y_dy 把旧 Y左 delta 翻成 Y右（dy→-dy, dyaw→-dyaw+wrap），
# 故 SFT 标签须同款翻号，base 先验才可用（旧版未翻→dy/dyaw 通道与 base 反号，
# 直行为主低 loss 掩盖转弯符号错→闭环撞墙，见 docs/实验记录.md#legacy-docs-20260921 2026-08-13）。
# input_waypoint 的 y 翻号由 ego_relative_xy 内部按同开关处理（build 与部署共用）。
_BASE_CONV = os.environ.get("ABOTN_BASE_CONVENTION", "1") == "1"


def _flip_to_base(arr, has_yaw: bool):
    """Y左 delta → Y右 delta（dy→-dy, has_yaw 时 dyaw→-dyaw + wrap[-π,π]）。
    对齐 raw_waypoints._flip_y_dy。arr: (N,2|3) ndarray，返回同形 float32。"""
    out = np.array(arr, dtype=np.float32).copy()
    out[:, 1] = -out[:, 1]
    if has_yaw and out.shape[1] >= 3:
        out[:, 2] = -out[:, 2]
        out[:, 2] = (out[:, 2] + np.pi) % (2 * np.pi) - np.pi  # wrap [-π,π)
    return out

# 部署同款 token（fm_policy:53-58），prompt 串逐字对齐 _build_prompt_ids。
IMAGE_TOKEN = "<image>"
WP_TOKEN = "<input_waypoint>"

# ---- prompt 变体（对齐 0729 base 训练分布；可选，env PROMPT_VARIANT 切换）----
# 0729 base 训练数据有 3 种 prompt（见 docs/rl使用手册 §2.2 审计）：
#   A(VLN CE AR):  <image>×8 Inter instruction: "...". You are given 8 historical observations. Predict the goal point and trajectory.  | gpt: goal point: [x,y]; <flow_matching_policy>
#   B(FM stop):    <image>×8 <input_waypoint>×8 Inter instruction: "...". You are given 8 historical observations. Predict the trajectory.       | gpt: <flow_matching_policy>
#   C(walk_around):<image>×8 <input_waypoint>×8 anonymous Instruction: walk around\nPredict the goal point and keep moving.                            | gpt: goal point: [x,y]; <flow_matching_policy>
# 本仓历史 prompt = `anonymous Instruction: {instr}\nPredict the goal point and trajectory.` + gpt 裸 <flow_matching_policy>
# —— 是个 0729 任一变体里都不原样出现的混合体（anonymous 前缀来自 C，结尾来自 A）。
# PROMPT_VARIANT:
#   legacy(默认) = 上述历史混合体，行为不变（向后兼容）。
#   base_c       = 结尾改 `Predict the goal point and keep moving.`（变体 C 结尾），使
#                  (前缀 anonymous Instruction + 结尾) 组合精确落到变体 C。gpt 仍裸
#                  <flow_matching_policy>（=变体 B 的 gpt，base 分布内）。
# ⚠ 残留 gap：变体 A/C 的 gpt 带 `goal point: [x,y]; ` 前缀，但那个 [x,y] 是 base 的
#   [0,1000) 画布坐标，abotn/sage goal 是世界米、画布约定未验 + abotn SFT 无 occupancy
#   bounds 算不出 → 暂不塞（塞错刻度比不塞糟），留 follow-up。详见 docs/rl使用手册 §2.2。
_PROMPT_VARIANT = os.environ.get("PROMPT_VARIANT", "legacy").lower()
_PROMPT_ENDINGS = {
    "legacy": "\nPredict the goal point and trajectory.",    # 变体 A 结尾（历史混合体，非 0729 原样）
    "base_c": "\nPredict the goal point and keep moving.",   # 变体 C 结尾（0729 walk_around 原样）
}

# ---- _ZYT_WP_STATS（= _CE_WP_STATS，和 RL 部署 fm_policy._ZYT 同一套）----
ZYT_STATS = {
    "dx": {"q01": -0.10273411870002747, "q99": 0.1031249612569809},
    "dy": {"q01": -0.10309220850467682, "q99": 0.10309189558029175},
    "dyaw": {"q01": -0.5668500065803528, "q99": 0.5747644901275635},
}

# abotn set → (zytoss 子路径, scene 前缀)
SETS = {
    "poi": ("ABotN-POIBench/annotations", "poi-"),
    "in":  ("ABotN-PointBench/Indoor/annotations", "in-"),
    "out": ("ABotN-PointBench/Outdoor/annotations", "out-"),
}

ZYTOSS = os.environ.get("ABOTN_ROOT", "/workspace/agentic_sim/zytoss")
CAMERA_HEIGHT = 0.65      # abotn camera z（env.CAMERA_HEIGHT）
W, H, HFOV_DEG = 640, 480, 120.0
FX = W / (2.0 * math.tan(math.radians(HFOV_DEG) / 2.0))
CAM_INTRINSICS = f"1 PINHOLE {W} {H} {FX:.4f} {FX:.4f} {W/2:.1f} {H/2:.1f}"


def normalize_waypoints(wp: np.ndarray) -> np.ndarray:
    """data.py _normalize_waypoints 同公式：quantile→[-1,1] clip[-2,2]，每 dim 独立。"""
    q01 = np.array([ZYT_STATS["dx"]["q01"], ZYT_STATS["dy"]["q01"], ZYT_STATS["dyaw"]["q01"]], np.float32)
    q99 = np.array([ZYT_STATS["dx"]["q99"], ZYT_STATS["dy"]["q99"], ZYT_STATS["dyaw"]["q99"]], np.float32)
    norm = (wp - q01) / (q99 - q01 + 1e-8) * 2.0 - 1.0
    return np.clip(norm, -2.0, 2.0)


def render_frame(render_url: str, scene_id: str, x: float, y: float, z: float, yaw: float,
                 preloaded: set, cam_intrinsics: str = None) -> bytes:
    """POST /render_gs_xyz_euler → JPEG bytes。首次用 scene 先 /preload。

    ``cam_intrinsics`` 默认 None → 用本模块 abotn 的 CAM_INTRINSICS（120° HFOV）。
    其他后端（sage 105.3°）传自己的内参串复用同一路径，行为对 abotn 不变。"""
    if cam_intrinsics is None:
        cam_intrinsics = CAM_INTRINSICS
    if scene_id not in preloaded:
        try:
            req = urllib.request.Request(render_url + "/preload",
                                         data=json.dumps({"scene_id": scene_id}).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=60).read()
        except Exception:
            pass  # render server 可能已 load，忽略
        preloaded.add(scene_id)
    payload = json.dumps({"scene_id": scene_id, "x": x, "y": y, "z": z,
                          "roll": 0.0, "pitch": 0.0, "yaw": yaw,
                          "degree_flag": False, "cam_intrinsics": cam_intrinsics}).encode()
    req = urllib.request.Request(render_url + "/render_gs_xyz_euler", data=payload,
                                 headers={"Content-Type": "application/json"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(1)
    raise RuntimeError(f"render {scene_id} ({x:.2f},{y:.2f},{yaw:.2f}) failed: {last}")


def arc_resample_future(xs, ys, yaws, i: int, step_m=0.1, horizon_m=3.2, n=32):
    """从 path 第 i 点起，按弧长取未来 horizon_m 内 n 步的 (x,y,yaw)。
    path 点间距 ~0.15m，每步 0.1m → 插值。yaw unwrap 后线性插值。"""
    x0, y0 = xs[i], ys[i]
    # 累积弧长（从 i 起）
    seg = np.array([math.hypot(xs[j+1]-xs[j], ys[j+1]-ys[j]) for j in range(i, len(xs)-1)])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < horizon_m * 0.5:  # 剩余路径太短（不到 1.6m），跳过
        return None
    # 目标弧长点（step, 2*step, ..., n*step）——**不含 arc=0 起点**，对齐 base
    # _interpolate_and_resample_trajectory 的"origin + N 个未来点"结构：filtered 用
    # start_point(0,0) + traj[1:]（排除当前 pose），resample 到 N+1 含 origin，再
    # _xy_to_delta_xyt diff 出 N 个真实 delta（poses[0] 是真实首步，无前导零）。
    # 若含 arc=0，cum[0]=origin，relative_future 的 prepend+diff 后 delta[0]=0
    # （前导零 no-op），与 base 结构错配。故 targets 从 step 起，clip 到 cum[-1]。
    targets = np.minimum(np.arange(1, n + 1) * step_m, cum[-1])
    xs_i, ys_i = np.array(xs[i:]), np.array(ys[i:])
    yaws_i = np.array(yaws[i:])
    # unwrap yaw
    yaws_unwrapped = np.unwrap(yaws_i)
    fx = np.interp(targets, cum, xs_i)
    fy = np.interp(targets, cum, ys_i)
    fyaw = np.interp(targets, cum, yaws_unwrapped)
    return fx, fy, fyaw


def relative_future(fx, fy, fyaw, x0, y0, yaw0):
    """future 点的**单步 ego delta** (dx,dy,dyaw)，对齐 base/canon 的"单步增量"GT。

    先算未来点相对 current 的**累积** ego 轨迹（2D 旋转到 robot 系，x 前/y 左；
    yaw from +x→+y（abotn），R(-yaw0)=[[cos,sin],[-sin,cos]]），再 prepend 原点
    (0,0,0) 后 np.diff 成**单步增量**（相邻点之差，每步 ~0.1m 落进 _ZYT q99≈0.103
    量程不饱和）。

    对齐 SocialNav vln_ce_dataset._xy_to_delta_xyt（resample 到 N+1 含 origin →
    diff → N 个单步 delta）+ convert_deploy_to_jsonl._normalize_gt_wp_from_dump
    （output_waypoints 累计绝对轨迹 → prepend origin → np.diff → 单步增量 →
    归一化）。执行侧 fm_action.execute_waypoint 把每个 waypoint 当"从当前 pose
    出发的独立 ego 增量"逐点执行（turn→fwd→turn ×32），故 GT 必须是单步 delta，
    不能是累积位置（旧版累积→40.7% 撞 clip 饱和→闭环原地蹭/撞墙 success=0）。

    dyaw 同理取相邻差：diff 后 = fyaw[k]-fyaw[k-1] = 单步 heading 变化（fyaw 已
    unwrap，无 wrap 问题）。
    """
    c, s = math.cos(yaw0), math.sin(yaw0)
    dx = c * (fx - x0) + s * (fy - y0)
    dy = -s * (fx - x0) + c * (fy - y0)
    dyaw = fyaw - yaw0  # 累积 heading 变化（已 unwrap）
    cum = np.stack([dx, dy, dyaw], axis=1).astype(np.float32)  # (n,3) 累积 ego 轨迹
    # prepend 原点 (0,0,0) → np.diff → 单步 delta (n,3)，与执行侧"逐点独立增量"自洽
    pts = np.vstack([np.zeros((1, 3), dtype=np.float32), cum])  # (n+1,3)
    return np.diff(pts, axis=0).astype(np.float32)             # (n,3) 单步 delta


def build_prompt(instruction: str, n: int) -> str:
    """部署同款 prompt（fm_policy._build_prompt_ids 的 content 段，pre-replace）。
    含 `<image>` → data.py auto-fill 跳过 → replace_image_tokens 转 vision token（同部署）。
    结尾随 PROMPT_VARIANT 切换（legacy/base_c），见模块 _PROMPT_ENDINGS 注释。"""
    ending = _PROMPT_ENDINGS.get(_PROMPT_VARIANT, _PROMPT_ENDINGS["legacy"])
    return (IMAGE_TOKEN * n + " " + WP_TOKEN * n +
            f" anonymous Instruction: {instruction}{ending}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-url", default="http://127.0.0.1:7037")
    ap.add_argument("--abotn-set", choices=list(SETS), default="poi")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stride", type=int, default=6, help="滑窗步（原 path 点数）")
    ap.add_argument("--history-len", type=int, default=8,
                    help="历史窗口 N（8=对齐 SocialNav/部署；1=退化单帧兼容旧 SFT）")
    ap.add_argument("--action-chunk", type=int, default=32)
    ap.add_argument("--step-m", type=float, default=0.1)
    ap.add_argument("--horizon-m", type=float, default=3.2)
    ap.add_argument("--max-tasks", type=int, default=0, help="0=全量；>0 限 smoke")
    args = ap.parse_args()

    H = args.history_len
    sub, prefix = SETS[args.abotn_set]
    ann_root = os.path.join(ZYTOSS, sub)
    traj_files = sorted(glob.glob(os.path.join(ann_root, "*", "traj_*.json")))
    if args.max_tasks:
        traj_files = traj_files[:args.max_tasks]
    print(f"[sft-data] set={args.abotn_set} tasks={len(traj_files)} stride={args.stride} "
          f"history={H} chunk={args.action_chunk} step={args.step_m}m horizon={args.horizon_m}m")

    os.makedirs(os.path.join(args.out_dir, "frames"), exist_ok=True)
    preloaded: set = set()
    jsonl_path = os.path.join(args.out_dir, "abotn_sft.jsonl")
    n_samples = n_skip = n_renders = 0
    all_gt_norm: list = []  # 归一化 GT (32,3)，末尾出饱和自检
    all_gt_real: list = []  # 真实 delta (32,3) 米/弧度，末尾出量程自检
    t0 = time.time()
    with open(jsonl_path, "w") as jf:
        for ti, tf in enumerate(traj_files):
            scene_dir = os.path.basename(os.path.dirname(tf))
            scene_id = prefix + scene_dir
            task_id = os.path.basename(tf)[:-5].replace("traj_", "")
            with open(tf, encoding="utf-8") as f:
                d = json.load(f)
            traj = d.get("trajectory") or []
            instr = (d.get("instruction") or "").strip()
            if len(traj) < 10 or not instr:
                continue
            xs = [float(p["x"]) for p in traj]
            ys = [float(p["y"]) for p in traj]
            yaws = [float(p.get("yaw", 0.0)) for p in traj]
            n_pts = len(traj)

            # per-task 帧缓存：path point idx -> frame 相对路径。8 帧窗口重叠 7 帧，
            # 缓存后每 task 渲染量 ≈ n_pts（而非 8×samples）。
            frame_cache: dict[int, str] = {}

            def _frame_for(pt_idx: int) -> str | None:
                nonlocal n_renders
                if pt_idx in frame_cache:
                    return frame_cache[pt_idx]
                if pt_idx < 0 or pt_idx >= n_pts:
                    return None
                rel = f"frames/{scene_id}_{task_id}_pt{pt_idx}.jpg"
                # 磁盘命中复用（翻号重建等场景：帧不变，jsonl 重写即可，不重渲染）
                disk = os.path.join(args.out_dir, rel)
                if os.path.exists(disk) and os.path.getsize(disk) > 0:
                    frame_cache[pt_idx] = rel
                    return rel
                z = CAMERA_HEIGHT
                try:
                    jpg = render_frame(args.render_url, scene_id, xs[pt_idx], ys[pt_idx],
                                       z, yaws[pt_idx], preloaded)
                except Exception as e:
                    print(f"  [skip render] {scene_id}:{task_id} pt={pt_idx}: {e}")
                    return None
                n_renders += 1
                with open(disk, "wb") as pf:
                    pf.write(jpg)
                frame_cache[pt_idx] = rel
                return rel

            # 滑窗：每 stride 个 path 点一个 sample，留够 future horizon
            for i in range(0, n_pts - 1, args.stride):
                fut = arc_resample_future(xs, ys, yaws, i, args.step_m,
                                          args.horizon_m, args.action_chunk)
                if fut is None:
                    n_skip += 1
                    continue
                fx, fy, fyaw = fut
                rel = relative_future(fx, fy, fyaw, xs[i], ys[i], yaws[i])  # (32,3) 单步 delta
                if _BASE_CONV:
                    rel = _flip_to_base(rel, has_yaw=True)  # Y左→Y右 对齐 0729 base
                gt_norm = normalize_waypoints(rel)                          # (32,3) 归一化
                gt = gt_norm.tolist()
                all_gt_norm.append(gt_norm)
                all_gt_real.append(rel)

                # 历史窗口 [i-H+1 .. i]，i<H 时左侧 clamp 到 0（对齐 HistoryBuffer.reset
                # 用首帧重复 N 份；部署 chunk t<H 时 buffer=[pose0×(H-t), pose1..poset]）。
                hist_idx = [max(0, i - H + 1 + j) for j in range(H)]
                # 渲染 8 帧（缓存命中只读已有文件）
                hist_paths = [_frame_for(k) for k in hist_idx]
                if any(p is None for p in hist_paths):
                    n_skip += 1
                    continue
                # 8 个历史位姿相对当前 i 的 ego (dx,dy)，复用部署 ego_relative_xy
                hist_xy = np.array([[xs[k], ys[k]] for k in hist_idx], np.float32)  # (H,2)
                ego = ego_relative_xy(hist_xy, xs[i], ys[i], yaws[i])             # (H,2) 米
                wp_norm = _norm_input_wp(ego, _ZYT)                                # (H,2) 除 range
                rec = {
                    "train_branch": "fm",
                    "traj_type": "anonymous",
                    "images": hist_paths,                       # H 帧（时序旧→新，末帧=当前）
                    "conversations": [
                        {"from": "human", "value": build_prompt(instr, H)},
                        {"from": "gpt", "value": "<flow_matching_policy>"},
                    ],
                    "meta_data": {
                        "input_waypoints": wp_norm.tolist(),     # (H,2) 归一化，与部署同款
                        "gt_waypoints": gt,
                        "waypoint_mask": [True] * args.action_chunk,
                        "arrive": 0.0,
                    },
                }
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_samples += 1
            if (ti + 1) % 10 == 0:
                print(f"  [{ti+1}/{len(traj_files)}] samples={n_samples} skip={n_skip} "
                      f"renders={n_renders} elapsed={time.time()-t0:.0f}s")
    print(f"[sft-data] done: samples={n_samples} skip={n_skip} renders={n_renders} "
          f"elapsed={time.time()-t0:.0f}s")
    print(f"  jsonl: {jsonl_path}")
    print(f"  frames: {os.path.join(args.out_dir, 'frames')}/")
    # 开环 GT 饱和自检（验收门槛：|norm|>=2 饱和应 <5%，对照旧累积版 40.7%）。
    # 饱和=被 clip[-2,2] 抹平的常数监督目标，比例高则 GT 退化、闭环无导航意义。
    if all_gt_norm:
        narr = np.concatenate(all_gt_norm, axis=0)  # (N*32, 3) 归一化后
        abs_n = np.abs(narr)
        sat2 = float((abs_n >= 2.0).mean())
        over1 = float((abs_n > 1.0).mean())
        print(f"[sft-data] GT selfcheck: |norm|>=2 satur={sat2*100:.1f}%  "
              f"|norm|>1={over1*100:.1f}%  (expect satur<5%; old cumulative=40.7%)")
        # 真实单步 delta 量程（应每步 ~0.1m，落进 q99≈0.103 量程；旧累积版涨到 3m 致饱和）
        rarr = np.concatenate(all_gt_real, axis=0)  # (N*32, 3) 真实米/弧度
        abs_r = np.abs(rarr)
        print(f"  real per-step |delta| mean: dx={abs_r[:,0].mean():.3f}m "
              f"dy={abs_r[:,1].mean():.3f}m dyaw={abs_r[:,2].mean():.3f}rad | "
              f"max: dx={abs_r[:,0].max():.3f}m dy={abs_r[:,1].max():.3f}m "
              f"dyaw={abs_r[:,2].max():.3f}rad (q99 range dx/dy~0.103m, dyaw~0.575rad)")
        if sat2 >= 0.05:
            print(f"  [WARN] satur {sat2*100:.1f}%>=5% — GT 仍饱和，检查 relative_future diff 逻辑")


if __name__ == "__main__":
    main()
