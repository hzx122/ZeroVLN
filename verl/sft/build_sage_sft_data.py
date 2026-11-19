#!/usr/bin/env python3
"""sage domain-matched SFT 数据生成（带历史上下文 N=8，对齐闭环部署）。

与 ``build_abotn_sft_data.py`` 同构——差异只在**数据源**（sage GVLN `SageSet.tasks`
而非 abotn traj_*.json）与**相机参数**（sage z=1.2 / HFOV=105.3，对齐
``sage/env.py`` 部署 camera）。其余逐字复用 abotn builder：

* ``render_frame``（带 sage cam_intrinsics）/ ``arc_resample_future`` /
  ``relative_future`` / ``normalize_waypoints`` / ``build_prompt`` /
  ``_flip_to_base`` 全部 import 自 ``build_abotn_sft_data``，零重造。
* 8 帧历史窗口 + 32 步**单步 ego delta** GT + per-task 帧缓存（磁盘命中复用）+
  MinimalDataset record 格式 + 末尾 GT 饱和自检——与 abotn 逐字一致。
* 坐标约定复用 ``ABOTN_BASE_CONVENTION``（sage yaw 系「from +x→+y」与 abotn 同，
  且 FM base ckpt 同为 0729-baseline，故同款 Y左→Y右 翻号）。

数据源：``sage.datasets.SageSet(split=--sage-split)`` 的 ``tasks(scene, limit=...)``
已把 GVLN traj 解析成 ``Task(scene, waypoints(x,y,z), yaws, instruction, goal_xy,
ref_len)``——``waypoints``/``yaws`` 已含 ``frame_offset``（几何系，与 ply/occupancy
同系），``scene`` 是 ``sage-<id>``（与 render server ``sage_gs_root`` 场景名一致）。

用法（render server 已起 sage 7038 sh3，pong OK）：
  python3 build_sage_sft_data.py --render-url http://127.0.0.1:7038 \
      --sage-split instr --out-dir /data/storage/.../vln_verl/sft_data/sage_instr
  # smoke：--max-tasks 1 --stride 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
import numpy as np

# ---- 复用 abotn builder 的全部核心逻辑（同目录 import）----
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from build_abotn_sft_data import (  # noqa: E402
    render_frame, arc_resample_future, relative_future, normalize_waypoints,
    build_prompt, _flip_to_base, _BASE_CONV, IMAGE_TOKEN, WP_TOKEN, ZYT_STATS,
)

# ---- sage 后端数据源（vln_demo/src 在 sys.path）----
_REPO = os.path.dirname(os.path.dirname(_HERE))
_VLN_SRC = os.path.join(_REPO, "vln_demo", "src")
if _VLN_SRC not in sys.path:
    sys.path.insert(0, _VLN_SRC)
from sage.datasets import SageSet, SPLITS  # noqa: E402

# ---- sage 相机参数：从统一配置 verl/dlc/backends.py 读（单一来源），env 覆盖 ----
# sage 地面 flat z=0，部署 camera_z = ground_z + 1.2 = 1.2；SFT 帧沿 reference path
# 渲染须用同款 camera_z 才与部署分布一致（waypoint 存的 z=0.5 是 agent 位姿非相机）。
_DLC = os.path.join(_REPO, "verl", "dlc")
if _DLC not in sys.path:
    sys.path.insert(0, _DLC)
from backends import fields as _backend_fields  # noqa: E402
_SAGE_CFG = _backend_fields("sage")
CAMERA_HEIGHT = float(os.environ.get("SAGE_CAMERA_HEIGHT", _SAGE_CFG["camera_height"]))
W, H = 640, 480
HFOV_DEG = float(os.environ.get("SAGE_HFOV", _SAGE_CFG["hfov"]))
FX = W / (2.0 * math.tan(math.radians(HFOV_DEG) / 2.0))
SAGE_CAM_INTRINSICS = f"1 PINHOLE {W} {H} {FX:.4f} {FX:.4f} {W/2:.1f} {H/2:.1f}"


def _arc_resample_partial(xs, ys, yaws, i: int, step_m: float, n: int,
                          min_future_m: float):
    """Resample the available suffix and mask endpoint padding.

    The legacy helper requires at least half of a 3.2 m horizon.  P0 routes are
    only 1.2--2.4 m long, so that rule silently removes late fast replans.  We
    retain the fixed model shape while excluding repeated endpoint padding from
    the loss with ``waypoint_mask``.
    """
    seg = np.asarray([
        math.hypot(xs[j + 1] - xs[j], ys[j + 1] - ys[j])
        for j in range(i, len(xs) - 1)
    ], dtype=np.float64)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    remaining = float(cum[-1])
    if remaining + 1e-8 < min_future_m:
        return None
    valid = min(n, max(1, int(math.ceil(remaining / step_m - 1e-8))))
    targets = np.minimum(np.arange(1, n + 1, dtype=np.float64) * step_m,
                         remaining)
    yaw_unwrapped = np.unwrap(np.asarray(yaws[i:], dtype=np.float64))
    fx = np.interp(targets, cum, np.asarray(xs[i:], dtype=np.float64))
    fy = np.interp(targets, cum, np.asarray(ys[i:], dtype=np.float64))
    fyaw = np.interp(targets, cum, yaw_unwrapped)
    mask = np.arange(n) < valid
    return fx, fy, fyaw, mask, remaining


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-url", default="http://127.0.0.1:7038")
    ap.add_argument("--sage-split", choices=list(SPLITS), default="instr",
                    help="train(929)/instr(15)/scene(15)/traj(14)——按 gs_root 已建 ply 取")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--jsonl-name", default="sage_sft.jsonl",
        help="Output path relative to --out-dir. This lets independent GPU "
             "render workers publish disjoint raw parts into one shard root.")
    ap.add_argument("--stride", type=int, default=6, help="滑窗步（原 path 点数）")
    ap.add_argument("--history-len", type=int, default=8,
                    help="历史窗口 N（8=对齐 SocialNav/部署；1=退化单帧兼容旧 SFT）")
    ap.add_argument("--action-chunk", type=int, default=32)
    ap.add_argument("--step-m", type=float, default=0.1)
    ap.add_argument("--horizon-m", type=float, default=3.2)
    ap.add_argument("--allow-partial-horizon", action="store_true",
                    help="Keep short route suffixes and mask repeated endpoint padding.")
    ap.add_argument("--min-future-m", type=float, default=0.4,
                    help="Minimum usable future arc when partial horizons are enabled.")
    ap.add_argument("--max-tasks", type=int, default=0, help="0=全量；>0 限 smoke（每场景前 N task）")
    ap.add_argument("--scene", default="",
                    help="只构建一个场景（接受 840483 或 sage-840483）")
    ap.add_argument("--task-id", default="", help="只构建一个 task")
    ap.add_argument("--instruction-index", type=int, default=-1,
                    help=">=0 时每条物理轨迹只保留该 instruction index")
    ap.add_argument("--eps-per-scene", type=int, default=0,
                    help="每场景限量解析的 episode 数（0=全量；train 大 split 必设 >0 省 RAM）")
    ap.add_argument("--split-manifest", default="",
                    help="可选：按 manifest 的 train/validation 场景确定性抽取短基线子集")
    ap.add_argument("--max-train-scenes", type=int, default=0)
    ap.add_argument("--max-validation-scenes", type=int, default=0)
    ap.add_argument("--scene-selection-seed", type=int, default=20260826)
    ap.add_argument("--scene-shard-index", type=int, default=0)
    ap.add_argument("--scene-shard-count", type=int, default=1)
    ap.add_argument("--max-scenes", type=int, default=0,
                    help="After deterministic sharding, keep at most N scenes (smoke only).")
    ap.add_argument("--fail-on-render-error", action="store_true",
                    help="Abort instead of silently dropping samples when rendering fails.")
    args = ap.parse_args()
    if args.scene_shard_count <= 0 or not 0 <= args.scene_shard_index < args.scene_shard_count:
        raise ValueError("scene shard must satisfy 0 <= index < count")
    jsonl_relative = Path(args.jsonl_name)
    if jsonl_relative.is_absolute() or ".." in jsonl_relative.parts:
        raise ValueError("--jsonl-name must stay inside --out-dir")

    H = args.history_len
    ds = SageSet(split=args.sage_split)
    # sage 的 ply 在 render server 的 scenes_root（sage_gs_root，来自配置）下
    # ``<root>/<scene>/gs/0/<scene>.ply``，不在 InteriorGS（ds.scene_dir 指那里，只有
    # occupancy.json）；用配置的 scenes_root 做已建 ply 预过滤，对齐 run_local_sage.sh。
    gs_root = _SAGE_CFG["scenes_root"]
    scenes = [s for s in ds.list_scenes()
              if os.path.exists(os.path.join(gs_root, s, "gs", "0", s + ".ply"))]
    if args.scene:
        wanted = args.scene if args.scene.startswith("sage-") else "sage-" + args.scene
        scenes = [scene for scene in scenes if scene == wanted]
    if args.split_manifest:
        if args.scene:
            raise ValueError("--scene and --split-manifest cannot be combined")
        manifest = json.loads(open(args.split_manifest, encoding="utf-8").read())
        split_scenes = manifest["splits"]

        def _select(values, limit):
            available = set(scenes)
            ranked = sorted(
                (scene for scene in values if scene in available),
                key=lambda scene: hashlib.sha256(
                    f"{args.scene_selection_seed}:{scene}".encode()).hexdigest())
            return ranked[:limit] if limit else ranked

        train_scenes = _select(
            split_scenes["train_scenes"], args.max_train_scenes)
        validation_scenes = _select(
            split_scenes["validation_scenes"], args.max_validation_scenes)
        scenes = train_scenes + validation_scenes
        print("[sft-data] deterministic scene subset: "
              f"train={len(train_scenes)} validation={len(validation_scenes)} "
              f"seed={args.scene_selection_seed}")
    if not scenes:
        raise SystemExit(
            f"[sft-data] split={args.sage_split} 无已建 ply 场景（gs_root={gs_root}）；"
            f"先 make_scenes_root --split {args.sage_split} --source nurec")
    scenes = sorted(scenes)[args.scene_shard_index::args.scene_shard_count]
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    if not scenes:
        raise SystemExit(
            f"[sft-data] empty scene shard {args.scene_shard_index}/"
            f"{args.scene_shard_count}")
    # 把 --max-tasks 理解成「每场景前 N task」（与 abotn --max-tasks 取前 N 文件同款 smoke 语义）；
    # --eps-per-scene 透传给 SageSet.tasks(limit=) 限量大 split 启动 RAM。
    limit = args.eps_per_scene if args.eps_per_scene > 0 else args.max_tasks
    print(f"[sft-data] split={args.sage_split} scenes={len(scenes)} "
          f"shard={args.scene_shard_index}/{args.scene_shard_count} stride={args.stride} "
          f"history={H} chunk={args.action_chunk} step={args.step_m}m horizon={args.horizon_m}m "
          f"max_tasks={args.max_tasks} eps_per_scene={args.eps_per_scene} cam_z={CAMERA_HEIGHT} "
          f"hfov={HFOV_DEG}")

    os.makedirs(os.path.join(args.out_dir, "frames"), exist_ok=True)
    preloaded: set = set()
    jsonl_path = os.path.join(args.out_dir, str(jsonl_relative))
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    jsonl_temporary = jsonl_path + ".tmp"
    n_samples = n_skip = n_renders = 0
    all_gt_norm: list = []
    all_gt_real: list = []
    t0 = time.time()
    with open(jsonl_temporary, "w") as jf:
        for si, scene in enumerate(scenes):
            tasks = ds.tasks(scene, limit=limit)
            if args.task_id:
                tasks = [task for task in tasks if task.task_id == args.task_id]
            if args.instruction_index >= 0:
                suffix = f"_{args.instruction_index}"
                tasks = [task for task in tasks if task.task_id.endswith(suffix)]
            # --max-tasks>0 时再裁到每场景前 N（--eps-per_scene=0 时 limit=max_tasks 已裁；
            # eps_per_scene>0 时 limit=eps_per_scene，max_tasks 再裁一层）
            if args.max_tasks and not args.eps_per_scene:
                tasks = tasks[:args.max_tasks]
            for ti, t in enumerate(tasks):
                wps = t.waypoints
                yaws = t.yaws
                instr = (t.instruction or "").strip()
                if len(wps) < 10 or not yaws or not instr:
                    continue
                xs = [p[0] for p in wps]
                ys = [p[1] for p in wps]
                yawsl = [float(y) for y in yaws]
                n_pts = len(wps)
                scene_id = t.scene
                task_id = t.task_id
                path_arc = np.concatenate((
                    np.zeros(1, dtype=np.float64),
                    np.cumsum([
                        math.hypot(xs[j + 1] - xs[j], ys[j + 1] - ys[j])
                        for j in range(n_pts - 1)
                    ]),
                ))

                frame_cache: dict[int, str] = {}

                def _frame_for(pt_idx: int) -> str | None:
                    nonlocal n_renders
                    if pt_idx in frame_cache:
                        return frame_cache[pt_idx]
                    if pt_idx < 0 or pt_idx >= n_pts:
                        return None
                    rel = f"frames/{scene_id}_{task_id}_pt{pt_idx}.jpg"
                    disk = os.path.join(args.out_dir, rel)
                    if os.path.exists(disk) and os.path.getsize(disk) > 0:
                        frame_cache[pt_idx] = rel
                        return rel
                    try:
                        jpg = render_frame(args.render_url, scene_id, xs[pt_idx],
                                           ys[pt_idx], CAMERA_HEIGHT, yawsl[pt_idx],
                                           preloaded, cam_intrinsics=SAGE_CAM_INTRINSICS)
                    except Exception as e:
                        print(f"  [skip render] {scene_id}:{task_id} pt={pt_idx}: {e}")
                        if args.fail_on_render_error:
                            raise RuntimeError(
                                f"render failure is fatal for this dataset build: "
                                f"{scene_id}:{task_id} pt={pt_idx}") from e
                        return None
                    n_renders += 1
                    with open(disk, "wb") as pf:
                        pf.write(jpg)
                    frame_cache[pt_idx] = rel
                    return rel

                for i in range(0, n_pts - 1, args.stride):
                    if args.allow_partial_horizon:
                        fut = _arc_resample_partial(
                            xs, ys, yawsl, i, args.step_m, args.action_chunk,
                            args.min_future_m)
                    else:
                        legacy = arc_resample_future(
                            xs, ys, yawsl, i, args.step_m,
                            args.horizon_m, args.action_chunk)
                        fut = None if legacy is None else (
                            *legacy,
                            np.ones(args.action_chunk, dtype=np.bool_),
                            float(path_arc[-1] - path_arc[i]),
                        )
                    if fut is None:
                        n_skip += 1
                        continue
                    fx, fy, fyaw, waypoint_mask, remaining_arc_m = fut
                    rel = relative_future(fx, fy, fyaw, xs[i], ys[i], yawsl[i])
                    if _BASE_CONV:
                        rel = _flip_to_base(rel, has_yaw=True)
                    gt_norm = normalize_waypoints(rel)
                    gt = gt_norm.tolist()
                    all_gt_norm.append(gt_norm)
                    all_gt_real.append(rel)

                    hist_idx = [max(0, i - H + 1 + j) for j in range(H)]
                    hist_paths = [_frame_for(k) for k in hist_idx]
                    if any(p is None for p in hist_paths):
                        n_skip += 1
                        continue
                    # ego 相对位姿复用部署 ego_relative_xy（与 abotn builder 同款 import 路径）
                    from history_util import ego_relative_xy  # noqa: E402
                    from fm_policy import _norm_input_wp, _ZYT  # noqa: E402
                    hist_xy = np.array([[xs[k], ys[k]] for k in hist_idx], np.float32)
                    ego = ego_relative_xy(hist_xy, xs[i], ys[i], yawsl[i])
                    wp_norm = _norm_input_wp(ego, _ZYT)
                    rec = {
                        "train_branch": "fm",
                        "traj_type": "anonymous",
                        "images": hist_paths,
                        "conversations": [
                            {"from": "human", "value": build_prompt(instr, H)},
                            {"from": "gpt", "value": "<flow_matching_policy>"},
                        ],
                        "meta_data": {
                            "source_scene": scene_id,
                            "source_task_id": task_id,
                            "source_point_index": i,
                            "source_arc_m": float(path_arc[i]),
                            "remaining_arc_m": remaining_arc_m,
                            "source_pose_xyyaw": [float(xs[i]), float(ys[i]),
                                                  float(yawsl[i])],
                            "input_waypoints": wp_norm.tolist(),
                            "gt_waypoints": gt,
                            "waypoint_mask": waypoint_mask.tolist(),
                            "valid_action_steps": int(waypoint_mask.sum()),
                            "arrive": 0.0,
                        },
                    }
                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_samples += 1
            if (si + 1) % 5 == 0:
                print(f"  [{si+1}/{len(scenes)}] scenes samples={n_samples} skip={n_skip} "
                      f"renders={n_renders} elapsed={time.time()-t0:.0f}s")
    os.replace(jsonl_temporary, jsonl_path)
    print(f"[sft-data] done: samples={n_samples} skip={n_skip} renders={n_renders} "
          f"elapsed={time.time()-t0:.0f}s")
    print(f"  jsonl: {jsonl_path}")
    print(f"  frames: {os.path.join(args.out_dir, 'frames')}/")
    if all_gt_norm:
        narr = np.concatenate(all_gt_norm, axis=0)
        abs_n = np.abs(narr)
        sat2 = float((abs_n >= 2.0).mean())
        over1 = float((abs_n > 1.0).mean())
        print(f"[sft-data] GT selfcheck: |norm|>=2 satur={sat2*100:.1f}%  "
              f"|norm|>1={over1*100:.1f}%  (expect satur<5%)")
        rarr = np.concatenate(all_gt_real, axis=0)
        abs_r = np.abs(rarr)
        print(f"  real per-step |delta| mean: dx={abs_r[:,0].mean():.3f}m "
              f"dy={abs_r[:,1].mean():.3f}m dyaw={abs_r[:,2].mean():.3f}rad | "
              f"max: dx={abs_r[:,0].max():.3f}m dy={abs_r[:,1].max():.3f}m "
              f"dyaw={abs_r[:,2].max():.3f}rad")
        if sat2 >= 0.05:
            print(f"  [WARN] satur {sat2*100:.1f}%>=5% — GT 仍饱和，检查 relative_future diff 逻辑")


if __name__ == "__main__":
    main()
