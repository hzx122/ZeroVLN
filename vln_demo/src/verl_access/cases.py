# -*- coding: utf-8 -*-
"""cases — 5 个用例的编排（verl 端访问层闭环验证）。

每个用例：构造 policy + bench，跑一个 episode（或一组合成动作），dump
jsonl + 末帧 jpg 到 outputs/verl_access/<date>/，最后 analysis 汇总。

用例：
  1 VLM ReAct 闭环 (2B)
  2 VLM ReAct 闭环对照 (35B)
  3 FM 映射几何（合成 chunk，无模型）— 只需 bench
  4 FM 真模型单发 + denorm/range（0729 hf_16k）
  5 FM 真模型闭环（0729 hf_16k）
"""
import os
import sys
import json
import math
import time
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
# fm_policy 已搬到 verl_nav/workers（解耦：verl 侧 runtime 在 date_mfm/verl_nav，
# vln_demo 只留 bench 服务）。dev 验证脚本跨指 verl_nav，加 path 后 flat import。
_VERL_NAV = "/workspace/date_mfm/projects/MegaTE/vla/verl_nav"
for _p in (os.path.join(_VERL_NAV, "workers"), os.path.join(_VERL_NAV, "bench")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from verl_access.bench_client import BenchClient  # noqa: E402
from verl_access.vlm_policy import VLMAgent  # noqa: E402
import prompts as P  # noqa: E402

import vln_gym.fm_action as FMA  # bench 侧 decompose，用例3 验它

OUT_ROOT = os.path.join(os.path.dirname(_SRC), "outputs", "verl_access")
BENCH_URL = os.environ.get("VLN_BENCH_URL", "http://127.0.0.1:8907")


def _out_dir(date=None):
    d = os.path.join(OUT_ROOT, date or time.strftime("%Y%m%d"))
    os.makedirs(d, exist_ok=True)
    return d


def _save(out_dir, name, traj, bench=None, last_frame=None):
    fn = os.path.join(out_dir, f"{name}.jsonl")
    with open(fn, "w") as f:
        json.dump(traj, f, ensure_ascii=False, indent=2)
    print(f"[case] saved {fn}")
    if last_frame is not None and bench is not None:
        try:
            arr = bench.render("rgb")
            from PIL import Image
            Image.fromarray(arr).save(os.path.join(out_dir, f"{name}_final.jpg"))
        except Exception as e:
            print(f"[case] render save failed: {e}")
    return fn


# ----------------------------------------------------------------------------
# 用例 3：FM 合成 chunk → bench fm_trajectory → 验 decompose 不变量
# ----------------------------------------------------------------------------
def case3_synthetic_fm(scene="2azQ1b91cZZ", episode_id="11"):
    """4 种手造 chunk 喂 bench，验 bench 侧 fm_action 映射几何正确（隔离模型）。"""
    print("\n===== 用例 3: FM 合成 chunk 映射几何 =====")
    bench = BenchClient(BENCH_URL)
    out_dir = _out_dir()
    # 每个 chunk：reset 后执行一次，记 gps/yaw 前后增量，对照 decompose 不变量。
    # normalized=False 直接给真实米/弧度，绕开 bench 硬编码 CE stats。
    chunks = {
        "straight_0p1x32": [[0.1, 0.0, 0.0] for _ in range(32)],   # 直行 3.2m
        "turn_left_0p05x32": [[0.0, 0.0, 0.05] for _ in range(32)],  # 左转 ~91°
        "side_left_0p05x32": [[0.0, 0.05, 0.0] for _ in range(32)],  # 侧移 1.6m
        "mixed": ([[0.1, 0.0, 0.0]] * 8 + [[0.0, 0.0, 0.1]] * 4  # 前进0.8 +左转4*5.7°
                  + [[0.1, 0.0, 0.0]] * 8 + [[0.0, 0.0, -0.1]] * 4
                  + [[0.1, 0.0, 0.0]] * 8),
    }
    results = []
    for name, wps in chunks.items():
        obs, info = bench.reset(scene=scene, episode_id=episode_id)
        gps0, yaw0 = info.get("gps"), info.get("yaw_deg")
        dist0 = info.get("distance_to_goal")
        obs, r, term, trunc, info = bench.step_fm(wps, normalized=False)
        gps1, yaw1 = info.get("gps"), info.get("yaw_deg")
        wl = info.get("waypoint_log") or []
        # decompose 不变量核验：对每个 wp，turn1+turn2 == dyaw
        inv_ok = []
        for w in wps:
            segs = FMA.decompose(w[0], w[1], w[2])
            tsum = sum((s["degrees"] if s["kind"] == "turn" else 0.0) *
                       (1 if s.get("direction") == "left" else -1)
                       for s in segs)
            inv_ok.append(abs(tsum - math.degrees(w[2])) < 1e-6)
        # 实际位移/转角增量
        import math as _m
        if gps0 and gps1:
            dx = gps1[0] - gps0[0]
            dy = gps1[1] - gps0[1]
            moved = _m.hypot(dx, dy)
        else:
            dx = dy = moved = None
        dyaw = ((yaw1 - yaw0 + 180) % 360) - 180 if yaw0 is not None else None
        rec = {"chunk": name, "n_wp": len(wps), "waypoints_executed": len(wl),
               "gps0": gps0, "gps1": gps1, "yaw0": yaw0, "yaw1": yaw1,
               "moved_m": moved, "dyaw_deg": dyaw,
               "collided": info.get("collided_this_step"),
               "blocked": info.get("blocked_this_step"),
               "distance_to_goal_0": dist0,
               "distance_to_goal_1": info.get("distance_to_goal"),
               "decompose_invariant_ok": all(inv_ok),
               "reward": r, "terminated": term, "truncated": trunc}
        print(f"[case3] {name}: moved={moved} dyaw={dyaw} collided={rec['collided']} "
              f"dist {dist0}->{info.get('distance_to_goal')} inv_ok={all(inv_ok)}")
        results.append(rec)
    out = {"case": 3, "scene": scene, "episode_id": episode_id,
           "note": "合成 chunk normalized=False（绕开 bench CE stats），"
                   "验 fm_action.decompose 几何不变量 + bench 实际位移/转角",
           "results": results}
    _save(out_dir, "case3_synthetic_fm", out)
    bench.close()
    return out


# ----------------------------------------------------------------------------
# 用例 1/2：VLM ReAct 闭环
# ----------------------------------------------------------------------------
def case_vlm(model_key, scene="2azQ1b91cZZ", episode_id="11",
             max_turns=25, instruction=None):
    label, url, served, _ = P.MODELS[model_key]
    print(f"\n===== 用例: VLM ReAct 闭环 ({label}) =====")
    bench = BenchClient(BENCH_URL)
    ag = VLMAgent(bench, vllm_url=url, model_name=served,
                  mode="think", verbose=True)
    traj = ag.rollout(scene, episode_id, instruction=instruction,
                      max_turns=max_turns)
    _save(_out_dir(), f"case_vlm_{model_key}", traj, bench=bench)
    bench.close()
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True,
                    choices=["1", "2", "3", "4", "5", "all_vlm"])
    ap.add_argument("--scene", default="2azQ1b91cZZ")
    ap.add_argument("--episode_id", default="11")
    ap.add_argument("--max_turns", type=int, default=25)
    args = ap.parse_args()
    if args.case == "3":
        case3_synthetic_fm(args.scene, args.episode_id)
    elif args.case == "1":
        case_vlm("mobot2b", args.scene, args.episode_id, args.max_turns)
    elif args.case == "2":
        case_vlm("qwen35b", args.scene, args.episode_id, args.max_turns)
    elif args.case == "all_vlm":
        case_vlm("mobot2b", args.scene, args.episode_id, args.max_turns)
        case_vlm("qwen35b", args.scene, args.episode_id, args.max_turns)
    elif args.case in ("4", "5"):
        from fm_policy import run_fm_case  # verl_nav/workers（path 已加）
        run_fm_case(int(args.case), args.scene, args.episode_id,
                    closed_loop=(args.case == "5"))


if __name__ == "__main__":
    main()
