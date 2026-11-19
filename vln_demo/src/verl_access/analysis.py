# -*- coding: utf-8 -*-
"""analysis — 跑完用例后的汇总：映射正确率 / 动作逻辑性 / FM chunk 健康度。

读 outputs/verl_access/<date>/*.jsonl，出文本摘要 + summary.md。
"""
import os
import sys
import json
import glob
import math

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _load(path):
    with open(path) as f:
        return json.load(f)


def analyze_vlm(traj):
    """VLM 闭环轨迹 -> 摘要 dict。"""
    turns = traj.get("turns", [])
    n_turns = len(turns)
    all_calls = [c for t in turns for c in t.get("calls", [])]
    n_calls = len(all_calls)
    stop_calls = [c for c in all_calls if c.get("name") == "stop"]
    # 映射正确率：每个 call 都有 info（执行成功）= 没抛异常
    executed = [c for c in all_calls if "info" in c]
    map_ok = len(executed) / n_calls if n_calls else 0.0
    # 距离曲线（每 turn 末）
    dists = [c.get("info", {}).get("distance_to_goal") for c in all_calls
             if "info" in c]
    dists = [d for d in dists if d is not None]
    d0 = dists[0] if dists else None
    d1 = dists[-1] if dists else None
    # 距离是否单调下降（允许一次反弹）
    decr = sum(1 for i in range(1, len(dists)) if dists[i] <= dists[i-1] + 0.05)
    decr_frac = decr / max(1, len(dists) - 1) if len(dists) > 1 else 0.0
    # 碰撞率
    collided = sum(1 for c in executed if c.get("info", {}).get("collided_this_step"))
    coll_frac = collided / max(1, len(executed))
    final = traj.get("final", {})
    return {
        "model": traj.get("model"), "n_turns": n_turns, "n_calls": n_calls,
        "n_executed": len(executed), "map_success_rate": round(map_ok, 3),
        "stop_called": len(stop_calls) > 0, "why": traj.get("why"),
        "dist_start": d0, "dist_end": d1,
        "dist_decrease_frac": round(decr_frac, 3),
        "collision_rate": round(coll_frac, 3),
        "success": final.get("success"), "spl": final.get("spl"),
        "steps_used": final.get("steps_used"),
        "final": final,
    }


def analyze_fm_closed(traj):
    """FM 闭环轨迹 -> 摘要。"""
    steps = traj.get("steps", [])
    chunks = [s for s in steps if s.get("chunk_norm") is not None]
    nonzero = sum(1 for s in chunks if any(abs(v) > 1e-4 for v in s["chunk_norm"].get("flat", [])))
    ranges = []
    for s in chunks:
        flat = s["chunk_norm"].get("flat", [])
        if flat:
            ranges.append((min(flat), max(flat)))
    dists = [s.get("distance_to_goal") for s in steps
             if s.get("distance_to_goal") is not None]
    d0 = dists[0] if dists else None
    d1 = dists[-1] if dists else None
    collided = sum(1 for s in steps if s.get("collided"))
    return {
        "n_steps": len(steps), "n_chunks": len(chunks),
        "nonzero_chunk_frac": round(nonzero / max(1, len(chunks)), 3),
        "chunk_norm_range": {"min": min(r[0] for r in ranges) if ranges else None,
                             "max": max(r[1] for r in ranges) if ranges else None},
        "dist_start": d0, "dist_end": d1,
        "collision_rate": round(collided / max(1, len(steps)), 3),
        "success": traj.get("final", {}).get("success"),
        "why": traj.get("why"),
    }


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else None
    if not d:
        root = os.path.join(os.path.dirname(_SRC), "outputs", "verl_access")
        dirs = sorted(glob.glob(os.path.join(root, "*")))
        d = dirs[-1] if dirs else None
    if not d or not os.path.isdir(d):
        print("no out dir"); return
    lines = [f"# verl_access 汇总 ({d})\n"]
    for fn in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        traj = _load(fn)
        lines.append(f"## {os.path.basename(fn)}")
        if "turns" in traj:
            s = analyze_vlm(traj)
            lines.append(json.dumps(s, ensure_ascii=False, indent=2))
        elif "steps" in traj:
            s = analyze_fm_closed(traj)
            lines.append(json.dumps(s, ensure_ascii=False, indent=2))
        elif "results" in traj:
            lines.append(json.dumps(traj, ensure_ascii=False, indent=2)[:2000])
        lines.append("")
    out = os.path.join(d, "summary.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
