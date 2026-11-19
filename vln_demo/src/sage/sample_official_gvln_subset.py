#!/usr/bin/env python3
"""从 official-gvln episode 池纯随机抽 N 条 trajectory，
按场景分组写精简 demo.json（只含抽中 sample）+ manifest + 抽样报告。

复用 sage.datasets 的目录布局契约：子集目录/<scene_bare>/<scene_bare>_demo.json，
console 侧通过在 SPLITS 注册新 split 即可零改加载。

默认参数保持历史 sample50 行为；正式测试可通过参数选择 candidate 的
validation scene，并把输出标为 validation split。
"""
import argparse
import glob
import json
import os
import random
import sys

# ---- 内联参数（改一处即全局生效）----
SEED = 20260907
N = 50
SRC_ROOT = "/data/zerovln/SAGE-3D/official_gvln_action_head_p0_v1"
SRC_TRAJ = os.path.join(SRC_ROOT, "Trajectory_demo")
SRC_MANIFEST = os.path.join(SRC_ROOT, "manifest.json")
DST_ROOT = "/data/zerovln/SAGE-3D/official_gvln_action_head_p0_sample50_%s" % SEED
SPLIT_NAME = "official_gvln_action_head_p0_sample50"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=SRC_ROOT)
    parser.add_argument("--source-split", choices=("train", "validation"),
                        default="train")
    parser.add_argument("--output-split", choices=("train", "validation"),
                        default="train")
    parser.add_argument("--count", type=int, default=N)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--split-name", default=SPLIT_NAME)
    return parser.parse_args()


def _scene_list(manifest, split):
    splits = manifest.get("splits")
    if isinstance(splits, dict):
        return splits.get(f"{split}_scenes") or []
    scene_split = manifest.get("scene_split")
    if isinstance(scene_split, dict):
        return scene_split.get(f"{split}_scenes") or []
    raise ValueError("source manifest has neither splits nor scene_split")


def _scene_json(source_trajectory_root, bare):
    canonical = os.path.join(
        source_trajectory_root, bare, "%s_demo.json" % bare)
    if os.path.isfile(canonical):
        return canonical
    candidates = sorted(glob.glob(os.path.join(source_trajectory_root, bare, "*.json")))
    if len(candidates) != 1:
        raise FileNotFoundError(
            "expected one source JSON for scene %s, found %d" %
            (bare, len(candidates)))
    return candidates[0]


def _instruction_text(sample):
    instruction = (sample.get("instructions") or [{}])[0]
    return instruction.get("generated_instruction") or instruction.get("instruction") or ""


def main():
    args = _parse_args()
    source_root = os.path.abspath(args.source_root)
    source_trajectory_root = os.path.join(source_root, "Trajectory_demo")
    source_manifest = os.path.join(source_root, "manifest.json")
    output_root = os.path.abspath(args.output_root or DST_ROOT)
    m = json.load(open(source_manifest, encoding="utf-8"))
    source_scenes = _scene_list(m, args.source_split)
    assert source_scenes, "source manifest split is empty"
    if args.count <= 0:
        raise ValueError("--count must be positive")

    # 1. 收集所有 (scene_bare, sample) 对。每 sample = 一条 trajectory（含 1 instruction）。
    all_pairs = []  # list of (scene_bare, sample_dict)
    missing = []
    for sc_name in source_scenes:
        bare = sc_name[len("sage-"):] if sc_name.startswith("sage-") else sc_name
        try:
            fp = _scene_json(source_trajectory_root, bare)
        except FileNotFoundError:
            missing.append(bare)
            continue
        d = json.load(open(fp, encoding="utf-8"))
        for sc_obj in d.get("scenes", []):
            for smp in sc_obj.get("samples", []):
                if not smp.get("points") or len(smp["points"]) < 2:
                    continue
                all_pairs.append((bare, smp))
    if missing:
        sys.stderr.write("WARN: %d train scenes missing demo.json: %s\n" % (len(missing), missing[:5]))
    assert all_pairs, "未收集到任何 trajectory"

    # 2. 纯随机抽 N（允许同场景多条）
    random.seed(args.seed)
    picked = random.sample(all_pairs, min(args.count, len(all_pairs)))

    # 3. 按场景分组写精简 demo.json
    by_scene = {}
    for bare, smp in picked:
        by_scene.setdefault(bare, []).append(smp)

    os.makedirs(output_root, exist_ok=True)
    scenes_manifest = {}
    for bare, samples in by_scene.items():
        scdir = os.path.join(output_root, bare)
        os.makedirs(scdir, exist_ok=True)
        # 读原 demo 拿 pipeline + scenes 结构，只保留抽中 trajectory_id 的 sample
        orig_fp = _scene_json(source_trajectory_root, bare)
        orig = json.load(open(orig_fp, encoding="utf-8"))
        picked_tids = {s.get("trajectory_id") for s in samples}
        new_scenes = []
        for sc_obj in orig.get("scenes", []):
            kept = [s for s in sc_obj.get("samples", []) if s.get("trajectory_id") in picked_tids]
            if kept:
                new_scenes.append({**sc_obj, "samples": kept})
        new_d = {"pipeline": orig.get("pipeline"), "scenes": new_scenes}
        with open(os.path.join(scdir, "%s_demo.json" % bare), "w", encoding="utf-8") as f:
            json.dump(new_d, f, ensure_ascii=False)
        scenes_manifest[bare] = len(samples)

    # 4. 写 manifest + 抽样报告
    manifest = {
        "schema": "sage_action_head_manifest_subset_v1",
        "split_unit": "physical_trajectory",
        "split_name": args.split_name,
        "seed": args.seed,
        "n_picked": len(picked),
        "n_scenes": len(by_scene),
        "source_root": source_root,
        "source_manifest": source_manifest,
        "source_split": args.source_split,
        "source_split_scenes": len(source_scenes),
        "source_total_trajectories": len(all_pairs),
        "dest_root": output_root,
        "sampling": "uniform random without replacement; same-scene multi-trajectory allowed",
        "splits": {
            "salt": "sample-official-gvln-%s-%s" % (args.source_split, args.seed),
            "validation_percent": 100 if args.output_split == "validation" else 0,
            "train_scenes": (["sage-" + b for b in sorted(by_scene)]
                             if args.output_split == "train" else []),
            "validation_scenes": (["sage-" + b for b in sorted(by_scene)]
                                  if args.output_split == "validation" else []),
        },
        "scenes": scenes_manifest,
        "picked": [
            {"scene": b, "trajectory_id": s.get("trajectory_id"),
             "instruction": _instruction_text(s)}
            for b, s in picked
        ],
    }
    with open(os.path.join(output_root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 5. 终端报告
    print("=== 抽样完成 ===")
    print("seed:", args.seed)
    print("source", args.source_split, "scenes:", len(source_scenes),
          "| source trajectories:", len(all_pairs))
    print("picked trajectories:", len(picked), "across", len(by_scene), "scenes")
    print("dest:", output_root)
    print("split_name:", args.split_name)
    print("同场景多条样本的场景数:", sum(1 for v in by_scene.values() if len(v) > 1))
    print("\n前10条抽样:")
    for i, (b, s) in enumerate(picked[:10]):
        ins = _instruction_text(s)
        print("  %2d  scene=%s  tid=%s  instr=%s" % (i + 1, b, s.get("trajectory_id"), ins[:60]))


if __name__ == "__main__":
    main()
