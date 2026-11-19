"""Build action expert demo episodes from cleaned official SAGE-3D GVLN.

Combines:
  * **Trajectory**: re-planned with ECM centered A* + collision-aware smoothing
    + physical gate chain (plan / smooth / swept / execution_replay / route
    gate), reusing ``build_demo_episodes._try_candidate``'s sub-functions.
    Start/goal pose + goal object are taken from the physically cleaned
    official GVLN trajectories (``official_gvln_train_candidate_v1``).  The
    path is NOT the official raw one -- it is re-planned so it is centered,
    smoothed, and provably collision-free (usd_rooms v4 ``object_obstacle``
    blocks furniture, verified ``free ∩ object_obstacle == 0``; replay-passed
    routes do not穿家具 or stick on furniture).
  * **Instruction**: the official verbatim ``Relative_Relationship``
    instruction ("go to X near Y") of that trajectory -- the official type
    closest to a navigation/action description (official GVLN has no
    turn-left/right action type; a 28k-instruction scan found zero turn
    phrases).  Using the official instruction verbatim sidesteps the
    self-generated Maneuver-based instruction's direction-sign bug.

The terminal VLM visibility / bearing gates of r7 are skipped (downstream SFT
chain replays a small VLM window at every slow anchor, not endpoint).  Pure
CPU; no render server or VLM needed.  Length / turn gates widened for the
official long navigation trajectories (mean 6.67m, multi-turn, cross-door).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Environment + config overrides applied BEFORE importing build_demo_episodes,
# so its module-level `pipeline_config_from_argv()` picks up the widened
# length/turn/clearance gates (action_head_p0 profile; instruction generation
# is NOT used -- official RR instruction is kept verbatim).
# --------------------------------------------------------------------------- #
SAGE_ROOT = "/data/zerovln/SAGE-3D"
DEFAULT_OFFICIAL_ROOT = os.path.join(
    SAGE_ROOT, "official_gvln_train_candidate_v1", "Trajectory_demo")
DEFAULT_OUTPUT_ROOT = os.path.join(
    SAGE_ROOT, "official_gvln_action_head_p0_v1")
P0P2_TESTSET_ROOT = "/data/zerovln/action_head/curriculum/testset/Trajectory_demo"
INSTRUCTION_TYPE = "Relative_Relationship"

os.environ.setdefault("SAGE_COLLISION", "usd_rooms")
os.environ.setdefault("SAGE_FURNITURE_COLLISION", "none")
os.environ.setdefault("SAGE_ROBOT_RADIUS", "0.07")
os.environ.setdefault("SAGE_CAMERA_HEIGHT", "1.2")
os.environ.setdefault("SAGE_HFOV", "115")
os.environ.setdefault("SAGE_USD_COLLISION_MASK_ROOT",
                      os.path.join(SAGE_ROOT, "sage_usd_rooms_masks"))

_PRE_IMPORT_ARGV = [
    "--set", "capability_profile.name=action_head_p0",
    "--set", "capability_profile.require_cross_room=false",
    "--set", "capability_profile.start_target_visibility_max_arc_m=0",
    "--set", "capability_profile.min_turns=0",
    "--set", "capability_profile.max_turns=999",
    "--set", "capability_profile.min_doors=0",
    "--set", "capability_profile.max_doors=999",
    "--set", "capability_profile.min_decisions=0",
    "--set", "capability_profile.min_clearance_m=0.12",
    "--set", "capability_profile.max_clearance_m=999.0",
    "--set", "sampling.pair_euclidean_m=[1.2,40.0]",
    "--set", "sampling.pair_geodesic_m=[1.2,40.0]",
    "--set", "runtime.trace_rejections=false",
]

_USER_ARGV = list(sys.argv)
sys.argv = [sys.argv[0]] + _PRE_IMPORT_ARGV
REPO = Path("/workspace/ZeroVLN")
sys.path.insert(0, str(REPO / "vln_demo"))
from sage import build_demo_episodes as bde  # noqa: E402
from sage.datasets import _to_geometry_frame  # noqa: E402
from sage import route_extractor  # noqa: E402
from sage.planner import passable_grid, plan_path, smooth_path  # noqa: E402
sys.argv = _USER_ARGV


def _select_scenes(args) -> List[str]:
    if args.scenes_file:
        return [ln.strip() for ln in open(args.scenes_file) if ln.strip()]
    import random
    cand = json.load(open(os.path.join(
        SAGE_ROOT, "official_gvln_train_candidate_v1", "manifest.json")))
    sp = cand["scene_split"]
    p0p2 = set(s for s in os.listdir(P0P2_TESTSET_ROOT)
               if os.path.isdir(os.path.join(P0P2_TESTSET_ROOT, s)))
    train = [s for s in sp["train_scenes"] if s not in p0p2]
    val = [s for s in sp["validation_scenes"] if s not in p0p2]
    rng = random.Random(args.seed)
    rng.shuffle(train); rng.shuffle(val)
    n_val = max(1, int(round(args.scene_limit * 0.1)))
    return train[:args.scene_limit - n_val] + val[:n_val]


def _load_official_scene_trajectories(scene: str, official_root: str) -> List[dict]:
    """Each trajectory: start_xy/goal_xy (geometry frame), start/goal obj, RR instr."""
    import glob
    fs = glob.glob(os.path.join(official_root, scene, "*.json"))
    if not fs:
        return []
    payload = json.load(open(fs[0]))
    out = []
    for sc in payload.get("scenes", []):
        for smp in sc.get("samples", []):
            pts = smp.get("points") or []
            if len(pts) < 2:
                continue
            instrs = smp.get("instructions") or []
            rr = next((i for i in instrs
                       if (i.get("instruction_type") or "").strip() == INSTRUCTION_TYPE), None)
            if rr is None:
                continue
            se = [(i.get("start"), i.get("end")) for i in instrs]
            start_obj, goal_obj = max(set(se), key=se.count)
            p0, pN = pts[0]["position"], pts[-1]["position"]
            gx0, gy0, _ = _to_geometry_frame(float(p0[0]), float(p0[1]), 0.0)
            gxN, gyN, _ = _to_geometry_frame(float(pN[0]), float(pN[1]), 0.0)
            out.append({
                "trajectory_id": str(smp.get("trajectory_id")),
                "start_xy": (float(gx0), float(gy0)),
                "goal_xy": (float(gxN), float(gyN)),
                "start_obj": start_obj, "goal_obj": goal_obj,
                "rr_instruction": rr,
            })
    return out


def _replan_candidate(bare, occ, comp, room_lab, label_index,
                      start_xy, goal_xy, goal_obj, start_obj, rr_instruction
                      ) -> Tuple[Optional[dict], Optional[str]]:
    """ECM A* + smooth + physical gates; instruction = official RR verbatim."""
    if goal_obj not in label_index:
        return None, "goal_obj_not_in_labels"
    # 1. ECM centered A*
    raw = plan_path(occ, start_xy, goal_xy, bde.PLANNING_RADIUS_M,
                    mode=bde.PLANNER_MODE, centered_config=bde.CENTERED_CONFIG)
    if raw is None or len(raw) < 2:
        return None, "plan_fail"
    raw = list(raw)
    if math.hypot(raw[-1][0] - goal_xy[0], raw[-1][1] - goal_xy[1]) > 1e-6:
        raw.append(goal_xy)
    # 2. collision-aware smoothing
    try:
        wp, smoothing = smooth_path(occ, raw, bde.PLANNING_RADIUS_M, bde.SMOOTHING_CONFIG)
    except ValueError:
        return None, "smooth_fail"
    if smoothing.get("blend_strength", 0.0) + 1e-9 < bde.MIN_ACCEPTED_SMOOTHING_BLEND:
        return None, "smooth_blend_low:%.3g" % smoothing.get("blend_strength", 0.0)
    # 3. swept collision
    if not bde._swept_free(occ, wp):
        return None, "swept_block"
    # 4. execution replay (true kinematic -- catches家具 sticking / doorway block)
    ok, replay = bde._execution_replay_free(occ, wp)
    if not ok:
        return None, "replay_block:%s" % replay.get("reason", "")
    # 5. route gate (turns/doors/landmarks/clearance)
    route = route_extractor.extract(wp, occ, label_index, bare,
                                    start_obj or "", goal_obj, turn_waypoints=raw)
    rt = len([x for x in route.get("turns", []) if x != "straight"])
    rd = int(route.get("doors_crossed") or 0)
    rlm = len(route.get("landmarks") or [])
    mc = min(float(occ.clearance(x, y)) for x, y in wp)
    if not (bde.MIN_ROUTE_TURNS <= rt <= bde.MAX_ROUTE_TURNS
            and bde.MIN_ROUTE_DOORS <= rd <= bde.MAX_ROUTE_DOORS
            and rt + rd >= bde.MIN_ROUTE_DECISIONS
            and rlm >= bde.MIN_ROUTE_LANDMARKS
            and bde.MIN_ROUTE_CLEARANCE_M <= mc <= bde.MAX_ROUTE_CLEARANCE_M):
        return None, "route_gate(turns=%d doors=%d lm=%d clear=%.3f)" % (rt, rd, rlm, mc)

    length = sum(math.hypot(wp[i+1][0]-wp[i][0], wp[i+1][1]-wp[i][1]) for i in range(len(wp)-1))
    yaws = bde._waypoint_yaws(wp)
    # instruction = official RR verbatim (NOT _gen_instructions -- avoids the
    # self-generated Maneuver-based direction-sign bug)
    instrs = [dict(rr_instruction)]
    traj_id = "official_%s_%s" % (bare, id(start_xy) % 100000)  # caller retags
    sample = bde._assemble_sample(traj_id, wp, yaws, instrs)
    sample["route_grounding"] = route
    sample["goal_constraints"] = {
        "object": goal_obj,
        "planner": bde.PLANNER_MODE,
        "planning_radius_m": bde.PLANNING_RADIUS_M,
        "path_smoothing": smoothing,
        "execution_replay": replay,
        "route_min_clearance_m": round(mc, 4),
        "path_length_m": round(length, 3),
        "terminal_visibility_skipped": True,
        "instruction_source": "official Relative_Relationship verbatim",
    }
    return sample, None


def _build_scene(scene, official_root, out_vln, max_traj_per_scene, reject_log, max_tries=0):
    bare = scene
    occ = bde._load_occ(bare)
    label_index = bde._build_label_index(bare)
    comp = bde._connected_components(passable_grid(occ, bde.ROBOT_RADIUS_M))
    room_lab = route_extractor._room_label_map(occ)
    trajs = _load_official_scene_trajectories(bare, official_root)
    samples: List[dict] = []
    n_try = n_ok = 0
    for t in trajs:
        n_try += 1
        if max_tries and n_try > max_tries:
            break
        if 0 < max_traj_per_scene <= n_ok:
            break
        sample, reason = _replan_candidate(
            bare, occ, comp, room_lab, label_index,
            t["start_xy"], t["goal_xy"], t["goal_obj"], t["start_obj"],
            t["rr_instruction"])
        if sample is None:
            reject_log.append({"scene": bare, "traj": t["trajectory_id"], "reason": reason})
            continue
        sample["trajectory_id"] = "official_%s_%s" % (bare, t["trajectory_id"])
        sample["source_official"] = {
            "scene": bare, "trajectory_id": t["trajectory_id"],
            "instruction_type": INSTRUCTION_TYPE,
            "start": t["start_obj"], "goal": t["goal_obj"],
        }
        samples.append(sample); n_ok += 1
    return samples, {"scene": bare, "tried": n_try, "accepted": n_ok, "total_official": len(trajs)}


def _write_scene_json(scene, samples, out_vln):
    scene_dir = os.path.join(out_vln, scene)
    os.makedirs(scene_dir, exist_ok=True)
    out_path = os.path.join(scene_dir, "%s_demo.json" % scene)
    doc = {
        "pipeline": {
            "name": "sage_official_gvln_action_head_v1",
            "version": "2026-08-28-official-rr-replanned",
            "canonical": False,
            "source": "official_gvln_train_candidate_v1",
            "trajectory": "ECM centered A* + collision-aware smoothing (re-planned from official start/goal)",
            "instruction": "official Relative_Relationship verbatim (no generation)",
            "planner": bde.PLANNER_MODE,
            "planner_config": asdict(bde.CENTERED_CONFIG),
            "path_smoothing": asdict(bde.SMOOTHING_CONFIG),
            "robot_radius_m": bde.ROBOT_RADIUS_M,
            "planning_radius_m": bde.PLANNING_RADIUS_M,
            "execution_replay_step_m": bde.REPLAY_STEP_M,
            "collision": "usd_rooms + furniture=none (object_obstacle blocks furniture)",
            "gates": "plan + smooth + swept + replay + route(turns/doors/clearance); terminal VLM/bearing SKIPPED",
        },
        "scenes": [{"scene_id": "demo_%s" % scene, "scene_name": scene, "samples": samples}],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False); f.write("\n")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scenes-file", default=None)
    parser.add_argument("--scene-limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--max-traj-per-scene", type=int, default=0)
    parser.add_argument("--max-tries-per-scene", type=int, default=0)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()

    out_vln = os.path.join(args.output_root, "Trajectory_demo")
    os.makedirs(out_vln, exist_ok=True)
    scenes = _select_scenes(args)
    if args.smoke:
        scenes = scenes[:args.smoke]
    print("[official-action-head] %d scenes; output=%s" % (len(scenes), out_vln), flush=True)

    all_stats, reject_log = [], []
    for i, scene in enumerate(scenes, 1):
        try:
            samples, stats = _build_scene(scene, args.official_root, out_vln,
                                          args.max_traj_per_scene, reject_log,
                                          max_tries=args.max_tries_per_scene)
        except FileNotFoundError as exc:
            print("[%d/%d] %s SKIP %s" % (i, len(scenes), scene, exc), flush=True); continue
        if samples:
            _write_scene_json(scene, samples, out_vln)
        all_stats.append(stats)
        print("[%d/%d] %s tried=%d accepted=%d" % (i, len(scenes), scene, stats["tried"], stats["accepted"]), flush=True)

    total_ok = sum(s["accepted"] for s in all_stats)
    total_try = sum(s["tried"] for s in all_stats)
    from collections import Counter
    rej = Counter(r["reason"].split(":")[0] for r in reject_log)
    prov = {
        "schema": "sage_official_gvln_action_head_demo_v1",
        "source": "official_gvln_train_candidate_v1",
        "official_root": args.official_root, "output_root": args.output_root,
        "config_overrides": _PRE_IMPORT_ARGV,
        "collision_contract": {"collision": "usd_rooms", "furniture": "none",
                               "robot_radius_m": bde.ROBOT_RADIUS_M,
                               "planning_radius_m": bde.PLANNING_RADIUS_M,
                               "replay_step_m": bde.REPLAY_STEP_M,
                               "camera_height_m": bde.CAMERA_Z_M, "hfov_deg": bde.CAMERA_HFOV_DEG},
        "trajectory": "ECM A* + smooth re-planned from official start/goal",
        "instruction": "official Relative_Relationship verbatim",
        "scenes_processed": len(all_stats),
        "scenes_with_samples": sum(1 for s in all_stats if s["accepted"] > 0),
        "trajectories_tried": total_try, "trajectories_accepted": total_ok,
        "accept_rate": round(total_ok / total_try, 4) if total_try else 0.0,
        "reject_reasons": dict(rej), "scene_stats": all_stats,
    }
    prov_path = os.path.join(args.output_root, "build_provenance.json")
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(prov, f, ensure_ascii=False, indent=2); f.write("\n")
    print("[official-action-head] DONE tried=%d accepted=%d rate=%.3f rej=%s"
          % (total_try, total_ok, prov["accept_rate"], dict(rej)), flush=True)
    if args.write_manifest:
        _write_sft_manifest(args.output_root, out_vln)


def _write_sft_manifest(output_root, out_vln):
    present = sorted(d for d in os.listdir(out_vln) if os.path.isdir(os.path.join(out_vln, d)))
    cand = json.load(open(os.path.join(SAGE_ROOT, "official_gvln_train_candidate_v1", "manifest.json")))
    sp = cand["scene_split"]
    train_set, val_set = set(sp["train_scenes"]), set(sp["validation_scenes"])
    manifest = {
        "schema": "sage_action_head_manifest_v1", "split_unit": "physical_trajectory",
        "source_root": out_vln, "source_json_count": len(present), "scene_count": len(present),
        "instruction_policy": {"instruction_type": INSTRUCTION_TYPE,
                                "instruction_level": "official",
                                "one_instruction_per_physical_trajectory": True},
        "splits": {"salt": "sage-official-gvln-action-head-v1", "validation_percent": 10,
                   "train_scenes": ["sage-%s" % s for s in present if s in train_set],
                   "validation_scenes": ["sage-%s" % s for s in present if s in val_set]},
    }
    man_path = os.path.join(output_root, "manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2); f.write("\n")
    print("[official-action-head] manifest=%s" % man_path, flush=True)


if __name__ == "__main__":
    main()
