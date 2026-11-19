#!/usr/bin/env python3
"""Exercise isolated R2R/RxR HTTP gyms before model evaluation, without a model."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np

import v6_vlnce_eval as V

sys.path[:0] = [str(V.REPO / "vln_demo/src"), str(V.REPO / "verl")]
from action_head.render_client import RemoteRenderClient  # noqa: E402
from verl_access.bench_client import BenchClient  # noqa: E402
from vln_gym.vlnce_action_head import canonical_scene, NATIVE_SUCCESS_CONTRACT  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frame_record(observation, info):
    rgb = np.asarray(observation.get("rgb"))
    require(rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.size > 0,
            "current RGB frame missing or malformed")
    audit = copy.deepcopy(info.get("sensor_audit") or {})
    record = {
        "world_xy": info.get("world_xy"),
        "world_yaw_deg": info.get("world_yaw_deg"),
        "steps_used": info.get("steps_used"),
        "jpeg_decoded_rgb_shape": list(rgb.shape),
        "jpeg_decoded_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "sensor_audit": audit,
        "metrics": {key: info.get(key) for key in (
            "success", "spl", "distance_to_goal", "oracle_success",
            "oracle_navigation_error", "path_length_m", "shortest_path_m",
            "collided_this_step", "blocked_this_step", "episode_over")},
        "actual_scene": info.get("actual_scene", info.get("scene")),
        "actual_episode_id": info.get("actual_episode_id", info.get("episode_id")),
        "observation_profile": copy.deepcopy(info.get("observation_profile")),
    }
    return record


def validate_frame(record):
    pose = record["world_xy"]
    require(isinstance(pose, list) and len(pose) == 2, "actual planar pose missing")
    require(all(math.isfinite(float(value)) for value in [*pose, record["world_yaw_deg"]]),
            "actual planar pose nonfinite")
    require(isinstance(record["steps_used"], int) and record["steps_used"] >= 0,
            "step counter missing or negative")
    audit = record["sensor_audit"]
    require(audit.get("schema") == "habitat_same_frame_rgb_metric_depth_v1",
            "same-frame RGB/depth audit unavailable")
    require(audit.get("rgb_shape") == record["jpeg_decoded_rgb_shape"],
            "RGB audit and delivered frame dimensions differ")
    require(audit.get("depth_shape", [])[:2] == record["jpeg_decoded_rgb_shape"][:2],
            "RGB and metric depth dimensions differ")
    for key in ("rgb_sha256", "depth_sha256"):
        require(isinstance(audit.get(key), str) and len(audit[key]) == 64,
                f"missing {key}")
    require(audit.get("depth_finite_fraction") == 1., "depth contains nonfinite pixels")
    require(0. <= float(audit["depth_min_m"]) <= float(audit["depth_max_m"])
            and math.isfinite(float(audit["depth_max_m"])) and audit["depth_max_m"] > 0.,
            "metric depth range invalid")
    require(audit.get("world_xy") == pose
            and audit.get("world_yaw_deg") == record["world_yaw_deg"],
            "RGB/depth audit pose differs from current actual pose")
    for key in ("success", "spl", "distance_to_goal", "oracle_success",
                "oracle_navigation_error", "path_length_m", "shortest_path_m"):
        value = record["metrics"][key]
        require(isinstance(value, (int, float, bool)) and math.isfinite(float(value)),
                f"native metric {key} missing or nonfinite")
    require(record["metrics"]["path_length_m"] >= 0., "negative native path length")


def yaw_delta(before, after):
    return (after["world_yaw_deg"] - before["world_yaw_deg"] + 180.) % 360. - 180.


def validate_catalog(spec, cases, dataset):
    require(spec.get("backend") == dataset, "Gym backend differs from requested dataset")
    require(spec.get("success_contract") == NATIVE_SUCCESS_CONTRACT,
            "Gym is missing the Action Head native scoring contract")
    actual = {(canonical_scene(scene, dataset), str(episode_id))
              for scene, episodes in spec.get("scene_eps", {}).items()
              for episode_id in episodes}
    expected = {(canonical_scene(case["scene"], dataset), str(case["episode_id"]))
                for case in cases}
    require(actual == expected,
            f"full catalog mismatch: missing={sorted(expected - actual)[:5]} "
            f"extra={sorted(actual - expected)[:5]}")
    require(sum(map(len, spec["scene_eps"].values())) == len(actual),
            "Gym catalog contains duplicate episode identities")
    return {"expected_cases": len(expected), "available_cases": len(actual),
            "scenes": len({scene for scene, _ in actual}), "ok": True}


def run_smoke(dataset, cases, bench, render, record, selected=None):
    record["health"] = bench.health()
    require(record["health"].get("ok") and record["health"].get("env_loaded"),
            "Gym is not ready")
    record["catalog"] = validate_catalog(bench.spec(), cases, dataset)
    if selected is None:
        selected = [cases[0]]
        selected.append(next(case for case in cases if case["scene"] != selected[0]["scene"]))
    record["cases"] = []
    for case in selected:
        case_result = {"logical_id": case["logical_id"], "scene": case["scene"],
                       "episode_id": case["episode_id"], "instruction": case["instruction"],
                       "events": []}
        record["cases"].append(case_result)

        def capture(action, obs, info, **extra):
            event = {"action": action, "after": frame_record(obs, info), **extra}
            if case_result["events"]:
                event["before"] = copy.deepcopy(case_result["events"][-1]["after"])
            case_result["events"].append(event)
            validate_frame(event["after"])
            require(canonical_scene(event["after"]["actual_scene"], dataset)
                    == canonical_scene(case["scene"], dataset)
                    and str(event["after"]["actual_episode_id"]) == str(case["episode_id"]),
                    "motion changed the active episode identity")
            for key, value in render.ui_metrics().items():
                if isinstance(value, (int, float, bool)):
                    require(math.isfinite(float(value)), f"UI metric {key} nonfinite")
            return event["after"]

        def reset(label):
            obs, info = render.reset(scene=case["scene"], episode_id=case["episode_id"],
                                     instruction=case["instruction"], seed=case.get("seed", 0))
            V.validate_reset(case, obs, info)
            require(info.get("actual_instruction") == case["instruction"],
                    "actual Habitat instruction differs from manifest")
            value = capture(label, obs, info)
            require(value["steps_used"] == 0 and value["metrics"]["path_length_m"] == 0.,
                    "reset did not clear step/path counters")
            return value

        initial = reset("reset")
        before = initial
        for direction, signed_degrees in (("left", 15.), ("right", -15.)):
            obs, tool_result = render.execute_turn(direction, abs(signed_degrees))
            after = capture("turn", obs, render.last_info,
                            arguments={"direction": direction, "degrees": abs(signed_degrees)},
                            tool_result=tool_result)
            require(abs(yaw_delta(before, after) - signed_degrees) < .5,
                    f"{direction} turn has wrong measured yaw")
            require(after["steps_used"] > before["steps_used"], "turn did not advance steps")
            require(after["jpeg_decoded_rgb_sha256"] != before["jpeg_decoded_rgb_sha256"],
                    "turn returned unchanged RGB despite actual rotation")
            before = after

        prefixes = ("fm_turn", [[0., 0., math.radians(5.)]] * 2), ("fm_forward", [[.08, 0., 0.]] * 2)
        for label, waypoints in prefixes:
            obs, reward, terminated, truncated, info = render.execute_prefix(waypoints)
            after = capture(label, obs, info, waypoints=waypoints,
                            reward=reward, terminated=terminated, truncated=truncated,
                            waypoint_log=copy.deepcopy(info.get("waypoint_log")))
            require(not terminated and not truncated, "small prefix unexpectedly ended episode")
            require(len(info.get("waypoint_log") or []) == len(waypoints),
                    "FM prefix omitted waypoint execution records")
            require(after["steps_used"] > before["steps_used"], "FM prefix did not advance steps")
            delta = yaw_delta(before, after)
            expected_yaw = 10. if label == "fm_turn" else 0.
            require(abs(delta - expected_yaw) < .5, "FM prefix yaw sign or magnitude incorrect")
            moved = math.dist(before["world_xy"], after["world_xy"])
            if label == "fm_turn" or moved > .02:
                require(after["jpeg_decoded_rgb_sha256"] != before["jpeg_decoded_rgb_sha256"],
                        "FM prefix returned stale RGB after measured movement")
            if label == "fm_forward" and not (info.get("blocked_this_step") or info.get("collided_this_step")):
                require(abs(moved - .16) < .03, "unblocked prefix did not execute requested displacement")
            before = after

        # STOP exercises the unchanged Habitat terminal scoring. It is an
        # acceptance action and is never counted as a model rollout.
        obs, reward, terminated, truncated, info = bench.step_tool("stop", {"reason": "HTTP acceptance probe"})
        render.last_info = info
        capture("stop", obs, info, reward=reward, terminated=terminated, truncated=truncated)
        require(terminated and info.get("episode_over"), "native STOP did not terminate episode")
        clean = reset("reset_after_stop")
        require(math.dist(clean["world_xy"], initial["world_xy"]) < 1e-5
                and abs(yaw_delta(initial, clean)) < 1e-4,
                "reset did not restore original actual pose")
        require(clean["jpeg_decoded_rgb_sha256"] == initial["jpeg_decoded_rgb_sha256"],
                "reset did not restore original camera observation")
        case_result["ok"] = True
    record["ok"] = True
    return record


def run_reset_audit(cases, bench, render, record, audit_path, progress_path):
    """Validate all affected resets on the real task, then exercise movement."""
    audit = json.loads(audit_path.read_text())
    require(audit['schema'] == 'vlnce_native_reset_distance_audit_v1', 'invalid reset audit')
    by_id = {case['logical_id']: case for case in cases}
    expected = {row['logical_id'] for row in audit['rows']}
    require(expected <= by_id.keys(), 'audit contains cases outside the locked evaluation set')
    record['reset_audit'] = dict(input=str(audit_path), completed=0,
                                previously_invalid_completed=0, results=[])
    selected, seen_starts = [], set()
    for row in audit['rows']:
        case = by_id[row['logical_id']]
        obs, info = render.reset(scene=case['scene'], episode_id=case['episode_id'],
                                 instruction=case['instruction'], seed=case.get('seed', 0))
        V.validate_reset(case, obs, info)
        frame = frame_record(obs, info)
        validate_frame(frame)
        require(info.get('actual_instruction') == case['instruction'], 'instruction changed')
        mesh_identity = info.get('navmesh_identity', {})
        require(mesh_identity.get('navmesh_sha256') == audit['meshes'][case['scene']]['sha256'],
                'Gym loaded a different navmesh than the audited candidate')
        require(math.dist(info['oracle_habitat_geometry']['position'], row['start']['position']) < 1e-5,
                'reset changed the benchmark start position')
        distance = row['multi']['distance_m']
        require(distance is not None and abs(info['distance_to_goal']-distance) < 1e-4,
                'native task distance differs from independent PathFinder audit')
        require(abs(info['shortest_path_m']-distance) < 1e-4 and info['path_length_m'] == 0.,
                'native SPL reset denominator differs from audited distance')
        record['reset_audit']['results'].append(dict(logical_id=case['logical_id'],
            episode_id=case['episode_id'], previously_invalid=row['previously_invalid'],
            distance_to_goal=info['distance_to_goal'], navmesh_identity=mesh_identity,
            sensor_audit=info['sensor_audit']))
        record['reset_audit']['completed'] += 1
        record['reset_audit']['previously_invalid_completed'] += int(row['previously_invalid'])
        V.atomic_json(progress_path, {k: v for k, v in record['reset_audit'].items() if k != 'results'})
        start = tuple(row['start']['position'])
        if row['previously_invalid'] and start not in seen_starts and len(selected) < 2:
            selected.append(case)
            seen_starts.add(start)
    require(len(selected) == 2, 'need two distinct previously failed start poses')
    # Leave and return to the repaired scene to exercise the scene-load hook.
    unaffected = next(case for case in cases if case['scene'] != selected[0]['scene'])
    run_smoke('rxr', cases, bench, render, record,
              selected=[selected[0], unaffected, selected[1]])


def run_backend_switch_smoke(cases_by_dataset, bench, render, record):
    """Verify that one actor can switch datasets without carrying episode state."""
    record["ok"] = False
    record["backend_sequence"] = ["r2r", "rxr", "r2r"]
    record["stages"] = []
    initial = None
    for index, dataset in enumerate(record["backend_sequence"]):
        cases = cases_by_dataset[dataset]
        case = cases[0]
        stage = {"dataset": dataset, "phase": "switch_and_motion" if index < 2
                 else "return_to_r2r", "ok": False,
                 "requested": {key: case[key] for key in
                               ("logical_id", "scene", "episode_id", "instruction")}}
        record["stages"].append(stage)
        obs, info = bench.reset(backend=dataset, scene=case["scene"],
                                episode_id=case["episode_id"],
                                instruction=case["instruction"], seed=case.get("seed", 0))
        render.last_info = copy.deepcopy(info)
        stage["reset"] = frame_record(obs, info)
        stage["actual_backend"] = info.get("backend")
        stage["actual_instruction"] = info.get("actual_instruction")
        V.validate_reset(case, obs, info)
        require(info.get("backend") == dataset, "backend switch did not select requested dataset")
        require(info.get("actual_instruction") == case["instruction"],
                "backend switch loaded a different Habitat instruction")
        validate_frame(stage["reset"])
        require(stage["reset"]["steps_used"] == 0
                and stage["reset"]["metrics"]["path_length_m"] == 0.
                and not stage["reset"]["metrics"]["episode_over"],
                "backend switch retained previous episode counters or terminal state")
        stage["catalog"] = validate_catalog(bench.spec(), cases, dataset)
        if index < 2:
            if index == 0:
                initial = copy.deepcopy(stage["reset"])
            stage["motion_probe"] = {}
            run_smoke(dataset, cases, bench, render, stage["motion_probe"])
        else:
            clean = stage["reset"]
            stage["restored_r2r"] = {
                "position_error_m": math.dist(clean["world_xy"], initial["world_xy"]),
                "yaw_error_deg": abs(yaw_delta(initial, clean)),
                "rgb_matches_initial": (clean["jpeg_decoded_rgb_sha256"]
                                        == initial["jpeg_decoded_rgb_sha256"]),
                "depth_matches_initial": (clean["sensor_audit"]["depth_sha256"]
                                          == initial["sensor_audit"]["depth_sha256"]),
            }
            restored = stage["restored_r2r"]
            require(restored["position_error_m"] < 1e-5 and restored["yaw_error_deg"] < 1e-4,
                    "backend round trip did not restore R2R initial pose")
            require(restored["rgb_matches_initial"] and restored["depth_matches_initial"],
                    "backend round trip did not restore R2R initial RGB/depth")
        stage["ok"] = True
    record["ok"] = True
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-url", required=True)
    parser.add_argument("--dataset", choices=("r2r", "rxr", "both"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--native-reset-audit', type=Path,
                        help='Audit all listed RxR resets and smoke two formerly failed poses')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(V.DATA_ROOT):
        parser.error("--output must be inside the authorized data disk root")
    record = {"schema": "v6_vlnce_http_acceptance_v1", "ok": False,
              "dataset": args.dataset, "bench_url": args.bench_url,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_root": str(V.DEFAULT_SET),
              "manifest_lock_sha256": V.sha256_file(V.DEFAULT_SET / "lock.json")}
    try:
        bench = BenchClient(args.bench_url, timeout=600)
        render = RemoteRenderClient(args.bench_url, timeout=600)
        if args.dataset == "both":
            cases = {dataset: V.load_cases(V.DEFAULT_SET, dataset)
                     for dataset in ("r2r", "rxr")}
            run_backend_switch_smoke(cases, bench, render, record)
        else:
            cases = V.load_cases(V.DEFAULT_SET, args.dataset)
            if args.native_reset_audit:
                require(args.dataset == 'rxr', 'reset audit requires RxR')
                run_reset_audit(cases, bench, render, record, args.native_reset_audit,
                                output.with_suffix('.progress.json'))
            else:
                run_smoke(args.dataset, cases, bench, render, record)
    except Exception as error:
        record.update(error=str(error), traceback=traceback.format_exc())
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        V.atomic_json(output, record)
    print(f"[v6_vlnce_smoke] dataset={args.dataset} ok={record['ok']} output={output}", flush=True)
    if not record["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
