#!/usr/bin/env python3
"""Guarded, single-attempt serial runner for a frozen local comparison.

Default is a read-only launch-gate snapshot. --run must be explicitly requested.
No process is killed and an observation timeout never causes a replacement run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def process_identity(pid, *, proc_root=Path("/proc")):
    try:
        fields = (proc_root / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": int(pid), "start_ticks": fields[19], "state": fields[0]}
    except FileNotFoundError:
        return None


def same_process_alive(expected, *, proc_root=Path("/proc")):
    actual = process_identity(expected["pid"], proc_root=proc_root)
    return bool(actual and actual["start_ticks"] == str(expected["start_ticks"])
                and actual["state"] not in ("Z", "X"))


def verify_package(plan):
    launch = Path(plan["launch_dir"])
    for name, wanted in plan["control_sha256"].items():
        if digest(launch / name) != wanted:
            raise ValueError(f"frozen control file changed: {name}")
    sources = json.loads((launch / "source_sha256.json").read_text())
    for name, record in sources.items():
        if "sha256" not in record:
            continue
        path = (launch / name.removeprefix("@launch/") if name.startswith("@launch/") else
                launch / name if name.startswith("upstream/") else launch / "source" / name)
        if digest(path) != record["sha256"]:
            raise ValueError(f"frozen source/dependency changed: {name}")


def gpu_processes():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=30)
    rows = []
    for line in result.stdout.splitlines():
        fields = [x.strip() for x in line.split(",")]
        if len(fields) != 2 or not fields[0].isdigit() or not fields[1].startswith("GPU-"):
            raise ValueError(f"unrecognized GPU process row: {line!r}")
        rows.append({"pid": int(fields[0]), "gpu_uuid": fields[1]})
    return rows


def verify_normal_final_checkpoint(plan):
    """Audit a stopped baseline's complete window and committed FP32 state.

    Called only after the recorded driver/launcher have exited and GPUs have
    drained, so polling never rereads large rollout files during training.
    Optimizer shards are checked without loading their large tensor payloads.
    """
    run = Path(plan["baseline_run"])
    first, target = plan.get("comparison_steps", [17, 22])
    if (first, target) != (17, 22):
        raise ValueError("normal final gate requires the prespecified 17..22 window")
    contract = json.loads((run / "contract.json").read_text())
    if (contract.get("train_scope") != "full" or contract.get("harness_version") != "v6"
            or contract.get("learner_parameter_dtype") != "fp32"
            or contract.get("learner_world_size") != 12
            or contract.get("episodes_per_step") != 32
            or contract.get("num_groups_per_step") != 8 or contract.get("group_size") != 4
            or contract.get("max_tool_calls") != 50
            or contract.get("multimodal_replay_revision") != "exact_input_ids_mm_token_types_v1"
            or contract.get("tool_credit") is not None
            or contract.get("reward", {}).get("normalization") != "same_case_trajectory_outcome_grpo"):
        raise ValueError("normal final baseline must be the repaired 12-rank episode-GRPO control")
    resume = contract.get("resume", {})
    if (resume.get("start_step") != first - 1 or resume.get("exact_trainer_state") is not True
            or resume.get("source") != plan["initial_checkpoint"]):
        raise ValueError("normal final baseline did not resume the frozen actor/optimizer source")

    def complete_jsonl(path):
        data = path.read_bytes()
        if not data.endswith(b"\n"):
            raise ValueError(f"stopped baseline has an incomplete JSONL tail: {path}")
        return [json.loads(line) for line in data.splitlines() if line.strip()], hashlib.sha256(data).hexdigest()

    metric_rows, metric_sha = complete_jsonl(run / "metrics.jsonl")
    metrics = {}
    for row in metric_rows:
        step = int(row["step"])
        if step in metrics:
            raise ValueError("duplicate baseline optimizer metric")
        update = row.get("updates", {})
        if (row.get("episodes") != 32 or update.get("optimizer_steps") != 1
                or update.get("logical_valid_chunks") != 32
                or not math.isfinite(float(update.get("grad_norm", float("nan"))))):
            raise ValueError("baseline metric lacks its complete finite optimizer update")
        rank_metrics = row.get("timing", {}).get("learner_rank_metrics")
        if (not isinstance(rank_metrics, list) or len(rank_metrics) != 12
                or any(not isinstance(rank, dict) or type(rank.get("rank")) is not int
                       for rank in rank_metrics)
                or {rank["rank"] for rank in rank_metrics} != set(range(12))):
            raise ValueError("baseline metric requires exactly one result from each of its 12 learner ranks")
        if any(rank.get("optimizer_steps") != 1 or rank.get("logical_valid_chunks") != 32
               or not math.isfinite(float(rank.get("grad_norm", float("nan"))))
               for rank in rank_metrics):
            raise ValueError("every baseline learner rank must complete one finite update with denominator 32")
        metrics[step] = row
    expected_steps = set(range(first, target + 1))
    if set(metrics) != expected_steps:
        raise ValueError("normal baseline lacks the exact complete 17..22 optimizer window")
    checkpoint = run / f"global_step_{target:06d}"
    if (metrics[target].get("checkpoint_committed") is not True
            or metrics[target].get("checkpoint", {}).get("checkpoint") != str(checkpoint)):
        raise ValueError("final optimizer metric does not reference the committed step-22 checkpoint")
    rollout_rows, rollout_sha = complete_jsonl(run / "rollouts.jsonl")
    episodes = set()
    for row in rollout_rows:
        key = (int(row["step"]), int(row["group_index"]), int(row["sample_index"]))
        native = row.get("reward_components", {}).get("terminal_success")
        result = row.get("result", {})
        if (key in episodes or result.get("evaluation_valid") is not True
                or not isinstance(native, bool) or result.get("success") is not native
                or not math.isclose(float(row["reward"]), float(native))):
            raise ValueError("baseline contains a duplicate or invalid terminal-reward rollout")
        episodes.add(key)
    grid = {(step, group, sample) for step in expected_steps for group in range(8) for sample in range(4)}
    if episodes != grid:
        raise ValueError("normal baseline lacks all 32 x 6 valid comparison rollouts")
    if int((run / "latest_checkpointed_iteration.txt").read_text()) != target:
        raise ValueError("normal baseline final checkpoint is not committed in the tracker")

    source = Path(plan.get("snapshot_repo", Path(__file__).resolve().parents[2]))
    paths = [str(source / "verl")]
    if plan.get("snapshot_upstream"):
        paths.append(plan["snapshot_upstream"])
    sys.path[:0] = [path for path in paths if path not in sys.path]
    import torch
    from action_head_rl.checkpoint import resolve_resume_checkpoint, _validate_fp32_policy_payload

    resolved = resolve_resume_checkpoint(mode="resume_path", output_dir=run,
        resume_from_path=str(checkpoint), train_scope="full", learner_world_size=12,
        allow_weights_only=False)
    report = json.loads((checkpoint / "report.json").read_text())
    if (not resolved.exact or resolved.step != target
            or report.get("publication_kind") != "checkpoint"
            or report.get("qwen_export_dtype") != "preserve"):
        raise ValueError("normal baseline final publication is not an exact FP32 checkpoint")
    qwen_index_sha = _validate_fp32_policy_payload(resolved)
    ranks = []
    for rank in range(12):
        optim = resolved.trainer_state / f"optim_world_size_12_rank_{rank}.pt"
        extra_path = resolved.trainer_state / f"extra_state_world_size_12_rank_{rank}.pt"
        if optim.stat().st_size <= 0 or extra_path.stat().st_size <= 0:
            raise ValueError("normal baseline has an empty distributed state shard")
        extra = torch.load(extra_path, map_location="cpu", weights_only=False)
        if (extra.get("lr_scheduler", {}).get("last_epoch") != target
                or not {"cpu", "cuda", "numpy", "random"} <= set(extra.get("rng", {}))):
            raise ValueError(f"normal baseline scheduler/RNG rank {rank} is incomplete or at the wrong step")
        ranks.append({"rank": rank, "optimizer_bytes": optim.stat().st_size,
                      "scheduler_last_epoch": target, "rng_keys": sorted(extra["rng"])})
    return {"checkpoint": str(checkpoint), "step": target, "episodes": len(episodes),
            "metrics_sha256": metric_sha, "rollouts_sha256": rollout_sha,
            "checkpoint_report_sha256": digest(checkpoint / "report.json"),
            "qwen_index_sha256": qwen_index_sha, "ranks": ranks}


def gate_snapshot(plan, *, proc_root=Path("/proc"), query_gpu=gpu_processes):
    """Require positive checkpoint proof plus authoritative process teardown."""
    mode = plan.get("completion_mode", "handoff22")
    if mode not in ("handoff22", "normal_final_checkpoint"):
        raise ValueError("unknown baseline completion gate")
    if mode == "normal_final_checkpoint":
        pending = [name for name in ("baseline_driver", "baseline_launcher")
                   if same_process_alive(plan[name], proc_root=proc_root)]
        if pending:
            return {"ready": False, "reason": "waiting_for_actual_process_exit", "processes": pending}
        renderer = plan["renderer"]
        if not same_process_alive(renderer, proc_root=proc_root):
            raise RuntimeError("the specific owned renderer is no longer live")
        rows = query_gpu()
        if rows != [{"pid": renderer["pid"], "gpu_uuid": renderer["gpu_uuid"]}]:
            return {"ready": False, "reason": "waiting_for_gpu_teardown", "gpu_processes": rows}
        try:
            audit = verify_normal_final_checkpoint(plan)
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"stopped baseline lacks valid normal completion evidence: {exc}") from exc
        return {"ready": True, "reason": "baseline_final_checkpoint_and_exited_only_owned_renderer",
                "final_checkpoint_audit": audit, "gpu_processes": rows}
    handoff = Path(plan["handoff_directory"])
    if (handoff / "handoff_failure.json").exists():
        raise RuntimeError("baseline handoff has an explicit failure; manual reconciliation required")
    result_path = handoff / "handoff_result.json"
    if not result_path.exists():
        if not any(same_process_alive(plan[name], proc_root=proc_root)
                   for name in ("baseline_driver", "baseline_launcher")):
            # The supported helper commits this marker before interrupting
            # the source driver. Both recorded identities disappearing first
            # is therefore failure evidence, not an indefinitely pending save.
            raise RuntimeError("baseline driver and launcher exited without a committed handoff result")
        return {"ready": False, "reason": "waiting_for_handoff_result"}
    result = json.loads(result_path.read_text())
    expected_checkpoint = str(handoff / "global_step_000022")
    if (result.get("state") != "checkpoint_saved" or result.get("step") != 22
            or result.get("source") != plan["baseline_run"]
            or result.get("checkpoint") != expected_checkpoint):
        raise ValueError("baseline handoff result does not match the frozen step-22 gate")
    identity = result.get("driver_identity") or {}
    if any(str(identity.get(key)) != str(plan["baseline_driver"][key]) for key in ("pid", "start_ticks")):
        raise ValueError("handoff driver identity differs from the frozen baseline driver")
    checkpoint = Path(expected_checkpoint)
    if (not (checkpoint / "report.json").is_file()
            or not (checkpoint / "trainer_state").is_dir()
            or int((handoff / "latest_checkpointed_iteration.txt").read_text()) != 22):
        raise ValueError("checkpoint_saved marker lacks checkpoint/trainer state/tracker")
    pending = [name for name in ("baseline_driver", "baseline_launcher")
               if same_process_alive(plan[name], proc_root=proc_root)]
    if pending:
        return {"ready": False, "reason": "waiting_for_actual_process_exit", "processes": pending}
    renderer = plan["renderer"]
    if not same_process_alive(renderer, proc_root=proc_root):
        raise RuntimeError("the specific owned renderer is no longer live")
    rows = query_gpu()
    expected = [{"pid": renderer["pid"], "gpu_uuid": renderer["gpu_uuid"]}]
    if rows != expected:
        return {"ready": False, "reason": "waiting_for_gpu_teardown", "gpu_processes": rows}
    return {"ready": True, "reason": "baseline_saved_and_exited_only_owned_renderer",
            "handoff_sha256": digest(result_path), "gpu_processes": rows}


def cache_signature(plan):
    source = Path(plan["autotune_source_root"])
    return [(str(p.relative_to(source)), p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(source.glob("learner_world_12_rank_*/triton/*/*.autotune.json"))]


def frozen_environment(plan, *, validate_only=False):
    env = dict(os.environ)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
                "SAGE_BENCH_PROTOCOL_ROOT", "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
                "MASTER_ADDR", "MASTER_PORT", "ACTION_HEAD_TRITON_AUTOTUNE_SEED"):
        env.pop(key, None)
    env.update(plan["environment"])
    if validate_only:
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        seed = Path(plan["autotune_destination"])
        if (seed / "manifest.json").exists():
            env["ACTION_HEAD_TRITON_AUTOTUNE_SEED"] = str(seed)
        for key in plan["runtime_directory_keys"]:
            Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def exec_trainer(plan, *, validate_only):
    verify_package(plan)
    if not validate_only:
        launch = Path(plan["launch_dir"])
        attempt = json.loads((launch / "candidate_start_attempt.json").read_text())
        if (attempt["plan_sha256"] != digest(launch / "plan.json")
                or os.getppid() != attempt["monitor"]["pid"]
                or not same_process_alive(attempt["monitor"])):
            raise RuntimeError("trainer entry requires its live owning monitor's recorded attempt")
        if not gate_snapshot(plan)["ready"]:
            raise RuntimeError("launch gates changed before direct trainer exec")
    command = plan["trainer_command"] + (["--validate-only"] if validate_only else [])
    os.chdir(plan["snapshot_repo"])
    os.execvpe(command[0], command, frozen_environment(plan, validate_only=validate_only))


def run_monitor(plan):
    launch = Path(plan["launch_dir"])
    for name in ("monitor_process.json", "candidate_start_attempt.json", "candidate_process.json"):
        if (launch / name).exists():
            raise RuntimeError(f"prior monitor/launch record exists ({name}); never auto-restart")
    if Path(plan["candidate_run"]).exists():
        raise RuntimeError("candidate output already exists; never overwrite or auto-restart")
    if Path(plan["comparison_output"]).exists():
        raise RuntimeError("comparison output already exists; do not mix a stale result into this run")
    # O_EXCL covers two simultaneous starts without relying on a stale PID file.
    with (launch / "monitor.lock").open("x") as lock:
        lock.write(str(os.getpid()) + "\n")
    identity = process_identity(os.getpid())
    atomic_json(launch / "monitor_process.json", {**identity, "started_at": time.time(),
                "plan_sha256": digest(launch / "plan.json"), "command": sys.argv})
    verify_package(plan)
    previous_cache = None
    while True:
        try:
            gate = gate_snapshot(plan)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError, json.JSONDecodeError) as exc:
            # Poll failure is not proof that a process/job stopped.
            gate = {"ready": False, "reason": "observation_error_retrying_same_handles", "error": str(exc)}
        atomic_json(launch / "monitor_state.json", {"state": "waiting", "time": time.time(), **gate})
        if gate["ready"]:
            current_cache = cache_signature(plan)
            if previous_cache == current_cache:
                break
            previous_cache = current_cache
        else:
            previous_cache = None
        time.sleep(15)
    # The old cache is now quiescent. Copy tuning decisions only, never binaries
    # or cache-group files containing old absolute paths.
    if previous_cache:
        with (launch / "autotune-seed.log").open("x") as log:
            subprocess.run(plan["autotune_command"], check=True, stdout=log, stderr=subprocess.STDOUT,
                           cwd=plan["snapshot_repo"], env=frozen_environment(plan, validate_only=True))
    else:
        atomic_json(launch / "autotune-seed-report.json", {"seeded": False, "reason": "no_completed_decisions"})
    verify_package(plan)
    if not gate_snapshot(plan)["ready"]:
        raise RuntimeError("baseline/GPU gate changed while preparing seed; no candidate was launched")
    atomic_json(launch / "candidate_start_attempt.json", {"created_at": time.time(), "monitor": identity,
                "plan_sha256": digest(launch / "plan.json"), "single_attempt": True,
                "candidate_run": plan["candidate_run"], "trainer_command": plan["trainer_command"]})
    with (launch / "candidate.log").open("x") as log:
        child = subprocess.Popen(["bash", str(launch / "entry.sh")], stdout=log, stderr=subprocess.STDOUT,
                                 cwd=plan["snapshot_repo"], env=frozen_environment(plan, validate_only=False))
        child_identity = process_identity(child.pid)
        if child_identity is None:
            raise RuntimeError("candidate exited before its process identity could be recorded; do not retry")
        atomic_json(launch / "candidate_process.json", {**child_identity, "monitor": identity,
                    "started_at": time.time(), "command": ["bash", str(launch / "entry.sh")],
                    "effective_trainer_command": plan["trainer_command"]})
        while child.poll() is None:
            atomic_json(launch / "monitor_state.json", {"state": "candidate_running", "time": time.time(),
                        "child": child_identity})
            time.sleep(15)
        code = child.wait()
    with (launch / "comparison.log").open("x") as log:
        comparison = subprocess.run(plan["comparison_command"], stdout=log, stderr=subprocess.STDOUT,
                                    cwd=plan["snapshot_repo"], env=frozen_environment(plan, validate_only=True))
    atomic_json(launch / "monitor_state.json", {"state": "candidate_ended", "time": time.time(),
                "candidate_exit_code": code, "comparison_exit_code": comparison.returncode,
                "complete_window_compared": comparison.returncode == 0,
                "comparison_result": plan["comparison_output"] if comparison.returncode == 0 else None,
                "note": "A successful comparison is not itself a positive RL benefit conclusion."})
    return code or comparison.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--exec-trainer", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--validate-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan.get("schema") not in ("v6-local-credit-frozen-launch-v1", "v6-local-trial-frozen-launch-v1"):
        raise ValueError("wrong frozen launch schema")
    if args.exec_trainer:
        exec_trainer(plan, validate_only=args.validate_only)
    elif args.run:
        try:
            return run_monitor(plan)
        except Exception as exc:
            launch = Path(plan["launch_dir"])
            owner_path = launch / "monitor_process.json"
            owner = json.loads(owner_path.read_text()) if owner_path.exists() else None
            failure = (f"rejected_monitor_attempt_{os.getpid()}.json"
                       if owner and owner["pid"] != os.getpid() else "monitor_failure.json")
            atomic_json(launch / failure,
                        {"time": time.time(), "error": str(exc), "no_automatic_retry": True})
            raise
    else:
        verify_package(plan)
        print(json.dumps(gate_snapshot(plan), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
