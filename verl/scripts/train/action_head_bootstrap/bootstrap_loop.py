#!/usr/bin/env python3
"""Run cumulative v3 bootstrap rounds on one or more 16-GPU DLC workers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path


REPO = Path("/workspace/ZeroVLN")
HERE = REPO / "verl/scripts/train/action_head_bootstrap"
TRAIN_ENTRY = REPO / "scripts/train/action_head_dualvln_sft/entry.sh"
PYTHON = Path("/usr/local/bin/python3.12")
STEP_RE = re.compile(r"global_step_(\d+)$")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_step(path: Path, version: str = "v3") -> int:
    report_path = path / "report.json"
    if not report_path.is_file():
        raise ValueError(f"checkpoint has no report.json: {path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("ok") or report.get("action_head_as_tool_version") != version:
        raise ValueError(f"not a successful {version} checkpoint: {path}")
    match = STEP_RE.fullmatch(path.name)
    step = int(report.get("global_step", -1))
    if not match or int(match.group(1)) != step:
        raise ValueError(f"checkpoint directory/report step mismatch: {path}")
    for name in ("action_head", "dualvln_query"):
        if not (path / name).is_dir():
            raise ValueError(f"checkpoint component missing: {path / name}")
    return step


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("[bootstrap-loop] run:", " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)


def resolve_node_rank(nodes: int) -> int:
    for name in ("RANK", "PET_NODE_RANK", "NODE_RANK"):
        value = os.environ.get(name, "")
        if value.isdigit():
            rank = int(value)
            break
    else:
        hostname = socket.gethostname()
        if "-master-" in hostname:
            rank = 0
        else:
            match = re.search(r"-worker-(\d+)$", hostname)
            if not match:
                if nodes == 1:
                    return 0
                raise RuntimeError(f"cannot derive node rank from hostname={hostname}")
            rank = int(match.group(1)) + 1
    if not 0 <= rank < nodes:
        raise ValueError(f"node rank {rank} is outside configured nodes={nodes}")
    return rank


def barrier(root: Path, name: str, rank: int, nodes: int,
            timeout_s: int) -> None:
    # A resubmitted DLC job may intentionally reuse completed round artifacts,
    # but it must never consume another job's stale rendezvous files.
    run_id = os.environ.get("DLC_JOB_ID") or f"local-{socket.gethostname()}-{os.getpid()}"
    directory = root / "_barriers" / run_id / name
    atomic_json(directory / f"node_{rank:03d}.json", {
        "node_rank": rank, "nodes": nodes,
        "run_id": run_id,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    deadline = time.monotonic() + timeout_s
    while True:
        ready = list(directory.glob("node_*.json"))
        if len(ready) == nodes:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"barrier {name} timed out: {len(ready)}/{nodes} nodes")
        time.sleep(10)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-head-as-tool-version", choices=("v3",), required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--start-round", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--barrier-timeout", type=int, default=86400)
    parser.add_argument("--loop-root", required=True, type=Path)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--initial-checkpoint", required=True, type=Path)
    parser.add_argument("--backbone", required=True, type=Path)
    parser.add_argument("--slow-model", type=Path,
                        default=Path("/opt/external/user-e7a01947/Model/Qwen3.6-35B-A3B"))
    parser.add_argument("--steps-per-round", type=int, default=5000)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-cases", type=int, default=5203)
    parser.add_argument("--rollout-actors", type=int, default=160)
    parser.add_argument("--max-correction-anchors", type=int, default=1000)
    parser.add_argument("--build-lanes", type=int, default=8)
    parser.add_argument("--renderer-cache-scenes", type=int, default=64)
    parser.add_argument("--ah-max-batch-size", type=int, default=32)
    parser.add_argument("--ah-batch-wait-ms", type=int, default=50)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-tool-calls", type=int, default=12)
    parser.add_argument("--anchors-per-gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    if (args.rounds <= 0 or args.start_round <= 0 or args.nodes <= 0 or
            args.steps_per_round <= 0 or args.barrier_timeout <= 0):
        parser.error("rounds, start-round, nodes, steps and barrier timeout must be positive")
    if not 1 <= args.build_lanes <= 8:
        parser.error("build-lanes must be in [1, 8]")
    if args.rollout_actors <= 0 or args.max_correction_anchors <= 0:
        parser.error("rollout-actors and max-correction-anchors must be positive")
    if args.max_cases and args.max_cases < args.nodes:
        parser.error("nonzero max-cases must be at least the node count")
    for path in (args.base_manifest, args.initial_checkpoint,
                 args.backbone, args.slow_model):
        if not path.exists():
            parser.error(f"required path does not exist: {path}")
    return args


def main() -> None:
    args = arguments()
    node_rank = resolve_node_rank(args.nodes)
    injected_world = os.environ.get("WORLD_SIZE", "")
    if injected_world.isdigit() and int(injected_world) != args.nodes:
        raise ValueError(
            f"DLC WORLD_SIZE={injected_world} disagrees with --nodes={args.nodes}")
    if args.nodes > 1 and os.environ.get("MASTER_ADDR", "") in ("", "127.0.0.1", "localhost"):
        raise RuntimeError("multi-node loop requires DLC MASTER_ADDR")
    required_entries = (
        REPO / "scripts/train/action_head_bootstrap/v3_collect_entry.sh",
        HERE / "merge_v3_corrections.py",
        REPO / "verl/sft/collect_noncollision_option_corrections.py",
        REPO / "verl/sft/mix_action_head_bootstrap.py",
        TRAIN_ENTRY,
    )
    missing_entries = [str(path) for path in required_entries if not path.is_file()]
    if missing_entries:
        raise FileNotFoundError(f"bootstrap child entries missing: {missing_entries}")
    root = args.loop_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    current_manifest = args.base_manifest.resolve()
    current_checkpoint = args.initial_checkpoint.resolve()
    current_step = checkpoint_step(current_checkpoint)
    completed: list[dict] = []

    config = vars(args).copy()
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in config.items()}
    config_payload = {
        "schema": "action_head_bootstrap_loop_config_v1", **config,
    }
    config_path = root / "loop_config.json"
    if node_rank == 0:
        if config_path.is_file():
            existing_config = json.loads(config_path.read_text(encoding="utf-8"))
            comparable = {key: value for key, value in existing_config.items()
                          if key != "created_at"}
            if comparable != config_payload:
                raise ValueError(
                    f"loop configuration changed for existing root: {config_path}")
        else:
            atomic_json(config_path, {
                **config_payload,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })
    barrier(root, "config", node_rank, args.nodes, args.barrier_timeout)

    round_indices = range(args.start_round, args.start_round + args.rounds)
    for round_index in round_indices:
        round_dir = root / f"round_{round_index:02d}"
        collection = round_dir / "collection"
        node_collection = collection / "nodes" / f"node_{node_rank:03d}"
        mixed = round_dir / "mixed"
        training = round_dir / "training"
        expected_step = current_step + args.steps_per_round
        final_checkpoint = training / f"global_step_{expected_step:06d}"
        round_done = round_dir / "ROUND_DONE.json"
        plan_path = round_dir / "ROUND_PLAN.json"
        if node_rank == 0:
            round_plan = {
                "schema": "action_head_bootstrap_round_plan_v2",
                "bootstrap_round": round_index, "nodes": args.nodes,
                "input_manifest": str(current_manifest),
                "input_manifest_sha256": sha256(current_manifest),
                "input_checkpoint": str(current_checkpoint), "input_step": current_step,
                "expected_output_step": expected_step,
            }
            if plan_path.is_file():
                if json.loads(plan_path.read_text(encoding="utf-8")) != round_plan:
                    raise ValueError(f"round inputs changed for existing output: {plan_path}")
            else:
                atomic_json(plan_path, round_plan)
        barrier(root, f"round_{round_index:03d}_plan", node_rank,
                args.nodes, args.barrier_timeout)

        if round_done.is_file():
            prior = json.loads(round_done.read_text(encoding="utf-8"))
            current_manifest = Path(prior["output_manifest"])
            current_checkpoint = Path(prior["output_checkpoint"])
            current_step = checkpoint_step(current_checkpoint)
            completed.append(prior)
            print(f"[bootstrap-loop] resume: round {round_index} already complete", flush=True)
            continue

        if not (node_collection / "DONE").is_file():
            env = os.environ.copy()
            env.update({
                "OUTPUT_DIR": str(node_collection),
                "MANIFEST": str(current_manifest),
                "CHECKPOINT_DIR": str(current_checkpoint),
                "BACKBONE": str(args.backbone),
                "MAX_CASES": str(args.max_cases),
                "ROLLOUT_ACTORS": str(args.rollout_actors),
                "MAX_CORRECTION_ANCHORS": str(args.max_correction_anchors),
                "BUILD_LANES": str(args.build_lanes),
                "RENDERER_CACHE_SCENES": str(args.renderer_cache_scenes),
                "AH_MAX_BATCH_SIZE": str(args.ah_max_batch_size),
                "AH_BATCH_WAIT_MS": str(args.ah_batch_wait_ms),
                "MAX_TURNS": str(args.max_turns),
                "MAX_TOOL_CALLS": str(args.max_tool_calls),
                "CASE_SEED": str(args.seed),
                "BOOTSTRAP_NODE_INDEX": str(node_rank),
                "BOOTSTRAP_NODE_COUNT": str(args.nodes),
            })
            run(["bash", str(REPO / "scripts/train/action_head_bootstrap/v3_collect_entry.sh"),
                 "--action-head-as-tool-version", args.action_head_as_tool_version,
                 "--bootstrap-round", str(round_index)], env=env)

        barrier(root, f"round_{round_index:03d}_collection", node_rank,
                args.nodes, args.barrier_timeout)

        global_corrections = collection / "corrections"
        if node_rank == 0 and not (global_corrections / "DONE").is_file():
            command = [
                str(PYTHON), str(HERE / "merge_v3_corrections.py"),
                "--output-dir", str(global_corrections),
                "--max-correction-anchors", str(args.max_correction_anchors),
                "--bootstrap-round", str(round_index),
            ]
            for index in range(args.nodes):
                command.extend([
                    "--input-dir",
                    str(collection / "nodes" / f"node_{index:03d}" / "corrections")])
            run(command)
        barrier(root, f"round_{round_index:03d}_merge", node_rank,
                args.nodes, args.barrier_timeout)

        correction_manifest = global_corrections / "manifest.jsonl"
        correction_samples = count_jsonl(correction_manifest)
        correction_report = json.loads(
            (collection / "corrections/provenance.json").read_text(encoding="utf-8"))
        if int(correction_report.get("bootstrap_round", -1)) != round_index:
            raise ValueError(f"collection round identity mismatch: {collection}")
        correction_anchors = int(correction_report["anchors_selected"])
        if correction_samples == 0:
            state = {
                "schema": "action_head_bootstrap_loop_state_v2", "ok": True,
                "status": "converged_no_corrections", "requested_rounds": args.rounds,
                "start_round": args.start_round, "nodes": args.nodes,
                "completed_rounds": len(completed),
                "stopped_at_round": round_index,
                "final_manifest": str(current_manifest),
                "final_checkpoint": str(current_checkpoint), "final_step": current_step,
            }
            if node_rank == 0:
                atomic_json(root / "LOOP_COMPLETE.json", state)
                print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
            barrier(root, f"round_{round_index:03d}_no_corrections", node_rank,
                    args.nodes, args.barrier_timeout)
            return

        if node_rank == 0 and not (mixed / "DONE").is_file():
            run([str(PYTHON), str(REPO / "verl/sft/mix_action_head_bootstrap.py"),
                 "--golden-manifest", str(current_manifest),
                 "--recovery-manifest", str(correction_manifest),
                 "--output-dir", str(mixed), "--seed", str(args.seed + round_index - 1),
                 "--action-head-as-tool-version", args.action_head_as_tool_version,
                 "--bootstrap-round", str(round_index)])
        barrier(root, f"round_{round_index:03d}_mix", node_rank,
                args.nodes, args.barrier_timeout)
        next_manifest = mixed / "manifest.jsonl"

        if not final_checkpoint.is_dir():
            if node_rank == 0 and training.exists() and any(training.iterdir()):
                raise RuntimeError(
                    f"incomplete training output exists at {training}; use a new loop root "
                    "or remove only this incomplete round after inspection")
            barrier(root, f"round_{round_index:03d}_train_ready", node_rank,
                    args.nodes, args.barrier_timeout)
            env = os.environ.copy()
            training_env = {
                "NNODES": str(args.nodes), "NODE_RANK": str(node_rank),
                "NPROC_PER_NODE": "16",
                "GATE": "train", "MANIFEST": str(next_manifest),
                "SLOW_MODEL": str(args.slow_model),
                "HEAD_CHECKPOINT": str(current_checkpoint / "action_head"),
                "QUERY_CHECKPOINT": str(current_checkpoint / "dualvln_query"),
                "ACTION_HEAD_WARM_START_CHECKPOINT": str(current_checkpoint),
                "ACTION_HEAD_STEP_OFFSET": str(current_step),
                "OUTPUT_DIR": str(training), "MAX_STEPS": str(args.steps_per_round),
                "SAVE_FREQ": str(args.save_freq), "VAL_FREQ": "0",
                "LEARNING_RATE": str(args.learning_rate),
                "ACTION_HEAD_ANCHORS_PER_GPU": str(args.anchors_per_gpu),
            }
            if args.nodes == 1:
                training_env["MASTER_ADDR"] = "127.0.0.1"
            env.update(training_env)
            run(["bash", str(TRAIN_ENTRY), "--action-head-as-tool-version",
                 args.action_head_as_tool_version], env=env)
        barrier(root, f"round_{round_index:03d}_training", node_rank,
                args.nodes, args.barrier_timeout)

        if node_rank == 0:
            actual_step = checkpoint_step(final_checkpoint)
            if actual_step != expected_step:
                raise RuntimeError(f"expected step {expected_step}, got {actual_step}")
            mix_report = json.loads((mixed / "provenance.json").read_text(encoding="utf-8"))
            if int(mix_report.get("bootstrap_round", -1)) != round_index:
                raise ValueError(f"mix round identity mismatch: {mixed}")
            round_report = {
                "schema": "action_head_bootstrap_round_v2", "ok": True,
                "bootstrap_round": round_index, "nodes": args.nodes,
                "input_manifest": str(current_manifest),
                "input_checkpoint": str(current_checkpoint), "input_step": current_step,
                "correction_anchors": correction_anchors,
                "correction_samples": correction_samples,
                "output_manifest": str(next_manifest.resolve()),
                "aggregate_samples": int(mix_report["samples"]),
                "aggregate_anchors": int(mix_report["anchors"]),
                "output_checkpoint": str(final_checkpoint.resolve()),
                "output_step": actual_step,
            }
            atomic_json(round_done, round_report)
        barrier(root, f"round_{round_index:03d}_finalize", node_rank,
                args.nodes, args.barrier_timeout)
        round_report = json.loads(round_done.read_text(encoding="utf-8"))
        actual_step = checkpoint_step(Path(round_report["output_checkpoint"]))
        completed.append(round_report)
        current_manifest, current_checkpoint, current_step = (
            next_manifest.resolve(), final_checkpoint.resolve(), actual_step)
        if node_rank == 0:
            atomic_json(root / "LOOP_STATE.json", {
                "schema": "action_head_bootstrap_loop_state_v2", "ok": True,
                "status": "running", "requested_rounds": args.rounds,
                "start_round": args.start_round, "nodes": args.nodes,
                "completed_rounds": len(completed), "rounds": completed,
                "current_manifest": str(current_manifest),
                "current_checkpoint": str(current_checkpoint), "current_step": current_step,
            })

    state = {
        "schema": "action_head_bootstrap_loop_state_v2", "ok": True,
        "status": "complete", "requested_rounds": args.rounds,
        "start_round": args.start_round,
        "completed_rounds": len(completed), "rounds": completed,
        "final_manifest": str(current_manifest),
        "final_checkpoint": str(current_checkpoint), "final_step": current_step,
    }
    if node_rank == 0:
        state["nodes"] = args.nodes
        atomic_json(root / "LOOP_COMPLETE.json", state)
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    barrier(root, "loop_complete", node_rank, args.nodes, args.barrier_timeout)


if __name__ == "__main__":
    main()
