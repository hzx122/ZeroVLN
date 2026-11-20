#!/usr/bin/env python3
"""Consume configured complete snapshots from a bound upstream DLC job."""
from __future__ import annotations

import argparse
import atexit
import errno
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO / "verl"), str(REPO / "verl" / "dlc"),
                "/workspace/dlc_cli"]

from common import atomic_json, checkpoint_identity, sha256_file  # noqa: E402
from action_head import protocol as ACTION_HEAD_PROTOCOL  # noqa: E402
from action_head import protocol_v3 as ACTION_HEAD_PROTOCOL_V3  # noqa: E402
from sage.goal_region import official_success_contract_identity  # noqa: E402


TERMINAL_SUCCESS = {"Succeeded", "Completed"}
TERMINAL_FAILURE = {"Failed", "Stopped", "Cancelled"}
DEFAULT_IDLE_OCCUPIER = (
    REPO / "verl/scripts/train/action_head_sft_data/gpu_occupier.py")


def checkpoint_index_steps(index_path: Path, checkpoint_root: Path) -> tuple[int, ...]:
    """Load the producer-published, atomic list of completed round checkpoints."""
    for attempt in range(5):
        try:
            return _checkpoint_index_steps_once(index_path, checkpoint_root)
        except OSError as exc:
            if exc.errno != errno.ESTALE or attempt == 4:
                raise
            delay = 2 ** attempt
            print(f"[watcher] stale checkpoint index; reopen in {delay}s: "
                  f"{index_path} (attempt {attempt + 1}/5)", flush=True)
            time.sleep(delay)
    raise AssertionError("unreachable")


def _checkpoint_index_steps_once(index_path: Path, checkpoint_root: Path) -> tuple[int, ...]:
    # Each attempt opens the path again; never reuse a stale file descriptor.
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if not payload.get("ok"):
        raise ValueError(f"checkpoint index is not valid: {index_path}")
    published_root = Path(payload["checkpoint_root"]).resolve()
    if published_root != checkpoint_root.resolve():
        raise ValueError(
            "checkpoint index root mismatch: "
            f"published={published_root} requested={checkpoint_root.resolve()}")
    steps = tuple(int(step) for step in payload.get("steps", []))
    if any(step <= 0 for step in steps) or len(steps) != len(set(steps)):
        raise ValueError(f"checkpoint index has invalid steps: {steps}")
    if steps != tuple(sorted(steps)):
        raise ValueError(f"checkpoint index steps are not sorted: {steps}")
    expected_paths = tuple(
        str((checkpoint_root / f"global_step_{step:06d}").resolve())
        for step in steps)
    published_paths = tuple(
        str(Path(path).resolve()) for path in payload.get("checkpoints", []))
    if published_paths != expected_paths:
        raise ValueError("checkpoint index paths do not match its steps")
    return steps


def upstream_status(job_id: str) -> dict:
    snapshot_path = os.environ.get("UPSTREAM_STATUS_SNAPSHOT")
    if snapshot_path:
        from upstream_status_bridge import read_snapshot
        snapshot = read_snapshot(snapshot_path, job_id)
        if snapshot:
            return snapshot
    try:
        from _dlc_common import get_dlc_client
        from alibabacloud_pai_dlc20201203.models import GetJobRequest
        body = get_dlc_client().get_job(job_id, GetJobRequest()).body.to_map()
        return {"job_id": job_id, "display_name": body.get("DisplayName"),
                "status": body.get("Status"), "reason_code": body.get("ReasonCode"),
                "reason_message": body.get("ReasonMessage")}
    except Exception as exc:
        return {"job_id": job_id, "status": "Unknown", "error": repr(exc)}


def run(command: list[str], *, env: dict | None = None, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as sink:
        sink.write("\n[command] " + " ".join(command) + "\n")
        sink.flush()
        subprocess.run(command, check=True, env=env, stdout=sink,
                       stderr=subprocess.STDOUT)


def valid_closedloop_summary(path: Path, repeats: int) -> bool:
    if not path.is_file():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    contract = official_success_contract_identity()
    return (bool(summary.get("evaluation_valid")) and
            summary.get("invalid_tool_call_policy", "terminate") ==
            os.environ.get("ACTION_HEAD_INVALID_TOOL_CALL_POLICY", "feedback") and
            int(summary.get("repeats_per_case", 1)) == repeats and
            (summary.get("success_contract") or {}).get(
                "contract_sha256") == contract["contract_sha256"] and
            int(summary.get("success_contract_mismatches", -1)) == 0 and
            "mean_at_k" in summary and
            ("pass_at_k" in summary if os.environ.get("COMPANION_REPORT_PASS_AT_K", "1") == "1"
             else "pass_at_k" not in summary))


def _tool_execution_contract(value):
    """Ignore prompt-only descriptions, retaining every executable schema field."""
    if isinstance(value, dict):
        return {key: _tool_execution_contract(item) for key, item in value.items()
                if key != "description"}
    if isinstance(value, (list, tuple)):
        return [_tool_execution_contract(item) for item in value]
    return value


def assert_checkpoint_protocol(report: dict, requested_version: str, *,
                               allow_legacy_v43_prompt: bool = False) -> None:
    """Accept exact trained protocol or a legacy v3 checkpoint under v4 harness."""
    protocol = ACTION_HEAD_PROTOCOL.protocol_module(requested_version)
    if protocol is None:
        return
    recorded_version = report.get("action_head_as_tool_version")
    if (allow_legacy_v43_prompt and requested_version in ("v4.11", "v6")
            and recorded_version == "v4.3"):
        from action_head import protocol_v4_3_legacy as legacy
        legacy.assert_protocol_identity(report)
        if _tool_execution_contract(protocol.TOOLS) != _tool_execution_contract(legacy.TOOLS):
            raise ValueError("legacy v4.3 prompt transfer requires unchanged tools")
        return
    if recorded_version == requested_version:
        if (allow_legacy_v43_prompt and requested_version == "v4.3" and
                protocol.PROMPT_VERSION in ("slow_system_prompt_v4_3_r2", "slow_system_prompt_v4_3_r3",
                                            "slow_system_prompt_v4_3_react") and
                report.get("prompt_sha256") != protocol.PROMPT_SHA256):
            # Explicit inference-only prompt transfer. Never relabel old data
            # or relax the protocol's own strict manifest identity checks.
            from action_head import protocol_v4_3_legacy as legacy
            legacy.assert_protocol_identity(report)
            if _tool_execution_contract(protocol.TOOLS) != _tool_execution_contract(legacy.TOOLS):
                raise ValueError("legacy v4.3 prompt transfer requires unchanged tools")
            return
        protocol.assert_protocol_identity(report)
        return
    if (requested_version.startswith("v4") and
            recorded_version == ACTION_HEAD_PROTOCOL_V3.ACTION_HEAD_AS_TOOL_VERSION):
        ACTION_HEAD_PROTOCOL_V3.assert_protocol_identity(report)
        return
    raise ValueError(
        "checkpoint protocol does not match requested evaluation harness: "
        f"checkpoint={recorded_version!r} requested={requested_version!r}")


def step_complete(step_output: Path, repeats: int) -> bool:
    d2_data = os.environ.get("CAST_D2_EVAL_DATA", "")
    if d2_data:
        try:
            d2_result = json.loads((step_output / "cast_d2/completed.json").read_text())
            if (not d2_result.get("ok") or d2_result.get("cases_sha256") !=
                    sha256_file(Path(d2_data) / "eval_after/cases.json")):
                return False
        except (OSError, ValueError):
            return False
    cast_data = os.environ.get("CAST_EVAL_DATA", "")
    if cast_data:
        try:
            cast_result = json.loads((step_output / "cast/completed.json").read_text())
            if (not cast_result.get("ok") or cast_result.get("cases_sha256") !=
                    sha256_file(Path(cast_data) / "eval_after/cases.json")):
                return False
        except (OSError, ValueError):
            return False
    return ((step_output / "DONE").is_file() and
            valid_closedloop_summary(
                step_output / "closedloop" / "summary.json", repeats))


def process_group_members(pgid: int) -> list[tuple[int, str]]:
    """Return live (including zombie) processes in one process group."""
    result = subprocess.run(
        ["ps", "-eo", "pid=,pgid=,stat="], check=False,
        text=True, capture_output=True)
    members = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[1].isdigit() and int(fields[1]) == pgid:
            members.append((int(fields[0]), fields[2]))
    return members


def stop_vllm(pid_file: Path) -> None:
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text().strip())
    except Exception:
        return
    # The PID is also the process-group ID because the launcher uses setsid.
    # Waiting only for the group leader is insufficient: vLLM workers can keep
    # CUDA contexts alive after the shell/API process exits.
    subprocess.run(["kill", "-TERM", "--", f"-{pid}"], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        live = [(member, stat) for member, stat in process_group_members(pid)
                if not stat.startswith("Z")]
        if not live:
            break
        time.sleep(1)
    else:
        subprocess.run(["kill", "-KILL", "--", f"-{pid}"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(30):
            live = [(member, stat) for member, stat in process_group_members(pid)
                    if not stat.startswith("Z")]
            if not live:
                break
            time.sleep(1)
        else:
            raise RuntimeError(f"vLLM process group {pid} survived SIGKILL: {live}")
    pid_file.unlink(missing_ok=True)


def stop_process_group(process: subprocess.Popen | None, timeout: float = 30.0) -> None:
    """Stop a child session and every descendant holding a CUDA context."""
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def start_idle_gpu_occupier(args, output_root: Path) -> subprocess.Popen:
    """Occupy every GPU only while the watcher has no checkpoint to evaluate."""
    script = Path(args.idle_occupier_script)
    if not script.is_file():
        raise FileNotFoundError(f"idle GPU occupier missing: {script}")
    log = output_root / "logs" / "idle_gpu_occupier.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as sink:
        sink.write(f"\n[start] {time.strftime('%Y-%m-%dT%H:%M:%S%z')} all GPUs\n")
        sink.flush()
        process = subprocess.Popen(
            [args.idle_occupier_python, str(script), "--auto"],
            stdout=sink, stderr=subprocess.STDOUT, start_new_session=True)
    atomic_json(output_root / "idle_gpu_occupier_status.json", {
        "active": True, "pid": process.pid, "gpus": "all",
        "reason": "waiting_for_checkpoint",
    })
    return process


def stable_checkpoint(path: Path, seconds: int) -> dict:
    first = checkpoint_identity(path)
    condition_path = (path / "context_aggregator.pt" if first.get("condition_mode") == "v5_native"
                      else path / "dualvln_query" / "dualvln_query_embeddings.pt")
    files = (path / "report.json", path / "action_head" / "fast_slow_action_head.pt", condition_path)
    sizes = {str(file): file.stat().st_size for file in files}
    time.sleep(seconds)
    second = checkpoint_identity(path)
    sizes2 = {str(file): file.stat().st_size for file in files}
    if first != second or sizes != sizes2:
        raise RuntimeError(f"checkpoint changed during readiness check: {path}")
    return second


def evaluate_one(args, checkpoint: Path, step_output: Path) -> None:
    identity = stable_checkpoint(checkpoint, args.stability_seconds)
    head_config = json.loads((
        checkpoint / "action_head" / "fast_slow_action_head_config.json"
    ).read_text(encoding="utf-8"))
    bootstrap_layout = os.environ.get("EVAL_LAYOUT", "bootstrap") == "bootstrap"
    native = identity.get("condition_mode") == "v5_native"
    if native and not bootstrap_layout:
        raise ValueError("native companion requires bootstrap layout")
    if not native and head_config.get("condition_source", "query") not in ("query", "query_hidden"):
        raise ValueError("query companion requires a query-conditioned checkpoint; "
                         "native-context v5 uses its separate evaluator")
    expected_query_shape = [16, 384] if native else [
        int(head_config["query_num_queries"]),
        int(head_config["qwen_hidden_size"]),
    ]
    protocol = ACTION_HEAD_PROTOCOL.protocol_module(args.action_head_as_tool_version)
    if protocol is not None:
        report = json.loads((checkpoint / "report.json").read_text(encoding="utf-8"))
        rl_root = os.environ.get("COMPANION_RL_TRAIN_ROOT")
        rl_checkpoint = report.get("schema") == "action_head_ray_grpo_checkpoint"
        if rl_checkpoint:
            if (args.action_head_as_tool_version != "v6" or not rl_root
                    or checkpoint.parent.resolve() != Path(rl_root).resolve()
                    or os.environ.get("COMPANION_USE_CHECKPOINT_QWEN") != "1"):
                raise ValueError("RL evaluation requires the bound v6 training root and its Qwen")
            training = json.loads((checkpoint.parent / "contract.json").read_text())
            if (training.get("harness_version") != "v6" or
                    training.get("train_scope") != "full" or
                    training.get("vllm_query_tail") is not True):
                raise ValueError("RL training contract is not a full v6 query policy")
        allow_legacy_v43_prompt = os.environ.get("ALLOW_V43_LEGACY_CHECKPOINT_PROMPT") == "1"
        if allow_legacy_v43_prompt and not args.closedloop_only:
            raise ValueError("legacy v4.3 prompt transfer is closedloop-only; old manifests stay strict")
        if not rl_checkpoint:
            assert_checkpoint_protocol(report, args.action_head_as_tool_version,
                                       allow_legacy_v43_prompt=allow_legacy_v43_prompt)
        if allow_legacy_v43_prompt:
            atomic_json(step_output / "checkpoint_prompt_transfer.json", {
                "schema": "explicit-closedloop-prompt-transfer-v1",
                "checkpoint_report": str(checkpoint / "report.json"),
                "training_protocol": {key: report.get(key) for key in protocol.protocol_identity()},
                "evaluation_protocol": protocol.protocol_identity(),
                "checkpoint_weights_modified": False,
                "openloop_manifest_relabelled": False,
            })
    atomic_json(step_output / "checkpoint_identity.json", identity)
    log_dir = step_output / "logs"
    open_summary = step_output / "openloop" / "summary.json"
    if (not args.closedloop_only and
            (not open_summary.exists() or
             not json.loads(open_summary.read_text()).get("evaluation_valid"))):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = args.openloop_gpus
        env["PYTHONPATH"] = f"{REPO}:{REPO / 'verl'}:{env.get('PYTHONPATH', '')}"
        run([
            args.python, str(HERE / "openloop_eval.py"),
            "--manifest", args.sft_manifest,
            "--checkpoint-dir", str(checkpoint),
            "--slow-model", args.slow_model,
            "--output", str(step_output / "openloop"),
            "--num-random", str(args.num_random),
            "--max-anchors", str(args.max_openloop_anchors),
            "--action-head-as-tool-version", args.action_head_as_tool_version,
        ], env=env, log=log_dir / "openloop.log")

    runtime_root = Path(os.environ.get("TMPDIR", "/tmp")) / "companion_eval" / step_output.name
    query_cache = runtime_root / "query_cache"
    vllm_log = log_dir / "vllm_28899.log"
    # One persistent PID file spans checkpoints, so a new launch can always
    # clean a previous service even if the prior checkpoint exited abnormally.
    pid_file = Path(args.output_root) / "active_vllm_28899.pid"
    launch_env = dict(os.environ)
    launch_env.update({
        "CHECKPOINT_DIR": str(checkpoint),
        "QUERY_CACHE": str(query_cache),
        "VLLM_LOG": str(vllm_log),
        "VLLM_PID_FILE": str(pid_file),
    })
    if not bootstrap_layout:
        run(["bash", str(REPO / "scripts/eval/qwen36_4query_ckpt_sweep/launch_vllm_for_ckpt.sh")],
            env=launch_env, log=log_dir / "vllm_launcher.log")
    try:
        closed_summary = step_output / "closedloop" / "summary.json"
        resumed_closed_summary = valid_closedloop_summary(closed_summary, args.closedloop_repeats)
        if not resumed_closed_summary:
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = args.closedloop_gpus
            env["PYTHONPATH"] = (
                f"{REPO}:{REPO / 'verl'}:{REPO / 'vln_demo/src'}:{env.get('PYTHONPATH', '')}")
            closedloop_command = [
                args.python, str(HERE / "closedloop_eval.py"), "run",
                "--checkpoint-dir", str(checkpoint),
                "--subset-manifest", args.subset_manifest,
                "--backbone", args.backbone,
                "--query-cache", str(query_cache),
                "--bench-url", args.bench_url,
                "--vllm-url", "http://127.0.0.1:28899",
                "--output", str(step_output / "closedloop"),
                "--max-turns", str(args.max_turns),
                "--max-tool-calls", str(args.max_tool_calls),
                "--limit-cases", str(args.limit_closedloop_cases),
                "--repeats", str(args.closedloop_repeats),
                "--action-head-as-tool-version", args.action_head_as_tool_version,
            ]
            if args.allow_partial_closedloop:
                closedloop_command.append("--allow-partial")
            if os.environ.get("COMPANION_REPORT_PASS_AT_K") == "0":
                if args.closedloop_repeats != 1:
                    raise ValueError("no pass@k report requires one rollout per case")
                closedloop_command.append("--no-pass-at-k")
            if bootstrap_layout:
                env.pop("CUDA_VISIBLE_DEVICES", None)
                env.update({
                    "COMPANION_EVAL": "1", "ROLLOUT_ONLY": "1", "COLLECTION_PHASE": "all",
                    "COMPANION_CONDITION_MODE": "v5_native" if native else "query",
                    "OUTPUT_DIR": str(step_output / "bootstrap_runtime"),
                    "EVAL_OUTPUT": str(step_output / "closedloop"),
                    "MANIFEST": args.sft_manifest, "SUBSET_MANIFEST": args.subset_manifest,
                    "CHECKPOINT_DIR": str(checkpoint), "BACKBONE": args.backbone,
                    "NODE_CACHE_ROOT": str(runtime_root),
                    "VLLM_COMPILE_CACHE_ROOT": str(runtime_root.parent / "vllm_compile"),
                    "SAGE_ASSET_SET": os.environ.get("CLOSEDLOOP_ASSET_SET", "test"),
                    "ROLLOUT_ACTORS": "160", "VLLM_REPLICAS": "6",
                    "COLLECTION_GPU_START": "0", "COLLECTION_GPU_COUNT": "16",
                    "RENDERER_CACHE_SCENES": "128", "CASE_SCHEDULING": "scene",
                    "ROLLOUT_LAZY_EPISODES": "1", "MAX_TURNS": str(args.max_turns),
                    "MAX_TOOL_CALLS": str(args.max_tool_calls),
                    "CLOSEDLOOP_REPEATS": str(args.closedloop_repeats),
                    "LIMIT_CLOSEDLOOP_CASES": str(args.limit_closedloop_cases),
                    "ALLOW_PARTIAL_CLOSEDLOOP": str(int(args.allow_partial_closedloop)),
                })
                closedloop_command = ["bash", str(REPO / "scripts/train/action_head_bootstrap/v3_collect_entry.sh"),
                                      "--action-head-as-tool-version", args.action_head_as_tool_version,
                                      "--bootstrap-round", "1"]
            run(closedloop_command, env=env, log=log_dir / "closedloop.log")
        rollout_paths = list((step_output / "closedloop" / "rollouts").glob("**/*.json"))
        query_artifacts = 0
        for rollout_path in rollout_paths:
            rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
            for turn in rollout.get("turns", []):
                for call in turn.get("calls", []):
                    trace = call.get("action_trace") or {}
                    artifact = trace.get("condition_artifact" if native else "query_artifact")
                    if native and artifact:
                        if (artifact.get("aggregator_sha256") != identity["aggregator_sha256"]
                                or artifact.get("query_tail_requested") is not False
                                or not 1 <= artifact.get("images", 0) <= 11
                                or artifact.get("brain_images") != artifact.get("images")):
                            raise ValueError("native condition identity/image-window audit failed")
                    if artifact and artifact.get("shape") == expected_query_shape:
                        query_artifacts += 1
        if query_artifacts == 0:
            raise RuntimeError(
                "closed-loop run consumed no query artifact matching checkpoint "
                f"shape {expected_query_shape}")
        audit = {
            "schema": "qwen36_4query_service_audit_v1",
            "checkpoint": identity,
            "vllm_entry": str(REPO / "scripts/serve/vllm/vllm-action-head-query.sh"),
            "vllm_port": 28899,
            "vllm_gpus": "0-11 (6 x TP2)" if bootstrap_layout else "14,15",
            "eval_layout": "bootstrap" if bootstrap_layout else "serial",
            "renderer_gpus": "12,13,14" if bootstrap_layout else "10",
            "action_head_gpu": "15" if bootstrap_layout else args.closedloop_gpus,
            "actors": 160 if bootstrap_layout else 1,
            "mtp_tokens": 0,
            "query_tail": True,
            "expected_query_artifact_shape": expected_query_shape,
            "compile_cache": str(runtime_root.parent / "vllm_compile") if bootstrap_layout else "/tmp/action_head_vllm_cache",
            "consumed_query_artifact_count": query_artifacts,
            "query_artifact_lifecycle": "request-scoped safetensors, consumed exactly once",
            "vllm_log": str(step_output / "bootstrap_runtime/logs/vllm_*.log") if bootstrap_layout else str(vllm_log),
        }
        if native:
            audit.update(condition_mode="v5_native", query_tail=False,
                         vllm_entry=str(REPO / "scripts/eval/qwen36_4query_ckpt_sweep/launch_native_vllm.sh"),
                         expected_condition_shape=expected_query_shape,
                         consumed_condition_count=query_artifacts,
                         aggregator_sha256=identity["aggregator_sha256"],
                         condition_lifecycle="request-local full text/tool and at most 8+3 images; no disk feature cache")
            for key in ("expected_query_artifact_shape", "consumed_query_artifact_count", "query_artifact_lifecycle"):
                audit.pop(key, None)
        # Preserve the actual topology of an already completed serial run.
        if not resumed_closed_summary or not (step_output / "service_audit.json").exists():
            atomic_json(step_output / "service_audit.json", audit)
    finally:
        if not bootstrap_layout:
            stop_vllm(pid_file)

    cast_data = os.environ.get("CAST_EVAL_DATA", "")
    if cast_data:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0,1"
        env["ACTION_HEAD_ALLOW_FORCED_COMPLETION"] = "1"
        env["ACTION_HEAD_FLA_L2NORM_DYNAMIC_NB"] = "1"
        env["ACTION_HEAD_R3_CAPTURE_MODE"] = "legacy"
        env["VLLM_RPC_BASE_PATH"] = f"/tmp/action_head_rl/cast_eval_rpc_{os.getpid()}"
        baseline = os.environ.get("CAST_BASELINE_CHECKPOINT", "")
        if baseline:
            run([args.python, str(REPO / "scripts/eval/cast_v43_full_eval.py"),
                 "--data", cast_data, "--output", str(Path(args.output_root) / "cast_baseline"),
                 "--checkpoint", baseline], env=env, log=log_dir / "cast_baseline.log")
        run([args.python, str(REPO / "scripts/eval/cast_v43_full_eval.py"),
             "--data", cast_data, "--output", str(step_output / "cast"),
             "--checkpoint", str(checkpoint)], env=env, log=log_dir / "cast.log")

    d2_data = os.environ.get("CAST_D2_EVAL_DATA", "")
    if d2_data:
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES="0,1", ACTION_HEAD_ALLOW_FORCED_COMPLETION="1",
                   ACTION_HEAD_FLA_L2NORM_DYNAMIC_NB="1", ACTION_HEAD_R3_CAPTURE_MODE="legacy",
                   VLLM_RPC_BASE_PATH=f"/tmp/action_head_rl/cast_d2_eval_rpc_{os.getpid()}")
        baseline = os.environ.get("CAST_BASELINE_CHECKPOINT", "")
        if baseline:
            run([args.python, str(REPO / "scripts/eval/cast_d2_eval.py"),
                 "--data", d2_data, "--output", str(Path(args.output_root) / "cast_d2_baseline"),
                 "--checkpoint", baseline, "--wait-seconds", "604800"],
                env=env, log=log_dir / "cast_d2_baseline.log")
        run([args.python, str(REPO / "scripts/eval/cast_d2_eval.py"),
             "--data", d2_data, "--output", str(step_output / "cast_d2"),
             "--checkpoint", str(checkpoint), "--wait-seconds", "604800"],
            env=env, log=log_dir / "cast_d2.log")

    closed_data = json.loads((step_output / "closedloop" / "summary.json").read_text(encoding="utf-8"))
    if not closed_data.get("evaluation_valid"):
        raise RuntimeError("checkpoint evaluation did not produce a valid closed-loop summary")
    success_contract = official_success_contract_identity()
    if ((closed_data.get("success_contract") or {}).get("contract_sha256") !=
            success_contract["contract_sha256"]):
        raise RuntimeError("closed-loop summary used a different success contract")
    if not args.closedloop_only:
        open_data = json.loads(open_summary.read_text(encoding="utf-8"))
        if not open_data.get("evaluation_valid"):
            raise RuntimeError("checkpoint evaluation did not produce a valid open-loop summary")
    result_payload = {
        "schema": "qwen36_4query_checkpoint_sweep_result_v1",
        "action_head_as_tool_version": args.action_head_as_tool_version,
        **(protocol.protocol_identity() if protocol is not None else {}),
        "checkpoint": identity,
        "closedloop_stages": closed_data["stages"],
        "closedloop_macro_average": closed_data["macro_average"],
        "closedloop_overall_micro": closed_data["overall_micro"],
        "closedloop_repeats": args.closedloop_repeats,
        "closedloop_mean_at_k": closed_data["mean_at_k"],
        **({"closedloop_pass_at_1": closed_data["pass_at_1"],
            "closedloop_pass_at_k": closed_data["pass_at_k"]}
           if os.environ.get("COMPANION_REPORT_PASS_AT_K", "1") == "1"
           else {"report_pass_at_k": False}),
        "success_contract": success_contract,
    }
    if not args.closedloop_only:
        result_payload["openloop_primary"] = open_data["primary_random_mean"]
    if cast_data:
        result_payload["cast_paired"] = json.loads((step_output / "cast/completed.json").read_text())
        if os.environ.get("CAST_BASELINE_CHECKPOINT"):
            result_payload["cast_baseline"] = json.loads((Path(args.output_root) / "cast_baseline/completed.json").read_text())
    if d2_data:
        result_payload["cast_d2_paired"] = json.loads((step_output / "cast_d2/completed.json").read_text())
    atomic_json(step_output / "result.json", result_payload)
    (step_output / "DONE").write_text(json.dumps({
        **identity, "closedloop_repeats": args.closedloop_repeats,
        "success_contract_sha256": success_contract["contract_sha256"],
    }) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-job-id", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sft-manifest", required=True)
    parser.add_argument("--subset-manifest", required=True)
    parser.add_argument("--slow-model", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--bench-url", default="http://127.0.0.1:8938")
    parser.add_argument("--python", default="/usr/local/bin/python3.12")
    parser.add_argument("--openloop-gpus", default="0,1")
    parser.add_argument("--closedloop-gpus", default="11")
    parser.add_argument("--num-random", type=int, default=8)
    parser.add_argument("--max-openloop-anchors", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-tool-calls", type=int, default=30)
    parser.add_argument("--limit-closedloop-cases", type=int, default=0)
    parser.add_argument("--closedloop-repeats", type=int, default=4)
    parser.add_argument("--allow-partial-closedloop", action="store_true")
    parser.add_argument("--closedloop-only", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--stability-seconds", type=int, default=5)
    parser.add_argument("--max-eval-retries", type=int, default=2)
    parser.add_argument("--step-start", type=int, default=200)
    parser.add_argument("--step-stop", type=int, default=5000)
    parser.add_argument("--step-interval", type=int, default=200)
    parser.add_argument("--include-step", type=int, action="append", default=[])
    parser.add_argument("--checkpoint-index", default="")
    parser.add_argument("--expected-checkpoint-count", type=int, default=0)
    parser.add_argument("--disable-idle-gpu-occupier", action="store_true")
    parser.add_argument("--idle-occupier-python", default="/usr/local/bin/python3.12")
    parser.add_argument("--idle-occupier-script", default=str(DEFAULT_IDLE_OCCUPIER))
    parser.add_argument("--action-head-as-tool-version", choices=("v1", "v2", "v3", "v4", "v4.1", "v4.2", "v4.3", "v4.4", "v4.5", "v4.6", "v4.7", "v4.8", "v4.9", "v4.10", "v4.11", "v6"), default="v1")
    args = parser.parse_args()
    if (not args.checkpoint_index and
            (args.step_start < 0 or args.step_stop < args.step_start
             or args.step_interval <= 0)):
        parser.error("expected 0 <= step-start <= step-stop and step-interval > 0")
    if (args.max_openloop_anchors < 0 or args.limit_closedloop_cases < 0 or
            args.closedloop_repeats <= 0 or args.expected_checkpoint_count < 0):
        parser.error("open-loop and closed-loop limits must be non-negative")
    checkpoint_root, output_root = Path(args.checkpoint_root), Path(args.output_root)
    checkpoint_index = Path(args.checkpoint_index) if args.checkpoint_index else None
    expected_steps = (checkpoint_index_steps(checkpoint_index, checkpoint_root)
                      if checkpoint_index else tuple(sorted(set(
        range(args.step_start, args.step_stop + 1, args.step_interval)
    ).union(step for step in args.include_step if step > 0))))
    output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / "CANCELLED").is_file():
        print(f"[watcher] cancelled before evaluation: {output_root}", flush=True)
        return
    protocol = ACTION_HEAD_PROTOCOL.protocol_module(args.action_head_as_tool_version)
    success_contract = official_success_contract_identity()
    atomic_json(output_root / "run_manifest.json", {
        "schema": "qwen36_4query_checkpoint_sweep_run_v1",
        "action_head_as_tool_version": args.action_head_as_tool_version,
        **(protocol.protocol_identity() if protocol is not None else {}),
        "upstream": upstream_status(args.upstream_job_id),
        "checkpoint_root": str(checkpoint_root.resolve()),
        "expected_steps": expected_steps,
        "checkpoint_index": (str(checkpoint_index.resolve())
                             if checkpoint_index else None),
        "expected_checkpoint_count": args.expected_checkpoint_count,
        "sft_manifest": str(Path(args.sft_manifest).resolve()),
        "sft_manifest_sha256": sha256_file(args.sft_manifest),
        "max_openloop_anchors": args.max_openloop_anchors,
        "subset_manifest": str(Path(args.subset_manifest).resolve()),
        "subset_manifest_sha256": sha256_file(args.subset_manifest),
        "closedloop_repeats": args.closedloop_repeats,
        "closedloop_only": bool(args.closedloop_only),
        "eval_layout": os.environ.get("EVAL_LAYOUT", "bootstrap"),
        "brain_mode": os.environ.get("COMPANION_BRAIN_MODE", "sft"),
        "backbone": str(Path(args.backbone).resolve()),
        "max_turns": args.max_turns,
        "max_tool_calls": args.max_tool_calls,
        "success_contract": success_contract,
        "idle_gpu_occupier": {
            "enabled": not args.disable_idle_gpu_occupier,
            "scope": "all_visible_gpus",
            "lifecycle": "start_only_while_waiting; stop_before_checkpoint_stability_and_eval",
            "script": str(Path(args.idle_occupier_script).resolve()),
        },
    })
    attempts = {step: 0 for step in expected_steps}
    failed = {}
    idle_process: list[subprocess.Popen | None] = [None]

    def stop_idle(reason: str) -> None:
        process = idle_process[0]
        stop_process_group(process)
        idle_process[0] = None
        atomic_json(output_root / "idle_gpu_occupier_status.json", {
            "active": False, "reason": reason,
        })

    def cleanup_idle() -> None:
        if idle_process[0] is not None:
            stop_idle("watcher_exit")

    atexit.register(cleanup_idle)
    def handle_sigterm(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_sigterm)
    while True:
        if checkpoint_index:
            try:
                expected_steps = checkpoint_index_steps(
                    checkpoint_index, checkpoint_root)
            except OSError as exc:
                if exc.errno != errno.ESTALE:
                    raise
                # Keep the last validated index and evaluation attempt counts.
                # Do not decide sweep completion from a stale publication list.
                print(f"[watcher] checkpoint index still stale; defer poll, "
                      f"retain steps={expected_steps}: {exc}", flush=True)
                time.sleep(args.poll_seconds)
                continue
            if (args.expected_checkpoint_count and
                    len(expected_steps) > args.expected_checkpoint_count):
                raise RuntimeError(
                    "checkpoint index published more checkpoints than expected: "
                    f"published={len(expected_steps)} "
                    f"expected={args.expected_checkpoint_count}")
            for step in expected_steps:
                attempts.setdefault(step, 0)
        progress = False
        for step in expected_steps:
            checkpoint = checkpoint_root / f"global_step_{step:06d}"
            step_output = output_root / f"global_step_{step:06d}"
            if step_complete(step_output, args.closedloop_repeats) or not checkpoint.is_dir():
                continue
            try:
                checkpoint_identity(checkpoint)
            except Exception:
                continue
            if attempts[step] >= args.max_eval_retries:
                continue
            # Release all synthetic CUDA work before checkpoint validation,
            # model loading, open-loop evaluation, or vLLM startup.
            if idle_process[0] is not None:
                stop_idle(f"checkpoint_{step}_ready")
            attempts[step] += 1
            progress = True
            try:
                evaluate_one(args, checkpoint, step_output)
                failed.pop(step, None)
            except Exception as exc:
                failed[step] = repr(exc)
                atomic_json(step_output / "FAILED_ATTEMPT.json", {
                    "step": step, "attempt": attempts[step], "error": repr(exc)})
                stop_vllm(step_output / "vllm_28899.pid")
        done = [step for step in expected_steps
                if step_complete(
                    output_root / f"global_step_{step:06d}",
                    args.closedloop_repeats)]
        state = upstream_status(args.upstream_job_id)
        atomic_json(output_root / "watcher_status.json", {
            "upstream": state, "completed_steps": done,
            "published_steps": expected_steps,
            "expected_checkpoint_count": args.expected_checkpoint_count,
            "failed_steps": failed, "attempts": attempts,
        })
        sweep_ready = (len(done) == len(expected_steps) and
                       (not checkpoint_index or (
                           args.expected_checkpoint_count > 0 and
                           len(expected_steps) == args.expected_checkpoint_count)))
        if sweep_ready:
            if idle_process[0] is not None:
                stop_idle("sweep_complete")
            atomic_json(output_root / "SWEEP_COMPLETE.json", {
                "upstream": state, "completed_steps": done})
            return
        if state.get("status") in TERMINAL_FAILURE:
            raise RuntimeError(f"upstream terminated unsuccessfully: {state}")
        if state.get("status") in TERMINAL_SUCCESS:
            if (checkpoint_index and len(expected_steps) <
                    args.expected_checkpoint_count):
                raise RuntimeError(
                    "upstream succeeded before publishing every round checkpoint: "
                    f"published={len(expected_steps)} "
                    f"expected={args.expected_checkpoint_count}")
            exhausted = [step for step in expected_steps
                         if step not in done and attempts[step] >= args.max_eval_retries]
            missing = [step for step in expected_steps
                       if not (checkpoint_root / f"global_step_{step:06d}").is_dir()]
            if exhausted or missing:
                raise RuntimeError(f"sweep incomplete after upstream success: missing={missing} exhausted={exhausted}")
        if not progress:
            if not args.disable_idle_gpu_occupier:
                if idle_process[0] is None or idle_process[0].poll() is not None:
                    idle_process[0] = start_idle_gpu_occupier(args, output_root)
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
