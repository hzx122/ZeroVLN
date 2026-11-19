"""CPU planning and immutable-input checks for a zero-update joint-forward audit.

This module does not initialize CUDA/Ray. Execution is an explicit, separate
entry point; incomplete captures and live prerequisite processes fail closed.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading

import torch

from action_head_rl.joint_fixture import SCHEMA as FIXTURE_SCHEMA, actor_source_binding
from action_head_rl.joint_policy import build_joint_policy_groups, consecutive_joint_groups
from action_head_rl.joint_scheduling import shard_joint_policy_groups
from action_head_rl.text_policy import is_text_record


SCHEMA = "v6-zero-update-joint-audit-v1"
ARTIFACTS = Path("/data/zerovln/artifacts")
PREREQUISITES = {"inputfix_driver", "inputfix_launcher", "credit_driver", "credit_launcher"}
DEFAULT_PREREQUISITE_PROFILE = "after_inputfix_and_credit"
GAP_PREREQUISITE_PROFILE = "after_inputfix_before_credit"
PREREQUISITE_PROFILES = {
    DEFAULT_PREREQUISITE_PROFILE: PREREQUISITES,
    GAP_PREREQUISITE_PROFILE: {
        "pilot_fast_driver", "pilot_fast_launcher", "inputfix_driver", "inputfix_launcher"},
}
_CREDIT_START_MARKERS = ("monitor_process.json", "monitor.lock", "monitor_state.json",
                         "candidate_start_attempt.json", "candidate_process.json")
THRESHOLDS = {
    "gradient_atol": 1e-6, "gradient_rtol": 1e-3,
    "output_atol": 1e-2, "output_rtol": 1e-3,
    "loss_atol": 1e-6, "loss_rtol": 1e-3,
}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_new_json(path, value):
    """Never replace an earlier plan, attempt, or result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("x") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_source(source):
    from action_head_rl.checkpoint import resolve_resume_checkpoint, _validate_fp32_policy_payload
    path = Path(source).resolve()
    checkpoint = resolve_resume_checkpoint(mode="resume_path", output_dir=path.parent,
        resume_from_path=str(path), train_scope="full", learner_world_size=12, allow_weights_only=False)
    if checkpoint.step != 16:
        raise ValueError("audit requires the complete original step-16 source")
    contract_path = path.parent / "contract.json"
    contract = read_json(contract_path)
    if (contract.get("learner_world_size") != 12 or contract.get("learner_parameter_dtype") != "fp32"
            or contract.get("harness_version") != "v6"):
        raise ValueError("audit source must be the twelve-rank FP32 v6 policy")
    index_sha = _validate_fp32_policy_payload(checkpoint)
    source_binding = actor_source_binding(path, qwen=checkpoint.qwen, head=checkpoint.head, query=checkpoint.query)
    # Query producer may bind the actual .pt instead of its containing folder.
    source_binding["query"] = str(checkpoint.query / "dualvln_query_embeddings.pt")
    return {"checkpoint": str(path), "binding": source_binding,
            "qwen_index_sha256": index_sha, "contract": contract,
            "contract_sha256": sha256(contract_path)}


def _same_source(actual, expected):
    if not isinstance(actual, dict):
        return False
    actual, expected = dict(actual), dict(expected)
    for item in (actual, expected):
        query = Path(item.get("query", "/missing"))
        if query.name != "dualvln_query_embeddings.pt":
            query /= "dualvln_query_embeddings.pt"
        item["query"] = str(query)
    return actual == expected


def load_fixture(directory, *, source_binding):
    directory = Path(directory).resolve()
    complete = read_json(directory / "complete.json")
    targets = read_json(directory / "targets.json")
    if (complete.get("status") != "complete" or complete.get("schema") != FIXTURE_SCHEMA
            or targets.get("schema") != FIXTURE_SCHEMA
            or complete.get("raw_sha256") != sha256(directory / "raw.pt")
            or complete.get("targets_sha256") != sha256(directory / "targets.json")
            or targets.get("raw_sha256") != complete["raw_sha256"]):
        raise ValueError("fixture is incomplete or its raw/target hashes changed")
    # These are locally captured trusted training artifacts, with PIL/tensor
    # objects. Never load arbitrary downloaded pickle files with this runner.
    raw = torch.load(directory / "raw.pt", map_location="cpu", weights_only=False)
    capture = raw.get("capture", {})
    if (raw.get("schema") != FIXTURE_SCHEMA or targets.get("step") != 17
            or targets.get("policy_version") != 16 or raw.get("policy_version") != 16
            or targets.get("logical_denominator") != 32
            or capture.get("step") != 17 or capture.get("rollout_index") not in (0, 1)
            or targets.get("rollout_index") != capture["rollout_index"]
            or not _same_source(capture.get("actor_source"), source_binding)
            or not _same_source(targets.get("actor_source"), source_binding)
            or not _same_source(complete.get("actor_source"), source_binding)):
        raise ValueError("fixture source, policy, capture index, or original denominator differs")
    for key in ("native_request_id", "decision_index"):
        if raw.get(key) != targets.get(key) or raw.get(key) != complete.get(key):
            raise ValueError(f"fixture identity differs: {key}")
    fm, fm_targets = raw["fm_records"], targets["fm_targets"]
    if not fm or len(fm) != len(fm_targets):
        raise ValueError("fixture has incomplete FM chunk targets")
    if (len({row["chunk_index"] for row in fm}) != len(fm)
            or [row["chunk_index"] for row in fm] != complete.get("chunk_indices")):
        raise ValueError("fixture has duplicate or reordered chunks")
    rows = []
    for record, target in [*zip(fm, fm_targets), (raw["text_record"], targets["text_target"])]:
        if any(record.get(key) != target.get(key) for key in
               (("decision_index", "option_id", "policy_version") if is_text_record(record)
                else ("option_id", "chunk_index", "policy_version"))):
            raise ValueError("fixture target belongs to another decision/chunk")
        for key in ("advantage", "loss_weight", "sample_valid"):
            value = target.get(key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"fixture target lacks a finite {key}")
        if target["sample_valid"] != 1.0 or target["loss_weight"] <= 0 or not target.get("condition_id"):
            raise ValueError("fixture must contain valid original branch weights and identities")
        if is_text_record(record):
            if (target.get("loss_coef") != target.get("text_loss_coef")
                    or target.get("temperature") != target.get("text_temperature")):
                raise ValueError("fixture text coefficient/temperature targets differ")
        elif target.get("loss_coef") != 1.0:
            raise ValueError("fixture changes the FM objective coefficient")
        row = dict(record, **target)
        row["joint_episode_id"] = f"step=17/rollout={capture['rollout_index']}/policy=16"
        rows.append(row)
    descriptors, pairing = build_joint_policy_groups(rows)
    if len(descriptors) != 1 or descriptors[0]["family"] != "joint" or pairing["paired_decisions"] != 1:
        raise ValueError("fixture is not exactly one eligible native text/FM pair")
    return {"directory": str(directory), "rows": rows, "descriptor": descriptors[0],
            "denominator": 32.0, "rollout_index": capture["rollout_index"],
            "credit_mode": targets.get("credit_mode"), "critic_sha256": targets.get("critic_sha256"),
            "complete_sha256": sha256(directory / "complete.json"),
            "raw_sha256": complete["raw_sha256"], "targets_sha256": complete["targets_sha256"]}


def paired_rank_streams(fixtures, world=12):
    """Derive both branches from ONE scheduler, preserving rank and pad layout."""
    if world < 1 or not 1 <= len(fixtures) <= 2:
        raise ValueError("audit supports one or two captured pairs")
    if len({row["rollout_index"] for row in fixtures}) != len(fixtures):
        raise ValueError("fixture rollout indices must be unique")
    joint, schedule = shard_joint_policy_groups([row["descriptor"] for row in fixtures],
                                               world, logical_valid=32.0)
    separate = []
    for rank in joint:
        stream = []
        for group in consecutive_joint_groups(rank):
            fm = [dict(row) for row in group if not is_text_record(row)]
            text = [dict(row) for row in group if is_text_record(row)]
            queries = len(fm[0]["query_routed_experts"])
            text[0]["_condition_pack_token_width"] -= queries
            for record in fm + text:
                record.pop("_joint_family", None)
                record.pop("_joint_group_id", None)
                record["logical_batch_valid"] = 32.0
                stream.append(record)
        separate.append(stream)
    return {"separate": separate, "joint": joint, "schedule": schedule, "logical_valid": 32.0}


@contextmanager
def temporary_environment(**values):
    prior = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@temporary_environment(ACTION_HEAD_TEXT_POLICY="1")
def pack_rank_pair(joint_rows, separate_rows, processor, position_fn):
    from action_head_rl.worker import joint_group_to_tensordict, record_group_to_tensordict, text_record_to_tensordict
    from verl.utils.tensordict_utils import maybe_fix_3d_position_ids
    joint, separate = [], []
    by_condition = {}
    for row in separate_rows:
        by_condition.setdefault(row["condition_id"], []).append(row)
    for group in consecutive_joint_groups(joint_rows):
        identity = group[0]["_joint_group_id"]
        rows = [dict(row, logical_batch_valid=32.0) for row in group]
        data = joint_group_to_tensordict(rows, processor, position_fn)
        maybe_fix_3d_position_ids(data)
        joint.append((identity, "joint", data))
        fm = [row for row in rows if not is_text_record(row)]
        text = next(row for row in rows if is_text_record(row))
        data = record_group_to_tensordict(by_condition[fm[0]["condition_id"]], processor, "cpu")
        maybe_fix_3d_position_ids(data)
        separate.append((identity, "fm", data))
        data = text_record_to_tensordict(by_condition[text["condition_id"]][0], processor, "cpu")
        maybe_fix_3d_position_ids(data)
        separate.append((identity, "text", data))
    return {"joint": joint, "separate": separate}


def cpu_preflight(streams, source):
    """Real local processor and official MRoPE methods, without model weights."""
    from types import MethodType, SimpleNamespace
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeModel
    from action_head_rl.worker import _cpu_model_positions, joint_payload_signature
    config = AutoConfig.from_pretrained(source["binding"]["qwen"], local_files_only=True)
    processor = AutoProcessor.from_pretrained(source["binding"]["qwen"], local_files_only=True)
    expected_queries = read_json(Path(source["binding"]["head"]) / "fast_slow_action_head_config.json")["query_num_queries"]
    shell = SimpleNamespace(config=config, rope_deltas=None)
    for name in ("get_vision_position_ids", "get_rope_index", "compute_3d_position_ids"):
        setattr(shell, name, MethodType(getattr(Qwen3_5MoeModel, name), shell))
    signatures = []
    for joint_rows, separate_rows in zip(streams["joint"], streams["separate"]):
        if any(len(row["query_routed_experts"]) != expected_queries for row in joint_rows if not is_text_record(row)):
            raise ValueError("fixture query count differs from source policy")
        packed = pack_rank_pair(joint_rows, separate_rows, processor, lambda payload: _cpu_model_positions(shell, payload))
        local = []
        for (_, _, data), group, spec in zip(packed["joint"], consecutive_joint_groups(joint_rows), streams["schedule"]["rounds"]):
            local.append(joint_payload_signature(data, family="joint", width=spec["token_width"],
                fm_chunks=spec["fm_chunks_per_rank"], query_count=len(group[0]["query_routed_experts"]),
                layer_count=config.text_config.num_hidden_layers, top_k=config.text_config.num_experts_per_tok))
        signatures.append(local)
    if any(row != signatures[0] for row in signatures):
        raise ValueError("audit has unequal actual collective signatures across ranks")
    return {"passed": True, "rank_count": len(signatures), "joint_signatures": signatures[0]}


def prerequisite_labels(profile):
    if profile not in PREREQUISITE_PROFILES:
        raise ValueError("unknown audit prerequisite profile")
    return PREREQUISITE_PROFILES[profile]


def pending_credit_guard(pending, *, experiment_root=None):
    """Prepared packages are allowed; any launch/monitor attempt fails closed."""
    if not isinstance(pending, dict) or set(pending) != {"run_dir", "launch_dir"}:
        raise ValueError("gap audit requires explicit pending credit run and launch directories")
    paths = [Path(pending[key]) for key in ("run_dir", "launch_dir")]
    if any(not p.is_absolute() or p.is_symlink() or p.resolve() != p for p in paths):
        raise ValueError("pending credit paths must be canonical absolute paths without aliases")
    run, launch = paths
    if run == launch or run in launch.parents or launch in run.parents:
        raise ValueError("pending credit run and launch must be separate directories")
    if experiment_root is not None and any(p.parent != experiment_root for p in paths):
        raise ValueError("pending credit directories must belong to the same experiment root")
    if run.exists():
        raise RuntimeError("pending credit training directory already exists")
    if launch.exists() and not launch.is_dir():
        raise ValueError("pending credit launch is not a directory")
    if any((launch / name).exists() or (launch / name).is_symlink() for name in _CREDIT_START_MARKERS):
        raise RuntimeError("pending credit monitor or candidate has already been armed or started")
    if (launch / "plan.json").is_file():
        prepared = read_json(launch / "plan.json")
        if prepared.get("trial") != "credit" or prepared.get("candidate_run") != str(run):
            raise ValueError("prepared pending credit package belongs to another run")
    return run, launch


def validate_gap_binding(*, runtime_plan, source, fixtures, prerequisites, pending):
    """Bind the gap to already recorded input-fix and old pilot identities."""
    runtime_path = Path(runtime_plan)
    runtime = read_json(runtime_path)
    launch = Path(runtime["launch_dir"])
    inputfix, baseline = Path(runtime["candidate_run"]), Path(runtime["baseline_run"])
    if (runtime.get("schema") != "v6-local-trial-frozen-launch-v1"
            or runtime.get("trial") != "input-fix" or runtime_path != launch / "plan.json"
            or inputfix.name != "pilot_inputfix" or baseline != inputfix.parent / "pilot_fast"
            or runtime.get("initial_checkpoint") != source["checkpoint"]
            or runtime["environment"].get("ACTION_HEAD_JOINT_TEXT_FM_FORWARD") != "0"):
        raise ValueError("gap audit requires the frozen source16 input-fix runtime with shared forward disabled")
    run, credit_launch = pending_credit_guard(pending, experiment_root=inputfix.parent)
    if run in (inputfix, baseline) or credit_launch in (launch, inputfix, baseline):
        raise ValueError("pending credit paths overlap a prerequisite run")
    capture = runtime.get("joint_fixture_capture", {})
    if (capture.get("enabled") is not True or capture.get("step") != 17
            or capture.get("source_policy_version") != 16
            or capture.get("shared_forward_enabled") is not False):
        raise ValueError("input-fix runtime does not declare source16 first-batch observational capture")
    for fixture in fixtures:
        if (fixture.get("credit_mode") != "episode_grpo" or fixture.get("critic_sha256") is not None
                or not Path(fixture["directory"]).is_relative_to(Path(capture["directory"]))):
            raise ValueError("gap audit requires the original episode_grpo input-fix fixture")
    contract_path, process_path = launch / "validated_contract.json", launch / "candidate_process.json"
    contract, actual_process = read_json(contract_path), read_json(process_path)
    if (contract.get("resume", {}).get("start_step") != 16
            or contract.get("resume", {}).get("exact_trainer_state") is not True
            or contract.get("tool_credit") is not None
            or contract.get("reward", {}).get("normalization") != "same_case_trajectory_outcome_grpo"
            or contract.get("joint_text_fm_forward", False) is not False):
        raise ValueError("frozen input-fix contract changed its actor source or learning objective")
    identities = {row["label"]: row["identity"] for row in prerequisites}
    for label, wanted in (
            ("pilot_fast_driver", runtime["baseline_driver"]),
            ("pilot_fast_launcher", runtime["baseline_launcher"]),
            ("inputfix_driver", actual_process), ("inputfix_launcher", actual_process)):
        # This frozen entry.sh execs the trainer in place: its driver and launcher
        # share one actual recorded identity, not an invented future child PID.
        if identities.get(label) != {"pid": wanted["pid"], "start_ticks": str(wanted["start_ticks"])}:
            raise ValueError(f"gap prerequisite does not match the real recorded process: {label}")
    return [runtime_path, contract_path, process_path]


def prepare_plan(*, fixture_dirs, source_checkpoint, runtime_plan, prerequisites, output,
                 warmup=1, repeats=3, thresholds=None, rpc_timeout_seconds=3600,
                 prerequisite_profile=DEFAULT_PREREQUISITE_PROFILE,
                 pending_credit_run_dir=None, pending_credit_launch_dir=None):
    output, runtime_plan = Path(output).resolve(), Path(runtime_plan).resolve()
    if not output.is_relative_to(ARTIFACTS) or output == ARTIFACTS or output.exists():
        raise ValueError("audit output must be a new data-disk directory")
    if warmup < 1 or repeats < 2:
        raise ValueError("audit requires cold runs, at least one warmup and two timing repeats")
    if not math.isfinite(rpc_timeout_seconds) or rpc_timeout_seconds <= 0:
        raise ValueError("audit RPC deadline must be finite and positive")
    limits = dict(THRESHOLDS if thresholds is None else thresholds)
    if set(limits) != set(THRESHOLDS) or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in limits.values()):
        raise ValueError("all six numerical thresholds must be fixed finite positive values")
    source = validate_source(source_checkpoint)
    fixtures = [load_fixture(path, source_binding=source["binding"]) for path in fixture_dirs]
    streams = paired_rank_streams(fixtures)
    packed_audit = cpu_preflight(streams, source)
    identities = []
    if set(prerequisites) != prerequisite_labels(prerequisite_profile):
        raise ValueError("provide the exact driver and launcher records required by the prerequisite profile")
    for label, path in sorted(prerequisites.items()):
        path = Path(path).resolve()
        row = read_json(path)
        if type(row.get("pid")) is not int or row["pid"] <= 0 or not str(row.get("start_ticks", "")).isdigit():
            raise ValueError(f"invalid prerequisite process record: {label}")
        identities.append({"label": label, "path": str(path), "sha256": sha256(path),
                           "identity": {"pid": row["pid"], "start_ticks": str(row["start_ticks"])}})
    runtime = read_json(runtime_plan)
    pending = None
    gap_sources = []
    if prerequisite_profile == GAP_PREREQUISITE_PROFILE:
        pending = {"run_dir": str(pending_credit_run_dir or ""),
                   "launch_dir": str(pending_credit_launch_dir or "")}
        gap_sources = validate_gap_binding(runtime_plan=runtime_plan, source=source,
            fixtures=fixtures, prerequisites=identities, pending=pending)
    elif pending_credit_run_dir is not None or pending_credit_launch_dir is not None:
        raise ValueError("pending credit paths apply only to the before-credit prerequisite profile")
    environment = dict(runtime["environment"])
    old_runtime = environment["ACTION_HEAD_LOCAL_RUNTIME_ROOT"]
    cache_root = "/tmp/ah_joint/" + hashlib.sha256(str(output).encode()).hexdigest()[:12]
    environment = {key: value.replace(old_runtime, cache_root) for key, value in environment.items()}
    repo = Path(__file__).resolve().parents[2]
    upstream = Path(runtime["snapshot_upstream"])
    environment["PYTHONPATH"] = ":".join((environment["FAST_KERNEL_OVERLAY"], str(repo / "verl"),
                                         str(upstream), str(repo / "vln_demo/src")))
    environment.update(CUDA_VISIBLE_DEVICES=",".join(map(str, range(12))),
        ACTION_HEAD_HEAD_CHECKPOINT=source["binding"]["head"],
        ACTION_HEAD_QUERY_CHECKPOINT=source["binding"]["query"],
        ACTION_HEAD_TRAIN_SCOPE="full", ACTION_HEAD_TEXT_POLICY="1",
        ACTION_HEAD_JOINT_TEXT_FM_FORWARD="1", ACTION_HEAD_OLD_LOGPROB_MODE="single_forward",
        ACTION_HEAD_ROUTER_REPLAY="R3", ACTION_HEAD_CAPTURE_FULL_ROUTING="1",
        ACTION_HEAD_OPTION_CONDITION_REUSE="1", ACTION_HEAD_DEFER_GRADIENT_SYNC="0",
        ACTION_HEAD_MULTI_CONDITION_BATCH_SIZE="1", ACTION_HEAD_FULL_WEIGHT_RESIDENCY_GIB="0",
        ACTION_HEAD_FROZEN_WEIGHT_RESIDENCY="0", ACTION_HEAD_VLM_LR="1e-6")
    contract = source["contract"]
    learner = dict(slow_model=source["binding"]["qwen"], learner_parameter_dtype="fp32",
        gradient_checkpointing=contract["gradient_checkpointing"], fsdp_size=contract["fsdp_size"],
        fsdp_forward_prefetch=contract["fsdp_forward_prefetch"], learning_rate=contract["action_head_learning_rate"],
        weight_decay=contract["optimizer_weight_decay"], opt_steps=22)
    files = [repo / "scripts/eval/audit_v6_joint_forward.py"]
    for directory in (repo / "verl/action_head_rl", repo / "verl/models", repo / "verl/sft",
                      upstream / "verl/workers/engine", Path(environment["FAST_KERNEL_OVERLAY"])):
        files.extend(directory.rglob("*.py"))
    files.extend([upstream / "verl/workers/engine_workers.py", runtime_plan, *gap_sources])
    return {"schema": SCHEMA, "output": str(output), "source": source,
        "fixtures": [{key: value for key, value in row.items() if key not in ("rows", "descriptor")} for row in fixtures],
        "world_size": 12, "logical_denominator": 32.0, "optimizer_updates": 0,
        "thresholds": limits, "warmup": warmup, "repeats": repeats,
        "rpc_timeout_seconds": rpc_timeout_seconds,
        "nonzero_original_targets": {family: any(row["advantage"] != 0.0 for fixture in fixtures
            for row in fixture["rows"] if is_text_record(row) == (family == "text")) for family in ("fm", "text")},
        "prerequisites": identities, "prerequisite_profile": prerequisite_profile,
        **({"pending_credit": pending, "profile_runtime_plan": str(runtime_plan)} if pending else {}),
        "environment": environment, "learner": learner,
        "sources_sha256": {str(path): sha256(path) for path in sorted(set(files))},
        "cpu_preflight": packed_audit, "schedule": streams["schedule"],
        "scope": "one or two fixed native pairs; partial gradient contribution; no full-step throughput or RL benefit claim"}


def validate_plan(plan, *, preflight=True):
    profile = plan.get("prerequisite_profile", DEFAULT_PREREQUISITE_PROFILE)
    if (len(plan.get("prerequisites", [])) != 4
            or {row["label"] for row in plan["prerequisites"]} != prerequisite_labels(profile)):
        raise ValueError("prerequisite process labels differ from the declared profile")
    if profile != GAP_PREREQUISITE_PROFILE and ("pending_credit" in plan or "profile_runtime_plan" in plan):
        raise ValueError("pending credit state cannot be attached to the default prerequisite profile")
    if (plan.get("schema") != SCHEMA or plan.get("world_size") != 12
            or plan.get("optimizer_updates") != 0 or plan.get("logical_denominator") != 32.0
            or set(plan.get("thresholds", {})) != set(THRESHOLDS)
            or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                   for value in plan.get("thresholds", {}).values())):
        raise ValueError("invalid zero-update audit plan contract")
    if (not math.isfinite(plan.get("rpc_timeout_seconds", 0)) or plan.get("rpc_timeout_seconds", 0) <= 0
            or plan.get("warmup", 0) < 1 or plan.get("repeats", 0) < 2
            or plan["environment"].get("CUDA_VISIBLE_DEVICES") != ",".join(map(str, range(12)))
            or plan["learner"].get("slow_model") != plan["source"]["binding"]["qwen"]
            or plan["learner"].get("learner_parameter_dtype") != "fp32"
            or plan["environment"].get("ACTION_HEAD_HEAD_CHECKPOINT") != plan["source"]["binding"]["head"]
            or plan["environment"].get("ACTION_HEAD_QUERY_CHECKPOINT") != plan["source"]["binding"]["query"]):
        raise ValueError("audit runtime/source/timeout contract differs")
    output = Path(plan["output"]).resolve()
    if output == ARTIFACTS or not output.is_relative_to(ARTIFACTS):
        raise ValueError("audit output escaped the artifact directory")
    for name, wanted in plan["sources_sha256"].items():
        if sha256(name) != wanted:
            raise ValueError(f"audit source/dependency changed; prepare a new plan: {name}")
    for record in plan["prerequisites"]:
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("prerequisite process record changed since planning")
        actual = read_json(record["path"])
        if record["identity"] != {"pid": actual["pid"], "start_ticks": str(actual["start_ticks"])}:
            raise ValueError("prerequisite identity differs from its bound process record")
    source = validate_source(plan["source"]["checkpoint"])
    if source != plan["source"]:
        raise ValueError("source policy metadata changed since planning")
    fixtures = [load_fixture(row["directory"], source_binding=source["binding"]) for row in plan["fixtures"]]
    if profile == GAP_PREREQUISITE_PROFILE:
        gap_sources = validate_gap_binding(runtime_plan=plan["profile_runtime_plan"], source=source,
            fixtures=fixtures, prerequisites=plan["prerequisites"], pending=plan["pending_credit"])
        if any(str(path) not in plan["sources_sha256"] for path in gap_sources):
            raise ValueError("gap prerequisite runtime records are not hash-bound")
    for old, new in zip(plan["fixtures"], fixtures):
        if any(old[key] != new[key] for key in ("complete_sha256", "raw_sha256", "targets_sha256")):
            raise ValueError("captured fixture changed since planning")
    streams = paired_rank_streams(fixtures)
    if streams["schedule"] != plan["schedule"]:
        raise ValueError("fixed audit rank schedule changed")
    nonzero = {family: any(row["advantage"] != 0.0 for fixture in fixtures for row in fixture["rows"]
                          if is_text_record(row) == (family == "text")) for family in ("fm", "text")}
    if nonzero != plan["nonzero_original_targets"]:
        raise ValueError("audit target informativeness changed")
    if preflight and cpu_preflight(streams, source) != plan["cpu_preflight"]:
        raise ValueError("CPU packed-input preflight changed")
    return streams


def resolve_rpc(value, *, timeout_seconds):
    """One absolute deadline shared by all nested rank references; no retry."""
    import ray
    import time
    deadline = time.monotonic() + timeout_seconds
    def resolve(item):
        if isinstance(item, ray.ObjectRef):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("zero-update audit RPC deadline expired")
            return resolve(ray.get(item, timeout=remaining))
        if isinstance(item, (list, tuple)):
            return [resolve(child) for child in item]
        if isinstance(item, dict):
            return {key: resolve(child) for key, child in item.items()}
        return item
    return resolve(value)


@contextmanager
def wall_clock_deadline(seconds, label):
    """Bound upstream blocking Ray bootstrap calls without changing upstream.

    The standalone CLI owns its main thread. Refuse an already armed alarm
    instead of extending another caller's deadline or changing its semantics.
    """
    if (threading.current_thread() is not threading.main_thread()
            or not math.isfinite(seconds) or seconds <= 0
            or signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0)):
        raise RuntimeError("audit bootstrap deadline requires an unarmed main-thread timer")
    previous = signal.getsignal(signal.SIGALRM)
    def expired(_signum, _frame):
        raise TimeoutError(f"zero-update audit {label} deadline expired")
    try:
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@contextmanager
def gap_execution_lease(plan, *, lock_path=Path("/tmp/ah_joint/gpu0-11.audit.lock")):
    """Serialize cooperating audits; unrelated launchers still require the gate."""
    profile = plan.get("prerequisite_profile", DEFAULT_PREREQUISITE_PROFILE)
    prerequisite_labels(profile)
    if profile != GAP_PREREQUISITE_PROFILE:
        yield
        return
    import fcntl
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another joint audit holds the GPU 0..11 lease") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def execute_plan(plan_path, *, expected_sha256):
    if sha256(plan_path) != expected_sha256:
        raise ValueError("reviewed plan hash differs")
    with gap_execution_lease(read_json(plan_path)):
        return _execute_plan_with_lease(plan_path, expected_sha256=expected_sha256)


def _execute_plan_with_lease(plan_path, *, expected_sha256):
    """Single explicit attempt. Gate/query errors never become permission to run."""
    import statistics
    import sys
    import time
    from types import SimpleNamespace
    plan_path = Path(plan_path).resolve()
    if sha256(plan_path) != expected_sha256:
        raise ValueError("reviewed plan hash differs")
    plan = read_json(plan_path)
    streams = validate_plan(plan)
    gate = execution_gate(plan)
    if torch.cuda.is_initialized():
        raise RuntimeError("execute requires a fresh process without an initialized CUDA runtime")
    output = Path(plan["output"])
    write_new_json(output / "attempt.json", {"plan_sha256": expected_sha256, "pid": os.getpid(),
        "start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19],
        "gate": gate, "time": time.time(), "automatic_retry": False})
    for key in list(os.environ):
        if key.startswith("ACTION_HEAD_"):
            os.environ.pop(key)
    os.environ.update(plan["environment"])
    for name in reversed(os.environ["PYTHONPATH"].split(":")):
        if name:
            sys.path.insert(0, name)
    cache_root = Path(os.environ["ACTION_HEAD_LOCAL_RUNTIME_ROOT"])
    if not cache_root.is_relative_to("/tmp/ah_joint"):
        raise ValueError("audit runtime cache escaped its isolated local directory")
    for key, value in plan["environment"].items():
        if value.startswith(str(cache_root)) and ":" not in value:
            Path(value).mkdir(parents=True, exist_ok=True)
    # Recheck immediately before creating GPU actors. A query failure aborts.
    execution_gate(plan)
    import ray
    from action_head_rl.joint_audit_worker import JointForwardAuditWorker
    from action_head_rl.trainer import learner_config
    from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
    started = time.perf_counter()
    try:
        # Upstream RayWorkerGroup waits for pg.ready() and master-address RPCs
        # inside its constructor; these precede our explicit resolve_rpc calls.
        with wall_clock_deadline(plan["rpc_timeout_seconds"], "Ray bootstrap/constructor"):
            ray.init(address="local", num_gpus=12, include_dashboard=False,
                _temp_dir=str(cache_root / "ray"), runtime_env={"env_vars": plan["environment"]})
            learner = RayWorkerGroup(resource_pool=RayResourcePool(process_on_nodes=[12], max_colocate_count=4),
                ray_cls_with_init=RayClassWithInitArgs(ray.remote(max_restarts=0, max_task_retries=0)(JointForwardAuditWorker),
                    config=learner_config(SimpleNamespace(**plan["learner"]))), device_name="cuda")
        resolve = lambda value: resolve_rpc(value, timeout_seconds=plan["rpc_timeout_seconds"])
        # reset's regular bound method is blocking; use its async primitive so
        # a failed FSDP rank cannot strand the controller inside ray.get.
        resolve(learner.execute_all_async("reset"))
        initialization_seconds = time.perf_counter() - started
        fast = resolve(learner.execute_all_async("audit_qwen_fast_path"))
        ready = resolve(learner.prepare_audit(
            joint_rows=streams["joint"], separate_rows=streams["separate"]))
        if len(ready) != 12 or {row["rank"] for row in ready} != set(range(12)):
            raise RuntimeError("not every audit rank completed CPU preparation")
        # Identities differ for rank-padding copies, but actual branch and
        # tensor shapes must agree before any rank enters FSDP collectives.
        signatures = [{mode: [{key: value for key, value in batch.items() if key != "identity"}
                             for batch in rows] for mode, rows in row["shapes"].items()} for row in ready]
        if any(row != signatures[0] for row in signatures):
            raise RuntimeError("audit workers have different collective schedules")
        ranks = resolve(learner.zero_update_audit(plan=plan))
        if len(ranks) != 12 or {row["rank"] for row in ranks} != set(range(12)):
            raise RuntimeError("incomplete rank audit results")
        times = {mode: [max(next(row["seconds"] for row in rank["timings"]
                                  if row["repeat"] == repeat and row["mode"] == mode) for rank in ranks)
                        for repeat in range(plan["repeats"])] for mode in ("separate", "joint")}
        informative = (all(plan["nonzero_original_targets"].values())
                       and sum(rank["local_gradient_squared"] for rank in ranks) > 0)
        passed = all(rank["numeric_passed"] and rank["state_unchanged"] and rank["rng_restored"]
                     and not rank["forbidden_call_attempts"] for rank in ranks)
        summary = {"schema": SCHEMA, "plan_sha256": expected_sha256,
            "status": "passed" if passed and informative else "failed" if not passed else "inconclusive_zero_targets",
            "gradient_informative": informative, "optimizer_updates": 0,
            "initialization_seconds": initialization_seconds, "fast_path": fast,
            "thresholds": plan["thresholds"], "max_rank_seconds_by_repeat": times,
            "median_seconds": {mode: statistics.median(values) for mode, values in times.items()},
            "joint_over_separate_median_time": statistics.median(times["joint"]) / statistics.median(times["separate"]),
            "scope": plan["scope"], "ranks": ranks}
        write_new_json(output / "result.json", summary)
        return summary
    except Exception as exc:
        write_new_json(output / "failure.json", {"type": type(exc).__name__, "error": str(exc),
            "plan_sha256": expected_sha256, "automatic_retry": False})
        raise
    finally:
        if ray.is_initialized():
            ray.shutdown()


@contextmanager
def forbid_updates(engine):
    """Guard actual methods, including a caller that bypasses engine wrappers."""
    changed, attempts = [], []
    objects = [(engine, ("optimizer_step", "lr_scheduler_step")),
               (engine.optimizer, ("step", "step_with_ready_grads")), (engine.lr_scheduler, ("step",)),
               (getattr(engine, "checkpoint_manager", None), ("save_checkpoint",))]
    def forbidden(label):
        def call(*args, **kwargs):
            attempts.append(label)
            raise RuntimeError(f"zero-update audit forbids {label}")
        return call
    try:
        for obj, names in objects:
            if obj is None:
                continue
            for name in names:
                if hasattr(obj, name):
                    changed.append((obj, name, getattr(obj, name), name in vars(obj)))
                    setattr(obj, name, forbidden(f"{type(obj).__name__}.{name}"))
        yield attempts
    finally:
        for obj, name, value, owned in reversed(changed):
            if owned:
                setattr(obj, name, value)
            else:
                delattr(obj, name)


def tensor_difference(left, right, *, atol, rtol):
    if (left is None) != (right is None):
        return {"passed": False, "missing_gradient": True}
    if left is None:
        return {"passed": True, "both_none": True}
    if left.shape != right.shape:
        return {"passed": False, "shape_mismatch": [list(left.shape), list(right.shape)]}
    # Move shards before widening; a 35B audit must not allocate FP64 gradient
    # copies on the learner GPUs merely to compute CPU comparison statistics.
    left, right = left.detach().cpu().double(), right.detach().cpu().double()
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    if not finite:
        return {"passed": False, "nonfinite": True}
    delta = left - right
    error_sq, norm_sq = float(delta.square().sum()), float(left.square().sum())
    return {"passed": bool(torch.allclose(left, right, atol=atol, rtol=rtol)),
            "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
            "error_squared": error_sq, "baseline_squared": norm_sq,
            "relative_l2": math.sqrt(error_sq / max(norm_sq, 1e-30))}


def process_alive(identity, *, proc_root=Path("/proc")):
    try:
        fields = (proc_root / str(identity["pid"]) / "stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return False
    return fields[19] == str(identity["start_ticks"]) and fields[0] not in ("Z", "X")


def query_gpu_state():
    def rows(query):
        result = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"],
                                check=True, capture_output=True, text=True, timeout=30)
        return [tuple(piece.strip() for piece in line.split(","))
                for line in result.stdout.splitlines() if line.strip()]
    devices = rows("--query-gpu=index,uuid")
    apps = rows("--query-compute-apps=pid,gpu_uuid")
    if (any(len(row) != 2 or not row[0].isdigit() or not row[1].startswith("GPU-") for row in devices + apps)
            or len({row[0] for row in devices}) != len(devices)):
        raise ValueError("unrecognized authoritative GPU query")
    return {int(index): uuid for index, uuid in devices}, apps


def reject_pending_credit_processes(pending, *, proc_root):
    run, launch = pending_credit_guard(pending)
    for process in proc_root.iterdir():
        if not process.name.isdigit() or not process.is_dir():
            continue
        try:
            state = (process / "stat").read_text().rsplit(")", 1)[1].split()[0]
            argv = (process / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
        except FileNotFoundError:
            continue
        if state in ("Z", "X"):
            continue
        def argument(flag):
            return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None
        trainer = argument("-m") == "action_head_rl.trainer" and argument("--output-dir") == str(run)
        monitor = (argument("--plan") == str(launch / "plan.json") and "--run" in argv)
        entry = str(launch / "entry.sh") in argv
        if trainer or monitor or entry:
            raise RuntimeError(f"pending credit process is already active: pid={process.name}")


def execution_gate(plan, *, proc_root=Path("/proc"), query_gpu=query_gpu_state):
    profile = plan.get("prerequisite_profile", DEFAULT_PREREQUISITE_PROFILE)
    if (len(plan["prerequisites"]) != 4
            or {row["label"] for row in plan["prerequisites"]} != prerequisite_labels(profile)):
        raise ValueError("audit needs the driver and launcher identities required by its profile")
    live = [row["label"] for row in plan["prerequisites"] if process_alive(row["identity"], proc_root=proc_root)]
    if live:
        raise RuntimeError(f"prerequisite processes are still live: {live}")
    if profile == GAP_PREREQUISITE_PROFILE:
        if any(row.get("credit_mode") != "episode_grpo" or row.get("critic_sha256") is not None
               for row in plan.get("fixtures", [])) or not plan.get("fixtures"):
            raise ValueError("before-credit audit requires episode_grpo fixture targets")
        reject_pending_credit_processes(plan.get("pending_credit"), proc_root=proc_root)
    devices, apps = query_gpu()
    if any(index not in devices for index in range(12)):
        raise ValueError("GPU query did not prove all twelve target devices exist")
    target = {devices[index] for index in range(12)}
    occupied = [row for row in apps if row[1] in target]
    if occupied:
        raise RuntimeError(f"GPU 0..11 are occupied: {occupied}")
    if profile == GAP_PREREQUISITE_PROFILE:
        pending_credit_guard(plan["pending_credit"])
    return {"ready": True, "physical_gpus": list(range(12)), "remaining_gpu_processes": apps,
            "prerequisite_profile": profile,
            "race_boundary": "Point-in-time process/GPU checks plus a cooperating audit lock; no guarantee against an unrelated future launcher. Keep GAE monitors unarmed until audit completion."}
