#!/usr/bin/env python3
"""Run the existing v6 action expert policy on frozen English R2R/RxR episodes.

The model loop, query capture, FM policy and concurrent scene scheduler are
reused from the SAGE evaluator. Only dataset identity, terminal scoring and
full-denominator aggregation are adapted here. No pi-agent is involved.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.util
import json
import math
import multiprocessing
from multiprocessing.managers import BaseManager
import os
from pathlib import Path
import re
import sys
import threading
import time
import traceback
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/train'))
from dlc_control_io_retry import retry_estale
SWEEP = REPO / "verl/scripts/eval/qwen36_4query_ckpt_sweep"
DEFAULT_SET = REPO / "vln_demo/batch/eval_sets/english-val-unseen-v1"
COUNTS = {"r2r": 1839, "rxr": 3669}
DATA_ROOT = Path(os.environ.get("RL_ARTIFACT_ROOT", "/data/closed_loop_3dgs_rl")).resolve()
SCORE_CONTRACT = "vlnce_native_stop_and_geodesic_lt_3m_v1"
MERGED_SCORE_CONTRACT = "vlnce_native_terminal_stop_and_geodesic_lt_3m_v1"
SCORING_MODES = ("active_stop", "merged_batch")
SOURCE_FILES = (
    "scripts/eval/v6_vlnce_eval.py",
    "verl/scripts/eval/qwen36_4query_ckpt_sweep/closedloop_eval.py",
    "verl/scripts/eval/action_head_fm_p0_p2/curriculum_action_head_eval.py",
    "verl/action_head/agent_runtime.py", "verl/action_head/runtime.py",
    "verl/action_head/policy.py", "verl/action_head/render_client.py",
    "verl/action_head/protocol_v6.py", "verl/action_head/protocol_v4_11.py",
    "verl/action_head/trajectory_map.py", "vln_demo/src/agent_tools.py",
    "vln_demo/src/vln_gym/env.py", "vln_demo/src/vln_gym/fm_action.py",
    "vln_demo/src/vln_gym/vlnce_action_head.py",
    "vln_demo/src/vln_gym/observation_profile.py",
    "vln_demo/src/remote_habitat/client.py", "vln_demo/src/remote_habitat/server.py",
    "vln_demo/src/vln_demo.yaml", "vln_demo/src/rxr.py",
    "verl/scripts/train/action_head_bootstrap/case_scheduler.py",
    "vln_demo/batch/model_gateway.py",
    "vln_demo/batch/recovery.py",
)


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# These utilities have no model, environment or service initialization.
COMMON = load_file("_v6_vlnce_common", SWEEP / "common.py")
SETS = load_file("_v6_vlnce_sets", REPO / "vln_demo/batch/eval_sets.py")
atomic_json, sha256_file = COMMON.atomic_json, COMMON.sha256_file
sys.path.insert(0, str(REPO / "verl/scripts/train/action_head_bootstrap"))
from case_scheduler import SceneCaseScheduler


class PoolManager(BaseManager):
    """Only episode dictionaries cross processes; model tensors remain local."""


PoolManager.register("SceneCaseScheduler", SceneCaseScheduler)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def load_cases(directory, dataset, ladder_stage='full', case_manifest=None):
    """Use the exact locked full pool and verify both manifest and source data."""
    ladder = SETS.load_ladder(directory, verify_sources=True)
    full = ladder["full"]
    counts = {key: sum(row["dataset"] == key for row in full) for key in COUNTS}
    if counts != COUNTS:
        raise ValueError(f"full English evaluation counts changed: {counts}")
    cases = [{**row, "stage": row["dataset"]} for row in ladder[ladder_stage]
             if dataset == "both" or row["dataset"] == dataset]
    if any(not re.fullmatch(r"[a-f0-9]{20}", row["logical_id"]) for row in cases):
        raise ValueError("unsafe or invalid frozen logical_id")
    if case_manifest:
        # Arbitrary suite subsets must be exact members of the locked source.
        frozen = json.loads(Path(case_manifest).read_text())
        selected = frozen['cases']
        lookup = {row['logical_id']: row for row in cases}
        ids = [row['logical_id'] for row in selected]
        if len(ids) != len(set(ids)) or any(lookup.get(row['logical_id']) != row for row in selected):
            raise ValueError('suite cases are duplicated or differ from the locked evaluation source')
        cases = selected
    return cases


def shard_cases(cases, index, count):
    if count < 1 or not 0 <= index < count:
        raise ValueError("shard-index must be in [0, shard-count)")
    return cases[index::count]


def result_path(output, case):
    return Path(output) / "rollouts" / (case["logical_id"] + ".json")


def validate_reset(case, obs, info):
    from vln_gym.observation_profile import requested_profile, assert_profile
    profile = requested_profile()
    if profile:
        assert_profile(info.get('observation_profile') or {}, profile)
    expected_scene = case["scene"]
    actual_scene = info.get("actual_scene")
    allowed_scenes = {expected_scene}
    if case["dataset"] == "rxr":
        allowed_scenes.add("rxr-" + expected_scene)
    if (str(info.get("actual_episode_id")) != str(case["episode_id"])
            or actual_scene not in allowed_scenes):
        raise ValueError(f"reset identity mismatch for {case['logical_id']}: "
                         f"{actual_scene}/{info.get('actual_episode_id')}")
    if info.get("backend", case["dataset"]) != case["dataset"]:
        raise ValueError("reset backend differs from the claimed dataset")
    if obs.get("rgb") is None and not obs.get("rgb_b64"):
        raise ValueError("reset returned no current RGB observation")
    if obs.get("instruction") not in (None, case["instruction"]):
        raise ValueError("reset instruction differs from frozen manifest")
    navmesh_root = os.environ.get('VLNCE_NAVMESH_ROOT')
    actual_navmesh = info.get('navmesh_identity')
    expected_navmesh = None
    if navmesh_root:
        from vln_gym.vlnce_action_head import navmesh_override_identity
        identity = navmesh_override_identity(navmesh_root)
        asset = identity['assets'].get(expected_scene)
        if asset is not None:
            expected_navmesh = dict(scene=expected_scene,
                                   manifest_sha256=identity['manifest_sha256'], **asset)
    if actual_navmesh != expected_navmesh:
        raise ValueError('reset navmesh version differs from the evaluation contract')


def score_native(row, terminal_info, scoring_mode="active_stop"):
    """Keep model STOP diagnostic separate from the merged terminal STOP score."""
    if scoring_mode not in SCORING_MODES:
        raise ValueError(f"unknown scoring mode: {scoring_mode}")
    row = dict(row)
    required = ("success", "spl", "distance_to_goal", "oracle_success",
                "oracle_navigation_error")
    missing = [key for key in required if key not in terminal_info]
    invalid = [key for key in required if key in terminal_info and
               (not isinstance(terminal_info[key], (int, float, bool)) or
                not math.isfinite(float(terminal_info[key])))]
    if missing or invalid:
        row["evaluation_valid"] = False
        row["infra_error"] = row.get("infra_error") or (
            f"native terminal metrics unavailable: missing={missing}, invalid={invalid}")
    valid = bool(row.get("evaluation_valid")) and not bool(row.get("infra_error"))
    native_success = bool(valid and terminal_info.get("success"))
    active_success = bool(native_success and row.get("agent_stopped"))
    native_spl = float(terminal_info.get("spl", 0.0) or 0.0) if native_success else 0.0
    success = native_success if scoring_mode == "merged_batch" else active_success
    row.update(evaluation_valid=valid, success=success,
               spl=native_spl if success else 0.0,
               active_stop_success=active_success,
               active_stop_spl=native_spl if active_success else 0.0,
               scoring_mode=scoring_mode,
               success_contract=(MERGED_SCORE_CONTRACT if scoring_mode == "merged_batch"
                                 else SCORE_CONTRACT),
               native_terminal_metrics=copy.deepcopy(terminal_info),
               native_success_after_scoring_stop=bool(terminal_info.get("success")),
               termination_reason=row.get("why", "infrastructure_exception"))
    for key in ("distance_to_goal", "oracle_success", "oracle_navigation_error",
                "ndtw", "sdtw", "path_length_m", "shortest_path_m", "steps_used"):
        if key in terminal_info:
            row[key] = terminal_info[key]
    if "sdtw" in row:
        row["sdtw"] = float(row["sdtw"]) if success else 0.0
    return row


def evaluator_type(closedloop):
    class VLNCEEvaluator(closedloop.TracedEvaluator):
        def __init__(self, args):
            super().__init__(args)
            self.native_terminal = {}
            original_reset = self.runtime.reset_environment
            original_step = self.bench.step_tool

            # The shared HTTP endpoint already supports backend rebuilding.
            # Inject it into the one real reset request, retaining all runtime
            # reset hooks and avoiding an extra reset/observation round trip.
            render = getattr(self.runtime, "render_client", None)
            if render is not None:
                original_post = render._post

                def post(endpoint, body):
                    if endpoint == "/reset":
                        body = {**body, "backend": self.current_case["dataset"]}
                    return original_post(endpoint, body)

                render._post = post

            def reset(**kwargs):
                obs, info = original_reset(**kwargs)
                validate_reset(self.current_case, obs, info)
                if getattr(self.args, "scoring_mode", "active_stop") == "merged_batch":
                    distance = info.get("distance_to_goal")
                    if not isinstance(distance, (int, float)) or not math.isfinite(float(distance)):
                        raise ValueError("native reset distance is missing or nonfinite")
                return obs, info

            def step(name, arguments=None):
                value = original_step(name, arguments)
                if name == "stop":
                    self.native_terminal = copy.deepcopy(value[-1])
                return value

            self.runtime.reset_environment = reset
            self.bench.step_tool = step

        def rollout(self, case, seed):
            self.current_case = case
            self.native_terminal = {}
            try:
                return score_native(super().rollout(case, seed), self.native_terminal,
                    getattr(self.args, "scoring_mode", "active_stop"))
            finally:
                release = getattr(self.runtime.tool_executor, "end_episode", None)
                if release is not None:
                    release()
    return VLNCEEvaluator


def configure_protocol():
    for key, value in {
        "COMPANION_BRAIN_MODE": "sft", "ACTION_HEAD_QUERY_TEXT_BUDGET": "256",
        "ACTION_HEAD_INVALID_TOOL_CALL_POLICY": "feedback",
        "VLN_OBSERVATION_CONTRACT": "agent_v1",
    }.items():
        if os.environ.get(key, value) != value:
            raise ValueError(f"{key} must be {value!r} for this frozen evaluation")
        os.environ[key] = value
    if os.environ.get("SAGE_BENCH_PROTOCOL_ROOT"):
        raise ValueError("SAGE benchmark auto-stop must be disabled for VLN-CE")


def build_contract(args, cases):
    configure_protocol()
    sys.path[:0] = [str(REPO / "verl"), str(REPO / "vln_demo/src")]
    from action_head.protocol import protocol_module
    protocol_identity = protocol_module(os.environ.get('RULE_TEACHER_CONTEXT_VERSION') or 'v6').protocol_identity
    identity = COMMON.checkpoint_identity(args.checkpoint_dir)
    config_path = Path(identity["action_head_dir"]) / "fast_slow_action_head_config.json"
    config = json.loads(config_path.read_text())
    if (identity.get("condition_mode") == "v5_native" or
            config.get("condition_source", "query") not in ("query", "query_hidden")):
        raise ValueError("v6 VLN-CE requires the selected query-conditioned checkpoint")
    identity["action_head_config_sha256"] = sha256_file(config_path)
    query_count = int(config["query_num_queries"])
    if query_count < 1:
        raise ValueError("checkpoint must contain query embeddings")
    backbone = Path(args.backbone).resolve(strict=True)
    config_files = ("config.json", "tokenizer_config.json", "preprocessor_config.json")
    backbone_files = {name: sha256_file(backbone / name) for name in config_files
                      if (backbone / name).is_file()}
    if "config.json" not in backbone_files:
        raise ValueError("backbone config.json is missing")
    lock = Path(args.eval_set) / "lock.json"
    scoring_mode = getattr(args, "scoring_mode", "active_stop")
    navmesh_root = os.environ.get('VLNCE_NAVMESH_ROOT')
    navmesh_identity = None
    if navmesh_root:
        from vln_gym.vlnce_action_head import navmesh_override_identity
        navmesh_identity = navmesh_override_identity(navmesh_root)
    return {
        "schema": ("v6_vlnce_full_english_k1_v1" if getattr(args, 'ladder_stage', 'full') == 'full' and not getattr(args, 'case_manifest', None)
                   else "v6_vlnce_fixed_subset_english_k1_v1"), "dataset": args.dataset,
        "expected_episodes": len(cases), "checkpoint": identity,
        "backbone": str(backbone), "backbone_config_sha256": backbone_files,
        "manifest_lock_sha256": sha256_file(lock), "case_pool_sha256": digest(cases),
        "eval_set": str(Path(args.eval_set).resolve()), "repeats": getattr(args,'repeats',1),
        "max_tool_calls": args.max_tool_calls, "max_turns": args.max_turns,
        "scoring_mode": scoring_mode,
        "one_tool_call_per_turn": args.max_turns == 0,
        "keep_last_images": 8, "protocol": protocol_identity(),
        "navmesh_override": navmesh_identity,
        **({'observation_profile': json.loads(os.environ['VLN_OBSERVATION_PROFILE'])}
           if os.environ.get('VLN_OBSERVATION_PROFILE') else {}),
        **({'suite_cases_sha256': sha256_file(args.case_manifest)}
           if getattr(args, 'case_manifest', None) else {}),
        "query_count": query_count,
        "success_contract": (MERGED_SCORE_CONTRACT if scoring_mode == "merged_batch"
                             else SCORE_CONTRACT),
        "sampling": {"enable_thinking": False, "preserve_thinking": True,
                     "temperature": 0.7, "top_p": 0.95,
                     "text_max_tokens": 256, "max_tokens": 256 + query_count + 1},
        "invalid_tool_call_policy": "feedback", "observation_contract": "agent_v1",
        "source_sha256": {name: sha256_file(REPO / name) for name in SOURCE_FILES},
    }


def ensure_contract(path, contract):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError(f"evaluation contract mismatch: {path}; use a new output")
    else:
        atomic_json(path, contract)


def resumable(row, case, contract_hash, repeat_index=1):
    return bool(row.get("evaluation_valid") and not row.get("infra_error") and
        row.get("contract_sha256") == contract_hash and
        row.get("case_sha256") == digest(case) and
        row.get("logical_id") == case["logical_id"] and
        row.get("dataset") == case.get("dataset") and
        row.get("repeat_index") == repeat_index)


def read_result(path):
    try:
        return json.loads(retry_estale(Path(path).read_text,str(path)))
    except FileNotFoundError:
        return {}


def summarize(cases, rows, contract_hash, repeat_index=1):
    valid, invalid, missing = [], [], []
    for case in cases:
        row = rows.get(case["logical_id"])
        if row is None:
            missing.append(case["logical_id"])
        elif resumable(row, case, contract_hash, repeat_index):
            valid.append(row)
        else:
            invalid.append(case["logical_id"])
    total = len(cases)
    successes = sum(bool(row.get("success")) for row in valid)
    spl_sum = sum(float(row.get("spl", 0.0)) for row in valid)
    metric_names = ("distance_to_goal", "oracle_success", "oracle_navigation_error",
                    "ndtw", "sdtw", "path_length_m", "steps_used")
    metrics = {}
    for name in metric_names:
        values = [float(row[name]) for row in valid if
                  isinstance(row.get(name), (int, float, bool)) and
                  math.isfinite(float(row[name]))]
        if values:
            metrics[name] = {"mean": sum(values) / len(values), "n": len(values)}
    return {
        "schema": "v6_vlnce_summary_v1", "contract_sha256": contract_hash,
        "dataset": cases[0]["dataset"] if cases else None,
        "expected_episodes": total, "completed_valid": len(valid),
        "infrastructure_failed_or_incompatible": len(invalid), "missing": len(missing),
        "evaluation_valid": len(valid) == total, "score_final": len(valid) == total,
        "success_rate": {"value": successes / total if total else None,
                         "numerator": successes, "denominator": total},
        "spl": {"value": spl_sum / total if total else None,
                "sum": spl_sum, "denominator": total},
        "completed_valid_metrics": metrics,
        "agent_stop_count": sum(bool(row.get("agent_stopped")) for row in valid),
        "active_stop_success_rate": {
            "value": sum(bool(row.get("active_stop_success", row.get("success") and
                row.get("agent_stopped"))) for row in valid) / total if total else None,
            "denominator": total},
        "active_stop_spl": {
            "value": sum(float(row.get("active_stop_spl", row.get("spl", 0.0)
                if row.get("agent_stopped") else 0.0)) for row in valid) / total if total else None,
            "denominator": total},
        "termination_counts": {why: sum(row.get("termination_reason") == why for row in valid)
                               for why in sorted({str(row.get("termination_reason")) for row in valid})},
        "missing_logical_ids": missing, "invalid_logical_ids": invalid,
        "partial_score_note": "Incomplete scores use the full denominator; no final score until every episode is valid.",
    }


def repeat_output(output, repeat_index, repeats):
    return Path(output)/f'repeat_{repeat_index:02d}' if repeats > 1 else Path(output)


def summarize_repeats(cases, rounds, contract_hash):
    """Retain every native metric and pair the same logical cases across trials."""
    k, n = len(rounds), len(cases)
    parts = [summarize(cases,rows,contract_hash,i) for i,rows in enumerate(rounds,1)]
    result = dict(schema='vlnce_pooled_repeat_summary_v1', contract_sha256=contract_hash,
        dataset=cases[0]['dataset'] if cases else None, expected_cases=n,
        expected_episodes=n*k, repeats_per_case=k, expected_runs=n*k,
        completed_valid=sum(p['completed_valid'] for p in parts),
        evaluation_valid=all(p['evaluation_valid'] for p in parts),
        score_final=all(p['evaluation_valid'] for p in parts),
        repeat_summaries=parts,
        missing=sum(p['missing'] for p in parts),
        infrastructure_failed_or_incompatible=sum(p['infrastructure_failed_or_incompatible'] for p in parts))
    if not result['evaluation_valid']:
        return result
    from rule_teacher_repeat_companion import paired_metrics
    paired = paired_metrics(rounds,[c['logical_id'] for c in cases],k)
    metrics = {}
    for name in set().union(*(p['completed_valid_metrics'] for p in parts)):
        entries = [p['completed_valid_metrics'][name] for p in parts if name in p['completed_valid_metrics']]
        count = sum(e['n'] for e in entries)
        metrics[name] = dict(mean=sum(e['mean']*e['n'] for e in entries)/count,n=count)
    result.update(paired_metrics=paired, success_rate=paired['pass_at_1'],
        spl=dict(value=paired['spl'],sum=paired['spl']*n*k,denominator=n*k),
        completed_valid_metrics=metrics, agent_stop_count=sum(p['agent_stop_count'] for p in parts),
        termination_counts=dict(sum((Counter(p['termination_counts']) for p in parts),Counter())),
        active_stop_success_rate=dict(value=sum(p['active_stop_success_rate']['value'] for p in parts)/k,denominator=n*k),
        active_stop_spl=dict(value=sum(p['active_stop_spl']['value'] for p in parts)/k,denominator=n*k))
    return result


def aggregate(args, cases, contract):
    output = Path(args.output)
    contract_hash = digest(contract)
    repeats = contract.get('repeats',1)
    rounds = []
    for i in range(1,repeats+1):
        rows = {}
        for case in cases:
            path = result_path(repeat_output(output,i,repeats),case)
            if retry_estale(path.exists,str(path)):
                rows[case['logical_id']] = read_result(path)
        rounds.append(rows)
    summary = (summarize_repeats(cases,rounds,contract_hash) if repeats > 1
               else summarize(cases,rounds[0],contract_hash))
    atomic_json(output / "summary.json", summary)
    return summary


def run(args, cases, contract):
    urls = [url.strip().rstrip("/") for url in args.bench_urls.split(",") if url.strip()]
    if not urls or len(urls) != len(set(urls)):
        raise ValueError("bench-urls must contain distinct Gym URLs")
    selected = shard_cases(cases, args.shard_index, args.shard_count)
    output = Path(args.output)
    contract_hash = digest(contract)
    shard_dir = output / f"shard_{args.shard_index:02d}"
    ensure_contract(shard_dir / "contract.json", {
        "contract_sha256": contract_hash, "shard_index": args.shard_index,
        "shard_count": args.shard_count, "logical_ids": [x["logical_id"] for x in selected]})
    pending = [case for case in selected if not resumable(
        read_result(result_path(output, case)), case, contract_hash)]
    if not pending:
        return {"selected": len(selected), "resumed": len(selected), "executed": 0}
    sys.path[:0] = [str(SWEEP), str(REPO / "verl/scripts/train/action_head_bootstrap")]
    import closedloop_eval as CE
    from action_head.policy import DynamicBatchingPolicy, PolicyProxy
    from case_scheduler import SceneCaseScheduler
    CE.ACTIVE_QUERY_COUNT = contract["query_count"]
    CE.E.chat_query = CE.training_matched_chat_query
    evaluator_class = evaluator_type(CE)
    namespace = SimpleNamespace(
        checkpoint=contract["checkpoint"]["action_head_dir"], backbone=args.backbone,
        query_cache=args.query_cache, bench_url=urls[0], vllm_url=args.vllm_url,
        timeout=args.timeout, policy_seed=20260827, mode="instruct",
        max_turns=args.max_turns, max_tool_calls=args.max_tool_calls,
        one_tool_call_per_turn=args.max_turns == 0,
        scoring_mode=args.scoring_mode,
        keep_last_images=8, action_head_as_tool_version=os.environ.get('RULE_TEACHER_CONTEXT_VERSION') or 'v6')
    base = CE.E.ActionHeadEvaluator(namespace)
    policy = DynamicBatchingPolicy(base.runtime.tool_executor.runtime.policy,
        max_batch_size=args.max_batch_size, batch_wait_ms=args.batch_wait_ms)
    scheduler = SceneCaseScheduler(pending, replicas=1,
        max_scenes=len(urls), max_cases=len(urls))
    counters = {"selected": len(selected), "resumed": len(selected) - len(pending),
                "executed": 0, "valid": 0, "success": 0}
    started, lock = time.monotonic(), threading.Lock()

    def actor(url):
        try:
            local = SimpleNamespace(**vars(namespace))
            local.bench_url = url
            local.shared_action_head_policy = PolicyProxy(policy)
            evaluator = evaluator_class(local)
            while (case := scheduler.acquire()) is not None:
                try:
                    episode_started = time.monotonic()
                    try:
                        row = evaluator.rollout(case, int(case["seed"]))
                    except Exception as exc:
                        row = {**case, "evaluation_valid": False, "success": False,
                               "spl": 0.0, "infra_error": repr(exc),
                               "termination_reason": "infrastructure_exception",
                               "traceback": traceback.format_exc()}
                    row.update(contract_sha256=contract_hash, case_sha256=digest(case),
                               repeat_index=1, repeats_per_case=1,
                               checkpoint=contract["checkpoint"],
                               evaluation_elapsed_s=time.monotonic() - episode_started)
                    atomic_json(result_path(output, case), row)
                    with lock:
                        counters["executed"] += 1
                        counters["valid"] += int(bool(row.get("evaluation_valid")))
                        counters["success"] += int(bool(row.get("success")))
                        atomic_json(shard_dir / "progress.json", {
                            **counters, "elapsed_s": time.monotonic() - started,
                            "batching": policy.stats(), "scheduler": scheduler.stats()})
                    print(json.dumps({"logical_id": case["logical_id"],
                        "dataset": args.dataset, "valid": row["evaluation_valid"],
                        "success": row["success"], "why": row.get("termination_reason")}), flush=True)
                finally:
                    scheduler.release(case)
        except BaseException:
            scheduler.cancel()
            raise

    try:
        with ThreadPoolExecutor(max_workers=len(urls)) as pool:
            futures = [pool.submit(actor, url) for url in urls]
            for future in futures:
                future.result()
    finally:
        policy.close()
        atomic_json(shard_dir / "timing.json", {
            **counters, "elapsed_s": time.monotonic() - started,
            "batching": policy.stats(), "scheduler": scheduler.stats()})
    return counters


def csv_values(value):
    return [item.strip().rstrip("/") for item in value.split(",") if item.strip()]


def pool_layout(args):
    """Each actor owns one Gym and one stable brain binding for its lifetime."""
    gyms = csv_values(args.bench_urls)
    gpus = csv_values(args.fm_gpus)
    brains = csv_values(args.vllm_urls or args.vllm_url)
    if not gyms or len(gyms) > 64 or len(set(gyms)) != len(gyms):
        raise ValueError("run-pool requires 1..64 distinct Gym URLs")
    if (not gpus or len(set(gpus)) != len(gpus)
            or not all(re.fullmatch(r"[0-9]+", gpu) for gpu in gpus)
            or len(gyms) % len(gpus)):
        raise ValueError("distinct numeric FM GPUs must divide the Gym actor count")
    if not brains or len(set(brains)) != len(brains) or len(brains) > len(gyms):
        raise ValueError("run-pool requires distinct brain URLs, at most one per actor")
    per_worker = len(gyms) // len(gpus)
    return [{"worker_index": worker, "gpu": gpu,
             "actors": [{"actor_index": index, "bench_url": gyms[index],
                         "preferred_dataset": (getattr(args, 'dataset', 'both')
                             if getattr(args, 'dataset', 'both') != 'both' else
                             'r2r' if index < len(gyms) // 2 else 'rxr'),
                         "brain_index": index % len(brains),
                         "vllm_url": brains[index % len(brains)]}
                        for index in range(worker * per_worker, (worker + 1) * per_worker)]}
            for worker, gpu in enumerate(gpus)]


def assert_case_owner(case, case_lookup):
    expected = case_lookup.get(case.get("logical_id"))
    if expected is None or expected != case:
        raise ValueError(f"scheduler returned an unknown or changed case: {case.get('logical_id')}")
    if case.get("dataset") not in COUNTS or case.get("stage") != case["dataset"]:
        raise ValueError("scheduler case has the wrong dataset ownership")


def evaluate_attempts(evaluator, case, contract, output, *, worker_index, actor_index,
                      repeat_index=1, repeats=1):
    """Retry infrastructure once; keep both attempts, including invalid metrics."""
    sys.path.insert(0, str(REPO / "vln_demo/batch"))
    from recovery import allocate_attempt
    contract_hash = digest(contract)
    started = time.monotonic()
    # SAGE manifests have official_split and no VLN-CE dataset/role/language.
    # Adapt only the archive address; keep the frozen case and its hash intact.
    archive_case = case
    if contract.get("benchmark") == "sage-3d":
        archive_case = dict(case, dataset="sage-3d", split=case["official_split"],
                            role="instruction", language="en")
    for attempt in range(2):
        destination = allocate_attempt(output, archive_case)
        try:
            row = evaluator.rollout(case, int(case.get("seed", contract.get("seed", 20261008))))
            if any(row.get(key) != value for key, value in case.items()):
                raise ValueError("rollout identity differs from its claimed manifest case")
        except Exception as exc:
            row = {**case, "evaluation_valid": False, "success": False, "spl": 0.0,
                   "active_stop_success": False, "active_stop_spl": 0.0,
                   "infra_error": repr(exc), "termination_reason": "infrastructure_exception",
                   "traceback": traceback.format_exc()}
        row.update(contract_sha256=contract_hash, case_sha256=digest(case),
                   repeat_index=repeat_index, repeats_per_case=repeats, checkpoint=contract["checkpoint"],
                   evaluation_elapsed_s=time.monotonic() - started,
                   worker_index=worker_index, actor_index=actor_index,
                   attempt_number=attempt + 1, attempt_path=str(destination / "result.json"))
        atomic_json(destination / "result.json", row)
        if resumable(row, case, contract_hash,repeat_index):
            break
    atomic_json(result_path(output, case), row)
    return row


def actor_namespace(args, contract, actor):
    return SimpleNamespace(
        checkpoint=contract["checkpoint"]["action_head_dir"], backbone=args.backbone,
        query_cache=args.query_cache, bench_url=actor["bench_url"],
        vllm_url=actor["vllm_url"], timeout=args.timeout, policy_seed=20260827,
        mode="instruct", max_turns=args.max_turns, max_tool_calls=args.max_tool_calls,
        one_tool_call_per_turn=args.max_turns == 0, scoring_mode=args.scoring_mode,
        keep_last_images=8, action_head_as_tool_version=os.environ.get('RULE_TEACHER_CONTEXT_VERSION') or 'v6')


def pool_worker(args_values, layout, scheduler, cases, contracts):
    """Spawn target: select the device before any model/CUDA imports."""
    os.environ["CUDA_VISIBLE_DEVICES"] = layout["gpu"]
    args = SimpleNamespace(**args_values)
    sys.path[:0] = [str(SWEEP), str(REPO / "verl"), str(REPO / "vln_demo/src")]
    configure_protocol()
    import closedloop_eval as CE
    from action_head.policy import DynamicBatchingPolicy, PolicyProxy
    first_contract = next(iter(contracts.values()))
    CE.ACTIVE_QUERY_COUNT = first_contract["query_count"]
    CE.E.chat_query = CE.training_matched_chat_query
    evaluator_class = evaluator_type(CE)
    output = Path(args.output)
    worker_output = output / "workers" / f"worker_{layout['worker_index']:02d}"
    case_lookup = {case["logical_id"]: case for case in cases}
    namespace = actor_namespace(args, first_contract, layout["actors"][0])
    base = CE.E.ActionHeadEvaluator(namespace)
    policy = DynamicBatchingPolicy(base.runtime.tool_executor.runtime.policy,
        max_batch_size=args.max_batch_size, batch_wait_ms=args.batch_wait_ms)
    counters = {"executed": 0, "valid": 0, "success": 0, "infrastructure_failed": 0}
    started, lock = time.monotonic(), threading.Lock()

    def progress():
        atomic_json(worker_output / "progress.json", {**counters,
            "worker_index": layout["worker_index"], "gpu": layout["gpu"],
            "elapsed_s": time.monotonic() - started, "batching": policy.stats()})

    def actor(binding):
        try:
            local = actor_namespace(args, first_contract, binding)
            local.shared_action_head_policy = PolicyProxy(policy)
            evaluator = evaluator_class(local)
            while (task := scheduler.acquire(binding["preferred_dataset"])) is not None:
                try:
                    repeat_index = task.get('_repeat_index',1)
                    case = {k:v for k,v in task.items() if k != '_repeat_index'}
                    assert_case_owner(case, case_lookup)
                    dataset = case["dataset"]
                    row = evaluate_attempts(evaluator, case, contracts[dataset],
                        repeat_output(output/dataset,repeat_index,getattr(args,'repeats',1)),
                        worker_index=layout["worker_index"], actor_index=binding["actor_index"],
                        repeat_index=repeat_index,repeats=getattr(args,'repeats',1))
                    with lock:
                        counters["executed"] += 1
                        valid = resumable(row, case, digest(contracts[dataset]),repeat_index)
                        counters["valid"] += int(valid)
                        counters["success"] += int(valid and bool(row.get("success")))
                        counters["infrastructure_failed"] += int(not valid)
                        progress()
                    print(json.dumps({"logical_id": case["logical_id"],
                        "dataset": dataset, "valid": valid, "success": row["success"],
                        "worker_index": layout["worker_index"],
                        "actor_index": binding["actor_index"],
                        "attempts": row["attempt_number"],
                        "why": row.get("termination_reason")}), flush=True)
                finally:
                    scheduler.release(task)
        except BaseException:
            scheduler.cancel()
            raise

    try:
        progress()
        with ThreadPoolExecutor(max_workers=len(layout["actors"])) as pool:
            futures = [pool.submit(actor, binding) for binding in layout["actors"]]
            for future in futures:
                future.result()
    finally:
        policy.close()
        progress()
        atomic_json(worker_output / "timing.json", {**counters,
            "elapsed_s": time.monotonic() - started, "batching": policy.stats()})


def pool_progress(output, scheduler, layouts, expected, resumed, started, phase):
    stats = scheduler.stats()
    workers = [read_result(Path(output) / "workers" / f"worker_{x['worker_index']:02d}"
                           / "progress.json") for x in layouts]
    value = {"schema": "v6_vlnce_global_pool_progress_v1", "timestamp": time.time(),
             "phase": phase, "expected_episodes": expected, "resumed": resumed,
             "completed": resumed + stats["completed"],
             "active": sum(stats["active_cases"]), "queued": stats["pending"],
             "workers": workers, "scheduler": stats,
             "elapsed_s": time.monotonic() - started}
    atomic_json(Path(output) / "progress.json", value)
    print("VLN_POOL_PROGRESS " + json.dumps(value), flush=True)
    return value


def run_pool(args, cases, contracts):
    if args.dataset == 'both' and (args.shard_index != 0 or args.shard_count != 1):
        raise ValueError('mixed run-pool cannot be sharded; use one dataset per suite stage')
    if args.max_turns != 0:
        raise ValueError("run-pool requires max-turns=0; the sole decision bound is max-tool-calls")
    layouts = pool_layout(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".pool.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for dataset, contract in contracts.items():
            ensure_contract(output / dataset / "contract.json", contract)
        scheduling = {"schema": "v6_vlnce_global_pool_v1", "layouts": layouts,
                      "repeats": getattr(args,'repeats',1),
                      "case_pool_sha256": digest(cases), "infra_attempts": 2,
                      "admission": "per_brain_merged_batch_aimd_v1",
                      "backend_scheduling": "actor_preference_then_shared_tail_v1"}
        ensure_contract(output / "pool_contract.json", scheduling)
        lookup = {case["logical_id"]: case for case in cases}
        if len(lookup) != len(cases):
            raise ValueError("duplicate logical identity in global manifest")
        for case in cases:
            assert_case_owner(case, lookup)
        repeats = getattr(args,'repeats',1)
        pending = [{**case,'_repeat_index':i} for case in cases for i in range(1,repeats+1) if not resumable(
            read_result(result_path(repeat_output(output/case['dataset'],i,repeats),case)),case,
            digest(contracts[case['dataset']]),i)]
        resumed, started = len(cases)*repeats - len(pending), time.monotonic()
        if not pending:
            return {"selected": len(cases), "resumed": resumed, "executed": 0}
        sys.path.insert(0, str(REPO / "vln_demo/batch"))
        from model_gateway import model_gateway
        # spawn prevents children inheriting the gateways' threads or a CUDA
        # context. The Manager exposes the existing thread-safe scene queue.
        context = multiprocessing.get_context("spawn")
        # CPFS does not support AF_UNIX files and long workspace runtime paths
        # exceed sockaddr_un. This is CPU-only local episode metadata traffic.
        manager = PoolManager(address=('127.0.0.1', 0), ctx=context)
        processes = []
        manager.start()
        scheduler = manager.SceneCaseScheduler(pending, replicas=1,
            max_scenes=len(csv_values(args.bench_urls)), max_cases=len(csv_values(args.bench_urls)))
        try:
            with ExitStack() as gateways:
                brains = csv_values(args.vllm_urls or args.vllm_url)
                for index, upstream in enumerate(brains):
                    assigned = [actor for layout in layouts for actor in layout["actors"]
                                if actor["brain_index"] == index]
                    admission_output = output / "admission" / f"brain_{index}"
                    admission_output.mkdir(parents=True, exist_ok=True)
                    gateway = gateways.enter_context(model_gateway(
                        upstream, maximum=len(assigned), run=admission_output)).removesuffix("/v1")
                    for binding in assigned:
                        binding["vllm_url"] = gateway
                for layout in layouts:
                    process = context.Process(target=pool_worker,
                        args=(vars(args), layout, scheduler, cases, contracts),
                        name=f"vlnce-fm-{layout['worker_index']}")
                    process.start()
                    processes.append(process)
                while True:
                    failed = [p for p in processes if p.exitcode not in (None, 0)]
                    if failed:
                        raise RuntimeError("FM workers failed: " + ", ".join(
                            f"{p.name}={p.exitcode}" for p in failed))
                    active = any(p.is_alive() for p in processes)
                    pool_progress(output, scheduler, layouts, len(cases)*repeats, resumed,
                                  started, "running" if active else "finished")
                    if not active:
                        break
                    # Short joins keep process-failure detection responsive;
                    # publish a heartbeat at least every thirty seconds.
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline and any(p.is_alive() for p in processes):
                        if any(p.exitcode not in (None, 0) for p in processes):
                            break
                        time.sleep(min(1, max(0, deadline - time.monotonic())))
                return {"selected": len(cases), "resumed": resumed,
                        "executed": scheduler.stats()["completed"]}
        finally:
            scheduler.cancel()
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
            manager.shutdown()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "run-pool", "aggregate", "check-inputs"))
    parser.add_argument("--dataset", choices=(*COUNTS, "both"), required=True)
    parser.add_argument("--eval-set", default=str(DEFAULT_SET))
    parser.add_argument("--ladder-stage", choices=('100', '200', '500', '1000', 'full'), default='full')
    parser.add_argument('--case-manifest', help='Frozen arbitrary subset; exact locked-source membership required')
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tool-calls", type=int, default=200)
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--scoring-mode", choices=SCORING_MODES, default="active_stop")
    parser.add_argument("--bench-urls", default="")
    parser.add_argument("--vllm-url", default="")
    parser.add_argument("--vllm-urls", default="")
    parser.add_argument("--fm-gpus", default="12,13,14,15")
    parser.add_argument("--query-cache", default="")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--batch-wait-ms", type=float, default=12.0)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)
    if min(args.max_tool_calls, args.max_batch_size, args.timeout,args.repeats) <= 0 or args.max_turns < 0 or args.batch_wait_ms < 0:
        parser.error("tool budget, timeout and batch size must be positive; max-turns may be zero")
    if args.dataset == "both" and (args.shard_index != 0 or args.shard_count != 1):
        parser.error("dataset=both owns the full global manifest and cannot be sharded")
    if args.command == "run" and args.dataset == "both":
        parser.error("use run-pool for a mixed dataset")
    if args.command == 'run' and args.repeats != 1:
        parser.error('multiple attempts require run-pool')
    cases = load_cases(args.eval_set, args.dataset, args.ladder_stage, args.case_manifest)
    selected = shard_cases(cases, args.shard_index, args.shard_count)
    datasets = tuple(COUNTS) if args.dataset == "both" else (args.dataset,)
    contracts = {dataset: build_contract(SimpleNamespace(**{**vars(args), "dataset": dataset}),
        [case for case in cases if case["dataset"] == dataset]) for dataset in datasets}
    contract = contracts[datasets[0]]
    if args.command == "check-inputs":
        if args.dataset == "both":
            print(json.dumps({"contracts": contracts,
                "contract_sha256": {key: digest(value) for key, value in contracts.items()},
                "selected_episodes": len(selected)}, indent=2))
            return 0
        print(json.dumps({"contract": contract, "contract_sha256": digest(contract),
                          "selected_episodes": len(selected)}, indent=2))
        return 0
    output = Path(args.output).resolve()
    if not output.is_relative_to(DATA_ROOT):
        parser.error(f"formal evaluation output must be below {DATA_ROOT}")
    args.output = str(output)
    if args.command == "run-pool":
        if not args.bench_urls or not (args.vllm_urls or args.vllm_url) or not args.query_cache:
            parser.error("run-pool requires bench-urls, vllm-urls/vllm-url and query-cache")
        # Contract describes the global subset; this node evaluates its slice.
        result = run_pool(args, selected, contracts)
        if args.shard_count > 1:
            print(json.dumps(result, indent=2))
            return 0 if all(resumable(read_result(result_path(repeat_output(output/case['dataset'],i,args.repeats),case)),
                case,digest(contracts[case['dataset']]),i) for case in selected for i in range(1,args.repeats+1)) else 4
        summaries = {dataset: aggregate(SimpleNamespace(**{**vars(args),
            "dataset": dataset, "output": str(output / dataset)}),
            [case for case in cases if case["dataset"] == dataset], contracts[dataset])
            for dataset in datasets}
        print(json.dumps({**result, "summaries": summaries}, indent=2))
        return 0 if all(summary["evaluation_valid"] for summary in summaries.values()) else 4
    # Per-shard contract files avoid two concurrently starting workers sharing
    # the atomic-json temporary filename. Aggregation verifies all row hashes.
    if args.command == "run":
        if not all((args.bench_urls, args.vllm_url, args.query_cache)):
            parser.error("run requires --bench-urls, --vllm-url and --query-cache")
        result = run(args, cases, contract)
        print(json.dumps(result, indent=2))
        return 0 if result.get("executed", 0) == result.get("valid", 0) else 4
    if args.dataset == "both":
        summaries = {dataset: aggregate(SimpleNamespace(**{**vars(args),
            "dataset": dataset, "output": str(output / dataset)}),
            [case for case in cases if case["dataset"] == dataset], contracts[dataset])
            for dataset in datasets}
        print(json.dumps(summaries, indent=2))
        return 0 if all(summary["evaluation_valid"] for summary in summaries.values()) else 4
    summary = aggregate(args, cases, contract)
    print(json.dumps(summary, indent=2))
    return 0 if summary["evaluation_valid"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
