#!/usr/bin/env python3
"""Primitive ReAct baseline on the immutable balanced 50-case subset."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO), str(REPO / "vln_demo/src"), str(REPO / "verl")]

import agent_tools as A  # noqa: E402
import prompts as P  # noqa: E402
from verl_access.bench_client import (BenchClient, obs_image_part,  # noqa: E402
                                      tool_return_text)
from verl_access.vlm_policy import to_openai_messages  # noqa: E402

from common import atomic_json, mean, rate, safe, sha256_file  # noqa: E402
from action_head import primitive_trajectory_map as TM  # noqa: E402

BASE_PATH = REPO / "verl/scripts/eval/action_head_fm_p0_p2/curriculum_action_head_eval.py"
SPEC = importlib.util.spec_from_file_location("_base_curriculum_eval", BASE_PATH)
E = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(E)


def training_matched_plain_chat_query(vllm_url, messages, timeout):
    """Use the SFT brain sampling contract without the four-query tail."""
    payload = {
        "model": E.MODEL,
        "messages": to_openai_messages(messages),
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {
            "enable_thinking": False,
            "preserve_thinking": True,
        },
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 256,
    }
    request = urllib.request.Request(
        vllm_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    chunks, usage = [], {}
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"vLLM HTTP {exc.code}: {detail}") from None
    try:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            obj = json.loads(body)
            if obj.get("usage"):
                usage.update(obj["usage"])
            choice = (obj.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
            if piece:
                chunks.append(piece)
    finally:
        response.close()
    return "".join(chunks), "native-" + uuid.uuid4().hex, usage


def load_subset(path: Path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    cases = list(manifest.get("cases") or [])
    counts = {stage: sum(row.get("stage") == stage for row in cases)
              for stage in ("p0", "p1", "p2")}
    if counts != {"p0": 17, "p1": 17, "p2": 16} or len(cases) != 50:
        raise RuntimeError(f"fixed subset mismatch: counts={counts} total={len(cases)}")
    return manifest, cases


def result_path(output: Path, case: dict) -> Path:
    return output / "rollouts" / case["stage"] / safe(case["scene"]) / (
        safe(case["episode_id"]) + ".json")


def deterministic_env_seed(case: dict) -> int:
    key = f"{case['scene']}:{case['episode_id']}".encode()
    return 20260828 + int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def normalize_call(name: str, arguments: dict):
    return A.normalize_primitive_call(name, arguments)


class PrimitiveEvaluator:
    def __init__(self, args):
        self.args = args
        self.bench = BenchClient(args.bench_url, timeout=args.timeout)
        self.profile = A.get_tool_profile("primitive")

    def initial_messages(self, instruction, obs):
        image = obs_image_part(obs)
        if image is None:
            raise RuntimeError("reset did not return RGB")
        return [
            {"role": "system", "content": [{"type": "text", "text":
                TM.system_prompt(P.agent_system_prompt(tool_specs=self.profile.tool_specs),
                                 getattr(self.args, 'trajectory_map', False))}]},
            {"role": "user", "content": [image, {"type": "text", "text":
                P.agent_current_view_text() + "\n\n" + P.agent_task_text(instruction)}]},
        ]

    def rollout(self, case, seed):
        obs, initial_info = self.bench.reset(
            scene=case["scene"], episode_id=case["episode_id"],
            instruction=case["instruction"], seed=seed)
        previous_info = initial_info
        motion = {"path_dist": 0.0, "yaw": 0.0, "steps": 0}
        messages = self.initial_messages(case["instruction"], obs)
        trajectory_map = TM.TrajectoryMapState() if getattr(self.args, 'trajectory_map', False) else None
        calls_used, turns, agent_stopped = 0, [], False
        invalid_call_policy = A.invalid_tool_call_policy()
        why, infra_error = "", ""
        for turn_index in range(self.args.max_turns):
            if calls_used >= self.args.max_tool_calls:
                why = "tool_budget_exhausted"
                break
            sent = A.strip_old_images(messages, self.args.keep_last_images)
            if trajectory_map is not None:
                sent = TM.inject_trajectory_map(sent, trajectory_map)
            started = time.time()
            try:
                text, request_id, usage = training_matched_plain_chat_query(
                    self.args.vllm_url, sent, self.args.timeout)
            except Exception as exc:
                infra_error, why = str(exc), "model_error"
                break
            thinking, visible = A.split_thinking(text)
            parsed = A.parse_tool_calls(visible)
            calls = [call for call in parsed if call["name"] in self.profile.tools_by_name]
            record = {
                "turn": turn_index + 1,
                "assistant": str(visible or "").strip(),
                "reasoning": str(thinking or "").strip(),
                "request_id": request_id,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "dt_s": round(time.time() - started, 3),
                "calls": [],
            }
            if trajectory_map is not None:
                record['trajectory_map_state'] = {
                    'forward_m': trajectory_map.forward_m, 'left_m': trajectory_map.left_m,
                    'heading_deg': trajectory_map.heading_deg,
                    'points': list(trajectory_map.points),
                    'blocked_points': list(trajectory_map.blocked_points),
                    'collision_points': list(trajectory_map.collision_points)}
                record['request_image_count'] = sum(
                    part.get('type') in ('image', 'image_url') for message in sent
                    for part in message.get('content', []) if isinstance(part, dict))
            messages.append({"role": "assistant", "content": [{"type": "text",
                "text": A.assistant_history_content(text, visible, "keep_thinking") or text}]})
            format_error = A.tool_call_batch_error(visible, parsed, self.profile.tools_by_name)
            if getattr(self.args, 'one_tool_call_per_turn', False) and len(parsed) > 1:
                format_error = {'ok': False, 'error': 'Return exactly one tool_call per slow decision.'}
            if not calls or (invalid_call_policy == "feedback" and format_error):
                if invalid_call_policy == "terminate":
                    why = "no_valid_tool_call"
                    turns.append(record)
                    break
                calls_used += 1
                raw = format_error
                record["tool_call_error"] = raw
                turns.append(record)
                if calls_used >= self.args.max_tool_calls:
                    why = "tool_budget_exhausted"
                    break
                messages.append(A.tool_call_feedback_message(
                    raw, self.args.max_tool_calls - calls_used))
                continue
            results = []
            allowance = min(P.MAX_CALLS_PER_TURN,
                            self.args.max_tool_calls - calls_used)
            for call in calls[:allowance]:
                calls_used += 1
                name, requested = call["name"], call["arguments"]
                stop_rejection = (A.rejected_stop_result(requested)
                                  if name == "stop" and invalid_call_policy == "feedback" else None)
                if stop_rejection is not None:
                    record["calls"].append({"name": name, "arguments": requested,
                                            "ok": False, "tool_result": stop_rejection})
                    results.append((A.compact_json(stop_rejection), False))
                    continue
                if name == "stop":
                    agent_stopped = True
                    stop_obs, reward, terminated, truncated, score_info = \
                        self.bench.step_tool("stop", requested)
                    if stop_obs.get("rgb") is not None:
                        obs = stop_obs
                    record["calls"].append({
                        "name": name, "arguments": requested, "reward": reward,
                        "terminated": terminated, "truncated": truncated,
                        "score_info": score_info,
                    })
                    previous_info = score_info
                    why = "agent_stop"
                    break
                normalized, note = normalize_call(name, requested)
                if normalized is None:
                    text_result, raw = A.error_observation(
                        name, note, motion, None)
                    record["calls"].append({
                        "name": name, "arguments": requested,
                        "normalized_arguments": None, "ok": False,
                        "tool_result": raw,
                    })
                    results.append((text_result, False))
                    continue
                next_obs, reward, terminated, truncated, info = \
                    self.bench.step_tool(name, normalized)
                if next_obs.get("rgb") is not None:
                    obs = next_obs
                model_text = tool_return_text(
                    name, info, previous_info, normalized, motion)
                if trajectory_map is not None:
                    map_facts = TM.update_from_tool(
                        trajectory_map, name, normalized, info.get('tool_result') or {})
                    if map_facts:
                        model_text += '\n' + A.compact_json(map_facts)
                record["calls"].append({
                    "name": name, "arguments": requested,
                    "normalized_arguments": normalized, "ok": True,
                    "reward": reward, "terminated": terminated,
                    "truncated": truncated,
                    "tool_result": info.get("tool_result"),
                    "model_tool_result": model_text,
                    "blocked": bool(info.get("blocked_this_step")),
                })
                previous_info = info
                results.append((model_text, True))
                if getattr(self.args, 'stop_on_environment_termination', False) and (terminated or truncated):
                    why = 'environment_terminated'
                    break
            turns.append(record)
            if agent_stopped:
                break
            if why == 'environment_terminated':
                break
            if calls_used >= self.args.max_tool_calls:
                why = "tool_budget_exhausted"
                break
            if not results:
                why = "no_primitive_result"
                break
            budget_left = self.args.max_tool_calls - calls_used
            for index, (model_text, with_image) in enumerate(results):
                body = A.append_tool_budget(model_text, budget_left) \
                    if index == len(results) - 1 else model_text
                content = [{"type": "text", "text": "[工具返回]\n" + body}]
                if with_image and index == len(results) - 1:
                    image = obs_image_part(obs)
                    if image is not None:
                        content.insert(0, image)
                messages.append({"role": "user", "content": content})
        else:
            why = "turn_budget_exhausted"

        if agent_stopped:
            score_info = turns[-1]["calls"][-1].get("score_info") or {}
        else:
            try:
                _, _, _, _, score_info = self.bench.step_tool(
                    "stop", {"reason": "evaluator terminal scoring only"})
            except Exception as exc:
                score_info = {}
                infra_error = infra_error or f"terminal scoring failed: {exc}"
        oracle_success = bool(score_info.get("success", False))
        success = bool(agent_stopped and oracle_success and not infra_error)
        spl = float(score_info.get("spl", 0.0) or 0.0)
        official_detail = score_info.get('official_object_success_detail') or {}
        if getattr(self.args, 'success_metric', 'legacy_stop') == 'official':
            if 'official_object_success' not in score_info or not official_detail.get('available'):
                infra_error = infra_error or 'official object success unavailable'
            success = bool(score_info.get('official_object_success') and not infra_error)
            spl = E.success_weighted_spl(success, float(score_info.get('path_length_m', 0) or 0),
                                         float(score_info.get('shortest_path_m', 0) or 0))
        return {
            "schema": "qwen36_native_tool_closedloop_rollout_v1",
            **case,
            "model": E.MODEL,
            "tool_profile": "primitive",
            "seed": seed,
            "evaluation_valid": not bool(infra_error),
            "invalid_tool_call_policy": invalid_call_policy,
            "success": success,
            "success_contract": (official_detail.get('schema') if getattr(self.args, 'success_metric', 'legacy_stop') == 'official' else 'legacy_stop'),
            "official_object_success_detail": official_detail,
            "agent_stopped": agent_stopped,
            "oracle_success_at_end": oracle_success,
            "spl": spl,
            "path_length_m": score_info.get('path_length_m'),
            "collisions": score_info.get('collisions'),
            "distance_to_goal": score_info.get("distance_to_goal"),
            "steps_used": score_info.get("steps_used"),
            "calls_used": calls_used,
            "why": why,
            "infra_error": infra_error,
            "initial_info": initial_info,
            "turns": turns,
            "thinking_contract": {
                "enable_thinking": False, "temperature": 0.7,
                "top_p": 0.95, "max_tokens": 256,
                "omitted_parameters": ["seed", "top_k", "min_p", "presence_penalty"],
                "matches_sft_bundle_generation": True,
            },
        }


def metric_block(rows):
    valid = [row for row in rows if row.get("evaluation_valid")]
    return {
        "case_count": len(rows),
        "valid_rollouts": len(valid),
        "round_trajectory_consistency": {
            "value": None, "numerator": None, "denominator": None,
            "not_applicable_reason": "primitive profile has no FM option contract",
        },
        "stitched_option_trajectory_consistency": {
            "value": None, "numerator": None, "denominator": None,
            "not_applicable_reason": "primitive profile has no stitched FM option",
        },
        "success_rate": rate(row.get("success", False) for row in valid),
        "spl": mean(float(row.get("spl", 0.0) or 0.0) for row in valid),
        "mean_calls_used": mean(float(row.get("calls_used", 0) or 0) for row in valid),
        "mean_steps_used": mean(float(row.get("steps_used", 0) or 0) for row in valid),
        "termination_reasons": dict(Counter(str(row.get("why")) for row in valid)),
    }


def aggregate(output, subset_path, allow_partial=False):
    subset, cases = load_subset(subset_path)
    rows = []
    for case in cases:
        path = result_path(output, case)
        if path.exists():
            try:
                rows.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                pass
    stages = {stage.upper(): metric_block(
        [row for row in rows if row.get("stage") == stage])
        for stage in ("p0", "p1", "p2")}
    summary = {
        "schema": "qwen36_native_tool_closedloop_summary_v1",
        "baseline": "native_primitive_tool_use",
        "subset_manifest": str(subset_path.resolve()),
        "subset_manifest_sha256": sha256_file(subset_path),
        "expected_cases": 50,
        "completed_cases": len(rows),
        "invalid_tool_call_policy": A.invalid_tool_call_policy(),
        "invalid_tool_call_policy_mismatches": sum(
            row.get("invalid_tool_call_policy", "terminate") != A.invalid_tool_call_policy()
            for row in rows),
        "evaluation_valid": bool(rows) and all(
            row.get("evaluation_valid") and
            row.get("invalid_tool_call_policy", "terminate") == A.invalid_tool_call_policy()
            for row in rows) and (
                allow_partial or len(rows) == 50),
        "allow_partial": bool(allow_partial),
        "brain_sampling": {
            "enable_thinking": False, "temperature": 0.7,
            "top_p": 0.95, "max_tokens": 256, "seed": "omitted",
        },
        "stages": stages,
        "overall_micro": metric_block(rows),
        "primary_metrics": ["success_rate", "spl"],
        "not_comparable_metrics": [
            "round_trajectory_consistency",
            "stitched_option_trajectory_consistency",
        ],
        "subset_protocol": subset,
    }
    atomic_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    runp = sub.add_parser("run")
    runp.add_argument("--subset-manifest", required=True)
    runp.add_argument("--bench-url", required=True)
    runp.add_argument("--vllm-url", required=True)
    runp.add_argument("--output", required=True)
    runp.add_argument("--max-turns", type=int, default=30)
    runp.add_argument("--max-tool-calls", type=int, default=30)
    runp.add_argument("--keep-last-images", type=int, default=8)
    runp.add_argument("--timeout", type=int, default=600)
    runp.add_argument("--retries", type=int, default=3)
    runp.add_argument("--limit-cases", type=int, default=0)
    runp.add_argument("--allow-partial", action="store_true")
    aggp = sub.add_parser("aggregate")
    aggp.add_argument("--subset-manifest", required=True)
    aggp.add_argument("--output", required=True)
    aggp.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    output, subset_path = Path(args.output), Path(args.subset_manifest)
    if args.command == "aggregate":
        print(json.dumps(aggregate(output, subset_path, args.allow_partial),
                         ensure_ascii=False, indent=2))
        return
    _, cases = load_subset(subset_path)
    if args.limit_cases:
        cases = cases[:args.limit_cases]
    evaluator = PrimitiveEvaluator(SimpleNamespace(**vars(args)))
    for index, case in enumerate(cases, 1):
        path = result_path(output, case)
        if path.exists():
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                A.assert_tool_call_policy_matches(saved)
                if saved.get("evaluation_valid"):
                    print(f"[native] resume {index}/{len(cases)} {case['episode_id']}", flush=True)
                    continue
            except (OSError, ValueError):
                pass
        last = None
        for attempt in range(1, args.retries + 1):
            try:
                last = evaluator.rollout(case, deterministic_env_seed(case))
            except Exception as exc:
                last = {
                    "schema": "qwen36_native_tool_closedloop_rollout_v1", **case,
                    "evaluation_valid": False, "success": False,
                    "infra_error": repr(exc), "traceback": traceback.format_exc(),
                }
            last["attempt"] = attempt
            if last.get("evaluation_valid"):
                break
            time.sleep(min(5 * attempt, 15))
        atomic_json(path, last)
        print(f"[native] {index}/{len(cases)} {case['stage']} {case['episode_id']} "
              f"valid={last.get('evaluation_valid')} success={last.get('success')} "
              f"why={last.get('why')}", flush=True)
    summary = aggregate(output, subset_path, args.allow_partial)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary["evaluation_valid"]:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
