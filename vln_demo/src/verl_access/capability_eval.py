"""Run Qwen ReAct on a SAGE capability manifest and produce diagnostic scores."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_DEFAULT_SAGE_ROOT = "/data/zerovln/SAGE-3D"
os.environ.setdefault("SAGE_COLLISION", "usd_rooms")
os.environ.setdefault("SAGE_USD_COLLISION_MASK_ROOT",
                      os.path.join(_DEFAULT_SAGE_ROOT, "sage_usd_rooms_masks"))

import prompts as P
from sage.capability_benchmark import (aggregate_scores, plan_recovery_intervention,
                                       project_to_path, score_rollout)
from sage.goal_region import (SEMANTIC_TERMINAL_SCHEMA,
                              evaluate_semantic_terminal)
from sage.occmap import OccMap
from sage.planner import plan_path
from verl_access.bench_client import BenchClient, OBSERVATION_CONTRACTS
from verl_access.vlm_policy import VLMAgent


class InterventionBench:
    """Transparent BenchClient proxy that injects one reproducible macro error.

    The requested action is executed first.  Once its resulting pose crosses
    the configured golden progress, the forced sequence is applied and only the
    unexpected final observation is returned.  Thus the model retains its full
    instruction, image history, and tool-call history, exactly as required by a
    semantic recovery test.
    """

    def __init__(self, base: BenchClient, golden_path, recovery=None, occ=None):
        self.base = base
        self.golden_path = golden_path
        self.recovery = recovery or {}
        self.occ = occ
        self.event = {"applied": False, "attempts": []}
        self.agent_call_index = 0

    def reset(self, **kwargs):
        self.event = {"applied": False, "attempts": []}
        self.agent_call_index = 0
        return self.base.reset(**kwargs)

    def step_tool(self, name, arguments=None):
        obs, reward, term, trunc, info = self.base.step_tool(name, arguments)
        call_index = self.agent_call_index
        self.agent_call_index += 1
        if (not self.event["applied"] and self.recovery and not term and not trunc and
                name != "stop"):
            xy = info.get("world_xy") or info.get("gps")
            progress = project_to_path(xy, self.golden_path)["progress"] if xy else 0.0
            if progress >= float(self.recovery.get("trigger_progress", .4)):
                yaw = info.get("world_yaw_deg", info.get("yaw_deg"))
                planned = None
                if self.occ is not None and xy is not None and yaw is not None:
                    planned = plan_recovery_intervention(
                        self.occ, self.golden_path, xy, yaw, self.recovery,
                        return_path_fn=lambda start, goal: plan_path(
                            self.occ, start, goal,
                            float(self.recovery.get("robot_radius_m", .07)),
                            mode="astar"))
                self.event["attempts"].append({
                    "after_call_index": call_index + 1, "progress": progress,
                    "xy": xy, "planned": planned is not None})
                if planned is None:
                    return obs, reward, term, trunc, info
                executed, forced_blocked, forced_collided = [], False, False
                for forced in planned["sequence"]:
                    obs, reward, term, trunc, info = self.base.step_tool(
                        forced["name"], forced.get("arguments") or {})
                    executed.append(forced)
                    forced_blocked |= bool(info.get("blocked_this_step"))
                    forced_collided |= bool(info.get("collided_this_step"))
                    if term or trunc:
                        break
                post_xy = info.get("world_xy") or info.get("gps")
                post_lateral = (project_to_path(post_xy, self.golden_path)["distance_m"]
                                if post_xy is not None else None)
                self.event = {"applied": True, "after_call_index": call_index + 1,
                              "trigger_progress": progress, "forced_sequence": executed,
                              "post_xy": post_xy, "post_lateral_error_m": post_lateral,
                              "blocked": forced_blocked, "collided": forced_collided,
                              "plan": planned, "attempts": self.event["attempts"]}
                info = dict(info)
                info["benchmark_intervention"] = dict(self.event)
        return obs, reward, term, trunc, info

    def __getattr__(self, name):
        return getattr(self.base, name)


def run_case(case, args):
    base = BenchClient(args.bench_url, timeout=args.timeout)
    occ = None
    if case["dimension"] == "recovery":
        matches = list(Path(args.sage_root, "InteriorGS").glob("*" + str(case["scene"])))
        if len(matches) != 1:
            raise FileNotFoundError("expected one InteriorGS scene for " + str(case["scene"]))
        occ = OccMap(str(matches[0]), collision_mode="usd_rooms", furniture_mode="none")
    bench = InterventionBench(base, case["golden_path_xy"],
                              case.get("recovery") if case["dimension"] == "recovery" else None,
                              occ=occ)
    label, url, served, _ = P.MODELS[args.model]
    agent = VLMAgent(bench, vllm_url=url, model_name=served, mode=args.mode,
                     max_turns=args.max_turns, max_tool_calls=args.max_tool_calls,
                     keep_last_images=args.keep_last_images, verbose=not args.quiet,
                     observation_contract=args.observation_contract)
    rollout = agent.rollout(case["scene"], case["episode_id"],
                            instruction=case["instruction"], seed=args.seed)
    rollout["case_id"] = case["case_id"]
    rollout["dimension"] = case["dimension"]
    rollout["intervention"] = bench.event
    try:
        semantic = evaluate_semantic_terminal(case, rollout, args.sage_root)
    except Exception as exc:
        semantic = {"schema_version": SEMANTIC_TERMINAL_SCHEMA,
                    "available": False,
                    "reason": "semantic_region_error: %s" % exc}
    score = score_rollout(case, rollout, semantic_terminal=semantic)
    # Do not call BenchClient.close(): its HTTP /close endpoint closes the
    # shared simulator and would invalidate every later case in this batch.
    return rollout, score


def rescore_run(run_dir, output=None, sage_root=_DEFAULT_SAGE_ROOT):
    """Recompute scores from immutable case+rollout artifacts with current code.

    Historical summaries are deliberately left untouched.  This makes scorer
    evolution auditable and lets the same rollout be compared under one schema.
    """
    run_dir = Path(run_dir)
    artifacts = []
    for path in sorted(run_dir.glob("*.json")):
        if path.name.startswith("summary"):
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc.get("case"), dict) or not isinstance(doc.get("rollout"), dict):
            continue
        artifacts.append((path, doc))
    scores = []
    for _, doc in artifacts:
        try:
            semantic = evaluate_semantic_terminal(
                doc["case"], doc["rollout"], sage_root)
        except Exception as exc:
            semantic = {"schema_version": SEMANTIC_TERMINAL_SCHEMA,
                        "available": False,
                        "reason": "semantic_region_error: %s" % exc}
        scores.append(score_rollout(doc["case"], doc["rollout"],
                                    semantic_terminal=semantic))
    summary = aggregate_scores(scores)
    summary.update({"rescore_source": str(run_dir), "scores": scores})
    target = Path(output) if output else run_dir / "summary.rescored.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary, target


def main():
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--rescore-run", help="offline run directory; never contacts bench/model")
    ap.add_argument("--output", default=None)
    ap.add_argument("--bench-url", default=os.environ.get("VLN_BENCH_URL",
                                                          "http://127.0.0.1:8907"))
    ap.add_argument("--sage-root", default=os.environ.get(
        "SAGE_ROOT", _DEFAULT_SAGE_ROOT))
    ap.add_argument("--model", default="qwen35b", choices=sorted(P.MODELS))
    ap.add_argument("--mode", default="think", choices=sorted(P.MODES))
    ap.add_argument("--dimension", action="append", choices=(
        "perception", "grounding", "planning", "recovery", "execution"))
    ap.add_argument("--case-id", action="append")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--max-tool-calls", type=int, default=200)
    ap.add_argument("--keep-last-images", type=int, default=8)
    ap.add_argument("--observation-contract", choices=OBSERVATION_CONTRACTS,
                    default=os.environ.get("VLN_OBSERVATION_CONTRACT", "agent_v1"),
                    help="model-visible tool result contract; agent_v1 hides oracle fields")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.rescore_run:
        summary, target = rescore_run(args.rescore_run, args.output, args.sage_root)
        print(json.dumps({"output": str(target), **summary}, ensure_ascii=False, indent=2))
        return

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    cases = manifest.get("cases") or []
    if args.dimension:
        cases = [x for x in cases if x.get("dimension") in args.dimension]
    if args.case_id:
        wanted = set(args.case_id)
        cases = [x for x in cases if x.get("case_id") in wanted]
    if args.limit > 0:
        cases = cases[:args.limit]
    out = Path(args.output or (Path(args.manifest).parent / "runs" /
                              time.strftime("%Y%m%d_%H%M%S")))
    out.mkdir(parents=True, exist_ok=True)
    rollouts, scores = [], []
    for i, case in enumerate(cases, 1):
        print("[capability] %d/%d %s" % (i, len(cases), case["case_id"]), flush=True)
        try:
            rollout, score = run_case(case, args)
        except Exception as exc:
            rollout = {"case_id": case["case_id"], "dimension": case["dimension"],
                       "error": str(exc), "turns": [], "final": {}}
            score = {"schema_version": case.get("schema_version"),
                     "success_contract": (case.get("evaluation_contract") or {}).get(
                         "primary_success"),
                     "case_id": case["case_id"], "dimension": case["dimension"],
                     "evaluation_valid": False, "success": False,
                     "invalid_reason": "runtime_error", "error": str(exc)}
        rollouts.append(rollout)
        scores.append(score)
        (out / (case["case_id"] + ".json")).write_text(
            json.dumps({"case": case, "rollout": rollout, "score": score},
                       ensure_ascii=False, indent=2), encoding="utf-8")
    summary = aggregate_scores(scores)
    summary["manifest"] = args.manifest
    summary["model"] = args.model
    summary["observation_contract"] = args.observation_contract
    summary["scores"] = scores
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
